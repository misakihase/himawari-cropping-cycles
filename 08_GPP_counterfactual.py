#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Counterfactual GPP experiment for Shapley attribution.

Drivers:
    PAR, FPAR, Tmin, VPD

For each cropland pixel and day of year, 2016-2023 climatologies are first
calculated for the four drivers. For each target year, GPP is then calculated
for all 2^4 = 16 combinations in which each driver is either:

    - the value from the target year, or
    - the 2016-2023 day-of-year climatology.

The resulting annual GPP values v(S), S = 0..15, are saved per pixel for
subsequent Shapley decomposition.

Bitmask:
    PAR  = 1
    FPAR = 2
    Tmin = 4
    VPD  = 8

Thus:
    mask 0  = all drivers climatological
    mask 15 = all drivers from the target year
"""

from pathlib import Path
import argparse

import numpy as np
import tifffile
from tqdm import tqdm


EVI2_DIR = Path("/path/to/Cropcycle")
SRAD_DIR = Path("/path/to/SRAD/Daily")
TMIN_DIR = Path("/path/to/TMIN/Daily")
VPD_DIR = Path("/path/to/VPD/Daily")
LANDCOVER_FILE = Path("/path/to/Landcover/SE.tif")
MASK_DIR = Path("/path/to/Cropcycle")
OUTPUT_DIR = Path("/path/to/ROI_counterfactual")

YEARS = range(2016, 2024)
HEIGHT = 3200
WIDTH = 3500
N_DAYS = 365

CROPLAND_CLASS = 10
Q_LOW = 0.02
Q_HIGH = 0.98

PAR_FRACTION = 0.45

EPSILON_MAX = 0.001044
TMIN_MIN = -8.0
TMIN_MAX = 12.02
VPD_MIN = 650.0
VPD_MAX = 4300.0

REGION_MASK_FILES = {
    "ChaoPhraya": "ChaoPhraya_mask.tif",
    "Irrawaddy": "Irrawaddy_mask.tif",
    "Mekong": "Mekong_mask.tif",
    "Java": "Java_mask.tif",
}


def temperature_scalar(tmin: np.ndarray) -> np.ndarray:
    return np.clip(
        (tmin - TMIN_MIN) / (TMIN_MAX - TMIN_MIN),
        0.0,
        1.0,
    )


def vpd_scalar(vpd: np.ndarray) -> np.ndarray:
    return np.clip(
        (VPD_MAX - vpd) / (VPD_MAX - VPD_MIN),
        0.0,
        1.0,
    )


def read_daily_array(path: Path, dtype, scale: float) -> np.ndarray:
    return (
        np.fromfile(path, dtype=dtype)
        .reshape(HEIGHT, WIDTH)
        .astype(np.float32)
        * scale
    )


def read_evi2_year(year: int) -> np.memmap:
    return np.memmap(
        EVI2_DIR / f"LOESS_EVI2_{year}.flt",
        dtype=np.int16,
        mode="r",
        shape=(N_DAYS, HEIGHT, WIDTH),
    )


def evi2_to_fpar(evi2: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(
        (evi2 - low) / (high - low),
        0.0,
        1.0,
    ).astype(np.float32)


def read_par(year: int, doy: int, flat_idx: np.ndarray) -> np.ndarray:
    srad = read_daily_array(
        SRAD_DIR / f"{year}{doy:03d}.srad.SE.flt",
        np.int16,
        1.0 / 10000.0,
    ).ravel()[flat_idx]

    srad[srad < 0] = np.nan
    return srad * PAR_FRACTION


def read_tmin(year: int, doy: int, flat_idx: np.ndarray) -> np.ndarray:
    tmin = read_daily_array(
        TMIN_DIR / f"{year}{doy:03d}.tmp.SE.flt",
        np.int16,
        1.0 / 100.0,
    ).ravel()[flat_idx]

    tmin[(tmin < -100) | (tmin > 100)] = np.nan
    return tmin


def read_vpd(year: int, doy: int, flat_idx: np.ndarray) -> np.ndarray:
    vpd = read_daily_array(
        VPD_DIR / f"{year}{doy:03d}.vpd.SE.flt",
        np.int16,
        1.0 / 100.0,
    ).ravel()[flat_idx]

    vpd[vpd < 0] = np.nan
    return vpd * 1000.0



def update_evi2_histogram(
    histogram: np.ndarray,
    values_int16: np.ndarray,
):
    """Add valid scaled EVI2 values to an integer histogram."""
    valid = values_int16 > -9000

    if not np.any(valid):
        return

    values = np.clip(
        values_int16[valid].astype(np.int32),
        -10000,
        10000,
    )

    histogram += np.bincount(
        values + 10000,
        minlength=20001,
    ).astype(np.int64)


def histogram_quantile(
    histogram: np.ndarray,
    quantile: float,
) -> float:
    """Return an EVI2 quantile from the scaled-integer histogram."""
    cumulative = np.cumsum(histogram)
    target = quantile * histogram.sum()
    index = int(np.searchsorted(cumulative, target, side="left"))

    return (index - 10000) / 10000.0


def build_climatology(flat_idx: np.ndarray):
    """Build pixel-wise day-of-year climatology and fixed EVI2 bounds."""
    npix = flat_idx.size

    sum_par = np.zeros((N_DAYS, npix), dtype=np.float32)
    sum_tmin = np.zeros((N_DAYS, npix), dtype=np.float32)
    sum_vpd = np.zeros((N_DAYS, npix), dtype=np.float32)
    sum_evi2 = np.zeros((N_DAYS, npix), dtype=np.float32)

    count_par = np.zeros((N_DAYS, npix), dtype=np.uint8)
    count_tmin = np.zeros((N_DAYS, npix), dtype=np.uint8)
    count_vpd = np.zeros((N_DAYS, npix), dtype=np.uint8)
    count_evi2 = np.zeros((N_DAYS, npix), dtype=np.uint8)

    evi2_histogram = np.zeros(20001, dtype=np.int64)

    for year in YEARS:
        evi2_year = read_evi2_year(year)

        for day in tqdm(range(N_DAYS), desc=f"Climatology {year}", leave=False):
            doy = day + 1

            e_int = evi2_year[day].ravel()[flat_idx]
            evi2 = e_int.astype(np.float32) / 10000.0
            evi2[e_int <= -9000] = np.nan

            valid = np.isfinite(evi2)
            sum_evi2[day, valid] += evi2[valid]
            count_evi2[day, valid] += 1
            update_evi2_histogram(
                evi2_histogram,
                e_int,
            )

            par = read_par(year, doy, flat_idx)
            tmin = read_tmin(year, doy, flat_idx)
            vpd = read_vpd(year, doy, flat_idx)

            for data, total, count in (
                (par, sum_par, count_par),
                (tmin, sum_tmin, count_tmin),
                (vpd, sum_vpd, count_vpd),
            ):
                valid = np.isfinite(data)
                total[day, valid] += data[valid]
                count[day, valid] += 1

        del evi2_year

    def finalize(total, count):
        out = np.full(total.shape, np.nan, dtype=np.float32)
        valid = count > 0
        out[valid] = total[valid] / count[valid].astype(np.float32)
        return out

    clim_par = finalize(sum_par, count_par)
    clim_tmin = finalize(sum_tmin, count_tmin)
    clim_vpd = finalize(sum_vpd, count_vpd)
    clim_evi2 = finalize(sum_evi2, count_evi2)

    low = histogram_quantile(
        evi2_histogram,
        Q_LOW,
    )
    high = histogram_quantile(
        evi2_histogram,
        Q_HIGH,
    )

    clim_fpar = evi2_to_fpar(clim_evi2, low, high)

    return clim_par, clim_fpar, clim_tmin, clim_vpd, low, high


def calculate_daily_gpp(
    par: np.ndarray,
    fpar: np.ndarray,
    tmin: np.ndarray,
    vpd: np.ndarray,
) -> np.ndarray:
    epsilon = (
        EPSILON_MAX
        * temperature_scalar(tmin)
        * vpd_scalar(vpd)
    )

    return par * fpar * epsilon * 1000.0


def process_region(region: str):
    landcover = tifffile.imread(LANDCOVER_FILE)[:HEIGHT, :WIDTH]
    region_mask = tifffile.imread(
        MASK_DIR / REGION_MASK_FILES[region]
    )[:HEIGHT, :WIDTH] != 0

    roi = region_mask & (landcover == CROPLAND_CLASS)
    flat_idx = np.flatnonzero(roi.ravel())
    npix = flat_idx.size

    (
        clim_par,
        clim_fpar,
        clim_tmin,
        clim_vpd,
        low,
        high,
    ) = build_climatology(flat_idx)

    region_dir = OUTPUT_DIR / region
    region_dir.mkdir(parents=True, exist_ok=True)

    for year in YEARS:
        evi2_year = read_evi2_year(year)

        annual_gpp = np.zeros((16, npix), dtype=np.float32)
        has_valid = np.zeros((16, npix), dtype=bool)

        for day in tqdm(range(N_DAYS), desc=f"{region} {year}", leave=False):
            doy = day + 1

            e_int = evi2_year[day].ravel()[flat_idx]
            evi2 = e_int.astype(np.float32) / 10000.0
            evi2[e_int <= -9000] = np.nan
            fpar = evi2_to_fpar(evi2, low, high)

            par = read_par(year, doy, flat_idx)
            tmin = read_tmin(year, doy, flat_idx)
            vpd = read_vpd(year, doy, flat_idx)

            Pc = clim_par[day]
            Fc = clim_fpar[day]
            Tc = clim_tmin[day]
            Vc = clim_vpd[day]

            for mask in range(16):
                P_use = par if (mask & 1) else Pc
                F_use = fpar if (mask & 2) else Fc
                T_use = tmin if (mask & 4) else Tc
                V_use = vpd if (mask & 8) else Vc

                gpp = calculate_daily_gpp(
                    P_use,
                    F_use,
                    T_use,
                    V_use,
                )

                valid = np.isfinite(gpp)
                annual_gpp[mask, valid] += gpp[valid]
                has_valid[mask, valid] = True

        annual_gpp[~has_valid] = np.nan

        np.save(
            region_dir / f"counterfactual_GPP_{year}.npy",
            annual_gpp,
        )

        del evi2_year

    np.savez(
        region_dir / "counterfactual_metadata.npz",
        EVI2_q02=np.float32(low),
        EVI2_q98=np.float32(high),
        years=np.array(list(YEARS), dtype=np.int16),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "region",
        choices=REGION_MASK_FILES.keys(),
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    process_region(args.region)


if __name__ == "__main__":
    main()
