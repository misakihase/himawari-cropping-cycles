#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate annual mean FPAR and climate variables for one study region.

For each year from 2016 to 2023:
  1. Convert daily LOESS EVI2 to FPAR using fixed ROI-wide EVI2 bounds
     calculated from all years (2nd and 98th percentiles).
  2. Calculate annual mean FPAR, PAR, Tmin, and VPD for each cropland pixel.
  3. Calculate the regional mean and spatial standard deviation.

All input arrays are assumed to be preprocessed to the common
(3200, 3500) study grid.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm


EVI2_DIR = Path("/path/to/Cropcycle")
SRAD_DIR = Path("/path/to/SRAD/Daily")
TMIN_DIR = Path("/path/to/TMIN/Daily")
VPD_DIR = Path("/path/to/VPD/Daily")
LANDCOVER_FILE = Path("/path/to/Landcover/SE.tif")
MASK_DIR = Path("/path/to/Cropcycle")
OUTPUT_DIR = Path("/path/to/ROI_timeseries")

YEARS = range(2016, 2024)
HEIGHT = 3200
WIDTH = 3500
N_DAYS = 365

CROPLAND_CLASS = 10
Q_LOW = 0.02
Q_HIGH = 0.98
PAR_FRACTION = 0.45

REGION_MASK_FILES = {
    "ChaoPhraya": "ChaoPhraya_mask.tif",
    "Irrawaddy": "Irrawaddy_mask.tif",
    "Mekong": "Mekong_mask.tif",
    "Java": "Java_mask.tif",
}


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


def fixed_evi2_bounds(flat_idx: np.ndarray) -> tuple[float, float]:
    """Calculate ROI-pooled 2nd and 98th percentile EVI2 bounds."""
    histogram = np.zeros(20001, dtype=np.int64)

    for year in YEARS:
        evi2 = read_evi2_year(year)

        for day in tqdm(range(N_DAYS), desc=f"EVI2 bounds {year}", leave=False):
            e_int = evi2[day].ravel()[flat_idx]
            update_evi2_histogram(histogram, e_int)

        del evi2

    return (
        histogram_quantile(histogram, Q_LOW),
        histogram_quantile(histogram, Q_HIGH),
    )


def evi2_to_fpar(evi2: np.ndarray, low: float, high: float) -> np.ndarray:
    fpar = (evi2 - low) / (high - low)
    return np.clip(fpar, 0.0, 1.0).astype(np.float32)


def regional_mean_sd(values: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(values)

    if not np.any(valid):
        return np.nan, np.nan

    mean = float(np.nanmean(values))
    sd = (
        float(np.nanstd(values, ddof=1))
        if np.sum(valid) > 1
        else np.nan
    )

    return mean, sd


def annual_pixel_mean(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    out = np.full(total.shape, np.nan, dtype=np.float32)
    valid = count > 0
    out[valid] = total[valid] / count[valid].astype(np.float32)
    return out


def process_region(region: str):
    landcover = tifffile.imread(LANDCOVER_FILE)[:HEIGHT, :WIDTH]
    region_mask = tifffile.imread(
        MASK_DIR / REGION_MASK_FILES[region]
    )[:HEIGHT, :WIDTH] != 0

    roi = region_mask & (landcover == CROPLAND_CLASS)
    flat_idx = np.flatnonzero(roi.ravel())

    low, high = fixed_evi2_bounds(flat_idx)

    records = []

    for year in YEARS:
        evi2_year = read_evi2_year(year)
        npix = flat_idx.size

        fpar_sum = np.zeros(npix, dtype=np.float32)
        fpar_count = np.zeros(npix, dtype=np.uint16)

        par_sum = np.zeros(npix, dtype=np.float32)
        par_count = np.zeros(npix, dtype=np.uint16)

        tmin_sum = np.zeros(npix, dtype=np.float32)
        tmin_count = np.zeros(npix, dtype=np.uint16)

        vpd_sum = np.zeros(npix, dtype=np.float32)
        vpd_count = np.zeros(npix, dtype=np.uint16)

        for day in tqdm(range(N_DAYS), desc=f"{region} {year}", leave=False):
            doy = day + 1

            e_int = evi2_year[day].ravel()[flat_idx]
            evi2 = e_int.astype(np.float32) / 10000.0
            evi2[e_int <= -9000] = np.nan
            fpar = evi2_to_fpar(evi2, low, high)

            valid = np.isfinite(fpar)
            fpar_sum[valid] += fpar[valid]
            fpar_count[valid] += 1

            srad = read_daily_array(
                SRAD_DIR / f"{year}{doy:03d}.srad.SE.flt",
                np.int16,
                1.0 / 10000.0,
            ).ravel()[flat_idx]
            srad[srad < 0] = np.nan
            par = srad * PAR_FRACTION

            tmin = read_daily_array(
                TMIN_DIR / f"{year}{doy:03d}.tmp.SE.flt",
                np.int16,
                1.0 / 100.0,
            ).ravel()[flat_idx]
            tmin[(tmin < -100) | (tmin > 100)] = np.nan

            vpd = read_daily_array(
                VPD_DIR / f"{year}{doy:03d}.vpd.SE.flt",
                np.int16,
                1.0 / 100.0,
            ).ravel()[flat_idx]
            vpd[vpd < 0] = np.nan
            vpd *= 1000.0

            for data, total, count in (
                (par, par_sum, par_count),
                (tmin, tmin_sum, tmin_count),
                (vpd, vpd_sum, vpd_count),
            ):
                valid = np.isfinite(data)
                total[valid] += data[valid]
                count[valid] += 1

        fpar_mean = annual_pixel_mean(fpar_sum, fpar_count)
        par_mean = annual_pixel_mean(par_sum, par_count)
        tmin_mean = annual_pixel_mean(tmin_sum, tmin_count)
        vpd_mean = annual_pixel_mean(vpd_sum, vpd_count)

        fpar_mu, fpar_sd = regional_mean_sd(fpar_mean)
        par_mu, par_sd = regional_mean_sd(par_mean)
        tmin_mu, tmin_sd = regional_mean_sd(tmin_mean)
        vpd_mu, vpd_sd = regional_mean_sd(vpd_mean)

        records.append({
            "Region": region,
            "Year": year,
            "FPAR_mean": fpar_mu,
            "FPAR_sd": fpar_sd,
            "PAR_mean": par_mu,
            "PAR_sd": par_sd,
            "Tmin_mean": tmin_mu,
            "Tmin_sd": tmin_sd,
            "VPD_mean_Pa": vpd_mu,
            "VPD_sd_Pa": vpd_sd,
            "EVI2_q02": low,
            "EVI2_q98": high,
        })

        del evi2_year

    pd.DataFrame(records).to_csv(
        OUTPUT_DIR / f"{region}_annual_drivers.csv",
        index=False,
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
