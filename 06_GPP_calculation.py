#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate daily and annual cropland GPP from Himawari-derived EVI2.

Daily EVI2 is converted to FPAR using pixel-wise minimum-maximum scaling
calculated across 2016-2023. GPP is then estimated with a MOD17-type
light-use-efficiency model:

    GPP = PAR * FPAR * epsilon

where epsilon is reduced by minimum-temperature and vapor-pressure-deficit
scalars.

All input arrays are assumed to have been preprocessed to the common study
grid with shape (3200, 3500).

Outputs for each year:
    GPP_<year>.flt       daily GPP, scaled by 100
    FPAR_<year>.flt      daily FPAR, scaled by 100
    GPP_<year>.npy       annual GPP sum [g C m-2 yr-1]
    FPAR_<year>.npy      annual mean FPAR
"""

from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

EVI2_DIR = Path("/path/to/Cropcycle")
SRAD_DIR = Path("/path/to/SRAD/Daily")
TMIN_DIR = Path("/path/to/TMIN/Daily")
VPD_DIR = Path("/path/to/VPD/Daily")
LANDCOVER_FILE = Path("/path/to/Landcover/SE.tif")
OUTPUT_DIR = Path("/path/to/GPP")

YEARS = range(2016, 2024)

HEIGHT = 3200
WIDTH = 3500
CROPLAND_CLASS = 10

EVI2_SCALE = 1.0 / 10000.0
OUTPUT_SCALE = 100.0
FILL_VALUE = -32768

# MOD17 cropland BPLUT parameters
EPSILON_MAX = 0.001044  # kg C MJ-1
TMIN_MIN = -8.0         # deg C; epsilon = 0
TMIN_MAX = 12.02        # deg C; epsilon = epsilon_max
VPD_MIN = 650.0         # Pa; epsilon = epsilon_max
VPD_MAX = 4300.0        # Pa; epsilon = 0

PAR_FRACTION = 0.45


def temperature_scalar(
    temperature: np.ndarray,
) -> np.ndarray:
    """Calculate the MOD17 minimum-temperature scalar."""
    return np.clip(
        (temperature - TMIN_MIN)
        / (TMIN_MAX - TMIN_MIN),
        0.0,
        1.0,
    )


def vpd_scalar(
    vpd: np.ndarray,
) -> np.ndarray:
    """Calculate the MOD17 vapor-pressure-deficit scalar."""
    return np.clip(
        (VPD_MAX - vpd)
        / (VPD_MAX - VPD_MIN),
        0.0,
        1.0,
    )


def read_daily_array(
    filepath: Path,
    dtype,
    scale: float = 1.0,
) -> np.ndarray:
    """Read one daily array on the common 3200 x 3500 study grid."""
    return (
        np.fromfile(
            filepath,
            dtype=dtype,
        )
        .reshape(HEIGHT, WIDTH)
        .astype(np.float32)
        * scale
    )


def read_landcover() -> np.ndarray:
    """Read the land-cover map on the common study grid."""
    landcover = tifffile.imread(
        LANDCOVER_FILE
    )

    return landcover[:HEIGHT, :WIDTH]


def read_evi2_year(
    year: int,
) -> np.memmap:
    """Open one annual LOESS EVI2 file as a memory-mapped array."""
    filepath = EVI2_DIR / f"LOESS_EVI2_{year}.flt"

    return np.memmap(
        filepath,
        dtype=np.int16,
        mode="r",
        shape=(365, HEIGHT, WIDTH),
    )


def calculate_evi2_bounds(
    cropland: np.ndarray,
):
    """
    Calculate pixel-wise EVI2 minimum and maximum across 2016-2023.

    Only valid cropland EVI2 observations are included.
    """
    evi2_min = np.full(
        (HEIGHT, WIDTH),
        np.inf,
        dtype=np.float32,
    )

    evi2_max = np.full(
        (HEIGHT, WIDTH),
        -np.inf,
        dtype=np.float32,
    )

    for year in YEARS:
        evi2_year = read_evi2_year(
            year
        )

        for day in tqdm(
            range(365),
            desc=f"EVI2 bounds {year}",
            leave=False,
        ):
            evi2_int = evi2_year[day]

            valid = (
                cropland
                & (evi2_int > -9000)
            )

            if not np.any(valid):
                continue

            evi2 = (
                evi2_int.astype(np.float32)
                * EVI2_SCALE
            )

            evi2[~valid] = np.nan

            evi2_min = np.fmin(
                evi2_min,
                evi2,
            )

            evi2_max = np.fmax(
                evi2_max,
                evi2,
            )

        del evi2_year

    evi2_min[
        np.isinf(evi2_min)
    ] = np.nan

    evi2_max[
        np.isinf(-evi2_max)
    ] = np.nan

    evi2_range = (
        evi2_max - evi2_min
    ).astype(np.float32)

    evi2_range[
        (~np.isfinite(evi2_range))
        | (evi2_range <= 0)
    ] = np.nan

    return evi2_min, evi2_max, evi2_range


def evi2_to_fpar(
    evi2_int: np.ndarray,
    cropland: np.ndarray,
    evi2_min: np.ndarray,
    evi2_range: np.ndarray,
) -> np.ndarray:
    """Convert daily EVI2 to FPAR by pixel-wise min-max scaling."""
    evi2 = (
        evi2_int.astype(np.float32)
        * EVI2_SCALE
    )

    evi2[
        (evi2_int <= -9000)
        | (~cropland)
    ] = np.nan

    fpar = (
        (evi2 - evi2_min)
        / evi2_range
    )

    return np.clip(
        fpar,
        0.0,
        1.0,
    )


def read_daily_climate(
    year: int,
    doy: int,
    cropland: np.ndarray,
):
    """
    Read daily solar radiation, minimum temperature, and VPD.

    Expected units after scaling:
        solar radiation : MJ m-2 day-1
        temperature     : deg C
        VPD             : Pa
    """
    solar_radiation = read_daily_array(
        SRAD_DIR
        / f"{year}{doy:03d}.srad.SE.flt",
        dtype=np.int16,
        scale=1.0 / 10000.0,
    )

    temperature = read_daily_array(
        TMIN_DIR
        / f"{year}{doy:03d}.tmp.SE.flt",
        dtype=np.int16,
        scale=1.0 / 100.0,
    )

    vpd = read_daily_array(
        VPD_DIR
        / f"{year}{doy:03d}.vpd.SE.flt",
        dtype=np.int16,
        scale=1.0 / 100.0,
    )

    solar_radiation[
        (~cropland)
        | (solar_radiation < 0)
    ] = np.nan

    temperature[
        (~cropland)
        | (temperature < -100)
        | (temperature > 100)
    ] = np.nan

    vpd[
        (~cropland)
        | (vpd < 0)
    ] = np.nan

    par = (
        solar_radiation
        * PAR_FRACTION
    )

    # Input VPD is in kPa after the original scale conversion.
    vpd *= 1000.0

    return par, temperature, vpd


def calculate_year(
    year: int,
    cropland: np.ndarray,
    evi2_min: np.ndarray,
    evi2_range: np.ndarray,
):
    """Calculate daily GPP/FPAR and annual summaries for one year."""
    evi2_year = read_evi2_year(
        year
    )

    gpp_path = (
        OUTPUT_DIR
        / f"GPP_{year}.flt"
    )

    fpar_path = (
        OUTPUT_DIR
        / f"FPAR_{year}.flt"
    )

    daily_gpp = np.memmap(
        gpp_path,
        dtype=np.int16,
        mode="w+",
        shape=(365, HEIGHT, WIDTH),
    )

    daily_fpar = np.memmap(
        fpar_path,
        dtype=np.int16,
        mode="w+",
        shape=(365, HEIGHT, WIDTH),
    )

    annual_gpp = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.float32,
    )

    fpar_sum = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.float32,
    )

    valid_gpp_days = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.uint16,
    )

    valid_fpar_days = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.uint16,
    )

    for day in tqdm(
        range(365),
        desc=f"GPP {year}",
    ):
        doy = day + 1

        fpar = evi2_to_fpar(
            evi2_year[day],
            cropland,
            evi2_min,
            evi2_range,
        )

        par, temperature, vpd = (
            read_daily_climate(
                year,
                doy,
                cropland,
            )
        )

        f_temperature = (
            temperature_scalar(
                temperature
            )
        )

        f_vpd = vpd_scalar(
            vpd
        )

        epsilon = (
            EPSILON_MAX
            * f_temperature
            * f_vpd
        )

        # EPSILON_MAX is in kg C MJ-1; multiply by 1000 for g C MJ-1.
        gpp = (
            par
            * fpar
            * epsilon
            * 1000.0
        )

        valid_fpar = np.isfinite(
            fpar
        )

        valid_gpp = np.isfinite(
            gpp
        )

        gpp_output = np.full(
            (HEIGHT, WIDTH),
            FILL_VALUE,
            dtype=np.int16,
        )

        fpar_output = np.full(
            (HEIGHT, WIDTH),
            FILL_VALUE,
            dtype=np.int16,
        )

        gpp_output[
            valid_gpp
        ] = np.rint(
            gpp[valid_gpp]
            * OUTPUT_SCALE
        ).astype(np.int16)

        fpar_output[
            valid_fpar
        ] = np.rint(
            fpar[valid_fpar]
            * OUTPUT_SCALE
        ).astype(np.int16)

        daily_gpp[day] = gpp_output
        daily_fpar[day] = fpar_output

        annual_gpp[
            valid_gpp
        ] += gpp[
            valid_gpp
        ]

        valid_gpp_days[
            valid_gpp
        ] += 1

        fpar_sum[
            valid_fpar
        ] += fpar[
            valid_fpar
        ]

        valid_fpar_days[
            valid_fpar
        ] += 1

    daily_gpp.flush()
    daily_fpar.flush()

    mean_fpar = np.full(
        (HEIGHT, WIDTH),
        np.nan,
        dtype=np.float32,
    )

    valid_mean_fpar = (
        valid_fpar_days > 0
    )

    mean_fpar[
        valid_mean_fpar
    ] = (
        fpar_sum[
            valid_mean_fpar
        ]
        / valid_fpar_days[
            valid_mean_fpar
        ].astype(np.float32)
    )

    annual_gpp[
        (~cropland)
        | (valid_gpp_days == 0)
    ] = np.nan

    mean_fpar[
        ~cropland
    ] = np.nan

    np.save(
        OUTPUT_DIR
        / f"GPP_{year}.npy",
        annual_gpp,
    )

    np.save(
        OUTPUT_DIR
        / f"FPAR_{year}.npy",
        mean_fpar,
    )

    del (
        daily_gpp,
        daily_fpar,
        evi2_year,
    )


def main():
    """Calculate AHI-derived FPAR and GPP for 2016-2023."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    landcover = read_landcover()
    cropland = (
        landcover == CROPLAND_CLASS
    )

    evi2_min, _, evi2_range = (
        calculate_evi2_bounds(
            cropland
        )
    )

    for year in YEARS:
        calculate_year(
            year,
            cropland,
            evi2_min,
            evi2_range,
        )


if __name__ == "__main__":
    main()
