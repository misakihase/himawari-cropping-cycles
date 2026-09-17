#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate daily EVI2 composites from high-NDVI Himawari observations.

For each pixel, valid observations within a day are ranked by NDVI.
EVI2 values corresponding to the upper 10% of valid NDVI observations
are averaged to produce one daily EVI2 composite.

Input daily files are assumed to contain 36 observations over the full
study grid with shape (36, 3200, 3500).

Output files have shape (3200, 3500), are stored as int16 after
multiplication by 10,000, and use -10000 as the missing-value flag.
"""

from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import tifffile


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

INPUT_DIR = Path("/path/to/HIMAWARI")
AOT_DIR = Path("/path/to/AOT")
LANDCOVER_FILE = Path("/path/to/Landcover/SE.tif")
OUTPUT_DIR = Path("/path/to/90percentile_NDVI")

YEARS = range(2016, 2025)
N_JOBS = 1

N_OBSERVATIONS = 36
HEIGHT = 3200
WIDTH = 3500

MIN_CLOUD_MASK = 0.95
MAX_AOT = 0.6
MAX_RED_REFLECTANCE = 0.2

EXCLUDED_LANDCOVER_CLASS = 14

TOP_FRACTION = 0.10
SCALE_FACTOR = 10000.0
FILL_VALUE = -10000


def read_daily_band(filepath: Path) -> np.ndarray:
    """
    Read one day of Himawari observations.

    The binary file is assumed to contain 36 int16 observations with shape
    (36, 3200, 3500). Values are converted using the scale factor 1e-4.
    """
    data = np.memmap(
        filepath,
        dtype=np.int16,
        mode="r",
        shape=(N_OBSERVATIONS, HEIGHT, WIDTH),
    )

    return np.asarray(data).astype(np.float32) * 1e-4


def calculate_daily_composite(
    ndvi: np.ndarray,
    evi2: np.ndarray,
) -> np.ndarray:
    """
    Average EVI2 values corresponding to the upper 10% of valid NDVI values.

    The number of selected observations follows the original implementation:
    floor(0.10 * number of valid observations), with at least one observation
    selected when valid data are available.
    """
    ndvi_for_sorting = np.nan_to_num(ndvi, nan=-1.0)

    valid = ndvi_for_sorting > -1.0
    n_valid = valid.sum(axis=0)
    has_valid = n_valid > 0

    n_selected = np.where(
        has_valid,
        np.maximum(
            1,
            (n_valid * TOP_FRACTION).astype(np.int32),
        ),
        0,
    )

    sort_index = np.argsort(ndvi_for_sorting, axis=0)
    evi2_sorted = np.take_along_axis(
        evi2,
        sort_index,
        axis=0,
    )

    ranks = np.arange(
        N_OBSERVATIONS,
        dtype=np.int32,
    )[:, None, None]

    selected = ranks >= (
        N_OBSERVATIONS - n_selected
    )[None, :, :]

    selected &= has_valid[None, :, :]

    selected_evi2 = np.where(
        selected,
        evi2_sorted,
        np.nan,
    )

    valid_count = np.sum(
        np.isfinite(selected_evi2),
        axis=0,
    )
    value_sum = np.nansum(
        selected_evi2,
        axis=0,
    )

    composite = np.full(
        (HEIGHT, WIDTH),
        np.nan,
        dtype=np.float32,
    )

    np.divide(
        value_sum,
        valid_count,
        out=composite,
        where=valid_count > 0,
    )

    return composite


def save_composite(
    filepath: Path,
    composite: np.ndarray,
):
    """Save the daily EVI2 composite as scaled int16 binary data."""
    output = np.full(
        composite.shape,
        FILL_VALUE,
        dtype=np.int16,
    )

    valid = np.isfinite(composite) & (composite >= 0)

    output[valid] = np.rint(
        composite[valid] * SCALE_FACTOR
    ).astype(np.int16)

    output.tofile(filepath)


def process_day(
    year: int,
    doy: int,
    landcover: np.ndarray,
):
    """Generate the daily EVI2 composite for one day."""
    date_id = f"{year:04d}{doy:03d}"
    print(date_id)

    red = read_daily_band(
        INPUT_DIR / "RED" / f"{date_id}.SE.flt"
    )
    nir = read_daily_band(
        INPUT_DIR / "NIR" / f"{date_id}.SE.flt"
    )
    cloud_mask = read_daily_band(
        INPUT_DIR / "CM" / f"{date_id}.SE.flt"
    )
    aot = read_daily_band(
        AOT_DIR / f"{date_id}.SE.flt"
    )

    invalid = (
        (cloud_mask < MIN_CLOUD_MASK)
        | (aot > MAX_AOT)
        | (red > MAX_RED_REFLECTANCE)
        | (red <= 0)
        | (nir <= 0)
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        ndvi = (
            (nir - red)
            / (nir + red)
        )

        evi2 = (
            2.5
            * (nir - red)
            / (nir + 2.4 * red + 1.0)
        )

    ndvi = np.where(
        invalid
        | (ndvi <= 0)
        | (ndvi > 1),
        np.nan,
        ndvi,
    ).astype(np.float32)

    evi2 = np.where(
        invalid
        | (evi2 < 0)
        | (evi2 > 1),
        np.nan,
        evi2,
    ).astype(np.float32)

    composite = calculate_daily_composite(
        ndvi,
        evi2,
    )

    composite = np.where(
        landcover != EXCLUDED_LANDCOVER_CLASS,
        composite,
        np.nan,
    )

    save_composite(
        OUTPUT_DIR / f"90_EVI2_{date_id}.flt",
        composite,
    )


def main():
    """Process daily observations from 2016 through 2024."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    landcover = tifffile.imread(
        LANDCOVER_FILE
    )[:HEIGHT, :WIDTH]

    for year in YEARS:
        Parallel(
            n_jobs=N_JOBS,
            prefer="threads",
        )(
            delayed(process_day)(
                year,
                doy,
                landcover,
            )
            for doy in range(1, 366)
        )


if __name__ == "__main__":
    main()