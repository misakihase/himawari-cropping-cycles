#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate regional monthly SPI-3 from MSWEP precipitation.

Monthly precipitation is spatially averaged over four study regions from
1981 through 2024. SPI-3 is then calculated using a parametric gamma
distribution fitted separately for each calendar month over the 1991-2020
reference period.

One CSV file is written for each region with columns:
    Region, Year, Month, SPI-3
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm
import tifffile


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

PRECIP_DIR = Path("/path/to/MSWEP/Monthly")
MASK_DIR = Path("/path/to/Cropcycle")
OUTPUT_DIR = Path("/path/to/Cropcycle")

START_YEAR = 1981
END_YEAR = 2024

SPI_SCALE = 3
BASE_START = "1991-01-01"
BASE_END = "2020-12-01"
MIN_SAMPLES_PER_MONTH = 25

GRID_SHAPE = (3500, 7500)

REGION_MASK_FILES = {
    "ChaoPhraya": "ChaoPhraya_mask.tif",
    "Irrawaddy": "Irrawaddy_mask.tif",
    "Mekong": "Mekong_mask.tif",
    "Java": "Java_mask.tif",
}


def calculate_spi_gamma(
    precipitation: pd.Series,
    scale: int = SPI_SCALE,
    base_start: str = BASE_START,
    base_end: str = BASE_END,
    min_samples_per_month: int = MIN_SAMPLES_PER_MONTH,
) -> pd.Series:
    """
    Calculate parametric SPI using a gamma distribution.

    Parameters
    ----------
    precipitation
        Monthly precipitation with a pandas DatetimeIndex.
    scale
        Accumulation period in months. SPI-3 uses a 3-month moving sum.
    base_start, base_end
        Reference period used to estimate gamma-distribution parameters.
    min_samples_per_month
        Minimum number of valid reference-period samples required for each
        calendar month.

    Returns
    -------
    pandas.Series
        SPI values aligned with the input monthly time series.
    """
    if not isinstance(precipitation.index, pd.DatetimeIndex):
        raise TypeError(
            "precipitation must have a pandas DatetimeIndex."
        )

    precipitation = precipitation.sort_index().astype(float)

    # Accumulated precipitation for SPI-k.
    accumulated = precipitation.rolling(
        scale,
        min_periods=scale,
    ).sum()

    baseline = accumulated.loc[
        base_start:base_end
    ].dropna()

    if baseline.empty:
        raise ValueError(
            "No data are available in the reference period."
        )

    parameters = {}

    for month in range(1, 13):
        monthly_baseline = baseline[
            baseline.index.month == month
        ].dropna()

        if len(monthly_baseline) < min_samples_per_month:
            parameters[month] = None
            continue

        zero_probability = float(
            (monthly_baseline <= 0).mean()
        )

        positive_values = monthly_baseline[
            monthly_baseline > 0
        ]

        if len(positive_values) < min_samples_per_month:
            parameters[month] = None
            continue

        shape, _, scale_parameter = gamma.fit(
            positive_values.values,
            floc=0,
        )

        parameters[month] = (
            zero_probability,
            float(shape),
            float(scale_parameter),
        )

    spi = pd.Series(
        index=accumulated.index,
        dtype=float,
        name=f"SPI-{scale}",
    )

    epsilon = 1e-10

    for month in range(1, 13):
        params = parameters[month]

        if params is None:
            continue

        zero_probability, shape, scale_parameter = params

        month_mask = (
            (accumulated.index.month == month)
            & accumulated.notna()
        )

        values = accumulated.loc[
            month_mask
        ].values

        gamma_probability = gamma.cdf(
            values,
            shape,
            loc=0,
            scale=scale_parameter,
        )

        cumulative_probability = (
            zero_probability
            + (1.0 - zero_probability)
            * gamma_probability
        )

        cumulative_probability = np.clip(
            cumulative_probability,
            epsilon,
            1.0 - epsilon,
        )

        spi.loc[month_mask] = norm.ppf(
            cumulative_probability
        )

    return spi


def read_region_masks() -> dict[str, np.ndarray]:
    """Read binary masks for the four study regions."""
    masks = {}

    for region, filename in REGION_MASK_FILES.items():
        masks[region] = (
            tifffile.imread(MASK_DIR / filename) == 1
        )

    return masks


def calculate_regional_precipitation(
    masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Calculate monthly mean precipitation for each study region.

    Returns
    -------
    dict
        Region name -> array with shape (number of years, 12).
    """
    years = np.arange(
        START_YEAR,
        END_YEAR + 1,
    )

    regional_precipitation = {
        region: np.full(
            (years.size, 12),
            np.nan,
            dtype=np.float32,
        )
        for region in masks
    }

    for year_index, year in enumerate(years):
        for month_index in range(12):
            month = month_index + 1

            precipitation = np.fromfile(
                PRECIP_DIR
                / f"{year}{month:02d}.pre.SE.flt",
                dtype=np.float64,
            ).reshape(GRID_SHAPE)

            for region, mask in masks.items():
                regional_precipitation[
                    region
                ][year_index, month_index] = np.nanmean(
                    precipitation[mask]
                )

    return regional_precipitation


def save_spi_csv(
    region: str,
    monthly_precipitation: np.ndarray,
):
    """Calculate SPI-3 for one region and save the monthly time series."""
    values = monthly_precipitation.reshape(-1).astype(float)

    dates = pd.date_range(
        start=f"{START_YEAR}-01-01",
        periods=values.size,
        freq="MS",
    )

    precipitation_series = pd.Series(
        values,
        index=dates,
        name="precipitation_mm",
    )

    spi = calculate_spi_gamma(
        precipitation_series,
        scale=SPI_SCALE,
    )

    output = pd.DataFrame(
        {
            "Region": region,
            "Year": spi.index.year,
            "Month": spi.index.month,
            "SPI-3": spi.values,
        }
    )

    output.to_csv(
        OUTPUT_DIR / f"{region}_spi3.csv",
        index=False,
    )


def main():
    """Calculate and save regional SPI-3 for 1981-2024."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    masks = read_region_masks()

    regional_precipitation = (
        calculate_regional_precipitation(
            masks
        )
    )

    for region, precipitation in regional_precipitation.items():
        save_spi_csv(
            region,
            precipitation,
        )


if __name__ == "__main__":
    main()
