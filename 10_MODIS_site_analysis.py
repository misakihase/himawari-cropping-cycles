#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze MODIS EVI2 time series at the two representative sites.

Daily MODIS Terra/Aqua EVI2 time series exported from Google Earth Engine
are used as input. The script:

  1. reconstructs daily EVI2 time series for 2016-2023,
  2. fills missing observations by linear interpolation,
  3. applies LOWESS smoothing to two-year windows,
  4. detects heading, planting, and harvesting dates using the same
     phenology-detection procedure used for the AHI analysis, and
  5. saves smoothed EVI2 time series and detected crop-cycle dates.

Input CSV columns:
    year, doy, evi2

The CSV files are assumed to contain daily site-level EVI2 values exported
from Google Earth Engine after MODIS preprocessing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

INPUT_DIR = Path("/path/to/MODIS_EVI2")
OUTPUT_DIR = Path("/path/to/MODIS_results")

SITE_FILES = {
    "Site_A": "MOD_MYD_EVI2_daily_combined_site_a_16.47N_107.52E_2016_2023.csv",
    "Site_B": "MOD_MYD_EVI2_daily_combined_site_b_6.84S_107.27E_2016_2023.csv",
}

START_YEAR = 2016
END_YEAR = 2023
N_DAYS = 365

LOWESS_SPAN = 0.095
LOWESS_ITERATIONS = 0

MIN_HEADING_INTERVAL = 60

PLANT_MINIMUM_LAG = {
    "1-2": 50,
    "3-4": 45,
    "5-6": 40,
}

PLANT_INFLECTION_LAG = {
    "1-2": 55,
    "3-4": 50,
    "5-6": 45,
}

HARVEST_MINIMUM_LAG = {
    "1-2": 25,
    "3-4": 20,
    "5-6": 15,
}


def read_site_timeseries(filepath: Path) -> np.ndarray:
    """
    Read daily MODIS EVI2 and return an array with shape (number of years, 365).

    If multiple observations are present for the same year and DOY, their mean
    is used. Missing days remain NaN before temporal interpolation.
    """
    data = pd.read_csv(
        filepath,
        usecols=["year", "doy", "evi2"],
    )

    years = np.arange(
        START_YEAR,
        END_YEAR + 1,
    )

    evi2 = np.full(
        (years.size, N_DAYS),
        np.nan,
        dtype=np.float32,
    )

    for year_index, year in enumerate(years):
        subset = data[
            data["year"] == year
        ]

        for doy, values in subset.groupby("doy")["evi2"]:
            if 1 <= int(doy) <= N_DAYS:
                evi2[
                    year_index,
                    int(doy) - 1,
                ] = np.nanmean(
                    values.to_numpy(dtype=float)
                )

    return evi2


def interpolate_missing(
    y: np.ndarray,
    x: np.ndarray,
):
    """
    Fill missing values by one-dimensional linear interpolation.

    At the beginning and end of the series, numpy.interp extends the nearest
    valid value. At least three valid observations are required.
    """
    valid = np.isfinite(y)

    if valid.sum() < 3:
        return None

    if valid.all():
        return y.astype(
            np.float32,
            copy=False,
        )

    return np.interp(
        x,
        x[valid],
        y[valid],
    ).astype(
        np.float32,
        copy=False,
    )


def smooth_evi2(
    y: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Smooth a two-year EVI2 time series using LOWESS."""
    return sm.nonparametric.lowess(
        y,
        x,
        frac=LOWESS_SPAN,
        it=LOWESS_ITERATIONS,
        is_sorted=True,
        return_sorted=False,
    ).astype(
        np.float32,
        copy=False,
    )


def find_curve_features(
    y: np.ndarray,
    x: np.ndarray,
):
    """Identify local maxima, local minima, and inflection points."""
    first_diff = np.diff(y)

    peak_mask = (
        (first_diff[:-1] * first_diff[1:] < 0)
        & (first_diff[:-1] > 0)
    )

    minimum_mask = (
        (first_diff[:-1] * first_diff[1:] < 0)
        & (first_diff[:-1] < 0)
    )

    peak_idx = x[1:-1][peak_mask]
    peak_val = y[1:-1][peak_mask]

    minimum_idx = x[1:-1][minimum_mask]
    minimum_val = y[1:-1][minimum_mask]

    second_diff = np.diff(
        y,
        n=2,
    )

    inflection_idx = np.where(
        np.diff(
            np.sign(second_diff)
        ) != 0
    )[0]

    inflection_val = y[
        1:-1
    ][inflection_idx]

    return (
        peak_idx.astype(np.float32),
        peak_val.astype(np.float32),
        minimum_idx.astype(np.float32),
        minimum_val.astype(np.float32),
        inflection_idx.astype(np.float32),
        inflection_val.astype(np.float32),
    )


def select_heading_dates(
    y: np.ndarray,
    peak_idx: np.ndarray,
    peak_val: np.ndarray,
):
    """Select heading candidates and merge closely spaced peaks."""
    if peak_idx.size == 0:
        return peak_idx, peak_val

    threshold = (
        0.4
        if np.nanmean(y) < 0.4
        else np.nanmean(y)
    )

    keep = peak_val >= threshold

    heading_idx = peak_idx[
        keep
    ].copy()

    heading_val = peak_val[
        keep
    ].copy()

    while heading_idx.size > 1:
        close_pairs = np.where(
            np.diff(heading_idx)
            < MIN_HEADING_INTERVAL
        )[0]

        if close_pairs.size == 0:
            break

        i = close_pairs[0]

        heading_idx[i] = (
            heading_idx[i]
            + heading_idx[i + 1]
        ) / 2.0

        heading_val[i] = (
            heading_val[i]
            + heading_val[i + 1]
        ) / 2.0

        heading_idx = np.delete(
            heading_idx,
            i + 1,
        )

        heading_val = np.delete(
            heading_val,
            i + 1,
        )

    return heading_idx, heading_val


def threshold_group(
    n_heading: int,
) -> str:
    """Return threshold group according to headings detected in two years."""
    if n_heading <= 2:
        return "1-2"

    if n_heading <= 4:
        return "3-4"

    return "5-6"


def select_planting_dates(
    heading_idx: np.ndarray,
    minimum_idx: np.ndarray,
    minimum_val: np.ndarray,
    inflection_idx: np.ndarray,
    inflection_val: np.ndarray,
):
    """Estimate planting candidates preceding each heading date."""
    if heading_idx.size == 0:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    group = threshold_group(
        heading_idx.size
    )

    minimum_lag = (
        PLANT_MINIMUM_LAG[group]
    )

    inflection_lag = (
        PLANT_INFLECTION_LAG[group]
    )

    planting_idx = []
    planting_val = []

    for heading in heading_idx:
        minimum_delta = (
            minimum_idx - heading
        )

        inflection_delta = (
            inflection_idx - heading
        )

        minimum_candidates = np.where(
            minimum_delta <= -minimum_lag
        )[0]

        inflection_candidates = np.where(
            inflection_delta <= -inflection_lag
        )[0]

        best_minimum = None
        best_inflection = None

        if minimum_candidates.size > 0:
            i = minimum_candidates[
                np.argmax(
                    minimum_delta[
                        minimum_candidates
                    ]
                )
            ]

            best_minimum = (
                minimum_idx[i],
                minimum_val[i],
            )

        if inflection_candidates.size > 0:
            i = inflection_candidates[
                np.argmax(
                    inflection_delta[
                        inflection_candidates
                    ]
                )
            ]

            best_inflection = (
                inflection_idx[i],
                inflection_val[i],
            )

        if (
            best_minimum is not None
            and best_inflection is not None
        ):
            if (
                best_minimum[0]
                >= best_inflection[0]
            ):
                planting_idx.append(
                    best_minimum[0]
                )
                planting_val.append(
                    best_minimum[1]
                )
            else:
                planting_idx.append(
                    best_inflection[0]
                )
                planting_val.append(
                    best_inflection[1]
                )

        elif best_minimum is not None:
            planting_idx.append(
                best_minimum[0]
            )
            planting_val.append(
                best_minimum[1]
            )

        elif best_inflection is not None:
            planting_idx.append(
                best_inflection[0]
            )
            planting_val.append(
                best_inflection[1]
            )

    return (
        np.asarray(
            planting_idx,
            dtype=np.float32,
        ),
        np.asarray(
            planting_val,
            dtype=np.float32,
        ),
    )


def select_harvest_dates(
    heading_idx: np.ndarray,
    planting_idx: np.ndarray,
    inflection_idx: np.ndarray,
    inflection_val: np.ndarray,
):
    """Estimate harvesting candidates following each heading date."""
    if heading_idx.size == 0:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    group = threshold_group(
        heading_idx.size
    )

    minimum_lag = (
        HARVEST_MINIMUM_LAG[group]
    )

    harvest_idx = []
    harvest_val = []

    for i, heading in enumerate(
        heading_idx
    ):
        delta = (
            inflection_idx - heading
        )

        upper_bound = None

        if (
            i < heading_idx.size - 1
            and planting_idx.size > 0
        ):
            before_next_heading = (
                planting_idx[
                    planting_idx
                    < heading_idx[i + 1]
                ]
            )

            if (
                before_next_heading.size
                > 0
            ):
                next_planting = np.max(
                    before_next_heading
                )

                if next_planting > heading:
                    upper_bound = (
                        next_planting
                        - heading
                    )

        valid = (
            delta >= minimum_lag
        )

        if upper_bound is not None:
            valid &= (
                delta <= upper_bound
            )

        candidates = np.where(
            valid
        )[0]

        if candidates.size > 0:
            j = candidates[
                np.argmin(
                    delta[candidates]
                )
            ]

            harvest_idx.append(
                inflection_idx[j]
            )

            harvest_val.append(
                inflection_val[j]
            )

    return (
        np.asarray(
            harvest_idx,
            dtype=np.float32,
        ),
        np.asarray(
            harvest_val,
            dtype=np.float32,
        ),
    )


def detect_phenology(
    two_year_evi2: np.ndarray,
):
    """Detect phenological events from one two-year EVI2 window."""
    x = np.arange(
        N_DAYS * 2,
        dtype=np.float32,
    )

    interpolated = interpolate_missing(
        two_year_evi2,
        x,
    )

    if interpolated is None:
        return None

    smoothed = smooth_evi2(
        interpolated,
        x,
    )

    (
        peak_idx,
        peak_val,
        minimum_idx,
        minimum_val,
        inflection_idx,
        inflection_val,
    ) = find_curve_features(
        smoothed,
        x,
    )

    heading_idx, _ = (
        select_heading_dates(
            smoothed,
            peak_idx,
            peak_val,
        )
    )

    planting_idx, _ = (
        select_planting_dates(
            heading_idx,
            minimum_idx,
            minimum_val,
            inflection_idx,
            inflection_val,
        )
    )

    harvest_idx, _ = (
        select_harvest_dates(
            heading_idx,
            planting_idx,
            inflection_idx,
            inflection_val,
        )
    )

    return {
        "interpolated": interpolated,
        "smoothed": smoothed,
        "heading": heading_idx,
        "planting": planting_idx,
        "harvest": harvest_idx,
    }


def nearest_preceding(
    dates: np.ndarray,
    target: float,
):
    """Return the nearest date before target."""
    candidates = dates[
        dates < target
    ]

    if candidates.size == 0:
        return np.nan

    return float(
        candidates[-1]
    )


def nearest_following(
    dates: np.ndarray,
    target: float,
):
    """Return the nearest date after target."""
    candidates = dates[
        dates > target
    ]

    if candidates.size == 0:
        return np.nan

    return float(
        candidates[0]
    )


def extract_target_year_cycles(
    result: dict,
    target_part: int,
):
    """
    Pair each heading with the nearest preceding planting and following harvest.

    target_part:
        0 = first year in the two-year window
        1 = second year in the two-year window
    """
    start = target_part * N_DAYS
    end = start + N_DAYS

    rows = []

    target_headings = result[
        "heading"
    ][
        (result["heading"] >= start)
        & (result["heading"] < end)
    ]

    for heading in target_headings:
        planting = nearest_preceding(
            result["planting"],
            heading,
        )

        harvest = nearest_following(
            result["harvest"],
            heading,
        )

        # Assign the crop cycle according to the planting year.
        if (
            not np.isfinite(planting)
            or planting < start
            or planting >= end
        ):
            continue

        rows.append(
            {
                "Planting_DOY": (
                    planting - start + 1
                ),
                "Heading_DOY": (
                    heading - start + 1
                ),
                "Harvesting_DOY": (
                    harvest - start + 1
                    if np.isfinite(harvest)
                    else np.nan
                ),
            }
        )

    return rows


def analyze_site(
    site: str,
    filepath: Path,
):
    """Analyze one MODIS representative site."""
    annual_evi2 = read_site_timeseries(
        filepath
    )

    years = np.arange(
        START_YEAR,
        END_YEAR + 1,
    )

    smoothed_output = []
    phenology_output = []

    for target_year in years:
        # Use target year + following year when available.
        # For the final year, use the preceding year + target year because
        # the supplied MODIS time series ends in 2023.
        if target_year < END_YEAR:
            first_year = target_year
            second_year = target_year + 1
            target_part = 0
        else:
            first_year = target_year - 1
            second_year = target_year
            target_part = 1

        first_index = (
            first_year - START_YEAR
        )

        second_index = (
            second_year - START_YEAR
        )

        two_year = np.concatenate(
            (
                annual_evi2[first_index],
                annual_evi2[second_index],
            )
        )

        result = detect_phenology(
            two_year
        )

        if result is None:
            continue

        # Save the smoothed series only for the target-year portion.
        start = target_part * N_DAYS
        end = start + N_DAYS

        for day in range(N_DAYS):
            smoothed_output.append(
                {
                    "Site": site,
                    "Year": int(target_year),
                    "DOY": day + 1,
                    "EVI2_raw": float(
                        annual_evi2[
                            target_year - START_YEAR,
                            day,
                        ]
                    ),
                    "EVI2_interpolated": float(
                        result["interpolated"][
                            start + day
                        ]
                    ),
                    "EVI2_smoothed": float(
                        result["smoothed"][
                            start + day
                        ]
                    ),
                }
            )

        cycles = extract_target_year_cycles(
            result,
            target_part,
        )

        for cycle_number, cycle in enumerate(
            cycles,
            start=1,
        ):
            phenology_output.append(
                {
                    "Site": site,
                    "Year": int(target_year),
                    "Cycle": cycle_number,
                    **cycle,
                }
            )

    return (
        pd.DataFrame(
            smoothed_output
        ),
        pd.DataFrame(
            phenology_output
        ),
    )


def main():
    """Run MODIS site-level phenology analysis."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_smoothed = []
    all_phenology = []

    for site, filename in SITE_FILES.items():
        smoothed, phenology = (
            analyze_site(
                site,
                INPUT_DIR / filename,
            )
        )

        all_smoothed.append(
            smoothed
        )

        all_phenology.append(
            phenology
        )

    pd.concat(
        all_smoothed,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR
        / "MODIS_smoothed_EVI2.csv",
        index=False,
    )

    pd.concat(
        all_phenology,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR
        / "MODIS_phenology_dates.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
