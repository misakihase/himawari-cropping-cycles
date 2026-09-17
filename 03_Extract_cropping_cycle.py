#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detect heading, planting, and harvesting dates from two consecutive years
of daily EVI2 time series.

Two years of observations are analyzed together to provide temporal context
for identifying phenological events associated with individual years.

Event codes in the output:
    0 = no event
    1 = heading date
    2 = planting date
    3 = harvesting date
"""

from pathlib import Path
import argparse

from joblib import Parallel, delayed
import numpy as np
import statsmodels.api as sm
import tifffile
from tqdm import tqdm


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

INPUT_DIR = Path("/path/to/90percentile_NDVI")
OUTPUT_DIR = Path("/path/to/Cropcycle")
LANDCOVER_FILE = Path("/path/to/Landcover/SE.tif")

N_DAYS = 365
SOURCE_HEIGHT = 3200
SOURCE_WIDTH = 3700

# Analysis grid 
HEIGHT = 3200
WIDTH = 3500

CROPLAND_CLASS = 10
LOWESS_SPAN = 0.095
LOWESS_ITERATIONS = 0

# Minimum separation between adjacent heading candidates.
# This value is preserved from the original analysis code.
MIN_HEADING_INTERVAL = 60

# Event-selection thresholds (days) according to the number of heading dates
# detected in the two-year time series.
#
# 1-2 heading dates: approximately single cropping per year
# 3-4 heading dates: approximately double cropping per year
# 5-6 heading dates: approximately triple cropping per year
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


def get_threshold_group(n_heading: int) -> str:
    """Return the threshold group for the number of headings in two years."""
    if n_heading <= 2:
        return "1-2"
    if n_heading <= 4:
        return "3-4"
    return "5-6"


def read_year_evi2(year: int) -> np.ndarray:
    """
    Read one year of daily EVI2.

    Input files are int16 arrays with shape (3200, 3700), scaled by 10000.
    Values <= 0 are treated as missing.
    """
    evi2 = np.full((N_DAYS, HEIGHT, WIDTH), np.nan, dtype=np.float32)

    for day_index in tqdm(range(N_DAYS), desc=f"Read EVI2 {year}"):
        doy = day_index + 1
        path = INPUT_DIR / f"90_EVI2_{year}{doy:03d}.flt"

        arr = np.fromfile(path, dtype=np.int16).reshape(
            SOURCE_HEIGHT, SOURCE_WIDTH
        )
        arr = arr.astype(np.float32) / 10000.0
        arr[arr <= 0] = np.nan

        evi2[day_index] = arr

    return evi2


def interpolate_missing(y: np.ndarray, x: np.ndarray):
    """
    Fill missing values by one-dimensional linear interpolation.

    At the beginning and end of the series, numpy.interp extends the nearest
    valid value. At least three valid observations are required.
    """
    valid = np.isfinite(y)

    if valid.sum() < 3:
        return None

    if valid.all():
        return y.astype(np.float32, copy=False)

    return np.interp(x, x[valid], y[valid]).astype(np.float32, copy=False)


def smooth_evi2(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Smooth the two-year EVI2 time series using LOWESS."""
    return sm.nonparametric.lowess(
        y,
        x,
        frac=LOWESS_SPAN,
        it=LOWESS_ITERATIONS,
        is_sorted=True,
        return_sorted=False,
    ).astype(np.float32, copy=False)


def find_curve_features(y: np.ndarray, x: np.ndarray):
    """
    Identify local maxima, local minima, and inflection points
    from the smoothed EVI2 time series.
    """
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

    second_diff = np.diff(y, n=2)
    inflection_idx = np.where(np.diff(np.sign(second_diff)) != 0)[0]
    inflection_val = y[1:-1][inflection_idx]

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
    """
    Select heading-date candidates and merge closely spaced peaks.

    The peak-height criterion and close-peak treatment are retained from
    the original analysis code.
    """
    if peak_idx.size == 0:
        return peak_idx, peak_val

    threshold = 0.4 if np.nanmean(y) < 0.4 else np.nanmean(y)
    keep = peak_val >= threshold

    heading_idx = peak_idx[keep].copy()
    heading_val = peak_val[keep].copy()

    # Merge adjacent heading candidates that are too close in time.
    while heading_idx.size > 1:
        close_pairs = np.where(
            np.diff(heading_idx) < MIN_HEADING_INTERVAL
        )[0]

        if close_pairs.size == 0:
            break

        i = close_pairs[0]
        heading_idx[i] = (heading_idx[i] + heading_idx[i + 1]) / 2.0
        heading_val[i] = (heading_val[i] + heading_val[i + 1]) / 2.0

        heading_idx = np.delete(heading_idx, i + 1)
        heading_val = np.delete(heading_val, i + 1)

    return heading_idx, heading_val


def select_planting_dates(
    heading_idx: np.ndarray,
    minimum_idx: np.ndarray,
    minimum_val: np.ndarray,
    inflection_idx: np.ndarray,
    inflection_val: np.ndarray,
):
    """
    Estimate one planting-date candidate before each heading date.

    The required lag depends on the total number of heading dates detected
    in the two-year time series:
        1-2 headings: 50 / 55 days
        3-4 headings: 45 / 50 days
        5-6 headings: 40 / 45 days

    The first value refers to local minima and the second to inflection points.
    When both are available, the candidate closest to the heading date is used.
    """
    n_heading = heading_idx.size
    if n_heading == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    group = get_threshold_group(n_heading)
    minimum_lag = PLANT_MINIMUM_LAG[group]
    inflection_lag = PLANT_INFLECTION_LAG[group]

    planting_idx = []
    planting_val = []

    for heading in heading_idx:
        minimum_delta = minimum_idx - heading
        inflection_delta = inflection_idx - heading

        minimum_candidates = np.where(minimum_delta <= -minimum_lag)[0]
        inflection_candidates = np.where(inflection_delta <= -inflection_lag)[0]

        best_minimum = None
        best_inflection = None

        if minimum_candidates.size > 0:
            i = minimum_candidates[np.argmax(minimum_delta[minimum_candidates])]
            best_minimum = (minimum_idx[i], minimum_val[i])

        if inflection_candidates.size > 0:
            i = inflection_candidates[np.argmax(inflection_delta[inflection_candidates])]
            best_inflection = (inflection_idx[i], inflection_val[i])

        if best_minimum is not None and best_inflection is not None:
            # Choose the candidate closer to the heading date.
            if best_minimum[0] >= best_inflection[0]:
                planting_idx.append(best_minimum[0])
                planting_val.append(best_minimum[1])
            else:
                planting_idx.append(best_inflection[0])
                planting_val.append(best_inflection[1])

        elif best_minimum is not None:
            planting_idx.append(best_minimum[0])
            planting_val.append(best_minimum[1])

        elif best_inflection is not None:
            planting_idx.append(best_inflection[0])
            planting_val.append(best_inflection[1])

    return (
        np.asarray(planting_idx, dtype=np.float32),
        np.asarray(planting_val, dtype=np.float32),
    )


def select_harvest_dates(
    heading_idx: np.ndarray,
    planting_idx: np.ndarray,
    inflection_idx: np.ndarray,
    inflection_val: np.ndarray,
):
    """
    Estimate harvesting dates from inflection points after each heading date.

    The minimum heading-to-harvest interval is:
        1-2 headings: 25 days
        3-4 headings: 20 days
        5-6 headings: 15 days

    For all but the final heading, the selected harvesting date must occur
    before the planting date associated with the following crop cycle.
    """
    n_heading = heading_idx.size
    if n_heading == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    group = get_threshold_group(n_heading)
    minimum_lag = HARVEST_MINIMUM_LAG[group]

    harvest_idx = []
    harvest_val = []

    for i, heading in enumerate(heading_idx):
        delta = inflection_idx - heading

        # Determine the upper bound using the planting date before
        # the following heading, when available.
        upper_bound = None
        if i < n_heading - 1 and planting_idx.size > 0:
            before_next_heading = planting_idx[planting_idx < heading_idx[i + 1]]
            if before_next_heading.size > 0:
                next_planting = np.max(before_next_heading)
                if next_planting > heading:
                    upper_bound = next_planting - heading

        valid = delta >= minimum_lag
        if upper_bound is not None:
            valid &= delta <= upper_bound

        candidates = np.where(valid)[0]

        if candidates.size > 0:
            j = candidates[np.argmin(delta[candidates])]
            harvest_idx.append(inflection_idx[j])
            harvest_val.append(inflection_val[j])

    return (
        np.asarray(harvest_idx, dtype=np.float32),
        np.asarray(harvest_val, dtype=np.float32),
    )


def process_pixel(
    x_index: int,
    y_index: int,
    evi2_year1: np.ndarray,
    evi2_year2: np.ndarray,
    landcover: np.ndarray,
) -> np.ndarray:
    """Detect phenological events for one cropland pixel."""
    event_series = np.zeros(N_DAYS * 2, dtype=np.int8)

    if landcover[y_index, x_index] != CROPLAND_CLASS:
        return event_series

    y = np.concatenate(
        (
            evi2_year1[:, y_index, x_index],
            evi2_year2[:, y_index, x_index],
        )
    )

    x = np.arange(N_DAYS * 2, dtype=np.float32)

    y_interpolated = interpolate_missing(y, x)
    if y_interpolated is None:
        return event_series

    y_smoothed = smooth_evi2(y_interpolated, x)

    (
        peak_idx,
        peak_val,
        minimum_idx,
        minimum_val,
        inflection_idx,
        inflection_val,
    ) = find_curve_features(y_smoothed, x)

    heading_idx, heading_val = select_heading_dates(
        y_smoothed,
        peak_idx,
        peak_val,
    )

    planting_idx, planting_val = select_planting_dates(
        heading_idx,
        minimum_idx,
        minimum_val,
        inflection_idx,
        inflection_val,
    )

    harvest_idx, harvest_val = select_harvest_dates(
        heading_idx,
        planting_idx,
        inflection_idx,
        inflection_val,
    )

    event_series[heading_idx.astype(int)] = 1
    event_series[planting_idx.astype(int)] = 2
    event_series[harvest_idx.astype(int)] = 3

    return event_series


def main():
    parser = argparse.ArgumentParser(
        description="Detect crop phenological dates from two years of EVI2."
    )
    parser.add_argument(
        "year",
        type=int,
        help="First year of the two-year analysis period, e.g. 2022",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=16,
        help="Number of parallel workers",
    )
    args = parser.parse_args()

    first_year = args.year
    second_year = first_year + 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Land-cover mask for the same spatial grid used in the analysis.
    landcover = tifffile.imread(LANDCOVER_FILE)
    
    evi2_year1 = read_year_evi2(first_year)
    evi2_year2 = read_year_evi2(second_year)

    output_path = OUTPUT_DIR / f"Phenology_{first_year}_{second_year}.flt"

    # Write row by row to avoid holding the complete 3-D output in memory.
    output = np.memmap(
        output_path,
        mode="w+",
        dtype=np.int8,
        shape=(HEIGHT, WIDTH, N_DAYS * 2),
    )
    output[:] = 0

    for row in tqdm(range(HEIGHT), desc="Phenology detection"):
        row_events = Parallel(n_jobs=args.n_jobs, prefer="threads")(
            delayed(process_pixel)(
                col,
                row,
                evi2_year1,
                evi2_year2,
                landcover,
            )
            for col in range(WIDTH)
        )

        output[row] = np.asarray(row_events, dtype=np.int8)

    output.flush()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
