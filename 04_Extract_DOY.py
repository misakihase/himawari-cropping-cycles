#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract annual crop-cycle dates and cropping intensity from a two-year
phenology-event array.

The input contains two consecutive years (730 days) with the following codes:
    0 = no event
    1 = heading date
    2 = planting date
    3 = harvesting date

For each heading date, the nearest preceding planting date and nearest
subsequent harvesting date are identified. Crop cycles are assigned to a year
according to their planting date.

Outputs:
    Ddate_<year>_<year+1>.flt : heading dates, shape (2, 4, 3200, 3500)
    Tdate_<year>_<year+1>.flt : planting dates, shape (2, 4, 3200, 3500)
    Hdate_<year>_<year+1>.flt : harvesting dates, shape (2, 4, 3200, 3500)
    Num_<year>_<year+1>.flt   : number of crop cycles, shape (2, 3200, 3500)

Planting and harvesting dates are stored as zero-based annual day indices
(0-364). Heading dates retain their positions on the concatenated two-year
time axis (0-729), consistent with the original analysis. Missing dates are
stored as -9999.
"""

from pathlib import Path
import argparse

from joblib import Parallel, delayed
import numpy as np
import tifffile
from tqdm import tqdm


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

INPUT_DIR = Path("/path/to/Cropcycle")
OUTPUT_DIR = Path("/path/to/Cropcycle/Date")
LANDCOVER_FILE = Path("/path/to/Landcover/SE.tif")

HEIGHT = 3200
WIDTH = 3500
N_DAYS = 365
N_YEARS = 2
MAX_CYCLES_PER_YEAR = 4

HEADING_CODE = 1
PLANTING_CODE = 2
HARVEST_CODE = 3

EXCLUDED_LANDCOVER_CLASS = 14
FILL_VALUE = -9999


def extract_event_dates(
    event_series: np.ndarray,
    event_code: int,
) -> list[np.ndarray]:
    """
    Extract up to four dates per year for one event type.

    The returned dates remain on the two-year time axis:
    year 1 = 0-364, year 2 = 365-729.
    """
    dates_by_year = []

    for year_index in range(N_YEARS):
        start = year_index * N_DAYS
        end = start + N_DAYS

        dates = np.flatnonzero(
            event_series[start:end] == event_code
        ) + start

        dates_by_year.append(
            dates[:MAX_CYCLES_PER_YEAR]
        )

    return dates_by_year


def nearest_preceding(
    dates: np.ndarray,
    target: int,
):
    """Return the nearest date before the target, or None."""
    candidates = dates[dates < target]

    if candidates.size == 0:
        return None

    return int(candidates[-1])


def nearest_following(
    dates: np.ndarray,
    target: int,
):
    """Return the nearest date after the target, or None."""
    candidates = dates[dates > target]

    if candidates.size == 0:
        return None

    return int(candidates[0])


def process_pixel(
    event_series: np.ndarray,
):
    """
    Extract heading, planting, and harvesting dates for one pixel.

    Each heading is paired with the nearest preceding planting date and
    nearest following harvesting date. A crop cycle is assigned to the year
    in which its planting date occurs.
    """
    heading_by_year = extract_event_dates(
        event_series,
        HEADING_CODE,
    )
    planting_by_year = extract_event_dates(
        event_series,
        PLANTING_CODE,
    )
    harvest_by_year = extract_event_dates(
        event_series,
        HARVEST_CODE,
    )

    # Preserve the original analysis: use at most four events of each type
    # within each calendar year before pairing crop-cycle dates.
    all_headings = np.concatenate(heading_by_year)
    all_plantings = np.concatenate(planting_by_year)
    all_harvests = np.concatenate(harvest_by_year)

    heading_output = np.full(
        (N_YEARS, MAX_CYCLES_PER_YEAR),
        FILL_VALUE,
        dtype=np.int16,
    )
    planting_output = np.full_like(
        heading_output,
        FILL_VALUE,
    )
    harvest_output = np.full_like(
        heading_output,
        FILL_VALUE,
    )

    # Heading dates retain their positions on the two-year time axis,
    # matching the original Ddate output.
    for year_index, dates in enumerate(heading_by_year):
        heading_output[
            year_index,
            :dates.size,
        ] = dates.astype(np.int16)

    cycle_count = np.zeros(
        N_YEARS,
        dtype=np.int16,
    )

    for heading in all_headings:
        planting = nearest_preceding(
            all_plantings,
            heading,
        )

        if planting is None:
            continue

        harvesting = nearest_following(
            all_harvests,
            heading,
        )

        year_index = 0 if planting < N_DAYS else 1
        slot = int(cycle_count[year_index])

        if slot >= MAX_CYCLES_PER_YEAR:
            continue

        planting_output[
            year_index,
            slot,
        ] = planting - year_index * N_DAYS

        if harvesting is not None:
            harvest_output[
                year_index,
                slot,
            ] = harvesting - year_index * N_DAYS

        cycle_count[year_index] += 1

    return (
        heading_output,
        planting_output,
        harvest_output,
        cycle_count,
    )


def process_row(
    row_index: int,
    events: np.ndarray,
    landcover: np.ndarray,
):
    """Process one spatial row."""
    heading_row = np.full(
        (N_YEARS, MAX_CYCLES_PER_YEAR, WIDTH),
        FILL_VALUE,
        dtype=np.int16,
    )
    planting_row = np.full_like(
        heading_row,
        FILL_VALUE,
    )
    harvest_row = np.full_like(
        heading_row,
        FILL_VALUE,
    )
    count_row = np.full(
        (N_YEARS, WIDTH),
        FILL_VALUE,
        dtype=np.int16,
    )

    for col in range(WIDTH):
        if landcover[row_index, col] == EXCLUDED_LANDCOVER_CLASS:
            continue

        (
            heading,
            planting,
            harvest,
            count,
        ) = process_pixel(
            events[row_index, col],
        )

        heading_row[:, :, col] = heading
        planting_row[:, :, col] = planting
        harvest_row[:, :, col] = harvest
        count_row[:, col] = count

    return (
        row_index,
        heading_row,
        planting_row,
        harvest_row,
        count_row,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract annual crop-cycle dates from a two-year "
            "phenology-event array."
        )
    )
    parser.add_argument(
        "year",
        type=int,
        help="First year of the two-year period, e.g. 2016",
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    event_path = (
        INPUT_DIR
        / f"Phenology_{first_year}_{second_year}.flt"
    )

    events = np.memmap(
        event_path,
        dtype=np.int8,
        mode="r",
        shape=(HEIGHT, WIDTH, N_DAYS * N_YEARS),
    )

    landcover = tifffile.imread(
        LANDCOVER_FILE
    )[:HEIGHT, 200:3700]

    heading_dates = np.full(
        (
            N_YEARS,
            MAX_CYCLES_PER_YEAR,
            HEIGHT,
            WIDTH,
        ),
        FILL_VALUE,
        dtype=np.int16,
    )

    planting_dates = np.full_like(
        heading_dates,
        FILL_VALUE,
    )

    harvest_dates = np.full_like(
        heading_dates,
        FILL_VALUE,
    )

    crop_count = np.full(
        (N_YEARS, HEIGHT, WIDTH),
        FILL_VALUE,
        dtype=np.int16,
    )

    rows = Parallel(
        n_jobs=args.n_jobs,
        prefer="threads",
    )(
        delayed(process_row)(
            row,
            events,
            landcover,
        )
        for row in tqdm(
            range(HEIGHT),
            desc="Extract crop-cycle dates",
        )
    )

    for (
        row,
        heading_row,
        planting_row,
        harvest_row,
        count_row,
    ) in rows:
        heading_dates[:, :, row, :] = heading_row
        planting_dates[:, :, row, :] = planting_row
        harvest_dates[:, :, row, :] = harvest_row
        crop_count[:, row, :] = count_row

    heading_dates.tofile(
        OUTPUT_DIR
        / f"Ddate_{first_year}_{second_year}.flt"
    )

    planting_dates.tofile(
        OUTPUT_DIR
        / f"Tdate_{first_year}_{second_year}.flt"
    )

    harvest_dates.tofile(
        OUTPUT_DIR
        / f"Hdate_{first_year}_{second_year}.flt"
    )

    crop_count.tofile(
        OUTPUT_DIR
        / f"Num_{first_year}_{second_year}.flt"
    )

    print(
        f"Saved crop-cycle dates for "
        f"{first_year}-{second_year}"
    )


if __name__ == "__main__":
    main()
