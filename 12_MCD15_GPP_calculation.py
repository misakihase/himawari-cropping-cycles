#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate daily and annual cropland GPP from MODIS 8-day FPAR.

This is the MODIS counterpart of 05_GPP_calculation.py. The light-use-efficiency
model, its parameters and the daily meteorological forcing are identical; only
the FPAR source differs. Each LOESS-smoothed 8-day FPAR value is held constant
over its 8-day period, and GPP is integrated daily.

    GPP = PAR x FPAR x epsilon
    epsilon = epsilon_max x f(Tmin) x f(VPD)

Input
-----
    MCD15_FPAR_<year>_8day_loess.npy    from 11_MCD15_FPAR_preprocess.py
    daily shortwave radiation, minimum temperature and VPD on the analysis grid

Output
------
    MCD15_GPP_<year>.flt              daily GPP, int16, g C m-2 day-1 x 100
    MCD15_GPP_annual_<year>.npy       annual GPP, float32, g C m-2 year-1
    MCD15_GPP_valid_days_<year>.npy   days contributing to the annual sum
    MCD15_GPP_daily_slot_<year>.npy    day -> 8-day slot index

Pixels whose valid-day count is below MIN_VALID_DAYS are set to NaN in the
annual output, so that partially sampled years are not reported as annual totals.

Usage
-----
    python 12_MCD15_GPP_calculation.py 2018
"""

import argparse
import os

import numpy as np
import tifffile
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Analysis grid (must match 11_MCD15_FPAR_preprocess.py)
# ---------------------------------------------------------------------------
NY, NX = 3200, 3500
X0, X1 = 200, 3700
CROPLAND_CLASS = 10

# ---------------------------------------------------------------------------
# Paths. Edit these, or override with the environment variables of the same name.
# ---------------------------------------------------------------------------
LANDCOVER_PATH = os.environ.get("LANDCOVER_PATH", "/path/to/landcover/SE.tif")
OUTDIR = os.environ.get("OUTDIR", "/path/to/output")
SRAD_DIR = os.environ.get("SRAD_DIR", "/path/to/climate/srad/Daily")
TMIN_DIR = os.environ.get("TMIN_DIR", "/path/to/climate/tmin/Daily")
VPD_DIR = os.environ.get("VPD_DIR", "/path/to/climate/vpd/Daily")

# ---------------------------------------------------------------------------
# MOD17 cropland parameters (Biome Properties Look-Up Table).
# Names follow Eqs. 2-4 of the manuscript.
# ---------------------------------------------------------------------------
EPSILON_MAX = 0.001044   # kg C MJ-1
TMIN_MIN = -8.00         # deg C; f(Tmin) = 0 at or below
TMIN_MAX = 12.02         # deg C; f(Tmin) = 1 at or above
VPD_MIN = 650.0          # Pa;    f(VPD)  = 1 at or below
VPD_MAX = 4300.0         # Pa;    f(VPD)  = 0 at or above
PAR_FRACTION = 0.45      # PAR as a fraction of incoming shortwave radiation

# ---------------------------------------------------------------------------
# Input scaling and output encoding
# ---------------------------------------------------------------------------
SRAD_SCALE = 1.0 / 10000.0   # stored integer -> MJ m-2 day-1
TMIN_SCALE = 1.0 / 100.0     # stored integer -> deg C
VPD_SCALE = 1.0 / 100.0      # stored integer -> kPa
KPA_TO_PA = 1000.0
KGC_TO_GC = 1000.0

GPP_SCALE = 100.0            # daily GPP stored as int16 x 100
NODATA_INT16 = -32768
MIN_VALID_DAYS = 330         # below this, the annual total is set to NaN


# ---------------------------------------------------------------------------
# Raster readers
# ---------------------------------------------------------------------------
# The original analysis read the daily climate rasters through an in-house
# module. These two functions isolate that dependency: replace the body with
# whatever reader matches your own file format. Each must return an array of at
# least (NY, X1) so that the [:NY, X0:X1] slice below yields the analysis grid.
def read_float_raster(path: str) -> np.ndarray:
    """Read a daily raster stored as 32-bit values."""
    import Setting as Set  # in-house reader; substitute your own
    return Set.opendata6(path)


def read_int_raster(path: str) -> np.ndarray:
    """Read a daily raster stored as scaled integers."""
    import Setting as Set  # in-house reader; substitute your own
    return Set.opendata5(path)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
def is_leap_year(year: int) -> bool:
    return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))


def tmin_scalar(tmin: np.ndarray) -> np.ndarray:
    """Eq. 3: rises linearly from 0 at TMIN_MIN to 1 at TMIN_MAX."""
    return np.clip((tmin - TMIN_MIN) / (TMIN_MAX - TMIN_MIN), 0.0, 1.0)


def vpd_scalar(vpd: np.ndarray) -> np.ndarray:
    """Eq. 4: falls linearly from 1 at VPD_MIN to 0 at VPD_MAX."""
    return np.clip((VPD_MAX - vpd) / (VPD_MAX - VPD_MIN), 0.0, 1.0)


def build_daily_slot_index(n_days: int, n_slots: int) -> np.ndarray:
    """Map each day of year onto its 8-day compositing period."""
    return np.minimum(np.arange(n_days, dtype=np.int16) // 8,
                      n_slots - 1).astype(np.int16)


def load_cropland_mask() -> np.ndarray:
    land_cover = tifffile.imread(LANDCOVER_PATH)[:NY, X0:X1]
    return land_cover == CROPLAND_CLASS


def load_daily_field(path: str, cropland: np.ndarray, scale: float,
                     lower_bound: float, integer: bool) -> np.ndarray:
    """Read one daily raster, crop it to the analysis grid and mask it."""
    reader = read_int_raster if integer else read_float_raster
    field = reader(path).astype(np.float32)[:NY, X0:X1] * scale
    field[~cropland | (field < lower_bound)] = np.nan
    return field


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------
def calculate_gpp(year: int) -> None:
    os.makedirs(OUTDIR, exist_ok=True)

    n_days = 366 if is_leap_year(year) else 365

    fpar_path = os.path.join(OUTDIR, f"MCD15_FPAR_{year}_8day_loess.npy")
    fpar_slots = np.load(fpar_path)
    n_slots = fpar_slots.shape[0]

    slot_of_day = build_daily_slot_index(n_days, n_slots)

    out_daily = os.path.join(OUTDIR, f"MCD15_GPP_{year}.flt")
    out_annual = os.path.join(OUTDIR, f"MCD15_GPP_annual_{year}.npy")
    out_valid = os.path.join(OUTDIR, f"MCD15_GPP_valid_days_{year}.npy")
    out_slot = os.path.join(OUTDIR, f"MCD15_GPP_daily_slot_{year}.npy")
    np.save(out_slot, slot_of_day)

    cropland = load_cropland_mask()

    daily_gpp = np.memmap(out_daily, dtype=np.int16, mode="w+",
                          shape=(n_days, NY, NX))
    annual_gpp = np.zeros((NY, NX), dtype=np.float32)
    valid_days = np.zeros((NY, NX), dtype=np.uint16)
    day_buffer = np.empty((NY, NX), dtype=np.int16)

    n_skipped_fpar = 0
    n_skipped_climate = 0

    for day in tqdm(range(n_days), desc=f"Daily GPP {year}"):
        doy = day + 1
        fpar = fpar_slots[slot_of_day[day]]

        if not np.isfinite(fpar).any():
            n_skipped_fpar += 1
            day_buffer.fill(NODATA_INT16)
            daily_gpp[day] = day_buffer
            continue

        srad_path = os.path.join(SRAD_DIR, f"{year}{doy:03d}.srad.SE.flt")
        tmin_path = os.path.join(TMIN_DIR, f"{year}{doy:03d}.tmp.SE.flt")
        vpd_path = os.path.join(VPD_DIR, f"{year}{doy:03d}.vpd.SE.flt")

        if not all(os.path.exists(p) for p in (srad_path, tmin_path, vpd_path)):
            n_skipped_climate += 1
            day_buffer.fill(NODATA_INT16)
            daily_gpp[day] = day_buffer
            continue

        par = load_daily_field(srad_path, cropland, SRAD_SCALE, 0.0, integer=False)
        par *= PAR_FRACTION

        tmin = load_daily_field(tmin_path, cropland, TMIN_SCALE, -100.0, integer=True)
        vpd = load_daily_field(vpd_path, cropland, VPD_SCALE, 0.0, integer=True)
        vpd *= KPA_TO_PA

        epsilon = EPSILON_MAX * tmin_scalar(tmin) * vpd_scalar(vpd)
        gpp = par * fpar * epsilon * KGC_TO_GC   # g C m-2 day-1

        valid = cropland & np.isfinite(gpp)
        day_buffer.fill(NODATA_INT16)
        day_buffer[valid] = np.rint(gpp[valid] * GPP_SCALE).astype(np.int16)
        daily_gpp[day] = day_buffer

        annual_gpp[valid] += gpp[valid]
        valid_days[valid] += 1

        if day % 10 == 0:
            daily_gpp.flush()

    daily_gpp.flush()

    incomplete = cropland & (valid_days < MIN_VALID_DAYS)
    annual_gpp[incomplete] = np.nan
    annual_gpp[~cropland] = np.nan

    np.save(out_annual, annual_gpp)
    np.save(out_valid, valid_days)

    n_reported = int(np.isfinite(annual_gpp).sum())
    print(f"Days skipped: {n_skipped_fpar} for missing FPAR, "
          f"{n_skipped_climate} for missing climate forcing.")
    print(f"Annual GPP reported for {n_reported:,} of "
          f"{int(cropland.sum()):,} cropland pixels "
          f"(at least {MIN_VALID_DAYS} valid days).")
    for path in (out_daily, out_annual, out_valid):
        print("Saved:", path)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate cropland GPP from MODIS 8-day FPAR."
    )
    parser.add_argument("year", type=int, help="Four-digit year, e.g. 2018")
    args = parser.parse_args()
    calculate_gpp(args.year)


if __name__ == "__main__":
    main()
