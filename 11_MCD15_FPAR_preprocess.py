#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproject MODIS 8-day FPAR to the AHI analysis grid and apply LOESS smoothing.

Input
-----
GeoTIFF files exported from Google Earth Engine by 00_MCD15_FPAR_GEE.js,
one file per 8-day compositing period, named:

    MCD15_FPAR_<YYYYMMDD>.tif

where <YYYYMMDD> is the start date of the 8-day period (DOY 1, 9, 17, ..., 361).

Output
------
    MCD15_FPAR_<year>_8day_reprojected_raw.npy   reprojected FPAR, NaN where missing
    MCD15_FPAR_<year>_8day_reprojected_used.npy  after single-slot gap filling
    MCD15_FPAR_<year>_8day_available.npy         which slots had a source file
    MCD15_FPAR_<year>_8day_fillflag.npy          -1 missing, 0 observed, 1 filled
    MCD15_FPAR_<year>_8day_loess.npy             LOESS-smoothed FPAR at 8-day slots
    MCD15_FPAR_<year>_daily_loess.flt            LOESS-smoothed FPAR, daily, int16 x10000
    MCD15_FPAR_<year>_loess_valid_count.npy      slots contributing per pixel
    MCD15_FPAR_<year>_loess_daily_valid_count.npy

The daily output is produced only so that the LUE model can be run at a daily
time step; within each 8-day period the FPAR value is constant, matching the
temporal resolution of the source product.

Usage
-----
    python 11_MCD15_FPAR_preprocess.py 2018
    python 11_MCD15_FPAR_preprocess.py 2018 --frac_days 35 --n_jobs 16
"""

import os
import re
import glob
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
import statsmodels.api as sm
import tifffile
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Analysis grid (must match the AHI grid used throughout the study)
# ---------------------------------------------------------------------------
NY, NX = 3200, 3500          # rows, columns of the analysis grid
X0, X1 = 200, 3700           # column slice applied to the land-cover raster

LON_MIN, LON_MAX = 92.0, 127.0
LAT_MIN, LAT_MAX = -9.0, 23.0
RES = 0.01                   # degrees

CROPLAND_CLASS = 10          # land-cover code for cropland

# ---------------------------------------------------------------------------
# Paths. Edit these, or override with the environment variables of the same name.
# ---------------------------------------------------------------------------
MCD15_DIR = os.environ.get("MCD15_DIR", "/path/to/GEE_MCD15_FPAR_8day")
LANDCOVER_PATH = os.environ.get("LANDCOVER_PATH", "/path/to/landcover/SE.tif")
OUTDIR = os.environ.get("OUTDIR", "/path/to/output")

# ---------------------------------------------------------------------------
# Processing options
# ---------------------------------------------------------------------------
SRC_NODATA = -9999.0
DST_NODATA = -9999.0
OUT_NODATA_INT16 = -9999
DAILY_SCALE = 10000.0        # daily FPAR stored as int16 x 10000

# Resampling to the 0.01 deg grid. Keep consistent with the Methods text.
RESAMPLING = Resampling.nearest

# Gap filling of whole 8-day composites that are absent from the archive
FILL_SINGLE_MISSING_SLOT = True   # interpolate a slot whose neighbours both exist
FILL_EDGE_NEAREST = False         # carry the nearest slot into a missing year edge

MIN_FINITE_SLOTS = 3              # pixels with fewer valid slots are left unsmoothed


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def is_leap_year(year: int) -> bool:
    return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))


def build_8day_slots(year: int):
    """Return (n_days, slot start DOYs, slot start dates as YYYYMMDD strings)."""
    n_days = 366 if is_leap_year(year) else 365
    start_doys = np.arange(1, n_days + 1, 8, dtype=np.int32)
    jan1 = datetime(year, 1, 1)
    dates = np.array(
        [(jan1 + timedelta(days=int(d - 1))).strftime("%Y%m%d") for d in start_doys],
        dtype="U8",
    )
    return n_days, start_doys, dates


# ---------------------------------------------------------------------------
# Grid and mask
# ---------------------------------------------------------------------------
def build_destination_grid() -> dict:
    width = int(round((LON_MAX - LON_MIN) / RES))
    height = int(round((LAT_MAX - LAT_MIN) / RES))

    if width != NX or height != NY:
        raise ValueError(
            f"Grid mismatch: computed {height}x{width}, expected {NY}x{NX}. "
            "Check LON_MIN/LON_MAX/LAT_MIN/LAT_MAX/RES."
        )

    return {
        "width": width,
        "height": height,
        "crs": CRS.from_epsg(4326),
        "transform": from_origin(LON_MIN, LAT_MAX, RES, RES),
        "lon": LON_MIN + RES * (np.arange(width) + 0.5),
        "lat": LAT_MAX - RES * (np.arange(height) + 0.5),
    }


def load_cropland_mask() -> np.ndarray:
    land_cover = tifffile.imread(LANDCOVER_PATH)[:NY, X0:X1]
    return land_cover == CROPLAND_CLASS


def output_paths(year: int) -> dict:
    j = lambda name: os.path.join(OUTDIR, name)
    return {
        "fpar_raw": j(f"MCD15_FPAR_{year}_8day_reprojected_raw.npy"),
        "fpar_used": j(f"MCD15_FPAR_{year}_8day_reprojected_used.npy"),
        "dates": j(f"MCD15_FPAR_{year}_8day_dates.npy"),
        "start_doy": j(f"MCD15_FPAR_{year}_8day_startdoy.npy"),
        "available": j(f"MCD15_FPAR_{year}_8day_available.npy"),
        "fill_flag": j(f"MCD15_FPAR_{year}_8day_fillflag.npy"),
        "lon": j("MCD15_FPAR_lon.npy"),
        "lat": j("MCD15_FPAR_lat.npy"),
        "loess_slots": j(f"MCD15_FPAR_{year}_8day_loess.npy"),
        "loess_daily": j(f"MCD15_FPAR_{year}_daily_loess.flt"),
        "loess_valid_count": j(f"MCD15_FPAR_{year}_loess_valid_count.npy"),
        "loess_daily_count": j(f"MCD15_FPAR_{year}_loess_daily_valid_count.npy"),
    }


# ---------------------------------------------------------------------------
# Reprojection
# ---------------------------------------------------------------------------
def collect_source_files(year: int) -> dict:
    """Map YYYYMMDD -> file path for every MCD15A2H GeoTIFF belonging to `year`."""
    pattern = os.path.join(MCD15_DIR, "MCD15_FPAR_*.tif")
    file_map = {}
    for path in sorted(glob.glob(pattern)):
        match = re.search(r"MCD15_FPAR_(\d{8})\.tif$", os.path.basename(path))
        if match and match.group(1).startswith(str(year)):
            file_map[match.group(1)] = path
    return file_map


def reproject_one(path: str, grid: dict) -> np.ndarray:
    """Reproject one MCD15A2H FPAR GeoTIFF onto the analysis grid."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        arr[~np.isfinite(arr)] = np.nan

        source = np.where(np.isfinite(arr), arr, SRC_NODATA).astype(np.float32)
        destination = np.full((grid["height"], grid["width"]), DST_NODATA, np.float32)

        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=SRC_NODATA,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            dst_nodata=DST_NODATA,
            resampling=RESAMPLING,
        )

    destination[destination == DST_NODATA] = np.nan
    destination[(destination < 0.0) | (destination > 1.0)] = np.nan
    return destination


def fill_missing_slots(fpar_raw: np.ndarray, available: np.ndarray):
    """Fill 8-day composites that are entirely absent from the archive.

    Returns the filled array and a flag array: -1 still missing, 0 observed,
    1 filled by interpolation or edge extension.
    """
    n_slots = fpar_raw.shape[0]
    filled = fpar_raw.copy()
    flag = np.full(n_slots, -1, dtype=np.int8)
    flag[available] = 0

    if FILL_SINGLE_MISSING_SLOT:
        for i in range(1, n_slots - 1):
            if available[i] or not (available[i - 1] and available[i + 1]):
                continue

            before, after = fpar_raw[i - 1], fpar_raw[i + 1]
            interp = np.full_like(before, np.nan, dtype=np.float32)

            both = np.isfinite(before) & np.isfinite(after)
            only_before = np.isfinite(before) & ~np.isfinite(after)
            only_after = ~np.isfinite(before) & np.isfinite(after)

            interp[both] = 0.5 * (before[both] + after[both])
            interp[only_before] = before[only_before]
            interp[only_after] = after[only_after]

            filled[i] = interp
            flag[i] = 1

    if FILL_EDGE_NEAREST:
        if not available[0]:
            for j in range(1, n_slots):
                if np.isfinite(filled[j]).any():
                    filled[0], flag[0] = filled[j], 1
                    break
        if not available[-1]:
            for j in range(n_slots - 2, -1, -1):
                if np.isfinite(filled[j]).any():
                    filled[-1], flag[-1] = filled[j], 1
                    break

    return filled, flag


def load_or_build_reprojected(year: int, grid: dict, force: bool = False) -> dict:
    os.makedirs(OUTDIR, exist_ok=True)
    paths = output_paths(year)
    n_days, start_doys, dates = build_8day_slots(year)
    n_slots = len(dates)

    cached = [
        "fpar_raw", "fpar_used", "dates", "start_doy",
        "available", "fill_flag", "lon", "lat",
    ]
    if not force and all(os.path.exists(paths[k]) for k in cached):
        print("Reprojected MCD15A2H arrays found; loading cache.")
        return {
            "fpar_used": np.load(paths["fpar_used"]),
            "start_doys": np.load(paths["start_doy"]),
            "available": np.load(paths["available"]),
            "fill_flag": np.load(paths["fill_flag"]),
            "n_days": n_days,
        }

    file_map = collect_source_files(year)
    print(f"Year {year}: {n_days} days, {n_slots} 8-day slots, "
          f"{len(file_map)} source files found.")

    fpar_raw = np.full((n_slots, grid["height"], grid["width"]), np.nan, np.float32)
    available = np.zeros(n_slots, dtype=bool)

    for i, date in enumerate(tqdm(dates, desc="Reprojecting")):
        path = file_map.get(date)
        if path is None:
            continue
        fpar_raw[i] = reproject_one(path, grid)
        available[i] = True

    fpar_used, fill_flag = fill_missing_slots(fpar_raw, available)

    n_missing = int((fill_flag == -1).sum())
    n_filled = int((fill_flag == 1).sum())
    print(f"Slots observed {int(available.sum())}, filled {n_filled}, "
          f"still missing {n_missing}.")

    np.save(paths["fpar_raw"], fpar_raw)
    np.save(paths["fpar_used"], fpar_used)
    np.save(paths["dates"], dates)
    np.save(paths["start_doy"], start_doys)
    np.save(paths["available"], available)
    np.save(paths["fill_flag"], fill_flag)
    np.save(paths["lon"], grid["lon"])
    np.save(paths["lat"], grid["lat"])

    return {
        "fpar_used": fpar_used,
        "start_doys": start_doys,
        "available": available,
        "fill_flag": fill_flag,
        "n_days": n_days,
    }


# ---------------------------------------------------------------------------
# LOESS smoothing
# ---------------------------------------------------------------------------
def interpolate_gaps(y: np.ndarray, x: np.ndarray):
    """Linearly interpolate NaNs; extrapolate edges with the nearest value.

    Returns None if fewer than MIN_FINITE_SLOTS finite values are present.
    """
    finite = np.isfinite(y)
    n_finite = int(finite.sum())
    if n_finite < MIN_FINITE_SLOTS:
        return None
    if n_finite == y.size:
        return y.astype(np.float32, copy=False)
    return np.interp(x, x[finite], y[finite]).astype(np.float32, copy=False)


def smooth_pixel(y_slots, x_slots, x_daily, frac, it):
    """LOESS-smooth one pixel at 8-day resolution, then expand to daily."""
    y = interpolate_gaps(y_slots, x_slots)
    if y is None:
        return None, None

    z_slots = sm.nonparametric.lowess(
        y, x_slots, frac=frac, it=it, is_sorted=True, return_sorted=False
    ).astype(np.float32, copy=False)
    z_slots = np.clip(z_slots, 0.0, 1.0)

    z_daily = np.clip(np.interp(x_daily, x_slots, z_slots), 0.0, 1.0)
    return z_slots, z_daily.astype(np.float32, copy=False)


def smooth_chunk(flat_slots, idx_chunk, x_slots, x_daily, frac, it):
    n_slots, n_days = x_slots.size, x_daily.size

    out_slots = np.full((n_slots, idx_chunk.size), np.nan, np.float32)
    out_daily = np.full((n_days, idx_chunk.size), OUT_NODATA_INT16, np.int16)
    was_smoothed = np.zeros(idx_chunk.size, dtype=bool)

    y_chunk = flat_slots[:, idx_chunk]
    for j in range(idx_chunk.size):
        z_slots, z_daily = smooth_pixel(y_chunk[:, j], x_slots, x_daily, frac, it)
        if z_daily is None:
            continue
        out_slots[:, j] = z_slots
        out_daily[:, j] = np.rint(z_daily * DAILY_SCALE).astype(np.int16)
        was_smoothed[j] = True

    return idx_chunk, out_slots, out_daily, was_smoothed


def run_loess(year, fpar_used, start_doys, cropland,
              frac_days, it, n_jobs, chunk_size):
    paths = output_paths(year)
    n_days = 366 if is_leap_year(year) else 365
    n_slots = fpar_used.shape[0]

    x_slots = (start_doys - 1).astype(np.float32)
    x_daily = np.arange(n_days, dtype=np.float32)
    frac = frac_days / float(n_days)

    valid_idx = np.flatnonzero(cropland.ravel())
    print(f"Cropland pixels: {valid_idx.size:,} / {cropland.size:,}")
    print(f"LOWESS span {frac_days} days (frac={frac:.4f}), it={it}")

    flat_slots = fpar_used.reshape(n_slots, -1)

    daily = np.memmap(paths["loess_daily"], mode="w+",
                      dtype=np.int16, shape=(n_days, cropland.size))
    daily[:] = OUT_NODATA_INT16

    slots = np.full((n_slots, cropland.size), np.nan, np.float32)
    daily_count = np.zeros(cropland.size, dtype=np.uint16)
    slot_count = np.zeros(cropland.size, dtype=np.uint8)

    chunks = [valid_idx[i:i + chunk_size]
              for i in range(0, valid_idx.size, chunk_size)]

    n_smoothed = 0
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = [
            pool.submit(smooth_chunk, flat_slots, chunk, x_slots, x_daily, frac, it)
            for chunk in chunks
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="LOWESS"):
            idx, res_slots, res_daily, smoothed = future.result()
            slots[:, idx] = res_slots
            daily[:, idx] = res_daily
            daily_count[idx] = np.where(smoothed, n_days, 0).astype(np.uint16)
            slot_count[idx] = np.isfinite(res_slots).sum(axis=0).astype(np.uint8)
            n_smoothed += int(smoothed.sum())

    daily.flush()

    np.save(paths["loess_slots"], slots.reshape(n_slots, NY, NX))
    np.save(paths["loess_valid_count"], slot_count.reshape(NY, NX))
    np.save(paths["loess_daily_count"], daily_count.reshape(NY, NX))

    skipped = valid_idx.size - n_smoothed
    print(f"Smoothed {n_smoothed:,} cropland pixels; "
          f"{skipped:,} skipped (fewer than {MIN_FINITE_SLOTS} valid slots).")
    for key in ("loess_slots", "loess_daily", "loess_valid_count", "loess_daily_count"):
        print("Saved:", paths[key])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Reproject MODIS 8-day FPAR to the AHI grid and LOESS-smooth it."
    )
    p.add_argument("year", type=int, help="Four-digit year, e.g. 2018")
    p.add_argument("--frac_days", type=int, default=35,
                   help="LOWESS span in days (default 35, matching the AHI EVI2 "
                        "span of 0.095 on a 365-day series)")
    p.add_argument("--it", type=int, default=0,
                   help="LOWESS robustifying iterations (0 = none)")
    p.add_argument("--n_jobs", type=int, default=16, help="Worker threads")
    p.add_argument("--chunk", type=int, default=4000, help="Pixels per chunk")
    p.add_argument("--force_reproject", action="store_true",
                   help="Rebuild the reprojected arrays, ignoring any cache")
    p.add_argument("--skip_loess", action="store_true",
                   help="Stop after reprojection")
    return p.parse_args()


def main():
    args = parse_args()

    grid = build_destination_grid()
    cropland = load_cropland_mask()
    reprojected = load_or_build_reprojected(args.year, grid, args.force_reproject)

    if args.skip_loess:
        print("skip_loess set; stopping after reprojection.")
        return

    run_loess(
        year=args.year,
        fpar_used=reprojected["fpar_used"],
        start_doys=reprojected["start_doys"],
        cropland=cropland,
        frac_days=args.frac_days,
        it=args.it,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk,
    )


if __name__ == "__main__":
    main()
