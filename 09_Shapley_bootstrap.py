#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""
Shapley attribution of interannual GPP variability, with pixel bootstrap.
 
Decomposes the annual departure of cropland GPP from its 2016-2023 climatology
among four drivers - PAR, FPAR, Tmin and VPD - using exact Shapley values
computed from the 16 counterfactual simulations produced by
08_GPP_counterfactual.py.
 
Two things are reported:
 
  phi_i(year)   the contribution of driver i to that year's departure from
                climatology                                [g C m-2 year-1]
 
  SD_i          the standard deviation of phi_i across 2016-2023, i.e. how much
                interannual variability in GPP that driver generates
                                                           [g C m-2 year-1]
 
SD_i is the summary statistic for the main figure; phi_i(year) is the detail
behind it. SD_i is preferred to a normalised share because it keeps physical
units, is comparable across drivers and regions, and cannot diverge in years
where the annual departure happens to be near zero.
 
Bootstrap
---------
Cropland pixels are resampled ONCE per replicate and the same sample is applied
to every year, so replicates are paired across years. An unpaired bootstrap
cannot give a valid interval for a statistic computed across years.
 
Only pixels that are valid in every year are used, so that the same pixel
population underlies all eight years.
 
The resampling is implemented with multinomial counts rather than index arrays:
a bootstrap mean is a weighted mean with integer weights, so each replicate
reduces to one matrix-vector product. This makes a full n-out-of-n bootstrap
tractable at the full cropland pixel count.
 
Input
-----
    cache_<name>/vpix_fixed/vpix_<year>.npy     shape (16, n_pixels)
 
    Row index is the driver bitmask used by 08_GPP_counterfactual.py:
        PAR = 1, FPAR = 2, Tmin = 4, VPD = 8
        mask 0  = all drivers at their 2016-2023 climatology
        mask 15 = all drivers at their year-specific values
 
Output
------
    <name>_shapley_by_year.csv        phi and share per year, with intervals
    <name>_shapley_interannual.csv    SD of phi across years, with intervals
    boot_<name>/shapley_boot.npz      bootstrap replicates, for later reuse
 
Usage
-----
    python 09_Shapley_bootstrap.py --name ChaoPhraya
    python 09_Shapley_bootstrap.py --name Java --n_boot 2000 --seed 0
"""
 
import argparse
import os
from math import factorial
 
import numpy as np
import pandas as pd
 
 
DRIVERS = ["PAR", "FPAR", "Tmin", "VPD"]
BITS = [1, 2, 4, 8]
N_DRIVERS = 4
N_SUBSETS = 1 << N_DRIVERS
YEARS = list(range(2016, 2024))
 
CACHE_DIR = os.environ.get("CACHE_DIR", "/path/to/ROI_timeseries_pixelwise")
 
 
# ---------------------------------------------------------------------------
# Shapley decomposition
# ---------------------------------------------------------------------------
def shapley_matrix() -> np.ndarray:
    """Linear operator M such that phi = M @ v, for v of length 16.
 
    phi_i = sum over subsets S not containing i of
            |S|! (n-|S|-1)! / n! * [ v(S + i) - v(S) ]
 
    Because the Shapley value is linear in v, the whole decomposition is a
    fixed 4x16 matrix. Building it once turns every later evaluation into a
    single small matrix product, and makes the linearity explicit: the Shapley
    values of a pixel mean equal the pixel mean of the Shapley values.
    """
    weights = [
        factorial(k) * factorial(N_DRIVERS - k - 1) / factorial(N_DRIVERS)
        for k in range(N_DRIVERS)
    ]
 
    matrix = np.zeros((N_DRIVERS, N_SUBSETS), dtype=np.float64)
    for i, bit in enumerate(BITS):
        for subset in range(N_SUBSETS):
            if subset & bit:
                continue
            w = weights[bin(subset).count("1")]
            matrix[i, subset | bit] += w
            matrix[i, subset] -= w
    return matrix
 
 
SHAPLEY = shapley_matrix()
 
 
def phi_from_v(v: np.ndarray) -> np.ndarray:
    """Shapley values from subset values.
 
    v may be (16,) or (n_years, 16) or (n_years, 16, n_replicates).
    The subset axis must be axis 1 when v is not one-dimensional.
    """
    if v.ndim == 1:
        return SHAPLEY @ v
    return np.tensordot(SHAPLEY, v, axes=([1], [1]))
 
 
# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
def load_all_years(name: str):
    """Load vpix for every year, keeping only pixels valid in all of them.
 
    Returns an array of shape (n_years, 16, n_common).
    """
    vpix_dir = os.path.join(CACHE_DIR, f"cache_{name}", "vpix_fixed")
 
    blocks = []
    for year in YEARS:
        path = os.path.join(vpix_dir, f"vpix_{year}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        block = np.load(path)
        if block.ndim != 2 or block.shape[0] != N_SUBSETS:
            raise ValueError(
                f"{path}: expected shape (16, n_pixels), got {block.shape}"
            )
        blocks.append(block.astype(np.float32, copy=False))
 
    widths = {b.shape[1] for b in blocks}
    if len(widths) != 1:
        raise ValueError(f"Years have differing pixel counts: {sorted(widths)}")
 
    stack = np.stack(blocks, axis=0)              # (n_years, 16, n_pixels)
    valid = np.isfinite(stack).all(axis=(0, 1))   # valid in every year and subset
 
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise RuntimeError("No pixel is valid in all years.")
 
    print(f"Pixels valid in all {len(YEARS)} years: {n_valid:,} of {valid.size:,} "
          f"({100.0 * n_valid / valid.size:.1f}%)")
 
    return np.ascontiguousarray(stack[:, :, valid])
 
 
# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def interannual_sd(phi: np.ndarray) -> np.ndarray:
    """SD across years. phi is (n_drivers, n_years) or (n_drivers, n_years, n_rep)."""
    return phi.std(axis=1, ddof=1)
 
 
def variance_share(phi: np.ndarray) -> np.ndarray:
    """Squared-contribution share, summing to one over drivers (axis 0)."""
    total = np.sum(phi ** 2, axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total > 0, phi ** 2 / total, np.nan)
    return share
 
 
def bootstrap(stack: np.ndarray, n_boot: int, n_draw: int, seed: int):
    """Paired pixel bootstrap.
 
    Returns phi_boot (n_drivers, n_years, n_boot) and sd_boot (n_drivers, n_boot).
    """
    n_years, _, n_pixels = stack.shape
    rng = np.random.default_rng(seed)
 
    # (n_years * 16, n_pixels), so one matrix-vector product gives every
    # subset value for every year at once.
    flat = stack.reshape(n_years * N_SUBSETS, n_pixels)
 
    v_boot = np.empty((n_years, N_SUBSETS, n_boot), dtype=np.float64)
 
    for b in range(n_boot):
        # A bootstrap mean is a weighted mean with multinomial integer weights.
        counts = np.bincount(
            rng.integers(0, n_pixels, size=n_draw), minlength=n_pixels
        ).astype(np.float32)
        v_boot[:, :, b] = (flat @ counts / n_draw).reshape(n_years, N_SUBSETS)
 
        if (b + 1) % max(1, n_boot // 10) == 0:
            print(f"  bootstrap {b + 1}/{n_boot}")
 
    phi_boot = phi_from_v(v_boot)          # (n_drivers, n_years, n_boot)
    sd_boot = interannual_sd(phi_boot)     # (n_drivers, n_boot)
    return phi_boot, sd_boot
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Shapley attribution of interannual GPP variability."
    )
    parser.add_argument("--name", required=True, help="Region name, e.g. ChaoPhraya")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--target_n", type=int, default=0,
                        help="Pixels per replicate; 0 (default) uses all valid "
                             "pixels, which is the standard n-out-of-n bootstrap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ci_low", type=float, default=2.5)
    parser.add_argument("--ci_high", type=float, default=97.5)
    args = parser.parse_args()
 
    stack = load_all_years(args.name)
    n_years, _, n_valid = stack.shape
 
    n_draw = n_valid if args.target_n <= 0 else min(args.target_n, n_valid)
    if n_draw < n_valid:
        print(f"WARNING: drawing {n_draw:,} of {n_valid:,} pixels per replicate. "
              "This is an m-out-of-n bootstrap; its intervals are wider than the "
              "sampling uncertainty of the full-sample estimate. Use --target_n 0 "
              "unless that is intended.")
 
    # ---- point estimates, from all valid pixels
    v_hat = stack.mean(axis=2).astype(np.float64)   # (n_years, 16)
    phi_hat = phi_from_v(v_hat)                     # (n_drivers, n_years)
    sd_hat = interannual_sd(phi_hat)                # (n_drivers,)
    share_hat = variance_share(phi_hat)             # (n_drivers, n_years)
 
    d_gpp = v_hat[:, 15] - v_hat[:, 0]
    residual = np.abs(phi_hat.sum(axis=0) - d_gpp)
    print(f"Additivity check, max |sum(phi) - dGPP|: {residual.max():.3e} "
          f"(should be ~0)")
 
    # ---- bootstrap
    print(f"Bootstrap: {args.n_boot} replicates, {n_draw:,} pixels each")
    phi_boot, sd_boot = bootstrap(stack, args.n_boot, n_draw, args.seed)
    share_boot = variance_share(phi_boot)
 
    pct = lambda a, q, axis: np.percentile(a, q, axis=axis)
    phi_lo = pct(phi_boot, args.ci_low, 2)
    phi_hi = pct(phi_boot, args.ci_high, 2)
    share_lo = pct(share_boot, args.ci_low, 2)
    share_hi = pct(share_boot, args.ci_high, 2)
    sd_lo = pct(sd_boot, args.ci_low, 1)
    sd_hi = pct(sd_boot, args.ci_high, 1)
 
    # ---- output
    os.makedirs(CACHE_DIR, exist_ok=True)
    boot_dir = os.path.join(CACHE_DIR, f"boot_{args.name}")
    os.makedirs(boot_dir, exist_ok=True)
 
    rows = []
    for y, year in enumerate(YEARS):
        row = {
            "region": args.name,
            "year": year,
            "GPP_base": v_hat[y, 0],
            "GPP_all": v_hat[y, 15],
            "dGPP": d_gpp[y],
            "phi_check": phi_hat[:, y].sum() - d_gpp[y],
        }
        for i, driver in enumerate(DRIVERS):
            row[f"phi_{driver}"] = phi_hat[i, y]
            row[f"phi_{driver}_lo"] = phi_lo[i, y]
            row[f"phi_{driver}_hi"] = phi_hi[i, y]
            row[f"share_{driver}"] = share_hat[i, y]
            row[f"share_{driver}_lo"] = share_lo[i, y]
            row[f"share_{driver}_hi"] = share_hi[i, y]
        rows.append(row)
 
    by_year = pd.DataFrame(rows)
    by_year_path = os.path.join(CACHE_DIR, f"{args.name}_shapley_by_year.csv")
    by_year.to_csv(by_year_path, index=False)
 
    summary = pd.DataFrame({
        "region": args.name,
        "driver": DRIVERS,
        "sd_phi": sd_hat,
        "sd_phi_lo": sd_lo,
        "sd_phi_hi": sd_hi,
        "mean_abs_phi": np.abs(phi_hat).mean(axis=1),
        "units": "g C m-2 year-1",
        "n_pixels": n_valid,
        "n_draw": n_draw,
        "n_boot": args.n_boot,
    })
    summary_path = os.path.join(CACHE_DIR, f"{args.name}_shapley_interannual.csv")
    summary.to_csv(summary_path, index=False)
 
    np.savez_compressed(
        os.path.join(boot_dir, "shapley_boot.npz"),
        region=args.name,
        drivers=np.array(DRIVERS),
        years=np.array(YEARS),
        phi_hat=phi_hat.astype(np.float32),
        sd_hat=sd_hat.astype(np.float32),
        phi_boot=phi_boot.astype(np.float32),
        sd_boot=sd_boot.astype(np.float32),
        n_pixels=np.int64(n_valid),
        n_draw=np.int64(n_draw),
        n_boot=np.int32(args.n_boot),
    )
 
    print()
    print(summary[["driver", "sd_phi", "sd_phi_lo", "sd_phi_hi"]]
          .to_string(index=False, float_format=lambda x: f"{x:9.2f}"))
    print()
    print("Saved:", by_year_path)
    print("Saved:", summary_path)
    print("Saved:", os.path.join(boot_dir, "shapley_boot.npz"))
 
 
if __name__ == "__main__":
    main()
 