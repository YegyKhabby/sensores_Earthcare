#!/usr/bin/env python3
"""
Per-file and daily Zg reflectivity statistics — no masking applied.

Reports, for every tower.znc file in the directory:
  - max / min / mean (linear-averaged) / median
  - NaN percentage (below-noise pixels)

Usage:
    python3 zg_stats_all_file_day.py [date_dir]
"""

import sys
from pathlib import Path

import h5py
import h5py.h5s as h5s
import h5py.h5t as h5t
import numpy as np

DEFAULT_DATA_DIR   = "/data/obs/site/jue/joyrad35/2025/06/06"

# ── helpers ────────────────────────────────────────────────────────────────────

def read_f32(ds):
    shape = ds.shape
    buf   = np.zeros(shape, dtype="float32")
    fsp   = ds.id.get_space()
    msp   = h5s.create_simple(shape)
    ds.id.read(msp, fsp, buf, h5t.IEEE_F32LE)
    return buf

def mean_dbz(z_linear):
    """Mean dBZ from linear Z [mm⁶/m³].
    Zg is stored LINEAR → mean dBZ = 10·log10(nanmean(Zg_stored)).
    NaN values (below-noise pixels) are excluded."""
    m = np.nanmean(z_linear)
    if m <= 0 or np.isnan(m):
        return float("nan")
    return float(10.0 * np.log10(m))

def to_dbz(z_linear):
    """Convert linear Z [mm⁶/m³] → dBZ.  Returns NaN for zero/negative."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(z_linear > 0, 10.0 * np.log10(z_linear), np.nan)


# ── discover files ─────────────────────────────────────────────────────────────
data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR)
files    = sorted(data_dir.glob("*tower.znc"))

if not files:
    sys.exit(f"ERROR: No tower.znc files found in {data_dir}")

print(f"Directory : {data_dir}")
print(f"Files     : {len(files)}")
print(f"Zg storage: LINEAR Z [mm\u00b6/m\u00b3]  (db=1 is a display hint, NOT log storage)")
print(f"dBZ conv  : 10 * log10(Zg_stored)")
print(f"Masking   : Saturatedco == 1 (ADC-saturated pixels, always at range gate 1)\n")

# ── header ─────────────────────────────────────────────────────────────────────
HDR = (f"{'File':<35}  {'N_time':>7}  {'Max(dBZ)':>9}  "
       f"{'Min(dBZ)':>9}  {'Mean(dBZ)':>10}  {'Med(dBZ)':>9}  {'NaN%':>6}  {'Sat':>5}")
print(HDR)
print("-" * len(HDR))

# ── per-file loop ──────────────────────────────────────────────────────────────
all_files_zg = []

for fp in files:
    with h5py.File(fp, "r") as f:
        zg  = read_f32(f["Zg"]).astype("float64")          # linear Z [mm\u00b6/m\u00b3]
        sat = read_f32(f["Saturatedco"]).astype("float32")  # 0/1 flag

    n_sat = int((sat == 1.0).sum())
    zg[sat == 1.0] = np.nan                                 # mask ADC-saturated pixels

    n_time   = zg.shape[0]
    has_data = not np.all(np.isnan(zg))
    # all stats in dBZ via 10*log10 (Zg is linear)
    max_dbz  = float(10*np.log10(np.nanmax(zg)))    if has_data else float("nan")
    min_dbz  = float(10*np.log10(np.nanmin(zg)))    if has_data else float("nan")
    mean_val = mean_dbz(zg)                          if has_data else float("nan")
    med_dbz  = float(10*np.log10(np.nanmedian(zg))) if has_data else float("nan")
    nan_pct  = 100.0 * np.isnan(zg).sum() / zg.size

    print(f"{fp.name:<35}  {n_time:>7}  {max_dbz:>9.2f}  "
          f"{min_dbz:>9.2f}  {mean_val:>10.2f}  {med_dbz:>9.2f}  {nan_pct:>6.1f}  {n_sat:>5}")

    all_files_zg.append(zg)

# ── daily summary ──────────────────────────────────────────────────────────────
all_zg = np.vstack(all_files_zg)

print("\n" + "=" * len(HDR))
print("DAILY TOTALS  (Saturatedco masked)")
print("=" * len(HDR))
print(f"  Total time steps  : {all_zg.shape[0]}")
print(f"  Height gates      : {all_zg.shape[1]}")
print(f"  Total data points : {all_zg.size:,}")
print()
print(f"  Max dBZ           : {10*np.log10(np.nanmax(all_zg)):.2f} dBZ")
print(f"  Min dBZ           : {10*np.log10(np.nanmin(all_zg)):.2f} dBZ")
print(f"  Mean dBZ          : {mean_dbz(all_zg):.2f} dBZ  [= 10·log10(mean(Z_linear))]")
print(f"  Median dBZ        : {10*np.log10(np.nanmedian(all_zg)):.2f} dBZ")
print(f"  Missing / NaN     : {np.isnan(all_zg).sum():,}  "
      f"({100*np.isnan(all_zg).sum()/all_zg.size:.1f} %)")
print("=" * len(HDR))
