#!/usr/bin/env python3
"""
Convert Zg from dBZ (logarithmic) to linear Z [mm⁶/m³], save per-file .npz,
and print correct statistics alongside the (incorrect) dBZ-averaged values
for comparison.

No artefact masking is applied here — raw values are converted as-is.

Usage:
    python3 zg_linear_stats.py [date_dir] [out_dir]

    date_dir – directory with tower.znc files
               (default: /data/obs/site/jue/joyrad35/2025/06/06)
    out_dir  – where to write .npz files
               (default: ./linear_zg)

Output files:
    <out_dir>/<original_stem>.npz  with keys:
        Zg_dBZ      float32  (n_time, n_height)  – raw dBZ values
        Z_linear    float64  (n_time, n_height)  – linear Z  [mm⁶/m³]
        time_unix   float64  (n_time,)           – Unix timestamps [s]
        range_m     float32  (n_height,)         – range/height [m]
        filename    str                           – source filename
"""

import sys
from datetime import datetime
from pathlib import Path

import h5py
import h5py.h5s as h5s
import h5py.h5t as h5t
import numpy as np

DEFAULT_DATA_DIR = "/data/obs/site/jue/joyrad35/2025/06/06"
DEFAULT_OUT_DIR  = "/work/yegy_project/Feb2026_new/output/tower/npz"

# ── helpers ────────────────────────────────────────────────────────────────────

def read_f32(ds):
    shape = ds.shape
    buf   = np.zeros(shape, dtype="float32")
    fsp   = ds.id.get_space()
    msp   = h5s.create_simple(shape)
    ds.id.read(msp, fsp, buf, h5t.IEEE_F32LE)
    return buf

def read_time(ds):
    return ds.astype("float64")[:]

def to_dbz(z_linear):
    """Linear Z [mm⁶/m³] → dBZ = 10·log10(z).  NaN/zero → NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(z_linear > 0, 10.0 * np.log10(z_linear), np.nan)

def mean_dbz(z_linear):
    """Mean dBZ = 10·log10(nanmean(z_linear)).  Correct linear-domain mean."""
    m = np.nanmean(z_linear)
    return float(10.0 * np.log10(m)) if (m > 0 and not np.isnan(m)) else float("nan")

def parse_start_time(filename):
    p = Path(filename).name.split("_")
    return datetime(int(p[0][:4]), int(p[0][4:6]), int(p[0][6:8]),
                    int(p[1][:2]), int(p[1][2:4]), int(p[1][4:6]))

# ── setup ──────────────────────────────────────────────────────────────────────
data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR)
out_dir  = Path(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

files = sorted(data_dir.glob("*tower.znc"))
if not files:
    sys.exit(f"ERROR: No tower.znc files found in {data_dir}")

print(f"Source      : {data_dir}")
print(f"Output dir  : {out_dir}")
print(f"Files found : {len(files)}")
print(f"Zg storage  : LINEAR Z [mm⁶/m³]  (db=1 is a display hint, NOT log storage)")
print(f"dBZ conv    : Zg_dBZ = 10 * log10(Zg_stored)")
print(f"Masking     : Saturatedco == 1 (ADC-saturated pixels)\n")

# ── per-file table header ──────────────────────────────────────────────────────
FILE_W = 35
COL_W  = 14

h = (f"{'File':<{FILE_W}}  {'N_time':>7}  {'Sat':>4}  "
     f"{'Mean(dBZ)':>{COL_W}}  {'Max(dBZ)':>{COL_W}}  {'Min(dBZ)':>{COL_W}}  "
     f"{'Mean Z':>{COL_W}}  {'NaN%':>6}  Saved")
sep = (f"  {'':4}  {'[dBZ]':>{COL_W}}  {'[dBZ]':>{COL_W}}  "
       f"{'[dBZ]':>{COL_W}}  {'[mm⁶/m³]':>{COL_W}}")
print("=" * len(h))
print(f"PER-FILE STATISTICS  (Saturatedco masked, dBZ primary)")
print(f"  Zg_dBZ   = 10*log10(Z_linear)   ← dBZ from stored linear Z")
print(f"  Mean dBZ = 10*log10(nanmean(Z))  ← correct linear-domain average")
print("=" * len(h))
print(h)
print(" " * (FILE_W + 13) + sep)
print("-" * len(h))

# accumulate for daily totals
daily_lin_all  = []

for fp in files:
    fname = fp.name
    with h5py.File(fp, "r") as f:
        z_lin   = read_f32(f["Zg"]).astype("float64")   # already linear Z [mm⁶/m³]
        sat     = read_f32(f["Saturatedco"])
        range_m = read_f32(f["range"])
        try:
            t_unix = read_time(f["time"])
        except Exception:
            start  = parse_start_time(fname)
            t_unix = np.array([start.timestamp() + i for i in range(z_lin.shape[0])])

    n_sat = int((sat == 1.0).sum())
    z_lin[sat == 1.0] = np.nan                          # mask ADC-saturated pixels
    zg_dbz = to_dbz(z_lin)                             # dBZ = 10*log10(Z_linear)

    # ── per-file stats ─────────────────────────────────────────────────────────
    lin_mean  = float(np.nanmean(z_lin))
    lin_max   = float(np.nanmax(z_lin))
    lin_min   = float(np.nanmin(z_lin))
    mean_db   = mean_dbz(z_lin)                                   # 10*log10(nanmean)
    max_db    = float(10*np.log10(lin_max)) if lin_max > 0 else float("nan")
    min_db    = float(10*np.log10(lin_min)) if lin_min > 0 else float("nan")
    nan_pct   = 100.0 * np.isnan(z_lin).sum() / z_lin.size

    # ── save .npz ──────────────────────────────────────────────────────────────
    out_path = out_dir / (fp.stem + ".npz")
    np.savez_compressed(
        out_path,
        Zg_dBZ    = zg_dbz.astype("float32"),  # 10*log10(Z_linear)  [dBZ]
        Z_linear  = z_lin.astype("float32"),   # stored Zg = linear Z [mm⁶/m³]
        time_unix = t_unix,
        range_m   = range_m,
        filename  = np.array(fname),
    )

    # ── print row ──────────────────────────────────────────────────────────────
    print(f"{fname:<{FILE_W}}  {z_lin.shape[0]:>7}  {n_sat:>4}  "
          f"{mean_db:>{COL_W}.3f}  {max_db:>{COL_W}.3f}  {min_db:>{COL_W}.3f}  "
          f"{lin_mean:>{COL_W}.4e}  {nan_pct:>6.1f}  {out_path.name}")

    daily_lin_all.append(z_lin)

# ══════════════════════════════════════════════════════════════════════════════
# DAILY SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
all_lin = np.vstack(daily_lin_all)

print("-" * len(h))
print(f"\n{'='*80}")
print("DAILY TOTALS  (Saturatedco masked, all heights)")
print(f"{'='*80}")
print(f"  Total time steps : {all_lin.shape[0]:,}")
print(f"  Height gates     : {all_lin.shape[1]:,}")
print(f"  Total pixels     : {all_lin.size:,}")
print()
CW = 22
print(f"  {'Metric':<30}  {'Value':>{CW}}  Units")
print(f"  {'-'*60}")
print(f"  {'Mean Z (linear)':<30}  {np.nanmean(all_lin):>{CW}.4e}  mm⁶/m³")
print(f"  {'Mean Z (dBZ) = 10·log10(mean Z)':<30}  {mean_dbz(all_lin):>{CW}.3f}  dBZ")
print(f"  {'Median Z (linear)':<30}  {np.nanmedian(all_lin):>{CW}.4e}  mm⁶/m³")
print(f"  {'Max Z (linear)':<30}  {np.nanmax(all_lin):>{CW}.4e}  mm⁶/m³")
print(f"  {'Max Z (dBZ)':<30}  {float(10*np.log10(np.nanmax(all_lin))):>{CW}.2f}  dBZ")
print(f"  {'Min Z (linear)':<30}  {np.nanmin(all_lin):>{CW}.4e}  mm⁶/m³")
print(f"  {'Min Z (dBZ)':<30}  {float(10*np.log10(np.nanmin(all_lin))):>{CW}.2f}  dBZ")
print(f"  {'Missing (NaN)':<30}  {np.isnan(all_lin).sum():>{CW},}  pixels  "
      f"({100*np.isnan(all_lin).sum()/all_lin.size:.1f} %)")
print(f"{'='*80}")
print(f"\nSaved {len(files)} .npz files to: {out_dir}")
