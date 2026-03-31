#!/usr/bin/env python3
"""
Low-altitude bin analysis: statistics for the lowest N height bins,
per file and per bin, plus a time-resolved table.

Usage:
    python3 low_bins_stats.py [date_dir] [n_bins]

    date_dir – directory with tower.znc files
               (default: /data/obs/site/jue/joyrad35/2025/06/06)
    n_bins   – number of lowest bins to analyse  (default: 10)
"""

import sys
from datetime import datetime
from pathlib import Path

import h5py
import h5py.h5s as h5s
import h5py.h5t as h5t
import numpy as np

DEFAULT_DATA_DIR   = "/data/obs/site/jue/joyrad35/2025/06/06"
N_BINS_DEFAULT     = 10

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

def parse_start_time(filename):
    p = Path(filename).name.split("_")
    return datetime(int(p[0][:4]), int(p[0][4:6]), int(p[0][6:8]),
                    int(p[1][:2]), int(p[1][2:4]), int(p[1][4:6]))

def bar(value, vmin, vmax, width=20):
    """ASCII bar chart cell."""
    if np.isnan(value):
        return " " * width
    frac  = max(0.0, min(1.0, (value - vmin) / max(vmax - vmin, 1e-9)))
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)

def mean_dbz(z_linear):
    """Mean dBZ from linear Z [mm⁶/m³]: 10·log10(nanmean(z_linear)).
    Zg is stored LINEAR, so this is the only correct formula."""
    m = np.nanmean(z_linear)
    if m <= 0 or np.isnan(m):
        return float("nan")
    return float(10.0 * np.log10(m))

def to_dbz(z):
    """Element-wise linear Z → dBZ.  NaN/zero stay NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(z > 0, 10.0 * np.log10(z), np.nan)

# ── args ───────────────────────────────────────────────────────────────────────
data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR)
N_BINS   = int(sys.argv[2]) if len(sys.argv) > 2 else N_BINS_DEFAULT

files = sorted(data_dir.glob("*tower.znc"))
if not files:
    sys.exit(f"ERROR: No tower.znc files found in {data_dir}")

# ── read range axis once ───────────────────────────────────────────────────────
with h5py.File(files[0], "r") as f0:
    range_m = read_f32(f0["range"])[:N_BINS]      # lowest N bins

bin_labels = [f"{r/1000:.3f} km" for r in range_m]
BIN_W      = max(len(b) for b in bin_labels) + 2

# ── collect per-file data ──────────────────────────────────────────────────────
records = []          # one dict per file

for fp in files:
    fname = fp.name
    with h5py.File(fp, "r") as f:
        zg_full = read_f32(f["Zg"])                    # linear Z [mm⁶/m³]
        sat     = read_f32(f["Saturatedco"])            # 0/1 flag
        try:
            t_raw = read_time(f["time"])
            times = [datetime.utcfromtimestamp(float(t)) for t in t_raw]
        except Exception:
            start = parse_start_time(fname)
            times = [start]

    zg_full[sat == 1.0] = np.nan                       # mask ADC-saturated pixels
    zg = zg_full[:, :N_BINS]                           # (n_time, N_BINS)
    zg_dbz = to_dbz(zg)                               # (n_time, N_BINS) in dBZ

    rec = {
        "file"   : fname,
        "start"  : times[0],
        "end"    : times[-1],
        "n_time" : zg.shape[0],
        "zg"     : zg,                                 # linear Z [mm⁶/m³]
        # per-bin stats in dBZ (using 10*log10 correctly)
        "bin_mean"   : np.array([mean_dbz(zg[:, b]) for b in range(N_BINS)]),
        "bin_max"    : np.where(np.nanmax(zg, axis=0) > 0,
                                10*np.log10(np.nanmax(zg, axis=0)), np.nan),
        "bin_min"    : np.where(np.nanmin(zg, axis=0) > 0,
                                10*np.log10(np.nanmin(zg, axis=0)), np.nan),
        "bin_std"    : np.nanstd(zg_dbz, axis=0),     # std of dBZ values
        "bin_nan_pct": 100.0 * np.isnan(zg).sum(axis=0) / zg.shape[0],
        # overall stats for this file in dBZ
        "overall_mean" : mean_dbz(zg),
        "overall_max"  : float(10*np.log10(np.nanmax(zg))) if np.nanmax(zg) > 0 else float("nan"),
        "overall_min"  : float(10*np.log10(np.nanmin(zg))) if np.nanmin(zg) > 0 else float("nan"),
    }
    records.append(rec)


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 – Per-file × per-bin mean reflectivity
# ══════════════════════════════════════════════════════════════════════════════
FILE_W  = 35
CELL_W  = 7

print("\n" + "=" * 80)
print(f"TABLE 1 — Mean Zg (dBZ) per file × per bin  [lowest {N_BINS} bins, Saturatedco masked, 10·log10(mean(Z_linear))]"
      f"\n          Directory: {data_dir}")
print("=" * 80)

# header
header = f"{'File':<{FILE_W}}  {'Start':8}  {'End':8}  " + \
         "".join(f"{b:>{CELL_W}}" for b in bin_labels)
print(header)
print("-" * len(header))

for r in records:
    row  = f"{r['file']:<{FILE_W}}  "
    row += f"{r['start'].strftime('%H:%M:%S')}  {r['end'].strftime('%H:%M:%S')}  "
    row += "".join(
        f"{v:>{CELL_W}.2f}" if not np.isnan(v) else f"{'NaN':>{CELL_W}}"
        for v in r["bin_mean"]
    )
    print(row)

# bin totals row
print("-" * len(header))
all_stacked = np.vstack([r["zg"] for r in records])     # linear Z
daily_bin_mean = np.array([mean_dbz(all_stacked[:, b]) for b in range(N_BINS)])
row = f"{'DAILY MEAN':<{FILE_W}}  {'':8}  {'':8}  " + \
      "".join(f"{v:>{CELL_W}.2f}" for v in daily_bin_mean)
print(row)


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 – Per-bin summary across all files
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(f"TABLE 2 — Daily bin statistics  (all {len(files)} files combined)")
print("=" * 80)

all_low = np.vstack([r["zg"] for r in records])    # linear Z [mm⁶/m³]
all_low_dbz = to_dbz(all_low)                       # dBZ

col_w = 9
hdr2 = (f"{'Bin':>5}  {'Height':>10}  {'Mean(dBZ)':>{col_w}}  {'Med(dBZ)':>{col_w}}  "
        f"{'Max(dBZ)':>{col_w}}  {'Min(dBZ)':>{col_w}}  {'Std(dBZ)':>{col_w}}  "
        f"{'NaN%':>{col_w}}  {'Bar (mean)':25}")
print(hdr2)
print("-" * len(hdr2))

vmin_bar = float(np.nanmin(all_low_dbz))
vmax_bar = float(np.nanmax(all_low_dbz))

for b in range(N_BINS):
    col     = all_low[:, b]          # linear
    col_dbz = all_low_dbz[:, b]      # dBZ
    mean_v  = mean_dbz(col)
    med_v   = float(np.nanmedian(col_dbz))
    max_v   = float(np.nanmax(col_dbz))
    min_v   = float(np.nanmin(col_dbz))
    std_v   = float(np.nanstd(col_dbz))
    nan_p   = 100.0 * np.isnan(col).sum() / len(col)
    abar    = bar(mean_v, vmin_bar, vmax_bar, width=25)
    print(f"{b+1:>5}  {bin_labels[b]:>10}  {mean_v:>{col_w}.3f}  "
          f"{med_v:>{col_w}.3f}  {max_v:>{col_w}.3f}  "
          f"{min_v:>{col_w}.3f}  {std_v:>{col_w}.3f}  "
          f"{nan_p:>{col_w}.1f}  {abar}")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3 – Time series: per-file mean of lowest N bins
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(f"TABLE 3 — Time series: file-mean Zg across lowest {N_BINS} bins")
print("=" * 80)

hdr3 = (f"{'File':<{FILE_W}}  {'Start':8}  {'End':8}  "
        f"{'Mean':>7}  {'Max':>7}  {'Min':>7}  {'Std':>7}  "
        f"{'NaN%':>6}  {'Bar (mean)':25}")
print(hdr3)
print("-" * len(hdr3))

all_means = [r["overall_mean"] for r in records]
vm, vM    = min(all_means), max(all_means)

for r in records:
    nan_pct = 100.0 * np.isnan(r["zg"]).sum() / r["zg"].size
    abar    = bar(r["overall_mean"], vm, vM, width=25)
    print(f"{r['file']:<{FILE_W}}  "
          f"{r['start'].strftime('%H:%M:%S')}  {r['end'].strftime('%H:%M:%S')}  "
          f"{r['overall_mean']:>7.2f}  {r['overall_max']:>7.2f}  "
          f"{r['overall_min']:>7.2f}  "
          f"{float(np.nanstd(to_dbz(r['zg']))):>7.3f}  "
          f"{nan_pct:>6.1f}  {abar}")

# footer stats
print("-" * len(hdr3))
print(f"\n  Daily low-bin mean : {mean_dbz(all_low):.3f} dBZ  [= 10·log10(mean(Z_linear))]")
print(f"  Daily low-bin max  : {float(np.nanmax(all_low_dbz)):.3f} dBZ")
print(f"  Daily low-bin min  : {float(np.nanmin(all_low_dbz)):.3f} dBZ")
print(f"  Daily low-bin std  : {float(np.nanstd(all_low_dbz)):.3f} dBZ")
print(f"  Total profiles     : {all_low.shape[0]}")
print(f"  Height range       : {range_m[0]/1000:.3f} – {range_m[-1]/1000:.3f} km")
print("=" * 80)
