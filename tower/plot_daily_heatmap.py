#!/usr/bin/env python3
"""
Daily reflectivity heatmap: concatenate all *tower.znc files for one day.

Usage:
    python3 plot_daily_heatmap.py [date_dir]

    date_dir  – path containing tower.znc files
                (default: /data/obs/site/jue/joyrad35/2025/06/06)
"""

import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import h5py.h5s as h5s
import h5py.h5t as h5t
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── configuration ──────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = "/data/obs/site/jue/joyrad35/2025/06/06"
OUTPUT_DBZ  = "/work/yegy_project/Feb2026_new/output/tower/zg_daily_heatmap_dBZ.png"
OUTPUT_Z    = "/work/yegy_project/Feb2026_new/output/tower/zg_daily_heatmap_Z.png"

data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR)

# ── helpers ────────────────────────────────────────────────────────────────────

def read_f32(ds):
    """Read an HDF5 dataset into a float32 ndarray using the low-level API."""
    shape = ds.shape
    buf   = np.zeros(shape, dtype="float32")
    fsp   = ds.id.get_space()
    msp   = h5s.create_simple(shape)
    ds.id.read(msp, fsp, buf, h5t.IEEE_F32LE)
    return buf


def read_time(ds):
    """Read Unix timestamps safely, even when h5py reports H5T_NO_CLASS."""
    return ds.astype("float64")[:]


def to_dbz(z_linear):
    """Linear Z [mm⁶/m³] → dBZ = 10·log10(z).  Zero/negative/NaN → NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(z_linear > 0, 10.0 * np.log10(z_linear), np.nan)


def parse_start_time(filename: str) -> datetime:
    """Extract start datetime from  YYYYMMDD_HHMMSS_tower.znc filename."""
    parts = Path(filename).name.split("_")
    return datetime(
        int(parts[0][:4]),  int(parts[0][4:6]),  int(parts[0][6:8]),
        int(parts[1][:2]),  int(parts[1][2:4]),  int(parts[1][4:6]),
    )


# ── discover and sort files ────────────────────────────────────────────────────
files = sorted(data_dir.glob("*tower.znc"))
if not files:
    sys.exit(f"ERROR: No tower.znc files found in {data_dir}")

print(f"Found {len(files)} files in {data_dir}")

# ── read range axis from first file (assumed identical for all) ────────────────
with h5py.File(files[0], "r") as f0:
    range_m = read_f32(f0["range"])          # shape (445,) in metres

# ── concatenate time + Zg across all files ────────────────────────────────────
all_times    = []   # flat list of datetime objects (all files)
all_zg       = []   # list of 2-D dBZ arrays per file (n_time, n_height)
all_zg_lin   = []   # list of 2-D linear Z arrays per file [mm⁶/m³]
time_blocks  = []   # same datetimes but kept per-file for edge building

for fp in files:
    fname = fp.name
    try:
        with h5py.File(fp, "r") as f:
            zg    = read_f32(f["Zg"]).astype("float64")   # linear Z [mm⁶/m³]
            sat   = read_f32(f["Saturatedco"])            # 0/1 ADC-saturation flag
            t_raw = read_time(f["time"])                  # Unix seconds
            times = [datetime.utcfromtimestamp(float(t)) for t in t_raw]
    except Exception as e:
        # Fall back to filename-based time estimate
        print(f"  WARNING {fname}: {e}  → using filename timestamps")
        start = parse_start_time(fname)
        with h5py.File(fp, "r") as f:
            zg  = read_f32(f["Zg"]).astype("float64")
            sat = read_f32(f["Saturatedco"])
        n     = zg.shape[0]
        times = [start + timedelta(seconds=i) for i in range(n)]

    # Mask ADC-saturated pixels (Saturatedco == 1; typically 1 pixel/step at
    # range gate 1, ~0.18 km — persistent near-field / tower clutter)
    zg[sat == 1.0] = np.nan

    # Convert stored linear Z [mm⁶/m³] → dBZ  (Zg storage is linear, not dBZ)
    zg_dbz = to_dbz(zg)   # NaN where zg is 0 / negative / already NaN

    all_times.extend(times)
    all_zg.append(zg_dbz)
    all_zg_lin.append(zg.copy())    # zg is already sat-masked linear Z
    time_blocks.append(times)
    print(f"  {fname}  →  {len(times)} time steps  "
          f"[{times[0].strftime('%H:%M:%S')} – {times[-1].strftime('%H:%M:%S')}]")

n_heights = len(range_m)
print(f"\nTotal profiles: {sum(len(t) for t in time_blocks)}")
print(f"Day coverage: {all_times[0]}  →  {all_times[-1]}")

# ── build expanded edge arrays for shading='flat' ─────────────────────────────
# Each ~2-minute file block is expanded to fill VISUAL_WIDTH_MIN minutes so
# the data is clearly visible even on a full-day x-axis.  A single NaN column
# is inserted in the remaining gap to keep the space blank.
VISUAL_WIDTH_MIN = 20.0          # minutes each file block occupies visually

nan_col      = np.full((1, n_heights), np.nan)
edges_list   = []   # X cell-edge values (n_total_cols + 1,)
zg_cols      = []   # dBZ column blocks
zg_lin_cols  = []   # linear Z column blocks [mm⁶/m³]

for i, (zg_block, zlin_block, t_block) in enumerate(zip(all_zg, all_zg_lin, time_blocks)):
    n  = len(t_block)
    t0 = mdates.date2num(t_block[0])
    t1 = t0 + VISUAL_WIDTH_MIN / 1440.0          # expand to visual width
    file_edges = np.linspace(t0, t1, n + 1)      # N+1 edges for N columns

    if i == 0:
        edges_list.extend(file_edges.tolist())
    else:
        # NaN separator column spanning gap from prev visual end → this t0
        edges_list.append(float(file_edges[0]))   # right edge of separator
        zg_cols.append(nan_col)
        zg_lin_cols.append(nan_col)
        edges_list.extend(file_edges[1:].tolist())

    zg_cols.append(zg_block)       # (n, n_heights)  dBZ
    zg_lin_cols.append(zlin_block) # (n, n_heights)  linear Z

zg_plot_dBZ  = np.vstack(zg_cols)      # (total_cols, n_heights)  dBZ
zg_plot_lin  = np.vstack(zg_lin_cols)  # (total_cols, n_heights)  linear Z [mm⁶/m³]
edges_arr    = np.array(edges_list)    # (total_cols + 1,)

# Height cell edges from centres (uniform gate spacing)
dh      = float(np.mean(np.diff(range_m)))
h_edges = np.concatenate([range_m - dh / 2, [range_m[-1] + dh / 2]])

print(f"Plot columns: {zg_plot_dBZ.shape[0]}  "
      f"({len(all_zg)} file blocks + {len(all_zg)-1} NaN separators)")

# ── plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 7))

# shading='flat' + explicit edges: each file block fills VISUAL_WIDTH_MIN
# minutes; gap columns are NaN → rendered as white (blank).
pcm = ax.pcolormesh(
    edges_arr,      # (n_cols + 1,) X edges
    h_edges,        # (n_heights + 1,) Y edges
    zg_plot_dBZ.T,  # dBZ values
    cmap="viridis",
    vmin=-35,
    vmax=20,
    shading="flat",
    rasterized=True,
)

# ── time axis formatting ───────────────────────────────────────────────────────
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

# Span the full day on x-axis so gaps between files are visible
day = all_times[0].date()
ax.set_xlim(
    mdates.date2num(datetime(day.year, day.month, day.day, 0, 0, 0)),
    mdates.date2num(datetime(day.year, day.month, day.day, 23, 59, 59)),
)

# ── height / y-axis ───────────────────────────────────────────────────────────
km_ticks = np.arange(0, range_m[-1] + 1, 2000)          # every 2 km
ax.set_yticks(km_ticks)
ax.set_yticklabels([f"{v/1000:.0f}" for v in km_ticks])

# ── labels ─────────────────────────────────────────────────────────────────────
day_str = all_times[0].strftime("%Y-%m-%d")
ax.set_xlabel(f"Time UTC  ({day_str})", fontsize=12)
ax.set_ylabel("Height (km)", fontsize=12)
ax.set_title(
    f"Daily Reflectivity Heatmap – Zg  |  {day_str}  |  Jülich 35 GHz",
    fontsize=14, fontweight="bold",
)

cbar = plt.colorbar(pcm, ax=ax, pad=0.01, label="Zg (dBZ) = 10\u00b7log\u2081\u2080(Z)")

plt.tight_layout()
plt.savefig(OUTPUT_DBZ, dpi=150, bbox_inches="tight")
print(f"\nSaved (dBZ) → {OUTPUT_DBZ}")
plt.close(fig)

# ── PLOT 2: Linear Z [mm⁶/m³], logarithmic colorscale ─────────────────────────────
fig2, ax2 = plt.subplots(figsize=(18, 7))
pcm2 = ax2.pcolormesh(
    edges_arr, h_edges, zg_plot_lin.T,
    cmap="plasma",
    norm=mcolors.LogNorm(vmin=1e-3, vmax=1e3),
    shading="flat", rasterized=True,
)
ax2.xaxis_date()
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax2.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax2.set_xlim(
    mdates.date2num(datetime(day.year, day.month, day.day, 0, 0, 0)),
    mdates.date2num(datetime(day.year, day.month, day.day, 23, 59, 59)),
)
km_ticks2 = np.arange(0, range_m[-1] + 1, 2000)
ax2.set_yticks(km_ticks2)
ax2.set_yticklabels([f"{v/1000:.0f}" for v in km_ticks2])
ax2.set_xlabel(f"Time UTC  ({day_str})", fontsize=12)
ax2.set_ylabel("Height (km)", fontsize=12)
ax2.set_title(
    f"Daily Reflectivity Heatmap – Linear Z  |  {day_str}  |  Jülich 35 GHz",
    fontsize=14, fontweight="bold",
)
plt.colorbar(pcm2, ax=ax2, pad=0.01, label="Z  [mm⁶/m³]  (log scale)")
plt.tight_layout()
plt.savefig(OUTPUT_Z, dpi=150, bbox_inches="tight")
print(f"Saved  (Z)  → {OUTPUT_Z}")
plt.close(fig2)

# ── summary ────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DAILY SUMMARY")
print("=" * 70)
n_real = sum(len(t) for t in time_blocks)
print(f"  Files processed : {len(files)}")
print(f"  Total time steps: {n_real}")
print(f"  Height gates    : {n_heights}  "
      f"({range_m[0]/1000:.2f} – {range_m[-1]/1000:.2f} km)")
print(f"  dBZ min / max   : {np.nanmin(zg_plot_dBZ):.1f} / {np.nanmax(zg_plot_dBZ):.1f} dBZ")
print(f"  Z   min / max   : {np.nanmin(zg_plot_lin):.3e} / {np.nanmax(zg_plot_lin):.3e} mm⁶/m³")
print(f"  Missing (NaN)   : {np.isnan(zg_plot_dBZ).sum()} "
      f"({100*np.isnan(zg_plot_dBZ).sum()/zg_plot_dBZ.size:.1f} %)")
print(f"  Start           : {all_times[0]}")
print(f"  End             : {all_times[-1]}")
print("=" * 70)
