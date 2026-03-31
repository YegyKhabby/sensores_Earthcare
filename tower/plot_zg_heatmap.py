#!/usr/bin/env python3
"""
Single-file (~2-min) reflectivity heatmap for one *tower.znc file.

Produces TWO output figures:
  1.  zg_heatmap_Z_2min.png   – linear Z [mm⁶/m³], logarithmic colorscale
  2.  zg_heatmap_dBZ_2min.png – Zg (dBZ) = 10·log10(Z_linear)

Storage fact: Zg is stored as LINEAR Z [mm⁶/m³] (db=1 is a display hint).
Saturatedco == 1 pixels are masked (ADC-saturated, near-field clutter).

Usage:
    python3 plot_zg_heatmap.py [znc_file]

    znc_file – path to a single *tower.znc file
               (default: first file found in DEFAULT_DATA_DIR)
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import h5py.h5s as h5s
import h5py.h5t as h5t
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEFAULT_DATA_DIR = "/data/obs/site/jue/joyrad35/2025/06/06"
OUT_DIR          = Path("/work/yegy_project/Feb2026_new/output/tower")

# ── helpers ─────────────────────────────────────────────────────────────────────────────
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
    """Linear Z [mm⁶/m³] → dBZ = 10·log10(z).  Zero/NaN → NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(z_linear > 0, 10.0 * np.log10(z_linear), np.nan)

# ── find file ─────────────────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    fp = Path(sys.argv[1])
else:
    files = sorted(Path(DEFAULT_DATA_DIR).glob("*tower.znc"))
    if not files:
        sys.exit(f"ERROR: No tower.znc files in {DEFAULT_DATA_DIR}")
    fp = files[0]

print(f"File: {fp.name}")

# ── read data ─────────────────────────────────────────────────────────────────────────────
with h5py.File(fp, "r") as f:
    z_lin   = read_f32(f["Zg"]).astype("float64")   # linear Z [mm⁶/m³]
    sat     = read_f32(f["Saturatedco"])
    range_m = read_f32(f["range"])
    t_unix  = read_time(f["time"])

n_sat = int((sat == 1.0).sum())
z_lin[sat == 1.0] = np.nan                  # mask ADC-saturated pixels
zg_dbz = to_dbz(z_lin)                     # dBZ = 10·log10(Z_linear)
times  = [datetime.utcfromtimestamp(float(t)) for t in t_unix]

print(f"  Shape     : {z_lin.shape}  (time × height)")
print(f"  Duration  : {times[0].strftime('%H:%M:%S')} – {times[-1].strftime('%H:%M:%S')} UTC")
print(f"  Sat masked: {n_sat} pixels")
print(f"  Z range   : {np.nanmin(z_lin):.3e} – {np.nanmax(z_lin):.3e} mm⁶/m³")
print(f"  dBZ range : {np.nanmin(zg_dbz):.1f} – {np.nanmax(zg_dbz):.1f} dBZ")

# ── pcolormesh edges ──────────────────────────────────────────────────────────────────────────
t_num   = mdates.date2num(times)
dt      = np.diff(t_num)
t_edges = np.concatenate([[t_num[0] - dt[0]/2],
                           (t_num[:-1] + t_num[1:]) / 2,
                           [t_num[-1] + dt[-1]/2]])

dh      = float(np.mean(np.diff(range_m)))
h_edges = np.concatenate([range_m - dh/2, [range_m[-1] + dh/2]])

# ── shared axis formatting ───────────────────────────────────────────────────────────────────
def format_ax(ax, pcm, cb_label, title):
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.set_xlabel(f"Time UTC  ({times[0].strftime('%Y-%m-%d')})", fontsize=11)
    ax.set_ylabel("Height (km)", fontsize=11)
    km_ticks = np.arange(0, range_m[-1] + 1, 2000)
    ax.set_yticks(km_ticks)
    ax.set_yticklabels([f"{v/1000:.0f}" for v in km_ticks])
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.colorbar(pcm, ax=ax, label=cb_label)

# ── PLOT 1: Linear Z [mm⁶/m³], logarithmic colorscale ────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(14, 6))
pcm1 = ax1.pcolormesh(
    t_edges, h_edges, z_lin.T,
    cmap="plasma",
    norm=mcolors.LogNorm(vmin=1e-3, vmax=1e3),
    shading="flat", rasterized=True,
)
format_ax(ax1, pcm1,
          cb_label="Z  [mm⁶/m³]  (log scale)",
          title=f"Reflectivity – Linear Z  |  {fp.name}")
plt.tight_layout()
out_Z = OUT_DIR / "zg_heatmap_Z_2min.png"
plt.savefig(out_Z, dpi=150, bbox_inches="tight")
print(f"\nSaved (Z)   → {out_Z}")
plt.close(fig1)

# ── PLOT 2: dBZ = 10·log10(Z_linear) ───────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 6))
pcm2 = ax2.pcolormesh(
    t_edges, h_edges, zg_dbz.T,
    cmap="viridis",
    vmin=-35, vmax=20,
    shading="flat", rasterized=True,
)
format_ax(ax2, pcm2,
          cb_label="Zg (dBZ) = 10·log₁₀(Z)",
          title=f"Reflectivity – Zg (dBZ)  |  {fp.name}")
plt.tight_layout()
out_dBZ = OUT_DIR / "zg_heatmap_dBZ_2min.png"
plt.savefig(out_dBZ, dpi=150, bbox_inches="tight")
print(f"Saved (dBZ) → {out_dBZ}")
plt.close(fig2)

# ── summary ──────────────────────────────────────────────────────────────────────────────
lin_mean = float(np.nanmean(z_lin))
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  File       : {fp.name}")
print(f"  Time steps : {z_lin.shape[0]}")
print(f"  Hgt gates  : {z_lin.shape[1]}  ({range_m[0]/1000:.2f} – {range_m[-1]/1000:.2f} km)")
print(f"  Sat masked : {n_sat}")
print(f"  Z  min/max : {np.nanmin(z_lin):.3e} / {np.nanmax(z_lin):.3e} mm⁶/m³")
print(f"  dBZ min/max: {np.nanmin(zg_dbz):.1f} / {np.nanmax(zg_dbz):.1f} dBZ")
print(f"  Mean Z     : {lin_mean:.3e} mm⁶/m³")
print(f"  Mean dBZ   : {float(10*np.log10(lin_mean)):.2f} dBZ  [= 10·log10(mean Z)]")
print(f"  Missing    : {np.isnan(z_lin).sum()} px  "
      f"({100*np.isnan(z_lin).sum()/z_lin.size:.1f} %)")
print("=" * 60)
