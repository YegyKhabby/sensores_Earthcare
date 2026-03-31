#!/usr/bin/env python3
"""
Compare Tower JOYRAD-35 Ze with CloudNet Ze for the same day.

Steps performed:
  1. Load CloudNet categorization NetCDF → Z [dBZ] on height-AGL grid
  2. Load tower .npz files → Zg_dBZ on range grid (add +120 m for AGL)
  3. Align time axes (nearest-neighbour interpolation, ±30 s tolerance)
  4. Align height axes (interpolate tower onto CloudNet height grid)
  5. Produce outputs:
       a. Side-by-side daily heatmaps (CloudNet Z | Tower Zg)
       b. Scatter plot: CloudNet Z vs Tower Zg (coloured by height)
       c. Bias/RMSE profile: mean(CloudNet Z - Tower Zg) per height bin

Usage:
    python3 compare_ze.py <cloudnet_file.nc> <tower_date_dir>

    cloudnet_file  – path to CloudNet categorization .nc file
    tower_date_dir – directory with *tower.znc files (or path to output/tower/npz/)

Output:
    output/comparison/Ze_heatmap_side_by_side_<date>.png
    output/comparison/Ze_scatter_<date>.png
    output/comparison/Ze_bias_profile_<date>.png

Height correction:
    Tower JOYRAD-35 sits at +120 m AGL on the Jülich tower.
    Applied as:  tower_height_AGL = range_m + 120

TODO: set CLOUDNET_FILE and TOWER_DATE_DIR below once paths are known.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
SITE_ALT_M            = 111.0    # JOYCE site altitude AMSL [m]
# ZNC range = vertical height above ground (SITE_ALT_M reference).
# CloudNet height = AMSL.  Relationship: ZNC range + SITE_ALT_M = CloudNet height.
TIME_TOL_S            = 30.0      # max allowed time mismatch for pairing [s]
OUT_DIR = Path("/work/yegy_project/Feb2026_new/output/comparison")

DEFAULT_CLOUDNET_FILE  = None     # e.g. "/data/cloudnet/jue/20250606_jue_categorize.nc"
DEFAULT_TOWER_NPZ_DIR  = "/work/yegy_project/Feb2026_new/output/tower/npz"

# ── parse args ────────────────────────────────────────────────────────────────
cloudnet_file = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(DEFAULT_CLOUDNET_FILE) if DEFAULT_CLOUDNET_FILE else None)
tower_npz_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(DEFAULT_TOWER_NPZ_DIR)

if cloudnet_file is None:
    print("Usage: python3 compare_ze.py <cloudnet_file.nc> [tower_npz_dir]")
    print("\nOnce you have a CloudNet file, set DEFAULT_CLOUDNET_FILE at the top of this script.")
    print(f"\nTower .npz files available in: {tower_npz_dir}")
    npz_files = sorted(tower_npz_dir.glob("*.npz"))
    print(f"  {len(npz_files)} files found")
    if npz_files:
        d = np.load(npz_files[0])
        print(f"  Keys: {list(d.keys())}")
        print(f"  Shape: {d['Zg_dBZ'].shape}  (time × height)")
        r = d['range_m']
        h_amsl = r + SITE_ALT_M   # ZNC range + site altitude = AMSL (= CloudNet height)
        print(f"  Range (above ground): {r[0]:.0f} – {r[-1]:.0f} m")
        print(f"  Height AMSL         : {h_amsl[0]:.0f} – {h_amsl[-1]:.0f} m"
              f"  (range + {SITE_ALT_M:.0f} m site altitude)")
    print("\nScript is ready — provide a CloudNet file to run the comparison.")
    sys.exit(0)

# ── everything below runs once cloudnet_file is set ───────────────────────────
try:
    import netCDF4
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.dates as mdates
    from scipy.interpolate import interp1d
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n  pip install netCDF4 scipy matplotlib")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load CloudNet ──────────────────────────────────────────────────────────
print(f"Loading CloudNet: {cloudnet_file.name}")
with netCDF4.Dataset(cloudnet_file, "r") as ds:
    t_raw   = ds.variables["time"][:]
    t_units = ds.variables["time"].units
    _dates   = netCDF4.num2date(t_raw, t_units)
    import calendar
    cn_times = np.array([calendar.timegm(t.timetuple()) for t in _dates])  # Unix

    # Try radar_reflectivity first (raw Ze, most comparable), then Z (corrected)
    ze_var  = "radar_reflectivity" if "radar_reflectivity" in ds.variables else "Z"
    cn_ze   = np.ma.filled(ds.variables[ze_var][:], np.nan)   # (n_time, n_height) dBZ
    cn_h    = ds.variables["height"][:]                        # AMSL [m]
    print(f"  Ze variable : {ze_var}  shape {cn_ze.shape}")
    print(f"  Height      : {cn_h[0]:.0f} – {cn_h[-1]:.0f} m AMSL")
    print(f"  Time        : {netCDF4.num2date(t_raw[0], t_units)} – "
          f"{netCDF4.num2date(t_raw[-1], t_units)}")

# ── 2. Load Tower .npz files ──────────────────────────────────────────────────
print(f"\nLoading tower .npz: {tower_npz_dir}")
npz_files = sorted(tower_npz_dir.glob("*.npz"))
if not npz_files:
    sys.exit(f"ERROR: no .npz files in {tower_npz_dir}")

tw_ze_list, tw_t_list = [], []
for fp in npz_files:
    d = np.load(fp)
    tw_ze_list.append(d["Zg_dBZ"])
    tw_t_list.append(d["time_unix"])

tw_ze    = np.vstack(tw_ze_list)   # (n_time, n_height)  dBZ
tw_t     = np.concatenate(tw_t_list)
tw_range = np.load(npz_files[0])["range_m"]
# ZNC range = vertical height above ground (SITE_ALT_M reference).
# Adding site altitude gives AMSL, matching CloudNet height grid directly.
tw_h_amsl = tw_range + SITE_ALT_M
print(f"  Files       : {len(npz_files)}")
print(f"  Shape       : {tw_ze.shape}")
print(f"  Height AMSL : {tw_h_amsl[0]:.0f} – {tw_h_amsl[-1]:.0f} m")

# ── 3. Time alignment ─────────────────────────────────────────────────────────
print("\nAligning time axes...")
matched_cn, matched_tw = [], []
for i, t_cn in enumerate(cn_times):
    diffs = np.abs(tw_t - t_cn)
    j     = int(np.argmin(diffs))
    if diffs[j] <= TIME_TOL_S:
        matched_cn.append(i)
        matched_tw.append(j)

matched_cn = np.array(matched_cn)
matched_tw = np.array(matched_tw)
print(f"  Matched pairs: {len(matched_cn)}  (tolerance ±{TIME_TOL_S:.0f} s)")

if len(matched_cn) == 0:
    sys.exit("ERROR: no overlapping time steps found between CloudNet and tower data.")

cn_ze_m  = cn_ze[matched_cn]    # (n_match, n_cn_height)
tw_ze_m  = tw_ze[matched_tw]    # (n_match, n_tw_height)
t_match  = tw_t[matched_tw]
datetimes = [datetime.utcfromtimestamp(float(t)) for t in t_match]

# ── 4. Height alignment (interpolate tower → CloudNet grid) ───────────────────
# CloudNet heights are AGL at the JOYCE site.
# Both grids are now on the same AMSL reference.
# ZNC tw_h_amsl = CloudNet cn_h — same values, same spacing.
print("Aligning height axes...")
cn_h     = np.array(cn_h)   # ensure plain ndarray (not masked)
h_common = cn_h[(cn_h >= tw_h_amsl[0]) & (cn_h <= tw_h_amsl[-1])]
interp       = interp1d(tw_h_amsl, tw_ze_m, axis=1,
                        bounds_error=False, fill_value=np.nan)
tw_ze_interp = interp(h_common)                    # (n_match, n_common)
cn_ze_common = cn_ze_m[:, np.isin(cn_h, h_common)] # (n_match, n_common)
print(f"  Common height range (AMSL): {h_common[0]:.0f} – {h_common[-1]:.0f} m  "
      f"({len(h_common)} gates)")

# ── 5a. Side-by-side heatmap ──────────────────────────────────────────────────
print("\nPlotting side-by-side heatmap...")
t_num  = mdates.date2num(datetimes)
fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
kw = dict(cmap="viridis", vmin=-35, vmax=20, shading="nearest", rasterized=True)

pcm0 = axes[0].pcolormesh(t_num, h_common / 1000, cn_ze_common.T, **kw)
pcm1 = axes[1].pcolormesh(t_num, h_common / 1000, tw_ze_interp.T, **kw)

for ax, title in zip(axes, [f"CloudNet  {ze_var}", "Tower JOYRAD-35  Zg"]):
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.set_xlabel(f"Time UTC  ({datetimes[0].strftime('%Y-%m-%d')})", fontsize=11)
    ax.set_ylabel("Height (km)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.colorbar(pcm0, ax=ax, label="Ze (dBZ)")

plt.suptitle("CloudNet vs Tower Ze comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
date_str = datetimes[0].strftime("%Y%m%d")
out = OUT_DIR / f"Ze_heatmap_side_by_side_{date_str}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"  Saved → {out}")
plt.close(fig)

# ── 5b. Scatter plot ──────────────────────────────────────────────────────────
print("Plotting scatter...")
x = cn_ze_common.ravel()
y = tw_ze_interp.ravel()
mask = np.isfinite(x) & np.isfinite(y)
x, y = x[mask], y[mask]

h_rep = np.tile(h_common, (tw_ze_interp.shape[0], 1)).ravel()[mask]

bias = float(np.nanmean(y - x))
rmse = float(np.sqrt(np.nanmean((y - x)**2)))

fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(x, y, c=h_rep/1000, cmap="turbo", s=1, alpha=0.3, rasterized=True)
lim = (-40, 25)
ax.plot(lim, lim, "k--", lw=1, label="1:1")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(f"CloudNet Ze  ({ze_var})  [dBZ]", fontsize=12)
ax.set_ylabel("Tower Zg  [dBZ]", fontsize=12)
ax.set_title(f"Ze Scatter  |  {date_str}\nBias = {bias:+.2f} dBZ  |  RMSE = {rmse:.2f} dBZ",
             fontsize=12, fontweight="bold")
plt.colorbar(sc, ax=ax, label="Height (km)")
ax.legend()
plt.tight_layout()
out = OUT_DIR / f"Ze_scatter_{date_str}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"  Saved → {out}")
plt.close(fig)

# ── 5c. Bias profile ──────────────────────────────────────────────────────────
print("Plotting bias profile...")
diff    = tw_ze_interp - cn_ze_common     # Tower − CloudNet
bias_h  = np.nanmean(diff, axis=0)
std_h   = np.nanstd(diff, axis=0)
n_h     = np.sum(np.isfinite(diff), axis=0)

fig, ax = plt.subplots(figsize=(6, 9))
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.fill_betweenx(h_common/1000, bias_h - std_h, bias_h + std_h,
                 alpha=0.25, color="steelblue", label="±1 std")
ax.plot(bias_h, h_common/1000, color="steelblue", lw=2, label="Mean bias")
ax.set_xlabel("Tower − CloudNet  [dBZ]", fontsize=12)
ax.set_ylabel("Height (km)", fontsize=12)
ax.set_title(f"Ze Bias Profile  |  {date_str}", fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
out = OUT_DIR / f"Ze_bias_profile_{date_str}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"  Saved → {out}")
plt.close(fig)

print("\nDone.")
