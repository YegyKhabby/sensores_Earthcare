#!/usr/bin/env python3
"""
Heatmap of radar reflectivity Ze [dBZ] vs time for
20250606_juelich_categorize.nc

Three panels:
  (a) Raw Ze  – everything the radar sees
  (b) Rain-only Ze  – liquid falling hydrometeors, no ice/insects/clutter
  (c) Rain mask  – shows which pixels pass the rain filter

Mask logic (from category_bits):
  KEEP if:
    (category_bits & 2)  != 0   → falling hydrometeors
    (category_bits & 32) == 0   → not insects / birds
  NOT applied: quality_bits & 4 (ground clutter), insect_prob < 0.5,
               wet-bulb / liquid-phase mask (category_bits & 4).

Output: output/cloudnet/ze_heatmap_20250606.png
"""

from pathlib import Path
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone

# ── paths ─────────────────────────────────────────────────────────────────────
FPATH   = Path("/data/obs/site/jue/cloudnet/clu_processing/categorize/2025"
               "/20250606_juelich_categorize.nc")
OUT_DIR = Path(__file__).resolve().parents[1] / "output" / "cloudnet"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUT_DIR / "ze_heatmap_20250606.png"

# ── load data ─────────────────────────────────────────────────────────────────
with nc.Dataset(FPATH, "r") as ds:
    # time → datetime objects
    tvar   = ds.variables["time"]
    tunits = getattr(tvar, "units", "hours since 2025-06-06 00:00:00 +00:00")
    tcal   = getattr(tvar, "calendar", "standard")
    times  = nc.num2date(tvar[:], tunits, calendar=tcal)
    # convert to Python datetimes for matplotlib
    dts = np.array([datetime(t.year, t.month, t.day,
                             t.hour, t.minute, t.second,
                             tzinfo=timezone.utc) for t in times])

    # height [km AMSL]
    height_m  = ds.variables["height"][:]          # m AMSL
    height_km = height_m / 1e3

    # Ze [dBZ]  – fill masked/missing with NaN
    Z_raw = ds.variables["Z"][:]
    Z = np.ma.filled(Z_raw.astype(float), np.nan)  # (time, height)

    # bitmasks
    cat  = ds.variables["category_bits"][:]        # int32

# ── rain mask ─────────────────────────────────────────────────────────────────
rain_mask = (
    ((cat  & 2)  != 0) &   # falling hydrometeors
    ((cat  & 32) == 0)     # no insects / birds
    # Removed: quality_bits & 4 (ground clutter) — not applied
    # Removed: insect_prob < 0.5 — redundant with category_bits bit 5
    # Not applied: wet-bulb / liquid-phase mask (cat & 4)
)

Z_rain = np.where(rain_mask, Z, np.nan)

# ── plot ──────────────────────────────────────────────────────────────────────
HEIGHT_MAX_KM = 10.0          # clip display at 10 km
h_idx = height_km <= HEIGHT_MAX_KM

fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True,
                         gridspec_kw={"hspace": 0.08})

# common Ze colour range
ZE_MIN, ZE_MAX = -30, 30
cmap_ze = plt.get_cmap("turbo").copy()
cmap_ze.set_bad(color="white")          # NaN → white

cmap_mask = plt.get_cmap("Blues").copy()
cmap_mask.set_bad(color="white")

def draw_panel(ax, data, title, cmap, vmin, vmax, cbar_label):
    im = ax.pcolormesh(
        dts,
        height_km[h_idx],
        data[:, h_idx].T,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        shading="nearest",
    )
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
    cb.set_label(cbar_label, fontsize=9)
    ax.set_ylabel("Height AMSL [km]", fontsize=9)
    ax.set_ylim(0, HEIGHT_MAX_KM)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    # 0 °C line placeholder – would need Tw but approximate from inspection
    ax.axhline(2.5, color="cyan", lw=0.8, ls="--", label="~0 °C isotherm")
    return im

# panel (a) raw Ze
draw_panel(axes[0], Z,
           "(a) Raw Ze – all radar echoes",
           cmap_ze, ZE_MIN, ZE_MAX, "Ze [dBZ]")

# panel (b) Ze after insect + clutter masking (no wet-bulb filter)
draw_panel(axes[1], Z_rain,
           "(b) Masked Ze – insects/clutter removed (no wet-bulb filter)",
           cmap_ze, ZE_MIN, ZE_MAX, "Ze [dBZ]")

# panel (c) insect + clutter mask
draw_panel(axes[2], rain_mask.astype(float),
           "(c) Insect/clutter mask  (1 = passes filter)",
           cmap_mask, 0, 1, "mask")

# x-axis formatting
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
axes[2].xaxis.set_major_locator(mdates.HourLocator(interval=2))
axes[2].xaxis.set_minor_locator(mdates.HourLocator(interval=1))
axes[2].set_xlabel("Time UTC  –  2025-06-06", fontsize=10)
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=8)

fig.suptitle("Jülich CloudNet categorize  –  2025-06-06\n"
             "METEK MIRA-35  |  35.5 GHz",
             fontsize=11, y=0.995)

plt.savefig(OUTFILE, dpi=150, bbox_inches="tight")
print(f"Saved → {OUTFILE}")
plt.show()
