#!/usr/bin/env python3
"""
plot_ze_heatmap_monthly.py
==========================
Monthly heatmap of radar reflectivity Ze [dBZ] vs time,
concatenating all available Cloudnet categorize files for a given month.

Three panels (same logic as plot_ze_heatmap_20250606.py):
  (a) Raw Ze           – everything the radar sees
  (b) Masked Ze        – falling hydrometeors, no insects/birds
  (c) Rain mask        – pixels passing the filter

Mask logic (category_bits):
  KEEP if  (category_bits & 2)  != 0   → falling hydrometeors
           (category_bits & 32) == 0   → no insects / birds

Usage
-----
  python3 plot_ze_heatmap_monthly.py           # defaults to 202506
  python3 plot_ze_heatmap_monthly.py 202506
  python3 plot_ze_heatmap_monthly.py 202507

Output
------
  output/cloudnet/ze_heatmap_monthly_<YYYYMM>.png
"""

import sys
import calendar
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── config ────────────────────────────────────────────────────────────────────
CLOUDNET_BASE = Path("/data/obs/site/jue/cloudnet/clu_processing/categorize")
OUT_DIR       = Path(__file__).resolve().parents[1] / "output" / "cloudnet"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEIGHT_MAX_KM = 10.0
ZE_MIN, ZE_MAX = -30, 30


# ── helpers ───────────────────────────────────────────────────────────────────
def load_day(nc_path):
    """
    Load one Cloudnet categorize file.
    Returns (dts, height_km, Z, cat) or None if the file is unreadable.
      dts        : (T,)      UTC datetime objects
      height_km  : (H,)      height in km AMSL
      Z          : (T, H)    Ze [dBZ], NaN where masked/missing
      cat        : (T, H)    category_bits int array
    """
    with nc.Dataset(nc_path, "r") as ds:
        tvar   = ds.variables["time"]
        tunits = getattr(tvar, "units", "")
        tcal   = getattr(tvar, "calendar", "standard")
        times  = nc.num2date(tvar[:], tunits, calendar=tcal)
        dts    = np.array([
            datetime(t.year, t.month, t.day,
                     t.hour, t.minute, t.second, tzinfo=timezone.utc)
            for t in times
        ])
        height_km = ds.variables["height"][:] / 1e3
        Z_raw     = ds.variables["Z"][:]
        Z         = np.ma.filled(Z_raw.astype(float), np.nan)
        cat       = np.array(ds.variables["category_bits"][:])
    return dts, height_km, Z, cat


# ── main ──────────────────────────────────────────────────────────────────────
def make_monthly_heatmap(year_month_str):
    yyyy = int(year_month_str[:4])
    mm   = int(year_month_str[4:6])
    n_days = calendar.monthrange(yyyy, mm)[1]
    month_label = f"{yyyy}-{mm:02d}"

    print(f"Loading {month_label} ({n_days} days) …")

    all_dts, all_Z, all_cat = [], [], []
    height_km = None
    days_loaded = 0

    for d in range(1, n_days + 1):
        date_str = f"{yyyy}{mm:02d}{d:02d}"
        nc_path  = CLOUDNET_BASE / str(yyyy) / f"{date_str}_juelich_categorize.nc"
        if not nc_path.exists():
            print(f"  {date_str}  SKIP (file not found)")
            continue
        try:
            dts, hkm, Z, cat = load_day(nc_path)
        except Exception as e:
            print(f"  {date_str}  ERROR: {e}")
            continue

        if height_km is None:
            height_km = hkm
        all_dts.append(dts)
        all_Z.append(Z)
        all_cat.append(cat)
        days_loaded += 1
        print(f"  {date_str}  ✓  {len(dts)} time steps")

    if days_loaded == 0:
        print("No data found — aborting.")
        return

    print(f"\nConcatenating {days_loaded} days …")
    dts_all = np.concatenate(all_dts)
    Z_all   = np.concatenate(all_Z,   axis=0)
    cat_all = np.concatenate(all_cat, axis=0)

    # sort by time (safety, in case files are out of order)
    order   = np.argsort(dts_all)
    dts_all = dts_all[order]
    Z_all   = Z_all[order]
    cat_all = cat_all[order]

    # rain mask
    rain_mask = (
        ((cat_all & 2)  != 0) &
        ((cat_all & 32) == 0)
    )
    Z_rain = np.where(rain_mask, Z_all, np.nan)

    # height selection
    h_idx = height_km <= HEIGHT_MAX_KM

    # ── plot ─────────────────────────────────────────────────────────────────
    print("Plotting …")
    cmap_ze   = plt.get_cmap("turbo").copy();  cmap_ze.set_bad(color="white")
    cmap_mask = plt.get_cmap("Blues").copy(); cmap_mask.set_bad(color="white")

    fig, axes = plt.subplots(3, 1, figsize=(20, 13), sharex=True,
                             gridspec_kw={"hspace": 0.06})

    def draw_panel(ax, data, title, cmap, vmin, vmax, cbar_label):
        im = ax.pcolormesh(
            dts_all,
            height_km[h_idx],
            data[:, h_idx].T,
            cmap=cmap, vmin=vmin, vmax=vmax,
            shading="nearest",
        )
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.015)
        cb.set_label(cbar_label, fontsize=9)
        ax.set_ylabel("Height AMSL [km]", fontsize=9)
        ax.set_ylim(0, HEIGHT_MAX_KM)
        ax.yaxis.set_tick_params(labelsize=8)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.axhline(2.5, color="cyan", lw=0.7, ls="--", label="~0 °C isotherm")

    draw_panel(axes[0], Z_all,
               "(a) Raw Ze – all radar echoes",
               cmap_ze, ZE_MIN, ZE_MAX, "Ze [dBZ]")

    draw_panel(axes[1], Z_rain,
               "(b) Masked Ze – insects/clutter removed",
               cmap_ze, ZE_MIN, ZE_MAX, "Ze [dBZ]")

    draw_panel(axes[2], rain_mask.astype(float),
               "(c) Rain mask  (1 = passes filter)",
               cmap_mask, 0, 1, "mask")

    # x-axis: show day ticks
    axes[2].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%-d %b", tz=timezone.utc))
    axes[2].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    axes[2].set_xlabel(f"Date UTC  –  {month_label}", fontsize=10)
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    fig.suptitle(
        f"Jülich CloudNet categorize  –  {month_label}\n"
        f"METEK MIRA-35  |  35.5 GHz  |  {days_loaded} days loaded",
        fontsize=11, y=0.998,
    )

    outfile = OUT_DIR / f"ze_heatmap_monthly_{year_month_str}.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {outfile}")


if __name__ == "__main__":
    ym = sys.argv[1] if len(sys.argv) > 1 else "202506"
    if len(ym) != 6 or not ym.isdigit():
        print(f"Usage: {sys.argv[0]} YYYYMM"); sys.exit(1)
    make_monthly_heatmap(ym)
