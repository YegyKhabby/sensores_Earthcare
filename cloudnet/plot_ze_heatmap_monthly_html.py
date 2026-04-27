#!/usr/bin/env python3
"""
plot_ze_heatmap_monthly_html.py
================================
Interactive HTML heatmap of Cloudnet Ze for a full month.
One row per day (Raw Ze + Masked Ze side by side), scroll vertically through month.
Each day spans the full 0–24 h x-axis so nothing is compressed.

Usage
-----
  python3 plot_ze_heatmap_monthly_html.py           # defaults to 202506
  python3 plot_ze_heatmap_monthly_html.py 202507

Output
------
  output/cloudnet/ze_heatmap_monthly_<YYYYMM>.html
"""

import sys
import calendar
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import netCDF4 as nc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CLOUDNET_BASE = Path("/data/obs/site/jue/cloudnet/clu_processing/categorize")
OUT_DIR       = Path(__file__).resolve().parents[1] / "output" / "cloudnet"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEIGHT_MAX_KM = 10.0
ZE_MIN, ZE_MAX = -60, 30
PX_PER_DAY     = 300   # height in pixels for each day row-pair


def load_day(nc_path):
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
        Z   = np.ma.filled(ds.variables["Z"][:].astype(float), np.nan)
        cat = np.array(ds.variables["category_bits"][:])
    return dts, height_km, Z, cat


def make_html(year_month_str):
    yyyy = int(year_month_str[:4])
    mm   = int(year_month_str[4:6])
    n_days     = calendar.monthrange(yyyy, mm)[1]
    month_label = f"{yyyy}-{mm:02d}"

    print(f"Loading {month_label} …")

    # load each day separately — one subplot row per day
    days = []   # list of (date_str, dts, hkm, Z_raw, Z_masked)
    height_km = None

    for d in range(1, n_days + 1):
        date_str = f"{yyyy}{mm:02d}{d:02d}"
        nc_path  = CLOUDNET_BASE / str(yyyy) / f"{date_str}_juelich_categorize.nc"
        if not nc_path.exists():
            print(f"  {date_str}  SKIP")
            continue
        try:
            dts, hkm, Z, cat = load_day(nc_path)
        except Exception as e:
            print(f"  {date_str}  ERROR: {e}")
            continue
        if height_km is None:
            height_km = hkm

        rain_mask = ((cat & 2) != 0) & ((cat & 32) == 0)
        Z_masked  = np.where(rain_mask, Z, np.nan)

        # express time as fractional hours (0–24) for a common x-axis
        day_start = datetime(yyyy, mm, d, 0, 0, 0, tzinfo=timezone.utc)
        hours = np.array([(t - day_start).total_seconds() / 3600.0 for t in dts])

        days.append((date_str, hours, Z, Z_masked))
        print(f"  {date_str}  ✓  {len(dts)} steps")

    if not days:
        print("No data — aborting.")
        return

    h_idx    = height_km <= HEIGHT_MAX_KM
    hkm_plot = height_km[h_idx][::2]   # every 2nd gate to keep size down
    n_rows   = len(days)

    print(f"Building interactive plot ({n_rows} day rows) …")

    # two columns: raw | masked
    subplot_titles = []
    for date_str, *_ in days:
        d_label = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        subplot_titles += [f"{d_label}  Raw Ze", f"{d_label}  Masked Ze"]

    fig = make_subplots(
        rows=n_rows, cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.01,
    )

    colorscale = "Turbo"
    hover_tmpl = "Hour: %{x:.2f}<br>Height: %{y:.2f} km<br>Ze: %{z:.1f} dBZ<extra></extra>"

    show_cb = True   # only show colorbar on first trace
    for row_i, (date_str, hours, Z_raw, Z_masked) in enumerate(days, start=1):
        Zr = Z_raw[:, h_idx][:, ::2]
        Zm = Z_masked[:, h_idx][:, ::2]

        cb_kw = dict(colorbar=dict(
            title="Ze [dBZ]", thickness=12, len=0.3,
            x=1.01, y=1.0, yanchor="top",
        )) if show_cb else dict(showscale=False)
        show_cb = False

        for col_i, (Z_panel, name) in enumerate([(Zr, "Raw"), (Zm, "Masked")], start=1):
            fig.add_trace(
                go.Heatmap(
                    z=Z_panel.T,
                    x=hours,
                    y=hkm_plot,
                    name=name,
                    colorscale=colorscale,
                    zmin=ZE_MIN, zmax=ZE_MAX,
                    zsmooth=False,
                    hoverongaps=False,
                    hovertemplate=hover_tmpl,
                    showscale=(col_i == 2 and row_i == 1),
                    **({} if col_i != 2 or row_i != 1 else
                       {"colorbar": dict(title="Ze [dBZ]", thickness=12,
                                         x=1.02, len=0.15, y=1.0, yanchor="top")}),
                ),
                row=row_i, col=col_i,
            )

    # shared x-axis ticks (hours 0–24)
    hour_ticks = list(range(0, 25, 3))
    fig.update_xaxes(
        range=[0, 24],
        tickvals=hour_ticks,
        ticktext=[f"{h:02d}:00" for h in hour_ticks],
        tickfont=dict(size=9),
    )
    fig.update_yaxes(range=[0, HEIGHT_MAX_KM], tickfont=dict(size=8))

    # only show x-axis label on bottom row
    fig.update_xaxes(title_text="Time UTC [h]", row=n_rows, col=1)
    fig.update_xaxes(title_text="Time UTC [h]", row=n_rows, col=2)

    total_height = max(600, PX_PER_DAY * n_rows)
    fig.update_layout(
        title=dict(
            text=f"Jülich CloudNet – {month_label} | METEK MIRA-35 35.5 GHz | "
                 f"{n_rows} days  "
                 "<sup>(scroll vertically · zoom/pan each panel · hover for values)</sup>",
            font=dict(size=13),
        ),
        height=total_height,
        width=1400,
        plot_bgcolor="white",
        margin=dict(l=60, r=80, t=60, b=40),
    )

    # suppress duplicate annotation font size warnings
    for ann in fig.layout.annotations:
        ann.font = dict(size=9)

    outfile = OUT_DIR / f"ze_heatmap_monthly_{year_month_str}.html"
    fig.write_html(str(outfile), include_plotlyjs="cdn")
    print(f"Saved → {outfile}")
    print(f"Open in browser: file://{outfile.resolve()}")


if __name__ == "__main__":
    ym = sys.argv[1] if len(sys.argv) > 1 else "202506"
    if len(ym) != 6 or not ym.isdigit():
        print(f"Usage: {sys.argv[0]} YYYYMM"); sys.exit(1)
    make_html(ym)
