#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare exact wawa-code agreement between:
  - BONN Parsivel log    (/parsivel_304640/.../parsivel_jue_YYYYMMDD.log)
  - JOYCE Parsivel NetCDF(/parsivel/.../sups_joy_dm00_l1_any_v00_YYYYMMDD.nc)
for all days in year 2025.

Rules requested:
  - exact code match (not grouped classes)
  - if either daily file is missing: skip that day entirely
  - produce monthly plots (daily agreement bars for each month)
  - produce overall plots

Outputs:
  output/wawa_compare/
    daily_agreement_2025.csv
    monthly_summary_2025.csv
    overall_summary_2025.csv
    plots/
      month_01_daily_agreement.png ... month_12_daily_agreement.png
      monthly_agreement_2025.png
      overall_agreement_2025.png
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import calendar

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4 as nc

YEAR = 2025
MERGE_TOLERANCE_SECONDS = 40  # nearest timestamp matching tolerance

OUT_DIR = Path('/work/yegy_project/Feb2026_new/output/wawa_compare')
PLOT_DIR = OUT_DIR / 'plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def path_bonn(d: date):
    """BONN LOG path"""
    y, m, dd = d.year, d.month, d.day
    ymd = f"{y:04d}{m:02d}{dd:02d}"
    return f"/data/obs/site/jue/parsivel_304640/l1/{y:04d}/{m:02d}/{dd:02d}/parsivel_jue_{ymd}.log"


def path_nc(d: date):
    """JOYCE NetCDF path"""
    y, m, dd = d.year, d.month, d.day
    ymd = f"{y:04d}{m:02d}{dd:02d}"
    return f"/data/obs/site/jue/parsivel/l1/{y:04d}/{m:02d}/{dd:02d}/sups_joy_dm00_l1_any_v00_{ymd}.nc"


def load_bonn_wawa(log_path: str) -> pd.DataFrame:
    """Read BONN log and return DataFrame with UTC time + wawa code."""
    times = []
    wawa_vals = []

    with open(log_path, 'r', encoding='latin-1') as fh:
        _ = fh.readline()  # header
        for line in fh:
            parts = line.rstrip('\n').split(';')
            if len(parts) < 4:
                continue

            t = pd.to_datetime(
                parts[0].strip(),
                format='%Y-%m-%d %H:%M:%S',
                utc=True,
                errors='coerce'
            )
            w = pd.to_numeric(parts[3].strip(), errors='coerce')
            if pd.isna(t) or pd.isna(w):
                continue

            times.append(t)
            wawa_vals.append(w)

    df = pd.DataFrame({'time': times, 'wawa_bonn': wawa_vals})
    df = df.sort_values('time').reset_index(drop=True)
    return df


def load_joyce_wawa(nc_path: str) -> pd.DataFrame:
    """Read JOYCE NetCDF and return DataFrame with UTC time + wawa code."""
    ds = nc.Dataset(nc_path)

    # time (s since 1970-01-01), often with fill endpoints < 0.
    # Some files may contain unrealistic large sentinels; filter by sane range.
    t_raw = np.array(ds.variables['time'][:], dtype=float)
    t_raw[(t_raw < 0) | (t_raw < 946684800) | (t_raw > 2082758400)] = np.nan
    if len(t_raw) >= 2:
        if np.isnan(t_raw[0]):
            t_raw[0] = t_raw[1] - 60.0
        if np.isnan(t_raw[-1]):
            t_raw[-1] = t_raw[-2] + 60.0

    wawa = np.array(ds.variables['wawa'][:], dtype=float)
    ds.close()

    time_pd = pd.to_datetime(t_raw, unit='s', utc=True, errors='coerce')

    df = pd.DataFrame({'time': time_pd, 'wawa_joyce': wawa})
    df['wawa_joyce'] = pd.to_numeric(df['wawa_joyce'], errors='coerce')
    df = df.dropna(subset=['time', 'wawa_joyce']).sort_values('time').reset_index(drop=True)
    return df


def compare_day(d: date) -> dict | None:
    """Compare one day, return metrics dict; None if file missing or no matches."""
    p_log = path_bonn(d)
    p_nc = path_nc(d)
    if not Path(p_log).exists() or not Path(p_nc).exists():
        return None

    bonn = load_bonn_wawa(p_log)
    joy = load_joyce_wawa(p_nc)
    if bonn.empty or joy.empty:
        return None

    # nearest-time match: BONN rows matched to nearest JOYCE row
    merged = pd.merge_asof(
        bonn.sort_values('time'),
        joy.sort_values('time'),
        on='time',
        direction='nearest',
        tolerance=pd.Timedelta(seconds=MERGE_TOLERANCE_SECONDS),
    )
    merged = merged.dropna(subset=['wawa_joyce'])
    if merged.empty:
        return None

    # exact code agreement (integer code equality)
    b = merged['wawa_bonn'].astype(int).to_numpy()
    j = merged['wawa_joyce'].astype(int).to_numpy()
    agree = (b == j)

    n_total = int(agree.size)
    n_agree = int(agree.sum())
    pct = 100.0 * n_agree / n_total if n_total > 0 else np.nan

    return {
        'date': d.isoformat(),
        'month': d.month,
        'day': d.day,
        'n_total': n_total,
        'n_agree': n_agree,
        'agreement_pct': pct,
    }


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def plot_month_daily(month_df: pd.DataFrame, month: int):
    """Bar plot of day-by-day agreement for one month."""
    if month_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    x = month_df['day'].to_numpy()
    y = month_df['agreement_pct'].to_numpy()

    ax.bar(x, y, color='steelblue', edgecolor='black', linewidth=0.4)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Day of month')
    ax.set_ylabel('Agreement [%]')
    ax.set_title(f'{calendar.month_name[month]} {YEAR}: exact wawa agreement (BONN vs JOYCE)')
    ax.grid(axis='y', alpha=0.3)

    # annotate month weighted mean
    n_agree = month_df['n_agree'].sum()
    n_total = month_df['n_total'].sum()
    m_pct = (100.0 * n_agree / n_total) if n_total > 0 else np.nan
    ax.text(0.99, 0.95,
            f'Weighted month mean = {m_pct:.2f}%\nN={n_total}',
            transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f'month_{month:02d}_daily_agreement.png', dpi=150)
    plt.close(fig)


def plot_monthly_summary(monthly: pd.DataFrame):
    """Bar plot of monthly weighted agreement (12 bars)."""
    fig, ax = plt.subplots(figsize=(11, 4.5))

    x = monthly['month']
    y = monthly['agreement_pct']
    ax.bar(x, y, color='darkorange', edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])
    ax.set_ylim(0, 100)
    ax.set_xlabel('Month')
    ax.set_ylabel('Agreement [%]')
    ax.set_title(f'{YEAR}: monthly exact wawa agreement (BONN vs JOYCE)')
    ax.grid(axis='y', alpha=0.3)

    for _, row in monthly.iterrows():
        ax.text(row['month'], row['agreement_pct'] + 1,
                f"{row['agreement_pct']:.1f}%",
                ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f'monthly_agreement_{YEAR}.png', dpi=150)
    plt.close(fig)


def plot_overall(overall_pct: float, n_total: int):
    """Single-bar overall agreement plot."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.bar([0], [overall_pct], color='seagreen', edgecolor='black')
    ax.set_xticks([0])
    ax.set_xticklabels([str(YEAR)])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Agreement [%]')
    ax.set_title('Overall exact wawa agreement\nBONN vs JOYCE')
    ax.grid(axis='y', alpha=0.3)
    ax.text(0, overall_pct + 1,
            f'{overall_pct:.2f}%\nN={n_total}',
            ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f'overall_agreement_{YEAR}.png', dpi=150)
    plt.close(fig)


def main():
    start = date(YEAR, 1, 1)
    end = date(YEAR, 12, 31)

    rows = []
    missing_days = 0
    for d in daterange(start, end):
        r = compare_day(d)
        if r is None:
            missing_days += 1
            continue
        rows.append(r)

    if not rows:
        print('No comparable days found (all missing or no valid matches).')
        return

    daily = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    daily_csv = OUT_DIR / f'daily_agreement_{YEAR}.csv'
    daily.to_csv(daily_csv, index=False, float_format='%.3f')

    # monthly weighted summary
    monthly = (
        daily.groupby('month', as_index=False)[['n_total', 'n_agree']]
        .sum()
        .assign(agreement_pct=lambda x: 100.0 * x['n_agree'] / x['n_total'])
    )

    # include only months with data in CSV/plot
    monthly_csv = OUT_DIR / f'monthly_summary_{YEAR}.csv'
    monthly.to_csv(monthly_csv, index=False, float_format='%.3f')

    # overall
    n_total = int(daily['n_total'].sum())
    n_agree = int(daily['n_agree'].sum())
    overall_pct = 100.0 * n_agree / n_total if n_total > 0 else np.nan

    overall_df = pd.DataFrame([{
        'year': YEAR,
        'n_days_compared': int(len(daily)),
        'n_days_skipped': int(missing_days),
        'n_total': n_total,
        'n_agree': n_agree,
        'agreement_pct': overall_pct,
    }])
    overall_csv = OUT_DIR / f'overall_summary_{YEAR}.csv'
    overall_df.to_csv(overall_csv, index=False, float_format='%.3f')

    # plots: one per month + overall plots
    for m in sorted(daily['month'].unique()):
        plot_month_daily(daily[daily['month'] == m], int(m))

    plot_monthly_summary(monthly)
    plot_overall(overall_pct, n_total)

    print('Done.')
    print(f'Compared days: {len(daily)}')
    print(f'Skipped days : {missing_days} (missing file or no valid matches)')
    print(f'Overall exact wawa agreement: {overall_pct:.2f}%  (N={n_total})')
    print(f'\nOutputs:')
    print(f'  {daily_csv}')
    print(f'  {monthly_csv}')
    print(f'  {overall_csv}')
    print(f'  {PLOT_DIR}')


if __name__ == '__main__':
    main()
