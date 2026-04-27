# -*- coding: utf-8 -*-
"""
compare_ze_fwd_cloudnet_20250606.py
====================================
Compare forward-simulated Ze (Parsivel M-matrix, 35.6 GHz T-matrix, 293.15 K)
against JOYRAD-35 Ze from the Cloudnet categorize file for 2025-06-06.

Three figures are produced:
  1. Ze vs Time       – one panel per height bin, FWD and Cloudnet overlaid
  2. Ze vs Rain rate  – scatter, one panel per height bin
  3. Ze_fwd vs Ze_cloudnet – scatter with 1:1 line, bias/RMSE, one panel per bin

All outputs → /work/yegy_project/Feb2026_new/output/comparison/
"""

import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
CLOUDNET_NC = Path('/data/obs/site/jue/cloudnet/clu_processing/categorize/2025/'
                   '20250606_juelich_categorize.nc')
FWD_CSV     = Path('/work/yegy_project/Feb2026_new/output/comparison/'
                   'fwd_sim_jue_parsivel_20250606.csv')
OUT_DIR     = Path('/work/yegy_project/Feb2026_new/output/comparison')
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BINS   = 10     # lowest N cloudnet height gates
RR_MIN   = 0.1   # mm h⁻¹ rain-rate threshold for scatter plots
COLOURS  = plt.cm.viridis_r(np.linspace(0.05, 0.9, N_BINS))


# =============================================================================
# 1 – Load forward simulation
# =============================================================================
print("Loading forward simulation CSV …")
fwd = pd.read_csv(FWD_CSV, index_col=0, parse_dates=True)
if fwd.index.tz is None:
    fwd.index = fwd.index.tz_localize('UTC')
print(f"  {len(fwd)} parsivel steps  |  {(fwd['rr'] >= RR_MIN).sum()} rainy")

# =============================================================================
# 2 – Load Cloudnet Ze at lowest N_BINS height bins
# =============================================================================
print("Loading Cloudnet categorize NetCDF …")
with nc.Dataset(CLOUDNET_NC, 'r') as ds:
    t_hrs = np.array(ds.variables['time'][:])
    t_cn  = (pd.to_datetime('2025-06-06')
             + pd.to_timedelta(t_hrs, unit='h')).tz_localize('UTC')
    Z_raw = np.ma.filled(ds.variables['Z'][:, :N_BINS], np.nan)  # (t, N_BINS)
    h_m   = np.array(ds.variables['height'][:N_BINS])            # m AMSL

h_labels = [f"{h:.0f} m" for h in h_m]   # AMSL heights used for column keys
Z_cn_df  = pd.DataFrame(Z_raw, index=t_cn, columns=h_labels)
print(f"  {len(t_cn)} cloudnet steps @ ~30 s  |  heights {h_m[0]:.0f}–{h_m[-1]:.0f} m AMSL")

# =============================================================================
# 3 – Align: merge_asof (nearest cloudnet step within ±60 s of each parsivel step)
# =============================================================================
print("Aligning time axes (nearest, ±60 s) …")
merged = pd.merge_asof(
    fwd.reset_index().rename(columns={'index': 'time'}).sort_values('time'),
    Z_cn_df.reset_index().rename(columns={'index': 'time'}).sort_values('time'),
    on='time', tolerance=pd.Timedelta('60s'), direction='nearest'
).set_index('time')
merged.index = pd.DatetimeIndex(merged.index, tz='UTC')

rain_mask = merged['rr'] >= RR_MIN
print(f"  {rain_mask.sum()} matched rainy steps in merged table")

# =============================================================================
# PLOT 1 – Ze vs Time
# =============================================================================
print("\nPlot 1 – Ze vs time …")
fig, axes = plt.subplots(N_BINS, 1, figsize=(14, 2.4 * N_BINS), sharex=True)
fig.suptitle(
    'Ze vs Time – Jülich 2025-06-06\n'
    'Cloudnet JOYRAD-35 (solid) vs Parsivel FWD Ze_tmm 35.6 GHz / 293.15 K (dashed red)',
    fontsize=11, y=1.002)

for i, (ax, hlabel) in enumerate(zip(axes, h_labels)):
    # shade rainy periods
    ax.fill_between(merged.index, 0, 1,
                    where=rain_mask,
                    transform=ax.get_xaxis_transform(),
                    color='skyblue', alpha=0.35, label='rr ≥ 0.1 mm h⁻¹')
    ax.plot(merged.index, merged[hlabel],
            color=COLOURS[i], lw=1.3, label=f'Cloudnet {hlabel}')
    ax.plot(merged.index, merged['Ze_tmm'],
            color='crimson', lw=0.9, ls='--', alpha=0.85, label='FWD Ze_tmm')
    ax.set_ylabel('Ze [dBZ]', fontsize=8)
    ax.set_ylim(-20, 45)
    ax.set_title(f'Height bin {i}  –  {hlabel} AMSL',
                fontsize=9, pad=2)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc='upper right', ncol=3)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
axes[-1].set_xlabel('Time UTC [2025-06-06]')
plt.tight_layout()
out1 = OUT_DIR / 'compare_ze_vs_time_20250606.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out1.name}")

# =============================================================================
# PLOT 2 – Ze vs Rain Rate  (rainy steps only)
# =============================================================================
print("Plot 2 – Ze vs rain rate …")
ncols = 2
nrows = int(np.ceil(N_BINS / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.8 * nrows))
axes_flat = axes.flatten()
fig.suptitle(
    'Ze vs Rain Rate – Jülich 2025-06-06  (rr ≥ 0.1 mm h⁻¹)\n'
    'Cloudnet JOYRAD-35 (●, coloured) vs Parsivel FWD Ze_tmm (✕ red)',
    fontsize=10)

rr_r = merged.loc[rain_mask, 'rr'].values
for i, (ax, hlabel) in enumerate(zip(axes_flat[:N_BINS], h_labels)):
    z_cn  = merged.loc[rain_mask, hlabel].values
    z_fwd = merged.loc[rain_mask, 'Ze_tmm'].values
    valid = np.isfinite(z_cn) & np.isfinite(rr_r)
    ax.scatter(rr_r[valid], z_cn[valid],
               c=[COLOURS[i]], s=22, alpha=0.85, zorder=3, label='Cloudnet')
    ax.scatter(rr_r[valid], z_fwd[valid],
               marker='x', s=22, color='crimson', alpha=0.7, label='FWD Ze_tmm')
    ax.set_xlabel('Rain rate [mm h⁻¹]', fontsize=8)
    ax.set_ylabel('Ze [dBZ]', fontsize=8)
    ax.set_title(f'Bin {i}  –  {hlabel} AMSL', fontsize=9)
    ax.set_ylim(-5, 45)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

for ax in axes_flat[N_BINS:]:
    ax.set_visible(False)
plt.tight_layout()
out2 = OUT_DIR / 'compare_ze_vs_rr_20250606.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out2.name}")

# =============================================================================
# PLOT 3 – Ze_fwd vs Ze_cloudnet  scatter + 1:1
# =============================================================================
print("Plot 3 – Ze_fwd vs Ze_cloudnet …")
fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.8 * nrows))
axes_flat = axes.flatten()
fig.suptitle(
    'Ze FWD vs Ze Cloudnet – Jülich 2025-06-06\n'
    'T-matrix 35.6 GHz / 293.15 K  |  colour = rain rate [mm h⁻¹]',
    fontsize=10)

lim = [-5, 45]
for i, (ax, hlabel) in enumerate(zip(axes_flat[:N_BINS], h_labels)):
    z_cn  = merged.loc[rain_mask, hlabel].values
    z_fwd = merged.loc[rain_mask, 'Ze_tmm'].values
    rr_r2 = merged.loc[rain_mask, 'rr'].values
    valid = np.isfinite(z_cn) & np.isfinite(z_fwd)
    n_v   = valid.sum()

    if n_v > 0:
        sc = ax.scatter(z_fwd[valid], z_cn[valid],
                        c=rr_r2[valid], cmap='plasma',
                        vmin=RR_MIN, vmax=np.nanmax(rr_r2) + 0.1,
                        s=28, alpha=0.85)
        plt.colorbar(sc, ax=ax, label='rr [mm h⁻¹]', fraction=0.046, pad=0.04)
        bias = float(np.mean(z_cn[valid] - z_fwd[valid]))
        rmse = float(np.sqrt(np.mean((z_cn[valid] - z_fwd[valid])**2)))
        ax.text(0.04, 0.96,
                f'N = {n_v}\nbias = {bias:+.1f} dB\nRMSE = {rmse:.1f} dB',
                transform=ax.transAxes, fontsize=7.5, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75))

    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.set_xlabel('Ze_tmm FWD [dBZ]', fontsize=8)
    ax.set_ylabel('Ze Cloudnet [dBZ]', fontsize=8)
    ax.set_title(f'Bin {i}  –  {hlabel} AMSL', fontsize=9)
    ax.grid(True, alpha=0.25)

for ax in axes_flat[N_BINS:]:
    ax.set_visible(False)
plt.tight_layout()
out3 = OUT_DIR / 'compare_ze_fwd_vs_cloudnet_20250606.png'
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out3.name}")

# =============================================================================
# Summary table
# =============================================================================
print()
print("── Bias & RMSE (Ze_cloudnet − Ze_fwd, rainy steps) ──────────────────")
print(f"  {'Height':>10}  {'N':>4}  {'Bias [dB]':>10}  {'RMSE [dB]':>10}")
print(f"  {'─'*44}")
for hlabel in h_labels:
    z_cn  = merged.loc[rain_mask, hlabel].values
    z_fwd = merged.loc[rain_mask, 'Ze_tmm'].values
    valid = np.isfinite(z_cn) & np.isfinite(z_fwd)
    if valid.sum() > 0:
        d = z_cn[valid] - z_fwd[valid]
        print(f"  {hlabel:>10}  {valid.sum():>4}  {d.mean():>+10.2f}  "
              f"{np.sqrt((d**2).mean()):>10.2f}")
    else:
        print(f"  {hlabel:>10}  {'0':>4}  {'—':>10}  {'—':>10}")
print("──────────────────────────────────────────────────────────────────────")
