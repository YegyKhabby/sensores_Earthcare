# -*- coding: utf-8 -*-
"""
compare_ze_fwd_cloudnet_masked_20250606.py
==========================================
Same comparison as compare_ze_fwd_cloudnet_20250606.py but with two extra masks:

  Parsivel / FWD mask
    • Only D-bins with centre 1.0 – 5.0 mm contribute to the forward simulation
      (bins 8–19 of the OTT Parsivel 2: D = 1.062 … 4.750 mm)
      → removes small drizzle/noise drops (<1 mm) and rare large-drop outliers
        or margin-faller artefacts (>5 mm)
    • Only time steps with rr ≥ 1.0 mm h⁻¹ are included in the scatter/bias
      analysis (proper rain, not just background drizzle)

  Cloudnet category_bits mask  (applied per pixel before averaging)
    • Bit 1 must be SET   → falling hydrometeors present (rain / drizzle)
    • Bit 5 must be CLEAR → no insects
    → Pixels that fail either condition are set to NaN

Outputs (new filenames, previous plots are NOT overwritten):
  compare_ze_vs_time_masked_20250606.png
  compare_ze_vs_rr_masked_20250606.png
  compare_ze_fwd_vs_cloudnet_masked_20250606.png
"""

import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ── raincoat ──────────────────────────────────────────────────────────────────
from raincoat.FWD_sim import FWD_sim
import raincoat.disdrometer.pars_class as pc

# ── paths ─────────────────────────────────────────────────────────────────────
PARSIVEL_NC = Path('/data/obs/site/jue/parsivel/l1/2025/06/06/'
                   'sups_joy_dm00_l1_any_v00_20250606.nc')
CLOUDNET_NC = Path('/data/obs/site/jue/cloudnet/clu_processing/categorize/2025/'
                   '20250606_juelich_categorize.nc')
SCAT_TABLE  = Path('/work/yegy_project/Feb2026_new/raincoat/samplefiles/'
                   'scattering/293.15_35.6GHz.csv')
OUT_DIR     = Path('/work/yegy_project/Feb2026_new/output/comparison')
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BINS  = 10    # lowest N cloudnet height gates
RR_MIN  = 1.0  # mm h⁻¹ – only proper rain events
D_MIN   = 1.0  # mm – lower diameter limit for FWD integration
D_MAX   = 5.0  # mm – upper diameter limit for FWD integration

A_SENSOR = 54e-4   # m²  OTT Parsivel2 effective sampling area
DT       = 60.0    # s   sample interval

COLOURS  = plt.cm.viridis_r(np.linspace(0.05, 0.9, N_BINS))

# =============================================================================
# 1 – Load parsivel NC, reconstruct N from M, apply D filter
# =============================================================================
print("STEP 1  Loading Parsivel NC and reconstructing N from M …")
with nc.Dataset(PARSIVEL_NC, 'r') as ds:
    t_raw    = np.array(ds.variables['time'][:], dtype=float)
    M        = np.array(ds.variables['M'][:])          # (v=32, D=32, t=1441)
    vclasses = np.array(ds.variables['vclasses'][:])   # m s⁻¹
    dclasses = np.array(ds.variables['dclasses'][:])   # mm
    dwidth   = np.array(ds.variables['dwidth'][:])     # mm
    Ze_inst  = np.array(ds.variables['Ze'][:])         # dBZ
    rr       = np.array(ds.variables['rr'][:])         # mm h⁻¹

# Fix fill-value timestamps
n_bad = int(np.sum(t_raw < 0))
if np.isnan(t_raw[0]) or t_raw[0] < 0:
    t_raw[0] = t_raw[1] - DT
if np.isnan(t_raw[-1]) or t_raw[-1] < 0:
    t_raw[-1] = t_raw[-2] + DT
# replace remaining negatives gracefully
t_raw[t_raw < 0] = np.nan
time_pd = pd.to_datetime(t_raw, unit='s', utc=True)
print(f"        {len(time_pd)} time steps  ({n_bad} fill endpoints fixed)")

# Reconstruct N from M:  N(D_i,t) = Σ_j M[j,i,t] / (v_j · A · Δt · ΔD_i)
v_safe  = np.where(vclasses > 0, vclasses, np.inf)          # avoid /0
M_vsum  = (M / v_safe[:, np.newaxis, np.newaxis]).sum(axis=0)  # (D, t)
N_full  = M_vsum / (A_SENSOR * DT * dwidth[:, np.newaxis])     # m⁻³ mm⁻¹

# Apply D filter: zero out bins outside [D_MIN, D_MAX]
d_mask = (dclasses >= D_MIN) & (dclasses <= D_MAX)
N_filtered = N_full.copy()
N_filtered[~d_mask, :] = 0.0
n_kept = d_mask.sum()
print(f"        D filter [{D_MIN}–{D_MAX} mm]: keeping {n_kept}/32 bins  "
      f"(D = {dclasses[d_mask][0]:.3f} – {dclasses[d_mask][-1]:.3f} mm)")

with np.errstate(divide='ignore', invalid='ignore'):
    log10_N = np.where(N_filtered > 0, np.log10(N_filtered), np.nan)

# =============================================================================
# 2 – Build parsivel bin edges and run FWD_sim
# =============================================================================
print("\nSTEP 2  Running FWD_sim on D-filtered N …")
pclass, bin_edges = pc.pars_class()
fwd = FWD_sim(str(SCAT_TABLE), time_pd, log10_N, bin_edges)
fwd['rr'] = pd.Series(rr, index=time_pd).values
print(f"        Done.  Rainy steps (rr ≥ {RR_MIN} mm/h): "
      f"{(fwd['rr'] >= RR_MIN).sum()}")

# =============================================================================
# 3 – Load Cloudnet Ze with category_bits mask
# =============================================================================
print("\nSTEP 3  Loading Cloudnet Ze with category_bits mask …")
with nc.Dataset(CLOUDNET_NC, 'r') as ds:
    t_hrs = np.array(ds.variables['time'][:])
    t_cn  = (pd.to_datetime('2025-06-06')
             + pd.to_timedelta(t_hrs, unit='h')).tz_localize('UTC')

    Z_raw  = np.ma.filled(ds.variables['Z'][:, :N_BINS], np.nan)   # (t, N_BINS)
    cat    = np.array(ds.variables['category_bits'][:, :N_BINS])   # (t, N_BINS)
    h_m    = np.array(ds.variables['height'][:N_BINS])             # m AMSL

# Mask: bit1 set (hydrometeors) AND bit5 clear (no insects)
hydro_ok  = ((cat >> 1) & 1).astype(bool)
insect_ok = ~((cat >> 5) & 1).astype(bool)
good      = hydro_ok & insect_ok
Z_masked  = np.where(good, Z_raw, np.nan)   # NaN where not good

h_labels = [f"{h:.0f} m" for h in h_m]   # AMSL heights used for column keys
Z_cn_df  = pd.DataFrame(Z_masked, index=t_cn, columns=h_labels)

for i, hl in enumerate(h_labels):
    n_good = int(np.sum(good[:, i]))
    print(f"        {hl}: {n_good}/{len(t_cn)} pixels kept after category mask")

# =============================================================================
# 4 – Align time axes  (nearest cloudnet step, ±60 s tolerance)
# =============================================================================
print("\nSTEP 4  Aligning time axes …")
merged = pd.merge_asof(
    fwd.reset_index().rename(columns={'index': 'time'}).sort_values('time'),
    Z_cn_df.reset_index().rename(columns={'index': 'time'}).sort_values('time'),
    on='time', tolerance=pd.Timedelta('60s'), direction='nearest'
).set_index('time')
merged.index = pd.DatetimeIndex(merged.index, tz='UTC')

rain_mask = merged['rr'] >= RR_MIN
print(f"        {rain_mask.sum()} parsivel steps with rr ≥ {RR_MIN} mm/h")

# =============================================================================
# PLOT 1 – Ze vs Time
# =============================================================================
print("\nPlot 1 – Ze vs time (masked) …")
fig, axes = plt.subplots(N_BINS, 1, figsize=(14, 2.4 * N_BINS), sharex=True)
fig.suptitle(
    'Ze vs Time – Jülich 2025-06-06  [MASKED]\n'
    'Cloudnet: bit1=hydrometeors, bit5=no insects  |  '
    'FWD: D = 1–5 mm, 293.15 K / 35.6 GHz  |  rain shading: rr ≥ 1 mm h⁻¹',
    fontsize=10, y=1.002)

for i, (ax, hlabel) in enumerate(zip(axes, h_labels)):
    ax.fill_between(merged.index, 0, 1, where=rain_mask,
                    transform=ax.get_xaxis_transform(),
                    color='skyblue', alpha=0.4, label=f'rr ≥ {RR_MIN} mm h⁻¹')
    ax.plot(merged.index, merged[hlabel],
            color=COLOURS[i], lw=1.3, label=f'Cloudnet {hlabel} (masked)')
    ax.plot(merged.index, merged['Ze_tmm'],
            color='crimson', lw=0.9, ls='--', alpha=0.85,
            label='FWD Ze_tmm (D 1–5 mm)')
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
out1 = OUT_DIR / 'compare_ze_vs_time_masked_20250606.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out1.name}")

# =============================================================================
# PLOT 2 – Ze vs Rain Rate  (rainy steps only)
# =============================================================================
print("Plot 2 – Ze vs rain rate (masked) …")
ncols = 2
nrows = int(np.ceil(N_BINS / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.8 * nrows))
axes_flat = axes.flatten()
fig.suptitle(
    f'Ze vs Rain Rate – Jülich 2025-06-06  [MASKED]  (rr ≥ {RR_MIN} mm h⁻¹)\n'
    'Cloudnet: hydrometeors & no insects  |  FWD: D = 1–5 mm',
    fontsize=10)

rr_r = merged.loc[rain_mask, 'rr'].values
for i, (ax, hlabel) in enumerate(zip(axes_flat[:N_BINS], h_labels)):
    z_cn  = merged.loc[rain_mask, hlabel].values
    z_fwd = merged.loc[rain_mask, 'Ze_tmm'].values
    valid = np.isfinite(z_cn) & np.isfinite(rr_r)
    ax.scatter(rr_r[valid], z_cn[valid],
               c=[COLOURS[i]], s=25, alpha=0.85, zorder=3,
               label='Cloudnet (masked)')
    ax.scatter(rr_r[valid], z_fwd[valid],
               marker='x', s=25, color='crimson', alpha=0.7,
               label='FWD Ze_tmm (D 1–5 mm)')
    ax.set_xlabel('Rain rate [mm h⁻¹]', fontsize=8)
    ax.set_ylabel('Ze [dBZ]', fontsize=8)
    ax.set_title(f'Bin {i}  –  {hlabel} AMSL', fontsize=9)
    ax.set_ylim(-5, 45)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

for ax in axes_flat[N_BINS:]:
    ax.set_visible(False)
plt.tight_layout()
out2 = OUT_DIR / 'compare_ze_vs_rr_masked_20250606.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out2.name}")

# =============================================================================
# PLOT 3 – Ze_fwd vs Ze_cloudnet  scatter + 1:1
# =============================================================================
print("Plot 3 – Ze_fwd vs Ze_cloudnet (masked) …")
fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.8 * nrows))
axes_flat = axes.flatten()
fig.suptitle(
    'Ze FWD vs Ze Cloudnet – Jülich 2025-06-06  [MASKED]\n'
    'T-matrix 35.6 GHz / 293.15 K, D = 1–5 mm  |  colour = rain rate [mm h⁻¹]',
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
                        s=30, alpha=0.85)
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
    ax.set_xlabel('Ze_tmm FWD [dBZ]  (D 1–5 mm)', fontsize=8)
    ax.set_ylabel('Ze Cloudnet [dBZ]  (masked)', fontsize=8)
    ax.set_title(f'Bin {i}  –  {hlabel} AMSL', fontsize=9)
    ax.grid(True, alpha=0.25)

for ax in axes_flat[N_BINS:]:
    ax.set_visible(False)
plt.tight_layout()
out3 = OUT_DIR / 'compare_ze_fwd_vs_cloudnet_masked_20250606.png'
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out3.name}")

# =============================================================================
# Summary table
# =============================================================================
print()
print(f"── Bias & RMSE  [Ze_cloudnet(masked) − Ze_fwd(D 1–5mm)]  rr ≥ {RR_MIN} mm/h ──")
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
