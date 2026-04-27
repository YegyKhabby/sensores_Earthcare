# -*- coding: utf-8 -*-
"""
compare_ze_fwd_cloudnet_masked.py
==================================
Same as compare_ze_fwd_cloudnet.py but with three additional masks:

  Parsivel / FWD mask
    • Only D-bins with centre D_MIN – D_MAX mm contribute to the FWD simulation
      → removes small drizzle/noise drops and margin-faller artefacts
    • Rain rate threshold raised to RR_MIN = 1.0 mm h⁻¹
      → only proper rain, no drizzle

  Cloudnet category_bits mask (applied per pixel before time-averaging)
    • Bit 1 must be SET   → falling hydrometeors present
    • Bit 5 must be CLEAR → no insects

Time alignment, plot structure, and everything else is identical to
compare_ze_fwd_cloudnet.py.

Usage
-----
  python3 compare_ze_fwd_cloudnet_masked.py          # runs TEST_DATES
  python3 compare_ze_fwd_cloudnet_masked.py 20250607 # single date

Outputs per date (in output/comparison/cloudnet_vs_fwd_masked/<date>/):
  Ze_vs_time_<date>.png
  Ze_vs_rr_<date>.png
  Ze_fwd_vs_cloudnet_<date>.png
  alignment_diagnostics_<date>.png
"""

import sys
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ── raincoat ───────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / 'raincoat'))
from raincoat.FWD_sim import FWD_sim
import raincoat.disdrometer.pars_class as pc

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
PARSIVEL_BASE  = Path('/data/obs/site/jue/parsivel/l1')
CLOUDNET_BASE  = Path('/data/obs/site/jue/cloudnet/clu_processing/categorize')
SCAT_TABLE     = Path('/work/yegy_project/Feb2026_new/raincoat/samplefiles/'
                      'scattering/293.15_35.6GHz.csv')
OUT_BASE       = Path('/work/yegy_project/Feb2026_new/output/comparison/cloudnet_vs_fwd_masked')

N_BINS    = 10     # lowest N Cloudnet height gates to compare
RR_MIN    = 1.0   # mm/h  — proper rain only (was 0.1 in unmasked version)
TOL_S     = 30.0  # seconds  ±window for Cloudnet averaging
A_SENSOR  = 54e-4 # m²  OTT Parsivel2 sampling area
DT        = 60.0  # s   Parsivel integration interval

D_MIN     = 1.0   # mm  lower diameter cut — removes noise/drizzle drops
D_MAX     = 5.0   # mm  upper diameter cut — removes margin-faller artefacts

TEST_DATES = ['20250605', '20250606', '20250607']

COLOURS = plt.cm.viridis_r(np.linspace(0.05, 0.9, N_BINS))


# ══════════════════════════════════════════════════════════════════════════════
# STEP A – FWD simulation with D filter
# ══════════════════════════════════════════════════════════════════════════════
def run_fwd_sim(date_str):
    """
    Run T-matrix FWD simulation for one day with D-range filter applied.
    Only drops with D_MIN <= D <= D_MAX contribute to Ze_tmm.
    Returns a DataFrame indexed by UTC timestamps with columns:
      Ze_tmm [dBZ], Ze_ray [dBZ], A [dB/km], rr [mm/h]
    """
    yyyy, mm, dd = date_str[:4], date_str[4:6], date_str[6:]
    pars_nc = PARSIVEL_BASE / yyyy / mm / dd / f'sups_joy_dm00_l1_any_v00_{date_str}.nc'
    if not pars_nc.exists():
        raise FileNotFoundError(f'Parsivel file not found: {pars_nc}')

    with nc.Dataset(pars_nc) as ds:
        t_raw    = np.array(ds.variables['time'][:], dtype=float)
        M        = np.array(ds.variables['M'][:])
        vclasses = np.array(ds.variables['vclasses'][:])
        dwidth   = np.array(ds.variables['dwidth'][:])
        dclasses = np.array(ds.variables['dclasses'][:])
        rr       = np.array(ds.variables['rr'][:])

    # fix fill-value endpoints
    t_raw[t_raw < 0] = np.nan
    if np.isnan(t_raw[0]):
        t_raw[0] = t_raw[1] - DT
    if np.isnan(t_raw[-1]):
        t_raw[-1] = t_raw[-2] + DT
    time_pd = pd.to_datetime(t_raw, unit='s', utc=True)

    # reconstruct N(D,t) from M
    v_safe   = np.where(vclasses > 0, vclasses, np.inf)
    M_vsum   = (M / v_safe[:, np.newaxis, np.newaxis]).sum(axis=0)  # (D, t)
    N_from_M = M_vsum / (A_SENSOR * DT * dwidth[:, np.newaxis])     # m⁻³ mm⁻¹

    # D filter: zero out bins outside [D_MIN, D_MAX]
    d_mask = (dclasses >= D_MIN) & (dclasses <= D_MAX)
    N_filtered = N_from_M.copy()
    N_filtered[~d_mask, :] = 0.0
    n_kept = int(d_mask.sum())
    print(f'        D filter [{D_MIN}–{D_MAX} mm]: {n_kept}/32 bins kept '
          f'(D = {dclasses[d_mask][0]:.3f} – {dclasses[d_mask][-1]:.3f} mm)')

    with np.errstate(divide='ignore', invalid='ignore'):
        log10_N = np.where(N_filtered > 0, np.log10(N_filtered), np.nan)

    _, bin_edges = pc.pars_class()
    fwd = FWD_sim(str(SCAT_TABLE), time_pd, log10_N, bin_edges)
    fwd['rr'] = pd.Series(rr, index=time_pd).values
    return fwd


# ══════════════════════════════════════════════════════════════════════════════
# STEP B – Load Cloudnet Ze with category_bits mask
# ══════════════════════════════════════════════════════════════════════════════
def load_cloudnet(date_str, n_bins=N_BINS):
    """
    Load Ze from Cloudnet categorize NetCDF with category_bits mask applied.
    Per-pixel mask: bit 1 SET (hydrometeors) AND bit 5 CLEAR (no insects).
    Pixels failing either condition are set to NaN before time-averaging.
    Returns (Z_cn_df, h_labels, h_m).
    """
    yyyy = date_str[:4]
    cn_nc = CLOUDNET_BASE / yyyy / f'{date_str}_juelich_categorize.nc'
    if not cn_nc.exists():
        raise FileNotFoundError(f'Cloudnet file not found: {cn_nc}')

    with nc.Dataset(cn_nc) as ds:
        t_hrs = np.array(ds.variables['time'][:])
        t_cn  = (pd.to_datetime(f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}')
                 + pd.to_timedelta(t_hrs, unit='h')).tz_localize('UTC')
        Z_raw = np.ma.filled(ds.variables['Z'][:, :n_bins], np.nan)
        cat   = np.array(ds.variables['category_bits'][:, :n_bins])
        h_m   = np.array(ds.variables['height'][:n_bins])

    # category_bits mask
    hydro_ok  = ((cat >> 1) & 1).astype(bool)   # bit 1 set = hydrometeors
    insect_ok = ~((cat >> 5) & 1).astype(bool)  # bit 5 clear = no insects
    good      = hydro_ok & insect_ok
    Z_masked  = np.where(good, Z_raw, np.nan)

    h_labels = [f'{h:.0f} m' for h in h_m]
    Z_cn_df  = pd.DataFrame(Z_masked, index=t_cn, columns=h_labels)

    for i, hl in enumerate(h_labels):
        n_good = int(good[:, i].sum())
        print(f'        {hl}: {n_good}/{len(t_cn)} pixels kept after category mask')

    return Z_cn_df, h_labels, h_m


# ══════════════════════════════════════════════════════════════════════════════
# STEP C – Improved time alignment (unchanged from unmasked version)
# ══════════════════════════════════════════════════════════════════════════════
def align_windowed(fwd, Z_cn_df, tol_s=TOL_S):
    """
    For each Parsivel step, average all Cloudnet snapshots within ±tol_s seconds
    in LINEAR reflectivity units, then convert back to dBZ.
    Parsivel steps with 0 Cloudnet matches are dropped.
    """
    t_p_unix  = fwd.index.astype('int64').values / 1e9
    t_cn_unix = Z_cn_df.index.astype('int64').values / 1e9

    dt    = np.abs(t_p_unix[:, np.newaxis] - t_cn_unix[np.newaxis, :])
    match = dt <= tol_s
    n_cn  = match.sum(axis=1)

    Z_linear    = 10.0 ** (Z_cn_df.values / 10.0)
    Z_linear    = np.where(np.isfinite(Z_cn_df.values), Z_linear, np.nan)
    Z_linear_nz = np.where(np.isfinite(Z_linear), Z_linear, 0.0)
    valid_counts = np.where(np.isfinite(Z_linear), 1.0, 0.0)

    sum_Z     = match.astype(float) @ Z_linear_nz
    sum_valid = match.astype(float) @ valid_counts

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_Z_linear = np.where(sum_valid > 0, sum_Z / sum_valid, np.nan)
        mean_Ze_dBZ   = np.where(mean_Z_linear > 0,
                                  10.0 * np.log10(mean_Z_linear),
                                  np.nan)

    cn_aligned = pd.DataFrame(mean_Ze_dBZ, index=fwd.index, columns=Z_cn_df.columns)
    cn_aligned['n_cn'] = n_cn

    merged = pd.concat([fwd, cn_aligned], axis=1)

    n_before = len(merged)
    merged = merged[merged['n_cn'] > 0].copy()
    n_dropped = n_before - len(merged)
    if n_dropped > 0:
        print(f'        Dropped {n_dropped} Parsivel steps with no Cloudnet match')

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# STEP D – Plots (identical structure, titles updated to reflect masking)
# ══════════════════════════════════════════════════════════════════════════════
def make_plots(merged, h_labels, date_str, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rain_mask  = merged['rr'] >= RR_MIN
    date_label = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
    mask_note  = f'D {D_MIN}–{D_MAX} mm | rr ≥ {RR_MIN} mm h⁻¹ | Cloudnet: hydrometeors + no insects'

    # ── Plot 1 : Ze vs Time ───────────────────────────────────────────────────
    fig, axes = plt.subplots(N_BINS, 1, figsize=(14, 2.4 * N_BINS), sharex=True)
    fig.suptitle(
        f'Ze vs Time – Jülich {date_label}  [±{TOL_S:.0f} s window mean]  [MASKED]\n'
        f'Cloudnet: hydrometeors + no insects  |  FWD: {mask_note}',
        fontsize=10, y=1.002)

    for i, (ax, hl) in enumerate(zip(axes, h_labels)):
        ax.fill_between(merged.index, 0, 1, where=rain_mask,
                        transform=ax.get_xaxis_transform(),
                        color='skyblue', alpha=0.35, label=f'rr ≥ {RR_MIN} mm h⁻¹')
        ax.plot(merged.index, merged[hl],
                color=COLOURS[i], lw=1.3, label=f'Cloudnet {hl} (masked)')
        ax.plot(merged.index, merged['Ze_tmm'],
                color='crimson', lw=0.9, ls='--', alpha=0.85,
                label=f'FWD Ze_tmm (D {D_MIN}–{D_MAX} mm)')
        ax.set_ylabel('Ze [dBZ]', fontsize=8)
        ax.set_ylim(-20, 45)
        ax.set_title(f'Height bin {i}  –  {hl} AMSL', fontsize=9, pad=2)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc='upper right', ncol=3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
    axes[-1].set_xlabel(f'Time UTC [{date_label}]')
    plt.tight_layout()
    p = out_dir / f'Ze_vs_time_{date_str}.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'        Saved → {p.name}')

    # ── Plot 2 : Ze vs Rain Rate ──────────────────────────────────────────────
    ncols = 2
    nrows = int(np.ceil(N_BINS / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.8 * nrows))
    axes_flat = axes.flatten()
    fig.suptitle(
        f'Ze vs Rain Rate – Jülich {date_label}  [±{TOL_S:.0f} s window mean]  [MASKED]\n'
        f'Cloudnet: hydrometeors + no insects  (●)  vs  FWD Ze_tmm D {D_MIN}–{D_MAX} mm  (✕ red)  '
        f'rr ≥ {RR_MIN} mm h⁻¹',
        fontsize=10)

    rr_r = merged.loc[rain_mask, 'rr'].values
    for i, (ax, hl) in enumerate(zip(axes_flat[:N_BINS], h_labels)):
        z_cn  = merged.loc[rain_mask, hl].values
        z_fwd = merged.loc[rain_mask, 'Ze_tmm'].values
        valid = np.isfinite(z_cn) & np.isfinite(z_fwd) & np.isfinite(rr_r)
        ax.scatter(rr_r[valid], z_cn[valid],
                   c=[COLOURS[i]], s=22, alpha=0.85, zorder=3, label='Cloudnet (masked)')
        ax.scatter(rr_r[valid], z_fwd[valid],
                   marker='x', s=22, color='crimson', alpha=0.7,
                   label=f'FWD Ze_tmm (D {D_MIN}–{D_MAX} mm)')
        ax.set_xlabel('Rain rate [mm h⁻¹]', fontsize=8)
        ax.set_ylabel('Ze [dBZ]', fontsize=8)
        ax.set_title(f'Bin {i}  –  {hl} AMSL', fontsize=9)
        ax.set_ylim(-5, 45)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for ax in axes_flat[N_BINS:]:
        ax.set_visible(False)
    plt.tight_layout()
    p = out_dir / f'Ze_vs_rr_{date_str}.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'        Saved → {p.name}')

    # ── Plot 3 : Ze_fwd vs Ze_cloudnet scatter ────────────────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.8 * nrows))
    axes_flat = axes.flatten()
    fig.suptitle(
        f'Ze FWD vs Ze Cloudnet – Jülich {date_label}  [±{TOL_S:.0f} s window mean]  [MASKED]\n'
        f'FWD: D {D_MIN}–{D_MAX} mm, 35.6 GHz / 293.15 K  |  '
        f'Cloudnet: hydrometeors + no insects  |  colour = rain rate [mm h⁻¹]',
        fontsize=10)

    lim = [-5, 45]
    for i, (ax, hl) in enumerate(zip(axes_flat[:N_BINS], h_labels)):
        z_cn  = merged.loc[rain_mask, hl].values
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
        ax.set_xlabel(f'Ze_tmm FWD [dBZ]  (D {D_MIN}–{D_MAX} mm)', fontsize=8)
        ax.set_ylabel('Ze Cloudnet [dBZ]  (masked)', fontsize=8)
        ax.set_title(f'Bin {i}  –  {hl} AMSL', fontsize=9)
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[N_BINS:]:
        ax.set_visible(False)
    plt.tight_layout()
    p = out_dir / f'Ze_fwd_vs_cloudnet_{date_str}.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'        Saved → {p.name}')

    # ── Plot 4 : alignment diagnostics ───────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(merged.index, merged['n_cn'], color='steelblue', lw=1)
    axes[0].axhline(2, color='k', ls='--', lw=0.8, label='expected 2')
    axes[0].set_ylabel('# Cloudnet snapshots\naveraged per Parsivel step')
    axes[0].set_ylim(0, 5)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title(
        f'Alignment diagnostics – {date_label}  (TOL = ±{TOL_S:.0f} s)  [MASKED]')

    axes[1].fill_between(merged.index, 0, 1, where=rain_mask,
                         transform=axes[1].get_xaxis_transform(),
                         color='skyblue', alpha=0.5, label=f'rr ≥ {RR_MIN} mm h⁻¹')
    axes[1].plot(merged.index, merged['rr'], color='teal', lw=0.8)
    axes[1].set_ylabel('Rain rate [mm h⁻¹]')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha='right')
    axes[1].set_xlabel(f'Time UTC [{date_label}]')
    plt.tight_layout()
    p = out_dir / f'alignment_diagnostics_{date_str}.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'        Saved → {p.name}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def process_date(date_str):
    print(f'\n{"="*60}')
    print(f'  Processing {date_str}  [MASKED]')
    print(f'{"="*60}')

    out_dir = OUT_BASE / date_str

    print('STEP A  Running FWD simulation (D filter applied) …')
    fwd = run_fwd_sim(date_str)
    n_rain = int((fwd['rr'] >= RR_MIN).sum())
    print(f'        {len(fwd)} Parsivel steps  |  {n_rain} rainy (rr ≥ {RR_MIN} mm/h)')

    print('STEP B  Loading Cloudnet Ze (category_bits mask applied) …')
    Z_cn_df, h_labels, h_m = load_cloudnet(date_str)
    print(f'        {len(Z_cn_df)} Cloudnet steps  |  '
          f'heights {h_m[0]:.0f}–{h_m[-1]:.0f} m AMSL')

    print(f'STEP C  Aligning (±{TOL_S:.0f} s window mean, linear units) …')
    merged = align_windowed(fwd, Z_cn_df, tol_s=TOL_S)
    n_cn_vals = merged['n_cn'].value_counts().sort_index()
    print(f'        {len(merged)} steps after alignment')
    print(f'        Cloudnet snapshots per step: {n_cn_vals.to_dict()}')

    print('STEP D  Producing plots …')
    make_plots(merged, h_labels, date_str, out_dir)

    rain_mask = merged['rr'] >= RR_MIN
    print(f'\n── Bias & RMSE  [Ze_cloudnet(masked) − Ze_fwd(D {D_MIN}–{D_MAX} mm)]  '
          f'rr ≥ {RR_MIN} mm/h ──')
    print(f'  {"Height":>10}  {"N":>4}  {"Bias [dB]":>10}  {"RMSE [dB]":>10}')
    print(f'  {"─"*44}')
    for hl in h_labels:
        z_cn  = merged.loc[rain_mask, hl].values
        z_fwd = merged.loc[rain_mask, 'Ze_tmm'].values
        valid = np.isfinite(z_cn) & np.isfinite(z_fwd)
        if valid.sum() > 0:
            d = z_cn[valid] - z_fwd[valid]
            print(f'  {hl:>10}  {valid.sum():>4}  {d.mean():>+10.2f}  '
                  f'{np.sqrt((d**2).mean()):>10.2f}')
        else:
            print(f'  {hl:>10}  {"0":>4}  {"—":>10}  {"—":>10}')
    print('──────────────────────────────────────────────────────────────')


# ══════════════════════════════════════════════════════════════════════════════
# MONTHLY – collect all days then plot
# ══════════════════════════════════════════════════════════════════════════════
def make_monthly_plots(merged, h_labels, year_month_str, out_dir):
    """Produce Ze_vs_rr and Ze_fwd_vs_cloudnet plots for a full month."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rain_mask  = merged['rr'] >= RR_MIN
    month_label = f'{year_month_str[:4]}-{year_month_str[4:6]}'
    ncols = 2
    nrows = int(np.ceil(N_BINS / ncols))

    # ── Plot 2 : Ze vs Rain Rate ──────────────────────────────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.8 * nrows))
    axes_flat = axes.flatten()
    fig.suptitle(
        f'Ze vs Rain Rate – Jülich {month_label}  [±{TOL_S:.0f} s window mean]  [MASKED]\n'
        f'Cloudnet: hydrometeors + no insects  (●)  vs  FWD Ze_tmm D {D_MIN}–{D_MAX} mm  (✕ red)  '
        f'rr ≥ {RR_MIN} mm h⁻¹',
        fontsize=10)

    rr_r = merged.loc[rain_mask, 'rr'].values
    for i, (ax, hl) in enumerate(zip(axes_flat[:N_BINS], h_labels)):
        z_cn  = merged.loc[rain_mask, hl].values
        z_fwd = merged.loc[rain_mask, 'Ze_tmm'].values
        valid = np.isfinite(z_cn) & np.isfinite(z_fwd) & np.isfinite(rr_r)
        ax.scatter(rr_r[valid], z_cn[valid],
                   c=[COLOURS[i]], s=8, alpha=0.5, zorder=3, label='Cloudnet (masked)')
        ax.scatter(rr_r[valid], z_fwd[valid],
                   marker='x', s=8, color='crimson', alpha=0.4,
                   label=f'FWD Ze_tmm (D {D_MIN}–{D_MAX} mm)')
        ax.set_xlabel('Rain rate [mm h⁻¹]', fontsize=8)
        ax.set_ylabel('Ze [dBZ]', fontsize=8)
        ax.set_title(f'Bin {i}  –  {hl} AMSL  (N={valid.sum()})', fontsize=9)
        ax.set_ylim(-5, 45)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for ax in axes_flat[N_BINS:]:
        ax.set_visible(False)
    plt.tight_layout()
    p = out_dir / f'Ze_vs_rr_{year_month_str}.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'        Saved → {p.name}')

    # ── Plot 3 : Ze_fwd vs Ze_cloudnet scatter ────────────────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.8 * nrows))
    axes_flat = axes.flatten()
    fig.suptitle(
        f'Ze FWD vs Ze Cloudnet – Jülich {month_label}  [±{TOL_S:.0f} s window mean]  [MASKED]\n'
        f'FWD: D {D_MIN}–{D_MAX} mm, 35.6 GHz / 293.15 K  |  '
        f'Cloudnet: hydrometeors + no insects  |  colour = rain rate [mm h⁻¹]',
        fontsize=10)

    lim = [-5, 45]
    for i, (ax, hl) in enumerate(zip(axes_flat[:N_BINS], h_labels)):
        z_cn  = merged.loc[rain_mask, hl].values
        z_fwd = merged.loc[rain_mask, 'Ze_tmm'].values
        rr_r2 = merged.loc[rain_mask, 'rr'].values
        valid = np.isfinite(z_cn) & np.isfinite(z_fwd)
        n_v   = valid.sum()

        if n_v > 0:
            sc = ax.scatter(z_fwd[valid], z_cn[valid],
                            c=rr_r2[valid], cmap='plasma',
                            vmin=RR_MIN, vmax=np.nanmax(rr_r2) + 0.1,
                            s=6, alpha=0.5)
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
        ax.set_xlabel(f'Ze_tmm FWD [dBZ]  (D {D_MIN}–{D_MAX} mm)', fontsize=8)
        ax.set_ylabel('Ze Cloudnet [dBZ]  (masked)', fontsize=8)
        ax.set_title(f'Bin {i}  –  {hl} AMSL', fontsize=9)
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[N_BINS:]:
        ax.set_visible(False)
    plt.tight_layout()
    p = out_dir / f'Ze_fwd_vs_cloudnet_{year_month_str}.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'        Saved → {p.name}')


def process_month(year_month_str):
    """
    Collect all available days in a month, run FWD+alignment for each,
    concatenate and produce monthly Ze_vs_rr and Ze_fwd_vs_cloudnet plots.
    year_month_str e.g. '202506'
    """
    import calendar
    yyyy, mm = int(year_month_str[:4]), int(year_month_str[4:6])
    n_days = calendar.monthrange(yyyy, mm)[1]
    month_label = f'{yyyy}-{mm:02d}'
    out_dir = OUT_BASE / 'monthly'

    print(f'\n{"="*60}')
    print(f'  Monthly processing  {month_label}  ({n_days} days)  [MASKED]')
    print(f'{"="*60}')

    all_merged = []
    h_labels = h_m = None

    for day in range(1, n_days + 1):
        date_str = f'{yyyy}{mm:02d}{day:02d}'
        try:
            fwd = run_fwd_sim(date_str)
            Z_cn_df, h_labels, h_m = load_cloudnet(date_str)
            merged = align_windowed(fwd, Z_cn_df)
            all_merged.append(merged)
            n_rain = int((merged['rr'] >= RR_MIN).sum())
            print(f'  {date_str}  ✓  rainy steps: {n_rain}')
        except FileNotFoundError as e:
            print(f'  {date_str}  SKIP: {e}')
        except Exception as e:
            print(f'  {date_str}  ERROR: {e}')

    if not all_merged:
        print('  No data found for this month.')
        return

    merged_all = pd.concat(all_merged).sort_index()
    total_rain = int((merged_all['rr'] >= RR_MIN).sum())
    print(f'\n  Total days loaded : {len(all_merged)}')
    print(f'  Total rainy steps : {total_rain}')
    print(f'  Producing monthly plots …')
    make_monthly_plots(merged_all, h_labels, year_month_str, out_dir)


if __name__ == '__main__':
    args = sys.argv[1:] if len(sys.argv) > 1 else TEST_DATES
    for arg in args:
        if len(arg) == 6:   # YYYYMM → monthly
            process_month(arg)
        elif len(arg) == 8: # YYYYMMDD → single day
            try:
                process_date(arg)
            except FileNotFoundError as e:
                print(f'  SKIP {arg}: {e}')
        else:
            print(f'  Unknown argument: {arg}  (expected YYYYMMDD or YYYYMM)')
