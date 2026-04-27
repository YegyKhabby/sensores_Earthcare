# -*- coding: utf-8 -*-
"""
compare_ze_fwd_tower_znc.py
===========================
Date-generic comparison of forward-simulated Ze (tower Parsivel 452070, T-matrix)
against MIRA-35 tower.znc Zg, for all range gates.

Geometry
--------
MIRA-35 tower beam: elevation = 19°, site altitude = 114 m ASL
Height formula: height_ASL [m] = range_slant [m] * sin(19°) + 114
Tower Parsivel (452070) is at 211 m ASL.
→ Gate closest to disdrometer found automatically (typically gate 4, ~208 m ASL).

Time alignment
--------------
For each Parsivel step (60 s integration, stamped at end):
  - Find all radar snapshots with |t_radar − t_parsivel| ≤ TOL_S (30 s)
  - Average in LINEAR reflectivity units (Z = 10^(Zg/10)), then back to dBZ
  - If no valid radar snapshot found within window: drop that Parsivel step

The tower.znc files run in burst mode (~2 min every ~30 min, ~1 s resolution),
so most Parsivel steps will have no radar match — that is expected.

Usage
-----
  python3 compare_ze_fwd_tower_znc.py 20250606     # single date
  python3 compare_ze_fwd_tower_znc.py 202506       # full month

Outputs per date (output/comparison/tower_znc_vs_fwd/<date>/):
  Ze_vs_rr_<date>.png
  Ze_fwd_vs_tower_<date>.png

Outputs per month (output/comparison/tower_znc_vs_fwd/monthly/):
  Ze_vs_rr_<month>.png
  Ze_fwd_vs_tower_<month>.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import h5py.h5s as h5s
import h5py.h5t as h5t

# ── raincoat ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / 'raincoat'))
from raincoat.FWD_sim import FWD_sim
import raincoat.disdrometer.pars_class as pc

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
TOWER_PARS_BASE = Path('/data/obs/site/jue-tower1/parsivel_452070/l1')
RADAR_BASE      = Path('/data/obs/site/jue/joyrad35')
SCAT_TABLE      = Path('/work/yegy_project/Feb2026_new/raincoat/samplefiles/'
                       'scattering/293.15_35.5GHz_elev19deg.csv')
OUT_BASE        = Path('/work/yegy_project/Feb2026_new/output/comparison/'
                       'tower_znc_vs_fwd')

N_GATES      = 10      # compare first N range gates
RR_MIN       = 0.1    # mm/h  rain-rate threshold
TOL_S        = 30.0   # s     ±window for radar snapshot averaging
ELEV_DEG     = 19.0   # °     MIRA-35 tower beam elevation
SITE_ALT_M   = 114.0  # m ASL radar site altitude
DISDRO_ALT_M = 211.0  # m ASL tower Parsivel 452070 altitude

A_SENSOR  = 54e-4   # m²  OTT Parsivel2 sampling area
DT        = 60.0    # s   Parsivel integration interval

ELEV_RAD = np.deg2rad(ELEV_DEG)

# Standard OTT Parsivel2 bin arrays (same for all units)
VCLASSES = np.array([
    0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
    1.10, 1.30, 1.50, 1.70, 1.90, 2.20, 2.60, 3.00, 3.40, 3.80,
    4.40, 5.20, 6.00, 6.80, 7.60, 8.80, 10.4, 12.0, 13.6, 15.2,
    17.6, 20.8,
])   # m/s

DCLASSES = np.array([
    0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187,
    1.375, 1.625, 1.875, 2.125, 2.375, 2.750, 3.250, 3.750, 4.250, 4.750,
    5.500, 6.500, 7.500, 8.500, 9.500, 11.00, 13.00, 15.00, 17.00, 19.00,
    21.50, 24.50,
])   # mm

DWIDTH = np.array([
    0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
    0.250, 0.250, 0.250, 0.250, 0.250,
    0.500, 0.500, 0.500, 0.500, 0.500,
    1.000, 1.000, 1.000, 1.000, 1.000,
    2.000, 2.000, 2.000, 2.000, 2.000,
    3.000, 3.000,
])   # mm


# ══════════════════════════════════════════════════════════════════════════════
# HDF5 low-level reader (bypasses non-standard float type in METEK files)
# ══════════════════════════════════════════════════════════════════════════════
def read_f32(ds):
    shape = ds.shape
    if shape == ():
        buf = np.zeros(1, dtype='float32')
        fsp = ds.id.get_space()
        msp = h5s.create_simple((1,))
        ds.id.read(msp, fsp, buf, h5t.IEEE_F32LE)
        return float(buf[0])
    buf = np.zeros(shape, dtype='float32')
    fsp = ds.id.get_space()
    msp = h5s.create_simple(shape)
    ds.id.read(msp, fsp, buf, h5t.IEEE_F32LE)
    return buf


# ══════════════════════════════════════════════════════════════════════════════
# STEP A – FWD simulation from tower Parsivel 452070 CSV
# ══════════════════════════════════════════════════════════════════════════════
def run_fwd_sim(date_str):
    """
    Parse tower Parsivel 452070 CSV, reconstruct N(D,t), run T-matrix FWD sim.
    Returns DataFrame indexed by UTC timestamps with columns:
      Ze_tmm [dBZ], Ze_ray [dBZ], A [dB/km], rr [mm/h]
    Non-rainy steps have Ze_tmm = Ze_ray = NaN.
    """
    yyyy, mm, dd = date_str[:4], date_str[4:6], date_str[6:]
    csv_path = TOWER_PARS_BASE / yyyy / mm / dd / f'{date_str}_parsivel_tower.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'Tower Parsivel CSV not found: {csv_path}')

    # ── parse CSV ────────────────────────────────────────────────────────────
    with open(csv_path, 'r', encoding='latin-1') as fh:
        fh.readline()          # skip header row
        raw_lines = fh.readlines()

    timestamps, rr_list, M_list = [], [], []
    n_skipped = 0
    for line in raw_lines:
        parts = line.rstrip('\n').split(';')
        if len(parts) < 1106:
            n_skipped += 1
            continue
        try:
            ts = pd.Timestamp(parts[0].strip(), tz='UTC')
        except Exception:
            n_skipped += 1
            continue
        timestamps.append(ts)
        try:
            rr_list.append(float(parts[1]))
        except ValueError:
            rr_list.append(np.nan)
        m_flat = np.array(parts[82:1106], dtype=float)
        M_list.append(m_flat.reshape(32, 32))

    if n_skipped:
        print(f'        WARNING: {n_skipped} lines skipped (short/malformed)')

    time_idx = pd.DatetimeIndex(timestamps)
    rr_arr   = np.array(rr_list, dtype=float)
    M        = np.stack(M_list, axis=2)            # (32v, 32D, N_t)

    # ── reconstruct N(D,t) from M matrix ─────────────────────────────────────
    # N(D_i,t) [m⁻³ mm⁻¹] = Σ_j M[j,i,t] / (v_j · A_sensor · Δt · ΔD_i)
    v_safe   = np.where(VCLASSES > 0, VCLASSES, np.inf)
    M_vsum   = (M / v_safe[:, np.newaxis, np.newaxis]).sum(axis=0)  # (32D, N_t)
    N_from_M = M_vsum / (A_SENSOR * DT * DWIDTH[:, np.newaxis])     # m⁻³ mm⁻¹

    with np.errstate(divide='ignore', invalid='ignore'):
        log10_N = np.where(N_from_M > 0, np.log10(N_from_M), np.nan)

    # ── T-matrix FWD simulation ───────────────────────────────────────────────
    _, bin_edges = pc.pars_class()
    fwd = FWD_sim(str(SCAT_TABLE), time_idx, log10_N, bin_edges)
    fwd['rr'] = rr_arr

    # mask non-rain periods
    rain_mask = rr_arr >= RR_MIN
    fwd.loc[~rain_mask, 'Ze_tmm'] = np.nan
    fwd.loc[~rain_mask, 'Ze_ray'] = np.nan

    n_rainy = int(rain_mask.sum())
    print(f'        {len(time_idx)} steps loaded, {n_rainy} rainy (rr ≥ {RR_MIN} mm/h)')
    return fwd


# ══════════════════════════════════════════════════════════════════════════════
# STEP B – Load all *_tower.znc files for one day
# ══════════════════════════════════════════════════════════════════════════════
def load_znc_files(date_str):
    """
    Load all *_tower.znc files, apply Saturatedco mask, concatenate.
    Returns:
      t_rad_unix : (N_r,)            float64  unix seconds
      Zg_lin_nz  : (N_r, N_gates)   float64  linear Zg, 0 where masked/invalid
      valid_cnts : (N_r, N_gates)   float64  1 where valid, 0 elsewhere
      ranges_m   : (N_gates,)        slant range [m] from first valid file
    """
    yyyy, mm, dd = date_str[:4], date_str[4:6], date_str[6:]
    radar_dir = RADAR_BASE / yyyy / mm / dd
    znc_files = sorted(radar_dir.glob('*_tower.znc'))
    if not znc_files:
        raise FileNotFoundError(f'No *_tower.znc files in {radar_dir}')

    t_list, Zg_list, sat_list = [], [], []
    ranges_m = None
    n_loaded = 0

    for znc_path in znc_files:
        try:
            f = h5py.File(str(znc_path), 'r')
        except Exception as e:
            print(f'        WARNING: cannot open {znc_path.name}: {e}')
            continue

        t_r  = read_f32(f['time']).astype('float64')   # (T,)
        Zg   = read_f32(f['Zg'])                        # (T, R) linear mm⁶/m³
        Sat  = read_f32(f['Saturatedco'])               # (T, R)
        if ranges_m is None:
            r_all    = read_f32(f['range'])             # (R,) slant range [m]
            ranges_m = r_all[:N_GATES].astype('float64')
        f.close()

        if Zg.ndim < 2 or Zg.shape[1] < N_GATES:
            continue

        t_list.append(t_r)
        Zg_list.append(Zg[:, :N_GATES].astype('float64'))
        sat_list.append(Sat[:, :N_GATES].astype('float64'))
        n_loaded += 1

    print(f'        {n_loaded}/{len(znc_files)} znc files loaded')

    t_rad_unix = np.concatenate(t_list)                        # (N_r,)
    Zg_all     = np.concatenate(Zg_list, axis=0)              # (N_r, N_gates)
    Sat_all    = np.concatenate(sat_list, axis=0)             # (N_r, N_gates)

    # sort by time
    order      = np.argsort(t_rad_unix)
    t_rad_unix = t_rad_unix[order]
    Zg_all     = Zg_all[order]
    Sat_all    = Sat_all[order]

    # apply Saturatedco mask: set saturated samples to NaN
    Zg_all = np.where(Sat_all == 0, Zg_all, np.nan)

    # build arrays for matrix multiply (0 where NaN/non-positive)
    valid_cnts = np.where(np.isfinite(Zg_all) & (Zg_all > 0), 1.0, 0.0)
    Zg_lin_nz  = np.where(valid_cnts > 0, Zg_all, 0.0)

    print(f'        {len(t_rad_unix)} total radar time steps')
    return t_rad_unix, Zg_lin_nz, valid_cnts, ranges_m


# ══════════════════════════════════════════════════════════════════════════════
# STEP C – Time alignment: ±TOL_S windowed mean in linear units
# ══════════════════════════════════════════════════════════════════════════════
def align_windowed(fwd, t_rad_unix, Zg_lin_nz, valid_cnts):
    """
    For each Parsivel step, average all radar snapshots within ±TOL_S in
    linear units (avoids dBZ averaging artefacts).  Drop steps with no match.

    Returns dict:
      ze_fwd  : (N,)         FWD Ze [dBZ]
      ze_rad  : (N, N_gates) radar Ze [dBZ] per gate (NaN if no data for that gate)
      rr      : (N,)         rain rate [mm/h]
      t_unix  : (N,)         Parsivel step unix timestamps
    """
    t_p_unix = np.array(fwd.index.view('int64'), dtype='float64') / 1e9  # (N_pars,)

    # match matrix: shape (N_pars, N_radar)
    dt    = np.abs(t_p_unix[:, np.newaxis] - t_rad_unix[np.newaxis, :])
    match = (dt <= TOL_S).astype(float)

    # linear-unit sums per gate: (N_pars, N_gates)
    sum_Z = match @ Zg_lin_nz    # (N_pars, N_gates)
    sum_v = match @ valid_cnts   # (N_pars, N_gates)

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_Z_lin = np.where(sum_v > 0, sum_Z / np.where(sum_v > 0, sum_v, 1.0), np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_Ze_dBZ = np.where(mean_Z_lin > 0, 10.0 * np.log10(mean_Z_lin), np.nan)
    # shape: (N_pars, N_gates)

    ze_fwd_all = fwd['Ze_tmm'].values
    rr_all     = fwd['rr'].values

    # keep only steps that are rainy (finite Ze_fwd) AND have ≥1 gate with data
    has_radar  = np.any(sum_v > 0, axis=1)
    valid_step = has_radar & np.isfinite(ze_fwd_all)

    n_drop = int(np.sum(~has_radar & np.isfinite(ze_fwd_all)))
    print(f'        Dropped {n_drop} rainy Parsivel steps with no radar match')

    idx = np.where(valid_step)[0]
    return {
        'ze_fwd': ze_fwd_all[idx],
        'ze_rad': mean_Ze_dBZ[idx],   # (N, N_gates)
        'rr':     rr_all[idx],
        't_unix': t_p_unix[idx],
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP D – Gate geometry helper
# ══════════════════════════════════════════════════════════════════════════════
def gate_info(ranges_m):
    """
    Compute height ASL per gate and find gate closest to disdrometer.
    Returns (heights_asl, gate_closest_index).
    """
    heights_asl  = ranges_m * np.sin(ELEV_RAD) + SITE_ALT_M
    gate_closest = int(np.argmin(np.abs(heights_asl - DISDRO_ALT_M)))
    return heights_asl, gate_closest


def _gate_label(g, ranges_m, heights_asl):
    return (f'Gate {g}  |  {ranges_m[g]:.0f} m slant range  |  '
            f'{heights_asl[g]:.0f} m ASL')


# ══════════════════════════════════════════════════════════════════════════════
# STEP E – Plots
# ══════════════════════════════════════════════════════════════════════════════
def _scatter_gate(ax, ze_fwd, ze_rad_g, rr, g, ranges_m, heights_asl, g_close):
    """Fill one subplot axes for gate g. Returns True if plotted."""
    mask = np.isfinite(ze_fwd) & np.isfinite(ze_rad_g) & (rr >= RR_MIN)
    N_g  = int(mask.sum())

    lbl = _gate_label(g, ranges_m, heights_asl)

    # highlight gate closest to disdrometer with red frame
    lw_spine = 2.5 if g == g_close else 0.8
    ec_spine = 'firebrick' if g == g_close else '#bbbbbb'
    for spine in ax.spines.values():
        spine.set_linewidth(lw_spine)
        spine.set_edgecolor(ec_spine)

    if N_g < 2:
        ax.set_title(lbl + f'\nN={N_g}', fontsize=7, pad=4)
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='gray')
        ax.set_xlabel('Ze FWD [dBZ]', fontsize=7)
        ax.set_ylabel('Zg radar [dBZ]', fontsize=7)
        return False

    bias = float(np.mean(ze_fwd[mask] - ze_rad_g[mask]))
    rmse = float(np.sqrt(np.mean((ze_fwd[mask] - ze_rad_g[mask])**2)))

    sc = ax.scatter(ze_fwd[mask], ze_rad_g[mask],
                    c=rr[mask], cmap='plasma',
                    vmin=RR_MIN, vmax=max(float(rr[mask].max()), 1.0),
                    s=5, alpha=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, pad=0.02).set_label('rr [mm/h]', fontsize=7)

    all_v = np.concatenate([ze_fwd[mask], ze_rad_g[mask]])
    lim   = [np.nanpercentile(all_v, 1) - 2, np.nanpercentile(all_v, 99) + 2]
    ax.plot(lim, lim, 'k--', lw=0.8, zorder=2, label='1:1')
    ax.plot(lim, [v + bias for v in lim], color='firebrick', lw=0.8, ls=':',
            zorder=2, label=f'bias {bias:+.2f} dB')
    ax.set_xlim(lim); ax.set_ylim(lim)

    title_str = f'{lbl}\nN={N_g}  bias={bias:+.2f} dB  RMSE={rmse:.2f} dB'
    if g == g_close:
        title_str += '\n★ closest to disdrometer'
    ax.set_title(title_str, fontsize=7, pad=4)
    ax.set_xlabel('Ze FWD [dBZ]\n(Parsivel 452070, T-matrix 35.5 GHz 19°)', fontsize=7)
    ax.set_ylabel('Zg MIRA-35 tower.znc [dBZ]\n(Saturatedco masked, linear mean)', fontsize=7)
    ax.legend(fontsize=6, loc='upper left')
    ax.grid(True, alpha=0.2)
    return True


def _make_gate_figure(ze_fwd, ze_rad, rr, ranges_m, suptitle):
    heights_asl, g_close = gate_info(ranges_m)
    ncols = 5
    nrows = int(np.ceil(N_GATES / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for g in range(N_GATES):
        _scatter_gate(axes[g], ze_fwd, ze_rad[:, g], rr, g,
                      ranges_m, heights_asl, g_close)

    for g in range(N_GATES, len(axes)):
        axes[g].set_visible(False)

    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def make_daily_plots(date_str, aligned, ranges_m):
    yyyy_mm_dd = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
    out_dir = OUT_BASE / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    heights_asl, g_close = gate_info(ranges_m)
    ze_fwd = aligned['ze_fwd']
    ze_rad = aligned['ze_rad']
    rr     = aligned['rr']
    rainy  = rr >= RR_MIN

    # ── Plot 1: Ze FWD vs rain rate ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = rainy & np.isfinite(ze_fwd)
    if valid.sum() > 0:
        sc = ax.scatter(rr[valid], ze_fwd[valid],
                        c=ze_fwd[valid], cmap='viridis',
                        s=8, alpha=0.6)
        plt.colorbar(sc, ax=ax, label='Ze FWD [dBZ]')
    ax.set_xlabel('Rain rate [mm h⁻¹]')
    ax.set_ylabel('Ze FWD simulated [dBZ]')
    ax.set_title(
        f'Ze FWD vs rain rate — Jülich {yyyy_mm_dd}\n'
        f'Tower Parsivel 452070 | T-matrix 35.5 GHz, 288.15 K, 19° elev\n'
        f'N rainy steps with radar match = {int(valid.sum())}'
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fname1 = out_dir / f'Ze_vs_rr_{date_str}.png'
    fig.savefig(fname1, dpi=150)
    plt.close(fig)
    print(f'        Saved → {fname1.name}')

    # ── Plot 2: per-gate scatter ───────────────────────────────────────────────
    suptitle = (
        f'MIRA-35 tower.znc vs FWD Ze — Jülich {yyyy_mm_dd}\n'
        f'Tower Parsivel 452070 | Saturatedco masked | ±{TOL_S:.0f} s windowed mean in linear units\n'
        f'Height ASL = range × sin({ELEV_DEG:.0f}°) + {SITE_ALT_M:.0f} m  |  '
        f'Disdrometer at {DISDRO_ALT_M:.0f} m ASL  (★ gate {g_close}, '
        f'{ranges_m[g_close]:.0f} m slant range, '
        f'{heights_asl[g_close]:.0f} m ASL)'
    )
    fig = _make_gate_figure(ze_fwd, ze_rad, rr, ranges_m, suptitle)
    fname2 = out_dir / f'Ze_fwd_vs_tower_{date_str}.png'
    fig.savefig(fname2, dpi=150)
    plt.close(fig)
    print(f'        Saved → {fname2.name}')


def make_monthly_plots(month_str, all_ze_fwd, all_ze_rad, all_rr, ranges_m):
    yyyy, mm = month_str[:4], month_str[4:6]
    out_dir = OUT_BASE / 'monthly'
    out_dir.mkdir(parents=True, exist_ok=True)

    heights_asl, g_close = gate_info(ranges_m)
    rainy = all_rr >= RR_MIN

    # ── Plot 1: Ze FWD vs rain rate ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = rainy & np.isfinite(all_ze_fwd)
    if valid.sum() > 0:
        sc = ax.scatter(all_rr[valid], all_ze_fwd[valid],
                        c=all_ze_fwd[valid], cmap='viridis',
                        s=4, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='Ze FWD [dBZ]')
    ax.set_xlabel('Rain rate [mm h⁻¹]')
    ax.set_ylabel('Ze FWD simulated [dBZ]')
    ax.set_title(
        f'Ze FWD vs rain rate — {yyyy}-{mm} (monthly)\n'
        f'Tower Parsivel 452070 | N = {int(valid.sum())} matched rainy steps'
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fname1 = out_dir / f'Ze_vs_rr_{month_str}.png'
    fig.savefig(fname1, dpi=150)
    plt.close(fig)
    print(f'        Saved → {fname1.name}')

    # ── Plot 2: per-gate scatter ───────────────────────────────────────────────
    suptitle = (
        f'MIRA-35 tower.znc vs FWD Ze — {yyyy}-{mm} (monthly)\n'
        f'Tower Parsivel 452070 | Saturatedco masked | ±{TOL_S:.0f} s windowed mean in linear units\n'
        f'Height ASL = range × sin({ELEV_DEG:.0f}°) + {SITE_ALT_M:.0f} m  |  '
        f'Disdrometer at {DISDRO_ALT_M:.0f} m ASL  (★ gate {g_close}, '
        f'{ranges_m[g_close]:.0f} m slant range, '
        f'{heights_asl[g_close]:.0f} m ASL)'
    )
    fig = _make_gate_figure(all_ze_fwd, all_ze_rad, all_rr, ranges_m, suptitle)
    fname2 = out_dir / f'Ze_fwd_vs_tower_{month_str}.png'
    fig.savefig(fname2, dpi=150)
    plt.close(fig)
    print(f'        Saved → {fname2.name}')


# ══════════════════════════════════════════════════════════════════════════════
# STEP F – process_date / process_month
# ══════════════════════════════════════════════════════════════════════════════
def process_date(date_str):
    """
    Process one day. Returns (ze_fwd, ze_rad, rr, ranges_m) or (None,)*4 on skip.
    """
    print(f'\n  {date_str}', end='  ')
    try:
        fwd = run_fwd_sim(date_str)
    except FileNotFoundError as e:
        print(f'SKIP (no Parsivel): {e}')
        return None, None, None, None

    try:
        t_rad, Zg_lin_nz, valid_cnts, ranges_m = load_znc_files(date_str)
    except FileNotFoundError as e:
        print(f'SKIP (no znc): {e}')
        return None, None, None, None

    aligned = align_windowed(fwd, t_rad, Zg_lin_nz, valid_cnts)
    n_rainy = int(np.sum(aligned['rr'] >= RR_MIN))
    print(f'  ✓  rainy steps: {n_rainy}')

    if n_rainy > 0:
        make_daily_plots(date_str, aligned, ranges_m)
    else:
        print('        No rainy steps with radar match — skipping daily plots')

    return aligned['ze_fwd'], aligned['ze_rad'], aligned['rr'], ranges_m


def process_month(month_str):
    import calendar
    yyyy = int(month_str[:4])
    mm   = int(month_str[4:6])
    n_days = calendar.monthrange(yyyy, mm)[1]

    print(f'\n{"="*60}')
    print(f'  Monthly processing  {month_str[:4]}-{month_str[4:6]}  ({n_days} days)')
    print(f'{"="*60}')

    all_ze_fwd, all_ze_rad, all_rr = [], [], []
    ranges_m_ref = None
    n_days_ok = 0

    for d in range(1, n_days + 1):
        date_str = f'{month_str}{d:02d}'
        ze_fwd, ze_rad, rr, ranges_m = process_date(date_str)
        if ze_fwd is None:
            continue
        all_ze_fwd.append(ze_fwd)
        all_ze_rad.append(ze_rad)
        all_rr.append(rr)
        if ranges_m_ref is None:
            ranges_m_ref = ranges_m
        n_days_ok += 1

    print(f'\n  Total days loaded : {n_days_ok}')
    if n_days_ok == 0 or ranges_m_ref is None:
        print('  No data — skipping monthly plots')
        return

    all_ze_fwd = np.concatenate(all_ze_fwd)
    all_ze_rad = np.concatenate(all_ze_rad, axis=0)
    all_rr     = np.concatenate(all_rr)

    n_rainy = int(np.sum(all_rr >= RR_MIN))
    print(f'  Total rainy steps matched to radar: {n_rainy}')
    print('  Producing monthly plots …')
    make_monthly_plots(month_str, all_ze_fwd, all_ze_rad, all_rr, ranges_m_ref)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    if len(sys.argv) < 2:
        # default: run single test date
        process_date('20250606')
    elif len(sys.argv[1]) == 6:
        process_month(sys.argv[1])
    elif len(sys.argv[1]) == 8:
        process_date(sys.argv[1])
    else:
        print(f'Usage: {sys.argv[0]} YYYYMMDD | YYYYMM')
        sys.exit(1)
