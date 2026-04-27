# -*- coding: utf-8 -*-
"""
compare_ze_fwd_both.py
======================
Unified comparison of forward-simulated Ze (T-matrix from Parsivel DSDs)
against two radars at Jülich, with identical matching/masking logic.

  (1) Cloudnet JOYRAD-35  (vertical, 35.6 GHz)  vs surface Parsivel (114 m ASL)
  (2) MIRA-35 tower beam  (19° elev, 35.5 GHz)  vs tower Parsivel 452070 (211 m ASL)

Science goal:
  (2) is the near-zero-separation reference (~3 m between sensors).
  (1) has a 141 m vertical gap. The bias difference between (1) and (2)
  isolates the height-separation error.

Differences from parent scripts
---------------------------------
Vs compare_ze_fwd_cloudnet_masked_shift30.py:
  • D-filter removed  (was D_MIN=1.0, D_MAX=5.0 mm applied to FWD only)
  • RR threshold removed  (was RR_MIN=1.0 mm/h; now all matched steps kept)
  • Parsivel midpoint fix: t_p_unix -= 30.0 before distance matrix
    → ±30 s window covers [t−60, t] matching the Parsivel integration window
    (previously window was centred on end-of-interval stamp, overshooting 30 s)

Vs compare_ze_fwd_tower_znc.py:
  • RR threshold removed  (was RR_MIN=0.1 mm/h; Ze_tmm no longer masked to NaN)
  • Parsivel midpoint fix: t_p_unix -= 30.0 before distance matrix

Unchanged from both parent scripts:
  • Cloudnet: category_bits mask  (bit 1 SET=hydrometeors, bit 5 CLEAR=no insects)
  • Cloudnet: +30 s fall-time shift on Cloudnet timestamps before matching
  • Tower: Saturatedco == 0 mask only, no time shift
  • Linear-unit averaging (mm⁶/m³) before converting back to dBZ
  • ±TOL_S = 30 s matching window (after midpoint fix)
  • Scattering tables: 293.15 K (20°C) for both
  • DSD reconstruction: N(D,t) = ΣM / (v·A·Δt·ΔD), same formula

Usage
-----
  python3 compare_ze_fwd_both.py 20250606       # single date
  python3 compare_ze_fwd_both.py 202506         # full month

Outputs per date  (output/comparison/cloudnet_vs_tower_mira/<date>/):
  Ze_side_by_side_<date>.png       — 2-panel scatter: CN gate 0 | Tower closest gate
  Ze_all_cn_gates_<date>.png       — scatter per Cloudnet gate
  Ze_all_tower_gates_<date>.png    — scatter per Tower gate

Outputs per month  (output/comparison/cloudnet_vs_tower_mira/monthly/):
  monthly_Ze_side_by_side_<YYYYMM>.png
  monthly_Ze_all_cn_gates_<YYYYMM>.png
  monthly_Ze_all_tower_gates_<YYYYMM>.png
"""

import sys
import calendar
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import h5py.h5s as h5s
import h5py.h5t as h5t

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'raincoat'))
from raincoat.FWD_sim import FWD_sim
import raincoat.disdrometer.pars_class as pc

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
PARSIVEL_BASE   = Path('/data/obs/site/jue/parsivel/l1')
TOWER_PARS_BASE = Path('/data/obs/site/jue-tower1/parsivel_452070/l1')
CLOUDNET_BASE   = Path('/data/obs/site/jue/cloudnet/clu_processing/categorize')
RADAR_BASE      = Path('/data/obs/site/jue/joyrad35')

SCAT_CN = Path('/work/yegy_project/Feb2026_new/raincoat/samplefiles/'
               'scattering/293.15_35.6GHz.csv')
SCAT_TW = Path('/work/yegy_project/Feb2026_new/raincoat/samplefiles/'
               'scattering/293.15_35.5GHz_elev19deg.csv')

OUT_BASE = Path('/work/yegy_project/Feb2026_new/output/comparison/'
                'cloudnet_vs_tower_mira')

N_CN_BINS  = 10    # lowest N Cloudnet height gates
N_TW_GATES = 10   # first N tower range gates

TOL_S           = 30.0   # s   ±window for time matching
CN_TIME_SHIFT_S = 30.0   # s   fall-time shift on Cloudnet timestamps (physical)

ELEV_DEG     = 19.0    # ° MIRA-35 tower elevation
SITE_ALT_M   = 114.0   # m ASL radar site altitude
DISDRO_ALT_M = 211.0   # m ASL tower Parsivel 452070 altitude

A_SENSOR = 54e-4   # m²  OTT Parsivel2 sampling area
DT       = 60.0    # s   Parsivel integration interval

ELEV_RAD = np.deg2rad(ELEV_DEG)

TOWER_GATE = 6   # gate closest to tower Parsivel 452070 (211 m ASL); fixed hardware

# Standard OTT Parsivel2 bin arrays — used for tower CSV which has no metadata
VCLASSES = np.array([
    0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
    1.10, 1.30, 1.50, 1.70, 1.90, 2.20, 2.60, 3.00, 3.40, 3.80,
    4.40, 5.20, 6.00, 6.80, 7.60, 8.80, 10.4, 12.0, 13.6, 15.2,
    17.6, 20.8,
])
DCLASSES = np.array([
    0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187,
    1.375, 1.625, 1.875, 2.125, 2.375, 2.750, 3.250, 3.750, 4.250, 4.750,
    5.500, 6.500, 7.500, 8.500, 9.500, 11.00, 13.00, 15.00, 17.00, 19.00,
    21.50, 24.50,
])
DWIDTH = np.array([
    0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
    0.250, 0.250, 0.250, 0.250, 0.250,
    0.500, 0.500, 0.500, 0.500, 0.500,
    1.000, 1.000, 1.000, 1.000, 1.000,
    2.000, 2.000, 2.000, 2.000, 2.000,
    3.000, 3.000,
])


# ══════════════════════════════════════════════════════════════════════════════
# HDF5 low-level reader  (unchanged from compare_ze_fwd_tower_znc.py)
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
# STEP A1 – FWD from surface Parsivel NetCDF  (used for Cloudnet comparison)
# ══════════════════════════════════════════════════════════════════════════════
def run_fwd_cn(date_str):
    """
    T-matrix FWD from surface Parsivel NetCDF.
    CHANGE vs parent: D-filter removed; RR masking removed.
    Logic otherwise identical to compare_ze_fwd_cloudnet_masked_shift30.py.
    """
    yyyy, mm, dd = date_str[:4], date_str[4:6], date_str[6:]
    pars_nc = (PARSIVEL_BASE / yyyy / mm / dd /
               f'sups_joy_dm00_l1_any_v00_{date_str}.nc')
    if not pars_nc.exists():
        raise FileNotFoundError(f'Surface Parsivel not found: {pars_nc}')

    with nc.Dataset(pars_nc) as ds:
        t_raw    = np.array(ds.variables['time'][:], dtype=float)
        M        = np.array(ds.variables['M'][:])
        vclasses = np.array(ds.variables['vclasses'][:])
        dwidth   = np.array(ds.variables['dwidth'][:])
        rr       = np.array(ds.variables['rr'][:])

    t_raw[t_raw < 0] = np.nan
    if np.isnan(t_raw[0]):  t_raw[0]  = t_raw[1]  - DT
    if np.isnan(t_raw[-1]): t_raw[-1] = t_raw[-2] + DT
    time_pd = pd.to_datetime(t_raw, unit='s', utc=True)

    v_safe   = np.where(vclasses > 0, vclasses, np.inf)
    M_vsum   = (M / v_safe[:, np.newaxis, np.newaxis]).sum(axis=0)   # (D, t)
    N_from_M = M_vsum / (A_SENSOR * DT * dwidth[:, np.newaxis])      # m⁻³ mm⁻¹

    with np.errstate(divide='ignore', invalid='ignore'):
        log10_N = np.where(N_from_M > 0, np.log10(N_from_M), np.nan)

    _, bin_edges = pc.pars_class()
    fwd = FWD_sim(str(SCAT_CN), time_pd, log10_N, bin_edges)
    fwd['rr'] = pd.Series(rr, index=time_pd).values
    n_fin = int(np.isfinite(fwd['Ze_tmm'].values).sum())
    print(f'        [CN FWD] {len(fwd)} steps  |  {n_fin} with finite Ze_tmm')
    return fwd


# ══════════════════════════════════════════════════════════════════════════════
# STEP A2 – FWD from tower Parsivel 452070 CSV  (used for Tower comparison)
# ══════════════════════════════════════════════════════════════════════════════
def run_fwd_tw(date_str):
    """
    T-matrix FWD from tower Parsivel 452070 CSV.
    CHANGE vs parent: RR masking removed (Ze_tmm no longer forced to NaN for rr < 0.1).
    Logic otherwise identical to compare_ze_fwd_tower_znc.py.
    """
    yyyy, mm, dd = date_str[:4], date_str[4:6], date_str[6:]
    csv_path = (TOWER_PARS_BASE / yyyy / mm / dd /
                f'{date_str}_parsivel_tower.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f'Tower Parsivel CSV not found: {csv_path}')

    with open(csv_path, 'r', encoding='latin-1') as fh:
        fh.readline()
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
    M        = np.stack(M_list, axis=2)   # (32v, 32D, N_t)

    v_safe   = np.where(VCLASSES > 0, VCLASSES, np.inf)
    M_vsum   = (M / v_safe[:, np.newaxis, np.newaxis]).sum(axis=0)   # (32D, N_t)
    N_from_M = M_vsum / (A_SENSOR * DT * DWIDTH[:, np.newaxis])      # m⁻³ mm⁻¹

    with np.errstate(divide='ignore', invalid='ignore'):
        log10_N = np.where(N_from_M > 0, np.log10(N_from_M), np.nan)

    _, bin_edges = pc.pars_class()
    fwd = FWD_sim(str(SCAT_TW), time_idx, log10_N, bin_edges)
    fwd['rr'] = rr_arr
    n_fin = int(np.isfinite(fwd['Ze_tmm'].values).sum())
    print(f'        [TW FWD] {len(fwd)} steps  |  {n_fin} with finite Ze_tmm')
    return fwd


# ══════════════════════════════════════════════════════════════════════════════
# STEP B1 – Load Cloudnet Ze with category_bits mask
# ══════════════════════════════════════════════════════════════════════════════
def load_cloudnet(date_str):
    """
    Unchanged from compare_ze_fwd_cloudnet_masked_shift30.py.
    Mask: bit 1 SET (hydrometeors) AND bit 5 CLEAR (no insects).
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
        Z_raw = np.ma.filled(ds.variables['Z'][:, :N_CN_BINS], np.nan)
        cat   = np.array(ds.variables['category_bits'][:, :N_CN_BINS])
        h_m   = np.array(ds.variables['height'][:N_CN_BINS])

    hydro_ok  = ((cat >> 1) & 1).astype(bool)
    insect_ok = ~((cat >> 5) & 1).astype(bool)
    Z_masked  = np.where(hydro_ok & insect_ok, Z_raw, np.nan)

    h_labels = [f'{h:.0f} m' for h in h_m]
    Z_cn_df  = pd.DataFrame(Z_masked, index=t_cn, columns=h_labels)
    print(f'        [CN load] {len(Z_cn_df)} steps  |  '
          f'heights {h_m[0]:.0f}–{h_m[-1]:.0f} m AMSL')
    return Z_cn_df, h_labels, h_m


# ══════════════════════════════════════════════════════════════════════════════
# STEP B2 – Load tower ZNC files with Saturatedco mask
# ══════════════════════════════════════════════════════════════════════════════
def load_znc(date_str):
    """
    Unchanged from compare_ze_fwd_tower_znc.py.
    Saturatedco mask applied. Returns arrays for matrix multiplication.
    """
    yyyy, mm, dd = date_str[:4], date_str[4:6], date_str[6:]
    radar_dir = RADAR_BASE / yyyy / mm / dd
    znc_files = sorted(radar_dir.glob('*_tower.znc'))
    if not znc_files:
        raise FileNotFoundError(f'No *_tower.znc files in {radar_dir}')

    t_list, Zg_list, sat_list = [], [], []
    ranges_m = None

    for znc_path in znc_files:
        try:
            f = h5py.File(str(znc_path), 'r')
        except Exception as e:
            print(f'        WARNING: cannot open {znc_path.name}: {e}')
            continue
        t_r = read_f32(f['time']).astype('float64')
        Zg  = read_f32(f['Zg'])
        Sat = read_f32(f['Saturatedco'])
        if ranges_m is None:
            ranges_m = read_f32(f['range'])[:N_TW_GATES].astype('float64')
        f.close()
        if Zg.ndim < 2 or Zg.shape[1] < N_TW_GATES:
            continue
        t_list.append(t_r)
        Zg_list.append(Zg[:, :N_TW_GATES].astype('float64'))
        sat_list.append(Sat[:, :N_TW_GATES].astype('float64'))

    if not t_list:
        raise FileNotFoundError(f'No usable znc data in {radar_dir}')

    t_rad  = np.concatenate(t_list)
    Zg_all = np.concatenate(Zg_list, axis=0)
    Sat_all= np.concatenate(sat_list, axis=0)

    order  = np.argsort(t_rad)
    t_rad  = t_rad[order]
    Zg_all = Zg_all[order]
    Sat_all= Sat_all[order]

    Zg_all     = np.where(Sat_all == 0, Zg_all, np.nan)
    valid_cnts = np.where(np.isfinite(Zg_all) & (Zg_all > 0), 1.0, 0.0)
    Zg_lin_nz  = np.where(valid_cnts > 0, Zg_all, 0.0)

    print(f'        [TW ZNC] {len(znc_files)} files  |  {len(t_rad)} radar steps')
    return t_rad, Zg_lin_nz, valid_cnts, ranges_m


# ══════════════════════════════════════════════════════════════════════════════
# STEP C1 – Align Cloudnet (with fall-time shift AND midpoint fix)
# ══════════════════════════════════════════════════════════════════════════════
def align_cn(fwd, Z_cn_df):
    """
    Match Parsivel steps to Cloudnet snapshots within ±TOL_S seconds.
    Average in linear units, convert back to dBZ.

    CHANGE vs parent: Parsivel midpoint fix  t_p_unix -= 30.0  (NEW)
    UNCHANGED: Cloudnet +30 s fall-time shift still applied.
    """
    # Parsivel midpoint fix: stamp at end of 60 s window → shift to midpoint
    # This makes ±30 s window cover [t−60, t] = the actual integration interval
    t_p_unix = fwd.index.astype('int64').values / 1e9 - 30.0

    t_cn_shifted = Z_cn_df.index + pd.to_timedelta(CN_TIME_SHIFT_S, unit='s')
    t_cn_unix    = t_cn_shifted.astype('int64').values / 1e9

    dt    = np.abs(t_p_unix[:, np.newaxis] - t_cn_unix[np.newaxis, :])
    match = dt <= TOL_S

    Z_linear    = np.where(np.isfinite(Z_cn_df.values),
                            10.0 ** (Z_cn_df.values / 10.0), np.nan)
    Z_linear_nz = np.where(np.isfinite(Z_linear), Z_linear, 0.0)
    valid_cnts  = np.where(np.isfinite(Z_linear), 1.0, 0.0)

    sum_Z = match.astype(float) @ Z_linear_nz
    sum_v = match.astype(float) @ valid_cnts

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_Z   = np.where(sum_v > 0, sum_Z / np.where(sum_v > 0, sum_v, 1.0), np.nan)
        mean_dBZ = np.where(mean_Z > 0, 10.0 * np.log10(mean_Z), np.nan)

    n_cn = match.sum(axis=1)
    cn_aligned = pd.DataFrame(mean_dBZ, index=fwd.index, columns=Z_cn_df.columns)
    cn_aligned['n_cn'] = n_cn

    merged = pd.concat([fwd, cn_aligned], axis=1)
    n_before = len(merged)
    merged   = merged[merged['n_cn'] > 0].copy()
    print(f'        [CN align] {n_before} → {len(merged)} steps '
          f'(dropped {n_before - len(merged)} with no CN match)')
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# STEP C2 – Align Tower (no time shift, midpoint fix)
# ══════════════════════════════════════════════════════════════════════════════
def align_tw(fwd, t_rad_unix, Zg_lin_nz, valid_cnts_rad):
    """
    Match Parsivel steps to tower radar snapshots within ±TOL_S seconds.
    Average in linear units, convert back to dBZ.

    CHANGE vs parent: Parsivel midpoint fix  t_p_unix -= 30.0  (NEW)
    UNCHANGED: no Cloudnet-style time shift (sensors are co-located).
    """
    # Parsivel midpoint fix: stamp at end of 60 s window → shift to midpoint
    t_p_unix = fwd.index.view('int64').astype('float64') / 1e9 - 30.0

    dt    = np.abs(t_p_unix[:, np.newaxis] - t_rad_unix[np.newaxis, :])
    match = (dt <= TOL_S).astype(float)

    sum_Z = match @ Zg_lin_nz      # (N_pars, N_gates)
    sum_v = match @ valid_cnts_rad # (N_pars, N_gates)

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_Z   = np.where(sum_v > 0, sum_Z / np.where(sum_v > 0, sum_v, 1.0), np.nan)
        mean_dBZ = np.where(mean_Z > 0, 10.0 * np.log10(mean_Z), np.nan)

    ze_fwd_all = fwd['Ze_tmm'].values
    rr_all     = fwd['rr'].values

    has_radar  = np.any(sum_v > 0, axis=1)
    valid_step = has_radar & np.isfinite(ze_fwd_all)

    n_drop = int(np.sum(~has_radar & np.isfinite(ze_fwd_all)))
    print(f'        [TW align] dropped {n_drop} steps with finite FWD Ze but no radar match')

    idx = np.where(valid_step)[0]
    return {
        'ze_fwd': ze_fwd_all[idx],
        'ze_rad': mean_dBZ[idx],   # (N, N_gates)
        'rr':     rr_all[idx],
        't_unix': t_p_unix[idx],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Gate geometry helper
# ══════════════════════════════════════════════════════════════════════════════
def gate_heights(ranges_m):
    return ranges_m * np.sin(ELEV_RAD) + SITE_ALT_M


# ══════════════════════════════════════════════════════════════════════════════
# STEP D – Scatter plot helpers
# ══════════════════════════════════════════════════════════════════════════════
def _scatter_one(ax, ze_x, ze_y, rr, label_x, label_y, title, lim=(-10, 50)):
    """Draw a single scatter panel. Returns (N, bias, rmse)."""
    mask = np.isfinite(ze_x) & np.isfinite(ze_y)
    N    = int(mask.sum())
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(label_x, fontsize=7.5)
    ax.set_ylabel(label_y, fontsize=7.5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.plot(lim, lim, 'k--', lw=0.8, label='1:1')
    ax.grid(True, alpha=0.25)

    if N < 2:
        ax.text(0.5, 0.5, f'N={N}\nno data', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='gray')
        return N, np.nan, np.nan

    bias = float(np.mean(ze_y[mask] - ze_x[mask]))
    rmse = float(np.sqrt(np.mean((ze_y[mask] - ze_x[mask])**2)))

    rr_v = rr[mask] if rr is not None else None
    if rr_v is not None:
        sc = ax.scatter(ze_x[mask], ze_y[mask], c=rr_v, cmap='plasma',
                        vmin=0, vmax=max(float(np.nanmax(rr_v)), 1.0),
                        s=10, alpha=0.6, zorder=3)
        plt.colorbar(sc, ax=ax, label='rr [mm/h]', fraction=0.046, pad=0.04)
    else:
        ax.scatter(ze_x[mask], ze_y[mask], color='steelblue',
                   s=10, alpha=0.6, zorder=3)

    ax.plot(lim, [v + bias for v in lim], 'r:', lw=0.8,
            label=f'bias {bias:+.2f} dB')
    ax.text(0.04, 0.96,
            f'N={N}\nbias={bias:+.2f} dB\nRMSE={rmse:.2f} dB',
            transform=ax.transAxes, fontsize=7.5, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75))
    ax.legend(fontsize=7, loc='lower right')
    return N, bias, rmse


def _print_bias_table(label, ze_x_arr, ze_y_arr, gate_labels):
    print(f'\n── Bias & RMSE  [{label}] ──')
    print(f'  {"Gate/Height":>15}  {"N":>5}  {"Bias [dB]":>10}  {"RMSE [dB]":>10}')
    print(f'  {"─"*46}')
    for gl, ze_x, ze_y in zip(gate_labels, ze_x_arr, ze_y_arr):
        mask  = np.isfinite(ze_x) & np.isfinite(ze_y)
        N_v   = int(mask.sum())
        if N_v > 0:
            d = ze_y[mask] - ze_x[mask]
            print(f'  {gl:>15}  {N_v:>5}  {d.mean():>+10.2f}  '
                  f'{np.sqrt((d**2).mean()):>10.2f}')
        else:
            print(f'  {gl:>15}  {"0":>5}  {"—":>10}  {"—":>10}')
    print('─' * 50)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 – Side-by-side scatter (key output)
# ══════════════════════════════════════════════════════════════════════════════
def plot_side_by_side(merged_cn, h_labels, h_m, aligned_tw, ranges_m,
                      label, out_dir):
    """
    2-panel scatter: Cloudnet gate 0 (left) | Tower closest gate (right).
    Same axis limits — biases directly comparable.
    """
    heights_asl = gate_heights(ranges_m)
    out_dir.mkdir(parents=True, exist_ok=True)

    ze_fwd_cn = merged_cn['Ze_tmm'].values
    ze_cn_g0  = merged_cn[h_labels[0]].values
    rr_cn     = merged_cn['rr'].values

    ze_fwd_tw = aligned_tw['ze_fwd']
    ze_tw_gc  = aligned_tw['ze_rad'][:, TOWER_GATE]
    rr_tw     = aligned_tw['rr']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        f'Ze FWD vs Radar — {label}\n'
        f'Left: Cloudnet JOYRAD-35 (141 m separation) | '
        f'Right: MIRA-35 tower.znc (~3 m separation)\n'
        f'No D-filter | No RR threshold | ±{TOL_S:.0f} s window | '
        f'Parsivel midpoint fix applied',
        fontsize=10,
    )

    _scatter_one(
        axes[0], ze_fwd_cn, ze_cn_g0, rr_cn,
        label_x=f'Ze FWD surface [dBZ]\n(Parsivel {SITE_ALT_M:.0f} m ASL, '
                f'T-matrix 35.6 GHz, 293.15 K)',
        label_y=f'Ze Cloudnet [dBZ]\n(JOYRAD-35 vertical, {h_m[0]:.0f} m AMSL, '
                f'category_bits masked)',
        title=f'Cloudnet gate 0  –  {h_labels[0]} AMSL\n'
              f'CN shift +{CN_TIME_SHIFT_S:.0f} s (fall-time correction)',
    )

    _scatter_one(
        axes[1], ze_fwd_tw, ze_tw_gc, rr_tw,
        label_x=f'Ze FWD tower [dBZ]\n(Parsivel 452070 {DISDRO_ALT_M:.0f} m ASL, '
                f'T-matrix 35.5 GHz 19°, 293.15 K)',
        label_y=f'Ze Tower ZNC [dBZ]\n(MIRA-35 19°, {heights_asl[TOWER_GATE]:.0f} m ASL, '
                f'Saturatedco masked)',
        title=f'Tower gate {TOWER_GATE}  –  {ranges_m[TOWER_GATE]:.0f} m slant range  '
              f'{heights_asl[TOWER_GATE]:.0f} m ASL  ★ closest to disdrometer',
    )

    plt.tight_layout()
    tag = label.replace('-', '')
    p = out_dir / f'Ze_side_by_side_{tag}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'        Saved → {p.name}')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 – All Cloudnet gates scatter
# ══════════════════════════════════════════════════════════════════════════════
def plot_cn_gates(merged_cn, h_labels, h_m, label, out_dir):
    """Scatter Ze_FWD vs Ze_Cloudnet for each of the N_CN_BINS gates."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ze_fwd = merged_cn['Ze_tmm'].values
    rr     = merged_cn['rr'].values
    ncols  = 2
    nrows  = int(np.ceil(N_CN_BINS / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.0 * nrows))
    axes_flat = axes.flatten()

    fig.suptitle(
        f'Ze FWD (surface Parsivel) vs Cloudnet JOYRAD-35 — {label}\n'
        f'category_bits mask | CN shift +{CN_TIME_SHIFT_S:.0f} s | '
        f'±{TOL_S:.0f} s window | No D-filter | No RR threshold',
        fontsize=10,
    )

    ze_x_arr, ze_y_arr, gl_arr = [], [], []
    for i, (ax, hl) in enumerate(zip(axes_flat[:N_CN_BINS], h_labels)):
        ze_cn = merged_cn[hl].values
        _scatter_one(
            ax, ze_fwd, ze_cn, rr,
            label_x='Ze FWD surface [dBZ]',
            label_y='Ze Cloudnet [dBZ]',
            title=f'Gate {i}  –  {hl} AMSL',
        )
        ze_x_arr.append(ze_fwd); ze_y_arr.append(ze_cn); gl_arr.append(hl)

    for ax in axes_flat[N_CN_BINS:]:
        ax.set_visible(False)
    plt.tight_layout()
    tag = label.replace('-', '')
    p = out_dir / f'Ze_all_cn_gates_{tag}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'        Saved → {p.name}')
    _print_bias_table(f'Cloudnet {label}', ze_x_arr, ze_y_arr, gl_arr)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 – All Tower gates scatter
# ══════════════════════════════════════════════════════════════════════════════
def plot_tower_gates(aligned_tw, ranges_m, label, out_dir):
    """Scatter Ze_FWD_tower vs Ze_tower for each of the N_TW_GATES gates."""
    out_dir.mkdir(parents=True, exist_ok=True)
    heights_asl = gate_heights(ranges_m)
    ze_fwd = aligned_tw['ze_fwd']
    rr     = aligned_tw['rr']
    ncols  = 5
    nrows  = int(np.ceil(N_TW_GATES / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    axes_flat = axes.flatten()

    fig.suptitle(
        f'Ze FWD (tower Parsivel) vs MIRA-35 tower.znc — {label}\n'
        f'Saturatedco mask | ±{TOL_S:.0f} s window | No D-filter | No RR threshold\n'
        f'Height ASL = range × sin({ELEV_DEG:.0f}°) + {SITE_ALT_M:.0f} m  |  '
        f'Disdrometer at {DISDRO_ALT_M:.0f} m ASL',
        fontsize=10,
    )

    ze_x_arr, ze_y_arr, gl_arr = [], [], []
    for g in range(N_TW_GATES):
        ax  = axes_flat[g]
        ze_rad_g = aligned_tw['ze_rad'][:, g]
        lbl = f'Gate {g} | {ranges_m[g]:.0f} m | {heights_asl[g]:.0f} m ASL'
        if g == TOWER_GATE:
            lbl += '\n★ closest'
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)
                spine.set_edgecolor('firebrick')

        _scatter_one(
            ax, ze_fwd, ze_rad_g, rr,
            label_x='Ze FWD tower [dBZ]',
            label_y='Ze Tower ZNC [dBZ]',
            title=lbl,
        )
        ze_x_arr.append(ze_fwd); ze_y_arr.append(ze_rad_g); gl_arr.append(lbl.split('\n')[0])

    for g in range(N_TW_GATES, len(axes_flat)):
        axes_flat[g].set_visible(False)
    plt.tight_layout()
    tag = label.replace('-', '')
    p = out_dir / f'Ze_all_tower_gates_{tag}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'        Saved → {p.name}')
    _print_bias_table(f'Tower ZNC {label}', ze_x_arr, ze_y_arr, gl_arr)


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS ONE DATE
# ══════════════════════════════════════════════════════════════════════════════
def process_date(date_str):
    """
    Run both comparisons for one date.
    Each comparison is attempted independently — if one sensor is missing the
    other still runs. Side-by-side plot requires both.
    Returns (merged_cn, h_labels, h_m, aligned_tw, ranges_m) — any may be None.
    """
    date_label = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
    print(f'\n{"="*62}')
    print(f'  {date_str}')
    print(f'{"="*62}')

    out_dir = OUT_BASE / date_str

    # ── Cloudnet branch ──────────────────────────────────────────────────────
    merged_cn = h_labels = h_m = None
    try:
        print('STEP A1  FWD surface Parsivel …')
        fwd_cn = run_fwd_cn(date_str)
        print('STEP B1  Load Cloudnet (category_bits mask) …')
        Z_cn_df, h_labels, h_m = load_cloudnet(date_str)
        print(f'STEP C1  Align Cloudnet (±{TOL_S:.0f} s, CN shift +{CN_TIME_SHIFT_S:.0f} s, midpoint fix) …')
        merged_cn = align_cn(fwd_cn, Z_cn_df)
    except FileNotFoundError as e:
        print(f'        SKIP Cloudnet branch: {e}')
    except Exception as e:
        print(f'        ERROR Cloudnet branch: {e}')

    # ── Tower branch ─────────────────────────────────────────────────────────
    aligned_tw = ranges_m = None
    try:
        print('STEP A2  FWD tower Parsivel 452070 …')
        fwd_tw = run_fwd_tw(date_str)
        print('STEP B2  Load tower ZNC (Saturatedco mask) …')
        t_rad, Zg_lin_nz, valid_cnts, ranges_m = load_znc(date_str)
        print(f'STEP C2  Align Tower (±{TOL_S:.0f} s, no time shift, midpoint fix) …')
        aligned_tw = align_tw(fwd_tw, t_rad, Zg_lin_nz, valid_cnts)
    except FileNotFoundError as e:
        print(f'        SKIP Tower branch: {e}')
    except Exception as e:
        print(f'        ERROR Tower branch: {e}')

    # ── Plots ────────────────────────────────────────────────────────────────
    print('STEP D  Plots …')
    if merged_cn is not None:
        plot_cn_gates(merged_cn, h_labels, h_m, date_label, out_dir)

    if aligned_tw is not None and ranges_m is not None:
        plot_tower_gates(aligned_tw, ranges_m, date_label, out_dir)

    if merged_cn is not None and aligned_tw is not None and ranges_m is not None:
        plot_side_by_side(merged_cn, h_labels, h_m, aligned_tw, ranges_m,
                          date_label, out_dir)

    return merged_cn, h_labels, h_m, aligned_tw, ranges_m


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS ONE MONTH
# ══════════════════════════════════════════════════════════════════════════════
def process_month(year_month_str):
    """
    Process all days in a month. Saves per-month plots to monthly/.
    Returns accumulated data dict for use by process_period (or None values if no data).
    """
    yyyy = int(year_month_str[:4])
    mm   = int(year_month_str[4:6])
    n_days = calendar.monthrange(yyyy, mm)[1]
    month_label = f'{yyyy}-{mm:02d}'
    out_dir_monthly = OUT_BASE / 'monthly'

    print(f'\n{"="*62}')
    print(f'  Monthly  {month_label}  ({n_days} days)')
    print(f'{"="*62}')

    all_cn_fwd, all_cn_rr, all_cn_cols, h_labels_ref, h_m_ref = [], [], {}, None, None
    all_cn_heights_list = []  # track heights for each day
    all_tw_fwd, all_tw_rad, all_tw_rr                         = [], [], []
    ranges_m_ref = None

    for d in range(1, n_days + 1):
        date_str = f'{yyyy}{mm:02d}{d:02d}'
        merged_cn, h_labels, h_m, aligned_tw, ranges_m = process_date(date_str)

        if merged_cn is not None:
            all_cn_fwd.append(merged_cn['Ze_tmm'].values)
            all_cn_rr.append(merged_cn['rr'].values)
            all_cn_heights_list.append(set(h_labels))
            if h_labels_ref is None:
                h_labels_ref = h_labels
                h_m_ref      = h_m
                for hl in h_labels:
                    all_cn_cols[hl] = []
            for hl in h_labels_ref:
                if hl in h_labels:
                    all_cn_cols[hl].append(merged_cn[hl].values)

        if aligned_tw is not None:
            all_tw_fwd.append(aligned_tw['ze_fwd'])
            all_tw_rad.append(aligned_tw['ze_rad'])
            all_tw_rr.append(aligned_tw['rr'])
            if ranges_m_ref is None:
                ranges_m_ref = ranges_m

    cn_fwd_all = tw_fwd_all = tw_rad_all = tw_rr_all = cn_rr_all = cn_cols_all = None

    # ── Monthly Cloudnet plots ────────────────────────────────────────────────
    if all_cn_fwd and h_labels_ref is not None:
        cn_fwd_all  = np.concatenate(all_cn_fwd)
        cn_rr_all   = np.concatenate(all_cn_rr)
        # Only keep heights that have data (filter out None/empty entries)
        cn_cols_all = {}
        for hl in h_labels_ref:
            if hl in all_cn_cols and len(all_cn_cols[hl]) > 0:
                cn_cols_all[hl] = np.concatenate(all_cn_cols[hl])

        merged_all = pd.DataFrame({'Ze_tmm': cn_fwd_all, 'rr': cn_rr_all})
        for hl, arr in cn_cols_all.items():
            merged_all[hl] = arr

        plot_cn_gates(merged_all, h_labels_ref, h_m_ref, month_label, out_dir_monthly)

    # ── Monthly Tower plots ───────────────────────────────────────────────────
    if all_tw_fwd and ranges_m_ref is not None:
        tw_fwd_all = np.concatenate(all_tw_fwd)
        tw_rad_all = np.concatenate(all_tw_rad, axis=0)
        tw_rr_all  = np.concatenate(all_tw_rr)

        plot_tower_gates({'ze_fwd': tw_fwd_all, 'ze_rad': tw_rad_all, 'rr': tw_rr_all},
                         ranges_m_ref, month_label, out_dir_monthly)

    # ── Monthly side-by-side ─────────────────────────────────────────────────
    if cn_fwd_all is not None and tw_fwd_all is not None:
        merged_all_cn = pd.DataFrame({'Ze_tmm': cn_fwd_all, 'rr': cn_rr_all})
        for hl, arr in cn_cols_all.items():
            merged_all_cn[hl] = arr

        plot_side_by_side(merged_all_cn, h_labels_ref, h_m_ref,
                          {'ze_fwd': tw_fwd_all, 'ze_rad': tw_rad_all, 'rr': tw_rr_all},
                          ranges_m_ref, month_label, out_dir_monthly)

    print(f'\n  Monthly done. Days with CN data: {len(all_cn_fwd)}  '
          f'| Days with Tower data: {len(all_tw_fwd)}')

    return {
        'cn_fwd':    cn_fwd_all,
        'cn_rr':     cn_rr_all,
        'cn_cols':   cn_cols_all,
        'h_labels':  h_labels_ref,
        'h_m':       h_m_ref,
        'tw_fwd':    tw_fwd_all,
        'tw_rad':    tw_rad_all,
        'tw_rr':     tw_rr_all,
        'ranges_m':  ranges_m_ref,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS MULTIPLE MONTHS (period)
# ══════════════════════════════════════════════════════════════════════════════
def process_period(month_list):
    """
    Run process_month for each YYYYMM in month_list (saves per-month plots),
    then pool all data and save combined plots to  output/.../all/.
    Usage: process_period(['202502','202503',...,'202507'])
    """
    label = f'{month_list[0]}–{month_list[-1]}'
    out_dir_all = OUT_BASE / 'all'

    print(f'\n{"="*62}')
    print(f'  Period  {label}  ({len(month_list)} months)')
    print(f'{"="*62}')

    all_cn_fwd, all_cn_rr, all_cn_cols = [], [], {}
    all_tw_fwd, all_tw_rad, all_tw_rr  = [], [], []
    h_labels_ref = h_m_ref = ranges_m_ref = None

    for ym in month_list:
        data = process_month(ym)

        if data['cn_fwd'] is not None:
            all_cn_fwd.append(data['cn_fwd'])
            all_cn_rr.append(data['cn_rr'])
            if h_labels_ref is None:
                h_labels_ref = data['h_labels']
                h_m_ref      = data['h_m']
                for hl in h_labels_ref:
                    all_cn_cols[hl] = []
            for hl in h_labels_ref:
                all_cn_cols[hl].append(data['cn_cols'][hl])

        if data['tw_fwd'] is not None:
            all_tw_fwd.append(data['tw_fwd'])
            all_tw_rad.append(data['tw_rad'])
            all_tw_rr.append(data['tw_rr'])
            if ranges_m_ref is None:
                ranges_m_ref = data['ranges_m']

    print(f'\n{"="*62}')
    print(f'  Period plots  {label}')
    print(f'{"="*62}')

    if all_cn_fwd and h_labels_ref is not None:
        cn_fwd_all  = np.concatenate(all_cn_fwd)
        cn_rr_all   = np.concatenate(all_cn_rr)
        cn_cols_all = {hl: np.concatenate(v) for hl, v in all_cn_cols.items()}
        merged_all_cn = pd.DataFrame({'Ze_tmm': cn_fwd_all, 'rr': cn_rr_all})
        for hl, arr in cn_cols_all.items():
            merged_all_cn[hl] = arr
        plot_cn_gates(merged_all_cn, h_labels_ref, h_m_ref, label, out_dir_all)

    if all_tw_fwd and ranges_m_ref is not None:
        tw_fwd_all = np.concatenate(all_tw_fwd)
        tw_rad_all = np.concatenate(all_tw_rad, axis=0)
        tw_rr_all  = np.concatenate(all_tw_rr)
        plot_tower_gates({'ze_fwd': tw_fwd_all, 'ze_rad': tw_rad_all, 'rr': tw_rr_all},
                         ranges_m_ref, label, out_dir_all)

    if all_cn_fwd and all_tw_fwd:
        plot_side_by_side(merged_all_cn, h_labels_ref, h_m_ref,
                          {'ze_fwd': tw_fwd_all, 'ze_rad': tw_rad_all, 'rr': tw_rr_all},
                          ranges_m_ref, label, out_dir_all)

    print(f'\n  Period done. Months with CN data: {len(all_cn_fwd)}  '
          f'| Months with Tower data: {len(all_tw_fwd)}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    if len(sys.argv) < 2:
        process_date('20250606')
        sys.exit(0)

    args = sys.argv[1:]
    dates  = [a for a in args if len(a) == 8 and a.isdigit()]
    months = [a for a in args if len(a) == 6 and a.isdigit()]
    bad    = [a for a in args if a not in dates + months]

    if bad:
        print(f'Unknown argument(s): {bad}  (expected YYYYMMDD or YYYYMM)')
        sys.exit(1)

    for d in dates:
        process_date(d)

    if len(months) == 1:
        process_month(months[0])
    elif len(months) > 1:
        process_period(sorted(months))
