# -*- coding: utf-8 -*-
"""
test_matching.py
================
Tests the Parsivel-as-master matching strategy on 2025-06-06.

Uses pre-computed FWD CSV (tower Parsivel) + live radar ZNC files.
No FWD re-simulation needed — just testing the temporal matching logic.

Run with:
  PYTHONPATH=/tmp/testpkgs python3.12 test_matching.py
"""

import sys
sys.path.insert(0, '/tmp/testpkgs')

import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import h5py.h5s as h5s
import h5py.h5t as h5t

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATE_STR    = '20250606'
GATE        = 6       # gate closest to Tower Parsivel
TOL_S       = 30.0    # ±30 s matching window
RR_MIN      = 0.1     # mm/h

FWD_CSV     = Path('output/comparison/fwd_sim_tower_parsivel_20250606.csv')
RADAR_DIR   = Path('/data/obs/site/jue/joyrad35/2025/06/06')
CLOUDNET_CSV= Path('output/comparison/cloudnet_vs_fwd_masked/20250607/all_points_20250607.csv')


def read_f32(ds):
    shape = ds.shape
    buf = np.zeros(shape, dtype='float32')
    fsp = ds.id.get_space()
    msp = h5s.create_simple(shape)
    ds.id.read(msp, fsp, buf, h5t.IEEE_F32LE)
    return buf


# ── STEP 1: Load pre-computed FWD CSV ─────────────────────────────────────────
print('=' * 60)
print(f'TOWER COMPARISON — {DATE_STR}')
print(f'Parsivel-as-master | gate {GATE} | ±{TOL_S:.0f} s window')
print('=' * 60)

fwd = pd.read_csv(FWD_CSV, index_col=0, parse_dates=True)
fwd.index = pd.to_datetime(fwd.index, utc=True)
rainy_mask = fwd['rr'] >= RR_MIN
fwd.loc[~rainy_mask, 'Ze_tmm'] = np.nan

n_total = len(fwd)
n_rainy = int(rainy_mask.sum())
print(f'\nFWD CSV: {n_total} Parsivel minutes total, {n_rainy} rainy (rr >= {RR_MIN} mm/h)')
print(f'Rainy periods:')
# show contiguous rainy blocks
in_rain = False
for t, rr in zip(fwd.index, fwd['rr']):
    if rr >= RR_MIN and not in_rain:
        print(f'  Rain starts: {t.strftime("%H:%M")} UTC')
        in_rain = True
    elif rr < RR_MIN and in_rain:
        print(f'  Rain ends:   {t.strftime("%H:%M")} UTC')
        in_rain = False
if in_rain:
    print(f'  Rain ends:   (end of day)')


# ── STEP 2: Load radar ZNC files, extract gate 6 ──────────────────────────────
print(f'\nLoading radar ZNC files from {RADAR_DIR}...')
znc_files = sorted(RADAR_DIR.glob('*_tower.znc'))
print(f'  Found {len(znc_files)} burst files')

t_list, Zg_list, Sat_list = [], [], []
for znc_path in znc_files:
    try:
        f = h5py.File(str(znc_path), 'r')
    except Exception as e:
        continue
    t_r  = read_f32(f['time']).astype('float64')
    Zg   = read_f32(f['Zg'])
    Sat  = read_f32(f['Saturatedco'])
    f.close()
    if Zg.ndim < 2 or Zg.shape[1] <= GATE:
        continue
    t_list.append(t_r)
    Zg_list.append(Zg[:, GATE].astype('float64'))
    Sat_list.append(Sat[:, GATE].astype('float64'))

t_rad = np.concatenate(t_list)
Zg_g  = np.concatenate(Zg_list)
Sat_g = np.concatenate(Sat_list)

order = np.argsort(t_rad)
t_rad = t_rad[order]
Zg_g  = Zg_g[order]
Sat_g = Sat_g[order]

# saturation mask
Zg_g = np.where(Sat_g == 0, Zg_g, np.nan)

# identify bursts (gap > 10 s between radar snapshots)
gaps         = np.diff(t_rad)
burst_starts = np.where(np.concatenate([[True], gaps > 10]))[0]
burst_ends   = np.append(burst_starts[1:] - 1, len(t_rad) - 1)

print(f'  {len(t_rad)} radar snapshots across {len(burst_starts)} bursts')
print(f'  Burst durations: min={np.min(t_rad[burst_ends]-t_rad[burst_starts]):.0f}s  '
      f'max={np.max(t_rad[burst_ends]-t_rad[burst_starts]):.0f}s  '
      f'mean={np.mean(t_rad[burst_ends]-t_rad[burst_starts]):.0f}s')


# ── STEP 3: Parsivel-as-master matching ───────────────────────────────────────
print(f'\nMatching (Parsivel minute as master, ±{TOL_S:.0f} s):')

t_p_unix = np.array(fwd.index.view('int64'), dtype='float64') / 1e9

matched_fwd, matched_rad, matched_rr, matched_t = [], [], [], []
n_rainy_no_radar = 0

for i in range(len(fwd)):
    ze_fwd = fwd['Ze_tmm'].iloc[i]
    rr     = fwd['rr'].iloc[i]

    if np.isnan(ze_fwd):
        continue  # not rainy

    tp = t_p_unix[i]
    in_window = np.abs(t_rad - tp) <= TOL_S
    Zg_win    = Zg_g[in_window]
    valid     = Zg_win[np.isfinite(Zg_win) & (Zg_win > 0)]

    if len(valid) == 0:
        n_rainy_no_radar += 1
        continue

    ze_rad_dBZ = 10.0 * np.log10(np.mean(valid))
    matched_fwd.append(ze_fwd)
    matched_rad.append(ze_rad_dBZ)
    matched_rr.append(rr)
    matched_t.append(fwd.index[i])

n_matched = len(matched_fwd)
print(f'  Rainy Parsivel minutes with radar match:    {n_matched}')
print(f'  Rainy Parsivel minutes WITHOUT radar match: {n_rainy_no_radar}')
print(f'  Match rate: {100*n_matched/(n_matched+n_rainy_no_radar):.1f}%')

if n_matched >= 2:
    diff  = np.array(matched_rad) - np.array(matched_fwd)
    bias  = np.mean(diff)
    rmse  = np.sqrt(np.mean(diff**2))

    print(f'\nResults at gate {GATE}:')
    print(f'  N pairs:                  {n_matched}')
    print(f'  Bias (Ze_radar - Ze_fwd): {bias:+.2f} dB')
    print(f'  RMSE:                     {rmse:.2f} dB')

    print(f'\nMatched pairs detail:')
    print(f'  {"Time(UTC)":>10}  {"Ze_fwd":>8}  {"Ze_radar":>9}  {"RR":>7}  {"diff":>6}  {"n_rad_snaps":>11}')
    for i, (t, zf, zr, rr) in enumerate(zip(matched_t, matched_fwd,
                                             matched_rad, matched_rr)):
        tp = t.value / 1e9
        n_snaps = int(np.sum(np.abs(t_rad - tp) <= TOL_S))
        print(f'  {t.strftime("%H:%M"):>10}  {zf:>8.2f}  {zr:>9.2f}  '
              f'{rr:>7.2f}  {zr-zf:>+6.2f}  {n_snaps:>11}')


# ── STEP 4: CloudNet reference ────────────────────────────────────────────────
print()
print('=' * 60)
print('CLOUDNET COMPARISON (reference, 20250607, lowest gate bin=0)')
print('=' * 60)

import csv, math
with open(CLOUDNET_CSV) as f:
    rows = [r for r in csv.DictReader(f) if r['bin'] == '0']

n_cn    = len(rows)
deltas  = [float(r['delta']) for r in rows]  # Ze_cloudnet - Ze_fwd
bias_cn = sum(deltas) / n_cn
rmse_cn = math.sqrt(sum(d**2 for d in deltas) / n_cn)

print(f'N pairs (lowest gate):    {n_cn}')
print(f'Bias (Ze_radar - Ze_fwd): {bias_cn:+.2f} dB')
print(f'RMSE:                     {rmse_cn:.2f} dB')
print()
print('Summary:')
print(f'  Tower   bias = {bias:+.2f} dB  (N={n_matched}, gate {GATE}, 2025-06-06)')
print(f'  CloudNet bias = {bias_cn:+.2f} dB  (N={n_cn}, lowest gate, 2025-06-07)')
print(f'  => Vertical separation estimate = {bias_cn - bias:+.2f} dB  (CloudNet - Tower)')
print('     (rough: only 1 date each, not yet pooled)')
