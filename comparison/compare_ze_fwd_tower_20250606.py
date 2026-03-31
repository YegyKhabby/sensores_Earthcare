# -*- coding: utf-8 -*-
"""
Compare FWD-simulated Ze (tower Parsivel 304640) vs MIRA-35 tower.znc Zg
for 2025-06-06, gates 0–9 (height above radar ≈ 47–152 m).

Strategy
--------
The tower.znc files have ~0.97 s time resolution but only cover ~2 min every
30 min (48 files × ~133 steps = ~6400 radar snapshots for the whole day).
The Parsivel log has 1-min averages.

For each radar time stamp t_r:
  - Find the parsivel step whose timestamp is closest to t_r.
  - Accept the pair only if |t_r − t_pars| < MATCH_TOL_S (90 s).
  - Apply Saturatedco mask: set Zg = NaN where Saturatedco = 1.
  - Accumulate (Ze_fwd_dBZ, Zg_dBZ, rr) per gate.

One output image per gate (scatter + coloured by rain rate) is saved to
  output/comparison/tower_gate_comparison/gate_NN_<height>m_agl.png

Summary CSV:
  output/comparison/tower_gate_comparison/gate_stats.csv
  columns: gate, height_agl_m, N, bias_dB, rmse_dB

Inputs
------
  FWD CSV : output/comparison/fwd_sim_tower_parsivel_20250606.csv
  Radar   : /data/obs/site/jue/joyrad35/2025/06/06/*_tower.znc
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import h5py.h5s as h5s
import h5py.h5t as h5t

# ── CONFIG ─────────────────────────────────────────────────────────────────────
FWD_CSV      = Path('/work/yegy_project/Feb2026_new/output/comparison/'
                    'fwd_sim_tower_parsivel_20250606.csv')
RADAR_DIR    = Path('/data/obs/site/jue/joyrad35/2025/06/06')
OUT_DIR      = Path('/work/yegy_project/Feb2026_new/output/comparison/'
                    'tower_gate_comparison')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MATCH_TOL_S  = 60.0    # max |t_radar − t_parsivel| to accept a pair [s]
N_GATES      = 10      # compare first N gates
RR_MIN       = 0.1     # minimum rain rate to include in plots [mm/h]
SITE_ALT_M   = 111.0   # JOYCE site altitude AMSL [m]

# ── low-level HDF5 reader ──────────────────────────────────────────────────────
def read_f32(ds):
    """Read HDF5 dataset as float32 (bypasses non-standard type issue)."""
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

# ── STEP 1: load FWD simulation results ───────────────────────────────────────
print("STEP 1  Loading FWD simulation CSV …")
fwd = pd.read_csv(FWD_CSV, index_col=0, parse_dates=True)
fwd.index = fwd.index.tz_convert('UTC')
print(f"        {len(fwd)} parsivel steps  "
      f"({fwd.index[0].strftime('%H:%M')} → {fwd.index[-1].strftime('%H:%M')} UTC)")
print(f"        Columns: {list(fwd.columns)}")

# parsivel unix timestamps for fast nearest-neighbour lookup
t_pars_unix = fwd.index.astype('int64') / 1e9   # float seconds

# ── STEP 2: load all tower.znc files ──────────────────────────────────────────
print("\nSTEP 2  Loading tower.znc files …")
znc_files = sorted(RADAR_DIR.glob('*_tower.znc'))
print(f"        {len(znc_files)} files found")

# Per-gate accumulators: list of (Ze_fwd, Zg_radar, rr) tuples
accum = {g: {'Ze_fwd': [], 'Zg_rad': [], 'rr': []} for g in range(N_GATES)}

n_files_loaded = 0
n_pairs_total  = 0

for znc_path in znc_files:
    try:
        f = h5py.File(str(znc_path), 'r')
    except Exception as e:
        print(f"        WARNING: cannot open {znc_path.name}: {e}")
        continue

    # radar timestamps
    t_rad = read_f32(f['time']).astype('float64')   # unix seconds

    # Zg (linear mm⁶/m³) and Saturatedco flag  → shape (time, range)
    Zg  = read_f32(f['Zg'])           # (T, R)
    Sat = read_f32(f['Saturatedco'])  # (T, R)  0=ok, 1=saturated

    # range array: vertical height above ground [m]  (ZNC stores height, not slant range)
    r_slant = read_f32(f['range'])    # (R,)
    f.close()

    # apply saturation mask: set saturated samples to NaN
    Zg = np.where(Sat == 0, Zg, np.nan)

    # convert to dBZ: 10 * log10(Zg), NaN where Zg ≤ 0
    with np.errstate(divide='ignore', invalid='ignore'):
        Zg_dBZ = np.where(Zg > 0, 10.0 * np.log10(Zg), np.nan)

    # match each radar time step to nearest parsivel step
    for i_t, t_r in enumerate(t_rad):
        idx_pars = int(np.argmin(np.abs(t_pars_unix - t_r)))
        dt       = abs(t_pars_unix[idx_pars] - t_r)
        if dt > MATCH_TOL_S:
            continue

        ze_fwd = fwd['Ze_tmm'].iloc[idx_pars]
        rr_val = fwd['rr'].iloc[idx_pars]

        # require rain
        if np.isnan(ze_fwd) or np.isnan(rr_val) or rr_val < RR_MIN:
            continue

        for g in range(N_GATES):
            zg_val = Zg_dBZ[i_t, g]
            if np.isnan(zg_val):
                continue
            accum[g]['Ze_fwd'].append(ze_fwd)
            accum[g]['Zg_rad'].append(zg_val)
            accum[g]['rr'].append(rr_val)

        n_pairs_total += 1

    n_files_loaded += 1

print(f"        {n_files_loaded} files loaded")
print(f"        {n_pairs_total} radar time steps matched to rainy parsivel steps")

# ── STEP 3: compute gate heights ──────────────────────────────────────────────
# re-open first file just for range array
f0 = h5py.File(str(znc_files[0]), 'r')
r_slant = read_f32(f0['range'])
f0.close()
# ZNC range = vertical height above ground (JOYCE site, 111 m AMSL).
# No trig conversion needed — METEK stores height directly.
# AMSL = range + SITE_ALT_M  (= CloudNet height grid)
heights_above_ground = r_slant[:N_GATES]

# ── STEP 4: one plot per gate ──────────────────────────────────────────────────
print(f"\nSTEP 3  Producing {N_GATES} scatter plots (one per gate) …")
print(f"        Output dir: {OUT_DIR}")

stats_rows = []

for g in range(N_GATES):
    h_agl   = heights_above_ground[g]
    Ze_fwd  = np.array(accum[g]['Ze_fwd'])
    Zg_rad  = np.array(accum[g]['Zg_rad'])
    rr_arr  = np.array(accum[g]['rr'])
    N       = len(Ze_fwd)

    h_str   = f"{h_agl:.0f}"   # "47", "59", ...
    gate_lbl = f"gate {g:02d}  ({h_str} m above ground,  AMSL {h_agl + SITE_ALT_M:.0f} m)"
    print(f"        Gate {g:2d}: N={N:4d}  h={h_agl:.1f} m above ground", end='')

    if N < 2:
        print("  → too few points, skipping")
        stats_rows.append({'gate': g, 'height_above_ground_m': h_agl,
                           'N': N, 'bias_dB': np.nan, 'rmse_dB': np.nan})
        continue

    bias = float(np.mean(Ze_fwd - Zg_rad))
    rmse = float(np.sqrt(np.mean((Ze_fwd - Zg_rad)**2)))
    print(f"  bias={bias:+.2f} dB  RMSE={rmse:.2f} dB")
    stats_rows.append({'gate': g, 'height_above_ground_m': h_agl,
                       'N': N, 'bias_dB': bias, 'rmse_dB': rmse})

    # ── scatter plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    sc = ax.scatter(Ze_fwd, Zg_rad,
                    c=rr_arr, cmap='plasma',
                    vmin=0.1, vmax=max(rr_arr.max(), 1.0),
                    s=6, alpha=0.6, zorder=3)

    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label('Rain rate [mm h⁻¹]', fontsize=9)

    # 1:1 line
    all_vals = np.concatenate([Ze_fwd, Zg_rad])
    vmin_plot = np.nanpercentile(all_vals, 1) - 2
    vmax_plot = np.nanpercentile(all_vals, 99) + 2
    lim = [vmin_plot, vmax_plot]
    ax.plot(lim, lim, 'k--', lw=1.0, label='1:1 line', zorder=2)

    # bias line
    ax.plot(lim, [l + bias for l in lim], color='firebrick', lw=1.0,
            ls=':', label=f'bias {bias:+.2f} dB', zorder=2)

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Ze FWD simulated [dBZ]\n(tower Parsivel 304640, T-matrix 35.5 GHz 19°)',
                  fontsize=9)
    ax.set_ylabel('Zg MIRA-35 tower.znc [dBZ]\n(linear Zg → 10·log10, Saturatedco masked)',
                  fontsize=9)

    title_lines = (
        f'MIRA-35 vs tower Parsivel FWD  –  Jülich 2025-06-06\n'
        f'{gate_lbl}\n'
        f'N = {N}  |  bias = {bias:+.2f} dB  |  RMSE = {rmse:.2f} dB\n'
        f'(pairs: |Δt| < {MATCH_TOL_S:.0f} s, rr ≥ {RR_MIN} mm h⁻¹)'
    )
    ax.set_title(title_lines, fontsize=9, pad=8)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.25)

    fname = f'gate_{g:02d}_{h_str}m_above_ground.png'
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)

# ── STEP 5: summary CSV ────────────────────────────────────────────────────────
stats_df = pd.DataFrame(stats_rows)
stats_csv = OUT_DIR / 'gate_stats.csv'
stats_df.to_csv(stats_csv, index=False, float_format='%.3f')
print(f"\nSTEP 4  Summary stats saved to:\n        {stats_csv}")
print(stats_df.to_string(index=False))
