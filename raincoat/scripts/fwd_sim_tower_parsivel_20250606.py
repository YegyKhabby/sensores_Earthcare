# -*- coding: utf-8 -*-
"""
Forward simulation of Ze (35.5 GHz) from the tower Parsivel (serial 304640)
for 2025-06-06, using the pre-computed T-matrix scattering table at 288.15 K
and 19° elevation angle (matching the MIRA-35 tower.znc beam geometry).

What this script does, step by step
=====================================

INPUT FILES
-----------
  Parsivel log  : /data/obs/site/jue/parsivel_304640/l1/2025/06/06/parsivel_jue_20250606.log
  Scatter table : raincoat/samplefiles/scattering/288.15_35.5GHz_elev19deg.csv

Why 19° and 288.15 K?
  The MIRA-35 tower.znc files use elv=19° (instrument azi=65°, true azimuth≈288° WNW).
  At 19° elevation the radar beam intersects the tower at ~47–320 m slant range
  (vertical gate spacing ≈ 11.7 m AGL).  The T-matrix geometry uses θ_zenith = 71°
  (= 90° − 19°), which affects the backscattering cross-section of oblate raindrops.
  The Parsivel itself is orientation-agnostic; only the scattering table cares about
  the radar's viewing angle.
  288.15 K = 15 °C — typical surface temperature at JOYCE in early June.

STEP 1  – Read tower Parsivel log file
    File has one header line + 1440 data lines (1-min cadence, latin-1 encoding).
    Field layout per line (1106 fields total):
        col  0        datetime_utc   (e.g. 2025-06-06 00:01:20)
        col  1        rain rate      [mm h⁻¹]
        cols 2–17     other metadata (accum, wawa, Z, vis, interval, amplitude,
                      n_particles, T_sensor, serial, firmware, heating, voltage,
                      status, station, rain_abs, error)
        cols 18–49    N[0:32]        velocity spectrum   (−9.999 = empty bin)
        cols 50–81    v[0:32]        diameter spectrum   (−9.999 = empty bin)
        cols 82–1105  M[32×32]       raw particle counts (v × D, row-major)

STEP 1b – Reconstruct N(D,t) from M matrix
    Same formula as the JOYCE rooftop Parsivel:
        N(D_i,t) [m⁻³ mm⁻¹] = Σ_j  M[j,i,t] / (v_j · A · Δt · ΔD_i)
    Constants:  A = 54 cm² = 54×10⁻⁴ m²,  Δt = 60 s
    Bin arrays are the standard OTT Parsivel2 values (identical for all units).

STEP 2  – Build standard parsivel bin edges (pars_class)

STEP 3  – Load T-matrix scattering table (288.15 K, 35.5 GHz, 19° elevation)

STEP 4  – Integration constant  λ⁴ / (π⁵ · |K|²)

STEP 5-8 – Run FWD_sim for all 1440 time steps
    Outputs per time step:  Ze_tmm [dBZ], Ze_ray [dBZ], A [dB km⁻¹], Ze_inst [dBZ]

STEP 9  – Mask non-rain periods (rr < 0.1 mm h⁻¹)

STEP 10 – Save CSV and plots
    Output CSV: output/comparison/fwd_sim_tower_parsivel_20250606.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
LOG_FILE   = Path('/data/obs/site/jue/parsivel_304640/l1/2025/06/06/'
                  'parsivel_jue_20250606.log')
SCAT_TABLE = Path('/work/yegy_project/Feb2026_new/raincoat/samplefiles/'
                  'scattering/288.15_35.5GHz_elev19deg.csv')
OUT_DIR    = Path('/work/yegy_project/Feb2026_new/output/comparison')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── raincoat imports ──────────────────────────────────────────────────────────
from raincoat.FWD_sim import FWD_sim
import raincoat.disdrometer.pars_class as pc

# =============================================================================
# STEP 1 – Read tower Parsivel log file
# =============================================================================
print("STEP 1  Reading tower Parsivel log file …")
print(f"        {LOG_FILE}")

# Standard OTT Parsivel2 bin arrays (identical for all units; sourced from
# the JOYCE reference NC: sups_joy_dm00_l1_any_v00_20250606.nc)
vclasses = np.array([
    0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
    1.10, 1.30, 1.50, 1.70, 1.90, 2.20, 2.60, 3.00, 3.40, 3.80,
    4.40, 5.20, 6.00, 6.80, 7.60, 8.80, 10.4, 12.0, 13.6, 15.2,
    17.6, 20.8
])   # m s⁻¹

dclasses = np.array([
    0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187,
    1.375, 1.625, 1.875, 2.125, 2.375, 2.750, 3.250, 3.750, 4.250, 4.750,
    5.500, 6.500, 7.500, 8.500, 9.500, 11.00, 13.00, 15.00, 17.00, 19.00,
    21.50, 24.50
])   # mm

dwidth = np.array([
    0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
    0.250, 0.250, 0.250, 0.250, 0.250,
    0.500, 0.500, 0.500, 0.500, 0.500,
    1.000, 1.000, 1.000, 1.000, 1.000,
    2.000, 2.000, 2.000, 2.000, 2.000,
    3.000, 3.000
])   # mm

# Read log file (latin-1: contains ° symbol in header field names)
with open(LOG_FILE, 'r', encoding='latin-1') as fh:
    header_line = fh.readline()          # skip the field-name header row
    raw_lines   = fh.readlines()

print(f"        {len(raw_lines)} data records found")

# Containers
timestamps = []
rr_list    = []
M_list     = []   # each entry: (32, 32) array
Ze_inst_list = []

n_skipped = 0
for line in raw_lines:
    parts = line.rstrip('\n').split(';')
    if len(parts) < 1106:
        n_skipped += 1
        continue

    # timestamp (col 0)  e.g. "2025-06-06 00:01:20"
    ts = pd.Timestamp(parts[0].strip(), tz='UTC')
    timestamps.append(ts)

    # rain rate (col 1)
    try:
        rr_val = float(parts[1])
    except ValueError:
        rr_val = np.nan
    rr_list.append(rr_val)

    # instrument Ze (col 4, field "Reflectivity [dBZ]")
    try:
        ze_val = float(parts[4])
    except ValueError:
        ze_val = np.nan
    Ze_inst_list.append(ze_val)

    # M matrix (cols 82–1105): 1024 integer counts, v-major order
    m_flat = np.array(parts[82:1106], dtype=float)   # (1024,)
    M_list.append(m_flat.reshape(32, 32))             # (v=32, D=32)

if n_skipped:
    print(f"        WARNING: {n_skipped} lines had fewer than 1106 fields – skipped")

time_pd  = pd.DatetimeIndex(timestamps)
rr_arr   = np.array(rr_list, dtype=float)            # (N,)
Ze_inst  = np.array(Ze_inst_list, dtype=float)       # (N,)
M        = np.stack(M_list, axis=2)                  # (32v, 32D, N)

print(f"        {len(time_pd)} records parsed  "
      f"({time_pd[0].strftime('%Y-%m-%d %H:%M')} → "
      f"{time_pd[-1].strftime('%H:%M')} UTC)")
print(f"        M shape: {M.shape}  (vclasses × dclasses × time)")
n_rain_raw = int(np.sum(rr_arr > 0.001))
print(f"        Rainy records (rr > 0.001 mm h⁻¹): {n_rain_raw}")

# =============================================================================
# STEP 1b – Reconstruct N(D,t) from M matrix
# =============================================================================
print("\nSTEP 1b Reconstructing N(D,t) from M matrix …")
# M axis order: M[v_j, D_i, t]  (confirmed by header M_0_0, M_0_1 layout)
# N(D_i,t) [m⁻³ mm⁻¹] = Σ_j  M[j,i,t] / (v_j · A_sensor · Δt · ΔD_i)
A_SENSOR = 54e-4   # m²  (OTT Parsivel2 effective sampling area)
DT       = 60.0    # s   (1-min records)

# Avoid division by zero for v=0 bins
v_safe = np.where(vclasses > 0, vclasses, np.inf)   # (32,)

# Weighted sum over velocity axis → (32_D, N_t)
M_vsum = (M / v_safe[:, np.newaxis, np.newaxis]).sum(axis=0)

# Divide by (A_sensor · Δt · ΔD_i) → N in m⁻³ mm⁻¹  shape (32_D, N_t)
N_from_M = M_vsum / (A_SENSOR * DT * dwidth[:, np.newaxis])

# log10(N); set zero/negative to NaN
with np.errstate(divide='ignore', invalid='ignore'):
    log10_N = np.where(N_from_M > 0, np.log10(N_from_M), np.nan)

n_nonzero = int(np.sum(N_from_M > 0))
print(f"        log10_N shape: {log10_N.shape}  (dclasses × time)")
print(f"        Non-zero N entries: {n_nonzero}  "
      f"({n_nonzero / log10_N.size * 100:.1f}% of all bins)")

# =============================================================================
# STEP 2 – Build standard parsivel bin edges
# =============================================================================
print("\nSTEP 2  Building parsivel bin edges via pars_class() …")
pclass, bin_edges = pc.pars_class()
print(f"        bin_edges shape: {bin_edges.shape}  (33 edges for 32 bins)")
print(f"        First 5 edges [mm]: {bin_edges[:5,0]}")
print(f"        Last  5 edges [mm]: {bin_edges[-5:,0]}")

# =============================================================================
# STEP 3 – Show scattering table header
# =============================================================================
print(f"\nSTEP 3  Reading scattering table header from:\n"
      f"        {SCAT_TABLE.name}")
with open(SCAT_TABLE, 'r') as fh:
    header = eval(fh.readline())
print(f"        Frequency   : {header['frequency']} GHz")
print(f"        Wavelength  : {header['wl']:.4f} mm")
print(f"        |K|²        : {header['K2']:.4f}")
print(f"        Elevation   : {header['elevation']}°  (MIRA-35 tower beam; θ_zenith = 71°)")
print(f"        Canting σ   : {header['canting']}°")
print(f"        Shape func  : {header['aspect_ratio_func']}")

# =============================================================================
# STEP 4 – Integration constant
# =============================================================================
wl = header['wl']
K2 = header['K2']
int_const = wl**4 / (np.pi**5 * K2)
print(f"\nSTEP 4  int_const = λ⁴ / (π⁵·|K|²)"
      f" = {wl:.4f}⁴ / (π⁵·{K2:.4f}) = {int_const:.4e}  [mm⁴]")

# =============================================================================
# STEP 5-8 – Run FWD_sim for all time steps
# =============================================================================
print(f"\nSTEP 5-8  Running FWD_sim for {len(time_pd)} time steps …")
print(f"          Scattering table: {SCAT_TABLE.name}")

fwd = FWD_sim(str(SCAT_TABLE), time_pd, log10_N, bin_edges)
print(f"          Done.  Output columns: {list(fwd.columns)}")

# =============================================================================
# STEP 9 – Mask non-rain periods
# =============================================================================
print("\nSTEP 9  Masking non-rain time steps (rr < 0.1 mm h⁻¹) …")
rr_series = pd.Series(rr_arr, index=time_pd)
rain_mask = rr_series >= 0.1

n_rain = int(rain_mask.sum())
print(f"        {n_rain} rainy steps  ({n_rain/len(time_pd)*100:.1f}% of day)")

Ze_tmm_rain = fwd['Ze_tmm'].where(rain_mask)
Ze_ray_rain  = fwd['Ze_ray'].where(rain_mask)

# Instrument Ze: flag bad values (< −99)
Ze_inst_s    = pd.Series(np.where(Ze_inst > -99, Ze_inst, np.nan), index=time_pd)
Ze_inst_rain = Ze_inst_s.where(rain_mask)

if n_rain > 0:
    sample = fwd[rain_mask].dropna().head(8)
    print("\n        Sample forward-simulated Ze (rainy steps only):")
    print("        " + sample[['Ze_tmm','Ze_ray','A']].to_string().replace('\n','\n        '))

# =============================================================================
# STEP 10 – Save CSV and plots
# =============================================================================
out_csv = OUT_DIR / 'fwd_sim_tower_parsivel_20250606.csv'
fwd['Ze_inst'] = Ze_inst_s
fwd['rr']      = rr_series.values
fwd.to_csv(out_csv)
print(f"\nSTEP 10 Saved results to:\n        {out_csv}")

# --- Plot 1: full-day time series ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Tower Parsivel (304640) Forward Simulation – Jülich 2025-06-06\n'
             'Scattering table: 288.15 K, 35.5 GHz, 19° elevation  '
             '(T-matrix, Thurai 2007)\n'
             'N(D) reconstructed from raw M matrix (v×D counts)',
             fontsize=11)

ax0, ax1, ax2 = axes

ax0.plot(fwd.index, Ze_inst_rain,  color='steelblue',  lw=1.2,
         label='Parsivel Ze (instrument, D⁶)')
ax0.plot(fwd.index, Ze_ray_rain,   color='orange',     lw=1.0, ls='--',
         label='Ze_ray (Rayleigh D⁶ from PSD)')
ax0.plot(fwd.index, Ze_tmm_rain,   color='crimson',    lw=1.2,
         label='Ze_tmm (T-matrix, 35.5 GHz, 19° elev)')
ax0.set_ylabel('Ze [dBZ]')
ax0.legend(fontsize=8, loc='upper right')
ax0.set_ylim(-10, 40)
ax0.grid(True, alpha=0.3)
ax0.set_title('Reflectivity Ze')

ax1.plot(fwd.index, fwd['A'].where(rain_mask), color='purple', lw=1.0)
ax1.set_ylabel('A [dB km⁻¹]')
ax1.set_title('One-way attenuation at 35.5 GHz')
ax1.grid(True, alpha=0.3)

ax2.plot(fwd.index, rr_series, color='teal', lw=1.0)
ax2.set_ylabel('Rain rate [mm h⁻¹]')
ax2.set_title('Rain rate (from tower Parsivel 304640)')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax2.set_xlabel('Time UTC [2025-06-06]')

plt.tight_layout()
out_ts = OUT_DIR / 'fwd_sim_tower_timeseries_20250606.png'
plt.savefig(out_ts, dpi=150)
print(f"        Time-series plot saved to:\n        {out_ts}")

# --- Plot 2: scatter Ze_inst vs Ze_tmm (rainy steps only) ---
mask_both = rain_mask & fwd['Ze_tmm'].notna() & ~np.isnan(Ze_inst_s)
if mask_both.sum() > 0:
    fig2, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(Ze_inst_s[mask_both], fwd['Ze_tmm'][mask_both],
                    c=rr_series[mask_both], cmap='plasma',
                    s=8, alpha=0.7, label='Ze_tmm')
    ax.scatter(Ze_inst_s[mask_both], Ze_ray_rain[mask_both],
               s=4, alpha=0.3, color='orange', label='Ze_ray (Rayleigh)')
    lim = [-5, 45]
    ax.plot(lim, lim, 'k--', lw=1, label='1:1 line')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Parsivel Ze instrument [dBZ]')
    ax.set_ylabel('FWD simulated Ze [dBZ]')
    ax.set_title('Tower Parsivel (304640): instrument Ze vs forward-simulated Ze\n'
                 'Jülich 2025-06-06 | 35.5 GHz | 288.15 K | 19° elevation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='Rain rate [mm h⁻¹]')
    plt.tight_layout()
    out_sc = OUT_DIR / 'fwd_sim_tower_scatter_20250606.png'
    plt.savefig(out_sc, dpi=150)
    print(f"        Scatter plot saved to:\n        {out_sc}")

    # --- Summary stats ---
    diff     = Ze_inst_s[mask_both] - fwd['Ze_tmm'][mask_both]
    diff_ray = Ze_inst_s[mask_both] - Ze_ray_rain[mask_both]
    print(f"\n── Summary (rainy steps, Ze_inst − Ze_tmm) ──────────────────────")
    print(f"   N rainy steps used : {mask_both.sum()}")
    print(f"   Mean bias          : {diff.mean():.2f} dB  (instrument − T-matrix)")
    print(f"   RMSE               : {np.sqrt((diff**2).mean()):.2f} dB")
    print(f"   Mean bias (vs Ray) : {diff_ray.mean():.2f} dB  (instrument − Rayleigh)")
    print(f"   Max attenuation A  : {fwd['A'].where(rain_mask).max():.3f} dB km⁻¹")
    print("────────────────────────────────────────────────────────────────────")
else:
    print("\n        No steps passed both rain_mask and Ze_inst validity — "
          "skipping scatter plot and stats.")
