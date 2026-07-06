# -*- coding: utf-8 -*-
"""
Forward simulation of Ze (35.6 GHz) from the Jülich Parsivel DSD
for 2025-06-06, using the pre-computed T-matrix scattering table at 283.15 K.

What this script does, step by step
=====================================

STEP 1 – Load the parsivel NetCDF
    File: sups_joy_dm00_l1_any_v00_20250606.nc
    Variables we need:
        time      [s since 1970-01-01 UTC]  shape (1441,)
        M         [counts]                  shape (32, 32, 1441)  ← raw particle matrix (v × D × t)
        Ze        [dBZ]                     shape (1441,)         ← parsivel's own Ze (D^6)
        rr        [mm h-1]                  shape (1441,)         ← rain rate (for masking)
        vclasses  [m s-1]                   shape (32,)           ← velocity bin centres
        dwidth    [mm]                      shape (32,)           ← diameter bin widths
    Note: the first and last time entries are fill values (-9999); they are
    masked to NaT before any processing.

STEP 1b – Reconstruct N(D) from M
    The raw M matrix holds particle counts n(v_j, D_i) per 60 s sample.
    Axis order in the file: M[v_j, D_i, t]  (confirmed by reconstruction test).
    Converting to number concentration:
        N(D_i) [m⁻³ mm⁻¹] = Σ_j  M(v_j, D_i) / (v_j · A · Δt · ΔD_i)
    where  A = 54 cm² = 54×10⁻⁴ m²  (OTT Parsivel2 effective sampling area)
           Δt = 60 s  (sample interval, constant for this file)
    Velocity bins with v = 0 are skipped (dead channel).
    The result is stored as log10(N) for input to FWD_sim.

STEP 2 – Build the parsivel bin edges
    pars_class() returns:
        pars_class  [mm]  (32, 2): column 0 = bin centre, column 1 = bin width
        bin_edges   [mm]  (33, 1): left edge of bin 0 = 0, right edge of bin i
                          = centre[i] + width[i]/2
    These edges define the intervals used later by Binned(bin_edges, N) to
    reconstruct a step-wise PSD at any diameter D.

STEP 3 – Load the T-matrix scattering table (283.15 K, 35.6 GHz)
    The CSV header row contains a Python dict with:
        wl       – wavelength [mm]   = c / (35.6e9) * 1e3 ≈ 8.42 mm
        K2       – radar dielectric factor |K|²
        elevation, canting, frequency, prefactor
    Body columns:
        diameter[mm], radarXSh[mm²], radarXSv[mm²], extxs[mm²], ray[mm²],
        sKdp[mm²], aspect_ratio

STEP 4 – The integration constant
    Ze is defined as:
        Ze [mm⁶ m⁻³] = (λ⁴ / (π⁵ · |K|²)) · ∫ σ_b(D) · N(D) dD
    So:
        int_const = λ⁴ / (π⁵ · |K|²)
    FWD_sim reads wl and K2 from the CSV header to compute this automatically.

STEP 5 – PSD upscaling via Binned()
    The Parsivel has only 32 non-uniform bins (coarser at large D).
    The scattering table is on a fine 0.01 mm grid.
    For each time step, Binned(bin_edges, N(t)) creates a step-wise PSD
    object that returns N(D) = N_bin  for any D inside a bin,  0 outside.
    This PSD is then evaluated on the fine 0.01 mm grid (diameter_ups).

STEP 6 – Ze_tmm  (T-matrix reflectivity)
    Ze_tmm [mm⁶ m⁻³] = int_const · Σ_D  σ_b(D) · N(D) · ΔD
    where σ_b = radarXSh from the T-matrix table (backward scattering, H-pol).
    Converted to dBZ:  Ze_tmm_dBZ = 10 · log10(Ze_tmm)

STEP 7 – Ze_ray  (Rayleigh D^6 reference)
    Ze_ray [mm⁶ m⁻³] = Σ_D  D⁶ · N(D) · ΔD
    This is the classical approximation valid when D ≪ λ.
    At 35.6 GHz (λ ≈ 8.4 mm) and D > 2 mm the non-Rayleigh regime begins,
    so Ze_ray overestimates Ze_tmm for heavy rain.

STEP 8 – Attenuation A  (one-way, dB km⁻¹)
    A [dB km⁻¹] = 8.686e-3 · Σ_D  σ_ext(D) · N(D) · ΔD
    where σ_ext = extxs from the forward-scattering geometry in the table.
    Note: FWD_sim multiplies by 2 (two-way) in the caller's attenuation
    correction formula;  here A is the one-way value per km.

STEP 9 – Mask non-rain periods
    We keep only time steps with rain rate > 0.1 mm h⁻¹ for the plots.

STEP 10 – Save results to CSV and plot
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
PARSIVEL_NC  = Path('/data/obs/site/jue/parsivel/l1/2025/06/06/'
                    'sups_joy_dm00_l1_any_v00_20250606.nc')
SCAT_TABLE   = Path('/work/yegy_project/Feb2026_new/raincoat/samplefiles/'
                    'scattering/293.15_35.6GHz.csv')
OUT_DIR      = Path('/work/yegy_project/Feb2026_new/output/comparison')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── raincoat imports ──────────────────────────────────────────────────────────
from raincoat.FWD_sim import FWD_sim
import raincoat.disdrometer.pars_class as pc

# =============================================================================
# STEP 1 – Load parsivel NetCDF
# =============================================================================
print("STEP 1  Loading parsivel NetCDF …")
ds = nc.Dataset(str(PARSIVEL_NC))

# Unix timestamps → pandas DatetimeIndex UTC
# NOTE: first and last entries are fill values (-9999).
# Infer them from their neighbours using the constant 60 s cadence.
t_raw = np.array(ds.variables['time'][:], dtype=float)
n_bad = int(np.sum(t_raw < 0))
t_raw[t_raw < 0] = np.nan
if np.isnan(t_raw[0]):
    t_raw[0] = t_raw[1] - 60.0
if np.isnan(t_raw[-1]):
    t_raw[-1] = t_raw[-2] + 60.0
time_pd = pd.to_datetime(t_raw, unit='s', utc=True)
print(f"        {len(time_pd)} time steps  ({n_bad} fill-value endpoints fixed)")
print(f"        {time_pd[0].strftime('%Y-%m-%d %H:%M')} → "
      f"{time_pd[-1].strftime('%H:%M')} UTC")

# Raw particle count matrix: shape (vclasses=32, dclasses=32, time=1441)
M        = np.array(ds.variables['M'][:])          # (v, D, t)  units: counts
vclasses = np.array(ds.variables['vclasses'][:])   # m s⁻¹
dclasses = np.array(ds.variables['dclasses'][:])   # mm
dwidth   = np.array(ds.variables['dwidth'][:])     # mm
print(f"        M shape: {M.shape}  (vclasses × dclasses × time)")

# Parsivel's own Ze (D^6, from instrument) and rain rate
Ze_pars_inst = np.array(ds.variables['Ze'][:])   # dBZ
rr           = np.array(ds.variables['rr'][:])   # mm h-1
ds.close()
print(f"        Parsivel bin centres range: {dclasses[0]:.3f} – {dclasses[-1]:.1f} mm")

# =============================================================================
# STEP 1b – Reconstruct log10 N(D,t) from raw M matrix
# =============================================================================
print("\nSTEP 1b Reconstructing N(D,t) from M matrix …")
# M axis order confirmed: M[v_j, D_i, t]
# N(D_i,t) [m⁻³ mm⁻¹] = Σ_j M[j,i,t] / (v_j · A_sensor · Δt · ΔD_i)
A_SENSOR = 54e-4   # m²  (OTT Parsivel2 effective sampling area: 54 cm²)
DT       = 60.0    # s   (sample interval, constant for this file)

# Avoid division by zero for v=0 bins
v_safe = np.where(vclasses > 0, vclasses, np.inf)   # (32,)

# Weighted sum over velocity axis: M[j,i,t] / v_j → (32d, 1441t)
M_vsum = (M / v_safe[:, np.newaxis, np.newaxis]).sum(axis=0)   # (32d, 1441t)

# Divide by (A_sensor · Δt · ΔD_i) → N in m⁻³ mm⁻¹
N_from_M = M_vsum / (A_SENSOR * DT * dwidth[:, np.newaxis])    # (32d, 1441t)

# Convert to log10; set N=0 bins to NaN (consistent with FWD_sim expectation)
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
# pars_class() hard-codes the standard OTT Parsivel bin geometry:
#   bins 0-9:   width 0.125 mm
#   bins 10-14: width 0.250 mm
#   bins 15-19: width 0.500 mm
#   bins 20-24: width 1.000 mm
#   bins 25-29: width 2.000 mm
#   bins 30-31: width 3.000 mm
pclass, bin_edges = pc.pars_class()
print(f"        bin_edges shape: {bin_edges.shape}  (33 edges for 32 bins)")
print(f"        First 5 edges [mm]: {bin_edges[:5,0]}")
print(f"        Last  5 edges [mm]: {bin_edges[-5:,0]}")

# =============================================================================
# STEP 3 – Show scattering table header
# =============================================================================
print(f"\nSTEP 3  Reading scattering table header from:\n"
      f"        {SCAT_TABLE.name}")
with open(SCAT_TABLE, 'r') as f:
    header = eval(f.readline())
print(f"        Frequency   : {header['frequency']} GHz")
print(f"        Wavelength  : {header['wl']:.4f} mm")
print(f"        |K|²        : {header['K2']:.4f}")
print(f"        Elevation   : {header['elevation']}°  (default 90° = zenith)")
print(f"        Canting σ   : {header['canting']}°")
print(f"        Shape func  : {header['aspect_ratio_func']}")

# =============================================================================
# STEP 4 – Integration constant
# =============================================================================
wl = header['wl']   # mm
K2 = header['K2']
int_const = wl**4 / (np.pi**5 * K2)
print(f"\nSTEP 4  int_const = λ⁴ / (π⁵·|K|²)"
      f" = {wl:.4f}⁴ / (π⁵·{K2:.4f}) = {int_const:.4e}  [mm⁴]")
print(f"        Multiplied by ∫ σ_b · N dD [mm² · m⁻³] → Ze in mm⁶ m⁻³")

# =============================================================================
# STEP 5-8 – Run FWD_sim for all 1441 time steps
# =============================================================================
print(f"\nSTEP 5-8  Running FWD_sim for {len(time_pd)} time steps …")
print(f"          (upscaling {32} Parsivel bins → 0.01 mm grid, "
      f"integrating σ_b·N·dD at each step)")

fwd = FWD_sim(str(SCAT_TABLE), time_pd, log10_N, bin_edges)
print(f"          Done.  Output columns: {list(fwd.columns)}")

# =============================================================================
# STEP 9 – Mask non-rain periods
# =============================================================================
print("\nSTEP 9  Masking non-rain time steps (rr < 0.1 mm h-1) …")
rr_series = pd.Series(rr, index=time_pd)
rain_mask = rr_series >= 0.1

n_rain = rain_mask.sum()
print(f"        {n_rain} rainy steps  ({n_rain/len(time_pd)*100:.1f}% of day)")

Ze_tmm_rain = fwd['Ze_tmm'].where(rain_mask)
Ze_ray_rain  = fwd['Ze_ray'].where(rain_mask)
Ze_inst_s    = pd.Series(np.where(Ze_pars_inst > -99, Ze_pars_inst, np.nan),
                         index=time_pd)
Ze_inst_rain = Ze_inst_s.where(rain_mask)

# Print a few rainy rows
print("\n        Sample forward-simulated Ze (rainy steps only):")
sample = fwd[rain_mask].dropna().head(8)
print("        " + sample[['Ze_tmm','Ze_ray','A']].to_string().replace('\n','\n        '))

# =============================================================================
# STEP 10 – Save CSV and plot
# =============================================================================
out_csv = OUT_DIR / 'fwd_sim_jue_parsivel_20250606.csv'
fwd['Ze_inst'] = Ze_inst_s
fwd['rr']      = rr_series.values
fwd.to_csv(out_csv)
print(f"\nSTEP 10 Saved results to:\n        {out_csv}")

# --- Plot 1: full-day time series ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Parsivel Forward Simulation – Jülich 2025-06-06\n'
             'Scattering table: 293.15 K, 35.6 GHz  (T-matrix, Thurai 2007)\n'
             'N(D) reconstructed from raw M matrix (v×D counts)',
             fontsize=12)

ax0, ax1, ax2 = axes

ax0.plot(fwd.index, Ze_inst_rain,  color='steelblue',  lw=1.2,
         label='Parsivel Ze (instrument, D⁶)')
ax0.plot(fwd.index, Ze_ray_rain,   color='orange',     lw=1.0, ls='--',
         label='Ze_ray (Rayleigh D⁶ from PSD)')
ax0.plot(fwd.index, Ze_tmm_rain,   color='crimson',    lw=1.2,
         label='Ze_tmm (T-matrix, 35.6 GHz)')
ax0.set_ylabel('Ze [dBZ]')
ax0.legend(fontsize=8, loc='upper right')
ax0.set_ylim(-10, 40)
ax0.grid(True, alpha=0.3)
ax0.set_title('Reflectivity Ze')

ax1.plot(fwd.index, fwd['A'].where(rain_mask), color='purple', lw=1.0)
ax1.set_ylabel('A [dB km⁻¹]')
ax1.set_title('One-way attenuation at 35.6 GHz')
ax1.grid(True, alpha=0.3)

ax2.plot(fwd.index, rr_series, color='teal', lw=1.0)
ax2.set_ylabel('Rain rate [mm h⁻¹]')
ax2.set_title('Rain rate (from Parsivel)')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax2.set_xlabel('Time UTC [2025-06-06]')

plt.tight_layout()
out_ts = OUT_DIR / 'fwd_sim_timeseries_20250606.png'
plt.savefig(out_ts, dpi=150)
print(f"        Time-series plot saved to:\n        {out_ts}")

# --- Plot 2: scatter Ze_inst vs Ze_tmm ---
fig2, ax = plt.subplots(figsize=(6, 6))
mask_both = rain_mask & fwd['Ze_tmm'].notna() & ~np.isnan(Ze_inst_s)
ax.scatter(Ze_inst_s[mask_both], fwd['Ze_tmm'][mask_both],
           c=rr_series[mask_both], cmap='plasma',
           s=8, alpha=0.6, label='Ze_tmm')
ax.scatter(Ze_inst_s[mask_both], Ze_ray_rain[mask_both],
           s=4, alpha=0.3, color='orange', label='Ze_ray (Rayleigh)')
lim = [-5, 40]
ax.plot(lim, lim, 'k--', lw=1, label='1:1 line')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Parsivel Ze instrument [dBZ]')
ax.set_ylabel('FWD simulated Ze [dBZ]')
ax.set_title('Parsivel inst. Ze vs forward-simulated Ze\nJülich 2025-06-06, 35.6 GHz, 293.15 K  (N from M matrix)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
sm = plt.cm.ScalarMappable(cmap='plasma',
     norm=plt.Normalize(rr_series[mask_both].min(), rr_series[mask_both].max()))
plt.colorbar(sm, ax=ax, label='Rain rate [mm h⁻¹]')
plt.tight_layout()
out_sc = OUT_DIR / 'fwd_sim_scatter_20250606.png'
plt.savefig(out_sc, dpi=150)
print(f"        Scatter plot saved to:\n        {out_sc}")

# --- Summary stats ---
diff = Ze_inst_s[mask_both] - fwd['Ze_tmm'][mask_both]
print(f"\n── Summary (rainy steps, Ze_inst − Ze_tmm) ──────────────────────")
print(f"   N rainy steps used : {mask_both.sum()}")
print(f"   Mean bias          : {diff.mean():.2f} dB  (instrument − T-matrix)")
print(f"   RMSE               : {np.sqrt((diff**2).mean()):.2f} dB")
diff_ray = Ze_inst_s[mask_both] - Ze_ray_rain[mask_both]
print(f"   Mean bias (vs Ray) : {diff_ray.mean():.2f} dB  (instrument − Rayleigh)")
print(f"   Max attenuation A  : {fwd['A'].where(rain_mask).max():.3f} dB km⁻¹")
print("────────────────────────────────────────────────────────────────────")
