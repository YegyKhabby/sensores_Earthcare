# -*- coding: utf-8 -*-
"""
Build a T-matrix scattering table for:
    Frequency  : 35.5 GHz   (JOYRAD-35 Ka-band)
    Temperature: 15 °C  →  288.15 K
    Elevation  : 19°

What this script does, step by step
=====================================

STEP 1 – Temperature conversion
    The dielectric model for water (Ellison 2005) requires temperature in Kelvin.
        T_K = 15 + 273.15 = 288.15 K

STEP 2 – Complex refractive index  n = n' + i·n''
    water.n(T_K, f_Hz) calls the Ellison (2005) model which returns the
    complex refractive index of liquid water at the given T and f.
    Internally it computes the double-Debye dielectric permittivity ε = ε' - i·ε''
    and then n = sqrt(ε).
    We pass frequency in Hz   →   f_Hz = 35.5e9 Hz.

STEP 3 – Radar dielectric factor |K|²
    K = (ε - 1) / (ε + 2)     (Clausius–Mossotti / Rayleigh factor)
    |K|² = real( K · K* )
    This is the normalisation factor in the reflectivity equation:
        Ze = (λ⁴ / (π⁵ · |K|²)) · ∫ σ_b(D) · N(D) dD
    raincoat computes this automatically inside scatTable.__init__ via
    utilities.K2(n²).

STEP 4 – Drop shape  (aspect ratio)
    Real raindrops are oblate spheroids whose axis ratio b/a depends on size.
    We use the empirical Thurai et al. (2007) relation:
        tmatrix_aux.dsr_thurai_2007(D)  →  b/a  (< 1, oblate)
    Note: scatTable stores the INVERSE (a/b = 1 / (b/a)) as 'axis_ratio'
    because pytmatrix defines axis_ratio = c/a where c is the symmetry axis.

STEP 5 – Canting angle  (orientation averaging)
    Falling drops are not perfectly oriented; they wobble around the horizontal.
    canting=7.0 means a Gaussian PDF with σ = 7° standard deviation around the
    mean canting angle = 0° (horizontal symmetry axis).
    Inside _compute_single_size the orientation averaging is done by
        rain.or_pdf  = orientation.gaussian_pdf(std=canting)
        rain.orient  = orientation.orient_averaged_fixed

STEP 6 – Elevation angle and scattering geometry
    elevation=19° means the radar beam points 19° above the horizon.
    Internally:  θ₀ = 90° − elevation = 71°
    The T-matrix geometry is set as a tuple
        (θᵢ, θₛ, φᵢ, φₛ, α, β)
    • Backward scattering (reflectivity):
        (θ₀,  180°−θ₀,  0°, 180°, 0°, 0°)
        θᵢ = θₛ = 71°  →  the scattered wave goes back toward the radar.
        From this:  radarXSh, radarXSv  (H and V polarisation radar cross sections [mm²])
    • Forward scattering (attenuation + phase shift):
        (θ₀,  θ₀,  0°, 0°, 0°, 0°)
        θᵢ = θₛ, φᵢ = φₛ  →  wave continues straight through.
        From this:  extxs  (extinction cross section [mm²])
                    sKdp   (specific differential phase contribution [mm²])

STEP 7 – Rayleigh reference
    ray = (π⁵ · |K|² / λ⁴) · D⁶   [mm²]
    This is the classical D⁶ approximation that holds for D ≪ λ.
    It is stored alongside the T-matrix result so you can compare them.

STEP 8 – Save
    The header row of the CSV contains the key parameters as a Python dict
    (wavelength, K², frequency, elevation, canting, aspect_ratio_func name).
    FWD_sim.py reads this header to recover wl and K² for the Ze integral.

Output columns in the CSV
--------------------------
    diameter[mm]   – equivolume diameter of the drop
    radarXSh[mm²]  – backward-scattering cross section, H polarisation
    radarXSv[mm²]  – backward-scattering cross section, V polarisation
    extxs[mm²]     – extinction cross section (forward scattering)
    ray[mm²]       – Rayleigh D⁶ approximation (reference)
    sKdp[mm²]      – specific Kdp contribution
    aspect_ratio   – actual b/a used for this drop size
"""

import numpy as np
from pathlib import Path

from raincoat.scatTable import water
from raincoat.scatTable.TMMrain import scatTable
from raincoat.scatTable import utilities
from pytmatrix import tmatrix_aux

# ── STEP 1: Temperature ────────────────────────────────────────────────────────
T_C  = 15.0                  # degrees Celsius
T_K  = T_C + 273.15          # Kelvin  →  288.15 K
print(f"STEP 1  Temperature: {T_C} °C  =  {T_K} K")

# ── STEP 2: Frequency ──────────────────────────────────────────────────────────
f_GHz = 35.5                 # GHz  (JOYRAD-35 Ka-band)
f_Hz  = f_GHz * 1.0e9       # Hz  — needed by the Ellison model
print(f"STEP 2  Frequency  : {f_GHz} GHz  =  {f_Hz:.3e} Hz")

# ── STEP 3: Complex refractive index of liquid water ──────────────────────────
# water.n() calls the Ellison (2005) double-Debye model and returns n = sqrt(ε)
n = water.n(T_K, f_Hz)       # complex:  n' + i·n''
eps = n * n                  # dielectric permittivity ε = n²
K2  = utilities.K2(eps)      # radar dielectric factor |K|²  (real scalar)
print(f"STEP 3  Refractive index n = {n.real:.4f} + {n.imag:.4f}j")
print(f"        Permittivity    ε = {eps.real:.4f} + {eps.imag:.4f}j")
print(f"        |K|²            = {K2:.4f}  (used to normalise Ze)")

# ── STEP 4: Drop sizes ─────────────────────────────────────────────────────────
# Equivolume diameter from 0.01 mm to 8.49 mm in 0.01 mm steps (849 drops).
# The upper limit covers the largest observed raindrops (~8 mm).
sizes = np.arange(0.01, 8.5, 0.01)   # mm
print(f"STEP 4  Drop sizes : {sizes[0]} – {sizes[-1]} mm, "
      f"{len(sizes)} steps of {sizes[1]-sizes[0]:.2f} mm")

# ── STEP 5: Elevation ──────────────────────────────────────────────────────────
elev_deg = 19.0              # radar elevation angle [°]
theta0   = 90.0 - elev_deg  # zenith angle θ₀ used by T-matrix geometry = 71°
print(f"STEP 5  Elevation  : {elev_deg}°  →  θ₀ = {theta0}° (zenith angle for T-matrix)")

# ── STEP 6: Drop shape ─────────────────────────────────────────────────────────
# Thurai et al. (2007): empirical axis ratio b/a as function of equivolume D.
# Small drops are nearly spherical (b/a ≈ 1), large drops are oblate (b/a < 1).
print(f"STEP 6  Drop shape : Thurai et al. (2007) axis-ratio relation")
print(f"        Example  D=1 mm → b/a = {tmatrix_aux.dsr_thurai_2007(1.0):.3f}")
print(f"        Example  D=3 mm → b/a = {tmatrix_aux.dsr_thurai_2007(3.0):.3f}")
print(f"        Example  D=6 mm → b/a = {tmatrix_aux.dsr_thurai_2007(6.0):.3f}")

# ── STEP 7: Canting angle ──────────────────────────────────────────────────────
canting_std = 7.0            # Gaussian std [°]; typical value for rain
print(f"STEP 7  Canting    : Gaussian PDF σ = {canting_std}° around 0° (horizontal)")

# ── STEP 8: Build the scatTable object ────────────────────────────────────────
print("\nBuilding scatTable …")
table = scatTable(
    frequency        = f_GHz,                          # GHz
    n                = n,                              # complex refractive index
    sizes            = sizes,                          # mm
    canting          = canting_std,                    # deg
    elevation        = elev_deg,                       # deg
    aspect_ratio_func= tmatrix_aux.dsr_thurai_2007,    # Thurai 2007 shape model
)
print(f"  wavelength λ  = {table.wl:.4f} mm")
print(f"  |K|²          = {table.K2:.4f}")
print(f"  prefactor     = {table.prefactor:.4e}  (= π⁵·|K|²/λ⁴, used for Rayleigh Ze)")

# ── STEP 9: Run T-matrix computation ──────────────────────────────────────────
# For each drop size, _compute_single_size is called:
#   • backward geometry → radar cross sections (radarXSh, radarXSv)
#   • forward  geometry → extinction cross section (extxs) and Kdp term
#   • Rayleigh D⁶ reference
print("\nRunning T-matrix … (this may take a few minutes)")
table.compute(verbose=False, procs=1)
print("  Done.")

# Quick sanity check — print a few rows
sample_diameters = [0.5, 1.0, 2.0, 3.0, 5.0]
print("\nSample rows from the scattering table:")
print(f"  {'D [mm]':>8}  {'radarXSh [mm²]':>16}  {'radarXSv [mm²]':>16}  "
      f"{'extxs [mm²]':>14}  {'ray [mm²]':>12}  {'b/a':>6}")
for d in sample_diameters:
    row = table.table.loc[d]
    print(f"  {d:8.2f}  {float(row['radarXSh[mm2]']):16.6e}  "
          f"{float(row['radarXSv[mm2]']):16.6e}  "
          f"{float(row['extxs[mm2]']):14.6e}  "
          f"{float(row['ray[mm2]']):12.6e}  "
          f"{float(row['aspect_ratio']):6.3f}")

# ── STEP 10: Save ──────────────────────────────────────────────────────────────
out_dir  = Path(__file__).parent.parent / "samplefiles" / "scattering"
out_file = out_dir / f"{T_K}_{f_GHz}GHz_elev{int(elev_deg)}deg.csv"
table.save_text_scat_table(str(out_file))
print(f"\nSaved scattering table to:\n  {out_file}")
print("\nHeader parameters written to CSV row 1 (read back by FWD_sim):")
print(f"  wl={table.wl:.4f} mm,  K2={table.K2:.4f},  "
      f"elevation={table.elevation}°,  canting={table.canting}°,  "
      f"frequency={table.frequency} GHz")
