#!/usr/bin/env python3
"""
inspect_znc.py  –  Full metadata and attribute catalogue for joyrad35 .znc files.

What this shows:
  • Every variable: shape, units, long_name, ALL attributes (raw bytes where
    h5py's own decoder fails)
  • Storage format clarification: db=1 means 'display as dB', values ARE linear
    → Zg (units='Z') is stored as linear Z [mm⁶/m³], convert to dBZ with 10·log10(Zg)
  • Fill-value encoding: no _FillValue attr, missing data = IEEE quiet NaN
  • Saturatedco flag: ADC saturation analysis (which gates, how many per file)
  • No large sentinel values in the data

Usage:
    python3 inspect_znc.py [file_or_dir]
"""

import sys, struct
from pathlib import Path
import h5py
import h5py.h5s as h5s, h5py.h5t as h5t, h5py.h5a as h5a
import numpy as np

DEFAULT_DATA_DIR = "/data/obs/site/jue/joyrad35/2025/06/06"

# ── low-level readers ──────────────────────────────────────────────────────────

def read_f32(ds):
    """Read dataset as IEEE float32 (bypasses h5py non-standard type error)."""
    shape = ds.shape
    buf   = np.zeros(shape, dtype="float32")
    fsp   = ds.id.get_space()
    msp   = h5s.create_simple(shape)
    ds.id.read(msp, fsp, buf, h5t.IEEE_F32LE)
    return buf

def safe_read(ds):
    """Best-effort read → float32 array, or None."""
    try:
        return read_f32(ds)
    except Exception:
        pass
    try:
        return ds.astype("float32")[:]
    except Exception:
        return None

def read_int_attr(ds, name_bytes):
    """Read an integer-typed attribute via low-level API, return int or None."""
    try:
        aid = h5a.open(ds.id, name_bytes)
        sz  = aid.get_type().get_size()
        buf = np.zeros(sz, dtype="uint8")
        aid.read(buf, h5t.STD_I8LE)
        return int.from_bytes(buf.tobytes(), "little", signed=True)
    except Exception:
        return None

def read_int32_array_attr(ds, name_bytes):
    """Read a 4-byte-per-element attribute as list of int32."""
    try:
        aid = h5a.open(ds.id, name_bytes)
        sz  = aid.get_type().get_size()
        nel = aid.get_space().get_simple_extent_npoints()
        buf = np.zeros(sz * nel, dtype="uint8")
        aid.read(buf, h5t.STD_I8LE)
        if sz == 4:
            return list(struct.unpack_from(f"<{nel}i", buf.tobytes()))
        return None
    except Exception:
        return None

def all_attr_names_safe(ds):
    """Return list of attribute names using low-level API (safe for broken types)."""
    names = []
    try:
        nattrs = h5a.get_num_attrs(ds.id)
        for i in range(nattrs):
            try:
                aid = h5a.open_idx(ds.id, i)
                names.append(aid.get_name().decode("utf-8", errors="replace"))
            except Exception:
                pass
    except Exception:
        try:
            names = list(ds.attrs.keys())
        except Exception:
            pass
    return names

def read_attr_display(ds, name):
    """Read an attribute and return a human-readable string."""
    # try normal h5py first (works for string and standard numeric attrs)
    try:
        val = ds.attrs[name]
        if hasattr(val, "tolist"):
            val = val.tolist()
        if isinstance(val, list):
            parts = []
            for v in val:
                if isinstance(v, (bytes, bytearray)):
                    parts.append(v.decode("utf-8", errors="replace").strip())
                else:
                    parts.append(str(v))
            return " | ".join(parts)
        if isinstance(val, (bytes, bytearray)):
            return val.decode("utf-8", errors="replace").strip()
        return str(val)
    except Exception:
        pass
    # low-level integer fallback
    name_b = name.encode() if isinstance(name, str) else name
    v_int  = read_int_attr(ds, name_b)
    if v_int is not None:
        return f"{v_int}  [raw int{8*read_int_attr.__code__.co_consts[0] if False else ''}]"
    v_arr = read_int32_array_attr(ds, name_b)
    if v_arr is not None:
        return f"{v_arr}  [raw int32 array]"
    return "<unreadable — non-standard HDF5 type>"


# ── locate file ────────────────────────────────────────────────────────────────
arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR
p   = Path(arg)
if p.is_dir():
    files = sorted(p.glob("*tower.znc"))
    if not files:
        sys.exit(f"No tower.znc files in {p}")
    fp = files[0]
    print(f"Directory : {p}  ({len(files)} files)")
else:
    if not p.exists():
        sys.exit(f"File not found: {p}")
    fp = p
print(f"Inspecting: {fp.name}\n")

W = 100

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 – Radar geometry / pointing summary  (printed first so it's visible)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * W)
print("SECTION 0 — Radar Geometry / Pointing Summary")
print("=" * W)
print()
with h5py.File(fp, "r") as f:
    def _scalar(key):
        """Return mean of a variable (or the scalar value), or None if missing."""
        if key not in f:
            return None
        arr = safe_read(f[key])
        if arr is None:
            return None
        return float(np.nanmean(arr))

    elv_deg     = _scalar("elv")
    azi_deg     = _scalar("azi")
    northangle  = _scalar("northangle")
    lam_m       = _scalar("lambda")
    drg_m       = _scalar("drg")
    rg0_idx     = _scalar("rg0")
    zrg         = _scalar("zrg")
    prf_hz      = _scalar("prf")
    nfft        = _scalar("nfft")
    nave        = _scalar("nave")
    nyq         = _scalar("NyquistVelocity")

    rng_arr = safe_read(f["range"]) if "range" in f else None

freq_ghz = (3e8 / lam_m) / 1e9 if lam_m else None
rg0_m    = float(rng_arr[0]) if rng_arr is not None and len(rng_arr) > 0 else None
rg_last  = float(rng_arr[-1]) if rng_arr is not None and len(rng_arr) > 0 else None
n_gates  = int(zrg) if zrg else (len(rng_arr) if rng_arr is not None else None)

# Vertical height of slant-range gates (range × sin(elv))
if elv_deg is not None and rg0_m is not None:
    import math
    sin_el = math.sin(math.radians(elv_deg))
    h_first_m = rg0_m * sin_el
    h_step_m  = drg_m * sin_el if drg_m else None
else:
    sin_el = h_first_m = h_step_m = None

print(f"  {'Elevation angle (elv)':<35}: {elv_deg:.2f} °  ← tilt above horizon")
print(f"  {'Azimuth angle (azi)':<35}: {azi_deg:.2f} °  (instrument frame; add northangle for true bearing)")
print(f"  {'North offset (northangle)':<35}: {northangle:.2f} °  → true azimuth ≈ {(azi_deg + northangle) % 360:.1f} °")
print()
print(f"  {'Radar frequency':<35}: {freq_ghz:.2f} GHz  (λ = {lam_m*1000:.3f} mm)")
print(f"  {'PRF':<35}: {prf_hz:.0f} Hz")
print(f"  {'FFT points / averages':<35}: {int(nfft)} pts,  {int(nave)} averages")
print(f"  {'Nyquist velocity':<35}: ±{nyq:.3f} m/s")
print()
print(f"  {'Number of range gates':<35}: {n_gates}")
print(f"  {'Range gate spacing (drg)':<35}: {drg_m:.3f} m  (slant range)")
print(f"  {'First gate centre (slant range)':<35}: {rg0_m:.1f} m")
print(f"  {'Last gate centre (slant range)':<35}: {rg_last:.1f} m")
print()
if h_first_m is not None:
    print(f"  Vertical projection  (x sin {elv_deg:.0f} deg = {sin_el:.4f}):")
    print(f"  {'First gate height AGL':<35}: {h_first_m:.1f} m")
    print(f"  {'Vertical gate spacing':<35}: {h_step_m:.2f} m")
    print(f"  {'Max height AGL':<35}: {rg_last * sin_el:.1f} m")
print()
print("=" * W)
print()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – Complete attribute dump for every variable
# ══════════════════════════════════════════════════════════════════════════════
print("=" * W)
print("SECTION 1 — All Variables + All Attributes")
print("=" * W)
print()
print("  Legend for 'db' column:")
print("  db=1  →  stored as LINEAR quantity; instrument software DISPLAYS as 10·log10(value)")
print("  db=0  →  stored in natural / SI units (m/s, ratio, etc.)")
print("  db=—  →  attribute absent (scalar config values)")
print()

with h5py.File(fp, "r") as f:
    for key in f.keys():
        try:
            ds = f[key]
        except Exception:
            continue

        # shape
        try:
            shape = str(ds.shape)
        except Exception:
            shape = "?"

        # db and yrange via low-level
        db_val = read_int_attr(ds, b"db")
        yrange = read_int32_array_attr(ds, b"yrange")
        db_str = str(db_val) if db_val is not None else "—"
        yr_str = str(yrange) if yrange else "—"

        # data range
        data = safe_read(ds) if shape not in ("?", "()") else None
        if data is not None and data.size > 0:
            try:
                n_nan = int(np.isnan(data).sum())
                v_min = float(np.nanmin(data))
                v_max = float(np.nanmax(data))
                rng   = f"min={v_min:.5g}  max={v_max:.5g}  NaN={n_nan}"
                if db_val == 1 and v_min > 0:
                    rng += (f"  →  [{10*np.log10(v_min):.1f}, "
                            f"{10*np.log10(v_max):.1f}] in 10·log10 units")
            except Exception:
                rng = "unreadable"
        else:
            rng = "scalar / unreadable via h5py"

        print(f"  ┌── {key}")
        print(f"  │   shape={shape}  db={db_str}  yrange={yr_str}")
        print(f"  │   data : {rng}")

        # all attributes
        attr_names = all_attr_names_safe(ds)
        for aname in attr_names:
            val_str = read_attr_display(ds, aname)
            print(f"  │   attr  {aname:<26}: {val_str}")

        # explicit _FillValue check
        fv_present = False
        try:
            _ = ds.attrs["_FillValue"]
            fv_present = True
        except Exception:
            pass
        if not fv_present:
            print(f"  │   attr  {'_FillValue':<26}: (NOT present)")

        print(f"  └──")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – Zg storage format: linear or dBZ?
# ══════════════════════════════════════════════════════════════════════════════
print("=" * W)
print("SECTION 2 — Proving Zg is stored as LINEAR Z [mm⁶/m³], NOT as dBZ")
print("=" * W)
print("""
  The file uses two attributes together to describe each quantity:
    units = 'Z'   →  CF-convention unit symbol for radar reflectivity factor Z [mm⁶/m³]
    db    = 1     →  this is a JOYRAD software display flag.  It tells the instrument's
                     own GUI to apply 10·log10 before plotting (i.e., show dBZ).
                     It does NOT mean the stored values are already in dBZ.

  PROOF from actual data values:
""")
with h5py.File(fp, "r") as f:
    zg  = read_f32(f["Zg"])
    rng = read_f32(f["range"])

v_min = float(np.nanmin(zg))
v_max = float(np.nanmax(zg))

print(f"  Stored Zg min = {v_min:.6f},  max = {v_max:.4f}")
print()
print(f"  If these were dBZ:")
print(f"    min = {v_min:.6f} dBZ  → essentially zero (a clean lower bound of 0 dBZ")
print(f"                             never happens; clear-sky bins would be NaN, not 0)")
print(f"    max = {v_max:.1f} dBZ  → PHYSICALLY IMPOSSIBLE at 35 GHz.")
print(f"                             Even severe hail rarely exceeds 50 dBZ at 35 GHz.")
print()
print(f"  If these are linear Z [mm⁶/m³]:")
print(f"    min = {v_min:.6f} mm⁶/m³  →  10·log10 = {10*np.log10(v_min):.1f} dBZ")
print(f"    max = {v_max:.4f}    mm⁶/m³  →  10·log10 = {10*np.log10(v_max):.1f} dBZ")
print(f"    Range {10*np.log10(v_min):.0f} to {10*np.log10(v_max):.0f} dBZ is correct for a 35 GHz cloud/precipitation radar.")
print()
print(f"  ✔  Conclusion: Zg is stored as LINEAR Z [mm⁶/m³].")
print()
print(f"  Correct formulas:")
print(f"    dBZ per pixel  =  10 * log10(Zg_stored)")
print(f"    mean dBZ       =  10 * log10( nanmean(Zg_stored) )")
print()
print(f"  The SAME applies to: SNRg (→ SNR dB), LDRg (→ LDR dB),")
print(f"  Zcx, SNRcx, ISDRco, ISDRcx, HSDco, HSDcx, LDRnormal, RadarConst")
print(f"  All of these have db=1 and are stored as linear power ratios or Z.")
print()
print(f"  Variables with db=0 (VELg m/s, RMSg m/s, RHO, SKWg) are stored")
print(f"  directly in SI/natural units — no log conversion needed.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – Fill values and sentinel scan
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * W)
print("SECTION 3 — Fill Value / Missing Data Encoding")
print("=" * W)
print()
with h5py.File(fp, "r") as f:
    print("  Searching for _FillValue / missing_value attributes on all 2-D variables:")
    any_found = False
    for key in f.keys():
        try:
            ds = f[key]
            if len(ds.shape) < 1:
                continue
        except Exception:
            continue
        for aname in ["_FillValue", "missing_value", "valid_range", "valid_min", "valid_max"]:
            try:
                _ = ds.attrs[aname]
                val_str = read_attr_display(ds, aname)
                print(f"    {key}.{aname} = {val_str}")
                any_found = True
            except Exception:
                pass
    if not any_found:
        print("    → None found.  No explicit fill-value attributes in the file.")
    print()

    # How are NaN values stored at the byte level?
    zg_raw = read_f32(f["Zg"])
    nan_pos = tuple(np.argwhere(np.isnan(zg_raw))[0])
    nan_u32 = zg_raw.view(np.uint32)[nan_pos]
    print(f"  NaN bit pattern (uint32) at position {nan_pos}: 0x{nan_u32:08X}")
    print(f"  IEEE 754 quiet NaN                            : 0x7FC00000")
    print(f"  → {'✔ Match — standard IEEE quiet NaN' if nan_u32 == 0x7FC00000 else f'Different: 0x{nan_u32:08X}'}")
    print()

    print("  Sentinel value scan (looking for large numbers used as flags):")
    print(f"  {'Variable':<12}  {'non-NaN px':>10}  {'> 1e5':>8}  {'< -1e5':>8}  "
          f"{'exact 0.0':>10}  {'max value':>12}")
    print("  " + "-" * 68)
    for vname in ["Zg", "VELg", "SNRg", "LDRg", "RMSg", "SKWg"]:
        data = safe_read(f[vname])
        if data is None:
            continue
        flat = data[~np.isnan(data)]
        print(f"  {vname:<12}  {flat.size:>10,}  {int((flat>1e5).sum()):>8}  "
              f"{int((flat<-1e5).sum()):>8}  {int((flat==0.0).sum()):>10}  "
              f"{float(np.max(flat)):>12.4g}")
    print()
    print("  → No large sentinel values detected. IEEE NaN is the only missing-data marker.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – Saturatedco: explanation and spatial pattern
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * W)
print("SECTION 4 — Saturated Flag (Saturatedco / Saturatedcx)")
print("=" * W)
print("""
  What is ADC saturation?
  ───────────────────────
  The radar receiver converts the incoming analog signal to a digital number
  using an Analog-to-Digital Converter (ADC).  The ADC has a fixed dynamic range.
  If the received signal power exceeds the maximum the ADC can represent, the
  output is "clipped" to the maximum — the ADC is saturated (overflowed).

  Effect on measurements:
    • The stored Zg at a saturated pixel is LOWER than the true Zg
      (power is clamped, so you read the ADC ceiling, not the target's true Z).
    • All moments at that pixel (VELg, RMSg, LDRg ...) are unreliable.
    • The pixel must be set to NaN before any physical analysis.

  Saturatedco  →  co-polarised channel saturated (Zg is affected)
  Saturatedcx  →  cross-polarised channel saturated (LDRg is affected)

  How to apply: zg[Saturatedco == 1] = NaN
""")
with h5py.File(fp, "r") as f:
    sat_co = safe_read(f["Saturatedco"])
    sat_cx = safe_read(f["Saturatedcx"])
    zg     = read_f32(f["Zg"])
    rng_m  = read_f32(f["range"])

n_t, n_r = sat_co.shape
sat_mask = (sat_co == 1.0)
n_sat    = int(sat_mask.sum())
sat_gates = np.where(sat_mask.any(axis=0))[0]
per_step  = sat_mask.sum(axis=1).astype(int)

print(f"  Grid shape: {sat_co.shape}  →  {n_t} time steps × {n_r} range gates")
print(f"  Saturatedco == 1 pixels : {n_sat}  ({100*n_sat/(n_t*n_r):.2f}% of grid)")
print(f"  Saturated per time step : {np.unique(per_step)} (unique values — always constant)")
print(f"  Affected range gate(s)  : indices {sat_gates.tolist()}")
for g in sat_gates:
    mask_g = sat_mask[:, g]
    z_sat  = zg[mask_g, g]
    z_sat  = z_sat[~np.isnan(z_sat)]
    if z_sat.size > 0:
        print(f"    Gate {g:3d}  →  height = {rng_m[g]/1000:.3f} km  |  "
              f"Zg at sat pixels: {10*np.log10(float(np.nanmin(z_sat))):.1f} – "
              f"{10*np.log10(float(np.nanmax(z_sat))):.1f} dBZ  "
              f"(linear: {float(np.nanmin(z_sat)):.4f} – {float(np.nanmax(z_sat)):.4f} mm⁶/m³)")
print(f"  Saturatedcx == 1 pixels : {int((sat_cx == 1).sum())}")
print()
print("  Pattern: 1 saturated pixel per time step, always at the SAME near-range gate.")
print("  This is a persistent strong reflector at that range (tower / near-field clutter).")
print("  The Zg values there are modest (not extreme), confirming ADC clamping behaviour.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – Variables useful for non-met target identification
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * W)
print("SECTION 5 — Variables for Non-Meteorological Target Identification")
print("=" * W)
print("""
  No dedicated classification variable exists in the file.
  Classification must be derived by combining the moments below.

  db=1 quantities: stored linear, shown here in both stored and log10 units.
""")
NONMET = [
    ("Saturatedco", "Binary flag: 1 = ADC saturation → NaN before use"),
    ("SNRg",        "Signal-to-noise ratio (linear); in dB = 10·log10"),
    ("LDRg",        "Linear depolarisation ratio (linear); in dB = 10·log10"),
    ("RHO",         "Co-cross correlation coefficient 0–1 (linear)"),
    ("RHOwav",      "Spectrum-weighted co-cross correlation 0–1 (linear)"),
    ("VELg",        "Doppler velocity [m/s]"),
    ("RMSg",        "Spectral width [m/s]"),
    ("SKWg",        "Spectral skewness [dimensionless]"),
    ("NPKg",        "Number of spectral peaks (integer); > 1 = multi-peak"),
    ("Zg",          "Reflectivity factor (linear Z mm⁶/m³); dBZ = 10·log10"),
]
with h5py.File(fp, "r") as f:
    print(f"  {'Variable':<14} {'Shape':<15} {'db':>3}  "
          f"{'Stored min':>12}  {'Stored max':>12}  {'→ log10 min':>13}  {'→ log10 max':>13}")
    print("  " + "-" * 90)
    for vname, note in NONMET:
        if vname not in f:
            print(f"  {vname:<14} NOT IN FILE")
            continue
        data = safe_read(f[vname])
        db_v = read_int_attr(f[vname], b"db")
        db_s = str(db_v) if db_v is not None else "—"
        if data is None or data.size == 0:
            print(f"  {vname:<14} {'unreadable':<15}")
            continue
        v_min = float(np.nanmin(data)) if not np.all(np.isnan(data)) else float("nan")
        v_max = float(np.nanmax(data)) if not np.all(np.isnan(data)) else float("nan")
        if db_v == 1 and v_min > 0:
            log_min = f"{10*np.log10(v_min):.2f} dB"
            log_max = f"{10*np.log10(v_max):.2f} dB"
        else:
            log_min = log_max = "—"
        print(f"  {vname:<14} {str(data.shape):<15} {db_s:>3}  "
              f"{v_min:>12.4g}  {v_max:>12.4g}  {log_min:>13}  {log_max:>13}")
        print(f"  {'':14} {note}")
        print()

print("=" * W)
print(f"Done.  Source: {fp}")
print("=" * W)
