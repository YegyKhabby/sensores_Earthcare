#!/usr/bin/env python3
"""
Inspect /data/obs/site/jue/cloudnet/clu_processing/categorize/2025/
        20250606_juelich_categorize.nc

Prints
------
  1. Global attributes  (instrument, location, date, software …)
  2. Dimensions
  3. Variables table    (name, shape, dtype, units, long_name)
  4. Variable-level attributes for every variable
  5. Head (first 5 rows / samples) of every variable
"""

from pathlib import Path
import numpy as np
import netCDF4 as nc

FPATH = Path(
    "/data/obs/site/jue/cloudnet/clu_processing/categorize/2025"
    "/20250606_juelich_categorize.nc"
)

SEP  = "=" * 72
sep2 = "-" * 72

# ── helpers ───────────────────────────────────────────────────────────────────
def head(arr, n=5):
    """Return first n elements along the first axis."""
    arr = np.ma.filled(arr, np.nan) if hasattr(arr, "mask") else np.asarray(arr)
    if arr.ndim == 0:
        return arr.item()
    return arr.flat[:n].tolist()          # works for any shape


# ── open ──────────────────────────────────────────────────────────────────────
if not FPATH.exists():
    raise FileNotFoundError(f"File not found: {FPATH}")

print(f"File : {FPATH.name}")
print(f"Path : {FPATH.parent}")
print(f"Size : {FPATH.stat().st_size / 1e6:.2f} MB")
print()

with nc.Dataset(FPATH, "r") as ds:

    # ══ 1. GLOBAL ATTRIBUTES ══════════════════════════════════════════════════
    print(SEP)
    print("1.  GLOBAL ATTRIBUTES")
    print(SEP)
    for attr in ds.ncattrs():
        val = getattr(ds, attr)
        print(f"  {attr:<35} {val}")

    # ══ 2. DIMENSIONS ═════════════════════════════════════════════════════════
    print()
    print(SEP)
    print("2.  DIMENSIONS")
    print(SEP)
    for name, dim in ds.dimensions.items():
        print(f"  {name:<25}  size = {len(dim)}")

    # ══ 3. VARIABLES TABLE ════════════════════════════════════════════════════
    print()
    print(SEP)
    print("3.  VARIABLES  (name | shape | dtype | units | long_name)")
    print(SEP)
    W = 32
    fmt = f"  {{:<{W}}}  {{:<22}}  {{:<10}}  {{:<15}}  {{}}"
    print(fmt.format("Name", "Shape", "dtype", "units", "long_name"))
    print(f"  {sep2}")
    for vname, var in ds.variables.items():
        units     = getattr(var, "units",     "—")
        long_name = getattr(var, "long_name", "—")
        print(fmt.format(
            vname, str(var.shape), str(var.dtype),
            str(units)[:15], str(long_name)[:70]
        ))

    # ══ 4. VARIABLE-LEVEL ATTRIBUTES + HEAD ═══════════════════════════════════
    print()
    print(SEP)
    print("4.  PER-VARIABLE ATTRIBUTES  &  HEAD (first 5 values)")
    print(SEP)

    for vname, var in ds.variables.items():
        print()
        print(f"  ▸ {vname}   shape={var.shape}  dtype={var.dtype}")
        # attributes
        for aname in var.ncattrs():
            aval = getattr(var, aname)
            print(f"      {aname:<28} = {aval}")
        # head
        raw  = var[:]
        vals = head(raw, n=5)
        # for 2-D vars, show first 5 rows as well
        if np.asarray(raw).ndim >= 2:
            arr2 = np.ma.filled(raw, np.nan) if hasattr(raw, "mask") else np.asarray(raw, float)
            rows = arr2[:5]            # (up-to-5, n_height)
            print(f"      head[0:5, :]  shape = {rows.shape}")
            for i, row in enumerate(rows):
                # show first 10 values of each row to keep output readable
                snippet = np.round(row[:10], 4).tolist()
                print(f"        row[{i}]: {snippet} ...")
        else:
            print(f"      head[0:5]  = {vals}")

    # ══ 5. TIME & HEIGHT SUMMARY ══════════════════════════════════════════════
    print()
    print(SEP)
    print("5.  TIME AXIS")
    print(SEP)
    if "time" in ds.variables:
        tvar   = ds.variables["time"]
        tunits = getattr(tvar, "units", "unknown")
        tcal   = getattr(tvar, "calendar", "standard")
        t      = tvar[:]
        print(f"  units    : {tunits}")
        print(f"  calendar : {tcal}")
        print(f"  n_times  : {len(t)}")
        try:
            times = nc.num2date(t, tunits, calendar=tcal)
            print(f"  first 5  : {[str(x) for x in times[:5]]}")
            print(f"  start    : {times[0]}")
            print(f"  end      : {times[-1]}")
        except Exception as e:
            print(f"  (decode error: {e})")
            print(f"  raw[0:5] : {t[:5].tolist()}")

    print()
    print(SEP)
    print("6.  HEIGHT AXIS  (AGL from surface)")
    print(SEP)
    for hname in ["height", "range", "altitude"]:
        if hname in ds.variables:
            h      = ds.variables[hname][:]
            hunits = getattr(ds.variables[hname], "units", "m")
            print(f"  variable : {hname}  [{hunits}]")
            print(f"  n_gates  : {len(h)}")
            print(f"  min      : {np.min(h):.1f}  ({np.min(h)/1e3:.3f} km)")
            print(f"  max      : {np.max(h):.1f}  ({np.max(h)/1e3:.3f} km)")
            print(f"  spacing  : ~{np.mean(np.diff(h)):.1f} {hunits}")
            print(f"  first 5  : {np.round(h[:5], 1).tolist()}")
            print(f"  NOTE: tower JOYRAD-35 sits at +120 m AGL → add 120 m when comparing")
            break
