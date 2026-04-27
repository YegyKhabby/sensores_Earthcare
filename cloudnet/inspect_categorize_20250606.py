#!/usr/bin/env python3

from pathlib import Path
import netCDF4 as nc

FPATH = Path(
    "/data/obs/site/jue/cloudnet/clu_processing/categorize/2025"
    "/20250606_juelich_categorize.nc"
)

if not FPATH.exists():
    raise FileNotFoundError(f"File not found: {FPATH}")

with nc.Dataset(FPATH, "r") as ds:

    print("global attributes:")
    for attr in ds.ncattrs():
        print(f"  {attr}: {getattr(ds, attr)}")

    print()
    print("dimensions:")
    for name, dim in ds.dimensions.items():
        print(f"  {name}: {len(dim)}")

    print()
    print("variables:")
    for vname, var in ds.variables.items():
        print(f"  {vname}: shape={var.shape} dtype={var.dtype}")

    print()
    print("location variables:")
    for vname in ["latitude", "longitude", "altitude"]:
        if vname in ds.variables:
            print(f"  {vname}: {ds.variables[vname][:]}")

    print()
    print("height[:10]:")
    if "height" in ds.variables:
        h = ds.variables["height"][:]
        for i in range(min(10, len(h))):
            print(f"  gate {i:>3d}  {float(h[i])}")

    print()
    print("variable attributes:")
    for vname, var in ds.variables.items():
        attrs = {a: getattr(var, a) for a in var.ncattrs()}
        if attrs:
            print(f"  {vname}:")
            for k, v in attrs.items():
                print(f"    {k}: {v}")
