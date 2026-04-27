#!/usr/bin/env python3

import sys, struct
from pathlib import Path
import h5py
import h5py.h5s as h5s, h5py.h5t as h5t, h5py.h5a as h5a
import numpy as np

DEFAULT_DATA_DIR = "/data/obs/site/jue/joyrad35/2025/06/06"

def read_f32(ds):
    shape = ds.shape
    buf   = np.zeros(shape, dtype="float32")
    fsp   = ds.id.get_space()
    msp   = h5s.create_simple(shape)
    ds.id.read(msp, fsp, buf, h5t.IEEE_F32LE)
    return buf

def safe_read(ds):
    try:
        return read_f32(ds)
    except Exception:
        pass
    try:
        return ds.astype("float32")[:]
    except Exception:
        return None

def read_int_attr(ds, name_bytes):
    try:
        aid = h5a.open(ds.id, name_bytes)
        sz  = aid.get_type().get_size()
        buf = np.zeros(sz, dtype="uint8")
        aid.read(buf, h5t.STD_I8LE)
        return int.from_bytes(buf.tobytes(), "little", signed=True)
    except Exception:
        return None

def read_int32_array_attr(ds, name_bytes):
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
    try:
        return list(ds.attrs.keys())
    except Exception:
        return []

def read_attr_display(ds, name):
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
    name_b = name.encode() if isinstance(name, str) else name
    v_int  = read_int_attr(ds, name_b)
    if v_int is not None:
        return str(v_int)
    v_arr = read_int32_array_attr(ds, name_b)
    if v_arr is not None:
        return str(v_arr)
    return "<unreadable>"


arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR
p   = Path(arg)
if p.is_dir():
    files = sorted(p.glob("*tower.znc"))
    if not files:
        sys.exit(f"No tower.znc files in {p}")
    fp = files[0]
else:
    if not p.exists():
        sys.exit(f"File not found: {p}")
    fp = p

print(f"file: {fp}")
print()

with h5py.File(fp, "r") as f:

    print("root attributes:")
    for k, v in f.attrs.items():
        val = v.decode("utf-8", errors="replace") if isinstance(v, (bytes, bytearray, np.bytes_)) else v
        print(f"  {k}: {val}")
    print()

    print("keys:")
    print(" ", sorted(f.keys()))
    print()

    print("range[:10]:")
    if "range" in f:
        rng = safe_read(f["range"])
        for i in range(min(10, len(rng))):
            print(f"  gate {i:>3d}  {float(rng[i]):.4f} m")
    print()

    print("variable attributes:")
    for key in f.keys():
        try:
            ds = f[key]
        except Exception:
            continue
        attr_names = all_attr_names_safe(ds)
        printable = [a for a in attr_names if a not in ("DIMENSION_LIST", "_Netcdf4Dimid", "REFERENCE_LIST", "CLASS")]
        if not printable:
            continue
        print(f"  {key}:")
        for aname in printable:
            val_str = read_attr_display(ds, aname)
            print(f"    {aname}: {val_str}")
