#!/usr/bin/env python3
import json
import sys

import h5py

FILE_PATH = "/data/obs/site/jue/joyrad35/2025/06/06/20250606_000647_tower.znc"
ROWS = 10
COLS = 5


def main():
    path = FILE_PATH
    if len(sys.argv) > 2:
        print("Usage: show_zg_time.py [file]", file=sys.stderr)
        return 1
    if len(sys.argv) == 2:
        path = sys.argv[1]

    with h5py.File(path, "r") as f:
        zg_key = "zg" if "zg" in f else ("Zg" if "Zg" in f else None)
        if zg_key is None or "time" not in f:
            print(json.dumps({"error": "missing dataset", "available": list(f.keys())[:50]}, indent=2))
            return 2

        zg = f[zg_key]
        time = f["time"]

        def read_1d(ds, n, mtype, dtype):
            try:
                fspace = ds.id.get_space()
                fspace.select_hyperslab((0,), (n,))
                mspace = h5py.h5s.create_simple((n,))
                out = __import__("numpy").empty((n,), dtype=dtype)
                ds.id.read(mspace, fspace, out, mtype)
                return out, None
            except Exception as exc:
                return None, str(exc)

        def read_2d(ds, nrows, ncols, mtype, dtype):
            try:
                fspace = ds.id.get_space()
                fspace.select_hyperslab((0, 0), (nrows, ncols))
                mspace = h5py.h5s.create_simple((nrows, ncols))
                out = __import__("numpy").empty((nrows, ncols), dtype=dtype)
                ds.id.read(mspace, fspace, out, mtype)
                return out, None
            except Exception as exc:
                return None, str(exc)

        n = min(len(time), ROWS)
        zg_rows = min(zg.shape[0], ROWS)
        zg_cols = min(zg.shape[1], COLS)

        zg_vals, zg_err = read_2d(zg, zg_rows, zg_cols, h5py.h5t.IEEE_F32LE, "float32")
        time_vals, time_err = read_1d(time, n, h5py.h5t.NATIVE_INT32, "int32")

    data = {
        "file": path,
        "zg_key": zg_key,
        "time_key": "time",
        "zg_preview": {
            "rows": zg_rows,
            "cols": zg_cols,
            "data": zg_vals.tolist() if zg_vals is not None else None,
        },
        "time_preview": {
            "count": n,
            "data": time_vals.tolist() if time_vals is not None else None,
        },
    }
    if zg_err:
        data["zg_error"] = zg_err
    if time_err:
        data["time_error"] = time_err
    print(json.dumps(data, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
