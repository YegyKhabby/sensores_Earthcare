import json
import os
import shutil
import sys


def print_table(items):
    if not items:
        return
    width = shutil.get_terminal_size(fallback=(120, 24)).columns
    max_len = max(len(n) for n in items)
    col_width = max_len + 2
    cols = max(1, width // col_width)

    for i, name in enumerate(items, start=1):
        end = "\n" if i % cols == 0 else ""
        print(name.ljust(col_width), end=end)
    if len(items) % cols != 0:
        print()


def list_hdf5(path):
    try:
        import h5py
    except Exception:
        print("Missing dependency: h5py. Install with `pip install h5py`.", file=sys.stderr)
        sys.exit(2)

    items = []
    file_meta = {}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            shape = "x".join(str(s) for s in obj.shape)
            items.append({"name": name, "shape": shape})
        else:
            items.append({"name": f"{name}/", "shape": ""})

    with h5py.File(path, "r") as f:
        if f.attrs:
            for k in f.attrs:
                try:
                    v = f.attrs[k]
                    if hasattr(v, "tolist"):
                        v = v.tolist()
                    file_meta[k] = v
                except Exception as exc:
                    file_meta[k] = f"<unreadable: {exc}>"
        f.visititems(visitor)

    data = {
        "file": path,
        "metadata": file_meta,
        "datasets": items,
    }
    print(json.dumps(data, indent=2, default=str))

directory = "/data/obs/site/jue/joyrad35/2025/06/06"
if len(sys.argv) > 2:
    print("Usage: list_tower.py [directory|file]", file=sys.stderr)
    sys.exit(1)
if len(sys.argv) == 2:
    directory = sys.argv[1]

if os.path.isfile(directory):
    list_hdf5(directory)
    sys.exit(0)

names = []
for name in sorted(os.listdir(directory)):
    path = os.path.join(directory, name)
    if name.endswith("tower.znc") and os.path.isfile(path):
        names.append(name)

print(json.dumps({"directory": directory, "files": names}, indent=2))
