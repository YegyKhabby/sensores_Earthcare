# Jülich 35 GHz Tower Radar – Zg Processing & CloudNet Comparison

**Instrument:** JOYRAD-35 cloud radar, Jülich Research Centre  
**Tower height:** 120 m AGL (all range gates require `+120 m` correction for AGL)  
**Observation day used for development:** 2025-06-06  
**Source directory:** `/data/obs/site/jue/joyrad35/2025/06/06`

---

## Repository structure

```
Feb2026_new/
├── README.md
├── .gitignore
│
├── tower/                          ← JOYRAD-35 tower radar scripts
│   ├── inspect_znc.py              ← metadata dump & storage-format proof
│   ├── list_tower.py               ← list HDF5 contents of tower.znc files
│   ├── check_zg_dimensions.py      ← quick dimension check
│   ├── show_zg_time.py             ← preview Zg + time from a single file
│   ├── plot_zg_heatmap.py          ← single-file heatmap (Z + dBZ)
│   ├── plot_daily_heatmap.py       ← full-day heatmap (Z + dBZ)
│   ├── zg_stats_all_file_day.py    ← per-file dBZ statistics
│   ├── zg_db_stats.py              ← dBZ stats + .npz export
│   └── low_bins_stats.py           ← low-altitude bin analysis
│
├── cloudnet/                       ← CloudNet data analysis
│   └── inspect_cloudnet.py         ← metadata overview for CloudNet .nc files
│
├── comparison/                     ← Tower ↔ CloudNet cross-comparison
│   └── compare_ze.py               ← Ze scatter, bias profile, side-by-side heatmap
│
└── output/
    ├── tower/                      ← figures from tower scripts
    │   ├── zg_heatmap_Z_2min.png
    │   ├── zg_heatmap_dBZ_2min.png
    │   ├── zg_daily_heatmap_Z.png
    │   ├── zg_daily_heatmap_dBZ.png
    │   └── npz/                    ← derived arrays (gitignored, regenerate with zg_db_stats.py)
    ├── cloudnet/                   ← figures from cloudnet scripts
    └── comparison/                 ← figures from comparison scripts
```

---

## Critical storage fact: Zg is LINEAR, not dBZ

The `Zg` dataset in `*tower.znc` files is stored as **linear Z [mm⁶/m³]**.  
The attribute `db=1` is a **display hint** for the instrument GUI, not a storage format flag.  
The attribute `units='Z'` is the CF-convention symbol for linear reflectivity factor.

| What you see in file | What it means |
|---|---|
| `Zg[i,j] = 0.000279` | Minimum signal: 10·log10(0.000279) = **−35.5 dBZ** ✓ |
| `Zg[i,j] = 77.2` | Strong cloud: 10·log10(77.2) = **+18.9 dBZ** ✓ |
| `db = 1` | "Show in log scale in GUI" — does NOT mean stored in dBZ |

### Correct conversion formulas

```python
# Pixel-wise:
Zg_dBZ = 10 * np.log10(Zg_stored)          # stored linear → dBZ

# Correct mean dBZ (average in linear domain FIRST, then convert):
mean_dBZ = 10 * np.log10(np.nanmean(Zg_stored))

# WRONG (direct dBZ average — gives geometric mean, underestimates):
# mean_dBZ = np.nanmean(10 * np.log10(Zg_stored))   ← DO NOT USE
```

---

## Masking

The only quality flag used is **`Saturatedco`** (0/1 per pixel):

| Value | Meaning | Action |
|---|---|---|
| `0` | Normal measurement | **Keep** |
| `1` | ADC saturated — signal clipped | **Mask → NaN** |

In this dataset: exactly 1 pixel per time step is flagged, always at range gate 1
(~0.18 km range) — persistent near-field / tower clutter.

```python
zg[sat == 1.0] = np.nan   # applied in all scripts
```

---

## Tower scripts (`tower/`)

### `plot_zg_heatmap.py`
Single-file (~2 min) reflectivity heatmap. Produces **two PNG files**:
- `output/tower/zg_heatmap_Z_2min.png` — linear Z, plasma colormap + LogNorm
- `output/tower/zg_heatmap_dBZ_2min.png` — Zg (dBZ), viridis, vmin=−35 / vmax=+20

```bash
python3 tower/plot_zg_heatmap.py [path/to/file.znc]
```

### `plot_daily_heatmap.py`
Full-day heatmap from all 48 files. Produces **two PNG files**:
- `output/tower/zg_daily_heatmap_dBZ.png` — viridis, −35 to +20 dBZ
- `output/tower/zg_daily_heatmap_Z.png` — plasma + LogNorm(1e-3, 1e3) mm⁶/m³

```bash
python3 tower/plot_daily_heatmap.py [date_dir]
```

Design: each ~2-minute file block is stretched to 20 min visual width on the
24-hour x-axis so profiles are visible; gaps between files appear as white bands.

### `zg_stats_all_file_day.py`
Per-file and daily dBZ statistics table.
Columns: `Max(dBZ)`, `Min(dBZ)`, `Mean(dBZ)`, `Median(dBZ)`, `NaN%`, `Sat`

```bash
python3 tower/zg_stats_all_file_day.py [date_dir]
```

### `zg_db_stats.py`
Per-file dBZ stats **+ saves `.npz` archives** to `output/tower/npz/`.

```bash
python3 tower/zg_db_stats.py [date_dir] [out_dir]
```

Each `.npz` contains:

| Key | Shape | dtype | Description |
|---|---|---|---|
| `Zg_dBZ` | (n_time, n_height) | float32 | 10·log10(Z_linear)  [dBZ] |
| `Z_linear` | (n_time, n_height) | float32 | Stored linear Z  [mm⁶/m³] |
| `time_unix` | (n_time,) | float64 | Unix timestamps [s] |
| `range_m` | (n_height,) | float32 | Range from radar [m] |
| `filename` | scalar | str | Source filename |

Load example:
```python
import numpy as np
d = np.load("output/tower/npz/20250606_000647_tower.npz")
Zg_dbz  = d["Zg_dBZ"]    # dBZ  (n_time, n_height)
Z_lin   = d["Z_linear"]  # mm⁶/m³
ELV_DEG = 19.0
# Slant range → vertical height AGL at tower base
h_agl   = d["range_m"] * np.sin(np.deg2rad(ELV_DEG)) + 120.0
# To get equivalent height AGL at the JOYCE/CloudNet site (97 m higher, 330 m away):
h_agl_joyce = h_agl - 97.0
```

### `low_bins_stats.py`
Statistics for the lowest N height bins (default: 10, covering 0.14 – 0.47 km).

```bash
python3 tower/low_bins_stats.py [date_dir] [n_bins]
```

### `inspect_znc.py`
Full metadata dump, storage-format proof, fill-value scan, Saturatedco analysis.

```bash
python3 tower/inspect_znc.py <file.znc>
```

---

## CloudNet scripts (`cloudnet/`)

### `inspect_cloudnet.py`
Prints all variables, dimensions, global attributes, and key statistics from
a CloudNet categorization or classification `.nc` file.

```bash
python3 cloudnet/inspect_cloudnet.py <cloudnet_file.nc>
```

**Important:** CloudNet heights are AGL from the **surface** (z=0 at ground).
When comparing with tower data, apply: `tower_h_AGL = range_m + 120`.

---

## Comparison scripts (`comparison/`)

### `compare_ze.py`
Cross-compares Tower Zg with CloudNet Ze. Requires a CloudNet categorization
`.nc` file and the tower `.npz` files.

```bash
python3 comparison/compare_ze.py <cloudnet_file.nc> [output/tower/npz]
```

Without a CloudNet file, running the script prints a diagnostic about available
tower data (useful for checking the .npz layout before you have the CloudNet file).

**Outputs** (saved to `output/comparison/`):
- `Ze_heatmap_side_by_side_<date>.png` — CloudNet | Tower side-by-side daily heatmap
- `Ze_scatter_<date>.png` — scatter plot, bias and RMSE in title
- `Ze_bias_profile_<date>.png` — mean (Tower − CloudNet) per height bin ± 1 std

---

## Daily statistics (2025-06-06, Saturatedco masked)

| Metric | Value |
|---|---|
| Files | 48 |
| Total time steps | 6,372 |
| Height gates | 445  (0.14 – 16.12 km range) |
| Max Zg | +28.7 dBZ |
| Min Zg | −60.2 dBZ |
| **Mean Zg** | **+7.4 dBZ**  `= 10·log10(nanmean(Z_linear))` |
| Median Zg | −21.6 dBZ |
| Missing / NaN | 67.1 % |
| Saturatedco masked | 6,372 pixels  (1 per time step, gate 1, ~0.18 km) |

---

## Known HDF5 quirk (fixed in all scripts)

The `time` dataset has dtype `int32` but reports `H5T_NO_CLASS` (class id 0)
in the low-level h5py API, causing `ds[:]` to raise:
```
Unsupported integer size (0)
```
**Fix used everywhere:** `ds.astype('float64')[:]`
