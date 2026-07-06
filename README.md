# Ze Vertical Separation Error — JOYCE / JOYRAD-35

**Research question:** How much does radar reflectivity (Ze) change between the surface and the lowest radar gate, purely due to vertical separation? We estimate this by running two parallel comparisons — one that controls for instrument error alone, and one that combines instrument error with vertical separation — and taking the difference.

**Site:** JOYCE (Jülich Observatory for Cloud Evolution), Jülich Research Centre, Germany.  
**Period analysed:** February 2025 and June–July 2025.

---

## Scientific background

Ze at the surface is measured via a Parsivel disdrometer, which records the raindrop size distribution (DSD). From the DSD we forward-simulate Ze using T-matrix scattering at 35.5 GHz. Simultaneously, the JOYRAD-35 Ka-band cloud radar measures Ze directly at its lowest range gate (~255 m ASL), several hundred metres above the surface.

The question is: how different is Ze at the surface vs at the lowest gate, and how do we separate that vertical signal from instrument calibration error?

**Answer: two comparisons, one differential.**

| | Parsivel | Radar beam | Purpose |
|---|---|---|---|
| **Comparison A (Tower)** | Tower Parsivel, 211 m ASL | JOYRAD-35 tilted 19°, gate 6 (~231 m ASL) | Control — same air volume, quantifies instrument error only |
| **Comparison B (CloudNet)** | JOYCE Parsivel, 114 m ASL | CloudNet Ze, lowest gate (~255 m ASL) | Main measurement — instrument error + vertical separation |

```
Vertical separation error = Bias_B − Bias_A
                          = Bias(CloudNet) − Bias(Tower)
```

Both comparisons use the same physical radar (JOYRAD-35), so any systematic calibration bias cancels in the differential.

**Expected results:** Both comparisons show a large shared bias of approximately −6 to −7 dB (Ze_radar < Ze_FWD) — this is likely a known Parsivel overestimation of Ze, not a bug. The scientifically interesting quantity, the differential, is approximately **~0.2 dB**.

---

## Instruments

| Instrument | Location | Altitude | Temporal resolution |
|---|---|---|---|
| JOYRAD-35 (vertical mode) | JOYCE site | — | 30 sec (~25 min / 30 min cycle) |
| JOYRAD-35 (tilted 19°, Tower mode) | JOYCE site | — | ~1 sec (2 min burst / 30 min) |
| JOYCE Parsivel | Surface, JOYCE site | 114 m ASL | 1 min |
| Tower Parsivel (452070) | Tower | 211 m ASL | 1 min |

The JOYRAD-35 alternates between vertical scanning (feeds CloudNet) and 19°-tilted scanning (feeds Tower ZNC files) within each 30-minute cycle. It is the **same physical radar with a single calibration** for both modes.

---

## Repository structure

```
Feb2026_new/
├── README.md                              ← this file
├── comparison_spec.md                     ← full workflow specification (authoritative)
├── AUDIT.md                               ← status of every script
├── filters_reference.md                   ← all filters applied / considered
├── data_paths.md                          ← example data file paths (2025-06-06)
│
├── comparison/
│   └── cloudnet_vs_tower_mira/            ← CURRENT production scripts
│       ├── compare_ze_fwd_both.py         ← main comparison script (Comparison A + B)
│       ├── bias_stats.py                  ← statistics module (N, bias, CI, RMSE, r)
│       ├── test_bias_stats.py             ← unit tests for bias_stats (16 tests)
│       ├── plot_bias_html.py              ← interactive HTML builder (draft, needs rewrite)
│       └── COMPARISON_LOGIC_AUDIT.md      ← history of all logic decisions and fixes
│
├── cloudnet/                              ← CloudNet data inspection and heatmaps
│   ├── plot_ze_heatmap_monthly_html.py    ← monthly Ze heatmap (interactive HTML)
│   ├── plot_ze_heatmap_monthly.py         ← monthly Ze heatmap (static PNG)
│   ├── plot_ze_heatmap_20250606.py        ← single-day Ze heatmap (diagnostic)
│   ├── inspect_categorize_20250606.py     ← NetCDF variable dump (diagnostic)
│   ├── analyze_saturation_jun05_06_07.py  ← Ze saturation investigation (diagnostic)
│   └── check_max_ze_gates_0to7.py         ← max Ze per gate May-Jul 2025 (diagnostic)
│
├── tower/                                 ← Tower radar ZNC file tools
│   ├── inspect_znc.py                     ← full metadata dump + HDF5 quirk workaround
│   ├── plot_daily_heatmap.py              ← full-day reflectivity heatmap
│   ├── plot_zg_heatmap.py                 ← single-file heatmap
│   ├── zg_stats_all_file_day.py           ← per-file dBZ statistics
│   ├── zg_db_stats.py                     ← dBZ stats + .npz export
│   ├── low_bins_stats.py                  ← stats for lowest N height bins
│   ├── list_tower.py, check_zg_dimensions.py, show_zg_time.py  ← inspection tools
│
├── wawa_compare/                          ← WaWa precipitation-type code comparison
│   ├── compare_wawa_joyce_bonn_2025.py    ← JOYCE vs Bonn Parsivel WaWa agreement
│   └── compare_wawa_bonn_tower_2025.py    ← Bonn vs Tower Parsivel WaWa agreement
│
├── comparison/                            ← Superseded scripts (reference only)
│   ├── compare_ze_fwd_cloudnet*.py
│   ├── compare_ze_fwd_tower*.py
│   └── initialscript_*.py
│
├── download_data.py                        ← bulk rsync download from remote SSH host
├── test_matching.py                        ← diagnostic: temporal matching test on 2025-06-06
└── raincoat/                              ← external library (not part of this project)
```

Output folders (`output/`) are gitignored — all figures and CSVs are regeneratable.

---

## Running the main comparison

### Primary script: `compare_ze_fwd_both.py`

Runs both Comparison A (Tower) and Comparison B (CloudNet) for one or more dates.
Produces per-day scatter PNGs and pooled paired CSVs.

```bash
cd comparison/cloudnet_vs_tower_mira
python3 compare_ze_fwd_both.py
```

Key constants at the top of the script:

| Constant | Value | Meaning |
|---|---|---|
| `TOWER_GATE` | 6 | Hardware-fixed gate in Tower ZNC files (~231 m ASL) |
| `TOL_S` | 30.0 s | Matching window half-width (±30 s around each Parsivel minute) |
| `CN_TIME_SHIFT_S` | 30.0 s | CloudNet timestamps shifted back for hydrometeor fall time |
| `ELEV_DEG` | 19.0° | JOYRAD-35 tilt angle in Tower mode |
| `SITE_ALT_M` | 114.0 m | JOYCE site altitude ASL |
| `DISDRO_ALT_M` | 211.0 m | Tower Parsivel altitude ASL |

**Timestamp corrections applied before matching:**
- Parsivel timestamps mark the *end* of the 1-minute window: subtract 30 s to get the midpoint.
- CloudNet Ze at ~255 m ASL was measured ~30 s before the drops reach the surface Parsivel: subtract 30 s from CloudNet timestamps before matching.

**Quality filters applied:**
- Rain rate >= 0.1 mm/h (Parsivel)
- `Saturatedco == 0` (Tower radar)
- CloudNet `category_bits` bit 1 set (falling hydrometeors present)
- CloudNet `category_bits` bit 5 clear (no insects)

### Statistics module

```python
from comparison.cloudnet_vs_tower_mira.bias_stats import compute_stats

stats = compute_stats(ze_fwd_array, ze_radar_array, n_boot=5000)
# Returns: N, mean, median, std, ci_lo, ci_hi, rmse, slope, intercept, r, p
```

### CloudNet heatmaps

```bash
python3 cloudnet/plot_ze_heatmap_monthly.py       # static PNG
python3 cloudnet/plot_ze_heatmap_monthly_html.py  # interactive HTML
```

---

## Key physics details

### Ze storage in Tower ZNC files

`Zg` is stored as **linear Z [mm6/m3]**, not dBZ. The attribute `db=1` is a GUI display hint, not a storage format flag.

```python
Zg_dBZ = 10 * np.log10(Zg_stored)  # correct conversion

# Correct mean over a window — average linearly, then convert:
mean_dBZ = 10 * np.log10(np.nanmean(Zg_window))
# WRONG: np.nanmean(10 * np.log10(Zg_window))  ← never average dBZ directly
```

### Known HDF5 quirk in ZNC files

The `time` dataset has dtype `int32` but reports `H5T_NO_CLASS` in the low-level h5py API, causing `ds[:]` to fail. Fix used everywhere:

```python
t = ds.astype('float64')[:]
```

### Forward simulation

Ze is simulated from the Parsivel DSD using **T-matrix scattering at 35.5 GHz**. The elevation angle (19° for Tower, 90° for CloudNet vertical beam) is passed to the simulation because it affects the effective dielectric factor and scattering geometry.

---

## Available rain dates

### Tower comparison (Comparison A) — 18 dates
February 2025: 0216, 0222, 0224, 0225, 0226, 0227, 0228  
June 2025: 0601, 0605, 0606, 0607, 0608, 0614, 0615, 0623, 0624, 0626, 0627

### CloudNet comparison (Comparison B) — 3 dates processed so far
20250605, 20250606, 20250607 (more dates to be added to match Tower date range)

---

## Pending work

1. **Add `rain_block_min` column** to paired CSVs in `compare_ze_fwd_both.py` (see `comparison_spec.md` Step 1)
2. **Rewrite `plot_bias_html.py`** following Steps 3–11 in `comparison_spec.md` (interactive Plotly dashboard: scatter, Bland-Altman, violin per gate, bias profile, monthly evolution)
3. **Extend CloudNet comparison** to cover all 18 Tower dates

See [`comparison_spec.md`](comparison_spec.md) for the full step-by-step implementation plan.  
See [`AUDIT.md`](AUDIT.md) for the current status of every script and output folder.

---

## Dependencies

```
numpy
scipy
pandas
matplotlib
plotly
h5py
netCDF4
pytmatrix   (T-matrix forward simulation)
```

Data files are on the Jülich Research Centre file system at `/data/obs/site/jue/`.  
Use `download_data.py` to rsync from the remote SSH host.
