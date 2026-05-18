# Comparison Specification: Vertical Separation Error

## Research Question

What error do we make when we evaluate radar reflectivity (Ze) at the lowest
vertical range gate using Ze calculated from a surface disdrometer next to the
radar? In other words: how much does Ze change between the surface and the
lowest radar gate, purely due to vertical separation?

---

## Site and Instruments

### JOYCE site

JOYCE (Jülich Observatory for Cloud Evolution) is the radar site. Altitude: **114 m ASL**.
The JOYRAD-35 radar is located here. It operates at **35.5 GHz (Ka-band, ~8.5 mm wavelength)**.

### Radar — JOYRAD-35 (same physical instrument for both comparisons)

JOYRAD-35 alternates between two scanning modes:
- **Vertical mode (~25 min of each 30-min cycle):** feeds CloudNet categorize files (Comparison B)
- **Tilted mode (19° elevation, ~2 min burst per cycle):** records Tower ZNC files (Comparison A)

Because it is the **same physical radar with a single calibration**, any systematic radar calibration
bias is **identical in both comparisons and cancels** when computing the differential
`Bias_B − Bias_A`. The differential isolates the vertical-separation signal alone.

### Instrument table

| Instrument | Location | Temporal resolution | Availability |
|---|---|---|---|
| JOYCE Parsivel | Surface, next to radar (~114 m ASL) | 1 min | Continuous |
| Tower Parsivel (452070) | Tower, **211 m ASL** | 1 min | Continuous |
| JOYRAD-35 vertical beam | JOYCE site | 30 sec | ~25 min / 30 min (nearly continuous) |
| JOYRAD-35 tilted beam (19°) | JOYCE site → tower direction | ~1 sec | 2 min burst every ~30 min |

---

## Two Comparisons

### Comparison A — Tower (same volume, control)

**Purpose:** Quantify the instrument error alone, with no vertical separation.
Parsivel and radar are measuring the same air volume.

- Ze_fwd: forward-simulated from **Tower Parsivel** (T-matrix, 35.5 GHz, 19° elevation)
- Ze_radar: **Tower radar Zg** at **gate 6** (117 m slant range above JOYCE, **231 m ASL**) — hardware-fixed gate, closest to Tower Parsivel altitude (211 m ASL, ~20 m below)

### Comparison B — CloudNet (different heights, the real measurement)

**Purpose:** Quantify instrument error + vertical separation error combined.
Parsivel is at the surface; radar sees a higher air volume.

- Ze_fwd: forward-simulated from **JOYCE Parsivel** (T-matrix, 35.5 GHz, 90° elevation)
- Ze_radar: **CloudNet Ze** at the lowest range gate

---

## Pairing Strategy

The same rule applies to both comparisons:

> **The Parsivel minute is the unit.** For each 1-minute Parsivel step,
> collect all radar observations that fall within ±30 s of that step,
> average them in **linear Ze** (not dBZ), then convert back to dBZ.
> If no radar observation falls within the window, drop that Parsivel step.

### Consequence for Comparison A (Tower)
Most Parsivel minutes will be dropped because the tilted beam is only active
for ~2 min every ~30 min. Only Parsivel minutes that overlap with a radar burst
**and** coincide with rain are kept. The exact number of pairs per burst varies.
This is expected — do not treat dropped steps as missing data, they simply had
no radar observation. The tower comparison will have **far fewer matched pairs**
than the CloudNet comparison. This is why all 18 available rain dates must be
**pooled** before computing statistics — per-day statistics will not be reliable.

### Consequence for Comparison B (CloudNet)
Almost no Parsivel minutes are dropped. The vertical beam provides 2 CloudNet
snapshots per Parsivel minute (30-sec resolution, nearly continuous).

### Timestamp corrections applied before pairing

**Parsivel end-of-interval convention:** Parsivel timestamps mark the *end* of the 1-minute
integration window, not the centre. Correct by subtracting 30 s before matching:
```python
t_p_unix -= 30.0  # shift to mid-interval
```

**CloudNet fall-time shift:** CloudNet Ze at the lowest gate (~255 m ASL) was measured
~144 m above the surface. At typical fall speeds of 5–6 m/s the hydrometeors measured
by the surface Parsivel left the radar gate ~25–30 s earlier. Correct by shifting the
CloudNet timestamps *back* by 30 s (i.e. a CloudNet observation at time T is paired
with a Parsivel observation at T+30 s):
```python
CN_TIME_SHIFT_S = 30.0  # subtract from CloudNet timestamps before matching
```
No such shift is applied for the Tower comparison because the radar gate (231 m ASL)
and Parsivel (211 m ASL) are nearly co-located vertically.

### Key configuration constants

| Constant | Value | Meaning |
|---|---|---|
| `TOWER_GATE` | 6 | Hardware-fixed gate index in Tower ZNC files |
| `TOL_S` | 30.0 s | Matching window half-width (±30 s) |
| `CN_TIME_SHIFT_S` | 30.0 s | CloudNet timestamp shift for fall time |
| `ELEV_DEG` | 19.0° | JOYRAD-35 tilt angle for Tower mode |
| `SITE_ALT_M` | 114.0 m | JOYCE site altitude ASL |
| `DISDRO_ALT_M` | 211.0 m | Tower Parsivel altitude ASL |

### Averaging in linear units
Always average Ze in linear units (mm⁶/m³), then convert:

```
Ze_mean_dBZ = 10 * log10( mean(10^(Ze_i / 10)) )
```

Never average dBZ values directly.

---

## Quality Filters (applied to both comparisons before pairing)

| Filter | Value | Reason |
|---|---|---|
| Rain rate (Parsivel) | RR ≥ 0.1 mm/h | Exclude non-rain periods |
| Tower radar saturation | Saturatedco == 0 | Remove saturated pixels |
| CloudNet category_bits bit 1 | Must be SET | Falling hydrometeors present |
| CloudNet category_bits bit 5 | Must be CLEAR | Exclude insects |
| Ze_radar > 0 (linear) | Must be positive | Avoid log(0) |

---

## Statistics

Compute for each comparison, **pooled across all available dates**:

| Statistic | Formula | Unit |
|---|---|---|
| N | count of valid pairs | — |
| Bias | mean(Ze_radar − Ze_fwd) | dBZ |
| RMSE | sqrt( mean( (Ze_radar − Ze_fwd)² ) ) | dBZ |
| Bootstrap 95% CI on bias | 5000 resamples with replacement | dBZ |

Compute in **dBZ domain** (after averaging linearly and converting).

Per-date statistics should also be saved, but the primary result is pooled.

---

## Final Answer

```
Vertical separation error = Bias_B − Bias_A
                          = Bias(CloudNet) − Bias(Tower)
```

With combined uncertainty from the two bootstrap confidence intervals.

**Interpretation:**
- If Bias_B ≈ Bias_A → vertical separation has little effect on Ze
- If Bias_B > Bias_A → Ze increases from surface to lowest gate (e.g. coalescence)
- If Bias_B < Bias_A → Ze decreases from surface to lowest gate (e.g. evaporation)

---

## Available Dates

### Tower comparison (Comparison A)
18 dates with output already generated:
Feb 2025: 0216, 0222, 0224, 0225, 0226, 0227, 0228
Jun 2025: 0601, 0605, 0606, 0607, 0608, 0614, 0615, 0623, 0624, 0626, 0627

### CloudNet comparison (Comparison B)
3 dates currently processed: 20250605, 20250606, 20250607
More dates should be added to match the tower date range.

---

## Resolved Questions

1. **CloudNet lowest gate:** 255 m AMSL (gate 0 of the categorize file height grid).
2. **No diameter filter:** FWD uses all drop sizes. A D-filter was considered but removed because it creates an asymmetry the radar cannot replicate — the radar sees all hydrometeors regardless of size.
3. **Tower gate:** Tower Parsivel (452070) is at 211 m ASL. Gate 6 is at 231 m ASL (~20 m above), confirmed by inspecting actual ZNC range arrays. This is a hardware-fixed gate; do not change it.
4. **Same physical radar:** JOYRAD-35 scans vertical (~25 min) then tilts to 19° (~2 min burst). Same calibration constant in both modes. Therefore, systematic radar bias is identical in Comparison A and B and cancels in the differential `Bias_B − Bias_A`.
5. **Expected bias magnitude:** Both comparisons show a large shared bias of approximately **−6 to −7 dB** (Ze_radar < Ze_FWD). This is not a bug — it likely reflects a systematic Parsivel overestimation of Ze (known in the literature). The scientifically interesting quantity is the **differential** between the two comparisons, which was observed to be approximately **~0.2 dB** — that is the actual vertical-separation signal. The large shared bias is irrelevant to the research question.
6. **Forward simulation:** T-matrix scattering at 35.5 GHz. Elevation angle (19° for Tower, 90° for CloudNet) is passed to the simulation because it affects the effective dielectric factor and scattering geometry.

---

## Interactive HTML Bias Analysis — Specification

### Purpose

One self-contained HTML file per period (e.g. `bias_analysis_202502–202507.html`) that lets the user explore the bias between Ze_FWD and Ze_radar with full interactivity. Built with Plotly.

### Input data

Paired CSV files produced by `compare_ze_fwd_both.py process_period()`:
- `output/comparison/cloudnet_vs_tower_mira/all/cn_paired_<period>.csv`
  columns: `month, ze_fwd, rr, <height_label>...` (one Ze column per Cloudnet gate)
- `output/comparison/cloudnet_vs_tower_mira/all/tw_paired_<period>.csv`
  columns: `month, ze_fwd, rr, ze_gate0_161m, ze_gate1_173m, ...` (one Ze column per Tower gate)

Both CSVs must also include a `rain_block_min` column (integer, minutes) marking the duration of the continuous rain block each row belongs to. Rows not in any rain block get `rain_block_min = 0`.

### Continuous rain event definition

Adopted from `/work/yegy_project/new_notebooks_git/continuous_rain_ratio_analysis.py` → `filter_by_continuous_blocks()`:

- Work on **Parsivel timestamps** (1-minute steps).
- Group consecutive steps into blocks. A block boundary occurs when the gap between two consecutive timestamps exceeds **2 minutes** (i.e. > (max_gap_minutes+1) × 60 s, with max_gap_minutes=1).
- Allow up to **2 short internal gaps** (≤ 2 min) per block before splitting — tolerance for occasional missing minutes.
- Each step is tagged with its block's total duration in minutes (`rain_block_min`).
- Filter categories exposed in the HTML:
  - **All events** (no filter)
  - **> 90 min** continuous rain
  - **> 60 min** continuous rain
  - **> 30 min** continuous rain
  - **< 30 min** (isolated showers)

This column must be computed and written into both paired CSVs by `compare_ze_fwd_both.py`.

### Controls (affect all panels simultaneously)

| Control | Type | What it filters |
|---|---|---|
| Gate / height | Dropdown | Which radar gate is shown in scatter, Bland-Altman, and violin |
| Rain rate | Range slider (0–30 mm/h) | Min RR threshold |
| Rain event duration | Radio buttons | All / >90 min / >60 min / >30 min / <30 min |
| Month | Legend click (toggle) | Show/hide individual months in scatter and monthly panel |

### Panels (CN and Tower side by side in each panel)

| # | Panel | Description |
|---|---|---|
| 0 | Stats table | One row per gate: N, mean bias, median bias, 95% bootstrap CI, RMSE, std, slope, intercept, Pearson r. Updates when gate dropdown or filters change. |
| 1 | Scatter | Ze_FWD (x) vs Ze_radar (y). 1:1 line + OLS regression line. Points colored by RR. Month toggle in legend. |
| 2 | Bland-Altman | x = (Ze_FWD + Ze_radar)/2, y = Ze_radar − Ze_FWD. Horizontal lines at mean bias, mean ± 1.96 SD (limits of agreement). Shows whether bias is constant or magnitude-dependent. |
| 3 | Violin per gate | One violin per height gate showing distribution of (Ze_radar − Ze_FWD). Box + median line inside. Zero-bias reference line. |
| 4 | Bias profile | Mean bias (filled markers) + median bias (open markers) vs height ASL. Horizontal error bars = 95% bootstrap CI. |
| 5 | Monthly evolution | Monthly mean bias + 95% CI (error bars) and median bias (open markers) for the selected gate, both CN and Tower on the same axes. |

### Statistics computed per gate

- **N** — number of valid matched pairs
- **Mean bias** — mean(Ze_radar − Ze_FWD) in dBZ
- **Median bias** — median(Ze_radar − Ze_FWD) in dBZ (robust, less sensitive to outliers)
- **Std** — standard deviation of the bias distribution
- **Bootstrap 95% CI on mean bias** — 5000 resamples with replacement, no normality assumption
- **RMSE** — sqrt(mean((Ze_radar − Ze_FWD)²))
- **OLS slope and intercept** — Ze_radar = slope × Ze_FWD + intercept; slope ≠ 1 indicates multiplicative bias
- **Pearson r** — linear correlation

---

## Implementation Plan (step by step, with tests after each step)

### Step 1 — Add `rain_block_min` to paired CSVs

Modify `compare_ze_fwd_both.py`:
- Extract Parsivel unix timestamps from the already-aligned data (`t_unix` in `aligned_tw` / CN equivalent)
- Apply `filter_by_continuous_blocks` logic (port from `continuous_rain_ratio_analysis.py`) to assign block duration in minutes to each paired row
- Write `rain_block_min` column into both `cn_paired_*.csv` and `tw_paired_*.csv`

**Test:** Load the CSV, check `rain_block_min` is non-negative integer, check that rows inside a known rain event (e.g. 20250606) get durations > 0, and that dry-period rows get 0.

### Step 2 — Statistics module

Create `comparison/cloudnet_vs_tower_mira/bias_stats.py` with a single function:
```python
def compute_stats(ze_fwd, ze_rad, n_boot=5000) -> dict
```
Returns: N, mean, median, std, ci_lo, ci_hi, rmse, slope, intercept, r, p.

**Test:** Synthetic data — known constant offset, verify mean=offset and CI contains it; zero-variance data; all-NaN data.

### Step 3 — Build stats table panel

In `plot_bias_html.py`, build a Plotly Table showing stats for all gates, for current filter state. Start with the full dataset (no interactivity yet). Verify numbers match manual calculation.

**Test:** Run on real CSVs, compare table output to numbers printed by `compare_ze_fwd_both.py`.

### Step 4 — Violin panel (static first)

Add violin plot of bias per gate (CN left, Tower right), no filter controls yet. Verify distributions look reasonable (centre around −6 to −7 dB, spread ~10 dB).

**Test:** Check violin x-axis labels match gate heights in CSV columns.

### Step 5 — Scatter + Bland-Altman panels (static)

Add scatter and Bland-Altman for the primary gate (CN gate 0, Tower gate 6). Static, no gate dropdown yet. Verify 1:1 line is at y=x, regression line is visible.

**Test:** Check that the regression slope and intercept shown in the annotation match `compute_stats` output.

### Step 6 — Bias profile panel (static)

Add horizontal bias profile with CI error bars for all gates, both CN and Tower overlaid.

### Step 7 — Monthly evolution panel (static)

Add monthly panel showing mean + CI + median per month for CN gate 0 and Tower gate 6.

### Step 8 — Gate dropdown (interactive)

Add a Plotly `updatemenus` dropdown that switches which gate's data is shown in scatter, Bland-Altman, violin highlight, and stats table annotation. All other gates still visible in violin and profile panels.

### Step 9 — Rain rate slider

Add a range slider for minimum RR. Uses Plotly frames or JavaScript `restyle` to filter visible points.

### Step 10 — Rain event duration filter

Add radio buttons (or dropdown) for event duration category using `rain_block_min`. Filters scatter, Bland-Altman, and stats table. Violin shows all gates filtered.

### Step 11 — Final layout and styling

Combine all panels into one scrollable HTML. Add header with period info, instrument descriptions, and notes on assumptions. Add download-as-PNG buttons on each panel.

---

## Files in this workflow

| File | Role |
|---|---|
| `comparison/cloudnet_vs_tower_mira/compare_ze_fwd_both.py` | Main comparison + paired CSV output |
| `comparison/cloudnet_vs_tower_mira/bias_stats.py` | Statistics helper (Step 2) |
| `comparison/cloudnet_vs_tower_mira/plot_bias_html.py` | Interactive HTML builder |
| `output/comparison/cloudnet_vs_tower_mira/all/cn_paired_<period>.csv` | Cloudnet paired data |
| `output/comparison/cloudnet_vs_tower_mira/all/tw_paired_<period>.csv` | Tower paired data |
| `output/comparison/cloudnet_vs_tower_mira/all/bias_analysis_<period>.html` | Final output |

## HTML output decisions

- **No static PNGs embedded** in the HTML. The PNGs produced by `compare_ze_fwd_both.py` remain in their output folders and are opened separately.
- **PNG links** will be added in a later phase (to be discussed — which PNGs are worth linking and from where).
- **Two HTML files per run:**
  - `bias_analysis_<YYYYMM>.html` — single month, all data shown in scatter
  - `bias_analysis_<period>.html` — pooled period (e.g. 202502–202507); scatter caps at 3000 randomly sampled points for display performance, all statistics computed on the full dataset. A note on the scatter panel states how many points are shown vs used for stats.
