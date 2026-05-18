# Project Audit
*Written 2026-05-18*

This document records the status of every script, output folder, and tracked file
in the project after a full logic review. "Superseded" means the logic was replaced
by a better version — not that it was wrong at the time it was written.

---

## Scripts

### CURRENT — production quality, correct logic

| File | Purpose | Notes |
|---|---|---|
| `comparison/cloudnet_vs_tower_mira/compare_ze_fwd_both.py` | Primary comparison script (CN + Tower) | All fixes applied: Parsivel −30 s midpoint, no D-filter, RR≥0.1, CN +30 s fall-time shift, Saturatedco mask, linear averaging |
| `comparison/cloudnet_vs_tower_mira/bias_stats.py` | Statistics module | `compute_stats()` → N, mean, median, std, CI, RMSE, slope, intercept, r, p |
| `comparison/cloudnet_vs_tower_mira/test_bias_stats.py` | Unit tests for bias_stats | 16 tests, all passing |
| `comparison/cloudnet_vs_tower_mira/plot_bias_html.py` | Interactive HTML builder | Draft written before spec was finalised — **needs rewriting** per `comparison_spec.md` steps 3–11 |
| `cloudnet/plot_ze_heatmap_monthly_html.py` | CloudNet monthly Ze heatmap (interactive) | ZE_MIN/MAX = ±60, zsmooth=False, correct |
| `cloudnet/plot_ze_heatmap_monthly.py` | CloudNet monthly Ze heatmap (static PNG) | Parallel to HTML version, still useful |

### DIAGNOSTIC — valid logic, single-purpose or inspection tools

These are not broken. They were written for a specific investigation and are not run
regularly, but the logic is sound and they are worth keeping as reference tools.

| File | Purpose | Notes |
|---|---|---|
| `cloudnet/plot_ze_heatmap_20250606.py` | Single-day Ze heatmap with category_bits mask | Hardcoded to 2025-06-06. Valid logic. |
| `cloudnet/inspect_categorize_20250606.py` | Dumps all variables/attributes from a CloudNet NC file | Hardcoded path but trivially reusable. |
| `cloudnet/analyze_saturation_jun05_06_07.py` | Investigated CloudNet Ze saturation Jun 5–7 2025 | Outputs in `output/cloudnet/ze_saturation_analysis_jun05-07.*`. Valid. |
| `cloudnet/check_max_ze_gates_0to7.py` | Max Ze per gate, May–Jul 2025 | Output in `output/cloudnet/cloudnet_max_ze_gates_0to7_may_jun_jul_2025.csv`. |
| `tower/inspect_znc.py` | Full metadata dump of a `.znc` file with HDF5 workaround | Essential for understanding any new ZNC file. |
| `tower/list_tower.py` | Lists HDF5 contents | Quick file inspection. |
| `tower/check_zg_dimensions.py` | Quick dimension check | |
| `tower/show_zg_time.py` | Preview Zg + time from a single file | |
| `tower/plot_zg_heatmap.py` | Single-file (~2 min) reflectivity heatmap | |
| `tower/plot_daily_heatmap.py` | Full-day heatmap from all 48 ZNC files | Visual QC of any date. |
| `tower/zg_stats_all_file_day.py` | Per-file dBZ statistics table | |
| `tower/zg_db_stats.py` | Per-file dBZ stats + saves `.npz` archives | Note: `compare_ze_fwd_both.py` reads ZNC directly, not NPZ. NPZ output is no longer required by the main workflow but may be useful for lightweight exploration. |
| `tower/low_bins_stats.py` | Stats for the lowest N height bins | |
| `wawa_compare/compare_wawa_joyce_bonn_2025.py` | WaWa code agreement: JOYCE vs Bonn Parsivel | Separate analysis, unrelated to Ze comparison. Complete. |
| `wawa_compare/compare_wawa_bonn_tower_2025.py` | WaWa code agreement: Bonn vs Tower Parsivel | Same. |
| `download_data.py` | Bulk rsync download from remote SSH host | Essential utility. |
| `comparison/stack_monthly_images.py` | Stacks two monthly PNGs vertically | Utility for old masked-shift30 outputs. Still functional. |

### NEEDS LOGIC CHECK before reuse

| File | Issue | Verdict |
|---|---|---|
| `comparison/cloudnet_vs_tower_mira/compare_ze.py` | Direct CloudNet↔Tower-ZNC comparison (no FWD, different science question). Uses `SITE_ALT_M = 111.0` (spec says 114 m). No Parsivel midpoint fix (irrelevant here — no Parsivel). No category_bits mask. | Different analysis from everything else (both instruments are radars). Not superseded per se, but has wrong site altitude. OK for rough visual inspection only. |

### SUPERSEDED — replaced by `compare_ze_fwd_both.py`

These were steps in the evolution toward the current script. The logic issues are
documented here so the outputs can be interpreted correctly if someone looks at them.

| File | Logic issues vs current | What outputs it produced |
|---|---|---|
| `comparison/initialscript_compare_ze_fwd_cloudnet_20250606.py` | Hardcoded to 2025-06-06. No category_bits mask. No Parsivel midpoint fix. No fall-time shift. | `output/comparison/compare_ze_fwd_vs_cloudnet_20250606.png` etc. |
| `comparison/initialscript_compare_ze_fwd_cloudnet_masked_20250606.py` | Hardcoded. D-filter present (wrong). No midpoint fix. | `output/comparison/compare_ze_fwd_vs_cloudnet_masked_20250606.png` etc. |
| `comparison/compare_ze_fwd_cloudnet.py` | No D-filter (good) but no Parsivel midpoint fix, no fall-time shift. | `output/comparison/cloudnet_vs_fwd/` |
| `comparison/compare_ze_fwd_cloudnet_masked.py` | D-filter present (wrong). No midpoint fix. RR_MIN=1.0 (too strict). | `output/comparison/cloudnet_vs_fwd_masked/` |
| `comparison/compare_ze_fwd_cloudnet_masked_shift30.py` | D-filter present (wrong). No midpoint fix. Has +30 s CN shift (correct direction). | `output/comparison/cloudnet_vs_fwd_masked_shift30/` |
| `comparison/compare_ze_fwd_cloudnet_masked_shift45.py` | Same as shift30 but +45 s. Experimental. Only ran for 20250607. | `output/comparison/cloudnet_vs_fwd_masked_shift45/20250607/` |
| `comparison/compare_ze_fwd_tower_20250606.py` | Hardcoded to 20250606. MATCH_TOL_S=60 s (too wide). Auto-detects gate by argmin → produces gate 4 (~208 m ASL) not gate 6 (231 m ASL, correct). | `output/comparison/tower_gate_comparison/` |
| `comparison/compare_ze_fwd_tower_znc.py` | **No Parsivel midpoint fix**. Gate auto-detection via argmin says "typically gate 4" — gate 6 is the correct hardware-fixed gate. | `output/comparison/tower_znc_vs_fwd/` |

---

## Documentation files

| File | Status |
|---|---|
| `comparison_spec.md` | **CURRENT**. Full specification for the interactive HTML workflow. Just updated with radar/site details, timestamp corrections, config constants, bias magnitude context. |
| `comparison/cloudnet_vs_tower_mira/COMPARISON_LOGIC_AUDIT.md` | **HISTORICAL REFERENCE**. Documents all logic decisions and fixes that led to `compare_ze_fwd_both.py`. The TODO list at the bottom is fully done. Keep as audit trail. |
| `data_paths.md` | Keep. Shows file path conventions for 2025-06-06. |
| `filters_reference.md` | Keep. Comprehensive reference for all filters (applied and not). Records decisions with rationale. |
| `README.md` | **STALE**. Shows the original repo structure from before `comparison/cloudnet_vs_tower_mira/` was created. Needs rewriting. |

---

## Output folders

| Folder | Source script | Status | Action |
|---|---|---|---|
| `output/cloudnet/` | `cloudnet/` scripts | Current. Gitignored. | Keep |
| `output/comparison/cloudnet_vs_tower_mira/` | `compare_ze_fwd_both.py` | **CURRENT**. 539 per-day PNGs + 3 pooled PNGs in `all/`. Gitignored. Note: `all/` has no CSVs yet — those require Step 1 (`rain_block_min`). | Keep |
| `output/comparison/tower_gate_comparison/` | `compare_ze_fwd_tower_20250606.py` | Gate geometry reference (per-gate scatter + stats CSV for 20250606). Logic has auto-detection bug but the height array it printed is still useful reference. Gitignored. | Keep (reference) |
| `output/wawa_compare/` | `wawa_compare/*.py` | Separate analysis. Complete. Gitignored. | Keep |
| `output/comparison/tower_znc_vs_fwd/` | `compare_ze_fwd_tower_znc.py` | Superseded script, wrong gate (4 not 6), no midpoint fix. Gitignored. | Can delete (disk) |
| `output/comparison/cloudnet_vs_fwd/` | `compare_ze_fwd_cloudnet.py` | No mask, no midpoint fix. Superseded. Gitignored. | Can delete (disk) |
| `output/comparison/cloudnet_vs_fwd_masked/` | `compare_ze_fwd_cloudnet_masked.py` | D-filter, no midpoint fix. Superseded. Has a `monthly/` sub-folder covering all 6 months. Gitignored. | Can delete (disk) |
| `output/comparison/cloudnet_vs_fwd_masked_shift30/` | `compare_ze_fwd_cloudnet_masked_shift30.py` | D-filter, no midpoint fix. Closest precursor to current. Has `monthly/` and `stacked/`. Gitignored. | Can delete (disk) |
| `output/comparison/cloudnet_vs_fwd_masked_shift45/` | `compare_ze_fwd_cloudnet_masked_shift45.py` | Shift45 experiment, only 20250607. Gitignored. | Can delete (disk) |

---

## Tracked output files — need `git rm --cached`

The gitignore was updated to cover `output/**/*.{png,html,csv}` but 53 files were
already tracked before that change. They are still committed in git and will appear
as "D" (staged for deletion) after `git rm --cached`. They need to be removed so
git stops tracking them.

Run:
```bash
git rm --cached $(git ls-files output/)
git commit -m "Stop tracking regeneratable output files"
```

The files affected are in:
- `output/cloudnet/ze_heatmap_20250606.png`
- `output/comparison/compare_ze_fwd_vs_cloudnet_20250606.png` (and 4 more old comparison PNGs)
- `output/comparison/fwd_sim_*.png / .csv` (6 files — early FWD simulation outputs)
- `output/comparison/geometry_side_view.png`
- `output/comparison/tower_gate_comparison/` (10 gate PNGs + 1 gate_stats.csv)
- `output/wawa_compare/` (6 CSV + 22 PNG files)

---

## Other files

| File | Status |
|---|---|
| `20251009_parsivel_tower.csv` | A single-row raw Tower Parsivel CSV from 2025-10-09. Probably a sample/test file. Only the header row (no data rows). Should be moved to a `data/` or `samples/` subfolder or removed from the project root. |
| `raincoat/` | External library / submodule. Not audited here. |

---

## Pending work (from spec)

1. **Step 1**: Add `rain_block_min` to paired CSVs (`compare_ze_fwd_both.py`) — not done yet
2. **Steps 3–11**: Rewrite `plot_bias_html.py` per spec (one step at a time, with tests)
3. **`git rm --cached`** the 53 tracked output files
4. **Update `README.md`** to reflect current structure
5. **Delete superseded output folders** (disk cleanup, optional)
6. **Discuss which PNGs** to link from the HTML (deferred, to be discussed)
