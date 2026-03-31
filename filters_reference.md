# Data Filters Reference — Jülich Ze Comparison Project

#filters #cloudnet #parsivel #radar #juelich

---

## Overview

This note documents all filtering options relevant to this dataset and records which ones are currently applied in the analysis scripts.

**Instruments:**
- JOYRAD-35 tower radar (`*tower.znc`) — 35.5 GHz, 120 m tower, 19° elevation
- CloudNet categorize NetCDF — JOYCE site, 97 m higher + 330 m from tower
- OTT Parsivel² disdrometers — JOYCE (NC), Tower (CSV), BONN (log)

---

## 1. CloudNet `category_bits`

Each pixel in the CloudNet categorize file has an integer bitmask. Individual bits are tested with `(category_bits & value) != 0`.

| Bit | Value | Meaning | Applied? |
|-----|-------|---------|----------|
| 0 | 1 | Small liquid cloud droplets present | ❌ not used |
| 1 | **2** | **Falling hydrometeors present (rain, drizzle, snow)** | ✅ **KEEP if set** |
| 2 | 4 | Wet-bulb temperature < 0 °C → hydrometeors are ice/snow | ❌ not used |
| 3 | 8 | Melting layer present | ❌ not used |
| 4 | 16 | Aerosol | ❌ not used |
| 5 | **32** | **Insects / biological scatterers** | ✅ **REJECT if set** |
| 6 | 64 | Radar clear-air signal (no cloud/precip) | ❌ not used |

> [!note] Wet-bulb / liquid-phase bit (bit 2, value 4)
> Setting `(category_bits & 4) == 0` would keep only liquid-phase pixels (rain, drizzle) and discard ice and snow. This is **not currently applied**. For a Parsivel-based comparison (Parsivel only measures liquid drops at the surface) this filter would make the comparison cleaner — it avoids comparing Parsivel Ze with radar Ze from ice aloft that has not yet melted. Worth considering if the analysis extends to winter months or convective cases with mixed-phase columns.

> [!note] Melting layer bit (bit 3, value 8)
> The melting layer ("bright band") produces anomalously high Ze due to partially melted ice. Excluding melting-layer pixels (`(category_bits & 8) == 0`) would remove this artefact from Ze comparisons, but also removes valid data near the 0 °C isotherm. Not currently applied.

---

## 2. CloudNet `quality_bits`

| Bit | Value | Meaning | Applied? |
|-----|-------|---------|----------|
| 0 | 1 | Radar signal attenuated by liquid water | ❌ not used |
| 1 | 2 | Attenuation has been corrected | ❌ not used |
| 2 | 4 | Ground clutter / processing artefact | ❌ **removed** (was in `plot_ze_heatmap`) |
| 3 | 8 | Lidar signal detected at this pixel | ❌ not used |
| 4 | 16 | Lidar signal attenuated | ❌ not used |
| 5 | 32 | Radar signal detected | ❌ not used |

> [!note] Ground clutter (bit 2, value 4)
> `(quality_bits & 4) == 0` rejects pixels flagged by the CloudNet algorithm as ground clutter or a retrieval artefact. Most relevant at the lowest height gates. **Removed from the scripts** — the decision was to not apply it, keeping more data in the low-level comparison.

> [!note] Liquid water attenuation (bits 0–1)
> At 35 GHz, liquid water causes significant two-way path attenuation (roughly 0.05–0.3 dB/km per mm/h of rain, more in heavy convection). Bit 1 indicates the correction has been applied. If you are using the corrected `Z` variable from the CloudNet file, the attenuation is already accounted for. If using `radar_reflectivity` (uncorrected), this becomes important.

---

## 3. CloudNet `insect_prob`

A continuous variable (0–1) giving the classifier's probability that a pixel is an insect echo rather than precipitation.

| Threshold | Effect | Applied? |
|-----------|--------|----------|
| `insect_prob < 0.5` | Extra guard for borderline insect pixels missed by the hard bit | ❌ **removed** |

> [!note] Redundancy with category_bits bit 5
> The hard insect bit (`category_bits & 32`) is derived from `insect_prob` — typically set when the probability exceeds an internal threshold near 0.5. Applying both is mostly redundant. Removed in favour of only using the hard bit.

---

## 4. JOYRAD-35 radar (`tower.znc`) hardware flags

| Flag | Condition | Meaning | Applied? |
|------|-----------|---------|----------|
| `Saturatedco` | `== 1` | ADC saturation — signal clipped, value unreliable | ✅ **always masked → NaN** |

> [!note] Other radar quality variables (not currently loaded)
> The `.znc` files contain additional spectral-moment variables that could be used for further QC:
> - **`SNRg`** — signal-to-noise ratio. A minimum SNR threshold (e.g. SNR > −20 dB) would remove noise-floor detections.
> - **`VELg`** — mean Doppler velocity. Rain has negative velocity (falling); filtering on velocity sign would reject updraughts or insects hovering.
> - **`RMSg`** — spectrum width. Very narrow spectra may indicate point targets (insects, birds) rather than hydrometeors.
> - **`NPKg`** — number of spectral peaks. Values > 1 indicate multi-modal spectra (e.g. simultaneous rain + insects). Not currently used.

---

## 5. Parsivel disdrometer filters

### 5a. Drop diameter range

| Parameter | Value | Applied? |
|-----------|-------|----------|
| `D_MIN` | 1.0 mm | ✅ in `compare_ze_fwd_cloudnet_masked` |
| `D_MAX` | 5.0 mm | ✅ in `compare_ze_fwd_cloudnet_masked` |

> [!note] Choice of diameter limits
> - **Lower limit (1 mm):** The first few Parsivel size bins (< ~0.5 mm) are unreliable due to the beam margin effect — drops partially crossing the laser edge are miscounted. 1 mm is conservative; 0.5 mm is defensible if the instrument is well-calibrated.
> - **Upper limit (5 mm):** Drops above ~5 mm are physically rare and are often "margin fallers" — large drops partially intersecting the laser that appear too large. The Parsivel² effective beam narrows for large drops, making counts unreliable above 5 mm.
> - These limits are applied to the DSD before forward-simulating Ze — they do **not** affect the rain rate `rr` loaded directly from the NetCDF.

### 5b. Rain rate threshold

| Script | `RR_MIN` | Applied? |
|--------|----------|----------|
| `compare_ze_fwd_cloudnet` (unmasked) | 0.1 mm/h | ✅ |
| `compare_ze_fwd_cloudnet_masked` | 1.0 mm/h | ✅ |
| `compare_ze_fwd_tower` | 0.1 mm/h | ✅ |

> [!note] Choosing the right threshold
> - **0.1 mm/h** is drizzle — very few drops, DSD is statistically unreliable, Ze is highly variable. FWD Ze at this rate will be noisy.
> - **1.0 mm/h** selects proper rain with a meaningful DSD. Recommended as the standard threshold.
> - Worth considering **0.5 mm/h** as a compromise if you want to include light rain.

### 5c. Velocity plausibility filter (not applied)

> [!note]
> Each Parsivel bin has an associated fall velocity. The instrument records the actual measured velocity per drop. A common QC step is to reject drops whose measured velocity deviates significantly from the expected terminal velocity for their size (e.g. Atlas et al. 1973 relationship). This filters out splashing drops, margin fallers, and strong wind effects. **Not currently applied** in any script — would require reading the full M-matrix velocity dimension, which is available in the JOYCE NetCDF.

### 5d. Sensor status / wind filter (not applied)

> [!note]
> The Parsivel has a `status_sensor` flag in the raw log files. Values other than 0 indicate instrument issues (lens contamination, heating failure, etc.). The raincoat library's conversion script replaces these with NaN, but the analysis scripts load pre-processed files where this is already handled. High wind speeds can also blow drops through the beam at an angle, distorting the velocity measurement — no wind speed filter is currently applied.

---

## 6. Time matching tolerances

| Script | Tolerance | Instrument pair | Applied? |
|--------|-----------|----------------|----------|
| `compare_ze` | ±30 s | CloudNet ↔ tower ZNC | ✅ |
| `compare_ze_fwd_cloudnet*` | ±60 s | Parsivel ↔ CloudNet | ✅ |
| `compare_ze_fwd_tower` | ±60 s | Parsivel ↔ tower ZNC | ✅ |
| `compare_wawa_*` | ±40 s | BONN ↔ JOYCE / Tower | ✅ |

> [!note] Rationale
> The Parsivel averages over 1-minute intervals; ±60 s is one full interval — tighter than the previous ±90 s. The CloudNet grid is ~30 s; ±30 s is one timestep. The wawa instruments have 1-minute logs; ±40 s is conservative.

---

## 7. Geometric / spatial filters

| Offset | Value | Where accounted for |
|--------|-------|-------------------|
| Tower height | 120 m | All tower ZNC height calculations |
| Elevation angle | 19° | `range × sin(19°)` for vertical height |
| JOYCE–tower vertical | 97 m | Subtracted in `compare_ze.py` to align on JOYCE reference |
| JOYCE–tower horizontal | 330 m | Noted in constants; not spatially interpolated |

> [!note] Horizontal separation — not corrected
> The 330 m horizontal separation between JOYCE and the tower means the radar and Parsivel are not sampling the exact same atmospheric column. For stratiform rain (spatially homogeneous) this is minor. For convective cells this can introduce large disagreements unrelated to instrument calibration. No spatial interpolation is applied — all comparisons assume the rain field is horizontally uniform at the scale of 330 m.

---

## 8. Summary table — currently applied filters

| Filter | Value | Scripts |
|--------|-------|---------|
| Saturatedco == 1 → NaN | hardware flag | all tower ZNC scripts |
| category_bits bit 1 (hydrometeors) | must be SET | `compare_ze_fwd_cloudnet_masked`, `plot_ze_heatmap` |
| category_bits bit 5 (insects) | must be CLEAR | `compare_ze_fwd_cloudnet_masked`, `plot_ze_heatmap` |
| Rain rate ≥ 1.0 mm/h | RR_MIN = 1.0 | `compare_ze_fwd_cloudnet_masked`, `compare_ze_fwd_tower` |
| Rain rate ≥ 0.1 mm/h | RR_MIN = 0.1 | `compare_ze_fwd_cloudnet` (unmasked) |
| Drop diameter 1–5 mm | D_MIN/D_MAX | `compare_ze_fwd_cloudnet_masked` |
| Time tolerance ±30 s | | `compare_ze` |
| Time tolerance ±60 s | | `compare_ze_fwd_*` |
| Time tolerance ±40 s | | `compare_wawa_*` |
| Zg ≤ 0 → NaN | from log10 | all tower ZNC scripts |
| dropna on wawa | missing → drop | `compare_wawa_*` |

---

## 9. Filters available but NOT applied

| Filter | Reason not applied |
|--------|-------------------|
| category_bits bit 2 — ice/snow (wet-bulb < 0 °C) | Not applied; would make Parsivel comparison cleaner but removes valid data |
| category_bits bit 3 — melting layer | Not applied; bright-band pixels inflate Ze |
| quality_bits bit 2 — ground clutter | Removed by choice; kept more data at low levels |
| insect_prob < 0.5 | Removed; redundant with category_bits bit 5 |
| SNR threshold on ZNC | SNRg not loaded; would remove noise-floor detections |
| Doppler velocity filter on ZNC | VELg not loaded; could reject insects/updraughts |
| Parsivel velocity plausibility | M-matrix velocity not used; would filter margin fallers |
| Parsivel wind speed filter | No wind data ingested |
| Liquid water attenuation correction check | Using corrected Z from CloudNet file |
