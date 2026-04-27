# Comparison Logic Audit & New Folder Plan
*Written 2026-04-26*

---

## Science Goal

| Comparison | Sensors | Separation | Purpose |
|---|---|---|---|
| Cloudnet vs FWD | JOYRAD-35 (vertical) vs Parsivel (surface, 114 m ASL) | 144 m vertical | Measure **height-separation error** |
| Tower ZNC vs FWD | MIRA-35 (19° tilted) vs Parsivel (tower, 211 m ASL) | ~3 m | **Near-zero reference** |

The tower comparison is the control: both sensors at almost the same altitude → residual bias ≈ instrument/simulation error only. Any *additional* bias in the Cloudnet comparison = vertical-separation effect.

---

## Assumptions in Current Comparisons

### 1. Time stamping — Parsivel

The Parsivel integrates over 60 s windows **stamped at the end** of the interval. So a timestamp of 12:01:00 represents rain that fell between 12:00:00 and 12:01:00. This is not corrected — we treat the stamp as if it represents an instantaneous measurement at that moment.

---

### 2. Time matching — both comparisons

Both use a ±30 s symmetric window: for each Parsivel step, all radar/Cloudnet snapshots within ±30 s of the Parsivel timestamp are collected and averaged.

- The window is centred on the **end** of the Parsivel integration, not the middle (which would be −60 s to 0 s)
- If the true representative time of the Parsivel step is its midpoint (t − 30 s), then a ±30 s window around the end-stamp coincidentally covers `[t−60, t+30]` — this **overshoots 30 s into the future**
- **Fix (new folder):** subtract 30 s from Parsivel unix timestamps before computing the distance matrix so the window covers `[t−60, t]`:
  ```python
  # instead of:
  t_p_unix = fwd.index.astype('int64').values / 1e9
  # use:
  t_p_unix = fwd.index.astype('int64').values / 1e9 - 30.0  # shift to midpoint
  ```
  Apply to both comparisons consistently.

---

### 3. Averaging in linear units

When multiple radar/Cloudnet snapshots fall inside the ±30 s window, they are averaged in linear reflectivity (mm⁶/m³) and then converted back to dBZ. This is physically correct — averaging dBZ directly would be wrong (Jensen's inequality). The assumption is that all snapshots within the window are equally valid — no weighting by time proximity.

---

### 4. Cloudnet time shift (+30 s) — physically motivated

In masked/shifted scripts, Cloudnet timestamps are shifted **forward by +30 s** before matching.

**Motivation:** Rain measured at gate 0 (~255 m AMSL) needs ~25 s to fall 141 m at ~5–6 m/s to reach the surface Parsivel (114 m AMSL). Shifting Cloudnet +30 s aligns the radar measurement with when that air parcel reached the disdrometer.

- This is a **physical fall-time correction**, not a timestamp convention fix
- +30 s is physically motivated and roughly correct
- The shift45 variant (+45 s) was also explored
- Tower needs **no shift** — separation is only ~3 m

---

### 5. FWD simulation — DSD reconstruction from M matrix

N(Dᵢ, t) = Σⱼ M[j,i,t] / (vⱼ · A · Δt · ΔDᵢ)

Assumptions:
- Fall velocity vⱼ is taken from the Parsivel velocity class centres (not measured per drop)
- A = 54 cm² is the nominal Parsivel sampling area — not corrected for wind, oblique drops, or instrument aging
- Δt = 60 s fixed — even if the actual integration window varies slightly
- Drops landing on velocity-class boundaries are assigned to one bin with no interpolation

---

### 6. FWD simulation — temperature

| Comparison | Table | Freq | Elev | T |
|---|---|---|---|---|
| Cloudnet | `293.15_35.6GHz.csv` | 35.6 GHz | 90° | 20°C |
| Tower | `293.15_35.5GHz_elev19deg.csv` | 35.5 GHz | 19° | 20°C |

- Tower table was previously at 15°C — **corrected** to 20°C (bias was ~0.3–0.5 dB)
- Temperature is fixed for the whole dataset — no adjustment for actual air temperature per event

---

### 7. D-filter (D_MIN=1 mm, D_MAX=5 mm) — current masked scripts only

Only drops with 1 ≤ D ≤ 5 mm contribute to FWD Ze:
- Drops below 1 mm treated as noise/drizzle — but **the radar sees them**
- Drops above 5 mm treated as margin-fallers — but they may be real and dominate Ze at D⁶
- The radar cannot be diameter-filtered — the D-filter creates an inherent asymmetry that cannot be fixed
- **Fix (new folder):** remove D-filter entirely. FWD uses all drop sizes, matching what the radar physically measures.

---

### 8. Rain rate threshold

| Script | RR threshold | Source |
|---|---|---|
| Cloudnet masked | 1.0 mm/h | Parsivel rr variable |
| Tower | 0.1 mm/h | Parsivel rr variable |

- rr comes from the Parsivel file's own rain rate variable — not recomputed from M matrix (may not be fully consistent)
- **New folder:** no rr threshold on either comparison

---

### 9. Cloudnet category_bits mask

Keep pixels where:
- Bit 1 set: falling hydrometeors present
- Bit 5 clear: no insects

Not checked:
- Quality flags (ground clutter, bit 6)
- Wet-bulb / melting layer

Assumption: Cloudnet's classification algorithm is correct. During heavy rain, misclassification is unlikely but possible.

---

### 10. Tower radar — Saturatedco mask

Radar snapshots where `Saturatedco > 0` are set to NaN. Assumption: this flag reliably identifies all saturated returns and only those — no further quality control (no clutter flag, no SNR threshold).

---

### 11. Tower geometry

| Parameter | Value | Source |
|---|---|---|
| Radar site altitude | 114 m ASL | hardcoded |
| Tower Parsivel altitude | 211 m ASL | hardcoded |
| Beam elevation | 19° | hardcoded |

- Height of gate g: `range[g] × sin(19°) + 114 m`
- Closest gate found by minimising `|height_ASL[g] − 211 m|`
- No correction for beam width (gate treated as a point, not a volume)
- Horizontal offset not accounted for: at 19° elevation the beam drifts `r × cos(19°)` horizontally from the radar. We assume the disdrometer is directly under the sampled air volume.

---

## Bugs Fixed in This Session

| Bug | File | Fix |
|---|---|---|
| `np.abs()` on linear Ze before log10 — reflected negative values to positive | `raincoat/FWD_sim.py` | `np.where(Z > 0, Z, np.nan)` |
| Tower scattering table at 15°C instead of 20°C | `compare_ze_fwd_tower_znc.py` + new table | Generated `293.15_35.5GHz_elev19deg.csv`, updated script |
| Negative n_inf diagnostic counts | diagnostic print only | `(~np.isfinite(col)).sum()` (parenthesis placement) |

---

## New Folder Plan: `comparison/cloudnet_vs_tower_mira/`

**Script:** `compare_ze_fwd_both.py`  
**Output:** `output/comparison/cloudnet_vs_tower_mira/`

### Logic — identical base, radar-specific masks only

```
Both comparisons:
  - Parsivel midpoint fix: t_p_unix -= 30.0
  - No D-filter on FWD Ze
  - No rain rate threshold
  - ±30 s matching window (after midpoint fix → covers [t-60, t])
  - Linear averaging, then convert to dBZ

Cloudnet-specific:
  - category_bits mask: bit 1 set AND bit 5 clear
  - +30 s fall-time shift applied to Cloudnet timestamps before matching

Tower-specific:
  - Saturatedco == 0 mask
  - No time shift
```

### Output per date
- `Ze_scatter_<date>_side_by_side.png` — two scatter panels on one figure
- `Ze_vs_time_<date>.png` — time series of all three Ze signals
- `monthly_Ze_fwd_vs_cloudnet_<YYYYMM>.png`
- `monthly_Ze_fwd_vs_tower_<YYYYMM>.png`

---

## Scattering Tables

| Sensor | File | Freq | Elev | T | K² |
|---|---|---|---|---|---|
| JOYRAD-35 (Cloudnet) | `293.15_35.6GHz.csv` | 35.6 GHz | 90° | 20°C | ~0.91 |
| MIRA-35 (Tower) | `293.15_35.5GHz_elev19deg.csv` | 35.5 GHz | 19° | 20°C | 0.9078 |

Both: Thurai 2007 drop shape, canting σ=7°, Parsivel DSD.

---

## Key Physical Numbers

| Parameter | Cloudnet | Tower |
|---|---|---|
| Radar freq | 35.6 GHz | 35.5 GHz |
| Elevation | 90° (vertical) | 19° |
| Lowest gate | ~255 m AMSL | ~208 m ASL |
| Disdrometer alt | 114 m AMSL | 211 m ASL |
| Separation | 141 m | ~3 m |
| Fall time | ~25–28 s | negligible |
| Time shift | +30 s | 0 s |
| Radar sampling | continuous | burst ~2 min/30 min |

---

## TODO

- [ ] Create `comparison/cloudnet_vs_tower_mira/compare_ze_fwd_both.py`
- [ ] Apply Parsivel midpoint fix (`t_p_unix -= 30.0`) in new script
- [ ] No D-filter, no RR threshold
- [ ] Cloudnet: category_bits mask + +30 s shift
- [ ] Tower: Saturatedco mask only, no shift
- [ ] Side-by-side scatter plots per date and monthly
- [ ] Run for June 2025, compare biases
