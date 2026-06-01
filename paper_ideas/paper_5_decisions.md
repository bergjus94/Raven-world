# Paper 5 — Locked Methodological Decisions

A clean, current-state summary of every locked decision for Paper 5.
The companion `paper_5_methodology.md` has full rationale and the
history of how we got here; **this document is the short reference**.

Last updated: **2026-06-01**

---

## 1. Paper question

Methodological study of **model-structural controls on baseflow
representation in glacierized mountain catchments across climate
regimes**. We isolate three orthogonal structural axes inside a single
modelling framework (Raven), calibrate every combination to the same
multi-objective target, and ask: which structural changes actually
improve winter baseflow, and does the answer depend on climate regime?

---

## 2. Experimental design at a glance

| Axis | Levels | Realised as |
|---|---|---|
| Architecture (A) | 3 — HBV-linear / HBV-threshold / SPHY-faithful | `perc_option` + `fast_reservoir_release` + `land_surface_routing` |
| Glacier-GW destination (B) | 3 — no link / link to SLOW / link to FAST | `glacier_routing: none / split_to_slow / split_to_fast` |
| Climate regime (C) | 3 — Swiss alpine / UIB monsoon-influenced / (Nepal pending) | Catchment selection |

Phase 1 runs the full **3 × 3 = 9 subsurface structures** on **5
catchments** spanning **2 of the 3 regimes** (Nepal data still pending).

Total Phase-1 calibration runs: **5 × 9 = 45 NSGAII Paretos**.

---

## 3. Subsurface structures (S1–S9)

The 3 × 3 factorial. All use `subsurface_structure: gw_2_layer`
(FAST_RES + SLOW_RES). Parameter counts reflect the calibrated set
after gating optional parameters via `default_params.yaml` conditions.

| Structure | Architecture (A) | Glacier-GW destination (B) | Config key | # params |
|---|---|---|---|---|
| **S1** | HBV-linear | none | `glogem_subdaily_opt1` | 12 |
| **S2** | HBV-linear | SLOW | `glogem_subdaily_opt1_glaciergw` | 13 |
| **S3** | HBV-threshold | none | `glogem_subdaily_opt1_threshold` | 14 |
| **S4** | HBV-threshold | SLOW | `glogem_subdaily_opt1_threshold_glaciergw` | 15 |
| **S5** | SPHY-faithful | none | `glogem_subdaily_opt2_sphy_faithful` | 14 |
| **S6** | SPHY-faithful | SLOW | `glogem_subdaily_opt2_sphy_faithful_glaciergw` | 15 |
| **S7** | HBV-linear | FAST | `glogem_subdaily_opt1_glaciergw_fast` | 13 |
| **S8** | HBV-threshold | FAST | `glogem_subdaily_opt1_threshold_glaciergw_fast` | 15 |
| **S9** | SPHY-faithful | FAST | `glogem_subdaily_opt2_sphy_faithful_glaciergw_fast` | 15 |

**Architectural definitions:**
- *HBV-linear* — single linear release K1 from FAST_RES; `:Flush`
  PONDED_WATER → FAST_RESERVOIR on land HRUs; single `PERC_CONSTANT`
  FAST → SLOW.
- *HBV-threshold* — HBV-linear plus a second `:Baseflow BASE_THRESH_STOR`
  outlet on FAST_RES above UZL (HBV-Light Q0 mechanism, Seibert & Vis
  2012). Adds X17 (UZL) and X18 (K0).
- *SPHY-faithful* — replaces flush-to-fast with direct overland
  `:CatchmentRoute TRIANGULAR_UH`; cascade percolation
  `TOPSOIL → FAST → SLOW` via `:Percolation PERC_POWER_LAW` /
  `PERC_LINEAR`. Adds X12, X13, X14; drops X11.

**Glacier-GW destination definitions:**
- *none* — no `:Split`. Glacier melt enters as PONDED_WATER on
  MASKED_GLACIER HRUs and routes via the same surface path as land HRUs.
- *SLOW* — `:Split RAVEN_DEFAULT PONDED_WATER SURFACE_WATER
  SLOW_RESERVOIR GlacROF`. GlacROF·melt to direct surface; (1-GlacROF)·melt
  to deep GW.
- *FAST* — same `:Split` line but target = FAST_RESERVOIR. (1-GlacROF)·melt
  enters the fast store; drains at K1 (HBV-linear) or via K0/K1
  (HBV-threshold) or cascade (SPHY-faithful).

---

## 4. Catchments

| ID | Name | Region | Sim period | Cali period | Warm-up |
|---|---|---|---|---|---|
| **0102** | Hunza @ Dainyor | UIB monsoon-influenced | 1980–2014 | 1980–1999 | 10-yr cycle from 1980–1989 |
| **0130** | Chenab @ Tandi | UIB lower-elevation monsoon | 1980–2019 | 1980–2002 | 10-yr cycle from 1980–1989 |
| **2161** | Massa @ Blatten | Swiss alpine, heavily glacierized (Aletsch) | 2000–2020 | 2000–2010 | Real 1990–1999 |
| **2256** | Rosegbach @ Pontresina | Swiss alpine, moderate | 2000–2020 | 2000–2010 | Real 1990–1999 |
| **2268** | Rhone @ Gletsch | Swiss alpine, small/steep | 2000–2020 | 2000–2010 | Real 1990–1999 |

**Nepal catchments:** prerequisite delineation + gauge data + GloGEM
forcing still pending; Phase-1 expansion when ready.

---

## 4b. Forcing data [LOCKED 2026-06-01]

Each region uses the **highest-quality bias-corrected precipitation
product available for its domain**. Glacier forcing is GloGEM (regional)
across all catchments.

| Region | meteo_source | precip_source | Rationale |
|---|---|---|---|
| **UIB (Hunza, Chenab)** | ERA5-Land (temp) | **TPHiPr** (Yang et al. 2023) | Published ML-bias-corrected precip for HMA addressing ERA5's high-elevation underestimation. Matches Huang et al. 2026 Pamir SPHY setup directly. TPHiPr coverage 25.75–41.35°N, 61.05–105.65°E. |
| **Swiss (Massa, Rosegbach, Rhone)** | **MeteoSwiss** (RhiresD + TabsD) | (MeteoSwiss) | Gauge-anchored Swiss-specific gridded products (~1 km); peer-reviewed and routine in Swiss alpine hydrology. TPHiPr does not cover Switzerland; ERA5-Land is inferior to MeteoSwiss for this domain. |

**In-model precipitation multiplier (`precip_correction: false` everywhere):**
The bias correction lives in the input data (TPHiPr / MeteoSwiss), not as a
free calibration parameter. Reasoning:

1. **Identifiability:** an in-model Cx couples tightly with melt factor
   (X02) and HBV-β (X05) — adding it as a free parameter introduces
   weakly-identifiable degrees of freedom that absorb model deficiencies.
2. **Structural attribution:** the paper compares structures; Cx as a free
   parameter lets each structure "tune away" its weaknesses via forcing,
   confounding the structural test.
3. **Literature match:** Huang 2026 uses TPHiPr without an additional
   in-model Cx; we match that convention.

Note: the SPHY `default_params.yaml` block does not define an X19/X20/X21
rain-correction entry (the legacy `preprocess_SPHY.py` lookup was looking
for X20/X21 that don't exist for the SPHY model, so `:RainCorrection 1.0`
was being written regardless of namelist flag). Setting
`precip_correction: false` makes this explicit and correct.

---

## 5. Calibration setup [LOCKED]

| Item | Choice |
|---|---|
| **Algorithm** | NSGAII (multi-objective Pareto) |
| **Iterations** | 10000 per structure |
| **Population size** | 50 |
| **Generations** | 200 (= 10000 / 50) |
| **Objectives** | Q, snow, baseflow (3-objective Pareto) |
| **Output label** | `paper5_pareto` (routes to `SPHY_paper5_pareto/`) |
| **Diagnostic log** | ON when snow in objectives |
| **Precip correction** | ON for all catchments |

Wall clock per structure: ~14–35 h depending on catchment size.
Wave-1 (3 catchments × 9 structures = 27 Paretos): ~36 h on hydrolinux
(28 cores). Wave-2 (2 catchments × 9): another ~33 h.

---

## 6. Per-objective metrics [LOCKED 2026-06-01]

| Objective | Metric | Window | Rationale |
|---|---|---|---|
| **Streamflow (Q)** | `KGE_NP` (Pool 2018) | full calibration period | Spearman rank correlation + FDC-based α; outlier-robust; Cinkus 2023 recommended over standard KGE. |
| **Snow (fSCA)** | `nRMSE` | full calibration period | Bounded variable; mean-normalized RMSE rescales to KGE-shaped [≤1, →1=perfect] for Tchebycheff compatibility. |
| **Baseflow (winter Q)** | `KGE_NP` | `deep_winter` (DJF, 3 months) | Same metric family as Q for consistency; non-parametric α handles the narrow winter distribution cleanly. |

KGE_NP formula:
`1 − sqrt((r_S − 1)² + (α_NP − 1)² + (β − 1)²)`
where `r_S` is Spearman rank correlation and
`α_NP = 1 − 0.5·Σ|sorted_sim_norm − sorted_obs_norm|` (FDC-based shape distance).

Snow aggregation: `elevation_band`, `band_width_m: 100`,
`min_pixels_per_band: 5`, `band_aggregation: area_weighted_mean`,
`cloud_threshold: 0.5`.

---

## 7. Baseflow target [LOCKED]

We use **raw winter Q** as the direct baseflow target — no filter
applied. Implementation: `method: raw_winter`, `window: deep_winter`.

**Why raw_winter and not a filter (Eckhardt, Lyne-Hollick, Sliding-Min)?**
For our 4 calibration catchments (2268, 2256, 2161, 0102), filter
comparison shows Eckhardt (BFI_max=0.95) and Sliding-Min agree with raw
winter Q to within 3–8 %. Lyne-Hollick disagrees more (7–22 %) but is
method-inherent — LH identifies quickflow components physically absent
in cold deep-winter Q. Raw is parameter-free and equally accurate.

**Chenab caveat:** filter comparison shows ~25 % non-baseflow content in
DJF for Chenab (lower-elevation, partial snow contribution). We accept
this as a catchment-specific finding and keep raw_winter as the
consistent cross-catchment target so structures can be compared on the
same target.

---

## 8. Parameter bounds [LOCKED]

| Param | Symbol | Bounds | Init | Source |
|---|---|---|---|---|
| X07 | K1 (FAST baseflow coeff) | [0.01, 0.4] /day | 0.276 | HBV-Light (Seibert & Vis 2012) |
| X08 | K2 (SLOW baseflow coeff) | [0.001, 0.10] /day | 0.050 | HBV-Light |
| X11 | FAST `MAX_PERC_RATE` | (opt 1 only) | | |
| X12 | TOPSOIL `MAX_PERC_RATE` | (opt 2 only) | | |
| X13 | `PERC_N` | (opt 2 only) | | |
| X14 | `PERC_COEFF` | [0.05, 1.0] | | (opt 2 only) |
| X16 | GlacROF | [0.5, 0.9] | 0.7 | Lutz 2014 (SPHY HMA) — see caveat below |
| X17 | UZL | [5, 50] mm | | HBV-Light upper, Seibert & Vis 2012 |
| X18 | K0 (threshold release) | [0.1, 0.5] /day | | HBV-Light |

**GlacROF caveat for S7/S8/S9 (fast-routing):** the Lutz 2014 range
[0.5, 0.9] was empirically derived for slow-routing (split_to_slow)
SPHY runs. No literature range exists for split_to_fast. We apply the
same bounds for clean structural comparison; if Pareto fronts show
GlacROF pinned to a bound or uniformly spread (unidentifiable) in
S7/S8/S9, that is itself a publishable finding about partition
behavior when both destinations are fast.

---

## 9. Pareto selection rules [LOCKED]

For each Pareto front, three selection rules will be reported and
compared in the paper:

1. **Theoretical-bounds Tchebycheff** (primary): minimize worst-case
   rescaled shortfall, with theoretical metric bounds [0, 1] for
   KGE-family / nRMSE (clipped). Cleanest cross-catchment comparability.
2. **Weighted-sum SCEUA** (sanity check):
   `0.4·Q + 0.3·snow + 0.3·baseflow` with explicit hydrologic-priority
   weights, calibrated separately by SCEUA. Used to check whether the
   weighted-sum optimum lies on the NSGAII Pareto front.
3. **ε-constraint** (alternative): maximize Q-KGE_NP subject to
   `snow ≥ 0.6` AND `baseflow ≥ 0.6`. Reveals trade-off elbow.

The paper reports how much the selection rule affects the structural
conclusions — likely answer: less than the structural choice itself,
but enough to discuss.

---

## 10. Hypotheses tested

**Main effects** (averaged over the off-axis factor):
- **H_arch:** Is the SPHY-faithful architecture necessary, or does the
  simpler HBV-threshold mechanism (Q0 + Q1) achieve the same baseflow
  improvement? Tested via (S3, S4) vs (S5, S6).
- **H_link:** Does adding a glacier-GW connection improve baseflow at
  all? Tested via (S2,…,S6,S7,…,S9) vs (S1, S3, S5).
- **H_destination:** Given a connection, does the destination
  (SLOW vs FAST) matter? Tested via (S2, S4, S6) vs (S7, S8, S9).

**Interactions:**
- *Architecture × link:* does glacier-GW benefit depend on architecture?
  Tested via (S2−S1) vs (S4−S3) vs (S6−S5).
- *Architecture × destination:* does the SLOW-vs-FAST contrast
  generalize across architectures? Tested via (S2−S7) vs (S4−S8) vs
  (S6−S9). Especially interesting for S4 vs S8 — threshold + FAST
  routing sends summer melt through the K0 outlet (very fast release).

**Climate-regime contrast:**
- *Architecture × regime:* does the "right" architecture depend on
  climate? Tested via Swiss vs UIB main-effect comparison.
- Nepal monsoon catchments to be added once data prerequisites are
  resolved.

---

## 11. What is running now (2026-06-01 17:13 UTC)

**Wave 1** (launched 2026-06-01 16:13 UTC on hydrolinux):
- 0102 Hunza — Phase B in progress (27 NSGAII children)
- 0130 Chenab — Phase A (preprocessing, ~1–2 h remaining)
- 2268 Rhone — Phase B in progress

**Wave 2** (queued, manual launch when wave 1 finishes):
- 2161 Massa
- 2256 Rosegbach

**Nepal** — blocked on data. Add when prerequisites resolved.

---

## 12. Files of record

| File | Purpose |
|---|---|
| `paper_ideas/paper_5_methodology.md` | Full rationale + decision history (the long-form journal) |
| `paper_ideas/paper_5_decisions.md` | **This file** — locked decisions summary |
| `paper_ideas/paper_5_literature.md` | Reference notes for cited literature |
| `namelists/catchment_{0102,0130,2161,2256,2268}_*.yaml` | Per-catchment launch configs |
| `src/config/layers/configurations/glogem_subdaily_*` | Per-structure config layers (S1–S9) |
| `src/config/default_params.yaml` | Parameter names, bounds, init, gating conditions |
| `src/calibration_objectives.py` | METRICS registry (KGE, KGE_NP, NSE, LogKGE, RMSE, nRMSE, MAE, PBIAS, CSI) |
| `scripts/cross_catchment_analysis.py` | Pareto post-processing + plots |
| `scripts/baseflow_filter_comparison.py` | Filter validation (Huang-style) |
| `scripts/smoke_test_S7.sh` | Sanity check for split_to_fast wiring |
