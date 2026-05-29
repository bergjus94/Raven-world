# Paper 5 — Implementation Plan: 6-structure Raven setups

**Status**: planning. To be executed after user confirmation.
**Date**: 2026-05-29
**Goal**: produce working, smoke-tested Raven model setups for all 6 structures (S1–S6 in `paper_5_methodology.md`) and verify they run on a small test catchment before launching Phase 1 calibrations.

---

## Phase 0 — Raven framework research [DONE]

### Threshold release mechanism for S3/S4

Looking at `RavenHydroFramework/src/Baseflow.cpp` (line 268+), Raven has **`BASE_THRESH_STOR`** which is exactly the HBV-Light Q0 formula:

```cpp
else if(type==BASE_THRESH_STOR) {
    K     = pSoil->baseflow_coeff2;     // linear baseflow coeff [1/d]
    tstor = pSoil->storage_threshold;   // threshold storage [mm]
    if (stor > tstor) { rates[0] = K * (stor - tstor); }
}
```

Parameters Raven expects: `BASEFLOW_COEFF2` (1/d) and `STORAGE_THRESHOLD` (mm). We map these to our `X18` (K0) and `X17` (UZL) respectively.

For S3, FAST_RES will have **two concurrent** `:Baseflow` processes:
```
:Baseflow  BASE_LINEAR        FAST_RES → SURFACE_WATER    # Q1 (existing X07/BASEFLOW_COEFF)
:Baseflow  BASE_THRESH_STOR   FAST_RES → SURFACE_WATER    # Q0 (new X17/X18)
```

The alternative `BASE_THRESH_POWER` was rejected — it uses a fractional saturation threshold (`tsat = stor/max_stor`) with power-law form, which doesn't match HBV-Light's storage-based linear-above-threshold formulation.

### Cascade percolation for S5/S6 — already implemented

`preprocess_SPHY.py` already supports `perc_option=2` which emits:
```
:Percolation  PERC_POWER_LAW   TOPSOIL  →  FAST_RES        # X12 (rate), X13 (n)
:Percolation  PERC_LINEAR      FAST_RES →  SLOW_RES        # X14 (coeff)
```

No new percolation code needed. Just needs to be paired with the new "direct overland" flag.

---

## Phase 1 — Code changes to `preprocess_SPHY.py`

### 1.1 Add two new config flags

Both default to current behavior so existing namelists keep working unchanged:

```python
# Added to __init__ after subsurface_structure/glacier_routing handling
self.fast_reservoir_release = namelist.get('fast_reservoir_release', 'linear')
if self.fast_reservoir_release not in ('linear', 'threshold'):
    raise ValueError(f"fast_reservoir_release must be 'linear' or 'threshold', got '{self.fast_reservoir_release}'")

self.land_surface_routing = namelist.get('land_surface_routing', 'flush_to_fast')
if self.land_surface_routing not in ('flush_to_fast', 'direct'):
    raise ValueError(f"land_surface_routing must be 'flush_to_fast' or 'direct', got '{self.land_surface_routing}'")
```

### 1.2 Gate the `:Flush SURFACE_WATER → FAST_RESERVOIR` block

In the `:HydrologicProcesses` emission (around line 1649 in current code), wrap the flush block:

```python
*(["   # Surface water routing on land HRUs (skip glacier/masked-glacier/rock/lake)",
   "   :Flush             RAVEN_DEFAULT      SURFACE_WATER   FAST_RESERVOIR",
   "       :-->Conditional HRU_TYPE IS_NOT GLACIER",
   "       :-->Conditional HRU_TYPE IS_NOT MASKED_GLACIER",
   "       :-->Conditional HRU_TYPE IS_NOT ROCK",
   "       :-->Conditional HRU_TYPE IS_NOT LAKE",
   ""]
  if self.land_surface_routing == 'flush_to_fast' else []),
```

When `land_surface_routing == 'direct'`, no flush is emitted — sat-excess SURFACE_WATER on land HRUs routes directly to outlet via the existing `:CatchmentRoute TRIANGULAR_UH` (which is always active and HRU-type-agnostic).

### 1.3 Add second `:Baseflow` from FAST_RES when threshold mode

Update the baseflow emission block (around line 1664 currently):

```python
# Current single BASE_LINEAR from FAST_RES:
"   :Baseflow          BASE_LINEAR        FAST_RESERVOIR  SURFACE_WATER",
"   :Baseflow          BASE_LINEAR        SLOW_RESERVOIR  SURFACE_WATER",

# Changes to:
"   :Baseflow          BASE_LINEAR        FAST_RESERVOIR  SURFACE_WATER",
*(["   :Baseflow          BASE_THRESH_STOR   FAST_RESERVOIR  SURFACE_WATER  # HBV-Light Q0"]
  if self.fast_reservoir_release == 'threshold' else []),
"   :Baseflow          BASE_LINEAR        SLOW_RESERVOIR  SURFACE_WATER",
```

### 1.4 Add UZL and K0 columns to the .rvp `:SoilParameterList` when threshold mode

In `_build_soil_class_parameters()` (around line 1372–1410), the soil parameter block currently branches on `perc_option`. We add a second branching dimension on `fast_reservoir_release`. For threshold mode, append two columns:

```
:Parameters POROSITY FIELD_CAPACITY SAT_WILT HBV_BETA MAX_CAP_RISE_RATE MAX_PERC_RATE BASEFLOW_COEFF [PERC_N PERC_COEFF if opt2] BASEFLOW_COEFF2 STORAGE_THRESHOLD
:Units      none     none           none     none     mm/d              mm/d           1/d            [none   1/d        if opt2] 1/d              mm
[per-soil rows]
   FAST_RES, ... X07 (BASEFLOW_COEFF), ... X18 (BASEFLOW_COEFF2), X17 (STORAGE_THRESHOLD)
   SLOW_RES, ... X08 (BASEFLOW_COEFF), ... _DEFAULT,              _DEFAULT
   TOPSOIL,  ... 0,                    ... _DEFAULT,              _DEFAULT
```

Only FAST_RES gets non-default values for BASEFLOW_COEFF2 and STORAGE_THRESHOLD. Other soil classes get `_DEFAULT` (Raven sees this as "process inactive for this soil").

**Estimated lines of code**: ~80–120 LOC across preprocess_SPHY.py modifications.

---

## Phase 2 — Parameter definitions in `default_params.yaml`

Add to the SPHY section:

```yaml
SPHY:
  names:
    # ... existing X01–X16 ...
    X17: Sphy_UZL_Threshold              # FAST_RES storage threshold for Q0 [mm]
    X18: Sphy_K0_Fast_Threshold          # FAST_RES above-threshold release rate [1/d]

  init:
    # ... existing ...
    X17: 20.0                            # UZL ~20 mm (HBV-Light typical mid-range)
    X18: 0.3                             # K0 ~0.3 /day (HBV-Light fast rate)

  lower:
    # ... existing ...
    X17: 5.0                             # UZL ≥ 5 mm (Seibert & Vis 2012 lower)
    X18: 0.1                             # K0 ≥ 0.1 /day

  upper:
    # ... existing ...
    X17: 50.0                            # UZL ≤ 50 mm
    X18: 0.5                             # K0 ≤ 0.5 /day

  optional_params:
    # X17, X18 only included in calibration when fast_reservoir_release == 'threshold'
    X17: 'fast_reservoir_release_threshold'
    X18: 'fast_reservoir_release_threshold'
```

The `'fast_reservoir_release_threshold'` condition needs to be added to the namelist-condition logic in `spotpy_optimize.py` so X17/X18 are auto-included/excluded based on the config flag.

---

## Phase 3 — Create 5 new configuration files

Per structure (S1 stays as `glogem_subdaily_opt1.yaml`, already exists):

| Structure | New config filename | Settings vs S1 |
|---|---|---|
| **S2** | `glogem_subdaily_opt1_glaciergw.yaml` | `glacier_routing: 'split_to_slow'` |
| **S3** | `glogem_subdaily_opt1_threshold.yaml` | `fast_reservoir_release: 'threshold'` |
| **S4** | `glogem_subdaily_opt1_threshold_glaciergw.yaml` | both S2 + S3 |
| **S5** | `glogem_subdaily_opt2_sphy_faithful.yaml` | `perc_option: 2`, `land_surface_routing: 'direct'` |
| **S6** | `glogem_subdaily_opt2_sphy_faithful_glaciergw.yaml` | S5 + `glacier_routing: 'split_to_slow'` |

All five inherit the rest from `glogem_subdaily_opt1.yaml` (coupled GloGEM TSLA, ERA5 forcing, subdaily melt, SPHY model). Color codes and display names should be distinct so they're identifiable in plots.

---

## Phase 4 — Test catchment selection

**Test catchment**: **2256 Rosegbach @ Pontresina** (~30 km², Swiss alpine, smallest in fleet, ~20% glacierized).

Reasoning:
- Smallest catchment → fastest Raven runs (~3-4 s/run) → fast iteration during debugging
- Already configured (production calibration in flight)
- Glacierized enough to exercise the glacier-GW pathway changes
- Test-catchment convention already established (per memory)

If 2256 works, scale to one more (perhaps 2161 Massa to test on heavier glaciation) before the full Phase 1 launch.

---

## Phase 5 — Smoke test protocol per structure

For each of S2–S6 (S1 already in production):

### 5.1 Generate Raven input files
- Use a temporary test namelist on 2256
- Run the orchestrator with `--skip-calibration --skip-future` to just produce a single forward run with **default parameter values** (the `init` values in default_params.yaml)
- Verify .rvi, .rvp, .rvh, .rvt files are generated without errors

### 5.2 Verify Raven runs to completion
- Invoke Raven on the generated files
- Confirm no `Raven_errors.txt` is produced
- Confirm `Hydrographs.csv` is written with non-NaN values for the full simulation period

### 5.3 Sanity-check the output
- Confirm simulated Q is non-zero and within plausible range (annual mean within ±50% of observed)
- Compare simulated Q time series visually for S1 vs each variant:
  - **S1 vs S2**: S2 should show smoother winter recession (glacier-GW baseflow contribution)
  - **S1 vs S3**: S3 should show modestly flashier peak responses (Q0 threshold activates above storage threshold)
  - **S1 vs S5**: S5 should show different peak shape (sat-excess routes directly to outlet — less buffered)
  - **S2 vs S6**: S6 should combine effects of S2 + S5
- Spot-check the baseflow component: winter Q (Nov–Mar) should differ between S1 and S2/S4/S6 (glacier-GW variants)

### 5.4 Failure handling
- If Raven errors out: read `Raven_errors.txt`, identify the issue (likely undefined parameter, wrong process pairing, or HRU-type conditional issue), fix, re-run
- If output is implausible (e.g., all-zero Q, NaN throughout): debug the .rvi/.rvp emission, likely a process ordering issue
- Document each fix in a smoke-test log

---

## Phase 6 — Integration verification

After all 6 structures smoke-test cleanly on 2256:

### 6.1 Calibration-pipeline integration check
- Run a small SCEUA calibration (~50 iterations, NOT full 3000) on S2 and S3 to verify:
  - Parameter sampling works correctly for the new X16 / X17 / X18 params
  - Conditional inclusion logic in `spotpy_optimize.py` correctly handles `fast_reservoir_release: 'threshold'` to add X17/X18 to the calibrated set
  - The calibration completes and produces a `VERIFIED_best_params.csv`

### 6.2 Run all 6 structures on a second test catchment
- 2161 Massa (heavier glaciation) — confirms structures behave correctly with significant glacier-derived flow
- Same smoke-test protocol

### 6.3 Document smoke-test results
- Append a "Smoke test outcomes" section to `paper_5_methodology.md` summarizing:
  - Wall-clock runtime per structure
  - Any structure-specific calibration parameter issues found
  - Sanity-check outcomes (did each structure produce the expected directional change?)
  - Any deviations from the plan that needed adjustment

---

## Estimated effort

| Phase | Work | Wall clock |
|---|---|---|
| 1 — preprocess_SPHY.py changes | ~80–120 LOC across 4 edits | ~3-4 hours |
| 2 — default_params.yaml + spotpy_optimize.py conditional | ~30 LOC | ~1 hour |
| 3 — Create 5 new config files | Copy + edit | ~30 min |
| 4 — Test catchment setup | Already done (2256 in production) | 0 |
| 5 — Smoke tests on 2256 | 5 × ~30 min iteration | ~3 hours including debugging |
| 6 — Integration verification | 2 × small calibrations + 1 second-catchment smoke | ~3 hours |
| **Total** | | **~10-12 hours of focused work** |

Spread over 2 working days with debugging buffer.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| BASE_THRESH_STOR + BASE_LINEAR both from FAST_RES might cause Raven mass-balance warnings | Medium | Check Raven manual on multi-baseflow-process behavior; if needed, run with `:DebugMode` to see flux accounting |
| `:Flush` removal in S5 might break MASKED_GLACIER routing if the existing conditional set is replaced rather than added to | Medium | Carefully preserve glacier-routing conditionals; test S5 glacier melt accounting first |
| New parameters X17/X18 might confuse the `_get_tied_parameters` logic in `spotpy_optimize.py` | Low | Smoke-test the parameter-template substitution before full calibration |
| The cascade percolation (perc_option=2) was previously paired with `:Flush` — combining with direct routing is genuinely new. The TOPSOIL-fed FAST_RES might fill differently than expected | Medium | Inspect S5 storage time series — TOPSOIL should fill, then percolate down; FAST_RES storage should be smaller than in S1 |
| The Raven model on hydrolinux might be an older version that lacks BASE_THRESH_STOR | Low | Verify Raven version on hydrolinux supports this baseflow algorithm before relying on it |

---

## Open questions before executing

1. **Test catchment choice** — confirm 2256 Rosegbach is the right first choice, or use a smaller/different one?
2. **Parameter bounds for X17 (UZL) and X18 (K0)** — confirm proposed ranges (UZL ∈ [5, 50] mm; K0 ∈ [0.1, 0.5] /day) are acceptable or specify different ones
3. **Naming convention for config files** — confirm the proposed filenames or suggest alternatives
4. **Order of execution** — should I implement S2 first (smallest change, uses existing infrastructure) as a proof-of-concept before tackling S3/S5 (which require new code)? Or implement all the preprocess changes at once then test cell by cell?
