# Research Ideas — Upper Indus Glacio-Hydrology

## Setup
- 30 catchments with streamflow data (10–20 years each, not all same period)
- Raven model with 4 structures: HBV, HYMOD, HMETS, MOHYSE
- GloGEM glacier runoff for 5 ISIMIP3b GCMs
- Delta-H glacier geometry method in Raven (coupled vs uncoupled)
- Meteo forcings: ERA5-Land, HAR, TPHiPr
- Climate projections: CMIP6 (5 GCMs) + CORDEX with QDM bias correction

---

## Idea 1: Multi-Structure Precipitation Back-Estimation (top pick)

Use 4 model structures x 3 forcings x 30 catchments to constrain the true precipitation field from streamflow. Where all structures agree a forcing is biased, that's robust. Where they disagree, that's structural uncertainty. GloGEM glacier melt can be subtracted from total discharge to isolate rain+snowmelt, making the inversion more constrained.

**Key reference:** "Can discharge be used to inversely correct precipitation?" (HESS 2025, https://hess.copernicus.org/articles/29/6115/2025/) — used LSTMs in non-glacierized catchments. Our approach: process-based, glacierized, multi-structure.

**Target journals:** Nature Water, HESS

---

## Idea 2: Model Structure as Process Diagnostic Across Glacier Gradient

Not "which model is best" but "where does each model fail and why?" Run 4 structures across 30 catchments spanning 0–30%+ glacier cover. Diagnose whether failure patterns correlate with glacier coverage, elevation, or precipitation regime (westerlies vs monsoon). MOHYSE was only formally published in 2025 and has never been tested in glacierized terrain. Model disagreement itself is the scientific result.

**Key reference:** MOHYSE published in Canadian J. Civil Eng 2025 (https://www.tandfonline.com/doi/full/10.1080/07011784.2025.2536023)

**Target journals:** WRR, HESS

---

## Idea 3: Non-Stationarity Detection Linked to Glacier Change

Calibrate each structure on early vs late period (or wet vs dry years via DSST), test whether parameter drift correlates with glacier coverage or Karakoram anomaly status. Multi-structure angle: does HBV drift differently than HYMOD?

**Limitation:** 10–20 year records are tight for time splits. DSST (wet/dry years) is more feasible. Weakest of the three ideas given data constraints.

**Key reference:** Karakoram anomaly persistence (Geo-spatial Info Science 2025, https://www.tandfonline.com/doi/full/10.1080/15481603.2025.2548059)

**Target journals:** WRR, The Cryosphere

---

## Other Angles Considered (less novel)
- Forcing uncertainty decomposition (ANOVA) — done by von der Esch et al. (HESS 2025) for 1 catchment/1 model, but the general approach is well-trodden
- Coupled vs uncoupled glacier comparison — interesting but more of a methods paper
- Peak water mapping at sub-catchment scale — incremental over existing basin-scale studies
- Dense streamflow network value — interesting but more of a data paper
- Differentiable parameter learning (dPL) for glacierized regionalization — novel but requires ML infrastructure

---

*Last updated: 2026-03-09*
