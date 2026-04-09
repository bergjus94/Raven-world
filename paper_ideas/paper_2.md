# Paper 2: Glacier-Hydrology Coupling Uncertainty in Future Streamflow Projections

## Research Questions

### Overarching question

**"How does the choice of glacier-hydrology coupling approach affect streamflow simulations and future projections across a glacierization gradient in High Mountain Asia, and how does this uncertainty compare to meteorological forcing uncertainty?"**

### Specific research questions

**RQ1 — Historical performance:**
How sensitive are historical calibration performance (KGE, NSE) and calibrated parameter values to the choice of glacier-hydrology coupling approach? Do coupling approaches that yield similar streamflow metrics produce different internal process representations (e.g., glacier melt contribution, snow dynamics)?

**RQ2 — Coupling vs forcing uncertainty:**
How does glacier coupling uncertainty compare to meteorological forcing uncertainty (ERA5 vs HAR vs TPHiPr) for historical streamflow, and do these two sources interact? (i.e., does the effect of coupling approach depend on which forcing is used?)

**RQ3 — Uncertainty shift along the glacierization gradient (headline finding):**
How does the relative importance of different uncertainty sources shift as a continuous function of catchment glacier cover fraction? Existing studies (Addor et al. 2014: 6 Swiss catchments; Dolk et al. 2020: 2 UIB catchments) only show a binary "glacierized vs non-glacierized" comparison. With 12 UIB catchments spanning a continuous gradient from <5% (Swat, Astore) to >30% (Shigar, Hunza) glacier cover, this study can for the first time show the continuous relationship between glacier fraction and uncertainty source dominance — and identify the glacier cover threshold above which coupling approach becomes a dominant uncertainty source.

**RQ3b — Glacier size distribution:**
Does the glacier size distribution within a catchment modulate the sensitivity of streamflow projections to the coupling approach? Catchments with the same glacier fraction can have very different glacier populations (few large glaciers vs many small ones). Small glaciers respond faster to climate change, pass peak water sooner, and disappear entirely within the projection period. The uncoupled model calibrates glacier parameters as catchment-wide values (implicitly assuming all glaciers behave similarly), while GloGEM models each glacier individually — does this difference matter more in catchments dominated by small glaciers? Catchments are characterized by median glacier size, fraction of glacier area from glaciers < 0.5 km², and number of glaciers, in addition to total glacier fraction.

**RQ4 — Future divergence:**
Do coupling approaches that perform similarly in the historical period diverge under future climate forcing? How large is this divergence relative to GCM forcing uncertainty for future streamflow projections?

**RQ5 — Value of external glacier information:**
Does external coupling with GloGEM (which is calibrated on geodetic mass balance at global scale) provide equivalent streamflow improvement to multi-objective calibration of the uncoupled model with elevation-band geodetic mass balance data? If so, external coupling offers the same constraint without requiring catchment-level mass balance data — what are the practical implications for glacio-hydrological modeling in data-scarce regions?

---

## Research Gap

### What exists

- **ANOVA decomposition** of GCM × RCP × downscaling × hydro model structure (Bosshard et al. 2013, Addor et al. 2014)
- **Glacier model intercomparisons** for runoff (Wimberly et al. 2025, Ultee et al. 2026) — compare GloGEM vs OGGM vs PyGEM, but not how they're coupled to hydro models
- **Coupled vs uncoupled** comparisons — each study tests ONE coupling approach vs no coupling (GloGEM–PCR-GLOBWB, OGGM–CWatM, WaSiM–OGGM)
- **Parameter uncertainty amplification** in glaciated UIB catchments (Dolk et al. 2020) — but with internal glacier model only
- **Review of 145 glacio-hydro studies** (Tiel et al. 2020) — found no systematic comparison of coupling approaches

### What doesn't exist

1. **Multiple coupling approaches compared side-by-side** in the same hydrological framework, same catchments, same forcing
2. **Glacier coupling method as an explicit uncertainty factor** in a decomposition alongside GCM/forcing uncertainty
3. **Continuous relationship between glacier cover and uncertainty dominance** — Addor et al. (2014) showed hydro model matters more in glacierized Swiss catchments and Dolk et al. (2020) showed the same for 2 UIB catchments, but both are binary comparisons (glacierized vs not) with very few catchments. Nobody has shown how uncertainty sources shift *continuously* across a glacierization gradient — is it linear? Is there a threshold? Does coupling uncertainty plateau above a certain glacier fraction? 12 catchments spanning <5% to >30% glacier cover can answer this for the first time.
4. **Analysis of how glacier size distribution modulates coupling uncertainty** — small glaciers (~80% by count in HMA) are where uncoupled models with catchment-wide glacier parameters are most likely to fail, and where the advantage of individually-modeled glaciers (GloGEM) should be largest. No study has tested this.

---

## Our Unique Setup

### Four coupling approaches in Raven

| Approach | Key | Glacier representation | Calibration data | Calibration freedom |
|---|---|---|---|---|
| **Uncoupled** | `baseline` | Raven internal degree-day + Δh | Streamflow only | Full — glacier params calibrated with hydro params |
| **Uncoupled + MB** | `baseline_mb` | Raven internal degree-day + Δh | Streamflow + geodetic MB per elevation band | Constrained — glacier params must match observed mass balance profile |
| **GloGEM TSLA** | `glogem` | External transient snowline altitude forcing | Streamflow only | Reduced — glacier melt prescribed by GloGEM |
| **GloGEM GMB** | `glogem_gmb` | External glacier mass balance forcing | Streamflow only | Reduced — different coupling variable |
| **Icemelt** | `icemelt` | External melt injection, glacier HRUs as ROCK | Streamflow only | Minimal — glacier processes fully external |

### Multi-objective calibration for uncoupled + MB variant

**Reviewer-motivated addition:** To ensure a fair comparison between uncoupled and coupled approaches, a multi-objective calibration variant is included that constrains glacier parameters against observed geodetic mass balance.

- **Data source:** Elevation-dependent geodetic mass balance rates from Hugonnet et al. (2021), aggregated to Raven's glacier elevation bands
- **Why per elevation band, not catchment average:** A single catchment-average MB (e.g., -0.5 m w.e./yr across 2000 glaciers) is too weak — it prevents extreme equifinality but doesn't constrain the elevation distribution of melt or temperature sensitivity/lapse rate. Per-elevation-band targets (4–6 bands) constrain the mass balance *profile*, which is where the real equifinality lives in uncoupled models.
- **Calibration approach:** Multi-objective SPOTPY calibration combining streamflow (KGE) + elevation-band geodetic mass balance (RMSE or similar)
- **Purpose:** This creates a gradient of "how much glacier information enters the model":

  | Approach | Glacier information source | Modeler effort |
  |---|---|---|
  | Uncoupled (streamflow only) | None — glacier params from streamflow equifinality | Low |
  | Uncoupled + MB calibration | Direct — geodetic MB per elevation band | High (requires data + multi-obj setup) |
  | GloGEM TSLA/GMB/Icemelt | Indirect/direct — via GloGEM (already calibrated on MB at global scale) | Low (no additional catchment-level calibration data needed) |

- **Expected finding:** Multi-objective calibration and external coupling achieve similar improvements over uncoupled-streamflow-only, but coupling requires no additional calibration data at the catchment level. This demonstrates the practical value of external coupling — it provides mass balance constraint "for free."

### Additional uncertainty dimensions

- **Meteorological forcing:** ERA5 vs HAR vs TPHiPr (3 datasets)
- **PET method:** ERA5-derived vs Oudin
- **Catchments:** 12 across the UIB, ranging from heavily glacierized (Shigar, Hunza) to low glacierization (Swat, Astore)
- **Future climate:** Multiple GloGEM climate models for projections

### Factorial design

- 5 coupling approaches × 3 forcings × 12 catchments for historical (including uncoupled + MB variant)
- Future projections with multiple GCMs per coupling approach
- Both KGE and NSE calibrations available

---

## Key Literature

### Coupling studies (each uses ONE approach)

- **GloGEM–PCR-GLOBWB 2 (HESS 2022)** — https://hess.copernicus.org/articles/26/5971/2022/
  Global scale. Coupling prevents underestimation of glacier runoff. Uncoupled vs coupled diverge for future but not historical.

- **OGGM–CWatM (GMD 2024)** — https://gmd.copernicus.org/articles/17/5123/2024/
  Global scale framework coupling at 5 and 30 arcmin resolution.

- **WaSiM–OGGM (Frontiers in Water 2023)** — https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2023.1296344/full
  Catchment scale. VA scaling and OGGM coupling agree historically but diverge massively for future: 10–19% vs 26–41% peak reduction. Closest to what we do but only 2 approaches in 1 catchment.

- **SWAT-GL (HESS 2025)** — https://hess.copernicus.org/articles/29/3227/2025/hess-29-3227-2025.pdf
  SWAT with glacier module, evaluation of merits and limits.

### Review

- **Tiel, Stahl, Freudiger & Seibert (2020)** *WIREs Water* — https://wires.onlinelibrary.wiley.com/doi/10.1002/wat2.1483
  Reviewed 145 glacio-hydro studies. Most use internal glacier representations (degree-day + VA scaling). No systematic comparison of coupling approaches found.

### Uncertainty decomposition (foundations)

- **Bosshard et al. (2013)** *WRR* — https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2011WR011533
  ANOVA framework. GCM-RCM dominates, but hydro model contribution is considerable in glacierized catchments.

- **Addor et al. (2014)** *WRR* — https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2014WR015549
  Swiss catchments. In partially glacierized catchments, hydro model explained comparable variance to GCM-RCMs.

### HMA-specific

- **Dolk, Penton & Ahmad (2020)** *Hydrological Processes*, 34, 2200–2218 — https://onlinelibrary.wiley.com/doi/abs/10.1002/hyp.13718
  Upper Indus, 3 hydro models, 2 catchments. **Parameter selection was the most significant uncertainty source in the glaciated catchment** and amplified climate model uncertainty. For one projection, parameter choice gave +54% to +125% future streamflow range. Great justification for fixing glacier melt from an external glacier model — removes the amplification effect.

- **Nair et al. (2025)** *Frontiers in Water* — https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2025.1611141/full
  Climate downscaling effects on hydrologic response in glacierized HMA catchments.

- **Immerzeel et al. (2020)** *Science* — https://www.science.org/doi/10.1126/science.abf3668
  Glaciohydrology of the Himalaya-Karakoram. Data gaps in upper reaches are primary uncertainty source.

### Glacier model intercomparisons

- **Wimberly et al. (2025)** *The Cryosphere* — First intercomparison of glacier runoff (GloGEM, OGGM, PyGEM) for 75 basins. Absolute runoff differs up to 3.8×, but normalized percent change agrees better.

- **Ultee et al. (2026)** *The Cryosphere* — GCM spread exceeds glacier model spread by factors of 1.7–133.

### Calibration metric / objective function

- **Khatami et al. (2023)** *HSJ* — https://www.tandfonline.com/doi/full/10.1080/02626667.2023.2231434
  12 objective functions across diverse catchments. Metric choice matters more in drier catchments.

- **Höge et al. (2023)** *HESS* — https://hess.copernicus.org/articles/27/2397/2023/
  Critical evaluation of performance criteria — metric choice shapes parameter identifiability.

---

## Study Design

### Factorial design

Two-factor design per catchment: **Coupling approach × Meteorological forcing**

|  | ERA5 | HAR | TPHiPr |
|---|---|---|---|
| **Uncoupled** (Raven internal) | `baseline` | `har` | `tphipr` |
| **Uncoupled + MB** (Raven + geodetic MB calibration) | `baseline_mb` | `har_mb`* | `tphipr_mb`* |
| **GloGEM TSLA** (external snowline) | `glogem` | `glogem_har` | `glogem_tphipr` |
| **GloGEM GMB** (external mass balance) | `glogem_gmb` | `glogem_gmb_har`* | `glogem_gmb_tphipr`* |
| **Icemelt** (external melt injection) | `icemelt` | `icemelt_har`* | `icemelt_tphipr`* |

*Runs marked with \* are currently missing and would be needed for a fully balanced design. Without them, options are: (a) add the missing configs × 12 catchments, (b) use unbalanced ANOVA (QE-ANOVA), or (c) restrict to the complete sub-factorial of available runs.*

Applied across **12 catchments** spanning a glacierization gradient in the Upper Indus Basin (UIB).

### Response variables (future change signal)

Not raw future Q, but the **change signal** (future minus/over historical):
- % change in mean annual discharge
- % change in seasonal discharge (DJF, MAM, JJA, SON)
- Shift in peak flow timing (days)
- Change in low-flow quantile (Q95)
- Change in high-flow quantile (Q5)

### Variance decomposition (two-way ANOVA)

Following the framework of Bosshard et al. (2013), decompose variance in the change signal **per catchment** into:

1. **Main effect: Coupling** — how much does the glacier coupling approach matter?
2. **Main effect: Forcing** — how much does the meteorological forcing matter?
3. **Interaction: Coupling × Forcing** — does the coupling effect depend on which forcing is used? (i.e., does fixing glacier melt externally reduce sensitivity to forcing uncertainty?)
4. **Residual**

Express as **fraction of total variance** per term.

### Key analysis

**Headline figure — uncertainty shift along glacierization gradient:**
Plot the ANOVA variance fractions (coupling / forcing / interaction / residual) across the 12 catchments, ordered by glacier cover fraction on the x-axis. This produces a continuous curve showing how the uncertainty composition shifts from forcing-dominated (low glacier cover) to coupling-dominated (high glacier cover) — or reveals a threshold effect, or shows that the relationship is non-linear. This is the central novel result: the first continuous mapping of uncertainty source dominance vs glacier fraction, extending the binary findings of Addor et al. (2014) and Dolk et al. (2020) to a gradient.

Additionally, test whether the interaction term (coupling × forcing) grows with glacier cover — i.e., does the choice of coupling approach become more consequential depending on which forcing is used, and is this effect stronger in more glacierized catchments?

Second axis: characterize each catchment not only by total glacier fraction but also by **glacier size distribution** (median glacier area, fraction of glacier area from glaciers < 0.5 km², total glacier count). This tests whether catchments dominated by many small glaciers show different coupling sensitivity than catchments with fewer large glaciers — even at similar total glacier fractions. The hypothesis is that uncoupled models (which apply catchment-wide glacier parameters) underperform most in small-glacier-dominated catchments, while GloGEM (which models each glacier individually) is relatively insensitive to size distribution.

### Why NOT include hydrological model structure

- **Practical cost:** Adding even 2 model structures (e.g., HBV + HMETS) would mean 4 × 3 × 2 = 24 configs × 12 catchments = 288 runs, each needing separate SPOTPY calibration
- **Focus:** The novelty is coupling approach as an uncertainty source — adding model structure dilutes this into "yet another comprehensive uncertainty study"
- **Already covered in literature:** Model structure uncertainty is well-quantified (Bosshard 2013, Addor 2014, Dolk 2020)
- **Framing:** Acknowledge as limitation, cite existing evidence, position as future work

### Expected outputs

1. **Historical performance comparison** — KGE/NSE heatmaps and boxplots across coupling × forcing × catchment (already implemented in `postprocessing_catchments.py`)
2. **Future change signal matrices** — per coupling × forcing × catchment
3. **ANOVA variance partition plots** — stacked bars or pie charts per catchment showing coupling / forcing / interaction contributions
4. **Glacierization gradient plot** — variance fractions vs glacier area fraction, the headline figure
5. **Glacier size distribution plot** — variance fractions vs median glacier size / small-glacier fraction, testing whether coupling uncertainty depends on the glacier population structure (second headline figure)
6. **Hydrograph/regime divergence plots** — showing where coupling approaches agree/disagree under future forcing

---

## Contribution

This would be the **first study to**:
1. Compare multiple glacier-hydrology coupling approaches side-by-side in the same modelling framework
2. Treat glacier coupling method as an explicit factor in uncertainty decomposition alongside meteorological forcing
3. Map the continuous relationship between glacier cover fraction and uncertainty source dominance across 12 catchments — extending the binary "glacierized vs not" findings of Addor et al. (2014) and Dolk et al. (2020) to a gradient, and identifying critical glacier cover thresholds
4. Show whether fixing glacier melt externally (as justified by Dolk et al. 2020) actually reduces or reshapes total projection uncertainty
5. Demonstrate whether external coupling provides equivalent constraint to multi-objective calibration with geodetic mass balance — without requiring catchment-level mass balance data
