# Literature: Uncertainty Propagation in Glacio-Hydrological Modeling

## Key Hierarchy of Uncertainty Sources

The literature converges on a consistent ranking:

1. **GCM choice** — consistently the largest source, especially for precipitation-dependent metrics and longer horizons
2. **Emission scenario (RCP/SSP)** — becomes increasingly dominant post-2060
3. **Reference/historical climate data** — underappreciated but potentially dominant (Aguayo et al. 2024)
4. **Hydrological model structure** — matters more in glacierized/high-elevation catchments
5. **Glacier model choice** — large absolute differences but better agreement on relative changes
6. **Glacier model parameters** — significant at individual glacier scale, less so regionally
7. **Natural climate variability** — meaningful near-term (20-30 yr), overwhelmed at longer horizons

---

## Foundational / Methodological Papers

- **Bosshard et al. (2013)** *Water Resources Research* — ANOVA framework decomposing uncertainty from GCM-RCMs, postprocessing, and hydrological models in Swiss Alpine catchments. GCM-RCM dominates, but hydrological model contribution is considerable in glacierized catchments.

- **Addor et al. (2014)** *Water Resources Research* — Factorial design across Swiss catchments. Hydrological model uncertainty increases with elevation/glacierization. Emission scenario dominates by end of century.

- **Schaefli et al. (2007)** *HESS* — Early full modeling chain (climate → discharge → glacier → hydropower) for a Swiss glacierized catchment.

- **Finger et al. (2012)** *Water Resources Research* — ANOVA decomposition for Vispa valley, Switzerland. Despite large uncertainty, consistent trends (spring increase, summer decline) are robust.

---

## High Mountain Asia

- **Immerzeel et al. (2020)** *Nature* — Identified Indus as the most important and vulnerable "water tower" globally. Climate models don't adequately capture monsoon-westerly dynamics.

- **Lutz et al. (2014)** *Nature Climate Change* — Projected runoff for Indus, Ganges, Brahmaputra, Salween, Mekong using 8 GCMs. Consistent runoff increases until ~2050 despite large inter-GCM precipitation spread.

- **Rounce et al. (2020a)** *Journal of Glaciology* — Bayesian calibration of PyGEM for all HMA glaciers. Climate forcing dominates at regional scale; parameter uncertainty matters at individual glacier scale.

- **Rounce et al. (2020b)** *Frontiers in Earth Science* — HMA glacier mass loss 29±12% (RCP2.6) to 67±10% (RCP8.5) by 2100. Monsoon-fed basins reach peak water before 2050; westerlies-fed basins (Indus) after 2050.

- **Dolk, Penton & Ahmad (2020)** *Hydrological Processes*, 34, 2200–2218. DOI: 10.1002/hyp.13718 — Three hydrological models in two Upper Indus catchments. **Parameter selection was the most significant uncertainty source in the glaciated catchment** and amplified climate model uncertainty, whereas GCM choice dominated in the rainfall-dominated catchment. For one climate projection, the choice among good parameter sets resulted in projected future streamflow ranging from +54% to +125%. Recommend presenting results from at least the bookends: models with low vs high sensitivity to ice-melt. **This is a great justification for fixing glacier melt from an external glacier model (e.g., GloGEM) — removing glacier parameter uncertainty from the hydrological model reduces the amplification effect they describe.**

- **Kraaijenbrink et al. (2017)** *Nature* — Even 1.5°C warming (→2.1°C in HMA) preserves only 64% of ice.

- **Nie, Pritchard et al. (2021)** *Nature Reviews Earth & Environment* — Heterogeneous retreat patterns create large runoff uncertainty; debris-covered glacier processes and precipitation are critical knowledge gaps.

---

## Glacier Model Intercomparisons

- **Hock et al. (2019)** *Journal of Glaciology* — GlacierMIP: 6 global glacier models × 25 GCMs × 4 RCPs. Large inter-model differences especially at regional scales.

- **Wimberly et al. (2025)** *The Cryosphere* — First intercomparison of glacier *runoff* (GloGEM, OGGM, PyGEM) for 75 basins. Absolute runoff differs up to 3.8x, but normalized percent change agrees much better. Peak water timing: GCM forcing (40-yr range) vs glacier model choice (6-yr range).

- **Ultee et al. (2026)** *The Cryosphere* — GCM spread exceeds glacier model spread by factors of 1.7–133. The commonly used 11-member GCM ensemble undersamples CMIP6 spread.

---

## Other Regions

- **Aguayo et al. (2024)** *The Cryosphere*, 18, 5383–5406. DOI: 10.5194/tc-18-5383-2024
  - **Authors:** Aguayo, R., Maussion, F., Schuster, L., Schaefer, M., Caro, A., Schmitt, P., Mackay, J., Ultee, L., Leon-Munoz, J., and Aguayo, M.
  - **Region:** Patagonian Andes (40–56°S), 329 catchments, every glacier >1 km²
  - **Model:** OGGM (Open Global Glacier Model)
  - **Design:** Full factorial, 1,920 scenarios = 2 glacier inventories (RGI6, RGI7) × 2 ice thickness datasets (Farinotti 2019, Millan 2022) × 4 reference climate datasets (ERA5, CR2MET v2.5, MSWEP/MSWX, PMET v1.0) × 10 CMIP6 GCMs × 4 SSPs × 3 bias correction methods (MVA, QDM, MBCn)
  - **Method:** Random forest regression with permutation feature importance (not classical ANOVA) to decompose variance across the 6 uncertainty sources for 10 glacio-hydrological signatures
  - **Key finding:** Reference climate dataset was the **dominant** uncertainty source in **69% ± 22%** of glacier area. Future sources (GCMs + SSPs + bias correction) dominated in only 17% of glacier area.
  - **Why reference climate dominates:** The 4 datasets disagreed massively on precipitation — ERA5 overestimated solid precip by +47% vs PMET, MSWEP underestimated by -56%. These differences propagate through calibration into different temperature sensitivity parameters, initial glacier geometries, and baselines.
  - **Other results:** 34% of glacier area has already passed peak water. Volume loss by 2100: 46 ± 9% (SSP1-2.6) to 67 ± 11% (SSP5-8.5). GCMs/SSPs only dominate for long-term trend/change metrics.
  - **Implication:** For medium-term projections (to ~2050), improving reference climate data yields greater uncertainty reduction than expanding GCM ensembles. After 2050, emission scenario divergence takes over.
  - **Code:** https://doi.org/10.5281/zenodo.14177951 | **Data:** https://doi.org/10.5281/zenodo.11353065

- **Ragettli et al. (2016)** *PNAS* — Contrasting Andes vs Himalaya: similar glacier area decreases but remarkably different hydrological response.

- **Huss & Hock (2018)** *Nature Climate Change* — Global-scale: about half of 56 basins haven't yet reached peak water. By 2100, one-third may see >10% melt-season runoff decrease.

---

## Downscaling / Bias Correction Uncertainty

### Foundational / Methodological

- **Teutschbein & Seibert (2012)** *J. Hydrology* — Compared 6 bias correction methods (linear scaling, local intensity scaling, power transformation, variance scaling, distribution mapping, quantile mapping) for Swedish catchments. Quantile mapping performed best overall; method choice significantly affected simulated extremes.

- **Cannon et al. (2015)** *J. Climate* — Introduced quantile delta mapping (QDM). Showed standard quantile mapping can **distort the climate change signal**, while trend-preserving methods (QDM, scaled distribution mapping) preserve it.

- **Maraun (2016)** *Current Climate Change Reports* — Argued bias correction cannot fix fundamental GCM process errors; warns against over-reliance on statistical post-processing.

- **Gutmann et al. (2014)** *Water Resources Research* — Compared BCSD, BCCA, delta method across CONUS. Precipitation extremes differed by a **factor of 2–3** between methods.

### Quantified Impact on Future Hydrological Projections

#### Variance decomposition studies (% of total projection uncertainty)

| Study | Region | Downscaling/BC contribution | Notes |
|---|---|---|---|
| **Chegwidden et al. (2019)** *Earth's Future* | Columbia River | **20–30%** of total variance | Sometimes exceeded GCM as dominant source for winter runoff timing |
| **Bosshard et al. (2013)** *Water Resources Research* | Swiss Alps | **5–25%** | Up to ~25% for summer discharge when melt matters most |
| **Vetter et al. (2017)** *Climatic Change* | Multi-catchment global | **5–40%** | Up to ~40% for low-flow indicators |
| **Muerth et al. (2013)** *Climatic Change* | Canadian catchments | **10–20%** | Dominant source for timing of spring flood |

#### Absolute differences in projections between methods

| Study | Region | How much projections differ |
|---|---|---|
| **Hagemann et al. (2011)** *J. Hydrometeorology* | Global | Changed the **sign** of projected runoff change in some regions |
| **Chen et al. (2011)** *J. Hydrology* | Canada | Projected change ranged from -5% to -20% depending on method; seasonal differences **1.5–2x** |
| **Mpelasoka & Chiew (2009)** *J. Hydrology* | Australia | **5–15 percentage points** difference between methods |
| **Teng et al. (2015)** *J. Hydrology* | Australia | **10–25 percentage points** difference; larger for drier catchments |
| **Grillakis et al. (2017)** *Climatic Change* | Mediterranean | **10–20 pp** for mean annual runoff; **15–30 pp** for extremes |

### Statistical vs Dynamical Downscaling

- **Fowler et al. (2007)** *Int. J. Climatology* — Statistical and dynamical downscaling can produce divergent precipitation projections with differences of 10–30% in seasonal precipitation that amplify through hydrological models.

- **Giorgi & Gutowski (2015)** *Annual Review of Environment and Resources* — RCMs add value in complex terrain but also introduce their own biases; unclear whether dynamical downscaling consistently reduces uncertainty vs statistical methods for hydrology.

### Summary

Downscaling/bias correction is typically a **secondary** uncertainty source (5–30% of total variance), below GCM choice and emission scenario. However:
- It can be **dominant** for specific metrics: low flows, flood timing, seasonal shifts
- It can change the **sign** of projected change (Hagemann et al. 2011)
- In glacierized mountain catchments it is **amplified** because temperature biases propagate nonlinearly through melt (Bosshard et al. 2013 showed up to 25% for summer discharge in the Alps)
- It is likely even more important in data-sparse regions like HMA where precipitation gradients are steep and poorly observed

---

## Raven-Specific

- **Jabbari et al. (2024, preprint)** — Raven HBV-EC ensemble of model structures, assessing model structure + input data uncertainties (Canadian basins).

- **Lovekin et al. (2021)** *Hydrological Processes* — Raven coupled with glacier model for Columbia River headwaters, Canada.

---

## Additional References: Reference Climate Uncertainty in HMA

Studies that document the problem but **do not** propagate it through models into future projections:

- **Dahri et al. (2016, 2018)** *Science of the Total Environment* / *Int. J. Climatology* — Compared multiple gridded precipitation products against adjusted station observations in the upper Indus. Products differ by **factors of 2–5x** at high elevations. Most products severely underestimate high-altitude precipitation.

- **Immerzeel et al. (2015)** *HESS* — Used glacier mass balance to infer precipitation must be **2–10x higher** than gridded products at high elevations in the upper Indus. Inverse approach, not a systematic model propagation study.

- **Palazzi et al. (2013)** *J. Geophys. Res. Atmospheres* — Compared observational/reanalysis precipitation over HKH. Large spread between products.

Closest to a systematic decomposition:

- **Lutz et al. (2016)** *J. Hydrometeorology* — "Climate change impacts on the upper Indus hydrology: Sources of uncertainty and its relative importance." Used SPHY and decomposed uncertainty from reference climate, GCMs, RCPs, and parameters. **Closest existing work**, but used a merged/corrected reference dataset rather than running the full chain with each product independently (unlike Aguayo's approach).

- **Wortmann et al. (2018)** *J. Hydrometeorology* — Forced the same glacio-hydrological model (SWIM) with multiple precipitation products in the Tien Shan. **Systematic multi-product forcing**, but did not propagate into future projections.

**Assessment:** Nobody has formally connected "products disagree by 2–5x in HMA" with "here's how much that changes your 2100 projections" — despite the upper Indus arguably having the largest precipitation uncertainty anywhere on Earth.

---

## Research Gaps

### Gap 1: No full uncertainty decomposition for glacierized HMA catchments

Aguayo et al. (2024) did this for Patagonia, Bosshard (2013) and Addor (2014) for the Swiss Alps — but no equivalent study exists for the Indus/HKH, despite it being the world's most important and vulnerable water tower (Immerzeel et al. 2020). The existing closest work (Lutz et al. 2016) uses a single merged reference climate and does not systematically vary hydrological model structure.

### Gap 2: Hydrological model structure uncertainty in glacierized HMA catchments

Bosshard (2013) and Addor (2014) showed that hydrological model structure uncertainty is **larger in glacierized/high-elevation catchments** than in lowland ones. Yet all HMA studies use a single hydrological model (SPHY, TOPKAPI-ETH, VIC, etc.). No study has tested multiple hydrological model structures in the same glacierized HMA catchment. Raven's flexibility (HBV, HMETS, HYMOD, MOHYSE) is uniquely suited to fill this gap.

### Gap 3: Interaction between glacier forcing and hydrological model structure

Existing studies either examine glacier model uncertainty (GlacierMIP, Wimberly et al. 2025) OR hydrological model uncertainty (Bosshard, Addor), but almost nobody examines how uncertainties **interact** when a glacier model is coupled into a hydrological model. Does the choice of hydrological model amplify or dampen the glacier melt signal? Does it matter more for some metrics (e.g., low flows, seasonality) than others?

### Gap 4: Reference climate uncertainty propagation in HMA

Dahri et al. (2016, 2018) documented that precipitation products disagree by 2–5x in the upper Indus. Aguayo et al. (2024) showed that reference climate dominates projection uncertainty in Patagonia. But nobody has done the Aguayo-style propagation study for HMA, where the problem is arguably worse. (Note: this would require re-running GloGEM with different forcings, which is not currently feasible with a single GloGEM calibration.)

### Gap 5: Scale dependence of uncertainty sources

Most decomposition studies work at either the individual glacier scale (Rounce 2020) or large basin scale (Lutz 2014, Huss & Hock 2018). How do the relative contributions shift at the sub-catchment scale (~500–5000 km²) that is relevant for water management? The upper Indus sub-catchments (0101, 0102, 0118, 0130) with different glacier coverage fractions could reveal this.

### Gap 6: Temporal evolution of uncertainty dominance

Which source of uncertainty dominates changes over time (natural variability near-term → GCM mid-century → SSP late-century). But this has not been shown for glacierized HMA catchments where the glacier contribution itself is non-stationary (peak water). How does the declining glacier signal interact with the growing emission scenario divergence?

---

## Proposed Experimental Design (given current setup)

### What can be varied

| Source | Options | n |
|---|---|---|
| Hydrological model structure | HBV, HMETS, HYMOD, MOHYSE | 4 |
| GCMs (ISIMIP3b) | GFDL-ESM4, IPSL-CM6A-LR, MPI-ESM1-2-HR, MRI-ESM2-0, UKESM1-0-LL | 5 |
| SSPs | SSP1-2.6, SSP5-8.5 | 2 |
| Calibration parameters | Behavioral parameter sets from SCEUA (e.g., top 5–10) | 5–10 |
| Catchments | 0101, 0102, 0130 (+ server catchments) | 3–8 |

### What is held constant (and justification)

| Source | Value | Justification |
|---|---|---|
| Glacier model (GloGEM) | 1 calibration | Wimberly et al. (2025) and Ultee et al. (2026) showed GCM spread outweighs glacier model spread; peak water timing uncertainty from GCM forcing is 40 years vs 6 years from glacier model choice |
| Reference climate | ERA5 | Single product available; acknowledged as limitation, motivates future work |
| Bias correction | Method used in pipeline | Could be varied but adds complexity |

### Factorial design

**Minimum design:** 4 models × 5 GCMs × 2 SSPs = **40 future runs per catchment**
- Plus 4 calibrations per catchment (~3 hours each)
- Total per catchment: ~12h calibration + ~40h forward runs ≈ **~52 hours**
- For 3 catchments: **~156 hours (~6.5 days)**

**Extended design (with parameter uncertainty):** 4 models × 5 GCMs × 2 SSPs × 5 parameter sets = **200 runs per catchment**
- Total per catchment: ~12h calibration + ~200h forward runs ≈ **~212 hours (~9 days)**
- For 3 catchments: **~636 hours (~26 days)**
- Note: forward runs are embarrassingly parallel — with 5 cores, wall-clock time drops to ~42h per catchment

### Variance decomposition method

Following Bosshard et al. (2013): **ANOVA decomposition** of total variance into:
- Main effects: hydrological model structure, GCM, SSP, (calibration parameters)
- Interaction effects: model×GCM, model×SSP, GCM×SSP

Or following Aguayo et al. (2024): **Random forest with permutation feature importance** — more flexible, handles non-linear interactions better.

### Target metrics (analogous to Aguayo's 10 signatures)

1. Mean annual discharge (reference period + future periods)
2. Peak water year and magnitude
3. Seasonal cycle shift (monthly climatology change)
4. Low flow magnitude and timing (Q95)
5. High flow magnitude (Q5)
6. Inter-annual variability
7. Glacier melt contribution to total runoff (%)
8. Long-term trend (linear)
9. Change in monsoon vs winter flow ratio
10. Drought buffering capacity (dry-season glacier contribution)

### Research questions this design can answer

1. **Does hydrological model structure matter more in glacierized HMA catchments than in the Swiss Alps?** (Compare relative contribution to Bosshard 2013 / Addor 2014)
2. **Does model structure uncertainty grow or shrink in future projections?** (As glacier contribution declines, does it matter less which model you use?)
3. **Which metrics are most sensitive to model structure vs GCM choice?** (Seasonal timing? Low flows? Annual totals?)
4. **Does glacier coverage fraction modulate the uncertainty hierarchy?** (Compare across catchments with different glacier fractions)
5. **Is the uncertainty hierarchy in HMA different from the Alps/Patagonia?** (Monsoon dynamics, extreme elevation range, debris cover)

### What this does NOT cover (acknowledged limitations)

- Glacier model uncertainty (single GloGEM run) — cite Wimberly/Ultee to justify
- Reference climate uncertainty — cite Dahri/Aguayo, flag as critical future work
- Downscaling/bias correction uncertainty — could be added later
- Ice thickness / glacier inventory uncertainty — held constant

---

## Novel / "Out of the Box" Research Gaps

### Gap 7: Parameter non-stationarity in glacierized catchments

Models calibrated on historical data (e.g., 1980–2014) assume parameters remain valid in a fundamentally different future climate. As glaciers retreat, the system shifts from melt-dominated to rain-dominated — a regime change that fixed parameters cannot represent.

**Key literature:**
- **Merz et al. (2011)** *Water Resources Research* — Parameters calibrated on different sub-periods vary significantly, questioning their use for future projections.
- **Coron et al. (2012)** *Water Resources Research* — "Crash testing hydrological models in contrasted climate conditions" using differential split-sample tests on 216 Australian catchments. Significant performance degradation when transferring parameters across contrasting climates.
- **Broderick et al. (2016)** — Equifinal parameter sets that perform identically on historical data can **diverge by 20–40%** in future projections.

**Gap:** Almost all non-stationarity work is in temperate/Australian catchments. For glacierized HMA catchments where the regime change is arguably most extreme, this is essentially unexplored.

**Feasibility:** High — re-calibrate Raven on contrasting sub-periods (high-melt vs low-melt years) and show parameter divergence. Combine with equifinality analysis (multiple behavioral parameter sets from SCEUA) to show how they fan out for 2100.

### Gap 8: Signal-to-noise ratio in the glacio-hydrological uncertainty cascade

At what point does the projection signal disappear into the noise of accumulated uncertainties? The Hawkins & Sutton (2009, 2011) framework partitions total uncertainty into model uncertainty, scenario uncertainty, and internal variability at different time horizons — but this has never been applied to glacierized catchment runoff.

**Key literature:**
- **Hawkins & Sutton (2009, 2011)** *Bull. Amer. Meteorol. Soc.* — Classic decomposition showing different uncertainty sources dominate at different time horizons for temperature/precipitation.
- **Clark et al. (2016)** *Current Climate Change Reports* — Extended to the hydrology chain; found hydrological model uncertainty can dominate GCM uncertainty for some variables.

**Gap:** Nobody has applied the Hawkins-Sutton framework to the GCM→downscaling→GloGEM→Raven cascade for glacierized catchments. For which variables and time horizons does a meaningful signal survive?

**Feasibility:** High — the multi-model ensemble infrastructure already exists. Apply ANOVA time-slice decomposition to the existing runs.

### Gap 9: Decision-relevance of uncertainty

The hydrology community produces enormous uncertainty envelopes but rarely asks: "Does this uncertainty matter for the decision at hand?"

**Key literature:**
- **Brown et al. (2012)** — "Decision scaling: Linking bottom-up vulnerability analysis with climate projections." Invert the approach: start from the decision threshold and ask which futures cross it.
- **Wilby & Dessai (2010)** *Weather* — Many adaptation decisions are "no-regret" regardless of precise climate change magnitude.
- **Herman et al. (2015)** — Many-Objective Robust Decision Making (MORDM) for water resources under deep uncertainty.

**Gap:** Almost non-existent for glacierized HMA catchments. Upper Indus water management has specific thresholds (hydropower minimum flows, irrigation allocations, flood warnings). Does the choice of hydrological model actually change whether those thresholds are crossed? Some uncertainty sources may be irrelevant to decisions even if large in absolute terms.

**Feasibility:** Medium — needs identification of stakeholder-relevant thresholds. Could frame existing ensemble output around decision thresholds rather than just reporting uncertainty bands.

### Gap 10: Paraglacial landscape evolution

As glaciers retreat they expose new terrain (moraines, bedrock, proglacial lakes) with fundamentally different hydrological properties. Standard models treat deglaciated area as static.

**Key literature:**
- **Ballantyne (2002)** *Quaternary Science Reviews* — Foundational review on paraglacial geomorphology.
- **Lane et al. (2017)** *Geomorphology* — Newly exposed proglacial areas have fundamentally different infiltration/groundwater behavior.
- **Milner et al. (2017)** *PNAS* — Ecological and hydrological changes in rivers as glacier cover diminishes.

**Gap:** The geomorphology community knows this matters, but it has barely been connected to hydrological modeling. For the upper Indus, essentially unexplored.

**Feasibility:** Medium — could approximate in Raven with time-varying land cover classes (replacing glacier HRUs with bare rock → moraine → sparse vegetation over decades). Even a simple sensitivity analysis would be a first.

### Gap 11: Compound events in glacierized catchments

Simultaneous glacier melt + extreme precipitation → flooding. Standard uncertainty analyses look at average seasonal flows and miss tail risk.

**Key literature:**
- **Zscheischler et al. (2018)** *Nature Climate Change* — Foundational framework for compound event analysis.
- **Veh et al. (2020)** *Nature Climate Change* — Studied compound GLOF triggers in the Himalaya.

**Gap:** The question "what happens when a heat wave drives anomalous melt simultaneously with a monsoon extreme" has barely been quantified in a modeling framework for HMA.

### Gap 12: Permafrost thaw effects on mountain hydrology

**Key literature:**
- **Gruber et al. (2017)** *The Cryosphere* — Mapped permafrost in HKH and expected thaw trajectories.
- **Walvoord & Kurylyk (2016)** *Vadose Zone Journal* — Permafrost thaw increases baseflow and creates new subsurface pathways.
- **Hayashi (2020)** *Groundwater* — Mountain groundwater systematically underrepresented in models.

**Gap:** Permafrost is extensive in the upper Indus (especially Karakoram) but not represented in standard Raven setups.

### Gap 13: Black carbon / aerosol effects on glacier melt

**Key literature:**
- **Yasunari et al. (2010)** *Atmos. Chem. Phys.* — BC deposition can enhance glacier melt by **10–30%** in parts of the Himalaya.
- **Kaspari et al. (2011)** — Documented increasing BC trend from Mt. Everest ice core.

**Gap:** Never propagated through the glacier→hydrology chain. Standard GloGEM uses temperature-index melt without BC-induced albedo effects.

### Feasibility Summary

| Angle | Feasibility | Novelty | Could be a paper? |
|---|---|---|---|
| Parameter non-stationarity | High — re-calibrate on sub-periods | High | Yes, combined with main uncertainty decomposition |
| Signal-to-noise decomposition | High — use existing ensemble | Very high | Yes, standalone or combined |
| Decision-relevance framing | Medium — needs thresholds | Very high | Yes, especially with stakeholder angle |
| Paraglacial landscape change | Medium — modify HRU classes | Very high | Yes, standalone sensitivity study |
| Compound events | Medium — extract from simulations | High | Possibly, depends on event frequency |
| Permafrost thaw | Low — needs new processes | High | Future work recommendation |
| Black carbon | Low — needs GloGEM re-runs | High | Future work recommendation |

---

## ★ HIGHLIGHT: Supply-Demand Threshold Analysis With ISIMIP Data ★

**Core idea:** Combine your Raven-simulated discharge (SUPPLY, with full uncertainty from GCMs × hydro models × SSPs) with ISIMIP3b or Khan et al. (2023) gridded water demand data (DEMAND) to create a decision-relevant threshold analysis for glacierized upper Indus catchments.

**Why this is powerful:** ISIMIP3b uses the **exact same 5 GCMs** as your GloGEM runs, enabling a fully consistent chain:

```
Same GCM → GloGEM (glacier melt) → Raven (streamflow = SUPPLY)
Same GCM → ISIMIP water models (water demand = DEMAND)
→ Water Stress Index = DEMAND / SUPPLY
```

**Standard WSI thresholds** (Falkenmark / Hanasaki et al. 2018):
- WSI < 0.2 → Low stress
- 0.2–0.4 → Medium stress
- WSI > 0.4 → **High stress**

**This would be novel because:**
- A study in *Global Sustainability* (likely Wild/Hejazi et al., PNNL — used OGGM + Xanthos) noted "most previous literature has not explicitly modeled glacier runoff to assess future water scarcity" — global water models have simplified glacier routines
- Your process-based GloGEM + calibrated Raven gives better supply estimates than any ISIMIP GHM
- Nobody has combined catchment-scale glacier-hydrology uncertainty quantification with demand-side scenarios for HMA

### Two Water Demand Data Options

#### Option A: ISIMIP3b Water Sector Output (recommended for consistency)

**Access:** https://data.isimip.org/ — crop to your bounding box, download as NetCDF

**Models:** H08, WaterGAP2, CWatM, and others (0.5° monthly resolution, 1850–2100)

**Key variables:**

| Variable | Description |
|---|---|
| `atotww` | Actual total water withdrawal |
| `airrww` | Actual irrigation water withdrawal |
| `adomww` | Actual domestic water withdrawal |
| `amanww` | Actual manufacturing water withdrawal |
| `atotuse` | Actual total water consumption |

**How demand is generated in ISIMIP:**
- **Irrigation demand:** Computed **internally** by each model — climate-sensitive (driven by soil moisture deficit, ET, crop type). Hotter/drier → more demand.
- **Domestic/Industrial:** Driven primarily by **socioeconomic** variables (population, GDP, SSP). Models with own demand modules (H08, WaterGAP, CWatM) compute these internally. Models without demand modules receive a **prescribed multi-model mean** from WaterGAP, H08, PCR-GLOBWB (ISIMIP2b, SSP2) — these are NOT climate-sensitive.

**Advantage:** Same 5 GCMs as your runs → fully consistent climate forcing chain.
**Limitation:** 0.5° resolution is coarse for your sub-catchments (~500–5000 km²). Need area-weighted extraction.

#### Option B: Khan et al. (2023) — GCAM-Tethys Dataset

**Citation:** Khan, Z., Wild, T.B., Silva Carrazzone, M.E., Giri, S., Yarlagadda, B., Hejazi, M.I., Burger, J., Kim, S., & Vernon, C.R. (2023). "Global monthly sectoral water use for 2010–2100 at 0.5° resolution across alternative futures." *Scientific Data*, 10, 201. DOI: 10.1038/s41597-023-02086-2

**Methodology:** Uses **GCAM** (integrated assessment model, PNNL) for regional-level demand, spatially downscaled to 0.5° using **Tethys** (proxy-based: population maps, irrigated area, livestock density). Monthly resolution via historical seasonality profiles.

**Coverage:** 75 scenarios (5 SSPs × multiple climate/demand variants). 6 sectors: irrigation, domestic, industrial, electricity, livestock, mining. Both withdrawal and consumption.

**Hosted on:** Zenodo (linked from the paper)

**Key differences from ISIMIP:**

| | ISIMIP3b Water Models | Khan et al. (2023) / GCAM-Tethys |
|---|---|---|
| **Approach** | Process-based GHMs (physical hydrology + demand) | Integrated Assessment Model (economic system → demand) |
| **Irrigation demand** | Climate-sensitive (soil moisture, ET) | Climate-sensitive (crop water requirements) |
| **Domestic/industrial** | Socioeconomic + minor climate sensitivity | Purely socioeconomic (GDP, population, technology) |
| **Spatial downscaling** | Native 0.5° model grid | Statistical downscaling from regional GCAM output |
| **# scenarios** | ~5 GCMs × 2-4 SSPs per model | 75 scenarios (more demand-side variants) |
| **Consistency with your GCMs** | Same 5 GCMs ✓ | Different GCMs (GCAM's own climate inputs) |
| **Independence** | Model-based, coupled hydrology+demand | Independent from ISIMIP |
| **Strength** | Physically consistent supply-demand | More demand scenarios, economic feedbacks |

**Recommendation:** Use **ISIMIP3b for primary analysis** (GCM consistency), **Khan et al. as cross-check** (independent methodology, more demand scenarios).

### Practical Implementation

1. Download ISIMIP3b `airrww` + `adomww` + `amanww` for your 5 GCMs × 2 SSPs, cropped to catchment bbox
2. Area-weight demand to your catchment polygons
3. For each of your 40 ensemble members (4 hydro models × 5 GCMs × 2 SSPs):
   - SUPPLY = Raven monthly discharge
   - DEMAND = ISIMIP demand for same GCM × SSP
   - WSI = DEMAND / SUPPLY (monthly)
4. Map response surface: for which ΔT × ΔP combinations does WSI > 0.4?
5. Overlay GCM ensemble on the response surface
6. Decompose: which uncertainty source (GCM? hydro model? SSP?) determines whether the threshold is crossed?

### Additional Thresholds to Test

| Threshold | Value | Source |
|---|---|---|
| WSI > 0.4 (high water stress) | Falkenmark / Hanasaki et al. 2018 |
| E-flow lean season | 20% of avg 4 leanest months | India CWC/EAC regulation |
| E-flow monsoon | 30% of inflows (90% dep. year) | India CWC/EAC |
| Hydropower Q30 | Flow exceeded 30% of time | Frontiers in Water 2023 |
| Hydropower Q80 (small plants) | Flow exceeded 80% of time | Frontiers in Water 2023 |

### Key Supporting Literature

- **Schewe et al. (2014)** *PNAS* — Landmark ISIMIP water scarcity paper. 11 GHMs × 5 GCMs. 2°C warming exposes ~15% more of global population to severe water scarcity.
- **Hanasaki et al. (2018)** *Water Resources Research* — Evaluated WSI thresholds using H08 at 0.5° daily resolution.
- **Calvo-Gallardo et al. (2025)** *Earth's Future* — Combines glacier runoff with agricultural water demand for Indus, Amu Darya, Tarim.
- **Calvo-Gallardo et al. (2026)** *Global Sustainability*, DOI: 10.1017/sus.2025.10046 — "An integrated assessment of water scarcity and glacier runoff changes in Asian and South American glacierized basins." Coupled OGGM + Xanthos (global 0.5° water balance) + GCAM (demand). Studied Indus, Amu Darya, Tarim + South American basins. Found **socioeconomic pathway matters as much or more than glacier/climate change** — SSP3-7.0 is more water-scarce than SSP5-8.5 (high population + low technology outweighs worse climate). **Limitations:** Xanthos is a simple 4-parameter ABCD bucket model at 0.5°, not calibrated against gauge data, single hydrological model, no uncertainty decomposition, macro-basin scale only ("Indus" = entire basin as one unit). Our work addresses these gaps with process-based GloGEM+Raven at sub-catchment scale, multiple model structures, gauge calibration, and formal uncertainty decomposition.
- **Companion paper:** Calvo-Gallardo et al. (2025) *Earth's Future*, DOI: 10.1029/2024EF005064 — "Assessing the Effect of Glacier Runoff Changes on Basin Runoff and Agricultural Production in the Indus, Amu Darya, and Tarim Interior Basins."

**Positioning statement:** Calvo-Gallardo et al. (2026) provided a valuable first-order assessment at the macro-basin scale, but their reliance on a global water balance model without gauge calibration, at 0.5° resolution, with a single model structure and no uncertainty decomposition, cannot capture the process dynamics or quantify the uncertainty cascade that governs water availability in specific glacierized headwater catchments. We address these limitations by coupling GloGEM with multiple calibrated Raven model structures at the sub-catchment scale, with formal uncertainty decomposition, at the spatial scale where water management decisions are actually made.

### Comprehensive Literature: Glacier Runoff × Water Demand/Scarcity

#### Glacier runoff + water supply projections (HMA / Indus)

- **Immerzeel et al. (2010)** *Science* — "Climate Change Will Affect the Asian Water Towers." SPHY model for Indus, Ganges, Brahmaputra, Yangtze, Yellow. Indus and Brahmaputra most dependent on glacier/snow melt. Compared meltwater supply to downstream irrigated food production. Basin scale; process-based glacier modeling; simplified demand assessment.

- **Immerzeel et al. (2013)** *Nature Geoscience* — "Rising river flows throughout the twenty-first century in two Himalayan glacierized watersheds." High-resolution glacio-hydrological modeling of Baltoro and Langtang (~300–600 km²). Runoff increases through mid-century then declines. Process-based glacier dynamics; no demand analysis.

- **Lutz et al. (2014)** *Nature Climate Change* — "Consistent increase in High Asia's runoff due to increasing glacier melt and precipitation." SPHY for upper Indus, Ganges, Brahmaputra, Salween, Mekong. Consistent runoff increases to 2050 across all basins. Basin scale; process-based; no demand analysis.

- **Lutz et al. (2016)** *PLOS ONE* — "Climate Change Impacts on the Upper Indus Hydrology." Decomposed contributions from glacier melt, snowmelt, rainfall under RCP4.5/8.5. Peak water ~2040–2060. Sub-basin scale; process-based; supply-focused.

- **Biemans et al. (2019)** *Nature Sustainability* — "Importance of snow and glacier meltwater for agriculture on the Indo-Gangetic Plain." **Coupled SPHY with LPJmL crop model** — one of the few studies formally linking glacier runoff to agricultural demand. Snow/glacier melt supports ~60% of irrigation in western Indus Plain. Basin-to-regional scale.

- **Pritchard (2019)** *Nature* — "Asia's shrinking glaciers protect large populations from drought stress." Glacier melt provides natural drought reserve compensating for precipitation deficits. Glacier drought buffering protects ~220 million people in the Indus. Observational; no projections or demand modeling.

#### Peak water and water security

- **Huss & Hock (2018)** *Nature Climate Change* — "Global-scale hydrological response to future glacier mass loss." Definitive peak water study using GloGEM for all ~215,000 glaciers. Peak water already passed for 45–60% of basins in some regions; HMA peaks ~2030–2050. Global per-glacier scale; process-based; no demand analysis.

- **Huss & Hock (2015)** *Frontiers in Earth Science* — GloGEM model description paper. Glacier geometry evolution, mass balance, runoff for every RGI glacier. Global; process-based.

- **Kaser et al. (2010)** *PNAS* — "Contribution potential of glaciers to water availability in different climate regimes." Glacier contributions most critical in arid regions where they compensate for low dry-season precipitation. Global; simplified glacier modeling; introduced "contribution potential" concept.

#### Water scarcity / stress projections (Indus)

- **Laghari et al. (2012)** *HESS* — "The Indus basin in the framework of current and future water resources management." Indus faces severe water stress by 2025 even without climate change, driven by population growth and inefficient irrigation. Review/synthesis; both supply and demand; no process-based glacier modeling.

- **Yu et al. (2013)** *World Bank Report* — "The Indus Basin of Pakistan: Impacts of Climate Risks on Water and Agriculture." System dynamics model linking water supply (incl. glacier/snow) to agricultural demand and economics. Climate change + growing demand could reduce agricultural GDP by 2%. Basin scale; simplified glaciers; comprehensive demand including economics.

#### ISIMIP water sector

- **Schewe et al. (2014)** *PNAS* — Landmark ISIMIP paper. 11 GHMs × 5 GCMs. ~40% of global population faces absolute water scarcity by end of century under high emissions. Global 0.5°; no explicit glacier modeling; demand from population projections.

- **Warszawski et al. (2014)** *PNAS* — ISIMIP framework paper. Established the multi-model protocol. Glacier processes poorly represented in participating GHMs — known limitation for mountain basins.

- **Gosling & Arnell (2016)** *Climatic Change* — Global water scarcity assessment. Climate model uncertainty dominates, but hydrological model structural uncertainty is substantial. No glacier processes.

- **Veldkamp et al. (2017)** *Nature Communications* — "Water scarcity hotspots travel downstream due to human interventions." Demand-side uncertainty comparable to or larger than supply-side climate uncertainty. Global; multiple GHMs and demand scenarios; no glacier modeling.

#### Vulnerability and mountain water dependence

- **Viviroli et al. (2020)** *Nature Sustainability* — "Increasing dependence of lowland populations on mountain water resources." ~1.5 billion people depend on mountain water that cannot be substituted. Indus has the highest lowland dependence. Global; statistical; no process-based glacier modeling.

- **Immerzeel et al. (2020)** *Nature* — "Importance of the Asian water towers." Ranked Indus as most important and most vulnerable water tower. Composite index combining supply and demand indicators. Global; not process-based coupled modeling.

- **Farinotti et al. (2019)** *Nature* — "Large hydropower and water-storage potential in future ice-free cirques." ~900 sites globally where new reservoirs could compensate for lost glacier storage. Global; GloGEM-derived projections; infrastructure-focused.

- **Kraaijenbrink et al. (2017)** *Nature* — "Impact of a global temperature rise of 1.5°C on Asia's glaciers." 1.5°C → 36% mass loss; 2°C → ~50%. Process-based; no demand analysis.

#### Supply-demand in other glacierized regions

- **Ragettli et al. (2016)** *PNAS* — Contrasting Himalayan vs Andean catchment responses. Himalaya: runoff increases; Andes: severe decline. Catchment scale (~50–200 km²); process-based; no demand.

- **Brunner et al. (2019)** *HESS* — "Future shifts in extreme flow regimes in Alpine regions." Increasing drought severity in late summer due to reduced glacier melt. Swiss Alps; process-based; no demand.

#### Decision scaling / vulnerability in mountain water

- **Brown et al. (2012)** *Water Resources Research* — Decision scaling methodology. Foundational; not glacier-specific.

- **Ray & Brown (2015)** *World Bank* — Decision Tree Framework for climate uncertainty in water infrastructure. Applicable to mountain systems.

- **Wijngaard et al. (2017)** *PLOS ONE* — "Future changes in hydro-climatic extremes in the Upper Indus, Ganges, and Brahmaputra." SPHY; process-based glacier/snow; focused on extremes for hydropower/irrigation implications.

#### Uncertainty in water scarcity projections

- **Huss et al. (2017)** *Earth's Future* — "Toward mountains without permanent snow and ice." Review of cascading consequences of glacier/snow loss. Discussed uncertainty sources qualitatively.

### The Scale Problem: A "Missing Middle" in the Literature

Most Indus studies work at the full basin or major sub-basin scale, but water management decisions happen at the tributary/headwater scale. The literature has a clear gap:

| Scale | Coverage | Examples |
|---|---|---|
| Pan-HKH / Full basin (~1M km²) | **Very well covered** | Immerzeel 2010, Lutz 2014, Kraaijenbrink 2017, Pritchard 2019 |
| Major sub-basins (50,000–200,000 km²) | **Moderately covered** | Archer & Fowler 2004, Dahri 2016, Mukhopadhyay & Khan 2014 |
| Individual glacier (<10 km²) | **Growing** | Azam et al. (Chhota Shigri), geodetic studies |
| **Headwater catchments (500–5,000 km²)** | **Very sparse** | Immerzeel 2013, Ragettli 2015 (both Nepal/Karakoram) |
| **Headwater catchments in western Himalaya/Chenab** | **Essentially absent** | **← Your work** |

#### Studies at headwater catchment scale in HKH

- **Immerzeel et al. (2013)** *Nature Geoscience* — Baltoro (~1,500 km²) and Langtang (~350 km²). One of the few truly catchment-scale studies. Showed opposite hydrological trajectories under climate change in neighboring catchments — key paper for arguing that basin aggregation masks heterogeneity.

- **Ragettli et al. (2015)** *J. Hydrology* — "Unraveling the hydrology of a Himalayan catchment through integration of high resolution in situ data." Langtang (~350 km²). TOPKAPI-ETH model. Best example of detailed catchment-scale modeling in HKH.

- **Ragettli et al. (2016)** *PNAS* — Contrasting Langtang (Nepal) vs Juncal (Chile). Opposite hydrological responses despite similar glacier retreat.

- **Nepal et al. (2014)** *J. Hydrology* — J2000 model in Dudh Koshi (~3,700 km²), Nepal.

- **Soncini et al. (2015)** — Shigar basin in the Karakoram.

- **Engelhardt et al. (2017)** — Chhota Shigri glacier catchment (~45 km²), very small scale.

**For Chenab/Chandra/Beas specifically:** Almost nothing at headwater catchment scale with explicit glacier dynamics. Some glacier-scale studies exist (Azam et al. on Chhota Shigri) but no systematic catchment-scale hydrological modeling with future projections.

#### Why aggregation masks critical information

- **Dahri et al. (2016)** showed precipitation varies by a **factor of 10+** across upper Indus sub-basins — basin-average precipitation is essentially meaningless.

- **Huss & Hock (2018)** showed peak water timing varies by **decades** depending on glacier size distribution — adjacent tributaries can have very different trajectories.

- Glacier coverage ranges from **<1% to >30%** across neighboring tributaries, meaning glacier melt contribution to flow varies enormously.

- The "Karakoram anomaly" (glacier stability/growth) versus rapid Himalayan retreat means even within the upper Indus, trajectories diverge fundamentally.

- **Viviroli et al. (2011)** *Global and Planetary Change* — Explicitly argued for finer-scale studies for mountain water management.

- **Pellicciotti et al. (2012)** — Discussed the challenges of headwater-scale work in HKH due to data scarcity.

#### Scale framing statement

"While basin-scale studies have established that the upper Indus is highly vulnerable to glacier change (Immerzeel et al. 2010, 2020), water management decisions — hydropower design, irrigation diversions, flood warnings — are made at the tributary scale, where glacier contributions, peak water timing, and hydrological regime shifts can differ fundamentally from basin averages (Immerzeel et al. 2013; Huss & Hock 2018). This mismatch between the scale of scientific assessment and the scale of decision-making represents a critical gap, particularly in the western Himalaya (Chandra/Chenab system) where catchment-scale glacio-hydrological modeling with future projections is essentially absent."

### What No Existing Study Does (= Your Contribution)

| Capability | Closest existing study | What's missing |
|---|---|---|
| Process-based glacier + hydrology coupling | Biemans et al. 2019 (SPHY+LPJmL) | Single model structure, no uncertainty decomposition |
| Formal uncertainty decomposition of supply | Schewe et al. 2014; Gosling & Arnell 2016 | No glacier processes, global scale |
| Supply-demand threshold at sub-catchment scale | Calvo-Gallardo et al. 2026 (OGGM+Xanthos+GCAM) | 0.5° global model, no gauge calibration, macro-basin |
| Multiple hydrological model structures + glaciers | Nobody | **This is your unique contribution** |
| Decision-relevant framing for glacierized HMA | Nobody | Brown et al. (2012) methodology exists but never applied here |
| **Headwater catchment scale in western Himalaya** | Ragettli 2015 (Langtang, Nepal only) | **No equivalent in Chenab/Chandra system** |
| **Raven model in HKH** | Nobody | **First application of Raven in this region** |

---

## Deep Dive: Parameter Non-Stationarity

### The Differential Split-Sample Test (DSST)

**Klemes (1986)** *Hydrological Sciences Journal* introduced the hierarchical testing scheme:
- **Split-sample test (SST):** Calibrate on one period, validate on another
- **Proxy-basin test:** Calibrate on one basin, validate on another
- **Differential split-sample test (DSST):** Split the record into climatologically *different* subperiods (wet vs dry), calibrate on one, validate on the other
- **Proxy-basin DSST:** Transfer across both space and climate simultaneously (most stringent)

The argument: if your model can't reproduce behavior under *observed* climate variability, you have no business using it for climate change projections.

### Key Studies With Quantified Results

- **Coron et al. (2012)** *Water Resources Research* — "Crash testing hydrological models" on 216 Australian catchments. Calibrated on wet/dry 5-year periods and tested on the opposite. **Models calibrated on wet periods performed significantly worse on dry periods, and vice versa.** The degradation was asymmetric — wet-to-dry transfers were worse. Systematic bias in soil moisture accounting store capacity.

- **Vaze et al. (2010)** *J. Hydrology* — Tested 4 models on Australian catchments. Found models calibrated on periods with **mean annual rainfall within ~15% of the projection period** performed acceptably. Beyond that threshold, performance degraded substantially. Practical guideline: if your future climate is >15% wetter/drier than calibration, beware.

- **Merz et al. (2011)** *Water Resources Research* — Austrian catchments, HBV-type model on 5-year moving windows across ~40 years. Found **significant temporal trends in calibrated parameters**, particularly degree-day factor and soil moisture parameters — suggesting parameters compensate for unrepresented processes.

- **Broderick et al. (2016)** *Water Resources Research* — Irish catchments, 4 conceptual models. Equifinal parameter sets diverged by **10–20% of mean flow** in projected runoff under different climate conditions — sometimes comparable to GCM spread.

- **Saft et al. (2015, 2016)** *Water Resources Research* — Australian Millennium Drought. Prolonged drought caused a **shift in the rainfall-runoff relationship** that persisted after drought ended. Models calibrated on pre-drought data **overestimated runoff during drought by 50–100%** in some catchments.

- **Fowler et al. (2016, 2018)** *Water Resources Research* — Challenged the narrative somewhat: showed that some apparent non-stationarity was due to precipitation measurement errors changing over time. Important caveat for HMA where precipitation measurement is notoriously poor.

### In Glacierized Catchments Specifically

Very sparse literature:
- **Konz & Seibert (2010)** *J. Hydrology* — Including glacier mass balance as a calibration constraint reduced equifinality in Alpine catchments. Without glacier-specific constraints, parameters compensate.
- **Gabbi et al. (2014)** — Showed degree-day factors for glacierized catchments are not constant over time.
- **Hock (2003)** — Classic review noting degree-day factors vary seasonally, inter-annually, and with changing glacier surface conditions.

**No one has done a rigorous Klemes-style DSST for glacierized HMA catchments.** This is a genuine gap.

### Mechanisms Causing Non-Stationarity in Glacierized Catchments

1. **Changing glacier fraction** — the melt contribution changes fundamentally as ice area shrinks
2. **Rain/snow ratio shift** — warming raises the rain/snow partition elevation, changing timing and storage
3. **New land surface exposure** — bare rock, moraine, soil have different infiltration/evaporation properties
4. **Vegetation migration upward** — documented in the Himalayas; changes ET, interception, soil development
5. **Permafrost degradation** — changes subsurface storage and flow paths
6. **Debris cover evolution** — debris-covered glaciers melt differently; debris often increases as glaciers thin
7. **Glacier hypsometry changes** — area-elevation distribution changes affect temperature-driven melt

### Practical Application for Your Catchments

1. Take catchment 0101 (most tested)
2. Split calibration period into early/late halves (or high-melt vs low-melt years)
3. Calibrate Raven on each half separately
4. Compare: (a) parameter values, (b) cross-validation performance, (c) future projections from each parameter set
5. If projections diverge substantially → non-stationarity problem worth addressing

**Advantage of your setup:** Raven+GloGEM already handles time-varying glacier geometry explicitly — so one of the biggest non-stationarity sources is already represented. The remaining question is whether the *hydrological* parameters (soil, baseflow, ET) are also non-stationary.

### Approaches to Address Non-Stationarity

- **Multi-objective calibration:** Add glacier mass balance, MODIS snow cover, ET as calibration targets (Konz & Seibert 2010; Parajka & Blöschl 2008) — constrains equifinality
- **Trading space for time:** Your multiple catchments (0101, 0102, 0118, 0130) with different glacier fractions can serve as proxies for temporal change
- **Behavioral ensemble:** Use all acceptable parameter sets rather than one "best" set; weight by cross-period performance

---

## Deep Dive: Signal-to-Noise / Variance Decomposition

### The Hawkins & Sutton Framework — Mathematical Details

For climate variable Y at time t, model m, scenario s, realization r:

**Step 1: Fit smooth signal.** For each model-scenario combination, fit a 4th-order polynomial:
Y(t,m,s,r) = f(t,m,s) + ε(t,m,s,r)

**Step 2: Internal variability.**
σ²_I = Var(ε), estimated from residuals pooled across models/scenarios

**Step 3: Model uncertainty.** At each time t:
σ²_M(t) = Var_m[Y_m(t) - Y̅(t)] (spread of model means around grand mean)

**Step 4: Scenario uncertainty.**
σ²_S(t) = Var_s[Y_s(t) - Y̅(t)] (spread of scenario means around grand mean)

**Step 5: Total = σ²_M(t) + σ²_S(t) + σ²_I(t)**

Express as fractions → the classic "wedge" plots showing how dominance shifts over time.

**Time of emergence** = when signal-to-noise ratio > 1 (multi-model mean change exceeds the combined noise).

### Signal Emergence Timelines (from literature)

| Variable | Signal emerges (~S/N>1) | Notes |
|---|---|---|
| Temperature | 2020–2040 (already emerged) | Scenario dominates by end of century |
| Mean annual precipitation | 2050–2100+ (many regions never) | Model uncertainty dominates |
| Mean annual discharge | 2040–2070 | GCM dominates |
| Snow fraction / melt timing | 2030–2050 | Strong temperature signal |
| Glacier runoff | Peak water is relatively robust | GCM for timing, SSP for magnitude |
| Flood magnitude | Often never before 2100 | Internal variability + GCM |
| Low flows / drought | 2050–2080 in some regions | GCM + hydro model |

### ANOVA for Your 4×5×2 Design

Your design: 4 hydro models × 5 GCMs × 2 SSPs = 40 simulations.

**Model:** Y_ijk = μ + α_i + β_j + γ_k + (αβ)_ij + (αγ)_ik + (βγ)_jk + ε_ijk

where: α = hydro model effect, β = GCM effect, γ = SSP effect

**Variance partition:**
```python
SS_hydro = n_gcm * n_ssp * Σ(α_i²)
SS_gcm   = n_hydro * n_ssp * Σ(β_j²)
SS_ssp   = n_hydro * n_gcm * Σ(γ_k²)
SS_total  = Σ(Y_ijk - μ)²
frac_X    = SS_X / SS_total
```

**Practical note:** With 40 cells and no replication, pool the 3-way interaction into the error term (12 df). This is standard practice (Bosshard et al. did similar).

**Add internal variability** via quasi-ergodic approach: fit smooth curve to each of the 40 annual time series, compute variance of residuals, pool across runs.

**Time-evolving analysis:** Apply ANOVA to rolling 30-year windows centered on each decade → Hawkins-Sutton-style wedge plots.

### Alternative: QUALYPSO

**Evin et al. (2019)** — Bayesian framework specifically designed for climate impact studies. Handles unbalanced designs, trends, and internal variability simultaneously. Available as an R package. Worth considering as a more sophisticated alternative to classical ANOVA.

### Alternative: Random Forest (Aguayo approach)

Train RF where target = hydrological metric, features = categorical factors (model, GCM, SSP). Permutation importance gives non-parametric variance decomposition. Easy to implement with scikit-learn. Good as a robustness check alongside ANOVA.

---

## Deep Dive: Decision-Relevance of Uncertainty

### Decision Scaling Methodology (Brown et al. 2012)

**The inversion:** Instead of GCM → downscaling → hydrology → impact → decision, it goes:

1. **Define the system + performance thresholds** (e.g., "irrigation reliability must exceed 90%")
2. **Stress-test across climate state space** — systematically vary ΔT, ΔP, Δseasonality across a regular grid (not just where GCMs land)
3. **Map a response surface** — run system model for each climate perturbation; identify where the system fails
4. **Overlay GCM projections** as probability markers on top of the response surface

| Traditional (top-down) | Decision scaling (bottom-up) |
|---|---|
| Start from GCM scenarios | Start from decision thresholds |
| Uncertainty cascades and expands | Uncertainty characterized relative to thresholds |
| Result: wide range of outcomes | Result: "here is where your system fails" |

### Key Additional References

- **Lempert et al. (2003)** — Robust Decision Making (RDM) foundational text. Embraces deep uncertainty where probability distributions are unknown.
- **Herman et al. (2015)** *J. Water Resources Planning & Management* — Tested different robustness definitions (satisficing, regret-based); the definition itself changes which strategy looks best.
- **Poff et al. (2016)** *Nature Climate Change* — Eco-Engineering Decision Scaling (EEDS): extends to ecological thresholds alongside engineering ones.
- **Ray & Brown (2015)** — World Bank Decision Tree Framework, operationalized for infrastructure project appraisal.
- **Shepherd et al. (2018)** *Climatic Change* — "Storyline" approach: construct physically coherent causal chains rather than relying on ensemble statistics.

### Studies Where Uncertainty Was Irrelevant for the Decision

- **Dessai & Hulme (2007)** *Global Environmental Change* — Water resources in East of England: the decision to invest in demand management was robust across all climate scenarios.
- **Brown et al. (2012)** — Some water systems have the entire GCM ensemble in the "safe" zone of the response surface, meaning large uncertainty is decision-irrelevant.
- **Culley et al. (2016)** *Water Resources Research* — Some systems have adaptive capacity exceeding the range of projected changes.

### Decision-Relevant Thresholds for Upper Indus

| Decision | Key variable | Threshold type |
|---|---|---|
| Run-of-river hydropower | Minimum monthly flow | Design flow for turbines |
| Irrigation season supply | Apr–Sep total volume | Canal system design capacity |
| Flood warning | Peak daily flow | Dam spillway / warning levels |
| Environmental flows | Dry-season minimum | Ecological flow requirements |
| Reservoir operations | Timing of peak inflow | Operating rule curves |

### What This Would Look Like for Your Catchments

1. **Define 2–3 decisions** for your sub-catchments (e.g., "Will minimum dry-season flow remain above X m³/s?")
2. **Build a 3D response surface:** ΔT (0 to +6°C) × ΔP (−30% to +30%) × glacier area reduction (0–100%)
3. **Map decision thresholds** on this surface
4. **Overlay your GCM × hydro model ensemble** — do they cluster in safe zone, failure zone, or straddle the boundary?
5. **Key finding:** If all ensemble members fall on one side → uncertainty is decision-irrelevant. If they straddle → identify which axis of uncertainty matters for the decision.

**Nobody has applied decision scaling to glacierized HMA catchments.** This would be genuinely novel — combining peak water dynamics with bottom-up vulnerability analysis.
