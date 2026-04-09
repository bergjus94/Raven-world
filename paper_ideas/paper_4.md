# Paper 4: Decision-Relevant Uncertainty in Glacierized Catchment Projections

## Research Questions

### Overarching question

**"For which water management decisions in the Upper Indus Basin does the choice of glacio-hydrological modeling approach actually matter, and at which planning horizons?"**

### Specific research questions

**RQ1 — Time-evolving uncertainty decomposition:**
How does the relative importance of different uncertainty sources (GCM choice, glacier coupling approach, meteorological forcing, internal variability) evolve over time for key streamflow metrics in glacierized UIB catchments? At which planning horizons does each source dominate?

**RQ2 — Decision thresholds:**
Do the different modeling chains agree on whether critical water management thresholds (hydropower minimum flows, irrigation allocations, flood levels) are crossed under future climate? For which thresholds and time horizons does modeling uncertainty flip the decision?

**RQ3 — Actionable uncertainty reduction:**
Given finite resources, where should the modeling community invest effort — better glacier models, more GCMs, improved precipitation data, or refined hydrological parameters — to most effectively reduce decision-relevant uncertainty at different planning horizons?

---

## Research Gap

### What exists

**Time-evolving uncertainty decomposition:**
- **Hawkins & Sutton (2009, 2011)** — Classic framework partitioning projection uncertainty into model, scenario, and internal variability components that evolve over time. Applied to global temperature and precipitation — never to glacierized catchment streamflow.
- **Bosshard et al. (2013)**, **Addor et al. (2014)** — ANOVA decomposition for streamflow in Swiss catchments, but as static time slices, not the time-evolving Hawkins-Sutton visualization.
- **Clark et al. (2016)** — Extended the framework conceptually to hydrology; found hydrological model uncertainty can dominate GCM uncertainty for some variables.

**Decision scaling and robust decision making:**
- **Brown et al. (2012)** *WRR* — https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2011WR011212
  *"Decision scaling: Linking bottom-up vulnerability analysis with climate projections in the water sector"* — Three-step approach: model the decision, identify thresholds, assess which climate states cross them. Inverts the traditional top-down approach.
- **Herman et al. (2020)** *WRR* — https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019WR025502
  *"Climate Adaptation as a Control Problem"* — Review of dynamic water resources planning under uncertainty. MORDM framework.
- **Wilby & Dessai (2010)** *Weather* — Many adaptation decisions are "no-regret" regardless of precise climate change magnitude.
- **Gil-Garcia et al. (2024)** *HESS* — https://hess.copernicus.org/articles/28/4501/2024/
  *"Actionable human-water system modelling under uncertainty"* — Combined climate, hydrological, and socio-economic ensembles with decision support. Found marginal climate changes can trigger non-linear allocation responses. Applied to Douro basin, Spain.

**UIB water management context:**
- **Frontiers in Water (2023)** — https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2023.1256249/full
  *"Quantification of run-of-river hydropower potential in the Upper Indus basin under climate change"* — Directly relevant: quantifies how hydropower potential changes, but doesn't frame it as a decision-relevant uncertainty problem.
- **Indus Waters Treaty** — Run-of-river hydropower constraints, minimum flow requirements (e.g., 9 cumecs at Kishanganga). Treaty under pressure from climate change and geopolitical tensions (suspended April 2025). ~80% of Pakistan's agriculture and ~28% of electricity depends on Indus water.

### What doesn't exist

1. **Hawkins-Sutton style time-evolving decomposition for glacierized catchment streamflow** — nobody has shown how the relative importance of uncertainty sources shifts over time for runoff in catchments where the glacier signal itself is non-stationary (peak water reversal)
2. **Decision-relevant uncertainty analysis for glacierized HMA catchments** — the decision scaling / MORDM frameworks have never been applied to glacier-fed water systems in HMA
3. **Connection between modeling choices and actual decision outcomes** — existing studies report uncertainty envelopes but never ask "does this uncertainty change the answer to a management question?"
4. **Guidance on where to invest modeling effort** for UIB water planning — nobody has shown whether better glacier models, more GCMs, or improved precipitation data would most effectively reduce decision-relevant uncertainty

---

## Study Design

### Approach: combine Hawkins-Sutton decomposition with decision scaling

**Step 1 — Time-evolving uncertainty decomposition (RQ1):**

For each streamflow metric and each catchment, compute the Hawkins-Sutton decomposition across time slices (2020s, 2030s, ..., 2090s):
- Fractional variance from: GCM choice / glacier coupling approach / meteorological forcing / internal variability
- Visualize as stacked area plots (time on x-axis, fraction of variance on y-axis)
- Key streamflow metrics: mean annual Q, summer Q (JJA), winter Q (DJF), peak flow timing, low-flow quantile (Q95), high-flow quantile (Q5)
- Compare across catchments ordered by glacierization fraction

**Step 2 — Identify decision thresholds (RQ2):**

Define real-world thresholds relevant to UIB water management:
- **Hydropower:** minimum flow for run-of-river operation (from project design documents)
- **Irrigation:** seasonal water allocation triggers (Indus Water Treaty, canal command areas)
- **Flood risk:** high-flow return period thresholds
- **Ecological:** minimum environmental flows

For each threshold, overlay the multi-model projection ensemble and count:
- What fraction of modeling chains project threshold exceedance?
- Does the answer change depending on which glacier coupling / forcing / GCM is used?
- At which time horizon does the modeling choice start to matter (or stop mattering)?

**Step 3 — Uncertainty reduction priorities (RQ3):**

For each decision threshold, compute which uncertainty source most frequently flips the answer:
- If 60% of GCMs say the threshold is crossed but 40% don't → GCM uncertainty is decision-relevant
- If all GCMs agree but coupled vs uncoupled disagree → coupling approach is decision-relevant
- If all modeling chains agree → uncertainty is large but decision-irrelevant (no-regret situation)

Produce a "decision-relevance matrix": decision thresholds × uncertainty sources × time horizons, showing where investment in uncertainty reduction would change management decisions.

### Water demand estimation: methodology

**Approach:** Combine FAO crop water requirement (Tier 2) with Zha et al. (2025) IDC indicator (Tier 3).

**Tier 2 — FAO crop water requirement:**
Net irrigation requirement = (Kc × ETo) − effective rainfall

| Data | What exactly | Source | Status |
|---|---|---|---|
| **Temperature** (min, max, mean) | Daily/monthly per catchment | ERA5-Land / CMIP6 projections | Have |
| **ETo** | Reference evapotranspiration | Penman-Monteith or Hargreaves from temperature; Raven computes PET | Have / computable |
| **Precipitation** | Daily/monthly | ERA5-Land / CMIP6 projections | Have |
| **Effective rainfall** | Fraction of precip usable by crops | FAO empirical formula (function of total precip and ETo) | Computable |
| **Crop coefficients (Kc)** | Per crop, per growth stage | FAO 56 tables (standard values for wheat, rice, maize, etc.) | Publicly available |
| **Crop types & areas** | What's grown where, how much area | Pakistan Bureau of Statistics, FAO GAEZ, MIRCA2000, or MapSPAM | **Need to find** |
| **Cropping calendar** | Planting/harvest dates per crop | FAO or local agricultural data | **Need to find** |
| **Irrigation efficiency** | Surface/sprinkler/drip fraction | FAO estimates: ~60% surface (dominant in Pakistan), 75% sprinkler, 90% drip | Published estimates |

**Tier 3 — Zha IDC (Irrigation Dependence on runoff Component) indicator:**
Adds supply-demand timing mismatch analysis on top of Tier 2.

| Data | What exactly | Source | Status |
|---|---|---|---|
| **Separated runoff components** | Monthly snowmelt, glacier melt, rain runoff | Raven-GloGEM output | Have |
| **Monthly irrigation demand** | From Tier 2 (Kc × ETo − effective rainfall) | Computed | Computable |

The IDC indicator asks: for each runoff component, how much does it contribute to meeting irrigation demand **in the months when demand exists**? This reveals whether glacier melt or snowmelt is more critical for agriculture — and how this changes under future climate.

**Practical shortcut:** Use a single representative crop (wheat — dominant irrigated crop in UIB) with FAO standard Kc values and a generic regional cropping calendar, rather than mapping every crop per catchment. The point is whether modeling uncertainty changes the supply-demand answer, not precise demand estimation.

**Key reference:** Zha et al. (2025) *WRR* — https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023WR036898

### Data requirements (supply side)

- Multi-model projection ensembles from Papers 2 and 3 (multiple GCMs × coupling approaches × forcings × catchments)
- Separated runoff components (snowmelt, glacier melt, rain) from Raven-GloGEM
- Glacier area / volume trajectories for each projection (from GloGEM)

### Data requirements (demand side)

#### Available datasets

**FAO GAEZ v4** (recommended — primary dataset)
- **Resolution:** 5 arcmin (~9 km), global
- **Content:** 51 crops, suitability indices, attainable yields, **crop water indicators** under current AND future RCP climates
- **Key advantage:** Has future crop suitability and water demand projections under climate change — directly gives climate-sensitive demand that shifts with warming. This is essential since warmer future = higher ETo = more irrigation needed, creating a supply-demand double squeeze.
- **Download:** https://gaez.fao.org/
- **For this study:** Use crop water indicators for dominant UIB crops under current and future climates to derive irrigation demand per catchment downstream area.

**MIRCA-OS (2024 update of MIRCA2000)** (supplement — monthly cropping calendars)
- **Resolution:** 5 arcmin (~9 km), global
- **Content:** 23 crop classes, **monthly** irrigated and rainfed areas — provides the cropping calendar built in (which months each crop is growing)
- **Covers:** 2000–2015
- **Download:** https://www.hydroshare.org/resource/60a890eb841c460192c03bb590687145/
- **Reference:** Kebede et al. (2024) *Scientific Data* — https://www.nature.com/articles/s41597-024-04313-w
- **For this study:** Extract monthly growing calendars to compute seasonal timing mismatch with runoff components (Tier 3 / IDC indicator).

**MapSPAM 2020** (supplement — most current crop distribution)
- **Resolution:** 10 × 10 km, global
- **Content:** 46 crops, split by irrigated/rainfed, with harvested area, production, yield
- **Download:** https://www.mapspam.info/data/ (country-level download for Pakistan/India)
- **For this study:** Cross-check crop areas and validate against GAEZ/MIRCA-OS.

#### Recommended approach

1. **FAO GAEZ v4** as the primary demand dataset — provides both current and future crop water requirements under different climate scenarios, matching the future projection framework of Papers 2–3
2. **MIRCA-OS** for monthly cropping calendars — needed for the Tier 3 supply-demand timing analysis (IDC indicator)
3. For each catchment, define the downstream irrigated area and extract dominant crops, their water requirements, and seasonal calendars
4. Demand becomes climate-sensitive: warmer future → higher ETo → more irrigation needed, even if crop areas stay constant

---

## Key Literature

### Time-evolving uncertainty decomposition

- **Hawkins, E. & Sutton, R. (2009)** *Bull. Amer. Meteorol. Soc.* — The Potential to Narrow Uncertainty in Regional Climate Predictions. Classic framework: internal variability dominates near-term, model uncertainty mid-century, scenario uncertainty late-century.

- **Hawkins, E. & Sutton, R. (2011)** *Climate Dynamics* — Extended to precipitation projections.

- **Clark et al. (2016)** *Current Climate Change Reports* — https://link.springer.com/article/10.1007/s40641-016-0034-x
  Conceptual extension to the hydrology chain. Hydrological model uncertainty can dominate GCM uncertainty for some variables.

- **Uncertainty in high-resolution hydrological projections (Hydrological Processes)** — https://onlinelibrary.wiley.com/doi/full/10.1002/hyp.14695
  Partitioning climate model and natural variability influence on streamflow projections.

### Decision scaling and robust decision making

- **Brown, C., Ghile, Y., Laverty, M. & Li, K. (2012)** *WRR* — https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2011WR011212
  Decision scaling: bottom-up vulnerability analysis linked to climate projections. Identify thresholds, then ask which climate states cross them.

- **Herman, J.D. et al. (2020)** *WRR* — https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019WR025502
  Climate adaptation as a control problem. Review of dynamic water resources planning under uncertainty. MORDM framework.

- **Wilby, R.L. & Dessai, S. (2010)** *Weather* — Many adaptation decisions are "no-regret" regardless of climate projection precision.

- **Gil-Garcia et al. (2024)** *HESS*, 28, 4501 — https://hess.copernicus.org/articles/28/4501/2024/
  *"Actionable human-water system modelling under uncertainty"* — Combined ensembles with decision support. Marginal climate changes trigger non-linear allocation responses. Douro basin, Spain.

- **Paton et al. (2015)** *Nature Climate Change* — https://www.nature.com/articles/nclimate2765
  *"Sustainable water management under future uncertainty with eco-engineering decision scaling"*

### UIB water resources and hydropower

- **Frontiers in Water (2023)** — https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2023.1256249/full
  *"Quantification of run-of-river hydropower potential in the Upper Indus basin under climate change"* — Directly quantifies hydropower changes but without decision-framing.

- **Indus Waters Treaty context** — Run-of-river constraints, minimum flows. ~80% Pakistan agriculture, ~28% electricity from Indus. Treaty suspended April 2025 — heightened geopolitical relevance.

### Uncertainty in glacierized catchment projections

- **Hester et al. (2025)** *JAWRA* — https://onlinelibrary.wiley.com/doi/10.1111/1752-1688.70020
  Intersection of hydrologic change and hydropower. Identifies need for decision-relevant uncertainty quantification.

- **High uncertainty in 21st century runoff projections from glacierized basins (JoH)** — https://www.sciencedirect.com/science/article/abs/pii/S0022169413009141
  Demonstrated high projection uncertainty but didn't connect it to decisions.

---

## Contribution

This would be the **first study to**:
1. Apply the Hawkins-Sutton time-evolving uncertainty decomposition to glacierized catchment streamflow — where the glacier signal itself is non-stationary (peak water)
2. Apply decision scaling / threshold analysis to glacier-fed water systems in HMA
3. Show for which real-world water management decisions in the UIB the choice of modeling approach actually matters — and for which it doesn't
4. Provide actionable guidance on where to invest modeling effort to reduce decision-relevant (not just total) uncertainty for UIB water planning

## Connection to Papers 2 and 3

This paper builds directly on the ensemble outputs from:
- **Paper 2** — provides the coupling approach × forcing × GCM ensemble needed for the decomposition
- **Paper 3** — provides the drought buffering analysis that feeds into drought/low-flow threshold assessment

Paper 4 is the **synthesis paper** that translates the uncertainty quantification from Papers 2–3 into actionable guidance for water managers.
