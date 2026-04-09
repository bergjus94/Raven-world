# Paper 3: Glacier Drought Buffering Across the Upper Indus Basin

## Research Questions

### Overarching question

**"How do snow droughts and glacier drought buffering interact across catchments with different glacier and climate characteristics in the Upper Indus Basin, and how will buffering capacity decline under climate change?"**

### Specific research questions

**RQ1 — Snow drought characterization:**
What types of snow droughts (warm, dry, compound) occur across the Upper Indus Basin, how do they differ between westerly-dominated and monsoon-influenced catchments, and how has their frequency changed over the observational period?

**RQ2 — Glacier drought buffering:**
To what extent do glaciers compensate for reduced snowmelt during snow drought years, and how does this buffering capacity vary with glacier fraction, glacier size distribution, and climatic regime across 12 UIB catchments?

**RQ3 — Future loss of buffering:**
Under future climate scenarios, when does each catchment lose its effective glacier drought buffering capacity, and are small-glacier-dominated catchments more vulnerable to earlier loss of drought resilience?

**RQ4 — Snow-ice feedback:**
Does snow drought accelerate glacier melt through reduced snow insulation and albedo feedbacks, and does this feedback — and its potential to bring forward peak water timing — vary across catchments with different glacier and climate characteristics?

---

## Research Gap

### What exists

- **Van Tiel et al. (2021, HESS)** — *"Do glaciers compensate?"* — 50 catchments in Norway, Canada, Alps. Compensation >100% at >5–15% glacier cover. Highly variable; antecedent winter snowfall explains much of the spread. **But not HMA.**
- **Van Tiel, Huss et al. (2026, HESS)** — Swiss 2022 drought case study. 60–80% of glacier melt from net mass loss. Full compensation at ~15% glacierization. But total melt volumes declining. **Single event, Swiss Alps only.**
- **Pritchard (2019, Nature)** — Asia's glaciers protect 221M people from drought stress. Meltwater dominates upper Indus during droughts. **But coarse basin-scale analysis, no catchment-level variation, no future projections.**
- **Nepal et al. (2025, JoH)** — *"Quantifying the impact of snow drought on glacier melting at a Himalayan mountain basin"* — Karnali River. Snow drought → snowmelt decreases, glacier melt doubles, melt timing shifts 1–2 months earlier. **But single basin, no multi-catchment comparison.**
- **Ougahi et al. (2025)** — Water resource vulnerabilities from climate-induced tipping point in the Karakoram Anomaly region. **Focused on the Karakoram specifically.**

### What doesn't exist

1. **Snow drought typology for the UIB** — the dry/warm/compound classification (Han et al. 2025, Dierauer et al. 2019) has not been applied across UIB sub-catchments with their distinct westerly vs monsoon regimes
2. **Multi-catchment analysis of drought buffering across a gradient of glacier characteristics in HMA** — Van Tiel (2021) did this for Europe/Canada but nobody has done it for HMA
3. **Future projections of when drought buffering is lost** per catchment — Pritchard (2019) showed current buffering matters but didn't project its decline
4. **Link between glacier size distribution and buffering capacity/decline** — small glaciers disappear first, so buffering loss should depend on the glacier population structure, but this hasn't been tested
5. **The snow drought → glacier melt feedback across multiple catchments** — Nepal et al. (2025) showed the mechanism in one basin; does it operate differently depending on glacier fraction, size, and elevation?

---

## Study Design

### Data and setup

- **Catchments:** 12 UIB catchments with varying glacier fraction and glacier size distributions
- **Historical simulations:** GloGEM-coupled Raven runs with separated runoff components (snowmelt, glacier melt, rain runoff)
- **Future projections:** Multiple GCM-forced GloGEM projections coupled with Raven
- **Glacier inventory data:** RGI for glacier count, size distribution, median area, elevation range per catchment

### Catchment characterization

For each catchment, derive:
- Total glacier fraction (% of catchment area)
- Number of glaciers
- Median glacier area
- Fraction of glacier area from small glaciers (< 0.5 km²)
- Dominant precipitation regime (westerly vs monsoon)
- Elevation distribution of glaciers

### Part 1: Historical drought buffering (RQ1)

1. **Identify snow drought years** per catchment — years with winter (DJF or ONDJFM depending on regime) precipitation significantly below the long-term mean (e.g., below 1 standard deviation, or lowest quartile)
2. **Quantify glacier compensation** — compare glacier melt contribution in drought years vs normal years. Calculate a compensation ratio following Van Tiel (2021): ratio of actual summer streamflow to expected streamflow without glacier melt surplus
3. **Relate compensation to catchment characteristics** — scatter plots / regression of compensation ratio vs glacier fraction, glacier size distribution, climatic regime
4. **Seasonal analysis** — does compensation peak in early summer (snow-dominated) or late summer (glacier-dominated)? How does this vary across catchments?

### Part 2: Future decline of buffering (RQ2)

1. **Repeat the compensation analysis for future periods** (e.g., 2030–2060, 2060–2090) under multiple GCM projections
2. **Track when each catchment crosses critical thresholds** — e.g., when compensation ratio drops below 100% (glaciers can no longer fully offset drought), or below 50%
3. **Relate timing of buffering loss to glacier characteristics** — do small-glacier-dominated catchments lose buffering earlier? Plot "year of buffering loss" vs median glacier size / small glacier fraction
4. **Quantify changes in interannual variability** — as buffering declines, does year-to-year streamflow variability increase? This is the water security metric.

### Part 3: Snow drought feedback (RQ3)

1. **Quantify the snow cover → glacier melt feedback** — in snow drought years, does reduced snow cover lead to earlier/enhanced glacier melt? Compare snow-covered period and glacier melt onset in drought vs normal years
2. **Test whether the feedback varies with catchment characteristics** — small glaciers at lower elevations may be more exposed to this effect
3. **Assess whether the feedback accelerates peak water** — does the additional melt during snow droughts draw down glacier storage faster, bringing forward the transition to declining runoff?

### Expected outputs

1. **Compensation ratio map** — showing buffering capacity across the 12 UIB catchments for historical period
2. **Compensation vs glacier characteristics scatter plots** — relating buffering to glacier fraction, size distribution, and climate regime
3. **Future buffering decline timeseries** — per catchment, showing when buffering is lost under different GCM projections
4. **"Year of buffering loss" vs glacier size distribution** — the headline figure linking small glacier vulnerability to water security
5. **Snow drought feedback analysis** — showing enhanced glacier melt and timing shift during drought years
6. **Interannual variability change** — how streamflow variability increases as buffering declines

---

## Key Literature

### Snow drought typology and trends

- **Dierauer, J.R. et al. (2019)** *Eos* — https://eos.org/opinions/defining-snow-drought-and-why-it-matters
  Established the dry vs warm snow drought classification framework. Dry = below-normal precipitation; warm = above-normal temperature causing rain instead of snow or early melt.

- **Han et al. (2025)** *Water Resources Research* — https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024WR037492
  *"Changes in Snow Drought and the Impacts on Streamflow Across Northern Catchments"* — 57% of snow droughts are compound (warm + dry). Dry snow droughts cause immediate persistent water deficits; warm snow droughts temporarily alleviate shortages but intensify spring-summer scarcity.

- **GRL (2025)** — https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2025GL114641
  *"Patterns of Snow Drought Under Climate Change: From Dry to Warm Dominance"* — Climate change shifts snow droughts from precipitation-driven to temperature-driven.

- **Schmitt et al. (2024)** *JGR Atmospheres* — https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JD039754
  *"Illuminating Snow Droughts: The Future of Western United States Snowpack in the SPEAR Large Ensemble"* — Snow drought frequency could triple (SSP2-4.5) to quadruple (SSP5-8.5) by 2100.

- **Livneh, B. & Badger, A.M. (2020)** *Nature Climate Change* — https://www.nature.com/articles/s41558-020-0754-8
  *"Drought less predictable under declining future snowpack"* — By late century, 83% of snowmelt-dominated areas lose the ability of snow to predict seasonal drought.

- **Snow droughts 1951–2021 (2025)** — https://www.sciencedirect.com/science/article/abs/pii/S0169809525003291
  *"Snow droughts over 1951–2021 show a decreasing and then increasing trend"* — Long-term global analysis showing reversal in snow drought trends.

### Glacier drought buffering (general)

- **Van Tiel, M., Van Loon, A.F., Seibert, J. & Stahl, K. (2021)** *HESS*, 25, 3245–3265 — https://hess.copernicus.org/articles/25/3245/2021/
  50 catchments, Norway/Canada/Alps. Compensation >100% at >5–15% glacier cover. Variable; depends on antecedent winter snowfall. Framework for compensation ratio calculation.

- **Van Tiel, M., Huss, M., Zappa, M., Jonas, T. & Farinotti, D. (2026)** *HESS*, 30, 23–43 — https://hess.copernicus.org/articles/30/23/2026/
  Swiss 2022 drought. 60–80% of melt from net mass loss. Full compensation at ~15% glacierization. But melt volumes declining — buffering capacity is being spent.

- **Pritchard, H.D. (2019)** *Nature*, 569, 649–654 — https://www.nature.com/articles/s41586-019-1240-1
  Asia's glaciers protect 221M people from drought stress. 36 km³/yr meltwater. Meltwater dominates upper Indus, Aral, Chu basins during droughts. Uniquely drought-resilient source.

- **Han, J., Liu, Z., Woods, R.A., McVicar, T., Yang, D. et al. (2024)** *Nature*, 629, 8014 — https://www.nature.com/articles/s41586-024-07299-y
  *"Streamflow seasonality in a snow-dwindling world"* — Global analysis of declining snow storage reshaping streamflow timing.

### HMA-specific: snow drought occurrence

- **Singh, H., Varade, D. et al. (2025)** *Scientific Reports*, 15, 36101 — https://www.nature.com/articles/s41598-025-21257-2
  *"Intensified occurrences of snow droughts are related to the snow cover dynamics in the Hindu Kush Himalayas"* — Snow drought intensification across 11 HKH basins (SWEI, 1999–2016). Moderate to severe snow droughts in 2008, 2011, 2015, 2016 in the Indus, Amu-Darya, Salween, Mekong basins. Links snow drought occurrence to snow cover dynamics.

- **ICIMOD Snow Update (2025)** — https://www.icimod.org/press-release/risk-of-water-shortages-builds-up-as-hindu-kush-himalaya-faces-23-year-record-low-snow-persistence-in-the-third-consecutive-year-of-below-normal-seasonal-snow/
  HKH in 3rd consecutive below-normal snow year. Snow persistence at 23-year lowest (-23.6%). Indus Basin hit 20-year lowest (-24.5%) in 2024. Decline most pronounced at 3,000–6,000 m — the elevation band where most glaciers sit. Snow cover days dropping ~5 days per decade.

- **Kuttippurath, J., Patel, V.K. & Sharma, B.R. (2024)** *npj Climate and Atmospheric Science*, 7(1), 162 — https://www.nature.com/articles/s41612-024-00710-5
  *"Observed changes in the climate and snow dynamics of the Third Pole"* — Documents declining snowfall, earlier melt, shorter snow cover periods across HMA (1980–2020).

### HMA-specific: snow drought → glacier melt / hydrology

- **Nepal, J., Bhlon, R., Wang, L. & Shrestha, M. (2025)** *Journal of Hydrology* — https://www.sciencedirect.com/science/article/abs/pii/S0022169425010741
  *"Quantifying the impact of snow drought on glacier melting at a Himalayan mountain basin"* — Karnali River, 2003–2019. Snowmelt decreased, glacier melt doubled during winter droughts. Timing shifted 1–2 months earlier. Snow acts as insulating buffer for ice. **Only study connecting snow drought to glacier melt compensation in HMA — single basin.**

- **Ougahi, J.H. et al. (2025)** — https://www.sciencedirect.com/science/article/pii/S2214581825002113
  *"Water resource vulnerabilities from climate-induced tipping point behaviour in runoff volumes and seasonality in the region of the 'Karakoram Anomaly'"* — Hunza River Basin. Glacier area projected to decline from 4,270 km² (2010) to 2,730–3,540 km² by 2100. SWE projected to decline across all seasons. Examines what happens if winter westerly precipitation patterns shift.

- **Pritchard (2019)** — Upper Indus specifically highlighted as most glacier-dependent during droughts.

- **Immerzeel et al. (2020)** *Science* — https://www.science.org/doi/10.1126/science.abf3668
  Glaciohydrology of the Himalaya-Karakoram. Data gaps, process understanding challenges.

---

## Contribution

This would be the **first study to**:
1. Quantify glacier drought buffering across multiple catchments in HMA with varying glacier characteristics
2. Project when each catchment loses effective drought buffering under climate change
3. Link buffering capacity and its future decline to glacier size distribution — showing that small-glacier-dominated catchments are most vulnerable
4. Test the snow drought → glacier melt acceleration feedback across a gradient of catchment types in HMA
5. Provide catchment-specific timelines for loss of drought resilience — directly relevant for water resource planning in the UIB
