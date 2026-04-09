# Paper 5: Baseflow Representation in Glacierized Mountain Catchments

## Research Questions

*To be refined after diagnostic analysis of current model outputs*

### Preliminary questions

**RQ1 — Diagnostic:**
How well do current conceptual model configurations (HBV in Raven) represent winter baseflow across the 12 UIB catchments? Is the systematic underestimation of winter low flows related to catchment characteristics (glacier fraction, elevation, geology)?

**RQ2 — Structural improvement:**
Does adding a deeper groundwater storage component (third reservoir with multi-year residence time) improve baseflow simulation without degrading summer peak performance? How sensitive are the results to the structure of the deep reservoir (linear vs non-linear, single vs dual)?

**RQ3 — Implications for projections:**
Does correcting the baseflow representation change future streamflow projections — particularly winter low flows and dry-season water availability? Does it interact with the glacier coupling approach (Paper 2)?

**RQ4 — Groundwater as the "next buffer":**
As glaciers retreat, does groundwater storage partially replace glacier buffering? Can we quantify when groundwater buffering capacity is exhausted (following Somers et al. 2019)?

---

## Research Gap

### The paradigm shift: groundwater in mountains matters more than we thought

Two recent high-profile papers have fundamentally changed the understanding:

- **Carroll, R.W.H. et al. (2024)** *Nature Water*, 2, 419–433 — https://www.nature.com/articles/s44221-024-00239-0
  *"Declining groundwater storage expected to amplify mountain streamflow reductions in a warmer world"* — Used integrated model extending 400 m into subsurface in Colorado River headwaters. **Including groundwater decline nearly DOUBLED projected streamflow reductions** compared to models without deep groundwater. Without representing groundwater, you underestimate the problem by half.

- **Communications Earth & Environment (2025)** — https://www.nature.com/articles/s43247-025-02303-3
  *"Groundwater dominates snowmelt runoff and controls streamflow efficiency in the western United States"* — **58% of snowmelt runoff is actually old groundwater** (average age 5.7 ± 4.3 years). Snowmelt pushes out stored groundwater rather than flowing directly to streams. Geology controls this: hard rock = younger water, less storage; sandstone = older water, more storage.

### What exists

**Reviews and foundations:**
- **Somers, L.D. & McKenzie, J.M. (2020)** *WIREs Water* — https://wires.onlinelibrary.wiley.com/doi/full/10.1002/wat2.1475
  *"A review of groundwater in high mountain environments"* — Established that groundwater in alpine zones was long considered negligible but is actually ubiquitous (talus, moraine, fractured bedrock aquifers). Severely understudied outside European Alps.

- **Somers, L.D. et al. (2019)** *GRL* — https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019GL084730
  *"Groundwater Buffers Decreasing Glacier Melt in an Andean Watershed—But Not Forever"* — Groundwater can temporarily replace glacier buffering as glaciers retreat, but the aquifer eventually depletes too. Critical finding for the glacier → groundwater transition.

**Baseflow model structure:**
- **Stoelzle, M. & Weiler, M. (2015)** *Hydrological Processes* — https://onlinelibrary.wiley.com/doi/abs/10.1002/hyp.10251
  *"Is there a superior conceptual groundwater model structure for baseflow simulation?"* — Tested 9 different two-parameter conceptual GW models across different aquifer types. Found aquifer-specific optimal structures: fractured/karstic aquifers need different models than porous aquifers. No universal best structure.

**Alpine groundwater dynamics:**
- **Müller, T., Lane, S.N. & Schaefli, B. (2022)** *HESS*, 26, 6029–6054 — https://hess.copernicus.org/articles/26/6029/2022/
  *"Towards a hydrogeomorphological understanding of proglacial catchments"* — Otemma glacier, Swiss Alps. Steep zones store water for days; flatter zones sustain baseflow for weeks. Identified both fluvial and bedrock aquifer storage in proglacial areas.

- **Vincent, A. et al. (2024)** *HESS*, 28, 3475–3494 — https://hess.copernicus.org/articles/28/3475/2024/
  *"A hydrogeological conceptual model of aquifers in catchments headed by temperate glaciers"* — Iceland (Vatnajökull). Two distinct aquifers identified. Subglacial recharge 4× higher than proglacial. Demonstrates that aquifer structure differs fundamentally from what conceptual models assume.

- **HESS (2024)** — https://hess.copernicus.org/articles/28/735/2024/
  *"Current and future roles of meltwater–groundwater dynamics in a proglacial Alpine outwash plain"* — Stream infiltration is the dominant recharge process. Outwash plains can sustain groundwater levels for months without glacier input.

- **Fan, X., Hofmeister, F., Schaefli, B. & Chiogna, G. (2025)** *EGUsphere* — https://egusphere.copernicus.org/preprints/2025/egusphere-2025-1500/
  *"Physics-based simulation of hydrological processes in a high-elevation glaciated environment focusing on groundwater"* — WaSiM with 2D groundwater module in the Martell Valley, European Alps. One of the first physics-based groundwater simulations in a glaciated environment.

**Water age and residence times:**
- **Betterle, A. et al. (2024)** *WRR* — https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024WR037407
  *"Morphological and Hydrogeological Controls of Groundwater Flows and Water Age Distribution in Mountain Aquifers and Streams"* — Water residence times range from 1 to 50+ years in fractured mountain bedrock.

**Climate change impacts on mountain groundwater:**
- **npj Climate and Atmospheric Science (2024)** — https://www.nature.com/articles/s41612-024-00840-w
  *"The slowdown of increasing groundwater storage in response to climate warming in the Tibetan Plateau"* — GRACE-based. Groundwater storage was increasing (permafrost thaw + melt recharge) but is now slowing.

- **HESS (2025)** — https://hess.copernicus.org/articles/29/4055/2025/
  *"Trends in hydroclimate extremes: how changes in winter affect water storage and baseflow"* — Warmer winters increase mid-winter snowmelt but reduce recharge that sustains summer baseflow.

- **Nature Water (2024)** — https://www.nature.com/articles/s44221-024-00243-4
  *"Mountain streamflow threatened by irreversible simulated groundwater declines"* — Commentary on Carroll et al., emphasizing the irreversibility of groundwater depletion.

**Permafrost-groundwater interactions:**
- **Frontiers in Earth Science (2023)** — https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2023.1254309/full
  *"Permafrost and groundwater interaction: current state and future perspective"* — Thawing permafrost increases active layer thickness, creates new aquifers, amplifies groundwater flow and baseflow. Relevant for Karakoram where permafrost is extensive.

### What doesn't exist

1. **Systematic assessment of baseflow bias in conceptual models across glacierized HMA catchments** — the baseflow problem is well-known anecdotally but nobody has documented it systematically across multiple catchments with different characteristics
2. **Structural model improvements for deep groundwater tested in HMA** — all the recent physics-based groundwater work (Carroll 2024, Fan 2025) is in the Americas or European Alps, not HMA
3. **Connection between baseflow representation and future projection reliability** — Carroll (2024) showed it matters for Colorado; does it matter for UIB where glacier melt dominates summer flow?
4. **The glacier → groundwater buffering transition in HMA** — Somers (2019) showed it for one Andean watershed; nobody has tested whether HMA catchments have sufficient aquifer storage to buffer glacier loss even temporarily

---

## The Problem in Your Model

### Observed behavior
- SLOW_RESERVOIR (SOIL[2]) shows annual fluctuations but **persistent rising trend** — accumulating water year over year
- Winter baseflow underestimated — observed streamflow shows sustained winter flow that the model cannot reproduce
- HBV's two-reservoir structure (FAST_RES + SLOW_RES) with linear baseflow recession is structurally inadequate for multi-year groundwater dynamics

### Why this happens
- Summer melt percolates into SLOW_RES via PERC_CONSTANT
- BASEFLOW_COEFF_SLOW is either too low (water doesn't drain) or the optimizer can't simultaneously match summer peaks and winter baseflow
- No mechanism for multi-year storage or water with residence times of 5+ years (as documented by the 2025 Comms Earth paper)
- KGE/NSE calibration weights summer peaks heavily — no incentive to get winter baseflow right

### Potential fixes within Raven
1. **Add a DEEP_GW layer** — third reservoir with very slow recession (0.001/day, ~3 year residence time)
2. **Add deep seepage loss** — loss term from SLOW_RES for regional groundwater that exits the catchment
3. **Widen parameter bounds** for BASEFLOW_COEFF_SLOW
4. **Multi-objective calibration** including winter baseflow as explicit target

---

## Study Design

*Preliminary — to be refined after diagnostic analysis*

### Phase 1: Diagnostic (RQ1)
1. Extract SOIL[2] timeseries from existing calibrated runs across all 12 catchments
2. Quantify winter baseflow bias (observed vs simulated) for each catchment
3. Relate bias to catchment characteristics (glacier fraction, elevation, area, geology if available)
4. Check whether bias differs between coupling approaches (uncoupled vs GloGEM-coupled)

### Phase 2: Structural improvement (RQ2)
1. Add DEEP_GW layer to HBV in Raven (SOIL[3] with BASE_LINEAR, very slow recession)
2. Re-calibrate with multi-objective function (KGE on full hydrograph + winter baseflow component)
3. Compare: does the 4-layer model improve winter flow without degrading summer performance?
4. Test across all 12 catchments — is the improvement universal or catchment-dependent?

### Phase 3: Projection implications (RQ3)
1. Run future projections with both 3-layer (original) and 4-layer (improved) structures
2. Compare: do future low-flow / winter flow projections differ?
3. Quantify: how much does baseflow representation uncertainty contribute to total projection uncertainty relative to coupling/forcing uncertainty (Paper 2)?

### Phase 4: Groundwater as buffer (RQ4)
1. Track deep groundwater storage trajectories under future climate
2. Identify when groundwater recharge (from declining glacier melt) can no longer sustain the deep reservoir
3. Compare with glacier buffering decline timelines from Paper 3

---

## Contribution

This would be the **first study to**:
1. Systematically document baseflow bias in conceptual models across multiple glacierized HMA catchments
2. Test structural groundwater improvements (deep reservoir) in a glacierized HMA context — extending Carroll et al. (2024) findings from Colorado to the UIB
3. Quantify how baseflow representation affects future projection reliability for winter/dry-season flows
4. Assess whether groundwater can temporarily buffer glacier retreat in HMA — extending Somers et al. (2019) from the Andes

## Connection to other papers
- **Paper 2**: Does coupling approach affect the baseflow problem? Does correcting baseflow change which configuration is "best"?
- **Paper 3**: Groundwater buffering is the "next chapter" after glacier buffering declines
- **Paper 4**: Winter low flows are a decision-relevant threshold — if baseflow is wrong, the threshold analysis is wrong
