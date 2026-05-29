# Multi-Objective Calibration Literature — Snow/Glacier Hydrology

Compiled 2026-05-29 to support Justine's SPHY-in-Raven calibration for Swiss alpine + Upper Indus catchments (3 objectives: Q-KGE, MODIS fSCA per elevation band, Eckhardt winter baseflow KGE; SCEUA weighted-sum vs NSGA-II Pareto). Grouped by the four axes you specified. Entries marked **[DOI not verified]** were inferred from search results rather than confirmed on the publisher landing page. PDFs in `multiobj_pdfs/` are flagged with **[PDF]**.

Anchor references you have already read (van Tiel et al. 2020, Konz & Seibert 2010, Parajka & Blöschl 2008, Finger et al. 2015, Hugonnet 2021, Huang 2026, Mourot 2025, Kissel 2024) are *not* re-summarised; cross-references to them appear under the appropriate axis where their methodological position is relevant.

---

## Axis 1 — Algorithm family (multi-objective optimisation)

### 1.1 Foundational evolutionary / Pareto-front algorithms

- **Deb, K., Pratap, A., Agarwal, S., Meyarivan, T. (2002)** *IEEE Trans. Evolutionary Computation* 6, 182–197 — "A fast and elitist multiobjective genetic algorithm: NSGA-II." DOI: 10.1109/4235.996017
- **Vrugt, J.A., Robinson, B.A. (2007)** *PNAS* 104, 708–711 — "Improved evolutionary optimization from genetically adaptive multimethod search (AMALGAM)." DOI: 10.1073/pnas.0610471104 **[DOI not verified]**
- **Vrugt, J.A., Gupta, H.V., Bastidas, L.A., Bouten, W., Sorooshian, S. (2003)** *WRR* 39, 1214 — "Effective and efficient algorithm for multiobjective optimization of hydrologic models (MOSCEM-UA)." DOI: 10.1029/2002WR001746 **[DOI not verified]**
- **Hadka, D., Reed, P. (2013)** *Evolutionary Computation* 21, 231–259 — "Borg: An Auto-Adaptive Many-Objective Evolutionary Computing Framework." DOI: 10.1162/EVCO_a_00075. Auto-adaptive operator selection + ε-dominance + ε-progress + randomized restarts. Outperforms NSGA-II/εNSGA-II on most DTLZ/WFG benchmarks especially as the number of objectives grows past 3.
- **Asadzadeh, M., Tolson, B.A. (2013)** *Engineering Optimization* 45, 1489–1509 — "Pareto archived dynamically dimensioned search (PADDS) with hypervolume-based selection." DOI: 10.1080/0305215X.2012.748046 **[DOI not verified]**. The Pareto-version of DDS — explicitly designed for *expensive* models with budget < 10 000 evaluations, exactly the regime SPHY-in-Raven sits in.
- **Tolson, B.A., Shoemaker, C.A. (2007)** *WRR* 43, W01413 — "Dynamically dimensioned search algorithm for computationally efficient watershed model calibration (DDS)." DOI: 10.1029/2005WR004723. Foundational single-objective basis for PADDS; one tuning parameter; requires only 15–20 % of SCE-UA evaluations.

### 1.2 Multi-objective calibration in hydrology — methodology / comparisons

- **Shafii, M., De Smedt, F. (2009)** *HESS* 13, 2137–2149 — "Multi-objective calibration of a distributed hydrological model (WetSpa) using a genetic algorithm." DOI: 10.5194/hess-13-2137-2009. **[PDF]** NSGA-II with {NSE, NSE(log Q)} — canonical worked example of two-objective Pareto calibration in HESS.
- **Krauße, T., Cullmann, J., Saile, P., Schmitz, G.H. (2012)** *HESS* 16, 3579–3606 — "Robust multi-objective calibration strategies — possibilities for improving flood forecasting." DOI: 10.5194/hess-16-3579-2012. **[PDF]** MOROPE method; combines multi-obj with depth-based parameter sampling for robustness — useful framing of *which* points on the Pareto front to pick.
- **Monteil, C., Zaoui, F., Le Moine, N., Hendrickx, F. (2020)** *HESS* 24, 3189–3209 — "Multi-objective calibration by combination of stochastic and gradient-like parameter generation rules — the caRamel algorithm." DOI: 10.5194/hess-24-3189-2020. **[PDF]** caRamel hybridises NSGA-II-style global search with Delaunay-triangulation-based local moves. On hydrological test cases caRamel reaches a similar Pareto front to NSGA-II but with faster hypervolume convergence — relevant if NSGA-II runtime starts to bite at HKH-scale.
- **Efstratiadis, A., Koutsoyiannis, D. (2010)** *Hydrol. Sci. J.* 55, 58–78 — "One decade of multi-objective calibration approaches in hydrological modelling: a review." DOI: 10.1080/02626660903526292 **[DOI not verified]**. Still the most-cited review of the algorithm-level landscape.
- **Pesce, M., Viglione, A., von Hardenberg, J., Tarasova, L., Basso, S., Merz, R., Parajka, J., Tong, R. (2024)** *Proc. IAHS* 385, 65–70 — "Regional multi-objective calibration for distributed hydrological modelling: a decision tree based approach." DOI: 10.5194/piahs-385-65-2024 **[DOI not verified]**. Decision-tree to pre-screen which objective combinations a given catchment "needs" — interesting for ungauged HKH extension.

### 1.3 Weighted-sum vs Pareto — the methodological debate

- The classical mathematical-optimisation result: **weighted-sum aggregation only spans convex parts of the Pareto front**; non-convex regions are unreachable regardless of weight choice (Das & Dennis 1997 *SIAM J. Optim.*; Marler & Arora 2010 *Struct. Multidiscip. Optim.* 41, 853–862 [DOI 10.1007/s00158-009-0460-7]). In hydrology this matters whenever trade-offs between objectives are competitive rather than cooperative — which is *always* the case once snow, baseflow, and Q-peaks are mixed.
- **Madsen, H. (2000, 2003)** *J. Hydrol.* and *Adv. Water Resour.* — early advocate of Pareto-front methods for rainfall-runoff calibration; demonstrated that weighted-aggregation hides trade-offs that practitioners need to see.
- **Khu, S.T., Madsen, H. (2005)** *WRR* 41, W03004 — "Multiobjective calibration with Pareto preference ordering: an application to rainfall-runoff model calibration." DOI: 10.1029/2004WR003041 **[DOI not verified]**. Pareto + a posteriori preference structure as a defensible alternative to a priori weights.
- See also Cinkus et al. 2023 (Axis 2) on how weighted-aggregation can hide compensating bias errors inside individual KGE components.

### 1.4 ML / surrogate-augmented calibration

- **Farahani, M.A., Wood, A.W., Tang, G., Mizukami, N. (2025)** *HESS* 29, 4515–4537 — "Calibrating a large-domain land/hydrology process model in the age of AI: the SUMMA CAMELS emulator experiments." DOI: 10.5194/hess-29-4515-2025. **[PDF]** Joint ML emulator trained across all CAMELS basins outperforms per-basin emulators (median KGE 0.76 cal / 0.69 val vs 0.69 / 0.65) and cuts compute by a factor of ~600 (6 emulators vs 3 763). State-of-the-art for the "many catchments × expensive forward model" regime.
- **Tang, G., Wood, A.W., et al. (2025)** *WRR* 61, e2024WR039525 — "On using AI-based large-sample emulators for land/hydrology model calibration and regionalization." DOI: 10.1029/2024WR039525. Paywalled — find preprint. Companion / extension of Farahani 2025; explicitly addresses regionalisation to ungauged basins via emulator.
- **Jiang, S., Zheng, Y., Solomatine, D. (2020)** *GRL* 47, e2020GL088229 — "Improving AI system awareness of geoscience knowledge: symbiotic integration of physical approaches and deep learning." Hybrid physics+ML calibration framing. DOI: 10.1029/2020GL088229 **[DOI not verified]**
- **Tsai, W.-P., Feng, D., Pan, M., et al. (2021)** *Nature Communications* 12, 5988 — "From calibration to parameter learning: harnessing the scaling effects of big data in geoscientific modeling." DOI: 10.1038/s41467-021-26107-z. The "differentiable parameter learning" idea — calibrate per-catchment-attribute *mappings* rather than per-catchment parameter sets. Becoming mainstream for CAMELS-scale work, not yet for individual alpine catchments.
- **Wu, S., Tetzlaff, D., et al. (2025)** *WRR* 61, e2024WR037656 — "Revising common approaches for calibration: insights from a 1-D tracer-aided hydrological model with high-dimensional parameters and objectives." DOI: 10.1029/2024WR037656 **[DOI not verified]**. Explicitly tests common aggregation choices in the multi-tracer regime.

### 1.5 Behavioural / GLUE-style alternatives

- **Beven, K. (2006)** *J. Hydrol.* 320, 18–36 — "A manifesto for the equifinality thesis." DOI: 10.1016/j.jhydrol.2005.07.007
- **Khatami, S., Peel, M.C., Peterson, T.J., Western, A.W. (2019)** *WRR* 55, 8922–8941 — "Equifinality and flux mapping: a new approach to model evaluation and process representation under uncertainty." DOI: 10.1029/2018WR023750. Paywalled — find preprint via Monash repository (https://research.monash.edu/en/publications/equifinality-and-flux-mapping-a-new-approach-to-model-evaluation-/). Argues that "best-fit" parameter sets hide enormous flux-partitioning equifinality; instead, *map* the behavioural ensemble onto flux-space and use that as the calibration product. Directly relevant to Justine's question of whether SCEUA's single weighted-sum optimum is informative.
- **Shannon, S., Payne, A., Freer, J., Coxon, G., Kauzlaric, M., Kriegel, D., Harrison, S. (2023)** *HESS* 27, 453–480 — "A snow and glacier hydrological model for large catchments — case study for the Naryn River, Central Asia." DOI: 10.5194/hess-27-453-2023. **[PDF]** Explicit GLUE implementation with 6 equally-weighted likelihoods (NSE, PBIAS, RSR seasonal) on a 56 000-km² snow/glacier catchment in the Tien Shan. Best 0.5 % of 150 100 LHS samples retained. Shows the GLUE workflow scaled to HKH-style basins.
- **Stefnisdóttir, S., Sikorska-Senoner, A.E., Ásgeirsson, E.I., Finger, D.C. (2021)** *HESS Discuss.* preprint hess-2021-325 — "Improving the Pareto Frontier in multi-dataset calibration of hydrological models using metaheuristics." **[PDF]** (preprint, not accepted). HBV on a small Rhone-glacier catchment, three objectives {Q-NSE, SCA, glacier MB-RMSE}. Compares Monte Carlo vs Simulated Annealing vs Genetic Algorithm — GA wins (narrowest CIs across all three variables, best Pareto coverage). Worth reading as a direct methodological precedent even though peer review rejected it.
- **Coxon, G., Freer, J., Wagener, T., Odoni, N.A., Clark, M.P. (2014)** *Hydrol. Process.* 28, 6135–6150 — "Diagnostic evaluation of multiple hypotheses in a limits-of-acceptability framework for 24 UK catchments." DOI: 10.1002/hyp.10096 **[DOI not verified]**

### 1.6 Hierarchical / stepwise calibration

- This is the dominant SPHY workflow in HKH (Terink 2015, Lutz 2014, Wijngaard 2018, Khanal 2021): calibrate snow params against MODIS first, then glacier melt against geodetic MB, then baseflow + routing against Q. **van Tiel et al. 2020** is the canonical critique of this — they note that stepwise approaches risk locking in compensating biases from earlier steps and recommend simultaneous multi-criteria calibration when computational budget allows.
- **Khanal, S., Nick, F., Fiddes, J., Kraaijenbrink, P., Immerzeel, W., Droogers, P., van Ravesteyn, P., Schults, T., Hunink, J.E. (2025)** FutureWater Report 265 — "Guidelines for Glacio-hydrological Modelling in High Mountain Asia." **[PDF]** Current best-practice stepwise SPHY calibration recipe from the developers; explicit on what to calibrate against what in HMA.
- **Pellicciotti, F., Buergi, C., Immerzeel, W.W., Konz, M., Shrestha, A.B. (2012)** *Mountain Research and Development* 32, 39–50 — "Challenges and uncertainties in hydrological modeling of remote HKH basins: suggestions for calibration strategies." DOI: 10.1659/MRD-JOURNAL-D-11-00092.1. **[PDF]** Still the canonical statement of *why* HKH calibration goes wrong (precipitation under-estimation compensated by melt over-estimation) and *why* multi-objective constraints are essential there.
- **Konz, M., Seibert, J. (2010)** *J. Hydrol.* 385, 238–246 — "On the value of glacier mass balances for hydrological model calibration." DOI: 10.1016/j.jhydrol.2010.02.025. **[PDF]** The foundational paper showing that Q-only calibration in glacierized basins is structurally biased; adding MB constrains the partitioning.

---

## Axis 2 — Objective scale & combination (THE scale-mismatch problem)

This is where the most decision-relevant recent literature sits for Justine.

### 2.1 Critical evaluation of single performance metrics

- **Cinkus, G., Mazzilli, N., Jourde, H., Wunsch, A., Liesch, T., Ravbar, N., Chen, Z., Goldscheider, N. (2023)** *HESS* 27, 2397–2411 — "When best is the enemy of good — critical evaluation of performance criteria in hydrological models." DOI: 10.5194/hess-27-2397-2023. **[PDF]** *The* paper on how KGE/KGE′/KGE′′/LME can mask compensating errors in bias and variability (they sum to ~2/3 of the criterion weight, and can cancel each other to inflate the score). Recommends modified index of agreement d₁, non-parametric KGE (KGE-NP), or diagnostic efficiency; advocates multi-criteria + expert assessment over any single metric. Directly relevant to Justine's choice to use KGE on Q.
- **Pool, S., Vis, M., Seibert, J. (2018)** *Hydrol. Sci. J.* 63, 1941–1953 — "Evaluating model performance: towards a non-parametric variant of the Kling-Gupta efficiency." DOI: 10.1080/02626667.2018.1552002. KGE-NP replaces Pearson r with Spearman rank and σ with the FDC mean-absolute-error, removing distributional assumptions. Worth adopting for the Q objective.
- **Gupta, H.V., Kling, H., Yilmaz, K.K., Martinez, G.F. (2009)** *J. Hydrol.* 377, 80–91 — "Decomposition of the mean squared error and NSE performance criteria: implications for improving hydrological modelling." DOI: 10.1016/j.jhydrol.2009.08.003 **[DOI not verified]**. The original KGE paper — already standard, but worth re-reading in light of Cinkus 2023.

### 2.2 Calibration-metric *selection* in snow-dominated mountains (your case)

- **Araya, D., Mendoza, P.A., Muñoz-Castro, E., McPhee, J. (2023)** *HESS* 27, 4385–4408 — "Towards robust seasonal streamflow forecasts in mountainous catchments: impact of calibration metric selection in hydrological modeling." DOI: 10.5194/hess-27-4385-2023. **[PDF]** 22 Chilean snow-mountain catchments × 3 conceptual models (GR4J, TUW, Sacramento) × 12 objective functions including seasonal melt-period metrics. Conclusion: **KGE(Q) + NSE(log Q)** gives the best compromise between hydrological consistency and seasonal forecast skill, with only ~5 % median forecast-skill loss. The most directly transferable empirical result for your Q-objective choice in snow-fed catchments.
- **Pool, S., Vis, M.J.P., Knight, R.R., Seibert, J. (2017)** *HESS* 21, 5443–5457 — "Streamflow characteristics from modeled runoff time series — importance of calibration criteria selection." DOI: 10.5194/hess-21-5443-2017. **[PDF]** The "you only get what you optimise for" paper. If you want to reproduce SFC X, calibrate against an objective that targets SFC X explicitly; relying on KGE alone biases low-flow, high-flow, and FDC slope simulations differently. Argues *for* multi-objective frameworks with hydrological-signature objectives.
- **Mendoza, P.A., et al. (and follow-up)** *Hydrol. Sci. J.* 2023 — "Exploring parameter (dis)agreement due to calibration metric selection in conceptual rainfall-runoff models." DOI: 10.1080/02626667.2023.2231434. Companion to Araya 2023; same group; quantifies how *parameter values* (not just performance) depend on metric choice.

### 2.3 Multi-objective aggregation, weighting, scaling

- **Efstratiadis, A., Koutsoyiannis, D. (2010)** — see Axis 1.2. Reviews aggregation schemes used in hydrology, including transformation-based normalisation (each objective rescaled to [0, 1] via min/max from a baseline run before weighting).
- **Hanus, S., Schuster, L., Burek, P., Maussion, F., Wada, Y., Viviroli, D. (2024)** *GMD* 17, 5123–5144 — "Coupling a large-scale glacier and hydrological model (OGGM v1.5.3 and CWatM V1.08)." DOI: 10.5194/gmd-17-5123-2024. **[PDF]** Uses NSGA-II with a 0.8/0.2 weighting of KGE(Q) and a "snow-cover penalty" — but this is a *penalty* term (large only when snow accumulates unphysically), not a symmetric objective. Useful as an example of how to inject a snow constraint without it dominating the Q signal. *Caveat* the authors found: parameters calibrated under this weighting compensated for absent glacier dynamics in the calibration period and over-estimated future discharge under reduced melt scenarios.
- **Pechlivanidis, I.G., Jackson, B., McMillan, H., Gupta, H.V. (2014)** *WRR* 50, 8066–8083 — "Use of an entropy-based metric in multi-objective calibration to improve model performance." DOI: 10.1002/2013WR014537 **[DOI not verified]**. Conditional Entropy Difference (CED) as a multi-criteria signature; behavioural-set selection with cutoff thresholds (KGE > 0.75, > 0.85; CED thresholds). Methodology directly applicable to defining a "behavioural" subset of NSGA-II Pareto solutions.
- **Westerberg, I.K., McMillan, H.K. (2015)** *HESS* 19, 3951–3968 — "Uncertainty in hydrological signatures." DOI: 10.5194/hess-19-3951-2015. Quantifies that signatures have ±10–40 % relative uncertainty — sets a floor below which improving multi-objective metrics is meaningless.
- **Khatami, S., et al. (2019)** — see Axis 1.5. Flux-mapping as the practical answer to "two objectives give similar scores but very different process partitions."

### 2.4 The structural-scale problem in your specific 3-objective setup

Practical scale-normalisation schemes catalogued from the above:

1. *Min–max rescaling from a baseline run.* Run NSGA-II for 50 generations on equal weights, observe the achievable range of each objective, then min-max rescale so each metric lives in [0, 1]. Used in Hanus 2024.
2. *Behavioural-threshold ε-aggregation.* Adopt ε-NSGA-II logic (Borg, PADDS): require each objective above a behavioural ε before parameter is considered. Equivalent to the limits-of-acceptability framework (Beven 2006, Coxon 2014).
3. *Hierarchical Pareto with preference ordering.* Khu & Madsen 2005. Two-stage: pick Pareto subset where Q-KGE > 0.7, then within that subset rank by snow/baseflow.
4. *Native [0, 1] transformations.* Replace 1−RMSE/μ for snow with KGE(SCA per band); use NSE(log Q) winter-only instead of KGE(BFsim, BFobs); both naturally bounded near 1 and structurally comparable to KGE(Q). This is what Araya 2023 ends up recommending.

---

## Axis 3 — Which calibration targets are actually used in glacier/snow modelling?

### 3.1 Streamflow variants

- Standard KGE / NSE / KGE-NP. Pool 2018, Cinkus 2023 (Axis 2.1).
- Seasonal sub-windows (snowmelt period, baseflow recession). Araya 2023 (Axis 2.2), Huang 2026 (winter-only baseflow KGE).
- Log-transformed (NSE-log) for low-flow emphasis. Araya 2023.
- Split-period KGE — calibrate independently on melt-season vs accumulation-season. Used by Stahl group, Schaefli group.

### 3.2 Snow

- **Basin-mean MODIS SCA.** Parajka & Blöschl 2008 (canonical).
- **Per-elevation-band MODIS fSCA** — Finger et al. 2015 (canonical, your reference).
- **Spatially distributed SWE.** Tiwari, D., Trudel, M., Leconte, R. (2024) *HESS* 28, 1127–1146 — "On optimization of calibrations of a distributed hydrological model with spatially distributed information on snow." DOI: 10.5194/hess-28-1127-2024. **[PDF]** PADDS with NSE(Q) + RMSE(SWE-mean) + SPAEF(SWE-spatial) on a Canadian sub-arctic basin. Headline: **spatial SWE > spatially-averaged SWE > Q-only**. Validates the choice to use per-elevation-band rather than basin-mean snow data.
- **In-situ SWE / SnowCAST / OSHD (Switzerland).** Magnusson, Jonas, Lehning et al. — OSHD operational product widely used in Swiss alpine hydrology.
- **MODIS Terra+Aqua merged (cloud-reduced).** Parajka & Blöschl 2008; Tong et al. 2021 (Axis 3.6).
- **Snow line elevation (SLE) from MODIS.** Used as an alternative target when fSCA is saturated; references in van Tiel 2020.
- **Sentinel-2 / Landsat snow line** (Rastner et al. 2019) — sub-seasonal MB constraint at glacier scale.
- **Gyawali, D.R., Bárdossy, A. (2022)** *HESS* 26, 3055–3077 — "Development and parameter estimation of snowmelt models using spatial snow-cover observations from MODIS." DOI: 10.5194/hess-26-3055-2022. Two-stage calibration: snow first against Brier-score on MODIS, then Q. Useful structural recipe.

### 3.3 Glacier mass balance

- **Konz & Seibert 2010** — canonical for adding glacier-wide annual MB.
- **Hugonnet et al. 2021** (your reference) — global per-glacier elevation-band geodetic MB, now the standard remote-sensing constraint for HKH where in-situ MB doesn't exist.
- **Hanus et al. 2024** (Axis 1.4 / 2.3) — calibrates a *separate* temperature-index MB model per glacier against Hugonnet 2000–2019 mass change, then feeds glacier runoff into CWatM. The two-step / pre-calibrated approach is increasingly standard for large-domain studies.
- **Sentinel-2 sub-seasonal MB.** Constraining sub-seasonal glacier MB in the Swiss Alps using Sentinel-2-derived snow-cover observations (Cambridge J. Glaciology, Saber & Huss?). Worth searching when adding finer temporal MB constraint.

### 3.4 ET (evapotranspiration)

- **GLEAM.** Generally outperforms MOD16 as a calibration target — incorporating GLEAM AET in a calibration leads the model to reproduce ET closer *and* simultaneously improves Q simulations (sciencedirect S2214581822001057, multiple studies).
- **MOD16.** Mixed performance; sometimes degrades Q calibration when added.
- For glacier catchments ET is a small term in the water balance and *probably not worth adding* unless the catchment includes large forested or wetland areas (most Swiss catchments do at lower elevations).

### 3.5 TWS (GRACE / GRACE-FO)

- **Bai, P., Liu, X., Liu, C. (2018)** *J. Hydrol.* 557, 291–304 — "Improving hydrological simulations by incorporating GRACE data for model calibration." DOI: 10.1016/j.jhydrol.2017.12.025 **[DOI not verified]**. Methodology paper.
- **Chen, X., Long, D., Hong, Y., et al. (2017)** *WRR* 53, 2431–2466 — "Improved modeling of snow and glacier melting by a progressive two-stage calibration strategy with GRACE and multisource data" in the Upper Brahmaputra. DOI: 10.1002/2016WR019656 **[DOI not verified]**. Direct HMA precedent.
- Caveat: GRACE footprint (~300–500 km) is too coarse for individual UIB sub-basins like Hunza; useful only at whole-Indus scale, where it would constrain the *aggregate* multi-catchment ensemble rather than a single calibration.

### 3.6 Multi-source combined satellite

- **Tong, R., Parajka, J., Salentinig, A., et al. (2021)** *HESS* 25, 1389–1410 — "The value of ASCAT soil moisture and MODIS snow cover data for calibrating a conceptual hydrologic model." DOI: 10.5194/hess-25-1389-2021. **[PDF]** Three calibration variants across 213 Austrian catchments. Adding ASCAT helps soil moisture, adding MODIS helps snow — both can be added together without degrading the other. Q-efficiency stable across variants. Demonstrates that auxiliary data improves the *internal* state representation more than it improves Q itself.
- **Avesani, D., Nan, Y., Tian, F. (2025)** *HESS* 29, 5755–5775 — "Reducing hydrological uncertainty in large mountainous basins: the role of isotope, snow cover, and glacier dynamics in capturing streamflow seasonality." DOI: 10.5194/hess-29-5755-2025. **[PDF]** Bayesian GLUE-like multi-likelihood (Q + SCA + GMB + δ¹⁸O isotope). Key result: **isotopes do the most uncertainty-reduction work for low flows / baseflow** — SCA and GMB are useful only during their respective active seasons. Justifies adding isotope tracers as a high-value 4th constraint *if available*. For UIB, isotope data is largely absent; for Switzerland, several Alpine catchments now have multi-year δ¹⁸O.

### 3.7 Signature-based

- **McMillan, H.K. (2021)** *WIREs Water* 8, e1499 — "A review of hydrologic signatures and their applications." DOI: 10.1002/wat2.1499. Catalogue of >200 signatures.
- **Gnann, S.J., Howden, N.J.K., Woods, R.A. (2021)** *WRR* 57, e2020WR028354 — "Including Regional Knowledge Improves Baseflow Signature Predictions in Large Sample Hydrology." DOI: 10.1029/2020WR028354. Treats BFI as a calibration signature.
- **Euser, T., Winsemius, H.C., Hrachowitz, M., Fenicia, F., Uhlenbrook, S., Savenije, H.H.G. (2013)** *HESS* 17, 1893–1912 — "A framework to assess the realism of model structures using hydrological signatures." DOI: 10.5194/hess-17-1893-2013. Signature-based model-structure rejection.
- **Hanus, S. (2024)** GMD (Axis 1.4 / 2.3) — uses a "snow penalty" which is essentially a signature constraint.

### 3.8 Information content / which target is "worth" adding?

Direct evidence from multi-likelihood GLUE-style studies (Avesani 2025, Shannon 2023, Tong 2021, Finger 2015): roughly ranked by information content for snow/glacier basins —

1. **Q (full record)** — sets the overall water balance; necessary but not sufficient.
2. **Per-elevation-band snow (MODIS fSCA or in-situ SWE)** — strongest second constraint. Tiwari 2024 shows spatial > basin-mean.
3. **Glacier-wide annual or geodetic MB** — constrains the snow vs ice melt partition; essential in glacierised catchments (Konz & Seibert 2010).
4. **Winter baseflow KGE / BFI** — Huang 2026 result: critical for the Pamir/HKH winter discharge regime; constrains the slow storage compartment.
5. **Isotopes (δ¹⁸O, δ²H)** — best for low-flow / baseflow age constraint (Avesani 2025), but data-limited.
6. **ET (GLEAM > MOD16)** — useful only when the catchment has substantial vegetated fraction.
7. **GRACE TWS** — only useful at large catchment aggregations (>1000 km² typically much more).

---

## Axis 4 — SPHY-specific and Indus/HKH-specific multi-objective work

### 4.1 SPHY model papers

- **Terink, W., Lutz, A.F., Simons, G.W.H., Immerzeel, W.W., Droogers, P. (2015)** *GMD* 8, 2009–2034 — SPHY v2.0. DOI: 10.5194/gmd-8-2009-2015.
- **Lutz, A.F., Immerzeel, W.W., Shrestha, A.B., Bierkens, M.F.P. (2014)** *Nature Climate Change* 4, 587–592 — UIB SPHY application; baseline for the standard SPHY HKH calibration recipe.
- **Wijngaard, R.R., Lutz, A.F., Nepal, S., Khanal, S., Pradhananga, S., Shrestha, A.B., Immerzeel, W.W. (2017)** *PLOS ONE* 12, e0190224 — Future hydro-climatic extremes in UIB, Ganges, Brahmaputra. DOI: 10.1371/journal.pone.0190224 **[DOI not verified]**. Stepwise SPHY calibration: snow first against MODIS, then Q at two UIB gauges.
- **Khanal, S., Lutz, A.F., Kraaijenbrink, P.D.A., et al. (2021)** *WRR* 57, e2020WR029266 — "Variable 21st century climate change response for rivers in HMA." DOI: 10.1029/2020WR029266. SPHY scaled across multiple HMA basins.
- **Huang, Y., Saidi, A., et al. (2026)** *WRR* 62, e2025WR040043 — "Winter Baseflow Calibration's Critical Role in Hydrological Modeling for the Pamir Region." **[PDF — EarthArXiv preprint]**. Modifies SPHY with a second linear GW reservoir; shows that calibrating against {daily Q + winter baseflow} *alone* recovers good performance on snow + MB without those targets being in the objective set. Strong endorsement of the winter-baseflow constraint that you have already adopted.
- **Khanal, S., Nick, F., Fiddes, J., Kraaijenbrink, P., Immerzeel, W., Droogers, P., et al. (2025)** FutureWater Report 265 — Guidelines for Glacio-hydrological Modelling in High Mountain Asia. **[PDF]**. Current developer-recommended SPHY workflow; useful sanity-check for what's "default" and where Justine deviates.
- **Khanal, S., Lutz, A.F., Eekhout, J., Terink, W. (2024)** FutureWater Report 248 — SPHY v3.1 manual. **[PDF link in references]**

### 4.2 HKH / UIB calibration practice — non-SPHY

- **Pellicciotti, F., et al. (2012)** — Axis 1.6. Canonical statement of HKH calibration challenges.
- **Bocchiola, D., Diolaiuti, G., Soncini, A., et al. (2011)** *HESS* 15, 2059–2075 — "Prediction of future hydrological regimes in poorly gauged high altitude basins: the case study of the upper Indus, Pakistan." DOI: 10.5194/hess-15-2059-2011. Open-access; high-altitude UIB.
- **Immerzeel, W.W., Wijngaard, R.R., Lutz, A.F., et al. (2015)** *HESS* 19, 4673–4687 — "Reconciling high-altitude precipitation in the Upper Indus basin with glacier mass balances and runoff." DOI: 10.5194/hess-19-4673-2015. **[PDF]** Uses MB + runoff as joint constraint to back-correct precipitation forcing — a *forcing-side* multi-objective inversion rather than a parameter calibration.
- **Tahir, A.A., et al. (2011, 2016, 2019)** — multiple SRM + MODIS papers on Hunza/Gilgit; primarily Q-only calibration.
- **Shrestha, M., Koike, T., Hirabayashi, Y., Xue, Y., Wang, L., Rasul, G., Ahmad, B. (2015)** *JGR-Atmos.* 120, 4889–4919 — "Integrated simulation of snow and glacier melt in water and energy balance-based, distributed hydrological modeling framework at Hunza River Basin." DOI: 10.1002/2014JD022666 **[DOI not verified]**. Physically based energy-balance Hunza calibration.
- **Garee, K., Chen, X., Bao, A., Wang, Y., Meng, F. (2017)** *Water* 9, 17 — Hunza SWAT calibration. DOI: 10.3390/w9010017 **[DOI not verified]**.
- **Khan, A., Richards, K.S., Parker, G.T., et al. (2014)** *Adv. Water Resour.* 71, 47–57 — Shyok runoff component partitioning. DOI: 10.1016/j.advwatres.2014.05.014 **[DOI not verified]**.
- **Hayat, H., Akbar, T.A., Tahir, A.A., et al. (2019)** *Water* 11, 761 — "Simulating current and future river-flows in the Karakoram and Himalayan regions of Pakistan using snowmelt-runoff model and RCP scenarios." DOI: 10.3390/w11040761.
- **Pradhananga, D., Pomeroy, J.W. (2022)** *J. Hydrol.* 608, 127545 — CRHM-Glacier. Process-based alternative to SPHY in HKH; uses geodetic MB calibration.
- **Adhikari, A., Pradhananga, D., et al. (2024)** *Proc. IAHS* 387, 25–31 — Coupling GDM with CRHM in the Langtang basin (Nepal). DOI: 10.5194/piahs-387-25-2024.

### 4.3 Established practice for HKH calibration (synthesis)

Across this literature, the modal HKH calibration recipe is:

1. **Stepwise** — snow params calibrated first against MODIS SCA, then glacier melt against MB, then baseflow + routing against Q. (Lutz, Wijngaard, Khanal, Huang.)
2. **Multi-gauge** when intermediate gauges exist (Hunza, Astore, Shigar within UIB).
3. **MODIS SCA basin-mean or elevation-banded.** Per-band is becoming more common.
4. **Geodetic MB** (Hugonnet) increasingly standard since 2022; replaces the IceSat-based MB used in early SPHY work.
5. **Q-only on KGE or NSE.** Winter-baseflow as a separate target is *new* (Huang 2026) and Justine's adoption of it puts her ahead of the typical practice.

The major underexploited target in HKH is **isotopes** — almost no calibration studies use them, despite Avesani 2025 (Tibetan Plateau) showing they are the single strongest constraint on low-flow uncertainty. Roy et al. 2024 (W-Himalaya tracers) is a recent data-side advance.

---

## Synthesis & recommendations for Justine's setup

### What the literature actually says, as bullets

1. **Pareto > weighted-sum, mathematically.** Weighted-sum cannot reach non-convex Pareto fronts and hides trade-offs the user needs to see. The methodological literature unanimously favours Pareto/ε-Pareto when computationally affordable.
2. **NSGA-II is fine but not always best for expensive models.** For SPHY-in-Raven at HKH scale, PADDS (Asadzadeh & Tolson 2013), Borg (Hadka & Reed 2013), or caRamel (Monteil 2020) converge faster on hypervolume with the same number of model evaluations. Probably not worth switching mid-PhD, but worth knowing.
3. **Weighted-sum is defensible *if* (a) you scale-normalise the components before weighting, (b) you justify the weights, (c) you check that the chosen weights are not in the dominated region of the implicit Pareto front.** Min-max rescaling using a baseline NSGA-II run is the standard fix.
4. **KGE has known compensating-bias pathologies** (Cinkus 2023). For Q, consider KGE-NP (Pool 2018) or KGE+NSE(log Q) (Araya 2023). Plain KGE is not wrong, but if NSGA-II's Pareto solutions all sit at the "KGE looks good but bias and variability cancel" corner, you need to know.
5. **Per-elevation-band snow > basin-mean snow** for parameter identifiability (Tiwari 2024, Finger 2015). Justine has this already.
6. **Winter baseflow as a 4th-ish target is novel and well-justified.** Huang 2026 plus Avesani 2025 (isotopes) both show winter low-flow is the most under-constrained part of the hydrograph in HKH. Justine's choice is on the methodological frontier.
7. **Stepwise calibration risks compensating biases.** van Tiel 2020 review is unambiguous: simultaneous multi-criteria > stepwise when computational budget allows. NSGA-II naturally avoids the stepwise problem.
8. **Geodetic mass balance from Hugonnet is the single best auxiliary constraint Justine isn't yet using.** Konz & Seibert 2010, van Tiel 2020, Hanus 2024 all agree. Adding it as a 4th NSGA-II objective is the highest-value next step. SCA + MB + Q + winter-baseflow is the current state-of-the-art constraint set for glacierised catchments.
9. **GRACE TWS, ET, isotopes — secondary priorities given Justine's catchments.** GRACE too coarse for Swiss / individual UIB sub-basins; ET small term in cryospheric water balance; isotopes great in principle but no data for UIB.

### Specific recommendations to Justine's 3-objective SCEUA + NSGA-II

**Keep doing in parallel both algorithms** — the methodological literature explicitly recommends comparing weighted-sum and Pareto to surface whether they converge on the same parameter region. They usually don't, and the disagreement is informative.

**For SCEUA (weighted-sum):**
- Your switch from raw `1−RMSE` to `1−RMSE/μ` for snow is correct and well-motivated; it's a min-max-style normalisation that brings the metric into a comparable range to KGE.
- Consider running a "diagnostic NSGA-II" first to get an empirical estimate of the achievable range for each objective, then min-max rescale all three before weighted-summing in SCEUA. This is the Hanus 2024 / Efstratiadis-Koutsoyiannis recipe.
- The 0.4 / 0.3 / 0.3 weights are not obviously wrong but should be justified against either (a) data uncertainty (Westerberg & McMillan 2015) or (b) decision-maker preference. "Equal-ish" is a reasonable default in the absence of either.

**For NSGA-II (Pareto):**
- 200 generations × 50 population = 10 000 evaluations. Adequate for 3 objectives; would be marginal for 4–5. Borg or PADDS would be more efficient.
- Post-Pareto-selection: read Pechlivanidis 2014 on the cutoff-threshold / behavioural-set logic for picking a single "best" parameter set or ensemble from the Pareto front. Recommended: ε-thresholds (e.g. require KGE_Q > 0.7, snow-score > some min, baseflow-KGE > some min), then median over the surviving set.
- Visualise the Pareto front to verify it is well-distributed (hypervolume metric). If NSGA-II is producing knee solutions all in one corner you may have an implicit weighting from objective-magnitude differences — same problem as SCEUA.

**Next steps in priority order:**
1. **Add Hugonnet 2021 geodetic MB as a 4th objective** (per-glacier or basin-aggregated). Strongest literature support. Cite Konz & Seibert 2010, Hanus 2024, van Tiel 2020.
2. **Run a 50-generation NSGA-II first** purely to estimate achievable objective ranges, then use those for SCEUA min-max normalisation.
3. **Replace KGE on Q with KGE-NP or KGE(Q) + NSE(log Q)** — defends against the Cinkus 2023 compensating-bias issue, and Araya 2023 shows this combo wins in snow-dominated mountain catchments.
4. **For Swiss catchments only:** add in-situ SLF/OSHD SWE as a higher-quality replacement / complement to MODIS fSCA in the snow objective. Switzerland has the best SWE data in the world; ignore at peril.
5. **Look up Avesani 2025 isotope methodology** if/when you ever get δ¹⁸O for any catchment — it would single-handedly resolve a chunk of the baseflow equifinality.
6. **Don't add ET, GRACE, or surrogate ML** unless the project scope explicitly requires it; marginal value vs setup cost for your catchment sizes and objectives.

### Honest gaps in this review

- I did not exhaustively search the **flood-forecasting** multi-objective literature (event-based modelling, Krauße 2012 was the only entry). Probably not relevant to your seasonal-cycle / climate-projection use case.
- I did not find a single calibration study that uses **all** of Q + per-band snow + MB + winter-baseflow on a glacierised catchment. Justine's setup, if extended with MB, would be a publishable methodological contribution in its own right.
- The Khatami 2019 flux-mapping work is paywalled at the publisher; the Monash repository version should have an open copy — fetch via institutional access.

---

## PDFs downloaded to `/home/jberg/Raven-world/paper_ideas/multiobj_pdfs/`

| File | Why it's here |
|---|---|
| `vanTiel_2020_glaciohydro_calibration_review.pdf` | The canonical review of glacio-hydrological calibration practice — methodological anchor for the whole project. |
| `Konz_Seibert_2010_glacier_MB_value.pdf` | Foundational result that Q-only calibration in glacierised basins is structurally biased; motivates adding MB. |
| `Pellicciotti_2012_HKH_calibration_challenges.pdf` | The canonical "why HKH calibration is hard" paper; precipitation-melt compensation; required reading for the UIB phase. |
| `Khanal_2025_FW265_SPHY_HMA_guidelines.pdf` | Current developer-recommended SPHY calibration recipe for HMA — what Justine is deviating from and why. |
| `Huang_2025_EarthArXiv_Pamir_winter_baseflow_preprint.pdf` | EarthArXiv preprint of Huang 2026 WRR — direct methodological precedent for the winter-baseflow objective. |
| `Pool_2017_calibration_criteria_streamflow_characteristics.pdf` | "You only get what you optimise for" — empirical evidence that objective choice biases parameter values and SFCs. |
| `Cinkus_2023_KGE_critical_evaluation.pdf` | The KGE-compensating-bias paper; required reading before defending plain KGE as the Q objective. |
| `Araya_2023_calibration_metrics_mountain.pdf` | 22 Chilean mountain catchments × 12 objectives; concludes KGE(Q)+NSE(logQ) is the best mountain-Q combo. Directly transferable. |
| `Tiwari_2024_distributed_snow_calibration.pdf` | PADDS multi-objective with spatial vs averaged SWE — empirical evidence that per-elevation-band > basin-mean snow data. |
| `Tong_2021_ASCAT_MODIS_calibration.pdf` | 213 Austrian catchments multi-source calibration — auxiliary data improves internal state more than Q. |
| `Avesani_2025_isotope_snow_glacier_calibration.pdf` | Bayesian multi-likelihood (Q+SCA+GMB+isotopes) on Tibetan Plateau — isotopes most informative for low-flow uncertainty. |
| `Shannon_2023_Naryn_snow_glacier_GLUE.pdf` | GLUE multi-likelihood on a large HKH-style snow/glacier catchment; behavioural-ensemble alternative to Pareto. |
| `Stefnisdottir_2021_preprint_Pareto_HBV_glacier.pdf` | Direct precedent: HBV with Q+SCA+MB three-objective Pareto on a Rhone glacier catchment (GA wins). Preprint only. |
| `Hanus_2024_OGGM_CWatM.pdf` | NSGA-II with weighted KGE + snow penalty; large-scale OGGM-CWatM coupling. Example of weighted-sum used carefully. |
| `Monteil_2020_caRamel.pdf` | caRamel multi-objective algorithm — faster Pareto convergence than NSGA-II, drop-in replacement when budget tightens. |
| `Shafii_2009_WetSpa_NSGAII.pdf` | Foundational NSGA-II application in hydrology; HESS open-access worked example. |
| `EfstratiadisKoutsoyiannis_2012_robust_multiobj.pdf` | "Robust" multi-objective with depth-based sampling — framework for picking from the Pareto set. |
| `Farahani_2025_SUMMA_emulator.pdf` | ML emulator calibration across CAMELS — state-of-the-art for when you eventually scale to many catchments. |
| `Immerzeel_2015_HighAlt_precip_glacier_MB.pdf` | UIB precipitation back-correction using MB + Q jointly; forcing-side multi-objective inversion in HKH. |
