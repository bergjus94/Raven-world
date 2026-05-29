# Deep-read memo: Huang 2025/2026, Cinkus 2023, Araya 2023

**Date:** 2026-05-29
**Scope:** Three PDFs that were supposed to inform the next round of
multi-objective SPHY-in-Raven calibration (SCEUA weighted-sum + NSGAII
Pareto, on glacierized Swiss + UIB catchments, with objectives `Q`, `snow`,
`baseflow`).

> **PDF integrity issue (urgent):** the file
> `paper_ideas/multiobj_pdfs/Huang_2025_EarthArXiv_Pamir_winter_baseflow_preprint.pdf`
> is **not** the Huang Pamir baseflow paper. All 24 pages are
> "Can fiddler crab bioturbation activity *in situ* modify the distribution
> of microplastics in sediments and the influence on their bioaccumulation?"
> by Capparelli et al. (preprint for *Marine Pollution Bulletin*). The
> filename is wrong; the actual Huang paper is not in
> `paper_ideas/multiobj_pdfs/`. Re-download required before any deep-read of
> Huang is possible. Everything in Section 1 below is based on this fact —
> there is no Huang content I can responsibly summarise.

## TL;DR — concrete code changes proposed

1. **Drop `RMSE` from the snow objective on SCEUA runs; use `nRMSE` you
   already added.** Edit `namelists/catchment_2161_SPHY_Q_snow_baseflow.yaml`
   (and the other `*_SPHY_Q_snow*.yaml` SCEUA namelists): change
   `snow.metric: RMSE` → `snow.metric: nRMSE`. This is the scale fix you
   identified independently; Cinkus 2023 supports it indirectly (RMSE has no
   inherent scale to compose with relative metrics like KGE).
2. **Add `KGE(Q) + NSE(log Q)` as a single composite `Q` objective.** In
   `src/calibration_objectives.py::q_objective`, support
   `metric: 'KGE+NSElog'` (or a structured spec `metric: [KGE, NSElog]` with
   internal 0.5/0.5 averaging). This is Araya 2023's recommended best
   compromise across snow-dominated mountain catchments for retaining
   hydrological signature consistency without breaking flood-peak skill. Use
   it in the SCEUA namelists in place of the current `Q.metric: KGE`. **Keep
   it as one objective**, not as a fourth Pareto axis — see Section 3.
3. **Keep the current weighted-sum weights (0.4 / 0.3 / 0.3) as-is.** Neither
   Cinkus nor Araya gives a defensible reason to change them. Don't tune
   weights in response to literature you haven't seen (Huang).
4. **For NSGAII (Pareto): switch `Q` from plain `KGE` to `KGE_NP`
   (non-parametric KGE, Pool et al. 2018).** This is Cinkus 2023's single
   strongest concrete recommendation: of the KGE family, `KGE_NP` and `DE'`
   are the *least* affected by counterbalancing errors. Pareto algorithms
   are exactly where false high-scoring solutions hurt most because they
   can't be down-weighted by composition with another axis. This requires
   adding `_kge_np` to the `METRICS` dict in `calibration_objectives.py`
   (~15 lines; spearman rank correlation in the `r` slot, FDC-based
   `α_NP` in the `α` slot — Cinkus eqns 9 & 15).
5. **Log raw KGE *components* (α, β, r), not just the composite score, in
   the SCEUA sidecar.** The `raw_diagnostics` you already have logs
   r/rmse/mae/pbias; add α (σ_sim/σ_obs) and β (μ_sim/μ_obs) so you can
   *see* counterbalancing errors when they happen (a high KGE with
   α ≈ 1.3, β ≈ 0.7 is the diagnostic signature Cinkus warns about).
   ~5-line change in `raw_diagnostics()`.
6. **Do NOT add MB as a fourth objective.** GloGEM already provides this
   constraint externally. Restated explicitly in Section "What NOT to change"
   because no literature read here proposed it, but it's the kind of thing
   that gets bolted on by accident.

---

## 1. Huang 2025/2026 (Pamir SPHY winter baseflow)

### What they did (methodology)
**Unable to summarise.** The PDF at the expected path contains a marine
ecology paper on fiddler crabs and microplastics (Capparelli et al.,
*Marine Pollution Bulletin* preprint). There is no Huang Pamir baseflow
content in `paper_ideas/multiobj_pdfs/` at all.

### What they found
N/A — see above. Re-download the actual preprint before relying on this
section.

### How it differs from Justine's setup
N/A.

### Recommendations for her setup
**One** recommendation, which is procedural rather than scientific:

- Re-download Huang 2025 (or 2026 if the version has moved) from EarthArXiv
  with a quick MIME / first-page check on the file before saving. The
  current file passes a `.pdf` extension test but the content is unrelated.
  Until that's fixed, **the Pamir baseflow methodology that motivated the
  whole "baseflow as a third objective" design is not actually documented
  in any PDF you have on disk**. This is a real risk: the entire framing
  of the third objective is justified by Huang in your project memory, but
  you cannot at this moment cite a specific filter / window / metric choice
  from that paper.

In the meantime, the baseflow objective as currently implemented (Eckhardt
on both obs and sim, KGE on Nov–Mar) is methodologically defensible on its
own terms and consistent with the broader baseflow-evaluation literature
(Eckhardt 2005, Stoelzle et al. 2013, Gnann et al. 2021). It does not need
Huang as a justification to be sensible — but if Huang is going to be the
cited precedent in the paper, you need the paper.

---

## 2. Cinkus 2023 (KGE compensating-bias pathology)

### What they did
Two-part study, published in *HESS* 27, 2397–2411, 2023:

1. **Synthetic experiment.** Created a reference hydrograph (two flood
   events) and 361² = 130,321 transformations by scaling each event
   independently with coefficients ω₁, ω₂ ∈ [10⁻⁰·³⁶, 10⁰·³⁶], i.e.
   discharge between ~½× and ~2×. Classified each transformed series as
   "bad–bad" (both events distorted), "bad–good" (one perfect, one wrong),
   etc. Scored all transformations with **nine metrics**: NSE, KGE (2009),
   KGE' (2012, with γ = CV ratio instead of α = σ ratio), KGE'' (2021, with
   normalised bias β_n instead of β), KGE_NP (non-parametric, FDC-based α
   and Spearman r), DE' (Schwemmle 2021 diagnostic efficiency), LME (Liu
   2020), LCE (Lee & Choi 2022), and Willmott's modified d₁.
2. **Real-data check** on the Unica karst spring (Slovenia, ~820 km²), an
   ANN vs a KarstMod bucket-type model evaluated against the same nine
   metrics on a 1-year validation period.

### What they found
The headline result is precise and quantitative:

- **All KGE variants reward "bad–bad" (BB) transformations** — i.e. models
  with errors in both flood events — **over the BG (bad–good) reference**,
  as long as the BB errors are in *opposite* directions (one over, one
  under). The compensation drives β to ~1.0 and *only* α and r register
  the damage, which together carry one-third of the score.
- Quantified impact (their Table 1): impact on score, scale of `–` (none)
  to `+++` (strong):
  - NSE: – (no counterbalancing — it uses squared errors)
  - d₁: – (no counterbalancing — uses absolute errors)
  - KGE_NP: + (mild — non-parametric robust to outliers but still relative)
  - DE': + (mild — FDC-based, includes diagnostic terms)
  - KGE' (2012): + (mild — γ instead of α reduces α–β cross-correlation)
  - **KGE: ++** (moderate)
  - KGE'' (2021): ++
  - LCE: ++
  - LME: +++ (strong — infinite-solution problem)
- Real-data confirmation: at Unica, both models had similar overall
  hydrographs but the bucket-type model **overestimated several floods and
  underestimated several recessions** (classic counterbalancing). It
  scored higher than the ANN on KGE, KGE', KGE'', DE', LME, LCE
  despite the ANN being clearly the better hydrograph by visual
  inspection and by NSE/d₁/KGE_NP.
- The mechanism: in KGE, β and α together contribute 2/3 of the weight
  (each parameter is one term in a Euclidean distance). Both are *relative*
  ratios. Two opposing errors of the same magnitude push β back to 1.0
  exactly, and they push α only slightly. Only r is unambiguously degraded.

### Specific recommendations from the paper
Cinkus explicitly says:
1. **Use criteria that are not / less prone to counterbalancing**: d₁,
   KGE_NP, DE', or modified KGE'.
2. **Or — use scaling factors** in the KGE equation to re-weight the
   bias/variability/correlation terms. Their eqn 22:
   `KGE_s = 1 − √[ s_α(α−1)² + s_β(β−1)² + s_r(r−1)² ]`.
   The default 1-1-1 weights make β and α the dominant terms; a 1-1-4 or
   1-1-3 ratio (more weight on r) markedly reduces counterbalancing in
   their case study (Fig. 8).

### Whether it matters for glacierized alpine streamflow
**Partially — but less than for karst.** Two pieces of context matter:

1. **The Unica karst system is a worst case.** Karst has wildly flashy
   floods + long stable baseflows + slow recessions, all of which create
   independent error modes that are easy to cross-cancel. Glacierized
   alpine Q has a far more dominant signal (the summer melt peak), which
   anchors α and β strongly. You won't get the full pathology because the
   ~July–August peak controls 50–70% of annual volume and ~80% of variance.
2. **You'll get a milder version anyway.** Anywhere your snowmelt timing
   is off by a few weeks (which Raven+SPHY routinely produces because of
   lapse-rate uncertainty), you can simultaneously *over*-predict July
   while *under*-predicting October recession — that's textbook
   counterbalancing on the seasonal scale rather than the event scale.
   This will show up as: high KGE-on-Q + visibly wrong seasonal timing in
   the hydrograph.

### Recommendations for her setup

**Do this (concrete, ordered by ease):**

1. **Log α and β components** in the SCEUA diagnostic sidecar so you can
   *detect* counterbalancing. Currently you log `r, rmse, mae, pbias` in
   `raw_diagnostics`. Add:
   ```python
   'alpha': float(sim.std() / obs.std()),
   'beta':  float(sim.mean() / obs.mean()),
   ```
   When the best member of an SCEUA population has KGE > 0.85 with α
   outside [0.85, 1.15] or β outside [0.90, 1.10], you have a
   counterbalancing problem. This is a diagnostic, not a calibration
   change — and it costs ~5 lines.
2. **For NSGAII (Pareto), swap KGE for KGE_NP on the Q axis.** The Pareto
   front in particular needs an objective that doesn't accept fake-good
   solutions, because there's no second axis pulling Q back. Add an
   implementation:
   ```python
   def _kge_np(obs, sim):
       # Eckhardt-style non-parametric KGE (Pool 2018, eqn for KGE_NP)
       r_s = float(spearmanr(obs, sim).correlation)
       beta = float(sim.mean() / obs.mean())
       # FDC-based variability term (Pool eqn 9 in Cinkus):
       fdc_o = np.sort(obs)[::-1] / (len(obs) * obs.mean())
       fdc_s = np.sort(sim)[::-1] / (len(sim) * sim.mean())
       alpha_np = 1.0 - 0.5 * np.sum(np.abs(fdc_s - fdc_o))
       return 1.0 - np.sqrt((alpha_np - 1)**2 + (beta - 1)**2 + (r_s - 1)**2)
   ```
3. **Do NOT switch SCEUA away from KGE on Q.** SCEUA's weighted-sum
   composition with snow + baseflow already mitigates the worst of
   counterbalancing, because if you over-predict July and under-predict
   October, your snow objective (off-glacier fSCA) will *also* register
   the error — snow melting too fast in spring, snow building up too slow
   in autumn. The cross-term safety is exactly the value-add of the
   multi-objective setup.

**Don't do this:**
- Don't switch to LME — Cinkus rates it +++ (worst). Don't switch to d₁ as
  the primary metric — it's safer than KGE, but it doesn't reflect
  variability, which you care about in a melt-dominated regime.
- Don't tweak the KGE scaling factors (s_α, s_β, s_r). It's a real lever
  but it requires picking weights, which trades one tuning knob for
  another. Not worth it given that you have the cleaner KGE_NP option.

---

## 3. Araya & Mendoza 2023 (calibration metrics for mountain catchments)

### What they did
*HESS* 27, 4385–4408, 2023. Highly relevant — this is almost a direct
analog for your setup, modulo the glacier component.

- **22 mountain catchments** in central Chile (28–37° S), spanning a
  hydroclimatic gradient from semi-arid (PET/P > 1) to wet temperate. Snow
  controls runoff seasonality in all of them; **no glacier component**
  (Andean catchments at these latitudes are largely glacier-free or have
  small influence; this is the main caveat for transferring to your
  setup).
- **Three hydrological models**: GR4J + CemaNeige, TUW (HBV-like), and
  Sacramento-SMA + SNOW-17. Bucket-type conceptual, parameter counts 6 /
  15 / 28.
- **12 calibration objective functions** in five families (their Table 1):
  1. NSE
  2. KGE family: KGE, KGE', ModKGE (Mizukami 2019), KGE''
  3. Split-KGE (Fowler 2018a) — KGE per year, then averaged
  4. Meta-objectives with transforms: KGE(Q) + KGE(1/Q), and
     **KGE(Q) + NSE(log Q)**
  5. Seasonal: VE-Sep (Sep–Mar volume RMSE), VE-Oct, KGEV-Sep, KGEV-Oct
- **Single-objective** SCE-UA calibration on each of the 12 OFs, separately
  for each (catchment × model). 19-year calibration (Apr 1994–Mar 2013),
  14-year evaluation.
- They scored hindcasts (ESP method, 5 initialization times) and also
  evaluated *hydrological consistency* via 5 FDC-based signatures (RR,
  FHV, FLV, FMS, FMM).

### What they found
The paper's main results, in the order that matters for your decision:

1. **Five "best representative" OFs identified** for further analysis:
   NSE, ModKGE, Split-KGE, VE-Sep, and **KGE(Q) + NSE(log Q)**.
2. **For *seasonal hindcast skill* (CRPSS)**, seasonal OFs (VE-Sep
   especially) win — but with a brutal trade-off: VE-Sep yields
   *unacceptable* hydrological consistency (median daily KGE = −0.27 to
   0.40 at Maipo with three models; their Fig 4 a3) and gross signature
   biases (44% bias in FLV, 27% in FMM during evaluation period).
3. **For *hydrological consistency***, KGE(Q) + NSE(log Q) is the winner
   during the calibration period; Split-KGE is best during evaluation.
4. **The Araya-recommended overall compromise** (their words, paraphrased
   from the Conclusions): **KGE(Q) + NSE(log Q) is the best balance** —
   only ~5% loss in CRPSS vs the best seasonal OF, while keeping good
   hydrological consistency (low FLV / FMM biases, reasonable daily KGE).
5. **Catchment-attribute correlations**: seasonal forecast skill correlates
   most strongly with **baseflow index** (ρ ≈ 0.2–0.8 across models) and
   **inter-annual runoff variability**. Their Fig 10 shows BFI is by
   far the strongest catchment predictor of which calibration metric
   wins, which is highly relevant for you — Massa, Hunza et al. have
   *different* BFI than central Chilean basins.
6. **Split-KGE evaluation result is interesting and worth flagging**:
   Split-KGE has the *highest* mean annual runoff biases during
   calibration (8.6%) but the *lowest* biases during evaluation (11.8%).
   That's a sign of better temporal transferability — which matters for
   your future-scenario work.

### Specific recommendations from the paper
The Araya conclusion is unusually clean: *"We could identify at least one
objective function (KGE(Q) + NSE(log(Q))) that yields a reasonable balance
between hydrological consistency and hindcast performance."*

This is a **linear combination of two metrics**, not a Pareto. They
average them with equal weight (they don't say explicitly, but it's
implied by the "meta-objective" framing and the way they sum the two
components into one OF in SCE-UA).

### Generalization to glacierized catchments
**Partial.** Three relevant differences for your setup:

1. **No glaciers in their study domain.** Andean catchments at 28–37° S
   are largely glacier-free or near-zero. Yours have 10–80% glacier cover.
   In glacierized catchments the summer Q peak is dominated by melt
   rather than precipitation, which changes the relative importance of:
   - the high-Q tail (where KGE on raw Q is most informative) — *more*
     important for you
   - the low-Q recession (where NSE on log Q is most informative) — *also*
     more important for you, because winter baseflow is the diagnostic
     signal everyone is interested in.
   This means the KGE + NSE(log Q) trade-off is *more* useful for you, not
   less.
2. **Their baseflow signature (FLV) IS evaluated** but only as a *check
   after calibration*, not as a calibration target. Your design (baseflow
   as a third calibration target) is actually more ambitious than theirs.
3. **They used SCE-UA single-objective only**, no Pareto. So their
   recommendation transfers cleanly to your SCEUA runs but not directly
   to NSGAII.

### Recommendations for her setup

**Do this:**

1. **Add `KGE(Q) + NSE(log Q)` as a Q-metric option** in
   `src/calibration_objectives.py`. Implementation should be one
   composite metric, not a fourth Pareto axis:
   ```python
   def _kge_nse_log(obs, sim):
       k = _kge(obs, sim)
       n = _nse(np.log(np.where(obs > 0, obs, np.nan)),
                np.log(np.where(sim > 0, sim, np.nan)))
       return 0.5 * (k + n)
   METRICS['KGE+NSElog'] = _kge_nse_log
   ```
   And add a sanity-check: it should refuse to score if more than ~5% of
   the Q series is zero (log of zero is undefined). For Massa / UIB
   catchments daily Q is essentially always > 0 so this won't trigger,
   but for ephemeral semi-arid catchments it would.
2. **Use it in the SCEUA namelists.** Change `Q.metric: KGE` →
   `Q.metric: KGE+NSElog` in `catchment_2161_SPHY_Q_snow_baseflow.yaml`
   and the other SCEUA configs. Keep weights 0.4/0.3/0.3.
3. **Keep `Q` as a single objective in NSGAII Pareto too.** Don't split.
   Araya didn't run Pareto, and a 4-axis Pareto with both KGE and
   NSE(log Q) as separate axes will dilute the front. Justine has
   established 3-axis Pareto is already at the edge of interpretability
   with 50 generations of NSGA-II on ~10-12 params.

**Don't do this:**

- Don't use VE-Sep or Split-KGE as your Q metric. VE-Sep destroys daily
  consistency (their result, very clear). Split-KGE looks nice for
  transferability but it's annual-resolution; you have a daily-resolution
  story.
- Don't add the FLV / FMS / FMM signature biases as Pareto axes. They're
  evaluation diagnostics, not calibration targets, in Araya's design. Add
  them to your **post-calibration diagnostic sidecar** instead — these
  are essentially what your baseflow target already covers (FLV ≈ winter
  baseflow volume).

---

## Synthesis: what to actually change in the next calibration run

### Files affected
- `src/calibration_objectives.py`
- `namelists/catchment_*_SPHY_Q_snow_baseflow.yaml` (SCEUA configs)
- `namelists/catchment_*_SPHY_Q_snow_baseflow_pareto.yaml` (NSGAII configs)

### Specific edits

**1. `src/calibration_objectives.py` — add 2 metrics + extend
diagnostics (~30 lines total).**

```python
# In the natural-form section:
def _spearman_r(obs, sim):
    from scipy.stats import spearmanr
    return float(spearmanr(obs, sim).correlation)

def _kge_np(obs, sim):
    """Non-parametric KGE (Pool et al. 2018). Cinkus 2023 rates this
    as least affected by counterbalancing errors of the KGE family."""
    r_s = _spearman_r(obs, sim)
    beta = float(sim.mean() / obs.mean())
    # FDC-based variability (Cinkus eqn 9 of α_NP):
    n = len(obs)
    fdc_o = np.sort(obs)[::-1] / (n * obs.mean())
    fdc_s = np.sort(sim)[::-1] / (n * sim.mean())
    alpha_np = 1.0 - 0.5 * np.sum(np.abs(fdc_s - fdc_o))
    return 1.0 - np.sqrt((alpha_np - 1)**2 + (beta - 1)**2 + (r_s - 1)**2)

def _kge_nse_log(obs, sim):
    """Araya & Mendoza 2023 best-compromise: KGE(Q) + NSE(log Q),
    equal-weighted."""
    k = _kge(obs, sim)
    obs_l = np.log(np.where(obs > 0, obs, np.nan))
    sim_l = np.log(np.where(sim > 0, sim, np.nan))
    mask = np.isfinite(obs_l) & np.isfinite(sim_l)
    if mask.sum() < 30 or mask.sum() < 0.95 * len(obs):
        # > 5% zeros / NaNs → refuse to score
        return float('nan')
    n = float(sof.nashsutcliffe(obs_l[mask], sim_l[mask]))
    return 0.5 * (k + n)

# Add to METRICS registry:
METRICS['KGE_NP']     = _kge_np
METRICS['KGE+NSElog'] = _kge_nse_log

# Extend raw_diagnostics() to log Gupta-decomposition α and β:
def raw_diagnostics(obs, sim):
    ...
    return {
        'r':     _r_pearson(o, s),
        'rmse':  _rmse_raw(o, s),
        'mae':   _mae_raw(o, s),
        'pbias': _pbias_raw(o, s),
        'alpha': float(np.std(s) / np.std(o)) if np.std(o) > 0 else float('nan'),
        'beta':  float(np.mean(s) / np.mean(o)) if np.mean(o) != 0 else float('nan'),
        'n':     int(len(df)),
    }
```

**2. SCEUA namelists (e.g. `catchment_2161_SPHY_Q_snow_baseflow.yaml`):**

```yaml
# BEFORE
Q:
  metric: KGE
snow:
  metric: RMSE
  aggregation: elevation_band
  ...
baseflow:
  metric: KGE
  method: eckhardt
  window: winter

# AFTER
Q:
  metric: KGE+NSElog      # Araya 2023: best hydro-consistency + skill compromise
snow:
  metric: nRMSE           # use scale-matched form Justine added; was RMSE
  aggregation: elevation_band
  ...
baseflow:
  metric: KGE             # keep — Cinkus pathology mitigated by winter window
  method: eckhardt        #   (low Q, low variance → less room for counterbalancing)
  window: winter
```

**3. NSGAII (Pareto) namelists (e.g. the `_pareto.yaml` sibling):**

```yaml
# BEFORE
Q:
  metric: KGE
snow:
  metric: RMSE
  ...

# AFTER
Q:
  metric: KGE_NP          # Cinkus 2023: KGE_NP least subject to counterbalancing
snow:
  metric: nRMSE
  ...
baseflow:
  metric: KGE_NP          # consistency with Q axis; KGE family stays in the front
  method: eckhardt
  window: winter
```

Rationale for split SCEUA-vs-NSGAII metric choice: in a weighted-sum the
3 axes can mutually constrain each other (snow + baseflow catch what KGE
on Q misses), so the KGE+NSElog composite is the right Q metric. In Pareto
each axis lives independently, so each axis needs to be robust on its own
— hence KGE_NP.

### Expected impact
- Snow objective scale now matches Q scale (nRMSE instead of RMSE) — fixes
  the over-weighting Justine independently identified.
- Q objective rewards both flow timing (KGE component) and recession /
  low-flow consistency (NSE-log component) — directly addresses what
  Araya 2023 identified as the calibration weakness of plain KGE in
  snow-dominated mountain basins.
- Pareto NSGAII is more robust to fake-good solutions on the Q axis (the
  axis with the most variance and most parameter sensitivity).
- Diagnostic sidecar now lets you *detect* counterbalancing post-hoc by
  inspecting α and β for any "best" SCEUA member.

### What this does NOT change
- The weighted-sum weights stay 0.4 / 0.3 / 0.3.
- The Eckhardt filter stays as the baseflow separator (no literature
  read here gives a reason to change it).
- The Nov–Mar winter window stays.
- The MODIS fSCA per-band area-weighted aggregation stays.
- The number of objectives stays at 3 (do not split Q into two Pareto
  axes).

---

## What NOT to change

These are things the literature *might* suggest at first glance but that
either don't apply to your setup or actively harm it.

1. **DO NOT add mass balance as a fourth objective.** This is the most
   important "do not". GloGEM provides the glacier melt and is itself
   calibrated against transient snow-line + geodetic mass balance
   externally. Adding MB at the Raven-SPHY-coupling stage either:
   (a) duplicates the constraint already in the system → no information
   gain, or
   (b) creates a fight between two MB sources (your GloGEM-derived ice
   melt forcing vs whatever observational MB target you'd use) →
   degrades both Q and snow performance. Neither Araya 2023, Cinkus
   2023, nor the absent-Huang paper recommends MB as a calibration
   target for this kind of coupled setup. Don't add it.
2. **DO NOT switch the Q metric to LME or LCE.** Cinkus rates LME as
   +++ (worst) for counterbalancing, and LCE as ++. Both look attractive
   because they emphasize extremes (which seems right for melt-driven
   summer peaks) but the counterbalancing pathology fully dominates.
3. **DO NOT add VE-Sep or Split-KGE as a calibration target.** Araya
   explicitly shows VE-Sep destroys daily hydrological consistency
   (median KGE on daily Q drops to −0.27 at Maipo). Split-KGE has nice
   inter-annual stability but is too low-resolution to inform Raven
   parameters that operate at daily / sub-daily scale.
4. **DO NOT split Q into a 2-axis Pareto (KGE + NSElog).** Araya's
   recommendation is a *composite* OF, not a Pareto. Splitting it would
   make Pareto a 4-axis problem; with NSGA-II on ~10–12 parameters, 4
   axes is past the regime where the front converges in <200
   generations.
5. **DO NOT use Cinkus's "scaling factors" recommendation
   (s_α, s_β, s_r in the KGE equation).** It's a real lever but it
   trades one tuning knob for another, and KGE_NP achieves the same
   counterbalancing reduction without an arbitrary weight choice.
6. **DO NOT replace the Eckhardt filter** based on any of these three
   papers — none of them evaluates baseflow separator choices, and the
   actual Huang paper (which presumably does) is not on disk.
