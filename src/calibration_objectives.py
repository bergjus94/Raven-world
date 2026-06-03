"""
Calibration objective functions for Raven multi-objective calibration.

Each objective takes simulated and observed timeseries (or paths to them) and
returns a single scalar metric value where **higher is better** (e.g. KGE).
The caller is responsible for combining them — a weighted sum for single-
objective algorithms (SCEUA / DDS / DREAM) or a list for Pareto algorithms
(NSGAII / PA-DDS).

Three objectives are supported:
  - q_objective:        discharge — KGE/NSE on daily Q over the cal period
  - snow_objective:     basin-mean off-glacier fSCA — Raven SNOW_FRAC BY_HRU
                        vs MODIS basin-mean CSV (8-day, cloud-masked)
  - baseflow_objective: winter baseflow — Eckhardt (or other) separator on
                        both obs and sim Q, KGE on the winter portion

The "winter window" for baseflow defaults to Nov-Mar, matching the existing
baseflow_separation.py convention (commit e8cbb25).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
import spotpy.objectivefunctions as sof

from baseflow_separation import BaseflowSeparator  # type: ignore


# Months treated as "winter" for baseflow / low-flow objectives.
# Matches WINTER_MONTHS in baseflow_separation.py (Nov-Mar in NH).
WINTER_MONTHS = (11, 12, 1, 2, 3)


# ---------------------------------------------------------------------------
# Metric registry — every entry returns a HIGHER-IS-BETTER score where 1.0 is
# perfect.  The "natural" form of error metrics (RMSE, MAE, PBIAS) is
# converted to a score so all metrics compose cleanly in the weighted-sum
# multi-objective framework.  Raw forms are also exposed via `raw_metrics`
# for the diagnostic sidecar.
# ---------------------------------------------------------------------------

# ---- Natural forms (raw values, range varies by metric) -------------------

def _kge(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(sof.kge(obs, sim))


def _kge_np(obs: np.ndarray, sim: np.ndarray) -> float:
    """Non-parametric KGE (Pool, Vis, Knoben & Seibert 2018).

    Replaces:
      - Pearson r → Spearman rank correlation (robust to outliers/non-Gaussian)
      - α = σ_sim/σ_obs → α_NP = 1 − 0.5·Σ|sorted_sim_norm − sorted_obs_norm|
        (flow-duration-curve based; directly compares the shape of the
        distribution rather than just its standard deviation)
    β (mean ratio) stays the same.

    Recommended by Cinkus et al. 2023 as the KGE variant least susceptible
    to compensating bias-and-variability errors. Useful for catchments
    where flow distributions are heavy-tailed (glacier-snowmelt regimes).
    """
    from scipy.stats import spearmanr
    if obs.std() == 0 or sim.std() == 0:
        return float('nan')
    # Spearman rank correlation
    r_s, _ = spearmanr(obs, sim)
    # Bias ratio (same as KGE)
    obs_mean = obs.mean()
    if obs_mean == 0:
        return float('nan')
    beta = sim.mean() / obs_mean
    # FDC-based alpha (descending sort, normalize by sum)
    obs_sorted = np.sort(obs)[::-1]
    sim_sorted = np.sort(sim)[::-1]
    obs_sum = obs_sorted.sum()
    sim_sum = sim_sorted.sum()
    if obs_sum == 0 or sim_sum == 0:
        return float('nan')
    obs_fdc = obs_sorted / obs_sum
    sim_fdc = sim_sorted / sim_sum
    alpha_np = 1.0 - 0.5 * float(np.sum(np.abs(sim_fdc - obs_fdc)))
    return float(1.0 - np.sqrt((r_s - 1.0) ** 2 + (alpha_np - 1.0) ** 2 + (beta - 1.0) ** 2))


def _nse(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(sof.nashsutcliffe(obs, sim))


def _logkge(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.where(obs > 0, obs, np.nan)
    sim = np.where(sim > 0, sim, np.nan)
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 30:
        return float('nan')
    return float(sof.kge(np.log(obs[mask]), np.log(sim[mask])))


def _rmse_raw(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(np.sqrt(np.mean((sim - obs) ** 2)))


def _nrmse_raw(obs: np.ndarray, sim: np.ndarray) -> float:
    """Mean-normalized RMSE: RMSE / mean(obs).  Returns NaN if mean(obs) is
    zero or non-finite (no signal to normalize against)."""
    mean_obs = obs.mean()
    if mean_obs == 0 or not np.isfinite(mean_obs):
        return float('nan')
    return float(_rmse_raw(obs, sim) / mean_obs)


def _mae_raw(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(np.mean(np.abs(sim - obs)))


def _pbias_raw(obs: np.ndarray, sim: np.ndarray) -> float:
    """Percent bias (Moriasi 2007).  0 = perfect, negative = under-prediction."""
    mean_obs = obs.mean()
    if mean_obs == 0:
        return float('nan')
    return float(100.0 * (sim - obs).sum() / obs.sum())


def _r_pearson(obs: np.ndarray, sim: np.ndarray) -> float:
    if obs.std() == 0 or sim.std() == 0:
        return float('nan')
    return float(np.corrcoef(obs, sim)[0, 1])


def _csi(obs: np.ndarray, sim: np.ndarray, threshold: float = 0.5) -> float:
    """Critical Success Index for snow/no-snow binary skill.

    fSCA > threshold treated as 'snow present'.  CSI = hits / (hits + misses +
    false alarms).  Range [0, 1], 1 = perfect agreement.  Designed for
    bounded categorical-like targets; degenerate for streamflow.
    """
    obs_b = (obs > threshold)
    sim_b = (sim > threshold)
    hits   = int(np.sum(obs_b & sim_b))
    misses = int(np.sum(obs_b & ~sim_b))
    fa     = int(np.sum(~obs_b & sim_b))
    denom  = hits + misses + fa
    return float(hits / denom) if denom > 0 else float('nan')


# ---- Score forms (higher-is-better, target 1.0) --------------------------
#
# For bounded variables (fSCA in [0, 1]), 1 - RMSE / 1 - MAE are in [0, 1]
# and behave as natural skill scores.  For unbounded variables the score
# form can dip negative, which still works inside _combine_weighted (NaN
# wouldn't, but a negative number is fine and just down-weights).

def _rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    return 1.0 - _rmse_raw(obs, sim)


def _nrmse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Score form of mean-normalized RMSE: 1 - RMSE/mean(obs).

    KGE-shaped scale so this can be combined with KGE-on-Q in a weighted-sum
    objective without the structural over-weighting that plain (1-RMSE) gives
    on bounded variables like fSCA.  Good fits on fSCA bands typically score
    0.5-0.8, mediocre ~0, bad goes negative (consistent with KGE behaviour).
    """
    return 1.0 - _nrmse_raw(obs, sim)


def _mae(obs: np.ndarray, sim: np.ndarray) -> float:
    return 1.0 - _mae_raw(obs, sim)


def _pbias_score(obs: np.ndarray, sim: np.ndarray) -> float:
    """Convert PBIAS (in %) to a [0, 1] skill score: 0% bias → 1.0, ±100%
    bias → 0.0, clamped at 0 for worse.
    """
    pb = abs(_pbias_raw(obs, sim))
    if not np.isfinite(pb):
        return float('nan')
    return float(max(0.0, 1.0 - min(1.0, pb / 100.0)))


METRICS = {
    'KGE':    _kge,
    'KGE_NP': _kge_np,       # non-parametric KGE (Pool 2018; Cinkus 2023 recommended)
    'NSE':    _nse,
    'LogKGE': _logkge,
    'RMSE':   _rmse,         # score form: 1 - RMSE (assumes bounded variable)
    'nRMSE':  _nrmse,        # score form: 1 - RMSE/mean(obs); KGE-shaped scale
    'MAE':    _mae,          # score form: 1 - MAE
    'PBIAS':  _pbias_score,  # score form: 1 - |PBIAS|/100
    'CSI':    _csi,
}


def _apply_metric(metric_name: str, obs: pd.Series, sim: pd.Series) -> float:
    df = pd.concat([obs.rename('obs'), sim.rename('sim')], axis=1).dropna()
    if len(df) < 30:
        return float('nan')
    if metric_name not in METRICS:
        raise ValueError(
            f"Unknown metric '{metric_name}'. Available: {sorted(METRICS)}"
        )
    return METRICS[metric_name](df['obs'].values, df['sim'].values)


def raw_diagnostics(obs: pd.Series, sim: pd.Series) -> dict:
    """Compute *raw* diagnostic metrics (natural units, not scored).

    Returned dict has keys: r, rmse, mae, pbias, n.  Intended for the
    sidecar CSV that logs alongside whichever metric is the calibration
    target — surfaces the underlying error structure regardless of which
    score the SCEUA loop is optimising.
    """
    df = pd.concat([obs.rename('obs'), sim.rename('sim')], axis=1).dropna()
    if len(df) < 30:
        return {'r': float('nan'), 'rmse': float('nan'),
                'mae': float('nan'), 'pbias': float('nan'), 'n': int(len(df))}
    o = df['obs'].values
    s = df['sim'].values
    return {
        'r':     _r_pearson(o, s),
        'rmse':  _rmse_raw(o, s),
        'mae':   _mae_raw(o, s),
        'pbias': _pbias_raw(o, s),
        'n':     int(len(df)),
    }


# ---------------------------------------------------------------------------
# Discharge objective
# ---------------------------------------------------------------------------

def q_objective(
    obs_Q: pd.Series,
    sim_Q: pd.Series,
    metric: str = 'KGE',
    date_range: Optional[tuple] = None,
) -> float:
    """KGE/NSE between observed and simulated daily discharge.

    Parameters
    ----------
    obs_Q, sim_Q : pd.Series indexed by date.
    metric       : 'KGE' | 'NSE' | 'LogKGE'.
    date_range   : optional (start, end) inclusive.
    """
    obs, sim = obs_Q, sim_Q
    if date_range is not None:
        obs = obs.loc[date_range[0]:date_range[1]]
        sim = sim.loc[date_range[0]:date_range[1]]
    return _apply_metric(metric, obs, sim)


# ---------------------------------------------------------------------------
# Snow objective
# ---------------------------------------------------------------------------

def load_modis_fsca(
    csv_path: Union[str, Path],
    cloud_threshold: float = 0.5,
) -> pd.Series:
    """Load MODIS basin-mean fSCA from a long-format CSV produced by
    ``scripts/derive_basin_fsca.py`` (or the legacy single-band ``fsca`` CSV).

    Rows where ``n_cloud / n_total > cloud_threshold`` are masked NaN.

    If the CSV is the multi-band form (band_m has multiple values per date),
    this collapses to a single area-weighted basin mean across bands using
    n_total as the weight. For per-band access use ``load_modis_fsca_bands``.
    """
    df = pd.read_csv(csv_path, parse_dates=['date'])
    if 'band_m' in df.columns and df['band_m'].nunique() > 1:
        # Multi-band CSV — collapse to basin mean weighted by n_total
        df['n_cloud_frac'] = df['n_cloud'] / df['n_total'].replace(0, np.nan)
        df['fsca'] = df['fsca'].where(df['n_cloud_frac'] <= cloud_threshold,
                                       other=np.nan)
        # Per date: sum(fsca * n_total) / sum(n_total) over valid bands
        df = df.dropna(subset=['fsca'])
        per_date = df.groupby('date').apply(
            lambda g: float(np.average(g['fsca'], weights=g['n_total']))
            if g['n_total'].sum() > 0 else np.nan,
            include_groups=False,
        )
        per_date.name = 'fsca_obs'
        return per_date

    # Single-band (or legacy) CSV — apply cloud threshold per row
    cloud_frac = df['n_cloud'] / df['n_total'].replace(0, np.nan)
    fsca = df['fsca'].astype(float)
    fsca = fsca.where(cloud_frac <= cloud_threshold, other=np.nan)
    return pd.Series(fsca.values, index=df['date'], name='fsca_obs')


def load_modis_fsca_bands(
    csv_path: Union[str, Path],
    cloud_threshold: float = 0.5,
    min_pixels_per_band: int = 30,
) -> pd.DataFrame:
    """Load per-elevation-band MODIS fSCA as a DataFrame.

    Returns a DataFrame indexed by date, columns are elevation-band lower
    edges in metres (int). Cells where ``n_cloud / n_total > cloud_threshold``
    or ``n_valid < min_pixels_per_band`` are NaN.

    Raises ValueError if the CSV is a basin-mean (single-band) file.
    """
    df = pd.read_csv(csv_path, parse_dates=['date'])
    if 'band_m' not in df.columns:
        raise ValueError(f"{csv_path} has no `band_m` column — it's a "
                         f"basin-mean CSV, not per-band.")
    if df['band_m'].nunique() <= 1:
        raise ValueError(f"{csv_path} has only one band ({df['band_m'].unique()}); "
                         f"use load_modis_fsca() for basin-mean.")

    cloud_frac = df['n_cloud'] / df['n_total'].replace(0, np.nan)
    df['fsca'] = df['fsca'].where(cloud_frac <= cloud_threshold, other=np.nan)
    df['fsca'] = df['fsca'].where(df['n_valid'] >= min_pixels_per_band,
                                   other=np.nan)
    df['band_m'] = df['band_m'].astype(int)

    # Defend against year-boundary duplicates that would otherwise produce
    # a cryptic "Index contains duplicate entries, cannot reshape" from
    # df.pivot. preprocess_modis_fsca.derive_for_catchment already dedupes
    # at write time; this is a backstop for stale CSVs from old runs.
    if df.duplicated(subset=['date', 'band_m']).any():
        dup_n = int(df.duplicated(subset=['date', 'band_m']).sum())
        df = df.drop_duplicates(subset=['date', 'band_m'], keep='first')
        # Loud warning so the user knows to re-derive when convenient.
        import warnings
        warnings.warn(
            f"{csv_path}: {dup_n} duplicate (date, band_m) rows dropped "
            f"during load (kept first). Re-run derive_basin_fsca.py to "
            f"refresh the CSV.", stacklevel=2,
        )

    return df.pivot(index='date', columns='band_m', values='fsca')


def _load_raven_by_hru_csv(src: Path) -> pd.DataFrame:
    """Parse a Raven ``:CustomOutput DAILY AVERAGE <X> BY_HRU`` CSV.

    Three formats observed in the wild:

    A. Title-row prefix:
           :CustomOutput DAILY AVERAGE SNOW_FRAC BY_HRU [-]
           time,date,hour,1,2,3,...

    B. Two-row header (current SPHY/HBV output):
           ,HRU:,1,2,3,...,N,
           time,day,mean,mean,...,mean,
           1,1990-01-01,0.5,...

    C. Plain CSV (test fixtures):
           time,date,hour,1,2,3,...

    Returns a DataFrame indexed by 'date' (parsed datetime) with one
    column per HRU (integer-named).
    """
    with open(src) as f:
        line0 = f.readline().rstrip('\n')
        line1 = f.readline().rstrip('\n')

    # Format B: line 0 has 'HRU:' as its second token. Line 1 is the
    # aggregation labels (all 'mean'). HRU IDs live in line 0.
    if 'HRU:' in line0 and ('mean' in line1 or 'day' in line1):
        raw_cols = line0.split(',')
        if raw_cols and raw_cols[-1] == '':
            raw_cols = raw_cols[:-1]
        df = pd.read_csv(src, skiprows=2, header=None)
        if df.shape[1] == len(raw_cols) + 1 and df.iloc[:, -1].isna().all():
            df = df.iloc[:, :-1]
        df.columns = raw_cols
        # Position 0 = time index, position 1 = date — name them.
        df.rename(columns={raw_cols[0]: 'time', raw_cols[1]: 'date'},
                  inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')

    # Format A: line 0 is a Raven directive. Skip it.
    if line0.lstrip().startswith(':'):
        df = pd.read_csv(src, skiprows=1)
        if 'day' in df.columns and 'date' not in df.columns:
            df = df.rename(columns={'day': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')

    # Format C: plain header on line 0.
    df = pd.read_csv(src)
    if 'day' in df.columns and 'date' not in df.columns:
        df = df.rename(columns={'day': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')


def swe_to_fsca(swe_mm: Union[float, np.ndarray, pd.Series],
                d_scale: float = 50.0) -> Union[float, np.ndarray, pd.Series]:
    """Convert snow water equivalent (mm) to fractional snow cover area [0, 1].

    Linear ramp: fSCA = min(SWE / d_scale, 1.0).

    Why this is needed: Raven's `SNOW_FRAC` variable is the snowfall-fraction
    *forcing* (controlled by RainSnow_Temp), and `SNOW_COVER` (the proper fSCA
    state) is only updated by algorithms like SNOBAL_CEMA_NEIGE — our
    SNOBAL_SIMPLE_MELT leaves it at zero. So we output the SWE state and
    derive fSCA here.

    The default d_scale=50 mm matches typical alpine depletion thresholds in
    operational snow models (Liston 2004 family). A smaller value makes the
    objective more sensitive to melt timing; a larger value smooths the
    transition. d_scale is a structural parameter of the metric, not a
    calibrated model parameter.

    Parameters
    ----------
    swe_mm   : SWE in millimetres (any array-like).
    d_scale  : depth (mm) above which fSCA saturates at 1.

    Returns
    -------
    fSCA in [0, 1], same shape as input.
    """
    return np.minimum(swe_mm / float(d_scale), 1.0)


def load_raven_snow_frac(
    output_dir: Union[str, Path],
    hru_areas: Dict[int, float],
    glacier_hrus: Optional[Iterable[int]] = None,
    d_scale_mm: float = 50.0,
) -> pd.Series:
    """Read Raven SNOW BY_HRU (SWE), derive fSCA via ``swe_to_fsca``, and
    aggregate to basin-mean off-glacier fractional snow cover.

    Parameters
    ----------
    output_dir   : Raven output directory.
    hru_areas    : dict HRU id (int) -> area_km2.
    glacier_hrus : optional set/list of HRU ids treated as glacier and
                   excluded from the basin mean.
    d_scale_mm   : SWE saturation threshold for the depth-area function.
    """
    output_dir = Path(output_dir)
    # Match exact ..._SNOW_Daily_Average*ByHRU* — i.e. excludes SNOW_FRAC,
    # SNOWFALL, SNOW_LIQ, etc.
    matches = sorted(output_dir.glob('*_SNOW_Daily_Average*ByHRU*.csv'))
    if not matches:
        raise FileNotFoundError(
            f"No SNOW ByHRU CSV in {output_dir}.  "
            "Make sure ':CustomOutput DAILY AVERAGE SNOW BY_HRU' is in "
            "the .rvi during calibration."
        )
    df = _load_raven_by_hru_csv(matches[0])
    # Convert SWE → fSCA in place on the HRU columns
    drop_cols = {'time', 'hour', 'tag', 'HRU:'}
    hru_cols_all = [c for c in df.columns if c not in drop_cols]
    for c in hru_cols_all:
        try:
            int(c)
        except (ValueError, TypeError):
            continue
        df[c] = swe_to_fsca(pd.to_numeric(df[c], errors='coerce'), d_scale_mm)

    # HRU columns are integer IDs; drop non-data columns.
    drop_cols = {'time', 'hour', 'tag', 'HRU:'}
    hru_cols_raw = [c for c in df.columns if c not in drop_cols]
    hru_cols: list = []
    for c in hru_cols_raw:
        try:
            int(c)
            hru_cols.append(c)
        except (ValueError, TypeError):
            continue

    if glacier_hrus is not None:
        glacier_set = {int(h) for h in glacier_hrus}
        hru_cols = [c for c in hru_cols if int(c) not in glacier_set]
    if not hru_cols:
        raise ValueError("No non-glacier HRUs left after filtering")

    weights = np.array([hru_areas[int(c)] for c in hru_cols], dtype=float)
    if weights.sum() <= 0:
        raise ValueError("All HRU areas are zero")
    weights = weights / weights.sum()

    snowfrac = df[hru_cols].astype(float).values @ weights
    return pd.Series(snowfrac, index=df.index, name='fsca_sim')


def load_raven_snow_frac_per_band(
    output_dir: Union[str, Path],
    hru_areas: Dict[int, float],
    hru_elevations: Dict[int, float],
    glacier_hrus: Optional[Iterable[int]] = None,
    band_width_m: int = 100,
    d_scale_mm: float = 50.0,
) -> pd.DataFrame:
    """Read Raven SNOW BY_HRU (SWE), derive fSCA, aggregate per elevation band.

    Returns a DataFrame indexed by date, columns are band lower edges in
    metres (int). Each cell is the area-weighted mean fSCA over the HRUs in
    that band (glacier HRUs excluded). fSCA is derived from SWE via
    ``swe_to_fsca`` because Raven's `SNOW_FRAC` is the snowfall-fraction
    forcing, not fractional snow cover.
    """
    output_dir = Path(output_dir)
    # Match exact ..._SNOW_Daily_Average*ByHRU* — i.e. excludes SNOW_FRAC,
    # SNOWFALL, SNOW_LIQ, etc.
    matches = sorted(output_dir.glob('*_SNOW_Daily_Average*ByHRU*.csv'))
    if not matches:
        raise FileNotFoundError(
            f"No SNOW ByHRU CSV in {output_dir}.  "
            "Make sure ':CustomOutput DAILY AVERAGE SNOW BY_HRU' is in "
            "the .rvi during calibration."
        )
    df = _load_raven_by_hru_csv(matches[0])
    # Convert SWE → fSCA in place on the HRU columns
    drop_cols = {'time', 'hour', 'tag', 'HRU:'}
    for c in df.columns:
        if c in drop_cols:
            continue
        try:
            int(c)
        except (ValueError, TypeError):
            continue
        df[c] = swe_to_fsca(pd.to_numeric(df[c], errors='coerce'), d_scale_mm)

    drop_cols = {'time', 'hour', 'tag', 'HRU:'}
    glacier_set = {int(h) for h in (glacier_hrus or [])}

    # Build per-HRU columns list (integer IDs), excluding glaciers
    hru_ids = []
    for c in df.columns:
        if c in drop_cols:
            continue
        try:
            hid = int(c)
        except (ValueError, TypeError):
            continue
        if hid in glacier_set or hid not in hru_areas or hid not in hru_elevations:
            continue
        hru_ids.append(hid)

    if not hru_ids:
        raise ValueError("No HRUs with both area and elevation, "
                         "and not flagged glacier")

    # Per HRU: band id
    hru_to_band = {hid: int(np.floor(hru_elevations[hid] / band_width_m)
                            * band_width_m)
                   for hid in hru_ids}
    bands = sorted(set(hru_to_band.values()))

    per_band = {}
    for b in bands:
        hrus_in = [h for h in hru_ids if hru_to_band[h] == b]
        if not hrus_in:
            continue
        areas = np.array([hru_areas[h] for h in hrus_in], dtype=float)
        w = areas / areas.sum() if areas.sum() > 0 else None
        vals = df[[str(h) for h in hrus_in]].astype(float).values
        if w is None:
            per_band[b] = pd.Series(vals.mean(axis=1), index=df.index)
        else:
            per_band[b] = pd.Series(vals @ w, index=df.index)

    return pd.DataFrame(per_band)


def _compute_band_areas(
    hru_areas: Dict[int, float],
    hru_elevations: Dict[int, float],
    band_width_m: int,
    glacier_hrus: Optional[Iterable[int]] = None,
) -> Dict[int, float]:
    glacier_set = {int(h) for h in (glacier_hrus or [])}
    out: Dict[int, float] = {}
    for hid, area in hru_areas.items():
        if hid in glacier_set or hid not in hru_elevations:
            continue
        b = int(np.floor(hru_elevations[hid] / band_width_m) * band_width_m)
        out[b] = out.get(b, 0.0) + float(area)
    return out


def _write_sidecar(diagnostic_log: Path, row: dict) -> None:
    """Append a one-row diagnostic record to a CSV (create with header if new)."""
    log = Path(diagnostic_log)
    log.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log.exists()
    pd.DataFrame([row]).to_csv(log, mode='a', header=write_header, index=False)


def snow_objective(
    obs_fsca_csv: Union[str, Path],
    sim_output_dir: Union[str, Path],
    hru_areas: Dict[int, float],
    glacier_hrus: Optional[Iterable[int]] = None,
    metric: str = 'KGE',
    cloud_threshold: float = 0.5,
    date_range: Optional[tuple] = None,
    obs_tolerance_days: int = 4,
    aggregation: str = 'basin_mean',
    band_width_m: int = 100,
    min_pixels_per_band: int = 30,
    band_aggregation: str = 'area_weighted_mean',
    hru_elevations: Optional[Dict[int, float]] = None,
    diagnostic_log: Optional[Union[str, Path]] = None,
) -> float:
    """Compare MODIS fSCA against Raven SNOW_FRAC.

    Two aggregation modes:

    - **basin_mean** (default): area-weighted basin mean obs vs basin mean
      sim, one scalar metric on the matched series. Backward-compatible
      with the legacy single-band path.

    - **elevation_band**: per-band obs vs per-band sim, per-band metric,
      rolled up to a single score via ``band_aggregation``. Requires
      ``hru_elevations`` (HRU id → elevation in m). The obs CSV must be the
      per-band long-format file emitted by ``preprocess_modis_fsca``.

    Parameters
    ----------
    metric            : KGE | NSE | LogKGE | RMSE | nRMSE | MAE | PBIAS | CSI.
                        Higher-is-better in all cases (error metrics are
                        score-converted).  For fSCA in [0,1], prefer nRMSE
                        over RMSE when combining with KGE-on-Q in a weighted
                        sum — plain (1-RMSE) sits in [0.6, 0.9] and
                        structurally over-weights snow in the combination.
    band_aggregation  : area_weighted_mean | mean | median | min.
    min_pixels_per_band : skip a (date, band) cell if fewer valid pixels.
    diagnostic_log    : optional path; if set, appends one row per call
                        with raw r/rmse/mae/pbias diagnostics.
    """
    if aggregation not in ('basin_mean', 'elevation_band'):
        raise ValueError(f"aggregation must be 'basin_mean' or "
                         f"'elevation_band'; got {aggregation!r}")

    if aggregation == 'basin_mean':
        obs = load_modis_fsca(obs_fsca_csv, cloud_threshold=cloud_threshold)
        sim = load_raven_snow_frac(sim_output_dir, hru_areas, glacier_hrus)
        sim_at_obs = sim.reindex(
            obs.index, method='nearest',
            tolerance=pd.Timedelta(days=obs_tolerance_days),
        )
        if date_range is not None:
            obs = obs.loc[date_range[0]:date_range[1]]
            sim_at_obs = sim_at_obs.loc[date_range[0]:date_range[1]]
        score = _apply_metric(metric, obs, sim_at_obs)
        if diagnostic_log:
            diag = raw_diagnostics(obs, sim_at_obs)
            diag.update({'metric': metric, 'score': score, 'band': 'basin'})
            _write_sidecar(diagnostic_log, diag)
        return score

    # elevation_band mode -------------------------------------------------
    if hru_elevations is None:
        raise ValueError("hru_elevations is required for "
                         "aggregation='elevation_band'.")

    obs_bands = load_modis_fsca_bands(
        obs_fsca_csv, cloud_threshold=cloud_threshold,
        min_pixels_per_band=min_pixels_per_band,
    )
    sim_bands = load_raven_snow_frac_per_band(
        sim_output_dir, hru_areas, hru_elevations,
        glacier_hrus=glacier_hrus, band_width_m=band_width_m,
    )
    band_area = _compute_band_areas(hru_areas, hru_elevations, band_width_m,
                                    glacier_hrus=glacier_hrus)

    common_bands = sorted(set(obs_bands.columns) & set(sim_bands.columns))
    if not common_bands:
        return float('nan')

    per_band_scores: Dict[int, float] = {}
    for b in common_bands:
        obs_b = obs_bands[b].dropna()
        if len(obs_b) < 30:
            continue
        sim_b = sim_bands[b].reindex(
            obs_b.index, method='nearest',
            tolerance=pd.Timedelta(days=obs_tolerance_days),
        )
        if date_range is not None:
            obs_b = obs_b.loc[date_range[0]:date_range[1]]
            sim_b = sim_b.loc[date_range[0]:date_range[1]]
        score = _apply_metric(metric, obs_b, sim_b)
        if not np.isnan(score):
            per_band_scores[b] = score
        if diagnostic_log:
            diag = raw_diagnostics(obs_b, sim_b)
            diag.update({'metric': metric, 'score': score, 'band': int(b),
                         'area_km2': float(band_area.get(b, 0.0))})
            _write_sidecar(diagnostic_log, diag)

    if not per_band_scores:
        return float('nan')

    bands = list(per_band_scores.keys())
    scores = np.array([per_band_scores[b] for b in bands], dtype=float)

    if band_aggregation == 'area_weighted_mean':
        w = np.array([band_area.get(b, 0.0) for b in bands], dtype=float)
        if w.sum() == 0:
            return float(scores.mean())
        return float(np.average(scores, weights=w))
    if band_aggregation == 'mean':
        return float(scores.mean())
    if band_aggregation == 'median':
        return float(np.median(scores))
    if band_aggregation == 'min':
        return float(scores.min())
    raise ValueError(f"Unknown band_aggregation: {band_aggregation!r}")


# ---------------------------------------------------------------------------
# Baseflow objective
# ---------------------------------------------------------------------------

def _resolve_window(window) -> tuple:
    """Convert a window spec to a tuple of month integers.

    Accepted forms:
      - 'winter'       → (11, 12, 1, 2, 3)   (NDJFM, 5 months)
      - 'raw_winter'   → (12, 1, 2, 3)        (DJFM, 4 months — legacy, Hunza pilot)
      - 'deep_winter'  → (12, 1, 2)           (DJF, 3 months — paper-5 Phase 1+)
                                                Reliably below freezing across all
                                                cold high-mountain study catchments
                                                regardless of regime.
      - 'huang_winter' → (11, 12, 1, 2)       (NDJF — matches Huang et al. 2026 Pamir)
      - 'all'          → (1, 2, …, 12)
      - list/tuple of month ints, e.g. [12, 1, 2] for DJF only
    """
    if isinstance(window, (list, tuple)) and not isinstance(window, str):
        months = tuple(int(m) for m in window)
        if not months:
            raise ValueError("Custom baseflow window is empty.")
        bad = [m for m in months if not 1 <= m <= 12]
        if bad:
            raise ValueError(
                f"Custom baseflow window months must be 1-12; got {bad}."
            )
        return months

    if window == 'winter':
        return WINTER_MONTHS
    if window == 'raw_winter':
        return (12, 1, 2, 3)
    if window == 'deep_winter':
        return (12, 1, 2)
    if window == 'huang_winter':
        return (11, 12, 1, 2)
    if window == 'all':
        return tuple(range(1, 13))

    raise ValueError(
        f"Unknown window {window!r}. Use a preset "
        f"('winter' | 'raw_winter' | 'deep_winter' | 'huang_winter' | 'all') "
        f"or an explicit list of month integers (e.g. [12, 1, 2])."
    )


def baseflow_objective(
    obs_Q: pd.Series,
    sim_Q: pd.Series,
    method: str = 'eckhardt',
    window='winter',
    metric: str = 'KGE',
    date_range: Optional[tuple] = None,
    eckhardt_kwargs: Optional[dict] = None,
) -> float:
    """KGE between observed and simulated baseflow.

    Both Q series run through the same filter (eckhardt / lyne_hollick /
    sliding_min) — or, for ``method='raw_winter'``, no filter is applied
    and the raw Q series is used directly under the mass-balance
    assumption Q ≈ baseflow during the chosen window.

    Parameters
    ----------
    method : {'eckhardt', 'lyne_hollick', 'sliding_min', 'raw_winter'}
    window : str or list of int
        Months to evaluate the metric on. Presets: 'winter' (Nov-Mar),
        'raw_winter' (Dec-Mar — tighter; recommended companion for
        method='raw_winter'), 'all' (12 months). Or an explicit list of
        month integers, e.g. ``[12, 1, 2]`` for DJF.

        With ``method='raw_winter'`` and ``window='winter'`` (the legacy
        default), the window is auto-tightened to Dec-Mar to avoid
        autumn-drainage residuals. Any explicit list of months is taken
        as-is (the user chose it, so we don't second-guess).
    """
    eck = eckhardt_kwargs or {}

    if method == 'raw_winter':
        obs_bf = obs_Q.copy()
        sim_bf = sim_Q.copy()
        # Auto-tighten only if the user is on the default preset 'winter'.
        # Explicit windows (list or other preset) are respected as-is.
        if window == 'winter':
            window = 'raw_winter'
    else:
        obs_sep = BaseflowSeparator(obs_Q.dropna())
        sim_sep = BaseflowSeparator(sim_Q.dropna())

        if method == 'eckhardt':
            obs_bf = obs_sep.eckhardt(**eck)
            sim_bf = sim_sep.eckhardt(**eck)
        elif method in ('lyne_hollick', 'lyne-hollick'):
            obs_bf = obs_sep.lyne_hollick()
            sim_bf = sim_sep.lyne_hollick()
        elif method in ('sliding_min', 'sliding_minimum'):
            obs_bf = obs_sep.sliding_minimum()
            sim_bf = sim_sep.sliding_minimum()
        else:
            raise ValueError(f"Unknown baseflow method '{method}'")

    months = _resolve_window(window)
    obs_bf = obs_bf[obs_bf.index.month.isin(months)]
    sim_bf = sim_bf[sim_bf.index.month.isin(months)]

    if date_range is not None:
        obs_bf = obs_bf.loc[date_range[0]:date_range[1]]
        sim_bf = sim_bf.loc[date_range[0]:date_range[1]]
    return _apply_metric(metric, obs_bf, sim_bf)


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

def resolve_modis_fsca_path(
    gauge_id: str,
    display_name: Optional[str] = None,
    explicit_path: Optional[Union[str, Path]] = None,
    product: str = 'MOD10A2',
    smb_root: Optional[Union[str, Path]] = None,
    main_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve the MODIS fSCA CSV for a catchment.

    Precedence:
      1. `explicit_path`             — caller wins absolutely.
      2. `smb_root` if given         — legacy SMB layout with display_name prefix.
      3. `main_dir`                  — local layout produced by
         scripts/derive_basin_fsca.py at
         <main_dir>/01_data/snow/MODIS/basins/<gauge>/fsca_<product>_<gauge>.csv.

    The local-main_dir branch is the canonical one for new runs; the SMB
    branch is kept for back-compat with the older
    `<smb_root>/basins/<display>_<gauge>/` directory layout.
    """
    if explicit_path is not None:
        return Path(explicit_path)

    if smb_root is not None:
        # Legacy SMB layout: basins are named "<display_name>_<gauge>/".
        smb_root = Path(smb_root)
        folder_name = (f"{display_name}_{gauge_id}" if display_name
                       else str(gauge_id))
        return smb_root / "basins" / folder_name / f"fsca_{product}_{gauge_id}.csv"

    if main_dir is not None:
        return (Path(main_dir) / '01_data' / 'snow' / 'MODIS' / 'basins'
                / str(gauge_id) / f"fsca_{product}_{gauge_id}.csv")

    raise ValueError(
        "resolve_modis_fsca_path needs at least one of: explicit_path, "
        "smb_root (legacy), or main_dir (canonical). Pass main_dir from the "
        "catchment namelist or set snow.fsca_csv explicitly in the namelist."
    )
