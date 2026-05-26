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
    'NSE':    _nse,
    'LogKGE': _logkge,
    'RMSE':   _rmse,         # score form: 1 - RMSE (assumes bounded variable)
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
    return df.pivot(index='date', columns='band_m', values='fsca')


def load_raven_snow_frac(
    output_dir: Union[str, Path],
    hru_areas: Dict[int, float],
    glacier_hrus: Optional[Iterable[int]] = None,
) -> pd.Series:
    """Read Raven `:CustomOutput DAILY AVERAGE SNOW_FRAC BY_HRU` and aggregate
    to basin-mean off-glacier snow cover fraction.

    Raven names the file `<prefix>_SNOW_FRAC_Daily_Average_ByHRU.csv` with
    columns: 'time', 'date', 'hour', then one column per HRU.

    Parameters
    ----------
    output_dir   : Raven output directory.
    hru_areas    : dict HRU id (int) -> area_km2.
    glacier_hrus : optional set/list of HRU ids treated as glacier and
                   excluded from the basin mean.
    """
    output_dir = Path(output_dir)
    matches = sorted(output_dir.glob('*SNOW_FRAC*Daily_Average*ByHRU*.csv'))
    if not matches:
        raise FileNotFoundError(
            f"No SNOW_FRAC ByHRU CSV in {output_dir}.  "
            "Make sure ':CustomOutput DAILY AVERAGE SNOW_FRAC BY_HRU' is in "
            "the .rvi during calibration."
        )

    # Raven CustomOutput files prepend a one-line title (":CustomOutput …")
    # before the actual CSV header. Other writers omit it. Peek line 1 and
    # skip it iff it's a Raven directive.
    src = matches[0]
    with open(src) as f:
        first = f.readline().lstrip()
    skip = 1 if first.startswith(':') else 0
    df = pd.read_csv(src, skiprows=skip)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # HRU columns are integer IDs; drop non-data columns.
    drop_cols = {'time', 'hour', 'tag'}
    hru_cols_raw = [c for c in df.columns if c not in drop_cols]
    hru_cols: list = []
    for c in hru_cols_raw:
        try:
            int(c)
            hru_cols.append(c)
        except ValueError:
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
) -> pd.DataFrame:
    """Read Raven SNOW_FRAC BY_HRU and aggregate per elevation band.

    Returns a DataFrame indexed by date, columns are band lower edges in
    metres (int). Each cell is the area-weighted mean SNOW_FRAC over the
    HRUs in that band (glacier HRUs excluded).
    """
    output_dir = Path(output_dir)
    matches = sorted(output_dir.glob('*SNOW_FRAC*Daily_Average*ByHRU*.csv'))
    if not matches:
        raise FileNotFoundError(
            f"No SNOW_FRAC ByHRU CSV in {output_dir}.  "
            "Make sure ':CustomOutput DAILY AVERAGE SNOW_FRAC BY_HRU' is in "
            "the .rvi during calibration."
        )

    src = matches[0]
    with open(src) as f:
        first = f.readline().lstrip()
    skip = 1 if first.startswith(':') else 0
    df = pd.read_csv(src, skiprows=skip)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    drop_cols = {'time', 'hour', 'tag'}
    glacier_set = {int(h) for h in (glacier_hrus or [])}

    # Build per-HRU columns list (integer IDs), excluding glaciers
    hru_ids = []
    for c in df.columns:
        if c in drop_cols:
            continue
        try:
            hid = int(c)
        except ValueError:
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
    metric            : KGE | NSE | LogKGE | RMSE | MAE | PBIAS | CSI.
                        Higher-is-better in all cases (error metrics are
                        score-converted).
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

def baseflow_objective(
    obs_Q: pd.Series,
    sim_Q: pd.Series,
    method: str = 'eckhardt',
    window: str = 'winter',
    metric: str = 'KGE',
    date_range: Optional[tuple] = None,
    eckhardt_kwargs: Optional[dict] = None,
) -> float:
    """KGE between observed and simulated baseflow.

    Both Q series run through the same filter; metric is then computed on
    the configured window (winter = Nov-Mar default).

    method  : 'eckhardt' | 'lyne_hollick' | 'sliding_min' | 'raw_winter'
              ('raw_winter' assumes Q ≡ baseflow for high-altitude
               truly-frozen catchments — Dec-Mar window only.)
    window  : 'winter' (Nov-Mar) | 'all'
    """
    eck = eckhardt_kwargs or {}

    if method == 'raw_winter':
        obs_bf = obs_Q.copy()
        sim_bf = sim_Q.copy()
        # raw_winter forces a tighter Dec-Mar window
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

    if window == 'winter':
        months = WINTER_MONTHS
    elif window == 'raw_winter':
        months = (12, 1, 2, 3)
    elif window == 'all':
        months = tuple(range(1, 13))
    else:
        raise ValueError(f"Unknown window '{window}'")

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
