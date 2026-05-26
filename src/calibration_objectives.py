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
# Metric registry — higher is better.  Add more as needed.
# ---------------------------------------------------------------------------

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


METRICS = {
    'KGE':    _kge,
    'NSE':    _nse,
    'LogKGE': _logkge,
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
    """Load MODIS basin-mean fSCA produced by `downloads/download_MODIS.py`.

    Rows where `n_cloud / n_total > cloud_threshold` are masked NaN.
    """
    df = pd.read_csv(csv_path, parse_dates=['date'])
    cloud_frac = df['n_cloud'] / df['n_total'].replace(0, np.nan)
    fsca = df['fsca'].astype(float)
    fsca = fsca.where(cloud_frac <= cloud_threshold, other=np.nan)
    return pd.Series(fsca.values, index=df['date'], name='fsca_obs')


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


def snow_objective(
    obs_fsca_csv: Union[str, Path],
    sim_output_dir: Union[str, Path],
    hru_areas: Dict[int, float],
    glacier_hrus: Optional[Iterable[int]] = None,
    metric: str = 'KGE',
    cloud_threshold: float = 0.5,
    date_range: Optional[tuple] = None,
    obs_tolerance_days: int = 4,
) -> float:
    """KGE/NSE between MODIS basin-mean fSCA and Raven basin-mean off-glacier
    SNOW_FRAC at MODIS observation dates.

    MODIS MOD10A2 is an 8-day max-snow composite.  We sample the simulated
    daily timeseries at each MODIS date (nearest match within `obs_tolerance_days`).
    """
    obs = load_modis_fsca(obs_fsca_csv, cloud_threshold=cloud_threshold)
    sim = load_raven_snow_frac(sim_output_dir, hru_areas, glacier_hrus)

    sim_at_obs = sim.reindex(
        obs.index,
        method='nearest',
        tolerance=pd.Timedelta(days=obs_tolerance_days),
    )

    if date_range is not None:
        obs = obs.loc[date_range[0]:date_range[1]]
        sim_at_obs = sim_at_obs.loc[date_range[0]:date_range[1]]
    return _apply_metric(metric, obs, sim_at_obs)


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
) -> Path:
    """Resolve the MODIS fSCA CSV for a catchment.

    Precedence: explicit_path > smb_root/basins/<name>_<gauge>/<csv>.

    `display_name` is the basin name used as the folder prefix (e.g. 'Hunza').
    If omitted, falls back to just the gauge_id.
    """
    if explicit_path is not None:
        return Path(explicit_path)

    if smb_root is None:
        smb_root = (f"/run/user/{os.getuid()}/gvfs/"
                    f"smb-share:server=hydroshare.giub.unibe.ch,share=data"
                    f"/Meteorology/Global/MODIS")
    smb_root = Path(smb_root)

    folder_name = (f"{display_name}_{gauge_id}" if display_name
                   else str(gauge_id))
    return smb_root / "basins" / folder_name / f"fsca_{product}_{gauge_id}.csv"
