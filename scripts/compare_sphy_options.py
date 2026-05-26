"""Compare SPHY perc-option × lateral-storage variants across catchments.

Loads each (catchment, configuration, calibration_metric) calibration run and
emits two kinds of analysis:

1. Streamflow performance — KGE / NSE / KGE_winter / KGE_WB on the validation
   period, per config, per calibration metric. CSV tables + bar plots.

2. Subsurface storage — area-weighted catchment-mean and cross-HRU spread of
   FAST_RESERVOIR (SOIL[1]) and SLOW_RESERVOIR (SOIL[2]) for each config.
   This is where opt1↔opt2 (different percolation) and shared↔per-HRU
   (lateral_equilibrate) actually show up.

Designed to run on the GIUB server where the calibration outputs live. Uses
config_merge.load_config so paths and dates come from the same source the
calibration pipeline used.

Usage
-----
    python scripts/compare_sphy_options.py                     # all SPHY catchments
    python scripts/compare_sphy_options.py --catchments 0102 2268
    python scripts/compare_sphy_options.py --configs glogem_subdaily_opt1 \\
                                            glogem_subdaily_opt2
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from config_merge import load_config  # noqa: E402
from paths import get_paths  # noqa: E402

DEFAULT_METRICS = ['KGE', 'KGE_WB']
MODEL_TYPE = 'SPHY'

CONFIG_COLORS = {
    'glogem_subdaily_opt1':         '#1f77b4',
    'glogem_subdaily_opt2':         '#ff7f0e',
    'glogem_subdaily_opt1_shared':  '#2ca02c',
    'glogem_subdaily_opt2_shared':  '#d62728',
    'glogem_tphipr_opt1':           '#9467bd',
    'glogem_tphipr_opt2':           '#8c564b',
    'glogem_tphipr_opt1_shared':    '#e377c2',
    'glogem_tphipr_opt2_shared':    '#7f7f7f',
}

FORCING_COLORS = {
    'ERA5-Land':  '#1f77b4',
    'TPHiPr':     '#d62728',
    'MeteoSwiss': '#2ca02c',
}


# ── Config classification ──────────────────────────────────────────────────

def classify_config(config_key: str, region: Optional[str] = None) -> dict:
    """Split a config key into its three orthogonal axes.

    Returns dict with keys: forcing, perc_option, subsurface.
      forcing     : 'ERA5-Land' | 'TPHiPr' | 'MeteoSwiss' | 'unknown'
      perc_option : 'opt1' | 'opt2' | 'unknown'
      subsurface  : 'shared' | 'per-HRU'
    """
    key = config_key.lower()
    if 'tphipr' in key:
        forcing = 'TPHiPr'
    elif region and region.lower() == 'switzerland':
        # `glogem_subdaily_*` uses MeteoSwiss for Swiss catchments
        # via the region layer override (see src/config/layers/region/switzerland.yaml)
        forcing = 'MeteoSwiss'
    elif 'subdaily' in key:
        forcing = 'ERA5-Land'
    else:
        forcing = 'unknown'

    if '_opt1' in key:
        perc = 'opt1'
    elif '_opt2' in key:
        perc = 'opt2'
    else:
        perc = 'unknown'

    subsurface = 'shared' if 'shared' in key else 'per-HRU'
    return {'forcing': forcing, 'perc_option': perc, 'subsurface': subsurface}


def configs_from_namelist(gauge_id: str) -> List[str]:
    """Read the `configurations:` list from a catchment's SPHY namelist.

    Falls back to the legacy 4-config default if the SPHY namelist isn't
    present or doesn't list a configurations block.
    """
    import yaml
    nml_path = ROOT / 'namelists' / f'catchment_{gauge_id}_SPHY.yaml'
    if not nml_path.exists():
        nml_path = ROOT / 'namelists' / f'catchment_{gauge_id}.yaml'
    if not nml_path.exists():
        return ['glogem_subdaily_opt1', 'glogem_subdaily_opt2',
                'glogem_subdaily_opt1_shared', 'glogem_subdaily_opt2_shared']
    with open(nml_path) as f:
        nml = yaml.safe_load(f) or {}
    cfgs = nml.get('configurations', [])
    return [c.strip() for c in cfgs if isinstance(c, str)]


# ── Metric calculation ──────────────────────────────────────────────────────

def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    """Kling-Gupta efficiency (standard 2009 formulation)."""
    mask = ~(np.isnan(sim) | np.isnan(obs))
    sim, obs = sim[mask], obs[mask]
    if len(sim) < 30 or obs.std() == 0 or obs.mean() == 0:
        return float('nan')
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    mask = ~(np.isnan(sim) | np.isnan(obs))
    sim, obs = sim[mask], obs[mask]
    if len(sim) < 30:
        return float('nan')
    denom = ((obs - obs.mean()) ** 2).sum()
    if denom == 0:
        return float('nan')
    return float(1 - ((sim - obs) ** 2).sum() / denom)


WINTER_MONTHS    = (11, 12, 1, 2, 3)        # baseflow-dominated
NONWINTER_MONTHS = (4, 5, 6, 7, 8, 9, 10)   # snowmelt + glacier-melt + rainfall


def compute_metrics(df: pd.DataFrame, val_start: str, val_end: str) -> dict:
    """Return seasonal KGE/NSE breakdown on the validation window.

    Returns
    -------
    dict with keys:
      KGE, NSE             — full-year validation
      KGE_winter           — Nov–Mar (winter baseflow signal)
      KGE_nonwinter        — Apr–Oct (peak melt / rainfall season)
      KGE_WB               — 0.5*KGE + 0.5*KGE_winter (matches diagnostic.py)

    All values are NaN-safe: if a season has <30 valid days, that season's
    metric is NaN and KGE_WB falls through to NaN too.
    """
    mask = (df['date'] >= val_start) & (df['date'] <= val_end)
    df = df.loc[mask].dropna(subset=['sim_Q', 'obs_Q'])
    keys = ('KGE', 'NSE', 'KGE_winter', 'KGE_nonwinter', 'KGE_WB')
    if df.empty:
        return {m: float('nan') for m in keys}

    sim = df['sim_Q'].to_numpy(dtype=float)
    obs = df['obs_Q'].to_numpy(dtype=float)
    k = kge(sim, obs)
    n = nse(sim, obs)

    winter = df['date'].dt.month.isin(WINTER_MONTHS)
    k_w = kge(df.loc[winter,  'sim_Q'].to_numpy(dtype=float),
              df.loc[winter,  'obs_Q'].to_numpy(dtype=float))
    k_nw = kge(df.loc[~winter, 'sim_Q'].to_numpy(dtype=float),
               df.loc[~winter, 'obs_Q'].to_numpy(dtype=float))

    k_wb = (0.5 * k + 0.5 * k_w
            if not math.isnan(k_w) and not math.isnan(k) else float('nan'))

    return {'KGE': k, 'NSE': n,
            'KGE_winter': k_w, 'KGE_nonwinter': k_nw, 'KGE_WB': k_wb}


# ── File loaders ────────────────────────────────────────────────────────────

def load_hydrograph(output_dir: Path, gauge_id: str) -> Optional[pd.DataFrame]:
    """Load Raven hydrograph CSV → DataFrame[date, sim_Q, obs_Q]."""
    f = output_dir / f"{gauge_id}_{MODEL_TYPE}_Hydrographs.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, skiprows=[1])
    df['date'] = pd.to_datetime(df['date'])
    sim_col, obs_col = None, None
    for c in df.columns:
        if c.endswith('(observed) [m3/s]'):
            obs_col = c
        elif c.endswith('[m3/s]') and 'observed' not in c.lower():
            sim_col = c
    if sim_col is None or obs_col is None:
        return None
    return pd.DataFrame({'date': df['date'], 'sim_Q': df[sim_col], 'obs_Q': df[obs_col]})


def load_soil_by_hru(output_dir: Path, soil_idx: int) -> Optional[pd.DataFrame]:
    """Load Raven ':CustomOutput DAILY AVERAGE SOIL[N] BY_HRU' CSV (the
    storage state, not the cumulative fluxes).

    Brackets are literal in the filename so we glob and substring-match.
    Critically, several files share the SOIL[N]_Daily_Average_ByHRU
    suffix — we want the *storage* one and must exclude the flux variants:

        2268_SPHY_SOIL[1]_Daily_Average_ByHRU.csv               ← storage  ✓
        2268_SPHY_FROM_SOIL[1]_Daily_Average_ByHRU.csv          ← cumulative out
        2268_SPHY_BETWEEN_SOIL[1]_AND_SURFACE_WATER_..._ByHRU   ← cumulative between

    Filtering by substring alone was picking FROM_SOIL alphabetically first.

    Returns DataFrame indexed by date, columns = HRU IDs (int), values = mm.
    """
    needle = f"_SOIL[{soil_idx}]_Daily_Average_ByHRU"
    exclude = (f"FROM_SOIL[{soil_idx}]", f"BETWEEN_SOIL[{soil_idx}]")
    matches = sorted(
        p for p in output_dir.glob('*.csv')
        if needle in p.name and not any(x in p.name for x in exclude)
    )
    if not matches:
        return None
    df = pd.read_csv(matches[0], skiprows=1)
    if 'date' not in df.columns and 'day' in df.columns:
        df['date'] = pd.to_datetime(df['day'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        return None
    drop = {'time', 'day', 'date', 'hour'}
    hru_cols = [c for c in df.columns if c not in drop]
    out = df.set_index('date')[hru_cols].apply(pd.to_numeric, errors='coerce')
    # Column names look like "1", "2", ... or "HRU 1", "HRU 2", ... — extract int
    new_cols = []
    for c in out.columns:
        digits = ''.join(ch for ch in str(c) if ch.isdigit())
        new_cols.append(int(digits) if digits else c)
    out.columns = new_cols
    return out


def load_hru_areas(topo_dir: Path) -> Optional[pd.Series]:
    """Load HRU_ID → Area_km2 from topo_files/<variant>/HRU_table.csv."""
    f = topo_dir / 'HRU_table.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f)
    if 'HRU_ID' not in df.columns or 'Area_km2' not in df.columns:
        return None
    return df.set_index('HRU_ID')['Area_km2']


# ── Per-catchment analysis ──────────────────────────────────────────────────

def build_nml(catchment: str, config_key: str, metric: str, env: str) -> dict:
    """Use the same merge order the pipeline uses, so paths line up.

    `load_config` returns `(merged_dict, temp_yaml_path)` — we only need the
    dict for plotting / paths, so drop the temp path.
    """
    merged, _tmp_path = load_config(
        catchment=catchment,
        configuration=config_key,
        model=MODEL_TYPE,
        env=env,
        overrides={'_calibration_metric': metric},
    )
    return merged


def analyze_catchment(
    gauge_id: str,
    configs: list[str],
    metrics: list[str],
    env: str,
) -> dict:
    """Return {'performance': DataFrame, 'storage': DataFrame, 'regime': DataFrame}.

    performance: long-format (cali_metric, config, eval_metric, value)
    storage: long-format (cali_metric, config, reservoir, date, mean_mm, std_mm)
    regime: long-format (cali_metric, config, month, sim_Q, obs_Q)
    """
    perf_rows, storage_rows, regime_rows = [], [], []

    for cali_metric in metrics:
        for cfg_key in configs:
            try:
                nml = build_nml(gauge_id, cfg_key, cali_metric, env)
            except FileNotFoundError as e:
                print(f"  [{gauge_id}/{cfg_key}/{cali_metric}] skip: {e}")
                continue

            paths = get_paths(nml)
            output_dir = paths['output_dir']
            if not output_dir.exists():
                print(f"  [{gauge_id}/{cfg_key}/{cali_metric}] missing: {output_dir}")
                continue

            cali_end = nml.get('cali_end_date')
            end = nml.get('end_date')
            if not cali_end or not end:
                print(f"  [{gauge_id}/{cfg_key}/{cali_metric}] no validation dates")
                continue

            # --- Performance ---
            hydro = load_hydrograph(output_dir, gauge_id)
            if hydro is None:
                print(f"  [{gauge_id}/{cfg_key}/{cali_metric}] no Hydrographs.csv")
                continue
            m = compute_metrics(hydro, cali_end, end)
            for ename, val in m.items():
                perf_rows.append({
                    'gauge_id': gauge_id, 'cali_metric': cali_metric,
                    'config': cfg_key, 'eval_metric': ename, 'value': val,
                })

            # --- Regime (validation period monthly means) ---
            val_mask = (hydro['date'] >= cali_end) & (hydro['date'] <= end)
            v = hydro.loc[val_mask].copy()
            if not v.empty:
                v['month'] = v['date'].dt.month
                monthly = v.groupby('month').agg(sim_Q=('sim_Q', 'mean'),
                                                 obs_Q=('obs_Q', 'mean')).reset_index()
                for _, row in monthly.iterrows():
                    regime_rows.append({
                        'gauge_id': gauge_id, 'cali_metric': cali_metric, 'config': cfg_key,
                        'month': int(row['month']),
                        'sim_Q': float(row['sim_Q']), 'obs_Q': float(row['obs_Q']),
                    })

            # --- Storage (FAST = SOIL[1], SLOW = SOIL[2]) ---
            areas = load_hru_areas(paths['topo_dir'])
            for soil_idx, label in [(1, 'FAST_RESERVOIR'), (2, 'SLOW_RESERVOIR')]:
                soil = load_soil_by_hru(output_dir, soil_idx)
                if soil is None:
                    continue
                # Align HRUs that exist in both areas and the SOIL table
                if areas is not None:
                    common = [h for h in soil.columns if h in areas.index]
                    if not common:
                        continue
                    soil = soil[common]
                    w = areas.loc[common].to_numpy(dtype=float)
                    w = w / w.sum()
                    mean = soil.to_numpy(dtype=float) @ w
                else:
                    mean = soil.to_numpy(dtype=float).mean(axis=1)
                std = soil.to_numpy(dtype=float).std(axis=1)
                ts = pd.DataFrame({
                    'date': soil.index, 'mean_mm': mean, 'std_mm': std,
                })
                ts['gauge_id'] = gauge_id
                ts['cali_metric'] = cali_metric
                ts['config'] = cfg_key
                ts['reservoir'] = label
                storage_rows.append(ts)

    return {
        'performance': pd.DataFrame(perf_rows),
        'regime': pd.DataFrame(regime_rows),
        'storage': pd.concat(storage_rows, ignore_index=True) if storage_rows else pd.DataFrame(),
    }


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_catchment_performance(perf: pd.DataFrame, out_path: Path) -> None:
    """Per catchment: 4-panel bar chart, one panel per eval metric
    (KGE, KGE_winter, KGE_nonwinter, KGE_WB), bars per config × cali_metric.

    KGE_winter is the baseflow proxy; KGE_nonwinter is the peak-melt-season
    fit. Plotted side by side so it's easy to see which config does well
    where.
    """
    if perf.empty:
        return
    eval_metrics = ['KGE', 'KGE_winter', 'KGE_nonwinter', 'KGE_WB']
    cali_metrics = sorted(perf['cali_metric'].unique())
    configs = sorted(perf['config'].unique())

    fig, axes = plt.subplots(1, len(eval_metrics),
                             figsize=(4.5 * len(eval_metrics), 4),
                             sharey=True, squeeze=False)
    x = np.arange(len(configs))
    width = 0.8 / max(len(cali_metrics), 1)

    for ax, em in zip(axes[0], eval_metrics):
        for i, cm in enumerate(cali_metrics):
            vals = [perf[(perf['config'] == c)
                         & (perf['cali_metric'] == cm)
                         & (perf['eval_metric'] == em)]['value'].mean()
                    for c in configs]
            offset = (i - (len(cali_metrics) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=f'cal: {cm}')
        ax.set_title(em)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('glogem_', '') for c in configs],
                           rotation=25, ha='right', fontsize=8)
        ax.set_ylim(min(0, *[v for v in ax.containers[0].datavalues if not np.isnan(v)] or [0]), 1)
        ax.axhline(0.5, color='grey', lw=0.5, ls='--')
        ax.grid(axis='y', alpha=0.3)
    axes[0, 0].set_ylabel('Validation score')
    axes[0, 0].legend(fontsize=8, loc='lower left')
    fig.suptitle(f"{perf['gauge_id'].iloc[0]} — SPHY option performance "
                 f"(seasonal breakdown)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_forcing_comparison(perf: pd.DataFrame, region: Optional[str],
                            out_path: Path) -> None:
    """For catchments with multiple forcings (e.g. Indus: ERA5-Land vs TPHiPr),
    show paired bars per (catchment, eval_metric, forcing).

    Skips emit if only one forcing source is present.
    """
    if perf.empty:
        return
    # Classify each row
    perf = perf.copy()
    perf['forcing'] = perf['config'].map(
        lambda c: classify_config(c, region)['forcing']
    )
    forcings = sorted(perf['forcing'].unique())
    if len(forcings) < 2:
        return  # nothing to compare

    eval_metrics = ['KGE', 'KGE_winter', 'KGE_nonwinter', 'KGE_WB']
    catchments = sorted(perf['gauge_id'].unique())

    # For each (catchment, eval_metric, forcing), take the MAX score across
    # the configs sharing that forcing — i.e. "best variant of this forcing".
    # This isolates the forcing axis from the perc/shared axes.
    fig, axes = plt.subplots(1, len(eval_metrics),
                             figsize=(4.5 * len(eval_metrics), 4),
                             sharey=True, squeeze=False)
    x = np.arange(len(catchments))
    width = 0.8 / len(forcings)

    for ax, em in zip(axes[0], eval_metrics):
        for i, f in enumerate(forcings):
            vals = []
            for gid in catchments:
                sub = perf[(perf['gauge_id'] == gid)
                           & (perf['eval_metric'] == em)
                           & (perf['forcing'] == f)]
                vals.append(sub['value'].max() if not sub.empty else np.nan)
            offset = (i - (len(forcings) - 1) / 2) * width
            ax.bar(x + offset, vals, width,
                   label=f, color=FORCING_COLORS.get(f, None))
        ax.set_title(em)
        ax.set_xticks(x)
        ax.set_xticklabels(catchments, rotation=0)
        ax.axhline(0.5, color='grey', lw=0.5, ls='--')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
    axes[0, 0].set_ylabel('Best validation score across perc/shared variants')
    axes[0, 0].legend(fontsize=9, loc='lower left')
    fig.suptitle('Forcing source comparison — best variant per forcing',
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_catchment_regime(regime: pd.DataFrame, out_path: Path) -> None:
    if regime.empty:
        return
    cali_metrics = sorted(regime['cali_metric'].unique())
    fig, axes = plt.subplots(1, len(cali_metrics), figsize=(6 * len(cali_metrics), 4),
                             sharey=True, squeeze=False)
    for ax, cm in zip(axes[0], cali_metrics):
        sub = regime[regime['cali_metric'] == cm]
        # observed (same for all configs in this catchment+metric — take first)
        obs_first = sub.drop_duplicates(subset=['month']).sort_values('month')
        ax.plot(obs_first['month'], obs_first['obs_Q'], 'k-', lw=2, label='observed')
        for cfg in sorted(sub['config'].unique()):
            r = sub[sub['config'] == cfg].sort_values('month')
            ax.plot(r['month'], r['sim_Q'],
                    color=CONFIG_COLORS.get(cfg, None),
                    label=cfg.replace('glogem_subdaily_', ''))
        ax.set_title(f'calibrated on {cm}')
        ax.set_xlabel('month')
        ax.set_xticks(range(1, 13))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel('Q [m³/s]')
    fig.suptitle(f"{regime['gauge_id'].iloc[0]} — validation regime", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_catchment_storage(storage: pd.DataFrame, out_path: Path) -> None:
    """Storage time series — mean line + ±1σ cross-HRU spread band.

    Faceted: rows = (FAST, SLOW), cols = cali_metrics. One catchment per file.
    """
    if storage.empty:
        return
    cali_metrics = sorted(storage['cali_metric'].unique())
    reservoirs = ['FAST_RESERVOIR', 'SLOW_RESERVOIR']
    fig, axes = plt.subplots(len(reservoirs), len(cali_metrics),
                             figsize=(7 * len(cali_metrics), 3.5 * len(reservoirs)),
                             sharex=True, squeeze=False)
    for r_i, res in enumerate(reservoirs):
        for c_i, cm in enumerate(cali_metrics):
            ax = axes[r_i, c_i]
            sub = storage[(storage['reservoir'] == res) & (storage['cali_metric'] == cm)]
            for cfg in sorted(sub['config'].unique()):
                s = sub[sub['config'] == cfg].sort_values('date')
                color = CONFIG_COLORS.get(cfg, None)
                ax.plot(s['date'], s['mean_mm'], color=color, lw=1.0,
                        label=cfg.replace('glogem_subdaily_', ''))
                ax.fill_between(s['date'], s['mean_mm'] - s['std_mm'],
                                s['mean_mm'] + s['std_mm'], color=color, alpha=0.15)
            ax.set_title(f'{res} — calibrated on {cm}')
            ax.set_ylabel('storage [mm]')
            ax.grid(alpha=0.3)
            if r_i == 0 and c_i == 0:
                ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f"{storage['gauge_id'].iloc[0]} — subsurface storage "
                 f"(line = area-weighted mean, band = ±1σ across HRUs)", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_cross_catchment_heatmap(perf: pd.DataFrame, out_path: Path) -> None:
    """Heatmap: rows = catchments, cols = configs, value = validation eval_metric.

    Two-panel: KGE and KGE_WB validation scores for each cali_metric × config.
    Cells annotated with the value; values are evaluated on the validation
    period, taken from the run calibrated on the same metric (KGE-calibrated
    → KGE row, KGE_WB-calibrated → KGE_WB row).
    """
    if perf.empty:
        return
    # Use each metric's diagonal: cali_metric == eval_metric (the natural read).
    pairs = [('KGE', 'KGE'), ('KGE_WB', 'KGE_WB')]
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 0.4 * perf['gauge_id'].nunique() + 2.5),
                             squeeze=False)
    for ax, (cm, em) in zip(axes[0], pairs):
        sub = perf[(perf['cali_metric'] == cm) & (perf['eval_metric'] == em)]
        if sub.empty:
            ax.set_visible(False)
            continue
        pivot = sub.pivot_table(index='gauge_id', columns='config', values='value')
        im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([c.replace('glogem_subdaily_', '') for c in pivot.columns],
                           rotation=25, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f'{em} (calibrated on {cm})')
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            color='black' if v > 0.4 else 'white', fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle('SPHY options — validation scores across catchments', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ── Winners table ───────────────────────────────────────────────────────────

def winners_table(perf: pd.DataFrame) -> pd.DataFrame:
    """For each (gauge_id, cali_metric, eval_metric) find the best config.

    Drops NaN values before idxmax — some (catchment, eval_metric) groups
    can be all-NaN (e.g. KGE_nonwinter when a catchment has too few non-winter
    valid days), and idxmax returns NaN there, which then trips perf.loc.
    """
    if perf.empty:
        return pd.DataFrame()
    perf = perf.dropna(subset=['value'])
    if perf.empty:
        return pd.DataFrame()
    idx = perf.groupby(['gauge_id', 'cali_metric', 'eval_metric'])['value'].idxmax()
    return perf.loc[idx].reset_index(drop=True)


# ── CLI ─────────────────────────────────────────────────────────────────────

def discover_sphy_catchments() -> list[str]:
    """Find all SPHY-bearing catchment configs (those with _SPHY namelist)."""
    nlists = sorted(ROOT.glob('namelists/catchment_*_SPHY.yaml'))
    return [p.stem.split('_')[1] for p in nlists]


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--catchments', nargs='+', default=None,
                    help='Gauge IDs (default: all *_SPHY namelists)')
    ap.add_argument('--configs', nargs='+', default=None,
                    help='Configuration keys (default: auto-discover from each '
                         "catchment's namelist `configurations:` block)")
    ap.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS,
                    help=f'Calibration metrics (default: {DEFAULT_METRICS})')
    ap.add_argument('--env', default=None,
                    help='Environment layer (server/local). Default: autodetect')
    ap.add_argument('--output-dir', default=None,
                    help='Output dir (default: <main_dir>/cross_catchment_plots/sphy_option_comparison)')
    args = ap.parse_args(argv)

    catchments = args.catchments or discover_sphy_catchments()
    if not catchments:
        print('No catchments to analyze.', file=sys.stderr)
        return 1

    # Per-catchment config sets: if --configs given, use that for all; else
    # auto-discover from each namelist.
    if args.configs:
        per_catchment_configs = {gid: args.configs for gid in catchments}
    else:
        per_catchment_configs = {gid: configs_from_namelist(gid)
                                 for gid in catchments}

    all_configs_seen = sorted({c for cfgs in per_catchment_configs.values()
                               for c in cfgs})

    print(f"Catchments: {catchments}")
    print(f"Configs (union across catchments): {all_configs_seen}")
    print(f"Metrics:    {args.metrics}")

    # Resolve output dir from first catchment's main_dir if not given
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        first_cfg = per_catchment_configs[catchments[0]][0]
        first = build_nml(catchments[0], first_cfg, args.metrics[0], args.env)
        out_dir = Path(first['main_dir']) / 'cross_catchment_plots' / 'sphy_option_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'per_catchment').mkdir(exist_ok=True)
    print(f"Output:     {out_dir}\n")

    all_perf, all_regime, all_storage = [], [], []
    # Track region per catchment for forcing classification
    catchment_regions: Dict[str, Optional[str]] = {}

    for gid in catchments:
        cfgs = per_catchment_configs[gid]
        print(f"=== Catchment {gid} ({len(cfgs)} configs) ===")
        res = analyze_catchment(gid, cfgs, args.metrics, args.env)

        # Stash region from the first successfully-built namelist (cheap)
        try:
            sample_nml = build_nml(gid, cfgs[0], args.metrics[0], args.env)
            catchment_regions[gid] = sample_nml.get('region')
        except Exception:
            catchment_regions[gid] = None

        if not res['performance'].empty:
            all_perf.append(res['performance'])
        if not res['regime'].empty:
            all_regime.append(res['regime'])
        if not res['storage'].empty:
            all_storage.append(res['storage'])

        # Per-catchment plots
        if not res['performance'].empty:
            plot_catchment_performance(res['performance'],
                                       out_dir / 'per_catchment' / f'{gid}_performance.png')
            # Optional forcing comparison per catchment (Indus has both forcings)
            plot_forcing_comparison(res['performance'],
                                    catchment_regions[gid],
                                    out_dir / 'per_catchment' / f'{gid}_forcing.png')
        if not res['regime'].empty:
            plot_catchment_regime(res['regime'],
                                  out_dir / 'per_catchment' / f'{gid}_regime.png')
        if not res['storage'].empty:
            plot_catchment_storage(res['storage'],
                                   out_dir / 'per_catchment' / f'{gid}_storage.png')

    if not all_perf:
        print('No data loaded for any catchment.', file=sys.stderr)
        return 2

    perf = pd.concat(all_perf, ignore_index=True)

    # Annotate with the three classification axes for downstream slicing
    perf['region']      = perf['gauge_id'].map(catchment_regions)
    perf['forcing']     = perf.apply(
        lambda r: classify_config(r['config'], r['region'])['forcing'], axis=1)
    perf['perc_option'] = perf['config'].map(
        lambda c: classify_config(c)['perc_option'])
    perf['subsurface']  = perf['config'].map(
        lambda c: classify_config(c)['subsurface'])

    perf.to_csv(out_dir / 'performance_long.csv', index=False)
    perf.pivot_table(
        index=['gauge_id', 'config'], columns=['cali_metric', 'eval_metric'],
        values='value',
    ).round(3).to_csv(out_dir / 'performance_wide.csv')
    winners_table(perf).to_csv(out_dir / 'winners.csv', index=False)
    plot_cross_catchment_heatmap(perf, out_dir / 'heatmap_validation.png')

    # Cross-catchment forcing comparison if any catchment had ≥2 forcings
    has_multiple_forcings = (
        perf.groupby('gauge_id')['forcing'].nunique() >= 2
    ).any()
    if has_multiple_forcings:
        # Restrict to catchments that actually have multiple forcings
        multi = perf.groupby('gauge_id').filter(lambda g: g['forcing'].nunique() >= 2)
        plot_forcing_comparison(
            multi, region=None,
            out_path=out_dir / 'forcing_comparison.png',
        )

    if all_storage:
        storage = pd.concat(all_storage, ignore_index=True)
        summary = (storage
                   .groupby(['gauge_id', 'cali_metric', 'config', 'reservoir'])
                   .agg(mean_storage_mm=('mean_mm', 'mean'),
                        median_storage_mm=('mean_mm', 'median'),
                        cross_hru_std_mm=('std_mm', 'mean'))
                   .round(2).reset_index())
        summary.to_csv(out_dir / 'storage_summary.csv', index=False)

    print(f"\nDone. Wrote results to {out_dir}")
    print(' - performance_long.csv   (per row: catchment × config × cali × eval_metric')
    print('                           + region / forcing / perc_option / subsurface)')
    print(' - performance_wide.csv   (pivot, easy to eyeball)')
    print(' - winners.csv            (best config per catchment × metric pair)')
    print(' - storage_summary.csv    (mean storage + cross-HRU spread)')
    print(' - heatmap_validation.png (KGE / KGE_WB grid across catchments)')
    if has_multiple_forcings:
        print(' - forcing_comparison.png (ERA5 vs TPHiPr paired bars,')
        print('                           best-of-perc/shared per forcing)')
    print(' - per_catchment/<gid>_performance.png   (seasonal KGE breakdown)')
    print(' - per_catchment/<gid>_forcing.png       (forcing source comparison)')
    print(' - per_catchment/<gid>_regime.png        (monthly mean Q)')
    print(' - per_catchment/<gid>_storage.png       (subsurface storage)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
