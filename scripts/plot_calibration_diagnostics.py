"""Multi-objective calibration diagnostic plotter.

Auto-detects which objectives were active for a catchment namelist and
emits the matching plot panels into <output_dir>/plots_calibration/multiobj/.

Designed for the quick-look "did this calibration do what I asked?" check
after a SCEUA run — not a publication plot. One PNG per concern; all
panels are stand-alone so missing inputs (e.g. no sidecar log) just skip
that panel rather than crashing the whole run.

Usage
-----
    python scripts/plot_calibration_diagnostics.py namelists/catchment_2268_SPHY.yaml
    python scripts/plot_calibration_diagnostics.py namelists/2268.yaml \\
        --config glogem_subdaily_opt1 --metric KGE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from config_merge import load_config  # noqa: E402
from paths import get_paths  # noqa: E402


# ── Helpers ─────────────────────────────────────────────────────────────────

def _kge(sim: np.ndarray, obs: np.ndarray) -> float:
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]
    if len(sim) < 30 or obs.std() == 0 or obs.mean() == 0:
        return float('nan')
    r = float(np.corrcoef(sim, obs)[0, 1])
    return float(1 - np.sqrt(
        (r - 1) ** 2
        + (sim.std() / obs.std() - 1) ** 2
        + (sim.mean() / obs.mean() - 1) ** 2
    ))


def _nse(sim: np.ndarray, obs: np.ndarray) -> float:
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]
    if len(sim) < 30:
        return float('nan')
    denom = ((obs - obs.mean()) ** 2).sum()
    return float(1 - ((sim - obs) ** 2).sum() / denom) if denom > 0 else float('nan')


def _bias(sim: np.ndarray, obs: np.ndarray) -> float:
    mask = np.isfinite(sim) & np.isfinite(obs)
    if mask.sum() == 0:
        return float('nan')
    return float(100.0 * (sim[mask] - obs[mask]).sum() / obs[mask].sum())


def _load_hydrograph(output_dir: Path, gauge_id: str, model: str) -> Optional[pd.DataFrame]:
    f = output_dir / f"{gauge_id}_{model}_Hydrographs.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, skiprows=[1])
    df['date'] = pd.to_datetime(df['date'])
    sim_col = obs_col = None
    for c in df.columns:
        if c.endswith('(observed) [m3/s]'):
            obs_col = c
        elif c.endswith('[m3/s]') and 'observed' not in c.lower():
            sim_col = c
    if sim_col is None or obs_col is None:
        return None
    return pd.DataFrame({'date': df['date'],
                         'sim_Q': df[sim_col].astype(float),
                         'obs_Q': df[obs_col].astype(float)})


def _load_results(output_dir: Path) -> Optional[pd.DataFrame]:
    """Latest calibration_results_*.csv."""
    matches = sorted(output_dir.glob('calibration_results_*.csv'),
                     key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    return pd.read_csv(matches[-1])


# ── Plot panels ─────────────────────────────────────────────────────────────

def plot_convergence(results: pd.DataFrame, objectives: List[str], out: Path) -> None:
    """SCEUA score trajectory. Best-so-far line + per-iter scatter. If
    multi-obj, separate trace for each obj_* column."""
    if 'objective' not in results.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(results)
    iters = np.arange(1, n + 1)

    # Combined/objective trace
    ax.scatter(iters, results['objective'], s=8, alpha=0.35, color='black',
               label='per-iter combined')
    best = np.maximum.accumulate(results['objective'].values)
    ax.plot(iters, best, color='black', lw=2, label='best-so-far')

    # Per-objective columns
    colors = {'Q': '#1f77b4', 'snow': '#2ca02c', 'baseflow': '#d62728'}
    for o in objectives:
        col = f'obj_{o}'
        if col in results.columns:
            ax.scatter(iters, results[col], s=5, alpha=0.25,
                       color=colors.get(o, 'grey'), label=f'obj_{o}')
            ob = np.maximum.accumulate(results[col].fillna(-999).values)
            ax.plot(iters, ob, color=colors.get(o, 'grey'), lw=1.2, alpha=0.7)

    ax.set_xlabel('SCEUA iteration')
    ax.set_ylabel('score (higher = better)')
    ax.set_title(f"Convergence  ({n} iterations)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_hydrograph(hydro: pd.DataFrame, cali_end: str, end: str, out: Path) -> None:
    """Full cal+val hydrograph with sim/obs lines and shaded validation band."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(hydro['date'], hydro['obs_Q'], color='black', lw=0.8, label='observed')
    ax.plot(hydro['date'], hydro['sim_Q'], color='#1f77b4', lw=0.8, alpha=0.85,
            label='simulated')

    # Shade cal/val periods
    cali_dt = pd.to_datetime(cali_end)
    end_dt = pd.to_datetime(end)
    ax.axvspan(hydro['date'].min(), cali_dt, color='#ffeecc', alpha=0.35,
               label='calibration')
    ax.axvspan(cali_dt, end_dt, color='#cce6ff', alpha=0.35,
               label='validation')

    ax.set_ylabel('Q [m³/s]')
    ax.set_title('Hydrograph (full record, sim vs obs)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_regime(hydro: pd.DataFrame, cali_end: str, end: str, out: Path) -> None:
    """Monthly mean Q for cal vs val period, obs vs sim."""
    cali_dt = pd.to_datetime(cali_end)
    end_dt = pd.to_datetime(end)
    hydro = hydro.copy()
    hydro['month'] = hydro['date'].dt.month
    cal = hydro[hydro['date'] < cali_dt]
    val = hydro[(hydro['date'] >= cali_dt) & (hydro['date'] <= end_dt)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, period, df in zip(axes, ['calibration', 'validation'], [cal, val]):
        if df.empty:
            ax.set_visible(False)
            continue
        m = df.groupby('month').agg(obs=('obs_Q', 'mean'),
                                     sim=('sim_Q', 'mean')).reset_index()
        ax.plot(m['month'], m['obs'], 'k-', lw=2, label='observed')
        ax.plot(m['month'], m['sim'], '#1f77b4', lw=2, label='simulated')
        ax.set_title(f'{period} period')
        ax.set_xlabel('month')
        ax.set_xticks(range(1, 13))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel('Q [m³/s]')
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)


def metrics_summary(hydro: pd.DataFrame, cali_end: str, end: str, out: Path) -> None:
    """Plain-text summary of KGE/NSE/bias for cal + val."""
    cali_dt = pd.to_datetime(cali_end)
    end_dt = pd.to_datetime(end)
    lines = ["Calibration diagnostics (final-best run)", "=" * 50, ""]
    for label, mask in [
        ('calibration', hydro['date'] < cali_dt),
        ('validation',  (hydro['date'] >= cali_dt) & (hydro['date'] <= end_dt)),
    ]:
        sub = hydro[mask]
        if sub.empty:
            continue
        sim = sub['sim_Q'].to_numpy(float)
        obs = sub['obs_Q'].to_numpy(float)
        winter = sub['date'].dt.month.isin([11, 12, 1, 2, 3])
        sim_w, obs_w = sub.loc[winter, 'sim_Q'].to_numpy(float), sub.loc[winter, 'obs_Q'].to_numpy(float)
        sim_nw, obs_nw = sub.loc[~winter, 'sim_Q'].to_numpy(float), sub.loc[~winter, 'obs_Q'].to_numpy(float)
        lines += [
            f"-- {label} period ({sub['date'].min().date()} → {sub['date'].max().date()}, n={len(sub)}) --",
            f"  KGE             : {_kge(sim, obs):.4f}",
            f"  NSE             : {_nse(sim, obs):.4f}",
            f"  PBIAS           : {_bias(sim, obs):+.2f} %",
            f"  KGE_winter      : {_kge(sim_w, obs_w):.4f}    (Nov-Mar, n={int(winter.sum())})",
            f"  KGE_nonwinter   : {_kge(sim_nw, obs_nw):.4f}   (Apr-Oct, n={int((~winter).sum())})",
            "",
        ]
    out.write_text("\n".join(lines))


def plot_snow_per_band(obs_fsca_csv: Path, sim_output_dir: Path,
                       hru_areas: dict, hru_elevations: dict,
                       glacier_hrus: set, band_width_m: int,
                       cloud_threshold: float, min_pixels_per_band: int,
                       cali_end: str, end: str, out: Path) -> None:
    """Per-band fSCA: obs (MODIS, dots) vs sim (Raven, lines), 6 representative
    elevation bands."""
    from calibration_objectives import (load_modis_fsca_bands,
                                         load_raven_snow_frac_per_band)
    try:
        obs_b = load_modis_fsca_bands(obs_fsca_csv,
                                       cloud_threshold=cloud_threshold,
                                       min_pixels_per_band=min_pixels_per_band)
        sim_b = load_raven_snow_frac_per_band(
            sim_output_dir, hru_areas, hru_elevations,
            glacier_hrus=glacier_hrus, band_width_m=band_width_m,
        )
    except Exception as e:
        print(f"  skip snow_per_band: {e}")
        return

    common = sorted(set(obs_b.columns) & set(sim_b.columns))
    if not common:
        print("  skip snow_per_band: no overlapping bands")
        return

    # Pick up to 6 bands evenly spaced across the elevation range
    n = min(6, len(common))
    pick = [common[i] for i in np.linspace(0, len(common) - 1, n).astype(int)]

    fig, axes = plt.subplots(n, 1, figsize=(11, 1.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    end_dt = pd.to_datetime(end)
    cali_dt = pd.to_datetime(cali_end)
    for ax, b in zip(axes, pick):
        o = obs_b[b].dropna()
        s = sim_b[b]
        # Trim to namelist date range so the cal/val window is visible
        o = o.loc[:end_dt]
        s = s.loc[:end_dt]
        ax.plot(s.index, s.values, color='#1f77b4', lw=0.7,
                label='sim (daily)', alpha=0.85)
        ax.scatter(o.index, o.values, s=12, color='black',
                   label='obs MODIS', zorder=5)
        ax.axvline(cali_dt, color='grey', ls='--', lw=0.5)
        ax.set_ylabel(f'{b}m', rotation=0, ha='right', va='center',
                      labelpad=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_title('Per-band fSCA: observed (MODIS) vs simulated (Raven)')
    axes[-1].set_xlabel('date')
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_snowline(obs_fsca_csv: Path, sim_output_dir: Path,
                  hru_areas: dict, hru_elevations: dict,
                  glacier_hrus: set, band_width_m: int,
                  cloud_threshold: float, min_pixels_per_band: int,
                  cali_end: str, end: str, out: Path) -> None:
    """Snowline elevation time series: lowest band where fSCA > 0.5."""
    from calibration_objectives import (load_modis_fsca_bands,
                                         load_raven_snow_frac_per_band)
    try:
        obs_b = load_modis_fsca_bands(obs_fsca_csv,
                                       cloud_threshold=cloud_threshold,
                                       min_pixels_per_band=min_pixels_per_band)
        sim_b = load_raven_snow_frac_per_band(
            sim_output_dir, hru_areas, hru_elevations,
            glacier_hrus=glacier_hrus, band_width_m=band_width_m,
        )
    except Exception as e:
        print(f"  skip snowline: {e}")
        return

    def _snowline_per_t(df: pd.DataFrame) -> pd.Series:
        """For each row (date), find the lowest band whose fSCA >= 0.5."""
        bands = sorted(df.columns)
        bands_arr = np.array(bands)
        out = []
        for _, row in df.iterrows():
            vals = row[bands].values.astype(float)
            mask = vals >= 0.5
            out.append(bands_arr[mask].min() if mask.any() else np.nan)
        return pd.Series(out, index=df.index)

    end_dt = pd.to_datetime(end)
    obs_sl = _snowline_per_t(obs_b.loc[:end_dt])
    sim_sl = _snowline_per_t(sim_b.loc[:end_dt])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.scatter(obs_sl.index, obs_sl.values, s=14, color='black',
               label='obs MODIS', zorder=5)
    ax.plot(sim_sl.index, sim_sl.values, color='#1f77b4', lw=0.7,
            label='sim (daily)', alpha=0.85)
    ax.axvline(pd.to_datetime(cali_end), color='grey', ls='--', lw=0.6,
               label='cal/val boundary')
    ax.set_ylabel('snowline elevation [m]')
    ax.set_title('Lowest elevation band with fSCA ≥ 0.5')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_sidecar(sidecar_csv: Path, out: Path) -> None:
    """Per-iteration raw r / RMSE / bias for the snow objective. Faceted
    per band if band column present."""
    df = pd.read_csv(sidecar_csv)
    if df.empty:
        return

    # The sidecar appends one row per (iter × band) call. Reconstruct an
    # iteration counter by grouping consecutive rows per band.
    has_band = 'band' in df.columns and df['band'].nunique() > 1
    metrics = ['r', 'rmse', 'mae', 'pbias']
    metrics = [m for m in metrics if m in df.columns]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 2.2 * len(metrics)),
                             sharex=True, squeeze=False)

    if has_band:
        bands = sorted(df['band'].unique())
        n = min(6, len(bands))
        pick = [bands[i] for i in np.linspace(0, len(bands) - 1, n).astype(int)]
        for ax, m in zip(axes[:, 0], metrics):
            for b in pick:
                sub = df[df['band'] == b].reset_index(drop=True)
                sub['iter'] = np.arange(1, len(sub) + 1)
                ax.plot(sub['iter'], sub[m], lw=0.7, alpha=0.7, label=f'{b}m')
            ax.set_ylabel(m)
            ax.grid(alpha=0.3)
        axes[0, 0].legend(loc='upper right', fontsize=8, ncol=3)
    else:
        df = df.reset_index(drop=True)
        df['iter'] = np.arange(1, len(df) + 1)
        for ax, m in zip(axes[:, 0], metrics):
            ax.plot(df['iter'], df[m], lw=0.8, color='black')
            ax.set_ylabel(m)
            ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel('snow-objective call (chronological)')
    axes[0, 0].set_title('Sidecar diagnostics: raw r / RMSE / MAE / PBIAS '
                          'per iteration')
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_baseflow(hydro: pd.DataFrame, method: str, window,
                  cali_end: str, end: str, out: Path) -> None:
    """Eckhardt-separated baseflow: obs vs sim, in the chosen window."""
    from baseflow_separation import BaseflowSeparator
    obs = hydro.set_index('date')['obs_Q'].astype(float).dropna()
    sim = hydro.set_index('date')['sim_Q'].astype(float).dropna()

    if method == 'raw_winter':
        obs_bf, sim_bf = obs.copy(), sim.copy()
    else:
        obs_bf = getattr(BaseflowSeparator(obs), method)()
        sim_bf = getattr(BaseflowSeparator(sim), method)()

    # Resolve window to months (mirrors calibration_objectives._resolve_window)
    if isinstance(window, (list, tuple)):
        months = tuple(int(m) for m in window)
    elif window == 'winter':
        months = (11, 12, 1, 2, 3)
    elif window == 'raw_winter':
        months = (12, 1, 2, 3)
    elif window == 'all':
        months = tuple(range(1, 13))
    else:
        months = (11, 12, 1, 2, 3)

    cali_dt = pd.to_datetime(cali_end)
    end_dt = pd.to_datetime(end)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(obs.index, obs.values, color='lightgrey', lw=0.5, label='obs Q')
    ax.plot(obs_bf.index, obs_bf.values, color='black', lw=0.9,
            label=f'obs baseflow ({method})')
    ax.plot(sim_bf.index, sim_bf.values, color='#d62728', lw=0.9,
            label='sim baseflow')

    # Shade the window months
    yr_min, yr_max = obs.index.min().year, end_dt.year
    for y in range(yr_min, yr_max + 1):
        for m in months:
            start = pd.Timestamp(year=y, month=m, day=1)
            stop = start + pd.offsets.MonthEnd()
            if start > end_dt:
                continue
            ax.axvspan(start, stop, color='#cce6ff', alpha=0.12)

    ax.axvline(cali_dt, color='grey', ls='--', lw=0.6)
    ax.set_ylabel('Q [m³/s]')
    ax.set_title(f"Baseflow: {method}, window={window!r}  "
                 "(shaded = window months)")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ── Main dispatch ───────────────────────────────────────────────────────────

def run_for_config(nml: dict, config: str, metric: str, env: Optional[str]) -> None:
    from preprocess_modis_fsca import resolve_region  # noqa
    # Compose overrides from the orchestrator namelist (the file the user
    # actually runs run_full_pipeline against). Mirrors what
    # run_full_pipeline._run_multi_config does so the merged config sees
    # the user's calibration.objectives / weights / per-objective blocks.
    overrides = {'_calibration_metric': metric}
    for key in ('start_date', 'end_date', 'cali_end_date', 'warm_up_date',
                'display_name', 'gauge_id', 'warmup', 'precip_correction'):
        if key in nml:
            overrides[key] = nml[key]
    if 'calibration' in nml:
        overrides['calibration'] = nml['calibration']
    merged, _tmp = load_config(
        catchment=nml['catchment'], configuration=config,
        model=nml.get('models', ['HBV'])[0], env=env,
        overrides=overrides,
    )
    paths = get_paths(merged)
    output_dir = paths['output_dir']
    gauge_id = merged['gauge_id']
    model = merged.get('model_type', 'HBV')

    print(f"\n=== {gauge_id}/{config}/{metric} ===")
    print(f"output_dir: {output_dir}")

    plot_dir = output_dir / 'plots_calibration' / 'multiobj'
    plot_dir.mkdir(parents=True, exist_ok=True)

    cali_end = merged.get('cali_end_date')
    end = merged.get('end_date')

    cal_cfg = merged.get('calibration', {}) or {}
    objectives = cal_cfg.get('objectives', ['Q'])
    snow_cfg = cal_cfg.get('snow', {}) or {}
    bf_cfg = cal_cfg.get('baseflow', {}) or {}

    # ── Always-on panels ───────────────────────────────────────────────
    hydro = _load_hydrograph(output_dir, gauge_id, model)
    if hydro is not None:
        print("  • hydrograph.png")
        plot_hydrograph(hydro, cali_end, end, plot_dir / 'hydrograph.png')
        print("  • regime.png")
        plot_regime(hydro, cali_end, end, plot_dir / 'regime.png')
        print("  • metrics_summary.txt")
        metrics_summary(hydro, cali_end, end, plot_dir / 'metrics_summary.txt')
    else:
        print("  ⚠️ no Hydrographs.csv — skipping hydrograph/regime/metrics")

    results = _load_results(output_dir)
    if results is not None:
        print("  • convergence.png")
        plot_convergence(results, objectives, plot_dir / 'convergence.png')
    else:
        print("  ⚠️ no calibration_results CSV — skipping convergence")

    # ── snow panels ────────────────────────────────────────────────────
    if 'snow' in objectives:
        main_dir = Path(merged['main_dir'])
        product = snow_cfg.get('product', 'MOD10A2')
        obs_fsca = (main_dir / '01_data' / 'snow' / 'MODIS'
                    / 'basins' / gauge_id / f'fsca_{product}_{gauge_id}.csv')
        # HRU areas + elevations from the .rvh — use the comma-aware parser
        # defined below (mirrors spotpy_optimize._load_hru_info).
        rvh = paths['model_dir'] / f'{gauge_id}_{model}.rvh'
        hru_areas, hru_elev, glacier_hrus = _parse_rvh_hrus(rvh)

        if obs_fsca.exists() and hru_areas:
            print("  • snow_per_band.png")
            plot_snow_per_band(
                obs_fsca, output_dir, hru_areas, hru_elev, glacier_hrus,
                band_width_m=int(snow_cfg.get('band_width_m', 100)),
                cloud_threshold=float(snow_cfg.get('cloud_threshold', 0.5)),
                min_pixels_per_band=int(snow_cfg.get('min_pixels_per_band', 30)),
                cali_end=cali_end, end=end,
                out=plot_dir / 'snow_per_band.png',
            )
            print("  • snowline.png")
            plot_snowline(
                obs_fsca, output_dir, hru_areas, hru_elev, glacier_hrus,
                band_width_m=int(snow_cfg.get('band_width_m', 100)),
                cloud_threshold=float(snow_cfg.get('cloud_threshold', 0.5)),
                min_pixels_per_band=int(snow_cfg.get('min_pixels_per_band', 30)),
                cali_end=cali_end, end=end,
                out=plot_dir / 'snowline.png',
            )
        else:
            print(f"  ⚠️ skipping snow panels (obs CSV exists={obs_fsca.exists()}, "
                  f"hru count={len(hru_areas)})")

        # Sidecar trajectories
        sidecar = output_dir / 'snow_sidecar.csv'
        if sidecar.exists():
            print("  • sidecar.png")
            plot_sidecar(sidecar, plot_dir / 'sidecar.png')
        else:
            print(f"  ⚠️ no sidecar at {sidecar} — skipping")

    # ── baseflow panel ─────────────────────────────────────────────────
    if 'baseflow' in objectives and hydro is not None:
        method = bf_cfg.get('method', 'eckhardt')
        window = bf_cfg.get('window', 'winter')
        print("  • baseflow.png")
        try:
            plot_baseflow(hydro, method, window, cali_end, end,
                          plot_dir / 'baseflow.png')
        except Exception as e:
            print(f"  ⚠️ baseflow plot failed: {e}")

    print(f"  → {plot_dir}")


def _parse_rvh_hrus(rvh: Path) -> Tuple[dict, dict, set]:
    """Comma-aware parser matching spotpy_optimize._load_hru_info."""
    hru_areas: dict = {}
    hru_elev: dict = {}
    glaciers: set = set()
    if not rvh.exists():
        return hru_areas, hru_elev, glaciers
    in_block = False
    for raw in rvh.read_text().splitlines():
        line = raw.strip()
        if line.startswith(':HRUs'):
            in_block = True
            continue
        if line.startswith(':EndHRUs'):
            in_block = False
            continue
        if not in_block or not line or line.startswith(':') or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 7:
            continue
        try:
            hru_id = int(parts[0])
            area = float(parts[1])
            elev = float(parts[2])
        except ValueError:
            continue
        hru_areas[hru_id] = area
        hru_elev[hru_id] = elev
        lu = parts[6].upper() if len(parts) > 6 else ''
        if 'GLACIER' in lu or 'ICE' in lu:
            glaciers.add(hru_id)
    return hru_areas, hru_elev, glaciers


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('namelist', help='Path to a catchment namelist (multi-config or single)')
    ap.add_argument('--config', default=None,
                    help='Restrict to a specific configuration key '
                         '(default: all in the namelist)')
    ap.add_argument('--metric', default=None,
                    help='Restrict to a specific calibration metric '
                         '(default: KGE or what is listed)')
    ap.add_argument('--env', default=None,
                    help='Environment layer (server/local). Default: autodetect')
    args = ap.parse_args(argv)

    with open(args.namelist) as f:
        nml = yaml.safe_load(f)

    configs = [args.config] if args.config else nml.get('configurations', [None])
    metrics = [args.metric] if args.metric else (
        nml.get('calibration', {}).get('metrics', ['KGE'])
    )
    if isinstance(metrics, str):
        metrics = [metrics]

    for config in configs:
        for metric in metrics:
            run_for_config(nml, config, metric, args.env)

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
