# Compare calibration metrics for the same model+config combination
# April 2026

#--------------------------------------------------------------------------------
################################## packages #####################################
#--------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import List, Dict, Optional
import yaml

from paths import get_paths
import postprocessing

#--------------------------------------------------------------------------------
################################## constants ####################################
#--------------------------------------------------------------------------------

KNOWN_METRICS = ['KGE', 'LogKGE', 'KGE_WB', 'KGE_lowFDC']

METRIC_COLORS = {
    'KGE': '#1b9e77',
    'LogKGE': '#d95f02',
    'KGE_WB': '#7570b3',
    'KGE_lowFDC': '#e7298a',
}

METRIC_DISPLAY_NAMES = {
    'KGE': 'KGE',
    'LogKGE': 'Log-KGE',
    'KGE_WB': 'KGE-WB (winter)',
    'KGE_lowFDC': 'KGE-lowFDC',
}

#--------------------------------------------------------------------------------
################################## discovery ####################################
#--------------------------------------------------------------------------------

def discover_available_metrics(gauge_id, config_key, model_type, main_dir):
    """Scan filesystem for which metric-calibrated runs have completed output.

    Returns list of metric strings (e.g. ['KGE', 'LogKGE', 'KGE_lowFDC']).
    """
    config_dir = Path(main_dir) / 'model_runs' / f'catchment_{gauge_id}' / 'configs' / config_key
    available = []
    for metric in KNOWN_METRICS:
        dir_name = model_type if metric == 'KGE' else f"{model_type}_{metric}"
        hydro_file = config_dir / dir_name / 'output' / f"{gauge_id}_{model_type}_Hydrographs.csv"
        if hydro_file.exists():
            available.append(metric)
    return available


def _build_metric_config(base_config, metric):
    """Build a single-run config dict for a specific metric."""
    cfg = dict(base_config)
    cfg['_calibration_metric'] = metric
    return cfg

#--------------------------------------------------------------------------------
################################## setup ########################################
#--------------------------------------------------------------------------------

def setup_metric_comparison_directories(gauge_id, config_key, main_dir):
    """Create output directories for metric comparison plots."""
    catchment_dir = Path(main_dir) / 'model_runs' / f'catchment_{gauge_id}'
    base_dir = catchment_dir / 'metric_comparisons' / config_key

    plot_dirs = {
        'hydrographs': base_dir / 'hydrographs',
        'performance': base_dir / 'performance',
        'flow_duration': base_dir / 'flow_duration',
        'parameters': base_dir / 'parameters',
        'water_balance': base_dir / 'water_balance',
        'storage': base_dir / 'storage',
    }

    for d in plot_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return plot_dirs

#--------------------------------------------------------------------------------
################################## data loading #################################
#--------------------------------------------------------------------------------

def load_all_metric_data(base_config, metrics):
    """Load hydrograph data for each metric.

    Returns dict {metric_name: DataFrame} using postprocessing.load_hydrograph_data.
    """
    all_data = {}
    for metric in metrics:
        cfg = _build_metric_config(base_config, metric)
        data = postprocessing.load_hydrograph_data(cfg)
        if data is not None:
            all_data[metric] = data
        else:
            print(f"  Warning: No data for metric {metric}")
    return all_data

#--------------------------------------------------------------------------------
################################## plot functions ################################
#--------------------------------------------------------------------------------

def plot_metric_hydrograph_regime(base_config, metrics, all_data, plot_dirs,
                                  validation_start, validation_end, unit='mm'):
    """Overlay monthly regime for each metric calibration."""
    gauge_id = base_config['gauge_id']
    model_type = base_config['model_type']
    config_key = base_config.get('_config_key', '')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get catchment area for mm conversion
    catchment_area = None
    if unit == 'mm':
        cfg0 = _build_metric_config(base_config, metrics[0])
        paths = get_paths(cfg0)
        try:
            rvh_files = list(paths['model_dir'].glob(f'*{model_type}.rvh'))
            if rvh_files:
                with open(rvh_files[0]) as f:
                    for line in f:
                        if 'AREA' in line.upper() and 'TOTAL' not in line.upper():
                            parts = line.split(',')
                            for p in parts:
                                try:
                                    val = float(p.strip())
                                    if val > 10:
                                        catchment_area = val
                                        break
                                except ValueError:
                                    continue
        except Exception:
            pass

    for period_idx, (period_name, start, end) in enumerate([
        ('Calibration', base_config.get('start_date'), base_config.get('cali_end_date')),
        ('Validation', base_config.get('cali_end_date'), validation_end),
    ]):
        ax = axes[period_idx]
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        # Plot observed (same for all metrics)
        first_data = list(all_data.values())[0]
        mask = (first_data['date'] >= start_dt) & (first_data['date'] <= end_dt)
        obs_period = first_data[mask].copy()
        obs_period['month'] = obs_period['date'].dt.month
        obs_monthly = obs_period.groupby('month')['obs_Q'].mean()

        if unit == 'mm' and catchment_area:
            obs_monthly = obs_monthly * 86.4 / catchment_area

        ax.plot(obs_monthly.index, obs_monthly.values, 'k-', linewidth=2,
                label='Observed', zorder=10)

        # Plot each metric
        for metric in metrics:
            if metric not in all_data:
                continue
            data = all_data[metric]
            mask = (data['date'] >= start_dt) & (data['date'] <= end_dt)
            sim_period = data[mask].copy()
            sim_period['month'] = sim_period['date'].dt.month
            sim_monthly = sim_period.groupby('month')['sim_Q'].mean()

            if unit == 'mm' and catchment_area:
                sim_monthly = sim_monthly * 86.4 / catchment_area

            color = METRIC_COLORS.get(metric, '#333333')
            label = METRIC_DISPLAY_NAMES.get(metric, metric)
            ax.plot(sim_monthly.index, sim_monthly.values, '-', color=color,
                    linewidth=1.5, label=label)

        ax.set_xlabel('Month')
        ylabel = 'Discharge [mm/day]' if unit == 'mm' else 'Discharge [m$^3$/s]'
        ax.set_ylabel(ylabel)
        ax.set_title(f'{period_name}')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Hydrological Regime — {gauge_id} {model_type} ({config_key})', fontsize=12)
    plt.tight_layout()
    save_path = plot_dirs['hydrographs'] / f'regime_metric_comparison_{gauge_id}_{model_type}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_metric_flow_duration_curve(base_config, metrics, all_data, plot_dirs,
                                     validation_start, validation_end):
    """FDC comparison — key plot for evaluating low-flow metric calibrations."""
    gauge_id = base_config['gauge_id']
    model_type = base_config['model_type']
    config_key = base_config.get('_config_key', '')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for period_idx, (period_name, start, end) in enumerate([
        ('Calibration', base_config.get('start_date'), base_config.get('cali_end_date')),
        ('Validation', base_config.get('cali_end_date'), validation_end),
    ]):
        ax = axes[period_idx]
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        # Observed FDC
        first_data = list(all_data.values())[0]
        mask = (first_data['date'] >= start_dt) & (first_data['date'] <= end_dt)
        obs = first_data.loc[mask, 'obs_Q'].dropna().values
        obs_sorted = np.sort(obs)[::-1]
        obs_exceed = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted) * 100
        ax.plot(obs_exceed, obs_sorted, 'k-', linewidth=2, label='Observed', zorder=10)

        # Simulated FDC per metric
        for metric in metrics:
            if metric not in all_data:
                continue
            data = all_data[metric]
            mask = (data['date'] >= start_dt) & (data['date'] <= end_dt)
            # Align sim with valid obs dates
            sim_obs = data.loc[mask, ['sim_Q', 'obs_Q']].dropna()
            sim = sim_obs['sim_Q'].values
            sim_sorted = np.sort(sim)[::-1]
            sim_exceed = np.arange(1, len(sim_sorted) + 1) / len(sim_sorted) * 100

            color = METRIC_COLORS.get(metric, '#333333')
            label = METRIC_DISPLAY_NAMES.get(metric, metric)
            ax.plot(sim_exceed, sim_sorted, '-', color=color, linewidth=1.5, label=label)

        ax.set_xlabel('Exceedance Probability [%]')
        ax.set_ylabel('Discharge [m$^3$/s]')
        ax.set_title(f'{period_name}')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Flow Duration Curve — {gauge_id} {model_type} ({config_key})', fontsize=12)
    plt.tight_layout()
    save_path = plot_dirs['flow_duration'] / f'fdc_metric_comparison_{gauge_id}_{model_type}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_metric_low_flow_comparison(base_config, metrics, all_data, plot_dirs,
                                     validation_start, validation_end):
    """Winter baseflow zoom — Nov-Mar hydrograph comparison."""
    gauge_id = base_config['gauge_id']
    model_type = base_config['model_type']
    config_key = base_config.get('_config_key', '')

    vali_start_dt = pd.to_datetime(base_config.get('cali_end_date'))
    vali_end_dt = pd.to_datetime(validation_end)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Helper: insert NaN rows at gaps >1 day so matplotlib doesn't draw
    # straight lines across the summer months
    def _break_at_gaps(dates, values):
        df_tmp = pd.DataFrame({'date': dates, 'val': values}).sort_values('date')
        gaps = df_tmp['date'].diff() > pd.Timedelta(days=2)
        if not gaps.any():
            return df_tmp['date'].values, df_tmp['val'].values
        # Insert NaN rows at each gap
        pieces = []
        for chunk in np.split(df_tmp, np.where(gaps)[0]):
            pieces.append(chunk)
            pieces.append(pd.DataFrame({'date': [chunk['date'].iloc[-1] + pd.Timedelta(days=1)],
                                        'val': [np.nan]}))
        result = pd.concat(pieces[:-1], ignore_index=True)
        return result['date'].values, result['val'].values

    # Plot observed winter flows
    first_data = list(all_data.values())[0]
    mask = (first_data['date'] >= vali_start_dt) & (first_data['date'] <= vali_end_dt)
    vali_data = first_data[mask].copy()
    winter_mask = vali_data['date'].dt.month.isin([11, 12, 1, 2, 3])
    obs_winter = vali_data[winter_mask]
    obs_dates, obs_vals = _break_at_gaps(obs_winter['date'].values, obs_winter['obs_Q'].values)
    ax.plot(obs_dates, obs_vals, 'k-', linewidth=1.5,
            label='Observed', alpha=0.8)

    # Plot each metric's winter simulation
    for metric in metrics:
        if metric not in all_data:
            continue
        data = all_data[metric]
        mask = (data['date'] >= vali_start_dt) & (data['date'] <= vali_end_dt)
        vali = data[mask].copy()
        winter = vali[vali['date'].dt.month.isin([11, 12, 1, 2, 3])]

        sim_dates, sim_vals = _break_at_gaps(winter['date'].values, winter['sim_Q'].values)
        color = METRIC_COLORS.get(metric, '#333333')
        label = METRIC_DISPLAY_NAMES.get(metric, metric)
        ax.plot(sim_dates, sim_vals, '-', color=color,
                linewidth=1, label=label, alpha=0.8)

    ax.set_xlabel('Date')
    ax.set_ylabel('Discharge [m$^3$/s]')
    ax.set_title(f'Winter Baseflow (Nov-Mar) — {gauge_id} {model_type} ({config_key})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = plot_dirs['hydrographs'] / f'winter_baseflow_comparison_{gauge_id}_{model_type}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_metric_performance_table(base_config, metrics, all_data, plot_dirs,
                                   validation_start, validation_end):
    """Bar chart comparing diagnostic metrics for each calibration metric."""
    gauge_id = base_config['gauge_id']
    model_type = base_config['model_type']
    config_key = base_config.get('_config_key', '')

    vali_start_dt = pd.to_datetime(base_config.get('cali_end_date'))
    vali_end_dt = pd.to_datetime(validation_end)

    # Compute performance for each metric
    perf_metrics = ['NSE', 'KGE', 'KGE_NP']
    results = {}
    for metric in metrics:
        if metric not in all_data:
            continue
        data = all_data[metric]
        m = postprocessing.calculate_performance_metrics(
            data, vali_start_dt, vali_end_dt, f"Validation ({metric})"
        )
        if m:
            results[metric] = m

    if not results:
        print("  No valid performance results to plot")
        return

    # Bar chart
    n_metrics = len(results)
    n_perf = len(perf_metrics)
    x = np.arange(n_perf)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, m) in enumerate(results.items()):
        values = [m.get(pm, np.nan) for pm in perf_metrics]
        color = METRIC_COLORS.get(metric, '#333333')
        label = METRIC_DISPLAY_NAMES.get(metric, metric)
        bars = ax.bar(x + i * width - 0.4 + width/2, values, width,
                      color=color, label=label, alpha=0.8)
        # Add value labels
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(perf_metrics)
    ax.set_ylabel('Metric Value')
    ax.set_title(f'Validation Performance — {gauge_id} {model_type} ({config_key})')
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_path = plot_dirs['performance'] / f'performance_comparison_{gauge_id}_{model_type}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_metric_parameter_comparison(base_config, metrics, plot_dirs):
    """Compare optimized parameters across calibration metrics."""
    gauge_id = base_config['gauge_id']
    model_type = base_config['model_type']
    config_key = base_config.get('_config_key', '')

    # Load best params for each metric
    all_params = {}
    for metric in metrics:
        cfg = _build_metric_config(base_config, metric)
        paths = get_paths(cfg)
        verified = paths['output_dir'] / f"{gauge_id}_{model_type}_VERIFIED_best_params.csv"
        best = paths['output_dir'] / f"{gauge_id}_{model_type}_best_params.csv"
        param_file = verified if verified.exists() else best
        if param_file.exists():
            try:
                df = pd.read_csv(param_file)
                # Drop non-parameter columns
                param_cols = [c for c in df.columns if c not in
                              ['objective', 'obj_function', 'timestamp', 'validation_obj', 'iteration']]
                all_params[metric] = df[param_cols].iloc[0]
            except Exception as e:
                print(f"  Warning: Could not load params for {metric}: {e}")

    if len(all_params) < 2:
        print("  Need at least 2 metric runs to compare parameters")
        return

    # Build comparison dataframe
    param_df = pd.DataFrame(all_params)
    param_names = param_df.index.tolist()

    # Normalize to [0,1] range for comparison
    param_norm = param_df.copy()
    for pname in param_names:
        pmin = param_df.loc[pname].min()
        pmax = param_df.loc[pname].max()
        if pmax > pmin:
            param_norm.loc[pname] = (param_df.loc[pname] - pmin) / (pmax - pmin)
        else:
            param_norm.loc[pname] = 0.5

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(all_params) * 2), max(6, len(param_names) * 0.4)))
    display_names = [METRIC_DISPLAY_NAMES.get(m, m) for m in param_norm.columns]

    im = ax.imshow(param_norm.values, aspect='auto', cmap='viridis')

    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_names, fontsize=8)

    # Add actual values as text
    for i in range(len(param_names)):
        for j in range(len(all_params)):
            metric = list(all_params.keys())[j]
            val = param_df.iloc[i, j]
            ax.text(j, i, f'{val:.3g}', ha='center', va='center', fontsize=6,
                    color='white' if param_norm.iloc[i, j] < 0.5 else 'black')

    ax.set_title(f'Parameter Comparison — {gauge_id} {model_type} ({config_key})')
    plt.colorbar(im, ax=ax, label='Normalized value', shrink=0.8)
    plt.tight_layout()
    save_path = plot_dirs['parameters'] / f'parameter_comparison_{gauge_id}_{model_type}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_metric_parameter_boxplots(base_config, metrics, plot_dirs, top_n=100):
    """Boxplots of top-N parameter sets for each calibration metric, side by side.

    Mirrors plot_parameter_boxplots_comparison from postprocessing_configurations
    but compares across calibration metrics instead of configurations.
    """
    gauge_id = base_config['gauge_id']
    model_type = base_config['model_type']
    config_key = base_config.get('_config_key', '')

    # Load top-N parameter sets per metric
    metric_results = {}
    all_param_names = set()

    for metric in metrics:
        cfg = _build_metric_config(base_config, metric)
        try:
            param_data = postprocessing.load_parameter_values(cfg, top_n)
            if param_data is None:
                print(f"  Warning: No parameter data for {metric}")
                continue
            metric_results[metric] = param_data
            all_param_names.update(param_data['parameters'].keys())
            print(f"  {metric}: loaded {param_data['n_sets']} sets, "
                  f"{len(param_data['parameters'])} params")
        except Exception as e:
            print(f"  Warning: Could not load params for {metric}: {e}")

    if len(metric_results) < 2:
        print("  Need at least 2 metrics with parameter data to compare")
        return

    param_names = sorted(all_param_names)
    n_params = len(param_names)

    n_cols = int(np.ceil(np.sqrt(n_params)))
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_params == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()

    available_metrics = list(metric_results.keys())

    for i, param_name in enumerate(param_names):
        ax = axes_flat[i]

        plot_data = []
        labels = []
        colors = []

        for metric in available_metrics:
            params = metric_results[metric]['parameters']
            values = params.get(param_name, [])
            plot_data.append(values)
            labels.append(METRIC_DISPLAY_NAMES.get(metric, metric))
            colors.append(METRIC_COLORS.get(metric, '#333333'))

        if not any(len(d) > 0 for d in plot_data):
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12)
            ax.set_title(param_name.replace(f"{model_type}_", ""),
                         fontsize=16, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')

        display_name = param_name.replace(f"{model_type}_", "")
        ax.set_title(display_name, fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax.set_ylabel('Parameter Value', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=13)
        if len(labels) > 2:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=13)
        else:
            plt.setp(ax.get_xticklabels(), fontsize=13)

    for i in range(n_params, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    save_path = plot_dirs['parameters'] / f'parameter_boxplots_metric_comparison_{gauge_id}_{model_type}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_metric_soil_storage(base_config, metrics, plot_dirs,
                             validation_start, validation_end):
    """Compare Soil Water[1] and Soil Water[2] timeseries across calibration metrics."""
    gauge_id = base_config['gauge_id']
    model_type = base_config['model_type']
    config_key = base_config.get('_config_key', '')

    soil_cols = ['Soil Water[1] [mm]', 'Soil Water[2] [mm]']
    soil_labels = ['SOIL[1] (fast reservoir)', 'SOIL[2] (slow reservoir)']

    # Load storage data per metric
    all_storage = {}
    for metric in metrics:
        cfg = _build_metric_config(base_config, metric)
        storage = postprocessing.load_storage_data(cfg)
        if storage is not None:
            all_storage[metric] = storage

    if len(all_storage) < 2:
        print("  Need at least 2 metrics with storage data")
        return

    start_dt = pd.to_datetime(base_config.get('start_date'))
    end_dt = pd.to_datetime(validation_end)

    fig, axes = plt.subplots(len(soil_cols), 1, figsize=(14, 4 * len(soil_cols)), sharex=True)
    if len(soil_cols) == 1:
        axes = [axes]

    for ax, col, label in zip(axes, soil_cols, soil_labels):
        for metric in metrics:
            if metric not in all_storage:
                continue
            df = all_storage[metric]
            if col not in df.columns:
                continue
            mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
            sub = df[mask]
            color = METRIC_COLORS.get(metric, '#333333')
            ax.plot(sub['date'], sub[col], color=color, linewidth=0.8,
                    label=METRIC_DISPLAY_NAMES.get(metric, metric), alpha=0.8)

        # Vertical line at calibration / validation split
        cali_end = pd.to_datetime(base_config.get('cali_end_date'))
        if cali_end:
            ax.axvline(cali_end, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)

        ax.set_ylabel('Storage [mm]')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    fig.suptitle(f'Soil Storage — {gauge_id} {model_type} ({config_key})', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = plot_dirs['storage'] / f'soil_storage_metric_comparison_{gauge_id}_{model_type}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def _extract_recession_segments(dates, Q, min_length=5, winter_only=True):
    """Extract monotonically decreasing flow segments.

    Parameters
    ----------
    dates : array-like of datetime
    Q : array-like of float
    min_length : int
        Minimum number of consecutive decreasing days.
    winter_only : bool
        If True, only keep segments that fall entirely in Nov-Mar.

    Returns list of dicts with keys 'dates', 'Q', 'Q_norm' (Q/Q0), 'k' (recession constant).
    """
    dates = pd.to_datetime(dates)
    Q = np.asarray(Q, dtype=float)

    # Remove NaN
    valid = ~np.isnan(Q)
    dates = dates[valid]
    Q = Q[valid]

    if len(Q) < min_length:
        return []

    # Winter filter
    if winter_only:
        winter = np.isin(dates.month, [11, 12, 1, 2, 3])
        dates = dates[winter]
        Q = Q[winter]

    # Find decreasing runs
    decreasing = np.diff(Q) < 0
    segments = []
    start = None

    for i, dec in enumerate(decreasing):
        if dec:
            if start is None:
                start = i
        else:
            if start is not None:
                length = i - start + 1
                if length >= min_length:
                    seg_dates = dates[start:i + 1]
                    seg_Q = Q[start:i + 1]
                    # Check for continuous dates (no summer gaps)
                    if (seg_dates[-1] - seg_dates[0]).days <= length + 2:
                        segments.append({'dates': seg_dates, 'Q': seg_Q})
                start = None

    # Handle final segment
    if start is not None:
        length = len(decreasing) - start + 1
        if length >= min_length:
            seg_dates = dates[start:]
            seg_Q = Q[start:]
            if (seg_dates[-1] - seg_dates[0]).days <= length + 2:
                segments.append({'dates': seg_dates, 'Q': seg_Q})

    # Normalize and fit recession constant for each segment
    for seg in segments:
        Q0 = seg['Q'][0]
        seg['Q_norm'] = seg['Q'] / Q0 if Q0 > 0 else seg['Q'] * 0
        t = np.arange(len(seg['Q']), dtype=float)
        # Fit k: Q/Q0 = exp(-k*t) → log(Q/Q0) = -k*t
        with np.errstate(divide='ignore', invalid='ignore'):
            log_Qn = np.log(seg['Q_norm'])
        valid_fit = np.isfinite(log_Qn) & (seg['Q_norm'] > 0)
        if np.sum(valid_fit) >= 3:
            # Simple least-squares: k = -slope of log(Q/Q0) vs t
            slope, _ = np.polyfit(t[valid_fit], log_Qn[valid_fit], 1)
            seg['k'] = -slope  # positive k means decay
        else:
            seg['k'] = np.nan

    return segments


def plot_metric_recession_analysis(base_config, metrics, all_data, plot_dirs,
                                   validation_start, validation_end):
    """Recession analysis comparing baseflow behavior across calibration metrics.

    Left panel: normalized recession curves (Q/Q0 vs days) overlaid.
    Right panel: boxplots of fitted recession constants per metric.
    """
    gauge_id = base_config['gauge_id']
    model_type = base_config['model_type']
    config_key = base_config.get('_config_key', '')

    vali_start_dt = pd.to_datetime(base_config.get('cali_end_date'))
    vali_end_dt = pd.to_datetime(validation_end)

    # Extract recession segments for observed and each metric
    first_data = list(all_data.values())[0]
    mask = (first_data['date'] >= vali_start_dt) & (first_data['date'] <= vali_end_dt)
    vali_obs = first_data[mask]
    obs_segments = _extract_recession_segments(
        vali_obs['date'].values, vali_obs['obs_Q'].values
    )

    metric_segments = {}
    for metric in metrics:
        if metric not in all_data:
            continue
        data = all_data[metric]
        mask = (data['date'] >= vali_start_dt) & (data['date'] <= vali_end_dt)
        vali = data[mask]
        segs = _extract_recession_segments(
            vali['date'].values, vali['sim_Q'].values
        )
        if segs:
            metric_segments[metric] = segs

    if not obs_segments and not metric_segments:
        print("  No recession segments found (validation period may be too short)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left panel: normalized recession overlay ---
    ax = axes[0]

    # Plot observed segments
    for seg in obs_segments:
        t = np.arange(len(seg['Q_norm']))
        ax.plot(t, seg['Q_norm'], 'k-', linewidth=0.6, alpha=0.4)
    # Dummy for legend
    if obs_segments:
        ax.plot([], [], 'k-', linewidth=1.5, label='Observed')

    # Plot each metric's segments
    for metric in metrics:
        if metric not in metric_segments:
            continue
        color = METRIC_COLORS.get(metric, '#333333')
        label = METRIC_DISPLAY_NAMES.get(metric, metric)
        for seg in metric_segments[metric]:
            t = np.arange(len(seg['Q_norm']))
            ax.plot(t, seg['Q_norm'], '-', color=color, linewidth=0.6, alpha=0.4)
        # Dummy for legend
        ax.plot([], [], '-', color=color, linewidth=1.5, label=label)

    # Reference exponential curves
    t_ref = np.arange(30)
    for k_ref, ls in [(0.02, ':'), (0.05, '--'), (0.10, ':')]:
        ax.plot(t_ref, np.exp(-k_ref * t_ref), color='grey', linestyle=ls,
                linewidth=0.8, alpha=0.5)
        ax.text(t_ref[-1] + 0.5, np.exp(-k_ref * t_ref[-1]),
                f'k={k_ref}', fontsize=7, color='grey', va='center')

    ax.set_xlabel('Days since recession start')
    ax.set_ylabel('Q / Q$_0$')
    ax.set_title('Normalized Recession Curves (Nov-Mar)')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Right panel: recession constant boxplots ---
    ax = axes[1]

    box_data = []
    box_labels = []
    box_colors = []

    # Observed
    obs_k = [s['k'] for s in obs_segments if np.isfinite(s['k'])]
    if obs_k:
        box_data.append(obs_k)
        box_labels.append('Observed')
        box_colors.append('black')

    # Each metric
    for metric in metrics:
        if metric not in metric_segments:
            continue
        k_vals = [s['k'] for s in metric_segments[metric] if np.isfinite(s['k'])]
        if k_vals:
            box_data.append(k_vals)
            box_labels.append(METRIC_DISPLAY_NAMES.get(metric, metric))
            box_colors.append(METRIC_COLORS.get(metric, '#333333'))

    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')

        # Add individual points
        for i, (data, color) in enumerate(zip(box_data, box_colors)):
            x = np.random.normal(i + 1, 0.04, size=len(data))
            ax.scatter(x, data, color=color, s=15, alpha=0.5, zorder=5,
                       edgecolors='none')

        # Add count labels
        for i, data in enumerate(box_data):
            ax.text(i + 1, ax.get_ylim()[1] * 0.95, f'n={len(data)}',
                    ha='center', fontsize=8, color='grey')

    ax.set_ylabel('Recession constant k [1/day]')
    ax.set_title('Recession Constants')
    ax.grid(True, alpha=0.3, axis='y')
    if len(box_labels) > 3:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    fig.suptitle(f'Recession Analysis — {gauge_id} {model_type} ({config_key})', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = plot_dirs['storage'] / f'recession_analysis_{gauge_id}_{model_type}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_metric_hydrograph_timeseries(base_config, metrics, all_data, plot_dirs,
                                       validation_start, validation_end):
    """Full hydrograph timeseries overlay for validation period."""
    gauge_id = base_config['gauge_id']
    model_type = base_config['model_type']
    config_key = base_config.get('_config_key', '')

    vali_start_dt = pd.to_datetime(base_config.get('cali_end_date'))
    vali_end_dt = pd.to_datetime(validation_end)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Observed
    first_data = list(all_data.values())[0]
    mask = (first_data['date'] >= vali_start_dt) & (first_data['date'] <= vali_end_dt)
    vali = first_data[mask]
    ax.plot(vali['date'], vali['obs_Q'], 'k-', linewidth=1, label='Observed', alpha=0.7)

    # Each metric
    for metric in metrics:
        if metric not in all_data:
            continue
        data = all_data[metric]
        mask = (data['date'] >= vali_start_dt) & (data['date'] <= vali_end_dt)
        vali = data[mask]
        color = METRIC_COLORS.get(metric, '#333333')
        label = METRIC_DISPLAY_NAMES.get(metric, metric)
        ax.plot(vali['date'], vali['sim_Q'], '-', color=color,
                linewidth=0.8, label=label, alpha=0.7)

    ax.set_xlabel('Date')
    ax.set_ylabel('Discharge [m$^3$/s]')
    ax.set_title(f'Validation Hydrograph — {gauge_id} {model_type} ({config_key})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = plot_dirs['hydrographs'] / f'hydrograph_timeseries_comparison_{gauge_id}_{model_type}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


#--------------------------------------------------------------------------------
################################## orchestrator #################################
#--------------------------------------------------------------------------------

def run_complete_metric_postprocessing(gauge_id, config_key, model_type='HBV',
                                        main_dir=None, metrics=None,
                                        validation_start=None, validation_end=None,
                                        env=None, skip_errors=True):
    """Run all metric comparison plots for a given model+config.

    Parameters
    ----------
    gauge_id : str
    config_key : str
        Configuration name (e.g. 'subdaily', 'glogem_subdaily')
    model_type : str
    main_dir : str or Path
    metrics : list, optional
        If None, auto-discovers available metrics
    validation_start, validation_end : str, optional
    env : str, optional
    skip_errors : bool
    """
    print(f"\n{'='*60}")
    print(f"METRIC COMPARISON: {gauge_id} — {model_type} — {config_key}")
    print(f"{'='*60}")

    # Resolve main_dir
    if main_dir is None:
        from config_merge import load_config
        nml, tmp = load_config(gauge_id, config_key, model_type, env=env)
        main_dir = nml['main_dir']
        Path(tmp).unlink(missing_ok=True)

    # Discover metrics
    if metrics is None:
        metrics = discover_available_metrics(gauge_id, config_key, model_type, main_dir)
    print(f"  Available metrics: {metrics}")

    if len(metrics) < 2:
        print(f"  Need at least 2 metrics to compare. Skipping.")
        return

    # Build base config (no metric set — individual functions will set it)
    base_config = {
        'main_dir': str(main_dir),
        'gauge_id': gauge_id,
        'model_type': model_type,
        '_config_key': config_key,
    }

    # Load dates from first available metric config
    from config_merge import load_config
    nml, tmp = load_config(gauge_id, config_key, model_type, env=env)
    base_config['start_date'] = nml.get('start_date')
    base_config['end_date'] = nml.get('end_date')
    base_config['cali_end_date'] = nml.get('cali_end_date')
    Path(tmp).unlink(missing_ok=True)

    if validation_start is None:
        validation_start = base_config['cali_end_date']
    if validation_end is None:
        validation_end = base_config['end_date']

    # Setup directories
    plot_dirs = setup_metric_comparison_directories(gauge_id, config_key, main_dir)

    # Load all data
    print(f"\n  Loading data for {len(metrics)} metrics...")
    all_data = load_all_metric_data(base_config, metrics)

    if len(all_data) < 2:
        print(f"  Only {len(all_data)} metrics loaded. Need at least 2. Skipping.")
        return

    # Run plots
    plot_functions = [
        ('Hydrograph Regime', plot_metric_hydrograph_regime,
         [base_config, metrics, all_data, plot_dirs, validation_start, validation_end]),
        ('Flow Duration Curve', plot_metric_flow_duration_curve,
         [base_config, metrics, all_data, plot_dirs, validation_start, validation_end]),
        ('Low Flow Comparison', plot_metric_low_flow_comparison,
         [base_config, metrics, all_data, plot_dirs, validation_start, validation_end]),
        ('Hydrograph Timeseries', plot_metric_hydrograph_timeseries,
         [base_config, metrics, all_data, plot_dirs, validation_start, validation_end]),
        ('Performance Table', plot_metric_performance_table,
         [base_config, metrics, all_data, plot_dirs, validation_start, validation_end]),
        ('Parameter Comparison', plot_metric_parameter_comparison,
         [base_config, metrics, plot_dirs]),
        ('Parameter Boxplots', plot_metric_parameter_boxplots,
         [base_config, metrics, plot_dirs]),
        ('Soil Storage', plot_metric_soil_storage,
         [base_config, metrics, plot_dirs, validation_start, validation_end]),
        ('Recession Analysis', plot_metric_recession_analysis,
         [base_config, metrics, all_data, plot_dirs, validation_start, validation_end]),
    ]

    results = {}
    for name, func, args in plot_functions:
        try:
            print(f"\n  [{name}]")
            func(*args)
            results[name] = True
        except Exception as e:
            results[name] = False
            if skip_errors:
                print(f"  ERROR in {name}: {e}")
            else:
                raise

    success = sum(1 for v in results.values() if v)
    print(f"\n  Completed: {success}/{len(results)} plots")
    return results


def run_all_for_catchment(namelist_path, env=None, skip_errors=True):
    """Run metric comparison for all config × model combinations in a catchment namelist.

    Usage:
        run_all_for_catchment('namelists/catchment_0118.yaml')
    """
    with open(namelist_path) as f:
        nml = yaml.safe_load(f)

    gauge_id = str(nml['catchment'])
    configurations = nml.get('configurations', ['baseline'])
    models = nml.get('models', ['HBV'])

    for config_key in configurations:
        for model_type in models:
            run_complete_metric_postprocessing(
                gauge_id=gauge_id,
                config_key=config_key,
                model_type=model_type,
                env=env,
                skip_errors=skip_errors,
            )


#--------------------------------------------------------------------------------
################################## CLI ##########################################
#--------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

    parser = argparse.ArgumentParser(description="Compare calibration metrics for a catchment")
    parser.add_argument('namelist', nargs='?', default=None,
                        help='Path to catchment namelist YAML (runs all configs × models)')
    parser.add_argument('--gauge-id', '-g', default=None, help='Catchment gauge ID')
    parser.add_argument('--config', '-c', default=None, help='Configuration key (e.g. subdaily)')
    parser.add_argument('--model', '-m', default='HBV', help='Model type (default: HBV)')
    parser.add_argument('--main-dir', default=None, help='Main data directory')
    parser.add_argument('--metrics', nargs='+', default=None,
                        help='Metrics to compare (auto-discovers if omitted)')
    parser.add_argument('--env', default=None, help='Environment (local/server)')
    args = parser.parse_args()

    if args.namelist:
        # Run all configs × models from namelist
        run_all_for_catchment(args.namelist, env=args.env)
    elif args.gauge_id:
        if args.config:
            # Single config
            run_complete_metric_postprocessing(
                gauge_id=args.gauge_id,
                config_key=args.config,
                model_type=args.model,
                main_dir=args.main_dir,
                metrics=args.metrics,
                env=args.env,
            )
        else:
            parser.error("Either provide a namelist or --gauge-id with --config")
    else:
        parser.error("Provide a namelist path or --gauge-id")
