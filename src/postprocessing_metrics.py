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

    # Plot observed winter flows
    first_data = list(all_data.values())[0]
    mask = (first_data['date'] >= vali_start_dt) & (first_data['date'] <= vali_end_dt)
    vali_data = first_data[mask].copy()
    winter_mask = vali_data['date'].dt.month.isin([11, 12, 1, 2, 3])
    obs_winter = vali_data[winter_mask]
    ax.plot(obs_winter['date'], obs_winter['obs_Q'], 'k-', linewidth=1.5,
            label='Observed', alpha=0.8)

    # Plot each metric's winter simulation
    for metric in metrics:
        if metric not in all_data:
            continue
        data = all_data[metric]
        mask = (data['date'] >= vali_start_dt) & (data['date'] <= vali_end_dt)
        vali = data[mask].copy()
        winter = vali[vali['date'].dt.month.isin([11, 12, 1, 2, 3])]

        color = METRIC_COLORS.get(metric, '#333333')
        label = METRIC_DISPLAY_NAMES.get(metric, metric)
        ax.plot(winter['date'], winter['sim_Q'], '-', color=color,
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
