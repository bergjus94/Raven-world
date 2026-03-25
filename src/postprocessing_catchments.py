# Cross-catchment postprocessing for multi-configuration comparison
# March 2026

#--------------------------------------------------------------------------------
################################## packages #####################################
#--------------------------------------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from postprocessing import load_hydrograph_data, calculate_performance_metrics, load_glogem_data
from postprocessing_configurations import (
    load_configurations, _build_individual_config,
    run_complete_multi_postprocessing
)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Global plot style for cross-catchment plots
plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'legend.title_fontsize': 12,
})
from pathlib import Path
import yaml
import argparse
import time
import csv


#--------------------------------------------------------------------------------
################################## data loading #################################
#--------------------------------------------------------------------------------

def load_catchment_registry(yaml_path):
    """Load the catchments block from configurations.yaml."""
    with open(yaml_path, 'r') as f:
        registry = yaml.safe_load(f)
    return registry.get('catchments', [])


def load_all_catchment_data(yaml_path, catchment_filter=None, config_filter=None):
    """
    Load metrics and regime data for all catchments × configurations.

    Returns catchment_results dict keyed by gauge_id, plus config_registry list.
    Uses load-compute-discard: each hydrograph CSV is freed after metric extraction.
    """
    with open(yaml_path, 'r') as f:
        registry = yaml.safe_load(f)

    catchments = registry.get('catchments', [])
    config_registry = registry['configurations']

    # Auto-assign catchment colors if not specified in YAML
    _catchment_palette = [
        '#1b9e77', '#d95f02', '#7570b3', '#e7298a',
        '#66a61e', '#e6ab02', '#a6761d', '#666666',
    ]
    for i, catch in enumerate(catchments):
        if 'color' not in catch:
            catch['color'] = _catchment_palette[i % len(_catchment_palette)]

    if catchment_filter:
        catchments = [c for c in catchments if c['gauge_id'] in catchment_filter]
    if config_filter:
        config_registry_filtered = [c for c in config_registry if c['key'] in config_filter]
    else:
        config_registry_filtered = config_registry

    catchment_results = {}

    for catch in catchments:
        gauge_id = catch['gauge_id']
        print(f"\n{'='*60}")
        print(f"Loading catchment {gauge_id} ({catch['display_name']})")
        print(f"{'='*60}")

        # Load multi_config for this catchment (resolves namelists, config_dirs)
        multi_config = load_configurations(yaml_path, gauge_id)
        if multi_config is None:
            print(f"  WARNING: Could not load configurations for {gauge_id}, skipping")
            continue

        validation_start = catch['validation_start']
        validation_end = catch['validation_end']

        # Build a key->config_dir mapping for this catchment
        key_to_dir = {}
        for cfg in config_registry:
            key = cfg['key']
            # Find matching config_dir in multi_config
            for cdir in multi_config['configs']:
                if multi_config['config_names'].get(cdir) == cfg['display_name']:
                    key_to_dir[key] = cdir
                    break

        metrics = {}
        regime_data = {}

        for cfg in config_registry_filtered:
            key = cfg['key']
            config_dir = key_to_dir.get(key)
            if config_dir is None:
                print(f"  - {key}: not available for {gauge_id}")
                continue

            individual_config = _build_individual_config(multi_config, config_dir)

            try:
                data = load_hydrograph_data(individual_config)
                if data is None:
                    continue

                # Calculate metrics
                start_dt = pd.to_datetime(validation_start)
                end_dt = pd.to_datetime(validation_end)
                m = calculate_performance_metrics(data, start_dt, end_dt, f"{key}")
                if m is not None:
                    metrics[key] = m

                # Calculate monthly regime for validation period
                val_mask = (data['date'] >= validation_start) & (data['date'] <= validation_end)
                df_val = data[val_mask].copy()
                if len(df_val) > 0 and 'sim_Q' in df_val.columns:
                    df_val['month'] = df_val['date'].dt.month
                    monthly = df_val.groupby('month').agg(
                        sim_Q=('sim_Q', 'mean'),
                        obs_Q=('obs_Q', 'mean') if 'obs_Q' in df_val.columns else ('sim_Q', 'mean')
                    ).reset_index()
                    regime_data[key] = monthly

            except Exception as e:
                print(f"  Error loading {key} for {gauge_id}: {e}")
                continue

            # Discard data (load-compute-discard)
            del data

        catchment_results[gauge_id] = {
            'multi_config': multi_config,
            'metrics': metrics,
            'regime_data': regime_data,
            'catchment_info': catch,
            'key_to_dir': key_to_dir,
        }

        n_loaded = len(metrics)
        print(f"  Loaded {n_loaded} configurations with metrics")

    return catchment_results, config_registry_filtered


def create_cross_catchment_plot_dir(yaml_path):
    """Create output directory for cross-catchment plots."""
    with open(yaml_path, 'r') as f:
        registry = yaml.safe_load(f)
    main_dir = Path(registry['main_dir'])
    plot_dir = main_dir / 'cross_catchment_plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


#--------------------------------------------------------------------------------
########################### Plot 1: Performance Heatmap #########################
#--------------------------------------------------------------------------------

def plot_performance_heatmap(catchment_results, config_registry, plot_dir):
    """
    Separate heatmaps for KGE and NSE.
    Catchments on y-axis, configs on x-axis. Cells annotated with values.
    """
    catchment_ids = list(catchment_results.keys())
    config_keys = [c['key'] for c in config_registry]
    config_names = {c['key']: c['display_name'] for c in config_registry}

    # Build matrices
    kge_matrix = np.full((len(catchment_ids), len(config_keys)), np.nan)
    nse_matrix = np.full((len(catchment_ids), len(config_keys)), np.nan)

    for i, gid in enumerate(catchment_ids):
        metrics = catchment_results[gid]['metrics']
        for j, key in enumerate(config_keys):
            if key in metrics:
                kge_matrix[i, j] = metrics[key]['KGE']
                nse_matrix[i, j] = metrics[key]['NSE']

    # Filter out configs with no data at all
    valid_cols = ~np.all(np.isnan(kge_matrix), axis=0)
    kge_matrix = kge_matrix[:, valid_cols]
    nse_matrix = nse_matrix[:, valid_cols]
    config_keys_valid = [k for k, v in zip(config_keys, valid_cols) if v]

    if len(config_keys_valid) == 0:
        print("No valid data for heatmap")
        return

    y_labels = [catchment_results[gid]['catchment_info']['display_name'] for gid in catchment_ids]
    x_labels = [config_names[k] for k in config_keys_valid]

    for matrix, metric_name, filename in [
        (kge_matrix, 'KGE', 'cross_catchment_heatmap_KGE.png'),
        (nse_matrix, 'NSE', 'cross_catchment_heatmap_NSE.png'),
    ]:
        # Compute color scale from actual data range
        valid_vals = matrix[~np.isnan(matrix)]
        if len(valid_vals) == 0:
            continue
        vmin = max(np.floor(valid_vals.min() * 10) / 10, -1.0)
        vmax = min(np.ceil(valid_vals.max() * 10) / 10, 1.0)

        fig, ax = plt.subplots(figsize=(max(12, len(config_keys_valid) * 1.2),
                                        max(4, len(catchment_ids) * 1.0)))

        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)

        # Annotate cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, '—', ha='center', va='center', fontsize=9, color='gray')
                else:
                    # Choose text color for readability
                    color = 'white' if val < (vmin + (vmax - vmin) * 0.3) else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9,
                            fontweight='bold', color=color)

        # Hatching for NaN cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isnan(matrix[i, j]):
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                               fill=True, facecolor='lightgray',
                                               edgecolor='gray', hatch='//'))

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
        fig.colorbar(im, ax=ax, shrink=0.8, label=metric_name)

        plt.tight_layout()
        save_path = plot_dir / filename
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
######################## Plot 2: Configuration Ranking ##########################
#--------------------------------------------------------------------------------

def plot_configuration_ranking(catchment_results, config_registry, plot_dir):
    """
    Grouped bar chart: for each config, bars grouped by catchment showing KGE.
    Sorted by mean performance across catchments.
    """
    catchment_ids = list(catchment_results.keys())
    config_keys = [c['key'] for c in config_registry]
    config_names = {c['key']: c['display_name'] for c in config_registry}
    catch_colors = {gid: catchment_results[gid]['catchment_info']['color'] for gid in catchment_ids}
    catch_names = {gid: catchment_results[gid]['catchment_info']['display_name'] for gid in catchment_ids}

    # Collect KGE per config per catchment
    kge_data = {}
    for key in config_keys:
        vals = []
        for gid in catchment_ids:
            m = catchment_results[gid]['metrics'].get(key)
            vals.append(m['KGE'] if m else np.nan)
        if not all(np.isnan(v) for v in vals):
            kge_data[key] = vals

    if not kge_data:
        print("No data for configuration ranking")
        return

    # Sort by mean KGE (descending)
    sorted_keys = sorted(kge_data.keys(),
                         key=lambda k: np.nanmean(kge_data[k]), reverse=True)

    n_catchments = len(catchment_ids)
    n_configs = len(sorted_keys)
    bar_width = 0.8 / n_catchments
    x = np.arange(n_configs)

    fig, ax = plt.subplots(figsize=(max(12, n_configs * 1.2), 7))

    for i, gid in enumerate(catchment_ids):
        vals = [kge_data[key][catchment_ids.index(gid)] for key in sorted_keys]
        offset = (i - n_catchments / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=catch_names[gid],
                      color=catch_colors[gid], edgecolor='black', linewidth=0.5)

    # Add mean line markers
    for j, key in enumerate(sorted_keys):
        mean_val = np.nanmean(kge_data[key])
        ax.plot(j, mean_val, 'k_', markersize=20, markeredgewidth=2.5, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([config_names[k] for k in sorted_keys], rotation=45, ha='right')
    ax.set_ylabel('KGE')
    ax.legend(title='Catchment')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    save_path = plot_dir / 'cross_catchment_config_ranking.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
###################### Plot 3: Regime Comparison ################################
#--------------------------------------------------------------------------------

def plot_regime_comparison(catchment_results, config_registry, plot_dir,
                           regime_configs=None):
    """
    One subplot per catchment showing observed + simulated monthly regime
    for key configurations.
    """
    catchment_ids = list(catchment_results.keys())
    config_names = {c['key']: c['display_name'] for c in config_registry}
    config_colors_map = {c['key']: c['color'] for c in config_registry}

    if regime_configs is None:
        # Default: show baseline + glogem + glogem_subdaily (or all if few)
        regime_configs = [c['key'] for c in config_registry]

    n_catch = len(catchment_ids)
    n_cols = min(n_catch, 2)
    n_rows = (n_catch + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), squeeze=False)

    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    for idx, gid in enumerate(catchment_ids):
        ax = axes[idx // n_cols][idx % n_cols]
        info = catchment_results[gid]['catchment_info']
        regime = catchment_results[gid]['regime_data']

        # Plot observed from first available config
        obs_plotted = False
        for key in regime_configs:
            if key in regime and 'obs_Q' in regime[key].columns:
                monthly = regime[key]
                ax.plot(monthly['month'], monthly['obs_Q'], 'k-', linewidth=2.5,
                        label='Observed', zorder=10)
                obs_plotted = True
                break

        # Plot simulated for each config
        for key in regime_configs:
            if key not in regime:
                continue
            monthly = regime[key]
            if 'sim_Q' not in monthly.columns:
                continue
            color = config_colors_map.get(key, 'gray')
            name = config_names.get(key, key)

            # Add KGE to label if available
            m = catchment_results[gid]['metrics'].get(key)
            if m:
                label = f"{name} (KGE={m['KGE']:.2f})"
            else:
                label = name

            ax.plot(monthly['month'], monthly['sim_Q'], color=color,
                    linewidth=2, label=label)

        ax.set_title(f"{info['display_name']} ({gid})", fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_labels)
        ax.set_xlabel('Month')
        ax.set_ylabel('Discharge (m³/s)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

    # Hide unused subplots
    for idx in range(n_catch, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    plt.tight_layout()
    save_path = plot_dir / 'cross_catchment_regime_comparison.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
################### Plot 4: Glacier Contribution Comparison #####################
#--------------------------------------------------------------------------------

def plot_glacier_contribution_comparison(catchment_results, config_registry, plot_dir):
    """
    Bar chart of glacier melt fraction of total simulated Q for each catchment,
    across ALL configurations (coupled and uncoupled).
    Uses Raven's GLACIERMELT_ALLMassLoadings.csv which is produced by all runs.
    """
    catchment_ids = list(catchment_results.keys())
    catch_names = {gid: catchment_results[gid]['catchment_info']['display_name'] for gid in catchment_ids}
    config_names = {c['key']: c['display_name'] for c in config_registry}
    config_keys = [c['key'] for c in config_registry]

    glacier_fractions = {}  # {config_key: {gauge_id: fraction}}

    for key in config_keys:
        glacier_fractions[key] = {}
        for gid in catchment_ids:
            mc = catchment_results[gid].get('multi_config')
            key_to_dir = catchment_results[gid].get('key_to_dir', {})
            config_dir = key_to_dir.get(key)
            if mc is None or config_dir is None:
                continue

            individual_config = _build_individual_config(mc, config_dir)
            try:
                info = catchment_results[gid]['catchment_info']
                gauge_id = mc['gauge_id']
                model_type = mc['model_type']
                config_path = Path(mc['main_dir']) / config_dir

                # Load GLACIERMELT_ALLMassLoadings.csv (produced by all Raven runs)
                glacier_file = (config_path / f"catchment_{gauge_id}" / model_type /
                                "output" / f"{gauge_id}_{model_type}_GLACIERMELT_ALLMassLoadings.csv")
                if not glacier_file.exists():
                    continue

                glacier_df = pd.read_csv(glacier_file)
                glacier_df['date'] = pd.to_datetime(glacier_df['date'])
                glacier_m3s_col = f"{gauge_id} m3/s"
                if glacier_m3s_col not in glacier_df.columns:
                    continue

                # Filter to validation period
                val_mask = ((glacier_df['date'] >= info['validation_start']) &
                            (glacier_df['date'] <= info['validation_end']))
                glacier_val = glacier_df[val_mask]
                mean_glacier_m3s = glacier_val[glacier_m3s_col].mean()

                # Load simulated Q for the same period
                data = load_hydrograph_data(individual_config)
                if data is None:
                    continue
                val_mask_q = ((data['date'] >= info['validation_start']) &
                              (data['date'] <= info['validation_end']))
                mean_sim_q = data[val_mask_q]['sim_Q'].mean()
                del data

                if mean_sim_q > 0:
                    glacier_fractions[key][gid] = mean_glacier_m3s / mean_sim_q

            except Exception as e:
                print(f"  Glacier contribution error for {key}/{gid}: {e}")
                continue

    # Filter to configs that have data for at least one catchment
    valid_keys = [k for k in config_keys if glacier_fractions.get(k)]
    if not valid_keys:
        print("No glacier contribution data available")
        return

    n_catchments = len(catchment_ids)
    n_configs = len(valid_keys)
    bar_width = 0.8 / n_catchments
    x = np.arange(n_configs)

    fig, ax = plt.subplots(figsize=(max(10, n_configs * 1.5), 6))

    for i, gid in enumerate(catchment_ids):
        vals = [glacier_fractions[key].get(gid, np.nan) * 100 for key in valid_keys]
        offset = (i - n_catchments / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=catch_names[gid],
               color=catchment_results[gid]['catchment_info']['color'],
               edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([config_names[k] for k in valid_keys], rotation=45, ha='right')
    ax.set_ylabel('Glacier Melt Fraction of Simulated Q (%)')
    ax.legend(title='Catchment')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = plot_dir / 'cross_catchment_glacier_contribution.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
###################### Plot 5: Coupling Effect (Delta-KGE) ######################
#--------------------------------------------------------------------------------

def plot_coupling_effect(catchment_results, config_registry, plot_dir):
    """
    For each catchment, show KGE change when switching from uncoupled to coupled.
    Diverging bars: green=improvement, red=degradation.
    """
    catchment_ids = list(catchment_results.keys())
    catch_names = {gid: catchment_results[gid]['catchment_info']['display_name'] for gid in catchment_ids}

    # Define coupling pairs: (uncoupled_key, coupled_key, pair_label)
    pairs = [
        ('baseline', 'glogem', 'Baseline → GloGEM'),
        ('subdaily', 'glogem_subdaily', 'Subdaily → GloGEM Subdaily'),
        ('icimod', 'glogem_icimod', 'ICIMOD → GloGEM ICIMOD'),
        ('har', 'glogem_har', 'HAR → GloGEM HAR'),
        ('subdaily_aspect', 'glogem_subdaily_aspect', 'Aspect → GloGEM Aspect'),
        ('oudin', 'glogem_oudin', 'Oudin → GloGEM Oudin'),
    ]

    # Filter pairs to those that exist in at least one catchment
    valid_pairs = []
    for uncoupled, coupled, label in pairs:
        for gid in catchment_ids:
            m = catchment_results[gid]['metrics']
            if uncoupled in m and coupled in m:
                valid_pairs.append((uncoupled, coupled, label))
                break

    if not valid_pairs:
        print("No coupling pairs found")
        return

    n_pairs = len(valid_pairs)
    n_catchments = len(catchment_ids)
    bar_width = 0.8 / n_catchments
    x = np.arange(n_pairs)

    fig, ax = plt.subplots(figsize=(max(10, n_pairs * 1.8), 6))

    for i, gid in enumerate(catchment_ids):
        deltas = []
        for uncoupled, coupled, label in valid_pairs:
            m = catchment_results[gid]['metrics']
            if uncoupled in m and coupled in m:
                delta = m[coupled]['KGE'] - m[uncoupled]['KGE']
            else:
                delta = np.nan
            deltas.append(delta)

        offset = (i - n_catchments / 2 + 0.5) * bar_width
        colors = ['#2ca02c' if d >= 0 else '#d62728' for d in deltas]
        ax.bar(x + offset, deltas, bar_width, color=colors,
               edgecolor='black', linewidth=0.5, label=catch_names[gid] if i == 0 or True else '')

    # Custom legend: one entry per catchment + improvement/degradation
    legend_handles = []
    for gid in catchment_ids:
        legend_handles.append(Patch(facecolor=catchment_results[gid]['catchment_info']['color'],
                                    edgecolor='black', label=catch_names[gid]))

    # Re-do with catchment colors instead of green/red per bar
    # Actually, let's use catchment colors with positive/negative direction
    fig, ax = plt.subplots(figsize=(max(10, n_pairs * 1.8), 6))

    for i, gid in enumerate(catchment_ids):
        deltas = []
        for uncoupled, coupled, label in valid_pairs:
            m = catchment_results[gid]['metrics']
            if uncoupled in m and coupled in m:
                delta = m[coupled]['KGE'] - m[uncoupled]['KGE']
            else:
                delta = np.nan
            deltas.append(delta)

        offset = (i - n_catchments / 2 + 0.5) * bar_width
        catch_color = catchment_results[gid]['catchment_info']['color']
        ax.bar(x + offset, deltas, bar_width, color=catch_color,
               edgecolor='black', linewidth=0.5, label=catch_names[gid])

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, label in valid_pairs], rotation=45, ha='right')
    ax.set_ylabel('ΔKGE (coupled − uncoupled)')
    ax.legend(title='Catchment')
    ax.grid(True, alpha=0.3, axis='y')

    # Shade background
    ax.axhspan(0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.1,
               alpha=0.05, color='green', zorder=0)
    ax.axhspan(ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -0.1, 0,
               alpha=0.05, color='red', zorder=0)

    plt.tight_layout()
    save_path = plot_dir / 'cross_catchment_coupling_effect.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
############## Plot 5b: Subdaily Effect #########################################
#--------------------------------------------------------------------------------

def plot_subdaily_effect(catchment_results, config_registry, plot_dir):
    """Boxplot comparing KGE of daily vs subdaily timestep configs."""
    groups = [
        ('Uncoupled', [
            ('baseline', 'Daily'),
            ('subdaily', 'Subdaily'),
        ]),
        ('Coupled (GloGEM)', [
            ('glogem', 'Daily'),
            ('glogem_subdaily', 'Subdaily'),
        ]),
        ('Icemelt', [
            ('icemelt', 'Daily'),
            ('icemelt_subdaily', 'Subdaily'),
        ]),
    ]
    _plot_grouped_boxplot(catchment_results, groups, plot_dir,
                          'cross_catchment_subdaily_effect.png')


#--------------------------------------------------------------------------------
############# Plot 5c: Meteorological Forcing Comparison ########################
#--------------------------------------------------------------------------------

def _plot_grouped_boxplot(catchment_results, groups, plot_dir, filename, ylabel='KGE'):
    """
    Boxplot comparing KGE across grouped configurations.
    Each box shows the spread across catchments for one config.
    Individual catchment points overlaid as colored dots.
    """
    catchment_ids = list(catchment_results.keys())
    catch_colors = {gid: catchment_results[gid]['catchment_info']['color'] for gid in catchment_ids}
    catch_names = {gid: catchment_results[gid]['catchment_info']['display_name'] for gid in catchment_ids}

    # Filter to groups/configs that have data
    valid_groups = []
    for group_label, configs in groups:
        valid_configs = []
        for key, config_label in configs:
            for gid in catchment_ids:
                if key in catchment_results[gid]['metrics']:
                    valid_configs.append((key, config_label))
                    break
        if valid_configs:
            valid_groups.append((group_label, valid_configs))

    if not valid_groups:
        print(f"No data for {filename}")
        return

    # Build flat lists
    x_labels = []
    config_key_list = []
    box_data = []
    group_colors = {'Uncoupled': '#aec7e8', 'Coupled (GloGEM)': '#ffbb78',
                    'Daily': '#aec7e8', 'Subdaily': '#ffbb78',
                    'Icemelt': '#c5b0d5'}

    for group_label, configs in valid_groups:
        for key, config_label in configs:
            x_labels.append(f"{config_label}\n({group_label})")
            config_key_list.append(key)
            vals = [catchment_results[gid]['metrics'][key]['KGE']
                    for gid in catchment_ids
                    if key in catchment_results[gid]['metrics']]
            box_data.append(vals)

    n_items = len(config_key_list)
    fig, ax = plt.subplots(figsize=(max(10, n_items * 1.8), 7))

    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))

    # Color boxes by group
    box_idx = 0
    for group_label, configs in valid_groups:
        color = group_colors.get(group_label, '#d9d9d9')
        for _ in configs:
            bp['boxes'][box_idx].set_facecolor(color)
            bp['boxes'][box_idx].set_alpha(0.5)
            box_idx += 1

    # Overlay individual catchment points
    for i, key in enumerate(config_key_list):
        for gid in catchment_ids:
            m = catchment_results[gid]['metrics'].get(key)
            if m:
                ax.scatter(i + 1, m['KGE'], color=catch_colors[gid],
                           s=60, zorder=5, edgecolors='black', linewidth=0.5)

    # Legend for catchments
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=catch_colors[gid],
                                  markeredgecolor='black', markersize=8,
                                  label=catch_names[gid])
                      for gid in catchment_ids]
    ax.legend(handles=legend_handles, title='Catchment', loc='best')

    ax.set_xticks(range(1, n_items + 1))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # Add vertical separators between groups
    cumulative = 0
    for gi, (group_label, configs) in enumerate(valid_groups):
        if gi > 0:
            ax.axvline(x=cumulative + 0.5, color='gray', linestyle='--',
                       linewidth=1, alpha=0.5)
        cumulative += len(configs)

    plt.tight_layout()
    save_path = plot_dir / filename
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_meteo_forcing_comparison(catchment_results, config_registry, plot_dir):
    """Boxplot comparing KGE across meteorological forcing datasets."""
    groups = [
        ('Uncoupled', [
            ('baseline', 'ERA5'),
            ('har', 'HAR'),
            ('tphipr', 'TPHiPr'),
        ]),
        ('Coupled (GloGEM)', [
            ('glogem', 'ERA5'),
            ('glogem_har', 'HAR'),
            ('glogem_tphipr', 'TPHiPr'),
        ]),
    ]
    _plot_grouped_boxplot(catchment_results, groups, plot_dir,
                          'cross_catchment_meteo_forcing.png')


#--------------------------------------------------------------------------------
############ Plot 5d: Coupling Method Comparison ################################
#--------------------------------------------------------------------------------

def plot_coupling_method_comparison(catchment_results, config_registry, plot_dir):
    """Boxplot comparing KGE across glacier coupling methods."""
    groups = [
        ('Daily', [
            ('baseline', 'Uncoupled'),
            ('glogem', 'TSLA'),
            ('glogem_gmb', 'GMB'),
            ('icemelt', 'Icemelt'),
        ]),
        ('Subdaily', [
            ('subdaily', 'Uncoupled'),
            ('glogem_subdaily', 'TSLA'),
            ('icemelt_subdaily', 'Icemelt'),
        ]),
    ]
    _plot_grouped_boxplot(catchment_results, groups, plot_dir,
                          'cross_catchment_coupling_method.png')


#--------------------------------------------------------------------------------
############ Plot 5e: ET Method Comparison ######################################
#--------------------------------------------------------------------------------

def plot_et_method_comparison(catchment_results, config_registry, plot_dir):
    """Boxplot comparing KGE across PET methods."""
    groups = [
        ('Uncoupled', [
            ('baseline', 'ERA5'),
            ('har', 'HAR'),
            ('oudin', 'Oudin'),
        ]),
        ('Coupled (GloGEM)', [
            ('glogem', 'ERA5'),
            ('glogem_har', 'HAR'),
            ('glogem_oudin', 'Oudin'),
        ]),
    ]
    _plot_grouped_boxplot(catchment_results, groups, plot_dir,
                          'cross_catchment_et_method.png')


#--------------------------------------------------------------------------------
############ Plot 5f: Landuse Effect ############################################
#--------------------------------------------------------------------------------

def plot_landuse_effect(catchment_results, config_registry, plot_dir):
    """Boxplot comparing KGE across landuse datasets (ESA vs ICIMOD)."""
    groups = [
        ('Uncoupled', [
            ('baseline', 'ESA'),
            ('icimod', 'ICIMOD'),
        ]),
        ('Coupled (GloGEM)', [
            ('glogem', 'ESA'),
            ('glogem_icimod', 'ICIMOD'),
        ]),
    ]
    _plot_grouped_boxplot(catchment_results, groups, plot_dir,
                          'cross_catchment_landuse_effect.png')


#--------------------------------------------------------------------------------
########### Sensitivity factors definition ######################################
#--------------------------------------------------------------------------------

def _define_sensitivity_factors():
    """
    Define modeling choice factors and their comparison pairs.

    Each factor is a dict with:
        name:  display name
        pairs: list of (key_a, key_b, pair_label) where ΔKGE = KGE(key_b) - KGE(key_a)
               For multi-option factors, pairs cover all relevant comparisons.
        sign_label: (negative_meaning, positive_meaning) for interpretation
    """
    return [
        {
            'name': 'Uncoupled →\nGloGEM TSLA',
            'pairs': [
                ('baseline', 'glogem', 'Daily ERA5'),
                ('subdaily', 'glogem_subdaily', 'Subdaily ERA5'),
                ('har', 'glogem_har', 'Daily HAR'),
                ('oudin', 'glogem_oudin', 'Daily Oudin'),
                ('subdaily_aspect', 'glogem_subdaily_aspect', 'Subdaily Aspect'),
                ('icimod', 'glogem_icimod', 'ICIMOD'),
            ],
            'sign_label': ('Uncoupled better', 'Coupled better'),
        },
        {
            'name': 'Daily →\nSubdaily',
            'pairs': [
                ('baseline', 'subdaily', 'Uncoupled ERA5'),
                ('glogem', 'glogem_subdaily', 'Coupled ERA5'),
                ('icemelt', 'icemelt_subdaily', 'Icemelt'),
            ],
            'sign_label': ('Daily better', 'Subdaily better'),
        },
        {
            'name': 'ERA5 →\nHAR Forcing',
            'pairs': [
                ('baseline', 'har', 'Uncoupled'),
                ('glogem', 'glogem_har', 'Coupled'),
            ],
            'sign_label': ('ERA5 better', 'HAR better'),
        },
        {
            'name': 'ERA5 PET →\nOudin PET',
            'pairs': [
                ('baseline', 'oudin', 'Uncoupled'),
                ('glogem', 'glogem_oudin', 'Coupled'),
            ],
            'sign_label': ('ERA5 better', 'Oudin better'),
        },
        {
            'name': 'ERA5 Precip →\nTPHiPr Precip',
            'pairs': [
                ('baseline', 'tphipr', 'Uncoupled'),
                ('glogem', 'glogem_tphipr', 'Coupled'),
            ],
            'sign_label': ('ERA5 better', 'TPHiPr better'),
        },
        {
            'name': 'GloGEM TSLA →\nIcemelt',
            'pairs': [
                ('glogem', 'icemelt', 'Daily'),
                ('glogem_subdaily', 'icemelt_subdaily', 'Subdaily'),
            ],
            'sign_label': ('TSLA better', 'Icemelt better'),
        },
        {
            'name': 'GloGEM TSLA →\nGloGEM GMB',
            'pairs': [
                ('glogem', 'glogem_gmb', 'Daily'),
            ],
            'sign_label': ('TSLA better', 'GMB better'),
        },
        {
            'name': 'ESA Landuse →\nICIMOD Landuse',
            'pairs': [
                ('baseline', 'icimod', 'Uncoupled'),
                ('glogem', 'glogem_icimod', 'Coupled'),
            ],
            'sign_label': ('ESA better', 'ICIMOD better'),
        },
        {
            'name': 'No Aspect →\nAspect HRU Split',
            'pairs': [
                ('subdaily', 'subdaily_aspect', 'Uncoupled'),
                ('glogem_subdaily', 'glogem_subdaily_aspect', 'Coupled'),
            ],
            'sign_label': ('Without better', 'Aspect better'),
        },
    ]


def _compute_factor_deltas(catchment_results, factors):
    """
    Compute ΔKGE per factor per catchment.

    Returns dict: {factor_name: {gauge_id: [list of ΔKGE values across valid pairs]}}
    """
    catchment_ids = list(catchment_results.keys())
    factor_deltas = {}

    for factor in factors:
        factor_deltas[factor['name']] = {}
        for gid in catchment_ids:
            m = catchment_results[gid]['metrics']
            deltas = []
            for key_a, key_b, _ in factor['pairs']:
                if key_a in m and key_b in m:
                    deltas.append(m[key_b]['KGE'] - m[key_a]['KGE'])
            if deltas:
                factor_deltas[factor['name']][gid] = deltas

    return factor_deltas


#--------------------------------------------------------------------------------
############ Plot 5f: Sensitivity Tornado #######################################
#--------------------------------------------------------------------------------

def plot_sensitivity_tornado(catchment_results, config_registry, plot_dir):
    """
    Tornado plot showing the range and mean of ΔKGE for each modeling factor.
    Bars show min-to-max range across all catchments and pairs; marker shows mean.
    """
    factors = _define_sensitivity_factors()
    factor_deltas = _compute_factor_deltas(catchment_results, factors)

    # Collect stats per factor
    factor_stats = []
    for factor in factors:
        name = factor['name']
        all_deltas = []
        for gid_deltas in factor_deltas.get(name, {}).values():
            all_deltas.extend(gid_deltas)
        if all_deltas:
            factor_stats.append({
                'name': name,
                'mean': np.mean(all_deltas),
                'mean_abs': np.mean(np.abs(all_deltas)),
                'min': np.min(all_deltas),
                'max': np.max(all_deltas),
                'median': np.median(all_deltas),
                'all': all_deltas,
                'sign_label': factor['sign_label'],
            })

    if not factor_stats:
        print("No data for sensitivity tornado")
        return

    # Sort by mean absolute effect (largest at top)
    factor_stats.sort(key=lambda f: f['mean_abs'])

    n = len(factor_stats)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.8)))

    y = np.arange(n)
    for i, fs in enumerate(factor_stats):
        # Draw range bar (min to max)
        color = '#2ca02c' if fs['mean'] >= 0 else '#d62728'
        ax.barh(i, fs['max'] - fs['min'], left=fs['min'], height=0.6,
                color=color, alpha=0.3, edgecolor=color, linewidth=1)
        # Draw individual points
        for d in fs['all']:
            ax.plot(d, i, 'o', color='gray', markersize=4, alpha=0.5, zorder=3)
        # Draw mean marker
        ax.plot(fs['mean'], i, 'D', color='black', markersize=8, zorder=5)
        # Draw median marker
        ax.plot(fs['median'], i, '|', color='black', markersize=15,
                markeredgewidth=2, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels([fs['name'] for fs in factor_stats])
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('ΔKGE')
    ax.grid(True, alpha=0.3, axis='x')

    # Add sign labels on edges
    xlim = ax.get_xlim()
    for i, fs in enumerate(factor_stats):
        neg_label, pos_label = fs['sign_label']
        ax.text(xlim[0], i + 0.35, neg_label, fontsize=8, color='#d62728',
                ha='left', va='bottom', alpha=0.7)
        ax.text(xlim[1], i + 0.35, pos_label, fontsize=8, color='#2ca02c',
                ha='right', va='bottom', alpha=0.7)

    plt.tight_layout()
    save_path = plot_dir / 'cross_catchment_sensitivity_tornado.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
############ Plot 5g: Factor Importance Heatmap #################################
#--------------------------------------------------------------------------------

def plot_factor_importance_heatmap(catchment_results, config_registry, plot_dir):
    """
    Heatmap with catchments on y-axis and modeling factors on x-axis.
    Cells colored by mean ΔKGE for that factor in that catchment.
    Marginal bar on top shows mean |ΔKGE| across catchments (factor importance).
    """
    factors = _define_sensitivity_factors()
    factor_deltas = _compute_factor_deltas(catchment_results, factors)
    catchment_ids = list(catchment_results.keys())

    # Filter to factors with data in at least one catchment
    valid_factors = [f for f in factors if factor_deltas.get(f['name'])]
    if not valid_factors:
        print("No data for factor importance heatmap")
        return

    # Sort factors by mean absolute effect (largest first)
    factor_mean_abs = {}
    for f in valid_factors:
        all_deltas = []
        for gid_deltas in factor_deltas[f['name']].values():
            all_deltas.extend(gid_deltas)
        factor_mean_abs[f['name']] = np.mean(np.abs(all_deltas)) if all_deltas else 0
    valid_factors.sort(key=lambda f: factor_mean_abs[f['name']], reverse=True)

    factor_names = [f['name'] for f in valid_factors]
    n_factors = len(factor_names)
    n_catch = len(catchment_ids)

    # Build matrix: mean ΔKGE per catchment per factor
    matrix = np.full((n_catch, n_factors), np.nan)
    for j, f in enumerate(valid_factors):
        for i, gid in enumerate(catchment_ids):
            deltas = factor_deltas[f['name']].get(gid, [])
            if deltas:
                matrix[i, j] = np.mean(deltas)

    y_labels = [catchment_results[gid]['catchment_info']['display_name']
                for gid in catchment_ids]
    x_labels = [f['name'].replace('\n', ' ') for f in valid_factors]
    # Add sign_label to x-axis: "Base → Alternative\n(−: base better, +: alt better)"
    x_labels_annotated = []
    for f in valid_factors:
        neg, pos = f['sign_label']
        x_labels_annotated.append(f"{f['name'].replace(chr(10), ' ')}\n(− {neg} / + {pos})")
    x_labels = x_labels_annotated

    # Diverging colorscale centered at 0
    abs_max = np.nanmax(np.abs(matrix))
    if abs_max == 0:
        abs_max = 0.1

    fig, (ax_bar, ax_heat) = plt.subplots(
        2, 1, figsize=(max(12, n_factors * 1.5), max(6, n_catch * 0.8 + 3)),
        gridspec_kw={'height_ratios': [1, 3]}, sharex=True)

    # Top bar: mean ΔKGE per factor (signed, colored by direction)
    mean_deltas = []
    for f in valid_factors:
        all_d = []
        for gid_deltas in factor_deltas.get(f['name'], {}).values():
            all_d.extend(gid_deltas)
        mean_deltas.append(np.mean(all_d) if all_d else 0)
    bar_colors = ['#2ca02c' if d >= 0 else '#d62728' for d in mean_deltas]
    ax_bar.bar(range(n_factors), mean_deltas, color=bar_colors, edgecolor='black',
               linewidth=0.5, width=0.7)
    ax_bar.set_ylabel('Mean ΔKGE')
    ax_bar.axhline(y=0, color='black', linewidth=0.5)
    ax_bar.grid(True, alpha=0.3, axis='y')
    # Annotate bars
    for j, d in enumerate(mean_deltas):
        va = 'bottom' if d >= 0 else 'top'
        offset = abs_max * 0.02 if d >= 0 else -abs_max * 0.02
        ax_bar.text(j, d + offset, f'{d:+.3f}', ha='center', va=va,
                    fontsize=10, fontweight='bold')

    # Bottom heatmap: ΔKGE per catchment × factor
    im = ax_heat.imshow(matrix, cmap='RdYlGn', aspect='auto',
                        vmin=-abs_max, vmax=abs_max)

    # Annotate cells
    for i in range(n_catch):
        for j in range(n_factors):
            val = matrix[i, j]
            if np.isnan(val):
                ax_heat.text(j, i, '—', ha='center', va='center',
                             fontsize=9, color='gray')
            else:
                color = 'white' if abs(val) > abs_max * 0.7 else 'black'
                ax_heat.text(j, i, f'{val:+.3f}', ha='center', va='center',
                             fontsize=9, fontweight='bold', color=color)

    # Hatching for NaN cells
    for i in range(n_catch):
        for j in range(n_factors):
            if np.isnan(matrix[i, j]):
                ax_heat.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=True,
                    facecolor='lightgray', edgecolor='gray', hatch='//'))

    ax_heat.set_xticks(range(n_factors))
    ax_heat.set_xticklabels(x_labels, rotation=45, ha='right')
    ax_heat.set_yticks(range(n_catch))
    ax_heat.set_yticklabels(y_labels)
    fig.colorbar(im, ax=ax_heat, shrink=0.8, label='Mean ΔKGE')

    plt.tight_layout()
    save_path = plot_dir / 'cross_catchment_factor_importance.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
#################### Plot 6: KGE Boxplot per Configuration ######################
#--------------------------------------------------------------------------------

def plot_kge_boxplot(catchment_results, config_registry, plot_dir):
    """
    Boxplot with one box per configuration, showing KGE spread across catchments.
    Individual catchment points overlaid as colored dots.
    """
    catchment_ids = list(catchment_results.keys())
    config_keys = [c['key'] for c in config_registry]
    config_names = {c['key']: c['display_name'] for c in config_registry}
    config_colors = {c['key']: c['color'] for c in config_registry}
    catch_colors = {gid: catchment_results[gid]['catchment_info']['color'] for gid in catchment_ids}
    catch_names = {gid: catchment_results[gid]['catchment_info']['display_name'] for gid in catchment_ids}

    # Collect KGE values per config
    box_data = []
    box_labels = []
    box_colors = []
    valid_keys = []

    for key in config_keys:
        vals = [catchment_results[gid]['metrics'][key]['KGE']
                for gid in catchment_ids
                if key in catchment_results[gid]['metrics']]
        if vals:
            box_data.append(vals)
            box_labels.append(config_names[key])
            box_colors.append(config_colors[key])
            valid_keys.append(key)

    if not box_data:
        print("No data for boxplot")
        return

    fig, ax = plt.subplots(figsize=(max(12, len(valid_keys) * 1.0), 7))

    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    # Overlay individual catchment points
    for i, key in enumerate(valid_keys):
        for gid in catchment_ids:
            m = catchment_results[gid]['metrics'].get(key)
            if m:
                ax.scatter(i + 1, m['KGE'], color=catch_colors[gid],
                           s=60, zorder=5, edgecolors='black', linewidth=0.5)

    # Legend for catchments
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=catch_colors[gid],
                                  markeredgecolor='black', markersize=8,
                                  label=catch_names[gid])
                      for gid in catchment_ids]
    ax.legend(handles=legend_handles, title='Catchment', loc='best')

    ax.set_xticks(range(1, len(valid_keys) + 1))
    ax.set_xticklabels(box_labels, rotation=45, ha='right')
    ax.set_ylabel('KGE')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    save_path = plot_dir / 'cross_catchment_kge_boxplot.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
##################### Plot 7: KGE Component Radar ##############################
#--------------------------------------------------------------------------------

def plot_kge_component_radar(catchment_results, config_registry, plot_dir,
                              radar_configs=None):
    """
    Radar plot with axes r, alpha, beta for selected configs across all catchments.
    One subplot per config.
    """
    catchment_ids = list(catchment_results.keys())
    catch_names = {gid: catchment_results[gid]['catchment_info']['display_name'] for gid in catchment_ids}
    catch_colors = {gid: catchment_results[gid]['catchment_info']['color'] for gid in catchment_ids}
    config_names = {c['key']: c['display_name'] for c in config_registry}

    if radar_configs is None:
        # Default: pick configs available in most catchments
        config_counts = {}
        for c in config_registry:
            count = sum(1 for gid in catchment_ids if c['key'] in catchment_results[gid]['metrics'])
            if count > 0:
                config_counts[c['key']] = count
        radar_configs = sorted(config_counts, key=config_counts.get, reverse=True)[:6]

    if not radar_configs:
        print("No configs for radar plot")
        return

    categories = ['r (correlation)', 'α (variability)', 'β (bias)']
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    n_configs = len(radar_configs)
    n_cols = min(n_configs, 3)
    n_rows = (n_configs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                             subplot_kw=dict(polar=True), squeeze=False)

    for idx, key in enumerate(radar_configs):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.set_title(config_names.get(key, key), fontweight='bold', pad=20)

        for gid in catchment_ids:
            m = catchment_results[gid]['metrics'].get(key)
            if m is None:
                continue
            values = [m['r'], m['alpha'], m['beta']]
            values += values[:1]  # Close polygon
            ax.plot(angles, values, color=catch_colors[gid], linewidth=2,
                    label=catch_names[gid])
            ax.fill(angles, values, color=catch_colors[gid], alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, max(2.0, ax.get_ylim()[1]))

        # Reference circle at 1.0 (ideal)
        ideal = [1.0] * n_cats + [1.0]
        ax.plot(angles, ideal, 'k--', linewidth=1, alpha=0.5, label='Ideal')

        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)

    # Hide unused subplots
    for idx in range(n_configs, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    plt.tight_layout()
    save_path = plot_dir / 'cross_catchment_kge_radar.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
####################### Plot 8: Summary Table ###################################
#--------------------------------------------------------------------------------

def create_summary_table(catchment_results, config_registry, plot_dir):
    """
    CSV + matplotlib table figure.
    Rows=configs, columns=catchments with KGE/NSE/KGE_NP sub-columns.
    Bold best per catchment.
    """
    catchment_ids = list(catchment_results.keys())
    catch_names = {gid: catchment_results[gid]['catchment_info']['display_name'] for gid in catchment_ids}
    config_names = {c['key']: c['display_name'] for c in config_registry}

    # Build CSV data
    rows = []
    for cfg in config_registry:
        key = cfg['key']
        row = {'Configuration': config_names[key]}
        has_data = False
        for gid in catchment_ids:
            m = catchment_results[gid]['metrics'].get(key)
            name = catch_names[gid]
            if m:
                row[f'{name} KGE'] = round(m['KGE'], 3)
                row[f'{name} NSE'] = round(m['NSE'], 3)
                row[f'{name} KGE_NP'] = round(m['KGE_NP'], 3)
                has_data = True
            else:
                row[f'{name} KGE'] = ''
                row[f'{name} NSE'] = ''
                row[f'{name} KGE_NP'] = ''
        if has_data:
            rows.append(row)

    if not rows:
        print("No data for summary table")
        return

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = plot_dir / 'cross_catchment_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Find best per catchment for bolding
    best_per_col = {}
    for col in df.columns:
        if col == 'Configuration':
            continue
        numeric_vals = pd.to_numeric(df[col], errors='coerce')
        if numeric_vals.notna().any():
            best_per_col[col] = numeric_vals.idxmax()

    # Create matplotlib table figure
    fig, ax = plt.subplots(figsize=(max(14, len(df.columns) * 1.8), max(6, len(rows) * 0.5)))
    ax.axis('off')

    cell_text = df.values.tolist()
    col_labels = df.columns.tolist()

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Style header
    for j, col in enumerate(col_labels):
        cell = table[0, j]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')

    # Bold best values
    for col_name, best_row_idx in best_per_col.items():
        col_idx = col_labels.index(col_name)
        cell = table[best_row_idx + 1, col_idx]  # +1 for header row
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#D6E4F0')

    # Alternate row colors
    for i in range(len(cell_text)):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            if (i + 1, j) not in [(best_per_col.get(col_labels[j], -1) + 1, j)
                                   for j2 in range(len(col_labels))
                                   if col_labels[j2] in best_per_col]:
                if i % 2 == 0:
                    cell.set_facecolor('#F2F2F2')

    save_path = plot_dir / 'cross_catchment_summary_table.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
############################# Main runner ########################################
#--------------------------------------------------------------------------------

def run_cross_catchment_postprocessing(yaml_path, catchment_filter=None,
                                        config_filter=None,
                                        skip_per_catchment=False,
                                        skip_errors=True,
                                        skip_existing=False):
    """
    Run complete cross-catchment postprocessing.

    Parameters
    ----------
    yaml_path : str or Path
        Path to configurations.yaml
    catchment_filter : list of str, optional
        Only process these gauge_ids
    config_filter : list of str, optional
        Only process these config keys
    skip_per_catchment : bool
        If True, skip per-catchment multi-config postprocessing
    skip_errors : bool
        If True, continue on individual plot failures
    skip_existing : bool
        If True, skip plots whose output files already exist
    """
    start_time = time.time()

    print("=" * 80)
    print("CROSS-CATCHMENT POSTPROCESSING")
    print("=" * 80)

    yaml_path = Path(yaml_path)
    plot_dir = create_cross_catchment_plot_dir(yaml_path)
    print(f"Output directory: {plot_dir}")

    # ========================================================================
    # Phase 1: Per-catchment postprocessing (reuse existing)
    # ========================================================================
    if not skip_per_catchment:
        catchments = load_catchment_registry(yaml_path)
        if catchment_filter:
            catchments = [c for c in catchments if c['gauge_id'] in catchment_filter]

        for catch in catchments:
            gid = catch['gauge_id']
            print(f"\n{'#'*80}")
            print(f"# Per-catchment postprocessing: {gid} ({catch['display_name']})")
            print(f"{'#'*80}")
            try:
                run_complete_multi_postprocessing(
                    str(yaml_path), gid,
                    validation_start=catch['validation_start'],
                    validation_end=catch['validation_end'],
                    skip_errors=skip_errors
                )
            except Exception as e:
                print(f"ERROR in per-catchment postprocessing for {gid}: {e}")
                if not skip_errors:
                    raise

    # ========================================================================
    # Phase 2: Load all catchment data
    # ========================================================================
    print(f"\n{'='*80}")
    print("LOADING DATA FOR CROSS-CATCHMENT ANALYSIS")
    print(f"{'='*80}")

    catchment_results, config_registry = load_all_catchment_data(
        yaml_path, catchment_filter, config_filter
    )

    if not catchment_results:
        print("ERROR: No catchment data loaded")
        return None

    # ========================================================================
    # Phase 3: Cross-catchment plots
    # ========================================================================
    errors = []
    skipped = []

    def run_plot(name, func, output_files, *args, **kwargs):
        if skip_existing and output_files:
            existing = [f for f in output_files if (plot_dir / f).exists()]
            if len(existing) == len(output_files):
                print(f"\n  Skipping {name} (all {len(existing)} output files exist)")
                skipped.append(name)
                return
        print(f"\n{'='*60}")
        print(f"Creating: {name}")
        print(f"{'='*60}")
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            errors.append(name)
            if not skip_errors:
                raise
            import traceback
            traceback.print_exc()

    run_plot("Performance Heatmap",
             plot_performance_heatmap,
             ['cross_catchment_heatmap_KGE.png', 'cross_catchment_heatmap_NSE.png'],
             catchment_results, config_registry, plot_dir)

    run_plot("Configuration Ranking",
             plot_configuration_ranking,
             ['cross_catchment_config_ranking.png'],
             catchment_results, config_registry, plot_dir)

    run_plot("Regime Comparison",
             plot_regime_comparison,
             ['cross_catchment_regime_comparison.png'],
             catchment_results, config_registry, plot_dir)

    run_plot("Glacier Contribution Comparison",
             plot_glacier_contribution_comparison,
             ['cross_catchment_glacier_contribution.png'],
             catchment_results, config_registry, plot_dir)

    run_plot("Coupling Effect (Delta-KGE)",
             plot_coupling_effect,
             ['cross_catchment_coupling_effect.png'],
             catchment_results, config_registry, plot_dir)

    run_plot("Subdaily Effect (Delta-KGE)",
             plot_subdaily_effect,
             ['cross_catchment_subdaily_effect.png'],
             catchment_results, config_registry, plot_dir)

    run_plot("Meteorological Forcing Comparison",
             plot_meteo_forcing_comparison,
             ['cross_catchment_meteo_forcing.png'],
             catchment_results, config_registry, plot_dir)

    run_plot("Coupling Method Comparison",
             plot_coupling_method_comparison,
             ['cross_catchment_coupling_method.png'],
             catchment_results, config_registry, plot_dir)

    run_plot("ET Method Comparison",
             plot_et_method_comparison,
             ['cross_catchment_et_method.png'],
             catchment_results, config_registry, plot_dir)

    run_plot("Landuse Effect",
             plot_landuse_effect,
             ['cross_catchment_landuse_effect.png'],
             catchment_results, config_registry, plot_dir)

    run_plot("Sensitivity Tornado",
             plot_sensitivity_tornado,
             ['cross_catchment_sensitivity_tornado.png'],
             catchment_results, config_registry, plot_dir)

    run_plot("Factor Importance Heatmap",
             plot_factor_importance_heatmap,
             ['cross_catchment_factor_importance.png'],
             catchment_results, config_registry, plot_dir)

    run_plot("KGE Boxplot",
             plot_kge_boxplot,
             ['cross_catchment_kge_boxplot.png'],
             catchment_results, config_registry, plot_dir)

    run_plot("KGE Component Radar",
             plot_kge_component_radar,
             ['cross_catchment_kge_radar.png'],
             catchment_results, config_registry, plot_dir)

    run_plot("Summary Table",
             create_summary_table,
             ['cross_catchment_summary_table.png', 'cross_catchment_summary.csv'],
             catchment_results, config_registry, plot_dir)

    # ========================================================================
    # Summary
    # ========================================================================
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print("CROSS-CATCHMENT POSTPROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Catchments: {list(catchment_results.keys())}")
    print(f"Configurations: {len(config_registry)}")
    print(f"Output: {plot_dir}")
    print(f"Time: {elapsed:.1f} seconds")
    if skipped:
        print(f"Skipped (already exist): {skipped}")
    if errors:
        print(f"Failed plots: {errors}")
    else:
        print("All plots generated successfully")
    print(f"{'='*80}")

    return catchment_results


#--------------------------------------------------------------------------------
################################## CLI ##########################################
#--------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cross-catchment multi-configuration postprocessing')
    parser.add_argument('yaml_path', help='Path to configurations.yaml')
    parser.add_argument('--catchments', nargs='+', default=None,
                        help='Filter to specific gauge_ids (e.g., 0101 0102)')
    parser.add_argument('--configs', nargs='+', default=None,
                        help='Filter to specific config keys (e.g., baseline glogem)')
    parser.add_argument('--skip-per-catchment', action='store_true',
                        help='Skip per-catchment multi-config postprocessing')
    parser.add_argument('--skip-errors', action='store_true', default=True,
                        help='Continue on individual plot failures (default: True)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip plots whose output files already exist')

    args = parser.parse_args()

    run_cross_catchment_postprocessing(
        args.yaml_path,
        catchment_filter=args.catchments,
        config_filter=args.configs,
        skip_per_catchment=args.skip_per_catchment,
        skip_errors=args.skip_errors,
        skip_existing=args.skip_existing,
    )
