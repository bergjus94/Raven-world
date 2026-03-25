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
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=10)
        ax.set_title(f'{metric_name} Across Catchments and Configurations',
                     fontsize=14, fontweight='bold')
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
    ax.set_xticklabels([config_names[k] for k in sorted_keys], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('KGE', fontsize=12)
    ax.set_title('Configuration Ranking by KGE (sorted by mean, black markers)',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Catchment', fontsize=10)
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

        ax.set_title(f"{info['display_name']} ({gid})\n"
                     f"Validation: {info['validation_start']} — {info['validation_end']}",
                     fontsize=11, fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_labels)
        ax.set_xlabel('Month')
        ax.set_ylabel('Discharge (m³/s)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    # Hide unused subplots
    for idx in range(n_catch, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle('Monthly Regime Comparison Across Catchments',
                 fontsize=15, fontweight='bold')
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
    Bar chart of glacier melt fraction of total Q for each catchment,
    grouped by coupled configuration.
    """
    catchment_ids = list(catchment_results.keys())
    catch_names = {gid: catchment_results[gid]['catchment_info']['display_name'] for gid in catchment_ids}
    config_names = {c['key']: c['display_name'] for c in config_registry}
    config_colors_map = {c['key']: c['color'] for c in config_registry}

    # Only coupled configs have glacier contributions
    coupled_keys = [c['key'] for c in config_registry if c['coupled']]

    glacier_fractions = {}  # {config_key: {gauge_id: fraction}}

    for key in coupled_keys:
        glacier_fractions[key] = {}
        for gid in catchment_ids:
            mc = catchment_results[gid].get('multi_config')
            key_to_dir = catchment_results[gid].get('key_to_dir', {})
            config_dir = key_to_dir.get(key)
            if mc is None or config_dir is None:
                continue

            individual_config = _build_individual_config(mc, config_dir)
            try:
                glogem = load_glogem_data(individual_config, plot=False)
                if glogem is None:
                    continue

                info = catchment_results[gid]['catchment_info']
                val_mask = ((glogem['date'] >= info['validation_start']) &
                            (glogem['date'] <= info['validation_end']))
                glogem_val = glogem[val_mask]

                # Glacier melt contribution (catchment-normalized) as fraction of mean obs Q
                mean_glacier_melt = glogem_val['glacier_melt_normalized'].mean()

                # Get mean observed Q in mm/day from regime data
                regime = catchment_results[gid]['regime_data'].get(key)
                if regime is not None and 'obs_Q' in regime.columns:
                    # Mean obs Q is in m3/s, glacier melt is in mm/day
                    # We compare both in mm/day using the glacier_melt_normalized vs sim_Q
                    mean_sim = glogem_val['glacier_melt_normalized'].sum()
                    # Use simulated total Q to get fraction
                    data = load_hydrograph_data(individual_config)
                    if data is not None:
                        val_mask_q = ((data['date'] >= info['validation_start']) &
                                      (data['date'] <= info['validation_end']))
                        mean_obs_q = data[val_mask_q]['obs_Q'].mean()
                        # Convert glacier melt mm/day to m3/s for comparison
                        # Actually, just show the normalized glacier melt as fraction of sim Q
                        mean_sim_q = data[val_mask_q]['sim_Q'].mean()
                        if mean_sim_q > 0:
                            # glacier melt normalized is in mm/day over catchment
                            # sim_Q is in m3/s - need common units
                            # Use the ratio of glogem melt to total sim Q both in mm/day
                            # For simplicity: glacier_melt_normalized / mean_total_glogem_output_normalized
                            total_glogem = (glogem_val['icemelt_normalized'].mean() +
                                           glogem_val['snowmelt_normalized'].mean() +
                                           glogem_val['rainfall_normalized'].mean())
                            if total_glogem > 0:
                                frac = glogem_val['icemelt_normalized'].mean() / total_glogem
                            else:
                                frac = 0
                            glacier_fractions[key][gid] = frac
                        del data

            except Exception as e:
                print(f"  Glacier contribution error for {key}/{gid}: {e}")
                continue

    # Filter to configs that have data for at least one catchment
    valid_keys = [k for k in coupled_keys if glacier_fractions.get(k)]
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
    ax.set_ylabel('Ice Melt Fraction of GloGEM Output (%)', fontsize=11)
    ax.set_title('Glacier Ice Melt Contribution by Catchment and Configuration',
                 fontsize=14, fontweight='bold')
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
    ax.set_xticklabels([label for _, _, label in valid_pairs], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('ΔKGE (coupled − uncoupled)', fontsize=12)
    ax.set_title('Effect of GloGEM Coupling on KGE',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Catchment', fontsize=10)
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
    ax.legend(handles=legend_handles, title='Catchment', fontsize=10, loc='best')

    ax.set_xticks(range(1, len(valid_keys) + 1))
    ax.set_xticklabels(box_labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('KGE', fontsize=12)
    ax.set_title('KGE Distribution Across Catchments per Configuration',
                 fontsize=14, fontweight='bold')
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
        ax.set_title(config_names.get(key, key), fontsize=11, fontweight='bold', pad=20)

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
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, max(2.0, ax.get_ylim()[1]))

        # Reference circle at 1.0 (ideal)
        ideal = [1.0] * n_cats + [1.0]
        ax.plot(angles, ideal, 'k--', linewidth=1, alpha=0.5, label='Ideal')

        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

    # Hide unused subplots
    for idx in range(n_configs, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle('KGE Components Across Catchments', fontsize=15, fontweight='bold')
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

    ax.set_title('Cross-Catchment Performance Summary',
                 fontsize=14, fontweight='bold', pad=20)

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
