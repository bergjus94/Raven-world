# Future ensemble postprocessing for multi-model CMIP6 projections
# March 2026

#--------------------------------------------------------------------------------
################################## packages #####################################
#--------------------------------------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from paths import get_paths

# GloGEM model ID mapping
GLOGEM_MODEL_ID = {
    "GFDL-ESM4": "gfdl-esm4",
    "IPSL-CM6A-LR": "ipsl-cm6a-lr",
    "MPI-ESM1-2-HR": "mpi-esm1-2-hr",
    "MRI-ESM2-0": "mri-esm2-0",
    "UKESM1-0-LL": "ukesm1-0-ll",
}

# Period definitions
PERIODS = {
    'early': ('2015', '2044', 'Early (2015–2044)'),
    'mid':   ('2045', '2074', 'Mid (2045–2074)'),
    'late':  ('2070', '2099', 'Late (2070–2099)'),
}


#--------------------------------------------------------------------------------
################################## data loading #################################
#--------------------------------------------------------------------------------

def load_future_namelist(yaml_path):
    """Load future namelist and return config dict."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def load_ensemble_hydrographs(nml):
    """
    Load Raven hydrograph CSVs for all climate models.

    Returns dict: {model_id: DataFrame with DatetimeIndex and 'Q_sim' column}
    """
    gauge_id = str(nml['gauge_id'])
    model_type = nml['model_type']
    paths = get_paths(nml)
    output_dir = paths['output_dir']

    models = nml.get('cmip6_models', list(GLOGEM_MODEL_ID.keys()))
    all_hydro = {}

    for model_id in models:
        glogem_id = GLOGEM_MODEL_ID.get(model_id, model_id.lower())
        hydro_file = output_dir / glogem_id / f"{gauge_id}_{model_type}_Hydrographs.csv"

        if not hydro_file.exists():
            print(f"  WARNING: Missing hydrograph for {model_id}: {hydro_file}")
            continue

        try:
            df = pd.read_csv(hydro_file, skiprows=[1], parse_dates=['date'])
            sim_cols = [c for c in df.columns if '[m3/s]' in c and 'observed' not in c.lower()]
            if sim_cols:
                all_hydro[model_id] = df[['date', sim_cols[0]]].rename(
                    columns={sim_cols[0]: 'Q_sim'}
                ).set_index('date')
                print(f"  Loaded {model_id}: {len(df)} timesteps")
        except Exception as e:
            print(f"  ERROR loading {model_id}: {e}")

    print(f"  Loaded {len(all_hydro)}/{len(models)} model hydrographs")
    return all_hydro


def load_ensemble_glogem(nml):
    """
    Load GloGEM catchment-averaged data for all climate models.

    Returns dict: {model_id: DataFrame with date column and component columns}
    Component columns: icemelt_all_catchment, snowmelt_all_catchment,
                       rain_all_catchment, melt_all_catchment (all in mm/day)
    """
    gauge_id = str(nml['gauge_id'])
    paths = get_paths(nml)
    topo_dir = paths['topo_dir']

    models = nml.get('cmip6_models', list(GLOGEM_MODEL_ID.keys()))
    all_glogem = {}

    glogem_file = topo_dir / 'GloGEM_catchment_averaged.csv'

    if not glogem_file.exists():
        print(f"  WARNING: GloGEM file not found: {glogem_file}")
        for model_id in models:
            glogem_id = GLOGEM_MODEL_ID.get(model_id, model_id.lower())
            model_glogem = topo_dir / f'GloGEM_catchment_averaged_{glogem_id}.csv'
            if model_glogem.exists():
                try:
                    df = pd.read_csv(model_glogem, parse_dates=['date'])
                    all_glogem[model_id] = df
                    print(f"  Loaded GloGEM for {model_id}: {len(df)} records")
                except Exception as e:
                    print(f"  ERROR loading GloGEM for {model_id}: {e}")
        return all_glogem

    # Single file — the GloGEM data is the same file regenerated per model run
    # For future runs, each model has its own GloGEM processing
    # Check if there are model-specific files first
    for model_id in models:
        glogem_id = GLOGEM_MODEL_ID.get(model_id, model_id.lower())
        model_glogem = topo_dir / f'GloGEM_catchment_averaged_{glogem_id}.csv'

        if model_glogem.exists():
            glogem_path = model_glogem
        else:
            glogem_path = glogem_file  # Fallback to shared file

        try:
            df = pd.read_csv(glogem_path, parse_dates=['date'])
            all_glogem[model_id] = df
            print(f"  Loaded GloGEM for {model_id}: {len(df)} records")
        except Exception as e:
            print(f"  ERROR loading GloGEM for {model_id}: {e}")

    print(f"  Loaded {len(all_glogem)}/{len(models)} GloGEM datasets")
    return all_glogem


def load_historical_observed(nml):
    """
    Load historical observed discharge from the calibration config.

    Returns DataFrame with DatetimeIndex and 'Q_obs' column, or None.
    """
    gauge_id = str(nml['gauge_id'])
    model_type = nml['model_type']
    hist_nml = dict(nml)
    if 'config_dir' in hist_nml:
        hist_nml['config_dir'] = hist_nml['config_dir'].replace('_future', '')
    if '_config_key' in hist_nml:
        hist_nml['_config_key'] = hist_nml['_config_key'].replace('_future', '')
    hist_paths = get_paths(hist_nml)
    hist_output = hist_paths['output_dir'] / f"{gauge_id}_{model_type}_Hydrographs.csv"

    if not hist_output.exists():
        print(f"  No historical hydrograph found: {hist_output}")
        return None

    try:
        df = pd.read_csv(hist_output, skiprows=[1], parse_dates=['date'])
        obs_cols = [c for c in df.columns if '[m3/s]' in c and 'observed' in c.lower()]
        sim_cols = [c for c in df.columns if '[m3/s]' in c and 'observed' not in c.lower()]
        result = df[['date']].copy()
        if obs_cols:
            result['Q_obs'] = pd.to_numeric(df[obs_cols[0]], errors='coerce')
        if sim_cols:
            result['Q_sim_hist'] = pd.to_numeric(df[sim_cols[0]], errors='coerce')
        result = result.set_index('date')
        print(f"  Loaded historical data: {len(result)} timesteps")
        return result
    except Exception as e:
        print(f"  ERROR loading historical data: {e}")
        return None


def create_future_plot_dir(nml):
    """Create output directory for future ensemble plots."""
    paths = get_paths(nml)
    plot_dir = paths['output_dir'] / 'future_plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


#--------------------------------------------------------------------------------
######################### helper functions ######################################
#--------------------------------------------------------------------------------

def _ensemble_stats(all_hydro, col='Q_sim'):
    """Compute ensemble mean, min, max from dict of DataFrames."""
    combined = pd.concat({k: v[col] for k, v in all_hydro.items()}, axis=1)
    return pd.DataFrame({
        'mean': combined.mean(axis=1),
        'min': combined.min(axis=1),
        'max': combined.max(axis=1),
        'std': combined.std(axis=1),
    })

def _monthly_regime(df, col, start=None, end=None):
    """Compute monthly mean regime for a period."""
    if start and end:
        df = df.loc[start:end]
    return df[col].groupby(df.index.month).mean()


MODEL_COLORS = {
    'GFDL-ESM4': '#1f77b4',
    'IPSL-CM6A-LR': '#ff7f0e',
    'MPI-ESM1-2-HR': '#2ca02c',
    'MRI-ESM2-0': '#d62728',
    'UKESM1-0-LL': '#9467bd',
}

MONTH_LABELS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']


#--------------------------------------------------------------------------------
#################### Plot 1: Ensemble Envelope ##################################
#--------------------------------------------------------------------------------

def plot_ensemble_envelope(all_hydro, gauge_id, plot_dir):
    """
    Time series with ensemble mean line + min/max shading.
    Individual model lines shown faintly underneath.
    """
    if not all_hydro:
        return

    stats = _ensemble_stats(all_hydro)
    # Resample to monthly for cleaner visualization
    stats_monthly = stats.resample('ME').mean()

    fig, ax = plt.subplots(figsize=(16, 6))

    # Shading
    ax.fill_between(stats_monthly.index, stats_monthly['min'], stats_monthly['max'],
                     alpha=0.2, color='steelblue', label='Model range')
    # Ensemble mean
    ax.plot(stats_monthly.index, stats_monthly['mean'], color='steelblue',
            linewidth=2, label='Ensemble mean')

    # Individual models (faint)
    for model_id, df in all_hydro.items():
        monthly = df['Q_sim'].resample('ME').mean()
        ax.plot(monthly.index, monthly.values, alpha=0.3, linewidth=0.5,
                color=MODEL_COLORS.get(model_id, 'gray'))

    ax.set_ylabel('Discharge (m³/s)', fontsize=12)
    ax.set_title(f'Catchment {gauge_id} — Future Ensemble Envelope (monthly mean)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = plot_dir / f'ensemble_envelope_{gauge_id}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
#################### Plot 2: 30-Year Moving Average #############################
#--------------------------------------------------------------------------------

def plot_moving_average(all_hydro, gauge_id, plot_dir, window=30):
    """
    30-year moving average with ensemble spread.
    Shows long-term trend without interannual noise.
    """
    if not all_hydro:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    annual_dfs = {}
    for model_id, df in all_hydro.items():
        annual = df['Q_sim'].resample('YE').mean()
        ma = annual.rolling(window=window, center=True, min_periods=15).mean()
        annual_dfs[model_id] = ma
        ax.plot(ma.index, ma.values, alpha=0.4, linewidth=1,
                color=MODEL_COLORS.get(model_id, 'gray'), label=model_id)

    # Ensemble mean moving average
    combined = pd.concat(annual_dfs, axis=1)
    ens_mean = combined.mean(axis=1)
    ens_min = combined.min(axis=1)
    ens_max = combined.max(axis=1)

    ax.fill_between(ens_mean.index, ens_min, ens_max, alpha=0.15, color='steelblue')
    ax.plot(ens_mean.index, ens_mean.values, color='steelblue', linewidth=2.5,
            label='Ensemble mean')

    ax.set_ylabel('Discharge (m³/s)', fontsize=12)
    ax.set_title(f'Catchment {gauge_id} — {window}-Year Moving Average',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = plot_dir / f'moving_average_{window}yr_{gauge_id}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
################# Plot 3: Regime by Period ######################################
#--------------------------------------------------------------------------------

def plot_regime_by_period(all_hydro, gauge_id, plot_dir, hist_data=None):
    """
    Monthly regime for 3 periods (early/mid/late century).
    Each panel shows ensemble mean + spread, with historical observed if available.
    """
    if not all_hydro:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (period_key, (start, end, label)) in zip(axes, PERIODS.items()):
        # Ensemble regime for this period
        regimes = {}
        for model_id, df in all_hydro.items():
            period_df = df.loc[start:end]
            if len(period_df) > 0:
                regimes[model_id] = period_df['Q_sim'].groupby(period_df.index.month).mean()

        if not regimes:
            ax.set_title(f'{label}\n(no data)')
            continue

        combined = pd.DataFrame(regimes)
        ens_mean = combined.mean(axis=1)
        ens_min = combined.min(axis=1)
        ens_max = combined.max(axis=1)

        ax.fill_between(ens_mean.index, ens_min, ens_max,
                         alpha=0.2, color='steelblue', label='Model range')
        ax.plot(ens_mean.index, ens_mean.values, color='steelblue',
                linewidth=2.5, label='Ensemble mean')

        # Individual models
        for model_id, regime in regimes.items():
            ax.plot(regime.index, regime.values, alpha=0.3, linewidth=1,
                    color=MODEL_COLORS.get(model_id, 'gray'))

        # Historical observed
        if hist_data is not None and 'Q_obs' in hist_data.columns:
            obs_clean = hist_data['Q_obs'].dropna()
            obs_regime = obs_clean.groupby(obs_clean.index.month).mean()
            ax.plot(obs_regime.index, obs_regime.values, 'k-', linewidth=2,
                    label='Historical observed')

        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_LABELS)
        ax.set_xlabel('Month')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Discharge (m³/s)', fontsize=12)
    axes[0].legend(fontsize=9)

    fig.suptitle(f'Catchment {gauge_id} — Monthly Regime by Period',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = plot_dir / f'regime_by_period_{gauge_id}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
################# Plot 4: Regime Shift vs Historical ############################
#--------------------------------------------------------------------------------

def plot_regime_shift(all_hydro, gauge_id, plot_dir, hist_data=None):
    """
    Overlay historical observed regime with future ensemble mean regimes
    for early and late century. Shows how the hydrograph shape changes.
    """
    if not all_hydro:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    period_colors = {'early': '#2ca02c', 'late': '#d62728'}

    for period_key, (start, end, label) in PERIODS.items():
        if period_key == 'mid':
            continue  # Skip mid for clarity

        regimes = {}
        for model_id, df in all_hydro.items():
            period_df = df.loc[start:end]
            if len(period_df) > 0:
                regimes[model_id] = period_df['Q_sim'].groupby(period_df.index.month).mean()

        if not regimes:
            continue

        combined = pd.DataFrame(regimes)
        ens_mean = combined.mean(axis=1)
        ens_min = combined.min(axis=1)
        ens_max = combined.max(axis=1)

        color = period_colors.get(period_key, 'gray')
        ax.fill_between(ens_mean.index, ens_min, ens_max, alpha=0.15, color=color)
        ax.plot(ens_mean.index, ens_mean.values, color=color,
                linewidth=2.5, label=f'Future {label}')

    # Historical
    if hist_data is not None and 'Q_obs' in hist_data.columns:
        obs_clean = hist_data['Q_obs'].dropna()
        obs_regime = obs_clean.groupby(obs_clean.index.month).mean()
        ax.plot(obs_regime.index, obs_regime.values, 'k-', linewidth=2.5,
                label='Historical observed')

    if hist_data is not None and 'Q_sim_hist' in hist_data.columns:
        sim_clean = hist_data['Q_sim_hist'].dropna()
        sim_regime = sim_clean.groupby(sim_clean.index.month).mean()
        ax.plot(sim_regime.index, sim_regime.values, 'k--', linewidth=1.5,
                label='Historical simulated')

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Discharge (m³/s)', fontsize=12)
    ax.set_title(f'Catchment {gauge_id} — Regime Shift: Historical vs Future',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = plot_dir / f'regime_shift_{gauge_id}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
################# Plot 5: Flow Duration Curves ##################################
#--------------------------------------------------------------------------------

def plot_flow_duration_curves(all_hydro, gauge_id, plot_dir, hist_data=None):
    """
    Flow duration curves for historical vs future periods.
    Shows changes in high/low flow frequency.
    """
    if not all_hydro:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    period_colors = {'early': '#2ca02c', 'mid': '#ff7f0e', 'late': '#d62728'}

    for period_key, (start, end, label) in PERIODS.items():
        all_sorted = []
        for model_id, df in all_hydro.items():
            period_df = df.loc[start:end]
            if len(period_df) > 0:
                sorted_q = np.sort(period_df['Q_sim'].dropna().values)[::-1]
                exceedance = np.arange(1, len(sorted_q) + 1) / len(sorted_q) * 100
                all_sorted.append(np.interp(np.linspace(0, 100, 1000),
                                             exceedance, sorted_q))

        if all_sorted:
            combined = np.array(all_sorted)
            ens_mean = combined.mean(axis=0)
            ens_min = combined.min(axis=0)
            ens_max = combined.max(axis=0)
            x = np.linspace(0, 100, 1000)

            color = period_colors.get(period_key, 'gray')
            ax.fill_between(x, ens_min, ens_max, alpha=0.1, color=color)
            ax.plot(x, ens_mean, color=color, linewidth=2, label=f'Future {label}')

    # Historical observed
    if hist_data is not None and 'Q_obs' in hist_data.columns:
        obs = hist_data['Q_obs'].dropna().values
        sorted_obs = np.sort(obs)[::-1]
        exceedance = np.arange(1, len(sorted_obs) + 1) / len(sorted_obs) * 100
        ax.plot(exceedance, sorted_obs, 'k-', linewidth=2, label='Historical observed')

    ax.set_xlabel('Exceedance Probability (%)', fontsize=12)
    ax.set_ylabel('Discharge (m³/s)', fontsize=12)
    ax.set_title(f'Catchment {gauge_id} — Flow Duration Curves',
                 fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    save_path = plot_dir / f'flow_duration_curves_{gauge_id}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
################# Plot 6: Seasonal Water Availability ###########################
#--------------------------------------------------------------------------------

def plot_seasonal_water_availability(all_hydro, gauge_id, plot_dir):
    """
    Stacked area: seasonal contribution (DJF, MAM, JJAS, ON) as % of annual,
    how this shifts over time (decadal).
    """
    if not all_hydro:
        return

    seasons = {
        'Winter (DJF)': [12, 1, 2],
        'Pre-monsoon (MAM)': [3, 4, 5],
        'Monsoon (JJAS)': [6, 7, 8, 9],
        'Post-monsoon (ON)': [10, 11],
    }
    season_colors = ['#3182bd', '#31a354', '#e6550d', '#756bb1']

    # Compute decadal seasonal fractions (ensemble mean)
    decades = list(range(2010, 2100, 10))
    decade_labels = [f'{d}s' for d in decades]

    season_fractions = {name: [] for name in seasons}

    for decade_start in decades:
        decade_end = decade_start + 9
        start_str = str(decade_start)
        end_str = str(decade_end)

        decade_totals = {name: [] for name in seasons}

        for model_id, df in all_hydro.items():
            dec_df = df.loc[start_str:end_str]
            if len(dec_df) == 0:
                continue
            annual_mean = dec_df['Q_sim'].mean()
            if annual_mean <= 0:
                continue

            for name, months in seasons.items():
                seasonal_mean = dec_df[dec_df.index.month.isin(months)]['Q_sim'].mean()
                decade_totals[name].append(seasonal_mean / annual_mean * 100)

        for name in seasons:
            vals = decade_totals[name]
            season_fractions[name].append(np.mean(vals) if vals else 0)

    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(len(decades))
    for (name, fracs), color in zip(season_fractions.items(), season_colors):
        fracs = np.array(fracs)
        ax.bar(range(len(decades)), fracs, bottom=bottom, color=color,
               label=name, edgecolor='white', linewidth=0.5)
        bottom += fracs

    ax.set_xticks(range(len(decades)))
    ax.set_xticklabels(decade_labels, rotation=45)
    ax.set_ylabel('Share of Annual Discharge (%)', fontsize=12)
    ax.set_title(f'Catchment {gauge_id} — Seasonal Water Availability Over Time',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = plot_dir / f'seasonal_water_availability_{gauge_id}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
############# Plot 7: Stacked Regime Decomposition ##############################
#--------------------------------------------------------------------------------

def plot_stacked_regime_decomposition(all_hydro, all_glogem, gauge_id, plot_dir,
                                       hist_data=None):
    """
    Monthly regime decomposed into: ice melt, snowmelt, rain (from GloGEM),
    one panel per period. Shows how the glacier subsidy shrinks.
    """
    if not all_glogem:
        print("  No GloGEM data for regime decomposition")
        return

    components = [
        ('icemelt_all_catchment', 'Ice Melt', '#4393c3'),
        ('snowmelt_all_catchment', 'Snowmelt', '#92c5de'),
        ('rain_all_catchment', 'Rainfall', '#d6604d'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (period_key, (start, end, label)) in zip(axes, PERIODS.items()):
        component_regimes = {name: [] for _, name, _ in components}

        for model_id, gdf in all_glogem.items():
            gdf_indexed = gdf.set_index('date') if 'date' in gdf.columns else gdf
            period_gdf = gdf_indexed.loc[start:end]
            if len(period_gdf) == 0:
                continue

            for col, name, _ in components:
                if col in period_gdf.columns:
                    monthly = period_gdf[col].groupby(period_gdf.index.month).mean()
                    component_regimes[name].append(monthly)

        months = np.arange(1, 13)
        bottom = np.zeros(12)

        for col, name, color in components:
            if component_regimes[name]:
                combined = pd.DataFrame(component_regimes[name]).mean()
                vals = combined.values if len(combined) == 12 else np.zeros(12)
                ax.bar(months, vals, bottom=bottom, color=color, label=name,
                       edgecolor='white', linewidth=0.3, width=0.8)
                bottom += vals

        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_LABELS)
        ax.set_xlabel('Month')
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel('GloGEM Output (mm/day, catchment area)', fontsize=11)
    axes[0].legend(fontsize=9)

    fig.suptitle(f'Catchment {gauge_id} — Regime Decomposition by Period',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = plot_dir / f'regime_decomposition_{gauge_id}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
############# Plot 8: Component Regime Shift ####################################
#--------------------------------------------------------------------------------

def plot_component_regime_shift(all_glogem, gauge_id, plot_dir):
    """
    Overlay ice melt and snowmelt monthly curves for early vs late century.
    Shows timing shifts.
    """
    if not all_glogem:
        return

    components = [
        ('icemelt_all_catchment', 'Ice Melt'),
        ('snowmelt_all_catchment', 'Snowmelt'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, (col, comp_name) in zip(axes, components):
        for period_key in ['early', 'late']:
            start, end, label = PERIODS[period_key]
            color = '#2ca02c' if period_key == 'early' else '#d62728'

            regimes = []
            for model_id, gdf in all_glogem.items():
                gdf_indexed = gdf.set_index('date') if 'date' in gdf.columns else gdf
                period_gdf = gdf_indexed.loc[start:end]
                if len(period_gdf) > 0 and col in period_gdf.columns:
                    monthly = period_gdf[col].groupby(period_gdf.index.month).mean()
                    regimes.append(monthly)

            if regimes:
                combined = pd.DataFrame(regimes)
                ens_mean = combined.mean()
                ens_min = combined.min()
                ens_max = combined.max()

                ax.fill_between(ens_mean.index, ens_min, ens_max, alpha=0.15, color=color)
                ax.plot(ens_mean.index, ens_mean.values, color=color,
                        linewidth=2.5, label=label)

        ax.set_title(comp_name, fontsize=12, fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_LABELS)
        ax.set_xlabel('Month')
        ax.set_ylabel('mm/day (catchment area)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Catchment {gauge_id} — Component Regime Shift',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = plot_dir / f'component_regime_shift_{gauge_id}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
############# Plot 9: Ice Melt Contribution Trend (Peak Water) ##################
#--------------------------------------------------------------------------------

def plot_ice_melt_trend(all_hydro, all_glogem, gauge_id, plot_dir):
    """
    Annual ice melt as % of total discharge over time.
    The key 'peak water' plot — shows when glacier contribution peaks and declines.
    """
    if not all_glogem or not all_hydro:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    for model_id in all_glogem:
        if model_id not in all_hydro:
            continue

        gdf = all_glogem[model_id]
        gdf_indexed = gdf.set_index('date') if 'date' in gdf.columns else gdf

        if 'melt_all_catchment' not in gdf_indexed.columns:
            continue

        # Annual glacier melt (catchment-normalized, mm/day → mm/year via mean*365)
        annual_melt = gdf_indexed['icemelt_all_catchment'].resample('YE').mean()

        # Annual total GloGEM output
        annual_total = gdf_indexed['melt_all_catchment'].resample('YE').mean()

        # Ice melt as fraction of total GloGEM output
        fraction = (annual_melt / annual_total * 100).replace([np.inf, -np.inf], np.nan)

        # Smooth with 10-year moving average
        fraction_smooth = fraction.rolling(window=10, center=True, min_periods=5).mean()

        color = MODEL_COLORS.get(model_id, 'gray')
        ax.plot(fraction_smooth.index, fraction_smooth.values, color=color,
                linewidth=1.5, alpha=0.7, label=model_id)

    ax.set_ylabel('Ice Melt / Total GloGEM Output (%)', fontsize=12)
    ax.set_title(f'Catchment {gauge_id} — Ice Melt Contribution Trend (10-yr smoothed)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = plot_dir / f'ice_melt_trend_{gauge_id}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
############# Plot 10: Dry Season Glacier Dependency ############################
#--------------------------------------------------------------------------------

def plot_dry_season_dependency(all_glogem, gauge_id, plot_dir):
    """
    Glacier melt as % of total GloGEM output during Oct-Mar (dry season),
    trending over time. Shows increasing vulnerability as glaciers retreat.
    """
    if not all_glogem:
        return

    dry_months = [10, 11, 12, 1, 2, 3]

    fig, ax = plt.subplots(figsize=(14, 6))

    for model_id, gdf in all_glogem.items():
        gdf_indexed = gdf.set_index('date') if 'date' in gdf.columns else gdf

        if 'icemelt_all_catchment' not in gdf_indexed.columns:
            continue

        dry = gdf_indexed[gdf_indexed.index.month.isin(dry_months)]

        annual_icemelt = dry['icemelt_all_catchment'].resample('YE').mean()
        annual_total = dry['melt_all_catchment'].resample('YE').mean()

        fraction = (annual_icemelt / annual_total * 100).replace([np.inf, -np.inf], np.nan)
        fraction_smooth = fraction.rolling(window=10, center=True, min_periods=5).mean()

        color = MODEL_COLORS.get(model_id, 'gray')
        ax.plot(fraction_smooth.index, fraction_smooth.values, color=color,
                linewidth=1.5, alpha=0.7, label=model_id)

    ax.set_ylabel('Ice Melt / Total GloGEM (dry season, %)', fontsize=12)
    ax.set_title(f'Catchment {gauge_id} — Dry Season Glacier Dependency (Oct–Mar, 10-yr smoothed)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = plot_dir / f'dry_season_dependency_{gauge_id}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
############# Plot 11: Annual Max/Min Trends ####################################
#--------------------------------------------------------------------------------

def plot_annual_extremes(all_hydro, gauge_id, plot_dir):
    """
    Trend in annual max and min monthly discharge across ensemble.
    Shows changes in flood risk and low flow.
    """
    if not all_hydro:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for model_id, df in all_hydro.items():
        color = MODEL_COLORS.get(model_id, 'gray')
        monthly = df['Q_sim'].resample('ME').mean()

        annual_max = monthly.resample('YE').max()
        annual_min = monthly.resample('YE').min()

        axes[0].plot(annual_max.index, annual_max.values, color=color,
                     alpha=0.5, linewidth=1)
        axes[1].plot(annual_min.index, annual_min.values, color=color,
                     alpha=0.5, linewidth=1, label=model_id)

    # Ensemble mean
    for ax_idx, agg_func in enumerate(['max', 'min']):
        combined = []
        for model_id, df in all_hydro.items():
            monthly = df['Q_sim'].resample('ME').mean()
            annual = getattr(monthly.resample('YE'), agg_func)()
            combined.append(annual)
        ens = pd.concat(combined, axis=1)
        ens_mean = ens.mean(axis=1)
        ens_smooth = ens_mean.rolling(window=10, center=True, min_periods=5).mean()
        axes[ax_idx].plot(ens_smooth.index, ens_smooth.values, 'k-', linewidth=2.5,
                          label='Ensemble mean (10-yr)')

    axes[0].set_ylabel('Annual Max Monthly Q (m³/s)', fontsize=11)
    axes[0].set_title(f'Catchment {gauge_id} — Annual Peak Flow Trend', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel('Annual Min Monthly Q (m³/s)', fontsize=11)
    axes[1].set_title('Annual Low Flow Trend', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = plot_dir / f'annual_extremes_{gauge_id}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


#--------------------------------------------------------------------------------
############################# Main runner ########################################
#--------------------------------------------------------------------------------

def run_future_postprocessing(yaml_path, skip_errors=True):
    """
    Run complete future ensemble postprocessing.
    """
    start_time = time.time()

    print("=" * 80)
    print("FUTURE ENSEMBLE POSTPROCESSING")
    print("=" * 80)

    nml = load_future_namelist(yaml_path)
    gauge_id = str(nml['gauge_id'])
    plot_dir = create_future_plot_dir(nml)
    print(f"Catchment: {gauge_id}")
    print(f"Output: {plot_dir}")

    # Load data
    print("\nLoading ensemble hydrographs...")
    all_hydro = load_ensemble_hydrographs(nml)

    print("\nLoading GloGEM data...")
    all_glogem = load_ensemble_glogem(nml)

    print("\nLoading historical data...")
    hist_data = load_historical_observed(nml)

    if not all_hydro:
        print("ERROR: No hydrograph data loaded")
        return None

    # Generate plots
    errors = []

    def run_plot(name, func, *args, **kwargs):
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

    # Discharge plots
    run_plot("Ensemble Envelope",
             plot_ensemble_envelope, all_hydro, gauge_id, plot_dir)

    run_plot("30-Year Moving Average",
             plot_moving_average, all_hydro, gauge_id, plot_dir)

    run_plot("Regime by Period",
             plot_regime_by_period, all_hydro, gauge_id, plot_dir, hist_data)

    run_plot("Regime Shift vs Historical",
             plot_regime_shift, all_hydro, gauge_id, plot_dir, hist_data)

    run_plot("Flow Duration Curves",
             plot_flow_duration_curves, all_hydro, gauge_id, plot_dir, hist_data)

    run_plot("Seasonal Water Availability",
             plot_seasonal_water_availability, all_hydro, gauge_id, plot_dir)

    run_plot("Annual Extremes",
             plot_annual_extremes, all_hydro, gauge_id, plot_dir)

    # Component / glacier plots
    run_plot("Stacked Regime Decomposition",
             plot_stacked_regime_decomposition, all_hydro, all_glogem, gauge_id, plot_dir, hist_data)

    run_plot("Component Regime Shift",
             plot_component_regime_shift, all_glogem, gauge_id, plot_dir)

    run_plot("Ice Melt Contribution Trend",
             plot_ice_melt_trend, all_hydro, all_glogem, gauge_id, plot_dir)

    run_plot("Dry Season Glacier Dependency",
             plot_dry_season_dependency, all_glogem, gauge_id, plot_dir)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print("FUTURE ENSEMBLE POSTPROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Catchment: {gauge_id}")
    print(f"Models: {len(all_hydro)} hydrographs, {len(all_glogem)} GloGEM")
    print(f"Output: {plot_dir}")
    print(f"Time: {elapsed:.1f} seconds")
    if errors:
        print(f"Failed plots: {errors}")
    else:
        print("All plots generated successfully")
    print(f"{'='*80}")

    return all_hydro


#--------------------------------------------------------------------------------
################################## CLI ##########################################
#--------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Future ensemble postprocessing for CMIP6 projections')
    parser.add_argument('yaml_path', help='Path to future namelist YAML')
    parser.add_argument('--skip-errors', action='store_true', default=True,
                        help='Continue on individual plot failures (default: True)')

    args = parser.parse_args()
    run_future_postprocessing(args.yaml_path, skip_errors=args.skip_errors)
