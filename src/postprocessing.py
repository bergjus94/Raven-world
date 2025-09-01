# This script is postprocessing Raven output from multiple model configurations to compare output
# July 2025

#--------------------------------------------------------------------------------
################################## packages ######################################
#--------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import re
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import glob
from matplotlib import cm
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib.dates import DateFormatter
import math
from typing import Tuple, List, Dict
import geopandas as gpd
import geopandas as gpd
from datetime import datetime, timedelta


#--------------------------------------------------------------------------------
################################### general #####################################
#--------------------------------------------------------------------------------

def get_available_catchments(base_dir, configs):
    """
    Get all available catchment IDs by scanning the base directory
    for folders matching the pattern 'catchment_XXXX_*'.
    
    Parameters
    ----------
    base_dir : str
        Base directory path
    configs : list
        List of configuration names to check
        
    Returns
    -------
    list
        List of unique catchment IDs found
    """
    catchment_ids = set()
    
    # Convert base_dir to Path object
    base_path = Path(base_dir)
    
    # Look for directories matching the pattern
    for config in configs:
        pattern = f"catchment_*_{config}"
        matching_dirs = list(base_path.glob(pattern))
        
        for dir_path in matching_dirs:
            # Extract catchment ID from directory name
            match = re.search(r'catchment_(\d+)_', dir_path.name)
            if match:
                catchment_id = match.group(1)
                catchment_ids.add(catchment_id)
    
    print(f"Found {len(catchment_ids)} catchments: {sorted(catchment_ids)}")
    return sorted(list(catchment_ids))

#--------------------------------------------------------------------------------
################################## hydrograph ###################################
#--------------------------------------------------------------------------------

def load_hydrograph_data(gauge_id, config, base_dir):
    """Load hydrograph data from a specific configuration"""
    # Path for hydrograph file
    model_dir = Path(base_dir) / f"catchment_{gauge_id}_{config}" / "HBV"
    hydro_file = model_dir / "output" / f"{gauge_id}_HBV_Hydrographs.csv"
    
    print(f"Loading hydrograph data for {config}:")
    print(f"  - File: {hydro_file}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(hydro_file)
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Identify the simulated and observed columns
        sim_col = None
        obs_col = None
        
        # Look for columns matching the pattern for simulated and observed flow
        for col in df.columns:
            if '[m3/s]' in col and 'observed' not in col.lower():
                sim_col = col
            elif '[m3/s]' in col and 'observed' in col.lower():
                obs_col = col
        
        if not sim_col:
            # Try alternative column naming patterns
            for col in df.columns:
                if col.endswith('[m3/s]') and not col.endswith('(observed) [m3/s]'):
                    sim_col = col
                elif col.endswith('(observed) [m3/s]'):
                    obs_col = col
        
        # Rename columns for consistency
        renamed_df = df.copy()
        if sim_col:
            renamed_df['sim_Q'] = df[sim_col]
        if obs_col:
            renamed_df['obs_Q'] = df[obs_col]
            
        print(f"  - Found columns: sim={sim_col}, obs={obs_col}")
        print(f"  - Data range: {renamed_df['date'].min()} to {renamed_df['date'].max()}")
        
        return renamed_df
    
    except Exception as e:
        print(f"  - Error loading data: {e}")
        return None
    
#---------------------------------------------------------------------------------


def plot_hydrological_regime_all_configs(
    gauge_id, configs, base_dir, 
    validation_start='2010-01-01', validation_end='2020-12-31',
    config_colors=None, config_names=None):
    """
    Plot the hydrological regime (monthly mean) for all configurations
    for the validation period
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Load data for all configurations
    all_data = {}
    for config in configs:
        data = load_hydrograph_data(gauge_id, config, base_dir)
        if data is not None:
            all_data[config] = data

    if not all_data:
        print("No data loaded for any configuration")
        return

    # Create a DataFrame for the monthly means
    monthly_data = {}

    # Get monthly means for each configuration
    for config, data in all_data.items():
        validation_mask = (data['date'] >= validation_start) & (data['date'] <= validation_end)
        df_validation = data[validation_mask].copy()

        if len(df_validation) == 0:
            print(f"Warning: No data found for validation period {validation_start} to {validation_end} in {config}")
            continue

        df_validation['month'] = df_validation['date'].dt.month

        if 'sim_Q' in df_validation.columns:
            monthly_means = df_validation.groupby('month')['sim_Q'].mean()
            monthly_data[f'sim_Q_{config}'] = monthly_means

        if 'obs_Q' in df_validation.columns and 'obs_Q' not in monthly_data:
            monthly_means = df_validation.groupby('month')['obs_Q'].mean()
            monthly_data['obs_Q'] = monthly_means

    monthly_df = pd.DataFrame(monthly_data)

    # Plotting
    plt.figure(figsize=(12, 7))

    # Plot observed data if available
    if 'obs_Q' in monthly_df.columns:
        plt.plot(monthly_df.index, monthly_df['obs_Q'], 'k-', linewidth=2.5, label='Observed')

    # Plot simulated data for each configuration
    sim_cols = [col for col in monthly_df.columns if col.startswith('sim_Q_')]

    # Handle flexible color and name mapping
    if config_colors is None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        config_colors = {cfg: color_cycle[i % len(color_cycle)] for i, cfg in enumerate(configs)}
    if config_names is None:
        config_names = {cfg: cfg for cfg in configs}

    for i, col in enumerate(sim_cols):
        config_name = col.replace('sim_Q_', '')
        color = config_colors.get(config_name, f"C{i}")
        label = config_names.get(config_name, config_name)
        plt.plot(monthly_df.index, monthly_df[col], color=color, linewidth=2, label=f'{label}')

    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Discharge (m³/s)', fontsize=14)
    plt.title(f'Hydrological Regime - Monthly Mean for Validation Period ({validation_start} to {validation_end})\nCatchment {gauge_id}', fontsize=16)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)

    # Add performance metrics text if available
    perf_text = "Validation Period Performance:\n"

    for config, data in all_data.items():
        validation_mask = (data['date'] >= validation_start) & (data['date'] <= validation_end)
        val_data = data[validation_mask]

        if 'obs_Q' in val_data.columns and 'sim_Q' in val_data.columns:
            obs = val_data['obs_Q'].values
            sim = val_data['sim_Q'].values
            obs_mean = np.mean(obs)
            nse = 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - obs_mean) ** 2))

            mean_sim = np.mean(sim)
            mean_obs = np.mean(obs)
            std_sim = np.std(sim)
            std_obs = np.std(obs)
            corr = np.corrcoef(sim, obs)[0, 1]
            alpha = std_sim / std_obs
            beta = mean_sim / mean_obs
            kge = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

            perf_text += f"{config_names.get(config, config)}: NSE={nse:.3f}, KGE={kge:.3f}\n"

    plt.figtext(0.02, 0.02, perf_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'hydrological_regime_{gauge_id}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return monthly_df

#--------------------------------------------------------------------------------

def plot_hydrological_regime_subplots(
    gauge_id, configs, base_dir, 
    validation_start='2010-01-01', validation_end='2020-12-31',
    config_colors=None, config_names=None):
    """
    Plot the hydrological regime (monthly mean) with a subplot for each configuration
    during the validation period.

    Parameters
    ----------
    gauge_id : str
        ID of the gauge to analyze
    configs : list
        List of configuration names
    base_dir : str
        Base directory containing model outputs
    validation_start : str
        Start date for validation period
    validation_end : str
        End date for validation period
    config_colors : dict, optional
        Dict mapping config name to color (e.g. {'nc': '#712423', ...})
    config_names : dict, optional
        Dict mapping config name to display name (e.g. {'nc': 'HBV', ...})

    Returns
    -------
    pandas.DataFrame
        DataFrame containing monthly mean values
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import os

    # Load data for all configurations
    all_data = {}
    for config in configs:
        data = load_hydrograph_data(gauge_id, config, base_dir)
        if data is not None:
            all_data[config] = data

    if not all_data:
        print("No data loaded for any configuration")
        return

    # Create a DataFrame for the monthly means
    monthly_data = {}

    # Get monthly means for each configuration
    for config, data in all_data.items():
        validation_mask = (data['date'] >= validation_start) & (data['date'] <= validation_end)
        df_validation = data[validation_mask].copy()

        if len(df_validation) == 0:
            print(f"Warning: No data found for validation period {validation_start} to {validation_end} in {config}")
            continue

        df_validation['month'] = df_validation['date'].dt.month

        if 'sim_Q' in df_validation.columns:
            monthly_means = df_validation.groupby('month')['sim_Q'].mean()
            monthly_data[f'sim_Q_{config}'] = monthly_means

        if 'obs_Q' in df_validation.columns and 'obs_Q' not in monthly_data:
            monthly_means = df_validation.groupby('month')['obs_Q'].mean()
            monthly_data['obs_Q'] = monthly_means

    monthly_df = pd.DataFrame(monthly_data)

    n_configs = len([col for col in monthly_df.columns if col.startswith('sim_Q_')])
    if n_configs == 0:
        print("No configuration data available for plotting")
        return monthly_df

    n_cols = min(2, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), sharex=True, sharey=True)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Handle flexible color and name mapping
    if config_colors is None:
        # Use matplotlib default color cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        config_colors = {cfg: color_cycle[i % len(color_cycle)] for i, cfg in enumerate(configs)}
    if config_names is None:
        config_names = {cfg: cfg for cfg in configs}

    performance_metrics = {}

    for i, col in enumerate([col for col in monthly_df.columns if col.startswith('sim_Q_')]):
        ax = axes[i]
        config_name = col.replace('sim_Q_', '')

        # Plot observed data
        if 'obs_Q' in monthly_df.columns:
            ax.plot(monthly_df.index, monthly_df['obs_Q'], 'k-', linewidth=2, label='Observed')

        # Plot simulated data
        color = config_colors.get(config_name, f"C{i}")
        display_name = config_names.get(config_name, config_name)
        ax.plot(monthly_df.index, monthly_df[col], color=color, linewidth=2, label='Simulated')

        ax.set_title(display_name, fontsize=14)
        if i % n_cols == 0:
            ax.set_ylabel('Discharge (m³/s)', fontsize=12)
        if i >= n_configs - n_cols:
            ax.set_xlabel('Month', fontsize=12)

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')

        # Calculate performance metrics
        if config_name in all_data:
            data = all_data[config_name]
            validation_mask = (data['date'] >= validation_start) & (data['date'] <= validation_end)
            val_data = data[validation_mask]

            if 'obs_Q' in val_data.columns and 'sim_Q' in val_data.columns:
                obs = val_data['obs_Q'].values
                sim = val_data['sim_Q'].values
                valid_mask = ~np.isnan(obs) & ~np.isnan(sim)

                if np.sum(valid_mask) > 0:
                    obs = obs[valid_mask]
                    sim = sim[valid_mask]
                    obs_mean = np.mean(obs)
                    nse = 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - obs_mean) ** 2))
                    mean_sim = np.mean(sim)
                    mean_obs = np.mean(obs)
                    std_sim = np.std(sim)
                    std_obs = np.std(obs)
                    corr = np.corrcoef(sim, obs)[0, 1]
                    alpha = std_sim / std_obs
                    beta = mean_sim / mean_obs
                    kge = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                    performance_metrics[config_name] = {'NSE': nse, 'KGE': kge}
                    metrics_text = f"NSE = {nse:.3f}\nKGE = {kge:.3f}"
                    ax.text(0.05, 0.8, metrics_text, transform=ax.transAxes, fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.8))

    for i in range(n_configs, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(f'Hydrological Regime - Catchment {gauge_id}\nValidation Period: {validation_start} to {validation_end}',
                 fontsize=16, y=1.02)
    plt.tight_layout()

    # Save results to catchment-specific results folder
    for config in configs:
        result_dir = Path(base_dir) / f"catchment_{gauge_id}_{config}" / "plots_results"
        os.makedirs(result_dir, exist_ok=True)
        plt.savefig(result_dir / f'hydrological_regime_{gauge_id}.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot to {result_dir}/hydrological_regime_{gauge_id}.png")

    plt.show()
    return monthly_df

#--------------------------------------------------------------------------------

def process_regime_all_catchments(base_dir, configs, validation_start='2010-01-01', validation_end='2020-12-31', config_colors=None, config_names=None):
    """
    Process all available catchments in the base directory
    
    Parameters
    ----------
    base_dir : str
        Base directory path
    configs : list
        List of configuration names
    validation_start : str
        Start date for validation period
    validation_end : str
        End date for validation period
    """
    # Get all available catchment IDs
    catchment_ids = get_available_catchments(base_dir, configs)
    
    # Process each catchment
    for gauge_id in catchment_ids:
        print(f"\n{'='*40}\nProcessing catchment {gauge_id}\n{'='*40}")
        
        try:
            # Generate plot for this catchment
            plot_hydrological_regime_subplots(
                gauge_id, 
                configs, 
                base_dir,
                validation_start=validation_start, 
                validation_end=validation_end,
                config_colors=config_colors,
                config_names=config_names
            )

        except Exception as e:
            print(f"Error processing catchment {gauge_id}: {e}")
            
    print("\nProcessing complete!")

#--------------------------------------------------------------------------------

def plot_hydrograph_comparison(gauge_id, configs, base_dir, start_date=None, end_date=None,
                              config_colors=None, config_names=None, plot_type='timeseries',
                              figsize=None, save_plots=True):
    """
    Load hydrograph data and plot simulated vs observed streamflow for multiple configurations.
    
    Parameters:
    -----------
    gauge_id : str
        ID of the gauge to analyze
    configs : list
        List of configuration names to compare (e.g., ['nc', 'nc_sr', 'c', 'c_sr'])
    base_dir : str or Path
        Base directory containing model outputs
    start_date : str, optional
        Start date for filtering (format: 'YYYY-MM-DD')
    end_date : str, optional
        End date for filtering (format: 'YYYY-MM-DD')
    config_colors : dict, optional
        Dict mapping config names to colors (e.g., {'nc': '#712423'})
    config_names : dict, optional
        Dict mapping config names to friendly display names (e.g., {'nc': 'HBV'})
    plot_type : str, optional
        Type of plot: 'timeseries', 'scatter', or 'both' (default: 'timeseries')
    figsize : tuple, optional
        Figure size as (width, height). If None, calculates based on plot type.
    save_plots : bool, optional
        Whether to save plots to results folders (default: True)
        
    Returns:
    --------
    dict
        Dictionary with configuration names as keys and hydrograph DataFrames as values
    """
    
    # Set default colors and names if not provided
    if config_colors is None:
        config_colors = {
            'nc': '#712423',
            'c': '#976c03',
            'c_sr': '#82b182',
            'nc_sr': '#356891'
        }
    
    if config_names is None:
        config_names = {
            'nc': 'HBV',
            'c': 'HBV-GloGEM',
            'c_sr': 'HBV-GloGEM-SR',
            'nc_sr': 'HBV-SR'
        }
    
    # Load hydrograph data for all configurations
    all_hydro_data = {}
    
    print(f"Loading hydrograph data for gauge {gauge_id}...")
    
    for config in configs:
        print(f"  - Loading {config}...")
        hydro_data = load_hydrograph_data(gauge_id, config, base_dir)
        
        if hydro_data is not None:
            # Filter by date range if provided
            if start_date:
                hydro_data = hydro_data[hydro_data['date'] >= pd.to_datetime(start_date)]
            if end_date:
                hydro_data = hydro_data[hydro_data['date'] <= pd.to_datetime(end_date)]
            
            # Check if we have data after filtering
            if len(hydro_data) > 0:
                all_hydro_data[config] = hydro_data
                print(f"    ✅ Loaded {len(hydro_data)} records")
            else:
                print(f"    ❌ No data in specified date range")
        else:
            print(f"    ❌ Failed to load data")
    
    if not all_hydro_data:
        print("❌ No hydrograph data found for any configuration")
        return None
    
    # Create period string for titles
    period_str = ""
    if start_date and end_date:
        period_str = f" ({start_date} to {end_date})"
    elif start_date:
        period_str = f" (from {start_date})"
    elif end_date:
        period_str = f" (to {end_date})"
    
    # Plot 1: Time series comparison
    if plot_type in ['timeseries', 'both']:
        if figsize is None:
            figsize = (16, 8)
        
        plt.figure(figsize=figsize)
        
        # Plot observed data (only once, should be same for all configs)
        first_config = list(all_hydro_data.keys())[0]
        if 'obs_Q' in all_hydro_data[first_config].columns:
            plt.plot(all_hydro_data[first_config]['date'], 
                    all_hydro_data[first_config]['obs_Q'], 
                    'k-', linewidth=2.5, label='Observed', alpha=0.8)
        
        # Plot simulated data for each configuration
        for config, hydro_data in all_hydro_data.items():
            if 'sim_Q' in hydro_data.columns:
                color = config_colors.get(config, f"C{list(all_hydro_data.keys()).index(config)}")
                label = config_names.get(config, config)
                
                plt.plot(hydro_data['date'], hydro_data['sim_Q'], 
                        color=color, linewidth=2, label=f'{label}', alpha=0.8)
        
        # Remove x-axis label as requested
        # plt.xlabel('Date', fontsize=14)  # ✅ Commented out
        plt.ylabel('Discharge (m³/s)', fontsize=18)  # ✅ Bigger font size
        plt.title(f'Hydrograph Comparison - Gauge {gauge_id}{period_str}', fontsize=20)  # ✅ Bigger font size
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=16)  # ✅ Bigger font size
        
        # Format x-axis dates with bigger font
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, fontsize=14)  # ✅ Bigger font size
        plt.yticks(fontsize=14)  # ✅ Bigger font size
        
        # Add performance metrics in upper left corner
        perf_text = "Performance Metrics:\n" + "-" * 20 + "\n"
        for config, hydro_data in all_hydro_data.items():
            if 'obs_Q' in hydro_data.columns and 'sim_Q' in hydro_data.columns:
                obs = hydro_data['obs_Q'].values
                sim = hydro_data['sim_Q'].values
                
                # Remove NaN values
                valid_mask = ~np.isnan(obs) & ~np.isnan(sim)
                if np.sum(valid_mask) > 0:
                    obs_clean = obs[valid_mask]
                    sim_clean = sim[valid_mask]
                    
                    # Calculate NSE
                    obs_mean = np.mean(obs_clean)
                    nse = 1 - (np.sum((obs_clean - sim_clean) ** 2) / np.sum((obs_clean - obs_mean) ** 2))
                    
                    # Calculate KGE
                    mean_sim = np.mean(sim_clean)
                    mean_obs = np.mean(obs_clean)
                    std_sim = np.std(sim_clean)
                    std_obs = np.std(obs_clean)
                    corr = np.corrcoef(sim_clean, obs_clean)[0, 1]
                    alpha = std_sim / std_obs
                    beta = mean_sim / mean_obs
                    kge = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                    
                    config_name = config_names.get(config, config)
                    perf_text += f"{config_name}: NSE={nse:.3f}, KGE={kge:.3f}\n"
        
        # ✅ Move text box to upper left corner with bigger font
        plt.figtext(0.02, 0.98, perf_text, fontsize=14, 
                   bbox=dict(facecolor='white', alpha=0.8), 
                   verticalalignment='top')
        
        plt.tight_layout()
        
        # Save plot
        if save_plots:
            for config in configs:
                save_dir = Path(base_dir) / f"catchment_{gauge_id}_{config}" / "plots_results"
                save_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_dir / f"hydrograph_timeseries_{gauge_id}.png", 
                           dpi=300, bbox_inches='tight')
            print(f"📁 Saved timeseries plots to plots_results folders")
        
        plt.show()
    
    # Plot 2: Scatter plot comparison (Observed vs Simulated)
    if plot_type in ['scatter', 'both']:
        n_configs = len(all_hydro_data)
        n_cols = min(3, n_configs)
        n_rows = (n_configs + n_cols - 1) // n_cols
        
        if figsize is None:
            figsize = (6 * n_cols, 5 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        fig.suptitle(f'Observed vs Simulated - Gauge {gauge_id}{period_str}', fontsize=20)  # ✅ Bigger font size
        
        for i, (config, hydro_data) in enumerate(all_hydro_data.items()):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            if 'obs_Q' in hydro_data.columns and 'sim_Q' in hydro_data.columns:
                obs = hydro_data['obs_Q'].values
                sim = hydro_data['sim_Q'].values
                
                # Remove NaN values
                valid_mask = ~np.isnan(obs) & ~np.isnan(sim)
                obs_clean = obs[valid_mask]
                sim_clean = sim[valid_mask]
                
                if len(obs_clean) > 0:
                    # Create scatter plot
                    color = config_colors.get(config, f"C{i}")
                    ax.scatter(obs_clean, sim_clean, alpha=0.6, color=color, s=20)
                    
                    # Add 1:1 line
                    min_val = min(obs_clean.min(), sim_clean.min())
                    max_val = max(obs_clean.max(), sim_clean.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
                    
                    # Calculate and display metrics
                    obs_mean = np.mean(obs_clean)
                    nse = 1 - (np.sum((obs_clean - sim_clean) ** 2) / np.sum((obs_clean - obs_mean) ** 2))
                    
                    mean_sim = np.mean(sim_clean)
                    mean_obs = np.mean(obs_clean)
                    std_sim = np.std(sim_clean)
                    std_obs = np.std(obs_clean)
                    corr = np.corrcoef(sim_clean, obs_clean)[0, 1]
                    alpha = std_sim / std_obs
                    beta = mean_sim / mean_obs
                    kge = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                    
                    # Add metrics text in upper left corner with bigger font
                    metrics_text = f'NSE = {nse:.3f}\nKGE = {kge:.3f}\nr = {corr:.3f}'
                    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                           fontsize=14, verticalalignment='top',  # ✅ Bigger font size
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Set labels and title with bigger fonts
            config_name = config_names.get(config, config)
            ax.set_title(f'{config_name}', fontsize=16)  # ✅ Bigger font size
            ax.set_xlabel('Observed Discharge (m³/s)', fontsize=14)  # ✅ Bigger font size
            ax.set_ylabel('Simulated Discharge (m³/s)', fontsize=14)  # ✅ Bigger font size
            ax.grid(True, alpha=0.3)
            
            # Increase tick label font size
            ax.tick_params(axis='both', which='major', labelsize=12)  # ✅ Bigger font size
        
        # Hide empty subplots
        for i in range(n_configs, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        if save_plots:
            for config in configs:
                save_dir = Path(base_dir) / f"catchment_{gauge_id}_{config}" / "plots_results"
                save_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_dir / f"hydrograph_scatter_{gauge_id}.png", 
                           dpi=300, bbox_inches='tight')
            print(f"📁 Saved scatter plots to plots_results folders")
        
        plt.show()
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics for Gauge {gauge_id}{period_str}")
    print("=" * 60)
    
    for config, hydro_data in all_hydro_data.items():
        config_name = config_names.get(config, config)
        
        if 'obs_Q' in hydro_data.columns and 'sim_Q' in hydro_data.columns:
            obs = hydro_data['obs_Q'].values
            sim = hydro_data['sim_Q'].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(obs) & ~np.isnan(sim)
            obs_clean = obs[valid_mask]
            sim_clean = sim[valid_mask]
            
            if len(obs_clean) > 0:
                # Calculate metrics
                obs_mean = np.mean(obs_clean)
                sim_mean = np.mean(sim_clean)
                nse = 1 - (np.sum((obs_clean - sim_clean) ** 2) / np.sum((obs_clean - obs_mean) ** 2))
                
                # KGE calculation
                std_sim = np.std(sim_clean)
                std_obs = np.std(obs_clean)
                corr = np.corrcoef(sim_clean, obs_clean)[0, 1]
                alpha = std_sim / std_obs
                beta = sim_mean / obs_mean
                kge = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                
                # Percent bias
                pbias = 100 * (np.sum(sim_clean - obs_clean) / np.sum(obs_clean))
                
                print(f"{config_name:15s}: NSE={nse:6.3f}, KGE={kge:6.3f}, PBIAS={pbias:6.1f}%, "
                      f"r={corr:6.3f}, n={len(obs_clean):6d}")
        else:
            print(f"{config_name:15s}: Missing observed or simulated data")
    
    return all_hydro_data




#--------------------------------------------------------------------------------
################################### metrics #####################################
#--------------------------------------------------------------------------------

def calculate_performance_metrics(data, start_date, end_date, period_name=""):
    """
    Calculate NSE, KGE, and KGE_NP for a specific period
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing 'date', 'sim_Q', and 'obs_Q' columns
    start_date : datetime
        Start date for the analysis period
    end_date : datetime
        End date for the analysis period
    period_name : str
        Name of the period (for display purposes)
        
    Returns:
    --------
    dict
        Dictionary containing calculated metrics
    """
    # Filter the data for the specified period
    period_mask = (data['date'] >= start_date) & (data['date'] <= end_date)
    period_data = data[period_mask].copy()
    
    if len(period_data) == 0:
        print(f"  - No data found for {period_name} period ({start_date} to {end_date})")
        return None
        
    # Extract observed and simulated values
    # First ensure data is numeric
    try:
        period_data['obs_Q'] = pd.to_numeric(period_data['obs_Q'], errors='coerce')
        period_data['sim_Q'] = pd.to_numeric(period_data['sim_Q'], errors='coerce')
        
        obs = period_data['obs_Q'].values
        sim = period_data['sim_Q'].values
        
        # Now check for NaN values
        valid_mask = ~np.isnan(obs) & ~np.isnan(sim)
        if np.sum(valid_mask) == 0:
            print(f"  - No valid data points for {period_name} period (all NaN)")
            return None
        
        # Use only valid data points
        obs = obs[valid_mask]
        sim = sim[valid_mask]
        
        print(f"  - Working with {np.sum(valid_mask)} valid data points for {period_name} period")
        
        # Calculate NSE
        obs_mean = np.mean(obs)
        nse = 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - obs_mean) ** 2))
        
        # Calculate KGE components
        mean_sim = np.mean(sim)
        mean_obs = np.mean(obs)
        std_sim = np.std(sim)
        std_obs = np.std(obs)
        
        # Pearson correlation
        corr = np.corrcoef(sim, obs)[0, 1]
        
        # Calculate KGE components
        alpha = std_sim / std_obs
        beta = mean_sim / mean_obs
        
        # KGE calculation
        kge = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        
        # Calculate KGE_NP (non-parametric version)
        # Sort values for rank correlation
        sim_sorted = np.sort(sim)
        obs_sorted = np.sort(obs)
        
        # Calculate Spearman rank correlation
        # Do this by creating sorted indices and then using them to reorder data
        sim_ranks = np.argsort(np.argsort(sim))
        obs_ranks = np.argsort(np.argsort(obs))
        spearman_corr = np.corrcoef(sim_ranks, obs_ranks)[0, 1]
        
        # Alpha NP - ratio of flow duration curve slopes
        # Add small epsilon to avoid division by zero
        alpha_np = np.mean(np.abs(np.diff(sim_sorted)) + 1e-10) / np.mean(np.abs(np.diff(obs_sorted)) + 1e-10)
        
        # Beta NP - remains the same as KGE
        beta_np = beta
        
        # Calculate KGE_NP
        kge_np = 1 - np.sqrt((spearman_corr - 1)**2 + (alpha_np - 1)**2 + (beta_np - 1)**2)
        
        print(f"  - {period_name} period metrics:")
        print(f"    NSE: {nse:.3f}")
        print(f"    KGE: {kge:.3f} (r={corr:.3f}, α={alpha:.3f}, β={beta:.3f})")
        print(f"    KGE_NP: {kge_np:.3f} (r_s={spearman_corr:.3f}, α_np={alpha_np:.3f}, β={beta_np:.3f})")
        print(f"    Data points: {len(obs)}")
        
        return {
            'NSE': nse,
            'KGE': kge,
            'KGE_NP': kge_np,
            'r': corr,
            'r_spearman': spearman_corr,
            'alpha': alpha,
            'alpha_np': alpha_np,
            'beta': beta,
            'n_points': len(obs)
        }
    
    except Exception as e:
        print(f"  - Error calculating metrics for {period_name} period: {e}")
        print("  - Check that 'sim_Q' and 'obs_Q' columns contain valid numeric data")
        return None


#--------------------------------------------------------------------------------

def analyze_hydrological_performance(base_dir, configs, calibration_period=('2000-01-01', '2009-12-31'), 
                                    validation_period=('2010-01-01', '2020-12-31')):
    """
    Analyze hydrological performance metrics (KGE, KGE_NP, NSE) across all catchments
    for different model configurations, for both calibration and validation periods.
    
    Parameters:
    -----------
    base_dir : str
        Base directory path containing model outputs
    configs : list
        List of configuration names to analyze
    calibration_period : tuple
        Start and end dates for calibration period
    validation_period : tuple
        Start and end dates for validation period
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing performance metrics for all catchments and configurations
    """
    # Get all available catchment IDs
    catchment_ids = get_available_catchments(base_dir, configs)
    
    # Convert period strings to datetime
    cal_start, cal_end = pd.to_datetime(calibration_period[0]), pd.to_datetime(calibration_period[1])
    val_start, val_end = pd.to_datetime(validation_period[0]), pd.to_datetime(validation_period[1])
    
    # Prepare results storage
    results = []
    
    # Process each catchment
    for gauge_id in catchment_ids:
        print(f"\n{'='*60}\nProcessing catchment {gauge_id}\n{'='*60}")
        
        # Process each configuration
        for config in configs:
            print(f"\n{'-'*40}\nAnalyzing {gauge_id} - {config}\n{'-'*40}")
            
            # Load hydrograph data
            hydro_data = load_hydrograph_data(gauge_id, config, base_dir)
            
            if hydro_data is None or 'sim_Q' not in hydro_data.columns or 'obs_Q' not in hydro_data.columns:
                print(f"  - Skipping {config} - Missing required data")
                continue
                
            # Calculate metrics for calibration period
            cal_metrics = calculate_performance_metrics(
                hydro_data, 
                start_date=cal_start,
                end_date=cal_end,
                period_name="Calibration"
            )
            
            # Calculate metrics for validation period
            val_metrics = calculate_performance_metrics(
                hydro_data, 
                start_date=val_start,
                end_date=val_end,
                period_name="Validation"
            )
            
            # Store results
            if cal_metrics and val_metrics:
                results.append({
                    'Catchment': gauge_id,
                    'Configuration': config,
                    'Cal_NSE': cal_metrics['NSE'],
                    'Cal_KGE': cal_metrics['KGE'],
                    'Cal_KGE_NP': cal_metrics['KGE_NP'],
                    'Val_NSE': val_metrics['NSE'],
                    'Val_KGE': val_metrics['KGE'],
                    'Val_KGE_NP': val_metrics['KGE_NP'],
                })
    
    # Convert to DataFrame
    if not results:
        print("No valid results found.")
        return None
        
    results_df = pd.DataFrame(results)
    
    # Print summary statistics
    print("\n\n" + "="*80)
    print("SUMMARY OF HYDROLOGICAL PERFORMANCE METRICS")
    print("="*80)
    
    # Group by configuration and calculate mean values
    summary = results_df.groupby('Configuration').agg({
        'Cal_NSE': ['mean', 'std', 'min', 'max'],
        'Cal_KGE': ['mean', 'std', 'min', 'max'],
        'Cal_KGE_NP': ['mean', 'std', 'min', 'max'],
        'Val_NSE': ['mean', 'std', 'min', 'max'],
        'Val_KGE': ['mean', 'std', 'min', 'max'],
        'Val_KGE_NP': ['mean', 'std', 'min', 'max'],
    })
    
    print(summary)
    
    # Save results to CSV
    csv_file = "hydrological_performance_metrics.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"\nSaved performance metrics to: {csv_file}")
    
    # Save summary to CSV
    summary_file = "hydrological_performance_summary.csv"
    summary.to_csv(summary_file)
    print(f"Saved summary statistics to: {summary_file}")
    
    return results_df


#--------------------------------------------------------------------------------


def plot_configuration_comparison(results_df, config_x, config_y, metrics=None, period='Val_', 
                                config_colors=None, config_names=None, output_dir=None):
    """
    Create plots comparing performance metrics between two configurations
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing performance metrics
    config_x : str
        Configuration name for x-axis
    config_y : str
        Configuration name for y-axis
    metrics : list or None
        List of metrics to plot. If None, plots KGE, KGE_NP, NSE.
    period : str
        Period prefix ('Cal_' or 'Val_')
    config_colors : dict, optional
        Dict mapping config names to colors
    config_names : dict, optional
        Dict mapping config names to friendly display names
    output_dir : str or None
        Directory to save plots. If None, saves to current directory.
    """
    if results_df is None or len(results_df) == 0:
        print("No data available for plotting")
        return
        
    if metrics is None:
        metrics = ['KGE', 'KGE_NP', 'NSE']
    
    
    # Filter data for the two configurations
    config_x_data = results_df[results_df['Configuration'] == config_x]
    config_y_data = results_df[results_df['Configuration'] == config_y]
    
    if len(config_x_data) == 0:
        print(f"No data found for configuration: {config_x}")
        return
    if len(config_y_data) == 0:
        print(f"No data found for configuration: {config_y}")
        return
    
    # Merge data on catchment ID to ensure we compare the same catchments
    merged_data = pd.merge(config_x_data, config_y_data, on='Catchment', suffixes=('_x', '_y'))
    
    if len(merged_data) == 0:
        print(f"No common catchments found between {config_x} and {config_y}")
        return
        
    # Set up figure
    fig, axes = plt.subplots(1, len(metrics), figsize=(7*len(metrics), 6), sharey=False)
    if len(metrics) == 1:
        axes = [axes]  # Make it iterable for single plot
        
    # Plot for each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        x_col = f'{period}{metric}_x'
        y_col = f'{period}{metric}_y'
        
        # Check if columns exist
        if x_col not in merged_data.columns or y_col not in merged_data.columns:
            print(f"Warning: Columns {x_col} or {y_col} not found in data")
            continue
        
        # Get min/max for setting equal axis limits
        min_val = min(merged_data[x_col].min(), merged_data[y_col].min())
        max_val = max(merged_data[x_col].max(), merged_data[y_col].max())
        
        # Add some margin
        range_val = max_val - min_val
        min_val -= range_val * 0.1
        max_val += range_val * 0.1
        
        # Plot scatter points
        ax.scatter(merged_data[x_col], merged_data[y_col], 
                  color='#2a5674', alpha=0.8, s=60, edgecolors='k', linewidth=0.5)
        
        # Add perfect performance point
        ax.scatter(1, 1, marker='*', color='red', s=200, 
                  label='Perfect', edgecolors='k', zorder=10)
        
        # Add diagonal line (1:1)
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='1:1 line')
        
        # Add horizontal and vertical reference lines at y=0 and x=0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        # Set equal axes limits
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
        # Get friendly names for axis labels
        x_name = config_names.get(config_x, config_x)
        y_name = config_names.get(config_y, config_y)
        
        # Set labels and title
        period_name = "Validation" if period == "Val_" else "Calibration"
        ax.set_xlabel(f'{x_name} {metric}', fontsize=14)
        ax.set_ylabel(f'{y_name} {metric}', fontsize=14)
        ax.set_title(f'{metric} {period_name}: {x_name} vs {y_name}', fontsize=16)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add background shading to indicate which configuration performs better
        if min_val < 0:
            # Above diagonal: y-config better than x-config
            ax.fill_between([min_val, max_val], [min_val, max_val], max_val, 
                           color=config_colors.get(config_y, '#82b182'), alpha=0.1, 
                           label=f'{y_name} better')
            # Below diagonal: x-config better than y-config  
            ax.fill_between([min_val, max_val], min_val, [min_val, max_val], 
                           color=config_colors.get(config_x, '#976c03'), alpha=0.1,
                           label=f'{x_name} better')
        
        # Annotate points with catchment IDs
        for _, row in merged_data.iterrows():
            ax.annotate(row['Catchment'], 
                       (row[x_col], row[y_col]),
                       fontsize=8, alpha=0.7,
                       xytext=(5, 5), textcoords='offset points')
    
    # Add legend (only in the last subplot)
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Determine output path
    period_str = "validation" if period == "Val_" else "calibration"
    filename = f'{config_x}_vs_{config_y}_{period_str}_{"-".join(metrics)}.png'
    if output_dir:
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
    else:
        output_path = filename
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also save to base_dir/plots_results if base_dir is available
    try:
        import os
        from pathlib import Path
        base_dir = "/home/jberg/OneDrive/Raven_Switzerland/09_runs_paper_1"
        plots_results_dir = os.path.join(base_dir, "plots_results")
        os.makedirs(plots_results_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_results_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Plot also saved to: {plots_results_dir}/{filename}")
    except Exception as e:
        print(f"Note: Could not save to base_dir/plots_results: {e}")
    
    plt.show()


#--------------------------------------------------------------------------------


def plot_performance_comparison(results_df, configurations=None, config_colors=None, config_names=None,
                              metric_prefix='Val_', metrics=None, plot_type='boxplot', output_dir=None):
    """
    Create visualization of performance metrics across configurations
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing performance metrics
    configurations : list or None
        List of configurations to include in the plot. If None, uses all configurations in data.
    config_colors : dict, optional
        Dict mapping config names to colors (e.g., {'nc': '#712423'})
    config_names : dict, optional
        Dict mapping config names to friendly display names (e.g., {'nc': 'HBV'})
    metric_prefix : str
        Prefix for the metrics to plot ('Cal_' or 'Val_')
    metrics : list or None
        List of metrics to plot (without prefix). If None, plots KGE, KGE_NP, NSE.
    plot_type : str
        Type of plot ('boxplot' or 'barplot')
    output_dir : str or None
        Directory to save plots. If None, saves to current directory.
    """
    if results_df is None or len(results_df) == 0:
        print("No data available for plotting")
        return
        
    if metrics is None:
        metrics = ['KGE', 'KGE_NP', 'NSE']
    
    # Set default configurations if not provided
    if configurations is None:
        configurations = results_df['Configuration'].unique().tolist()
    
    # Filter data for specified configurations
    filtered_df = results_df[results_df['Configuration'].isin(configurations)].copy()
    
    if len(filtered_df) == 0:
        print("No data found for the specified configurations")
        return
    
    period_name = "Validation" if metric_prefix == "Val_" else "Calibration"
    
    # Prepare data for plotting in long format
    plot_data = []
    for _, row in filtered_df.iterrows():
        for metric in metrics:
            col_name = f"{metric_prefix}{metric}"
            if col_name in row:
                config = row['Configuration']
                plot_data.append({
                    'Catchment': row['Catchment'],
                    'Configuration': config,
                    'ConfigDisplay': config_names.get(config, config),
                    'Metric': metric,
                    'Value': row[col_name]
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure - wider plots
    fig, axes = plt.subplots(1, len(metrics), figsize=(8*len(metrics), 8), sharey=False)
    if len(metrics) == 1:
        axes = [axes]  # Make it iterable for single plot
    
    # Create plots for each metric
    for i, metric in enumerate(metrics):
        metric_data = plot_df[plot_df['Metric'] == metric]
        
        if plot_type == 'boxplot':
            # Create palette for the configurations, using fallback colors if not defined
            palette = {}
            for j, config in enumerate(metric_data['Configuration'].unique()):
                if config in config_colors:
                    palette[config] = config_colors[config]
                else:
                    # Use matplotlib default colors as fallback
                    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                    palette[config] = color_cycle[j % len(color_cycle)]
            
            # Create boxplot
            ax = sns.boxplot(x='Configuration', y='Value', data=metric_data, 
                            palette=palette, ax=axes[i])
            sns.swarmplot(x='Configuration', y='Value', data=metric_data, 
                         color='black', alpha=0.7, size=6, ax=axes[i])
                
            # Set plot properties with bigger fonts
            ax.set_title(f'{period_name} {metric}', fontsize=22)
            ax.set_xlabel('', fontsize=1)  # Remove x-axis label
            ax.set_ylabel(f'{metric}', fontsize=20)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Increase tick label font size
            ax.tick_params(axis='both', which='major', labelsize=16)
            
            # Use friendly configuration names for x-tick labels
            configs = [c for c in metric_data['Configuration'].unique()]
            labels = [config_names.get(c, c) for c in configs]
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16)
            
            # Add reference line for perfect performance
            if metric in ['KGE', 'KGE_NP', 'NSE']:
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.6, linewidth=2)
                ax.text(ax.get_xlim()[1]*0.95, 1.0, 'Perfect', 
                       ha='right', va='bottom', color='red', fontsize=16)
            
        elif plot_type == 'barplot':
            # Calculate mean and standard error for each configuration
            grouped = metric_data.groupby('Configuration')['Value'].agg(['mean', 'std', 'count'])
            grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
            
            # Create barplot
            ax = axes[i]
            
            # Use the consistent color palette with fallbacks
            colors = []
            for j, config in enumerate(grouped.index):
                if config in config_colors:
                    colors.append(config_colors[config])
                else:
                    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                    colors.append(color_cycle[j % len(color_cycle)])
            
            bars = ax.bar(grouped.index, grouped['mean'], yerr=grouped['se'],
                   color=colors, capsize=5, alpha=0.8, ecolor='black', linewidth=1.5)
            
            # Set plot properties with bigger fonts
            ax.set_title(f'{period_name} Mean {metric}', fontsize=22)
            ax.set_xlabel('', fontsize=1)  # Remove x-axis label
            ax.set_ylabel(f'{metric}', fontsize=20)
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Increase tick label font size
            ax.tick_params(axis='both', which='major', labelsize=16)
            
            # Use friendly configuration names for x-tick labels
            labels = [config_names.get(c, c) for c in grouped.index]
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16)
            
            # Add text with mean values - bigger font
            for bar, mean in zip(bars, grouped['mean']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=14)
            
            # Add reference line
            if metric in ['KGE', 'KGE_NP', 'NSE']:
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.6, linewidth=2)
                
    # Adjust layout
    plt.tight_layout()
    
    # Create filename that includes configuration names
    config_suffix = "_".join(configurations) if len(configurations) <= 3 else f"{len(configurations)}configs"
    filename = f'performance_{metric_prefix.replace("_", "")}_{plot_type}_{config_suffix}.png'
    if output_dir:
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
    else:
        output_path = filename
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also save to base_dir/plots_results if base_dir is available
    try:
        import os
        from pathlib import Path
        base_dir = "/home/jberg/OneDrive/Raven_Switzerland/09_runs_paper_1"
        plots_results_dir = os.path.join(base_dir, "plots_results")
        os.makedirs(plots_results_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_results_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Plot also saved to: {plots_results_dir}/{filename}")
    except Exception as e:
        print(f"Note: Could not save to base_dir/plots_results: {e}")
    
    plt.show()


#--------------------------------------------------------------------------------
################################### parameter ###################################
#--------------------------------------------------------------------------------


def load_parameter_values(gauge_id, configs, base_dir, top_n=100):
    """
    Load parameter values from different model configurations for analysis.
    Selects the best top_n parameter sets based on objective function.
    
    Parameters:
    -----------
    gauge_id : str
        ID of the gauge to analyze
    configs : list
        List of configuration names to analyze (e.g., 'nc_single', 'c_multi')
    base_dir : str or Path
        Base directory containing model outputs
    top_n : int
        Number of top parameter sets to select
    
    Returns:
    --------
    dict
        Dictionary containing parameter values for each configuration
    """
    
    # Dictionary to store parameter values for each configuration
    param_values = {}
    
    # Process each configuration
    for config in configs:
        print(f"\n{'-'*40}\nAnalyzing parameters for {gauge_id} - {config}\n{'-'*40}")
        
        # Build path to model output directory
        model_dir = Path(base_dir) / f"catchment_{gauge_id}_{config}" / "HBV" / "output"
        
        # Look for calibration results files
        calibration_files = list(model_dir.glob(f"calibration_results_{gauge_id}_HBV_*.csv"))
        
        found = False
        if calibration_files:
            # Use the first file if multiple exist
            cal_file = calibration_files[0]
            print(f"Found calibration file: {cal_file}")
            
            try:
                df = pd.read_csv(cal_file)
                print(f"Loaded {len(df)} parameter sets")
                
                # Check for objective column
                if 'objective' in df.columns:
                    print(f"Using 'objective' column for parameter selection")
                    
                    # Sort by objective (higher is better) and get top N
                    df = df.sort_values('objective', ascending=False).head(top_n)
                    print(f"Selected top {len(df)} parameter sets")
                    print(f"Objective range: {df['objective'].min():.4f} to {df['objective'].max():.4f}")
                else:
                    print(f"Warning: 'objective' column not found")
                    # Try to find alternative columns
                    obj_col = None
                    for possible_col in ['KGE', 'obj_function_value', 'KGE_NP']:
                        if possible_col in df.columns:
                            obj_col = possible_col
                            break
                            
                    if obj_col:
                        print(f"Using alternative objective column: {obj_col}")
                        # Sort by objective and get top N
                        df = df.sort_values(obj_col, ascending=False).head(top_n)
                        print(f"Selected top {len(df)} parameter sets based on {obj_col}")
                    else:
                        print(f"Warning: No objective function column found. Using first {top_n} rows")
                        df = df.head(top_n)
                
                found = True
                
                # Extract parameter columns (starting with 'HBV_')
                param_cols = [col for col in df.columns if col.startswith('HBV_')]
                
                if len(param_cols) == 0:
                    print(f"Warning: No HBV parameter columns found")
                    continue
                    
                print(f"Found {len(param_cols)} parameter columns: {', '.join(param_cols[:5])}...")
                
                # Store parameter values for all selected top runs
                param_values[config] = {}
                
                # For each parameter, store all values from the top runs
                for col in param_cols:
                    param_values[config][col] = df[col].values.tolist()
                
                # Calculate basic statistics for each parameter
                param_stats = {}
                for col in param_cols:
                    param_stats[col] = {
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'mean': df[col].mean(),
                        'median': df[col].median(),
                        'std': df[col].std()
                    }
                
                # Store statistics
                param_values[config]['stats'] = param_stats
                
            except Exception as e:
                print(f"Error reading file {cal_file}: {e}")
                found = False
        
        if not found:
            print(f"No valid calibration files found for {config}")
    
    return param_values

#--------------------------------------------------------------------------------

def create_parameter_comparison_plot(param_name, param_data, catchments, configs, base_dir, 
                                   config_colors=None, config_names=None):
    """
    Create a detailed plot for a single parameter across all catchments and configurations.
    
    Parameters:
    -----------
    param_name : str
        Name of the parameter to plot
    param_data : pandas.DataFrame
        DataFrame containing the parameter data
    catchments : list
        List of catchment IDs
    configs : list
        List of configurations
    base_dir : str or Path
        Base directory where plots_results folder will be created
    config_colors : dict, optional
        Dict mapping config names to colors (e.g., {'nc': '#712423'})
    config_names : dict, optional
        Dict mapping config names to friendly display names (e.g., {'nc': 'HBV'})
    """
    
    custom_palette = []
    for i, config in enumerate(configs):
        if config in config_colors:
            custom_palette.append(config_colors[config])
    
    # Create plots_results directory if it doesn't exist
    plots_dir = Path(base_dir) / "plots_results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Set figure size based on number of catchments
    plt.figure(figsize=(max(12, len(catchments)*0.8), 8))
    
    # Create boxplot with custom palette
    ax = sns.boxplot(x='CatchmentID', y='Value', hue='Configuration', data=param_data, palette=custom_palette)
    
    # Add individual points with jitter
    sns.stripplot(x='CatchmentID', y='Value', hue='Configuration', data=param_data, 
                 dodge=True, alpha=0.3, jitter=True, size=3, palette=custom_palette, legend=False)
    
    # Set titles and labels
    plt.title(f'Parameter: {param_name} Across Catchments (Top 100 Sets)', fontsize=16)
    plt.xlabel('Catchment ID', fontsize=14)
    plt.ylabel(f'{param_name} Value', fontsize=14)
    
    # Improve legend with friendly configuration names
    handles, labels = ax.get_legend_handles_labels()
    # Replace configuration codes with friendly names
    friendly_labels = [config_names.get(label, label) for label in labels[:len(configs)]]
    # Only keep the legend items for the boxplot
    ax.legend(handles[:len(configs)], friendly_labels, 
             title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Calculate and display statistics
    grouped_stats = param_data.groupby(['CatchmentID', 'Configuration'])['Value'].agg(['mean', 'median', 'std']).reset_index()
    
    # Show mean values on the plot
    for idx, row in grouped_stats.iterrows():
        catchment = row['CatchmentID']
        config = row['Configuration']
        mean_val = row['mean']
        
        # Get the x-position for this catchment-config combination
        x_pos = catchments.index(catchment)
        config_idx = configs.index(config)
        
        # Adjust x position within the catchment group
        box_width = 0.8 / len(configs)
        adjusted_x = x_pos - 0.4 + (config_idx + 0.5) * box_width
        
        # Add text annotation for mean
        ax.text(adjusted_x, mean_val, f"{mean_val:.2f}", 
               ha='center', va='bottom', fontsize=7, alpha=0.7)
    
    # Add reference lines if this is a known parameter with typical ranges
    if 'RainSnow_Temp' in param_name:
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Freezing Point')
        plt.axhline(y=2, color='b', linestyle='--', alpha=0.5, label='Typical Range')
        plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save to plots_results directory
    save_path = plots_dir / f"param_{param_name.replace('/', '_').replace(':', '_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.show()
    plt.close()
    
    return save_path

#--------------------------------------------------------------------------------

def analyze_parameters_across_catchments(base_dir, configs, config_colors=None, config_names=None,
                                        top_n=100):
    """
    Main function to analyze parameter values across all catchments and configurations.
    Selects top parameter sets based on objective functions.
    
    Parameters:
    -----------
    base_dir : str or Path
        Base directory containing model outputs
    configs : list
        List of configuration names to analyze
    config_colors : dict, optional
        Dict mapping config names to colors (e.g., {'nc': '#712423'})
    config_names : dict, optional
        Dict mapping config names to friendly display names (e.g., {'nc': 'HBV'})
    top_n : int
        Number of top parameter sets to select
    
    Returns:
    --------
    tuple
        (all_param_data, param_stats) containing compiled parameter data and statistics
    """
    
    print(f"Starting parameter analysis across catchments...")
    print(f"Base directory: {base_dir}")
    print(f"Configurations: {configs}")
    print(f"Selecting top {top_n} parameter sets")
    
    # Get all available catchments
    catchments = get_available_catchments(base_dir, configs)
    if not catchments:
        print("No catchments found.")
        return None, None
    
    print(f"Found {len(catchments)} catchments: {', '.join(catchments)}")
    
    # Load all parameter data
    all_data = {}
    all_param_names = set()
    
    for gauge_id in catchments:
        print(f"\nProcessing catchment {gauge_id}...")
        param_values = load_parameter_values(
            gauge_id, configs, base_dir, 
            top_n=top_n
        )
        
        if param_values:
            all_data[gauge_id] = param_values
            
            # Collect parameter names
            for config, values in param_values.items():
                all_param_names.update([p for p in values.keys() if p != 'stats'])
    
    # Convert parameter names to sorted list
    all_param_names = sorted(list(all_param_names))
    print(f"Found {len(all_param_names)} unique parameters")
    
    # Create DataFrame with all parameter values - now handle lists of values
    data_rows = []
    
    for gauge_id, params_by_config in all_data.items():
        for config, params in params_by_config.items():
            for param_name in all_param_names:
                if param_name in params and param_name != 'stats':
                    param_values = params[param_name]
                    
                    # Check if we have a list of values (should be the case now)
                    if isinstance(param_values, list):
                        for value in param_values:
                            data_rows.append({
                                'CatchmentID': gauge_id,
                                'Configuration': config,
                                'Parameter': param_name.replace('HBV_', ''),
                                'FullParameter': param_name,
                                'Value': value
                            })
                    else:
                        # Fallback for single values
                        data_rows.append({
                            'CatchmentID': gauge_id,
                            'Configuration': config,
                            'Parameter': param_name.replace('HBV_', ''),
                            'FullParameter': param_name,
                            'Value': param_values
                        })
    
    all_param_data = pd.DataFrame(data_rows)
    
    # Create plots_results directory in base_dir for combined outputs
    base_plots_dir = Path(base_dir) / "plots_results"
    base_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save complete dataset to base plots_results directory
    all_param_data.to_csv(base_plots_dir / "all_parameter_values.csv", index=False)
    
    # Process each parameter
    for param in all_param_names:
        param_display = param.replace('HBV_', '')
        print(f"Creating plot for {param_display}...")
        
        # Filter data for this parameter
        param_data = all_param_data[all_param_data['FullParameter'] == param]
        
        if len(param_data) == 0:
            print(f"  No data for {param_display}, skipping")
            continue
        
        # Create the plot and save to base plots_results directory
        save_path = create_parameter_comparison_plot(
            param_display, 
            param_data, 
            catchments, 
            configs, 
            base_dir,
            config_colors=config_colors,
            config_names=config_names
        )
        
        # Also save to each catchment's plots_results folder
        for gauge_id in catchments:
            for config in configs:
                catchment_plots_dir = Path(base_dir) / f"catchment_{gauge_id}_{config}" / "plots_results"
                catchment_plots_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy the plot to each catchment's directory
                import shutil
                dest_path = catchment_plots_dir / f"param_{param_display.replace('/', '_').replace(':', '_')}.png"
                try:
                    shutil.copy2(save_path, dest_path)
                    print(f"  Copied plot to {dest_path}")
                except Exception as e:
                    print(f"  Warning: Could not copy plot to {dest_path}: {e}")
    
    # Create summary statistics - now handling multiple values per parameter-configuration-catchment
    param_stats = all_param_data.groupby(['Parameter', 'Configuration', 'CatchmentID']).agg({
        'Value': ['count', 'min', 'max', 'mean', 'median', 'std']
    }).reset_index()
    
    # Save statistics to base plots_results directory
    param_stats.to_csv(base_plots_dir / "parameter_statistics.csv")
    
    # Also save statistics to each catchment's plots_results folder
    for gauge_id in catchments:
        # Filter statistics for this catchment
        catchment_stats = param_stats[param_stats['CatchmentID'] == gauge_id]
        
        for config in configs:
            catchment_plots_dir = Path(base_dir) / f"catchment_{gauge_id}_{config}" / "plots_results"
            catchment_plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Save catchment-specific statistics
            catchment_stats.to_csv(catchment_plots_dir / f"parameter_statistics_{gauge_id}.csv", index=False)
    
    print(f"\nAnalysis complete!")
    print(f"Main plots and data saved to: {base_plots_dir}")
    print(f"Plots also copied to individual catchment plots_results folders")
    
    return all_param_data, param_stats


#--------------------------------------------------------------------------------
###################################### SWE ######################################
#--------------------------------------------------------------------------------

def load_swe_data(gauge_id, config, base_dir):
    """
    Load simulated SWE data files for a specific configuration.
    
    Parameters:
    -----------
    gauge_id : str
        ID of the gauge to analyze
    config : str
        Configuration name (e.g., "nc", "nc_sr")
    base_dir : Path or str
        Base directory containing model outputs
        
    Returns:
    --------
    pandas.DataFrame or None
        Simulation data or None if loading fails
    """
    # Construct path to simulated data
    model_dir = Path(base_dir) / f"catchment_{gauge_id}_{config}" / "HBV" 
    sim_file = model_dir / "output" / f"{gauge_id}_HBV_SNOW_Daily_Average_ByHRUGroup.csv"
    
    print(f"Loading SWE data for {config}:")
    print(f"  - Simulated data: {sim_file}")
    
    # Check if file exists
    if not sim_file.exists():
        print(f"ERROR: Simulation file missing for configuration {config}")
        return None
    
    try:
        # Load simulated data with special header handling
        # Read first two lines to get headers
        with open(sim_file, 'r') as f:
            header_line = f.readline().strip()
            units_line = f.readline().strip()
        
        # Read the data, skipping the units line
        sim_data = pd.read_csv(sim_file, skiprows=[1], header=0)
        
        # Clean up column names - first column usually doesn't have a header
        if sim_data.columns[0] == '':
            sim_data = sim_data.rename(columns={sim_data.columns[0]: 'row_id'})
        
        # Try to find date column
        date_cols = [col for col in sim_data.columns if any(keyword in col.lower() for keyword in ['day', 'time', 'date'])]
        if date_cols:
            sim_data['date'] = pd.to_datetime(sim_data[date_cols[0]])
        elif len(sim_data.columns) > 0:
            # If no clear date column, create sequential dates
            print("  - Warning: No date column found, creating sequential dates")
            sim_data['date'] = pd.date_range(start='2000-01-01', periods=len(sim_data), freq='D')
        
        return sim_data
        
    except Exception as e:
        print(f"Error loading SWE data: {e}")
        return None


def process_swe_data(sim_data):
    """
    Process simulated SWE data.
    
    Parameters:
    -----------
    sim_data : pandas.DataFrame
        Simulation data
        
    Returns:
    --------
    dict
        Dictionary containing processed data and metadata
    """
    if sim_data is None:
        return None
    
    # Get elevation band columns efficiently using regex pattern
    sim_elev_pattern = re.compile(r'\d+-\d+m')
    sim_elev_cols = [col for col in sim_data.columns if sim_elev_pattern.search(col)]
    
    print(f"Found {len(sim_elev_cols)} simulation elevation bands")
    
    # Convert data to numeric
    for col in sim_elev_cols:
        sim_data[col] = pd.to_numeric(sim_data[col], errors='coerce')
    
    # Return processed data and metadata
    return {
        'sim_data': sim_data,
        'sim_elev_cols': sim_elev_cols
    }


def plot_swe_time_series_by_elevation(base_dir, gauge_id, model_type, configs=['nc', 'nc_sr'], 
                                     config_colors=None, config_names=None, water_year=None):
    """
    Plot time series of simulated SWE for each elevation band comparing 
    various model configurations, with plots arranged in a single column.
    
    Args:
        base_dir: Directory containing model files
        gauge_id: ID of the gauge
        model_type: Type of model (e.g. 'HBV')
        configs: List of configurations to compare (e.g. 'nc', 'nc_sr', etc.)
        config_colors: Dict mapping config names to colors (e.g., {'nc': '#712423'})
        config_names: Dict mapping config names to friendly display names (e.g., {'nc': 'HBV'})
        water_year: Optional water year to filter (e.g. 2018 for 2018-2019 water year)
    """
    
    # Load data for all configurations
    data_dict = {}
    
    for config in configs:
        sim_data = load_swe_data(gauge_id, config, base_dir)
        if sim_data is not None:
            processed_data = process_swe_data(sim_data)
            if processed_data is not None:
                data_dict[config] = processed_data
    
    if not data_dict:
        print(f"No SWE data available for gauge {gauge_id}")
        return
    
    # Get all elevation bands from one of the configurations
    config = list(data_dict.keys())[0]
    sim_elev_cols = data_dict[config]['sim_elev_cols']
    
    # Sort elevation bands by altitude
    elev_bands = sorted(sim_elev_cols, key=lambda x: int(x.split('-')[0]) if '-' in x else 0)
    
    if not elev_bands:
        print("No elevation bands found in simulation data")
        return
    
    # Optional: Filter for water year
    if water_year is not None:
        start_date = f"{water_year}-10-01"
        end_date = f"{water_year+1}-09-30"
        
        for config in data_dict:
            sim_data = data_dict[config]['sim_data']
            if 'date' in sim_data.columns:
                mask = (sim_data['date'] >= start_date) & (sim_data['date'] <= end_date)
                data_dict[config]['sim_data'] = sim_data[mask].copy()
    
    # Calculate number of subplots needed - use 1 column
    n_bands = len(elev_bands)
    n_cols = 1  # Single column layout
    n_rows = n_bands
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows), sharex=True)
    
    # Make axes iterable if there's only one plot
    if n_bands == 1:
        axes = np.array([axes])
    
    # Plot each elevation band
    for i, band in enumerate(elev_bands):
        ax = axes[i]
        
        # Plot each configuration
        for config in configs:
            if config in data_dict and band in data_dict[config]['sim_data'].columns:
                sim_data = data_dict[config]['sim_data']
                
                # Use provided colors and names
                color = config_colors.get(config, 'gray') if config_colors else None
                label = config_names.get(config, config) if config_names else config
                
                ax.plot(sim_data['date'], sim_data[band], 
                        linestyle='-', color=color, label=label, linewidth=1.2)
        
        # Add labels and legend
        ax.set_title(f'Elevation Band: {band}')
        ax.set_ylabel('SWE (mm)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(DateFormatter('%b-%d'))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
        
        # Add legend to the first plot only
        if i == 0:
            ax.legend(loc='upper right')
    
    # Add overall title
    if water_year:
        title = f'Simulated SWE by Elevation Band - Gauge {gauge_id} - Water Year {water_year}-{water_year+1}'
    else:
        title = f'Simulated SWE by Elevation Band - Gauge {gauge_id}'
        
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Add more space between subplots
    plt.subplots_adjust(hspace=0.3)
    
    # Create filename based on configurations and water year
    config_suffix = "_".join(configs)
    if water_year:
        filename = f"swe_time_series_by_elevation_{water_year}_{config_suffix}.png"
    else:
        filename = f"swe_time_series_by_elevation_{config_suffix}.png"
    
    # Save to each config's plots_results folder
    for config in configs:
        results_dir = f"{base_dir}/catchment_{gauge_id}_{config}/{model_type}/plots_results"
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(f"{results_dir}/{filename}", dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {results_dir}/{filename}")
    
    return fig


#--------------------------------------------------------------------------------
################################# contributions #################################
#--------------------------------------------------------------------------------

def process_glogem_data_optimized(gauge_id, base_dir, start_date, end_date, unit='mm', 
                                 plot=True, force_reprocess=False, chunk_size=10000):
    """
    Optimized version that processes large GloGEM files efficiently and caches results.
    
    Parameters:
    -----------
    gauge_id : str
        Gauge identifier
    base_dir : str or Path
        Base directory path
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    unit : str, optional
        Unit for output ('mm' or 'm3/s', default: 'mm')
    plot : bool, optional
        Whether to create plots (default: True)
    force_reprocess : bool, optional
        Force reprocessing even if cached files exist (default: False)
    chunk_size : int, optional
        Number of lines to read at once (default: 10000)
        
    Returns:
    --------
    pandas.DataFrame
        Processed GloGEM data
    """
    import pickle
    import os
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    from datetime import datetime, timedelta
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # Base directory paths
    glogem_dir = Path(r"/home/jberg/OneDrive/Raven_worldwide/01_data/GloGEM")
    topo_dir = Path(base_dir) / f"catchment_{gauge_id}_c" / "topo_files"
    
    # Create cache directory for processed files
    cache_dir = glogem_dir / "processed_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Cache file paths (using pickle for faster I/O)
    cache_files = {
        'icemelt': cache_dir / f"icemelt_{gauge_id}_processed.pkl",
        'snowmelt': cache_dir / f"snowmelt_{gauge_id}_processed.pkl", 
        'output': cache_dir / f"output_{gauge_id}_processed.pkl",
        'final_result': cache_dir / f"final_result_{gauge_id}_{start_date}_{end_date}_{unit}.pkl"
    }
    
    # Original file paths
    file_paths = {
        'icemelt': glogem_dir / f"GloGEM_icemelt_{gauge_id}.dat",
        'snowmelt': glogem_dir / f"GloGEM_snowmelt_{gauge_id}.dat",
        'output': glogem_dir / f"GloGEM_output_{gauge_id}.dat"
    }
    catchment_shape_file = topo_dir / "HRU.shp"
    
    # Check if final result already exists and is recent
    if cache_files['final_result'].exists() and not force_reprocess:
        print(f"Loading cached final result from {cache_files['final_result']}")
        try:
            with open(cache_files['final_result'], 'rb') as f:
                scaled_df = pickle.load(f)
            print(f"✅ Loaded {len(scaled_df)} records from cache")
            
            if plot:
                create_glogem_plots(scaled_df, gauge_id, start_date, end_date, unit)
            
            return scaled_df
        except Exception as e:
            print(f"Error loading cache: {e}. Will reprocess...")
    
    # Check if files exist
    for file_type, file_path in file_paths.items():
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Processing GloGEM data for gauge {gauge_id}...")
    
    # Parse start and end dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    def parse_glogem_file_chunked(file_path, cache_file, file_type):
        """Parse large GloGEM file in chunks and cache the result"""
        
        # Check if we have a cached version
        if cache_file.exists() and not force_reprocess:
            print(f"Loading cached {file_type} data from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache for {file_type}: {e}. Will reprocess...")
        
        print(f"Processing {file_type} file: {file_path}")
        print("This may take a few minutes for large files...")
        
        data_chunks = []
        areas = {}
        total_lines = 0
        
        # First pass: count lines for progress tracking
        print("Counting lines...")
        with open(file_path, 'r') as f:
            total_lines = sum(1 for line in f if not line.startswith(("ID", "//")))
        
        print(f"Processing {total_lines:,} data lines in chunks of {chunk_size:,}")
        
        # Second pass: process data in chunks
        with open(file_path, 'r') as f:
            lines_processed = 0
            chunk_data = []
            
            for line_num, line in enumerate(f):
                # Skip header lines
                if line.startswith("ID") or line.startswith("//"):
                    continue
                
                parts = line.strip().split()
                if len(parts) < 5:  # Skip malformed lines
                    continue
                
                try:
                    glacier_id = parts[0]
                    year = int(parts[1])
                    area = float(parts[2])
                    areas[glacier_id] = area
                    
                    # Daily values start from index 3
                    daily_values = []
                    for val in parts[3:]:
                        try:
                            daily_values.append(float(val) if val != '*' else 0.0)
                        except ValueError:
                            daily_values.append(0.0)
                    
                    # Create date for each value (starting Oct 1st of hydrological year)
                    start_date_hydro = datetime(year-1, 10, 1)
                    
                    for day, value in enumerate(daily_values):
                        date = start_date_hydro + timedelta(days=day)
                        chunk_data.append({
                            'glacier_id': glacier_id,
                            'date': date,
                            'area': area,
                            'value': value
                        })
                
                except (ValueError, IndexError) as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
                
                lines_processed += 1
                
                # Process chunk when it reaches the specified size
                if len(chunk_data) >= chunk_size:
                    data_chunks.append(pd.DataFrame(chunk_data))
                    chunk_data = []
                    
                    # Progress update
                    if lines_processed % 1000 == 0:
                        progress = (lines_processed / total_lines) * 100
                        print(f"Progress: {progress:.1f}% ({lines_processed:,}/{total_lines:,} lines)")
            
            # Process remaining data
            if chunk_data:
                data_chunks.append(pd.DataFrame(chunk_data))
        
        # Combine all chunks
        print("Combining chunks...")
        if data_chunks:
            df = pd.concat(data_chunks, ignore_index=True)
        else:
            df = pd.DataFrame()
        
        print(f"✅ Processed {len(df):,} records for {file_type}")
        
        # Cache the result
        print(f"Caching processed data to {cache_file}")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((df, areas), f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✅ Cached {file_type} data")
        except Exception as e:
            print(f"Warning: Could not cache {file_type} data: {e}")
        
        return df, areas
    
    # Parse all three files with caching
    print("\n" + "="*50)
    print("PROCESSING GLOGEM FILES")
    print("="*50)
    
    icemelt_df, icemelt_areas = parse_glogem_file_chunked(
        file_paths['icemelt'], cache_files['icemelt'], 'icemelt'
    )
    
    snowmelt_df, snowmelt_areas = parse_glogem_file_chunked(
        file_paths['snowmelt'], cache_files['snowmelt'], 'snowmelt'
    )
    
    output_df, output_areas = parse_glogem_file_chunked(
        file_paths['output'], cache_files['output'], 'output'
    )
    
    print("\n" + "="*50)
    print("PROCESSING CATCHMENT DATA")
    print("="*50)
    
    # Filter for the date range
    print(f"Filtering data for period {start_date} to {end_date}")
    icemelt_df = icemelt_df[(icemelt_df['date'] >= start) & (icemelt_df['date'] <= end)]
    snowmelt_df = snowmelt_df[(snowmelt_df['date'] >= start) & (snowmelt_df['date'] <= end)]
    output_df = output_df[(output_df['date'] >= start) & (output_df['date'] <= end)]
    
    print(f"After filtering: icemelt={len(icemelt_df):,}, snowmelt={len(snowmelt_df):,}, output={len(output_df):,}")
    
    # Read catchment shapefile
    print("Reading catchment shapefile...")
    catchment = gpd.read_file(catchment_shape_file)
    
    # Create glacier areas mapping
    glacier_areas = catchment.groupby('Glacier_Cl').agg({'Area_km2': 'sum'}).reset_index()
    glacier_area = glacier_areas['Area_km2'].sum()
    total_area = catchment['Area_km2'].sum()
    percentage = (glacier_area / total_area) * 100
    
    print(f"Glacier area: {glacier_area:.2f} km²")
    print(f"Total catchment area: {total_area:.2f} km²")
    print(f"Glaciated percentage: {percentage:.1f}%")
    
    # Create area dictionary
    area_dict = {}
    for _, row in glacier_areas.iterrows():
        glacier_id = row['Glacier_Cl'][-5:]  # Extract last 5 digits
        area_dict[glacier_id] = row['Area_km2']
    
    # Update icemelt_df with correct areas
    print("Updating ice melt areas...")
    for glacier_id, new_area in area_dict.items():
        mask = icemelt_df['glacier_id'] == glacier_id
        if mask.any():
            icemelt_df.loc[mask, 'area'] = new_area
    
    # Calculate catchment average runoff (memory efficient)
    def calculate_catchment_average_efficient(df, areas):
        print(f"Calculating catchment average for {len(df):,} records...")
        catchment_area = sum(areas.values())
        
        # Use vectorized operations for efficiency
        df = df.copy()  # Avoid SettingWithCopyWarning
        df['weighted_value'] = df['value'] * df['area']
        
        # Group by date and calculate catchment average
        daily_sum = df.groupby('date')['weighted_value'].sum().reset_index()
        daily_sum['catchment_avg'] = daily_sum['weighted_value'] / catchment_area
        
        return daily_sum[['date', 'catchment_avg']]
    
    print("\nCalculating catchment averages...")
    icemelt_avg = calculate_catchment_average_efficient(icemelt_df, area_dict)
    snowmelt_avg = calculate_catchment_average_efficient(snowmelt_df, area_dict)
    output_avg = calculate_catchment_average_efficient(output_df, area_dict)
    
    # Combine dataframes efficiently
    print("Merging dataframes...")
    result_df = pd.merge(icemelt_avg, snowmelt_avg, on='date', suffixes=('_icemelt', '_snowmelt'))
    result_df = pd.merge(result_df, output_avg, on='date')
    result_df.columns = ['date', 'glacier_melt', 'snowmelt', 'total_output']
    
    # Apply scaling factor
    scaling_factor = percentage / 100
    scaled_df = result_df.copy()
    scaled_df['glacier_melt'] = scaled_df['glacier_melt'] * scaling_factor
    scaled_df['snowmelt'] = scaled_df['snowmelt'] * scaling_factor
    scaled_df['total_output'] = scaled_df['total_output'] * scaling_factor
    
    # Convert units if necessary
    if unit == 'm3':
        catchment_area_m2 = total_area * 1000000  # Convert km² to m²
        scaled_df['glacier_melt'] = scaled_df['glacier_melt'] * catchment_area_m2 / 1000
        scaled_df['snowmelt'] = scaled_df['snowmelt'] * catchment_area_m2 / 1000
        scaled_df['total_output'] = scaled_df['total_output'] * catchment_area_m2 / 1000
    
    # Cache the final result
    print(f"Caching final result to {cache_files['final_result']}")
    try:
        with open(cache_files['final_result'], 'wb') as f:
            pickle.dump(scaled_df, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✅ Cached final result")
    except Exception as e:
        print(f"Warning: Could not cache final result: {e}")
    
    print(f"\n✅ Processing complete! Final dataset: {len(scaled_df):,} records")
    
    # Create plots if requested
    if plot:
        create_glogem_plots(scaled_df, gauge_id, start_date, end_date, unit)
    
    return scaled_df


def create_glogem_plots(scaled_df, gauge_id, start_date, end_date, unit):
    """Create plots for GloGEM data"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # Add time columns
    scaled_df['year'] = scaled_df['date'].dt.year
    scaled_df['month'] = scaled_df['date'].dt.month
    
    # ✅ Define unit_label based on unit parameter
    unit_label = 'mm' if unit.lower() == 'mm' else 'm³/s'
    
    # Monthly regime plot
    monthly_regime = scaled_df.groupby('month').agg({
        'glacier_melt': 'mean',
        'snowmelt': 'mean',
        'total_output': 'mean'
    }).reset_index()
    
    print("Creating monthly regime plot...")
    plt.figure(figsize=(12, 6))
    
    # ✅ Updated colors: light grey for glacier melt, light blue for snowmelt
    plt.plot(monthly_regime['month'], monthly_regime['glacier_melt'], 
             color='lightgrey', label='Glacier Melt', linewidth=2)
    plt.plot(monthly_regime['month'], monthly_regime['snowmelt'], 
             color='lightblue', label='Snowmelt', linewidth=2)
    plt.plot(monthly_regime['month'], monthly_regime['total_output'], 
             'k-', label='Total Output', linewidth=2)
    
    # ✅ Removed x-axis label
    # plt.xlabel('Month', fontsize=12)  # Commented out
    plt.ylabel(f'Average Runoff ({unit_label})', fontsize=18)  # ✅ Bigger font size
    plt.title(f'Monthly Regime for Catchment {gauge_id} ({start_date} to {end_date})', fontsize=20)  # ✅ Bigger font size
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=16)  # ✅ Bigger font size
    plt.yticks(fontsize=16)  # ✅ Bigger font size
    plt.legend(fontsize=16)  # ✅ Bigger font size
    plt.tight_layout()
    plt.show()
    
    # Daily average plot
    scaled_df['day_of_year'] = scaled_df['date'].dt.dayofyear
    daily_avg = scaled_df.groupby('day_of_year').agg({
        'glacier_melt': 'mean',
        'snowmelt': 'mean', 
        'total_output': 'mean'
    }).reset_index()
    
    daily_avg['date'] = pd.to_datetime('2001-01-01') + pd.to_timedelta(daily_avg['day_of_year'] - 1, unit='days')
    
    print("Creating daily average plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # ✅ Updated colors and bigger fonts
    ax.plot(daily_avg['date'], daily_avg['total_output'], 'k-', label='Total Output', linewidth=2)
    ax.plot(daily_avg['date'], daily_avg['snowmelt'], color='lightblue', label='Snowmelt', linewidth=1.5)
    ax.plot(daily_avg['date'], daily_avg['glacier_melt'], color='lightgrey', label='Glacier Melt', linewidth=1.5)
    
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # ✅ Removed x-axis label and increased font sizes
    # ax.set_xlabel('Month', fontsize=12)  # Commented out
    ax.set_ylabel(f'Average Runoff ({unit_label})', fontsize=18)  # ✅ Bigger font size
    ax.set_title('Daily Average Runoff Values (Averaged Over All Years)', fontsize=20)  # ✅ Bigger font size
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=16)  # ✅ Bigger font size
    
    # ✅ Bigger tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    plt.show()


def clear_glogem_cache(gauge_id=None):
    """
    Clear cached GloGEM data to force reprocessing.
    
    Parameters:
    -----------
    gauge_id : str, optional
        If provided, only clear cache for this gauge. If None, clear all cache.
    """
    cache_dir = Path(r"/home/jberg/OneDrive/Raven_worldwide/01_data/GloGEM/processed_cache")
    
    if not cache_dir.exists():
        print("No cache directory found")
        return
    
    if gauge_id:
        # Clear cache for specific gauge
        cache_files = list(cache_dir.glob(f"*{gauge_id}*"))
        for file in cache_files:
            file.unlink()
            print(f"Deleted: {file}")
    else:
        # Clear all cache
        cache_files = list(cache_dir.glob("*.pkl"))
        for file in cache_files:
            file.unlink()
            print(f"Deleted: {file}")
    
    print(f"✅ Cache cleared!")


def get_cache_info(gauge_id=None):
    """
    Get information about cached GloGEM files.
    
    Parameters:
    -----------
    gauge_id : str, optional
        If provided, show info only for this gauge
    """
    cache_dir = Path(r"/home/jberg/OneDrive/Raven_worldwide/01_data/GloGEM/processed_cache")
    
    if not cache_dir.exists():
        print("No cache directory found")
        return
    
    if gauge_id:
        cache_files = list(cache_dir.glob(f"*{gauge_id}*"))
    else:
        cache_files = list(cache_dir.glob("*.pkl"))
    
    if not cache_files:
        print("No cache files found")
        return
    
    print("Cache files:")
    print("-" * 50)
    for file in cache_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(file.stat().st_mtime)
        print(f"{file.name:<40} {size_mb:>8.1f} MB  {modified.strftime('%Y-%m-%d %H:%M')}")
    
    total_size = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
    print("-" * 50)
    print(f"Total cache size: {total_size:.1f} MB")

#--------------------------------------------------------------------------------

def get_hydrograph_data(gauge_id, config=None, start_date=None, end_date=None, unit='m3s', base_dir=None, catchment_area=None, plot=True):
    """
    Read and process HBV hydrograph data for a specific gauge ID with support for multiple configurations.
    
    Parameters:
    ----------
    gauge_id : str or int
        The gauge ID to process
    config : str or list, default=None
        Configuration(s) to process. Can be a single string ('c', 'nc', 'c_multi', etc.) 
        or a list of configurations. If None, uses 'nc'.
    start_date : str, default=None
        Start date for filtering data (format: 'YYYY-MM-DD')
    end_date : str, default=None
        End date for filtering data (format: 'YYYY-MM-DD')
    unit : str, default='m3s'
        Unit for streamflow values, either 'm3s' (cubic meters per second) or 'mm'
    base_dir : str, default=None
        Base directory for simulation results, if None uses current directory
    catchment_area : float, default=None
        Catchment area in km². If provided, this will be used for unit conversion.
        If None, will use predefined values or attempt to find the area from HRU file.
    plot : bool, default=True
        Whether to create and display a plot
        
    Returns:
    -------
    dict
        Dictionary with configuration names as keys and DataFrames with hydrograph data as values
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os
    
    # Set base directory if not provided
    if base_dir is None:
        base_dir = os.getcwd()
    
    # Format gauge_id as string
    gauge_id = str(gauge_id)
    
    # Handle config parameter
    if config is None:
        config = ['nc']
    elif isinstance(config, str):
        config = [config]
    
    # Dictionary to store results
    results = {}
    area_km2 = None
    
    # Process each configuration
    for cfg in config:
        try:
            # Build path to the data file
            folder_path = Path(base_dir) / f"catchment_{gauge_id}_{cfg}" / "HBV" / "output"
            file_path = folder_path / f"{gauge_id}_HBV_Hydrographs.csv"
            
            print(f"Processing configuration: {cfg}")
            print(f"Reading file: {file_path}")
            
            # Check if file exists
            if not file_path.exists():
                print(f"Data file not found at {file_path}")
                continue
            
            # Read the data
            df = pd.read_csv(file_path)
            
            print(f"Available columns: {df.columns.tolist()}")
            
            # Convert the date column to datetime format
            df['Date'] = pd.to_datetime(df['date'])
            
            # Filter data by date range if specified
            if start_date:
                df = df[df['Date'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['Date'] <= pd.to_datetime(end_date)]
            
            # Ensure we have data after filtering
            if df.empty:
                print(f"No data found for the specified date range in configuration {cfg}")
                continue
            
            # Find observed and simulated columns based on gauge_id
            obs_col = None
            sim_col = None
            
            # Expected column name patterns
            expected_obs_pattern = f"{gauge_id} (observed)"
            expected_sim_pattern = f"{gauge_id} [m3/s]"
            
            # Look for matching columns
            for col in df.columns:
                if expected_obs_pattern in col:
                    obs_col = col
                elif col == expected_sim_pattern or (gauge_id in col and 'observed' not in col and '[m3/s]' in col):
                    sim_col = col
            
            if not obs_col:
                print(f"Warning: Observed column not found. Expected pattern: '{expected_obs_pattern}'")
                print(f"Will try to find any column with 'observed' in the name")
                for col in df.columns:
                    if 'observed' in col:
                        obs_col = col
                        break
            
            if not sim_col:
                print(f"Warning: Simulated column not found. Expected pattern: '{expected_sim_pattern}'")
                print(f"Will try to find any column with the gauge ID and [m3/s]")
                for col in df.columns:
                    if gauge_id in col and 'observed' not in col and '[m3/s]' in col:
                        sim_col = col
                        break
            
            if not obs_col or not sim_col:
                print(f"Required columns not found. Available columns: {df.columns.tolist()}")
                continue
            
            print(f"Using observed column: {obs_col}")
            print(f"Using simulated column: {sim_col}")
            
            # Extract observed and simulated columns
            result_df = df[['Date', obs_col, sim_col]].copy()
            result_df.columns = ['Date', 'Qobs', 'Qsim']
            
            # Convert units if necessary (assuming input data is in m3/s)
            if unit.lower() == 'mm':
                # Determine catchment area (only need to do this once)
                if catchment_area is not None:
                    area_km2 = catchment_area
                elif area_km2 is None:  # Only calculate if not already done
                    # Try to find area from predefined values
                    catchment_areas = {
                        # area in km²
                        "2269": 285.0,  # Example value - replace with actual area
                        # Add other gauge IDs as needed
                    }
                    
                    if gauge_id in catchment_areas:
                        area_km2 = catchment_areas[gauge_id]
                    else:
                        # Try to find area from shapefile if available
                        try:
                            import geopandas as gpd
                            shapefile_path = Path(base_dir) / f"catchment_{gauge_id}_{cfg}" / "topo_files" / "HRU.shp"
                            if shapefile_path.exists():
                                gdf = gpd.read_file(shapefile_path)
                                if 'Area_km2' in gdf.columns:
                                    area_km2 = gdf['Area_km2'].sum()
                                else:
                                    # Calculate area if not available directly
                                    area_km2 = gdf.to_crs({'proj': 'cea'}).area.sum() / 1e6
                                print(f"Catchment area determined from shapefile: {area_km2:.2f} km²")
                            else:
                                print(f"Catchment area not available for gauge {gauge_id}")
                                # Use a default value or continue without conversion
                                area_km2 = 100.0  # Default value
                        except ImportError:
                            print(f"geopandas not installed and catchment area not provided for gauge {gauge_id}")
                            continue
                        except Exception as e:
                            print(f"Error determining catchment area for gauge {gauge_id}: {e}")
                            continue
                
                # Convert from m³/s to mm/day
                # 1 m³/s = 86400 seconds in a day / (area in m²) * 1000 mm/m
                conversion_factor = 86400 / (area_km2 * 10**6) * 1000
                
                result_df['Qobs'] = result_df['Qobs'] * conversion_factor
                result_df['Qsim'] = result_df['Qsim'] * conversion_factor
            
            # Add monthly info for later plotting
            result_df['Month'] = result_df['Date'].dt.month
            result_df['Year'] = result_df['Date'].dt.year
            
            # Calculate performance metrics
            obs_mean = result_df['Qobs'].mean()
            sim_mean = result_df['Qsim'].mean()
            
            # Nash-Sutcliffe Efficiency
            nse = 1 - (np.sum((result_df['Qobs'] - result_df['Qsim'])**2) / 
                       np.sum((result_df['Qobs'] - obs_mean)**2))
            
            # Percent Bias
            pbias = 100 * (np.sum(result_df['Qsim'] - result_df['Qobs']) / np.sum(result_df['Qobs']))
            
            # Store metrics in the dataframe
            result_df.attrs['nse'] = nse
            result_df.attrs['pbias'] = pbias
            result_df.attrs['obs_mean'] = obs_mean
            result_df.attrs['sim_mean'] = sim_mean
            result_df.attrs['config'] = cfg
            
            if unit.lower() == 'mm' and area_km2 is not None:
                result_df.attrs['area_km2'] = area_km2
            
            # Add the result to the dictionary
            results[cfg] = result_df
            
        except Exception as e:
            print(f"Error processing configuration {cfg}: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("No valid data found for any configuration.")
        return None
    
    if plot:
        unit_label = "mm/day" if unit.lower() == 'mm' else "m³/s"
        
        # Create monthly averages for plotting
        plt.figure(figsize=(12, 8))
        
        # Colors and line styles for different configurations
        colors = ['#75D054FF', '#24868EFF', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
        
        # Plot each configuration
        for i, (cfg, df) in enumerate(results.items()):
            monthly_avg = df.groupby('Month')[['Qobs', 'Qsim']].mean().reset_index()
            
            # Add month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_avg['MonthName'] = monthly_avg['Month'].apply(lambda x: month_names[x-1])
            
            # Plot simulated data with different colors/styles
            color_idx = i % len(colors)
            style_idx = i % len(line_styles)
            plt.plot(monthly_avg['Month'], monthly_avg['Qsim'], 
                   color=colors[color_idx], linestyle=line_styles[style_idx],
                   linewidth=2, label=f'Sim - {cfg}')
            
            # Plot observed data only once (should be the same for all configurations)
            if i == 0:
                plt.plot(monthly_avg['Month'], monthly_avg['Qobs'], 'k-', 
                       linewidth=2.5, label='Observed')
        
        # Set x-ticks to month names
        plt.xticks(range(1, 13), month_names)
        
        # Add labels and title
        plt.title(f"Monthly Average Streamflow - Gauge {gauge_id}", fontsize=16)
        plt.xlabel("Month", fontsize=14)
        plt.ylabel(f"Streamflow ({unit_label})", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=12)
        
        # Add performance metrics text box
        metrics_text = "Performance Metrics:\n" + "-" * 20 + "\n"
        for cfg, df in results.items():
            metrics_text += f"{cfg}: NSE={df.attrs['nse']:.3f}, PBIAS={df.attrs['pbias']:.2f}%\n"
        
        if unit.lower() == 'mm' and area_km2 is not None:
            metrics_text += f"\nCatchment Area: {area_km2:.2f} km²"
            
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                  bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    return results

#--------------------------------------------------------------------------------

def analyze_glacier_melt(gauge_id, start_date, end_date, base_dir, config=None, unit='mm', 
                         catchment_area=None, plot=True):
    """
    Analyze glacier melt data for a specific gauge and time period.
    
    Parameters:
    -----------
    gauge_id : str
        Identifier for the gauge
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    base_dir : str
        Base directory path
    config : str or list, default=None
        Configuration(s) to process. Can be a single string ('c', 'nc', 'c_multi', etc.) 
        or a list of configurations. If None, uses 'nc'.
    unit : str, optional
        Unit for the output ('mm' or 'm3/s', default: 'mm')
    catchment_area : float, optional
        Catchment area in km², required if unit is 'm3/s'
    plot : bool, optional
        Whether to create and display plots (default: True)
        
    Returns:
    --------
    dict
        Dictionary with configuration names as keys and DataFrames with glacier melt data as values
    """
    # Validate inputs
    if unit == 'm3/s' and catchment_area is None:
        raise ValueError("Catchment area must be provided when unit is 'm3/s'")
    
    # Handle config parameter
    if config is None:
        config = ['nc']  # Default to non-coupled
    elif isinstance(config, str):
        config = [config]  # Convert single string to list
    
    # Dictionary to store results for each configuration
    results = {}
    
    # Process each configuration
    for cfg in config:
        try:
            # Construct file path
            file_path = os.path.join(base_dir, f"catchment_{gauge_id}_{cfg}", "HBV", "output", 
                                   f"{gauge_id}_HBV_FROM_GLACIER_ICE_Daily_Average_BySubbasin.csv")
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File not found for configuration '{cfg}': {file_path}")
                continue
            
            print(f"Processing configuration: {cfg}")
            print(f"Reading file: {file_path}")
            
            # Read the file
            df = pd.read_csv(file_path, skiprows=1, parse_dates=['day'])
            df['date'] = pd.to_datetime(df['day'])
            
            # Sort by date first (important for calculating daily melt)
            df = df.sort_values('date')
            
            # Important: Get cumulative data for full period BEFORE filtering dates
            # This ensures we can calculate correct daily melt for the start of our period
            df['glacier_melt_full'] = df['mean'].diff().fillna(0)  # Use 0 for first record
            
            # Apply negative values check/correction (glacier melt should be positive)
            df['glacier_melt_full'] = df['glacier_melt_full'].clip(lower=0)
            
            # Now filter by date range after calculating differences
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df['date'] >= start) & (df['date'] <= end)]
            
            # Check if we have data after filtering
            if df.empty:
                print(f"No data found for the specified date range in configuration {cfg}")
                continue
            
            # Copy the pre-calculated melt values to our working column
            df['glacier_melt'] = df['glacier_melt_full']
            
            # Handle first value specifically
            # If the first date is January, February, or March (winter months), 
            # set the value to zero as glacier melt is unlikely
            first_month = df.iloc[0]['date'].month
            if first_month in [1, 2, 3, 12]:  # Winter months
                print(f"First value is in month {first_month}, setting to zero (winter month)")
                df.loc[df.index[0], 'glacier_melt'] = 0
            
            # Add month and year columns for aggregation
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            
            # Convert units if necessary
            if unit == 'm3/s':
                # Convert from mm/day to m³/s
                # 1 mm over catchment area (km²) = catchment_area * 1000 m³
                # Divide by 86400 seconds per day
                df['glacier_melt'] = df['glacier_melt'] * catchment_area * 1000 / 86400
                unit_label = 'm³/s'
            else:
                unit_label = 'mm/day'
            
            # Create result DataFrame with only date and glacier melt
            result_df = df[['date', 'glacier_melt', 'month', 'year']]
            
            # Add configuration info to the dataframe attributes
            result_df.attrs['config'] = cfg
            result_df.attrs['unit'] = unit
            result_df.attrs['unit_label'] = unit_label
            
            # Store the result in the dictionary
            results[cfg] = result_df
            
            # Create plot if requested and only for this configuration
            if plot:
                # Calculate monthly regime (average across all years for each month)
                monthly_regime = df.groupby('month')['glacier_melt'].mean().reset_index()
                
                # Create month names for x-axis
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                # Set up the plot style
                plt.figure(figsize=(12, 6))
                sns.set_style("whitegrid")
                
                # Plot monthly regime
                plt.plot(range(1, 13), monthly_regime['glacier_melt'], marker='o', 
                         linestyle='-', linewidth=2, markersize=8, color='#24868EFF')
                
                # Set x-axis labels to month names
                plt.xticks(range(1, 13), month_names)
                
                # Set plot labels and title
                plt.title(f'Monthly Glacier Melt Regime for Gauge {gauge_id} - {cfg} Configuration', fontsize=16)
                plt.xlabel('Month', fontsize=12)
                plt.ylabel(f'Glacier Melt ({unit_label})', fontsize=12)
                
                # Add grid for better readability
                plt.grid(True, alpha=0.3)
                
                # Calculate annual statistics for display
                annual_melt = monthly_regime['glacier_melt'].mean() * 365/12
                max_month = monthly_regime.loc[monthly_regime['glacier_melt'].idxmax()]
                
                # Add statistics text box
                stats_text = (
                    f"Configuration: {cfg}\n"
                    f"Period: {start_date} to {end_date}\n"
                    f"Annual glacier melt: {annual_melt:.1f} {unit_label.split('/')[0]}/year\n"
                    f"Peak month: {month_names[int(max_month['month'])-1]} ({max_month['glacier_melt']:.2f} {unit_label})"
                )
                
                # Add text box to the plot
                plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
                         fontsize=10, bbox=dict(facecolor='white', alpha=0.8),
                         verticalalignment='top')
                
                plt.tight_layout()
                plt.show()
                
                # Also create a time series plot to visualize the data
                plt.figure(figsize=(14, 6))
                plt.plot(df['date'], df['glacier_melt'], '-', linewidth=1, alpha=0.7)
                plt.title(f'Daily Glacier Melt - Gauge {gauge_id} ({cfg})', fontsize=16)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel(f'Glacier Melt ({unit_label})', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
        
        except Exception as e:
            print(f"Error processing configuration {cfg}: {e}")
    
    # If we have multiple configurations and plot is True, create a comparison plot
    if plot and len(results) > 1:
        try:
            plt.figure(figsize=(12, 6))
            
            # Colors for different configurations
            colors = ['#75D054FF', '#24868EFF', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            # Plot monthly regime for each configuration
            for i, (cfg, result) in enumerate(results.items()):
                monthly_regime = result.groupby('month')['glacier_melt'].mean().reset_index()
                color_idx = i % len(colors)
                
                plt.plot(range(1, 13), monthly_regime['glacier_melt'], marker='o', 
                        linestyle='-', linewidth=2, markersize=6, color=colors[color_idx], 
                        label=f'Config: {cfg}')
            
            # Set x-axis labels to month names
            plt.xticks(range(1, 13), month_names)
            
            # Set plot labels and title
            plt.title(f'Monthly Glacier Melt Regime Comparison for Gauge {gauge_id}', fontsize=16)
            plt.xlabel('Month', fontsize=12)
            plt.ylabel(f'Glacier Melt ({unit_label})', fontsize=12)
            
            # Add grid and legend
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add statistics text for each configuration
            stats_text = f"Period: {start_date} to {end_date}\n\n"
            for i, (cfg, result) in enumerate(results.items()):
                monthly_regime = result.groupby('month')['glacier_melt'].mean().reset_index()
                annual_melt = monthly_regime['glacier_melt'].mean() * 365/12
                max_month = monthly_regime.loc[monthly_regime['glacier_melt'].idxmax()]
                
                stats_text += (
                    f"Config {cfg}:\n"
                    f"Annual melt: {annual_melt:.1f} {unit_label.split('/')[0]}/year\n"
                    f"Peak month: {month_names[int(max_month['month'])-1]}\n\n"
                )
            
            # Add text box to the plot
            plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
                      bbox=dict(facecolor='white', alpha=0.8), verticalalignment='bottom')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating comparison plot: {e}")
    
    # Return dictionary of results
    if not results:
        print("No valid data found for any configuration.")
        return None
        
    return results

#---------------------------------------------------------------------------------

def analyze_snow_melt(gauge_id, start_date, end_date, base_dir, config=None, unit='mm', 
                     catchment_area=None, plot=True):
    """
    Analyze snow melt data for a specific gauge and time period.
    
    Parameters:
    -----------
    gauge_id : str
        Identifier for the gauge
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    base_dir : str
        Base directory path
    config : str or list, default=None
        Configuration(s) to process. Can be a single string ('c', 'nc', 'c_multi', etc.) 
        or a list of configurations. If None, uses 'nc'.
    unit : str, optional
        Unit for the output ('mm' or 'm3/s', default: 'mm')
    catchment_area : float, optional
        Catchment area in km², required if unit is 'm3/s'
    plot : bool, optional
        Whether to create and display plots (default: True)
        
    Returns:
    --------
    dict
        Dictionary with configuration names as keys and DataFrames with snow melt data as values
    """
    # Validate inputs
    if unit == 'm3/s' and catchment_area is None:
        raise ValueError("Catchment area must be provided when unit is 'm3/s'")
    
    # Handle config parameter
    if config is None:
        config = ['nc']  # Default to non-coupled
    elif isinstance(config, str):
        config = [config]  # Convert single string to list
    
    # Dictionary to store results for each configuration
    results = {}
    
    # Process each configuration
    for cfg in config:
        try:
            # Construct file path
            file_path = os.path.join(base_dir, f"catchment_{gauge_id}_{cfg}", "HBV", "output", 
                                   f"{gauge_id}_HBV_TO_LAKE_STORAGE_Daily_Average_BySubbasin.csv")
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File not found for configuration '{cfg}': {file_path}")
                continue
            
            print(f"Processing configuration: {cfg}")
            print(f"Reading file: {file_path}")
            
            # Read the file
            df = pd.read_csv(file_path, skiprows=1, parse_dates=['day'])
            df['date'] = pd.to_datetime(df['day'])
            
            # Sort by date first (important for calculating daily melt)
            df = df.sort_values('date')
            
            # Calculate daily snow melt from cumulative values
            df['snow_melt_full'] = df['mean'].diff().fillna(0)  # Use 0 for first record
            
            # Apply negative values check/correction (snowmelt should be positive)
            df['snow_melt_full'] = df['snow_melt_full'].clip(lower=0)
            
            # Now filter by date range after calculating differences
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df['date'] >= start) & (df['date'] <= end)]
            
            # Check if we have data after filtering
            if df.empty:
                print(f"No data found for the specified date range in configuration {cfg}")
                continue
            
            # Copy the pre-calculated melt values to our working column
            df['snow_melt'] = df['snow_melt_full']
            
            # Handle first value specifically
            # If the first date is in summer, set the value appropriately
            first_month = df.iloc[0]['date'].month
            if first_month in [6, 7, 8]:  # Summer months, might need to adjust
                print(f"First value is in month {first_month}, may need adjustment")
            
            # Add month and year columns for aggregation
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            
            # Convert units if necessary
            if unit == 'm3/s':
                # Convert from mm/day to m³/s
                # 1 mm over catchment area (km²) = catchment_area * 1000 m³
                # Divide by 86400 seconds per day
                df['snow_melt'] = df['snow_melt'] * catchment_area * 1000 / 86400
                unit_label = 'm³/s'
            else:
                unit_label = 'mm/day'
            
            # Create result DataFrame with only date and snow melt
            result_df = df[['date', 'snow_melt', 'month', 'year']]
            
            # Add configuration info to the dataframe attributes
            result_df.attrs['config'] = cfg
            result_df.attrs['unit'] = unit
            result_df.attrs['unit_label'] = unit_label
            
            # Store the result in the dictionary
            results[cfg] = result_df
            
            # Create plot if requested and only for this configuration
            if plot:
                # Calculate monthly regime (average across all years for each month)
                monthly_regime = df.groupby('month')['snow_melt'].mean().reset_index()
                
                # Create month names for x-axis
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                # Set up the plot style
                plt.figure(figsize=(12, 6))
                sns.set_style("whitegrid")
                
                # Plot monthly regime
                plt.plot(range(1, 13), monthly_regime['snow_melt'], marker='o', 
                         linestyle='-', linewidth=2, markersize=8, color='#24868EFF')
                
                # Set x-axis labels to month names
                plt.xticks(range(1, 13), month_names)
                
                # Set plot labels and title
                plt.title(f'Monthly Snow Melt Regime for Gauge {gauge_id} - {cfg} Configuration', fontsize=16)
                plt.xlabel('Month', fontsize=12)
                plt.ylabel(f'Snow Melt ({unit_label})', fontsize=12)
                
                # Add grid for better readability
                plt.grid(True, alpha=0.3)
                
                # Calculate annual statistics for display
                annual_melt = monthly_regime['snow_melt'].mean() * 365/12
                max_month = monthly_regime.loc[monthly_regime['snow_melt'].idxmax()]
                
                # Add statistics text box
                stats_text = (
                    f"Configuration: {cfg}\n"
                    f"Period: {start_date} to {end_date}\n"
                    f"Annual snow melt: {annual_melt:.1f} {unit_label.split('/')[0]}/year\n"
                    f"Peak month: {month_names[int(max_month['month'])-1]} ({max_month['snow_melt']:.2f} {unit_label})"
                )
                
                # Add text box to the plot
                plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
                         fontsize=10, bbox=dict(facecolor='white', alpha=0.8),
                         verticalalignment='top')
                
                plt.tight_layout()
                plt.show()
                
                # Also create a time series plot to visualize the data
                plt.figure(figsize=(14, 6))
                plt.plot(df['date'], df['snow_melt'], '-', linewidth=1, alpha=0.7)
                plt.title(f'Daily Snow Melt - Gauge {gauge_id} ({cfg})', fontsize=16)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel(f'Snow Melt ({unit_label})', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
        
        except Exception as e:
            print(f"Error processing configuration {cfg}: {e}")
    
    # If we have multiple configurations and plot is True, create a comparison plot
    if plot and len(results) > 1:
        try:
            plt.figure(figsize=(12, 6))
            
            # Colors for different configurations
            colors = ['#75D054FF', '#24868EFF', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            # Plot monthly regime for each configuration
            for i, (cfg, result) in enumerate(results.items()):
                monthly_regime = result.groupby('month')['snow_melt'].mean().reset_index()
                color_idx = i % len(colors)
                
                plt.plot(range(1, 13), monthly_regime['snow_melt'], marker='o', 
                        linestyle='-', linewidth=2, markersize=6, color=colors[color_idx], 
                        label=f'Config: {cfg}')
            
            # Set x-axis labels to month names
            plt.xticks(range(1, 13), month_names)
            
            # Set plot labels and title
            plt.title(f'Monthly Snow Melt Regime Comparison for Gauge {gauge_id}', fontsize=16)
            plt.xlabel('Month', fontsize=12)
            plt.ylabel(f'Snow Melt ({unit_label})', fontsize=12)
            
            # Add grid and legend
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add statistics text for each configuration
            stats_text = f"Period: {start_date} to {end_date}\n\n"
            for i, (cfg, result) in enumerate(results.items()):
                monthly_regime = result.groupby('month')['snow_melt'].mean().reset_index()
                annual_melt = monthly_regime['snow_melt'].mean() * 365/12
                max_month = monthly_regime.loc[monthly_regime['snow_melt'].idxmax()]
                
                stats_text += (
                    f"Config {cfg}:\n"
                    f"Annual melt: {annual_melt:.1f} {unit_label.split('/')[0]}/year\n"
                    f"Peak month: {month_names[int(max_month['month'])-1]}\n\n"
                )
            
            # Add text box to the plot
            plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
                      bbox=dict(facecolor='white', alpha=0.8), verticalalignment='bottom')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating comparison plot: {e}")
    
    # Return dictionary of results
    if not results:
        print("No valid data found for any configuration.")
        return None
        
    return results

def contributions_nc(gauge_id, start_date, end_date, base_dir, unit='mm', catchment_area=None):
    """
    Analyze contributions to streamflow for a specific gauge and time period.
    
    Parameters:
    -----------
    gauge_id : str
        Identifier for the gauge
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    base_dir : str
        Base directory path
    unit : str, optional
        Unit for the output ('mm' or 'm3/s', default: 'mm')
    catchment_area : float, optional
        Catchment area in km², required if unit is 'm3/s'
        
    Returns:
    --------
    DataFrame
        DataFrame with date and contributions values
    """

    # define paths
    topo_dir = Path(base_dir) / f"catchment_{gauge_id}_nc" / "topo_files"
    output_dir = Path(base_dir) / "HBV" / "output"
    
    # define files
    catchment_shape_file = topo_dir / "HRU.shp"
    streamflow_file = output_dir / f"{gauge_id}_HBV_Hydrographs.csv"
    
    # Calculate catchment area from shapefile
    if not os.path.exists(catchment_shape_file):
        raise FileNotFoundError(f"Shapefile not found: {catchment_shape_file}")
        
    gdf = gpd.read_file(catchment_shape_file)
    # Area is typically in the units of the CRS - convert to km²
    if 'Area_km2' in gdf.columns:
            catchment_area_km2 = gdf['Area_km2'].sum()
    else:
        # Calculate area if not available directly
        catchment_area_km2 = gdf.to_crs({'proj': 'cea'}).area.sum() / 1e6
     
    # calculate snowmelt for whole catchment    
    snow_melt = analyze_snow_melt(gauge_id, start_date, end_date, base_dir, coupled=False, 
                                 unit=unit, catchment_area=catchment_area_km2)
    
    # calculate glacier melt for the whole catchment
    glacier_melt = analyze_glacier_melt(gauge_id, start_date, end_date, base_dir, 
                                      unit=unit, catchment_area=catchment_area_km2)
    
    #  Calculate streamflow for whole catchment
    streamflow = get_hydrograph_data(gauge_id, coupled=False, start_date=start_date, end_date=end_date,
                                   unit=unit, base_dir=base_dir, catchment_area=catchment_area_km2)
    
    # Make sure all dataframes have datetime index for proper merging
    snow_melt['date'] = pd.to_datetime(snow_melt['date'])
    glacier_melt['date'] = pd.to_datetime(glacier_melt['date'])
    streamflow['Date'] = pd.to_datetime(streamflow['Date'])

    # Rename columns for consistency
    streamflow = streamflow.rename(columns={'Date': 'date'})

    # Merge all dataframes
    merged_df = pd.merge(snow_melt, glacier_melt, on='date', suffixes=('_snow', '_glacier'))
    merged_df = pd.merge(merged_df, streamflow, on='date')

    # Keep only necessary columns
    merged_df = merged_df[['date', 'snow_melt', 'glacier_melt', 'Qobs', 'Qsim', 
                          'month_snow', 'year_snow']]  # Using snow month/year as they should be the same

    # Rename for clarity
    merged_df = merged_df.rename(columns={'month_snow': 'month', 'year_snow': 'year'})

    # Handle any NaN values
    merged_df = merged_df.fillna(0)
    
    # Calculate monthly averages
    monthly_avgs = merged_df.groupby('month').agg({
        'snow_melt': 'mean',
        'glacier_melt': 'mean',
        'Qobs': 'mean',
        'Qsim': 'mean'
    }).reset_index()

    # Define month names for better x-axis labels
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Set the y-axis label based on unit
    if unit.lower() == 'mm':
        y_label = 'Flow (mm/day)'
    elif unit.lower() == 'm3/s':
        y_label = 'Flow (m³/s)'
    else:
        y_label = f'Flow ({unit})'

    # Create plot
    plt.figure(figsize=(12, 7))

    # Plot individual melt components as filled areas with transparency
    # Plot glacier melt
    plt.plot(monthly_avgs['month'], monthly_avgs['glacier_melt'], 
            color='#75D054', linewidth=2, label='Glacier Melt')
    plt.fill_between(monthly_avgs['month'], 0, monthly_avgs['glacier_melt'], 
                    color='#75D054', alpha=0.4)

    # Plot snow melt
    plt.plot(monthly_avgs['month'], monthly_avgs['snow_melt'], 
            color='#AED6F1', linewidth=2, label='Snow Melt')
    plt.fill_between(monthly_avgs['month'], 0, monthly_avgs['snow_melt'], 
                    color='#AED6F1', alpha=0.4)

    # Plot observed and simulated streamflow as lines
    plt.plot(monthly_avgs['month'], monthly_avgs['Qobs'], 'k-', linewidth=2.5, 
            label='Observed Streamflow')
    plt.plot(monthly_avgs['month'], monthly_avgs['Qsim'], color='#24868E', linewidth=2, 
            linestyle='--', label='Simulated Streamflow')

    # Configure plot
    plt.xticks(monthly_avgs['month'], month_names)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel(y_label, fontsize=12)  # Dynamic y-axis label
    plt.title('Monthly Hydrological Regime with Melt Components', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.show()
    
    return merged_df

#--------------------------------------------------------------------------------

def contribution_plots(contribution_df):
    
    # Define month names for better x-axis labels
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Calculate monthly averages
    monthly_avgs = contribution_df.groupby('month').agg({
        'snow_melt': 'mean',
        'glacier_melt': 'mean',
        'Qobs': 'mean',
        'Qsim': 'mean'
    }).reset_index()
    
    # Create a bar chart comparing snow and glacier melt
    plt.figure(figsize=(12, 6))

    bar_width = 0.35
    index = np.arange(len(month_names))

    plt.bar(index - bar_width/2, monthly_avgs['glacier_melt'], bar_width, 
            label='Glacier Melt', color='#75D054', alpha=0.8)
    plt.bar(index + bar_width/2, monthly_avgs['snow_melt'], bar_width, 
            label='Snow Melt', color='#AED6F1', alpha=0.8)

    # Add lines to show total melt
    plt.plot(index, monthly_avgs['glacier_melt'] + monthly_avgs['snow_melt'], 
            'r--', linewidth=1.5, label='Total Melt')

    # Configure plot
    plt.xticks(index, month_names)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Melt (mm/day)', fontsize=12)
    plt.title('Monthly Snow vs. Glacier Melt Comparison', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    plt.tight_layout()

    plt.show()
    
    
    # Create a pivot table with years as rows and months as columns
    pivot_df = contribution_df.pivot_table(
    index='year', 
    columns='month',
    values=['snow_melt', 'glacier_melt', 'Qsim']
    )

        # Plot heatmap for glacier melt
    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    sns.heatmap(pivot_df['glacier_melt'], cmap='Greens', 
        xticklabels=month_names, annot=False, fmt=".1f", cbar_kws={'label': 'Glacier Melt (mm/day)'})
    plt.title('Glacier Melt - Monthly Values by Year', fontsize=14)
    plt.ylabel('Year', fontsize=12)

    plt.subplot(3, 1, 2)
    sns.heatmap(pivot_df['snow_melt'], cmap='Blues', 
                xticklabels=month_names, annot=False, fmt=".1f", cbar_kws={'label': 'Snow Melt (mm/day)'})
    plt.title('Snow Melt - Monthly Values by Year', fontsize=14)
    plt.ylabel('Year', fontsize=12)

    plt.subplot(3, 1, 3)
    sns.heatmap(pivot_df['Qsim'], cmap='YlOrBr', 
        xticklabels=month_names, annot=False, fmt=".1f", cbar_kws={'label': 'Streamflow (mm/day)'})
    plt.title('Simulated Streamflow - Monthly Values by Year', fontsize=14)
    plt.ylabel('Year', fontsize=12)
    plt.xlabel('Month', fontsize=12)

    plt.tight_layout()
    plt.show()
    
    # Calculate the relative contribution by month
    monthly_contribs = contribution_df.groupby('month').agg({
        'snow_melt': 'sum',
        'glacier_melt': 'sum',
        'Qsim': 'sum'
    }).reset_index()

    # Calculate other water sources (rainfall, groundwater, etc.)
    monthly_contribs['other_sources'] = monthly_contribs['Qsim'] - (monthly_contribs['snow_melt'] + monthly_contribs['glacier_melt'])
    monthly_contribs['other_sources'] = monthly_contribs['other_sources'].clip(lower=0)  # Ensure no negative values

    # Calculate percentages
    monthly_contribs['snow_pct'] = monthly_contribs['snow_melt'] / monthly_contribs['Qsim'] * 100
    monthly_contribs['glacier_pct'] = monthly_contribs['glacier_melt'] / monthly_contribs['Qsim'] * 100
    monthly_contribs['other_pct'] = monthly_contribs['other_sources'] / monthly_contribs['Qsim'] * 100

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Plot stacked percentage area
    ax.stackplot(monthly_contribs['month'], 
                monthly_contribs['other_pct'],
                monthly_contribs['snow_pct'], 
                monthly_contribs['glacier_pct'],
                labels=['Other Sources', 'Snow Melt', 'Glacier Melt'],
                colors=['#F8C471', '#AED6F1', '#75D054'], alpha=0.8)

    # Format plot
    ax.set_xticks(monthly_contribs['month'])
    ax.set_xticklabels(month_names)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Contribution to Streamflow (%)', fontsize=12)
    ax.set_title('Monthly Relative Contribution to Streamflow', fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12)

    plt.tight_layout()
    plt.show()
    
    # Create dataframe for analysis
    scatter_df = contribution_df.copy()

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Snow melt vs. Streamflow scatter
    ax1.scatter(scatter_df['snow_melt'], scatter_df['Qsim'], 
                alpha=0.5, color='#AED6F1', edgecolor='none')
    # Add regression line
    z1 = np.polyfit(scatter_df['snow_melt'], scatter_df['Qsim'], 1)
    p1 = np.poly1d(z1)
    x_range = np.linspace(0, scatter_df['snow_melt'].max(), 100)
    ax1.plot(x_range, p1(x_range), color='#3498DB', linewidth=2)
    r1 = np.corrcoef(scatter_df['snow_melt'], scatter_df['Qsim'])[0,1]
    ax1.text(0.05, 0.95, f'r = {r1:.2f}', transform=ax1.transAxes, fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.8))
    ax1.set_xlabel('Snow Melt (mm/day)', fontsize=12)
    ax1.set_ylabel('Streamflow (mm/day)', fontsize=12)
    ax1.set_title('Snow Melt vs. Streamflow', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Glacier melt vs. Streamflow scatter
    ax2.scatter(scatter_df['glacier_melt'], scatter_df['Qsim'], 
                alpha=0.5, color='#75D054', edgecolor='none')
    # Add regression line
    z2 = np.polyfit(scatter_df['glacier_melt'], scatter_df['Qsim'], 1)
    p2 = np.poly1d(z2)
    x_range = np.linspace(0, scatter_df['glacier_melt'].max(), 100)
    ax2.plot(x_range, p2(x_range), color='#2ECC71', linewidth=2)
    r2 = np.corrcoef(scatter_df['glacier_melt'], scatter_df['Qsim'])[0,1]
    ax2.text(0.05, 0.95, f'r = {r2:.2f}', transform=ax2.transAxes, fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.8))
    ax2.set_xlabel('Glacier Melt (mm/day)', fontsize=12)
    ax2.set_ylabel('Streamflow (mm/day)', fontsize=12)
    ax2.set_title('Glacier Melt vs. Streamflow', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    # Create a year-over-year comparison
    yearly_df = contribution_df.copy()
    yearly_df['year'] = yearly_df['date'].dt.year

    # Calculate annual statistics
    annual_stats = yearly_df.groupby('year').agg({
        'snow_melt': 'sum',
        'glacier_melt': 'sum',
        'Qobs': 'sum',
        'Qsim': 'sum'
    }).reset_index()

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Set bar width and positions
    bar_width = 0.8
    years = annual_stats['year']
    indices = np.arange(len(years))

    # Create stacked bars
    p1 = ax.bar(indices, annual_stats['snow_melt'], bar_width, 
                color='#AED6F1', label='Snow Melt')
    p2 = ax.bar(indices, annual_stats['glacier_melt'], bar_width, 
                bottom=annual_stats['snow_melt'], color='#75D054', label='Glacier Melt')

    # Calculate other contributions
    annual_stats['other'] = annual_stats['Qsim'] - (annual_stats['snow_melt'] + annual_stats['glacier_melt'])
    annual_stats['other'] = annual_stats['other'].clip(lower=0)  # Ensure no negative values

    # Add other contributions to the stacked bar
    p3 = ax.bar(indices, annual_stats['other'], bar_width, 
                bottom=annual_stats['snow_melt'] + annual_stats['glacier_melt'], 
                color='#F8C471', label='Other Sources')

    # Add observed streamflow line
    ax.plot(indices, annual_stats['Qobs'], 'ko-', label='Observed Streamflow', linewidth=2)

    # Format plot
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Annual Water Volume (mm)', fontsize=12)
    ax.set_title('Annual Contribution of Different Water Sources', fontsize=14)
    ax.set_xticks(indices)
    ax.set_xticklabels(years, rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()
    
    # Analyze lag between peak melt and peak streamflow
    lag_df = contribution_df.copy()
    lag_df['year'] = lag_df['date'].dt.year
    lag_df['day_of_year'] = lag_df['date'].dt.dayofyear

    # Calculate rolling means to smooth data
    window_size = 7  # 7-day window
    lag_df['snow_melt_smooth'] = lag_df['snow_melt'].rolling(window=window_size, center=True).mean()
    lag_df['glacier_melt_smooth'] = lag_df['glacier_melt'].rolling(window=window_size, center=True).mean()
    lag_df['Qsim_smooth'] = lag_df['Qsim'].rolling(window=window_size, center=True).mean()

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Group by day of year and calculate mean across years
    daily_means = lag_df.groupby('day_of_year').agg({
        'snow_melt_smooth': 'mean',
        'glacier_melt_smooth': 'mean',
        'Qsim_smooth': 'mean'
    }).reset_index()

    # Plot smoothed time series
    ax.plot(daily_means['day_of_year'], daily_means['Qsim_smooth'], 'k-', 
            linewidth=2.5, label='Streamflow')
    ax.plot(daily_means['day_of_year'], daily_means['snow_melt_smooth'], 
            color='#AED6F1', linewidth=2, label='Snow Melt')
    ax.plot(daily_means['day_of_year'], daily_means['glacier_melt_smooth'], 
            color='#75D054', linewidth=2, label='Glacier Melt')

    # Find and mark peak days
    peak_snow = daily_means.loc[daily_means['snow_melt_smooth'].idxmax()]
    peak_glacier = daily_means.loc[daily_means['glacier_melt_smooth'].idxmax()]
    peak_flow = daily_means.loc[daily_means['Qsim_smooth'].idxmax()]

    # Mark peaks with vertical lines
    ax.axvline(x=peak_snow['day_of_year'], color='#AED6F1', linestyle='--', alpha=0.7)
    ax.axvline(x=peak_glacier['day_of_year'], color='#75D054', linestyle='--', alpha=0.7)
    ax.axvline(x=peak_flow['day_of_year'], color='black', linestyle='--', alpha=0.7)

    # Add annotations for lag times
    ax.annotate(f"Snow Peak: Day {peak_snow['day_of_year']:.0f}", 
                xy=(peak_snow['day_of_year'], peak_snow['snow_melt_smooth']),
                xytext=(peak_snow['day_of_year']+10, peak_snow['snow_melt_smooth']),
                arrowprops=dict(arrowstyle="->", color='#3498DB'),
                fontsize=10, color='#3498DB')

    ax.annotate(f"Glacier Peak: Day {peak_glacier['day_of_year']:.0f}", 
                xy=(peak_glacier['day_of_year'], peak_glacier['glacier_melt_smooth']),
                xytext=(peak_glacier['day_of_year']+10, peak_glacier['glacier_melt_smooth']),
                arrowprops=dict(arrowstyle="->", color='#2ECC71'),
                fontsize=10, color='#2ECC71')

    ax.annotate(f"Flow Peak: Day {peak_flow['day_of_year']:.0f}", 
                xy=(peak_flow['day_of_year'], peak_flow['Qsim_smooth']),
                xytext=(peak_flow['day_of_year']+10, peak_flow['Qsim_smooth']),
                arrowprops=dict(arrowstyle="->", color='black'),
                fontsize=10, color='black')

    # Calculate and display lag times
    snow_lag = peak_flow['day_of_year'] - peak_snow['day_of_year']
    glacier_lag = peak_flow['day_of_year'] - peak_glacier['day_of_year']

    lag_text = (f"Snow to Flow Lag: {snow_lag:.0f} days\n"
                f"Glacier to Flow Lag: {glacier_lag:.0f} days")
    ax.text(0.02, 0.95, lag_text, transform=ax.transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')

    # Format plot
    ax.set_xlabel('Day of Year', fontsize=12)
    ax.set_ylabel('Water Flux (mm/day)', fontsize=12)
    ax.set_title('Seasonal Timing of Melt Components and Streamflow', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_xlim(1, 366)

    plt.tight_layout()
    plt.show()

#--------------------------------------------------------------------------------

def merge_hydrological_dataframes(streamflow_df, snow_melt_df, glogem_df):
    """
    Merge three hydrological dataframes into two different formats.
    
    Parameters:
    -----------
    streamflow_df : DataFrame
        DataFrame with columns 'Date', 'Qobs', 'Qsim'
    snow_melt_df : DataFrame
        DataFrame with columns 'date', 'snow_melt', 'month', 'year'
    glogem_df : DataFrame
        DataFrame with columns 'date', 'glacier_melt', 'snowmelt', 'total_output', etc.
        
    Returns:
    --------
    tuple
        (merged_df, combined_snowmelt_df) - Two merged dataframes
    """
    # Make copies to avoid modifying original dataframes
    streamflow = streamflow_df.copy()
    snow_melt = snow_melt_df.copy()
    glogem = glogem_df.copy()
    
    # Ensure consistent date column names
    streamflow.rename(columns={'Date': 'date'}, inplace=True)
    
    # Ensure date columns are datetime objects
    streamflow['date'] = pd.to_datetime(streamflow['date'])
    snow_melt['date'] = pd.to_datetime(snow_melt['date'])
    glogem['date'] = pd.to_datetime(glogem['date'])
    
    # 1. First merged dataframe - keep all original columns
    # Rename glogem's snowmelt to avoid conflict
    glogem.rename(columns={'snowmelt': 'glogem_snowmelt'}, inplace=True)
    
    # Merge all dataframes
    merged_df = pd.merge(streamflow, snow_melt[['date', 'snow_melt']], on='date', how='outer')
    merged_df = pd.merge(merged_df, glogem, on='date', how='outer')
    
    # Ensure month and year columns exist and convert to integers
    merged_df['month'] = merged_df['date'].dt.month.astype(int)
    merged_df['year'] = merged_df['date'].dt.year.astype(int)
    
    # Sort by date
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    
    # 2. Second dataframe with combined snowmelt
    # Start with basic columns
    combined_snowmelt_df = pd.DataFrame({
        'date': merged_df['date'],
        'Qobs': merged_df['Qobs'],
        'Qsim': merged_df['Qsim'],
        'glacier_melt': merged_df['glacier_melt'],
        'month': merged_df['month'],
        'year': merged_df['year']
    })
    
    # Combine the two snowmelt columns
    # Fill NaNs with 0 before adding
    snow_melt_val = merged_df['snow_melt'].fillna(0)
    glogem_snowmelt_val = merged_df['glogem_snowmelt'].fillna(0)
    combined_snowmelt_df['snow_melt'] = snow_melt_val + glogem_snowmelt_val
    
    # Reorder columns to match requested format
    combined_snowmelt_df = combined_snowmelt_df[['date', 'snow_melt', 'glacier_melt', 'Qobs', 'Qsim', 'month', 'year']]
    
    # Fill NaNs with 0 for better usability
    combined_snowmelt_df = combined_snowmelt_df.fillna(0)
    
    # Ensure month and year are integers
    combined_snowmelt_df['month'] = combined_snowmelt_df['month'].astype(int)
    combined_snowmelt_df['year'] = combined_snowmelt_df['year'].astype(int)
    
    return merged_df, combined_snowmelt_df


#--------------------------------------------------------------------------------

def contributions_c(gauge_id, start_date, end_date, base_dir, unit='mm', catchment_area=None):
    """
    Analyze contributions to streamflow for a specific gauge and time period.
    
    Parameters:
    -----------
    gauge_id : str
        Identifier for the gauge
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    base_dir : str
        Base directory path
    unit : str, optional
        Unit for the output ('mm' or 'm3/s', default: 'mm')
    catchment_area : float, optional
        Catchment area in km², required if unit is 'm3/s'
        
    Returns:
    --------
    DataFrame
        DataFrame with date and contributions values
    """

    # define paths
    topo_dir = Path(base_dir) / f"catchment_{gauge_id}_c" / "topo_files"
    output_dir = Path(base_dir) / "HBV" / "output"
    
    # define files
    catchment_shape_file = topo_dir / "HRU.shp"
    streamflow_file = output_dir / f"{gauge_id}_HBV_Hydrographs.csv"
    
    # Calculate catchment area from shapefile
    if not os.path.exists(catchment_shape_file):
        raise FileNotFoundError(f"Shapefile not found: {catchment_shape_file}")
        
    gdf = gpd.read_file(catchment_shape_file)
    # Area is typically in the units of the CRS - convert to km²
    if 'Area_km2' in gdf.columns:
            catchment_area_km2 = gdf['Area_km2'].sum()
    else:
        # Calculate area if not available directly
        catchment_area_km2 = gdf.to_crs({'proj': 'cea'}).area.sum() / 1e6
     
    # calculate snowmelt for whole catchment    
    snow_melt = analyze_snow_melt(gauge_id, start_date, end_date, base_dir, coupled=True, 
                                 unit=unit, catchment_area=catchment_area_km2)
    
    # calculate glogem melt for the whole catchment
    glogem = process_glogem_data_optimized('2268', base_dir, '2000-01-01', '2020-12-31', unit='mm')
    
    #  Calculate streamflow for whole catchment
    streamflow = get_hydrograph_data(gauge_id, coupled=True, start_date=start_date, end_date=end_date,
                                   unit=unit, base_dir=base_dir, catchment_area=catchment_area_km2)
    
    merged_df, combined_snowmelt_df = merge_hydrological_dataframes(streamflow, snow_melt, glogem)
    
    # Calculate monthly averages
    monthly_avgs = combined_snowmelt_df.groupby('month').agg({
        'snow_melt': 'mean',
        'glacier_melt': 'mean',
        'Qobs': 'mean',
        'Qsim': 'mean'
    }).reset_index()

    # Define month names for better x-axis labels
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Set the y-axis label based on unit
    if unit.lower() == 'mm':
        y_label = 'Flow (mm/day)'
    elif unit.lower() == 'm3/s':
        y_label = 'Flow (m³/s)'
    else:
        y_label = f'Flow ({unit})'

    # Create plot
    plt.figure(figsize=(12, 7))

    # Plot individual melt components as filled areas with transparency
    # Plot glacier melt
    plt.plot(monthly_avgs['month'], monthly_avgs['glacier_melt'], 
            color='#75D054', linewidth=2, label='Glacier Melt')
    plt.fill_between(monthly_avgs['month'], 0, monthly_avgs['glacier_melt'], 
                    color='#75D054', alpha=0.4)

    # Plot snow melt
    plt.plot(monthly_avgs['month'], monthly_avgs['snow_melt'], 
            color='#AED6F1', linewidth=2, label='Snow Melt')
    plt.fill_between(monthly_avgs['month'], 0, monthly_avgs['snow_melt'], 
                    color='#AED6F1', alpha=0.4)

    # Plot observed and simulated streamflow as lines
    plt.plot(monthly_avgs['month'], monthly_avgs['Qobs'], 'k-', linewidth=2.5, 
            label='Observed Streamflow')
    plt.plot(monthly_avgs['month'], monthly_avgs['Qsim'], color='#24868E', linewidth=2, 
            linestyle='--', label='Simulated Streamflow')

    # Configure plot
    plt.xticks(monthly_avgs['month'], month_names)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel(y_label, fontsize=12)  # Dynamic y-axis label
    plt.title('Monthly Hydrological Regime with Melt Components', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.show()
    
    return merged_df, combined_snowmelt_df


#--------------------------------------------------------------------------------

def analyze_contributions(gauge_id, start_date, end_date, base_dir, config=None, unit='mm', catchment_area=None, plot=True):
    """
    Analyze contributions to streamflow for a specific gauge and time period.
    Automatically handles both coupled and non-coupled configurations.
    
    Parameters:
    -----------
    gauge_id : str
        Identifier for the gauge
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    base_dir : str
        Base directory path
    config : str or list, default=None
        Configuration to process. Can specify the exact configuration (e.g. 'c_single', 'nc_multi').
        If None, will try to find an appropriate configuration automatically.
    unit : str, optional
        Unit for the output ('mm' or 'm3/s', default: 'mm')
    catchment_area : float, optional
        Catchment area in km², required if unit is 'm3/s'
    plot : bool, optional
        Whether to create and display plots (default: True)
        
    Returns:
    --------
    tuple or DataFrame
        For coupled configs: (merged_df, combined_snowmelt_df)
        For non-coupled configs: merged_df containing contributions data
    """
    # If config is not provided, try to determine it
    if config is None:
        # Check if we have a coupled configuration
        c_dir = Path(base_dir) / f"catchment_{gauge_id}_c"
        c_single_dir = Path(base_dir) / f"catchment_{gauge_id}_c_single"
        c_multi_dir = Path(base_dir) / f"catchment_{gauge_id}_c_multi"
        
        # Check if we have a non-coupled configuration
        nc_dir = Path(base_dir) / f"catchment_{gauge_id}_nc"
        nc_single_dir = Path(base_dir) / f"catchment_{gauge_id}_nc_single"
        nc_multi_dir = Path(base_dir) / f"catchment_{gauge_id}_nc_multi"
        
        # Check which directories exist
        if c_single_dir.exists():
            config = "c_single"
        elif c_multi_dir.exists():
            config = "c_multi"
        elif c_dir.exists():
            config = "c"
        elif nc_single_dir.exists():
            config = "nc_single"
        elif nc_multi_dir.exists():
            config = "nc_multi"
        elif nc_dir.exists():
            config = "nc"
        else:
            raise FileNotFoundError(f"Could not find any valid configuration directories for gauge {gauge_id}")
    
    # Determine if this is a coupled or non-coupled configuration
    is_coupled = any(config.startswith(c) for c in ["c", "c_", "couple"])
    
    print(f"Processing gauge {gauge_id} with {'coupled' if is_coupled else 'non-coupled'} configuration: {config}")
    
    # Define paths
    topo_dir = Path(base_dir) / f"catchment_{gauge_id}_{config}" / "topo_files"
    
    # Calculate catchment area from shapefile
    catchment_shape_file = topo_dir / "HRU.shp"
    if not os.path.exists(catchment_shape_file):
        raise FileNotFoundError(f"Shapefile not found: {catchment_shape_file}")
        
    gdf = gpd.read_file(catchment_shape_file)
    # Get catchment area
    if 'Area_km2' in gdf.columns:
        catchment_area_km2 = gdf['Area_km2'].sum()
    else:
        # Calculate area if not available directly
        catchment_area_km2 = gdf.to_crs({'proj': 'cea'}).area.sum() / 1e6
        
    print(f"Catchment area: {catchment_area_km2:.2f} km²")
    
    # For coupled configurations
    if is_coupled:
        print(f"Using coupled workflow for configuration: {config}")
        # Calculate snowmelt 
        snow_melt_df = analyze_snow_melt(
            gauge_id, start_date, end_date, base_dir, 
            config=config, unit=unit, catchment_area=catchment_area_km2, plot=False
        )
        
        if not isinstance(snow_melt_df, dict) or not snow_melt_df:
            print(f"Warning: No snow melt data found for {gauge_id}")
            snow_melt_data = None
        else:
            # Extract the relevant dataframe from the dictionary
            snow_melt_data = list(snow_melt_df.values())[0]
        
        # Get GloGEM data for glacier melt
        try:
            glogem_data = process_glogem_data_optimized(
                gauge_id, base_dir, start_date, end_date, unit=unit, plot=False
            )
        except Exception as e:
            print(f"Error loading GloGEM data: {e}")
            glogem_data = None
            
        # Calculate streamflow
        streamflow_data = get_hydrograph_data(
            gauge_id, config=config, start_date=start_date, end_date=end_date,
            unit=unit, base_dir=base_dir, catchment_area=catchment_area_km2, plot=False
        )
        
        if not isinstance(streamflow_data, dict) or not streamflow_data:
            print(f"Warning: No streamflow data found for {gauge_id}")
            streamflow = None
        else:
            # Extract the relevant dataframe from the dictionary
            streamflow = list(streamflow_data.values())[0]
        
        # Check if we have all the required data
        if snow_melt_data is None or glogem_data is None or streamflow is None:
            print("Missing required data for coupled analysis")
            return None
        
        # Merge the dataframes
        merged_df, combined_snowmelt_df = merge_hydrological_dataframes(
            streamflow, snow_melt_data, glogem_data
        )
        
        if plot:
            # Calculate monthly averages
            monthly_avgs = combined_snowmelt_df.groupby('month').agg({
                'snow_melt': 'mean',
                'glacier_melt': 'mean',
                'Qobs': 'mean',
                'Qsim': 'mean'
            }).reset_index()
            
            # Define month names for better x-axis labels
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Set the y-axis label based on unit
            y_label = 'Flow (mm/day)' if unit.lower() == 'mm' else 'Flow (m³/s)'
            
            # Create plot
            plt.figure(figsize=(12, 7))
            
            # Plot individual melt components as filled areas with transparency
            # Plot glacier melt
            plt.plot(monthly_avgs['month'], monthly_avgs['glacier_melt'], 
                    color='#75D054', linewidth=2, label='Glacier Melt')
            plt.fill_between(monthly_avgs['month'], 0, monthly_avgs['glacier_melt'], 
                            color='#75D054', alpha=0.4)
            
            # Plot snow melt
            plt.plot(monthly_avgs['month'], monthly_avgs['snow_melt'], 
                    color='#AED6F1', linewidth=2, label='Snow Melt')
            plt.fill_between(monthly_avgs['month'], 0, monthly_avgs['snow_melt'], 
                            color='#AED6F1', alpha=0.4)
            
            # Plot observed and simulated streamflow as lines
            plt.plot(monthly_avgs['month'], monthly_avgs['Qobs'], 'k-', linewidth=2.5, 
                    label='Observed Streamflow')
            plt.plot(monthly_avgs['month'], monthly_avgs['Qsim'], color='#24868E', linewidth=2, 
                    linestyle='--', label='Simulated Streamflow')
            
            # Configure plot
            plt.xticks(monthly_avgs['month'], month_names)
            plt.xlabel('Month', fontsize=12)
            plt.ylabel(y_label, fontsize=12)  # Dynamic y-axis label
            plt.title(f'Monthly Hydrological Regime with Melt Components - {gauge_id} (Coupled)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left')
            plt.tight_layout()
            plt.show()
            
            # Generate additional plots if needed
            contribution_plots(combined_snowmelt_df)
            
        return merged_df, combined_snowmelt_df
    
    # For non-coupled configurations
    else:
        print(f"Using non-coupled workflow for configuration: {config}")
        # Calculate snowmelt 
        snow_melt_df = analyze_snow_melt(
            gauge_id, start_date, end_date, base_dir, 
            config=config, unit=unit, catchment_area=catchment_area_km2, plot=False
        )
        
        if not isinstance(snow_melt_df, dict) or not snow_melt_df:
            print(f"Warning: No snow melt data found for {gauge_id}")
            snow_melt = None
        else:
            # Extract the relevant dataframe from the dictionary
            snow_melt = list(snow_melt_df.values())[0]
        
        # Calculate glacier melt
        glacier_melt_df = analyze_glacier_melt(
            gauge_id, start_date, end_date, base_dir, 
            config=config, unit=unit, catchment_area=catchment_area_km2, plot=False
        )
        
        if not isinstance(glacier_melt_df, dict) or not glacier_melt_df:
            print(f"Warning: No glacier melt data found for {gauge_id}")
            glacier_melt = None
        else:
            # Extract the relevant dataframe from the dictionary
            glacier_melt = list(glacier_melt_df.values())[0]
        
        # Calculate streamflow
        streamflow_data = get_hydrograph_data(
            gauge_id, config=config, start_date=start_date, end_date=end_date,
            unit=unit, base_dir=base_dir, catchment_area=catchment_area_km2, plot=False
        )
        
        if not isinstance(streamflow_data, dict) or not streamflow_data:
            print(f"Warning: No streamflow data found for {gauge_id}")
            streamflow = None
        else:
            # Extract the relevant dataframe from the dictionary
            streamflow = list(streamflow_data.values())[0]
        
        # Check if we have all the required data
        if snow_melt is None or glacier_melt is None or streamflow is None:
            print("Missing required data for non-coupled analysis")
            return None
        
        # Ensure all dataframes have datetime index for proper merging
        snow_melt['date'] = pd.to_datetime(snow_melt['date'])
        glacier_melt['date'] = pd.to_datetime(glacier_melt['date'])
        
        # Handle the streamflow dataframe more carefully
        # First check the columns to identify which one is the date column
        print(f"Original streamflow columns: {streamflow.columns.tolist()}")
        
        # Create a clean streamflow dataframe with only the columns we need
        streamflow_cleaned = pd.DataFrame()
        
        # Add date column (could be 'Date' or 'date')
        if 'Date' in streamflow.columns:
            streamflow_cleaned['date'] = pd.to_datetime(streamflow['Date'])
        elif 'date' in streamflow.columns:
            # Get the first 'date' column if there are duplicates
            date_col_idx = streamflow.columns.get_loc('date')
            if isinstance(date_col_idx, (list, np.ndarray)):
                # Multiple columns found, take the first one
                date_col_idx = date_col_idx[0]
            streamflow_cleaned['date'] = pd.to_datetime(streamflow.iloc[:, date_col_idx])
        
        # Add other needed columns
        if 'Qobs' in streamflow.columns:
            streamflow_cleaned['Qobs'] = streamflow['Qobs']
        if 'Qsim' in streamflow.columns:
            streamflow_cleaned['Qsim'] = streamflow['Qsim']
        
        # Use streamflow_cleaned instead of the original dataframe
        streamflow = streamflow_cleaned
        print(f"Cleaned streamflow columns: {streamflow.columns.tolist()}")
        
        # Print column names for debugging
        print(f"Snow melt columns: {snow_melt.columns.tolist()}")
        print(f"Glacier melt columns: {glacier_melt.columns.tolist()}")
        print(f"Streamflow columns: {streamflow.columns.tolist()}")

        # Merge dataframes with clear suffixes
        merged_df = pd.merge(snow_melt, glacier_melt, on='date', suffixes=('_snow', '_glacier'))
        merged_df = pd.merge(merged_df, streamflow, on='date')
        
        # Keep only necessary columns
        merged_df = merged_df[['date', 'snow_melt', 'glacier_melt', 'Qobs', 'Qsim', 
                              'month_snow', 'year_snow']]  # Using snow month/year as they should be the same
        
        # Rename for clarity
        merged_df = merged_df.rename(columns={'month_snow': 'month', 'year_snow': 'year'})
        
        # Handle any NaN values
        merged_df = merged_df.fillna(0)
        
        if plot:
            # Calculate monthly averages
            monthly_avgs = merged_df.groupby('month').agg({
                'snow_melt': 'mean',
                'glacier_melt': 'mean',
                'Qobs': 'mean',
                'Qsim': 'mean'
            }).reset_index()
            
            # Define month names for better x-axis labels
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Set the y-axis label based on unit
            y_label = 'Flow (mm/day)' if unit.lower() == 'mm' else 'Flow (m³/s)'
            
            # Create plot
            plt.figure(figsize=(12, 7))
            
            # Plot individual melt components as filled areas with transparency
            # Plot glacier melt
            plt.plot(monthly_avgs['month'], monthly_avgs['glacier_melt'], 
                    color='#75D054', linewidth=2, label='Glacier Melt')
            plt.fill_between(monthly_avgs['month'], 0, monthly_avgs['glacier_melt'], 
                            color='#75D054', alpha=0.4)
            
            # Plot snow melt
            plt.plot(monthly_avgs['month'], monthly_avgs['snow_melt'], 
                    color='#AED6F1', linewidth=2, label='Snow Melt')
            plt.fill_between(monthly_avgs['month'], 0, monthly_avgs['snow_melt'], 
                            color='#AED6F1', alpha=0.4)
            
            # Plot observed and simulated streamflow as lines
            plt.plot(monthly_avgs['month'], monthly_avgs['Qobs'], 'k-', linewidth=2.5, 
                    label='Observed Streamflow')
            plt.plot(monthly_avgs['month'], monthly_avgs['Qsim'], color='#24868E', linewidth=2, 
                    linestyle='--', label='Simulated Streamflow')
            
            # Configure plot
            plt.xticks(monthly_avgs['month'], month_names)
            plt.xlabel('Month', fontsize=12)
            plt.ylabel(y_label, fontsize=12)  # Dynamic y-axis label
            plt.title(f'Monthly Hydrological Regime with Melt Components - {gauge_id} (Non-Coupled)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left')
            plt.tight_layout()
            plt.show()
            
            # Generate additional plots
            contribution_plots(merged_df)
        
        return merged_df


#--------------------------------------------------------------------------------

def analyze_contributions_multi(gauge_id, start_date, end_date, base_dir, configs=None, unit='mm', 
                              catchment_area=None, plot=True, figsize=None, config_colors=None, config_names=None):
    """
    Analyze and compare contributions to streamflow across multiple configurations.
    Creates a subplot for each configuration organized in 2 rows.
    
    Parameters:
    -----------
    gauge_id : str
        Identifier for the gauge
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    base_dir : str
        Base directory path
    configs : list, default=None
        List of configurations to analyze. If None, will try to find available configurations.
    unit : str, optional
        Unit for the output ('mm' or 'm3/s', default: 'mm')
    catchment_area : float, optional
        Catchment area in km², required if unit is 'm3/s'
    plot : bool, optional
        Whether to create and display plots (default: True)
    figsize : tuple, optional
        Figure size as (width, height). If None, calculates based on number of configs.
    config_colors : dict, optional
        Dict mapping config names to colors (e.g., {'nc': '#712423'})
    config_names : dict, optional
        Dict mapping config names to friendly display names (e.g., {'nc': 'HBV'})
        
    Returns:
    --------
    dict
        Dictionary with configuration names as keys and contribution data as values
    dict
        Dictionary with configuration names as keys and monthly contribution statistics
    """
    from pathlib import Path
    
    # Handle config parameter
    if configs is None:
        # Try to find available configurations
        print("No configurations provided. Searching for available configurations...")
        available_configs = []
        
        # Check for possible configurations
        patterns = ['c', 'nc', 'c_single', 'c_multi', 'nc_single', 'nc_multi', 'c_single_sb', 'c_sr', 'nc_sr']
        
        for pattern in patterns:
            config_dir = os.path.join(base_dir, f"catchment_{gauge_id}_{pattern}")
            if os.path.exists(config_dir):
                available_configs.append(pattern)
        
        if not available_configs:
            raise ValueError(f"No configurations found for gauge {gauge_id}")
        
        configs = available_configs
        print(f"Found configurations: {configs}")
    
    elif isinstance(configs, str):
        configs = [configs]  # Convert single string to list
    
    
    # Store results for each configuration
    all_results = {}
    
    # Create a dictionary to store monthly contribution statistics for each configuration
    monthly_contributions = {}
    
    # Process each configuration
    for config in configs:
        print(f"\nAnalyzing contributions for gauge {gauge_id}, configuration {config}...")
        
        try:
            # Use the analyze_contributions function to process each config
            result = analyze_contributions(
                gauge_id, start_date, end_date, base_dir, 
                config=config, unit=unit, catchment_area=catchment_area, 
                plot=False  # Don't create individual plots yet
            )
            
            # Store the result
            if result is not None:
                all_results[config] = result
                
                # Calculate monthly contribution statistics
                # Get the appropriate dataframe
                if isinstance(result, tuple):  # Coupled configuration returns a tuple
                    df = result[1]  # combined_snowmelt_df
                else:
                    df = result  # Non-coupled configuration returns a dataframe
                
                # Calculate monthly averages
                monthly_avgs = df.groupby('month').agg({
                    'snow_melt': 'mean',
                    'glacier_melt': 'mean',
                    'Qobs': 'mean',
                    'Qsim': 'mean'
                }).reset_index()
                
                # Calculate contribution percentages
                monthly_avgs['snow_contribution'] = monthly_avgs['snow_melt'] / monthly_avgs['Qsim'] * 100
                monthly_avgs['glacier_contribution'] = monthly_avgs['glacier_melt'] / monthly_avgs['Qsim'] * 100
                
                # Handle cases where Qsim is very small or zero (avoid division by zero)
                monthly_avgs['snow_contribution'] = monthly_avgs['snow_contribution'].fillna(0).clip(0, 100)
                monthly_avgs['glacier_contribution'] = monthly_avgs['glacier_contribution'].fillna(0).clip(0, 100)
                
                # Calculate other contributions (rainfall, groundwater, etc.)
                monthly_avgs['other_contribution'] = 100 - (monthly_avgs['snow_contribution'] + monthly_avgs['glacier_contribution'])
                monthly_avgs['other_contribution'] = monthly_avgs['other_contribution'].clip(0, 100)  # Ensure between 0-100%
                
                # Store in the monthly contributions dictionary
                monthly_contributions[config] = monthly_avgs
                
        except Exception as e:
            print(f"Error processing configuration {config}: {str(e)}")
            continue
        
    # If plot is requested, create a comparison plot with subplots
    if plot and all_results:
        # Calculate subplot layout
        n_configs = len(all_results)
        n_cols = min(2, n_configs)  # Maximum 2 columns
        n_rows = (n_configs + n_cols - 1) // n_cols  # Ceiling division
        
        # Set figure size
        if figsize is None:
            figsize = (6 * n_cols, 5 * n_rows)
        
        # Replace the plotting section in analyze_contributions_multi with this:

        # Create the subplot figure for components
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        fig.suptitle(f'Monthly Hydrological Regime Comparison - Gauge {gauge_id}', fontsize=16)

        # Plot each configuration
        for i, config in enumerate(configs):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Get monthly average data
            monthly_avgs = monthly_contributions[config]
            
            # Get colors and name
            color = config_colors.get(config, 'gray') if config_colors else 'gray'
            name = config_names.get(config, config) if config_names else config
            
            # ✅ Plot only snow melt and glacier melt filled areas
            # Snow melt - independent filled area
            ax.fill_between(monthly_avgs['month'], 0, monthly_avgs['snow_melt'], 
                        alpha=0.6, color='lightblue', label='Snow Melt')
            
            # Glacier melt - independent filled area (NOT stacked on snow melt)
            ax.fill_between(monthly_avgs['month'], 0, monthly_avgs['glacier_melt'], 
                        alpha=0.6, color='lightgrey', label='Glacier Melt')
            
            # ✅ Removed the "Other Sources" filled area
            
            # Plot total simulated flow as a line
            ax.plot(monthly_avgs['month'], monthly_avgs['Qsim'], 
                color=color, linewidth=2.5, label=f'{name} Total', alpha=0.9)
            
            # Plot observed flow as a line
            ax.plot(monthly_avgs['month'], monthly_avgs['Qobs'], 
                'k--', linewidth=2, label='Observed', alpha=0.9)
            
            # Configure subplot
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticks(monthly_avgs['month'])
            ax.set_xticklabels(month_names)
            ax.set_xlabel('Month')
            ax.set_ylabel(f'Flow ({unit})')
            ax.set_title(f'{name}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')

        # Hide empty subplots if any
        for i in range(n_configs, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()
        
        # Create percentage contributions plot
        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)  # ✅ squeeze=False here too
        fig2.suptitle(f'Relative Contributions to Streamflow - Gauge {gauge_id}', fontsize=16)
        
        for i, config in enumerate(configs):
            row = i // n_cols
            col = i % n_cols
            ax = axes2[row, col]  # ✅ Now this will work correctly
            
            # Get monthly average data
            monthly_avgs = monthly_contributions[config]
            
            # Calculate percentages
            total_flow = monthly_avgs['Qsim']
            snow_pct = (monthly_avgs['snow_melt'] / total_flow * 100).fillna(0)
            glacier_pct = (monthly_avgs['glacier_melt'] / total_flow * 100).fillna(0)
            other_pct = 100 - snow_pct - glacier_pct
            
            # Create stacked bar chart
            months = monthly_avgs['month']
            ax.bar(months, snow_pct, label='Snow Melt', color='lightblue', alpha=0.8)
            ax.bar(months, glacier_pct, bottom=snow_pct, label='Glacier Melt', color='lightcoral', alpha=0.8)
            ax.bar(months, other_pct, bottom=snow_pct + glacier_pct, label='Other Sources', color='lightgreen', alpha=0.8)
            
            # Configure subplot
            name = config_names.get(config, config) if config_names else config
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticks(months)
            ax.set_xticklabels(month_names)
            ax.set_xlabel('Month')
            ax.set_ylabel('Contribution (%)')
            ax.set_title(f'{name}')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
        
        # Hide empty subplots if any
        for i in range(n_configs, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes2[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Print a summary of monthly contributions for each configuration
    print("\n===== Monthly Contribution Summary =====")
    for config, monthly_data in monthly_contributions.items():
        config_name = config_names.get(config, config)
        print(f"\nConfiguration: {config_name}")
        
        # Calculate seasonal statistics
        winter = monthly_data[monthly_data['month'].isin([12, 1, 2])].mean()
        spring = monthly_data[monthly_data['month'].isin([3, 4, 5])].mean()
        summer = monthly_data[monthly_data['month'].isin([6, 7, 8])].mean()
        fall = monthly_data[monthly_data['month'].isin([9, 10, 11])].mean()
        annual = monthly_data.mean()
        
        # Print seasonal contribution summary
        seasons = {'Annual': annual, 'Winter': winter, 'Spring': spring, 'Summer': summer, 'Fall': fall}
        
        print(f"{'Season':<8} {'Snow %':>8} {'Glacier %':>10} {'Other %':>10}")
        print("-" * 40)
        
        for season_name, stats in seasons.items():
            print(f"{season_name:<8} {stats['snow_contribution']:>8.1f} {stats['glacier_contribution']:>10.1f} {stats['other_contribution']:>10.1f}")
    
    return all_results, monthly_contributions


#--------------------------------------------------------------------------------
################################### storages ####################################
#--------------------------------------------------------------------------------

def load_storage_data(gauge_id, config, base_dir):
    """
    Load watershed storage data for a specific configuration.
    
    Parameters:
    -----------
    gauge_id : str
        ID of the gauge to analyze
    config : str
        Configuration name (e.g., "nc_single", "c_multi")
    base_dir : str or Path
        Base directory containing model outputs
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing storage data with datetime index
    """
    # Construct path to storage file
    model_dir = Path(base_dir) / f"catchment_{gauge_id}_{config}" / "HBV" / "output"
    storage_file = model_dir / f"{gauge_id}_HBV_WatershedStorage.csv"
    
    print(f"Loading storage data for {config}:")
    print(f"  - File: {storage_file}")
    
    if not storage_file.exists():
        print(f"ERROR: Storage file not found: {storage_file}")
        return None
    
    try:
        # Read the CSV file with the second row skipped
        df = pd.read_csv(storage_file,  skiprows = [1])

        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'])
        else:
            # Look for any column that might be a date
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    df['date'] = pd.to_datetime(df[col])
                    break
        
        # Filter out unwanted columns
        columns_to_exclude = [
            'time [d]',
            'hour', 
            'Channel Storage [mm]', 
            'Reservoir Storage [mm]', 
            'Surface Water [mm]', 
            'Canopy [mm]', 
            'Canopy Snow [mm]', 
            'Net Lake Storage [mm]'
        ]
        
        # Keep only the columns we want
        columns_to_keep = [col for col in df.columns if col not in columns_to_exclude]
        df = df[columns_to_keep]
        
        # Add month and year columns for analysis
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        print(f"  - Loaded {len(df)} records from {df['date'].min()} to {df['date'].max()}")
        print(f"  - Storage columns: {[col for col in df.columns if col not in ['date', 'month', 'year']]}")
        
        return df
        
    except Exception as e:
        print(f"  - Error loading storage data: {e}")
        return None


#--------------------------------------------------------------------------------

def plot_storage_timeseries(gauge_id, config, base_dir, start_date=None, end_date=None, 
                           config_colors=None, config_names=None):
    """
    Plot time series of watershed storage components for a single configuration.
    
    Parameters:
    -----------
    gauge_id : str
        ID of the gauge to analyze
    config : str
        Configuration name
    base_dir : str or Path
        Base directory containing model outputs
    start_date : str, optional
        Start date for filtering (format: 'YYYY-MM-DD')
    end_date : str, optional
        End date for filtering (format: 'YYYY-MM-DD')
    config_colors : dict, optional
        Dict mapping config names to colors
    config_names : dict, optional
        Dict mapping config names to friendly display names
    """
    # Load storage data
    storage_df = load_storage_data(gauge_id, config, base_dir)
    
    if storage_df is None:
        print(f"No storage data available for {gauge_id} - {config}")
        return
    
    # Filter by date range if provided
    if start_date:
        storage_df = storage_df[storage_df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        storage_df = storage_df[storage_df['date'] <= pd.to_datetime(end_date)]
    
    # Get storage columns (exclude date, month, year)
    storage_cols = [col for col in storage_df.columns if col not in ['date', 'month', 'year']]
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(storage_cols), 1, figsize=(14, 4*len(storage_cols)), sharex=True)
    
    if len(storage_cols) == 1:
        axes = [axes]  # Make it iterable
    
    # Get config color
    color = config_colors.get(config, '#2a5674') if config_colors else '#2a5674'
    config_name = config_names.get(config, config) if config_names else config
    
    # Plot each storage component
    for i, col in enumerate(storage_cols):
        ax = axes[i]
        
        # Handle precipitation data with better y-axis scaling
        if col in ['snowfall [mm/d]', 'rainfall [mm/d]']:
            data = storage_df[col]
            # Set y-axis limit to 95th percentile to avoid extreme outliers dominating the scale
            y_max = np.percentile(data[data > 0], 95) if len(data[data > 0]) > 0 else data.max()
            ax.set_ylim(0, y_max * 1.1)
            
            if col == 'snowfall [mm/d]':
                ax.fill_between(storage_df['date'], 0, data, color='skyblue', alpha=0.7, edgecolor='lightblue')
            else:  # rainfall
                ax.fill_between(storage_df['date'], 0, data, color='navy', alpha=0.7, edgecolor='darkblue')
        else:
            # For other variables, use the full range
            if 'snow storage [mm]' in col:
                ax.fill_between(storage_df['date'], 0, storage_df[col], color='white', alpha=0.8, edgecolor='lightgray')
            else:
                ax.plot(storage_df['date'], storage_df[col], color=color, linewidth=1.5)
        
        ax.set_title(f'{col} - {config_name}', fontsize=14)
        ax.set_ylabel(col, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    
    # Set x-label only on bottom subplot
    axes[-1].set_xlabel('Date', fontsize=12)
    
    # Add overall title
    period_str = ""
    if start_date and end_date:
        period_str = f" ({start_date} to {end_date})"
    elif start_date:
        period_str = f" (from {start_date})"
    elif end_date:
        period_str = f" (to {end_date})"
    
    fig.suptitle(f'Watershed Storage Components - Gauge {gauge_id}{period_str}', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save plot to catchment's plots_results folder
    save_dir = Path(base_dir) / f"catchment_{gauge_id}_{config}" / "plots_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"storage_timeseries_{gauge_id}.png", dpi=300, bbox_inches='tight')
    print(f"Saved storage plot to {save_dir / f'storage_timeseries_{gauge_id}.png'}")
    
    plt.show()


#--------------------------------------------------------------------------------

def plot_storage_seasonal_patterns(gauge_id, config, base_dir, config_colors=None, config_names=None):
    """
    Plot seasonal patterns of watershed storage components.
    
    Parameters:
    -----------
    gauge_id : str
        ID of the gauge to analyze
    config : str
        Configuration name
    base_dir : str or Path
        Base directory containing model outputs
    config_colors : dict, optional
        Dict mapping config names to colors
    config_names : dict, optional
        Dict mapping config names to friendly display names
    """
    # Load storage data
    storage_df = load_storage_data(gauge_id, config, base_dir)
    
    if storage_df is None:
        print(f"No storage data available for {gauge_id} - {config}")
        return
    
    # Get storage columns
    storage_cols = [col for col in storage_df.columns if col not in ['date', 'month', 'year']]
    
    # Calculate monthly averages
    monthly_avgs = storage_df.groupby('month')[storage_cols].mean().reset_index()
    
    # Month names for x-axis
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Get config color and name
    color = config_colors.get(config, '#2a5674') if config_colors else '#2a5674'
    config_name = config_names.get(config, config) if config_names else config
    
    # Create figure
    fig, axes = plt.subplots(len(storage_cols), 1, figsize=(12, 4*len(storage_cols)), sharex=True)
    
    if len(storage_cols) == 1:
        axes = [axes]
    
    # Plot each storage component
    for i, col in enumerate(storage_cols):
        ax = axes[i]
        ax.plot(monthly_avgs['month'], monthly_avgs[col], 
               color=color, linewidth=2, marker='o', markersize=6)
        ax.set_title(f'{col} - Monthly Average', fontsize=14)
        ax.set_ylabel('Storage (mm)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names)
    
    # Add overall title
    fig.suptitle(f'Seasonal Storage Patterns - Gauge {gauge_id} ({config_name})', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save plot
    save_dir = Path(base_dir) / f"catchment_{gauge_id}_{config}" / "plots_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"storage_seasonal_{gauge_id}.png", dpi=300, bbox_inches='tight')
    print(f"Saved seasonal storage plot to {save_dir / f'storage_seasonal_{gauge_id}.png'}")
    
    plt.show()

#--------------------------------------------------------------------------------

def analyze_storage_multi(gauge_id, configs, base_dir, start_date=None, end_date=None,
                         config_colors=None, config_names=None, plot=True):
    """
    Analyze and compare watershed storage across multiple configurations.
    
    Parameters:
    -----------
    gauge_id : str
        ID of the gauge to analyze
    configs : list
        List of configuration names to compare
    base_dir : str or Path
        Base directory containing model outputs
    start_date : str, optional
        Start date for filtering (format: 'YYYY-MM-DD')
    end_date : str, optional
        End date for filtering (format: 'YYYY-MM-DD')
    config_colors : dict, optional
        Dict mapping config names to colors
    config_names : dict, optional
        Dict mapping config names to friendly display names
    plot : bool, optional
        Whether to create plots (default: True)
        
    Returns:
    --------
    dict
        Dictionary with configuration names as keys and storage DataFrames as values
    """
    # Set default colors and names if not provided
    if config_colors is None:
        config_colors = {
            'nc_single': '#712423',
            'nc': '#712423',
            'c_single': '#976c03',
            'c': '#976c03',
            'c_single_sb': '#82b182',
            'c_sr': '#82b182',
            'nc_sr': '#356891',
            'c_multi': '#356891'
        }
    
    if config_names is None:
        config_names = {
            'nc_single': 'HBV',
            'nc': 'HBV',
            'c_single': 'HBV-GloGEM',
            'c': 'HBV-GloGEM',
            'c_single_sb': 'HBV-GloGEM-SR',
            'c_sr': 'HBV-GloGEM-SR',
            'nc_sr': 'HBV-SR',
            'c_multi': 'HBV-GloGEM-Multi'
        }
    
    # Load storage data for all configurations
    all_storage_data = {}
    
    for config in configs:
        print(f"\nProcessing storage for {gauge_id} - {config}")
        storage_df = load_storage_data(gauge_id, config, base_dir)
        
        if storage_df is not None:
            # Filter by date range if provided
            if start_date:
                storage_df = storage_df[storage_df['date'] >= pd.to_datetime(start_date)]
            if end_date:
                storage_df = storage_df[storage_df['date'] <= pd.to_datetime(end_date)]
            
            all_storage_data[config] = storage_df
    
    if not all_storage_data:
        print("No storage data found for any configuration")
        return None
    
    if plot:
        # Get all storage columns (should be the same across configurations)
        first_config = list(all_storage_data.keys())[0]
        storage_cols = [col for col in all_storage_data[first_config].columns 
                       if col not in ['date', 'month', 'year']]
        
        # Create comparison time series plots
        fig, axes = plt.subplots(len(storage_cols), 1, figsize=(14, 4*len(storage_cols)), sharex=True)
        
        if len(storage_cols) == 1:
            axes = [axes]
        
        # Plot each storage component across all configurations
        for i, col in enumerate(storage_cols):
            ax = axes[i]
            
            for config, storage_df in all_storage_data.items():
                if col in storage_df.columns:
                    color = config_colors.get(config, f"C{list(all_storage_data.keys()).index(config)}")
                    label = config_names.get(config, config)
                    
                    ax.plot(storage_df['date'], storage_df[col], 
                           color=color, linewidth=1.5, label=label, alpha=0.8)
            
            ax.set_title(f'{col} - Configuration Comparison', fontsize=14)
            ax.set_ylabel('Storage (mm)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
        
        # Set x-label only on bottom subplot
        axes[-1].set_xlabel('Date', fontsize=12)
        
        # Add overall title
        period_str = ""
        if start_date and end_date:
            period_str = f" ({start_date} to {end_date})"
        
        fig.suptitle(f'Storage Comparison - Gauge {gauge_id}{period_str}', 
                    fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        # Save plot to each configuration's plots_results folder
        for config in configs:
            save_dir = Path(base_dir) / f"catchment_{gauge_id}_{config}" / "plots_results"
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f"storage_comparison_{gauge_id}.png", dpi=300, bbox_inches='tight')
        
        print(f"Saved storage comparison plots")
        plt.show()
        
        # Create seasonal comparison plot
        fig, axes = plt.subplots(len(storage_cols), 1, figsize=(12, 4*len(storage_cols)), sharex=True)
        
        if len(storage_cols) == 1:
            axes = [axes]
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for i, col in enumerate(storage_cols):
            ax = axes[i]
            
            for config, storage_df in all_storage_data.items():
                if col in storage_df.columns:
                    # Calculate monthly averages
                    monthly_avg = storage_df.groupby('month')[col].mean()
                    
                    color = config_colors.get(config, f"C{list(all_storage_data.keys()).index(config)}")
                    label = config_names.get(config, config)
                    
                    ax.plot(monthly_avg.index, monthly_avg.values, 
                           color=color, linewidth=2, marker='o', markersize=6, 
                           label=label, alpha=0.8)
            
            ax.set_title(f'{col} - Seasonal Pattern Comparison', fontsize=14)
            ax.set_ylabel('Storage (mm)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(month_names)
        
        fig.suptitle(f'Seasonal Storage Patterns - Gauge {gauge_id}', 
                    fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        # Save seasonal comparison plots
        for config in configs:
            save_dir = Path(base_dir) / f"catchment_{gauge_id}_{config}" / "plots_results"
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f"storage_seasonal_comparison_{gauge_id}.png", dpi=300, bbox_inches='tight')
        
        print(f"Saved seasonal storage comparison plots")
        plt.show()
    
    return all_storage_data