# This script is postprocessing Raven output from a single model configuration using namelist
# August 2025

#--------------------------------------------------------------------------------
################################## packages #####################################
#--------------------------------------------------------------------------------

import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use('Agg')  # Use non-interactive backend to prevent image viewer
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
from datetime import datetime, timedelta
import csv
import yaml

#--------------------------------------------------------------------------------
################################## setup ########################################
#--------------------------------------------------------------------------------

def load_namelist(namelist_path='namelist.yaml'):
    """Load configuration from namelist.yaml"""
    with open(namelist_path, 'r') as file:
        return yaml.safe_load(file)

#--------------------------------------------------------------------------------

def setup_output_directories(config):
    """Create output directories for different plot types"""
    gauge_id = config['gauge_id']
    config_dir = Path(config['main_dir']) / config['config_dir']
    base_plots_dir = config_dir / f"catchment_{gauge_id}" / config['model_type'] / "output" / "plots"

    
    plot_dirs = {
        'hydrographs': base_plots_dir / "streamflow",
        'swe': base_plots_dir / "swe",
        'contributions': base_plots_dir / "contributions", 
        'parameters': base_plots_dir / "parameters",
        'storage': base_plots_dir / "storage",
        'forcing': base_plots_dir / "forcing"
    }
    
    # Create directories
    for plot_dir in plot_dirs.values():
        plot_dir.mkdir(parents=True, exist_ok=True)
    
    return plot_dirs

#--------------------------------------------------------------------------------
################################## hydrograph ###################################
#--------------------------------------------------------------------------------

def load_hydrograph_data(config):
    """Load hydrograph data from model directory"""
    config_dir = Path(config['main_dir']) / config['config_dir']
    gauge_id = config['gauge_id']
    hydro_file = config_dir/ f"catchment_{gauge_id}" / config['model_type'] / "output" / f"{gauge_id}_{config['model_type']}_Hydrographs.csv"

    print(f"Loading hydrograph data:")
    print(f"  - File: {hydro_file}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(hydro_file, skiprows=[1])

        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Identify the simulated and observed columns
        sim_col = None
        obs_col = None
        precip_col = None
        
        # Look for columns matching the pattern for simulated, observed flow, and precipitation
        for col in df.columns:
            if '[m3/s]' in col and 'observed' not in col.lower():
                sim_col = col
            elif '[m3/s]' in col and 'observed' in col.lower():
                obs_col = col
            elif 'precip' in col.lower() and '[mm/day]' in col:  # ✅ Look for precip with units
                precip_col = col
        
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
        if precip_col:
            renamed_df['precip'] = df[precip_col] 
            
        print(f"  - Found columns: sim={sim_col}, obs={obs_col}, precip={precip_col}")
        print(f"  - Data range: {renamed_df['date'].min()} to {renamed_df['date'].max()}")
        
        return renamed_df
    
    except Exception as e:
        print(f"  - Error loading data: {e}")
        return None
    
#--------------------------------------------------------------------------------

def plot_hydrological_regime(config, plot_dirs, validation_start=None, validation_end=None, unit='mm'):
    """
    Plot the hydrological regime (monthly mean) for the catchment.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    unit : str, optional
        Unit for discharge ('mm' for mm/day, 'm3' for m³/s), default is 'mm'
    """
    # Use dates from namelist if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    # Load data
    data = load_hydrograph_data(config)
    if data is None:
        print("No hydrograph data loaded")
        return None

    # Load catchment area for conversion if unit='mm'
    conversion_factor = None
    if unit == 'mm':
        config_dir = Path(config['main_dir']) / config['config_dir']
        gauge_id = config['gauge_id']
        topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
        catchment_shape_file = topo_dir / "HRU.shp"
        
        try:
            if catchment_shape_file.exists():
                hru_gdf = gpd.read_file(catchment_shape_file)
                total_area_km2 = hru_gdf['Area_km2'].sum()
                # Conversion factor: m³/s to mm/day
                conversion_factor = 86400 / (total_area_km2 * 1000000) * 1000
                print(f"  - Catchment area: {total_area_km2:.2f} km²")
                print(f"  - Conversion factor: {conversion_factor:.6f}")
            else:
                print(f"  - Warning: Catchment shapefile not found, using m³/s instead")
                unit = 'm3'
        except Exception as e:
            print(f"  - Warning: Could not load catchment area: {e}, using m³/s instead")
            unit = 'm3'

    # Filter for validation period
    validation_mask = (data['date'] >= validation_start) & (data['date'] <= validation_end)
    df_validation = data[validation_mask].copy()

    if len(df_validation) == 0:
        print(f"Warning: No data found for validation period {validation_start} to {validation_end}")
        return None

    # Convert discharge based on unit selection
    if unit == 'mm' and conversion_factor is not None:
        if 'sim_Q' in df_validation.columns:
            df_validation['sim_Q_converted'] = df_validation['sim_Q'] * conversion_factor
        if 'obs_Q' in df_validation.columns:
            df_validation['obs_Q_converted'] = df_validation['obs_Q'] * conversion_factor
        unit_label = 'mm/day'
    else:
        # Keep original units (m³/s)
        if 'sim_Q' in df_validation.columns:
            df_validation['sim_Q_converted'] = df_validation['sim_Q']
        if 'obs_Q' in df_validation.columns:
            df_validation['obs_Q_converted'] = df_validation['obs_Q']
        unit_label = 'm³/s'

    # Calculate monthly means
    df_validation['month'] = df_validation['date'].dt.month
    monthly_data = {}

    if 'sim_Q_converted' in df_validation.columns:
        monthly_data['sim_Q'] = df_validation.groupby('month')['sim_Q_converted'].mean()

    if 'obs_Q_converted' in df_validation.columns:
        monthly_data['obs_Q'] = df_validation.groupby('month')['obs_Q_converted'].mean()

    monthly_df = pd.DataFrame(monthly_data)

    # Plotting
    plt.figure(figsize=(12, 7))

    # Plot observed data if available
    if 'obs_Q' in monthly_df.columns:
        plt.plot(monthly_df.index, monthly_df['obs_Q'], 'k-', linewidth=2.5, label='Observed')

    # Plot simulated data
    if 'sim_Q' in monthly_df.columns:
        plt.plot(monthly_df.index, monthly_df['sim_Q'], 'C0', linewidth=2, label='Simulated')

    plt.xlabel('Month', fontsize=14)
    plt.ylabel(f'Discharge ({unit_label})', fontsize=14)
    plt.title(f'Hydrological Regime - Monthly Mean for Validation Period ({validation_start} to {validation_end})\nCatchment {config["gauge_id"]}', fontsize=16)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)

    # Add performance metrics if both sim and obs are available
    if 'obs_Q_converted' in df_validation.columns and 'sim_Q_converted' in df_validation.columns:
        obs = df_validation['obs_Q_converted'].values
        sim = df_validation['sim_Q_converted'].values
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

        perf_text = f"Validation Performance:\nNSE={nse:.3f}, KGE={kge:.3f}"
        plt.figtext(0.02, 0.02, perf_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

    # Save plot
    save_path = plot_dirs['hydrographs'] / f'hydrological_regime_{unit}_{config["gauge_id"]}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved hydrological regime plot ({unit_label}) to: {save_path}")
    plt.show()

    return monthly_df

#--------------------------------------------------------------------------------

def plot_hydrograph_timeseries(config, plot_dirs, validation_start=None, validation_end=None, random_seed=42, unit='mm'):
    """
    Plot the hydrograph time series for calibration and validation periods in two subplots,
    and plot a random year from the validation period.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    random_seed : int, optional
        Seed for random year selection
    unit : str, optional
        Unit for discharge ('mm' for mm/day, 'm3' for m³/s), default is 'mm'
    """
    # Load data
    data = load_hydrograph_data(config)
    if data is None:
        print("No hydrograph data loaded")
        return None

    # Use dates from namelist if not provided
    cali_start = config.get('start_date', '2000-01-01')
    cali_end = config.get('cali_end_date', '2009-12-31')
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')

    # Load catchment area for conversion if unit='mm'
    conversion_factor = None
    if unit == 'mm':
        config_dir = Path(config['main_dir']) / config['config_dir']
        gauge_id = config['gauge_id']
        topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
        catchment_shape_file = topo_dir / "HRU.shp"
        
        try:
            if catchment_shape_file.exists():
                hru_gdf = gpd.read_file(catchment_shape_file)
                total_area_km2 = hru_gdf['Area_km2'].sum()
                # Conversion factor: m³/s to mm/day
                conversion_factor = 86400 / (total_area_km2 * 1000000) * 1000
                print(f"  - Catchment area: {total_area_km2:.2f} km²")
                print(f"  - Conversion factor: {conversion_factor:.6f}")
            else:
                print(f"  - Warning: Catchment shapefile not found, using m³/s instead")
                unit = 'm3'
        except Exception as e:
            print(f"  - Warning: Could not load catchment area: {e}, using m³/s instead")
            unit = 'm3'

    # Convert discharge based on unit selection
    if unit == 'mm' and conversion_factor is not None:
        if 'sim_Q' in data.columns:
            data['sim_Q_converted'] = data['sim_Q'] * conversion_factor
        if 'obs_Q' in data.columns:
            data['obs_Q_converted'] = data['obs_Q'] * conversion_factor
        unit_label = 'mm/day'
    else:
        # Keep original units (m³/s)
        if 'sim_Q' in data.columns:
            data['sim_Q_converted'] = data['sim_Q']
        if 'obs_Q' in data.columns:
            data['obs_Q_converted'] = data['obs_Q']
        unit_label = 'm³/s'

    # Calibration and validation masks
    cali_mask = (data['date'] >= cali_start) & (data['date'] <= cali_end)
    val_mask = (data['date'] >= validation_start) & (data['date'] <= validation_end)

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    
    # Calibration period
    ax = axes[0]
    if 'obs_Q_converted' in data.columns:
        ax.plot(data[cali_mask]['date'], data[cali_mask]['obs_Q_converted'], 'k-', label='Observed')
    if 'sim_Q_converted' in data.columns:
        ax.plot(data[cali_mask]['date'], data[cali_mask]['sim_Q_converted'], 'C0', label='Simulated')
    ax.set_title(f'Calibration Period ({cali_start} to {cali_end})')
    ax.set_ylabel(f'Discharge ({unit_label})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # Validation period
    ax = axes[1]
    if 'obs_Q_converted' in data.columns:
        ax.plot(data[val_mask]['date'], data[val_mask]['obs_Q_converted'], 'k-', label='Observed')
    if 'sim_Q_converted' in data.columns:
        ax.plot(data[val_mask]['date'], data[val_mask]['sim_Q_converted'], 'C0', label='Simulated')
    ax.set_title(f'Validation Period ({validation_start} to {validation_end})')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'Discharge ({unit_label})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = plot_dirs['hydrographs'] / f'hydrograph_timeseries_split_{unit}_{config["gauge_id"]}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved split hydrograph time series plot ({unit_label}) to: {save_path}")
    plt.show()

    # Pick a random year in validation period
    val_years = pd.Series(data[val_mask]['date'].dt.year.unique())
    if len(val_years) == 0:
        print("No years found in validation period.")
        return
    np.random.seed(random_seed)
    rand_year = np.random.choice(val_years)
    year_mask = (data['date'].dt.year == rand_year) & val_mask
    
    plt.figure(figsize=(14, 6))
    if 'obs_Q_converted' in data.columns:
        plt.plot(data[year_mask]['date'], data[year_mask]['obs_Q_converted'], 'k-', label='Observed')
    if 'sim_Q_converted' in data.columns:
        plt.plot(data[year_mask]['date'], data[year_mask]['sim_Q_converted'], 'C0', label='Simulated')
    plt.xlabel('Date')
    plt.ylabel(f'Discharge ({unit_label})')
    plt.title(f'Hydrograph for Random Validation Year {rand_year} - Catchment {config["gauge_id"]}')
    plt.legend()
    plt.tight_layout()
    save_path = plot_dirs['hydrographs'] / f'hydrograph_random_year_{rand_year}_{unit}_{config["gauge_id"]}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved hydrograph for random year plot ({unit_label}) to: {save_path}")
    plt.show()

#--------------------------------------------------------------------------------

def plot_streamflow_scatter(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Create scatter plot for observed vs simulated streamflow.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    dict
        Dictionary containing statistics and plot path
    """
    
    gauge_id = config['gauge_id']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"Creating streamflow scatter plot for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    
    # Load streamflow data
    data = load_hydrograph_data(config)
    if data is None:
        print("ERROR: Could not load hydrograph data")
        return None
    
    # Filter for validation period
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    
    mask = (data['date'] >= start_date) & (data['date'] <= end_date)
    df = data[mask].copy()
    
    if len(df) == 0:
        print(f"ERROR: No data found for period {validation_start} to {validation_end}")
        return None
    
    # Check for required columns
    if 'obs_Q' not in df.columns or 'sim_Q' not in df.columns:
        print("ERROR: Hydrograph file must contain 'obs_Q' and 'sim_Q' columns")
        return None
    
    # Remove NaN values
    df = df.dropna(subset=['obs_Q', 'sim_Q'])
    
    if len(df) == 0:
        print("ERROR: No valid data points after removing NaN values")
        return None
    
    print(f"  - Found {len(df)} valid data points")
    
    obs = df['obs_Q'].values
    sim = df['sim_Q'].values
    
    # Calculate statistics
    from scipy.stats import linregress
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(obs, sim)
    
    # Performance metrics
    obs_mean = np.mean(obs)
    nse = 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - obs_mean) ** 2))
    
    # KGE
    std_sim = np.std(sim)
    std_obs = np.std(obs)
    mean_sim = np.mean(sim)
    mean_obs = np.mean(obs)
    corr = np.corrcoef(sim, obs)[0, 1]
    alpha = std_sim / std_obs
    beta = mean_sim / mean_obs
    kge = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    # RMSE and Bias
    rmse = np.sqrt(np.mean((obs - sim)**2))
    bias = np.mean(sim - obs)
    relative_bias = (bias / mean_obs) * 100
    
    print(f"  - R² = {r_value**2:.3f}")
    print(f"  - NSE = {nse:.3f}")
    print(f"  - KGE = {kge:.3f}")
    print(f"  - RMSE = {rmse:.3f} m³/s")
    print(f"  - Bias = {bias:.3f} m³/s ({relative_bias:+.1f}%)")
    
    # Create scatter plot
    plt.figure(figsize=(10, 10))
    
    # Plot data points
    plt.scatter(obs, sim, alpha=0.5, s=20, c='steelblue', edgecolors='navy', linewidth=0.5)
    
    # Add 1:1 line
    min_val = min(obs.min(), sim.min())
    max_val = max(obs.max(), sim.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, 
            label='1:1 Line', zorder=10)
    
    # Add regression line
    line_x = np.array([min_val, max_val])
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, 'r-', linewidth=2, 
            label=f'Regression (R²={r_value**2:.3f})', zorder=9)
    
    # Formatting
    plt.xlabel('Observed Streamflow (m³/s)', fontsize=12, fontweight='bold')
    plt.ylabel('Simulated Streamflow (m³/s)', fontsize=12, fontweight='bold')
    plt.title(f'Observed vs Simulated Streamflow\nCatchment {gauge_id}', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add statistics text box
    stats_text = (f"Statistics:\n"
                 f"R² = {r_value**2:.3f}\n"
                 f"NSE = {nse:.3f}\n"
                 f"KGE = {kge:.3f}\n"
                 f"RMSE = {rmse:.3f} m³/s\n"
                 f"Bias = {bias:+.3f} m³/s\n"
                 f"n = {len(df)}")
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['hydrographs'] / f'streamflow_scatter_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved scatter plot to: {save_path}")
    plt.show()
    
    # Print detailed summary
    print(f"\n{'='*60}")
    print(f"STREAMFLOW SCATTER PLOT SUMMARY - CATCHMENT {gauge_id}")
    print(f"{'='*60}")
    print(f"Period: {validation_start} to {validation_end}")
    print(f"Number of data points: {len(df)}")
    
    print(f"\nObserved Streamflow:")
    print(f"  Mean: {mean_obs:.3f} m³/s")
    print(f"  Std Dev: {std_obs:.3f} m³/s")
    print(f"  Min: {obs.min():.3f} m³/s")
    print(f"  Max: {obs.max():.3f} m³/s")
    
    print(f"\nSimulated Streamflow:")
    print(f"  Mean: {mean_sim:.3f} m³/s")
    print(f"  Std Dev: {std_sim:.3f} m³/s")
    print(f"  Min: {sim.min():.3f} m³/s")
    print(f"  Max: {sim.max():.3f} m³/s")
    
    print(f"\nPerformance Metrics:")
    print(f"  R² (coefficient of determination): {r_value**2:.3f}")
    print(f"  NSE (Nash-Sutcliffe Efficiency): {nse:.3f}")
    print(f"  KGE (Kling-Gupta Efficiency): {kge:.3f}")
    print(f"    - Correlation (r): {corr:.3f}")
    print(f"    - Variability ratio (α): {alpha:.3f}")
    print(f"    - Bias ratio (β): {beta:.3f}")
    print(f"  RMSE (Root Mean Square Error): {rmse:.3f} m³/s")
    print(f"  Bias: {bias:+.3f} m³/s ({relative_bias:+.1f}%)")
    
    print(f"\nRegression:")
    print(f"  Slope: {slope:.3f}")
    print(f"  Intercept: {intercept:.3f} m³/s")
    print(f"  p-value: {p_value:.6f}")
    
    print(f"{'='*60}\n")
    
    # Return results
    return {
        'statistics': {
            'r_squared': r_value**2,
            'nse': nse,
            'kge': kge,
            'correlation': corr,
            'alpha': alpha,
            'beta': beta,
            'rmse': rmse,
            'bias': bias,
            'relative_bias_pct': relative_bias,
            'slope': slope,
            'intercept': intercept,
            'p_value': p_value,
            'n_points': len(df)
        },
        'observed': {
            'mean': mean_obs,
            'std': std_obs,
            'min': obs.min(),
            'max': obs.max()
        },
        'simulated': {
            'mean': mean_sim,
            'std': std_sim,
            'min': sim.min(),
            'max': sim.max()
        },
        'save_path': save_path
    }

#--------------------------------------------------------------------------------

def plot_streamflow_residuals(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Create residual plot for streamflow (Simulated - Observed vs Observed).
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    dict
        Dictionary containing residual statistics and plot path
    """
    
    gauge_id = config['gauge_id']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"Creating streamflow residual plot for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    
    # Load streamflow data
    data = load_hydrograph_data(config)
    if data is None:
        print("ERROR: Could not load hydrograph data")
        return None
    
    # Filter for validation period
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    
    mask = (data['date'] >= start_date) & (data['date'] <= end_date)
    df = data[mask].copy()
    
    if len(df) == 0:
        print(f"ERROR: No data found for period {validation_start} to {validation_end}")
        return None
    
    # Check for required columns
    if 'obs_Q' not in df.columns or 'sim_Q' not in df.columns:
        print("ERROR: Hydrograph file must contain 'obs_Q' and 'sim_Q' columns")
        return None
    
    # Remove NaN values
    df = df.dropna(subset=['obs_Q', 'sim_Q'])
    
    if len(df) == 0:
        print("ERROR: No valid data points after removing NaN values")
        return None
    
    print(f"  - Found {len(df)} valid data points")
    
    # Calculate residuals
    df['residual'] = df['sim_Q'] - df['obs_Q']
    
    obs = df['obs_Q'].values
    residuals = df['residual'].values
    
    # Calculate residual statistics
    bias = np.mean(residuals)
    std_residual = np.std(residuals)
    min_residual = residuals.min()
    max_residual = residuals.max()
    median_residual = np.median(residuals)
    
    # Calculate percentage of points within ±2σ
    within_2sigma = np.sum(np.abs(residuals) <= 2*std_residual)
    pct_within_2sigma = (within_2sigma / len(residuals)) * 100
    
    print(f"  - Mean bias: {bias:+.3f} m³/s")
    print(f"  - Std Dev: {std_residual:.3f} m³/s")
    print(f"  - Within ±2σ: {pct_within_2sigma:.1f}%")
    
    # Create residual plot
    plt.figure(figsize=(12, 8))
    
    # Plot residuals vs observed
    plt.scatter(obs, residuals, alpha=0.5, s=20, 
               c='coral', edgecolors='darkred', linewidth=0.5)
    
    # Add zero line (perfect prediction)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Zero residual')
    
    # Add mean bias line
    plt.axhline(y=bias, color='red', linestyle='-', linewidth=2, alpha=0.7,
               label=f'Mean Bias = {bias:+.3f} m³/s')
    
    # Add ±2 standard deviations lines
    plt.axhline(y=2*std_residual, color='red', linestyle=':', linewidth=1.5, alpha=0.5,
               label=f'±2σ = ±{2*std_residual:.3f} m³/s')
    plt.axhline(y=-2*std_residual, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Formatting
    plt.xlabel('Observed Streamflow (m³/s)', fontsize=12, fontweight='bold')
    plt.ylabel('Residual (Sim - Obs) (m³/s)', fontsize=12, fontweight='bold')
    plt.title(f'Streamflow Residual Plot\nCatchment {gauge_id}', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add residual statistics text box
    residual_stats = (f"Residual Statistics:\n"
                     f"Mean: {bias:+.3f} m³/s\n"
                     f"Std Dev: {std_residual:.3f} m³/s\n"
                     f"Min: {min_residual:+.3f} m³/s\n"
                     f"Max: {max_residual:+.3f} m³/s\n"
                     f"Median: {median_residual:+.3f} m³/s\n"
                     f"Within ±2σ: {pct_within_2sigma:.1f}%")
    
    plt.text(0.02, 0.98, residual_stats, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['hydrographs'] / f'streamflow_residuals_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved residual plot to: {save_path}")
    plt.show()
    
    # Print detailed summary
    print(f"\n{'='*60}")
    print(f"STREAMFLOW RESIDUAL ANALYSIS - CATCHMENT {gauge_id}")
    print(f"{'='*60}")
    print(f"Period: {validation_start} to {validation_end}")
    print(f"Number of data points: {len(df)}")
    
    print(f"\nResidual Statistics:")
    print(f"  Mean (Bias): {bias:+.3f} m³/s")
    print(f"  Std Dev: {std_residual:.3f} m³/s")
    print(f"  Min: {min_residual:+.3f} m³/s")
    print(f"  Max: {max_residual:+.3f} m³/s")
    print(f"  Median: {median_residual:+.3f} m³/s")
    print(f"  Range: {max_residual - min_residual:.3f} m³/s")
    
    print(f"\nDistribution:")
    print(f"  Points within ±1σ: {np.sum(np.abs(residuals) <= std_residual)}/{len(residuals)} ({np.sum(np.abs(residuals) <= std_residual)/len(residuals)*100:.1f}%)")
    print(f"  Points within ±2σ: {within_2sigma}/{len(residuals)} ({pct_within_2sigma:.1f}%)")
    print(f"  Points within ±3σ: {np.sum(np.abs(residuals) <= 3*std_residual)}/{len(residuals)} ({np.sum(np.abs(residuals) <= 3*std_residual)/len(residuals)*100:.1f}%)")
    
    # Check for systematic patterns
    print(f"\nSystematic Patterns:")
    
    # Check if bias is significantly different from zero (simple t-test approximation)
    t_stat = bias / (std_residual / np.sqrt(len(residuals)))
    if abs(t_stat) > 2:  # Rough significance at 95% confidence
        print(f"  ⚠️  Significant systematic bias detected (t={t_stat:+.2f})")
    else:
        print(f"  ✓ No significant systematic bias (t={t_stat:+.2f})")
    
    # Check for heteroscedasticity (residuals increasing with magnitude)
    # Split data into low and high flow
    median_obs = np.median(obs)
    low_flow_std = np.std(residuals[obs <= median_obs])
    high_flow_std = np.std(residuals[obs > median_obs])
    
    if high_flow_std > 1.5 * low_flow_std:
        print(f"  ⚠️  Heteroscedasticity detected (higher errors at high flows)")
        print(f"     Low flow std: {low_flow_std:.3f}, High flow std: {high_flow_std:.3f}")
    else:
        print(f"  ✓ Relatively homogeneous error distribution")
        print(f"     Low flow std: {low_flow_std:.3f}, High flow std: {high_flow_std:.3f}")
    
    print(f"{'='*60}\n")
    
    # Return results
    return {
        'residuals': {
            'mean': bias,
            'std': std_residual,
            'min': min_residual,
            'max': max_residual,
            'median': median_residual,
            'range': max_residual - min_residual,
            'pct_within_2sigma': pct_within_2sigma
        },
        'distribution': {
            'within_1sigma': np.sum(np.abs(residuals) <= std_residual),
            'within_2sigma': within_2sigma,
            'within_3sigma': np.sum(np.abs(residuals) <= 3*std_residual)
        },
        'systematic_patterns': {
            't_statistic': t_stat,
            'significant_bias': abs(t_stat) > 2,
            'low_flow_std': low_flow_std,
            'high_flow_std': high_flow_std,
            'heteroscedasticity': high_flow_std > 1.5 * low_flow_std
        },
        'n_points': len(df),
        'save_path': save_path
    }

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
    try:
        period_data['obs_Q'] = pd.to_numeric(period_data['obs_Q'], errors='coerce')
        period_data['sim_Q'] = pd.to_numeric(period_data['sim_Q'], errors='coerce')
        
        obs = period_data['obs_Q'].values
        sim = period_data['sim_Q'].values
        
        # Check for NaN values
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
        sim_ranks = np.argsort(np.argsort(sim))
        obs_ranks = np.argsort(np.argsort(obs))
        spearman_corr = np.corrcoef(sim_ranks, obs_ranks)[0, 1]
        
        # Alpha NP - ratio of flow duration curve slopes
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

def plot_performance_metrics_summary(config, plot_dirs):
    """
    Calculate and display performance metrics for calibration and validation periods
    in a text-based plot.
    """
    # Load hydrograph data
    data = load_hydrograph_data(config)
    if data is None:
        print("No hydrograph data loaded for metrics calculation")
        return None
    
    # Check if both observed and simulated data are available
    if 'obs_Q' not in data.columns or 'sim_Q' not in data.columns:
        print("Both observed and simulated discharge data are required for metrics calculation")
        return None
    
    # Get date ranges from config
    cali_start = pd.to_datetime(config.get('start_date', '2000-01-01'))
    cali_end = pd.to_datetime(config.get('cali_end_date', '2009-12-31'))
    val_start = pd.to_datetime(config.get('cali_end_date', '2010-01-01'))
    val_end = pd.to_datetime(config.get('end_date', '2020-12-31'))
    
    print(f"\nCalculating performance metrics for Catchment {config['gauge_id']}:")
    print("=" * 60)
    
    # Calculate metrics for calibration period
    cali_metrics = calculate_performance_metrics(data, cali_start, cali_end, "Calibration")
    
    # Calculate metrics for validation period
    val_metrics = calculate_performance_metrics(data, val_start, val_end, "Validation")
    
    # Calculate metrics for entire period
    entire_start = data['date'].min()
    entire_end = data['date'].max()
    entire_metrics = calculate_performance_metrics(data, entire_start, entire_end, "Entire Period")
    
    # Create a text-based summary plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')  # Turn off axes
    
    # Create summary text
    summary_text = f"HYDROLOGICAL MODEL PERFORMANCE SUMMARY\n"
    summary_text += f"Catchment: {config['gauge_id']}\n"
    summary_text += f"Model: {config.get('model_type', 'N/A')}\n"
    summary_text += "=" * 50 + "\n\n"
    
    # Helper function to format metrics
    def format_metrics_section(metrics, period_name, start_date, end_date):
        if metrics is None:
            return f"{period_name}:\n  No data available\n\n"
        
        section = f"{period_name}:\n"
        section += f"  Period: {start_date.date()} to {end_date.date()}\n"
        section += f"  Data Points: {metrics['n_points']}\n"
        section += f"  NSE:     {metrics['NSE']:7.3f}\n"
        section += f"  KGE:     {metrics['KGE']:7.3f}\n"
        section += f"  KGE_NP:  {metrics['KGE_NP']:7.3f}\n"
        section += f"\n  KGE Components:\n"
        section += f"    Correlation (r):     {metrics['r']:7.3f}\n"
        section += f"    Variability (α):     {metrics['alpha']:7.3f}\n"
        section += f"    Bias (β):            {metrics['beta']:7.3f}\n"
        section += f"\n  KGE_NP Components:\n"
        section += f"    Spearman Corr. (rs): {metrics['r_spearman']:7.3f}\n"
        section += f"    Variability (α_np):  {metrics['alpha_np']:7.3f}\n"
        section += f"    Bias (β):            {metrics['beta']:7.3f}\n"
        section += "\n"
        return section
    
    # Add calibration metrics
    summary_text += format_metrics_section(cali_metrics, "CALIBRATION PERIOD", cali_start, cali_end)
    
    # Add validation metrics
    summary_text += format_metrics_section(val_metrics, "VALIDATION PERIOD", val_start, val_end)
    
    # Add entire period metrics
    summary_text += format_metrics_section(entire_metrics, "ENTIRE PERIOD", entire_start, entire_end)
    
    # Add performance interpretation
    summary_text += "PERFORMANCE INTERPRETATION:\n"
    summary_text += "  NSE:    > 0.75 = very good, > 0.65 = good, > 0.50 = satisfactory\n"
    summary_text += "  KGE:    > 0.75 = very good, > 0.65 = good, > 0.50 = satisfactory\n"
    summary_text += "  KGE_NP: > 0.75 = very good, > 0.65 = good, > 0.50 = satisfactory\n\n"
    
    # Add component interpretation
    summary_text += "COMPONENT INTERPRETATION:\n"
    summary_text += "  α, α_np: measures variability ratio (1 = perfect)\n"
    summary_text += "  β:       measures bias ratio (1 = perfect)\n"
    summary_text += "  r, rs:   measures correlation (1 = perfect)\n"
    
    # Display text on plot
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.title(f'Performance Metrics Summary - Catchment {config["gauge_id"]}', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Save plot
    save_path = plot_dirs['hydrographs'] / f'performance_metrics_summary_{config["gauge_id"]}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved performance metrics summary to: {save_path}")
    plt.show()
    
    # Return metrics for further use
    return {
        'calibration': cali_metrics,
        'validation': val_metrics,
        'entire_period': entire_metrics
    }

#--------------------------------------------------------------------------------
################################### forcing #####################################
#--------------------------------------------------------------------------------

def load_forcing_by_hrugroup(config, forcing_type='SNOWFALL'):
    """
    Load forcing data by HRU group (e.g., SNOWFALL or RAINFALL)
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    forcing_type : str
        Type of forcing data ('SNOWFALL' or 'RAINFALL')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with date and NO_GLACIER columns
    """
    config_dir = Path(config['main_dir']) / config['config_dir']
    gauge_id = config['gauge_id']
    forcing_file = config_dir / f"catchment_{gauge_id}" / config['model_type'] / "output" / f"{gauge_id}_{config['model_type']}_{forcing_type}_Daily_Average_ByHRUGroup.csv"
    
    print(f"Loading {forcing_type} data:")
    print(f"  - File: {forcing_file}")
    
    try:
        # Read the CSV file, skipping the second row (row index 1)
        df = pd.read_csv(forcing_file, skiprows=[1])
        
        # Rename the date column - it's called 'HRUGroup:' in the file
        if 'HRUGroup:' in df.columns:
            df = df.rename(columns={'HRUGroup:': 'date'})
        else:
            print(f"  - Warning: 'HRUGroup:' column not found")
            print(f"  - Available columns: {df.columns.tolist()}")
            return None
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Check if NO_GLACIER column exists
        if 'NO_GLACIER' not in df.columns:
            print(f"  - Warning: NO_GLACIER column not found in {forcing_type} file")
            print(f"  - Available columns: {df.columns.tolist()}")
            return None
        
        print(f"  - Found {len(df)} days of data")
        print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  - Mean {forcing_type}: {df['NO_GLACIER'].mean():.2f} mm/day")
        
        return df[['date', 'NO_GLACIER']].copy()
    
    except Exception as e:
        print(f"  - Error loading {forcing_type} data: {e}")
        import traceback
        traceback.print_exc()
        return None

#--------------------------------------------------------------------------------

def plot_precipitation_partitioning(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot monthly rainfall vs snowfall partitioning for non-glacier area.
    Creates a stacked bar chart showing total precipitation with colors for rain and snow.
    Scales precipitation from non-glacier area to total catchment area for comparison with total runoff.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for analysis period
    validation_end : str, optional
        End date for analysis period
    """
    gauge_id = config['gauge_id']
    config_dir = Path(config['main_dir']) / config['config_dir']
    
    # Use dates from namelist if not provided
    if validation_start is None:
        validation_start = config.get('start_date', '2000-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"\nCreating precipitation partitioning plot for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    
    # Load HRU shapefile to get areas
    topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
    hru_shapefile = topo_dir / "HRU.shp"
    
    print(f"  - Loading HRU shapefile: {hru_shapefile}")
    
    if not hru_shapefile.exists():
        print(f"ERROR: HRU shapefile not found: {hru_shapefile}")
        return None
    
    try:
        import geopandas as gpd
        hru_gdf = gpd.read_file(hru_shapefile)
        original_count = len(hru_gdf)
        print(f"  - Loaded {original_count} HRUs from shapefile")
        
        # Calculate total catchment area
        total_area_km2 = hru_gdf['Area_km2'].sum()
        
        # Calculate glacier and non-glacier areas
        # Glacier HRUs have Landuse_Cl values of 7 or 8
        if 'Landuse_Cl' in hru_gdf.columns:
            glacier_area_km2 = hru_gdf[hru_gdf['Landuse_Cl'].isin([7, 8])]['Area_km2'].sum()
            non_glacier_area_km2 = hru_gdf[~hru_gdf['Landuse_Cl'].isin([7, 8])]['Area_km2'].sum()
            glacier_hru_count = len(hru_gdf[hru_gdf['Landuse_Cl'].isin([7, 8])])
            non_glacier_hru_count = len(hru_gdf[~hru_gdf['Landuse_Cl'].isin([7, 8])])
            print(f"  - Glacier HRUs (Landuse_Cl 7, 8): {glacier_hru_count}")
            print(f"  - Non-glacier HRUs: {non_glacier_hru_count}")
        else:
            print("  - Warning: 'Landuse_Cl' column not found, cannot identify glacier areas")
            print("  - Assuming all area is non-glacier")
            non_glacier_area_km2 = total_area_km2
            glacier_area_km2 = 0.0
        
        area_fraction = non_glacier_area_km2 / total_area_km2 if total_area_km2 > 0 else 1.0
        
        print(f"  - Total catchment area: {total_area_km2:.2f} km²")
        print(f"  - Non-glacier area: {non_glacier_area_km2:.2f} km² ({area_fraction*100:.1f}%)")
        print(f"  - Glacier area: {glacier_area_km2:.2f} km² ({(1-area_fraction)*100:.1f}%)")
        
    except Exception as e:
        print(f"ERROR: Could not load HRU shapefile: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Load snowfall data
    snowfall_df = load_forcing_by_hrugroup(config, 'SNOWFALL')
    if snowfall_df is None:
        print("ERROR: Could not load snowfall data")
        return None
    
    # Load rainfall data
    rainfall_df = load_forcing_by_hrugroup(config, 'RAINFALL')
    if rainfall_df is None:
        print("ERROR: Could not load rainfall data")
        return None
    
    # Merge the datasets
    df = pd.merge(
        snowfall_df.rename(columns={'NO_GLACIER': 'snowfall'}),
        rainfall_df.rename(columns={'NO_GLACIER': 'rainfall'}),
        on='date',
        how='inner'
    )
    
    print(f"\n  - Merged dataset has {len(df)} days")
    
    # Filter for analysis period
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_period = df[mask].copy()
    
    if len(df_period) == 0:
        print(f"ERROR: No data found for period {validation_start} to {validation_end}")
        return None
    
    print(f"  - Analysis period has {len(df_period)} days")
    
    # Calculate total precipitation (non-glacier area values)
    df_period['total_precip_nonglacier'] = df_period['snowfall'] + df_period['rainfall']
    
    # Scale to catchment area (multiply by area fraction to get catchment-wide values)
    df_period['snowfall_scaled'] = df_period['snowfall'] * area_fraction
    df_period['rainfall_scaled'] = df_period['rainfall'] * area_fraction
    df_period['total_precip_scaled'] = df_period['total_precip_nonglacier'] * area_fraction
    
    # Extract month
    df_period['month'] = df_period['date'].dt.month
    
    # Calculate monthly averages for SCALED values (mean of daily values)
    monthly_snowfall = df_period.groupby('month')['snowfall_scaled'].mean()
    monthly_rainfall = df_period.groupby('month')['rainfall_scaled'].mean()
    monthly_total = df_period.groupby('month')['total_precip_scaled'].mean()
    
    # Also keep non-scaled for reference
    monthly_snowfall_nonglacier = df_period.groupby('month')['snowfall'].mean()
    monthly_rainfall_nonglacier = df_period.groupby('month')['rainfall'].mean()
    monthly_total_nonglacier = df_period.groupby('month')['total_precip_nonglacier'].mean()
    
    # Calculate percentages (these remain the same whether scaled or not)
    snow_fraction = (monthly_snowfall / monthly_total * 100).fillna(0)
    rain_fraction = (monthly_rainfall / monthly_total * 100).fillna(0)
    
    print(f"\n  NON-GLACIER AREA VALUES (original):")
    print(f"  - Mean daily precipitation: {df_period['total_precip_nonglacier'].mean():.2f} mm/day")
    print(f"  - Mean daily snowfall: {df_period['snowfall'].mean():.2f} mm/day ({df_period['snowfall'].mean()/df_period['total_precip_nonglacier'].mean()*100:.1f}%)")
    print(f"  - Mean daily rainfall: {df_period['rainfall'].mean():.2f} mm/day ({df_period['rainfall'].mean()/df_period['total_precip_nonglacier'].mean()*100:.1f}%)")
    
    print(f"\n  SCALED TO CATCHMENT AREA:")
    print(f"  - Mean daily precipitation: {df_period['total_precip_scaled'].mean():.2f} mm/day")
    print(f"  - Mean daily snowfall: {df_period['snowfall_scaled'].mean():.2f} mm/day ({df_period['snowfall_scaled'].mean()/df_period['total_precip_scaled'].mean()*100:.1f}%)")
    print(f"  - Mean daily rainfall: {df_period['rainfall_scaled'].mean():.2f} mm/day ({df_period['rainfall_scaled'].mean()/df_period['total_precip_scaled'].mean()*100:.1f}%)")
    print(f"  - Scaling factor: {area_fraction:.4f} (non-glacier fraction)")

    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Month labels
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create stacked bar chart
    # Bottom part: rainfall (blue)
    bars_rain = ax.bar(months, monthly_rainfall, 
                       color='steelblue', label='Rainfall', 
                       edgecolor='navy', linewidth=0.5)
    
    # Top part: snowfall (white/light blue)
    bars_snow = ax.bar(months, monthly_snowfall, 
                       bottom=monthly_rainfall,
                       color='lightcyan', label='Snowfall',
                       edgecolor='darkblue', linewidth=0.5)
    
    # Formatting
    ax.set_xlabel('Month', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Daily Precipitation (mm/day)', fontsize=14, fontweight='bold')
    ax.set_title(f'Monthly Mean Precipitation Partitioning (Scaled to Catchment Area)\nNon-Glacier Area × {area_fraction:.2f} - Catchment {gauge_id}\n{validation_start} to {validation_end}',
                fontsize=16, fontweight='bold')
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for i, month in enumerate(months):
        total = monthly_total.get(month, 0)
        if total > 0:
            # Snow percentage at top
            snow_pct = snow_fraction.get(month, 0)
            if snow_pct > 5:  # Only show if >5%
                y_pos = monthly_rainfall.get(month, 0) + monthly_snowfall.get(month, 0) * 0.5
                ax.text(month, y_pos, f'{snow_pct:.0f}%',
                       ha='center', va='center', fontsize=9, fontweight='bold',
                       color='darkblue')
            
            # Rain percentage at bottom
            rain_pct = rain_fraction.get(month, 0)
            if rain_pct > 5:  # Only show if >5%
                y_pos = monthly_rainfall.get(month, 0) * 0.5
                ax.text(month, y_pos, f'{rain_pct:.0f}%',
                       ha='center', va='center', fontsize=9, fontweight='bold',
                       color='white')
    
    # Add statistics text box
    stats_text = (f"Catchment Statistics:\n"
                 f"Total Area: {total_area_km2:.2f} km²\n"
                 f"Non-Glacier: {non_glacier_area_km2:.2f} km² ({area_fraction*100:.1f}%)\n"
                 f"Glacier: {glacier_area_km2:.2f} km² ({(1-area_fraction)*100:.1f}%)\n"
                 f"\n"
                 f"Scaled Precipitation:\n"
                 f"Mean Daily: {df_period['total_precip_scaled'].mean():.2f} mm/day\n"
                 f"Snowfall: {df_period['snowfall_scaled'].mean():.2f} mm/day ({df_period['snowfall_scaled'].mean()/df_period['total_precip_scaled'].mean()*100:.1f}%)\n"
                 f"Rainfall: {df_period['rainfall_scaled'].mean():.2f} mm/day ({df_period['rainfall_scaled'].mean()/df_period['total_precip_scaled'].mean()*100:.1f}%)\n"
                 f"Days: {len(df_period)}")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['forcing'] / f'precipitation_partitioning_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved precipitation partitioning plot to: {save_path}")
    plt.show()
    
    # Print detailed monthly summary
    print(f"\n{'='*100}")
    print(f"MONTHLY MEAN PRECIPITATION PARTITIONING - CATCHMENT {gauge_id}")
    print(f"{'='*100}")
    print(f"Period: {validation_start} to {validation_end}")
    print(f"Scaled to Catchment Area (Non-Glacier Fraction: {area_fraction:.4f})")
    print(f"Total Catchment Area: {total_area_km2:.2f} km² | Non-Glacier: {non_glacier_area_km2:.2f} km² | Glacier: {glacier_area_km2:.2f} km²")
    print(f"\n{'Month':<10} {'Rainfall':<18} {'Snowfall':<18} {'Total':<18} {'Rain %':<10} {'Snow %':<10}")
    print(f"{'-'*100}")
    
    for month in months:
        month_name = month_names[month-1]
        rain = monthly_rainfall.get(month, 0)
        snow = monthly_snowfall.get(month, 0)
        total = monthly_total.get(month, 0)
        rain_pct = rain_fraction.get(month, 0)
        snow_pct = snow_fraction.get(month, 0)
        
        print(f"{month_name:<10} {rain:>13.2f} mm/day {snow:>13.2f} mm/day {total:>13.2f} mm/day {rain_pct:>8.1f}% {snow_pct:>8.1f}%")
    
    print(f"{'-'*100}")
    print(f"{'ANNUAL AVG':<10} {monthly_rainfall.mean():>13.2f} mm/day {monthly_snowfall.mean():>13.2f} mm/day {monthly_total.mean():>13.2f} mm/day {monthly_rainfall.sum()/monthly_total.sum()*100:>8.1f}% {monthly_snowfall.sum()/monthly_total.sum()*100:>8.1f}%")
    print(f"{'='*100}\n")
    
    # Return summary statistics
    return {
        'catchment_info': {
            'total_area_km2': total_area_km2,
            'non_glacier_area_km2': non_glacier_area_km2,
            'glacier_area_km2': glacier_area_km2,
            'area_fraction': area_fraction
        },
        'monthly_rainfall_scaled': monthly_rainfall.to_dict(),
        'monthly_snowfall_scaled': monthly_snowfall.to_dict(),
        'monthly_total_scaled': monthly_total.to_dict(),
        'monthly_rainfall_nonglacier': monthly_rainfall_nonglacier.to_dict(),
        'monthly_snowfall_nonglacier': monthly_snowfall_nonglacier.to_dict(),
        'monthly_total_nonglacier': monthly_total_nonglacier.to_dict(),
        'mean_daily_rainfall_scaled': df_period['rainfall_scaled'].mean(),
        'mean_daily_snowfall_scaled': df_period['snowfall_scaled'].mean(),
        'mean_daily_precip_scaled': df_period['total_precip_scaled'].mean(),
        'mean_daily_rainfall_nonglacier': df_period['rainfall'].mean(),
        'mean_daily_snowfall_nonglacier': df_period['snowfall'].mean(),
        'mean_daily_precip_nonglacier': df_period['total_precip_nonglacier'].mean(),
        'rainfall_fraction': df_period['rainfall_scaled'].mean() / df_period['total_precip_scaled'].mean(),
        'snowfall_fraction': df_period['snowfall_scaled'].mean() / df_period['total_precip_scaled'].mean(),
        'n_days': len(df_period),
        'save_path': save_path
    }

#--------------------------------------------------------------------------------

def plot_actual_evapotranspiration(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot monthly actual evapotranspiration (AET) for non-glacier area.
    Creates a bar chart showing monthly average AET.
    Scales AET from non-glacier area to total catchment area for comparison with water balance.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for analysis period
    validation_end : str, optional
        End date for analysis period
    """
    gauge_id = config['gauge_id']
    config_dir = Path(config['main_dir']) / config['config_dir']
    
    # Use dates from namelist if not provided
    if validation_start is None:
        validation_start = config.get('start_date', '2000-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"\nCreating actual evapotranspiration plot for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    
    # Load HRU shapefile to get areas
    topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
    hru_shapefile = topo_dir / "HRU.shp"
    
    print(f"  - Loading HRU shapefile: {hru_shapefile}")
    
    if not hru_shapefile.exists():
        print(f"ERROR: HRU shapefile not found: {hru_shapefile}")
        return None
    
    try:
        import geopandas as gpd
        hru_gdf = gpd.read_file(hru_shapefile)
        original_count = len(hru_gdf)
        print(f"  - Loaded {original_count} HRUs from shapefile")
        
        # Calculate total catchment area
        total_area_km2 = hru_gdf['Area_km2'].sum()
        
        # Calculate glacier and non-glacier areas
        # Glacier HRUs have Landuse_Cl values of 7 or 8
        if 'Landuse_Cl' in hru_gdf.columns:
            glacier_area_km2 = hru_gdf[hru_gdf['Landuse_Cl'].isin([7, 8])]['Area_km2'].sum()
            non_glacier_area_km2 = hru_gdf[~hru_gdf['Landuse_Cl'].isin([7, 8])]['Area_km2'].sum()
            glacier_hru_count = len(hru_gdf[hru_gdf['Landuse_Cl'].isin([7, 8])])
            non_glacier_hru_count = len(hru_gdf[~hru_gdf['Landuse_Cl'].isin([7, 8])])
            print(f"  - Glacier HRUs (Landuse_Cl 7, 8): {glacier_hru_count}")
            print(f"  - Non-glacier HRUs: {non_glacier_hru_count}")
        else:
            print("  - Warning: 'Landuse_Cl' column not found, cannot identify glacier areas")
            print("  - Assuming all area is non-glacier")
            non_glacier_area_km2 = total_area_km2
            glacier_area_km2 = 0.0
        
        area_fraction = non_glacier_area_km2 / total_area_km2 if total_area_km2 > 0 else 1.0
        
        print(f"  - Total catchment area: {total_area_km2:.2f} km²")
        print(f"  - Non-glacier area: {non_glacier_area_km2:.2f} km² ({area_fraction*100:.1f}%)")
        print(f"  - Glacier area: {glacier_area_km2:.2f} km² ({(1-area_fraction)*100:.1f}%)")
        
    except Exception as e:
        print(f"ERROR: Could not load HRU shapefile: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Load AET data
    aet_file = config_dir / f"catchment_{gauge_id}" / config['model_type'] / "output" / f"{gauge_id}_{config['model_type']}_AET_Daily_Average_ByHRUGroup.csv"
    
    print(f"  - Loading AET data: {aet_file}")
    
    if not aet_file.exists():
        print(f"ERROR: AET file not found: {aet_file}")
        return None
    
    try:
        # Read the CSV file, skipping the second row (row index 1)
        df = pd.read_csv(aet_file, skiprows=[1])
        
        # Rename the date column - it's called 'HRUGroup:' in the file
        if 'HRUGroup:' in df.columns:
            df = df.rename(columns={'HRUGroup:': 'date'})
        else:
            print(f"  - Warning: 'HRUGroup:' column not found")
            print(f"  - Available columns: {df.columns.tolist()}")
            return None
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Check if NO_GLACIER column exists
        if 'NO_GLACIER' not in df.columns:
            print(f"  - Warning: NO_GLACIER column not found in AET file")
            print(f"  - Available columns: {df.columns.tolist()}")
            return None
        
        print(f"  - Found {len(df)} days of data")
        print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Rename NO_GLACIER to aet for clarity
        df = df.rename(columns={'NO_GLACIER': 'aet'})
        
    except Exception as e:
        print(f"ERROR: Could not load AET data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Filter for analysis period
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_period = df[mask].copy()
    
    if len(df_period) == 0:
        print(f"ERROR: No data found for period {validation_start} to {validation_end}")
        return None
    
    print(f"  - Analysis period has {len(df_period)} days")
    
    # Scale to catchment area (multiply by area fraction to get catchment-wide values)
    df_period['aet_scaled'] = df_period['aet'] * area_fraction
    
    # Extract month
    df_period['month'] = df_period['date'].dt.month
    
    # Calculate monthly averages for SCALED values (mean of daily values)
    monthly_aet = df_period.groupby('month')['aet_scaled'].mean()
    
    # Also keep non-scaled for reference
    monthly_aet_nonglacier = df_period.groupby('month')['aet'].mean()
    
    print(f"\n  NON-GLACIER AREA VALUES (original):")
    print(f"  - Mean daily AET: {df_period['aet'].mean():.2f} mm/day")
    
    print(f"\n  SCALED TO CATCHMENT AREA:")
    print(f"  - Mean daily AET: {df_period['aet_scaled'].mean():.2f} mm/day")
    print(f"  - Scaling factor: {area_fraction:.4f} (non-glacier fraction)")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Month labels
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create bar chart
    bars = ax.bar(months, monthly_aet, 
                  color='forestgreen', label='Actual Evapotranspiration', 
                  edgecolor='darkgreen', linewidth=0.5)
    
    # Formatting
    ax.set_xlabel('Month', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Daily AET (mm/day)', fontsize=14, fontweight='bold')
    ax.set_title(f'Monthly Mean Actual Evapotranspiration (Scaled to Catchment Area)\nNon-Glacier Area × {area_fraction:.2f} - Catchment {gauge_id}\n{validation_start} to {validation_end}',
                fontsize=16, fontweight='bold')
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text box
    stats_text = (f"Catchment Statistics:\n"
                 f"Total Area: {total_area_km2:.2f} km²\n"
                 f"Non-Glacier: {non_glacier_area_km2:.2f} km² ({area_fraction*100:.1f}%)\n"
                 f"Glacier: {glacier_area_km2:.2f} km² ({(1-area_fraction)*100:.1f}%)\n"
                 f"\n"
                 f"Scaled AET:\n"
                 f"Mean Daily: {df_period['aet_scaled'].mean():.2f} mm/day\n"
                 f"Annual Total: {df_period['aet_scaled'].mean() * 365:.1f} mm/year\n"
                 f"Days: {len(df_period)}")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['forcing'] / f'actual_evapotranspiration_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved actual evapotranspiration plot to: {save_path}")
    plt.show()
    
    # Print detailed monthly summary
    print(f"\n{'='*100}")
    print(f"MONTHLY MEAN ACTUAL EVAPOTRANSPIRATION - CATCHMENT {gauge_id}")
    print(f"{'='*100}")
    print(f"Period: {validation_start} to {validation_end}")
    print(f"Scaled to Catchment Area (Non-Glacier Fraction: {area_fraction:.4f})")
    print(f"Total Catchment Area: {total_area_km2:.2f} km² | Non-Glacier: {non_glacier_area_km2:.2f} km² | Glacier: {glacier_area_km2:.2f} km²")
    print(f"\n{'Month':<10} {'AET (mm/day)':<20} {'AET (mm/month)':<20}")
    print(f"{'-'*100}")
    
    for month in months:
        month_name = month_names[month-1]
        aet = monthly_aet.get(month, 0)
        # Approximate monthly total (days per month varies, but rough estimate)
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month-1]
        aet_monthly = aet * days_in_month
        
        print(f"{month_name:<10} {aet:>15.2f} {aet_monthly:>18.1f}")
    
    print(f"{'-'*100}")
    print(f"{'ANNUAL AVG':<10} {monthly_aet.mean():>15.2f} {monthly_aet.mean() * 365:>18.1f} mm/year")
    print(f"{'='*100}\n")
    
    # Return summary statistics
    return {
        'catchment_info': {
            'total_area_km2': total_area_km2,
            'non_glacier_area_km2': non_glacier_area_km2,
            'glacier_area_km2': glacier_area_km2,
            'area_fraction': area_fraction
        },
        'monthly_aet_scaled': monthly_aet.to_dict(),
        'monthly_aet_nonglacier': monthly_aet_nonglacier.to_dict(),
        'mean_daily_aet_scaled': df_period['aet_scaled'].mean(),
        'mean_daily_aet_nonglacier': df_period['aet'].mean(),
        'annual_total_aet_scaled': df_period['aet_scaled'].mean() * 365,
        'annual_total_aet_nonglacier': df_period['aet'].mean() * 365,
        'n_days': len(df_period),
        'save_path': save_path
    }

#--------------------------------------------------------------------------------

def plot_precipitation_and_aet_combined(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot monthly precipitation (rain + snow) and AET side by side.
    Creates grouped bar chart with precipitation (stacked rain/snow) and AET for each month.
    All values scaled to catchment area for water balance comparison.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for analysis period
    validation_end : str, optional
        End date for analysis period
    """
    gauge_id = config['gauge_id']
    config_dir = Path(config['main_dir']) / config['config_dir']
    
    # Use dates from namelist if not provided
    if validation_start is None:
        validation_start = config.get('start_date', '2000-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"\nCreating combined precipitation and AET plot for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    
    # Load HRU shapefile to get areas
    topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
    hru_shapefile = topo_dir / "HRU.shp"
    
    print(f"  - Loading HRU shapefile: {hru_shapefile}")
    
    if not hru_shapefile.exists():
        print(f"ERROR: HRU shapefile not found: {hru_shapefile}")
        return None
    
    try:
        import geopandas as gpd
        hru_gdf = gpd.read_file(hru_shapefile)
        original_count = len(hru_gdf)
        print(f"  - Loaded {original_count} HRUs from shapefile")
        
        # Calculate total catchment area
        total_area_km2 = hru_gdf['Area_km2'].sum()
        
        # Calculate glacier and non-glacier areas
        if 'Landuse_Cl' in hru_gdf.columns:
            glacier_area_km2 = hru_gdf[hru_gdf['Landuse_Cl'].isin([7, 8])]['Area_km2'].sum()
            non_glacier_area_km2 = hru_gdf[~hru_gdf['Landuse_Cl'].isin([7, 8])]['Area_km2'].sum()
            glacier_hru_count = len(hru_gdf[hru_gdf['Landuse_Cl'].isin([7, 8])])
            non_glacier_hru_count = len(hru_gdf[~hru_gdf['Landuse_Cl'].isin([7, 8])])
            print(f"  - Glacier HRUs (Landuse_Cl 7, 8): {glacier_hru_count}")
            print(f"  - Non-glacier HRUs: {non_glacier_hru_count}")
        else:
            print("  - Warning: 'Landuse_Cl' column not found, cannot identify glacier areas")
            print("  - Assuming all area is non-glacier")
            non_glacier_area_km2 = total_area_km2
            glacier_area_km2 = 0.0
        
        area_fraction = non_glacier_area_km2 / total_area_km2 if total_area_km2 > 0 else 1.0
        
        print(f"  - Total catchment area: {total_area_km2:.2f} km²")
        print(f"  - Non-glacier area: {non_glacier_area_km2:.2f} km² ({area_fraction*100:.1f}%)")
        print(f"  - Glacier area: {glacier_area_km2:.2f} km² ({(1-area_fraction)*100:.1f}%)")
        
    except Exception as e:
        print(f"ERROR: Could not load HRU shapefile: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Load snowfall data
    snowfall_df = load_forcing_by_hrugroup(config, 'SNOWFALL')
    if snowfall_df is None:
        print("ERROR: Could not load snowfall data")
        return None
    
    # Load rainfall data
    rainfall_df = load_forcing_by_hrugroup(config, 'RAINFALL')
    if rainfall_df is None:
        print("ERROR: Could not load rainfall data")
        return None
    
    # Load AET data
    aet_file = config_dir / f"catchment_{gauge_id}" / config['model_type'] / "output" / f"{gauge_id}_{config['model_type']}_AET_Daily_Average_ByHRUGroup.csv"
    
    print(f"  - Loading AET data: {aet_file}")
    
    if not aet_file.exists():
        print(f"ERROR: AET file not found: {aet_file}")
        return None
    
    try:
        # Read the CSV file, skipping the second row
        aet_df = pd.read_csv(aet_file, skiprows=[1])
        
        # Rename the date column
        if 'HRUGroup:' in aet_df.columns:
            aet_df = aet_df.rename(columns={'HRUGroup:': 'date'})
        else:
            print(f"  - Warning: 'HRUGroup:' column not found in AET file")
            return None
        
        # Convert date column to datetime
        aet_df['date'] = pd.to_datetime(aet_df['date'])
        
        # Check if NO_GLACIER column exists
        if 'NO_GLACIER' not in aet_df.columns:
            print(f"  - Warning: NO_GLACIER column not found in AET file")
            return None
        
        aet_df = aet_df.rename(columns={'NO_GLACIER': 'aet'})
        print(f"  - Loaded AET data: {len(aet_df)} days")
        
    except Exception as e:
        print(f"ERROR: Could not load AET data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Merge all datasets
    df = pd.merge(
        snowfall_df.rename(columns={'NO_GLACIER': 'snowfall'}),
        rainfall_df.rename(columns={'NO_GLACIER': 'rainfall'}),
        on='date',
        how='inner'
    )
    df = pd.merge(df, aet_df[['date', 'aet']], on='date', how='inner')
    
    print(f"\n  - Merged dataset has {len(df)} days")
    
    # Filter for analysis period
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_period = df[mask].copy()
    
    if len(df_period) == 0:
        print(f"ERROR: No data found for period {validation_start} to {validation_end}")
        return None
    
    print(f"  - Analysis period has {len(df_period)} days")
    
    # Calculate total precipitation and scale all values to catchment area
    df_period['total_precip'] = df_period['snowfall'] + df_period['rainfall']
    df_period['snowfall_scaled'] = df_period['snowfall'] * area_fraction
    df_period['rainfall_scaled'] = df_period['rainfall'] * area_fraction
    df_period['total_precip_scaled'] = df_period['total_precip'] * area_fraction
    df_period['aet_scaled'] = df_period['aet'] * area_fraction
    
    # Extract month
    df_period['month'] = df_period['date'].dt.month
    
    # Calculate monthly averages (mean of daily values)
    monthly_snowfall = df_period.groupby('month')['snowfall_scaled'].mean()
    monthly_rainfall = df_period.groupby('month')['rainfall_scaled'].mean()
    monthly_total_precip = df_period.groupby('month')['total_precip_scaled'].mean()
    monthly_aet = df_period.groupby('month')['aet_scaled'].mean()
    
    # Calculate percentages for precipitation partitioning
    snow_fraction = (monthly_snowfall / monthly_total_precip * 100).fillna(0)
    rain_fraction = (monthly_rainfall / monthly_total_precip * 100).fillna(0)
    
    print(f"\n  SCALED TO CATCHMENT AREA:")
    print(f"  - Mean daily precipitation: {df_period['total_precip_scaled'].mean():.2f} mm/day")
    print(f"  - Mean daily AET: {df_period['aet_scaled'].mean():.2f} mm/day")
    print(f"  - P-AET balance: {df_period['total_precip_scaled'].mean() - df_period['aet_scaled'].mean():.2f} mm/day")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Month labels
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Bar width and positions
    bar_width = 0.35
    x_pos = np.arange(len(months))
    
    # Create precipitation bars (stacked: rainfall bottom, snowfall top)
    bars_rain = ax.bar(x_pos - bar_width/2, monthly_rainfall, bar_width,
                       color='steelblue', label='Rainfall', 
                       edgecolor='navy', linewidth=0.8)
    
    bars_snow = ax.bar(x_pos - bar_width/2, monthly_snowfall, bar_width,
                       bottom=monthly_rainfall,
                       color='lightcyan', label='Snowfall',
                       edgecolor='darkblue', linewidth=0.8)
    
    # Create AET bars
    bars_aet = ax.bar(x_pos + bar_width/2, monthly_aet, bar_width,
                      color='forestgreen', label='AET',
                      edgecolor='darkgreen', linewidth=0.8)
    
    # Formatting
    ax.set_xlabel('Month', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Daily Water Flux (mm/day)', fontsize=14, fontweight='bold')
    ax.set_title(f'Monthly Mean Precipitation and Evapotranspiration (Scaled to Catchment Area)\nNon-Glacier Area × {area_fraction:.2f} - Catchment {gauge_id}\n{validation_start} to {validation_end}',
                fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(month_names)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text box
    stats_text = (f"Catchment Statistics:\n"
                 f"Total Area: {total_area_km2:.2f} km²\n"
                 f"Non-Glacier: {non_glacier_area_km2:.2f} km² ({area_fraction*100:.1f}%)\n"
                 f"Glacier: {glacier_area_km2:.2f} km² ({(1-area_fraction)*100:.1f}%)\n"
                 f"\n"
                 f"Mean Daily (Scaled):\n"
                 f"Precipitation: {df_period['total_precip_scaled'].mean():.2f} mm/day\n"
                 f"  Rainfall: {df_period['rainfall_scaled'].mean():.2f} mm/day ({df_period['rainfall_scaled'].mean()/df_period['total_precip_scaled'].mean()*100:.1f}%)\n"
                 f"  Snowfall: {df_period['snowfall_scaled'].mean():.2f} mm/day ({df_period['snowfall_scaled'].mean()/df_period['total_precip_scaled'].mean()*100:.1f}%)\n"
                 f"AET: {df_period['aet_scaled'].mean():.2f} mm/day\n"
                 f"P - AET: {df_period['total_precip_scaled'].mean() - df_period['aet_scaled'].mean():.2f} mm/day\n"
                 f"Days: {len(df_period)}")
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['forcing'] / f'precipitation_aet_combined_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined precipitation and AET plot to: {save_path}")
    plt.show()
    
    # Print detailed monthly summary
    print(f"\n{'='*120}")
    print(f"MONTHLY MEAN PRECIPITATION AND AET - CATCHMENT {gauge_id}")
    print(f"{'='*120}")
    print(f"Period: {validation_start} to {validation_end}")
    print(f"Scaled to Catchment Area (Non-Glacier Fraction: {area_fraction:.4f})")
    print(f"Total Catchment Area: {total_area_km2:.2f} km² | Non-Glacier: {non_glacier_area_km2:.2f} km² | Glacier: {glacier_area_km2:.2f} km²")
    print(f"\n{'Month':<10} {'Rainfall':<15} {'Snowfall':<15} {'Total P':<15} {'AET':<15} {'P-AET':<15} {'Rain %':<10} {'Snow %':<10}")
    print(f"{'-'*120}")
    
    for month in months:
        month_name = month_names[month-1]
        rain = monthly_rainfall.get(month, 0)
        snow = monthly_snowfall.get(month, 0)
        precip = monthly_total_precip.get(month, 0)
        aet = monthly_aet.get(month, 0)
        balance = precip - aet
        rain_pct = rain_fraction.get(month, 0)
        snow_pct = snow_fraction.get(month, 0)
        
        print(f"{month_name:<10} {rain:>10.2f} mm/d {snow:>10.2f} mm/d {precip:>10.2f} mm/d {aet:>10.2f} mm/d {balance:>10.2f} mm/d {rain_pct:>8.1f}% {snow_pct:>8.1f}%")
    
    print(f"{'-'*120}")
    avg_rain = monthly_rainfall.mean()
    avg_snow = monthly_snowfall.mean()
    avg_precip = monthly_total_precip.mean()
    avg_aet = monthly_aet.mean()
    avg_balance = avg_precip - avg_aet
    print(f"{'ANNUAL AVG':<10} {avg_rain:>10.2f} mm/d {avg_snow:>10.2f} mm/d {avg_precip:>10.2f} mm/d {avg_aet:>10.2f} mm/d {avg_balance:>10.2f} mm/d {avg_rain/avg_precip*100:>8.1f}% {avg_snow/avg_precip*100:>8.1f}%")
    print(f"{'='*120}\n")
    
    # Return summary statistics
    return {
        'catchment_info': {
            'total_area_km2': total_area_km2,
            'non_glacier_area_km2': non_glacier_area_km2,
            'glacier_area_km2': glacier_area_km2,
            'area_fraction': area_fraction
        },
        'monthly_rainfall_scaled': monthly_rainfall.to_dict(),
        'monthly_snowfall_scaled': monthly_snowfall.to_dict(),
        'monthly_precip_scaled': monthly_total_precip.to_dict(),
        'monthly_aet_scaled': monthly_aet.to_dict(),
        'mean_daily_precip_scaled': df_period['total_precip_scaled'].mean(),
        'mean_daily_aet_scaled': df_period['aet_scaled'].mean(),
        'mean_daily_balance': df_period['total_precip_scaled'].mean() - df_period['aet_scaled'].mean(),
        'rainfall_fraction': df_period['rainfall_scaled'].mean() / df_period['total_precip_scaled'].mean(),
        'snowfall_fraction': df_period['snowfall_scaled'].mean() / df_period['total_precip_scaled'].mean(),
        'n_days': len(df_period),
        'save_path': save_path
    }

#--------------------------------------------------------------------------------

def plot_temperature_by_elevation(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot average daily temperature by elevation band.
    Creates an "average year" by taking the mean of each day-of-year across all years.
    Shows one line for each elevation band and identifies bands that never exceed 0°C.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for analysis period
    validation_end : str, optional
        End date for analysis period
    """
    gauge_id = config['gauge_id']
    config_dir = Path(config['main_dir']) / config['config_dir']
    
    # Use dates from namelist if not provided
    if validation_start is None:
        validation_start = config.get('start_date', '2000-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"\nCreating temperature by elevation band plot for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    
    # Load temperature data
    temp_file = config_dir / f"catchment_{gauge_id}" / config['model_type'] / "output" / f"{gauge_id}_{config['model_type']}_TEMP_AVE_Daily_Average_ByHRUGroup.csv"
    
    print(f"  - Loading temperature data: {temp_file}")
    
    if not temp_file.exists():
        print(f"ERROR: Temperature file not found: {temp_file}")
        return None
    
    try:
        # Read the CSV file, skipping the second row (row index 1)
        df = pd.read_csv(temp_file, skiprows=[1])
        
        # Rename the date column - it's called 'HRUGroup:' in the file
        if 'HRUGroup:' in df.columns:
            df = df.rename(columns={'HRUGroup:': 'date'})
        else:
            print(f"  - Warning: 'HRUGroup:' column not found")
            print(f"  - Available columns: {df.columns.tolist()}")
            return None
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Identify elevation band columns ONLY
        # Elevation bands have format like "2200-2300m", "3900-4000m", etc.
        # Must have: digit-digit followed by 'm'
        import re
        all_columns = df.columns.tolist()
        
        # Pattern to match elevation bands: starts with digits, has hyphen, more digits, ends with 'm'
        elevation_pattern = re.compile(r'^\d+-\d+m$', re.IGNORECASE)
        elevation_bands = [col for col in all_columns if elevation_pattern.match(col)]
        
        if len(elevation_bands) == 0:
            print(f"  - Warning: No elevation band columns found")
            print(f"  - Available columns: {all_columns}")
            print(f"  - Looking for pattern: ####-####m (e.g., 2200-2300m)")
            return None
        
        # Sort elevation bands by lower elevation value for logical ordering
        def get_lower_elevation(band_name):
            match = re.match(r'^(\d+)-\d+m$', band_name, re.IGNORECASE)
            return int(match.group(1)) if match else 0
        
        elevation_bands = sorted(elevation_bands, key=get_lower_elevation)
        
        print(f"  - Found {len(elevation_bands)} elevation bands:")
        for band in elevation_bands:
            print(f"    • {band}")
        
        print(f"  - Total records: {len(df)}")
        print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
        
    except Exception as e:
        print(f"ERROR: Could not load temperature data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Filter for analysis period
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_period = df[mask].copy()
    
    if len(df_period) == 0:
        print(f"ERROR: No data found for period {validation_start} to {validation_end}")
        return None
    
    print(f"  - Analysis period has {len(df_period)} days")
    
    # Add day of year (1-366)
    df_period['day_of_year'] = df_period['date'].dt.dayofyear
    
    # Calculate average for each day of year across all years
    print(f"\n  - Calculating average year (mean of each day-of-year)...")
    
    # Group by day_of_year and calculate mean for each elevation band
    avg_year = df_period.groupby('day_of_year')[elevation_bands].mean()
    
    # Identify bands that never exceed 0°C
    bands_never_above_zero = []
    bands_max_temps = {}
    
    for band in elevation_bands:
        max_temp = avg_year[band].max()
        bands_max_temps[band] = max_temp
        if max_temp <= 0.0:
            bands_never_above_zero.append(band)
    
    print(f"\n  - Temperature statistics by elevation band:")
    for band in elevation_bands:
        mean_temp = avg_year[band].mean()
        min_temp = avg_year[band].min()
        max_temp = avg_year[band].max()
        above_zero = "✗ NEVER ABOVE 0°C" if band in bands_never_above_zero else "✓"
        print(f"    • {band}: Mean={mean_temp:>6.2f}°C, Min={min_temp:>6.2f}°C, Max={max_temp:>6.2f}°C {above_zero}")
    
    if len(bands_never_above_zero) > 0:
        print(f"\n  - {len(bands_never_above_zero)} elevation band(s) never exceed 0°C:")
        for band in bands_never_above_zero:
            print(f"    • {band}")
    else:
        print(f"\n  - All elevation bands exceed 0°C at some point during the year")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Use a colormap to distinguish elevation bands
    # Reverse the colormap so lowest elevation (warmest) = red, highest elevation (coldest) = blue
    colors = plt.cm.coolwarm(np.linspace(1, 0, len(elevation_bands)))
    
    # Plot each elevation band
    for idx, band in enumerate(elevation_bands):
        linestyle = '--' if band in bands_never_above_zero else '-'
        linewidth = 1.5 if band in bands_never_above_zero else 2.0
        alpha = 0.7 if band in bands_never_above_zero else 1.0
        
        ax.plot(avg_year.index, avg_year[band], 
               color=colors[idx], linestyle=linestyle, linewidth=linewidth, 
               alpha=alpha, label=band)
    
    # Add 0°C reference line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label='0°C')
    
    # Formatting
    ax.set_xlabel('Day of Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=14, fontweight='bold')
    ax.set_title(f'Average Daily Temperature by Elevation Band\nCatchment {gauge_id} - {validation_start} to {validation_end}',
                fontsize=16, fontweight='bold')
    
    # Add month labels on x-axis
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_names)
    ax.set_xlim(1, 366)
    
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = (f"Elevation Bands: {len(elevation_bands)}\n"
                 f"Never above 0°C: {len(bands_never_above_zero)}\n"
                 f"Period: {validation_start} to {validation_end}\n"
                 f"Days analyzed: {len(df_period)}\n"
                 f"Years: {df_period['date'].dt.year.nunique()}")
    
    if len(bands_never_above_zero) > 0:
        stats_text += f"\n\nAlways frozen:\n"
        for band in bands_never_above_zero[:5]:  # Show first 5 to avoid crowding
            stats_text += f"{band}\n"
        if len(bands_never_above_zero) > 5:
            stats_text += f"... and {len(bands_never_above_zero) - 5} more"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['forcing'] / f'temperature_by_elevation_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved temperature by elevation plot to: {save_path}")
    plt.show()
    
    # Print detailed summary
    print(f"\n{'='*100}")
    print(f"TEMPERATURE BY ELEVATION BAND - CATCHMENT {gauge_id}")
    print(f"{'='*100}")
    print(f"Period: {validation_start} to {validation_end}")
    print(f"Number of elevation bands: {len(elevation_bands)}")
    print(f"Bands that never exceed 0°C: {len(bands_never_above_zero)}")
    print(f"\n{'Elevation Band':<20} {'Mean Temp (°C)':<20} {'Min Temp (°C)':<20} {'Max Temp (°C)':<20} {'Status':<20}")
    print(f"{'-'*100}")
    
    for band in elevation_bands:
        mean_temp = avg_year[band].mean()
        min_temp = avg_year[band].min()
        max_temp = avg_year[band].max()
        status = "NEVER ABOVE 0°C" if band in bands_never_above_zero else "Above 0°C"
        
        print(f"{band:<20} {mean_temp:>18.2f} {min_temp:>18.2f} {max_temp:>18.2f} {status:<20}")
    
    print(f"{'='*100}\n")
    
    # Return summary statistics
    return {
        'elevation_bands': elevation_bands,
        'n_bands': len(elevation_bands),
        'bands_never_above_zero': bands_never_above_zero,
        'n_bands_never_above_zero': len(bands_never_above_zero),
        'avg_year_temperatures': {band: avg_year[band].to_dict() for band in elevation_bands},
        'band_statistics': {
            band: {
                'mean': avg_year[band].mean(),
                'min': avg_year[band].min(),
                'max': avg_year[band].max(),
                'never_above_zero': band in bands_never_above_zero
            } for band in elevation_bands
        },
        'n_days': len(df_period),
        'n_years': df_period['date'].dt.year.nunique(),
        'save_path': save_path
    }



#--------------------------------------------------------------------------------
#################################### GloGEM #####################################
#--------------------------------------------------------------------------------

def load_glogem_data(config, unit='mm', plot=True):
    """
    Load catchment-averaged GloGEM data from the new preprocessing format.
    The new format has separate columns for all components (icemelt, snowmelt, rain, melt)
    with both glacier area and catchment area normalized values.
    """
    
    config_dir = Path(config['main_dir']) / config['config_dir']
    gauge_id = config['gauge_id']
    
    # Define topo_dir where GloGEM files are located
    topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
    
    # NEW: Look for the catchment-averaged file with ALL components
    catchment_avg_file = topo_dir / "GloGEM_catchment_averaged.csv"
    
    if not catchment_avg_file.exists():
        print(f"ERROR: Catchment-averaged GloGEM file not found: {catchment_avg_file}")
        print(f"Please run the GloGEM preprocessing with create_catchment_averaged_melt() first")
        return None
    
    # Get date range from config
    start_date = config.get('start_date', '2000-01-01')
    end_date = config.get('end_date', '2020-12-31')
    
    print(f"Loading catchment-averaged GloGEM data for catchment {gauge_id}:")
    print(f"  - File: {catchment_avg_file}")
    print(f"  - Period: {start_date} to {end_date}")
    
    try:
        # Load the catchment-averaged file
        glogem_df = pd.read_csv(catchment_avg_file, parse_dates=['date'])
        
        print(f"  - Loaded {len(glogem_df)} records")
        print(f"  - Columns: {glogem_df.columns.tolist()}")
        
        # Filter for date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        glogem_filtered = glogem_df[(glogem_df['date'] >= start) & (glogem_df['date'] <= end)].copy()
        
        if len(glogem_filtered) == 0:
            print(f"ERROR: No data found for period {start_date} to {end_date}")
            return None
        
        # NEW: The file now has columns for:
        # - icemelt_all, icemelt_large, icemelt_small (glacier area normalized)
        # - snowmelt_all, snowmelt_large, snowmelt_small (glacier area normalized)
        # - rain_all, rain_large, rain_small (glacier area normalized)
        # - melt_all, melt_large, melt_small (glacier area normalized - total melt = ice + snow)
        # - *_catchment versions (catchment area normalized)
        
        # Use the "all" columns (includes both large and small glaciers)
        result_df = pd.DataFrame({
            'date': glogem_filtered['date'],
            'icemelt': glogem_filtered['icemelt_all'],
            'snowmelt': glogem_filtered['snowmelt_all'],
            'rainfall': glogem_filtered['rain_all'],
            'glacier_melt': glogem_filtered['melt_all'],
            'total_output': glogem_filtered['melt_all'],
            # Also include catchment-normalized versions
            'icemelt_normalized': glogem_filtered['icemelt_all_catchment'],
            'snowmelt_normalized': glogem_filtered['snowmelt_all_catchment'],
            'rainfall_normalized': glogem_filtered['rain_all_catchment'],
            'glacier_melt_normalized': glogem_filtered['melt_all_catchment'],
            # Include large and small glacier components separately
            'icemelt_large': glogem_filtered['icemelt_large'],
            'icemelt_small': glogem_filtered['icemelt_small'],
            'snowmelt_large': glogem_filtered['snowmelt_large'],
            'snowmelt_small': glogem_filtered['snowmelt_small'],
            'rain_large': glogem_filtered['rain_large'],
            'rain_small': glogem_filtered['rain_small'],
            'melt_large': glogem_filtered['melt_large'],
            'melt_small': glogem_filtered['melt_small']
        })
        
        print(f"  ✓ Successfully loaded catchment-averaged GloGEM data")
        print(f"  - Date range: {result_df['date'].min()} to {result_df['date'].max()}")
        print(f"\n  Mean values (glacier area - all glaciers):")
        print(f"    Ice melt: {result_df['icemelt'].mean():.3f} mm/day")
        print(f"    Snow melt: {result_df['snowmelt'].mean():.3f} mm/day")
        print(f"    Rainfall: {result_df['rainfall'].mean():.3f} mm/day")
        print(f"    Total melt: {result_df['glacier_melt'].mean():.3f} mm/day")
        print(f"\n  Mean values (catchment area - all glaciers):")
        print(f"    Ice melt: {result_df['icemelt_normalized'].mean():.3f} mm/day")
        print(f"    Snow melt: {result_df['snowmelt_normalized'].mean():.3f} mm/day")
        print(f"    Rainfall: {result_df['rainfall_normalized'].mean():.3f} mm/day")
        print(f"    Total melt: {result_df['glacier_melt_normalized'].mean():.3f} mm/day")
        print(f"\n  Breakdown by glacier size (glacier area):")
        print(f"    Large glacier ice melt: {result_df['icemelt_large'].mean():.3f} mm/day")
        print(f"    Small glacier ice melt: {result_df['icemelt_small'].mean():.3f} mm/day")
        
        return result_df
        
    except Exception as e:
        print(f"ERROR: Failed to load GloGEM data: {e}")
        import traceback
        traceback.print_exc()
        return None

#--------------------------------------------------------------------------------

def plot_glogem_component_validation(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot GloGEM components showing time series and monthly regime.
    Shows ice melt, snowmelt, and rainfall from GloGEM data.
    """
    
    gauge_id = config['gauge_id']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"Plotting GloGEM components for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    
    # Load GloGEM data
    glogem_df = load_glogem_data(config, unit='mm', plot=False)
    
    if glogem_df is None:
        print("ERROR: Could not load GloGEM data")
        return None
    
    # Filter for validation period
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    
    mask = (glogem_df['date'] >= start_date) & (glogem_df['date'] <= end_date)
    glogem_filtered = glogem_df[mask].copy()
    
    if len(glogem_filtered) == 0:
        print(f"ERROR: No data found for period {validation_start} to {validation_end}")
        return None
    
    print(f"  - Loaded {len(glogem_filtered)} records")
    
    # Print statistics
    print(f"\n  Component Statistics (catchment-normalized, mm/day):")
    print(f"    Ice melt mean: {glogem_filtered['icemelt_normalized'].mean():.6f}")
    print(f"    Snowmelt mean: {glogem_filtered['snowmelt_normalized'].mean():.6f}")
    print(f"    Rainfall mean: {glogem_filtered['rainfall_normalized'].mean():.6f}")
    print(f"    Total mean: {glogem_filtered['glacier_melt_normalized'].mean():.6f}")
    
    # Calculate monthly regimes
    glogem_filtered['month'] = glogem_filtered['date'].dt.month
    
    ice_regime = glogem_filtered.groupby('month')['icemelt_normalized'].mean()
    snow_regime = glogem_filtered.groupby('month')['snowmelt_normalized'].mean()
    rain_regime = glogem_filtered.groupby('month')['rainfall_normalized'].mean()
    total_regime = glogem_filtered.groupby('month')['glacier_melt_normalized'].mean()
    
    # Create plots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # =============================
    # PLOT 1: TIME SERIES OF COMPONENTS
    # =============================
    
    ax1.plot(glogem_filtered['date'], glogem_filtered['icemelt_normalized'], 
            'grey', linewidth=2, label='Ice Melt', alpha=0.8)
    ax1.plot(glogem_filtered['date'], glogem_filtered['snowmelt_normalized'], 
            'lightblue', linewidth=2, label='Snowmelt', alpha=0.8)
    ax1.plot(glogem_filtered['date'], glogem_filtered['rainfall_normalized'], 
            'darkblue', linewidth=2, label='Rainfall', alpha=0.8)
    ax1.plot(glogem_filtered['date'], glogem_filtered['glacier_melt_normalized'], 
            'red', linewidth=2.5, label='Total', alpha=0.7, linestyle='--')
    
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('GloGEM Output (mm/day)', fontsize=12, fontweight='bold')
    ax1.set_title('Time Series: GloGEM Components (Catchment-Normalized)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Add statistics text
    stats_text = (f"Period Mean (mm/day):\n"
                 f"Ice Melt: {glogem_filtered['icemelt_normalized'].mean():.4f}\n"
                 f"Snowmelt: {glogem_filtered['snowmelt_normalized'].mean():.4f}\n"
                 f"Rainfall: {glogem_filtered['rainfall_normalized'].mean():.4f}\n"
                 f"Total: {glogem_filtered['glacier_melt_normalized'].mean():.4f}")
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    # =============================
    # PLOT 2: MONTHLY REGIME
    # =============================
    
    # Plot stacked components
    ax2.fill_between(months, 0, ice_regime.values, 
                    label='Ice Melt', color='grey', alpha=0.7)
    ax2.fill_between(months, ice_regime.values, 
                    ice_regime.values + snow_regime.values, 
                    label='Snowmelt', color='lightblue', alpha=0.7)
    ax2.fill_between(months, ice_regime.values + snow_regime.values, 
                    ice_regime.values + snow_regime.values + rain_regime.values, 
                    label='Rainfall', color='darkblue', alpha=0.7)
    
    # Plot total as line
    ax2.plot(months, total_regime.values, 'r-', 
            linewidth=3, label='Total', marker='o', markersize=8)
    
    ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax2.set_ylabel('GloGEM Output (mm/day)', fontsize=12, fontweight='bold')
    ax2.set_title('Monthly Regime: GloGEM Components (Catchment-Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xticks(months)
    ax2.set_xticklabels(month_names, fontsize=11)
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    fig.suptitle(f'GloGEM Components - Catchment {gauge_id}\n'
                f'Period: {validation_start} to {validation_end}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'glogem_components_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved GloGEM components plot to: {save_path}")
    plt.show()
    
    # =============================
    # PRINT DETAILED SUMMARY
    # =============================
    
    print(f"\n{'='*80}")
    print(f"GLOGEM COMPONENTS SUMMARY")
    print(f"{'='*80}")
    print(f"Catchment: {gauge_id}")
    print(f"Period: {validation_start} to {validation_end}")
    print(f"Records: {len(glogem_filtered)}")
    
    print(f"\nMonthly Regime (mm/day):")
    print(f"{'Month':<6} {'Ice Melt':<12} {'Snowmelt':<12} {'Rainfall':<12} {'Total':<12}")
    print(f"{'-'*80}")
    
    for month, ice, snow, rain, total in zip(
        month_names, 
        ice_regime.values, 
        snow_regime.values, 
        rain_regime.values,
        total_regime.values
    ):
        print(f"{month:<6} {ice:>10.4f} {snow:>10.4f} {rain:>10.4f} {total:>10.4f}")
    
    print(f"\n{'Annual Mean':<6} {ice_regime.mean():>10.4f} {snow_regime.mean():>10.4f} {rain_regime.mean():>10.4f} {total_regime.mean():>10.4f}")
    print(f"{'='*80}")
    
    return {
        'glogem_filtered': glogem_filtered,
        'monthly_regimes': {
            'ice': ice_regime,
            'snow': snow_regime,
            'rain': rain_regime,
            'total': total_regime
        },
        'statistics': {
            'mean_ice': glogem_filtered['icemelt_normalized'].mean(),
            'mean_snow': glogem_filtered['snowmelt_normalized'].mean(),
            'mean_rain': glogem_filtered['rainfall_normalized'].mean(),
            'mean_total': glogem_filtered['glacier_melt_normalized'].mean()
        },
        'save_path': save_path
    }

#--------------------------------------------------------------------------------

def plot_glogem_regime(config, plot_dirs, unit='mm'):
    """
    Plot GloGEM monthly regime for the catchment.
    Updated to work with new catchment-averaged format with ALL components.
    """
    
    # Load GloGEM data
    glogem_df = load_glogem_data(config, unit=unit, plot=False)
    if glogem_df is None:
        print("No GloGEM data available for plotting")
        return None
    
    # Add time columns
    glogem_df['year'] = glogem_df['date'].dt.year
    glogem_df['month'] = glogem_df['date'].dt.month
    glogem_df['day_of_year'] = glogem_df['date'].dt.dayofyear
    
    # Calculate monthly regime for all components
    monthly_regime = glogem_df.groupby('month').agg({
        'icemelt': 'mean',
        'snowmelt': 'mean',
        'rainfall': 'mean',
        'glacier_melt': 'mean',
        'icemelt_normalized': 'mean',
        'snowmelt_normalized': 'mean',
        'rainfall_normalized': 'mean',
        'glacier_melt_normalized': 'mean'
    }).reset_index()
    
    # Calculate daily regime (averaged over all years)
    daily_regime = glogem_df.groupby('day_of_year').agg({
        'icemelt': 'mean',
        'snowmelt': 'mean',
        'rainfall': 'mean',
        'glacier_melt': 'mean',
        'icemelt_normalized': 'mean',
        'snowmelt_normalized': 'mean',
        'rainfall_normalized': 'mean',
        'glacier_melt_normalized': 'mean'
    }).reset_index()
    
    # Create date series for daily regime (using non-leap year)
    daily_regime['date'] = pd.to_datetime('2001-01-01') + pd.to_timedelta(daily_regime['day_of_year'] - 1, unit='days')
    
    # Create subplots - 2x2 grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Unit label
    unit_label = 'mm/day' if unit == 'mm' else 'm³/day'
    
    # PLOT 1: Monthly regime - Glacier area
    ax1.plot(monthly_regime['month'], monthly_regime['rainfall'], 'g-', 
             label='Rainfall', linewidth=2, marker='o')
    ax1.plot(monthly_regime['month'], monthly_regime['snowmelt'], 'b-', 
             label='Snow Melt', linewidth=2, marker='^')
    ax1.plot(monthly_regime['month'], monthly_regime['icemelt'], 'r-', 
             label='Ice Melt', linewidth=2, marker='s')
    ax1.plot(monthly_regime['month'], monthly_regime['glacier_melt'], 'darkred', 
             label='Total Glacier Melt', linewidth=3, marker='D', linestyle='--')
    
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel(f'Contribution ({unit_label})', fontsize=12)
    ax1.set_title(f'Monthly GloGEM Regime (Glacier Area) - Catchment {config["gauge_id"]}', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax1.legend(loc='best')
    
    # PLOT 2: Monthly regime - Catchment normalized
    ax2.plot(monthly_regime['month'], monthly_regime['rainfall_normalized'], 'g--', 
             label='Rainfall', linewidth=2, marker='o')
    ax2.plot(monthly_regime['month'], monthly_regime['snowmelt_normalized'], 'b--', 
             label='Snow Melt', linewidth=2, marker='^')
    ax2.plot(monthly_regime['month'], monthly_regime['icemelt_normalized'], 'r--', 
             label='Ice Melt', linewidth=2, marker='s')
    ax2.plot(monthly_regime['month'], monthly_regime['glacier_melt_normalized'], 'darkred', 
             label='Total Glacier Melt', linewidth=3, marker='D', linestyle=':')
    
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel(f'Contribution ({unit_label})', fontsize=12)
    ax2.set_title('Monthly GloGEM Regime (Catchment Normalized)', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax2.legend(loc='best')
    
    # PLOT 3: Daily regime - Glacier area
    ax3.plot(daily_regime['date'], daily_regime['rainfall'], 'g-', 
             label='Rainfall', linewidth=1.5)
    ax3.plot(daily_regime['date'], daily_regime['snowmelt'], 'b-', 
             label='Snow Melt', linewidth=1.5)
    ax3.plot(daily_regime['date'], daily_regime['icemelt'], 'r-', 
             label='Ice Melt', linewidth=1.5)
    ax3.plot(daily_regime['date'], daily_regime['glacier_melt'], 'darkred', 
             label='Total Glacier Melt', linewidth=2, linestyle='--')
    
    # Format x-axis
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    ax3.set_xlabel('Month', fontsize=12)
    ax3.set_ylabel(f'Contribution ({unit_label})', fontsize=12)
    ax3.set_title('Daily Average GloGEM (Glacier Area)', fontsize=14, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='best')
    
    # PLOT 4: Daily regime - Catchment normalized
    ax4.plot(daily_regime['date'], daily_regime['rainfall_normalized'], 'g--', 
             label='Rainfall', linewidth=1.5)
    ax4.plot(daily_regime['date'], daily_regime['snowmelt_normalized'], 'b--', 
             label='Snow Melt', linewidth=1.5)
    ax4.plot(daily_regime['date'], daily_regime['icemelt_normalized'], 'r--', 
             label='Ice Melt', linewidth=1.5)
    ax4.plot(daily_regime['date'], daily_regime['glacier_melt_normalized'], 'darkred', 
             label='Total Glacier Melt', linewidth=2, linestyle=':')
    
    # Format x-axis
    ax4.xaxis.set_major_locator(mdates.MonthLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    ax4.set_xlabel('Month', fontsize=12)
    ax4.set_ylabel(f'Contribution ({unit_label})', fontsize=12)
    ax4.set_title('Daily Average GloGEM (Catchment Normalized)', fontsize=14, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(loc='best')
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'glogem_regime_{config["gauge_id"]}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved GloGEM regime plot to: {save_path}")
    plt.show()
    
    # Print summary statistics
    total_annual = glogem_df.groupby('year')[['icemelt', 'snowmelt', 'rainfall', 'glacier_melt', 
                                              'icemelt_normalized', 'snowmelt_normalized', 
                                              'rainfall_normalized', 'glacier_melt_normalized']].sum().mean()
    
    print(f"\nGloGEM Analysis Summary for Catchment {config['gauge_id']}:")
    print(f"  Period: {glogem_df['date'].min().date()} to {glogem_df['date'].max().date()}")
    print(f"\n  Annual averages (glacier area):")
    print(f"    Rainfall: {total_annual['rainfall']:.1f} mm/year")
    print(f"    Snow melt: {total_annual['snowmelt']:.1f} mm/year")
    print(f"    Ice melt: {total_annual['icemelt']:.1f} mm/year")
    print(f"    Total glacier melt: {total_annual['glacier_melt']:.1f} mm/year")
    print(f"\n  Annual averages (catchment normalized):")
    print(f"    Rainfall: {total_annual['rainfall_normalized']:.1f} mm/year")
    print(f"    Snow melt: {total_annual['snowmelt_normalized']:.1f} mm/year")
    print(f"    Ice melt: {total_annual['icemelt_normalized']:.1f} mm/year")
    print(f"    Total glacier melt: {total_annual['glacier_melt_normalized']:.1f} mm/year")
    
    return fig

#--------------------------------------------------------------------------------

def plot_glogem_vs_observed_regime(config, plot_dirs, start_date=None, end_date=None, min_data_threshold=0.95):
    """
    Plot GloGEM catchment average melt regime vs observed runoff regime.
    This allows comparison to see if GloGEM data makes sense.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing paths and settings
    plot_dirs : dict
        Dictionary of plot directories
    start_date : str, optional
        Start date for analysis (format: 'YYYY-MM-DD'). If None, uses config['start_date']
    end_date : str, optional
        End date for analysis (format: 'YYYY-MM-DD'). If None, uses config['end_date']
    min_data_threshold : float, optional
        Minimum fraction of valid data required per year (default: 0.95 = 95%)
        Years with less data availability are excluded from regime calculation
        
    Returns:
    --------
    dict : Summary statistics and plot path
    """
    
    gauge_id = config['gauge_id']
    
    # Set date range
    if start_date is None:
        start_date = config.get('start_date', '2000-01-01')
    if end_date is None:
        end_date = config.get('end_date', '2020-12-31')
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    print(f"\n{'='*100}")
    print(f"PLOTTING GLOGEM VS OBSERVED RUNOFF REGIME - CATCHMENT {gauge_id}")
    print(f"{'='*100}\n")
    print(f"Analysis period: {start_date} to {end_date}")
    print(f"Minimum data threshold: {min_data_threshold*100:.0f}% per year\n")
    
    # Load GloGEM data (catchment-normalized, in mm/day)
    print("Loading GloGEM data...")
    glogem_df = load_glogem_data(config, unit='mm', plot=False)
    if glogem_df is None:
        print("ERROR: Could not load GloGEM data")
        return None
    
    # Filter GloGEM data by date range
    glogem_df = glogem_df[(glogem_df['date'] >= start) & (glogem_df['date'] <= end)].copy()
    print(f"  - GloGEM data: {len(glogem_df)} records in period")
    
    # Load observed runoff
    print("\nLoading observed runoff data...")
    hydro_df = load_hydrograph_data(config)
    if hydro_df is None or 'obs_Q' not in hydro_df.columns:
        print("ERROR: Could not load observed runoff data")
        return None
    
    # Filter observed data by date range
    hydro_df = hydro_df[(hydro_df['date'] >= start) & (hydro_df['date'] <= end)].copy()
    print(f"  - Observed data: {len(hydro_df)} records in period")
    
    # Filter observed data by date range
    hydro_df = hydro_df[(hydro_df['date'] >= start) & (hydro_df['date'] <= end)].copy()
    print(f"  - Observed data: {len(hydro_df)} records in period")
    
    # Load catchment area from HRU shapefile
    print("\nLoading catchment area from HRU shapefile...")
    config_dir = Path(config['main_dir']) / config['config_dir']
    topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
    catchment_shape_file = topo_dir / "HRU.shp"
    
    try:
        if not catchment_shape_file.exists():
            print(f"ERROR: HRU shapefile not found: {catchment_shape_file}")
            return None
        
        hru_gdf = gpd.read_file(catchment_shape_file)
        catchment_area_km2 = hru_gdf['Area_km2'].sum()
        print(f"  - Catchment area: {catchment_area_km2:.2f} km²")
        
    except Exception as e:
        print(f"ERROR: Could not load catchment area from shapefile: {e}")
        return None
    
    # Convert observed runoff from m3/s to mm/day
    # Q [m3/s] * 86400 [s/day] / (Area [km2] * 1e6 [m2/km2]) * 1000 [mm/m] = mm/day
    conversion_factor = 86400.0 / (catchment_area_km2 * 1e6) * 1000.0
    hydro_df['obs_Q_mm'] = hydro_df['obs_Q'] * conversion_factor
    
    print(f"  - Conversion factor: {conversion_factor:.6f}")
    
    # Merge the two dataframes on date
    merged_df = pd.merge(glogem_df[['date', 'glacier_melt_normalized']], 
                         hydro_df[['date', 'obs_Q_mm']], 
                         on='date', 
                         how='inner')
    
    if len(merged_df) == 0:
        print("ERROR: No overlapping dates between GloGEM and observed data")
        return None
    
    print(f"\n  - Overlapping period: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"  - Number of overlapping days: {len(merged_df)}")
    
    # Add year and month columns
    merged_df['year'] = merged_df['date'].dt.year
    merged_df['month'] = merged_df['date'].dt.month
    
    # ============================================================================
    # FILTER OUT YEARS WITH INSUFFICIENT DATA
    # ============================================================================
    print(f"\n{'='*80}")
    print(f"FILTERING YEARS BY DATA AVAILABILITY (Minimum: {min_data_threshold*100:.0f}%)")
    print(f"{'='*80}")
    
    # Calculate expected days per year
    year_stats = []
    for year in sorted(merged_df['year'].unique()):
        # Determine expected days for this year (accounting for leap years and partial years)
        year_start = max(pd.Timestamp(f"{year}-01-01"), start)
        year_end = min(pd.Timestamp(f"{year}-12-31"), end)
        expected_days = (year_end - year_start).days + 1
        
        # Get data for this year
        year_data = merged_df[merged_df['year'] == year].copy()
        actual_days = len(year_data)
        
        # Count valid (non-NaN) values
        glogem_valid = year_data['glacier_melt_normalized'].notna().sum()
        obs_valid = year_data['obs_Q_mm'].notna().sum()
        
        # Calculate data availability
        glogem_availability = glogem_valid / expected_days if expected_days > 0 else 0
        obs_availability = obs_valid / expected_days if expected_days > 0 else 0
        
        # Year is valid if BOTH datasets meet the threshold
        is_valid = (glogem_availability >= min_data_threshold) and (obs_availability >= min_data_threshold)
        
        year_stats.append({
            'year': year,
            'expected_days': expected_days,
            'actual_days': actual_days,
            'glogem_valid': glogem_valid,
            'obs_valid': obs_valid,
            'glogem_availability': glogem_availability,
            'obs_availability': obs_availability,
            'is_valid': is_valid
        })
        
        status = "✓ INCLUDED" if is_valid else "✗ EXCLUDED"
        print(f"Year {year}: GloGEM={glogem_availability*100:>5.1f}%, Observed={obs_availability*100:>5.1f}% ({actual_days}/{expected_days} days) - {status}")
    
    year_stats_df = pd.DataFrame(year_stats)
    valid_years = year_stats_df[year_stats_df['is_valid']]['year'].tolist()
    excluded_years = year_stats_df[~year_stats_df['is_valid']]['year'].tolist()
    
    print(f"\n  - Valid years: {len(valid_years)} ({', '.join(map(str, valid_years))})")
    if excluded_years:
        print(f"  - Excluded years: {len(excluded_years)} ({', '.join(map(str, excluded_years))})")
    
    # Filter merged dataframe to only include valid years
    merged_df_filtered = merged_df[merged_df['year'].isin(valid_years)].copy()
    
    if len(merged_df_filtered) == 0:
        print("\nERROR: No years meet the minimum data availability threshold")
        return None
    
    print(f"  - Total days after filtering: {len(merged_df_filtered)} (from {len(valid_years)} complete years)")
    
    # Calculate monthly regime (average for each month across valid years only)
    monthly_regime = merged_df_filtered.groupby('month').agg({
        'glacier_melt_normalized': 'mean',
        'obs_Q_mm': 'mean'
    }).reset_index()
    
    # Calculate annual totals correctly (monthly averages * days per month)
    # Days per month (using average for simplicity)
    days_per_month = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    monthly_regime['days'] = monthly_regime['month'].apply(lambda m: days_per_month[m-1])
    
    # Calculate monthly totals (mm/day * days = mm/month)
    monthly_regime['glogem_monthly_total'] = monthly_regime['glacier_melt_normalized'] * monthly_regime['days']
    monthly_regime['obs_monthly_total'] = monthly_regime['obs_Q_mm'] * monthly_regime['days']
    
    # Sum to get annual totals
    glogem_total = monthly_regime['glogem_monthly_total'].sum()
    obs_total = monthly_regime['obs_monthly_total'].sum()
    correlation = merged_df_filtered['glacier_melt_normalized'].corr(merged_df_filtered['obs_Q_mm'])
    
    print(f"\n{'='*80}")
    print(f"STATISTICS (based on {len(valid_years)} complete years)")
    print(f"{'='*80}")
    print(f"  - Annual total GloGEM melt: {glogem_total:.2f} mm/year")
    print(f"  - Annual total observed runoff: {obs_total:.2f} mm/year")
    print(f"  - GloGEM / Observed ratio: {glogem_total/obs_total:.2%}")
    print(f"  - Correlation (daily): {correlation:.3f}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Plot bars
    x = monthly_regime['month']
    width = 0.35
    
    ax.bar(x - width/2, monthly_regime['glacier_melt_normalized'], width, 
           label='GloGEM Total Melt', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, monthly_regime['obs_Q_mm'], width, 
           label='Observed Runoff', color='coral', alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Water Input [mm/day]', fontsize=12, fontweight='bold')
    
    
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add statistics box
    stats_text = (
        f'Annual Total:\n'
        f'  GloGEM: {glogem_total:.1f} mm/yr\n'
        f'  Observed: {obs_total:.1f} mm/yr\n'
        f'  Ratio: {glogem_total/obs_total:.2%}\n'
        f'  Correlation: {correlation:.3f}\n'
        f'  Valid years: {len(valid_years)}'
    )
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    save_path = plot_dirs['contributions'] / f"{gauge_id}_glogem_vs_observed_regime.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved: {save_path}")
    
    # Show the plot
    plt.show()
    plt.close()
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"MONTHLY REGIME COMPARISON")
    print(f"{'='*80}")
    print(f"{'Month':<10} {'GloGEM [mm/day]':<20} {'Observed [mm/day]':<20} {'Difference [mm/day]':<20}")
    print(f"{'-'*80}")
    
    for _, row in monthly_regime.iterrows():
        month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][int(row['month'])-1]
        glogem_val = row['glacier_melt_normalized']
        obs_val = row['obs_Q_mm']
        diff = glogem_val - obs_val
        
        print(f"{month_name:<10} {glogem_val:>18.3f} {obs_val:>18.3f} {diff:>18.3f}")
    
    print(f"{'='*80}\n")
    
    # Return summary
    return {
        'monthly_regime': monthly_regime,
        'glogem_annual_total': glogem_total,
        'observed_annual_total': obs_total,
        'ratio': glogem_total / obs_total,
        'correlation': correlation,
        'overlapping_days': len(merged_df_filtered),
        'period_start': start,
        'period_end': end,
        'valid_years': valid_years,
        'excluded_years': excluded_years,
        'n_valid_years': len(valid_years),
        'n_excluded_years': len(excluded_years),
        'year_stats': year_stats_df,
        'save_path': save_path
    }

#--------------------------------------------------------------------------------

def create_irrigation_timeseries(config):
    """
    Read irrigation NetCDF file, calculate area-weighted average for each HRU,
    and save as time series in data_obs folder.
    
    The NetCDF structure has:
    - time dimension
    - x, y dimensions (spatial grid or placeholder)
    - 'data' variable where each row corresponds to an HRU (row 0 = HRU 1, row 1 = HRU 2, etc.)
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with date and area-weighted irrigation values (mm/day)
    """
    
    config_dir = Path(config['main_dir']) / config['config_dir']
    gauge_id = config['gauge_id']
    model_type = config['model_type']
    
    # Define paths
    data_obs_dir = config_dir / f"catchment_{gauge_id}" / model_type / "data_obs"
    irrigation_file = data_obs_dir / "irrigation.nc"
    
    topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
    hru_shapefile = topo_dir / "HRU.shp"
    
    print(f"Creating irrigation time series for catchment {gauge_id}:")
    print(f"  - Irrigation file: {irrigation_file}")
    print(f"  - HRU shapefile: {hru_shapefile}")
    
    # Check if files exist
    if not irrigation_file.exists():
        print(f"ERROR: Irrigation file not found: {irrigation_file}")
        return None
    
    if not hru_shapefile.exists():
        print(f"ERROR: HRU shapefile not found: {hru_shapefile}")
        return None
    
    try:
        # Load HRU shapefile to get areas
        import geopandas as gpd
        hru_gdf = gpd.read_file(hru_shapefile)
        print(f"  - Loaded {len(hru_gdf)} HRUs from shapefile")
        
        # Get HRU areas in order (HRU_ID should match row index)
        # Sort by HRU_ID to ensure correct order
        hru_gdf_sorted = hru_gdf.sort_values('HRU_ID').reset_index(drop=True)
        
        hru_areas = hru_gdf_sorted['Area_km2'].values
        total_area = hru_areas.sum()
        
        print(f"  - Total catchment area: {total_area:.2f} km²")
        print(f"  - Number of HRUs: {len(hru_areas)}")
        
    except Exception as e:
        print(f"ERROR: Could not load HRU shapefile: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    try:
        # Load irrigation NetCDF file
        import xarray as xr
        
        ds = xr.open_dataset(irrigation_file)
        print(f"  - Loaded irrigation NetCDF file")
        print(f"  - Variables: {list(ds.variables.keys())}")
        print(f"  - Dimensions: {list(ds.dims.keys())}")
        
        # Get time dimension
        if 'time' not in ds.dims:
            print(f"ERROR: No time dimension found in NetCDF file")
            return None
        
        # Get time values
        time_values = pd.to_datetime(ds['time'].values)
        n_timesteps = len(time_values)
        
        print(f"  - Time steps: {n_timesteps}")
        print(f"  - Date range: {time_values.min()} to {time_values.max()}")
        
        # Get irrigation data variable
        if 'data' not in ds.variables:
            print(f"ERROR: 'data' variable not found in NetCDF file")
            print(f"  Available variables: {list(ds.variables.keys())}")
            return None
        
        irrigation_data = ds['data']
        
        print(f"  - Irrigation data shape: {irrigation_data.shape}")
        print(f"  - Irrigation data dimensions: {irrigation_data.dims}")
        
        # Determine data structure
        # Expected: (time, x, y) where x corresponds to HRU rows
        # or (time, hru) or similar
        
        # Get the spatial dimensions
        spatial_dims = [dim for dim in irrigation_data.dims if dim != 'time']
        
        if len(spatial_dims) == 0:
            print(f"ERROR: No spatial dimensions found")
            return None
        
        print(f"  - Spatial dimensions: {spatial_dims}")
        
        # Calculate area-weighted average for each time step
        irrigation_timeseries = []
        
        for t_idx in range(n_timesteps):
            # Get irrigation values for all HRUs at this time step
            irrig_slice = irrigation_data.isel(time=t_idx)
            
            # Flatten to 1D array (handles both 1D and 2D spatial grids)
            irrig_values = irrig_slice.values.flatten()
            
            # Check if we have the right number of HRUs
            if len(irrig_values) != len(hru_areas):
                print(f"  WARNING: Mismatch in HRU count at timestep {t_idx}")
                print(f"    Irrigation data points: {len(irrig_values)}")
                print(f"    HRU areas: {len(hru_areas)}")
                
                # Truncate or pad to match
                if len(irrig_values) > len(hru_areas):
                    irrig_values = irrig_values[:len(hru_areas)]
                else:
                    # Pad with zeros
                    irrig_values = np.pad(irrig_values, 
                                         (0, len(hru_areas) - len(irrig_values)), 
                                         'constant', constant_values=0)
            
            # Calculate area-weighted average
            # Each HRU row corresponds to: row_0 = HRU_1, row_1 = HRU_2, etc.
            weighted_sum = np.sum(irrig_values * hru_areas)
            weighted_avg = weighted_sum / total_area
            
            irrigation_timeseries.append({
                'date': time_values[t_idx],
                'irrigation_mm_day': weighted_avg
            })
            
            # Print progress every 365 days
            if (t_idx + 1) % 365 == 0:
                print(f"  - Processed {t_idx + 1}/{n_timesteps} time steps...")
        
        # Create DataFrame
        irrigation_df = pd.DataFrame(irrigation_timeseries)
        
        print(f"  - Created time series with {len(irrigation_df)} records")
        print(f"  - Mean irrigation: {irrigation_df['irrigation_mm_day'].mean():.4f} mm/day")
        print(f"  - Max irrigation: {irrigation_df['irrigation_mm_day'].max():.4f} mm/day")
        print(f"  - Min irrigation: {irrigation_df['irrigation_mm_day'].min():.4f} mm/day")
        
        # Check for any issues
        zero_count = (irrigation_df['irrigation_mm_day'] == 0).sum()
        negative_count = (irrigation_df['irrigation_mm_day'] < 0).sum()
        
        print(f"  - Zero irrigation days: {zero_count}/{len(irrigation_df)}")
        if negative_count > 0:
            print(f"  - WARNING: {negative_count} days with negative irrigation!")
        
        # Save to CSV in data_obs folder
        output_file = data_obs_dir / "irrigation_timeseries.csv"
        irrigation_df.to_csv(output_file, index=False)
        print(f"  ✓ Saved irrigation time series to: {output_file}")
        
        # Close NetCDF file
        ds.close()
        
        return irrigation_df
        
    except Exception as e:
        print(f"ERROR: Failed to process irrigation NetCDF file: {e}")
        import traceback
        traceback.print_exc()
        return None

#--------------------------------------------------------------------------------

def plot_irrigation_vs_glogem_regime(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Compare irrigation regime with GloGEM glacier melt regime.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    dict
        Dictionary containing both regimes and comparison statistics
    """
    
    gauge_id = config['gauge_id']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"Comparing irrigation vs GloGEM regime for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    
    # 1. Load or create irrigation time series
    config_dir = Path(config['main_dir']) / config['config_dir']
    model_type = config['model_type']
    data_obs_dir = config_dir / f"catchment_{gauge_id}" / model_type / "data_obs"
    irrigation_ts_file = data_obs_dir / "irrigation_timeseries.csv"
    
    if not irrigation_ts_file.exists():
        print("  - Irrigation time series not found, creating it...")
        irrigation_df = create_irrigation_timeseries(config)
        if irrigation_df is None:
            print("ERROR: Could not create irrigation time series")
            return None
    else:
        print("  - Loading existing irrigation time series...")
        try:
            irrigation_df = pd.read_csv(irrigation_ts_file, parse_dates=['date'])
            print(f"  ✓ Loaded {len(irrigation_df)} irrigation records")
        except Exception as e:
            print(f"ERROR: Could not load irrigation time series: {e}")
            return None
    
    # 2. Load GloGEM data
    print("  - Loading GloGEM data...")
    glogem_df = load_glogem_data(config, unit='mm', plot=False)
    
    if glogem_df is None:
        print("ERROR: Could not load GloGEM data")
        return None
    
    # 3. Filter both datasets for validation period
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    
    # Filter irrigation
    irrig_mask = (irrigation_df['date'] >= start_date) & (irrigation_df['date'] <= end_date)
    irrigation_filtered = irrigation_df[irrig_mask].copy()
    
    if len(irrigation_filtered) == 0:
        print(f"ERROR: No irrigation data found for period {validation_start} to {validation_end}")
        return None
    
    # Filter GloGEM
    glogem_mask = (glogem_df['date'] >= start_date) & (glogem_df['date'] <= end_date)
    glogem_filtered = glogem_df[glogem_mask].copy()
    
    if len(glogem_filtered) == 0:
        print(f"ERROR: No GloGEM data found for period {validation_start} to {validation_end}")
        return None
    
    # 4. Calculate monthly regimes
    irrigation_filtered['month'] = irrigation_filtered['date'].dt.month
    glogem_filtered['month'] = glogem_filtered['date'].dt.month
    
    irrigation_regime = irrigation_filtered.groupby('month')['irrigation_mm_day'].mean()
    glogem_regime = glogem_filtered.groupby('month')['glacier_melt_normalized'].mean()
    
    print(f"  - Irrigation regime mean: {irrigation_regime.mean():.4f} mm/day")
    print(f"  - GloGEM regime mean: {glogem_regime.mean():.4f} mm/day")
    
    # 5. Create comparison plot
    plt.figure(figsize=(14, 8))
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot irrigation regime
    plt.plot(months, irrigation_regime.values, 'g-', linewidth=3, 
            label='Irrigation', marker='o', markersize=8)
    
    # Plot GloGEM glacier melt regime
    plt.plot(months, glogem_regime.values, 'C3--', linewidth=3, 
            label='GloGEM Glacier Melt', marker='s', markersize=8)
    
    # Fill area between to show difference
    plt.fill_between(months, irrigation_regime.values, glogem_regime.values, 
                     alpha=0.2, color='gray', label='Difference')
    
    # Formatting
    plt.xlabel('Month', fontsize=14, fontweight='bold')
    plt.ylabel('Water Input (mm/day)', fontsize=14, fontweight='bold')
    plt.title(f'Irrigation vs GloGEM Glacier Melt Regime\nCatchment {gauge_id}', 
             fontsize=16, fontweight='bold')
    plt.xticks(months, month_names, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    
    # Calculate and display statistics
    correlation = np.corrcoef(irrigation_regime.values, glogem_regime.values)[0, 1]
    bias = irrigation_regime.mean() - glogem_regime.mean()
    relative_bias = (bias / glogem_regime.mean()) * 100
    rmse = np.sqrt(np.mean((irrigation_regime.values - glogem_regime.values)**2))
    
    stats_text = (f"Statistics:\n"
                 f"Irrigation mean: {irrigation_regime.mean():.4f} mm/day\n"
                 f"GloGEM mean: {glogem_regime.mean():.4f} mm/day\n"
                 f"Correlation: {correlation:.3f}\n"
                 f"Bias: {bias:.4f} mm/day ({relative_bias:+.1f}%)\n"
                 f"RMSE: {rmse:.4f} mm/day")
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'irrigation_vs_glogem_regime_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {save_path}")
    plt.show()
    
    # Print detailed comparison
    print(f"\nIrrigation vs GloGEM Regime Comparison:")
    print(f"  Period: {validation_start} to {validation_end}")
    print(f"  Correlation: {correlation:.3f}")
    print(f"  Mean bias: {bias:.4f} mm/day ({relative_bias:+.1f}%)")
    print(f"  RMSE: {rmse:.4f} mm/day")
    print(f"  Irrigation peak month: {month_names[irrigation_regime.idxmax()-1]}")
    print(f"  GloGEM peak month: {month_names[glogem_regime.idxmax()-1]}")
    
    print(f"\nMonthly Comparison:")
    for month, irrig_val, glogem_val in zip(month_names, irrigation_regime.values, glogem_regime.values):
        diff = irrig_val - glogem_val
        diff_pct = (diff / glogem_val * 100) if glogem_val > 0 else 0
        print(f"  {month}: Irrigation={irrig_val:.4f}, GloGEM={glogem_val:.4f}, "
              f"Diff={diff:+.4f} ({diff_pct:+.1f}%)")
    
    # Return results
    return {
        'irrigation_regime': irrigation_regime,
        'glogem_regime': glogem_regime,
        'correlation': correlation,
        'bias': bias,
        'relative_bias_pct': relative_bias,
        'rmse': rmse,
        'irrigation_peak_month': month_names[irrigation_regime.idxmax()-1],
        'glogem_peak_month': month_names[glogem_regime.idxmax()-1],
        'validation_period': {
            'start': validation_start,
            'end': validation_end
        }
    }


#--------------------------------------------------------------------------------
################################# contributions #################################
#--------------------------------------------------------------------------------

def load_snowmelt_mass_loadings(config, validation_start=None, validation_end=None, unit='mm'):
    """
    Load snowmelt mass loadings data from Raven output.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    unit : str, optional
        Unit for output ('mm' for mm/day, 'm3' for m³/s), default is 'mm'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing date and snowmelt in specified units
    """
    
    config_dir = Path(config['main_dir']) / config['config_dir']
    gauge_id = config['gauge_id']
    model_type = config['model_type']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', config.get('start_date', '2000-01-01'))
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"Loading snowmelt mass loadings for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    print(f"  - Requested unit: {unit}")
    
    # Load catchment area for unit conversion
    conversion_m3s_to_mm_day = None
    if unit == 'mm':
        topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
        catchment_shape_file = topo_dir / "HRU.shp"
        
        try:
            if catchment_shape_file.exists():
                import geopandas as gpd
                hru_gdf = gpd.read_file(catchment_shape_file)
                total_area_km2 = hru_gdf['Area_km2'].sum()
                # Conversion factor: m³/s to mm/day
                conversion_m3s_to_mm_day = 86400 / (total_area_km2 * 1000000) * 1000
                print(f"  - Catchment area: {total_area_km2:.2f} km²")
                print(f"  - Conversion factor (m³/s to mm/day): {conversion_m3s_to_mm_day:.6f}")
            else:
                print(f"ERROR: Catchment shapefile not found: {catchment_shape_file}")
                print(f"  Falling back to m³/s")
                unit = 'm3'
        except Exception as e:
            print(f"ERROR: Could not load catchment area: {e}")
            print(f"  Falling back to m³/s")
            unit = 'm3'
    
    # Load snowmelt data file
    mass_loadings_file = config_dir / f"catchment_{gauge_id}" / model_type / "output" / f"{gauge_id}_{model_type}_SNOWMELTMassLoadings.csv"
    
    if not mass_loadings_file.exists():
        print(f"ERROR: Mass loadings file not found: {mass_loadings_file}")
        return None
    
    try:
        # Read mass loadings file
        df = pd.read_csv(mass_loadings_file)
        print(f"  - Loaded mass loadings: {df.shape}")
        print(f"  - Columns: {df.columns.tolist()}")
        
        # Parse dates
        if 'date' not in df.columns:
            print(f"ERROR: 'date' column not found in mass loadings file")
            return None
        df['date'] = pd.to_datetime(df['date'])
        
        # Find the gauge column
        gauge_col = f"{gauge_id} m3/s"
        
        if gauge_col not in df.columns:
            print(f"ERROR: Column '{gauge_col}' not found in mass loadings file")
            print(f"  Available columns: {df.columns.tolist()}")
            return None
        
        # Store data in m³/s (original units)
        df['snowmelt_m3s'] = df[gauge_col]
        
        # Convert to mm/day if requested
        if unit == 'mm' and conversion_m3s_to_mm_day is not None:
            df['snowmelt_mm_day'] = df['snowmelt_m3s'] * conversion_m3s_to_mm_day
            snowmelt_col = 'snowmelt_mm_day'
            unit_label = 'mm/day'
        else:
            snowmelt_col = 'snowmelt_m3s'
            unit_label = 'm³/s'
        
        print(f"  - Successfully loaded snowmelt data in {unit_label}")
        print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Filter for validation period
        start_date = pd.to_datetime(validation_start)
        end_date = pd.to_datetime(validation_end)
        
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        df_filtered = df[mask].copy()
        
        if len(df_filtered) == 0:
            print(f"ERROR: No data found for period {validation_start} to {validation_end}")
            return None
        
        # Count statistics
        zero_count = (df_filtered[snowmelt_col] == 0).sum()
        nonzero_count = (df_filtered[snowmelt_col] > 0).sum()
        
        print(f"  - Filtered to {len(df_filtered)} records")
        print(f"  - Zero snowmelt days: {zero_count}")
        print(f"  - Non-zero snowmelt days: {nonzero_count}")
        print(f"  - Mean snowmelt: {df_filtered[snowmelt_col].mean():.4f} {unit_label}")
        print(f"  - Max snowmelt: {df_filtered[snowmelt_col].max():.4f} {unit_label}")
        print(f"  - Sample values (first 5 days):")
        for idx, row in df_filtered.head().iterrows():
            q = row[snowmelt_col]
            if q == 0:
                print(f"      {row['date'].date()}: 0.0000 {unit_label} (no snowmelt)")
            else:
                print(f"      {row['date'].date()}: {q:.4f} {unit_label}")
        
        # Keep date and both columns (for flexibility)
        if unit == 'mm' and conversion_m3s_to_mm_day is not None:
            result_df = df_filtered[['date', 'snowmelt_m3s', 'snowmelt_mm_day']].copy()
        else:
            result_df = df_filtered[['date', 'snowmelt_m3s']].copy()
        
        print(f"  - Final valid records: {len(result_df)}")
        
        return result_df
        
    except Exception as e:
        print(f"ERROR: Failed to load snowmelt mass loadings: {e}")
        import traceback
        traceback.print_exc()
        return None

#--------------------------------------------------------------------------------

def plot_snowmelt_timeseries(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot time series of snowmelt mass loadings in mm/day.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    pandas.DataFrame
        Snowmelt data with date and mm/day
    """
    
    gauge_id = config['gauge_id']
    
    # Load snowmelt data (now in mm/day)
    snowmelt_df = load_snowmelt_mass_loadings(config, validation_start, validation_end)
    
    if snowmelt_df is None:
        print("No snowmelt mass loadings data available for plotting")
        return None
    
    # Create plot
    plt.figure(figsize=(16, 8))
    
    # Plot snowmelt time series
    plt.plot(snowmelt_df['date'], snowmelt_df['snowmelt_mm_day'], 
             'deepskyblue', linewidth=1.5, label='Snowmelt')
    
    # Formatting
    plt.xlabel('Date', fontsize=14, fontweight='bold')
    plt.ylabel('Snowmelt (mm/day)', fontsize=14, fontweight='bold')
    plt.title(f'Snowmelt Mass Loadings - Time Series\nCatchment {gauge_id}', 
             fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gcf().autofmt_xdate()
    
    # Add statistics text box
    mean_val = snowmelt_df['snowmelt_mm_day'].mean()
    max_val = snowmelt_df['snowmelt_mm_day'].max()
    total_val = snowmelt_df['snowmelt_mm_day'].sum()
    
    stats_text = (f"Statistics:\n"
                 f"Mean: {mean_val:.4f} mm/day\n"
                 f"Max: {max_val:.4f} mm/day\n"
                 f"Total: {total_val:.2f} mm")
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'snowmelt_mass_loadings_timeseries_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved snowmelt time series plot to: {save_path}")
    plt.show()
    
    # Print summary statistics
    print(f"\nSnowmelt Mass Loadings Time Series Summary for Catchment {gauge_id}:")
    print(f"  Period: {snowmelt_df['date'].min().date()} to {snowmelt_df['date'].max().date()}")
    print(f"  Mean snowmelt: {mean_val:.4f} mm/day")
    print(f"  Max snowmelt: {max_val:.4f} mm/day")
    print(f"  Total snowmelt: {total_val:.2f} mm")
    
    return snowmelt_df

#--------------------------------------------------------------------------------

def plot_snowmelt_regime(config, plot_dirs, validation_start=None, validation_end=None, unit='mm'):
    """
    Plot monthly regime of snowmelt mass loadings.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    unit : str, optional
        Unit for display ('mm' for mm/day, 'm3' for m³/s), default is 'mm'
        
    Returns:
    --------
    pandas.DataFrame
        Monthly mean snowmelt data
    """
    
    gauge_id = config['gauge_id']
    
    # Load snowmelt data in requested units
    snowmelt_df = load_snowmelt_mass_loadings(config, validation_start, validation_end, unit=unit)
    
    if snowmelt_df is None:
        print("No snowmelt mass loadings data available for plotting")
        return None
    
    # Add month column
    snowmelt_df['month'] = snowmelt_df['date'].dt.month
    snowmelt_df['year'] = snowmelt_df['date'].dt.year
    
    # Determine which column to use based on requested unit
    if unit == 'mm' and 'snowmelt_mm_day' in snowmelt_df.columns:
        snowmelt_col = 'snowmelt_mm_day'
        unit_label = 'mm/day'
    else:
        snowmelt_col = 'snowmelt_m3s'
        unit_label = 'm³/s'
    
    # Calculate monthly mean regime
    monthly_regime = snowmelt_df.groupby('month')[snowmelt_col].mean()
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot monthly regime
    plt.plot(months, monthly_regime.values, 'deepskyblue', linewidth=3, marker='o', 
             markersize=8, label='Snowmelt')
    
    # Fill under the curve for visual effect
    plt.fill_between(months, 0, monthly_regime.values, alpha=0.3, color='lightblue')
    
    # Formatting
    plt.xlabel('Month', fontsize=14, fontweight='bold')
    plt.ylabel(f'Snowmelt ({unit_label})', fontsize=14, fontweight='bold')
    plt.title(f'Snowmelt Mass Loadings - Monthly Regime\nCatchment {gauge_id}', 
             fontsize=16, fontweight='bold')
    plt.xticks(months, month_names, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Add statistics text box
    mean_annual = snowmelt_df.groupby('year')[snowmelt_col].sum().mean()
    max_daily = snowmelt_df[snowmelt_col].max()
    
    stats_text = (f"Statistics:\n"
                 f"Mean annual total: {mean_annual:.1f} {unit_label}·year\n"
                 f"Max daily: {max_daily:.4f} {unit_label}\n"
                 f"Peak month: {month_names[monthly_regime.idxmax()-1]}")
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'snowmelt_mass_loadings_regime_{unit}_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved snowmelt regime plot to: {save_path}")
    plt.show()
    
    # Print summary statistics
    print(f"\nSnowmelt Mass Loadings Summary for Catchment {gauge_id}:")
    print(f"  Period: {snowmelt_df['date'].min().date()} to {snowmelt_df['date'].max().date()}")
    print(f"  Mean annual snowmelt: {mean_annual:.1f} {unit_label}·year")
    print(f"  Max daily snowmelt: {max_daily:.4f} {unit_label}")
    print(f"  Peak month: {month_names[monthly_regime.idxmax()-1]} ({monthly_regime.max():.4f} {unit_label})")
    print(f"  Min month: {month_names[monthly_regime.idxmin()-1]} ({monthly_regime.min():.4f} {unit_label})")
    
    # Calculate seasonal distribution
    winter_months = [12, 1, 2]
    spring_months = [3, 4, 5]
    summer_months = [6, 7, 8]
    fall_months = [9, 10, 11]
    
    winter_mean = monthly_regime[monthly_regime.index.isin(winter_months)].mean()
    spring_mean = monthly_regime[monthly_regime.index.isin(spring_months)].mean()
    summer_mean = monthly_regime[monthly_regime.index.isin(summer_months)].mean()
    fall_mean = monthly_regime[monthly_regime.index.isin(fall_months)].mean()
    
    print(f"\nSeasonal distribution:")
    print(f"  Winter (Dec-Feb): {winter_mean:.4f} {unit_label}")
    print(f"  Spring (Mar-May): {spring_mean:.4f} {unit_label}")
    print(f"  Summer (Jun-Aug): {summer_mean:.4f} {unit_label}")
    print(f"  Fall (Sep-Nov): {fall_mean:.4f} {unit_label}")
    
    return monthly_regime


#--------------------------------------------------------------------------------

def plot_snowmelt_comparison_lake_vs_mass(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Compare snowmelt from two different sources:
    1. BETWEEN_SNOW_LIQ_AND_PONDED_WATER file - shows snowmelt flux (CUMULATIVE -> convert to daily rate)
    2. Mass loadings file (SNOWMELTMassLoadings) - shows snowmelt contribution to streamflow (already daily rate)
    
    Creates both time series and regime comparison plots.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    plot_dirs : dict
        Dictionary of plot directories
    validation_start : str, optional
        Start date for analysis
    validation_end : str, optional
        End date for analysis
        
    Returns:
    --------
    dict : Results including combined data, regimes, and statistics
    """
    
    config_dir = Path(config['main_dir']) / config['config_dir']
    gauge_id = config['gauge_id']
    model_type = config['model_type']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', config.get('start_date', '2000-01-01'))
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    validation_start = pd.to_datetime(validation_start)
    validation_end = pd.to_datetime(validation_end)
    
    print(f"\n{'='*100}")
    print(f"COMPARING SNOWMELT: SNOW_LIQ->PONDED_WATER vs MASS LOADINGS - CATCHMENT {gauge_id}")
    print(f"{'='*100}\n")
    print(f"Period: {validation_start.date()} to {validation_end.date()}")
    
    # =============================
    # 1. LOAD SNOW_LIQ->PONDED_WATER SNOWMELT (CUMULATIVE)
    # =============================
    
    output_dir = config_dir / f"catchment_{gauge_id}" / model_type / "output"
    snow_flux_file = output_dir / f"{gauge_id}_{model_type}_BETWEEN_SNOW_LIQ_AND_PONDED_WATER_Daily_Average_BySubbasin.csv"
 
    if not snow_flux_file.exists():
        print(f"ERROR: Snow flux file not found: {snow_flux_file}")
        return None
    
    try:
        # Read file, skip first row (row 0), second row becomes header
        df_flux = pd.read_csv(snow_flux_file, skiprows=[0])
        print(f"Loading snow flux data (BETWEEN_SNOW_LIQ_AND_PONDED_WATER):")
        print(f"  - Loaded data: {df_flux.shape}")
        print(f"  - Columns: {df_flux.columns.tolist()}")
        
        # The 'day' column contains the actual dates
        if 'day' not in df_flux.columns:
            print(f"ERROR: 'day' column not found in snow flux file")
            print(f"  Available columns: {df_flux.columns.tolist()}")
            return None
        
        if 'mean' not in df_flux.columns:
            print(f"ERROR: 'mean' column not found in snow flux file")
            print(f"  Available columns: {df_flux.columns.tolist()}")
            return None
        
        # Parse dates from 'day' column (which contains actual dates like '1983-01-01')
        df_flux['date'] = pd.to_datetime(df_flux['day'], errors='coerce')
        
        print(f"  - Date range: {df_flux['date'].min()} to {df_flux['date'].max()}")
        
        # Convert CUMULATIVE snowmelt to DAILY RATE
        # The 'mean' column contains cumulative snowmelt flux
        df_flux['cumulative_snowmelt'] = pd.to_numeric(df_flux['mean'], errors='coerce')
        
        # Calculate daily snowmelt rate as the difference between consecutive days
        df_flux['snowmelt_flux'] = df_flux['cumulative_snowmelt'].diff().fillna(0)
        
        # Set negative values to zero (can happen at the start or with numerical issues)
        df_flux['snowmelt_flux'] = df_flux['snowmelt_flux'].clip(lower=0)
        
        print(f"  - Converted cumulative snowmelt to daily rate")
        
        # Filter for validation period
        flux_mask = (df_flux['date'] >= validation_start) & (df_flux['date'] <= validation_end)
        df_flux_filtered = df_flux[flux_mask].copy()
        
        if len(df_flux_filtered) == 0:
            print(f"ERROR: No snow flux data found for period {validation_start.date()} to {validation_end.date()}")
            return None
        
        print(f"  - Filtered to {len(df_flux_filtered)} records")
        print(f"  - Mean snow flux: {df_flux_filtered['snowmelt_flux'].mean():.4f} mm/day")
        print(f"  - Max snow flux: {df_flux_filtered['snowmelt_flux'].max():.4f} mm/day")
        
    except Exception as e:
        print(f"ERROR: Failed to load snow flux data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # =============================
    # 2. LOAD MASS LOADINGS SNOWMELT
    # =============================
    
    print(f"\nLoading mass loadings snowmelt data...")
    df_mass = load_snowmelt_mass_loadings(config, validation_start, validation_end, unit='mm')
    
    if df_mass is None:
        print(f"ERROR: Could not load mass loadings snowmelt data")
        return None
    
    # Ensure we have the mm/day column
    if 'snowmelt_mm_day' not in df_mass.columns:
        print(f"ERROR: 'snowmelt_mm_day' column not found in mass loadings data")
        print(f"  Available columns: {df_mass.columns.tolist()}")
        return None
    
    print(f"  - Mean mass loadings snowmelt: {df_mass['snowmelt_mm_day'].mean():.4f} mm/day")
    print(f"  - Max mass loadings snowmelt: {df_mass['snowmelt_mm_day'].max():.4f} mm/day")
    
    # =============================
    # 3. MERGE THE TWO DATASETS
    # =============================
    
    # Merge on date
    df_combined = pd.merge(
        df_flux_filtered[['date', 'snowmelt_flux']], 
        df_mass[['date', 'snowmelt_mm_day']], 
        on='date', 
        how='inner'
    )
    
    if len(df_combined) == 0:
        print(f"\nERROR: No overlapping dates between the two datasets")
        return None
    
    print(f"\n  - Combined dataset: {len(df_combined)} records")
    
    # Calculate statistics
    correlation = np.corrcoef(df_combined['snowmelt_flux'].values, 
                            df_combined['snowmelt_mm_day'].values)[0, 1]
    bias = df_combined['snowmelt_flux'].mean() - df_combined['snowmelt_mm_day'].mean()
    rmse = np.sqrt(np.mean((df_combined['snowmelt_flux'] - df_combined['snowmelt_mm_day'])**2))
    
    print(f"\n  Comparison Statistics:")
    print(f"    Correlation: {correlation:.3f}")
    print(f"    Mean bias: {bias:.4f} mm/day")
    print(f"    RMSE: {rmse:.4f} mm/day")
    
    # =============================
    # PLOT 1: TIME SERIES COMPARISON
    # =============================
    
    print(f"\nCreating time series comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # Top plot: Snow flux
    ax1.fill_between(df_combined['date'], 0, df_combined['snowmelt_flux'], 
                     color='steelblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
    ax1.set_ylabel('Snowmelt Flux (mm/day)', fontsize=12, fontweight='bold')
    ax1.set_title('SNOW_LIQ → PONDED_WATER Flux', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text box
    flux_stats = (f"Statistics:\n"
                 f"Mean: {df_combined['snowmelt_flux'].mean():.4f} mm/day\n"
                 f"Max: {df_combined['snowmelt_flux'].max():.4f} mm/day\n"
                 f"Total: {df_combined['snowmelt_flux'].sum():.2f} mm")
    ax1.text(0.02, 0.98, flux_stats, transform=ax1.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    # Bottom plot: Mass loadings snowmelt
    ax2.fill_between(df_combined['date'], 0, df_combined['snowmelt_mm_day'], 
                     color='deepskyblue', alpha=0.7, edgecolor='blue', linewidth=0.5)
    ax2.set_ylabel('Snowmelt Mass Loadings (mm/day)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_title('Snowmelt Mass Loadings to Streamflow', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text box
    mass_stats = (f"Statistics:\n"
                 f"Mean: {df_combined['snowmelt_mm_day'].mean():.4f} mm/day\n"
                 f"Max: {df_combined['snowmelt_mm_day'].max():.4f} mm/day\n"
                 f"Total: {df_combined['snowmelt_mm_day'].sum():.2f} mm")
    ax2.text(0.02, 0.98, mass_stats, transform=ax2.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    # Format x-axis dates
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gcf().autofmt_xdate()
    
    # Overall title
    fig.suptitle(f'Snowmelt Comparison: Snow Flux vs Mass Loadings\nCatchment {gauge_id}\n'
                f'Period: {validation_start.date()} to {validation_end.date()}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save time series plot
    save_path_ts = plot_dirs['contributions'] / f'{gauge_id}_snowmelt_comparison_timeseries.png'
    plt.savefig(save_path_ts, dpi=300, bbox_inches='tight')
    print(f"✓ Saved time series comparison plot: {save_path_ts}")
    plt.show()
    
    # =============================
    # PLOT 2: MONTHLY REGIME COMPARISON
    # =============================
    
    print(f"\nCreating regime comparison plot...")
    
    # Calculate monthly regimes
    df_combined['month'] = df_combined['date'].dt.month
    
    flux_regime = df_combined.groupby('month')['snowmelt_flux'].mean()
    mass_regime = df_combined.groupby('month')['snowmelt_mm_day'].mean()
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot both regimes
    plt.plot(months, flux_regime.values, 'steelblue', linewidth=3, 
            label='SNOW_LIQ→PONDED_WATER Flux', marker='o', markersize=8)
    
    plt.plot(months, mass_regime.values, 'deepskyblue', linewidth=3, 
            label='Mass Loadings Snowmelt', marker='s', markersize=8, linestyle='--')
    
    # Fill area between to show difference
    plt.fill_between(months, flux_regime.values, mass_regime.values, 
                     alpha=0.2, color='gray', label='Difference')
    
    # Formatting
    plt.xlabel('Month', fontsize=14, fontweight='bold')
    plt.ylabel('Snowmelt (mm/day)', fontsize=14, fontweight='bold')
    plt.title(f'Snowmelt Monthly Regime Comparison\nCatchment {gauge_id}', 
             fontsize=16, fontweight='bold')
    plt.xticks(months, month_names, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.legend(fontsize=12, loc='best')
    
    # Add statistics text box
    stats_text = (f"Comparison Statistics:\n"
                 f"Correlation: {correlation:.3f}\n"
                 f"Mean bias: {bias:.4f} mm/day\n"
                 f"RMSE: {rmse:.4f} mm/day\n\n"
                 f"Snow Flux mean: {flux_regime.mean():.4f} mm/day\n"
                 f"Mass Loadings mean: {mass_regime.mean():.4f} mm/day\n\n"
                 f"Flux peak: {month_names[flux_regime.idxmax()-1]}\n"
                 f"Mass peak: {month_names[mass_regime.idxmax()-1]}")
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save regime plot
    save_path_regime = plot_dirs['contributions'] / f'{gauge_id}_snowmelt_comparison_regime.png'
    plt.savefig(save_path_regime, dpi=300, bbox_inches='tight')
    print(f"✓ Saved regime comparison plot: {save_path_regime}")
    plt.show()
    
    # =============================
    # PRINT SUMMARY STATISTICS
    # =============================
    
    print(f"\n{'='*80}")
    print(f"SNOWMELT COMPARISON SUMMARY - CATCHMENT {gauge_id}")
    print(f"{'='*80}")
    print(f"Period: {validation_start.date()} to {validation_end.date()}")
    print(f"Number of records: {len(df_combined)}")
    
    print(f"\nDaily Averages:")
    print(f"  Snow Flux: {df_combined['snowmelt_flux'].mean():.4f} mm/day")
    print(f"  Mass Loadings: {df_combined['snowmelt_mm_day'].mean():.4f} mm/day")
    print(f"  Difference: {bias:.4f} mm/day")
    
    print(f"\nTotal Snowmelt:")
    print(f"  Snow Flux: {df_combined['snowmelt_flux'].sum():.2f} mm")
    print(f"  Mass Loadings: {df_combined['snowmelt_mm_day'].sum():.2f} mm")
    print(f"  Difference: {df_combined['snowmelt_flux'].sum() - df_combined['snowmelt_mm_day'].sum():.2f} mm")
    
    print(f"\nComparison Metrics:")
    print(f"  Correlation: {correlation:.3f}")
    print(f"  RMSE: {rmse:.4f} mm/day")
    print(f"  Mean bias: {bias:.4f} mm/day")
    print(f"  Relative bias: {(bias/df_combined['snowmelt_mm_day'].mean()*100):.1f}%")
    
    print(f"\nPeak Months:")
    print(f"  Snow Flux: {month_names[flux_regime.idxmax()-1]} ({flux_regime.max():.4f} mm/day)")
    print(f"  Mass Loadings: {month_names[mass_regime.idxmax()-1]} ({mass_regime.max():.4f} mm/day)")
    
    print(f"{'='*80}\n")
    
    return {
        'combined_data': df_combined,
        'flux_regime': flux_regime,
        'mass_regime': mass_regime,
        'statistics': {
            'correlation': correlation,
            'rmse': rmse,
            'bias': bias
        },
        'plots': {
            'timeseries': save_path_ts,
            'regime': save_path_regime
        }
    }

#--------------------------------------------------------------------------------

def load_glacier_melt_mass_loadings(config, validation_start=None, validation_end=None, unit='m3'):
    """
    Load glacier melt mass loadings data from Raven output.
    Loads data for SMALL, LARGE, and ALL glacier types.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    unit : str, optional
        Unit for output ('mm' for mm/day, 'm3' for m³/s), default is 'm3'
        
    Returns:
    --------
    dict
        Dictionary containing dataframes for each glacier type with data in specified units
    """
    
    config_dir = Path(config['main_dir']) / config['config_dir']
    gauge_id = config['gauge_id']
    model_type = config['model_type']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', config.get('start_date', '2000-01-01'))
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"Loading glacier melt mass loadings for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    print(f"  - Requested unit: {unit}")
    
    # Load catchment area for unit conversion
    conversion_m3s_to_mm_day = None
    if unit == 'mm':
        topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
        catchment_shape_file = topo_dir / "HRU.shp"
        
        try:
            if catchment_shape_file.exists():
                import geopandas as gpd
                hru_gdf = gpd.read_file(catchment_shape_file)
                total_area_km2 = hru_gdf['Area_km2'].sum()
                # Conversion factor: m³/s to mm/day
                conversion_m3s_to_mm_day = 86400 / (total_area_km2 * 1000000) * 1000
                print(f"  - Catchment area: {total_area_km2:.2f} km²")
                print(f"  - Conversion factor (m³/s to mm/day): {conversion_m3s_to_mm_day:.6f}")
            else:
                print(f"ERROR: Catchment shapefile not found: {catchment_shape_file}")
                print(f"  Falling back to m³/s")
                unit = 'm3'
        except Exception as e:
            print(f"ERROR: Could not load catchment area: {e}")
            print(f"  Falling back to m³/s")
            unit = 'm3'
    
    # Define file paths for all three glacier types
    glacier_types = {
        'small': 'GLACIERMELT_SMALL',
        'large': 'GLACIERMELT_LARGE',
        'all': 'GLACIERMELT_ALL'
    }
    
    results = {}
    
    for glacier_type, file_suffix in glacier_types.items():
        glacier_file = config_dir / f"catchment_{gauge_id}" / model_type / "output" / f"{gauge_id}_{model_type}_{file_suffix}MassLoadings.csv"
        
        # Check if file exists
        if not glacier_file.exists():
            print(f"  WARNING: {glacier_type.upper()} glacier file not found: {glacier_file}")
            results[glacier_type] = None
            continue
        
        try:
            # Read mass loadings file
            df = pd.read_csv(glacier_file)
            print(f"  - Loaded {glacier_type.upper()} glacier data: {df.shape}")
            
            # Parse dates
            if 'date' not in df.columns:
                print(f"  ERROR: 'date' column not found in {glacier_type} file")
                results[glacier_type] = None
                continue
            
            df['date'] = pd.to_datetime(df['date'])
            
            # Find the gauge column
            gauge_col = f"{gauge_id} m3/s"
            
            if gauge_col not in df.columns:
                print(f"  ERROR: Column '{gauge_col}' not found in {glacier_type} file")
                print(f"    Available columns: {df.columns.tolist()}")
                results[glacier_type] = None
                continue
            
            # Store data in m³/s (original units)
            df['glacier_melt_m3s'] = df[gauge_col]
            
            # Convert to mm/day if requested
            if unit == 'mm' and conversion_m3s_to_mm_day is not None:
                df['glacier_melt_mm_day'] = df['glacier_melt_m3s'] * conversion_m3s_to_mm_day
                glacier_melt_col = 'glacier_melt_mm_day'
                unit_label = 'mm/day'
            else:
                glacier_melt_col = 'glacier_melt_m3s'
                unit_label = 'm³/s'
            
            print(f"    - Successfully loaded data in {unit_label}")
            
            # Filter for validation period
            start_date = pd.to_datetime(validation_start)
            end_date = pd.to_datetime(validation_end)
            
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            df_filtered = df[mask].copy()
            
            if len(df_filtered) == 0:
                print(f"  ERROR: No {glacier_type} data found for period {validation_start} to {validation_end}")
                results[glacier_type] = None
                continue
            
            # Count statistics
            zero_count = (df_filtered[glacier_melt_col] == 0).sum()
            nonzero_count = (df_filtered[glacier_melt_col] > 0).sum()
            
            print(f"    ✓ Filtered to {len(df_filtered)} records")
            print(f"    - Zero glacier melt days: {zero_count}")
            print(f"    - Non-zero glacier melt days: {nonzero_count}")
            print(f"    - Mean: {df_filtered[glacier_melt_col].mean():.4f} {unit_label}")
            print(f"    - Max: {df_filtered[glacier_melt_col].max():.4f} {unit_label}")
            
            # Keep date and both columns (for flexibility)
            if unit == 'mm' and conversion_m3s_to_mm_day is not None:
                result_df = df_filtered[['date', 'glacier_melt_m3s', 'glacier_melt_mm_day']].copy()
            else:
                result_df = df_filtered[['date', 'glacier_melt_m3s']].copy()
            
            results[glacier_type] = result_df
            
        except Exception as e:
            print(f"  ERROR: Failed to load {glacier_type} glacier data: {e}")
            import traceback
            traceback.print_exc()
            results[glacier_type] = None
    
    # Check if we successfully loaded at least one dataset
    successful_loads = sum(1 for df in results.values() if df is not None)
    
    if successful_loads == 0:
        print(f"  ✗ Failed to load any glacier melt data")
        return None
    
    print(f"  ✓ Successfully loaded {successful_loads}/3 glacier melt datasets")
    
    return results

#--------------------------------------------------------------------------------

def plot_glacier_melt_regime(config, plot_dirs, validation_start=None, validation_end=None, unit='mm'):
    """
    Plot monthly regime of glacier melt mass loadings for all three glacier types.
    Creates a single plot with three lines (SMALL, LARGE, ALL).
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    unit : str, optional
        Unit for display ('mm' for mm/day, 'm3' for m³/s), default is 'mm'
        
    Returns:
    --------
    dict
        Dictionary containing monthly mean glacier melt data for each type
    """
    
    gauge_id = config['gauge_id']
    
    # Load glacier melt data for all types in requested units
    glacier_data = load_glacier_melt_mass_loadings(config, validation_start, validation_end, unit=unit)
    
    if glacier_data is None:
        print("No glacier melt mass loadings data available for plotting")
        return None
    
    # Check if we have at least one dataset
    if not any(df is not None for df in glacier_data.values()):
        print("No valid glacier melt data loaded")
        return None
    
    # Determine which column to use based on requested unit
    if unit == 'mm':
        glacier_melt_col = 'glacier_melt_mm_day'
        unit_label = 'mm/day'
    else:
        glacier_melt_col = 'glacier_melt_m3s'
        unit_label = 'm³/s'
    
    # Calculate monthly regimes for each glacier type
    monthly_regimes = {}
    
    for glacier_type, df in glacier_data.items():
        if df is not None:
            # Check if the column exists
            if glacier_melt_col not in df.columns:
                print(f"  WARNING: Column '{glacier_melt_col}' not found for {glacier_type}, using m³/s instead")
                glacier_melt_col = 'glacier_melt_m3s'
                unit_label = 'm³/s'
            
            # Add month column
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            
            # Calculate monthly mean regime
            monthly_regime = df.groupby('month')[glacier_melt_col].mean()
            monthly_regimes[glacier_type] = monthly_regime
    
    if len(monthly_regimes) == 0:
        print("No monthly regimes could be calculated")
        return None
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Define colors for each glacier type
    colors = {
        'small': 'lightgray',
        'large': 'darkgray',
        'all': 'black'
    }
    
    line_styles = {
        'small': '--',
        'large': ':',
        'all': '-'
    }
    
    linewidths = {
        'small': 2,
        'large': 2,
        'all': 3
    }
    
    # Plot each glacier type
    for glacier_type, monthly_regime in monthly_regimes.items():
        plt.plot(months, monthly_regime.values, 
                color=colors[glacier_type],
                linestyle=line_styles[glacier_type],
                linewidth=linewidths[glacier_type],
                marker='o' if glacier_type == 'all' else None,
                markersize=8 if glacier_type == 'all' else 0,
                label=f'{glacier_type.upper()} Glaciers')
    
    # Formatting
    plt.xlabel('Month', fontsize=14, fontweight='bold')
    plt.ylabel(f'Glacier Melt ({unit_label})', fontsize=14, fontweight='bold')
    plt.title(f'Glacier Melt Mass Loadings - Monthly Regime\nCatchment {gauge_id}', 
             fontsize=16, fontweight='bold')
    plt.xticks(months, month_names, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    
    # Add statistics text box
    stats_lines = []
    for glacier_type, monthly_regime in monthly_regimes.items():
        # Calculate annual total correctly (monthly averages * days per month)
        days_per_month = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        annual_total = sum(monthly_regime.iloc[i-1] * days_per_month[i-1] for i in range(1, 13))
        
        max_daily = glacier_data[glacier_type][glacier_melt_col].max()
        peak_month = month_names[monthly_regime.idxmax()-1]
        
        stats_lines.append(f"{glacier_type.upper()}:")
        stats_lines.append(f"  Annual: {annual_total:.1f} {unit_label.split('/')[0]}/year")
        stats_lines.append(f"  Max: {max_daily:.4f} {unit_label}")
        stats_lines.append(f"  Peak: {peak_month}")
    
    stats_text = '\n'.join(stats_lines)
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'glacier_melt_regime_{unit}_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved glacier melt regime plot to: {save_path}")
    plt.show()
    
    # Print summary statistics
    print(f"\nGlacier Melt Mass Loadings Summary for Catchment {gauge_id}:")
    print(f"  Period: {glacier_data[list(monthly_regimes.keys())[0]]['date'].min().date()} to {glacier_data[list(monthly_regimes.keys())[0]]['date'].max().date()}")
    
    for glacier_type, monthly_regime in monthly_regimes.items():
        df = glacier_data[glacier_type]
        
        # Calculate annual total correctly (monthly averages * days per month)
        days_per_month = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        annual_total = sum(monthly_regime.iloc[i-1] * days_per_month[i-1] for i in range(1, 13))
        
        max_daily = df[glacier_melt_col].max()
        peak_month = month_names[monthly_regime.idxmax()-1]
        min_month = month_names[monthly_regime.idxmin()-1]
        
        print(f"\n  {glacier_type.upper()} Glaciers:")
        print(f"    Mean annual glacier melt: {annual_total:.1f} {unit_label.split('/')[0]}/year")
        print(f"    Max daily glacier melt: {max_daily:.4f} {unit_label}")
        print(f"    Peak month: {peak_month} ({monthly_regime.max():.4f} {unit_label})")
        print(f"    Min month: {min_month} ({monthly_regime.min():.4f} {unit_label})")
        
        # Calculate seasonal distribution
        winter_months = [12, 1, 2]
        spring_months = [3, 4, 5]
        summer_months = [6, 7, 8]
        fall_months = [9, 10, 11]
        
        winter_mean = monthly_regime[monthly_regime.index.isin(winter_months)].mean()
        spring_mean = monthly_regime[monthly_regime.index.isin(spring_months)].mean()
        summer_mean = monthly_regime[monthly_regime.index.isin(summer_months)].mean()
        fall_mean = monthly_regime[monthly_regime.index.isin(fall_months)].mean()
        
        print(f"    Seasonal distribution:")
        print(f"      Winter (Dec-Feb): {winter_mean:.4f} {unit_label}")
        print(f"      Spring (Mar-May): {spring_mean:.4f} {unit_label}")
        print(f"      Summer (Jun-Aug): {summer_mean:.4f} {unit_label}")
        print(f"      Fall (Sep-Nov): {fall_mean:.4f} {unit_label}")
    
    return monthly_regimes

#--------------------------------------------------------------------------------

def plot_glacier_melt_timeseries(config, plot_dirs, validation_start=None, validation_end=None, unit='mm'):
    """
    Plot time series of glacier melt mass loadings for all three glacier types.
    Creates three subplots stacked vertically.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    unit : str, optional
        Unit for display ('mm' for mm/day, 'm3' for m³/s), default is 'mm'
        
    Returns:
    --------
    dict
        Dictionary containing glacier melt data for each type
    """
    
    gauge_id = config['gauge_id']
    
    # Load glacier melt data in requested units
    glacier_data = load_glacier_melt_mass_loadings(config, validation_start, validation_end, unit=unit)
    
    if glacier_data is None:
        print("No glacier melt mass loadings data available for plotting")
        return None
    
    # Count how many valid datasets we have
    valid_datasets = {k: v for k, v in glacier_data.items() if v is not None}
    
    if len(valid_datasets) == 0:
        print("No valid glacier melt datasets loaded")
        return None
    
    # Determine which column to use based on requested unit
    if unit == 'mm':
        glacier_melt_col = 'glacier_melt_mm_day'
        unit_label = 'mm/day'
    else:
        glacier_melt_col = 'glacier_melt_m3s'
        unit_label = 'm³/s'
    
    # Create subplots
    n_plots = len(valid_datasets)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 4*n_plots), sharex=True)
    
    # Make axes iterable if only one plot
    if n_plots == 1:
        axes = [axes]
    
    # Define colors for each glacier type
    colors = {
        'small': 'lightgray',
        'large': 'darkgray',
        'all': 'black'
    }
    
    # Plot each glacier type
    for i, (glacier_type, df) in enumerate(valid_datasets.items()):
        ax = axes[i]
        
        # Check if the column exists
        if glacier_melt_col not in df.columns:
            print(f"  WARNING: Column '{glacier_melt_col}' not found for {glacier_type}, using m³/s instead")
            glacier_melt_col = 'glacier_melt_m3s'
            unit_label = 'm³/s'
        
        # Plot time series
        ax.plot(df['date'], df[glacier_melt_col], 
               color=colors[glacier_type], linewidth=1.5, 
               label=f'{glacier_type.upper()} Glaciers')
        
        # Formatting
        ax.set_ylabel(f'Glacier Melt ({unit_label})', fontsize=12, fontweight='bold')
        ax.set_title(f'{glacier_type.upper()} Glacier Melt - Time Series', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=11)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        
        # Add statistics text box
        mean_val = df[glacier_melt_col].mean()
        max_val = df[glacier_melt_col].max()
        total_val = df[glacier_melt_col].sum()
        
        stats_text = (f"Statistics:\n"
                     f"Mean: {mean_val:.4f} {unit_label}\n"
                     f"Max: {max_val:.4f} {unit_label}\n"
                     f"Total: {total_val:.2f} {unit_label}·days")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    # Set x-label only on bottom subplot
    axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
    plt.gcf().autofmt_xdate()
    
    # Add overall title
    fig.suptitle(f'Glacier Melt Mass Loadings - Time Series\nCatchment {gauge_id}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'glacier_melt_timeseries_{unit}_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved glacier melt time series plot to: {save_path}")
    plt.show()
    
    return glacier_data

#--------------------------------------------------------------------------------

def plot_streamflow_with_all_glacier_snowmelt_regime(config, plot_dirs, validation_start=None, validation_end=None, unit='m3'):
    """
    Plot streamflow regime with total glacier melt and snowmelt contributions.
    Shows: Simulated streamflow, Observed streamflow, Snowmelt, and Total Glacier Melt (ALL)
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    unit : str, optional
        Unit for display ('mm' for mm/day, 'm3' for m³/s), default is 'm3'
        
    Returns:
    --------
    dict
        Dictionary containing all monthly regime data
    """
    
    gauge_id = config['gauge_id']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"Creating streamflow regime with all glacier melt and snowmelt for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    print(f"  - Unit: {unit}")
    
    # Load catchment area for unit conversion
    conversion_m3s_to_mm_day = None
    if unit == 'mm':
        config_dir = Path(config['main_dir']) / config['config_dir']
        topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
        catchment_shape_file = topo_dir / "HRU.shp"
        
        try:
            if catchment_shape_file.exists():
                import geopandas as gpd
                hru_gdf = gpd.read_file(catchment_shape_file)
                total_area_km2 = hru_gdf['Area_km2'].sum()
                # Conversion factor: m³/s to mm/day
                conversion_m3s_to_mm_day = 86400 / (total_area_km2 * 1000000) * 1000
                print(f"  - Catchment area: {total_area_km2:.2f} km²")
                print(f"  - Conversion factor (m³/s to mm/day): {conversion_m3s_to_mm_day:.6f}")
            else:
                print(f"ERROR: Catchment shapefile not found: {catchment_shape_file}")
                print(f"  Falling back to m³/s")
                unit = 'm3'
        except Exception as e:
            print(f"ERROR: Could not load catchment area: {e}")
            print(f"  Falling back to m³/s")
            unit = 'm3'
    
    # Set unit label
    unit_label = 'mm/day' if unit == 'mm' else 'm³/s'
    
    # 1. Load streamflow data
    streamflow_data = load_hydrograph_data(config)
    if streamflow_data is None:
        print("ERROR: Could not load streamflow data")
        return None
    
    # Filter for validation period
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    
    streamflow_mask = (streamflow_data['date'] >= start_date) & (streamflow_data['date'] <= end_date)
    streamflow_filtered = streamflow_data[streamflow_mask].copy()
    
    if len(streamflow_filtered) == 0:
        print(f"ERROR: No streamflow data found for period {validation_start} to {validation_end}")
        return None
    
    # Convert streamflow if needed
    if unit == 'mm' and conversion_m3s_to_mm_day is not None:
        streamflow_filtered['obs_Q_converted'] = streamflow_filtered['obs_Q'] * conversion_m3s_to_mm_day
        streamflow_filtered['sim_Q_converted'] = streamflow_filtered['sim_Q'] * conversion_m3s_to_mm_day
    else:
        streamflow_filtered['obs_Q_converted'] = streamflow_filtered['obs_Q']
        streamflow_filtered['sim_Q_converted'] = streamflow_filtered['sim_Q']
    
    # Calculate monthly regime for streamflow
    streamflow_filtered['month'] = streamflow_filtered['date'].dt.month
    
    streamflow_regime = {}
    if 'obs_Q_converted' in streamflow_filtered.columns:
        streamflow_regime['observed'] = streamflow_filtered.groupby('month')['obs_Q_converted'].mean()
    if 'sim_Q_converted' in streamflow_filtered.columns:
        streamflow_regime['simulated'] = streamflow_filtered.groupby('month')['sim_Q_converted'].mean()
    
    # 2. Load snowmelt data
    snowmelt_df = load_snowmelt_mass_loadings(config, validation_start, validation_end, unit=unit)
    if snowmelt_df is None:
        print("ERROR: Could not load snowmelt data")
        return None
    
    # Determine snowmelt column based on unit
    snowmelt_col = 'snowmelt_mm_day' if unit == 'mm' and 'snowmelt_mm_day' in snowmelt_df.columns else 'snowmelt_m3s'
    
    # Calculate monthly regime for snowmelt
    snowmelt_df['month'] = snowmelt_df['date'].dt.month
    snowmelt_regime = snowmelt_df.groupby('month')[snowmelt_col].mean()
    
    # 3. Load glacier melt data (ALL)
    glacier_data = load_glacier_melt_mass_loadings(config, validation_start, validation_end, unit=unit)
    if glacier_data is None or glacier_data.get('all') is None:
        print("ERROR: Could not load ALL glacier melt data")
        return None
    
    # Determine glacier melt column based on unit
    glacier_melt_col = 'glacier_melt_mm_day' if unit == 'mm' and 'glacier_melt_mm_day' in glacier_data['all'].columns else 'glacier_melt_m3s'
    
    # Calculate monthly regime for ALL glacier melt
    glacier_all_df = glacier_data['all']
    glacier_all_df['month'] = glacier_all_df['date'].dt.month
    glacier_all_regime = glacier_all_df.groupby('month')[glacier_melt_col].mean()
    
    # 4. Create plot
    plt.figure(figsize=(14, 8))
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot filled polygons FIRST (bottom layer)
    # Plot snowmelt as filled polygon - softer light blue
    plt.fill_between(months, 0, snowmelt_regime.values, 
                     color='#B3D9FF', alpha=0.7, label='Snowmelt', zorder=1, edgecolor='#6DB3F2', linewidth=1.5)
    
    # Plot total glacier melt as filled polygon - warmer brown/orange tone
    plt.fill_between(months, 0, glacier_all_regime.values, 
                     color='#C17817', alpha=0.6, label='Glacier Runoff', zorder=2, edgecolor='#8B5A00', linewidth=1.5)
    
    # Plot observed streamflow (line without markers)
    if 'observed' in streamflow_regime:
        plt.plot(months, streamflow_regime['observed'].values, 'k-', 
                linewidth=3, label='Observed Streamflow', zorder=4)
    
    # Plot simulated streamflow (dashed line without markers)
    if 'simulated' in streamflow_regime:
        plt.plot(months, streamflow_regime['simulated'].values, 'C0--', 
                linewidth=2.5, label='Simulated Streamflow', zorder=3)
    
    # Formatting
    plt.xlabel('Month', fontsize=14, fontweight='bold')
    plt.ylabel(f'Discharge ({unit_label})', fontsize=14, fontweight='bold')
    plt.title(f'Streamflow Regime with Melt Contributions\nCatchment {gauge_id}', 
             fontsize=16, fontweight='bold')
    plt.xticks(months, month_names, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.legend(fontsize=12, loc='best')
    
    # Add statistics text box
    stats_lines = []
    if 'observed' in streamflow_regime:
        stats_lines.append(f"Obs. Streamflow: {streamflow_regime['observed'].mean():.4f} {unit_label} (mean)")
    if 'simulated' in streamflow_regime:
        stats_lines.append(f"Sim. Streamflow: {streamflow_regime['simulated'].mean():.4f} {unit_label} (mean)")
    stats_lines.append(f"Snowmelt: {snowmelt_regime.mean():.4f} {unit_label} (mean)")
    stats_lines.append(f"Glacier Melt: {glacier_all_regime.mean():.4f} {unit_label} (mean)")
    
    stats_text = '\n'.join(stats_lines)
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'streamflow_all_glacier_snowmelt_regime_{unit}_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved streamflow regime with all glacier melt plot to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\nStreamflow Regime Summary:")
    print(f"  Period: {validation_start} to {validation_end}")
    print(f"  Unit: {unit_label}")
    if 'observed' in streamflow_regime:
        print(f"  Observed streamflow: {streamflow_regime['observed'].mean():.4f} {unit_label} (mean)")
    if 'simulated' in streamflow_regime:
        print(f"  Simulated streamflow: {streamflow_regime['simulated'].mean():.4f} {unit_label} (mean)")
    print(f"  Snowmelt: {snowmelt_regime.mean():.4f} {unit_label} (mean)")
    print(f"  Glacier melt: {glacier_all_regime.mean():.4f} {unit_label} (mean)")
    
    # Return all data
    return {
        'streamflow': streamflow_regime,
        'snowmelt': snowmelt_regime,
        'glacier_all': glacier_all_regime,
        'unit': unit_label
    }

#--------------------------------------------------------------------------------

def plot_streamflow_with_separated_glacier_snowmelt_regime(config, plot_dirs, validation_start=None, validation_end=None, unit='m3'):
    """
    Plot streamflow regime with separated glacier melt (SMALL, LARGE) and snowmelt contributions.
    Shows: Simulated streamflow, Observed streamflow, Snowmelt, Small Glacier Melt, Large Glacier Melt
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    unit : str, optional
        Unit for display ('mm' for mm/day, 'm3' for m³/s), default is 'm3'
        
    Returns:
    --------
    dict
        Dictionary containing all monthly regime data
    """
    
    gauge_id = config['gauge_id']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"Creating streamflow regime with separated glacier melt and snowmelt for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    print(f"  - Unit: {unit}")
    
    # Load catchment area for unit conversion
    conversion_m3s_to_mm_day = None
    if unit == 'mm':
        config_dir = Path(config['main_dir']) / config['config_dir']
        topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
        catchment_shape_file = topo_dir / "HRU.shp"
        
        try:
            if catchment_shape_file.exists():
                import geopandas as gpd
                hru_gdf = gpd.read_file(catchment_shape_file)
                total_area_km2 = hru_gdf['Area_km2'].sum()
                # Conversion factor: m³/s to mm/day
                conversion_m3s_to_mm_day = 86400 / (total_area_km2 * 1000000) * 1000
                print(f"  - Catchment area: {total_area_km2:.2f} km²")
                print(f"  - Conversion factor (m³/s to mm/day): {conversion_m3s_to_mm_day:.6f}")
            else:
                print(f"ERROR: Catchment shapefile not found: {catchment_shape_file}")
                print(f"  Falling back to m³/s")
                unit = 'm3'
        except Exception as e:
            print(f"ERROR: Could not load catchment area: {e}")
            print(f"  Falling back to m³/s")
            unit = 'm3'
    
    # Set unit label
    unit_label = 'mm/day' if unit == 'mm' else 'm³/s'
    
    # 1. Load streamflow data
    streamflow_data = load_hydrograph_data(config)
    if streamflow_data is None:
        print("ERROR: Could not load streamflow data")
        return None
    
    # Filter for validation period
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    
    streamflow_mask = (streamflow_data['date'] >= start_date) & (streamflow_data['date'] <= end_date)
    streamflow_filtered = streamflow_data[streamflow_mask].copy()
    
    if len(streamflow_filtered) == 0:
        print(f"ERROR: No streamflow data found for period {validation_start} to {validation_end}")
        return None
    
    # Convert streamflow if needed
    if unit == 'mm' and conversion_m3s_to_mm_day is not None:
        streamflow_filtered['obs_Q_converted'] = streamflow_filtered['obs_Q'] * conversion_m3s_to_mm_day
        streamflow_filtered['sim_Q_converted'] = streamflow_filtered['sim_Q'] * conversion_m3s_to_mm_day
    else:
        streamflow_filtered['obs_Q_converted'] = streamflow_filtered['obs_Q']
        streamflow_filtered['sim_Q_converted'] = streamflow_filtered['sim_Q']
    
    # Calculate monthly regime for streamflow
    streamflow_filtered['month'] = streamflow_filtered['date'].dt.month
    
    streamflow_regime = {}
    if 'obs_Q_converted' in streamflow_filtered.columns:
        streamflow_regime['observed'] = streamflow_filtered.groupby('month')['obs_Q_converted'].mean()
    if 'sim_Q_converted' in streamflow_filtered.columns:
        streamflow_regime['simulated'] = streamflow_filtered.groupby('month')['sim_Q_converted'].mean()
    
    # 2. Load snowmelt data
    snowmelt_df = load_snowmelt_mass_loadings(config, validation_start, validation_end, unit=unit)
    if snowmelt_df is None:
        print("ERROR: Could not load snowmelt data")
        return None
    
    # Determine snowmelt column based on unit
    snowmelt_col = 'snowmelt_mm_day' if unit == 'mm' and 'snowmelt_mm_day' in snowmelt_df.columns else 'snowmelt_m3s'
    
    # Calculate monthly regime for snowmelt
    snowmelt_df['month'] = snowmelt_df['date'].dt.month
    snowmelt_regime = snowmelt_df.groupby('month')[snowmelt_col].mean()
    
    # 3. Load glacier melt data (SMALL and LARGE)
    glacier_data = load_glacier_melt_mass_loadings(config, validation_start, validation_end, unit=unit)
    if glacier_data is None:
        print("ERROR: Could not load glacier melt data")
        return None
    
    # Determine glacier melt column based on unit
    glacier_melt_col = 'glacier_melt_mm_day' if unit == 'mm' else 'glacier_melt_m3s'
    
    # Calculate monthly regimes for SMALL and LARGE glacier melt
    glacier_regimes = {}
    
    if glacier_data.get('small') is not None:
        glacier_small_df = glacier_data['small']
        
        # Check if column exists
        if glacier_melt_col not in glacier_small_df.columns:
            print(f"  WARNING: '{glacier_melt_col}' not found for SMALL glacier, using m³/s")
            glacier_melt_col = 'glacier_melt_m3s'
        
        glacier_small_df['month'] = glacier_small_df['date'].dt.month
        glacier_regimes['small'] = glacier_small_df.groupby('month')[glacier_melt_col].mean()
    else:
        print("WARNING: SMALL glacier data not available")
    
    if glacier_data.get('large') is not None:
        glacier_large_df = glacier_data['large']
        
        # Check if column exists
        if glacier_melt_col not in glacier_large_df.columns:
            print(f"  WARNING: '{glacier_melt_col}' not found for LARGE glacier, using m³/s")
            glacier_melt_col = 'glacier_melt_m3s'
        
        glacier_large_df['month'] = glacier_large_df['date'].dt.month
        glacier_regimes['large'] = glacier_large_df.groupby('month')[glacier_melt_col].mean()
    else:
        print("WARNING: LARGE glacier data not available")
    
    if len(glacier_regimes) == 0:
        print("ERROR: No glacier melt data available")
        return None
    
    # 4. Create plot
    plt.figure(figsize=(14, 8))
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot filled polygons FIRST (bottom layer)
    # Plot snowmelt as filled polygon - softer light blue
    plt.fill_between(months, 0, snowmelt_regime.values, 
                     color='#B3D9FF', alpha=0.7, label='Snowmelt', zorder=1, edgecolor='#6DB3F2', linewidth=1.5)
    
    # Plot small glacier melt as filled polygon - lighter orange/tan
    if 'small' in glacier_regimes:
        plt.fill_between(months, 0, glacier_regimes['small'].values, 
                        color='#E6A85C', alpha=0.7, label='Small Glacier Runoff', zorder=2, 
                        edgecolor='#CC8800', linewidth=1.5)
    
    # Plot large glacier melt as filled polygon - darker brown/rust
    if 'large' in glacier_regimes:
        plt.fill_between(months, 0, glacier_regimes['large'].values, 
                        color='#8B4513', alpha=0.7, label='Large Glacier Runoff', zorder=2,
                        edgecolor='#654321', linewidth=1.5)
    
    # Plot observed streamflow (line without markers)
    if 'observed' in streamflow_regime:
        plt.plot(months, streamflow_regime['observed'].values, 'k-', 
                linewidth=3, label='Observed Streamflow', zorder=5)
    
    # Plot simulated streamflow (dashed line without markers)
    if 'simulated' in streamflow_regime:
        plt.plot(months, streamflow_regime['simulated'].values, 'C0--', 
                linewidth=2.5, label='Simulated Streamflow', zorder=4)
    
    # Formatting
    plt.xlabel('Month', fontsize=14, fontweight='bold')
    plt.ylabel(f'Discharge ({unit_label})', fontsize=14, fontweight='bold')
    plt.title(f'Streamflow Regime with Separated Glacier Melt\nCatchment {gauge_id}', 
             fontsize=16, fontweight='bold')
    plt.xticks(months, month_names, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.legend(fontsize=12, loc='best')
    
    # Add statistics text box
    stats_lines = []
    if 'observed' in streamflow_regime:
        stats_lines.append(f"Obs. Streamflow: {streamflow_regime['observed'].mean():.4f} {unit_label} (mean)")
    if 'simulated' in streamflow_regime:
        stats_lines.append(f"Sim. Streamflow: {streamflow_regime['simulated'].mean():.4f} {unit_label} (mean)")
    stats_lines.append(f"Snowmelt: {snowmelt_regime.mean():.4f} {unit_label} (mean)")
    
    if 'small' in glacier_regimes:
        stats_lines.append(f"Small Glacier: {glacier_regimes['small'].mean():.4f} {unit_label} (mean)")
    if 'large' in glacier_regimes:
        stats_lines.append(f"Large Glacier: {glacier_regimes['large'].mean():.4f} {unit_label} (mean)")
    
    stats_text = '\n'.join(stats_lines)
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'streamflow_separated_glacier_snowmelt_regime_{unit}_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved streamflow regime with separated glacier melt plot to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\nStreamflow Regime Summary:")
    print(f"  Period: {validation_start} to {validation_end}")
    print(f"  Unit: {unit_label}")
    
    if 'observed' in streamflow_regime:
        print(f"  Observed streamflow: {streamflow_regime['observed'].mean():.4f} {unit_label} (mean)")
        print(f"    Peak: {month_names[streamflow_regime['observed'].idxmax()-1]} ({streamflow_regime['observed'].max():.4f} {unit_label})")
    
    if 'simulated' in streamflow_regime:
        print(f"  Simulated streamflow: {streamflow_regime['simulated'].mean():.4f} {unit_label} (mean)")
        print(f"    Peak: {month_names[streamflow_regime['simulated'].idxmax()-1]} ({streamflow_regime['simulated'].max():.4f} {unit_label})")
    
    print(f"  Snowmelt: {snowmelt_regime.mean():.4f} {unit_label} (mean)")
    print(f"    Peak: {month_names[snowmelt_regime.idxmax()-1]} ({snowmelt_regime.max():.4f} {unit_label})")
    
    if 'small' in glacier_regimes:
        print(f"  Small glacier melt: {glacier_regimes['small'].mean():.4f} {unit_label} (mean)")
        print(f"    Peak: {month_names[glacier_regimes['small'].idxmax()-1]} ({glacier_regimes['small'].max():.4f} {unit_label})")
    
    if 'large' in glacier_regimes:
        print(f"  Large glacier melt: {glacier_regimes['large'].mean():.4f} {unit_label} (mean)")
        print(f"    Peak: {month_names[glacier_regimes['large'].idxmax()-1]} ({glacier_regimes['large'].max():.4f} {unit_label})")
    
    # Return all data
    return {
        'streamflow': streamflow_regime,
        'snowmelt': snowmelt_regime,
        'glacier_small': glacier_regimes.get('small'),
        'glacier_large': glacier_regimes.get('large'),
        'unit': unit_label
    }

#--------------------------------------------------------------------------------ss

def plot_streamflow_with_glogem_icemelt_and_total_snowmelt_regime(config, plot_dirs, validation_start=None, validation_end=None, unit='mm'):
    """
    Plot streamflow regime with GloGEM ice melt and total snowmelt contributions.
    
    Components shown:
    - Observed streamflow
    - Simulated streamflow
    - GloGEM ice melt (glacier area)
    - Total snowmelt = GloGEM snowmelt + HBV snowmelt mass loadings (combined)
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    unit : str, optional
        Unit for display ('mm' for mm/day, 'm3' for m³/s), default is 'mm'
        
    Returns:
    --------
    dict
        Dictionary containing all monthly regime data
    """
    
    gauge_id = config['gauge_id']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"Creating streamflow regime with GloGEM ice melt and total snowmelt for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    print(f"  - Unit: {unit}")
    
    # Load catchment area for unit conversion
    conversion_m3s_to_mm_day = None
    if unit == 'mm':
        config_dir = Path(config['main_dir']) / config['config_dir']
        topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
        catchment_shape_file = topo_dir / "HRU.shp"
        
        try:
            if catchment_shape_file.exists():
                import geopandas as gpd
                hru_gdf = gpd.read_file(catchment_shape_file)
                total_area_km2 = hru_gdf['Area_km2'].sum()
                # Conversion factor: m³/s to mm/day
                conversion_m3s_to_mm_day = 86400 / (total_area_km2 * 1000000) * 1000
                print(f"  - Catchment area: {total_area_km2:.2f} km²")
                print(f"  - Conversion factor (m³/s to mm/day): {conversion_m3s_to_mm_day:.6f}")
            else:
                print(f"ERROR: Catchment shapefile not found: {catchment_shape_file}")
                print(f"  Falling back to m³/s")
                unit = 'm3'
        except Exception as e:
            print(f"ERROR: Could not load catchment area: {e}")
            print(f"  Falling back to m³/s")
            unit = 'm3'
    
    # Set unit label
    unit_label = 'mm/day' if unit == 'mm' else 'm³/s'
    
    # =============================
    # 1. LOAD STREAMFLOW DATA
    # =============================
    
    streamflow_data = load_hydrograph_data(config)
    if streamflow_data is None:
        print("ERROR: Could not load streamflow data")
        return None
    
    # Filter for validation period
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    
    streamflow_mask = (streamflow_data['date'] >= start_date) & (streamflow_data['date'] <= end_date)
    streamflow_filtered = streamflow_data[streamflow_mask].copy()
    
    if len(streamflow_filtered) == 0:
        print(f"ERROR: No streamflow data found for period {validation_start} to {validation_end}")
        return None
    
    # Convert streamflow if needed
    if unit == 'mm' and conversion_m3s_to_mm_day is not None:
        streamflow_filtered['obs_Q_converted'] = streamflow_filtered['obs_Q'] * conversion_m3s_to_mm_day
        streamflow_filtered['sim_Q_converted'] = streamflow_filtered['sim_Q'] * conversion_m3s_to_mm_day
    else:
        streamflow_filtered['obs_Q_converted'] = streamflow_filtered['obs_Q']
        streamflow_filtered['sim_Q_converted'] = streamflow_filtered['sim_Q']
    
    # Calculate monthly regime for streamflow
    streamflow_filtered['month'] = streamflow_filtered['date'].dt.month
    
    streamflow_regime = {}
    if 'obs_Q_converted' in streamflow_filtered.columns:
        streamflow_regime['observed'] = streamflow_filtered.groupby('month')['obs_Q_converted'].mean()
    if 'sim_Q_converted' in streamflow_filtered.columns:
        streamflow_regime['simulated'] = streamflow_filtered.groupby('month')['sim_Q_converted'].mean()
    
    # =============================
    # 2. LOAD GLOGEM DATA
    # =============================
    
    print(f"\n  - Loading GloGEM data...")
    glogem_df = load_glogem_data(config, unit='mm', plot=False)
    
    if glogem_df is None:
        print("ERROR: Could not load GloGEM data")
        return None
    
    # Filter GloGEM data for validation period
    glogem_mask = (glogem_df['date'] >= start_date) & (glogem_df['date'] <= end_date)
    glogem_filtered = glogem_df[glogem_mask].copy()
    
    if len(glogem_filtered) == 0:
        print(f"ERROR: No GloGEM data found for period {validation_start} to {validation_end}")
        return None
    
    # Calculate monthly regime for GloGEM components
    glogem_filtered['month'] = glogem_filtered['date'].dt.month
    
    # Use NORMALIZED (catchment area) values for fair comparison with streamflow
    glogem_icemelt_regime = glogem_filtered.groupby('month')['icemelt_normalized'].mean()
    glogem_snowmelt_regime = glogem_filtered.groupby('month')['snowmelt_normalized'].mean()
    
    print(f"  - GloGEM ice melt mean: {glogem_icemelt_regime.mean():.4f} mm/day")
    print(f"  - GloGEM snowmelt mean: {glogem_snowmelt_regime.mean():.4f} mm/day")
    
    # =============================
    # 3. LOAD HBV SNOWMELT MASS LOADINGS
    # =============================
    
    print(f"\n  - Loading HBV snowmelt mass loadings...")
    hbv_snowmelt_df = load_snowmelt_mass_loadings(config, validation_start, validation_end, unit=unit)
    
    if hbv_snowmelt_df is None:
        print("ERROR: Could not load HBV snowmelt mass loadings")
        return None
    
    # Determine snowmelt column based on unit
    snowmelt_col = 'snowmelt_mm_day' if unit == 'mm' and 'snowmelt_mm_day' in hbv_snowmelt_df.columns else 'snowmelt_m3s'
    
    # Calculate monthly regime for HBV snowmelt
    hbv_snowmelt_df['month'] = hbv_snowmelt_df['date'].dt.month
    hbv_snowmelt_regime = hbv_snowmelt_df.groupby('month')[snowmelt_col].mean()
    
    print(f"  - HBV snowmelt mean: {hbv_snowmelt_regime.mean():.4f} {unit_label}")
    
    # =============================
    # 4. COMBINE SNOWMELT SOURCES
    # =============================
    
    # Total snowmelt = GloGEM snowmelt + HBV snowmelt mass loadings
    total_snowmelt_regime = glogem_snowmelt_regime.add(hbv_snowmelt_regime, fill_value=0)
    
    print(f"  - Total snowmelt mean: {total_snowmelt_regime.mean():.4f} {unit_label}")
    print(f"    (GloGEM: {glogem_snowmelt_regime.mean():.4f} + HBV: {hbv_snowmelt_regime.mean():.4f})")
    
    # =============================
    # 5. CREATE PLOT
    # =============================
    
    plt.figure(figsize=(14, 8))
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot filled polygons FIRST (bottom layer, in order of magnitude)
    # Plot total snowmelt as filled polygon - light blue
    plt.fill_between(months, 0, total_snowmelt_regime.values, 
                     color='#B3D9FF', alpha=0.7, label='Total Snowmelt (GloGEM+HBV)', 
                     zorder=1, edgecolor='#6DB3F2', linewidth=1.5)
    
    # Plot GloGEM ice melt as filled polygon - grey/brown
    plt.fill_between(months, 0, glogem_icemelt_regime.values, 
                     color='#C17817', alpha=0.6, label='GloGEM Ice Melt', 
                     zorder=2, edgecolor='#8B5A00', linewidth=1.5)
    
    # Plot observed streamflow (line without markers)
    if 'observed' in streamflow_regime:
        plt.plot(months, streamflow_regime['observed'].values, 'k-', 
                linewidth=3, label='Observed Streamflow', zorder=4)
    
    # Plot simulated streamflow (dashed line without markers)
    if 'simulated' in streamflow_regime:
        plt.plot(months, streamflow_regime['simulated'].values, 'C0--', 
                linewidth=2.5, label='Simulated Streamflow', zorder=3)
    
    # Formatting
    plt.xlabel('Month', fontsize=14, fontweight='bold')
    plt.ylabel(f'Discharge ({unit_label})', fontsize=14, fontweight='bold')
    plt.title(f'Streamflow Regime with GloGEM Ice Melt and Total Snowmelt\nCatchment {gauge_id}', 
             fontsize=16, fontweight='bold')
    plt.xticks(months, month_names, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.legend(fontsize=12, loc='best')
    
    # Add statistics text box
    stats_lines = []
    if 'observed' in streamflow_regime:
        stats_lines.append(f"Obs. Streamflow: {streamflow_regime['observed'].mean():.4f} {unit_label} (mean)")
    if 'simulated' in streamflow_regime:
        stats_lines.append(f"Sim. Streamflow: {streamflow_regime['simulated'].mean():.4f} {unit_label} (mean)")
    stats_lines.append(f"Total Snowmelt: {total_snowmelt_regime.mean():.4f} {unit_label} (mean)")
    stats_lines.append(f"  - GloGEM: {glogem_snowmelt_regime.mean():.4f} {unit_label}")
    stats_lines.append(f"  - HBV: {hbv_snowmelt_regime.mean():.4f} {unit_label}")
    stats_lines.append(f"GloGEM Ice Melt: {glogem_icemelt_regime.mean():.4f} {unit_label} (mean)")
    
    stats_text = '\n'.join(stats_lines)
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'streamflow_glogem_icemelt_total_snowmelt_regime_{unit}_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved streamflow regime plot to: {save_path}")
    plt.show()
    
    # =============================
    # 6. PRINT SUMMARY
    # =============================
    
    print(f"\n{'='*60}")
    print(f"STREAMFLOW REGIME WITH GLOGEM AND TOTAL SNOWMELT SUMMARY")
    print(f"{'='*60}")
    print(f"Catchment: {gauge_id}")
    print(f"Period: {validation_start} to {validation_end}")
    print(f"Unit: {unit_label}")
    
    if 'observed' in streamflow_regime:
        print(f"\nObserved Streamflow:")
        print(f"  Mean: {streamflow_regime['observed'].mean():.4f} {unit_label}")
        print(f"  Peak: {month_names[streamflow_regime['observed'].idxmax()-1]} ({streamflow_regime['observed'].max():.4f} {unit_label})")
    
    if 'simulated' in streamflow_regime:
        print(f"\nSimulated Streamflow:")
        print(f"  Mean: {streamflow_regime['simulated'].mean():.4f} {unit_label}")
        print(f"  Peak: {month_names[streamflow_regime['simulated'].idxmax()-1]} ({streamflow_regime['simulated'].max():.4f} {unit_label})")
    
    print(f"\nGloGEM Ice Melt:")
    print(f"  Mean: {glogem_icemelt_regime.mean():.4f} {unit_label}")
    print(f"  Peak: {month_names[glogem_icemelt_regime.idxmax()-1]} ({glogem_icemelt_regime.max():.4f} {unit_label})")
    
    print(f"\nTotal Snowmelt (GloGEM + HBV):")
    print(f"  Mean: {total_snowmelt_regime.mean():.4f} {unit_label}")
    print(f"  Peak: {month_names[total_snowmelt_regime.idxmax()-1]} ({total_snowmelt_regime.max():.4f} {unit_label})")
    print(f"  Components:")
    print(f"    GloGEM snowmelt: {glogem_snowmelt_regime.mean():.4f} {unit_label} ({(glogem_snowmelt_regime.mean()/total_snowmelt_regime.mean()*100):.1f}%)")
    print(f"    HBV snowmelt: {hbv_snowmelt_regime.mean():.4f} {unit_label} ({(hbv_snowmelt_regime.mean()/total_snowmelt_regime.mean()*100):.1f}%)")
    
    # Calculate contribution percentages (relative to simulated streamflow)
    if 'simulated' in streamflow_regime:
        sim_mean = streamflow_regime['simulated'].mean()
        icemelt_pct = (glogem_icemelt_regime.mean() / sim_mean) * 100
        snowmelt_pct = (total_snowmelt_regime.mean() / sim_mean) * 100
        
        print(f"\nContributions to Simulated Streamflow:")
        print(f"  Ice melt: {icemelt_pct:.1f}%")
        print(f"  Total snowmelt: {snowmelt_pct:.1f}%")
        print(f"  Combined melt: {icemelt_pct + snowmelt_pct:.1f}%")
    
    print(f"{'='*60}\n")
    
    # Return all data
    return {
        'streamflow': streamflow_regime,
        'glogem_icemelt': glogem_icemelt_regime,
        'glogem_snowmelt': glogem_snowmelt_regime,
        'hbv_snowmelt': hbv_snowmelt_regime,
        'total_snowmelt': total_snowmelt_regime,
        'unit': unit_label,
        'save_path': save_path
    }

#--------------------------------------------------------------------------------

def load_rainfall_mass_loadings(config, validation_start=None, validation_end=None):
    """
    Load rainfall mass loadings data from Raven output in m³/s and convert to mm/day.
    
    The file contains direct discharge values in m³/s from the RAIN tracer at the catchment outlet.
    We convert to mm/day for consistency with other data sources.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing date and rainfall in mm/day
    """
    
    config_dir = Path(config['main_dir']) / config['config_dir']
    gauge_id = config['gauge_id']
    model_type = config['model_type']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', config.get('start_date', '2000-01-01'))
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"Loading rainfall mass loadings for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    
    # =============================
    # 1. LOAD CATCHMENT AREA FOR UNIT CONVERSION
    # =============================
    
    topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
    catchment_shape_file = topo_dir / "HRU.shp"
    
    try:
        if catchment_shape_file.exists():
            import geopandas as gpd
            hru_gdf = gpd.read_file(catchment_shape_file)
            total_area_km2 = hru_gdf['Area_km2'].sum()
            # Conversion factor: m³/s to mm/day
            conversion_m3s_to_mm_day = 86400 / (total_area_km2 * 1000000) * 1000
            print(f"  - Catchment area: {total_area_km2:.2f} km²")
            print(f"  - Conversion factor (m³/s to mm/day): {conversion_m3s_to_mm_day:.6f}")
        else:
            print(f"ERROR: Catchment shapefile not found: {catchment_shape_file}")
            return None
    except Exception as e:
        print(f"ERROR: Could not load catchment area: {e}")
        return None
    
    # =============================
    # 2. LOAD RAINFALL DATA FILE
    # =============================
    
    # Define file path
    rainfall_file = config_dir / f"catchment_{gauge_id}" / model_type / "output" / f"{gauge_id}_{model_type}_RAINMassLoadings.csv"
    
    # Check if file exists
    if not rainfall_file.exists():
        print(f"ERROR: Rainfall mass loadings file not found: {rainfall_file}")
        return None
    
    try:
        # Read mass loadings file
        df = pd.read_csv(rainfall_file)
        print(f"  - Loaded mass loadings: {df.shape}")
        print(f"  - Columns: {df.columns.tolist()}")
        
        # Parse dates
        if 'date' not in df.columns:
            print(f"ERROR: 'date' column not found in mass loadings file")
            return None
        df['date'] = pd.to_datetime(df['date'])
        
        # Find the gauge column - looking for m3/s format
        gauge_col = f"{gauge_id} m3/s"
        
        if gauge_col not in df.columns:
            print(f"ERROR: Column '{gauge_col}' not found in mass loadings file")
            print(f"  Available columns: {df.columns.tolist()}")
            return None
        
        # ✅ CONVERT from m³/s to mm/day
        df['rainfall_m3s'] = df[gauge_col]
        df['rainfall_mm_day'] = df['rainfall_m3s'] * conversion_m3s_to_mm_day
        
        print(f"  - Successfully loaded and converted rainfall data")
        print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Filter for validation period
        start_date = pd.to_datetime(validation_start)
        end_date = pd.to_datetime(validation_end)
        
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        df_filtered = df[mask].copy()
        
        if len(df_filtered) == 0:
            print(f"ERROR: No data found for period {validation_start} to {validation_end}")
            return None
        
        # Count statistics
        zero_count = (df_filtered['rainfall_mm_day'] == 0).sum()
        nonzero_count = (df_filtered['rainfall_mm_day'] > 0).sum()
        
        print(f"  - Filtered to {len(df_filtered)} records")
        print(f"  - Zero rainfall days: {zero_count}")
        print(f"  - Non-zero rainfall days: {nonzero_count}")
        print(f"  - Mean rainfall: {df_filtered['rainfall_mm_day'].mean():.4f} mm/day")
        print(f"  - Max rainfall: {df_filtered['rainfall_mm_day'].max():.4f} mm/day")
        print(f"  - Sample values (first 5 days):")
        for idx, row in df_filtered.head().iterrows():
            q = row['rainfall_mm_day']
            if q == 0:
                print(f"      {row['date'].date()}: 0.0000 mm/day (no rainfall)")
            else:
                print(f"      {row['date'].date()}: {q:.4f} mm/day")
        
        # Keep only date and rainfall columns (in mm/day)
        result_df = df_filtered[['date', 'rainfall_mm_day']].copy()
        
        print(f"  - Final valid records: {len(result_df)}")
        
        return result_df
        
    except Exception as e:
        print(f"ERROR: Failed to load rainfall mass loadings: {e}")
        import traceback
        traceback.print_exc()
        return None

#--------------------------------------------------------------------------------

def plot_comprehensive_annual_water_balance(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Create comprehensive annual water balance analysis combining all sources.
    
    Components:
    - Observed and simulated streamflow
    - Ice melt from GloGEM
    - Snowmelt from GloGEM + HBV model
    - Rainfall from GloGEM + HBV model
    
    All components are converted to consistent units (mm/year) for comparison.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    pandas.DataFrame
        Annual water balance data with all components
    """
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    gauge_id = config['gauge_id']
    coupled = config.get('coupled', False)
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE ANNUAL WATER BALANCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Catchment: {gauge_id}")
    print(f"Period: {validation_start} to {validation_end}")
    print(f"Coupled mode: {coupled}")
    print(f"{'='*60}\n")
    
    # =============================
    # 1. LOAD CATCHMENT AREA FOR UNIT CONVERSION
    # =============================
    
    config_dir = Path(config['main_dir']) / config['config_dir']
    topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
    catchment_shape_file = topo_dir / "HRU.shp"
    
    try:
        if catchment_shape_file.exists():
            import geopandas as gpd
            hru_gdf = gpd.read_file(catchment_shape_file)
            total_area_km2 = hru_gdf['Area_km2'].sum()
            # Conversion factor: m³/s to mm/day
            conversion_m3s_to_mm_day = 86400 / (total_area_km2 * 1000000) * 1000
            print(f"1. Catchment area: {total_area_km2:.2f} km²")
            print(f"   Conversion factor (m³/s to mm/day): {conversion_m3s_to_mm_day:.6f}")
        else:
            print(f"ERROR: Catchment shapefile not found: {catchment_shape_file}")
            return None
    except Exception as e:
        print(f"ERROR: Could not load catchment area: {e}")
        return None
    
    # =============================
    # 2. LOAD STREAMFLOW DATA (m³/s -> mm/year)
    # =============================
    
    print(f"\n2. Loading streamflow data...")
    streamflow_data = load_hydrograph_data(config)
    if streamflow_data is None:
        print("   ERROR: Could not load streamflow data")
        return None
    
    # Filter for validation period
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    
    streamflow_mask = (streamflow_data['date'] >= start_date) & (streamflow_data['date'] <= end_date)
    streamflow_filtered = streamflow_data[streamflow_mask].copy()
    
    if len(streamflow_filtered) == 0:
        print(f"   ERROR: No streamflow data found for period")
        return None
    
    # Convert streamflow from m³/s to mm/day
    streamflow_filtered['obs_Q_mm_day'] = streamflow_filtered['obs_Q'] * conversion_m3s_to_mm_day
    streamflow_filtered['sim_Q_mm_day'] = streamflow_filtered['sim_Q'] * conversion_m3s_to_mm_day
    
    # Calculate annual sums (mm/year)
    streamflow_filtered['year'] = streamflow_filtered['date'].dt.year
    streamflow_annual = streamflow_filtered.groupby('year').agg({
        'obs_Q_mm_day': 'sum',
        'sim_Q_mm_day': 'sum'
    }).reset_index()
    streamflow_annual.columns = ['year', 'obs_streamflow_mm', 'sim_streamflow_mm']
    
    print(f"   ✓ Loaded streamflow data: {len(streamflow_annual)} years")
    print(f"     Mean annual observed streamflow: {streamflow_annual['obs_streamflow_mm'].mean():.1f} mm/year")
    print(f"     Mean annual simulated streamflow: {streamflow_annual['sim_streamflow_mm'].mean():.1f} mm/year")
    
    # =============================
    # 3. LOAD GLOGEM DATA (already in mm/day -> mm/year)
    # =============================

    print(f"\n3. Loading GloGEM data...")
    glogem_df = load_glogem_data(config, unit='mm', plot=False)

    if glogem_df is not None:
        # Filter for validation period
        glogem_mask = (glogem_df['date'] >= start_date) & (glogem_df['date'] <= end_date)
        glogem_filtered = glogem_df[glogem_mask].copy()
        
        if len(glogem_filtered) > 0:
            # ✅ FIX: Use GLACIER AREA values (not normalized catchment values)
            # Calculate annual sums (mm/year) - data is already in mm/day over glacier area
            glogem_filtered['year'] = glogem_filtered['date'].dt.year
            glogem_annual = glogem_filtered.groupby('year').agg({
                'icemelt_normalized': 'sum',        
                'snowmelt_normalized': 'sum',       
                'rainfall_normalized': 'sum'       
            }).reset_index()
            glogem_annual.columns = ['year', 'glogem_icemelt_mm', 'glogem_snowmelt_mm', 'glogem_rainfall_mm']
            
            print(f"   ✓ Loaded GloGEM data: {len(glogem_annual)} years")
            print(f"     Mean annual GloGEM ice melt (glacier area): {glogem_annual['glogem_icemelt_mm'].mean():.1f} mm/year")
            print(f"     Mean annual GloGEM snowmelt (glacier area): {glogem_annual['glogem_snowmelt_mm'].mean():.1f} mm/year")
            print(f"     Mean annual GloGEM rainfall (glacier area): {glogem_annual['glogem_rainfall_mm'].mean():.1f} mm/year")
        else:
            print(f"   WARNING: No GloGEM data found for validation period")
            glogem_annual = None
    else:
        print(f"   WARNING: Could not load GloGEM data")
        glogem_annual = None
    
    # =============================
    # 4. LOAD HBV SNOWMELT DATA (m³/s -> mm/year)
    # =============================
    
    print(f"\n4. Loading HBV snowmelt data...")
    snowmelt_df = load_snowmelt_mass_loadings(config, validation_start, validation_end)
    
    if snowmelt_df is not None:
        # Convert from m³/s to mm/day
        snowmelt_df['snowmelt_mm_day'] = snowmelt_df['snowmelt_m3s'] * conversion_m3s_to_mm_day
        
        # Calculate annual sums (mm/year)
        snowmelt_df['year'] = snowmelt_df['date'].dt.year
        snowmelt_annual = snowmelt_df.groupby('year')['snowmelt_mm_day'].sum().reset_index()
        snowmelt_annual.columns = ['year', 'hbv_snowmelt_mm']
        
        print(f"   ✓ Loaded HBV snowmelt data: {len(snowmelt_annual)} years")
        print(f"     Mean annual HBV snowmelt: {snowmelt_annual['hbv_snowmelt_mm'].mean():.1f} mm/year")
    else:
        print(f"   WARNING: Could not load HBV snowmelt data")
        snowmelt_annual = None
    
    # =============================
    # 5. LOAD HBV RAINFALL DATA (mm/day -> mm/year)
    # =============================
    
    print(f"\n5. Loading HBV rainfall data...")
    rainfall_df = load_rainfall_mass_loadings(config, validation_start, validation_end)
    
    if rainfall_df is not None:
        # Data is already in mm/day, just sum to get mm/year
        rainfall_df['year'] = rainfall_df['date'].dt.year
        rainfall_annual = rainfall_df.groupby('year')['rainfall_mm_day'].sum().reset_index()
        rainfall_annual.columns = ['year', 'hbv_rainfall_mm']
        
        print(f"   ✓ Loaded HBV rainfall data: {len(rainfall_annual)} years")
        print(f"     Mean annual HBV rainfall: {rainfall_annual['hbv_rainfall_mm'].mean():.1f} mm/year")
    else:
        print(f"   WARNING: Could not load rainfall mass loadings file")
        print(f"   Trying alternative: RAINFALL_Daily_Average_ByHRUGroup.csv (NO_GLACIER column)...")
        
        # Fallback: Load from ByHRUGroup file (same as plot_precipitation_partitioning)
        try:
            rainfall_hrugroup_df = load_forcing_by_hrugroup(config, 'RAINFALL')
            
            if rainfall_hrugroup_df is not None and 'NO_GLACIER' in rainfall_hrugroup_df.columns:
                # Filter for validation period
                rainfall_mask = (rainfall_hrugroup_df['date'] >= start_date) & (rainfall_hrugroup_df['date'] <= end_date)
                rainfall_filtered = rainfall_hrugroup_df[rainfall_mask].copy()
                
                # Load HRU shapefile to calculate area scaling factor
                hru_shapefile = topo_dir / "HRU.shp"
                if hru_shapefile.exists():
                    import geopandas as gpd
                    hru_gdf = gpd.read_file(hru_shapefile)
                    
                    # Calculate non-glacier area fraction
                    if 'Landuse_Cl' in hru_gdf.columns:
                        glacier_area = hru_gdf[hru_gdf['Landuse_Cl'].isin([7, 8])]['Area_km2'].sum()
                        non_glacier_area = hru_gdf[~hru_gdf['Landuse_Cl'].isin([7, 8])]['Area_km2'].sum()
                    else:
                        non_glacier_area = total_area_km2
                        glacier_area = 0.0
                    
                    area_fraction = non_glacier_area / total_area_km2 if total_area_km2 > 0 else 1.0
                    
                    # Scale rainfall from non-glacier areas to catchment
                    rainfall_filtered['rainfall_mm_day'] = rainfall_filtered['NO_GLACIER'] * area_fraction
                    
                    # Calculate annual sums
                    rainfall_filtered['year'] = rainfall_filtered['date'].dt.year
                    rainfall_annual = rainfall_filtered.groupby('year')['rainfall_mm_day'].sum().reset_index()
                    rainfall_annual.columns = ['year', 'hbv_rainfall_mm']
                    
                    print(f"   ✓ Loaded rainfall from ByHRUGroup file (NO_GLACIER column)")
                    print(f"     Non-glacier area fraction: {area_fraction*100:.1f}%")
                    print(f"     Mean annual HBV rainfall: {rainfall_annual['hbv_rainfall_mm'].mean():.1f} mm/year")
                else:
                    print(f"   WARNING: Could not load HRU shapefile for area scaling")
                    rainfall_annual = None
            else:
                print(f"   WARNING: Could not load rainfall from ByHRUGroup file")
                rainfall_annual = None
        except Exception as e:
            print(f"   WARNING: Failed to load rainfall from ByHRUGroup file: {e}")
            rainfall_annual = None
    
    # =============================
    # 6. COMBINE ALL DATA INTO SINGLE DATAFRAME
    # =============================
    
    print(f"\n6. Combining all data sources...")
    
    # Start with streamflow data
    annual_balance = streamflow_annual.copy()
    
    # Merge GloGEM data if available
    if glogem_annual is not None:
        annual_balance = pd.merge(annual_balance, glogem_annual, on='year', how='inner')
    
    # Merge HBV snowmelt if available
    if snowmelt_annual is not None:
        annual_balance = pd.merge(annual_balance, snowmelt_annual, on='year', how='inner')
    
    # Merge HBV rainfall if available
    if rainfall_annual is not None:
        annual_balance = pd.merge(annual_balance, rainfall_annual, on='year', how='inner')
    
    if len(annual_balance) == 0:
        print(f"   ERROR: No overlapping years found between datasets")
        return None
    
    # =============================
    # 7. CALCULATE COMBINED COMPONENTS
    # =============================
    
    print(f"\n7. Calculating combined components...")
    
    # Total ice melt (only GloGEM for now)
    annual_balance['total_icemelt_mm'] = annual_balance.get('glogem_icemelt_mm', 0)
    
    # Total snowmelt (GloGEM + HBV)
    annual_balance['total_snowmelt_mm'] = (
        annual_balance.get('glogem_snowmelt_mm', 0) + 
        annual_balance.get('hbv_snowmelt_mm', 0)
    )
    
    # Total rainfall (GloGEM + HBV)
    annual_balance['total_rainfall_mm'] = (
        annual_balance.get('glogem_rainfall_mm', 0) + 
        annual_balance.get('hbv_rainfall_mm', 0)
    )
    
    # Total precipitation (rainfall + snowfall, if we had it)
    annual_balance['total_precipitation_mm'] = annual_balance['total_rainfall_mm']
    
    # Total input (precipitation + ice melt)
    annual_balance['total_input_mm'] = (
        annual_balance['total_precipitation_mm'] + 
        annual_balance['total_icemelt_mm'] +
        annual_balance['total_snowmelt_mm']
    )
    
    print(f"   ✓ Combined data for {len(annual_balance)} years")
    print(f"\n   Mean annual components (mm/year):")
    print(f"     Total ice melt: {annual_balance['total_icemelt_mm'].mean():.1f}")
    print(f"     Total snowmelt: {annual_balance['total_snowmelt_mm'].mean():.1f}")
    print(f"     Total rainfall: {annual_balance['total_rainfall_mm'].mean():.1f}")
    print(f"     Total input: {annual_balance['total_input_mm'].mean():.1f}")
    print(f"     Observed streamflow: {annual_balance['obs_streamflow_mm'].mean():.1f}")
    print(f"     Simulated streamflow: {annual_balance['sim_streamflow_mm'].mean():.1f}")
    
    # =============================
    # 8. CREATE BAR PLOT
    # =============================
    
    print(f"\n8. Creating bar plot...")
    
    fig, ax = plt.subplots(figsize=(max(14, len(annual_balance) * 1.2), 10))
    
    x = np.arange(len(annual_balance))
    width = 0.15  # Width of bars
    
    # Plot bars for each component
    bars1 = ax.bar(x - 2.5*width, annual_balance['total_rainfall_mm'], width, 
                   label='Rainfall', color='navy', alpha=0.8, edgecolor='black', linewidth=1)
    
    bars2 = ax.bar(x - 1.5*width, annual_balance['total_snowmelt_mm'], width, 
                   label='Snowmelt', color='lightblue', alpha=0.8, edgecolor='black', linewidth=1)
    
    bars3 = ax.bar(x - 0.5*width, annual_balance['total_icemelt_mm'], width, 
                   label='Ice Melt (GloGEM)', color='grey', alpha=0.8, edgecolor='black', linewidth=1)
    
    bars4 = ax.bar(x + 0.5*width, annual_balance['total_input_mm'], width, 
                   label='Total Input', color='darkgreen', alpha=0.8, edgecolor='black', linewidth=1)
    
    bars5 = ax.bar(x + 1.5*width, annual_balance['obs_streamflow_mm'], width, 
                   label='Obs. Streamflow', color='black', alpha=0.8, edgecolor='white', linewidth=1)
    
    bars6 = ax.bar(x + 2.5*width, annual_balance['sim_streamflow_mm'], width, 
                   label='Sim. Streamflow', color='orange', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Formatting
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Annual Sum (mm/year)', fontsize=14, fontweight='bold')
    ax.set_title(f'Comprehensive Annual Water Balance - Catchment {gauge_id}\n'
                f'Period: {validation_start} to {validation_end} ({"Coupled" if coupled else "Uncoupled"})', 
                fontsize=16, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(year)) for year in annual_balance['year']], rotation=45)
    ax.legend(fontsize=11, loc='upper left', ncol=2)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save bar plot
    save_path_bars = plot_dirs['contributions'] / f'comprehensive_annual_water_balance_bars_{gauge_id}.png'
    plt.savefig(save_path_bars, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved bar plot to: {save_path_bars}")
    plt.show()
    
    # =============================
    # 9. CREATE STACKED AREA PLOT
    # =============================

    print(f"\n9. Creating stacked area plot...")

    fig, ax = plt.subplots(figsize=(14, 8))

    # ✅ FIX: Insert NaN values for missing years to create gaps in the plot
    # Create complete year range
    year_min = annual_balance['year'].min()
    year_max = annual_balance['year'].max()
    all_years = np.arange(year_min, year_max + 1)

    # Reindex dataframe to include all years, filling missing years with NaN
    annual_balance_complete = annual_balance.set_index('year').reindex(all_years).reset_index()
    annual_balance_complete.columns = ['year'] + list(annual_balance.columns[1:])

    # Create stacked area plot for inputs (NaN values will create gaps)
    ax.fill_between(annual_balance_complete['year'], 0, annual_balance_complete['total_rainfall_mm'], 
                    label='Rainfall', color='navy', alpha=0.6)

    ax.fill_between(annual_balance_complete['year'], annual_balance_complete['total_rainfall_mm'], 
                    annual_balance_complete['total_rainfall_mm'] + annual_balance_complete['total_snowmelt_mm'], 
                    label='Snowmelt', color='lightblue', alpha=0.6)

    ax.fill_between(annual_balance_complete['year'], 
                    annual_balance_complete['total_rainfall_mm'] + annual_balance_complete['total_snowmelt_mm'], 
                    annual_balance_complete['total_rainfall_mm'] + annual_balance_complete['total_snowmelt_mm'] + annual_balance_complete['total_icemelt_mm'], 
                    label='Ice Melt', color='grey', alpha=0.6)

    # Plot observed and simulated streamflow as lines (NaN values will create gaps)
    ax.plot(annual_balance_complete['year'], annual_balance_complete['obs_streamflow_mm'], 
        'k-', linewidth=3, label='Obs. Streamflow', zorder=10)

    ax.plot(annual_balance_complete['year'], annual_balance_complete['sim_streamflow_mm'], 
        'orange', linewidth=2.5, linestyle='--', label='Sim. Streamflow', zorder=9)

    # Formatting
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Annual Sum (mm/year)', fontsize=14, fontweight='bold')
    ax.set_title(f'Annual Water Balance Components (Stacked) - Catchment {gauge_id}\n'
                f'Period: {validation_start} to {validation_end}', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, zorder=0)

    # ✅ FIX: Set x-axis limits to show full range including gaps
    ax.set_xlim(year_min - 0.5, year_max + 0.5)

    plt.tight_layout()
    
    # =============================
    # 10. CREATE WATER BALANCE RATIO PLOT
    # =============================

    print(f"\n10. Creating water balance ratio plot...")

    # ✅ FIX: Calculate melt contribution ratios for ALL years (use full dataset)
    print(f"  - Calculating melt contributions for all {len(annual_balance)} years")

    # Calculate contribution fractions relative to SIMULATED STREAMFLOW (for all years)
    annual_balance['icemelt_fraction'] = annual_balance['total_icemelt_mm'] / annual_balance['sim_streamflow_mm']
    annual_balance['snowmelt_fraction'] = annual_balance['total_snowmelt_mm'] / annual_balance['sim_streamflow_mm']
    annual_balance['rainfall_fraction'] = annual_balance['total_rainfall_mm'] / annual_balance['sim_streamflow_mm']

    # Remove any infinite values that might result from division by zero
    annual_balance = annual_balance.replace([np.inf, -np.inf], np.nan)

    # ✅ FIX: For RUNOFF RATIO ONLY, filter years with sufficient observed data
    # Check data availability for each year
    min_data_fraction = 0.8  # Require 80% of days

    # Get streamflow data availability
    streamflow_filtered_copy = streamflow_filtered.copy()
    streamflow_filtered_copy['days_count'] = 1

    yearly_data_availability = streamflow_filtered_copy.groupby('year').agg({
        'obs_Q': 'count',
        'days_count': 'sum'
    }).reset_index()

    yearly_data_availability['days_in_year'] = yearly_data_availability['year'].apply(
        lambda y: 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
    )

    yearly_data_availability['data_fraction'] = yearly_data_availability['obs_Q'] / yearly_data_availability['days_in_year']

    # Get list of years with sufficient observed data for runoff ratio
    valid_years_for_runoff_ratio = yearly_data_availability[
        yearly_data_availability['data_fraction'] >= min_data_fraction
    ]['year'].values

    print(f"  - Years with >{min_data_fraction*100:.0f}% observed data (for runoff ratio): {len(valid_years_for_runoff_ratio)}/{len(annual_balance)}")

    # ✅ Create a filtered dataset ONLY for runoff ratio calculation
    annual_balance_runoff_ratio = annual_balance[annual_balance['year'].isin(valid_years_for_runoff_ratio)].copy()

    # Calculate runoff ratio only for years with sufficient observed data
    if len(annual_balance_runoff_ratio) > 0:
        annual_balance_runoff_ratio['runoff_ratio'] = annual_balance_runoff_ratio['obs_streamflow_mm'] / annual_balance_runoff_ratio['sim_streamflow_mm']
        annual_balance_runoff_ratio = annual_balance_runoff_ratio.replace([np.inf, -np.inf], np.nan)
        annual_balance_runoff_ratio = annual_balance_runoff_ratio.dropna(subset=['runoff_ratio'])

    # ✅ Create the plots using DIFFERENT datasets for different purposes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ==============================================
    # LEFT PLOT: Runoff Ratio (ONLY years with obs data)
    # ==============================================

    if len(annual_balance_runoff_ratio) > 0:
        ax1.plot(annual_balance_runoff_ratio['year'], annual_balance_runoff_ratio['runoff_ratio'], 'o-', 
                linewidth=2, markersize=8, color='darkblue')
        ax1.axhline(y=annual_balance_runoff_ratio['runoff_ratio'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {annual_balance_runoff_ratio["runoff_ratio"].mean():.3f}')
        ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Runoff Ratio (Obs Q / Sim Q)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Runoff Ratio Over Time\n({len(annual_balance_runoff_ratio)} years with >{min_data_fraction*100:.0f}% obs data)', 
                    fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No years with sufficient observed data', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=14)
        ax1.set_title('Runoff Ratio (No Sufficient Data)', fontsize=14, fontweight='bold')

    # ==============================================
    # RIGHT PLOT: Melt Contributions (ALL years)
    # ==============================================

    # ✅ Use FULL dataset for melt contributions (all years with simulated data)
    ax2.plot(annual_balance['year'], annual_balance['icemelt_fraction'] * 100, 'o-', 
            linewidth=2, markersize=6, color='grey', label='Ice Melt (% of Sim. Streamflow)')
    ax2.plot(annual_balance['year'], annual_balance['snowmelt_fraction'] * 100, 's-', 
            linewidth=2, markersize=6, color='lightblue', label='Snowmelt (% of Sim. Streamflow)')
    ax2.plot(annual_balance['year'], annual_balance['rainfall_fraction'] * 100, '^-', 
            linewidth=2, markersize=6, color='navy', label='Rainfall (% of Sim. Streamflow)')

    # Add 100% reference line
    ax2.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5, label='100%')

    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage of Simulated Streamflow (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Melt Contributions to Simulated Streamflow\n(All {len(annual_balance)} years)', 
                fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save ratio plot
    save_path_ratios = plot_dirs['contributions'] / f'comprehensive_annual_water_balance_ratios_{gauge_id}.png'
    plt.savefig(save_path_ratios, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved ratio plot to: {save_path_ratios}")
    plt.show()


    # =============================
    # 11. SAVE DATAFRAME TO CSV
    # =============================

    print(f"\n11. Saving data to CSV...")

    results_dir = config_dir / f"catchment_{gauge_id}" / config['model_type'] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f'comprehensive_annual_water_balance_{gauge_id}.csv'
    annual_balance.to_csv(csv_path, index=False)
    print(f"   ✓ Saved data to: {csv_path}")

    # =============================
    # 12. PRINT SUMMARY STATISTICS
    # =============================

    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Period: {int(annual_balance['year'].min())} - {int(annual_balance['year'].max())}")
    print(f"Number of years: {len(annual_balance)}")
    print(f"\nMean annual values (mm/year):")
    print(f"  Rainfall (total): {annual_balance['total_rainfall_mm'].mean():.1f} ± {annual_balance['total_rainfall_mm'].std():.1f}")
    if 'glogem_rainfall_mm' in annual_balance.columns:
        print(f"    - GloGEM: {annual_balance['glogem_rainfall_mm'].mean():.1f}")
    if 'hbv_rainfall_mm' in annual_balance.columns:
        print(f"    - HBV: {annual_balance['hbv_rainfall_mm'].mean():.1f}")

    print(f"  Snowmelt (total): {annual_balance['total_snowmelt_mm'].mean():.1f} ± {annual_balance['total_snowmelt_mm'].std():.1f}")
    if 'glogem_snowmelt_mm' in annual_balance.columns:
        print(f"    - GloGEM: {annual_balance['glogem_snowmelt_mm'].mean():.1f}")
    if 'hbv_snowmelt_mm' in annual_balance.columns:
        print(f"    - HBV: {annual_balance['hbv_snowmelt_mm'].mean():.1f}")

    print(f"  Ice melt (GloGEM): {annual_balance['total_icemelt_mm'].mean():.1f} ± {annual_balance['total_icemelt_mm'].std():.1f}")
    print(f"  Total input: {annual_balance['total_input_mm'].mean():.1f} ± {annual_balance['total_input_mm'].std():.1f}")
    print(f"  Observed streamflow: {annual_balance['obs_streamflow_mm'].mean():.1f} ± {annual_balance['obs_streamflow_mm'].std():.1f}")
    print(f"  Simulated streamflow: {annual_balance['sim_streamflow_mm'].mean():.1f} ± {annual_balance['sim_streamflow_mm'].std():.1f}")

    # ✅ FIX: Print melt contributions for ALL years
    print(f"\n✅ Mean contributions to simulated streamflow (all {len(annual_balance)} years):")

    # Calculate statistics on the contribution fractions (removing inf/nan)
    icemelt_frac_clean = annual_balance['icemelt_fraction'].replace([np.inf, -np.inf], np.nan).dropna()
    snowmelt_frac_clean = annual_balance['snowmelt_fraction'].replace([np.inf, -np.inf], np.nan).dropna()
    rainfall_frac_clean = annual_balance['rainfall_fraction'].replace([np.inf, -np.inf], np.nan).dropna()

    if len(icemelt_frac_clean) > 0:
        print(f"  Ice melt: {icemelt_frac_clean.mean()*100:.1f}% ± {icemelt_frac_clean.std()*100:.1f}%")
    if len(snowmelt_frac_clean) > 0:
        print(f"  Snowmelt: {snowmelt_frac_clean.mean()*100:.1f}% ± {snowmelt_frac_clean.std()*100:.1f}%")
    if len(rainfall_frac_clean) > 0:
        print(f"  Rainfall: {rainfall_frac_clean.mean()*100:.1f}% ± {rainfall_frac_clean.std()*100:.1f}%")
    if len(icemelt_frac_clean) > 0 and len(snowmelt_frac_clean) > 0:
        print(f"  Total melt (ice+snow): {(icemelt_frac_clean.mean() + snowmelt_frac_clean.mean())*100:.1f}%")

    # ✅ Print runoff ratio ONLY for years with sufficient observed data
    if len(annual_balance_runoff_ratio) > 0:
        print(f"\n✅ Runoff ratio (Obs/Sim) - {len(annual_balance_runoff_ratio)} years with >{min_data_fraction*100:.0f}% obs data:")
        print(f"  Mean: {annual_balance_runoff_ratio['runoff_ratio'].mean():.3f} ± {annual_balance_runoff_ratio['runoff_ratio'].std():.3f}")
    else:
        print(f"\n⚠️  Runoff ratio: No years with sufficient observed data (>{min_data_fraction*100:.0f}%)")

    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE WATER BALANCE ANALYSIS COMPLETE")
    print(f"{'='*60}\n")

#--------------------------------------------------------------------------------
################################### parameter ###################################
#--------------------------------------------------------------------------------

def load_parameter_values(config, top_n=100):
    """
    Load parameter values from model configuration for analysis.
    Selects the best top_n parameter sets based on objective function.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    top_n : int
        Number of top parameter sets to select
    
    Returns:
    --------
    dict
        Dictionary containing parameter values and statistics
    """
    
    gauge_id = config['gauge_id']
    model_type = config['model_type']
    config_dir = Path(config['main_dir']) / config['config_dir']
    
    print(f"\n{'-'*40}\nAnalyzing parameters for {gauge_id}\n{'-'*40}")
    
    # Build path to model output directory
    model_dir = config_dir / f"catchment_{gauge_id}" / model_type / "output"
    
    # Look for calibration results files
    calibration_files = list(model_dir.glob(f"calibration_results_{gauge_id}_{model_type}_*.csv"))
    
    if not calibration_files:
        # Try alternative file patterns
        calibration_files = list(model_dir.glob(f"*calibration*.csv"))
        if not calibration_files:
            calibration_files = list(model_dir.glob(f"*parameter*.csv"))
    
    if not calibration_files:
        print(f"No calibration files found in {model_dir}")
        return None
    
    # Use the first file if multiple exist
    cal_file = calibration_files[0]
    print(f"Found calibration file: {cal_file}")
    
    try:
        df = pd.read_csv(cal_file)
        print(f"Loaded {len(df)} parameter sets")
        
        # Check for objective column
        obj_col = None
        if 'objective' in df.columns:
            obj_col = 'objective'
            print(f"Using 'objective' column for parameter selection")
        else:
            # Try to find alternative columns
            for possible_col in ['KGE', 'obj_function_value', 'KGE_NP', 'NSE']:
                if possible_col in df.columns:
                    obj_col = possible_col
                    break
        
        if obj_col:
            print(f"Using objective column: {obj_col}")
            # Sort by objective (higher is better) and get top N
            df = df.sort_values(obj_col, ascending=False).head(top_n)
            print(f"Selected top {len(df)} parameter sets")
            print(f"Objective range: {df[obj_col].min():.4f} to {df[obj_col].max():.4f}")
        else:
            print(f"Warning: No objective function column found. Using first {top_n} rows")
            df = df.head(top_n)
        
        # Extract parameter columns (starting with model type prefix)
        param_cols = [col for col in df.columns if col.startswith(f'{model_type}_')]
        
        if len(param_cols) == 0:
            print(f"Warning: No {model_type} parameter columns found")
            # Try without prefix
            param_cols = [col for col in df.columns if col not in ['objective', 'KGE', 'NSE', 'KGE_NP', 'obj_function_value']]
        
        if len(param_cols) == 0:
            print(f"Error: No parameter columns found")
            return None
        
        print(f"Found {len(param_cols)} parameter columns: {', '.join(param_cols[:5])}...")
        
        # Store parameter values for all selected top runs
        param_data = {}
        
        # For each parameter, store all values from the top runs
        for col in param_cols:
            param_data[col] = df[col].values.tolist()
        
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
        
        # Store everything
        result = {
            'parameters': param_data,
            'stats': param_stats,
            'objective_column': obj_col,
            'n_sets': len(df)
        }
        
        return result
        
    except Exception as e:
        print(f"Error reading file {cal_file}: {e}")
        return None

#--------------------------------------------------------------------------------

def plot_parameter_boxplots(config, plot_dirs, top_n=100):
    """
    Create boxplots for each parameter showing the distribution of the top parameter sets.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    top_n : int
        Number of top parameter sets to analyze
    """
    
    # Load parameter data
    param_data = load_parameter_values(config, top_n)
    if param_data is None:
        print("No parameter data available for plotting")
        return None
    
    parameters = param_data['parameters']
    stats = param_data['stats']
    
    # Get parameter names and clean them for display
    param_names = list(parameters.keys())
    n_params = len(param_names)
    
    if n_params == 0:
        print("No parameters to plot")
        return None
    
    # Calculate optimal subplot layout
    # Try to make it roughly square
    n_cols = int(np.ceil(np.sqrt(n_params)))
    n_rows = int(np.ceil(n_params / n_cols))
    
    print(f"Creating {n_rows}x{n_cols} subplot layout for {n_params} parameters")
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    
    # Handle case where there's only one subplot
    if n_params == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Plot each parameter
    for i, param_name in enumerate(param_names):
        ax = axes_flat[i]
        
        # Get parameter values
        values = parameters[param_name]
        
        # Create boxplot
        box_plot = ax.boxplot(values, patch_artist=True)
        
        # Customize boxplot appearance
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][0].set_alpha(0.7)
        
        # Clean parameter name for display (remove model prefix)
        display_name = param_name.replace(f"{config['model_type']}_", "")
        ax.set_title(f'{display_name}', fontsize=11, fontweight='bold')
        
        # Add statistics text
        param_stat = stats[param_name]
        stats_text = (f"Mean: {param_stat['mean']:.3f}\n"
                     f"Median: {param_stat['median']:.3f}\n"
                     f"Std: {param_stat['std']:.3f}")
        
        # Position text box
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=8,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Format y-axis
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylabel('Parameter Value', fontsize=10)
        
        # Remove x-axis labels (not meaningful for boxplots)
        ax.set_xticks([])
    
    # Hide empty subplots
    for i in range(n_params, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'Parameter Distribution - Catchment {config["gauge_id"]}\n'
                f'Top {param_data["n_sets"]} Parameter Sets (by {param_data["objective_column"]})', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save plot
    save_path = plot_dirs['parameters'] / f'parameter_boxplots_{config["gauge_id"]}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved parameter boxplots to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\nParameter Analysis Summary for Catchment {config['gauge_id']}:")
    print(f"  Number of parameters: {n_params}")
    print(f"  Number of parameter sets: {param_data['n_sets']}")
    print(f"  Objective function: {param_data['objective_column']}")
    
    # Print parameter ranges
    print(f"\nParameter Ranges:")
    for param_name in param_names:
        display_name = param_name.replace(f"{config['model_type']}_", "")
        stat = stats[param_name]
        print(f"  {display_name:15}: {stat['min']:.3f} - {stat['max']:.3f} (mean: {stat['mean']:.3f})")
    
    return fig


#--------------------------------------------------------------------------------
################################### storages ####################################
#--------------------------------------------------------------------------------

def load_storage_data(config):
    """
    Load watershed storage data for the configuration.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing storage data with datetime index
    """
    config_dir = Path(config['main_dir']) / config['config_dir']
    gauge_id = config['gauge_id']
    model_type = config['model_type']
    
    # Construct path to storage file
    storage_file = config_dir / f"catchment_{gauge_id}" / model_type / "output" / f"{gauge_id}_{model_type}_WatershedStorage.csv"
    
    print(f"Loading storage data:")
    print(f"  - File: {storage_file}")
    
    if not storage_file.exists():
        print(f"ERROR: Storage file not found: {storage_file}")
        return None
    
    try:
        # Read the CSV file with the second row skipped (units row)
        df = pd.read_csv(storage_file, skiprows=[1])

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
        storage_cols = [col for col in df.columns if col not in ['date', 'month', 'year']]
        print(f"  - Storage columns: {storage_cols}")
        
        return df
        
    except Exception as e:
        print(f"  - Error loading storage data: {e}")
        return None

#--------------------------------------------------------------------------------

def plot_storage_timeseries(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot time series of watershed storage components for a single catchment configuration.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    """
    # Load storage data
    storage_df = load_storage_data(config)
    
    if storage_df is None:
        print(f"No storage data available for catchment {config['gauge_id']}")
        return None
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    # Filter by validation period
    validation_start = pd.to_datetime(validation_start)
    validation_end = pd.to_datetime(validation_end)
    
    val_mask = (storage_df['date'] >= validation_start) & (storage_df['date'] <= validation_end)
    storage_df = storage_df[val_mask].copy()
    
    if len(storage_df) == 0:
        print(f"No storage data found for validation period: {validation_start} to {validation_end}")
        return None
    
    # Get storage columns (exclude date, month, year)
    storage_cols = [col for col in storage_df.columns if col not in ['date', 'month', 'year']]
    
    if len(storage_cols) == 0:
        print("No storage columns found in data")
        return None
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(storage_cols), 1, figsize=(14, 3.5*len(storage_cols)), sharex=True)
    
    if len(storage_cols) == 1:
        axes = [axes]  # Make it iterable
    
    # Define colors for different storage types
    storage_colors = {
        'snowfall': 'skyblue',
        'rainfall': 'navy',
        'snow storage': 'white',
        'soil': 'brown',
        'groundwater': 'blue',
        'depression': 'lightblue',
        'ponded': 'cyan',
        'fast': 'orange',
        'slow': 'darkblue'
    }
    
    # Plot each storage component
    for i, col in enumerate(storage_cols):
        ax = axes[i]
        
        # Determine color based on column name
        color = '#2a5674'  # default color
        for key, storage_color in storage_colors.items():
            if key in col.lower():
                color = storage_color
                break
        
        # Handle different types of data
        if 'snowfall' in col.lower() and '[mm/d]' in col:
            # Snowfall - use fill_between with sky blue
            data = storage_df[col]
            y_max = np.percentile(data[data > 0], 95) if len(data[data > 0]) > 0 else data.max()
            ax.set_ylim(0, y_max * 1.1)
            ax.fill_between(storage_df['date'], 0, data, color='skyblue', alpha=0.7, edgecolor='lightblue', linewidth=0.5)
            
        elif 'rainfall' in col.lower() and '[mm/d]' in col:
            # Rainfall - use fill_between with navy
            data = storage_df[col]
            y_max = np.percentile(data[data > 0], 95) if len(data[data > 0]) > 0 else data.max()
            ax.set_ylim(0, y_max * 1.1)
            ax.fill_between(storage_df['date'], 0, data, color='navy', alpha=0.7, edgecolor='darkblue', linewidth=0.5)
            
        elif 'snow storage' in col.lower():
            # Snow storage - use fill_between with white/light gray
            ax.fill_between(storage_df['date'], 0, storage_df[col], 
                           color='white', alpha=0.9, edgecolor='lightgray', linewidth=1)
            
        else:
            # Other storage components - use line plots
            ax.plot(storage_df['date'], storage_df[col], color=color, linewidth=1.5)
        
        # Clean up column name for title
        clean_title = col.replace('[mm]', '(mm)').replace('[mm/d]', '(mm/d)')
        ax.set_title(f'{clean_title}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Storage (mm)', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add some basic statistics as text
        mean_val = storage_df[col].mean()
        max_val = storage_df[col].max()
        stats_text = f"Mean: {mean_val:.1f} mm\nMax: {max_val:.1f} mm"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    
    # Set x-label only on bottom subplot
    axes[-1].set_xlabel('Date', fontsize=12)
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add overall title
    fig.suptitle(f'Watershed Storage Components - Catchment {config["gauge_id"]}\n'
                f'Validation Period: {validation_start.date()} to {validation_end.date()}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.3)
    
    # Save plot
    save_path = plot_dirs['storage'] / f'storage_timeseries_{config["gauge_id"]}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved storage plot to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\nStorage Analysis Summary for Catchment {config['gauge_id']}:")
    print(f"  Period: {validation_start.date()} to {validation_end.date()}")
    print(f"  Number of storage components: {len(storage_cols)}")
    print(f"  Storage components:")
    for col in storage_cols:
        mean_val = storage_df[col].mean()
        max_val = storage_df[col].max()
        print(f"    {col}: Mean={mean_val:.1f} mm, Max={max_val:.1f} mm")
    
    return fig

#--------------------------------------------------------------------------------
###################################### SWE ######################################
#--------------------------------------------------------------------------------


def load_swe_data(config):
    """
    Load SWE data for ALL HRU groups from model output.
    Returns both the full dataframe and the area data.
    
    Returns:
    --------
    tuple: (sim_data, area_data)
        sim_data: DataFrame with date and all HRU group SWE data
        area_data: DataFrame with HRU areas (or None if not available)
    """
    config_dir = Path(config['main_dir']) / config['config_dir']
    gauge_id = config['gauge_id']
    model_type = config['model_type']
    
    # Construct path to SWE file
    swe_file = config_dir / f"catchment_{gauge_id}" / model_type / "output" / f"{gauge_id}_{model_type}_SNOW_Daily_Average_ByHRUGroup.csv"
    
    print(f"Loading SWE data for all HRU groups:")
    print(f"  - File: {swe_file}")
    
    if not swe_file.exists():
        print(f"ERROR: SWE file not found: {swe_file}")
        return None, None
    
    try:
        # Read the CSV file, skipping the units row (row index 1)
        df = pd.read_csv(swe_file, skiprows=[1])
        
        print(f"  - Loaded data shape: {df.shape}")
        print(f"  - Columns: {df.columns.tolist()}")
        
        # Get dates from the 'HRUGroup:' column (same as other ByHRUGroup files)
        if 'HRUGroup:' in df.columns:
            df['date'] = pd.to_datetime(df['HRUGroup:'])
            print(f"  - Using 'HRUGroup:' column from file for dates")
        elif 'day' in df.columns:
            df['date'] = pd.to_datetime(df['day'])
            print(f"  - Using 'day' column from file for dates")
        else:
            print(f"ERROR: No date column found in file")
            return None, None
        
        print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Identify HRU group columns (exclude 'date', 'day', etc.)
        exclude_cols = ['date', 'day', 'time', 'Unnamed: 0', 'HRUGroup:']
        hru_groups = [col for col in df.columns if col not in exclude_cols]
        
        print(f"  - Found {len(hru_groups)} HRU groups: {hru_groups}")
        
        # Convert all HRU groups from m to mm if needed
        for col in hru_groups:
            if df[col].mean() < 10 and df[col].max() < 20:
                df[col] = df[col] * 1000
        
        print(f"  ✓ Successfully loaded SWE data for all HRU groups")
        
        # Try to load area data from HRU shapefile
        area_data = None
        try:
            topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
            hru_shapefile = topo_dir / "HRU.shp"
            if hru_shapefile.exists():
                import geopandas as gpd
                hru_gdf = gpd.read_file(hru_shapefile)
                area_data = hru_gdf[['Area_km2']].copy()
                print(f"  ✓ Loaded HRU area data: {len(area_data)} HRUs")
        except Exception as e:
            print(f"  WARNING: Could not load HRU area data: {e}")
        
        return df, area_data
        
    except Exception as e:
        print(f"ERROR: Failed to load SWE data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

#--------------------------------------------------------------------------------

def plot_area_weighted_swe_timeseries(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot catchment-average SWE time series using the NO_GLACIER HRU group.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with date and catchment-average SWE
    """
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_start_date', config.get('start_date', '2000-01-01'))
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    validation_start = pd.to_datetime(validation_start)
    validation_end = pd.to_datetime(validation_end)
    
    gauge_id = config['gauge_id']
    
    print(f"Plotting catchment-average SWE for catchment {gauge_id}:")
    print(f"  - Period: {validation_start.date()} to {validation_end.date()}")
    
    # Load SWE data using the comprehensive loader
    df, area_data = load_swe_data(config)
    
    if df is None:
        print("ERROR: Failed to load SWE data")
        return None
    
    if 'NO_GLACIER' not in df.columns:
        print("ERROR: NO_GLACIER column not available for catchment average")
        return None
    
    # Filter for validation period
    mask = (df['date'] >= validation_start) & (df['date'] <= validation_end)
    df_filtered = df[mask].copy()
    
    if len(df_filtered) == 0:
        print(f"ERROR: No data found for period {validation_start.date()} to {validation_end.date()}")
        return None
    
    print(f"  - Filtered to {len(df_filtered)} records")
    
    # Get catchment average SWE
    catchment_avg_swe = df_filtered[['date', 'NO_GLACIER']].copy()
    catchment_avg_swe.columns = ['date', 'catchment_avg_swe']
    
    print(f"  - Mean SWE: {catchment_avg_swe['catchment_avg_swe'].mean():.1f} mm")
    print(f"  - Max SWE: {catchment_avg_swe['catchment_avg_swe'].max():.1f} mm")
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot catchment-average SWE
    plt.plot(catchment_avg_swe['date'], catchment_avg_swe['catchment_avg_swe'], 
             'C0', linewidth=2, label='Catchment-Average SWE (NO_GLACIER HRU Group)')
    
    # Format plot
    plt.title(f'Catchment-Average SWE Time Series - Catchment {gauge_id}\n'
              f'Period: {validation_start.date()} to {validation_end.date()}', 
              fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Snow Water Equivalent (mm)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gcf().autofmt_xdate()
    
    # Add summary statistics text box
    mean_swe = catchment_avg_swe['catchment_avg_swe'].mean()
    max_swe = catchment_avg_swe['catchment_avg_swe'].max()
    max_date = catchment_avg_swe.loc[catchment_avg_swe['catchment_avg_swe'].idxmax(), 'date']
    
    stats_text = (f"Statistics:\n"
                 f"Mean SWE: {mean_swe:.1f} mm\n"
                 f"Max SWE: {max_swe:.1f} mm\n"
                 f"Max date: {max_date.date()}")
    
    # Place text box in top right corner
    plt.figtext(0.98, 0.98, stats_text, fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
               verticalalignment='top', horizontalalignment='right')
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['swe'] / f'catchment_average_swe_timeseries_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved catchment-average SWE plot to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\nSWE Analysis Results for Catchment {gauge_id}:")
    print(f"  Mean SWE: {mean_swe:.2f} mm")
    print(f"  Max SWE: {max_swe:.2f} mm")
    print(f"  Max SWE date: {max_date.date()}")
    
    # Return filtered data
    return catchment_avg_swe

#--------------------------------------------------------------------------------

def plot_swe_time_series_by_elevation(config, plot_dirs, water_year=None, validation_start=None, validation_end=None):
    """
    Plot time series of simulated SWE for each elevation band.
    Simplified version that just plots the data without area weighting.
    """
    
    config_dir = Path(config['main_dir']) / config['config_dir']
    gauge_id = config['gauge_id']
    model_type = config['model_type']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', config.get('start_date', '2000-01-01'))
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"Loading SWE data by elevation bands:")
    print(f"  - Period: {validation_start} to {validation_end}")
    
    # Load SWE file
    sim_file = config_dir / f"catchment_{gauge_id}" / model_type / "output" / f"{gauge_id}_{model_type}_SNOW_Daily_Average_ByHRUGroup.csv"
    
    if not sim_file.exists():
        print(f"ERROR: SWE file not found: {sim_file}")
        return None
    
    try:
        # Read the file, skipping the units row (row index 1)
        sim_data = pd.read_csv(sim_file, skiprows=[1])
        
        print(f"  - Loaded data shape: {sim_data.shape}")
        print(f"  - Columns: {sim_data.columns.tolist()[:10]}...")
        
        # Find the time/date column (first column, which may be unnamed)
        date_col = sim_data.columns[0]
        print(f"  - Date column: '{date_col}'")
        
        # Parse dates - THIS WAS THE BUG!
        # The column might contain actual date strings or row numbers
        try:
            # Try to parse as dates
            sim_data['date'] = pd.to_datetime(sim_data[date_col])
            
            # Check if dates make sense (not all 1970)
            if sim_data['date'].min().year == 1970:
                print(f"  - Warning: Dates parsed as 1970, trying alternative method...")
                # Try reading the raw file to get actual dates
                with open(sim_file, 'r') as f:
                    first_line = f.readline()  # Header with band names
                    second_line = f.readline()  # Units line (skip)
                    
                # Re-read without skipping unit line to see actual dates
                temp_df = pd.read_csv(sim_file)
                if temp_df.columns[0].lower() in ['time', 'date']:
                    sim_data['date'] = pd.to_datetime(temp_df.iloc[:, 0])
                else:
                    # If still doesn't work, create date range from config
                    print(f"  - Creating date range from config dates...")
                    start = pd.to_datetime(config.get('start_date', '2000-01-01'))
                    sim_data['date'] = pd.date_range(start=start, periods=len(sim_data), freq='D')
            
            print(f"  - Date range: {sim_data['date'].min()} to {sim_data['date'].max()}")
            
        except Exception as e:
            print(f"  - Error parsing dates: {e}")
            print(f"  - Creating date range from config...")
            start = pd.to_datetime(config.get('start_date', '2000-01-01'))
            sim_data['date'] = pd.date_range(start=start, periods=len(sim_data), freq='D')
        
        # Get elevation band columns (exclude 'AllHRUs', 'date', 'day', etc.)
        exclude_cols = ['date', 'day', 'time', 'AllHRUs', 'HRUGroup:', date_col]
        if 'Unnamed' in str(date_col):
            exclude_cols.append(date_col)
        
        elev_bands = [col for col in sim_data.columns if col not in exclude_cols and '-' in col and 'm' in col]
        
        if len(elev_bands) == 0:
            print(f"ERROR: No elevation band columns found")
            print(f"  Available columns: {sim_data.columns.tolist()}")
            return None
        
        print(f"  - Found {len(elev_bands)} elevation bands: {elev_bands[:5]}...")
        
        # Filter for time period
        start_date = pd.to_datetime(validation_start)
        end_date = pd.to_datetime(validation_end)
        
        sim_mask = (sim_data['date'] >= start_date) & (sim_data['date'] <= end_date)
        sim_filtered = sim_data[sim_mask].copy()
        
        if len(sim_filtered) == 0:
            print(f"ERROR: No data found for period {start_date} to {end_date}")
            print(f"  Data date range: {sim_data['date'].min()} to {sim_data['date'].max()}")
            return None
        
        print(f"  - Filtered to {len(sim_filtered)} records")
        
        # Sort elevation bands by altitude
        elev_bands_sorted = sorted(elev_bands, key=lambda x: int(x.split('-')[0]))
        
        # Create subplots - one row per elevation band
        n_bands = len(elev_bands_sorted)
        fig, axes = plt.subplots(n_bands, 1, figsize=(14, 3*n_bands), sharex=True)
        
        # Make axes iterable if there's only one
        if n_bands == 1:
            axes = [axes]
        
        # Plot each elevation band
        for i, band in enumerate(elev_bands_sorted):
            ax = axes[i]
            
            # Plot SWE data
            swe_values = sim_filtered[band].values
            
            # Convert to mm if needed (if values are in meters)
            if swe_values.mean() < 10 and swe_values.max() < 20:
                swe_values = swe_values * 1000
            
            ax.plot(sim_filtered['date'], swe_values, 'C0', linewidth=1.5)
            
            # Calculate statistics
            mean_val = swe_values.mean()
            max_val = swe_values.max()
            
            # Add statistics text box
            stats_text = f"Mean: {mean_val:.1f} mm\nMax: {max_val:.1f} mm"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Formatting
            ax.set_title(f'Elevation Band: {band}', fontsize=12, fontweight='bold')
            ax.set_ylabel('SWE (mm)', fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        # Add overall title
        title = f'SWE by Elevation Band - Catchment {gauge_id}\n'
        title += f'Period: {start_date.date()} to {end_date.date()}'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Format x-axis for bottom plot
        axes[-1].set_xlabel('Date', fontsize=12)
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.subplots_adjust(hspace=0.3)
        
        # Save figure
        filename = f'swe_time_series_by_elevation_{gauge_id}.png'
        save_path = plot_dirs['swe'] / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved SWE elevation band plot to: {save_path}")
        plt.show()
        
        # Print summary
        print(f"\nSWE Elevation Band Analysis:")
        print(f"  - Period: {start_date.date()} to {end_date.date()}")
        print(f"  - Number of elevation bands: {len(elev_bands_sorted)}")
        print(f"  - Elevation range: {elev_bands_sorted[0]} to {elev_bands_sorted[-1]}")
        
        return fig
        
    except Exception as e:
        print(f"ERROR: Failed to process SWE data: {e}")
        import traceback
        traceback.print_exc()
        return None

#--------------------------------------------------------------------------------

def analyze_peak_swe(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Analyze peak SWE timing and magnitude for the catchment using catchment-average SWE.
    Uses the NO_GLACIER HRU group which contains catchment-average SWE.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    dict
        Dictionary containing peak SWE analysis results
    """
    
    # Load SWE data using new tuple format
    full_data, area_data = load_swe_data(config)
    if full_data is None:
        print("Failed to load SWE data for peak analysis")
        return None
    
    # Check if NO_GLACIER column exists (catchment-average SWE)
    if 'NO_GLACIER' not in full_data.columns:
        print("NO_GLACIER column not available for catchment-average SWE")
        return None
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', config.get('start_date', '2000-01-01'))
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    validation_start = pd.to_datetime(validation_start)
    validation_end = pd.to_datetime(validation_end)
    
    gauge_id = config['gauge_id']
    
    print(f"Analyzing peak SWE for catchment {gauge_id}:")
    print(f"  - Period: {validation_start.date()} to {validation_end.date()}")
    print(f"  - Using NO_GLACIER HRU group (catchment-average SWE)")
    
    # Filter for validation period
    val_mask = (full_data['date'] >= validation_start) & (full_data['date'] <= validation_end)
    val_data = full_data[val_mask].copy()
    
    if len(val_data) == 0:
        print("No data found for validation period")
        return None
    
    # Get catchment-average SWE column
    val_data['catchment_avg_swe'] = val_data['NO_GLACIER']
    
    # Analyze peak SWE by water year (October 1 - September 30)
    def get_water_year(date):
        if date.month >= 10:
            return date.year + 1
        else:
            return date.year
    
    val_data['water_year'] = val_data['date'].apply(get_water_year)
    
    # Find peak SWE for each water year
    peak_swe_results = {
        'simulated': {},
        'summary': {}
    }
    
    # Analyze peak SWE
    water_years = sorted(val_data['water_year'].unique())
    
    for wy in water_years:
        wy_data = val_data[val_data['water_year'] == wy].copy()
        
        if len(wy_data) > 0:
            # Find peak SWE
            peak_idx = wy_data['catchment_avg_swe'].idxmax()
            peak_swe = wy_data.loc[peak_idx]
            
            # Calculate statistics for this water year
            mean_swe = wy_data['catchment_avg_swe'].mean()
            max_swe = wy_data['catchment_avg_swe'].max()
            
            # Find snow season length (days with SWE > 10 mm)
            snow_days = len(wy_data[wy_data['catchment_avg_swe'] > 10])
            
            peak_swe_results['simulated'][wy] = {
                'peak_date': peak_swe['date'],
                'peak_swe': peak_swe['catchment_avg_swe'],
                'peak_doy': peak_swe['date'].dayofyear,
                'mean_swe': mean_swe,
                'max_swe': max_swe,
                'snow_days': snow_days,
                'data_points': len(wy_data)
            }
    
    # Calculate summary statistics
    if peak_swe_results['simulated']:
        sim_peak_dates = [result['peak_doy'] for result in peak_swe_results['simulated'].values()]
        sim_peak_swe = [result['peak_swe'] for result in peak_swe_results['simulated'].values()]
        
        peak_swe_results['summary']['simulated'] = {
            'mean_peak_doy': np.mean(sim_peak_dates),
            'std_peak_doy': np.std(sim_peak_dates),
            'median_peak_doy': np.median(sim_peak_dates),
            'mean_peak_swe': np.mean(sim_peak_swe),
            'std_peak_swe': np.std(sim_peak_swe),
            'median_peak_swe': np.median(sim_peak_swe),
            'n_years': len(sim_peak_dates)
        }
    
    # Create visualization
    plt.figure(figsize=(16, 10))
    
    # Plot 1: Peak SWE timing by water year
    plt.subplot(2, 2, 1)
    
    if peak_swe_results['simulated']:
        sim_years = list(peak_swe_results['simulated'].keys())
        sim_doys = [peak_swe_results['simulated'][wy]['peak_doy'] for wy in sim_years]
        plt.scatter(sim_years, sim_doys, color='C0', s=80, alpha=0.7, label='Simulated', edgecolors='black', linewidth=1)
    
    plt.xlabel('Water Year', fontsize=12)
    plt.ylabel('Peak SWE Day of Year', fontsize=12)
    plt.title('Peak SWE Timing by Water Year', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add month labels on y-axis
    month_doys = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.yticks(month_doys, month_labels)
    
    # Plot 2: Peak SWE magnitude by water year
    plt.subplot(2, 2, 2)
    
    if peak_swe_results['simulated']:
        sim_magnitudes = [peak_swe_results['simulated'][wy]['peak_swe'] for wy in sim_years]
        plt.scatter(sim_years, sim_magnitudes, color='C0', s=80, alpha=0.7, label='Simulated', edgecolors='black', linewidth=1)
    
    plt.xlabel('Water Year', fontsize=12)
    plt.ylabel('Peak SWE (mm)', fontsize=12)
    plt.title('Peak SWE Magnitude by Water Year', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Peak timing distribution
    plt.subplot(2, 2, 3)
    
    if peak_swe_results['simulated']:
        plt.hist(sim_doys, bins=15, alpha=0.7, color='C0', edgecolor='black', label='Simulated')
    
    plt.xlabel('Peak SWE Day of Year', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Peak SWE Timing Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Peak magnitude distribution
    plt.subplot(2, 2, 4)
    
    if peak_swe_results['simulated']:
        plt.hist(sim_magnitudes, bins=15, alpha=0.7, color='C0', edgecolor='black', label='Simulated')
    
    plt.xlabel('Peak SWE (mm)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Peak SWE Magnitude Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Peak SWE Analysis - Catchment {gauge_id}\n'
                f'Period: {validation_start.date()} to {validation_end.date()} (Catchment Average)', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['swe'] / f'peak_swe_analysis_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved peak SWE analysis plot to: {save_path}")
    plt.show()
    
    # Print summary statistics
    print(f"\nPeak SWE Analysis Summary for Catchment {gauge_id}:")
    print(f"  Period: {validation_start.date()} to {validation_end.date()}")
    
    if 'simulated' in peak_swe_results['summary']:
        sim_summary = peak_swe_results['summary']['simulated']
        print(f"\n  Simulated Peak SWE:")
        print(f"    Number of water years: {sim_summary['n_years']}")
        print(f"    Mean peak timing: DOY {sim_summary['mean_peak_doy']:.1f} ± {sim_summary['std_peak_doy']:.1f}")
        print(f"    Median peak timing: DOY {sim_summary['median_peak_doy']:.1f}")
        print(f"    Mean peak SWE: {sim_summary['mean_peak_swe']:.1f} ± {sim_summary['std_peak_swe']:.1f} mm")
        print(f"    Median peak SWE: {sim_summary['median_peak_swe']:.1f} mm")
        
        # Get actual month from the peak_swe data
        earliest_peak = min(sim_doys)
        latest_peak = max(sim_doys)
        
        earliest_peak_row = [result for result in peak_swe_results['simulated'].values() 
                            if result['peak_doy'] == earliest_peak][0]
        latest_peak_row = [result for result in peak_swe_results['simulated'].values() 
                          if result['peak_doy'] == latest_peak][0]
        
        earliest_month = month_labels[earliest_peak_row['peak_date'].month - 1]
        latest_month = month_labels[latest_peak_row['peak_date'].month - 1]
        
        print(f"    Earliest peak: DOY {earliest_peak} ({earliest_month} {earliest_peak_row['peak_date'].day})")
        print(f"    Latest peak: DOY {latest_peak} ({latest_month} {latest_peak_row['peak_date'].day})")
    
    return peak_swe_results

#--------------------------------------------------------------------------------

def plot_spatial_swe_distribution(config, plot_dirs, validation_start=None, validation_end=None, 
                                  example_years=None, figsize=(20, 16)):
    """
    Plot spatial SWE distribution for SIMULATED data only (no observed data).
    Creates plots on the date of simulated peak SWE for each year.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    example_years : list, optional
        List of years to plot (default: first and last year of validation period)
    figsize : tuple, optional
        Figure size for the plots
        
    Returns:
    --------
    dict
        Dictionary containing plot information and peak SWE data
    """
    
    config_dir = Path(config['main_dir']) / config['config_dir']
    gauge_id = config['gauge_id']
    model_type = config['model_type']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    validation_start_dt = pd.to_datetime(validation_start)
    validation_end_dt = pd.to_datetime(validation_end)
    
    print(f"Creating spatial SWE peak plots for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    print(f"  - Simulated data only (no observations)")
    print(f"  - Excluding glacier areas (Landuse_Cl 7 and 8)")
    print(f"  - Excluding HRUs with SWE > 1000mm")
    
    # 1. Load HRU shapefile for spatial information
    topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
    hru_shapefile = topo_dir / "HRU.shp"
    
    if not hru_shapefile.exists():
        print(f"ERROR: HRU shapefile not found: {hru_shapefile}")
        return None
    
    try:
        hru_gdf = gpd.read_file(hru_shapefile)
        original_count = len(hru_gdf)
        print(f"  - Loaded {original_count} HRUs from shapefile")
        
        # Filter out glacier HRUs (landuse classes 7 and 8)
        if 'Landuse_Cl' in hru_gdf.columns:
            hru_gdf = hru_gdf[~hru_gdf['Landuse_Cl'].isin([7, 8])].copy()
            print(f"  - Filtered to {len(hru_gdf)} non-glacier HRUs")
        else:
            print(f"  - Warning: 'Landuse_Cl' column not found, cannot filter glacier areas")
            
    except Exception as e:
        print(f"ERROR: Failed to load HRU shapefile: {e}")
        return None
    
    # 2. Load simulated SWE data
    model_dir = config_dir / f"catchment_{gauge_id}" / model_type
    sim_swe_file = model_dir / "output" / f"{gauge_id}_{model_type}_SNOW_Daily_Average_ByHRU.csv"
    
    if not sim_swe_file.exists():
        print(f"ERROR: Simulated SWE file not found: {sim_swe_file}")
        return None
    
    try:
        # Handle header properly
        with open(sim_swe_file, 'r') as f:
            header_line = f.readline().strip()
            units_line = f.readline().strip()
        
        sim_swe_df = pd.read_csv(sim_swe_file, skiprows=[1], header=0)
        
        # Create date from start_date in config
        start_date = pd.to_datetime(config.get('start_date', '2000-01-01'))
        sim_swe_df['date'] = pd.date_range(start=start_date, periods=len(sim_swe_df), freq='D')
        
        # Filter for validation period
        sim_mask = (sim_swe_df['date'] >= validation_start_dt) & (sim_swe_df['date'] <= validation_end_dt)
        sim_swe_df = sim_swe_df[sim_mask].copy()
        
        if len(sim_swe_df) == 0:
            print("ERROR: No simulated SWE data found for validation period")
            return None
        
        print(f"  - Loaded simulated SWE data: {len(sim_swe_df)} records")
        
    except Exception as e:
        print(f"ERROR: Failed to load simulated SWE data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 3. Get HRU columns and create mapping
    sim_hru_cols = [col for col in sim_swe_df.columns if col not in ['date', 'day', 'time', 'row_id', 'HRU:'] and col != '']
    
    # Create mapping between HRU columns and HRU IDs (only non-glacier HRUs)
    hru_mapping = {}
    non_glacier_hru_ids = set(hru_gdf['HRU_ID'].astype(str))
    
    for sim_col in sim_hru_cols:
        try:
            if sim_col.isdigit():
                sim_hru_id = int(sim_col)
                if str(sim_hru_id) in non_glacier_hru_ids:
                    hru_mapping[sim_col] = sim_hru_id
        except:
            continue
    
    if len(hru_mapping) == 0:
        print("ERROR: No matching non-glacier HRU columns found")
        return None
    
    print(f"  - Found {len(hru_mapping)} matching non-glacier HRU columns")
    
    # 4. Convert SWE units if needed (from m to mm)
    for sim_col in hru_mapping.keys():
        if sim_col in sim_swe_df.columns:
            sim_vals = sim_swe_df[sim_col]
            if sim_vals.mean() < 10 and sim_vals.max() < 20:
                sim_swe_df[sim_col] = sim_vals * 1000
    
    # 5. Determine example years
    available_years = sorted(sim_swe_df['date'].dt.year.unique())
    
    if example_years is None:
        if len(available_years) >= 2:
            example_years = [available_years[0], available_years[-1]]
        else:
            example_years = available_years[:1]
    
    print(f"  - Example years: {example_years}")
    
    # 6. Find peak SWE dates for each year and create plots
    plot_info = {}
    
    for year in example_years:
        print(f"\n  Processing year {year}...")
        
        # Define water year period (Oct 1 to Sep 30)
        wy_start = pd.to_datetime(f"{year-1}-10-01")
        wy_end = pd.to_datetime(f"{year}-09-30")
        
        # Filter data for this water year
        sim_wy_mask = (sim_swe_df['date'] >= wy_start) & (sim_swe_df['date'] <= wy_end)
        sim_wy = sim_swe_df[sim_wy_mask].copy()
        
        if len(sim_wy) == 0:
            print(f"    Warning: No data for water year {year}")
            continue
        
        # Calculate area-weighted mean SWE for peak detection
        hru_weights = {}
        for sim_col in hru_mapping.keys():
            try:
                hru_id = hru_mapping[sim_col]
                hru_match = hru_gdf[hru_gdf['HRU_ID'] == hru_id]
                if len(hru_match) > 0:
                    hru_weights[sim_col] = hru_match['Area_km2'].iloc[0]
                else:
                    hru_weights[sim_col] = 1.0
            except:
                hru_weights[sim_col] = 1.0
        
        # Calculate weighted mean SWE for simulated data
        sim_wy['weighted_swe'] = 0
        total_weight = sum(hru_weights.values())
        for sim_col in hru_mapping.keys():
            if sim_col in sim_wy.columns:
                weight = hru_weights[sim_col] / total_weight
                sim_wy['weighted_swe'] += sim_wy[sim_col].fillna(0) * weight
        
        # Find peak SWE date
        sim_peak_idx = sim_wy['weighted_swe'].idxmax()
        sim_peak_date = sim_wy.loc[sim_peak_idx, 'date']
        sim_peak_swe = sim_wy.loc[sim_peak_idx, 'weighted_swe']
        
        print(f"    Simulated peak: {sim_peak_date.date()} ({sim_peak_swe:.1f} mm)")
        
        # Create plot for simulated peak date
        print(f"    Creating plot for simulated peak date...")
        
        # Get SWE data for simulated peak date
        sim_sim_peak_data = sim_swe_df.loc[sim_swe_df['date'] == sim_peak_date]
        
        if len(sim_sim_peak_data) > 0:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            
            # Prepare HRU data for simulated peak date
            hru_gdf_sim = hru_gdf.copy()
            hru_gdf_sim['sim_swe'] = np.nan
            
            # Add SWE data to geodataframe
            for sim_col in hru_mapping.keys():
                try:
                    hru_id = hru_mapping[sim_col]
                    hru_mask = hru_gdf_sim['HRU_ID'] == hru_id
                    
                    if hru_mask.any() and len(sim_sim_peak_data) > 0:
                        hru_gdf_sim.loc[hru_mask, 'sim_swe'] = sim_sim_peak_data[sim_col].iloc[0]
                except:
                    continue
            
            # Filter out HRUs with extremely high SWE (>1000mm) or no data
            plot_mask = ~hru_gdf_sim['sim_swe'].isna()
            hru_gdf_sim = hru_gdf_sim[plot_mask].copy()
            
            high_swe_mask = hru_gdf_sim['sim_swe'] <= 1000
            hru_gdf_sim = hru_gdf_sim[high_swe_mask].copy()
            
            if len(hru_gdf_sim) > 0:
                # Determine color scale
                valid_sim = hru_gdf_sim['sim_swe'].dropna()
                
                if len(valid_sim) > 0:
                    vmin = max(0, valid_sim.min())
                    vmax = min(1000, valid_sim.max())
                else:
                    vmin, vmax = 0, 100
                
                # Plot simulated SWE
                hru_gdf_sim.plot(column='sim_swe', ax=ax, cmap='Blues', 
                               vmin=vmin, vmax=vmax, legend=False, 
                               edgecolor='black', linewidth=0.5)
                
                ax.set_title(f'Simulated SWE on Peak Date\n'
                             f'{sim_peak_date.strftime("%Y-%m-%d")} (Water Year {year})\n'
                             f'Peak SWE: {sim_peak_swe:.1f} mm', 
                             fontsize=14, fontweight='bold')
                ax.set_xlabel('Longitude', fontsize=12)
                ax.set_ylabel('Latitude', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Add colorbar
                from matplotlib import cm
                from matplotlib.colors import Normalize
                sm = cm.ScalarMappable(cmap='Blues', norm=Normalize(vmin=vmin, vmax=vmax))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Snow Water Equivalent (mm)', fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                
                # Save plot
                save_path = plot_dirs['swe'] / f'spatial_swe_sim_peak_WY{year}_{gauge_id}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"    Saved simulated peak plot to: {save_path}")
                plt.show()
                
                # Store plot information
                plot_info[year] = {
                    'sim_peak_date': sim_peak_date,
                    'sim_peak_swe': sim_peak_swe,
                    'sim_peak_plot': save_path
                }
    
    # Print summary
    print(f"\nSpatial SWE Peak Analysis Summary:")
    print(f"  Catchment: {gauge_id}")
    print(f"  Validation period: {validation_start} to {validation_end}")
    print(f"  Example years processed: {list(plot_info.keys())}")
    print(f"  Original HRUs in shapefile: {original_count}")
    print(f"  Non-glacier HRUs: {len(hru_gdf)}")
    print(f"  HRUs with SWE data: {len(hru_mapping)}")
    
    for year, info in plot_info.items():
        print(f"\n  Water Year {year}:")
        print(f"    Simulated peak: {info['sim_peak_date'].strftime('%Y-%m-%d')} ({info['sim_peak_swe']:.1f} mm)")
    
    return {
        'plot_info': plot_info,
        'hru_mapping': hru_mapping,
        'n_hrus_total': original_count,
        'n_hrus_non_glacier': len(hru_gdf),
        'n_hrus_with_data': len(hru_mapping),
        'example_years': example_years,
        'filtering_applied': {
            'glacier_areas_excluded': True,
            'high_swe_threshold': 1000,
            'landuse_classes_excluded': [7, 8]
        }
    }


#--------------------------------------------------------------------------------
#################################### run all ####################################
#--------------------------------------------------------------------------------

def run_complete_postprocessing(config, validation_start=None, validation_end=None, 
                                skip_errors=True, verbose=True):
    """
    Run complete postprocessing analysis - all plotting and analysis functions.
    
    This function is designed to be run after calibration and model execution 
    to automatically generate all output plots and analyses.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    validation_start : str, optional
        Start date for validation period (defaults to cali_end_date or start_date from config)
    validation_end : str, optional
        End date for validation period (defaults to end_date from config)
    skip_errors : bool, optional
        If True, continue processing even if individual functions fail (default: True)
    verbose : bool, optional
        If True, print detailed progress information (default: True)
        
    Returns:
    --------
    dict
        Dictionary containing results from all analyses and list of any errors
    """
    
    import time
    start_time = time.time()
    
    # Initialize results and error tracking
    results = {}
    errors = []
    
    # Set up plot directories
    if verbose:
        print("="*80)
        print("RUNNING COMPLETE POSTPROCESSING ANALYSIS")
        print("="*80)
        print(f"\nCatchment: {config.get('gauge_id', 'Unknown')}")
        print(f"Model: {config.get('model_type', 'Unknown')}")
        if validation_start:
            print(f"Validation period: {validation_start} to {validation_end}")
        print("\n" + "="*80 + "\n")
    
    plot_dirs = setup_plot_directories(config)
    
    # Helper function to run a function with error handling
    def run_function(func_name, func, *args, **kwargs):
        if verbose:
            print(f"\n{'='*80}")
            print(f"Running: {func_name}")
            print(f"{'='*80}")
        try:
            result = func(*args, **kwargs)
            if verbose:
                print(f"✓ {func_name} completed successfully")
            return result
        except Exception as e:
            error_msg = f"✗ ERROR in {func_name}: {str(e)}"
            print(error_msg)
            if not skip_errors:
                raise
            errors.append({'function': func_name, 'error': str(e)})
            import traceback
            if verbose:
                traceback.print_exc()
            return None
    
    # ========================================================================
    # 1. STREAMFLOW ANALYSIS
    # ========================================================================
    if verbose:
        print("\n" + "#"*80)
        print("# 1. STREAMFLOW ANALYSIS")
        print("#"*80)
    
    results['hydrological_regime'] = run_function(
        'plot_hydrological_regime',
        plot_hydrological_regime,
        config, plot_dirs, validation_start, validation_end
    )
    
    results['hydrograph_timeseries'] = run_function(
        'plot_hydrograph_timeseries',
        plot_hydrograph_timeseries,
        config, plot_dirs, validation_start, validation_end
    )
    
    results['streamflow_scatter'] = run_function(
        'plot_streamflow_scatter',
        plot_streamflow_scatter,
        config, plot_dirs, validation_start, validation_end
    )
    
    results['streamflow_residuals'] = run_function(
        'plot_streamflow_residuals',
        plot_streamflow_residuals,
        config, plot_dirs, validation_start, validation_end
    )
    
    results['performance_metrics'] = run_function(
        'plot_performance_metrics_summary',
        plot_performance_metrics_summary,
        config, plot_dirs
    )
    
    # ========================================================================
    # 2. PRECIPITATION AND EVAPOTRANSPIRATION
    # ========================================================================
    if verbose:
        print("\n" + "#"*80)
        print("# 2. PRECIPITATION AND EVAPOTRANSPIRATION")
        print("#"*80)
    
    results['precipitation_partitioning'] = run_function(
        'plot_precipitation_partitioning',
        plot_precipitation_partitioning,
        config, plot_dirs, validation_start, validation_end
    )
    
    results['actual_evapotranspiration'] = run_function(
        'plot_actual_evapotranspiration',
        plot_actual_evapotranspiration,
        config, plot_dirs, validation_start, validation_end
    )
    
    results['precipitation_aet_combined'] = run_function(
        'plot_precipitation_and_aet_combined',
        plot_precipitation_and_aet_combined,
        config, plot_dirs, validation_start, validation_end
    )
    
    # ========================================================================
    # 3. TEMPERATURE
    # ========================================================================
    if verbose:
        print("\n" + "#"*80)
        print("# 3. TEMPERATURE ANALYSIS")
        print("#"*80)
    
    results['temperature_by_elevation'] = run_function(
        'plot_temperature_by_elevation',
        plot_temperature_by_elevation,
        config, plot_dirs, validation_start, validation_end
    )
    
    # ========================================================================
    # 4. GLACIER ANALYSIS (GloGEM)
    # ========================================================================
    if verbose:
        print("\n" + "#"*80)
        print("# 4. GLACIER ANALYSIS (GloGEM)")
        print("#"*80)
    
    results['glogem_component_validation'] = run_function(
        'plot_glogem_component_validation',
        plot_glogem_component_validation,
        config, plot_dirs, validation_start, validation_end
    )
    
    results['glogem_regime'] = run_function(
        'plot_glogem_regime',
        plot_glogem_regime,
        config, plot_dirs
    )
    
    results['glogem_vs_observed_regime'] = run_function(
        'plot_glogem_vs_observed_regime',
        plot_glogem_vs_observed_regime,
        config, plot_dirs, validation_start, validation_end
    )
    
    results['irrigation_vs_glogem_regime'] = run_function(
        'plot_irrigation_vs_glogem_regime',
        plot_irrigation_vs_glogem_regime,
        config, plot_dirs, validation_start, validation_end
    )
    
    # ========================================================================
    # 5. SNOWMELT ANALYSIS
    # ========================================================================
    if verbose:
        print("\n" + "#"*80)
        print("# 5. SNOWMELT ANALYSIS")
        print("#"*80)
    
    results['snowmelt_timeseries'] = run_function(
        'plot_snowmelt_timeseries',
        plot_snowmelt_timeseries,
        config, plot_dirs, validation_start, validation_end
    )
    
    results['snowmelt_regime'] = run_function(
        'plot_snowmelt_regime',
        plot_snowmelt_regime,
        config, plot_dirs, validation_start, validation_end
    )
    
    results['snowmelt_comparison_lake_vs_mass'] = run_function(
        'plot_snowmelt_comparison_lake_vs_mass',
        plot_snowmelt_comparison_lake_vs_mass,
        config, plot_dirs, validation_start, validation_end
    )
    
    # ========================================================================
    # 6. GLACIER MELT ANALYSIS
    # ========================================================================
    if verbose:
        print("\n" + "#"*80)
        print("# 6. GLACIER MELT ANALYSIS")
        print("#"*80)
    
    results['glacier_melt_regime'] = run_function(
        'plot_glacier_melt_regime',
        plot_glacier_melt_regime,
        config, plot_dirs, validation_start, validation_end
    )
    
    results['glacier_melt_timeseries'] = run_function(
        'plot_glacier_melt_timeseries',
        plot_glacier_melt_timeseries,
        config, plot_dirs, validation_start, validation_end
    )
    
    # ========================================================================
    # 7. COMBINED STREAMFLOW REGIME ANALYSES
    # ========================================================================
    if verbose:
        print("\n" + "#"*80)
        print("# 7. COMBINED STREAMFLOW REGIME ANALYSES")
        print("#"*80)
    
    results['streamflow_with_all_glacier_snowmelt'] = run_function(
        'plot_streamflow_with_all_glacier_snowmelt_regime',
        plot_streamflow_with_all_glacier_snowmelt_regime,
        config, plot_dirs, validation_start, validation_end
    )
    
    results['streamflow_with_separated_glacier_snowmelt'] = run_function(
        'plot_streamflow_with_separated_glacier_snowmelt_regime',
        plot_streamflow_with_separated_glacier_snowmelt_regime,
        config, plot_dirs, validation_start, validation_end
    )
    
    results['streamflow_with_glogem_icemelt_snowmelt'] = run_function(
        'plot_streamflow_with_glogem_icemelt_and_total_snowmelt_regime',
        plot_streamflow_with_glogem_icemelt_and_total_snowmelt_regime,
        config, plot_dirs, validation_start, validation_end
    )
    
    # ========================================================================
    # 8. COMPREHENSIVE WATER BALANCE
    # ========================================================================
    if verbose:
        print("\n" + "#"*80)
        print("# 8. COMPREHENSIVE WATER BALANCE")
        print("#"*80)
    
    results['comprehensive_annual_water_balance'] = run_function(
        'plot_comprehensive_annual_water_balance',
        plot_comprehensive_annual_water_balance,
        config, plot_dirs, validation_start, validation_end
    )
    
    # ========================================================================
    # 9. SWE ANALYSIS
    # ========================================================================
    if verbose:
        print("\n" + "#"*80)
        print("# 9. SNOW WATER EQUIVALENT (SWE) ANALYSIS")
        print("#"*80)
    
    results['area_weighted_swe_timeseries'] = run_function(
        'plot_area_weighted_swe_timeseries',
        plot_area_weighted_swe_timeseries,
        config, plot_dirs, validation_start, validation_end
    )
    
    results['swe_time_series_by_elevation'] = run_function(
        'plot_swe_time_series_by_elevation',
        plot_swe_time_series_by_elevation,
        config, plot_dirs, None, validation_start, validation_end
    )
    
    results['peak_swe_analysis'] = run_function(
        'analyze_peak_swe',
        analyze_peak_swe,
        config, plot_dirs, validation_start, validation_end
    )
    
    results['spatial_swe_distribution'] = run_function(
        'plot_spatial_swe_distribution',
        plot_spatial_swe_distribution,
        config, plot_dirs, validation_start, validation_end
    )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    elapsed_time = time.time() - start_time
    
    if verbose:
        print("\n" + "="*80)
        print("POSTPROCESSING COMPLETE")
        print("="*80)
        print(f"\nTotal time: {elapsed_time/60:.1f} minutes")
        print(f"Successful analyses: {sum(1 for r in results.values() if r is not None)}/{len(results)}")
        
        if errors:
            print(f"\n⚠ Errors encountered: {len(errors)}")
            for error in errors:
                print(f"  - {error['function']}: {error['error']}")
        else:
            print("\n✓ All analyses completed successfully!")
        
        print(f"\nPlots saved to:")
        for key, path in plot_dirs.items():
            print(f"  - {key}: {path}")
        
        print("\n" + "="*80 + "\n")
    
    return {
        'results': results,
        'errors': errors,
        'elapsed_time': elapsed_time,
        'plot_directories': plot_dirs,
        'config': config
    }


