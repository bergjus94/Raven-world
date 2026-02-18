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
        'hydrographs': base_plots_dir / "hydrographs",
        'swe': base_plots_dir / "swe",
        'contributions': base_plots_dir / "contributions", 
        'parameters': base_plots_dir / "parameters",
        'storage': base_plots_dir / "storage"
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
###################################### SWE ######################################
#--------------------------------------------------------------------------------

def load_swe_data(config):
    """
    Load SWE data for ALL HRU groups from model output.
    Returns both the full dataframe and the specific NO_GLACIER column for catchment average.
    
    Returns:
    --------
    dict containing:
        'full_data': DataFrame with all HRU groups
        'catchment_avg': Series with catchment-average SWE (NO_GLACIER column)
        'hru_groups': List of HRU group column names
        'date_range': Tuple of (start_date, end_date)
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
        return None
    
    try:
        # Read the CSV file, skipping the units row (row index 1)
        df = pd.read_csv(swe_file, skiprows=[1])
        
        print(f"  - Loaded data shape: {df.shape}")
        print(f"  - Columns: {df.columns.tolist()}")
        
        # Create date range from config
        start_date = pd.to_datetime(config.get('start_date', '2000-01-01'))
        df['date'] = pd.date_range(start=start_date, periods=len(df), freq='D')
        
        print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Identify HRU group columns (exclude 'date', 'day', etc.)
        exclude_cols = ['date', 'day', 'time', 'Unnamed: 0', 'HRUGroup:']
        hru_groups = [col for col in df.columns if col not in exclude_cols]
        
        print(f"  - Found {len(hru_groups)} HRU groups: {hru_groups}")
        
        # Check if NO_GLACIER column exists
        if 'NO_GLACIER' not in df.columns:
            print(f"WARNING: 'NO_GLACIER' column not found in SWE file")
            print(f"  Available columns: {df.columns.tolist()}")
            catchment_avg = None
        else:
            catchment_avg = df['NO_GLACIER'].copy()
            # Convert from m to mm if needed
            if catchment_avg.mean() < 10 and catchment_avg.max() < 20:
                catchment_avg = catchment_avg * 1000
                print(f"  - Converted catchment average SWE from m to mm")
        
        # Convert all HRU groups from m to mm if needed
        for col in hru_groups:
            if df[col].mean() < 10 and df[col].max() < 20:
                df[col] = df[col] * 1000
        
        print(f"  ✓ Successfully loaded SWE data for all HRU groups")
        
        return {
            'full_data': df,
            'catchment_avg': catchment_avg,
            'hru_groups': hru_groups,
            'date_range': (df['date'].min(), df['date'].max())
        }
        
    except Exception as e:
        print(f"ERROR: Failed to load SWE data: {e}")
        import traceback
        traceback.print_exc()
        return None

#--------------------------------------------------------------------------------

def process_swe_data(sim_data, area_data=None):
    """
    Process simulated SWE data for single catchment analysis.
    Modified to work without observed data.
    """
    if sim_data is None:
        return None
    
    # Get elevation band columns efficiently using regex pattern
    sim_elev_pattern = re.compile(r'\d+-\d+m')
    sim_elev_cols = [col for col in sim_data.columns if sim_elev_pattern.search(col)]
    
    print(f"Found {len(sim_elev_cols)} simulation elevation bands")
    
    # Create area mapping if available
    area_mapping = {}
    if area_data is not None:
        print(f"  - Area data columns: {area_data.columns.tolist()}")
        
        # Check if the first column is unnamed but contains the elevation bands
        if 'Unnamed: 0' in area_data.columns and 'area_km2' in area_data.columns:
            area_dict = dict(zip(area_data['Unnamed: 0'].astype(str), area_data['area_km2']))
            
            matched_bands = 0
            for band in sim_elev_cols:
                if band in area_dict:
                    area_mapping[band] = area_dict[band]
                    matched_bands += 1
                    print(f"  - Found area for band {band}: {area_mapping[band]} km²")
                else:
                    print(f"  - Warning: No area data found for band {band}")
            
            print(f"  - Successfully matched {matched_bands}/{len(sim_elev_cols)} bands with area data")
    
    # Convert data to numeric
    for col in sim_elev_cols:
        sim_data[col] = pd.to_numeric(sim_data[col], errors='coerce')
    
    return {
        'sim_data': sim_data,
        'sim_elev_cols': sim_elev_cols,
        'area_mapping': area_mapping
    }

#--------------------------------------------------------------------------------

def calculate_area_weighted_swe(df, area_mapping):
    """
    Calculate area-weighted SWE for each time step using correct methodology.
    """
    swe_cols = [col for col in df.columns if col in area_mapping]
    if not swe_cols:
        return pd.Series(index=df.index, dtype=float)
    
    swe_array = df[swe_cols].copy()
    
    # Convert to mm if needed
    for col in swe_cols:
        vals = swe_array[col]
        if vals.mean() < 10 and vals.max() < 20:
            swe_array[col] = vals * 1000
    
    # Multiply each band by its area
    for col in swe_cols:
        swe_array[col] = swe_array[col] * area_mapping[col]
    
    # Calculate numerator (sum of SWE*area for valid bands)
    numerator = swe_array.sum(axis=1)
    
    # Calculate denominator (sum of areas for valid bands at each time step)
    valid_mask = df[swe_cols].notnull()
    denominator = valid_mask.astype(float) @ np.array([area_mapping[col] for col in swe_cols])
    
    # Calculate area-weighted average
    area_weighted_swe = numerator / denominator.replace(0, np.nan)
    
    return area_weighted_swe

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
    swe_data = load_swe_data(config)
    
    if swe_data is None:
        print("ERROR: Failed to load SWE data")
        return None
    
    if swe_data['catchment_avg'] is None:
        print("ERROR: NO_GLACIER column not available for catchment average")
        return None
    
    # Extract the full dataframe and catchment average
    df = swe_data['full_data']
    
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
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
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
    
    # ✅ FIX: Load SWE data using new format
    swe_data = load_swe_data(config)
    if swe_data is None:
        print("Failed to load SWE data for peak analysis")
        return None
    
    # Extract components from the dictionary
    full_data = swe_data['full_data']
    catchment_avg_swe = swe_data['catchment_avg']
    
    if catchment_avg_swe is None:
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
################################# contributions #################################
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
        
        # NEW: The file now has columns for all components:
        # - icemelt_glacier_area, icemelt_catchment_area
        # - snowmelt_glacier_area, snowmelt_catchment_area
        # - rain_glacier_area, rain_catchment_area
        # - melt_glacier_area, melt_catchment_area
        
        # Rename columns to match expected format (using glacier area values by default)
        result_df = pd.DataFrame({
            'date': glogem_filtered['date'],
            'icemelt': glogem_filtered['icemelt_glacier_area'],
            'snowmelt': glogem_filtered['snowmelt_glacier_area'],
            'rainfall': glogem_filtered['rain_glacier_area'],
            'glacier_melt': glogem_filtered['melt_glacier_area'],
            'total_output': glogem_filtered['melt_glacier_area'],
            # Also include catchment-normalized versions
            'icemelt_normalized': glogem_filtered['icemelt_catchment_area'],
            'snowmelt_normalized': glogem_filtered['snowmelt_catchment_area'],
            'rainfall_normalized': glogem_filtered['rain_catchment_area'],
            'glacier_melt_normalized': glogem_filtered['melt_catchment_area']
        })
        
        print(f"  ✓ Successfully loaded catchment-averaged GloGEM data")
        print(f"  - Date range: {result_df['date'].min()} to {result_df['date'].max()}")
        print(f"\n  Mean values (glacier area):")
        print(f"    Ice melt: {result_df['icemelt'].mean():.3f} mm/day")
        print(f"    Snow melt: {result_df['snowmelt'].mean():.3f} mm/day")
        print(f"    Rainfall: {result_df['rainfall'].mean():.3f} mm/day")
        print(f"    Total melt: {result_df['glacier_melt'].mean():.3f} mm/day")
        print(f"\n  Mean values (catchment area):")
        print(f"    Ice melt: {result_df['icemelt_normalized'].mean():.3f} mm/day")
        print(f"    Snow melt: {result_df['snowmelt_normalized'].mean():.3f} mm/day")
        print(f"    Rainfall: {result_df['rainfall_normalized'].mean():.3f} mm/day")
        print(f"    Total melt: {result_df['glacier_melt_normalized'].mean():.3f} mm/day")
        
        return result_df
        
    except Exception as e:
        print(f"ERROR: Failed to load GloGEM data: {e}")
        import traceback
        traceback.print_exc()
        return None

#--------------------------------------------------------------------------------

def plot_glogem_component_validation(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Diagnostic plot to verify that GloGEM components add up correctly.
    Compares:
    1. Total GloGEM output (glacier_melt_normalized from load_glogem_data)
    2. Sum of components (icemelt_normalized + snowmelt_normalized + rainfall_normalized)
    
    This helps identify any inconsistencies in the GloGEM data processing.
    """
    
    gauge_id = config['gauge_id']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"Validating GloGEM components for catchment {gauge_id}:")
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
    print(f"  - Columns: {glogem_filtered.columns.tolist()}")
    
    # Calculate sum of components (normalized values)
    glogem_filtered['calculated_total'] = (glogem_filtered['icemelt_normalized'] + 
                                           glogem_filtered['snowmelt_normalized'] + 
                                           glogem_filtered['rainfall_normalized'])
    
    # Get the reported total
    glogem_filtered['reported_total'] = glogem_filtered['glacier_melt_normalized']
    
    # Calculate difference
    glogem_filtered['difference'] = glogem_filtered['calculated_total'] - glogem_filtered['reported_total']
    
    # Print diagnostic statistics
    print(f"\n  Component Statistics (normalized, mm/day):")
    print(f"    Ice melt mean: {glogem_filtered['icemelt_normalized'].mean():.6f}")
    print(f"    Snowmelt mean: {glogem_filtered['snowmelt_normalized'].mean():.6f}")
    print(f"    Rainfall mean: {glogem_filtered['rainfall_normalized'].mean():.6f}")
    print(f"    Calculated total mean: {glogem_filtered['calculated_total'].mean():.6f}")
    print(f"    Reported total mean: {glogem_filtered['reported_total'].mean():.6f}")
    print(f"    Difference mean: {glogem_filtered['difference'].mean():.6f}")
    print(f"    Difference std: {glogem_filtered['difference'].std():.6f}")
    print(f"    Difference max: {glogem_filtered['difference'].max():.6f}")
    print(f"    Difference min: {glogem_filtered['difference'].min():.6f}")
    
    # Calculate monthly regimes
    glogem_filtered['month'] = glogem_filtered['date'].dt.month
    
    ice_regime = glogem_filtered.groupby('month')['icemelt_normalized'].mean()
    snow_regime = glogem_filtered.groupby('month')['snowmelt_normalized'].mean()
    rain_regime = glogem_filtered.groupby('month')['rainfall_normalized'].mean()
    calculated_total_regime = glogem_filtered.groupby('month')['calculated_total'].mean()
    reported_total_regime = glogem_filtered.groupby('month')['reported_total'].mean()
    
    # Create diagnostic plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # =============================
    # PLOT 1: TIME SERIES COMPARISON
    # =============================
    
    ax1.plot(glogem_filtered['date'], glogem_filtered['reported_total'], 
            'b-', linewidth=2, label='Reported Total (glacier_melt_normalized)', alpha=0.7)
    ax1.plot(glogem_filtered['date'], glogem_filtered['calculated_total'], 
            'r--', linewidth=2, label='Calculated Total (ice+snow+rain)', alpha=0.7)
    
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('GloGEM Output (mm/day)', fontsize=12, fontweight='bold')
    ax1.set_title('Time Series: Reported vs Calculated Total', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # =============================
    # PLOT 2: DIFFERENCE TIME SERIES
    # =============================
    
    ax2.plot(glogem_filtered['date'], glogem_filtered['difference'], 
            'purple', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Difference (mm/day)', fontsize=12, fontweight='bold')
    ax2.set_title('Difference: Calculated - Reported', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Add statistics text
    stats_text = (f"Difference Statistics:\n"
                 f"Mean: {glogem_filtered['difference'].mean():.6f}\n"
                 f"Std: {glogem_filtered['difference'].std():.6f}\n"
                 f"Max: {glogem_filtered['difference'].max():.6f}\n"
                 f"Min: {glogem_filtered['difference'].min():.6f}")
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    # =============================
    # PLOT 3: MONTHLY REGIME COMPARISON
    # =============================
    
    # Plot stacked components
    ax3.fill_between(months, 0, ice_regime.values, 
                    label='Ice Melt', color='grey', alpha=0.7)
    ax3.fill_between(months, ice_regime.values, 
                    ice_regime.values + snow_regime.values, 
                    label='Snowmelt', color='lightblue', alpha=0.7)
    ax3.fill_between(months, ice_regime.values + snow_regime.values, 
                    ice_regime.values + snow_regime.values + rain_regime.values, 
                    label='Rainfall', color='darkblue', alpha=0.7)
    
    # Plot reported total as line
    ax3.plot(months, reported_total_regime.values, 'r-', 
            linewidth=3, label='Reported Total', marker='o', markersize=8)
    
    ax3.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax3.set_ylabel('GloGEM Output (mm/day)', fontsize=12, fontweight='bold')
    ax3.set_title('Monthly Regime: Components vs Reported Total', fontsize=14, fontweight='bold')
    ax3.set_xticks(months)
    ax3.set_xticklabels(month_names, fontsize=11)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # =============================
    # PLOT 4: SCATTER PLOT
    # =============================
    
    ax4.scatter(glogem_filtered['reported_total'], glogem_filtered['calculated_total'], 
               alpha=0.5, s=20, c='steelblue', edgecolors='navy', linewidth=0.5)
    
    # Add 1:1 line
    max_val = max(glogem_filtered['reported_total'].max(), glogem_filtered['calculated_total'].max())
    min_val = min(glogem_filtered['reported_total'].min(), glogem_filtered['calculated_total'].min())
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, 
            label='1:1 Line', zorder=10)
    
    ax4.set_xlabel('Reported Total (mm/day)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Calculated Total (mm/day)', fontsize=12, fontweight='bold')
    ax4.set_title('Scatter: Reported vs Calculated', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # Calculate correlation
    corr = np.corrcoef(glogem_filtered['reported_total'], glogem_filtered['calculated_total'])[0, 1]
    
    # Add statistics text
    scatter_stats = (f"Statistics:\n"
                    f"R = {corr:.6f}\n"
                    f"Mean diff: {glogem_filtered['difference'].mean():.6f}\n"
                    f"RMSE: {np.sqrt(np.mean(glogem_filtered['difference']**2)):.6f}")
    ax4.text(0.02, 0.98, scatter_stats, transform=ax4.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    # Overall title
    fig.suptitle(f'GloGEM Component Validation - Catchment {gauge_id}\n'
                f'Period: {validation_start} to {validation_end}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'glogem_component_validation_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved component validation plot to: {save_path}")
    plt.show()
    
    # =============================
    # PRINT DETAILED COMPARISON
    # =============================
    
    print(f"\n{'='*60}")
    print(f"GLOGEM COMPONENT VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Catchment: {gauge_id}")
    print(f"Period: {validation_start} to {validation_end}")
    print(f"Records: {len(glogem_filtered)}")
    
    print(f"\nMonthly Regime Comparison (mm/day):")
    print(f"{'Month':<6} {'Ice':<8} {'Snow':<8} {'Rain':<8} {'Calc':<8} {'Report':<8} {'Diff':<8}")
    print(f"{'-'*60}")
    
    for month, ice, snow, rain, calc, report in zip(
        month_names, 
        ice_regime.values, 
        snow_regime.values, 
        rain_regime.values,
        calculated_total_regime.values,
        reported_total_regime.values
    ):
        diff = calc - report
        print(f"{month:<6} {ice:>7.4f} {snow:>7.4f} {rain:>7.4f} {calc:>7.4f} {report:>7.4f} {diff:>+7.4f}")
    
    # Check if components match
    max_diff = abs(glogem_filtered['difference']).max()
    mean_diff = abs(glogem_filtered['difference']).mean()
    
    print(f"\n{'='*60}")
    if max_diff < 0.001 and mean_diff < 0.0001:
        print(f"✅ VALIDATION PASSED: Components add up correctly!")
        print(f"   Max difference: {max_diff:.8f} mm/day")
        print(f"   Mean difference: {mean_diff:.8f} mm/day")
    else:
        print(f"⚠️  VALIDATION WARNING: Components don't match perfectly!")
        print(f"   Max difference: {max_diff:.8f} mm/day")
        print(f"   Mean difference: {mean_diff:.8f} mm/day")
        print(f"   This could indicate:")
        print(f"     - Different data sources used for total vs components")
        print(f"     - Rounding errors in normalization")
        print(f"     - Issues in GloGEM data processing")
    print(f"{'='*60}")
    
    return {
        'glogem_filtered': glogem_filtered,
        'monthly_regimes': {
            'ice': ice_regime,
            'snow': snow_regime,
            'rain': rain_regime,
            'calculated_total': calculated_total_regime,
            'reported_total': reported_total_regime
        },
        'statistics': {
            'max_difference': max_diff,
            'mean_difference': mean_diff,
            'correlation': corr,
            'rmse': np.sqrt(np.mean(glogem_filtered['difference']**2))
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

def plot_snowfall_rainfall_comparison(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot snowfall vs rainfall comparison for non-glacier areas (NO_GLACIER HRU group).
    Creates both time series and regime plots.
    
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
        DataFrame with date, snowfall, and rainfall data
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
    
    print(f"Loading snowfall and rainfall data for catchment {gauge_id}:")
    print(f"  - Period: {validation_start.date()} to {validation_end.date()}")
    
    # Define file paths
    output_dir = config_dir / f"catchment_{gauge_id}" / model_type / "output"
    snowfall_file = output_dir / f"{gauge_id}_{model_type}_SNOWFALL_Daily_Average_ByHRUGroup.csv"
    rainfall_file = output_dir / f"{gauge_id}_{model_type}_RAINFALL_Daily_Average_ByHRUGroup.csv"
    
    # Check if files exist
    if not snowfall_file.exists():
        print(f"ERROR: Snowfall file not found: {snowfall_file}")
        return None
    
    if not rainfall_file.exists():
        print(f"ERROR: Rainfall file not found: {rainfall_file}")
        return None
    
    try:
        # Load snowfall data (skip second row with units)
        df_snowfall = pd.read_csv(snowfall_file, skiprows=[1])
        print(f"  - Loaded snowfall data: {df_snowfall.shape}")
        print(f"  - Snowfall columns: {df_snowfall.columns.tolist()}")
        
        # Load rainfall data (skip second row with units)
        df_rainfall = pd.read_csv(rainfall_file, skiprows=[1])
        print(f"  - Loaded rainfall data: {df_rainfall.shape}")
        print(f"  - Rainfall columns: {df_rainfall.columns.tolist()}")
        
        # Create date column from start_date
        start_date = pd.to_datetime(config.get('start_date', '2000-01-01'))
        df_snowfall['date'] = pd.date_range(start=start_date, periods=len(df_snowfall), freq='D')
        df_rainfall['date'] = pd.date_range(start=start_date, periods=len(df_rainfall), freq='D')
        
        print(f"  - Date range: {df_snowfall['date'].min()} to {df_snowfall['date'].max()}")
        
        # Check if NO_GLACIER column exists in both files
        if 'NO_GLACIER' not in df_snowfall.columns:
            print(f"ERROR: 'NO_GLACIER' column not found in snowfall file")
            print(f"  Available columns: {df_snowfall.columns.tolist()}")
            return None
        
        if 'NO_GLACIER' not in df_rainfall.columns:
            print(f"ERROR: 'NO_GLACIER' column not found in rainfall file")
            print(f"  Available columns: {df_rainfall.columns.tolist()}")
            return None
        
        # Extract NO_GLACIER columns
        combined_df = pd.DataFrame({
            'date': df_snowfall['date'],
            'snowfall': df_snowfall['NO_GLACIER'],
            'rainfall': df_rainfall['NO_GLACIER']
        })
        
        # Filter for validation period
        mask = (combined_df['date'] >= validation_start) & (combined_df['date'] <= validation_end)
        df_filtered = combined_df[mask].copy()
        
        if len(df_filtered) == 0:
            print(f"ERROR: No data found for period {validation_start.date()} to {validation_end.date()}")
            return None
        
        print(f"  - Filtered to {len(df_filtered)} records")
        print(f"  - Mean snowfall: {df_filtered['snowfall'].mean():.2f} mm/day")
        print(f"  - Mean rainfall: {df_filtered['rainfall'].mean():.2f} mm/day")
        print(f"  - Max snowfall: {df_filtered['snowfall'].max():.2f} mm/day")
        print(f"  - Max rainfall: {df_filtered['rainfall'].max():.2f} mm/day")
        
    except Exception as e:
        print(f"ERROR: Failed to load precipitation data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # =============================
    # PLOT 1: TIME SERIES
    # =============================
    
    print(f"\nCreating time series plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # Top plot: Snowfall
    ax1.fill_between(df_filtered['date'], 0, df_filtered['snowfall'], 
                     color='lightblue', alpha=0.7, edgecolor='blue', linewidth=0.5)
    ax1.set_ylabel('Snowfall (mm/day)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Snowfall on Non-Glacier Areas (NO_GLACIER HRU Group)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text box
    snowfall_stats = (f"Statistics:\n"
                     f"Mean: {df_filtered['snowfall'].mean():.2f} mm/day\n"
                     f"Max: {df_filtered['snowfall'].max():.2f} mm/day\n"
                     f"Total: {df_filtered['snowfall'].sum():.1f} mm")
    ax1.text(0.02, 0.98, snowfall_stats, transform=ax1.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    # Bottom plot: Rainfall
    ax2.fill_between(df_filtered['date'], 0, df_filtered['rainfall'], 
                     color='darkblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
    ax2.set_ylabel('Rainfall (mm/day)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_title(f'Rainfall on Non-Glacier Areas (NO_GLACIER HRU Group)', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text box
    rainfall_stats = (f"Statistics:\n"
                     f"Mean: {df_filtered['rainfall'].mean():.2f} mm/day\n"
                     f"Max: {df_filtered['rainfall'].max():.2f} mm/day\n"
                     f"Total: {df_filtered['rainfall'].sum():.1f} mm")
    ax2.text(0.02, 0.98, rainfall_stats, transform=ax2.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gcf().autofmt_xdate()
    
    # Overall title
    fig.suptitle(f'Snowfall vs Rainfall Time Series - Catchment {gauge_id}\n'
                f'Period: {validation_start.date()} to {validation_end.date()}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save time series plot
    save_path_ts = plot_dirs['contributions'] / f'snowfall_rainfall_timeseries_{gauge_id}.png'
    plt.savefig(save_path_ts, dpi=300, bbox_inches='tight')
    print(f"Saved time series plot to: {save_path_ts}")
    plt.show()
    
    # =============================
    # PLOT 2: MONTHLY REGIME
    # =============================
    
    print(f"\nCreating regime plot...")
    
    # Calculate monthly regime
    df_filtered['month'] = df_filtered['date'].dt.month
    
    snowfall_regime = df_filtered.groupby('month')['snowfall'].mean()
    rainfall_regime = df_filtered.groupby('month')['rainfall'].mean()
    
    # Calculate total precipitation regime
    total_precip_regime = snowfall_regime + rainfall_regime
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot stacked bar chart
    width = 0.8
    
    # Snowfall bars (bottom)
    plt.bar(months, snowfall_regime.values, width, 
           label='Snowfall', color='lightblue', alpha=0.8, 
           edgecolor='blue', linewidth=1.5)
    
    # Rainfall bars (on top of snowfall)
    plt.bar(months, rainfall_regime.values, width, 
           bottom=snowfall_regime.values,
           label='Rainfall', color='darkblue', alpha=0.8, 
           edgecolor='navy', linewidth=1.5)
    
    # Plot total precipitation as a line
    plt.plot(months, total_precip_regime.values, 'ro-', 
            linewidth=3, markersize=8, label='Total Precipitation')
    
    # Formatting
    plt.xlabel('Month', fontsize=14, fontweight='bold')
    plt.ylabel('Precipitation (mm/day)', fontsize=14, fontweight='bold')
    plt.title(f'Snowfall vs Rainfall Monthly Regime - Catchment {gauge_id}\n'
             f'Non-Glacier Areas (NO_GLACIER HRU Group)', 
             fontsize=16, fontweight='bold')
    plt.xticks(months, month_names, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y', zorder=0)
    plt.legend(fontsize=12, loc='best')
    
    # Add statistics text box
    total_snowfall_annual = df_filtered.groupby(df_filtered['date'].dt.year)['snowfall'].sum().mean()
    total_rainfall_annual = df_filtered.groupby(df_filtered['date'].dt.year)['rainfall'].sum().mean()
    total_precip_annual = total_snowfall_annual + total_rainfall_annual
    snowfall_fraction = (total_snowfall_annual / total_precip_annual) * 100
    rainfall_fraction = (total_rainfall_annual / total_precip_annual) * 100
    
    regime_stats = (f"Annual Averages:\n"
                   f"Snowfall: {total_snowfall_annual:.1f} mm/year ({snowfall_fraction:.1f}%)\n"
                   f"Rainfall: {total_rainfall_annual:.1f} mm/year ({rainfall_fraction:.1f}%)\n"
                   f"Total: {total_precip_annual:.1f} mm/year\n\n"
                   f"Peak Months:\n"
                   f"Snowfall: {month_names[snowfall_regime.idxmax()-1]}\n"
                   f"Rainfall: {month_names[rainfall_regime.idxmax()-1]}\n"
                   f"Total: {month_names[total_precip_regime.idxmax()-1]}")
    
    plt.text(0.02, 0.98, regime_stats, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save regime plot
    save_path_regime = plot_dirs['contributions'] / f'snowfall_rainfall_regime_{gauge_id}.png'
    plt.savefig(save_path_regime, dpi=300, bbox_inches='tight')
    print(f"Saved regime plot to: {save_path_regime}")
    plt.show()
    
    # =============================
    # PRINT SUMMARY STATISTICS
    # =============================
    
    print(f"\n{'='*60}")
    print(f"SNOWFALL VS RAINFALL SUMMARY - CATCHMENT {gauge_id}")
    print(f"{'='*60}")
    print(f"Period: {validation_start.date()} to {validation_end.date()}")
    print(f"Area: Non-glacier areas (NO_GLACIER HRU group)")
    
    print(f"\nDaily Averages:")
    print(f"  Snowfall: {df_filtered['snowfall'].mean():.2f} mm/day")
    print(f"  Rainfall: {df_filtered['rainfall'].mean():.2f} mm/day")
    print(f"  Total: {(df_filtered['snowfall'] + df_filtered['rainfall']).mean():.2f} mm/day")
    
    print(f"\nAnnual Totals:")
    print(f"  Snowfall: {total_snowfall_annual:.1f} mm/year ({snowfall_fraction:.1f}%)")
    print(f"  Rainfall: {total_rainfall_annual:.1f} mm/year ({rainfall_fraction:.1f}%)")
    print(f"  Total: {total_precip_annual:.1f} mm/year")
    
    print(f"\nPeak Months:")
    print(f"  Snowfall: {month_names[snowfall_regime.idxmax()-1]} ({snowfall_regime.max():.2f} mm/day)")
    print(f"  Rainfall: {month_names[rainfall_regime.idxmax()-1]} ({rainfall_regime.max():.2f} mm/day)")
    print(f"  Total: {month_names[total_precip_regime.idxmax()-1]} ({total_precip_regime.max():.2f} mm/day)")
    
    print(f"\nSeasonal Distribution:")
    winter_months = [12, 1, 2]
    spring_months = [3, 4, 5]
    summer_months = [6, 7, 8]
    fall_months = [9, 10, 11]
    
    for season, season_months in [('Winter', winter_months), ('Spring', spring_months), 
                                   ('Summer', summer_months), ('Fall', fall_months)]:
        snow_mean = snowfall_regime[snowfall_regime.index.isin(season_months)].mean()
        rain_mean = rainfall_regime[rainfall_regime.index.isin(season_months)].mean()
        total_mean = snow_mean + rain_mean
        
        print(f"  {season}:")
        print(f"    Snowfall: {snow_mean:.2f} mm/day ({(snow_mean/total_mean*100):.1f}%)")
        print(f"    Rainfall: {rain_mean:.2f} mm/day ({(rain_mean/total_mean*100):.1f}%)")
    
    print(f"{'='*60}")
    
    return df_filtered

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
    1. Lake storage file (TO_LAKE_STORAGE) - shows snowmelt going to lake storage (CUMULATIVE -> convert to daily rate)
    2. Mass loadings file (SNOWMELTMassLoadings) - shows snowmelt contribution to streamflow (already daily rate)
    
    Creates both time series and regime comparison plots.
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
    
    print(f"Loading and comparing snowmelt data for catchment {gauge_id}:")
    print(f"  - Period: {validation_start.date()} to {validation_end.date()}")
    
    # =============================
    # 1. LOAD LAKE STORAGE SNOWMELT (CUMULATIVE)
    # =============================
    
    output_dir = config_dir / f"catchment_{gauge_id}" / model_type / "output"
    lake_storage_file = output_dir / f"{gauge_id}_{model_type}_BETWEEN_SNOW_LIQ_AND_PONDED_WATER_Daily_Average_BySubbasin.csv"
 
    if not lake_storage_file.exists():
        print(f"ERROR: Lake storage file not found: {lake_storage_file}")
        return None
    
    try:
        # Read file, skip first row, second row has headers
        df_lake = pd.read_csv(lake_storage_file, skiprows=[0])
        print(f"  - Loaded lake storage data: {df_lake.shape}")
        print(f"  - Lake storage columns: {df_lake.columns.tolist()}")
        
        # Handle both 'date' and 'day' columns
        date_col = None
        if 'date' in df_lake.columns:
            date_col = 'date'
        elif 'day' in df_lake.columns:
            date_col = 'day'
        else:
            print(f"ERROR: No date/day column found in lake storage file")
            print(f"  Available columns: {df_lake.columns.tolist()}")
            return None
        
        # Check if mean column exists
        if 'mean' not in df_lake.columns:
            print(f"ERROR: 'mean' column not found in lake storage file")
            print(f"  Available columns: {df_lake.columns.tolist()}")
            return None
        
        # Parse dates - create from config start_date if needed
        try:
            df_lake['date'] = pd.to_datetime(df_lake[date_col])
        except:
            # If date parsing fails, create date range from config
            start_date_config = pd.to_datetime(config.get('start_date', '2000-01-01'))
            df_lake['date'] = pd.date_range(start=start_date_config, periods=len(df_lake), freq='D')
            print(f"  - Created date range from config start_date")
        
        # ✅ FIX: Convert CUMULATIVE snowmelt to DAILY RATE
        # The 'mean' column contains cumulative snowmelt, so we need to take the difference
        df_lake['cumulative_snowmelt'] = df_lake['mean']
        
        # Calculate daily snowmelt rate as the difference between consecutive days
        df_lake['snowmelt_lake'] = df_lake['cumulative_snowmelt'].diff().fillna(0)
        
        # Set negative values to zero (can happen at the start or with numerical issues)
        df_lake['snowmelt_lake'] = df_lake['snowmelt_lake'].clip(lower=0)
        
        print(f"  - Converted cumulative snowmelt to daily rate")
        print(f"  - Date range: {df_lake['date'].min()} to {df_lake['date'].max()}")
        
        # Filter for validation period
        lake_mask = (df_lake['date'] >= validation_start) & (df_lake['date'] <= validation_end)
        df_lake_filtered = df_lake[lake_mask].copy()
        
        if len(df_lake_filtered) == 0:
            print(f"ERROR: No lake storage data found for period {validation_start.date()} to {validation_end.date()}")
            return None
        
        print(f"  - Filtered to {len(df_lake_filtered)} records")
        print(f"  - Mean lake storage snowmelt: {df_lake_filtered['snowmelt_lake'].mean():.4f} mm/day")
        print(f"  - Max lake storage snowmelt: {df_lake_filtered['snowmelt_lake'].max():.4f} mm/day")
        print(f"  - Sample values (first 5 days after conversion):")
        for idx, row in df_lake_filtered.head().iterrows():
            print(f"      {row['date'].date()}: {row['snowmelt_lake']:.4f} mm/day (cumulative: {row['cumulative_snowmelt']:.4f})")
        
    except Exception as e:
        print(f"ERROR: Failed to load lake storage data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # =============================
    # 2. LOAD MASS LOADINGS SNOWMELT (already daily rate)
    # =============================
    
    print(f"\n  - Loading mass loadings snowmelt data...")
    df_mass = load_snowmelt_mass_loadings(config, validation_start, validation_end, unit='mm')
    
    if df_mass is None:
        print(f"ERROR: Could not load mass loadings snowmelt data")
        return None
    
    # Ensure we have the mm/day column
    if 'snowmelt_mm_day' not in df_mass.columns:
        print(f"ERROR: 'snowmelt_mm_day' column not found in mass loadings data")
        return None
    
    print(f"  - Mean mass loadings snowmelt: {df_mass['snowmelt_mm_day'].mean():.4f} mm/day")
    print(f"  - Max mass loadings snowmelt: {df_mass['snowmelt_mm_day'].max():.4f} mm/day")
    
    # =============================
    # 3. MERGE THE TWO DATASETS
    # =============================
    
    # Merge on date
    df_combined = pd.merge(
        df_lake_filtered[['date', 'snowmelt_lake']], 
        df_mass[['date', 'snowmelt_mm_day']], 
        on='date', 
        how='inner'
    )
    
    if len(df_combined) == 0:
        print(f"ERROR: No overlapping dates between the two datasets")
        return None
    
    print(f"\n  - Combined dataset: {len(df_combined)} records")
    
    # Calculate statistics
    correlation = np.corrcoef(df_combined['snowmelt_lake'].values, 
                            df_combined['snowmelt_mm_day'].values)[0, 1]
    bias = df_combined['snowmelt_lake'].mean() - df_combined['snowmelt_mm_day'].mean()
    rmse = np.sqrt(np.mean((df_combined['snowmelt_lake'] - df_combined['snowmelt_mm_day'])**2))
    
    print(f"\n  Comparison Statistics:")
    print(f"    Correlation: {correlation:.3f}")
    print(f"    Mean bias: {bias:.4f} mm/day")
    print(f"    RMSE: {rmse:.4f} mm/day")
    
    # =============================
    # PLOT 1: TIME SERIES COMPARISON
    # =============================
    
    print(f"\nCreating time series comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # Top plot: Lake storage snowmelt
    ax1.fill_between(df_combined['date'], 0, df_combined['snowmelt_lake'], 
                     color='steelblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
    ax1.set_ylabel('Snowmelt to Lake Storage (mm/day)', fontsize=12, fontweight='bold')
    ax1.set_title('Snowmelt to Lake Storage (TO_LAKE_STORAGE)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text box
    lake_stats = (f"Statistics:\n"
                 f"Mean: {df_combined['snowmelt_lake'].mean():.4f} mm/day\n"
                 f"Max: {df_combined['snowmelt_lake'].max():.4f} mm/day\n"
                 f"Total: {df_combined['snowmelt_lake'].sum():.2f} mm")
    ax1.text(0.02, 0.98, lake_stats, transform=ax1.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    # Bottom plot: Mass loadings snowmelt
    ax2.fill_between(df_combined['date'], 0, df_combined['snowmelt_mm_day'], 
                     color='deepskyblue', alpha=0.7, edgecolor='blue', linewidth=0.5)
    ax2.set_ylabel('Snowmelt Mass Loadings (mm/day)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_title('Snowmelt Mass Loadings to Streamflow (SNOWMELTMassLoadings)', 
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
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gcf().autofmt_xdate()
    
    # Overall title
    fig.suptitle(f'Snowmelt Comparison: Lake Storage vs Mass Loadings\nCatchment {gauge_id}\n'
                f'Period: {validation_start.date()} to {validation_end.date()}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save time series plot
    save_path_ts = plot_dirs['contributions'] / f'snowmelt_comparison_timeseries_{gauge_id}.png'
    plt.savefig(save_path_ts, dpi=300, bbox_inches='tight')
    print(f"Saved time series comparison plot to: {save_path_ts}")
    plt.show()
    
    # =============================
    # PLOT 2: MONTHLY REGIME COMPARISON
    # =============================
    
    print(f"\nCreating regime comparison plot...")
    
    # Calculate monthly regimes
    df_combined['month'] = df_combined['date'].dt.month
    
    lake_regime = df_combined.groupby('month')['snowmelt_lake'].mean()
    mass_regime = df_combined.groupby('month')['snowmelt_mm_day'].mean()
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot both regimes
    plt.plot(months, lake_regime.values, 'steelblue', linewidth=3, 
            label='Lake Storage Snowmelt', marker='o', markersize=8)
    
    plt.plot(months, mass_regime.values, 'deepskyblue', linewidth=3, 
            label='Mass Loadings Snowmelt', marker='s', markersize=8, linestyle='--')
    
    # Fill area between to show difference
    plt.fill_between(months, lake_regime.values, mass_regime.values, 
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
                 f"Lake Storage mean: {lake_regime.mean():.4f} mm/day\n"
                 f"Mass Loadings mean: {mass_regime.mean():.4f} mm/day\n\n"
                 f"Lake peak: {month_names[lake_regime.idxmax()-1]}\n"
                 f"Mass peak: {month_names[mass_regime.idxmax()-1]}")
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save regime plot
    save_path_regime = plot_dirs['contributions'] / f'snowmelt_comparison_regime_{gauge_id}.png'
    plt.savefig(save_path_regime, dpi=300, bbox_inches='tight')
    print(f"Saved regime comparison plot to: {save_path_regime}")
    plt.show()
    
    # =============================
    # PLOT 3: SCATTER PLOT
    # =============================
    
    print(f"\nCreating scatter plot...")
    
    plt.figure(figsize=(10, 10))
    
    # Create scatter plot
    plt.scatter(df_combined['snowmelt_mm_day'], df_combined['snowmelt_lake'], 
               alpha=0.5, s=20, c='steelblue', edgecolors='navy', linewidth=0.5)
    
    # Add 1:1 line
    max_val = max(df_combined['snowmelt_mm_day'].max(), df_combined['snowmelt_lake'].max())
    min_val = min(df_combined['snowmelt_mm_day'].min(), df_combined['snowmelt_lake'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, 
            label='1:1 Line', zorder=10)
    
    # Add regression line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(
        df_combined['snowmelt_mm_day'].values, 
        df_combined['snowmelt_lake'].values
    )
    
    line_x = np.array([min_val, max_val])
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, 'r-', linewidth=2, 
            label=f'Regression (R²={r_value**2:.3f})', zorder=9)
    
    # Formatting
    plt.xlabel('Mass Loadings Snowmelt (mm/day)', fontsize=14, fontweight='bold')
    plt.ylabel('Lake Storage Snowmelt (mm/day)', fontsize=14, fontweight='bold')
    plt.title(f'Snowmelt Comparison Scatter Plot\nCatchment {gauge_id}', 
             fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add statistics text box
    scatter_stats = (f"Statistics:\n"
                    f"R² = {r_value**2:.3f}\n"
                    f"Slope = {slope:.3f}\n"
                    f"Intercept = {intercept:.3f}\n"
                    f"Correlation = {correlation:.3f}\n"
                    f"RMSE = {rmse:.4f} mm/day\n"
                    f"Bias = {bias:.4f} mm/day\n"
                    f"n = {len(df_combined)}")
    
    plt.text(0.02, 0.98, scatter_stats, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save scatter plot
    save_path_scatter = plot_dirs['contributions'] / f'snowmelt_comparison_scatter_{gauge_id}.png'
    plt.savefig(save_path_scatter, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot to: {save_path_scatter}")
    plt.show()
    
    # =============================
    # PRINT SUMMARY STATISTICS
    # =============================
    
    print(f"\n{'='*60}")
    print(f"SNOWMELT COMPARISON SUMMARY - CATCHMENT {gauge_id}")
    print(f"{'='*60}")
    print(f"Period: {validation_start.date()} to {validation_end.date()}")
    print(f"Number of records: {len(df_combined)}")
    
    print(f"\nDaily Averages:")
    print(f"  Lake Storage: {df_combined['snowmelt_lake'].mean():.4f} mm/day")
    print(f"  Mass Loadings: {df_combined['snowmelt_mm_day'].mean():.4f} mm/day")
    print(f"  Difference: {bias:.4f} mm/day")
    
    print(f"\nTotal Snowmelt:")
    print(f"  Lake Storage: {df_combined['snowmelt_lake'].sum():.2f} mm")
    print(f"  Mass Loadings: {df_combined['snowmelt_mm_day'].sum():.2f} mm")
    print(f"  Difference: {df_combined['snowmelt_lake'].sum() - df_combined['snowmelt_mm_day'].sum():.2f} mm")
    
    print(f"\nComparison Metrics:")
    print(f"  Correlation: {correlation:.3f}")
    print(f"  R²: {r_value**2:.3f}")
    print(f"  RMSE: {rmse:.4f} mm/day")
    print(f"  Mean bias: {bias:.4f} mm/day")
    print(f"  Relative bias: {(bias/df_combined['snowmelt_mm_day'].mean()*100):.1f}%")
    
    print(f"\nPeak Months:")
    print(f"  Lake Storage: {month_names[lake_regime.idxmax()-1]} ({lake_regime.max():.4f} mm/day)")
    print(f"  Mass Loadings: {month_names[mass_regime.idxmax()-1]} ({mass_regime.max():.4f} mm/day)")
    
    print(f"{'='*60}\n")
    
    return {
        'combined_data': df_combined,
        'lake_regime': lake_regime,
        'mass_regime': mass_regime,
        'statistics': {
            'correlation': correlation,
            'r_squared': r_value**2,
            'rmse': rmse,
            'bias': bias,
            'slope': slope,
            'intercept': intercept
        },
        'plots': {
            'timeseries': save_path_ts,
            'regime': save_path_regime,
            'scatter': save_path_scatter
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
        mean_annual = glacier_data[glacier_type].groupby('year')[glacier_melt_col].sum().mean()
        max_daily = glacier_data[glacier_type][glacier_melt_col].max()
        peak_month = month_names[monthly_regime.idxmax()-1]
        
        stats_lines.append(f"{glacier_type.upper()}:")
        stats_lines.append(f"  Annual: {mean_annual:.1f} {unit_label}·year")
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
        mean_annual = df.groupby('year')[glacier_melt_col].sum().mean()
        max_daily = df[glacier_melt_col].max()
        peak_month = month_names[monthly_regime.idxmax()-1]
        min_month = month_names[monthly_regime.idxmin()-1]
        
        print(f"\n  {glacier_type.upper()} Glaciers:")
        print(f"    Mean annual glacier melt: {mean_annual:.1f} {unit_label}·year")
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

#--------------------------------------------------------------------------------

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
        print(f"   WARNING: Could not load HBV rainfall data")
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

def plot_glogem_vs_observed_streamflow_regime(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot monthly regime comparing GloGEM total output vs observed streamflow.
    Shows how well GloGEM's total glacier contribution matches observed discharge patterns.
    
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
        Dictionary containing monthly regime data for both datasets
    """
    
    gauge_id = config['gauge_id']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"Creating GloGEM vs Observed Streamflow regime for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    
    # 1. Load GloGEM data
    glogem_df = load_glogem_data(config, unit='mm', plot=False)
    if glogem_df is None:
        print("ERROR: Could not load GloGEM data")
        return None
    
    # Filter GloGEM data for validation period
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    
    glogem_mask = (glogem_df['date'] >= start_date) & (glogem_df['date'] <= end_date)
    glogem_filtered = glogem_df[glogem_mask].copy()
    
    if len(glogem_filtered) == 0:
        print(f"ERROR: No GloGEM data found for period {validation_start} to {validation_end}")
        return None
    
    # ✅ FIX: Use the SAME columns as in the validation function!
    # The validation function uses: icemelt_normalized, snowmelt_normalized, rainfall_normalized
    # Calculate monthly regime for GloGEM total output
    glogem_filtered['month'] = glogem_filtered['date'].dt.month
    
    # ✅ Use the SAME calculation as in plot_glogem_component_validation
    glogem_filtered['calculated_total'] = (glogem_filtered['icemelt_normalized'] + 
                                           glogem_filtered['snowmelt_normalized'] + 
                                           glogem_filtered['rainfall_normalized'])
    
    # Calculate monthly mean regime using the CALCULATED total (sum of components)
    glogem_regime = glogem_filtered.groupby('month')['calculated_total'].mean()
    
    print(f"  - GloGEM total output mean (catchment-normalized): {glogem_regime.mean():.4f} mm/day")
    
    # 2. Load observed streamflow data
    streamflow_data = load_hydrograph_data(config)
    if streamflow_data is None:
        print("ERROR: Could not load streamflow data")
        return None
    
    # Filter streamflow for validation period
    streamflow_mask = (streamflow_data['date'] >= start_date) & (streamflow_data['date'] <= end_date)
    streamflow_filtered = streamflow_data[streamflow_mask].copy()
    
    if len(streamflow_filtered) == 0:
        print(f"ERROR: No streamflow data found for period {validation_start} to {validation_end}")
        return None
    
    # Convert observed streamflow from m³/s to mm/day
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
            streamflow_filtered['obs_Q_mm_day'] = streamflow_filtered['obs_Q'] * conversion_m3s_to_mm_day
            print(f"  - Catchment area: {total_area_km2:.2f} km²")
            print(f"  - Conversion factor: {conversion_m3s_to_mm_day:.6f}")
        else:
            print(f"ERROR: Catchment shapefile not found: {catchment_shape_file}")
            return None
    except Exception as e:
        print(f"ERROR: Could not convert streamflow units: {e}")
        return None
    
    # Calculate monthly regime for observed streamflow
    streamflow_filtered['month'] = streamflow_filtered['date'].dt.month
    obs_regime = streamflow_filtered.groupby('month')['obs_Q_mm_day'].mean()
    
    print(f"  - Observed streamflow mean: {obs_regime.mean():.4f} mm/day")
    
    # 3. Create plot
    plt.figure(figsize=(14, 8))
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot observed streamflow
    plt.plot(months, obs_regime.values, 'k-', linewidth=3, 
            label='Observed Streamflow', marker='o', markersize=8, zorder=4)
    
    # Plot GloGEM total output (catchment-normalized, sum of components)
    plt.plot(months, glogem_regime.values, 'C3--', linewidth=2.5, 
            label='GloGEM Total Output (Ice+Snow+Rain)', marker='s', markersize=8, zorder=3)
    
    # Fill area between the two lines to show difference
    plt.fill_between(months, obs_regime.values, glogem_regime.values, 
                     alpha=0.2, color='gray', label='Difference')
    
    # Formatting
    plt.xlabel('Month', fontsize=14, fontweight='bold')
    plt.ylabel('Discharge (mm/day)', fontsize=14, fontweight='bold')
    plt.title(f'GloGEM Total Output vs Observed Streamflow Regime\nCatchment {gauge_id} (Catchment-Normalized)', 
             fontsize=16, fontweight='bold')
    plt.xticks(months, month_names, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.legend(fontsize=12, loc='best')
    
    # Add statistics text box
    # Calculate correlation and bias
    corr = np.corrcoef(glogem_regime.values, obs_regime.values)[0, 1]
    bias = glogem_regime.mean() - obs_regime.mean()
    relative_bias = (bias / obs_regime.mean()) * 100
    rmse = np.sqrt(np.mean((glogem_regime.values - obs_regime.values)**2))
    
    stats_text = (f"Statistics:\n"
                 f"GloGEM mean: {glogem_regime.mean():.4f} mm/day\n"
                 f"Observed mean: {obs_regime.mean():.4f} mm/day\n"
                 f"Correlation: {corr:.3f}\n"
                 f"Bias: {bias:.4f} mm/day ({relative_bias:+.1f}%)\n"
                 f"RMSE: {rmse:.4f} mm/day")
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'glogem_vs_observed_regime_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved GloGEM vs Observed regime plot to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\nGloGEM vs Observed Streamflow Regime Summary:")
    print(f"  Period: {validation_start} to {validation_end}")
    print(f"  GloGEM total output mean (sum of ice+snow+rain): {glogem_regime.mean():.4f} mm/day")
    print(f"  Observed streamflow mean: {obs_regime.mean():.4f} mm/day")
    print(f"  Correlation: {corr:.3f}")
    print(f"  Mean bias: {bias:.4f} mm/day ({relative_bias:+.1f}%)")
    print(f"  RMSE: {rmse:.4f} mm/day")
    print(f"  Peak month (GloGEM): {month_names[glogem_regime.idxmax()-1]}")
    print(f"  Peak month (Observed): {month_names[obs_regime.idxmax()-1]}")
    
    # Monthly comparison
    print(f"\nMonthly Comparison:")
    for month, glogem_val, obs_val in zip(month_names, glogem_regime.values, obs_regime.values):
        diff = glogem_val - obs_val
        diff_pct = (diff / obs_val * 100) if obs_val > 0 else 0
        print(f"  {month}: GloGEM={glogem_val:.4f}, Obs={obs_val:.4f}, "
              f"Diff={diff:+.4f} ({diff_pct:+.1f}%)")
    
    # Return data
    return {
        'glogem_regime': glogem_regime,
        'observed_regime': obs_regime,
        'correlation': corr,
        'bias': bias,
        'relative_bias_pct': relative_bias,
        'rmse': rmse,
        'glogem_peak_month': month_names[glogem_regime.idxmax()-1],
        'observed_peak_month': month_names[obs_regime.idxmax()-1]
    }

#--------------------------------------------------------------------------------
################################# Uncertainties #################################
#--------------------------------------------------------------------------------

def fill_templates_with_parameters(run_dir, param_cols, param_values):
    """
    Fill template files in run_dir with parameter values and rename them properly.
    Template files should have .tpl extension and contain parameter placeholders.
    Also calculates and replaces tied parameters.
    """
    run_dir = Path(run_dir)
    
    # Find all .tpl files
    template_files = list(run_dir.glob("*.tpl"))
    
    if not template_files:
        print(f"No .tpl template files found in {run_dir}")
        available_files = list(run_dir.glob("*"))
        print(f"Available files: {[f.name for f in available_files]}")
        return False
    
    print(f"Processing {len(template_files)} template files...")
    
    # Create parameter dictionary from columns and values
    params_dict = {}
    for col, val in zip(param_cols, param_values):
        # Remove 'par' prefix if present
        if col.startswith('par'):
            param_name = col[3:]  # Remove 'par' prefix
        else:
            param_name = col
        params_dict[param_name] = val
    
    # Add tied parameters
    print("Calculating tied parameters...")
    
    # 1. Add HBV_Time_To_Peak (tied to HBV_T_Conc_Max_Bas)
    if 'HBV_T_Conc_Max_Bas' in params_dict:
        time_to_peak = 0.5 * params_dict['HBV_T_Conc_Max_Bas']
        params_dict['HBV_Time_To_Peak'] = time_to_peak
        print(f"  Calculated HBV_Time_To_Peak = 0.5 * {params_dict['HBV_T_Conc_Max_Bas']} = {time_to_peak}")
    
    # 2. Add HBV_Initial_Thickness_Topsoil (tied to HBV_Thickness_Topsoil)
    if 'HBV_Thickness_Topsoil' in params_dict:
        initial_thickness = 500 * params_dict['HBV_Thickness_Topsoil']
        params_dict['HBV_Initial_Thickness_Topsoil'] = initial_thickness
        print(f"  Calculated HBV_Initial_Thickness_Topsoil = 500 * {params_dict['HBV_Thickness_Topsoil']} = {initial_thickness}")
    
    # Debug: Print all parameters (including tied ones)
    print(f"Parameters to replace (including tied): {len(params_dict)}")
    for param_name, param_value in list(params_dict.items())[:7]:  # Show first 7 for debugging
        print(f"  {param_name} = {param_value}")
    
    replacements_made = 0
    
    for template_file in template_files:
        try:
            # Read template content
            with open(template_file, 'r') as f:
                content = f.read()
            
            original_content = content
            file_replacements = 0
            
            # Debug: Show a sample of the template content
            print(f"\nProcessing {template_file.name}...")
            print(f"Template content preview (first 200 chars):")
            print(content[:200])
            
            # Replace parameter placeholders with actual values (including tied parameters)
            for param_name, param_value in params_dict.items():
                print(f"  Looking for parameter: {param_name}")
                
                # Try different placeholder formats that are commonly used
                placeholders = [
                    f"{{{param_name}}}",           # {HBV_RainSnow_Temp}
                    f"${{{param_name}}}",          # ${HBV_RainSnow_Temp}
                    f"@{param_name}@",             # @HBV_RainSnow_Temp@
                    f"#{param_name}#",             # #HBV_RainSnow_Temp#
                    f"%{param_name}%",             # %HBV_RainSnow_Temp%
                    f"<{param_name}>",             # <HBV_RainSnow_Temp>
                    f"[{param_name}]",             # [HBV_RainSnow_Temp]
                    param_name,                    # HBV_RainSnow_Temp (direct replacement)
                    f"__{param_name}__",           # __HBV_RainSnow_Temp__
                ]
                
                for placeholder in placeholders:
                    if placeholder in content:
                        old_content = content
                        content = content.replace(placeholder, str(param_value))
                        if content != old_content:
                            print(f"  ✓ Replaced {placeholder} with {param_value}")
                            file_replacements += 1
                            replacements_made += 1
                            break  # Found the right format, no need to try others for this parameter
            
            # Create output filename by removing .tpl extension
            output_file = template_file.with_suffix('')
            
            # Write the filled template to the new file
            with open(output_file, 'w') as f:
                f.write(content)
            
            print(f"  Created: {output_file.name} ({file_replacements} replacements)")
            
            # If no replacements were made in this file, show some content for debugging
            if file_replacements == 0:
                print(f"  WARNING: No replacements made in {template_file.name}")
                print(f"  File contains: {content[:500]}...")
                
                # Look for any parameter names that might be in the file
                found_params = []
                for param_name in params_dict.keys():
                    if param_name in content:
                        found_params.append(param_name)
                
                if found_params:
                    print(f"  Found parameter names in file: {found_params[:5]}")
                    print("  These might need different placeholder formats")
                
        except Exception as e:
            print(f"Error processing template {template_file}: {e}")
            return False
    
    print(f"\nTotal replacements made across all files: {replacements_made}")
    
    if replacements_made == 0:
        print("\n❌ WARNING: NO PARAMETER REPLACEMENTS WERE MADE!")
        print("This means the placeholder format in your template files doesn't match what we're looking for.")
        print("Please check your template files to see what format they use for parameter placeholders.")
        
        # Show content of the first template file for inspection
        if template_files:
            print(f"\nSample content from {template_files[0].name}:")
            with open(template_files[0], 'r') as f:
                sample_content = f.read()
            print(sample_content[:1000])  # Show first 1000 characters
        
        return False
    
    return True

#--------------------------------------------------------------------------------

def setup_raven_run_directory(run_dir, config):
    """
    Set up the proper directory structure for a Raven run.
    Copy data_obs folder and create output directory.
    """
    run_dir = Path(run_dir)
    
    # Source data_obs directory from config
    config_dir = Path(config['main_dir']) / config['config_dir']
    gauge_id = config['gauge_id']
    model_type = config['model_type']
    
    source_data_obs = config_dir / f"catchment_{gauge_id}" / model_type / "data_obs"
    
    if not source_data_obs.exists():
        print(f"Error: Source data_obs directory not found: {source_data_obs}")
        return False
    
    # Copy data_obs directory to run directory
    target_data_obs = run_dir / "data_obs"
    if target_data_obs.exists():
        import shutil
        shutil.rmtree(target_data_obs)
    
    try:
        import shutil
        shutil.copytree(source_data_obs, target_data_obs)
        print(f"  Copied data_obs directory")
    except Exception as e:
        print(f"  Error copying data_obs: {e}")
        return False
    
    # Create output directory
    output_dir = run_dir / "output"
    output_dir.mkdir(exist_ok=True)
    print(f"  Created output directory")
    
    return True

#--------------------------------------------------------------------------------

def cleanup_raven_run_directory(run_dir):
    """
    Clean up the run directory by removing data_obs folder but keeping output.
    """
    run_dir = Path(run_dir)
    
    # Remove data_obs directory to save space
    data_obs_dir = run_dir / "data_obs"
    if data_obs_dir.exists():
        try:
            import shutil
            shutil.rmtree(data_obs_dir)
            print(f"  Cleaned up data_obs directory")
        except Exception as e:
            print(f"  Warning: Could not remove data_obs: {e}")
    
    # Remove template files (.tpl) to save space
    for tpl_file in run_dir.glob("*.tpl"):
        try:
            tpl_file.unlink()
        except Exception as e:
            print(f"  Warning: Could not remove {tpl_file}: {e}")

#--------------------------------------------------------------------------------

def plot_test_results(config, hydrographs, validation_start, validation_end, 
                     color_best, color_others, sim_results_dir):
    """
    Plot test results from the first few runs to verify the setup is working.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    hydrographs : list
        List of monthly mean hydrograph data from test runs
    validation_start : str
        Start date for validation period
    validation_end : str
        End date for validation period
    color_best : str
        Color for the best simulation
    color_others : str
        Color for other simulations
    sim_results_dir : Path
        Directory where simulation results are stored
    """
    
    if len(hydrographs) == 0:
        print("No hydrographs to plot")
        return
    
    gauge_id = config['gauge_id']
    
    # Load observed data for comparison
    obs_data = load_hydrograph_data(config)
    if obs_data is None:
        print("Warning: Could not load observed hydrograph data for test plot")
        obs_mean = None
    else:
        mask = (obs_data['date'] >= validation_start) & (obs_data['date'] <= validation_end)
        obs_monthly = obs_data[mask].copy()
        obs_monthly['month'] = obs_monthly['date'].dt.month
        obs_mean = obs_monthly.groupby('month')['obs_Q'].mean()

    # Create test plot
    plt.figure(figsize=(12, 7))
    
    # Plot observed data if available
    if obs_mean is not None:
        plt.plot(obs_mean.index, obs_mean.values, 'k-', linewidth=2.5, label='Observed')
    
    # Plot test hydrographs
    for i, monthly_mean in enumerate(hydrographs):
        if i == 0:
            plt.plot(monthly_mean.index, monthly_mean.values, color=color_best, 
                    linewidth=2, label='Best Test Run')
        else:
            plt.plot(monthly_mean.index, monthly_mean.values, color=color_others, 
                    linewidth=1, alpha=0.7, label='Other Test Runs' if i == 1 else '')
    
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Discharge (m³/s)', fontsize=12)
    plt.title(f'Test Results - {len(hydrographs)} Successful Runs - Gauge {gauge_id}', fontsize=14)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save test plot
    test_plot_path = sim_results_dir / f'test_regime_{len(hydrographs)}_runs_{gauge_id}.png'
    plt.savefig(test_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved test plot to: {test_plot_path}")
    plt.show()
    
    print(f"Test plot created successfully with {len(hydrographs)} hydrographs")

#--------------------------------------------------------------------------------

def plot_regime_100_best_runs(config, plot_dirs, template_dir=None, raven_exe=None, n_runs=100,
                             validation_start=None, validation_end=None,
                             color_best='#24868E', color_others='grey'):
    """
    Plot regime for the best runs from SCEUA calibration using namelist configuration.
    Now includes streamflow, glacier melt, and snowmelt uncertainty analysis.
    Uses GloGEM data for glacier/snowmelt when coupled=True.
    """
    
    import subprocess
    import shutil
    
    gauge_id = config['gauge_id']
    model_type = config['model_type']
    coupled = config.get('coupled', False)  # Get coupled setting
    config_dir = Path(config['main_dir']) / config['config_dir']
    
    print(f"Running regime uncertainty analysis for {n_runs} best runs:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Model: {model_type}")
    print(f"  - Coupled mode: {coupled}")
    print(f"  - Validation period: {validation_start} to {validation_end}")
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    # Get paths from config if not provided
    if template_dir is None:
        template_dir = config_dir / f"catchment_{gauge_id}" / model_type / "templates"
    else:
        template_dir = Path(template_dir)
    
    # Get Raven executable from config
    if raven_exe is None:
        raven_exe = config.get('raven_executable', '/path/to/raven.exe')
        if raven_exe == '/path/to/raven.exe':
            print("Warning: Using default Raven executable path. Please check your namelist.")

    print(f"  - Template directory: {template_dir}")
    print(f"  - Raven executable: {raven_exe}")
    
    # Load catchment area for GloGEM normalization if coupled
    conversion_factor = None
    if coupled:
        topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
        catchment_shape_file = topo_dir / "HRU.shp"
        
        if catchment_shape_file.exists():
            try:
                import geopandas as gpd
                hru_gdf = gpd.read_file(catchment_shape_file)
                total_area_km2 = hru_gdf['Area_km2'].sum()
                # Convert km² to m² for GloGEM normalization
                conversion_factor = total_area_km2 * 1000000
                print(f"  - Catchment area for GloGEM: {total_area_km2:.2f} km²")
            except Exception as e:
                print(f"  - Warning: Could not load catchment area: {e}")
                conversion_factor = None
    
    # Load GloGEM data if coupled (using parsed file)
    glogem_glacier_data = None
    glogem_snowmelt_data = None
    
    if coupled and conversion_factor is not None:
        try:
            topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
            glogem_parsed_file = topo_dir / f"GloGEM_parsed_{gauge_id}.csv"
            
            if glogem_parsed_file.exists():
                print(f"  - Loading parsed GloGEM data from: {glogem_parsed_file}")
                
                # Load parsed GloGEM data (this has the correct column names)
                glogem_df = pd.read_csv(glogem_parsed_file, parse_dates=['date'])
                print(f"    Raw GloGEM data shape: {glogem_df.shape}")
                print(f"    Raw GloGEM data columns: {glogem_df.columns.tolist()}")
                print(f"    Raw GloGEM data date range: {glogem_df['date'].min()} to {glogem_df['date'].max()}")
                
                # Filter by validation period
                start_date = pd.to_datetime(validation_start)
                end_date = pd.to_datetime(validation_end)
                glogem_filtered = glogem_df[(glogem_df['date'] >= start_date) & 
                                           (glogem_df['date'] <= end_date)].copy()
                print(f"    Filtered GloGEM data shape: {glogem_filtered.shape}")
                
                if len(glogem_filtered) > 0:
                    # Process glacier melt data
                    if 'glacier_melt' in glogem_filtered.columns:
                        glogem_filtered['month'] = glogem_filtered['date'].dt.month
                        glogem_glacier_data = glogem_filtered.groupby('month')['glacier_melt'].mean()
                        print(f"    ✓ Loaded GloGEM glacier data: {len(glogem_filtered)} records")
                        print(f"    Monthly glacier melt values: {glogem_glacier_data.to_dict()}")
                    else:
                        print(f"    ✗ 'glacier_melt' column not found in parsed GloGEM file")
                        print(f"    Available columns: {glogem_filtered.columns.tolist()}")
                        glogem_glacier_data = None
                    
                    # Process snowmelt data
                    if 'snowmelt' in glogem_filtered.columns:
                        glogem_snowmelt_data = glogem_filtered.groupby('month')['snowmelt'].mean()
                        print(f"    ✓ Loaded GloGEM snowmelt data: {len(glogem_filtered)} records")
                        print(f"    Monthly snowmelt values: {glogem_snowmelt_data.to_dict()}")
                    else:
                        print(f"    ✗ 'snowmelt' column not found in parsed GloGEM file")
                        print(f"    Available columns: {glogem_filtered.columns.tolist()}")
                        glogem_snowmelt_data = None
                else:
                    print(f"    ✗ No GloGEM data found for validation period")
                    glogem_glacier_data = None
                    glogem_snowmelt_data = None
                
            else:
                print(f"  - Warning: Parsed GloGEM file not found: {glogem_parsed_file}")
                print(f"  - Please run load_glogem_data() first to create the parsed file")
                glogem_glacier_data = None
                glogem_snowmelt_data = None
                
        except Exception as e:
            print(f"  - Error loading parsed GloGEM data: {e}")
            import traceback
            traceback.print_exc()
            glogem_glacier_data = None
            glogem_snowmelt_data = None
    
    # MODIFY THE TEMPLATE FILE FIRST
    print(f"  - Modifying template file for glacier/snowmelt output...")
    rvi_template_file = template_dir / f"{gauge_id}_{model_type}.rvi.tpl"
    
    if rvi_template_file.exists():
        try:
            # Read the template file
            with open(rvi_template_file, 'r') as f:
                content = f.read()
            
            # Replace the output options
            old_output = "#Output Options\n  :EvaluationMetrics RMSE KLING_GUPTA NASH_SUTCLIFFE \n  :CustomOutput DAILY AVERAGE SNOW BY_HRU_GROUP"
            new_output = "#Output Options\n  :CustomOutput DAILY AVERAGE From:GLACIER_ICE BY_BASIN\n  :CustomOutput DAILY AVERAGE To:LAKE_STORAGE BY_BASIN"
            
            if old_output in content:
                content = content.replace(old_output, new_output)
                print(f"    ✓ Replaced evaluation metrics with glacier/snowmelt output")
            else:
                # Try alternative patterns
                import re
                # Pattern to match the output section
                pattern = r'#Output Options\s*:EvaluationMetrics[^\n]*\n\s*:CustomOutput[^\n]*'
                if re.search(pattern, content):
                    content = re.sub(pattern, new_output, content)
                    print(f"    ✓ Found and replaced output section using pattern matching")
                else:
                    print(f"    ⚠ Could not find output section to replace")
                    print(f"    Template content preview:")
                    print(content[-500:])  # Show last 500 chars where output usually is
            
            # Write the modified content back
            with open(rvi_template_file, 'w') as f:
                f.write(content)
            
            print(f"    ✓ Modified template file: {rvi_template_file}")
            
        except Exception as e:
            print(f"    ✗ Error modifying template file: {e}")
            return None
    else:
        print(f"    ✗ Template file not found: {rvi_template_file}")
        return None

    # 1. Read SCEUA results file
    output_dir = config_dir / f"catchment_{gauge_id}" / model_type / "output"
    results_file = output_dir / f"raven_sceua_{gauge_id}_{model_type}.csv"
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        # Try alternative file patterns
        alt_files = list(output_dir.glob(f"*sceua*.csv"))
        if alt_files:
            results_file = alt_files[0]
            print(f"Using alternative results file: {results_file}")
        else:
            print(f"No SCEUA results files found in {output_dir}")
            return None

    # READ THE CSV FILE - THIS WAS MISSING!
    try:
        df = pd.read_csv(results_file)
        print(f"Successfully loaded CSV with shape: {df.shape}")
        
    except pd.errors.ParserError as e:
        print(f"CSV parsing error: {e}")
        print(f"Attempting to fix parsing issues...")
        
        try:
            # Method 1: Use python engine with flexible separator detection
            df = pd.read_csv(results_file, sep=None, engine='python', on_bad_lines='skip')
            print(f"Successfully loaded with python engine: {df.shape}")
            
        except Exception as e2:
            try:
                # Method 2: Skip bad lines and use comma separator
                df = pd.read_csv(results_file, sep=',', on_bad_lines='skip')
                print(f"Successfully loaded skipping bad lines: {df.shape}")
                
            except Exception as e3:
                print(f"All CSV reading attempts failed:")
                print(f"  - Normal reading: {e}")
                print(f"  - Python engine: {e2}")
                print(f"  - Skip bad lines: {e3}")
                print(f"Please check the CSV file format manually: {results_file}")
                return None

    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    # Validate the loaded data
    if df is None or len(df) == 0:
        print("Error: Empty or invalid results file")
        return None

    print(f"Loaded {len(df)} parameter sets from results file")

    # 2. Convert negative KGE to positive KGE - WITH ERROR HANDLING
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample 'like1' values: {df['like1'].head().tolist()}")
    print(f"Data type of 'like1': {df['like1'].dtype}")

    # Convert like1 column to numeric, handling any non-numeric values
    try:
        # First, try to convert to numeric (this will handle strings that represent numbers)
        df['like1_numeric'] = pd.to_numeric(df['like1'], errors='coerce')
        
        # Check for any NaN values that resulted from conversion
        nan_count = df['like1_numeric'].isna().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} non-numeric values found in 'like1' column, dropping these rows")
            # Show some examples of problematic values
            problematic = df[df['like1_numeric'].isna()]['like1'].head()
            print(f"Examples of problematic values: {problematic.tolist()}")
            
            # Drop rows with NaN values
            df = df.dropna(subset=['like1_numeric'])
        
        # Remove failed runs (like1 = 999999 means failed run in SCEUA)
        print(f"Original dataset: {len(df)} parameter sets")
        valid_mask = (df['like1_numeric'] != 999999)
        df_valid = df[valid_mask].copy()
        
        print(f"After removing failed runs (999999): {len(df_valid)} parameter sets")
        
        if len(df_valid) == 0:
            print("ERROR: No valid parameter sets found after filtering!")
            print("All runs appear to have failed (like1 = 999999)")
            print("Check your SCEUA calibration - it may not have converged properly.")
            return None
        
        # Now convert negative KGE to positive KGE (SCEUA minimizes, so negative KGE becomes positive)
        df_valid['KGE'] = -df_valid['like1_numeric']
        
        print(f"Successfully processed {len(df_valid)} valid parameter sets")
        print(f"KGE range: {df_valid['KGE'].min():.4f} to {df_valid['KGE'].max():.4f}")
        
        # Use the cleaned dataframe for the rest of the function
        df = df_valid
        
    except Exception as e:
        print(f"Error converting 'like1' to numeric: {e}")
        return None

    # 3. Select best runs
    best_runs = df.sort_values('KGE', ascending=False).head(n_runs)
    param_cols = [col for col in df.columns if col not in ['like1', 'like1_numeric', 'KGE']]
    
    print(f"Parameter columns: {param_cols}")
    print(f"Best KGE range: {best_runs['KGE'].min():.4f} to {best_runs['KGE'].max():.4f}")

    # 4. Check template directory
    if not template_dir.exists():
        print(f"Error: Template directory not found: {template_dir}")
        return None
        
    template_files = list(template_dir.glob("*.tpl"))
    if not template_files:
        print(f"No .tpl files found in template directory.")
        all_files = list(template_dir.glob("*"))
        print(f"Available files: {[f.name for f in all_files]}")
        return None
    else:
        print(f"Found template files: {[f.name for f in template_files]}")

    # 5. Prepare output folder for simulations
    sim_results_dir = output_dir / f"best_{n_runs}_simulations_{gauge_id}"
    sim_results_dir.mkdir(exist_ok=True)

    # 6. Process all runs and collect streamflow, glacier melt, and snowmelt data
    print(f"\nProcessing all {n_runs} runs...")
    hydrographs = []
    glacier_melts = []
    snowmelts = []
    successful_runs = 0
    failed_runs = 0
    
    for i, (idx, row) in enumerate(best_runs.iterrows()):
        if i % 10 == 0:  # Progress indicator
            print(f"Processing run {i+1}/{n_runs}...")
            
        run_dir = sim_results_dir / f"run_{idx}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        
        try:
            # Set up run directory
            shutil.copytree(template_dir, run_dir)
            
            if not fill_templates_with_parameters(run_dir, param_cols, row[param_cols]):
                failed_runs += 1
                continue
                
            if not setup_raven_run_directory(run_dir, config):
                failed_runs += 1
                continue
            
            # Run Raven
            model_file = run_dir / f"{gauge_id}_{model_type}"
            run_output_dir = run_dir / "output"
            run_output_dir.mkdir(exist_ok=True)
            
            cmd = [str(raven_exe), str(model_file), "-o", str(run_output_dir)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Define file paths
                hydro_file = run_output_dir / f"{gauge_id}_{model_type}_Hydrographs.csv"
                
                # For glacier melt: use GloGEM if coupled, otherwise HBV output
                if coupled and glogem_glacier_data is not None:
                    # Use pre-loaded GloGEM glacier data (same for all runs)
                    monthly_mean_glacier = glogem_glacier_data
                    glacier_file_exists = True
                else:
                    # Use HBV glacier file
                    glacier_file = run_output_dir / f"{gauge_id}_{model_type}_FROM_GLACIER_ICE_Daily_Average_BySubbasin.csv"
                    glacier_file_exists = glacier_file.exists()
                
                # For snowmelt: combine GloGEM + HBV if coupled, otherwise just HBV
                snowmelt_file = run_output_dir / f"{gauge_id}_{model_type}_TO_LAKE_STORAGE_Daily_Average_BySubbasin.csv"
                
                # Check if required files exist
                if hydro_file.exists() and glacier_file_exists and snowmelt_file.exists():
                    
                    # Process hydrograph data (unchanged)
                    try:
                        df_hydro = pd.read_csv(hydro_file)
                        df_hydro['date'] = pd.to_datetime(df_hydro['date'])
                        mask = (df_hydro['date'] >= validation_start) & (df_hydro['date'] <= validation_end)
                        monthly_hydro = df_hydro[mask].copy()
                        
                        if len(monthly_hydro) > 0:
                            monthly_hydro['month'] = monthly_hydro['date'].dt.month
                            
                            sim_col = None
                            for col in df_hydro.columns:
                                if '[m3/s]' in col and 'observed' not in col.lower():
                                    sim_col = col
                                    break
                            
                            if sim_col:
                                monthly_mean_hydro = monthly_hydro.groupby('month')[sim_col].mean()
                                hydrographs.append(monthly_mean_hydro)
                            else:
                                print(f"    Warning: No simulation column found in hydrograph file for run {idx}")
                                failed_runs += 1
                                continue
                        else:
                            failed_runs += 1
                            continue
                            
                    except Exception as e:
                        print(f"    Error processing hydrograph data for run {idx}: {e}")
                        failed_runs += 1
                        continue
                    
                    # Process glacier melt data
                    try:
                        if coupled and glogem_glacier_data is not None:
                            # Use pre-loaded GloGEM data (same for all runs)
                            monthly_mean_glacier = glogem_glacier_data
                            glacier_melts.append(monthly_mean_glacier)
                            print(f"    Using GloGEM glacier data for run {idx}")
                        else:
                            # Use HBV glacier data (existing code)
                            df_glacier = pd.read_csv(glacier_file, skiprows=1)
                            
                            # Handle date column
                            if 'day' in df_glacier.columns:
                                df_glacier['date'] = pd.to_datetime(df_glacier['day'])
                            elif 'date' in df_glacier.columns:
                                df_glacier['date'] = pd.to_datetime(df_glacier['date'])
                            else:
                                print(f"    Warning: No date column found in glacier file for run {idx}")
                                failed_runs += 1
                                continue
                            
                            # Filter for validation period and calculate rates
                            glacier_mask = (df_glacier['date'] >= validation_start) & (df_glacier['date'] <= validation_end)
                            monthly_glacier = df_glacier[glacier_mask].copy()
                            
                            if len(monthly_glacier) > 0:
                                monthly_glacier['month'] = monthly_glacier['date'].dt.month
                                
                                if 'mean' in df_glacier.columns:
                                    monthly_glacier['glacier_melt_rate'] = monthly_glacier['mean'].diff().fillna(0)
                                    monthly_glacier['glacier_melt_rate'] = monthly_glacier['glacier_melt_rate'].clip(lower=0)
                                    
                                    monthly_mean_glacier = monthly_glacier.groupby('month')['glacier_melt_rate'].mean()
                                    glacier_melts.append(monthly_mean_glacier)
                                else:
                                    print(f"    Warning: No 'mean' column found in glacier file for run {idx}")
                                    failed_runs += 1
                                    continue
                            else:
                                failed_runs += 1
                                continue
                        
                    except Exception as e:
                        print(f"    Error processing glacier data for run {idx}: {e}")
                        failed_runs += 1
                        continue
                    
                    # Process snowmelt data
                    try:
                        # Always load HBV snowmelt
                        df_snowmelt = pd.read_csv(snowmelt_file, skiprows=1)
                        
                        # Handle date column
                        if 'day' in df_snowmelt.columns:
                            df_snowmelt['date'] = pd.to_datetime(df_snowmelt['day'])
                        elif 'date' in df_snowmelt.columns:
                            df_snowmelt['date'] = pd.to_datetime(df_snowmelt['date'])
                        else:
                            print(f"    Warning: No date column found in snowmelt file for run {idx}")
                            failed_runs += 1
                            continue
                        
                        # Filter for validation period
                        snowmelt_mask = (df_snowmelt['date'] >= validation_start) & (df_snowmelt['date'] <= validation_end)
                        monthly_snowmelt = df_snowmelt[snowmelt_mask].copy()
                        
                        if len(monthly_snowmelt) > 0:
                            monthly_snowmelt['month'] = monthly_snowmelt['date'].dt.month
                            
                            # Calculate daily snowmelt rates from cumulative values
                            if 'mean' in df_snowmelt.columns:
                                monthly_snowmelt['snowmelt_rate'] = monthly_snowmelt['mean'].diff().fillna(0)
                                monthly_snowmelt['snowmelt_rate'] = monthly_snowmelt['snowmelt_rate'].clip(lower=0)
                                
                                # Calculate monthly mean HBV snowmelt
                                hbv_snowmelt = monthly_snowmelt.groupby('month')['snowmelt_rate'].mean()
                                
                                # If coupled, add GloGEM snowmelt
                                if coupled and glogem_snowmelt_data is not None:
                                    # Combine HBV + GloGEM snowmelt
                                    combined_snowmelt = hbv_snowmelt.add(glogem_snowmelt_data, fill_value=0)
                                    snowmelts.append(combined_snowmelt)
                                    print(f"    Combined HBV + GloGEM snowmelt for run {idx}")
                                else:
                                    # Use only HBV snowmelt
                                    snowmelts.append(hbv_snowmelt)
                                
                            else:
                                print(f"    Warning: No 'mean' column found in snowmelt file for run {idx}")
                                failed_runs += 1
                                continue
                        else:
                            failed_runs += 1
                            continue
                            
                    except Exception as e:
                        print(f"    Error processing snowmelt data for run {idx}: {e}")
                        failed_runs += 1
                        continue
                    
                    # If we got here, all three datasets were processed successfully
                    successful_runs += 1
                    
                    # Clean up run directory
                    cleanup_raven_run_directory(run_dir)
                        
                else:
                    if not hydro_file.exists():
                        print(f"    Warning: Hydrograph file not found for run {idx}")
                    if not glacier_file_exists:
                        print(f"    Warning: Glacier file not found for run {idx}")
                    if not snowmelt_file.exists():
                        print(f"    Warning: Snowmelt file not found for run {idx}")
                    failed_runs += 1
            else:
                print(f"    Warning: Raven execution failed for run {idx}")
                print(f"    Error: {result.stderr}")
                failed_runs += 1
                        
        except Exception as e:
            print(f"    Error in run {idx}: {e}")
            failed_runs += 1
            continue

    print(f"Successfully processed {successful_runs} out of {n_runs} runs ({failed_runs} failed)")
    
    if len(hydrographs) == 0 or len(glacier_melts) == 0 or len(snowmelts) == 0:
        print("ERROR: Not all datasets were successfully generated. Cannot create plots.")
        print(f"  Hydrographs: {len(hydrographs)}")
        print(f"  Glacier melt: {len(glacier_melts)} ({'GloGEM' if coupled else 'HBV'})")
        print(f"  Snowmelt: {len(snowmelts)} ({'HBV+GloGEM' if coupled else 'HBV'})")
        return None

    # 7. Load observed data for hydrograph comparison
    obs_data = load_hydrograph_data(config)
    if obs_data is None:
        print("Warning: Could not load observed hydrograph data")
        obs_mean = None
    else:
        mask = (obs_data['date'] >= validation_start) & (obs_data['date'] <= validation_end)
        obs_monthly = obs_data[mask].copy()
        obs_monthly['month'] = obs_monthly['date'].dt.month
        obs_mean = obs_monthly.groupby('month')['obs_Q'].mean()

    # 8. Create three separate plots
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # PLOT 1: Streamflow uncertainty
    plt.figure(figsize=(14, 8))
    
    # Plot observed data first (if available)
    if obs_mean is not None:
        plt.plot(obs_mean.index, obs_mean.values, 'k-', linewidth=3, label='Observed', zorder=4)
    
    # Plot all simulations except the best one (in grey)
    for i, monthly_mean in enumerate(hydrographs[1:], 1):  # Skip first (best) simulation
        plt.plot(monthly_mean.index, monthly_mean.values, color=color_others, 
                linewidth=1, alpha=0.5, zorder=1)
    
    # Plot the best simulation on top
    if len(hydrographs) > 0:
        plt.plot(hydrographs[0].index, hydrographs[0].values, color=color_best, 
                linewidth=3, label='Best Simulation', zorder=3)
    
    # Add grey line to legend
    if len(hydrographs) > 1:
        plt.plot([], [], color=color_others, linewidth=1, alpha=0.5, 
                label=f'Other {len(hydrographs)-1} Simulations')
    
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Discharge (m³/s)', fontsize=14)
    plt.title(f'Streamflow Regime Uncertainty - {len(hydrographs)} Best Runs\nCatchment {gauge_id}', 
             fontsize=16, fontweight='bold')
    plt.xticks(months, month_names)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.tight_layout()
    
    # Save streamflow plot
    save_path_hydro = plot_dirs['hydrographs'] / f'streamflow_uncertainty_{len(hydrographs)}_best_runs_{gauge_id}.png'
    plt.savefig(save_path_hydro, dpi=300, bbox_inches='tight')
    print(f"Saved streamflow uncertainty plot to: {save_path_hydro}")
    plt.show()
    
    # PLOT 2: Glacier melt uncertainty
    plt.figure(figsize=(14, 8))
    
    # Plot all glacier melt simulations except the best one (in grey)
    for i, monthly_glacier in enumerate(glacier_melts[1:], 1):  # Skip first (best) simulation
        plt.plot(monthly_glacier.index, monthly_glacier.values, color=color_others, 
                linewidth=1, alpha=0.5, zorder=1)
    
    # Plot the best glacier melt simulation on top
    if len(glacier_melts) > 0:
        plt.plot(glacier_melts[0].index, glacier_melts[0].values, color='red', 
                linewidth=3, label='Best Glacier Melt', zorder=3)
    
    # Add grey line to legend
    if len(glacier_melts) > 1:
        plt.plot([], [], color=color_others, linewidth=1, alpha=0.5, 
                label=f'Other {len(glacier_melts)-1} Simulations')
    
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Glacier Melt Rate (mm/day)', fontsize=14)
    plt.title(f'Glacier Melt Uncertainty - {len(glacier_melts)} Best Runs\nCatchment {gauge_id} ({'GloGEM' if coupled else 'HBV'})', 
             fontsize=16, fontweight='bold')
    plt.xticks(months, month_names)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.tight_layout()
    
    # Save glacier melt plot
    save_path_glacier = plot_dirs['contributions'] / f'glacier_melt_uncertainty_{len(glacier_melts)}_best_runs_{gauge_id}.png'
    plt.savefig(save_path_glacier, dpi=300, bbox_inches='tight')
    print(f"Saved glacier melt uncertainty plot to: {save_path_glacier}")
    plt.show()
    
    # PLOT 3: Snowmelt uncertainty
    plt.figure(figsize=(14, 8))
    
    # Plot all snowmelt simulations except the best one (in grey)
    for i, monthly_snowmelt in enumerate(snowmelts[1:], 1):  # Skip first (best) simulation
        plt.plot(monthly_snowmelt.index, monthly_snowmelt.values, color=color_others, 
                linewidth=1, alpha=0.5, zorder=1)
    
    # Plot the best snowmelt simulation on top
    if len(snowmelts) > 0:
        plt.plot(snowmelts[0].index, snowmelts[0].values, color='deepskyblue', 
                linewidth=3, label='Best Snowmelt', zorder=3)
    
    # Add grey line to legend
    if len(snowmelts) > 1:
        plt.plot([], [], color=color_others, linewidth=1, alpha=0.5, 
                label=f'Other {len(snowmelts)-1} Simulations')
    
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Snowmelt Rate (mm/day)', fontsize=14)
    plt.title(f'Snowmelt Uncertainty - {len(snowmelts)} Best Runs\nCatchment {gauge_id} ({'HBV+GloGEM' if coupled else 'HBV'})', 
             fontsize=16, fontweight='bold')
    plt.xticks(months, month_names)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.tight_layout()
    
    # Save snowmelt plot
    save_path_snowmelt = plot_dirs['contributions'] / f'snowmelt_uncertainty_{len(snowmelts)}_best_runs_{gauge_id}.png'
    plt.savefig(save_path_snowmelt, dpi=300, bbox_inches='tight')
    print(f"Saved snowmelt uncertainty plot to: {save_path_snowmelt}")
    plt.show()
    
    # Print comprehensive summary
    print(f"\nUncertainty Analysis Summary for Catchment {gauge_id}:")
    print(f"  Successfully processed: {successful_runs}/{n_runs} runs")
    print(f"  Mode: {'Coupled (GloGEM+HBV)' if coupled else 'Uncoupled (HBV only)'}")
    print(f"  Validation period: {validation_start} to {validation_end}")
    print(f"  Best KGE: {best_runs['KGE'].iloc[0]:.4f}")
    print(f"  Worst KGE in selection: {best_runs['KGE'].iloc[-1]:.4f}")
    
    # Calculate uncertainty statistics for all three components
    if len(glacier_melts) > 1:
        glacier_array = np.array([gm.values for gm in glacier_melts])
        glacier_mean = np.mean(glacier_array, axis=0)
        glacier_std = np.std(glacier_array, axis=0)
        
        print(f"\nGlacier Melt Uncertainty Statistics:")
        for month, mean_val, std_val in zip(month_names, glacier_mean, glacier_std):
            cv = (std_val/mean_val*100) if mean_val > 0 else 0
            print(f"  {month}: {mean_val:.3f} ± {std_val:.3f} mm/day (CV: {cv:.1f}%)")
    
    if len(snowmelts) > 1:
        snowmelt_array = np.array([sm.values for sm in snowmelts])
        snowmelt_mean = np.mean(snowmelt_array, axis=0)
        snowmelt_std = np.std(snowmelt_array, axis=0)
        
        print(f"\nSnowmelt Uncertainty Statistics:")
        for month, mean_val, std_val in zip(month_names, snowmelt_mean, snowmelt_std):
            cv = (std_val/mean_val*100) if mean_val > 0 else 0
            print(f"  {month}: {mean_val:.3f} ± {std_val:.3f} mm/day (CV: {cv:.1f}%)")
    
    return {
        'hydrographs': hydrographs,
        'glacier_melts': glacier_melts,
        'snowmelts': snowmelts,
        'best_kge': best_runs['KGE'].iloc[0],
        'successful_runs': successful_runs,
        'save_paths': {
            'streamflow': save_path_hydro,
            'glacier_melt': save_path_glacier,
            'snowmelt': save_path_snowmelt
        }
    }

#--------------------------------------------------------------------------------

def plot_combined_uncertainty_with_envelope(config, plot_dirs, template_dir=None, raven_exe=None, n_runs=100,
                                          validation_start=None, validation_end=None):
    """
    Create three combined uncertainty plots with presentation-style formatting:
    1. Best run only (clean for presentations)
    2. Spaghetti plot with all runs as individual lines
    3. Best run with full uncertainty envelope (min-max range)
    
    Also calculates and displays contribution statistics.
    """
    
    # First run the main uncertainty analysis to get the data
    uncertainty_results = plot_regime_100_best_runs(
        config, plot_dirs, template_dir, raven_exe, n_runs,
        validation_start, validation_end
    )
    
    if uncertainty_results is None:
        print("ERROR: Could not generate uncertainty data")
        return None
    
    hydrographs = uncertainty_results['hydrographs']
    glacier_melts = uncertainty_results['glacier_melts']
    snowmelts = uncertainty_results['snowmelts']
    gauge_id = config['gauge_id']
    coupled = config.get('coupled', False)
    
    # Load catchment area for unit conversion
    config_dir = Path(config['main_dir']) / config['config_dir']
    topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
    catchment_shape_file = topo_dir / "HRU.shp"
    
    conversion_factor = None
    if catchment_shape_file.exists():
        try:
            import geopandas as gpd
            hru_gdf = gpd.read_file(catchment_shape_file)
            total_area_km2 = hru_gdf['Area_km2'].sum()
            conversion_factor = total_area_km2 * 1000000 / 1000 / 86400
            print(f"  - Catchment area: {total_area_km2:.2f} km²")
        except Exception as e:
            print(f"  - Warning: Could not load catchment area: {e}")
            conversion_factor = None
    
    # Convert glacier melt and snowmelt from mm/day to m³/s if possible
    if conversion_factor is not None:
        glacier_melts_converted = []
        for glacier_melt in glacier_melts:
            glacier_melt_converted = glacier_melt * conversion_factor
            glacier_melts_converted.append(glacier_melt_converted)
        glacier_melts = glacier_melts_converted
        
        snowmelts_converted = []
        for snowmelt in snowmelts:
            snowmelt_converted = snowmelt * conversion_factor
            snowmelts_converted.append(snowmelt_converted)
        snowmelts = snowmelts_converted
        
        print(f"  - Successfully converted glacier melt and snowmelt to m³/s")
    
    # Load observed data for streamflow comparison
    obs_data = load_hydrograph_data(config)
    obs_mean = None
    if obs_data is not None:
        if validation_start is None:
            validation_start = config.get('cali_end_date', '2010-01-01')
        if validation_end is None:
            validation_end = config.get('end_date', '2020-12-31')
        
        mask = (obs_data['date'] >= validation_start) & (obs_data['date'] <= validation_end)
        obs_monthly = obs_data[mask].copy()
        obs_monthly['month'] = obs_monthly['date'].dt.month
        obs_mean = obs_monthly.groupby('month')['obs_Q'].mean()
    
    months = range(1, 13)
    month_names_short = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    
    # Set colors based on coupled setting
    sim_color = '#82b182' if coupled else '#976c03'
    
    # Calculate min-max envelope and best run
    def calculate_envelope(data_list):
        if len(data_list) < 2:
            return None, None, None
        
        data_array = np.array([sim.reindex(months, fill_value=0).values for sim in data_list])
        
        min_vals = np.min(data_array, axis=0)
        max_vals = np.max(data_array, axis=0)
        best_vals = data_array[0]
        
        return min_vals, max_vals, best_vals
    
    # Calculate envelopes for each component
    hydro_min, hydro_max, hydro_best = calculate_envelope(hydrographs)
    glacier_min, glacier_max, glacier_best = calculate_envelope(glacier_melts)
    snow_min, snow_max, snow_best = calculate_envelope(snowmelts)
    
    # =================================================================
    # CALCULATE CONTRIBUTION STATISTICS
    # =================================================================
    
    print(f"\n{'='*60}")
    print(f"CALCULATING CONTRIBUTION STATISTICS")
    print(f"{'='*60}")
    
    contribution_data = []
    contribution_stats = None
    unit_label = 'm³/s' if conversion_factor is not None else 'mm/day'
    
    # Calculate yearly contribution percentages for all runs
    if len(hydrographs) > 0 and len(glacier_melts) > 0 and len(snowmelts) > 0:
        
        for i in range(len(hydrographs)):
            if i >= len(glacier_melts) or i >= len(snowmelts):
                continue
                
            # Get annual means for this run
            hydro_annual = hydrographs[i].mean()  # Mean monthly discharge
            glacier_annual = glacier_melts[i].mean()  # Mean monthly glacier melt
            snowmelt_annual = snowmelts[i].mean()  # Mean monthly snowmelt
            
            # Skip if any values are zero or invalid
            if hydro_annual <= 0 or not np.isfinite(hydro_annual):
                continue
            
            # Calculate contributions as percentages of streamflow
            glacier_contribution = (glacier_annual / hydro_annual) * 100
            snowmelt_contribution = (snowmelt_annual / hydro_annual) * 100
            total_melt_contribution = glacier_contribution + snowmelt_contribution
            
            # Store the data
            contribution_data.append({
                'run': i,
                'streamflow_annual': hydro_annual,
                'glacier_melt_annual': glacier_annual,
                'snowmelt_annual': snowmelt_annual,
                'glacier_contribution_pct': glacier_contribution,
                'snowmelt_contribution_pct': snowmelt_contribution,
                'total_melt_contribution_pct': total_melt_contribution
            })
        
        if len(contribution_data) > 1:
            # Convert to DataFrame for easier statistics
            contrib_df = pd.DataFrame(contribution_data)
            
            # Calculate statistics
            contribution_stats = {
                'glacier_contribution': {
                    'min': contrib_df['glacier_contribution_pct'].min(),
                    'max': contrib_df['glacier_contribution_pct'].max(),
                    'mean': contrib_df['glacier_contribution_pct'].mean(),
                    'median': contrib_df['glacier_contribution_pct'].median(),
                    'std': contrib_df['glacier_contribution_pct'].std(),
                    'range': contrib_df['glacier_contribution_pct'].max() - contrib_df['glacier_contribution_pct'].min()
                },
                'snowmelt_contribution': {
                    'min': contrib_df['snowmelt_contribution_pct'].min(),
                    'max': contrib_df['snowmelt_contribution_pct'].max(),
                    'mean': contrib_df['snowmelt_contribution_pct'].mean(),
                    'median': contrib_df['snowmelt_contribution_pct'].median(),
                    'std': contrib_df['snowmelt_contribution_pct'].std(),
                    'range': contrib_df['snowmelt_contribution_pct'].max() - contrib_df['snowmelt_contribution_pct'].min()
                },
                'total_melt_contribution': {
                    'min': contrib_df['total_melt_contribution_pct'].min(),
                    'max': contrib_df['total_melt_contribution_pct'].max(),
                    'mean': contrib_df['total_melt_contribution_pct'].mean(),
                    'median': contrib_df['total_melt_contribution_pct'].median(),
                    'std': contrib_df['total_melt_contribution_pct'].std(),
                    'range': contrib_df['total_melt_contribution_pct'].max() - contrib_df['total_melt_contribution_pct'].min()
                }
            }
            
            print(f"Successfully calculated contribution statistics for {len(contribution_data)} runs")
        else:
            print("Not enough valid runs for contribution statistics")
            contribution_stats = None
    
    # ==============================================
    # PLOT 1: BEST RUNS ONLY - PRESENTATION STYLE
    # ==============================================
    
    plt.figure(figsize=(16, 12))
    
    # Plot observed streamflow first (if available)
    if obs_mean is not None:
        plt.plot(months, obs_mean.reindex(months).values, 'k-', linewidth=6, 
                label='Observed', zorder=10)
    
    # Plot the best runs
    if hydro_best is not None:
        plt.plot(months, hydro_best, color=sim_color, linewidth=5, 
                label='Simulated', zorder=8)
    
    if glacier_best is not None:
        plt.plot(months, glacier_best, color='darkgray', linewidth=5, 
                label='Glacier Melt', zorder=6)
    
    if snow_best is not None:
        plt.plot(months, snow_best, color='lightblue', linewidth=5, 
                label='Snowmelt', zorder=4)
    
    # Formatting for presentation
    plt.ylabel('Discharge (m³/s)', fontsize=32, fontweight='bold')
    plt.xticks(months, month_names_short, fontsize=32, fontweight='bold')
    plt.yticks(fontsize=32, fontweight='bold')
    plt.grid(True, alpha=0.3, zorder=0)
    plt.legend(loc='best', fontsize=28)
    
    plt.tight_layout()
    
    # Save best runs only plot
    save_path_best = plot_dirs['contributions'] / f'best_run_only_presentation_{gauge_id}.png'
    plt.savefig(save_path_best, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved best runs only plot to: {save_path_best}")
    plt.show()
    
    # ==============================================
    # PLOT 2: SPAGHETTI PLOT - ALL INDIVIDUAL RUNS
    # ==============================================
    
    plt.figure(figsize=(16, 12))
    
    # Plot all individual runs as background lines (spaghetti)
    # Streamflow - all runs except the best one
    for i, monthly_mean in enumerate(hydrographs[1:], 1):
        alpha_val = max(0.1, min(0.4, 20/len(hydrographs)))  # Adjust alpha based on number of runs
        plt.plot(months, monthly_mean.reindex(months, fill_value=0).values, 
                color=sim_color, linewidth=2, alpha=alpha_val, zorder=2)
    
    # Glacier melt - all runs except the best one
    for i, monthly_glacier in enumerate(glacier_melts[1:], 1):
        alpha_val = max(0.1, min(0.4, 20/len(glacier_melts)))
        plt.plot(months, monthly_glacier.reindex(months, fill_value=0).values, 
                color='darkgray', linewidth=2, alpha=alpha_val, zorder=2)
    
    # Snowmelt - all runs except the best one
    for i, monthly_snow in enumerate(snowmelts[1:], 1):
        alpha_val = max(0.1, min(0.4, 20/len(snowmelts)))
        plt.plot(months, monthly_snow.reindex(months, fill_value=0).values, 
                color='lightblue', linewidth=2, alpha=alpha_val, zorder=2)
    
    # Plot observed streamflow (if available)
    if obs_mean is not None:
        plt.plot(months, obs_mean.reindex(months).values, 'k-', linewidth=6, 
                label='Observed', zorder=10)
    
    # Plot best runs on top with thicker lines
    if hydro_best is not None:
        plt.plot(months, hydro_best, color=sim_color, linewidth=5, 
                label='Simulated', zorder=8)
    
    if glacier_best is not None:
        plt.plot(months, glacier_best, color='darkgray', linewidth=5, 
                label='Glacier Melt', zorder=6)
    
    if snow_best is not None:
        plt.plot(months, snow_best, color='lightblue', linewidth=5, 
                label='Snowmelt', zorder=4)
    
    # Add background lines to legend (invisible lines just for legend)
    #if len(hydrographs) > 1:
    #    plt.plot([], [], color='gray', linewidth=2, alpha=0.3, 
    #            label=f'{len(hydrographs)-1} Other Runs')
    
    # Formatting for presentation
    plt.ylabel('Discharge (m³/s)', fontsize=32, fontweight='bold')
    plt.xticks(months, month_names_short, fontsize=32, fontweight='bold')
    plt.yticks(fontsize=32, fontweight='bold')
    plt.grid(True, alpha=0.3, zorder=0)
    plt.legend(loc='best', fontsize=28)
    
    plt.tight_layout()
    
    # Save spaghetti plot
    save_path_spaghetti = plot_dirs['contributions'] / f'spaghetti_plot_presentation_{gauge_id}.png'
    plt.savefig(save_path_spaghetti, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved spaghetti plot to: {save_path_spaghetti}")
    plt.show()
    
    # ==============================================
    # PLOT 3: UNCERTAINTY ENVELOPE - PRESENTATION STYLE
    # ==============================================
    
    plt.figure(figsize=(16, 12))
    
    # Plot uncertainty envelopes (fill between min and max)
    if hydro_min is not None and hydro_max is not None:
        plt.fill_between(months, hydro_min, hydro_max, alpha=0.4, color=sim_color, 
                        zorder=1)
    
    if glacier_min is not None and glacier_max is not None:
        plt.fill_between(months, glacier_min, glacier_max, alpha=0.4, color='darkgray', 
                        zorder=1)
    
    if snow_min is not None and snow_max is not None:
        plt.fill_between(months, snow_min, snow_max, alpha=0.4, color='lightblue', 
                        zorder=1)
    
    # Plot observed streamflow (if available)
    if obs_mean is not None:
        plt.plot(months, obs_mean.reindex(months).values, 'k-', linewidth=6, 
                label='Observed', zorder=10)
    
    # Plot best runs on top
    if hydro_best is not None:
        plt.plot(months, hydro_best, color=sim_color, linewidth=5, 
                label='Simulated', zorder=8)
    
    if glacier_best is not None:
        plt.plot(months, glacier_best, color='darkgray', linewidth=5, 
                label='Glacier Melt', zorder=6)
    
    if snow_best is not None:
        plt.plot(months, snow_best, color='lightblue', linewidth=5, 
                label='Snowmelt', zorder=4)
    
    # Formatting for presentation
    plt.ylabel('Discharge (m³/s)', fontsize=32, fontweight='bold')
    plt.xticks(months, month_names_short, fontsize=32, fontweight='bold')
    plt.yticks(fontsize=32, fontweight='bold')
    plt.grid(True, alpha=0.3, zorder=0)
    plt.legend(loc='best', fontsize=28)
    
    plt.tight_layout()
    
    # Save uncertainty envelope plot
    save_path_envelope = plot_dirs['contributions'] / f'uncertainty_envelope_presentation_{gauge_id}.png'
    plt.savefig(save_path_envelope, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved uncertainty envelope plot to: {save_path_envelope}")
    plt.show()
    
    # =================================================================
    # PRINT CONTRIBUTION STATISTICS
    # =================================================================
    if contribution_stats:
        print(f"\n{'='*60}")
        print(f"YEARLY CONTRIBUTION STATISTICS - CATCHMENT {gauge_id}")
        print(f"{'='*60}")
        print(f"Number of runs analyzed: {len(contribution_data)}")
        print(f"Validation period: {validation_start} to {validation_end}")
        
        print(f"\nGLACIER MELT CONTRIBUTION TO STREAMFLOW:")
        print(f"  Range: {contribution_stats['glacier_contribution']['min']:.1f}% - {contribution_stats['glacier_contribution']['max']:.1f}%")
        print(f"  Mean: {contribution_stats['glacier_contribution']['mean']:.1f}% ± {contribution_stats['glacier_contribution']['std']:.1f}%")
        print(f"  Median: {contribution_stats['glacier_contribution']['median']:.1f}%")
        print(f"  Variation range: {contribution_stats['glacier_contribution']['range']:.1f} percentage points")
        
        print(f"\nSNOWMELT CONTRIBUTION TO STREAMFLOW:")
        print(f"  Range: {contribution_stats['snowmelt_contribution']['min']:.1f}% - {contribution_stats['snowmelt_contribution']['max']:.1f}%")
        print(f"  Mean: {contribution_stats['snowmelt_contribution']['mean']:.1f}% ± {contribution_stats['snowmelt_contribution']['std']:.1f}%")
        print(f"  Median: {contribution_stats['snowmelt_contribution']['median']:.1f}%")
        print(f"  Variation range: {contribution_stats['snowmelt_contribution']['range']:.1f} percentage points")
        
        print(f"\nTOTAL MELT CONTRIBUTION TO STREAMFLOW:")
        print(f"  Range: {contribution_stats['total_melt_contribution']['min']:.1f}% - {contribution_stats['total_melt_contribution']['max']:.1f}%")
        print(f"  Mean: {contribution_stats['total_melt_contribution']['mean']:.1f}% ± {contribution_stats['total_melt_contribution']['std']:.1f}%")
        print(f"  Median: {contribution_stats['total_melt_contribution']['median']:.1f}%")
        print(f"  Variation range: {contribution_stats['total_melt_contribution']['range']:.1f} percentage points")
    
    # Print summary statistics
    if hydro_min is not None and hydro_max is not None:
        print(f"\nUncertainty Range Summary for Catchment {gauge_id}:")
        print(f"  Number of simulations: {len(hydrographs)}")
        print(f"  Units: {unit_label}")
        if conversion_factor is not None:
            print(f"  Conversion factor: {conversion_factor:.6f}")
        print(f"  Streamflow uncertainty range:")
        for month, min_val, max_val, best_val in zip(month_names_short, hydro_min, hydro_max, hydro_best):
            range_val = max_val - min_val
            print(f"    {month}: {min_val:.2f} - {max_val:.2f} (range: {range_val:.2f}, best: {best_val:.2f})")
    
    return {
        'best_run_plot': save_path_best,
        'spaghetti_plot': save_path_spaghetti,
        'uncertainty_plot': save_path_envelope,
        'successful_runs': uncertainty_results['successful_runs'],
        'coupled': coupled,
        'sim_color': sim_color,
        'conversion_factor': conversion_factor,
        'unit_label': unit_label,
        'contribution_statistics': contribution_stats,
        'contribution_data': contribution_data if contribution_data else None,
        'uncertainty_ranges': {
            'streamflow': {'min': hydro_min, 'max': hydro_max, 'best': hydro_best},
            'glacier_melt': {'min': glacier_min, 'max': glacier_max, 'best': glacier_best},
            'snowmelt': {'min': snow_min, 'max': snow_max, 'best': snow_best}
        },
        'summary': {
            'glacier_range': f"{contribution_stats['glacier_contribution']['min']:.1f}% - {contribution_stats['glacier_contribution']['max']:.1f}%" if contribution_stats else "N/A",
            'snowmelt_range': f"{contribution_stats['snowmelt_contribution']['min']:.1f}% - {contribution_stats['snowmelt_contribution']['max']:.1f}%" if contribution_stats else "N/A",
            'total_melt_range': f"{contribution_stats['total_melt_contribution']['min']:.1f}% - {contribution_stats['total_melt_contribution']['max']:.1f}%" if contribution_stats else "N/A",
            'glacier_variation': f"{contribution_stats['glacier_contribution']['range']:.1f} percentage points" if contribution_stats else "N/A",
            'snowmelt_variation': f"{contribution_stats['snowmelt_contribution']['range']:.1f} percentage points" if contribution_stats else "N/A",
            'total_melt_variation': f"{contribution_stats['total_melt_contribution']['range']:.1f} percentage points" if contribution_stats else "N/A"
        }
    }

#--------------------------------------------------------------------------------

def plot_combined_uncertainty_boxplots(config, plot_dirs, template_dir=None, raven_exe=None, n_runs=100,
                                     validation_start=None, validation_end=None):
    """
    Create box plots showing uncertainty distributions for each component by month.
    """
    
    # Get uncertainty data
    uncertainty_results = plot_regime_100_best_runs(
        config, plot_dirs, template_dir, raven_exe, n_runs,
        validation_start, validation_end
    )
    
    if uncertainty_results is None:
        return None
    
    hydrographs = uncertainty_results['hydrographs']
    glacier_melts = uncertainty_results['glacier_melts']
    snowmelts = uncertainty_results['snowmelts']
    gauge_id = config['gauge_id']
    
    months = range(1, 13)
    month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    
    # Prepare data for box plots
    def prepare_boxplot_data(data_list):
        boxplot_data = []
        for month in months:
            month_values = [sim.get(month, 0) for sim in data_list if month in sim.index]
            boxplot_data.append(month_values)
        return boxplot_data
    
    hydro_boxdata = prepare_boxplot_data(hydrographs)
    glacier_boxdata = prepare_boxplot_data(glacier_melts)
    snow_boxdata = prepare_boxplot_data(snowmelts)
    
    # Create three subplot box plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Streamflow box plots
    bp1 = ax1.boxplot(hydro_boxdata, positions=months, widths=0.6, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('#24868E')
        patch.set_alpha(0.7)
    ax1.set_ylabel('Discharge (m³/s)', fontsize=12)
    ax1.set_title(f'Streamflow Uncertainty Distribution - Catchment {gauge_id}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Glacier melt box plots
    bp2 = ax2.boxplot(glacier_boxdata, positions=months, widths=0.6, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('red')
        patch.set_alpha(0.7)
    ax2.set_ylabel('Glacier Melt (mm/day)', fontsize=12)
    ax2.set_title('Glacier Melt Uncertainty Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Snowmelt box plots
    bp3 = ax3.boxplot(snow_boxdata, positions=months, widths=0.6, patch_artist=True)
    for patch in bp3['boxes']:
        patch.set_facecolor('deepskyblue')
        patch.set_alpha(0.7)
    ax3.set_ylabel('Snowmelt (mm/day)', fontsize=12)
    ax3.set_title('Snowmelt Uncertainty Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Month', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis
    ax3.set_xticks(months)
    ax3.set_xticklabels(month_names)
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'combined_uncertainty_boxplots_{len(hydrographs)}_best_runs_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined uncertainty boxplots to: {save_path}")
    plt.show()
    
    return fig


#--------------------------------------------------------------------------------
################################ water balance #################################
#--------------------------------------------------------------------------------


def plot_yearly_precipitation_streamflow(config, plot_dirs, validation_start=None, validation_end=None, min_data_fraction=0.8):
    """
    Plot yearly summed precipitation vs yearly summed observed and simulated streamflow.
    Excludes years with insufficient data (less than min_data_fraction of days).
    
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
    min_data_fraction : float, optional
        Minimum fraction of valid days required for a year to be included (default: 0.8 = 80%)
    """
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    gauge_id = config['gauge_id']

    print(f"Creating yearly precipitation vs streamflow plot for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    print(f"  - Minimum data requirement: {min_data_fraction*100:.0f}% of days per year")

    # Load hydrograph data
    df = load_hydrograph_data(config)
    if df is None:
        print("ERROR: Could not load hydrograph data")
        return None

    # Filter for validation period
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df = df[mask].copy()
    
    if len(df) == 0:
        print(f"ERROR: No data found for period {validation_start} to {validation_end}")
        return None

    # Check required columns
    if not all(col in df.columns for col in ['obs_Q', 'sim_Q', 'precip']):
        print("ERROR: Hydrograph file must contain 'obs_Q', 'sim_Q', and 'precip' columns")
        print(f"  Available columns: {df.columns.tolist()}")
        return None

    # Convert streamflow from m³/s to mm/day using catchment area
    config_dir = Path(config['main_dir']) / config['config_dir']
    topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
    catchment_shape_file = topo_dir / "HRU.shp"
    
    try:
        if catchment_shape_file.exists():
            hru_gdf = gpd.read_file(catchment_shape_file)
            total_area_km2 = hru_gdf['Area_km2'].sum()
            conversion_factor = 86400 / (total_area_km2 * 1000000) * 1000
            df['obs_Q_mm'] = df['obs_Q'] * conversion_factor
            df['sim_Q_mm'] = df['sim_Q'] * conversion_factor
            print(f"  - Catchment area: {total_area_km2:.2f} km²")
        else:
            print("WARNING: Could not load catchment area, keeping streamflow in m³/s")
            df['obs_Q_mm'] = df['obs_Q']
            df['sim_Q_mm'] = df['sim_Q']
            conversion_factor = None
    except Exception as e:
        print(f"WARNING: Error converting streamflow units: {e}")
        df['obs_Q_mm'] = df['obs_Q']
        df['sim_Q_mm'] = df['sim_Q']
        conversion_factor = None

    # Add year column
    df['year'] = df['date'].dt.year
    
    # ✅ FIX: Calculate data availability per year and filter incomplete years
    yearly_stats = df.groupby('year').agg({
        'obs_Q': 'count',  # Count non-null values
        'precip': ['count', 'sum'],  # Count and sum
        'obs_Q_mm': 'sum',
        'sim_Q_mm': 'sum'
    })
    
    # Flatten column names
    yearly_stats.columns = ['obs_count', 'precip_count', 'precip_sum_mm', 'obs_streamflow_mm', 'sim_streamflow_mm']
    yearly_stats = yearly_stats.reset_index()
    
    # Calculate expected number of days per year
    yearly_stats['days_in_year'] = yearly_stats['year'].apply(
        lambda y: 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
    )
    
    # Calculate data fraction
    yearly_stats['obs_data_fraction'] = yearly_stats['obs_count'] / yearly_stats['days_in_year']
    yearly_stats['precip_data_fraction'] = yearly_stats['precip_count'] / yearly_stats['days_in_year']
    
    # ✅ FIX: Filter years with insufficient data
    print(f"\n  Data availability by year:")
    for _, row in yearly_stats.iterrows():
        print(f"    {int(row['year'])}: Obs={row['obs_data_fraction']*100:.1f}%, Precip={row['precip_data_fraction']*100:.1f}% ({int(row['obs_count'])}/{int(row['days_in_year'])} days)")
    
    # Keep only years with sufficient data for BOTH obs and precip
    valid_years = yearly_stats[
        (yearly_stats['obs_data_fraction'] >= min_data_fraction) & 
        (yearly_stats['precip_data_fraction'] >= min_data_fraction)
    ].copy()
    
    if len(valid_years) == 0:
        print(f"ERROR: No years with sufficient data (>{min_data_fraction*100:.0f}% coverage)")
        return None
    
    # ✅ FIX: Exclude first and last year if they're incomplete
    first_year = valid_years['year'].min()
    last_year = valid_years['year'].max()
    
    # Check if first year starts at beginning of year
    first_year_data = df[df['year'] == first_year]
    if first_year_data['date'].min().month != 1 or first_year_data['date'].min().day != 1:
        print(f"  - Excluding {first_year} (incomplete - starts {first_year_data['date'].min().date()})")
        valid_years = valid_years[valid_years['year'] != first_year]
    
    # Check if last year ends at end of year
    last_year_data = df[df['year'] == last_year]
    if last_year_data['date'].max().month != 12 or last_year_data['date'].max().day != 31:
        print(f"  - Excluding {last_year} (incomplete - ends {last_year_data['date'].max().date()})")
        valid_years = valid_years[valid_years['year'] != last_year]
    
    if len(valid_years) == 0:
        print(f"ERROR: No complete years remaining after filtering")
        return None
    
    # Rename columns for clarity
    yearly = valid_years.rename(columns={
        'precip_sum_mm': 'precip_mm',
        'obs_streamflow_mm': 'streamflow_mm',
        'sim_streamflow_mm': 'sim_streamflow_mm'
    })[['year', 'precip_mm', 'streamflow_mm', 'sim_streamflow_mm']].copy()

    print(f"\n  ✓ Found {len(yearly)} complete years with sufficient data")
    print(f"    Year range: {int(yearly['year'].min())} - {int(yearly['year'].max())}")

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Precipitation vs Observed Streamflow
    ax1.scatter(yearly['precip_mm'], yearly['streamflow_mm'], color='darkblue', s=80, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Calculate 1:1 line limits
    min_val = min(yearly['precip_mm'].min(), yearly['streamflow_mm'].min()) * 0.95
    max_val = max(yearly['precip_mm'].max(), yearly['streamflow_mm'].max()) * 1.05
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='1:1 line')
    
    # Add regression line
    z = np.polyfit(yearly['precip_mm'], yearly['streamflow_mm'], 1)
    p = np.poly1d(z)
    ax1.plot(yearly['precip_mm'], p(yearly['precip_mm']), 'r-', alpha=0.8, linewidth=2, label='Linear fit')
    
    # Calculate correlation
    corr_obs = np.corrcoef(yearly['precip_mm'], yearly['streamflow_mm'])[0, 1]
    
    # Add year labels
    for _, row in yearly.iterrows():
        ax1.annotate(str(int(row['year'])), (row['precip_mm'], row['streamflow_mm']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    ax1.set_xlabel('Annual Precipitation (mm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Annual Observed Streamflow (mm)' if conversion_factor else 'Annual Observed Streamflow (m³/s)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Precipitation vs Observed Streamflow\nR = {corr_obs:.3f}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right plot: Precipitation vs Simulated Streamflow
    ax2.scatter(yearly['precip_mm'], yearly['sim_streamflow_mm'], color='orange', s=80, alpha=0.7, edgecolors='black', linewidth=1)
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='1:1 line')
    
    # Add regression line
    z = np.polyfit(yearly['precip_mm'], yearly['sim_streamflow_mm'], 1)
    p = np.poly1d(z)
    ax2.plot(yearly['precip_mm'], p(yearly['precip_mm']), 'r-', alpha=0.8, linewidth=2, label='Linear fit')
    
    # Calculate correlation
    corr_sim = np.corrcoef(yearly['precip_mm'], yearly['sim_streamflow_mm'])[0, 1]
    
    # Add year labels
    for _, row in yearly.iterrows():
        ax2.annotate(str(int(row['year'])), (row['precip_mm'], row['sim_streamflow_mm']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    ax2.set_xlabel('Annual Precipitation (mm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Annual Simulated Streamflow (mm)' if conversion_factor else 'Annual Simulated Streamflow (m³/s)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Precipitation vs Simulated Streamflow\nR = {corr_sim:.3f}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(f'Annual Precipitation vs Streamflow - Catchment {gauge_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save plot
    save_path = plot_dirs['contributions'] / f'yearly_precipitation_streamflow_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved yearly precipitation vs streamflow plot to: {save_path}")
    plt.show()

    # Print summary statistics
    print(f"\nYearly Precipitation vs Streamflow Analysis:")
    print(f"  Period: {int(yearly['year'].min())} - {int(yearly['year'].max())}")
    print(f"  Number of complete years: {len(yearly)}")
    print(f"  Mean annual precipitation: {yearly['precip_mm'].mean():.1f} mm")
    print(f"  Mean annual observed streamflow: {yearly['streamflow_mm'].mean():.1f} {'mm' if conversion_factor else 'm³/s'}")
    print(f"  Mean annual simulated streamflow: {yearly['sim_streamflow_mm'].mean():.1f} {'mm' if conversion_factor else 'm³/s'}")
    print(f"  Correlation (precip vs obs): {corr_obs:.3f}")
    print(f"  Correlation (precip vs sim): {corr_sim:.3f}")

    if conversion_factor:
        obs_ratio = yearly['streamflow_mm'].mean() / yearly['precip_mm'].mean()
        sim_ratio = yearly['sim_streamflow_mm'].mean() / yearly['precip_mm'].mean()
        print(f"  Observed runoff ratio: {obs_ratio:.3f}")
        print(f"  Simulated runoff ratio: {sim_ratio:.3f}")

    return yearly

    #--------------------------------------------------------------------------------

def plot_precipitation_glacier_melt_vs_streamflow_scatter(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Create a scatter plot comparing input vs observed streamflow for each year.
    - Coupled mode: Uses precipitation (which already includes glacier melt)
    - Uncoupled mode: Uses precipitation + separate glacier melt
    
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
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    gauge_id = config['gauge_id']
    coupled = config.get('coupled', False)
    
    print(f"Creating input vs streamflow scatter plot for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    print(f"  - Coupled mode: {coupled}")
    
    # Load streamflow data
    streamflow_data = load_hydrograph_data(config)
    if streamflow_data is None:
        print("ERROR: Could not load streamflow data")
        return None
    
    # Filter streamflow data for validation period
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    
    streamflow_mask = (streamflow_data['date'] >= start_date) & (streamflow_data['date'] <= end_date)
    streamflow_filtered = streamflow_data[streamflow_mask].copy()
    
    if len(streamflow_filtered) == 0:
        print(f"ERROR: No streamflow data found for period {validation_start} to {validation_end}")
        return None
    
    # Check required columns
    if not all(col in streamflow_filtered.columns for col in ['obs_Q', 'sim_Q', 'precip']):
        print("ERROR: Hydrograph file must contain 'obs_Q', 'sim_Q', and 'precip' columns")
        return None
    
    # Convert streamflow to mm/year
    try:
        config_dir = Path(config['main_dir']) / config['config_dir']
        topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
        catchment_shape_file = topo_dir / "HRU.shp"
        
        if catchment_shape_file.exists():
            import geopandas as gpd
            hru_gdf = gpd.read_file(catchment_shape_file)
            total_area_km2 = hru_gdf['Area_km2'].sum()
            conversion_factor = 86400 / (total_area_km2 * 1000000) * 1000
            streamflow_filtered['obs_Q_mm'] = streamflow_filtered['obs_Q'] * conversion_factor
            streamflow_filtered['sim_Q_mm'] = streamflow_filtered['sim_Q'] * conversion_factor
            print(f"  - Catchment area: {total_area_km2:.2f} km²")
        else:
            print("WARNING: Could not load catchment area")
            return None
    except Exception as e:
        print(f"ERROR: Error converting streamflow units: {e}")
        return None
    
    # Calculate yearly sums
    streamflow_filtered['year'] = streamflow_filtered['date'].dt.year
    
    # Yearly sums for streamflow and precipitation
    streamflow_yearly = streamflow_filtered.groupby('year').agg({
        'obs_Q_mm': 'sum',
        'sim_Q_mm': 'sum',
        'precip': 'sum'
    }).reset_index()
    
    # Calculate input based on coupled/uncoupled mode
    if coupled:
        # COUPLED MODE: Precipitation already includes glacier melt
        streamflow_yearly['input_total'] = streamflow_yearly['precip']
        input_label = 'Annual Precipitation (incl. Glacier Melt)'
        data_source_label = 'Coupled (GloGEM+HBV)'
        print(f"  - Using precipitation data (glacier melt already included)")
        
    else:
        # UNCOUPLED MODE: Need to add separate glacier melt
        print("Loading combined contributions data for uncoupled mode...")
        glacier_df, _ = create_combined_contributions_dataframes(
            config, plot_dirs, validation_start, validation_end
        )
        
        if glacier_df is None:
            print("ERROR: Could not load glacier contributions data for uncoupled mode")
            return None
        
        # Calculate glacier melt yearly sums
        glacier_df['year'] = glacier_df['date'].dt.year
        glacier_yearly = glacier_df.groupby('year')['glaciermelt'].sum().reset_index()
        glacier_yearly.columns = ['year', 'glacier_melt']
        
        # Merge with streamflow data
        streamflow_yearly = pd.merge(streamflow_yearly, glacier_yearly, on='year', how='inner')
        
        # Calculate total input
        streamflow_yearly['input_total'] = streamflow_yearly['precip'] + streamflow_yearly['glacier_melt']
        input_label = 'Annual Precipitation + Glacier Melt'
        data_source_label = 'Uncoupled (HBV only)'
        print(f"  - Added separate glacier melt to precipitation")
    
    if len(streamflow_yearly) == 0:
        print("ERROR: No overlapping years found between datasets")
        return None
    
    print(f"  - Found {len(streamflow_yearly)} years of complete data")
    print(f"  - Data source: {data_source_label}")
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    plt.scatter(streamflow_yearly['input_total'], streamflow_yearly['obs_Q_mm'], 
               color='darkblue', s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add 1:1 line
    min_val = min(streamflow_yearly['input_total'].min(), streamflow_yearly['obs_Q_mm'].min()) * 0.95
    max_val = max(streamflow_yearly['input_total'].max(), streamflow_yearly['obs_Q_mm'].max()) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2, label='1:1 line')
    
    # Add trend line
    z = np.polyfit(streamflow_yearly['input_total'], streamflow_yearly['obs_Q_mm'], 1)
    p = np.poly1d(z)
    plt.plot(streamflow_yearly['input_total'], p(streamflow_yearly['input_total']), 'r-', alpha=0.8, linewidth=2, label='Trend line')
    
    # Calculate correlation
    corr = np.corrcoef(streamflow_yearly['input_total'], streamflow_yearly['obs_Q_mm'])[0, 1]
    
    # Add year labels to points
    for _, row in streamflow_yearly.iterrows():
        plt.annotate(str(int(row['year'])), (row['input_total'], row['obs_Q_mm']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
    
    plt.xlabel(f'{input_label} (mm/year)', fontsize=14, fontweight='bold')
    plt.ylabel('Annual Observed Streamflow (mm/year)', fontsize=14, fontweight='bold')
    plt.title(f'Input vs Output Water Balance - Catchment {gauge_id}\n'
             f'R = {corr:.3f} | Period: {validation_start} to {validation_end} ({data_source_label})', 
             fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set equal aspect ratio and limits
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'input_vs_streamflow_scatter_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot to: {save_path}")
    plt.show()
    
    # Print summary statistics
    mean_input = streamflow_yearly['input_total'].mean()
    mean_output = streamflow_yearly['obs_Q_mm'].mean()
    mean_precip = streamflow_yearly['precip'].mean()
    runoff_ratio = mean_output / mean_input
    
    print(f"\nWater Balance Analysis Summary:")
    print(f"  Data source: {data_source_label}")
    print(f"  Mean annual precipitation: {mean_precip:.1f} mm/year")
    
    if not coupled and 'glacier_melt' in streamflow_yearly.columns:
        mean_glacier_melt = streamflow_yearly['glacier_melt'].mean()
        glacier_contribution_pct = (mean_glacier_melt / mean_input) * 100
        print(f"  Mean annual glacier melt: {mean_glacier_melt:.1f} mm/year")
        print(f"  Glacier contribution to input: {glacier_contribution_pct:.1f}%")
    else:
        print(f"  Glacier melt: Already included in precipitation")
    
    print(f"  Mean annual total input: {mean_input:.1f} mm/year")
    print(f"  Mean annual output (observed streamflow): {mean_output:.1f} mm/year")
    print(f"  Runoff ratio: {runoff_ratio:.3f}")
    print(f"  Correlation coefficient: {corr:.3f}")
    
    return streamflow_yearly

#--------------------------------------------------------------------------------

def plot_annual_water_balance_bars(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Create a bar plot showing annual water balance components for each year.
    
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
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    gauge_id = config['gauge_id']
    coupled = config.get('coupled', False)
    
    print(f"Creating annual water balance bar plot for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    print(f"  - Coupled mode: {coupled}")
    
    # Load streamflow data
    streamflow_data = load_hydrograph_data(config)
    if streamflow_data is None:
        print("ERROR: Could not load streamflow data")
        return None
    
    # Filter streamflow data for validation period
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    
    streamflow_mask = (streamflow_data['date'] >= start_date) & (streamflow_data['date'] <= end_date)
    streamflow_filtered = streamflow_data[streamflow_mask].copy()
    
    if len(streamflow_filtered) == 0:
        print(f"ERROR: No streamflow data found for period {validation_start} to {validation_end}")
        return None
    
    # Check required columns
    if not all(col in streamflow_filtered.columns for col in ['obs_Q', 'sim_Q', 'precip']):
        print("ERROR: Hydrograph file must contain 'obs_Q', 'sim_Q', and 'precip' columns")
        return None
    
    # Use the combined contributions function to get all contributions
    print("Loading combined contributions data...")
    glacier_df, nonglacier_df = create_combined_contributions_dataframes(
        config, plot_dirs, validation_start, validation_end
    )
    
    if glacier_df is None or nonglacier_df is None:
        print("ERROR: Could not load combined contributions data")
        return None
    
    # Convert streamflow to mm/year
    try:
        config_dir = Path(config['main_dir']) / config['config_dir']
        topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
        catchment_shape_file = topo_dir / "HRU.shp"
        
        if catchment_shape_file.exists():
            import geopandas as gpd
            hru_gdf = gpd.read_file(catchment_shape_file)
            total_area_km2 = hru_gdf['Area_km2'].sum()
            conversion_factor = 86400 / (total_area_km2 * 1000000) * 1000
            streamflow_filtered['obs_Q_mm'] = streamflow_filtered['obs_Q'] * conversion_factor
            streamflow_filtered['sim_Q_mm'] = streamflow_filtered['sim_Q'] * conversion_factor
            print(f"  - Catchment area: {total_area_km2:.2f} km²")
        else:
            print("WARNING: Could not load catchment area")
            return None
    except Exception as e:
        print(f"ERROR: Error converting streamflow units: {e}")
        return None
    
    # Calculate yearly sums
    streamflow_filtered['year'] = streamflow_filtered['date'].dt.year
    glacier_df['year'] = glacier_df['date'].dt.year
    nonglacier_df['year'] = nonglacier_df['date'].dt.year
    
    # Yearly sums for streamflow and precipitation
    streamflow_yearly = streamflow_filtered.groupby('year').agg({
        'obs_Q_mm': 'sum',
        'sim_Q_mm': 'sum',
        'precip': 'sum'
    }).reset_index()
    
    # Yearly sums for glacier contributions (all normalized to catchment scale)
    glacier_contributions = {}
    for component in ['rainfall', 'snowmelt', 'glaciermelt']:
        if component in glacier_df.columns:
            glacier_yearly = glacier_df.groupby('year')[component].sum().reset_index()
            glacier_yearly.columns = ['year', f'glacier_{component}']
            glacier_contributions[component] = glacier_yearly
    
    # Handle snowfall if present (for non-coupled runs)
    if 'snowfall' in glacier_df.columns:
        glacier_snowfall_yearly = glacier_df.groupby('year')['snowfall'].sum().reset_index()
        glacier_snowfall_yearly.columns = ['year', 'glacier_snowfall']
        glacier_contributions['snowfall'] = glacier_snowfall_yearly
    
    # Yearly sums for non-glacier contributions (all normalized to catchment scale)
    nonglacier_contributions = {}
    for component in ['rainfall', 'snowfall', 'snowmelt']:
        if component in nonglacier_df.columns:
            nonglacier_yearly = nonglacier_df.groupby('year')[component].sum().reset_index()
            nonglacier_yearly.columns = ['year', f'nonglacier_{component}']
            nonglacier_contributions[component] = nonglacier_yearly
    
    # Start with streamflow data
    yearly_data = streamflow_yearly.copy()
    
    # Merge all glacier contributions
    for component, contrib_df in glacier_contributions.items():
        yearly_data = pd.merge(yearly_data, contrib_df, on='year', how='inner')
    
    # Merge all non-glacier contributions
    for component, contrib_df in nonglacier_contributions.items():
        yearly_data = pd.merge(yearly_data, contrib_df, on='year', how='inner')
    
    if len(yearly_data) == 0:
        print("ERROR: No overlapping years found between datasets")
        return None
    
    # Calculate combined components
    # Total precipitation (glacier + non-glacier areas)
    total_rainfall = yearly_data.get('glacier_rainfall', 0) + yearly_data.get('nonglacier_rainfall', 0)
    
    # Handle snowfall (if present)
    total_snowfall = 0
    if 'glacier_snowfall' in yearly_data.columns:
        total_snowfall += yearly_data['glacier_snowfall']
    if 'nonglacier_snowfall' in yearly_data.columns:
        total_snowfall += yearly_data['nonglacier_snowfall']
    
    yearly_data['total_precipitation'] = total_rainfall + total_snowfall
    
    # Total snowmelt (glacier + non-glacier areas)
    total_snowmelt = yearly_data.get('glacier_snowmelt', 0) + yearly_data.get('nonglacier_snowmelt', 0)
    yearly_data['total_snowmelt'] = total_snowmelt
    
    # Glacier melt (only from glacier areas)
    yearly_data['total_glacier_melt'] = yearly_data.get('glacier_glaciermelt', 0)
    
    print(f"  - Found {len(yearly_data)} years of complete data")
    print(f"  - Available components:")
    print(f"    Total precipitation: {yearly_data['total_precipitation'].mean():.1f} mm/year")
    print(f"    Total snowmelt: {yearly_data['total_snowmelt'].mean():.1f} mm/year")
    print(f"    Glacier melt: {yearly_data['total_glacier_melt'].mean():.1f} mm/year")
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(max(12, len(yearly_data) * 1.0), 8))
    
    x = np.arange(len(yearly_data))
    width = 0.2  # Width of bars
    
    # Plot bars
    bars1 = ax.bar(x - 1.5*width, yearly_data['total_precipitation'], width, 
                   label='Total Precipitation', color='darkblue', alpha=0.8, 
                   edgecolor='black', linewidth=1)
    
    bars2 = ax.bar(x - 0.5*width, yearly_data['total_snowmelt'], width, 
                   label='Total Snowmelt', color='lightblue', alpha=0.8, 
                   edgecolor='black', linewidth=1)
    
    bars3 = ax.bar(x + 0.5*width, yearly_data['total_glacier_melt'], width, 
                   label='Glacier Melt', color='grey', alpha=0.8, 
                   edgecolor='black', linewidth=1)
    
    bars4 = ax.bar(x + 1.5*width, yearly_data['obs_Q_mm'], width, 
                   label='Observed Streamflow', color='black', alpha=0.8, 
                   edgecolor='white', linewidth=1)
    
    # Add value labels on bars (only show if not too crowded)
    if len(yearly_data) <= 12:  # Only add labels if not too many years
        def add_value_labels(bars, values):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if height > 0:  # Only add label if bar has height
                    ax.text(bar.get_x() + bar.get_width()/2., height + 
                           yearly_data[['total_precipitation', 'total_snowmelt', 'total_glacier_melt', 'obs_Q_mm']].max().max() * 0.01,
                           f'{value:.0f}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        add_value_labels(bars1, yearly_data['total_precipitation'])
        add_value_labels(bars2, yearly_data['total_snowmelt'])
        add_value_labels(bars3, yearly_data['total_glacier_melt'])
        add_value_labels(bars4, yearly_data['obs_Q_mm'])
    
    # Formatting
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Annual Sum (mm/year)', fontsize=14, fontweight='bold')
    ax.set_title(f'Annual Water Balance Components - Catchment {gauge_id}\n'
                f'Period: {validation_start} to {validation_end} ({"Coupled" if coupled else "Uncoupled"})', 
                fontsize=16, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(year)) for year in yearly_data['year']], rotation=45)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'annual_water_balance_bars_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved annual bar plot to: {save_path}")
    plt.show()
    
    # Print summary statistics
    yearly_data['input_total'] = yearly_data['total_precipitation'] + yearly_data['total_glacier_melt']
    yearly_data['runoff_ratio'] = yearly_data['obs_Q_mm'] / yearly_data['input_total']
    yearly_data['glacier_contribution_pct'] = (yearly_data['total_glacier_melt'] / yearly_data['input_total']) * 100
    yearly_data['snowmelt_contribution_pct'] = (yearly_data['total_snowmelt'] / yearly_data['obs_Q_mm']) * 100
    
    print(f"\nAnnual Water Balance Summary:")
    print(f"  Data source: {'GloGEM + HBV' if coupled else 'HBV only'}")
    print(f"  Mean annual precipitation: {yearly_data['total_precipitation'].mean():.1f} ± {yearly_data['total_precipitation'].std():.1f} mm/year")
    print(f"  Mean annual snowmelt: {yearly_data['total_snowmelt'].mean():.1f} ± {yearly_data['total_snowmelt'].std():.1f} mm/year")
    print(f"  Mean annual glacier melt: {yearly_data['total_glacier_melt'].mean():.1f} ± {yearly_data['total_glacier_melt'].std():.1f} mm/year")
    print(f"  Mean annual streamflow: {yearly_data['obs_Q_mm'].mean():.1f} ± {yearly_data['obs_Q_mm'].std():.1f} mm/year")
    print(f"  Mean runoff ratio: {yearly_data['runoff_ratio'].mean():.3f} ± {yearly_data['runoff_ratio'].std():.3f}")
    print(f"  Mean glacier contribution: {yearly_data['glacier_contribution_pct'].mean():.1f} ± {yearly_data['glacier_contribution_pct'].std():.1f}%")
    print(f"  Mean snowmelt contribution: {yearly_data['snowmelt_contribution_pct'].mean():.1f} ± {yearly_data['snowmelt_contribution_pct'].std():.1f}%")
    
    return yearly_data


#--------------------------------------------------------------------------------
#################################### run all ####################################
#--------------------------------------------------------------------------------

def run_complete_postprocessing(config, validation_start=None, validation_end=None):
    """
    Run complete postprocessing analysis for a single model configuration.
    Creates all diagnostic plots and analyses.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    validation_start : str, optional
        Start date for validation period (defaults to cali_end_date from config)
    validation_end : str, optional
        End date for validation period (defaults to end_date from config)
        
    Returns:
    --------
    dict
        Dictionary containing results from all analyses
    """
    
    gauge_id = config['gauge_id']
    model_type = config.get('model_type', 'HBV')
    coupled = config.get('coupled', False)
    
    print(f"\n{'='*80}")
    print(f"COMPLETE POSTPROCESSING ANALYSIS")
    print(f"{'='*80}")
    print(f"Catchment: {gauge_id}")
    print(f"Model: {model_type}")
    print(f"Coupled: {coupled}")
    print(f"Validation: {validation_start or config.get('cali_end_date', 'auto')} to {validation_end or config.get('end_date', 'auto')}")
    print(f"{'='*80}")
    
    # Setup output directories
    print("\n1. Setting up output directories...")
    try:
        plot_dirs = setup_output_directories(config)
        print(f"   ✓ Created plot directories")
    except Exception as e:
        print(f"   ✗ Error setting up directories: {e}")
        return None
    
    # Store results
    results = {
        'catchment_id': gauge_id,
        'model_type': model_type,
        'coupled': coupled,
        'success': {},
        'errors': {}
    }
    
    # ========================
    # HYDROGRAPH ANALYSIS
    # ========================
    print("\n2. Hydrograph Analysis...")
    
    try:
        print("   2.1 Performance Metrics Summary...")
        metrics_result = plot_performance_metrics_summary(config, plot_dirs)
        results['performance_metrics'] = metrics_result
        results['success']['performance_metrics'] = True
        print("       ✓ Performance metrics calculated and plotted")
    except Exception as e:
        print(f"       ✗ Error with performance metrics: {e}")
        results['errors']['performance_metrics'] = str(e)
        results['success']['performance_metrics'] = False
    
    try:
        print("   2.2 Hydrological Regime...")
        regime_result = plot_hydrological_regime(config, plot_dirs, validation_start, validation_end)
        results['hydrological_regime'] = regime_result
        results['success']['hydrological_regime'] = True
        print("       ✓ Hydrological regime plotted")
    except Exception as e:
        print(f"       ✗ Error with hydrological regime: {e}")
        results['errors']['hydrological_regime'] = str(e)
        results['success']['hydrological_regime'] = False
    
    try:
        print("   2.3 Hydrograph Time Series...")
        timeseries_result = plot_hydrograph_timeseries(config, plot_dirs, validation_start, validation_end)
        results['hydrograph_timeseries'] = timeseries_result
        results['success']['hydrograph_timeseries'] = True
        print("       ✓ Hydrograph time series plotted")
    except Exception as e:
        print(f"       ✗ Error with hydrograph time series: {e}")
        results['errors']['hydrograph_timeseries'] = str(e)
        results['success']['hydrograph_timeseries'] = False
    
    # ========================
    # SWE ANALYSIS
    # ========================
    print("\n3. Snow Water Equivalent Analysis...")
    
    try:
        print("   3.1 Area-Weighted SWE Time Series...")
        swe_timeseries_result = plot_area_weighted_swe_timeseries(config, plot_dirs, validation_start, validation_end)
        results['swe_timeseries'] = swe_timeseries_result
        results['success']['swe_timeseries'] = True
        print("       ✓ SWE time series plotted")
    except Exception as e:
        print(f"       ✗ Error with SWE time series: {e}")
        results['errors']['swe_timeseries'] = str(e)
        results['success']['swe_timeseries'] = False
    
    try:
        print("   3.2 SWE by Elevation Bands...")
        swe_elevation_result = plot_swe_time_series_by_elevation(config, plot_dirs, validation_start=validation_start, validation_end=validation_end)
        results['swe_elevation'] = swe_elevation_result
        results['success']['swe_elevation'] = True
        print("       ✓ SWE elevation bands plotted")
    except Exception as e:
        print(f"       ✗ Error with SWE elevation analysis: {e}")
        results['errors']['swe_elevation'] = str(e)
        results['success']['swe_elevation'] = False
    
    try:
        print("   3.3 Peak SWE Analysis...")
        peak_swe_result = analyze_peak_swe(config, plot_dirs, validation_start, validation_end)
        results['peak_swe_analysis'] = peak_swe_result
        results['success']['peak_swe_analysis'] = True
        print("       ✓ Peak SWE analysis completed")
    except Exception as e:
        print(f"       ✗ Error with peak SWE analysis: {e}")
        results['errors']['peak_swe_analysis'] = str(e)
        results['success']['peak_swe_analysis'] = False
    
    try:
        print("   3.4 Spatial SWE Distribution...")
        spatial_swe_result = plot_spatial_swe_distribution(config, plot_dirs, validation_start, validation_end)
        results['spatial_swe_distribution'] = spatial_swe_result
        results['success']['spatial_swe_distribution'] = True
        print("       ✓ Spatial SWE distribution plotted")
    except Exception as e:
        print(f"       ✗ Error with spatial SWE distribution: {e}")
        results['errors']['spatial_swe_distribution'] = str(e)
        results['success']['spatial_swe_distribution'] = False
    
    # ========================
    # PARAMETER ANALYSIS
    # ========================
    print("\n4. Parameter Analysis...")
    
    try:
        print("   4.1 Parameter Distributions...")
        param_result = plot_parameter_boxplots(config, plot_dirs, top_n=100)
        results['parameter_distributions'] = param_result
        results['success']['parameter_distributions'] = True
        print("       ✓ Parameter distributions plotted")
    except Exception as e:
        print(f"       ✗ Error with parameter analysis: {e}")
        results['errors']['parameter_distributions'] = str(e)
        results['success']['parameter_distributions'] = False
    
    # ========================
    # STORAGE ANALYSIS
    # ========================
    print("\n5. Storage Analysis...")
    
    try:
        print("   5.1 Storage Time Series...")
        storage_result = plot_storage_timeseries(config, plot_dirs, validation_start, validation_end)
        results['storage_timeseries'] = storage_result
        results['success']['storage_timeseries'] = True
        print("       ✓ Storage time series plotted")
    except Exception as e:
        print(f"       ✗ Error with storage analysis: {e}")
        results['errors']['storage_timeseries'] = str(e)
        results['success']['storage_timeseries'] = False
    
    # ========================
    # CONTRIBUTIONS ANALYSIS
    # ========================
    print("\n6. Contributions Analysis...")
    
    # Only run detailed contributions analysis if we have the required data
    if coupled:
        try:
            print("   6.1 GloGEM Regime Analysis...")
            glogem_result = plot_glogem_regime(config, plot_dirs, unit='mm')
            results['glogem_regime'] = glogem_result
            results['success']['glogem_regime'] = True
            print("       ✓ GloGEM regime analysis completed")
        except Exception as e:
            print(f"       ✗ Error with GloGEM analysis: {e}")
            results['errors']['glogem_regime'] = str(e)
            results['success']['glogem_regime'] = False
    else:
        print("   6.1 GloGEM Analysis skipped (uncoupled mode)")
        results['success']['glogem_regime'] = None
    
    try:
        print("   6.2 Glacier Contributions Regime...")
        glacier_contrib_result = plot_glacier_contributions_regime(config, plot_dirs, validation_start, validation_end)
        results['glacier_contributions'] = glacier_contrib_result
        results['success']['glacier_contributions'] = True
        print("       ✓ Glacier contributions regime plotted")
    except Exception as e:
        print(f"       ✗ Error with glacier contributions: {e}")
        results['errors']['glacier_contributions'] = str(e)
        results['success']['glacier_contributions'] = False
    
    try:
        print("   6.3 Non-Glacier Contributions Regime...")
        nonglacier_contrib_result = plot_nonglacier_contributions_regime(config, plot_dirs, validation_start, validation_end)
        results['nonglacier_contributions'] = nonglacier_contrib_result
        results['success']['nonglacier_contributions'] = True
        print("       ✓ Non-glacier contributions regime plotted")
    except Exception as e:
        print(f"       ✗ Error with non-glacier contributions: {e}")
        results['errors']['nonglacier_contributions'] = str(e)
        results['success']['nonglacier_contributions'] = False
    
    try:
        print("   6.4 Combined Contributions Dataframes...")
        glacier_df, nonglacier_df = create_combined_contributions_dataframes(config, plot_dirs, validation_start, validation_end)
        results['contributions_dataframes'] = {'glacier': glacier_df, 'nonglacier': nonglacier_df}
        results['success']['contributions_dataframes'] = True
        print("       ✓ Contributions dataframes created")
    except Exception as e:
        print(f"       ✗ Error creating contributions dataframes: {e}")
        results['errors']['contributions_dataframes'] = str(e)
        results['success']['contributions_dataframes'] = False
    
    try:
        print("   6.5 Combined Contributions Comparison...")
        combined_contrib_result = plot_combined_contributions_comparison(config, plot_dirs, validation_start, validation_end)
        results['combined_contributions'] = combined_contrib_result
        results['success']['combined_contributions'] = True
        print("       ✓ Combined contributions comparison plotted")
    except Exception as e:
        print(f"       ✗ Error with combined contributions: {e}")
        results['errors']['combined_contributions'] = str(e)
        results['success']['combined_contributions'] = False
    
    try:
        print("   6.6 Streamflow Contributions Regime...")
        streamflow_contrib_result = plot_streamflow_contributions_regime(config, plot_dirs, validation_start, validation_end)
        results['streamflow_contributions'] = streamflow_contrib_result
        results['success']['streamflow_contributions'] = True
        print("       ✓ Streamflow contributions regime plotted")
    except Exception as e:
        print(f"       ✗ Error with streamflow contributions: {e}")
        results['errors']['streamflow_contributions'] = str(e)
        results['success']['streamflow_contributions'] = False
    
    # ========================
    # WATER BALANCE ANALYSIS
    # ========================
    print("\n7. Water Balance Analysis...")
    
    try:
        print("   7.1 Yearly Precipitation vs Streamflow...")
        yearly_precip_result = plot_yearly_precipitation_streamflow(config, plot_dirs, validation_start, validation_end)
        results['yearly_precipitation_streamflow'] = yearly_precip_result
        results['success']['yearly_precipitation_streamflow'] = True
        print("       ✓ Yearly precipitation vs streamflow plotted")
    except Exception as e:
        print(f"       ✗ Error with yearly precipitation analysis: {e}")
        results['errors']['yearly_precipitation_streamflow'] = str(e)
        results['success']['yearly_precipitation_streamflow'] = False
    
    try:
        print("   7.2 Input vs Streamflow Scatter Plot...")
        scatter_result = plot_precipitation_glacier_melt_vs_streamflow_scatter(config, plot_dirs, validation_start, validation_end)
        results['input_streamflow_scatter'] = scatter_result
        results['success']['input_streamflow_scatter'] = True
        print("       ✓ Input vs streamflow scatter plot created")
    except Exception as e:
        print(f"       ✗ Error with scatter plot: {e}")
        results['errors']['input_streamflow_scatter'] = str(e)
        results['success']['input_streamflow_scatter'] = False
    
    try:
        print("   7.3 Annual Water Balance Bars...")
        bar_result = plot_annual_water_balance_bars(config, plot_dirs, validation_start, validation_end)
        results['annual_water_balance_bars'] = bar_result
        results['success']['annual_water_balance_bars'] = True
        print("       ✓ Annual water balance bars plotted")
    except Exception as e:
        print(f"       ✗ Error with annual water balance bars: {e}")
        results['errors']['annual_water_balance_bars'] = str(e)
        results['success']['annual_water_balance_bars'] = False
    
    try:
        print("   7.4 Average Yearly Water Balance...")
        avg_balance_result = plot_average_yearly_water_balance(config, plot_dirs, validation_start, validation_end)
        results['average_yearly_water_balance'] = avg_balance_result
        results['success']['average_yearly_water_balance'] = True
        print("       ✓ Average yearly water balance plotted")
    except Exception as e:
        print(f"       ✗ Error with average yearly water balance: {e}")
        results['errors']['average_yearly_water_balance'] = str(e)
        results['success']['average_yearly_water_balance'] = False
    
    # ========================
    # UNCERTAINTY ANALYSIS
    # ========================
    print("\n8. Uncertainty Analysis...")
    
    #try:
    #    print("   8.1 Combined Uncertainty with Envelope...")
        #uncertainty_result = plot_combined_uncertainty_with_envelope(config, plot_dirs, n_runs=100, validation_start=validation_start, validation_end=validation_end)
        #results['uncertainty_analysis'] = uncertainty_result
        #results['success']['uncertainty_analysis'] = True
        #print("       ✓ Uncertainty analysis completed")
    #except Exception as e:
        #print(f"       ✗ Error with uncertainty analysis: {e}")
        #results['errors']['uncertainty_analysis'] = str(e)
        #results['success']['uncertainty_analysis'] = False
    
    # ========================
    # SUMMARY
    # ========================
    print(f"\n{'='*80}")
    print(f"POSTPROCESSING SUMMARY")
    print(f"{'='*80}")
    
    success_count = sum(1 for success in results['success'].values() if success is True)
    total_analyses = len([k for k, v in results['success'].items() if v is not None])
    error_count = len(results['errors'])
    
    print(f"Catchment: {gauge_id}")
    print(f"Model: {model_type} ({'Coupled' if coupled else 'Uncoupled'})")
    print(f"Successful analyses: {success_count}/{total_analyses}")
    print(f"Failed analyses: {error_count}")
    
    if error_count > 0:
        print(f"\nErrors encountered:")
        for analysis, error in results['errors'].items():
            print(f"  - {analysis}: {error}")
    
    # List successful outputs
    config_dir = Path(config['main_dir']) / config['config_dir']
    output_base = config_dir / f"catchment_{gauge_id}" / model_type / "output"
    
    print(f"\nOutput locations:")
    print(f"  - Plots: {output_base / 'plots'}")
    print(f"  - Results: {output_base / 'results'}")
    
    # Print key results if available
    if results['success'].get('performance_metrics') and results['performance_metrics']:
        try:
            val_metrics = results['performance_metrics'].get('validation')
            if val_metrics:
                print(f"\nKey Performance Metrics (Validation):")
                print(f"  - NSE: {val_metrics['NSE']:.3f}")
                print(f"  - KGE: {val_metrics['KGE']:.3f}")
                print(f"  - KGE_NP: {val_metrics['KGE_NP']:.3f}")
        except:
            pass
    
    if results['success'].get('average_yearly_water_balance') and results['average_yearly_water_balance']:
        try:
            wb = results['average_yearly_water_balance']
            print(f"\nKey Water Balance Results:")
            print(f"  - Glacier contribution: {wb['glacier_pct']:.1f}% of simulated streamflow")
            print(f"  - Snowmelt contribution: {wb['snowmelt_pct']:.1f}% of simulated streamflow")
            print(f"  - Total melt contribution: {wb['glacier_pct'] + wb['snowmelt_pct']:.1f}% of simulated streamflow")
        except:
            pass
    
    if results['success'].get('uncertainty_analysis') and results['uncertainty_analysis']:
        try:
            unc = results['uncertainty_analysis']
            if unc.get('contribution_statistics'):
                stats = unc['contribution_statistics']
                print(f"\nUncertainty Ranges:")
                print(f"  - Glacier contribution: {stats['glacier_contribution']['min']:.1f}% - {stats['glacier_contribution']['max']:.1f}%")
                print(f"  - Snowmelt contribution: {stats['snowmelt_contribution']['min']:.1f}% - {stats['snowmelt_contribution']['max']:.1f}%")
                print(f"  - Successful runs analyzed: {unc['successful_runs']}")
        except:
            pass
    
    # Calculate success rate
    success_rate = (success_count / total_analyses * 100) if total_analyses > 0 else 0
    
    if success_rate >= 80:
        print(f"\n🎉 POSTPROCESSING COMPLETED SUCCESSFULLY! ({success_rate:.1f}% success rate)")
    elif success_rate >= 60:
        print(f"\n⚠️  POSTPROCESSING COMPLETED WITH SOME ISSUES ({success_rate:.1f}% success rate)")
    else:
        print(f"\n❌ POSTPROCESSING COMPLETED WITH SIGNIFICANT ISSUES ({success_rate:.1f}% success rate)")
    
    print(f"{'='*80}")
    
    return results


#--------------------------------------------------------------------------------
################################### forcing #####################################
#--------------------------------------------------------------------------------


