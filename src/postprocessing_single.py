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
            elif col.strip().lower() == 'precip [mm/day]':
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

def plot_hydrological_regime(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot the hydrological regime (monthly mean) for the catchment
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

    # Filter for validation period
    validation_mask = (data['date'] >= validation_start) & (data['date'] <= validation_end)
    df_validation = data[validation_mask].copy()

    if len(df_validation) == 0:
        print(f"Warning: No data found for validation period {validation_start} to {validation_end}")
        return None

    # Calculate monthly means
    df_validation['month'] = df_validation['date'].dt.month
    monthly_data = {}

    if 'sim_Q' in df_validation.columns:
        monthly_data['sim_Q'] = df_validation.groupby('month')['sim_Q'].mean()

    if 'obs_Q' in df_validation.columns:
        monthly_data['obs_Q'] = df_validation.groupby('month')['obs_Q'].mean()

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
    plt.ylabel('Discharge (m³/s)', fontsize=14)
    plt.title(f'Hydrological Regime - Monthly Mean for Validation Period ({validation_start} to {validation_end})\nCatchment {config["gauge_id"]}', fontsize=16)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)

    # Add performance metrics if both sim and obs are available
    if 'obs_Q' in df_validation.columns and 'sim_Q' in df_validation.columns:
        obs = df_validation['obs_Q'].values
        sim = df_validation['sim_Q'].values
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
    save_path = plot_dirs['hydrographs'] / f'hydrological_regime_{config["gauge_id"]}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved hydrological regime plot to: {save_path}")
    plt.show()

    return monthly_df

#--------------------------------------------------------------------------------

def plot_hydrograph_timeseries(config, plot_dirs, validation_start=None, validation_end=None, random_seed=42):
    """
    Plot the hydrograph time series for calibration and validation periods in two subplots,
    and plot a random year from the validation period.
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

    # Calibration and validation masks
    cali_mask = (data['date'] >= cali_start) & (data['date'] <= cali_end)
    val_mask = (data['date'] >= validation_start) & (data['date'] <= validation_end)

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    
    # Calibration period
    ax = axes[0]
    if 'obs_Q' in data.columns:
        ax.plot(data[cali_mask]['date'], data[cali_mask]['obs_Q'], 'k-', label='Observed')
    if 'sim_Q' in data.columns:
        ax.plot(data[cali_mask]['date'], data[cali_mask]['sim_Q'], 'C0', label='Simulated')
    ax.set_title(f'Calibration Period ({cali_start} to {cali_end})')
    ax.set_ylabel('Discharge (m³/s)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # Validation period
    ax = axes[1]
    if 'obs_Q' in data.columns:
        ax.plot(data[val_mask]['date'], data[val_mask]['obs_Q'], 'k-', label='Observed')
    if 'sim_Q' in data.columns:
        ax.plot(data[val_mask]['date'], data[val_mask]['sim_Q'], 'C0', label='Simulated')
    ax.set_title(f'Validation Period ({validation_start} to {validation_end})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Discharge (m³/s)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = plot_dirs['hydrographs'] / f'hydrograph_timeseries_split_{config["gauge_id"]}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved split hydrograph time series plot to: {save_path}")
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
    if 'obs_Q' in data.columns:
        plt.plot(data[year_mask]['date'], data[year_mask]['obs_Q'], 'k-', label='Observed')
    if 'sim_Q' in data.columns:
        plt.plot(data[year_mask]['date'], data[year_mask]['sim_Q'], 'C0', label='Simulated')
    plt.xlabel('Date')
    plt.ylabel('Discharge (m³/s)')
    plt.title(f'Hydrograph for Random Validation Year {rand_year} - Catchment {config["gauge_id"]}')
    plt.legend()
    plt.tight_layout()
    save_path = plot_dirs['hydrographs'] / f'hydrograph_random_year_{rand_year}_{config["gauge_id"]}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved hydrograph for random year plot to: {save_path}")
    plt.show()

#--------------------------------------------------------------------------------
###################################### SWE ######################################
#--------------------------------------------------------------------------------

def load_swe_data(config):
    """
    Load SWE data files with improved efficiency.
    Also loads elevation band areas for weighted metrics.
    Makes observed data optional - returns None if not available.
    """
    config_dir = Path(config['main_dir']) / config['config_dir']
    gauge_id = config['gauge_id']
    model_type = config['model_type']
    
    sim_file = config_dir / f"catchment_{gauge_id}" / model_type / "output" / f"{gauge_id}_{model_type}_SNOW_Daily_Average_ByHRUGroup.csv"
    obs_file = config_dir / f"catchment_{gauge_id}" / model_type / "data_obs" / "swe_by_elevation_band.csv"
    area_file = config_dir / f"catchment_{gauge_id}" / model_type / "data_obs" / "elevation_band_areas.csv"
    
    print(f"Loading SWE data:")
    print(f"  - Simulated data: {sim_file}")
    print(f"  - Observed data: {obs_file}")
    print(f"  - Area data: {area_file}")
    
    # Check if simulated file exists (required)
    if not sim_file.exists():
        print(f"ERROR: Required simulated SWE file missing: {sim_file}")
        return None, None, None
    
    try:
        # Load simulated data (required)
        with open(sim_file, 'r') as f:
            header_line = f.readline().strip()
            units_line = f.readline().strip()
        
        sim_data = pd.read_csv(sim_file, skiprows=[1], header=0)
        
        if sim_data.columns[0] == '':
            sim_data = sim_data.rename(columns={sim_data.columns[0]: 'row_id'})
        
        print(f"  ✅ Successfully loaded simulated SWE data ({len(sim_data)} records)")
        
        # Load observed data (optional)
        obs_data = None
        if obs_file.exists():
            try:
                obs_data = pd.read_csv(obs_file)
                obs_data['time'] = pd.to_datetime(obs_data['time'])
                
                # Ensure sim_data has proper date alignment with obs_data
                min_length = min(len(sim_data), len(obs_data))
                sim_data['date'] = obs_data['time'].iloc[:min_length].values
                
                print(f"  ✅ Successfully loaded observed SWE data ({len(obs_data)} records)")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not load observed SWE data: {e}")
                obs_data = None
                # Create date column from row index for sim_data
                sim_data['date'] = pd.date_range(start='2000-01-01', periods=len(sim_data), freq='D')
        else:
            print(f"  ℹ️  Observed SWE file not found (optional): {obs_file}")
            obs_data = None
            # Create date column from row index for sim_data
            sim_data['date'] = pd.date_range(start='2000-01-01', periods=len(sim_data), freq='D')
        
        # Load area data (optional)
        area_data = None
        if area_file.exists():
            try:
                area_data = pd.read_csv(area_file)
                print(f"  ✅ Successfully loaded elevation band areas")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not load elevation band areas: {e}")
        else:
            print(f"  ℹ️  Elevation band area file not found (optional): {area_file}")
        
        return sim_data, obs_data, area_data
        
    except Exception as e:
        print(f"ERROR: Failed to load SWE data: {e}")
        return None, None, None
    

#--------------------------------------------------------------------------------

def process_swe_data(sim_data, obs_data, area_data=None):
    """
    Process and align simulated and observed SWE data for single catchment analysis.
    Now handles cases where obs_data is None.
    """
    if sim_data is None:
        return None
    
    # Get elevation band columns efficiently using regex pattern
    sim_elev_pattern = re.compile(r'\d+-\d+m')
    sim_elev_cols = [col for col in sim_data.columns if sim_elev_pattern.search(col)]
    
    print(f"Found {len(sim_elev_cols)} simulation elevation bands")
    
    # Process observed data if available
    obs_elev_cols = []
    band_mapping = {}
    
    if obs_data is not None:
        obs_elev_cols = [col for col in obs_data.columns if sim_elev_pattern.search(col)]
        print(f"Found {len(obs_elev_cols)} observation elevation bands")
        
        # Create mapping between sim and obs bands
        band_mapping = {band: band for band in sim_elev_cols if band in obs_elev_cols}
        print(f"Found {len(band_mapping)} matching elevation bands")
        
        # Convert observed data to numeric
        for col in obs_elev_cols:
            obs_data[col] = pd.to_numeric(obs_data[col], errors='coerce')
    else:
        print("No observed data available - processing simulated data only")
        # For simulated-only analysis, we can still use the elevation bands
        band_mapping = {band: band for band in sim_elev_cols}
    
    # Create area mapping if available
    area_mapping = {}
    if area_data is not None:
        print(f"  - Area data columns: {area_data.columns.tolist()}")
        
        # Check if the first column is unnamed but contains the elevation bands
        if 'Unnamed: 0' in area_data.columns and 'area_km2' in area_data.columns:
            area_dict = dict(zip(area_data['Unnamed: 0'].astype(str), area_data['area_km2']))
            
            matched_bands = 0
            for band in sim_elev_cols:  # Use sim_elev_cols instead of band_mapping
                if band in area_dict:
                    area_mapping[band] = area_dict[band]
                    matched_bands += 1
                    print(f"  - Found area for band {band}: {area_mapping[band]} km²")
                else:
                    print(f"  - Warning: No area data found for band {band}")
            
            print(f"  - Successfully matched {matched_bands}/{len(sim_elev_cols)} bands with area data")
        else:
            # Try other column combinations
            for band in sim_elev_cols:
                if band in area_data.columns:
                    area_mapping[band] = area_data[band].iloc[0]
    else:
        # Create equal-weight area mapping for all elevation bands
        area_mapping = {band: 1.0 for band in sim_elev_cols}
        print(f"  - Using equal weights for {len(sim_elev_cols)} elevation bands")
    
    # Convert simulated data to numeric
    for col in sim_elev_cols:
        sim_data[col] = pd.to_numeric(sim_data[col], errors='coerce')
    
    return {
        'sim_data': sim_data,
        'obs_data': obs_data,  # Can be None
        'sim_elev_cols': sim_elev_cols,
        'obs_elev_cols': obs_elev_cols,
        'band_mapping': band_mapping,
        'area_mapping': area_mapping
    }

#--------------------------------------------------------------------------------

def calculate_swe_metrics(sim_data, obs_data, sim_cols, obs_cols, band_mapping, area_mapping=None):
    """
    Calculate SWE comparison metrics between simulated and observed data.
    """
    metrics = {
        'rmse_by_band': {},
        'bias_by_band': {},
        'corr_by_band': {},
        'overall_rmse': None,
        'overall_bias': None,
        'overall_corr': None,
        'area_weighted_rmse': None,
        'area_weighted_bias': None,    # NEW
        'area_weighted_corr': None,    # NEW
        'total_area': None,
        'used_bands': []
    }
    
    all_sim_values = []
    all_obs_values = []
    band_metrics_values = {}  # Store RMSE, bias, and correlation with areas
    used_area = 0
    
    # Calculate metrics for each band
    for sim_band, obs_band in band_mapping.items():
        # Create merged dataframe
        merged = pd.DataFrame({
            'date': sim_data['date'],
            'sim': sim_data[sim_band]
        })
        
        # Add observed values
        merged = pd.merge(
            merged, 
            obs_data[['time', obs_band]].rename(columns={'time': 'date', obs_band: 'obs'}),
            on='date', 
            how='inner'
        ).dropna()
        
        if len(merged) > 0:
            # Convert from model units to mm if needed
            if merged['sim'].mean() < 10 and merged['sim'].max() < 20:
                merged['sim'] = merged['sim'] * 1000
            
            if merged['obs'].mean() < 10 and merged['obs'].max() < 20:
                merged['obs'] = merged['obs'] * 1000
            
            # Calculate metrics
            diff = merged['sim'] - merged['obs']
            rmse = np.sqrt(np.mean(diff**2))
            bias = np.mean(diff)
            corr = np.corrcoef(merged['sim'], merged['obs'])[0, 1] if len(merged) > 2 else np.nan
            
            # Store metrics
            metrics['rmse_by_band'][sim_band] = rmse
            metrics['bias_by_band'][sim_band] = bias
            metrics['corr_by_band'][sim_band] = corr
            
            # Collect for overall metrics
            all_sim_values.extend(merged['sim'].values)
            all_obs_values.extend(merged['obs'].values)
            
            # Store for area-weighted calculation
            if area_mapping and sim_band in area_mapping:
                area = area_mapping[sim_band]
                band_metrics_values[sim_band] = (rmse, bias, corr, area)  # Store all metrics with area
                used_area += area
                metrics['used_bands'].append(sim_band)
    
    # Calculate overall metrics
    if all_sim_values and all_obs_values:
        all_sim = np.array(all_sim_values)
        all_obs = np.array(all_obs_values)
        all_diff = all_sim - all_obs
        
        metrics['overall_rmse'] = np.sqrt(np.mean(all_diff**2))
        metrics['overall_bias'] = np.mean(all_diff)
        metrics['overall_corr'] = np.corrcoef(all_sim, all_obs)[0, 1]
    
    # Calculate area-weighted metrics
    if band_metrics_values and used_area > 0:
        weighted_rmse_sum = 0
        weighted_bias_sum = 0
        weighted_corr_sum = 0
        
        for band, (rmse, bias, corr, area) in band_metrics_values.items():
            weight = area / used_area
            weighted_rmse_sum += rmse * weight
            weighted_bias_sum += bias * weight
            # Only add correlation if it's not NaN
            if not np.isnan(corr):
                weighted_corr_sum += corr * weight
        
        metrics['area_weighted_rmse'] = weighted_rmse_sum
        metrics['area_weighted_bias'] = weighted_bias_sum
        metrics['area_weighted_corr'] = weighted_corr_sum
        metrics['total_area'] = used_area
    
    return metrics


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
    Plot area-weighted SWE time series for single catchment with metrics displayed.
    """
    # Load and process data
    sim_data, obs_data, area_data = load_swe_data(config)
    if sim_data is None or obs_data is None:
        print("Failed to load SWE data")
        return None
    
    processed = process_swe_data(sim_data, obs_data, area_data)
    if processed is None:
        print("Failed to process SWE data")
        return None
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    validation_start = pd.to_datetime(validation_start)
    validation_end = pd.to_datetime(validation_end)
    
    # Get processed components
    band_mapping = processed['band_mapping']
    area_mapping = processed['area_mapping']
    sim_data = processed['sim_data']
    obs_data = processed['obs_data']
    
    # Filter for validation period
    sim_data['date'] = pd.to_datetime(sim_data['date'])
    obs_data['time'] = pd.to_datetime(obs_data['time'])
    
    val_sim_mask = (sim_data['date'] >= validation_start) & (sim_data['date'] <= validation_end)
    val_obs_mask = (obs_data['time'] >= validation_start) & (obs_data['time'] <= validation_end)
    
    val_sim = sim_data[val_sim_mask].copy()
    val_obs = obs_data[val_obs_mask].copy()
    
    if len(val_sim) == 0:
        print("No simulation data found for validation period")
        return None
    
    # Calculate area-weighted SWE
    val_sim['area_weighted_swe'] = calculate_area_weighted_swe(val_sim, area_mapping)
    
    if len(val_obs) > 0:
        val_obs['area_weighted_swe'] = calculate_area_weighted_swe(val_obs, area_mapping)
    
    # Calculate metrics
    metrics = calculate_swe_metrics(
        sim_data, obs_data,
        processed['sim_elev_cols'], processed['obs_elev_cols'],
        band_mapping, area_mapping
    )
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot simulated area-weighted SWE
    plt.plot(val_sim['date'], val_sim['area_weighted_swe'], 
             'C0', linewidth=2, label='Simulated Area-Weighted SWE')
    
    # Plot observed area-weighted SWE if available
    if len(val_obs) > 0 and 'area_weighted_swe' in val_obs.columns:
        plt.plot(val_obs['time'], val_obs['area_weighted_swe'], 
                 'k-', linewidth=2, label='Observed Area-Weighted SWE')
    
    # Format plot
    plt.title(f'Area-Weighted SWE Time Series - Catchment {config["gauge_id"]}\n'
              f'Validation Period: {validation_start.date()} to {validation_end.date()}', 
              fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Snow Water Equivalent (mm)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gcf().autofmt_xdate()
    
    # Add metrics text box with area-weighted metrics
    if metrics['overall_rmse'] is not None:
        metric_text = (
            f"SWE Metrics:\n"
            f"RMSE: {metrics['overall_rmse']:.1f} mm\n"
            f"Bias: {metrics['overall_bias']:.1f} mm\n"
            f"Correlation: {metrics['overall_corr']:.3f}"
        )
        
        if metrics['area_weighted_rmse'] is not None:
            metric_text += f"\n\nArea-Weighted Metrics:"
            metric_text += f"\nRMSE: {metrics['area_weighted_rmse']:.1f} mm"
            metric_text += f"\nBias: {metrics['area_weighted_bias']:.1f} mm"
            metric_text += f"\nCorrelation: {metrics['area_weighted_corr']:.3f}"
            metric_text += f"\nTotal area: {metrics['total_area']:.1f} km²"
        
        plt.figtext(0.02, 0.02, metric_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['swe'] / f'area_weighted_swe_timeseries_{config["gauge_id"]}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved area-weighted SWE plot to: {save_path}")
    plt.show()
    
    # Print summary with area-weighted metrics
    if metrics['overall_rmse'] is not None:
        print(f"\nSWE Analysis Results for Catchment {config['gauge_id']}:")
        print(f"  Overall RMSE: {metrics['overall_rmse']:.2f} mm")
        print(f"  Overall Bias: {metrics['overall_bias']:.2f} mm")
        print(f"  Overall Correlation: {metrics['overall_corr']:.3f}")
        if metrics['area_weighted_rmse'] is not None:
            print(f"  Area-weighted RMSE: {metrics['area_weighted_rmse']:.2f} mm")
            print(f"  Area-weighted Bias: {metrics['area_weighted_bias']:.2f} mm")
            print(f"  Area-weighted Correlation: {metrics['area_weighted_corr']:.3f}")
            print(f"  Total catchment area: {metrics['total_area']:.2f} km²")
    
    return val_sim

#--------------------------------------------------------------------------------

def plot_swe_time_series_by_elevation(config, plot_dirs, water_year=None, validation_start=None, validation_end=None):
    """
    Plot time series of SWE for each elevation band comparing observed and simulated data
    for a single catchment configuration.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    water_year : int, optional
        Optional water year to filter (e.g. 2018 for 2018-2019 water year)
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    """
    
    # Load and process data
    sim_data, obs_data, area_data = load_swe_data(config)
    if sim_data is None or obs_data is None:
        print("Failed to load SWE data")
        return None
    
    processed = process_swe_data(sim_data, obs_data, area_data)
    if processed is None:
        print("Failed to process SWE data")
        return None
    
    # Get processed components
    sim_data = processed['sim_data']
    obs_data = processed['obs_data']
    sim_elev_cols = processed['sim_elev_cols']
    obs_elev_cols = processed['obs_elev_cols']
    band_mapping = processed['band_mapping']
    
    if not band_mapping:
        print("No matching elevation bands found")
        return None
    
    # Convert date columns to datetime
    sim_data['date'] = pd.to_datetime(sim_data['date'])
    obs_data['time'] = pd.to_datetime(obs_data['time'])
    
    # Filter data based on time period
    if water_year is not None:
        # Water year: October 1 to September 30
        start_date = pd.to_datetime(f"{water_year}-10-01")
        end_date = pd.to_datetime(f"{water_year+1}-09-30")
        period_label = f"Water Year {water_year}-{water_year+1}"
    else:
        # Use validation period or config dates
        if validation_start is None:
            validation_start = config.get('cali_end_date', '2010-01-01')
        if validation_end is None:
            validation_end = config.get('end_date', '2020-12-31')
        
        start_date = pd.to_datetime(validation_start)
        end_date = pd.to_datetime(validation_end)
        period_label = f"Validation Period ({start_date.date()} to {end_date.date()})"
    
    # Filter data
    sim_mask = (sim_data['date'] >= start_date) & (sim_data['date'] <= end_date)
    obs_mask = (obs_data['time'] >= start_date) & (obs_data['time'] <= end_date)
    
    sim_filtered = sim_data[sim_mask].copy()
    obs_filtered = obs_data[obs_mask].copy()
    
    if len(sim_filtered) == 0 or len(obs_filtered) == 0:
        print(f"No data found for the specified period: {start_date} to {end_date}")
        return None
    
    # Sort elevation bands by altitude
    elev_bands = sorted(band_mapping.keys(), key=lambda x: int(x.split('-')[0]) if '-' in x else 0)
    
    # Calculate number of subplots needed - use 1 column
    n_bands = len(elev_bands)
    n_cols = 1  # Single column layout
    n_rows = n_bands
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3*n_rows), sharex=True)
    
    # Make axes iterable if there's only one plot
    if n_bands == 1:
        axes = np.array([axes])
    
    # Plot each elevation band
    for i, band in enumerate(elev_bands):
        ax = axes[i]
        
        # Plot observed data
        if band in obs_filtered.columns:
            obs_values = obs_filtered[band].copy()
            # Convert to mm if needed
            if obs_values.mean() < 10 and obs_values.max() < 20:
                obs_values *= 1000
            
            ax.plot(obs_filtered['time'], obs_values, 
                    'k-', label='Observed', linewidth=2)
        
        # Plot simulated data
        if band in sim_filtered.columns:
            sim_values = sim_filtered[band].copy()
            # Convert to mm if needed
            if sim_values.mean() < 10 and sim_values.max() < 20:
                sim_values *= 1000
            
            ax.plot(sim_filtered['date'], sim_values, 
                    'C0', label='Simulated', linewidth=1.5)
        
        # Calculate band-specific metrics if both obs and sim data exist
        if band in obs_filtered.columns and band in sim_filtered.columns:
            # Merge data for metrics calculation
            merged = pd.merge(
                sim_filtered[['date', band]].rename(columns={'date': 'time', band: 'sim'}),
                obs_filtered[['time', band]].rename(columns={band: 'obs'}),
                on='time', how='inner'
            ).dropna()
            
            if len(merged) > 0:
                # Convert units if needed
                if merged['sim'].mean() < 10:
                    merged['sim'] *= 1000
                if merged['obs'].mean() < 10:
                    merged['obs'] *= 1000
                
                # Calculate metrics
                rmse = np.sqrt(np.mean((merged['sim'] - merged['obs'])**2))
                bias = np.mean(merged['sim'] - merged['obs'])
                corr = np.corrcoef(merged['sim'], merged['obs'])[0, 1] if len(merged) > 2 else np.nan
                
                # Add metrics text to plot
                metrics_text = f"RMSE: {rmse:.1f} mm\nBias: {bias:.1f} mm\nR: {corr:.3f}"
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_title(f'Elevation Band: {band}', fontsize=12)
        ax.set_ylabel('SWE (mm)', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        # Add legend only to the first plot
        if i == 0:
            ax.legend(loc='upper right', fontsize=10)
    
    # Add overall title
    title = f'SWE by Elevation Band - Catchment {config["gauge_id"]}\n{period_label}'
    fig.suptitle(title, fontsize=16)
    
    # Format x-axis for the bottom plot
    axes[-1].set_xlabel('Date', fontsize=12)
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.3)
    
    # Save figure
    if water_year:
        filename = f'swe_time_series_by_elevation_WY{water_year}_{config["gauge_id"]}.png'
    else:
        filename = f'swe_time_series_by_elevation_{config["gauge_id"]}.png'
    
    save_path = plot_dirs['swe'] / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved SWE elevation band plot to: {save_path}")
    plt.show()
    
    # Print summary statistics
    print(f"\nSWE Elevation Band Analysis for Catchment {config['gauge_id']}:")
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    print(f"  Number of elevation bands: {len(elev_bands)}")
    print(f"  Elevation bands: {', '.join(elev_bands)}")
    
    return fig


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
    Load and process GloGEM data from the preprocessed CSV file.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    unit : str
        Output unit ('mm' or 'm3')
    plot : bool
        Whether to create plots
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing processed GloGEM data with columns:
        ['date', 'glacier_melt', 'snowmelt', 'rainfall', 'total_output']
    """
    
    config_dir = Path(config['main_dir']) / config['config_dir']
    gauge_id = config['gauge_id']
    
    print(f"Loading GloGEM data for catchment {gauge_id}:")
    
    # Load the preprocessed GloGEM CSV file from topo_files
    topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
    glogem_csv_file = topo_dir / "GloGEM_melt.csv"
    
    if not glogem_csv_file.exists():
        print(f"ERROR: GloGEM CSV file not found: {glogem_csv_file}")
        print("Please run the preprocess_glogem script first to generate this file.")
        return None
    
    print(f"  - Loading from: {glogem_csv_file}")
    
    # Get date range from config
    start_date = config.get('start_date', '2000-01-01')
    end_date = config.get('end_date', '2020-12-31')
    
    try:
        # Load the individual glacier records from CSV
        glogem_df = pd.read_csv(glogem_csv_file, dtype={'id': str})
        glogem_df['date'] = pd.to_datetime(glogem_df['date'])
        
        print(f"  - Loaded {len(glogem_df)} records for {glogem_df['id'].nunique()} glaciers")
        print(f"  - Date range in file: {glogem_df['date'].min()} to {glogem_df['date'].max()}")
        
        # Filter for the specified date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        date_mask = (glogem_df['date'] >= start) & (glogem_df['date'] <= end)
        glogem_filtered = glogem_df[date_mask].copy()
        
        if len(glogem_filtered) == 0:
            print(f"ERROR: No GloGEM data found for period {start_date} to {end_date}")
            return None
        
        print(f"  - Filtered to {len(glogem_filtered)} records for period {start_date} to {end_date}")
        
        # Load catchment shapefile to get areas and scaling
        catchment_shape_file = topo_dir / "HRU.shp"
        
        if not catchment_shape_file.exists():
            print(f"ERROR: Catchment shapefile not found: {catchment_shape_file}")
            return None
        
        try:
            catchment = gpd.read_file(catchment_shape_file)
        except Exception as e:
            print(f"ERROR: Failed to read catchment shapefile: {e}")
            return None
        
        # Calculate catchment areas for scaling
        glacier_areas = catchment.groupby('Glacier_Cl').agg({'Area_km2': 'sum'}).reset_index()
        glacier_area = glacier_areas['Area_km2'].sum()
        total_area = catchment['Area_km2'].sum()
        percentage = (glacier_area / total_area) * 100
        
        print(f"  - Total catchment area: {total_area:.2f} km²")
        print(f"  - Glaciated area: {glacier_area:.2f} km² ({percentage:.1f}%)")
        
        # Create area mapping for individual glaciers
        area_dict = {}
        for _, row in glacier_areas.iterrows():
            if pd.notna(row['Glacier_Cl']):
                # Extract the numeric part of glacier ID (remove RGI prefix)
                glacier_id = str(row['Glacier_Cl']).split('.')[-1] if '.' in str(row['Glacier_Cl']) else str(row['Glacier_Cl'])
                area_dict[glacier_id] = row['Area_km2']
        
        print(f"  - Found area mapping for {len(area_dict)} glaciers")
        
        # Calculate area-weighted daily averages
        print(f"  - Calculating area-weighted catchment averages...")
        
        # Add area information to glacier records
        glogem_filtered['area'] = glogem_filtered['id'].map(area_dict)
        
        # Fill missing areas with mean area (for glaciers not in shapefile)
        missing_area_mask = glogem_filtered['area'].isna()
        if missing_area_mask.any():
            mean_area = glogem_filtered['area'].mean()
            glogem_filtered.loc[missing_area_mask, 'area'] = mean_area
            print(f"    Warning: {missing_area_mask.sum()} records had missing area data, used mean area {mean_area:.3f} km²")
        
        # Calculate area-weighted daily totals
        def calculate_daily_totals(df):
            """Calculate area-weighted daily totals for the catchment"""
            total_area = df['area'].sum()
            if total_area == 0:
                return 0
            
            # Weight by area and sum
            weighted_total = (df['q'] * df['area']).sum()
            # Return as catchment average (weighted by total glacier area)
            return weighted_total / total_area
        
        # Group by date and calculate catchment average
        daily_averages = glogem_filtered.groupby('date').apply(calculate_daily_totals).reset_index()
        daily_averages.columns = ['date', 'glacier_melt']
        
        # Create the output dataframe in the expected format
        # Note: Since we only have total melt from GloGEM, we'll use it for glacier_melt
        # and set other components to appropriate values
        result_df = daily_averages.copy()
        result_df['snowmelt'] = 0.0  # GloGEM total melt already includes snow melt
        result_df['rainfall'] = 0.0  # GloGEM doesn't provide rainfall component separately
        result_df['total_output'] = result_df['glacier_melt']  # Total output is the same as glacier melt
        
        # Apply scaling factor based on glaciated percentage of catchment
        scaling_factor = percentage / 100
        result_df['glacier_melt'] = result_df['glacier_melt'] * scaling_factor
        result_df['snowmelt'] = result_df['snowmelt'] * scaling_factor
        result_df['total_output'] = result_df['total_output'] * scaling_factor
        
        print(f"  - Applied scaling factor: {scaling_factor:.3f} (glaciated percentage)")
        
        # Convert units if necessary
        if unit == 'm3':
            catchment_area_m2 = total_area * 1000000  # Convert km² to m²
            for col in ['glacier_melt', 'snowmelt', 'rainfall', 'total_output']:
                result_df[col] = result_df[col] * catchment_area_m2 / 1000  # mm to m³
            print(f"  - Converted units to m³/day")
        
        print(f"  - Final dataset: {len(result_df)} daily records")
        print(f"  - Mean glacier melt: {result_df['glacier_melt'].mean():.3f} {unit}/day")
        print(f"  - Max glacier melt: {result_df['glacier_melt'].max():.3f} {unit}/day")
        print(f"  - Total glacier melt: {result_df['glacier_melt'].sum():.1f} {unit}")
        
        # Create a cached parsed file for compatibility
        parsed_file = topo_dir / f"GloGEM_parsed_{gauge_id}.csv"
        result_df.to_csv(parsed_file, index=False)
        print(f"  - Saved parsed result to: {parsed_file}")
        
        return result_df
        
    except Exception as e:
        print(f"ERROR: Failed to process GloGEM CSV file: {e}")
        import traceback
        traceback.print_exc()
        return None

#--------------------------------------------------------------------------------

def plot_glogem_regime(config, plot_dirs, unit='mm'):
    """
    Plot GloGEM monthly and daily regimes for the catchment.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    unit : str
        Output unit ('mm' or 'm3')
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
    
    # Calculate monthly regime
    monthly_regime = glogem_df.groupby('month').agg({
        'glacier_melt': 'mean',
        'snowmelt': 'mean',
        'rainfall': 'mean',
        'total_output': 'mean'
    }).reset_index()
    
    # Calculate daily regime (averaged over all years)
    daily_regime = glogem_df.groupby('day_of_year').agg({
        'glacier_melt': 'mean',
        'snowmelt': 'mean',
        'rainfall': 'mean',
        'total_output': 'mean'
    }).reset_index()
    
    # Create date series for daily regime (using non-leap year)
    daily_regime['date'] = pd.to_datetime('2001-01-01') + pd.to_timedelta(daily_regime['day_of_year'] - 1, unit='days')
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Unit label
    unit_label = 'mm/day' if unit == 'mm' else 'm³/day'
    
    # Monthly regime plot
    ax1.plot(monthly_regime['month'], monthly_regime['glacier_melt'], 'b-', 
             label='Glacier Melt', linewidth=2, marker='o')
    ax1.plot(monthly_regime['month'], monthly_regime['snowmelt'], 'c-', 
             label='Snowmelt', linewidth=2, marker='s')
    ax1.plot(monthly_regime['month'], monthly_regime['rainfall'], 'g-', 
             label='Rainfall', linewidth=2, marker='^')
    ax1.plot(monthly_regime['month'], monthly_regime['total_output'], 'k-', 
             label='Total Output', linewidth=2.5, marker='D')
    
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel(f'Average Runoff ({unit_label})', fontsize=12)
    ax1.set_title(f'Monthly Regime - Catchment {config["gauge_id"]}', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax1.legend(loc='best')
    
    # Daily regime plot
    ax2.plot(daily_regime['date'], daily_regime['glacier_melt'], 'b-', 
             label='Glacier Melt', linewidth=1.5)
    ax2.plot(daily_regime['date'], daily_regime['snowmelt'], 'c-', 
             label='Snowmelt', linewidth=1.5)
    ax2.plot(daily_regime['date'], daily_regime['rainfall'], 'g-', 
             label='Rainfall', linewidth=1.5)
    ax2.plot(daily_regime['date'], daily_regime['total_output'], 'k-', 
             label='Total Output', linewidth=2)
    
    # Format x-axis for daily plot
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel(f'Average Runoff ({unit_label})', fontsize=12)
    ax2.set_title('Daily Average Runoff (Averaged Over All Years)', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='best')
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'glogem_regime_{config["gauge_id"]}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved GloGEM regime plot to: {save_path}")
    plt.show()
    
    # Print summary statistics
    total_annual = glogem_df.groupby('year')[['glacier_melt', 'snowmelt', 'rainfall', 'total_output']].sum().mean()
    
    print(f"\nGloGEM Analysis Summary for Catchment {config['gauge_id']}:")
    print(f"  Period: {glogem_df['date'].min().date()} to {glogem_df['date'].max().date()}")
    print(f"  Annual averages ({unit_label.replace('/day', '/year')}):")
    print(f"    Glacier melt: {total_annual['glacier_melt']:.1f}")
    print(f"    Snowmelt: {total_annual['snowmelt']:.1f}")
    print(f"    Rainfall: {total_annual['rainfall']:.1f}")
    print(f"    Total output: {total_annual['total_output']:.1f}")
    
    return fig

#--------------------------------------------------------------------------------

def load_glacier_contributions_data(config, validation_start=None, validation_end=None):
    """
    Extract all glacier contribution data (precipitation, snowmelt, and ice melt) from HBV model output.
    
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
        DataFrame containing all glacier contribution data with monthly statistics
    """
    
    config_dir = Path(config['main_dir']) / config['config_dir']
    gauge_id = config['gauge_id']
    model_type = config['model_type']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"Loading glacier contributions data for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    
    # Load HRU shapefile to identify glacier HRUs
    topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
    catchment_shape_file = topo_dir / "HRU.shp"
    
    if not catchment_shape_file.exists():
        print(f"ERROR: HRU shapefile not found: {catchment_shape_file}")
        return None
    
    try:
        hru_gdf = gpd.read_file(catchment_shape_file)
    except Exception as e:
        print(f"ERROR: Failed to read HRU shapefile: {e}")
        return None
    
    # Calculate total catchment area
    total_catchment_area = hru_gdf['Area_km2'].sum()
    
    # Identify glacier HRUs using Landuse_Cl == 7
    glacier_hrus = hru_gdf[hru_gdf['Landuse_Cl'] == 7]
    
    if len(glacier_hrus) == 0:
        print("WARNING: No glacier HRUs found (Landuse_Cl == 7)")
        return None
    
    glacier_hru_ids = glacier_hrus['HRU_ID'].astype(str).tolist()
    glacier_hru_areas = dict(zip(glacier_hrus['HRU_ID'].astype(str), glacier_hrus['Area_km2']))
    
    # Calculate glacier area and area ratio
    glacier_area = glacier_hrus['Area_km2'].sum()
    area_ratio = glacier_area / total_catchment_area
    
    print(f"  - Total catchment area: {total_catchment_area:.2f} km²")
    print(f"  - Glacier area: {glacier_area:.2f} km² ({area_ratio:.1%})")
    print(f"  - Number of glacier HRUs: {len(glacier_hrus)}")
    
    # Define file paths
    model_dir = config_dir / f"catchment_{gauge_id}" / model_type
    files = {
        'rainfall': model_dir / "output" / f"{gauge_id}_{model_type}_RAINFALL_Daily_Average_ByHRU.csv",
        'snowfall': model_dir / "output" / f"{gauge_id}_{model_type}_SNOWFALL_Daily_Average_ByHRU.csv",
        'snowmelt': model_dir / "output" / f"{gauge_id}_{model_type}_TO_LAKE_STORAGE_Daily_Average_ByHRU.csv",
        'icemelt': model_dir / "output" / f"{gauge_id}_{model_type}_FROM_GLACIER_ICE_Daily_Average_BySubbasin.csv"
    }
    
    # Check if files exist
    for name, file_path in files.items():
        if not file_path.exists():
            print(f"ERROR: {name} file not found: {file_path}")
            return None
    
    try:
        # Initialize result dataframe
        result_df = None
        
        # Process HRU-level data (rainfall, snowfall, snowmelt)
        for data_type in ['rainfall', 'snowfall', 'snowmelt']:
            file_path = files[data_type]
            
            # Read header to get HRU ID mapping
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                hru_id_row = next(reader)
                colname_row = next(reader)
            
            # Read the actual data
            df = pd.read_csv(file_path, header=1)
            
            # Create mapping between column names and HRU IDs
            hru_colnames = [col for col in df.columns if col not in ['time', 'date', 'day']]
            hru_id_map = dict(zip(hru_colnames, hru_id_row[2:]))  # skip time/date columns
            
            # Find columns corresponding to glacier HRUs
            glacier_cols = [col for col in hru_colnames if hru_id_map[col] in glacier_hru_ids]
            
            if not glacier_cols:
                print(f"ERROR: No glacier HRU columns found in {data_type} file")
                return None
            
            print(f"  - Found {len(glacier_cols)} glacier HRU columns in {data_type} data")
            
            # Ensure date column is present and parse it
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'day' in df.columns:
                df['date'] = pd.to_datetime(df['day'])
            else:
                print(f"ERROR: {data_type} file must have a 'date' or 'day' column")
                return None
            
            # Filter by date range
            start_date = pd.to_datetime(validation_start)
            end_date = pd.to_datetime(validation_end)
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            if len(df) == 0:
                print(f"ERROR: No {data_type} data found for period {validation_start} to {validation_end}")
                return None
            
            # For snowmelt, calculate daily rates from cumulative values
            if data_type == 'snowmelt':
                for col in glacier_cols:
                    df[col] = df[col].diff().fillna(0)
                    df[col] = df[col].clip(lower=0)  # Ensure no negative melt rates
            
            # Calculate area-weighted average daily values over glacier HRUs
            area_sum = sum([glacier_hru_areas[hru_id_map[col]] for col in glacier_cols])
            df[f'glacier_{data_type}'] = 0.0
            
            for col in glacier_cols:
                hru_id = hru_id_map[col]
                hru_area = glacier_hru_areas[hru_id]
                df[f'glacier_{data_type}'] += df[col] * hru_area
            
            df[f'glacier_{data_type}'] /= area_sum
            
            # Normalize to catchment area
            df[f'glacier_{data_type}_normalized'] = df[f'glacier_{data_type}'] * area_ratio
            
            # Initialize or merge with result dataframe
            if result_df is None:
                result_df = df[['date', f'glacier_{data_type}', f'glacier_{data_type}_normalized']].copy()
            else:
                result_df = pd.merge(result_df, 
                                   df[['date', f'glacier_{data_type}', f'glacier_{data_type}_normalized']], 
                                   on='date', how='inner')
        
        # Process ice melt data (subbasin level)
        icemelt_df = pd.read_csv(files['icemelt'], skiprows=1)
        
        # Ensure date column is present and parse it
        if 'day' in icemelt_df.columns:
            icemelt_df['date'] = pd.to_datetime(icemelt_df['day'])
        elif 'date' in icemelt_df.columns:
            icemelt_df['date'] = pd.to_datetime(icemelt_df['date'])
        else:
            print("ERROR: Ice melt file must have a 'day' or 'date' column")
            return None
        
        # Filter by date range
        icemelt_df = icemelt_df[(icemelt_df['date'] >= start_date) & (icemelt_df['date'] <= end_date)]
        
        if len(icemelt_df) == 0:
            print(f"ERROR: No ice melt data found for period {validation_start} to {validation_end}")
            return None
        
        # Calculate daily melt rates from cumulative values
        if 'mean' not in icemelt_df.columns:
            print("ERROR: Expected 'mean' column not found in ice melt file")
            return None
        
        icemelt_df['glacier_icemelt'] = icemelt_df['mean'].diff().fillna(0)
        icemelt_df['glacier_icemelt'] = icemelt_df['glacier_icemelt'].clip(lower=0)
        
        # Ice melt is already at catchment scale, so normalized version is the same
        icemelt_df['glacier_icemelt_normalized'] = icemelt_df['glacier_icemelt']
        
        # Merge ice melt data
        result_df = pd.merge(result_df, 
                           icemelt_df[['date', 'glacier_icemelt', 'glacier_icemelt_normalized']], 
                           on='date', how='inner')
        
        # Calculate combined metrics
        result_df['glacier_precip'] = result_df['glacier_rainfall'] + result_df['glacier_snowfall']
        result_df['glacier_precip_normalized'] = result_df['glacier_rainfall_normalized'] + result_df['glacier_snowfall_normalized']
        result_df['glacier_total_melt'] = result_df['glacier_snowmelt'] + result_df['glacier_icemelt']
        result_df['glacier_total_melt_normalized'] = result_df['glacier_snowmelt_normalized'] + result_df['glacier_icemelt_normalized']
        
        # Add time columns for analysis
        result_df['month'] = result_df['date'].dt.month
        result_df['year'] = result_df['date'].dt.year
        
        print(f"  - Processed {len(result_df)} daily records")
        print(f"  - Mean glacier rainfall: {result_df['glacier_rainfall'].mean():.2f} mm/day")
        print(f"  - Mean glacier snowfall: {result_df['glacier_snowfall'].mean():.2f} mm/day")
        print(f"  - Mean glacier snowmelt: {result_df['glacier_snowmelt'].mean():.2f} mm/day")
        print(f"  - Mean glacier ice melt: {result_df['glacier_icemelt'].mean():.2f} mm/day")
        
        return result_df
        
    except Exception as e:
        print(f"ERROR: Failed to process glacier contributions data: {e}")
        return None


#--------------------------------------------------------------------------------

def load_nonglacier_contributions_data(config, validation_start=None, validation_end=None):
    """
    Extract all non-glacier contribution data (precipitation and snowmelt) from HBV model output.
    
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
        DataFrame containing all non-glacier contribution data with monthly statistics
    """
    
    config_dir = Path(config['main_dir']) / config['config_dir']
    gauge_id = config['gauge_id']
    model_type = config['model_type']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"Loading non-glacier contributions data for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    
    # Load HRU shapefile to identify non-glacier HRUs
    topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
    catchment_shape_file = topo_dir / "HRU.shp"
    
    if not catchment_shape_file.exists():
        print(f"ERROR: HRU shapefile not found: {catchment_shape_file}")
        return None
    
    try:
        hru_gdf = gpd.read_file(catchment_shape_file)
    except Exception as e:
        print(f"ERROR: Failed to read HRU shapefile: {e}")
        return None
    
    # Calculate total catchment area
    total_catchment_area = hru_gdf['Area_km2'].sum()
    
    # Identify non-glacier HRUs (Landuse_Cl != 7 and != 8)
    nonglacier_hrus = hru_gdf[~hru_gdf['Landuse_Cl'].isin([7, 8])]
    
    if len(nonglacier_hrus) == 0:
        print("WARNING: No non-glacier HRUs found (all HRUs are glacier or snow)")
        return None
    
    nonglacier_hru_ids = nonglacier_hrus['HRU_ID'].astype(str).tolist()
    nonglacier_hru_areas = dict(zip(nonglacier_hrus['HRU_ID'].astype(str), nonglacier_hrus['Area_km2']))
    
    # Calculate non-glacier area and area ratio
    nonglacier_area = nonglacier_hrus['Area_km2'].sum()
    area_ratio = nonglacier_area / total_catchment_area
    
    print(f"  - Total catchment area: {total_catchment_area:.2f} km²")
    print(f"  - Non-glacier area: {nonglacier_area:.2f} km² ({area_ratio:.1%})")
    print(f"  - Number of non-glacier HRUs: {len(nonglacier_hrus)}")
    
    # Print landuse distribution for information
    landuse_distribution = hru_gdf.groupby('Landuse_Cl')['Area_km2'].sum()
    print(f"  - Landuse distribution:")
    for landuse, area in landuse_distribution.items():
        pct = area / total_catchment_area * 100
        landuse_name = {7: 'Glacier', 8: 'Snow/Firn'}.get(landuse, f'Other ({landuse})')
        print(f"    {landuse_name}: {area:.2f} km² ({pct:.1f}%)")
    
    # Define file paths
    model_dir = config_dir / f"catchment_{gauge_id}" / model_type
    files = {
        'rainfall': model_dir / "output" / f"{gauge_id}_{model_type}_RAINFALL_Daily_Average_ByHRU.csv",
        'snowfall': model_dir / "output" / f"{gauge_id}_{model_type}_SNOWFALL_Daily_Average_ByHRU.csv",
        'snowmelt': model_dir / "output" / f"{gauge_id}_{model_type}_TO_LAKE_STORAGE_Daily_Average_ByHRU.csv"
    }
    
    # Check if files exist
    for name, file_path in files.items():
        if not file_path.exists():
            print(f"ERROR: {name} file not found: {file_path}")
            return None
    
    try:
        # Initialize result dataframe
        result_df = None
        
        # Process HRU-level data (rainfall, snowfall, snowmelt)
        for data_type in ['rainfall', 'snowfall', 'snowmelt']:
            file_path = files[data_type]
            
            # Read header to get HRU ID mapping
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                hru_id_row = next(reader)
                colname_row = next(reader)
            
            # Read the actual data
            df = pd.read_csv(file_path, header=1)
            
            # Create mapping between column names and HRU IDs
            hru_colnames = [col for col in df.columns if col not in ['time', 'date', 'day']]
            hru_id_map = dict(zip(hru_colnames, hru_id_row[2:]))  # skip time/date columns
            
            # Find columns corresponding to non-glacier HRUs
            nonglacier_cols = [col for col in hru_colnames if hru_id_map[col] in nonglacier_hru_ids]
            
            if not nonglacier_cols:
                print(f"ERROR: No non-glacier HRU columns found in {data_type} file")
                return None
            
            print(f"  - Found {len(nonglacier_cols)} non-glacier HRU columns in {data_type} data")
            
            # Ensure date column is present and parse it
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'day' in df.columns:
                df['date'] = pd.to_datetime(df['day'])
            else:
                print(f"ERROR: {data_type} file must have a 'date' or 'day' column")
                return None
            
            # Filter by date range
            start_date = pd.to_datetime(validation_start)
            end_date = pd.to_datetime(validation_end)
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            if len(df) == 0:
                print(f"ERROR: No {data_type} data found for period {validation_start} to {validation_end}")
                return None
            
            # For snowmelt, calculate daily rates from cumulative values
            if data_type == 'snowmelt':
                for col in nonglacier_cols:
                    df[col] = df[col].diff().fillna(0)
                    df[col] = df[col].clip(lower=0)  # Ensure no negative melt rates
            
            # Calculate area-weighted average daily values over non-glacier HRUs
            area_sum = sum([nonglacier_hru_areas[hru_id_map[col]] for col in nonglacier_cols])
            df[f'nonglacier_{data_type}'] = 0.0
            
            for col in nonglacier_cols:
                hru_id = hru_id_map[col]
                hru_area = nonglacier_hru_areas[hru_id]
                df[f'nonglacier_{data_type}'] += df[col] * hru_area
            
            df[f'nonglacier_{data_type}'] /= area_sum
            
            # Normalize to catchment area
            df[f'nonglacier_{data_type}_normalized'] = df[f'nonglacier_{data_type}'] * area_ratio
            
            # Initialize or merge with result dataframe
            if result_df is None:
                result_df = df[['date', f'nonglacier_{data_type}', f'nonglacier_{data_type}_normalized']].copy()
            else:
                result_df = pd.merge(result_df, 
                                   df[['date', f'nonglacier_{data_type}', f'nonglacier_{data_type}_normalized']], 
                                   on='date', how='inner')
        
        # Calculate combined metrics
        result_df['nonglacier_precip'] = result_df['nonglacier_rainfall'] + result_df['nonglacier_snowfall']
        result_df['nonglacier_precip_normalized'] = result_df['nonglacier_rainfall_normalized'] + result_df['nonglacier_snowfall_normalized']
        
        # Add time columns for analysis
        result_df['month'] = result_df['date'].dt.month
        result_df['year'] = result_df['date'].dt.year
        
        print(f"  - Processed {len(result_df)} daily records")
        print(f"  - Mean non-glacier rainfall: {result_df['nonglacier_rainfall'].mean():.2f} mm/day")
        print(f"  - Mean non-glacier snowfall: {result_df['nonglacier_snowfall'].mean():.2f} mm/day")
        print(f"  - Mean non-glacier snowmelt: {result_df['nonglacier_snowmelt'].mean():.2f} mm/day")
        
        return result_df
        
    except Exception as e:
        print(f"ERROR: Failed to process non-glacier contributions data: {e}")
        return None


#--------------------------------------------------------------------------------

def plot_glacier_contributions_regime(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot comprehensive glacier contributions analysis with all components.
    
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
    
    # Load glacier contributions data
    glacier_df = load_glacier_contributions_data(config, validation_start, validation_end)
    if glacier_df is None:
        print("No glacier contributions data available for plotting")
        return None
    
    # Calculate monthly regime
    monthly_regime = glacier_df.groupby('month').agg({
        'glacier_rainfall': 'mean',
        'glacier_snowfall': 'mean',
        'glacier_precip': 'mean',
        'glacier_snowmelt': 'mean',
        'glacier_icemelt': 'mean',
        'glacier_total_melt': 'mean',
        'glacier_rainfall_normalized': 'mean',
        'glacier_snowfall_normalized': 'mean',
        'glacier_precip_normalized': 'mean',
        'glacier_snowmelt_normalized': 'mean',
        'glacier_icemelt_normalized': 'mean',
        'glacier_total_melt_normalized': 'mean'
    }).reset_index()
    
    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Monthly Precipitation Regime - Glacier Areas
    ax1.plot(monthly_regime['month'], monthly_regime['glacier_rainfall'], 'g-', 
             label='Rainfall', linewidth=2.5, marker='o')
    ax1.plot(monthly_regime['month'], monthly_regime['glacier_snowfall'], 'c-', 
             label='Snowfall', linewidth=2.5, marker='s')
    ax1.plot(monthly_regime['month'], monthly_regime['glacier_precip'], 'b-', 
             label='Total Precipitation', linewidth=3, marker='D')
    
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Precipitation (mm/day)', fontsize=12)
    ax1.set_title(f'Monthly Precipitation on Glacier Areas\nCatchment {config["gauge_id"]}', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax1.legend(loc='best')
    
    # 2. Monthly Melt Regime - Glacier Areas
    ax2.plot(monthly_regime['month'], monthly_regime['glacier_snowmelt'], 'deepskyblue', 
             label='Snow Melt', linewidth=2.5, marker='o')
    ax2.plot(monthly_regime['month'], monthly_regime['glacier_icemelt'], 'red', 
             label='Ice Melt', linewidth=2.5, marker='s')
    ax2.plot(monthly_regime['month'], monthly_regime['glacier_total_melt'], 'darkred', 
             label='Total Melt', linewidth=3, marker='D')
    
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Melt (mm/day)', fontsize=12)
    ax2.set_title('Monthly Melt from Glacier Areas', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax2.legend(loc='best')
    
    # 3. Monthly Precipitation - Catchment Normalized
    ax3.plot(monthly_regime['month'], monthly_regime['glacier_rainfall_normalized'], 'g--', 
             label='Rainfall', linewidth=2.5, marker='o')
    ax3.plot(monthly_regime['month'], monthly_regime['glacier_snowfall_normalized'], 'c--', 
             label='Snowfall', linewidth=2.5, marker='s')
    ax3.plot(monthly_regime['month'], monthly_regime['glacier_precip_normalized'], 'b--', 
             label='Total Precipitation', linewidth=3, marker='D')
    
    ax3.set_xlabel('Month', fontsize=12)
    ax3.set_ylabel('Precipitation (mm/day)', fontsize=12)
    ax3.set_title('Monthly Precipitation - Catchment Normalized', 
                 fontsize=14, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax3.legend(loc='best')
    
    # 4. Monthly Melt - Catchment Normalized
    ax4.plot(monthly_regime['month'], monthly_regime['glacier_snowmelt_normalized'], 'deepskyblue', 
             linestyle='--', label='Snow Melt', linewidth=2.5, marker='o')
    ax4.plot(monthly_regime['month'], monthly_regime['glacier_icemelt_normalized'], 'red', 
             linestyle='--', label='Ice Melt', linewidth=2.5, marker='s')
    ax4.plot(monthly_regime['month'], monthly_regime['glacier_total_melt_normalized'], 'darkred', 
             linestyle='--', label='Total Melt', linewidth=3, marker='D')
    
    ax4.set_xlabel('Month', fontsize=12)
    ax4.set_ylabel('Melt (mm/day)', fontsize=12)
    ax4.set_title('Monthly Melt - Catchment Normalized', 
                 fontsize=14, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax4.legend(loc='best')
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'glacier_contributions_regime_{config["gauge_id"]}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved glacier contributions regime plot to: {save_path}")
    plt.show()
    
    # Print comprehensive summary statistics
    annual_totals = glacier_df.groupby('year')[['glacier_rainfall', 'glacier_snowfall', 'glacier_precip',
                                               'glacier_snowmelt', 'glacier_icemelt', 'glacier_total_melt']].sum().mean()
    
    print(f"\nGlacier Contributions Analysis Summary for Catchment {config['gauge_id']}:")
    print(f"  Period: {glacier_df['date'].min().date()} to {glacier_df['date'].max().date()}")
    print(f"  Annual averages (glacier areas):")
    print(f"    Rainfall: {annual_totals['glacier_rainfall']:.1f} mm/year")
    print(f"    Snowfall: {annual_totals['glacier_snowfall']:.1f} mm/year")
    print(f"    Total Precipitation: {annual_totals['glacier_precip']:.1f} mm/year")
    print(f"    Snow Melt: {annual_totals['glacier_snowmelt']:.1f} mm/year")
    print(f"    Ice Melt: {annual_totals['glacier_icemelt']:.1f} mm/year")
    print(f"    Total Melt: {annual_totals['glacier_total_melt']:.1f} mm/year")
    
    # Calculate fractions
    snow_fraction = annual_totals['glacier_snowfall'] / annual_totals['glacier_precip'] if annual_totals['glacier_precip'] > 0 else 0
    snowmelt_fraction = annual_totals['glacier_snowmelt'] / annual_totals['glacier_total_melt'] if annual_totals['glacier_total_melt'] > 0 else 0
    
    print(f"  Composition:")
    print(f"    Snow fraction of precipitation: {snow_fraction:.1%}")
    print(f"    Snow melt fraction of total melt: {snowmelt_fraction:.1%}")
    print(f"    Ice melt fraction of total melt: {1-snowmelt_fraction:.1%}")
    
    return fig

#--------------------------------------------------------------------------------

def plot_nonglacier_contributions_regime(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot comprehensive non-glacier contributions analysis with all components.
    
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
    
    # Load non-glacier contributions data
    nonglacier_df = load_nonglacier_contributions_data(config, validation_start, validation_end)
    if nonglacier_df is None:
        print("No non-glacier contributions data available for plotting")
        return None
    
    # Calculate monthly regime
    monthly_regime = nonglacier_df.groupby('month').agg({
        'nonglacier_rainfall': 'mean',
        'nonglacier_snowfall': 'mean',
        'nonglacier_precip': 'mean',
        'nonglacier_snowmelt': 'mean',
        'nonglacier_rainfall_normalized': 'mean',
        'nonglacier_snowfall_normalized': 'mean',
        'nonglacier_precip_normalized': 'mean',
        'nonglacier_snowmelt_normalized': 'mean'
    }).reset_index()
    
    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Monthly Precipitation Regime - Non-Glacier Areas
    ax1.plot(monthly_regime['month'], monthly_regime['nonglacier_rainfall'], 'orange', 
             label='Rainfall', linewidth=2.5, marker='o')
    ax1.plot(monthly_regime['month'], monthly_regime['nonglacier_snowfall'], 'lightblue', 
             label='Snowfall', linewidth=2.5, marker='s')
    ax1.plot(monthly_regime['month'], monthly_regime['nonglacier_precip'], 'purple', 
             label='Total Precipitation', linewidth=3, marker='D')
    
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Precipitation (mm/day)', fontsize=12)
    ax1.set_title(f'Monthly Precipitation on Non-Glacier Areas\nCatchment {config["gauge_id"]}', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax1.legend(loc='best')
    
    # 2. Monthly Snowmelt Regime - Non-Glacier Areas
    ax2.plot(monthly_regime['month'], monthly_regime['nonglacier_snowmelt'], 'lightblue', 
             label='Snow Melt', linewidth=2.5, marker='o')
    
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Snowmelt (mm/day)', fontsize=12)
    ax2.set_title('Monthly Snowmelt from Non-Glacier Areas', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax2.legend(loc='best')
    
    # 3. Monthly Precipitation - Catchment Normalized
    ax3.plot(monthly_regime['month'], monthly_regime['nonglacier_rainfall_normalized'], 'orange', 
             linestyle='--', label='Rainfall', linewidth=2.5, marker='o')
    ax3.plot(monthly_regime['month'], monthly_regime['nonglacier_snowfall_normalized'], 'lightblue', 
             linestyle='--', label='Snowfall', linewidth=2.5, marker='s')
    ax3.plot(monthly_regime['month'], monthly_regime['nonglacier_precip_normalized'], 'purple', 
             linestyle='--', label='Total Precipitation', linewidth=3, marker='D')
    
    ax3.set_xlabel('Month', fontsize=12)
    ax3.set_ylabel('Precipitation (mm/day)', fontsize=12)
    ax3.set_title('Monthly Precipitation - Catchment Normalized', 
                 fontsize=14, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax3.legend(loc='best')
    
    # 4. Monthly Snowmelt - Catchment Normalized
    ax4.plot(monthly_regime['month'], monthly_regime['nonglacier_snowmelt_normalized'], 'lightblue', 
             linestyle='--', label='Snow Melt', linewidth=2.5, marker='o')
    
    ax4.set_xlabel('Month', fontsize=12)
    ax4.set_ylabel('Snowmelt (mm/day)', fontsize=12)
    ax4.set_title('Monthly Snowmelt - Catchment Normalized', 
                 fontsize=14, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax4.legend(loc='best')
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'nonglacier_contributions_regime_{config["gauge_id"]}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved non-glacier contributions regime plot to: {save_path}")
    plt.show()
    
    # Print comprehensive summary statistics
    annual_totals = nonglacier_df.groupby('year')[['nonglacier_rainfall', 'nonglacier_snowfall', 
                                                  'nonglacier_precip', 'nonglacier_snowmelt']].sum().mean()
    
    print(f"\nNon-Glacier Contributions Analysis Summary for Catchment {config['gauge_id']}:")
    print(f"  Period: {nonglacier_df['date'].min().date()} to {nonglacier_df['date'].max().date()}")
    print(f"  Annual averages (non-glacier areas):")
    print(f"    Rainfall: {annual_totals['nonglacier_rainfall']:.1f} mm/year")
    print(f"    Snowfall: {annual_totals['nonglacier_snowfall']:.1f} mm/year")
    print(f"    Total Precipitation: {annual_totals['nonglacier_precip']:.1f} mm/year")
    print(f"    Snow Melt: {annual_totals['nonglacier_snowmelt']:.1f} mm/year")
    
    # Calculate fractions
    snow_fraction = annual_totals['nonglacier_snowfall'] / annual_totals['nonglacier_precip'] if annual_totals['nonglacier_precip'] > 0 else 0
    
    print(f"  Composition:")
    print(f"    Snow fraction of precipitation: {snow_fraction:.1%}")
    print(f"    Rain fraction of precipitation: {1-snow_fraction:.1%}")
    
    return fig

#--------------------------------------------------------------------------------

def create_combined_contributions_dataframes(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Create combined contributions dataframes for glacier and non-glacier areas.
    If coupled=True, uses GloGEM data for glacier contributions.
    If coupled=False, uses HBV model data for glacier contributions.
    Always uses HBV model data for non-glacier contributions.
    
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
    tuple
        (glacier_df, nonglacier_df) - Two DataFrames with combined contributions
    """
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    coupled = config.get('coupled', False)
    gauge_id = config['gauge_id']
    
    print(f"Creating combined contributions dataframes for catchment {gauge_id}:")
    print(f"  - Coupled setting: {coupled}")
    print(f"  - Period: {validation_start} to {validation_end}")
    
    # Create results directory
    config_dir = Path(config['main_dir']) / config['config_dir']
    results_dir = config_dir / f"catchment_{gauge_id}" / config['model_type'] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================
    # GLACIER CONTRIBUTIONS
    # ========================
    
    glacier_df = None
    
    if coupled:
        print("  - Loading glacier contributions from GloGEM data...")
        try:
            # Load GloGEM data
            glogem_df = load_glogem_data(config, unit='mm', plot=False)
            if glogem_df is None:
                print("ERROR: Failed to load GloGEM data")
                return None, None
            
            # Filter by validation period
            start_date = pd.to_datetime(validation_start)
            end_date = pd.to_datetime(validation_end)
            glogem_filtered = glogem_df[(glogem_df['date'] >= start_date) & (glogem_df['date'] <= end_date)].copy()
            
            if len(glogem_filtered) == 0:
                print(f"ERROR: No GloGEM data found for period {validation_start} to {validation_end}")
                return None, None
            
            # Create glacier contributions dataframe with required columns
            glacier_df = pd.DataFrame({
                'date': glogem_filtered['date'],
                'rainfall': glogem_filtered['rainfall'],
                'snowmelt': glogem_filtered['snowmelt'],
                'glaciermelt': glogem_filtered['glacier_melt']
            })
            
            print(f"    ✓ Successfully loaded GloGEM glacier contributions ({len(glacier_df)} records)")
            
        except Exception as e:
            print(f"ERROR: Failed to process GloGEM data: {e}")
            return None, None
    
    else:
        print("  - Loading glacier contributions from HBV model data...")
        try:
            # Load HBV glacier contributions
            hbv_glacier_df = load_glacier_contributions_data(config, validation_start, validation_end)
            if hbv_glacier_df is None:
                print("ERROR: Failed to load HBV glacier contributions data")
                return None, None
            
            # Create glacier contributions dataframe with required columns
            # Note: using normalized values for catchment-scale comparison
            glacier_df = pd.DataFrame({
                'date': hbv_glacier_df['date'],
                'rainfall': hbv_glacier_df['glacier_rainfall_normalized'],
                'snowfall': hbv_glacier_df['glacier_snowfall_normalized'],
                'snowmelt': hbv_glacier_df['glacier_snowmelt_normalized'],
                'glaciermelt': hbv_glacier_df['glacier_icemelt_normalized']
            })
            
            print(f"    ✓ Successfully loaded HBV glacier contributions ({len(glacier_df)} records)")
            
        except Exception as e:
            print(f"ERROR: Failed to process HBV glacier data: {e}")
            return None, None
    
    # ============================
    # NON-GLACIER CONTRIBUTIONS
    # ============================
    
    print("  - Loading non-glacier contributions from HBV model data...")
    try:
        # Load HBV non-glacier contributions
        hbv_nonglacier_df = load_nonglacier_contributions_data(config, validation_start, validation_end)
        if hbv_nonglacier_df is None:
            print("ERROR: Failed to load HBV non-glacier contributions data")
            return glacier_df, None
        
        # Create non-glacier contributions dataframe with required columns
        # Note: using normalized values for catchment-scale comparison
        nonglacier_df = pd.DataFrame({
            'date': hbv_nonglacier_df['date'],
            'rainfall': hbv_nonglacier_df['nonglacier_rainfall_normalized'],
            'snowfall': hbv_nonglacier_df['nonglacier_snowfall_normalized'],
            'snowmelt': hbv_nonglacier_df['nonglacier_snowmelt_normalized'],
            'glaciermelt': 0.0  # Non-glacier areas don't have glacier melt
        })
        
        print(f"    ✓ Successfully loaded HBV non-glacier contributions ({len(nonglacier_df)} records)")
        
    except Exception as e:
        print(f"ERROR: Failed to process HBV non-glacier data: {e}")
        return glacier_df, None
    
    # ============================
    # SAVE DATAFRAMES TO CSV
    # ============================
    
    print("  - Saving dataframes to results directory...")
    
    # Generate filenames based on coupled setting
    if coupled:
        glacier_filename = f"glacier_contributions_glogem_{gauge_id}.csv"
    else:
        glacier_filename = f"glacier_contributions_hbv_{gauge_id}.csv"
    
    nonglacier_filename = f"nonglacier_contributions_hbv_{gauge_id}.csv"
    
    # Save glacier contributions
    if glacier_df is not None:
        glacier_path = results_dir / glacier_filename
        glacier_df.to_csv(glacier_path, index=False)
        print(f"    ✓ Saved glacier contributions to: {glacier_path}")
        
        # Print summary statistics
        print(f"      Glacier contributions summary:")
        print(f"        Mean rainfall: {glacier_df['rainfall'].mean():.3f} mm/day")
        if 'snowfall' in glacier_df.columns:
            print(f"        Mean snowfall: {glacier_df['snowfall'].mean():.3f} mm/day")
        print(f"        Mean snowmelt: {glacier_df['snowmelt'].mean():.3f} mm/day")
        print(f"        Mean glacier melt: {glacier_df['glaciermelt'].mean():.3f} mm/day")
    
    # Save non-glacier contributions
    if nonglacier_df is not None:
        nonglacier_path = results_dir / nonglacier_filename
        nonglacier_df.to_csv(nonglacier_path, index=False)
        print(f"    ✓ Saved non-glacier contributions to: {nonglacier_path}")
        
        # Print summary statistics
        print(f"      Non-glacier contributions summary:")
        print(f"        Mean rainfall: {nonglacier_df['rainfall'].mean():.3f} mm/day")
        print(f"        Mean snowfall: {nonglacier_df['snowfall'].mean():.3f} mm/day")
        print(f"        Mean snowmelt: {nonglacier_df['snowmelt'].mean():.3f} mm/day")
        print(f"        Mean glacier melt: {nonglacier_df['glaciermelt'].mean():.3f} mm/day (should be 0.0)")
    
    # ============================
    # CREATE SUMMARY COMPARISON
    # ============================
    
    if glacier_df is not None and nonglacier_df is not None:
        print(f"\n  Combined Contributions Analysis Summary:")
        print(f"    Period: {validation_start} to {validation_end}")
        print(f"    Data source: {'GloGEM + HBV' if coupled else 'HBV only'}")
        
        # Calculate annual totals
        glacier_annual = glacier_df.groupby(glacier_df['date'].dt.year)[['rainfall', 'snowmelt', 'glaciermelt']].sum().mean()
        nonglacier_annual = nonglacier_df.groupby(nonglacier_df['date'].dt.year)[['rainfall', 'snowfall', 'snowmelt']].sum().mean()
        
        if 'snowfall' in glacier_df.columns:
            glacier_annual_snowfall = glacier_df.groupby(glacier_df['date'].dt.year)['snowfall'].sum().mean()
            glacier_total_precip = glacier_annual['rainfall'] + glacier_annual_snowfall
        else:
            glacier_total_precip = glacier_annual['rainfall']
        
        nonglacier_total_precip = nonglacier_annual['rainfall'] + nonglacier_annual['snowfall']
        
        print(f"    Annual totals (mm/year):")
        print(f"      Glacier areas:")
        print(f"        Total precipitation: {glacier_total_precip:.1f}")
        print(f"        Snowmelt: {glacier_annual['snowmelt']:.1f}")
        print(f"        Glacier melt: {glacier_annual['glaciermelt']:.1f}")
        print(f"      Non-glacier areas:")
        print(f"        Total precipitation: {nonglacier_total_precip:.1f}")
        print(f"        Snowmelt: {nonglacier_annual['snowmelt']:.1f}")
    
    return glacier_df, nonglacier_df

#--------------------------------------------------------------------------------

def plot_combined_contributions_comparison(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Create a comparison plot of glacier vs non-glacier contributions.
    
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
    
    # Create combined dataframes
    glacier_df, nonglacier_df = create_combined_contributions_dataframes(
        config, plot_dirs, validation_start, validation_end
    )
    
    if glacier_df is None or nonglacier_df is None:
        print("Cannot create comparison plot - missing data")
        return None
    
    # Add month column for regime analysis
    glacier_df['month'] = glacier_df['date'].dt.month
    nonglacier_df['month'] = nonglacier_df['date'].dt.month
    
    # Calculate monthly regimes
    glacier_monthly = glacier_df.groupby('month')[['rainfall', 'snowmelt', 'glaciermelt']].mean()
    nonglacier_monthly = nonglacier_df.groupby('month')[['rainfall', 'snowmelt']].mean()
    
    if 'snowfall' in glacier_df.columns:
        glacier_monthly_snowfall = glacier_df.groupby('month')['snowfall'].mean()
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Glacier contributions
    months = range(1, 13)
    month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    
    ax1.plot(months, glacier_monthly['rainfall'], 'g-', label='Rainfall', linewidth=2, marker='o')
    ax1.plot(months, glacier_monthly['snowmelt'], 'b-', label='Snowmelt', linewidth=2, marker='^')
    ax1.plot(months, glacier_monthly['glaciermelt'], 'r-', label='Glacier Melt', linewidth=2, marker='D')
    
    ax1.set_title(f'Glacier Area Contributions\n({"GloGEM" if config.get("coupled", False) else "HBV"})')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Contribution (mm/day)')
    ax1.set_xticks(months)
    ax1.set_xticklabels(month_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Non-glacier contributions
    ax2.plot(months, nonglacier_monthly['rainfall'], 'orange', label='Rainfall', linewidth=2, marker='o')
    ax2.plot(months, nonglacier_monthly['snowmelt'], 'navy', label='Snowmelt', linewidth=2, marker='^')
    
    ax2.set_title('Non-Glacier Area Contributions\n(HBV)')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Contribution (mm/day)')
    ax2.set_xticks(months)
    ax2.set_xticklabels(month_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Catchment Contributions Comparison - Gauge {config["gauge_id"]}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'combined_contributions_comparison_{config["gauge_id"]}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined contributions comparison plot to: {save_path}")
    plt.show()
    
    return fig

#--------------------------------------------------------------------------------

def plot_streamflow_contributions_regime(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot streamflow contributions regime showing observed vs simulated streamflow 
    and the contributions from glacier melt and total snowmelt as filled areas.
    
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
    
    gauge_id = config['gauge_id']
    coupled = config.get('coupled', False)
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    print(f"Creating streamflow contributions regime plot for catchment {gauge_id}:")
    print(f"  - Coupled setting: {coupled}")
    print(f"  - Period: {validation_start} to {validation_end}")
    
    # Load streamflow data
    streamflow_data = load_hydrograph_data(config)
    if streamflow_data is None:
        print("ERROR: Could not load streamflow data")
        return None
    
    # Filter streamflow data for validation period
    validation_start_dt = pd.to_datetime(validation_start)
    validation_end_dt = pd.to_datetime(validation_end)
    
    streamflow_mask = (streamflow_data['date'] >= validation_start_dt) & (streamflow_data['date'] <= validation_end_dt)
    streamflow_filtered = streamflow_data[streamflow_mask].copy()
    
    if len(streamflow_filtered) == 0:
        print(f"ERROR: No streamflow data found for period {validation_start} to {validation_end}")
        return None
    
    # Load saved contribution dataframes
    config_dir = Path(config['main_dir']) / config['config_dir']
    results_dir = config_dir / f"catchment_{gauge_id}" / config['model_type'] / "results"
    
    # Determine glacier contributions filename based on coupled setting
    if coupled:
        glacier_filename = f"glacier_contributions_glogem_{gauge_id}.csv"
    else:
        glacier_filename = f"glacier_contributions_hbv_{gauge_id}.csv"
    
    nonglacier_filename = f"nonglacier_contributions_hbv_{gauge_id}.csv"
    
    glacier_file = results_dir / glacier_filename
    nonglacier_file = results_dir / nonglacier_filename
    
    # Check if contribution files exist
    if not glacier_file.exists():
        print(f"ERROR: Glacier contributions file not found: {glacier_file}")
        print("Please run create_combined_contributions_dataframes() first")
        return None
    
    if not nonglacier_file.exists():
        print(f"ERROR: Non-glacier contributions file not found: {nonglacier_file}")
        print("Please run create_combined_contributions_dataframes() first")
        return None
    
    try:
        # Load contribution dataframes
        glacier_df = pd.read_csv(glacier_file, parse_dates=['date'])
        nonglacier_df = pd.read_csv(nonglacier_file, parse_dates=['date'])
        
        print(f"  - Loaded glacier contributions: {len(glacier_df)} records")
        print(f"  - Loaded non-glacier contributions: {len(nonglacier_df)} records")
        
        # Filter contribution data for validation period
        glacier_mask = (glacier_df['date'] >= validation_start_dt) & (glacier_df['date'] <= validation_end_dt)
        nonglacier_mask = (nonglacier_df['date'] >= validation_start_dt) & (nonglacier_df['date'] <= validation_end_dt)
        
        glacier_filtered = glacier_df[glacier_mask].copy()
        nonglacier_filtered = nonglacier_df[nonglacier_mask].copy()
        
        if len(glacier_filtered) == 0 or len(nonglacier_filtered) == 0:
            print(f"ERROR: No contribution data found for validation period")
            return None
        
        # Add month column for regime calculation
        streamflow_filtered['month'] = streamflow_filtered['date'].dt.month
        glacier_filtered['month'] = glacier_filtered['date'].dt.month
        nonglacier_filtered['month'] = nonglacier_filtered['date'].dt.month
        
        # Calculate monthly regimes for streamflow
        streamflow_monthly = streamflow_filtered.groupby('month').agg({
            'obs_Q': 'mean',
            'sim_Q': 'mean'
        }).reset_index()
        
        # Calculate monthly regimes for contributions
        glacier_monthly = glacier_filtered.groupby('month').agg({
            'glaciermelt': 'mean',
            'snowmelt': 'mean'
        }).reset_index()
        
        nonglacier_monthly = nonglacier_filtered.groupby('month').agg({
            'snowmelt': 'mean'
        }).reset_index()
        
        # Combine snowmelt from glacier and non-glacier areas
        # Merge on month to ensure alignment
        combined_monthly = pd.merge(glacier_monthly, nonglacier_monthly, on='month', suffixes=('_glacier', '_nonglacier'))
        combined_monthly['total_snowmelt'] = combined_monthly['snowmelt_glacier'] + combined_monthly['snowmelt_nonglacier']
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        months = range(1, 13)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Convert contributions from mm/day to m³/s
        # Load catchment area for conversion
        topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
        catchment_shape_file = topo_dir / "HRU.shp"
        
        conversion_factor = 1.0  # Default to 1 if we can't get area
        
        if catchment_shape_file.exists():
            try:
                import geopandas as gpd
                hru_gdf = gpd.read_file(catchment_shape_file)
                total_area_km2 = hru_gdf['Area_km2'].sum()
                # Convert mm/day to m³/s: mm/day * km² * 1000 m²/km² * 1000 mm/m / 86400 s/day
                conversion_factor = total_area_km2 * 1000 * 1000 / 86400 / 1000
                print(f"  - Catchment area: {total_area_km2:.2f} km²")
                print(f"  - Conversion factor (mm/day to m³/s): {conversion_factor:.3f}")
            except Exception as e:
                print(f"  - Warning: Could not load catchment area for unit conversion: {e}")
                print(f"  - Contributions will be plotted in mm/day")
        
        # Convert contributions
        glacier_melt_converted = combined_monthly['glaciermelt'] * conversion_factor
        total_snowmelt_converted = combined_monthly['total_snowmelt'] * conversion_factor
        
        # Plot contributions as filled areas (polygons)
        # Plot total snowmelt first (bottom layer) as filled area
        plt.fill_between(combined_monthly['month'], 0, total_snowmelt_converted, 
                        color='lightblue', alpha=0.7, label='Total Snowmelt', zorder=1)
        
        # Plot glacier melt as filled area on top
        plt.fill_between(combined_monthly['month'], 0, glacier_melt_converted, 
                        color='grey', alpha=0.8, label='Glacier Melt', zorder=2)
        
        # Plot observed streamflow (black line)
        if 'obs_Q' in streamflow_monthly.columns:
            plt.plot(streamflow_monthly['month'], streamflow_monthly['obs_Q'], 
                    'k-', linewidth=3, label='Observed Streamflow', zorder=4)
        
        # Plot simulated streamflow (colored line)
        if 'sim_Q' in streamflow_monthly.columns:
            plt.plot(streamflow_monthly['month'], streamflow_monthly['sim_Q'], 
                    'C0', linewidth=2.5, label='Simulated Streamflow', zorder=3)
        
        # Formatting
        plt.xlabel('Month', fontsize=14)
        if conversion_factor != 1.0:
            plt.ylabel('Discharge (m³/s)', fontsize=14)
        else:
            plt.ylabel('Discharge (m³/s) / Contributions (mm/day)', fontsize=14)
        
        plt.title(f'Streamflow and Contributions Regime - Catchment {gauge_id}\n'
                 f'Validation Period: {validation_start} to {validation_end} '
                 f'({"GloGEM+HBV" if coupled else "HBV"})', 
                 fontsize=16, fontweight='bold')
        
        plt.xticks(months, month_names)
        plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        save_path = plot_dirs['contributions'] / f'streamflow_contributions_regime_{gauge_id}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved streamflow contributions regime plot to: {save_path}")
        plt.show()
        
        # Print summary statistics
        print(f"\nStreamflow Contributions Analysis Summary:")
        if 'obs_Q' in streamflow_monthly.columns and 'sim_Q' in streamflow_monthly.columns:
            mean_obs = streamflow_monthly['obs_Q'].mean()
            mean_sim = streamflow_monthly['sim_Q'].mean()
            print(f"  Mean observed streamflow: {mean_obs:.2f} m³/s")
            print(f"  Mean simulated streamflow: {mean_sim:.2f} m³/s")
        
        mean_glacier_melt = glacier_melt_converted.mean()
        mean_total_snowmelt = total_snowmelt_converted.mean()
        
        if conversion_factor != 1.0:
            print(f"  Mean glacier melt contribution: {mean_glacier_melt:.2f} m³/s")
            print(f"  Mean total snowmelt contribution: {mean_total_snowmelt:.2f} m³/s")
            
            # Calculate percentages of streamflow
            if 'sim_Q' in streamflow_monthly.columns:
                mean_sim = streamflow_monthly['sim_Q'].mean()
                if mean_sim > 0:
                    glacier_pct = (mean_glacier_melt / mean_sim) * 100
                    snowmelt_pct = (mean_total_snowmelt / mean_sim) * 100
                    print(f"  Glacier melt as % of simulated flow: {glacier_pct:.1f}%")
                    print(f"  Total snowmelt as % of simulated flow: {snowmelt_pct:.1f}%")
        else:
            print(f"  Mean glacier melt contribution: {mean_glacier_melt:.3f} mm/day")
            print(f"  Mean total snowmelt contribution: {mean_total_snowmelt:.3f} mm/day")
        
        return plt.gcf()
        
    except Exception as e:
        print(f"ERROR: Failed to process contribution data: {e}")
        return None

#--------------------------------------------------------------------------------



#--------------------------------------------------------------------------------

def plot_average_yearly_water_balance(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot average yearly water balance components: precipitation, observed streamflow, simulated streamflow,
    and a stacked bar of simulated streamflow components (rainfall, snowmelt, glacier melt).
    """
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    
    gauge_id = config['gauge_id']
    coupled = config.get('coupled', False)
    
    print(f"Creating average yearly water balance plot for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")
    print(f"  - Coupled mode: {coupled}")
    
    # Load streamflow data
    print("  - Loading streamflow data...")
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
        print("ERROR: No streamflow data in validation period")
        return None
    
    # Check for existing contribution dataframes
    config_dir = Path(config['main_dir']) / config['config_dir']
    results_dir = config_dir / f"catchment_{gauge_id}" / config['model_type'] / "results"
    
    if coupled:
        glacier_filename = f"glacier_contributions_glogem_{gauge_id}.csv"
    else:
        glacier_filename = f"glacier_contributions_hbv_{gauge_id}.csv"
    
    nonglacier_filename = f"nonglacier_contributions_hbv_{gauge_id}.csv"
    
    glacier_file = results_dir / glacier_filename
    nonglacier_file = results_dir / nonglacier_filename
    
    # If contribution files don't exist, create them automatically
    if not glacier_file.exists() or not nonglacier_file.exists():
        print("  - Contribution files not found. Creating them automatically...")
        try:
            glacier_df, nonglacier_df = create_combined_contributions_dataframes(
                config, plot_dirs, validation_start, validation_end
            )
            if glacier_df is None or nonglacier_df is None:
                print("ERROR: Failed to create contribution dataframes")
                return None
            print("  - ✓ Contribution files created successfully")
        except Exception as e:
            print(f"ERROR: Failed to create contribution files: {e}")
            return None
    
    try:
        # Load contribution dataframes
        glacier_df = pd.read_csv(glacier_file, parse_dates=['date'])
        nonglacier_df = pd.read_csv(nonglacier_file, parse_dates=['date'])
        
        # Filter by validation period
        glacier_mask = (glacier_df['date'] >= start_date) & (glacier_df['date'] <= end_date)
        nonglacier_mask = (nonglacier_df['date'] >= start_date) & (nonglacier_df['date'] <= end_date)
        
        glacier_filtered = glacier_df[glacier_mask].copy()
        nonglacier_filtered = nonglacier_df[nonglacier_mask].copy()
        
        if len(glacier_filtered) == 0 or len(nonglacier_filtered) == 0:
            print("ERROR: No contribution data found for validation period")
            return None
        
        print(f"    ✓ Loaded contribution data successfully")
        
    except Exception as e:
        print(f"ERROR: Failed to load contribution data: {e}")
        return None
    
    # Convert streamflow to mm/year if possible
    try:
        topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
        catchment_shape_file = topo_dir / "HRU.shp"
        
        if catchment_shape_file.exists():
            import geopandas as gpd
            hru_gdf = gpd.read_file(catchment_shape_file)
            total_area_km2 = hru_gdf['Area_km2'].sum()
            
            # Convert m³/s to mm/year
            conversion_factor = 86400 * 365.25 / (total_area_km2 * 1000000) * 1000
            
            print(f"    ✓ Catchment area: {total_area_km2:.2f} km²")
        else:
            print("    Warning: Could not load catchment area")
            conversion_factor = 365.25  # Rough conversion
            
    except Exception as e:
        print(f"    Warning: Error with catchment area: {e}")
        conversion_factor = 365.25
    
    # Calculate yearly averages
    streamflow_filtered['year'] = streamflow_filtered['date'].dt.year
    glacier_filtered['year'] = glacier_filtered['date'].dt.year
    nonglacier_filtered['year'] = nonglacier_filtered['date'].dt.year
    
    # Average streamflow (convert to mm/year)
    streamflow_yearly = streamflow_filtered.groupby('year').agg({
        'obs_Q': 'mean',
        'sim_Q': 'mean'
    }).mean()
    
    obs_streamflow = streamflow_yearly['obs_Q'] * conversion_factor if 'obs_Q' in streamflow_yearly else None
    sim_streamflow = streamflow_yearly['sim_Q'] * conversion_factor if 'sim_Q' in streamflow_yearly else None
    
    # --- Build aggregation dict dynamically ---
    glacier_agg = {'rainfall': 'sum', 'snowmelt': 'sum', 'glaciermelt': 'sum'}
    if 'snowfall' in glacier_filtered.columns:
        glacier_agg['snowfall'] = 'sum'
    glacier_yearly = glacier_filtered.groupby('year').agg(glacier_agg).mean()
    
    nonglacier_agg = {'rainfall': 'sum', 'snowmelt': 'sum'}
    if 'snowfall' in nonglacier_filtered.columns:
        nonglacier_agg['snowfall'] = 'sum'
    nonglacier_yearly = nonglacier_filtered.groupby('year').agg(nonglacier_agg).mean()
    
    # Calculate combined values
    total_precipitation = (
        glacier_yearly.get('rainfall', 0) +
        glacier_yearly.get('snowfall', 0) +
        nonglacier_yearly.get('rainfall', 0) +
        nonglacier_yearly.get('snowfall', 0)
    )
    total_snowmelt = glacier_yearly.get('snowmelt', 0) + nonglacier_yearly.get('snowmelt', 0)
    glacier_melt = glacier_yearly.get('glaciermelt', 0)
    total_rainfall = glacier_yearly.get('rainfall', 0) + nonglacier_yearly.get('rainfall', 0)
    
    # Prepare data for plotting
    bar_labels = ['Precipitation', 'Observed\nStreamflow', 'Simulated\nStreamflow', 'Simulated\nComponents']
    bar_positions = np.arange(len(bar_labels))
    
    # Stacked bar components for simulated streamflow
    stack_rain = total_rainfall
    stack_snowmelt = total_snowmelt
    stack_glacier = glacier_melt
    stack_total = stack_rain + stack_snowmelt + stack_glacier
    
    # Main bars
    bar_values = [
        total_precipitation,
        obs_streamflow,
        sim_streamflow,
        stack_total
    ]
    
    # Colors for bars and stacks
    bar_colors = ['darkblue', 'black', 'orange', None]  # Last is stacked
    stack_colors = ['forestgreen', 'deepskyblue', 'grey']  # Rain, snowmelt, glacier melt
    
    plt.figure(figsize=(10, 8))
    
    # Plot main bars
    plt.bar(bar_positions[:3], bar_values[:3], color=bar_colors[:3], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Plot stacked bar for simulated components
    plt.bar(bar_positions[3], stack_rain, color=stack_colors[0], edgecolor='black', linewidth=1.5, label='Rainfall')
    plt.bar(bar_positions[3], stack_snowmelt, bottom=stack_rain, color=stack_colors[1], edgecolor='black', linewidth=1.5, label='Snowmelt')
    plt.bar(bar_positions[3], stack_glacier, bottom=stack_rain + stack_snowmelt, color=stack_colors[2], edgecolor='black', linewidth=1.5, label='Glacier Melt')
    
    # Add value labels on bars
    for i in range(3):
        plt.text(bar_positions[i], bar_values[i] + max(bar_values)*0.01,
                 f'{bar_values[i]:.0f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # Add value and percentage label inside stacked bar for snowmelt and glacier melt
    if sim_streamflow and stack_total:
        snowmelt_perc = stack_snowmelt / sim_streamflow * 100 if sim_streamflow > 0 else 0
        glacier_perc = stack_glacier / sim_streamflow * 100 if sim_streamflow > 0 else 0
        label_text = (f'Snowmelt: {stack_snowmelt:.0f} mm\n({snowmelt_perc:.1f}%)\n'
                      f'Glacier: {stack_glacier:.0f} mm\n({glacier_perc:.1f}%)')
        plt.text(bar_positions[3], stack_total/2, 
                 label_text, 
                 ha='center', va='center', fontsize=13, fontweight='bold', color='black')
    
    # Formatting
    plt.xticks(bar_positions, bar_labels, fontsize=15)
    plt.ylabel('Annual Average (mm/year)', fontsize=15)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(['Rainfall', 'Snowmelt', 'Glacier Melt'], fontsize=13, loc='upper right')
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'average_yearly_water_balance_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved average yearly water balance plot to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\nAverage Yearly Water Balance Summary for Catchment {gauge_id}:")
    print(f"  Period: {validation_start} to {validation_end}")
    print(f"  Mode: {'Coupled (GloGEM+HBV)' if coupled else 'Uncoupled (HBV only)'}")
    print(f"  Precipitation: {total_precipitation:.1f} mm/year")
    print(f"  Observed streamflow: {obs_streamflow:.1f} mm/year")
    print(f"  Simulated streamflow: {sim_streamflow:.1f} mm/year")
    print(f"  Simulated components (rain/snowmelt/glacier): {stack_rain:.1f} / {stack_snowmelt:.1f} / {stack_glacier:.1f} mm/year")
    print(f"  Snowmelt: {stack_snowmelt:.1f} mm/year ({snowmelt_perc:.1f}% of simulated streamflow)")
    print(f"  Glacier melt: {stack_glacier:.1f} mm/year ({glacier_perc:.1f}% of simulated streamflow)")
    
    return {
        'precipitation': total_precipitation,
        'obs_streamflow': obs_streamflow,
        'sim_streamflow': sim_streamflow,
        'rainfall': stack_rain,
        'snowmelt_total': stack_snowmelt,
        'glacier_melt': stack_glacier,
        'sim_components_total': stack_total,
        'snowmelt_pct': snowmelt_perc,
        'glacier_pct': glacier_perc
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
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from namelist
    plot_dirs : dict
        Dictionary containing plot directory paths
    template_dir : str or Path, optional
        Directory containing template files (will use config if not provided)
    raven_exe : str or Path, optional
        Path to Raven executable (will use config if not provided)
    n_runs : int
        Number of best runs to process (default: 100)
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    color_best : str
        Color for the best simulation
    color_others : str
        Color for other simulations
    """
    
    import subprocess
    import shutil
    
    gauge_id = config['gauge_id']
    model_type = config['model_type']
    config_dir = Path(config['main_dir']) / config['config_dir']
    
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
    
    # Get Raven executable from config (note: key is 'raven_executable' not 'raven_exe')
    if raven_exe is None:
        raven_exe = config.get('raven_executable', '/path/to/raven.exe')
        if raven_exe == '/path/to/raven.exe':
            print("Warning: Using default Raven executable path. Please check your namelist.")

    print(f"Running regime analysis for {n_runs} best runs:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Model: {model_type}")
    print(f"  - Validation period: {validation_start} to {validation_end}")
    print(f"  - Template directory: {template_dir}")
    print(f"  - Raven executable: {raven_exe}")
    
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
        
    df = pd.read_csv(results_file)
    if 'like1' not in df.columns:
        print("Error: 'like1' column not found in results file.")
        print(f"Available columns: {df.columns.tolist()}")
        return None

    print(f"Loaded {len(df)} parameter sets from results file")

    # 2. Convert negative KGE to positive KGE
    df['KGE'] = -df['like1']

    # 3. Select best runs
    best_runs = df.sort_values('KGE', ascending=False).head(n_runs)
    param_cols = [col for col in df.columns if col not in ['like1', 'KGE']]
    
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

    # 6. Process all runs directly
    print(f"\nProcessing all {n_runs} runs...")
    hydrographs = []
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
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                hydro_file = run_output_dir / f"{gauge_id}_{model_type}_Hydrographs.csv"
                if hydro_file.exists():
                    df_hydro = pd.read_csv(hydro_file)
                    df_hydro['date'] = pd.to_datetime(df_hydro['date'])
                    mask = (df_hydro['date'] >= validation_start) & (df_hydro['date'] <= validation_end)
                    monthly = df_hydro[mask].copy()
                    monthly['month'] = monthly['date'].dt.month
                    
                    sim_col = None
                    for col in df_hydro.columns:
                        if '[m3/s]' in col and 'observed' not in col.lower():
                            sim_col = col
                            break
                    
                    if sim_col:
                        monthly_mean = monthly.groupby('month')[sim_col].mean()
                        hydrographs.append(monthly_mean)
                        successful_runs += 1
                        
                        # Clean up run directory
                        cleanup_raven_run_directory(run_dir)
                    else:
                        failed_runs += 1
                else:
                    failed_runs += 1
            else:
                failed_runs += 1
                        
        except Exception as e:
            failed_runs += 1
            continue

    print(f"Successfully processed {successful_runs} out of {n_runs} runs ({failed_runs} failed)")
    
    if len(hydrographs) == 0:
        print("No hydrographs generated. Cannot create plot.")
        return None

    # 7. Load observed data and create final plot
    obs_data = load_hydrograph_data(config)
    if obs_data is None:
        print("Error: Could not load observed hydrograph data")
        return None
    
    mask = (obs_data['date'] >= validation_start) & (obs_data['date'] <= validation_end)
    obs_monthly = obs_data[mask].copy()
    obs_monthly['month'] = obs_monthly['date'].dt.month
    obs_mean = obs_monthly.groupby('month')['obs_Q'].mean()

    # 8. Create final plot
    plt.figure(figsize=(12, 7))
    
    # Plot observed data first
    plt.plot(obs_mean.index, obs_mean.values, 'k-', linewidth=2.5, label='Observed')
    
    # Plot all simulations except the best one (in grey)
    for i, monthly_mean in enumerate(hydrographs[1:], 1):  # Skip first (best) simulation
        plt.plot(monthly_mean.index, monthly_mean.values, color=color_others, 
                linewidth=1, alpha=0.5, zorder=1)
    
    # Plot the best simulation on top (highest zorder)
    if len(hydrographs) > 0:
        plt.plot(hydrographs[0].index, hydrographs[0].values, color=color_best, 
                linewidth=2, label='Best Simulation', zorder=3)
    
    # Add a single grey line to legend
    if len(hydrographs) > 1:
        plt.plot([], [], color=color_others, linewidth=1, alpha=0.5, 
                label=f'Other {len(hydrographs)-1} Simulations')
    
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Discharge (m³/s)', fontsize=14)
    plt.title(f'Regime for {len(hydrographs)} Best Runs - Catchment {gauge_id}', fontsize=16)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['hydrographs'] / f'regime_{len(hydrographs)}_best_runs_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved regime plot to: {save_path}")
    plt.show()
    
    print(f"\nRegime Analysis Summary for Catchment {gauge_id}:")
    print(f"  Successfully processed: {len(hydrographs)}/{n_runs} runs")
    print(f"  Validation period: {validation_start} to {validation_end}")
    print(f"  Best KGE: {best_runs['KGE'].iloc[0]:.4f}")
    print(f"  Worst KGE in selection: {best_runs['KGE'].iloc[-1]:.4f}")
    
    return hydrographs

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
    
    try:
        print("   6.1 Combined Contributions Dataframes...")
        glacier_df, nonglacier_df = create_combined_contributions_dataframes(config, plot_dirs, validation_start, validation_end)
        results['contributions_dataframes'] = {'glacier': glacier_df, 'nonglacier': nonglacier_df}
        results['success']['contributions_dataframes'] = True
        print("       ✓ Contributions dataframes created")
    except Exception as e:
        print(f"       ✗ Error creating contributions dataframes: {e}")
        results['errors']['contributions_dataframes'] = str(e)
        results['success']['contributions_dataframes'] = False
    
    try:
        print("   6.2 Combined Contributions Comparison...")
        combined_contrib_result = plot_combined_contributions_comparison(config, plot_dirs, validation_start, validation_end)
        results['combined_contributions'] = combined_contrib_result
        results['success']['combined_contributions'] = True
        print("       ✓ Combined contributions comparison plotted")
    except Exception as e:
        print(f"       ✗ Error with combined contributions: {e}")
        results['errors']['combined_contributions'] = str(e)
        results['success']['combined_contributions'] = False
    
    try:
        print("   6.3 Streamflow Contributions Regime...")
        streamflow_contrib_result = plot_streamflow_contributions_regime(config, plot_dirs, validation_start, validation_end)
        results['streamflow_contributions'] = streamflow_contrib_result
        results['success']['streamflow_contributions'] = True
        print("       ✓ Streamflow contributions regime plotted")
    except Exception as e:
        print(f"       ✗ Error with streamflow contributions: {e}")
        results['errors']['streamflow_contributions'] = str(e)
        results['success']['streamflow_contributions'] = False
    
    # Only run detailed contributions analysis if we have the required data
    if coupled:
        try:
            print("   6.4 GloGEM Regime Analysis...")
            glogem_result = plot_glogem_regime(config, plot_dirs, unit='mm')
            results['glogem_regime'] = glogem_result
            results['success']['glogem_regime'] = True
            print("       ✓ GloGEM regime analysis completed")
        except Exception as e:
            print(f"       ✗ Error with GloGEM analysis: {e}")
            results['errors']['glogem_regime'] = str(e)
            results['success']['glogem_regime'] = False
    else:
        print("   6.4 GloGEM Analysis skipped (uncoupled mode)")
        results['success']['glogem_regime'] = None
    
    try:
        print("   6.5 Glacier Contributions Regime...")
        glacier_contrib_result = plot_glacier_contributions_regime(config, plot_dirs, validation_start, validation_end)
        results['glacier_contributions'] = glacier_contrib_result
        results['success']['glacier_contributions'] = True
        print("       ✓ Glacier contributions regime plotted")
    except Exception as e:
        print(f"       ✗ Error with glacier contributions: {e}")
        results['errors']['glacier_contributions'] = str(e)
        results['success']['glacier_contributions'] = False
    
    try:
        print("   6.6 Non-Glacier Contributions Regime...")
        nonglacier_contrib_result = plot_nonglacier_contributions_regime(config, plot_dirs, validation_start, validation_end)
        results['nonglacier_contributions'] = nonglacier_contrib_result
        results['success']['nonglacier_contributions'] = True
        print("       ✓ Non-glacier contributions regime plotted")
    except Exception as e:
        print(f"       ✗ Error with non-glacier contributions: {e}")
        results['errors']['nonglacier_contributions'] = str(e)
        results['success']['nonglacier_contributions'] = False
    
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
################################ water balance #################################
#--------------------------------------------------------------------------------


def plot_yearly_precipitation_streamflow(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot yearly summed precipitation vs yearly summed observed and simulated streamflow.
    """
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = config.get('end_date', '2020-12-31')
    gauge_id = config['gauge_id']

    print(f"Creating yearly precipitation vs streamflow plot for catchment {gauge_id}:")
    print(f"  - Period: {validation_start} to {validation_end}")

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

    # Calculate yearly sums
    df['year'] = df['date'].dt.year
    yearly = df.groupby('year').agg({
        'precip': 'sum',
        'obs_Q_mm': 'sum',
        'sim_Q_mm': 'sum'
    }).reset_index()

    if len(yearly) == 0:
        print("ERROR: No yearly data found")
        return None

    print(f"  - Found {len(yearly)} years of complete data")

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Precipitation vs Observed Streamflow
    ax1.scatter(yearly['precip'], yearly['obs_Q_mm'], color='darkblue', s=80, alpha=0.7, edgecolors='black', linewidth=1)
    min_val = min(yearly['precip'].min(), yearly['obs_Q_mm'].min())
    max_val = max(yearly['precip'].max(), yearly['obs_Q_mm'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='1:1 line')
    z = np.polyfit(yearly['precip'], yearly['obs_Q_mm'], 1)
    p = np.poly1d(z)
    ax1.plot(yearly['precip'], p(yearly['precip']), 'r-', alpha=0.8, linewidth=2)
    corr_obs = np.corrcoef(yearly['precip'], yearly['obs_Q_mm'])[0, 1]
    ax1.set_xlabel('Annual Precipitation (mm)', fontsize=12)
    ax1.set_ylabel('Annual Observed Streamflow (mm)' if conversion_factor else 'Annual Observed Streamflow (m³/s)', fontsize=12)
    ax1.set_title(f'Precipitation vs Observed Streamflow\nR = {corr_obs:.3f}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right plot: Precipitation vs Simulated Streamflow
    ax2.scatter(yearly['precip'], yearly['sim_Q_mm'], color='orange', s=80, alpha=0.7, edgecolors='black', linewidth=1)
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='1:1 line')
    z = np.polyfit(yearly['precip'], yearly['sim_Q_mm'], 1)
    p = np.poly1d(z)
    ax2.plot(yearly['precip'], p(yearly['precip']), 'r-', alpha=0.8, linewidth=2)
    corr_sim = np.corrcoef(yearly['precip'], yearly['sim_Q_mm'])[0, 1]
    ax2.set_xlabel('Annual Precipitation (mm)', fontsize=12)
    ax2.set_ylabel('Annual Simulated Streamflow (mm)' if conversion_factor else 'Annual Simulated Streamflow (m³/s)', fontsize=12)
    ax2.set_title(f'Precipitation vs Simulated Streamflow\nR = {corr_sim:.3f}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(f'Annual Precipitation vs Streamflow - Catchment {gauge_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save plot
    save_path = plot_dirs['contributions'] / f'yearly_precipitation_streamflow_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved yearly precipitation vs streamflow plot to: {save_path}")
    plt.show()

    # Print summary statistics
    print(f"\nYearly Precipitation vs Streamflow Analysis:")
    print(f"  Period: {yearly['year'].min()} - {yearly['year'].max()}")
    print(f"  Number of years: {len(yearly)}")
    print(f"  Mean annual precipitation: {yearly['precip'].mean():.1f} mm")
    print(f"  Mean annual observed streamflow: {yearly['obs_Q_mm'].mean():.1f} {'mm' if conversion_factor else 'm³/s'}")
    print(f"  Mean annual simulated streamflow: {yearly['sim_Q_mm'].mean():.1f} {'mm' if conversion_factor else 'm³/s'}")
    print(f"  Correlation (precip vs obs): {corr_obs:.3f}")
    print(f"  Correlation (precip vs sim): {corr_sim:.3f}")

    if conversion_factor:
        obs_ratio = yearly['obs_Q_mm'].mean() / yearly['precip'].mean()
        sim_ratio = yearly['sim_Q_mm'].mean() / yearly['precip'].mean()
        print(f"  Observed runoff ratio: {obs_ratio:.3f}")
        print(f"  Simulated runoff ratio: {sim_ratio:.3f}")

    return yearly

    #--------------------------------------------------------------------------------

def plot_precipitation_glacier_melt_vs_streamflow_scatter(config, plot_dirs, validation_start=None, validation_end=None):
    """
    Create a scatter plot comparing (precipitation + glacier melt) vs observed streamflow for each year.
    
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
    
    print(f"Creating precipitation + glacier melt vs streamflow scatter plot for catchment {gauge_id}:")
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
    
    # Load glacier contributions data
    glacier_df = load_glacier_contributions_data(config, validation_start, validation_end)
    if glacier_df is None:
        print("ERROR: Could not load glacier contributions data")
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
    
    # Yearly sums for streamflow and precipitation
    streamflow_yearly = streamflow_filtered.groupby('year').agg({
        'obs_Q_mm': 'sum',
        'sim_Q_mm': 'sum',
        'precip': 'sum'
    }).reset_index()
    
    # Yearly sums for glacier melt (normalized to catchment scale)
    glacier_yearly = glacier_df.groupby('year')['glacier_icemelt_normalized'].sum().reset_index()
    glacier_yearly.columns = ['year', 'glacier_melt']
    
    # Merge data
    yearly_data = pd.merge(streamflow_yearly, glacier_yearly, on='year', how='inner')
    
    if len(yearly_data) == 0:
        print("ERROR: No overlapping years found between datasets")
        return None
    
    # Calculate input (precipitation + glacier melt)
    yearly_data['input_total'] = yearly_data['precip'] + yearly_data['glacier_melt']
    
    print(f"  - Found {len(yearly_data)} years of complete data")
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    plt.scatter(yearly_data['input_total'], yearly_data['obs_Q_mm'], 
               color='darkblue', s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add 1:1 line
    min_val = min(yearly_data['input_total'].min(), yearly_data['obs_Q_mm'].min()) * 0.95
    max_val = max(yearly_data['input_total'].max(), yearly_data['obs_Q_mm'].max()) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2, label='1:1 line')
    
    # Add trend line
    z = np.polyfit(yearly_data['input_total'], yearly_data['obs_Q_mm'], 1)
    p = np.poly1d(z)
    plt.plot(yearly_data['input_total'], p(yearly_data['input_total']), 'r-', alpha=0.8, linewidth=2, label='Trend line')
    
    # Calculate correlation
    corr = np.corrcoef(yearly_data['input_total'], yearly_data['obs_Q_mm'])[0, 1]
    
    # Add year labels to points
    for _, row in yearly_data.iterrows():
        plt.annotate(str(int(row['year'])), (row['input_total'], row['obs_Q_mm']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
    
    plt.xlabel('Annual Precipitation + Glacier Melt (mm/year)', fontsize=14, fontweight='bold')
    plt.ylabel('Annual Observed Streamflow (mm/year)', fontsize=14, fontweight='bold')
    plt.title(f'Input vs Output Water Balance - Catchment {gauge_id}\n'
             f'R = {corr:.3f} | Period: {validation_start} to {validation_end}', 
             fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set equal aspect ratio and limits
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'precipitation_glacier_melt_vs_streamflow_scatter_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot to: {save_path}")
    plt.show()
    
    # Print summary statistics
    mean_input = yearly_data['input_total'].mean()
    mean_output = yearly_data['obs_Q_mm'].mean()
    runoff_ratio = mean_output / mean_input
    
    print(f"\nWater Balance Analysis Summary:")
    print(f"  Mean annual input (precip + glacier melt): {mean_input:.1f} mm/year")
    print(f"  Mean annual output (observed streamflow): {mean_output:.1f} mm/year")
    print(f"  Runoff ratio: {runoff_ratio:.3f}")
    print(f"  Correlation coefficient: {corr:.3f}")
    
    return yearly_data

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
    
    # Load glacier contributions data
    glacier_df = load_glacier_contributions_data(config, validation_start, validation_end)
    if glacier_df is None:
        print("ERROR: Could not load glacier contributions data")
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
    
    # Yearly sums for streamflow and precipitation
    streamflow_yearly = streamflow_filtered.groupby('year').agg({
        'obs_Q_mm': 'sum',
        'sim_Q_mm': 'sum',
        'precip': 'sum'
    }).reset_index()
    
    # Yearly sums for glacier melt (normalized to catchment scale)
    glacier_yearly = glacier_df.groupby('year')['glacier_icemelt_normalized'].sum().reset_index()
    glacier_yearly.columns = ['year', 'glacier_melt']
    
    # Merge data
    yearly_data = pd.merge(streamflow_yearly, glacier_yearly, on='year', how='inner')
    
    if len(yearly_data) == 0:
        print("ERROR: No overlapping years found between datasets")
        return None
    
    print(f"  - Found {len(yearly_data)} years of complete data")
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(max(12, len(yearly_data) * 0.8), 8))
    
    x = np.arange(len(yearly_data))
    width = 0.25  # Width of bars
    
    # Plot bars
    bars1 = ax.bar(x - width, yearly_data['precip'], width, label='Precipitation', 
                   color='darkblue', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, yearly_data['glacier_melt'], width, label='Glacier Melt', 
                   color='grey', alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + width, yearly_data['obs_Q_mm'], width, label='Observed Streamflow', 
                   color='black', alpha=0.8, edgecolor='white', linewidth=1)
    
    # Add value labels on bars (only show if not too crowded)
    if len(yearly_data) <= 15:  # Only add labels if not too many years
        def add_value_labels(bars, values):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(yearly_data[['precip', 'glacier_melt', 'obs_Q_mm']].max()) * 0.01,
                       f'{value:.0f}', ha='center', va='bottom', fontsize=9, rotation=90)
        
        add_value_labels(bars1, yearly_data['precip'])
        add_value_labels(bars2, yearly_data['glacier_melt'])
        add_value_labels(bars3, yearly_data['obs_Q_mm'])
    
    # Formatting
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Annual Sum (mm/year)', fontsize=14, fontweight='bold')
    ax.set_title(f'Annual Water Balance Components - Catchment {gauge_id}\n'
                f'Period: {validation_start} to {validation_end}', 
                fontsize=16, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(year)) for year in yearly_data['year']], rotation=45)
    ax.legend(fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'annual_water_balance_bars_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved annual bar plot to: {save_path}")
    plt.show()
    
    # Print summary statistics
    yearly_data['input_total'] = yearly_data['precip'] + yearly_data['glacier_melt']
    yearly_data['runoff_ratio'] = yearly_data['obs_Q_mm'] / yearly_data['input_total']
    yearly_data['glacier_contribution_pct'] = (yearly_data['glacier_melt'] / yearly_data['input_total']) * 100
    
    print(f"\nAnnual Water Balance Summary:")
    print(f"  Mean annual precipitation: {yearly_data['precip'].mean():.1f} ± {yearly_data['precip'].std():.1f} mm/year")
    print(f"  Mean annual glacier melt: {yearly_data['glacier_melt'].mean():.1f} ± {yearly_data['glacier_melt'].std():.1f} mm/year")
    print(f"  Mean annual streamflow: {yearly_data['obs_Q_mm'].mean():.1f} ± {yearly_data['obs_Q_mm'].std():.1f} mm/year")
    print(f"  Mean runoff ratio: {yearly_data['runoff_ratio'].mean():.3f} ± {yearly_data['runoff_ratio'].std():.3f}")
    print(f"  Mean glacier contribution: {yearly_data['glacier_contribution_pct'].mean():.1f} ± {yearly_data['glacier_contribution_pct'].std():.1f}%")
    
    return yearly_data



#--------------------------------------------------------------------------------
##################################### Forcing ###################################
#--------------------------------------------------------------------------------

def plot_glacier_hru_temperatures(shp_path, temp_csv_path, output_dir=None):
    """
    Plot temperature data for glacier HRUs (landuse class 7)
    
    Parameters:
    shp_path (str): Path to the HRU.shp file
    temp_csv_path (str): Path to the temperature CSV file
    output_dir (str): Optional directory to save plots
    """
    
    # Load HRU shapefile
    print("Loading HRU shapefile...")
    hru_gdf = gpd.read_file(shp_path)
    
    # Filter for glacier HRUs (landuse class 7)
    glacier_hrus = hru_gdf[hru_gdf['Landuse_Cl'] == 7]
    glacier_hru_ids = glacier_hrus.index.tolist()  # Assuming index corresponds to HRU ID
    
    # If the HRU ID is in a different column, adjust accordingly
    # glacier_hru_ids = glacier_hrus['HRU_ID'].tolist()  # uncomment if HRU_ID column exists
    
    print(f"Found {len(glacier_hru_ids)} glacier HRUs: {glacier_hru_ids}")
    
    # Load temperature data
    print("Loading temperature data...")
    temp_df = pd.read_csv(temp_csv_path)
    
    # The first row contains HRU numbers as column names, second row contains 'time', 'day', 'mean', etc.
    # Let's read it properly
    temp_df_header = pd.read_csv(temp_csv_path, nrows=1)
    column_names = temp_df_header.columns.tolist()
    
    # Read the actual data starting from row 2
    temp_df = pd.read_csv(temp_csv_path, skiprows=1)
    temp_df.columns = column_names
    
    # Convert the first column to datetime if it's a date column
    if 'time' in temp_df.columns[0].lower() or 'date' in temp_df.columns[0].lower():
        try:
            temp_df[temp_df.columns[0]] = pd.to_datetime(temp_df[temp_df.columns[0]])
        except:
            print("Could not convert first column to datetime, using as is")
    
    # Plot temperatures for glacier HRUs
    plt.figure(figsize=(15, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(glacier_hru_ids)))
    
    for i, hru_id in enumerate(glacier_hru_ids):
        # Find the column corresponding to this HRU
        hru_col = None
        for col in temp_df.columns[1:]:  # Skip the first column (time/date)
            try:
                if int(col) == hru_id or str(hru_id) in str(col):
                    hru_col = col
                    break
            except:
                continue
        
        if hru_col is not None:
            plt.plot(temp_df[temp_df.columns[0]], temp_df[hru_col], 
                    label=f'HRU {hru_id}', color=colors[i], alpha=0.7)
            
            # Check for winter melting (temperature > 0 in winter months)
            if temp_df[temp_df.columns[0]].dtype == 'datetime64[ns]':
                winter_mask = temp_df[temp_df.columns[0]].dt.month.isin([12, 1, 2])
                winter_positive = temp_df[winter_mask & (temp_df[hru_col] > 0)]
                if len(winter_positive) > 0:
                    print(f"WARNING: HRU {hru_id} has {len(winter_positive)} days with positive temperatures in winter!")
        else:
            print(f"Warning: Could not find temperature data for HRU {hru_id}")
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Freezing point')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.title('Mean Temperature for Glacier HRUs (Landuse Class 7)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'glacier_hru_temperatures.png'), 
                   dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Create a summary plot showing temperature statistics
    plt.figure(figsize=(12, 8))
    
    temp_stats = []
    hru_labels = []
    
    for hru_id in glacier_hru_ids:
        hru_col = None
        for col in temp_df.columns[1:]:
            try:
                if int(col) == hru_id or str(hru_id) in str(col):
                    hru_col = col
                    break
            except:
                continue
        
        if hru_col is not None:
            temps = temp_df[hru_col].dropna()
            temp_stats.append(temps)
            hru_labels.append(f'HRU {hru_id}')
    
    if temp_stats:
        plt.boxplot(temp_stats, labels=hru_labels)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Freezing point')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature Distribution for Glacier HRUs')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'glacier_hru_temperature_boxplot.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    return glacier_hru_ids