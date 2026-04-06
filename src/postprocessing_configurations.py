# This script is postprocessing Raven output from a multiple model configurations using namelist
# August 2025

#--------------------------------------------------------------------------------
################################## packages #####################################
#--------------------------------------------------------------------------------

# Import all functions from postprocessing.py
from postprocessing import *
from paths import get_paths

import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use('Agg')  # Use non-interactive backend to prevent image viewer
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

#--------------------------------------------------------------------------------
################################### general #####################################
#--------------------------------------------------------------------------------

def create_multi_plot_dir(multi_config):
    """
    Create plot directory for multi-configuration analysis.
    Placed alongside the config folders by finding their common directory prefix.

    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary

    Returns:
    --------
    Path
        Path to the multi-configuration plot directory
    """

    configs = multi_config['configs']

    if configs:
        ind_config = _build_individual_config(multi_config, configs[0])
        paths = get_paths(ind_config)
        plot_dir = paths['plots_dir']
    else:
        main_dir = Path(multi_config['main_dir'])
        gauge_id = multi_config['gauge_id']
        plot_dir = main_dir / 'model_runs' / f'catchment_{gauge_id}' / 'plots'

    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plot directory: {plot_dir}")
    return plot_dir

#--------------------------------------------------------------------------------
################################## hydrograph ###################################
#--------------------------------------------------------------------------------

def plot_hydrological_regime_comparison(multi_config, validation_start=None, validation_end=None):
    """
    Plot hydrological regime comparison across multiple configurations.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary containing:
        - 'main_dir': main directory path
        - 'gauge_id': gauge identifier
        - 'configs': list of configuration directory names
        - 'config_colors': dict mapping config names to colors
        - 'config_names': dict mapping config names to display names
        - 'start_date', 'end_date', 'cali_end_date': date strings
        - 'model_type': model type (default 'HBV')
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    dict
        Dictionary containing monthly data for each configuration
    """
    
    # Use dates from multi_config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']
    model_type = multi_config.get('model_type', 'HBV')
    
    print(f"Creating hydrological regime comparison for {len(configs)} configurations:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Validation period: {validation_start} to {validation_end}")
    
    # Create plot directory
    plot_dir = create_multi_plot_dir(multi_config)
    
    # Store results for each configuration
    config_results = {}
    obs_data = None  # Store observed data (should be same for all configs)
    
    # Process each configuration
    for config_dir in configs:
        print(f"\nProcessing configuration: {config_dir}")
        
        individual_config = _build_individual_config(multi_config, config_dir)

        try:
            # Load hydrograph data for this configuration
            data = load_hydrograph_data(individual_config)
            if data is None:
                print(f"  Warning: No hydrograph data loaded for {config_dir}")
                continue

            # Filter for validation period
            validation_mask = (data['date'] >= validation_start) & (data['date'] <= validation_end)
            df_validation = data[validation_mask].copy()

            if len(df_validation) == 0:
                print(f"  Warning: No data found for validation period in {config_dir}")
                continue

            # Calculate monthly means
            df_validation['month'] = df_validation['date'].dt.month
            monthly_data = {}
            
            if 'sim_Q' in df_validation.columns:
                monthly_data['sim_Q'] = df_validation.groupby('month')['sim_Q'].mean()
                print(f"  ✓ Loaded simulated data: {len(monthly_data['sim_Q'])} months")
            
            if 'obs_Q' in df_validation.columns:
                monthly_data['obs_Q'] = df_validation.groupby('month')['obs_Q'].mean()
                if obs_data is None:  # Store observed data from first config
                    obs_data = monthly_data['obs_Q'].copy()
                print(f"  ✓ Loaded observed data: {len(monthly_data['obs_Q'])} months")
            
            # Calculate performance metrics if both available
            performance = {}
            if 'obs_Q' in df_validation.columns and 'sim_Q' in df_validation.columns:
                obs = df_validation['obs_Q'].values
                sim = df_validation['sim_Q'].values
                
                # NSE
                obs_mean = np.mean(obs)
                nse = 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - obs_mean) ** 2))
                
                # KGE
                mean_sim = np.mean(sim)
                mean_obs = np.mean(obs)
                std_sim = np.std(sim)
                std_obs = np.std(obs)
                corr = np.corrcoef(sim, obs)[0, 1]
                alpha = std_sim / std_obs
                beta = mean_sim / mean_obs
                kge = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                
                performance = {'NSE': nse, 'KGE': kge}
                print(f"  ✓ Performance: NSE={nse:.3f}, KGE={kge:.3f}")
            
            monthly_df = pd.DataFrame(monthly_data)
            config_results[config_dir] = {
                'monthly_data': monthly_df,
                'performance': performance,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir)
            }
            
        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue
    
    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None
    
    # Create comparison plot
    plt.figure(figsize=(14, 8))
    
    # Plot observed data first (if available)
    if obs_data is not None:
        plt.plot(obs_data.index, obs_data.values, 'k-', linewidth=3, 
                label='Observed', zorder=10)
    
    # Plot each configuration
    for config_dir, result in config_results.items():
        monthly_df = result['monthly_data']
        color = result['color']
        name = result['name']
        perf = result['performance']
        
        if 'sim_Q' in monthly_df.columns:
            # Create label with performance metrics if available
            if perf:
                label = f"{name} (NSE={perf['NSE']:.3f}, KGE={perf['KGE']:.3f})"
            else:
                label = name
            
            plt.plot(monthly_df.index, monthly_df['sim_Q'], 
                    color=color, linewidth=2.5, label=label, zorder=5)
    
    # Formatting
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Discharge (m³/s)', fontsize=14)
    plt.title(f'Hydrological Regime Comparison - Catchment {gauge_id}\n'
             f'Validation Period: {validation_start} to {validation_end}', 
             fontsize=16, fontweight='bold')
    
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    
    # Save plot in the main directory
    save_path = plot_dir / f'hydrological_regime_comparison_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved hydrological regime comparison plot to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\nHydrological Regime Comparison Summary:")
    print(f"  Configurations processed: {len(config_results)}")
    print(f"  Validation period: {validation_start} to {validation_end}")
    
    if obs_data is not None:
        print(f"  Mean observed discharge: {obs_data.mean():.2f} m³/s")
    
    print(f"  Configuration performance:")
    for config_dir, result in config_results.items():
        name = result['name']
        perf = result['performance']
        if perf:
            print(f"    - {name}: NSE={perf['NSE']:.3f}, KGE={perf['KGE']:.3f}")
        else:
            print(f"    - {name}: No performance metrics available")
    
    return config_results

#--------------------------------------------------------------------------------

def plot_hydrological_regime_subplots(multi_config, validation_start=None, validation_end=None, unit='m3s'):
    """
    Plot hydrological regime for each configuration in separate subplots.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    unit : str, optional
        Unit for discharge ('m3s' for m³/s or 'mm' for mm/day)
        
    Returns:
    --------
    dict
        Dictionary containing monthly data for each configuration
    """
    
    # Use dates from multi_config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']
    model_type = multi_config.get('model_type', 'HBV')
    
    # Determine unit label
    unit_label = 'm³/s' if unit == 'm3s' else 'mm/day'
    
    print(f"Creating individual hydrological regime plots for {len(configs)} configurations:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Validation period: {validation_start} to {validation_end}")
    print(f"  - Unit: {unit_label}")
    
    # Create plot directory
    plot_dir = create_multi_plot_dir(multi_config)
    
    # Store results for each configuration
    config_results = {}
    obs_data = None  # Store observed data (should be same for all configs)
    
    # Process each configuration
    for config_dir in configs:
        print(f"\nProcessing configuration: {config_dir}")
        
        individual_config = _build_individual_config(multi_config, config_dir)
        
        try:
            # Load hydrograph data for this configuration
            data = load_hydrograph_data(individual_config)
            if data is None:
                print(f"  Warning: No hydrograph data loaded for {config_dir}")
                continue
            
            # Filter for validation period
            validation_mask = (data['date'] >= validation_start) & (data['date'] <= validation_end)
            df_validation = data[validation_mask].copy()
            
            if len(df_validation) == 0:
                print(f"  Warning: No data found for validation period in {config_dir}")
                continue
            
            # Convert to mm/day if needed
            if unit == 'mm':
                paths = get_paths(individual_config)
                topo_dir = paths['topo_dir']
                hru_shapefile = topo_dir / "HRU.shp"
                if hru_shapefile.exists():
                    import geopandas as gpd
                    hru_gdf = gpd.read_file(hru_shapefile)
                    total_area_km2 = hru_gdf['Area_km2'].sum()
                    # Conversion factor: m³/s to mm/day
                    # mm/day = (m³/s * 86400 s/day) / (area_m² ) * 1000 mm/m
                    conversion = 86400 / (total_area_km2 * 1000000) * 1000

                    if 'sim_Q' in df_validation.columns:
                        df_validation['sim_Q'] = df_validation['sim_Q'] * conversion
                    if 'obs_Q' in df_validation.columns:
                        df_validation['obs_Q'] = df_validation['obs_Q'] * conversion

                    print(f"  ✓ Converted discharge to mm/day (catchment area: {total_area_km2:.2f} km²)")
                else:
                    print(f"  ⚠️  Warning: Could not find HRU shapefile for area calculation, using m³/s")
            
            # Calculate monthly means
            df_validation['month'] = df_validation['date'].dt.month
            monthly_data = {}
            
            if 'sim_Q' in df_validation.columns:
                monthly_data['sim_Q'] = df_validation.groupby('month')['sim_Q'].mean()
                print(f"  ✓ Loaded simulated data: {len(monthly_data['sim_Q'])} months")
            
            if 'obs_Q' in df_validation.columns:
                monthly_data['obs_Q'] = df_validation.groupby('month')['obs_Q'].mean()
                if obs_data is None:  # Store observed data from first config
                    obs_data = monthly_data['obs_Q'].copy()
                print(f"  ✓ Loaded observed data: {len(monthly_data['obs_Q'])} months")
            
            # Calculate performance metrics if both available
            performance = {}
            if 'obs_Q' in df_validation.columns and 'sim_Q' in df_validation.columns:
                obs = df_validation['obs_Q'].values
                sim = df_validation['sim_Q'].values
                
                # NSE
                obs_mean = np.mean(obs)
                nse = 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - obs_mean) ** 2))
                
                # KGE
                mean_sim = np.mean(sim)
                mean_obs = np.mean(obs)
                std_sim = np.std(sim)
                std_obs = np.std(obs)
                corr = np.corrcoef(sim, obs)[0, 1]
                alpha = std_sim / std_obs
                beta = mean_sim / mean_obs
                kge = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                
                performance = {'NSE': nse, 'KGE': kge}
                print(f"  ✓ Performance: NSE={nse:.3f}, KGE={kge:.3f}")
            
            monthly_df = pd.DataFrame(monthly_data)
            config_results[config_dir] = {
                'monthly_data': monthly_df,
                'performance': performance,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir)
            }
            
        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue
    
    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None
    
    # Calculate subplot layout
    n_configs = len(config_results)
    if n_configs <= 2:
        n_rows, n_cols = 1, n_configs
        figsize = (7 * n_configs, 6)
    elif n_configs <= 4:
        n_rows, n_cols = 2, 2
        figsize = (14, 10)
    elif n_configs <= 6:
        n_rows, n_cols = 2, 3
        figsize = (18, 10)
    elif n_configs <= 9:
        n_rows, n_cols = 3, 3
        figsize = (18, 15)
    else:
        # For more than 9 configs, use 4 columns
        n_cols = 4
        n_rows = (n_configs + n_cols - 1) // n_cols
        figsize = (20, 5 * n_rows)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    
    # Handle single subplot case
    if n_configs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_configs > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each configuration in its own subplot
    for i, (config_dir, result) in enumerate(config_results.items()):
        ax = axes[i]
        monthly_df = result['monthly_data']
        color = result['color']
        name = result['name']
        perf = result['performance']
        
        # Plot observed data first (if available)
        if obs_data is not None:
            ax.plot(obs_data.index, obs_data.values, 'k-', linewidth=3.5, 
                   label='Observed', zorder=10)
        
        # Plot simulated data for this configuration with dashed line
        if 'sim_Q' in monthly_df.columns:
            ax.plot(monthly_df.index, monthly_df['sim_Q'], 
                   color=color, linewidth=3, linestyle='--', label='Simulated', zorder=5)
        
        # Formatting for this subplot
        ax.set_title(f'{name}', fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        
        # Only show legend in the first plot (i==0)
        if i == 0:
            ax.legend(loc='best', fontsize=13)
        
        # Add performance metrics as text
        if perf:
            perf_text = f"NSE={perf['NSE']:.3f}\nKGE={perf['KGE']:.3f}"
            ax.text(0.02, 0.98, perf_text, transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Set x-axis labels for bottom row
        if i >= (n_rows - 1) * n_cols or i >= n_configs - n_cols:
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=13)
        
        # Set y-axis label for leftmost column
        if i % n_cols == 0:
            ax.set_ylabel(f'Discharge ({unit_label})', fontsize=15, fontweight='bold')
    
    # Hide unused subplots
    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title and labels
    #fig.suptitle(f'Hydrological Regime Comparison by Configuration - Catchment {gauge_id}\n'
    #            f'Validation Period: {validation_start} to {validation_end}', 
    #            fontsize=16, fontweight='bold', y=0.98)
    
    # Add common x-label
    fig.text(0.5, 0.02, 'Month', ha='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.08)
    
    # Save plot
    save_path = plot_dir / f'hydrological_regime_subplots_{gauge_id}_{unit}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved hydrological regime subplots to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\nHydrological Regime Subplots Summary:")
    print(f"  Configurations processed: {len(config_results)}")
    print(f"  Layout: {n_rows} rows × {n_cols} columns")
    print(f"  Validation period: {validation_start} to {validation_end}")
    print(f"  Unit: {unit_label}")
    
    if obs_data is not None:
        print(f"  Mean observed discharge: {obs_data.mean():.2f} {unit_label}")
    
    print(f"  Configuration performance:")
    for config_dir, result in config_results.items():
        name = result['name']
        perf = result['performance']
        if perf:
            print(f"    - {name}: NSE={perf['NSE']:.3f}, KGE={perf['KGE']:.3f}")
        else:
            print(f"    - {name}: No performance metrics available")
    
    return config_results

#--------------------------------------------------------------------------------

def plot_hydrograph_timeseries_comparison(multi_config, validation_start=None, validation_end=None, 
                                        random_seed=42, n_years=2):
    """
    Plot hydrograph time series comparison for random years from validation period 
    across multiple configurations.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    random_seed : int
        Random seed for reproducible year selection
    n_years : int
        Number of random years to plot (default: 2)
        
    Returns:
    --------
    dict
        Dictionary containing selected years and config results
    """
    
    # Use dates from multi_config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']
    model_type = multi_config.get('model_type', 'HBV')
    
    print(f"Creating hydrograph timeseries comparison for {len(configs)} configurations:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Validation period: {validation_start} to {validation_end}")
    print(f"  - Number of random years: {n_years}")
    
    # Create plot directory
    plot_dir = create_multi_plot_dir(multi_config)
    
    # Store results for each configuration
    config_results = {}
    available_years = None
    
    # First pass: Load data and find available years
    for config_dir in configs:
        print(f"\nProcessing configuration: {config_dir}")
        
        individual_config = _build_individual_config(multi_config, config_dir)
        
        try:
            # Load hydrograph data for this configuration
            data = load_hydrograph_data(individual_config)
            if data is None:
                print(f"  Warning: No hydrograph data loaded for {config_dir}")
                continue
            
            # Filter for validation period
            validation_mask = (data['date'] >= validation_start) & (data['date'] <= validation_end)
            df_validation = data[validation_mask].copy()
            
            if len(df_validation) == 0:
                print(f"  Warning: No data found for validation period in {config_dir}")
                continue
            
            # Get available years for this configuration
            val_years = df_validation['date'].dt.year.unique()
            if available_years is None:
                available_years = set(val_years)
            else:
                available_years = available_years.intersection(set(val_years))
            
            config_results[config_dir] = {
                'data': data,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir)
            }
            
            print(f"  ✓ Loaded data with years: {sorted(val_years)}")
            
        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue
    
    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None
    
    if not available_years or len(available_years) < n_years:
        print(f"Not enough common years available. Found: {sorted(available_years) if available_years else 'None'}")
        return None
    
    # Select random years that are available in all configurations
    np.random.seed(random_seed)
    selected_years = sorted(np.random.choice(list(available_years), size=n_years, replace=False))
    
    print(f"\nSelected random years: {selected_years}")
    
    # Create subplots for each year
    fig, axes = plt.subplots(n_years, 1, figsize=(16, 6*n_years), sharex=False)
    if n_years == 1:
        axes = [axes]  # Make it iterable for single subplot
    
    for year_idx, year in enumerate(selected_years):
        ax = axes[year_idx]
        
        # Plot observed data first (should be same for all configs)
        obs_plotted = False
        
        for config_dir, result in config_results.items():
            data = result['data']
            color = result['color']
            name = result['name']
            
            # Filter for this specific year
            year_mask = (data['date'].dt.year == year) & \
                       (data['date'] >= validation_start) & \
                       (data['date'] <= validation_end)
            year_data = data[year_mask].copy()
            
            if len(year_data) == 0:
                continue
            
            # Plot observed data once
            if 'obs_Q' in year_data.columns and not obs_plotted:
                ax.plot(year_data['date'], year_data['obs_Q'], 'k-', 
                       linewidth=2.5, label='Observed', zorder=10)
                obs_plotted = True
            
            # Plot simulated data for this configuration
            if 'sim_Q' in year_data.columns:
                ax.plot(year_data['date'], year_data['sim_Q'], 
                       color=color, linewidth=2, label=name, zorder=5)
        
        # Formatting for this subplot
        ax.set_ylabel('Discharge (m³/s)', fontsize=12)
        ax.set_title(f'Hydrograph for Year {year} - Catchment {gauge_id}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        ax.legend(loc='best', fontsize=10)
        
        # Format x-axis
        if year_idx == len(selected_years) - 1:  # Last subplot
            ax.set_xlabel('Date', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    years_str = '_'.join(map(str, selected_years))
    save_path = plot_dir / f'hydrograph_timeseries_comparison_{years_str}_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved hydrograph timeseries comparison plot to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\nHydrograph Timeseries Comparison Summary:")
    print(f"  Configurations processed: {len(config_results)}")
    print(f"  Selected years: {selected_years}")
    print(f"  Available years in all configs: {sorted(available_years)}")
    
    return {
        'selected_years': selected_years,
        'available_years': sorted(available_years),
        'config_results': config_results
    }

#--------------------------------------------------------------------------------
###################################### SWE ######################################
#--------------------------------------------------------------------------------

def plot_swe_timeseries_comparison(multi_config, validation_start=None, validation_end=None):
    """
    Plot area-weighted SWE time series comparison across multiple configurations.
    Works with or without observed data - plots simulated data only if no observed data available.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    dict
        Dictionary containing SWE data and metrics for each configuration
    """
    
    # Use dates from multi_config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']
    model_type = multi_config.get('model_type', 'HBV')
    
    print(f"Creating SWE timeseries comparison for {len(configs)} configurations:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Validation period: {validation_start} to {validation_end}")
    
    # Create plot directory
    plot_dir = create_multi_plot_dir(multi_config)
    
    # Store results for each configuration
    config_results = {}
    obs_swe_data = None  # Store observed data (should be same for all configs)
    has_observed_data = False
    
    # Process each configuration
    for config_dir in configs:
        print(f"\nProcessing configuration: {config_dir}")
        
        individual_config = _build_individual_config(multi_config, config_dir)
        
        try:
            # Load SWE data for this configuration (obs_data can be None now)
            sim_data, obs_data, area_data = load_swe_data(individual_config)
            
            if sim_data is None:
                print(f"  ❌ Warning: Failed to load simulated SWE data for {config_dir}")
                continue
            
            # Check if we have observed data
            has_obs_this_config = (obs_data is not None)
            if not has_obs_this_config:
                print(f"  ℹ️  No observed SWE data found for {config_dir} - using simulated data only")
            
            # Process the data
            processed = process_swe_data(sim_data, obs_data, area_data)
            if processed is None:
                print(f"  ❌ Warning: Failed to process SWE data for {config_dir}")
                continue
            
            # Get processed components
            band_mapping = processed['band_mapping']
            area_mapping = processed['area_mapping']
            sim_data_proc = processed['sim_data']
            obs_data_proc = processed['obs_data']  # Can be None
            
            # Convert validation dates to datetime
            validation_start_dt = pd.to_datetime(validation_start)
            validation_end_dt = pd.to_datetime(validation_end)
            
            # Filter simulated data for validation period
            sim_data_proc['date'] = pd.to_datetime(sim_data_proc['date'])
            val_sim_mask = (sim_data_proc['date'] >= validation_start_dt) & (sim_data_proc['date'] <= validation_end_dt)
            val_sim = sim_data_proc[val_sim_mask].copy()
            
            if len(val_sim) == 0:
                print(f"  ❌ Warning: No simulation data found for validation period in {config_dir}")
                continue
            
            # Calculate area-weighted SWE for simulated data
            val_sim['area_weighted_swe'] = calculate_area_weighted_swe(val_sim, area_mapping)
            
            # Process observed data if available
            val_obs = None
            if has_obs_this_config and obs_data_proc is not None:
                obs_data_proc['time'] = pd.to_datetime(obs_data_proc['time'])
                val_obs_mask = (obs_data_proc['time'] >= validation_start_dt) & (obs_data_proc['time'] <= validation_end_dt)
                val_obs = obs_data_proc[val_obs_mask].copy()
                
                if len(val_obs) > 0:
                    val_obs['area_weighted_swe'] = calculate_area_weighted_swe(val_obs, area_mapping)
                    # Store observed data from first successful config
                    if obs_swe_data is None:
                        obs_swe_data = val_obs.copy()
                        has_observed_data = True
            
            # Calculate metrics if both sim and obs are available
            metrics = {'overall_rmse': None, 'overall_bias': None, 'overall_corr': None,
                      'area_weighted_rmse': None, 'area_weighted_bias': None, 'area_weighted_corr': None}
            
            if has_obs_this_config and val_obs is not None and len(val_obs) > 0:
                try:
                    metrics = calculate_swe_metrics(
                        sim_data, obs_data,
                        processed['sim_elev_cols'], processed['obs_elev_cols'],
                        band_mapping, area_mapping
                    )
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not calculate metrics for {config_dir}: {e}")
            
            config_results[config_dir] = {
                'swe_data': val_sim,
                'obs_data': val_obs,
                'metrics': metrics,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir),
                'has_obs': has_obs_this_config
            }
            
            print(f"  ✅ Processed SWE data: {len(val_sim)} records")
            if has_obs_this_config and metrics['overall_rmse'] is not None:
                print(f"    📏 RMSE: {metrics['overall_rmse']:.1f} mm, Bias: {metrics['overall_bias']:.1f} mm, Corr: {metrics['overall_corr']:.3f}")
            else:
                print(f"    📊 Simulated SWE: Mean={val_sim['area_weighted_swe'].mean():.1f} mm, Max={val_sim['area_weighted_swe'].max():.1f} mm")
            
        except Exception as e:
            print(f"  ❌ Error processing {config_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(config_results) == 0:
        print("❌ No configurations processed successfully")
        return None
    
    # Create comparison plot
    plt.figure(figsize=(16, 10))
    
    # Plot observed SWE data first (if available)
    if has_observed_data and obs_swe_data is not None and 'area_weighted_swe' in obs_swe_data.columns:
        plt.plot(obs_swe_data['time'], obs_swe_data['area_weighted_swe'], 
                'k-', linewidth=3, label='Observed Area-Weighted SWE', zorder=10)
        plot_title_suffix = "with Observations"
    else:
        plot_title_suffix = "Simulated Only"
        print("ℹ️  No observed data available - plotting simulated SWE only")
    
    # Plot each configuration
    for config_dir, result in config_results.items():
        swe_data = result['swe_data']
        color = result['color']
        name = result['name']
        metrics = result['metrics']
        has_obs = result['has_obs']
        
        if 'area_weighted_swe' in swe_data.columns:
            # Create label with metrics if available, otherwise just configuration name
            if has_obs and metrics['overall_rmse'] is not None:
                label = f"{name} (RMSE={metrics['overall_rmse']:.1f} mm, Bias={metrics['overall_bias']:.1f} mm)"
            else:
                label = name
            
            plt.plot(swe_data['date'], swe_data['area_weighted_swe'], 
                    color=color, linewidth=2.5, label=label, zorder=5)
    
    # Formatting
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Snow Water Equivalent (mm)', fontsize=14)
    plt.title(f'Area-Weighted SWE Time Series Comparison - Catchment {gauge_id} ({plot_title_suffix})\n'
             f'Validation Period: {validation_start} to {validation_end}', 
             fontsize=16, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.legend(loc='best', fontsize=11)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=6))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dir / f'swe_timeseries_comparison_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved SWE timeseries comparison plot to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\n📊 SWE Timeseries Comparison Summary:")
    print(f"  Configurations processed: {len(config_results)}")
    print(f"  Validation period: {validation_start} to {validation_end}")
    print(f"  Observed data available: {'Yes' if has_observed_data else 'No'}")
    
    if has_observed_data and obs_swe_data is not None:
        mean_obs = obs_swe_data['area_weighted_swe'].mean()
        max_obs = obs_swe_data['area_weighted_swe'].max()
        print(f"  Observed SWE: Mean={mean_obs:.1f} mm, Max={max_obs:.1f} mm")
    
    print(f"\n  Configuration performance:")
    for config_dir, result in config_results.items():
        name = result['name']
        metrics = result['metrics']
        swe_data = result['swe_data']
        has_obs = result['has_obs']
        
        mean_sim = swe_data['area_weighted_swe'].mean()
        max_sim = swe_data['area_weighted_swe'].max()
        
        if has_obs and metrics['overall_rmse'] is not None:
            print(f"    - {name}:")
            print(f"      📏 SWE: Mean={mean_sim:.1f} mm, Max={max_sim:.1f} mm")
            print(f"      📊 Metrics: RMSE={metrics['overall_rmse']:.1f} mm, Bias={metrics['overall_bias']:.1f} mm, Corr={metrics['overall_corr']:.3f}")
            if metrics['area_weighted_rmse'] is not None:
                print(f"      🎯 Area-weighted: RMSE={metrics['area_weighted_rmse']:.1f} mm, Bias={metrics['area_weighted_bias']:.1f} mm, Corr={metrics['area_weighted_corr']:.3f}")
        else:
            print(f"    - {name}: SWE: Mean={mean_sim:.1f} mm, Max={max_sim:.1f} mm (Simulated only)")
    
    return config_results

#--------------------------------------------------------------------------------

def plot_swe_elevation_bands_comparison(multi_config, validation_start=None, validation_end=None, water_year=None):
    """
    Plot SWE time series by elevation bands comparison across multiple configurations.
    Each elevation band gets its own subplot with all configurations overlaid.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    water_year : int, optional
        Optional water year to filter (e.g. 2018 for 2018-2019 water year)
        
    Returns:
    --------
    dict
        Dictionary containing SWE elevation data for each configuration
    """
    
    # Determine time period
    if water_year is not None:
        # Water year: October 1 to September 30
        start_date = pd.to_datetime(f"{water_year}-10-01")
        end_date = pd.to_datetime(f"{water_year+1}-09-30")
        period_label = f"Water Year {water_year}-{water_year+1}"
    else:
        # Use validation period or config dates
        if validation_start is None:
            validation_start = multi_config.get('cali_end_date', '2010-01-01')
        if validation_end is None:
            validation_end = multi_config.get('end_date', '2020-12-31')
        
        start_date = pd.to_datetime(validation_start)
        end_date = pd.to_datetime(validation_end)
        period_label = f"Validation Period ({start_date.date()} to {end_date.date()})"
    
    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']
    model_type = multi_config.get('model_type', 'HBV')
    
    print(f"Creating SWE elevation bands comparison for {len(configs)} configurations:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Period: {period_label}")
    
    # Create plot directory
    plot_dir = create_multi_plot_dir(multi_config)
    
    # Store results for each configuration
    config_results = {}
    obs_swe_data = None  # Store observed data (should be same for all configs)
    common_elevation_bands = None
    
    # Process each configuration
    for config_dir in configs:
        print(f"\nProcessing configuration: {config_dir}")
        
        individual_config = _build_individual_config(multi_config, config_dir)
        
        try:
            # Load and process SWE data for this configuration
            sim_data, obs_data, area_data = load_swe_data(individual_config)
            if sim_data is None or obs_data is None:
                print(f"  Warning: Failed to load SWE data for {config_dir}")
                continue
            
            processed = process_swe_data(sim_data, obs_data, area_data)
            if processed is None:
                print(f"  Warning: Failed to process SWE data for {config_dir}")
                continue
            
            # Get processed components
            sim_data_proc = processed['sim_data']
            obs_data_proc = processed['obs_data']
            sim_elev_cols = processed['sim_elev_cols']
            obs_elev_cols = processed['obs_elev_cols']
            band_mapping = processed['band_mapping']
            
            if not band_mapping:
                print(f"  Warning: No matching elevation bands found for {config_dir}")
                continue
            
            # Convert date columns to datetime
            sim_data_proc['date'] = pd.to_datetime(sim_data_proc['date'])
            obs_data_proc['time'] = pd.to_datetime(obs_data_proc['time'])
            
            # Filter data for the specified period
            sim_mask = (sim_data_proc['date'] >= start_date) & (sim_data_proc['date'] <= end_date)
            obs_mask = (obs_data_proc['time'] >= start_date) & (obs_data_proc['time'] <= end_date)
            
            sim_filtered = sim_data_proc[sim_mask].copy()
            obs_filtered = obs_data_proc[obs_mask].copy()
            
            if len(sim_filtered) == 0:
                print(f"  Warning: No simulation data found for specified period in {config_dir}")
                continue
            
            # Store observed data from first successful config
            if obs_swe_data is None and len(obs_filtered) > 0:
                obs_swe_data = obs_filtered.copy()
            
            # Get elevation bands for this configuration
            elev_bands = set(band_mapping.keys())
            if common_elevation_bands is None:
                common_elevation_bands = elev_bands
            else:
                common_elevation_bands = common_elevation_bands.intersection(elev_bands)
            
            config_results[config_dir] = {
                'sim_data': sim_filtered,
                'obs_data': obs_filtered,
                'band_mapping': band_mapping,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir)
            }
            
            print(f"  ✓ Processed SWE elevation data: {len(sim_filtered)} records")
            print(f"    Elevation bands: {sorted(elev_bands)}")
            
        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue
    
    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None
    
    if not common_elevation_bands:
        print("No common elevation bands found across configurations")
        return None
    
    # Sort elevation bands by altitude
    elev_bands = sorted(common_elevation_bands, key=lambda x: int(x.split('-')[0]) if '-' in x else 0)
    
    print(f"\nCommon elevation bands: {elev_bands}")
    
    # Calculate subplot layout
    n_bands = len(elev_bands)
    n_cols = 1  # Single column layout for better readability
    n_rows = n_bands
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows), sharex=True)
    
    # Make axes iterable if there's only one plot
    if n_bands == 1:
        axes = np.array([axes])
    
    # Plot each elevation band
    for i, band in enumerate(elev_bands):
        ax = axes[i]
        
        # Plot observed data first (if available)
        obs_plotted = False
        if obs_swe_data is not None and band in obs_swe_data.columns:
            obs_values = obs_swe_data[band].copy()
            # Convert to mm if needed
            if obs_values.mean() < 10 and obs_values.max() < 20:
                obs_values *= 1000
            
            ax.plot(obs_swe_data['time'], obs_values, 
                   'k-', label='Observed', linewidth=3, zorder=10)
            obs_plotted = True
        
        # Plot each configuration
        for config_dir, result in config_results.items():
            sim_data = result['sim_data']
            color = result['color']
            name = result['name']
            
            if band in sim_data.columns:
                sim_values = sim_data[band].copy()
                # Convert to mm if needed
                if sim_values.mean() < 10 and sim_values.max() < 20:
                    sim_values *= 1000
                
                ax.plot(sim_data['date'], sim_values, 
                       color=color, label=name, linewidth=2, zorder=5)
        
        # Formatting for this subplot
        ax.set_title(f'Elevation Band: {band}', fontsize=12, fontweight='bold')
        ax.set_ylabel('SWE (mm)', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        
        # Format x-axis dates
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        # Add legend only to the first plot
        if i == 0:
            ax.legend(loc='best', fontsize=10)
    
    # Add overall title
    fig.suptitle(f'SWE by Elevation Band Comparison - Catchment {gauge_id}\n{period_label}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Format x-axis for the bottom plot
    axes[-1].set_xlabel('Date', fontsize=12)
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.3)
    
    # Save figure
    if water_year:
        filename = f'swe_elevation_bands_comparison_WY{water_year}_{gauge_id}.png'
    else:
        filename = f'swe_elevation_bands_comparison_{gauge_id}.png'
    
    save_path = plot_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved SWE elevation bands comparison plot to: {save_path}")
    plt.show()
    
    # Calculate and print summary statistics with metrics per elevation band
    print(f"\nSWE Elevation Bands Comparison Summary:")
    print(f"  Configurations processed: {len(config_results)}")
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    print(f"  Common elevation bands: {len(elev_bands)}")
    print(f"  Elevation bands: {', '.join(elev_bands)}")
    
    # Calculate metrics for each configuration and elevation band
    print(f"\nPerformance by elevation band:")
    for band in elev_bands:
        print(f"\n  Elevation Band {band}:")
        
        for config_dir, result in config_results.items():
            name = result['name']
            sim_data = result['sim_data']
            
            if band in sim_data.columns:
                # Calculate metrics if observed data is available
                if obs_swe_data is not None and band in obs_swe_data.columns:
                    # Merge data for metrics calculation
                    merged = pd.merge(
                        sim_data[['date', band]].rename(columns={'date': 'time', band: 'sim'}),
                        obs_swe_data[['time', band]].rename(columns={band: 'obs'}),
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
                        
                        mean_sim = merged['sim'].mean()
                        mean_obs = merged['obs'].mean()
                        
                        print(f"    - {name}: Mean SWE={mean_sim:.1f}mm (obs={mean_obs:.1f}mm), RMSE={rmse:.1f}mm, Bias={bias:.1f}mm, R={corr:.3f}")
                    else:
                        print(f"    - {name}: No overlapping data for metrics calculation")
                else:
                    # Just show mean values without metrics
                    sim_values = sim_data[band].copy()
                    if sim_values.mean() < 10:
                        sim_values *= 1000
                    mean_sim = sim_values.mean()
                    print(f"    - {name}: Mean SWE={mean_sim:.1f}mm (no observed data for comparison)")
            else:
                print(f"    - {name}: No data for this elevation band")
    
    return config_results

#--------------------------------------------------------------------------------
################################### metrics #####################################
#--------------------------------------------------------------------------------

def plot_streamflow_metrics_comparison(multi_config, validation_start=None, validation_end=None):
    """
    Plot streamflow metrics comparison across multiple configurations using bar plots.
    Each metric gets its own subplot due to different scales.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    dict
        Dictionary containing metrics for each configuration
    """
    
    # Use dates from multi_config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']
    model_type = multi_config.get('model_type', 'HBV')
    
    print(f"Creating streamflow metrics comparison for {len(configs)} configurations:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Validation period: {validation_start} to {validation_end}")
    
    # Create plot directory
    plot_dir = create_multi_plot_dir(multi_config)
    
    # Store results for each configuration
    config_results = {}
    
    # Process each configuration
    for config_dir in configs:
        print(f"\nProcessing configuration: {config_dir}")
        
        individual_config = _build_individual_config(multi_config, config_dir)
        
        try:
            # Load hydrograph data for this configuration
            data = load_hydrograph_data(individual_config)
            if data is None:
                print(f"  Warning: No hydrograph data loaded for {config_dir}")
                continue
            
            # Check if both observed and simulated data are available
            if 'obs_Q' not in data.columns or 'sim_Q' not in data.columns:
                print(f"  Warning: Missing observed or simulated data for {config_dir}")
                continue
            
            # Calculate metrics for validation period
            start_date = pd.to_datetime(validation_start)
            end_date = pd.to_datetime(validation_end)
            
            val_metrics = calculate_performance_metrics(
                data, start_date, end_date, "Validation"
            )
            
            if val_metrics is None:
                print(f"  Warning: Could not calculate metrics for {config_dir}")
                continue
            
            config_results[config_dir] = {
                'metrics': val_metrics,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir)
            }
            
            print(f"  ✓ Calculated metrics: NSE={val_metrics['NSE']:.3f}, KGE={val_metrics['KGE']:.3f}, KGE_NP={val_metrics['KGE_NP']:.3f}")
            
        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue
    
    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None
    
    # Extract data for plotting
    config_labels = []
    colors = []
    nse_values = []
    kge_values = []
    kge_np_values = []
    r_values = []
    alpha_values = []
    beta_values = []
    
    for config_dir, result in config_results.items():
        config_labels.append(result['name'])
        colors.append(result['color'])
        metrics = result['metrics']
        
        nse_values.append(metrics['NSE'])
        kge_values.append(metrics['KGE'])
        kge_np_values.append(metrics['KGE_NP'])
        r_values.append(metrics['r'])
        alpha_values.append(metrics['alpha'])
        beta_values.append(metrics['beta'])
    
    # Create subplot layout (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Define bar width and positions
    x_pos = np.arange(len(config_labels))
    bar_width = 0.6
    
    # Plot 1: NSE
    ax = axes[0]
    bars = ax.bar(x_pos, nse_values, width=bar_width, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Nash-Sutcliffe Efficiency (NSE)', fontsize=12, fontweight='bold')
    ax.set_ylabel('NSE', fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.75, color='green', linestyle='--', alpha=0.7, label='Very good (>0.75)')
    ax.axhline(y=0.65, color='orange', linestyle='--', alpha=0.7, label='Good (>0.65)')
    ax.axhline(y=0.50, color='red', linestyle='--', alpha=0.7, label='Satisfactory (>0.50)')
    
    # Add value labels on bars
    for bar, value in zip(bars, nse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.legend(loc='best', fontsize=8)
    
    # Plot 2: KGE
    ax = axes[1]
    bars = ax.bar(x_pos, kge_values, width=bar_width, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Kling-Gupta Efficiency (KGE)', fontsize=12, fontweight='bold')
    ax.set_ylabel('KGE', fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.75, color='green', linestyle='--', alpha=0.7, label='Very good (>0.75)')
    ax.axhline(y=0.65, color='orange', linestyle='--', alpha=0.7, label='Good (>0.65)')
    ax.axhline(y=0.50, color='red', linestyle='--', alpha=0.7, label='Satisfactory (>0.50)')
    
    # Add value labels on bars
    for bar, value in zip(bars, kge_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.legend(loc='best', fontsize=8)
    
    # Plot 3: KGE_NP
    ax = axes[2]
    bars = ax.bar(x_pos, kge_np_values, width=bar_width, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Non-Parametric KGE (KGE_NP)', fontsize=12, fontweight='bold')
    ax.set_ylabel('KGE_NP', fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.75, color='green', linestyle='--', alpha=0.7, label='Very good (>0.75)')
    ax.axhline(y=0.65, color='orange', linestyle='--', alpha=0.7, label='Good (>0.65)')
    ax.axhline(y=0.50, color='red', linestyle='--', alpha=0.7, label='Satisfactory (>0.50)')
    
    # Add value labels on bars
    for bar, value in zip(bars, kge_np_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.legend(loc='best', fontsize=8)
    
    # Plot 4: Correlation (r)
    ax = axes[3]
    bars = ax.bar(x_pos, r_values, width=bar_width, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Correlation Coefficient (r)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation (r)', fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect (1.0)')
    ax.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bar, value in zip(bars, r_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.legend(loc='best', fontsize=8)
    
    # Plot 5: Variability ratio (α)
    ax = axes[4]
    bars = ax.bar(x_pos, alpha_values, width=bar_width, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Variability Ratio (α)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Alpha (α)', fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect (1.0)')
    
    # Add value labels on bars
    for bar, value in zip(bars, alpha_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.legend(loc='best', fontsize=8)
    
    # Plot 6: Bias ratio (β)
    ax = axes[5]
    bars = ax.bar(x_pos, beta_values, width=bar_width, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Bias Ratio (β)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Beta (β)', fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect (1.0)')
    
    # Add value labels on bars
    for bar, value in zip(bars, beta_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.legend(loc='best', fontsize=8)
    
    # Add overall title
    fig.suptitle(f'Streamflow Performance Metrics Comparison - Catchment {gauge_id}\n'
                f'Validation Period: {validation_start} to {validation_end}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save plot
    save_path = plot_dir / f'streamflow_metrics_comparison_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved streamflow metrics comparison plot to: {save_path}")
    plt.show()
    
    # Print comprehensive summary
    print(f"\nStreamflow Metrics Comparison Summary:")
    print(f"  Configurations processed: {len(config_results)}")
    print(f"  Validation period: {validation_start} to {validation_end}")
    print(f"\n  Performance by configuration:")
    
    # Create a summary table
    print(f"{'Configuration':<20} {'NSE':<8} {'KGE':<8} {'KGE_NP':<8} {'r':<8} {'α':<8} {'β':<8}")
    print(f"{'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    
    for config_dir, result in config_results.items():
        name = result['name']
        metrics = result['metrics']
        print(f"{name:<20} {metrics['NSE']:7.3f} {metrics['KGE']:7.3f} {metrics['KGE_NP']:7.3f} "
              f"{metrics['r']:7.3f} {metrics['alpha']:7.3f} {metrics['beta']:7.3f}")
    
    # Find best performing configuration for each metric
    print(f"\n  Best performing configurations:")
    
    best_nse_idx = np.argmax(nse_values)
    best_kge_idx = np.argmax(kge_values)
    best_kge_np_idx = np.argmax(kge_np_values)
    best_r_idx = np.argmax(r_values)
    
    # For alpha and beta, best is closest to 1.0
    best_alpha_idx = np.argmin(np.abs(np.array(alpha_values) - 1.0))
    best_beta_idx = np.argmin(np.abs(np.array(beta_values) - 1.0))
    
    print(f"    NSE:    {config_labels[best_nse_idx]} ({nse_values[best_nse_idx]:.3f})")
    print(f"    KGE:    {config_labels[best_kge_idx]} ({kge_values[best_kge_idx]:.3f})")
    print(f"    KGE_NP: {config_labels[best_kge_np_idx]} ({kge_np_values[best_kge_np_idx]:.3f})")
    print(f"    r:      {config_labels[best_r_idx]} ({r_values[best_r_idx]:.3f})")
    print(f"    α:      {config_labels[best_alpha_idx]} ({alpha_values[best_alpha_idx]:.3f})")
    print(f"    β:      {config_labels[best_beta_idx]} ({beta_values[best_beta_idx]:.3f})")
    
    return config_results

#--------------------------------------------------------------------------------
################################### parameter ###################################
#--------------------------------------------------------------------------------

def plot_parameter_boxplots_comparison(multi_config, top_n=100):
    """
    Create boxplots for each parameter showing the distribution across multiple configurations.
    Each parameter gets its own subplot with separate boxes for each configuration.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    top_n : int
        Number of top parameter sets to analyze for each configuration
        
    Returns:
    --------
    dict
        Dictionary containing parameter data for each configuration
    """
    
    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']
    model_type = multi_config.get('model_type', 'HBV')
    
    print(f"Creating parameter boxplots comparison for {len(configs)} configurations:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Top {top_n} parameter sets per configuration")
    
    # Create plot directory
    plot_dir = create_multi_plot_dir(multi_config)
    
    # Store results for each configuration
    config_results = {}
    all_param_names = set()
    
    # Process each configuration
    for config_dir in configs:
        print(f"\nProcessing configuration: {config_dir}")
        
        individual_config = _build_individual_config(multi_config, config_dir)
        
        try:
            # Load parameter data for this configuration
            param_data = load_parameter_values(individual_config, top_n)
            if param_data is None:
                print(f"  Warning: No parameter data available for {config_dir}")
                continue
            
            config_results[config_dir] = {
                'param_data': param_data,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir)
            }
            
            # Collect all parameter names
            all_param_names.update(param_data['parameters'].keys())
            
            print(f"  ✓ Loaded {len(param_data['parameters'])} parameters with {param_data['n_sets']} parameter sets")
            
        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue
    
    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None
    
    if len(all_param_names) == 0:
        print("No parameters found across configurations")
        return None
    
    # Sort parameter names for consistent ordering
    param_names = sorted(all_param_names)
    n_params = len(param_names)
    
    print(f"\nFound {n_params} unique parameters across all configurations")
    
    # Calculate optimal subplot layout
    n_cols = int(np.ceil(np.sqrt(n_params)))
    n_rows = int(np.ceil(n_params / n_cols))
    
    print(f"Creating {n_rows}x{n_cols} subplot layout")
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
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
        
        # Collect data for this parameter from all configurations
        plot_data = []
        labels = []
        colors = []
        
        for config_dir, result in config_results.items():
            param_data = result['param_data']
            color = result['color']
            name = result['name']
            
            if param_name in param_data['parameters']:
                values = param_data['parameters'][param_name]
                plot_data.append(values)
                labels.append(name)
                colors.append(color)
            else:
                # Add empty data if parameter doesn't exist in this configuration
                plot_data.append([])
                labels.append(name)
                colors.append(color)
        
        if not any(len(data) > 0 for data in plot_data):
            # No data for this parameter in any configuration
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{param_name.replace(f"{model_type}_", "")}', fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Create boxplot with multiple boxes
        box_plot = ax.boxplot(plot_data, labels=labels, patch_artist=True)
        
        # Customize boxplot colors
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
        
        # Clean parameter name for display
        display_name = param_name.replace(f"{model_type}_", "")
        ax.set_title(f'{display_name}', fontsize=16, fontweight='bold')
        
        # Format axes
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax.set_ylabel('Parameter Value', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=13)
        
        # Rotate x-axis labels if needed
        if len(labels) > 2:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=13)
        else:
            plt.setp(ax.get_xticklabels(), fontsize=13)
    
    # Hide empty subplots
    for i in range(n_params, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # No main title - removed for better readability
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dir / f'parameter_boxplots_comparison_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved parameter boxplots comparison to: {save_path}")
    plt.show()
    
    # Print comprehensive summary
    print(f"\nParameter Boxplots Comparison Summary:")
    print(f"  Configurations processed: {len(config_results)}")
    print(f"  Total unique parameters: {n_params}")
    print(f"  Top parameter sets per config: {top_n}")
    
    # Print parameter statistics by configuration
    print(f"\nParameter Statistics by Configuration:")
    for param_name in param_names:
        display_name = param_name.replace(f"{model_type}_", "")
        print(f"\n  Parameter: {display_name}")
        
        for config_dir, result in config_results.items():
            param_data = result['param_data']
            name = result['name']
            
            if param_name in param_data['parameters'] and len(param_data['parameters'][param_name]) > 0:
                values = param_data['parameters'][param_name]
                stats = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
                print(f"    {name:20}: Mean={stats['mean']:7.3f}, Std={stats['std']:7.3f}, Range=[{stats['min']:.3f}, {stats['max']:.3f}]")
            else:
                print(f"    {name:20}: No data available")
    
    # Find parameters with highest variability across configurations
    print(f"\nParameter Variability Analysis:")
    param_variability = {}
    
    for param_name in param_names:
        config_means = []
        for config_dir, result in config_results.items():
            param_data = result['param_data']
            if param_name in param_data['parameters'] and len(param_data['parameters'][param_name]) > 0:
                config_means.append(np.mean(param_data['parameters'][param_name]))
        
        if len(config_means) > 1:
            variability = np.std(config_means) / np.mean(config_means) if np.mean(config_means) != 0 else 0
            param_variability[param_name] = variability
    
    if param_variability:
        # Sort by variability (coefficient of variation)
        sorted_params = sorted(param_variability.items(), key=lambda x: x[1], reverse=True)
        print(f"  Most variable parameters across configurations (by coefficient of variation):")
        for param_name, cv in sorted_params[:5]:  # Top 5 most variable
            display_name = param_name.replace(f"{model_type}_", "")
            print(f"    {display_name:20}: CV = {cv:.3f}")
    
    return config_results

#--------------------------------------------------------------------------------
################################### storage ####################################
#--------------------------------------------------------------------------------

def plot_storage_timeseries_comparison(multi_config, validation_start=None, validation_end=None):
    """
    Plot watershed storage components comparison across multiple configurations.
    Each storage component gets its own subplot with all configurations overlaid.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    dict
        Dictionary containing storage data for each configuration
    """
    
    # Use dates from multi_config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']
    model_type = multi_config.get('model_type', 'HBV')
    
    print(f"Creating storage timeseries comparison for {len(configs)} configurations:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Validation period: {validation_start} to {validation_end}")
    
    # Create plot directory
    plot_dir = create_multi_plot_dir(multi_config)
    
    # Store results for each configuration
    config_results = {}
    all_storage_cols = set()
    
    # Process each configuration
    for config_dir in configs:
        print(f"\nProcessing configuration: {config_dir}")
        
        individual_config = _build_individual_config(multi_config, config_dir)
        
        try:
            # Load storage data for this configuration
            storage_df = load_storage_data(individual_config)
            if storage_df is None:
                print(f"  Warning: No storage data available for {config_dir}")
                continue
            
            # Filter by validation period
            validation_start_dt = pd.to_datetime(validation_start)
            validation_end_dt = pd.to_datetime(validation_end)
            
            val_mask = (storage_df['date'] >= validation_start_dt) & (storage_df['date'] <= validation_end_dt)
            storage_filtered = storage_df[val_mask].copy()
            
            if len(storage_filtered) == 0:
                print(f"  Warning: No storage data found for validation period in {config_dir}")
                continue
            
            # Get storage columns (exclude date, month, year)
            storage_cols = [col for col in storage_filtered.columns if col not in ['date', 'month', 'year']]
            
            if len(storage_cols) == 0:
                print(f"  Warning: No storage columns found in {config_dir}")
                continue
            
            config_results[config_dir] = {
                'storage_data': storage_filtered,
                'storage_cols': storage_cols,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir)
            }
            
            # Collect all storage column names
            all_storage_cols.update(storage_cols)
            
            print(f"  ✓ Loaded storage data: {len(storage_filtered)} records, {len(storage_cols)} components")
            
        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue
    
    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None
    
    if len(all_storage_cols) == 0:
        print("No storage components found across configurations")
        return None
    
    # Sort storage columns for consistent ordering
    storage_cols = sorted(all_storage_cols)
    n_storage = len(storage_cols)
    
    print(f"\nFound {n_storage} unique storage components across all configurations")
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_storage, 1, figsize=(16, 3.5*n_storage), sharex=True)
    
    if n_storage == 1:
        axes = [axes]  # Make it iterable
    
    # Define colors for different storage types (background styling)
    storage_colors = {
        'snowfall': 'skyblue',
        'rainfall': 'navy',
        'snow storage': 'lightcyan',
        'soil': 'brown',
        'groundwater': 'blue',
        'depression': 'lightblue',
        'ponded': 'cyan',
        'fast': 'orange',
        'slow': 'darkblue'
    }
    
    # Plot each storage component
    for i, storage_col in enumerate(storage_cols):
        ax = axes[i]
        
        # Determine background color based on column name
        bg_color = None
        for key, color in storage_colors.items():
            if key in storage_col.lower():
                bg_color = color
                break
        
        # Track if this is a precipitation component for special handling
        is_precipitation = 'snowfall' in storage_col.lower() or 'rainfall' in storage_col.lower()
        
        # Plot each configuration for this storage component
        for config_dir, result in config_results.items():
            storage_data = result['storage_data']
            color = result['color']
            name = result['name']
            
            if storage_col in storage_data.columns:
                data = storage_data[storage_col]
                
                if is_precipitation and '[mm/d]' in storage_col:
                    # For precipitation, use filled areas with config-specific colors but transparent
                    ax.fill_between(storage_data['date'], 0, data, 
                                   color=color, alpha=0.4, label=name)
                elif 'snow storage' in storage_col.lower():
                    # For snow storage, use filled areas
                    ax.fill_between(storage_data['date'], 0, data, 
                                   color=color, alpha=0.3, label=name)
                else:
                    # For other storage components, use line plots
                    ax.plot(storage_data['date'], data, 
                           color=color, linewidth=2, label=name, alpha=0.8)
        
        # Clean up column name for title
        clean_title = storage_col.replace('[mm]', '(mm)').replace('[mm/d]', '(mm/d)')
        ax.set_title(f'{clean_title}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Storage (mm)', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend only to the first subplot
        if i == 0:
            ax.legend(loc='best', fontsize=10)
        
        # Add summary statistics for each configuration
        stats_text = ""
        for config_dir, result in config_results.items():
            storage_data = result['storage_data']
            name = result['name']
            
            if storage_col in storage_data.columns:
                mean_val = storage_data[storage_col].mean()
                max_val = storage_data[storage_col].max()
                stats_text += f"{name}: μ={mean_val:.1f}, max={max_val:.1f} mm\n"
        
        if stats_text:
            ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Format x-axis
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    
    # Set x-label only on bottom subplot
    axes[-1].set_xlabel('Date', fontsize=12)
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add overall title
    fig.suptitle(f'Watershed Storage Components Comparison - Catchment {gauge_id}\n'
                f'Validation Period: {validation_start} to {validation_end}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.3)
    
    # Save plot
    save_path = plot_dir / f'storage_timeseries_comparison_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved storage timeseries comparison plot to: {save_path}")
    plt.show()
    
    # Print comprehensive summary
    print(f"\nStorage Timeseries Comparison Summary:")
    print(f"  Configurations processed: {len(config_results)}")
    print(f"  Validation period: {validation_start} to {validation_end}")
    print(f"  Storage components: {n_storage}")
    
    # Print detailed statistics by storage component and configuration
    print(f"\nStorage Component Statistics by Configuration:")
    for storage_col in storage_cols:
        clean_name = storage_col.replace('[mm]', '').replace('[mm/d]', '').strip()
        print(f"\n  {clean_name}:")
        
        for config_dir, result in config_results.items():
            storage_data = result['storage_data']
            name = result['name']
            
            if storage_col in storage_data.columns:
                data = storage_data[storage_col]
                stats = {
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max()
                }
                print(f"    {name:20}: Mean={stats['mean']:7.1f} mm, Std={stats['std']:7.1f} mm, Range=[{stats['min']:.1f}, {stats['max']:.1f}] mm")
            else:
                print(f"    {name:20}: No data available")
    
    # Analyze differences between configurations
    print(f"\nStorage Component Variability Analysis:")
    component_variability = {}
    
    for storage_col in storage_cols:
        config_means = []
        config_names_list = []
        
        for config_dir, result in config_results.items():
            storage_data = result['storage_data']
            name = result['name']
            
            if storage_col in storage_data.columns:
                config_means.append(storage_data[storage_col].mean())
                config_names_list.append(name)
        
        if len(config_means) > 1:
            variability = np.std(config_means) / np.mean(config_means) if np.mean(config_means) != 0 else 0
            component_variability[storage_col] = variability
    
    if component_variability:
        # Sort by variability (coefficient of variation)
        sorted_components = sorted(component_variability.items(), key=lambda x: x[1], reverse=True)
        print(f"  Most variable storage components across configurations (by coefficient of variation):")
        for storage_col, cv in sorted_components[:5]:  # Top 5 most variable
            clean_name = storage_col.replace('[mm]', '').replace('[mm/d]', '').strip()
            print(f"    {clean_name:30}: CV = {cv:.3f}")
    
    return config_results

#--------------------------------------------------------------------------------
############################### contributions ###################################
#--------------------------------------------------------------------------------

def plot_streamflow_glogem_snowmelt_regime_subplots(multi_config, validation_start=None, validation_end=None, unit='mm'):
    """
    Plot streamflow regime with GloGEM ice melt and total snowmelt for multiple configurations.
    Each configuration gets its own subplot showing stacked contributions.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary containing:
        - 'main_dir': main directory path
        - 'gauge_id': gauge identifier
        - 'configs': list of configuration directory names
        - 'config_colors': dict mapping config names to colors
        - 'config_names': dict mapping config names to display names
        - 'model_type': model type (default 'HBV')
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    unit : str, optional
        Unit for display ('mm' for mm/day, 'm3' for m³/s), default is 'mm'
        
    Returns:
    --------
    dict
        Dictionary containing regime data for each configuration
    """
    
    # Use dates from multi_config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']
    model_type = multi_config.get('model_type', 'HBV')
    
    print(f"Creating streamflow regime with GloGEM and snowmelt subplots for {len(configs)} configurations:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Validation period: {validation_start} to {validation_end}")
    print(f"  - Unit: {unit}")
    
    # Create plot directory
    plot_dir = create_multi_plot_dir(multi_config)
    
    # Set unit label
    unit_label = 'mm/day' if unit == 'mm' else 'm³/s'
    
    # Store results for each configuration
    config_results = {}
    obs_data = None  # Store observed streamflow (should be same for all configs)
    
    # Process each configuration
    for config_dir in configs:
        print(f"\n{'='*60}")
        print(f"Processing: {config_names.get(config_dir, config_dir)}")
        print(f"{'='*60}")
        
        individual_config = _build_individual_config(multi_config, config_dir)
        
        try:
            # Load observed streamflow data (same for all configs)
            if obs_data is None:
                streamflow_data = load_hydrograph_data(individual_config)
                if streamflow_data is not None and 'obs_Q' in streamflow_data.columns:
                    # Filter for validation period
                    val_start = pd.to_datetime(validation_start)
                    val_end = pd.to_datetime(validation_end)
                    val_mask = (streamflow_data['date'] >= val_start) & (streamflow_data['date'] <= val_end)
                    obs_data = streamflow_data[val_mask].copy()
                    
                    # Convert to mm/day if needed
                    if unit == 'mm':
                        paths = get_paths(individual_config)
                        topo_dir = paths['topo_dir']
                        hru_shapefile = topo_dir / "HRU.shp"
                        if hru_shapefile.exists():
                            import geopandas as gpd
                            hru_gdf = gpd.read_file(hru_shapefile)
                            total_area_km2 = hru_gdf['Area_km2'].sum()
                            conversion = 86400 / (total_area_km2 * 1000000) * 1000
                            obs_data['obs_Q_converted'] = obs_data['obs_Q'] * conversion

            # Load simulated streamflow
            streamflow_data = load_hydrograph_data(individual_config)
            if streamflow_data is None:
                print(f"  ⚠️  Could not load streamflow data for {config_dir}")
                continue
            
            # Filter for validation period
            val_start = pd.to_datetime(validation_start)
            val_end = pd.to_datetime(validation_end)
            val_mask = (streamflow_data['date'] >= val_start) & (streamflow_data['date'] <= val_end)
            sim_data = streamflow_data[val_mask].copy()
            
            # Convert simulated to mm/day if needed
            if unit == 'mm':
                paths = get_paths(individual_config)
                topo_dir = paths['topo_dir']
                hru_shapefile = topo_dir / "HRU.shp"
                if hru_shapefile.exists():
                    import geopandas as gpd
                    hru_gdf = gpd.read_file(hru_shapefile)
                    total_area_km2 = hru_gdf['Area_km2'].sum()
                    conversion = 86400 / (total_area_km2 * 1000000) * 1000
                    sim_data['sim_Q_converted'] = sim_data['sim_Q'] * conversion
                else:
                    sim_data['sim_Q_converted'] = sim_data['sim_Q']
            else:
                sim_data['sim_Q_converted'] = sim_data['sim_Q']
            
            # Load snowmelt mass loadings - EXACT SAME AS YOUR WORKING FUNCTION
            print(f"\n  - Loading HBV snowmelt mass loadings...")
            hbv_snowmelt_df = load_snowmelt_mass_loadings(individual_config, validation_start, validation_end, unit=unit)
            
            if hbv_snowmelt_df is None:
                print(f"  ⚠️  Could not load snowmelt data for {config_dir}")
                continue
            
            # Determine snowmelt column based on unit - EXACT SAME AS YOUR WORKING FUNCTION
            snowmelt_col = 'snowmelt_mm_day' if unit == 'mm' and 'snowmelt_mm_day' in hbv_snowmelt_df.columns else 'snowmelt_m3s'
            
            print(f"  - Using snowmelt column: {snowmelt_col}")
            
            # Add month column for aggregation
            hbv_snowmelt_df['month'] = hbv_snowmelt_df['date'].dt.month
            sim_data['month'] = sim_data['date'].dt.month
            
            # ========================================
            # COUPLED vs NON-COUPLED CONFIGURATIONS
            # ========================================
            
            if individual_config['coupled']:
                # COUPLED: Use GloGEM for ice melt and snowmelt on glaciers
                print(f"  - Loading GloGEM data (coupled configuration)...")
                glogem_data = load_glogem_data(individual_config, unit=unit, plot=False)
                if glogem_data is None:
                    print(f"  ⚠️  Could not load GloGEM data for {config_dir}")
                    continue

                # Filter GloGEM for validation period
                glogem_mask = (glogem_data['date'] >= val_start) & (glogem_data['date'] <= val_end)
                glogem_filtered = glogem_data[glogem_mask].copy()

                # Check which GloGEM columns are available
                if 'snowmelt' not in glogem_filtered.columns or 'icemelt' not in glogem_filtered.columns:
                    print(f"  ⚠️  GloGEM data missing required columns for {config_dir}")
                    print(f"     Available columns: {glogem_filtered.columns.tolist()}")
                    continue

                # Calculate monthly regime for each component
                glogem_filtered['month'] = glogem_filtered['date'].dt.month

                # Use NORMALIZED (catchment area) values for GloGEM
                glacier_icemelt_regime = glogem_filtered.groupby('month')['icemelt_normalized'].mean()
                glogem_snowmelt_regime = glogem_filtered.groupby('month')['snowmelt_normalized'].mean()
                hbv_snowmelt_regime = hbv_snowmelt_df.groupby('month')[snowmelt_col].mean()
                sim_regime = sim_data.groupby('month')['sim_Q_converted'].mean()

                # Check if this is an icemelt-mode config
                is_icemelt = multi_config.get('config_icemelt_mode', {}).get(config_dir, False)

                if is_icemelt:
                    # ICEMELT MODE: glacier HRUs are ROCK, so Raven already simulates
                    # snowmelt on glacier areas. Do NOT add GloGEM snowmelt (would double-count).
                    total_snowmelt_regime = hbv_snowmelt_regime
                    print(f"  - Icemelt mode: using HBV snowmelt only (includes glacier-ROCK HRUs)")
                else:
                    # STANDARD COUPLED: glacier HRUs are MASKED_GLACIER, Raven does not
                    # simulate snowmelt there. Add GloGEM snowmelt for glacier areas.
                    total_snowmelt_regime = glogem_snowmelt_regime.add(hbv_snowmelt_regime, fill_value=0)

                print(f"  - GloGEM ice melt mean: {glacier_icemelt_regime.mean():.4f} {unit_label}")
                print(f"  - GloGEM snowmelt mean: {glogem_snowmelt_regime.mean():.4f} {unit_label}")
                print(f"  - HBV snowmelt mean: {hbv_snowmelt_regime.mean():.4f} {unit_label}")
                print(f"  - Total snowmelt mean: {total_snowmelt_regime.mean():.4f} {unit_label}")
                
            else:
                # NON-COUPLED: Use mass loadings files for both glacier melt and snowmelt
                print(f"  - Loading glacier melt mass loadings (non-coupled configuration)...")
                
                # Load glacier melt mass loadings - SAME STRUCTURE AS HYDROGRAPHS (has gauge_id column)
                paths = get_paths(individual_config)
                glacier_file = paths['output_dir'] / f"{gauge_id}_{model_type}_GLACIERMELT_ALLMassLoadings.csv"
                
                if not glacier_file.exists():
                    print(f"  ⚠️  Glacier melt file not found: {glacier_file}")
                    continue
                
                try:
                    glacier_df = pd.read_csv(glacier_file)
                    glacier_df['date'] = pd.to_datetime(glacier_df['date'])
                    
                    # Filter for validation period
                    glacier_mask = (glacier_df['date'] >= val_start) & (glacier_df['date'] <= val_end)
                    glacier_filtered = glacier_df[glacier_mask].copy()
                    
                    # The glacier melt file has the same structure as hydrographs: gauge_id column in m³/s
                    # Column name is like '0118 m3/s'
                    glacier_m3s_col = f"{gauge_id} m3/s"
                    
                    if glacier_m3s_col not in glacier_filtered.columns:
                        print(f"  ⚠️  Expected column '{glacier_m3s_col}' not found in glacier melt data")
                        print(f"     Available columns: {glacier_filtered.columns.tolist()}")
                        continue
                    
                    # Convert to mm/day if needed (same as for streamflow)
                    if unit == 'mm':
                        # Use the same conversion factor as for streamflow
                        paths = get_paths(individual_config)
                        topo_dir = paths['topo_dir']
                        hru_shapefile = topo_dir / "HRU.shp"
                        if hru_shapefile.exists():
                            import geopandas as gpd
                            hru_gdf = gpd.read_file(hru_shapefile)
                            total_area_km2 = hru_gdf['Area_km2'].sum()
                            conversion = 86400 / (total_area_km2 * 1000000) * 1000
                            glacier_filtered['glacier_melt_converted'] = glacier_filtered[glacier_m3s_col] * conversion
                        else:
                            glacier_filtered['glacier_melt_converted'] = glacier_filtered[glacier_m3s_col]
                    else:
                        glacier_filtered['glacier_melt_converted'] = glacier_filtered[glacier_m3s_col]
                    
                    glacier_filtered['month'] = glacier_filtered['date'].dt.month
                    
                    # Calculate monthly regimes
                    glacier_icemelt_regime = glacier_filtered.groupby('month')['glacier_melt_converted'].mean()
                    hbv_snowmelt_regime = hbv_snowmelt_df.groupby('month')[snowmelt_col].mean()
                    sim_regime = sim_data.groupby('month')['sim_Q_converted'].mean()
                    
                    # For non-coupled, total snowmelt is just HBV snowmelt (no GloGEM snowmelt component)
                    total_snowmelt_regime = hbv_snowmelt_regime
                    
                    print(f"  - Glacier ice melt mean: {glacier_icemelt_regime.mean():.4f} {unit_label}")
                    print(f"  - HBV snowmelt mean: {hbv_snowmelt_regime.mean():.4f} {unit_label}")
                    print(f"  - Total snowmelt mean: {total_snowmelt_regime.mean():.4f} {unit_label}")
                    
                except Exception as e:
                    print(f"  ⚠️  Error loading glacier melt data: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Create monthly regime DataFrame (same structure for both coupled and non-coupled)
            monthly_regime = pd.DataFrame({
                'month': range(1, 13),
                'sim_Q_converted': [sim_regime.get(m, 0) for m in range(1, 13)],
                'total_snowmelt': [total_snowmelt_regime.get(m, 0) for m in range(1, 13)],
                'glacier_icemelt': [glacier_icemelt_regime.get(m, 0) for m in range(1, 13)]
            })
            
            config_results[config_dir] = {
                'monthly_regime': monthly_regime,
                'name': config_names.get(config_dir, config_dir),
                'color': config_colors.get(config_dir, 'gray'),
                'coupled': individual_config['coupled']
            }
            
            print(f"  ✅ Successfully processed {config_names.get(config_dir, config_dir)}")
            
        except Exception as e:
            print(f"  ⚠️  Error processing {config_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(config_results) == 0:
        print("\n❌ No configurations were successfully processed")
        return None
    
    # ===================================
    # CREATE COMPARISON PLOT
    # ===================================
    
    # Calculate subplot layout
    n_configs = len(config_results)
    if n_configs <= 2:
        n_rows, n_cols = 1, n_configs
        figsize = (10*n_configs, 7)
    elif n_configs <= 4:
        n_rows, n_cols = 2, 2
        figsize = (16, 12)
    elif n_configs <= 6:
        n_rows, n_cols = 2, 3
        figsize = (20, 12)
    elif n_configs <= 9:
        n_rows, n_cols = 3, 3
        figsize = (20, 16)
    elif n_configs <= 12:
        n_rows, n_cols = 3, 4
        figsize = (22, 16)
    else:
        n_cols = 4
        n_rows = (n_configs + n_cols - 1) // n_cols
        figsize = (22, 5 * n_rows)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)

    # Handle single subplot case
    if n_configs == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot each configuration in its own subplot
    for i, (config_dir, result) in enumerate(config_results.items()):
        ax = axes[i]
        monthly = result['monthly_regime']
        config_name = result['name']
        is_coupled = result['coupled']
        config_color = result['color']  # Get config-specific color
        
        # EXACT SAME AS YOUR WORKING FUNCTION - Plot filled polygons FIRST (bottom layer, in order of magnitude)
        # Plot total snowmelt as filled polygon - light blue
        # Label changes based on coupled vs non-coupled
        snowmelt_label = 'Total Snowmelt (GloGEM+HBV)' if is_coupled else 'Total Snowmelt (HBV)'
        ax.fill_between(monthly['month'], 0, monthly['total_snowmelt'], 
                        color='#B3D9FF', alpha=0.7, label=snowmelt_label, 
                        zorder=1, edgecolor='#6DB3F2', linewidth=1.5)
        
        # Plot glacier ice melt as filled polygon - GREY FOR ALL PANELS
        ax.fill_between(monthly['month'], 0, monthly['glacier_icemelt'], 
                        color='#A9A9A9', alpha=0.6, label='Glacier Ice Melt', 
                        zorder=2, edgecolor='#696969', linewidth=1.5)
        
        # Plot observed streamflow (line without markers) - EXACT SAME AS YOUR WORKING FUNCTION
        if obs_data is not None and 'obs_Q_converted' in obs_data.columns:
            obs_data['month'] = pd.to_datetime(obs_data['date']).dt.month
            obs_monthly = obs_data.groupby('month')['obs_Q_converted'].mean()
            ax.plot(obs_monthly.index, obs_monthly.values, 'k-', 
                   linewidth=3, label='Observed Streamflow', zorder=4)
        
        # Plot simulated streamflow (dashed line) - USE CONFIG-SPECIFIC COLOR
        ax.plot(monthly['month'], monthly['sim_Q_converted'], '--', 
               color=config_color, linewidth=2.5, label='Simulated Streamflow', zorder=3)
        
        # Formatting
        ax.set_title(config_name, fontsize=18, fontweight='bold')  # Bigger panel titles
        ax.set_xticks(months)
        ax.set_xticklabels(month_names, rotation=0, fontsize=14)  # Bigger tick labels
        ax.tick_params(axis='y', labelsize=14)  # Bigger y-axis tick labels
        ax.grid(True, alpha=0.3, zorder=0)
        
        # Only show legend on first subplot
        if i == 0:
            ax.legend(loc='best', fontsize=13, framealpha=0.9)  # Bigger legend font
        
        # Only show y-label on leftmost subplots
        if i % n_cols == 0:
            ax.set_ylabel(f'Discharge ({unit_label})', fontsize=16, fontweight='bold')  # Bigger y-axis label
    
    # Hide unused subplots
    for i in range(n_configs, len(axes)):
        axes[i].axis('off')
    
    # Add common x-label
    fig.text(0.5, 0.02, 'Month', ha='center', fontsize=18, fontweight='bold')  # Bigger x-axis label
    
    # NO MAIN TITLE - removed as requested
    
    plt.tight_layout(rect=[0, 0.04, 1, 1.0])  # Adjusted since no title
    
    # Save plot
    save_path = plot_dir / f'streamflow_glogem_snowmelt_regime_subplots_{unit}_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved streamflow regime subplots to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"STREAMFLOW REGIME SUBPLOTS SUMMARY")
    print(f"{'='*60}")
    print(f"Catchment: {gauge_id}")
    print(f"Period: {validation_start} to {validation_end}")
    print(f"Unit: {unit_label}")
    print(f"Configurations processed: {len(config_results)}")
    print(f"Layout: {n_rows} rows × {n_cols} columns")
    
    for config_dir, result in config_results.items():
        monthly = result['monthly_regime']
        print(f"\n  {result['name']}:")
        print(f"    Mean glacier ice melt: {monthly['glacier_icemelt'].mean():.2f} {unit_label}")
        print(f"    Mean total snowmelt: {monthly['total_snowmelt'].mean():.2f} {unit_label}")
        print(f"    Mean simulated Q: {monthly['sim_Q_converted'].mean():.2f} {unit_label}")
    
    print(f"{'='*60}\n")
    
    return config_results


#--------------------------------------------------------------------------------
#################################### forcing ####################################
#--------------------------------------------------------------------------------

def plot_glacier_hru_temperatures_comparison(multi_config, validation_start=None, validation_end=None):
    """
    Plot glacier HRU temperatures comparison across multiple configurations.
    Identifies glacier HRUs (landuse class 7) and plots their temperatures for each configuration.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    dict
        Dictionary containing glacier temperature data for each configuration
    """
    
    # Use dates from multi_config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']
    model_type = multi_config.get('model_type', 'HBV')
    
    print(f"Creating glacier HRU temperature comparison for {len(configs)} configurations:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Validation period: {validation_start} to {validation_end}")
    
    # Create plot directory
    plot_dir = create_multi_plot_dir(multi_config)
    
    # Store results for each configuration
    config_results = {}
    glacier_hru_ids = None  # Will be determined from first successful config
    
    # Process each configuration
    for config_dir in configs:
        print(f"\nProcessing configuration: {config_dir}")
        
        individual_config = _build_individual_config(multi_config, config_dir)
        paths = get_paths(individual_config)

        try:
            # Load HRU shapefile to identify glacier HRUs
            topo_dir = paths['topo_dir']
            hru_shp_path = topo_dir / "HRU.shp"

            if not hru_shp_path.exists():
                print(f"  ❌ HRU shapefile not found: {hru_shp_path}")
                continue

            # Load shapefile and find glacier HRUs (only need to do this once)
            if glacier_hru_ids is None:
                import geopandas as gpd
                hru_gdf = gpd.read_file(hru_shp_path)

                # Find glacier HRUs (landuse class 7)
                glacier_hrus = hru_gdf[hru_gdf['Landuse_Cl'] == 7]
                if len(glacier_hrus) == 0:
                    print(f"  ⚠️  No glacier HRUs found (landuse class 7) in shapefile")
                    continue

                # Get HRU IDs - try different possible column names
                if 'HRU_ID' in hru_gdf.columns:
                    glacier_hru_ids = glacier_hrus['HRU_ID'].tolist()
                elif 'OBJECTID' in hru_gdf.columns:
                    glacier_hru_ids = glacier_hrus['OBJECTID'].tolist()
                else:
                    # Use index + 1 (assuming 1-based HRU numbering)
                    glacier_hru_ids = [idx + 1 for idx in glacier_hrus.index.tolist()]

                print(f"  📍 Found {len(glacier_hru_ids)} glacier HRUs: {glacier_hru_ids}")

            # Load temperature data
            output_dir = paths['output_dir']
            temp_csv_path = output_dir / f"{gauge_id}_{model_type}_TEMP_AVE_Daily_Average_ByHRU.csv"
            
            if not temp_csv_path.exists():
                print(f"  ❌ Temperature file not found: {temp_csv_path}")
                continue
            
            print(f"  📁 Loading temperature data: {temp_csv_path}")
            
            # Read temperature data with CORRECT header handling for your CSV structure
            # Row 1: HRU numbers (column headers)
            # Row 2: 'time', 'day', 'mean', 'mean', etc. (descriptive headers)
            # Row 3+: Actual data
            
            # First, read the HRU numbers from row 1
            hru_numbers_df = pd.read_csv(temp_csv_path, nrows=1, header=None)
            hru_column_names = hru_numbers_df.iloc[0].tolist()
            
            # Then, read the actual data starting from row 3 (skip rows 0 and 1)
            temp_df = pd.read_csv(temp_csv_path, skiprows=2, header=None)
            temp_df.columns = hru_column_names
            
            print(f"  📊 CSV structure - HRU columns: {hru_column_names[:5]}... (showing first 5)")
            print(f"  📊 Data shape: {temp_df.shape}")
            
            # The first column should be the date/time column
            date_col = temp_df.columns[1]
            
            # Convert the first column to datetime if possible
            try:
                temp_df[date_col] = pd.to_datetime(temp_df[date_col])
                print(f"  ✅ Successfully converted {date_col} to datetime")
            except Exception as e:
                print(f"  ⚠️  Could not convert {date_col} to datetime: {e}")
                # Try creating a date range if conversion fails
                temp_df[date_col] = pd.date_range(start='2000-01-01', periods=len(temp_df), freq='D')
            
            # Filter for validation period
            if temp_df[date_col].dtype == 'datetime64[ns]':
                validation_start_dt = pd.to_datetime(validation_start)
                validation_end_dt = pd.to_datetime(validation_end)
                
                mask = (temp_df[date_col] >= validation_start_dt) & (temp_df[date_col] <= validation_end_dt)
                temp_filtered = temp_df[mask].copy()
            else:
                temp_filtered = temp_df.copy()
            
            if len(temp_filtered) == 0:
                print(f"  ⚠️  No temperature data found for validation period")
                continue
            
            print(f"  📅 Filtered to {len(temp_filtered)} records for validation period")
            
            # Find temperature columns for glacier HRUs
            glacier_temp_data = {}
            found_hrus = []
            
            for hru_id in glacier_hru_ids:
                # Look for column that matches this HRU ID
                hru_col = None
                for col in temp_df.columns[2:]:  # Skip the date column (first column)
                    try:
                        # Convert column name to string and compare
                        col_str = str(col).strip()
                        hru_id_str = str(hru_id).strip()
                        
                        # Try exact match first
                        if col_str == hru_id_str:
                            hru_col = col
                            break
                        # Try with integer conversion
                        elif col_str.isdigit() and hru_id_str.isdigit():
                            if int(col_str) == int(hru_id_str):
                                hru_col = col
                                break
                    except:
                        continue
                
                if hru_col is not None:
                    # Convert temperature data to numeric, handling any non-numeric values
                    temp_data = pd.to_numeric(temp_filtered[hru_col], errors='coerce')
                    glacier_temp_data[hru_id] = temp_data.values
                    found_hrus.append(hru_id)
                    print(f"    ✅ Found temperature data for HRU {hru_id} in column '{hru_col}'")
                else:
                    print(f"    ⚠️  Could not find temperature data for HRU {hru_id}")
                    print(f"        Available columns: {list(temp_df.columns[1:6])}... (showing first 5)")
            
            if len(glacier_temp_data) == 0:
                print(f"  ❌ No glacier HRU temperature data found for {config_dir}")
                print(f"      Glacier HRUs needed: {glacier_hru_ids}")
                print(f"      Available columns: {list(temp_df.columns)}")
                continue
            
            # Check for winter melting conditions
            winter_melting_days = 0
            if temp_filtered[date_col].dtype == 'datetime64[ns]':
                winter_months = [12, 1, 2]  # December, January, February
                winter_mask = temp_filtered[date_col].dt.month.isin(winter_months)
                
                for hru_id, temps in glacier_temp_data.items():
                    winter_temps = temps[winter_mask]
                    # Remove NaN values before counting positive temperatures
                    valid_winter_temps = winter_temps[~pd.isna(winter_temps)]
                    positive_winter_days = (valid_winter_temps > 0).sum()
                    winter_melting_days += positive_winter_days
                    
                    if positive_winter_days > 0:
                        print(f"    ⚠️ HRU {hru_id}: {positive_winter_days} days with T > 0°C in winter")
            
            config_results[config_dir] = {
                'temp_data': temp_filtered,
                'glacier_temp_data': glacier_temp_data,
                'found_hrus': found_hrus,
                'winter_melting_days': winter_melting_days,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir),
                'date_col': date_col
            }
            
            print(f"  ✅ Processed temperature data: {len(temp_filtered)} records for {len(found_hrus)} glacier HRUs")
            if winter_melting_days > 0:
                print(f"  ❄️ Warning: {winter_melting_days} total winter melting days found across all glacier HRUs")
            
        except Exception as e:
            print(f"  ❌ Error processing {config_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(config_results) == 0:
        print("❌ No configurations processed successfully")
        return None
    
    if glacier_hru_ids is None:
        print("❌ No glacier HRUs identified")
        return None
    
    # Create plots
    n_hrus = len(glacier_hru_ids)
    
    # 1. Time series plot - all glacier HRUs for all configurations
    plt.figure(figsize=(16, 10))
    
    for config_dir, result in config_results.items():
        temp_data = result['temp_data']
        glacier_temp_data = result['glacier_temp_data']
        config_color = result['color']
        config_name = result['name']
        date_col = result['date_col']
        
        for i, hru_id in enumerate(glacier_hru_ids):
            if hru_id in glacier_temp_data:
                # Use config-specific alpha and line style
                alpha = 0.7
                linewidth = 1.5
                
                # Create label only for first HRU of each config
                if i == 0:
                    label = config_name
                else:
                    label = None
                
                plt.plot(temp_data[date_col], glacier_temp_data[hru_id], 
                        color=config_color, alpha=alpha, linewidth=linewidth, 
                        label=label)
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Freezing point')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Temperature (°C)', fontsize=14)
    plt.title(f'Glacier HRU Temperatures Comparison - Catchment {gauge_id}\n'
             f'Validation Period: {validation_start} to {validation_end}', 
             fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save time series plot
    save_path = plot_dir / f'glacier_hru_temperatures_timeseries_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved glacier HRU temperatures time series to: {save_path}")
    plt.show()
    
    # 2. Box plot comparison - temperature distribution by configuration
    plt.figure(figsize=(14, 8))
    
    # Prepare data for box plots
    box_data = []
    box_labels = []
    box_colors = []
    
    for config_dir, result in config_results.items():
        glacier_temp_data = result['glacier_temp_data']
        config_name = result['name']
        config_color = result['color']
        
        # Combine all glacier HRU temperatures for this configuration
        all_temps = []
        for hru_id, temps in glacier_temp_data.items():
            valid_temps = temps[~pd.isna(temps)]  # Remove NaN values
            all_temps.extend(valid_temps)
        
        if len(all_temps) > 0:
            box_data.append(all_temps)
            box_labels.append(config_name)
            box_colors.append(config_color)
    
    if box_data:
        box_plot = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Customize box colors
        for patch, color in zip(box_plot['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Freezing point')
    plt.ylabel('Temperature (°C)', fontsize=14)
    plt.title(f'Glacier HRU Temperature Distribution Comparison - Catchment {gauge_id}', 
             fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save box plot
    save_path = plot_dir / f'glacier_hru_temperatures_boxplot_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved glacier HRU temperatures box plot to: {save_path}")
    plt.show()
    
    # 3. Monthly regime comparison
    plt.figure(figsize=(14, 8))
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for config_dir, result in config_results.items():
        temp_data = result['temp_data']
        glacier_temp_data = result['glacier_temp_data']
        config_color = result['color']
        config_name = result['name']
        date_col = result['date_col']
        
        # Calculate monthly means across all glacier HRUs
        if temp_data[date_col].dtype == 'datetime64[ns]':
            temp_data_copy = temp_data.copy()
            temp_data_copy['month'] = temp_data_copy[date_col].dt.month
            
            monthly_means = []
            for month in range(1, 13):
                month_mask = temp_data_copy['month'] == month
                month_temps = []
                
                for hru_id, temps in glacier_temp_data.items():
                    month_temps_hru = temps[month_mask]
                    valid_month_temps = month_temps_hru[~pd.isna(month_temps_hru)]
                    month_temps.extend(valid_month_temps)
                
                if month_temps:
                    monthly_means.append(np.mean(month_temps))
                else:
                    monthly_means.append(np.nan)
            
            plt.plot(range(1, 13), monthly_means, marker='o', linewidth=2.5, 
                    color=config_color, label=config_name)
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Freezing point')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Mean Temperature (°C)', fontsize=14)
    plt.title(f'Monthly Temperature Regime - Glacier HRUs - Catchment {gauge_id}', 
             fontsize=16, fontweight='bold')
    plt.xticks(range(1, 13), months)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save monthly regime plot
    save_path = plot_dir / f'glacier_hru_temperatures_monthly_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved glacier HRU temperatures monthly regime to: {save_path}")
    plt.show()
    
    # Print comprehensive summary
    print(f"\n📊 Glacier HRU Temperature Analysis Summary:")
    print(f"  Configurations processed: {len(config_results)}")
    print(f"  Validation period: {validation_start} to {validation_end}")
    print(f"  Total glacier HRUs: {len(glacier_hru_ids)}")
    print(f"  Glacier HRU IDs: {glacier_hru_ids}")
    
    print(f"\n  Temperature Statistics by Configuration:")
    for config_dir, result in config_results.items():
        glacier_temp_data = result['glacier_temp_data']
        config_name = result['name']
        found_hrus = result['found_hrus']
        winter_melting_days = result['winter_melting_days']
        
        # Calculate overall statistics
        all_temps = []
        for hru_id, temps in glacier_temp_data.items():
            valid_temps = temps[~pd.isna(temps)]
            all_temps.extend(valid_temps)
        
        if all_temps:
            mean_temp = np.mean(all_temps)
            min_temp = np.min(all_temps)
            max_temp = np.max(all_temps)
            positive_days = sum(1 for t in all_temps if t > 0)
            total_days = len(all_temps)
            
            print(f"\n    {config_name}:")
            print(f"      HRUs found: {len(found_hrus)}/{len(glacier_hru_ids)}")
            print(f"      Temperature range: {min_temp:.1f}°C to {max_temp:.1f}°C")
            print(f"      Mean temperature: {mean_temp:.1f}°C")
            print(f"      Days above freezing: {positive_days}/{total_days} ({positive_days/total_days*100:.1f}%)")
            if winter_melting_days > 0:
                print(f"      ❄️ Winter melting days: {winter_melting_days}")
            
            # Individual HRU statistics
            print(f"      Individual HRU stats:")
            for hru_id in found_hrus:
                temps = glacier_temp_data[hru_id]
                valid_temps = temps[~pd.isna(temps)]
                if len(valid_temps) > 0:
                    hru_mean = np.mean(valid_temps)
                    hru_positive = sum(1 for t in valid_temps if t > 0)
                    print(f"        HRU {hru_id}: Mean={hru_mean:.1f}°C, Positive days={hru_positive}")
    
    # Warn about potential issues
    total_winter_melting = sum(result['winter_melting_days'] for result in config_results.values())
    if total_winter_melting > 0:
        print(f"\n  ⚠️ WARNING: Total winter melting days across all configurations: {total_winter_melting}")
        print(f"     This may indicate unrealistic temperature conditions causing glacier melting in winter.")
        print(f"     Consider reviewing temperature forcing data or model parameters.")
    
    return config_results

#--------------------------------------------------------------------------------

def plot_hru_group_temperatures_comparison(multi_config, validation_start=None, validation_end=None):
    """
    Plot HRU group temperatures comparison across multiple configurations.
    Each configuration gets its own subplot, showing every third HRU group for better readability.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    dict
        Dictionary containing HRU group temperature data for each configuration
    """
    
    # Use dates from multi_config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']
    model_type = multi_config.get('model_type', 'HBV')
    
    print(f"Creating HRU group temperature comparison for {len(configs)} configurations:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Validation period: {validation_start} to {validation_end}")
    
    # Create plot directory
    plot_dir = create_multi_plot_dir(multi_config)
    
    # Store results for each configuration
    config_results = {}
    hru_group_names = None  # Will be determined from first successful config
    
    # Process each configuration
    for config_dir in configs:
        print(f"\nProcessing configuration: {config_dir}")
        
        individual_config = _build_individual_config(multi_config, config_dir)
        paths = get_paths(individual_config)

        try:
            # Load HRU group temperature data
            output_dir = paths['output_dir']
            temp_csv_path = output_dir / f"{gauge_id}_{model_type}_TEMP_AVE_Daily_Average_ByHRUGroup.csv"
            
            if not temp_csv_path.exists():
                print(f"  ❌ HRU group temperature file not found: {temp_csv_path}")
                continue
            
            print(f"  📁 Loading HRU group temperature data: {temp_csv_path}")
            
            # Read temperature data with CORRECT header handling for your CSV structure
            # Row 1: HRU group names (column headers)
            # Row 2: 'time' (delete), 'day' (date column), 'mean', 'mean', etc. (descriptive headers)
            # Row 3+: Actual data
            
            # First, read the HRU group names from row 1
            hru_group_df = pd.read_csv(temp_csv_path, nrows=1, header=None)
            hru_group_column_names = hru_group_df.iloc[0].tolist()
            
            # Then, read the actual data starting from row 3 (skip rows 0 and 1)
            temp_df = pd.read_csv(temp_csv_path, skiprows=2, header=None)
            temp_df.columns = hru_group_column_names
            
            print(f"  📊 CSV structure - HRU group columns: {hru_group_column_names[:5]}... (showing first 5)")
            print(f"  📊 Data shape: {temp_df.shape}")
            
            # The first column is time step (delete), second column is date
            # Remove the first column (time step)
            temp_df = temp_df.drop(temp_df.columns[0], axis=1)
            hru_group_column_names = hru_group_column_names[1:]  # Remove first column name too
            temp_df.columns = hru_group_column_names
            
            # The first column should now be the date column
            date_col = temp_df.columns[0]  # This should be 'day' or similar
            
            # Convert the date column to datetime if possible
            try:
                temp_df[date_col] = pd.to_datetime(temp_df[date_col])
                print(f"  ✅ Successfully converted {date_col} to datetime")
            except Exception as e:
                print(f"  ⚠️  Could not convert {date_col} to datetime: {e}")
                # Try creating a date range if conversion fails
                temp_df[date_col] = pd.date_range(start='2000-01-01', periods=len(temp_df), freq='D')
            
            # Get HRU group names (exclude the date column and filter out NaN values)
            hru_groups = []
            for col in temp_df.columns:
                if col != date_col:
                    # Check if column name is a string and not NaN
                    if isinstance(col, str) and col.strip():
                        hru_groups.append(col)
                    elif pd.notna(col) and str(col) != 'nan':
                        # Convert to string if it's not NaN
                        col_str = str(col).strip()
                        if col_str and col_str != 'nan':
                            hru_groups.append(col_str)
                    else:
                        print(f"    ⚠️  Skipping invalid column name: {col} (type: {type(col)})")
            
            # Store group names from first successful config
            if hru_group_names is None:
                hru_group_names = hru_groups
                print(f"  📍 Found {len(hru_group_names)} valid HRU groups: {hru_group_names[:10]}{'...' if len(hru_group_names) > 10 else ''}")
            
            # Filter for validation period
            if temp_df[date_col].dtype == 'datetime64[ns]':
                validation_start_dt = pd.to_datetime(validation_start)
                validation_end_dt = pd.to_datetime(validation_end)
                
                mask = (temp_df[date_col] >= validation_start_dt) & (temp_df[date_col] <= validation_end_dt)
                temp_filtered = temp_df[mask].copy()
            else:
                temp_filtered = temp_df.copy()
            
            if len(temp_filtered) == 0:
                print(f"  ⚠️  No temperature data found for validation period")
                continue
            
            print(f"  📅 Filtered to {len(temp_filtered)} records for validation period")
            
            # Process temperature data for HRU groups
            hru_group_temp_data = {}
            found_groups = []
            
            for group_name in hru_groups:
                # Double-check that group_name is valid
                if not isinstance(group_name, str):
                    print(f"    ⚠️  Skipping non-string group name: {group_name}")
                    continue
                    
                if group_name in temp_filtered.columns:
                    # Convert temperature data to numeric, handling any non-numeric values
                    temp_data = pd.to_numeric(temp_filtered[group_name], errors='coerce')
                    hru_group_temp_data[group_name] = temp_data.values
                    found_groups.append(group_name)
                    print(f"    ✅ Found temperature data for HRU group '{group_name}'")
                else:
                    print(f"    ⚠️  Could not find temperature data for HRU group '{group_name}'")
            
            if len(hru_group_temp_data) == 0:
                print(f"  ❌ No HRU group temperature data found for {config_dir}")
                continue
            
            # Check for winter melting conditions
            winter_melting_days = 0
            glacier_groups = []  # Track which groups might be glacier-related
            
            if temp_filtered[date_col].dtype == 'datetime64[ns]':
                winter_months = [12, 1, 2]  # December, January, February
                winter_mask = temp_filtered[date_col].dt.month.isin(winter_months)
                
                for group_name, temps in hru_group_temp_data.items():
                    # Ensure group_name is a string before calling .lower()
                    if not isinstance(group_name, str):
                        continue
                        
                    winter_temps = temps[winter_mask]
                    # Remove NaN values before counting positive temperatures
                    valid_winter_temps = winter_temps[~pd.isna(winter_temps)]
                    positive_winter_days = (valid_winter_temps > 0).sum()
                    winter_melting_days += positive_winter_days
                    
                    # Check if this might be a glacier group (by name or winter melting)
                    group_name_lower = group_name.lower()
                    if 'glacier' in group_name_lower or 'ice' in group_name_lower:
                        glacier_groups.append(group_name)
                        if positive_winter_days > 0:
                            print(f"    ⚠️ HRU group '{group_name}': {positive_winter_days} days with T > 0°C in winter")
                    elif positive_winter_days > 10:  # Arbitrary threshold for concern
                        print(f"    ⚠️ HRU group '{group_name}': {positive_winter_days} days with T > 0°C in winter")
            
            config_results[config_dir] = {
                'temp_data': temp_filtered,
                'hru_group_temp_data': hru_group_temp_data,
                'found_groups': found_groups,
                'glacier_groups': glacier_groups,
                'winter_melting_days': winter_melting_days,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir),
                'date_col': date_col
            }
            
            print(f"  ✅ Processed temperature data: {len(temp_filtered)} records for {len(found_groups)} HRU groups")
            if glacier_groups:
                print(f"  🏔️  Identified potential glacier groups: {glacier_groups}")
            if winter_melting_days > 0:
                print(f"  ❄️ Warning: {winter_melting_days} total winter melting days found across all HRU groups")
            
        except Exception as e:
            print(f"  ❌ Error processing {config_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(config_results) == 0:
        print("❌ No configurations processed successfully")
        return None
    
    if hru_group_names is None or len(hru_group_names) == 0:
        print("❌ No valid HRU groups identified")
        return None
    
    # Select every third HRU group for plotting (better readability)
    selected_groups = hru_group_names[::3]  # Every third group
    print(f"\n📊 Selected {len(selected_groups)} HRU groups for plotting (every 3rd): {selected_groups[:10]}{'...' if len(selected_groups) > 10 else ''}")
    
    # Calculate subplot layout for configurations
    n_configs = len(config_results)
    if n_configs <= 2:
        n_rows, n_cols = 1, n_configs
        figsize = (8 * n_configs, 8)
    elif n_configs <= 4:
        n_rows, n_cols = 2, 2
        figsize = (16, 12)
    elif n_configs <= 6:
        n_rows, n_cols = 2, 3
        figsize = (20, 12)
    else:
        # For more configurations, use more rows
        n_cols = 3
        n_rows = (n_configs + n_cols - 1) // n_cols
        figsize = (20, 6 * n_rows)
    
    # Create subplots - one for each configuration
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    
    # Handle single subplot case
    if n_configs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_configs > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Use different colors for different HRU groups
    group_colors = plt.cm.tab20(np.linspace(0, 1, len(selected_groups)))  # Use tab20 for more colors
    
    # Plot each configuration in its own subplot
    for i, (config_dir, result) in enumerate(config_results.items()):
        ax = axes[i]
        
        temp_data = result['temp_data']
        hru_group_temp_data = result['hru_group_temp_data']
        config_color = result['color']
        config_name = result['name']
        date_col = result['date_col']
        winter_melting_days = result['winter_melting_days']
        
        # Plot each selected HRU group for this configuration
        for j, group_name in enumerate(selected_groups):
            if group_name in hru_group_temp_data:
                # Use different colors for different groups
                color = group_colors[j]
                alpha = 0.8
                linewidth = 1.5
                
                # Create label for legend (show first few groups)
                if j < 8:  # Only label first 8 groups to avoid legend clutter
                    label = group_name
                else:
                    label = None
                
                ax.plot(temp_data[date_col], hru_group_temp_data[group_name], 
                       color=color, alpha=alpha, linewidth=linewidth, label=label)
        
        # Add freezing line
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Freezing point')
        
        # Formatting for this subplot
        ax.set_title(f'{config_name}\n({len([g for g in selected_groups if g in hru_group_temp_data])} HRU groups)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend only to the first subplot
        if i == 0:
            ax.legend(loc='best', fontsize=8, ncol=2)
        
        # Set y-axis label for leftmost column
        if i % n_cols == 0:
            ax.set_ylabel('Temperature (°C)', fontsize=11)
        
        # Set x-axis label for bottom row
        if i >= (n_rows - 1) * n_cols or i >= n_configs - n_cols:
            ax.set_xlabel('Date', fontsize=11)
            # Format x-axis dates
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add winter melting warning as text annotation
        if winter_melting_days > 0:
            warning_text = f"⚠️ {winter_melting_days} winter melting days"
            ax.text(0.02, 0.98, warning_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'HRU Group Temperature Comparison by Configuration - Catchment {gauge_id}\n'
                f'Validation Period: {validation_start} to {validation_end} (Every 3rd HRU Group)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save time series plot
    save_path = plot_dir / f'hru_group_temperatures_by_config_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved HRU group temperatures by configuration to: {save_path}")
    plt.show()
    
    # 2. Create a separate monthly regime comparison plot
    plt.figure(figsize=(16, 10))
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create subplots for monthly regime - one per configuration
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    
    # Handle single subplot case
    if n_configs == 1:
        axes2 = [axes2]
    elif n_rows == 1:
        axes2 = axes2 if n_configs > 1 else [axes2]
    else:
        axes2 = axes2.flatten()
    
    # Plot monthly regime for each configuration
    for i, (config_dir, result) in enumerate(config_results.items()):
        ax = axes2[i]
        
        temp_data = result['temp_data']
        hru_group_temp_data = result['hru_group_temp_data']
        config_name = result['name']
        date_col = result['date_col']
        
        # Calculate monthly means for selected HRU groups
        if temp_data[date_col].dtype == 'datetime64[ns]':
            temp_data_copy = temp_data.copy()
            temp_data_copy['month'] = temp_data_copy[date_col].dt.month
            
            # Plot selected representative groups only
            representative_groups = []
            if 'AllHRUs' in hru_group_temp_data:
                representative_groups.append('AllHRUs')
            
            # Add some elevation bands
            low_elev = [g for g in selected_groups if g in hru_group_temp_data and ('1500-' in g or '2000-' in g)]
            mid_elev = [g for g in selected_groups if g in hru_group_temp_data and ('3000-' in g or '3500-' in g)]
            high_elev = [g for g in selected_groups if g in hru_group_temp_data and ('5000-' in g or '5500-' in g)]
            
            if low_elev:
                representative_groups.append(low_elev[0])
            if mid_elev:
                representative_groups.append(mid_elev[0])
            if high_elev:
                representative_groups.append(high_elev[0])
            
            # Use different line styles for different elevation bands
            line_styles = ['-', '--', '-.', ':', '-']
            colors = ['blue', 'green', 'orange', 'purple', 'brown']
            
            for j, group_name in enumerate(representative_groups):
                if group_name in hru_group_temp_data:
                    monthly_means = []
                    for month in range(1, 13):
                        month_mask = temp_data_copy['month'] == month
                        month_temps = hru_group_temp_data[group_name][month_mask]
                        valid_month_temps = month_temps[~pd.isna(month_temps)]
                        
                        if len(valid_month_temps) > 0:
                            monthly_means.append(np.mean(valid_month_temps))
                        else:
                            monthly_means.append(np.nan)
                    
                    line_style = line_styles[j % len(line_styles)]
                    color = colors[j % len(colors)]
                    ax.plot(range(1, 13), monthly_means, marker='o', linewidth=2, 
                           color=color, linestyle=line_style, label=group_name)
        
        # Add freezing line
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Freezing point')
        
        # Formatting
        ax.set_title(f'{config_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months)
        ax.grid(True, alpha=0.3)
        
        # Add legend only to the first subplot
        if i == 0:
            ax.legend(loc='best', fontsize=9)
        
        # Set y-axis label for leftmost column
        if i % n_cols == 0:
            ax.set_ylabel('Mean Temperature (°C)', fontsize=11)
        
        # Set x-axis label for bottom row
        if i >= (n_rows - 1) * n_cols or i >= n_configs - n_cols:
            ax.set_xlabel('Month', fontsize=11)
    
    # Hide unused subplots
    for i in range(n_configs, len(axes2)):
        axes2[i].set_visible(False)
    
    # Add overall title
    fig2.suptitle(f'Monthly Temperature Regime by Configuration - Catchment {gauge_id}\n'
                 f'Representative HRU Groups', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save monthly regime plot
    save_path = plot_dir / f'hru_group_temperatures_monthly_by_config_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved HRU group temperatures monthly regime by configuration to: {save_path}")
    plt.show()
    
    # Print comprehensive summary
    print(f"\n📊 HRU Group Temperature Analysis Summary:")
    print(f"  Configurations processed: {len(config_results)}")
    print(f"  Validation period: {validation_start} to {validation_end}")
    print(f"  Total valid HRU groups: {len(hru_group_names)}")
    print(f"  Selected HRU groups for plotting: {len(selected_groups)} (every 3rd)")
    
    print(f"\n  Temperature Statistics by Configuration:")
    for config_dir, result in config_results.items():
        hru_group_temp_data = result['hru_group_temp_data']
        config_name = result['name']
        found_groups = result['found_groups']
        glacier_groups = result['glacier_groups']
        winter_melting_days = result['winter_melting_days']
        
        print(f"\n    {config_name}:")
        print(f"      HRU groups found: {len(found_groups)}/{len(hru_group_names)}")
        print(f"      Selected groups plotted: {len([g for g in selected_groups if g in found_groups])}/{len(selected_groups)}")
        if glacier_groups:
            print(f"      Potential glacier groups: {glacier_groups}")
        if winter_melting_days > 0:
            print(f"      ❄️ Winter melting days: {winter_melting_days}")
        
        # Show statistics for selected groups
        selected_found = [g for g in selected_groups if g in found_groups]
        if selected_found:
            print(f"      Selected group stats (first 5):")
            for group_name in selected_found[:5]:  # Limit output
                temps = hru_group_temp_data[group_name]
                valid_temps = temps[~pd.isna(temps)]
                if len(valid_temps) > 0:
                    group_mean = np.mean(valid_temps)
                    group_min = np.min(valid_temps)
                    group_max = np.max(valid_temps)
                    group_positive = sum(1 for t in valid_temps if t > 0)
                    print(f"        {group_name}: Mean={group_mean:.1f}°C, Range=[{group_min:.1f}, {group_max:.1f}]°C, Positive days={group_positive}")
    
    # Warn about potential issues
    total_winter_melting = sum(result['winter_melting_days'] for result in config_results.values())
    if total_winter_melting > 0:
        print(f"\n  ⚠️ WARNING: Total winter melting days across all configurations: {total_winter_melting}")
        print(f"     This may indicate unrealistic temperature conditions.")
        print(f"     Consider reviewing temperature forcing data or model parameters.")
    
    # Identify potentially problematic groups
    problematic_groups = set()
    for config_dir, result in config_results.items():
        problematic_groups.update(result['glacier_groups'])
    
    if problematic_groups:
        print(f"\n  🏔️  Groups requiring attention (potential glaciers): {list(problematic_groups)}")
        print(f"     Monitor these groups for unrealistic winter melting.")
    
    return config_results

#--------------------------------------------------------------------------------

def check_temperature_netcdf_flipping(netcdf_path, sample_date=None):
    """
    Check if temperature NetCDF data is flipped with three simple plots:
    1. Plot with lat/lon coordinates
    2. Plot with cell indices (original data orientation)
    3. Plot with cell indices starting from 0 at bottom (flipped orientation)
    
    Also checks elevation data if present in the same file.
    
    Parameters:
    -----------
    netcdf_path : str
        Path to the temperature NetCDF file
    sample_date : str, optional
        Specific date to plot (format: 'YYYY-MM-DD'). If None, uses first available date.
    """
    import xarray as xr
    import matplotlib.pyplot as plt
    import numpy as np
    
    print(f"🌡️ Analyzing NetCDF file: {netcdf_path}")
    
    try:
        # Load the NetCDF file
        ds = xr.open_dataset(netcdf_path)
        print(f"📊 Dataset info:")
        print(f"  Variables: {list(ds.variables.keys())}")
        print(f"  Dimensions: {dict(ds.dims)}")
        
        # Find the temperature variable
        temp_var = None
        for var in ds.data_vars:
            if any(temp_name in var.lower() for temp_name in ['temp', 't2m', 'air']):
                temp_var = var
                break
        
        # Find the elevation variable
        elev_var = None
        for var in ds.data_vars:
            if any(elev_name in var.lower() for elev_name in ['elevation', 'elev', 'altitude', 'height']):
                elev_var = var
                break
        
        # If no specific variables found, use available data variables
        if temp_var is None and elev_var is None and len(ds.data_vars) >= 1:
            # Try to identify by shape and characteristics
            for var in ds.data_vars:
                var_data = ds[var]
                if 'time' in var_data.dims:
                    temp_var = var  # Assume time-varying data is temperature
                else:
                    elev_var = var  # Assume time-invariant data is elevation
                if temp_var and elev_var:
                    break
        
        if temp_var is None and elev_var is None:
            print(f"❌ Could not identify temperature or elevation variables from: {list(ds.data_vars)}")
            return None
        
        print(f"🌡️ Temperature variable: '{temp_var}'" if temp_var else "❌ No temperature variable found")
        print(f"🏔️ Elevation variable: '{elev_var}'" if elev_var else "❌ No elevation variable found")
        
        # === TEMPERATURE ANALYSIS ===
        temp_analysis = {}
        if temp_var:
            print(f"\n{'='*60}")
            print(f"🌡️ TEMPERATURE ANALYSIS")
            print(f"{'='*60}")
            
            # Get temperature data
            temp_data = ds[temp_var]
            
            # Select a sample date for plotting
            if 'time' in temp_data.dims:
                if sample_date is not None:
                    try:
                        temp_sample = temp_data.sel(time=sample_date, method='nearest')
                        actual_date = temp_sample.time.values
                        print(f"📅 Selected date: {sample_date} (actual: {actual_date})")
                    except:
                        print(f"⚠️  Could not find date {sample_date}, using first available date")
                        temp_sample = temp_data.isel(time=0)
                        actual_date = temp_sample.time.values
                        print(f"📅 Using first available date: {actual_date}")
                else:
                    temp_sample = temp_data.isel(time=0)
                    actual_date = temp_sample.time.values
                    print(f"📅 Using first available date: {actual_date}")
            else:
                temp_sample = temp_data
                actual_date = "No time dimension"
            
            # Convert temperature to Celsius if needed
            temp_values = temp_sample.values
            temp_units = temp_sample.attrs.get('units', 'unknown')
            print(f"🌡️ Temperature units: {temp_units}")
            print(f"🌡️ Temperature range: {np.nanmin(temp_values):.2f} to {np.nanmax(temp_values):.2f}")
            
            # Convert Kelvin to Celsius if needed
            if temp_units.lower() in ['k', 'kelvin'] or np.nanmin(temp_values) > 200:
                temp_values = temp_values - 273.15
                temp_units = '°C'
                print(f"🔄 Converted from Kelvin to Celsius")
                print(f"🌡️ Temperature range (°C): {np.nanmin(temp_values):.2f} to {np.nanmax(temp_values):.2f}")
            
            temp_analysis = {
                'values': temp_values,
                'units': temp_units,
                'sample': temp_sample,
                'date': actual_date
            }
        
        # === ELEVATION ANALYSIS ===
        elev_analysis = {}
        if elev_var:
            print(f"\n{'='*60}")
            print(f"🏔️ ELEVATION ANALYSIS")
            print(f"{'='*60}")
            
            # Get elevation data
            elev_data = ds[elev_var]
            elev_values = elev_data.values
            elev_units = elev_data.attrs.get('units', 'unknown')
            
            print(f"🏔️ Elevation units: {elev_units}")
            print(f"🏔️ Elevation range: {np.nanmin(elev_values):.1f} to {np.nanmax(elev_values):.1f}")
            print(f"🏔️ Elevation shape: {elev_values.shape}")
            print(f"🏔️ Elevation dimensions: {elev_data.dims}")
            
            # Sanity checks for elevation
            if np.nanmin(elev_values) < -500:
                print(f"  ⚠️ Very low elevations found ({np.nanmin(elev_values):.1f} {elev_units})")
            if np.nanmax(elev_values) > 9000:
                print(f"  ⚠️ Very high elevations found ({np.nanmax(elev_values):.1f} {elev_units})")
            if np.nanmean(elev_values) < 0:
                print(f"  ⚠️ Mean elevation is below sea level ({np.nanmean(elev_values):.1f} {elev_units})")
            
            elev_analysis = {
                'values': elev_values,
                'units': elev_units,
                'data': elev_data
            }
        
        # === COORDINATE ANALYSIS ===
        print(f"\n{'='*60}")
        print(f"📍 COORDINATE ANALYSIS")
        print(f"{'='*60}")
        
        # Use temperature sample for coordinate analysis, fallback to elevation
        sample_data = temp_analysis.get('sample') if temp_analysis else elev_analysis.get('data')
        
        # Get spatial coordinates
        spatial_dims = [dim for dim in sample_data.dims if dim not in ['time']]
        print(f"📍 Spatial dimensions: {spatial_dims}")
        
        # Identify latitude and longitude
        lat_dim = None
        lon_dim = None
        
        for dim in spatial_dims:
            if 'lat' in dim.lower() or 'y' in dim.lower():
                lat_dim = dim
            elif 'lon' in dim.lower() or 'x' in dim.lower():
                lon_dim = dim
        
        lat_values = None
        lon_values = None
        if lat_dim and lon_dim:
            lat_values = sample_data[lat_dim].values
            lon_values = sample_data[lon_dim].values
            print(f"🗺️  Latitude ({lat_dim}): {lat_values.min():.4f} to {lat_values.max():.4f}")
            print(f"🗺️  Longitude ({lon_dim}): {lon_values.min():.4f} to {lon_values.max():.4f}")
            print(f"🗺️  Latitude direction: {'Decreasing' if lat_values[0] > lat_values[-1] else 'Increasing'}")
            print(f"🗺️  Longitude direction: {'Decreasing' if lon_values[0] > lon_values[-1] else 'Increasing'}")
        
        # === PLOTTING ===
        print(f"\n{'='*60}")
        print(f"📊 CREATING COMPARISON PLOTS")
        print(f"{'='*60}")
        
        # Determine number of subplots needed
        n_vars = len([x for x in [temp_analysis, elev_analysis] if x])
        
        if n_vars == 2:
            # Both temperature and elevation - create 2 rows of 3 plots each
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            temp_axes = axes[0, :]
            elev_axes = axes[1, :]
        else:
            # Only one variable - create 1 row of 3 plots
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            if temp_analysis:
                temp_axes = axes
                elev_axes = None
            else:
                elev_axes = axes
                temp_axes = None
        
        # === TEMPERATURE PLOTS ===
        if temp_analysis and temp_axes is not None:
            temp_values = temp_analysis['values']
            temp_units = temp_analysis['units']
            temp_sample = temp_analysis['sample']
            
            # Plot 1: With lat/lon coordinates (if available)
            ax1 = temp_axes[0]
            if lat_values is not None and lon_values is not None:
                temp_sample.plot(ax=ax1, cmap='RdBu_r', add_colorbar=True, 
                               cbar_kwargs={'label': f'Temperature ({temp_units})'})
                ax1.set_title('Temperature: Lat/Lon Coordinates\n(XArray default plotting)')
                ax1.set_xlabel('Longitude')
                ax1.set_ylabel('Latitude')
            else:
                ax1.text(0.5, 0.5, 'No lat/lon coordinates available', 
                        transform=ax1.transAxes, ha='center', va='center', fontsize=14)
                ax1.set_title('Temperature: No Coordinates Available')
            
            # Plot 2: With cell indices (original data orientation)
            ax2 = temp_axes[1]
            im2 = ax2.imshow(temp_values, cmap='RdBu_r', aspect='auto', origin='upper')
            plt.colorbar(im2, ax=ax2, label=f'Temperature ({temp_units})')
            ax2.set_title('Temperature: Cell Indices (Original)\nY-axis: 0 at top, increasing downward')
            ax2.set_xlabel('Column Index (X)')
            ax2.set_ylabel('Row Index (Y)')
            
            # Add corner values as annotations
            if temp_values.shape[0] > 1 and temp_values.shape[1] > 1:
                ax2.text(0.02, 0.98, f'Top-left (0,0): {temp_values[0,0]:.1f}°C', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                ax2.text(0.98, 0.02, f'Bottom-right (-1,-1): {temp_values[-1,-1]:.1f}°C', 
                        transform=ax2.transAxes, verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Plot 3: With cell indices flipped (0 at bottom)
            ax3 = temp_axes[2]
            im3 = ax3.imshow(temp_values, cmap='RdBu_r', aspect='auto', origin='lower')
            plt.colorbar(im3, ax=ax3, label=f'Temperature ({temp_units})')
            ax3.set_title('Temperature: Cell Indices (Flipped)\nY-axis: 0 at bottom, increasing upward')
            ax3.set_xlabel('Column Index (X)')
            ax3.set_ylabel('Row Index (Y)')
            
            # Add corner values as annotations for flipped version
            if temp_values.shape[0] > 1 and temp_values.shape[1] > 1:
                ax3.text(0.02, 0.02, f'Bottom-left (array[0,0]): {temp_values[0,0]:.1f}°C', 
                        transform=ax3.transAxes, verticalalignment='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                ax3.text(0.98, 0.98, f'Top-right (array[-1,-1]): {temp_values[-1,-1]:.1f}°C', 
                        transform=ax3.transAxes, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # === ELEVATION PLOTS ===
        if elev_analysis and elev_axes is not None:
            elev_values = elev_analysis['values']
            elev_units = elev_analysis['units']
            elev_data = elev_analysis['data']
            
            # Plot 1: With lat/lon coordinates (if available)
            ax1 = elev_axes[0]
            if lat_values is not None and lon_values is not None:
                elev_data.plot(ax=ax1, cmap='terrain', add_colorbar=True, 
                              cbar_kwargs={'label': f'Elevation ({elev_units})'})
                ax1.set_title('Elevation: Lat/Lon Coordinates\n(XArray default plotting)')
                ax1.set_xlabel('Longitude')
                ax1.set_ylabel('Latitude')
            else:
                ax1.text(0.5, 0.5, 'No lat/lon coordinates available', 
                        transform=ax1.transAxes, ha='center', va='center', fontsize=14)
                ax1.set_title('Elevation: No Coordinates Available')
            
            # Plot 2: With cell indices (original data orientation)
            ax2 = elev_axes[1]
            im2 = ax2.imshow(elev_values, cmap='terrain', aspect='auto', origin='upper')
            plt.colorbar(im2, ax=ax2, label=f'Elevation ({elev_units})')
            ax2.set_title('Elevation: Cell Indices (Original)\nY-axis: 0 at top, increasing downward')
            ax2.set_xlabel('Column Index (X)')
            ax2.set_ylabel('Row Index (Y)')
            
            # Add corner values as annotations
            if elev_values.shape[0] > 1 and elev_values.shape[1] > 1:
                ax2.text(0.02, 0.98, f'Top-left (0,0): {elev_values[0,0]:.0f}m', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                ax2.text(0.98, 0.02, f'Bottom-right (-1,-1): {elev_values[-1,-1]:.0f}m', 
                        transform=ax2.transAxes, verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Plot 3: With cell indices flipped (0 at bottom)
            ax3 = elev_axes[2]
            im3 = ax3.imshow(elev_values, cmap='terrain', aspect='auto', origin='lower')
            plt.colorbar(im3, ax=ax3, label=f'Elevation ({elev_units})')
            ax3.set_title('Elevation: Cell Indices (Flipped)\nY-axis: 0 at bottom, increasing upward')
            ax3.set_xlabel('Column Index (X)')
            ax3.set_ylabel('Row Index (Y)')
            
            # Add corner values as annotations for flipped version
            if elev_values.shape[0] > 1 and elev_values.shape[1] > 1:
                ax3.text(0.02, 0.02, f'Bottom-left (array[0,0]): {elev_values[0,0]:.0f}m', 
                        transform=ax3.transAxes, verticalalignment='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                ax3.text(0.98, 0.98, f'Top-right (array[-1,-1]): {elev_values[-1,-1]:.0f}m', 
                        transform=ax3.transAxes, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # === ANALYSIS SUMMARY ===
        print(f"\n{'='*60}")
        print(f"📊 DATA ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        # Coordinate analysis
        needs_flip = lat_values is not None and lat_values[0] > lat_values[-1]
        
        if lat_values is not None and lon_values is not None:
            print(f"  Coordinate ranges:")
            print(f"    Latitude: {lat_values.min():.4f} to {lat_values.max():.4f}")
            print(f"    Longitude: {lon_values.min():.4f} to {lon_values.max():.4f}")
            
            # Check coordinate direction
            if lat_values[0] > lat_values[-1]:
                print(f"  ⚠️  Latitude decreases from first to last element")
                print(f"     This suggests data might need vertical flipping")
            else:
                print(f"  ✅ Latitude increases from first to last element")
        
        # Temperature analysis
        if temp_analysis:
            temp_values = temp_analysis['values']
            temp_units = temp_analysis['units']
            print(f"\n  Temperature Analysis:")
            print(f"    Data shape: {temp_values.shape} (rows, columns)")
            print(f"    Temperature range: {np.nanmin(temp_values):.2f} to {np.nanmax(temp_values):.2f} {temp_units}")
            
            # Temperature corner analysis
            if temp_values.shape[0] > 1 and temp_values.shape[1] > 1:
                temp_corners = {
                    'Array[0,0] (top-left in middle plot)': temp_values[0, 0],
                    'Array[0,-1] (top-right in middle plot)': temp_values[0, -1],
                    'Array[-1,0] (bottom-left in middle plot)': temp_values[-1, 0],
                    'Array[-1,-1] (bottom-right in middle plot)': temp_values[-1, -1],
                }
                
                print(f"    Corner Values:")
                for corner, value in temp_corners.items():
                    print(f"      {corner}: {value:.2f}°C")
        
        # Elevation analysis
        if elev_analysis:
            elev_values = elev_analysis['values']
            elev_units = elev_analysis['units']
            print(f"\n  Elevation Analysis:")
            print(f"    Data shape: {elev_values.shape} (rows, columns)")
            print(f"    Elevation range: {np.nanmin(elev_values):.1f} to {np.nanmax(elev_values):.1f} {elev_units}")
            print(f"    Mean elevation: {np.nanmean(elev_values):.1f} {elev_units}")
            
            # Elevation corner analysis
            if elev_values.shape[0] > 1 and elev_values.shape[1] > 1:
                elev_corners = {
                    'Array[0,0] (top-left in middle plot)': elev_values[0, 0],
                    'Array[0,-1] (top-right in middle plot)': elev_values[0, -1],
                    'Array[-1,0] (bottom-left in middle plot)': elev_values[-1, 0],
                    'Array[-1,-1] (bottom-right in middle plot)': elev_values[-1, -1],
                }
                
                print(f"    Corner Values:")
                for corner, value in elev_corners.items():
                    print(f"      {corner}: {value:.1f}m")
                
                # Check for elevation gradient patterns
                print(f"    Elevation Gradient Analysis:")
                # Check if elevation increases from south to north (typical)
                north_south_gradient = elev_values[-1, :].mean() - elev_values[0, :].mean()
                east_west_gradient = elev_values[:, -1].mean() - elev_values[:, 0].mean()
                print(f"      North-South gradient: {north_south_gradient:.1f}m (+ means higher in north)")
                print(f"      East-West gradient: {east_west_gradient:.1f}m (+ means higher in east)")
        
        print(f"\n💡 Interpretation Guide:")
        print(f"  - Left plots: Show how xarray interprets your coordinates")
        print(f"  - Middle plots: Raw array with array[0,0] at top-left (origin='upper')")
        print(f"  - Right plots: Same data with array[0,0] at bottom-left (origin='lower')")
        print(f"  ")
        if needs_flip:
            print(f"  ⚠️  COORDINATE FLIPPING DETECTED!")
            print(f"      Your latitude coordinates decrease, suggesting data needs vertical flipping")
            print(f"      Compare middle vs right plots to see which orientation makes sense")
            print(f"      🔧 Solution: Use coordinate flipping in your data processing")
        else:
            print(f"  ✅ Coordinates appear to be in standard orientation")
        
        print(f"  ")
        print(f"  📏 For elevation data:")
        print(f"      - Higher elevations should typically be in mountainous areas")
        print(f"      - Check if elevation patterns match expected topography")
        print(f"      - Corner values can help identify correct orientation")
        
        # Return comprehensive results
        results = {
            'coordinates': {'lat': lat_values, 'lon': lon_values},
            'needs_flip': needs_flip,
            'dataset': ds
        }
        
        if temp_analysis:
            results['temperature'] = {
                'values': temp_analysis['values'],
                'units': temp_analysis['units'],
                'corners': temp_corners if 'temp_corners' in locals() else {}
            }
        
        if elev_analysis:
            results['elevation'] = {
                'values': elev_analysis['values'],
                'units': elev_analysis['units'],
                'corners': elev_corners if 'elev_corners' in locals() else {}
            }
        
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing NetCDF file: {e}")
        import traceback
        traceback.print_exc()
        return None


#--------------------------------------------------------------------------------
########################## configuration loader #################################
#--------------------------------------------------------------------------------

def load_configurations(gauge_id, model='HBV', env=None, configs=None):
    """
    Load configuration registry and build multi_config dict for a given catchment.

    Uses the composable config layer system (config_merge) — no individual
    namelist files needed.

    Parameters:
    -----------
    gauge_id : str
        Gauge identifier (e.g., '0101')
    model : str
        Hydrological model type (default 'HBV')
    env : str, optional
        Environment ('local' or 'server'). Auto-detected if None.
    configs : list, optional
        List of config keys to include. If None, loads all from layers.

    Returns:
    --------
    dict
        multi_config dictionary with keys: 'main_dir', 'gauge_id', 'model_type',
        'start_date', 'end_date', 'cali_end_date', 'configs', 'config_colors',
        'config_names', 'config_coupled', 'config_glacier_source', 'glogem_dir'
    """
    from config_merge import load_config, load_configurations_registry

    print(f"Loading configurations for gauge {gauge_id}")

    registry = load_configurations_registry()
    if configs:
        registry = [cfg for cfg in registry if cfg['key'] in configs]
    print(f"Found {len(registry)} configurations in registry")

    config_keys = []
    config_colors = {}
    config_names = {}
    config_coupled = {}
    config_glacier_source = {}
    config_icemelt_mode = {}

    main_dir = None
    model_type = None
    start_date = None
    end_date = None
    cali_end_date = None
    glogem_dir = None

    for cfg in registry:
        key = cfg['key']

        try:
            nml, tmp_path = load_config(str(gauge_id), key, model, env=env)
            # Clean up temp file immediately
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        except Exception as e:
            print(f"  WARNING: Could not load config {key} for {gauge_id}: {e}")
            continue

        # Check that output actually exists before including
        paths = get_paths(nml)
        hydro_file = paths['output_dir'] / f"{gauge_id}_{model}_Hydrographs.csv"
        if not hydro_file.exists():
            print(f"  - {cfg['display_name']:40s}  (no output, skipping)")
            continue

        if model_type is None:
            main_dir = nml['main_dir']
            model_type = nml.get('model_type', 'HBV')
            start_date = nml.get('start_date')
            end_date = nml.get('end_date')
            cali_end_date = nml.get('cali_end_date')
            glogem_dir = nml.get('glogem_dir')

        config_keys.append(key)
        config_colors[key] = cfg['color']
        config_names[key] = cfg['display_name']
        config_coupled[key] = cfg['coupled']
        config_glacier_source[key] = cfg['glacier_source']
        config_icemelt_mode[key] = cfg.get('icemelt_mode', False)

        print(f"  + {cfg['display_name']:40s} -> {key}")

    if len(config_keys) == 0:
        print("ERROR: No configurations were successfully loaded")
        return None

    multi_config = {
        'main_dir': main_dir,
        'gauge_id': gauge_id,
        'model_type': model_type,
        'start_date': start_date,
        'end_date': end_date,
        'cali_end_date': cali_end_date,
        'configs': config_keys,
        'config_colors': config_colors,
        'config_names': config_names,
        'config_coupled': config_coupled,
        'config_glacier_source': config_glacier_source,
        'config_icemelt_mode': config_icemelt_mode,
        'glogem_dir': glogem_dir,
    }

    print(f"\nSuccessfully loaded {len(config_keys)} configurations")
    return multi_config


#--------------------------------------------------------------------------------
######################### subplot layout helper #################################
#--------------------------------------------------------------------------------

def _calc_subplot_layout(n_configs):
    """Calculate subplot grid layout and figure size for n configurations."""
    if n_configs <= 2:
        n_rows, n_cols = 1, n_configs
        figsize = (8 * n_configs, 6)
    elif n_configs <= 4:
        n_rows, n_cols = 2, 2
        figsize = (16, 12)
    elif n_configs <= 6:
        n_rows, n_cols = 2, 3
        figsize = (20, 12)
    elif n_configs <= 9:
        n_rows, n_cols = 3, 3
        figsize = (20, 16)
    elif n_configs <= 12:
        n_rows, n_cols = 3, 4
        figsize = (22, 16)
    else:
        n_cols = 4
        n_rows = (n_configs + n_cols - 1) // n_cols
        figsize = (22, 5 * n_rows)
    return n_rows, n_cols, figsize


def _build_individual_config(multi_config, config_key):
    """Build an individual config dict from multi_config for a given config_key."""
    return {
        'main_dir': multi_config['main_dir'],
        '_config_key': config_key,
        'gauge_id': multi_config['gauge_id'],
        'start_date': multi_config.get('start_date', '2000-01-01'),
        'end_date': multi_config.get('end_date', '2020-12-31'),
        'cali_end_date': multi_config.get('cali_end_date', '2009-12-31'),
        'model_type': multi_config.get('model_type', 'HBV'),
        'coupled': multi_config.get('config_coupled', {}).get(config_key, False),
        'glogem_dir': multi_config.get('glogem_dir'),
    }


#--------------------------------------------------------------------------------
##################### scatter plot subplots ######################################
#--------------------------------------------------------------------------------

def plot_streamflow_scatter_subplots(multi_config, validation_start=None, validation_end=None):
    """
    Plot sim vs obs scatter for each configuration in separate subplots.

    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period

    Returns:
    --------
    dict
        Dictionary containing scatter statistics for each configuration
    """

    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')

    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']

    print(f"Creating streamflow scatter subplots for {len(configs)} configurations")

    plot_dir = create_multi_plot_dir(multi_config)
    config_results = {}

    # Load data for each config
    for config_dir in configs:
        individual_config = _build_individual_config(multi_config, config_dir)
        try:
            data = load_hydrograph_data(individual_config)
            if data is None:
                continue

            start_dt = pd.to_datetime(validation_start)
            end_dt = pd.to_datetime(validation_end)
            mask = (data['date'] >= start_dt) & (data['date'] <= end_dt)
            df = data[mask].dropna(subset=['obs_Q', 'sim_Q'])

            if len(df) == 0:
                continue

            obs = df['obs_Q'].values
            sim = df['sim_Q'].values

            # Performance metrics
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(obs, sim)
            obs_mean = np.mean(obs)
            nse = 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - obs_mean) ** 2))
            corr = np.corrcoef(sim, obs)[0, 1]
            alpha = np.std(sim) / np.std(obs)
            beta = np.mean(sim) / np.mean(obs)
            kge = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

            config_results[config_dir] = {
                'obs': obs, 'sim': sim,
                'r_squared': r_value**2, 'nse': nse, 'kge': kge,
                'slope': slope, 'intercept': intercept,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir),
            }
            print(f"  + {config_names.get(config_dir, config_dir)}: R2={r_value**2:.3f}, NSE={nse:.3f}, KGE={kge:.3f}")

        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue

    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None

    # Create subplot grid
    n_configs = len(config_results)
    n_rows, n_cols, figsize = _calc_subplot_layout(n_configs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_configs == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    for i, (config_dir, result) in enumerate(config_results.items()):
        ax = axes[i]
        obs = result['obs']
        sim = result['sim']
        color = result['color']

        ax.scatter(obs, sim, alpha=0.4, s=15, c=color, edgecolors='none')

        # 1:1 line
        min_val = min(obs.min(), sim.min())
        max_val = max(obs.max(), sim.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, zorder=10)

        # Regression line
        line_x = np.array([min_val, max_val])
        ax.plot(line_x, result['slope'] * line_x + result['intercept'],
                color='red', linewidth=1.5, alpha=0.7, zorder=9)

        ax.set_title(result['name'], fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Metrics text
        stats_text = f"R²={result['r_squared']:.3f}\nNSE={result['nse']:.3f}\nKGE={result['kge']:.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        if i % n_cols == 0:
            ax.set_ylabel('Simulated (m³/s)', fontsize=13, fontweight='bold')
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Observed (m³/s)', fontsize=13, fontweight='bold')

    # Hide unused subplots
    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    save_path = plot_dir / f'streamflow_scatter_subplots_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved scatter subplots to: {save_path}")
    plt.show()

    return config_results


#--------------------------------------------------------------------------------
##################### residuals subplots ########################################
#--------------------------------------------------------------------------------

def plot_streamflow_residuals_subplots(multi_config, validation_start=None, validation_end=None):
    """
    Plot streamflow residuals (Sim - Obs vs Obs) for each configuration in separate subplots.

    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period

    Returns:
    --------
    dict
        Dictionary containing residual statistics for each configuration
    """

    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')

    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']

    print(f"Creating streamflow residual subplots for {len(configs)} configurations")

    plot_dir = create_multi_plot_dir(multi_config)
    config_results = {}

    for config_dir in configs:
        individual_config = _build_individual_config(multi_config, config_dir)
        try:
            data = load_hydrograph_data(individual_config)
            if data is None:
                continue

            start_dt = pd.to_datetime(validation_start)
            end_dt = pd.to_datetime(validation_end)
            mask = (data['date'] >= start_dt) & (data['date'] <= end_dt)
            df = data[mask].dropna(subset=['obs_Q', 'sim_Q'])

            if len(df) == 0:
                continue

            obs = df['obs_Q'].values
            residuals = df['sim_Q'].values - obs
            bias = np.mean(residuals)
            std_res = np.std(residuals)

            config_results[config_dir] = {
                'obs': obs, 'residuals': residuals,
                'bias': bias, 'std': std_res,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir),
            }
            print(f"  + {config_names.get(config_dir, config_dir)}: Bias={bias:+.3f}, Std={std_res:.3f}")

        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue

    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None

    n_configs = len(config_results)
    n_rows, n_cols, figsize = _calc_subplot_layout(n_configs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_configs == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    for i, (config_dir, result) in enumerate(config_results.items()):
        ax = axes[i]
        obs = result['obs']
        residuals = result['residuals']
        color = result['color']
        bias = result['bias']
        std_res = result['std']

        ax.scatter(obs, residuals, alpha=0.4, s=15, c=color, edgecolors='none')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=bias, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
        ax.axhline(y=2*std_res, color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(y=-2*std_res, color='red', linestyle=':', linewidth=1, alpha=0.5)

        ax.set_title(result['name'], fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        stats_text = f"Bias={bias:+.2f}\nStd={std_res:.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        if i % n_cols == 0:
            ax.set_ylabel('Residual (m³/s)', fontsize=13, fontweight='bold')
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Observed (m³/s)', fontsize=13, fontweight='bold')

    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    save_path = plot_dir / f'streamflow_residuals_subplots_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved residual subplots to: {save_path}")
    plt.show()

    return config_results


#--------------------------------------------------------------------------------
################# precipitation partitioning subplots ###########################
#--------------------------------------------------------------------------------

def plot_precipitation_partitioning_subplots(multi_config, validation_start=None, validation_end=None):
    """
    Plot precipitation partitioning (rainfall vs snowfall) for each configuration in separate subplots.

    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period

    Returns:
    --------
    dict
        Dictionary containing precip data for each configuration
    """

    if validation_start is None:
        validation_start = multi_config.get('start_date', '2000-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')

    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']

    print(f"Creating precipitation partitioning subplots for {len(configs)} configurations")

    plot_dir = create_multi_plot_dir(multi_config)
    config_results = {}

    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for config_dir in configs:
        individual_config = _build_individual_config(multi_config, config_dir)
        try:
            # Load snowfall and rainfall data
            snowfall_df = load_forcing_by_hrugroup(individual_config, 'SNOWFALL')
            rainfall_df = load_forcing_by_hrugroup(individual_config, 'RAINFALL')

            if snowfall_df is None or rainfall_df is None:
                print(f"  Skipping {config_dir}: missing forcing data")
                continue

            # Merge
            df = pd.merge(
                snowfall_df.rename(columns={'NO_GLACIER': 'snowfall'}),
                rainfall_df.rename(columns={'NO_GLACIER': 'rainfall'}),
                on='date', how='inner'
            )

            # Filter period
            start_dt = pd.to_datetime(validation_start)
            end_dt = pd.to_datetime(validation_end)
            mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
            df_period = df[mask].copy()

            if len(df_period) == 0:
                continue

            # Load area scaling
            paths = get_paths(individual_config)
            topo_dir = paths['topo_dir']
            hru_shapefile = topo_dir / "HRU.shp"

            area_fraction = 1.0
            if hru_shapefile.exists():
                import geopandas as gpd
                hru_gdf = gpd.read_file(hru_shapefile)
                total_area = hru_gdf['Area_km2'].sum()
                if 'Landuse_Cl' in hru_gdf.columns:
                    non_glacier_area = hru_gdf[~hru_gdf['Landuse_Cl'].isin([7, 8])]['Area_km2'].sum()
                    area_fraction = non_glacier_area / total_area if total_area > 0 else 1.0

            df_period['snowfall_scaled'] = df_period['snowfall'] * area_fraction
            df_period['rainfall_scaled'] = df_period['rainfall'] * area_fraction
            df_period['month'] = df_period['date'].dt.month

            monthly_snow = df_period.groupby('month')['snowfall_scaled'].mean()
            monthly_rain = df_period.groupby('month')['rainfall_scaled'].mean()

            config_results[config_dir] = {
                'monthly_snow': monthly_snow,
                'monthly_rain': monthly_rain,
                'area_fraction': area_fraction,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir),
            }
            print(f"  + {config_names.get(config_dir, config_dir)}: mean precip={monthly_snow.mean() + monthly_rain.mean():.2f} mm/day")

        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue

    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None

    n_configs = len(config_results)
    n_rows, n_cols, figsize = _calc_subplot_layout(n_configs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    if n_configs == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    for i, (config_dir, result) in enumerate(config_results.items()):
        ax = axes[i]
        monthly_rain = result['monthly_rain']
        monthly_snow = result['monthly_snow']

        ax.bar(months, monthly_rain, color='steelblue', label='Rainfall', edgecolor='navy', linewidth=0.5)
        ax.bar(months, monthly_snow, bottom=monthly_rain, color='lightcyan', label='Snowfall',
               edgecolor='darkblue', linewidth=0.5)

        ax.set_title(result['name'], fontsize=14, fontweight='bold')
        ax.set_xticks(months)
        ax.set_xticklabels(month_names, rotation=45, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        if i == 0:
            ax.legend(fontsize=11)
        if i % n_cols == 0:
            ax.set_ylabel('Precip (mm/day)', fontsize=13, fontweight='bold')

    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)

    fig.text(0.5, 0.02, 'Month', ha='center', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 1.0])

    save_path = plot_dir / f'precipitation_partitioning_subplots_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved precipitation partitioning subplots to: {save_path}")
    plt.show()

    return config_results


#--------------------------------------------------------------------------------
##################### snowmelt regime subplots ##################################
#--------------------------------------------------------------------------------

def plot_snowmelt_regime_subplots(multi_config, validation_start=None, validation_end=None, unit='mm'):
    """
    Plot monthly snowmelt regime for each configuration in separate subplots.

    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    unit : str, optional
        Unit for display ('mm' for mm/day, 'm3' for m³/s)

    Returns:
    --------
    dict
        Dictionary containing snowmelt regime data for each configuration
    """

    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')

    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']

    unit_label = 'mm/day' if unit == 'mm' else 'm³/s'

    print(f"Creating snowmelt regime subplots for {len(configs)} configurations")

    plot_dir = create_multi_plot_dir(multi_config)
    config_results = {}

    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for config_dir in configs:
        individual_config = _build_individual_config(multi_config, config_dir)
        try:
            snowmelt_df = load_snowmelt_mass_loadings(individual_config, validation_start, validation_end, unit=unit)

            if snowmelt_df is None:
                print(f"  Skipping {config_dir}: no snowmelt data")
                continue

            snowmelt_col = 'snowmelt_mm_day' if unit == 'mm' and 'snowmelt_mm_day' in snowmelt_df.columns else 'snowmelt_m3s'
            snowmelt_df['month'] = snowmelt_df['date'].dt.month
            monthly_regime = snowmelt_df.groupby('month')[snowmelt_col].mean()

            config_results[config_dir] = {
                'monthly_regime': monthly_regime,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir),
            }
            print(f"  + {config_names.get(config_dir, config_dir)}: peak={monthly_regime.max():.4f} {unit_label}")

        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue

    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None

    n_configs = len(config_results)
    n_rows, n_cols, figsize = _calc_subplot_layout(n_configs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    if n_configs == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    for i, (config_dir, result) in enumerate(config_results.items()):
        ax = axes[i]
        regime = result['monthly_regime']
        color = result['color']

        ax.plot(months, [regime.get(m, 0) for m in months], color=color, linewidth=2.5, marker='o', markersize=5)
        ax.fill_between(months, 0, [regime.get(m, 0) for m in months], alpha=0.3, color=color)

        ax.set_title(result['name'], fontsize=14, fontweight='bold')
        ax.set_xticks(months)
        ax.set_xticklabels(month_names, rotation=45, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

        if i % n_cols == 0:
            ax.set_ylabel(f'Snowmelt ({unit_label})', fontsize=13, fontweight='bold')

    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)

    fig.text(0.5, 0.02, 'Month', ha='center', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 1.0])

    save_path = plot_dir / f'snowmelt_regime_subplots_{unit}_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved snowmelt regime subplots to: {save_path}")
    plt.show()

    return config_results


#--------------------------------------------------------------------------------
##################### glacier melt regime subplots ##############################
#--------------------------------------------------------------------------------

def plot_glacier_melt_regime_subplots(multi_config, validation_start=None, validation_end=None, unit='mm'):
    """
    Plot monthly glacier melt regime for each configuration in separate subplots.
    Skips configurations with glacier_source='none' (baseline).
    For coupled configs, uses GloGEM icemelt. For deltah configs, uses mass loadings.

    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    unit : str, optional
        Unit for display ('mm' for mm/day, 'm3' for m³/s)

    Returns:
    --------
    dict
        Dictionary containing glacier melt regime data for each configuration
    """

    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')

    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']
    config_glacier_source = multi_config.get('config_glacier_source', {})
    model_type = multi_config.get('model_type', 'HBV')

    unit_label = 'mm/day' if unit == 'mm' else 'm³/s'

    print(f"Creating glacier melt regime subplots for {len(configs)} configurations")

    plot_dir = create_multi_plot_dir(multi_config)
    config_results = {}

    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for config_dir in configs:
        glacier_source = config_glacier_source.get(config_dir, 'none')

        # Skip baseline (no glacier melt)
        if glacier_source == 'none':
            print(f"  Skipping {config_names.get(config_dir, config_dir)}: no glacier source")
            continue

        individual_config = _build_individual_config(multi_config, config_dir)

        try:
            if glacier_source == 'glogem':
                # Coupled: use GloGEM icemelt
                glogem_data = load_glogem_data(individual_config, unit=unit, plot=False)
                if glogem_data is None:
                    print(f"  Skipping {config_dir}: no GloGEM data")
                    continue

                val_start = pd.to_datetime(validation_start)
                val_end = pd.to_datetime(validation_end)
                glogem_mask = (glogem_data['date'] >= val_start) & (glogem_data['date'] <= val_end)
                glogem_filtered = glogem_data[glogem_mask].copy()

                if 'icemelt_normalized' not in glogem_filtered.columns:
                    print(f"  Skipping {config_dir}: no icemelt_normalized column")
                    continue

                glogem_filtered['month'] = glogem_filtered['date'].dt.month
                monthly_regime = glogem_filtered.groupby('month')['icemelt_normalized'].mean()

            elif glacier_source == 'raven_deltah':
                # DeltaH: use glacier melt mass loadings
                paths = get_paths(individual_config)
                glacier_file = paths['output_dir'] / f"{gauge_id}_{model_type}_GLACIERMELT_ALLMassLoadings.csv"

                if not glacier_file.exists():
                    print(f"  Skipping {config_dir}: glacier melt file not found")
                    continue

                glacier_df = pd.read_csv(glacier_file)
                glacier_df['date'] = pd.to_datetime(glacier_df['date'])

                val_start = pd.to_datetime(validation_start)
                val_end = pd.to_datetime(validation_end)
                glacier_mask = (glacier_df['date'] >= val_start) & (glacier_df['date'] <= val_end)
                glacier_filtered = glacier_df[glacier_mask].copy()

                glacier_m3s_col = f"{gauge_id} m3/s"
                if glacier_m3s_col not in glacier_filtered.columns:
                    print(f"  Skipping {config_dir}: column '{glacier_m3s_col}' not found")
                    continue

                # Convert to mm/day if needed
                if unit == 'mm':
                    topo_dir = paths['topo_dir']
                    hru_shapefile = topo_dir / "HRU.shp"
                    if hru_shapefile.exists():
                        import geopandas as gpd
                        hru_gdf = gpd.read_file(hru_shapefile)
                        total_area_km2 = hru_gdf['Area_km2'].sum()
                        conversion = 86400 / (total_area_km2 * 1000000) * 1000
                        glacier_filtered['glacier_melt_converted'] = glacier_filtered[glacier_m3s_col] * conversion
                    else:
                        glacier_filtered['glacier_melt_converted'] = glacier_filtered[glacier_m3s_col]
                else:
                    glacier_filtered['glacier_melt_converted'] = glacier_filtered[glacier_m3s_col]

                glacier_filtered['month'] = glacier_filtered['date'].dt.month
                monthly_regime = glacier_filtered.groupby('month')['glacier_melt_converted'].mean()

            config_results[config_dir] = {
                'monthly_regime': monthly_regime,
                'glacier_source': glacier_source,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir),
            }
            print(f"  + {config_names.get(config_dir, config_dir)}: peak={monthly_regime.max():.4f} {unit_label} ({glacier_source})")

        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None

    n_configs = len(config_results)
    n_rows, n_cols, figsize = _calc_subplot_layout(n_configs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    if n_configs == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    for i, (config_dir, result) in enumerate(config_results.items()):
        ax = axes[i]
        regime = result['monthly_regime']
        color = result['color']

        ax.plot(months, [regime.get(m, 0) for m in months], color=color, linewidth=2.5, marker='o', markersize=5)
        ax.fill_between(months, 0, [regime.get(m, 0) for m in months], alpha=0.3, color=color)

        source_label = 'GloGEM' if result['glacier_source'] == 'glogem' else 'DeltaH'
        ax.set_title(f"{result['name']} ({source_label})", fontsize=13, fontweight='bold')
        ax.set_xticks(months)
        ax.set_xticklabels(month_names, rotation=45, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

        if i % n_cols == 0:
            ax.set_ylabel(f'Glacier Melt ({unit_label})', fontsize=13, fontweight='bold')

    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)

    fig.text(0.5, 0.02, 'Month', ha='center', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 1.0])

    save_path = plot_dir / f'glacier_melt_regime_subplots_{unit}_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved glacier melt regime subplots to: {save_path}")
    plt.show()

    return config_results


#--------------------------------------------------------------------------------
##################### water balance subplots ####################################
#--------------------------------------------------------------------------------

def plot_water_balance_subplots(multi_config, validation_start=None, validation_end=None):
    """
    Plot annual water balance for each configuration in separate subplots.
    Shows observed Q, simulated Q, precipitation, snowmelt, and glacier melt as bars.

    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period

    Returns:
    --------
    dict
        Dictionary containing water balance data for each configuration
    """

    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')

    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']
    model_type = multi_config.get('model_type', 'HBV')

    print(f"Creating water balance subplots for {len(configs)} configurations")

    plot_dir = create_multi_plot_dir(multi_config)
    config_results = {}

    for config_dir in configs:
        individual_config = _build_individual_config(multi_config, config_dir)

        paths = get_paths(individual_config)

        try:
            # Load catchment area for unit conversion
            topo_dir = paths['topo_dir']
            hru_shapefile = topo_dir / "HRU.shp"

            if not hru_shapefile.exists():
                print(f"  Skipping {config_dir}: HRU shapefile not found")
                continue

            import geopandas as gpd
            hru_gdf = gpd.read_file(hru_shapefile)
            total_area_km2 = hru_gdf['Area_km2'].sum()
            conversion_m3s_to_mm_day = 86400 / (total_area_km2 * 1000000) * 1000

            # Load streamflow
            data = load_hydrograph_data(individual_config)
            if data is None:
                continue

            start_dt = pd.to_datetime(validation_start)
            end_dt = pd.to_datetime(validation_end)
            mask = (data['date'] >= start_dt) & (data['date'] <= end_dt)
            df = data[mask].copy()

            if len(df) == 0:
                continue

            df['obs_Q_mm'] = df['obs_Q'] * conversion_m3s_to_mm_day
            df['sim_Q_mm'] = df['sim_Q'] * conversion_m3s_to_mm_day
            df['year'] = df['date'].dt.year

            annual = df.groupby('year').agg({
                'obs_Q_mm': 'sum',
                'sim_Q_mm': 'sum'
            }).reset_index()

            # Calculate means
            mean_obs = annual['obs_Q_mm'].mean()
            mean_sim = annual['sim_Q_mm'].mean()

            # Load snowmelt
            mean_snowmelt = 0
            try:
                snowmelt_df = load_snowmelt_mass_loadings(individual_config, validation_start, validation_end, unit='mm')
                if snowmelt_df is not None and 'snowmelt_mm_day' in snowmelt_df.columns:
                    snowmelt_df['year'] = snowmelt_df['date'].dt.year
                    mean_snowmelt = snowmelt_df.groupby('year')['snowmelt_mm_day'].sum().mean()
            except Exception:
                pass

            # Load glacier melt
            mean_glacier_melt = 0
            glacier_source = multi_config.get('config_glacier_source', {}).get(config_dir, 'none')
            try:
                if glacier_source == 'glogem':
                    glogem_data = load_glogem_data(individual_config, unit='mm', plot=False)
                    if glogem_data is not None and 'icemelt_normalized' in glogem_data.columns:
                        glogem_mask = (glogem_data['date'] >= start_dt) & (glogem_data['date'] <= end_dt)
                        glogem_filtered = glogem_data[glogem_mask].copy()
                        glogem_filtered['year'] = glogem_filtered['date'].dt.year
                        mean_glacier_melt = glogem_filtered.groupby('year')['icemelt_normalized'].sum().mean()
                elif glacier_source == 'raven_deltah':
                    glacier_file = paths['output_dir'] / f"{gauge_id}_{model_type}_GLACIERMELT_ALLMassLoadings.csv"
                    if glacier_file.exists():
                        glacier_df = pd.read_csv(glacier_file)
                        glacier_df['date'] = pd.to_datetime(glacier_df['date'])
                        glacier_mask = (glacier_df['date'] >= start_dt) & (glacier_df['date'] <= end_dt)
                        glacier_filtered = glacier_df[glacier_mask].copy()
                        glacier_m3s_col = f"{gauge_id} m3/s"
                        if glacier_m3s_col in glacier_filtered.columns:
                            glacier_filtered['glacier_mm'] = glacier_filtered[glacier_m3s_col] * conversion_m3s_to_mm_day
                            glacier_filtered['year'] = glacier_filtered['date'].dt.year
                            mean_glacier_melt = glacier_filtered.groupby('year')['glacier_mm'].sum().mean()
            except Exception:
                pass

            # Load precipitation
            mean_precip = 0
            try:
                snowfall_df = load_forcing_by_hrugroup(individual_config, 'SNOWFALL')
                rainfall_df = load_forcing_by_hrugroup(individual_config, 'RAINFALL')
                if snowfall_df is not None and rainfall_df is not None:
                    precip_df = pd.merge(
                        snowfall_df.rename(columns={'NO_GLACIER': 'snowfall'}),
                        rainfall_df.rename(columns={'NO_GLACIER': 'rainfall'}),
                        on='date', how='inner'
                    )
                    precip_mask = (precip_df['date'] >= start_dt) & (precip_df['date'] <= end_dt)
                    precip_period = precip_df[precip_mask].copy()

                    # Scale by non-glacier fraction
                    area_fraction = 1.0
                    if 'Landuse_Cl' in hru_gdf.columns:
                        non_glacier_area = hru_gdf[~hru_gdf['Landuse_Cl'].isin([7, 8])]['Area_km2'].sum()
                        area_fraction = non_glacier_area / total_area_km2 if total_area_km2 > 0 else 1.0

                    precip_period['total'] = (precip_period['snowfall'] + precip_period['rainfall']) * area_fraction
                    precip_period['year'] = precip_period['date'].dt.year
                    mean_precip = precip_period.groupby('year')['total'].sum().mean()
            except Exception:
                pass

            config_results[config_dir] = {
                'mean_obs_Q': mean_obs,
                'mean_sim_Q': mean_sim,
                'mean_snowmelt': mean_snowmelt,
                'mean_glacier_melt': mean_glacier_melt,
                'mean_precip': mean_precip,
                'annual_data': annual,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir),
            }
            print(f"  + {config_names.get(config_dir, config_dir)}: Q_obs={mean_obs:.0f}, Q_sim={mean_sim:.0f}, P={mean_precip:.0f}, Glacier={mean_glacier_melt:.0f} mm/yr")

        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue

    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None

    n_configs = len(config_results)
    n_rows, n_cols, figsize = _calc_subplot_layout(n_configs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    if n_configs == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    for i, (config_dir, result) in enumerate(config_results.items()):
        ax = axes[i]
        color = result['color']

        components = ['Obs Q', 'Sim Q', 'Precip', 'Snowmelt', 'Glacier']
        values = [result['mean_obs_Q'], result['mean_sim_Q'], result['mean_precip'], result['mean_snowmelt'], result['mean_glacier_melt']]
        bar_colors = ['black', color, 'steelblue', 'deepskyblue', '#4292c6']

        bars = ax.bar(components, values, color=bar_colors, edgecolor='black', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=10)

        ax.set_title(result['name'], fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=30, labelsize=11)

        if i % n_cols == 0:
            ax.set_ylabel('mm/year', fontsize=13, fontweight='bold')

    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    save_path = plot_dir / f'water_balance_subplots_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved water balance subplots to: {save_path}")
    plt.show()

    return config_results


#--------------------------------------------------------------------------------
############ hydrograph timeseries subplots (configs as rows) ###################
#--------------------------------------------------------------------------------

def plot_hydrograph_timeseries_subplots(multi_config, validation_start=None, validation_end=None,
                                        random_seed=42, n_years=2):
    """
    Plot hydrograph time series with each configuration as a row and years as columns.
    Observed streamflow in black, simulated in config color.

    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    random_seed : int
        Random seed for reproducible year selection
    n_years : int
        Number of years to plot as columns (default: 2)

    Returns:
    --------
    dict
        Dictionary with selected years and config results
    """

    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')

    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']

    print(f"Creating hydrograph timeseries subplots (configs as rows) for {len(configs)} configurations")
    print(f"  - Validation period: {validation_start} to {validation_end}")
    print(f"  - Number of years (columns): {n_years}")

    plot_dir = create_multi_plot_dir(multi_config)

    # First pass: load data and find common years
    config_results = {}
    available_years = None

    for config_dir in configs:
        individual_config = _build_individual_config(multi_config, config_dir)
        try:
            data = load_hydrograph_data(individual_config)
            if data is None:
                continue

            val_mask = (data['date'] >= pd.to_datetime(validation_start)) & \
                       (data['date'] <= pd.to_datetime(validation_end))
            df_val = data[val_mask].copy()

            if len(df_val) == 0:
                continue

            # Only include years with at least 30 days of data
            year_counts = df_val['date'].dt.year.value_counts()
            val_years = set(year_counts[year_counts >= 30].index)

            config_results[config_dir] = {
                'data': data,
                'val_years': val_years,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir),
            }
            print(f"  + {config_names.get(config_dir, config_dir)}: years {sorted(val_years)}")

        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue

    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None

    # Find years available in the majority of configs (>50%)
    from collections import Counter
    year_counter = Counter()
    for result in config_results.values():
        year_counter.update(result['val_years'])
    threshold = len(config_results) * 0.5
    available_years = {y for y, count in year_counter.items() if count >= threshold}

    if not available_years or len(available_years) < n_years:
        print(f"Not enough common years (need {n_years}). Found: {sorted(available_years) if available_years else 'None'}")
        if available_years and len(available_years) > 0:
            n_years = len(available_years)
        else:
            return None

    # Select years from the middle of the validation period
    sorted_years = sorted(available_years)
    mid = len(sorted_years) // 2
    # Take n_years centered around the middle
    half = n_years // 2
    start_idx = max(0, mid - half)
    # Ensure we don't go past the end, but also avoid first/last years
    if start_idx + n_years > len(sorted_years):
        start_idx = len(sorted_years) - n_years
    # Avoid picking the very first or last year if possible
    if len(sorted_years) > n_years + 1:
        start_idx = max(1, start_idx)
        if start_idx + n_years >= len(sorted_years):
            start_idx = len(sorted_years) - n_years - 1
    selected_years = sorted_years[start_idx:start_idx + n_years]
    print(f"\nSelected years (middle of validation): {selected_years}")

    # Create figure: rows = configs, cols = years
    n_configs = len(config_results)
    fig, axes = plt.subplots(n_configs, n_years, figsize=(8 * n_years, 3.5 * n_configs),
                              sharex='col', sharey=True)

    # Handle edge cases for axes shape
    if n_configs == 1 and n_years == 1:
        axes = np.array([[axes]])
    elif n_configs == 1:
        axes = axes[np.newaxis, :]
    elif n_years == 1:
        axes = axes[:, np.newaxis]

    for row_idx, (config_dir, result) in enumerate(config_results.items()):
        data = result['data']
        color = result['color']
        name = result['name']

        for col_idx, year in enumerate(selected_years):
            ax = axes[row_idx, col_idx]

            # Filter for this year within validation period
            year_mask = (data['date'].dt.year == year) & \
                        (data['date'] >= pd.to_datetime(validation_start)) & \
                        (data['date'] <= pd.to_datetime(validation_end))
            year_data = data[year_mask].copy()

            if len(year_data) == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
                continue

            # Plot observed
            if 'obs_Q' in year_data.columns:
                ax.plot(year_data['date'], year_data['obs_Q'], 'k-',
                        linewidth=2, label='Observed', zorder=10)

            # Plot simulated
            if 'sim_Q' in year_data.columns:
                ax.plot(year_data['date'], year_data['sim_Q'], '--',
                        color=color, linewidth=2, label='Simulated', zorder=5)

            ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

            # Column header (year) on top row
            if row_idx == 0:
                ax.set_title(f'{year}', fontsize=16, fontweight='bold')

            # Row label (config name) on right side
            if col_idx == n_years - 1:
                ax.annotate(name, xy=(1.02, 0.5), xycoords='axes fraction',
                           fontsize=12, fontweight='bold', va='center', rotation=-90)

            # Y-axis label on leftmost column
            if col_idx == 0:
                ax.set_ylabel('Q (m³/s)', fontsize=12)

            # Legend only on first cell
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper right', fontsize=10)

            # Format x-axis dates
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    plt.tight_layout()
    plt.subplots_adjust(right=0.92)  # Make room for row labels

    save_path = plot_dir / f'hydrograph_timeseries_subplots_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved hydrograph timeseries subplots to: {save_path}")
    plt.show()

    return {'selected_years': selected_years, 'config_results': config_results}


#--------------------------------------------------------------------------------
########## catchment-average SWE timeseries (all configs overlay) ###############
#--------------------------------------------------------------------------------

def plot_swe_catchment_average_comparison(multi_config, validation_start=None, validation_end=None):
    """
    Plot catchment-average SWE time series for all configurations overlaid on one plot.
    Reads SWE directly from the ByHRUGroup CSV (AllHRUs column) to avoid load_swe_data issues.

    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period

    Returns:
    --------
    dict
        Dictionary containing SWE data for each configuration
    """

    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')

    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']
    model_type = multi_config.get('model_type', 'HBV')

    print(f"Creating catchment-average SWE comparison for {len(configs)} configurations")
    print(f"  - Validation period: {validation_start} to {validation_end}")

    plot_dir = create_multi_plot_dir(multi_config)
    config_results = {}

    val_start = pd.to_datetime(validation_start)
    val_end = pd.to_datetime(validation_end)

    for config_dir in configs:
        try:
            ind_config = _build_individual_config(multi_config, config_dir)
            paths = get_paths(ind_config)
            swe_file = paths['output_dir'] / f"{gauge_id}_{model_type}_SNOW_Daily_Average_ByHRUGroup.csv"

            if not swe_file.exists():
                print(f"  Skipping {config_names.get(config_dir, config_dir)}: SWE file not found")
                continue

            # Read the CSV - skip the units row (row 2)
            df = pd.read_csv(swe_file, skiprows=[1])

            # Parse dates from the second column (HRUGroup: or similar)
            date_col = df.columns[1]
            df['date'] = pd.to_datetime(df[date_col])

            # Get the AllHRUs column (catchment-average SWE)
            if 'AllHRUs' not in df.columns:
                print(f"  Skipping {config_dir}: no AllHRUs column")
                continue

            df['swe'] = pd.to_numeric(df['AllHRUs'], errors='coerce')

            # Filter for validation period
            mask = (df['date'] >= val_start) & (df['date'] <= val_end)
            df_val = df[mask][['date', 'swe']].dropna().copy()

            if len(df_val) == 0:
                print(f"  Skipping {config_dir}: no data in validation period")
                continue

            config_results[config_dir] = {
                'date': df_val['date'].values,
                'swe': df_val['swe'].values,
                'mean_swe': df_val['swe'].mean(),
                'max_swe': df_val['swe'].max(),
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir),
            }
            print(f"  + {config_names.get(config_dir, config_dir)}: mean={df_val['swe'].mean():.1f} mm, max={df_val['swe'].max():.1f} mm")

        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue

    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None

    # Create overlay plot
    fig, ax = plt.subplots(figsize=(18, 8))

    for config_dir, result in config_results.items():
        ax.plot(result['date'], result['swe'],
                color=result['color'], linewidth=1.8, label=result['name'], alpha=0.85)

    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Snow Water Equivalent (mm)', fontsize=14, fontweight='bold')
    ax.set_title(f'Catchment-Average SWE - All Configurations\nCatchment {gauge_id} ({validation_start} to {validation_end})',
                 fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.legend(loc='upper right', fontsize=9, ncol=2)

    # Format x-axis
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()

    plt.tight_layout()

    save_path = plot_dir / f'swe_catchment_average_comparison_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved catchment-average SWE comparison to: {save_path}")
    plt.show()

    return config_results


#--------------------------------------------------------------------------------
##################### AET subplots ##############################################
#--------------------------------------------------------------------------------

def plot_aet_subplots(multi_config, validation_start=None, validation_end=None):
    """
    Plot monthly actual evapotranspiration (AET) for each configuration in separate subplots.
    AET is scaled by non-glacier area fraction.

    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for analysis period
    validation_end : str, optional
        End date for analysis period

    Returns:
    --------
    dict
        Dictionary containing AET data for each configuration
    """

    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')

    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']
    model_type = multi_config.get('model_type', 'HBV')

    print(f"Creating AET subplots for {len(configs)} configurations")

    plot_dir = create_multi_plot_dir(multi_config)
    config_results = {}

    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for config_dir in configs:
        try:
            ind_config = _build_individual_config(multi_config, config_dir)
            paths = get_paths(ind_config)
            output_dir = paths['output_dir']

            # Load AET data
            aet_file = output_dir / f"{gauge_id}_{model_type}_AET_Daily_Average_ByHRUGroup.csv"
            if not aet_file.exists():
                print(f"  Skipping {config_names.get(config_dir, config_dir)}: AET file not found")
                continue

            df = pd.read_csv(aet_file, skiprows=[1])
            if 'HRUGroup:' in df.columns:
                df = df.rename(columns={'HRUGroup:': 'date'})
            else:
                continue
            df['date'] = pd.to_datetime(df['date'])

            if 'NO_GLACIER' not in df.columns:
                print(f"  Skipping {config_names.get(config_dir, config_dir)}: NO_GLACIER column missing")
                continue

            df = df.rename(columns={'NO_GLACIER': 'aet'})

            # Load area scaling
            topo_dir = paths['topo_dir']
            hru_shapefile = topo_dir / "HRU.shp"

            area_fraction = 1.0
            total_area_km2 = 0.0
            non_glacier_area_km2 = 0.0
            if hru_shapefile.exists():
                import geopandas as gpd
                hru_gdf = gpd.read_file(hru_shapefile)
                total_area_km2 = hru_gdf['Area_km2'].sum()
                if 'Landuse_Cl' in hru_gdf.columns:
                    non_glacier_area_km2 = hru_gdf[~hru_gdf['Landuse_Cl'].isin([7, 8])]['Area_km2'].sum()
                    area_fraction = non_glacier_area_km2 / total_area_km2 if total_area_km2 > 0 else 1.0
                else:
                    non_glacier_area_km2 = total_area_km2

            # Filter period
            start_dt = pd.to_datetime(validation_start)
            end_dt = pd.to_datetime(validation_end)
            mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
            df_period = df[mask].copy()

            if len(df_period) == 0:
                continue

            # Scale AET
            df_period['aet_scaled'] = df_period['aet'] * area_fraction
            df_period['month'] = df_period['date'].dt.month
            monthly_aet = df_period.groupby('month')['aet_scaled'].mean()

            config_results[config_dir] = {
                'monthly_aet': monthly_aet,
                'area_fraction': area_fraction,
                'mean_daily_aet': df_period['aet_scaled'].mean(),
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir),
            }
            print(f"  + {config_names.get(config_dir, config_dir)}: mean AET={df_period['aet_scaled'].mean():.3f} mm/day")

        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue

    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None

    n_configs = len(config_results)
    n_rows, n_cols, figsize = _calc_subplot_layout(n_configs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    if n_configs == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    for i, (config_dir, result) in enumerate(config_results.items()):
        ax = axes[i]
        monthly_aet = result['monthly_aet']

        ax.bar(months, [monthly_aet.get(m, 0) for m in months],
               color='forestgreen', edgecolor='darkgreen', linewidth=0.5)

        ax.set_title(result['name'], fontsize=14, fontweight='bold')
        ax.set_xticks(months)
        ax.set_xticklabels(month_names, rotation=45, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add mean daily AET annotation
        ax.text(0.98, 0.95, f"Mean: {result['mean_daily_aet']:.3f} mm/d\nArea frac: {result['area_fraction']:.2f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if i % n_cols == 0:
            ax.set_ylabel('AET (mm/day)', fontsize=13, fontweight='bold')

    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Monthly Mean AET (Scaled to Catchment Area) - Catchment {gauge_id}\n{validation_start} to {validation_end}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = plot_dir / f'aet_subplots_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved AET subplots to: {save_path}")
    plt.show()

    return config_results


#--------------------------------------------------------------------------------
################# precipitation and AET combined subplots #######################
#--------------------------------------------------------------------------------

def plot_precipitation_and_aet_combined_subplots(multi_config, validation_start=None, validation_end=None):
    """
    Plot monthly precipitation (stacked rain/snow) and AET side by side for each configuration.

    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for analysis period
    validation_end : str, optional
        End date for analysis period

    Returns:
    --------
    dict
        Dictionary containing precip+AET data for each configuration
    """

    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')

    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']
    model_type = multi_config.get('model_type', 'HBV')

    print(f"Creating precipitation + AET combined subplots for {len(configs)} configurations")

    plot_dir = create_multi_plot_dir(multi_config)
    config_results = {}

    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for config_dir in configs:
        individual_config = _build_individual_config(multi_config, config_dir)
        try:
            # Load snowfall and rainfall
            snowfall_df = load_forcing_by_hrugroup(individual_config, 'SNOWFALL')
            rainfall_df = load_forcing_by_hrugroup(individual_config, 'RAINFALL')

            if snowfall_df is None or rainfall_df is None:
                print(f"  Skipping {config_names.get(config_dir, config_dir)}: missing forcing data")
                continue

            # Load AET data
            paths = get_paths(individual_config)
            output_dir = paths['output_dir']
            aet_file = output_dir / f"{gauge_id}_{model_type}_AET_Daily_Average_ByHRUGroup.csv"

            if not aet_file.exists():
                print(f"  Skipping {config_names.get(config_dir, config_dir)}: AET file not found")
                continue

            aet_df = pd.read_csv(aet_file, skiprows=[1])
            if 'HRUGroup:' in aet_df.columns:
                aet_df = aet_df.rename(columns={'HRUGroup:': 'date'})
            else:
                continue
            aet_df['date'] = pd.to_datetime(aet_df['date'])
            if 'NO_GLACIER' not in aet_df.columns:
                continue
            aet_df = aet_df.rename(columns={'NO_GLACIER': 'aet'})

            # Merge all datasets
            df = pd.merge(
                snowfall_df.rename(columns={'NO_GLACIER': 'snowfall'}),
                rainfall_df.rename(columns={'NO_GLACIER': 'rainfall'}),
                on='date', how='inner'
            )
            df = pd.merge(df, aet_df[['date', 'aet']], on='date', how='inner')

            # Filter period
            start_dt = pd.to_datetime(validation_start)
            end_dt = pd.to_datetime(validation_end)
            mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
            df_period = df[mask].copy()

            if len(df_period) == 0:
                continue

            # Load area scaling
            topo_dir = paths['topo_dir']
            hru_shapefile = topo_dir / "HRU.shp"

            area_fraction = 1.0
            if hru_shapefile.exists():
                import geopandas as gpd
                hru_gdf = gpd.read_file(hru_shapefile)
                total_area = hru_gdf['Area_km2'].sum()
                if 'Landuse_Cl' in hru_gdf.columns:
                    non_glacier_area = hru_gdf[~hru_gdf['Landuse_Cl'].isin([7, 8])]['Area_km2'].sum()
                    area_fraction = non_glacier_area / total_area if total_area > 0 else 1.0

            # Scale values
            df_period['snowfall_scaled'] = df_period['snowfall'] * area_fraction
            df_period['rainfall_scaled'] = df_period['rainfall'] * area_fraction
            df_period['aet_scaled'] = df_period['aet'] * area_fraction
            df_period['month'] = df_period['date'].dt.month

            monthly_snow = df_period.groupby('month')['snowfall_scaled'].mean()
            monthly_rain = df_period.groupby('month')['rainfall_scaled'].mean()
            monthly_aet = df_period.groupby('month')['aet_scaled'].mean()

            config_results[config_dir] = {
                'monthly_snow': monthly_snow,
                'monthly_rain': monthly_rain,
                'monthly_aet': monthly_aet,
                'area_fraction': area_fraction,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir),
            }
            mean_precip = monthly_snow.mean() + monthly_rain.mean()
            print(f"  + {config_names.get(config_dir, config_dir)}: precip={mean_precip:.3f}, AET={monthly_aet.mean():.3f} mm/day")

        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue

    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None

    n_configs = len(config_results)
    n_rows, n_cols, figsize = _calc_subplot_layout(n_configs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    if n_configs == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    for i, (config_dir, result) in enumerate(config_results.items()):
        ax = axes[i]
        monthly_rain = result['monthly_rain']
        monthly_snow = result['monthly_snow']
        monthly_aet = result['monthly_aet']

        bar_width = 0.35
        x_pos = np.arange(1, 13)

        # Precipitation bars (stacked)
        ax.bar(x_pos - bar_width/2, [monthly_rain.get(m, 0) for m in months], bar_width,
               color='steelblue', label='Rainfall', edgecolor='navy', linewidth=0.5)
        ax.bar(x_pos - bar_width/2, [monthly_snow.get(m, 0) for m in months], bar_width,
               bottom=[monthly_rain.get(m, 0) for m in months],
               color='lightcyan', label='Snowfall', edgecolor='darkblue', linewidth=0.5)

        # AET bars
        ax.bar(x_pos + bar_width/2, [monthly_aet.get(m, 0) for m in months], bar_width,
               color='forestgreen', label='AET', edgecolor='darkgreen', linewidth=0.5)

        ax.set_title(result['name'], fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(month_names, rotation=45, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        if i == 0:
            ax.legend(fontsize=9)
        if i % n_cols == 0:
            ax.set_ylabel('Water Flux (mm/day)', fontsize=13, fontweight='bold')

    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Monthly Precipitation and AET (Scaled to Catchment Area) - Catchment {gauge_id}\n{validation_start} to {validation_end}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = plot_dir / f'precipitation_aet_combined_subplots_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved precipitation + AET combined subplots to: {save_path}")
    plt.show()

    return config_results


#--------------------------------------------------------------------------------
################# GloGEM vs observed regime subplots #############################
#--------------------------------------------------------------------------------

def plot_glogem_vs_observed_regime_subplots(multi_config, validation_start=None, validation_end=None,
                                             min_data_threshold=0.95):
    """
    Plot GloGEM catchment-average melt regime vs observed runoff for each coupled configuration.
    Each subplot shows 2-panel: total GloGEM vs observed bars, and stacked GloGEM components.
    Skips non-coupled configs (baseline, deltah) since they don't have GloGEM data.

    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    validation_start : str, optional
        Start date for analysis period
    validation_end : str, optional
        End date for analysis period
    min_data_threshold : float, optional
        Minimum fraction of valid data per year (default: 0.95)

    Returns:
    --------
    dict
        Dictionary containing GloGEM vs observed regime data for each configuration
    """

    if validation_start is None:
        validation_start = multi_config.get('start_date', '2000-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')

    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_colors = multi_config['config_colors']
    config_names = multi_config['config_names']
    config_coupled = multi_config.get('config_coupled', {})

    print(f"Creating GloGEM vs observed regime subplots for {len(configs)} configurations")

    plot_dir = create_multi_plot_dir(multi_config)
    config_results = {}

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    start = pd.to_datetime(validation_start)
    end = pd.to_datetime(validation_end)

    for config_dir in configs:
        # Only process coupled configs (they have GloGEM data)
        if not config_coupled.get(config_dir, False):
            print(f"  Skipping {config_names.get(config_dir, config_dir)}: not coupled (no GloGEM data)")
            continue

        individual_config = _build_individual_config(multi_config, config_dir)

        try:
            # Load GloGEM data
            glogem_df = load_glogem_data(individual_config, unit='mm', plot=False)
            if glogem_df is None:
                print(f"  Skipping {config_names.get(config_dir, config_dir)}: no GloGEM data")
                continue

            glogem_df = glogem_df[(glogem_df['date'] >= start) & (glogem_df['date'] <= end)].copy()

            # Load observed runoff
            hydro_df = load_hydrograph_data(individual_config)
            if hydro_df is None or 'obs_Q' not in hydro_df.columns:
                print(f"  Skipping {config_names.get(config_dir, config_dir)}: no observed data")
                continue

            hydro_df = hydro_df[(hydro_df['date'] >= start) & (hydro_df['date'] <= end)].copy()

            # Get catchment area for Q conversion
            paths = get_paths(individual_config)
            topo_dir = paths['topo_dir']
            hru_shapefile = topo_dir / "HRU.shp"

            if not hru_shapefile.exists():
                print(f"  Skipping {config_names.get(config_dir, config_dir)}: HRU shapefile not found")
                continue

            import geopandas as gpd
            hru_gdf = gpd.read_file(hru_shapefile)
            catchment_area_km2 = hru_gdf['Area_km2'].sum()

            # Convert Q from m3/s to mm/day
            conversion_factor = 86400.0 / (catchment_area_km2 * 1e6) * 1000.0
            hydro_df['obs_Q_mm'] = hydro_df['obs_Q'] * conversion_factor

            # Merge
            merged_df = pd.merge(
                glogem_df[['date', 'glacier_melt_normalized', 'icemelt_normalized',
                           'snowmelt_normalized', 'rainfall_normalized']],
                hydro_df[['date', 'obs_Q_mm']],
                on='date', how='inner'
            )

            if len(merged_df) == 0:
                continue

            merged_df['year'] = merged_df['date'].dt.year
            merged_df['month'] = merged_df['date'].dt.month

            # Filter years by data availability
            valid_years = []
            for year in sorted(merged_df['year'].unique()):
                year_start = max(pd.Timestamp(f"{year}-01-01"), start)
                year_end = min(pd.Timestamp(f"{year}-12-31"), end)
                expected_days = (year_end - year_start).days + 1
                year_data = merged_df[merged_df['year'] == year]
                glogem_avail = year_data['glacier_melt_normalized'].notna().sum() / expected_days
                obs_avail = year_data['obs_Q_mm'].notna().sum() / expected_days
                if glogem_avail >= min_data_threshold and obs_avail >= min_data_threshold:
                    valid_years.append(year)

            if not valid_years:
                print(f"  Skipping {config_names.get(config_dir, config_dir)}: no valid years")
                continue

            merged_filtered = merged_df[merged_df['year'].isin(valid_years)].copy()

            # Monthly regime
            monthly_regime = merged_filtered.groupby('month').agg({
                'glacier_melt_normalized': 'mean',
                'icemelt_normalized': 'mean',
                'snowmelt_normalized': 'mean',
                'rainfall_normalized': 'mean',
                'obs_Q_mm': 'mean'
            }).reset_index()

            # Annual totals
            days_per_month = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            monthly_regime['days'] = monthly_regime['month'].apply(lambda m: days_per_month[m-1])
            glogem_total = (monthly_regime['glacier_melt_normalized'] * monthly_regime['days']).sum()
            obs_total = (monthly_regime['obs_Q_mm'] * monthly_regime['days']).sum()
            ratio = glogem_total / obs_total if obs_total > 0 else float('nan')

            config_results[config_dir] = {
                'monthly_regime': monthly_regime,
                'glogem_total': glogem_total,
                'obs_total': obs_total,
                'ratio': ratio,
                'valid_years': valid_years,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir),
            }
            print(f"  + {config_names.get(config_dir, config_dir)}: GloGEM={glogem_total:.0f}, Obs={obs_total:.0f} mm/yr, ratio={ratio:.2%}")

        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue

    if len(config_results) == 0:
        print("No coupled configurations processed successfully")
        return None

    # Each config gets 2 rows (GloGEM vs obs, component breakdown)
    n_configs = len(config_results)
    n_cols = min(n_configs, 4)
    n_rows = (n_configs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(6 * n_cols, 5 * n_rows * 2))

    if n_rows * 2 == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows * 2 == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, (config_dir, result) in enumerate(config_results.items()):
        col = i % n_cols
        row_base = (i // n_cols) * 2
        ax1 = axes[row_base, col]
        ax2 = axes[row_base + 1, col]

        regime = result['monthly_regime']
        x = regime['month']

        # Panel 1: GloGEM total vs observed
        width = 0.35
        ax1.bar(x - width/2, regime['glacier_melt_normalized'], width,
                label='GloGEM Total', color='steelblue', alpha=0.8)
        ax1.bar(x + width/2, regime['obs_Q_mm'], width,
                label='Observed', color='coral', alpha=0.8)

        ax1.set_title(result['name'], fontsize=13, fontweight='bold')
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(month_labels, fontsize=8, rotation=45)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylabel('mm/day', fontsize=11)

        stats_text = f"GloGEM: {result['glogem_total']:.0f}\nObs: {result['obs_total']:.0f}\nRatio: {result['ratio']:.0%}"
        ax1.text(0.98, 0.95, stats_text, transform=ax1.transAxes,
                 fontsize=8, va='top', ha='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        if i == 0:
            ax1.legend(fontsize=8, loc='upper left')

        # Panel 2: GloGEM components (stacked)
        icemelt = regime['icemelt_normalized'].values
        snowmelt = regime['snowmelt_normalized'].values
        rain = regime['rainfall_normalized'].values

        ax2.bar(x, icemelt, color='#4292c6', alpha=0.85, label='Ice Melt')
        ax2.bar(x, snowmelt, bottom=icemelt, color='#9ecae1', alpha=0.85, label='Snow Melt')
        ax2.bar(x, rain, bottom=icemelt + snowmelt, color='#74c476', alpha=0.85, label='Rain')

        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(month_labels, fontsize=8, rotation=45)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylabel('mm/day', fontsize=11)

        if i == 0:
            ax2.legend(fontsize=8, loc='upper left')

    # Hide unused subplots
    for i in range(n_configs, n_rows * n_cols):
        col = i % n_cols
        row_base = (i // n_cols) * 2
        axes[row_base, col].set_visible(False)
        axes[row_base + 1, col].set_visible(False)

    fig.suptitle(f'GloGEM vs Observed Runoff Regime (Coupled Configs) - Catchment {gauge_id}\n{validation_start} to {validation_end}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = plot_dir / f'glogem_vs_observed_regime_subplots_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved GloGEM vs observed regime subplots to: {save_path}")
    plt.show()

    return config_results


#--------------------------------------------------------------------------------
##################### orchestrator ##############################################
#--------------------------------------------------------------------------------

def run_complete_multi_postprocessing(configurations_yaml_path, gauge_id,
                                      validation_start=None, validation_end=None,
                                      skip_errors=True, unit='mm'):
    """
    Run complete multi-configuration postprocessing analysis.

    Loads all configurations from the YAML registry and generates all comparison
    plots for the specified catchment.

    Parameters:
    -----------
    configurations_yaml_path : str or Path
        Path to configurations.yaml
    gauge_id : str
        Gauge identifier (e.g., '0101')
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    skip_errors : bool, optional
        If True, continue even if individual plot functions fail (default: True)
    unit : str, optional
        Unit for regime plots ('mm' or 'm3'), default 'mm'

    Returns:
    --------
    dict
        Dictionary with results from each plot function and error summary
    """

    import time
    start_time = time.time()

    print("=" * 80)
    print("RUNNING COMPLETE MULTI-CONFIGURATION POSTPROCESSING")
    print("=" * 80)

    # Step 1: Load configurations
    multi_config = load_configurations(configurations_yaml_path, gauge_id)
    if multi_config is None:
        print("ERROR: Failed to load configurations")
        return None

    # Override validation dates if provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2020-12-31')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2022-12-31')

    print(f"\nValidation period: {validation_start} to {validation_end}")
    print(f"Number of configurations: {len(multi_config['configs'])}")
    print("=" * 80 + "\n")

    results = {}
    errors = []

    def run_function(func_name, func, *args, **kwargs):
        print(f"\n{'=' * 80}")
        print(f"Running: {func_name}")
        print(f"{'=' * 80}")
        try:
            result = func(*args, **kwargs)
            print(f"  {func_name} completed successfully")
            return result
        except Exception as e:
            error_msg = f"  ERROR in {func_name}: {str(e)}"
            print(error_msg)
            if not skip_errors:
                raise
            errors.append({'function': func_name, 'error': str(e)})
            import traceback
            traceback.print_exc()
            return None

    # ========================================================================
    # 1. STREAMFLOW COMPARISONS (overlay plots)
    # ========================================================================
    print("\n" + "#" * 80)
    print("# 1. STREAMFLOW ANALYSIS")
    print("#" * 80)

    results['hydrological_regime_comparison'] = run_function(
        'plot_hydrological_regime_comparison',
        plot_hydrological_regime_comparison,
        multi_config, validation_start, validation_end
    )

    results['hydrological_regime_subplots'] = run_function(
        'plot_hydrological_regime_subplots',
        plot_hydrological_regime_subplots,
        multi_config, validation_start, validation_end, unit
    )

    results['hydrograph_timeseries_comparison'] = run_function(
        'plot_hydrograph_timeseries_comparison',
        plot_hydrograph_timeseries_comparison,
        multi_config, validation_start, validation_end
    )

    results['hydrograph_timeseries_subplots'] = run_function(
        'plot_hydrograph_timeseries_subplots',
        plot_hydrograph_timeseries_subplots,
        multi_config, validation_start, validation_end
    )

    results['streamflow_scatter_subplots'] = run_function(
        'plot_streamflow_scatter_subplots',
        plot_streamflow_scatter_subplots,
        multi_config, validation_start, validation_end
    )

    results['streamflow_residuals_subplots'] = run_function(
        'plot_streamflow_residuals_subplots',
        plot_streamflow_residuals_subplots,
        multi_config, validation_start, validation_end
    )

    results['streamflow_metrics_comparison'] = run_function(
        'plot_streamflow_metrics_comparison',
        plot_streamflow_metrics_comparison,
        multi_config, validation_start, validation_end
    )

    # ========================================================================
    # 2. SWE ANALYSIS
    # ========================================================================
    print("\n" + "#" * 80)
    print("# 2. SNOW WATER EQUIVALENT (SWE) ANALYSIS")
    print("#" * 80)

    results['swe_timeseries_comparison'] = run_function(
        'plot_swe_timeseries_comparison',
        plot_swe_timeseries_comparison,
        multi_config, validation_start, validation_end
    )

    results['swe_elevation_bands_comparison'] = run_function(
        'plot_swe_elevation_bands_comparison',
        plot_swe_elevation_bands_comparison,
        multi_config, validation_start, validation_end
    )

    results['swe_catchment_average_comparison'] = run_function(
        'plot_swe_catchment_average_comparison',
        plot_swe_catchment_average_comparison,
        multi_config, validation_start, validation_end
    )

    # ========================================================================
    # 3. PRECIPITATION ANALYSIS
    # ========================================================================
    print("\n" + "#" * 80)
    print("# 3. PRECIPITATION ANALYSIS")
    print("#" * 80)

    results['precipitation_partitioning_subplots'] = run_function(
        'plot_precipitation_partitioning_subplots',
        plot_precipitation_partitioning_subplots,
        multi_config, validation_start, validation_end
    )

    results['aet_subplots'] = run_function(
        'plot_aet_subplots',
        plot_aet_subplots,
        multi_config, validation_start, validation_end
    )

    results['precipitation_aet_combined_subplots'] = run_function(
        'plot_precipitation_and_aet_combined_subplots',
        plot_precipitation_and_aet_combined_subplots,
        multi_config, validation_start, validation_end
    )

    # ========================================================================
    # 4. SNOWMELT AND GLACIER MELT ANALYSIS
    # ========================================================================
    print("\n" + "#" * 80)
    print("# 4. SNOWMELT AND GLACIER MELT ANALYSIS")
    print("#" * 80)

    results['snowmelt_regime_subplots'] = run_function(
        'plot_snowmelt_regime_subplots',
        plot_snowmelt_regime_subplots,
        multi_config, validation_start, validation_end, unit
    )

    results['glacier_melt_regime_subplots'] = run_function(
        'plot_glacier_melt_regime_subplots',
        plot_glacier_melt_regime_subplots,
        multi_config, validation_start, validation_end, unit
    )

    results['streamflow_glogem_snowmelt_regime_subplots'] = run_function(
        'plot_streamflow_glogem_snowmelt_regime_subplots',
        plot_streamflow_glogem_snowmelt_regime_subplots,
        multi_config, validation_start, validation_end, unit
    )

    results['glogem_vs_observed_regime_subplots'] = run_function(
        'plot_glogem_vs_observed_regime_subplots',
        plot_glogem_vs_observed_regime_subplots,
        multi_config, validation_start, validation_end
    )

    # ========================================================================
    # 5. WATER BALANCE
    # ========================================================================
    print("\n" + "#" * 80)
    print("# 5. WATER BALANCE ANALYSIS")
    print("#" * 80)

    results['water_balance_subplots'] = run_function(
        'plot_water_balance_subplots',
        plot_water_balance_subplots,
        multi_config, validation_start, validation_end
    )

    # ========================================================================
    # 6. PARAMETER AND STORAGE ANALYSIS
    # ========================================================================
    print("\n" + "#" * 80)
    print("# 6. PARAMETER AND STORAGE ANALYSIS")
    print("#" * 80)

    results['parameter_boxplots_comparison'] = run_function(
        'plot_parameter_boxplots_comparison',
        plot_parameter_boxplots_comparison,
        multi_config
    )

    results['storage_timeseries_comparison'] = run_function(
        'plot_storage_timeseries_comparison',
        plot_storage_timeseries_comparison,
        multi_config, validation_start, validation_end
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("MULTI-CONFIGURATION POSTPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Catchment: {gauge_id}")
    print(f"Validation: {validation_start} to {validation_end}")
    print(f"Configurations: {len(multi_config['configs'])}")
    print(f"Time elapsed: {elapsed:.1f} seconds")

    n_success = sum(1 for v in results.values() if v is not None)
    n_total = len(results)
    n_failed = len(errors)

    print(f"\nPlots: {n_success}/{n_total} succeeded, {n_failed} failed")

    if errors:
        print(f"\nFailed plots:")
        for err in errors:
            print(f"  - {err['function']}: {err['error'][:80]}")

    print("=" * 80)

    results['_errors'] = errors
    results['_multi_config'] = multi_config
    return results