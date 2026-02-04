# This script is postprocessing Raven output from a multiple model configurations using namelist
# August 2025

#--------------------------------------------------------------------------------
################################## packages #####################################
#--------------------------------------------------------------------------------

import postprocessing_single
import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use('Agg')  # Use non-interactive backend to prevent image viewer
import matplotlib.pyplot as plt
from pathlib import Path

#--------------------------------------------------------------------------------
################################### general #####################################
#--------------------------------------------------------------------------------

def create_multi_plot_dir(multi_config):
    """
    Create plot directory for multi-configuration analysis in the main directory.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
        
    Returns:
    --------
    Path
        Path to the multi-configuration plot directory
    """
    
    main_dir = Path(multi_config['main_dir'])
    gauge_id = multi_config['gauge_id']
    
    # Create plot directory in main_dir (where all config folders are)
    plot_dir = main_dir / f"multi_config_plots_catchment_{gauge_id}"
    plot_dir.mkdir(exist_ok=True)
    
    print(f"Created multi-configuration plot directory: {plot_dir}")
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
        
        # Create individual config dictionary for this configuration
        individual_config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': config_dir,
            'gauge_id': gauge_id,
            'start_date': multi_config.get('start_date', '2000-01-01'),
            'end_date': multi_config.get('end_date', '2020-12-31'),
            'cali_end_date': multi_config.get('cali_end_date', '2009-12-31'),
            'model_type': model_type,
            'coupled': 'coupled' in config_dir.lower(),  # Assume coupled if 'coupled' in name
        }
        
        try:
            # Load hydrograph data for this configuration
            data = postprocessing_single.load_hydrograph_data(individual_config)
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

def plot_hydrological_regime_subplots(multi_config, validation_start=None, validation_end=None):
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
    
    print(f"Creating individual hydrological regime plots for {len(configs)} configurations:")
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
        
        # Create individual config dictionary for this configuration
        individual_config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': config_dir,
            'gauge_id': gauge_id,
            'start_date': multi_config.get('start_date', '2000-01-01'),
            'end_date': multi_config.get('end_date', '2020-12-31'),
            'cali_end_date': multi_config.get('cali_end_date', '2009-12-31'),
            'model_type': model_type,
            'coupled': 'coupled' in config_dir.lower(),
        }
        
        try:
            # Load hydrograph data for this configuration
            data = postprocessing_single.load_hydrograph_data(individual_config)
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
        
        # Plot simulated data for this configuration
        if 'sim_Q' in monthly_df.columns:
            ax.plot(monthly_df.index, monthly_df['sim_Q'], 
                   color=color, linewidth=3, label='Simulated', zorder=5)
        
        # Formatting for this subplot
        ax.set_title(f'{name}', fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
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
            ax.set_ylabel('Discharge (m³/s)', fontsize=15, fontweight='bold')
    
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
    save_path = plot_dir / f'hydrological_regime_subplots_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved hydrological regime subplots to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\nHydrological Regime Subplots Summary:")
    print(f"  Configurations processed: {len(config_results)}")
    print(f"  Layout: {n_rows} rows × {n_cols} columns")
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
        
        # Create individual config dictionary for this configuration
        individual_config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': config_dir,
            'gauge_id': gauge_id,
            'start_date': multi_config.get('start_date', '2000-01-01'),
            'end_date': multi_config.get('end_date', '2020-12-31'),
            'cali_end_date': multi_config.get('cali_end_date', '2009-12-31'),
            'model_type': model_type,
            'coupled': 'coupled' in config_dir.lower(),
        }
        
        try:
            # Load hydrograph data for this configuration
            data = postprocessing_single.load_hydrograph_data(individual_config)
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
        
        # Create individual config dictionary for this configuration
        individual_config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': config_dir,
            'gauge_id': gauge_id,
            'start_date': multi_config.get('start_date', '2000-01-01'),
            'end_date': multi_config.get('end_date', '2020-12-31'),
            'cali_end_date': multi_config.get('cali_end_date', '2009-12-31'),
            'model_type': model_type,
            'coupled': 'coupled' in config_dir.lower(),
        }
        
        try:
            # Load SWE data for this configuration (obs_data can be None now)
            sim_data, obs_data, area_data = postprocessing_single.load_swe_data(individual_config)
            
            if sim_data is None:
                print(f"  ❌ Warning: Failed to load simulated SWE data for {config_dir}")
                continue
            
            # Check if we have observed data
            has_obs_this_config = (obs_data is not None)
            if not has_obs_this_config:
                print(f"  ℹ️  No observed SWE data found for {config_dir} - using simulated data only")
            
            # Process the data
            processed = postprocessing_single.process_swe_data(sim_data, obs_data, area_data)
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
            val_sim['area_weighted_swe'] = postprocessing_single.calculate_area_weighted_swe(val_sim, area_mapping)
            
            # Process observed data if available
            val_obs = None
            if has_obs_this_config and obs_data_proc is not None:
                obs_data_proc['time'] = pd.to_datetime(obs_data_proc['time'])
                val_obs_mask = (obs_data_proc['time'] >= validation_start_dt) & (obs_data_proc['time'] <= validation_end_dt)
                val_obs = obs_data_proc[val_obs_mask].copy()
                
                if len(val_obs) > 0:
                    val_obs['area_weighted_swe'] = postprocessing_single.calculate_area_weighted_swe(val_obs, area_mapping)
                    # Store observed data from first successful config
                    if obs_swe_data is None:
                        obs_swe_data = val_obs.copy()
                        has_observed_data = True
            
            # Calculate metrics if both sim and obs are available
            metrics = {'overall_rmse': None, 'overall_bias': None, 'overall_corr': None,
                      'area_weighted_rmse': None, 'area_weighted_bias': None, 'area_weighted_corr': None}
            
            if has_obs_this_config and val_obs is not None and len(val_obs) > 0:
                try:
                    metrics = postprocessing_single.calculate_swe_metrics(
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
        
        # Create individual config dictionary for this configuration
        individual_config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': config_dir,
            'gauge_id': gauge_id,
            'start_date': multi_config.get('start_date', '2000-01-01'),
            'end_date': multi_config.get('end_date', '2020-12-31'),
            'cali_end_date': multi_config.get('cali_end_date', '2009-12-31'),
            'model_type': model_type,
            'coupled': 'coupled' in config_dir.lower(),
        }
        
        try:
            # Load and process SWE data for this configuration
            sim_data, obs_data, area_data = postprocessing_single.load_swe_data(individual_config)
            if sim_data is None or obs_data is None:
                print(f"  Warning: Failed to load SWE data for {config_dir}")
                continue
            
            processed = postprocessing_single.process_swe_data(sim_data, obs_data, area_data)
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
        
        # Create individual config dictionary for this configuration
        individual_config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': config_dir,
            'gauge_id': gauge_id,
            'start_date': multi_config.get('start_date', '2000-01-01'),
            'end_date': multi_config.get('end_date', '2020-12-31'),
            'cali_end_date': multi_config.get('cali_end_date', '2009-12-31'),
            'model_type': model_type,
            'coupled': 'coupled' in config_dir.lower(),
        }
        
        try:
            # Load hydrograph data for this configuration
            data = postprocessing_single.load_hydrograph_data(individual_config)
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
            
            val_metrics = postprocessing_single.calculate_performance_metrics(
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
        
        # Create individual config dictionary for this configuration
        individual_config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': config_dir,
            'gauge_id': gauge_id,
            'start_date': multi_config.get('start_date', '2000-01-01'),
            'end_date': multi_config.get('end_date', '2020-12-31'),
            'cali_end_date': multi_config.get('cali_end_date', '2009-12-31'),
            'model_type': model_type,
            'coupled': 'coupled' in config_dir.lower(),
        }
        
        try:
            # Load parameter data for this configuration
            param_data = postprocessing_single.load_parameter_values(individual_config, top_n)
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
        ax.set_title(f'{display_name}', fontsize=11, fontweight='bold')
        
        # Format axes
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax.set_ylabel('Parameter Value', fontsize=10)
        
        # Rotate x-axis labels if needed
        if len(labels) > 2:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add statistics for each configuration
        stats_text = ""
        for j, (config_dir, result) in enumerate(config_results.items()):
            param_data = result['param_data']
            name = result['name']
            
            if param_name in param_data['parameters'] and len(param_data['parameters'][param_name]) > 0:
                values = param_data['parameters'][param_name]
                mean_val = np.mean(values)
                std_val = np.std(values)
                stats_text += f"{name}: μ={mean_val:.3f}, σ={std_val:.3f}\n"
        
        if stats_text:
            ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for i in range(n_params, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'Parameter Distribution Comparison - Catchment {gauge_id}\n'
                f'Top {top_n} Parameter Sets per Configuration', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
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
        
        # Create individual config dictionary for this configuration
        individual_config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': config_dir,
            'gauge_id': gauge_id,
            'start_date': multi_config.get('start_date', '2000-01-01'),
            'end_date': multi_config.get('end_date', '2020-12-31'),
            'cali_end_date': multi_config.get('cali_end_date', '2009-12-31'),
            'model_type': model_type,
            'coupled': 'coupled' in config_dir.lower(),
        }
        
        try:
            # Load storage data for this configuration
            storage_df = postprocessing_single.load_storage_data(individual_config)
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

def create_combined_contributions_for_all_configs(multi_config, validation_start=None, validation_end=None):
    """
    Create combined contributions dataframes for all configurations.
    
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
        Dictionary containing contribution data for each configuration
    """
    
    # Use dates from multi_config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    gauge_id = multi_config['gauge_id']
    configs = multi_config['configs']
    config_names = multi_config['config_names']
    model_type = multi_config.get('model_type', 'HBV')
    
    print(f"Creating contributions dataframes for {len(configs)} configurations:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Period: {validation_start} to {validation_end}")
    
    # Store results for each configuration
    config_contributions = {}
    
    # Process each configuration
    for config_dir in configs:
        print(f"\nProcessing configuration: {config_dir}")
        
        # Create individual config dictionary for this configuration
        individual_config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': config_dir,
            'gauge_id': gauge_id,
            'start_date': multi_config.get('start_date', '2000-01-01'),
            'end_date': multi_config.get('end_date', '2020-12-31'),
            'cali_end_date': multi_config.get('cali_end_date', '2009-12-31'),
            'model_type': model_type,
            'glogem_dir': multi_config.get('glogem_dir'),  # <-- ADD THIS LINE
            'coupled': multi_config.get('config_coupled', {}).get(config_dir, False),
        }
        
        try:
            # Create plot_dirs for the individual config (needed by postprocessing_single function)
            temp_plot_dirs = postprocessing_single.setup_output_directories(individual_config)
            
            # Create combined contributions dataframes using postprocessing_single function
            glacier_df, nonglacier_df = postprocessing_single.create_combined_contributions_dataframes(
                individual_config, temp_plot_dirs, validation_start, validation_end
            )
            
            if glacier_df is None or nonglacier_df is None:
                print(f"  Warning: Could not create contribution dataframes for {config_dir}")
                continue
            
            # Combine glacier and non-glacier contributions
            # Add month column for regime analysis
            glacier_df['month'] = glacier_df['date'].dt.month
            nonglacier_df['month'] = nonglacier_df['date'].dt.month
            
            # Calculate total contributions
            combined_df = pd.DataFrame({
                'date': glacier_df['date'],
                'month': glacier_df['month'],
                'glacier_melt': glacier_df['glaciermelt'],
                'snowmelt_glacier': glacier_df['snowmelt'],
                'snowmelt_nonglacier': nonglacier_df['snowmelt'],
                'total_snowmelt': glacier_df['snowmelt'] + nonglacier_df['snowmelt']
            })
            
            # Load streamflow data for this configuration
            streamflow_data = postprocessing_single.load_hydrograph_data(individual_config)
            if streamflow_data is not None:
                # Filter streamflow for validation period
                validation_start_dt = pd.to_datetime(validation_start)
                validation_end_dt = pd.to_datetime(validation_end)
                
                streamflow_mask = (streamflow_data['date'] >= validation_start_dt) & (streamflow_data['date'] <= validation_end_dt)
                streamflow_filtered = streamflow_data[streamflow_mask].copy()
                
                if len(streamflow_filtered) > 0:
                    streamflow_filtered['month'] = streamflow_filtered['date'].dt.month
                    
                    # Calculate monthly streamflow means
                    streamflow_monthly = streamflow_filtered.groupby('month').agg({
                        'obs_Q': 'mean',
                        'sim_Q': 'mean'
                    }).reset_index()
                    
                    combined_df['obs_Q'] = streamflow_monthly['obs_Q'].values if 'obs_Q' in streamflow_monthly.columns else None
                    combined_df['sim_Q'] = streamflow_monthly['sim_Q'].values if 'sim_Q' in streamflow_monthly.columns else None
            
            # Calculate monthly means for contributions
            monthly_contributions = combined_df.groupby('month').agg({
                'glacier_melt': 'mean',
                'total_snowmelt': 'mean'
            }).reset_index()
            
            # Get catchment area for unit conversion (mm/day to m³/s)
            conversion_factor = 1.0
            try:
                config_dir_path = Path(individual_config['main_dir']) / individual_config['config_dir']
                topo_dir = config_dir_path / f"catchment_{gauge_id}" / "topo_files"
                catchment_shape_file = topo_dir / "HRU.shp"
                
                if catchment_shape_file.exists():
                    import geopandas as gpd
                    hru_gdf = gpd.read_file(catchment_shape_file)
                    total_area_km2 = hru_gdf['Area_km2'].sum()
                    # Convert mm/day to m³/s
                    conversion_factor = total_area_km2 * 1000 * 1000 / 86400 / 1000
                    print(f"  ✓ Catchment area: {total_area_km2:.2f} km², conversion factor: {conversion_factor:.3f}")
            except Exception as e:
                print(f"  Warning: Could not load catchment area: {e}")
            
            # Store results for this configuration
            config_contributions[config_dir] = {
                'combined_data': combined_df,
                'monthly_contributions': monthly_contributions,
                'monthly_streamflow': streamflow_monthly if 'streamflow_monthly' in locals() else None,
                'conversion_factor': conversion_factor,
                'name': config_names.get(config_dir, config_dir),
                'coupled': individual_config['coupled']
            }
            
            print(f"  ✓ Successfully processed contributions for {config_dir}")
            print(f"    Mean glacier melt: {monthly_contributions['glacier_melt'].mean():.3f} mm/day")
            print(f"    Mean total snowmelt: {monthly_contributions['total_snowmelt'].mean():.3f} mm/day")
            
        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue
    
    if len(config_contributions) == 0:
        print("No configurations processed successfully")
        return None
    
    print(f"\nSuccessfully processed {len(config_contributions)} configurations")
    return config_contributions

#--------------------------------------------------------------------------------

def plot_streamflow_contributions_comparison(multi_config, validation_start=None, validation_end=None):
    """
    Plot streamflow contributions comparison across multiple configurations.
    Each configuration gets its own subplot showing observed/simulated streamflow
    and contributions as filled polygons.
    
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
        Dictionary containing contribution data for each configuration
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
    
    print(f"Creating streamflow contributions comparison for {len(configs)} configurations:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Validation period: {validation_start} to {validation_end}")
    
    # Create plot directory
    plot_dir = create_multi_plot_dir(multi_config)
    
    # Store results for each configuration
    config_contributions = {}
    
    # Process each configuration
    for config_dir in configs:
        print(f"\nProcessing configuration: {config_dir}")
        
        # Create individual config dictionary for this configuration
        individual_config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': config_dir,
            'gauge_id': gauge_id,
            'start_date': multi_config.get('start_date', '2000-01-01'),
            'end_date': multi_config.get('end_date', '2020-12-31'),
            'cali_end_date': multi_config.get('cali_end_date', '2009-12-31'),
            'model_type': multi_config.get('model_type', 'HBV'),
            'glogem_dir': multi_config.get('glogem_dir'),
            'coupled': multi_config.get('config_coupled', {}).get(config_dir, False),
        }
        
        try:
            # Create plot_dirs for the individual config (needed by postprocessing_single function)
            temp_plot_dirs = postprocessing_single.setup_output_directories(individual_config)
            
            # Create combined contributions dataframes using postprocessing_single function
            glacier_df, nonglacier_df = postprocessing_single.create_combined_contributions_dataframes(
                individual_config, temp_plot_dirs, validation_start, validation_end
            )
            
            if glacier_df is None or nonglacier_df is None:
                print(f"  Warning: Could not create contribution dataframes for {config_dir}")
                continue
            
            # Combine glacier and non-glacier contributions
            # Add month column for regime analysis
            glacier_df['month'] = glacier_df['date'].dt.month
            nonglacier_df['month'] = nonglacier_df['date'].dt.month
            
            # Calculate monthly means for contributions (this is the key fix!)
            monthly_contributions = pd.DataFrame()
            monthly_contributions['month'] = range(1, 13)
            
            # Calculate monthly means from daily data
            glacier_monthly = glacier_df.groupby('month').agg({
                'glaciermelt': 'mean',
                'snowmelt': 'mean'
            }).reset_index()
            
            nonglacier_monthly = nonglacier_df.groupby('month').agg({
                'snowmelt': 'mean'
            }).reset_index()
            
            # Merge monthly contributions
            monthly_contributions = monthly_contributions.merge(glacier_monthly, on='month', how='left')
            monthly_contributions = monthly_contributions.merge(
                nonglacier_monthly, on='month', how='left', suffixes=('_glacier', '_nonglacier')
            )
            
            # Calculate total snowmelt
            monthly_contributions['glacier_melt'] = monthly_contributions['glaciermelt']
            monthly_contributions['total_snowmelt'] = (
                monthly_contributions['snowmelt_glacier'] + monthly_contributions['snowmelt_nonglacier']
            )
            
            # Load streamflow data for this configuration
            streamflow_data = postprocessing_single.load_hydrograph_data(individual_config)
            monthly_streamflow = None
            
            if streamflow_data is not None:
                # Filter streamflow for validation period
                validation_start_dt = pd.to_datetime(validation_start)
                validation_end_dt = pd.to_datetime(validation_end)
                
                streamflow_mask = (streamflow_data['date'] >= validation_start_dt) & (streamflow_data['date'] <= validation_end_dt)
                streamflow_filtered = streamflow_data[streamflow_mask].copy()
                
                if len(streamflow_filtered) > 0:
                    streamflow_filtered['month'] = streamflow_filtered['date'].dt.month
                    
                    # Calculate monthly streamflow means
                    monthly_streamflow = streamflow_filtered.groupby('month').agg({
                        'obs_Q': 'mean',
                        'sim_Q': 'mean'
                    }).reset_index()
            
            # Get catchment area for unit conversion (mm/day to m³/s)
            conversion_factor = 1.0
            try:
                config_dir_path = Path(individual_config['main_dir']) / individual_config['config_dir']
                topo_dir = config_dir_path / f"catchment_{gauge_id}" / "topo_files"
                catchment_shape_file = topo_dir / "HRU.shp"
                
                if catchment_shape_file.exists():
                    import geopandas as gpd
                    hru_gdf = gpd.read_file(catchment_shape_file)
                    total_area_km2 = hru_gdf['Area_km2'].sum()
                    # Convert mm/day to m³/s
                    conversion_factor = total_area_km2 * 1000 * 1000 / 86400 / 1000
                    print(f"  ✓ Catchment area: {total_area_km2:.2f} km², conversion factor: {conversion_factor:.3f}")
            except Exception as e:
                print(f"  Warning: Could not load catchment area: {e}")
            
            # Store results for this configuration
            config_contributions[config_dir] = {
                'monthly_contributions': monthly_contributions,
                'monthly_streamflow': monthly_streamflow,
                'conversion_factor': conversion_factor,
                'name': config_names.get(config_dir, config_dir),
                'coupled': individual_config['coupled']
            }
            
            print(f"  ✓ Successfully processed contributions for {config_dir}")
            print(f"    Mean glacier melt: {monthly_contributions['glacier_melt'].mean():.3f} mm/day")
            print(f"    Mean total snowmelt: {monthly_contributions['total_snowmelt'].mean():.3f} mm/day")
            
        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue
    
    if len(config_contributions) == 0:
        print("No configurations processed successfully")
        return None
    
    # Calculate subplot layout
    n_configs = len(config_contributions)
    if n_configs <= 2:
        n_rows, n_cols = 1, n_configs
        figsize = (8 * n_configs, 6)
    elif n_configs <= 4:
        n_rows, n_cols = 2, 2
        figsize = (16, 10)
    elif n_configs <= 6:
        n_rows, n_cols = 2, 3
        figsize = (20, 10)
    else:
        # For more configurations, use more rows
        n_cols = 3
        n_rows = (n_configs + n_cols - 1) // n_cols
        figsize = (20, 6 * n_rows)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    
    # Handle single subplot case
    if n_configs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_configs > 1 else [axes]
    else:
        axes = axes.flatten()
    
    months = range(1, 13)
    month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    
    # Plot each configuration in its own subplot
    for i, (config_dir, contrib_data) in enumerate(config_contributions.items()):
        ax = axes[i]
        
        monthly_contributions = contrib_data['monthly_contributions']
        monthly_streamflow = contrib_data['monthly_streamflow']
        conversion_factor = contrib_data['conversion_factor']
        name = contrib_data['name']
        coupled = contrib_data['coupled']
        
        # Convert contributions from mm/day to m³/s
        glacier_melt_converted = monthly_contributions['glacier_melt'] * conversion_factor
        total_snowmelt_converted = monthly_contributions['total_snowmelt'] * conversion_factor
        
        # Plot contributions as filled areas (polygons)
        # Plot total snowmelt first (bottom layer)
        ax.fill_between(monthly_contributions['month'], 0, total_snowmelt_converted, 
                       color='lightblue', alpha=0.7, label='Total Snowmelt', zorder=1)
        
        # Plot glacier melt on top
        ax.fill_between(monthly_contributions['month'], 0, glacier_melt_converted, 
                       color='grey', alpha=0.8, label='Glacier Melt', zorder=2)
        
        # Plot streamflow if available
        if monthly_streamflow is not None:
            # Plot observed streamflow (black line)
            if 'obs_Q' in monthly_streamflow.columns:
                ax.plot(monthly_streamflow['month'], monthly_streamflow['obs_Q'], 
                       'k-', linewidth=3, label='Observed', zorder=4)
            
            # Plot simulated streamflow (colored line)
            if 'sim_Q' in monthly_streamflow.columns:
                color = config_colors.get(config_dir, 'C0')
                ax.plot(monthly_streamflow['month'], monthly_streamflow['sim_Q'], 
                       color=color, linewidth=2.5, label='Simulated', zorder=3)
        
        # Formatting for this subplot
        ax.set_title(f'{name}\n({"GloGEM+HBV" if coupled else "HBV"})', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(months)
        ax.set_xticklabels(month_names)
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        
        # Add legend only to the first subplot
        if i == 0:
            ax.legend(loc='best', fontsize=10)
        
        # Set y-axis label for leftmost column
        if i % n_cols == 0:
            ax.set_ylabel('Discharge (m³/s)', fontsize=11)
        
        # Set x-axis label for bottom row
        if i >= (n_rows - 1) * n_cols or i >= n_configs - n_cols:
            ax.set_xlabel('Month', fontsize=11)
        
        # Add contribution statistics as text
        mean_glacier = glacier_melt_converted.mean()
        mean_snowmelt = total_snowmelt_converted.mean()
        
        if monthly_streamflow is not None and 'sim_Q' in monthly_streamflow.columns:
            mean_sim = monthly_streamflow['sim_Q'].mean()
            if mean_sim > 0:
                glacier_pct = (mean_glacier / mean_sim) * 100
                snowmelt_pct = (mean_snowmelt / mean_sim) * 100
                stats_text = f"Glacier: {glacier_pct:.1f}%\nSnowmelt: {snowmelt_pct:.1f}%"
            else:
                stats_text = f"Glacier: {mean_glacier:.1f} m³/s\nSnowmelt: {mean_snowmelt:.1f} m³/s"
        else:
            stats_text = f"Glacier: {mean_glacier:.1f} m³/s\nSnowmelt: {mean_snowmelt:.1f} m³/s"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'Streamflow Contributions Comparison - Catchment {gauge_id}\n'
                f'Validation Period: {validation_start} to {validation_end}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    save_path = plot_dir / f'streamflow_contributions_comparison_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved streamflow contributions comparison plot to: {save_path}")
    plt.show()
    
    # Print comprehensive summary
    print(f"\nStreamflow Contributions Comparison Summary:")
    print(f"  Configurations processed: {len(config_contributions)}")
    print(f"  Validation period: {validation_start} to {validation_end}")
    
    print(f"\nContribution Analysis by Configuration:")
    for config_dir, contrib_data in config_contributions.items():
        name = contrib_data['name']
        monthly_contributions = contrib_data['monthly_contributions']
        monthly_streamflow = contrib_data['monthly_streamflow']
        conversion_factor = contrib_data['conversion_factor']
        coupled = contrib_data['coupled']
        
        mean_glacier = monthly_contributions['glacier_melt'].mean() * conversion_factor
        mean_snowmelt = monthly_contributions['total_snowmelt'].mean() * conversion_factor
        
        print(f"\n  {name} ({'GloGEM+HBV' if coupled else 'HBV'}):")
        print(f"    Mean glacier melt: {mean_glacier:.2f} m³/s")
        print(f"    Mean total snowmelt: {mean_snowmelt:.2f} m³/s")
        
        if monthly_streamflow is not None and 'sim_Q' in monthly_streamflow.columns:
            mean_sim = monthly_streamflow['sim_Q'].mean()
            if mean_sim > 0:
                glacier_pct = (mean_glacier / mean_sim) * 100
                snowmelt_pct = (mean_snowmelt / mean_sim) * 100
                print(f"    Mean simulated flow: {mean_sim:.2f} m³/s")
                print(f"    Glacier contribution: {glacier_pct:.1f}% of simulated flow")
                print(f"    Snowmelt contribution: {snowmelt_pct:.1f}% of simulated flow")
    
    return config_contributions

#--------------------------------------------------------------------------------

def plot_streamflow_glogem_snowmelt_regime_comparison(multi_config, validation_start=None, validation_end=None, unit='mm'):
    """
    Plot streamflow regime with GloGEM ice melt and total snowmelt for multiple configurations.
    
    Shows stacked contributions in separate subplots for each configuration:
    - Total Snowmelt (GloGEM snowmelt + HBV snowmelt) - filled area
    - GloGEM ice melt - filled area
    - Observed streamflow - solid line
    - Simulated streamflow - dashed line
    
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
    
    print(f"Creating streamflow regime with GloGEM and snowmelt comparison for {len(configs)} configurations:")
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
        config_path = Path(multi_config['main_dir']) / config_dir
        catchment_dir = config_path / f"catchment_{gauge_id}" / model_type
        
        print(f"\n{'='*60}")
        print(f"Processing: {config_names.get(config_dir, config_dir)}")
        print(f"{'='*60}")
        
        # Create temporary config for single-config functions
        temp_config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': config_dir,
            'gauge_id': gauge_id,
            'model_type': model_type,
            'start_date': multi_config.get('start_date', '2000-01-01'),
            'end_date': multi_config.get('end_date', '2020-12-31'),
            'cali_end_date': multi_config.get('cali_end_date', '2010-12-31')
        }
        
        try:
            # Load catchment area for unit conversion
            conversion_m3s_to_mm_day = None
            if unit == 'mm':
                topo_dir = config_path / f"catchment_{gauge_id}" / "topo_files"
                catchment_shape_file = topo_dir / "HRU.shp"
                
                if catchment_shape_file.exists():
                    import geopandas as gpd
                    hru_gdf = gpd.read_file(catchment_shape_file)
                    total_area_km2 = hru_gdf['Area_km2'].sum()
                    conversion_m3s_to_mm_day = 86400 / (total_area_km2 * 1000000) * 1000
                    print(f"  Catchment area: {total_area_km2:.2f} km²")
            
            # 1. LOAD STREAMFLOW DATA
            from postprocessing_single import load_hydrograph_data
            streamflow_data = load_hydrograph_data(temp_config)
            if streamflow_data is None:
                print(f"  ⚠️  Skipping {config_dir}: Could not load streamflow data")
                continue
            
            # Filter for validation period
            start_date = pd.to_datetime(validation_start)
            end_date = pd.to_datetime(validation_end)
            
            streamflow_mask = (streamflow_data['date'] >= start_date) & (streamflow_data['date'] <= end_date)
            streamflow_filtered = streamflow_data[streamflow_mask].copy()
            
            if len(streamflow_filtered) == 0:
                print(f"  ⚠️  Skipping {config_dir}: No streamflow data in validation period")
                continue
            
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
                # Store observed data for plotting (same for all configs)
                if obs_data is None:
                    obs_data = streamflow_regime['observed']
            
            if 'sim_Q_converted' in streamflow_filtered.columns:
                streamflow_regime['simulated'] = streamflow_filtered.groupby('month')['sim_Q_converted'].mean()
            
            # 2. LOAD GLOGEM DATA
            from postprocessing_single import load_glogem_data
            glogem_df = load_glogem_data(temp_config, unit='mm', plot=False)
            
            if glogem_df is None:
                print(f"  ⚠️  Skipping {config_dir}: Could not load GloGEM data")
                continue
            
            # Filter GloGEM data for validation period
            glogem_mask = (glogem_df['date'] >= start_date) & (glogem_df['date'] <= end_date)
            glogem_filtered = glogem_df[glogem_mask].copy()
            
            if len(glogem_filtered) == 0:
                print(f"  ⚠️  Skipping {config_dir}: No GloGEM data in validation period")
                continue
            
            # Calculate monthly regime for GloGEM components
            glogem_filtered['month'] = glogem_filtered['date'].dt.month
            
            # Use NORMALIZED (catchment area) values for fair comparison with streamflow
            glogem_icemelt_regime = glogem_filtered.groupby('month')['icemelt_normalized'].mean()
            glogem_snowmelt_regime = glogem_filtered.groupby('month')['snowmelt_normalized'].mean()
            
            # 3. LOAD HBV SNOWMELT MASS LOADINGS
            from postprocessing_single import load_snowmelt_mass_loadings
            hbv_snowmelt_df = load_snowmelt_mass_loadings(temp_config, validation_start, validation_end, unit=unit)
            
            if hbv_snowmelt_df is None:
                print(f"  ⚠️  Skipping {config_dir}: Could not load HBV snowmelt data")
                continue
            
            # Determine snowmelt column based on unit
            snowmelt_col = 'snowmelt_mm_day' if unit == 'mm' and 'snowmelt_mm_day' in hbv_snowmelt_df.columns else 'snowmelt_m3s'
            
            # Calculate monthly regime for HBV snowmelt
            hbv_snowmelt_df['month'] = hbv_snowmelt_df['date'].dt.month
            hbv_snowmelt_regime = hbv_snowmelt_df.groupby('month')[snowmelt_col].mean()
            
            # 4. COMBINE SNOWMELT SOURCES
            total_snowmelt_regime = glogem_snowmelt_regime.add(hbv_snowmelt_regime, fill_value=0)
            
            # Store results
            config_results[config_dir] = {
                'streamflow': streamflow_regime,
                'glogem_icemelt': glogem_icemelt_regime,
                'glogem_snowmelt': glogem_snowmelt_regime,
                'hbv_snowmelt': hbv_snowmelt_regime,
                'total_snowmelt': total_snowmelt_regime,
                'color': config_colors.get(config_dir, '#1f77b4'),
                'name': config_names.get(config_dir, config_dir)
            }
            
            print(f"  ✓ Successfully processed {config_names.get(config_dir, config_dir)}")
            print(f"    GloGEM ice melt mean: {glogem_icemelt_regime.mean():.4f} {unit_label}")
            print(f"    Total snowmelt mean: {total_snowmelt_regime.mean():.4f} {unit_label}")
            
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
        figsize = (14*n_configs, 8)
    elif n_configs <= 4:
        n_rows, n_cols = 2, 2
        figsize = (24, 16)
    elif n_configs <= 6:
        n_rows, n_cols = 2, 3
        figsize = (30, 16)
    else:
        n_cols = 3
        n_rows = (n_configs + n_cols - 1) // n_cols
        figsize = (30, 8*n_rows)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    
    # Handle single subplot case
    if n_configs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot each configuration in its own subplot
    for i, (config_dir, result) in enumerate(config_results.items()):
        ax = axes[i]
        
        streamflow_regime = result['streamflow']
        glogem_icemelt = result['glogem_icemelt']
        total_snowmelt = result['total_snowmelt']
        name = result['name']
        
        # Plot filled polygons FIRST (bottom layer, in order of magnitude)
        # Total snowmelt as filled polygon - light blue
        ax.fill_between(months, 0, total_snowmelt.values, 
                        color='#B3D9FF', alpha=0.7, label='Total Snowmelt (GloGEM+HBV)', 
                        zorder=1, edgecolor='#6DB3F2', linewidth=1.5)
        
        # GloGEM ice melt as filled polygon - brown/orange
        ax.fill_between(months, 0, glogem_icemelt.values, 
                        color='#C17817', alpha=0.6, label='GloGEM Ice Melt', 
                        zorder=2, edgecolor='#8B5A00', linewidth=1.5)
        
        # Plot observed streamflow (thick solid line)
        if 'observed' in streamflow_regime:
            ax.plot(months, streamflow_regime['observed'].values, 'k-', 
                   linewidth=3.5, label='Observed', zorder=4)
        
        # Plot simulated streamflow (dashed line)
        if 'simulated' in streamflow_regime:
            ax.plot(months, streamflow_regime['simulated'].values, 'C0--', 
                   linewidth=3, label='Simulated', zorder=3)
        
        # Formatting for this subplot
        ax.set_title(f'{name}', fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        ax.legend(loc='best', fontsize=13)
        
        # Set x-axis labels for bottom row
        if i >= (n_rows - 1) * n_cols or i >= n_configs - n_cols:
            ax.set_xticks(months)
            ax.set_xticklabels(month_names, fontsize=13)
        
        # Set y-axis label for leftmost column
        if i % n_cols == 0:
            ax.set_ylabel(f'Discharge ({unit_label})', fontsize=15, fontweight='bold')
        
        # Add statistics text box
        stats_lines = []
        if 'simulated' in streamflow_regime:
            sim_mean = streamflow_regime['simulated'].mean()
            icemelt_mean = glogem_icemelt.mean()
            snowmelt_mean = total_snowmelt.mean()
            
            if sim_mean > 0:
                icemelt_pct = (icemelt_mean / sim_mean) * 100
                snowmelt_pct = (snowmelt_mean / sim_mean) * 100
                
                stats_lines.append(f"Sim: {sim_mean:.2f} {unit_label}")
                stats_lines.append(f"Ice: {icemelt_pct:.1f}%")
                stats_lines.append(f"Snow: {snowmelt_pct:.1f}%")
        
        if stats_lines:
            stats_text = '\n'.join(stats_lines)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # Hide unused subplots
    for i in range(n_configs, len(axes)):
        axes[i].axis('off')
    
    # Add common x-label
    fig.text(0.5, 0.02, 'Month', ha='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.08)
    
    # Save plot
    save_path = plot_dir / f'streamflow_glogem_snowmelt_regime_comparison_{unit}_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved streamflow regime comparison plot to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"STREAMFLOW REGIME WITH GLOGEM AND SNOWMELT COMPARISON")
    print(f"{'='*60}")
    print(f"Catchment: {gauge_id}")
    print(f"Period: {validation_start} to {validation_end}")
    print(f"Unit: {unit_label}")
    print(f"Configurations processed: {len(config_results)}")
    
    for config_dir, result in config_results.items():
        name = result['name']
        streamflow = result['streamflow']
        icemelt = result['glogem_icemelt']
        snowmelt = result['total_snowmelt']
        
        print(f"\n{name}:")
        if 'simulated' in streamflow:
            sim_mean = streamflow['simulated'].mean()
            icemelt_mean = icemelt.mean()
            snowmelt_mean = snowmelt.mean()
            
            print(f"  Simulated flow: {sim_mean:.4f} {unit_label}")
            print(f"  GloGEM ice melt: {icemelt_mean:.4f} {unit_label} ({(icemelt_mean/sim_mean*100):.1f}%)")
            print(f"  Total snowmelt: {snowmelt_mean:.4f} {unit_label} ({(snowmelt_mean/sim_mean*100):.1f}%)")
            print(f"  Combined melt: {(icemelt_mean+snowmelt_mean)/sim_mean*100:.1f}%")
    
    print(f"{'='*60}\n")
    
    return config_results

#--------------------------------------------------------------------------------

def plot_streamflow_all_melt_regime_comparison(multi_config, validation_start=None, validation_end=None, unit='mm'):
    """
    Plot streamflow regime with total glacier melt and snowmelt for multiple configurations.
    
    Shows stacked contributions in separate subplots for each configuration:
    - Snowmelt (from mass loadings) - filled area
    - Total Glacier Melt (ALL) - filled area
    - Observed streamflow - solid line
    - Simulated streamflow - dashed line
    
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
    
    print(f"Creating streamflow regime with all melt contributions comparison for {len(configs)} configurations:")
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
        config_path = Path(multi_config['main_dir']) / config_dir
        catchment_dir = config_path / f"catchment_{gauge_id}" / model_type
        
        print(f"\n{'='*60}")
        print(f"Processing: {config_names.get(config_dir, config_dir)}")
        print(f"{'='*60}")
        
        # Create temporary config for single-config functions
        temp_config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': config_dir,
            'gauge_id': gauge_id,
            'model_type': model_type,
            'start_date': multi_config.get('start_date', '2000-01-01'),
            'end_date': multi_config.get('end_date', '2020-12-31'),
            'cali_end_date': multi_config.get('cali_end_date', '2010-12-31')
        }
        
        try:
            # Load catchment area for unit conversion
            conversion_m3s_to_mm_day = None
            if unit == 'mm':
                topo_dir = config_path / f"catchment_{gauge_id}" / "topo_files"
                catchment_shape_file = topo_dir / "HRU.shp"
                
                if catchment_shape_file.exists():
                    import geopandas as gpd
                    hru_gdf = gpd.read_file(catchment_shape_file)
                    total_area_km2 = hru_gdf['Area_km2'].sum()
                    conversion_m3s_to_mm_day = 86400 / (total_area_km2 * 1000000) * 1000
                    print(f"  Catchment area: {total_area_km2:.2f} km²")
            
            # 1. LOAD STREAMFLOW DATA
            from postprocessing_single import load_hydrograph_data
            streamflow_data = load_hydrograph_data(temp_config)
            if streamflow_data is None:
                print(f"  ⚠️  Skipping {config_dir}: Could not load streamflow data")
                continue
            
            # Filter for validation period
            start_date = pd.to_datetime(validation_start)
            end_date = pd.to_datetime(validation_end)
            
            streamflow_mask = (streamflow_data['date'] >= start_date) & (streamflow_data['date'] <= end_date)
            streamflow_filtered = streamflow_data[streamflow_mask].copy()
            
            if len(streamflow_filtered) == 0:
                print(f"  ⚠️  Skipping {config_dir}: No streamflow data in validation period")
                continue
            
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
                # Store observed data for plotting (same for all configs)
                if obs_data is None:
                    obs_data = streamflow_regime['observed']
            
            if 'sim_Q_converted' in streamflow_filtered.columns:
                streamflow_regime['simulated'] = streamflow_filtered.groupby('month')['sim_Q_converted'].mean()
            
            # 2. LOAD SNOWMELT DATA
            from postprocessing_single import load_snowmelt_mass_loadings
            snowmelt_df = load_snowmelt_mass_loadings(temp_config, validation_start, validation_end, unit=unit)
            
            if snowmelt_df is None:
                print(f"  ⚠️  Skipping {config_dir}: Could not load snowmelt data")
                continue
            
            # Determine snowmelt column based on unit
            snowmelt_col = 'snowmelt_mm_day' if unit == 'mm' and 'snowmelt_mm_day' in snowmelt_df.columns else 'snowmelt_m3s'
            
            # Calculate monthly regime for snowmelt
            snowmelt_df['month'] = snowmelt_df['date'].dt.month
            snowmelt_regime = snowmelt_df.groupby('month')[snowmelt_col].mean()
            
            # 3. LOAD GLACIER MELT DATA (ALL)
            from postprocessing_single import load_glacier_melt_mass_loadings
            glacier_data = load_glacier_melt_mass_loadings(temp_config, validation_start, validation_end, unit=unit)
            
            if glacier_data is None or glacier_data.get('all') is None:
                print(f"  ⚠️  Skipping {config_dir}: Could not load glacier melt data")
                continue
            
            # Determine glacier melt column based on unit
            glacier_melt_col = 'glacier_melt_mm_day' if unit == 'mm' and 'glacier_melt_mm_day' in glacier_data['all'].columns else 'glacier_melt_m3s'
            
            # Calculate monthly regime for ALL glacier melt
            glacier_all_df = glacier_data['all']
            glacier_all_df['month'] = glacier_all_df['date'].dt.month
            glacier_all_regime = glacier_all_df.groupby('month')[glacier_melt_col].mean()
            
            # Store results
            config_results[config_dir] = {
                'streamflow': streamflow_regime,
                'snowmelt': snowmelt_regime,
                'glacier_all': glacier_all_regime,
                'color': config_colors.get(config_dir, '#1f77b4'),
                'name': config_names.get(config_dir, config_dir)
            }
            
            print(f"  ✓ Successfully processed {config_names.get(config_dir, config_dir)}")
            print(f"    Snowmelt mean: {snowmelt_regime.mean():.4f} {unit_label}")
            print(f"    Glacier melt mean: {glacier_all_regime.mean():.4f} {unit_label}")
            
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
        figsize = (14*n_configs, 8)
    elif n_configs <= 4:
        n_rows, n_cols = 2, 2
        figsize = (24, 16)
    elif n_configs <= 6:
        n_rows, n_cols = 2, 3
        figsize = (30, 16)
    else:
        n_cols = 3
        n_rows = (n_configs + n_cols - 1) // n_cols
        figsize = (30, 8*n_rows)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    
    # Handle single subplot case
    if n_configs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot each configuration in its own subplot
    for i, (config_dir, result) in enumerate(config_results.items()):
        ax = axes[i]
        
        streamflow_regime = result['streamflow']
        snowmelt = result['snowmelt']
        glacier_all = result['glacier_all']
        name = result['name']
        
        # Plot filled polygons FIRST (bottom layer, in order of magnitude)
        # Snowmelt as filled polygon - light blue
        ax.fill_between(months, 0, snowmelt.values, 
                        color='#B3D9FF', alpha=0.7, label='Snowmelt', 
                        zorder=1, edgecolor='#6DB3F2', linewidth=1.5)
        
        # Glacier melt as filled polygon - brown/orange
        ax.fill_between(months, 0, glacier_all.values, 
                        color='#C17817', alpha=0.6, label='Glacier Runoff', 
                        zorder=2, edgecolor='#8B5A00', linewidth=1.5)
        
        # Plot observed streamflow (thick solid line)
        if 'observed' in streamflow_regime:
            ax.plot(months, streamflow_regime['observed'].values, 'k-', 
                   linewidth=3.5, label='Observed', zorder=4)
        
        # Plot simulated streamflow (dashed line)
        if 'simulated' in streamflow_regime:
            ax.plot(months, streamflow_regime['simulated'].values, 'C0--', 
                   linewidth=3, label='Simulated', zorder=3)
        
        # Formatting for this subplot
        ax.set_title(f'{name}', fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        ax.legend(loc='best', fontsize=13)
        
        # Set x-axis labels for bottom row
        if i >= (n_rows - 1) * n_cols or i >= n_configs - n_cols:
            ax.set_xticks(months)
            ax.set_xticklabels(month_names, fontsize=13)
        
        # Set y-axis label for leftmost column
        if i % n_cols == 0:
            ax.set_ylabel(f'Discharge ({unit_label})', fontsize=15, fontweight='bold')
        
        # Add statistics text box
        stats_lines = []
        if 'simulated' in streamflow_regime:
            sim_mean = streamflow_regime['simulated'].mean()
            snowmelt_mean = snowmelt.mean()
            glacier_mean = glacier_all.mean()
            
            if sim_mean > 0:
                snowmelt_pct = (snowmelt_mean / sim_mean) * 100
                glacier_pct = (glacier_mean / sim_mean) * 100
                
                stats_lines.append(f"Sim: {sim_mean:.2f} {unit_label}")
                stats_lines.append(f"Snow: {snowmelt_pct:.1f}%")
                stats_lines.append(f"Glacier: {glacier_pct:.1f}%")
        
        if stats_lines:
            stats_text = '\n'.join(stats_lines)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # Hide unused subplots
    for i in range(n_configs, len(axes)):
        axes[i].axis('off')
    
    # Add common x-label
    fig.text(0.5, 0.02, 'Month', ha='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.08)
    
    # Save plot
    save_path = plot_dir / f'streamflow_all_melt_regime_comparison_{unit}_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved streamflow all melt regime comparison plot to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"STREAMFLOW REGIME WITH ALL MELT CONTRIBUTIONS COMPARISON")
    print(f"{'='*60}")
    print(f"Catchment: {gauge_id}")
    print(f"Period: {validation_start} to {validation_end}")
    print(f"Unit: {unit_label}")
    print(f"Configurations processed: {len(config_results)}")
    
    for config_dir, result in config_results.items():
        name = result['name']
        streamflow = result['streamflow']
        snowmelt = result['snowmelt']
        glacier = result['glacier_all']
        
        print(f"\n{name}:")
        if 'simulated' in streamflow:
            sim_mean = streamflow['simulated'].mean()
            snowmelt_mean = snowmelt.mean()
            glacier_mean = glacier.mean()
            
            print(f"  Simulated flow: {sim_mean:.4f} {unit_label}")
            print(f"  Snowmelt: {snowmelt_mean:.4f} {unit_label} ({(snowmelt_mean/sim_mean*100):.1f}%)")
            print(f"  Glacier melt: {glacier_mean:.4f} {unit_label} ({(glacier_mean/sim_mean*100):.1f}%)")
            print(f"  Combined melt: {(snowmelt_mean+glacier_mean)/sim_mean*100:.1f}%")
    
    print(f"{'='*60}\n")
    
    return config_results

#--------------------------------------------------------------------------------

def plot_comprehensive_annual_water_balance_comparison(multi_config, validation_start=None, validation_end=None):
    """
    Plot comprehensive annual water balance comparison across multiple configurations.
    
    Creates side-by-side bar plots for each configuration showing:
    - Rainfall (GloGEM + HBV)
    - Snowmelt (GloGEM + HBV)
    - Ice melt (GloGEM)
    - Total input
    - Observed streamflow
    - Simulated streamflow
    
    All components are in mm/year for direct comparison.
    
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
        
    Returns:
    --------
    dict
        Dictionary containing annual water balance data for each configuration
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
    
    print(f"Creating comprehensive annual water balance comparison for {len(configs)} configurations:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Validation period: {validation_start} to {validation_end}")
    
    # Create plot directory
    plot_dir = create_multi_plot_dir(multi_config)
    
    # Store results for each configuration
    config_results = {}
    
    # Process each configuration
    for config_dir in configs:
        config_path = Path(multi_config['main_dir']) / config_dir
        catchment_dir = config_path / f"catchment_{gauge_id}" / model_type
        
        print(f"\n{'='*60}")
        print(f"Processing: {config_names.get(config_dir, config_dir)}")
        print(f"{'='*60}")
        
        # Create temporary config for single-config functions
        temp_config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': config_dir,
            'gauge_id': gauge_id,
            'model_type': model_type,
            'start_date': multi_config.get('start_date', '2000-01-01'),
            'end_date': multi_config.get('end_date', '2020-12-31'),
            'cali_end_date': multi_config.get('cali_end_date', '2010-12-31'),
            'coupled': multi_config.get('config_coupled', {}).get(config_dir, False)
        }
        
        try:
            # Load catchment area for unit conversion
            topo_dir = config_path / f"catchment_{gauge_id}" / "topo_files"
            catchment_shape_file = topo_dir / "HRU.shp"
            
            if not catchment_shape_file.exists():
                print(f"  ⚠️  Skipping {config_dir}: Catchment shapefile not found")
                continue
            
            import geopandas as gpd
            hru_gdf = gpd.read_file(catchment_shape_file)
            total_area_km2 = hru_gdf['Area_km2'].sum()
            conversion_m3s_to_mm_day = 86400 / (total_area_km2 * 1000000) * 1000
            print(f"  Catchment area: {total_area_km2:.2f} km²")
            
            # 1. LOAD STREAMFLOW DATA
            from postprocessing_single import load_hydrograph_data
            streamflow_data = load_hydrograph_data(temp_config)
            if streamflow_data is None:
                print(f"  ⚠️  Skipping {config_dir}: Could not load streamflow data")
                continue
            
            # Filter for validation period
            start_date = pd.to_datetime(validation_start)
            end_date = pd.to_datetime(validation_end)
            
            streamflow_mask = (streamflow_data['date'] >= start_date) & (streamflow_data['date'] <= end_date)
            streamflow_filtered = streamflow_data[streamflow_mask].copy()
            
            if len(streamflow_filtered) == 0:
                print(f"  ⚠️  Skipping {config_dir}: No streamflow data in validation period")
                continue
            
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
            
            # 2. LOAD GLOGEM DATA
            from postprocessing_single import load_glogem_data
            glogem_df = load_glogem_data(temp_config, unit='mm', plot=False)
            
            glogem_annual = None
            if glogem_df is not None:
                glogem_mask = (glogem_df['date'] >= start_date) & (glogem_df['date'] <= end_date)
                glogem_filtered = glogem_df[glogem_mask].copy()
                
                if len(glogem_filtered) > 0:
                    glogem_filtered['year'] = glogem_filtered['date'].dt.year
                    glogem_annual = glogem_filtered.groupby('year').agg({
                        'icemelt_normalized': 'sum',
                        'snowmelt_normalized': 'sum',
                        'rainfall_normalized': 'sum'
                    }).reset_index()
                    glogem_annual.columns = ['year', 'glogem_icemelt_mm', 'glogem_snowmelt_mm', 'glogem_rainfall_mm']
            
            # 3. LOAD HBV SNOWMELT DATA
            from postprocessing_single import load_snowmelt_mass_loadings
            snowmelt_df = load_snowmelt_mass_loadings(temp_config, validation_start, validation_end, unit='mm')
            
            snowmelt_annual = None
            if snowmelt_df is not None:
                snowmelt_col = 'snowmelt_mm_day' if 'snowmelt_mm_day' in snowmelt_df.columns else 'snowmelt_m3s'
                
                if snowmelt_col == 'snowmelt_m3s':
                    snowmelt_df['snowmelt_mm_day'] = snowmelt_df['snowmelt_m3s'] * conversion_m3s_to_mm_day
                
                snowmelt_df['year'] = snowmelt_df['date'].dt.year
                snowmelt_annual = snowmelt_df.groupby('year')['snowmelt_mm_day'].sum().reset_index()
                snowmelt_annual.columns = ['year', 'hbv_snowmelt_mm']
            
            # 4. LOAD HBV RAINFALL DATA
            from postprocessing_single import load_rainfall_mass_loadings
            rainfall_df = load_rainfall_mass_loadings(temp_config, validation_start, validation_end)
            
            rainfall_annual = None
            if rainfall_df is not None:
                rainfall_df['year'] = rainfall_df['date'].dt.year
                rainfall_annual = rainfall_df.groupby('year')['rainfall_mm_day'].sum().reset_index()
                rainfall_annual.columns = ['year', 'hbv_rainfall_mm']
            
            # 5. COMBINE ALL DATA
            annual_balance = streamflow_annual.copy()
            
            if glogem_annual is not None:
                annual_balance = pd.merge(annual_balance, glogem_annual, on='year', how='inner')
            if snowmelt_annual is not None:
                annual_balance = pd.merge(annual_balance, snowmelt_annual, on='year', how='inner')
            if rainfall_annual is not None:
                annual_balance = pd.merge(annual_balance, rainfall_annual, on='year', how='inner')
            
            if len(annual_balance) == 0:
                print(f"  ⚠️  Skipping {config_dir}: No overlapping years found")
                continue
            
            # 6. CALCULATE COMBINED COMPONENTS
            annual_balance['total_icemelt_mm'] = annual_balance.get('glogem_icemelt_mm', 0)
            annual_balance['total_snowmelt_mm'] = (
                annual_balance.get('glogem_snowmelt_mm', 0) + 
                annual_balance.get('hbv_snowmelt_mm', 0)
            )
            annual_balance['total_rainfall_mm'] = (
                annual_balance.get('glogem_rainfall_mm', 0) + 
                annual_balance.get('hbv_rainfall_mm', 0)
            )
            annual_balance['total_input_mm'] = (
                annual_balance['total_rainfall_mm'] + 
                annual_balance['total_icemelt_mm'] +
                annual_balance['total_snowmelt_mm']
            )
            
            # Store results
            config_results[config_dir] = {
                'annual_balance': annual_balance,
                'color': config_colors.get(config_dir, '#1f77b4'),
                'name': config_names.get(config_dir, config_dir)
            }
            
            print(f"  ✓ Successfully processed {config_names.get(config_dir, config_dir)}")
            print(f"    Years: {len(annual_balance)}")
            print(f"    Mean annual values (mm/year):")
            print(f"      Ice melt: {annual_balance['total_icemelt_mm'].mean():.1f}")
            print(f"      Snowmelt: {annual_balance['total_snowmelt_mm'].mean():.1f}")
            print(f"      Rainfall: {annual_balance['total_rainfall_mm'].mean():.1f}")
            print(f"      Sim. streamflow: {annual_balance['sim_streamflow_mm'].mean():.1f}")
            
        except Exception as e:
            print(f"  ⚠️  Error processing {config_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(config_results) == 0:
        print("\n❌ No configurations were successfully processed")
        return None
    
    # ===================================
    # CREATE COMPARISON BAR PLOT
    # ===================================
    
    # Calculate subplot layout
    n_configs = len(config_results)
    if n_configs <= 2:
        n_rows, n_cols = 1, n_configs
        figsize = (12*n_configs, 10)
    elif n_configs <= 4:
        n_rows, n_cols = 2, 2
        figsize = (24, 20)
    elif n_configs <= 6:
        n_rows, n_cols = 2, 3
        figsize = (30, 20)
    else:
        n_cols = 3
        n_rows = (n_configs + n_cols - 1) // n_cols
        figsize = (30, 10*n_rows)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False, sharey=True)
    
    # Handle single subplot case
    if n_configs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Plot each configuration in its own subplot
    for i, (config_dir, result) in enumerate(config_results.items()):
        ax = axes[i]
        annual_balance = result['annual_balance']
        name = result['name']
        
        x = np.arange(len(annual_balance))
        width = 0.15
        
        # Plot bars for each component
        ax.bar(x - 2.5*width, annual_balance['total_rainfall_mm'], width, 
               label='Rainfall', color='navy', alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.bar(x - 1.5*width, annual_balance['total_snowmelt_mm'], width, 
               label='Snowmelt', color='lightblue', alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.bar(x - 0.5*width, annual_balance['total_icemelt_mm'], width, 
               label='Ice Melt', color='grey', alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.bar(x + 0.5*width, annual_balance['total_input_mm'], width, 
               label='Total Input', color='darkgreen', alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.bar(x + 1.5*width, annual_balance['obs_streamflow_mm'], width, 
               label='Obs. Q', color='black', alpha=0.8, edgecolor='white', linewidth=1)
        
        ax.bar(x + 2.5*width, annual_balance['sim_streamflow_mm'], width, 
               label='Sim. Q', color='orange', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Formatting for this subplot
        ax.set_title(f'{name}', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(year)) for year in annual_balance['year']], rotation=45, fontsize=11)
        ax.grid(True, axis='y', alpha=0.3, zorder=0)
        
        # Set y-axis label for leftmost column
        if i % n_cols == 0:
            ax.set_ylabel('Annual Sum (mm/year)', fontsize=15, fontweight='bold')
        
        # Set x-axis label for bottom row
        if i >= (n_rows - 1) * n_cols or i >= n_configs - n_cols:
            ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        
        # Add legend to first subplot
        if i == 0:
            ax.legend(fontsize=11, loc='upper left', ncol=2)
    
    # Hide unused subplots
    for i in range(n_configs, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dir / f'comprehensive_annual_water_balance_comparison_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved water balance comparison plot to: {save_path}")
    plt.show()
    
    # ===================================
    # CREATE SUMMARY STATISTICS TABLE
    # ===================================
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ANNUAL WATER BALANCE COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Catchment: {gauge_id}")
    print(f"Period: {validation_start} to {validation_end}")
    print(f"Configurations processed: {len(config_results)}")
    
    for config_dir, result in config_results.items():
        name = result['name']
        annual_balance = result['annual_balance']
        
        print(f"\n{name}:")
        print(f"  Years: {len(annual_balance)} ({int(annual_balance['year'].min())} - {int(annual_balance['year'].max())})")
        print(f"  Mean annual values (mm/year):")
        print(f"    Rainfall: {annual_balance['total_rainfall_mm'].mean():.1f} ± {annual_balance['total_rainfall_mm'].std():.1f}")
        print(f"    Snowmelt: {annual_balance['total_snowmelt_mm'].mean():.1f} ± {annual_balance['total_snowmelt_mm'].std():.1f}")
        print(f"    Ice melt: {annual_balance['total_icemelt_mm'].mean():.1f} ± {annual_balance['total_icemelt_mm'].std():.1f}")
        print(f"    Total input: {annual_balance['total_input_mm'].mean():.1f} ± {annual_balance['total_input_mm'].std():.1f}")
        print(f"    Obs. streamflow: {annual_balance['obs_streamflow_mm'].mean():.1f} ± {annual_balance['obs_streamflow_mm'].std():.1f}")
        print(f"    Sim. streamflow: {annual_balance['sim_streamflow_mm'].mean():.1f} ± {annual_balance['sim_streamflow_mm'].std():.1f}")
        
        # Calculate contribution percentages
        sim_mean = annual_balance['sim_streamflow_mm'].mean()
        if sim_mean > 0:
            ice_pct = (annual_balance['total_icemelt_mm'].mean() / sim_mean) * 100
            snow_pct = (annual_balance['total_snowmelt_mm'].mean() / sim_mean) * 100
            rain_pct = (annual_balance['total_rainfall_mm'].mean() / sim_mean) * 100
            
            print(f"  Contributions to simulated streamflow:")
            print(f"    Ice melt: {ice_pct:.1f}%")
            print(f"    Snowmelt: {snow_pct:.1f}%")
            print(f"    Rainfall: {rain_pct:.1f}%")
            print(f"    Total: {ice_pct + snow_pct + rain_pct:.1f}%")
    
    print(f"{'='*80}\n")
    
    return config_results

#--------------------------------------------------------------------------------
################################ uncertainties ##################################
#--------------------------------------------------------------------------------

def plot_uncertainties_comparison(multi_config, n_runs=50, validation_start=None, validation_end=None):
    """
    Plot uncertainty analysis comparison across multiple configurations.
    Runs the model n_runs times with best parameter sets for each configuration
    and plots the uncertainty envelope for each configuration in separate subplots.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-configuration dictionary
    n_runs : int
        Number of best parameter sets to run for each configuration
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    dict
        Dictionary containing uncertainty results for each configuration
    """
    
    import subprocess
    import shutil
    
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
    
    # Get Raven executable from multi_config
    raven_executable = multi_config.get('raven_executable')
    if not raven_executable:
        print("Error: 'raven_executable' not found in multi_config")
        return None
    
    print(f"Creating uncertainty analysis comparison for {len(configs)} configurations:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Number of runs per config: {n_runs}")
    print(f"  - Validation period: {validation_start} to {validation_end}")
    print(f"  - Raven executable: {raven_executable}")
    
    # Check if Raven executable exists
    if not Path(raven_executable).exists():
        print(f"Error: Raven executable not found at: {raven_executable}")
        return None
    
    # Create plot directory
    plot_dir = create_multi_plot_dir(multi_config)
    
    # Store results for each configuration
    config_results = {}
    obs_data = None  # Store observed data (should be same for all configs)
    
    # Process each configuration
    for config_dir in configs:
        print(f"\nProcessing configuration: {config_dir}")
        
        # Create individual config dictionary for this configuration
        individual_config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': config_dir,
            'gauge_id': gauge_id,
            'start_date': multi_config.get('start_date', '2000-01-01'),
            'end_date': multi_config.get('end_date', '2020-12-31'),
            'cali_end_date': multi_config.get('cali_end_date', '2009-12-31'),
            'model_type': model_type,
            'raven_executable': raven_executable,  # Use from multi_config
            'coupled': multi_config.get('config_coupled', {}).get(config_dir, False),
        }
        
        try:
            # Set up paths
            config_dir_path = Path(individual_config['main_dir']) / individual_config['config_dir']
            template_dir = config_dir_path / f"catchment_{gauge_id}" / model_type / "templates"
            output_dir = config_dir_path / f"catchment_{gauge_id}" / model_type / "output"
            
            # Check if template directory exists
            if not template_dir.exists():
                print(f"  Error: Template directory not found: {template_dir}")
                continue
            
            template_files = list(template_dir.glob("*.tpl"))
            if not template_files:
                print(f"  Error: No .tpl files found in template directory")
                continue
            
            print(f"  Found template files: {[f.name for f in template_files]}")
            
            # Load calibration results
            results_file = output_dir / f"raven_sceua_{gauge_id}_{model_type}.csv"
            if not results_file.exists():
                # Try alternative file patterns
                alt_files = list(output_dir.glob(f"*sceua*.csv"))
                if alt_files:
                    results_file = alt_files[0]
                    print(f"  Using alternative results file: {results_file}")
                else:
                    print(f"  Error: No SCEUA results files found in {output_dir}")
                    continue
            
            df = pd.read_csv(results_file)
            if 'like1' not in df.columns:
                print(f"  Error: 'like1' column not found in results file")
                continue
            
            print(f"  Loaded {len(df)} parameter sets from results file")
            
            # Convert negative KGE to positive KGE
            df['KGE'] = -df['like1']
            
            # Select best runs
            best_runs = df.sort_values('KGE', ascending=False).head(n_runs)
            param_cols = [col for col in df.columns if col not in ['like1', 'KGE']]
            
            print(f"  Best KGE range: {best_runs['KGE'].min():.4f} to {best_runs['KGE'].max():.4f}")
            
            # Prepare output folder for simulations
            sim_results_dir = output_dir / f"uncertainty_{n_runs}_simulations_{gauge_id}"
            sim_results_dir.mkdir(exist_ok=True)
            
            # Run simulations
            print(f"  Running {n_runs} simulations...")
            hydrographs = []
            successful_runs = 0
            failed_runs = 0
            
            # Test first run with verbose output
            first_run = True
            
            for i, (idx, row) in enumerate(best_runs.iterrows()):
                if i % 10 == 0:  # Progress indicator
                    print(f"    Processing run {i+1}/{n_runs}...")
                
                run_dir = sim_results_dir / f"run_{idx}"
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                
                try:
                    # Set up run directory
                    shutil.copytree(template_dir, run_dir)
                    
                    if not postprocessing_single.fill_templates_with_parameters(run_dir, param_cols, row[param_cols]):
                        failed_runs += 1
                        if first_run:
                            print(f"    First run failed at template filling stage")
                        continue
                    
                    if not postprocessing_single.setup_raven_run_directory(run_dir, individual_config):
                        failed_runs += 1
                        if first_run:
                            print(f"    First run failed at setup stage")
                        continue
                    
                    # Run Raven
                    model_file = run_dir / f"{gauge_id}_{model_type}"
                    run_output_dir = run_dir / "output"
                    run_output_dir.mkdir(exist_ok=True)
                    
                    # Use the raven_executable from multi_config
                    cmd = [str(raven_executable), str(model_file), "-o", str(run_output_dir)]
                    
                    if first_run:
                        print(f"    First run command: {' '.join(cmd)}")
                        print(f"    Working directory: {run_dir}")
                        print(f"    Model file exists: {model_file.exists()}")
                        print(f"    Output dir exists: {run_output_dir.exists()}")
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=run_dir)
                    
                    if first_run:
                        print(f"    First run return code: {result.returncode}")
                        if result.stdout:
                            print(f"    Stdout (first 500 chars): {result.stdout[:500]}")
                        if result.stderr:
                            print(f"    Stderr (first 500 chars): {result.stderr[:500]}")
                    
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
                                
                                if first_run:
                                    print(f"    First run successful! Generated {len(monthly_mean)} monthly values")
                                
                                # Clean up run directory
                                postprocessing_single.cleanup_raven_run_directory(run_dir)
                            else:
                                failed_runs += 1
                                if first_run:
                                    print(f"    First run failed: No simulation column found in hydrograph file")
                                    print(f"    Available columns: {df_hydro.columns.tolist()}")
                        else:
                            failed_runs += 1
                            if first_run:
                                print(f"    First run failed: Hydrograph file not generated")
                                print(f"    Files in output dir: {list(run_output_dir.glob('*'))}")
                    else:
                        failed_runs += 1
                        if first_run:
                            print(f"    First run failed: Raven execution failed")
                    
                    first_run = False
                
                except Exception as e:
                    failed_runs += 1
                    if first_run:
                        print(f"    First run exception: {e}")
                    first_run = False
                    continue
            
            print(f"  Successfully processed {successful_runs} out of {n_runs} runs ({failed_runs} failed)")
            
            if len(hydrographs) == 0:
                print(f"  Warning: No successful runs for {config_dir}")
                continue
            
            # Load observed data (only once)
            if obs_data is None:
                obs_hydro_data = postprocessing_single.load_hydrograph_data(individual_config)
                if obs_hydro_data is not None and 'obs_Q' in obs_hydro_data.columns:
                    mask = (obs_hydro_data['date'] >= validation_start) & (obs_hydro_data['date'] <= validation_end)
                    obs_monthly = obs_hydro_data[mask].copy()
                    obs_monthly['month'] = obs_monthly['date'].dt.month
                    obs_data = obs_monthly.groupby('month')['obs_Q'].mean()
                    print(f"  ✓ Loaded observed data for reference")
            
            # Store results for this configuration
            config_results[config_dir] = {
                'hydrographs': hydrographs,
                'best_kge': best_runs['KGE'].iloc[0],
                'worst_kge': best_runs['KGE'].iloc[-1],
                'successful_runs': successful_runs,
                'color': config_colors.get(config_dir, 'C0'),
                'name': config_names.get(config_dir, config_dir),
                'coupled': individual_config['coupled']
            }
            
            print(f"  ✓ Successfully processed uncertainty analysis for {config_dir}")
            print(f"    Successful runs: {successful_runs}/{n_runs}")
            print(f"    Best KGE: {best_runs['KGE'].iloc[0]:.4f}")
            
        except Exception as e:
            print(f"  Error processing {config_dir}: {e}")
            continue
    
    if len(config_results) == 0:
        print("No configurations processed successfully")
        return None
    
    # [Rest of the plotting code remains the same...]
    
    # Calculate subplot layout
    n_configs = len(config_results)
    if n_configs <= 2:
        n_rows, n_cols = 1, n_configs
        figsize = (8 * n_configs, 6)
    elif n_configs <= 4:
        n_rows, n_cols = 2, 2
        figsize = (16, 10)
    elif n_configs <= 6:
        n_rows, n_cols = 2, 3
        figsize = (20, 10)
    else:
        # For more configurations, use more rows
        n_cols = 3
        n_rows = (n_configs + n_cols - 1) // n_cols
        figsize = (20, 6 * n_rows)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    
    # Handle single subplot case
    if n_configs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_configs > 1 else [axes]
    else:
        axes = axes.flatten()
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot each configuration in its own subplot
    for i, (config_dir, result) in enumerate(config_results.items()):
        ax = axes[i]
        
        hydrographs = result['hydrographs']
        color = result['color']
        name = result['name']
        coupled = result['coupled']
        successful_runs = result['successful_runs']
        best_kge = result['best_kge']
        
        # Plot observed data first (same for all configurations)
        if obs_data is not None:
            ax.plot(obs_data.index, obs_data.values, 'k-', linewidth=3, 
                   label='Observed', zorder=5)
        
        # Plot all simulations except the best one (in grey)
        for j, monthly_mean in enumerate(hydrographs[1:], 1):  # Skip first (best) simulation
            ax.plot(monthly_mean.index, monthly_mean.values, color='grey', 
                   linewidth=1, alpha=0.3, zorder=1)
        
        # Plot the best simulation on top
        if len(hydrographs) > 0:
            ax.plot(hydrographs[0].index, hydrographs[0].values, color=color, 
                   linewidth=2.5, label='Best Simulation', zorder=4)
        
        # Add a single grey line to legend for other simulations
        if len(hydrographs) > 1:
            ax.plot([], [], color='grey', linewidth=1, alpha=0.3, 
                   label=f'Other {len(hydrographs)-1} Simulations')
        
        # Formatting for this subplot
        ax.set_title(f'{name}\n({"GloGEM+HBV" if coupled else "HBV"}, {successful_runs} runs)', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(months)
        ax.set_xticklabels(month_names)
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        
        # Add legend only to the first subplot
        if i == 0:
            ax.legend(loc='best', fontsize=10)
        
        # Set y-axis label for leftmost column
        if i % n_cols == 0:
            ax.set_ylabel('Discharge (m³/s)', fontsize=11)
        
        # Set x-axis label for bottom row
        if i >= (n_rows - 1) * n_cols or i >= n_configs - n_cols:
            ax.set_xlabel('Month', fontsize=11)
        
        # Add performance statistics as text
        stats_text = f"Best KGE: {best_kge:.3f}\nSuccessful: {successful_runs}/{n_runs}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'Parameter Uncertainty Analysis - Catchment {gauge_id}\n'
                f'Validation Period: {validation_start} to {validation_end} ({n_runs} runs per config)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    save_path = plot_dir / f'uncertainties_comparison_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved uncertainty analysis comparison plot to: {save_path}")
    plt.show()
    
    # Print comprehensive summary
    print(f"\nParameter Uncertainty Analysis Summary:")
    print(f"  Configurations processed: {len(config_results)}")
    print(f"  Validation period: {validation_start} to {validation_end}")
    print(f"  Runs per configuration: {n_runs}")
    
    print(f"\nUncertainty Analysis by Configuration:")
    for config_dir, result in config_results.items():
        name = result['name']
        successful_runs = result['successful_runs']
        best_kge = result['best_kge']
        worst_kge = result['worst_kge']
        coupled = result['coupled']
        
        print(f"\n  {name} ({'GloGEM+HBV' if coupled else 'HBV'}):")
        print(f"    Successful runs: {successful_runs}/{n_runs} ({successful_runs/n_runs*100:.1f}%)")
        print(f"    Best KGE: {best_kge:.4f}")
        print(f"    Worst KGE in selection: {worst_kge:.4f}")
        print(f"    KGE range: {worst_kge:.4f} to {best_kge:.4f}")
    
    # Compare uncertainty levels between configurations
    if len(config_results) > 1:
        print(f"\nConfiguration Performance Ranking (by best KGE):")
        sorted_configs = sorted(config_results.items(), key=lambda x: x[1]['best_kge'], reverse=True)
        for i, (config_dir, result) in enumerate(sorted_configs, 1):
            name = result['name']
            best_kge = result['best_kge']
            successful_runs = result['successful_runs']
            print(f"  {i}. {name}: KGE={best_kge:.4f} ({successful_runs}/{n_runs} runs)")
    
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
        
        # Create paths for this configuration
        config_dir_path = Path(multi_config['main_dir']) / config_dir
        
        try:
            # Load HRU shapefile to identify glacier HRUs
            topo_dir = config_dir_path / f"catchment_{gauge_id}" / "topo_files"
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
            output_dir = config_dir_path / f"catchment_{gauge_id}" / model_type / "output"
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
        
        # Create paths for this configuration
        config_dir_path = Path(multi_config['main_dir']) / config_dir
        
        try:
            # Load HRU group temperature data
            output_dir = config_dir_path / f"catchment_{gauge_id}" / model_type / "output"
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