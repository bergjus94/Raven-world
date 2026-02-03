#--------------------------------------------------------------------------------
################################## packages #####################################
#--------------------------------------------------------------------------------

import postprocessing_single
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import matplotlib.dates as mdates

#--------------------------------------------------------------------------------
################################### setup #######################################
#--------------------------------------------------------------------------------

def setup_multi_model_directories(multi_config):
    """
    Set up directory structure for multi-model comparison outputs.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-model configuration dictionary containing:
        - main_dir: base directory path
        - config: config directory name
        - gauge_id: catchment identifier
        - model_type: list of model types to compare
        
    Returns:
    --------
    dict
        Dictionary containing plot directory paths for comparisons
    """
    
    gauge_id = multi_config['gauge_id']
    config_dir = Path(multi_config['main_dir']) / multi_config['config']
    
    # Create base comparison directory
    comparison_base = config_dir / f"catchment_{gauge_id}" / "model_comparisons"
    
    # Create subdirectories for different plot types
    plot_dirs = {
        'hydrographs': comparison_base / "hydrographs",
        'swe': comparison_base / "swe",
        'contributions': comparison_base / "contributions",
        'performance': comparison_base / "performance",
        'water_balance': comparison_base / "water_balance",
        'uncertainty': comparison_base / "uncertainty",
        'storage': comparison_base / "storage"  # ✅ ADD THIS LINE
    }
    
    # Create all directories
    for plot_dir in plot_dirs.values():
        plot_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"MULTI-MODEL COMPARISON SETUP")
    print(f"{'='*60}")
    print(f"Catchment: {gauge_id}")
    print(f"Models to compare: {', '.join(multi_config['model_type'])}")
    print(f"Output directory: {comparison_base}")
    print(f"{'='*60}\n")
    
    return plot_dirs


#--------------------------------------------------------------------------------
################################ hydrographs ####################################
#--------------------------------------------------------------------------------

def plot_multi_model_hydrograph_regime(multi_config, plot_dirs, validation_start=None, validation_end=None, unit='mm'):
    """
    Plot hydrological regime (monthly mean) for multiple models on the same plot.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-model configuration dictionary
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    unit : str, optional
        Unit for discharge ('mm' for mm/day, 'm3' for m³/s), default is 'mm'
        
    Returns:
    --------
    dict
        Dictionary containing regime data for all models
    """
    
    gauge_id = multi_config['gauge_id']
    model_types = multi_config['model_type']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    print(f"Creating multi-model hydrograph regime comparison:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Period: {validation_start} to {validation_end}")
    print(f"  - Unit: {unit}")
    print(f"  - Models: {', '.join(model_types)}")
    
    # Store results for each model
    all_regimes = {}
    obs_regime = None
    
    # Load data for each model
    for model_type in model_types:
        print(f"\n  Loading {model_type} data...")
        
        # Create single-model config
        config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': multi_config['config'],
            'gauge_id': gauge_id,
            'model_type': model_type,
            'start_date': multi_config['start_date'],
            'end_date': multi_config['end_date'],
            'cali_end_date': multi_config['cali_end_date'],
            'coupled': multi_config['config_coupled'].get(model_type, False)
        }
        
        # Load hydrograph data
        data = postprocessing_single.load_hydrograph_data(config)
        
        if data is None:
            print(f"    ✗ No data found for {model_type}")
            continue
        
        # Filter for validation period
        validation_mask = (data['date'] >= validation_start) & (data['date'] <= validation_end)
        df_validation = data[validation_mask].copy()
        
        if len(df_validation) == 0:
            print(f"    ✗ No data in validation period for {model_type}")
            continue
        
        # Convert to requested unit
        conversion_factor = None
        if unit == 'mm':
            config_dir = Path(multi_config['main_dir']) / multi_config['config']
            topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
            catchment_shape_file = topo_dir / "HRU.shp"
            
            try:
                import geopandas as gpd
                if catchment_shape_file.exists():
                    hru_gdf = gpd.read_file(catchment_shape_file)
                    total_area_km2 = hru_gdf['Area_km2'].sum()
                    conversion_factor = 86400 / (total_area_km2 * 1000000) * 1000
                    print(f"    - Catchment area: {total_area_km2:.2f} km²")
            except Exception as e:
                print(f"    ⚠️  Could not load catchment area: {e}")
        
        # Convert discharge
        if unit == 'mm' and conversion_factor is not None:
            if 'sim_Q' in df_validation.columns:
                df_validation['sim_Q_converted'] = df_validation['sim_Q'] * conversion_factor
            if 'obs_Q' in df_validation.columns:
                df_validation['obs_Q_converted'] = df_validation['obs_Q'] * conversion_factor
            unit_label = 'mm/day'
        else:
            if 'sim_Q' in df_validation.columns:
                df_validation['sim_Q_converted'] = df_validation['sim_Q']
            if 'obs_Q' in df_validation.columns:
                df_validation['obs_Q_converted'] = df_validation['obs_Q']
            unit_label = 'm³/s'
        
        # Calculate monthly means
        df_validation['month'] = df_validation['date'].dt.month
        
        if 'sim_Q_converted' in df_validation.columns:
            sim_regime = df_validation.groupby('month')['sim_Q_converted'].mean()
            all_regimes[model_type] = sim_regime
            print(f"    ✓ Loaded {model_type} regime (mean: {sim_regime.mean():.4f} {unit_label})")
        
        # Store observed data (only once, should be same for all models)
        if obs_regime is None and 'obs_Q_converted' in df_validation.columns:
            obs_regime = df_validation.groupby('month')['obs_Q_converted'].mean()
            print(f"    ✓ Loaded observed regime (mean: {obs_regime.mean():.4f} {unit_label})")
    
    if len(all_regimes) == 0:
        print("\n✗ ERROR: No valid model data found for comparison")
        return None
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot observed data first (if available)
    if obs_regime is not None:
        plt.plot(months, obs_regime.values, 'k-', linewidth=3, 
                label='Observed', marker='o', markersize=8, zorder=100)
    
    # Plot each model
    for model_type, regime in all_regimes.items():
        color = multi_config['config_colors'].get(model_type, 'gray')
        name = multi_config['config_names'].get(model_type, model_type)
        
        plt.plot(months, regime.values, color=color, linewidth=2.5, 
                label=name, marker='s', markersize=6, alpha=0.8)
    
    # Formatting
    plt.xlabel('Month', fontsize=14, fontweight='bold')
    plt.ylabel(f'Discharge ({unit_label})', fontsize=14, fontweight='bold')
    plt.title(f'Hydrological Regime Comparison - Catchment {gauge_id}\n'
             f'Validation Period: {validation_start} to {validation_end}', 
             fontsize=16, fontweight='bold')
    plt.xticks(months, month_names, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['hydrographs'] / f'multi_model_regime_{unit}_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved multi-model regime plot to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\nMulti-Model Regime Summary:")
    print(f"  Period: {validation_start} to {validation_end}")
    print(f"  Unit: {unit_label}")
    
    if obs_regime is not None:
        print(f"\n  Observed:")
        print(f"    Mean: {obs_regime.mean():.4f} {unit_label}")
        print(f"    Peak month: {month_names[obs_regime.idxmax()-1]} ({obs_regime.max():.4f} {unit_label})")
    
    print(f"\n  Models:")
    for model_type, regime in all_regimes.items():
        name = multi_config['config_names'].get(model_type, model_type)
        print(f"    {name:10s}: Mean={regime.mean():.4f} {unit_label}, Peak={month_names[regime.idxmax()-1]} ({regime.max():.4f} {unit_label})")
    
    return {
        'observed': obs_regime,
        'models': all_regimes,
        'unit': unit_label,
        'save_path': save_path
    }

#--------------------------------------------------------------------------------

def plot_multi_model_hydrograph_timeseries(multi_config, plot_dirs, validation_start=None, validation_end=None, unit='mm'):
    """
    Plot hydrograph time series for multiple models on the same plot.
    Shows both calibration and validation periods in separate subplots.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-model configuration dictionary
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    unit : str, optional
        Unit for discharge ('mm' for mm/day, 'm3' for m³/s), default is 'mm'
        
    Returns:
    --------
    dict
        Dictionary containing time series data for all models
    """
    
    gauge_id = multi_config['gauge_id']
    model_types = multi_config['model_type']
    
    # Use dates from config if not provided
    cali_start = multi_config.get('start_date', '2000-01-01')
    cali_end = multi_config.get('cali_end_date', '2009-12-31')
    
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    print(f"Creating multi-model hydrograph time series comparison:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Calibration: {cali_start} to {cali_end}")
    print(f"  - Validation: {validation_start} to {validation_end}")
    print(f"  - Unit: {unit}")
    print(f"  - Models: {', '.join(model_types)}")
    
    # Get conversion factor
    conversion_factor = None
    if unit == 'mm':
        config_dir = Path(multi_config['main_dir']) / multi_config['config']
        topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
        catchment_shape_file = topo_dir / "HRU.shp"
        
        try:
            import geopandas as gpd
            if catchment_shape_file.exists():
                hru_gdf = gpd.read_file(catchment_shape_file)
                total_area_km2 = hru_gdf['Area_km2'].sum()
                conversion_factor = 86400 / (total_area_km2 * 1000000) * 1000
        except Exception as e:
            print(f"  ⚠️  Could not load catchment area: {e}")
    
    unit_label = 'mm/day' if unit == 'mm' and conversion_factor is not None else 'm³/s'
    
    # Store results for each model
    all_data = {}
    obs_data = None
    
    # Load data for each model
    for model_type in model_types:
        print(f"\n  Loading {model_type} data...")
        
        # Create single-model config
        config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': multi_config['config'],
            'gauge_id': gauge_id,
            'model_type': model_type,
            'start_date': multi_config['start_date'],
            'end_date': multi_config['end_date'],
            'cali_end_date': multi_config['cali_end_date'],
            'coupled': multi_config['config_coupled'].get(model_type, False)
        }
        
        # Load hydrograph data
        data = postprocessing_single.load_hydrograph_data(config)
        
        if data is None:
            print(f"    ✗ No data found for {model_type}")
            continue
        
        # Convert discharge
        if unit == 'mm' and conversion_factor is not None:
            if 'sim_Q' in data.columns:
                data['sim_Q_converted'] = data['sim_Q'] * conversion_factor
            if 'obs_Q' in data.columns:
                data['obs_Q_converted'] = data['obs_Q'] * conversion_factor
        else:
            if 'sim_Q' in data.columns:
                data['sim_Q_converted'] = data['sim_Q']
            if 'obs_Q' in data.columns:
                data['obs_Q_converted'] = data['obs_Q']
        
        all_data[model_type] = data
        print(f"    ✓ Loaded {model_type} data ({len(data)} records)")
        
        # Store observed data (only once)
        if obs_data is None and 'obs_Q_converted' in data.columns:
            obs_data = data[['date', 'obs_Q_converted']].copy()
    
    if len(all_data) == 0:
        print("\n✗ ERROR: No valid model data found for comparison")
        return None
    
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    
    # Plot calibration period
    ax = axes[0]
    
    if obs_data is not None:
        cali_mask = (obs_data['date'] >= cali_start) & (obs_data['date'] <= cali_end)
        ax.plot(obs_data[cali_mask]['date'], obs_data[cali_mask]['obs_Q_converted'], 
               'k-', linewidth=2, label='Observed', alpha=0.8)
    
    for model_type, data in all_data.items():
        color = multi_config['config_colors'].get(model_type, 'gray')
        name = multi_config['config_names'].get(model_type, model_type)
        
        cali_mask = (data['date'] >= cali_start) & (data['date'] <= cali_end)
        ax.plot(data[cali_mask]['date'], data[cali_mask]['sim_Q_converted'], 
               color=color, linewidth=1.5, label=name, alpha=0.7)
    
    ax.set_title(f'Calibration Period ({cali_start} to {cali_end})', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Discharge ({unit_label})', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot validation period
    ax = axes[1]
    
    if obs_data is not None:
        val_mask = (obs_data['date'] >= validation_start) & (obs_data['date'] <= validation_end)
        ax.plot(obs_data[val_mask]['date'], obs_data[val_mask]['obs_Q_converted'], 
               'k-', linewidth=2, label='Observed', alpha=0.8)
    
    for model_type, data in all_data.items():
        color = multi_config['config_colors'].get(model_type, 'gray')
        name = multi_config['config_names'].get(model_type, model_type)
        
        val_mask = (data['date'] >= validation_start) & (data['date'] <= validation_end)
        ax.plot(data[val_mask]['date'], data[val_mask]['sim_Q_converted'], 
               color=color, linewidth=1.5, label=name, alpha=0.7)
    
    ax.set_title(f'Validation Period ({validation_start} to {validation_end})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'Discharge ({unit_label})', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Overall title
    fig.suptitle(f'Multi-Model Hydrograph Comparison - Catchment {gauge_id}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    save_path = plot_dirs['hydrographs'] / f'multi_model_timeseries_{unit}_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved multi-model time series plot to: {save_path}")
    plt.show()
    
    return {
        'observed': obs_data,
        'models': all_data,
        'unit': unit_label,
        'save_path': save_path
    }

#--------------------------------------------------------------------------------

def plot_multi_model_random_year_hydrograph(multi_config, plot_dirs, validation_start=None, validation_end=None, 
                                           random_seed=42, unit='mm'):
    """
    Plot hydrograph for a random year in the validation period comparing multiple models.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-model configuration dictionary
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
    random_seed : int, optional
        Seed for random year selection (default: 42)
    unit : str, optional
        Unit for discharge ('mm' for mm/day, 'm3' for m³/s), default is 'mm'
        
    Returns:
    --------
    dict
        Dictionary containing random year data for all models
    """
    
    gauge_id = multi_config['gauge_id']
    model_types = multi_config['model_type']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    print(f"Creating multi-model random year hydrograph comparison:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Period: {validation_start} to {validation_end}")
    print(f"  - Random seed: {random_seed}")
    print(f"  - Unit: {unit}")
    print(f"  - Models: {', '.join(model_types)}")
    
    # Get conversion factor if needed
    conversion_factor = None
    if unit == 'mm':
        config_dir = Path(multi_config['main_dir']) / multi_config['config']
        topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
        catchment_shape_file = topo_dir / "HRU.shp"
        
        try:
            import geopandas as gpd
            if catchment_shape_file.exists():
                hru_gdf = gpd.read_file(catchment_shape_file)
                total_area_km2 = hru_gdf['Area_km2'].sum()
                conversion_factor = 86400 / (total_area_km2 * 1000000) * 1000
        except Exception as e:
            print(f"  ⚠️  Could not load catchment area: {e}")
    
    unit_label = 'mm/day' if unit == 'mm' and conversion_factor is not None else 'm³/s'
    
    # Store results for each model
    all_data = {}
    obs_data = None
    rand_year = None
    
    # Load data for each model
    for model_type in model_types:
        print(f"\n  Loading {model_type} data...")
        
        # Create single-model config
        config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': multi_config['config'],
            'gauge_id': gauge_id,
            'model_type': model_type,
            'start_date': multi_config['start_date'],
            'end_date': multi_config['end_date'],
            'cali_end_date': multi_config['cali_end_date'],
            'coupled': multi_config['config_coupled'].get(model_type, False)
        }
        
        # Load hydrograph data
        data = postprocessing_single.load_hydrograph_data(config)
        
        if data is None:
            print(f"    ✗ No data found for {model_type}")
            continue
        
        # Filter for validation period
        validation_mask = (data['date'] >= validation_start) & (data['date'] <= validation_end)
        df_validation = data[validation_mask].copy()
        
        if len(df_validation) == 0:
            print(f"    ✗ No data in validation period for {model_type}")
            continue
        
        # Convert discharge
        if unit == 'mm' and conversion_factor is not None:
            if 'sim_Q' in data.columns:
                data['sim_Q_converted'] = data['sim_Q'] * conversion_factor
            if 'obs_Q' in data.columns:
                data['obs_Q_converted'] = data['obs_Q'] * conversion_factor
        else:
            if 'sim_Q' in data.columns:
                data['sim_Q_converted'] = data['sim_Q']
            if 'obs_Q' in data.columns:
                data['obs_Q_converted'] = data['obs_Q']
        
        all_data[model_type] = data
        print(f"    ✓ Loaded {model_type} data ({len(data)} records)")
        
        # Pick random year (only once, use same year for all models)
        if rand_year is None:
            val_years = pd.Series(df_validation['date'].dt.year.unique())
            if len(val_years) > 0:
                np.random.seed(random_seed)
                rand_year = np.random.choice(val_years)
                print(f"    ✓ Selected random year: {rand_year}")
        
        # Store observed data (only once)
        if obs_data is None and 'obs_Q_converted' in data.columns:
            obs_data = data[['date', 'obs_Q_converted']].copy()
    
    if len(all_data) == 0:
        print("\n✗ ERROR: No valid model data found for comparison")
        return None
    
    if rand_year is None:
        print("\n✗ ERROR: Could not select random year")
        return None
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot observed data for random year
    if obs_data is not None:
        year_mask = (obs_data['date'].dt.year == rand_year)
        plt.plot(obs_data[year_mask]['date'], obs_data[year_mask]['obs_Q_converted'], 
                'k-', linewidth=2.5, label='Observed', alpha=0.8, zorder=10)
    
    # Plot each model for random year
    for model_type, data in all_data.items():
        color = multi_config['config_colors'].get(model_type, 'gray')
        name = multi_config['config_names'].get(model_type, model_type)
        
        year_mask = (data['date'].dt.year == rand_year)
        plt.plot(data[year_mask]['date'], data[year_mask]['sim_Q_converted'], 
                color=color, linewidth=2, label=name, alpha=0.7)
    
    # Formatting
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel(f'Discharge ({unit_label})', fontsize=12, fontweight='bold')
    plt.title(f'Multi-Model Hydrograph Comparison - Random Year {rand_year}\nCatchment {gauge_id}', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['hydrographs'] / f'multi_model_random_year_{rand_year}_{unit}_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved random year comparison plot to: {save_path}")
    plt.show()
    
    return {
        'observed': obs_data,
        'models': all_data,
        'random_year': rand_year,
        'unit': unit_label,
        'save_path': save_path
    }

#--------------------------------------------------------------------------------

def plot_multi_model_scatter(multi_config, plot_dirs, validation_start=None, validation_end=None):
    """
    Create scatter plots (observed vs simulated) for multiple models on the same figure.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-model configuration dictionary
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    dict
        Dictionary containing statistics for all models
    """
    
    gauge_id = multi_config['gauge_id']
    model_types = multi_config['model_type']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    print(f"Creating multi-model scatter plot comparison:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Period: {validation_start} to {validation_end}")
    print(f"  - Models: {', '.join(model_types)}")
    
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    
    # Store results for each model
    all_stats = {}
    
    # Calculate grid dimensions
    n_models = len(model_types)
    n_cols = min(2, n_models)  # Max 2 columns
    n_rows = int(np.ceil(n_models / n_cols))
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 7*n_rows), squeeze=False)
    axes_flat = axes.flatten()
    
    # Find global min/max for consistent axis limits
    global_min = float('inf')
    global_max = float('-inf')
    
    # First pass: collect data and find limits
    model_data = {}
    for model_type in model_types:
        config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': multi_config['config'],
            'gauge_id': gauge_id,
            'model_type': model_type,
            'start_date': multi_config['start_date'],
            'end_date': multi_config['end_date'],
            'cali_end_date': multi_config['cali_end_date'],
            'coupled': multi_config['config_coupled'].get(model_type, False)
        }
        
        data = postprocessing_single.load_hydrograph_data(config)
        if data is None:
            continue
        
        mask = (data['date'] >= start_date) & (data['date'] <= end_date)
        df = data[mask].copy()
        df = df.dropna(subset=['obs_Q', 'sim_Q'])
        
        if len(df) > 0:
            model_data[model_type] = df
            global_min = min(global_min, df['obs_Q'].min(), df['sim_Q'].min())
            global_max = max(global_max, df['obs_Q'].max(), df['sim_Q'].max())
    
    # Add margin
    margin = (global_max - global_min) * 0.05
    global_min -= margin
    global_max += margin
    
    # Second pass: create plots
    for i, model_type in enumerate(model_types):
        ax = axes_flat[i]
        
        if model_type not in model_data:
            ax.set_visible(False)
            continue
        
        df = model_data[model_type]
        obs = df['obs_Q'].values
        sim = df['sim_Q'].values
        
        color = multi_config['config_colors'].get(model_type, 'gray')
        name = multi_config['config_names'].get(model_type, model_type)
        
        # Calculate statistics
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(obs, sim)
        
        obs_mean = np.mean(obs)
        nse = 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - obs_mean) ** 2))
        
        std_sim = np.std(sim)
        std_obs = np.std(obs)
        mean_sim = np.mean(sim)
        mean_obs = np.mean(obs)
        corr = np.corrcoef(sim, obs)[0, 1]
        alpha = std_sim / std_obs
        beta = mean_sim / mean_obs
        kge = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        
        rmse = np.sqrt(np.mean((obs - sim)**2))
        bias = np.mean(sim - obs)
        
        all_stats[model_type] = {
            'r_squared': r_value**2,
            'nse': nse,
            'kge': kge,
            'rmse': rmse,
            'bias': bias,
            'n_points': len(df)
        }
        
        # Plot scatter
        ax.scatter(obs, sim, alpha=0.5, s=20, c=color, edgecolors='navy', linewidth=0.5)
        
        # Add 1:1 line
        ax.plot([global_min, global_max], [global_min, global_max], 'k--', 
               linewidth=2, label='1:1 Line', zorder=10)
        
        # Add regression line
        line_x = np.array([global_min, global_max])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, 'r-', linewidth=2, 
               label=f'Regression (R²={r_value**2:.3f})', zorder=9)
        
        # Formatting
        ax.set_xlabel('Observed (m³/s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Simulated (m³/s)', fontsize=11, fontweight='bold')
        ax.set_title(f'{name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        ax.set_aspect('equal')
        
        # Add statistics text
        stats_text = (f"NSE={nse:.3f}\nKGE={kge:.3f}\n"
                     f"RMSE={rmse:.2f}\nn={len(df)}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # Hide unused subplots
    for i in range(len(model_types), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Overall title
    fig.suptitle(f'Multi-Model Scatter Plot Comparison - Catchment {gauge_id}\n'
                f'Validation Period: {validation_start} to {validation_end}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    save_path = plot_dirs['performance'] / f'multi_model_scatter_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved multi-model scatter plot to: {save_path}")
    plt.show()
    
    # Print comparison
    print(f"\nModel Performance Comparison:")
    print(f"{'Model':<15} {'R²':<8} {'NSE':<8} {'KGE':<8} {'RMSE':<10} {'n':<8}")
    print(f"{'-'*65}")
    for model_type, stats in all_stats.items():
        name = multi_config['config_names'].get(model_type, model_type)
        print(f"{name:<15} {stats['r_squared']:<8.3f} {stats['nse']:<8.3f} "
              f"{stats['kge']:<8.3f} {stats['rmse']:<10.2f} {stats['n_points']:<8}")
    
    return {
        'statistics': all_stats,
        'save_path': save_path
    }

#--------------------------------------------------------------------------------

def plot_multi_model_residuals(multi_config, plot_dirs, validation_start=None, validation_end=None):
    """
    Create residual plots for multiple models on the same figure.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-model configuration dictionary
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    dict
        Dictionary containing residual statistics for all models
    """
    
    gauge_id = multi_config['gauge_id']
    model_types = multi_config['model_type']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    print(f"Creating multi-model residual plot comparison:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Period: {validation_start} to {validation_end}")
    print(f"  - Models: {', '.join(model_types)}")
    
    start_date = pd.to_datetime(validation_start)
    end_date = pd.to_datetime(validation_end)
    
    # Store results for each model
    all_residual_stats = {}
    
    # Calculate grid dimensions
    n_models = len(model_types)
    n_cols = min(2, n_models)
    n_rows = int(np.ceil(n_models / n_cols))
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 7*n_rows), squeeze=False)
    axes_flat = axes.flatten()
    
    # Process each model
    for i, model_type in enumerate(model_types):
        ax = axes_flat[i]
        
        config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': multi_config['config'],
            'gauge_id': gauge_id,
            'model_type': model_type,
            'start_date': multi_config['start_date'],
            'end_date': multi_config['end_date'],
            'cali_end_date': multi_config['cali_end_date'],
            'coupled': multi_config['config_coupled'].get(model_type, False)
        }
        
        data = postprocessing_single.load_hydrograph_data(config)
        if data is None:
            ax.set_visible(False)
            continue
        
        mask = (data['date'] >= start_date) & (data['date'] <= end_date)
        df = data[mask].copy()
        df = df.dropna(subset=['obs_Q', 'sim_Q'])
        
        if len(df) == 0:
            ax.set_visible(False)
            continue
        
        # Calculate residuals
        df['residual'] = df['sim_Q'] - df['obs_Q']
        obs = df['obs_Q'].values
        residuals = df['residual'].values
        
        color = multi_config['config_colors'].get(model_type, 'gray')
        name = multi_config['config_names'].get(model_type, model_type)
        
        # Calculate statistics
        bias = np.mean(residuals)
        std_residual = np.std(residuals)
        within_2sigma = np.sum(np.abs(residuals) <= 2*std_residual)
        pct_within_2sigma = (within_2sigma / len(residuals)) * 100
        
        all_residual_stats[model_type] = {
            'mean': bias,
            'std': std_residual,
            'min': residuals.min(),
            'max': residuals.max(),
            'pct_within_2sigma': pct_within_2sigma,
            'n_points': len(df)
        }
        
        # Plot residuals
        ax.scatter(obs, residuals, alpha=0.5, s=20, c=color, 
                  edgecolors='darkred', linewidth=0.5)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=2, 
                  alpha=0.7, label='Zero residual')
        
        # Add mean bias line
        ax.axhline(y=bias, color='red', linestyle='-', linewidth=2, 
                  alpha=0.7, label=f'Mean bias={bias:+.2f}')
        
        # Add ±2σ lines
        ax.axhline(y=2*std_residual, color='red', linestyle=':', 
                  linewidth=1.5, alpha=0.5, label=f'±2σ=±{2*std_residual:.2f}')
        ax.axhline(y=-2*std_residual, color='red', linestyle=':', 
                  linewidth=1.5, alpha=0.5)
        
        # Formatting
        ax.set_xlabel('Observed (m³/s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residual (Sim - Obs) (m³/s)', fontsize=11, fontweight='bold')
        ax.set_title(f'{name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Add statistics text
        stats_text = (f"Mean: {bias:+.3f}\nStd: {std_residual:.3f}\n"
                     f"±2σ: {pct_within_2sigma:.1f}%\nn={len(df)}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # Hide unused subplots
    for i in range(len(model_types), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Overall title
    fig.suptitle(f'Multi-Model Residual Plot Comparison - Catchment {gauge_id}\n'
                f'Validation Period: {validation_start} to {validation_end}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    save_path = plot_dirs['performance'] / f'multi_model_residuals_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved multi-model residual plot to: {save_path}")
    plt.show()
    
    # Print comparison
    print(f"\nModel Residual Comparison:")
    print(f"{'Model':<15} {'Mean':<10} {'Std':<10} {'Within ±2σ':<12} {'n':<8}")
    print(f"{'-'*65}")
    for model_type, stats in all_residual_stats.items():
        name = multi_config['config_names'].get(model_type, model_type)
        print(f"{name:<15} {stats['mean']:+<10.3f} {stats['std']:<10.3f} "
              f"{stats['pct_within_2sigma']:<12.1f}% {stats['n_points']:<8}")
    
    return {
        'residual_statistics': all_residual_stats,
        'save_path': save_path
    }

#--------------------------------------------------------------------------------
################################### SWE #########################################
#--------------------------------------------------------------------------------

def plot_multi_model_area_weighted_swe_timeseries(multi_config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot catchment-average SWE time series for multiple models on the same plot.
    Uses the NO_GLACIER HRU group which contains catchment-average SWE.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-model configuration dictionary
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    dict
        Dictionary containing SWE data for all models
    """
    
    gauge_id = multi_config['gauge_id']
    model_types = multi_config['model_type']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    validation_start = pd.to_datetime(validation_start)
    validation_end = pd.to_datetime(validation_end)
    
    print(f"Creating multi-model area-weighted SWE comparison:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Period: {validation_start.date()} to {validation_end.date()}")
    print(f"  - Models: {', '.join(model_types)}")
    
    # Store results for each model
    all_swe_data = {}
    
    # Load data for each model
    for model_type in model_types:
        print(f"\n  Loading {model_type} SWE data...")
        
        # Create single-model config
        config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': multi_config['config'],
            'gauge_id': gauge_id,
            'model_type': model_type,
            'start_date': multi_config['start_date'],
            'end_date': multi_config['end_date'],
            'cali_end_date': multi_config['cali_end_date'],
            'coupled': multi_config['config_coupled'].get(model_type, False)
        }
        
        # Load SWE data using the comprehensive loader
        swe_data = postprocessing_single.load_swe_data(config)
        
        if swe_data is None:
            print(f"    ✗ No SWE data found for {model_type}")
            continue
        
        if swe_data['catchment_avg'] is None:
            print(f"    ✗ NO_GLACIER column not available for {model_type}")
            continue
        
        # Extract the full dataframe
        df = swe_data['full_data']
        
        # Filter for validation period
        mask = (df['date'] >= validation_start) & (df['date'] <= validation_end)
        df_filtered = df[mask].copy()
        
        if len(df_filtered) == 0:
            print(f"    ✗ No data in validation period for {model_type}")
            continue
        
        # Store catchment average SWE
        all_swe_data[model_type] = df_filtered[['date', 'NO_GLACIER']].copy()
        all_swe_data[model_type].columns = ['date', 'catchment_avg_swe']
        
        print(f"    ✓ Loaded {model_type} SWE data ({len(df_filtered)} records)")
        print(f"      Mean SWE: {all_swe_data[model_type]['catchment_avg_swe'].mean():.1f} mm")
        print(f"      Max SWE: {all_swe_data[model_type]['catchment_avg_swe'].max():.1f} mm")
    
    if len(all_swe_data) == 0:
        print("\n✗ ERROR: No valid SWE data found for any model")
        return None
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot each model's SWE time series
    for model_type, swe_df in all_swe_data.items():
        color = multi_config['config_colors'].get(model_type, 'gray')
        name = multi_config['config_names'].get(model_type, model_type)
        
        plt.plot(swe_df['date'], swe_df['catchment_avg_swe'], 
                color=color, linewidth=2, label=name, alpha=0.8)
    
    # Formatting
    plt.xlabel('Date', fontsize=14, fontweight='bold')
    plt.ylabel('Snow Water Equivalent (mm)', fontsize=14, fontweight='bold')
    plt.title(f'Multi-Model Catchment-Average SWE Comparison - Catchment {gauge_id}\n'
             f'Period: {validation_start.date()} to {validation_end.date()}', 
             fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    
    # Format x-axis dates
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gcf().autofmt_xdate()
    
    # Add summary statistics text box
    stats_lines = []
    for model_type, swe_df in all_swe_data.items():
        name = multi_config['config_names'].get(model_type, model_type)
        mean_swe = swe_df['catchment_avg_swe'].mean()
        max_swe = swe_df['catchment_avg_swe'].max()
        stats_lines.append(f"{name}: Mean={mean_swe:.1f} mm, Max={max_swe:.1f} mm")
    
    stats_text = '\n'.join(stats_lines)
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    save_path = plot_dirs['swe'] / f'multi_model_area_weighted_swe_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved multi-model SWE comparison to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\nMulti-Model SWE Summary:")
    print(f"  Period: {validation_start.date()} to {validation_end.date()}")
    
    for model_type, swe_df in all_swe_data.items():
        name = multi_config['config_names'].get(model_type, model_type)
        mean_swe = swe_df['catchment_avg_swe'].mean()
        max_swe = swe_df['catchment_avg_swe'].max()
        max_date = swe_df.loc[swe_df['catchment_avg_swe'].idxmax(), 'date']
        
        print(f"\n  {name}:")
        print(f"    Mean SWE: {mean_swe:.2f} mm")
        print(f"    Max SWE: {max_swe:.2f} mm")
        print(f"    Max SWE date: {max_date.date()}")
    
    return {
        'swe_data': all_swe_data,
        'save_path': save_path
    }

#--------------------------------------------------------------------------------

def plot_multi_model_swe_by_elevation(multi_config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot SWE time series by elevation bands for multiple models.
    Creates separate subplots for each model to show elevation band distribution.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-model configuration dictionary
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    dict
        Dictionary containing elevation band SWE data for all models
    """
    
    gauge_id = multi_config['gauge_id']
    model_types = multi_config['model_type']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    print(f"Creating multi-model SWE by elevation comparison:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Period: {validation_start} to {validation_end}")
    print(f"  - Models: {', '.join(model_types)}")
    
    # Calculate grid dimensions for subplots
    n_models = len(model_types)
    n_cols = min(2, n_models)  # Max 2 columns
    n_rows = int(np.ceil(n_models / n_cols))
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14*n_cols, 8*n_rows), squeeze=False)
    axes_flat = axes.flatten()
    
    # Store results for each model
    all_elevation_data = {}
    
    # Process each model
    for i, model_type in enumerate(model_types):
        ax = axes_flat[i]
        
        print(f"\n  Processing {model_type}...")
        
        # Create single-model config
        config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': multi_config['config'],
            'gauge_id': gauge_id,
            'model_type': model_type,
            'start_date': multi_config['start_date'],
            'end_date': multi_config['end_date'],
            'cali_end_date': multi_config['cali_end_date'],
            'coupled': multi_config['config_coupled'].get(model_type, False)
        }
        
        # Load SWE data
        config_dir = Path(multi_config['main_dir']) / multi_config['config']
        sim_file = config_dir / f"catchment_{gauge_id}" / model_type / "output" / f"{gauge_id}_{model_type}_SNOW_Daily_Average_ByHRUGroup.csv"
        
        if not sim_file.exists():
            print(f"    ✗ SWE file not found for {model_type}")
            ax.set_visible(False)
            continue
        
        try:
            # Read the file, skipping the units row
            sim_data = pd.read_csv(sim_file, skiprows=[1])
            
            # Create date from config start date
            start_date_config = pd.to_datetime(config.get('start_date', '2000-01-01'))
            sim_data['date'] = pd.date_range(start=start_date_config, periods=len(sim_data), freq='D')
            
            # Get elevation band columns (exclude 'AllHRUs', 'date', etc.)
            date_col = sim_data.columns[0]
            exclude_cols = ['date', 'day', 'time', 'AllHRUs', 'HRUGroup:', date_col, 'NO_GLACIER']
            if 'Unnamed' in str(date_col):
                exclude_cols.append(date_col)
            
            elev_bands = [col for col in sim_data.columns if col not in exclude_cols and '-' in col and 'm' in col]
            
            if len(elev_bands) == 0:
                print(f"    ✗ No elevation bands found for {model_type}")
                ax.set_visible(False)
                continue
            
            # Filter for validation period
            start_date_val = pd.to_datetime(validation_start)
            end_date_val = pd.to_datetime(validation_end)
            
            sim_mask = (sim_data['date'] >= start_date_val) & (sim_data['date'] <= end_date_val)
            sim_filtered = sim_data[sim_mask].copy()
            
            if len(sim_filtered) == 0:
                print(f"    ✗ No data in validation period for {model_type}")
                ax.set_visible(False)
                continue
            
            # Sort elevation bands
            elev_bands_sorted = sorted(elev_bands, key=lambda x: int(x.split('-')[0]))
            
            # Store data
            all_elevation_data[model_type] = {
                'data': sim_filtered,
                'elevation_bands': elev_bands_sorted
            }
            
            # Plot elevation bands for this model
            for band in elev_bands_sorted:
                # Convert from m to mm if needed
                if sim_filtered[band].mean() < 10:  # Likely in meters
                    band_data = sim_filtered[band] * 1000
                else:
                    band_data = sim_filtered[band]
                
                ax.plot(sim_filtered['date'], band_data, linewidth=1.5, label=band, alpha=0.7)
            
            # Formatting for this subplot
            color = multi_config['config_colors'].get(model_type, 'gray')
            name = multi_config['config_names'].get(model_type, model_type)
            
            ax.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax.set_ylabel('SWE (mm)', fontsize=11, fontweight='bold')
            ax.set_title(f'{name} - SWE by Elevation Band', fontsize=13, fontweight='bold', 
                        color=color)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best', ncol=2)
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            print(f"    ✓ Plotted {len(elev_bands_sorted)} elevation bands for {model_type}")
            
        except Exception as e:
            print(f"    ✗ Error processing {model_type}: {e}")
            ax.set_visible(False)
            continue
    
    # Hide unused subplots
    for i in range(len(model_types), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Overall title
    fig.suptitle(f'Multi-Model SWE by Elevation Bands - Catchment {gauge_id}\n'
                f'Period: {validation_start} to {validation_end}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    save_path = plot_dirs['swe'] / f'multi_model_swe_by_elevation_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved multi-model SWE by elevation plot to: {save_path}")
    plt.show()
    
    # Print summary
    print(f"\nMulti-Model SWE by Elevation Summary:")
    print(f"  Period: {validation_start} to {validation_end}")
    
    for model_type, data_dict in all_elevation_data.items():
        name = multi_config['config_names'].get(model_type, model_type)
        n_bands = len(data_dict['elevation_bands'])
        
        print(f"\n  {name}:")
        print(f"    Elevation bands: {n_bands}")
        print(f"    Range: {data_dict['elevation_bands'][0]} to {data_dict['elevation_bands'][-1]}")
    
    return {
        'elevation_data': all_elevation_data,
        'save_path': save_path
    }

#--------------------------------------------------------------------------------
################################# contributions #################################
#--------------------------------------------------------------------------------

def plot_multi_model_glogem_icemelt_snowmelt_regime(multi_config, plot_dirs, validation_start=None, validation_end=None, unit='mm'):
    """
    Plot streamflow regime with GloGEM ice melt and total snowmelt for multiple models.
    Creates separate panels for each model for better visibility.
    
    Creates a comparison showing:
    - Observed streamflow (if available)
    - Simulated streamflow for each model (in separate panels)
    - GloGEM ice melt (same for all models if coupled)
    - Total snowmelt (GloGEM + HBV) for each model
    
    Parameters:
    -----------
    multi_config : dict
        Multi-model configuration dictionary
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
        Dictionary containing regime data for all models and components
    """
    
    gauge_id = multi_config['gauge_id']
    model_types = multi_config['model_type']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    print(f"Creating multi-model GloGEM + snowmelt regime comparison:")
    print(f"  - Catchment: {gauge_id}")
    print(f"  - Period: {validation_start} to {validation_end}")
    print(f"  - Unit: {unit}")
    print(f"  - Models: {', '.join(model_types)}")
    
    # Load catchment area for unit conversion
    config_dir = Path(multi_config['main_dir']) / multi_config['config']
    topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
    catchment_shape_file = topo_dir / "HRU.shp"
    
    conversion_m3s_to_mm_day = None
    if unit == 'mm':
        try:
            import geopandas as gpd
            if catchment_shape_file.exists():
                hru_gdf = gpd.read_file(catchment_shape_file)
                total_area_km2 = hru_gdf['Area_km2'].sum()
                conversion_m3s_to_mm_day = 86400 / (total_area_km2 * 1000000) * 1000
                print(f"  - Catchment area: {total_area_km2:.2f} km²")
        except Exception as e:
            print(f"  ⚠️  Could not load catchment area: {e}")
    
    unit_label = 'mm/day' if unit == 'mm' and conversion_m3s_to_mm_day is not None else 'm³/s'
    
    # =============================
    # 1. LOAD OBSERVED STREAMFLOW (once, shared across all models)
    # =============================
    
    obs_regime = None
    
    # Try to load observed data from the first model
    first_model = model_types[0]
    config_first = {
        'main_dir': multi_config['main_dir'],
        'config_dir': multi_config['config'],
        'gauge_id': gauge_id,
        'model_type': first_model,
        'start_date': multi_config['start_date'],
        'end_date': multi_config['end_date'],
        'cali_end_date': multi_config['cali_end_date'],
        'coupled': multi_config['config_coupled'].get(first_model, False)
    }
    
    streamflow_data = postprocessing_single.load_hydrograph_data(config_first)
    
    if streamflow_data is not None:
        start_date = pd.to_datetime(validation_start)
        end_date = pd.to_datetime(validation_end)
        
        streamflow_mask = (streamflow_data['date'] >= start_date) & (streamflow_data['date'] <= end_date)
        streamflow_filtered = streamflow_data[streamflow_mask].copy()
        
        if len(streamflow_filtered) > 0 and 'obs_Q' in streamflow_filtered.columns:
            # Convert to requested units
            if unit == 'mm' and conversion_m3s_to_mm_day is not None:
                streamflow_filtered['obs_Q_converted'] = streamflow_filtered['obs_Q'] * conversion_m3s_to_mm_day
            else:
                streamflow_filtered['obs_Q_converted'] = streamflow_filtered['obs_Q']
            
            # Calculate monthly regime
            streamflow_filtered['month'] = streamflow_filtered['date'].dt.month
            obs_regime = streamflow_filtered.groupby('month')['obs_Q_converted'].mean()
            print(f"  ✓ Loaded observed streamflow regime (mean: {obs_regime.mean():.4f} {unit_label})")
    
    # =============================
    # 2. LOAD GLOGEM DATA (once, shared across all coupled models)
    # =============================
    
    glogem_icemelt_regime = None
    glogem_snowmelt_regime = None
    
    # Check if any model is coupled
    if any(multi_config['config_coupled'].get(model, False) for model in model_types):
        print(f"\n  Loading GloGEM data (shared across coupled models)...")
        
        glogem_df = postprocessing_single.load_glogem_data(config_first, unit='mm', plot=False)
        
        if glogem_df is not None:
            # Filter for validation period
            glogem_mask = (glogem_df['date'] >= start_date) & (glogem_df['date'] <= end_date)
            glogem_filtered = glogem_df[glogem_mask].copy()
            
            if len(glogem_filtered) > 0:
                # Calculate monthly regime for GloGEM components (normalized)
                glogem_filtered['month'] = glogem_filtered['date'].dt.month
                
                glogem_icemelt_regime = glogem_filtered.groupby('month')['icemelt_normalized'].mean()
                glogem_snowmelt_regime = glogem_filtered.groupby('month')['snowmelt_normalized'].mean()
                
                print(f"  ✓ GloGEM ice melt mean: {glogem_icemelt_regime.mean():.4f} mm/day")
                print(f"  ✓ GloGEM snowmelt mean: {glogem_snowmelt_regime.mean():.4f} mm/day")
    
    # =============================
    # 3. LOAD MODEL-SPECIFIC DATA
    # =============================
    
    model_streamflow_regimes = {}
    model_snowmelt_regimes = {}
    
    for model_type in model_types:
        print(f"\n  Loading {model_type} data...")
        
        config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': multi_config['config'],
            'gauge_id': gauge_id,
            'model_type': model_type,
            'start_date': multi_config['start_date'],
            'end_date': multi_config['end_date'],
            'cali_end_date': multi_config['cali_end_date'],
            'coupled': multi_config['config_coupled'].get(model_type, False)
        }
        
        # Load simulated streamflow
        streamflow_data = postprocessing_single.load_hydrograph_data(config)
        
        if streamflow_data is not None:
            streamflow_mask = (streamflow_data['date'] >= start_date) & (streamflow_data['date'] <= end_date)
            streamflow_filtered = streamflow_data[streamflow_mask].copy()
            
            if len(streamflow_filtered) > 0 and 'sim_Q' in streamflow_filtered.columns:
                # Convert to requested units
                if unit == 'mm' and conversion_m3s_to_mm_day is not None:
                    streamflow_filtered['sim_Q_converted'] = streamflow_filtered['sim_Q'] * conversion_m3s_to_mm_day
                else:
                    streamflow_filtered['sim_Q_converted'] = streamflow_filtered['sim_Q']
                
                # Calculate monthly regime
                streamflow_filtered['month'] = streamflow_filtered['date'].dt.month
                sim_regime = streamflow_filtered.groupby('month')['sim_Q_converted'].mean()
                
                model_streamflow_regimes[model_type] = sim_regime
                print(f"    ✓ {model_type} streamflow mean: {sim_regime.mean():.4f} {unit_label}")
        
        # Load HBV snowmelt mass loadings
        snowmelt_df = postprocessing_single.load_snowmelt_mass_loadings(config, validation_start, validation_end, unit=unit)
        
        if snowmelt_df is not None:
            snowmelt_col = 'snowmelt_mm_day' if unit == 'mm' and 'snowmelt_mm_day' in snowmelt_df.columns else 'snowmelt_m3s'
            
            snowmelt_df['month'] = snowmelt_df['date'].dt.month
            hbv_snowmelt_regime = snowmelt_df.groupby('month')[snowmelt_col].mean()
            
            # Combine with GloGEM snowmelt if coupled
            if multi_config['config_coupled'].get(model_type, False) and glogem_snowmelt_regime is not None:
                total_snowmelt_regime = glogem_snowmelt_regime.add(hbv_snowmelt_regime, fill_value=0)
                print(f"    ✓ {model_type} total snowmelt mean: {total_snowmelt_regime.mean():.4f} {unit_label}")
                print(f"      (GloGEM: {glogem_snowmelt_regime.mean():.4f} + HBV: {hbv_snowmelt_regime.mean():.4f})")
            else:
                total_snowmelt_regime = hbv_snowmelt_regime
                print(f"    ✓ {model_type} HBV snowmelt mean: {hbv_snowmelt_regime.mean():.4f} {unit_label}")
            
            model_snowmelt_regimes[model_type] = total_snowmelt_regime
    
    # =============================
    # 4. CREATE MULTI-PANEL PLOT
    # =============================
    
    if len(model_streamflow_regimes) == 0:
        print("\n✗ ERROR: No valid model data found for plotting")
        return None
    
    # Calculate grid dimensions
    n_models = len(model_types)
    n_cols = min(2, n_models)  # Max 2 columns
    n_rows = int(np.ceil(n_models / n_cols))
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10*n_cols, 7*n_rows), squeeze=False)
    axes_flat = axes.flatten()
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Consistent colors for melt components (same for all models)
    snowmelt_color = '#B3D9FF'  # Light blue for snowmelt
    icemelt_color = '#C17817'   # Orange/brown for ice melt
    
    # Plot each model in its own panel
    for i, model_type in enumerate(model_types):
        ax = axes_flat[i]
        
        if model_type not in model_streamflow_regimes:
            ax.set_visible(False)
            continue
        
        # Get model-specific data
        sim_regime = model_streamflow_regimes[model_type]
        snowmelt_regime = model_snowmelt_regimes.get(model_type)
        
        color = multi_config['config_colors'].get(model_type, 'gray')
        name = multi_config['config_names'].get(model_type, model_type)
        
        # Plot filled areas first (bottom layer)
        # Plot total snowmelt as filled polygon (same color for all models)
        if snowmelt_regime is not None:
            ax.fill_between(months, 0, snowmelt_regime.values, 
                           color=snowmelt_color, alpha=0.7, 
                           label='Total Snowmelt (GloGEM+HBV)' if multi_config['config_coupled'].get(model_type, False) else 'Snowmelt (HBV)', 
                           zorder=1, edgecolor='#6DB3F2', linewidth=1.5)
        
        # Plot GloGEM ice melt as filled polygon (if coupled, same color for all models)
        if multi_config['config_coupled'].get(model_type, False) and glogem_icemelt_regime is not None:
            ax.fill_between(months, 0, glogem_icemelt_regime.values, 
                           color=icemelt_color, alpha=0.6, label='GloGEM Ice Melt', 
                           zorder=2, edgecolor='#8B5A00', linewidth=1.5)
        
        # Plot observed streamflow (if available)
        if obs_regime is not None:
            ax.plot(months, obs_regime.values, 'k-', linewidth=3.5, 
                   label='Observed', marker='o', markersize=8, zorder=10)
        
        # Plot simulated streamflow for this model (model-specific color)
        ax.plot(months, sim_regime.values, color=color, linewidth=3, 
               label=f'{name} Simulated', marker='s', markersize=7, 
               alpha=0.8, linestyle='--', zorder=9)
        
        # Formatting - BIGGER FONTS
        ax.set_xlabel('Month', fontsize=16, fontweight='bold')
        ax.set_ylabel(f'Discharge ({unit_label})', fontsize=16, fontweight='bold')
        ax.set_title(f'{name}', fontsize=18, fontweight='bold')
        ax.set_xticks(months)
        ax.set_xticklabels(month_names, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        ax.legend(fontsize=13, loc='best')
        
        # Add statistics text box - BIGGER FONT
        stats_lines = []
        
        if obs_regime is not None:
            stats_lines.append(f"Obs: {obs_regime.mean():.4f} {unit_label}")
        
        stats_lines.append(f"Sim: {sim_regime.mean():.4f} {unit_label}")
        
        if snowmelt_regime is not None:
            stats_lines.append(f"Snowmelt: {snowmelt_regime.mean():.4f} {unit_label}")
        
        if multi_config['config_coupled'].get(model_type, False) and glogem_icemelt_regime is not None:
            stats_lines.append(f"Ice: {glogem_icemelt_regime.mean():.4f} {unit_label}")
        
        stats_text = '\n'.join(stats_lines)
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    # Hide unused subplots
    for i in range(len(model_types), len(axes_flat)):
        axes_flat[i].set_visible(False)

    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    save_path = plot_dirs['contributions'] / f'multi_model_glogem_icemelt_snowmelt_regime_{unit}_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved multi-model GloGEM + snowmelt regime plot to: {save_path}")
    plt.show()
    
    # =============================
    # 5. PRINT SUMMARY
    # =============================
    
    print(f"\n{'='*60}")
    print(f"MULTI-MODEL GLOGEM + SNOWMELT REGIME SUMMARY")
    print(f"{'='*60}")
    print(f"Catchment: {gauge_id}")
    print(f"Period: {validation_start} to {validation_end}")
    print(f"Unit: {unit_label}")
    print(f"Models: {', '.join(model_types)}")
    
    if obs_regime is not None:
        print(f"\nObserved Streamflow:")
        print(f"  Mean: {obs_regime.mean():.4f} {unit_label}")
        print(f"  Peak: {month_names[obs_regime.idxmax()-1]} ({obs_regime.max():.4f} {unit_label})")
    
    print(f"\nSimulated Streamflow:")
    for model_type, sim_regime in model_streamflow_regimes.items():
        name = multi_config['config_names'].get(model_type, model_type)
        print(f"  {name}:")
        print(f"    Mean: {sim_regime.mean():.4f} {unit_label}")
        print(f"    Peak: {month_names[sim_regime.idxmax()-1]} ({sim_regime.max():.4f} {unit_label})")
    
    if glogem_icemelt_regime is not None:
        print(f"\nGloGEM Ice Melt:")
        print(f"  Mean: {glogem_icemelt_regime.mean():.4f} {unit_label}")
        print(f"  Peak: {month_names[glogem_icemelt_regime.idxmax()-1]} ({glogem_icemelt_regime.max():.4f} {unit_label})")
    
    print(f"\nTotal Snowmelt (GloGEM + HBV):")
    for model_type, snowmelt_regime in model_snowmelt_regimes.items():
        name = multi_config['config_names'].get(model_type, model_type)
        coupled = multi_config['config_coupled'].get(model_type, False)
        print(f"  {name} ({'Coupled' if coupled else 'HBV only'}):")
        print(f"    Mean: {snowmelt_regime.mean():.4f} {unit_label}")
        print(f"    Peak: {month_names[snowmelt_regime.idxmax()-1]} ({snowmelt_regime.max():.4f} {unit_label})")
    
    print(f"{'='*60}\n")
    
    # Return all data
    return {
        'observed': obs_regime,
        'streamflow': model_streamflow_regimes,
        'glogem_icemelt': glogem_icemelt_regime,
        'snowmelt': model_snowmelt_regimes,
        'unit': unit_label,
        'save_path': save_path
    }

#--------------------------------------------------------------------------------
##################################### Storage ###################################
#--------------------------------------------------------------------------------

def plot_multi_model_storage_timeseries(multi_config, plot_dirs, validation_start=None, validation_end=None):
    """
    Plot time series of watershed storage components for multiple models.
    Creates one panel per storage type with a line for each model.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-model configuration dictionary
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    dict
        Dictionary containing storage data for all models
    """
    
    gauge_id = multi_config['gauge_id']
    model_types = multi_config['model_type']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    validation_start = pd.to_datetime(validation_start)
    validation_end = pd.to_datetime(validation_end)
    
    print(f"Creating multi-model storage time series for catchment {gauge_id}:")
    print(f"  - Period: {validation_start.date()} to {validation_end.date()}")
    print(f"  - Models: {', '.join(model_types)}")
    
    # =============================
    # 1. LOAD STORAGE DATA FOR ALL MODELS
    # =============================
    
    all_storage_data = {}
    common_storage_cols = None
    
    for model_type in model_types:
        print(f"\n  Loading {model_type} storage data...")
        
        # Create single-model config
        config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': multi_config['config'],
            'gauge_id': gauge_id,
            'model_type': model_type,
            'start_date': multi_config['start_date'],
            'end_date': multi_config['end_date'],
            'cali_end_date': multi_config['cali_end_date'],
            'coupled': multi_config['config_coupled'].get(model_type, False)
        }
        
        # Load storage data
        storage_df = postprocessing_single.load_storage_data(config)
        
        if storage_df is None:
            print(f"    ✗ No storage data found for {model_type}")
            continue
        
        # Filter for validation period
        mask = (storage_df['date'] >= validation_start) & (storage_df['date'] <= validation_end)
        storage_filtered = storage_df[mask].copy()
        
        if len(storage_filtered) == 0:
            print(f"    ✗ No data in validation period for {model_type}")
            continue
        
        # Get storage columns (exclude date, month, year)
        storage_cols = [col for col in storage_filtered.columns if col not in ['date', 'month', 'year']]
        
        if len(storage_cols) == 0:
            print(f"    ✗ No storage columns found for {model_type}")
            continue
        
        # Store data
        all_storage_data[model_type] = {
            'data': storage_filtered,
            'storage_cols': storage_cols
        }
        
        print(f"    ✓ Loaded {len(storage_filtered)} records with {len(storage_cols)} storage types")
        
        # Find common storage columns across all models
        if common_storage_cols is None:
            common_storage_cols = set(storage_cols)
        else:
            common_storage_cols = common_storage_cols.intersection(set(storage_cols))
    
    if len(all_storage_data) == 0:
        print("\n✗ ERROR: No valid storage data found for any model")
        return None
    
    # Convert common columns to sorted list
    common_storage_cols = sorted(list(common_storage_cols))
    
    if len(common_storage_cols) == 0:
        print("\n✗ ERROR: No common storage columns found across models")
        print("  Available columns per model:")
        for model_type, data_dict in all_storage_data.items():
            print(f"    {model_type}: {data_dict['storage_cols']}")
        return None
    
    print(f"\n  Found {len(common_storage_cols)} common storage types across all models:")
    for col in common_storage_cols:
        print(f"    - {col}")
    
    # =============================
    # 2. CREATE MULTI-PANEL PLOT
    # =============================
    
    n_storage_types = len(common_storage_cols)
    
    # Create subplots - one row per storage type
    fig, axes = plt.subplots(n_storage_types, 1, figsize=(16, 4*n_storage_types), sharex=True)
    
    # Make axes iterable if there's only one
    if n_storage_types == 1:
        axes = [axes]
    
    # Plot each storage type
    for i, storage_col in enumerate(common_storage_cols):
        ax = axes[i]
        
        # Plot each model's data for this storage type
        for model_type, data_dict in all_storage_data.items():
            storage_filtered = data_dict['data']
            
            if storage_col not in storage_filtered.columns:
                continue
            
            color = multi_config['config_colors'].get(model_type, 'gray')
            name = multi_config['config_names'].get(model_type, model_type)
            
            # Plot time series
            ax.plot(storage_filtered['date'], storage_filtered[storage_col], 
                   color=color, linewidth=2, label=name, alpha=0.8)
        
        # Clean up column name for title
        clean_title = storage_col.replace('[mm]', '(mm)').replace('[mm/d]', '(mm/d)')
        ax.set_title(f'{clean_title}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Storage (mm)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='best')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    
    # Set x-label only on bottom subplot
    axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add overall title
    fig.suptitle(f'Multi-Model Storage Time Series - Catchment {gauge_id}\n'
                f'Period: {validation_start.date()} to {validation_end.date()}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.3)
    
    # Save plot
    save_path = plot_dirs['storage'] / f'multi_model_storage_timeseries_{gauge_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved multi-model storage time series to: {save_path}")
    plt.show()
    
    # =============================
    # 3. PRINT SUMMARY STATISTICS
    # =============================
    
    print(f"\n{'='*60}")
    print(f"MULTI-MODEL STORAGE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Catchment: {gauge_id}")
    print(f"Period: {validation_start.date()} to {validation_end.date()}")
    print(f"Models: {', '.join(all_storage_data.keys())}")
    print(f"Common storage types: {len(common_storage_cols)}")
    
    for storage_col in common_storage_cols:
        print(f"\n  {storage_col}:")
        
        for model_type, data_dict in all_storage_data.items():
            storage_filtered = data_dict['data']
            
            if storage_col not in storage_filtered.columns:
                continue
            
            name = multi_config['config_names'].get(model_type, model_type)
            mean_val = storage_filtered[storage_col].mean()
            max_val = storage_filtered[storage_col].max()
            min_val = storage_filtered[storage_col].min()
            
            print(f"    {name:10s}: Mean={mean_val:7.1f} mm, Min={min_val:7.1f} mm, Max={max_val:7.1f} mm")
    
    print(f"{'='*60}\n")
    
    return {
        'storage_data': all_storage_data,
        'common_storage_types': common_storage_cols,
        'save_path': save_path
    }

#--------------------------------------------------------------------------------
########################## Water Balance Comparison #############################
#--------------------------------------------------------------------------------

def plot_multi_model_comprehensive_annual_water_balance(multi_config, plot_dirs, validation_start=None, validation_end=None):
    """
    Create comprehensive annual water balance analysis for multiple models.
    Creates separate plots for each model showing all components.
    
    Components (same for all models):
    - Observed and simulated streamflow
    - Ice melt from GloGEM (if coupled)
    - Snowmelt from GloGEM + HBV model (if coupled) or HBV only
    - Rainfall from GloGEM + HBV model (if coupled) or HBV only
    
    All components are converted to consistent units (mm/year) for comparison.
    
    Parameters:
    -----------
    multi_config : dict
        Multi-model configuration dictionary
    plot_dirs : dict
        Dictionary containing plot directory paths
    validation_start : str, optional
        Start date for validation period
    validation_end : str, optional
        End date for validation period
        
    Returns:
    --------
    dict
        Dictionary containing annual balance data for all models
    """
    
    gauge_id = multi_config['gauge_id']
    model_types = multi_config['model_type']
    
    # Use dates from config if not provided
    if validation_start is None:
        validation_start = multi_config.get('cali_end_date', '2010-01-01')
    if validation_end is None:
        validation_end = multi_config.get('end_date', '2020-12-31')
    
    print(f"\n{'='*60}")
    print(f"MULTI-MODEL COMPREHENSIVE ANNUAL WATER BALANCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Catchment: {gauge_id}")
    print(f"Period: {validation_start} to {validation_end}")
    print(f"Models: {', '.join(model_types)}")
    print(f"{'='*60}\n")
    
    # Store results for all models
    all_model_results = {}
    
    # Process each model separately
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Processing {model_type}")
        print(f"{'='*60}")
        
        # Create single-model config
        config = {
            'main_dir': multi_config['main_dir'],
            'config_dir': multi_config['config'],
            'gauge_id': gauge_id,
            'model_type': model_type,
            'start_date': multi_config['start_date'],
            'end_date': multi_config['end_date'],
            'cali_end_date': multi_config['cali_end_date'],
            'coupled': multi_config['config_coupled'].get(model_type, False)
        }
        
        coupled = config['coupled']
        
        # =============================
        # 1. LOAD DATA FOR THIS MODEL
        # =============================
        
        # Load catchment area (same for all models)
        config_dir = Path(multi_config['main_dir']) / multi_config['config']
        topo_dir = config_dir / f"catchment_{gauge_id}" / "topo_files"
        catchment_shape_file = topo_dir / "HRU.shp"
        
        try:
            import geopandas as gpd
            if catchment_shape_file.exists():
                hru_gdf = gpd.read_file(catchment_shape_file)
                total_area_km2 = hru_gdf['Area_km2'].sum()
                conversion_m3s_to_mm_day = 86400 / (total_area_km2 * 1000000) * 1000
                print(f"  Catchment area: {total_area_km2:.2f} km²")
            else:
                print(f"  ERROR: Catchment shapefile not found")
                continue
        except Exception as e:
            print(f"  ERROR: Could not load catchment area: {e}")
            continue
        
        # Load streamflow data
        print(f"\n  Loading streamflow data...")
        streamflow_data = postprocessing_single.load_hydrograph_data(config)
        if streamflow_data is None:
            print(f"  ERROR: Could not load streamflow data for {model_type}")
            continue
        
        # Filter for validation period
        start_date = pd.to_datetime(validation_start)
        end_date = pd.to_datetime(validation_end)
        
        streamflow_mask = (streamflow_data['date'] >= start_date) & (streamflow_data['date'] <= end_date)
        streamflow_filtered = streamflow_data[streamflow_mask].copy()
        
        if len(streamflow_filtered) == 0:
            print(f"  ERROR: No streamflow data in validation period for {model_type}")
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
        
        print(f"  ✓ Loaded streamflow data: {len(streamflow_annual)} years")
        
        # Load GloGEM data if coupled
        glogem_annual = None
        if coupled:
            print(f"\n  Loading GloGEM data (coupled mode)...")
            glogem_df = postprocessing_single.load_glogem_data(config, unit='mm', plot=False)
            
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
                    print(f"  ✓ Loaded GloGEM data: {len(glogem_annual)} years")
        
        # Load HBV snowmelt
        print(f"\n  Loading HBV snowmelt data...")
        snowmelt_df = postprocessing_single.load_snowmelt_mass_loadings(config, validation_start, validation_end)
        
        snowmelt_annual = None
        if snowmelt_df is not None:
            snowmelt_df['year'] = snowmelt_df['date'].dt.year
            snowmelt_annual = snowmelt_df.groupby('year')['snowmelt_mm_day'].sum().reset_index()
            snowmelt_annual.columns = ['year', 'hbv_snowmelt_mm']
            print(f"  ✓ Loaded HBV snowmelt: {len(snowmelt_annual)} years")
        
        # Load HBV rainfall
        print(f"\n  Loading HBV rainfall data...")
        rainfall_df = postprocessing_single.load_rainfall_mass_loadings(config, validation_start, validation_end)
        
        rainfall_annual = None
        if rainfall_df is not None:
            rainfall_df['year'] = rainfall_df['date'].dt.year
            rainfall_annual = rainfall_df.groupby('year')['rainfall_mm_day'].sum().reset_index()
            rainfall_annual.columns = ['year', 'hbv_rainfall_mm']
            print(f"  ✓ Loaded HBV rainfall: {len(rainfall_annual)} years")
        
        # =============================
        # 2. COMBINE ALL DATA
        # =============================
        
        print(f"\n  Combining all data sources...")
        
        # Start with streamflow data
        annual_balance = streamflow_annual.copy()
        
        # Merge GloGEM data if available
        if glogem_annual is not None:
            annual_balance = annual_balance.merge(glogem_annual, on='year', how='left')
        
        # Merge HBV snowmelt if available
        if snowmelt_annual is not None:
            annual_balance = annual_balance.merge(snowmelt_annual, on='year', how='left')
        
        # Merge HBV rainfall if available
        if rainfall_annual is not None:
            annual_balance = annual_balance.merge(rainfall_annual, on='year', how='left')
        
        # Calculate combined components
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
        
        print(f"  ✓ Combined data for {len(annual_balance)} years")
        
        # Store results for this model
        all_model_results[model_type] = annual_balance
        
        # =============================
        # 3. CREATE PLOTS FOR THIS MODEL
        # =============================
        
        # Get model color and name
        color = multi_config['config_colors'].get(model_type, 'gray')
        name = multi_config['config_names'].get(model_type, model_type)
        
        # PLOT 1: Bar plot for this model
        print(f"\n  Creating bar plot for {model_type}...")
        
        fig, ax = plt.subplots(figsize=(max(14, len(annual_balance) * 1.2), 10))
        
        x = np.arange(len(annual_balance))
        width = 0.15
        
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
                       label='Sim. Streamflow', color=color, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Annual Sum (mm/year)', fontsize=14, fontweight='bold')
        ax.set_title(f'{name} - Annual Water Balance\nCatchment {gauge_id} ({"Coupled" if coupled else "Uncoupled"})', 
                    fontsize=16, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(year)) for year in annual_balance['year']], rotation=45)
        ax.legend(fontsize=11, loc='upper left', ncol=2)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        save_path_bars = plot_dirs['water_balance'] / f'{model_type}_annual_water_balance_bars_{gauge_id}.png'
        plt.savefig(save_path_bars, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved bar plot to: {save_path_bars}")
        plt.show()
        
        # PLOT 2: Stacked area plot for this model
        print(f"\n  Creating stacked area plot for {model_type}...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create complete year range with NaN for missing years
        year_min = annual_balance['year'].min()
        year_max = annual_balance['year'].max()
        all_years = np.arange(year_min, year_max + 1)
        
        annual_balance_complete = annual_balance.set_index('year').reindex(all_years).reset_index()
        annual_balance_complete.columns = ['year'] + list(annual_balance.columns[1:])
        
        # Stacked area plot
        ax.fill_between(annual_balance_complete['year'], 0, annual_balance_complete['total_rainfall_mm'], 
                        label='Rainfall', color='navy', alpha=0.6)
        
        ax.fill_between(annual_balance_complete['year'], annual_balance_complete['total_rainfall_mm'], 
                        annual_balance_complete['total_rainfall_mm'] + annual_balance_complete['total_snowmelt_mm'], 
                        label='Snowmelt', color='lightblue', alpha=0.6)
        
        ax.fill_between(annual_balance_complete['year'], 
                        annual_balance_complete['total_rainfall_mm'] + annual_balance_complete['total_snowmelt_mm'], 
                        annual_balance_complete['total_rainfall_mm'] + annual_balance_complete['total_snowmelt_mm'] + annual_balance_complete['total_icemelt_mm'], 
                        label='Ice Melt', color='grey', alpha=0.6)
        
        ax.plot(annual_balance_complete['year'], annual_balance_complete['obs_streamflow_mm'], 
               'k-', linewidth=3, label='Obs. Streamflow', zorder=10)
        
        ax.plot(annual_balance_complete['year'], annual_balance_complete['sim_streamflow_mm'], 
               color=color, linewidth=2.5, linestyle='--', label='Sim. Streamflow', zorder=9)
        
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Annual Sum (mm/year)', fontsize=14, fontweight='bold')
        ax.set_title(f'{name} - Annual Water Balance (Stacked)\nCatchment {gauge_id}', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_xlim(year_min - 0.5, year_max + 0.5)
        
        plt.tight_layout()
        
        save_path_stacked = plot_dirs['water_balance'] / f'{model_type}_annual_water_balance_stacked_{gauge_id}.png'
        plt.savefig(save_path_stacked, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved stacked area plot to: {save_path_stacked}")
        plt.show()
        
        # Print summary for this model
        print(f"\n  {model_type} Summary:")
        print(f"    Mean annual rainfall: {annual_balance['total_rainfall_mm'].mean():.1f} mm/year")
        print(f"    Mean annual snowmelt: {annual_balance['total_snowmelt_mm'].mean():.1f} mm/year")
        print(f"    Mean annual ice melt: {annual_balance['total_icemelt_mm'].mean():.1f} mm/year")
        print(f"    Mean annual input: {annual_balance['total_input_mm'].mean():.1f} mm/year")
        print(f"    Mean annual obs streamflow: {annual_balance['obs_streamflow_mm'].mean():.1f} mm/year")
        print(f"    Mean annual sim streamflow: {annual_balance['sim_streamflow_mm'].mean():.1f} mm/year")
    
    # =============================
    # 4. CREATE COMPARISON PLOT
    # =============================
    
    if len(all_model_results) > 1:
        print(f"\n{'='*60}")
        print(f"Creating multi-model comparison plot...")
        print(f"{'='*60}")
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Find common years across all models
        common_years = set(all_model_results[model_types[0]]['year'])
        for model_type in model_types[1:]:
            common_years = common_years.intersection(set(all_model_results[model_type]['year']))
        
        common_years = sorted(list(common_years))
        
        if len(common_years) > 0:
            print(f"  Found {len(common_years)} common years across all models")
            
            x = np.arange(len(common_years))
            width = 0.8 / len(model_types)
            
            # Plot observed streamflow once
            obs_data = all_model_results[model_types[0]][all_model_results[model_types[0]]['year'].isin(common_years)]
            ax.bar(x - width*(len(model_types)/2 + 0.5), obs_data['obs_streamflow_mm'], width, 
                   label='Observed Streamflow', color='black', alpha=0.8, edgecolor='white', linewidth=1)
            
            # Plot simulated streamflow for each model
            for i, model_type in enumerate(model_types):
                model_data = all_model_results[model_type][all_model_results[model_type]['year'].isin(common_years)]
                color = multi_config['config_colors'].get(model_type, 'gray')
                name = multi_config['config_names'].get(model_type, model_type)
                
                offset = - width*(len(model_types)/2 - 0.5) + i*width
                ax.bar(x + offset, model_data['sim_streamflow_mm'], width, 
                       label=f'{name} Simulated', color=color, alpha=0.8, edgecolor='black', linewidth=1)
            
            ax.set_xlabel('Year', fontsize=14, fontweight='bold')
            ax.set_ylabel('Annual Streamflow (mm/year)', fontsize=14, fontweight='bold')
            ax.set_title(f'Multi-Model Streamflow Comparison\nCatchment {gauge_id}', 
                        fontsize=16, fontweight='bold')
            
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(year)) for year in common_years], rotation=45)
            ax.legend(fontsize=11, loc='upper left')
            ax.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            save_path_comparison = plot_dirs['water_balance'] / f'multi_model_streamflow_comparison_{gauge_id}.png'
            plt.savefig(save_path_comparison, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved comparison plot to: {save_path_comparison}")
            plt.show()
    
    # =============================
    # 5. PRINT OVERALL SUMMARY
    # =============================
    
    print(f"\n{'='*60}")
    print(f"MULTI-MODEL WATER BALANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Catchment: {gauge_id}")
    print(f"Period: {validation_start} to {validation_end}")
    print(f"Models processed: {len(all_model_results)}/{len(model_types)}")
    
    for model_type, annual_balance in all_model_results.items():
        name = multi_config['config_names'].get(model_type, model_type)
        coupled = multi_config['config_coupled'].get(model_type, False)
        
        print(f"\n{name} ({'Coupled' if coupled else 'Uncoupled'}):")
        print(f"  Years: {len(annual_balance)}")
        print(f"  Mean annual components (mm/year):")
        print(f"    Rainfall: {annual_balance['total_rainfall_mm'].mean():.1f}")
        print(f"    Snowmelt: {annual_balance['total_snowmelt_mm'].mean():.1f}")
        print(f"    Ice melt: {annual_balance['total_icemelt_mm'].mean():.1f}")
        print(f"    Total input: {annual_balance['total_input_mm'].mean():.1f}")
        print(f"    Observed streamflow: {annual_balance['obs_streamflow_mm'].mean():.1f}")
        print(f"    Simulated streamflow: {annual_balance['sim_streamflow_mm'].mean():.1f}")
    
    print(f"{'='*60}\n")
    
    return all_model_results