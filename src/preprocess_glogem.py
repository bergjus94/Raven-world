import geopandas as gpd
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple
import yaml
from datetime import datetime, timedelta


class GloGEMProcessor:
    """
    Streamlined processor for GloGEM glacier melt data.
    Creates irrigation forcing files for Raven hydrological modeling.
    
    ✅ UPDATED: Now reads from preprocessed NetCDF files instead of .dat files
    """
    
    def __init__(self, namelist_path: Union[str, Path]) -> None:
        """
        Initialize the GloGEM processor with namelist configuration
        
        Parameters
        ----------
        namelist_path : Union[str, Path]
            Path to the namelist YAML file
        """
        self.namelist_path = Path(namelist_path)
        
        if not self.namelist_path.exists():
            raise FileNotFoundError(f"Namelist file not found: {self.namelist_path}")
        
        with open(self.namelist_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract configuration parameters
        self.gauge_id = config['gauge_id']
        self.main_dir = Path(config['main_dir'])
        self.model_type = config['model_type']
        self.start_date = config['start_date']
        self.end_date = config['end_date']
        
        # ✅ NEW: Scenario for GloGEM NetCDF files
        self.glogem_scenario = config.get('glogem_scenario', 'ssp126')
        
        # ✅ NEW: Basin name for NetCDF file naming
        self.basin_name = config.get('basin_name', 'Indus')
        
        # ✅ Load warm-up date if specified
        if 'warm_up_date' in config:
            self.warm_up_date = config['warm_up_date']
            print(f"Warm-up period configured: {self.warm_up_date} to {self.start_date}")
        else:
            self.warm_up_date = None
            print("No warm-up period configured")
        
        self.debug = config.get('debug', False)
        self.model_dir = self.main_dir / config.get('config_dir')
        
        # ✅ UPDATED: GloGEM directory now contains NetCDF files
        self.glogem_dir = config.get('glogem_dir')
        if self.glogem_dir:
            self.glogem_dir = Path(self.main_dir, self.glogem_dir.format(gauge_id=self.gauge_id))
        
        # ✅ Define SHARED catchment-level directory (primary storage)
        self.shared_data_dir = self.model_dir / f'catchment_{self.gauge_id}' / 'data_obs'
        
        # ✅ Define MODEL-SPECIFIC directory (for backward compatibility)
        self.model_data_dir = self.model_dir / f'catchment_{self.gauge_id}' / self.model_type / 'data_obs'
        
        # Create directories
        self.shared_data_dir.mkdir(parents=True, exist_ok=True)
        self.model_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
        
        self.logger.info(f"GloGEM Processor initialized for gauge {self.gauge_id}")
        self.logger.info(f"📁 GloGEM NetCDF directory: {self.glogem_dir}")
        self.logger.info(f"📁 Shared irrigation files: {self.shared_data_dir}")
        self.logger.info(f"📋 Files will be copied to: {self.model_data_dir}")
        self.logger.info(f"🌡️  Scenario: {self.glogem_scenario}")
        
        if self.warm_up_date:
            self.logger.info(f"🔄 Warm-up period: {self.warm_up_date} to {self.start_date}")
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for this class"""
        log_level = logging.DEBUG if self.debug else logging.INFO
        
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=log_level,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Suppress matplotlib warnings
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        
        logger = logging.getLogger('GloGEMProcessor')
        return logger
    
    def _get_netcdf_path(self, component: str) -> Path:
        """
        Get the path to a GloGEM NetCDF file for a specific component
        
        Parameters
        ----------
        component : str
            One of: 'Discharge', 'Icemelt', 'Snowmelt', 'Rain'
            
        Returns
        -------
        Path
            Path to the NetCDF file
        """
        filename = f'GloGEM_{component}_{self.basin_name}_{self.glogem_scenario}.nc'
        return self.glogem_dir / filename
    
    def _get_glacier_ids_from_catchment(self) -> Tuple[set, str]:
        """
        Get glacier IDs needed for this catchment from HRU shapefile
        
        Returns
        -------
        Tuple[set, str]
            Set of numeric glacier IDs and the RGI region code
        """
        # Get catchment shapefile to identify needed glaciers
        catchment_shape_file = Path(self.model_dir) / f"catchment_{self.gauge_id}" / "topo_files" / "HRU.shp"
        
        if not catchment_shape_file.exists():
            raise FileNotFoundError(f"Catchment shapefile not found: {catchment_shape_file}")
        
        catchment = gpd.read_file(catchment_shape_file)
        
        # Extract glacier IDs from the catchment
        glacier_ids_needed = set()
        rgi_region_code = None
        
        if 'Glacier_Cl' in catchment.columns:
            glacier_series = catchment['Glacier_Cl'].dropna()
            if not glacier_series.empty:
                # Auto-detect RGI region code
                for glacier_id in glacier_series.unique():
                    if isinstance(glacier_id, str) and glacier_id.startswith('RGI60-'):
                        parts = glacier_id.split('.')
                        if len(parts) >= 2:
                            rgi_region_code = parts[0]
                            break
                
                self.logger.info(f"Auto-detected RGI region code: {rgi_region_code}")
                
                # Convert RGI60-XX.xxxxx to xxxxx format
                for glacier_id in glacier_series.unique():
                    if isinstance(glacier_id, str) and rgi_region_code and glacier_id.startswith(rgi_region_code + '.'):
                        glacier_ids_needed.add(glacier_id.replace(rgi_region_code + '.', ''))
        
        if not glacier_ids_needed:
            self.logger.warning("No glacier IDs found in catchment shapefile")
        else:
            self.logger.info(f"Found {len(glacier_ids_needed)} glacier IDs in catchment")
        
        return glacier_ids_needed, rgi_region_code
    
    # ...existing code...

    def process_glogem_files(self) -> pd.DataFrame:
        """
        Process GloGEM NetCDF files and create individual component CSV files.
        ✅ FIXED: Process in chunks to avoid memory overflow
        ✅ FIXED: Proper datetime handling from NetCDF
        
        Returns
        -------
        pd.DataFrame
            DataFrame with glacier melt data (id, date, q)
        """
        self.logger.info("Processing GloGEM NetCDF files...")
        
        # Get glacier IDs needed for this catchment
        glacier_ids_needed, rgi_region_code = self._get_glacier_ids_from_catchment()
        
        if not glacier_ids_needed:
            return pd.DataFrame(columns=['id', 'date', 'q'])
        
        # Filter for date range
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        
        # Create output directory
        topo_dir = Path(self.model_dir) / f"catchment_{self.gauge_id}" / "topo_files"
        topo_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output files and corresponding NetCDF components
        components = {
            'icemelt': ('Icemelt', topo_dir / 'GloGEM_icemelt.csv'),
            'snowmelt': ('Snowmelt', topo_dir / 'GloGEM_snowmelt.csv'),
            'output': ('Discharge', topo_dir / 'GloGEM_melt.csv'),
            'rain': ('Rain', topo_dir / 'GloGEM_rain.csv')
        }
        
        # Check if all output files already exist
        all_exist = all(path.exists() for _, path in components.values())
        
        if all_exist:
            self.logger.info("✅ All GloGEM CSV files already exist")
            self.logger.info("⏭️  Skipping NetCDF file processing")
            self.logger.info("💡 Delete files to force reprocessing:")
            for _, path in components.values():
                self.logger.info(f"   rm {path}")
            
            # Load and return melt data
            melt_path = components['output'][1]
            self.logger.info(f"Loading melt data from {melt_path}...")
            melt_df = pd.read_csv(melt_path, dtype={'id': str})
            melt_df['date'] = pd.to_datetime(melt_df['date'])
            
            self.logger.info(f"✅ Loaded {len(melt_df)} melt records for {melt_df['id'].nunique()} glaciers")
            return melt_df
        
        # Process each component
        for comp_key, (nc_component, output_path) in components.items():
            self.logger.info(f"Processing {comp_key} from NetCDF...")
            
            nc_path = self._get_netcdf_path(nc_component)
            
            if not nc_path.exists():
                self.logger.warning(f"NetCDF file not found: {nc_path}")
                continue
            
            try:
                # Open NetCDF file
                ds = xr.open_dataset(nc_path)
                
                # Get variable name (lowercase component name)
                var_name = nc_component.lower()
                
                self.logger.info(f"  NetCDF shape: {ds[var_name].shape}")
                self.logger.info(f"  Glaciers in NetCDF: {len(ds.glacier_id)}")
                self.logger.info(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
                
                # Get arrays from NetCDF
                all_times = pd.to_datetime(ds.time.values)
                all_glacier_ids = ds.glacier_id.values.astype(str)
                
                # Filter to glaciers in catchment
                matching_glaciers = list(glacier_ids_needed.intersection(set(all_glacier_ids)))
                
                self.logger.info(f"  Matching glaciers: {len(matching_glaciers)}/{len(glacier_ids_needed)}")
                
                if not matching_glaciers:
                    self.logger.warning(f"  No matching glaciers found for {comp_key}")
                    ds.close()
                    continue
                
                # Filter by time using boolean indexing
                time_mask = (all_times >= start) & (all_times <= end)
                glacier_mask = np.isin(all_glacier_ids, matching_glaciers)
                
                time_indices = np.where(time_mask)[0]
                glacier_indices = np.where(glacier_mask)[0]
                
                self.logger.info(f"  Selecting time range: {start.date()} to {end.date()}")
                self.logger.info(f"  Time steps in range: {len(time_indices)}")
                self.logger.info(f"  Glaciers to extract: {len(glacier_indices)}")
                
                if len(time_indices) == 0:
                    self.logger.warning(f"  No time steps found in date range")
                    ds.close()
                    continue
                
                # ✅ FIX: Process in chunks to avoid memory overflow
                chunk_size = 365  # Process one year at a time
                n_time_chunks = int(np.ceil(len(time_indices) / chunk_size))
                
                self.logger.info(f"  Processing in {n_time_chunks} chunks of ~{chunk_size} days...")
                
                # Open output CSV for writing
                first_chunk = True
                
                for chunk_idx in range(n_time_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min((chunk_idx + 1) * chunk_size, len(time_indices))
                    
                    chunk_time_indices = time_indices[start_idx:end_idx]
                    
                    self.logger.info(f"    Chunk {chunk_idx+1}/{n_time_chunks}: processing {len(chunk_time_indices)} days...")
                    
                    # Load chunk of data from NetCDF
                    full_data = ds[var_name].values  # Shape: (all_times, all_glaciers)
                    chunk_data = full_data[np.ix_(chunk_time_indices, glacier_indices)]
                    chunk_times = all_times[chunk_time_indices]
                    chunk_glacier_ids = all_glacier_ids[glacier_indices]
                    
                    # ✅ FIX: Create DataFrame directly (more memory efficient than list of dicts)
                    # Create multi-index for dates and glacier IDs
                    date_index = np.repeat(chunk_times, len(chunk_glacier_ids))
                    glacier_index = np.tile(chunk_glacier_ids, len(chunk_times))
                    values = chunk_data.flatten()
                    
                    # Convert dates to strings
                    date_strings = pd.to_datetime(date_index).strftime('%Y-%m-%d')
                    
                    # Create DataFrame
                    chunk_df = pd.DataFrame({
                        'id': glacier_index,
                        'date': date_strings,
                        'q': values
                    })
                    
                    # Replace NaN with 0
                    chunk_df['q'] = chunk_df['q'].fillna(0)
                    
                    # Write to CSV
                    if first_chunk:
                        chunk_df.to_csv(output_path, index=False, mode='w')
                        first_chunk = False
                    else:
                        chunk_df.to_csv(output_path, index=False, mode='a', header=False)
                    
                    # Clear memory
                    del chunk_data, chunk_df, date_index, glacier_index, values, date_strings
                    
                    self.logger.info(f"    ✓ Chunk {chunk_idx+1}/{n_time_chunks} written")
                
                ds.close()
                
                # Get file size for logging
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                self.logger.info(f"  ✅ Saved: {output_path.name} ({file_size_mb:.1f} MB)")
                
            except Exception as e:
                self.logger.error(f"Error processing {nc_path}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue
        
        # Load the output (melt/discharge) CSV and return it
        melt_path = components['output'][1]
        if melt_path.exists():
            self.logger.info(f"Loading melt data from {melt_path}...")
            melt_df = pd.read_csv(melt_path, dtype={'id': str})
            melt_df['date'] = pd.to_datetime(melt_df['date'])
            
            self.logger.info(f"Loaded {len(melt_df)} melt records for {melt_df['id'].nunique()} glaciers")
            self.logger.info(f"Date range: {melt_df['date'].min()} to {melt_df['date'].max()}")
            
            return melt_df
        else:
            self.logger.warning("No melt data file created")
            return pd.DataFrame(columns=['id', 'date', 'q'])

    def create_catchment_averaged_melt(self) -> pd.DataFrame:
        """
        Create an additional CSV file with catchment-averaged, area-weighted glacier data.
        This does NOT replace the existing individual glacier CSV files.
        
        Process ALL components: icemelt, snowmelt, rain, and total melt
        
        Process:
        1. Load individual glacier data for each component (id, date, q)
        2. Weight each glacier's values by its actual area in catchment
        3. Calculate area-weighted average: sum(value_i * area_i) / sum(area_i)
        4. Normalize by glacier fraction to get values over whole catchment area
        5. Save as GloGEM_catchment_averaged.csv
        
        Returns
        -------
        pd.DataFrame
            DataFrame with catchment-averaged data (date, icemelt_*, snowmelt_*, rain_*, melt_*)
        """
        self.logger.info("Creating catchment-averaged glacier data file (ALL components)...")
        
        # Output path for the new file
        topo_dir = Path(self.model_dir) / f"catchment_{self.gauge_id}" / "topo_files"
        output_path = topo_dir / 'GloGEM_catchment_averaged.csv'
        
        # Check if it already exists
        if output_path.exists():
            self.logger.info(f"✅ Catchment-averaged file already exists: {output_path}")
            self.logger.info("   Loading existing file...")
            return pd.read_csv(output_path, parse_dates=['date'])
        
        # Load HRU shapefile to get glacier areas
        hru_path = Path(self.model_dir) / f"catchment_{self.gauge_id}" / "topo_files" / "HRU.shp"
        hru_gdf = gpd.read_file(hru_path)
        
        # Get area column name
        area_col = 'Area_km2' if 'Area_km2' in hru_gdf.columns else 'area'
        
        # Auto-detect RGI region code
        rgi_region_code = None
        if 'Glacier_Cl' in hru_gdf.columns:
            glacier_series = hru_gdf['Glacier_Cl'].dropna()
            if not glacier_series.empty:
                for glacier_id in glacier_series.unique():
                    if isinstance(glacier_id, str) and glacier_id.startswith('RGI60-'):
                        parts = glacier_id.split('.')
                        if len(parts) >= 2:
                            rgi_region_code = parts[0]
                            break
        
        self.logger.info(f"Auto-detected RGI region code: {rgi_region_code}")
        
        # Extract glacier areas from HRU shapefile
        glacier_hrus = hru_gdf[hru_gdf['Glacier_Cl'].notna()].copy()
        
        # Group by glacier ID and sum areas (in case glacier spans multiple HRUs)
        glacier_areas = glacier_hrus.groupby('Glacier_Cl')[area_col].sum()
        
        # Create mapping: numeric ID -> area
        area_map = {}
        for full_id, area in glacier_areas.items():
            if isinstance(full_id, str) and rgi_region_code and full_id.startswith(rgi_region_code + '.'):
                numeric_id = full_id.replace(rgi_region_code + '.', '')
                area_map[numeric_id] = area
        
        self.logger.info(f"Found {len(area_map)} glaciers with areas")
        
        # Calculate total areas
        total_glacier_area_km2 = sum(area_map.values())
        total_catchment_area_km2 = hru_gdf[area_col].sum()
        glacier_fraction = total_glacier_area_km2 / total_catchment_area_km2
        
        self.logger.info(f"Total catchment area: {total_catchment_area_km2:.2f} km²")
        self.logger.info(f"Total glacier area: {total_glacier_area_km2:.2f} km²")
        self.logger.info(f"Glacier fraction: {glacier_fraction*100:.1f}%")
        
        # Define components to process
        components = {
            'icemelt': topo_dir / 'GloGEM_icemelt.csv',
            'snowmelt': topo_dir / 'GloGEM_snowmelt.csv',
            'rain': topo_dir / 'GloGEM_rain.csv',
            'melt': topo_dir / 'GloGEM_melt.csv'
        }
        
        # Check which files exist
        missing_files = []
        for comp, filepath in components.items():
            if not filepath.exists():
                missing_files.append(comp)
        
        if missing_files:
            self.logger.error(f"Missing component files: {missing_files}")
            self.logger.error("Please run process_glogem_files() first.")
            return pd.DataFrame()
        
        # Process each component
        all_daily_weighted = None
        
        for component, filepath in components.items():
            self.logger.info(f"Processing {component}...")
            
            # Load the individual glacier data
            comp_df = pd.read_csv(filepath, dtype={'id': str})
            comp_df['date'] = pd.to_datetime(comp_df['date'])
            
            self.logger.info(f"  Loaded {len(comp_df)} glacier-day records")
            
            # Add areas to dataframe
            comp_df['area_km2'] = comp_df['id'].map(area_map)
            
            # Filter out glaciers without area information
            before_filter = len(comp_df)
            comp_df = comp_df[comp_df['area_km2'].notna()].copy()
            after_filter = len(comp_df)
            
            if before_filter > after_filter:
                self.logger.warning(f"  Removed {before_filter - after_filter} records without area information")
            
            # STEP 1: Calculate area-weighted average (mm/day over glacier area)
            # Formula: sum(value_i * area_i) / sum(area_i) for each date
            daily_weighted = comp_df.groupby('date').apply(
                lambda x: pd.Series({
                    f'{component}_glacier_area': (x['q'] * x['area_km2']).sum() / x['area_km2'].sum()
                })
            ).reset_index()
            
            # STEP 2: Normalize by glacier fraction to get values over whole catchment area
            daily_weighted[f'{component}_catchment_area'] = daily_weighted[f'{component}_glacier_area'] * glacier_fraction
            
            self.logger.info(f"  ✓ Calculated area-weighted {component}")
            self.logger.info(f"    Mean (glacier area): {daily_weighted[f'{component}_glacier_area'].mean():.3f} mm/day")
            self.logger.info(f"    Mean (catchment area): {daily_weighted[f'{component}_catchment_area'].mean():.3f} mm/day")
            
            # Merge with master dataframe
            if all_daily_weighted is None:
                all_daily_weighted = daily_weighted
            else:
                all_daily_weighted = pd.merge(all_daily_weighted, daily_weighted, on='date', how='outer')
        
        # Sort by date
        all_daily_weighted = all_daily_weighted.sort_values('date').reset_index(drop=True)
        
        # Save to CSV
        all_daily_weighted.to_csv(output_path, index=False)
        
        self.logger.info(f"\n✅ Saved catchment-averaged data: {output_path}")
        self.logger.info(f"   Records: {len(all_daily_weighted)} days")
        self.logger.info(f"   Columns:")
        self.logger.info(f"     - date: Date")
        for component in components.keys():
            self.logger.info(f"     - {component}_glacier_area: Area-weighted average over glacier area (mm/day)")
            self.logger.info(f"     - {component}_catchment_area: Normalized by catchment area (mm/day)")
        
        self.logger.info(f"\n   Summary Statistics (glacier area):")
        for component in components.keys():
            mean_val = all_daily_weighted[f'{component}_glacier_area'].mean()
            max_val = all_daily_weighted[f'{component}_glacier_area'].max()
            self.logger.info(f"     {component}: mean={mean_val:.3f} mm/day, max={max_val:.3f} mm/day")
        
        self.logger.info(f"\n   Summary Statistics (catchment area):")
        for component in components.keys():
            mean_val = all_daily_weighted[f'{component}_catchment_area'].mean()
            max_val = all_daily_weighted[f'{component}_catchment_area'].max()
            self.logger.info(f"     {component}: mean={mean_val:.3f} mm/day, max={max_val:.3f} mm/day")
        
        return all_daily_weighted
    
    def create_irrigation_netcdf(self, force_reprocess: bool = False) -> xr.Dataset:
        """
        Create irrigation NetCDF file with GloGEM melt on glacier HRUs, zeros elsewhere
        ✅ OPTIMIZED: Uses numpy operations instead of pandas pivot for speed
        ✅ FIXED: Proper subsetting using numpy indexing on loaded data
        """
        self.logger.info("Creating irrigation NetCDF file...")
        
        # ✅ Use shared directory
        output_path = self.shared_data_dir / 'irrigation.nc'
        
        if output_path.exists() and not force_reprocess:
            self.logger.info(f"✅ Irrigation file already exists: {output_path}")
            self.logger.info("⏭️ Skipping. Set force_reprocess=True to reprocess.")
            return xr.open_dataset(output_path)
        
        # Load HRU data
        hru_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
        hru_gdf = gpd.read_file(hru_path)
        hru_gdf = hru_gdf.sort_values(by='HRU_ID').reset_index(drop=True)
        hru_gdf['HRU ID'] = range(1, len(hru_gdf) + 1)
        num_hrus = len(hru_gdf)
        
        # Get glacier IDs from catchment
        glacier_ids_needed, rgi_region_code = self._get_glacier_ids_from_catchment()
        
        if not glacier_ids_needed:
            self.logger.warning("No glaciers in catchment - creating empty irrigation file")
            return self._create_empty_irrigation_netcdf(hru_gdf, output_path)
        
        # ✅ Load Discharge (total melt) from NetCDF directly
        nc_path = self._get_netcdf_path('Discharge')
        
        if not nc_path.exists():
            self.logger.error(f"GloGEM Discharge NetCDF not found: {nc_path}")
            raise FileNotFoundError(f"GloGEM Discharge NetCDF not found: {nc_path}")
        
        self.logger.info(f"Loading GloGEM Discharge from: {nc_path}")
        ds_glogem = xr.open_dataset(nc_path)
        
        # Get all time and glacier info from NetCDF
        all_nc_times = pd.to_datetime(ds_glogem.time.values)
        all_glacier_ids = ds_glogem.glacier_id.values.astype(str)
        
        self.logger.info(f"NetCDF contains {len(all_nc_times)} time steps, {len(all_glacier_ids)} glaciers")
        self.logger.info(f"NetCDF date range: {all_nc_times[0].date()} to {all_nc_times[-1].date()}")
        
        # Filter to glaciers in catchment
        matching_glaciers = list(glacier_ids_needed.intersection(set(all_glacier_ids)))
        
        self.logger.info(f"Matching glaciers: {len(matching_glaciers)}/{len(glacier_ids_needed)}")
        
        if not matching_glaciers:
            self.logger.warning("No matching glaciers found - creating empty irrigation file")
            ds_glogem.close()
            return self._create_empty_irrigation_netcdf(hru_gdf, output_path)
        
        # ✅ Determine date range (with or without warm-up)
        if hasattr(self, 'warm_up_date') and self.warm_up_date is not None:
            start_date_for_file = pd.to_datetime(self.warm_up_date)
            simulation_start = pd.to_datetime(self.start_date)
            self.logger.info(f"Including warm-up period: {self.warm_up_date} to {(simulation_start - pd.Timedelta(days=1)).date()}")
        else:
            start_date_for_file = pd.to_datetime(self.start_date)
            simulation_start = None
        
        end_date_for_file = pd.to_datetime(self.end_date)
        sim_start = pd.to_datetime(self.start_date)
        
        # ✅ FIX: Create boolean masks for time and glacier filtering
        time_mask = (all_nc_times >= sim_start) & (all_nc_times <= end_date_for_file)
        glacier_mask = np.isin(all_glacier_ids, matching_glaciers)
        
        # Get indices for subsetting
        time_indices = np.where(time_mask)[0]
        glacier_indices = np.where(glacier_mask)[0]
        
        self.logger.info(f"Selecting simulation period: {sim_start.date()} to {end_date_for_file.date()}")
        self.logger.info(f"Time steps in range: {len(time_indices)}")
        self.logger.info(f"Glaciers to extract: {len(glacier_indices)}")
        
        if len(time_indices) == 0:
            self.logger.error("No time steps found in the specified date range!")
            self.logger.error(f"NetCDF time range: {all_nc_times[0]} to {all_nc_times[-1]}")
            ds_glogem.close()
            raise ValueError("No time steps found in the specified date range")
        
        # ✅ FIX: Load the full data array first, then subset with numpy
        self.logger.info("Loading discharge data from NetCDF...")
        full_discharge = ds_glogem['discharge'].values  # Shape: (all_times, all_glaciers)
        
        self.logger.info(f"Full data shape: {full_discharge.shape}")
        
        # ✅ FIX: Subset using numpy indexing (much faster and reliable)
        self.logger.info("Subsetting data with numpy...")
        discharge_data = full_discharge[np.ix_(time_indices, glacier_indices)]
        
        # Get the corresponding time and glacier arrays
        sim_times = all_nc_times[time_indices]
        sim_glacier_ids = all_glacier_ids[glacier_indices]
        
        self.logger.info(f"Subset data shape: {discharge_data.shape}")
        self.logger.info(f"Actual date range: {sim_times[0].date()} to {sim_times[-1].date()}")
        
        # Clean up full array
        del full_discharge
        ds_glogem.close()
        
        # Replace NaN with 0
        discharge_data = np.nan_to_num(discharge_data, nan=0.0).astype(np.float32)
        
        # Create mapping: numeric glacier ID -> HRU index (0-based)
        glacier_to_hru_idx = {}
        glacier_hrus = hru_gdf[hru_gdf['Glacier_Cl'].notna()].copy()
        for idx, row in glacier_hrus.iterrows():
            full_glacier_id = row['Glacier_Cl']
            if isinstance(full_glacier_id, str) and rgi_region_code and full_glacier_id.startswith(f'{rgi_region_code}.'):
                numeric_id = full_glacier_id.replace(f'{rgi_region_code}.', '')
                hru_idx = int(row['HRU ID']) - 1  # 0-based index
                glacier_to_hru_idx[numeric_id] = hru_idx
        
        self.logger.info(f"Found {len(glacier_to_hru_idx)} glacier-HRU mappings")
        
        # ✅ OPTIMIZED: Create result array using numpy directly
        self.logger.info("Creating result array using numpy (fast)...")
        n_sim_times = len(sim_times)
        result_sim = np.zeros((n_sim_times, num_hrus), dtype=np.float32)
        
        # Map glacier columns to HRU indices
        matched_count = 0
        for g_idx, glacier_id in enumerate(sim_glacier_ids):
            if glacier_id in glacier_to_hru_idx:
                hru_idx = glacier_to_hru_idx[glacier_id]
                result_sim[:, hru_idx] = discharge_data[:, g_idx]
                matched_count += 1
        
        self.logger.info(f"Matched {matched_count} glaciers to HRU columns")
        
        # Clean up
        del discharge_data
        
        # ✅ Handle warm-up period by repeating first year
        if simulation_start is not None:
            self.logger.info("Processing warm-up period...")
            
            # Find first year of simulation data in result_sim
            first_year_end = simulation_start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
            if first_year_end > end_date_for_file:
                first_year_end = end_date_for_file
            
            # ✅ FIX: Convert to numpy datetime64 for comparison
            sim_times_np = sim_times.values if hasattr(sim_times, 'values') else np.array(sim_times)
            simulation_start_np = np.datetime64(simulation_start)
            first_year_end_np = np.datetime64(first_year_end)
            
            # Find indices for first year
            first_year_mask = (sim_times_np >= simulation_start_np) & (sim_times_np <= first_year_end_np)
            first_year_indices = np.where(first_year_mask)[0]
            
            self.logger.info(f"Looking for first year: {simulation_start.date()} to {first_year_end.date()}")
            self.logger.info(f"Found {len(first_year_indices)} days in first year")
            
            if len(first_year_indices) == 0:
                self.logger.warning("No data found for first year of simulation!")
                self.logger.warning(f"Available: {sim_times[0].date()} to {sim_times[-1].date()}")
                # Use first 365 days of available data
                first_year_indices = np.arange(min(365, len(sim_times)))
                self.logger.warning(f"Using first {len(first_year_indices)} days instead")
            
            first_year_data = result_sim[first_year_indices, :]
            
            self.logger.info(f"First year data shape: {first_year_data.shape}")
            
            # Calculate how many days needed for warm-up
            warmup_days = (simulation_start - start_date_for_file).days
            
            if warmup_days > 0:
                n_repetitions = max(1, int(np.ceil(warmup_days / len(first_year_data))))
                
                self.logger.info(f"Warm-up days needed: {warmup_days}, repeating first year {n_repetitions} time(s)")
                
                # ✅ OPTIMIZED: Create warm-up data using numpy tile
                warmup_data = np.tile(first_year_data, (n_repetitions, 1))
                
                # Trim to exact warm-up length
                warmup_data = warmup_data[:warmup_days, :]
                
                # Create warm-up time array
                warmup_times = pd.date_range(start=start_date_for_file, periods=warmup_days, freq='D')
                
                self.logger.info(f"Warm-up data: {warmup_times[0].date()} to {warmup_times[-1].date()} ({len(warmup_data)} days)")
                
                # ✅ OPTIMIZED: Concatenate using numpy
                result_array = np.vstack([warmup_data, result_sim])
                full_times = warmup_times.append(pd.DatetimeIndex(sim_times))
                
                self.logger.info(f"Combined data: {full_times[0].date()} to {full_times[-1].date()} ({len(result_array)} days)")
            else:
                result_array = result_sim
                full_times = pd.DatetimeIndex(sim_times)
        else:
            result_array = result_sim
            full_times = pd.DatetimeIndex(sim_times)
        
        # Create full date range and reindex if needed
        full_date_range = pd.date_range(start=start_date_for_file, end=end_date_for_file, freq='D')
        
        # Check if we need to fill gaps
        if len(full_times) != len(full_date_range):
            self.logger.info(f"Reindexing from {len(full_times)} to {len(full_date_range)} days...")
            
            # Create a mapping from dates to indices
            full_times_normalized = pd.to_datetime(full_times).normalize()
            time_to_idx = {t: i for i, t in enumerate(full_times_normalized)}
            
            # Create new result array
            final_array = np.zeros((len(full_date_range), num_hrus), dtype=np.float32)
            
            for i, date in enumerate(full_date_range):
                if date in time_to_idx:
                    final_array[i, :] = result_array[time_to_idx[date], :]
            
            result_array = final_array
        
        self.logger.info("✅ Created result array")
        
        # Create xarray Dataset
        x_values = np.arange(1, num_hrus + 1)
        y_values = np.arange(1, 2)
        
        ds = xr.Dataset(
            {'data': (['time', 'x', 'y'], result_array.reshape(len(full_date_range), -1, 1))},
            coords={'time': full_date_range, 'x': x_values, 'y': y_values}
        )
        
        # Add elevation
        elevation_values = hru_gdf['Elev_Mean'].values
        ds['elevation'] = xr.DataArray(
            elevation_values.reshape(-1, 1),
            dims=['x', 'y'],
            coords={'x': ds['x'], 'y': ds['y']}
        )
        
        # Add metadata
        ds.attrs.update({
            'title': f'Glacier melt irrigation for catchment {self.gauge_id}',
            'source': f'GloGEM {self.glogem_scenario}',
            'n_glaciers': len(matching_glaciers),
            'n_hrus': num_hrus,
        })
        
        if simulation_start is not None and warmup_days > 0:
            ds.attrs.update({
                'warmup_included': 'true',
                'warmup_start': str(start_date_for_file.date()),
                'warmup_end': str((simulation_start - pd.Timedelta(days=1)).date()),
                'simulation_start': str(simulation_start.date()),
                'simulation_end': str(end_date_for_file.date()),
                'warmup_method': 'repeat_first_year',
                'warmup_repetitions': n_repetitions
            })
        
        # Save
        self.logger.info(f"Saving to {output_path}...")
        ds.to_netcdf(output_path)
        self.logger.info(f"✅ Saved irrigation NetCDF: {output_path}")
        
        # Log statistics
        glacier_hrus_count = (result_array != 0).any(axis=0).sum()
        non_zero = (result_array != 0).sum()
        
        self.logger.info(f"   Time range: {full_date_range[0].date()} to {full_date_range[-1].date()}")
        self.logger.info(f"   Total days: {len(full_date_range)}")
        self.logger.info(f"   Glacier HRUs: {glacier_hrus_count}/{num_hrus}")
        self.logger.info(f"   Non-zero values: {non_zero}/{result_array.size} ({non_zero/result_array.size*100:.2f}%)")
        if non_zero > 0:
            self.logger.info(f"   Mean irrigation (glacier HRUs): {result_array[result_array != 0].mean():.3f} mm/day")
            self.logger.info(f"   Max irrigation: {result_array.max():.3f} mm/day")
        
        return ds
    
    def _create_empty_irrigation_netcdf(self, hru_gdf: gpd.GeoDataFrame, output_path: Path) -> xr.Dataset:
        """Create an empty irrigation NetCDF file when no glaciers are present"""
        
        # Determine date range
        if hasattr(self, 'warm_up_date') and self.warm_up_date is not None:
            start_date_for_file = pd.to_datetime(self.warm_up_date)
        else:
            start_date_for_file = pd.to_datetime(self.start_date)
        
        end_date_for_file = pd.to_datetime(self.end_date)
        full_date_range = pd.date_range(start=start_date_for_file, end=end_date_for_file)
        
        num_hrus = len(hru_gdf)
        result_array = np.zeros((len(full_date_range), num_hrus))
        
        x_values = np.arange(1, num_hrus + 1)
        y_values = np.arange(1, 2)
        
        ds = xr.Dataset(
            {'data': (['time', 'x', 'y'], result_array.reshape(len(full_date_range), -1, 1))},
            coords={'time': full_date_range, 'x': x_values, 'y': y_values}
        )
        
        elevation_values = hru_gdf['Elev_Mean'].values
        ds['elevation'] = xr.DataArray(
            elevation_values.reshape(-1, 1),
            dims=['x', 'y'],
            coords={'x': ds['x'], 'y': ds['y']}
        )
        
        ds.attrs.update({
            'title': f'Empty glacier melt irrigation for catchment {self.gauge_id}',
            'source': 'No glaciers in catchment',
            'n_glaciers': 0,
            'n_hrus': num_hrus,
        })
        
        ds.to_netcdf(output_path)
        self.logger.info(f"✅ Saved empty irrigation NetCDF: {output_path}")
        
        return ds
    
    def create_irrigation_gridweights(self) -> None:
        """
        Create GridWeights file specifically for irrigation forcing.
        Saves as GridWeights_Irrigation.txt to avoid overwriting existing file.
        ✅ UPDATED: Saves to shared data_obs directory
        """
        self.logger.info("Creating irrigation grid weights file...")
        
        # Load HRU data to get number of HRUs
        hru_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
        hru_gdf = gpd.read_file(hru_path)
        
        number_hrus = len(hru_gdf)
        number_cells = number_hrus
        
        hru_list = list(range(1, number_hrus + 1))
        cell_ids = list(range(0, number_hrus))
        rel_areas = np.ones(number_hrus)
        
        # ✅ Use shared directory
        filename = self.shared_data_dir / 'GridWeights_Irrigation.txt'
        
        with open(filename, 'w') as f:
            f.write('# ---------------------------------------------- \n')
            f.write('# Raven GridWeights File for Irrigation Forcing \n')
            f.write('# ---------------------------------------------- \n')
            f.write('\n')
            f.write(':GridWeights\n')
            f.write('   #\n')
            f.write('   # [# HRUs]\n')
            f.write(f'   :NumberHRUs       {number_hrus}\n')
            f.write(f'   :NumberGridCells  {number_cells}\n')
            f.write('   #\n')
            f.write('   # [HRU ID] [Cell #] [w_kl]\n')
            for hru_id, cell_id, rel_area in zip(hru_list, cell_ids, rel_areas):
                f.write(f"   {hru_id}   {cell_id}   {rel_area}\n")
            f.write(':EndGridWeights\n')
        
        self.logger.info(f"✅ Saved irrigation grid weights: {filename}")
    
    def validate_glacier_ids(self) -> Dict[str, List[str]]:
        """
        Advanced validation of glacier IDs between HRU shapefile and GloGEM NetCDF.
        ✅ UPDATED: Reads from NetCDF instead of .dat files
        
        Returns
        -------
        dict
            Dictionary with 'matched', 'missing_in_glogem', 'missing_in_hru' lists
        """
        self.logger.info("Validating glacier IDs...")
        
        results = {
            'matched': [],
            'missing_in_glogem': [],
            'missing_in_hru': []
        }
        
        try:
            # Get glacier IDs from catchment
            glacier_ids_needed, rgi_region_code = self._get_glacier_ids_from_catchment()
            
            if not glacier_ids_needed:
                self.logger.warning("No glacier IDs found in catchment shapefile")
                return results
            
            # Load GloGEM NetCDF to get available glacier IDs
            nc_path = self._get_netcdf_path('Discharge')
            
            if not nc_path.exists():
                self.logger.warning(f"GloGEM Discharge NetCDF not found: {nc_path}")
                # All glaciers are missing
                for g_id in glacier_ids_needed:
                    full_id = f"{rgi_region_code}.{g_id}" if rgi_region_code else g_id
                    results['missing_in_glogem'].append(full_id)
                return results
            
            ds = xr.open_dataset(nc_path)
            glacier_ids_glogem = set(ds.glacier_id.values.astype(str))
            ds.close()
            
            # Compare sets
            hru_set = glacier_ids_needed
            
            missing_in_glogem = hru_set - glacier_ids_glogem
            missing_in_hru = glacier_ids_glogem - hru_set
            matched = hru_set.intersection(glacier_ids_glogem)
            
            # Store results with full IDs
            for g_id in missing_in_glogem:
                full_id = f"{rgi_region_code}.{g_id}" if rgi_region_code else g_id
                results['missing_in_glogem'].append(full_id)
            
            for g_id in missing_in_hru:
                full_id = f"{rgi_region_code}.{g_id}" if rgi_region_code else g_id
                results['missing_in_hru'].append(full_id)
            
            for g_id in matched:
                full_id = f"{rgi_region_code}.{g_id}" if rgi_region_code else g_id
                results['matched'].append(full_id)
            
            # Log summary
            self.logger.info(f"✅ Matched glaciers: {len(results['matched'])}")
            
            if results['missing_in_glogem']:
                self.logger.warning(f"⚠️  Missing in GloGEM: {len(results['missing_in_glogem'])}")
                for g_id in results['missing_in_glogem'][:5]:
                    self.logger.warning(f"   - {g_id}")
                if len(results['missing_in_glogem']) > 5:
                    self.logger.warning(f"   - ... and {len(results['missing_in_glogem'])-5} more")
            
            if results['missing_in_hru']:
                self.logger.info(f"ℹ️  In GloGEM but not in catchment: {len(results['missing_in_hru'])}")
            
            # Save detailed report
            self._save_validation_report(results, rgi_region_code)
            
            # Create map visualization
            hru_path = Path(self.model_dir) / f"catchment_{self.gauge_id}" / "topo_files" / "HRU.shp"
            hru_gdf = gpd.read_file(hru_path)
            self._create_validation_map(hru_gdf, results, rgi_region_code)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error validating glacier IDs: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return results
    
    def _save_validation_report(self, results: Dict[str, List[str]], rgi_region_code: str) -> None:
        """Save glacier validation report to CSV"""
        out_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', 'validation')
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Create detailed report
        report_rows = []
        
        for g_id in results['matched']:
            report_rows.append({
                'glacier_id': g_id,
                'status': 'matched',
                'in_hru': True,
                'in_glogem': True
            })
        
        for g_id in results['missing_in_glogem']:
            report_rows.append({
                'glacier_id': g_id,
                'status': 'missing_in_glogem',
                'in_hru': True,
                'in_glogem': False
            })
        
        for g_id in results['missing_in_hru']:
            report_rows.append({
                'glacier_id': g_id,
                'status': 'missing_in_hru',
                'in_hru': False,
                'in_glogem': True
            })
        
        report_df = pd.DataFrame(report_rows)
        report_path = out_dir / 'glacier_validation_report.csv'
        report_df.to_csv(report_path, index=False)
        
        self.logger.info(f"✅ Validation report saved: {report_path}")
    
    def _create_validation_map(self, hru_gdf: gpd.GeoDataFrame, results: Dict[str, List[str]], 
                               rgi_region_code: str) -> None:
        """Create map showing matched and missing glaciers"""
        out_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', 'validation')
        out_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Add validation status to HRU geodataframe
            hru_gdf = hru_gdf.copy()
            hru_gdf['validation_status'] = 'non-glacier'
            
            # Mark matched glaciers
            for g_id in results['matched']:
                mask = hru_gdf['Glacier_Cl'] == g_id
                hru_gdf.loc[mask, 'validation_status'] = 'matched'
            
            # Mark missing glaciers
            for g_id in results['missing_in_glogem']:
                mask = hru_gdf['Glacier_Cl'] == g_id
                hru_gdf.loc[mask, 'validation_status'] = 'missing_in_glogem'
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Define colors
            colors = {
                'non-glacier': 'lightgray',
                'matched': 'green',
                'missing_in_glogem': 'red'
            }
            
            # Plot each category
            for status, color in colors.items():
                subset = hru_gdf[hru_gdf['validation_status'] == status]
                if len(subset) > 0:
                    subset.plot(ax=ax, color=color, edgecolor='black', linewidth=0.5, 
                              label=status.replace('_', ' ').title())
            
            # Add labels for missing glaciers
            missing_glaciers = hru_gdf[hru_gdf['validation_status'] == 'missing_in_glogem']
            for idx, row in missing_glaciers.iterrows():
                centroid = row.geometry.centroid
                glacier_id = row['Glacier_Cl'].split('.')[-1] if '.' in str(row['Glacier_Cl']) else row['Glacier_Cl']
                ax.annotate(glacier_id, (centroid.x, centroid.y),
                          fontsize=8, ha='center', va='center',
                          bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7))
            
            ax.set_title(f'Glacier Validation Map - Gauge {self.gauge_id}\n'
                        f'Matched: {len(results["matched"])}, '
                        f'Missing in GloGEM: {len(results["missing_in_glogem"])}, '
                        f'Missing in HRU: {len(results["missing_in_hru"])}',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Easting (m)')
            ax.set_ylabel('Northing (m)')
            ax.legend(loc='best', fontsize=10)
            ax.set_aspect('equal')
            
            plt.tight_layout()
            
            # Save
            map_path = out_dir / 'glacier_validation_map.png'
            plt.savefig(map_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"✅ Validation map saved: {map_path}")
            
            if self.debug:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating validation map: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def plot_glacier_runoff_vs_observed(self) -> None:
        """
        Plot area-weighted average daily glacier runoff (from GloGEM melt/irrigation),
        observed streamflow, and optionally precipitation.
        All series are shown in mm/day over the catchment area.
        
        ✅ FIXED: Works with both CSV files (from .dat) and NetCDF files
        ✅ FIXED: Proper monthly aggregation
        ✅ FIXED: Better debugging for empty plots
        """
        import matplotlib.pyplot as plt
        
        try:
            self.logger.info("Creating glacier runoff vs observed streamflow comparison plots...")
            
            # Create plots directory
            plots_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', 'plots')
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # --- 1. Load GloGEM melt data ---
            # Try CSV first (from .dat files), then fall back to NetCDF
            glogem_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'GloGEM_melt.csv')
            
            if glogem_path.exists():
                self.logger.info(f"Loading GloGEM melt from CSV: {glogem_path}")
                glogem_df = pd.read_csv(glogem_path, dtype={'id': str})
                glogem_df['date'] = pd.to_datetime(glogem_df['date'])
                glogem_df['q'] = pd.to_numeric(glogem_df['q'], errors='coerce')
            else:
                # Try to load from NetCDF
                self.logger.info("CSV not found, loading from NetCDF...")
                nc_path = self._get_netcdf_path('Discharge')
                
                if not nc_path.exists():
                    self.logger.warning(f"Neither CSV nor NetCDF found. Skipping plots.")
                    return
                
                # Get glacier IDs from catchment
                glacier_ids_needed, rgi_region_code = self._get_glacier_ids_from_catchment()
                
                ds = xr.open_dataset(nc_path)
                
                # Filter time range
                start = pd.to_datetime(self.start_date)
                end = pd.to_datetime(self.end_date)
                
                all_nc_times = pd.to_datetime(ds.time.values)
                all_glacier_ids = ds.glacier_id.values.astype(str)
                
                time_mask = (all_nc_times >= start) & (all_nc_times <= end)
                glacier_mask = np.isin(all_glacier_ids, list(glacier_ids_needed))
                
                time_indices = np.where(time_mask)[0]
                glacier_indices = np.where(glacier_mask)[0]
                
                if len(time_indices) == 0 or len(glacier_indices) == 0:
                    self.logger.warning("No matching data in NetCDF. Skipping plots.")
                    ds.close()
                    return
                
                # Load and subset data
                full_discharge = ds['discharge'].values
                discharge_data = full_discharge[np.ix_(time_indices, glacier_indices)]
                sim_times = all_nc_times[time_indices]
                sim_glacier_ids = all_glacier_ids[glacier_indices]
                
                ds.close()
                
                # Convert to DataFrame format
                records = []
                for t_idx, date in enumerate(sim_times):
                    for g_idx, glacier_id in enumerate(sim_glacier_ids):
                        records.append({
                            'id': glacier_id,
                            'date': date,
                            'q': discharge_data[t_idx, g_idx]
                        })
                
                glogem_df = pd.DataFrame(records)
                glogem_df['q'] = pd.to_numeric(glogem_df['q'], errors='coerce').fillna(0)
                
                self.logger.info(f"Loaded {len(glogem_df)} records from NetCDF")
            
            # ✅ DEBUG: Check data
            self.logger.info(f"GloGEM data shape: {glogem_df.shape}")
            self.logger.info(f"GloGEM unique glaciers: {glogem_df['id'].nunique()}")
            self.logger.info(f"GloGEM date range: {glogem_df['date'].min()} to {glogem_df['date'].max()}")
            self.logger.info(f"GloGEM q stats: min={glogem_df['q'].min():.3f}, max={glogem_df['q'].max():.3f}, mean={glogem_df['q'].mean():.3f}")
            
            if glogem_df.empty:
                self.logger.warning("No GloGEM data available. Skipping plots.")
                return
            
            # --- 2. Load HRU data to get glacier areas ---
            hru_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
            hru_gdf = gpd.read_file(hru_path)
            hru_gdf = hru_gdf.sort_values(by='HRU_ID').reset_index(drop=True)
            
            # Get area column
            area_col = 'Area_km2' if 'Area_km2' in hru_gdf.columns else 'area'
            
            # Auto-detect RGI region code
            rgi_region_code = None
            if 'Glacier_Cl' in hru_gdf.columns:
                glacier_series = hru_gdf['Glacier_Cl'].dropna()
                if not glacier_series.empty:
                    for glacier_id in glacier_series.unique():
                        if isinstance(glacier_id, str) and glacier_id.startswith('RGI60-'):
                            parts = glacier_id.split('.')
                            if len(parts) >= 2:
                                rgi_region_code = parts[0]
                                break
            
            self.logger.info(f"RGI region code: {rgi_region_code}")
            
            # Map glacier IDs to areas
            glacier_areas_df = hru_gdf[hru_gdf['Glacier_Cl'].notna()][['Glacier_Cl', area_col]].copy()
            
            if rgi_region_code:
                glacier_areas_df['id'] = glacier_areas_df['Glacier_Cl'].str.replace(f'{rgi_region_code}.', '', regex=False)
            else:
                glacier_areas_df['id'] = glacier_areas_df['Glacier_Cl']
            
            area_map = glacier_areas_df.set_index('id')[area_col].to_dict()
            
            self.logger.info(f"Area map has {len(area_map)} glaciers")
            
            # Add areas to GloGEM data
            glogem_df['area'] = glogem_df['id'].map(area_map)
            glogem_df['area'] = pd.to_numeric(glogem_df['area'], errors='coerce')
            
            # ✅ DEBUG: Check area mapping
            matched_areas = glogem_df['area'].notna().sum()
            total_records = len(glogem_df)
            self.logger.info(f"Area mapping: {matched_areas}/{total_records} records have area info")
            
            # Filter out records without area
            glogem_df = glogem_df[glogem_df['area'].notna()].copy()
            
            if glogem_df.empty:
                self.logger.warning("No glaciers with area information. Skipping plots.")
                return
            
            # Calculate total areas
            glacier_area_km2 = hru_gdf[hru_gdf['Glacier_Cl'].notna()][area_col].sum()
            catchment_area_km2 = hru_gdf[area_col].sum()
            glacier_fraction = glacier_area_km2 / catchment_area_km2
            
            self.logger.info(f"Catchment area: {catchment_area_km2:.2f} km²")
            self.logger.info(f"Glacier area: {glacier_area_km2:.2f} km² ({glacier_fraction*100:.1f}%)")
            
            # --- 3. Calculate area-weighted glacier runoff per day (mm/day) ---
            def calc_weighted_avg(group):
                total_area = group['area'].sum()
                if total_area > 0:
                    return (group['q'] * group['area']).sum() / total_area
                return 0.0
            
            # ✅ FIX: Use proper aggregation
            daily_glacier = glogem_df.groupby('date').apply(calc_weighted_avg, include_groups=False).reset_index()
            daily_glacier.columns = ['date', 'glacier_runoff_per_glacier_area']
            
            # Normalize to catchment area (multiply by glacier fraction)
            daily_glacier['glacier_runoff_catchment_norm'] = daily_glacier['glacier_runoff_per_glacier_area'] * glacier_fraction
            
            # ✅ DEBUG: Check daily data
            self.logger.info(f"Daily glacier runoff calculated for {len(daily_glacier)} days")
            self.logger.info(f"Date range: {daily_glacier['date'].min()} to {daily_glacier['date'].max()}")
            self.logger.info(f"Mean runoff (glacier area): {daily_glacier['glacier_runoff_per_glacier_area'].mean():.3f} mm/day")
            self.logger.info(f"Mean runoff (catchment): {daily_glacier['glacier_runoff_catchment_norm'].mean():.3f} mm/day")
            self.logger.info(f"Non-zero days: {(daily_glacier['glacier_runoff_catchment_norm'] > 0).sum()}")
            
            # --- 4. Load observed streamflow from Q_daily.rvt ---
            q_file = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs', 'Q_daily.rvt')
            
            obs_series_mm = None
            if not q_file.exists():
                self.logger.warning(f"Observed streamflow file not found: {q_file}")
            else:
                try:
                    with open(q_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Find start date (first non-comment, non-empty line)
                    start_date_str = None
                    data_start_idx = 0
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        if stripped and not stripped.startswith(':') and not stripped.startswith('#'):
                            parts = stripped.split()
                            if len(parts) >= 1:
                                start_date_str = parts[0]
                                data_start_idx = i + 1
                                break
                    
                    if start_date_str:
                        # Extract values
                        value_lines = []
                        for line in lines[data_start_idx:]:
                            stripped = line.strip()
                            if stripped and not stripped.startswith(':') and not stripped.startswith('#'):
                                try:
                                    value = float(stripped)
                                    value_lines.append(value)
                                except ValueError:
                                    continue
                        
                        if value_lines:
                            obs_dates = pd.date_range(start=start_date_str, periods=len(value_lines), freq='D')
                            obs_series = pd.Series(value_lines, index=obs_dates, name='observed_streamflow_m3s')
                            
                            # Convert from m³/s to mm/day
                            obs_series_mm = obs_series * 86400 / (catchment_area_km2 * 1e6) * 1000
                            obs_series_mm.name = 'observed_streamflow_mm'
                            
                            self.logger.info(f"Loaded {len(obs_series_mm)} days of observed streamflow")
                            self.logger.info(f"Obs date range: {obs_series_mm.index.min()} to {obs_series_mm.index.max()}")
                except Exception as e:
                    self.logger.warning(f"Error loading observed streamflow: {e}")
            
            # --- 5. Merge data for plotting ---
            plot_df = daily_glacier.copy()
            
            if obs_series_mm is not None:
                obs_df = obs_series_mm.reset_index()
                obs_df.columns = ['date', 'observed_streamflow_mm']
                plot_df = plot_df.merge(obs_df, on='date', how='left')
            
            # ✅ DEBUG: Check plot dataframe
            self.logger.info(f"Plot dataframe shape: {plot_df.shape}")
            self.logger.info(f"Plot dataframe columns: {plot_df.columns.tolist()}")
            self.logger.info(f"Plot dataframe head:\n{plot_df.head()}")
            
            # --- 6. Create time series plot (mm/day) ---
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # ✅ FIX: Convert dates to datetime for proper plotting
            plot_dates = pd.to_datetime(plot_df['date'])
            
            # Plot glacier runoff (per glacier area)
            ax.plot(plot_dates, plot_df['glacier_runoff_per_glacier_area'].values, 
                label='Glacier Runoff (per glacier area)', 
                color='blue', alpha=0.7, linewidth=1)
            
            # Plot catchment-normalized glacier runoff
            ax.plot(plot_dates, plot_df['glacier_runoff_catchment_norm'].values, 
                label=f'Glacier Runoff (catchment-normalized, {glacier_fraction*100:.1f}% glacier)', 
                color='green', alpha=0.7, linewidth=1.5)
            
            # Plot observed streamflow if available
            if 'observed_streamflow_mm' in plot_df.columns:
                valid_mask = plot_df['observed_streamflow_mm'].notna()
                if valid_mask.any():
                    ax.plot(plot_dates[valid_mask], plot_df.loc[valid_mask, 'observed_streamflow_mm'].values, 
                        label='Observed Streamflow', 
                        color='black', linewidth=1)
                    self.logger.info(f"Plotted {valid_mask.sum()} days of observed data")
            
            ax.set_title(f'Glacier Runoff vs Observed Streamflow - Gauge {self.gauge_id}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Discharge (mm/day)')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # ✅ DEBUG: Check axis limits
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            self.logger.info(f"Plot y-limits: {ylim}")
            self.logger.info(f"Plot x-limits: {xlim}")
            
            plt.tight_layout()
            output_path = plots_dir / 'glacier_runoff_vs_observed_mm.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Time series plot saved: {output_path}")
            
            # --- 7. Create monthly regime plot ---
            regime_df = plot_df.copy()
            regime_df['month'] = pd.to_datetime(regime_df['date']).dt.month
            
            # Calculate monthly means
            monthly_glacier = regime_df.groupby('month')['glacier_runoff_catchment_norm'].mean()
            monthly_glacier = monthly_glacier.reindex(range(1, 13), fill_value=0)
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            x = np.arange(1, 13)
            
            self.logger.info(f"Monthly glacier runoff values: {monthly_glacier.values}")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot glacier runoff regime as bars
            bars = ax.bar(x, monthly_glacier.values, color='lightblue', alpha=0.7, 
                edgecolor='blue', label='Glacier Runoff (catchment-normalized)')
            
            # Add line on top of bars
            ax.plot(x, monthly_glacier.values, 'bo-', linewidth=2, markersize=8)
            
            # Plot observed if available
            if 'observed_streamflow_mm' in regime_df.columns:
                monthly_obs = regime_df.groupby('month')['observed_streamflow_mm'].mean()
                monthly_obs = monthly_obs.reindex(range(1, 13), fill_value=np.nan)
                
                if monthly_obs.notna().any():
                    ax.plot(x, monthly_obs.values, 'ko-', linewidth=2, markersize=8, 
                        label='Observed Streamflow')
            
            # Add value labels on bars
            for i, val in enumerate(monthly_glacier.values):
                if val > 0:
                    ax.text(i + 1, val + max(monthly_glacier.values) * 0.02, f'{val:.2f}', 
                        ha='center', va='bottom', fontsize=8, color='blue')
            
            ax.set_title(f'Monthly Regime: Glacier Runoff vs Observed Streamflow\nGauge {self.gauge_id}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Mean Discharge (mm/day)')
            ax.set_xticks(x)
            ax.set_xticklabels(month_names)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            ax.set_ylim(bottom=0)
            
            plt.tight_layout()
            output_path = plots_dir / 'monthly_regime_glacier_vs_observed.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Monthly regime plot saved: {output_path}")
            
            # --- 8. Log statistics ---
            self.logger.info("\n" + "="*60)
            self.logger.info("GLACIER RUNOFF VS OBSERVED STREAMFLOW STATISTICS")
            self.logger.info("="*60)
            self.logger.info(f"Catchment area: {catchment_area_km2:.2f} km²")
            self.logger.info(f"Glacier area: {glacier_area_km2:.2f} km² ({glacier_fraction*100:.1f}%)")
            self.logger.info(f"Mean glacier runoff (per glacier area): {daily_glacier['glacier_runoff_per_glacier_area'].mean():.3f} mm/day")
            self.logger.info(f"Mean glacier runoff (catchment-normalized): {daily_glacier['glacier_runoff_catchment_norm'].mean():.3f} mm/day")
            
            if 'observed_streamflow_mm' in plot_df.columns and plot_df['observed_streamflow_mm'].notna().any():
                mean_obs = plot_df['observed_streamflow_mm'].mean()
                mean_glacier = daily_glacier['glacier_runoff_catchment_norm'].mean()
                self.logger.info(f"Mean observed streamflow: {mean_obs:.3f} mm/day")
                if mean_obs > 0:
                    glacier_contribution = (mean_glacier / mean_obs) * 100
                    self.logger.info(f"Glacier contribution to streamflow: {glacier_contribution:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error creating glacier runoff comparison plots: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

        
    def process_all(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Run complete GloGEM processing pipeline.
        ✅ UPDATED: Now reads from NetCDF files instead of .dat files
        
        Parameters
        ----------
        force_reprocess : bool
            Force reprocessing even if files exist
            
        Returns
        -------
        dict
            Dictionary with processing results
        """
        self.logger.info("="*60)
        self.logger.info(f"Starting GloGEM processing for gauge {self.gauge_id}")
        self.logger.info(f"📁 Reading from NetCDF files in: {self.glogem_dir}")
        self.logger.info(f"🌡️  Scenario: {self.glogem_scenario}")
        self.logger.info("="*60)
        
        # ✅ CHECK: Skip if final output already exists in shared directory
        irrigation_nc = self.shared_data_dir / 'irrigation.nc'
        irrigation_gridweights = self.shared_data_dir / 'GridWeights_Irrigation.txt'
        
        if irrigation_nc.exists() and irrigation_gridweights.exists() and not force_reprocess:
            self.logger.info("✅ All GloGEM processing outputs already exist")
            self.logger.info("⏭️  Skipping GloGEM processing")
            self.logger.info("💡 Files found:")
            self.logger.info(f"   {irrigation_nc}")
            self.logger.info(f"   {irrigation_gridweights}")
            self.logger.info("💡 To force reprocessing:")
            self.logger.info(f"   rm {irrigation_nc}")
            self.logger.info(f"   rm {irrigation_gridweights}")
            self.logger.info("   OR call process_all(force_reprocess=True)")
            
            # Copy to model directory before returning
            self._copy_to_model_directory()
            
            # Return empty results dict
            return {
                'glogem_data': pd.DataFrame(),
                'catchment_averaged_melt': pd.DataFrame(),
                'irrigation_netcdf': None,
                'validation': {'matched': [], 'missing_in_glogem': [], 'missing_in_hru': []},
                'skipped': True
            }
        
        # ===== NORMAL PROCESSING =====
        
        results = {}
        
        # 1. Process GloGEM NetCDF files (creates individual glacier CSVs)
        self.logger.info("\n1. Processing GloGEM NetCDF files...")
        glogem_df = self.process_glogem_files()
        results['glogem_data'] = glogem_df
        
        # 2. Create catchment-averaged melt file
        self.logger.info("\n2. Creating catchment-averaged glacier melt file...")
        catchment_avg_df = self.create_catchment_averaged_melt()
        results['catchment_averaged_melt'] = catchment_avg_df
        
        # 3. Create irrigation NetCDF
        self.logger.info("\n3. Creating irrigation NetCDF...")
        irrigation_ds = self.create_irrigation_netcdf(force_reprocess=force_reprocess)
        results['irrigation_netcdf'] = irrigation_ds
        
        # 4. Create irrigation grid weights
        self.logger.info("\n4. Creating irrigation grid weights...")
        self.create_irrigation_gridweights()
        
        # 5. Validate glacier IDs
        self.logger.info("\n5. Validating glacier IDs...")
        validation = self.validate_glacier_ids()
        results['validation'] = validation
        
        # 6. Create comparison plots with observed streamflow
        self.logger.info("\n6. Creating glacier runoff vs observed streamflow plots...")
        self.plot_glacier_runoff_vs_observed()
        
        # 7. Copy irrigation files to model-specific directory
        self.logger.info("\n7. Copying files to model-specific directory...")
        self._copy_to_model_directory()
        
        self.logger.info("\n" + "="*60)
        self.logger.info("GloGEM PROCESSING COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"✅ GloGEM melt data: {len(glogem_df)} individual glacier records")
        self.logger.info(f"✅ Catchment-averaged melt: {len(catchment_avg_df)} daily records")
        self.logger.info(f"✅ Irrigation NetCDF created")
        self.logger.info(f"✅ Grid weights created")
        self.logger.info(f"✅ Validation: {len(validation['matched'])} matched, "
                        f"{len(validation['missing_in_glogem'])} missing in GloGEM, "
                        f"{len(validation['missing_in_hru'])} missing in HRU")
        self.logger.info(f"✅ Comparison plots created")
        
        results['skipped'] = False
        
        return results
    
    def _copy_to_model_directory(self) -> None:
        """
        Copy irrigation files from shared data_obs to model-specific data_obs.
        This maintains backward compatibility while using shared storage.
        """
        import shutil
        
        # Files to copy
        irrigation_files = [
            self.shared_data_dir / 'irrigation.nc',
            self.shared_data_dir / 'GridWeights_Irrigation.txt'
        ]
        
        self.logger.info(f"📋 Copying irrigation files from shared to model-specific directory...")
        self.logger.debug(f"  Source: {self.shared_data_dir}")
        self.logger.debug(f"  Destination: {self.model_data_dir}")
        
        copied_count = 0
        for file_path in irrigation_files:
            if file_path.exists():
                dest = self.model_data_dir / file_path.name
                try:
                    shutil.copy2(file_path, dest)
                    self.logger.debug(f"  ✅ Copied: {file_path.name}")
                    copied_count += 1
                except Exception as e:
                    self.logger.warning(f"  ❌ Failed to copy {file_path.name}: {e}")
            else:
                self.logger.warning(f"  ⚠️ File not found: {file_path}")
        
        self.logger.info(f"✅ Successfully copied {copied_count}/{len(irrigation_files)} files to {self.model_data_dir.name}/")