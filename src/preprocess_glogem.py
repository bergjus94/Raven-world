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
import traceback


# Maps namelist irrigation_variable value → topo_files CSV filename
_GLOGEM_CSV = {
    'discharge': 'GloGEM_melt.csv',   # historical naming kept for compat
    'icemelt':   'GloGEM_icemelt.csv',
    'snowmelt':  'GloGEM_snowmelt.csv',
    'rain':      'GloGEM_rain.csv',
}

# Human-readable label used in plot titles / axis labels
_GLOGEM_LABEL = {
    'discharge': 'Glacier Runoff (discharge)',
    'icemelt':   'Ice Melt',
    'snowmelt':  'Snow Melt',
    'rain':      'Glacier Rain',
}

_VALID_IRRIGATION_VARS = {'discharge', 'icemelt', 'snowmelt', 'rain'}


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

        # ✅ NEW: Climate model name for NetCDF file naming (e.g. 'gfdl-esm4')
        self.glogem_model = config.get('glogem_model', None)
        
        # ✅ Load warm-up date if specified
        if 'warm_up_date' in config:
            self.warm_up_date = config['warm_up_date']
        else:
            self.warm_up_date = None
        
        self.debug = config.get('debug', False)

        self.irrigation_variable = config.get('irrigation_variable', 'discharge').lower()
        if self.irrigation_variable not in _VALID_IRRIGATION_VARS:
            raise ValueError(
                f"irrigation_variable must be one of {_VALID_IRRIGATION_VARS}, "
                f"got '{self.irrigation_variable}'"
            )

        self.coupled = config.get('coupled', False)

        self.model_dir = self.main_dir / config.get('config_dir')

        # ✅ UPDATED: GloGEM directory now contains NetCDF files
        self.glogem_dir = config.get('glogem_dir')
        if self.glogem_dir:
            self.glogem_dir = Path(self.main_dir, self.glogem_dir.format(gauge_id=self.gauge_id))
        
        # Shared catchment-level directory (canonical location for irrigation files)
        self.shared_data_dir = self.model_dir / f'catchment_{self.gauge_id}' / 'data_obs'
        self.shared_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
        
        self.logger.info(f"GloGEM Processor initialized for gauge {self.gauge_id}")
        self.logger.info(f"📁 GloGEM NetCDF directory: {self.glogem_dir}")
        self.logger.info(f"📁 Irrigation files: {self.shared_data_dir}")
        self.logger.info(f"🌡️  Scenario: {self.glogem_scenario}")
        if self.glogem_model:
            self.logger.info(f"🌐 Climate model: {self.glogem_model}")

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

    @property
    def icemelt_mode(self) -> bool:
        """True when coupled + icemelt: HRUs are ROCK with elevation discretization."""
        return self.coupled and self.irrigation_variable == 'icemelt'

    def _load_glacier_hru_overlap(self) -> pd.DataFrame:
        """Load glacier_hru_overlap.csv (icemelt mode pixel-overlap table)."""
        overlap_path = (
            Path(self.model_dir) / f"catchment_{self.gauge_id}" / "topo_files" / "glacier_hru_overlap.csv"
        )
        if not overlap_path.exists():
            raise FileNotFoundError(
                f"glacier_hru_overlap.csv not found: {overlap_path}\n"
                "Run preprocess_catchment_hru.py with icemelt mode first."
            )
        return pd.read_csv(overlap_path)

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
        if self.glogem_model:
            filename = f'GloGEM_{component}_{self.basin_name}_{self.glogem_model}_{self.glogem_scenario}.nc'
        else:
            filename = f'GloGEM_{component}_{self.basin_name}_{self.glogem_scenario}.nc'
        return self.glogem_dir / filename

    def _get_glacier_ids_from_catchment(self) -> Tuple[set, str, Dict[str, List[str]]]:
        """
        Get glacier IDs needed for this catchment from glacier_id_mapping.csv.
        ✅ UPDATED: Reads from glacier_id_mapping.csv instead of HRU.shp (no character limits!)
        
        Returns
        -------
        Tuple[set, str, Dict[str, List[str]]]
            - Set of numeric glacier IDs (with leading zeros, e.g., "00029")
            - RGI region code  
            - Dictionary mapping glacier size category ('large', 'small', 'all') to list of glacier IDs
        """
        self.logger.info("Getting glacier IDs from catchment...")
        
        # ✅ NEW: Read from glacier_id_mapping.csv instead of HRU shapefile
        mapping_path = Path(self.model_dir) / f"catchment_{self.gauge_id}" / "topo_files" / "glacier_id_mapping.csv"
        
        if not mapping_path.exists():
            self.logger.error(f"❌ Glacier ID mapping not found: {mapping_path}")
            self.logger.error("Please run preprocess_catchment.py first to generate the mapping file!")
            raise FileNotFoundError(f"Missing glacier_id_mapping.csv at {mapping_path}")
        
        # Load the mapping CSV
        mapping_df = pd.read_csv(mapping_path)
        
        self.logger.info(f"✅ Loaded {len(mapping_df)} glacier IDs from mapping file")
        
        # Extract glacier IDs and classify by size
        glacier_ids_needed = set()
        large_glacier_ids = []
        small_glacier_ids = []
        rgi_region_code = None
        
        for _, row in mapping_df.iterrows():
            rgi_id = row['RGIId']
            numeric_id = str(row['numeric_id'])  # Ensure it's a string
            is_large = row.get('is_large', False)  # Default to False if column missing
            
            # Auto-detect RGI region code from first glacier
            if rgi_region_code is None and rgi_id.startswith('RGI60-'):
                parts = rgi_id.split('.')
                if len(parts) >= 2:
                    rgi_region_code = parts[0]
            
            # Ensure numeric ID has leading zeros (5 digits) to match NetCDF format
            numeric_id_padded = numeric_id.zfill(5)
            glacier_ids_needed.add(numeric_id_padded)
            
            # Add to size category lists
            if is_large:
                large_glacier_ids.append(numeric_id_padded)
            else:
                small_glacier_ids.append(numeric_id_padded)
        
        # Create size category mapping
        glacier_categories = {
            'all': list(glacier_ids_needed),
            'large': large_glacier_ids,
            'small': small_glacier_ids
        }
        
        self.logger.info(f"✅ Found {len(glacier_ids_needed)} glacier IDs in catchment")
        self.logger.info(f"   - Large glaciers (>=2 km²): {len(large_glacier_ids)}")
        self.logger.info(f"   - Small glaciers (<2 km²): {len(small_glacier_ids)}")
        self.logger.info(f"   - RGI region code: {rgi_region_code}")
        
        # Debug: Show sample IDs
        if self.debug and len(glacier_ids_needed) > 0:
            sample_ids = list(glacier_ids_needed)[:5]
            self.logger.debug(f"   Sample glacier IDs: {sample_ids}")
        
        return glacier_ids_needed, rgi_region_code, glacier_categories

    def _load_area_map(self) -> Dict[str, float]:
        """Load glacier area mapping (padded numeric_id → area_km2) from glacier_id_mapping.csv."""
        mapping_path = (
            Path(self.model_dir) / f"catchment_{self.gauge_id}" / "topo_files" / "glacier_id_mapping.csv"
        )
        if not mapping_path.exists():
            raise FileNotFoundError(f"Glacier ID mapping not found: {mapping_path}")
        df = pd.read_csv(mapping_path)
        return dict(zip(df['numeric_id'].astype(str).str.zfill(5), df['area_km2']))

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
        glacier_ids_needed, rgi_region_code, glacier_categories = self._get_glacier_ids_from_catchment()
        
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

                # Pre-select only the needed glacier columns using xarray lazy indexing
                # This avoids loading the full 3+ GB array into memory
                da_subset = ds[var_name][:, glacier_indices]
                chunk_glacier_ids = all_glacier_ids[glacier_indices]

                # Open output CSV for writing
                first_chunk = True

                for chunk_idx in range(n_time_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min((chunk_idx + 1) * chunk_size, len(time_indices))

                    chunk_time_indices = time_indices[start_idx:end_idx]

                    self.logger.info(f"    Chunk {chunk_idx+1}/{n_time_chunks}: processing {len(chunk_time_indices)} days...")

                    # Load only this time chunk for the pre-selected glaciers
                    chunk_data = da_subset[chunk_time_indices, :].values
                    chunk_times = all_times[chunk_time_indices]
                    
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
        Create catchment-averaged, area-weighted glacier data for ALL glaciers, LARGE glaciers, and SMALL glaciers.
        ✅ UPDATED: Calculates 3 separate weighted averages (all, large >=2km², small <2km²)
        
        Process ALL components: icemelt, snowmelt, rain, and total melt
        
        Process:
        1. Load individual glacier data for each component (id, date, q)
        2. Weight each glacier's values by its actual area in catchment
        3. Calculate 3 area-weighted averages: 
           - All glaciers: sum(value_i * area_i) / sum(area_i)
           - Large glaciers (>=2km²): same but only for large glaciers
           - Small glaciers (<2km²): same but only for small glaciers
        4. Save as GloGEM_catchment_averaged.csv
        
        Returns
        -------
        pd.DataFrame
            DataFrame with catchment-averaged data (date, component_all, component_large, component_small)
        """
        self.logger.info("Creating catchment-averaged glacier data (ALL, LARGE, SMALL)...")
        
        # Output path for the new file
        topo_dir = Path(self.model_dir) / f"catchment_{self.gauge_id}" / "topo_files"
        output_path = topo_dir / 'GloGEM_catchment_averaged.csv'
        
        # Check if it already exists
        if output_path.exists():
            self.logger.info(f"✅ Catchment-averaged file already exists: {output_path}")
            self.logger.info("   Loading existing file...")
            return pd.read_csv(output_path, parse_dates=['date'])
        
        # Get glacier IDs by category
        glacier_ids_needed, rgi_region_code, glacier_categories = self._get_glacier_ids_from_catchment()
        
        area_map = self._load_area_map()
        self.logger.info(f"Loaded {len(area_map)} glacier areas from mapping CSV")
        
        # Calculate total areas by category
        area_all = sum(area_map.get(gid, 0.0) for gid in glacier_categories['all'])
        area_large = sum(area_map.get(gid, 0.0) for gid in glacier_categories['large'])
        area_small = sum(area_map.get(gid, 0.0) for gid in glacier_categories['small'])
        
        self.logger.info(f"Total glacier area: {area_all:.2f} km²")
        self.logger.info(f"  - Large glaciers (>=2 km²): {area_large:.2f} km² ({len(glacier_categories['large'])} glaciers)")
        self.logger.info(f"  - Small glaciers (<2 km²): {area_small:.2f} km² ({len(glacier_categories['small'])} glaciers)")
        
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
            
            # Classify glaciers as large or small
            comp_df['glacier_size'] = comp_df['id'].apply(
                lambda x: 'large' if x in glacier_categories['large'] else 'small'
            )
            
            # ✅ Calculate 3 area-weighted averages: ALL, LARGE, SMALL
            def calc_weighted_avg(group):
                total_area = group['area_km2'].sum()
                if total_area > 0:
                    return (group['q'] * group['area_km2']).sum() / total_area
                return 0.0
            
            # ALL glaciers
            daily_all = comp_df.groupby('date').apply(calc_weighted_avg, include_groups=False).reset_index()
            daily_all.columns = ['date', f'{component}_all']
            
            # LARGE glaciers
            comp_large = comp_df[comp_df['glacier_size'] == 'large']
            if not comp_large.empty:
                daily_large = comp_large.groupby('date').apply(calc_weighted_avg, include_groups=False).reset_index()
                daily_large.columns = ['date', f'{component}_large']
            else:
                daily_large = pd.DataFrame({'date': daily_all['date'], f'{component}_large': 0.0})
            
            # SMALL glaciers
            comp_small = comp_df[comp_df['glacier_size'] == 'small']
            if not comp_small.empty:
                daily_small = comp_small.groupby('date').apply(calc_weighted_avg, include_groups=False).reset_index()
                daily_small.columns = ['date', f'{component}_small']
            else:
                daily_small = pd.DataFrame({'date': daily_all['date'], f'{component}_small': 0.0})
            
            # Merge all three categories
            daily_weighted = daily_all.merge(daily_large, on='date', how='outer').merge(daily_small, on='date', how='outer')
            
            self.logger.info(f"  ✓ Calculated area-weighted {component}")
            self.logger.info(f"    Mean (all glaciers): {daily_weighted[f'{component}_all'].mean():.3f} mm/day")
            self.logger.info(f"    Mean (large glaciers): {daily_weighted[f'{component}_large'].mean():.3f} mm/day")
            self.logger.info(f"    Mean (small glaciers): {daily_weighted[f'{component}_small'].mean():.3f} mm/day")
            
            # Merge with master dataframe
            if all_daily_weighted is None:
                all_daily_weighted = daily_weighted
            else:
                all_daily_weighted = pd.merge(all_daily_weighted, daily_weighted, on='date', how='outer')
        
        # Sort by date
        all_daily_weighted = all_daily_weighted.sort_values('date').reset_index(drop=True)
        
        # ✅ NEW: Calculate catchment-scaled values
        # Load HRU shapefile to get total catchment area and glacier HRU areas
        hru_path = Path(self.model_dir) / f"catchment_{self.gauge_id}" / "topo_files" / "HRU.shp"
        hru_gdf = gpd.read_file(hru_path)
        area_col = 'Area_km2' if 'Area_km2' in hru_gdf.columns else 'area'
        total_catchment_area_km2 = hru_gdf[area_col].sum()
        
        # Get glacier HRU areas from HRU shapefile
        glacier_hrus = hru_gdf[hru_gdf['Glacier_Cl'].notna()].copy()

        all_glacier_area_km2 = 0.0
        large_glacier_area_km2 = 0.0
        small_glacier_area_km2 = 0.0

        if len(glacier_hrus) == 0:
            # Icemelt mode: glacier pixels are ROCK (Landuse_Cl=5), so Glacier_Cl is absent.
            # Fall back to clipped_glacier.shp and compute area from GEOMETRY (not the
            # 'Area' attribute, which stores original RGI polygon areas before clipping
            # and is larger than the rasterized HRU-based area used by standard coupled configs).
            glacier_shp_path = Path(self.model_dir) / f"catchment_{self.gauge_id}" / "topo_files" / "clipped_glacier.shp"
            if glacier_shp_path.exists():
                glacier_gdf = gpd.read_file(glacier_shp_path)
                # Project to UTM for accurate area calculation, then convert m² to km²
                glacier_gdf_proj = glacier_gdf.to_crs(glacier_gdf.estimate_utm_crs())
                for idx, row in glacier_gdf_proj.iterrows():
                    area_km2 = row.geometry.area / 1e6  # m² -> km²
                    rgi_id = row.get('RGIId', '')
                    all_glacier_area_km2 += area_km2
                    short_id = rgi_id.replace(rgi_region_code + '.', '') if rgi_region_code and isinstance(rgi_id, str) else rgi_id
                    if short_id in glacier_categories.get('large', set()):
                        large_glacier_area_km2 += area_km2
                    else:
                        small_glacier_area_km2 += area_km2
                self.logger.info(f"  Icemelt mode: computed glacier area from clipped_glacier.shp geometry ({len(glacier_gdf)} glaciers)")
            else:
                self.logger.warning(f"  No Glacier_Cl in HRUs and no clipped_glacier.shp found — glacier fraction will be 0")
        else:
            for idx, row in glacier_hrus.iterrows():
                hru_area = row[area_col]
                glacier_id_str = row['Glacier_Cl']
                all_glacier_area_km2 += hru_area

                # Determine if this HRU contains large or small glaciers
                # by checking the first glacier ID in the pipe-separated list
                if isinstance(glacier_id_str, str) and '|' in glacier_id_str:
                    id_list = glacier_id_str.split('|')
                    first_glacier_id = id_list[0].replace(rgi_region_code + '.', '') if rgi_region_code else id_list[0]
                    if first_glacier_id in glacier_categories['large']:
                        large_glacier_area_km2 += hru_area
                    else:
                        small_glacier_area_km2 += hru_area
                elif isinstance(glacier_id_str, str):
                    glacier_id = glacier_id_str.replace(rgi_region_code + '.', '') if rgi_region_code else glacier_id_str
                    if glacier_id in glacier_categories['large']:
                        large_glacier_area_km2 += hru_area
                    else:
                        small_glacier_area_km2 += hru_area
        
        # Calculate glacier fractions
        fraction_all = all_glacier_area_km2 / total_catchment_area_km2 if total_catchment_area_km2 > 0 else 0
        fraction_large = large_glacier_area_km2 / total_catchment_area_km2 if total_catchment_area_km2 > 0 else 0
        fraction_small = small_glacier_area_km2 / total_catchment_area_km2 if total_catchment_area_km2 > 0 else 0
        
        self.logger.info(f"\nCatchment scaling factors:")
        self.logger.info(f"  Total catchment area: {total_catchment_area_km2:.2f} km²")
        self.logger.info(f"  All glacier area (from HRU): {all_glacier_area_km2:.2f} km² → fraction: {fraction_all*100:.2f}%")
        self.logger.info(f"  Large glacier area (from HRU): {large_glacier_area_km2:.2f} km² → fraction: {fraction_large*100:.2f}%")
        self.logger.info(f"  Small glacier area (from HRU): {small_glacier_area_km2:.2f} km² → fraction: {fraction_small*100:.2f}%")
        
        # Add catchment-scaled columns for each component
        for component in components.keys():
            all_daily_weighted[f'{component}_all_catchment'] = all_daily_weighted[f'{component}_all'] * fraction_all
            all_daily_weighted[f'{component}_large_catchment'] = all_daily_weighted[f'{component}_large'] * fraction_large
            all_daily_weighted[f'{component}_small_catchment'] = all_daily_weighted[f'{component}_small'] * fraction_small
        
        # Save to CSV
        all_daily_weighted.to_csv(output_path, index=False)
        
        self.logger.info(f"\n✅ Saved catchment-averaged data: {output_path}")
        self.logger.info(f"   Records: {len(all_daily_weighted)} days")
        self.logger.info(f"   Columns:")
        self.logger.info(f"     - date: Date")
        for component in components.keys():
            self.logger.info(f"     - {component}_all: All glaciers (mm/day over glacier area)")
            self.logger.info(f"     - {component}_large: Large glaciers >=2km² (mm/day over large glacier area)")
            self.logger.info(f"     - {component}_small: Small glaciers <2km² (mm/day over small glacier area)")
        
        self.logger.info(f"\n   Summary Statistics:")
        for component in components.keys():
            self.logger.info(f"   {component}:")
            for category in ['all', 'large', 'small']:
                col = f'{component}_{category}'
                mean_val = all_daily_weighted[col].mean()
                max_val = all_daily_weighted[col].max()
                self.logger.info(f"     {category}: mean={mean_val:.3f} mm/day, max={max_val:.3f} mm/day")
        
        return all_daily_weighted
    
    def create_irrigation_netcdf(self, force_reprocess: bool = False) -> xr.Dataset:
        """
        Create irrigation NetCDF file with GloGEM melt on glacier HRUs, zeros elsewhere.
        ✅ UPDATED: Handles aggregated glacier HRUs (2 HRUs: large + small) instead of per-glacier HRUs
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
        
        # Get glacier IDs from catchment with size categories
        glacier_ids_needed, rgi_region_code, glacier_categories = self._get_glacier_ids_from_catchment()
        
        if not glacier_ids_needed:
            self.logger.warning("No glaciers in catchment - creating empty irrigation file")
            return self._create_empty_irrigation_netcdf(hru_gdf, output_path)
        
        area_map = self._load_area_map()
        self.logger.info(f"Loaded {len(area_map)} glacier areas from mapping CSV")
        
        # ✅ Load the configured irrigation variable from NetCDF
        nc_path = self._get_netcdf_path(self.irrigation_variable.capitalize())

        if not nc_path.exists():
            self.logger.error(f"GloGEM {self.irrigation_variable} NetCDF not found: {nc_path}")
            raise FileNotFoundError(f"GloGEM {self.irrigation_variable} NetCDF not found: {nc_path}")

        self.logger.info(f"Loading GloGEM {self.irrigation_variable} from: {nc_path}")
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
        
        # ✅ Create boolean masks for time and glacier filtering
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
            ds_glogem.close()
            raise ValueError("No time steps found in the specified date range")
        
        # ✅ Load and subset data — select glaciers lazily first to avoid
        # loading the full ~3 GB array before subsetting.
        self.logger.info(f"Loading {self.irrigation_variable} data from NetCDF...")
        da = ds_glogem[self.irrigation_variable][:, glacier_indices]   # lazy glacier selection
        discharge_data = da[time_indices, :].values            # materialise only needed slice

        sim_times = all_nc_times[time_indices]
        sim_glacier_ids = all_glacier_ids[glacier_indices]

        self.logger.info(f"Subset data shape: {discharge_data.shape}")

        ds_glogem.close()
        
        # Replace NaN with 0
        discharge_data = np.nan_to_num(discharge_data, nan=0.0).astype(np.float32)
        
        n_sim_times = len(sim_times)
        result_sim = np.zeros((n_sim_times, num_hrus), dtype=np.float32)

        # Build glacier_id → NetCDF column index mapping
        sim_glacier_id_to_idx = {gid: i for i, gid in enumerate(sim_glacier_ids)}

        if self.icemelt_mode:
            # ── ICEMELT MODE: distribute melt to elevation-discretized ROCK HRUs ──
            self.logger.info("Icemelt mode: distributing melt via glacier_hru_overlap.csv...")
            overlap_df = self._load_glacier_hru_overlap()

            # glacier_hru_overlap.csv has glacier_numeric_id as int; glacier_id_mapping
            # uses numeric_id (padded string).  Build int→padded mapping.
            mapping_path = (
                Path(self.model_dir) / f"catchment_{self.gauge_id}"
                / "topo_files" / "glacier_id_mapping.csv"
            )
            mapping_df = pd.read_csv(mapping_path)
            int_to_padded = {}
            for _, row in mapping_df.iterrows():
                int_to_padded[int(row['numeric_id'])] = str(row['numeric_id']).zfill(5)

            # For each HRU, compute pixel-weighted average melt across overlapping glaciers
            for hru_id, grp in overlap_df.groupby('HRU_ID'):
                hru_0idx = int(hru_id) - 1  # 0-based
                if hru_0idx < 0 or hru_0idx >= num_hrus:
                    continue

                glacier_nc_indices = []
                pixel_weights = []
                for _, ov_row in grp.iterrows():
                    padded = int_to_padded.get(int(ov_row['glacier_numeric_id']))
                    if padded and padded in sim_glacier_id_to_idx:
                        glacier_nc_indices.append(sim_glacier_id_to_idx[padded])
                        pixel_weights.append(float(ov_row['n_pixels']))

                if not glacier_nc_indices:
                    continue

                weights = np.array(pixel_weights, dtype=np.float32)
                total_w = weights.sum()
                if total_w > 0:
                    data = discharge_data[:, glacier_nc_indices]  # (time, n_overlapping)
                    result_sim[:, hru_0idx] = (data * weights).sum(axis=1) / total_w

            n_assigned = int((result_sim != 0).any(axis=0).sum())
            self.logger.info(f"Icemelt mode: assigned melt to {n_assigned} HRUs")

        else:
            # ── AGGREGATED MODE: 2 HRUs (large + small) via Glacier_Cl field ──
            self.logger.info("Calculating area-weighted averages for glacier HRUs...")

            large_glacier_set = set(glacier_categories['large'])
            small_glacier_set = set(glacier_categories['small'])

            large_indices = []
            large_areas = []
            small_indices = []
            small_areas = []

            for g_idx, glacier_id in enumerate(sim_glacier_ids):
                area = area_map.get(glacier_id, 0.0)
                if glacier_id in large_glacier_set:
                    large_indices.append(g_idx)
                    large_areas.append(area)
                elif glacier_id in small_glacier_set:
                    small_indices.append(g_idx)
                    small_areas.append(area)

            self.logger.info(f"Found {len(large_indices)} large glaciers and {len(small_indices)} small glaciers in data")

            large_weighted = np.zeros(n_sim_times, dtype=np.float32)
            small_weighted = np.zeros(n_sim_times, dtype=np.float32)

            if large_indices:
                large_data = discharge_data[:, large_indices]
                large_weights = np.array(large_areas, dtype=np.float32)
                total_large_area = large_weights.sum()
                if total_large_area > 0:
                    large_weighted = (large_data * large_weights).sum(axis=1) / total_large_area
                    self.logger.info(f"Large glacier mean irrigation: {large_weighted.mean():.3f} mm/day")

            if small_indices:
                small_data = discharge_data[:, small_indices]
                small_weights = np.array(small_areas, dtype=np.float32)
                total_small_area = small_weights.sum()
                if total_small_area > 0:
                    small_weighted = (small_data * small_weights).sum(axis=1) / total_small_area
                    self.logger.info(f"Small glacier mean irrigation: {small_weighted.mean():.3f} mm/day")

            # Find glacier HRU indices via pipe-separated Glacier_Cl field
            glacier_hrus = hru_gdf[hru_gdf['Glacier_Cl'].notna()].copy()
            large_hru_idx = None
            small_hru_idx = None

            for idx, row in glacier_hrus.iterrows():
                glacier_id_str = row['Glacier_Cl']
                hru_id = int(row['HRU ID']) - 1  # 0-based index

                if isinstance(glacier_id_str, str) and '|' in glacier_id_str:
                    id_list = glacier_id_str.split('|')
                    numeric_ids = []
                    for full_id in id_list:
                        if rgi_region_code and full_id.startswith(rgi_region_code + '.'):
                            numeric_id = full_id.replace(rgi_region_code + '.', '')
                            numeric_ids.append(numeric_id)

                    has_large = any(nid in large_glacier_set for nid in numeric_ids)
                    has_small = any(nid in small_glacier_set for nid in numeric_ids)

                    if has_large and not has_small:
                        large_hru_idx = hru_id
                        self.logger.info(f"Found LARGE glacier HRU at index {hru_id} (HRU ID {row['HRU ID']})")
                    elif has_small and not has_large:
                        small_hru_idx = hru_id
                        self.logger.info(f"Found SMALL glacier HRU at index {hru_id} (HRU ID {row['HRU ID']})")

            if large_hru_idx is not None:
                result_sim[:, large_hru_idx] = large_weighted
                self.logger.info(f"Assigned large glacier irrigation to HRU index {large_hru_idx}")

            if small_hru_idx is not None:
                result_sim[:, small_hru_idx] = small_weighted
                self.logger.info(f"Assigned small glacier irrigation to HRU index {small_hru_idx}")

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
            'source': f'GloGEM {self.irrigation_variable}',
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
        Advanced validation of glacier IDs between catchment and GloGEM NetCDF.
        ✅ FIXED: Uses actual RGI IDs from mapping CSV (not reconstructed IDs)
        
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
            # ✅ NEW: Load mapping CSV to get TRUE RGI IDs (not reconstructed!)
            mapping_path = Path(self.model_dir) / f"catchment_{self.gauge_id}" / "topo_files" / "glacier_id_mapping.csv"
            
            if not mapping_path.exists():
                self.logger.warning(f"Glacier ID mapping not found: {mapping_path}")
                return results
            
            mapping_df = pd.read_csv(mapping_path)
            
            # Create mapping: numeric_id (padded) → RGI ID
            numeric_to_rgi = {}
            rgi_region_code = None
            
            for _, row in mapping_df.iterrows():
                numeric_id = str(row['numeric_id']).zfill(5)
                rgi_id = row['RGIId']
                numeric_to_rgi[numeric_id] = rgi_id
                
                # Extract region code from first RGI ID
                if rgi_region_code is None and rgi_id.startswith('RGI60-'):
                    rgi_region_code = rgi_id.split('.')[0]
            
            glacier_ids_needed = set(numeric_to_rgi.keys())
            
            self.logger.debug(f"Loaded {len(glacier_ids_needed)} glacier IDs from mapping CSV")
            
            if not glacier_ids_needed:
                self.logger.warning("No glacier IDs found in mapping CSV")
                return results
            
            # Load GloGEM NetCDF to get available glacier IDs
            nc_path = self._get_netcdf_path('Discharge')
            
            if not nc_path.exists():
                self.logger.warning(f"GloGEM Discharge NetCDF not found: {nc_path}")
                # All glaciers are missing
                for numeric_id in glacier_ids_needed:
                    results['missing_in_glogem'].append(numeric_to_rgi[numeric_id])
                return results
            
            ds = xr.open_dataset(nc_path)
            glacier_ids_glogem = set(ds.glacier_id.values.astype(str))
            ds.close()
            
            # Compare sets (using NUMERIC IDs)
            missing_in_glogem = glacier_ids_needed - glacier_ids_glogem
            missing_in_hru = glacier_ids_glogem - glacier_ids_needed
            matched = glacier_ids_needed.intersection(glacier_ids_glogem)
            
            # ✅ Convert to ACTUAL RGI IDs for reporting (not reconstructed!)
            for numeric_id in missing_in_glogem:
                results['missing_in_glogem'].append(numeric_to_rgi[numeric_id])
            
            for numeric_id in matched:
                results['matched'].append(numeric_to_rgi[numeric_id])
            
            # For IDs in GloGEM but not in HRU, we need to reconstruct
            # (we don't have the actual RGI ID for these)
            for numeric_id in missing_in_hru:
                full_id = f"{rgi_region_code}.{numeric_id}" if rgi_region_code else numeric_id
                results['missing_in_hru'].append(full_id)
            
            # Log summary
            self.logger.info(f"✅ Matched glaciers: {len(results['matched'])}")
            
            if results['missing_in_glogem']:
                self.logger.warning(f"⚠️  Missing in GloGEM: {len(results['missing_in_glogem'])}")
                self.logger.warning(f"   Sample missing glaciers:")
                for g_id in results['missing_in_glogem'][:5]:
                    self.logger.warning(f"     - {g_id}")
                if len(results['missing_in_glogem']) > 5:
                    self.logger.warning(f"     - ... and {len(results['missing_in_glogem'])-5} more")
            
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
        """
        Create map showing matched and missing glaciers.
        ✅ UPDATED: Handles pipe-separated glacier IDs, no labels on map
        """
        out_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', 'validation')
        out_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Add validation status to HRU geodataframe
            hru_gdf = hru_gdf.copy()
            hru_gdf['validation_status'] = 'non-glacier'
            
            # ✅ NEW: Handle pipe-separated glacier IDs
            for idx, row in hru_gdf.iterrows():
                glacier_id_str = row.get('Glacier_Cl', None)
                if pd.isna(glacier_id_str):
                    continue
                
                # Parse pipe-separated IDs
                if isinstance(glacier_id_str, str) and '|' in glacier_id_str:
                    id_list = glacier_id_str.split('|')
                else:
                    id_list = [glacier_id_str]
                
                # Check status of each glacier in this HRU
                has_matched = any(g_id in results['matched'] for g_id in id_list)
                has_missing = any(g_id in results['missing_in_glogem'] for g_id in id_list)
                
                if has_missing:
                    hru_gdf.at[idx, 'validation_status'] = 'missing_in_glogem'
                elif has_matched:
                    hru_gdf.at[idx, 'validation_status'] = 'matched'
            
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
            
            # ✅ REMOVED: No glacier ID labels on map (as requested)
            
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
            
            # --- 1. Load GloGEM data ---
            # Try CSV first (from .dat files), then fall back to NetCDF
            csv_filename = _GLOGEM_CSV[self.irrigation_variable]
            glogem_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', csv_filename)

            if glogem_path.exists():
                self.logger.info(f"Loading GloGEM {self.irrigation_variable} from CSV: {glogem_path}")
                glogem_df = pd.read_csv(glogem_path, dtype={'id': str})
                glogem_df['date'] = pd.to_datetime(glogem_df['date'])
                glogem_df['q'] = pd.to_numeric(glogem_df['q'], errors='coerce')
            else:
                # Try to load from NetCDF
                self.logger.info("CSV not found, loading from NetCDF...")
                nc_path = self._get_netcdf_path(self.irrigation_variable.capitalize())

                if not nc_path.exists():
                    self.logger.warning(f"Neither CSV nor NetCDF found. Skipping plots.")
                    return

                # Get glacier IDs from catchment
                glacier_ids_needed, rgi_region_code, glacier_categories = self._get_glacier_ids_from_catchment()

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
                full_data = ds[self.irrigation_variable].values
                discharge_data = full_data[np.ix_(time_indices, glacier_indices)]
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
            
            # --- 2. Load HRU data to get catchment and glacier areas ---
            hru_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
            hru_gdf = gpd.read_file(hru_path)
            hru_gdf = hru_gdf.sort_values(by='HRU_ID').reset_index(drop=True)
            
            # Get area column
            area_col = 'Area_km2' if 'Area_km2' in hru_gdf.columns else 'area'
            
            # ✅ STEP 1: Calculate TOTAL catchment area from entire HRU shapefile
            total_catchment_area_km2 = hru_gdf[area_col].sum()
            self.logger.info(f"Total catchment area (from HRU.shp): {total_catchment_area_km2:.2f} km²")
            
            # Auto-detect RGI region code
            rgi_region_code = None
            if 'Glacier_Cl' in hru_gdf.columns:
                glacier_series = hru_gdf['Glacier_Cl'].dropna()
                if not glacier_series.empty:
                    for glacier_id in glacier_series.unique():
                        if isinstance(glacier_id, str) and glacier_id.startswith('RGI60-'):
                            # Handle pipe-separated IDs
                            first_id = glacier_id.split('|')[0]
                            parts = first_id.split('.')
                            if len(parts) >= 2:
                                rgi_region_code = parts[0]
                                break
            
            self.logger.info(f"RGI region code: {rgi_region_code}")
            
            # Load glacier shapefile to get individual glacier sizes for classification
            glacier_shp_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'clipped_glacier.shp')
            glacier_size_map = {}  # Maps RGI ID -> glacier size in km²
            if glacier_shp_path.exists():
                glacier_gdf = gpd.read_file(glacier_shp_path)
                for _, row in glacier_gdf.iterrows():
                    rgi_id = row.get('RGIId', None)
                    area_km2 = row.get('Area', 0.0)
                    if rgi_id:
                        glacier_size_map[rgi_id] = area_km2
                self.logger.info(f"Loaded {len(glacier_size_map)} glacier sizes from shapefile")
            
            # ✅ STEP 2: Calculate glacier HRU areas in catchment by category (from HRU.shp)
            glacier_hrus = hru_gdf[hru_gdf['Glacier_Cl'].notna()].copy()
            
            all_glacier_area_in_catchment_km2 = 0.0
            large_glacier_area_in_catchment_km2 = 0.0
            small_glacier_area_in_catchment_km2 = 0.0
            
            for idx, row in glacier_hrus.iterrows():
                hru_area = row[area_col]  # Area of this glacier HRU in catchment
                glacier_id_str = row['Glacier_Cl']
                
                all_glacier_area_in_catchment_km2 += hru_area  # Total glacier coverage
                
                # Handle pipe-separated IDs (aggregated format)
                if isinstance(glacier_id_str, str) and '|' in glacier_id_str:
                    # Aggregated HRU contains multiple glaciers
                    id_list = glacier_id_str.split('|')
                    
                    # Check if this is large or small HRU by checking first glacier
                    first_glacier_size = glacier_size_map.get(id_list[0], 0.0)
                    if first_glacier_size >= 2.0:
                        large_glacier_area_in_catchment_km2 += hru_area
                    else:
                        small_glacier_area_in_catchment_km2 += hru_area
                
                elif isinstance(glacier_id_str, str):
                    # Single glacier HRU
                    glacier_size = glacier_size_map.get(glacier_id_str, 0.0)
                    if glacier_size >= 2.0:
                        large_glacier_area_in_catchment_km2 += hru_area
                    else:
                        small_glacier_area_in_catchment_km2 += hru_area
            
            # ✅ STEP 3: Calculate glacier fractions of total catchment
            glacier_fraction_all = all_glacier_area_in_catchment_km2 / total_catchment_area_km2
            glacier_fraction_large = large_glacier_area_in_catchment_km2 / total_catchment_area_km2
            glacier_fraction_small = small_glacier_area_in_catchment_km2 / total_catchment_area_km2
            
            self.logger.info(f"Glacier coverage in catchment:")
            self.logger.info(f"  All glaciers: {all_glacier_area_in_catchment_km2:.2f} km² ({glacier_fraction_all*100:.2f}%)")
            self.logger.info(f"  Large glaciers (>=2km²): {large_glacier_area_in_catchment_km2:.2f} km² ({glacier_fraction_large*100:.2f}%)")
            self.logger.info(f"  Small glaciers (<2km²): {small_glacier_area_in_catchment_km2:.2f} km² ({glacier_fraction_small*100:.2f}%)")
            
            # Create mapping for GloGEM data (numeric ID -> glacier size for classification)
            area_map = {}
            for rgi_id, size in glacier_size_map.items():
                if rgi_region_code and rgi_id.startswith(rgi_region_code + '.'):
                    numeric_id = rgi_id.replace(rgi_region_code + '.', '')
                    area_map[numeric_id] = size
            
            # ✅ STEP 4: Add individual glacier areas to GloGEM data for weighted averaging
            glogem_df['area'] = glogem_df['id'].map(area_map)
            glogem_df['area'] = pd.to_numeric(glogem_df['area'], errors='coerce')
            
            # Classify glaciers by size for GloGEM data processing
            glogem_df['size_category'] = glogem_df['area'].apply(
                lambda x: 'large' if x >= 2.0 else 'small' if pd.notna(x) else None
            )
            
            # ✅ DEBUG: Check area mapping
            matched_areas = glogem_df['area'].notna().sum()
            total_records = len(glogem_df)
            self.logger.info(f"Area mapping for GloGEM: {matched_areas}/{total_records} records have area info")
            
            n_large = (glogem_df['size_category'] == 'large').sum()
            n_small = (glogem_df['size_category'] == 'small').sum()
            self.logger.info(f"Size classification in GloGEM: {n_large} large, {n_small} small glacier-day records")
            
            # Filter out records without area
            glogem_df = glogem_df[glogem_df['area'].notna()].copy()
            
            if glogem_df.empty:
                self.logger.warning("No glaciers with area information. Skipping plots.")
                return
            
            # --- 3. Calculate area-weighted glacier runoff per day (mm/day over glacier area) ---
            def calc_weighted_avg(group):
                total_area = group['area'].sum()
                if total_area > 0:
                    return (group['q'] * group['area']).sum() / total_area
                return 0.0
            
            # ✅ Calculate for ALL glaciers
            daily_all = glogem_df.groupby('date').apply(calc_weighted_avg, include_groups=False).reset_index()
            daily_all.columns = ['date', 'glacier_runoff_all_glacier_area']
            
            # ✅ Calculate for LARGE glaciers
            glogem_large = glogem_df[glogem_df['size_category'] == 'large']
            if not glogem_large.empty:
                daily_large = glogem_large.groupby('date').apply(calc_weighted_avg, include_groups=False).reset_index()
                daily_large.columns = ['date', 'glacier_runoff_large_glacier_area']
            else:
                daily_large = pd.DataFrame({'date': daily_all['date'], 'glacier_runoff_large_glacier_area': 0.0})
            
            # ✅ Calculate for SMALL glaciers
            glogem_small = glogem_df[glogem_df['size_category'] == 'small']
            if not glogem_small.empty:
                daily_small = glogem_small.groupby('date').apply(calc_weighted_avg, include_groups=False).reset_index()
                daily_small.columns = ['date', 'glacier_runoff_small_glacier_area']
            else:
                daily_small = pd.DataFrame({'date': daily_all['date'], 'glacier_runoff_small_glacier_area': 0.0})
            
            # Merge all three
            daily_glacier = daily_all.merge(daily_large, on='date', how='outer').merge(daily_small, on='date', how='outer')
            
            # ✅ NEW: Scale to catchment area using respective glacier fractions
            daily_glacier['glacier_runoff_all'] = daily_glacier['glacier_runoff_all_glacier_area'] * glacier_fraction_all
            daily_glacier['glacier_runoff_large'] = daily_glacier['glacier_runoff_large_glacier_area'] * glacier_fraction_large
            daily_glacier['glacier_runoff_small'] = daily_glacier['glacier_runoff_small_glacier_area'] * glacier_fraction_small
            
            # ✅ DEBUG: Check daily data
            self.logger.info(f"Daily glacier runoff calculated for {len(daily_glacier)} days")
            self.logger.info(f"Date range: {daily_glacier['date'].min()} to {daily_glacier['date'].max()}")
            self.logger.info(f"Mean runoff over glacier area:")
            self.logger.info(f"  All glaciers: {daily_glacier['glacier_runoff_all_glacier_area'].mean():.3f} mm/day")
            self.logger.info(f"  Large glaciers: {daily_glacier['glacier_runoff_large_glacier_area'].mean():.3f} mm/day")
            self.logger.info(f"  Small glaciers: {daily_glacier['glacier_runoff_small_glacier_area'].mean():.3f} mm/day")
            self.logger.info(f"Mean runoff over catchment area (scaled):")
            self.logger.info(f"  All glaciers: {daily_glacier['glacier_runoff_all'].mean():.3f} mm/day")
            self.logger.info(f"  Large glaciers: {daily_glacier['glacier_runoff_large'].mean():.3f} mm/day")
            self.logger.info(f"  Small glaciers: {daily_glacier['glacier_runoff_small'].mean():.3f} mm/day")
            
            # --- 4. Load observed streamflow from Q_daily.rvt ---
            # Check shared data_obs first, then model-specific dir
            q_file = self.shared_data_dir / 'Q_daily.rvt'
            if not q_file.exists():
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
                            
                            # Convert from m³/s to mm/day using total catchment area
                            obs_series_mm = obs_series * 86400 / (total_catchment_area_km2 * 1e6) * 1000
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
            var_label = _GLOGEM_LABEL[self.irrigation_variable]
            fig, ax = plt.subplots(figsize=(14, 6))

            # ✅ Convert dates to datetime for proper plotting
            plot_dates = pd.to_datetime(plot_df['date'])

            # ✅ NEW: Plot all 3 glacier categories (catchment-normalized)
            ax.plot(plot_dates, plot_df['glacier_runoff_all'].values,
                label=f'All Glaciers — {var_label} ({glacier_fraction_all*100:.1f}% of catchment)',
                color='blue', alpha=0.8, linewidth=1.5)

            ax.plot(plot_dates, plot_df['glacier_runoff_large'].values,
                label=f'Large Glaciers >=2 km2 — {var_label} ({glacier_fraction_large*100:.1f}% of catchment)',
                color='darkblue', alpha=0.7, linewidth=1, linestyle='--')

            ax.plot(plot_dates, plot_df['glacier_runoff_small'].values,
                label=f'Small Glaciers <2 km2 — {var_label} ({glacier_fraction_small*100:.1f}% of catchment)',
                color='cyan', alpha=0.7, linewidth=1, linestyle=':')

            # Plot observed streamflow if available
            if 'observed_streamflow_mm' in plot_df.columns:
                valid_mask = plot_df['observed_streamflow_mm'].notna()
                if valid_mask.any():
                    ax.plot(plot_dates[valid_mask], plot_df.loc[valid_mask, 'observed_streamflow_mm'].values,
                        label='Observed Streamflow',
                        color='black', linewidth=1)
                    self.logger.info(f"Plotted {valid_mask.sum()} days of observed data")

            ax.set_title(f'{var_label} vs Observed Streamflow (Catchment-Normalized) - Gauge {self.gauge_id}',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel(f'{var_label} (mm/day over catchment area)')
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
            
            # ✅ Calculate monthly means for all 3 categories
            monthly_all = regime_df.groupby('month')['glacier_runoff_all'].mean().reindex(range(1, 13), fill_value=0)
            monthly_large = regime_df.groupby('month')['glacier_runoff_large'].mean().reindex(range(1, 13), fill_value=0)
            monthly_small = regime_df.groupby('month')['glacier_runoff_small'].mean().reindex(range(1, 13), fill_value=0)
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            x = np.arange(1, 13)
            
            self.logger.info(f"Monthly glacier runoff (all): {monthly_all.values}")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # ✅ NEW: Plot all 3 categories
            ax.plot(x, monthly_all.values, 'bo-', linewidth=2.5, markersize=8, 
                   label='All Glaciers', alpha=0.8)
            ax.plot(x, monthly_large.values, 's--', color='darkblue', linewidth=2, markersize=6, 
                   label='Large Glaciers (>=2 km²)', alpha=0.7)
            ax.plot(x, monthly_small.values, 'd:', color='cyan', linewidth=2, markersize=6, 
                   label='Small Glaciers (<2 km²)', alpha=0.7)
            
            # Plot observed if available
            if 'observed_streamflow_mm' in regime_df.columns:
                monthly_obs = regime_df.groupby('month')['observed_streamflow_mm'].mean()
                monthly_obs = monthly_obs.reindex(range(1, 13), fill_value=np.nan)
                
                if monthly_obs.notna().any():
                    ax.plot(x, monthly_obs.values, 'ko-', linewidth=2, markersize=8, 
                        label='Observed Streamflow')
            
            ax.set_title(f'Monthly Regime: {var_label} (Catchment-Normalized) vs Observed\nGauge {self.gauge_id}',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel(f'Mean {var_label} (mm/day over catchment area)')
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
            self.logger.info(f"Catchment area: {total_catchment_area_km2:.2f} km²")
            self.logger.info(f"All glacier area: {all_glacier_area_in_catchment_km2:.2f} km² ({glacier_fraction_all*100:.1f}%)")
            self.logger.info(f"Large glacier area: {large_glacier_area_in_catchment_km2:.2f} km² ({glacier_fraction_large*100:.1f}%)")
            self.logger.info(f"Small glacier area: {small_glacier_area_in_catchment_km2:.2f} km² ({glacier_fraction_small*100:.1f}%)")
            self.logger.info(f"\nMean glacier runoff (over catchment area):")
            self.logger.info(f"  All glaciers: {daily_glacier['glacier_runoff_all'].mean():.3f} mm/day")
            self.logger.info(f"  Large glaciers: {daily_glacier['glacier_runoff_large'].mean():.3f} mm/day")
            self.logger.info(f"  Small glaciers: {daily_glacier['glacier_runoff_small'].mean():.3f} mm/day")
            
            if 'observed_streamflow_mm' in plot_df.columns and plot_df['observed_streamflow_mm'].notna().any():
                mean_obs = plot_df['observed_streamflow_mm'].mean()
                mean_glacier_all = daily_glacier['glacier_runoff_all'].mean()
                mean_glacier_large = daily_glacier['glacier_runoff_large'].mean()
                mean_glacier_small = daily_glacier['glacier_runoff_small'].mean()
                self.logger.info(f"\nMean observed streamflow: {mean_obs:.3f} mm/day")
                if mean_obs > 0:
                    contrib_all = (mean_glacier_all / mean_obs) * 100
                    contrib_large = (mean_glacier_large / mean_obs) * 100
                    contrib_small = (mean_glacier_small / mean_obs) * 100
                    self.logger.info(f"Glacier contribution to streamflow:")
                    self.logger.info(f"  All glaciers: {contrib_all:.1f}%")
                    self.logger.info(f"  Large glaciers: {contrib_large:.1f}%")
                    self.logger.info(f"  Small glaciers: {contrib_small:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error creating glacier runoff comparison plots: {e}")
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
    
class MultiSubbasinGloGEMProcessor:
    """
    GloGEM processor for multi-subbasin configurations.

    Reads per-subbasin glacier_id_mapping.csv files, computes area-weighted melt
    for large and small glacier HRUs in each subbasin from shared basin-wide NetCDF
    files, and writes one combined irrigation.nc covering all HRUs.
    """

    def __init__(self, namelist_path: Union[str, Path]) -> None:
        self.namelist_path = Path(namelist_path)
        with open(self.namelist_path, 'r') as f:
            config = yaml.safe_load(f)

        self.gauge_id = config['gauge_id']
        self.main_dir = Path(config['main_dir'])
        self.model_type = config['model_type']
        self.start_date = config['start_date']
        self.end_date = config['end_date']
        self.warm_up_date = config.get('warm_up_date', None)
        self.glogem_scenario = config.get('glogem_scenario', 'ssp126')
        self.basin_name = config.get('basin_name', 'Indus')
        self.glogem_model = config.get('glogem_model', None)
        self.debug = config.get('debug', False)

        self.coupled = config.get('coupled', False)
        self.irrigation_variable = config.get('irrigation_variable', 'discharge').lower()
        if self.irrigation_variable not in _VALID_IRRIGATION_VARS:
            raise ValueError(
                f"irrigation_variable must be one of {_VALID_IRRIGATION_VARS}, "
                f"got '{self.irrigation_variable}'"
            )

        self.model_dir = self.main_dir / config.get('config_dir')
        self.glogem_dir = config.get('glogem_dir')
        if self.glogem_dir:
            self.glogem_dir = Path(self.main_dir, self.glogem_dir.format(gauge_id=self.gauge_id))

        self.topo_dir = self.model_dir / f'catchment_{self.gauge_id}' / 'topo_files'
        self.shared_data_dir = self.model_dir / f'catchment_{self.gauge_id}' / 'data_obs'
        self.shared_data_dir.mkdir(parents=True, exist_ok=True)

        self.subbasin_configs = config.get('subbasins', [])

        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.DEBUG if self.debug else logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('MultiSubbasinGloGEMProcessor')

    @property
    def icemelt_mode(self) -> bool:
        """True when coupled + icemelt: HRUs are ROCK with elevation discretization."""
        return self.coupled and self.irrigation_variable == 'icemelt'

    def _get_netcdf_path(self, component: str) -> Path:
        """Return path to a GloGEM NetCDF file (same naming convention as GloGEMProcessor)."""
        if self.glogem_model:
            filename = f'GloGEM_{component}_{self.basin_name}_{self.glogem_model}_{self.glogem_scenario}.nc'
        else:
            filename = f'GloGEM_{component}_{self.basin_name}_{self.glogem_scenario}.nc'
        return self.glogem_dir / filename

    def process_all(self, force_reprocess: bool = False) -> dict:
        """
        Process GloGEM data for all subbasins and write a combined irrigation.nc.

        Returns dict with {'skipped': bool, 'n_subbasins': int, ...}.
        """
        output_nc = self.shared_data_dir / 'irrigation.nc'
        output_gw = self.shared_data_dir / 'GridWeights_Irrigation.txt'

        if output_nc.exists() and output_gw.exists() and not force_reprocess:
            self.logger.info("✅ irrigation.nc + GridWeights already exist — skipping.")
            return {'skipped': True}

        # --- Load merged HRU shapefile ---
        hru_shp = self.topo_dir / 'HRU.shp'
        if not hru_shp.exists():
            raise FileNotFoundError(f"Merged HRU.shp not found: {hru_shp}")

        hru_gdf = gpd.read_file(hru_shp)
        hru_gdf = hru_gdf.sort_values(by='HRU_ID').reset_index(drop=True)
        num_hrus = len(hru_gdf)
        self.logger.info(f"Loaded {num_hrus} HRUs from {hru_shp}")

        # --- Date ranges ---
        start_date_for_file = pd.to_datetime(self.warm_up_date if self.warm_up_date else self.start_date)
        end_date_for_file = pd.to_datetime(self.end_date)
        simulation_start = pd.to_datetime(self.start_date)
        self.logger.info(f"Date range: {start_date_for_file.date()} → {end_date_for_file.date()}")

        # --- Load GloGEM NetCDF ONCE for configured irrigation variable ---
        nc_path = self._get_netcdf_path(self.irrigation_variable.capitalize())
        if not nc_path.exists():
            raise FileNotFoundError(f"GloGEM {self.irrigation_variable} NetCDF not found: {nc_path}")

        self.logger.info(f"Loading GloGEM {self.irrigation_variable}: {nc_path}")
        ds_glogem = xr.open_dataset(nc_path)
        all_nc_times = pd.to_datetime(ds_glogem.time.values)
        all_glacier_ids = ds_glogem.glacier_id.values.astype(str)

        # Select simulation time window from NetCDF
        time_mask = (all_nc_times >= simulation_start) & (all_nc_times <= end_date_for_file)
        time_indices = np.where(time_mask)[0]
        if len(time_indices) == 0:
            ds_glogem.close()
            raise ValueError("No NetCDF time steps in simulation range.")

        sim_times = all_nc_times[time_indices]
        self.logger.info(f"Loaded {len(time_indices)} simulation time steps from NetCDF")

        # Load the configured variable — select sim-window time steps lazily before materialising.
        # All glacier columns are kept here since different subbasins use different subsets.
        full_discharge = ds_glogem[self.irrigation_variable][time_indices, :].values   # (n_sim, n_glaciers)
        ds_glogem.close()
        full_discharge = np.nan_to_num(full_discharge, nan=0.0).astype(np.float32)

        # --- Pre-allocate result (simulation window only; warm-up prepended later) ---
        n_sim = len(time_indices)
        result_sim = np.zeros((n_sim, num_hrus), dtype=np.float32)

        processed_subbasins = []

        # --- Per-subbasin processing ---
        for sb_config in self.subbasin_configs:
            sb_id = sb_config['id']
            self.logger.info(f"Processing subbasin {sb_id} ({sb_config.get('name', '')})")

            # CatchmentProcessor.get_path() appends 'topo_files/' to model_dir, and
            # each subbasin's model_dir is topo_dir/subbasin_{id}, so the actual
            # save path is topo_dir/subbasin_{id}/topo_files/glacier_id_mapping.csv
            mapping_path = self.topo_dir / f'subbasin_{sb_id}' / 'topo_files' / 'glacier_id_mapping.csv'
            if not mapping_path.exists():
                self.logger.warning(
                    f"  ⚠️ glacier_id_mapping.csv not found for subbasin {sb_id} — skipping\n"
                    f"      Looked at: {mapping_path}"
                )
                continue

            mapping_df = pd.read_csv(mapping_path)
            if mapping_df.empty:
                self.logger.warning(f"  ⚠️ Empty mapping for subbasin {sb_id} — skipping")
                continue

            # Build padded-numeric → area map and large/small sets
            large_glacier_set = set()
            small_glacier_set = set()
            area_map = {}
            int_to_padded = {}
            rgi_region_code = None

            for _, row in mapping_df.iterrows():
                numeric_id = str(row['numeric_id']).zfill(5)
                rgi_id = str(row['RGIId'])
                area_km2 = float(row['area_km2'])
                is_large = bool(row['is_large'])

                area_map[numeric_id] = area_km2
                int_to_padded[int(row['numeric_id'])] = numeric_id
                if is_large:
                    large_glacier_set.add(numeric_id)
                else:
                    small_glacier_set.add(numeric_id)

                if rgi_region_code is None and '.' in rgi_id:
                    rgi_region_code = rgi_id.rsplit('.', 1)[0]  # e.g. 'RGI60-14'

            all_sb_ids = large_glacier_set | small_glacier_set
            if not all_sb_ids:
                self.logger.warning(f"  No glacier IDs for subbasin {sb_id}")
                continue

            # Match against NetCDF glacier IDs
            glacier_id_to_ncidx = {gid: i for i, gid in enumerate(all_glacier_ids)}
            matching_ids = [gid for gid in all_sb_ids if gid in glacier_id_to_ncidx]
            if not matching_ids:
                self.logger.warning(f"  No matching glaciers in NetCDF for subbasin {sb_id}")
                continue

            self.logger.info(f"  Matched {len(matching_ids)}/{len(all_sb_ids)} glaciers in NetCDF")

            if self.icemelt_mode:
                # ── ICEMELT MODE: distribute via per-subbasin overlap CSV ──
                overlap_path = self.topo_dir / f'subbasin_{sb_id}' / 'topo_files' / 'glacier_hru_overlap.csv'
                if not overlap_path.exists():
                    self.logger.warning(f"  glacier_hru_overlap.csv not found for subbasin {sb_id} — skipping")
                    continue

                overlap_df = pd.read_csv(overlap_path)
                for hru_id, grp in overlap_df.groupby('HRU_ID'):
                    hru_0idx = int(hru_id) - 1
                    if hru_0idx < 0 or hru_0idx >= num_hrus:
                        continue

                    glacier_nc_indices = []
                    pixel_weights = []
                    for _, ov_row in grp.iterrows():
                        padded = int_to_padded.get(int(ov_row['glacier_numeric_id']))
                        if padded and padded in glacier_id_to_ncidx:
                            glacier_nc_indices.append(glacier_id_to_ncidx[padded])
                            pixel_weights.append(float(ov_row['n_pixels']))

                    if not glacier_nc_indices:
                        continue

                    weights = np.array(pixel_weights, dtype=np.float32)
                    total_w = weights.sum()
                    if total_w > 0:
                        data = full_discharge[:, glacier_nc_indices]
                        result_sim[:, hru_0idx] = (data * weights).sum(axis=1) / total_w

                n_assigned = int((result_sim != 0).any(axis=0).sum())
                self.logger.info(f"  Icemelt mode: assigned melt to {n_assigned} HRUs for subbasin {sb_id}")

            else:
                # ── AGGREGATED MODE: 2 HRUs (large + small) via Glacier_Cl field ──
                large_nc_idx, large_areas = [], []
                small_nc_idx, small_areas = [], []
                for gid in matching_ids:
                    nc_idx = glacier_id_to_ncidx[gid]
                    area = area_map.get(gid, 0.0)
                    if gid in large_glacier_set:
                        large_nc_idx.append(nc_idx)
                        large_areas.append(area)
                    else:
                        small_nc_idx.append(nc_idx)
                        small_areas.append(area)

                large_weighted = np.zeros(n_sim, dtype=np.float32)
                small_weighted = np.zeros(n_sim, dtype=np.float32)

                if large_nc_idx:
                    data = full_discharge[:, large_nc_idx]
                    weights = np.array(large_areas, dtype=np.float32)
                    total = weights.sum()
                    if total > 0:
                        large_weighted = (data * weights).sum(axis=1) / total
                        self.logger.info(f"  Large glacier mean melt: {large_weighted.mean():.3f} mm/day")

                if small_nc_idx:
                    data = full_discharge[:, small_nc_idx]
                    weights = np.array(small_areas, dtype=np.float32)
                    total = weights.sum()
                    if total > 0:
                        small_weighted = (data * weights).sum(axis=1) / total
                        self.logger.info(f"  Small glacier mean melt: {small_weighted.mean():.3f} mm/day")

                # Save per-subbasin intermediate CSV
                sb_out_dir = self.topo_dir / f'subbasin_{sb_id}' / 'topo_files'
                melt_df = pd.DataFrame({
                    'date': pd.to_datetime(sim_times).strftime('%Y-%m-%d'),
                    'large_melt_mm_day': large_weighted,
                    'small_melt_mm_day': small_weighted,
                })
                melt_csv = sb_out_dir / 'GloGEM_catchment_melt.csv'
                melt_df.to_csv(melt_csv, index=False)
                self.logger.info(f"  Saved per-subbasin melt CSV: {melt_csv}")

                # Find large/small HRU indices in merged HRU.shp via Glacier_Cl field
                large_hru_idx = None
                small_hru_idx = None

                glacier_hrus = hru_gdf[hru_gdf['Glacier_Cl'].notna()]
                for _, hru_row in glacier_hrus.iterrows():
                    glacier_cl = str(hru_row['Glacier_Cl'])
                    hru_0idx = int(hru_row['HRU_ID']) - 1

                    parts = glacier_cl.split('|')
                    numeric_ids = []
                    for full_id in parts:
                        full_id = full_id.strip()
                        if rgi_region_code and full_id.startswith(rgi_region_code + '.'):
                            numeric_ids.append(full_id.replace(rgi_region_code + '.', '').zfill(5))
                        elif '.' in full_id:
                            numeric_ids.append(full_id.rsplit('.', 1)[-1].zfill(5))

                    if not numeric_ids:
                        continue

                    if any(nid in large_glacier_set for nid in numeric_ids):
                        large_hru_idx = hru_0idx
                        self.logger.info(f"  → LARGE glacier HRU index {hru_0idx} (HRU_ID {hru_row['HRU_ID']})")
                    elif any(nid in small_glacier_set for nid in numeric_ids):
                        small_hru_idx = hru_0idx
                        self.logger.info(f"  → SMALL glacier HRU index {hru_0idx} (HRU_ID {hru_row['HRU_ID']})")

                if large_hru_idx is not None:
                    result_sim[:, large_hru_idx] = large_weighted
                elif large_nc_idx:
                    self.logger.warning(f"  Could not find LARGE glacier HRU in HRU.shp for subbasin {sb_id}")

                if small_hru_idx is not None:
                    result_sim[:, small_hru_idx] = small_weighted
                elif small_nc_idx:
                    self.logger.warning(f"  Could not find SMALL glacier HRU in HRU.shp for subbasin {sb_id}")

            processed_subbasins.append(sb_id)

        del full_discharge

        # --- Warm-up period: prepend repeated first year ---
        if self.warm_up_date:
            warmup_days = (simulation_start - start_date_for_file).days
            if warmup_days > 0:
                first_year_end = simulation_start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
                sim_times_np = np.array(pd.to_datetime(sim_times))
                first_year_mask = sim_times_np <= np.datetime64(first_year_end)
                first_year_indices = np.where(first_year_mask)[0]
                if len(first_year_indices) == 0:
                    first_year_indices = np.arange(min(365, n_sim))

                first_year_data = result_sim[first_year_indices, :]
                n_rep = max(1, int(np.ceil(warmup_days / len(first_year_data))))
                warmup_data = np.tile(first_year_data, (n_rep, 1))[:warmup_days, :]
                warmup_times = pd.date_range(start=start_date_for_file, periods=warmup_days, freq='D')

                result_array = np.vstack([warmup_data, result_sim])
                full_times = warmup_times.append(pd.DatetimeIndex(sim_times))
                self.logger.info(f"Warm-up prepended: {warmup_days} days")
            else:
                result_array = result_sim
                full_times = pd.DatetimeIndex(sim_times)
        else:
            result_array = result_sim
            full_times = pd.DatetimeIndex(sim_times)

        # Ensure contiguous date range; fill gaps with 0
        full_date_range = pd.date_range(start=start_date_for_file, end=end_date_for_file, freq='D')
        if len(full_times) != len(full_date_range):
            full_times_norm = pd.to_datetime(full_times).normalize()
            time_to_idx = {t: i for i, t in enumerate(full_times_norm)}
            final_array = np.zeros((len(full_date_range), num_hrus), dtype=np.float32)
            for i, date in enumerate(full_date_range):
                if date in time_to_idx:
                    final_array[i, :] = result_array[time_to_idx[date], :]
            result_array = final_array

        # --- Write irrigation.nc ---
        self.logger.info(f"Writing irrigation.nc ({result_array.shape})...")
        x_values = np.arange(1, num_hrus + 1)
        y_values = np.arange(1, 2)

        ds = xr.Dataset(
            {'data': (['time', 'x', 'y'], result_array.reshape(len(full_date_range), -1, 1))},
            coords={'time': full_date_range, 'x': x_values, 'y': y_values}
        )

        if 'Elev_Mean' in hru_gdf.columns:
            ds['elevation'] = xr.DataArray(
                hru_gdf['Elev_Mean'].values.reshape(-1, 1),
                dims=['x', 'y'],
                coords={'x': ds['x'], 'y': ds['y']}
            )

        ds.attrs.update({
            'title': f'Glacier melt irrigation for multi-subbasin catchment {self.gauge_id}',
            'source': f'GloGEM {self.irrigation_variable}',
            'n_subbasins': len(processed_subbasins),
            'n_hrus': num_hrus,
        })
        ds.to_netcdf(output_nc)
        self.logger.info(f"✅ Saved irrigation.nc: {output_nc}")

        # --- Write GridWeights_Irrigation.txt ---
        with open(output_gw, 'w') as f:
            f.write('# ---------------------------------------------- \n')
            f.write('# Raven GridWeights File for Irrigation Forcing\n')
            f.write('# ---------------------------------------------- \n\n')
            f.write(':GridWeights\n')
            f.write(f'   :NumberHRUs       {num_hrus}\n')
            f.write(f'   :NumberGridCells  {num_hrus}\n')
            f.write('   # [HRU ID] [Cell #] [w_kl]\n')
            for hru_id in range(1, num_hrus + 1):
                f.write(f'   {hru_id}   {hru_id - 1}   1.0\n')
            f.write(':EndGridWeights\n')
        self.logger.info(f"✅ Saved GridWeights_Irrigation.txt: {output_gw}")

        glacier_hrus_count = int((result_array != 0).any(axis=0).sum())
        self.logger.info(f"Summary: {glacier_hrus_count}/{num_hrus} HRUs have non-zero glacier melt")

        return {
            'skipped': False,
            'n_subbasins': len(processed_subbasins),
            'processed_subbasins': processed_subbasins,
        }