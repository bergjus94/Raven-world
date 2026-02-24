#### This file contains all functions for plotting and analyzing ERA5-Land meteorological data
#### Updated for plotting and time series analysis with namelist configuration
#### Justine Berg

#--------------------------------------------------------------------------------
############################### import packages #################################
#--------------------------------------------------------------------------------

import geopandas as gpd
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import seaborn as sns
import logging
import yaml
from typing import Dict, List, Union, Optional, Any, Tuple
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings('ignore')

#--------------------------------------------------------------------------------
############################### MeteoProcessor Class ############################
#--------------------------------------------------------------------------------

class ERA5LandAnalyzer:
    """
    A class for analyzing and plotting ERA5-Land meteorological data
    """
    def __init__(self, namelist_path: Union[str, Path], force_reprocess: bool = False) -> None:
        """
        Initialize the ERA5-Land data analyzer
        
        Parameters
        ----------
        namelist_path : str or Path
            Path to the namelist YAML configuration file
        force_reprocess : bool, optional
            If True, reprocess files even if they already exist (default: False)
        """
        # Store the force_reprocess flag
        self.force_reprocess = force_reprocess
        
        # Load configuration from namelist directly
        namelist_path = Path(namelist_path)
        
        if not namelist_path.exists():
            raise FileNotFoundError(f"Namelist file not found: {namelist_path}")
        
        with open(namelist_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Store configuration parameters
        self.main_dir = Path(self.config.get('main_dir'))
        self.gauge_id = self.config.get('gauge_id')
        self.basin_id = self.config.get('basin_id', self.gauge_id)
        self.start_date = pd.to_datetime(self.config.get('start_date'))
        self.end_date = pd.to_datetime(self.config.get('end_date'))
        self.model_type = self.config.get('model_type')
        self.debug = self.config.get('debug', False)
        self.coupled = self.config.get('coupled', False)
        self.model_dir = self.main_dir / self.config.get('config_dir')
        
        # ✅ NEW: Setup directories using basin_id for meteo data
        meteo_dir_template = self.config.get('meteo_dir', '01_data/meteo/gauge_{gauge_id}')
        self.era5_data_dir = Path(meteo_dir_template.format(basin_id=self.basin_id, gauge_id=self.basin_id))
        
        # Make absolute path if needed
        if not self.era5_data_dir.is_absolute():
            self.era5_data_dir = self.main_dir / self.era5_data_dir
        
        # ✅ NEW: Define SHARED catchment-level directories (primary storage)
        self.shared_data_dir = self.model_dir / f'catchment_{self.gauge_id}' / 'data_obs'
        self.shared_plots_dir = self.model_dir / f'catchment_{self.gauge_id}' / 'plots'
        
        # ✅ NEW: Define MODEL-SPECIFIC directories (for backward compatibility)
        self.model_data_dir = self.model_dir / f'catchment_{self.gauge_id}' / self.model_type / 'data_obs'
        
        # ✅ Use shared directories as primary output locations
        self.output_path = self.shared_data_dir
        self.plots_dir = self.shared_plots_dir
        
        # Create plots directories
        self.spatial_plots_dir = self.plots_dir / 'spatial_overview'
        self.timeseries_plots_dir = self.plots_dir / 'time_series'
        
        # Create all directories
        self.spatial_plots_dir.mkdir(parents=True, exist_ok=True)
        self.timeseries_plots_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.model_data_dir.mkdir(parents=True, exist_ok=True)  # ✅ Create model-specific dir too
        
        # Keep the old attribute name for backward compatibility
        self.processed_data_dir = self.output_path
        
        # ERA5-Land variable information
        self.era5_variables = {
            't2m': {
                'name': '2m Temperature',
                'units': '°C',
                'cmap': 'RdYlBu_r',
                'convert_kelvin': True
            },
            'tp': {
                'name': 'Total Precipitation',
                'units': 'mm',
                'cmap': 'Blues',
                'convert_kelvin': False
            },
            'pev': {
                'name': 'Potential Evapotranspiration',
                'units': 'mm',
                'cmap': 'Oranges',
                'convert_kelvin': False
            }
        }
        
        # ✅ MOVED UP: Setup logger FIRST
        self.logger = self._setup_logger()
        
        # ✅ NOW: Load warm-up date AFTER logger is created
        if 'warm_up_date' in self.config:
            self.warmup_date = pd.to_datetime(self.config.get('warm_up_date'))
            self.logger.info(f"Warm-up period configured: {self.warmup_date.date()} to {(self.start_date - pd.Timedelta(days=1)).date()}")
        else:
            self.warmup_date = None
            self.logger.debug("No warm-up period configured")
        
        # ✅ NEW: Load catchment shapefile for clipping
        self.catchment_extent = self._load_catchment_shapefile()
        
        # Log the output directory
        self.logger.info(f"Processing for gauge {self.gauge_id} using basin {self.basin_id} meteo data")
        self.logger.info(f"Model type: {self.model_type}")
        self.logger.info(f"Coupled mode: {self.coupled}")
        self.logger.info(f"Model directory: {self.model_dir}")
        self.logger.info(f"ERA5 data directory: {self.era5_data_dir}")
        self.logger.info(f"Plots will be saved to: {self.plots_dir}")
        self.logger.info(f"Processed meteo files will be saved to: {self.output_path}")
        
        # Check for existing files before processing
        existing_files = self._check_existing_files()
        existing_count = sum(existing_files.values())
        total_expected = len(existing_files)
        
        if existing_count == total_expected and not force_reprocess:
            self.logger.info(f"🎉 All {total_expected} processed files already exist!")
            self.logger.info("⏭️ Skipping processing. Set force_reprocess=True to reprocess anyway.")
            
            # Build list of existing files
            expected_files = {
                'temperature_mean': 'era5_land_temp_mean.nc',
                'temperature_min': 'era5_land_temp_min.nc',
                'temperature_max': 'era5_land_temp_max.nc',
                'precipitation': 'era5_land_precip.nc',
                'potential_evaporation': 'era5_land_pet.nc'
            }
            
            self.processed_files = []
            for file_type, exists in existing_files.items():
                if exists:
                    self.processed_files.append(self.output_path / expected_files[file_type])
            
            # ✅ NEW: Copy existing files to model-specific directory
            self.logger.info("📋 Copying existing files to model-specific directory...")
            self._copy_to_model_directory(self.processed_files)
                    
        elif existing_count > 0 and not force_reprocess:
            self.logger.info(f"📂 Found {existing_count}/{total_expected} existing files")
            self.logger.info("🔄 Will only process missing files. Set force_reprocess=True to reprocess all.")
            # Continue with normal processing
            self.processed_files = self._find_and_process_monthly_files()
        else:
            if force_reprocess and existing_count > 0:
                self.logger.info(f"🔄 Reprocessing all files (force_reprocess=True)")
            
            # Find and process monthly files
            self.processed_files = self._find_and_process_monthly_files()
        
        self.logger.info(f"Available files: {len(self.processed_files)} daily files for gauge {self.gauge_id}")

        # Automatically run analysis and create plots
        if self.processed_files:
            self.logger.info("Starting automatic analysis and plotting...")
            self.analyze_all_files()
        else:
            self.logger.warning("No files processed - skipping analysis")

    #---------------------------------------------------------------------------------

    def _load_catchment_shapefile(self) -> Optional[gpd.GeoDataFrame]:
        """
        Load the catchment shapefile for the gauge_id for clipping ERA5-Land data
        
        Returns
        -------
        Optional[gpd.GeoDataFrame]
            Catchment shapefile in WGS84 (EPSG:4326) or None if not found
        """
        self.logger.debug(f"Loading catchment shapefile for gauge ID: {self.gauge_id}")
        
        try:
            # Get shapefile path from config
            shape_dir_template = self.config.get('shape_dir', '01_data/topo/catchment_shapefile/catchment_shape_{gauge_id}.shp')
            shape_path = Path(shape_dir_template.format(gauge_id=self.gauge_id))
            
            # Make absolute path if needed
            if not shape_path.is_absolute():
                shape_path = self.main_dir / shape_path
            
            self.logger.debug(f"Looking for catchment shapefile at: {shape_path}")
            
            if not shape_path.exists():
                self.logger.warning(f"Catchment shapefile not found: {shape_path}")
                self.logger.warning("⚠️ Processing will continue without catchment clipping")
                return None
            
            # Read the catchment shapefile
            extent = gpd.read_file(shape_path)
            
            self.logger.info(f"✅ Loaded catchment shapefile with {len(extent)} features")
            self.logger.info(f"   Original CRS: {extent.crs}")
            
            # Reproject to WGS84 (EPSG:4326) to match ERA5-Land data
            if extent.crs != 'EPSG:4326':
                extent = extent.to_crs('EPSG:4326')
                self.logger.info("   Reprojected to WGS84 (EPSG:4326)")
            
            # Get catchment bounds for logging
            bounds = extent.total_bounds
            self.logger.info(f"   Catchment bounds: lon [{bounds[0]:.4f}, {bounds[2]:.4f}], lat [{bounds[1]:.4f}, {bounds[3]:.4f}]")
            
            return extent
            
        except Exception as e:
            self.logger.error(f"Error loading catchment shapefile: {e}")
            self.logger.warning("⚠️ Processing will continue without catchment clipping")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    #---------------------------------------------------------------------------------

    def _setup_logger(self) -> logging.Logger:
        """Set up and configure logger based on debug flag"""
        level = logging.DEBUG if self.debug else logging.INFO
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            force=True
        )
        
        # Suppress matplotlib logging
        matplotlib_logger = logging.getLogger('matplotlib')
        matplotlib_logger.setLevel(logging.WARNING)
        
        # Also suppress other common noisy loggers
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.colorbar').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
        
        return logging.getLogger(f'ERA5LandAnalyzer_Gauge_{self.gauge_id}')

    #---------------------------------------------------------------------------------

    def _check_existing_files(self) -> Dict[str, bool]:
        """
        Check if ERA5-Land processed files already exist in the output directory
        
        Returns
        -------
        Dict[str, bool]
            Dictionary indicating which file types already exist
        """
        existing_files = {
            'temperature_mean': False,
            'temperature_min': False,
            'temperature_max': False,
            'precipitation': False,
            'potential_evaporation': False
        }
        
        # Check for each expected output file
        expected_files = {
            'temperature_mean': 'era5_land_temp_mean.nc',
            'temperature_min': 'era5_land_temp_min.nc',
            'temperature_max': 'era5_land_temp_max.nc',
            'precipitation': 'era5_land_precip.nc',
            'potential_evaporation': 'era5_land_pet.nc'
        }
        
        for file_type, filename in expected_files.items():
            file_path = self.output_path / filename
            if file_path.exists():
                existing_files[file_type] = True
                self.logger.info(f"✅ Found existing file: {filename}")
            else:
                self.logger.debug(f"❌ Missing file: {filename}")
        
        return existing_files

    #---------------------------------------------------------------------------------

    def _find_monthly_files(self, need_temperature: bool = True, 
                        need_precipitation: bool = True, 
                        need_pet: bool = True) -> Dict[str, List[Path]]:
        """
        Find all monthly files in the ERA5-Land data directory, including geopotential
        MODIFIED: Only search for file types that are actually needed

        Parameters
        ----------
        need_temperature : bool
            Whether to search for temperature files
        need_precipitation : bool
            Whether to search for precipitation files
        need_pet : bool
            Whether to search for PET files

        Returns
        -------
        Dict[str, List[Path]]
            Dictionary with lists of files for each variable type
        """
        self.logger.info(f"Finding monthly files for period {self.start_date.date()} to {self.end_date.date()}")

        # Generate all year-month combinations needed
        start_year_month = (self.start_date.year, self.start_date.month)
        end_year_month = (self.end_date.year, self.end_date.month)

        year_months = []
        current_date = self.start_date.replace(day=1)  # Start from first day of start month
        while current_date <= self.end_date:
            year_months.append((current_date.year, current_date.month))
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)

        self.logger.info(f"Need data for {len(year_months)} months: {year_months[0]} to {year_months[-1]}")

        # Find files for each variable
        monthly_files = {
            'temperature': [],
            'precipitation': [],
            'potential_evaporation': [],
            'geopotential': []  # 🏔️ Add geopotential
        }

        missing_files = []

        for year, month in year_months:
            # ✅ Temperature files (only if needed)
            if need_temperature:
                temp_pattern = f"era5_land_2m_temperature_{year}_{month:02d}.nc"
                temp_file = self.era5_data_dir / temp_pattern

                if temp_file.exists() and temp_file.stat().st_size > 0:
                    monthly_files['temperature'].append(temp_file)
                    self.logger.debug(f"Found temperature file: {temp_file.name}")
                elif temp_file.exists() and temp_file.stat().st_size == 0:
                    missing_files.append(str(temp_file))
                    self.logger.warning(f"Empty temperature file (0 bytes): {temp_file}")
                else:
                    missing_files.append(str(temp_file))
                    self.logger.warning(f"Missing temperature file: {temp_file}")

            # ✅ Precipitation files (only if needed)
            if need_precipitation:
                precip_pattern = f"era5_land_total_precipitation_{year}_{month:02d}.nc"
                precip_file = self.era5_data_dir / precip_pattern

                if precip_file.exists() and precip_file.stat().st_size > 0:
                    monthly_files['precipitation'].append(precip_file)
                    self.logger.debug(f"Found precipitation file: {precip_file.name}")
                elif precip_file.exists() and precip_file.stat().st_size == 0:
                    missing_files.append(str(precip_file))
                    self.logger.warning(f"Empty precipitation file (0 bytes): {precip_file}")
                else:
                    missing_files.append(str(precip_file))
                    self.logger.warning(f"Missing precipitation file: {precip_file}")

            # ✅ PET files (only if needed)
            if need_pet:
                pet_pattern = f"era5_land_potential_evaporation_{year}_{month:02d}.nc"
                pet_file = self.era5_data_dir / pet_pattern

                if pet_file.exists() and pet_file.stat().st_size > 0:
                    monthly_files['potential_evaporation'].append(pet_file)
                    self.logger.debug(f"Found PET file: {pet_file.name}")
                elif pet_file.exists() and pet_file.stat().st_size == 0:
                    missing_files.append(str(pet_file))
                    self.logger.warning(f"Empty PET file (0 bytes): {pet_file}")
                else:
                    missing_files.append(str(pet_file))
                    self.logger.warning(f"Missing PET file: {pet_file}")

        # 🏔️ Look for geopotential file (only need to find it once since it's time-invariant)
        geopotential_file = self.era5_data_dir / "era5_land_geopotential.nc"
        if geopotential_file.exists() and geopotential_file.stat().st_size > 0:
            monthly_files['geopotential'].append(geopotential_file)
            self.logger.debug(f"Found geopotential file: {geopotential_file.name}")
        else:
            self.logger.warning(f"Geopotential file not found: {geopotential_file}")
            self.logger.warning("💡 Meteorological files will be processed without elevation data")

        # Report summary
        self.logger.info(f"Found {len(monthly_files['temperature'])} temperature files")
        self.logger.info(f"Found {len(monthly_files['precipitation'])} precipitation files")
        self.logger.info(f"Found {len(monthly_files['potential_evaporation'])} PET files")
        self.logger.info(f"Found {len(monthly_files['geopotential'])} geopotential files")

        if missing_files:
            self.logger.warning(f"Missing {len(missing_files)} files:")
            for missing in missing_files[:5]:  # Show first 5 missing files
                self.logger.warning(f"  - {missing}")
            if len(missing_files) > 5:
                self.logger.warning(f"  ... and {len(missing_files) - 5} more")

        return monthly_files

    #---------------------------------------------------------------------------------

    def _combine_monthly_files_with_clipping(self, file_list: List[Path], variable_type: str, 
                                    clip_bounds: Optional[Dict[str, float]]) -> Optional[xr.Dataset]:
        """
        Combine monthly files - CLIP each file but DON'T aggregate yet
        ✅ FIXED: Only clip, don't aggregate - aggregation happens later
        """
        if not file_list:
            self.logger.warning(f"No files to combine for {variable_type}")
            return None
        
        self.logger.info(f"Combining {len(file_list)} {variable_type} files...")
        
        try:
            sorted_files = sorted(file_list)
            
            # Validate files first (EXACT OLD METHOD)
            valid_files = []
            invalid_files = []
            
            for file_path in sorted_files:
                try:
                    with xr.open_dataset(file_path, engine='netcdf4') as test_ds:
                        if 'time' in test_ds.dims or 'valid_time' in test_ds.dims:
                            valid_files.append(file_path)
                        else:
                            invalid_files.append(file_path)
                            self.logger.warning(f"File missing time dimension: {file_path.name}")
                except Exception as e:
                    invalid_files.append(file_path)
                    self.logger.error(f"Cannot read file {file_path.name}: {str(e)}")
            
            if invalid_files:
                self.logger.warning(f"Found {len(invalid_files)} invalid files out of {len(sorted_files)}")
                
            if not valid_files:
                self.logger.error(f"No valid files found for {variable_type}")
                return None
                
            self.logger.info(f"Using {len(valid_files)} valid files out of {len(sorted_files)}")
            
            # Process in batches
            if len(valid_files) > 50:
                self.logger.info(f"Large file count ({len(valid_files)}) - using batch processing")
                
                batch_size = 12  # 1 year batches
                datasets = []
                
                for i in range(0, len(valid_files), batch_size):
                    batch_files = valid_files[i:i+batch_size]
                    batch_num = i//batch_size + 1
                    total_batches = (len(valid_files)-1)//batch_size + 1
                    
                    self.logger.debug(f"Processing batch {batch_num}/{total_batches}: {len(batch_files)} files")
                    
                    batch_datasets = []
                    for file_path in batch_files:
                        ds = xr.open_dataset(file_path, engine='netcdf4')
                        
                        # Standardize time coordinate name
                        time_coord = None
                        for coord in ['time', 'valid_time', 'datetime']:
                            if coord in ds.dims:
                                time_coord = coord
                                break
                        
                        if time_coord and time_coord != 'time':
                            ds = ds.rename({time_coord: 'time'})
                        
                        # ✅ ONLY CLIP - DON'T AGGREGATE YET
                        if clip_bounds is not None and 'latitude' in ds.coords:
                            lat_coords = ds.latitude.values
                            lat_increasing = lat_coords[0] < lat_coords[-1]
                            
                            if lat_increasing:
                                lat_slice = slice(clip_bounds['lat_min'], clip_bounds['lat_max'])
                            else:
                                lat_slice = slice(clip_bounds['lat_max'], clip_bounds['lat_min'])
                            
                            lon_slice = slice(clip_bounds['lon_min'], clip_bounds['lon_max'])
                            ds = ds.sel(latitude=lat_slice, longitude=lon_slice)
                        
                        batch_datasets.append(ds)
                    
                    # Concatenate batch
                    if batch_datasets:
                        batch_combined = xr.concat(batch_datasets, dim='time', combine_attrs='drop_conflicts')
                        batch_combined = batch_combined.sortby('time')
                        batch_combined = batch_combined.load()
                        datasets.append(batch_combined)
                        
                        for ds in batch_datasets:
                            ds.close()
                
                if not datasets:
                    self.logger.error(f"No datasets could be processed for {variable_type}")
                    return None
                
                # Combine all batches
                self.logger.info(f"Combining {len(datasets)} batches...")
                ds = xr.concat(datasets, dim='time', combine_attrs='drop_conflicts')
                ds = ds.sortby('time')
                
                for batch_ds in datasets:
                    batch_ds.close()
                    
            else:
                # Smaller file counts
                self.logger.info(f"Processing {len(valid_files)} files directly")
                
                datasets = []
                for file_path in valid_files:
                    ds = xr.open_dataset(file_path, engine='netcdf4')
                    
                    # Standardize time
                    time_coord = None
                    for coord in ['time', 'valid_time', 'datetime']:
                        if coord in ds.dims:
                            time_coord = coord
                            break
                    
                    if time_coord and time_coord != 'time':
                        ds = ds.rename({time_coord: 'time'})
                    
                    # ✅ ONLY CLIP - DON'T AGGREGATE YET
                    if clip_bounds is not None and 'latitude' in ds.coords:
                        lat_coords = ds.latitude.values
                        lat_increasing = lat_coords[0] < lat_coords[-1]
                        
                        if lat_increasing:
                            lat_slice = slice(clip_bounds['lat_min'], clip_bounds['lat_max'])
                        else:
                            lat_slice = slice(clip_bounds['lat_max'], clip_bounds['lat_min'])
                        
                        lon_slice = slice(clip_bounds['lon_min'], clip_bounds['lon_max'])
                        ds = ds.sel(latitude=lat_slice, longitude=lon_slice)
                    
                    datasets.append(ds)
                
                # Concatenate
                ds = xr.concat(datasets, dim='time', combine_attrs='drop_conflicts')
                
                for dataset in datasets:
                    dataset.close()
            
            # Final sort
            ds = ds.sortby('time')
            
            self.logger.info(f"Combined {variable_type} dataset shape: {dict(ds.dims)}")
            self.logger.info(f"Time range: {ds.time.min().values} to {ds.time.max().values}")
            self.logger.info(f"✅ Combined dataset is still HOURLY - will aggregate to daily next")
            
            return ds
            
        except Exception as e:
            self.logger.error(f"Error combining {variable_type} files: {str(e)}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None

    
    #---------------------------------------------------------------------------------

    def _filter_time_range_exact(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Filter dataset to exact time range specified in namelist
        
        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset
            
        Returns
        -------
        xr.Dataset
            Filtered dataset
        """
        if 'time' not in dataset.dims:
            self.logger.warning("No 'time' coordinate found in dataset")
            return dataset
        
        try:
            # Warm-up will be added later
            start_date_str = self.start_date.strftime('%Y-%m-%d')
            end_date_str = self.end_date.strftime('%Y-%m-%d')
            
            self.logger.debug(f"Filtering to simulation period: {start_date_str} to {end_date_str}")
            
            # Filter to date range
            filtered_ds = dataset.sel(time=slice(start_date_str, end_date_str))
            
            self.logger.info(f"Filtered time range: {filtered_ds.time.min().values} to {filtered_ds.time.max().values}")
            self.logger.info(f"Total days: {len(filtered_ds.time)}")
            
            return filtered_ds
            
        except Exception as e:
            self.logger.error(f"Error filtering time range: {str(e)}")
            self.logger.warning("Returning unfiltered dataset")
            return dataset

    #---------------------------------------------------------------------------------
    

    def _aggregate_to_daily(self, dataset: xr.Dataset, variable_type: str) -> Dict[str, xr.Dataset]:
        """
        Aggregate hourly data to daily values
        
        Parameters
        ----------
        dataset : xr.Dataset
            Hourly dataset
        variable_type : str
            'temperature', 'precipitation', or 'potential_evaporation'
            
        Returns
        -------
        Dict[str, xr.Dataset]
            Dictionary with aggregated datasets
        """
        self.logger.info(f"Aggregating {variable_type} to daily values...")
        
        try:
            if variable_type == 'temperature':
                # Temperature: create mean, min, max
                daily_mean = dataset.resample(time='1D').mean()
                daily_min = dataset.resample(time='1D').min()
                daily_max = dataset.resample(time='1D').max()
                
                # Convert from Kelvin to Celsius if needed
                temp_var = list(dataset.data_vars)[0]
                if dataset[temp_var].attrs.get('units') == 'K':
                    self.logger.debug("Converting temperature from Kelvin to Celsius")
                    daily_mean[temp_var] = daily_mean[temp_var] - 273.15
                    daily_min[temp_var] = daily_min[temp_var] - 273.15
                    daily_max[temp_var] = daily_max[temp_var] - 273.15
                    
                    # Update attributes
                    for ds in [daily_mean, daily_min, daily_max]:
                        ds[temp_var].attrs['units'] = 'degC'
                
                return {
                    'temp_mean': daily_mean,
                    'temp_min': daily_min,
                    'temp_max': daily_max
                }
                
            elif variable_type == 'precipitation':
                # ERA5-Land precipitation fix: compute differences first
                precip_var = list(dataset.data_vars)[0]
                
                self.logger.debug("ERA5-Land precipitation detected - computing hourly differences from cumulative data")
                
                # Compute differences along time dimension to get hourly precipitation
                precip_hourly = dataset[precip_var].diff('time', label='upper')
                
                # Handle negative values (can occur at forecast restart times)
                negative_mask = precip_hourly < 0
                if negative_mask.any():
                    self.logger.debug("Found negative differences (forecast restarts) - fixing...")
                    precip_hourly = precip_hourly.where(~negative_mask, dataset[precip_var].where(negative_mask))
                
                # Create new dataset with hourly precipitation
                hourly_dataset = dataset.copy()
                hourly_dataset[precip_var] = precip_hourly
                
                # Now sum hourly precipitation to daily
                daily_sum = hourly_dataset.resample(time='1D').sum()
                
                # Convert from m to mm if needed
                if dataset[precip_var].attrs.get('units') == 'm':
                    self.logger.debug("Converting precipitation from m to mm")
                    daily_sum[precip_var] = daily_sum[precip_var] * 1000.0
                    daily_sum[precip_var].attrs['units'] = 'mm'
                
                return {
                    'precip': daily_sum
                }
                
            elif variable_type == 'potential_evaporation':
                # PET: Handle ERA5-Land specific conventions more carefully
                pet_var = list(dataset.data_vars)[0]
                
                self.logger.debug("Processing potential evaporation data")
                self.logger.debug(f"Original PET units: {dataset[pet_var].attrs.get('units', 'Unknown')}")
                
                # Check some sample values to understand the data format
                sample_values = dataset[pet_var].isel(time=slice(0, 24)).mean(dim=['latitude', 'longitude']).values
                self.logger.debug(f"Sample hourly PET values: min={sample_values.min():.8f}, max={sample_values.max():.8f}")
                
                # Check if values are cumulative (like precipitation) or instantaneous rates
                # ERA5-Land PET is usually cumulative and in meters
                if dataset[pet_var].attrs.get('units') == 'm':
                    self.logger.debug("PET is in meters and likely cumulative - computing hourly differences")
                    
                    # First check if the data appears to be cumulative
                    first_day_values = dataset[pet_var].isel(time=slice(0, 24)).mean(dim=['latitude', 'longitude']).values
                    if len(first_day_values) > 1:
                        is_increasing = np.all(np.diff(first_day_values) >= 0)
                        self.logger.debug(f"Values appear to be {'cumulative' if is_increasing else 'non-cumulative'}")
                    
                    # Compute hourly differences from cumulative data
                    pet_hourly = dataset[pet_var].diff('time', label='upper')
                    
                    # Handle negative values at forecast restarts
                    negative_mask = pet_hourly < 0
                    if negative_mask.any():
                        self.logger.debug("Found negative differences in PET - fixing...")
                        pet_hourly = pet_hourly.where(~negative_mask, dataset[pet_var].where(negative_mask))
                    
                    # Create new dataset with hourly PET rates
                    hourly_dataset = dataset.copy()
                    hourly_dataset[pet_var] = pet_hourly
                else:
                    # Already in rate form
                    self.logger.debug("PET appears to be in rate form already")
                    hourly_dataset = dataset
                
                # Sum hourly PET to get daily totals
                daily_sum = hourly_dataset.resample(time='1D').sum()
                
                # ERA5-Land PET convention: negative values mean upward flux (evaporation)
                # Convert to positive values for standard PET interpretation
                pet_mean_value = daily_sum[pet_var].mean()
                if pet_mean_value < 0:
                    self.logger.debug("Converting negative PET to positive (ERA5-Land upward flux convention)")
                    daily_sum[pet_var] = -daily_sum[pet_var]
                
                # Convert from m to mm if needed
                if dataset[pet_var].attrs.get('units') == 'm':
                    self.logger.debug("Converting PET from m to mm")
                    daily_sum[pet_var] = daily_sum[pet_var] * 1000.0
                    daily_sum[pet_var].attrs['units'] = 'mm'
                
                # Additional check: if we still have very high values, might need different approach
                pet_stats = daily_sum[pet_var].mean(dim=['latitude', 'longitude'])
                daily_values = pet_stats.values
                daily_values = daily_values[~np.isnan(daily_values)]
                
                if len(daily_values) > 0:
                    self.logger.info(f"Daily PET statistics after processing:")
                    self.logger.info(f"  Min: {daily_values.min():.3f} mm/day")
                    self.logger.info(f"  Max: {daily_values.max():.3f} mm/day")
                    self.logger.info(f"  Mean: {daily_values.mean():.3f} mm/day")
                    self.logger.info(f"  Median: {np.median(daily_values):.3f} mm/day")
                    
                    # Sanity checks with better thresholds for high mountain regions
                    if daily_values.mean() > 10:
                        self.logger.warning("⚠️ Very high PET values (>10mm/day avg) - might be processing error!")
                        self.logger.warning("💡 For high mountain regions, expect 0.5-3 mm/day typically")
                    if daily_values.min() < 0:
                        self.logger.warning("⚠️ Still have negative PET values - check sign convention!")
                    if daily_values.max() > 30:
                        self.logger.error("❌ Extremely high PET values (>30mm/day max) - likely processing error!")
                        
                        # Additional debugging: check if we need to use mean instead of sum
                        self.logger.debug("🔍 Trying alternative aggregation (mean instead of sum)...")
                        daily_mean_test = hourly_dataset.resample(time='1D').mean()
                        if dataset[pet_var].attrs.get('units') == 'm':
                            daily_mean_test[pet_var] = daily_mean_test[pet_var] * 1000.0
                        if daily_mean_test[pet_var].mean() < 0:
                            daily_mean_test[pet_var] = -daily_mean_test[pet_var]
                            
                        mean_values = daily_mean_test[pet_var].mean(dim=['latitude', 'longitude']).values
                        mean_values = mean_values[~np.isnan(mean_values)]
                        if len(mean_values) > 0:
                            self.logger.debug(f"Alternative (mean) approach gives: {mean_values.mean():.3f} mm/day average")
                            if mean_values.mean() < 10:  # If mean approach gives more reasonable values
                                self.logger.warning("🔄 Using mean aggregation instead of sum for PET")
                                daily_sum = daily_mean_test
                
                return {
                    'pet': daily_sum
                }
            
        except Exception as e:
            self.logger.error(f"Error aggregating {variable_type}: {str(e)}")
            return {}
        
    #---------------------------------------------------------------------------------

    def process_geopotential_to_elevation(self) -> Optional[Path]:
        """
        Process geopotential NetCDF file to elevation and save as separate file with single elevation value per cell
        MODIFIED: Skip coordinate flipping here to match meteorological data processing timing
        
        Returns
        -------
        Optional[Path]
            Path to created elevation file, or None if failed
        """
        self.logger.info("🏔️ Processing geopotential to elevation...")
        
        # Find geopotential file in the same directory as monthly files
        geopotential_file = self.era5_data_dir / "era5_land_geopotential.nc"
        
        if not geopotential_file.exists():
            self.logger.error(f"❌ Geopotential file not found: {geopotential_file}")
            self.logger.error("💡 Expected file: era5_land_geopotential.nc in ERA5 data directory")
            return None
        
        try:
            self.logger.info(f"📂 Loading geopotential file: {geopotential_file}")
            
            # Open geopotential data
            ds_geo = xr.open_dataset(geopotential_file)
            
            self.logger.info(f"📊 Geopotential file contents:")
            self.logger.info(f"  Variables: {list(ds_geo.data_vars)}")
            self.logger.info(f"  Dimensions: {dict(ds_geo.dims)}")
            self.logger.info(f"  Coordinates: {list(ds_geo.coords)}")
            
            # Find geopotential variable (usually 'z' or 'geopotential')
            geo_var = None
            possible_vars = ['z', 'geopotential', 'Z']
            
            for var in ds_geo.data_vars:
                if var in possible_vars:
                    geo_var = var
                    break
            
            if geo_var is None:
                self.logger.error(f"❌ Could not find geopotential variable")
                self.logger.error(f"Available variables: {list(ds_geo.data_vars)}")
                self.logger.error(f"Expected one of: {possible_vars}")
                ds_geo.close()
                return None
            
            self.logger.info(f"✅ Using geopotential variable: '{geo_var}'")
            
            # Get geopotential data
            geo_data = ds_geo[geo_var]
            self.logger.info(f"  Original shape: {geo_data.shape}")
            self.logger.info(f"  Original dimensions: {geo_data.dims}")
            
            # Remove time dimension if present (geopotential is time-invariant)
            if 'time' in geo_data.dims:
                self.logger.info("⏰ Removing time dimension from geopotential data")
                geo_data = geo_data.isel(time=0)
                self.logger.info(f"  New shape: {geo_data.shape}")
            
            # Convert geopotential to elevation
            # Geopotential (m²/s²) ÷ standard gravity (9.80665 m/s²) = elevation (m)
            standard_gravity = 9.80665  # m/s²
            
            self.logger.info(f"🔄 Converting geopotential to elevation (÷ {standard_gravity} m/s²)")
            elevation_data = geo_data / standard_gravity
            
            # 🔧 SKIP COORDINATE FLIPPING HERE - let it happen in _save_daily_files
            # elevation_data = self._check_and_fix_coordinate_flipping_single_var(elevation_data)
            self.logger.info("⏭️ Skipping coordinate flipping for elevation - will be handled during save")
            
            # 🔧 IMPORTANT: Create a simple 2D dataset with just the elevation variable
            # No time dimension, just latitude and longitude with elevation values
            ds_elevation = xr.Dataset({
                'elevation': elevation_data
            })
            
            # ... rest of the function stays the same ...
            
            # Update elevation variable attributes
            ds_elevation['elevation'].attrs = {
                'units': 'm',
                'long_name': 'Surface elevation above sea level',
                'standard_name': 'surface_altitude',
                'source': 'ERA5-Land geopotential converted to elevation',
                'conversion_factor': f'Divided by standard gravity ({standard_gravity} m/s²)',
                'description': 'Average elevation for each ERA5-Land grid cell',
                'processing_date': pd.Timestamp.now().isoformat(),
                'grid_mapping': 'WGS84'
            }
            
            # Copy coordinate attributes from original data
            for coord in ds_elevation.coords:
                if coord in ds_geo.coords:
                    ds_elevation[coord].attrs = ds_geo[coord].attrs
            
            # Add global attributes
            ds_elevation.attrs.update({
                'title': 'ERA5-Land elevation data',
                'gauge_id': str(self.gauge_id),  # Ensure string type
                'processed_by': 'ERA5LandAnalyzer',
                'creation_date': pd.Timestamp.now().isoformat(),
                'source_file': str(geopotential_file),
                'coordinate_orientation': 'Same as source data - will be corrected during meteorological processing',
                'conventions': 'CF-1.8',
                'institution': 'Processed from ERA5-Land reanalysis',
                'comment': 'Single elevation value per grid cell, time-invariant'
            })
            
            # Save elevation file (without coordinate flipping)
            elevation_file = self.output_path / 'era5_land_elevation.nc'
            ds_elevation.to_netcdf(elevation_file)
            
            # Log elevation statistics
            elev_values = ds_elevation['elevation'].values
            self.logger.info(f"📊 Elevation statistics:")
            self.logger.info(f"  Min: {np.nanmin(elev_values):.1f} m")
            self.logger.info(f"  Max: {np.nanmax(elev_values):.1f} m")
            self.logger.info(f"  Mean: {np.nanmean(elev_values):.1f} m")
            self.logger.info(f"  Data shape: {elev_values.shape}")
            
            self.logger.info(f"✅ Elevation file saved: {elevation_file}")
            
            # Close datasets
            ds_geo.close()
            ds_elevation.close()
            
            return elevation_file
            
        except Exception as e:
            self.logger.error(f"❌ Error processing geopotential to elevation: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def _check_and_fix_coordinate_flipping_single_var(self, data_array: xr.DataArray) -> xr.DataArray:
        """
        Apply coordinate flipping to a single DataArray (for elevation processing)
        
        Parameters
        ----------
        data_array : xr.DataArray
            Input DataArray
            
        Returns
        -------
        xr.DataArray
            DataArray with corrected coordinate orientation
        """
        # Check for latitude coordinates
        lat_dim = None
        for dim in ['latitude', 'lat', 'y']:
            if dim in data_array.coords:
                lat_dim = dim
                break
        
        if lat_dim is None:
            self.logger.debug("No latitude coordinate found - skipping flip check")
            return data_array
        
        lat_coords = data_array.coords[lat_dim].values
        lat_decreasing = lat_coords[0] > lat_coords[-1]
        
        if lat_decreasing:
            self.logger.warning("⚠️  Latitude coordinates are decreasing - applying coordinate flip")
            
            # Flip latitude coordinates by reversing the slice
            data_corrected = data_array.sel({lat_dim: slice(None, None, -1)})
            
            # Verify the flip
            new_lat_coords = data_corrected.coords[lat_dim].values
            self.logger.info(f"✅ Latitude corrected: {lat_coords.min():.4f} to {lat_coords.max():.4f}")
            self.logger.info(f"   Original order: {lat_coords[0]:.4f} → {lat_coords[-1]:.4f}")
            self.logger.info(f"   Corrected order: {new_lat_coords[0]:.4f} → {new_lat_coords[-1]:.4f}")
            
            return data_corrected
        else:
            self.logger.debug("✅ Latitude coordinates are correctly oriented (increasing)")
            return data_array

    #---------------------------------------------------------------------------------

    def _check_and_fix_coordinate_flipping(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Check if coordinates are flipped and correct them consistently with GridWeightsGenerator
        
        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset
            
        Returns
        -------
        xr.Dataset
            Dataset with corrected coordinate orientation
        """
        # Check for latitude coordinates
        lat_dim = None
        for dim in ['latitude', 'lat', 'y']:
            if dim in dataset.coords:
                lat_dim = dim
                break
        
        if lat_dim is None:
            self.logger.debug("No latitude coordinate found - skipping flip check")
            return dataset
        
        lat_coords = dataset.coords[lat_dim].values
        lat_decreasing = lat_coords[0] > lat_coords[-1]
        
        if lat_decreasing:
            self.logger.warning("⚠️  Latitude coordinates are decreasing - applying coordinate flip")
            self.logger.warning("🔄 This matches the GridWeightsGenerator correction for consistent orientation")
            
            # Flip latitude coordinates by reversing the slice
            dataset_corrected = dataset.sel({lat_dim: slice(None, None, -1)})
            
            # Verify the flip
            new_lat_coords = dataset_corrected.coords[lat_dim].values
            self.logger.info(f"✅ Latitude corrected: {lat_coords.min():.4f} to {lat_coords.max():.4f}")
            self.logger.info(f"   Original order: {lat_coords[0]:.4f} → {lat_coords[-1]:.4f}")
            self.logger.info(f"   Corrected order: {new_lat_coords[0]:.4f} → {new_lat_coords[-1]:.4f}")
            
            return dataset_corrected
        else:
            self.logger.debug("✅ Latitude coordinates are correctly oriented (increasing)")
            return dataset
        
    #---------------------------------------------------------------------------------

    def _clip_to_catchment(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Clip ERA5-Land dataset to catchment extent
        
        Parameters
        ----------
        dataset : xr.Dataset
            ERA5-Land dataset to clip
            
        Returns
        -------
        xr.Dataset
            Clipped dataset
        """
        # ✅ NEW: If gauge_id equals basin_id, no clipping needed
        if self.gauge_id == self.basin_id:
            self.logger.debug(f"gauge_id ({self.gauge_id}) == basin_id ({self.basin_id}) - skipping clipping")
            self.logger.info("✅ Using meteo data from same basin - no clipping required")
            return dataset
        
        if self.catchment_extent is None:
            self.logger.debug("No catchment extent available - skipping clipping")
            return dataset
        
        try:
            self.logger.info("✂️ Clipping dataset to catchment extent...")
            
            # Get catchment bounds
            bounds = self.catchment_extent.total_bounds  # [minx, miny, maxx, maxy]
            
            self.logger.debug(f"Catchment bounds: lon [{bounds[0]:.4f}, {bounds[2]:.4f}], lat [{bounds[1]:.4f}, {bounds[3]:.4f}]")
            
            # Get data bounds before clipping
            if 'latitude' in dataset.coords and 'longitude' in dataset.coords:
                data_lat_min = float(dataset.latitude.min())
                data_lat_max = float(dataset.latitude.max())
                data_lon_min = float(dataset.longitude.min())
                data_lon_max = float(dataset.longitude.max())
                
                self.logger.debug(f"Data bounds before clipping: lon [{data_lon_min:.4f}, {data_lon_max:.4f}], lat [{data_lat_min:.4f}, {data_lat_max:.4f}]")
            
            # Add buffer to catchment extent (e.g., 0.1 degrees ~ 10km)
            buffer_deg = 0.1
            buffered_extent = self.catchment_extent.buffer(buffer_deg)
            
            self.logger.debug(f"Applied buffer of {buffer_deg}° to catchment extent")
            
            # Get buffered bounds
            buffered_bounds = buffered_extent.total_bounds
            
            # 🔧 FIX: Check if latitude coordinates are increasing or decreasing
            lat_coords = dataset.latitude.values
            lat_increasing = lat_coords[0] < lat_coords[-1]
            
            if lat_increasing:
                # Standard case: latitude increases (south to north)
                lat_slice = slice(buffered_bounds[1], buffered_bounds[3])
            else:
                # Reversed case: latitude decreases (north to south)
                # Need to reverse the slice bounds for proper selection
                lat_slice = slice(buffered_bounds[3], buffered_bounds[1])
            
            # Longitude should always work with standard slice
            lon_slice = slice(buffered_bounds[0], buffered_bounds[2])
            
            self.logger.debug(f"Using latitude slice: {lat_slice}")
            self.logger.debug(f"Using longitude slice: {lon_slice}")
            
            # Clip dataset to buffered catchment bounds
            clipped = dataset.sel(
                latitude=lat_slice,
                longitude=lon_slice
            )
            
            # Verify we got data
            if clipped.latitude.size == 0 or clipped.longitude.size == 0:
                self.logger.error("❌ Clipping resulted in empty dataset!")
                self.logger.error(f"Latitude size: {clipped.latitude.size}, Longitude size: {clipped.longitude.size}")
                self.logger.warning("Returning unclipped dataset")
                return dataset
            
            # Log clipped dataset info
            if 'latitude' in clipped.coords and 'longitude' in clipped.coords:
                clipped_lat_min = float(clipped.latitude.min())
                clipped_lat_max = float(clipped.latitude.max())
                clipped_lon_min = float(clipped.longitude.min())
                clipped_lon_max = float(clipped.longitude.max())
                
                self.logger.info(f"   Clipped bounds: lon [{clipped_lon_min:.4f}, {clipped_lon_max:.4f}], lat [{clipped_lat_min:.4f}, {clipped_lat_max:.4f}]")
                
                # Calculate reduction in data size
                original_size = dataset.latitude.size * dataset.longitude.size
                clipped_size = clipped.latitude.size * clipped.longitude.size
                reduction = (1 - clipped_size/original_size) * 100
                
                self.logger.info(f"✅ Dataset clipped: {original_size} → {clipped_size} grid cells ({reduction:.1f}% reduction)")
            
            return clipped
            
        except Exception as e:
            self.logger.error(f"Error clipping dataset to catchment: {e}")
            self.logger.warning("Returning unclipped dataset")
            import traceback
            self.logger.debug(traceback.format_exc())
            return dataset

    #---------------------------------------------------------------------------------

    def _save_daily_files(self, daily_datasets: Dict[str, xr.Dataset]) -> List[Path]:
        """
        Save daily aggregated datasets to NetCDF files with correct coordinate orientation and elevation data
        MODIFIED: Add elevation FIRST, then clip the combined dataset to ensure matching extents

        Parameters
        ----------
        daily_datasets : Dict[str, xr.Dataset]
            Dictionary of daily datasets

        Returns
        -------
        List[Path]
            List of saved file paths
        """
        saved_files = []

        # 🏔️ LOAD ELEVATION ONCE (outside the loop)
        elevation_dataset = None
        elevation_file = self.output_path / 'era5_land_elevation.nc'

        if elevation_file.exists():
            try:
                self.logger.debug(f"📍 Loading elevation data from: {elevation_file.name}")
                elevation_dataset = xr.open_dataset(elevation_file)

                if 'elevation' in elevation_dataset.data_vars:
                    # Apply coordinate flipping to elevation data
                    self.logger.debug("Applying coordinate flipping to elevation data...")
                    elevation_dataset = self._check_and_fix_coordinate_flipping(elevation_dataset)
                    self.logger.debug("✅ Elevation data coordinate-flipped")
                else:
                    self.logger.warning("❌ No 'elevation' variable found in elevation file")
                    elevation_dataset = None

            except Exception as e:
                self.logger.error(f"❌ Error loading elevation file: {e}")
                import traceback
                self.logger.debug(f"Full traceback: {traceback.format_exc()}")
                elevation_dataset = None
        else:
            self.logger.debug("📍 No elevation file found - saving without elevation data")

        for var_name, dataset in daily_datasets.items():
            try:
                self.logger.debug(f"Saving {var_name} dataset...")

                # 🔧 FIRST: Apply coordinate flipping to meteorological data
                self.logger.debug(f"Checking coordinate orientation for {var_name} before saving...")
                dataset_corrected = self._check_and_fix_coordinate_flipping(dataset)

                # ✅ NEW: Add elevation BEFORE clipping (if available)
                elevation_added = False
                if elevation_dataset is not None:
                    try:
                        # Get coordinates from meteorological data
                        meteo_lat = dataset_corrected.coords.get('latitude', dataset_corrected.coords.get('lat'))
                        meteo_lon = dataset_corrected.coords.get('longitude', dataset_corrected.coords.get('lon'))
                        elev_lat = elevation_dataset.coords.get('latitude', elevation_dataset.coords.get('lat'))
                        elev_lon = elevation_dataset.coords.get('longitude', elevation_dataset.coords.get('lon'))

                        if (meteo_lat is not None and meteo_lon is not None and
                            elev_lat is not None and elev_lon is not None):

                            # Check if coordinates match (before clipping)
                            lat_match = (len(meteo_lat) == len(elev_lat)) and np.allclose(meteo_lat.values, elev_lat.values, rtol=1e-8, atol=1e-8)
                            lon_match = (len(meteo_lon) == len(elev_lon)) and np.allclose(meteo_lon.values, elev_lon.values, rtol=1e-8, atol=1e-8)

                            if lat_match and lon_match:
                                # Direct assignment - coordinates match exactly
                                dataset_corrected['elevation'] = elevation_dataset['elevation']
                                elevation_added = True
                                self.logger.debug(f"✅ Added elevation to {var_name} (exact coordinate match)")

                            else:
                                # Interpolate elevation to match meteo grid
                                self.logger.debug("⚠️ Elevation grid doesn't match meteo grid - interpolating...")
                                try:
                                    elev_interp = elevation_dataset['elevation'].interp(
                                        latitude=meteo_lat,
                                        longitude=meteo_lon,
                                        method='linear'
                                    )
                                    dataset_corrected['elevation'] = elev_interp
                                    elevation_added = True
                                    self.logger.debug("✅ Successfully interpolated elevation to meteo grid")
                                except Exception as e:
                                    self.logger.error(f"❌ Error interpolating elevation: {e}")
                                    elevation_added = False

                    except Exception as e:
                        self.logger.error(f"❌ Error adding elevation: {e}")
                        import traceback
                        self.logger.debug(f"Full traceback: {traceback.format_exc()}")

                # ✅ NOW: Clip the COMBINED dataset (meteo + elevation together)
                self.logger.debug("Clipping combined dataset (meteo + elevation) to catchment extent...")
                dataset_corrected = self._clip_to_catchment(dataset_corrected)
                self.logger.debug("✅ Combined dataset clipped")

                # Log final coordinates
                final_lat = dataset_corrected.coords.get('latitude', dataset_corrected.coords.get('lat'))
                final_lon = dataset_corrected.coords.get('longitude', dataset_corrected.coords.get('lon'))
                
                if final_lat is not None and final_lon is not None:
                    self.logger.debug(f"Final {var_name} coordinates after clipping:")
                    self.logger.debug(f"  Lat: {final_lat.values[0]:.6f} to {final_lat.values[-1]:.6f} ({len(final_lat)} points)")
                    self.logger.debug(f"  Lon: {final_lon.values[0]:.6f} to {final_lon.values[-1]:.6f} ({len(final_lon)} points)")

                # Verify elevation is still in the dataset if it was added
                if elevation_added and 'elevation' in dataset_corrected.data_vars:
                    self.logger.debug(f"✅ Elevation successfully included in final {var_name} dataset")
                elif elevation_added and 'elevation' not in dataset_corrected.data_vars:
                    self.logger.warning(f"⚠️ Elevation was added but disappeared during clipping for {var_name}")
                    elevation_added = False

                # 🔧 CRITICAL FIX: Ensure we actually have data variables before saving
                if len(list(dataset_corrected.data_vars)) == 0:
                    self.logger.error(f"❌ Dataset {var_name} has no data variables! Skipping save.")
                    continue

                # Simplified filename without date range
                filename = f"era5_land_{var_name}.nc"
                output_file_path = self.output_path / filename

                # 🔧 FIX: Convert boolean to string for NetCDF compatibility
                elevation_status = "true" if elevation_added else "false"
                
                # ✅ UPDATED: Metadata to reflect warm-up inclusion
                if hasattr(self, 'warmup_date') and self.warmup_date is not None:
                    warmup_status = "true"
                    time_range_str = f"{self.warmup_date.date()} to {self.end_date.date()} (includes warm-up)"
                else:
                    warmup_status = "false"
                    time_range_str = f"{self.start_date.date()} to {self.end_date.date()}"

                # Add metadata with proper data types
                dataset_corrected.attrs.update({
                    'title': f'ERA5-Land {var_name} daily data',
                    'gauge_id': str(self.gauge_id),
                    'time_range': time_range_str,
                    'warmup_included': warmup_status,
                    'warmup_start': str(self.warmup_date.date()) if hasattr(self, 'warmup_date') and self.warmup_date else 'none',
                    'simulation_start': str(self.start_date.date()),
                    'simulation_end': str(self.end_date.date()),
                    'processed_by': 'ERA5LandAnalyzer',
                    'creation_date': pd.Timestamp.now().isoformat(),
                    'model_type': str(self.model_type),
                    'catchment': f'catchment_{self.gauge_id}',
                    'coordinate_orientation': 'Corrected for consistent north-up orientation',
                    'elevation_included': elevation_status,
                    'clipped_to_catchment': 'true' if self.catchment_extent is not None else 'false'
                })

                # Log what we're about to save
                self.logger.debug(f"About to save {var_name}:")
                self.logger.debug(f"  Data variables: {list(dataset_corrected.data_vars)}")
                self.logger.debug(f"  Dimensions: {dict(dataset_corrected.dims)}")
                self.logger.debug(f"  Coordinates: {list(dataset_corrected.coords)}")

                # Save corrected dataset
                dataset_corrected.to_netcdf(output_file_path)
                saved_files.append(output_file_path)

                elevation_status_msg = "with elevation" if elevation_added else "without elevation"
                self.logger.info(f"Saved {var_name}: {filename} ({elevation_status_msg})")
                self.logger.debug(f"Full path: {output_file_path}")

                # Close datasets to free memory
                dataset.close()
                dataset_corrected.close()

            except Exception as e:
                self.logger.error(f"Error saving {var_name}: {str(e)}")
                import traceback
                self.logger.error(f"Full traceback: {traceback.format_exc()}")

        # Close elevation dataset at the end
        if elevation_dataset is not None:
            elevation_dataset.close()

        return saved_files

    #---------------------------------------------------------------------------------

    def _find_and_process_monthly_files(self) -> List[Path]:
        """
        Main processing pipeline - RESTORED old method with clipping added
        ✅ Clips during combination, aggregates AFTER combination
        """
        self.logger.info("Starting monthly file processing pipeline...")
        
        # Check which files already exist
        expected_files = {
            'temperature_mean': 'era5_land_temp_mean.nc',
            'temperature_min': 'era5_land_temp_min.nc',
            'temperature_max': 'era5_land_temp_max.nc',
            'precipitation': 'era5_land_precip.nc',
            'potential_evaporation': 'era5_land_pet.nc'
        }
        
        existing_files = {}
        missing_files = []
        
        for var_type, filename in expected_files.items():
            file_path = self.output_path / filename
            if file_path.exists() and not self.force_reprocess:
                existing_files[var_type] = file_path
                self.logger.info(f"✅ Using existing file: {filename}")
            else:
                missing_files.append(var_type)
        
        if len(missing_files) == 0 and not self.force_reprocess:
            self.logger.info("✅ All required files already exist - skipping monthly file processing")
            return list(existing_files.values())
        
        # Find monthly files
        if missing_files or self.force_reprocess:
            self.logger.info(f"📂 Need to process: {missing_files if missing_files else 'all files (force_reprocess=True)'}")
            
            need_temperature = any('temperature' in mf for mf in missing_files) or self.force_reprocess
            need_precipitation = 'precipitation' in missing_files or self.force_reprocess
            need_pet = 'potential_evaporation' in missing_files or self.force_reprocess
            
            monthly_files = self._find_monthly_files(
                need_temperature=need_temperature,
                need_precipitation=need_precipitation, 
                need_pet=need_pet
            )
            
            if not any(monthly_files.values()):
                self.logger.error("No monthly files found!")
                return list(existing_files.values())
        else:
            monthly_files = {'temperature': [], 'precipitation': [], 'potential_evaporation': [], 'geopotential': []}

        # Calculate clip bounds
        clip_bounds = None
        
        if self.catchment_extent is not None:
            bounds = self.catchment_extent.total_bounds
            buffer_deg = 0.1
            
            clip_bounds = {
                'lon_min': bounds[0] - buffer_deg,
                'lon_max': bounds[2] + buffer_deg,
                'lat_min': bounds[1] - buffer_deg,
                'lat_max': bounds[3] + buffer_deg
            }
            
            self.logger.info(f"✂️ Will clip to bounds: lon [{clip_bounds['lon_min']:.2f}, {clip_bounds['lon_max']:.2f}], "
                            f"lat [{clip_bounds['lat_min']:.2f}, {clip_bounds['lat_max']:.2f}]")
        else:
            self.logger.info("⚠️ No catchment extent - processing full dataset")

        # Process geopotential to elevation
        elevation_file = self.output_path / 'era5_land_elevation.nc'
        
        if not elevation_file.exists() or self.force_reprocess:
            self.logger.info("🏔️ Processing geopotential to elevation...")
            elevation_file_created = self.process_geopotential_to_elevation()
            
            if elevation_file_created is not None:
                self.logger.info(f"✅ Elevation processing successful: {elevation_file_created.name}")
            else:
                self.logger.warning("⚠️ Elevation processing failed - meteorological files will be saved without elevation data")
        else:
            self.logger.info(f"✅ Using existing elevation file: {elevation_file.name}")
        
        all_daily_files = list(existing_files.values())
        
        # ===== TEMPERATURE =====
        temp_files_exist = all(
            var_type in existing_files 
            for var_type in ['temperature_mean', 'temperature_min', 'temperature_max']
        )
        
        if monthly_files['temperature'] and not temp_files_exist:
            self.logger.info("Processing temperature files...")
            
            # 1. Combine (with clipping)
            temp_combined = self._combine_monthly_files_with_clipping(
                monthly_files['temperature'], 
                'temperature',
                clip_bounds
            )
            
            if temp_combined is not None:
                # 2. Filter time range
                temp_filtered = self._filter_time_range_exact(temp_combined)
                
                # 3. Aggregate to daily (OLD METHOD)
                temp_daily = self._aggregate_to_daily(temp_filtered, 'temperature')
                
                # 4. Save
                if temp_daily:
                    temp_files = self._save_daily_files(temp_daily)
                    all_daily_files.extend(temp_files)
                
                temp_combined.close()
                temp_filtered.close()
        elif temp_files_exist:
            self.logger.info("⏭️ Skipping temperature processing - files already exist")
        
        # ===== PRECIPITATION =====
        if monthly_files['precipitation'] and 'precipitation' not in existing_files:
            self.logger.info("Processing precipitation files...")
            
            # 1. Combine (with clipping)
            precip_combined = self._combine_monthly_files_with_clipping(
                monthly_files['precipitation'], 
                'precipitation',
                clip_bounds
            )
            
            if precip_combined is not None:
                # 2. Filter time range
                precip_filtered = self._filter_time_range_exact(precip_combined)
                
                # 3. Aggregate to daily (OLD METHOD)
                precip_daily = self._aggregate_to_daily(precip_filtered, 'precipitation')
                
                # 4. Save
                if precip_daily:
                    precip_files = self._save_daily_files(precip_daily)
                    all_daily_files.extend(precip_files)
                
                precip_combined.close()
                precip_filtered.close()
        elif 'precipitation' in existing_files:
            self.logger.info("⏭️ Skipping precipitation processing - file already exists")
        
        # ===== PET =====
        if monthly_files['potential_evaporation'] and 'potential_evaporation' not in existing_files:
            self.logger.info("Processing potential evaporation files...")
            
            # 1. Combine (with clipping)
            pet_combined = self._combine_monthly_files_with_clipping(
                monthly_files['potential_evaporation'], 
                'potential_evaporation',
                clip_bounds
            )
            
            if pet_combined is not None:
                # 2. Filter time range
                pet_filtered = self._filter_time_range_exact(pet_combined)
                
                # 3. Aggregate to daily (OLD METHOD)
                pet_daily = self._aggregate_to_daily(pet_filtered, 'potential_evaporation')
                
                # 4. Save
                if pet_daily:
                    pet_files = self._save_daily_files(pet_daily)
                    all_daily_files.extend(pet_files)
                
                pet_combined.close()
                pet_filtered.close()
        elif 'potential_evaporation' in existing_files:
            self.logger.info("⏭️ Skipping PET processing - file already exists")
        
        self.logger.info(f"Processing pipeline complete! Total files available: {len(all_daily_files)}")
        
        for file_path in all_daily_files:
            self.logger.info(f"  - {file_path.name}")
        
        # Add warm-up period
        if hasattr(self, 'warmup_date') and self.warmup_date is not None:
            self.logger.info("\n🔄 Adding warm-up period to meteorological files...")
            all_daily_files = self._add_warmup_to_files(all_daily_files)
        
        # Copy to model directory
        self._copy_to_model_directory(all_daily_files)
        
        return all_daily_files
    
    #---------------------------------------------------------------------------------
    
    def _convert_units(self, dataset: xr.Dataset, var_name: str) -> xr.Dataset:
        """Convert ERA5-Land units to more readable formats"""
        ds = dataset.copy()
        
        if var_name in self.era5_variables and self.era5_variables[var_name]['convert_kelvin']:
            if var_name in ds and ds[var_name].attrs.get('units') == 'K':
                self.logger.debug(f"Converting {var_name} from Kelvin to Celsius")
                ds[var_name] = ds[var_name] - 273.15
                ds[var_name].attrs['units'] = 'degC'
        
        elif var_name == 'tp' and var_name in ds:
            # Convert precipitation from m to mm
            if ds[var_name].attrs.get('units') == 'm':
                self.logger.debug("Converting precipitation from m to mm")
                ds[var_name] = ds[var_name] * 1000.0
                ds[var_name].attrs['units'] = 'mm'
        
        return ds
    
    #---------------------------------------------------------------------------------

    def _copy_to_model_directory(self, files: List[Path]) -> None:
        """
        Copy processed files from shared data_obs to model-specific data_obs.
        This maintains backward compatibility while using shared storage.
        ✅ UPDATED: Also copies monthly average CSV files
        
        Parameters
        ----------
        files : List[Path]
            List of file paths to copy
        """
        import shutil
        
        if not files:
            self.logger.debug("No files to copy to model directory")
            return
        
        self.logger.info(f"📋 Copying {len(files)} files from shared to model-specific directory...")
        self.logger.debug(f"  Source: {self.shared_data_dir}")
        self.logger.debug(f"  Destination: {self.model_data_dir}")
        
        copied_count = 0
        for file_path in files:
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
        
        # ✅ NEW: Also copy monthly average CSV files if they exist
        monthly_files = [
            'monthly_temperature_averages.csv',
            'monthly_pet_averages.csv'
        ]
        
        for monthly_file in monthly_files:
            src = self.shared_data_dir / monthly_file
            if src.exists():
                dest = self.model_data_dir / monthly_file
                try:
                    shutil.copy2(src, dest)
                    self.logger.debug(f"  ✅ Copied: {monthly_file}")
                    copied_count += 1
                except Exception as e:
                    self.logger.warning(f"  ❌ Failed to copy {monthly_file}: {e}")
            else:
                self.logger.debug(f"  ⏭️ Monthly file not found (will be created later): {monthly_file}")
        
        self.logger.info(f"✅ Successfully copied {copied_count} files to {self.model_data_dir.name}/")
        
    #---------------------------------------------------------------------------------

    def _add_warmup_to_files(self, file_list: List[Path]) -> List[Path]:
        """
        Add warm-up period to processed meteorological files by repeating the first year
        ✅ FIXED: Properly handles elevation (keeps it 2D, time-invariant)
        ✅ FIXED: Uses temporary file to avoid permission errors
        
        Parameters
        ----------
        file_list : List[Path]
            List of processed daily NetCDF files
            
        Returns
        -------
        List[Path]
            List of files with warm-up period included
        """
        self.logger.info("Adding warm-up period to meteorological files...")
        
        # Calculate how many years of warm-up we need
        years_diff = (self.start_date - self.warmup_date).days / 365.25
        n_repetitions = max(1, int(np.ceil(years_diff)))
        
        self.logger.info(f"📅 Warm-up configuration:")
        self.logger.info(f"   Period: {self.warmup_date.date()} to {(self.start_date - pd.Timedelta(days=1)).date()}")
        self.logger.info(f"   Repetitions: {n_repetitions} year(s)")
        
        updated_files = []
        
        for file_path in file_list:
            # ✅ SKIP ELEVATION FILE - it has no time dimension
            if 'elevation' in file_path.name:
                self.logger.info(f"⏭️  Skipping {file_path.name} (elevation is time-invariant)")
                updated_files.append(file_path)
                continue
            
            try:
                self.logger.info(f"Processing {file_path.name}...")
                
                # Load the file
                ds = xr.open_dataset(file_path)
                
                # ✅ CRITICAL: Extract and REMOVE elevation BEFORE any processing
                has_elevation = 'elevation' in ds.data_vars
                elevation_data = None
                
                if has_elevation:
                    self.logger.debug("  Found elevation in dataset - extracting it")
                    elevation_data = ds['elevation'].copy()
                    
                    # Verify it has no time dimension
                    if 'time' in elevation_data.dims:
                        self.logger.error("❌ ELEVATION HAS TIME DIMENSION IN SAVED FILE!")
                        self.logger.error(f"   Elevation dims: {elevation_data.dims}")
                        self.logger.error("   Selecting first timestep to fix...")
                        elevation_data = elevation_data.isel(time=0)
                    
                    self.logger.debug(f"  Elevation shape: {elevation_data.shape}, dims: {elevation_data.dims}")
                    
                    # ✅ CRITICAL: Drop elevation from dataset BEFORE concatenation
                    ds = ds.drop_vars('elevation')
                    self.logger.debug("  ✅ Removed elevation from dataset before warm-up processing")
                
                # Get the main meteorological variable
                data_vars = [v for v in ds.data_vars]
                if not data_vars:
                    self.logger.warning(f"No data variables in {file_path.name}, skipping")
                    ds.close()
                    updated_files.append(file_path)
                    continue
                
                main_var = data_vars[0]
                
                # ✅ CHECK: Make sure the variable has a time dimension
                if 'time' not in ds[main_var].dims:
                    self.logger.warning(f"{main_var} has no time dimension, skipping warm-up")
                    ds.close()
                    updated_files.append(file_path)
                    continue
                
                # Extract first year of simulation data (NOW without elevation)
                first_year_end = self.start_date + pd.DateOffset(years=1) - pd.Timedelta(days=1)
                
                # Make sure we don't go beyond the end date
                if first_year_end > self.end_date:
                    first_year_end = self.end_date
                    self.logger.warning(f"First year extends beyond end_date for {file_path.name}")
                
                # Get first year of data (WITHOUT elevation)
                first_year = ds.sel(time=slice(self.start_date, first_year_end))
                
                self.logger.debug(f"  First year: {first_year.time.min().values} to {first_year.time.max().values}")
                self.logger.debug(f"  Days: {len(first_year.time)}")
                
                # Repeat the first year n times
                repeated_datasets = []
                current_time = pd.to_datetime(self.warmup_date)
                
                for i in range(n_repetitions):
                    # Create a copy with adjusted time coordinates
                    year_copy = first_year.copy(deep=True)
                    
                    # Calculate new time coordinates
                    time_deltas = first_year.time - first_year.time.values[0]
                    new_times = current_time + time_deltas
                    
                    # Update time coordinate
                    year_copy['time'] = new_times
                    repeated_datasets.append(year_copy)
                    
                    # Move to next year
                    current_time = new_times.values[-1] + pd.Timedelta(days=1)
                    
                    self.logger.debug(f"  Repetition {i+1}: {new_times.values[0]} to {new_times.values[-1]}")
                
                # ✅ CONCATENATE ONLY METEOROLOGICAL DATA (no elevation in either dataset)
                warmup_data = xr.concat(repeated_datasets, dim='time')
                
                # Trim warm-up to exact period (warmup_date to start_date - 1 day)
                warmup_end = self.start_date - pd.Timedelta(days=1)
                warmup_data = warmup_data.sel(time=slice(self.warmup_date, warmup_end))
                
                self.logger.info(f"  Warm-up: {warmup_data.time.min().values} to {warmup_data.time.max().values} ({len(warmup_data.time)} days)")
                
                # ✅ CONCATENATE (both datasets have NO elevation now)
                combined = xr.concat([warmup_data, ds], dim='time')
                
                self.logger.info(f"  Combined: {combined.time.min().values} to {combined.time.max().values} ({len(combined.time)} days)")
                
                # ✅ NOW: Add elevation AFTER concatenation (as time-invariant 2D array)
                if has_elevation and elevation_data is not None:
                    self.logger.debug("  Re-adding elevation (time-invariant)")
                    
                    # Verify elevation is 2D
                    if len(elevation_data.dims) != 2:
                        self.logger.error(f"❌ Elevation is not 2D: {elevation_data.dims}")
                        raise ValueError(f"Elevation must be 2D, got {elevation_data.dims}")
                    
                    # Add elevation as a simple 2D variable (no time dimension)
                    combined['elevation'] = elevation_data
                    
                    # ✅ VERIFICATION: Check elevation dimensions in combined dataset
                    if 'time' in combined['elevation'].dims:
                        self.logger.error("❌ ELEVATION GAINED TIME DIMENSION AFTER ADDING TO COMBINED!")
                        self.logger.error(f"   Combined dims: {combined['elevation'].dims}")
                        raise ValueError("Elevation must not have time dimension!")
                    else:
                        self.logger.debug(f"  ✅ Elevation correctly added: shape={combined['elevation'].shape}, dims={combined['elevation'].dims}")
                
                # Update metadata
                combined.attrs.update({
                    'warmup_included': 'true',
                    'warmup_start': str(self.warmup_date.date()),
                    'warmup_end': str(warmup_end.date()),
                    'warmup_days': len(warmup_data.time),
                    'simulation_start': str(self.start_date.date()),
                    'simulation_end': str(self.end_date.date()),
                    'simulation_days': len(ds.time),
                    'warmup_method': 'repeat_first_year',
                    'warmup_repetitions': n_repetitions,
                    'total_days': len(combined.time),
                    'elevation_included': 'true' if has_elevation else 'false'
                })
                
                # Load into memory FIRST while ds is still open, THEN close originals.
                # Closing before .load() causes a hang: combined still lazily references ds.
                combined = combined.load()
                ds.close()
                warmup_data.close()

                # ✅ NEW: Save to temporary file first, then replace original
                import tempfile
                import shutil
                
                temp_file = file_path.parent / f".tmp_{file_path.name}"
                
                self.logger.debug(f"Saving to temporary file: {temp_file}...")
                combined.to_netcdf(temp_file)
                
                # Close combined dataset
                combined.close()
                
                # Replace original file with temporary file
                self.logger.debug(f"Replacing original file...")
                shutil.move(str(temp_file), str(file_path))
                
                self.logger.info(f"✅ Updated {file_path.name} with warm-up period")
                
                updated_files.append(file_path)
                
            except Exception as e:
                self.logger.error(f"Error adding warm-up to {file_path.name}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                updated_files.append(file_path)
        
        self.logger.info(f"✅ Warm-up period added to {len(updated_files)} files")
        
        # ✅ NEW: Copy updated files to model-specific directory
        self._copy_to_model_directory(updated_files)
        
        return updated_files
    
    #---------------------------------------------------------------------------------
    
    def plot_spatial_overview(self, netcdf_file: Path) -> None:
        """
        Plot spatial overview of the first timestep for each variable in the NetCDF file
        INCLUDING elevation data if available
        
        Parameters
        ----------
        netcdf_file : Path
            Path to the NetCDF file to plot
        """
        self.logger.info(f"Creating spatial overview plots for {netcdf_file.name}")
        
        try:
            # Open the dataset
            ds = xr.open_dataset(netcdf_file, chunks={'time': 100})
            
            # Filter time range
            ds = self._filter_time_range_exact(ds)
            
            # Get data variables (exclude coordinates and elevation)
            data_vars = [var for var in ds.data_vars if len(ds[var].dims) >= 2 and var != 'elevation']
            
            # ✅ Check if elevation is available in this file
            has_elevation = 'elevation' in ds.data_vars
            
            if has_elevation:
                self.logger.info("✅ Elevation data found in file - will include in plots")
            else:
                self.logger.debug("No elevation data in this file")
            
            if not data_vars and not has_elevation:
                self.logger.warning(f"No suitable variables found in {netcdf_file.name}")
                return
            
            # ✅ Calculate total number of subplots (meteo vars + elevation if available)
            total_plots = len(data_vars) + (1 if has_elevation else 0)
            
            # Create subplots
            cols = min(3, total_plots)
            rows = (total_plots + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if total_plots == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            # Load catchment shapefile
            catchment_shape = None
            try:
                # Format the shape_dir path with gauge_id
                shape_dir_formatted = self.config['shape_dir'].format(gauge_id=self.gauge_id)
                shape_path = Path(shape_dir_formatted)
                
                if not shape_path.is_absolute():
                    shape_path = self.main_dir / shape_path
                    
                self.logger.info(f"Looking for catchment shapefile at: {shape_path}")
                
                if shape_path.exists():
                    catchment_shape = gpd.read_file(shape_path)
                    self.logger.info(f"Loaded catchment shapefile with {len(catchment_shape)} features")
                    self.logger.info(f"Original catchment CRS: {catchment_shape.crs}")
                    
                    # Get data bounds to determine target CRS
                    if hasattr(ds, 'longitude') and hasattr(ds, 'latitude'):
                        lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
                        lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
                        self.logger.info(f"Data bounds: lon [{lon_min:.3f}, {lon_max:.3f}], lat [{lat_min:.3f}, {lat_max:.3f}]")
                        
                        # Assume WGS84 for lat/lon data
                        target_crs = 'EPSG:4326'
                        catchment_shape = catchment_shape.to_crs(target_crs)
                        self.logger.info(f"Reprojected catchment to: {target_crs}")
                        
                        # Check if catchment bounds overlap with data bounds
                        cat_bounds = catchment_shape.total_bounds
                        self.logger.info(f"Catchment bounds: lon [{cat_bounds[0]:.3f}, {cat_bounds[2]:.3f}], lat [{cat_bounds[1]:.3f}, {cat_bounds[3]:.3f}]")
                        
                        # Check overlap
                        lon_overlap = not (cat_bounds[2] < lon_min or cat_bounds[0] > lon_max)
                        lat_overlap = not (cat_bounds[3] < lat_min or cat_bounds[1] > lat_max)
                        
                        if lon_overlap and lat_overlap:
                            self.logger.info("✅ Catchment and data bounds overlap - should be visible")
                        else:
                            self.logger.warning("⚠️ Catchment and data bounds do NOT overlap!")
                            self.logger.warning(f"  Data: lon [{lon_min:.3f}, {lon_max:.3f}], lat [{lat_min:.3f}, {lat_max:.3f}]")
                            self.logger.warning(f"  Catchment: lon [{cat_bounds[0]:.3f}, {cat_bounds[2]:.3f}], lat [{cat_bounds[1]:.3f}, {cat_bounds[3]:.3f}]")
                    
                else:
                    self.logger.warning(f"Catchment shapefile not found: {shape_path}")
                    
            except Exception as e:
                self.logger.error(f"Error loading catchment shapefile: {e}")
                import traceback
                traceback.print_exc()
            
            fig.suptitle(f'Spatial Overview - Gauge {self.gauge_id}\n{netcdf_file.stem} - First Timestep\nDate Range: {self.start_date.strftime("%Y-%m-%d")} to {self.end_date.strftime("%Y-%m-%d")}', 
                        fontsize=14, fontweight='bold')
            
            # Plot each meteorological variable
            for i, var_name in enumerate(data_vars):
                ax = axes[i]
                
                # Convert units if needed
                ds_converted = self._convert_units(ds, var_name)
                
                # Get first timestep
                data = ds_converted[var_name].isel(time=0)
                
                # Get variable info
                var_info = self.era5_variables.get(var_name, {
                    'name': var_name,
                    'units': data.attrs.get('units', ''),
                    'cmap': 'viridis'
                })
                
                # Create the plot with explicit coordinates
                if hasattr(ds, 'longitude') and hasattr(ds, 'latitude'):
                    im = data.plot(ax=ax, x='longitude', y='latitude', cmap=var_info['cmap'], add_colorbar=False)
                else:
                    im = data.plot(ax=ax, cmap=var_info['cmap'], add_colorbar=False)
                
                # Overlay catchment boundary if available
                if catchment_shape is not None:
                    try:
                        catchment_shape.boundary.plot(ax=ax, color='red', linewidth=3, alpha=0.9, label='Catchment boundary')
                        catchment_shape.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=3, alpha=0.9)
                        ax.legend(loc='upper right', fontsize=10, framealpha=0.8)
                    except Exception as e:
                        self.logger.error(f"Error plotting catchment boundary: {e}")
                
                # Add colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_label(f"{var_info['units']}", rotation=270, labelpad=15)
                
                # Customize plot
                ax.set_title(f"{var_info['name']}")
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
            
            # ✅ Plot elevation in the last subplot if available
            if has_elevation:
                ax = axes[len(data_vars)]  # Use next available subplot
                
                elevation_data = ds['elevation']
                
                self.logger.debug(f"Plotting elevation data - shape: {elevation_data.shape}, dims: {elevation_data.dims}")
                
                # Plot elevation
                if hasattr(ds, 'longitude') and hasattr(ds, 'latitude'):
                    im = elevation_data.plot(ax=ax, x='longitude', y='latitude', cmap='terrain', add_colorbar=False)
                else:
                    im = elevation_data.plot(ax=ax, cmap='terrain', add_colorbar=False)
                
                # Overlay catchment boundary
                if catchment_shape is not None:
                    try:
                        catchment_shape.boundary.plot(ax=ax, color='red', linewidth=3, alpha=0.9, label='Catchment boundary')
                        catchment_shape.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=3, alpha=0.9)
                        ax.legend(loc='upper right', fontsize=10, framealpha=0.8)
                    except Exception as e:
                        self.logger.error(f"Error plotting catchment boundary on elevation: {e}")
                
                # Add colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_label("m asl", rotation=270, labelpad=15)
                
                # Customize plot
                ax.set_title("Elevation")
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                
                self.logger.debug("✅ Elevation plot completed")
            
            # Hide unused subplots
            for i in range(total_plots, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save the plot
            save_path = self.spatial_plots_dir / f"spatial_overview_gauge_{self.gauge_id}_{netcdf_file.stem}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Spatial overview saved to {save_path}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating spatial overview for {netcdf_file.name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    #---------------------------------------------------------------------------------
    
    def calculate_spatial_average_timeseries(self, netcdf_file: Path) -> pd.DataFrame:
        """
        Calculate spatial average time series for each variable in the NetCDF file
        ✅ FIXED: Don't filter time range - use ALL data (including warm-up)
        
        Parameters
        ----------
        netcdf_file : Path
            Path to the NetCDF file
            
        Returns
        -------
        pd.DataFrame
            Time series of spatially averaged data
        """
        self.logger.info(f"Calculating spatial averages for {netcdf_file.name}")
        
        try:
            # Open the dataset
            ds = xr.open_dataset(netcdf_file, chunks={'time': 100})
            
            # ❌ DON'T FILTER TIME RANGE - WE WANT THE WARM-UP DATA TOO!
            # ds = self._filter_time_range_exact(ds)  # <-- REMOVE THIS LINE
            
            # Get data variables
            data_vars = [var for var in ds.data_vars if len(ds[var].dims) >= 2]
            
            if not data_vars:
                self.logger.warning(f"No suitable variables found in {netcdf_file.name}")
                return pd.DataFrame()
            
            # Calculate spatial means for each variable
            results = {}
            
            for var_name in data_vars:
                self.logger.debug(f"Processing variable: {var_name}")
                
                # Convert units if needed
                ds_converted = self._convert_units(ds, var_name)
                
                # Calculate spatial mean (average over lat/lon)
                spatial_dims = [dim for dim in ds_converted[var_name].dims if dim != 'time']
                spatial_mean = ds_converted[var_name].mean(dim=spatial_dims)
                
                # Convert to pandas series
                ts = spatial_mean.to_pandas()
                results[var_name] = ts
            
            # Combine into DataFrame
            df = pd.DataFrame(results)
            df.index.name = 'time'
            
            self.logger.info(f"Time series created: {df.index.min()} to {df.index.max()} ({len(df)} days)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating spatial averages for {netcdf_file.name}: {str(e)}")
            return pd.DataFrame()
        
    #---------------------------------------------------------------------------------
    
    def plot_timeseries(self, netcdf_file: Path, df_timeseries: pd.DataFrame) -> None:
        """
        Plot time series of spatially averaged variables
        ✅ MODIFIED: Highlights warm-up period with different color
        
        Parameters
        ----------
        netcdf_file : Path
            Path to the NetCDF file
        df_timeseries : pd.DataFrame
            Time series data to plot
        """
        if df_timeseries.empty:
            self.logger.warning(f"No data to plot for {netcdf_file.name}")
            return
        
        self.logger.info(f"Creating time series plots for {netcdf_file.name}")
        
        try:
            n_vars = len(df_timeseries.columns)
            
            # Create subplots
            fig, axes = plt.subplots(n_vars, 1, figsize=(14, 3*n_vars))
            if n_vars == 1:
                axes = [axes]
            
            # ✅ Updated title to show warm-up info
            title_warmup = ""
            if hasattr(self, 'warmup_date') and self.warmup_date is not None:
                title_warmup = f"\n🔄 Warm-up: {self.warmup_date.strftime('%Y-%m-%d')} to {(self.start_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}"
            
            fig.suptitle(f'Time Series - Gauge {self.gauge_id}\n{netcdf_file.stem} - Spatial Averages\n'
                        f'Period: {df_timeseries.index.min().strftime("%Y-%m-%d")} to {df_timeseries.index.max().strftime("%Y-%m-%d")}'
                        f'{title_warmup}', 
                        fontsize=14, fontweight='bold')
            
            for i, var_name in enumerate(df_timeseries.columns):
                ax = axes[i]
                
                # Get variable info
                var_info = self.era5_variables.get(var_name, {
                    'name': var_name,
                    'units': '',
                    'cmap': 'viridis'
                })
                
                # ✅ NEW: Split data into warm-up and simulation periods
                if hasattr(self, 'warmup_date') and self.warmup_date is not None:
                    # Separate warm-up and simulation data
                    warmup_mask = df_timeseries.index < self.start_date
                    simulation_mask = df_timeseries.index >= self.start_date
                    
                    warmup_data = df_timeseries.loc[warmup_mask, var_name]
                    simulation_data = df_timeseries.loc[simulation_mask, var_name]
                    
                    # Plot warm-up period in orange/gray
                    if len(warmup_data) > 0:
                        ax.plot(warmup_data.index, warmup_data.values, 
                            linewidth=1.5, color='orange', alpha=0.7, 
                            label='Warm-up period')
                    
                    # Plot simulation period in blue
                    if len(simulation_data) > 0:
                        ax.plot(simulation_data.index, simulation_data.values, 
                            linewidth=1, color='blue', 
                            label='Simulation period')
                    
                    # Add vertical line at start of simulation
                    ax.axvline(self.start_date, color='red', linestyle='--', 
                            linewidth=2, alpha=0.7, label='Simulation start')
                    
                    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
                    
                else:
                    # No warm-up - plot normally
                    df_timeseries[var_name].plot(ax=ax, linewidth=1, color='blue')
                
                # Customize plot
                ax.set_title(f"{var_info['name']}")
                ax.set_ylabel(f"{var_info['units']}")
                ax.grid(True, alpha=0.3)
                
                # Format x-axis based on data length
                date_range = (df_timeseries.index.max() - df_timeseries.index.min()).days
                
                if date_range > 365*2:  # More than 2 years
                    ax.xaxis.set_major_locator(mdates.YearLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                elif date_range > 365:  # More than 1 year
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                else:
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                
                # ✅ UPDATED: Statistics text box now shows warm-up vs simulation stats
                if hasattr(self, 'warmup_date') and self.warmup_date is not None and len(warmup_data) > 0:
                    warmup_mean = warmup_data.mean()
                    sim_mean = simulation_data.mean()
                    
                    stats_text = (f'Warm-up Mean: {warmup_mean:.2f}\n'
                                f'Simulation Mean: {sim_mean:.2f}\n'
                                f'Overall Min: {df_timeseries[var_name].min():.2f}\n'
                                f'Overall Max: {df_timeseries[var_name].max():.2f}')
                else:
                    mean_val = df_timeseries[var_name].mean()
                    std_val = df_timeseries[var_name].std()
                    min_val = df_timeseries[var_name].min()
                    max_val = df_timeseries[var_name].max()
                    
                    stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}'
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9)
            
            # Set x-label only on bottom plot
            axes[-1].set_xlabel('Date')
            
            plt.tight_layout()
            
            # Save the plot
            save_path = self.timeseries_plots_dir / f"timeseries_gauge_{self.gauge_id}_{netcdf_file.stem}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.debug(f"Time series plot saved to {save_path}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating time series plot for {netcdf_file.name}: {str(e)}")
    
    #---------------------------------------------------------------------------------
    
    def detect_anomalies(self, df_timeseries: pd.DataFrame, var_name: str, 
                        threshold: float = 3.0) -> pd.Series:
        """
        Detect anomalies in time series using z-score method
        
        Parameters
        ----------
        df_timeseries : pd.DataFrame
            Time series data
        var_name : str
            Variable name to check for anomalies
        threshold : float
            Z-score threshold for anomaly detection
            
        Returns
        -------
        pd.Series
            Boolean series indicating anomalies
        """
        if var_name not in df_timeseries.columns:
            return pd.Series(dtype=bool)
        
        data = df_timeseries[var_name].dropna()
        z_scores = np.abs((data - data.mean()) / data.std())
        
        return z_scores > threshold
    
    #---------------------------------------------------------------------------------
    
    def calculate_monthly_temperature_averages(self) -> pd.DataFrame:
        """
        Calculate monthly mean temperature averages (climatology) and save to CSV
        
        Returns
        -------
        pd.DataFrame
            Monthly temperature averages with columns: month, Temperature
        """
        self.logger.info("Calculating monthly temperature averages...")
        
        try:
            # Find the processed temperature mean file
            temp_mean_file = None
            for file_path in self.processed_files:
                if 'temp_mean' in file_path.name:
                    temp_mean_file = file_path
                    break
            
            if temp_mean_file is None:
                self.logger.error("Temperature mean file not found in processed files")
                return pd.DataFrame()
            
            self.logger.debug(f"Using temperature file: {temp_mean_file}")
            
            # Load processed temperature data
            ds = xr.open_dataset(temp_mean_file)
            temp_var = list(ds.data_vars)[0]  # Get the temperature variable name
            
            # ✅ NEW: Mask data to only include grid cells within catchment
            if self.catchment_extent is not None:
                self.logger.info("Masking temperature data to catchment extent...")
                
                # Get catchment bounds
                catchment_bounds = self.catchment_extent.total_bounds  # minx, miny, maxx, maxy
                
                # Create boolean masks using DataArray coordinates (proper broadcasting)
                lat_mask = (ds['latitude'] >= catchment_bounds[1]) & (ds['latitude'] <= catchment_bounds[3])
                lon_mask = (ds['longitude'] >= catchment_bounds[0]) & (ds['longitude'] <= catchment_bounds[2])
                
                # Apply mask to temperature data
                ds_masked = ds[temp_var].where(lat_mask & lon_mask, drop=False)
                
                # Calculate spatial average only over masked (catchment) area
                temp_spatial_avg = ds_masked.mean(dim=['latitude', 'longitude'], skipna=True)
                
                n_valid_cells = int(ds_masked.isel(time=0).count())
                self.logger.info(f"  Using {n_valid_cells} grid cells within catchment bounds")
            else:
                # Fallback: Calculate spatial average over entire grid
                self.logger.warning("No catchment shapefile - using entire grid")
                temp_spatial_avg = ds[temp_var].mean(dim=['latitude', 'longitude'])
            
            # Group by month and calculate mean for each month (climatology)
            # This creates a 12-month climatology by averaging all Januaries, all Februaries, etc.
            monthly_climatology = temp_spatial_avg.groupby('time.month').mean()
            
            # Convert to pandas DataFrame
            monthly_df = monthly_climatology.to_dataframe().reset_index()
            monthly_df = monthly_df.rename(columns={'month': 'month', temp_var: 'Temperature'})
            
            # Ensure we have exactly 12 months
            full_months = pd.DataFrame({'month': list(range(1, 13))})
            monthly_df = full_months.merge(monthly_df, on='month', how='left')
            
            # Fill any missing months with NaN or interpolate
            if monthly_df['Temperature'].isna().any():
                self.logger.warning("Some months have missing data - interpolating")
                monthly_df['Temperature'] = monthly_df['Temperature'].interpolate()
            
            # Round to reasonable precision
            monthly_df['Temperature'] = monthly_df['Temperature'].round(2)
            
            # Save to CSV
            output_file = self.output_path / 'monthly_temperature_averages.csv'
            monthly_df.to_csv(output_file, index=False)
            
            self.logger.info(f"Monthly temperature averages saved to: {output_file}")
            self.logger.info("Monthly temperature averages (°C):")
            
            # Log the values for verification
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            for i, row in monthly_df.iterrows():
                month_name = month_names[int(row['month']) - 1]
                temp = row['Temperature']
                self.logger.info(f"  {month_name}: {temp:.1f}°C")
            
            ds.close()
            return monthly_df
            
        except Exception as e:
            self.logger.error(f"Error calculating monthly temperature averages: {str(e)}")
            return pd.DataFrame()

    #---------------------------------------------------------------------------------

    def plot_monthly_temperature_climatology(self, monthly_df: pd.DataFrame) -> None:
        """
        Create a plot of monthly temperature climatology
        
        Parameters
        ----------
        monthly_df : pd.DataFrame
            Monthly temperature data
        """
        if monthly_df.empty:
            self.logger.warning("No monthly temperature data to plot")
            return
        
        self.logger.info("Creating monthly temperature climatology plot...")
        
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            # Create bar plot
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            bars = ax.bar(month_names, monthly_df['Temperature'], 
                         color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1)
            
            # Add value labels on bars
            for bar, temp in zip(bars, monthly_df['Temperature']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{temp:.1f}°C', ha='center', va='bottom', fontsize=10)
            
            # Customize plot
            ax.set_title(f'Monthly Temperature Climatology - Gauge {self.gauge_id}', 
                        fontsize=16, fontweight='bold')
            ax.set_ylabel('Temperature (°C)', fontsize=12)
            ax.set_xlabel('Month', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add horizontal line at 0°C
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1, label='0°C')
            ax.legend()
            
            # Set y-axis to include some padding
            temp_min, temp_max = monthly_df['Temperature'].min(), monthly_df['Temperature'].max()
            padding = (temp_max - temp_min) * 0.1
            ax.set_ylim(temp_min - padding - 2, temp_max + padding + 2)
            
            plt.tight_layout()
            
            # Save plot
            save_path = self.plots_dir / f'monthly_temperature_climatology_gauge_{self.gauge_id}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Monthly temperature climatology plot saved to: {save_path}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating monthly temperature climatology plot: {str(e)}")

    #---------------------------------------------------------------------------------

    def calculate_monthly_pet_averages(self) -> pd.DataFrame:
        """
        Calculate monthly mean PET averages (climatology) and save to CSV
        
        Returns
        -------
        pd.DataFrame
            Monthly PET averages with columns: month, PET_avg_mm_per_day
        """
        self.logger.info("Calculating monthly PET averages...")
        
        try:
            # Find the processed PET file
            pet_file = None
            for file_path in self.processed_files:
                if 'pet' in file_path.name:
                    pet_file = file_path
                    break
            
            if pet_file is None:
                self.logger.error("PET file not found in processed files")
                return pd.DataFrame()
            
            self.logger.debug(f"Using PET file: {pet_file}")
            
            # Load processed PET data
            ds = xr.open_dataset(pet_file)
            pet_var = list(ds.data_vars)[0]  # Get the PET variable name
            
            # ✅ NEW: Mask data to only include grid cells within catchment
            if self.catchment_extent is not None:
                self.logger.info("Masking PET data to catchment extent...")
                
                # Get catchment bounds
                catchment_bounds = self.catchment_extent.total_bounds  # minx, miny, maxx, maxy
                
                # Create boolean masks using DataArray coordinates (proper broadcasting)
                lat_mask = (ds['latitude'] >= catchment_bounds[1]) & (ds['latitude'] <= catchment_bounds[3])
                lon_mask = (ds['longitude'] >= catchment_bounds[0]) & (ds['longitude'] <= catchment_bounds[2])
                
                # Apply mask to PET data
                ds_masked = ds[pet_var].where(lat_mask & lon_mask, drop=False)
                
                # Calculate spatial average only over masked (catchment) area
                pet_spatial_avg = ds_masked.mean(dim=['latitude', 'longitude'], skipna=True)
                
                n_valid_cells = int(ds_masked.isel(time=0).count())
                self.logger.info(f"  Using {n_valid_cells} grid cells within catchment bounds")
            else:
                # Fallback: Calculate spatial average over entire grid
                self.logger.warning("No catchment shapefile - using entire grid")
                pet_spatial_avg = ds[pet_var].mean(dim=['latitude', 'longitude'])
            
            # Group by month and calculate mean for each month (climatology)
            # This creates a 12-month climatology by averaging all Januaries, all Februaries, etc.
            monthly_climatology = pet_spatial_avg.groupby('time.month').mean()
            
            # Convert to pandas DataFrame
            monthly_df = monthly_climatology.to_dataframe().reset_index()
            monthly_df = monthly_df.rename(columns={'month': 'month', pet_var: 'PET_avg_mm_per_day'})
            
            # Ensure we have exactly 12 months
            full_months = pd.DataFrame({'month': list(range(1, 13))})
            monthly_df = full_months.merge(monthly_df, on='month', how='left')
            
            # Fill any missing months with NaN or interpolate
            if monthly_df['PET_avg_mm_per_day'].isna().any():
                self.logger.warning("Some months have missing PET data - interpolating")
                monthly_df['PET_avg_mm_per_day'] = monthly_df['PET_avg_mm_per_day'].interpolate()
            
            # Round to reasonable precision
            monthly_df['PET_avg_mm_per_day'] = monthly_df['PET_avg_mm_per_day'].round(3)
            
            # Save to CSV
            output_file = self.output_path / 'monthly_pet_averages.csv'
            monthly_df.to_csv(output_file, index=False)
            
            self.logger.info(f"Monthly PET averages saved to: {output_file}")
            self.logger.info("Monthly PET averages (mm/day):")
            
            # Log the values for verification
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            for i, row in monthly_df.iterrows():
                month_name = month_names[int(row['month']) - 1]
                pet = row['PET_avg_mm_per_day']
                self.logger.info(f"  {month_name}: {pet:.3f} mm/day")
            
            ds.close()
            return monthly_df
            
        except Exception as e:
            self.logger.error(f"Error calculating monthly PET averages: {str(e)}")
            return pd.DataFrame()

    #---------------------------------------------------------------------------------

    def plot_monthly_pet_climatology(self, monthly_df: pd.DataFrame) -> None:
        """
        Create a plot of monthly PET climatology
        
        Parameters
        ----------
        monthly_df : pd.DataFrame
            Monthly PET data
        """
        if monthly_df.empty:
            self.logger.warning("No monthly PET data to plot")
            return
        
        self.logger.info("Creating monthly PET climatology plot...")
        
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            # Create bar plot
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            bars = ax.bar(month_names, monthly_df['PET_avg_mm_per_day'], 
                         color='orange', alpha=0.7, edgecolor='darkorange', linewidth=1)
            
            # Add value labels on bars
            for bar, pet in zip(bars, monthly_df['PET_avg_mm_per_day']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{pet:.2f}', ha='center', va='bottom', fontsize=10)
            
            # Customize plot
            ax.set_title(f'Monthly PET Climatology - Gauge {self.gauge_id}', 
                        fontsize=16, fontweight='bold')
            ax.set_ylabel('PET (mm/day)', fontsize=12)
            ax.set_xlabel('Month', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set y-axis to start from 0
            pet_max = monthly_df['PET_avg_mm_per_day'].max()
            ax.set_ylim(0, pet_max * 1.1)
            
            plt.tight_layout()
            
            # Save plot
            save_path = self.plots_dir / f'monthly_pet_climatology_gauge_{self.gauge_id}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Monthly PET climatology plot saved to: {save_path}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating monthly PET climatology plot: {str(e)}")

    #---------------------------------------------------------------------------------

    def get_output_directory_info(self) -> Dict[str, Path]:
        """
        Get information about output directories and structure
        
        Returns
        -------
        Dict[str, Path]
            Dictionary with directory information
        """
        return {
            'output_path': self.output_path,
            'catchment_dir': self.model_dir / f'catchment_{self.gauge_id}',
            'model_type_dir': self.model_dir / f'catchment_{self.gauge_id}' / self.model_type,
            'data_obs_dir': self.output_path,
            'plots_dir': self.plots_dir,
            'spatial_plots_dir': self.spatial_plots_dir,
            'timeseries_plots_dir': self.timeseries_plots_dir
        }

    #---------------------------------------------------------------------------------

    def analyze_all_files(self) -> None:
        """
        Main method to analyze all processed daily files
        """
        self.logger.info(f"Starting analysis of processed ERA5-Land data for gauge {self.gauge_id}")
        self.logger.info(f"Analysis period: {self.start_date.date()} to {self.end_date.date()}")
        self.logger.info(f"Output directory: {self.output_path}")
        
        if not self.processed_files:
            self.logger.warning("No processed files found to analyze")
            return
        
        all_timeseries = {}
        
        for processed_file in self.processed_files:
            try:
                self.logger.info(f"Analyzing file: {processed_file.name}")
                
                # Create spatial overview plot
                self.plot_spatial_overview(processed_file)
                
                # Calculate and plot time series
                df_timeseries = self.calculate_spatial_average_timeseries(processed_file)
                if not df_timeseries.empty:
                    self.plot_timeseries(processed_file, df_timeseries)
                    all_timeseries[processed_file.stem] = df_timeseries
                
            except Exception as e:
                self.logger.error(f"Error analyzing {processed_file.name}: {str(e)}")
                continue
        
        # Calculate and save monthly temperature averages
        monthly_temp_df = self.calculate_monthly_temperature_averages()
        if not monthly_temp_df.empty:
            self.plot_monthly_temperature_climatology(monthly_temp_df)
        
        # Calculate and save monthly PET averages
        monthly_pet_df = self.calculate_monthly_pet_averages()
        if not monthly_pet_df.empty:
            self.plot_monthly_pet_climatology(monthly_pet_df)
        
        self.logger.info("Analysis complete!")
        self.logger.info(f"Plots saved in: {self.plots_dir}")
        self.logger.info(f"Processed meteo files saved in: {self.output_path}")
        self.logger.info(f"Monthly temperature averages saved in: {self.output_path / 'monthly_temperature_averages.csv'}")
        self.logger.info(f"Monthly PET averages saved in: {self.output_path / 'monthly_pet_averages.csv'}")
        self.logger.info(f"Analyzed {len(self.processed_files)} files successfully")

    #---------------------------------------------------------------------------------

    def get_processed_files_info(self) -> Dict[str, Path]:
        """
        Get information about processed files
        
        Returns
        -------
        Dict[str, Path]
            Dictionary mapping variable names to file paths
        """
        file_info = {}
        
        for file_path in self.processed_files:
            filename = file_path.name
            
            if 'temp_mean' in filename:
                file_info['temperature_mean'] = file_path
            elif 'temp_min' in filename:
                file_info['temperature_min'] = file_path
            elif 'temp_max' in filename:
                file_info['temperature_max'] = file_path
            elif 'precip' in filename:
                file_info['precipitation'] = file_path
        
        return file_info
    
    #---------------------------------------------------------------------------------
    
    def debug_precipitation_values(self, netcdf_file: Path) -> None:
        """
        Debug precipitation values to understand the extreme values
        """
        self.logger.info(f"🔍 Debugging precipitation values in {netcdf_file.name}")
        
        try:
            # Open the processed daily file
            ds = xr.open_dataset(netcdf_file)
            
            # Get the precipitation variable
            precip_vars = [var for var in ds.data_vars if 'tp' in var or 'precip' in var.lower()]
            if not precip_vars:
                self.logger.warning("No precipitation variable found")
                return
                
            precip_var = precip_vars[0]
            precip_data = ds[precip_var]
            
            self.logger.info(f"Variable: {precip_var}")
            self.logger.info(f"Units: {precip_data.attrs.get('units', 'Unknown')}")
            self.logger.info(f"Data shape: {precip_data.shape}")
            
            # Calculate spatial average
            spatial_mean = precip_data.mean(dim=['latitude', 'longitude'])
            
            # Get statistics
            daily_values = spatial_mean.values
            daily_values = daily_values[~np.isnan(daily_values)]
            
            self.logger.info(f"\n📊 Precipitation Statistics:")
            self.logger.info(f"  Min: {np.min(daily_values):.3f} mm/day")
            self.logger.info(f"  Max: {np.max(daily_values):.3f} mm/day")
            self.logger.info(f"  Mean: {np.mean(daily_values):.3f} mm/day")
            self.logger.info(f"  Median: {np.median(daily_values):.3f} mm/day")
            self.logger.info(f"  95th percentile: {np.percentile(daily_values, 95):.3f} mm/day")
            self.logger.info(f"  99th percentile: {np.percentile(daily_values, 99):.3f} mm/day")
            
            # Find extreme days
            extreme_days = daily_values > 100  # More than 100mm/day
            if np.any(extreme_days):
                n_extreme = np.sum(extreme_days)
                self.logger.warning(f"⚠️ Found {n_extreme} days with >100mm precipitation!")
                
                # Show the most extreme values
                extreme_indices = np.where(extreme_days)[0]
                extreme_values = daily_values[extreme_indices]
                extreme_dates = pd.to_datetime(spatial_mean.time.values)[extreme_indices]
                
                self.logger.warning("Most extreme precipitation days:")
                for i, (date, value) in enumerate(zip(extreme_dates, extreme_values)):
                    if i < 10:  # Show top 10
                        self.logger.warning(f"  {date.strftime('%Y-%m-%d')}: {value:.1f} mm")
            
            # Check if values look like they're in wrong units
            if np.mean(daily_values) > 50:
                self.logger.error("❌ Average daily precipitation > 50mm - likely unit conversion error!")
            elif np.max(daily_values) > 500:
                self.logger.error("❌ Max daily precipitation > 500mm - likely accumulation error!")
            
        except Exception as e:
            self.logger.error(f"Error debugging precipitation: {e}")

    #---------------------------------------------------------------------------------

    def debug_hourly_precipitation(self, monthly_files: List[Path]) -> None:
        """
        Debug hourly precipitation values before daily aggregation
        """
        self.logger.info("🔍 Debugging original hourly precipitation data...")
        
        try:
            # Open first file to check
            sample_file = monthly_files[0]
            ds = xr.open_dataset(sample_file)
            
            # Find precipitation variable
            precip_vars = [var for var in ds.data_vars if 'tp' in var]
            if not precip_vars:
                self.logger.warning("No precipitation variable found")
                return
                
            precip_var = precip_vars[0]
            precip_data = ds[precip_var]
            
            self.logger.info(f"Original variable: {precip_var}")
            self.logger.info(f"Original units: {precip_data.attrs.get('units', 'Unknown')}")
            self.logger.info(f"Original shape: {precip_data.shape}")
            
            # Sample some values
            sample_values = precip_data.isel(time=slice(0, 24)).mean(dim=['latitude', 'longitude']).values
            
            self.logger.info(f"Sample hourly values (first 24 hours):")
            for i, val in enumerate(sample_values):
                self.logger.info(f"  Hour {i:2d}: {val:.6f} {precip_data.attrs.get('units', 'units')}")
            
            # Check if values are accumulation or rate
            daily_sum_sample = np.sum(sample_values)
            self.logger.info(f"Sum of first 24 hours: {daily_sum_sample:.6f}")
            
            if precip_data.attrs.get('units') == 'm':
                daily_sum_mm = daily_sum_sample * 1000
                self.logger.info(f"Sum converted to mm: {daily_sum_mm:.3f} mm")
                
                if daily_sum_mm > 100:
                    self.logger.error("❌ 24-hour sum > 100mm - this looks wrong!")
            
        except Exception as e:
            self.logger.error(f"Error debugging hourly precipitation: {e}")


#--------------------------------------------------------------------------------
############################### HARAnalyzer Class ###############################
#--------------------------------------------------------------------------------

class HARAnalyzer:
    """
    A class for analyzing and processing HAR (High Asia Refined) meteorological data
    
    HAR v2 characteristics:
    - 10km resolution (d10km domain)
    - Lambert Conformal Conic projection
    - Daily data in yearly files
    - Variables: t2 (temperature), prcp (precipitation), potevap (potential evaporation)
    """
    
    def __init__(self, namelist_path: Union[str, Path], force_reprocess: bool = False) -> None:
        """
        Initialize the HAR data analyzer
        
        Parameters
        ----------
        namelist_path : str or Path
            Path to the namelist YAML configuration file
        force_reprocess : bool, optional
            If True, reprocess files even if they already exist (default: False)
        """
        # Store the force_reprocess flag
        self.force_reprocess = force_reprocess
        
        # Load configuration from namelist directly
        namelist_path = Path(namelist_path)
        
        if not namelist_path.exists():
            raise FileNotFoundError(f"Namelist file not found: {namelist_path}")
        
        with open(namelist_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Store configuration parameters
        self.main_dir = Path(self.config.get('main_dir'))
        self.gauge_id = self.config.get('gauge_id')
        self.basin_id = self.config.get('basin_id', self.gauge_id)
        self.start_date = pd.to_datetime(self.config.get('start_date'))
        self.end_date = pd.to_datetime(self.config.get('end_date'))
        self.model_type = self.config.get('model_type')
        self.debug = self.config.get('debug', False)
        self.coupled = self.config.get('coupled', False)
        self.model_dir = self.main_dir / self.config.get('config_dir')
        
        # ✅ HAR-specific: Get HAR data directory
        har_dir_template = self.config.get('meteo_har_dir', '01_data/meteo/HAR')
        self.har_data_dir = Path(har_dir_template)
        
        # Make absolute path if needed
        if not self.har_data_dir.is_absolute():
            self.har_data_dir = self.main_dir / self.har_data_dir
        
        # ✅ Define SHARED catchment-level directories (primary storage)
        self.shared_data_dir = self.model_dir / f'catchment_{self.gauge_id}' / 'data_obs'
        self.shared_plots_dir = self.model_dir / f'catchment_{self.gauge_id}' / 'plots'
        
        # ✅ Define MODEL-SPECIFIC directories (for backward compatibility)
        self.model_data_dir = self.model_dir / f'catchment_{self.gauge_id}' / self.model_type / 'data_obs'
        
        # ✅ Use shared directories as primary output locations
        self.output_path = self.shared_data_dir
        self.plots_dir = self.shared_plots_dir
        
        # Create plots directories
        self.spatial_plots_dir = self.plots_dir / 'spatial_overview'
        self.timeseries_plots_dir = self.plots_dir / 'time_series'
        
        # Create all directories
        self.spatial_plots_dir.mkdir(parents=True, exist_ok=True)
        self.timeseries_plots_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.model_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Keep the old attribute name for backward compatibility
        self.processed_data_dir = self.output_path
        
        # HAR variable information
        self.har_variables = {
            't2_mean': {
                'name': '2m Temperature (Mean)',
                'units': '°C',
                'cmap': 'RdYlBu_r',
                'source_units': 'K',
                'file_pattern': 'HARv2_d10km_d_2d_t2_mean_{year}.nc'
            },
            't2_min': {
                'name': '2m Temperature (Min)',
                'units': '°C',
                'cmap': 'RdYlBu_r',
                'source_units': 'K',
                'file_pattern': 'HARv2_d10km_d_2d_t2_min_{year}.nc'
            },
            't2_max': {
                'name': '2m Temperature (Max)',
                'units': '°C',
                'cmap': 'RdYlBu_r',
                'source_units': 'K',
                'file_pattern': 'HARv2_d10km_d_2d_t2_max_{year}.nc'
            },
            'prcp': {
                'name': 'Total Precipitation',
                'units': 'mm/day',
                'cmap': 'Blues',
                'source_units': 'mm h-1',
                'file_pattern': 'HARv2_d10km_d_2d_prcp_{year}.nc'
            },
            'potevap': {
                'name': 'Potential Evapotranspiration',
                'units': 'mm/day',
                'cmap': 'Oranges',
                'source_units': 'mm h-1',
                'file_pattern': 'HARv2_d10km_d_2d_potevap_{year}.nc'
            },
            'hgt': {
                'name': 'Terrain Height',
                'units': 'm',
                'cmap': 'terrain',
                'source_units': 'm',
                'file_pattern': 'HARv2_d10km_static_hgt.nc'
            }
        }
        
        # ✅ Setup logger FIRST
        self.logger = self._setup_logger()
        
        # ✅ Load warm-up date AFTER logger is created
        if 'warm_up_date' in self.config:
            self.warmup_date = pd.to_datetime(self.config.get('warm_up_date'))
            self.logger.info(f"Warm-up period configured: {self.warmup_date.date()} to {(self.start_date - pd.Timedelta(days=1)).date()}")
        else:
            self.warmup_date = None
            self.logger.debug("No warm-up period configured")
        
        # ✅ Load catchment shapefile for clipping
        self.catchment_extent = self._load_catchment_shapefile()
        
        # Log the output directory
        self.logger.info(f"Processing HAR data for gauge {self.gauge_id}")
        self.logger.info(f"Model type: {self.model_type}")
        self.logger.info(f"HAR data directory: {self.har_data_dir}")
        self.logger.info(f"Plots will be saved to: {self.plots_dir}")
        self.logger.info(f"Processed meteo files will be saved to: {self.output_path}")
        
        # Check for existing files before processing
        existing_files = self._check_existing_files()
        existing_count = sum(existing_files.values())
        total_expected = len(existing_files)
        
        if existing_count == total_expected and not force_reprocess:
            self.logger.info(f"🎉 All {total_expected} processed HAR files already exist!")
            self.logger.info("⏭️ Skipping processing. Set force_reprocess=True to reprocess anyway.")
            
            # Build list of existing files
            expected_files = {
                'temperature_mean': 'har_temp_mean.nc',
                'temperature_min': 'har_temp_min.nc',
                'temperature_max': 'har_temp_max.nc',
                'precipitation': 'har_precip.nc',
                'potential_evaporation': 'har_pet.nc'
            }
            
            self.processed_files = []
            for file_type, exists in existing_files.items():
                if exists:
                    self.processed_files.append(self.output_path / expected_files[file_type])
            
            # Copy existing files to model-specific directory
            self.logger.info("📋 Copying existing files to model-specific directory...")
            self._copy_to_model_directory(self.processed_files)
                    
        elif existing_count > 0 and not force_reprocess:
            self.logger.info(f"📂 Found {existing_count}/{total_expected} existing files")
            self.logger.info("🔄 Will only process missing files. Set force_reprocess=True to reprocess all.")
            self.processed_files = self._find_and_process_yearly_files()
        else:
            if force_reprocess and existing_count > 0:
                self.logger.info(f"🔄 Reprocessing all files (force_reprocess=True)")
            
            # Find and process yearly files
            self.processed_files = self._find_and_process_yearly_files()
        
        self.logger.info(f"Available files: {len(self.processed_files)} daily files for gauge {self.gauge_id}")

        # Automatically run analysis and create plots
        if self.processed_files:
            self.logger.info("Starting automatic analysis and plotting...")
            self.analyze_all_files()
            # ✅ NEW: Create comprehensive missing values report
            missing_report = self.create_missing_values_report()
        else:
            self.logger.warning("No files processed - skipping analysis")

    #---------------------------------------------------------------------------------

    def _load_catchment_shapefile(self) -> Optional[gpd.GeoDataFrame]:
        """
        Load the catchment shapefile for the gauge_id for clipping HAR data
        
        Returns
        -------
        Optional[gpd.GeoDataFrame]
            Catchment shapefile in WGS84 (EPSG:4326) or None if not found
        """
        self.logger.debug(f"Loading catchment shapefile for gauge ID: {self.gauge_id}")
        
        try:
            # Get shapefile path from config
            shape_dir_template = self.config.get('shape_dir', '01_data/topo/catchment_shapefile/catchment_shape_{gauge_id}.shp')
            shape_path = Path(shape_dir_template.format(gauge_id=self.gauge_id))
            
            # Make absolute path if needed
            if not shape_path.is_absolute():
                shape_path = self.main_dir / shape_path
            
            self.logger.debug(f"Looking for catchment shapefile at: {shape_path}")
            
            if not shape_path.exists():
                self.logger.warning(f"Catchment shapefile not found: {shape_path}")
                self.logger.warning("⚠️ Processing will continue without catchment clipping")
                return None
            
            # Read the catchment shapefile
            extent = gpd.read_file(shape_path)
            
            self.logger.info(f"✅ Loaded catchment shapefile with {len(extent)} features")
            self.logger.info(f"   Original CRS: {extent.crs}")
            
            # Reproject to WGS84 (EPSG:4326) to match HAR lat/lon
            if extent.crs != 'EPSG:4326':
                extent = extent.to_crs('EPSG:4326')
                self.logger.info("   Reprojected to WGS84 (EPSG:4326)")
            
            # Get catchment bounds for logging
            bounds = extent.total_bounds
            self.logger.info(f"   Catchment bounds: lon [{bounds[0]:.4f}, {bounds[2]:.4f}], lat [{bounds[1]:.4f}, {bounds[3]:.4f}]")
            
            return extent
            
        except Exception as e:
            self.logger.error(f"Error loading catchment shapefile: {e}")
            self.logger.warning("⚠️ Processing will continue without catchment clipping")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    #---------------------------------------------------------------------------------

    def _setup_logger(self) -> logging.Logger:
        """Set up and configure logger based on debug flag"""
        level = logging.DEBUG if self.debug else logging.INFO
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            force=True
        )
        
        # Suppress matplotlib logging
        matplotlib_logger = logging.getLogger('matplotlib')
        matplotlib_logger.setLevel(logging.WARNING)
        
        # Also suppress other common noisy loggers
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.colorbar').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
        
        return logging.getLogger(f'HARAnalyzer_Gauge_{self.gauge_id}')

    #---------------------------------------------------------------------------------

    def _check_existing_files(self) -> Dict[str, bool]:
        """
        Check if HAR processed files already exist in the output directory
        
        Returns
        -------
        Dict[str, bool]
            Dictionary indicating which file types already exist
        """
        existing_files = {
            'temperature_mean': False,
            'temperature_min': False,
            'temperature_max': False,
            'precipitation': False,
            'potential_evaporation': False
        }
        
        # Check for each expected output file
        expected_files = {
            'temperature_mean': 'har_temp_mean.nc',
            'temperature_min': 'har_temp_min.nc',
            'temperature_max': 'har_temp_max.nc',
            'precipitation': 'har_precip.nc',
            'potential_evaporation': 'har_pet.nc'
        }
        
        for file_type, filename in expected_files.items():
            file_path = self.output_path / filename
            if file_path.exists():
                existing_files[file_type] = True
                self.logger.info(f"✅ Found existing file: {filename}")
            else:
                self.logger.debug(f"❌ Missing file: {filename}")
        
        return existing_files

    #---------------------------------------------------------------------------------

    def _find_yearly_files(self, variable: str) -> List[Path]:
        """
        Find all yearly HAR files for a specific variable within the date range
        
        Parameters
        ----------
        variable : str
            Variable name ('t2_mean', 't2_min', 't2_max', 'prcp', 'potevap')
            
        Returns
        -------
        List[Path]
            List of file paths found
        """
        if variable not in self.har_variables:
            self.logger.error(f"Unknown variable: {variable}")
            return []
        
        file_pattern = self.har_variables[variable]['file_pattern']
        
        # Determine years needed (including warm-up if configured)
        if self.warmup_date is not None:
            start_year = self.warmup_date.year
        else:
            start_year = self.start_date.year
        end_year = self.end_date.year
        
        files = []
        missing_years = []
        
        for year in range(start_year, end_year + 1):
            filename = file_pattern.format(year=year)
            filepath = self.har_data_dir / filename
            
            if filepath.exists() and filepath.stat().st_size > 0:
                files.append(filepath)
                self.logger.debug(f"Found {variable} file for {year}: {filename}")
            else:
                missing_years.append(year)
                self.logger.warning(f"Missing {variable} file for {year}: {filename}")
        
        if missing_years:
            self.logger.warning(f"Missing years for {variable}: {missing_years}")
        
        return sorted(files)

    #---------------------------------------------------------------------------------

    def _combine_yearly_files(self, file_list: List[Path], variable: str) -> Optional[xr.Dataset]:
        """
        Combine yearly HAR files into a single dataset
        FIXED: Preserve 2D lat/lon coordinates (don't let them become 3D)
        
        Parameters
        ----------
        file_list : List[Path]
            List of yearly NetCDF files
        variable : str
            Variable name
            
        Returns
        -------
        Optional[xr.Dataset]
            Combined dataset or None if failed
        """
        if not file_list:
            self.logger.warning(f"No files to combine for {variable}")
            return None
        
        self.logger.info(f"Combining {len(file_list)} {variable} files...")
        
        # Get clipping bounds
        sn_slice = None
        we_slice = None
        
        if self.catchment_extent is not None:
            bounds = self.catchment_extent.total_bounds
            buffer_deg = 0.2
            
            clip_bounds = {
                'lon_min': bounds[0] - buffer_deg,
                'lon_max': bounds[2] + buffer_deg,
                'lat_min': bounds[1] - buffer_deg,
                'lat_max': bounds[3] + buffer_deg
            }
            
            self.logger.info(f"✂️ Will clip to bounds: lon [{clip_bounds['lon_min']:.2f}, {clip_bounds['lon_max']:.2f}], "
                        f"lat [{clip_bounds['lat_min']:.2f}, {clip_bounds['lat_max']:.2f}]")
            
            # Determine clipping indices
            if self._clip_indices is not None:
                sn_slice = self._clip_indices['sn_slice']
                we_slice = self._clip_indices['we_slice']
                self.logger.info(f"📦 Reusing stored clip indices: south_north={sn_slice}, west_east={we_slice}")
            else:
                self.logger.debug("Determining clip indices from first file...")
                with xr.open_dataset(file_list[0]) as ds_first:
                    if 'lat' in ds_first.coords and 'lon' in ds_first.coords:
                        lat_2d = ds_first.coords['lat']
                        lon_2d = ds_first.coords['lon']
                        
                        mask = (
                            (lat_2d >= clip_bounds['lat_min']) & (lat_2d <= clip_bounds['lat_max']) &
                            (lon_2d >= clip_bounds['lon_min']) & (lon_2d <= clip_bounds['lon_max'])
                        )
                        
                        valid_south_north = mask.any(dim='west_east')
                        valid_west_east = mask.any(dim='south_north')
                        
                        sn_indices = np.where(valid_south_north.values)[0]
                        we_indices = np.where(valid_west_east.values)[0]
                        
                        if len(sn_indices) > 0 and len(we_indices) > 0:
                            sn_slice = slice(sn_indices.min(), sn_indices.max() + 1)
                            we_slice = slice(we_indices.min(), we_indices.max() + 1)
                            
                            self._clip_indices = {
                                'sn_slice': sn_slice,
                                'we_slice': we_slice,
                                'sn_min': sn_indices.min(),
                                'sn_max': sn_indices.max(),
                                'we_min': we_indices.min(),
                                'we_max': we_indices.max()
                            }
                            
                            original_size = ds_first.south_north.size * ds_first.west_east.size
                            clipped_size = (sn_indices.max() - sn_indices.min() + 1) * (we_indices.max() - we_indices.min() + 1)
                            reduction = (1 - clipped_size / original_size) * 100
                            
                            self.logger.info(f"📉 Memory optimization: {original_size:,} → {clipped_size:,} cells/timestep ({reduction:.1f}% reduction)")
                            self.logger.info(f"📦 Stored clip indices: south_north=[{sn_indices.min()}:{sn_indices.max()+1}], "
                                        f"west_east=[{we_indices.min()}:{we_indices.max()+1}]")
        
        try:
            datasets = []
            
            # ✅ KEY FIX: Extract 2D lat/lon from FIRST file BEFORE processing
            reference_lat = None
            reference_lon = None
            
            # Process files in batches
            batch_size = 5
            total_files = len(file_list)
            
            for batch_start in range(0, total_files, batch_size):
                batch_end = min(batch_start + batch_size, total_files)
                batch_files = file_list[batch_start:batch_end]
                
                self.logger.info(f"📦 Processing batch {batch_start//batch_size + 1}/{(total_files-1)//batch_size + 1}: "
                            f"years {batch_start+1}-{batch_end} of {total_files}")
                
                batch_datasets = []
                
                for file_path in batch_files:
                    self.logger.debug(f"Loading {file_path.name}...")
                    
                    ds = xr.open_dataset(file_path, chunks={'time': 30})
                    
                    # ✅ CRITICAL: Extract reference lat/lon from first file
                    if reference_lat is None and reference_lon is None:
                        if 'lat' in ds.coords and 'lon' in ds.coords:
                            reference_lat = ds.coords['lat'].values
                            reference_lon = ds.coords['lon'].values
                            self.logger.debug(f"Extracted reference lat/lon: {reference_lat.shape}")
                    
                    # Clip IMMEDIATELY
                    if sn_slice is not None and we_slice is not None:
                        ds = ds.isel(south_north=sn_slice, west_east=we_slice)
                    
                    # ✅ CRITICAL: Drop lat/lon coordinates before concatenation
                    # This prevents them from becoming time-dependent
                    ds = ds.drop_vars(['lat', 'lon'], errors='ignore')
                    
                    batch_datasets.append(ds)
                
                # Combine this batch
                if batch_datasets:
                    batch_combined = xr.concat(batch_datasets, dim='time', combine_attrs='drop_conflicts')
                    batch_combined = batch_combined.compute()
                    
                    for ds in batch_datasets:
                        ds.close()
                    
                    datasets.append(batch_combined)
                    self.logger.debug(f"  Batch combined: {len(batch_combined.time)} timesteps")
            
            if not datasets:
                self.logger.error(f"No datasets loaded for {variable}")
                return None
            
            # Final concatenation
            self.logger.info("Combining all batches...")
            combined = xr.concat(datasets, dim='time', combine_attrs='drop_conflicts')
            combined = combined.sortby('time')
            
            for ds in datasets:
                ds.close()
            
            # ✅ CRITICAL: Now add the 2D lat/lon coordinates AFTER concatenation
            if reference_lat is not None and reference_lon is not None:
                # Apply same clipping to reference coordinates
                if sn_slice is not None and we_slice is not None:
                    clipped_lat = reference_lat[sn_slice, we_slice]
                    clipped_lon = reference_lon[sn_slice, we_slice]
                else:
                    clipped_lat = reference_lat
                    clipped_lon = reference_lon
                
                # Add as 2D coordinates (NOT time-dependent!)
                combined = combined.assign_coords({
                    'lat': (['south_north', 'west_east'], clipped_lat),
                    'lon': (['south_north', 'west_east'], clipped_lon)
                })
                
                self.logger.debug(f"✅ Added 2D lat/lon coordinates: {clipped_lat.shape}")
            
            self.logger.info(f"✅ Combined {variable} dataset: {dict(combined.dims)}")
            self.logger.info(f"   Time range: {combined.time.min().values} to {combined.time.max().values}")
            self.logger.info(f"   Total timesteps: {len(combined.time)}")
            
            # Verify coordinate dimensions
            if 'lat' in combined.coords:
                self.logger.debug(f"   lat coordinate shape: {combined.coords['lat'].shape}, dims: {combined.coords['lat'].dims}")
            if 'lon' in combined.coords:
                self.logger.debug(f"   lon coordinate shape: {combined.coords['lon'].shape}, dims: {combined.coords['lon'].dims}")
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Error combining {variable} files: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    #---------------------------------------------------------------------------------

    def _add_warmup_to_files(self, file_list: List[Path]) -> List[Path]:
        """
        Add warm-up period to processed HAR meteorological files by repeating the first year
        ✅ FIXED: Properly handles elevation (keeps it 2D, time-invariant)
        ✅ FIXED: Uses temporary file to avoid permission errors
        ✅ FIXED: Ensures no duplicate timestamps at warm-up/simulation boundary
        
        Parameters
        ----------
        file_list : List[Path]
            List of processed daily NetCDF files
            
        Returns
        -------
        List[Path]
            List of files with warm-up period included
        """
        self.logger.info("Adding warm-up period to HAR meteorological files...")
        
        # Calculate how many years of warm-up we need
        years_diff = (self.start_date - self.warmup_date).days / 365.25
        n_repetitions = max(1, int(np.ceil(years_diff)))
        
        self.logger.info(f"📅 Warm-up configuration:")
        self.logger.info(f"   Period: {self.warmup_date.date()} to {(self.start_date - pd.Timedelta(days=1)).date()}")
        self.logger.info(f"   Repetitions: {n_repetitions} year(s)")
        
        updated_files = []
        
        for file_path in file_list:
            # ✅ SKIP ELEVATION FILE - it has no time dimension
            if 'elevation' in file_path.name.lower():
                self.logger.info(f"⏭️  Skipping {file_path.name} (elevation is time-invariant)")
                updated_files.append(file_path)
                continue

            try:
                self.logger.info(f"Processing {file_path.name}...")

                # Skip if warm-up was already added in a previous (possibly interrupted) run
                with xr.open_dataset(file_path) as _ds_check:
                    if _ds_check.attrs.get('warmup_included') == 'true':
                        self.logger.info(f"  ⏭️ Warm-up already present — skipping")
                        updated_files.append(file_path)
                        continue

                # Load the file
                ds = xr.open_dataset(file_path)
                
                # ✅ CRITICAL: Extract and REMOVE elevation BEFORE any processing
                has_elevation = 'elevation' in ds.data_vars
                elevation_data = None
                
                if has_elevation:
                    self.logger.debug("  Found elevation in dataset - extracting it")
                    elevation_data = ds['elevation'].copy()
                    
                    # Verify it has no time dimension
                    if 'time' in elevation_data.dims:
                        self.logger.error("❌ ELEVATION HAS TIME DIMENSION IN SAVED FILE!")
                        self.logger.error(f"   Elevation dims: {elevation_data.dims}")
                        self.logger.error("   Selecting first timestep to fix...")
                        elevation_data = elevation_data.isel(time=0)
                    
                    self.logger.debug(f"  Elevation shape: {elevation_data.shape}, dims: {elevation_data.dims}")
                    
                    # ✅ CRITICAL: Drop elevation from dataset BEFORE concatenation
                    ds = ds.drop_vars('elevation')
                    self.logger.debug("  ✅ Removed elevation from dataset before warm-up processing")
                
                # Get the main meteorological variable
                data_vars = [v for v in ds.data_vars]
                if not data_vars:
                    self.logger.warning(f"No data variables in {file_path.name}, skipping")
                    ds.close()
                    updated_files.append(file_path)
                    continue
                
                main_var = data_vars[0]
                
                # ✅ CHECK: Make sure the variable has a time dimension
                if 'time' not in ds[main_var].dims:
                    self.logger.warning(f"{main_var} has no time dimension, skipping warm-up")
                    ds.close()
                    updated_files.append(file_path)
                    continue
                
                # Load the entire dataset into memory NOW (sequential read = fast).
                # This must happen BEFORE any boolean isel() dedup, because calling
                # ds.isel(time=~bool_mask) on a lazy NetCDF dataset creates a fancy-
                # indexed view that forces per-row seeks when later .load()ed — which
                # is catastrophically slow on network filesystems.
                file_size_mb = file_path.stat().st_size / 1e6
                self.logger.info(f"  Loading {file_path.name} into memory ({file_size_mb:.1f} MB)...")
                ds = ds.load()
                self.logger.info(f"  ✅ Loaded into memory")

                # ✅ FIX: Check for existing duplicates in the original data (in-memory, fast)
                original_times = pd.to_datetime(ds.time.values)
                original_duplicates = original_times.duplicated()
                if original_duplicates.any():
                    self.logger.warning(f"⚠️ Original data has {original_duplicates.sum()} duplicate timestamps - removing them")
                    ds = ds.isel(time=~original_duplicates)  # in-memory operation, instant

                # Extract first year of simulation data (NOW without elevation)
                first_year_end = self.start_date + pd.DateOffset(years=1) - pd.Timedelta(days=1)
                
                # Make sure we don't go beyond the end date
                if first_year_end > self.end_date:
                    first_year_end = self.end_date
                    self.logger.warning(f"First year extends beyond end_date for {file_path.name}")
                
                # Get first year of data (WITHOUT elevation)
                first_year = ds.sel(time=slice(self.start_date, first_year_end))
                
                self.logger.debug(f"  First year: {first_year.time.min().values} to {first_year.time.max().values}")
                self.logger.debug(f"  Days: {len(first_year.time)}")
                
                # Repeat the first year n times
                repeated_datasets = []
                current_time = pd.to_datetime(self.warmup_date)
                
                for i in range(n_repetitions):
                    # Create a copy with adjusted time coordinates
                    year_copy = first_year.copy(deep=True)
                    
                    # Calculate new time coordinates
                    time_deltas = first_year.time - first_year.time.values[0]
                    new_times = current_time + time_deltas
                    
                    # Update time coordinate
                    year_copy['time'] = new_times
                    repeated_datasets.append(year_copy)
                    
                    # Move to next year
                    current_time = new_times.values[-1] + pd.Timedelta(days=1)
                    
                    self.logger.debug(f"  Repetition {i+1}: {new_times.values[0]} to {new_times.values[-1]}")
                
                # ✅ CONCATENATE ONLY METEOROLOGICAL DATA (no elevation in either dataset)
                warmup_data = xr.concat(repeated_datasets, dim='time')
                
                # ✅ FIX: Trim warm-up to EXACTLY end one day before simulation start
                # This ensures no overlap!
                warmup_end = self.start_date - pd.Timedelta(days=1)
                warmup_data = warmup_data.sel(time=slice(self.warmup_date, warmup_end))
                
                self.logger.info(f"  Warm-up: {warmup_data.time.min().values} to {warmup_data.time.max().values} ({len(warmup_data.time)} days)")
                
                # ✅ FIX: Verify no overlap before concatenation
                warmup_last_time = pd.to_datetime(warmup_data.time.values[-1])
                sim_first_time = pd.to_datetime(ds.time.values[0])
                
                self.logger.debug(f"  Last warm-up time: {warmup_last_time}")
                self.logger.debug(f"  First simulation time: {sim_first_time}")
                
                if warmup_last_time >= sim_first_time:
                    self.logger.warning(f"⚠️ Overlap detected! Trimming warm-up to avoid duplicate")
                    # Remove the last timestamp from warm-up if it overlaps
                    warmup_data = warmup_data.isel(time=slice(None, -1))
                    self.logger.info(f"  Trimmed warm-up: {warmup_data.time.min().values} to {warmup_data.time.max().values}")
                
                # Cache lengths before closing (used in metadata below)
                n_warmup_days = len(warmup_data.time)
                n_sim_days = len(ds.time)

                # ✅ CONCATENATE (both datasets have NO elevation now)
                combined = xr.concat([warmup_data, ds], dim='time')
                combined = combined.sortby('time')

                # Load into memory immediately — BEFORE any .time.values access and
                # before closing ds.  On a network FS, lazy evaluation of xr.concat
                # (which includes a boolean-indexed ds) hangs on the first .values call.
                combined = combined.load()
                ds.close()
                warmup_data.close()

                # Check for duplicates (all in-memory now — fast)
                combined_times = pd.to_datetime(combined.time.values)
                duplicates = combined_times.duplicated()
                if duplicates.any():
                    self.logger.warning(f"⚠️ Found {duplicates.sum()} duplicate timestamps after concatenation - removing them")
                    combined = combined.isel(time=~duplicates)

                self.logger.info(f"  Combined: {combined.time.min().values} to {combined.time.max().values} ({len(combined.time)} days)")

                # Check for consecutive time steps
                time_diffs = np.diff(combined.time.values).astype('timedelta64[D]').astype(int)
                non_consecutive = np.where(time_diffs != 1)[0]
                if len(non_consecutive) > 0:
                    self.logger.warning(f"⚠️ Found {len(non_consecutive)} non-consecutive time steps")
                    for idx in non_consecutive[:5]:
                        self.logger.warning(f"   Index {idx}: {combined.time.values[idx]} -> {combined.time.values[idx+1]} (gap: {time_diffs[idx]} days)")
                else:
                    self.logger.info(f"  ✅ All time steps are consecutive (1-day intervals)")

                # Re-add elevation after concatenation (time-invariant 2D array)
                if has_elevation and elevation_data is not None:
                    self.logger.debug("  Re-adding elevation (time-invariant)")
                    if len(elevation_data.dims) != 2:
                        self.logger.error(f"❌ Elevation is not 2D: {elevation_data.dims}")
                        raise ValueError(f"Elevation must be 2D, got {elevation_data.dims}")
                    combined['elevation'] = elevation_data
                    if 'time' in combined['elevation'].dims:
                        self.logger.error("❌ ELEVATION GAINED TIME DIMENSION AFTER ADDING TO COMBINED!")
                        raise ValueError("Elevation must not have time dimension!")
                    else:
                        self.logger.debug(f"  ✅ Elevation correctly added: shape={combined['elevation'].shape}, dims={combined['elevation'].dims}")

                # Update metadata
                combined.attrs.update({
                    'warmup_included': 'true',
                    'warmup_start': str(self.warmup_date.date()),
                    'warmup_end': str(warmup_end.date()),
                    'warmup_days': n_warmup_days,
                    'simulation_start': str(self.start_date.date()),
                    'simulation_end': str(self.end_date.date()),
                    'simulation_days': n_sim_days,
                    'warmup_method': 'repeat_first_year',
                    'warmup_repetitions': n_repetitions,
                    'total_days': len(combined.time),
                    'elevation_included': 'true' if has_elevation else 'false'
                })

                # ✅ NEW: Save to temporary file first, then replace original
                import shutil
                
                temp_file = file_path.parent / f".tmp_{file_path.name}"
                
                self.logger.debug(f"Saving to temporary file: {temp_file}...")
                combined.to_netcdf(temp_file)
                
                # Close combined dataset
                combined.close()
                
                # Replace original file with temporary file
                self.logger.debug(f"Replacing original file...")
                shutil.move(str(temp_file), str(file_path))
                
                self.logger.info(f"✅ Updated {file_path.name} with warm-up period")
                
                updated_files.append(file_path)
                
            except Exception as e:
                self.logger.error(f"Error adding warm-up to {file_path.name}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                updated_files.append(file_path)
        
        self.logger.info(f"✅ Warm-up period added to {len(updated_files)} files")
        
        # ✅ NEW: Copy updated files to model-specific directory
        self._copy_to_model_directory(updated_files)
        
        return updated_files

    #---------------------------------------------------------------------------------

    def _filter_time_range(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Filter dataset to the exact time range specified in namelist
        
        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset
            
        Returns
        -------
        xr.Dataset
            Filtered dataset
        """
        if 'time' not in dataset.dims:
            self.logger.warning("No 'time' coordinate found in dataset")
            return dataset
        
        try:
            # Use warm-up date if available, otherwise start_date
            if self.warmup_date is not None:
                filter_start = self.warmup_date
            else:
                filter_start = self.start_date
            
            start_date_str = filter_start.strftime('%Y-%m-%d')
            end_date_str = self.end_date.strftime('%Y-%m-%d')
            
            self.logger.debug(f"Filtering to period: {start_date_str} to {end_date_str}")
            
            # Filter to date range
            filtered_ds = dataset.sel(time=slice(start_date_str, end_date_str))
            
            self.logger.info(f"Filtered time range: {filtered_ds.time.min().values} to {filtered_ds.time.max().values}")
            self.logger.info(f"Total days: {len(filtered_ds.time)}")
            
            return filtered_ds
            
        except Exception as e:
            self.logger.error(f"Error filtering time range: {e}")
            self.logger.warning("Returning unfiltered dataset")
            return dataset

    #---------------------------------------------------------------------------------

    def _convert_units(self, dataset: xr.Dataset, variable: str) -> xr.Dataset:
        """
        Convert HAR units to standard units
        
        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset
        variable : str
            Variable name
            
        Returns
        -------
        xr.Dataset
            Dataset with converted units
        """
        ds = dataset.copy()
        var_info = self.har_variables.get(variable)
        
        if var_info is None:
            return ds
        
        # Find the data variable
        data_var = None
        for v in ds.data_vars:
            if v.lower() == variable.lower() or variable in v.lower():
                data_var = v
                break
        
        if data_var is None:
            self.logger.warning(f"Could not find variable {variable} in dataset")
            return ds
        
        source_units = var_info.get('source_units', '')
        target_units = var_info.get('units', '')
        
        # Temperature: Kelvin to Celsius
        if 't2' in variable and source_units == 'K':
            self.logger.debug(f"Converting {variable} from Kelvin to Celsius")
            ds[data_var] = ds[data_var] - 273.15
            ds[data_var].attrs['units'] = 'degC'
        
        # Precipitation/PET: mm/h to mm/day
        # HAR daily files already contain daily totals, so we just need to ensure units are correct
        elif source_units == 'mm h-1':
            # Check if values look like hourly rates or daily totals
            sample_mean = float(ds[data_var].mean())
            
            if 'prcp' in variable:
                # Daily precipitation should be ~0-50 mm/day typically
                # If values are small (< 1), they might be hourly rates
                if sample_mean < 1:
                    self.logger.debug(f"Converting {variable} from mm/h to mm/day (×24)")
                    ds[data_var] = ds[data_var] * 24.0
                else:
                    self.logger.debug(f"{variable} appears to already be in mm/day")
                ds[data_var].attrs['units'] = 'mm/day'
                
            elif 'potevap' in variable:
                # Daily PET should be ~0-10 mm/day typically
                if sample_mean < 0.5:
                    self.logger.debug(f"Converting {variable} from mm/h to mm/day (×24)")
                    ds[data_var] = ds[data_var] * 24.0
                    
                    # HAR PET can be negative (upward flux convention) - convert to positive
                    if float(ds[data_var].mean()) < 0:
                        self.logger.debug("Converting negative PET to positive values")
                        ds[data_var] = -ds[data_var]
                else:
                    self.logger.debug(f"{variable} appears to already be in mm/day")
                    
                ds[data_var].attrs['units'] = 'mm/day'
        
        return ds

    #---------------------------------------------------------------------------------

    def _clip_to_catchment(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Clip HAR dataset to catchment extent using lat/lon coordinates
        
        HAR data has curvilinear grid (2D lat/lon arrays), so we clip by masking
        
        Parameters
        ----------
        dataset : xr.Dataset
            HAR dataset to clip
            
        Returns
        -------
        xr.Dataset
            Clipped dataset
        """
        if self.catchment_extent is None:
            self.logger.debug("No catchment extent available - skipping clipping")
            return dataset
        
        try:
            self.logger.info("✂️ Clipping HAR dataset to catchment extent...")
            
            # Get catchment bounds with buffer
            bounds = self.catchment_extent.total_bounds  # [minx, miny, maxx, maxy]
            buffer_deg = 0.2  # ~20km buffer for HAR 10km grid
            
            lon_min = bounds[0] - buffer_deg
            lon_max = bounds[2] + buffer_deg
            lat_min = bounds[1] - buffer_deg
            lat_max = bounds[3] + buffer_deg
            
            self.logger.debug(f"Clip bounds (with buffer): lon [{lon_min:.4f}, {lon_max:.4f}], lat [{lat_min:.4f}, {lat_max:.4f}]")
            
            # HAR has 2D lat/lon coordinates
            if 'lat' in dataset.coords and 'lon' in dataset.coords:
                lat_2d = dataset.coords['lat']
                lon_2d = dataset.coords['lon']
                
                # Create mask based on lat/lon bounds
                mask = (
                    (lat_2d >= lat_min) & (lat_2d <= lat_max) &
                    (lon_2d >= lon_min) & (lon_2d <= lon_max)
                )
                
                # Find indices where mask is True
                valid_south_north = mask.any(dim='west_east')
                valid_west_east = mask.any(dim='south_north')
                
                # Get index ranges
                sn_indices = np.where(valid_south_north.values)[0]
                we_indices = np.where(valid_west_east.values)[0]
                
                if len(sn_indices) == 0 or len(we_indices) == 0:
                    self.logger.error("❌ No valid indices after clipping - returning unclipped")
                    return dataset
                
                sn_min, sn_max = sn_indices.min(), sn_indices.max() + 1
                we_min, we_max = we_indices.min(), we_indices.max() + 1
                
                self.logger.debug(f"Clipping indices: south_north [{sn_min}:{sn_max}], west_east [{we_min}:{we_max}]")
                
                # Clip the dataset
                clipped = dataset.isel(
                    south_north=slice(sn_min, sn_max),
                    west_east=slice(we_min, we_max)
                )
                
                # Calculate reduction
                original_size = dataset.south_north.size * dataset.west_east.size
                clipped_size = clipped.south_north.size * clipped.west_east.size
                reduction = (1 - clipped_size / original_size) * 100
                
                self.logger.info(f"✅ Dataset clipped: {original_size} → {clipped_size} grid cells ({reduction:.1f}% reduction)")
                
                return clipped
            else:
                self.logger.warning("Could not find lat/lon coordinates for clipping")
                return dataset
            
        except Exception as e:
            self.logger.error(f"Error clipping dataset: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return dataset

    #---------------------------------------------------------------------------------


    def _save_processed_file(self, dataset: xr.Dataset, variable: str, 
                            output_name: str) -> Optional[Path]:
        """
        Save processed HAR dataset to NetCDF file (WITHOUT elevation - added later)
        """
        try:
            output_file = self.output_path / output_name
            
            # Set encoding
            encoding = {}
            for var in dataset.data_vars:
                encoding[var] = {'zlib': True, 'complevel': 4}
            for coord in ['lat', 'lon']:
                if coord in dataset.coords:
                    encoding[coord] = {'dtype': 'float64'}
            
            # Add metadata
            dataset.attrs.update({
                'title': f'HAR v2 {variable} daily data',
                'source': 'High Asia Refined Analysis v2 (HAR v2)',
                'gauge_id': str(self.gauge_id),
                'processed_by': 'HARAnalyzer',
                'creation_date': pd.Timestamp.now().isoformat(),
                'projection': 'Lambert Conformal Conic',
                'resolution': '10km',
                'elevation_included': 'false'  # Will be updated when elevation is added
            })
            
            dataset.to_netcdf(output_file, encoding=encoding)
            
            self.logger.info(f"💾 Saved: {output_name}")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error saving {output_name}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
        
    #---------------------------------------------------------------------------------

    def _add_elevation_to_all_files_simple(self, file_list: List[Path]) -> None:
        """
        Add elevation to all HAR files by clipping elevation to match EACH file's exact grid.
        This works because we clip elevation individually for each file, ensuring perfect alignment.
        """
        self.logger.info("🏔️ Adding elevation to all processed files...")
        
        # Load full elevation data
        hgt_file = self.har_data_dir / 'HARv2_d10km_static_hgt.nc'
        
        if not hgt_file.exists():
            self.logger.warning(f"⚠️ HAR terrain height file not found: {hgt_file}")
            self.logger.warning("   Files will be saved without elevation data")
            return
        
        try:
            ds_hgt = xr.open_dataset(hgt_file)
            
            # Find elevation variable
            hgt_var = None
            for var in ['hgt', 'HGT', 'terrain', 'elevation', 'z']:
                if var in ds_hgt.data_vars:
                    hgt_var = var
                    break
            
            if hgt_var is None:
                self.logger.warning(f"⚠️ Could not find elevation variable in {hgt_file}")
                ds_hgt.close()
                return
            
            # Get full elevation data (keep it open for now)
            full_elevation = ds_hgt[hgt_var]
            self.logger.info(f"   Full elevation shape: {full_elevation.shape}")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading elevation file: {e}")
            return
        
        # Process each file individually
        for file_path in file_list:
            if 'elevation' in file_path.name.lower():
                self.logger.debug(f"   Skipping elevation file: {file_path.name}")
                continue
            
            try:
                self.logger.info(f"   Processing: {file_path.name}")
                
                # Open the meteorological file
                ds_meteo = xr.open_dataset(file_path)
                
                # Check if elevation already exists
                if 'elevation' in ds_meteo.data_vars:
                    self.logger.debug(f"   ✅ Elevation already exists in {file_path.name}")
                    ds_meteo.close()
                    continue
                
                # Get the grid dimensions from this file
                if 'south_north' in ds_meteo.dims and 'west_east' in ds_meteo.dims:
                    sn_size = ds_meteo.dims['south_north']
                    we_size = ds_meteo.dims['west_east']
                else:
                    # Try alternative dimension names
                    sn_dim = None
                    we_dim = None
                    for dim in ds_meteo.dims:
                        if 'south' in dim.lower() or 'y' in dim.lower() or 'lat' in dim.lower():
                            sn_dim = dim
                        elif 'west' in dim.lower() or 'x' in dim.lower() or 'lon' in dim.lower():
                            we_dim = dim
                    
                    if sn_dim and we_dim:
                        sn_size = ds_meteo.dims[sn_dim]
                        we_size = ds_meteo.dims[we_dim]
                    else:
                        self.logger.warning(f"   ⚠️ Could not determine grid dimensions for {file_path.name}")
                        ds_meteo.close()
                        continue
                
                self.logger.debug(f"   Meteo grid: {sn_size} x {we_size}")
                
                # Get lat/lon from the meteorological file
                if 'lat' in ds_meteo.coords and 'lon' in ds_meteo.coords:
                    meteo_lat = ds_meteo.coords['lat'].values
                    meteo_lon = ds_meteo.coords['lon'].values
                else:
                    self.logger.warning(f"   ⚠️ No lat/lon coordinates in {file_path.name}")
                    ds_meteo.close()
                    continue
                
                self.logger.debug(f"   Meteo lat shape: {meteo_lat.shape}, lon shape: {meteo_lon.shape}")
                
                # ✅ KEY FIX: Clip elevation to match THIS file's exact grid
                # Find the bounding box of the meteorological data
                lat_min, lat_max = meteo_lat.min(), meteo_lat.max()
                lon_min, lon_max = meteo_lon.min(), meteo_lon.max()
                
                self.logger.debug(f"   Meteo extent: lat [{lat_min:.4f}, {lat_max:.4f}], lon [{lon_min:.4f}, {lon_max:.4f}]")
                
                # Get full elevation lat/lon
                if 'lat' in ds_hgt.coords and 'lon' in ds_hgt.coords:
                    full_lat = ds_hgt.coords['lat'].values
                    full_lon = ds_hgt.coords['lon'].values
                else:
                    self.logger.warning(f"   ⚠️ No lat/lon in elevation file")
                    ds_meteo.close()
                    continue
                
                # Find indices that match the meteorological grid extent
                # Add small buffer to ensure we capture all cells
                buffer = 0.01  # degrees
                
                if len(full_lat.shape) == 2:
                    # 2D lat/lon (curvilinear grid)
                    # Find cells that fall within the bounding box
                    lat_mask = (full_lat >= lat_min - buffer) & (full_lat <= lat_max + buffer)
                    lon_mask = (full_lon >= lon_min - buffer) & (full_lon <= lon_max + buffer)
                    combined_mask = lat_mask & lon_mask
                    
                    # Find the bounding indices
                    rows_with_data = np.any(combined_mask, axis=1)
                    cols_with_data = np.any(combined_mask, axis=0)
                    
                    if not np.any(rows_with_data) or not np.any(cols_with_data):
                        self.logger.warning(f"   ⚠️ No overlap between elevation and meteo grids")
                        ds_meteo.close()
                        continue
                    
                    row_start = np.argmax(rows_with_data)
                    row_end = len(rows_with_data) - np.argmax(rows_with_data[::-1])
                    col_start = np.argmax(cols_with_data)
                    col_end = len(cols_with_data) - np.argmax(cols_with_data[::-1])
                    
                    # Clip elevation using these indices
                    clipped_elevation = full_elevation.isel(
                        south_north=slice(row_start, row_end),
                        west_east=slice(col_start, col_end)
                    )
                    
                else:
                    # 1D lat/lon (regular grid)
                    sn_mask = (full_lat >= lat_min - buffer) & (full_lat <= lat_max + buffer)
                    we_mask = (full_lon >= lon_min - buffer) & (full_lon <= lon_max + buffer)
                    
                    clipped_elevation = full_elevation.isel(
                        south_north=sn_mask,
                        west_east=we_mask
                    )
                
                self.logger.debug(f"   Clipped elevation shape: {clipped_elevation.shape}")
                
                # ✅ CRITICAL: Ensure clipped elevation matches meteo grid EXACTLY
                clipped_shape = clipped_elevation.shape
                expected_shape = (sn_size, we_size)
                
                if clipped_shape != expected_shape:
                    self.logger.warning(f"   ⚠️ Shape mismatch: elevation {clipped_shape} vs meteo {expected_shape}")
                    self.logger.info(f"   🔧 Resampling elevation to match meteo grid...")
                    
                    # Create new elevation array with correct shape
                    # Use the meteo file's lat/lon coordinates
                    new_elevation = xr.DataArray(
                        data=np.full(expected_shape, np.nan),
                        dims=['south_north', 'west_east'],
                        coords={
                            'lat': (['south_north', 'west_east'], meteo_lat),
                            'lon': (['south_north', 'west_east'], meteo_lon)
                        },
                        attrs={
                            'units': 'm',
                            'long_name': 'Terrain Height',
                            'source': 'HAR v2 static terrain height (resampled)'
                        }
                    )
                    
                    # Interpolate from clipped elevation to new grid
                    # Simple approach: use nearest neighbor based on lat/lon
                    from scipy.interpolate import griddata
                    
                    # Get source points and values
                    src_lat = ds_hgt.coords['lat'].values
                    src_lon = ds_hgt.coords['lon'].values
                    src_elev = full_elevation.values
                    
                    # Flatten source data
                    src_points = np.column_stack([src_lat.flatten(), src_lon.flatten()])
                    src_values = src_elev.flatten()
                    
                    # Remove NaN values
                    valid_mask = ~np.isnan(src_values)
                    src_points = src_points[valid_mask]
                    src_values = src_values[valid_mask]
                    
                    # Target points
                    tgt_points = np.column_stack([meteo_lat.flatten(), meteo_lon.flatten()])
                    
                    # Interpolate
                    interpolated_values = griddata(
                        src_points, src_values, tgt_points, 
                        method='nearest'
                    )
                    
                    # Reshape and assign
                    new_elevation.values = interpolated_values.reshape(expected_shape)
                    clipped_elevation = new_elevation
                    
                    self.logger.info(f"   ✅ Resampled elevation to shape: {clipped_elevation.shape}")
                
                # Add elevation to the meteorological dataset
                ds_meteo['elevation'] = clipped_elevation
                ds_meteo['elevation'].attrs.update({
                    'units': 'm',
                    'long_name': 'Terrain Height',
                    'source': 'HAR v2 static terrain height'
                })
                
                # Save to temporary file, then replace
                temp_file = file_path.with_suffix('.tmp.nc')
                ds_meteo.to_netcdf(temp_file)
                ds_meteo.close()
                
                # Replace original with updated file
                import shutil
                shutil.move(str(temp_file), str(file_path))
                
                self.logger.info(f"   ✅ Added elevation to {file_path.name}")
                
            except Exception as e:
                self.logger.error(f"   ❌ Error adding elevation to {file_path.name}: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
        
        # Close elevation dataset
        ds_hgt.close()
        
        self.logger.info("🏔️ Elevation processing complete!")

    #---------------------------------------------------------------------------------

    def _load_elevation_data(self) -> Optional[xr.DataArray]:
        """
        Load elevation from HAR static file (UNCLIPPED - will be clipped later)
        
        Returns
        -------
        Optional[xr.DataArray]
            Full elevation data array, or None if failed
        """
        hgt_file = self.har_data_dir / 'HARv2_d10km_static_hgt.nc'
        
        if not hgt_file.exists():
            self.logger.warning(f"❌ HAR terrain height file not found: {hgt_file}")
            return None
        
        try:
            self.logger.info(f"🏔️ Loading elevation from: {hgt_file.name}")
            
            ds_hgt = xr.open_dataset(hgt_file)
            
            # Find hgt variable
            hgt_var = None
            for var in ds_hgt.data_vars:
                if 'hgt' in var.lower():
                    hgt_var = var
                    break
            
            if hgt_var is None:
                self.logger.error("Could not find hgt variable")
                ds_hgt.close()
                return None
            
            elevation = ds_hgt[hgt_var]
            
            # Remove time dimension if present
            if 'time' in elevation.dims:
                elevation = elevation.isel(time=0)
                self.logger.debug(f"Removed time dim, shape: {elevation.shape}")
            
            # Keep a copy in memory (it's small - just 252x381 floats)
            elevation = elevation.load()
            
            self.logger.info(f"✅ Loaded elevation: shape={elevation.shape}, "
                           f"range=[{float(elevation.min()):.0f}, {float(elevation.max()):.0f}] m")
            
            ds_hgt.close()
            
            return elevation
            
        except Exception as e:
            self.logger.error(f"Error loading elevation: {e}")
            return None


    #---------------------------------------------------------------------------------

    def _find_and_process_yearly_files(self) -> List[Path]:
        """
        Main method to find yearly HAR files, combine them, and process
        SIMPLIFIED: Don't add elevation during processing - add it after ALL files are saved
        ✅ UPDATED: Adds warm-up period if configured
        """
        self.logger.info("Starting HAR file processing pipeline...")
        
        # Check which files already exist
        expected_files = {
            't2_mean': 'har_temp_mean.nc',
            't2_min': 'har_temp_min.nc',
            't2_max': 'har_temp_max.nc',
            'prcp': 'har_precip.nc',
            'potevap': 'har_pet.nc'
        }
        
        existing_files = {}
        missing_vars = []
        
        for var, filename in expected_files.items():
            file_path = self.output_path / filename
            if file_path.exists() and not self.force_reprocess:
                existing_files[var] = file_path
                self.logger.info(f"✅ Using existing file: {filename}")
            else:
                missing_vars.append(var)
        
        all_files = list(existing_files.values())
        
        # Initialize clip indices storage
        self._clip_indices = None
        
        # Process missing variables (WITHOUT elevation)
        for variable in missing_vars:
            self.logger.info(f"Processing {variable}...")
            
            yearly_files = self._find_yearly_files(variable)
            
            if not yearly_files:
                self.logger.warning(f"No files found for {variable}")
                continue
            
            # Combine yearly files (clips during loading, stores clip indices)
            combined = self._combine_yearly_files(yearly_files, variable)
            
            if combined is None:
                continue
            
            # Filter time range
            filtered = self._filter_time_range(combined)
            
            # Convert units
            converted = self._convert_units(filtered, variable)
            
            # Save WITHOUT elevation
            output_name = expected_files[variable]
            output_file = self._save_processed_file(converted, variable, output_name)
            
            if output_file:
                all_files.append(output_file)
            
            # Close datasets
            combined.close()
            filtered.close()
            converted.close()
        
        self.logger.info(f"Processing pipeline complete! Total files: {len(all_files)}")
        
        # ✅ NOW: Add elevation to ALL files using a simple, robust method
        self._add_elevation_to_all_files_simple(all_files)
        
        # ✅ NEW: Add warm-up period to each file if configured
        if hasattr(self, 'warmup_date') and self.warmup_date is not None:
            self.logger.info("\n🔄 Adding warm-up period to HAR meteorological files...")
            all_files = self._add_warmup_to_files(all_files)
        
        # Copy to model-specific directory
        self._copy_to_model_directory(all_files)
        
        return all_files

    #---------------------------------------------------------------------------------
    
    def check_missing_values_in_netcdf(self, netcdf_file: Path) -> Dict[str, Any]:
        """
        Check for missing values (NaN, inf, fill values) in NetCDF file
        
        Parameters
        ----------
        netcdf_file : Path
            Path to NetCDF file to check
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with missing value statistics
        """
        self.logger.info(f"🔍 Checking for missing values in {netcdf_file.name}")
        
        try:
            ds = xr.open_dataset(netcdf_file)
            
            results = {
                'file': netcdf_file.name,
                'variables': {},
                'has_missing': False,
                'total_missing_pct': 0.0
            }
            
            # Get data variables (exclude elevation and coordinates)
            data_vars = [v for v in ds.data_vars if 'time' in ds[v].dims]
            
            if not data_vars:
                self.logger.warning(f"No time-varying variables found in {netcdf_file.name}")
                ds.close()
                return results
            
            for var_name in data_vars:
                var_data = ds[var_name]
                
                # Get total number of values
                total_values = var_data.size
                
                # Count NaN values
                nan_count = np.isnan(var_data.values).sum()
                
                # Count infinite values
                inf_count = np.isinf(var_data.values).sum()
                
                # Check for fill values
                fill_value = var_data.attrs.get('_FillValue', None)
                if fill_value is not None:
                    fill_count = (var_data.values == fill_value).sum()
                else:
                    fill_count = 0
                
                # Total missing
                total_missing = nan_count + inf_count + fill_count
                missing_pct = (total_missing / total_values) * 100
                
                var_results = {
                    'total_values': int(total_values),
                    'nan_count': int(nan_count),
                    'inf_count': int(inf_count),
                    'fill_count': int(fill_count),
                    'total_missing': int(total_missing),
                    'missing_pct': float(missing_pct)
                }
                
                results['variables'][var_name] = var_results
                
                # Log results
                if total_missing > 0:
                    results['has_missing'] = True
                    self.logger.warning(f"⚠️ {var_name}: {total_missing:,} missing values ({missing_pct:.2f}%)")
                    if nan_count > 0:
                        self.logger.warning(f"   - NaN values: {nan_count:,}")
                    if inf_count > 0:
                        self.logger.warning(f"   - Inf values: {inf_count:,}")
                    if fill_count > 0:
                        self.logger.warning(f"   - Fill values: {fill_count:,}")
                else:
                    self.logger.info(f"✅ {var_name}: No missing values")
            
            # Calculate overall missing percentage
            total_all_values = sum(r['total_values'] for r in results['variables'].values())
            total_all_missing = sum(r['total_missing'] for r in results['variables'].values())
            
            if total_all_values > 0:
                results['total_missing_pct'] = (total_all_missing / total_all_values) * 100
            
            if results['has_missing']:
                self.logger.warning(f"📊 Overall: {total_all_missing:,} / {total_all_values:,} values missing ({results['total_missing_pct']:.2f}%)")
            else:
                self.logger.info(f"✅ File check complete: No missing values found")
            
            ds.close()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error checking missing values: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'file': netcdf_file.name, 'error': str(e)}

    #---------------------------------------------------------------------------------

    def create_missing_values_report(self) -> pd.DataFrame:
        """
        Create a comprehensive missing values report for all processed files
        
        Returns
        -------
        pd.DataFrame
            Summary report of missing values across all files
        """
        self.logger.info("📊 Creating missing values report...")
        
        if not self.processed_files:
            self.logger.warning("No processed files found")
            return pd.DataFrame()
        
        report_data = []
        
        for processed_file in self.processed_files:
            results = self.check_missing_values_in_netcdf(processed_file)
            
            if 'error' in results:
                continue
            
            for var_name, var_stats in results.get('variables', {}).items():
                report_data.append({
                    'File': results['file'],
                    'Variable': var_name,
                    'Total Values': var_stats['total_values'],
                    'NaN Count': var_stats['nan_count'],
                    'Inf Count': var_stats['inf_count'],
                    'Fill Count': var_stats['fill_count'],
                    'Total Missing': var_stats['total_missing'],
                    'Missing %': var_stats['missing_pct']
                })
        
        if not report_data:
            self.logger.info("No data for missing values report")
            return pd.DataFrame()
        
        report_df = pd.DataFrame(report_data)
        
        # Save to CSV
        report_file = self.output_path / 'missing_values_report.csv'
        report_df.to_csv(report_file, index=False)
        
        self.logger.info(f"Missing values report saved to: {report_file}")
        
        # Print summary
        self.logger.info("\n📋 Missing Values Summary:")
        self.logger.info(f"{'Variable':<20} {'File':<25} {'Missing %':>10}")
        self.logger.info("-" * 60)
        
        for _, row in report_df.iterrows():
            if row['Missing %'] > 0:
                self.logger.warning(f"{row['Variable']:<20} {row['File']:<25} {row['Missing %']:>9.2f}%")
        
        return report_df

    #---------------------------------------------------------------------------------

    def _copy_to_model_directory(self, files: List[Path]) -> None:
        """
        Copy processed files from shared data_obs to model-specific data_obs
        
        Parameters
        ----------
        files : List[Path]
            List of file paths to copy
        """
        import shutil
        
        if not files:
            self.logger.debug("No files to copy to model directory")
            return
        
        self.logger.info(f"📋 Copying {len(files)} files to model-specific directory...")
        
        copied_count = 0
        for file_path in files:
            if file_path.exists():
                dest = self.model_data_dir / file_path.name
                try:
                    shutil.copy2(file_path, dest)
                    self.logger.debug(f"  ✅ Copied: {file_path.name}")
                    copied_count += 1
                except Exception as e:
                    self.logger.warning(f"  ❌ Failed to copy {file_path.name}: {e}")
        
        self.logger.info(f"✅ Successfully copied {copied_count} files to {self.model_data_dir.name}/")

    #---------------------------------------------------------------------------------

    def plot_spatial_overview(self, netcdf_file: Path) -> None:
        """
        Plot spatial overview - FIXED to use 2D lat/lon coordinates properly
        """
        self.logger.info(f"Creating spatial overview plots for {netcdf_file.name}")
        
        try:
            ds = xr.open_dataset(netcdf_file)
            
            # Get variables to plot
            meteo_vars = [v for v in ds.data_vars if v != 'elevation' and len(ds[v].dims) >= 2]
            has_elevation = 'elevation' in ds.data_vars
            
            n_plots = len(meteo_vars) + (1 if has_elevation else 0)
            
            if n_plots == 0:
                self.logger.warning("No variables to plot")
                ds.close()
                return
            
            # Create figure
            cols = min(2, n_plots)
            rows = (n_plots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 6*rows))
            if n_plots == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            # ✅ Get 2D lat/lon for plotting
            lat_2d = ds.coords['lat'].values if 'lat' in ds.coords else None
            lon_2d = ds.coords['lon'].values if 'lon' in ds.coords else None
            
            plot_idx = 0
            
            # Plot meteorological variables
            for var_name in meteo_vars:
                ax = axes[plot_idx]
                
                # Get data (first timestep if time exists)
                data = ds[var_name]
                if 'time' in data.dims:
                    data = data.isel(time=0)
                
                # Get variable info
                var_info = self.har_variables.get(var_name, {'name': var_name, 'units': '', 'cmap': 'viridis'})
                
                # Plot
                if lat_2d is not None and lon_2d is not None:
                    im = ax.pcolormesh(lon_2d, lat_2d, data.values, cmap=var_info.get('cmap', 'viridis'), shading='auto')
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    
                    # Add catchment boundary
                    if self.catchment_extent is not None:
                        try:
                            self.catchment_extent.boundary.plot(ax=ax, color='red', linewidth=2)
                        except:
                            pass
                else:
                    im = ax.imshow(data.values, cmap=var_info.get('cmap', 'viridis'), origin='lower')
                
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label(var_info.get('units', ''))
                ax.set_title(var_info.get('name', var_name))
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
            
            # Plot elevation
            if has_elevation:
                ax = axes[plot_idx]
                elev = ds['elevation']
                
                if lat_2d is not None and lon_2d is not None:
                    im = ax.pcolormesh(lon_2d, lat_2d, elev.values, cmap='terrain', shading='auto')
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    
                    if self.catchment_extent is not None:
                        try:
                            self.catchment_extent.boundary.plot(ax=ax, color='red', linewidth=2)
                        except:
                            pass
                else:
                    im = ax.imshow(elev.values, cmap='terrain', origin='lower')
                
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('m')
                ax.set_title('Elevation')
                ax.grid(True, alpha=0.3)
            
            # Hide unused axes
            for i in range(n_plots, len(axes)):
                axes[i].set_visible(False)
            
            fig.suptitle(f"HAR Data - Gauge {self.gauge_id}\n{netcdf_file.stem}", fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            # Save
            save_path = self.spatial_plots_dir / f"har_spatial_{self.gauge_id}_{netcdf_file.stem}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved: {save_path.name}")
            
            plt.show()
            plt.close()
            ds.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    #---------------------------------------------------------------------------------

    def calculate_spatial_average_timeseries(self, netcdf_file: Path) -> pd.DataFrame:
        """
        Calculate spatial average time series with missing value checks
        ✅ FIXED: Don't call check_missing_values here - it's called separately now
        
        Parameters
        ----------
        netcdf_file : Path
            Path to the NetCDF file
            
        Returns
        -------
        pd.DataFrame
            Time series of spatially averaged data
        """
        self.logger.debug(f"Calculating spatial averages for {netcdf_file.name}")
        
        try:
            # Open the dataset
            ds = xr.open_dataset(netcdf_file, chunks={'time': 100})
            
            # Get data variables (exclude elevation)
            data_vars = [var for var in ds.data_vars if var != 'elevation' and len(ds[var].dims) >= 2]
            
            if not data_vars:
                self.logger.warning(f"No suitable variables found in {netcdf_file.name}")
                ds.close()
                return pd.DataFrame()
            
            # Calculate spatial means for each variable
            results = {}
            
            for var_name in data_vars:
                self.logger.debug(f"  Processing variable: {var_name}")
                
                # Calculate spatial mean (average over spatial dims, ignoring NaN)
                spatial_dims = [dim for dim in ds[var_name].dims if dim != 'time']
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    spatial_mean = ds[var_name].mean(dim=spatial_dims, skipna=True)
                
                # Convert to pandas series
                ts = spatial_mean.to_pandas()
                
                # Check for missing timesteps
                missing_ts = ts.isna().sum()
                if missing_ts > 0:
                    missing_ts_pct = (missing_ts / len(ts)) * 100
                    self.logger.debug(f"  {var_name}: {missing_ts} timesteps ({missing_ts_pct:.1f}%) have no valid data")
                
                results[var_name] = ts
            
            # Combine into DataFrame
            df = pd.DataFrame(results)
            df.index.name = 'time'
            
            self.logger.debug(f"  Timeseries: {len(df)} days from {df.index.min()} to {df.index.max()}")
            
            # Close dataset
            ds.close()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating spatial averages for {netcdf_file.name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    #---------------------------------------------------------------------------------

    def plot_timeseries(self, netcdf_file: Path, df_timeseries: pd.DataFrame) -> None:
        """
        Plot time series of spatially averaged HAR variables
        ✅ MODIFIED: Highlights warm-up period with different color
        
        Parameters
        ----------
        netcdf_file : Path
            Path to the NetCDF file
        df_timeseries : pd.DataFrame
            Time series data to plot
        """
        if df_timeseries.empty:
            self.logger.warning(f"No data to plot for {netcdf_file.name}")
            return
        
        self.logger.info(f"Creating time series plots for {netcdf_file.name}")
        
        try:
            n_vars = len(df_timeseries.columns)
            
            fig, axes = plt.subplots(n_vars, 1, figsize=(14, 3*n_vars))
            if n_vars == 1:
                axes = [axes]
            
            # ✅ Updated title to show warm-up info
            title_warmup = ""
            if hasattr(self, 'warmup_date') and self.warmup_date is not None:
                title_warmup = f"\n🔄 Warm-up: {self.warmup_date.strftime('%Y-%m-%d')} to {(self.start_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}"
            
            fig.suptitle(f'HAR Time Series - Gauge {self.gauge_id}\n{netcdf_file.stem} - Spatial Averages\n'
                        f'Period: {df_timeseries.index.min().strftime("%Y-%m-%d")} to {df_timeseries.index.max().strftime("%Y-%m-%d")}'
                        f'{title_warmup}', 
                        fontsize=14, fontweight='bold')
            
            for i, var_name in enumerate(df_timeseries.columns):
                ax = axes[i]
                
                # Get variable info
                var_info = None
                for key, info in self.har_variables.items():
                    if key in var_name.lower() or var_name.lower() in key:
                        var_info = info
                        break
                
                if var_info is None:
                    var_info = {'name': var_name, 'units': '', 'cmap': 'viridis'}
                
                # ✅ NEW: Split data into warm-up and simulation periods
                if hasattr(self, 'warmup_date') and self.warmup_date is not None:
                    # Separate warm-up and simulation data
                    warmup_mask = df_timeseries.index < self.start_date
                    simulation_mask = df_timeseries.index >= self.start_date
                    
                    warmup_data = df_timeseries.loc[warmup_mask, var_name]
                    simulation_data = df_timeseries.loc[simulation_mask, var_name]
                    
                    # Plot warm-up period in orange/gray
                    if len(warmup_data) > 0:
                        ax.plot(warmup_data.index, warmup_data.values, 
                            linewidth=1.5, color='orange', alpha=0.7, 
                            label='Warm-up period')
                    
                    # Plot simulation period in blue
                    if len(simulation_data) > 0:
                        ax.plot(simulation_data.index, simulation_data.values, 
                            linewidth=1, color='blue', 
                            label='Simulation period')
                    
                    # Add vertical line at start of simulation
                    ax.axvline(self.start_date, color='red', linestyle='--', 
                            linewidth=2, alpha=0.7, label='Simulation start')
                    
                    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
                    
                else:
                    # No warm-up - plot normally
                    df_timeseries[var_name].plot(ax=ax, linewidth=1, color='blue')
                
                # Customize
                ax.set_title(f"{var_info['name']}")
                ax.set_ylabel(f"{var_info['units']}")
                ax.grid(True, alpha=0.3)
                
                # Format x-axis
                date_range = (df_timeseries.index.max() - df_timeseries.index.min()).days
                
                if date_range > 365*2:
                    ax.xaxis.set_major_locator(mdates.YearLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                elif date_range > 365:
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                else:
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                
                # ✅ UPDATED: Statistics text box now shows warm-up vs simulation stats
                if hasattr(self, 'warmup_date') and self.warmup_date is not None and len(warmup_data) > 0:
                    warmup_mean = warmup_data.mean()
                    sim_mean = simulation_data.mean()
                    
                    stats_text = (f'Warm-up Mean: {warmup_mean:.2f}\n'
                                f'Simulation Mean: {sim_mean:.2f}\n'
                                f'Overall Min: {df_timeseries[var_name].min():.2f}\n'
                                f'Overall Max: {df_timeseries[var_name].max():.2f}')
                else:
                    mean_val = df_timeseries[var_name].mean()
                    std_val = df_timeseries[var_name].std()
                    min_val = df_timeseries[var_name].min()
                    max_val = df_timeseries[var_name].max()
                    
                    stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}'
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9)
            
            axes[-1].set_xlabel('Date')
            
            plt.tight_layout()
            
            # Save
            save_path = self.timeseries_plots_dir / f"har_timeseries_gauge_{self.gauge_id}_{netcdf_file.stem}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Time series plot saved to {save_path}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating time series plot: {e}")

    #---------------------------------------------------------------------------------

    def calculate_monthly_temperature_averages(self) -> pd.DataFrame:
        """
        Calculate monthly mean temperature averages (climatology) and save to CSV
        
        Returns
        -------
        pd.DataFrame
            Monthly temperature averages
        """
        self.logger.info("Calculating monthly temperature averages from HAR data...")
        
        try:
            # Find temperature mean file
            temp_file = None
            for file_path in self.processed_files:
                if 'temp_mean' in file_path.name:
                    temp_file = file_path
                    break
            
            if temp_file is None:
                self.logger.error("Temperature mean file not found")
                return pd.DataFrame()
            
            ds = xr.open_dataset(temp_file)
            
            # Get temperature variable
            temp_var = [v for v in ds.data_vars if 'time' in ds[v].dims][0]
            
            # ✅ NEW: Mask data to only include grid cells within catchment
            if self.catchment_extent is not None:
                self.logger.info("Masking temperature data to catchment extent...")
                
                # Get lat/lon coordinates (HAR has 2D lat/lon)
                if 'lat' in ds.coords:
                    lats = ds['lat'].values
                    lons = ds['lon'].values
                else:
                    lats = ds['latitude'].values
                    lons = ds['longitude'].values
                
                # Get catchment bounds
                catchment_bounds = self.catchment_extent.total_bounds  # minx, miny, maxx, maxy
                
                # Create boolean mask for grid cells within catchment bounds
                # HAR has 2D lat/lon arrays
                lat_mask = (lats >= catchment_bounds[1]) & (lats <= catchment_bounds[3])
                lon_mask = (lons >= catchment_bounds[0]) & (lons <= catchment_bounds[2])
                combined_mask = lat_mask & lon_mask
                
                # Apply mask to temperature data
                ds_masked = ds[temp_var].where(combined_mask, drop=False)
                
                # Calculate spatial average only over masked (catchment) area
                spatial_dims = [dim for dim in ds[temp_var].dims if dim != 'time']
                temp_spatial_avg = ds_masked.mean(dim=spatial_dims, skipna=True)
                
                n_valid_cells = int(combined_mask.sum())
                self.logger.info(f"  Using {n_valid_cells} grid cells within catchment bounds")
            else:
                # Fallback: Calculate spatial average over entire grid
                self.logger.warning("No catchment shapefile - using entire grid")
                spatial_dims = [dim for dim in ds[temp_var].dims if dim != 'time']
                temp_spatial_avg = ds[temp_var].mean(dim=spatial_dims)
            
            # Group by month and calculate mean climatology
            monthly_climatology = temp_spatial_avg.groupby('time.month').mean()
            
            # Convert to DataFrame
            monthly_df = monthly_climatology.to_dataframe().reset_index()
            monthly_df = monthly_df.rename(columns={'month': 'month', temp_var: 'Temperature'})
            
            # Ensure we have exactly 12 months
            full_months = pd.DataFrame({'month': list(range(1, 13))})
            monthly_df = full_months.merge(monthly_df, on='month', how='left')
            
            # Round
            monthly_df['Temperature'] = monthly_df['Temperature'].round(2)
            
            # Save to CSV
            output_file = self.output_path / 'har_monthly_temperature_averages.csv'
            monthly_df.to_csv(output_file, index=False)
            
            self.logger.info(f"Monthly temperature averages saved to: {output_file}")
            
            # Log values
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            for i, row in monthly_df.iterrows():
                month_name = month_names[int(row['month']) - 1]
                temp = row['Temperature']
                self.logger.info(f"  {month_name}: {temp:.1f}°C")
            
            ds.close()
            return monthly_df
            
        except Exception as e:
            self.logger.error(f"Error calculating monthly temperature averages: {e}")
            return pd.DataFrame()

    #---------------------------------------------------------------------------------

    def calculate_monthly_pet_averages(self) -> pd.DataFrame:
        """
        Calculate monthly mean PET averages (climatology) and save to CSV
        
        Returns
        -------
        pd.DataFrame
            Monthly PET averages
        """
        self.logger.info("Calculating monthly PET averages from HAR data...")
        
        try:
            # Find PET file
            pet_file = None
            for file_path in self.processed_files:
                if 'pet' in file_path.name:
                    pet_file = file_path
                    break
            
            if pet_file is None:
                self.logger.error("PET file not found")
                return pd.DataFrame()
            
            ds = xr.open_dataset(pet_file)
            
            # Get PET variable
            pet_var = [v for v in ds.data_vars if 'time' in ds[v].dims][0]
            
            # ✅ NEW: Mask data to only include grid cells within catchment
            if self.catchment_extent is not None:
                self.logger.info("Masking PET data to catchment extent...")
                
                # Get lat/lon coordinates (HAR has 2D lat/lon)
                if 'lat' in ds.coords:
                    lats = ds['lat'].values
                    lons = ds['lon'].values
                else:
                    lats = ds['latitude'].values
                    lons = ds['longitude'].values
                
                # Get catchment bounds
                catchment_bounds = self.catchment_extent.total_bounds  # minx, miny, maxx, maxy
                
                # Create boolean mask for grid cells within catchment bounds
                # HAR has 2D lat/lon arrays
                lat_mask = (lats >= catchment_bounds[1]) & (lats <= catchment_bounds[3])
                lon_mask = (lons >= catchment_bounds[0]) & (lons <= catchment_bounds[2])
                combined_mask = lat_mask & lon_mask
                
                # Apply mask to PET data
                ds_masked = ds[pet_var].where(combined_mask, drop=False)
                
                # Calculate spatial average only over masked (catchment) area
                spatial_dims = [dim for dim in ds[pet_var].dims if dim != 'time']
                pet_spatial_avg = ds_masked.mean(dim=spatial_dims, skipna=True)
                
                n_valid_cells = int(combined_mask.sum())
                self.logger.info(f"  Using {n_valid_cells} grid cells within catchment bounds")
            else:
                # Fallback: Calculate spatial average over entire grid
                self.logger.warning("No catchment shapefile - using entire grid")
                spatial_dims = [dim for dim in ds[pet_var].dims if dim != 'time']
                pet_spatial_avg = ds[pet_var].mean(dim=spatial_dims)
            
            # Group by month and calculate mean climatology
            monthly_climatology = pet_spatial_avg.groupby('time.month').mean()
            
            # Convert to DataFrame
            monthly_df = monthly_climatology.to_dataframe().reset_index()
            monthly_df = monthly_df.rename(columns={'month': 'month', pet_var: 'PET_avg_mm_per_day'})
            
            # Ensure we have exactly 12 months
            full_months = pd.DataFrame({'month': list(range(1, 13))})
            monthly_df = full_months.merge(monthly_df, on='month', how='left')
            
            # Round
            monthly_df['PET_avg_mm_per_day'] = monthly_df['PET_avg_mm_per_day'].round(3)
            
            # Save to CSV
            output_file = self.output_path / 'har_monthly_pet_averages.csv'
            monthly_df.to_csv(output_file, index=False)
            
            self.logger.info(f"Monthly PET averages saved to: {output_file}")
            
            # Log values
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            for i, row in monthly_df.iterrows():
                month_name = month_names[int(row['month']) - 1]
                pet = row['PET_avg_mm_per_day']
                self.logger.info(f"  {month_name}: {pet:.3f} mm/day")
            
            ds.close()
            return monthly_df
            
        except Exception as e:
            self.logger.error(f"Error calculating monthly PET averages: {e}")
            return pd.DataFrame()

    #---------------------------------------------------------------------------------

    def plot_monthly_temperature_climatology(self, monthly_df: pd.DataFrame) -> None:
        """
        Create a plot of monthly temperature climatology
        
        Parameters
        ----------
        monthly_df : pd.DataFrame
            Monthly temperature data
        """
        if monthly_df.empty:
            self.logger.warning("No monthly temperature data to plot")
            return
        
        self.logger.info("Creating monthly temperature climatology plot...")
        
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            bars = ax.bar(month_names, monthly_df['Temperature'], 
                         color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1)
            
            # Add value labels
            for bar, temp in zip(bars, monthly_df['Temperature']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{temp:.1f}°C', ha='center', va='bottom', fontsize=10)
            
            ax.set_title(f'HAR Monthly Temperature Climatology - Gauge {self.gauge_id}', 
                        fontsize=16, fontweight='bold')
            ax.set_ylabel('Temperature (°C)', fontsize=12)
            ax.set_xlabel('Month', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add horizontal line at 0°C
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1, label='0°C')
            ax.legend()
            
            plt.tight_layout()
            
            save_path = self.plots_dir / f'har_monthly_temperature_climatology_gauge_{self.gauge_id}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Monthly temperature climatology plot saved to: {save_path}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating monthly temperature climatology plot: {e}")

    #---------------------------------------------------------------------------------

    def plot_monthly_pet_climatology(self, monthly_df: pd.DataFrame) -> None:
        """
        Create a plot of monthly PET climatology
        
        Parameters
        ----------
        monthly_df : pd.DataFrame
            Monthly PET data
        """
        if monthly_df.empty:
            self.logger.warning("No monthly PET data to plot")
            return
        
        self.logger.info("Creating monthly PET climatology plot...")
        
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            bars = ax.bar(month_names, monthly_df['PET_avg_mm_per_day'], 
                         color='orange', alpha=0.7, edgecolor='darkorange', linewidth=1)
            
            # Add value labels
            for bar, pet in zip(bars, monthly_df['PET_avg_mm_per_day']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{pet:.2f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_title(f'HAR Monthly PET Climatology - Gauge {self.gauge_id}', 
                        fontsize=16, fontweight='bold')
            ax.set_ylabel('PET (mm/day)', fontsize=12)
            ax.set_xlabel('Month', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            ax.set_ylim(0, monthly_df['PET_avg_mm_per_day'].max() * 1.1)
            
            plt.tight_layout()
            
            save_path = self.plots_dir / f'har_monthly_pet_climatology_gauge_{self.gauge_id}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Monthly PET climatology plot saved to: {save_path}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating monthly PET climatology plot: {e}")

    #---------------------------------------------------------------------------------

    def analyze_all_files(self) -> None:
        """
        Main method to analyze all processed files
        ✅ FIXED: Actually calls calculate_spatial_average_timeseries before plotting!
        """
        self.logger.info(f"Starting analysis of processed data for gauge {self.gauge_id}")
        self.logger.info(f"Analysis period: {self.start_date.date()} to {self.end_date.date()}")
        
        if not self.processed_files:
            self.logger.warning("No processed files found to analyze")
            return
        
        # ✅ Run missing value checks on all files first
        self.logger.info("\n" + "="*80)
        self.logger.info("🔍 MISSING VALUE CHECK")
        self.logger.info("="*80)
        
        all_missing_results = {}
        files_with_issues = []
        
        for processed_file in self.processed_files:
            missing_results = self.check_missing_values_in_netcdf(processed_file)
            all_missing_results[processed_file.name] = missing_results
            
            if missing_results.get('has_missing', False):
                files_with_issues.append(processed_file.name)
        
        # Summary
        if files_with_issues:
            self.logger.warning(f"\n⚠️ {len(files_with_issues)} file(s) contain missing values:")
            for fname in files_with_issues:
                pct = all_missing_results[fname]['total_missing_pct']
                self.logger.warning(f"  - {fname}: {pct:.2f}% missing")
        else:
            self.logger.info("\n✅ All files passed missing value check - no missing data detected!")
        
        self.logger.info("="*80 + "\n")
        
        # ✅ FIX: Continue with normal analysis - ACTUALLY create timeseries DataFrames!
        all_timeseries = {}
        
        for processed_file in self.processed_files:
            try:
                self.logger.info(f"Analyzing file: {processed_file.name}")
                
                # Create spatial overview plot
                self.plot_spatial_overview(processed_file)
                
                # ✅ FIX: Calculate spatial average timeseries FIRST
                df_timeseries = self.calculate_spatial_average_timeseries(processed_file)
                
                # ✅ FIX: THEN plot it (only if we got data)
                if not df_timeseries.empty:
                    self.plot_timeseries(processed_file, df_timeseries)
                    all_timeseries[processed_file.stem] = df_timeseries
                else:
                    self.logger.warning(f"No timeseries data to plot for {processed_file.name}")
                
            except Exception as e:
                self.logger.error(f"Error analyzing {processed_file.name}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue
        
        # Calculate and save monthly averages
        monthly_temp_df = self.calculate_monthly_temperature_averages()
        if not monthly_temp_df.empty:
            self.plot_monthly_temperature_climatology(monthly_temp_df)
        
        monthly_pet_df = self.calculate_monthly_pet_averages()
        if not monthly_pet_df.empty:
            self.plot_monthly_pet_climatology(monthly_pet_df)
        
        self.logger.info("Analysis complete!")
        self.logger.info(f"Plots saved in: {self.plots_dir}")
        self.logger.info(f"Processed meteo files saved in: {self.output_path}")




#--------------------------------------------------------------------------------
############################# GridWeights Generator #############################
#--------------------------------------------------------------------------------

class GridWeightsGenerator:
    """
    A class for generating grid weights for ERA5-Land meteorological data preprocessing.
    Converts ERA5-Land netCDF data to polygon grids and calculates weights
    for each HRU (Hydrological Response Unit).
    """
    
    def __init__(self, namelist_path: Union[str, Path], force_reprocess: bool = False) -> None:
        """
        Initialize the GridWeights Generator
        
        Parameters
        ----------
        namelist_path : str or Path
            Path to the namelist YAML configuration file
        force_reprocess : bool, optional
            If True, reprocess files even if they already exist (default: False)
        """
        # Store the force_reprocess flag
        self.force_reprocess = force_reprocess
        
        # Load configuration from namelist directly
        namelist_path = Path(namelist_path)
        
        if not namelist_path.exists():
            raise FileNotFoundError(f"Namelist file not found: {namelist_path}")
        
        with open(namelist_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Store configuration parameters
        self.main_dir = Path(self.config.get('main_dir'))
        self.gauge_id = self.config.get('gauge_id')
        self.basin_id = self.config.get('basin_id', self.gauge_id)
        self.start_date = pd.to_datetime(self.config.get('start_date'))
        self.end_date = pd.to_datetime(self.config.get('end_date'))
        
        self.model_type = self.config.get('model_type')
        self.debug = self.config.get('debug', False)
        self.coupled = self.config.get('coupled', False)
        self.model_dir = self.main_dir / self.config.get('config_dir')
        
        # ✅ Setup directories using basin_id for meteo data
        meteo_dir_template = self.config.get('meteo_dir', '01_data/meteo/gauge_{gauge_id}')
        self.era5_data_dir = Path(meteo_dir_template.format(basin_id=self.basin_id, gauge_id=self.basin_id))
        
        # Make absolute path if needed
        if not self.era5_data_dir.is_absolute():
            self.era5_data_dir = self.main_dir / self.era5_data_dir
        
        # ✅ Define SHARED catchment-level directories (primary storage)
        self.shared_data_dir = self.model_dir / f'catchment_{self.gauge_id}' / 'data_obs'
        self.shared_plots_dir = self.model_dir / f'catchment_{self.gauge_id}' / 'plots'
        
        # ✅ Define MODEL-SPECIFIC directories (for backward compatibility)
        self.model_data_dir = self.model_dir / f'catchment_{self.gauge_id}' / self.model_type / 'data_obs'
        
        # ✅ Use shared directories as primary output locations
        self.output_path = self.shared_data_dir
        self.plots_dir = self.shared_plots_dir
        
        # ✅ Set paths for GridWeights generator
        self.out_dir = self.shared_data_dir  # Where GridWeights.txt will be saved
        self.out_HRU_shape_dir = self.model_dir / f'catchment_{self.gauge_id}' / 'topo_files' / 'HRU.shp'
        
        # Create plots directories
        self.spatial_plots_dir = self.plots_dir / 'spatial_overview'
        self.timeseries_plots_dir = self.plots_dir / 'time_series'
        
        # Create all directories
        self.spatial_plots_dir.mkdir(parents=True, exist_ok=True)
        self.timeseries_plots_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.model_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Keep the old attribute name for backward compatibility
        self.processed_data_dir = self.output_path
        
        # Setup logger FIRST
        self.logger = self._setup_logger()
        
        # ✅ Load warm-up date if specified (AFTER logger is initialized)
        if 'warm_up_date' in self.config:
            self.warmup_date = pd.to_datetime(self.config.get('warm_up_date'))
            self.logger.info(f"Warm-up period configured: {self.warmup_date.date()} to {(self.start_date - pd.Timedelta(days=1)).date()}")
        else:
            self.warmup_date = None
            self.logger.debug("No warm-up period configured")
        
        # Log the output directory
        self.logger.info(f"Processing for gauge {self.gauge_id} using basin {self.basin_id} meteo data")
        self.logger.info(f"Model type: {self.model_type}")
        self.logger.info(f"Model directory: {self.model_dir}")
        self.logger.info(f"ERA5 data directory: {self.era5_data_dir}")
        self.logger.info(f"GridWeights will be saved to: {self.out_dir}")
        self.logger.info(f"GridWeights will be copied to: {self.model_data_dir}")

    #---------------------------------------------------------------------------------

    def _setup_logger(self) -> logging.Logger:
        """
        Set up and configure logger based on debug flag
        
        Returns
        -------
        logging.Logger
            Configured logger instance
        """
        # Use DEBUG level only in debug mode, otherwise use WARNING
        log_level = logging.DEBUG if self.debug else logging.WARNING
        
        # Configure logging
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=log_level,
            datefmt='%Y-%m-%d %H:%M:%S',
            force=True  # Reset existing loggers
        )

        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
        
        # Create logger with appropriate name
        logger = logging.getLogger(f'GridWeightsGenerator_Gauge_{self.gauge_id}')

        # If you want to see your INFO messages even in non-debug mode, add a special handler
        if not self.debug:
            # Create console handler that shows INFO messages from this class only
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console.setFormatter(formatter)
            
            # Add a filter to only show messages from this class
            class ModuleFilter(logging.Filter):
                def filter(self, record):
                    return 'GridWeightsGenerator' in record.name and record.levelno >= logging.INFO
                    
            console.addFilter(ModuleFilter())
            logger.addHandler(console)
        
        return logger

    #---------------------------------------------------------------------------------  

    def netCDF_to_GeoDataFrame(self, netcdf: xr.Dataset) -> gpd.GeoDataFrame:
        """
        Transform ERA5-Land netCDF file into a GeoDataFrame with point data
        
        Parameters
        ----------
        netcdf : xr.Dataset
            ERA5-Land netCDF dataset to transform
            
        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with value and geometry data in WGS84 (EPSG:4326)
        """
        self.logger.debug("Converting ERA5-Land netCDF to GeoDataFrame")
        
        ds = netcdf
        
        # Select the first data variable 
        data_vars = [var for var in ds.data_vars]
        if data_vars:
            main_var = data_vars[0]
        else:
            raise ValueError("No data variables found in netCDF dataset")
            
        self.logger.debug(f"Using data variable: {main_var}")
        
        # Select first time step if time dimension exists
        if 'time' in ds.dims:
            ds = ds.isel(time=0)
            
        xarr = ds[main_var]
        
        # ERA5-Land uses latitude/longitude coordinates
        if 'latitude' in ds.coords and 'longitude' in ds.coords:
            lat_coords = ds.coords['latitude'].values
            lon_coords = ds.coords['longitude'].values
        elif 'lat' in ds.coords and 'lon' in ds.coords:
            lat_coords = ds.coords['lat'].values
            lon_coords = ds.coords['lon'].values
        else:
            raise ValueError("Cannot find latitude/longitude coordinates in ERA5-Land netCDF")
        
        self.logger.debug(f"Original latitude range: {lat_coords.min():.4f} to {lat_coords.max():.4f}")
        self.logger.debug(f"Original longitude range: {lon_coords.min():.4f} to {lon_coords.max():.4f}")
        
        # 🔧 FIX: Check if latitude coordinates are decreasing (indicating flip needed)
        lat_decreasing = lat_coords[0] > lat_coords[-1]
        
        if lat_decreasing:
            self.logger.warning("⚠️  Latitude coordinates are decreasing - data appears to be vertically flipped!")
            self.logger.warning("🔄 Flipping latitude coordinates and data array to correct orientation")
            
            # Flip latitude coordinates
            lat_coords = np.flipud(lat_coords)
            
            # Also flip the data array along the latitude dimension
            if 'latitude' in xarr.dims:
                xarr = xarr.sel(latitude=slice(None, None, -1))  # Reverse latitude dimension
            elif 'lat' in xarr.dims:
                xarr = xarr.sel(lat=slice(None, None, -1))  # Reverse lat dimension
            
            self.logger.info(f"✅ Corrected latitude range: {lat_coords.min():.4f} to {lat_coords.max():.4f}")
            self.logger.info("🗺️  Data orientation corrected - latitudes now increase northward")
        else:
            self.logger.debug("✅ Latitude coordinates are already correctly oriented")
        
        # Create meshgrid for all coordinate combinations
        LON, LAT = np.meshgrid(lon_coords, lat_coords)
        
        # Get data values - handle different array structures
        if xarr.dims == ('latitude', 'longitude') or xarr.dims == ('lat', 'lon'):
            data_values = xarr.values.flatten()
        else:
            # If scalar or other dimension structure
            if xarr.size == 1:
                data_values = np.full(LON.size, xarr.values)
            else:
                data_values = xarr.values.flatten()
        
        # Create DataFrame with all coordinate combinations
        df = pd.DataFrame({
            'longitude': LON.flatten(),
            'latitude': LAT.flatten(),
            main_var: data_values
        })
        
        # Convert the DataFrame to a GeoDataFrame using WGS84 (EPSG:4326)
        self.logger.debug("Creating points geometry from longitude, latitude coordinates")
        
        grid_points = gpd.GeoDataFrame(
            df[[main_var]],  # Keep the data variable as a column
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326"  # ERA5-Land is in WGS84
        )
        
        self.logger.info(f"Created GeoDataFrame with {len(grid_points)} points")
        
        # 🔍 VERIFICATION: Log some sample coordinates for verification
        if lat_decreasing:
            self.logger.debug("📍 Sample corrected coordinates (first 3 points):")
            for i in range(min(3, len(grid_points))):
                point = grid_points.iloc[i]
                self.logger.debug(f"  Point {i}: Lat={point.geometry.y:.4f}, Lon={point.geometry.x:.4f}, Value={point[main_var]:.3f}")
        
        return grid_points

    #---------------------------------------------------------------------------------

    def make_polygon_shape_from_point_shape(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Create a polygon grid from ERA5-Land point data (OPTIMIZED)
        """
        self.logger.debug("Creating polygon grid from ERA5-Land point data")
        
        # Get unique coordinates
        sorted_x = np.sort(gdf.geometry.x.unique())
        sorted_y = np.sort(gdf.geometry.y.unique())
        
        self.logger.debug(f"Grid: {len(sorted_x)} x {len(sorted_y)} cells")
        
        # Infer cell size
        width = np.median(np.diff(sorted_x)) if len(sorted_x) > 1 else 0.1
        height = np.median(np.diff(sorted_y)) if len(sorted_y) > 1 else 0.1
        
        self.logger.debug(f"Cell size: {width:.6f}° x {height:.6f}°")
        
        # ✅ OPTIMIZED: Create boundaries using numpy operations
        cols = np.concatenate([
            [sorted_x[0] - width/2],
            sorted_x + width/2
        ])
        
        rows = np.concatenate([
            [sorted_y[0] - height/2],
            sorted_y + height/2
        ])
        
        # ✅ OPTIMIZED: Vectorized polygon creation using list comprehension
        polygons = [
            Polygon([
                (cols[ix], rows[iy]),
                (cols[ix+1], rows[iy]),
                (cols[ix+1], rows[iy+1]),
                (cols[ix], rows[iy+1])
            ])
            for iy in range(len(rows)-1)
            for ix in range(len(cols)-1)
        ]
        
        # ✅ OPTIMIZED: Vectorized cell_id creation
        cell_ids = [str(iy * (len(cols)-1) + ix) 
                    for iy in range(len(rows)-1) 
                    for ix in range(len(cols)-1)]
        
        # Create GeoDataFrame directly
        grid = gpd.GeoDataFrame({
            'cell_id': cell_ids,
            'area_rel': 0,
            'geometry': polygons
        }, crs='EPSG:4326')
        
        self.logger.info(f"Created grid with {len(grid)} cells")
        
        return grid
    
    #---------------------------------------------------------------------------------
        
    def create_overlay(self, grid: gpd.GeoDataFrame, hru: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Create overlay of grid and HRU shapes, handling CRS transformation
        
        Parameters
        ----------
        grid : gpd.GeoDataFrame
            Grid polygons in WGS84 (EPSG:4326)
        hru : gpd.GeoDataFrame
            HRU polygons (any CRS)
            
        Returns
        -------
        gpd.GeoDataFrame
            Intersection of grid and HRU polygons
        """
        self.logger.debug("Creating overlay of ERA5-Land grid and HRU shapes")
        
        # Check and align CRS
        if hru.crs != grid.crs:
            self.logger.debug(f"Reprojecting HRU from {hru.crs} to {grid.crs}")
            hru_reprojected = hru.to_crs(grid.crs)
        else:
            hru_reprojected = hru.copy()
        
        self.logger.debug(f"Grid CRS: {grid.crs}, HRU CRS: {hru_reprojected.crs}")
        
        result = hru_reprojected.overlay(grid, how='intersection')
        self.logger.debug(f"Created overlay with {len(result)} intersections")
        
        result.set_index("cell_id")
        return result

    #---------------------------------------------------------------------------------

    def calc_relative_area(self, overlay: gpd.GeoDataFrame, hru: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calculate the relative area of each HRU in each ERA5-Land grid cell (VECTORIZED)
        """
        self.logger.debug("Calculating relative areas for ERA5-Land grid cells (vectorized)")
        
        # Find HRU ID column
        hru_id_col = None
        for col in overlay.columns:
            if 'hru' in col.lower() and 'id' in col.lower():
                hru_id_col = col
                break
        
        if hru_id_col is None:
            raise ValueError("Could not find HRU ID column in overlay data")
        
        self.logger.debug(f"Using HRU ID column: {hru_id_col}")
        
        # ✅ VECTORIZED: Calculate all areas at once
        overlay_copy = overlay.copy()
        
        # Calculate area in square meters (project if needed)
        if overlay_copy.crs.to_string() == 'EPSG:4326':
            overlay_proj = overlay_copy.to_crs('ESRI:54009')
            overlay_copy["area"] = overlay_proj.geometry.area
        else:
            overlay_copy["area"] = overlay_copy.geometry.area
        
        # ✅ VECTORIZED: Group by HRU and calculate relative areas using pandas
        # Calculate total area per HRU
        hru_totals = overlay_copy.groupby(hru_id_col)['area'].transform('sum')
        
        # Calculate relative area (avoiding division by zero)
        overlay_copy['area_rel'] = np.where(
            hru_totals > 0,
            overlay_copy['area'] / hru_totals,
            0
        )
        
        # Round to 5 decimal places
        overlay_copy['area_rel'] = overlay_copy['area_rel'].round(5)
        
        # ✅ VECTORIZED: Normalize within each HRU group
        # Group by HRU and normalize
        def normalize_group(group):
            total = group['area_rel'].sum()
            if total > 0:
                group['normalized_relative_area'] = (group['area_rel'] / total).round(5)
            else:
                group['normalized_relative_area'] = 0
            return group
        
        result_gdf = overlay_copy.groupby(hru_id_col, group_keys=False).apply(normalize_group)
        
        self.logger.debug("Relative area calculation completed")
        return result_gdf

    #---------------------------------------------------------------------------------

    def write_gridWeights(self, hru: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, relative_area: gpd.GeoDataFrame) -> None:
        """
        Write ERA5-Land grid weights to file
        
        Parameters
        ----------
        hru : gpd.GeoDataFrame
            HRU polygons
        grid : gpd.GeoDataFrame
            Grid polygons
        relative_area : gpd.GeoDataFrame
            Calculated relative areas
            
        Returns
        -------
        None
        """
        self.logger.debug("Writing ERA5-Land grid weights to file")
        
        # Find HRU ID column
        hru_id_col = None
        for col in relative_area.columns:
            if 'hru' in col.lower() and 'id' in col.lower():
                hru_id_col = col
                break
        
        if hru_id_col is None:
            raise ValueError("Could not find HRU ID column in relative area data")
        
        # list all parameters that are needed for gridweights file 
        number_HRUs = len(hru)
        number_cells = len(grid)
        HRU_list = list(relative_area[hru_id_col])
        cell_id = list(relative_area['cell_id'])
        rel_area = list(relative_area['normalized_relative_area'])
        filename = self.out_dir / 'GridWeights.txt'
        
        self.logger.debug(f"Writing to {filename}")
        self.logger.debug(f"Number of HRUs: {number_HRUs}, Number of cells: {number_cells}")
        self.logger.debug(f"Number of weight entries: {len(HRU_list)}")
        
        # Create directory if it doesn't exist
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the gridweights file for Raven
        with open(filename, 'w') as ff:
            ff.write('# ---------------------------------------------- \n')
            ff.write('# Raven GridWeights file for ERA5-Land data     \n')
            ff.write('# Generated by GridWeightsGenerator              \n')
            ff.write(f'# Catchment: {self.gauge_id}                    \n')
            ff.write(f'# Model type: {self.model_type}                 \n')
            ff.write('# ---------------------------------------------- \n')
            ff.write('\n')
            ff.write(':GridWeights                     \n')
            ff.write('   #                                \n')
            ff.write('   # [# HRUs]                       \n')
            ff.write('   :NumberHRUs       {}            \n'.format(number_HRUs))
            ff.write('   :NumberGridCells       {}            \n'.format(number_cells))
            ff.write('   #                                \n')
            ff.write('   # [HRU ID] [Cell #] [w_kl]       \n')
            for i in range(len(relative_area)):
                ff.write("   {}   {}   {}\n".format(HRU_list[i], cell_id[i], rel_area[i]))
            ff.write(':EndGridWeights \n')
            
        self.logger.info(f"ERA5-Land grid weights written to {filename}")

    #---------------------------------------------------------------------------------

    def generate(self, use_precipitation: bool = True) -> gpd.GeoDataFrame:
        """
        Generate grid weights file from ERA5-Land netCDF and HRU data
        
        Parameters
        ----------
        use_precipitation : bool, optional
            Whether to use precipitation file (True) or temperature file (False) as grid reference
            
        Returns
        -------
        gpd.GeoDataFrame
            Relative area calculations
        """
        self.logger.info(f"Generating ERA5-Land grid weights for catchment {self.gauge_id}")
        
        # ✅ CHECK 1: Skip if GridWeights.txt already exists
        gridweights_file = self.out_dir / 'GridWeights.txt'
        
        if gridweights_file.exists():
            self.logger.info(f"✅ GridWeights.txt already exists at {gridweights_file}")
            self.logger.info("⏭️  Skipping grid weights generation")
            self.logger.info("💡 Delete the file to regenerate: rm " + str(gridweights_file))
            
            # ✅ NEW: Copy to model directory even if skipping generation
            self._copy_gridweights_to_model_directory()
            
            # Return empty GeoDataFrame since we're skipping
            return gpd.GeoDataFrame()
        
        # ✅ CHECK 2: Check for cached overlay calculation
        cache_file = self.out_dir / 'grid_overlay_cache.pkl'
        
        if cache_file.exists():
            self.logger.info("📦 Loading cached overlay data...")
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                relative_area = cached_data['relative_area']
                HRU = cached_data['HRU']
                era5_grid_polygons = cached_data['era5_grid_polygons']
                
                self.logger.info("✅ Loaded from cache - skipping overlay calculations")
                
                # Jump straight to writing file
                self.write_gridWeights(HRU, era5_grid_polygons, relative_area)
                
                # ✅ NEW: Copy GridWeights.txt to model-specific directory
                self._copy_gridweights_to_model_directory()
                
                return relative_area
                
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to load cache: {e}, recalculating...")
                cache_file.unlink(missing_ok=True)
        
        # ===== NORMAL PROCESSING (if no cache) =====
        
        self.logger.info("Processing grid weights (no cache found)...")
        
        # Find ERA5-Land netCDF files
        if use_precipitation:
            era5_file = self.out_dir / f'era5_land_precip.nc'
            file_type = "precipitation"
        else:
            era5_file = self.out_dir / f'era5_land_temp_mean.nc'
            file_type = "temperature"
        
        if not era5_file.exists():
            raise FileNotFoundError(f"ERA5-Land {file_type} file not found: {era5_file}")
        
        self.logger.debug(f"Loading ERA5-Land {file_type} data from {era5_file}")
        xds = xr.open_dataset(era5_file)
        
        # Load HRU shapefile
        if not self.out_HRU_shape_dir.exists():
            raise FileNotFoundError(f"HRU shapefile not found: {self.out_HRU_shape_dir}")
            
        self.logger.debug(f"Loading HRU shapefile from {self.out_HRU_shape_dir}")
        HRU = gpd.read_file(self.out_HRU_shape_dir)
        
        # Ensure HRU ID column exists and is properly formatted
        if 'HRU_ID' in HRU.columns:
            HRU = HRU.sort_values(by='HRU_ID').reset_index(drop=True)
            HRU['HRU ID'] = HRU['HRU_ID']  # Create standardized column name
        elif 'HRU ID' in HRU.columns:
            HRU = HRU.sort_values(by='HRU ID').reset_index(drop=True)
        else:
            # Create HRU ID column if it doesn't exist
            HRU['HRU ID'] = list(range(1, len(HRU) + 1))
            
        self.logger.debug(f"Loaded {len(HRU)} HRUs")
        
        # Transform ERA5-Land netcdf to GeoDataFrame with point data
        era5_grid_points = self.netCDF_to_GeoDataFrame(xds)
        
        # Plot points over HRU shapefile if debug is enabled
        if self.debug:
            self.logger.debug("Plotting ERA5-Land grid points over HRU shapefile")
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Reproject HRU to WGS84 for plotting with ERA5 data
            HRU_wgs84 = HRU.to_crs('EPSG:4326')
            
            HRU_wgs84.plot(ax=ax, color='lightblue', edgecolor='blue', alpha=0.7, linewidth=1)
            era5_grid_points.plot(ax=ax, color='red', markersize=20, alpha=0.8)
            
            plt.title(f"ERA5-Land Grid Points for Catchment {self.gauge_id}", fontsize=14, fontweight='bold')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = self.plots_dir / 'era5_grid_points.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.debug(f"Grid points plot saved to {plot_path}")
            plt.show()
        
        # Transform point data into polygons
        era5_grid_polygons = self.make_polygon_shape_from_point_shape(era5_grid_points)
        
        # Plot polygons over HRU shapefile if debug is enabled
        if self.debug:
            self.logger.debug("Plotting ERA5-Land grid polygons over HRU shapefile")
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Reproject HRU to WGS84 for plotting
            HRU_wgs84 = HRU.to_crs('EPSG:4326')
            
            HRU_wgs84.plot(ax=ax, color='lightblue', edgecolor='blue', alpha=0.7, linewidth=1)
            era5_grid_polygons.plot(ax=ax, color='red', edgecolor='darkred', alpha=0.5, linewidth=0.5)
            
            plt.title(f"ERA5-Land Grid Polygons for Catchment {self.gauge_id}", fontsize=14, fontweight='bold')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = self.plots_dir / 'era5_grid_polygons.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.debug(f"Grid polygons plot saved to {plot_path}")
            plt.show()
        
        # Create union overlay GeoDataFrames
        res_union = self.create_overlay(era5_grid_polygons, HRU)
        
        # Calculate relative area for each HRU file
        relative_area = self.calc_relative_area(res_union, HRU)

        # ✅ SAVE TO CACHE for next time
        try:
            import pickle
            self.logger.info("💾 Saving overlay cache...")
            
            cached_data = {
                'relative_area': relative_area,
                'HRU': HRU,
                'era5_grid_polygons': era5_grid_polygons
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            
            cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
            self.logger.info(f"✅ Saved overlay cache ({cache_size_mb:.2f} MB)")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to save cache: {e}")
            # Don't fail if caching fails
            pass

        # Write gridweights file
        self.write_gridWeights(HRU, era5_grid_polygons, relative_area)
        
        # 🔍 VERIFICATION: Add detailed verification after generating grid weights
        if self.debug:
            self.logger.info("🔍 Verifying grid orientation with HRU bounds...")
            
            # Get bounds of HRU in WGS84
            HRU_wgs84 = HRU.to_crs('EPSG:4326')
            hru_bounds = HRU_wgs84.total_bounds  # [minx, miny, maxx, maxy]
            
            # Get bounds of ERA5 grid
            grid_bounds = era5_grid_polygons.total_bounds
            
            self.logger.info(f"HRU bounds: Lat [{hru_bounds[1]:.4f}, {hru_bounds[3]:.4f}], Lon [{hru_bounds[0]:.4f}, {hru_bounds[2]:.4f}]")
            self.logger.info(f"Grid bounds: Lat [{grid_bounds[1]:.4f}, {grid_bounds[3]:.4f}], Lon [{grid_bounds[0]:.4f}, {grid_bounds[2]:.4f}]")
            
            # Check for reasonable overlap
            lat_overlap = not (hru_bounds[3] < grid_bounds[1] or hru_bounds[1] > grid_bounds[3])
            lon_overlap = not (hru_bounds[2] < grid_bounds[0] or hru_bounds[0] > grid_bounds[2])
            
            if lat_overlap and lon_overlap:
                self.logger.info("✅ HRU and grid bounds overlap - orientation looks correct")
            else:
                self.logger.error("❌ HRU and grid bounds DO NOT overlap - possible orientation error!")
                self.logger.error("🚨 Check your data orientation - HRUs might be mapped to wrong grid cells!")
                
                # Additional debugging information
                lat_gap = max(0, max(hru_bounds[1] - grid_bounds[3], grid_bounds[1] - hru_bounds[3]))
                lon_gap = max(0, max(hru_bounds[0] - grid_bounds[2], grid_bounds[0] - hru_bounds[2]))
                
                if lat_gap > 0:
                    self.logger.error(f"  Latitude gap: {lat_gap:.4f} degrees")
                if lon_gap > 0:
                    self.logger.error(f"  Longitude gap: {lon_gap:.4f} degrees")
            
            # Check grid weight statistics
            self.logger.info("📊 Grid weights statistics:")
            self.logger.info(f"  Total grid cells: {len(era5_grid_polygons)}")
            self.logger.info(f"  Total HRUs: {len(HRU)}")
            self.logger.info(f"  Total weight entries: {len(relative_area)}")
            self.logger.info(f"  Sum of all weights: {relative_area['normalized_relative_area'].sum():.6f}")
            
            # Check if weights sum to number of HRUs (should be close to len(HRU))
            expected_sum = len(HRU)
            actual_sum = relative_area['normalized_relative_area'].sum()
            if abs(actual_sum - expected_sum) > 0.01:
                self.logger.warning(f"⚠️ Weight sum ({actual_sum:.6f}) differs from expected ({expected_sum})")
            else:
                self.logger.info(f"✅ Weight sum verification passed")
        
        self.logger.info("ERA5-Land grid weights generation completed successfully")
        
        # ✅ NEW: Copy GridWeights.txt to model-specific directory
        self._copy_gridweights_to_model_directory()
        
        return relative_area
    
    #---------------------------------------------------------------------------------
    
    def _copy_gridweights_to_model_directory(self) -> None:
        """
        Copy GridWeights.txt from shared data_obs to model-specific data_obs.
        This maintains backward compatibility while using shared storage.
        """
        import shutil
        
        gridweights_file = self.shared_data_dir / 'GridWeights.txt'
        
        if not gridweights_file.exists():
            self.logger.warning(f"⚠️ GridWeights.txt not found in shared directory: {gridweights_file}")
            return
        
        dest = self.model_data_dir / 'GridWeights.txt'
        
        try:
            shutil.copy2(gridweights_file, dest)
            self.logger.info(f"📋 Copied GridWeights.txt to: {self.model_data_dir}")
            self.logger.debug(f"  Source: {gridweights_file}")
            self.logger.debug(f"  Destination: {dest}")
        except Exception as e:
            self.logger.warning(f"❌ Failed to copy GridWeights.txt: {e}")


#--------------------------------------------------------------------------------
############################# HAR GridWeights Generator #########################
#--------------------------------------------------------------------------------

class HARGridWeightsGenerator:
    """
    A class for generating grid weights for HAR meteorological data.
    Handles the Lambert Conformal Conic projection and curvilinear grid.
    """
    
    def __init__(self, namelist_path: Union[str, Path], force_reprocess: bool = False) -> None:
        """
        Initialize the HAR GridWeights Generator
        
        Parameters
        ----------
        namelist_path : str or Path
            Path to the namelist YAML configuration file
        force_reprocess : bool, optional
            If True, reprocess files even if they already exist (default: False)
        """
        self.force_reprocess = force_reprocess
        
        # Load configuration
        namelist_path = Path(namelist_path)
        
        if not namelist_path.exists():
            raise FileNotFoundError(f"Namelist file not found: {namelist_path}")
        
        with open(namelist_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Store configuration parameters
        self.main_dir = Path(self.config.get('main_dir'))
        self.gauge_id = self.config.get('gauge_id')
        self.model_type = self.config.get('model_type')
        self.debug = self.config.get('debug', False)
        self.model_dir = self.main_dir / self.config.get('config_dir')
        
        # Setup directories
        self.shared_data_dir = self.model_dir / f'catchment_{self.gauge_id}' / 'data_obs'
        self.model_data_dir = self.model_dir / f'catchment_{self.gauge_id}' / self.model_type / 'data_obs'
        self.out_dir = self.shared_data_dir
        self.out_HRU_shape_dir = self.model_dir / f'catchment_{self.gauge_id}' / 'topo_files' / 'HRU.shp'
        self.plots_dir = self.model_dir / f'catchment_{self.gauge_id}' / 'plots'
        
        # Create directories
        self.shared_data_dir.mkdir(parents=True, exist_ok=True)
        self.model_data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
        
        self.logger.info(f"HAR GridWeights Generator initialized for gauge {self.gauge_id}")

    #---------------------------------------------------------------------------------

    def _setup_logger(self) -> logging.Logger:
        """Set up and configure logger"""
        level = logging.DEBUG if self.debug else logging.INFO
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            force=True
        )
        
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        
        return logging.getLogger(f'HARGridWeightsGenerator_Gauge_{self.gauge_id}')

    #---------------------------------------------------------------------------------

    def generate(self) -> gpd.GeoDataFrame:
        """
        Generate grid weights file for HAR data
        ✅ FIXED: Correct cell numbering and handle HRUs with no grid overlap
        
        Returns
        -------
        gpd.GeoDataFrame
            Relative area calculations
        """
        self.logger.info(f"Generating HAR grid weights for catchment {self.gauge_id}")
        
        # Check if GridWeights file already exists
        gridweights_file = self.out_dir / 'GridWeights_HAR.txt'
        
        if gridweights_file.exists() and not self.force_reprocess:
            self.logger.info(f"✅ GridWeights_HAR.txt already exists")
            self.logger.info("⏭️ Skipping grid weights generation")
            self._copy_gridweights_to_model_directory()
            return gpd.GeoDataFrame()
        
        # Find HAR precipitation file (use as grid reference)
        har_file = self.out_dir / 'har_precip.nc'
        
        if not har_file.exists():
            raise FileNotFoundError(f"HAR precipitation file not found: {har_file}")
        
        self.logger.info(f"Loading HAR grid from {har_file}")
        ds = xr.open_dataset(har_file)
        
        # Load HRU shapefile
        if not self.out_HRU_shape_dir.exists():
            raise FileNotFoundError(f"HRU shapefile not found: {self.out_HRU_shape_dir}")
        
        HRU = gpd.read_file(self.out_HRU_shape_dir)
        
        # Ensure HRU ID column exists
        if 'HRU_ID' in HRU.columns:
            HRU = HRU.sort_values(by='HRU_ID').reset_index(drop=True)
            HRU['HRU ID'] = HRU['HRU_ID']
        elif 'HRU ID' not in HRU.columns:
            HRU['HRU ID'] = list(range(1, len(HRU) + 1))
        
        self.logger.info(f"Loaded {len(HRU)} HRUs")
        
        # Reproject HRU to WGS84
        HRU_wgs84 = HRU.to_crs('EPSG:4326')
        
        # ✅ FIX: Get the ACTUAL grid dimensions from the NetCDF file
        # HAR uses south_north and west_east dimensions
        if 'south_north' in ds.dims and 'west_east' in ds.dims:
            ny = ds.dims['south_north']
            nx = ds.dims['west_east']
        else:
            # Fallback to lat/lon shape
            lat_2d = ds.coords['lat'].values
            ny, nx = lat_2d.shape
        
        self.logger.info(f"HAR grid dimensions: {ny} x {nx} = {ny * nx} cells")
        
        # Get 2D lat/lon coordinates
        lat_2d = ds.coords['lat'].values
        lon_2d = ds.coords['lon'].values
        
        # ✅ FIX: Calculate cell size from coordinate spacing
        # Use average spacing between adjacent points
        dlat = np.abs(np.diff(lat_2d, axis=0)).mean() / 2
        dlon = np.abs(np.diff(lon_2d, axis=1)).mean() / 2
        
        self.logger.info(f"Estimated half-cell size: dlat={dlat:.4f}°, dlon={dlon:.4f}°")
        
        # ✅ FIX: Create polygons for EACH grid cell (ny * nx cells)
        # Each polygon is centered on the grid point
        self.logger.info("Creating HAR grid polygons (catchment-extent only)...")

        # Pre-compute catchment bounding box for fast filtering
        hru_minx, hru_miny, hru_maxx, hru_maxy = HRU_wgs84.total_bounds
        # generous buffer: 2 full cells
        buf_lat = dlat * 4
        buf_lon = dlon * 4

        polygons = []
        cell_ids = []

        for j in range(ny):
            for i in range(nx):
                # Get center point of this cell
                center_lat = lat_2d[j, i]
                center_lon = lon_2d[j, i]

                # Skip cells clearly outside the catchment (+buffer)
                if (center_lon < hru_minx - buf_lon or center_lon > hru_maxx + buf_lon or
                        center_lat < hru_miny - buf_lat or center_lat > hru_maxy + buf_lat):
                    continue

                # Create polygon around the center point
                corners = [
                    (center_lon - dlon, center_lat - dlat),  # lower-left
                    (center_lon + dlon, center_lat - dlat),  # lower-right
                    (center_lon + dlon, center_lat + dlat),  # upper-right
                    (center_lon - dlon, center_lat + dlat),  # upper-left
                ]

                poly = Polygon(corners)
                polygons.append(poly)

                # ✅ FIX: Cell ID matches flattened index (row-major order)
                cell_id = j * nx + i
                cell_ids.append(str(cell_id))
        
        # Create GeoDataFrame
        har_grid = gpd.GeoDataFrame({
            'cell_id': cell_ids,
            'area_rel': 0,
            'geometry': polygons
        }, crs='EPSG:4326')
        
        self.logger.info(f"Created HAR grid with {len(har_grid)} cells")
        self.logger.info(f"Cell IDs range: 0 to {len(har_grid) - 1}")
        
        # ✅ Plot HAR grid polygons over HRU shapefile if debug is enabled
        if self.debug:
            self.logger.debug("Plotting HAR grid polygons over HRU shapefile")
            fig, ax = plt.subplots(figsize=(12, 10))

            # Simplify HRU geometries for plotting only (avoids slow render with 500+ complex polygons)
            HRU_plot = HRU_wgs84.copy()
            HRU_plot['geometry'] = HRU_plot.geometry.simplify(0.001)
            HRU_plot.plot(ax=ax, color='lightblue', edgecolor='blue', alpha=0.7, linewidth=1)

            # Plot HAR grid
            har_grid.plot(ax=ax, facecolor='none', edgecolor='red', alpha=0.5, linewidth=0.5)

            plt.title(f"HAR Grid Polygons for Catchment {self.gauge_id}\n({len(har_grid)} cells)",
                    fontsize=14, fontweight='bold')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, alpha=0.3)

            # Save plot
            plot_path = self.plots_dir / 'har_grid_polygons.png'
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            self.logger.info(f"HAR grid polygons plot saved to {plot_path}")
            plt.close()
        
        # Clip HAR grid to catchment bounding box before overlay (major speedup)
        minx, miny, maxx, maxy = HRU_wgs84.total_bounds
        buf = max(dlat, dlon) * 2  # two-cell buffer
        har_grid_clip = har_grid.cx[minx - buf : maxx + buf, miny - buf : maxy + buf]
        self.logger.info(
            f"Clipped HAR grid to catchment extent: {len(har_grid)} → {len(har_grid_clip)} cells"
        )

        # Create overlay
        self.logger.info("Creating overlay of HAR grid and HRU shapes...")
        res_union = HRU_wgs84.overlay(har_grid_clip, how='intersection')
        
        # Calculate relative areas
        self.logger.info("Calculating relative areas...")
        
        # Find HRU ID column
        hru_id_col = 'HRU ID' if 'HRU ID' in res_union.columns else 'HRU_ID'
        
        # Calculate areas
        res_union_proj = res_union.to_crs('ESRI:54009')
        res_union['area'] = res_union_proj.geometry.area
        
        # Calculate total area per HRU
        hru_totals = res_union.groupby(hru_id_col)['area'].transform('sum')
        res_union['area_rel'] = np.where(hru_totals > 0, res_union['area'] / hru_totals, 0)
        res_union['area_rel'] = res_union['area_rel'].round(5)
        
        # Normalize within each HRU
        def normalize_group(group):
            total = group['area_rel'].sum()
            if total > 0:
                group['normalized_relative_area'] = (group['area_rel'] / total).round(5)
            else:
                group['normalized_relative_area'] = 0
            return group
        
        relative_area = res_union.groupby(hru_id_col, group_keys=False).apply(normalize_group)
        
        # ✅ FIX: Check for HRUs with no grid overlap and assign nearest cell
        all_hru_ids = set(HRU_wgs84[hru_id_col].values)
        hrus_with_weights = set(relative_area[hru_id_col].values)
        missing_hrus = all_hru_ids - hrus_with_weights
        
        if missing_hrus:
            self.logger.warning(f"⚠️ {len(missing_hrus)} HRUs have no grid overlap!")
            self.logger.warning(f"   Missing HRU IDs: {sorted(missing_hrus)[:10]}{'...' if len(missing_hrus) > 10 else ''}")
            self.logger.info("🔧 Assigning nearest grid cell to each missing HRU...")
            
            # For each missing HRU, find the nearest grid cell centroid
            new_rows = []
            
            for hru_id in missing_hrus:
                # Get the HRU geometry
                hru_geom = HRU_wgs84[HRU_wgs84[hru_id_col] == hru_id].geometry.values[0]
                hru_centroid = hru_geom.centroid
                
                # Find nearest grid cell
                distances = har_grid.geometry.centroid.distance(hru_centroid)
                nearest_idx = distances.idxmin()
                nearest_cell_id = har_grid.loc[nearest_idx, 'cell_id']
                
                self.logger.debug(f"   HRU {hru_id}: assigned to cell {nearest_cell_id}")
                
                # Create new row with weight = 1.0
                new_row = {
                    hru_id_col: hru_id,
                    'cell_id': nearest_cell_id,
                    'area_rel': 1.0,
                    'normalized_relative_area': 1.0,
                    'area': 0,
                    'geometry': hru_geom
                }
                new_rows.append(new_row)
            
            # Add new rows to relative_area
            if new_rows:
                new_df = gpd.GeoDataFrame(new_rows, crs=relative_area.crs)
                relative_area = pd.concat([relative_area, new_df], ignore_index=True)
                self.logger.info(f"✅ Added {len(new_rows)} missing HRU assignments")
        
        # ✅ VERIFICATION: Check that all HRUs now have weights
        final_hrus_with_weights = set(relative_area[hru_id_col].values)
        still_missing = all_hru_ids - final_hrus_with_weights
        
        if still_missing:
            self.logger.error(f"❌ Still missing {len(still_missing)} HRUs after fix!")
        else:
            self.logger.info(f"✅ All {len(all_hru_ids)} HRUs have grid weights")
        
        # ✅ VERIFICATION: Check weight sums per HRU
        weight_sums = relative_area.groupby(hru_id_col)['normalized_relative_area'].sum()
        bad_sums = weight_sums[~np.isclose(weight_sums, 1.0, atol=0.001)]
        
        if len(bad_sums) > 0:
            self.logger.warning(f"⚠️ {len(bad_sums)} HRUs have weight sums != 1.0")
            for hru_id, weight_sum in bad_sums.head(5).items():
                self.logger.warning(f"   HRU {hru_id}: sum = {weight_sum:.6f}")
        else:
            self.logger.info(f"✅ All HRU weight sums equal 1.0")
        
        # Write grid weights file
        self._write_gridweights(HRU, har_grid, relative_area, hru_id_col)
        
        # Copy to model directory
        self._copy_gridweights_to_model_directory()
        
        ds.close()
        
        self.logger.info("HAR grid weights generation completed!")
        
        return relative_area

    #---------------------------------------------------------------------------------

    def _write_gridweights(self, hru: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, 
                          relative_area: gpd.GeoDataFrame, hru_id_col: str) -> None:
        """
        Write HAR grid weights to file
        """
        number_HRUs = len(hru)
        number_cells = len(grid)
        HRU_list = list(relative_area[hru_id_col])
        cell_id = list(relative_area['cell_id'])
        rel_area = list(relative_area['normalized_relative_area'])
        
        filename = self.out_dir / 'GridWeights_HAR.txt'
        
        self.logger.info(f"Writing HAR grid weights to {filename}")
        
        with open(filename, 'w') as ff:
            ff.write('# ---------------------------------------------- \n')
            ff.write('# Raven GridWeights file for HAR v2 data         \n')
            ff.write('# Generated by HARGridWeightsGenerator           \n')
            ff.write(f'# Catchment: {self.gauge_id}                    \n')
            ff.write(f'# Model type: {self.model_type}                 \n')
            ff.write('# ---------------------------------------------- \n')
            ff.write('\n')
            ff.write(':GridWeights                     \n')
            ff.write('   #                                \n')
            ff.write('   # [# HRUs]                       \n')
            ff.write('   :NumberHRUs       {}            \n'.format(number_HRUs))
            ff.write('   :NumberGridCells       {}            \n'.format(number_cells))
            ff.write('   #                                \n')
            ff.write('   # [HRU ID] [Cell #] [w_kl]       \n')
            for i in range(len(relative_area)):
                ff.write("   {}   {}   {}\n".format(HRU_list[i], cell_id[i], rel_area[i]))
            ff.write(':EndGridWeights \n')
        
        self.logger.info(f"HAR grid weights written to {filename}")

    #---------------------------------------------------------------------------------

    def _copy_gridweights_to_model_directory(self) -> None:
        """
        Copy GridWeights_HAR.txt to model-specific directory
        """
        import shutil
        
        gridweights_file = self.shared_data_dir / 'GridWeights_HAR.txt'
        
        if not gridweights_file.exists():
            self.logger.warning(f"GridWeights_HAR.txt not found")
            return
        
        dest = self.model_data_dir / 'GridWeights_HAR.txt'
        
        try:
            shutil.copy2(gridweights_file, dest)
            self.logger.info(f"📋 Copied GridWeights_HAR.txt to: {self.model_data_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to copy GridWeights_HAR.txt: {e}")