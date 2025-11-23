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
        self.start_date = pd.to_datetime(self.config.get('start_date'))
        self.end_date = pd.to_datetime(self.config.get('end_date'))
        self.model_type = self.config.get('model_type')
        self.debug = self.config.get('debug', False)
        self.coupled = self.config.get('coupled', False)
        self.model_dir = self.main_dir / self.config.get('config_dir')
        
        # Setup directories
        self.era5_data_dir = Path(self.config['meteo_dir'].format(gauge_id=self.gauge_id))
        
        # Updated plots directory structure
        self.plots_dir = self.model_dir / f'catchment_{self.gauge_id}' / self.model_type / 'plots'
        
        # Create plots directories
        self.spatial_plots_dir = self.plots_dir / 'spatial_overview'
        self.timeseries_plots_dir = self.plots_dir / 'time_series'
        
        # Updated output path for processed data
        self.output_path = self.model_dir / f'catchment_{self.gauge_id}' / self.model_type / 'data_obs'
        
        # Create all directories
        self.spatial_plots_dir.mkdir(parents=True, exist_ok=True)
        self.timeseries_plots_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
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
            'pev': {  # Add PET variable
                'name': 'Potential Evapotranspiration',
                'units': 'mm',
                'cmap': 'Oranges',
                'convert_kelvin': False
            }
        }
        
        # Setup logger
        self.logger = self._setup_logger()
        
        # Log the output directory
        self.logger.info(f"Processing for gauge {self.gauge_id}")
        self.logger.info(f"Model type: {self.model_type}")
        self.logger.info(f"Coupled mode: {self.coupled}")
        self.logger.info(f"Model directory: {self.model_dir}")
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

    def _find_monthly_files(self) -> Dict[str, List[Path]]:
        """
        Find all monthly files in the ERA5-Land data directory, including geopotential

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
            # Temperature files
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

            # Precipitation files
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

            # PET files
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

    def _combine_monthly_files(self, file_list: List[Path], variable_type: str) -> Optional[xr.Dataset]:
        """
        Combine monthly files into a single dataset, with better error handling
        
        Parameters
        ----------
        file_list : List[Path]
            List of monthly NetCDF files
        variable_type : str
            'temperature', 'precipitation', or 'potential_evaporation'
            
        Returns
        -------
        Optional[xr.Dataset]
            Combined dataset or None if failed
        """
        if not file_list:
            self.logger.warning(f"No files to combine for {variable_type}")
            return None
        
        self.logger.info(f"Combining {len(file_list)} {variable_type} files...")
        
        try:
            # Sort files to ensure chronological order
            sorted_files = sorted(file_list)
            
            # ✅ IMPROVED: First check which files are valid
            valid_files = []
            invalid_files = []
            
            for file_path in sorted_files:
                try:
                    # Quick test to see if file can be opened
                    with xr.open_dataset(file_path, engine='netcdf4') as test_ds:
                        # Check if it has the expected dimensions
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
            
            # ✅ COMPLETELY REWRITTEN: Use xarray's built-in concatenation with proper time handling
            if len(valid_files) > 50:  # For large numbers of files
                self.logger.info(f"Large file count ({len(valid_files)}) - using batch processing with proper time concatenation")
                
                # Process in smaller batches but with proper time coordinate handling
                batch_size = 12  # Monthly files, so 12 = 1 year batches
                datasets = []
                
                for i in range(0, len(valid_files), batch_size):
                    batch_files = valid_files[i:i+batch_size]
                    batch_num = i//batch_size + 1
                    total_batches = (len(valid_files)-1)//batch_size + 1
                    
                    self.logger.debug(f"Processing batch {batch_num}/{total_batches}: {len(batch_files)} files")
                    
                    try:
                        # Open and immediately combine this batch
                        batch_datasets = []
                        for file_path in batch_files:
                            ds = xr.open_dataset(file_path, engine='netcdf4')
                            
                            # Standardize time coordinate name immediately
                            time_coord = None
                            for coord in ['time', 'valid_time', 'datetime']:
                                if coord in ds.dims:
                                    time_coord = coord
                                    break
                            
                            if time_coord and time_coord != 'time':
                                ds = ds.rename({time_coord: 'time'})
                            
                            # Ensure time is properly decoded
                            if 'time' in ds.coords:
                                if not pd.api.types.is_datetime64_any_dtype(ds.time):
                                    # Try to decode time coordinate
                                    try:
                                        ds = xr.decode_cf(ds)
                                    except:
                                        self.logger.warning(f"Could not decode time for {file_path.name}")
                            
                            batch_datasets.append(ds)
                        
                        # Concatenate this batch along time dimension
                        if batch_datasets:
                            batch_combined = xr.concat(batch_datasets, dim='time', combine_attrs='drop_conflicts')
                            
                            # Sort by time to ensure chronological order
                            batch_combined = batch_combined.sortby('time')
                            
                            # Load into memory to free file handles
                            batch_combined = batch_combined.load()
                            datasets.append(batch_combined)
                            
                            # Close individual datasets
                            for ds in batch_datasets:
                                ds.close()
                        
                    except Exception as e:
                        self.logger.error(f"Error processing batch {batch_num}: {str(e)}")
                        continue
                
                if not datasets:
                    self.logger.error(f"No datasets could be processed for {variable_type}")
                    return None
                
                # Combine all batches
                self.logger.info(f"Combining {len(datasets)} batches...")
                ds = xr.concat(datasets, dim='time', combine_attrs='drop_conflicts')
                
                # Close batch datasets to free memory
                for batch_ds in datasets:
                    batch_ds.close()
                    
            else:
                # ✅ SIMPLIFIED: Direct approach for smaller file counts with proper time handling
                self.logger.info(f"Processing {len(valid_files)} files directly")
                
                # Open all files and standardize time coordinates
                datasets = []
                for file_path in valid_files:
                    ds = xr.open_dataset(file_path, engine='netcdf4')
                    
                    # Standardize time coordinate name
                    time_coord = None
                    for coord in ['time', 'valid_time', 'datetime']:
                        if coord in ds.dims:
                            time_coord = coord
                            break
                    
                    if time_coord and time_coord != 'time':
                        ds = ds.rename({time_coord: 'time'})
                    
                    # Ensure time is properly decoded
                    if 'time' in ds.coords:
                        if not pd.api.types.is_datetime64_any_dtype(ds.time):
                            try:
                                ds = xr.decode_cf(ds)
                            except:
                                self.logger.warning(f"Could not decode time for {file_path.name}")
                    
                    datasets.append(ds)
                
                # Concatenate all datasets
                ds = xr.concat(datasets, dim='time', combine_attrs='drop_conflicts')
                
                # Close individual datasets
                for dataset in datasets:
                    dataset.close()
            
            # ✅ FINAL TIME COORDINATE VERIFICATION AND CLEANUP
            if 'time' not in ds.dims:
                self.logger.error(f"No time coordinate found in combined {variable_type} dataset")
                self.logger.debug(f"Available dimensions: {list(ds.dims)}")
                self.logger.debug(f"Available coordinates: {list(ds.coords)}")
                return None
            
            # Ensure time is properly sorted
            ds = ds.sortby('time')
            
            # Verify we have a proper datetime index
            if not pd.api.types.is_datetime64_any_dtype(ds.time):
                self.logger.error(f"Time coordinate is not datetime type: {ds.time.dtype}")
                # Try one more time to decode
                try:
                    ds = xr.decode_cf(ds)
                    self.logger.debug("Successfully decoded time coordinate")
                except Exception as e:
                    self.logger.error(f"Failed to decode time coordinate: {e}")
                    return None
            
            self.logger.info(f"Combined {variable_type} dataset shape: {dict(ds.dims)}")
            self.logger.info(f"Time range: {ds.time.min().values} to {ds.time.max().values}")
            self.logger.info(f"Time coordinate type: {ds.time.dtype}")
            
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
        # The time coordinate should already be renamed to 'time' by _combine_monthly_files
        if 'time' not in dataset.dims:
            self.logger.warning("No 'time' coordinate found in dataset")
            return dataset
        
        try:
            # Convert start and end dates to the same format as the dataset time coordinate
            start_date_str = self.start_date.strftime('%Y-%m-%d')
            end_date_str = self.end_date.strftime('%Y-%m-%d')
            
            self.logger.debug(f"Filtering from {start_date_str} to {end_date_str}")
            
            # Filter to exact date range using string dates
            filtered_ds = dataset.sel(time=slice(start_date_str, end_date_str))
            
            self.logger.info(f"Filtered to exact range: {self.start_date.date()} to {self.end_date.date()}")
            self.logger.info(f"Filtered time range: {filtered_ds.time.min().values} to {filtered_ds.time.max().values}")
            self.logger.info(f"Filtered dataset shape: {dict(filtered_ds.dims)}")
            
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

    def plot_elevation_with_cell_numbers(self, elevation_file: Optional[Path] = None) -> None:
        """
        Plot elevation data using cell numbers - shows single elevation value per cell
        
        Parameters
        ----------
        elevation_file : Optional[Path]
            Path to elevation file. If None, looks for era5_land_elevation.nc in output directory
        """
        if elevation_file is None:
            elevation_file = self.output_path / 'era5_land_elevation.nc'
        
        if not elevation_file.exists():
            self.logger.error(f"❌ Elevation file not found: {elevation_file}")
            return
        
        self.logger.info(f"🗺️ Plotting elevation data with cell numbers: {elevation_file.name}")
        
        try:
            # Load elevation data
            ds = xr.open_dataset(elevation_file)
            elevation = ds['elevation']
            
            self.logger.info(f"📊 Elevation data info:")
            self.logger.info(f"  Shape: {elevation.shape}")
            self.logger.info(f"  Dimensions: {elevation.dims}")
            self.logger.info(f"  Coordinates: {list(elevation.coords)}")
            
            # Get coordinate information
            if 'latitude' in elevation.coords and 'longitude' in elevation.coords:
                lat_coords = elevation.coords['latitude'].values
                lon_coords = elevation.coords['longitude'].values
                lat_name, lon_name = 'latitude', 'longitude'
            elif 'lat' in elevation.coords and 'lon' in elevation.coords:
                lat_coords = elevation.coords['lat'].values
                lon_coords = elevation.coords['lon'].values
                lat_name, lon_name = 'lat', 'lon'
            else:
                self.logger.error("❌ Cannot find latitude/longitude coordinates")
                return
            
            self.logger.info(f"  Latitude range: {lat_coords.min():.4f} to {lat_coords.max():.4f}")
            self.logger.info(f"  Longitude range: {lon_coords.min():.4f} to {lon_coords.max():.4f}")
            self.logger.info(f"  Number of lat cells: {len(lat_coords)}")
            self.logger.info(f"  Number of lon cells: {len(lon_coords)}")
            
            # Get elevation values
            elev_values = elevation.values
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot 1: Elevation with cell numbers as coordinates
            ax1 = axes[0]
            
            # Create cell number arrays
            lon_cells = np.arange(len(lon_coords))  # X-axis: longitude cell numbers
            lat_cells = np.arange(len(lat_coords))  # Y-axis: latitude cell numbers
            
            # 🔧 SIMPLE APPROACH: Use pcolormesh for clean cell-by-cell display
            im1 = ax1.pcolormesh(lon_cells, lat_cells, elev_values, 
                                cmap='terrain', shading='nearest')
            
            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('Elevation (m)', rotation=270, labelpad=20)
            
            # Customize plot 1
            ax1.set_title(f'ERA5-Land Elevation - Cell Numbers\nGauge {self.gauge_id}', 
                        fontsize=12, fontweight='bold')
            ax1.set_xlabel('Longitude Cell Number')
            ax1.set_ylabel('Latitude Cell Number')
            ax1.grid(True, alpha=0.3)
            
            # Add cell number annotations on key points
            rows, cols = elev_values.shape
            for i in range(0, rows, max(1, rows//5)):  # Show ~5 labels per axis
                for j in range(0, cols, max(1, cols//5)):
                    if not np.isnan(elev_values[i, j]):
                        ax1.annotate(f'({j},{i})\n{elev_values[i,j]:.0f}m', 
                                (j, i), ha='center', va='center',
                                fontsize=7, alpha=0.8,
                                bbox=dict(boxstyle='round,pad=0.2', 
                                        facecolor='white', alpha=0.7))
            
            # Plot 2: Same data with geographic coordinates
            ax2 = axes[1]
            
            # Use pcolormesh for geographic coordinates too
            im2 = ax2.pcolormesh(lon_coords, lat_coords, elev_values,
                                cmap='terrain', shading='nearest')
            
            # Add colorbar
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
            cbar2.set_label('Elevation (m)', rotation=270, labelpad=20)
            
            # Customize plot 2
            ax2.set_title(f'ERA5-Land Elevation - Geographic Coordinates\nGauge {self.gauge_id}', 
                        fontsize=12, fontweight='bold')
            ax2.set_xlabel('Longitude (°)')
            ax2.set_ylabel('Latitude (°)')
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')
            
            # Add statistics text box
            stats_text = f'''Elevation Stats:
    Min: {np.nanmin(elev_values):.0f} m
    Max: {np.nanmax(elev_values):.0f} m
    Mean: {np.nanmean(elev_values):.0f} m
    Grid: {len(lon_coords)} × {len(lat_coords)} cells
    Type: Single value per cell'''
            
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            save_path = self.plots_dir / f'era5_elevation_cell_values_gauge_{self.gauge_id}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📊 Elevation plot saved: {save_path}")
            
            plt.show()
            
            # Close dataset
            ds.close()
            
        except Exception as e:
            self.logger.error(f"❌ Error plotting elevation: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")

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

    def _save_daily_files(self, daily_datasets: Dict[str, xr.Dataset]) -> List[Path]:
        """
        Save daily aggregated datasets to NetCDF files with correct coordinate orientation and elevation data

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

        # 🏔️ Load elevation data once (if it exists)
        elevation_dataset = None
        elevation_file = self.output_path / 'era5_land_elevation.nc'

        if elevation_file.exists():
            try:
                self.logger.info(f"📍 Loading elevation data from: {elevation_file.name}")
                elevation_dataset = xr.open_dataset(elevation_file)

                # Verify elevation data structure
                if 'elevation' in elevation_dataset.data_vars:
                    elev_shape = elevation_dataset['elevation'].shape
                    elev_dims = elevation_dataset['elevation'].dims
                    self.logger.info(f"   Elevation shape: {elev_shape}, dimensions: {elev_dims}")

                    # 🔧 CRITICAL FIX: Apply coordinate flipping to elevation data too!
                    self.logger.debug("Applying coordinate flipping to elevation data...")
                    elevation_corrected = self._check_and_fix_coordinate_flipping(elevation_dataset)
                    elevation_dataset.close()  # Close original
                    elevation_dataset = elevation_corrected
                    self.logger.debug("✅ Elevation data coordinate flipping applied")

                else:
                    self.logger.warning("❌ No 'elevation' variable found in elevation file")
                    elevation_dataset = None

            except Exception as e:
                self.logger.error(f"❌ Error loading elevation file: {e}")
                elevation_dataset = None
        else:
            self.logger.debug("📍 No elevation file found - saving without elevation data")

        for var_name, dataset in daily_datasets.items():
            try:
                self.logger.debug(f"Saving {var_name} dataset...")

                # 🔧 FIRST: Apply coordinate flipping to meteorological data
                self.logger.debug(f"Checking coordinate orientation for {var_name} before saving...")
                dataset_corrected = self._check_and_fix_coordinate_flipping(dataset)

                # 🏔️ ADD ELEVATION: Now both datasets should have matching coordinates
                elevation_added = False
                if elevation_dataset is not None:
                    try:
                        self.logger.debug(f"Adding elevation data to {var_name}...")

                        # Get coordinates from both datasets
                        dataset_lat = dataset_corrected.coords.get('latitude', dataset_corrected.coords.get('lat'))
                        dataset_lon = dataset_corrected.coords.get('longitude', dataset_corrected.coords.get('lon'))
                        elev_lat = elevation_dataset.coords.get('latitude', elevation_dataset.coords.get('lat'))
                        elev_lon = elevation_dataset.coords.get('longitude', elevation_dataset.coords.get('lon'))

                        if (dataset_lat is not None and dataset_lon is not None and
                            elev_lat is not None and elev_lon is not None):

                            # 🔧 DEBUG: Log coordinate values for comparison
                            self.logger.debug(f"Meteorological data coordinates (after flip):")
                            self.logger.debug(f"  Lat: {dataset_lat.values[0]:.6f} to {dataset_lat.values[-1]:.6f} ({len(dataset_lat)} points)")
                            self.logger.debug(f"  Lon: {dataset_lon.values[0]:.6f} to {dataset_lon.values[-1]:.6f} ({len(dataset_lon)} points)")

                            self.logger.debug(f"Elevation data coordinates (after flip):")
                            self.logger.debug(f"  Lat: {elev_lat.values[0]:.6f} to {elev_lat.values[-1]:.6f} ({len(elev_lat)} points)")
                            self.logger.debug(f"  Lon: {elev_lon.values[0]:.6f} to {elev_lon.values[-1]:.6f} ({len(elev_lon)} points)")

                            # Check if coordinates match with reasonable tolerance
                            lat_match = (len(dataset_lat) == len(elev_lat)) and np.allclose(dataset_lat.values, elev_lat.values, rtol=1e-8, atol=1e-8)
                            lon_match = (len(dataset_lon) == len(elev_lon)) and np.allclose(dataset_lon.values, elev_lon.values, rtol=1e-8, atol=1e-8)

                            if lat_match and lon_match:
                                # Add elevation variable to the dataset
                                dataset_corrected['elevation'] = elevation_dataset['elevation']
                                elevation_added = True

                                self.logger.debug(f"✅ Successfully added elevation to {var_name}")
                                self.logger.debug(f"   Final dataset variables: {list(dataset_corrected.data_vars)}")

                            else:
                                # 🔧 INTERPOLATE elevation if shapes do not match
                                self.logger.warning("⚠️ Elevation grid does not match meteo grid - interpolating elevation data")
                                try:
                                    elev_interp = elevation_dataset['elevation'].interp(
                                        latitude=dataset_lat,
                                        longitude=dataset_lon
                                    )
                                    dataset_corrected['elevation'] = elev_interp
                                    elevation_added = True
                                    self.logger.info("✅ Successfully interpolated elevation to meteo grid")
                                except Exception as e:
                                    self.logger.error(f"❌ Error interpolating elevation: {e}")
                                    elevation_added = False

                        else:
                            self.logger.warning(f"⚠️ Cannot find matching coordinates - skipping elevation for {var_name}")

                    except Exception as e:
                        self.logger.error(f"❌ Error adding elevation to {var_name}: {e}")
                        import traceback
                        self.logger.debug(f"Full traceback: {traceback.format_exc()}")

                # 🔧 CRITICAL FIX: Ensure we actually have data variables before saving
                if len(list(dataset_corrected.data_vars)) == 0:
                    self.logger.error(f"❌ Dataset {var_name} has no data variables! Skipping save.")
                    continue

                # Simplified filename without date range
                filename = f"era5_land_{var_name}.nc"
                output_file_path = self.output_path / filename

                # 🔧 FIX: Convert boolean to string for NetCDF compatibility
                elevation_status = "true" if elevation_added else "false"  # Convert boolean to string

                # Add metadata with proper data types
                dataset_corrected.attrs.update({
                    'title': f'ERA5-Land {var_name} daily data',
                    'gauge_id': str(self.gauge_id),  # Ensure string type
                    'time_range': f"{self.start_date.date()} to {self.end_date.date()}",
                    'processed_by': 'ERA5LandAnalyzer',
                    'creation_date': pd.Timestamp.now().isoformat(),
                    'model_type': str(self.model_type),  # Ensure string type
                    'catchment': f'catchment_{self.gauge_id}',
                    'coordinate_orientation': 'Corrected for consistent north-up orientation',
                    'elevation_included': elevation_status  # Use string instead of boolean
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

        # Close elevation dataset if it was loaded
        if elevation_dataset is not None:
            elevation_dataset.close()
            self.logger.debug("Closed elevation dataset")

        return saved_files

    #---------------------------------------------------------------------------------

    def _find_and_process_monthly_files(self) -> List[Path]:
        """
        Main method to find monthly files, combine them, filter time range, and aggregate to daily
        INCLUDING geopotential processing for elevation data
        
        Returns
        -------
        List[Path]
            List of processed daily files
        """
        self.logger.info("Starting monthly file processing pipeline...")
        
        # Step 1: Find monthly files
        monthly_files = self._find_monthly_files()
        
        if not any(monthly_files.values()):
            self.logger.error("No monthly files found!")
            return []
        
        # 🏔️ Step 2: Process geopotential to elevation FIRST (before meteorological processing)
        self.logger.info("🏔️ Processing geopotential to elevation...")
        elevation_file = self.process_geopotential_to_elevation()
        
        if elevation_file is not None:
            self.logger.info(f"✅ Elevation processing successful: {elevation_file.name}")
        else:
            self.logger.warning("⚠️ Elevation processing failed - meteorological files will be saved without elevation data")
        
        all_daily_files = []
        
        # Step 3: Process temperature files
        if monthly_files['temperature']:
            self.logger.info("Processing temperature files...")
            
            # Combine monthly files
            temp_combined = self._combine_monthly_files(monthly_files['temperature'], 'temperature')
            
            if temp_combined is not None:
                # Filter to exact time range
                temp_filtered = self._filter_time_range_exact(temp_combined)
                
                # Aggregate to daily
                temp_daily = self._aggregate_to_daily(temp_filtered, 'temperature')
                
                # Save daily files (elevation will be added automatically if available)
                if temp_daily:
                    temp_files = self._save_daily_files(temp_daily)
                    all_daily_files.extend(temp_files)
                
                # Close datasets
                temp_combined.close()
                temp_filtered.close()
        
        # Step 4: Process precipitation files
        if monthly_files['precipitation']:
            self.logger.info("Processing precipitation files...")
            
            # Combine monthly files
            precip_combined = self._combine_monthly_files(monthly_files['precipitation'], 'precipitation')
            
            if precip_combined is not None:
                # Filter to exact time range
                precip_filtered = self._filter_time_range_exact(precip_combined)
                
                # Aggregate to daily
                precip_daily = self._aggregate_to_daily(precip_filtered, 'precipitation')
                
                # Save daily files (elevation will be added automatically if available)
                if precip_daily:
                    precip_files = self._save_daily_files(precip_daily)
                    all_daily_files.extend(precip_files)
                
                # Close datasets
                precip_combined.close()
                precip_filtered.close()
        
        # Step 5: Process PET files
        if monthly_files['potential_evaporation']:
            self.logger.info("Processing potential evaporation files...")
            
            # Combine monthly files
            pet_combined = self._combine_monthly_files(monthly_files['potential_evaporation'], 'potential_evaporation')
            
            if pet_combined is not None:
                # Filter to exact time range
                pet_filtered = self._filter_time_range_exact(pet_combined)
                
                # Aggregate to daily
                pet_daily = self._aggregate_to_daily(pet_filtered, 'potential_evaporation')
                
                # Save daily files (elevation will be added automatically if available)
                if pet_daily:
                    pet_files = self._save_daily_files(pet_daily)
                    all_daily_files.extend(pet_files)
                
                # Close datasets
                pet_combined.close()
                pet_filtered.close()
        
        self.logger.info(f"Processing pipeline complete! Created {len(all_daily_files)} daily files:")
        for file_path in all_daily_files:
            self.logger.info(f"  - {file_path.name}")
        
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
    
    def _filter_time_range(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Filter dataset to the specified time range from namelist
        
        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset
            
        Returns
        -------
        xr.Dataset
            Filtered dataset
        """
        # Handle different time coordinate names
        time_coord = None
        for coord in ['time', 'valid_time', 'datetime']:
            if coord in dataset.dims:
                time_coord = coord
                break
        
        if time_coord is None:
            self.logger.warning("No time coordinate found in dataset")
            return dataset
        
        # Rename to standard 'time' if needed
        if time_coord != 'time':
            dataset = dataset.rename({time_coord: 'time'})
        
        # Filter to specified date range
        filtered_ds = dataset.sel(time=slice(self.start_date, self.end_date))
        
        self.logger.info(f"Filtered dataset from {self.start_date} to {self.end_date}")
        self.logger.info(f"Original time range: {dataset.time.min().values} to {dataset.time.max().values}")
        self.logger.info(f"Filtered time range: {filtered_ds.time.min().values} to {filtered_ds.time.max().values}")
        
        return filtered_ds
    
    #---------------------------------------------------------------------------------
    
    def process_and_save_netcdf(self, netcdf_file: Path) -> Optional[Path]:
        """
        Process NetCDF file (filter time range, convert units) and save
        
        Parameters
        ----------
        netcdf_file : Path
            Path to the NetCDF file to process
            
        Returns
        -------
        Path or None
            Path to processed file if successful, None otherwise
        """
        self.logger.info(f"Processing file: {netcdf_file.name}")
        
        try:
            # Open the dataset
            ds = xr.open_dataset(netcdf_file, chunks={'time': 100})
            
            # Filter time range
            ds_filtered = self._filter_time_range(ds)
            
            # Get data variables and convert units
            data_vars = [var for var in ds_filtered.data_vars if len(ds_filtered[var].dims) >= 2]
            
            for var_name in data_vars:
                ds_filtered = self._convert_units(ds_filtered, var_name)
            
            # Create output filename
            output_filename = f"processed_{self.gauge_id}_{netcdf_file.stem}_{self.start_date.strftime('%Y')}_{self.end_date.strftime('%Y')}.nc"
            output_path = self.processed_data_dir / output_filename
            
            # Save processed dataset
            ds_filtered.to_netcdf(output_path)
            self.logger.info(f"Processed file saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error processing {netcdf_file.name}: {str(e)}")
            return None
    
    #---------------------------------------------------------------------------------
    
    def plot_spatial_overview(self, netcdf_file: Path) -> None:
        """
        Plot spatial overview of the first timestep for each variable in the NetCDF file
        
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
            ds = self._filter_time_range(ds)
            
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
            
            # Get data variables (exclude coordinates)
            data_vars = [var for var in ds.data_vars if len(ds[var].dims) >= 2]
            
            if not data_vars:
                self.logger.warning(f"No suitable variables found in {netcdf_file.name}")
                return
            
            # Create subplots for each variable
            n_vars = len(data_vars)
            cols = min(3, n_vars)
            rows = (n_vars + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if n_vars == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            fig.suptitle(f'Spatial Overview - Gauge {self.gauge_id}\n{netcdf_file.stem} - First Timestep\nDate Range: {self.start_date.strftime("%Y-%m-%d")} to {self.end_date.strftime("%Y-%m-%d")}', 
                        fontsize=14, fontweight='bold')
            
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
                        self.logger.debug(f"Plotting catchment boundary on subplot {i+1}")
                        catchment_shape.boundary.plot(ax=ax, color='red', linewidth=3, alpha=0.9, label='Catchment boundary')
                        
                        # Also plot the filled area with transparency
                        catchment_shape.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=3, alpha=0.9)
                        
                        # Add legend
                        ax.legend(loc='upper right', fontsize=10, framealpha=0.8)
                        self.logger.debug("✅ Catchment boundary plotted successfully")
                        
                    except Exception as e:
                        self.logger.error(f"Error plotting catchment boundary: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Add colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_label(f"{var_info['units']}", rotation=270, labelpad=15)
                
                # Customize plot
                ax.set_title(f"{var_info['name']}")
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Set aspect ratio to equal for proper geographic display
                ax.set_aspect('equal')
            
            # Hide unused subplots
            for i in range(n_vars, len(axes)):
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
            
            # Filter time range
            ds = self._filter_time_range(ds)
            
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
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating spatial averages for {netcdf_file.name}: {str(e)}")
            return pd.DataFrame()
    
    #---------------------------------------------------------------------------------
    
    def plot_timeseries(self, netcdf_file: Path, df_timeseries: pd.DataFrame) -> None:
        """
        Plot time series of spatially averaged variables
        
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
            
            fig.suptitle(f'Time Series - Gauge {self.gauge_id}\n{netcdf_file.stem} - Spatial Averages\nPeriod: {self.start_date.strftime("%Y-%m-%d")} to {self.end_date.strftime("%Y-%m-%d")}', 
                        fontsize=14, fontweight='bold')
            
            for i, var_name in enumerate(df_timeseries.columns):
                ax = axes[i]
                
                # Get variable info
                var_info = self.era5_variables.get(var_name, {
                    'name': var_name,
                    'units': '',
                    'cmap': 'viridis'
                })
                
                # Plot the time series
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
                
                # Add statistics text box
                mean_val = df_timeseries[var_name].mean()
                std_val = df_timeseries[var_name].std()
                min_val = df_timeseries[var_name].min()
                max_val = df_timeseries[var_name].max()
                
                stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
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
            
            # Calculate spatial average over catchment area
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
            
            # Calculate spatial average over catchment area
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
############################### Main execution ################################
#--------------------------------------------------------------------------------

def main():
    """Example usage of the ERA5LandAnalyzer"""
    
    # Path to namelist configuration file
    namelist_path = '/home/jberg/OneDrive/Raven-world/namelist.yaml'
    
    # Create analyzer and run processing & analysis
    analyzer = ERA5LandAnalyzer(namelist_path)
    
    # Get info about processed files
    file_info = analyzer.get_processed_files_info()
    print("\n📁 Processed Files:")
    for var_name, file_path in file_info.items():
        print(f"  {var_name}: {file_path.name}")
    
    # Run analysis
    analyzer.analyze_all_files()

if __name__ == "__main__":
    main()



#--------------------------------------------------------------------------------
############################# GridWeights Generator #############################
#--------------------------------------------------------------------------------

class GridWeightsGenerator:
    """
    A class for generating grid weights for ERA5-Land meteorological data preprocessing.
    Converts ERA5-Land netCDF data to polygon grids and calculates weights
    for each HRU (Hydrological Response Unit).
    """
    
    def __init__(self, config: Union[Dict[str, Any], str, Path]) -> None:
        """
        Initialize the GridWeightsGenerator
        
        Parameters
        ----------
        config : Dict[str, Any] or str or Path
            Configuration dictionary OR path to namelist YAML file with the following keys:
            - model_dir: Directory for model outputs (constructed automatically if using namelist)
            - gauge_id: ID of the catchment gauge
            - model_type: Type of hydrological model
            - cell_size: (optional) Size of the grid cell (square) in meters
            - x_size: (optional) Width of the grid cell in meters
            - y_size: (optional) Height of the grid cell in meters
            - debug: (optional) Whether to enable detailed logging and plotting
        """
        # Load configuration from namelist if path provided
        if isinstance(config, (str, Path)):
            with open(config, 'r') as f:
                namelist_config = yaml.safe_load(f)
            
            # Extract parameters from namelist
            self.gauge_id = namelist_config['gauge_id']
            main_dir = Path(namelist_config['main_dir'])
            coupled = namelist_config.get('coupled', False)
            self.model_dir = main_dir / namelist_config['config_dir'] / f'catchment_{self.gauge_id}'
            self.model_type = namelist_config['model_type']
            
            # Optional parameters from namelist or defaults
            self.cell_size = namelist_config.get('cell_size')
            self.x_size = namelist_config.get('x_size')
            self.y_size = namelist_config.get('y_size')
            self.debug = namelist_config.get('debug', False)
            
        else:
            # Use config dictionary (backward compatibility)
            self.model_dir = Path(config['model_dir'])
            self.gauge_id = config['gauge_id']
            self.model_type = config['model_type']
            self.cell_size = config.get('cell_size')
            self.x_size = config.get('x_size')
            self.y_size = config.get('y_size')
            self.debug = config.get('debug', False)
        
        # Define output directory paths
        self.out_dir = self.model_dir / self.model_type / 'data_obs'
        self.out_HRU_shape_dir = self.model_dir / 'topo_files' / 'HRU.shp'
        self.plots_dir = self.model_dir / self.model_type / 'plots'
        
        # Create directories if they don't exist
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
        
        # Log configuration
        self.logger.debug(f"Initialized GridWeightsGenerator with:")
        self.logger.debug(f"  Model dir: {self.model_dir}")
        self.logger.debug(f"  Gauge ID: {self.gauge_id}")
        self.logger.debug(f"  Model type: {self.model_type}")
        self.logger.debug(f"  Output dir: {self.out_dir}")
        self.logger.debug(f"  HRU shapefile: {self.out_HRU_shape_dir}")

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
        Create a polygon grid from ERA5-Land point data
        
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame with point geometries in WGS84
            
        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with polygon geometries
        """
        self.logger.debug("Creating polygon grid from ERA5-Land point data")
        
        # From the GeoDataFrame that contains the netCDF grid, get the extent as points
        xmin, ymin, xmax, ymax = gdf.total_bounds
        self.logger.debug(f"Grid bounds (WGS84): xmin={xmin:.6f}, xmax={xmax:.6f}, ymin={ymin:.6f}, ymax={ymax:.6f}")
        
        # For ERA5-Land, infer cell size from the data points
        self.logger.debug("Inferring cell size from ERA5-Land grid spacing")
        
        # Sort points by coordinates to find adjacent points
        sorted_x = sorted(gdf.geometry.x.unique())
        sorted_y = sorted(gdf.geometry.y.unique())
        
        self.logger.debug(f"Number of unique X coordinates: {len(sorted_x)}")
        self.logger.debug(f"Number of unique Y coordinates: {len(sorted_y)}")
        
        # Calculate distances between adjacent points (in degrees)
        x_diffs = [sorted_x[i+1] - sorted_x[i] for i in range(len(sorted_x)-1)]
        y_diffs = [sorted_y[i+1] - sorted_y[i] for i in range(len(sorted_y)-1)]
        
        if x_diffs and y_diffs:
            width = np.median(x_diffs)
            height = np.median(y_diffs)
            self.logger.debug(f"Inferred ERA5-Land cell size: width={width:.6f}°, height={height:.6f}°")
        else:
            # Default ERA5-Land resolution is 0.1° x 0.1°
            width = 0.1
            height = 0.1
            self.logger.warning(f"Could not infer cell size, using default ERA5-Land resolution: {width}° x {height}°")
        
        # FIX: Create boundaries based on actual data points, not extended bounds
        # This ensures we only create polygons for actual data cells
        cols = []
        rows = []
        
        # Create column boundaries (X direction)
        for i, x in enumerate(sorted_x):
            if i == 0:
                # First column: start at x - width/2
                cols.append(x - width/2)
            # Add the right boundary of current cell
            cols.append(x + width/2)
        
        # Create row boundaries (Y direction) 
        for i, y in enumerate(sorted_y):
            if i == 0:
                # First row: start at y - height/2
                rows.append(y - height/2)
            # Add the top boundary of current cell
            rows.append(y + height/2)
        
        self.logger.debug(f"Creating grid with {len(cols)-1} columns and {len(rows)-1} rows")
        self.logger.debug(f"Expected total cells: {(len(cols)-1) * (len(rows)-1)}")
        self.logger.debug(f"Actual data points: {len(gdf)}")
        
        # Verify the grid matches the data
        expected_cells = (len(cols)-1) * (len(rows)-1)
        if expected_cells != len(gdf):
            self.logger.warning(f"Grid size mismatch: expected {expected_cells} cells, have {len(gdf)} data points")
        
        # initialize the Polygon list
        polygons = []
        cell_id = []
        
        # Create the GeoDataFrame with the grid
        grid_cols = ['row', 'col', 'cell_id', 'polygons', 'area', 'area_rel']
        grid = gpd.GeoDataFrame(columns=grid_cols, geometry='polygons')
        grid.set_crs(epsg='4326')  # ERA5-Land is in WGS84
        
        # Loop over each cell and create the corresponding Polygon
        # Use len(cols)-1 and len(rows)-1 to avoid creating extra cells
        for ix in range(len(cols)-1):
            for iy in range(len(rows)-1):
                x = cols[ix]
                y = rows[iy]
                x_next = cols[ix+1]
                y_next = rows[iy+1]
                
                polygons.append(Polygon([(x, y), (x_next, y), (x_next, y_next), (x, y_next)]))
                cid = int(iy * (len(cols)-1) + ix)
                cell_id.append(f"{str(cid)}")
        
        # Use the polygon list in the GeoDataFrame
        grid["polygons"] = polygons
        grid["cell_id"] = cell_id
        grid["area_rel"] = 0
        grid = grid.set_crs('EPSG:4326')  # WGS84 for ERA5-Land
        
        self.logger.info(f"Created grid with {len(grid)} cells")
        self.logger.info(f"This should match the number of data points: {len(gdf)}")
        
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
        Calculate the relative area of each HRU in each ERA5-Land grid cell
        
        Parameters
        ----------
        overlay : gpd.GeoDataFrame
            Overlay of grid and HRU shapes
        hru : gpd.GeoDataFrame
            HRU polygons
            
        Returns
        -------
        gpd.GeoDataFrame
            Overlay with calculated relative areas
        """
        self.logger.debug("Calculating relative areas for ERA5-Land grid cells")
        
        column_names_list = list(overlay.columns)
        result_gdf = gpd.GeoDataFrame(columns=column_names_list)
        
        # Get HRU ID column name (might be 'HRU_ID' or 'HRU ID')
        hru_id_col = None
        for col in overlay.columns:
            if 'hru' in col.lower() and 'id' in col.lower():
                hru_id_col = col
                break
        
        if hru_id_col is None:
            raise ValueError("Could not find HRU ID column in overlay data")
        
        self.logger.debug(f"Using HRU ID column: {hru_id_col}")
        
        # Get unique HRU IDs
        unique_hrus = sorted(overlay[hru_id_col].unique())
        self.logger.debug(f"Processing {len(unique_hrus)} HRUs: {unique_hrus}")
        
        # calculate relative area for each HRU
        for hru_id in unique_hrus:
            self.logger.debug(f"Processing HRU {hru_id}")
            
            # extract Geodataframe from each HRU
            mask = (overlay[hru_id_col] == hru_id)
            filtered_gdf = overlay[mask].copy()
        
            # calculate area in square meters (project to appropriate UTM if needed)
            if filtered_gdf.crs.to_string() == 'EPSG:4326':
                # For WGS84, use equal area projection for accurate area calculation
                # Use a general equal area projection (World Mollweide)
                filtered_gdf_proj = filtered_gdf.to_crs('ESRI:54009')
                filtered_gdf["area"] = filtered_gdf_proj.geometry.area
            else:
                filtered_gdf["area"] = filtered_gdf.geometry.area
                
            area_sum = float(filtered_gdf['area'].sum())
            
            if area_sum > 0:
                for index, row in filtered_gdf.iterrows():
                    # Calculate relative area
                    filtered_gdf.at[index, 'area_rel'] = filtered_gdf.loc[index]['area'] / area_sum
            
                filtered_gdf['area_rel'] = filtered_gdf['area_rel'].round(5)
            
                # Normalize the values in the 'relative_area' column
                sum_rel_area = filtered_gdf['area_rel'].sum()
                if sum_rel_area > 0:
                    filtered_gdf['normalized_relative_area'] = (filtered_gdf['area_rel'] / sum_rel_area)
                    filtered_gdf['normalized_relative_area'] = filtered_gdf['normalized_relative_area'].round(5)
                else:
                    filtered_gdf['normalized_relative_area'] = 0
            else:
                filtered_gdf['area_rel'] = 0
                filtered_gdf['normalized_relative_area'] = 0
        
            result_gdf = pd.concat([result_gdf, filtered_gdf], ignore_index=True)
        
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
        
        # 🔧 FIX: Return the relative_area GeoDataFrame
        return relative_area

#--------------------------------------------------------------------------------
# Convenience function for GridWeightsGenerator
#--------------------------------------------------------------------------------

def generate_era5_grid_weights(namelist_path: Union[str, Path], 
                              use_precipitation: bool = True,
                              debug: bool = False) -> gpd.GeoDataFrame:
    """
    Generate ERA5-Land grid weights using namelist configuration
    
    Parameters
    ----------
    namelist_path : str or Path
        Path to namelist YAML configuration file
    use_precipitation : bool, optional
        Whether to use precipitation file (True) or temperature file (False) as grid reference
    debug : bool, optional
        Whether to enable debug mode with plots
        
    Returns
    -------
    gpd.GeoDataFrame
        Relative area calculations
    """
    # Load namelist and add debug parameter
    with open(namelist_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['debug'] = debug
    
    generator = GridWeightsGenerator(config)
    return generator.generate(use_precipitation=use_precipitation)