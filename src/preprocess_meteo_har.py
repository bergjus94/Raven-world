#### This file contains all functions for plotting and analyzing ERA5-Land meteorological data
#### Updated for plotting and time series analysis with namelist configuration
#### Justine Berg

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

from preprocess_meteo_base import MeteoBase, normalize_coords

#--------------------------------------------------------------------------------
############################### HARAnalyzer Class ###############################
#--------------------------------------------------------------------------------

class HARAnalyzer(MeteoBase):
    """
    A class for analyzing and processing HAR (High Asia Refined) meteorological data

    HAR v2 characteristics:
    - 10km resolution (d10km domain)
    - Lambert Conformal Conic projection
    - Daily data in yearly files
    - Variables: t2 (temperature), prcp (precipitation), potevap (potential evaporation)
    """

    _logger_class_name = 'HARAnalyzer'
    _csv_prefix = 'har_'

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
        super().__init__(namelist_path, force_reprocess)

        # HAR-specific: data directory
        har_dir_template = self.config.get('meteo_har_dir', '01_data/meteo/HAR')
        self.har_data_dir = Path(har_dir_template)
        if not self.har_data_dir.is_absolute():
            self.har_data_dir = self.main_dir / self.har_data_dir

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
            
            self.logger.info(f"✅ Combined {variable} dataset: {dict(combined.sizes)}")
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
                    sn_size = ds_meteo.sizes['south_north']
                    we_size = ds_meteo.sizes['west_east']
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
                        sn_size = ds_meteo.sizes[sn_dim]
                        we_size = ds_meteo.sizes[we_dim]
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
############################# HAR GridWeights Generator #########################
#--------------------------------------------------------------------------------

class HARGridWeightsGenerator(MeteoBase):
    """
    A class for generating grid weights for HAR meteorological data.
    Handles the Lambert Conformal Conic projection and curvilinear grid.
    """

    _logger_class_name = 'HARGridWeightsGenerator'

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
        super().__init__(namelist_path, force_reprocess)

        # GridWeights-specific paths
        self.out_dir = self.shared_data_dir
        self.out_HRU_shape_dir = self.model_dir / f'catchment_{self.gauge_id}' / 'topo_files' / 'HRU.shp'

        self.logger.info(f"HAR GridWeights Generator initialized for gauge {self.gauge_id}")

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
            ny = ds.sizes['south_north']
            nx = ds.sizes['west_east']
        else:
            # Fallback to lat/lon shape
            lat_2d = ds.coords['lat'].values
            ny, nx = lat_2d.shape
        
        self.logger.info(f"HAR grid dimensions: {ny} x {nx} = {ny * nx} cells")
        self.total_grid_cells = ny * nx  # full NetCDF size — Raven requires this in :NumberGridCells
        
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
        # Raven validates that :NumberGridCells equals cols*rows from the NetCDF.
        # Use the full grid size recorded during generate(), not the catchment-filtered subset.
        number_cells = getattr(self, 'total_grid_cells', len(grid))
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

