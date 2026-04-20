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
from paths import get_paths
import warnings
warnings.filterwarnings('ignore')

#--------------------------------------------------------------------------------
############################### Helper functions #################################
#--------------------------------------------------------------------------------

def normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize ERA5-Land-style coordinate names and orientation.

    - Renames 'latitude'→'lat' and 'longitude'→'lon' (only when they are dimensions)
    - Flips lat if decreasing (ensures south→north ordering)

    HAR datasets (dims: south_north/west_east) pass through unchanged.
    """
    rename = {}
    if 'latitude' in ds.dims and 'lat' not in ds.dims:
        rename['latitude'] = 'lat'
    if 'longitude' in ds.dims and 'lon' not in ds.dims:
        rename['longitude'] = 'lon'
    if rename:
        ds = ds.rename(rename)
    lat_dim = next((n for n in ('lat', 'latitude') if n in ds.dims), None)
    if lat_dim and ds[lat_dim].size > 1 and float(ds[lat_dim][0]) > float(ds[lat_dim][-1]):
        ds = ds.isel({lat_dim: slice(None, None, -1)})
    return ds


#--------------------------------------------------------------------------------
############################### MeteoBase ########################################
#--------------------------------------------------------------------------------

class MeteoBase:
    """
    Shared base class for all meteorological data analyzers and grid weight generators.
    Handles common __init__ boilerplate: YAML loading, directory setup, logger,
    warm-up date, and catchment shapefile loading.
    """

    _logger_class_name = 'MeteoBase'
    _csv_prefix = ''
    _logger_class_name = 'MeteoBase'

    def __init__(self, namelist_path: Union[str, Path], force_reprocess: bool = False) -> None:
        """
        Initialize common meteorological processor setup.

        Parameters
        ----------
        namelist_path : str or Path
            Path to the namelist YAML configuration file
        force_reprocess : bool, optional
            If True, reprocess files even if they already exist (default: False)
        """
        self.force_reprocess = force_reprocess

        namelist_path = Path(namelist_path)
        if not namelist_path.exists():
            raise FileNotFoundError(f"Namelist file not found: {namelist_path}")

        with open(namelist_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Common configuration parameters
        self.main_dir = Path(self.config.get('main_dir'))
        self.gauge_id = self.config.get('gauge_id')
        self.basin_id = self.config.get('basin_id', self.gauge_id)
        self.start_date = pd.to_datetime(self.config.get('start_date'))
        self.end_date = pd.to_datetime(self.config.get('end_date'))
        self.model_type = self.config.get('model_type')
        self.debug = self.config.get('debug', False)
        self.coupled = self.config.get('coupled', False)

        # Centralized path construction
        paths = get_paths(self.config)
        self.shared_data_dir = paths['data_obs_dir']
        self.shared_plots_dir = paths['plots_dir']

        # Primary output locations (shared)
        self.output_path = self.shared_data_dir
        self.plots_dir = self.shared_plots_dir
        self.spatial_plots_dir = self.plots_dir / 'spatial_overview'
        self.timeseries_plots_dir = self.plots_dir / 'time_series'

        # Backward-compat aliases
        self.processed_data_dir = self.output_path
        self.model_dir = paths['catchment_dir'].parent  # model_runs/
        self.model_data_dir = self.shared_data_dir  # no longer separate

        # Create all directories
        self.spatial_plots_dir.mkdir(parents=True, exist_ok=True)
        self.timeseries_plots_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = self._setup_logger()

        # Warm-up date (from warm_up_date config key, or None)
        if 'warm_up_date' in self.config:
            self.warmup_date = pd.to_datetime(self.config.get('warm_up_date'))
            self.logger.info(f"Warm-up period configured: {self.warmup_date.date()} to {(self.start_date - pd.Timedelta(days=1)).date()}")
        else:
            self.warmup_date = None
            self.logger.debug("No warm-up period configured")

        # Warm-up strategy (see defaults.yaml docstring)
        warmup_cfg = self.config.get('warmup') or {}
        self.warmup_method = warmup_cfg.get('method', 'cycle')
        self.warmup_cycle_years = int(warmup_cfg.get('cycle_years', 1))
        if self.warmup_method not in ('cycle', 'real'):
            raise ValueError(
                f"warmup.method must be 'cycle' or 'real', got '{self.warmup_method}'"
            )
        if self.warmup_cycle_years < 1:
            raise ValueError(
                f"warmup.cycle_years must be >= 1, got {self.warmup_cycle_years}"
            )
        if self.warmup_date is not None:
            self.logger.info(
                f"Warm-up method: {self.warmup_method}"
                f"{f' (cycle_years={self.warmup_cycle_years})' if self.warmup_method == 'cycle' else ''}"
            )

        # Load catchment shapefile for spatial clipping
        self.catchment_extent = self._load_catchment_shapefile()

    #---------------------------------------------------------------------------------

    def _effective_start_date(self) -> pd.Timestamp:
        """
        Date from which source data should be clipped.

        Returns `warm_up_date` for `method='real'` (so analyzers can ingest
        pre-simulation data), otherwise `start_date` (synthetic warm-up will
        be prepended later by `_add_warmup_to_files`).
        """
        if self.warmup_method == 'real' and self.warmup_date is not None:
            return self.warmup_date
        return self.start_date

    #---------------------------------------------------------------------------------

    def _setup_logger(self) -> logging.Logger:
        """Set up and configure logger based on debug flag."""
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            force=True
        )

        # Suppress noisy third-party loggers
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.colorbar').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)

        return logging.getLogger(f'{self._logger_class_name}_Gauge_{self.gauge_id}')

    #---------------------------------------------------------------------------------

    def _load_catchment_shapefile(self) -> Optional[gpd.GeoDataFrame]:
        """
        Load the catchment shapefile for the gauge_id.

        Returns
        -------
        Optional[gpd.GeoDataFrame]
            Catchment shapefile in WGS84 (EPSG:4326) or None if not found
        """
        self.logger.debug(f"Loading catchment shapefile for gauge ID: {self.gauge_id}")

        try:
            shape_dir_template = self.config.get('shape_dir', '01_data/topo/catchment_shapefile/catchment_shape_{gauge_id}.shp')
            shape_path = Path(shape_dir_template.format(gauge_id=self.gauge_id))

            if not shape_path.is_absolute():
                shape_path = self.main_dir / shape_path

            self.logger.debug(f"Looking for catchment shapefile at: {shape_path}")

            if not shape_path.exists():
                self.logger.warning(f"Catchment shapefile not found: {shape_path}")
                self.logger.warning("⚠️ Processing will continue without catchment clipping")
                return None

            extent = gpd.read_file(shape_path)

            self.logger.info(f"✅ Loaded catchment shapefile with {len(extent)} features")
            self.logger.info(f"   Original CRS: {extent.crs}")

            if extent.crs != 'EPSG:4326':
                extent = extent.to_crs('EPSG:4326')
                self.logger.info("   Reprojected to WGS84 (EPSG:4326)")

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

    def _add_warmup_to_files(self, file_list: List[Path]) -> List[Path]:
        """
        Prepend warm-up period to processed meteorological files.

        Strategy controlled by `warmup.method` in the namelist:
          * 'real'  — require the file to already contain data back to
            `warm_up_date`. If not, raise an error (user asked for real
            data, pipeline refuses to silently synthesize).
          * 'cycle' — tile the first `warmup.cycle_years` years of
            simulation data forward (C1 direction) to fill the warm-up
            window. cycle_years=1 reproduces the legacy "repeat first
            year" behavior bit-for-bit.

        Preserves:
          - Elevation as 2D time-invariant
          - Temp-file atomic write
          - Duplicate / non-consecutive time step detection

        Parameters
        ----------
        file_list : List[Path]
            List of processed daily NetCDF files

        Returns
        -------
        List[Path]
            List of files with warm-up period included (or unchanged for
            'real' mode when files already cover the warm-up window).
        """
        self.logger.info("Adding warm-up period to meteorological files...")

        # Cycle-block length in days. The first cycle_years of simulation
        # data is tiled forward to fill the warm-up window.
        cycle_block_end = (
            self.start_date
            + pd.DateOffset(years=self.warmup_cycle_years)
            - pd.Timedelta(days=1)
        )
        if cycle_block_end > self.end_date:
            cycle_block_end = self.end_date

        # How much warm-up data we need
        warmup_days_needed = (self.start_date - self.warmup_date).days

        self.logger.info(f"📅 Warm-up configuration:")
        self.logger.info(f"   Period: {self.warmup_date.date()} to {(self.start_date - pd.Timedelta(days=1)).date()}")
        self.logger.info(f"   Method: {self.warmup_method}"
                         f"{f' (cycle_years={self.warmup_cycle_years})' if self.warmup_method == 'cycle' else ''}")
        
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

                # Check whether the file already covers the warm-up window
                # (either from method='real' with real pre-sim data or from a
                # source file that naturally extends back far enough).
                file_first_time = pd.Timestamp(ds.time.values[0])
                covers_warmup = file_first_time <= self.warmup_date

                if self.warmup_method == 'real':
                    if not covers_warmup:
                        raise ValueError(
                            f"warmup.method='real' but {file_path.name} starts "
                            f"{file_first_time.date()}, after warm_up_date "
                            f"{self.warmup_date.date()}. Extend the forcing record "
                            f"or set warmup.method='cycle'."
                        )
                    # Real data already present — no synthesis needed
                    self.logger.info(
                        f"  ✅ File already covers warm-up window "
                        f"({file_first_time.date()} ≤ {self.warmup_date.date()}) — no synthesis needed"
                    )
                    # Still need to re-attach elevation metadata and save
                    n_warmup_days = int(((self.start_date - self.warmup_date).days))
                    n_sim_days = len(ds.time) - n_warmup_days
                    combined = ds
                else:
                    # method == 'cycle': tile first cycle_years years of sim data
                    cycle_block = ds.sel(time=slice(self.start_date, cycle_block_end))
                    block_days = len(cycle_block.time)

                    if block_days == 0:
                        raise ValueError(
                            f"Cycle block empty for {file_path.name}: no data in "
                            f"[{self.start_date.date()}, {cycle_block_end.date()}]"
                        )

                    expected_block_days = (cycle_block_end - self.start_date).days + 1
                    # Allow some slack for calendar edge cases but refuse if we
                    # lost more than a few % — that indicates the user asked for
                    # more cycle_years than the simulation record actually has.
                    if block_days < 0.95 * expected_block_days:
                        raise ValueError(
                            f"warmup.cycle_years={self.warmup_cycle_years} requires "
                            f"{expected_block_days} days in the first {self.warmup_cycle_years} "
                            f"simulation year(s) but only {block_days} are available in "
                            f"{file_path.name}. Reduce cycle_years or extend the "
                            f"simulation end date."
                        )

                    n_repetitions = max(1, int(np.ceil(warmup_days_needed / block_days)))
                    self.logger.debug(
                        f"  Cycle block: {cycle_block.time.min().values} to "
                        f"{cycle_block.time.max().values} ({block_days} days)"
                    )
                    self.logger.debug(f"  Repetitions: {n_repetitions}")

                    # Tile the block forward, time-shifted to start at warmup_date
                    repeated_datasets = []
                    current_time = pd.to_datetime(self.warmup_date)

                    for i in range(n_repetitions):
                        block_copy = cycle_block.copy(deep=True)
                        time_deltas = cycle_block.time - cycle_block.time.values[0]
                        new_times = current_time + time_deltas
                        block_copy['time'] = new_times
                        repeated_datasets.append(block_copy)
                        current_time = new_times.values[-1] + pd.Timedelta(days=1)
                        self.logger.debug(
                            f"  Repetition {i+1}: {new_times.values[0]} to {new_times.values[-1]}"
                        )

                    # ✅ CONCATENATE ONLY METEOROLOGICAL DATA (no elevation in either dataset)
                    warmup_data = xr.concat(repeated_datasets, dim='time')

                    # ✅ FIX: Trim warm-up to EXACTLY end one day before simulation start
                    # This ensures no overlap!
                    warmup_end_trim = self.start_date - pd.Timedelta(days=1)
                    warmup_data = warmup_data.sel(time=slice(self.warmup_date, warmup_end_trim))

                    self.logger.info(f"  Warm-up: {warmup_data.time.min().values} to {warmup_data.time.max().values} ({len(warmup_data.time)} days)")

                    # ✅ FIX: Verify no overlap before concatenation
                    warmup_last_time = pd.to_datetime(warmup_data.time.values[-1])
                    sim_first_time = pd.to_datetime(ds.time.values[0])
                    self.logger.debug(f"  Last warm-up time: {warmup_last_time}")
                    self.logger.debug(f"  First simulation time: {sim_first_time}")

                    if warmup_last_time >= sim_first_time:
                        self.logger.warning(f"⚠️ Overlap detected! Trimming warm-up to avoid duplicate")
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

                # For 'real' mode, load the combined dataset (which is just ds)
                # into memory so subsequent .time.values access is fast.
                if self.warmup_method == 'real':
                    combined = combined.load()
                    ds.close()
                    self.logger.info(
                        f"  Data range: {combined.time.min().values} to "
                        f"{combined.time.max().values} ({len(combined.time)} days)"
                    )

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
                meta = {
                    'warmup_included': 'true',
                    'warmup_start': str(self.warmup_date.date()),
                    'warmup_end': str((self.start_date - pd.Timedelta(days=1)).date()),
                    'warmup_days': n_warmup_days,
                    'simulation_start': str(self.start_date.date()),
                    'simulation_end': str(self.end_date.date()),
                    'simulation_days': n_sim_days,
                    'warmup_method': self.warmup_method,
                    'total_days': len(combined.time),
                    'elevation_included': 'true' if has_elevation else 'false',
                }
                if self.warmup_method == 'cycle':
                    meta['warmup_cycle_years'] = self.warmup_cycle_years
                    meta['warmup_repetitions'] = n_repetitions
                combined.attrs.update(meta)

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
        return updated_files

    #---------------------------------------------------------------------------------

    def calculate_monthly_temperature_averages(self) -> pd.DataFrame:
        """
        Calculate monthly mean temperature averages (climatology) and save to CSV.

        Works for both 1-D ERA5-Land grids (lat/lon dims) and 2-D HAR grids
        (south_north/west_east dims with 2-D lat/lon coords), using the
        ``_csv_prefix`` class attribute to distinguish output filenames.

        Returns
        -------
        pd.DataFrame
            Monthly temperature averages with columns: month, Temperature
        """
        self.logger.info("Calculating monthly temperature averages...")

        try:
            # Find the processed temperature mean file
            temp_file = None
            for file_path in self.processed_files:
                if 'temp_mean' in file_path.name:
                    temp_file = file_path
                    break

            if temp_file is None:
                self.logger.error("Temperature mean file not found in processed files")
                return pd.DataFrame()

            self.logger.debug(f"Using temperature file: {temp_file}")

            # Load with coordinate normalisation (ERA5: latitude->lat; HAR: passes through)
            ds = normalize_coords(xr.open_dataset(temp_file))

            # First variable with a time dim — skips any time-invariant elevation var
            temp_var = [v for v in ds.data_vars if 'time' in ds[v].dims][0]

            if self.catchment_extent is not None:
                self.logger.info("Masking temperature data to catchment extent...")
                catchment_bounds = self.catchment_extent.total_bounds  # minx, miny, maxx, maxy

                # DataArray masks: work for 1-D ERA5 (lat dim) and 2-D HAR (lat coord) alike
                lat_mask = (ds['lat'] >= catchment_bounds[1]) & (ds['lat'] <= catchment_bounds[3])
                lon_mask = (ds['lon'] >= catchment_bounds[0]) & (ds['lon'] <= catchment_bounds[2])
                mask = lat_mask & lon_mask

                ds_masked = ds[temp_var].where(mask, drop=False)
                spatial_dims = [d for d in ds[temp_var].dims if d != 'time']
                temp_spatial_avg = ds_masked.mean(dim=spatial_dims, skipna=True)

                n_valid_cells = int(mask.sum())
                self.logger.info(f"  Using {n_valid_cells} grid cells within catchment bounds")
            else:
                self.logger.warning("No catchment shapefile - using entire grid")
                spatial_dims = [d for d in ds[temp_var].dims if d != 'time']
                temp_spatial_avg = ds[temp_var].mean(dim=spatial_dims)

            # Monthly climatology
            monthly_climatology = temp_spatial_avg.groupby('time.month').mean()
            monthly_df = monthly_climatology.to_dataframe().reset_index()
            monthly_df = monthly_df.rename(columns={temp_var: 'Temperature'})

            # Ensure exactly 12 months
            full_months = pd.DataFrame({'month': list(range(1, 13))})
            monthly_df = full_months.merge(monthly_df, on='month', how='left')

            if monthly_df['Temperature'].isna().any():
                self.logger.warning("Some months have missing data - interpolating")
                monthly_df['Temperature'] = monthly_df['Temperature'].interpolate()

            monthly_df['Temperature'] = monthly_df['Temperature'].round(2)

            output_file = self.output_path / f'{self._csv_prefix}monthly_temperature_averages.csv'
            monthly_df.to_csv(output_file, index=False)

            self.logger.info(f"Monthly temperature averages saved to: {output_file}")
            self.logger.info("Monthly temperature averages (degC):")
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for _, row in monthly_df.iterrows():
                self.logger.info(f"  {month_names[int(row['month']) - 1]}: {row['Temperature']:.1f} degC")

            ds.close()
            return monthly_df

        except Exception as e:
            self.logger.error(f"Error calculating monthly temperature averages: {str(e)}")
            return pd.DataFrame()

    #---------------------------------------------------------------------------------

    def calculate_monthly_pet_averages(self) -> pd.DataFrame:
        """
        Calculate monthly mean PET averages (climatology) and save to CSV.

        Works for both 1-D ERA5-Land grids (lat/lon dims) and 2-D HAR grids
        (south_north/west_east dims with 2-D lat/lon coords).

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

            # Load with coordinate normalisation
            ds = normalize_coords(xr.open_dataset(pet_file))

            # First variable with a time dim
            pet_var = [v for v in ds.data_vars if 'time' in ds[v].dims][0]

            if self.catchment_extent is not None:
                self.logger.info("Masking PET data to catchment extent...")
                catchment_bounds = self.catchment_extent.total_bounds

                lat_mask = (ds['lat'] >= catchment_bounds[1]) & (ds['lat'] <= catchment_bounds[3])
                lon_mask = (ds['lon'] >= catchment_bounds[0]) & (ds['lon'] <= catchment_bounds[2])
                mask = lat_mask & lon_mask

                ds_masked = ds[pet_var].where(mask, drop=False)
                spatial_dims = [d for d in ds[pet_var].dims if d != 'time']
                pet_spatial_avg = ds_masked.mean(dim=spatial_dims, skipna=True)

                n_valid_cells = int(mask.sum())
                self.logger.info(f"  Using {n_valid_cells} grid cells within catchment bounds")
            else:
                self.logger.warning("No catchment shapefile - using entire grid")
                spatial_dims = [d for d in ds[pet_var].dims if d != 'time']
                pet_spatial_avg = ds[pet_var].mean(dim=spatial_dims)

            # Monthly climatology
            monthly_climatology = pet_spatial_avg.groupby('time.month').mean()
            monthly_df = monthly_climatology.to_dataframe().reset_index()
            monthly_df = monthly_df.rename(columns={pet_var: 'PET_avg_mm_per_day'})

            # Ensure exactly 12 months
            full_months = pd.DataFrame({'month': list(range(1, 13))})
            monthly_df = full_months.merge(monthly_df, on='month', how='left')

            if monthly_df['PET_avg_mm_per_day'].isna().any():
                self.logger.warning("Some months have missing PET data - interpolating")
                monthly_df['PET_avg_mm_per_day'] = monthly_df['PET_avg_mm_per_day'].interpolate()

            monthly_df['PET_avg_mm_per_day'] = monthly_df['PET_avg_mm_per_day'].round(3)

            output_file = self.output_path / f'{self._csv_prefix}monthly_pet_averages.csv'
            monthly_df.to_csv(output_file, index=False)

            self.logger.info(f"Monthly PET averages saved to: {output_file}")
            self.logger.info("Monthly PET averages (mm/day):")
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for _, row in monthly_df.iterrows():
                self.logger.info(f"  {month_names[int(row['month']) - 1]}: {row['PET_avg_mm_per_day']:.3f} mm/day")

            ds.close()
            return monthly_df

        except Exception as e:
            self.logger.error(f"Error calculating monthly PET averages: {str(e)}")
            return pd.DataFrame()

    #---------------------------------------------------------------------------------

    def compute_monthly_averages_for_subbasin(self, sb_gauge_id: str,
                                               output_dir: Path) -> bool:
        """
        Compute monthly temperature and PET averages for a specific subbasin.

        Temporarily overrides self.catchment_extent and self.output_path to point
        at the subbasin shapefile and output directory, calls the existing
        calculate_monthly_temperature_averages() and calculate_monthly_pet_averages()
        methods, then restores the originals.

        Parameters
        ----------
        sb_gauge_id : str
            Gauge ID of the subbasin (used to locate its catchment shapefile).
        output_dir : Path
            Directory where the per-subbasin CSV files are written.

        Returns
        -------
        bool
            True if both CSV files were written successfully, False otherwise.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        shape_template = self.config.get('shape_dir', '')
        shp_path = self.main_dir / shape_template.format(gauge_id=sb_gauge_id)
        if not Path(shp_path).exists():
            self.logger.warning(
                f"Catchment shapefile not found for subbasin {sb_gauge_id}: {shp_path} — "
                f"skipping monthly averages"
            )
            return False

        original_extent = self.catchment_extent
        original_output = self.output_path
        try:
            self.catchment_extent = gpd.read_file(shp_path)
            self.output_path = output_dir
            self.calculate_monthly_temperature_averages()
            self.calculate_monthly_pet_averages()
            self.logger.info(
                f"Per-subbasin monthly averages saved to {output_dir}"
            )
            return True
        except Exception as e:
            self.logger.error(
                f"Error computing monthly averages for subbasin {sb_gauge_id}: {e}"
            )
            return False
        finally:
            self.catchment_extent = original_extent
            self.output_path = original_output


