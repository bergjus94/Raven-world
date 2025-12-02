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
        self.debug = config.get('debug', False)
        self.model_dir = self.main_dir / config.get('config_dir')
        
        # GloGEM specific parameters
        self.glogem_dir = config.get('glogem_dir')
        if self.glogem_dir:
            self.glogem_dir = Path(self.main_dir, self.glogem_dir.format(gauge_id=self.gauge_id))
        
        # Setup logger
        self.logger = self._setup_logger()
        
        self.logger.info(f"GloGEM Processor initialized for gauge {self.gauge_id}")
        
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
    
    def _get_glogem_base_dir(self) -> Path:
        """Get the base directory for GloGEM files"""
        if self.glogem_dir.is_file() or str(self.glogem_dir).endswith('.dat'):
            return self.glogem_dir.parent
        return self.glogem_dir
    
    def process_glogem_files(self) -> pd.DataFrame:
        """
        Process GloGEM .dat files and create individual component CSV files.
        Memory-efficient version that processes in chunks.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with glacier melt data (id, date, q)
        """
        self.logger.info("Processing GloGEM files...")
        
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
            return pd.DataFrame(columns=['id', 'date', 'q'])
        
        self.logger.info(f"Looking for {len(glacier_ids_needed)} glacier IDs")
        
        # Get GloGEM base directory
        glogem_base_dir = self._get_glogem_base_dir()
        
        # File patterns for different components
        file_patterns = {
            'icemelt': "GloGEM_icemelt_*.dat",
            'snowmelt': "GloGEM_snowmelt_*.dat",
            'output': "GloGEM_output_*.dat",
            'rain': "GloGEM_rain_*.dat"
        }
        
        # Filter for date range
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        # Create output directory
        topo_dir = Path(self.model_dir) / f"catchment_{self.gauge_id}" / "topo_files"
        topo_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output files
        output_files = {
            'icemelt': topo_dir / 'GloGEM_icemelt.csv',
            'snowmelt': topo_dir / 'GloGEM_snowmelt.csv',
            'output': topo_dir / 'GloGEM_melt.csv',
            'rain': topo_dir / 'GloGEM_rain.csv'
        }
        
        # Process each component type
        for component, pattern in file_patterns.items():
            self.logger.info(f"Processing {component} files...")
            
            component_files = list(glogem_base_dir.glob(pattern))
            self.logger.info(f"Found {len(component_files)} {component} files")
            
            if not component_files:
                self.logger.warning(f"No {component} files found with pattern: {pattern}")
                continue
            
            # Open output CSV file for writing
            output_path = output_files[component]
            csv_file = open(output_path, 'w')
            csv_file.write('id,date,q\n')  # Header
            
            records_written = 0
            glaciers_found = set()
            
            # Parse each file
            for dat_file in component_files:
                self.logger.info(f"Reading {dat_file.name} (memory-efficient mode)...")
                
                try:
                    with open(dat_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            # Skip headers and empty lines
                            if line.startswith("ID") or line.startswith("//") or line.strip() == "":
                                continue
                            
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            
                            glacier_id = parts[0]
                            
                            # Check if this glacier is needed
                            if glacier_id not in glacier_ids_needed:
                                continue
                            
                            glaciers_found.add(glacier_id)
                            
                            try:
                                year = int(parts[1])
                                # area = float(parts[2])  # Not needed for CSV output
                                
                                # Parse daily values
                                daily_values = []
                                for val in parts[3:]:
                                    try:
                                        daily_values.append(float(val) if val != '*' else 0.0)
                                    except ValueError:
                                        daily_values.append(0.0)
                                
                                # Create date range (hydrological year: Oct 1 to Sep 30)
                                start_date_hydro = datetime(year-1, 10, 1)
                                
                                # Write daily values within date range
                                for day, value in enumerate(daily_values):
                                    date = start_date_hydro + timedelta(days=day)
                                    
                                    # Only write if within date range
                                    if start <= date <= end:
                                        csv_file.write(f"{glacier_id},{date.strftime('%Y-%m-%d')},{value}\n")
                                        records_written += 1
                                
                                # Log progress every 1000 glaciers
                                if len(glaciers_found) % 100 == 0:
                                    self.logger.debug(f"  Processed {len(glaciers_found)} glaciers, {records_written} records written")
                                        
                            except (ValueError, IndexError) as e:
                                self.logger.warning(f"Error parsing line {line_num} in {dat_file.name}: {e}")
                                continue
                                
                except Exception as e:
                    self.logger.error(f"Error reading file {dat_file}: {e}")
                    continue
            
            # Close CSV file
            csv_file.close()
            
            self.logger.info(f"✅ {component.capitalize()}: {output_path.name} ({records_written} records, {len(glaciers_found)} glaciers)")
        
        # Load the output (melt) CSV and return it
        melt_path = output_files['output']
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
        Create irrigation NetCDF file with GloGEM melt on glacier HRUs, zeros elsewhere.
        
        Parameters
        ----------
        force_reprocess : bool
            Force reprocessing even if file exists
            
        Returns
        -------
        xr.Dataset
            The created NetCDF dataset
        """
        self.logger.info("Creating irrigation NetCDF file...")
        
        # Check if output exists
        out_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs')
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / 'irrigation.nc'
        
        if output_path.exists() and not force_reprocess:
            self.logger.info(f"✅ Irrigation file already exists: {output_path}")
            self.logger.info("⏭️ Skipping. Set force_reprocess=True to reprocess.")
            return xr.open_dataset(output_path)
        
        # Load GloGEM melt data
        glogem_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'GloGEM_melt.csv')
        
        if not glogem_path.exists():
            self.logger.info("GloGEM melt file not found, processing files first...")
            glogem_df = self.process_glogem_files()
        else:
            glogem_df = pd.read_csv(glogem_path, dtype={'id': str})
        
        glogem_df['date'] = pd.to_datetime(glogem_df['date'])
        
        # Load HRU data
        hru_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
        hru_gdf = gpd.read_file(hru_path)
        hru_gdf = hru_gdf.sort_values(by='HRU_ID').reset_index(drop=True)
        hru_gdf['HRU ID'] = range(1, len(hru_gdf) + 1)
        
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
        
        # Generate full date range
        full_date_range = pd.date_range(self.start_date, self.end_date)
        
        # Initialize result with zeros
        num_hrus = len(hru_gdf)
        result_df = pd.DataFrame(
            np.zeros((len(full_date_range), num_hrus)),
            columns=range(1, num_hrus + 1)
        )
        
        # Fill in GloGEM data for glacier HRUs
        unique_ids = glogem_df['id'].unique()
        self.logger.info(f"Processing {len(unique_ids)} glacier HRUs")
        
        for glogem_id in unique_ids:
            # Create full glacier ID
            full_glacier_id = f"{rgi_region_code}.{glogem_id}" if rgi_region_code else str(glogem_id)
            
            # Find matching HRU
            mask = hru_gdf['Glacier_Cl'].notna() & (hru_gdf['Glacier_Cl'] == full_glacier_id)
            if not mask.any():
                self.logger.warning(f"Glacier {full_glacier_id} not found in HRU data")
                continue
            
            # Filter and reindex GloGEM data
            filtered_glogem = glogem_df[glogem_df['id'] == glogem_id].copy()
            filtered_glogem = filtered_glogem.set_index('date').reindex(full_date_range, fill_value=0).reset_index()
            filtered_glogem.rename(columns={'index': 'date'}, inplace=True)
            
            # Assign to HRU
            hru_id = hru_gdf.loc[mask, 'HRU ID'].iloc[0]
            result_df[hru_id] = filtered_glogem['q'].values
            
            self.logger.debug(f"Assigned irrigation to HRU {hru_id} for glacier {full_glacier_id}")
        
        # Create xarray Dataset
        result_array = result_df.to_numpy()
        x_values = np.arange(1, result_array.shape[1] + 1)
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
        
        # Save
        ds.to_netcdf(output_path)
        self.logger.info(f"✅ Saved irrigation NetCDF: {output_path}")
        
        # Log statistics
        glacier_hrus = (result_array != 0).any(axis=0).sum()
        non_zero = (result_array != 0).sum()
        
        self.logger.info(f"   Glacier HRUs: {glacier_hrus}/{num_hrus}")
        self.logger.info(f"   Non-zero values: {non_zero}/{result_array.size} ({non_zero/result_array.size*100:.2f}%)")
        self.logger.info(f"   Mean irrigation (glacier HRUs): {result_array[result_array != 0].mean():.3f} mm/day")
        self.logger.info(f"   Max irrigation: {result_array.max():.3f} mm/day")
        
        return ds
    
    def create_irrigation_gridweights(self) -> None:
        """
        Create GridWeights file specifically for irrigation forcing.
        Saves as GridWeights_Irrigation.txt to avoid overwriting existing file.
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
        
        # Save to unique filename
        out_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs')
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = out_dir / 'GridWeights_Irrigation.txt'
        
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
        Advanced validation of glacier IDs between HRU shapefile and GloGEM data.
        Creates detailed report and map visualization.
        
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
            # Load HRU data
            hru_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
            if not hru_path.exists():
                self.logger.error(f"HRU shapefile not found: {hru_path}")
                return results
            
            hru_gdf = gpd.read_file(hru_path)
            
            # Extract glacier IDs from HRU
            glacier_ids_hru_numeric = []
            rgi_region_code = None
            
            if 'Glacier_Cl' in hru_gdf.columns:
                glacier_series = hru_gdf['Glacier_Cl'].dropna()
                if not glacier_series.empty:
                    # Auto-detect region code
                    for glacier_id in glacier_series.unique():
                        if isinstance(glacier_id, str) and glacier_id.startswith('RGI60-'):
                            parts = glacier_id.split('.')
                            if len(parts) >= 2:
                                rgi_region_code = parts[0]
                                break
                    
                    self.logger.info(f"Auto-detected RGI region code: {rgi_region_code}")
                    
                    # Extract numeric IDs
                    for g_id in glacier_series.unique():
                        if isinstance(g_id, str) and rgi_region_code and f'{rgi_region_code}.' in g_id:
                            glacier_ids_hru_numeric.append(g_id.replace(f'{rgi_region_code}.', ''))
            
            # Load GloGEM data
            glogem_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'GloGEM_melt.csv')
            
            if glogem_path.exists():
                glogem_df = pd.read_csv(glogem_path, dtype={'id': str})
                glacier_ids_glogem = glogem_df['id'].unique().tolist()
            else:
                self.logger.warning("GloGEM melt file not found, processing files...")
                glogem_df = self.process_glogem_files()
                glacier_ids_glogem = glogem_df['id'].unique().tolist()
            
            # Compare sets
            glogem_set = set(glacier_ids_glogem)
            hru_set = set(glacier_ids_hru_numeric)
            
            missing_in_glogem = hru_set - glogem_set
            missing_in_hru = glogem_set - hru_set
            matched = hru_set.intersection(glogem_set)
            
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
                self.logger.warning(f"⚠️  Missing in HRU: {len(results['missing_in_hru'])}")
                for g_id in results['missing_in_hru'][:5]:
                    self.logger.warning(f"   - {g_id}")
                if len(results['missing_in_hru']) > 5:
                    self.logger.warning(f"   - ... and {len(results['missing_in_hru'])-5} more")
            
            # Save detailed report
            self._save_validation_report(results, rgi_region_code)
            
            # Create map visualization
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
        """
        import matplotlib.pyplot as plt
        
        try:
            self.logger.info("Creating glacier runoff vs observed streamflow comparison plots...")
            
            # Create plots directory
            plots_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', 'plots')
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # --- 1. Load GloGEM melt data (individual glacier records) ---
            glogem_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'GloGEM_melt.csv')
            
            if not glogem_path.exists():
                self.logger.warning(f"GloGEM melt file not found: {glogem_path}")
                return
                
            glogem_df = pd.read_csv(glogem_path, dtype={'id': str})
            glogem_df['date'] = pd.to_datetime(glogem_df['date'])
            glogem_df['q'] = pd.to_numeric(glogem_df['q'], errors='coerce')
            
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
            
            # Map glacier IDs to areas
            glacier_areas_df = hru_gdf[hru_gdf['Glacier_Cl'].notna()][['Glacier_Cl', area_col]].copy()
            glacier_areas_df['id'] = glacier_areas_df['Glacier_Cl'].str.replace(f'{rgi_region_code}.', '', regex=False)
            area_map = glacier_areas_df.set_index('id')[area_col].to_dict()
            
            # Add areas to GloGEM data
            glogem_df['area'] = glogem_df['id'].map(area_map)
            glogem_df['area'] = pd.to_numeric(glogem_df['area'], errors='coerce')
            
            # Calculate total areas
            glacier_area_km2 = hru_gdf[hru_gdf['Glacier_Cl'].notna()][area_col].sum()
            catchment_area_km2 = hru_gdf[area_col].sum()
            glacier_fraction = glacier_area_km2 / catchment_area_km2
            
            self.logger.info(f"Catchment area: {catchment_area_km2:.2f} km²")
            self.logger.info(f"Glacier area: {glacier_area_km2:.2f} km² ({glacier_fraction*100:.1f}%)")
            
            # --- 3. Calculate area-weighted glacier runoff per day (mm/day) ---
            daily_glacier = glogem_df.groupby('date').apply(
                lambda x: np.nansum(x['q'] * x['area']) / np.nansum(x['area']) if np.nansum(x['area']) > 0 else np.nan
            ).rename('glacier_runoff_per_glacier_area').reset_index()
            
            # Normalize to catchment area (multiply by glacier fraction)
            daily_glacier['glacier_runoff_catchment_norm'] = daily_glacier['glacier_runoff_per_glacier_area'] * glacier_fraction
            
            # --- 4. Load observed streamflow from Q_daily.rvt ---
            q_file = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs', 'Q_daily.rvt')
            
            if not q_file.exists():
                self.logger.warning(f"Observed streamflow file not found: {q_file}")
                obs_series = None
            else:
                with open(q_file, 'r') as f:
                    lines = f.readlines()
                
                # Find start date
                start_date_line = None
                for line in lines:
                    if line.strip() and not line.strip().startswith(':'):
                        start_date_line = line.strip()
                        break
                
                if start_date_line is None:
                    self.logger.warning("Could not find start date in Q_daily.rvt")
                    obs_series = None
                else:
                    start_date = start_date_line.split()[0]
                    
                    # Extract values
                    value_lines = []
                    found_start = False
                    for line in lines:
                        if found_start:
                            stripped = line.strip()
                            if stripped and not stripped.startswith(':'):
                                try:
                                    value = float(stripped)
                                    value_lines.append(value)
                                except ValueError:
                                    continue
                        elif line.strip() == start_date_line:
                            found_start = True
                    
                    # Create time series
                    obs_dates = pd.date_range(start=start_date, periods=len(value_lines))
                    obs_series = pd.Series(value_lines, index=obs_dates, name='observed_streamflow_m3s')
                    
                    # Convert from m³/s to mm/day
                    obs_series_mm = obs_series * 86400 / (catchment_area_km2 * 1e6) * 1000
                    obs_series_mm.name = 'observed_streamflow_mm'
            
            # --- 5. Merge data for plotting ---
            plot_df = daily_glacier.copy()
            
            if obs_series is not None:
                plot_df = plot_df.merge(obs_series_mm, left_on='date', right_index=True, how='left')
            
            # --- 6. Create time series plot (mm/day) ---
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot glacier runoff (per glacier area)
            ax.plot(plot_df['date'], plot_df['glacier_runoff_per_glacier_area'], 
                label='Glacier Runoff (per glacier area, mm/day)', 
                color='blue', alpha=0.7, linewidth=1)
            
            # Plot catchment-normalized glacier runoff
            ax.plot(plot_df['date'], plot_df['glacier_runoff_catchment_norm'], 
                label=f'Glacier Runoff (catchment-normalized, {glacier_fraction*100:.1f}% glacier coverage)', 
                color='green', alpha=0.7, linewidth=1.5)
            
            # Plot observed streamflow if available
            if obs_series is not None and 'observed_streamflow_mm' in plot_df.columns:
                ax.plot(plot_df['date'], plot_df['observed_streamflow_mm'], 
                    label='Observed Streamflow (mm/day)', 
                    color='black', linewidth=1)
            
            ax.set_title(f'Glacier Runoff vs Observed Streamflow - Gauge {self.gauge_id}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Discharge (mm/day)')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'glacier_runoff_vs_observed_mm.png', dpi=300, bbox_inches='tight')
            if self.debug:
                plt.show()
            plt.close()
            
            self.logger.info(f"✅ Time series plot saved: {plots_dir / 'glacier_runoff_vs_observed_mm.png'}")
            
            # --- 7. Create monthly regime plot ---
            regime_df = plot_df.copy()
            regime_df['month'] = regime_df['date'].dt.month
            
            monthly_glacier = regime_df.groupby('month')['glacier_runoff_catchment_norm'].mean()
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            x = range(1, 13)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot glacier runoff regime
            ax.bar(x, monthly_glacier.values, color='green', alpha=0.7, 
                edgecolor='black', label='Glacier Runoff (catchment-normalized)')
            ax.plot(x, monthly_glacier.values, 'go-', linewidth=2, markersize=6)
            
            # Plot observed if available
            if obs_series is not None and 'observed_streamflow_mm' in plot_df.columns:
                monthly_obs = regime_df.groupby('month')['observed_streamflow_mm'].mean()
                ax.plot(x, monthly_obs.values, 'ko-', linewidth=2, markersize=6, 
                    label='Observed Streamflow')
                
                # Add value labels
                for i, (glacier_val, obs_val) in enumerate(zip(monthly_glacier.values, monthly_obs.values), 1):
                    ax.text(i, glacier_val + 0.1, f'{glacier_val:.2f}', 
                        ha='center', va='bottom', fontsize=8)
                    ax.text(i, obs_val + 0.1, f'{obs_val:.2f}', 
                        ha='center', va='bottom', fontsize=8)
            else:
                # Add value labels for glacier only
                for i, val in enumerate(monthly_glacier.values, 1):
                    ax.text(i, val + 0.1, f'{val:.2f}', 
                        ha='center', va='bottom', fontsize=8)
            
            ax.set_title(f'Monthly Regime: Glacier Runoff vs Observed Streamflow\nGauge {self.gauge_id}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Mean Discharge (mm/day)')
            ax.set_xticks(x)
            ax.set_xticklabels(month_names)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'monthly_regime_glacier_vs_observed.png', dpi=300, bbox_inches='tight')
            if self.debug:
                plt.show()
            plt.close()
            
            self.logger.info(f"✅ Monthly regime plot saved: {plots_dir / 'monthly_regime_glacier_vs_observed.png'}")
            
            # --- 8. Log statistics ---
            self.logger.info("\n" + "="*60)
            self.logger.info("GLACIER RUNOFF VS OBSERVED STREAMFLOW STATISTICS")
            self.logger.info("="*60)
            self.logger.info(f"Catchment area: {catchment_area_km2:.2f} km²")
            self.logger.info(f"Glacier area: {glacier_area_km2:.2f} km² ({glacier_fraction*100:.1f}%)")
            self.logger.info(f"Mean glacier runoff (per glacier area): {plot_df['glacier_runoff_per_glacier_area'].mean():.2f} mm/day")
            self.logger.info(f"Mean glacier runoff (catchment-normalized): {plot_df['glacier_runoff_catchment_norm'].mean():.2f} mm/day")
            
            if obs_series is not None and 'observed_streamflow_mm' in plot_df.columns:
                self.logger.info(f"Mean observed streamflow: {plot_df['observed_streamflow_mm'].mean():.2f} mm/day")
                glacier_contribution = (plot_df['glacier_runoff_catchment_norm'].mean() / 
                                    plot_df['observed_streamflow_mm'].mean() * 100)
                self.logger.info(f"Glacier contribution to streamflow: {glacier_contribution:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error creating glacier runoff comparison plots: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

        
    def process_all(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Run complete GloGEM processing pipeline.
        
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
        self.logger.info("="*60)
        
        results = {}
        
        # 1. Process GloGEM files (creates individual glacier CSVs)
        self.logger.info("\n1. Processing GloGEM .dat files...")
        glogem_df = self.process_glogem_files()
        results['glogem_data'] = glogem_df
        
        # 2. Create catchment-averaged melt file (NEW!)
        self.logger.info("\n2. Creating catchment-averaged glacier melt file...")
        catchment_avg_df = self.create_catchment_averaged_melt()
        results['catchment_averaged_melt'] = catchment_avg_df
        
        # 3. Create irrigation NetCDF (uses individual glacier data)
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
        self.logger.info(f"✅ Irrigation NetCDF created (uses individual glacier data)")
        self.logger.info(f"✅ Grid weights created")
        self.logger.info(f"✅ Validation: {len(validation['matched'])} matched, "
                        f"{len(validation['missing_in_glogem'])} missing in GloGEM, "
                        f"{len(validation['missing_in_hru'])} missing in HRU")
        self.logger.info(f"✅ Comparison plots created")
        
        return results

# Example usage
if __name__ == "__main__":
    namelist_path = "../namelist.yaml"
    
    processor = GloGEMProcessor(namelist_path)
    results = processor.process_all(force_reprocess=False)