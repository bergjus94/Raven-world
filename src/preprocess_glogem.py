import geopandas as gpd
from joblib import Parallel, delayed
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import multiprocessing
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple
import yaml
import glob

class GloGEMProcessor:
    """
    A class for processing GloGEM glacier melt data for hydrological modeling.
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
        
        # Extract configuration parameters from namelist
        self.gauge_id = config['gauge_id']
        self.main_dir = Path(config['main_dir'])
        self.model_type = config['model_type']
        self.start_date = config['start_date']
        self.end_date = config['end_date']
        self.debug = config.get('debug', False)
        self.coupled = config.get('coupled', False)
        self.model_dir = self.main_dir / config.get('config_dir')
        
        # GloGEM specific parameters (you may need to add these to your namelist)
        self.glogem_dir = config.get('glogem_dir')  # Add this to your namelist
        self.output_unit = config.get('output_unit', 'mm')
        self.glogem_dir = Path(self.main_dir, self.glogem_dir.format(gauge_id=self.gauge_id))

        # Path to save GloGEM melt data
        self.glogem_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'GloGEM_melt.csv')
        
        # Setup logger
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """
        Set up logger for this class
        
        Returns
        -------
        logging.Logger
            Configured logger
        """
        # Use DEBUG level only in debug mode, otherwise use WARNING
        log_level = logging.DEBUG if self.debug else logging.WARNING
        
        # Configure logging
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=log_level,
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
        
        # Create logger with appropriate name
        logger = logging.getLogger('GloGEMProcessor')
        
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
                    return record.name == 'GloGEMProcessor' and record.levelno >= logging.INFO
                    
            console.addFilter(ModuleFilter())
            logger.addHandler(console)
        
        return logger
    
    def get_dat_file(self) -> pd.DataFrame:
        """
        Read and parse GloGEM .dat file
        
        Returns
        -------
        pd.DataFrame
            Parsed DataFrame with GloGEM data
        """
        self.logger.debug(f"Reading GloGEM .dat file from {self.glogem_dir}")
        
        # read in .dat file
        df = pd.read_csv(self.glogem_dir, delimiter='\t', header=None, skiprows=1)

        # delete rows that have no melt output
        df.rename(columns={0: 'combined_column'}, inplace=True)
        df = df[~df.apply(lambda row: any('*' in cell for cell in row), axis=1)]

        # make datafram into something usable for further processing
        days_column_names = ['day' + str(i) for i in range(1, 366)]
        additional_column_names = ['id', 'year', 'area']
        all_column_names = additional_column_names + days_column_names
        split_columns = df['combined_column'].str.split('\s{1,}', expand=True)
        split_columns.columns = all_column_names
        
        return split_columns
    
    def build_glogem_melt_from_multiple_files(self):
        """
        If the expected GloGEM files for the catchment are missing, 
        search all .dat files in the GloGEM directory, extract glacier data 
        for glacier IDs present in the catchment shapefile, and build separate CSV files:
        - GloGEM_icemelt.csv
        - GloGEM_snowmelt.csv  
        - GloGEM_rain.csv
        - GloGEM_melt.csv (total/output)
        """
        from datetime import datetime, timedelta
        import pickle
        
        self.logger.info(f"Building GloGEM component data from multiple files for gauge {self.gauge_id}")
        
        # Get glacier IDs from catchment shapefile
        catchment_shape_file = Path(self.model_dir) / f"catchment_{self.gauge_id}" / "topo_files" / "HRU.shp"
        
        if not catchment_shape_file.exists():
            raise FileNotFoundError(f"Catchment shapefile not found: {catchment_shape_file}")
        
        catchment = gpd.read_file(catchment_shape_file)
        
        # Extract glacier IDs from the catchment
        glacier_ids_needed = set()
        rgi_region_code = None  # Will be auto-detected
        
        if 'Glacier_Cl' in catchment.columns:
            glacier_series = catchment['Glacier_Cl'].dropna()
            if not glacier_series.empty:
                # Auto-detect RGI region code from the first glacier ID
                for glacier_id in glacier_series.unique():
                    if isinstance(glacier_id, str) and glacier_id.startswith('RGI60-'):
                        # Extract the region code (e.g., 'RGI60-15' from 'RGI60-15.12345')
                        parts = glacier_id.split('.')
                        if len(parts) >= 2:
                            rgi_region_code = parts[0]  # e.g., 'RGI60-15'
                            break
                
                self.logger.info(f"Auto-detected RGI region code: {rgi_region_code}")
                
                # Convert RGI60-XX.xxxxx to xxxxx format for matching with GloGEM
                for glacier_id in glacier_series.unique():
                    if isinstance(glacier_id, str) and rgi_region_code and glacier_id.startswith(rgi_region_code + '.'):
                        glacier_ids_needed.add(glacier_id.replace(rgi_region_code + '.', ''))
        
        if not glacier_ids_needed:
            self.logger.warning("No glacier IDs found in catchment shapefile")
            return pd.DataFrame()
        
        self.logger.info(f"Looking for {len(glacier_ids_needed)} glacier IDs: {list(glacier_ids_needed)[:5]}...")
        
        # Get the base directory properly
        if self.glogem_dir.is_file() or str(self.glogem_dir).endswith('.dat'):
            # If glogem_dir points to a file, use its parent directory
            glogem_base_dir = self.glogem_dir.parent
        else:
            # If glogem_dir is already a directory, use it as is
            glogem_base_dir = self.glogem_dir
        
        # Define file patterns and corresponding component names
        file_patterns = {
            'icemelt': "GloGEM_icemelt_*.dat",
            'snowmelt': "GloGEM_snowmelt_*.dat", 
            'output': "GloGEM_output_*.dat",
            'rain': "GloGEM_rain_*.dat"  # Added rain component
        }
        
        # Store records by component and glacier ID
        component_records = {
            'icemelt': {},  # {glacier_id: {date: value}}
            'snowmelt': {},
            'output': {},
            'rain': {}
        }
        glacier_areas = {}
        
        # Process each component type
        for component, pattern in file_patterns.items():
            self.logger.info(f"Processing {component} files...")
            
            component_files = list(glogem_base_dir.glob(pattern))
            self.logger.info(f"Found {len(component_files)} {component} files")
            
            if not component_files:
                self.logger.warning(f"No {component} files found with pattern: {pattern}")
                continue
            
            # Search through all files for this component
            for dat_file in component_files:
                self.logger.debug(f"Searching {component} file: {dat_file.name}")
                
                # Read and parse the file
                try:
                    with open(dat_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            if line.startswith("ID") or line.startswith("//") or line.strip() == "":
                                continue
                            
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            
                            glacier_id = parts[0]
                            
                            # Check if this glacier ID is needed
                            if glacier_id not in glacier_ids_needed:
                                continue
                            
                            try:
                                year = int(parts[1])
                                area = float(parts[2])
                                glacier_areas[glacier_id] = area  # Store area for later use
                                
                                # Parse daily values (replace '*' with 0.0)
                                daily_values = []
                                for val in parts[3:]:
                                    try:
                                        daily_values.append(float(val) if val != '*' else 0.0)
                                    except ValueError:
                                        daily_values.append(0.0)
                                
                                # Create date range for hydrological year (Oct 1 to Sep 30)
                                start_date_hydro = datetime(year-1, 10, 1)
                                
                                # Initialize glacier record if not exists
                                if glacier_id not in component_records[component]:
                                    component_records[component][glacier_id] = {}
                                
                                # Store daily values for this glacier
                                for day, value in enumerate(daily_values):
                                    if day < len(daily_values):  # Safety check
                                        date = start_date_hydro + timedelta(days=day)
                                        component_records[component][glacier_id][date] = value
                                        
                            except (ValueError, IndexError) as e:
                                self.logger.warning(f"Error parsing line {line_num} in {dat_file.name}: {e}")
                                continue
                                
                except Exception as e:
                    self.logger.error(f"Error reading file {dat_file}: {e}")
                    continue
        
        # Check if we found any data
        total_components_found = sum(1 for records in component_records.values() if records)
        if total_components_found == 0:
            raise ValueError(f"No glacier data found for any component for glacier IDs: {glacier_ids_needed}")
        
        self.logger.info(f"Found data for components: {[comp for comp, records in component_records.items() if records]}")
        
        # Filter for the specified date range
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        # Create output directory
        topo_dir = Path(self.model_dir) / f"catchment_{self.gauge_id}" / "topo_files"
        topo_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output file paths
        output_files = {
            'icemelt': topo_dir / 'GloGEM_icemelt.csv',
            'snowmelt': topo_dir / 'GloGEM_snowmelt.csv',
            'output': topo_dir / 'GloGEM_melt.csv',  # Keep original name for backward compatibility
            'rain': topo_dir / 'GloGEM_rain.csv'
        }
        
        # Process each component and create CSV files
        component_dataframes = {}
        
        for component, glacier_records in component_records.items():
            if not glacier_records:
                self.logger.warning(f"No data found for {component} component, skipping...")
                continue
                
            self.logger.info(f"Creating {component} CSV file...")
            
            # Create final records for this component
            final_records = []
            
            for glacier_id, date_values in glacier_records.items():
                for date, value in date_values.items():
                    if start <= date <= end:
                        final_records.append({
                            'id': glacier_id,
                            'date': date,
                            'q': value
                        })
            
            if not final_records:
                self.logger.warning(f"No {component} records found within the specified date range")
                # Create empty dataframe with correct structure
                final_df = pd.DataFrame(columns=['id', 'date', 'q'])
            else:
                # Create DataFrame with individual glacier records
                final_df = pd.DataFrame(final_records)
                
                # Sort by glacier ID and date for consistency
                final_df = final_df.sort_values(['id', 'date']).reset_index(drop=True)
            
            # Save the component CSV file
            final_df.to_csv(output_files[component], index=False)
            component_dataframes[component] = final_df
            
            self.logger.info(f"✅ {component.capitalize()} file created: {output_files[component]} ({len(final_df)} records)")
            
            if not final_df.empty:
                num_glaciers = final_df['id'].nunique()
                date_range_str = f"{final_df['date'].min()} to {final_df['date'].max()}"
                self.logger.info(f"   Found {num_glaciers} individual glaciers, Date range: {date_range_str}")
        
        # Create summary report
        self.logger.info("\n" + "="*60)
        self.logger.info("GLOGEM COMPONENT PROCESSING SUMMARY")
        self.logger.info("="*60)
        
        for component, df in component_dataframes.items():
            if not df.empty:
                num_records = len(df)
                num_glaciers = df['id'].nunique()
                mean_value = df['q'].mean()
                max_value = df['q'].max()
                total_value = df['q'].sum()
                
                self.logger.info(f"{component.upper()}:")
                self.logger.info(f"  📁 File: {output_files[component].name}")
                self.logger.info(f"  📊 Records: {num_records}")
                self.logger.info(f"  🏔️  Glaciers: {num_glaciers}")
                self.logger.info(f"  📈 Stats: Mean={mean_value:.3f}, Max={max_value:.3f}, Total={total_value:.1f}")
            else:
                self.logger.info(f"{component.upper()}: No data available")
        
        # For backward compatibility, return the total output (melt) dataframe in aggregated format
        if 'output' in component_dataframes and not component_dataframes['output'].empty:
            output_df = component_dataframes['output']
            # Convert to datetime for processing
            output_df['date'] = pd.to_datetime(output_df['date'])
            
            # Create aggregated data for return (for backwards compatibility with plotting functions)
            daily_totals = output_df.groupby('date')['q'].sum().reset_index()
            daily_totals.columns = ['date', 'glacier_melt']
            daily_totals['snowmelt'] = 0.0  # Placeholder
            daily_totals['total_output'] = daily_totals['glacier_melt']
            
            return daily_totals
        else:
            # Return empty dataframe with expected columns
            return pd.DataFrame(columns=['date', 'glacier_melt', 'snowmelt', 'total_output'])
        
    def prepare_irrigation_coupled(self, force_reprocess: bool = False) -> pd.DataFrame:
            """
            Prepare irrigation data (GloGEM melt on glacier HRUs, zeros elsewhere) as a NetCDF file.
            This creates an irrigation forcing file with the same structure as precipitation/temperature.
            
            Parameters
            ----------
            force_reprocess : bool, optional
                If True, force reprocessing even if the output file exists (default is False).
            
            Returns
            -------
            pd.DataFrame
                DataFrame with irrigation values for each HRU
            """
            try:
                self.logger.info("Preparing coupled irrigation data (GloGEM melt as irrigation)")

                # Check if output file already exists
                out_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs')
                output_path = Path(out_dir, 'irrigation.nc')
                
                if output_path.exists() and not force_reprocess:
                    self.logger.info(f"✅ Irrigation file already exists: {output_path}")
                    self.logger.info("⏭️ Skipping processing. Set force_reprocess=True to reprocess.")
                    # Load and return existing data
                    ds = xr.open_dataset(output_path)
                    # Convert to DataFrame if needed
                    return pd.DataFrame()  # Or extract data from ds
                
                # Check if required fields are available
                required_fields = ['model_type', 'start_date', 'end_date']
                for field in required_fields:
                    if not getattr(self, field):
                        raise ValueError(f"Missing required field for coupled forcing: {field}")
                
                # Load GloGEM data
                glogem_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'GloGEM_melt.csv')
                if not glogem_dir.exists():
                    raise FileNotFoundError(f"GloGEM melt file not found: {glogem_dir}")
                    
                glogem = pd.read_csv(glogem_dir, dtype={'id': str})[['id', 'date', 'q']]
                glogem['date'] = pd.to_datetime(glogem['date'])

                # Load HRU data
                hru_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
                HRU = gpd.read_file(hru_dir)
                HRU = HRU.sort_values(by='HRU_ID').reset_index(drop=True)
                HRU['HRU ID'] = range(1, len(HRU) + 1)
                
                # Auto-detect RGI region code from glacier IDs
                rgi_region_code = None
                if 'Glacier_Cl' in HRU.columns:
                    glacier_series = HRU['Glacier_Cl'].dropna()
                    if not glacier_series.empty:
                        for glacier_id in glacier_series.unique():
                            if isinstance(glacier_id, str) and glacier_id.startswith('RGI60-'):
                                parts = glacier_id.split('.')
                                if len(parts) >= 2:
                                    rgi_region_code = parts[0]
                                    break
                        self.logger.info(f"Auto-detected RGI region code: {rgi_region_code}")
                
                # Generate the full date range
                full_date_range = pd.date_range(self.start_date, self.end_date)
                
                # Initialize result dataframe with zeros for all HRUs
                num_hrus = len(HRU)
                result_df = pd.DataFrame(
                    np.zeros((len(full_date_range), num_hrus)),
                    columns=range(1, num_hrus + 1)
                )
                
                # Fill in GloGEM data for glacier HRUs
                unique_ids = glogem['id'].unique()
                self.logger.info(f"Processing {len(unique_ids)} glacier HRUs for irrigation")
                
                for glogem_id in unique_ids:
                    # Create the full glacier ID string
                    full_glacier_id = f"{rgi_region_code}.{glogem_id}" if rgi_region_code else str(glogem_id)
                    
                    # Check if this glacier exists in the HRU DataFrame
                    mask = HRU['Glacier_Cl'].notna() & (HRU['Glacier_Cl'] == full_glacier_id)
                    if not mask.any():
                        self.logger.warning(f"Warning: Glacier ID {full_glacier_id} from GloGEM not found in HRU data")
                        continue
                    
                    # Filter glogem time series for this ID
                    filtered_glogem = glogem[glogem['id'] == glogem_id].copy()
                    filtered_glogem['date'] = pd.to_datetime(filtered_glogem['date'])
                    
                    # Reindex to match the full date range (fill missing dates with 0)
                    filtered_glogem = filtered_glogem.set_index('date').reindex(full_date_range, fill_value=0).reset_index()
                    filtered_glogem.rename(columns={'index': 'date'}, inplace=True)
                    
                    # Identify the HRU ID
                    hru_id = HRU.loc[mask, 'HRU ID'].iloc[0]
                    
                    # Assign glogem values (irrigation) to the corresponding HRU
                    result_df[hru_id] = filtered_glogem['q'].values
                    
                    self.logger.debug(f"Assigned irrigation to HRU {hru_id} for glacier {full_glacier_id}")
                
                # Create time index
                time_index = full_date_range
                
                # Format to numpy array
                result_array = result_df.to_numpy()
                
                # Create the dimension variables
                x_values = np.arange(1, result_array.shape[1] + 1)
                y_values = np.arange(1, 2)
                
                # Create xarray Dataset
                ds_new = xr.Dataset(
                    {'data': (['time', 'x', 'y'], result_array.reshape(len(time_index), -1, 1))},
                    coords={'time': time_index, 'x': x_values, 'y': y_values}
                )
                
                # Add elevation data
                elevation_values = HRU['Elev_Mean'].values
                elevation_da = xr.DataArray(
                    elevation_values.reshape(-1, 1),
                    dims=['x', 'y'], 
                    coords={'x': ds_new['x'], 'y': ds_new['y']}
                )
                ds_new['elevation'] = elevation_da
                
                # Save to NetCDF
                out_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs')
                out_dir.mkdir(parents=True, exist_ok=True)
                output_path = Path(out_dir, 'irrigation.nc')
                ds_new.to_netcdf(output_path)
                
                self.logger.info(f"✅ Saved irrigation data to {output_path}")
                
                # Log statistics
                glacier_hrus_count = (result_df != 0).any(axis=0).sum()
                non_zero_count = (result_array != 0).sum()
                total_count = result_array.size
                
                self.logger.info(f"Irrigation statistics:")
                self.logger.info(f"  Glacier HRUs with irrigation: {glacier_hrus_count}/{num_hrus}")
                self.logger.info(f"  Non-zero values: {non_zero_count}/{total_count} ({non_zero_count/total_count*100:.2f}%)")
                self.logger.info(f"  Mean irrigation (glacier HRUs only): {result_array[result_array != 0].mean():.3f} mm/day")
                self.logger.info(f"  Max irrigation: {result_array.max():.3f} mm/day")
                
                # Create plot if debug mode
                if self.debug:
                    self._plot_irrigation_timeseries_and_regime(ds_new, title="Irrigation (GloGEM Melt)")
                
                return result_df
                
            except Exception as e:
                self.logger.error(f"Error in prepare_irrigation_coupled: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise
    
    def process_glogem_data_optimized(self, force_reprocess: bool = False, chunk_size: int = 10000, plot: bool = True) -> pd.DataFrame:
        """
        Optimized version that processes large GloGEM files efficiently and caches results.
        """
        import pickle
        from datetime import datetime, timedelta

        # Setup directories and file paths
        topo_dir = Path(self.model_dir) / f"catchment_{self.gauge_id}" / "topo_files"
        
        # FIX: Get the parent directory of glogem_dir for cache and file operations
        if self.glogem_dir.is_file() or str(self.glogem_dir).endswith('.dat'):
            # If glogem_dir points to a file, use its parent directory
            glogem_base_dir = self.glogem_dir.parent
        else:
            # If glogem_dir is already a directory, use it as is
            glogem_base_dir = self.glogem_dir
        
        cache_dir = glogem_base_dir / "processed_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_files = {
            'icemelt': cache_dir / f"icemelt_{self.gauge_id}_processed.pkl",
            'snowmelt': cache_dir / f"snowmelt_{self.gauge_id}_processed.pkl", 
            'output': cache_dir / f"output_{self.gauge_id}_processed.pkl",
            'final_result': cache_dir / f"final_result_{self.gauge_id}_{self.start_date}_{self.end_date}_{self.output_unit}.pkl"
        }
        
        file_paths = {
            'icemelt': glogem_base_dir / f"GloGEM_icemelt_{self.gauge_id}.dat",
            'snowmelt': glogem_base_dir / f"GloGEM_snowmelt_{self.gauge_id}.dat",
            'output': glogem_base_dir / f"GloGEM_output_{self.gauge_id}.dat"
        }
        
        catchment_shape_file = topo_dir / "HRU.shp"

        # Check if expected files exist, if not, build from multiple files
        files_exist = any(path.exists() for path in file_paths.values())
        
        if not files_exist:
            self.logger.warning(f"Expected GloGEM files not found for gauge {self.gauge_id}")
            self.logger.info("Searching through all GloGEM files in directory...")
            
            # Use the new function to build from multiple files
            scaled_df = self.build_glogem_melt_from_multiple_files()
            
            if plot:
                self.create_glogem_plots(scaled_df)
            return scaled_df

        # Check if final result already exists and is recent
        if cache_files['final_result'].exists() and not force_reprocess:
            self.logger.info(f"Loading cached final result from {cache_files['final_result']}")
            with open(cache_files['final_result'], 'rb') as f:
                scaled_df = pickle.load(f)
            
            # ✅ NEW: Create individual CSV files even when loading from cache
            if not self.glogem_path.exists() or force_reprocess:
                self.logger.info("Creating missing CSV files from cached data...")
                
                # Create output directory
                topo_dir.mkdir(parents=True, exist_ok=True)
                
                # Try to load individual glacier data from cache
                component_cache_exists = all(cache_files[comp].exists() for comp in ['icemelt', 'snowmelt', 'output'])
                
                if component_cache_exists:
                    # Load cached component data
                    with open(cache_files['icemelt'], 'rb') as f:
                        icemelt_df, icemelt_areas = pickle.load(f)
                    with open(cache_files['snowmelt'], 'rb') as f:
                        snowmelt_df, snowmelt_areas = pickle.load(f)
                    with open(cache_files['output'], 'rb') as f:
                        output_df, output_areas = pickle.load(f)
                    
                    # Filter for date range
                    start = datetime.strptime(self.start_date, '%Y-%m-%d')
                    end = datetime.strptime(self.end_date, '%Y-%m-%d')
                    icemelt_df = icemelt_df[(icemelt_df['date'] >= start) & (icemelt_df['date'] <= end)]
                    snowmelt_df = snowmelt_df[(snowmelt_df['date'] >= start) & (snowmelt_df['date'] <= end)]
                    output_df = output_df[(output_df['date'] >= start) & (output_df['date'] <= end)]
                    
                    # Define output file paths for individual components
                    output_files = {
                        'icemelt': topo_dir / 'GloGEM_icemelt.csv',
                        'snowmelt': topo_dir / 'GloGEM_snowmelt.csv',
                        'output': topo_dir / 'GloGEM_melt.csv',  # Keep original name for backward compatibility
                    }
                    
                    # Create individual component CSV files
                    component_dfs = {
                        'icemelt': icemelt_df,
                        'snowmelt': snowmelt_df,
                        'output': output_df
                    }
                    
                    for component, df in component_dfs.items():
                        if not df.empty:
                            # Create final records for this component
                            final_records = []
                            for _, row in df.iterrows():
                                final_records.append({
                                    'id': row['glacier_id'],
                                    'date': row['date'],
                                    'q': row['value']
                                })
                            
                            # Create DataFrame and save to CSV
                            if final_records:
                                final_df = pd.DataFrame(final_records)
                                final_df = final_df.sort_values(['id', 'date']).reset_index(drop=True)
                                
                                # Save the component CSV file
                                final_df.to_csv(output_files[component], index=False)
                                
                                self.logger.info(f"✅ {component.capitalize()} file created from cache: {output_files[component].name} ({len(final_df)} records)")
                            else:
                                self.logger.warning(f"No records found for {component} component in cache")
                        else:
                            self.logger.warning(f"No data found for {component} component in cache")
                else:
                    self.logger.warning("Cannot create individual CSV files - component cache missing, will build from multiple files")
                    scaled_df = self.build_glogem_melt_from_multiple_files()
                    if plot:
                        self.create_glogem_plots(scaled_df)
                    return scaled_df
            
            if plot:
                self.create_glogem_plots(scaled_df)
            return scaled_df

        # Helper to parse large GloGEM file in chunks and cache the result
        def parse_glogem_file_chunked(file_path, cache_file, file_type):
            if cache_file.exists() and not force_reprocess:
                self.logger.info(f"Loading cached {file_type} data from {cache_file}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            self.logger.info(f"Processing {file_type} file: {file_path}")
            data_chunks = []
            areas = {}
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith("ID") or line.startswith("//"):
                        continue
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    glacier_id = parts[0]
                    year = int(parts[1])
                    area = float(parts[2])
                    areas[glacier_id] = area
                    daily_values = [float(val) if val != '*' else 0.0 for val in parts[3:]]
                    start_date_hydro = datetime(year-1, 10, 1)
                    for day, value in enumerate(daily_values):
                        date = start_date_hydro + timedelta(days=day)
                        data_chunks.append({
                            'glacier_id': glacier_id,
                            'date': date,
                            'area': area,
                            'value': value
                        })
            df = pd.DataFrame(data_chunks)
            with open(cache_file, 'wb') as f:
                pickle.dump((df, areas), f, protocol=pickle.HIGHEST_PROTOCOL)
            return df, areas

        # Parse all three files with caching
        icemelt_df, icemelt_areas = parse_glogem_file_chunked(
            file_paths['icemelt'], cache_files['icemelt'], 'icemelt'
        )
        snowmelt_df, snowmelt_areas = parse_glogem_file_chunked(
            file_paths['snowmelt'], cache_files['snowmelt'], 'snowmelt'
        )
        output_df, output_areas = parse_glogem_file_chunked(
            file_paths['output'], cache_files['output'], 'output'
        )

        # Filter for the date range
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        icemelt_df = icemelt_df[(icemelt_df['date'] >= start) & (icemelt_df['date'] <= end)]
        snowmelt_df = snowmelt_df[(snowmelt_df['date'] >= start) & (snowmelt_df['date'] <= end)]
        output_df = output_df[(output_df['date'] >= start) & (output_df['date'] <= end)]

        # Read catchment shapefile
        catchment = gpd.read_file(catchment_shape_file)
        glacier_areas = catchment.groupby('Glacier_Cl').agg({'Area_km2': 'sum'}).reset_index()
        glacier_area = glacier_areas['Area_km2'].sum()
        total_area = catchment['Area_km2'].sum()
        percentage = (glacier_area / total_area) * 100
        area_dict = {row['Glacier_Cl'][-5:]: row['Area_km2'] for _, row in glacier_areas.iterrows()}

        # Update icemelt_df with correct areas
        for glacier_id, new_area in area_dict.items():
            mask = icemelt_df['glacier_id'] == glacier_id
            if mask.any():
                icemelt_df.loc[mask, 'area'] = new_area

        # Calculate catchment average runoff (memory efficient)
        def calculate_catchment_average_efficient(df, areas):
            catchment_area = sum(areas.values())
            df = df.copy()
            df['weighted_value'] = df['value'] * df['area']
            daily_sum = df.groupby('date')['weighted_value'].sum().reset_index()
            daily_sum['catchment_avg'] = daily_sum['weighted_value'] / catchment_area
            return daily_sum[['date', 'catchment_avg']]

        icemelt_avg = calculate_catchment_average_efficient(icemelt_df, area_dict)
        snowmelt_avg = calculate_catchment_average_efficient(snowmelt_df, area_dict)
        output_avg = calculate_catchment_average_efficient(output_df, area_dict)

        # Combine dataframes efficiently
        result_df = pd.merge(icemelt_avg, snowmelt_avg, on='date', suffixes=('_icemelt', '_snowmelt'))
        result_df = pd.merge(result_df, output_avg, on='date')
        result_df.columns = ['date', 'glacier_melt', 'snowmelt', 'total_output']

        # Apply scaling factor
        scaling_factor = percentage / 100
        scaled_df = result_df.copy()
        scaled_df['glacier_melt'] = scaled_df['glacier_melt'] * scaling_factor
        scaled_df['snowmelt'] = scaled_df['snowmelt'] * scaling_factor
        scaled_df['total_output'] = scaled_df['total_output'] * scaling_factor

        # Convert units if necessary
        if self.output_unit == 'm3':
            catchment_area_m2 = total_area * 1000000
            scaled_df['glacier_melt'] = scaled_df['glacier_melt'] * catchment_area_m2 / 1000
            scaled_df['snowmelt'] = scaled_df['snowmelt'] * catchment_area_m2 / 1000
            scaled_df['total_output'] = scaled_df['total_output'] * catchment_area_m2 / 1000

        # ✅ CRITICAL FIX: CREATE INDIVIDUAL COMPONENT CSV FILES BEFORE CACHING
        self.logger.info("Creating individual component CSV files...")
        
        # Create output directory
        topo_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output file paths for individual components
        output_files = {
            'icemelt': topo_dir / 'GloGEM_icemelt.csv',
            'snowmelt': topo_dir / 'GloGEM_snowmelt.csv',
            'output': topo_dir / 'GloGEM_melt.csv',  # Keep original name for backward compatibility
        }
        
        # Create individual component CSV files
        component_dfs = {
            'icemelt': icemelt_df,
            'snowmelt': snowmelt_df,
            'output': output_df
        }
        
        for component, df in component_dfs.items():
            if not df.empty:
                # Create final records for this component
                final_records = []
                for _, row in df.iterrows():
                    final_records.append({
                        'id': row['glacier_id'],
                        'date': row['date'],
                        'q': row['value']
                    })
                
                # Create DataFrame and save to CSV
                if final_records:
                    final_df = pd.DataFrame(final_records)
                    final_df = final_df.sort_values(['id', 'date']).reset_index(drop=True)
                    
                    # Save the component CSV file
                    final_df.to_csv(output_files[component], index=False)
                    
                    self.logger.info(f"✅ {component.capitalize()} file created: {output_files[component].name} ({len(final_df)} records)")
                    
                    if not final_df.empty:
                        num_glaciers = final_df['id'].nunique()
                        date_range_str = f"{final_df['date'].min()} to {final_df['date'].max()}"
                        self.logger.info(f"   Found {num_glaciers} individual glaciers, Date range: {date_range_str}")
                else:
                    self.logger.warning(f"No records found for {component} component")
            else:
                self.logger.warning(f"No data found for {component} component")

        # Cache the final result
        with open(cache_files['final_result'], 'wb') as f:
            pickle.dump(scaled_df, f, protocol=pickle.HIGHEST_PROTOCOL)

        if plot:
            self.create_glogem_plots(scaled_df)
        return scaled_df
    
    def _process_hru(self, hru_id, grid_weights_np, array3d, time_dim):
        """
        Process a single HRU by applying grid weights
        
        Parameters
        ----------
        hru_id : int
            ID of the HRU to process
        grid_weights_np : np.ndarray
            Grid weights data
        array3d : np.ndarray
            3D array of data values
        time_dim : int
            Size of the time dimension
            
        Returns
        -------
        tuple
            HRU ID and corresponding values
        """
        # Filter rows for the current HRU
        filtered_rows = grid_weights_np[grid_weights_np[:, 0] == hru_id]
        
        # If no grid weights found for this HRU, return zeros
        if len(filtered_rows) == 0:
            return hru_id, np.zeros(time_dim, dtype=np.float32)
            
        cell_ids = filtered_rows[:, 1].astype(int)  # Flattened cell IDs
        weights = filtered_rows[:, 2]               # Relative areas as weights
        
        # Pre-allocate HRU values
        hru_values = np.zeros(time_dim, dtype=np.float32)
        
        # Handle 2D array case (time x cells) - common in NetCDF files with flattened spatial dimensions
        if len(array3d.shape) == 2:
            # In this case, we can directly index without unraveling
            for t in range(time_dim):
                # Make sure all cell_ids are within range
                valid_indices = cell_ids < array3d.shape[1]
                if not all(valid_indices):
                    self.logger.warning(f"Some cell IDs for HRU {hru_id} are out of range. Shape: {array3d.shape}")
                    cell_ids = cell_ids[valid_indices]
                    weights = weights[valid_indices]
                    
                if len(cell_ids) == 0:
                    continue
                    
                # Direct indexing for 2D array
                cell_values = array3d[t, cell_ids]
                hru_values[t] = np.sum(cell_values * weights)
        
        # Handle 3D array case (time x rows x cols)
        elif len(array3d.shape) >= 3:
            try:
                # Try to unravel indices for 3D array
                row_indices, col_indices = np.unravel_index(cell_ids, array3d.shape[1:])
                
                # Loop over time dimension
                for t in range(time_dim):
                    cell_values = array3d[t, row_indices, col_indices]
                    hru_values[t] = np.sum(cell_values * weights)
                    
            except ValueError as e:
                self.logger.error(f"Error unraveling indices for HRU {hru_id}: {e}")
                self.logger.error(f"Array shape: {array3d.shape}, Cell IDs: {cell_ids}")
                # Return zeros for this HRU
                return hru_id, np.zeros(time_dim, dtype=np.float32)
        
        # Handle any other array dimensionality
        else:
            self.logger.error(f"Unsupported array shape {array3d.shape} for HRU {hru_id}")
            return hru_id, np.zeros(time_dim, dtype=np.float32)
        
        return hru_id, hru_values
    
    def prepare_precip_coupled(self, force_reprocess: bool = False) -> pd.DataFrame:
            try:
                self.logger.info("Preparing coupled precipitation data")

                # ✅ ADD: Check if output file already exists
                out_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs')
                output_path = Path(out_dir, 'era5_land_precip_coupled.nc')
                
                if output_path.exists() and not force_reprocess:
                    self.logger.info(f"✅ Coupled precipitation file already exists: {output_path}")
                    self.logger.info("⏭️ Skipping processing. Set force_reprocess=True to reprocess.")
                    return pd.DataFrame()
                
                # Check if required fields are available
                required_fields = ['model_type', 'start_date', 'end_date']
                for field in required_fields:
                    if not getattr(self, field):
                        raise ValueError(f"Missing required field for coupled forcing: {field}")
                
                # ✅ Generate the full date range for the desired period FIRST
                full_date_range = pd.date_range(self.start_date, self.end_date)
                self.logger.info(f"Target date range: {self.start_date} to {self.end_date} ({len(full_date_range)} days)")
                        
                # Load necessary files
                self.logger.debug("Loading shapefile")
                shapefile_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
                gdf = gpd.read_file(shapefile_path)
                gdf['ID'] = gdf.reset_index().index

                self.logger.debug("Loading NetCDF file")
                out_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs')
                netcdf_path = Path(out_dir, 'era5_land_precip.nc')
                
                self.logger.debug(f"Opening NetCDF file: {netcdf_path}")
                ds = xr.open_mfdataset(netcdf_path)
                
                # Debug: Print NetCDF structure
                self.logger.debug(f"NetCDF dimensions: {ds.dims}")
                self.logger.debug(f"NetCDF variables: {list(ds.data_vars)}")
                
                # ✅ FIX: Filter NetCDF to match the desired date range
                time_coord_name = None
                for coord in ds.coords:
                    if coord.lower() in ['time', 'date']:
                        time_coord_name = coord
                        break
                
                if time_coord_name:
                    self.logger.info(f"Original NetCDF time range: {ds[time_coord_name].values[0]} to {ds[time_coord_name].values[-1]} ({len(ds[time_coord_name])} days)")
                    
                    # Filter to the desired date range
                    ds = ds.sel({time_coord_name: slice(self.start_date, self.end_date)})
                    
                    self.logger.info(f"Filtered NetCDF time range: {ds[time_coord_name].values[0]} to {ds[time_coord_name].values[-1]} ({len(ds[time_coord_name])} days)")
                else:
                    self.logger.warning("Could not find time coordinate in NetCDF, proceeding without filtering")

                self.logger.debug("Loading GloGEM data")
                glogem_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'GloGEM_melt.csv')
                glogem = pd.read_csv(glogem_dir, dtype={'id': str})[['id', 'date', 'q']]
                glogem['date'] = pd.to_datetime(glogem['date'])

                self.logger.debug("Loading HRU data")
                hru_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
                HRU = gpd.read_file(hru_dir)
                HRU = HRU.sort_values(by='HRU_ID').reset_index(drop=True)
                HRU['HRU ID'] = range(1, len(HRU) + 1)
                
                # Auto-detect RGI region code
                rgi_region_code = None
                if 'Glacier_Cl' in HRU.columns:
                    glacier_series = HRU['Glacier_Cl'].dropna()
                    if not glacier_series.empty:
                        for glacier_id in glacier_series.unique():
                            if isinstance(glacier_id, str) and glacier_id.startswith('RGI60-'):
                                parts = glacier_id.split('.')
                                if len(parts) >= 2:
                                    rgi_region_code = parts[0]
                                    break
                        self.logger.info(f"Auto-detected RGI region code: {rgi_region_code}")

                # Import GridWeightsGenerator
                from preprocess_meteo import GridWeightsGenerator
                
                self.logger.debug("Generating grid weights")
                grid_weights_generator = GridWeightsGenerator(self.namelist_path)
                grid_weights = grid_weights_generator.generate(use_precipitation=True)
                
                # Extract relevant arrays
                grid_weights_np = grid_weights[['HRU ID', 'cell_id', 'normalized_relative_area']].to_numpy()
                
                # Get variable names
                var_names = list(ds.data_vars.keys())
                self.logger.debug(f"NetCDF variables: {var_names}")
                
                if len(var_names) == 0:
                    raise ValueError("No data variables found in NetCDF file")
                
                # Find precipitation variable
                era5_precip_vars = ['tp', 'precip', 'precipitation', 'total_precipitation']
                var_name = None
                for era5_var in era5_precip_vars:
                    if era5_var in var_names:
                        var_name = era5_var
                        break
                
                if var_name is None:
                    non_coord_vars = [v for v in var_names if not any(coord in v.lower() for coord in ['coord', 'crs', 'lv95'])]
                    var_name = non_coord_vars[0] if non_coord_vars else var_names[0]
                
                self.logger.debug(f"Using variable: {var_name}")
                
                # Extract data
                array3d = ds[var_name].values
                self.logger.debug(f"Array shape: {array3d.shape}")
                
                if len(array3d.shape) == 1:
                    self.logger.warning(f"Array is 1D with shape {array3d.shape}. Reshaping to 2D.")
                    array3d = array3d.reshape(-1, 1)
                
                time_dim = array3d.shape[0]
                
                # ✅ CRITICAL CHECK: Verify time dimensions match
                if time_dim != len(full_date_range):
                    self.logger.error(f"Time dimension mismatch! NetCDF: {time_dim} days, Expected: {len(full_date_range)} days")
                    raise ValueError(f"NetCDF time dimension ({time_dim}) does not match desired date range ({len(full_date_range)})")
                
                # Process HRUs
                self.logger.debug("Calculating precipitation for each HRU")
                hru_range = range(1, max(grid_weights['HRU ID']) + 1)
                results = []
                
                for hru_id in hru_range:
                    self.logger.debug(f"Processing HRU {hru_id}")
                    try:
                        hru_result = self._process_hru(hru_id, grid_weights_np, array3d, time_dim)
                        results.append(hru_result)
                    except Exception as e:
                        self.logger.error(f"Error processing HRU {hru_id}: {str(e)}")
                        results.append((hru_id, np.zeros(time_dim, dtype=np.float32)))
                
                result_df = pd.DataFrame({hru_id: values for hru_id, values in results})
                
                # ✅ NOW THE LENGTHS MATCH: both result_df and filtered_glogem are len(full_date_range)
                # Process GloGEM data for glacier HRUs
                unique_ids = glogem['id'].unique()
                
                for glogem_id in unique_ids:
                    full_glacier_id = f"{rgi_region_code}.{glogem_id}" if rgi_region_code else str(glogem_id)
                    mask = HRU['Glacier_Cl'].notna() & (HRU['Glacier_Cl'] == full_glacier_id)
                    
                    if not mask.any():
                        self.logger.warning(f"Warning: Glacier ID {full_glacier_id} from GloGEM not found in HRU data")
                        continue
                    
                    # Filter and reindex GloGEM data
                    filtered_glogem = glogem[glogem['id'] == glogem_id].copy()
                    filtered_glogem['date'] = pd.to_datetime(filtered_glogem['date'])
                    filtered_glogem = filtered_glogem.set_index('date').reindex(full_date_range, fill_value=0).reset_index()
                    filtered_glogem.rename(columns={'index': 'date'}, inplace=True)
                    
                    hru_id = HRU.loc[mask, 'HRU ID'].iloc[0]
                    
                    # This should now work because lengths match
                    result_df[hru_id] = filtered_glogem['q'].values
                
                # Create time index
                time_index = full_date_range
                
                # Format to numpy
                result_array = result_df.to_numpy() 
                
                # Create xarray Dataset
                x_values = np.arange(1, result_array.shape[1] + 1)
                y_values = np.arange(1, 2)
                
                ds_new = xr.Dataset(
                    {'data': (['time', 'x', 'y'], result_array.reshape(len(time_index), -1, 1))},
                    coords={'time': time_index, 'x': x_values, 'y': y_values}
                )
                
                # Add elevation
                elevation_values = HRU['Elev_Mean'].values
                elevation_da = xr.DataArray(
                    elevation_values.reshape(-1, 1),
                    dims=['x', 'y'], coords={'x': ds_new['x'], 'y': ds_new['y']}
                )
                ds_new['elevation'] = elevation_da
                
                # Save
                output_path = Path(out_dir, 'era5_land_precip_coupled.nc')
                ds_new.to_netcdf(output_path)
                self.logger.info(f"Saved coupled precipitation data to {output_path}")

                return result_df
            
            except Exception as e:
                self.logger.error(f"Error in prepare_precip_coupled: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise

    def prepare_temperature_coupled(self, force_reprocess: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Prepare temperature data coupled with glacier information for hydrological modeling.
        
        Returns
        -------
        dict
            Dictionary containing temperature dataframes for each temperature variable
        """
        try:
            self.logger.info("Preparing coupled temperature data")

            # ✅ ADD: Check if all output files already exist
            out_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs')
            expected_files = [
                out_dir / 'era5_land_temp_mean_coupled.nc',
                out_dir / 'era5_land_temp_max_coupled.nc',
                out_dir / 'era5_land_temp_min_coupled.nc'
            ]
            
            all_exist = all(f.exists() for f in expected_files)
            
            if all_exist and not force_reprocess:
                self.logger.info(f"✅ All coupled temperature files already exist")
                self.logger.info("⏭️ Skipping processing. Set force_reprocess=True to reprocess.")
                return {}
            
            # Check if required fields are available
            required_fields = ['model_type', 'start_date', 'end_date']
            for field in required_fields:
                if not getattr(self, field):
                    raise ValueError(f"Missing required field for coupled forcing: {field}")
            
            # ✅ Generate the full date range for the desired period FIRST
            full_date_range = pd.date_range(self.start_date, self.end_date)
            self.logger.info(f"Target date range: {self.start_date} to {self.end_date} ({len(full_date_range)} days)")
                    
            # Load necessary files
            shapefile_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
            gdf = gpd.read_file(shapefile_path)
            
            # Auto-detect RGI region code from glacier IDs in the shapefile
            rgi_region_code = None
            if 'Glacier_Cl' in gdf.columns:
                glacier_series = gdf['Glacier_Cl'].dropna()
                if not glacier_series.empty:
                    for glacier_id in glacier_series.unique():
                        if isinstance(glacier_id, str) and glacier_id.startswith('RGI60-'):
                            parts = glacier_id.split('.')
                            if len(parts) >= 2:
                                rgi_region_code = parts[0]
                                break
                    self.logger.info(f"Auto-detected RGI region code: {rgi_region_code}")
            gdf['ID'] = gdf.reset_index().index

            glogem_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'GloGEM_melt.csv')
            glogem = pd.read_csv(glogem_dir, dtype={'id': str})[['id', 'date', 'q']]
            glogem['date'] = pd.to_datetime(glogem['date'])

            hru_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
            HRU = gpd.read_file(hru_dir)
            HRU = HRU.sort_values(by='HRU_ID').reset_index(drop=True)
            HRU['HRU ID'] = range(1, len(HRU) + 1)

            # Import GridWeightsGenerator from preprocess_meteo
            from preprocess_meteo import GridWeightsGenerator
            
            # Generate grid weights
            self.logger.debug("Generating grid weights")
            grid_weights_generator = GridWeightsGenerator(self.namelist_path)
            grid_weights = grid_weights_generator.generate(use_precipitation=True)

            # List of ERA5-Land temperature files to process
            file_names = ['era5_land_temp_mean.nc', 'era5_land_temp_max.nc', 'era5_land_temp_min.nc']
            temp_data = {}

            # Process each temperature file
            for file_name in file_names:
                self.logger.info(f"Processing temperature file: {file_name}")
                netcdf_path = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs', file_name)
                
                # Check if file exists, skip if not
                if not netcdf_path.exists():
                    self.logger.warning(f"Temperature file not found: {netcdf_path}, skipping...")
                    continue
                    
                ds = xr.open_mfdataset(netcdf_path)
                
                # Debug: Print NetCDF structure
                self.logger.debug(f"NetCDF dimensions: {ds.dims}")
                self.logger.debug(f"NetCDF variables: {list(ds.data_vars)}")
                
                # ✅ FIX: Filter NetCDF to match the desired date range
                time_coord_name = None
                for coord in ds.coords:
                    if coord.lower() in ['time', 'date']:
                        time_coord_name = coord
                        break
                
                if time_coord_name:
                    self.logger.info(f"Original NetCDF time range: {ds[time_coord_name].values[0]} to {ds[time_coord_name].values[-1]} ({len(ds[time_coord_name])} days)")
                    
                    # Filter to the desired date range
                    ds = ds.sel({time_coord_name: slice(self.start_date, self.end_date)})
                    
                    self.logger.info(f"Filtered NetCDF time range: {ds[time_coord_name].values[0]} to {ds[time_coord_name].values[-1]} ({len(ds[time_coord_name])} days)")
                else:
                    self.logger.warning("Could not find time coordinate in NetCDF, proceeding without filtering")
                
                # Get variable names and extract data
                var_names = list(ds.data_vars.keys())
                self.logger.debug(f"NetCDF variables: {var_names}")
                
                if len(var_names) == 0:
                    raise ValueError("No data variables found in NetCDF file")
                    
                # Look for ERA5-Land temperature variables
                if len(var_names) == 1:
                    var_name = var_names[0]
                else:
                    era5_temp_vars = ['t2m', 'temp', 'temperature', '2m_temperature']
                    var_name = None
                    for era5_var in era5_temp_vars:
                        if era5_var in var_names:
                            var_name = era5_var
                            break
                    
                    if var_name is None:
                        non_coord_vars = [v for v in var_names if not any(coord in v.lower() for coord in ['coord', 'crs', 'lv95'])]
                        if non_coord_vars:
                            var_name = non_coord_vars[0]
                        else:
                            var_name = var_names[0]
                
                self.logger.debug(f"Using variable: {var_name}")
                
                # Extract the data
                array3d = ds[var_name].values
                self.logger.debug(f"Array shape: {array3d.shape}")
                
                # Special handling for 1D or 2D arrays
                if len(array3d.shape) == 1:
                    self.logger.warning(f"Array is 1D with shape {array3d.shape}. Reshaping to 2D.")
                    array3d = array3d.reshape(-1, 1)
                
                time_dim = array3d.shape[0]
                
                # ✅ CRITICAL CHECK: Verify time dimensions match
                if time_dim != len(full_date_range):
                    self.logger.error(f"Time dimension mismatch! NetCDF: {time_dim} days, Expected: {len(full_date_range)} days")
                    raise ValueError(f"NetCDF time dimension ({time_dim}) does not match desired date range ({len(full_date_range)})")
                
                # Extract relevant arrays for efficiency
                grid_weights_np = grid_weights[['HRU ID', 'cell_id', 'normalized_relative_area']].to_numpy()

                # Process HRUs
                self.logger.debug(f"Processing HRUs in parallel using {max(1, multiprocessing.cpu_count() - 1)} cores")
                hru_range = range(1, max(grid_weights['HRU ID']) + 1)
                results = []
                
                for hru_id in hru_range:
                    try:
                        hru_result = self._process_hru(hru_id, grid_weights_np, array3d, time_dim)
                        results.append(hru_result)
                    except Exception as e:
                        self.logger.error(f"Error processing HRU {hru_id}: {str(e)}")
                        results.append((hru_id, np.zeros(time_dim, dtype=np.float32)))
                
                result_df = pd.DataFrame({hru_id: values for hru_id, values in results})
                temp_data[file_name] = result_df

            # ✅ NOW THE LENGTHS MATCH: Process GloGEM data for glacier HRUs
            unique_ids = glogem['id'].unique()
            for glogem_id in unique_ids:
                full_glacier_id = f"{rgi_region_code}.{glogem_id}" if rgi_region_code else str(glogem_id)
            
                mask = HRU['Glacier_Cl'].notna() & (HRU['Glacier_Cl'] == full_glacier_id)
                if not mask.any():
                    self.logger.warning(f"Warning: Glacier ID {full_glacier_id} from GloGEM not found in HRU data")
                    continue
                
                hru_id = HRU.loc[mask, 'HRU ID'].iloc[0]
            
                # Assign the constant value 20 to this HRU across all time steps
                for df in temp_data.values():
                    df[hru_id] = 20
                    
            # Save results to NetCDF for each temperature variable
            for file_name, result_df in temp_data.items():
                output_name = file_name.replace('.nc', '_coupled.nc')
                out_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs')
                output_path = Path(out_dir, output_name)

                # Use the full date range as time index
                time_index = full_date_range
                result_array = result_df.to_numpy()
                
                # Create the dimension variables
                x_values = np.arange(1, result_array.shape[1] + 1)
                y_values = np.arange(1, 2)
                
                ds_new = xr.Dataset(
                    {'data': (['time', 'x', 'y'], result_array.reshape(len(time_index), -1, 1))},
                    coords={'time': time_index, 'x': x_values, 'y': y_values}
                )

                # Add elevation data
                elevation_values = HRU['Elev_Mean'].values
                elevation_da = xr.DataArray(
                    elevation_values.reshape(-1, 1),
                    dims=['x', 'y'], coords={'x': ds_new['x'], 'y': ds_new['y']}
                )
                ds_new['elevation'] = elevation_da
                
                ds_new.to_netcdf(output_path)
                self.logger.info(f"Saved coupled temperature data to {output_path}")

                # Create plot if debug mode
                if self.debug:
                    self._plot_netcdf_first_timestep(ds_new, title=f"Temperature {file_name}")

            return temp_data
            
        except Exception as e:
            self.logger.error(f"Error in prepare_temperature_coupled: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def coupled_grid_weights_file(self) -> None:
        """
        Create grid weights file for coupled forcing
        """
        self.logger.info("Creating grid weights file for coupled forcing")
        
        # Check if required fields are available
        if not self.model_type:
            raise ValueError("model_type is required for creating grid weights file")
        
        # CHANGE: open ERA5-Land netcdf to get amount of cells
        netcdf_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs', 'era5_land_temp_mean_coupled.nc')
        ds = xr.open_mfdataset(netcdf_dir)
    
        # list all parameters that are needed for gridweights file 
        number_HRUs = len(ds['x'])
        number_cells = len(ds['x'])
    
        HRU_list = list(range(1, len(ds['x'])+1))
        cell_id = list(range(0, len(ds['x'])))
        rel_area = np.tile(1, len(ds['x']))
        out_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs')
        filename = Path(out_dir, 'GridWeights.txt')
        
        # The following section has been adapted from Juliane Mai's derive_grid_weights.py script
        with open(filename, 'w') as ff:
            ff.write('# ---------------------------------------------- \n')
            ff.write('# Raven Input file                               \n')
            ff.write('# ---------------------------------------------- \n')
            ff.write('\n')
            ff.write(':GridWeights                     \n')
            ff.write('   #                                \n')
            ff.write('   # [# HRUs]                       \n')
            ff.write('   :NumberHRUs       {}            \n'.format(number_HRUs))
            ff.write('   :NumberGridCells       {}            \n'.format(number_cells))
            ff.write('   #                                \n')
            ff.write('   # [HRU ID] [Cell #] [w_kl]       \n')
            for i in list(range(0, len(rel_area))):
                ff.write("   {}   {}   {}\n".format(HRU_list[i], cell_id[i], rel_area[i]))
            ff.write(':EndGridWeights \n')
            
        self.logger.info(f"Grid weights file created at {filename}")

    def _plot_netcdf_first_timestep(self, ds_new, title="First Time Step of Coupled Data"):
        """
        Plot the first time step of a NetCDF dataset
        
        Parameters
        ----------
        ds_new : xr.Dataset
            The NetCDF dataset
        title : str
            Plot title
        """
        try:
            # Extract the data variable
            data_variable = ds_new['data']
            
            # Select the first time step
            first_time_step = data_variable.isel(time=0)
            
            # Extract values as array and check dimensions
            values = first_time_step.values.squeeze()
            self.logger.debug(f"Data shape after squeeze: {values.shape}")
            
            # Get time info for title
            time_str = pd.to_datetime(ds_new.time.values[0]).strftime('%Y-%m-%d')
            
            # Create a colormap with distinct values for HRUs
            cmap = plt.cm.viridis
            
            # Create a bar chart (works for both 1D and 2D data)
            fig, ax = plt.subplots(figsize=(max(12, len(values) * 0.3), 6))  # ✅ Dynamic width
            
            # If data is 1D, use it directly
            if len(values.shape) == 1:
                hru_ids = np.arange(1, len(values) + 1)
                bars = ax.bar(hru_ids, values, color=cmap(values/max(values) if max(values) > 0 else values))
            # If data is 2D, flatten it
            elif len(values.shape) == 2:
                flattened_values = values.flatten()
                hru_ids = np.arange(1, len(flattened_values) + 1)
                bars = ax.bar(hru_ids, flattened_values, 
                            color=cmap(flattened_values/max(flattened_values) if max(flattened_values) > 0 else flattened_values))
            else:
                self.logger.warning(f"Unexpected data shape: {values.shape}")
                return
                
            # Add labels and title
            ax.set_xlabel("HRU ID")
            ax.set_ylabel("Value")
            ax.set_title(f"{title} - Bar Chart View\n{time_str}", fontsize=14)
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Adjust x-axis for better readability
            if len(hru_ids) > 10:
                plt.xticks(np.arange(1, len(hru_ids) + 1, step=max(1, len(hru_ids) // 10)))
            
            plt.tight_layout()
            plt.show()
            
            # ✅ IMPROVED: Create a better heatmap visualization for 1D data
            if len(values.shape) == 1:
                # Create a grid layout for better visualization
                n_hrus = len(values)
                
                # Calculate optimal grid dimensions (try to make it roughly square)
                n_cols = int(np.ceil(np.sqrt(n_hrus)))
                n_rows = int(np.ceil(n_hrus / n_cols))
                
                # Create 2D grid from 1D data
                grid_data = np.full((n_rows, n_cols), np.nan)
                for i, val in enumerate(values):
                    row = i // n_cols
                    col = i % n_cols
                    grid_data[row, col] = val
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(max(8, n_cols * 0.8), max(6, n_rows * 0.6)))
                im = ax.imshow(grid_data, cmap=cmap, interpolation='nearest')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Value')
                
                # Add HRU ID annotations
                for i in range(n_rows):
                    for j in range(n_cols):
                        hru_id = i * n_cols + j + 1
                        if hru_id <= n_hrus:
                            ax.text(j, i, str(hru_id), ha='center', va='center', 
                                fontsize=8, color='white' if not np.isnan(grid_data[i, j]) else 'black')
                
                # Add titles and labels
                ax.set_title(f"{title} - Grid View (HRU IDs shown)\n{time_str}", fontsize=14)
                ax.set_xlabel("Grid Column")
                ax.set_ylabel("Grid Row")
                
                # Remove tick marks for cleaner look
                ax.set_xticks([])
                ax.set_yticks([])
                
                plt.tight_layout()
                plt.show()
                
            # Only create traditional heatmap for 2D data
            elif len(values.shape) == 2:
                # Create a heatmap for 2D data
                fig, ax = plt.subplots(figsize=(12, 8))
                im = ax.imshow(values, cmap=cmap, interpolation='nearest')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Value')
                
                # Add titles and labels
                ax.set_title(f"{title} - Heatmap View\n{time_str}", fontsize=14)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                
                plt.tight_layout()
                plt.show()
            
            # Try to create spatial plot if HRU shapefile is available
            try:
                hru_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
                hru_gdf = gpd.read_file(hru_path)
                
                # Check column names and find the HRU ID column
                hru_id_col = None
                for col_name in ['HRU ID', 'HRU_ID', 'hru_id', 'hru id', 'ID', 'id', 'HRU']:
                    if col_name in hru_gdf.columns:
                        hru_id_col = col_name
                        break
                
                if hru_id_col is not None:
                    # Create a new column with the values
                    hru_gdf['value'] = 0
                    
                    # Flatten values if needed
                    if len(values.shape) == 2:
                        values_flat = values.flatten()
                    else:
                        values_flat = values
                    
                    # Assign values to corresponding HRUs
                    for i, value in enumerate(values_flat):
                        hru_id = i + 1  # HRU IDs start from 1
                        if i < len(values_flat):
                            # Find matching HRU
                            if isinstance(hru_id, int) and hru_gdf[hru_id_col].dtype != 'int64':
                                mask = hru_gdf[hru_id_col].astype(int) == hru_id
                            else:
                                mask = hru_gdf[hru_id_col] == hru_id
                                
                            if mask.any():
                                hru_gdf.loc[mask, 'value'] = value
                    
                    # Create a spatial plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    hru_gdf.plot(column='value', cmap=cmap, legend=True, ax=ax, 
                            edgecolor='black', linewidth=0.5)
                    
                    # Add HRU ID labels if not too many HRUs
                    if len(hru_gdf) <= 50:
                        for idx, row in hru_gdf.iterrows():
                            centroid = row.geometry.centroid
                            ax.annotate(str(row[hru_id_col]), (centroid.x, centroid.y), 
                                    fontsize=8, ha='center', va='center',
                                    bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7))
                    
                    ax.set_title(f"{title} - Spatial View\n{time_str}", fontsize=14)
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    plt.tight_layout()
                    plt.show()
                else:
                    self.logger.warning("Could not find HRU ID column in shapefile")
            except Exception as e:
                self.logger.warning(f"Could not create spatial plot: {e}")
            
        except Exception as e:
            self.logger.error(f"Error in NetCDF first time step plotting: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            

    def _plot_irrigation_timeseries_and_regime(self, ds_new, title="Irrigation (GloGEM Melt)"):
        """
        Plot time series and regime for irrigation (GloGEM melt) data
        
        Parameters
        ----------
        ds_new : xr.Dataset
            The NetCDF dataset
        title : str
            Plot title
        """
        try:
            # Create plots directory
            plots_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', 'plots')
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract the data variable
            data_variable = ds_new['data']
            
            # Get time dimension
            time_values = pd.to_datetime(ds_new.time.values)
            
            # Extract values and reshape
            values = data_variable.values.squeeze()  # Shape: (time, HRUs)
            self.logger.debug(f"Data shape after squeeze: {values.shape}")
            
            if len(values.shape) == 1:
                # If somehow it's 1D, reshape to (time, 1)
                values = values.reshape(-1, 1)
            
            # Load HRU data to get areas
            hru_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
            hru_gdf = gpd.read_file(hru_path)
            
            # ✅ Sort HRU data to match the order in the NetCDF (by HRU_ID)
            hru_gdf = hru_gdf.sort_values(by='HRU_ID').reset_index(drop=True)
            
            # Get total catchment area and glacier area
            area_col = 'Area_km2' if 'Area_km2' in hru_gdf.columns else 'area'
            total_catchment_area = hru_gdf[area_col].sum()
            
            # Identify glacier HRUs and their areas
            glacier_mask = hru_gdf['Glacier_Cl'].notna()
            glacier_hru_indices = hru_gdf[glacier_mask].index.tolist()  # Indices in the dataframe
            glacier_areas = hru_gdf.loc[glacier_mask, area_col].values  # Areas in km²
            total_glacier_area = glacier_areas.sum()
            glacier_fraction = total_glacier_area / total_catchment_area
            
            self.logger.info(f"Catchment area: {total_catchment_area:.2f} km²")
            self.logger.info(f"Glacier area: {total_glacier_area:.2f} km² ({glacier_fraction*100:.1f}% of catchment)")
            
            # ✅ Calculate area-weighted mean irrigation per day (only for glacier HRUs)
            # For each timestep, calculate: sum(irrigation_i * area_i) / sum(area_i)
            mean_irrigation_weighted = np.zeros(len(time_values))
            
            for t in range(len(time_values)):
                glacier_values = values[t, glacier_hru_indices]  # Irrigation values for glacier HRUs
                weighted_sum = np.sum(glacier_values * glacier_areas)  # mm/day * km²
                mean_irrigation_weighted[t] = weighted_sum / total_glacier_area if total_glacier_area > 0 else 0
            
            # ✅ Calculate catchment-normalized irrigation
            # This is the mean irrigation weighted by glacier fraction
            catchment_norm_irrigation = mean_irrigation_weighted * glacier_fraction
            
            # Find HRUs with non-zero irrigation (glacier HRUs)
            glacier_hrus_mask = (values != 0).any(axis=0)
            glacier_hru_ids = np.where(glacier_hrus_mask)[0] + 1  # +1 for 1-based indexing
            
            self.logger.info(f"Found {len(glacier_hru_ids)} glacier HRUs with irrigation data")
            
            # --- 1. TIME SERIES PLOT ---
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot 1a: Area-weighted mean irrigation per glacier HRU
            ax1.plot(time_values, mean_irrigation_weighted, 'b-', linewidth=1, alpha=0.7)
            ax1.fill_between(time_values, 0, mean_irrigation_weighted, color='blue', alpha=0.3)
            ax1.set_title(f'{title} - Area-Weighted Mean Irrigation per Glacier HRU\nGauge {self.gauge_id}', fontsize=14)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Mean Irrigation (mm/day)')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Add statistics text
            stats_text = (
                f"Mean: {mean_irrigation_weighted.mean():.2f} mm/day\n"
                f"Max: {mean_irrigation_weighted.max():.2f} mm/day\n"
                f"Glacier HRUs: {len(glacier_hru_ids)}\n"
                f"Total glacier area: {total_glacier_area:.2f} km²"
            )
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Plot 1b: Catchment-normalized irrigation (relative contribution)
            ax2.plot(time_values, catchment_norm_irrigation, 'orange', linewidth=1, alpha=0.7)
            ax2.fill_between(time_values, 0, catchment_norm_irrigation, color='orange', alpha=0.3)
            ax2.set_title(f'{title} - Catchment-Normalized Irrigation\n(Relative contribution to total catchment)', fontsize=14)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Catchment-Normalized Irrigation (mm/day)')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Add statistics text
            stats_text = (
                f"Mean: {catchment_norm_irrigation.mean():.2f} mm/day\n"
                f"Max: {catchment_norm_irrigation.max():.2f} mm/day\n"
                f"Glacier fraction: {glacier_fraction*100:.1f}%\n"
                f"Catchment area: {total_catchment_area:.2f} km²"
            )
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'irrigation_timeseries.png', dpi=300, bbox_inches='tight')
            if self.debug:
                plt.show()
            plt.close()
            
            self.logger.info(f"✅ Time series plot saved: {plots_dir / 'irrigation_timeseries.png'}")
            
            # --- 2. MONTHLY REGIME PLOT ---
            # Create DataFrame for easier manipulation
            df = pd.DataFrame({
                'date': time_values,
                'mean_irrigation_weighted': mean_irrigation_weighted,
                'catchment_norm_irrigation': catchment_norm_irrigation
            })
            df['month'] = df['date'].dt.month
            
            # Calculate monthly means
            monthly_mean_weighted = df.groupby('month')['mean_irrigation_weighted'].mean()
            monthly_catchment_norm = df.groupby('month')['catchment_norm_irrigation'].mean()
            
            # Create regime plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            x = range(1, 13)
            
            # Plot 2a: Area-weighted mean irrigation regime
            ax1.bar(x, monthly_mean_weighted.values, color='blue', alpha=0.7, edgecolor='black')
            ax1.plot(x, monthly_mean_weighted.values, 'b-o', linewidth=2, markersize=6)
            ax1.set_title(f'{title} - Monthly Regime (Area-Weighted Mean per Glacier HRU)\nGauge {self.gauge_id}', fontsize=14)
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Mean Irrigation (mm/day)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(month_names)
            ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Add value labels on bars
            for i, val in enumerate(monthly_mean_weighted.values, 1):
                ax1.text(i, val + 0.5, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
            
            # Plot 2b: Catchment-normalized regime
            ax2.bar(x, monthly_catchment_norm.values, color='orange', alpha=0.7, edgecolor='black')
            ax2.plot(x, monthly_catchment_norm.values, 'o-', color='darkorange', linewidth=2, markersize=6)
            ax2.set_title(f'{title} - Monthly Regime (Catchment-Normalized)\n(Relative contribution: {glacier_fraction*100:.1f}% glacier coverage)', fontsize=14)
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Catchment-Normalized Irrigation (mm/day)')
            ax2.set_xticks(x)
            ax2.set_xticklabels(month_names)
            ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Add value labels on bars
            for i, val in enumerate(monthly_catchment_norm.values, 1):
                ax2.text(i, val + 0.1, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'irrigation_regime.png', dpi=300, bbox_inches='tight')
            if self.debug:
                plt.show()
            plt.close()
            
            self.logger.info(f"✅ Regime plot saved: {plots_dir / 'irrigation_regime.png'}")
            
            # --- 3. OPTIONAL: INDIVIDUAL GLACIER HRU PLOT (if reasonable number) ---
            if len(glacier_hru_ids) > 0 and len(glacier_hru_ids) <= 20:
                fig, ax = plt.subplots(figsize=(14, 6))
                
                # Plot individual glacier HRU contributions (up to 10 for clarity)
                for i, hru_id in enumerate(glacier_hru_ids[:10]):
                    hru_idx = hru_id - 1  # Convert to 0-based indexing
                    hru_area = glacier_areas[i] if i < len(glacier_areas) else 1.0
                    ax.plot(time_values, values[:, hru_idx], 
                           alpha=0.5, linewidth=0.8, 
                           label=f'HRU {hru_id} ({hru_area:.2f} km²)')
                
                # Plot area-weighted mean
                ax.plot(time_values, mean_irrigation_weighted, 'b-', linewidth=2, 
                       label='Area-weighted mean', alpha=0.8)
                
                # Plot catchment-normalized
                ax.plot(time_values, catchment_norm_irrigation, color='orange', linewidth=2,
                       label='Catchment-normalized', alpha=0.8, linestyle='--')
                
                ax.set_title(f'{title} - Individual Glacier HRU Contributions\nGauge {self.gauge_id}', 
                           fontsize=14)
                ax.set_xlabel('Date')
                ax.set_ylabel('Irrigation (mm/day)')
                ax.legend(loc='upper right', fontsize=8, ncol=2)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'irrigation_hru_contributions.png', dpi=300, bbox_inches='tight')
                if self.debug:
                    plt.show()
                plt.close()
                
                self.logger.info(f"✅ HRU contributions plot saved: {plots_dir / 'irrigation_hru_contributions.png'}")
            
        except Exception as e:
            self.logger.error(f"Error in irrigation plotting: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _plot_time_series(self, result_df, title="Time Series by HRU"):
        """
        Plot time series data for each HRU
        
        Parameters
        ----------
        result_df : pd.DataFrame
            DataFrame containing values for each HRU
        title : str
            Plot title
        """
        try:
            # Create date range for x-axis
            dates = pd.date_range(self.start_date, self.end_date)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot each HRU time series
            for hru_id in result_df.columns:
                # Sample data points to avoid overcrowding (plot every 7th day = weekly samples)
                ax.plot(dates[::7], result_df[hru_id].values[::7], 
                        label=f"HRU {hru_id}" if hru_id <= 5 else "", 
                        alpha=0.7, linewidth=1)
            
            # Only show legend for first 5 HRUs to avoid overcrowding
            if len(result_df.columns) > 5:
                ax.plot([], [], label=f"+ {len(result_df.columns)-5} more HRUs", linestyle='')
                
            ax.legend(loc='upper right')
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            
            # Format x-axis to show dates nicely
            fig.autofmt_xdate()
            
            # Show grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            self.logger.error(f"Error in time series plotting: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def validate_glacier_ids(self) -> Dict[str, List[str]]:
        """
        Validate that glacier IDs in HRU shapefile match those in GloGEM data.
        """
        self.logger.info("Validating glacier IDs between HRU and GloGEM data")
        
        results = {
            'missing_in_glogem': [],
            'missing_in_hru': [],
            'matched': []
        }
        
        try:
            # Load HRU data
            hru_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
            if not hru_path.exists():
                self.logger.error(f"HRU shapefile not found at {hru_path}")
                return results
                
            hru_gdf = gpd.read_file(hru_path)
            
            # Extract glacier IDs from HRU data
            glacier_ids_hru = []
            glacier_ids_hru_numeric = []  # ✅ Initialize this variable
            rgi_region_code = None  # Auto-detect region code
            
            if 'Glacier_Cl' in hru_gdf.columns:
                # Filter out NaN values first
                glacier_series = hru_gdf['Glacier_Cl'].dropna()
                if not glacier_series.empty:
                    # Auto-detect RGI region code from the first glacier ID
                    for glacier_id in glacier_series.unique():
                        if isinstance(glacier_id, str) and glacier_id.startswith('RGI60-'):
                            parts = glacier_id.split('.')
                            if len(parts) >= 2:
                                rgi_region_code = parts[0]  # e.g., 'RGI60-15'
                                break
                    
                    self.logger.info(f"Auto-detected RGI region code: {rgi_region_code}")
                    
                    # Convert to standard format (e.g., RGI60-15.xxxxx)
                    glacier_ids_hru = glacier_series.unique().tolist()
                    # Also store the numeric part for comparison with GloGEM
                    for g_id in glacier_ids_hru:
                        if isinstance(g_id, str) and rgi_region_code and f'{rgi_region_code}.' in g_id:
                            glacier_ids_hru_numeric.append(g_id.replace(f'{rgi_region_code}.', ''))
                        else:
                            self.logger.warning(f"Unexpected glacier ID format in HRU: {g_id}")
            
            # Directly read the GloGEM file from the known path
            glogem_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'GloGEM_melt.csv')
            
            if glogem_path.exists():
                # Read the processed CSV file
                glogem_data = pd.read_csv(glogem_path, dtype={'id': str})
                glacier_ids_glogem = glogem_data['id'].unique().tolist()
            else:
                # Fall back to the original DAT file if needed
                self.logger.warning(f"Processed GloGEM file not found at {glogem_path}, reading from original DAT file")
                glogem_data = self.get_dat_file()
                glacier_ids_glogem = glogem_data['id'].unique().astype(str).tolist()
            
            # Compare the sets of IDs
            glogem_ids_set = set(glacier_ids_glogem)
            hru_ids_set = set(glacier_ids_hru_numeric)
            
            # Find mismatches
            missing_in_glogem = hru_ids_set - glogem_ids_set
            missing_in_hru = glogem_ids_set - hru_ids_set
            matched = hru_ids_set.intersection(glogem_ids_set)
            
            # Store results with full IDs for better clarity
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
            self.logger.info(f"Found {len(results['matched'])} matched glacier IDs")
            
            if results['missing_in_glogem']:
                self.logger.warning(f"{len(results['missing_in_glogem'])} glacier IDs in HRU are missing from GloGEM data")
                for g_id in results['missing_in_glogem'][:5]:  # Show first 5 only
                    self.logger.warning(f"  - {g_id}")
                if len(results['missing_in_glogem']) > 5:
                    self.logger.warning(f"  - ... and {len(results['missing_in_glogem'])-5} more")
            
            if results['missing_in_hru']:
                self.logger.warning(f"{len(results['missing_in_hru'])} glacier IDs in GloGEM are missing from HRU")
                for g_id in results['missing_in_hru'][:5]:  # Show first 5 only
                    self.logger.warning(f"  - {g_id}")
                if len(results['missing_in_hru']) > 5:
                    self.logger.warning(f"  - ... and {len(results['missing_in_hru'])-5} more")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error validating glacier IDs: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return results
        
    def validate_precipitation_files(self) -> None:
        """
        Create validation plots for the precipitation files to confirm the GloGEM 
        data is properly integrated.
        
        This function generates time series plots showing:
        1. Daily catchment average precipitation
        2. Monthly average precipitation
        3. Annual total precipitation
        
        The plots help verify that glacier melt has been correctly incorporated.
        """
        self.logger.info("Validating precipitation file modifications")
        
        try:
            # Define paths for original and modified precipitation files
            orig_precip_file = Path(self.model_dir, f'catchment_{self.gauge_id}', 
                                self.model_type, 'data_obs', 'era5_land_precip.nc')
            new_precip_file = Path(self.model_dir, f'catchment_{self.gauge_id}', 
                                self.model_type, 'data_obs', 'era5_land_precip_coupled.nc')
            
            # Check if both files exist
            if not orig_precip_file.exists():
                self.logger.warning(f"Original precipitation file not found at {orig_precip_file}")
                return
                
            if not new_precip_file.exists():
                self.logger.warning(f"Modified precipitation file not found at {new_precip_file}")
                return
                
            # Create plots directory
            plots_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', 'plots')
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Load datasets with more detailed logging
            self.logger.debug("Loading precipitation datasets")
            ds_orig = xr.open_dataset(orig_precip_file)
            ds_new = xr.open_dataset(new_precip_file)
            
            # Extract variables
            orig_var_names = list(ds_orig.data_vars)
            new_var_names = list(ds_new.data_vars)

            self.logger.debug(f"Original dataset variables: {orig_var_names}")
            self.logger.debug(f"New dataset variables: {new_var_names}")

            # Find the right precipitation variable in the original dataset
            orig_var_name = None
            for var_name in orig_var_names:
                # Check if this variable has dimensions
                if ds_orig[var_name].dims:
                    # Look for the variable that has a time dimension
                    for dim in ds_orig[var_name].dims:
                        if dim.lower() in ['time', 'date']:
                            orig_var_name = var_name
                            break
                    if orig_var_name:
                        break
                        
            if not orig_var_name:
                self.logger.error("Could not find a suitable variable with time dimension in original dataset")
                # Try to find any variable with dimensions as fallback
                for var_name in orig_var_names:
                    if ds_orig[var_name].dims:
                        orig_var_name = var_name
                        self.logger.warning(f"Using {orig_var_name} as fallback, but it may not have time dimension")
                        break
                        
                if not orig_var_name:
                    self.logger.error("No variable with dimensions found in original dataset")
                    return

            # For the new dataset, the first variable is likely correct, but let's verify
            new_var_name = new_var_names[0]
            for var_name in new_var_names:
                # Look for variable that has dimensions
                if ds_new[var_name].dims:
                    # Check if it has a time dimension
                    for dim in ds_new[var_name].dims:
                        if dim.lower() in ['time', 'date']:
                            new_var_name = var_name
                            break
                    if new_var_name:
                        break
            
            # More detailed debug info
            self.logger.debug(f"Original dataset dimensions: {ds_orig.dims}")
            self.logger.debug(f"Selected original dataset variable: {orig_var_name}")
            self.logger.debug(f"Original dataset variable dims: {ds_orig[orig_var_name].dims}")
            self.logger.debug(f"Original dataset variable shape: {ds_orig[orig_var_name].shape}")
            
            self.logger.debug(f"New dataset dimensions: {ds_new.dims}")
            self.logger.debug(f"Selected new dataset variable: {new_var_name}")
            self.logger.debug(f"New dataset variable dims: {ds_new[new_var_name].dims}")
            self.logger.debug(f"New dataset variable shape: {ds_new[new_var_name].shape}")
            
            # Get time dimension
            time_dim_orig = None
            time_dim_new = None
            
            for dim in ds_orig[orig_var_name].dims:
                if dim.lower() in ['time', 'date']:
                    time_dim_orig = dim
                    break
                    
            for dim in ds_new[new_var_name].dims:
                if dim.lower() in ['time', 'date']:
                    time_dim_new = dim
                    break
            
            if not time_dim_orig:
                self.logger.warning(f"Could not find time dimension in original dataset. Available dims: {ds_orig[orig_var_name].dims}")
                return
                
            if not time_dim_new:
                self.logger.warning(f"Could not find time dimension in new dataset. Available dims: {ds_new[new_var_name].dims}")
                return
            
            # Get spatial dimensions
            orig_spatial_dims = [dim for dim in ds_orig[orig_var_name].dims if dim != time_dim_orig]
            new_spatial_dims = [dim for dim in ds_new[new_var_name].dims if dim != time_dim_new]
            
            self.logger.debug(f"Original spatial dimensions: {orig_spatial_dims}")
            self.logger.debug(f"New spatial dimensions: {new_spatial_dims}")
            
            # Calculate catchment average time series with more careful handling
            self.logger.debug("Calculating catchment averages")
            
            # Process original dataset
            try:
                orig_time_series = ds_orig[orig_var_name].mean(dim=orig_spatial_dims, skipna=True)
                self.logger.debug(f"Original time series shape: {orig_time_series.shape}")
                
                # Check if result is a time series
                if time_dim_orig not in orig_time_series.dims:
                    self.logger.warning(f"Time dimension {time_dim_orig} missing in result. This is unexpected.")
                    # Try a different approach - explicitly preserve time
                    orig_time_series = ds_orig[orig_var_name].mean(dim=orig_spatial_dims, skipna=True, keep_attrs=True)
                    
                # Convert to pandas Series manually with more checks
                time_values = ds_orig[time_dim_orig].values
                self.logger.debug(f"Original time values shape: {time_values.shape}")
                self.logger.debug(f"First few time values: {time_values[:5]}")
                
                if len(orig_time_series.shape) == 0:  # 0-dimensional
                    self.logger.warning("Original time series is 0-dimensional, using a constant value")
                    # Create a constant series with the same time index
                    orig_series = pd.Series(
                        data=[float(orig_time_series.values)] * len(time_values),
                        index=time_values
                    )
                else:
                    orig_series = pd.Series(
                        data=orig_time_series.values,
                        index=time_values
                    )
            except Exception as e:
                self.logger.error(f"Error processing original dataset: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                orig_series = None
            
            # Process new dataset
            try:
                new_time_series = ds_new[new_var_name].mean(dim=new_spatial_dims, skipna=True)
                self.logger.debug(f"New time series shape: {new_time_series.shape}")
                
                # Check if result is a time series
                if time_dim_new not in new_time_series.dims:
                    self.logger.warning(f"Time dimension {time_dim_new} missing in result. This is unexpected.")
                    # Try a different approach - explicitly preserve time
                    new_time_series = ds_new[new_var_name].mean(dim=new_spatial_dims, skipna=True, keep_attrs=True)
                    
                # Convert to pandas Series manually with more checks
                time_values = ds_new[time_dim_new].values
                self.logger.debug(f"New time values shape: {time_values.shape}")
                self.logger.debug(f"First few time values: {time_values[:5]}")
                
                if len(new_time_series.shape) == 0:  # 0-dimensional
                    self.logger.warning("New time series is 0-dimensional, using a constant value")
                    # Create a constant series with the same time index
                    new_series = pd.Series(
                        data=[float(new_time_series.values)] * len(time_values),
                        index=time_values
                    )
                else:
                    new_series = pd.Series(
                        data=new_time_series.values,
                        index=time_values
                    )
            except Exception as e:
                self.logger.error(f"Error processing new dataset: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                new_series = None
            
            # Check if we have valid data to plot
            if orig_series is None and new_series is None:
                self.logger.error("Both time series extraction failed, cannot create plots")
                return
                
            if orig_series is None:
                self.logger.warning("Original time series extraction failed, will only plot modified data")
                # Create a dummy series with zeros
                orig_series = pd.Series(
                    data=np.zeros_like(new_series.values),
                    index=new_series.index
                )
                
            if new_series is None:
                self.logger.warning("Modified time series extraction failed, will only plot original data")
                # Create a dummy series with zeros
                new_series = pd.Series(
                    data=np.zeros_like(orig_series.values),
                    index=orig_series.index
                )
            
            # Ensure we have DateTimeIndex
            if not isinstance(orig_series.index, pd.DatetimeIndex):
                self.logger.warning("Converting original precipitation index to datetime")
                try:
                    orig_series.index = pd.to_datetime(orig_series.index)
                except Exception as e:
                    self.logger.warning(f"Failed to convert index to datetime: {str(e)}")
                    self.logger.warning("Using artificial time index")
                    orig_series.index = pd.date_range(self.start_date, periods=len(orig_series))
                    
            if not isinstance(new_series.index, pd.DatetimeIndex):
                self.logger.warning("Converting new precipitation index to datetime")
                try:
                    new_series.index = pd.to_datetime(new_series.index)
                except Exception as e:
                    self.logger.warning(f"Failed to convert index to datetime: {str(e)}")
                    self.logger.warning("Using artificial time index")
                    new_series.index = pd.date_range(self.start_date, periods=len(new_series))
            
            # 1. Daily time series comparison (sampled to avoid overcrowding)
            self.logger.debug("Creating daily time series plot")
            plt.figure(figsize=(14, 6))
            
            # Sample every 7 days for clearer visualization
            sample_freq = 7
            plt.plot(orig_series.index[::sample_freq], orig_series.values[::sample_freq], 
                    'b-', alpha=0.7, label='Original Precipitation', linewidth=1)
            plt.plot(new_series.index[::sample_freq], new_series.values[::sample_freq], 
                    'r-', alpha=0.7, label='With Glacier Melt', linewidth=1)
            
            plt.title(f'Catchment Average Precipitation - Original vs. Modified (Gauge {self.gauge_id})')
            plt.xlabel('Date')
            plt.ylabel('Precipitation (mm/day)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Format x-axis for better readability
            plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'precip_timeseries_comparison.png', dpi=300)
            if self.debug:
                plt.show()
            plt.close()
            
            # 2. Monthly average comparison
            self.logger.debug("Creating monthly average plot")
            
            # Calculate monthly averages
            orig_monthly = orig_series.groupby(orig_series.index.month).mean()
            new_monthly = new_series.groupby(new_series.index.month).mean()
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Set width for bars
            width = 0.35
            
            # Create month labels
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            x = np.arange(len(months))
            
            # Plot bars
            plt.bar(x - width/2, orig_monthly.values, width, label='Original Precipitation', color='blue', alpha=0.7)
            plt.bar(x + width/2, new_monthly.values, width, label='With Glacier Melt', color='red', alpha=0.7)
            
            # Add labels and legend
            plt.xlabel('Month')
            plt.ylabel('Average Precipitation (mm/day)')
            plt.title(f'Monthly Average Precipitation Comparison (Gauge {self.gauge_id})')
            plt.xticks(x, months)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Add value increase percentages
            for i, (orig, new) in enumerate(zip(orig_monthly.values, new_monthly.values)):
                if orig > 0:  # Avoid division by zero
                    pct_increase = ((new - orig) / orig) * 100
                    plt.text(i, max(orig, new) + 0.1, f"+{pct_increase:.1f}%", 
                            ha='center', va='bottom', fontsize=8, 
                            color='green' if pct_increase > 0 else 'red')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'precip_monthly_comparison.png', dpi=300)
            if self.debug:
                plt.show()
            plt.close()
            
            # 3. Annual sum comparison
            if (orig_series.index.max() - orig_series.index.min()).days > 365:
                self.logger.debug("Creating annual sum plot")
                
                # Calculate annual sums
                orig_annual = orig_series.groupby(orig_series.index.year).sum()
                new_annual = new_series.groupby(new_series.index.year).sum()
                
                # Create plot
                plt.figure(figsize=(14, 6))
                
                # Get years as x values
                years = orig_annual.index.astype(str)
                x = np.arange(len(years))
                
                # Plot bars
                plt.bar(x - width/2, orig_annual.values, width, label='Original Precipitation', color='blue', alpha=0.7)
                plt.bar(x + width/2, new_annual.values, width, label='With Glacier Melt', color='red', alpha=0.7)
                
                # Add labels and legend
                plt.xlabel('Year')
                plt.ylabel('Annual Precipitation Sum (mm/year)')
                plt.title(f'Annual Precipitation Sum Comparison (Gauge {self.gauge_id})')
                plt.xticks(x, years, rotation=45)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7, axis='y')
                
                # Add value increase percentages
                for i, (orig, new) in enumerate(zip(orig_annual.values, new_annual.values)):
                    if orig > 0:  # Avoid division by zero
                        pct_increase = ((new - orig) / orig) * 100
                        plt.text(i, max(orig, new) + 20, f"+{pct_increase:.1f}%", 
                                ha='center', va='bottom', fontsize=8, 
                                color='green' if pct_increase > 0 else 'red')
                
                # Add statistics
                stats_text = (
                    f"Original annual avg: {orig_annual.mean():.1f} mm\n"
                    f"Modified annual avg: {new_annual.mean():.1f} mm\n"
                    f"Average increase: {new_annual.mean() - orig_annual.mean():.1f} mm "
                    f"({(new_annual.mean() - orig_annual.mean()) / orig_annual.mean() * 100:.1f}%)"
                )
                plt.figtext(0.01, 0.01, stats_text, fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'precip_annual_comparison.png', dpi=300)
                if self.debug:
                    plt.show()
                plt.close()
            
            # 4. Create a glacier contribution plot
            self.logger.debug("Creating glacier contribution plot")
            
            # Calculate the difference (glacier contribution)
            diff_series = new_series - orig_series
            
            plt.figure(figsize=(14, 6))
            plt.plot(diff_series.index[::sample_freq], diff_series.values[::sample_freq], 
                    'g-', alpha=0.7, linewidth=1)
            plt.fill_between(diff_series.index[::sample_freq], 0, diff_series.values[::sample_freq], 
                            color='green', alpha=0.3)
            
            plt.title(f'Glacier Melt Contribution to Precipitation (Gauge {self.gauge_id})')
            plt.xlabel('Date')
            plt.ylabel('Glacier Contribution (mm/day)')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Format x-axis for better readability
            plt.gcf().autofmt_xdate()
            
            # Add statistics
            stats_text = (
                f"Mean contribution: {diff_series.mean():.3f} mm/day\n"
                f"Max contribution: {diff_series.max():.3f} mm/day\n"
                f"Total contribution: {diff_series.sum():.1f} mm"
            )
            plt.figtext(0.01, 0.01, stats_text, fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'glacier_contribution.png', dpi=300)
            if self.debug:
                plt.show()
            plt.close()
            
            self.logger.info("Precipitation validation plots created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating precipitation validation plots: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def plot_glacier_runoff_vs_observed(self):
        """
        Plot area-weighted average daily glacier runoff (from GloGEM melt),
        observed streamflow, and area-weighted glacier HRU precipitation from NetCDF.
        All series are shown in mm/day over the catchment area.
        """
        import matplotlib.pyplot as plt
        import xarray as xr

        # --- Load GloGEM melt data ---
        glogem_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'GloGEM_melt.csv')
        glogem_df = pd.read_csv(glogem_path)
        glogem_df['date'] = pd.to_datetime(glogem_df['date'])
        glogem_df['q'] = pd.to_numeric(glogem_df['q'], errors='coerce')
        
        # Get HRU areas
        hru_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
        hru_gdf = gpd.read_file(hru_path)
        
        if 'area' not in glogem_df.columns:
            glacier_areas = hru_gdf[['Glacier_Cl', 'Area_km2']].dropna()
            glacier_areas['id'] = glacier_areas['Glacier_Cl'].astype(str).str.split('.').str[-1]
            area_map = glacier_areas.set_index('id')['Area_km2'].to_dict()
            glogem_df['area'] = glogem_df['id'].map(area_map)
        glogem_df['area'] = pd.to_numeric(glogem_df['area'], errors='coerce')

        # --- Area-weighted glacier runoff per day (m³/s) ---
        daily = glogem_df.groupby('date').apply(
            lambda x: np.nansum(x['q'] * x['area']) / np.nansum(x['area']) if np.nansum(x['area']) > 0 else np.nan
        ).rename('glacier_runoff').reset_index()

        # --- Load observed streamflow from Q_daily.rvt ---
        q_file = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs', 'Q_daily.rvt')
        with open(q_file, 'r') as f:
            lines = f.readlines()
        start_date_line = None
        for line in lines:
            if line.strip() and not line.strip().startswith(':'):
                start_date_line = line.strip()
                break
        if start_date_line is None:
            raise ValueError("Could not find start date line in Q_daily.rvt file.")
        start_date = start_date_line.split()[0]
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
        obs_dates = pd.date_range(start=start_date, periods=len(value_lines))
        obs_series = pd.Series(value_lines, index=obs_dates, name='observed_streamflow')

        # --- Merge glacier runoff and observed streamflow ---
        merged = pd.merge(daily, obs_series.rename('observed_streamflow'), left_on='date', right_index=True, how='inner')

        # --- Load HRU shapefile and get glacier info ---
        hru_gdf = hru_gdf.sort_values(by='HRU_ID').reset_index(drop=True)
        glacier_mask = hru_gdf['Glacier_Cl'].notna()
        glacier_hru_indices = hru_gdf[glacier_mask].index.tolist()
        
        area_col = 'Area_km2' if 'Area_km2' in hru_gdf.columns else 'area'
        glacier_areas = hru_gdf.loc[glacier_mask, area_col].values
        glacier_area_km2 = np.sum(glacier_areas)
        catchment_area_km2 = hru_gdf[area_col].sum()
        glacier_fraction = glacier_area_km2 / catchment_area_km2

        # ✅ FIX: Load the COUPLED precipitation NetCDF (which has GloGEM data)
        nc_path = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs', 'era5_land_precip_coupled.nc')
        
        if not nc_path.exists():
            self.logger.warning(f"Coupled precipitation NetCDF not found at {nc_path}")
            self.logger.warning("Falling back to uncoupled precipitation")
            nc_path = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs', 'era5_land_precip.nc')
        
        ds = xr.open_dataset(nc_path)
        prec_data = ds['data']  # shape: (time, HRUs, 1)
        
        # ✅ FIX: Calculate area-weighted mean (same as irrigation regime plot)
        glacier_precip_weighted = np.zeros(prec_data.shape[0])
        
        for t in range(prec_data.shape[0]):
            # Extract values for glacier HRUs
            glacier_values = prec_data[t, glacier_hru_indices, 0].values
            # Calculate area-weighted mean
            weighted_sum = np.sum(glacier_values * glacier_areas)
            glacier_precip_weighted[t] = weighted_sum / glacier_area_km2 if glacier_area_km2 > 0 else 0
        
        # Create pandas series
        glacier_precip_series = pd.Series(
            glacier_precip_weighted,
            index=pd.to_datetime(ds['time'].values),
            name='glacier_precipitation'
        )
        
        # ✅ Normalize to catchment (same as irrigation regime plot)
        glacier_precip_series_catchment_norm = glacier_precip_series * glacier_fraction
        glacier_precip_series_catchment_norm.name = 'glacier_precipitation_catchment_norm'

        # --- Convert discharge/runoff from m³/s to mm/day over catchment ---
        def m3s_to_mmday(series, area_km2):
            return series * 86400 / (area_km2 * 1e6) * 1000

        merged['glacier_runoff_mm'] = m3s_to_mmday(merged['glacier_runoff'], catchment_area_km2)
        merged['observed_streamflow_mm'] = m3s_to_mmday(merged['observed_streamflow'], catchment_area_km2)

        # --- Merge all for plotting ---
        plot_df = merged.copy()
        plot_df = plot_df.merge(glacier_precip_series_catchment_norm, left_on='date', right_index=True, how='left')

        plots_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', 'plots_results')
        plots_dir.mkdir(parents=True, exist_ok=True)

        # --- Plot in mm/day ---
        plt.figure(figsize=(14, 6))
        plt.plot(plot_df['date'], plot_df['glacier_runoff_mm'], label='Glacier Runoff (GloGEM CSV, mm/day)', color='green', alpha=0.7)
        plt.plot(plot_df['date'], plot_df['observed_streamflow_mm'], label='Observed Streamflow (mm/day)', color='black')
        plt.plot(plot_df['date'], plot_df['glacier_precipitation_catchment_norm'], 
                label='Glacier Melt from Coupled Precip NC (Catchment-normalized, mm/day)', 
                color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        plt.title(f'Glacier Runoff (CSV vs Coupled Precip NC) and Observed Streamflow\nGauge {self.gauge_id}')
        plt.xlabel('Date')
        plt.ylabel('Discharge / Glacier Melt (mm/day)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'glacier_runoff_vs_observed_and_precip_mm.png', dpi=300)
        if self.debug:
            plt.show()
        plt.close()

        # --- Monthly regime plot ---
        regime_df = plot_df.copy()
        regime_df['month'] = regime_df['date'].dt.month
        monthly_obs = regime_df.groupby('month')['observed_streamflow_mm'].mean()
        monthly_glacier_precip = regime_df.groupby('month')['glacier_precipitation_catchment_norm'].mean()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 13), monthly_obs.values, label='Observed Streamflow (mm/day)', color='black', marker='o')
        plt.plot(range(1, 13), monthly_glacier_precip.values, 
                label='Glacier Melt (Catchment-normalized, mm/day)', 
                color='red', marker='o', linestyle='--', linewidth=2)
        
        plt.xticks(range(1, 13), month_names)
        plt.title(f'Monthly Regime: Observed Streamflow vs Glacier Melt\nGauge {self.gauge_id}')
        plt.xlabel('Month')
        plt.ylabel('Mean (mm/day)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'monthly_regime_obs_vs_glacier_melt.png', dpi=300)
        if self.debug:
            plt.show()
        plt.close()

        self.logger.info(f"✅ Glacier runoff vs observed plots saved to {plots_dir}")


    def prepare_coupled_forcing(self) -> Dict[str, Any]:
        """
        Prepare all coupled forcing data (precipitation, temperature, and irrigation)
        
        Returns
        -------
        dict
            Dictionary containing all generated datasets
        """
        self.logger.info(f"Preparing all coupled forcing data for gauge {self.gauge_id}")
        
        # Check if required fields are available
        required_fields = ['model_type', 'start_date', 'end_date']
        for field in required_fields:
            if not getattr(self, field):
                raise ValueError(f"Missing required field for coupled forcing: {field}")
                
        # First ensure we have the GloGEM time series
        if not Path(self.glogem_path).exists():
            self.logger.info("Processing GloGEM time series first")
            self.process_glogem_data_optimized(plot=False)

        # Validate glacier IDs
        validation_results = self.validate_glacier_ids()
            
        # Prepare precipitation data
        self.logger.info("Preparing precipitation data")
        precip_data = self.prepare_precip_coupled()
        
        # Prepare temperature data
        self.logger.info("Preparing temperature data")
        temp_data = self.prepare_temperature_coupled()
        
        # Prepare irrigation data (NEW)
        self.logger.info("Preparing irrigation data (GloGEM melt)")
        irrigation_data = self.prepare_irrigation_coupled()
        
        # Create grid weights file
        self.logger.info("Creating grid weights file")
        self.coupled_grid_weights_file()

        # Add validation plots for precipitation
        self.validate_precipitation_files()

        # plot the glacier melt vs streamflow
        self.plot_glacier_runoff_vs_observed()
        
        self.logger.info("All coupled forcing data prepared successfully")
        
        return {
            'precipitation': precip_data,
            'temperature': temp_data,
            'irrigation': irrigation_data  # NEW
        }