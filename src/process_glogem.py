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
        
        # Set model directory based on coupled/uncoupled
        if self.coupled:
            self.model_dir = self.main_dir / config['model_dirs']['coupled']
        else:
            self.model_dir = self.main_dir / config['model_dirs']['uncoupled']
        
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
    
    def process_glogem_data_optimized(self, force_reprocess: bool = False, chunk_size: int = 10000, plot: bool = True) -> pd.DataFrame:
        """
        Optimized version that processes large GloGEM files efficiently and caches results.
        """
        import pickle
        from datetime import datetime, timedelta

        # Setup directories and file paths
        topo_dir = Path(self.model_dir) / f"catchment_{self.gauge_id}" / "topo_files"
        glogem_dir = self.glogem_dir
        cache_dir = glogem_dir / "processed_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_files = {
            'icemelt': cache_dir / f"icemelt_{self.gauge_id}_processed.pkl",
            'snowmelt': cache_dir / f"snowmelt_{self.gauge_id}_processed.pkl", 
            'output': cache_dir / f"output_{self.gauge_id}_processed.pkl",
            'final_result': cache_dir / f"final_result_{self.gauge_id}_{self.start_date}_{self.end_date}_{self.output_unit}.pkl"
        }
        file_paths = {
            'icemelt': glogem_dir / f"GloGEM_icemelt_{self.gauge_id}.dat",
            'snowmelt': glogem_dir / f"GloGEM_snowmelt_{self.gauge_id}.dat",
            'output': glogem_dir / f"GloGEM_output_{self.gauge_id}.dat"
        }
        catchment_shape_file = topo_dir / "HRU.shp"

        # Check if final result already exists and is recent
        if cache_files['final_result'].exists() and not force_reprocess:
            self.logger.info(f"Loading cached final result from {cache_files['final_result']}")
            with open(cache_files['final_result'], 'rb') as f:
                scaled_df = pickle.load(f)
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
    
    def prepare_precip_coupled(self) -> pd.DataFrame:
        try:
            self.logger.info("Preparing coupled precipitation data")
            
            # Check if required fields are available
            required_fields = ['model_type', 'start_date', 'end_date']
            for field in required_fields:
                if not getattr(self, field):
                    raise ValueError(f"Missing required field for coupled forcing: {field}")
                    
            # Load necessary files
            self.logger.debug("Loading shapefile")
            shapefile_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
            gdf = gpd.read_file(shapefile_path)
            gdf['ID'] = gdf.reset_index().index

            self.logger.debug("Loading NetCDF file")
            out_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs')
            # CHANGE: Use ERA5-Land precipitation file instead of Meteoswiss
            netcdf_path = Path(out_dir, 'era5_land_precip.nc')
            
            # Add debug message before opening NetCDF file
            self.logger.debug(f"Opening NetCDF file: {netcdf_path}")
            ds = xr.open_mfdataset(netcdf_path)
            
            # Debug: Print NetCDF structure
            self.logger.debug(f"NetCDF dimensions: {ds.dims}")
            self.logger.debug(f"NetCDF variables: {list(ds.data_vars)}")
            self.logger.debug(f"NetCDF coordinates: {list(ds.coords)}")

            self.logger.debug("Loading GloGEM data")
            glogem_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'GloGEM_melt.csv')
            glogem = pd.read_csv(glogem_dir, dtype={'id': str})[['id', 'date', 'q']]
            glogem['date'] = pd.to_datetime(glogem['date'])

            self.logger.debug("Loading HRU data")
            hru_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
            HRU = gpd.read_file(hru_dir)
            HRU = HRU.sort_values(by='HRU_ID').reset_index(drop=True)
            HRU['HRU ID'] = range(1, len(HRU) + 1)
            
            # Print first few rows of HRU data to help diagnose issues
            self.logger.debug(f"HRU data first few rows: {HRU.head()}")
            self.logger.debug(f"HRU columns: {HRU.columns}")

            # Safely extract glacier IDs from HRU
            self.logger.debug("Extracting glacier IDs")
            glacier_ids_hru = []
            if 'Glacier_Cl' in HRU.columns:
                # First filter out NaN values
                glacier_series = HRU['Glacier_Cl'].dropna()
                # Only then apply string operations
                if not glacier_series.empty:
                    glacier_ids_hru = glacier_series.astype(str).str.replace('RGI60-14.', '').tolist()
                
                self.logger.debug(f"Found {len(glacier_ids_hru)} glaciers in HRU shapefile")
            else:
                self.logger.warning("No 'Glacier_Cl' column found in HRU shapefile")
            
            # Get GloGEM IDs
            glacier_ids_glogem = glogem['id'].unique()
            
            # Find missing glaciers (in HRU but not in GloGEM)
            missing_glaciers = set(glacier_ids_hru) - set(glacier_ids_glogem)
            
            if len(missing_glaciers) == 0:
                self.logger.info("No glaciers missing - all HRU glaciers are present in GloGEM data")
            else:
                self.logger.warning("The following glacier IDs are missing from GloGEM data:")
                for glacier_id in missing_glaciers:
                    self.logger.warning(f"RGI60-14.{glacier_id}")

            # Import GridWeightsGenerator with detailed error handling
            self.logger.debug("Importing GridWeightsGenerator")
            try:
                from preprocess_meteo import GridWeightsGenerator
            except ImportError as e:
                self.logger.error(f"Failed to import GridWeightsGenerator: {e}")
                raise
            
            # CHANGE: Use namelist-based configuration for GridWeightsGenerator
            self.logger.debug("Creating grid weights generator")
            # Create a temporary namelist-like config
            
            # Generate grid weights using the updated method
            self.logger.debug("Generating grid weights")
            grid_weights_generator = GridWeightsGenerator(self.namelist_path)
            grid_weights = grid_weights_generator.generate(use_precipitation=True)
            
            # Extract relevant arrays for efficiency
            self.logger.debug("Extracting grid weights and NetCDF data")
            grid_weights_np = grid_weights[['HRU ID', 'cell_id', 'normalized_relative_area']].to_numpy()
            
            # Get variable names and extract data with explicit error handling
            var_names = list(ds.data_vars.keys())
            self.logger.debug(f"NetCDF variables: {var_names}")
            
            if len(var_names) == 0:
                raise ValueError("No data variables found in NetCDF file")
                
            # Try the first variable if only one exists
            if len(var_names) == 1:
                var_name = var_names[0]
            else:
                # CHANGE: Look for ERA5-Land precipitation variable names
                era5_precip_vars = ['tp', 'precip', 'precipitation', 'total_precipitation']
                var_name = None
                for era5_var in era5_precip_vars:
                    if era5_var in var_names:
                        var_name = era5_var
                        break
                
                if var_name is None:
                    # Fallback to first non-coordinate variable
                    non_coord_vars = [v for v in var_names if not any(coord in v.lower() for coord in ['coord', 'crs', 'lv95'])]
                    if non_coord_vars:
                        var_name = non_coord_vars[0]
                    else:
                        var_name = var_names[0]
            
            self.logger.debug(f"Using variable: {var_name}")
            
            # Extract the data with detailed shape information
            array3d = ds[var_name].values
            self.logger.debug(f"Array shape: {array3d.shape}")
            
            # Special handling for 1D or 2D arrays
            if len(array3d.shape) == 1:
                self.logger.warning(f"Array is 1D with shape {array3d.shape}. Reshaping to 2D.")
                array3d = array3d.reshape(-1, 1)
            
            time_dim = array3d.shape[0]
            
            # Detailed info about grid weights and array shapes
            self.logger.debug(f"Grid weights shape: {grid_weights_np.shape}")
            self.logger.debug(f"Array3D shape: {array3d.shape}")
            self.logger.debug(f"Time dimension: {time_dim}")
            
            # Apply function that calculates precip for each HRU with explicit error handling
            self.logger.debug("Calculating precipitation for each HRU")
            num_cores = max(1, multiprocessing.cpu_count() - 1)
            self.logger.debug(f"Processing HRUs in parallel using {num_cores} cores")
            
            # CRITICAL CHANGE: Sequential processing first for easier debugging
            self.logger.debug("Trying sequential processing first to identify any issues")
            hru_range = range(1, max(grid_weights['HRU ID']) + 1)
            results = []
            
            for hru_id in hru_range:
                self.logger.debug(f"Processing HRU {hru_id}")
                try:
                    hru_result = self._process_hru(hru_id, grid_weights_np, array3d, time_dim)
                    results.append(hru_result)
                except Exception as e:
                    self.logger.error(f"Error processing HRU {hru_id}: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    # Continue with next HRU rather than failing
                    results.append((hru_id, np.zeros(time_dim, dtype=np.float32)))
            
            result_df = pd.DataFrame({hru_id: values for hru_id, values in results})
            
            # Process GloGEM data for glacier HRUs
            unique_ids = glogem['id'].unique()
            if self.debug:
                self.logger.debug(f"GloGEM unique IDs: {unique_ids}")
            
            # Generate the full date range for the desired period using date strings
            full_date_range = pd.date_range(self.start_date, self.end_date)
            
            # Put in glacier time series for glacier HRUs
            for glogem_id in glogem['id'].unique():
                # Create the full glacier ID string
                full_glacier_id = f"RGI60-14.{glogem_id}"
            
                # Check if this glacier exists in the HRU DataFrame - safely using a mask
                mask = HRU['Glacier_Cl'].notna() & (HRU['Glacier_Cl'] == full_glacier_id)
                if not mask.any():
                    self.logger.warning(f"Warning: Glacier ID {full_glacier_id} from GloGEM not found in HRU data")
                    continue
                
                # Filter glogem time series for this ID
                filtered_glogem = glogem[glogem['id'] == glogem_id].copy()
                filtered_glogem['date'] = pd.to_datetime(filtered_glogem['date'])
            
                # Reindex filtered_glogem to match the full date range
                filtered_glogem = filtered_glogem.set_index('date').reindex(full_date_range, fill_value=0).reset_index()
                filtered_glogem.rename(columns={'index': 'date'}, inplace=True)
            
                # Identify the HRU ID using the mask
                hru_id = HRU.loc[mask, 'HRU ID'].iloc[0]
            
                # Assign glogem values to the corresponding HRU
                result_df[hru_id] = filtered_glogem['q'].values
            
            # Create time index using date strings
            time_index = full_date_range
            
            # Format to numpy
            result_array = result_df.to_numpy() 
            
            # Create the dimension variables
            x_values = np.arange(1, result_array.shape[1] + 1)
            y_values = np.arange(1, 2)
            
            if self.debug:
                self.logger.debug(f"X values: {x_values}")
                self.logger.debug(f"Y values: {y_values}")
            
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
            
            # Save to NetCDF
            # CHANGE: Update output filename for ERA5-Land
            output_path = Path(out_dir, 'era5_land_precip_coupled.nc')
            ds_new.to_netcdf(output_path)
            self.logger.info(f"Saved coupled precipitation data to {output_path}")
            
            # Extract the data variable
            data_variable = ds_new['data']

            # Select the first time step
            first_time_step = data_variable.isel(time=0)

            # Plot the first time step as a raster
            plt.figure(figsize=(10, 6))
            plt.imshow(first_time_step, cmap='viridis', origin='lower')
            plt.colorbar(label='Value')
            plt.title('First Time Step of Data Variable')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.show()

            return result_df
        
        except Exception as e:
            self.logger.error(f"Error in prepare_precip_coupled: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def prepare_temperature_coupled(self) -> Dict[str, pd.DataFrame]:
        """
        Prepare temperature data coupled with glacier information for hydrological modeling.
        
        Returns
        -------
        dict
            Dictionary containing temperature dataframes for each temperature variable
        """
        try:
            self.logger.info("Preparing coupled temperature data")
            
            # Check if required fields are available
            required_fields = ['model_type', 'start_date', 'end_date']
            for field in required_fields:
                if not getattr(self, field):
                    raise ValueError(f"Missing required field for coupled forcing: {field}")
                    
            # Load necessary files
            shapefile_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
            gdf = gpd.read_file(shapefile_path)
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
            
            # CHANGE: Use namelist-based configuration for GridWeightsGenerator
            
            # Generate grid weights
            self.logger.debug("Generating grid weights")
            grid_weights_generator = GridWeightsGenerator(self.namelist_path)
            grid_weights = grid_weights_generator.generate(use_precipitation=True)

            # CHANGE: List of ERA5-Land temperature files to process
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
                self.logger.debug(f"NetCDF coordinates: {list(ds.coords)}")
                
                # Get variable names and extract data with explicit error handling
                var_names = list(ds.data_vars.keys())
                self.logger.debug(f"NetCDF variables: {var_names}")
                
                if len(var_names) == 0:
                    raise ValueError("No data variables found in NetCDF file")
                    
                # CHANGE: Look for ERA5-Land temperature variable names
                if len(var_names) == 1:
                    var_name = var_names[0]
                else:
                    # Look for ERA5-Land temperature variables
                    era5_temp_vars = ['t2m', 'temp', 'temperature', '2m_temperature']
                    var_name = None
                    for era5_var in era5_temp_vars:
                        if era5_var in var_names:
                            var_name = era5_var
                            break
                    
                    if var_name is None:
                        # Fallback to first non-coordinate variable
                        non_coord_vars = [v for v in var_names if not any(coord in v.lower() for coord in ['coord', 'crs', 'lv95'])]
                        if non_coord_vars:
                            var_name = non_coord_vars[0]
                        else:
                            var_name = var_names[0]
                
                self.logger.debug(f"Using variable: {var_name}")
                
                # Extract the data with detailed shape information
                array3d = ds[var_name].values
                self.logger.debug(f"Array shape: {array3d.shape}")
                
                # Special handling for 1D or 2D arrays
                if len(array3d.shape) == 1:
                    self.logger.warning(f"Array is 1D with shape {array3d.shape}. Reshaping to 2D.")
                    array3d = array3d.reshape(-1, 1)
                
                time_dim = array3d.shape[0]
                
                # Extract relevant arrays for efficiency
                grid_weights_np = grid_weights[['HRU ID', 'cell_id', 'normalized_relative_area']].to_numpy()

                # Apply function to calculate temperature for each HRU
                num_cores = max(1, multiprocessing.cpu_count() - 1)
                self.logger.debug(f"Processing HRUs in parallel using {num_cores} cores")
                
                # CRITICAL CHANGE: Sequential processing first for easier debugging
                self.logger.debug("Processing HRUs sequentially to identify any issues")
                hru_range = range(1, max(grid_weights['HRU ID']) + 1)
                results = []
                
                for hru_id in hru_range:
                    try:
                        hru_result = self._process_hru_debug(hru_id, grid_weights_np, array3d, time_dim)
                        results.append(hru_result)
                    except Exception as e:
                        self.logger.error(f"Error processing HRU {hru_id}: {str(e)}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                        # Continue with next HRU rather than failing
                        results.append((hru_id, np.zeros(time_dim, dtype=np.float32)))
                
                result_df = pd.DataFrame({hru_id: values for hru_id, values in results})
                temp_data[file_name] = result_df

            # Process GloGEM data for glacier HRUs
            unique_ids = glogem['id'].unique()
            for glogem_id in unique_ids:
                # Create the full glacier ID string
                full_glacier_id = f"RGI60-14.{glogem_id}"
            
                # Check if this glacier exists in the HRU DataFrame - safely using a mask
                mask = HRU['Glacier_Cl'].notna() & (HRU['Glacier_Cl'] == full_glacier_id)
                if not mask.any():
                    self.logger.warning(f"Warning: Glacier ID {full_glacier_id} from GloGEM not found in HRU data")
                    continue
                
                # Identify the HRU ID using the mask
                hru_id = HRU.loc[mask, 'HRU ID'].iloc[0]
            
                # Assign the constant value 20 to this HRU across all time steps
                for df in temp_data.values():
                    df[hru_id] = 20
                    
            # Save results to NetCDF for each temperature variable
            for file_name, result_df in temp_data.items():
                # CHANGE: Update output filenames for ERA5-Land
                output_name = file_name.replace('.nc', '_coupled.nc')
                out_dir = Path(self.model_dir, f'catchment_{self.gauge_id}', self.model_type, 'data_obs')
                output_path = Path(out_dir, output_name)

                # Generate the full date range for the desired period using date strings
                time_index = pd.date_range(self.start_date, self.end_date)
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

                # Extract the data variable
                data_variable = ds_new['data']

                # Select the first time step
                first_time_step = data_variable.isel(time=0)

                # Plot the first time step as a raster
                plt.figure(figsize=(10, 6))
                plt.imshow(first_time_step, cmap='viridis', origin='lower')
                plt.colorbar(label='Value')
                plt.title('First Time Step of Data Variable')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.show()

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
            
            if 'Glacier_Cl' in hru_gdf.columns:
                # Filter out NaN values first
                glacier_series = hru_gdf['Glacier_Cl'].dropna()
                if not glacier_series.empty:
                    # Convert to standard format (RGI60-14.xxxxx)
                    glacier_ids_hru = glacier_series.unique().tolist()
                    # Also store the numeric part for comparison with GloGEM
                    for g_id in glacier_ids_hru:
                        if isinstance(g_id, str) and 'RGI60-14.' in g_id:
                            glacier_ids_hru_numeric.append(g_id.replace('RGI60-14.', ''))
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
                results['missing_in_glogem'].append(f"RGI60-14.{g_id}")
            
            for g_id in missing_in_hru:
                results['missing_in_hru'].append(f"RGI60-14.{g_id}")
                
            for g_id in matched:
                results['matched'].append(f"RGI60-14.{g_id}")
            
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


    def prepare_coupled_forcing(self) -> Dict[str, Any]:
        """
        Prepare all coupled forcing data (precipitation and temperature)
        
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
            self.process_time_series()

        # Validate glacier IDs
        validation_results = self.validate_glacier_ids()
            
        # Prepare precipitation data
        self.logger.info("Preparing precipitation data")
        precip_data = self.prepare_precip_coupled()
        
        # Prepare temperature data
        self.logger.info("Preparing temperature data")
        temp_data = self.prepare_temperature_coupled()
        
        # Create grid weights file
        self.logger.info("Creating grid weights file")
        self.coupled_grid_weights_file()

        # Add validation plots for precipitation
        self.validate_precipitation_files()
        
        self.logger.info("All coupled forcing data prepared successfully")
        
        return {
            'precipitation': precip_data,
            'temperature': temp_data
        }
