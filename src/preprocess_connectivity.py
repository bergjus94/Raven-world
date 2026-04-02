#### HRU connectivity calculators for Raven hydrological model
#### Justine Berg

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
import rioxarray as rxr
import xrspatial as xrs
from pathlib import Path
import pandas as pd
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Any, Tuple
from shapely.geometry import shape, MultiPolygon
from rasterio.features import shapes, rasterize
from rasterio.transform import xy
from pyproj import Transformer
import yaml
from pysheds.grid import Grid as pyshedsGrid
from paths import get_paths

#--------------------------------------------------------------------------------
############################### HRU connectivity ################################
#--------------------------------------------------------------------------------

class HRUConnectivityCalculator:
    """
    A class for calculating connectivity between HRUs for Raven hydrological model
    Adapted for worldwide catchment processing setup
    """
    
    def __init__(self, config: Union[Dict[str, Any], str, Path]):
        """
        Initialize the HRUConnectivityCalculator
        
        Parameters
        ----------
        config : Dict[str, Any] or str or Path
            Configuration dictionary with parameters OR path to namelist YAML file:
            - model_dir : Path or str (optional if using namelist)
                Directory where model files are stored
            - gauge_id : str or int
                ID of the catchment gauge
            - mode : str, optional
                Mode for connectivity calculation ('single' or 'multiple'), default 'single'
            - min_area_threshold : float, optional
                Minimum area in km² for an HRU to receive flow, default 0.01
            - debug : bool, optional
                Whether to enable debug mode, default False
        """
        # Load configuration from namelist if path provided
        if isinstance(config, (str, Path)):
            with open(config, 'r') as f:
                namelist_config = yaml.safe_load(f)
            
            # Extract parameters from namelist
            self.gauge_id = namelist_config['gauge_id']
            self.model_type = namelist_config['model_type']
            coupled = namelist_config.get('coupled', False)
            paths = get_paths(namelist_config)
            self.model_dir = paths['catchment_dir']
            self._topo_dir = paths['topo_dir']
            self.shared_data_dir = paths['data_obs_dir']

            # Optional parameters from namelist or defaults
            self.mode = namelist_config.get('nconnect', 'single')
            self.min_area_threshold = namelist_config.get('min_area_threshold', 0.01)
            self.debug = namelist_config.get('debug', False)
            
        else:
            # Use config dictionary (backward compatibility)
            # Required parameters
            self.model_dir = Path(config['model_dir'])
            self.gauge_id = config['gauge_id']
            self._topo_dir = self.model_dir / 'topo_files'
            self.shared_data_dir = self.model_dir / 'data_obs'

            # Optional parameters
            self.mode = config.get('mode', config.get('nconnect', 'single'))
            self.min_area_threshold = config.get('min_area_threshold', 0.01)
            self.debug = config.get('debug', False)
        
        # ✅ NEW: Define MODEL-SPECIFIC directory (for backward compatibility)
        if hasattr(self, 'model_type'):
            self.model_data_dir = self.model_dir / self.model_type / 'data_obs'
        else:
            self.model_data_dir = None
        
        # Create directories
        self.shared_data_dir.mkdir(parents=True, exist_ok=True)
        if self.model_data_dir:
            self.model_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.dem_path = None
        self.hru_shapefile = None
        self.hru_raster = None
        self.dem_grid = None
        self.dem_data = None
        self.flow_dir = None
        self.flow_acc = None
        self.connectivity_df = None
        
        # Set up logging
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        Set up logger for this class
        
        Returns
        -------
        logging.Logger
            Configured logger
        """
        log_level = logging.DEBUG if self.debug else logging.INFO
        
        # Configure logging
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=log_level,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Disable verbose logging from various packages
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('rasterio').setLevel(logging.WARNING)
        logging.getLogger('fiona').setLevel(logging.WARNING)
        logging.getLogger('numba').setLevel(logging.WARNING)
        logging.getLogger('numba.core').setLevel(logging.WARNING)
        logging.getLogger('numba.core.ssa').setLevel(logging.WARNING)
        logging.getLogger('numba.core.interpreter').setLevel(logging.WARNING)
        logging.getLogger('numba.core.byteflow').setLevel(logging.WARNING)
        logging.getLogger('numba.core.compiler_lock').setLevel(logging.WARNING)
        
        logger = logging.getLogger(f'HRUConnectivityCalculator_Gauge_{self.gauge_id}')
        
        # ✅ NEW: Log directory structure
        logger.info(f"HRU Connectivity Calculator for gauge {self.gauge_id}")
        logger.info(f"📁 Shared connectivity files: {self.shared_data_dir}")
        if self.model_data_dir:
            logger.info(f"📋 Files will be copied to: {self.model_data_dir}")
        
        return logger


    def get_path(self, filename: str) -> Path:
        """
        Get path to a file in the appropriate directory including model type
        ✅ UPDATED: Uses shared data_obs directory for connectivity files
        
        Parameters
        ----------
        filename : str
            Name of the file
            
        Returns
        -------
        Path
            Full path to the file
        """
        # All connectivity files go to topo variant directory (HRU-dependent)
        return self._topo_dir / filename

    def load_data(self) -> None:
        """
        Load required data for connectivity calculation
        """
        self.logger.info("Loading data for connectivity calculation")
        
        # Set paths (adapted to your file structure)
        self.dem_path = self.get_path('clipped_dem.tif')
        hru_path = self.get_path('HRU.shp')
        
        # Check if files exist
        if not self.dem_path.exists():
            self.logger.error(f"DEM file not found at {self.dem_path}")
            raise FileNotFoundError(f"DEM file not found at {self.dem_path}")
            
        if not hru_path.exists():
            self.logger.error(f"HRU shapefile not found at {hru_path}")
            raise FileNotFoundError(f"HRU shapefile not found at {hru_path}")
        
        # Load HRU shapefile
        self.logger.debug(f"Loading HRU shapefile from {hru_path}")
        self.hru_shapefile = gpd.read_file(hru_path)
        
        # Initialize pysheds grid
        self.logger.debug(f"Loading DEM from {self.dem_path}")
        self.dem_grid = pyshedsGrid.from_raster(str(self.dem_path))
        self.dem_data = self.dem_grid.read_raster(str(self.dem_path))
        
        self.logger.info(f"Loaded data successfully: {len(self.hru_shapefile)} HRUs")

    #---------------------------------------------------------------------------------

    def prepare_dem_flow(self) -> None:
        """
        Prepare DEM for flow direction and accumulation
        """
        self.logger.info("Preparing DEM for flow calculation")
        
        # Fill pits and depressions in DEM and resolve flats
        self.logger.debug("Filling pits in DEM")
        pit_filled_dem = self.dem_grid.fill_pits(self.dem_data)
        
        self.logger.debug("Filling depressions in DEM")
        flooded_dem = self.dem_grid.fill_depressions(pit_filled_dem)
        
        self.logger.debug("Resolving flats in DEM")
        inflated_dem = self.dem_grid.resolve_flats(flooded_dem)
        
        # Calculate flow direction
        self.logger.debug("Calculating flow direction")
        self.flow_dir = self.dem_grid.flowdir(inflated_dem, routing='d8', nodata_out=np.int64(0))
        
        self.logger.info("DEM preparation complete")

    #---------------------------------------------------------------------------------

    def rasterize_hrus(self) -> np.ndarray:
        """
        Rasterize HRU shapefile to match DEM grid
        
        Returns
        -------
        np.ndarray
            Rasterized HRU data
        """
        self.logger.info("Rasterizing HRU shapefile")
        
        # Get raster metadata from DEM
        with rasterio.open(str(self.dem_path)) as src:
            transform = src.transform
            out_shape = (src.height, src.width)
            crs = src.crs
        
        # Ensure the shapefile is in the same CRS as the DEM
        hru_df = self.hru_shapefile.to_crs(crs)
        
        # Rasterize the shapefile based on HRU_ID
        self.logger.debug(f"Rasterizing {len(hru_df)} HRUs to shape {out_shape}")
        hru_raster = rasterize(
            [(geom, value) for geom, value in zip(hru_df.geometry, hru_df['HRU_ID'])],
            out_shape=out_shape,
            transform=transform,
            fill=np.nan,
            dtype='float32'
        )
        
        # Plot the rasterized HRUs if in debug mode
        if self.debug:
            plt.figure(figsize=(8, 6))
            plt.imshow(hru_raster, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='HRU ID')
            plt.title("Rasterized HRU")
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            
            # Save plot if debug mode
            plot_dir = self.model_dir / 'plots'
            plot_dir.mkdir(exist_ok=True)
            plot_path = plot_dir / f"rasterized_hrus_{self.gauge_id}.png"
            plt.savefig(plot_path)
            plt.show()
        
        self.hru_raster = hru_raster
        self.logger.info("HRU rasterization complete")
        return hru_raster
    
    #---------------------------------------------------------------------------------
    
    def calculate_flow_accumulation(self) -> np.ndarray:
        """
        Calculate flow accumulation for each HRU (OPTIMIZED - single accumulation)
        """
        self.logger.info("Calculating flow accumulation (optimized)")
        
        # Initialize DataFrame for connectivity
        hru_df = self.hru_shapefile[['HRU_ID']].copy()
        if 'Area_km2' in self.hru_shapefile.columns:
            hru_df['Area_km2'] = self.hru_shapefile['Area_km2']
        else:
            self.logger.warning("Area_km2 not found in HRU shapefile, using default values")
            hru_df['Area_km2'] = 1.0
            
        hru_df['connectivity'] = [{} for _ in range(len(hru_df))]
        
        # ✅ OPTIMIZATION: Calculate flow accumulation ONCE for entire catchment
        # Instead of once per HRU (huge speedup!)
        self.logger.info("Calculating catchment-wide flow accumulation...")
        
        flow_acc_total = self.dem_grid.accumulation(
            self.flow_dir,
            routing='d8',
            nodata_out=np.float64(0)
        )
        
        self.flow_acc = flow_acc_total.view(np.ndarray)
        self.connectivity_df = hru_df
        
        self.logger.info("Flow accumulation calculation complete")
        return self.flow_acc
    
    #---------------------------------------------------------------------------------

    def _sum_contributing_flow_acc(self) -> None:
        """
        Calculate connectivity between HRUs based on flow accumulation (VECTORIZED VERSION)
        """
        self.logger.info("Calculating connectivity between HRUs (vectorized)")
        
        flow_dir = self.flow_dir.view(np.ndarray)
        height, width = self.flow_acc.shape
        
        # ✅ OPTIMIZATION: Single combined mask (slightly faster)
        valid_mask = (
            ~np.isnan(self.hru_raster) & 
            (self.hru_raster > 0) & 
            (flow_dir > 0)
        )
        
        # Early exit if no valid cells
        if not np.any(valid_mask):
            self.logger.warning("No valid cells found for connectivity calculation")
            return
        
        # Extract valid cells (only once)
        valid_rows, valid_cols = np.where(valid_mask)
        valid_flow_dir = flow_dir[valid_mask]
        valid_hru_ids = self.hru_raster[valid_mask].astype(np.int32)  # int32 is faster than int
        valid_flow_acc = self.flow_acc[valid_mask]
        
        # Vectorized calculation of next cell coordinates
        # Flow direction mapping: [E, SE, S, SW, W, NW, N, NE] = [1, 2, 4, 8, 16, 32, 64, 128]
        delta_map = {
            1: (0, 1),    # E
            2: (1, 1),    # SE
            4: (1, 0),    # S
            8: (1, -1),   # SW
            16: (0, -1),  # W
            32: (-1, -1), # NW
            64: (-1, 0),  # N
            128: (-1, 1)  # NE
        }
        
        # Initialize arrays for next cell coordinates
        next_rows = np.zeros_like(valid_rows)
        next_cols = np.zeros_like(valid_cols)
        
        # Apply deltas based on flow direction
        for dir_code, (di, dj) in delta_map.items():
            mask = valid_flow_dir == dir_code
            next_rows[mask] = valid_rows[mask] + di
            next_cols[mask] = valid_cols[mask] + dj
        
        # Check bounds
        bounds_mask = (
            (next_rows >= 0) & (next_rows < height) &
            (next_cols >= 0) & (next_cols < width)
        )
        
        # Filter to valid next cells
        next_rows = next_rows[bounds_mask]
        next_cols = next_cols[bounds_mask]
        valid_hru_ids = valid_hru_ids[bounds_mask]
        valid_flow_acc = valid_flow_acc[bounds_mask]
        
        # Get next HRU IDs
        next_hru_ids = self.hru_raster[next_rows, next_cols]
        
        # Filter out NaN and same-HRU connections
        valid_next_mask = (~np.isnan(next_hru_ids)) & (next_hru_ids != valid_hru_ids)
        
        from_hrus = valid_hru_ids[valid_next_mask]
        to_hrus = next_hru_ids[valid_next_mask].astype(int)
        flow_values = valid_flow_acc[valid_next_mask]
        
        # Create unique connection pairs
        connection_pairs = np.column_stack([from_hrus, to_hrus])
        
        # Use numpy's unique with return_inverse for grouping
        unique_pairs, inverse_indices = np.unique(
            connection_pairs, axis=0, return_inverse=True
        )
        
        # Sum flow values for each unique pair using numpy (faster than pandas)
        aggregated_flow = np.zeros(len(unique_pairs), dtype=flow_values.dtype)
        np.add.at(aggregated_flow, inverse_indices, flow_values)
        
        connectivity_updates = {}
        for i, (from_hru, to_hru) in enumerate(unique_pairs):
            from_hru_int = int(from_hru)
            if from_hru_int not in connectivity_updates:
                connectivity_updates[from_hru_int] = {}
            connectivity_updates[from_hru_int][int(to_hru)] = float(aggregated_flow[i])
        
        # Update connectivity_df in one pass
        for idx, row in self.connectivity_df.iterrows():
            hru_id = row['HRU_ID']
            if hru_id in connectivity_updates:
                self.connectivity_df.at[idx, 'connectivity'] = connectivity_updates[hru_id]
        
        total_connections = len(unique_pairs)
        self.logger.info(f"Connectivity calculation complete: {total_connections} connections found")

    #---------------------------------------------------------------------------------

    def filter_small_hrus(self) -> None:
        """
        Filter out connections to small HRUs based on area threshold
        """
        if self.min_area_threshold <= 0:
            self.logger.debug("Skipping small HRU filtering (threshold <= 0)")
            return
            
        self.logger.info(f"Filtering out connections to small HRUs (< {self.min_area_threshold} km²)")
        
        # Identify small HRUs that shouldn't receive flow
        small_hrus = set(self.connectivity_df[self.connectivity_df['Area_km2'] < self.min_area_threshold]['HRU_ID'])
        
        if small_hrus:
            self.logger.debug(f"Found {len(small_hrus)} small HRUs to exclude from receiving flow")
            
            # Filter out connections to small HRUs
            for idx, row in self.connectivity_df.iterrows():
                connectivity = row['connectivity']
                
                # Remove connections where the target is a small HRU
                for target_id in list(connectivity.keys()):
                    if target_id in small_hrus:
                        self.logger.debug(f"Removing connection from HRU {row['HRU_ID']} to small HRU {target_id}")
                        del connectivity[target_id]
                        
                self.connectivity_df.at[idx, 'connectivity'] = connectivity
            
            self.logger.info(f"Removed connections to {len(small_hrus)} small HRUs")
        else:
            self.logger.debug("No small HRUs found to filter out")

    #---------------------------------------------------------------------------------

    def normalize_connectivity(self) -> None:
        """
        Normalize connectivity values for each HRU
        """
        self.logger.info("Normalizing connectivity values")
        
        def _normalize_row(row):
            connectivity = row['connectivity']
            if not connectivity:
                return row
                
            # If the maximum connectivity leaves the catchment, nullify the connectivity
            if 0 in connectivity and max(connectivity, key=connectivity.get) == 0:
                row['connectivity'] = {}
                return row
                
            # Remove the key 0 if it exists (flow leaving catchment)
            if 0 in connectivity:
                del connectivity[0]
                
            # Normalize the connectivity within the catchment
            total_flow = sum(connectivity.values())
            if total_flow == 0:
                return row
                
            for key in connectivity:
                connectivity[key] /= total_flow
                
            row['connectivity'] = connectivity
            return row
        
        self.connectivity_df = self.connectivity_df.apply(_normalize_row, axis=1)
        self.logger.info("Connectivity normalization complete")

    #---------------------------------------------------------------------------------

    def keep_highest_connectivity(self) -> None:
        """
        Keep only the highest connectivity for each HRU
        """
        self.logger.info("Keeping only highest connectivity for each HRU")
        
        def _keep_highest(row):
            connectivity = row['connectivity']
            if not connectivity:
                return row
                
            # If the maximum connectivity leaves the catchment, nullify the connectivity
            if 0 in connectivity and max(connectivity, key=connectivity.get) == 0:
                row['connectivity'] = {}
                return row
                
            # Remove the key 0 if it exists
            if 0 in connectivity:
                del connectivity[0]
                
            if not connectivity:
                return row
                
            # Keep only the highest connectivity
            max_key = max(connectivity, key=connectivity.get)
            connectivity = {max_key: 1.0}
            row['connectivity'] = connectivity
            return row
        
        self.connectivity_df = self.connectivity_df.apply(_keep_highest, axis=1)
        self.logger.info("Highest connectivity filtering complete")

    #---------------------------------------------------------------------------------

    def find_nearest_hru(self) -> None:
        """
        Find the nearest HRU for HRUs without connectivity (FIXED - pre-calculate centroids)
        """
        self.logger.info("Finding nearest HRUs for disconnected HRUs")
        
        # Identify HRUs with no connectivity
        hrus_without_connections = []
        for idx, row in self.connectivity_df.iterrows():
            if not row['connectivity']:
                hrus_without_connections.append(row['HRU_ID'])
        
        if not hrus_without_connections:
            self.logger.debug("No disconnected HRUs found")
            return
            
        self.logger.debug(f"Found {len(hrus_without_connections)} HRUs without connections")
        
        # ✅ FIX: Pre-calculate centroids for ALL HRUs ONCE (not in the loop!)
        unique_hrus = self.connectivity_df['HRU_ID'].unique()
        hru_centroids = {}
        
        self.logger.debug(f"Calculating centroids for {len(unique_hrus)} HRUs...")
        
        # Vectorized centroid calculation
        for hru_id in unique_hrus:
            mask = (self.hru_raster == hru_id)
            if np.any(mask):
                rows, cols = np.where(mask)
                hru_centroids[hru_id] = np.array([np.mean(rows), np.mean(cols)])
        
        self.logger.debug(f"Calculated {len(hru_centroids)} centroids")
        
        # ✅ FIX: Use scipy KDTree for nearest neighbor search (MUCH faster!)
        from scipy.spatial import cKDTree
        
        # Build KDTree from all HRU centroids
        all_hru_ids = list(hru_centroids.keys())
        all_centroids = np.array([hru_centroids[hru_id] for hru_id in all_hru_ids])
        
        self.logger.debug(f"Building KDTree with {len(all_centroids)} points...")
        tree = cKDTree(all_centroids)
        
        # For each disconnected HRU, find nearest neighbor
        for hru_id in hrus_without_connections:
            if hru_id not in hru_centroids:
                self.logger.warning(f"HRU {hru_id} not found in raster, skipping")
                continue
            
            src_centroid = hru_centroids[hru_id]
            
            # Query tree for 2 nearest neighbors (first will be itself)
            distances, indices = tree.query(src_centroid, k=2)
            
            # Get the second nearest (first is itself)
            closest_hru = all_hru_ids[indices[1]]
            
            self.logger.debug(f"Creating connection from HRU {hru_id} to closest HRU {closest_hru} (distance: {distances[1]:.2f})")
            
            # Get index for the HRU without connections
            idx = self.connectivity_df.index[self.connectivity_df['HRU_ID'] == hru_id].tolist()[0]
            
            # Add connection
            if self.mode == 'single':
                self.connectivity_df.at[idx, 'connectivity'] = {closest_hru: 1.0}
            else:
                connect = self.connectivity_df.at[idx, 'connectivity']
                connect[closest_hru] = 1.0
                self.connectivity_df.at[idx, 'connectivity'] = connect
        
        self.logger.info(f"Connected {len(hrus_without_connections)} disconnected HRUs to nearest neighbors")

    #---------------------------------------------------------------------------------

    def plot_connectivity_summary(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot HRU connectivity summary (FAST - no arrows!)
        Shows: HRU map + simple connectivity statistics
        
        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Figure size (width, height), default (12, 8)
        """
        self.logger.info("Plotting connectivity summary")
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # LEFT: HRU raster map
        im = axes[0].imshow(self.hru_raster, cmap='tab20', interpolation='nearest')
        axes[0].set_title('HRU Map')
        axes[0].set_xlabel('X Coordinate')
        axes[0].set_ylabel('Y Coordinate')
        plt.colorbar(im, ax=axes[0], label='HRU ID')
        
        # RIGHT: Connectivity statistics
        num_connections = [len(row['connectivity']) for _, row in self.connectivity_df.iterrows()]
        
        axes[1].hist(num_connections, bins=range(0, max(num_connections) + 2), 
                    edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Number of Connections per HRU')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Connectivity Distribution')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics text
        total_connections = sum(num_connections)
        avg_connections = total_connections / len(num_connections) if num_connections else 0
        
        stats_text = f"Total HRUs: {len(self.connectivity_df)}\n"
        stats_text += f"Total connections: {total_connections}\n"
        stats_text += f"Avg connections/HRU: {avg_connections:.2f}"
        
        axes[1].text(0.98, 0.98, stats_text,
                    transform=axes[1].transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save the plot
        plot_dir = self.model_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / f"connectivity_summary_{self.gauge_id}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved connectivity summary to {plot_path}")
        
        plt.show()
        plt.close()

    #---------------------------------------------------------------------------------

    def write_connectivity_file(self) -> None:
        """
        Write connectivity information to a Raven-compatible file
        """
        self.logger.info("Writing connectivity to Raven file")
        
        # Set output file path (adapted to your structure)
        output_file = self.get_path('connections.rvh')
        
        self.logger.debug(f"Writing connectivity to {output_file}")
        
        with open(output_file, 'w') as f:
            f.write(":LateralConnections  SNOW_REDISTRIBUTE\n")
            f.write("#HRU_ID\tConnected_HRU_ID\tWeight\n")
            
            connection_count = 0
            for idx, row in self.connectivity_df.iterrows():
                unit_id = row['HRU_ID']
                connectivity = row['connectivity']
                
                if connectivity:
                    for target_id, value in connectivity.items():
                        f.write(f"{unit_id}\t{target_id}\t{value:.6f}\n")
                        connection_count += 1
            
            f.write(":EndLateralConnections\n")
        
        self.logger.info(f"Wrote {connection_count} connections to {output_file}")
    #---------------------------------------------------------------------------------

    def _copy_to_model_directory(self) -> None:
        """
        Copy connectivity files from shared data_obs to model-specific data_obs.
        This maintains backward compatibility while using shared storage.
        """
        import shutil
        
        if not self.model_data_dir:
            self.logger.debug("No model-specific directory defined - skipping copy")
            return
        
        # Files to copy
        connectivity_files = [
            self.get_path('connections.rvh'),
            self.get_path('HRU_connectivity.csv')
        ]
        
        self.logger.info(f"📋 Copying connectivity files from shared to model-specific directory...")
        self.logger.debug(f"  Source: {self.shared_data_dir}")
        self.logger.debug(f"  Destination: {self.model_data_dir}")
        
        copied_count = 0
        for file_path in connectivity_files:
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
        
        self.logger.info(f"✅ Successfully copied {copied_count}/{len(connectivity_files)} files to {self.model_data_dir.name}/")

    #---------------------------------------------------------------------------------

    def calculate_connectivity(self) -> pd.DataFrame:
        """
        Calculate HRU connectivity using flow accumulation method
        
        Returns
        -------
        pd.DataFrame
            DataFrame with HRU connectivity information
        """
        self.logger.info(f"Calculating HRU connectivity for gauge {self.gauge_id}")
        
        # ✅ CHECK: Skip if connectivity file already exists
        connections_file = self.get_path('connections.rvh')
        connectivity_csv = self.get_path('HRU_connectivity.csv')
        
        if connections_file.exists() and connectivity_csv.exists():
            self.logger.info("✅ Connectivity files already exist")
            self.logger.info("⏭️  Skipping connectivity calculation")
            self.logger.info("💡 Delete files to force reprocessing:")
            self.logger.info(f"   rm {connections_file}")
            self.logger.info(f"   rm {connectivity_csv}")
            
            # Load existing connectivity DataFrame
            self.logger.info(f"Loading existing connectivity data from {connectivity_csv}")
            existing_connectivity = pd.read_csv(connectivity_csv)
            
            self.logger.info(f"✅ Loaded connectivity for {len(existing_connectivity)} HRUs")
            return existing_connectivity
        
        # ===== NORMAL PROCESSING (if files don't exist) =====
        
        # Step 1: Load required data
        self.load_data()
        
        # Step 2: Prepare DEM for flow calculation
        self.prepare_dem_flow()
        
        # Step 3: Rasterize HRU shapefile
        self.rasterize_hrus()
        
        # Step 4: Calculate flow accumulation
        self.calculate_flow_accumulation()
        
        # Step 5: Sum contributing flow accumulation
        self._sum_contributing_flow_acc()
        
        # Step 6: Filter small HRUs if threshold provided
        if self.min_area_threshold > 0:
            self.filter_small_hrus()
        
        # Step 7: Process connectivity based on mode
        if self.mode == 'multiple':
            self.normalize_connectivity()
        elif self.mode == 'single':
            self.keep_highest_connectivity()
        else:
            self.logger.warning(f"Unknown mode '{self.mode}', using 'single' mode")
            self.keep_highest_connectivity()
        
        # Step 8: Find nearest HRU for disconnected HRUs
        self.find_nearest_hru()
        
        # Step 9: Plot connectivity summary if in debug mode
        if self.debug:
            self.plot_connectivity_summary()
        
        # Step 10: Write connectivity file
        self.write_connectivity_file()
        
        # Step 11: Save connectivity DataFrame
        self.connectivity_df.to_csv(connectivity_csv, index=False)
        self.logger.debug(f"Saved connectivity DataFrame to {connectivity_csv}")
        self.logger.info("Connectivity calculation complete")
        return self.connectivity_df

#---------------------------------------------------------------------------------

# Convenience functions for backward compatibility and easy use

def calculate_connectivity(model_dir: Union[str, Path], gauge_id: str, mode: str = 'single', 
                         min_area_threshold: float = 0.01, debug: bool = False) -> pd.DataFrame:
    """
    Calculate the connectivity between HRUs using flow accumulation method
    
    Parameters
    ----------
    model_dir : str or Path
        Directory containing model files
    gauge_id : str
        ID of the catchment gauge
    mode : str, optional
        Mode for connectivity calculation ('single' or 'multiple'), default 'single'
    min_area_threshold : float, optional
        Minimum area in km² for an HRU to receive flow, default 0.01
    debug : bool, optional
        Whether to enable debug mode, default False
        
    Returns
    -------
    pd.DataFrame
        DataFrame with HRU connectivity information
    """
    config = {
        'model_dir': model_dir,
        'gauge_id': gauge_id,
        'mode': mode,
        'min_area_threshold': min_area_threshold,
        'debug': debug
    }
    
    calculator = HRUConnectivityCalculator(config)
    return calculator.calculate_connectivity()

#---------------------------------------------------------------------------------

def create_connection_file(model_dir: Union[str, Path], gauge_id: str, 
                          nconnect: str = 'single', debug: bool = False) -> pd.DataFrame:
    """
    Legacy function to calculate connectivity and write to a Raven-compatible file
    
    Parameters
    ----------
    model_dir : str or Path
        Directory containing model files
    gauge_id : str
        ID of the catchment gauge
    nconnect : str, optional
        Connection mode ('single' for highest only, 'multiple' for all), default 'single'
    debug : bool, optional
        Whether to enable debug mode, default False
        
    Returns
    -------
    pd.DataFrame
        DataFrame with HRU connectivity information
    """
    config = {
        'model_dir': model_dir,
        'gauge_id': gauge_id,
        'mode': nconnect,
        'debug': debug
    }
    
    calculator = HRUConnectivityCalculator(config)
    connectivity_df = calculator.calculate_connectivity()

    return connectivity_df



class MultiSubbasinConnectivityCalculator:
    """
    Calculates lateral HRU connectivity for multi-subbasin configurations.

    Each subbasin is processed independently using its own local-polygon DEM
    and HRU shapefile (already produced by MultiSubbasinProcessor).  After
    computing per-subbasin connectivity with local HRU IDs the IDs are shifted
    to the globally unique values from the merged HRU_table.csv so that all
    connections reference the same ID space as the rest of the Raven model.

    Per-subbasin intermediate files (local IDs) are saved alongside the merged
    global file:
        catchment_{gauge_id}/topo_files/subbasin_{id}/data_obs/connections.rvh
        catchment_{gauge_id}/topo_files/subbasin_{id}/data_obs/HRU_connectivity.csv
        catchment_{gauge_id}/data_obs/connections.rvh   ← merged, global IDs
    """

    def __init__(self, namelist_path: Union[str, Path]):
        with open(namelist_path, 'r') as f:
            namelist = yaml.safe_load(f)

        self.gauge_id     = namelist['gauge_id']
        self.model_type   = namelist.get('model_type', 'HBV')
        self.nconnect     = namelist.get('nconnect', 'single')
        self.min_area_threshold = namelist.get('min_area_threshold', 0.01)
        self.debug        = namelist.get('debug', False)
        self.subbasins_config = namelist.get('subbasins', [])

        paths = get_paths(namelist)
        self.catchment_dir = paths['catchment_dir']
        self.topo_dir      = paths['topo_shared_dir']
        self.output_dir    = paths['data_obs_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------

    def _get_hru_id_offsets(self) -> Dict[int, int]:
        """
        Read the merged HRU_table.csv and return the per-subbasin HRU ID
        offset, defined as  ``min_global_id - 1``  so that
        ``global_id = local_id + offset``.
        """
        hru_table_path = self.topo_dir / 'HRU_table.csv'
        if not hru_table_path.exists():
            raise FileNotFoundError(
                f"Merged HRU_table.csv not found: {hru_table_path}\n"
                "Run MultiSubbasinProcessor.process_all_subbasins() first."
            )

        merged_df = pd.read_csv(hru_table_path)
        offsets: Dict[int, int] = {}
        for sb in self.subbasins_config:
            sb_id = sb['id']
            sb_rows = merged_df[merged_df['BASIN_ID'] == sb_id]
            if len(sb_rows) > 0:
                offsets[sb_id] = int(sb_rows[':ATTRIBUTES'].min()) - 1
            else:
                self.logger.warning(
                    f"No HRUs found for subbasin {sb_id} in merged HRU_table.csv — offset set to 0"
                )
                offsets[sb_id] = 0
        return offsets

    # ------------------------------------------------------------------

    def _read_connections_rvh(self, filepath: Path, offset: int) -> list:
        """
        Parse a ``connections.rvh`` file and return a list of
        ``(from_global_id, to_global_id, weight)`` tuples with HRU IDs
        shifted by *offset*.
        """
        connections = []
        if not filepath.exists():
            self.logger.warning(f"connections.rvh not found: {filepath}")
            return connections

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith(':'):
                    continue
                parts = line.split()
                if len(parts) == 3:
                    try:
                        from_id = int(parts[0]) + offset
                        to_id   = int(parts[1]) + offset
                        weight  = float(parts[2])
                        connections.append((from_id, to_id, weight))
                    except ValueError:
                        pass
        return connections

    # ------------------------------------------------------------------

    def _write_merged_connections(self, connections: list) -> None:
        """
        Write the merged ``connections.rvh`` with global HRU IDs to
        ``catchment_{gauge_id}/data_obs/connections.rvh``.
        """
        output_file = self.output_dir / 'connections.rvh'
        with open(output_file, 'w') as f:
            f.write(":LateralConnections  SNOW_REDISTRIBUTE\n")
            f.write("#HRU_ID\tConnected_HRU_ID\tWeight\n")
            for from_id, to_id, weight in connections:
                f.write(f"{from_id}\t{to_id}\t{weight:.6f}\n")
            f.write(":EndLateralConnections\n")
        self.logger.info(f"Wrote {len(connections)} merged connections to {output_file}")

    # ------------------------------------------------------------------

    def calculate_connectivity(self) -> list:
        """
        Run ``HRUConnectivityCalculator`` for each subbasin (using its own
        local-polygon DEM + HRU shapefile), shift HRU IDs to global values,
        and write one merged ``connections.rvh``.

        Returns
        -------
        list of (from_global_id, to_global_id, weight)
        """
        offsets = self._get_hru_id_offsets()
        all_connections: list = []

        for sb in self.subbasins_config:
            sb_id       = sb['id']
            sb_gauge_id = str(sb['gauge_id'])
            offset      = offsets.get(sb_id, 0)

            sb_model_dir = self.topo_dir / f'subbasin_{sb_id}'
            self.logger.info(
                f"Computing connectivity for subbasin {sb_id} "
                f"(gauge {sb_gauge_id}), HRU ID offset={offset}"
            )

            calc = HRUConnectivityCalculator({
                'model_dir':           sb_model_dir,
                'gauge_id':            sb_gauge_id,
                'nconnect':            self.nconnect,
                'min_area_threshold':  self.min_area_threshold,
                'debug':               self.debug,
            })

            # Runs full pipeline (respects existing-file skip); writes
            # per-subbasin connections.rvh with local HRU IDs.
            calc.calculate_connectivity()

            # Read back from file (handles both fresh + cached paths)
            sb_connections_file = calc.get_path('connections.rvh')
            shifted = self._read_connections_rvh(sb_connections_file, offset)
            self.logger.info(
                f"  Subbasin {sb_id}: {len(shifted)} connections (local file: {sb_connections_file.name})"
            )
            all_connections.extend(shifted)

        self._write_merged_connections(all_connections)
        return all_connections