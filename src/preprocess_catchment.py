#### This file contains all functions for preprocessing catchment data for Raven hydrological model
#### Updated for worldwide data using SRTM DEM and ESA WorldCover
#### Justine Berg

#--------------------------------------------------------------------------------
############################### import packages #################################
#--------------------------------------------------------------------------------

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import xarray as xr
import rioxarray as rxr
import xrspatial as xrs
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, List, Union, Optional, Any, Tuple
import itertools
from shapely.geometry import shape, MultiPolygon
from rasterio.features import shapes, rasterize
from rasterio.transform import xy
from pyproj import Transformer
from matplotlib.patches import Arrow
import yaml
from pysheds.grid import Grid as pyshedsGrid

#--------------------------------------------------------------------------------
############################### Helper Functions ################################
#--------------------------------------------------------------------------------

def plot_raster(raster, title="Raster", cmap="viridis"):
    """
    Plot a raster
    
    Parameters
    ----------
    raster : xarray.DataArray
        Raster to plot
    title : str, optional
        Title for the plot, by default "Raster"
    cmap : str, optional
        Colormap to use, by default "viridis"
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(raster, cmap=cmap)
    plt.colorbar(label=title)
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

def plot_map(map_data, title="Map", cmap="tab20"):
    """
    Plot a map
    
    Parameters
    ----------
    map_data : numpy.ndarray
        Map data to plot
    title : str, optional
        Title for the plot, by default "Map"
    cmap : str, optional
        Colormap to use, by default "tab20"
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(map_data, cmap=cmap)
    plt.colorbar(label=title)
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

#--------------------------------------------------------------------------------
############################### CatchmentProcessor Class ########################
#--------------------------------------------------------------------------------

class CatchmentProcessor:
    """
    A class for preprocessing catchment data for Raven hydrological model using worldwide datasets
    """
    
    def __init__(self, namelist_path: Union[str, Path]):
        """
        Initialize the CatchmentProcessor with namelist configuration
        
        Parameters
        ----------
        namelist_path : str or Path
            Path to the namelist YAML configuration file
        """
        # Load configuration from namelist
        self.config = self._load_namelist(namelist_path)
        
        # Extract key parameters
        self.gauge_id = self.config['gauge_id']
        self.main_dir = Path(self.config['main_dir'])
        self.coupled = self.config.get('coupled', False)
        self.model_dirs = self.config['model_dirs']
        
        # CORRECT MODEL DIR CONSTRUCTION
        if self.coupled:
            self.model_dir = self.main_dir / self.model_dirs["coupled"] / f'catchment_{self.gauge_id}'
        else:
            self.model_dir = self.main_dir / self.model_dirs["uncoupled"] / f'catchment_{self.gauge_id}'

        # Data directories (format paths with gauge_id) - CORRECTED PATHS
        self.shape_dir = self.main_dir / self.config['shape_dir'].format(gauge_id=self.gauge_id)
        self.raster_dir = self.main_dir / self.config['raster_dir'].format(gauge_id=self.gauge_id)
        self.landuse_dir = self.main_dir / self.config['landuse_dir'].format(gauge_id=self.gauge_id)
        
        # Glacier directory (optional)
        if 'glacier_dir' in self.config:
            self.glacier_dir = self.main_dir / self.config['glacier_dir']  # No gauge_id formatting needed
        else:
            self.glacier_dir = None
        
        # Processing parameters
        criteria = self.config.get('criteria', ['elevation', 'landuse'])
        self.criteria = criteria if isinstance(criteria, list) else [criteria]
        self.elevation_distance = self.config.get('elevation_distance', 100)
        self.slope_distance = self.config.get('slope_distance', 10)
        self.debug = self.config.get('debug', False)
        
        # Initialize data containers
        self.catchment_extent = None
        self.dem_data = None
        self.slope_data = None
        self.aspect_data = None
        self.landuse_data = None
        self.glacier_data = None
        self.glacier_id_mapping = {}
        self.hru_df = None
        self.map_unit_ids = None
        self.hru_stats = None
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Create output directory
        self._create_output_dir()

    #---------------------------------------------------------------------------------

    def _load_namelist(self, namelist_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML namelist file"""
        namelist_path = Path(namelist_path)
        
        if not namelist_path.exists():
            raise FileNotFoundError(f"Namelist file not found: {namelist_path}")
        
        with open(namelist_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config

    #---------------------------------------------------------------------------------

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
        logging.getLogger('rasterio').setLevel(logging.WARNING)
        
        # Create logger with appropriate name
        logger = logging.getLogger(f'CatchmentProcessor_Gauge_{self.gauge_id}')
        
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
                    return record.name.startswith('CatchmentProcessor') and record.levelno >= logging.INFO
                    
            console.addFilter(ModuleFilter())
            logger.addHandler(console)
        
        return logger

    #---------------------------------------------------------------------------------

    def _create_output_dir(self) -> None:
        """
        Create output directory structure
        """
        topo_dir = Path(self.model_dir, 'topo_files')
        topo_dir.mkdir(parents=True, exist_ok=True)
        
        if self.debug:
            plot_dir = Path(self.model_dir, 'plots')
            plot_dir.mkdir(parents=True, exist_ok=True)
            
        self.logger.debug(f"Created directory structure at {topo_dir}")

    #---------------------------------------------------------------------------------

    def get_path(self, filename: str) -> Path:
        """
        Get path to a file in the topo_files directory
        
        Parameters
        ----------
        filename : str
            Name of the file
            
        Returns
        -------
        Path
            Full path to the file
        """
        return Path(self.model_dir, 'topo_files', filename)

    #---------------------------------------------------------------------------------

    def extract_catchment_shape(self) -> gpd.GeoDataFrame:
        """
        Extract a shapefile from the CAMELS shapefile corresponding with the gauging ID
        
        Returns
        -------
        gpd.GeoDataFrame
            Shapefile of desired catchment
        """
        self.logger.info(f"Extracting catchment shape for gauge ID {self.gauge_id}")
        
        if not self.shape_dir.exists():
            raise FileNotFoundError(f"Catchment shapefile not found: {self.shape_dir}")
        
        extent = gpd.read_file(self.shape_dir)
        
        # For single gauge shapefiles, just use the first (and likely only) feature
        if len(extent) == 1:
            extent_catchment = extent
        else:
            # If multiple features, try to filter by gauge_id
            if 'gauge_id' in extent.columns:
                extent_catchment = extent.loc[extent['gauge_id'] == self.gauge_id]
            else:
                self.logger.warning("Multiple features found but no gauge_id column. Using first feature.")
                extent_catchment = extent.iloc[[0]]
        
        if len(extent_catchment) == 0:
            self.logger.warning(f"No catchment found with gauge ID {self.gauge_id}")
            return None
            
        self.catchment_extent = extent_catchment
        return extent_catchment

    #---------------------------------------------------------------------------------

    def clip_srtm_dem(self) -> xr.DataArray:
        """
        Clip the SRTM DEM to the catchment shapefile
        
        Returns
        -------
        xr.DataArray
            Clipped DEM
        """
        self.logger.info(f"Clipping DEM for catchment {self.gauge_id}")
        
        # Check if output file already exists
        out_raster_path = self.get_path('clipped_dem.tif')
        if out_raster_path.exists():
            self.logger.debug(f"Using existing clipped DEM at {out_raster_path}")
            clipped_raster = rxr.open_rasterio(out_raster_path).squeeze()
            self.dem_data = clipped_raster
            return clipped_raster
        
        # Extract desired catchment shapefile if not already done
        if self.catchment_extent is None:
            self.extract_catchment_shape()
        
        if not self.raster_dir.exists():
            raise FileNotFoundError(f"SRTM DEM not found: {self.raster_dir}")
        
        # Open SRTM DEM data
        dem = rxr.open_rasterio(self.raster_dir).squeeze()
        
        # Get the original CRS of the DEM (usually EPSG:4326 for SRTM)
        dem_crs = dem.rio.crs
        self.logger.debug(f"DEM CRS: {dem_crs}")
        
        # Determine target CRS based on catchment location
        # Get catchment bounds to determine appropriate UTM zone
        catchment_bounds = self.catchment_extent.to_crs('EPSG:4326').total_bounds
        lon_center = (catchment_bounds[0] + catchment_bounds[2]) / 2
        lat_center = (catchment_bounds[1] + catchment_bounds[3]) / 2
        
        # Calculate UTM zone
        utm_zone = int((lon_center + 180) / 6) + 1
        utm_crs = f"EPSG:326{utm_zone:02d}" if lat_center >= 0 else f"EPSG:327{utm_zone:02d}"
        
        self.logger.debug(f"Using UTM CRS: {utm_crs}")
        
        # Reproject DEM and catchment to the same CRS (UTM)
        dem_proj = dem.rio.reproject(utm_crs)
        extent_proj = self.catchment_extent.to_crs(utm_crs)
        
        # Clip raster to catchment boundary
        clip_bound = extent_proj.geometry
        clipped_raster = dem_proj.rio.clip(clip_bound, from_disk=True)
        
        # Handle SRTM no-data values (typically -32768 or 0 for water)
        # Replace no-data values with NaN
        nodata_value = clipped_raster.rio.nodata
        if nodata_value is not None:
            clipped_raster = clipped_raster.where(clipped_raster != nodata_value, other=float('nan'))
        
        # Also handle common SRTM void values
        clipped_raster = clipped_raster.where(clipped_raster != -32768, other=float('nan'))
        clipped_raster = clipped_raster.where(clipped_raster > -9999, other=float('nan'))
        
        # Save clipped raster
        clipped_raster.rio.to_raster(out_raster_path)
        
        self.logger.info(f"DEM clipped and saved. Shape: {clipped_raster.shape}, "
                        f"Elevation range: {float(clipped_raster.min()):.1f} - {float(clipped_raster.max()):.1f} m")
        
        self.dem_data = clipped_raster
        return clipped_raster
    
    #---------------------------------------------------------------------------------

    def calculate_slope_and_aspect(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Calculate slope and aspect from SRTM DEM
        
        Returns
        -------
        Tuple[xr.DataArray, xr.DataArray]
            Slope and aspect data
        """
        self.logger.info("Calculating slope and aspect from SRTM DEM")
        
        # Check if output files already exist
        slope_path = self.get_path('slope.tif')
        aspect_path = self.get_path('aspect.tif')
        
        if slope_path.exists() and aspect_path.exists():
            self.logger.debug("Using existing slope and aspect files")
            slope = rxr.open_rasterio(slope_path).squeeze()
            aspect = rxr.open_rasterio(aspect_path).squeeze()
            self.slope_data = slope
            self.aspect_data = aspect
            return slope, aspect
        
        # Make sure we have DEM data
        if self.dem_data is None:
            self.clip_srtm_dem()
        
        # Create a copy of the DEM data to avoid in-place operations
        xr_dem = self.dem_data.copy(deep=True)
        
        # Handle SRTM-specific no-data values
        nodata_value = xr_dem.rio.nodata
        if nodata_value is not None:
            valid_mask = (xr_dem != nodata_value)
            xr_dem = xr_dem.where(valid_mask)
        
        # Additional SRTM void handling
        xr_dem = xr_dem.where(xr_dem != -32768)
        xr_dem = xr_dem.where(xr_dem > -9999)
        
        # Calculate slope using xrspatial
        self.logger.debug("Calculating slope")
        slope = xrs.slope(xr_dem, name='slope')
        
        # Calculate aspect using xrspatial
        self.logger.debug("Calculating aspect")
        aspect = xrs.aspect(xr_dem, name='aspect')
        
        # SAVE TO FILES - This is what you were missing!
        self.logger.debug(f"Saving slope to {slope_path}")
        slope.rio.to_raster(slope_path)
        
        self.logger.debug(f"Saving aspect to {aspect_path}")
        aspect.rio.to_raster(aspect_path)
        
        # Plot if debug is enabled
        if self.debug:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.title('SRTM DEM')
            plt.imshow(xr_dem, cmap='terrain')
            plt.colorbar(label='Elevation (m)')
            
            plt.subplot(1, 3, 2)
            plt.title('Slope')
            plt.imshow(slope, cmap='viridis')
            plt.colorbar(label='Slope (degrees)')
            
            plt.subplot(1, 3, 3)
            plt.title('Aspect')
            plt.imshow(aspect, cmap='twilight')
            plt.colorbar(label='Aspect (degrees)')
            
            plt.tight_layout()
            
            # Save plot if debug mode
            if self.debug:
                plot_dir = Path(self.model_dir, 'plots')
                plot_path = plot_dir / f"dem_slope_aspect_{self.gauge_id}.png"
                plt.savefig(plot_path)
            
            plt.show()
        
        self.slope_data = slope
        self.aspect_data = aspect
        return slope, aspect

    #---------------------------------------------------------------------------------

    def reclassify_landuse(self) -> xr.DataArray:
        """
        Reclassify the landuse raster from ESA WorldCover V2
        
        Returns
        -------
        xr.DataArray
            Reclassified landuse raster
        """
        self.logger.info("Reclassifying ESA WorldCover landuse")
        
        # Check if output file already exists
        landuse_path = self.get_path('reclassified_landuse.tif')
        if landuse_path.exists():
            self.logger.debug(f"Using existing reclassified landuse at {landuse_path}")
            landuse = rxr.open_rasterio(landuse_path).squeeze()
            # IMPORTANT: Make sure it matches the DEM's grid
            if self.dem_data is not None and landuse.shape != self.dem_data.shape:
                self.logger.warning(f"Resampling landuse to match DEM grid: {landuse.shape} -> {self.dem_data.shape}")
                landuse = landuse.rio.reproject_match(self.dem_data)
            self.landuse_data = landuse
            return landuse
        
        # Extract catchment shape if not already done
        if self.catchment_extent is None:
            self.extract_catchment_shape()
        
        if not self.landuse_dir.exists():
            raise FileNotFoundError(f"ESA WorldCover file not found: {self.landuse_dir}")
        
        # Open ESA WorldCover data
        landuse = rxr.open_rasterio(self.landuse_dir).squeeze()

        # CONVERT TO FLOAT FIRST, THEN HANDLE CLASS 0
        landuse_values = landuse.values.astype(np.float32)  # Convert to float first!
        landuse_values[landuse_values == 0] = np.nan  # Now we can assign NaN
        
        # Update the DataArray with cleaned values
        landuse = xr.DataArray(
            data=landuse_values,
            dims=landuse.dims,
            coords=landuse.coords,
            attrs=landuse.attrs
        )
        
        # Get target CRS from DEM (should be UTM)
        target_crs = self.dem_data.rio.crs if self.dem_data is not None else self.catchment_extent.crs
        landuse = landuse.rio.reproject(target_crs)
        
        # Clip raster to catchment extent
        extent = self.catchment_extent.to_crs(target_crs)
        clip_bound = extent.geometry
        landuse = landuse.rio.clip(clip_bound, from_disk=True)
        
        # IMPORTANT: Make sure the landuse data matches the DEM's grid
        if self.dem_data is not None and landuse.shape != self.dem_data.shape:
            self.logger.warning(f"Resampling landuse to match DEM grid: {landuse.shape} -> {self.dem_data.shape}")
            landuse = landuse.rio.reproject_match(self.dem_data)
        
        # Create a copy of the landuse values to avoid in-place operations
        lista = landuse.values.copy()
        
        # Define reclassification dictionary (same as your original)
        reclassification_dict = {
            '1': (lista == 10) | (lista == 95),  # Forest
            '2': (lista == 20) | (lista == 30) | (lista == 90),  # Open
            '3': (lista == 40),  # Crop
            '4': (lista == 50),  # Built
            '5': (lista == 60) | (lista == 70) | (lista == 100),  # Rock
            '6': (lista == 80),  # Lake
        }
        
        # Apply reclassification
        for value, condition in reclassification_dict.items():
            lista[condition] = int(value)
        
        # Create a new DataArray with the reclassified values
        landuse_new = xr.DataArray(
            data=lista,
            dims=landuse.dims,
            coords=landuse.coords,
            attrs=landuse.attrs
        )
        
        # SAVE RECLASSIFIED LANDUSE - Make sure this is saved!
        self.logger.debug(f"Saving reclassified landuse to {landuse_path}")
        landuse_new.rio.to_raster(landuse_path)
        
        # Plot if debug is enabled
        if self.debug:
            plot_raster(landuse_new, title="Reclassified ESA WorldCover", cmap="tab10")
        
        self.landuse_data = landuse_new
        return landuse_new

    #---------------------------------------------------------------------------------

    def clip_glacier_to_extent(self) -> gpd.GeoDataFrame:
        """
        Clip RGI6 glacier shapefile to catchment extent and save the clipped glacier file
        
        Returns
        -------
        gpd.GeoDataFrame
            Clipped glacier shapefile
        """
        self.logger.info("Clipping RGI6 glacier shapefile to catchment extent")
        
        # Check if output file already exists
        glacier_path = self.get_path('clipped_glacier.shp')
        if glacier_path.exists():
            self.logger.debug(f"Using existing clipped glacier at {glacier_path}")
            clipped_glacier = gpd.read_file(glacier_path)
            # ADD THIS CHECK: If existing file is empty, re-clip
            if len(clipped_glacier) == 0:
                self.logger.warning("Existing clipped glacier file is empty, re-clipping...")
                # Remove the empty files
                import glob
                for f in glob.glob(str(glacier_path).replace('.shp', '.*')):
                    try:
                        os.remove(f)
                        self.logger.debug(f"Removed empty file: {f}")
                    except:
                        pass
            else:
                return clipped_glacier
        
        # Make sure glacier_dir is provided
        if self.glacier_dir is None:
            self.logger.warning("No glacier directory provided in namelist, skipping glacier clipping")
            return gpd.GeoDataFrame()
            
        if not self.glacier_dir.exists():
            self.logger.warning(f"Glacier file not found: {self.glacier_dir}, skipping glacier clipping")
            return gpd.GeoDataFrame()
        
        # Extract catchment shape if not already done
        if self.catchment_extent is None:
            self.extract_catchment_shape()
        
        try:
            # Read the RGI6 glacier shapefile
            self.logger.debug(f"Reading glacier file: {self.glacier_dir}")
            glaciers = gpd.read_file(self.glacier_dir)
            self.logger.debug(f"Loaded {len(glaciers)} glaciers from file")
            
            # Get target CRS from DEM
            target_crs = self.dem_data.rio.crs if self.dem_data is not None else self.catchment_extent.crs
            
            # Reproject both to target CRS
            self.logger.debug(f"Reprojecting to target CRS: {target_crs}")
            glaciers_proj = glaciers.to_crs(target_crs)
            catchment_proj = self.catchment_extent.to_crs(target_crs)
            
            # First check for intersections before clipping
            catchment_geom = catchment_proj.geometry.iloc[0]
            intersecting_glaciers = glaciers_proj[glaciers_proj.intersects(catchment_geom)]
            
            self.logger.debug(f"Found {len(intersecting_glaciers)} glaciers that intersect catchment")
            
            if len(intersecting_glaciers) == 0:
                self.logger.warning("No glaciers intersect with catchment geometry")
                # Create empty shapefile
                empty_gdf = gpd.GeoDataFrame(columns=['RGIId', 'geometry'], crs=target_crs)
                empty_gdf.to_file(glacier_path)
                return empty_gdf
            
            # Clip the glacier shapefile to the catchment extent
            self.logger.debug("Performing geometric clipping...")
            clipped_glacier = gpd.clip(glaciers_proj, catchment_proj)
            
            if len(clipped_glacier) > 0:
                # Save the clipped glacier file
                clipped_glacier.to_file(glacier_path)
                self.logger.info(f"Found {len(clipped_glacier)} glaciers in catchment")
                
                # Log glacier IDs for verification
                if self.debug:
                    glacier_ids = clipped_glacier['RGIId'].tolist()
                    self.logger.debug(f"Glacier IDs: {glacier_ids[:5]}...")  # Show first 5
                    
            else:
                self.logger.warning("Clipping returned empty result")
                # Create empty shapefile for consistency
                empty_gdf = gpd.GeoDataFrame(columns=['RGIId', 'geometry'], crs=target_crs)
                empty_gdf.to_file(glacier_path)
                return empty_gdf
            
            # Plot if debug is enabled
            if self.debug and len(clipped_glacier) > 0:
                plt.figure(figsize=(10, 8))
                catchment_proj.plot(color='none', edgecolor='black', linewidth=2, label='Catchment')
                clipped_glacier.plot(ax=plt.gca(), color='blue', alpha=0.7, label='Glaciers')
                plt.title("RGI6 Glaciers in Catchment")
                plt.legend()
                
                # Save plot if debug mode
                plot_dir = Path(self.model_dir, 'plots')
                plot_path = plot_dir / f"glaciers_{self.gauge_id}.png"
                plt.savefig(plot_path)
                plt.show()
            
            return clipped_glacier
            
        except Exception as e:
            self.logger.error(f"Error clipping glacier shapefile: {e}")
            import traceback
            traceback.print_exc()
            
            # Create empty shapefile as fallback
            empty_gdf = gpd.GeoDataFrame(columns=['RGIId', 'geometry'], crs=target_crs)
            empty_gdf.to_file(glacier_path)
            return empty_gdf

    #---------------------------------------------------------------------------------

    def rasterize_glacier_shapefile(self) -> np.ndarray:
        """
        Rasterize the RGI6 glacier shapefile to match the DEM extent
        Each glacier gets a unique numeric ID for rasterization
        
        Returns
        -------
        np.ndarray
            Rasterized glacier data with unique IDs per glacier
        """
        self.logger.info("Rasterizing RGI6 glacier shapefile")
        
        # Check if DEM data exists
        if self.dem_data is None:
            self.logger.error("DEM data not loaded. Cannot rasterize glaciers.")
            return np.full((1, 1), np.nan, dtype='float32')
        
        # Get paths
        glacier_path = self.get_path('clipped_glacier.shp')
        glacier_raster_path = self.get_path('glacier_raster.tif')
        
        # Check if output file already exists
        if glacier_raster_path.exists():
            self.logger.debug(f"Using existing glacier raster at {glacier_raster_path}")
            with rasterio.open(glacier_raster_path) as src:
                glacier_raster = src.read(1)
            self.glacier_data = glacier_raster
            return glacier_raster
        
        # Ensure glacier clipping is done
        if not glacier_path.exists():
            self.logger.debug("Clipped glacier file not found, running clipping...")
            self.clip_glacier_to_extent()
        
        # Read the shapefile
        try:
            gdf = gpd.read_file(glacier_path)
        except Exception as e:
            self.logger.error(f"Error reading glacier file: {e}")
            glacier_raster = np.full(self.dem_data.shape, np.nan, dtype='float32')
            self.glacier_data = glacier_raster
            return glacier_raster
        
        # Initialize glacier array with NaN (same shape as DEM)
        glacier_shape = self.dem_data.shape
        glacier_raster = np.full(glacier_shape, np.nan, dtype='float32')
        
        self.logger.debug(f"Initializing glacier array with shape: {glacier_shape}")
        
        if len(gdf) == 0:
            self.logger.warning("No glaciers found in the catchment")
            # Save empty glacier raster
            glacier_xr = xr.DataArray(
                data=glacier_raster,
                dims=self.dem_data.dims,
                coords=self.dem_data.coords,
                attrs=self.dem_data.attrs
            )
            glacier_xr.rio.to_raster(glacier_raster_path)
            self.glacier_data = glacier_raster
            return glacier_raster
        
        self.logger.info(f"Rasterizing {len(gdf)} glaciers with unique IDs")
        
        try:
            # Get DEM transform and CRS for rasterization
            dem_transform = self.dem_data.rio.transform()
            dem_crs = self.dem_data.rio.crs
            
            # Ensure glaciers are in the same CRS as DEM
            if gdf.crs != dem_crs:
                self.logger.debug(f"Reprojecting glaciers from {gdf.crs} to {dem_crs}")
                gdf = gdf.to_crs(dem_crs)
            
            # ✅ FIX: Create unique numeric IDs for each glacier
            # Extract numeric part from RGI IDs for rasterization
            gdf['numeric_id'] = range(1, len(gdf) + 1)  # Simple sequential IDs
            
            # Store mapping between numeric IDs and RGI IDs
            self.glacier_id_mapping = dict(zip(gdf['numeric_id'], gdf['RGIId']))
            
            # Create shapes for rasterization with unique IDs
            shapes = [(geom, numeric_id) for geom, numeric_id in zip(gdf.geometry, gdf['numeric_id'])]
            
            self.logger.debug(f"Rasterizing {len(shapes)} glacier geometries with unique IDs")
            
            # Rasterize using rasterio.features.rasterize
            from rasterio.features import rasterize
            
            rasterized = rasterize(
                shapes=shapes,
                out_shape=glacier_shape,
                transform=dem_transform,
                fill=np.nan,  # Background value
                dtype='float32'
            )
            
            # Count glacier pixels
            glacier_pixels = np.count_nonzero(~np.isnan(rasterized))
            unique_glacier_ids = np.unique(rasterized[~np.isnan(rasterized)])
            
            self.logger.info(f"Successfully rasterized {glacier_pixels:,} glacier pixels")
            self.logger.info(f"Created {len(unique_glacier_ids)} unique glacier HRUs")
            
            # Log glacier ID mapping
            if self.debug:
                self.logger.debug("Glacier ID mapping:")
                for numeric_id, rgi_id in self.glacier_id_mapping.items():
                    self.logger.debug(f"  {numeric_id} -> {rgi_id}")
            
            # Store result
            self.glacier_data = rasterized
            
            # Save glacier raster
            self.logger.debug(f"Saving glacier raster to: {glacier_raster_path}")
            glacier_xr = xr.DataArray(
                data=rasterized,
                dims=self.dem_data.dims,
                coords=self.dem_data.coords,
                attrs=self.dem_data.attrs
            )
            glacier_xr.rio.to_raster(glacier_raster_path)
            
            # Also save the mapping as a CSV for reference
            mapping_df = pd.DataFrame(list(self.glacier_id_mapping.items()), 
                                    columns=['numeric_id', 'RGIId'])
            mapping_df.to_csv(self.get_path('glacier_id_mapping.csv'), index=False)
            
            return rasterized
            
        except Exception as e:
            self.logger.error(f"Error during glacier rasterization: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty array on error and save it
            glacier_xr = xr.DataArray(
                data=glacier_raster,
                dims=self.dem_data.dims,
                coords=self.dem_data.coords,
                attrs=self.dem_data.attrs
            )
            glacier_xr.rio.to_raster(glacier_raster_path)
            self.glacier_data = glacier_raster
            return glacier_raster

    #---------------------------------------------------------------------------------

    def discretize_catchment(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Discretize catchment based on given criteria with flexible glacier handling
        
        Returns
        -------
        Tuple[pd.DataFrame, np.ndarray]
            Tuple of (hru_df, map_unit_ids)
        """
        self.logger.info(f"Discretizing catchment using criteria: {self.criteria}")
        
        # Make sure all necessary data is available
        if self.dem_data is None:
            self.clip_srtm_dem()
        
        if self.slope_data is None or self.aspect_data is None:
            self.calculate_slope_and_aspect()
        
        if self.landuse_data is None and 'landuse' in self.criteria:
            self.reclassify_landuse()
        
        if self.glacier_data is None and self.coupled:
            self.rasterize_glacier_shapefile()

        # ✅ ADD DEBUG CODE HERE - after data loading but before processing
        if self.debug:
            print("🔍 Debugging catchment processing...")
            
            # Check if files exist
            import os
            from pathlib import Path
            
            topo_dir = Path(self.model_dir, 'topo_files')
            print(f"📁 Topo directory: {topo_dir}")
            print(f"   - Exists: {topo_dir.exists()}")
            
            if topo_dir.exists():
                files = list(topo_dir.glob('*'))
                print(f"   - Files: {[f.name for f in files]}")
                
                # Check glacier raster specifically
                glacier_raster = topo_dir / 'glacier_raster.tif'
                if glacier_raster.exists():
                    import rasterio
                    with rasterio.open(glacier_raster) as src:
                        data = src.read(1)
                        unique_vals = np.unique(data[~np.isnan(data)])
                        print(f"   - Glacier raster unique values: {unique_vals}")
                        print(f"   - Glacier raster shape: {data.shape}")
            
            # Debug loaded data arrays
            print(f"📊 Data array shapes:")
            print(f"   - DEM: {self.dem_data.shape if self.dem_data is not None else 'None'}")
            print(f"   - Slope: {self.slope_data.shape if self.slope_data is not None else 'None'}")
            print(f"   - Aspect: {self.aspect_data.shape if self.aspect_data is not None else 'None'}")
            print(f"   - Landuse: {self.landuse_data.shape if self.landuse_data is not None else 'None'}")
            print(f"   - Glacier: {self.glacier_data.shape if self.glacier_data is not None else 'None'}")
            
            # Check glacier mapping
            if hasattr(self, 'glacier_id_mapping'):
                print(f"🧊 Glacier ID mapping: {self.glacier_id_mapping}")
            else:
                print("🧊 No glacier ID mapping found")
        
        # If not coupled, treat glaciers as landuse class 7
        if not self.coupled:
            glacier_mask = ~np.isnan(self.glacier_data)
            self.landuse_data.values[glacier_mask] = 7
            self.logger.info(f"Marked {np.count_nonzero(glacier_mask)} glacier cells as landuse class 7")
        
        # Create criteria dictionary
        criteria_dict = {}
        
        # Helper function to calculate range bins
        def create_range_bins(data, distance):
            min_val = np.nanmin(data)
            min_val = min_val - (min_val % distance)
            max_val = np.nanmax(data)
            max_val = max_val + (distance - max_val % distance)
            values = np.arange(min_val, max_val + distance, distance)
            return [values[i:i+2] for i in range(len(values)-1)]
        
        # Fill criteria dictionary based on selected criteria
        if 'elevation' in self.criteria:
            criteria_dict['elevation'] = create_range_bins(self.dem_data, self.elevation_distance)
        
        if 'slope' in self.criteria:
            criteria_dict['slope'] = create_range_bins(self.slope_data, self.slope_distance)
        
        if 'aspect' in self.criteria:
            criteria_dict['aspect'] = ['N', 'E', 'S', 'W']
        
        if 'landuse' in self.criteria:
            criteria_dict['landuse'] = [1, 2, 3, 4, 5, 6] + ([7] if not self.coupled else [])
        
        # ✅ ADD DEBUG CODE HERE - after criteria setup
        if self.debug:
            print(f"🎯 Criteria dictionary: {criteria_dict}")
            for key, values in criteria_dict.items():
                print(f"   - {key}: {len(values)} values")
        
        # Initialize results storage
        columns = {
            'Unit ID': [],
            'Elevation': [], 'Elevation Min': [], 'Elevation Max': [],
            'Slope': [], 'Slope Min': [], 'Slope Max': [],
            'Aspect Class': [], 'Land Use Class': [], 'Glacier Class': [],
            'Latitude': [], 'Longitude': []
        }
        
        # Initialize HRU ID map
        map_unit_ids = np.zeros(self.dem_data.shape)
        unit_id = 1
        
        # Define aspect ranges for easier lookup
        aspect_ranges = {
            'N': ((self.aspect_data >= 315) & (self.aspect_data <= 360)) | ((self.aspect_data >= 0) & (self.aspect_data < 45)),
            'E': ((self.aspect_data >= 45) & (self.aspect_data < 135)),
            'S': ((self.aspect_data >= 135) & (self.aspect_data < 225)),
            'W': ((self.aspect_data >= 225) & (self.aspect_data < 315))
        }
        
        # Handle glaciers if coupled mode is active
        if self.coupled and self.glacier_data is not None:
            glacier_data_np = self.glacier_data
            unique_glaciers = np.unique(glacier_data_np[~np.isnan(glacier_data_np)])
            
            self.logger.info(f"Creating {len(unique_glaciers)} glacier HRUs")
            
            for numeric_glacier_id in unique_glaciers:
                mask_glacier = (glacier_data_np == numeric_glacier_id)
                if np.count_nonzero(mask_glacier) > 0:
                    # Assign HRU ID to glacier
                    map_unit_ids[mask_glacier] = unit_id
                    
                    # Calculate centroid
                    rows, cols = np.where(mask_glacier)
                    with rasterio.open(self.get_path('clipped_dem.tif')) as src:
                        transform = src.transform
                        lon, lat = xy(transform, np.mean(cols), np.mean(rows))
                    
                    # ✅ FIX: Get actual RGI ID from mapping
                    if hasattr(self, 'glacier_id_mapping') and numeric_glacier_id in self.glacier_id_mapping:
                        rgi_id = self.glacier_id_mapping[numeric_glacier_id]
                    else:
                        # Fallback if mapping doesn't exist
                        rgi_id = f"RGI60-11.{int(numeric_glacier_id):05d}"
                    
                    # Store glacier HRU attributes
                    columns['Unit ID'].append(unit_id)
                    columns['Glacier Class'].append(rgi_id)  # ✅ Use actual RGI ID
                    columns['Latitude'].append(lat)
                    columns['Longitude'].append(lon)
                    
                    # Fill other fields with NaN for glacier HRUs
                    for field in ['Elevation', 'Elevation Min', 'Elevation Max', 
                                'Slope', 'Slope Min', 'Slope Max', 
                                'Aspect Class', 'Land Use Class']:
                        columns[field].append(np.nan)
                    
                    self.logger.debug(f"Created glacier HRU {unit_id} for {rgi_id}")
                    unit_id += 1
        
            # Plot initial results if debug mode
            if self.debug:
                plot_map(map_unit_ids, title="Glacier HRUs")
        
        # Generate all combinations of criteria
        combinations = list(itertools.product(*criteria_dict.values()))
        combinations_keys = list(criteria_dict.keys())
        
        # ✅ ADD DEBUG CODE HERE - after combinations generation
        self.logger.info(f"Criteria dictionary: {criteria_dict}")
        self.logger.info(f"Number of combinations: {len(combinations)}")
        
        if len(combinations) > 1000:
            self.logger.warning(f"Very large number of combinations ({len(combinations)}), this may take a long time!")
            
        # Show first few combinations as examples
        if len(combinations) > 0:
            self.logger.debug(f"First 5 combinations: {combinations[:5]}")
        
        self.logger.debug(f"Processing {len(combinations)} combinations of criteria")
        
        # Process each combination to create HRUs
        for i, combination in enumerate(combinations):
            if self.debug and i % 50 == 0:
                self.logger.debug(f"Processing combination {i+1}/{len(combinations)}")
            
            # Create mask for current combination
            mask_unit = np.ones(self.dem_data.shape, dtype=bool)
            mask_unit[self.dem_data == -32768.0] = False
            
            # Apply each criterion to the mask
            for criterion_name, criterion in zip(combinations_keys, combination):
                if criterion_name == 'elevation':
                    # Extract data values directly to NumPy for masking
                    dem_values = self.dem_data.values
                    mask_unit &= (dem_values >= criterion[0]) & (dem_values < criterion[1])
                elif criterion_name == 'slope':
                    # Extract data values directly to NumPy for masking
                    slope_values = self.slope_data.values
                    mask_unit &= (slope_values >= criterion[0]) & (slope_values < criterion[1])
                elif criterion_name == 'aspect':
                    # Use the pre-calculated aspect ranges
                    mask_unit &= aspect_ranges[criterion].values
                elif criterion_name == 'landuse':
                    # Extract data values directly to NumPy for masking
                    landuse_values = self.landuse_data.values
                    mask_unit &= (landuse_values == criterion)
            
            # If coupled mode, exclude glacier cells
            if self.coupled and self.glacier_data is not None:
                mask_unit &= (map_unit_ids == 0)
                
            # Skip empty HRUs
            if np.count_nonzero(mask_unit) == 0:
                continue
                
            # Assign unit ID to this HRU
            map_unit_ids[mask_unit] = unit_id
            
            # Get criterion values for results
            columns['Unit ID'].append(unit_id)
            
            # Extract and store criterion values
            for criterion_name in self.criteria:
                if criterion_name == 'elevation':
                    idx = combinations_keys.index('elevation')
                    elev_range = combination[idx]
                    columns['Elevation'].append(round(float(np.mean(elev_range)), 2))
                    columns['Elevation Min'].append(round(float(elev_range[0]), 2))
                    columns['Elevation Max'].append(round(float(elev_range[1]), 2))
                elif criterion_name == 'slope':
                    idx = combinations_keys.index('slope')
                    slope_range = combination[idx]
                    columns['Slope'].append(round(float(np.mean(slope_range)), 2))
                    columns['Slope Min'].append(round(float(slope_range[0]), 2))
                    columns['Slope Max'].append(round(float(slope_range[1]), 2))
                elif criterion_name == 'aspect':
                    idx = combinations_keys.index('aspect')
                    columns['Aspect Class'].append(combination[idx])
                elif criterion_name == 'landuse':
                    idx = combinations_keys.index('landuse')
                    columns['Land Use Class'].append(combination[idx])
            
            # Fill in missing fields with NaN if criterion wasn't used
            for field in ['Elevation', 'Elevation Min', 'Elevation Max', 
                        'Slope', 'Slope Min', 'Slope Max', 
                        'Aspect Class', 'Land Use Class']:
                if field not in [f for c in self.criteria for f in [c, f"{c} Min", f"{c} Max"]]:
                    if field not in columns or len(columns[field]) < len(columns['Unit ID']):
                        columns[field].append(np.nan)
                        
            # Add glacier class (NaN for non-glacier HRUs)
            columns['Glacier Class'].append(np.nan)
            
            # Calculate centroid for geography
            rows, cols = np.where(mask_unit)
            with rasterio.open(self.get_path('clipped_dem.tif')) as src:
                transform = src.transform
                lon, lat = xy(transform, np.mean(cols), np.mean(rows))
            columns['Latitude'].append(lat)
            columns['Longitude'].append(lon)
            
            unit_id += 1
        
        self.logger.info(f"Created {unit_id-1} HRUs")
        
        # Plot final results if debug mode
        if self.debug:
            plot_map(map_unit_ids, title="Final HRUs")
        
        # Save HRU raster
        self.logger.debug("Saving HRU raster")
        hru_raster_dir = self.get_path('HRU_raster.tif')
        
        # Get the transform and CRS from the DEM
        with rasterio.open(self.get_path('clipped_dem.tif')) as src:
            transform = src.transform
            crs = src.crs
            
            # Create raster profile
            profile = src.profile
            profile.update(
                dtype=rasterio.float32,
                count=1,
                compress='lzw',
                nodata=-9999
            )
        
        # Write the map_unit_ids array to a GeoTIFF file
        with rasterio.open(hru_raster_dir, 'w', **profile) as dst:
            dst.write(map_unit_ids.astype(rasterio.float32), 1)
        
        # Create DataFrame from columns that have data
        valid_columns = {k: v for k, v in columns.items() if len(v) > 0}
        hru_df = pd.DataFrame(valid_columns)
        
        self.hru_df = hru_df
        self.map_unit_ids = map_unit_ids
        
        return hru_df, map_unit_ids

    #---------------------------------------------------------------------------------

    def calculate_hru_statistics(self) -> pd.DataFrame:
        """
        Calculate HRU statistics directly from raster data using the HRU ID map
        
        Returns
        -------
        pd.DataFrame
            DataFrame with statistics for each HRU
        """
        self.logger.info("Calculating HRU statistics")
        
        # Make sure discretization has been done
        if self.map_unit_ids is None or self.hru_df is None:
            self.discretize_catchment()
        
        # Get cell area from DEM
        with rasterio.open(self.get_path('clipped_dem.tif')) as src:
            transform = src.transform
            cell_area = abs(transform[0] * transform[4])  # Area in square meters
        
        # Get unique HRU IDs (excluding 0 and NaN)
        map_ids = self.map_unit_ids
        
        # Use numpy masking to safely handle the comparison
        mask = (map_ids > 0) & ~np.isnan(map_ids)
        if not np.any(mask):
            self.logger.warning("No valid HRUs found in the map")
            return pd.DataFrame()
            
        unique_ids = np.unique(map_ids[mask])
        
        # Create empty list for results
        results = []
        transformer = Transformer.from_crs(self.dem_data.rio.crs, "EPSG:4326", always_xy=True)
        
        self.logger.debug(f"Processing statistics for {len(unique_ids)} HRUs")
        
        # Process each HRU
        for hru_id in unique_ids:
            # Create mask for current HRU
            hru_mask = (map_ids == hru_id)
            
            # Skip if no pixels in this HRU (shouldn't happen but just to be safe)
            if not np.any(hru_mask):
                continue
            
            # Calculate area
            area = np.sum(hru_mask) * cell_area
            
            # ✅ FIXED DEM STATS - Handle small HRUs properly
            dem_masked = self.dem_data.values[hru_mask]
            
            # First, try to get valid elevation values (exclude SRTM nodata)
            valid_dem = dem_masked[(dem_masked != -32768) & ~np.isnan(dem_masked)]
            
            if len(valid_dem) > 0:
                # Use valid elevation data
                mean_elevation = np.nanmean(valid_dem)
            else:
                # If no valid elevation data in this HRU, use nearest neighbor interpolation
                self.logger.warning(f"HRU {hru_id} has no valid elevation data, using nearest neighbor interpolation")
                
                # Get HRU centroid
                rows, cols = np.where(hru_mask)
                center_row, center_col = np.mean(rows), np.mean(cols)
                
                # Find nearest valid elevation value
                dem_full = self.dem_data.values
                valid_mask = (dem_full != -32768) & ~np.isnan(dem_full)
                
                if np.any(valid_mask):
                    # Get coordinates of all valid pixels
                    valid_rows, valid_cols = np.where(valid_mask)
                    
                    # Calculate distances to HRU centroid
                    distances = np.sqrt((valid_rows - center_row)**2 + (valid_cols - center_col)**2)
                    
                    # Find nearest valid pixel
                    nearest_idx = np.argmin(distances)
                    mean_elevation = dem_full[valid_rows[nearest_idx], valid_cols[nearest_idx]]
                    
                    self.logger.debug(f"HRU {hru_id}: interpolated elevation {mean_elevation:.1f}m from nearest neighbor")
                else:
                    # Last resort: use a default elevation (shouldn't happen)
                    mean_elevation = 1000.0  # Default elevation in meters
                    self.logger.warning(f"HRU {hru_id}: using default elevation {mean_elevation}m")
            
            # Slope stats (keep your existing logic)
            slope_masked = self.slope_data.values[hru_mask]
            slope_masked = slope_masked[~np.isnan(slope_masked)]
            mean_slope = np.nanmean(slope_masked) if len(slope_masked) > 0 else np.nan
            
            # Aspect stats (circular mean) (keep your existing logic)
            aspect_masked = self.aspect_data.values[hru_mask]
            aspect_masked = aspect_masked[~np.isnan(aspect_masked)]
            
            if len(aspect_masked) > 0:
                aspect_rad = np.deg2rad(aspect_masked)
                mean_aspect = np.rad2deg(np.arctan2(
                    np.nanmean(np.sin(aspect_rad)),
                    np.nanmean(np.cos(aspect_rad))
                ))
                mean_aspect = (mean_aspect + 360) % 360
            else:
                mean_aspect = np.nan
            
            # Rest of your code remains the same...
            # Get values from hru_df for this HRU
            hru_row = self.hru_df[self.hru_df['Unit ID'] == hru_id]
            
            # Get landuse class (with default of 8 for NaN)
            landuse_class = np.nan
            if 'Land Use Class' in self.hru_df.columns and not hru_row.empty:
                landuse_class = hru_row['Land Use Class'].iloc[0]
                if pd.isna(landuse_class):
                    landuse_class = 8
            
            # Get glacier class
            glacier_class = np.nan
            if 'Glacier Class' in self.hru_df.columns and not hru_row.empty:
                glacier_class = hru_row['Glacier Class'].iloc[0]
            
            # Calculate centroid coordinates
            rows, cols = np.where(hru_mask)
            if len(rows) > 0:
                x, y = transform * (np.mean(cols), np.mean(rows))
                lon, lat = transformer.transform(x, y)
            else:
                lon, lat = np.nan, np.nan
            
            # Get min/max elevation from HRU dataframe if available
            elev_min = hru_row['Elevation Min'].iloc[0] if 'Elevation Min' in hru_row.columns and not hru_row.empty else np.nan
            elev_max = hru_row['Elevation Max'].iloc[0] if 'Elevation Max' in hru_row.columns and not hru_row.empty else np.nan
            
            # Store results
            results.append({
                'HRU_ID': int(hru_id),
                'Area_m2': float(area),
                'Area_km2': float(area / 1_000_000),
                'Elev_Mean': float(mean_elevation),  # Now guaranteed to be a valid value
                'Elev_Min': elev_min,
                'Elev_Max': elev_max,
                'Slope_deg': float(mean_slope),
                'Aspect_deg': float(mean_aspect),
                'Landuse_Cl': landuse_class,
                'Latitude': float(lat),
                'Longitude': float(lon),
                'Glacier_Cl': glacier_class
            })
        
        # Rest of your method remains the same...
        # Create DataFrame from results
        stats_df = pd.DataFrame(results)
        
        # Add aspect class if available in hru_df
        if 'Aspect Class' in self.hru_df.columns:
            # Merge aspect class from original dataframe
            aspect_mapping = self.hru_df.set_index('Unit ID')['Aspect Class'].to_dict()
            stats_df['Aspect_Cl'] = stats_df['HRU_ID'].map(aspect_mapping)
        
        # Sort by HRU ID and fix NaN values
        stats_df = stats_df.sort_values(by='HRU_ID')
        
        # Handle any NaN values safely
        if 'Slope_deg' in stats_df.columns:
            stats_df['Slope_deg'] = stats_df['Slope_deg'].interpolate(method='nearest').fillna(0)
            
        if 'Aspect_deg' in stats_df.columns:
            stats_df['Aspect_deg'] = stats_df['Aspect_deg'].interpolate(method='nearest').fillna(0)
        
        # ✅ ADDITIONAL CHECK: Ensure no NaN elevations remain
        if stats_df['Elev_Mean'].isna().any():
            self.logger.warning("Some HRUs still have NaN elevation, using interpolation")
            stats_df['Elev_Mean'] = stats_df['Elev_Mean'].interpolate(method='linear').fillna(stats_df['Elev_Mean'].mean())
        
        # Print summary statistics
        self.logger.info(f"Total number of HRUs: {len(stats_df)}")
        self.logger.info(f"Total area: {stats_df['Area_km2'].sum():.2f} km²")
        self.logger.info(f"Elevation range: {stats_df['Elev_Mean'].min():.2f} - {stats_df['Elev_Mean'].max():.2f} m")
        self.logger.info(f"HRUs with valid elevation: {(~stats_df['Elev_Mean'].isna()).sum()}/{len(stats_df)}")
        
        # Visualize results if in debug mode
        if self.debug:
            self._plot_hru_statistics(stats_df)
        
        # Save the stats dataframe
        stats_df.to_csv(self.get_path('HRU_statistics.csv'), index=False)
        
        self.hru_stats = stats_df
        return stats_df
    
    #---------------------------------------------------------------------------------

    def _plot_hru_statistics(self, stats_df: pd.DataFrame) -> None:
        """
        Plot HRU statistics
        
        Parameters
        ----------
        stats_df : pd.DataFrame
            DataFrame with HRU statistics
        """
        self.logger.debug("Plotting HRU statistics")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot key statistics
        axes[0, 0].bar(stats_df['HRU_ID'], stats_df['Elev_Mean'])
        axes[0, 0].set_title('Mean Elevation by HRU')
        axes[0, 0].set_xlabel('HRU ID')
        axes[0, 0].set_ylabel('Elevation (m)')
        
        axes[0, 1].bar(stats_df['HRU_ID'], stats_df['Area_km2'])
        axes[0, 1].set_title('Area by HRU')
        axes[0, 1].set_xlabel('HRU ID')
        axes[0, 1].set_ylabel('Area (km²)')
        
        axes[1, 0].bar(stats_df['HRU_ID'], stats_df['Slope_deg'])
        axes[1, 0].set_title('Mean Slope by HRU')
        axes[1, 0].set_xlabel('HRU ID')
        axes[1, 0].set_ylabel('Slope (degrees)')
        
        # Land use distribution if available
        if 'Landuse_Cl' in stats_df.columns and not stats_df['Landuse_Cl'].isna().all():
            landuse_counts = stats_df['Landuse_Cl'].value_counts()
            axes[1, 1].bar(landuse_counts.index.astype(str), landuse_counts.values)
            axes[1, 1].set_title('Land Use Class Distribution')
            axes[1, 1].set_xlabel('Land Use Class')
            axes[1, 1].set_ylabel('Number of HRUs')
        
        plt.tight_layout()
        
        # Save plot if debug mode
        plot_dir = Path(self.model_dir, 'plots')
        plot_path = plot_dir / f"hru_statistics_{self.gauge_id}.png"
        plt.savefig(plot_path)
        
        plt.show()

    #---------------------------------------------------------------------------------

    def create_hru_shapefile(self) -> gpd.GeoDataFrame:
        """
        Create a shapefile from the HRU map and HRU statistics
        
        Returns
        -------
        gpd.GeoDataFrame
            HRU shapefile
        """
        self.logger.info("Creating HRU shapefile")
        
        # Make sure we have HRU stats
        if self.hru_stats is None:
            self.calculate_hru_statistics()
        
        # Check if output file already exists
        output_shapefile_path = self.get_path('HRU.shp')
        if output_shapefile_path.exists():
            self.logger.debug(f"Using existing HRU shapefile at {output_shapefile_path}")
            return gpd.read_file(output_shapefile_path)
        
        # Read the DEM to get the transform and CRS
        with rasterio.open(self.get_path('clipped_dem.tif')) as src:
            transform = src.transform
            crs = src.crs
        
        # Convert the HRU map to polygons
        mask = self.map_unit_ids != 0  # Exclude no-data areas
        shapes_generator = shapes(self.map_unit_ids, mask=mask, transform=transform)
        
        # Create a dictionary to store polygons by HRU ID
        hru_polygons = {}
        for geom, value in shapes_generator:
            hru_id = int(value)
            if hru_id not in hru_polygons:
                hru_polygons[hru_id] = []
            hru_polygons[hru_id].append(shape(geom))
        
        # Aggregate polygons by HRU ID
        aggregated_polygons = []
        hru_ids = []
        for hru_id, polygons in hru_polygons.items():
            if len(polygons) > 1:
                aggregated_polygons.append(MultiPolygon(polygons))
            else:
                aggregated_polygons.extend(polygons)
                
            hru_ids.append(hru_id)
        
        # Create a GeoDataFrame from the aggregated polygons and HRU IDs
        gdf = gpd.GeoDataFrame({'HRU_ID': hru_ids, 'geometry': aggregated_polygons})
        
        # Merge the GeoDataFrame with the HRU stats
        gdf = gdf.merge(self.hru_stats, on='HRU_ID')
        
        # Set the CRS to match the DEM
        gdf.set_crs(crs, inplace=True)
        
        # Save the GeoDataFrame as a shapefile
        self.logger.debug(f"Saving HRU shapefile to {output_shapefile_path}")
        gdf.to_file(output_shapefile_path, driver='ESRI Shapefile')
        
        # Plot if in debug mode
        if self.debug:
            plt.figure(figsize=(10, 8))
            gdf.plot(column='Landuse_Cl', cmap='tab10', legend=True, 
                    legend_kwds={'label': 'Land Use Class'})
            plt.title(f"HRU Shapefile for Gauge ID {self.gauge_id}")
            
            # Save plot if debug mode
            plot_dir = Path(self.model_dir, 'plots')
            plot_path = plot_dir / f"hru_shapefile_{self.gauge_id}.png"
            plt.savefig(plot_path)
            
            plt.show()
        
        return gdf

    #---------------------------------------------------------------------------------

    def create_hru_table(self) -> pd.DataFrame:
        """
        Make HRU table with all the attributes necessary for Raven
        
        Returns
        -------
        pd.DataFrame
            HRU table for Raven
        """
        self.logger.info("Creating HRU table for Raven")
        
        # Make sure we have HRU stats
        if self.hru_stats is None:
            self.calculate_hru_statistics()
        
        # Create DataFrame for HRU attributes
        all_HRUs = pd.DataFrame()
        
        all_HRUs['SLOPE'] = self.hru_stats['Slope_deg']
        all_HRUs['ASPECT'] = self.hru_stats['Aspect_deg']
        all_HRUs['AREA'] = self.hru_stats['Area_km2']
        all_HRUs['ELEVATION'] = self.hru_stats['Elev_Mean']
        all_HRUs['LATITUDE'] = self.hru_stats['Latitude']
        all_HRUs['LONGITUDE'] = self.hru_stats['Longitude']
        all_HRUs['ID'] = self.hru_stats['HRU_ID']
        
        # Set default values
        all_HRUs['LAND_USE_CLASS'] = 'DEFAULT_L'
        all_HRUs['VEG_CLASS'] = 'DEFAULT_V'
        all_HRUs['SOIL_PROFILE'] = 'DEFAULT_P'
        all_HRUs['AQUIFER_PROFILE'] = '[NONE]'
        all_HRUs['TERRAIN_CLASS'] = '[NONE]'
        
        # Assign landuse classes
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 1, 'LAND_USE_CLASS'] = 'FOREST'
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 2, 'LAND_USE_CLASS'] = 'OPEN'
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 3, 'LAND_USE_CLASS'] = 'OPEN'
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 4, 'LAND_USE_CLASS'] = 'BUILT'
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 5, 'LAND_USE_CLASS'] = 'ROCK'
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 6, 'LAND_USE_CLASS'] = 'LAKE'
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 7, 'LAND_USE_CLASS'] = 'GLACIER'
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 8, 'LAND_USE_CLASS'] = 'MASKED_GLACIER'
        
        # Assign soil classes 
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 5, 'SOIL_PROFILE'] = 'ROCK'
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 6, 'SOIL_PROFILE'] = 'LAKE'
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 7, 'SOIL_PROFILE'] = 'GLACIER'
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 8, 'SOIL_PROFILE'] = 'MASKED_GLACIER'
        
        # Assign vegetation classes
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 1, 'VEG_CLASS'] = 'FOREST'
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 2, 'VEG_CLASS'] = 'GRAS'
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 3, 'VEG_CLASS'] = 'CROP'
        
        # Make final dataframe in Raven HRU format
        HRU = pd.DataFrame(
            {':ATTRIBUTES': all_HRUs['ID'],
             'AREA': all_HRUs['AREA'],
             'ELEVATION': all_HRUs['ELEVATION'],
             'LATITUDE': all_HRUs['LATITUDE'],
             'LONGITUDE': all_HRUs['LONGITUDE'],
             'BASIN_ID': [1] * len(all_HRUs),
             'LAND_USE_CLASS': all_HRUs['LAND_USE_CLASS'],
             'VEG_CLASS': all_HRUs['VEG_CLASS'],
             'SOIL_PROFILE': all_HRUs['SOIL_PROFILE'],
             'AQUIFER_PROFILE': all_HRUs['AQUIFER_PROFILE'],
             'TERRAIN_CLASS': all_HRUs['TERRAIN_CLASS'],
             'SLOPE': all_HRUs['SLOPE'],
             'ASPECT': all_HRUs['ASPECT']
            })
        
        # Save HRU table to CSV
        self.logger.debug("Saving HRU table to CSV")
        HRU.to_csv(self.get_path('HRU_table.csv'), index=False)
        
        # Also save as txt file with spaces as separator for Raven
        with open(self.get_path('HRU.txt'), 'w') as f:
            # Write header
            f.write(' '.join(HRU.columns) + '\n')
            # Write data
            for _, row in HRU.iterrows():
                f.write(' '.join(map(str, row)) + '\n')
        
        return HRU

    #---------------------------------------------------------------------------------

    def get_hrus_by_elevation_band(self) -> Dict[str, List[int]]:
        """
        Load HRU shapefile and create a list of HRU IDs for each elevation band
        
        Returns
        -------
        Dict[str, List[int]]
            Dictionary with elevation bands as keys and lists of HRU IDs as values
        """
        self.logger.info("Getting HRUs by elevation band")
        
        # Load HRU shapefile
        hru_path = self.get_path('HRU.shp')
        
        if not hru_path.exists():
            self.create_hru_shapefile()
        
        hru_gdf = gpd.read_file(hru_path)
        
        # Filter out landuse classes 7 and 8 if needed
        filtered_hru = hru_gdf[~hru_gdf['Landuse_Cl'].isin([7, 8])]
        
        # Create elevation band labels
        filtered_hru['ElevationBand'] = filtered_hru.apply(
            lambda row: f"{int(row['Elev_Min'])}-{int(row['Elev_Max'])}m", axis=1)
        
        # Group by elevation band and collect HRU IDs
        hrus_by_band = {}
        for band in filtered_hru['ElevationBand'].unique():
            band_hrus = filtered_hru[filtered_hru['ElevationBand'] == band]['HRU_ID'].tolist()
            hrus_by_band[band] = band_hrus
        
        # Print summary
        self.logger.info(f"Found {len(hrus_by_band)} elevation bands with a total of {len(filtered_hru)} HRUs")
        for band in sorted(hrus_by_band.keys(), key=lambda x: int(x.split('-')[0])):
            self.logger.info(f"  {band}: {len(hrus_by_band[band])} HRUs")
        
        return hrus_by_band
    
    #---------------------------------------------------------------------------------

    def process_catchment(self) -> pd.DataFrame:
        """
        Process the catchment from start to finish
        
        Returns
        -------
        pd.DataFrame
            HRU table for Raven
        """
        self.logger.info(f"Processing catchment {self.gauge_id}")
        
        try:
            # Step 1: Extract catchment shape
            self.extract_catchment_shape()
            
            # Step 2: Clip SRTM DEM to catchment
            self.clip_srtm_dem()
            
            # Step 3: Calculate slope and aspect
            self.calculate_slope_and_aspect()
            
            # Step 4: Reclassify landuse if needed
            if 'landuse' in self.criteria:
                self.reclassify_landuse()
            
            # Step 5: Handle glaciers if coupled
            self.clip_glacier_to_extent()
            self.rasterize_glacier_shapefile()
            
            # Step 6: Discretize catchment
            self.discretize_catchment()
            
            # Step 7: Calculate HRU statistics
            self.calculate_hru_statistics()
            
            # Step 8: Create HRU shapefile
            self.create_hru_shapefile()
            
            # Step 9: Create HRU table
            hru_table = self.create_hru_table()
            
            self.logger.info(f"Catchment processing completed for gauge ID {self.gauge_id}")
            
            return hru_table
            
        except Exception as e:
            self.logger.error(f"Error processing catchment: {str(e)}")
            raise

#---------------------------------------------------------------------------------

def main():
    """Example usage of the CatchmentProcessor"""
    
    # Path to namelist configuration file
    namelist_path = '/home/jberg/OneDrive/Raven-world/namelist.yaml'
    
    # Create processor and run analysis
    processor = CatchmentProcessor(namelist_path)
    hru_table = processor.process_catchment()
    
    print(f"Created HRU table with {len(hru_table)} HRUs")

if __name__ == "__main__":
    main()

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
            main_dir = Path(namelist_config['main_dir'])
            coupled = namelist_config.get('coupled', False)
            model_dirs = namelist_config['model_dirs']
            
            # CONSTRUCT MODEL_DIR THE SAME WAY AS CatchmentProcessor
            if coupled:
                self.model_dir = main_dir / model_dirs["coupled"] / f'catchment_{self.gauge_id}'
            else:
                self.model_dir = main_dir / model_dirs["uncoupled"] / f'catchment_{self.gauge_id}'
            
            # Optional parameters from namelist or defaults
            self.mode = namelist_config.get('mode', 'single')
            self.min_area_threshold = namelist_config.get('min_area_threshold', 0.01)
            self.debug = namelist_config.get('debug', False)
            
        else:
            # Use config dictionary (backward compatibility)
            # Required parameters
            self.model_dir = Path(config['model_dir'])
            self.gauge_id = config['gauge_id']
            
            # Optional parameters
            self.mode = config.get('mode', config.get('nconnect', 'single'))
            self.min_area_threshold = config.get('min_area_threshold', 0.01)
            self.debug = config.get('debug', False)
        
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
        
        return logging.getLogger(f'HRUConnectivityCalculator_Gauge_{self.gauge_id}')


    def get_path(self, filename: str) -> Path:
        """
        Get path to a file in the appropriate directory including model type
        
        Parameters
        ----------
        filename : str
            Name of the file
            
        Returns
        -------
        Path
            Full path to the file
        """
        # Save connections.rvh in model_type/data_obs directory
        if filename == 'connections.rvh':
            data_obs_dir = self.model_dir / self.model_type / 'data_obs'
            data_obs_dir.mkdir(parents=True, exist_ok=True)
            return data_obs_dir / filename
        else:
            # All other files go to topo_files
            return self.model_dir / 'topo_files' / filename

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
        Calculate flow accumulation for each HRU
        
        Returns
        -------
        np.ndarray
            Combined flow accumulation raster
        """
        self.logger.info("Calculating flow accumulation")
        
        # Initialize DataFrame for connectivity
        hru_df = self.hru_shapefile[['HRU_ID']].copy()
        if 'Area_km2' in self.hru_shapefile.columns:
            hru_df['Area_km2'] = self.hru_shapefile['Area_km2']
        else:
            self.logger.warning("Area_km2 not found in HRU shapefile, using default values")
            hru_df['Area_km2'] = 1.0
            
        hru_df['connectivity'] = [{} for _ in range(len(hru_df))]
        
        # Loop over every HRU and compute the flow accumulation
        flow_acc_tot = np.zeros_like(self.hru_raster)
        
        self.logger.debug(f"Processing flow accumulation for {len(hru_df)} HRUs")
        for unit_id in hru_df['HRU_ID']:
            mask_unit = self.hru_raster == unit_id
            flow_acc = self.dem_grid.accumulation(
                self.flow_dir, 
                mask=mask_unit, 
                routing='d8',
                nodata_out=np.float64(0)
            )
            flow_acc_np = flow_acc.view(np.ndarray)
            flow_acc_tot = np.maximum(flow_acc_tot, flow_acc_np)
        
        self.flow_acc = flow_acc_tot
        self.connectivity_df = hru_df
        
        self.logger.info("Flow accumulation calculation complete")
        return flow_acc_tot
    
    #---------------------------------------------------------------------------------

    def _sum_contributing_flow_acc(self) -> None:
        """
        Calculate connectivity between HRUs based on flow accumulation
        """
        self.logger.info("Calculating connectivity between HRUs")
        
        flow_dir = self.flow_dir.view(np.ndarray)
        
        # Loop over every cell in the flow accumulation grid
        height, width = self.flow_acc.shape
        
        processed_cells = 0
        total_cells = height * width
        
        for i in range(height):
            for j in range(width):
                # Get the unit id of the current cell
                if np.isnan(self.hru_raster[i, j]):
                    continue
                    
                unit_id = int(self.hru_raster[i, j])
                if unit_id == 0:
                    continue
                
                # Get the flow direction of the current cell
                flow_dir_cell = flow_dir[i, j]
                if flow_dir_cell <= 0:
                    continue
                
                # Get the unit id of the cell to which the current cell flows
                # Flow directions:
                # [N,  NE,  E, SE, S, SW, W, NW]
                # [64, 128, 1, 2,  4, 8, 16, 32]
                if flow_dir_cell == 1:
                    i_next, j_next = i, j + 1
                elif flow_dir_cell == 2:
                    i_next, j_next = i + 1, j + 1
                elif flow_dir_cell == 4:
                    i_next, j_next = i + 1, j
                elif flow_dir_cell == 8:
                    i_next, j_next = i + 1, j - 1
                elif flow_dir_cell == 16:
                    i_next, j_next = i, j - 1
                elif flow_dir_cell == 32:
                    i_next, j_next = i - 1, j - 1
                elif flow_dir_cell == 64:
                    i_next, j_next = i - 1, j
                elif flow_dir_cell == 128:
                    i_next, j_next = i - 1, j + 1
                else:
                    continue
                
                # Check if next cell is within grid boundaries
                if (i_next < 0 or i_next >= height or
                        j_next < 0 or j_next >= width):
                    continue
                
                # Check if next cell has a valid HRU ID
                if np.isnan(self.hru_raster[i_next, j_next]):
                    continue
                    
                unit_id_next = int(self.hru_raster[i_next, j_next])
                
                # Skip if the current cell flows to the same unit
                if unit_id_next == unit_id:
                    continue
                
                # Add connection from current HRU to next HRU
                idx = self.connectivity_df.index[self.connectivity_df['HRU_ID'] == unit_id].tolist()[0]
                connectivity = self.connectivity_df.at[idx, 'connectivity']
                
                if unit_id_next not in connectivity:
                    connectivity[unit_id_next] = 0
                    
                connectivity[unit_id_next] += float(self.flow_acc[i, j])
                self.connectivity_df.at[idx, 'connectivity'] = connectivity
                
                processed_cells += 1
                
                # Log progress periodically
                if processed_cells % 10000 == 0:
                    self.logger.debug(f"Processed {processed_cells}/{total_cells} cells ({processed_cells/total_cells:.1%})")
        
        self.logger.info(f"Connectivity calculation complete: processed {processed_cells} connections")

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
        Find the nearest HRU for HRUs without connectivity
        """
        self.logger.info("Finding nearest HRUs for disconnected HRUs")
        
        # First, identify HRUs with no connectivity
        hrus_without_connections = []
        for idx, row in self.connectivity_df.iterrows():
            if not row['connectivity']:  # Empty dictionary means no connections
                hrus_without_connections.append(row['HRU_ID'])
        
        if not hrus_without_connections:
            self.logger.debug("No disconnected HRUs found")
            return
            
        self.logger.debug(f"Found {len(hrus_without_connections)} HRUs without connections")
        
        # Create a mapping of HRU IDs to centroid coordinates
        hru_centroids = {}
        for hru_id in self.connectivity_df['HRU_ID'].unique():
            mask = (self.hru_raster == hru_id)
            if np.any(mask):  # Check if HRU exists in the raster
                rows, cols = np.where(mask)
                hru_centroids[hru_id] = (np.mean(rows), np.mean(cols))
        
        # For each HRU without connections, find the closest HRU
        for hru_id in hrus_without_connections:
            if hru_id not in hru_centroids:
                self.logger.warning(f"HRU {hru_id} not found in raster, skipping")
                continue
            
            src_centroid = hru_centroids[hru_id]
            min_distance = float('inf')
            closest_hru = None
            
            # Compute distance to all other HRUs
            for other_id, other_centroid in hru_centroids.items():
                if other_id == hru_id:
                    continue
                    
                # Compute Euclidean distance between centroids
                distance = ((src_centroid[0] - other_centroid[0])**2 + 
                             (src_centroid[1] - other_centroid[1])**2)**0.5
                
                if distance < min_distance:
                    min_distance = distance
                    closest_hru = other_id
            
            if closest_hru is not None:
                # Create a connection to the closest HRU
                self.logger.debug(f"Creating connection from HRU {hru_id} to closest HRU {closest_hru}")
                
                # Get index for the HRU without connections
                idx = self.connectivity_df.index[self.connectivity_df['HRU_ID'] == hru_id].tolist()[0]
                
                # Add connection with a reasonable weight
                # For single mode, just add one connection
                if self.mode == 'single':
                    self.connectivity_df.at[idx, 'connectivity'] = {closest_hru: 1.0}
                # For multiple mode, add a connection but allow for other connections to be added later
                else:
                    connect = self.connectivity_df.at[idx, 'connectivity']
                    connect[closest_hru] = 1.0
                    self.connectivity_df.at[idx, 'connectivity'] = connect
        
        self.logger.info(f"Connected {len(hrus_without_connections)} disconnected HRUs to nearest neighbors")

    #---------------------------------------------------------------------------------

    def plot_connectivity_map(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot HRU raster with arrows showing connectivity between units
        
        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Figure size (width, height), default (15, 10)
        """
        self.logger.info("Plotting connectivity map")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the HRU raster
        im = ax.imshow(self.hru_raster, cmap='tab20', interpolation='nearest')
        plt.colorbar(im, label='HRU ID')
        
        # Calculate centroids for each HRU
        centroids = {}
        for unit_id in np.unique(self.hru_raster[~np.isnan(self.hru_raster)]):
            unit_mask = self.hru_raster == unit_id
            y_coords, x_coords = np.where(unit_mask)
            if len(y_coords) > 0:  # Check if unit exists in raster
                centroids[unit_id] = (np.mean(x_coords), np.mean(y_coords))
        
        # Draw arrows for connectivity
        connection_count = 0
        for idx, row in self.connectivity_df.iterrows():
            unit_id = row['HRU_ID']
            if unit_id in centroids:
                start_point = centroids[unit_id]
                
                # Get connectivity dict and sort by value
                connections = row['connectivity']
                if isinstance(connections, dict) and connections:  # Check if connections exist
                    # Sort connections by strength
                    sorted_connections = sorted(connections.items(), key=lambda x: x[1], reverse=True)
                    
                    # Draw arrows for top 3 connections (or fewer if not available)
                    for target_id, strength in sorted_connections[:3]:
                        if target_id in centroids:
                            end_point = centroids[target_id]
                            
                            # Calculate arrow properties
                            dx = end_point[0] - start_point[0]
                            dy = end_point[1] - start_point[1]
                            
                            # Scale arrow length based on strength
                            scale_factor = 0.8  # Adjust this to change arrow length
                            dx *= scale_factor
                            dy *= scale_factor
                            
                            # Create arrow
                            arrow = Arrow(start_point[0], start_point[1], dx, dy,
                                        width=max(strength * 20, 1),  # Scale width by connectivity strength
                                        color='red',
                                        alpha=min(strength + 0.2, 0.8))  # Scale transparency by strength
                            ax.add_patch(arrow)
                            connection_count += 1
        
        # Add labels
        plt.title('HRU Connectivity Map')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # Add legend for arrow strength
        arrow_strengths = [0.2, 0.5, 1.0]
        legend_elements = [Arrow(0, 0, 1, 0, 
                               width=max(strength * 20, 1),
                               color='red', 
                               alpha=min(strength + 0.2, 0.8)) 
                          for strength in arrow_strengths]
        ax.legend(legend_elements, 
                 [f'Strength: {strength:.1f}' for strength in arrow_strengths],
                 loc='center left',
                 bbox_to_anchor=(1, 0.5))
        
        self.logger.info(f"Created connectivity map with {connection_count} connections")
        
        # Save the plot
        plot_dir = self.model_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / f"connectivity_map_{self.gauge_id}.png"
        plt.savefig(plot_path)
        self.logger.debug(f"Saved connectivity map to {plot_path}")
        
        plt.tight_layout()
        plt.show()

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
            f.write(":LateralConnections\n")
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

    def calculate_connectivity(self) -> pd.DataFrame:
        """
        Calculate HRU connectivity using flow accumulation method
        
        Returns
        -------
        pd.DataFrame
            DataFrame with HRU connectivity information
        """
        self.logger.info(f"Calculating HRU connectivity for gauge {self.gauge_id}")
        
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
        
        # Step 9: Plot connectivity map if in debug mode
        if self.debug:
            self.plot_connectivity_map()
        
        # Step 10: Write connectivity file
        self.write_connectivity_file()
        
        # Step 11: Save connectivity DataFrame
        connectivity_csv_path = self.get_path('HRU_connectivity.csv')
        self.connectivity_df.to_csv(connectivity_csv_path, index=False)
        self.logger.debug(f"Saved connectivity DataFrame to {connectivity_csv_path}")
        
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