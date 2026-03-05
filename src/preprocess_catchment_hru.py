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
        self.model_dir = self.main_dir / self.config.get('config_dir') / f"catchment_{self.gauge_id}"

        # Data directories (format paths with gauge_id) - CORRECTED PATHS
        self.shape_dir = self.main_dir / self.config['shape_dir'].format(gauge_id=self.gauge_id)
        self.raster_dir = self.main_dir / self.config['raster_dir'].format(gauge_id=self.gauge_id)
        self.landuse_dir = self.main_dir / self.config['landuse_dir'].format(gauge_id=self.gauge_id)
        
        # Glacier directory (optional)
        if 'glacier_dir' in self.config:
            self.glacier_dir = self.main_dir / self.config['glacier_dir']  # No gauge_id formatting needed
        else:
            self.glacier_dir = None
        
        # Landuse source: 'ESA', 'ICIMOD', or 'auto'
        self.landuse_source = self.config.get('landuse_source', 'auto').upper()
        
        # Processing parameters
        criteria = self.config.get('criteria', ['elevation', 'landuse'])
        self.criteria = criteria if isinstance(criteria, list) else [criteria]
        self.elevation_distance = self.config.get('elevation_distance', 100)
        self.slope_distance = self.config.get('slope_distance', 10)
        self.aspect_slope_threshold = self.config.get('aspect_slope_threshold', 15.0)
        self.debug = self.config.get('debug', False)
        
        # Basin ID for multi-subbasin support (default 1 for single-basin runs)
        self._basin_id = 1

        # Glacier HRU unit IDs (set during discretize_catchment, used in create_hru_table)
        self._large_glacier_hru_id = None
        self._small_glacier_hru_id = None

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
        logging.getLogger('fiona').setLevel(logging.WARNING)
        logging.getLogger('numba').setLevel(logging.WARNING)
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

    def _get_dem_meta(self):
        """Return (transform, crs, profile) of clipped DEM — cached after first call."""
        if not hasattr(self, '_cached_dem_meta'):
            with rasterio.open(self.get_path('clipped_dem.tif')) as src:
                self._cached_dem_meta = (src.transform, src.crs, src.profile.copy())
        return self._cached_dem_meta

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

    def _optimize_dem_clipping(self, dem, catchment_bounds):
        """
        Pre-clip DEM to rough bounding box before reprojection (much faster for large DEMs)
        
        Parameters
        ----------
        dem : xr.DataArray
            Full DEM to clip
        catchment_bounds : tuple
            Bounding box (minx, miny, maxx, maxy) in EPSG:4326
            
        Returns
        -------
        xr.DataArray
            Pre-clipped DEM (much smaller, faster to reproject)
        """
        # Add 0.5 degree buffer to catchment bounds (safety margin)
        minx, miny, maxx, maxy = catchment_bounds
        bbox_clip = (minx - 0.5, miny - 0.5, maxx + 0.5, maxy + 0.5)
        
        self.logger.debug(f"Pre-clipping DEM to bbox: {bbox_clip}")
        
        # Clip DEM to bounding box BEFORE reprojection (MUCH faster!)
        dem_bbox = dem.rio.clip_box(*bbox_clip)
        
        original_size = dem.shape
        clipped_size = dem_bbox.shape
        reduction = (1 - (clipped_size[0] * clipped_size[1]) / (original_size[0] * original_size[1])) * 100
        
        self.logger.debug(f"DEM size reduced: {original_size} -> {clipped_size} ({reduction:.1f}% reduction)")
        
        return dem_bbox

    #---------------------------------------------------------------------------------

    def clip_srtm_dem(self) -> xr.DataArray:
        """
        Clip the Upper Indus DEM to the catchment shapefile
        
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
        
        # ✅ SIMPLIFIED: Use raster_dir from namelist (already points to dem_Indus.tif)
        upper_indus_dem_path = self.main_dir / self.config['raster_dir']
        
        if not upper_indus_dem_path.exists():
            raise FileNotFoundError(
                f"Upper Indus DEM not found: {upper_indus_dem_path}\n"
                f"Expected path from namelist: {self.config['raster_dir']}\n"
                f"Please run download_SRTM_Indus.py first to download the DEM"
            )
        
        self.logger.debug(f"Opening Upper Indus DEM: {upper_indus_dem_path.name}")
        
        # Open Upper Indus DEM data (lazy loading - doesn't load full 2GB into memory!)
        dem = rxr.open_rasterio(upper_indus_dem_path, chunks='auto').squeeze()
        
        # Get the original CRS of the DEM (should be EPSG:4326 for SRTM)
        dem_crs = dem.rio.crs
        self.logger.debug(f"DEM CRS: {dem_crs}")
        
        # Get catchment bounds in EPSG:4326 for pre-clipping
        catchment_bounds = self.catchment_extent.to_crs('EPSG:4326').total_bounds
        lon_center = (catchment_bounds[0] + catchment_bounds[2]) / 2
        lat_center = (catchment_bounds[1] + catchment_bounds[3]) / 2
        
        # ✅ Pre-clip to bounding box BEFORE reprojection (HUGE speedup!)
        self.logger.debug("Pre-clipping DEM to catchment bounding box (this speeds up reprojection)")
        dem_bbox = self._optimize_dem_clipping(dem, catchment_bounds)
        
        # Calculate appropriate UTM zone for this catchment
        utm_zone = int((lon_center + 180) / 6) + 1
        utm_crs = f"EPSG:326{utm_zone:02d}" if lat_center >= 0 else f"EPSG:327{utm_zone:02d}"
        
        self.logger.debug(f"Using UTM CRS: {utm_crs} for catchment at ({lon_center:.2f}, {lat_center:.2f})")
        
        # Reproject the SMALL clipped DEM (much faster than reprojecting full 2GB!)
        self.logger.debug("Reprojecting clipped DEM to UTM...")
        dem_proj = dem_bbox.rio.reproject(utm_crs)
        
        # Reproject catchment to same UTM
        extent_proj = self.catchment_extent.to_crs(utm_crs)
        
        # Final clip to exact catchment boundary
        clip_bound = extent_proj.geometry
        self.logger.debug("Clipping to exact catchment boundary...")
        clipped_raster = dem_proj.rio.clip(clip_bound, from_disk=True)
        
        # Handle SRTM no-data values
        nodata_value = clipped_raster.rio.nodata
        if nodata_value is not None:
            clipped_raster = clipped_raster.where(clipped_raster != nodata_value, other=float('nan'))
        
        # Handle common SRTM void values
        clipped_raster = clipped_raster.where(clipped_raster != -32768, other=float('nan'))
        clipped_raster = clipped_raster.where(clipped_raster > -9999, other=float('nan'))
        
        # Save clipped raster
        self.logger.debug(f"Saving clipped DEM to: {out_raster_path}")
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
            # ✅ FIX: Downsample for plotting to avoid memory spike
            downsample_factor = max(1, max(slope.shape) // 2000)  # Max 2000 pixels in any dimension
            
            if downsample_factor > 1:
                self.logger.debug(f"Downsampling by factor {downsample_factor} for plotting (memory optimization)")
                dem_plot = self.dem_data[::downsample_factor, ::downsample_factor]
                slope_plot = slope[::downsample_factor, ::downsample_factor]
                aspect_plot = aspect[::downsample_factor, ::downsample_factor]
            else:
                dem_plot = self.dem_data
                slope_plot = slope
                aspect_plot = aspect
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.title('SRTM DEM')
            plt.imshow(dem_plot, cmap='terrain')
            plt.colorbar(label='Elevation (m)')
            
            plt.subplot(1, 3, 2)
            plt.title('Slope')
            plt.imshow(slope_plot, cmap='viridis')
            plt.colorbar(label='Slope (degrees)')
            
            plt.subplot(1, 3, 3)
            plt.title('Aspect')
            plt.imshow(aspect_plot, cmap='twilight')
            plt.colorbar(label='Aspect (degrees)')
            
            plt.tight_layout()
            
            # Save plot if debug mode
            plot_dir = Path(self.model_dir, 'plots')
            plot_path = plot_dir / f"dem_slope_aspect_{self.gauge_id}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            self.logger.debug(f"Saved plot to {plot_path}")
            plt.close()  # ✅ Close figure to free memory
            
            # Clear plot data from memory
            del dem_plot, slope_plot, aspect_plot
            import gc
            gc.collect()

        self.slope_data = slope
        self.aspect_data = aspect
        return slope, aspect

    #---------------------------------------------------------------------------------
    # ── Landuse reclassification schemes and auto-detection ──────────────────────
    #---------------------------------------------------------------------------------

    # ESA WorldCover V2 class values (10, 20, ..., 100)
    _ESA_SIGNATURE = {10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100}

    # ICIMOD HKH Land Cover class values (1–9)
    _ICIMOD_SIGNATURE = {1, 2, 3, 4, 5, 6, 7, 8, 9}

    # ── Reclassification to Raven landuse classes ────────────────────────────────
    #
    # Raven class  |  Meaning                          |  LAND_USE_CLASS  |  VEG_CLASS
    # -------------|-----------------------------------|------------------|----------
    #   1          |  Forest                           |  FOREST          |  FOREST
    #   2          |  Open vegetated (shrub/grass/wet) |  OPEN            |  GRAS
    #   3          |  Crop                             |  OPEN            |  CROP
    #   4          |  Built-up / urban                 |  BUILT           |  DEFAULT_V
    #   5          |  Rock / snow & ice                |  ROCK            |  DEFAULT_V
    #   6          |  Water (lake / river)             |  LAKE            |  DEFAULT_V
    #   9          |  Bare open (sparse veg / soil)    |  OPEN            |  DEFAULT_V
    #

    # ESA WorldCover V2 → Raven classes
    _RECLASSIFY_ESA = {
        1: {10, 95},       # Forest  (Tree cover, Mangroves)
        2: {20, 30, 90},   # Open    (Shrubland, Grassland, Herbaceous wetland)
        3: {40},           # Crop    (Cropland)
        4: {50},           # Built   (Built-up)
        5: {70},      # Rock    (Snow & Ice, Moss & lichen)
        6: {80},           # Water   (Permanent water bodies)
        9: {60, 100},           # Bare open (Bare / sparse vegetation)
    }

    # ICIMOD HKH Land Cover 2021 → Raven classes
    # Source codes:  1=Water, 2=Snow/glacier, 3=Forest, 4=Riverbed,
    #               5=Built-up, 6=Cropland, 7=Bare soil, 8=Bare rock, 9=Grassland
    _RECLASSIFY_ICIMOD = {
        1: {3},            # Forest  (ICIMOD: Forest)
        2: {9},            # Open    (ICIMOD: Grassland)
        3: {6},            # Crop    (ICIMOD: Cropland)
        4: {5},            # Built   (ICIMOD: Built-up area)
        5: {2, 8},         # Rock    (ICIMOD: Snow/glacier, Bare rock)
        6: {1, 4},         # Water   (ICIMOD: Water body, Riverbed)
        9: {7},            # Bare open (ICIMOD: Bare soil)
    }

    def _detect_landuse_source(self, raw_values: np.ndarray) -> str:
        """
        Auto-detect landuse classification scheme from unique pixel values.

        Parameters
        ----------
        raw_values : np.ndarray
            Raw landuse pixel values (may contain NaN / nodata)

        Returns
        -------
        str
            'ESA' or 'ICIMOD'
        """
        valid = raw_values[~np.isnan(raw_values)]
        if valid.size == 0:
            self.logger.warning("No valid landuse pixels – defaulting to ESA")
            return 'ESA'

        unique_vals = set(np.unique(valid).astype(int))
        # Exclude known nodata values that survived reprojection (e.g. ICIMOD nodata=15)
        unique_vals.discard(0)
        unique_vals.discard(15)

        overlap_esa    = len(unique_vals & self._ESA_SIGNATURE)
        overlap_icimod = len(unique_vals & self._ICIMOD_SIGNATURE)

        # ESA has values like 10, 20, …, 100 so any value > 9 is a strong ESA signal
        has_high_vals = any(v > 9 for v in unique_vals)

        if has_high_vals or overlap_esa > overlap_icimod:
            detected = 'ESA'
        else:
            detected = 'ICIMOD'

        self.logger.info(
            f"Auto-detected landuse source: {detected}  "
            f"(unique values: {sorted(unique_vals)}, "
            f"ESA overlap: {overlap_esa}, ICIMOD overlap: {overlap_icimod})"
        )
        return detected

    def _get_landuse_source(self, raw_values: np.ndarray) -> str:
        """
        Resolve which landuse scheme to use, optionally validating against
        auto-detection.

        Parameters
        ----------
        raw_values : np.ndarray
            Raw landuse raster values (float, NaN for nodata)

        Returns
        -------
        str
            'ESA' or 'ICIMOD'
        """
        declared = self.landuse_source  # already uppercased in __init__

        if declared == 'AUTO':
            return self._detect_landuse_source(raw_values)

        if declared not in ('ESA', 'ICIMOD'):
            raise ValueError(
                f"landuse_source '{declared}' not recognised. "
                f"Use 'ESA', 'ICIMOD', or 'auto'."
            )

        # Validate declared source against auto-detection (warn only, never override)
        detected = self._detect_landuse_source(raw_values)
        if detected != declared:
            self.logger.warning(
                f"⚠️  landuse_source in namelist is '{declared}' but auto-detection "
                f"says '{detected}'.  Trusting the namelist declaration.  "
                f"If reclassification produces all-NaN, check your landuse_dir path."
            )

        return declared

    def _apply_reclassification(self, raw: np.ndarray, source: str) -> np.ndarray:
        """
        Apply the correct reclassification mapping to raw landuse values.

        Parameters
        ----------
        raw : np.ndarray
            Float32 array with NaN for nodata
        source : str
            'ESA' or 'ICIMOD'

        Returns
        -------
        np.ndarray
            Reclassified array (same shape) with Raven classes 1–6
        """
        scheme = self._RECLASSIFY_ESA if source == 'ESA' else self._RECLASSIFY_ICIMOD

        result = np.full_like(raw, np.nan)  # start all-NaN

        for raven_class, src_values in scheme.items():
            mask = np.zeros(raw.shape, dtype=bool)
            for v in src_values:
                mask |= (raw == v)
            result[mask] = raven_class

        # Report unmapped pixels (excluding nodata)
        mapped = ~np.isnan(result)
        valid  = ~np.isnan(raw)
        n_unmapped = int(np.sum(valid & ~mapped))
        if n_unmapped > 0:
            unmapped_vals = np.unique(raw[valid & ~mapped]).astype(int)
            self.logger.warning(
                f"{n_unmapped:,} pixels have source values not in the {source} scheme "
                f"and were set to NaN.  Unmapped values: {unmapped_vals.tolist()}"
            )

        unique_out = np.unique(result[~np.isnan(result)]).astype(int)
        self.logger.info(
            f"Reclassified with {source} scheme → Raven classes present: {unique_out.tolist()}"
        )
        return result

    #---------------------------------------------------------------------------------

    def reclassify_landuse(self) -> xr.DataArray:
        """
        Reclassify the landuse raster to Raven landuse classes.
        
        Supports:
          - ESA WorldCover V2 (class values 10–100)
          - ICIMOD HKH Land Cover 2021 (class values 1–9)
        
        Which scheme is used is controlled by ``landuse_source`` in the namelist:
          - ``'ESA'``    → use ESA reclassification (validated against pixel values)
          - ``'ICIMOD'`` → use ICIMOD reclassification (validated against pixel values)
          - ``'auto'``   → auto-detect from the unique pixel values in the file
        
        Raven target classes (identical for both sources):
          1 = Forest,  2 = Open,  3 = Crop,  4 = Built,  5 = Rock/Snow,  6 = Water
        
        Returns
        -------
        xr.DataArray
            Reclassified landuse raster
        """
        self.logger.info("Reclassifying landuse")
        
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
        
        # Make sure DEM is loaded (we need it for CRS and grid matching)
        if self.dem_data is None:
            self.clip_srtm_dem()
        
        # Determine landuse source path
        landuse_source_path = self.main_dir / self.config['landuse_dir']
        
        if not landuse_source_path.exists():
            raise FileNotFoundError(f"Landuse file not found: {landuse_source_path}")
        
        # Check file size
        file_size_mb = landuse_source_path.stat().st_size / 1024 / 1024
        self.logger.info(f"Landuse file: {landuse_source_path.name} ({file_size_mb:.1f} MB)")
        
        # Use optimized windowed loading for large files (both ESA Indus and ICIMOD HKH are huge)
        use_optimized = (file_size_mb > 50) or ('indus' in landuse_source_path.name.lower()) \
                        or ('hkh' in landuse_source_path.name.lower())
        
        if use_optimized:
            self.logger.info(f"Using optimized windowed loading for large landuse file")
            
            # Get catchment bounds in EPSG:4326 with 0.5 degree buffer
            catchment_bounds = self.catchment_extent.to_crs('EPSG:4326').total_bounds
            minx, miny, maxx, maxy = catchment_bounds
            
            # Add 0.5 degree buffer (same as DEM clipping)
            bbox = (minx - 0.5, miny - 0.5, maxx + 0.5, maxy + 0.5)
            
            self.logger.debug(f"Catchment bounds: {catchment_bounds}")
            self.logger.debug(f"Pre-clipping landuse to bbox: {bbox}")
            
            try:
                # Use rasterio windowed reading for memory efficiency
                with rasterio.open(landuse_source_path) as src:
                    self.logger.debug(f"Landuse CRS: {src.crs}")
                    self.logger.debug(f"Landuse full shape: {src.shape} ({src.shape[0] * src.shape[1] / 1e6:.1f} million pixels)")
                    self.logger.debug(f"Landuse bounds: {src.bounds}")
                    
                    # Convert bbox to source CRS if needed
                    if src.crs and str(src.crs) != 'EPSG:4326':
                        from pyproj import Transformer
                        transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
                        # Transform corners
                        x1, y1 = transformer.transform(bbox[0], bbox[1])
                        x2, y2 = transformer.transform(bbox[2], bbox[3])
                        bbox_transformed = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                        self.logger.debug(f"Transformed bbox to {src.crs}: {bbox_transformed}")
                        window = rasterio.windows.from_bounds(*bbox_transformed, transform=src.transform)
                    else:
                        window = rasterio.windows.from_bounds(*bbox, transform=src.transform)
                    
                    # Round window to valid pixel coordinates
                    window = window.round_offsets().round_lengths()
                    
                    # Clamp window to valid bounds
                    window = rasterio.windows.Window(
                        col_off=max(0, window.col_off),
                        row_off=max(0, window.row_off),
                        width=min(window.width, src.width - max(0, window.col_off)),
                        height=min(window.height, src.height - max(0, window.row_off))
                    )
                    
                    self.logger.debug(f"Reading window: {window}")
                    self.logger.debug(f"Window size: {window.height} x {window.width} = {window.height * window.width / 1e6:.2f} million pixels")
                    
                    # Read only the windowed data
                    landuse_data = src.read(1, window=window)
                    landuse_transform = src.window_transform(window)
                    landuse_crs = src.crs
                    landuse_nodata = src.nodata
                    
                    self.logger.info(f"Loaded windowed landuse: {landuse_data.shape} "
                                f"(reduced from {src.shape})")
                
                # Calculate coordinates for the windowed data
                height, width = landuse_data.shape
                
                # Calculate coordinates from transform
                cols = np.arange(width)
                rows = np.arange(height)
                xs = landuse_transform[2] + cols * landuse_transform[0]
                ys = landuse_transform[5] + rows * landuse_transform[4]
                
                # Create xarray DataArray
                landuse = xr.DataArray(
                    data=landuse_data,
                    dims=['y', 'x'],
                    coords={'y': ys, 'x': xs}
                )
                landuse = landuse.rio.write_crs(landuse_crs)
                landuse = landuse.rio.write_transform(landuse_transform)
                if landuse_nodata is not None:
                    landuse = landuse.rio.write_nodata(landuse_nodata)

                self.logger.debug(f"Created xarray DataArray with shape {landuse.shape}")
                
                # Free the numpy array
                del landuse_data
                import gc
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Error during windowed reading: {e}")
                import traceback
                traceback.print_exc()
                raise
                
        else:
            # Small file, load normally
            self.logger.debug("Small landuse file, loading normally")
            landuse = rxr.open_rasterio(landuse_source_path).squeeze()

        # CONVERT TO FLOAT FIRST, THEN HANDLE NODATA
        # Read the raster's declared nodata value (e.g. ICIMOD=15, ESA=0)
        nodata_val = landuse.attrs.get('_FillValue', None)
        if nodata_val is None and hasattr(landuse, 'rio') and landuse.rio.nodata is not None:
            nodata_val = landuse.rio.nodata
        self.logger.debug(f"Landuse nodata value: {nodata_val}")

        landuse_values = landuse.values.astype(np.float32)
        landuse_values[landuse_values == 0] = np.nan
        if nodata_val is not None and nodata_val != 0:
            landuse_values[landuse_values == nodata_val] = np.nan
        
        # Update the DataArray with cleaned values
        landuse = xr.DataArray(
            data=landuse_values,
            dims=landuse.dims,
            coords=landuse.coords,
            attrs=landuse.attrs
        )
        
        # Free memory
        del landuse_values
        import gc
        gc.collect()
        
        # Ensure CRS is set
        if landuse.rio.crs is None:
            self.logger.warning("Landuse CRS was lost, re-setting")
            landuse = landuse.rio.write_crs(landuse_crs)
        
        # Get target CRS from DEM (should be UTM)
        target_crs = self.dem_data.rio.crs
        
        self.logger.debug(f"Reprojecting landuse from {landuse.rio.crs} to {target_crs}")
        landuse = landuse.rio.reproject(target_crs)
        
        # Clip raster to catchment extent
        extent = self.catchment_extent.to_crs(target_crs)
        clip_bound = extent.geometry
        
        self.logger.debug("Clipping landuse to exact catchment boundary")
        landuse = landuse.rio.clip(clip_bound, from_disk=True)
        
        # IMPORTANT: Make sure the landuse data matches the DEM's grid
        if self.dem_data is not None and landuse.shape != self.dem_data.shape:
            self.logger.info(f"Resampling landuse to match DEM grid: {landuse.shape} -> {self.dem_data.shape}")
            landuse = landuse.rio.reproject_match(self.dem_data)
        
        # ── Re-apply nodata masking after reprojection/resampling ──────────
        # Reprojection can reintroduce the original nodata value (e.g. ICIMOD nodata=15)
        if nodata_val is not None and nodata_val != 0:
            reintroduced = np.count_nonzero(landuse.values == nodata_val)
            if reintroduced > 0:
                self.logger.debug(f"Re-masking {reintroduced} pixels with nodata value {nodata_val} after reprojection")
                landuse.values[landuse.values == nodata_val] = np.nan

        # ── Detect / validate landuse source and apply reclassification ──────
        raw = landuse.values.copy()
        source = self._get_landuse_source(raw)
        self.logger.info(f"Using landuse reclassification scheme: {source}")
        lista = self._apply_reclassification(raw, source)
        del raw
        
        # Create a new DataArray with the reclassified values
        landuse_new = xr.DataArray(
            data=lista,
            dims=landuse.dims,
            coords=landuse.coords,
            attrs=landuse.attrs
        )
        
        # Ensure spatial reference is preserved
        landuse_new = landuse_new.rio.write_crs(target_crs)
        if hasattr(landuse, 'rio') and landuse.rio.transform() is not None:
            landuse_new = landuse_new.rio.write_transform(landuse.rio.transform())
        
        # SAVE RECLASSIFIED LANDUSE
        self.logger.debug(f"Saving reclassified landuse to {landuse_path}")
        landuse_new.rio.to_raster(landuse_path)
        
        # Log summary
        unique_values = np.unique(lista[~np.isnan(lista)])
        self.logger.info(f"Reclassified landuse saved. Unique classes: {unique_values}")
        
        # Plot if debug is enabled
        if self.debug:
            plot_raster(landuse_new, title=f"Reclassified Landuse ({source})", cmap="tab10")
        
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
            # Rebuild ID mappings from the clipped shapefile (needed for HRU creation)
            if glacier_path.exists() and not hasattr(self, 'glacier_int_to_str'):
                try:
                    gdf = gpd.read_file(glacier_path)
                    if len(gdf) > 0 and 'RGIId' in gdf.columns:
                        gdf['numeric_id_str'] = gdf['RGIId'].apply(lambda x: x.split('.')[-1])
                        gdf['numeric_id_int'] = gdf['numeric_id_str'].astype(int)
                        self.glacier_id_mapping = dict(zip(gdf['numeric_id_str'], gdf['RGIId']))
                        self.glacier_int_to_str = dict(zip(gdf['numeric_id_int'], gdf['numeric_id_str']))
                except Exception as e:
                    self.logger.warning(f"Could not rebuild glacier ID mappings: {e}")
                    self.glacier_id_mapping = {}
                    self.glacier_int_to_str = {}
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
            
            # ✅ FIX: Extract numeric IDs from RGI IDs (e.g., "RGI60-14.04481" → "04481")
            # This matches the format used in GloGEM NetCDF files
            gdf['numeric_id_str'] = gdf['RGIId'].apply(lambda x: x.split('.')[-1])
            
            # For rasterization, we need integer IDs (convert string to int)
            gdf['numeric_id_int'] = gdf['numeric_id_str'].astype(int)
            
            # Store mapping between numeric IDs (as strings with zero-padding) and RGI IDs
            # This is used for GloGEM validation
            self.glacier_id_mapping = dict(zip(gdf['numeric_id_str'], gdf['RGIId']))
            
            # Also store reverse mapping: integer ID → string ID (for raster lookups)
            self.glacier_int_to_str = dict(zip(gdf['numeric_id_int'], gdf['numeric_id_str']))
            
            # Create shapes for rasterization with integer IDs
            shapes = [(geom, numeric_id) for geom, numeric_id in zip(gdf.geometry, gdf['numeric_id_int'])]
            
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
            
            # ✅ NEW: Save mapping with size classification (area in km²)
            self.logger.info("Creating glacier ID mapping with size classification...")
            mapping_data = []
            
            for numeric_id, rgi_id in self.glacier_id_mapping.items():
                # Find this glacier in the glacier shapefile to get its area
                glacier_row = gdf[gdf['RGIId'] == rgi_id]
                
                if len(glacier_row) > 0:
                    area_km2 = glacier_row.iloc[0].get('Area', 0.0)
                    is_large = area_km2 >= 2.0
                else:
                    # Glacier not found in shapefile (shouldn't happen)
                    area_km2 = 0.0
                    is_large = False
                    self.logger.warning(f"Glacier {rgi_id} (ID {numeric_id}) not found in shapefile!")
                
                mapping_data.append({
                    'numeric_id': numeric_id,
                    'RGIId': rgi_id,
                    'area_km2': area_km2,
                    'is_large': is_large
                })
            
            mapping_df = pd.DataFrame(mapping_data)
            mapping_df.to_csv(self.get_path('glacier_id_mapping.csv'), index=False)
            
            # Log summary
            n_large = mapping_df['is_large'].sum()
            n_small = (~mapping_df['is_large']).sum()
            total_area = mapping_df['area_km2'].sum()
            large_area = mapping_df[mapping_df['is_large']]['area_km2'].sum()
            small_area = mapping_df[~mapping_df['is_large']]['area_km2'].sum()
            
            self.logger.info(f"✅ Saved glacier mapping with size classification:")
            self.logger.info(f"   Large glaciers (>=2 km²): {n_large} glaciers, {large_area:.2f} km²")
            self.logger.info(f"   Small glaciers (<2 km²): {n_small} glaciers, {small_area:.2f} km²")
            self.logger.info(f"   Total: {len(mapping_df)} glaciers, {total_area:.2f} km²")
            
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
        
        # Always load land use: needed for glacier class marking regardless of criteria
        if self.landuse_data is None:
            self.reclassify_landuse()
        
        if self.glacier_data is None and self.glacier_dir is not None:
            self.rasterize_glacier_shapefile()

        # When not coupled, burn glacier pixels as landuse class 7 so they
        # participate in criteria-based discretization (elevation bands, aspect, etc.)
        if self.glacier_data is not None and not self.coupled:
            glacier_mask = (self.glacier_data > 0) & np.isfinite(self.glacier_data)
            self.landuse_data.values[glacier_mask] = 7
            self.logger.info(f"Burned {int(np.count_nonzero(glacier_mask))} glacier pixels as landuse class 7 for criteria-based discretization")

        # Debug info after data loading but before processing
        if self.debug:
            self.logger.debug("Debugging catchment processing")

            topo_dir = Path(self.model_dir, 'topo_files')
            self.logger.debug(f"Topo directory: {topo_dir} (exists: {topo_dir.exists()})")

            if topo_dir.exists():
                files = list(topo_dir.glob('*'))
                self.logger.debug(f"Topo files: {[f.name for f in files]}")

                glacier_raster = topo_dir / 'glacier_raster.tif'
                if glacier_raster.exists():
                    with rasterio.open(glacier_raster) as src:
                        data = src.read(1)
                        unique_vals = np.unique(data[~np.isnan(data)])
                        self.logger.debug(f"Glacier raster unique values: {unique_vals}, shape: {data.shape}")

            self.logger.debug(
                f"Data array shapes — DEM: {self.dem_data.shape if self.dem_data is not None else None}, "
                f"Slope: {self.slope_data.shape if self.slope_data is not None else None}, "
                f"Aspect: {self.aspect_data.shape if self.aspect_data is not None else None}, "
                f"Landuse: {self.landuse_data.shape if self.landuse_data is not None else None}, "
                f"Glacier: {self.glacier_data.shape if self.glacier_data is not None else None}"
            )
            if hasattr(self, 'glacier_id_mapping'):
                self.logger.debug(f"Glacier ID mapping: {self.glacier_id_mapping}")
        
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
            # 1=Forest, 2=Open, 3=Crop, 4=Built, 5=Rock, 6=Water/Lake
            # 9=Bare Open (ICIMOD Bare Soil → class 9; must be included or entire
            #   high-altitude non-glacier areas are silently dropped)
            criteria_dict['landuse'] = [1, 2, 3, 4, 5, 6, 9]
            # When not coupled, glacier pixels (class 7) participate in normal discretization
            if self.glacier_data is not None and not self.coupled:
                criteria_dict['landuse'].append(7)
        
        if self.debug:
            self.logger.debug(f"Criteria dictionary: {criteria_dict}")
            for key, values in criteria_dict.items():
                self.logger.debug(f"  {key}: {len(values)} values")
        
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
        
        # Handle glaciers: create aggregated glacier HRUs only when coupled (GloGEM external melt)
        # When not coupled, glacier pixels participate in normal criteria-based discretization
        if self.glacier_data is not None and self.coupled:
            glacier_data_np = self.glacier_data
            # Get unique glacier IDs, filtering out NaN and infinite values
            valid_glacier_data = glacier_data_np[~np.isnan(glacier_data_np)]
            valid_glacier_data = valid_glacier_data[np.isfinite(valid_glacier_data)]
            unique_glaciers = np.unique(valid_glacier_data)
            
            self.logger.info(f"Creating 2 aggregated glacier HRUs from {len(unique_glaciers)} glaciers")
            self.logger.info(f"  - Large glaciers: >= 2 km²")
            self.logger.info(f"  - Small glaciers: < 2 km²")
            
            # Load glacier shapefile to get actual glacier areas
            glacier_shapefile_path = self.get_path('clipped_glacier.shp')
            
            if not glacier_shapefile_path.exists():
                self.logger.warning(f"Glacier shapefile not found: {glacier_shapefile_path}")
                self.logger.warning("Cannot classify glaciers by size, creating single aggregated HRU")
                
                # Fallback: Create single aggregated glacier HRU
                mask_all_glaciers = ~np.isnan(glacier_data_np)
                
                if np.count_nonzero(mask_all_glaciers) > 0:
                    map_unit_ids[mask_all_glaciers] = unit_id
                    
                    # Calculate centroid for all glaciers
                    rows, cols = np.where(mask_all_glaciers)
                    transform, _, _ = self._get_dem_meta()
                    lon, lat = xy(transform, np.mean(cols), np.mean(rows))
                    
                    # Get all RGI IDs
                    rgi_ids = []
                    for numeric_glacier_id_int in unique_glaciers:
                        # Skip invalid values
                        if not np.isfinite(numeric_glacier_id_int):
                            self.logger.warning(f"Skipping invalid glacier ID: {numeric_glacier_id_int}")
                            continue
                        
                        # Convert integer ID from raster to string ID with zero-padding
                        try:
                            numeric_id_str = self.glacier_int_to_str.get(int(numeric_glacier_id_int))
                            if numeric_id_str and numeric_id_str in self.glacier_id_mapping:
                                rgi_ids.append(self.glacier_id_mapping[numeric_id_str])
                            else:
                                # Fallback: create RGI ID from integer
                                rgi_ids.append(f"RGI60-11.{int(numeric_glacier_id_int):05d}")
                        except (ValueError, OverflowError) as e:
                            self.logger.warning(f"Cannot convert glacier ID {numeric_glacier_id_int}: {e}")
                            continue
                    
                    # Store as pipe-separated string
                    rgi_id_string = "|".join(rgi_ids)
                    
                    # Store glacier HRU attributes
                    columns['Unit ID'].append(unit_id)
                    columns['Glacier Class'].append(rgi_id_string)
                    columns['Latitude'].append(lat)
                    columns['Longitude'].append(lon)
                    
                    # Fill other fields with NaN (except Land Use Class)
                    for field in ['Elevation', 'Elevation Min', 'Elevation Max',
                                'Slope', 'Slope Min', 'Slope Max',
                                'Aspect Class']:
                        columns[field].append(np.nan)
                    # MASKED_GLACIER (8) when coupled (GloGEM external), GLACIER (7) when not
                    columns['Land Use Class'].append(8 if self.coupled else 7)

                    self.logger.info(f"Created aggregated glacier HRU {unit_id} with {len(rgi_ids)} glaciers")
                    unit_id += 1
                    
            else:
                # Load glacier shapefile to get glacier areas
                import geopandas as gpd
                glacier_gdf = gpd.read_file(glacier_shapefile_path)
                
                # Create mapping of RGI ID to area
                glacier_areas = {}
                for _, glacier_row in glacier_gdf.iterrows():
                    rgi_id = glacier_row.get('RGIId', None)
                    area_km2 = glacier_row.get('Area', 0.0)  # Area should be in km²
                    if rgi_id:
                        glacier_areas[rgi_id] = area_km2
                
                # Classify glaciers by size (2 km² threshold)
                large_glacier_ids = []
                small_glacier_ids = []
                large_glacier_mask = np.zeros(glacier_data_np.shape, dtype=bool)
                small_glacier_mask = np.zeros(glacier_data_np.shape, dtype=bool)
                
                for numeric_glacier_id_int in unique_glaciers:
                    # Skip invalid values
                    if not np.isfinite(numeric_glacier_id_int):
                        self.logger.warning(f"Skipping invalid glacier ID: {numeric_glacier_id_int}")
                        continue
                    
                    # Convert integer ID from raster to string ID with zero-padding
                    try:
                        numeric_id_str = self.glacier_int_to_str.get(int(numeric_glacier_id_int))
                        if numeric_id_str and numeric_id_str in self.glacier_id_mapping:
                            rgi_id = self.glacier_id_mapping[numeric_id_str]
                        else:
                            # Fallback: create RGI ID from integer
                            rgi_id = f"RGI60-11.{int(numeric_glacier_id_int):05d}"
                    except (ValueError, OverflowError) as e:
                        self.logger.warning(f"Cannot convert glacier ID {numeric_glacier_id_int} to integer: {e}")
                        continue
                    
                    # Get glacier area
                    area_km2 = glacier_areas.get(rgi_id, 0.0)
                    
                    # Get glacier mask
                    glacier_mask = (glacier_data_np == numeric_glacier_id_int)
                    
                    # Classify by size
                    if area_km2 >= 2.0:
                        large_glacier_ids.append(rgi_id)
                        large_glacier_mask |= glacier_mask
                    else:
                        small_glacier_ids.append(rgi_id)
                        small_glacier_mask |= glacier_mask
                
                self.logger.info(f"  - {len(large_glacier_ids)} large glaciers (>= 2 km²)")
                self.logger.info(f"  - {len(small_glacier_ids)} small glaciers (< 2 km²)")
                
                # Create LARGE glacier HRU (if any large glaciers exist)
                if np.count_nonzero(large_glacier_mask) > 0:
                    map_unit_ids[large_glacier_mask] = unit_id
                    
                    # Calculate centroid
                    rows, cols = np.where(large_glacier_mask)
                    transform, _, _ = self._get_dem_meta()
                    lon, lat = xy(transform, np.mean(cols), np.mean(rows))
                    
                    # Store all large glacier RGI IDs as pipe-separated string
                    rgi_id_string = "|".join(large_glacier_ids)
                    
                    # Store glacier HRU attributes
                    columns['Unit ID'].append(unit_id)
                    columns['Glacier Class'].append(rgi_id_string)
                    columns['Latitude'].append(lat)
                    columns['Longitude'].append(lon)
                    
                    # Fill other fields with NaN (except Land Use Class)
                    for field in ['Elevation', 'Elevation Min', 'Elevation Max',
                                'Slope', 'Slope Min', 'Slope Max',
                                'Aspect Class']:
                        columns[field].append(np.nan)
                    # MASKED_GLACIER (8) when coupled (GloGEM external), GLACIER (7) when not
                    columns['Land Use Class'].append(8 if self.coupled else 7)

                    self.logger.info(f"Created LARGE glacier HRU {unit_id} with {len(large_glacier_ids)} glaciers")
                    self._large_glacier_hru_id = unit_id
                    unit_id += 1

                # Create SMALL glacier HRU (if any small glaciers exist)
                if np.count_nonzero(small_glacier_mask) > 0:
                    map_unit_ids[small_glacier_mask] = unit_id
                    
                    # Calculate centroid
                    rows, cols = np.where(small_glacier_mask)
                    transform, _, _ = self._get_dem_meta()
                    lon, lat = xy(transform, np.mean(cols), np.mean(rows))
                    
                    # Store all small glacier RGI IDs as pipe-separated string
                    rgi_id_string = "|".join(small_glacier_ids)
                    
                    # Store glacier HRU attributes
                    columns['Unit ID'].append(unit_id)
                    columns['Glacier Class'].append(rgi_id_string)
                    columns['Latitude'].append(lat)
                    columns['Longitude'].append(lon)
                    
                    # Fill other fields with NaN (except Land Use Class)
                    for field in ['Elevation', 'Elevation Min', 'Elevation Max',
                                'Slope', 'Slope Min', 'Slope Max',
                                'Aspect Class']:
                        columns[field].append(np.nan)
                    # MASKED_GLACIER (8) when coupled (GloGEM external), GLACIER (7) when not
                    columns['Land Use Class'].append(8 if self.coupled else 7)

                    self.logger.info(f"Created SMALL glacier HRU {unit_id} with {len(small_glacier_ids)} glaciers")
                    self._small_glacier_hru_id = unit_id
                    unit_id += 1
        
            # Plot initial results if debug mode
            if self.debug:
                plot_map(map_unit_ids, title="Aggregated Glacier HRUs (Large + Small)")
        
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

        # Pre-extract numpy arrays once (avoids per-iteration DataArray overhead)
        dem_np = self.dem_data.values
        slope_np = self.slope_data.values if self.slope_data is not None else None
        landuse_np = self.landuse_data.values if self.landuse_data is not None else None
        aspect_np = {k: v.values for k, v in aspect_ranges.items()}

        # Process each combination to create HRUs
        for i, combination in enumerate(combinations):
            if self.debug and i % 50 == 0:
                self.logger.debug(f"Processing combination {i+1}/{len(combinations)}")

            # Create mask for current combination
            mask_unit = np.ones(dem_np.shape, dtype=bool)
            mask_unit[dem_np == -32768.0] = False

            # Apply each criterion to the mask
            for criterion_name, criterion in zip(combinations_keys, combination):
                if criterion_name == 'elevation':
                    mask_unit &= (dem_np >= criterion[0]) & (dem_np < criterion[1])
                elif criterion_name == 'slope':
                    mask_unit &= (slope_np >= criterion[0]) & (slope_np < criterion[1])
                elif criterion_name == 'aspect':
                    mask_unit &= aspect_np[criterion]
                elif criterion_name == 'landuse':
                    mask_unit &= (landuse_np == criterion)
            
            # Exclude glacier cells only when coupled (already assigned to aggregated glacier HRUs)
            if self.glacier_data is not None and self.coupled:
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
            transform, _, _ = self._get_dem_meta()
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
        transform, crs, profile = self._get_dem_meta()
        profile = profile.copy()
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
        ✅ OPTIMIZED: Vectorized processing for massive speedup
        
        Returns
        -------
        pd.DataFrame
            DataFrame with statistics for each HRU
        """
        self.logger.info("Calculating HRU statistics (vectorized)")
        
        # Make sure discretization has been done
        if self.map_unit_ids is None or self.hru_df is None:
            self.discretize_catchment()
        
        # Get cell area from DEM
        transform, _, _ = self._get_dem_meta()
        cell_area = abs(transform[0] * transform[4])  # Area in square meters
        
        # Get unique HRU IDs (excluding 0 and NaN)
        map_ids = self.map_unit_ids
        mask = (map_ids > 0) & ~np.isnan(map_ids)
        if not np.any(mask):
            self.logger.warning("No valid HRUs found in the map")
            return pd.DataFrame()
        
        unique_ids = np.unique(map_ids[mask])
        self.logger.debug(f"Processing statistics for {len(unique_ids)} HRUs (vectorized)")
        
        # ✅ VECTORIZED: Pre-extract all raster values (huge speedup!)
        dem_values = self.dem_data.values.copy()
        slope_values = self.slope_data.values.copy()
        aspect_values = self.aspect_data.values.copy()
        
        # Clean DEM values once (not per-HRU)
        valid_dem_mask = (dem_values != -32768) & ~np.isnan(dem_values)
        
        # Prepare transformer once
        transformer = Transformer.from_crs(self.dem_data.rio.crs, "EPSG:4326", always_xy=True)
        
        # ✅ VECTORIZED: Create mapping from hru_df
        hru_df_dict = self.hru_df.set_index('Unit ID').to_dict('index')
        
        # ✅ VECTORIZED: Process all HRUs at once using numpy operations
        results = []
        
        # Process HRUs in batches to avoid memory issues
        batch_size = 500
        num_batches = int(np.ceil(len(unique_ids) / batch_size))
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(unique_ids))
            batch_ids = unique_ids[start_idx:end_idx]
            
            if batch_idx % 5 == 0:
                self.logger.debug(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_ids)} HRUs)")
            
            for hru_id in batch_ids:
                # Create mask for this HRU
                hru_mask = (map_ids == hru_id)
                
                if not np.any(hru_mask):
                    continue
                
                # ✅ VECTORIZED: All extractions in one go
                area = np.sum(hru_mask) * cell_area
                
                # Elevation stats
                dem_masked = dem_values[hru_mask]
                valid_dem = dem_masked[valid_dem_mask[hru_mask]]
                
                if len(valid_dem) > 0:
                    mean_elevation = np.mean(valid_dem)
                else:
                    # Quick nearest neighbor fallback
                    rows, cols = np.where(hru_mask)
                    center_row, center_col = int(np.mean(rows)), int(np.mean(cols))
                    
                    # Simple search radius
                    radius = 5
                    search_area = dem_values[
                        max(0, center_row-radius):min(dem_values.shape[0], center_row+radius+1),
                        max(0, center_col-radius):min(dem_values.shape[1], center_col+radius+1)
                    ]
                    valid_search = search_area[valid_dem_mask[
                        max(0, center_row-radius):min(dem_values.shape[0], center_row+radius+1),
                        max(0, center_col-radius):min(dem_values.shape[1], center_col+radius+1)
                    ]]
                    
                    mean_elevation = np.mean(valid_search) if len(valid_search) > 0 else 1000.0
                
                # Slope stats
                slope_masked = slope_values[hru_mask]
                mean_slope = np.nanmean(slope_masked[~np.isnan(slope_masked)])
                
                # Aspect stats (circular mean)
                aspect_masked = aspect_values[hru_mask]
                aspect_masked = aspect_masked[~np.isnan(aspect_masked)]
                
                if len(aspect_masked) > 0:
                    aspect_rad = np.deg2rad(aspect_masked)
                    mean_aspect = np.rad2deg(np.arctan2(
                        np.mean(np.sin(aspect_rad)),
                        np.mean(np.cos(aspect_rad))
                    ))
                    mean_aspect = (mean_aspect + 360) % 360
                else:
                    mean_aspect = 0.0
                
                # Get values from hru_df
                hru_info = hru_df_dict.get(hru_id, {})
                landuse_class = hru_info.get('Land Use Class', 8)
                if pd.isna(landuse_class):
                    landuse_class = 8
                glacier_class = hru_info.get('Glacier Class', np.nan)
                elev_min = hru_info.get('Elevation Min', np.nan)
                elev_max = hru_info.get('Elevation Max', np.nan)
                
                # Calculate centroid
                rows, cols = np.where(hru_mask)
                x, y = transform * (np.mean(cols), np.mean(rows))
                lon, lat = transformer.transform(x, y)
                
                # Store results
                results.append({
                    'HRU_ID': int(hru_id),
                    'Area_m2': float(area),
                    'Area_km2': float(area / 1_000_000),
                    'Elev_Mean': float(mean_elevation),
                    'Elev_Min': elev_min,
                    'Elev_Max': elev_max,
                    'Slope_deg': float(mean_slope) if not np.isnan(mean_slope) else 0.0,
                    'Aspect_deg': float(mean_aspect),
                    'Landuse_Cl': landuse_class,
                    'Latitude': float(lat),
                    'Longitude': float(lon),
                    'Glacier_Cl': glacier_class
                })
        
        # Create DataFrame from results
        stats_df = pd.DataFrame(results)
        
        # Add aspect class if available
        if 'Aspect Class' in self.hru_df.columns:
            aspect_mapping = self.hru_df.set_index('Unit ID')['Aspect Class'].to_dict()
            stats_df['Aspect_Cl'] = stats_df['HRU_ID'].map(aspect_mapping)
        
        # Sort by HRU ID
        stats_df = stats_df.sort_values(by='HRU_ID')
        
        # Print summary
        self.logger.info(f"Total HRUs: {len(stats_df)}")
        self.logger.info(f"Total area: {stats_df['Area_km2'].sum():.2f} km²")
        self.logger.info(f"Elevation range: {stats_df['Elev_Mean'].min():.2f} - {stats_df['Elev_Mean'].max():.2f} m")
        
        # Visualize if debug
        if self.debug:
            self._plot_hru_statistics(stats_df)
        
        # Save
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
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 9, 'LAND_USE_CLASS'] = 'OPEN'
        
        # Assign soil classes 
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 5, 'SOIL_PROFILE'] = 'ROCK'
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 6, 'SOIL_PROFILE'] = 'LAKE'
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 7, 'SOIL_PROFILE'] = 'GLACIER'
        all_HRUs.loc[self.hru_stats['Landuse_Cl'] == 8, 'SOIL_PROFILE'] = 'MASKED_GLACIER'
        
        # Assign vegetation classes (class 9 = bare open keeps DEFAULT_V)
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
             'BASIN_ID': [self._basin_id] * len(all_HRUs),
             'LAND_USE_CLASS': all_HRUs['LAND_USE_CLASS'],
             'VEG_CLASS': all_HRUs['VEG_CLASS'],
             'SOIL_PROFILE': all_HRUs['SOIL_PROFILE'],
             'AQUIFER_PROFILE': all_HRUs['AQUIFER_PROFILE'],
             'TERRAIN_CLASS': all_HRUs['TERRAIN_CLASS'],
             'SLOPE': all_HRUs['SLOPE'],
             'ASPECT': all_HRUs['ASPECT']
            })
        
        # Add GLACIER_SIZE column to distinguish large vs small glacier HRUs
        HRU['GLACIER_SIZE'] = ''
        if self._large_glacier_hru_id is not None:
            HRU.loc[HRU[':ATTRIBUTES'] == self._large_glacier_hru_id, 'GLACIER_SIZE'] = 'LARGE'
        if self._small_glacier_hru_id is not None:
            HRU.loc[HRU[':ATTRIBUTES'] == self._small_glacier_hru_id, 'GLACIER_SIZE'] = 'SMALL'

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

    def merge_low_slope_hrus(self, hru_table: pd.DataFrame) -> pd.DataFrame:
        """
        Merge HRUs that share the same elevation band and landuse/veg/soil
        combination when their mean slope is below the aspect_slope_threshold.

        On gentle terrain, aspect orientation has negligible effect for all
        landuse types.  Instead of keeping up to 4 separate HRUs (one per
        aspect direction) we merge them into one, using area-weighted averages
        for all continuous properties.

        The slope threshold comes from the namelist key ``aspect_slope_threshold``
        (default 15°).

        Parameters
        ----------
        hru_table : pd.DataFrame
            The Raven HRU table produced by ``create_hru_table()``.

        Returns
        -------
        pd.DataFrame
            HRU table with low-slope HRUs merged (fewer rows).
        """
        threshold = self.aspect_slope_threshold
        if threshold <= 0 or self.hru_stats is None:
            self.logger.info("Aspect-slope merge disabled (threshold <= 0)")
            return hru_table

        # ── Build a working copy with elevation-band info ───────────────
        work = hru_table.copy()
        work['_elev_min'] = self.hru_stats['Elev_Min'].values
        work['_elev_max'] = self.hru_stats['Elev_Max'].values

        # All HRUs with mean slope below threshold are eligible — regardless of landuse
        is_low_slope = (work['SLOPE'] < threshold)
        eligible = is_low_slope

        n_eligible = int(eligible.sum())
        if n_eligible == 0:
            self.logger.info("No HRUs eligible for low-slope aspect merge")
            return hru_table

        self.logger.info(
            f"Low-slope merge: {n_eligible} HRUs eligible (slope < {threshold}°, all landuse classes)"
        )

        # ── Split into mergeable vs keep-as-is ──────────────────────────
        keep = work[~eligible].copy()
        to_merge = work[eligible].copy()

        # Group by elevation band + all Raven class columns
        group_cols = ['_elev_min', '_elev_max',
                      'LAND_USE_CLASS', 'VEG_CLASS', 'SOIL_PROFILE',
                      'AQUIFER_PROFILE', 'TERRAIN_CLASS', 'BASIN_ID']

        merged_rows = []
        for _key, grp in to_merge.groupby(group_cols, dropna=False):
            if len(grp) == 1:
                merged_rows.append(grp.iloc[0])
                continue

            area = grp['AREA']
            total_area = area.sum()
            w = area / total_area  # area weights

            # Area-weighted continuous properties
            merged_elev = float((grp['ELEVATION'] * w).sum())
            merged_lat  = float((grp['LATITUDE']  * w).sum())
            merged_lon  = float((grp['LONGITUDE'] * w).sum())
            merged_slope = float((grp['SLOPE']    * w).sum())

            # Area-weighted circular mean for aspect
            asp_rad = np.deg2rad(grp['ASPECT'].values)
            w_vals = w.values
            mean_sin = np.sum(np.sin(asp_rad) * w_vals)
            mean_cos = np.sum(np.cos(asp_rad) * w_vals)
            merged_aspect = float((np.rad2deg(np.arctan2(mean_sin, mean_cos)) + 360) % 360)

            # Take the lowest original ID for traceability
            merged_id = int(grp[':ATTRIBUTES'].min())

            row = grp.iloc[0].copy()
            row[':ATTRIBUTES'] = merged_id
            row['AREA']       = float(total_area)
            row['ELEVATION']  = merged_elev
            row['LATITUDE']   = merged_lat
            row['LONGITUDE']  = merged_lon
            row['SLOPE']      = merged_slope
            row['ASPECT']     = merged_aspect
            merged_rows.append(row)

        merged_df = pd.DataFrame(merged_rows)

        # ── Combine and re-number ───────────────────────────────────────
        result = pd.concat([keep, merged_df], ignore_index=True)
        # Drop helper columns
        result.drop(columns=['_elev_min', '_elev_max'], inplace=True)
        # Re-assign sequential IDs
        result = result.sort_values(':ATTRIBUTES').reset_index(drop=True)
        result[':ATTRIBUTES'] = range(1, len(result) + 1)

        n_removed = len(hru_table) - len(result)
        self.logger.info(
            f"Low-slope merge complete: {len(hru_table)} → {len(result)} HRUs "
            f"({n_removed} removed)"
        )

        # ── Overwrite saved files ───────────────────────────────────────
        result.to_csv(self.get_path('HRU_table.csv'), index=False)
        with open(self.get_path('HRU.txt'), 'w') as f:
            f.write(' '.join(result.columns) + '\n')
            for _, row in result.iterrows():
                f.write(' '.join(map(str, row)) + '\n')

        return result

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
        
        # ✅ CHECK: Skip if final output files already exist
        hru_shapefile = self.get_path('HRU.shp')
        hru_table = self.get_path('HRU_table.csv')
        hru_txt = self.get_path('HRU.txt')
        
        all_final_outputs_exist = (
            hru_shapefile.exists() and 
            hru_table.exists() and 
            hru_txt.exists()
        )
        
        if all_final_outputs_exist:
            self.logger.info("✅ All catchment processing outputs already exist")
            self.logger.info("⏭️  Skipping catchment processing")
            self.logger.info("💡 Delete files to force reprocessing:")
            self.logger.info(f"   rm {hru_shapefile}")
            self.logger.info(f"   rm {hru_table}")
            self.logger.info(f"   rm {hru_txt}")

            # Load and return existing HRU table
            self.logger.info(f"Loading existing HRU table from {hru_table}")
            existing_table = pd.read_csv(hru_table)

            # Also load the stats for consistency
            stats_file = self.get_path('HRU_statistics.csv')
            if stats_file.exists():
                self.hru_stats = pd.read_csv(stats_file)

            self.logger.info(f"✅ Loaded existing HRU table with {len(existing_table)} HRUs")
            return existing_table

        # ✅ PARTIAL COMPLETION: HRU.shp + statistics exist but table was not written
        # (e.g. process was killed after shapefile creation). Recreate table only.
        stats_file = self.get_path('HRU_statistics.csv')
        if (not all_final_outputs_exist and
                hru_shapefile.exists() and stats_file.exists() and
                not hru_table.exists()):
            self.logger.info("⚡ Partial completion detected: HRU.shp and statistics exist")
            self.logger.info("   Recreating HRU_table.csv/HRU.txt from cached statistics...")
            self.hru_stats = pd.read_csv(stats_file)
            hru_table_df = self.create_hru_table()
            self.logger.info(f"✅ HRU table recreated with {len(hru_table_df)} HRUs")
            return hru_table_df
        
        # ===== NORMAL PROCESSING (if files don't exist) =====
        
        try:
            # Step 1: Extract catchment shape (skip if pre-set by MultiSubbasinProcessor)
            if self.catchment_extent is None:
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
            hru_table_df = self.create_hru_table()
            
            # Step 10: Merge low-slope HRUs across aspects
            if self.aspect_slope_threshold > 0 and 'aspect' in self.criteria:
                hru_table_df = self.merge_low_slope_hrus(hru_table_df)
            
            self.logger.info(f"Catchment processing completed for gauge ID {self.gauge_id}")
            
            return hru_table_df
            
        except Exception as e:
            self.logger.error(f"Error processing catchment: {str(e)}")
            raise


