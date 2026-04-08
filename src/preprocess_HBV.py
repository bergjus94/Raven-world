"""
HBV Model Preprocessing Module for Raven Hydrological Modeling Framework

This module provides a comprehensive preprocessing class for setting up HBV model
runs in the Raven hydrological modeling framework. It handles the creation of
all necessary Raven input files (.rvh, .rvt, .rvp, .rvi, .rvc) with HBV-specific
configurations.

Author: Justine Berg
Date: August 2025
"""

#--------------------------------------------------------------------------------
#################################### packages ###################################
#--------------------------------------------------------------------------------

import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import date
import shutil
from typing import Dict, List, Optional, Union, Tuple
import os
import yaml
import preprocess_general
import numpy as np
import rasterio
import pyproj
from paths import get_paths, get_relative_data_obs, get_relative_topo

#--------------------------------------------------------------------------------
############################ HBV Preprocessor Class ############################
#--------------------------------------------------------------------------------

class HBVProcessor:
    """
    A comprehensive preprocessor for HBV model setup in Raven.
    
    This class handles all aspects of HBV model preprocessing including:
    - Creation of Raven input files (.rvh, .rvt, .rvp, .rvi, .rvc)
    - HRU management and elevation band grouping
    - Parameter handling for both template and initialized files
    - Forcing data configuration
    """
    
    def __init__(self, namelist_path: Union[str, Path]):
        """        
        Initialize the HBV preprocessor with namelist file.
        
        Args:
            namelist_path: Path to the YAML namelist file
        """

        import logging
        logging.getLogger('rasterio').setLevel(logging.WARNING)

        # Load namelist file
        namelist_path = Path(namelist_path)
        if not namelist_path.exists():
            raise FileNotFoundError(f"Namelist file not found: {namelist_path}")
        
        with open(namelist_path, 'r') as f:
            namelist = yaml.safe_load(f)
        
        # Store full namelist config (for multi-subbasin support)
        self.config = namelist

        # Store basic configuration
        self.gauge_id = str(namelist['gauge_id'])
        self.main_dir = Path(namelist['main_dir'])
        self.model_type = namelist['model_type']
        self.start_date = namelist['start_date']
        self.end_date = namelist['end_date']
        self.cali_end_date = namelist['cali_end_date']
        self.coupled = namelist.get('coupled', False)
        self._irrigation_variable = namelist.get('irrigation_variable', 'discharge').lower()
        self._pet_method = namelist.get('pet_method', None)
        self.author = namelist.get('author', 'Justine Berg')

        # ✅ NEW: Meteorological data source (ERA5 or HAR)
        self.meteo_source = namelist.get('meteo_source', 'ERA5').upper()
        if self.meteo_source not in ['ERA5', 'HAR']:
            raise ValueError(f"Invalid meteo_source: {self.meteo_source}. Must be 'ERA5' or 'HAR'")
        print(f"Meteorological data source: {self.meteo_source}")

        # ✅ ADD ONLY THIS - warm_up_date
        if 'warm_up_date' in namelist:
            self.warm_up_date = namelist['warm_up_date']
            print(f"Warm-up period configured: {self.warm_up_date} to {self.start_date}")
        else:
            self.warm_up_date = None
            print("No warm-up period configured")
        
        # ✅ LOAD PARAMETERS FROM NAMELIST
        params_path = namelist.get('params_dir', 'config/default_params.yaml')
        
        # Handle relative paths
        if not Path(params_path).is_absolute():
            params_path = Path(namelist_path).parent / params_path
        
        params_path = Path(params_path)
        if not params_path.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_path}")
        
        with open(params_path, 'r') as f:
            self.params = yaml.safe_load(f)
        
        print(f"✅ Loaded parameters from: {params_path}")
        
        self.gauge_lat = None
        self.gauge_lon = None
        self.station_elevation = None
        
        # Centralized path construction
        paths = get_paths(namelist)
        self.catchment_dir = paths['catchment_dir']
        self.hbv_dir = paths['model_dir']
        self.templates_dir = paths['template_dir']
        self.topo_files_dir = paths['topo_dir']
        self.shared_data_dir = paths['data_obs_dir']
        self.data_obs_dir = self.shared_data_dir  # backward-compat alias
        self._rel_data_obs = get_relative_data_obs(namelist)
        self._rel_topo = get_relative_topo(namelist)
        
        # Ensure directories exist
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.hbv_dir, self.templates_dir, self.shared_data_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def icemelt_mode(self) -> bool:
        """True when coupled + icemelt: glacier HRUs are ROCK with elevation discretization."""
        return self.coupled and self._irrigation_variable == 'icemelt'

    def _get_glacier_hru_ids_from_overlap(self, hru_df: pd.DataFrame) -> Dict[str, list]:
        """
        In icemelt mode, identify which ROCK HRUs originated from glaciers
        using glacier_hru_overlap.csv + glacier_id_mapping.csv.

        Returns dict with keys: 'all', 'large', 'small', 'no_glacier'.
        Values are lists of HRU attribute IDs (from ':ATTRIBUTES' column).
        """
        overlap_path = self.topo_files_dir / 'glacier_hru_overlap.csv'
        mapping_path = self.topo_files_dir / 'glacier_id_mapping.csv'

        if not overlap_path.exists() or not mapping_path.exists():
            print(f"WARNING: glacier_hru_overlap.csv or glacier_id_mapping.csv not found")
            all_hru_ids = hru_df[':ATTRIBUTES'].tolist()
            return {'all': [], 'large': [], 'small': [], 'no_glacier': all_hru_ids}

        overlap_df = pd.read_csv(overlap_path)
        mapping_df = pd.read_csv(mapping_path)

        # Build set of large glacier numeric IDs
        large_glacier_ints = set()
        for _, row in mapping_df.iterrows():
            if bool(row['is_large']):
                large_glacier_ints.add(int(row['numeric_id']))

        # For each HRU in overlap table, determine large/small
        glacier_hru_ids = set(overlap_df['HRU_ID'].unique())
        all_glacier_attrs = []
        large_glacier_attrs = []
        small_glacier_attrs = []

        for _, hru_row in hru_df.iterrows():
            hru_id = int(hru_row[':ATTRIBUTES'])
            if hru_id not in glacier_hru_ids:
                continue

            all_glacier_attrs.append(hru_id)

            # Check if any overlapping glacier is large
            hru_overlap = overlap_df[overlap_df['HRU_ID'] == hru_id]
            has_large = any(
                int(ov_row['glacier_numeric_id']) in large_glacier_ints
                for _, ov_row in hru_overlap.iterrows()
            )
            if has_large:
                large_glacier_attrs.append(hru_id)
            else:
                small_glacier_attrs.append(hru_id)

        all_hru_ids = set(hru_df[':ATTRIBUTES'].tolist())
        no_glacier_attrs = sorted(all_hru_ids - set(all_glacier_attrs))

        return {
            'all': all_glacier_attrs,
            'large': large_glacier_attrs,
            'small': small_glacier_attrs,
            'no_glacier': no_glacier_attrs,
        }

    def get_hrus_by_elevation_band(self) -> Dict[str, List[int]]:
        """
        Load HRU shapefile and create a dictionary of HRU IDs for each elevation band.
        
        Returns:
            Dictionary with elevation bands as keys and lists of HRU IDs as values.
            Returns empty dict if no elevation bands are available.
        """
        hru_path = self.topo_files_dir / 'HRU.shp'
        
        try:
            hru_gdf = gpd.read_file(hru_path)
        except Exception as e:
            print(f"Error reading HRU shapefile from {hru_path}: {e}")
            return {}
        
        # Check if elevation columns exist and have valid data
        if ('Elev_Min' not in hru_gdf.columns or 'Elev_Max' not in hru_gdf.columns or
            hru_gdf['Elev_Min'].isna().all() or hru_gdf['Elev_Max'].isna().all()):
            print(f"No elevation band data available for catchment {self.gauge_id}")
            return {}
        
        # Filter out landuse classes 7 and 8 if needed
        filtered_hru = hru_gdf[~hru_gdf['Landuse_Cl'].isin([7, 8])]
        
        # Filter out rows with NaN elevation values
        filtered_hru = filtered_hru.dropna(subset=['Elev_Min', 'Elev_Max'])
        
        if len(filtered_hru) == 0:
            print(f"No HRUs with valid elevation data for catchment {self.gauge_id}")
            return {}
        
        # Create elevation band labels
        filtered_hru['ElevationBand'] = filtered_hru.apply(
            lambda row: f"{int(row['Elev_Min'])}-{int(row['Elev_Max'])}m", axis=1)
        
        # Group by elevation band and collect HRU IDs
        hrus_by_band = {}
        for band in filtered_hru['ElevationBand'].unique():
            band_hrus = filtered_hru[filtered_hru['ElevationBand'] == band]['HRU_ID'].tolist()
            hrus_by_band[band] = band_hrus
        
        # Print summary
        print(f"Found {len(hrus_by_band)} elevation bands with a total of {len(filtered_hru)} HRUs")
        for band in sorted(hrus_by_band.keys(), key=lambda x: int(x.split('-')[0])):
            print(f"  {band}: {len(hrus_by_band[band])} HRUs")
        
        return hrus_by_band

    def _create_header(self, rvx_type: str) -> List[str]:
        """
        Create header info for .rvX files.
        
        Args:
            rvx_type: Type of Raven file (rvi, rvh, rvt, rvp, rvc)
            
        Returns:
            List of header lines
        """
        creation_date = date.today()
        header_line = "#########################################################################"
        file_type = f":FileType          {rvx_type} ASCII Raven 3.5"
        author_line = f":WrittenBy         {self.author}"
        creation_date_line = f":CreationDate      {creation_date}"
        description = [
            "#",
            f"# Emulation of HBV simulation of {self.gauge_id}",
            "#------------------------------------------------------------------------ \n"
        ]
        return [header_line, file_type, author_line, creation_date_line, *description]

    def _get_file_path(self, file_type: str, template: bool = False) -> Tuple[Path, str]:
        """
        Get file path and parameter naming convention.
        
        Args:
            file_type: Type of file (rvh, rvt, rvp, rvi, rvc)
            template: Whether to create template file
            
        Returns:
            Tuple of (file_path, param_or_name)
        """
        if template:
            param_or_name = "names"
            file_name = f"{self.gauge_id}_HBV.{file_type}.tpl"
            file_path = self.templates_dir / file_name
        else:
            param_or_name = "init"
            file_name = f"{self.gauge_id}_HBV.{file_type}"
            file_path = self.hbv_dir / file_name
        
        return file_path, param_or_name
    
    def get_gauge_location_from_dem(self) -> tuple:
        """
        Extract gauge location (lat, lon, elevation) from the lowest point in the DEM
        
        Returns
        -------
        tuple
            (latitude, longitude, elevation) of the lowest point in the DEM
        """
        # Path to clipped DEM
        dem_path = self.topo_files_dir / 'clipped_dem.tif'
        
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {dem_path}")
        
        try:
            with rasterio.open(dem_path) as dem:
                # Read DEM data
                dem_data = dem.read(1)  # Read first band
                
                print(f"DEM shape: {dem_data.shape}")
                print(f"DEM data type: {dem_data.dtype}")
                print(f"DEM nodata value: {dem.nodata}")
                print(f"DEM bounds: {dem.bounds}")
                print(f"DEM CRS: {dem.crs}")
                
                # Handle NoData values
                nodata = dem.nodata
                if nodata is not None:
                    valid_mask = (dem_data != nodata) & ~np.isnan(dem_data)
                    if not valid_mask.any():
                        raise ValueError("No valid data found in DEM - all pixels are NoData")
                    dem_data_valid = dem_data[valid_mask]
                else:
                    valid_mask = ~np.isnan(dem_data) & (dem_data > -9999)
                    if not valid_mask.any():
                        raise ValueError("No valid data found in DEM - all pixels are NaN or NoData")
                    dem_data_valid = dem_data[valid_mask]
                
                if len(dem_data_valid) == 0:
                    raise ValueError("No valid elevation data found after filtering")
                
                # Find minimum elevation from valid data
                min_elevation = np.min(dem_data_valid)
                print(f"Minimum elevation: {min_elevation:.1f} m")
                
                # Find the pixel coordinates of the minimum elevation
                min_indices = np.where((dem_data == min_elevation) & valid_mask)
                
                if len(min_indices[0]) == 0:
                    # Fallback: find closest value to minimum within tolerance
                    tolerance = 10.0  # 10m tolerance
                    close_indices = np.where(
                        (np.abs(dem_data - min_elevation) <= tolerance) & valid_mask
                    )
                    
                    if len(close_indices[0]) == 0:
                        raise ValueError(f"Could not find any valid pixels near minimum elevation {min_elevation}")
                    
                    min_indices = close_indices
                
                # Take the first pixel with minimum elevation
                row_idx = min_indices[0][0]
                col_idx = min_indices[1][0]
                
                # Convert pixel coordinates to geographic coordinates
                lon, lat = dem.xy(row_idx, col_idx)
                
                # Convert to WGS84 if needed
                if dem.crs != 'EPSG:4326':
                    transformer = pyproj.Transformer.from_crs(dem.crs, 'EPSG:4326', always_xy=True)
                    lon_wgs84, lat_wgs84 = transformer.transform(lon, lat)
                    lon, lat = lon_wgs84, lat_wgs84
                
                print(f"Gauge location: Lat={lat:.6f}, Lon={lon:.6f}, Elev={min_elevation:.1f}m")
                
                return lat, lon, min_elevation
                
        except Exception as e:
            print(f"Error reading DEM file: {e}")
            raise

    def create_rvh_file(self, template: bool = False):
        """
        Write Raven .rvh file for HBV model.
        
        Args:
            template: Whether to create a template file
        """
        file_path, param_or_name = self._get_file_path('rvh', template)
        # DEBUG: Print what param_or_name actually is
        print(f"DEBUG: template={template}, param_or_name={param_or_name}")
        print(f"DEBUG: Available keys in self.params['HBV']['{param_or_name}']: {list(self.params['HBV'][param_or_name].keys())[:5]}...")
            
        # Read HRU table from CSV file
        hru_table_path = self.topo_files_dir / 'HRU_table.csv'
        
        try:
            HRU = pd.read_csv(hru_table_path)
            print(f"Successfully read HRU table from {hru_table_path}")
        except Exception as e:
            print(f"Error reading HRU table from {hru_table_path}: {e}")
            return

        # Drop GLACIER_SIZE column (internal use only, not a Raven attribute)
        HRU = HRU.drop(columns=['GLACIER_SIZE'], errors='ignore')

        # Convert HRU table to formatted string
        x = HRU.to_string(header=False, index=False, index_names=False).split('\n')
        vals = [',\t'.join(ele.split()) for ele in x]

        # Create HRUs section
        hru_list = [
            ":HRUs",
            "  :Attributes, ID,  AREA, ELEVATION, LATITUDE, LONGITUDE, BASIN_ID,LAND_USE_CLASS, VEG_CLASS, SOIL_PROFILE, AQUIFER_PROFILE, TERRAIN_CLASS, SLOPE, ASPECT",
            "  :Units     , none,   km2,         m,      deg,       deg,     none,          none,      none,         none,            none,          none,   deg,    deg",
            *vals,
            ":EndHRUs"
        ]
        
        # Add lateral connections
        lateral_connections = [
            "",
            "#:LateralConnections",
            f":RedirectToFile  {self._rel_topo}/connections.rvh",
            "#:EndLateralConnections",
            ""
        ]
        
        # Create HRU groups
        hru_groups = self._create_hru_groups(HRU)

        # -------------------------------------------------------------------
        # SubBasins / SubBasinProperties – multi or single basin
        # -------------------------------------------------------------------
        subbasins_config = self.config.get('subbasins', None)

        if subbasins_config:
            # Multi-subbasin: read routing computed by MultiSubbasinProcessor
            routing_path = self.topo_files_dir / 'subbasin_routing.yaml'
            if not routing_path.exists():
                raise FileNotFoundError(
                    f"subbasin_routing.yaml not found at {routing_path}. "
                    "Run MultiSubbasinProcessor.process_all_subbasins() first."
                )
            with open(routing_path) as _f:
                routing = yaml.safe_load(_f)  # {id: downstream_id}

            sb_rows = []
            for sb in subbasins_config:
                sb_id = sb['id']
                downstream = routing[sb_id]
                reach = sb.get('reach_length', '_AUTO')
                profile = sb.get('profile', 'NONE')
                gauged = sb.get('gauged', 0)
                sb_rows.append(
                    f"  {sb_id:10d}, {sb['name']:20s}, {downstream:12d}, "
                    f"{profile:>10s}, {str(reach):>12s}, {gauged:>6d}"
                )

            subbasins = [
                ":SubBasins",
                "  :Attributes,          NAME, DOWNSTREAM_ID,   PROFILE, REACH_LENGTH, GAUGED",
                "  :Units     ,          none,          none,      none,           km,   none",
                *sb_rows,
                ":EndSubBasins",
            ]

            sp_rows = []
            for sb in subbasins_config:
                sp_rows.append(
                    f"  {sb['id']:10d},  {self.params['HBV'][param_or_name]['X11']},"
                    f"  {self.params['HBV'][param_or_name]['HBV_Time_To_Peak']},"
                )

            subbasin_properties = [
                ":SubBasinProperties",
                "#                       HBV_T_CONC_MAX_BAS, DERIVED FROM HBV_T_CONC_MAX_BAS,",
                "#                            MAXBAS,                 MAXBAS/2,",
                "   :Parameters,           TIME_CONC,             TIME_TO_PEAK,",
                "   :Units,                        d,                        d,",
                *sp_rows,
                ":EndSubBasinProperties",
            ]

        else:
            # Single-basin: existing behaviour unchanged
            subbasins = [
                ":SubBasins",
                "  :Attributes,          NAME, DOWNSTREAM_ID,PROFILE,REACH_LENGTH,       GAUGED",
                "  :Units     ,          none,          none,   none,          km,         none",
                f"            1,        {self.gauge_id},            -1,   NONE,       _AUTO,     1",
                ":EndSubBasins",
            ]

            subbasin_properties = [
                ":SubBasinProperties",
                "#                       HBV_T_CONC_MAX_BAS, DERIVED FROM HBV_T_CONC_MAX_BAS,",
                "#                            MAXBAS,                 MAXBAS/2,",
                "   :Parameters,           TIME_CONC,             TIME_TO_PEAK,",
                "   :Units,                        d,                        d,",
                f"              1,          {self.params['HBV'][param_or_name]['X11']},                  {self.params['HBV'][param_or_name]['HBV_Time_To_Peak']},",
                ":EndSubBasinProperties",
            ]

        # Write the file
        with open(file_path, 'w') as ff:
            ff.writelines(f"{line}\n" for line in self._create_header("rvh"))
            ff.writelines(f"{line}\n" for line in subbasins)
            ff.writelines(f"{line}\n" for line in subbasin_properties)
            ff.writelines(f"{line}\n" for line in hru_list)
            ff.writelines(f"{line}\n" for line in lateral_connections)
            ff.write("\n# HRU Groups\n")
            ff.writelines(f"{line}\n" for line in hru_groups)
                
        print(f"Successfully wrote HBV RVH file to {file_path}")

    def _create_hru_groups(self, hru_df: pd.DataFrame) -> List[str]:
        """
        Create HRU groups including filtered AllHRUs, elevation bands, and glacier groups.
        ✅ UPDATED: Handles aggregated glacier HRUs (2 HRUs: 1 large, 1 small)
        """
        
        def format_hru_list(hru_ids: List[int], max_per_line: int = 50) -> List[str]:
            """
            Format HRU IDs into multiple lines if needed.
            
            Args:
                hru_ids: List of HRU IDs
                max_per_line: Maximum number of HRU IDs per line
                
            Returns:
                List of formatted lines
            """
            if not hru_ids:
                return ["  # No HRUs available"]
            
            lines = []
            for i in range(0, len(hru_ids), max_per_line):
                chunk = hru_ids[i:i + max_per_line]
                lines.append("  " + " ".join(map(str, chunk)))
            
            return lines
        
        hru_groups = []
        
        # Different exclusion rules for different groups
        allhrus_excluded_landuse = ['GLACIER', 'ROCK', 'MASKED_GLACIER']
        elevation_excluded_landuse = ['GLACIER', 'MASKED_GLACIER']  # Only exclude glaciers for elevation bands
        
        # Get all HRU IDs excluding the specified land use classes for AllHRUs
        filtered_hrus = hru_df[~hru_df['LAND_USE_CLASS'].isin(allhrus_excluded_landuse)]
        filtered_hru_ids = filtered_hrus[':ATTRIBUTES'].tolist()
        
        print(f"Total HRUs: {len(hru_df)}")
        print(f"Excluded HRUs from AllHRUs group (land use {allhrus_excluded_landuse}): {len(hru_df) - len(filtered_hrus)}")
        print(f"AllHRUs group will contain: {len(filtered_hru_ids)} HRUs")
        
        # Add AllHRUs group with filtered HRUs (split across multiple lines)
        hru_groups.append(":HRUGroup AllHRUs")
        hru_groups.extend(format_hru_list(filtered_hru_ids))
        hru_groups.extend([":EndHRUGroup", ""])

        # ✅ Glacier HRU groups
        if self.icemelt_mode:
            # Icemelt mode: glacier HRUs are ROCK — identify via overlap CSV
            glacier_groups = self._get_glacier_hru_ids_from_overlap(hru_df)
            all_glacier_hrus = glacier_groups['all']
            large_glacier_hrus = glacier_groups['large']
            small_glacier_hrus = glacier_groups['small']
            no_glacier_hrus = glacier_groups['no_glacier']
            print(f"Icemelt mode: identified {len(all_glacier_hrus)} glacier-origin ROCK HRUs from overlap CSV")
        else:
            # Standard mode: glacier HRUs have GLACIER/MASKED_GLACIER land use
            all_glacier_hrus = hru_df[hru_df['LAND_USE_CLASS'].isin(['GLACIER', 'MASKED_GLACIER'])][':ATTRIBUTES'].tolist()

            # Use GLACIER_SIZE column if available (multi-subbasin); fallback to creation-order heuristic
            if 'GLACIER_SIZE' in hru_df.columns:
                large_glacier_hrus = hru_df[hru_df['GLACIER_SIZE'] == 'LARGE'][':ATTRIBUTES'].tolist()
                small_glacier_hrus = hru_df[hru_df['GLACIER_SIZE'] == 'SMALL'][':ATTRIBUTES'].tolist()
            else:
                large_glacier_hrus = [all_glacier_hrus[0]] if len(all_glacier_hrus) > 0 else []
                small_glacier_hrus = [all_glacier_hrus[1]] if len(all_glacier_hrus) > 1 else []

            no_glacier_hrus = hru_df[~hru_df['LAND_USE_CLASS'].isin(['GLACIER', 'MASKED_GLACIER'])][':ATTRIBUTES'].tolist()

        hru_groups.append(":HRUGroup ALL_GLACIER")
        hru_groups.extend(format_hru_list(all_glacier_hrus))
        hru_groups.extend([":EndHRUGroup", ""])
        print(f"ALL_GLACIER group: {len(all_glacier_hrus)} HRUs")

        hru_groups.append(":HRUGroup SMALL_GLACIER")
        hru_groups.extend(format_hru_list(small_glacier_hrus))
        hru_groups.extend([":EndHRUGroup", ""])
        print(f"SMALL_GLACIER group: {len(small_glacier_hrus)} HRUs")

        hru_groups.append(":HRUGroup LARGE_GLACIER")
        hru_groups.extend(format_hru_list(large_glacier_hrus))
        hru_groups.extend([":EndHRUGroup", ""])
        print(f"LARGE_GLACIER group: {len(large_glacier_hrus)} HRUs")

        hru_groups.append(":HRUGroup NO_GLACIER")
        hru_groups.extend(format_hru_list(no_glacier_hrus))
        hru_groups.extend([":EndHRUGroup", ""])
        print(f"NO_GLACIER group: {len(no_glacier_hrus)} HRUs")

        # Add elevation band groups with DIFFERENT filtering (only exclude glaciers)
        hrus_by_band = self.get_hrus_by_elevation_band()
        if hrus_by_band:
            for band, hru_ids in hrus_by_band.items():
                if hru_ids:
                    # Apply DIFFERENT filtering for elevation bands (only exclude glaciers)
                    band_hru_df = hru_df[hru_df[':ATTRIBUTES'].isin(hru_ids)]
                    filtered_band_hrus = band_hru_df[~band_hru_df['LAND_USE_CLASS'].isin(elevation_excluded_landuse)]
                    filtered_band_hru_ids = filtered_band_hrus[':ATTRIBUTES'].tolist()
                    
                    print(f"Elevation band {band}: {len(hru_ids)} total HRUs, {len(filtered_band_hru_ids)} after excluding {elevation_excluded_landuse}")
                    
                    # ✅ Split elevation bands across multiple lines too
                    hru_groups.append(f":HRUGroup {band}")
                    hru_groups.extend(format_hru_list(filtered_band_hru_ids))
                    hru_groups.extend([":EndHRUGroup", ""])
        else:
            print(f"No elevation bands available for catchment {self.gauge_id}")

        return hru_groups

    def _get_subbasin_mean_elevation(self, sb_gauge_id: str) -> Optional[float]:
        """Return area-weighted mean elevation for a subbasin from the merged HRU table."""
        try:
            hru_table_path = self.topo_files_dir / 'HRU_table.csv'
            if not hru_table_path.exists():
                return None
            hru_df = pd.read_csv(hru_table_path)
            # Find the subbasin id corresponding to this gauge_id
            sb_match = [sb for sb in self.config.get('subbasins', [])
                        if str(sb['gauge_id']) == str(sb_gauge_id)]
            if not sb_match:
                return None
            sb_id = sb_match[0]['id']
            sb_rows = hru_df[hru_df['BASIN_ID'] == sb_id]
            if sb_rows.empty or 'ELEVATION' not in sb_rows.columns:
                return None
            if 'AREA' in sb_rows.columns:
                total_area = sb_rows['AREA'].sum()
                if total_area > 0:
                    return round(float((sb_rows['ELEVATION'] * sb_rows['AREA']).sum() / total_area), 1)
            return round(float(sb_rows['ELEVATION'].mean()), 1)
        except Exception:
            return None

    def _get_subbasin_centroid_location(self, sb_gauge_id: str):
        """Return (lat, lon, elevation) for a subbasin from its catchment shapefile centroid."""
        shape_template = self.config.get('shape_dir', '')
        shp_path = Path(self.config['main_dir']) / shape_template.format(gauge_id=sb_gauge_id)
        if not shp_path.exists():
            return None, None, None
        try:
            gdf = gpd.read_file(shp_path).to_crs('EPSG:4326')
            centroid = gdf.geometry.unary_union.centroid
            elev = self._get_subbasin_mean_elevation(sb_gauge_id)
            return round(centroid.y, 4), round(centroid.x, 4), elev
        except Exception as e:
            print(f"  ⚠️ Could not compute centroid for subbasin {sb_gauge_id}: {e}")
            return None, None, None

    def _write_rvt_header(self, ff, param_or_name: str):
        """Write RVT file header + forcing blocks (shared by single and multi-subbasin)."""
        ff.writelines(f"{line}\n" for line in self._create_header("rvt"))
        ff.write(f"# Meteorological data source: {self.meteo_source}\n")
        if self.meteo_source == 'HAR':
            ff.write("# Using HAR v2 (High Asia Refined Analysis) 10km data\n")
        else:
            ff.write("# Using ERA5-Land reanalysis data\n")
        ff.write("#\n")
        ff.write("# meteorological forcings\n")
        forcing_data = self._create_forcing_block(param_or_name, self.coupled)
        for f in forcing_data.values():
            for t in f:
                ff.write(f"{t}\n")

    def create_rvt_file(self, template: bool = False):
        """
        Write Raven .rvt file for HBV model.
        ✅ UPDATED: Supports multi-subbasin mode (one :Gauge block per subbasin).
        """
        if self.config.get('subbasins'):
            self._create_rvt_file_multi_subbasin(template)
        else:
            self._create_rvt_file_single(template)

    def _create_rvt_file_single(self, template: bool = False):
        """Write .rvt for a single-basin namelist (original behaviour)."""
        file_path, param_or_name = self._get_file_path('rvt', template)

        print(f"Extracting gauge location from DEM for gauge {self.gauge_id}...")
        try:
            gauge_lat, gauge_lon, station_elevation = self.get_gauge_location_from_dem()
        except Exception as e:
            print(f"Error: Could not extract gauge location from DEM: {e}")
            return

        gauge_info = self._create_gauge_info(self.gauge_lat, self.gauge_lon,
                                              self.station_elevation, param_or_name)
        with open(file_path, 'w') as ff:
            self._write_rvt_header(ff, param_or_name)
            ff.writelines(gauge_info)
            ff.write(f":RedirectToFile {self._rel_data_obs}/Q_daily.rvt\n")

        print(f"✅ Successfully wrote HBV RVT file to {file_path}")
        print(f"   Meteo source: {self.meteo_source}")

    def _create_rvt_file_multi_subbasin(self, template: bool = False):
        """Write .rvt for a multi-subbasin namelist.

        One :Gauge block per subbasin (all subbasins, gauged and ungauged).
        A :RedirectToFile line is added only for subbasins with gauged=1.
        """
        file_path, param_or_name = self._get_file_path('rvt', template)
        subbasins_config = self.config['subbasins']

        with open(file_path, 'w') as ff:
            self._write_rvt_header(ff, param_or_name)

            for sb in subbasins_config:
                sb_id    = sb['id']
                sb_gid   = str(sb['gauge_id'])
                gauged   = sb.get('gauged', 0)

                # Per-subbasin monthly T/PET directory
                sb_data_dir = self.shared_data_dir / f'subbasin_{sb_id}'

                # Gauge location from subbasin shapefile centroid
                lat, lon, elev = self._get_subbasin_centroid_location(sb_gid)
                print(f"  Subbasin {sb_id} ({sb_gid}): lat={lat}, lon={lon}, elev={elev}")

                gauge_info = self._create_gauge_info(
                    lat, lon, elev, param_or_name,
                    gauge_id=sb_gid,
                    subbasin_data_dir=sb_data_dir
                )
                ff.writelines(gauge_info)

                if gauged:
                    ff.write(f":RedirectToFile {self._rel_data_obs}/Q_daily_{sb_gid}.rvt\n\n")
                    print(f"  → Streamflow redirect: {self._rel_data_obs}/Q_daily_{sb_gid}.rvt")

        print(f"✅ Successfully wrote multi-subbasin HBV RVT file to {file_path}")
        print(f"   Meteo source: {self.meteo_source}")

    def _create_gauge_info(self, gauge_lat: float, gauge_lon: float,
                            station_elevation: float, param_or_name: str,
                            gauge_id: str = None,
                            subbasin_data_dir: Path = None) -> List[str]:
        """Create gauge information section for RVT file.

        Parameters
        ----------
        gauge_id : str, optional
            Gauge ID to write in the :Gauge header.  Defaults to self.gauge_id.
        subbasin_data_dir : Path, optional
            Per-subbasin data_obs directory to search for monthly T/PET CSV files
            before falling back to the shared directories.
        """
        gid = gauge_id if gauge_id is not None else self.gauge_id
        gauge_info = [
            f":Gauge {gid}\n",
            f"  :Latitude    {gauge_lat}\n",
            f"  :Longitude {gauge_lon}\n",
            f"  :Elevation  {station_elevation}\n\n"
        ]

        # Add monthly data if available
        monthly_data = self._get_monthly_data(subbasin_data_dir=subbasin_data_dir)
        if monthly_data:
            gauge_info.extend(monthly_data)

        gauge_info.append(f":EndGauge\n\n")

        return gauge_info

    def _get_monthly_data(self, subbasin_data_dir: Path = None) -> List[str]:
        """
        Read and format monthly temperature and PET data if available.
        ✅ UPDATED: Chooses correct files based on meteo_source (ERA5 or HAR)
        ✅ UPDATED: Also checks shared data_obs directory
        ✅ UPDATED: Checks per-subbasin directory first when subbasin_data_dir is given
        """
        # Skip monthly PET data when using PET_OUDIN (computes PET from temperature internally)
        if self._pet_method and self._pet_method.upper() == 'OUDIN':
            return []

        # ✅ Choose file names based on meteo_source
        if self.meteo_source == 'HAR':
            temp_filename = 'har_monthly_temperature_averages.csv'
            pet_filename = 'har_monthly_pet_averages.csv'
        else:  # ERA5 (default)
            temp_filename = 'monthly_temperature_averages.csv'
            pet_filename = 'monthly_pet_averages.csv'

        # ✅ Check per-subbasin directory first (multi-subbasin mode)
        if subbasin_data_dir is not None:
            sb_temp = Path(subbasin_data_dir) / temp_filename
            sb_pet  = Path(subbasin_data_dir) / pet_filename
            if sb_temp.exists() and sb_pet.exists():
                monthly_temp_file = sb_temp
                monthly_pet_file  = sb_pet
                print(f"📂 Using per-subbasin monthly data: {subbasin_data_dir}")
            else:
                # Fall through to normal search
                monthly_temp_file = self.data_obs_dir / temp_filename
                monthly_pet_file  = self.data_obs_dir / pet_filename
        else:
            # ✅ Check model-specific directory first
            monthly_temp_file = self.data_obs_dir / temp_filename
            monthly_pet_file = self.data_obs_dir / pet_filename

        # ✅ If not found, check shared directory
        if not monthly_temp_file.exists() or not monthly_pet_file.exists():
            shared_data_dir = self.catchment_dir / 'data_obs'

            if not monthly_temp_file.exists():
                shared_temp = shared_data_dir / temp_filename
                if shared_temp.exists():
                    monthly_temp_file = shared_temp
                    print(f"📂 Found temperature file in shared directory: {shared_temp}")

            if not monthly_pet_file.exists():
                shared_pet = shared_data_dir / pet_filename
                if shared_pet.exists():
                    monthly_pet_file = shared_pet
                    print(f"📂 Found PET file in shared directory: {shared_pet}")
        
        # ✅ Final check
        if not monthly_temp_file.exists() or not monthly_pet_file.exists():
            print(f"⚠️ Monthly data files not found for {self.meteo_source}:")
            print(f"   Temperature: {temp_filename}")
            print(f"   - Checked: {self.data_obs_dir / temp_filename} (exists: {(self.data_obs_dir / temp_filename).exists()})")
            print(f"   - Checked: {self.catchment_dir / 'data_obs' / temp_filename} (exists: {(self.catchment_dir / 'data_obs' / temp_filename).exists()})")
            print(f"   PET: {pet_filename}")
            print(f"   - Checked: {self.data_obs_dir / pet_filename} (exists: {(self.data_obs_dir / pet_filename).exists()})")
            print(f"   - Checked: {self.catchment_dir / 'data_obs' / pet_filename} (exists: {(self.catchment_dir / 'data_obs' / pet_filename).exists()})")
            print(f"   💡 Run HARAnalyzer first to generate these files!")
            return []
        
        try:
            temp_df = pd.read_csv(monthly_temp_file)
            pet_df = pd.read_csv(monthly_pet_file)
            
            # ✅ Validate data
            if 'Temperature' not in temp_df.columns:
                print(f"⚠️ 'Temperature' column not found in {monthly_temp_file}")
                print(f"   Available columns: {list(temp_df.columns)}")
                return []
            
            if 'PET_avg_mm_per_day' not in pet_df.columns:
                print(f"⚠️ 'PET_avg_mm_per_day' column not found in {monthly_pet_file}")
                print(f"   Available columns: {list(pet_df.columns)}")
                return []
            
            temp_values = temp_df['Temperature'].values
            temp_str = ", ".join([f"{val:.1f}" for val in temp_values])
            
            pet_values = pet_df['PET_avg_mm_per_day'].values
            pet_str = ", ".join([f"{val:.3f}" for val in pet_values])
            
            print(f"✅ Loaded monthly data from {self.meteo_source} files:")
            print(f"   Temperature file: {monthly_temp_file.name}")
            print(f"   PET file: {monthly_pet_file.name}")
            
            return [
                "#                       Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec \n",
                f"  :MonthlyAveEvaporation, {pet_str} \n",
                f"  :MonthlyAveTemperature, {temp_str} \n"
            ]
        except Exception as e:
            print(f"❌ Error reading monthly data: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _create_forcing_block(self, param_or_name: str, coupled: bool) -> Dict[str, List[str]]:
        """
        Create forcing data configuration for RVT file.
        ✅ UPDATED: Supports both ERA5-Land and HAR meteorological data sources
        
        Parameters
        ----------
        param_or_name : str
            Parameter naming convention ('names' for template, 'init' for initialized)
        coupled : bool
            Whether the model is coupled
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary of forcing data blocks
        """
        
        # ✅ Choose file names and GridWeights based on meteo_source
        if self.meteo_source == 'HAR':
            # HAR file names and configuration
            precip_file = 'har_precip.nc'
            temp_mean_file = 'har_temp_mean.nc'
            temp_max_file = 'har_temp_max.nc'
            temp_min_file = 'har_temp_min.nc'
            grid_weights_file = f'{self._rel_topo}/GridWeights_HAR.txt'

            # HAR variable names
            precip_var = 'prcp'
            temp_var = 't2_mean'
            temp_max_var = 't2_max'
            temp_min_var = 't2_min'

            # HAR dimension names (curvilinear grid)
            dim_names = "west_east south_north time"

            print(f"📊 Using HAR meteorological data")

        else:  # ERA5 (default)
            # ERA5-Land file names and configuration
            precip_file = 'era5_land_precip.nc'
            temp_mean_file = 'era5_land_temp_mean.nc'
            temp_max_file = 'era5_land_temp_max.nc'
            temp_min_file = 'era5_land_temp_min.nc'
            grid_weights_file = f'{self._rel_topo}/GridWeights.txt'

            # ERA5-Land variable names
            precip_var = 'tp'
            temp_var = 't2m'
            temp_max_var = 't2m'
            temp_min_var = 't2m'

            # ERA5-Land dimension names (normalized to lon/lat by preprocess_meteo)
            dim_names = "lon lat time"

            print(f"📊 Using ERA5-Land meteorological data")

        # Precipitation weights/dims default to the meteo_source values
        precip_weights_file = grid_weights_file
        precip_dim_names = dim_names

        # Override all forcing files when running CORDEX climate projections.
        # CORDEX downscaled files use identical variable names and grid to
        # ERA5-Land, so only the filenames need to change.
        if self.config.get('future', False) and self.config.get('cordex_models'):
            models    = self.config['cordex_models']
            scenarios = self.config.get('cordex_scenarios', ['rcp45'])
            model_id  = models[0].replace('/', '_').replace(' ', '_')
            scenario  = scenarios[0]   # first scenario used for .rvt generation

            precip_file    = f'cordex_{model_id}_{scenario}_precip.nc'
            temp_mean_file = f'cordex_{model_id}_{scenario}_temp_mean.nc'
            temp_max_file  = f'cordex_{model_id}_{scenario}_temp_max.nc'
            temp_min_file  = f'cordex_{model_id}_{scenario}_temp_min.nc'
            # CORDEX downscaled files are on the ERA5-Land grid
            precip_var    = 'tp'
            temp_var      = 't2m'
            temp_max_var  = 't2m'
            temp_min_var  = 't2m'
            grid_weights_file = f'{self._rel_topo}/GridWeights.txt'
            precip_weights_file = grid_weights_file
            dim_names       = "lon lat time"
            precip_dim_names = dim_names
            print(f"📊 Using CORDEX downscaled forcing: {model_id} / {scenario}")

        # Override all forcing files when running CMIP6 climate projections.
        # CMIP6 downscaled output uses the same ERA5-Land variable names/grid.
        if self.config.get('future', False) and self.config.get('cmip6_models'):
            models    = self.config['cmip6_models']
            scenarios = self.config.get('cmip6_scenarios', ['ssp126'])
            model_id  = models[0].replace('/', '_').replace(' ', '_')
            future_scens = [s for s in scenarios if s != 'historical']
            scenario  = future_scens[0] if future_scens else scenarios[0]

            precip_file    = f'cmip6_{model_id}_{scenario}_precip.nc'
            temp_mean_file = f'cmip6_{model_id}_{scenario}_temp_mean.nc'
            temp_max_file  = f'cmip6_{model_id}_{scenario}_temp_max.nc'
            temp_min_file  = f'cmip6_{model_id}_{scenario}_temp_min.nc'
            precip_var    = 'tp'
            temp_var      = 't2m'
            temp_max_var  = 't2m'
            temp_min_var  = 't2m'
            grid_weights_file = f'{self._rel_topo}/GridWeights.txt'
            precip_weights_file = grid_weights_file
            dim_names       = "lon lat time"
            precip_dim_names = dim_names
            print(f"Using CMIP6 forcing: {model_id} / {scenario}")

        # ✅ Override precipitation source if TPHiPr is requested
        if self.config.get('precip_source', '').upper() == 'TPHIPR':
            precip_file = 'tphipr_precip.nc'
            precip_var = 'prcp'
            precip_weights_file = f'{self._rel_topo}/GridWeights_TPHiPr.txt'
            precip_dim_names = 'longitude latitude time'
            print(f"📊 Overriding precipitation source: TPHiPr")

        # Get optional parameters with default values
        rain_corr = self.params['HBV'][param_or_name].get('X20', 1.0)
        snow_corr = self.params['HBV'][param_or_name].get('X21', 1.0)
        use_precip_corr = self.config.get('precip_correction', False)
        comment = "" if use_precip_corr else "#"

        forcing_data = {}

        # Rainfall forcing block (uses precip-specific weights and dim names)
        # TPHiPr has no elevation variable (high-res product, no lapse correction needed)
        precip_elev_line = ("    :ElevationVarNameNC   elevation"
                            if self.config.get('precip_source', '').upper() != 'TPHIPR'
                            else "    # No ElevationVarNameNC for TPHiPr (high-res gridded product)")
        forcing_data['Rainfall'] = [
            f":GriddedForcing           Rainfall",
            f"    :ForcingType          RAINFALL",
            f"    :FileNameNC           {self._rel_data_obs}/{precip_file}",
            f"    :VarNameNC            {precip_var}",
            f"    :DimNamesNC           {precip_dim_names}",
            precip_elev_line,
            f"    {comment}:RainCorrection       {rain_corr}",
            f"    {comment}:SnowCorrection       {snow_corr}",
            f"    :RedirectToFile       {precip_weights_file}",
            ":EndGriddedForcing",
            ''
        ]

        # Average Temperature forcing block
        forcing_data['Average Temperature'] = [
            f":GriddedForcing           Average Temperature",
            f"    :ForcingType          TEMP_AVE",
            f"    :FileNameNC           {self._rel_data_obs}/{temp_mean_file}",
            f"    :VarNameNC            {temp_var}",
            f"    :DimNamesNC           {dim_names}",
            "    :ElevationVarNameNC   elevation",
            f"    :RedirectToFile       {grid_weights_file}",
            ":EndGriddedForcing",
            ''
        ]

        # Maximum Temperature forcing block
        forcing_data['Maximum Temperature'] = [
            f":GriddedForcing           Maximum Temperature",
            f"    :ForcingType          TEMP_MAX",
            f"    :FileNameNC           {self._rel_data_obs}/{temp_max_file}",
            f"    :VarNameNC            {temp_max_var}",
            f"    :DimNamesNC           {dim_names}",
            "    :ElevationVarNameNC   elevation",
            f"    :RedirectToFile       {grid_weights_file}",
            ":EndGriddedForcing",
            ''
        ]

        # Minimum Temperature forcing block
        forcing_data['Minimum Temperature'] = [
            f":GriddedForcing           Minimum Temperature",
            f"    :ForcingType          TEMP_MIN",
            f"    :FileNameNC           {self._rel_data_obs}/{temp_min_file}",
            f"    :VarNameNC            {temp_min_var}",
            f"    :DimNamesNC           {dim_names}",
            "    :ElevationVarNameNC   elevation",
            f"    :RedirectToFile       {grid_weights_file}",
            ":EndGriddedForcing",
            ''
        ]
        
        # ✅ Irrigation forcing block (only for coupled mode)
        if self.coupled:
            forcing_data['Irrigation'] = [
                ":GriddedForcing           Irrigation",
                "    :ForcingType          IRRIGATION",
                f"    :FileNameNC           {self._rel_data_obs}/irrigation.nc",
                "    :VarNameNC            data",
                "    :DimNamesNC           x y time     # must be in the order of (x,y,t)",
                "    :ElevationVarNameNC   elevation",
                f"    :RedirectToFile       {self._rel_topo}/GridWeights_Irrigation.txt",
                ":EndGriddedForcing",
                ''
            ]

        return forcing_data


    def _compute_avg_annual_runoff(self) -> Optional[float]:
        """
        Compute average annual runoff from the main gauge streamflow CSV.

        Returns
        -------
        float or None
            Average annual runoff in mm/yr, or None if the data are unavailable.
        """
        try:
            stream_dir = self.config.get('stream_dir', '')
            stream_path = self.main_dir / stream_dir.format(gauge_id=self.gauge_id)

            if not stream_path.exists():
                print(f"   ⚠️ Streamflow file not found for AVG_ANNUAL_RUNOFF: {stream_path}")
                return None

            df = pd.read_csv(stream_path)
            df['date'] = pd.to_datetime(df['date'])

            # Use simulation period only (exclude warm-up)
            start = pd.to_datetime(self.start_date)
            end   = pd.to_datetime(self.end_date)
            df = df[df['date'].between(start, end)]

            # Drop Raven fill value and any negatives
            df = df[df['Q_obs'] > 0]

            if len(df) == 0:
                print("   ⚠️ No valid streamflow data for AVG_ANNUAL_RUNOFF")
                return None

            mean_q_m3s = df['Q_obs'].mean()

            # Total catchment area from merged HRU table
            hru_table_path = self.topo_files_dir / 'HRU_table.csv'
            if not hru_table_path.exists():
                print(f"   ⚠️ HRU_table.csv not found for AVG_ANNUAL_RUNOFF: {hru_table_path}")
                return None

            hru_table = pd.read_csv(hru_table_path)
            total_area_km2 = hru_table['AREA'].sum()

            if total_area_km2 <= 0:
                return None

            # runoff [mm/yr] = Q [m³/s] × 31557600 [s/yr] / area [m²] × 1000 [mm/m]
            seconds_per_year = 365.25 * 24 * 3600
            runoff_mm_yr = mean_q_m3s * seconds_per_year / (total_area_km2 * 1e3)

            print(f"   📊 AVG_ANNUAL_RUNOFF: {runoff_mm_yr:.1f} mm/yr "
                  f"(mean Q={mean_q_m3s:.2f} m³/s, area={total_area_km2:.1f} km²)")
            return round(runoff_mm_yr, 1)

        except Exception as e:
            print(f"   ⚠️ Could not compute AVG_ANNUAL_RUNOFF: {e}")
            return None


    def create_rvp_file(self, template: bool = False):
        """
        Write Raven .rvp file for HBV model.
        
        Args:
            template: Whether to create template file
        """
        file_path, param_or_name = self._get_file_path('rvp', template)
        
        # Define land use and vegetation classes
        land_use_classes = [
            ":LandUseClasses",
            "   :Attributes, IMPERM, FOREST_COV",
            "   :Units, frac, frac",
            "   FOREST,    0.05, 1.0",
            "   OPEN,      0.0,  0.0",
            "   GLACIER,   0.0,  0.0",
            "   LAKE,      0.0,  0.0",
            "   ROCK,      0.15, 0.0",
            "   BUILT,     0.5,  0.0",
            "   DEFAULT_L  0.0,  0.0",
            "   MASKED_GLACIER, 1.0, 0.0",
            ":EndLandUseClasses"
        ]
        
        vegetation_classes = [
            ":VegetationClasses",
            "   :Attributes, MAX_HT, MAX_LAI, MAX_LEAF_COND",
            "   :Units, m, none, mm_per_s",
            "   DEFAULT_V, 0.0, 0.0, 0.0",
            "   FOREST,  20,  4.5, 3.5",
            "   GRAS,    0.6, 1.5, 2.5",
            "   CROP,    2.0, 3.0, 4.0",
            ":EndVegetationClasses"
        ]

        # Create HBV-specific parameter structure
        rvp_sections = self._create_rvp_sections(param_or_name, 
                                                land_use_classes, vegetation_classes)
        
        # Write the file
        with open(file_path, 'w') as ff:
            ff.writelines(f"{line}\n" for line in self._create_header("rvp"))
            for section, lines in rvp_sections.items():
                ff.write(f"{section}\n")
                ff.writelines(line + '\n' for line in lines)
                ff.write('\n')

    def _build_global_params(self, param_or_name: str) -> List[str]:
        """Build the :GlobalParameter lines, including AVG_ANNUAL_RUNOFF."""
        lines = [
            f":GlobalParameter RAINSNOW_TEMP       {self.params['HBV'][param_or_name]['X01']}",
            ":GlobalParameter RAINSNOW_DELTA      1.0 #constant",
            f":GlobalParameter PRECIP_LAPSE       2.0 # RECIP_LAPSE",
            f":GlobalParameter ADIABATIC_LAPSE    6.0 # ADIABATIC_LAPSE",
            f":GlobalParameter SNOW_SWI  {self.params['HBV'][param_or_name]['X04']} #SNOW_SWI",
        ]
        # DeltaH glacier method requires SNOW_TO_ICE_DATE (day of year for annual snow-to-ice conversion)
        if not self.coupled and self.config.get('glacier_method') == 'DeltaH':
            lines.append(":GlobalParameter SNOW_TO_ICE_DATE    274   # Oct 1 (day of year)")
        avg_runoff = self._compute_avg_annual_runoff()
        if avg_runoff is not None:
            lines.append(
                f":GlobalParameter AVG_ANNUAL_RUNOFF  {avg_runoff}  "
                f"# avg annual runoff [mm/yr] from gauge {self.gauge_id}"
            )
        else:
            lines.append(
                "# :GlobalParameter AVG_ANNUAL_RUNOFF  ???  "
                "# Could not compute — fill manually [mm/yr]"
            )
        return lines

    def _create_rvp_sections(self, param_or_name: str, 
                        land_use_classes: List[str], vegetation_classes: List[str]) -> Dict[str, List[str]]:
        """Create all sections for RVP file."""
        
        # Get optional glacier parameter with default
        glac_storage_coeff = self.params['HBV'][param_or_name].get('X19', 0.05)
        
        return {
            "#Soil Classes": [
                ":SoilClasses",
                "   :Attributes,",
                "   :Units,",
                "       TOPSOIL,      1.0,    0.0,       0",
                "       SLOW_RES,     1.0,    0.0,       0",
                "       FAST_RES,     1.0,    0.0,       0",
                ":EndSoilClasses"
            ],
            "#Soil Profiles": [
                "#     name,#horizons,{soiltype,thickness}x{#horizons}",
                "# ",
                ":SoilProfiles",
                "    GLACIER, 0",
                "    LAKE, 0",
                "    ROCK, 0",
                "    MASKED_GLACIER, 0",
                f"   DEFAULT_P,      3,    TOPSOIL,            {self.params['HBV'][param_or_name]['X17']},   FAST_RES,    100.0, SLOW_RES,    100.0",
                ":EndSoilProfiles"
            ],
            "#Vegetation Classes": vegetation_classes,
            "#Vegetation Parameters": [
                ":VegetationParameterList",
                "   :Parameters,  MAX_CAPACITY, MAX_SNOW_CAPACITY,  TFRAIN,  TFSNOW,",
                "   :Units,                 mm,                mm,    frac,    frac,",
                "       [DEFAULT],             10000,             10000,    0.88,    0.88,",
                ":EndVegetationParameterList"
            ],
            "#Land Use Classes": land_use_classes,
            "#Global Parameters": self._build_global_params(param_or_name),
            "#Land Use Parameters": [
                ":LandUseParameterList",
                "  :Parameters,   MELT_FACTOR, MIN_MELT_FACTOR,   HBV_MELT_FOR_CORR, REFREEZE_FACTOR, HBV_MELT_ASP_CORR",
                "  :Units     ,        mm/d/K,          mm/d/K,                none,          mm/d/K,              none",
                "  #              MELT_FACTOR,        CONSTANT,         HBV_MELT_FOR_CORR,     REFREEZE_FACTOR,          CONSTANT",
                f"    [DEFAULT],  {self.params['HBV'][param_or_name]['X02']},             2.2,        {self.params['HBV'][param_or_name]['X18']},    {self.params['HBV'][param_or_name]['X03']},              0.48",
                ":EndLandUseParameterList",
                "",
                # Glacier land use parameters: active when NOT coupled (Raven handles glacier melt),
                # commented when coupled (GloGEM handles glacier melt externally)
                (":LandUseParameterList" if not self.coupled else "#:LandUseParameterList"),
                ("  :Parameters, HBV_MELT_GLACIER_CORR,   HBV_GLACIER_KMIN, GLAC_STORAGE_COEFF, HBV_GLACIER_AG" if not self.coupled else
                 "# :Parameters, HBV_MELT_GLACIER_CORR,   HBV_GLACIER_KMIN, GLAC_STORAGE_COEFF, HBV_GLACIER_AG"),
                ("  :Units     ,                  none,                1/d,                1/d,           1/mm" if not self.coupled else
                 "# :Units     ,                  none,                1/d,                1/d,           1/mm"),
                ("  #                        CONSTANT,           CONSTANT,        GLAC_STORAGE_COEFF,       CONSTANT," if not self.coupled else
                 "#   #                       CONSTANT,           CONSTANT,        GLAC_STORAGE_COEFF,       CONSTANT,"),
                (f"    [DEFAULT],                  1.64,               0.05,       {glac_storage_coeff},           0.05" if not self.coupled else
                 f"#   [DEFAULT],                  1.64,               0.05,       {glac_storage_coeff},           0.05"),
                (":EndLandUseParameterList" if not self.coupled else "#:EndLandUseParameterList")
            ],
            "#Soil Parameters": [
                f"#For Ostrich:HBV_Alpha= {self.params['HBV'][param_or_name]['X15']}",
                ":SoilParameterList",
                "  :Parameters,                POROSITY,FIELD_CAPACITY,     SAT_WILT,     HBV_BETA, MAX_CAP_RISE_RATE,  MAX_PERC_RATE,  BASEFLOW_COEFF,            BASEFLOW_N",
                "  :Units     ,                    none,          none,         none,         none,              mm/d,           mm/d,             1/d,                  none",
                "  #                        POROSITY,   FIELD_CAPACITY,  SAT_WILT,  BETA,       MAX_CAP_RISE_RATE,       CONSTANT,        CONSTANT,              CONSTANT,",
                f"    [DEFAULT],            {self.params['HBV'][param_or_name]['X05']},  {self.params['HBV'][param_or_name]['X06']}, {self.params['HBV'][param_or_name]['X14']}, {self.params['HBV'][param_or_name]['X07']},      {self.params['HBV'][param_or_name]['X16']},            0.0,             0.0,                   0.0",
                "  #                                                        CONSTANT,                                     MAX_PERC_RATE,     BASEFLOW_COEFF_FAST, 1+BASEFLOW_N=1+ALPHA,",
                f"     FAST_RES,                _DEFAULT,      _DEFAULT,          0.0,     _DEFAULT,          _DEFAULT,   {self.params['HBV'][param_or_name]['X08']},    {self.params['HBV'][param_or_name]['X09']},              {self.params['HBV'][param_or_name]['X15']}",
                "  #                                                        CONSTANT,                                                      BASEFLOW_COEFF_SLOW,              CONSTANT,",
                f"     SLOW_RES,                _DEFAULT,      _DEFAULT,          0.0,     _DEFAULT,          _DEFAULT,       _DEFAULT,    {self.params['HBV'][param_or_name]['X10']},                   1.0",
                ":EndSoilParameterList"
            ]
        }

    def create_rvi_file(self, template: bool = False):
        """
        Write Raven .rvi file for HBV model.
        
        Args:
            template: Whether to create template file
        """
        file_path, param_or_name = self._get_file_path('rvi', template)
        
        # Get HRU groups definition
        hru_groups_definition = self._get_hru_groups_definition()
        
        # Use namelist dates
        rvi_sections = self._create_rvi_sections(self.start_date, self.end_date, 
                                            self.cali_end_date, hru_groups_definition)

        # Write the file
        with open(file_path, 'w') as ff:
            ff.writelines(f"{line}\n" for line in self._create_header("rvi"))
            for section, lines in rvi_sections.items():
                ff.write(f"{section}\n")
                ff.writelines(line + '\n' for line in lines)
                ff.write('\n')

    def _get_hru_groups_definition(self) -> str:
        """Get HRU groups definition string."""
        hrus_by_band = self.get_hrus_by_elevation_band()
        
        # ✅ ADD GLACIER GROUPS TO DEFINITION
        base_groups = "AllHRUs ALL_GLACIER SMALL_GLACIER LARGE_GLACIER NO_GLACIER"
        
        if hrus_by_band:
            elevation_bands = sorted(hrus_by_band.keys(), key=lambda x: int(x.split('-')[0]))
            return f":DefineHRUGroups {base_groups} " + " ".join(elevation_bands)
        else:
            print(f"No elevation bands available for catchment {self.gauge_id}, only defining base groups")
            return f":DefineHRUGroups {base_groups}"

    def _create_rvi_sections(self, start_date: str, end_date: str, cali_end_date: str,
                        hru_groups_definition: str) -> Dict[str, List[str]]:
        """Create all sections for RVI file."""
        
        # ✅ DETERMINE ACTUAL START DATE (warm-up or simulation)
        if hasattr(self, 'warm_up_date') and self.warm_up_date is not None:
            actual_start_date = self.warm_up_date
            print(f"RVI will use warm-up start date: {actual_start_date}")
        else:
            actual_start_date = start_date
            print(f"RVI will use simulation start date: {actual_start_date}")
        
        # ✅ CALCULATE DURATION (from actual start to end)
        from datetime import datetime
        start_dt = datetime.strptime(actual_start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        duration_days = (end_dt - start_dt).days
        
        return {
            "#Model Organisation": [
                f":StartDate             {actual_start_date} 00:00:00",  # ✅ Use warm-up date
                f":EndDate               {end_date} 00:00:00",
                ":TimeStep              1.0",
                f":RunName               {self.gauge_id}_HBV",
                f"# Duration: {duration_days} days (includes warm-up)" if self.warm_up_date else f"# Duration: {duration_days} days"
            ],
            "#Model Options": [
                ":Routing             	    ROUTE_NONE",
                ":CatchmentRoute      	    TRIANGULAR_UH",
                f":Evaporation         	    {'PET_OUDIN' if self._pet_method and self._pet_method.upper() == 'OUDIN' else 'PET_FROMMONTHLY'}",
                f":OW_Evaporation      	    {'PET_OUDIN' if self._pet_method and self._pet_method.upper() == 'OUDIN' else 'PET_FROMMONTHLY'}",
                ":SWRadiationMethod   	    SW_RAD_DEFAULT",
                ":SWCloudCorrect      	    SW_CLOUD_CORR_NONE",
                ":SWCanopyCorrect     	    SW_CANOPY_CORR_NONE",
                ":LWRadiationMethod   	    LW_RAD_DEFAULT",
                ":RainSnowFraction    	    RAINSNOW_HBV",
                ":PotentialMeltMethod 	    POTMELT_HBV",
                ":OroTempCorrect      	    OROCORR_HBV",
                ":OroPrecipCorrect    	    OROCORR_HBV",
                ":OroPETCorrect       	    OROCORR_HBV",
                ":CloudCoverMethod    	    CLOUDCOV_NONE",
                ":PrecipIceptFract    	    PRECIP_ICEPT_USER",
                ":MonthlyInterpolationMethod MONTHINT_LINEAR_21",
                *([f":SubdailyMethod          {self.config.get('subdaily_method')}"] if self.config.get('subdaily_method') else []),
                ":SoilModel                  SOIL_MULTILAYER 3",
                f":EvaluationPeriod   CALIBRATION   {start_date}   {cali_end_date}",  # ✅ Still use simulation dates
                f":EvaluationPeriod   VALIDATION    {cali_end_date}   {end_date}"      # ✅ Still use simulation dates
            ],
            "#Soil Alias Layer Definitions": [
                ":Alias       FAST_RESERVOIR SOIL[1]",
                ":Alias       SLOW_RESERVOIR SOIL[2]",
                ":LakeStorage SLOW_RESERVOIR"
            ],
            "#HRU Groups Definition": [
                hru_groups_definition
            ] if hru_groups_definition else [],
            "#Hydrologic Process Order": [
                ":HydrologicProcesses",
                "   :Flush             RAVEN_DEFAULT      SNOW            ATMOSPHERE",
                "       :-->Conditional HRU_TYPE IS MASKED_GLACIER",
                "   :SnowRefreeze      FREEZE_DEGREE_DAY  SNOW_LIQ        SNOW",
                "   :Precipitation     PRECIP_RAVEN       ATMOS_PRECIP    MULTIPLE",
                "   :CanopyEvaporation CANEVP_ALL         CANOPY          ATMOSPHERE",
                "   :CanopySnowEvap    CANEVP_ALL         CANOPY_SNOW     ATMOSPHERE",
                "   :SnowBalance       SNOBAL_SIMPLE_MELT SNOW            SNOW_LIQ",
                "       :-->Overflow     RAVEN_DEFAULT      SNOW_LIQ        PONDED_WATER",
                # Glacier processes: active when NOT coupled (Raven handles glacier melt),
                # commented when coupled (GloGEM handles glacier melt externally)
                ("   :Flush             RAVEN_DEFAULT      PONDED_WATER    GLACIER" if not self.coupled else
                 "#   :Flush             RAVEN_DEFAULT      PONDED_WATER    GLACIER"),
                ("       :-->Conditional HRU_TYPE IS GLACIER" if not self.coupled else
                 "#       :-->Conditional HRU_TYPE IS GLACIER"),
                ("   :GlacierMelt       GMELT_HBV          GLACIER_ICE     GLACIER" if not self.coupled else
                 "#   :GlacierMelt       GMELT_HBV          GLACIER_ICE     GLACIER"),
                ("   :GlacierRelease    GRELEASE_HBV_EC    GLACIER         SURFACE_WATER" if not self.coupled else
                 "#   :GlacierRelease    GRELEASE_HBV_EC    GLACIER         SURFACE_WATER"),
                # DeltaH glacier geometry update (only when glacier_method: 'DeltaH' and not coupled)
                *(["   :SnowToIce         ANNUAL             SNOW            GLACIER_ICE",
                   "   :GlacierDeltaH     HUSS               GLACIER_ICE     GLACIER_ICE"]
                  if not self.coupled and self.config.get('glacier_method') == 'DeltaH' else []),
                "   :Infiltration      INF_HBV            PONDED_WATER    MULTIPLE",
                "   :Flush             RAVEN_DEFAULT      SURFACE_WATER   FAST_RESERVOIR",
                "       :-->Conditional HRU_TYPE IS_NOT GLACIER",
                "       :-->Conditional HRU_TYPE IS_NOT MASKED_GLACIER",
                "       :-->Conditional HRU_TYPE IS_NOT ROCK",
                "       :-->Conditional HRU_TYPE IS_NOT LAKE",
                "   :SoilEvaporation   SOILEVAP_HBV       SOIL[0]         ATMOSPHERE",
                "   :CapillaryRise     RISE_HBV           FAST_RESERVOIR 	SOIL[0]",
                "   :LakeEvaporation   LAKE_EVAP_BASIC    SLOW_RESERVOIR  ATMOSPHERE",
                "   :Percolation       PERC_CONSTANT      FAST_RESERVOIR 	SLOW_RESERVOIR",
                "   :Baseflow          BASE_POWER_LAW     FAST_RESERVOIR  SURFACE_WATER",
                "   :Baseflow          BASE_LINEAR        SLOW_RESERVOIR  SURFACE_WATER",
                "   :SnowRedistribute  THRESHOLD          SNOW            35000.0",
                "   :LateralEquilibrate RAVEN_DEFAULT AllHRUs FAST_RESERVOIR 1.0",
                "   :LateralEquilibrate RAVEN_DEFAULT AllHRUs SLOW_RESERVOIR 1.0",
                ":EndHydrologicProcesses"
            ],
            "#Output Options": [
            ],
        }

    def create_rvc_file(self, template: bool = False):
        """
        Write Raven .rvc file for HBV model.
        
        Args:
            template: Whether to create template file
        """
        file_path, param_or_name = self._get_file_path('rvc', template)

        # Define RVC configuration
        rvc_sections = {
            "#Basin": [
                ":BasinInitialConditions",
                ":Attributes, ID,              Q",
                ":Units,      none,         m3/s",
                "#                  HBV_PARA_???",
                "1,             1.0",
                ":EndBasinInitialConditions"
            ],
            "#Lower Groundwater Storage": [
                "# Initial Lower groundwater storage - for each HRU",
                "",
                ":InitialConditions SOIL[2]",
                "# derived from thickness: HBV_THICKNESS_TOPSOIL [m] * 1000.0 / 2.0",
                f"{self.params['HBV'][param_or_name]['HBV_Initial_Thickness_Topsoil']}",
                ":EndInitialConditions"
            ]
        }

        # Write the file
        with open(file_path, 'w') as ff:
            ff.writelines(f"{line}\n" for line in self._create_header("rvc"))
            for section, lines in rvc_sections.items():
                ff.write(f"{section}\n")
                ff.writelines(line + '\n' for line in lines)
                ff.write('\n')

            # DeltaH glacier initial ice volumes
            if not self.coupled and self.config.get('glacier_method') == 'DeltaH':
                self._write_glacier_ice_initial_conditions(ff)

    def _write_glacier_ice_initial_conditions(self, ff):
        """Write :HRUStateVariableTable with GLACIER_ICE column (mm w.e.)."""
        ice_path = self.topo_files_dir / 'glacier_ice_thickness.csv'
        if not ice_path.exists():
            print(f"WARNING: {ice_path} not found, skipping GLACIER_ICE initial conditions")
            return

        ice_df = pd.read_csv(ice_path)
        hru_table = pd.read_csv(self.topo_files_dir / 'HRU_table.csv')

        # Build lookup: HRU_ID → ice thickness (mm w.e.)
        ice_lookup = dict(zip(ice_df['HRU_ID'], ice_df['ice_thickness_mm']))

        # HRU IDs are in ':ATTRIBUTES' column in the Raven HRU table
        hru_id_col = ':ATTRIBUTES' if ':ATTRIBUTES' in hru_table.columns else 'HRU_ID'

        ff.write("# Glacier ice initial conditions (DeltaH method)\n")
        ff.write(":HRUStateVariableTable\n")
        ff.write("  :Attributes,    GLACIER_ICE\n")
        ff.write("  :Units,         mm\n")
        for hru_id in hru_table[hru_id_col]:
            ice_val = ice_lookup.get(int(hru_id), 0.0)
            ff.write(f"  {int(hru_id)},         {ice_val:.2f}\n")
        ff.write(":EndHRUStateVariableTable\n\n")

    def create_all_files(self, template: bool = False):
        """
        Create all Raven input files for HBV model using namelist configuration.
        
        Args:
            template: Whether to create template files
        """
        print(f"Creating {'template' if template else 'initialized'} HBV files for catchment {self.gauge_id}")
        
        try:
            self.create_rvh_file(template)
            self.create_rvt_file(template)
            self.create_rvp_file(template)
            self.create_rvi_file(template)
            self.create_rvc_file(template)
            
            print(f"Successfully created all {'template' if template else 'initialized'} HBV files!")
            
        except Exception as e:
            print(f"Error creating HBV files: {e}")
            raise

