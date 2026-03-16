"""
HYMOD Model Preprocessing Module for Raven Hydrological Modeling Framework

This module provides a comprehensive preprocessing class for setting up HYMOD model
runs in the Raven hydrological modeling framework. It handles the creation of
all necessary Raven input files (.rvh, .rvt, .rvp, .rvi, .rvc) with HYMOD-specific
configurations.

Author: Justine Berg
Date: November 2025
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

#--------------------------------------------------------------------------------
############################ HYMOD Preprocessor Class ###########################
#--------------------------------------------------------------------------------

class HYMODPreprocessor:
    """
    A comprehensive preprocessor for HYMOD model setup in Raven.
    
    This class handles all aspects of HYMOD model preprocessing including:
    - Creation of Raven input files (.rvh, .rvt, .rvp, .rvi, .rvc)
    - HRU management and elevation band grouping
    - Parameter handling for both template and initialized files
    - Forcing data configuration
    """
    
    def __init__(self, namelist_path: Union[str, Path]):
        """
        Initialize the HYMOD preprocessor with namelist file.
        
        Args:
            namelist_path: Path to the YAML namelist file
        """
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
        self.config_dir = namelist['config_dir']
        self.model_type = namelist['model_type']
        self.start_date = namelist['start_date']
        self.end_date = namelist['end_date']
        self.cali_end_date = namelist['cali_end_date']
        self.coupled = namelist.get('coupled', False)
        self._irrigation_variable = namelist.get('irrigation_variable', 'discharge').lower()
        self._pet_method = namelist.get('pet_method', None)
        self.author = namelist.get('author', 'Justine Berg')
        self.meteo_source = namelist.get('meteo_source', 'ERA5').upper()

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
        
        # Remove gauge_info loading - we'll get it from DEM like HBV
        self.gauge_lat = None
        self.gauge_lon = None
        self.station_elevation = None
        
        # Set up directory structure
        self.model_dir = self.main_dir / self.config_dir
        self.catchment_dir = self.model_dir / f'catchment_{self.gauge_id}'
        self.hymod_dir = self.catchment_dir / self.model_type
        self.templates_dir = self.hymod_dir / 'templates'
        self.data_obs_dir = self.hymod_dir / 'data_obs'
        self.topo_files_dir = self.catchment_dir / 'topo_files'
        self.shared_data_dir = self.catchment_dir / 'data_obs'

        # Ensure directories exist
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.hymod_dir, self.templates_dir, self.data_obs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def icemelt_mode(self) -> bool:
        """True when coupled + icemelt: glacier HRUs are ROCK with elevation discretization."""
        return self.coupled and self._irrigation_variable == 'icemelt'

    def _get_glacier_hru_ids_from_overlap(self, hru_df) -> dict:
        """Identify glacier-origin ROCK HRUs via overlap CSV for icemelt mode."""
        import pandas as _pd
        overlap_path = self.topo_files_dir / 'glacier_hru_overlap.csv'
        mapping_path = self.topo_files_dir / 'glacier_id_mapping.csv'
        if not overlap_path.exists() or not mapping_path.exists():
            all_ids = hru_df[':ATTRIBUTES'].tolist()
            return {'all': [], 'large': [], 'small': [], 'no_glacier': all_ids}
        overlap_df = _pd.read_csv(overlap_path)
        mapping_df = _pd.read_csv(mapping_path)
        large_ints = {int(r['numeric_id']) for _, r in mapping_df.iterrows() if bool(r['is_large'])}
        glacier_hru_ids = set(overlap_df['HRU_ID'].unique())
        all_g, large_g, small_g = [], [], []
        for _, row in hru_df.iterrows():
            hid = int(row[':ATTRIBUTES'])
            if hid not in glacier_hru_ids:
                continue
            all_g.append(hid)
            hru_ov = overlap_df[overlap_df['HRU_ID'] == hid]
            if any(int(r['glacier_numeric_id']) in large_ints for _, r in hru_ov.iterrows()):
                large_g.append(hid)
            else:
                small_g.append(hid)
        no_g = sorted(set(hru_df[':ATTRIBUTES'].tolist()) - set(all_g))
        return {'all': all_g, 'large': large_g, 'small': small_g, 'no_glacier': no_g}

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
            f"# Emulation of HYMOD simulation of {self.gauge_id}",
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
            file_name = f"{self.gauge_id}_HYMOD.{file_type}.tpl"
            file_path = self.templates_dir / file_name
        else:
            param_or_name = "init"
            file_name = f"{self.gauge_id}_HYMOD.{file_type}"
            file_path = self.hymod_dir / file_name
        
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

    def get_hrus_by_elevation_band(self) -> Dict[str, List[int]]:
        """
        Load HRU shapefile and create a dictionary of HRU IDs for each elevation band.
        Excludes GLACIER and MASKED_GLACIER land use classes from elevation bands.
        
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
        
        # Define land use classes to exclude from elevation bands
        elevation_excluded_landuse = ['GLACIER', 'MASKED_GLACIER']
        
        # Filter out excluded landuse classes for elevation bands
        filtered_hru = hru_gdf[~hru_gdf['Landuse_Cl'].isin(elevation_excluded_landuse)]
        
        # Filter out rows with NaN elevation values
        filtered_hru = filtered_hru.dropna(subset=['Elev_Min', 'Elev_Max'])
        
        if len(filtered_hru) == 0:
            print(f"No HRUs with valid elevation data for catchment {self.gauge_id} after excluding {elevation_excluded_landuse}")
            return {}
        
        # Print info about excluded HRUs
        total_excluded = len(hru_gdf) - len(filtered_hru)
        if total_excluded > 0:
            print(f"Excluded {total_excluded} HRUs from elevation bands (land use: {elevation_excluded_landuse})")
        
        # Create elevation band labels
        filtered_hru['ElevationBand'] = filtered_hru.apply(
            lambda row: f"{int(row['Elev_Min'])}-{int(row['Elev_Max'])}m", axis=1)
        
        # Group by elevation band and collect HRU IDs
        hrus_by_band = {}
        for band in filtered_hru['ElevationBand'].unique():
            band_hrus = filtered_hru[filtered_hru['ElevationBand'] == band]['HRU_ID'].tolist()
            hrus_by_band[band] = band_hrus
        
        # Print summary
        print(f"Found {len(hrus_by_band)} elevation bands with a total of {len(filtered_hru)} HRUs (glaciers excluded)")
        for band in sorted(hrus_by_band.keys(), key=lambda x: int(x.split('-')[0])):
            print(f"  {band}: {len(hrus_by_band[band])} HRUs")
        
        return hrus_by_band

    def create_rvh_file(self, template: bool = False):
        """
        Write Raven .rvh file for HYMOD model.
        
        Args:
            template: Whether to create a template file
        """
        file_path, param_or_name = self._get_file_path('rvh', template)
        
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
            ":RedirectToFile  ../data_obs/connections.rvh",
            "#:EndLateralConnections",
            ""
        ]

        # Create HRU groups
        hru_groups = self._create_hru_groups(HRU)

        # Define subbasins (multi or single)
        subbasins_config = self.config.get('subbasins', None)

        if subbasins_config:
            routing_path = self.topo_files_dir / 'subbasin_routing.yaml'
            if not routing_path.exists():
                raise FileNotFoundError(
                    f"subbasin_routing.yaml not found at {routing_path}. "
                    "Run MultiSubbasinProcessor.process_all_subbasins() first."
                )
            with open(routing_path) as _f:
                routing = yaml.safe_load(_f)

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
                    f"  {sb['id']:10d},  {self.params['HYMOD'][param_or_name]['X01']},                  3,"
                )

            subbasin_properties = [
                ":SubBasinProperties",
                "#                         HYMOD_PARA_1,                  3,",
                "   :Parameters,           RES_CONSTANT,     NUM_RESERVOIRS,",
                "   :Units,                         1/d,                  -,",
                *sp_rows,
                ":EndSubBasinProperties",
            ]

        else:
            subbasins = [
                ":SubBasins",
                "  :Attributes,          NAME, DOWNSTREAM_ID,PROFILE,REACH_LENGTH,       GAUGED",
                "  :Units     ,          none,          none,   none,          km,         none",
                f"            1,        {self.gauge_id},            -1,   NONE,       _AUTO,     1",
                ":EndSubBasins"
            ]

            subbasin_properties = [
                ":SubBasinProperties",
                "#                         HYMOD_PARA_1,                  3,",
                "   :Parameters,           RES_CONSTANT,     NUM_RESERVOIRS,",
                "   :Units,                         1/d,                  -,",
                f"              1,          {self.params['HYMOD'][param_or_name]['X01']},                  3,",
                ":EndSubBasinProperties"
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
                
        print(f"Successfully wrote HYMOD RVH file to {file_path}")

    def _create_hru_groups(self, hru_df: pd.DataFrame) -> List[str]:
        """Create HRU groups including filtered AllHRUs, elevation bands, and glacier groups."""
        hru_groups = []
        
        # Different exclusion rules for different groups
        allhrus_excluded_landuse = ['GLACIER', 'ROCK', 'MASKED_GLACIER', 'LAKE']
        elevation_excluded_landuse = ['GLACIER', 'MASKED_GLACIER']  # Only exclude glaciers for elevation bands
        
        # Get all HRU IDs excluding the specified land use classes for AllHRUs
        filtered_hrus = hru_df[~hru_df['LAND_USE_CLASS'].isin(allhrus_excluded_landuse)]
        filtered_hru_ids = filtered_hrus[':ATTRIBUTES'].tolist()
        
        print(f"Total HRUs: {len(hru_df)}")
        print(f"Excluded HRUs from AllHRUs group (land use {allhrus_excluded_landuse}): {len(hru_df) - len(filtered_hrus)}")
        print(f"AllHRUs group will contain: {len(filtered_hru_ids)} HRUs")
        
        # Add AllHRUs group with filtered HRUs
        if filtered_hru_ids:
            hru_groups.extend([
                ":HRUGroup AllHRUs",
                f"  {' '.join(map(str, filtered_hru_ids))}",
                ":EndHRUGroup",
                ""
            ])
        else:
            print("WARNING: No HRUs remain after filtering for AllHRUs group!")
            hru_groups.extend([
                ":HRUGroup AllHRUs",
                "  # No HRUs available after filtering",
                ":EndHRUGroup",
                ""
            ])

        # ✅ GLACIER-RELATED HRU GROUPS
        if self.icemelt_mode:
            glacier_groups = self._get_glacier_hru_ids_from_overlap(hru_df)
            all_glacier_hrus = glacier_groups['all']
            large_glacier_hrus = glacier_groups['large']
            small_glacier_hrus = glacier_groups['small']
            no_glacier_hrus = glacier_groups['no_glacier']
            print(f"Icemelt mode: identified {len(all_glacier_hrus)} glacier-origin ROCK HRUs from overlap CSV")
        else:
            all_glacier_hrus = hru_df[hru_df['LAND_USE_CLASS'].isin(['GLACIER', 'MASKED_GLACIER'])][':ATTRIBUTES'].tolist()
            small_glacier_hrus = hru_df[
                (hru_df['LAND_USE_CLASS'].isin(['GLACIER', 'MASKED_GLACIER'])) &
                (hru_df['AREA'] < 2)
            ][':ATTRIBUTES'].tolist()
            large_glacier_hrus = hru_df[
                (hru_df['LAND_USE_CLASS'].isin(['GLACIER', 'MASKED_GLACIER'])) &
                (hru_df['AREA'] >= 2)
            ][':ATTRIBUTES'].tolist()
            no_glacier_hrus = hru_df[~hru_df['LAND_USE_CLASS'].isin(['GLACIER', 'MASKED_GLACIER'])][':ATTRIBUTES'].tolist()

        for group_name, group_hrus in [
            ('ALL_GLACIER', all_glacier_hrus),
            ('SMALL_GLACIER', small_glacier_hrus),
            ('LARGE_GLACIER', large_glacier_hrus),
            ('NO_GLACIER', no_glacier_hrus),
        ]:
            if group_hrus:
                hru_groups.extend([
                    f":HRUGroup {group_name}",
                    f"  {' '.join(map(str, group_hrus))}",
                    ":EndHRUGroup",
                    ""
                ])
            else:
                hru_groups.extend([
                    f":HRUGroup {group_name}",
                    f"  # No {group_name} HRUs available",
                    ":EndHRUGroup",
                    ""
                ])
            print(f"{group_name} group: {len(group_hrus)} HRUs")

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
                    
                    if filtered_band_hru_ids:
                        hru_groups.extend([
                            f":HRUGroup {band}",
                            f"  {' '.join(map(str, filtered_band_hru_ids))}",
                            ":EndHRUGroup",
                            ""
                        ])
        else:
            print(f"No elevation bands available for catchment {self.gauge_id}")

        return hru_groups

    def create_rvt_file(self, template: bool = False):
        """Write Raven .rvt file for HYMOD model."""
        if self.config.get('subbasins'):
            self._create_rvt_file_multi_subbasin(template)
        else:
            self._create_rvt_file_single(template)

    def _create_rvt_file_single(self, template: bool = False):
        """Write .rvt for a single-basin namelist."""
        file_path, param_or_name = self._get_file_path('rvt', template)

        print(f"Extracting gauge location from DEM for gauge {self.gauge_id}...")
        try:
            gauge_lat, gauge_lon, station_elevation = self.get_gauge_location_from_dem()
        except Exception as e:
            print(f"Error: Could not extract gauge location from DEM: {e}")
            return

        gauge_info = self._create_gauge_info(gauge_lat, gauge_lon, station_elevation)
        forcing_data = self._create_forcing_block(param_or_name)

        with open(file_path, 'w') as ff:
            ff.writelines(f"{line}\n" for line in self._create_header("rvt"))
            ff.write("# meteorological forcings\n")
            for f in forcing_data.values():
                for t in f:
                    ff.write(f"{t}\n")
            ff.writelines(gauge_info)
            ff.write(f":RedirectToFile ../data_obs/Q_daily.rvt\n")

    def _create_rvt_file_multi_subbasin(self, template: bool = False):
        """Write .rvt for a multi-subbasin namelist (one :Gauge block per subbasin)."""
        file_path, param_or_name = self._get_file_path('rvt', template)
        subbasins_config = self.config['subbasins']
        forcing_data = self._create_forcing_block(param_or_name)

        with open(file_path, 'w') as ff:
            ff.writelines(f"{line}\n" for line in self._create_header("rvt"))
            ff.write("# meteorological forcings\n")
            for f in forcing_data.values():
                for t in f:
                    ff.write(f"{t}\n")

            for sb in subbasins_config:
                sb_id = sb['id']
                sb_gid = str(sb['gauge_id'])
                gauged = sb.get('gauged', 0)

                sb_data_dir = self.shared_data_dir / f'subbasin_{sb_id}'
                lat, lon, elev = self._get_subbasin_centroid_location(sb_gid)
                print(f"  Subbasin {sb_id} ({sb_gid}): lat={lat}, lon={lon}, elev={elev}")

                gauge_info = self._create_gauge_info(lat, lon, elev,
                                                     gauge_id=sb_gid,
                                                     subbasin_data_dir=sb_data_dir)
                ff.writelines(gauge_info)

                if gauged:
                    ff.write(f":RedirectToFile ../data_obs/Q_daily_{sb_gid}.rvt\n\n")

    def _create_gauge_info(self, gauge_lat: float, gauge_lon: float,
                           station_elevation: float,
                           gauge_id: str = None,
                           subbasin_data_dir=None) -> List[str]:
        """Create gauge information section for RVT file."""
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

    def _get_monthly_data(self, subbasin_data_dir=None) -> List[str]:
        """Read and format monthly temperature and PET data if available."""
        # Skip monthly PET data when using PET_OUDIN (computes PET from temperature internally)
        if self._pet_method and self._pet_method.upper() == 'OUDIN':
            return []

        if self.meteo_source == 'HAR':
            temp_filename = 'har_monthly_temperature_averages.csv'
            pet_filename = 'har_monthly_pet_averages.csv'
        else:
            temp_filename = 'monthly_temperature_averages.csv'
            pet_filename = 'monthly_pet_averages.csv'

        if subbasin_data_dir is not None:
            sb_temp = Path(subbasin_data_dir) / temp_filename
            sb_pet = Path(subbasin_data_dir) / pet_filename
            if sb_temp.exists() and sb_pet.exists():
                monthly_temp_file, monthly_pet_file = sb_temp, sb_pet
            else:
                monthly_temp_file = self.data_obs_dir / temp_filename
                monthly_pet_file = self.data_obs_dir / pet_filename
        else:
            monthly_temp_file = self.data_obs_dir / temp_filename
            monthly_pet_file = self.data_obs_dir / pet_filename

        if not monthly_temp_file.exists() or not monthly_pet_file.exists():
            shared = self.catchment_dir / 'data_obs'
            if not monthly_temp_file.exists():
                s = shared / temp_filename
                if s.exists():
                    monthly_temp_file = s
            if not monthly_pet_file.exists():
                s = shared / pet_filename
                if s.exists():
                    monthly_pet_file = s

        if not (monthly_temp_file.exists() and monthly_pet_file.exists()):
            return []

        try:
            temp_df = pd.read_csv(monthly_temp_file)
            pet_df = pd.read_csv(monthly_pet_file)

            temp_values = temp_df['Temperature'].values
            temp_str = ", ".join([f"{val:.1f}" for val in temp_values])

            pet_values = pet_df['PET_avg_mm_per_day'].values
            pet_str = ", ".join([f"{val:.3f}" for val in pet_values])

            return [
                "#                       Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec \n",
                f"  :MonthlyAveEvaporation, {pet_str} \n",
                f"  :MonthlyAveTemperature, {temp_str} \n"
            ]
        except Exception as e:
            print(f"Error reading monthly data: {e}")
            return []

    def _create_forcing_block(self, param_or_name: str) -> Dict[str, List[str]]:
        """Create forcing data configuration for RVT file (ERA5-Land or HAR)."""
        if self.meteo_source == 'HAR':
            grid_weights_file_path = "../data_obs/GridWeights_HAR.txt"
            dim_names = "west_east south_north time"
            precip_tuple = ('Rainfall', 'RAINFALL', 'har_precip.nc', 'prcp')
            other_types = [
                ('Average Temperature', 'TEMP_AVE', 'har_temp_mean.nc', 't2_mean'),
                ('Maximum Temperature', 'TEMP_MAX', 'har_temp_max.nc',  't2_max'),
                ('Minimum Temperature', 'TEMP_MIN', 'har_temp_min.nc',  't2_min'),
            ]
        else:  # ERA5 (default)
            grid_weights_file_path = "../data_obs/GridWeights.txt"
            dim_names = "longitude latitude time"
            precip_tuple = ('Rainfall', 'RAINFALL', 'era5_land_precip.nc', 'tp')
            other_types = [
                ('Average Temperature', 'TEMP_AVE', 'era5_land_temp_mean.nc', 't2m'),
                ('Maximum Temperature', 'TEMP_MAX', 'era5_land_temp_max.nc',  't2m'),
                ('Minimum Temperature', 'TEMP_MIN', 'era5_land_temp_min.nc',  't2m'),
            ]

        # Precip weights/dims default to meteo_source values
        precip_weights_file = grid_weights_file_path
        precip_dim_names = dim_names

        # Override all forcing files when running CORDEX climate projections.
        if self.config.get('future', False) and self.config.get('cordex_models'):
            models    = self.config['cordex_models']
            scenarios = self.config.get('cordex_scenarios', ['rcp45'])
            model_id  = models[0].replace('/', '_').replace(' ', '_')
            scenario  = scenarios[0]
            precip_tuple = (
                'Rainfall', 'RAINFALL',
                f'cordex_{model_id}_{scenario}_precip.nc', 'tp',
            )
            other_types = [
                ('Average Temperature', 'TEMP_AVE',
                 f'cordex_{model_id}_{scenario}_temp_mean.nc', 't2m'),
                ('Maximum Temperature', 'TEMP_MAX',
                 f'cordex_{model_id}_{scenario}_temp_max.nc',  't2m'),
                ('Minimum Temperature', 'TEMP_MIN',
                 f'cordex_{model_id}_{scenario}_temp_min.nc',  't2m'),
            ]
            grid_weights_file_path = "../data_obs/GridWeights.txt"
            precip_weights_file    = grid_weights_file_path
            dim_names        = "longitude latitude time"
            precip_dim_names = dim_names

        # Override all forcing files when running CMIP6 climate projections.
        if self.config.get('future', False) and self.config.get('cmip6_models'):
            models    = self.config['cmip6_models']
            scenarios = self.config.get('cmip6_scenarios', ['ssp126'])
            model_id  = models[0].replace('/', '_').replace(' ', '_')
            future_scens = [s for s in scenarios if s != 'historical']
            scenario  = future_scens[0] if future_scens else scenarios[0]
            precip_tuple = (
                'Rainfall', 'RAINFALL',
                f'cmip6_{model_id}_{scenario}_precip.nc', 'tp',
            )
            other_types = [
                ('Average Temperature', 'TEMP_AVE',
                 f'cmip6_{model_id}_{scenario}_temp_mean.nc', 't2m'),
                ('Maximum Temperature', 'TEMP_MAX',
                 f'cmip6_{model_id}_{scenario}_temp_max.nc',  't2m'),
                ('Minimum Temperature', 'TEMP_MIN',
                 f'cmip6_{model_id}_{scenario}_temp_min.nc',  't2m'),
            ]
            grid_weights_file_path = "../data_obs/GridWeights.txt"
            precip_weights_file    = grid_weights_file_path
            dim_names        = "longitude latitude time"
            precip_dim_names = dim_names
            print(f"Using CMIP6 forcing: {model_id} / {scenario}")

        # ✅ Override precipitation source if TPHiPr is requested
        if self.config.get('precip_source', '').upper() == 'TPHIPR':
            precip_tuple = ('Rainfall', 'RAINFALL', 'tphipr_precip.nc', 'prcp')
            precip_weights_file = "../data_obs/GridWeights_TPHiPr.txt"
            precip_dim_names = "longitude latitude time"

        # Get optional precip correction parameters
        rain_corr = self.params['HYMOD'][param_or_name].get('X09', 1.0)
        snow_corr = self.params['HYMOD'][param_or_name].get('X10', 1.0)
        use_precip_corr = self.config.get('precip_correction', False)
        comment = "" if use_precip_corr else "#"

        forcing_data = {}

        # Rainfall block (potentially different weights/dims than T/PET)
        # TPHiPr has no elevation variable (high-res product, no lapse correction needed)
        precip_elev_line = ("    :ElevationVarNameNC   elevation"
                            if self.config.get('precip_source', '').upper() != 'TPHIPR'
                            else "    # No ElevationVarNameNC for TPHiPr (high-res gridded product)")
        name, forcing_type, filename, var_name = precip_tuple
        forcing_data[name] = [
            f":GriddedForcing           {name}",
            f"    :ForcingType          {forcing_type}",
            f"    :FileNameNC           ../data_obs/{filename}",
            f"    :VarNameNC            {var_name}",
            f"    :DimNamesNC           {precip_dim_names}",
            precip_elev_line,
            f"    {comment}:RainCorrection       {rain_corr}",
            f"    {comment}:SnowCorrection       {snow_corr}",
            f"    :RedirectToFile       {precip_weights_file}",
            ":EndGriddedForcing",
            ''
        ]

        # Temperature blocks (always from meteo_source)
        for name, forcing_type, filename, var_name in other_types:
            forcing_data[name] = [
                f":GriddedForcing           {name}",
                f"    :ForcingType          {forcing_type}",
                f"    :FileNameNC           ../data_obs/{filename}",
                f"    :VarNameNC            {var_name}",
                f"    :DimNamesNC           {dim_names}",
                "    :ElevationVarNameNC   elevation",
                f"    :RedirectToFile       {grid_weights_file_path}",
                ":EndGriddedForcing",
                ''
            ]

        # Irrigation is always ERA5-based (independent of meteo_source)
        forcing_data['Irrigation'] = [
            ":GriddedForcing           Irrigation",
            "    :ForcingType          IRRIGATION",
            "    :FileNameNC           ../data_obs/irrigation.nc",
            "    :VarNameNC            data",
            "    :DimNamesNC           x y time     # must be in the order of (x,y,t)",
            "    :ElevationVarNameNC   elevation",
            "    :RedirectToFile       ../data_obs/GridWeights_Irrigation.txt",
            ":EndGriddedForcing",
            ''
        ]

        return forcing_data

    def create_rvp_file(self, template: bool = False):
        """
        Write Raven .rvp file for HYMOD model.
        
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
            "   FOREST,  25,  6.0, 5.0",
            "   GRAS,    0.6, 2.0, 5.0",
            "   CROP,    2.0, 4.0, 5.0",
            ":EndVegetationClasses"
        ]

        # Create HYMOD-specific parameter structure
        rvp_sections = self._create_rvp_sections(param_or_name, 
                                                land_use_classes, vegetation_classes)
        
        # Write the file
        with open(file_path, 'w') as ff:
            ff.writelines(f"{line}\n" for line in self._create_header("rvp"))
            for section, lines in rvp_sections.items():
                ff.write(f"{section}\n")
                ff.writelines(line + '\n' for line in lines)
                ff.write('\n')

    def _create_rvp_sections(self, param_or_name: str, 
                           land_use_classes: List[str], vegetation_classes: List[str]) -> Dict[str, List[str]]:
        """Create all sections for RVP file."""
        return {
            "#Soil Classes": [
                ":SoilClasses",
                "   :Attributes",
                "   :Units",
                "       TOPSOIL",
                "       GWSOIL",
                ":EndSoilClasses"
            ],
            "#Soil Profiles": [
                "#     name,#horizons,{soiltype,thickness}x{#horizons}",
                "# ",
                ":SoilProfiles",
                "   LAKE, 0,",
                "   ROCK, 0,",
                "   GLACIER, 0,",
                "   MASKED_GLACIER, 0,",
                f"   DEFAULT_P, 2, TOPSOIL, {self.params['HYMOD'][param_or_name]['X02']}, GWSOIL, 10.0",
                ":EndSoilProfiles"
            ],
            "#Land Use Classes": land_use_classes,
            "#Vegetation Classes": vegetation_classes,
            "#Global Parameters": [
                f":GlobalParameter RAINSNOW_TEMP {self.params['HYMOD'][param_or_name]['X03']}"
            ],
            "#Soil Parameters": [
                ":SoilParameterList",
                "   :Parameters, POROSITY, PET_CORRECTION, BASEFLOW_COEFF,",
                "   :Units, -, -, 1 / d,",
                f"       TOPSOIL, 1.0, {self.params['HYMOD'][param_or_name]['X08']}, 0.0,",
                f"       GWSOIL, 1.0, 1.0, {self.params['HYMOD'][param_or_name]['X04']},",
                ":EndSoilParameterList"
            ],
            "#Land Use Parameters": [
                ":LandUseParameterList",
                "   :Parameters, MELT_FACTOR, DD_MELT_TEMP, PDM_B,",
                "   :Units, mm / d / K, degC, -,",
                f"       [DEFAULT], {self.params['HYMOD'][param_or_name]['X05']}, {self.params['HYMOD'][param_or_name]['X06']}, {self.params['HYMOD'][param_or_name]['X07']},",
                ":EndLandUseParameterList"
            ]
        }

    def create_rvi_file(self, template: bool = False):
        """
        Write Raven .rvi file for HYMOD model.
        
        Args:
            template: Whether to create template file
        """
        file_path, param_or_name = self._get_file_path('rvi', template)
        
        # Get HRU groups definition
        hru_groups_definition = self._get_hru_groups_definition()
        
        # Create RVI sections
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
        
        return {
            "#Model Organisation": [
                f":StartDate             {actual_start_date} 00:00:00",  # ✅ Use warm-up date
                f":EndDate               {end_date} 00:00:00",
                ":TimeStep              1.0",
                ":Method                ORDERED_SERIES",
                f":RunName               {self.gauge_id}_HYMOD"
            ],
            "#Model Options": [
                ":SoilModel                  SOIL_MULTILAYER 2",
                ":Routing                    ROUTE_NONE",
                ":CatchmentRoute             GAMMA_UH",
                f":Evaporation                {'PET_OUDIN' if self._pet_method and self._pet_method.upper() == 'OUDIN' else 'PET_OUDIN'}",
                ":RainSnowFraction           RAINSNOW_DINGMAN",
                ":PotentialMeltMethod        POTMELT_DEGREE_DAY",
                ":OroTempCorrect             OROCORR_SIMPLELAPSE",
                ":OroPrecipCorrect           OROCORR_SIMPLELAPSE",
                f":EvaluationPeriod   CALIBRATION   {start_date}   {cali_end_date}",  # ✅ Keep simulation dates
                f":EvaluationPeriod   VALIDATION    {cali_end_date}   {end_date}"
            ],
            "#Soil Layer Alias Definitions": [
                ":Alias       TOPSOIL SOIL[0]",
                ":Alias       GWSOIL  SOIL[1]"
            ],
            "#HRU Groups Definition": [
                hru_groups_definition
            ] if hru_groups_definition else [],
            "#Hydrologic Process Order": [
                ":HydrologicProcesses",
                "   :SnowBalance              SNOBAL_SIMPLE_MELT SNOW            SNOW_LIQ",
                "   :Flush                    RAVEN_DEFAULT      SNOW_LIQ        LAKE_STORAGE",
                "   :Flush                    RAVEN_DEFAULT      LAKE_STORAGE    PONDED_WATER",
                "   :Precipitation            PRECIP_RAVEN       ATMOS_PRECIP    MULTIPLE",
                "   :Infiltration             INF_PDM            PONDED_WATER    MULTIPLE",
                "   :Flush                    RAVEN_DEFAULT      SURFACE_WATER   GWSOIL",
                "           :-->Conditional HRU_TYPE IS_NOT MASKED_GLACIER",
                "           :-->Conditional HRU_TYPE IS_NOT ROCK",
                "           :-->Conditional HRU_TYPE IS_NOT LAKE",
                "   :SoilEvaporation          SOILEVAP_PDM       TOPSOIL         ATMOSPHERE",
                "   :Baseflow                 BASE_LINEAR        GWSOIL          SURFACE_WATER",
                "   :SnowRedistribute         THRESHOLD          SNOW            35000.0",
                ":EndHydrologicProcesses"
            ],
            "#Output Options": [
                        ],
            "#Transport for Snowmelt and Glacier Melt Tracking": [
                "",
                ":Transport SNOWMELT TRACER",
                ":FixedConcentration SNOWMELT ATMOS_PRECIP 0.0 1.0",
                ":FixedConcentration SNOWMELT SLOW_RESERVOIR 0.0",
                "",
                ":Transport GLACIERMELT_ALL TRACER",
                ":FixedConcentration GLACIERMELT_ALL PONDED_WATER 1.0 ALL_GLACIER",
                "",
                ":Transport GLACIERMELT_SMALL TRACER",
                ":FixedConcentration GLACIERMELT_SMALL PONDED_WATER 1.0 SMALL_GLACIER",
                "",
                ":Transport GLACIERMELT_LARGE TRACER",
                ":FixedConcentration GLACIERMELT_LARGE PONDED_WATER 1.0 LARGE_GLACIER"
            ]
        }

    def create_rvc_file(self, template: bool = False):
        """
        Write Raven .rvc file for HYMOD model.
        
        Args:
            template: Whether to create template file
        """
        file_path, param_or_name = self._get_file_path('rvc', template)

        # Define RVC configuration
        rvc_sections = {
            "#Basin Initial Conditions": [
                ":BasinInitialConditions",
                ":Attributes, ID,              Q",
                ":Units,      none,         m3/s",
                "1,             1.0",
                ":EndBasinInitialConditions"
            ]
        }

        # Write the file
        with open(file_path, 'w') as ff:
            ff.writelines(f"{line}\n" for line in self._create_header("rvc"))
            for section, lines in rvc_sections.items():
                ff.write(f"{section}\n")
                ff.writelines(line + '\n' for line in lines)
                ff.write('\n')

    def _get_subbasin_mean_elevation(self, sb_gauge_id: str):
        """Return area-weighted mean elevation for a subbasin from the merged HRU table."""
        try:
            hru_table_path = self.topo_files_dir / 'HRU_table.csv'
            if not hru_table_path.exists():
                return None
            hru_df = pd.read_csv(hru_table_path)
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
            print(f"  Could not compute centroid for subbasin {sb_gauge_id}: {e}")
            return None, None, None

    def create_all_files(self, template: bool = False):
        """
        Create all Raven input files for HYMOD model using namelist configuration.

        Args:
            template: Whether to create template files
        """
        print(f"Creating {'template' if template else 'initialized'} HYMOD files for catchment {self.gauge_id}")
        
        try:
            self.create_rvh_file(template)
            self.create_rvt_file(template)
            self.create_rvp_file(template)
            self.create_rvi_file(template)
            self.create_rvc_file(template)
            
            print(f"Successfully created all {'template' if template else 'initialized'} HYMOD files!")
            
        except Exception as e:
            print(f"Error creating HYMOD files: {e}")
            raise