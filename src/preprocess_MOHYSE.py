"""
MOHYSE Model Preprocessing Module for Raven Hydrological Modeling Framework

This module provides a comprehensive preprocessing class for setting up MOHYSE model
runs in the Raven hydrological modeling framework. It handles the creation of
all necessary Raven input files (.rvh, .rvt, .rvp, .rvi, .rvc) with MOHYSE-specific
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
import numpy as np
import rasterio
from pyproj import Transformer
import preprocess_general

#--------------------------------------------------------------------------------
############################ MOHYSE Preprocessor Class ##########################
#--------------------------------------------------------------------------------

class MOHYSEPreprocessor:
    """
    A comprehensive preprocessor for MOHYSE model setup in Raven.
    
    This class handles all aspects of MOHYSE model preprocessing including:
    - Creation of Raven input files (.rvh, .rvt, .rvp, .rvi, .rvc)
    - HRU management and elevation band grouping
    - Parameter handling for both template and initialized files
    - Forcing data configuration
    """
    
    def __init__(self, namelist_path: Union[str, Path]):
        """
        Initialize the MOHYSE preprocessor with namelist file.
        
        Args:
            namelist_path: Path to the YAML namelist file
        """
        # Load namelist file
        namelist_path = Path(namelist_path)
        if not namelist_path.exists():
            raise FileNotFoundError(f"Namelist file not found: {namelist_path}")
        
        with open(namelist_path, 'r') as f:
            namelist = yaml.safe_load(f)
        
        # Store basic configuration
        self.gauge_id = str(namelist['gauge_id'])
        self.main_dir = Path(namelist['main_dir'])
        self.config_dir = namelist['config_dir']
        self.model_type = namelist['model_type']
        self.start_date = namelist['start_date']
        self.end_date = namelist['end_date']
        self.cali_end_date = namelist['cali_end_date']
        self.coupled = namelist.get('coupled', False)
        self.author = namelist.get('author', 'Justine Berg')
        
        # ✅ LOAD PARAMETERS FROM NAMELIST
        params_path = namelist.get('params_dir', 'config/default_params.yaml')
        
        # Handle relative paths - try multiple locations
        if not Path(params_path).is_absolute():
            # Try relative to namelist file
            candidate1 = Path(namelist_path).parent / params_path
            # Try relative to current working directory
            candidate2 = Path.cwd() / params_path
            # Try relative to src directory (where this file is)
            candidate3 = Path(__file__).parent / params_path
            
            for candidate in [candidate1, candidate2, candidate3]:
                if candidate.exists():
                    params_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"Parameters file not found. Tried:\n"
                    f"  - {candidate1}\n"
                    f"  - {candidate2}\n"
                    f"  - {candidate3}"
                )
        else:
            params_path = Path(params_path)
            if not params_path.exists():
                raise FileNotFoundError(f"Parameters file not found: {params_path}")
        
        with open(params_path, 'r') as f:
            self.params = yaml.safe_load(f)
        
        print(f"✅ Loaded parameters from: {params_path}")
        
        # Remove gauge_info loading - we'll get it from DEM like HBV and HYMOD
        self.gauge_lat = None
        self.gauge_lon = None
        self.station_elevation = None
        
        # Set up directory structure
        self.model_dir = self.main_dir / self.config_dir
        self.catchment_dir = self.model_dir / f'catchment_{self.gauge_id}'
        self.mohyse_dir = self.catchment_dir / self.model_type
        self.templates_dir = self.mohyse_dir / 'templates'
        self.data_obs_dir = self.mohyse_dir / 'data_obs'
        self.topo_files_dir = self.catchment_dir / 'topo_files'
        
        # Ensure directories exist
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.mohyse_dir, self.templates_dir, self.data_obs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

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
            f"# Emulation of MOHYSE simulation of {self.gauge_id}",
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
            file_name = f"{self.gauge_id}_MOHYSE.{file_type}.tpl"
            file_path = self.templates_dir / file_name
        else:
            param_or_name = "init"
            file_name = f"{self.gauge_id}_MOHYSE.{file_type}"
            file_path = self.mohyse_dir / file_name
        
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
                        raise ValueError("No valid elevation data found in DEM (all nodata)")
                    dem_data_valid = dem_data[valid_mask]
                else:
                    valid_mask = ~np.isnan(dem_data) & (dem_data > -9999)
                    if not valid_mask.any():
                        raise ValueError("No valid elevation data found in DEM")
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
                        raise ValueError(f"Could not find pixel with minimum elevation {min_elevation:.1f}m")
                    min_indices = close_indices
                
                # Take the first pixel with minimum elevation
                row_idx = min_indices[0][0]
                col_idx = min_indices[1][0]
                
                # Convert pixel coordinates to geographic coordinates
                lon, lat = dem.xy(row_idx, col_idx)
                
                # Convert to WGS84 if needed
                if dem.crs != 'EPSG:4326':
                    transformer = Transformer.from_crs(dem.crs, 'EPSG:4326', always_xy=True)
                    lon, lat = transformer.transform(lon, lat)
                
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
        Write Raven .rvh file for MOHYSE model.
        
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
            ":RedirectToFile  data_obs/connections.rvh",
            "#:EndLateralConnections",
            ""
        ]
        
        # Create HRU groups
        hru_groups = self._create_hru_groups(HRU)
        
        # Define subbasins
        subbasins = [
            ":SubBasins",
            "  :Attributes,          NAME, DOWNSTREAM_ID,PROFILE,REACH_LENGTH,       GAUGED",
            "  :Units     ,          none,          none,   none,          km,         none",
            f"            1,        {self.gauge_id},            -1,   NONE,       _AUTO,     1",
            ":EndSubBasins"
        ]

        # Define subbasin properties for MOHYSE
        subbasin_properties = [
            ":SubBasinProperties",
            "#                         MOHYSE_PARA_1,                  3,",
            "   :Parameters,           GAMMA_SHAPE,     GAMMA_SCALE,",
            "   :Units,                         -,             1/d,",
            f"              1,          {self.params['MOHYSE'][param_or_name]['X09']},          {self.params['MOHYSE'][param_or_name]['Mohyse_Gamma_Scale']},",
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
                
        print(f"Successfully wrote MOHYSE RVH file to {file_path}")

    def _create_hru_groups(self, hru_df: pd.DataFrame) -> List[str]:
        """Create HRU groups including filtered AllHRUs and elevation bands."""
        hru_groups = []
        
        # Different exclusion rules for different groups
        allhrus_excluded_landuse = ['GLACIER', 'ROCK', 'MASKED_GLACIER', 'LAKE']
        elevation_excluded_landuse = ['GLACIER', 'MASKED_GLACIER']
        
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

        # Add elevation band groups
        hrus_by_band = self.get_hrus_by_elevation_band()
        if hrus_by_band:
            for band, hru_ids in hrus_by_band.items():
                if hru_ids:
                    hru_groups.extend([
                        f":HRUGroup {band}",
                        f"  {' '.join(map(str, hru_ids))}",
                        ":EndHRUGroup",
                        ""
                    ])
        else:
            print(f"No elevation bands available for catchment {self.gauge_id}")

        return hru_groups

    def create_rvt_file(self, template: bool = False):
        """
        Write Raven .rvt file for MOHYSE model.
        
        Args:
            template: Whether to create template file
        """
        file_path, param_or_name = self._get_file_path('rvt', template)

        print(f"Extracting gauge location from DEM for gauge {self.gauge_id}...")
        try:
            gauge_lat, gauge_lon, station_elevation = self.get_gauge_location_from_dem()
        except Exception as e:
            print(f"Error: Could not extract gauge location from DEM: {e}")
            return

        # Create gauge info
        gauge_info = self._create_gauge_info(gauge_lat, gauge_lon, station_elevation)
        
        # Create forcing data (no coupled parameter needed - always ERA5-Land)
        forcing_data = self._create_forcing_block()

        # Write the file
        with open(file_path, 'w') as ff:
            ff.writelines(f"{line}\n" for line in self._create_header("rvt"))
            ff.write("# meteorological forcings\n")
            for f in forcing_data.values():
                for t in f:
                    ff.write(f"{t}\n")
            ff.writelines(gauge_info)
            ff.write(f":RedirectToFile data_obs/Q_daily.rvt\n")

    def _create_gauge_info(self, gauge_lat: float, gauge_lon: float, 
                        station_elevation: float) -> List[str]:
        """Create gauge information section for RVT file."""
        gauge_info = [
            f":Gauge {self.gauge_id}\n",
            f"  :Latitude    {gauge_lat}\n",
            f"  :Longitude {gauge_lon}\n",
            f"  :Elevation  {station_elevation}\n\n"
        ]
        
        # Add monthly data if available
        monthly_data = self._get_monthly_data()
        if monthly_data:
            gauge_info.extend(monthly_data)
        
        gauge_info.extend([
            f":EndGauge\n\n"
        ])
        
        return gauge_info

    def _get_monthly_data(self) -> List[str]:
        """Read and format monthly temperature and PET data if available."""
        monthly_temp_file = self.data_obs_dir / 'monthly_temperature_averages.csv'
        monthly_pet_file = self.data_obs_dir / 'monthly_pet_averages.csv'
        
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

    def _create_forcing_block(self) -> Dict[str, List[str]]:
        """Create forcing data configuration for RVT file - ERA5-Land version."""
        grid_weights_file_path = "data_obs/GridWeights.txt"
        
        # ERA5-Land variable names (always the same, no coupled option)
        var_names = {
            'rainfall': 'tp',
            'temp_ave': 't2m',
            'temp_max': 't2m', 
            'temp_min': 't2m'
        }
        dim_names = "longitude latitude time"

        # ERA5-Land file names
        forcing_types = [
            ('Rainfall', 'RAINFALL', 'era5_land_precip.nc', var_names['rainfall']),
            ('Average Temperature', 'TEMP_AVE', 'era5_land_temp_mean.nc', var_names['temp_ave']),
            ('Maximum Temperature', 'TEMP_MAX', 'era5_land_temp_max.nc', var_names['temp_max']),
            ('Minimum Temperature', 'TEMP_MIN', 'era5_land_temp_min.nc', var_names['temp_min'])
        ]

        forcing_data = {}
        for name, forcing_type, filename, var_name in forcing_types:
            forcing_data[name] = [
                f":GriddedForcing           {name}",
                f"    :ForcingType          {forcing_type}",
                f"    :FileNameNC           data_obs/{filename}",
                f"    :VarNameNC            {var_name}",
                f"    :DimNamesNC           {dim_names}",
                "    :ElevationVarNameNC   elevation",
                f"    :RedirectToFile       {grid_weights_file_path}",
                ":EndGriddedForcing",
                ''
            ]
        
        # Add Irrigation forcing block
        forcing_data['Irrigation'] = [
            ":GriddedForcing           Irrigation",
            "    :ForcingType          IRRIGATION",
            "    :FileNameNC           data_obs/irrigation.nc",
            "    :VarNameNC            data",
            "    :DimNamesNC           x y time     # must be in the order of (x,y,t)",
            "    :ElevationVarNameNC   elevation",
            "    :RedirectToFile       data_obs/GridWeights_Irrigation.txt",
            ":EndGriddedForcing",
            ''
        ]

        return forcing_data

    def create_rvp_file(self, template: bool = False):
        """
        Write Raven .rvp file for MOHYSE model.
        
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

        # Create MOHYSE-specific parameter structure
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
                "   :Attributes,",
                "   :Units,",
                "       TOPSOIL",
                "       GWSOIL",
                ":EndSoilClasses"
            ],
            "#Soil Profiles": [
                "#  name,#horizons,{soiltype,thickness}x{#horizons}",
                "# ",
                ":SoilProfiles",
                "   LAKE, 0",
                "   ROCK, 0",
                "   GLACIER, 0",
                "   MASKED_GLACIER, 0",
                f"   DEFAULT_P,      2, TOPSOIL,     {self.params['MOHYSE'][param_or_name]['X05']}, GWSOIL, 10.0",
                ":EndSoilProfiles"
            ],
            "#Vegetation Classes": vegetation_classes,
            "#Vegetation Parameters": [
                ":VegetationParameterList",
                "   :Parameters,    SAI_HT_RATIO,  RAIN_ICEPT_PCT,  SNOW_ICEPT_PCT,",
                "   :Units,               -,               -,               -, ",
                "       [DEFAULT],             0.0,             0.0,             0.0,   ",
                ":EndVegetationParameterList"
            ],
            "#Land Use Classes": land_use_classes,
            "#Global Parameters": [
                "#:GlobalParameter      RAINSNOW_TEMP              -2.0",
                ":GlobalParameter       TOC_MULTIPLIER              1.0",
                f":GlobalParameter       MOHYSE_PET_COEFF         {self.params['MOHYSE'][param_or_name]['X01']}"
            ],
            "#Land Use Parameters": [
                ":LandUseParameterList",
                "   :Parameters,     MELT_FACTOR,       AET_COEFF, FOREST_SPARSENESS, DD_MELT_TEMP,",
                "   :Units,          mm/d/K,            mm/d,                 -,         degC,",
                "#      [DEFAULT],   MOHYSE_PARA_3,   MOHYSE_PARA_2,               0.0,MOHYSE_PARA_4, ",
                f"      [DEFAULT],          {self.params['MOHYSE'][param_or_name]['X03']},          {self.params['MOHYSE'][param_or_name]['X02']},               0.0,       {self.params['MOHYSE'][param_or_name]['X04']},",
                ":EndLandUseParameterList"
            ],
            "#Soil Parameters": [
                ":SoilParameterList",
                "   :Parameters,        POROSITY,  PET_CORRECTION,        HBV_BETA,  BASEFLOW_COEFF,      PERC_COEFF, ",
                "   :Units,               -,               -,               -,             1/d,             1/d, ",
                "#      TOPSOIL,            1.0 ,             1.0,             1.0,   TOPSOIL_BASEFLOW_COEFF,   TOPSOIL_PERC_COEFF,",
                "#      GWSOIL,            1.0 ,             1.0,             1.0,   GWSOIL_BASEFLOW_COEFF,             0.0,",
                f"      TOPSOIL,            1.0 ,             1.0,             1.0,          {self.params['MOHYSE'][param_or_name]['X07']},          {self.params['MOHYSE'][param_or_name]['X06']},",
                f"      GWSOIL,            1.0 ,             1.0,             1.0,          {self.params['MOHYSE'][param_or_name]['X08']},             0.0,",
                ":EndSoilParameterList"
            ]
        }

    def create_rvi_file(self, template: bool = False):
        """
        Write Raven .rvi file for MOHYSE model.
        
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
        
        if hrus_by_band:
            elevation_bands = sorted(hrus_by_band.keys(), key=lambda x: int(x.split('-')[0]))
            return ":DefineHRUGroups AllHRUs " + " ".join(elevation_bands)
        else:
            print(f"No elevation bands available for catchment {self.gauge_id}, only defining AllHRUs group")
            return ":DefineHRUGroups AllHRUs"

    def _create_rvi_sections(self, start_date: str, end_date: str, cali_end_date: str,
                           hru_groups_definition: str) -> Dict[str, List[str]]:
        """Create all sections for RVI file."""
        return {
            "#Model Organisation": [
                f":StartDate             {start_date} 00:00:00",
                f":EndDate               {end_date} 00:00:00",
                ":TimeStep              1.0",
                ":Method                ORDERED_SERIES",
                f":RunName               {self.gauge_id}_MOHYSE"
            ],
            "#Model Options": [
                ":SoilModel                  SOIL_MULTILAYER 2",
                ":Routing                    ROUTE_NONE",
                ":CatchmentRoute             TRIANGULAR_UH",
                ":Evaporation                PET_MOHYSE",
                ":RainSnowFraction           RAINSNOW_DINGMAN",
                ":PotentialMeltMethod        POTMELT_DEGREE_DAY",
                ":OroTempCorrect             OROCORR_SIMPLELAPSE",
                ":OroPrecipCorrect           OROCORR_SIMPLELAPSE",
                f":EvaluationPeriod   CALIBRATION   {start_date}   {cali_end_date}",
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
                "   :SnowBalance              SNOBAL_SIMPLE_MELT   SNOW            SNOW_LIQ",
                "   :Flush                    RAVEN_DEFAULT        SNOW_LIQ        LAKE_STORAGE",
                "   :Flush                    RAVEN_DEFAULT        LAKE_STORAGE    PONDED_WATER",
                "   :Precipitation            PRECIP_RAVEN         ATMOS_PRECIP    MULTIPLE",
                "   :Infiltration             INF_HBV              PONDED_WATER    MULTIPLE",
                "   :SoilEvaporation          SOILEVAP_LINEAR      TOPSOIL         ATMOSPHERE",
                "   :Percolation              PERC_LINEAR          TOPSOIL         GWSOIL",
                "   :Baseflow                 BASE_LINEAR          TOPSOIL         SURFACE_WATER",
                "   :Baseflow                 BASE_LINEAR          GWSOIL          SURFACE_WATER",
                "   :SnowRedistribute         THRESHOLD            SNOW            35000.0",
                ":EndHydrologicProcesses"
            ],
            "#Output Options": [
                "  :EvaluationMetrics RMSE KLING_GUPTA NASH_SUTCLIFFE ",
                "  :CustomOutput DAILY AVERAGE SNOW BY_HRU_GROUP",
            ]
        }
    
    def create_rvc_file(self, template: bool = False):
        """
        Write Raven .rvc file for MOHYSE model.
        
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

    def create_all_files(self, template: bool = False):
        """
        Create all Raven input files for MOHYSE model using namelist configuration.
        
        Args:
            template: Whether to create template files
        """
        print(f"Creating {'template' if template else 'initialized'} MOHYSE files for catchment {self.gauge_id}")
        
        try:
            self.create_rvh_file(template)
            self.create_rvt_file(template)
            self.create_rvp_file(template)
            self.create_rvi_file(template)
            self.create_rvc_file(template)
            
            print(f"Successfully created all {'template' if template else 'initialized'} MOHYSE files!")
            
        except Exception as e:
            print(f"Error creating MOHYSE files: {e}")
            raise