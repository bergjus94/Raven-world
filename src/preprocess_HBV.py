#--------------------------------------------------------------------------------
#################################### packages ###################################
#--------------------------------------------------------------------------------

import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import date
import numpy as np
import rasterio
import pyproj
import yaml
import logging
from typing import List  

#--------------------------------------------------------------------------------
############################### HBV Processor Class ############################
#--------------------------------------------------------------------------------

class HBVProcessor:
    """
    A class to process and generate HBV model input files for Raven hydrological modeling.
    
    This class handles the creation of all necessary HBV input files including:
    - .rvh (HRU and basin definitions)
    - .rvi (model configuration and process definitions)
    - .rvp (parameters and soil/vegetation properties)
    - .rvc (initial conditions)
    - .rvt (time series and forcing data)
    """
    
    def __init__(self, namelist_path: str, author: str = 'Justine Berg'):
        """
        Initialize the HBVProcessor with configuration from namelist.
        
        Parameters
        ----------
        namelist_path : str
            Path to the YAML configuration file
        author : str, optional
            Author name for file headers, by default 'Justine Berg'
        """
        self.namelist_path = namelist_path
        self.author = author
        
        # Load configuration
        with open(namelist_path, "r") as f:
            self.nml = yaml.safe_load(f)
        
        # Extract configuration
        self.gauge_id = self.nml["gauge_id"]
        self.coupled = self.nml["coupled"]
        self.main_dir = self.nml["main_dir"]
        self.model_dirs = self.nml["model_dirs"]
        self.model_type = "HBV"  # Fixed to HBV
        self.params_dir = self.nml["params_dir"]
        self.start_date = self.nml["start_date"]
        self.end_date = self.nml["end_date"]
        self.cali_end_date = self.nml["cali_end_date"]
        
        # Set model directory
        if self.coupled:
            self.model_dir = Path(self.main_dir, self.model_dirs["coupled"])
        else:
            self.model_dir = Path(self.main_dir, self.model_dirs["uncoupled"])
        
        # Load parameters
        with open(self.params_dir, "r") as f:
            self.params = yaml.load(f, Loader=yaml.FullLoader)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"HBVProcessor initialized for gauge {self.gauge_id}")
        self.logger.info(f"Model directory: {self.model_dir}")
        self.logger.info(f"Coupled mode: {self.coupled}")

    def create_header(self, rvx_type: str = "rvi") -> list:
        """
        Creates header info for .rvX files
        
        Parameters
        ----------
        rvx_type : str
            .rvX file type (rvi, rvp, rvh, rvc, rvt)
            
        Returns
        -------
        list
            List with header info
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
        
        header = [header_line, file_type, author_line, creation_date_line, *description]
        return header

    def get_hrus_by_elevation_band(self) -> dict:
        """
        Load HRU shapefile and create a list of HRU IDs for each elevation band.
        
        Returns
        -------
        dict
            Dictionary with elevation bands as keys and lists of HRU IDs as values
        """
        # Load HRU shapefile
        hru_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU.shp')
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

    def get_gauge_location_from_dem(self) -> tuple:
        """
        Extract gauge location (lat, lon, elevation) from the lowest point in the DEM
        
        Returns
        -------
        tuple
            (latitude, longitude, elevation) of the lowest point in the DEM
        """
        # Path to clipped DEM
        dem_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'clipped_dem.tif')
        
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {dem_path}")
        
        try:
            with rasterio.open(dem_path) as dem:
                # Read DEM data
                dem_data = dem.read(1)  # Read first band
                
                self.logger.info(f"DEM shape: {dem_data.shape}")
                self.logger.info(f"DEM data type: {dem_data.dtype}")
                self.logger.info(f"DEM nodata value: {dem.nodata}")
                self.logger.info(f"DEM bounds: {dem.bounds}")
                self.logger.info(f"DEM CRS: {dem.crs}")
                
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
                self.logger.info(f"Minimum elevation: {min_elevation:.1f} m")
                
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
                
                self.logger.info(f"Gauge location: Lat={lat:.6f}, Lon={lon:.6f}, Elev={min_elevation:.1f}m")
                
                return lat, lon, min_elevation
                
        except Exception as e:
            self.logger.error(f"Error reading DEM file: {e}")
            raise

    def _create_hru_groups(self, hru_df: pd.DataFrame) -> List[str]:
        """Create HRU groups including filtered AllHRUs and elevation bands."""
        hru_groups = []
        
        # Filter out unwanted land use classes for AllHRUs group
        excluded_landuse = ['GLACIER', 'ROCK', 'MASKED_GLACIER', 'LAKE']
        
        # ✅ FIXED: Use correct column name from HRU table
        # Get all HRU IDs excluding the specified land use classes
        filtered_hrus = hru_df[~hru_df['LAND_USE_CLASS'].isin(excluded_landuse)]
        filtered_hru_ids = filtered_hrus[':ATTRIBUTES'].tolist()  # Changed from ':ATTRIBUTES' to 'ID'
        
        print(f"Total HRUs: {len(hru_df)}")
        print(f"Excluded HRUs with land use {excluded_landuse}: {len(hru_df) - len(filtered_hrus)}")
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
            # Still create an empty AllHRUs group to avoid errors
            hru_groups.extend([
                ":HRUGroup AllHRUs",
                "  # No HRUs available after filtering",
                ":EndHRUGroup",
                ""
            ])

        # Add elevation band groups if available (also filtered)
        hrus_by_band = self.get_hrus_by_elevation_band()
        if hrus_by_band:
            for band, hru_ids in hrus_by_band.items():
                if hru_ids:
                    # Double-check that elevation band HRUs are also filtered
                    # (get_hrus_by_elevation_band already filters out landuse 7 and 8,
                    # but let's be explicit about all excluded classes)
                    band_hru_df = hru_df[hru_df[':ATTRIBUTES'].isin(hru_ids)]  # Changed from ':ATTRIBUTES' to 'ID'
                    filtered_band_hrus = band_hru_df[~band_hru_df['LAND_USE_CLASS'].isin(excluded_landuse)]
                    filtered_band_hru_ids = filtered_band_hrus[':ATTRIBUTES'].tolist()  # Changed from ':ATTRIBUTES' to 'ID'
                    
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

    def _get_hru_groups_definition(self) -> str:
        """Get HRU groups definition string."""
        hrus_by_band = self.get_hrus_by_elevation_band()
        
        if hrus_by_band:
            elevation_bands = sorted(hrus_by_band.keys(), key=lambda x: int(x.split('-')[0]))
            return ":DefineHRUGroups AllHRUs " + " ".join(elevation_bands)
        else:
            print(f"No elevation bands available for catchment {self.gauge_id}, only defining AllHRUs group")
            return ":DefineHRUGroups AllHRUs"

    def create_forcing_block(self) -> dict:
        """
        Create Dictionary of forcing data to write in RVT file for HBV.
        
        Returns
        -------
        dict
            Dictionary containing forcing data configuration
        """
        grid_weights_file_path = "data_obs/GridWeights.txt"
        
        if self.coupled:
            # Use transformed forcing files with 'data' variable names
            file_suffix = "_coupled.nc"
            var_names = {
                'rainfall': 'data',
                'temp_ave': 'data', 
                'temp_max': 'data',
                'temp_min': 'data'
            }
            dim_names = "x y time"
        else:
            # Use standard ERA5-Land forcing files with original variable names
            file_suffix = ".nc"
            var_names = {
                'rainfall': 'tp',
                'temp_ave': 't2m',
                'temp_max': 't2m', 
                'temp_min': 't2m'
            }
            dim_names = "longitude latitude time"

        forcing_types = [
            ('Rainfall', 'RAINFALL', f'era5_land_precip{file_suffix}', var_names['rainfall']),
            ('Average Temperature', 'TEMP_AVE', f'era5_land_temp_mean{file_suffix}', var_names['temp_ave']),
            ('Maximum Temperature', 'TEMP_MAX', f'era5_land_temp_max{file_suffix}', var_names['temp_max']),
            ('Minimum Temperature', 'TEMP_MIN', f'era5_land_temp_min{file_suffix}', var_names['temp_min'])
        ]

        forcing_data = {}
        for name, forcing_type, filename, var_name in forcing_types:
            forcing_data[name] = [
                f":GriddedForcing           {name}",
                f"    :ForcingType          {forcing_type}",
                f"    :FileNameNC           data_obs/{filename}",
                f"    :VarNameNC            {var_name}",
                f"    :DimNamesNC           {dim_names}     # must be in the order of (x,y,t)",
                "    :ElevationVarNameNC   elevation",
                f"    :RedirectToFile       {grid_weights_file_path}",
                ":EndGriddedForcing",
                ''
            ]

        return forcing_data

    def write_rvh_file(self, template: bool = False) -> None:
        """
        Write Raven .rvh file by reading the HRU_table.csv file.
        
        Parameters
        ----------
        template : bool, optional
            Whether to create a template file, by default False
        """
        # Determine file paths
        if not template:
            file_name = f"{self.gauge_id}_HBV.rvh"
            file_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'HBV', file_name)
        else:
            file_name = f"{self.gauge_id}_HBV.rvh.tpl"
            file_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'HBV', 'templates', file_name)

        # Read HRU table from CSV file
        hru_table_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'topo_files', 'HRU_table.csv')
        
        try:
            HRU = pd.read_csv(hru_table_path)
            self.logger.info(f"Successfully read HRU table from {hru_table_path}")
        except Exception as e:
            self.logger.error(f"Error reading HRU table from {hru_table_path}: {e}")
            return

        # Convert the HRU table to a formatted string for the RVH file
        x = HRU.to_string(header=False, index=False, index_names=False).split('\n')
        vals = [',\t'.join(ele.split()) for ele in x]

        # HRUs section
        hru_list_1 = [
            ":HRUs",
            "  :Attributes, ID,  AREA, ELEVATION, LATITUDE, LONGITUDE, BASIN_ID,LAND_USE_CLASS, VEG_CLASS, SOIL_PROFILE, AQUIFER_PROFILE, TERRAIN_CLASS, SLOPE, ASPECT",
            "  :Units     , none,   km2,         m,      deg,       deg,     none,          none,      none,         none,            none,          none,   deg,    deg",
        ]
        hru_list_2 = [":EndHRUs"]
        hru_list = [*hru_list_1, *vals, *hru_list_2]
        
        # Lateral connections block
        lateral_connections = [
            "",
            "#:LateralConnections",
            ":RedirectToFile  data_obs/connections.rvh",
            "#:EndLateralConnections",
            ""
        ]
        
        # ✅ UPDATED: Use the new HRU groups creation method
        hru_groups = self._create_hru_groups(HRU)  # Pass the HRU DataFrame

        # Subbasins section
        subbasins = [
            ":SubBasins",
            "  :Attributes,          NAME, DOWNSTREAM_ID,PROFILE,REACH_LENGTH,       GAUGED",
            "  :Units     ,          none,          none,   none,          km,         none",
            f"            1,        {self.gauge_id},            -1,   NONE,       _AUTO,     1",
            ":EndSubBasins"
        ]

        # HBV-specific subbasin properties
        param_or_name = "names" if template else "init"
        subbasin_properties = [
            ":SubBasinProperties",
            "#                       HBV_PARA_11, DERIVED FROM HBV_PARA_11,",
            "#                            MAXBAS,                 MAXBAS/2,",
            "   :Parameters,           TIME_CONC,             TIME_TO_PEAK,",
            "   :Units,                        d,                        d,",
            f"              1,          {self.params['HBV'][param_or_name]['HBV_Param_11']},                  {self.params['HBV'][param_or_name]['HBV_Param_11b']},",
            ":EndSubBasinProperties"
        ]

        # Write the file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as ff:
            ff.writelines(f"{line}\n" for line in self.create_header(rvx_type="rvh"))
            ff.writelines(f"{line}\n" for line in subbasins)
            ff.writelines(f"{line}\n" for line in subbasin_properties)
            ff.writelines(f"{line}\n" for line in hru_list)
            ff.writelines(f"{line}\n" for line in lateral_connections)
            ff.write("\n# HRU Groups (AllHRUs + Elevation Bands)\n")
            ff.writelines(f"{line}\n" for line in hru_groups)  # Use new method
            
        self.logger.info(f"Successfully wrote RVH file to {file_path}")

    def write_rvi_file(self, template: bool = False) -> None:
        """
        Write Raven .rvi file for HBV model.
        
        Parameters
        ----------
        template : bool, optional
            Whether to create a template file, by default False
        """
        # Determine file paths
        if not template:
            file_name = f"{self.gauge_id}_HBV.rvi"
            file_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'HBV', file_name)
        else:
            file_name = f"{self.gauge_id}_HBV.rvi.tpl"
            file_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'HBV', 'templates', file_name)

        # ✅ UPDATED: Use the new HRU groups definition method
        hru_groups_definition = self._get_hru_groups_definition()

        # HBV-specific RVI structure
        rvi_content = {
            "#Model Organisation": [
                f":StartDate             {self.start_date} 00:00:00",
                f":EndDate               {self.end_date} 00:00:00",
                ":TimeStep              1.0",
                f":RunName               {self.gauge_id}_HBV"
            ],
            "#Model Options": [
                ":Routing             	    ROUTE_NONE",
                ":CatchmentRoute      	    TRIANGULAR_UH",
                ":Evaporation         	    PET_FROMMONTHLY",
                ":OW_Evaporation      	    PET_FROMMONTHLY",
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
                ":SoilModel                  SOIL_MULTILAYER 3",
                f":EvaluationPeriod   CALIBRATION   {self.start_date}   {self.cali_end_date}",
                f":EvaluationPeriod   VALIDATION    {self.cali_end_date}   {self.end_date}"
            ],
            "#Soil Alias Layer Definitions": [
                ":Alias       FAST_RESERVOIR SOIL[1]",
                ":Alias       SLOW_RESERVOIR SOIL[2]",
                ":LakeStorage SLOW_RESERVOIR"
            ],
            "#HRU Groups Definition": [hru_groups_definition] if hru_groups_definition else [],
            "#Hydrologic Process Order": [
                ":HydrologicProcesses",
                "   :Flush             RAVEN_DEFAULT      SNOW            ATMOSPHERE",
                "       :-->Conditional HRU_TYPE IS MASKED_GLACIER",
                "   :SnowRefreeze      FREEZE_DEGREE_DAY  SNOW_LIQ        SNOW",
                "   :Precipitation     PRECIP_RAVEN       ATMOS_PRECIP    MULTIPLE",
                "   :CanopyEvaporation CANEVP_ALL         CANOPY          ATMOSPHERE",
                "   :CanopySnowEvap    CANEVP_ALL         CANOPY_SNOW     ATMOSPHERE",
                "   :SnowBalance       SNOBAL_SIMPLE_MELT SNOW            SNOW_LIQ",
                "       :-->Overflow     RAVEN_DEFAULT      SNOW_LIQ        LAKE_STORAGE",
                "   :Flush             RAVEN_DEFAULT      LAKE_STORAGE    PONDED_WATER",
                "   :Flush             RAVEN_DEFAULT      PONDED_WATER    GLACIER",
                "       :-->Conditional HRU_TYPE IS GLACIER",
                "   :GlacierMelt       GMELT_HBV          GLACIER_ICE     GLACIER",
                "   :GlacierRelease    GRELEASE_HBV_EC    GLACIER         SURFACE_WATER",
                "   :Infiltration      INF_HBV            PONDED_WATER    MULTIPLE",
                "   :Flush             RAVEN_DEFAULT      SURFACE_WATER   FAST_RESERVOIR",
                "       :-->Conditional HRU_TYPE IS_NOT GLACIER",
                "       :-->Conditional HRU_TYPE IS_NOT ROCK",
                "       :-->Conditional HRU_TYPE IS_NOT MASKED_GLACIER",
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
                "  :EvaluationMetrics RMSE KLING_GUPTA NASH_SUTCLIFFE ",
                "  :CustomOutput DAILY AVERAGE SNOW BY_HRU_GROUP",
            ]
        }

        # Write the file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as ff:
            ff.writelines(f"{line}\n" for line in self.create_header(rvx_type="rvi"))
            for section, lines in rvi_content.items():
                ff.write(f"{section}\n")
                ff.writelines(line + '\n' for line in lines)
                ff.write('\n')

        self.logger.info(f"Successfully wrote RVI file to {file_path}")

    def write_rvp_file(self, template: bool = False) -> None:
        """
        Write Raven .rvp file for HBV model.
        
        Parameters
        ----------
        template : bool, optional
            Whether to create a template file, by default False
        """
        # Determine file paths and parameter type
        if not template:
            param_or_name = "init"
            file_name = f"{self.gauge_id}_HBV.rvp"
            file_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'HBV', file_name)
        else:
            param_or_name = "names"
            file_name = f"{self.gauge_id}_HBV.rvp.tpl"
            file_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'HBV', 'templates', file_name)

        # Common class definitions
        land_use_classes = [
            ":LandUseClasses",
            "   :Attributes, IMPERM, FOREST_COV",
            "   :Units, frac, frac",
            "   FOREST,    0.05, 1.0",
            "   OPEN,      0.0,  0.1",
            "   GLACIER,   0.0,  0.0",
            "   LAKE,      0.0,  0.0",
            "   ROCK,      0.1, 0.0",
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

        # HBV-specific parameter structure
        rvp_content = {
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
                f"   DEFAULT_P,      3,    TOPSOIL,            {self.params['HBV'][param_or_name]['HBV_Param_17']},   FAST_RES,    100.0, SLOW_RES,    100.0",
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
            "#Global Parameters": [
                f":GlobalParameter RAINSNOW_TEMP       {self.params['HBV'][param_or_name]['HBV_Param_01']}",
                ":GlobalParameter RAINSNOW_DELTA      1.0 #constant",
                f":GlobalParameter PRECIP_LAPSE       2.0 # HBV_PARA_12=PCALT",
                f":GlobalParameter ADIABATIC_LAPSE    6.0 # HBV_PARA_13=TCALT",
                f":GlobalParameter SNOW_SWI  {self.params['HBV'][param_or_name]['HBV_Param_04']} #HBV_PARA_04"
            ],
            "#Land Use Parameters": [
                ":LandUseParameterList",
                "  :Parameters,   MELT_FACTOR, MIN_MELT_FACTOR,   HBV_MELT_FOR_CORR, REFREEZE_FACTOR, HBV_MELT_ASP_CORR",
                "  :Units     ,        mm/d/K,          mm/d/K,                none,          mm/d/K,              none",
                "  #              HBV_PARA_02,        CONSTANT,         HBV_PARA_18,     HBV_PARA_03,          CONSTANT",
                f"    [DEFAULT],  {self.params['HBV'][param_or_name]['HBV_Param_02']},             2.2,        {self.params['HBV'][param_or_name]['HBV_Param_18']},    {self.params['HBV'][param_or_name]['HBV_Param_03']},              0.48",
                ":EndLandUseParameterList",
                "",
                ":LandUseParameterList",
                " :Parameters, HBV_MELT_GLACIER_CORR,   HBV_GLACIER_KMIN, GLAC_STORAGE_COEFF, HBV_GLACIER_AG",
                " :Units     ,                  none,                1/d,                1/d,           1/mm",
                "   #                       CONSTANT,           CONSTANT,        HBV_PARA_19,       CONSTANT,",
                f"   [DEFAULT],                  1.64,               0.05,       {self.params['HBV'][param_or_name]['HBV_Param_19']},           0.05",
                ":EndLandUseParameterList"
            ],
            "#Soil Parameters": [
                f"#For Ostrich:HBV_Alpha= {self.params['HBV'][param_or_name]['HBV_Param_15']}",
                ":SoilParameterList",
                "  :Parameters,                POROSITY,FIELD_CAPACITY,     SAT_WILT,     HBV_BETA, MAX_CAP_RISE_RATE,  MAX_PERC_RATE,  BASEFLOW_COEFF,            BASEFLOW_N",
                "  :Units     ,                    none,          none,         none,         none,              mm/d,           mm/d,             1/d,                  none",
                "  #                        HBV_PARA_05,   HBV_PARA_06,  HBV_PARA_14,  HBV_PARA_07,       HBV_PARA_16,       CONSTANT,        CONSTANT,              CONSTANT,",
                f"    [DEFAULT],            {self.params['HBV'][param_or_name]['HBV_Param_05']},  {self.params['HBV'][param_or_name]['HBV_Param_06']}, {self.params['HBV'][param_or_name]['HBV_Param_14']}, {self.params['HBV'][param_or_name]['HBV_Param_07']},      {self.params['HBV'][param_or_name]['HBV_Param_16']},            0.0,             0.0,                   0.0",
                "  #                                                        CONSTANT,                                     HBV_PARA_08,     HBV_PARA_09, 1+HBV_PARA_15=1+ALPHA,",
                f"     FAST_RES,                _DEFAULT,      _DEFAULT,          0.0,     _DEFAULT,          _DEFAULT,   {self.params['HBV'][param_or_name]['HBV_Param_08']},    {self.params['HBV'][param_or_name]['HBV_Param_09']},              {self.params['HBV'][param_or_name]['HBV_Param_15']}",
                "  #                                                        CONSTANT,                                                      HBV_PARA_10,              CONSTANT,",
                f"     SLOW_RES,                _DEFAULT,      _DEFAULT,          0.0,     _DEFAULT,          _DEFAULT,       _DEFAULT,    {self.params['HBV'][param_or_name]['HBV_Param_10']},                   1.0",
                ":EndSoilParameterList"
            ]
        }

        # Write the file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as ff:
            ff.writelines(f"{line}\n" for line in self.create_header(rvx_type="rvp"))
            for section, lines in rvp_content.items():
                ff.write(f"{section}\n")
                ff.writelines(line + '\n' for line in lines)
                ff.write('\n')

        self.logger.info(f"Successfully wrote RVP file to {file_path}")

    def write_rvc_file(self, template: bool = False) -> None:
        """
        Write Raven .rvc file for HBV model.
        
        Parameters
        ----------
        template : bool, optional
            Whether to create a template file, by default False
        """
        # Determine file paths and parameter type
        if not template:
            param_or_name = "init"
            file_name = f"{self.gauge_id}_HBV.rvc"
            file_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'HBV', file_name)
        else:
            param_or_name = "names"
            file_name = f"{self.gauge_id}_HBV.rvc.tpl"
            file_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'HBV', 'templates', file_name)

        # HBV-specific initial conditions
        rvc_content = {
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
                "# derived from thickness: HBV_PARA_17 [m] * 1000.0 / 2.0",
                f"{self.params['HBV'][param_or_name]['HBV_Param_17b']}",
                ":EndInitialConditions"
            ]
        }

        # Write the file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as ff:
            ff.writelines(f"{line}\n" for line in self.create_header(rvx_type="rvc"))
            for section, lines in rvc_content.items():
                ff.write(f"{section}\n")
                ff.writelines(line + '\n' for line in lines)
                ff.write('\n')

        self.logger.info(f"Successfully wrote RVC file to {file_path}")

    def write_rvt_file(self, template: bool = False) -> None:
        """
        Write Raven .rvt file for HBV model.
        
        Parameters
        ----------
        template : bool, optional
            Whether to create a template file, by default False
        """
        # Determine file paths and parameter type
        if not template:
            param_or_name = "init"
            file_name = f"{self.gauge_id}_HBV.rvt"
            file_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'HBV', file_name)
        else:
            param_or_name = "names"
            file_name = f"{self.gauge_id}_HBV.rvt.tpl"
            file_path = Path(self.model_dir, f'catchment_{self.gauge_id}', 'HBV', 'templates', file_name)

        # Extract gauge location from DEM
        self.logger.info(f"Extracting gauge location from DEM for gauge {self.gauge_id}...")
        try:
            gauge_lat, gauge_lon, station_elevation = self.get_gauge_location_from_dem()
        except Exception as e:
            self.logger.error(f"Error: Could not extract gauge location from DEM: {e}")
            return

        # Gauge header and info
        gauge_header = f":Gauge {self.gauge_id}\n"
        gauge_end = f":EndGauge\n\n"
        gauge_info = [
            f"  :Latitude    {gauge_lat}\n",
            f"  :Longitude {gauge_lon}\n",
            f"  :Elevation  {station_elevation}\n\n",
            ' '
        ]

        # Try to read monthly temperature and evaporation data if available
        monthly_temp_file = Path(self.model_dir, f'catchment_{self.gauge_id}', 'HBV', 'data_obs', 'monthly_temperature_averages.csv')
        monthly_pet_file = Path(self.model_dir, f'catchment_{self.gauge_id}', 'HBV', 'data_obs', 'monthly_pet_averages.csv')
        
        monthly_data = []
        
        if monthly_temp_file.exists() and monthly_pet_file.exists():
            try:
                # Read monthly temperature data
                temp_df = pd.read_csv(monthly_temp_file)
                # Read monthly PET data
                pet_df = pd.read_csv(monthly_pet_file)
                
                # Format temperature values
                temp_values = temp_df['Temperature'].values
                temp_str = ", ".join([f"{val:.1f}" for val in temp_values])
                
                # Format PET values (convert from mm/day to mm/hour)
                pet_values = pet_df['PET_avg_mm_per_day'].values
                pet_str = ", ".join([f"{val:.3f}" for val in pet_values])
                
                # Create monthly data block
                monthly_data = [
                    "#                       Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec \n",
                    f"  :MonthlyAveEvaporation, {pet_str} \n",
                    f"  :MonthlyAveTemperature, {temp_str} \n"
                ]
            except Exception as e:
                self.logger.warning(f"Error reading monthly data: {e}")
                monthly_data = []

        # HBV-specific gauge corrections
        gauge_correction = [
            f"#  :RainCorrection    {self.params['HBV'][param_or_name]['HBV_Param_20']} \n",
            f"#  :SnowCorrection    {self.params['HBV'][param_or_name]['HBV_Param_21']}\n \n"
        ]

        # Complete gauge section
        gauge = [
            gauge_header,
            *gauge_info,
            *gauge_correction,
            *monthly_data,  # Add monthly data if available
            gauge_end
        ]

        # Discharge info
        flow_observation = [
            "# observed streamflow\n",
            f":RedirectToFile data_obs/Q_daily.rvt"
        ]

        # Write the file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as ff:
            ff.writelines(f"{line}\n" for line in self.create_header(rvx_type="rvt"))
            ff.write(f"# meteorological forcings\n")
            for f in self.create_forcing_block().values():
                for t in f:
                    ff.write(f"{t}\n")

            ff.writelines(gauge)
            ff.writelines(flow_observation)

        self.logger.info(f"Successfully wrote RVT file to {file_path}")
        self.logger.info(f"Used gauge location: Lat={gauge_lat:.6f}, Lon={gauge_lon:.6f}, Elev={station_elevation:.1f}m")

    def process_all_files(self, template: bool = False) -> None:
        """
        Process and generate all HBV model input files.
        
        Parameters
        ----------
        template : bool, optional
            Whether to create template files, by default False
        """
        self.logger.info(f"Starting HBV file generation for gauge {self.gauge_id}")
        self.logger.info(f"Template mode: {template}")
        
        try:
            # Create all HBV input files
            self.write_rvh_file(template=template)
            self.write_rvi_file(template=template)
            self.write_rvp_file(template=template)
            self.write_rvc_file(template=template)
            self.write_rvt_file(template=template)
            
            self.logger.info("✅ All HBV files generated successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Error during HBV file generation: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize HBV processor
    namelist_path = "/path/to/your/namelist.yaml"
    
    try:
        processor = HBVProcessor(namelist_path)
        
        # Generate all files (both template and regular versions)
        processor.process_all_files(template=False)
        processor.process_all_files(template=True)
        
        print("HBV processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")