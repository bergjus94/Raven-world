#--------------------------------------------------------------------------------
#################################### packages ###################################
#--------------------------------------------------------------------------------

import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import date
import numpy as np
import rasterio


#--------------------------------------------------------------------------------
############################### create model files ##############################
#--------------------------------------------------------------------------------

def create_header(gauge_id: str, model, author='Justine Berg',
                  rvx_type: str = "rvi"):
    """Creates header info for .rvX files

    Args:
        gauge_id: str
            Catchment id
        author: str
            Author name
        creation_date: str
            Date of file generation
        model: str
            Model type, e.g. 'GR4J'
        rvx_type: str
            .rvX file type

    Returns:
        header: list
            List with header info
    """
    creation_date = date.today()
    try:
        header_line = "#########################################################################"
        file_type = f":FileType          {rvx_type} ASCII Raven 3.5"
        author_line = f":WrittenBy         {author}"
        creation_date = f":CreationDate      {creation_date}"
        description = [
            "#",
            f"# Emulation of {model} simulation of {gauge_id}",
            "#------------------------------------------------------------------------ \n"]
        header = [header_line, file_type, author_line, creation_date, *description]
        return header
    except NameError:
        print("Probably project_config file could not be found...")
        pass


#---------------------------------------------------------------------------------

def get_hrus_by_elevation_band(model_dir, gauge_id):
    """
    Load HRU shapefile and create a list of HRU IDs for each elevation band.
    
    Args:
        model_dir: Directory containing model files
        gauge_id: ID of the gauge
        
    Returns:
        Dictionary with elevation bands as keys and lists of HRU IDs as values
    """
    # Load HRU shapefile
    hru_path = Path(model_dir, f'catchment_{gauge_id}', 'topo_files', 'HRU.shp')
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
    print(f"Found {len(hrus_by_band)} elevation bands with a total of {len(filtered_hru)} HRUs")
    for band in sorted(hrus_by_band.keys(), key=lambda x: int(x.split('-')[0])):
        print(f"  {band}: {len(hrus_by_band[band])} HRUs")
    
    return hrus_by_band

#---------------------------------------------------------------------------------


def create_rvh_file(model_dir, gauge_id, model_type, params, template):
    """Write Raven .rvh file by reading the HRU_table.csv file.

    Args:
        model_dir: Path or str
            Directory containing model files
        gauge_id: str
            ID of the gauge
        model_type: str
            Type of hydrological model
        params: dict
            Dictionary of model parameters
        template: bool
            Whether to create a template file (True) or a regular file (False)
    """
    # Determine file paths
    if template is False:
        param_or_name = "init"
        file_name: str = f"{gauge_id}_{model_type}.rvh"
        file_path: Path = Path(model_dir, f'catchment_{gauge_id}', model_type, file_name)
    else:
        param_or_name = "names"
        file_name: str = f"{gauge_id}_{model_type}.rvh.tpl"
        file_path: Path = Path(model_dir, f'catchment_{gauge_id}', model_type, 'templates', file_name)

    # Read HRU table from CSV file
    hru_table_path = Path(model_dir, f'catchment_{gauge_id}', 'topo_files', 'HRU_table.csv')
    
    try:
        # Try to read the HRU table file
        HRU = pd.read_csv(hru_table_path)
        print(f"Successfully read HRU table from {hru_table_path}")
    except Exception as e:
        print(f"Error reading HRU table from {hru_table_path}: {e}")
        return

    # Convert the HRU table to a formatted string for the RVH file
    x = HRU.to_string(header=False,
                index=False,
                index_names=False).split('\n')
    vals = [',\t'.join(ele.split()) for ele in x]

    # Make format HRUs rvh file
    hru_list_1 = [
        ":HRUs",
        "  :Attributes, ID,  AREA, ELEVATION, LATITUDE, LONGITUDE, BASIN_ID,LAND_USE_CLASS, VEG_CLASS, SOIL_PROFILE, AQUIFER_PROFILE, TERRAIN_CLASS, SLOPE, ASPECT",
        "  :Units     , none,   km2,         m,      deg,       deg,     none,          none,      none,         none,            none,          none,   deg,    deg",
    ]

    hru_list_2 = [":EndHRUs"]
    hru_list = [*hru_list_1, *vals, *hru_list_2]
    
    # Add lateral connections block after HRU list
    lateral_connections = [
        "",
        "#:LateralConnections",
        ":RedirectToFile  data_obs/connections.rvh",
        "#:EndLateralConnections",
        ""
    ]
    
    # Get HRUs by elevation band
    hrus_by_band = get_hrus_by_elevation_band(model_dir, gauge_id)
    
    # Create HRU groups section
    hru_groups = []

    for band, hru_ids in hrus_by_band.items():
        if hru_ids:
            # Format as ":HRUGroup band_name" followed by HRU IDs on next line and closing tag
            hru_groups.append(f":HRUGroup {band}")
            hru_groups.append(f"  {' '.join(map(str, hru_ids))}")
            hru_groups.append(":EndHRUGroup")
            hru_groups.append("")  # Add empty line between groups

    # Define subbasins section
    Subbasins = [
        ":SubBasins",
        "  :Attributes,          NAME, DOWNSTREAM_ID,PROFILE,REACH_LENGTH,       GAUGED",
        "  :Units     ,          none,          none,   none,          km,         none",
        f"            1,        {gauge_id},            -1,   NONE,       _AUTO,     1",
        ":EndSubBasins"
    ]

    # Define subbasin properties based on model type
    if model_type == 'HBV':
        Subbasin_properties = [
            ":SubBasinProperties",
            "#                       HBV_PARA_11, DERIVED FROM HBV_PARA_11,",
            "#                            MAXBAS,                 MAXBAS/2,",
            "   :Parameters,           TIME_CONC,             TIME_TO_PEAK,",
            "   :Units,                        d,                        d,",
            f"              1,          {params['HBV'][param_or_name]['HBV_Param_11']},                  {params['HBV'][param_or_name]['HBV_Param_11b']},",
            ":EndSubBasinProperties"
        ]
    elif model_type == 'HYMOD':
        Subbasin_properties = [
            ":SubBasinProperties",
            "#                         HYMOD_PARA_1,                  3,",
            "   :Parameters,           RES_CONSTANT,     NUM_RESERVOIRS,",
            "   :Units,                         1/d,                  -,",
            f"              1,          {params['HYMOD'][param_or_name]['HYMOD_Param_01']},                  3,",
            ":EndSubBasinProperties"
        ]
    elif model_type == 'MOHYSE':
        Subbasin_properties = [
            ":SubBasinProperties",
            "#          1.0 / MOHYSE_PARA_10,   MOHYSE_PARA_9",
            "   :Parameters,     GAMMA_SCALE,     GAMMA_SHAPE,",
            "   :Units,                  1/d,               -",
            f"              1,          {params['MOHYSE'][param_or_name]['MOHYSE_Param_10']},                  {params['MOHYSE'][param_or_name]['MOHYSE_Param_09']},",
            ":EndSubBasinProperties"
        ]
    else:  # GR4J or HMETS
        Subbasin_properties = [" "]

    # Write the file
    with open(file_path, 'w') as ff:
        ff.writelines(f"{line}\n" for line in
                     create_header(author='Justine Berg', model=model_type, gauge_id=gauge_id, rvx_type="rvh"))
        ff.writelines(f"{line}\n" for line in Subbasins)
        ff.writelines(f"{line}\n" for line in Subbasin_properties)
        ff.writelines(f"{line}\n" for line in hru_list)
        ff.writelines(f"{line}\n" for line in lateral_connections)
        ff.write("\n# Elevation Band HRU Groups\n")
        ff.writelines(f"{line}\n" for line in hru_groups)
        
    print(f"Successfully wrote RVH file to {file_path}")


#---------------------------------------------------------------------------------

def forcing_block(coupled):
    """Create Dictionary of forcing data to write in RVT file.

    This function creates a Dictionary of forcing data to be written into an RVT file.
    It configures forcing data paths based on whether the model is coupled with glacier
    information or not.

    Parameters
    ----------
    coupled : bool
        Whether the model is coupled with glacier information.
        If True, uses transformed forcing files (*_new.nc) with 'data' variable names.
        If False, uses standard forcing files with original variable names.

    Returns
    -------
    dict
        Dictionary containing forcing data configuration for rainfall and temperature
        in a format suitable for writing to a Raven RVT file.
    """

    grid_weights_file_path = f"data_obs/GridWeights.txt"
    
    if coupled:
        
        forcing_rainfall = [
          ":GriddedForcing           Rainfall",
            "    :ForcingType          RAINFALL",
            f"    :FileNameNC           data_obs/era5_land_precip_coupled.nc",
            "    :VarNameNC            data",
            "    :DimNamesNC           x y time     # must be in the order of (x,y,t) ",
            f"    :RedirectToFile       {grid_weights_file_path}",
            ":EndGriddedForcing",
            '']
        forcing_temp_ave = [
            ":GriddedForcing           Average Temperature",
            "    :ForcingType          TEMP_AVE",
            f"    :FileNameNC           data_obs/era5_land_temp_mean_coupled.nc",
            "    :VarNameNC            data",
            "    :DimNamesNC           x y time     # must be in the order of (x,y,t) ",
            f"    :RedirectToFile       {grid_weights_file_path}",
           ":EndGriddedForcing",
            '']
        forcing_temp_max = [
            ":GriddedForcing           Maximum Temperature",
           "    :ForcingType          TEMP_MAX",
           f"    :FileNameNC           data_obs/era5_land_temp_max_coupled.nc",
            "    :VarNameNC            data",
            "    :DimNamesNC           x y time     # must be in the order of (x,y,t) ",
            f"    :RedirectToFile       {grid_weights_file_path}",
            ":EndGriddedForcing",
            '']
        forcing_temp_min = [
           ":GriddedForcing           Minimum Temperature",
            "    :ForcingType          TEMP_MIN",
            f"    :FileNameNC           data_obs/era5_land_temp_min_coupled.nc",
            "    :VarNameNC            data",
            "    :DimNamesNC           x y time     # must be in the order of (x,y,t) ",
            f"    :RedirectToFile       {grid_weights_file_path}",
            ":EndGriddedForcing",
            '']
        
        
    if not coupled:
    
        forcing_rainfall = [
          ":GriddedForcing           Rainfall",
            "    :ForcingType          RAINFALL",
            f"    :FileNameNC           data_obs/era5_land_precip.nc",
            "    :VarNameNC            tp",
            "    :DimNamesNC           longitude latitude time     # must be in the order of (x,y,t) ",
            f"    :RedirectToFile       {grid_weights_file_path}",
            ":EndGriddedForcing",
            '']
        forcing_temp_ave = [
            ":GriddedForcing           Average Temperature",
            "    :ForcingType          TEMP_AVE",
            f"    :FileNameNC           data_obs/era5_land_temp_mean.nc",
            "    :VarNameNC            t2m",
            "    :DimNamesNC           longitude latitude time     # must be in the order of (x,y,t) ",
            f"    :RedirectToFile       {grid_weights_file_path}",
           ":EndGriddedForcing",
            '']
        forcing_temp_max = [
            ":GriddedForcing           Maximum Temperature",
           "    :ForcingType          TEMP_MAX",
           f"    :FileNameNC           data_obs/era5_land_temp_max.nc",
            "    :VarNameNC            t2m",
            "    :DimNamesNC           longitude latitude time     # must be in the order of (x,y,t) ",
            f"    :RedirectToFile       {grid_weights_file_path}",
            ":EndGriddedForcing",
            '']
        forcing_temp_min = [
           ":GriddedForcing           Minimum Temperature",
            "    :ForcingType          TEMP_MIN",
            f"    :FileNameNC           data_obs/era5_land_temp_min.nc",
            "    :VarNameNC            t2m",
            "    :DimNamesNC           longitude latitude time     # must be in the order of (x,y,t) ",
            f"    :RedirectToFile       {grid_weights_file_path}",
            ":EndGriddedForcing",
            '']

    forcing_data = {
        'Rainfall':
            forcing_rainfall,
        'Average Temperature':
            forcing_temp_ave,
        'Maximum Temperature':
            forcing_temp_max,
        'Minimum Temperature':
            forcing_temp_min
    }

    return forcing_data

#---------------------------------------------------------------------------------

def get_gauge_location_from_dem(model_dir, gauge_id):
    """
    Extract gauge location (lat, lon, elevation) from the lowest point in the DEM
    
    Parameters
    ----------
    model_dir : Path or str
        Directory containing model files
    gauge_id : str
        ID of the gauge
        
    Returns
    -------
    tuple
        (latitude, longitude, elevation) of the lowest point in the DEM
    """
    # Path to clipped DEM
    dem_path = Path(model_dir, f'catchment_{gauge_id}', 'topo_files', 'clipped_dem.tif')
    
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
            
            # Handle NoData values more carefully
            nodata = dem.nodata
            if nodata is not None:
                print(f"Found NoData value: {nodata}")
                # Create mask for valid data
                valid_mask = (dem_data != nodata) & ~np.isnan(dem_data)
                print(f"Valid pixels: {valid_mask.sum()} out of {dem_data.size}")
                
                if not valid_mask.any():
                    raise ValueError("No valid data found in DEM - all pixels are NoData")
                
                # Get valid data only
                dem_data_valid = dem_data[valid_mask]
            else:
                print("No NoData value specified")
                # Check for NaN values and common NoData values
                valid_mask = ~np.isnan(dem_data) & (dem_data > -9999)
                if not valid_mask.any():
                    raise ValueError("No valid data found in DEM - all pixels are NaN or NoData")
                dem_data_valid = dem_data[valid_mask]
            
            if len(dem_data_valid) == 0:
                raise ValueError("No valid elevation data found after filtering")
            
            print(f"Valid elevation range: {dem_data_valid.min():.1f} to {dem_data_valid.max():.1f} m")
            
            # Find minimum elevation from valid data
            min_elevation = np.min(dem_data_valid)
            print(f"Minimum elevation: {min_elevation:.1f} m")
            
            # Find the pixel coordinates of the minimum elevation
            if nodata is not None:
                # Search for minimum elevation in valid pixels only
                min_indices = np.where((dem_data == min_elevation) & valid_mask)
            else:
                min_indices = np.where((dem_data == min_elevation) & valid_mask)
            
            print(f"Found {len(min_indices[0])} pixels with minimum elevation")
            
            if len(min_indices[0]) == 0:
                # Fallback: find closest value to minimum within tolerance
                print("No exact match found, looking for closest value...")
                tolerance = 1.0  # Increase tolerance to 1m
                close_indices = np.where(
                    (np.abs(dem_data - min_elevation) <= tolerance) & valid_mask
                )
                
                if len(close_indices[0]) == 0:
                    # Further fallback: just take any pixel close to minimum
                    print(f"No pixels within {tolerance}m tolerance, expanding search...")
                    tolerance = 10.0  # 10m tolerance
                    close_indices = np.where(
                        (np.abs(dem_data - min_elevation) <= tolerance) & valid_mask
                    )
                    
                    if len(close_indices[0]) == 0:
                        raise ValueError(f"Could not find any valid pixels near minimum elevation {min_elevation}")
                
                min_indices = close_indices
                print(f"Found {len(min_indices[0])} pixels within {tolerance}m of minimum elevation")
            
            # Take the first pixel with minimum (or closest to minimum) elevation
            row_idx = min_indices[0][0]
            col_idx = min_indices[1][0]
            
            print(f"Selected pixel at row {row_idx}, col {col_idx}")
            print(f"Pixel elevation: {dem_data[row_idx, col_idx]:.1f} m")
            
            # Convert pixel coordinates to geographic coordinates
            # Using rasterio's pixel-to-coordinate transformation
            lon, lat = dem.xy(row_idx, col_idx)
            
            # Note: Your DEM is in UTM (EPSG:32643), convert to WGS84 if needed
            if dem.crs != 'EPSG:4326':
                print(f"Converting coordinates from {dem.crs} to WGS84...")
                import pyproj
                transformer = pyproj.Transformer.from_crs(dem.crs, 'EPSG:4326', always_xy=True)
                lon_wgs84, lat_wgs84 = transformer.transform(lon, lat)
                
                print(f"UTM coordinates: {lon:.1f}, {lat:.1f}")
                print(f"WGS84 coordinates: {lon_wgs84:.6f}, {lat_wgs84:.6f}")
                
                # Use WGS84 coordinates for Raven
                lon, lat = lon_wgs84, lat_wgs84
            
            print(f"Gauge location extracted from DEM:")
            print(f"  Latitude: {lat:.6f}")
            print(f"  Longitude: {lon:.6f}")
            print(f"  Elevation: {min_elevation:.1f} m")
            
            return lat, lon, min_elevation
            
    except Exception as e:
        print(f"Error reading DEM file: {e}")
        import traceback
        traceback.print_exc()
        raise



#---------------------------------------------------------------------------------

def write_rvt(gauge_id: str,
              model_dir,
              model_type,
              params,
              coupled = False,
              author='Justine Berg',
              template = False):
    """Write to Raven .rvt file.

    Args:
        gauge_id: str
            ID of the gauge
        model_dir: str
            Root directory of .rvX files
        model_type: str
            Model type, e.g. 'GR4J'
        params: dict
            Dictionary of model parameters
        coupled: bool
            Whether model is coupled
        author: str
            Author name
        template: bool
            Should a template file be created?
    """
    
    # Extract gauge location from DEM
    print(f"Extracting gauge location from DEM for gauge {gauge_id}...")
    try:
        gauge_lat, gauge_lon, station_elevation = get_gauge_location_from_dem(model_dir, gauge_id)
    except Exception as e:
        print(f"Error: Could not extract gauge location from DEM: {e}")
        print("Please check that the DEM file exists and is readable.")
        return
    
    # Determine parameter type and file paths
    if template is False:
        param_or_name = "init"
        file_name: str = f"{gauge_id}_{model_type}.rvt"
        file_path: Path = Path(model_dir,'catchment_' + str(gauge_id),model_type,file_name)
    if template is True:
        param_or_name = "names"
        file_name: str = f"{gauge_id}_{model_type}.rvt.tpl"
        file_path: Path = Path(model_dir,'catchment_' + str(gauge_id),model_type,'templates',file_name)

    # Gauge info
    # ----------
    gauge_header = f":Gauge {gauge_id}\n"
    gauge_end = f":EndGauge\n\n"
    gauge_info = [
        f"  :Latitude    {gauge_lat}\n",
        f"  :Longitude {gauge_lon}\n",
        f"  :Elevation  {station_elevation}\n\n",
        ' '
    ]

    # Discharge info
    # --------------
    flow_observation = [
        "# observed streamflow\n",
        f":RedirectToFile data_obs/Q_daily.rvt"
    ]

    if model_type == "HBV":
        # Try to read monthly temperature and evaporation data if available
        monthly_temp_file = Path(model_dir, f'catchment_{gauge_id}', model_type, 'data_obs', 'monthly_temperature_averages.csv')
        monthly_pet_file = Path(model_dir, f'catchment_{gauge_id}', model_type, 'data_obs', 'monthly_pet_averages.csv')
        
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
                print(f"Error reading monthly data: {e}")
                monthly_data = []
        
        gauge_correction = [
            f"#  :RainCorrection    {params['HBV'][param_or_name]['HBV_Param_20']} \n",
            f"#  :SnowCorrection    {params['HBV'][param_or_name]['HBV_Param_21']}\n \n"
        ]

        gauge = [
            gauge_header,
            *gauge_info,
            *gauge_correction,
            *monthly_data,  # Add monthly data if available
            gauge_end
        ]

    if not model_type == "HBV":
        gauge = [
            gauge_header,
            *gauge_info,
            gauge_end
        ]

    with open(file_path, 'w') as ff:
        ff.writelines(f"{line}\n" for line in
                      create_header(author=author, model=model_type, gauge_id=gauge_id, rvx_type="rvt"))
        ff.write(f"# meteorological forcings\n")
        for f in forcing_block(coupled).values():
            for t in f:
                ff.write(f"{t}\n")

        ff.writelines(gauge)
        ff.writelines(flow_observation)

    print(f"Successfully wrote RVT file to {file_path}")
    print(f"Used gauge location: Lat={gauge_lat:.6f}, Lon={gauge_lon:.6f}, Elev={station_elevation:.1f}m")

#---------------------------------------------------------------------------------

def write_rvp_file(model_type, params, gauge_id, model_dir, template):
    
    if template is False:
        param_or_name = "init"
        file_name: str = f"{gauge_id}_{model_type}.rvp"
        file_path: Path = Path(model_dir,'catchment_' + str(gauge_id),model_type, file_name)
    if template is True:
        param_or_name = "names"
        file_name: str = f"{gauge_id}_{model_type}.rvp.tpl"
        file_path: Path = Path(model_dir,'catchment_' + str(gauge_id),model_type,'templates',file_name)
    
    land_use_classes = \
            [
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
    vegetation_classes = \
            [
                ":VegetationClasses",
                "   :Attributes, MAX_HT, MAX_LAI, MAX_LEAF_COND",
                "   :Units, m, none, mm_per_s",
                "   DEFAULT_V, 0.0, 0.0, 0.0",
                "   FOREST,  25,  6.0, 5.0",
                "   GRAS,    0.6, 2.0, 5.0",
                "   CROP,    2.0, 4.0, 5.0",
                ":EndVegetationClasses"
            ]
    
    if model_type == 'GR4J':
        rvp = \
            {
                "#Soil Classes":
                    [
                        ":SoilClasses",
                        "   :Attributes",
                        "   :Units",
                        "       SOIL_PROD",
                        "       SOIL_ROUT",
                        "       SOIL_TEMP",
                        "       SOIL_GW",
                        ":EndSoilClasses"
                    ],
                "#Soil Profiles":
                    [
                        "#     name,#horizons,{soiltype,thickness}x{#horizons}",
                        "#     GR4J_X1 is thickness of first layer (SOIL_PROD), here 0.529",
                        ":SoilProfiles",
                        f"   GLACIER, 0",
                        f"   DEFAULT_P, 4, SOIL_PROD, {params['GR4J'][param_or_name]['GR4J_X1']}, SOIL_ROUT, 0.300, SOIL_TEMP, 1.000, SOIL_GW, 1.000,",
                        ":EndSoilProfiles"
                    ],
                "#Vegetation Classes":
                    vegetation_classes,
                "#Land Use Classes":
                    land_use_classes,
                "#Global Parameters":
                    [
                        ":GlobalParameter RAINSNOW_TEMP       0.0",
                        ":GlobalParameter RAINSNOW_DELTA      1.0",
                        f":GlobalParameter AIRSNOW_COEFF     {params['GR4J'][param_or_name]['Airsnow_Coeff']} # [1/d] = 1.0 - CEMANEIGE_X2 = 1.0 - x6",
                        f"#CN_X2 = {params['GR4J'][param_or_name]['GR4J_Cemaneige_X2']}",
                        f":GlobalParameter AVG_ANNUAL_SNOW    {params['GR4J'][param_or_name]['Cemaneige_X1']} # [mm]  =       CEMANEIGE_X1 =       x5",
                        ":GlobalParameter PRECIP_LAPSE     0.0004 I assume not necessary for gridded data",
                        ":GlobalParameter ADIABATIC_LAPSE  0.0065 not necessary for gridded data"
                    ],
                "#Soil Parameters":
                    [
                        ":SoilParameterList",
                        "   :Parameters, POROSITY, GR4J_X3, GR4J_X2",
                        "   :Units, none, mm, mm / d",
                        f"   [DEFAULT], 1.0, {params['GR4J'][param_or_name]['GR4J_X3']}, {params['GR4J'][param_or_name]['GR4J_X2']}",
                        ":EndSoilParameterList"
                    ],
                "#Land Use Parameters":
                    [
                        ":LandUseParameterList",
                        "   :Parameters, GR4J_X4, MELT_FACTOR",
                        "   :Units, d, mm / d / C",
                        f"   [DEFAULT], {params['GR4J'][param_or_name]['GR4J_X4']}, 3.5",
                        ":EndLandUseParameterList"
                    ]
            }
    
    if model_type == 'HMETS':
        rvp = \
            {
                "#Soil Classes":
                    [
                        ":SoilClasses",
                        "   :Attributes,",
                        "   :Units,",
                        "       TOPSOIL,",
                        "       PHREATIC,",
                        ":EndSoilClasses"
                    ],
                "#Soil Profiles":
                    [
                        "#     name,#horizons,{soiltype,thickness}x{#horizons}",
                        ":SoilProfiles",
                        "   LAKE, 0",
                        "   ROCK, 0",
                        "   GLACIER, 0",
                        "   # DEFAULT_P, 2, TOPSOIL,          x(20)/1000, PHREATIC,         x(21)/1000,",
                        f"  DEFAULT_P, 2, TOPSOIL,     {params['HMETS'][param_or_name]['HMETS_Param_20b']}, PHREATIC,     {params['HMETS'][param_or_name]['HMETS_Param_21b']},",
                        ":EndSoilProfiles"
                    ],
                "#Vegetation Classes":
                    vegetation_classes,
                "#Land Use Classes":
                    land_use_classes,
                "#Global Parameters":
                    [
                        f":GlobalParameter  SNOW_SWI_MIN {params['HMETS'][param_or_name]['HMETS_Param_09a']} # x(9)",
                        f":GlobalParameter  SNOW_SWI_MAX {params['HMETS'][param_or_name]['HMETS_Param_09b']} # x(9)+x(10) = {params['HMETS'][param_or_name]['HMETS_Param_09a']} + {params['HMETS'][param_or_name]['HMETS_Param_10']}",
                        f":GlobalParameter  SWI_REDUCT_COEFF {params['HMETS'][param_or_name]['HMETS_Param_11']} # x(11)",
                        ":GlobalParameter SNOW_SWI 0.05 #not sure why/if needed"
                    ],
                "#Vegetation Parameters":
                    [
                        ":VegetationParameterList",
                        "   :Parameters,  RAIN_ICEPT_PCT,  SNOW_ICEPT_PCT,",
                        "   :Units,               -,               -,",
                        "       [DEFAULT],             0.0,             0.0,",
                        ":EndVegetationParameterList"
                    ],
                "#Land Use Parameters":
                    [
                        ":LandUseParameterList",
                        "   :Parameters, MIN_MELT_FACTOR, MAX_MELT_FACTOR,    DD_MELT_TEMP,  DD_AGGRADATION, REFREEZE_FACTOR,    REFREEZE_EXP, DD_REFREEZE_TEMP, HMETS_RUNOFF_COEFF,",
                        "   :Units,          mm/d/C,          mm/d/C,               C,            1/mm,          mm/d/C,               -,                C,                  -,",
                        f"      [DEFAULT],  {params['HMETS'][param_or_name]['HMETS_Param_05a']}, {params['HMETS'][param_or_name]['HMETS_Param_05b']},  {params['HMETS'][param_or_name]['HMETS_Param_07']},  {params['HMETS'][param_or_name]['HMETS_Param_08']},  {params['HMETS'][param_or_name]['HMETS_Param_13']},  {params['HMETS'][param_or_name]['HMETS_Param_14']},   {params['HMETS'][param_or_name]['HMETS_Param_12']},     {params['HMETS'][param_or_name]['HMETS_Param_16']},",
                        f"#      x(5),       x(5)+x(6) = {params['HMETS'][param_or_name]['HMETS_Param_05a']} + {params['HMETS'][param_or_name]['HMETS_Param_06']},            x(7),            x(8),           x(13),           x(14),            x(12),              x(16),",
                        ":EndLandUseParameterList",
                        "",
                        ":LandUseParameterList",
                        "   :Parameters,     GAMMA_SHAPE,     GAMMA_SCALE,    GAMMA_SHAPE2,    GAMMA_SCALE2,",
                        "   :Units,               -,             1/d,               -,             1/d,",
                        f"      [DEFAULT],  {params['HMETS'][param_or_name]['HMETS_Param_01']},  {params['HMETS'][param_or_name]['HMETS_Param_02']},  {params['HMETS'][param_or_name]['HMETS_Param_03']},  {params['HMETS'][param_or_name]['HMETS_Param_04']},",
                        "#      x(1),            x(2),            x(3),            x(4),",
                        ":EndLandUseParameterList"
                    ],
                "#Soil Parameters":
                    [
                        ":SoilParameterList",
                        "   :Parameters,        POROSITY,      PERC_COEFF,  PET_CORRECTION, BASEFLOW_COEFF",
                        "   :Units,               -,             1/d,               -,            1/d",
                        f"      TOPSOIL,             1.0,  {params['HMETS'][param_or_name]['HMETS_Param_17']},  {params['HMETS'][param_or_name]['HMETS_Param_15']}, {params['HMETS'][param_or_name]['HMETS_Param_18']}",
                        f"      PHREATIC,             1.0,             0.0,             0.0, {params['HMETS'][param_or_name]['HMETS_Param_19']}",
                        "#      TOPSOIL,             1.0,           x(17),           x(15),          x(18)",
                        "#      PHREATIC,             1.0,             0.0,             0.0,          x(19)",
                        ":EndSoilParameterList"
                    ]
            }
            
            
    if model_type == 'HYMOD':  
        rvp = \
            {
                "#Soil Classes":
                    [
                        ":SoilClasses",
                        "   :Attributes",
                        "   :Units",
                        "       TOPSOIL",
                        "       GWSOIL",
                        ":EndSoilClasses"
                    ],
                "#Soil Profiles":
                    [
                        "#     name,#horizons,{soiltype,thickness}x{#horizons}",
                        "# ",
                        ":SoilProfiles",
                        "   LAKE, 0,",
                        "   ROCK, 0,",
                        "   GLACIER, 0,",
                        "   # DEFAULT_P,      2, TOPSOIL,  HYMOD_PARA_2, GWSOIL, 10.0",
                        f"   DEFAULT_P, 2, TOPSOIL, {params['HYMOD'][param_or_name]['HYMOD_Param_02']}, GWSOIL, 10.0",
                        ":EndSoilProfiles"
                    ],
                "#Land Use Classes":
                    land_use_classes,
                "#Vegetation Classes":
                    vegetation_classes,
                "#Global Parameters":
                    [
                        f":GlobalParameter RAINSNOW_TEMP {params['HYMOD'][param_or_name]['HYMOD_Param_03']}",
                        "   #:GlobalParameter      RAINSNOW_TEMP    HYMOD_PARA_3"
                    ],
                "#Soil Parameters":
                    [
                        ":SoilParameterList",
                        "   :Parameters, POROSITY, PET_CORRECTION, BASEFLOW_COEFF,",
                        "   :Units, -, -, 1 / d,",
                        "       # TOPSOIL,            1.0 ,    HYMOD_PARA_8,               0.0,",
                        "       #  GWSOIL,            1.0 ,             1.0,   HYMOD_PARA_4=Ks,",
                        f"       TOPSOIL, 1.0, {params['HYMOD'][param_or_name]['HYMOD_Param_08']}, 0.0,",
                        f"       GWSOIL, 1.0, 1.0, {params['HYMOD'][param_or_name]['HYMOD_Param_04']},",
                        ":EndSoilParameterList"
                    ],
                "#Land Use Parameters":
                    [
                        ":LandUseParameterList",
                        "   :Parameters, MELT_FACTOR, DD_MELT_TEMP, PDM_B,",
                        "   :Units, mm / d / K, degC, -,",
                        "       # [DEFAULT],    HYMOD_PARA_5,    HYMOD_PARA_6,  HYMOD_PARA_7=Bexp,",
                        f"       [DEFAULT], {params['HYMOD'][param_or_name]['HYMOD_Param_05']}, {params['HYMOD'][param_or_name]['HYMOD_Param_06']}, {params['HYMOD'][param_or_name]['HYMOD_Param_07']},",
                        ":EndLandUseParameterList"
                    ]
            }
        
    if model_type == 'MOHYSE':
        rvp = \
            {
                "#Soil Classes":
                    [
                        ":SoilClasses",
                        "   :Attributes,",
                        "   :Units,",
                        "       TOPSOIL",
                        "       GWSOIL",
                        ":EndSoilClasses"
                    ],
                "#Soil Profiles":
                    [
                        "#  name,#horizons,{soiltype,thickness}x{#horizons}",
                        "# ",
                        ":SoilProfiles",
                        "   LAKE, 0",
                        "   ROCK, 0",
                        "   GLACIER, 0",
                        "#  DEFAULT_P,      2, TOPSOIL, MOHYSE_PARA_5, GWSOIL, 10.0",
                        f"   DEFAULT_P,      2, TOPSOIL,     {params['MOHYSE'][param_or_name]['MOHYSE_Param_05']}, GWSOIL, 10.0",
                        ":EndSoilProfiles"
                    ],
                "#Vegetation Classes":
                    vegetation_classes,
                "#Vegetation Parameters":
                    [
                        ":VegetationParameterList",
                        "   :Parameters,    SAI_HT_RATIO,  RAIN_ICEPT_PCT,  SNOW_ICEPT_PCT,",
                        "   :Units,               -,               -,               -, ",
                        "       [DEFAULT],             0.0,             0.0,             0.0,   ",
                        ":EndVegetationParameterList"
                    ],
                "#Land Use Classes":
                    land_use_classes,
                "#Global Parameters":
                    [
                        "#:GlobalParameter      RAINSNOW_TEMP              -2.0",
                        ":GlobalParameter       TOC_MULTIPLIER              1.0",
                        f"# :GlobalParameter     MOHYSE_PET_COEFF  MOHYSE_PARA_01",
                        f":GlobalParameter       MOHYSE_PET_COEFF         {params['MOHYSE'][param_or_name]['MOHYSE_Param_01']}"
                    ],
                "#Land Use Parameters":
                    [
                        ":LandUseParameterList",
                        "   :Parameters,     MELT_FACTOR,       AET_COEFF, FOREST_SPARSENESS, DD_MELT_TEMP,",
                        "   :Units,          mm/d/K,            mm/d,                 -,         degC,",
                        "#      [DEFAULT],   MOHYSE_PARA_3,   MOHYSE_PARA_2,               0.0,MOHYSE_PARA_4, ",
                        f"      [DEFAULT],          {params['MOHYSE'][param_or_name]['MOHYSE_Param_03']},          {params['MOHYSE'][param_or_name]['MOHYSE_Param_02']},               0.0,       {params['MOHYSE'][param_or_name]['MOHYSE_Param_04']},",
                        ":EndLandUseParameterList"
                    ],
                "#Soil Parameters":
                    [
                        ":SoilParameterList",
                        "   :Parameters,        POROSITY,  PET_CORRECTION,        HBV_BETA,  BASEFLOW_COEFF,      PERC_COEFF, ",
                        "   :Units,               -,               -,               -,             1/d,             1/d, ",
                        "#      TOPSOIL,            1.0 ,             1.0,             1.0,   MOHYSE_PARA_7,   MOHYSE_PARA_6,",
                        "#      GWSOIL,            1.0 ,             1.0,             1.0,   MOHYSE_PARA_8,             0.0,",
                        f"      TOPSOIL,            1.0 ,             1.0,             1.0,          {params['MOHYSE'][param_or_name]['MOHYSE_Param_07']},          {params['MOHYSE'][param_or_name]['MOHYSE_Param_06']},",
                        f"      GWSOIL,            1.0 ,             1.0,             1.0,          {params['MOHYSE'][param_or_name]['MOHYSE_Param_08']},             0.0,",
                        ":EndSoilParameterList"
                    ]
            }

    if model_type == 'HBV':

        rvp = \
            {
                "#Soil Classes":
                    [
                        ":SoilClasses",
                        "   :Attributes,",
                        "   :Units,",
                        "       TOPSOIL,      1.0,    0.0,       0",
                        "       SLOW_RES,     1.0,    0.0,       0",
                        "       FAST_RES,     1.0,    0.0,       0",
                        ":EndSoilClasses"
                    ],
                "#Soil Profiles":
                    [
                        "#     name,#horizons,{soiltype,thickness}x{#horizons}",
                        "# ",
                        ":SoilProfiles",
                        "    GLACIER, 0",
                        "    LAKE, 0",
                        "    ROCK, 0",
                        "    MASKED_GLACIER, 0",
                        f"   DEFAULT_P,      3,    TOPSOIL,            {params['HBV'][param_or_name]['HBV_Param_17']},   FAST_RES,    100.0, SLOW_RES,    100.0",
                        ":EndSoilProfiles"
                    ],
                "#Vegetation Classes":
                    vegetation_classes,
                "#Vegetation Parameters":
                    [
                        ":VegetationParameterList",
                        "   :Parameters,  MAX_CAPACITY, MAX_SNOW_CAPACITY,  TFRAIN,  TFSNOW,",
                        "   :Units,                 mm,                mm,    frac,    frac,",
                        "       [DEFAULT],             10000,             10000,    0.88,    0.88,",
                        ":EndVegetationParameterList"
                    ],
                "#Land Use Classes":
                    land_use_classes,
                "#Global Parameters":
                    [
                        f":GlobalParameter RAINSNOW_TEMP       {params['HBV'][param_or_name]['HBV_Param_01']}",
                        ":GlobalParameter RAINSNOW_DELTA      1.0 #constant",
                        f":GlobalParameter PRECIP_LAPSE       2.0 # HBV_PARA_12=PCALT",
                        f":GlobalParameter ADIABATIC_LAPSE    6.0 # HBV_PARA_13=TCALT",
                        f":GlobalParameter SNOW_SWI  {params['HBV'][param_or_name]['HBV_Param_04']} #HBV_PARA_04"
                    ],
                "#Land Use Parameters":
                    [
                        ":LandUseParameterList",
                        "  :Parameters,   MELT_FACTOR, MIN_MELT_FACTOR,   HBV_MELT_FOR_CORR, REFREEZE_FACTOR, HBV_MELT_ASP_CORR",
                        "  :Units     ,        mm/d/K,          mm/d/K,                none,          mm/d/K,              none",
                        "  #              HBV_PARA_02,        CONSTANT,         HBV_PARA_18,     HBV_PARA_03,          CONSTANT",
                        f"    [DEFAULT],  {params['HBV'][param_or_name]['HBV_Param_02']},             2.2,        {params['HBV'][param_or_name]['HBV_Param_18']},    {params['HBV'][param_or_name]['HBV_Param_03']},              0.48",
                        ":EndLandUseParameterList",
                        "",
                        ":LandUseParameterList",
                        " :Parameters, HBV_MELT_GLACIER_CORR,   HBV_GLACIER_KMIN, GLAC_STORAGE_COEFF, HBV_GLACIER_AG",
                        " :Units     ,                  none,                1/d,                1/d,           1/mm",
                        "   #                       CONSTANT,           CONSTANT,        HBV_PARA_19,       CONSTANT,",
                        f"   [DEFAULT],                  1.64,               0.05,       {params['HBV'][param_or_name]['HBV_Param_19']},           0.05",
                        ":EndLandUseParameterList"

                    ],
                "#Soil Parameters":
                    [
                        f"#For Ostrich:HBV_Alpha= {params['HBV'][param_or_name]['HBV_Param_15']}",
                        ":SoilParameterList",
                        "  :Parameters,                POROSITY,FIELD_CAPACITY,     SAT_WILT,     HBV_BETA, MAX_CAP_RISE_RATE,  MAX_PERC_RATE,  BASEFLOW_COEFF,            BASEFLOW_N",
                        "  :Units     ,                    none,          none,         none,         none,              mm/d,           mm/d,             1/d,                  none",
                        "  #                        HBV_PARA_05,   HBV_PARA_06,  HBV_PARA_14,  HBV_PARA_07,       HBV_PARA_16,       CONSTANT,        CONSTANT,              CONSTANT,",
                        f"    [DEFAULT],            {params['HBV'][param_or_name]['HBV_Param_05']},  {params['HBV'][param_or_name]['HBV_Param_06']}, {params['HBV'][param_or_name]['HBV_Param_14']}, {params['HBV'][param_or_name]['HBV_Param_07']},      {params['HBV'][param_or_name]['HBV_Param_16']},            0.0,             0.0,                   0.0",
                        "  #                                                        CONSTANT,                                     HBV_PARA_08,     HBV_PARA_09, 1+HBV_PARA_15=1+ALPHA,",
                        f"     FAST_RES,                _DEFAULT,      _DEFAULT,          0.0,     _DEFAULT,          _DEFAULT,   {params['HBV'][param_or_name]['HBV_Param_08']},    {params['HBV'][param_or_name]['HBV_Param_09']},              {params['HBV'][param_or_name]['HBV_Param_15']}",
                        "  #                                                        CONSTANT,                                                      HBV_PARA_10,              CONSTANT,",
                        f"     SLOW_RES,                _DEFAULT,      _DEFAULT,          0.0,     _DEFAULT,          _DEFAULT,       _DEFAULT,    {params['HBV'][param_or_name]['HBV_Param_10']},                   1.0",
                        ":EndSoilParameterList"
                    ]
            }
            
    if model_type == 'UBCWM':
        
        rvp =  \
            {
                "#Soil Classes":
                    [
                        ":SoilClasses",
                        "   :Attributes,",
                        "   :Units,",
                        "       TOPSOIL,",
                        "       INT_SOIL,",
                        "       INT_SOIL2,",
                        "       INT_SOIL3,",
                        "       GWU_SOIL,",
                        "       GWL_SOIL,",
                        ":EndSoilClasses"
                    ],
                    
                "#Soil Profiles":
                    [
                        "#     all thicknesses 10m (large enough to not fill)",
                        "# name, layers, {soilClass, thickness} x layers",
                        "# ",
                        "SoilProfiles",
                        "Lake, 0",
                        "GLACIER,   6, TOPSOIL, 0.0, INT_SOIL,10.0, GWU_SOIL,10.0, GWL_SOIL,10.0, INT_SOIL2,10.0, INT_SOIL3,10.0",
                        "DEFAULT_P  6, TOPSOIL,10.0, INT_SOIL,10.0, GWU_SOIL,10.0, GWL_SOIL,10.0, INT_SOIL2,10.0, INT_SOIL3,10.0",
                        "EndSoilProfiles"
                    ],
                    
                "#Vegetation Classes":
                    vegetation_classes,
                
                "Land Use Classes":
                    land_use_classes,
                    
                "#Orographic Correction":
                    [
                        ":AdiabaticLapseRate     4               # A0TLZZ",
                        ":WetAdiabaticLapse      4.4684  5       # A0TLZP A0PPTP",
                        ":ReferenceMaxTemperatureRange  20       # A0TERM(1)",
                        ":UBCTempLapseRates      6.49446  0.34441  6.4  2  4.52405  0.0260184   # A0TLXM A0TLNM A0TLXH A0TLNH P0TEDL P0TEDU",
                        ":UBCPrecipLapseRates    161  775.119  1035.3  1.59"
                        
                    ]
                
                    
                
            }
        
        
    # write rvp file
    with open(file_path, 'w') as ff:
        ff.writelines(f"{line}\n" for line in
                      create_header(author= 'Justine Berg', model=model_type, gauge_id=gauge_id, rvx_type="rvp"))
        for section, lines in rvp.items():
            ff.write(f"{section}\n")
            ff.writelines(line + '\n' for line in lines)
            ff.write('\n')


#---------------------------------------------------------------------------------

def write_rvi_file(start_date, end_date, gauge_id, model_type, cali_end_date, model_dir, params, template):
    
    if template is False:
        param_or_name = "init"
        file_name: str = f"{gauge_id}_{model_type}.rvi"
        file_path: Path = Path(model_dir,'catchment_' + str(gauge_id),model_type, file_name)
    if template is True:
        param_or_name = "names"
        file_name: str = f"{gauge_id}_{model_type}.rvi.tpl"
        file_path: Path = Path(model_dir,'catchment_' + str(gauge_id),model_type,'templates',file_name)
        
    hrus_by_band = get_hrus_by_elevation_band(model_dir, gauge_id)
    # Create elevation bands definition string
    elevation_bands = sorted(hrus_by_band.keys(), key=lambda x: int(x.split('-')[0]))
    hru_groups_definition = ":DefineHRUGroups " + " ".join(elevation_bands)
    
    if model_type == 'GR4J':
        rvi = \
            {"#Model Organisation":
                [
                    f":StartDate             {start_date} 00:00:00",
                    f":EndDate               {end_date} 00:00:00",
                    ":TimeStep              1.0",
                    ":Method                ORDERED_SERIES",
                    f":RunName               {gauge_id}_GR4J"
                ],
                "#Model Options":
                    [
                        ":SoilModel             SOIL_MULTILAYER  4",
                        ":Routing               ROUTE_NONE",
                        ":CatchmentRoute        ROUTE_DUMP",
                        ":Evaporation           PET_HAMON",
                        ":RainSnowFraction      RAINSNOW_DINGMAN",
                        ":PotentialMeltMethod   POTMELT_DEGREE_DAY",
                        ":OroTempCorrect        OROCORR_SIMPLELAPSE",
                        ":OroPrecipCorrect      OROCORR_SIMPLELAPSE",
                        f":EvaluationPeriod CALIBRATION {start_date} {cali_end_date}",
                        f":EvaluationPeriod VALIDATION {cali_end_date} {end_date}"
                    ],
                "#Soil Layer Alias Definitions":
                    [
                        ":Alias PRODUCT_STORE      SOIL[0]",
                        ":Alias ROUTING_STORE      SOIL[1]",
                        ":Alias TEMP_STORE         SOIL[2]",
                        ":Alias GW_STORE           SOIL[3]"
                    ],
                "#HRU Groups Definition":
                    [hru_groups_definition] if hru_groups_definition else [],
                "#Hydrologic Process Order":
                    [
                ":HydrologicProcesses",
                        "   :Precipitation            PRECIP_RAVEN       ATMOS_PRECIP    MULTIPLE",
                        "   :SnowTempEvolve           SNOTEMP_NEWTONS    SNOW_TEMP",
                        "   :SnowBalance              SNOBAL_CEMA_NIEGE  SNOW            PONDED_WATER",
                        "   :OpenWaterEvaporation     OPEN_WATER_EVAP    PONDED_WATER    ATMOSPHERE     	 # Pn",
                        "   :Infiltration             INF_GR4J           PONDED_WATER    MULTIPLE       	 # Ps-",
                        "   :SoilEvaporation          SOILEVAP_GR4J      PRODUCT_STORE   ATMOSPHERE     	 # Es",
                        "   :Percolation              PERC_GR4J          PRODUCT_STORE   TEMP_STORE     	 # Perc",
                        "   :Flush                    RAVEN_DEFAULT      SURFACE_WATER   TEMP_STORE     	 # Pn-Ps",
                        "   :Split                    RAVEN_DEFAULT      TEMP_STORE      CONVOLUTION[0] CONVOLUTION[1] 0.9  # Split Pr",
                        "   :Convolve                 CONVOL_GR4J_1      CONVOLUTION[0]  ROUTING_STORE  	 # Q9",
                        "   :Convolve                 CONVOL_GR4J_2      CONVOLUTION[1]  TEMP_STORE     	 # Q1",
                        "   :Percolation              PERC_GR4JEXCH      ROUTING_STORE   GW_STORE       	 # F(x1)",
                        "   :Percolation              PERC_GR4JEXCH2     TEMP_STORE      GW_STORE       	 # F(x1)",
                        "   :Flush                    RAVEN_DEFAULT      TEMP_STORE      SURFACE_WATER  	 # Qd",
                        "   :Baseflow                 BASE_GR4J          ROUTING_STORE   SURFACE_WATER  	 # Qr",
                ":EndHydrologicProcesses"
            ],
                "#Output Options":
                    [
                       "  :EvaluationMetrics RMSE KLING_GUPTA NASH_SUTCLIFFE ",
                       "  :WriteForcingFunctions",
                       "  :CustomOutput DAILY AVERAGE SNOW BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SNOWMELT BY_HRU",
                       "  :CustomOutput DAILY AVERAGE RUNOFF BY_HRU",
                       "  :CustomOutput DAILY AVERAGE PRECIP BY_HRU",
                       "  :CustomOutput DAILY AVERAGE ATMOSPHERE BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SOIL[0] BY_HRU",
                       "  :CustomOutput DAILY AVERAGE AET BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SURFACE_WATER BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SNOW BY_HRU_GROUP",
                    ]
            }
        
    if model_type == 'HMETS':
        rvi = \
            {"#Model Organisation":
                [
                    f":StartDate               {start_date} 00:00:00",
                    f":EndDate                {end_date} 00:00:00",
                    ":TimeStep                1.0",
                    ":Method                  ORDERED_SERIES",
                    f":RunName                 {gauge_id}_HMETS"
                ],
                "#Model Options":
                    [
                        ":PotentialMeltMethod     POTMELT_HMETS",
                        ":RainSnowFraction        RAINSNOW_DATA",
                        "#:Evaporation             PET_DATA",
                        ":Evaporation            PET_OUDIN",
                        ":CatchmentRoute          ROUTE_DUMP",
                        ":Routing                 ROUTE_NONE",
                        ":SoilModel               SOIL_TWO_LAYER",
                        f":EvaluationPeriod   CALIBRATION   {start_date}   {cali_end_date}",
                        f":EvaluationPeriod   VALIDATION    {cali_end_date}   {end_date}"
                    ],
                "#Alias Definitions":
                    [
                        ":Alias DELAYED_RUNOFF CONVOLUTION[1]"
                    ],
                "#HRU Groups Definition":
                    [hru_groups_definition] if hru_groups_definition else [],
                "#Hydrologic Process Order":
                    [
                        ":HydrologicProcesses",
                        "   :SnowBalance     SNOBAL_HMETS    MULTIPLE     MULTIPLE",
                        "   :Precipitation   RAVEN_DEFAULT   ATMOS_PRECIP MULTIPLE",
                        "   :Infiltration    INF_HMETS       PONDED_WATER MULTIPLE",
                        "   :Overflow      OVERFLOW_RAVEN  SOIL[0]      DELAYED_RUNOFF",
                        "   :Baseflow        BASE_LINEAR     SOIL[0]      SURFACE_WATER   # interflow, really",
                        "   :Percolation     PERC_LINEAR     SOIL[0]      SOIL[1]         # recharge",
                        "   :Overflow      OVERFLOW_RAVEN  SOIL[1]      DELAYED_RUNOFF",
                        "   :SoilEvaporation SOILEVAP_ALL    SOIL[0]      ATMOSPHERE      # AET",
                        "   :Convolve        CONVOL_GAMMA    CONVOLUTION[0] SURFACE_WATER #'surface runoff'",
                        "   :Convolve        CONVOL_GAMMA_2  DELAYED_RUNOFF SURFACE_WATER #'delayed runoff'",
                        "   :Baseflow        BASE_LINEAR     SOIL[1]      SURFACE_WATER",
                        ":EndHydrologicProcesses"
                    ],
                "#Output Options":
                    [
                       "  :EvaluationMetrics RMSE KLING_GUPTA NASH_SUTCLIFFE ",
                       "  :WriteForcingFunctions",
                       "  :CustomOutput DAILY AVERAGE SNOW BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SNOWMELT BY_HRU",
                       "  :CustomOutput DAILY AVERAGE RUNOFF BY_HRU",
                       "  :CustomOutput DAILY AVERAGE PRECIP BY_HRU",
                       "  :CustomOutput DAILY AVERAGE ATMOSPHERE BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SOIL[0] BY_HRU",
                       "  :CustomOutput DAILY AVERAGE AET BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SURFACE_WATER BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SNOW BY_HRU_GROUP",
                    ]
            }
        
    if model_type == 'HYMOD':
        rvi = \
            {"#Model Organisation":
                [
                    f":StartDate          {start_date} 00:00:00",
                    f":EndDate            {end_date} 00:00:00",
                    ":TimeStep           1.0",
                    ":Method             ORDERED_SERIES",
                    f":RunName            {gauge_id}_HYMOD"
                ],
                "#Model Options":
                    [
                        ":Routing             ROUTE_NONE",
                        ":CatchmentRoute      ROUTE_RESERVOIR_SERIES",
                        ":Evaporation         PET_HAMON",
                        ":OW_Evaporation      PET_HAMON",
                        ":SWRadiationMethod   SW_RAD_NONE",
                        ":LWRadiationMethod   LW_RAD_NONE",
                        ":CloudCoverMethod    CLOUDCOV_NONE",
                        ":RainSnowFraction    RAINSNOW_THRESHOLD",
                        ":PotentialMeltMethod POTMELT_DEGREE_DAY",
                        ":PrecipIceptFract    PRECIP_ICEPT_NONE",
                        ":SoilModel           SOIL_MULTILAYER 2",
                        f":EvaluationPeriod   CALIBRATION   {start_date}   {cali_end_date}",
                        f":EvaluationPeriod   VALIDATION    {cali_end_date}   {end_date}"
                    ],
                "#HRU Groups Definition":
                    [hru_groups_definition] if hru_groups_definition else [],
                "#Hydrologic Process Order":
                    [
                        ":HydrologicProcesses",
                        "   :Precipitation     PRECIP_RAVEN       ATMOS_PRECIP    MULTIPLE",
                        "   :SnowBalance       SNOBAL_SIMPLE_MELT SNOW            PONDED_WATER",
                        "   :Infiltration      INF_PDM            PONDED_WATER    MULTIPLE",
                        "#  :Flush            RAVEN_DEFAULT      SURFACE_WATER   SOIL[1]   HYMOD_PARAM_9=ALPHA",
                        f"  :Flush             RAVEN_DEFAULT      SURFACE_WATER   SOIL[1]          {params['HYMOD'][param_or_name]['HYMOD_Param_09']}",
                        "   :SoilEvaporation   SOILEVAP_PDM       SOIL[0]         ATMOSPHERE",
                        "   :Baseflow          BASE_LINEAR        SOIL[1]         SURFACE_WATER",
                        ":EndHydrologicProcesses"
                    ],
                "#Output Options":
                    [
                       "  :EvaluationMetrics RMSE KLING_GUPTA NASH_SUTCLIFFE ",
                       "  :WriteForcingFunctions",
                       "  :CustomOutput DAILY AVERAGE SNOW BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SNOWMELT BY_HRU",
                       "  :CustomOutput DAILY AVERAGE RUNOFF BY_HRU",
                       "  :CustomOutput DAILY AVERAGE PRECIP BY_HRU",
                       "  :CustomOutput DAILY AVERAGE ATMOSPHERE BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SOIL[0] BY_HRU",
                       "  :CustomOutput DAILY AVERAGE AET BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SURFACE_WATER BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SNOW BY_HRU_GROUP",
                    ]
            }
            
    if model_type == 'MOHYSE':
        
        rvi = \
            {"#Model Organisation":
                [
                    f":StartDate               {start_date} 00:00:00",
                    f":EndDate                {end_date} 00:00:00",
                    ":TimeStep                1.0",
                    ":Method                  ORDERED_SERIES",
                    f":RunName                 {gauge_id}_MOHYSE"
                ],
                "#Model Options":
                    [
                        ":SoilModel             SOIL_TWO_LAYER",
                        ":PotentialMeltMethod   POTMELT_DEGREE_DAY",
                        ":Routing               ROUTE_NONE",
                        ":CatchmentRoute        ROUTE_GAMMA_CONVOLUTION",
                        ":Evaporation           PET_MOHYSE",
                        ":DirectEvaporation",
                        ":RainSnowFraction      RAINSNOW_DATA",
                        f":EvaluationPeriod   CALIBRATION   {start_date}   {cali_end_date}",
                        f":EvaluationPeriod   VALIDATION    {cali_end_date}   {end_date}"
                    ],
                "#HRU Groups Definition":
                    [hru_groups_definition] if hru_groups_definition else [],
                "#Hydrologic Process Order":
                    [
                        ":HydrologicProcesses",
                        "   :SoilEvaporation  SOILEVAP_LINEAR    SOIL[0]            ATMOSPHERE",
                        "   :SnowBalance      SNOBAL_SIMPLE_MELT SNOW PONDED_WATER",
                        "   :Precipitation    RAVEN_DEFAULT      ATMOS_PRECIP       MULTIPLE",
                        "   :Infiltration     INF_HBV            PONDED_WATER       SOIL[0]",
                        "   :Baseflow         BASE_LINEAR        SOIL[0]            SURFACE_WATER",
                        "   :Percolation      PERC_LINEAR        SOIL[0]            SOIL[1]",
                        "   :Baseflow         BASE_LINEAR        SOIL[1]            SURFACE_WATER",
                        ":EndHydrologicProcesses"
                    ],
                "#Output Options":
                    [
                       "  :EvaluationMetrics RMSE KLING_GUPTA NASH_SUTCLIFFE ",
                       "  :WriteForcingFunctions",
                       "  :CustomOutput DAILY AVERAGE SNOW BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SNOWMELT BY_HRU",
                       "  :CustomOutput DAILY AVERAGE RUNOFF BY_HRU",
                       "  :CustomOutput DAILY AVERAGE PRECIP BY_HRU",
                       "  :CustomOutput DAILY AVERAGE ATMOSPHERE BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SOIL[0] BY_HRU",
                       "  :CustomOutput DAILY AVERAGE AET BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SURFACE_WATER BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SNOW BY_HRU_GROUP",
                    ]
            }

    if model_type == 'HBV':

        rvi = \
            {"#Model Organisation":
                [
                    f":StartDate             {start_date} 00:00:00",
                    f":EndDate               {end_date} 00:00:00",
                    ":TimeStep              1.0",
                    f":RunName               {gauge_id}_HBV"
                ],
                "#Model Options":
                    [
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
                        f":EvaluationPeriod   CALIBRATION   {start_date}   {cali_end_date}",
                        f":EvaluationPeriod   VALIDATION    {cali_end_date}   {end_date}"
                    ],
                "#Soil Alias Layer Definitions":
                    [
                        ":Alias       FAST_RESERVOIR SOIL[1]",
                        ":Alias       SLOW_RESERVOIR SOIL[2]",
                        ":LakeStorage SLOW_RESERVOIR"
                    ],
                "#HRU Groups Definition":
                    [hru_groups_definition] if hru_groups_definition else [],
                "#Hydrologic Process Order":
                    [
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
                        ":EndHydrologicProcesses"
                    ],
                "#Output Options":
                    [
                       "  :EvaluationMetrics RMSE KLING_GUPTA NASH_SUTCLIFFE ",
                       "  :WriteForcingFunctions",
                       "  :CustomOutput DAILY AVERAGE SNOW BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SNOWMELT BY_HRU",
                       "  :CustomOutput DAILY AVERAGE RUNOFF BY_HRU",
                       "  :CustomOutput DAILY AVERAGE PRECIP BY_HRU",
                       "  :CustomOutput DAILY AVERAGE ATMOSPHERE BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SOIL[0] BY_HRU",
                       "  :CustomOutput DAILY AVERAGE AET BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SURFACE_WATER BY_HRU",
                       "  :CustomOutput DAILY AVERAGE SNOW BY_HRU_GROUP",
                    ]
            }
        

    with open(file_path, 'w') as ff:
        ff.writelines(f"{line}\n" for line in
                      create_header(author= 'Justine Berg', model=model_type, gauge_id=gauge_id, rvx_type="rvi"))
        for section, lines in rvi.items():
            ff.write(f"{section}\n")
            ff.writelines(line + '\n' for line in lines)
            ff.write('\n')

#---------------------------------------------------------------------------------         


def write_rvc_file(model_type, params, gauge_id, model_dir, template):
    
    
    if template is False:
        param_or_name = "init"
        file_name: str = f"{gauge_id}_{model_type}.rvc"
        file_path: Path = Path(model_dir,'catchment_' + str(gauge_id),model_type, file_name)
    if template is True:
        param_or_name = "names"
        file_name: str = f"{gauge_id}_{model_type}.rvc.tpl"
        file_path: Path = Path(model_dir,'catchment_' + str(gauge_id),model_type,'templates',file_name)
    
    
    if model_type == 'GR4J':
        
        rvc = \
            {
                "#Soil Profiles":
                    [
                        "# SOIL[0] = GR4J_X1 * 1000. / 2.0 (initialize to 1/2 full)",
                        "# SOIL[1] = 0.3m * 1000. / 2.0   (initialize to 1/2 full)"
                    ],
                "#HRU States":
                    [
                        ":HRUStateVariableTable",
                        "   :Attributes SOIL[0] SOIL[1]",
                        "   :Units      mm      mm",
                        f"   1           {params['GR4J'][param_or_name]['GR4J_Soil_0']},   15.0",
                        ":EndHRUStateVariableTable"
                    ]
            }
        
    if model_type == 'HMETS':
        
        rvc = \
            {"#Initial Storage":
                [
                    "# initialize to 1/2 full",
                    "# x(20b)/2",
                    f"#:UniformInitialConditions SOIL[0] {params['HMETS'][param_or_name]['HMETS_Param_20b']}",
                    "# x(21b)/2",
                    f"#:UniformInitialConditions SOIL[1] {params['HMETS'][param_or_name]['HMETS_Param_21b']}"
                ],
                "#HRUs":
                    [
                        ":HRUStateVariableTable # (according to rchlumsk-BMSC-cf9a83c, modelname.rvc.tpl: formerly :InitialConditionsTable)",
                        ":Attributes SOIL[0] SOIL[1]",
                        ":Units mm mm",
                        f"1 {params['HMETS'][param_or_name]['HMETS_Param_20a']} {params['HMETS'][param_or_name]['HMETS_Param_21a']}",
                        ":EndHRUStateVariableTable",
                    ]
            }
        
        
    if model_type == 'HYMOD':
        
        rvc = \
            {"#Empty":
                [
                    "# Nothing to set here."
                ]
            }
        
    if model_type == 'MOHYSE':
        
        rvc = \
            {"#Empty":
                [
                    "# Nothing to set here."
                ]
            }

    if model_type == 'HBV':

        rvc = \
            {
                "#Basin":
                    [
                        ":BasinInitialConditions",
                        ":Attributes, ID,              Q",
                        ":Units,      none,         m3/s",
                        "#                  HBV_PARA_???",
                        "1,             1.0",
                        ":EndBasinInitialConditions"
                    ],
                "#Lower Groundwater Storage":
                    [
                        "# Initial Lower groundwater storage - for each HRU",
                        "",
                        ":InitialConditions SOIL[2]",
                        "# derived from thickness: HBV_PARA_17 [m] * 1000.0 / 2.0",
                        f"{params['HBV'][param_or_name]['HBV_Param_17b']}",
                        ":EndInitialConditions"
                    ]
            }
        
    # write rvc file

    with open(file_path, 'w') as ff:
        ff.writelines(f"{line}\n" for line in
                      create_header(author= 'Justine Berg', model=model_type, gauge_id=gauge_id, rvx_type="rvc"))
        for section, lines in rvc.items():
            ff.write(f"{section}\n")
            ff.writelines(line + '\n' for line in lines)
            ff.write('\n')
    
