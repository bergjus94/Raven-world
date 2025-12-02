import cdsapi
import os
import traceback
import zipfile
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import time
import yaml


def read_namelist(namelist_path):
    """Read gauge_id, years, shape_dir, meteo_dir from namelist.yaml"""
    with open(namelist_path, 'r') as f:
        config = yaml.safe_load(f)
    
    gauge_id = str(config.get('gauge_id'))
    main_dir = config.get('main_dir', '')
    shape_dir = config.get('shape_dir')
    meteo_dir = config.get('meteo_dir')
    
    # FIXED: Get simulation period from start_date and end_date
    start_date = config.get('start_date')  # Changed from 'sim_start'
    end_date = config.get('end_date')      # Changed from 'sim_end'
    
    # Extract years from simulation period
    if start_date and end_date:
        # Parse dates in format 'YYYY-MM-DD'
        start_year = int(start_date.split('-')[0])
        end_year = int(end_date.split('-')[0])
        years = list(range(start_year, end_year + 1))
        
        print(f"📅 Extracted years from namelist: {years}")
        print(f"   Start: {start_date}")
        print(f"   End: {end_date}")
    else:
        print("WARNING: start_date or end_date not found in namelist, using default years")
        years = [2020]  # Default fallback
    
    # Build absolute paths if needed
    shape_path = shape_dir.format(gauge_id=gauge_id)
    meteo_path = meteo_dir.format(gauge_id=gauge_id)
    
    if not os.path.isabs(shape_path):
        shape_path = os.path.join(main_dir, shape_path)
    if not os.path.isabs(meteo_path):
        meteo_path = os.path.join(main_dir, meteo_path)
    
    return gauge_id, years, shape_path, meteo_path


def get_extent_from_shapefile(shapefile_path, buffer_degrees=0.1):
    """
    Read shapefile and extract bounding box extent
    
    Parameters:
    -----------
    shapefile_path : str
        Path to the shapefile
    buffer_degrees : float
        Buffer to add around the shapefile extent in degrees (default: 0.1°)
    
    Returns:
    --------
    list
        [North, West, South, East] coordinates in WGS84
    """
    try:
        # Read shapefile
        gdf = gpd.read_file(shapefile_path)
        
        # Convert to WGS84 (EPSG:4326) if not already
        if gdf.crs != 'EPSG:4326':
            print(f"Converting from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs('EPSG:4326')
        
        # Get bounding box
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        
        # Add buffer
        minx, miny, maxx, maxy = bounds
        minx -= buffer_degrees
        miny -= buffer_degrees
        maxx += buffer_degrees
        maxy += buffer_degrees
        
        # ERA5 format: [North, West, South, East]
        area = [maxy, minx, miny, maxx]
        
        print(f"Extracted extent: North={maxy:.3f}, West={minx:.3f}, South={miny:.3f}, East={maxx:.3f}")
        print(f"Buffer applied: {buffer_degrees}°")
        
        return area
        
    except Exception as e:
        print(f"Error reading shapefile: {str(e)}")
        traceback.print_exc()
        return None


def download_one_month(variable, year, month, output_dir, area):
    """
    Download a single month of ERA5-Land data
    
    Parameters:
    -----------
    variable : str
        ERA5 variable name ('2m_temperature' or 'total_precipitation')
    year : int
        Year to download
    month : int
        Month to download
    output_dir : str
        Output directory
    area : list
        [North, West, South, East] coordinates
        
    Returns:
    --------
    str or None
        Path to downloaded file if successful, None otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Format month as two-digit string
        month_str = str(month).zfill(2)
        
        # Generate filename
        var_str = variable.replace(' ', '_')
        filename = f"era5_land_{var_str}_{year}_{month_str}.nc"
        file_path = os.path.join(output_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(file_path):
            print(f"✅ File already exists: {filename}")
            return file_path
        
        print(f"📥 Downloading {variable} for {year}-{month_str}...")
        
        # Initialize CDS client
        c = cdsapi.Client()
        
        # Make the API request
        c.retrieve(
            'reanalysis-era5-land',
            {
                'variable': variable,
                'year': str(year),
                'month': month_str,
                'day': [f"{day:02d}" for day in range(1, 32)],  # All days
                'time': [f"{hour:02d}:00" for hour in range(24)],  # All hours
                'area': area,  # [North, West, South, East]
                'data_format': 'netcdf',
                'download_format': 'unarchived',
            },
            file_path
        )
        
        # Handle ZIP files if necessary
        if zipfile.is_zipfile(file_path):
            print("📦 File is ZIP archive - extracting...")
            extract_dir = os.path.join(output_dir, f"temp_extract_{year}_{month_str}")
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find the NetCDF file
            nc_files = [f for f in os.listdir(extract_dir) if f.endswith('.nc')]
            
            if nc_files:
                extracted_file = os.path.join(extract_dir, nc_files[0])
                os.replace(extracted_file, file_path)
                
                # Clean up
                for f in os.listdir(extract_dir):
                    os.remove(os.path.join(extract_dir, f))
                os.rmdir(extract_dir)
                print("✅ Extracted successfully")
            else:
                print("❌ No NetCDF files found in archive")
                return None
        
        print(f"✅ Downloaded: {filename}")
        return file_path
    
    except Exception as e:
        print(f"❌ Error downloading {variable} {year}-{month:02d}: {str(e)}")
        traceback.print_exc()
        return None


def download_multiple_months_parallel(variable, years, months, output_dir, area, max_workers=4):
    """
    Download multiple months of ERA5-Land data in parallel
    
    Parameters:
    -----------
    variable : str
        ERA5 variable name
    years : list
        List of years to download
    months : list
        List of months to download (1-12)
    output_dir : str
        Output directory
    area : list
        [North, West, South, East] coordinates
    max_workers : int
        Maximum number of parallel downloads (default: 4)
        
    Returns:
    --------
    list
        List of successfully downloaded file paths
    """
    downloaded_files = []
    total_files = len(years) * len(months)
    
    print(f"\n📋 Downloading {total_files} files for {variable} (parallel)")
    print(f"   Years: {years}")
    print(f"   Months: {months}")
    print(f"   Max workers: {max_workers}")
    
    # Create list of (year, month) combinations
    year_month_combinations = list(product(years, months))
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_params = {
            executor.submit(download_one_month, variable, year, month, output_dir, area): (year, month)
            for year, month in year_month_combinations
        }
        
        # Process completed downloads
        completed = 0
        for future in as_completed(future_to_params):
            year, month = future_to_params[future]
            completed += 1
            
            try:
                file_path = future.result()
                if file_path:
                    downloaded_files.append(file_path)
                    print(f"✅ [{completed}/{total_files}] Downloaded: {variable} {year}-{month:02d}")
                else:
                    print(f"❌ [{completed}/{total_files}] Failed: {variable} {year}-{month:02d}")
                    
            except Exception as e:
                print(f"❌ [{completed}/{total_files}] Error {variable} {year}-{month:02d}: {str(e)}")
    
    print(f"\n📊 Download summary for {variable}:")
    print(f"   ✅ Successfully downloaded: {len(downloaded_files)}/{total_files} files")
    print(f"   ❌ Failed: {total_files - len(downloaded_files)} files")
    
    return downloaded_files


def download_geopotential_once(output_dir, area):
    """
    Download geopotential data (only need one time step since it's time-invariant)
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        filename = "era5_land_geopotential.nc"
        file_path = os.path.join(output_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(file_path):
            print(f"✅ Geopotential file already exists: {filename}")
            return file_path
        
        print(f"📥 Downloading geopotential (elevation data)...")
        
        c = cdsapi.Client()
        
        # Only need one time step for geopotential
        c.retrieve(
            'reanalysis-era5-land',
            {
                'variable': 'geopotential',
                'year': '2020',        # Any recent year
                'month': '01',         # Any month  
                'day': '01',           # Any day
                'time': '00:00',       # Any time
                'area': area,
                'data_format': 'netcdf',
                'download_format': 'unarchived',
            },
            file_path
        )
        
        print(f"✅ Downloaded: {filename}")
        return file_path
        
    except Exception as e:
        print(f"❌ Error downloading geopotential: {str(e)}")
        return None


def process_gauge_parallel(gauge_id, years, months, variables, shapefile_path, base_output_dir, buffer_degrees=0.1, geo_buffer_degrees=0.2, max_workers=4):
    """
    Download ERA5-Land data for a specific gauge using parallel processing
    
    Parameters:
    -----------
    gauge_id : str
        Gauge identifier (e.g., "0001")
    years : list
        List of years to download
    months : list  
        List of months to download (1-12)
    variables : list
        List of ERA5 variables
    shapefile_path : str
        Path to the catchment shapefile
    base_output_dir : str
        Base directory for output files
    buffer_degrees : float
        Buffer around shapefile extent in degrees for regular variables
    geo_buffer_degrees : float
        Buffer around shapefile extent in degrees for geopotential (should be larger)
    max_workers : int
        Maximum number of parallel downloads per variable
        
    Returns:
    --------
    dict
        Dictionary with download results for each variable
    """
    print(f"\n{'='*80}")
    print(f"🎯 PROCESSING GAUGE {gauge_id} (PARALLEL MODE)")
    print(f"{'='*80}")
    
    if not os.path.exists(shapefile_path):
        print(f"❌ ERROR: Shapefile not found: {shapefile_path}")
        return None
    
    print(f"📍 Found shapefile: {shapefile_path}")
    
    # Extract extent from shapefile for regular variables
    area = get_extent_from_shapefile(shapefile_path, buffer_degrees)
    if area is None:
        print(f"❌ ERROR: Could not extract extent from shapefile")
        return None
    
    print(f"🗺️ Regular variables area (buffer={buffer_degrees}°): {area}")
    
    # Extract extent for geopotential with larger buffer
    geo_area = get_extent_from_shapefile(shapefile_path, geo_buffer_degrees)
    if geo_area is None:
        print(f"❌ ERROR: Could not extract extent for geopotential")
        return None
    
    print(f"🏔️ Geopotential area (buffer={geo_buffer_degrees}°): {geo_area}")
    
    # Create gauge-specific output directory
    gauge_output_dir = base_output_dir
    os.makedirs(gauge_output_dir, exist_ok=True)
    print(f"📁 Output directory: {gauge_output_dir}")
    
    # Initialize results dictionary
    results = {}
    
    # Handle geopotential separately (only download once with larger buffer)
    if 'geopotential' in variables:
        print(f"\n🏔️ Downloading geopotential (elevation) data with larger buffer ({geo_buffer_degrees}°)...")
        geopotential_file = download_geopotential_once(gauge_output_dir, geo_area)
        results['geopotential'] = [geopotential_file] if geopotential_file else []
        
        # Remove geopotential from variables list for regular processing
        variables = [var for var in variables if var != 'geopotential']
    
    # Process remaining variables normally (time-series data) with regular buffer
    for variable in variables:
        print(f"\n🌍 Processing variable: {variable} with buffer {buffer_degrees}°")
        
        downloaded_files = download_multiple_months_parallel(
            variable=variable,
            years=years, 
            months=months,
            output_dir=gauge_output_dir,
            area=area,  # Use regular area for other variables
            max_workers=max_workers
        )
        
        results[variable] = downloaded_files
        
        if downloaded_files:
            print(f"✅ {variable}: {len(downloaded_files)} files downloaded")
        else:
            print(f"❌ {variable}: No files downloaded")
    
    # Print final summary
    print(f"\n📊 FINAL SUMMARY FOR GAUGE {gauge_id}:")
    print(f"   Buffer settings:")
    print(f"     Regular variables: {buffer_degrees}°")
    print(f"     Geopotential: {geo_buffer_degrees}°")
    print(f"   Downloaded files:")
    total_files = 0
    for variable, files in results.items():
        file_count = len(files) if files else 0
        total_files += file_count
        print(f"     {variable}: {file_count} files")
    
    print(f"   TOTAL: {total_files} files downloaded")
    
    return results


def process_gauge_ultra_parallel(gauge_id, years, months, variables, shapefile_path, base_output_dir, buffer_degrees=0.1, max_workers=6):
    """
    Download ERA5-Land data with maximum parallelization (across variables AND months)
    
    WARNING: Use with caution - too many concurrent requests may hit API limits
    """
    print(f"\n{'='*80}")
    print(f"🎯 PROCESSING GAUGE {gauge_id} (ULTRA PARALLEL MODE)")
    print(f"{'='*80}")
    
    if not os.path.exists(shapefile_path):
        print(f"❌ ERROR: Shapefile not found: {shapefile_path}")
        return None
    
    area = get_extent_from_shapefile(shapefile_path, buffer_degrees)
    if area is None:
        return None
        
    gauge_output_dir = base_output_dir
    os.makedirs(gauge_output_dir, exist_ok=True)
    
    # Create all combinations of (variable, year, month)
    all_combinations = list(product(variables, years, months))
    total_downloads = len(all_combinations)
    
    print(f"📋 Total downloads: {total_downloads}")
    print(f"   Variables: {len(variables)}")
    print(f"   Years: {len(years)}")
    print(f"   Months: {len(months)}")
    print(f"   Max workers: {max_workers}")
    
    results = {var: [] for var in variables}
    
    # Submit all downloads at once
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {
            executor.submit(download_one_month, variable, year, month, gauge_output_dir, area): (variable, year, month)
            for variable, year, month in all_combinations
        }
        
        completed = 0
        for future in as_completed(future_to_params):
            variable, year, month = future_to_params[future]
            completed += 1
            
            try:
                file_path = future.result()
                if file_path:
                    results[variable].append(file_path)
                    print(f"✅ [{completed}/{total_downloads}] {variable} {year}-{month:02d}")
                else:
                    print(f"❌ [{completed}/{total_downloads}] Failed: {variable} {year}-{month:02d}")
                    
            except Exception as e:
                print(f"❌ [{completed}/{total_downloads}] Error {variable} {year}-{month:02d}: {str(e)}")
    
    return results


if __name__ == "__main__":
    # Read parameters from namelist.yaml
    namelist_path = "/home/jberg/OneDrive/Raven-world/namelist.yaml"
    gauge_id, years, shapefile_path, meteo_output_dir = read_namelist(namelist_path)
    
    # Configuration
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    variables = [
        'geopotential',
        '2m_temperature',
        'total_precipitation',
        'evaporation_from_vegetation_transpiration',
        'surface_solar_radiation_downwards',
        'surface_pressure',
        'total_evaporation',
    ]
    
    # Choose parallel mode
    max_workers = 4  # Start with 4, can increase if stable
    
    print(f"\n{'='*80}")
    print(f"ERA5-LAND DOWNLOAD CONFIGURATION")
    print(f"{'='*80}")
    print(f"Gauge ID: {gauge_id}")
    print(f"Years: {years}")
    print(f"Months: {months}")
    print(f"Shapefile: {shapefile_path}")
    print(f"Output directory: {meteo_output_dir}")
    print(f"Variables: {len(variables)}")
    for var in variables:
        print(f"  - {var}")
    
    print(f"\n🚀 Starting ERA5-Land download (PARALLEL)...")
    
    # Option 1: Moderate parallelization (recommended)
    results = process_gauge_parallel(
        gauge_id=gauge_id,
        years=years,
        months=months,
        variables=variables,
        shapefile_path=shapefile_path,
        base_output_dir=meteo_output_dir,
        max_workers=max_workers
    )
    
    # Option 2: Ultra parallelization (use with caution)
    # results = process_gauge_ultra_parallel(
    #     gauge_id=gauge_id,
    #     years=years,
    #     months=months,
    #     variables=variables,
    #     shapefile_path=shapefile_path,
    #     base_output_dir=meteo_output_dir,
    #     max_workers=6  # Be careful not to exceed API limits
    # )
    
    if results:
        print(f"\n🎉 Parallel download complete for gauge {gauge_id}!")
        for variable, files in results.items():
            print(f"   {variable}: {len(files)} files")
    else:
        print(f"\n❌ Download failed for gauge {gauge_id}")