import cdsapi
import os
import traceback
import zipfile
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import time


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

def process_gauge_parallel(gauge_id, years, months, variables, shapefile_dir, base_output_dir, buffer_degrees=0.1, max_workers=4):
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
    shapefile_dir : str
        Directory containing gauge shapefiles
    base_output_dir : str
        Base directory for output files
    buffer_degrees : float
        Buffer around shapefile extent in degrees
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
    
    # Find shapefile
    shapefile_path = os.path.join(shapefile_dir, f"catchment_shape_{gauge_id}.shp")
    
    if not os.path.exists(shapefile_path):
        print(f"❌ ERROR: Shapefile not found: {shapefile_path}")
        return None
    
    print(f"📍 Found shapefile: {shapefile_path}")
    
    # Extract extent from shapefile
    area = get_extent_from_shapefile(shapefile_path, buffer_degrees)
    if area is None:
        print(f"❌ ERROR: Could not extract extent from shapefile")
        return None
    
    # Create gauge-specific output directory
    gauge_output_dir = os.path.join(base_output_dir, f"gauge_{gauge_id}")
    os.makedirs(gauge_output_dir, exist_ok=True)
    print(f"📁 Output directory: {gauge_output_dir}")
    
    # Download data for each variable (can also parallelize across variables)
    results = {}
    
    # Option 1: Process variables sequentially, but months in parallel
    for variable in variables:
        print(f"\n🌍 Processing variable: {variable}")
        
        downloaded_files = download_multiple_months_parallel(
            variable=variable,
            years=years, 
            months=months,
            output_dir=gauge_output_dir,
            area=area,
            max_workers=max_workers
        )
        
        results[variable] = downloaded_files
        
        if downloaded_files:
            print(f"✅ {variable}: {len(downloaded_files)} files downloaded")
        else:
            print(f"❌ {variable}: No files downloaded")
    
    return results

def process_gauge_ultra_parallel(gauge_id, years, months, variables, shapefile_dir, base_output_dir, buffer_degrees=0.1, max_workers=6):
    """
    Download ERA5-Land data with maximum parallelization (across variables AND months)
    
    WARNING: Use with caution - too many concurrent requests may hit API limits
    """
    print(f"\n{'='*80}")
    print(f"🎯 PROCESSING GAUGE {gauge_id} (ULTRA PARALLEL MODE)")
    print(f"{'='*80}")
    
    # ... (same setup as above)
    
    # Find shapefile and extract area (same as above)
    shapefile_path = os.path.join(shapefile_dir, f"catchment_shape_{gauge_id}.shp")
    if not os.path.exists(shapefile_path):
        print(f"❌ ERROR: Shapefile not found: {shapefile_path}")
        return None
    
    area = get_extent_from_shapefile(shapefile_path, buffer_degrees)
    if area is None:
        return None
        
    gauge_output_dir = os.path.join(base_output_dir, f"gauge_{gauge_id}")
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
    # Configuration
    gauge_id = "0670"
    years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]  # Reduced for testing
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    variables = [
        'total_precipitation',
        '2m_temperature'
    ]
    
    # Directories
    shapefile_dir = "/home/jberg/OneDrive/Raven_worldwide/01_data/topo/catchment_shapefile"
    base_output_dir = "/run/user/1001/gvfs/smb-share:server=fileserv02.giub.unibe.ch,share=userdata/jberg/Meteo_data/Era5_worldwide"
    
    # Choose parallel mode
    max_workers = 4  # Start with 4, can increase if stable
    
    print("🚀 Starting ERA5-Land download (PARALLEL)...")
    
    # Option 1: Moderate parallelization (recommended)
    results = process_gauge_parallel(
        gauge_id=gauge_id,
        years=years,
        months=months,
        variables=variables,
        shapefile_dir=shapefile_dir,
        base_output_dir=base_output_dir,
        max_workers=max_workers
    )
    
    # Option 2: Ultra parallelization (use with caution)
    # results = process_gauge_ultra_parallel(
    #     gauge_id=gauge_id,
    #     years=years,
    #     months=months,
    #     variables=variables,
    #     shapefile_dir=shapefile_dir,
    #     base_output_dir=base_output_dir,
    #     max_workers=6  # Be careful not to exceed API limits
    # )
    
    if results:
        print(f"\n🎉 Parallel download complete for gauge {gauge_id}!")
        for variable, files in results.items():
            print(f"   {variable}: {len(files)} files")
    else:
        print(f"\n❌ Download failed for gauge {gauge_id}")