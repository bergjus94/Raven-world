"""
Unified data download script for catchment modeling
Downloads all necessary forcing data: DEM (SRTM), Land Use (ESA WorldCover), and Meteorological data (ERA5-Land)

Usage:
    python download_forcing.py <path_to_namelist.yaml>
"""

import os
import sys
import yaml
import traceback
from pathlib import Path

# Import functions from individual download scripts
from download_SRTM import (
    read_namelist as read_namelist_srtm,
    get_extent_from_shapefile as get_extent_srtm,
    download_dem,
    clip_raster_to_shapefile,
    print_dem_info
)

from download_esa_worldcover import (
    read_namelist as read_namelist_esa,
    get_extent_from_shapefile as get_extent_esa,
    download_esa_worldcover_direct,
    clip_raster_to_shapefile as clip_esa,
    aggregate_to_30m_mode_parallel,
    print_landuse_info
)

from download_era5land import (
    read_namelist as read_namelist_era5,
    get_extent_from_shapefile as get_extent_era5,
    process_gauge_parallel
)

import tempfile
import shutil


def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")


def download_srtm_data(namelist_path, buffer_degrees=0.01):
    """
    Download and process SRTM DEM data
    
    Parameters:
    -----------
    namelist_path : str
        Path to namelist.yaml
    buffer_degrees : float
        Buffer around shapefile extent in degrees
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    print_header("STEP 1: DOWNLOADING SRTM DEM DATA")
    
    try:
        # Read namelist
        gauge_id, shapefile_path, dem_path = read_namelist_srtm(namelist_path)
        
        print(f"Gauge ID: {gauge_id}")
        print(f"Shapefile: {shapefile_path}")
        print(f"DEM output: {dem_path}")
        
        # Check if DEM already exists
        if os.path.exists(dem_path) and os.path.getsize(dem_path) > 1000:
            print(f"✅ DEM already exists: {dem_path}")
            print_dem_info(dem_path)
            return True
        
        # Create output directory
        os.makedirs(os.path.dirname(dem_path), exist_ok=True)
        
        # Extract extent from shapefile
        bounds = get_extent_srtm(shapefile_path, buffer_degrees)
        if bounds is None:
            print("❌ ERROR: Could not extract extent from shapefile")
            return False
        
        # Download and clip DEM
        temp_dir = tempfile.mkdtemp(prefix=f"srtm_{gauge_id}_")
        temp_dem_path = os.path.join(temp_dir, f"full_dem_{gauge_id}.tif")
        
        try:
            if download_dem(bounds, temp_dem_path):
                if clip_raster_to_shapefile(temp_dem_path, shapefile_path, dem_path):
                    print(f"✅ Successfully created DEM: {dem_path}")
                    print_dem_info(dem_path)
                    return True
                else:
                    print("❌ ERROR: Failed to clip DEM to catchment")
                    return False
            else:
                print("❌ ERROR: Failed to download DEM")
                return False
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory")
                
    except Exception as e:
        print(f"❌ ERROR in SRTM download: {str(e)}")
        traceback.print_exc()
        return False


def download_esa_data(namelist_path, buffer_degrees=0.01):
    """
    Download and process ESA WorldCover land use data
    
    Parameters:
    -----------
    namelist_path : str
        Path to namelist.yaml
    buffer_degrees : float
        Buffer around shapefile extent in degrees
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    print_header("STEP 2: DOWNLOADING ESA WORLDCOVER LAND USE DATA")
    
    try:
        # Read namelist
        gauge_id, shapefile_path, landuse_path = read_namelist_esa(namelist_path)
        
        print(f"Gauge ID: {gauge_id}")
        print(f"Shapefile: {shapefile_path}")
        print(f"Land use output: {landuse_path}")
        
        # Check if shapefile exists
        if not os.path.exists(shapefile_path):
            print(f"❌ ERROR: Shapefile not found: {shapefile_path}")
            return False
        
        # Check if landuse already exists
        if os.path.exists(landuse_path) and os.path.getsize(landuse_path) > 1000:
            print(f"✅ Land use raster already exists: {landuse_path}")
            print_landuse_info(landuse_path)
            return True
        
        # Create output directory
        os.makedirs(os.path.dirname(landuse_path), exist_ok=True)
        
        # Extract extent from shapefile
        bounds = get_extent_esa(shapefile_path, buffer_degrees)
        if bounds is None:
            print("❌ ERROR: Could not extract extent from shapefile")
            return False
        
        # Download, clip, and aggregate
        temp_dir = tempfile.mkdtemp(prefix=f"esa_{gauge_id}_")
        
        try:
            # Step 1: Download ESA WorldCover tile
            temp_worldcover_path = os.path.join(temp_dir, f"worldcover_full_{gauge_id}.tif")
            
            if not download_esa_worldcover_direct(bounds, temp_worldcover_path):
                print("❌ ERROR: Failed to download ESA WorldCover")
                return False
            
            # Step 2: Clip to catchment extent
            clipped_path = os.path.join(temp_dir, f"worldcover_clipped_{gauge_id}.tif")
            
            if not clip_esa(temp_worldcover_path, shapefile_path, clipped_path):
                print("❌ ERROR: Failed to clip ESA WorldCover to catchment")
                return False
            
            # Step 3: Aggregate to 30m
            if aggregate_to_30m_mode_parallel(clipped_path, landuse_path):
                print(f"✅ Successfully created land use raster: {landuse_path}")
                print_landuse_info(landuse_path)
                return True
            else:
                print("❌ ERROR: Failed to aggregate to 30m")
                return False
                
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory")
                
    except Exception as e:
        print(f"❌ ERROR in ESA WorldCover download: {str(e)}")
        traceback.print_exc()
        return False


def download_era5_data(namelist_path, buffer_degrees=0.1, geo_buffer_degrees=0.2, max_workers=4):
    """
    Download ERA5-Land meteorological data
    
    Parameters:
    -----------
    namelist_path : str
        Path to namelist.yaml
    buffer_degrees : float
        Buffer around shapefile extent in degrees for regular variables
    geo_buffer_degrees : float
        Buffer around shapefile extent in degrees for geopotential
    max_workers : int
        Maximum number of parallel downloads
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    print_header("STEP 3: DOWNLOADING ERA5-LAND METEOROLOGICAL DATA")
    
    try:
        # Read namelist
        gauge_id, years, shapefile_path, meteo_output_dir = read_namelist_era5(namelist_path)
        
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
        
        print(f"Gauge ID: {gauge_id}")
        print(f"Years: {years}")
        print(f"Shapefile: {shapefile_path}")
        print(f"Output directory: {meteo_output_dir}")
        print(f"Variables: {len(variables)}")
        for var in variables:
            print(f"  - {var}")
        
        # Create output directory
        os.makedirs(meteo_output_dir, exist_ok=True)
        
        # Download data
        print(f"\n🚀 Starting ERA5-Land download (PARALLEL)...")
        
        results = process_gauge_parallel(
            gauge_id=gauge_id,
            years=years,
            months=months,
            variables=variables,
            shapefile_path=shapefile_path,
            base_output_dir=meteo_output_dir,
            buffer_degrees=buffer_degrees,
            geo_buffer_degrees=geo_buffer_degrees,
            max_workers=max_workers
        )
        
        if results:
            print(f"\n✅ ERA5-Land download complete for gauge {gauge_id}!")
            total_files = sum(len(files) for files in results.values())
            print(f"   Total files downloaded: {total_files}")
            for variable, files in results.items():
                print(f"   {variable}: {len(files)} files")
            return True
        else:
            print(f"\n❌ ERA5-Land download failed for gauge {gauge_id}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR in ERA5-Land download: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """
    Main execution function - downloads all required data
    """
    print_header("UNIFIED DATA DOWNLOAD SCRIPT FOR CATCHMENT MODELING")
    
    # Parse command line arguments
    if len(sys.argv) != 2:
        print("❌ ERROR: Invalid number of arguments")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <path_to_namelist.yaml>")
        print("\nExample:")
        print(f"  python {sys.argv[0]} /home/jberg/OneDrive/Raven-world/namelist.yaml")
        sys.exit(1)
    
    namelist_path = sys.argv[1]
    
    # Check if namelist exists
    if not os.path.exists(namelist_path):
        print(f"❌ ERROR: Namelist file not found: {namelist_path}")
        sys.exit(1)
    
    print(f"📄 Using namelist: {namelist_path}")
    
    # Load namelist to display configuration
    with open(namelist_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nConfiguration:")
    print(f"  Gauge ID: {config.get('gauge_id')}")
    print(f"  Start date: {config.get('start_date')}")
    print(f"  End date: {config.get('end_date')}")
    print(f"  Main directory: {config.get('main_dir')}")
    
    # Track success of each step
    results = {
        'SRTM DEM': False,
        'ESA WorldCover': False,
        'ERA5-Land': False
    }
    
    # Step 1: Download SRTM DEM
    results['SRTM DEM'] = download_srtm_data(namelist_path, buffer_degrees=0.01)
    
    # Step 2: Download ESA WorldCover
    results['ESA WorldCover'] = download_esa_data(namelist_path, buffer_degrees=0.01)
    
    # Step 3: Download ERA5-Land
    results['ERA5-Land'] = download_era5_data(
        namelist_path, 
        buffer_degrees=0.1, 
        geo_buffer_degrees=0.2, 
        max_workers=4
    )
    
    # Print final summary
    print_header("DOWNLOAD SUMMARY")
    
    all_successful = True
    for dataset, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{dataset:.<50} {status}")
        if not success:
            all_successful = False
    
    print(f"\n{'='*80}")
    if all_successful:
        print("🎉 ALL DOWNLOADS COMPLETED SUCCESSFULLY!")
    else:
        print("⚠️  SOME DOWNLOADS FAILED - CHECK LOGS ABOVE")
    print(f"{'='*80}\n")
    
    return 0 if all_successful else 1


if __name__ == "__main__":
    sys.exit(main())