import os
import requests
import rasterio
import rasterio.mask
import geopandas as gpd
import tempfile
import shutil
import traceback
import elevation

def get_extent_from_shapefile(shapefile_path, buffer_degrees=0.01):
    """
    Read shapefile and extract bounding box extent
    
    Parameters:
    -----------
    shapefile_path : str
        Path to the shapefile
    buffer_degrees : float
        Buffer to add around the shapefile extent in degrees
    
    Returns:
    --------
    tuple
        (minx, miny, maxx, maxy) coordinates in WGS84
    """
    try:
        # Read shapefile
        gdf = gpd.read_file(shapefile_path)
        
        # Convert to WGS84 (EPSG:4326) if not already
        if gdf.crs != 'EPSG:4326':
            print(f"Converting from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs('EPSG:4326')
        
        # Get bounding box and add buffer
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        minx, miny, maxx, maxy = bounds
        minx -= buffer_degrees
        miny -= buffer_degrees
        maxx += buffer_degrees
        maxy += buffer_degrees
        
        print(f"Shapefile extent: West={minx:.3f}, South={miny:.3f}, East={maxx:.3f}, North={maxy:.3f}")
        print(f"Buffer applied: {buffer_degrees}°")
        
        return (minx, miny, maxx, maxy)
        
    except Exception as e:
        print(f"Error reading shapefile: {str(e)}")
        traceback.print_exc()
        return None

def download_with_elevation_package(bounds, output_path):
    """
    Use the elevation package to download SRTM data
    
    Parameters:
    -----------
    bounds : tuple
        (minx, miny, maxx, maxy) in WGS84
    output_path : str
        Path for output DEM file
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        print("Downloading SRTM data using elevation package...")
        
        # Download using elevation package
        elevation.clip(bounds=bounds, output=output_path, product='SRTM1')
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            print(f"Successfully downloaded DEM: {output_path}")
            return True
        else:
            print("Elevation package download failed")
            return False
            
    except Exception as e:
        print(f"Error with elevation package: {str(e)}")
        return False

def download_opentopography_dem(bounds, output_path):
    """
    Download DEM from OpenTopography as backup
    
    Parameters:
    -----------
    bounds : tuple
        (minx, miny, maxx, maxy) in WGS84
    output_path : str
        Path for output file
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        print("Trying OpenTopography SRTM as backup...")
        
        minx, miny, maxx, maxy = bounds
        
        # OpenTopography API for SRTM 1 arc-second
        url = "https://portal.opentopography.org/API/globaldem"
        params = {
            'demtype': 'SRTM_GL1',
            'south': miny,
            'north': maxy,
            'west': minx,
            'east': maxx,
            'outputFormat': 'GTiff',
            'API_Key': 'demoapikeyot2022'
        }
        
        response = requests.get(url, params=params, timeout=120)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            # Verify it's a valid GeoTIFF
            try:
                with rasterio.open(output_path) as src:
                    if src.width > 0 and src.height > 0:
                        print(f"Successfully downloaded DEM from OpenTopography")
                        return True
            except:
                pass
        
        print(f"OpenTopography download failed: HTTP {response.status_code}")
        return False
        
    except Exception as e:
        print(f"Error downloading from OpenTopography: {str(e)}")
        return False

def download_dem(bounds, output_path):
    """
    Try multiple DEM sources in order of reliability
    
    Parameters:
    -----------
    bounds : tuple
        Bounding box coordinates (minx, miny, maxx, maxy)
    output_path : str
        Output file path
    
    Returns:
    --------
    bool
        True if any source succeeded, False otherwise
    """
    # Try elevation package first (most reliable)
    if download_with_elevation_package(bounds, output_path):
        return True
    
    # Try OpenTopography as backup
    if download_opentopography_dem(bounds, output_path):
        return True
    
    print("ERROR: All DEM sources failed")
    return False

def clip_raster_to_shapefile(raster_path, shapefile_path, output_path):
    """
    Clip raster to shapefile extent
    
    Parameters:
    -----------
    raster_path : str
        Path to input raster
    shapefile_path : str
        Path to shapefile for clipping
    output_path : str
        Path for clipped output raster
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        print("Clipping DEM to catchment extent...")
        
        # Read shapefile
        gdf = gpd.read_file(shapefile_path)
        
        # Clip raster to shapefile
        with rasterio.open(raster_path) as src:
            # Ensure CRS matches
            if gdf.crs != src.crs:
                print(f"Reprojecting shapefile from {gdf.crs} to {src.crs}")
                gdf = gdf.to_crs(src.crs)
            
            # Perform clipping
            out_image, out_transform = rasterio.mask.mask(src, gdf.geometry, crop=True)
            out_meta = src.meta.copy()
            
            # Update metadata
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw"
            })
        
        # Write clipped raster
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
        
        print(f"Successfully clipped DEM")
        return True
        
    except Exception as e:
        print(f"Error clipping raster: {str(e)}")
        traceback.print_exc()
        return False

def download_srtm_for_catchment(gauge_id, shapefile_dir, output_dir, buffer_degrees=0.01):
    """
    Download and process SRTM DEM for a specific catchment
    
    Parameters:
    -----------
    gauge_id : str
        Gauge ID (e.g., '0001')
    shapefile_dir : str
        Directory containing shapefiles
    output_dir : str
        Output directory for final DEM
    buffer_degrees : float
        Buffer around shapefile extent in degrees
    
    Returns:
    --------
    str or None
        Path to final clipped DEM or None if failed
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING DEM FOR GAUGE {gauge_id}")
    print(f"{'='*80}")
    
    # Find shapefile
    shapefile_path = os.path.join(shapefile_dir, f"shape_{gauge_id}.shp")
    
    if not os.path.exists(shapefile_path):
        print(f"ERROR: Shapefile not found: {shapefile_path}")
        return None
    
    print(f"Found shapefile: {shapefile_path}")
    
    # Extract extent from shapefile
    bounds = get_extent_from_shapefile(shapefile_path, buffer_degrees)
    if bounds is None:
        print("ERROR: Could not extract extent from shapefile")
        return None
    
    # Setup directories
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix=f"srtm_{gauge_id}_")
    
    try:
        # Download DEM for the area
        temp_dem_path = os.path.join(temp_dir, f"full_dem_{gauge_id}.tif")
        
        if download_dem(bounds, temp_dem_path):
            # Clip to catchment extent
            final_dem_path = os.path.join(output_dir, f"dem_{gauge_id}.tif")
            
            if clip_raster_to_shapefile(temp_dem_path, shapefile_path, final_dem_path):
                print(f"✅ Successfully created DEM: {final_dem_path}")
                return final_dem_path
            else:
                print("ERROR: Failed to clip DEM to catchment")
                return None
        else:
            print("ERROR: Failed to download DEM")
            return None
            
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory")

def print_dem_info(dem_path):
    """
    Print information about the DEM file
    
    Parameters:
    -----------
    dem_path : str
        Path to DEM file
    """
    try:
        with rasterio.open(dem_path) as src:
            print(f"\nDEM Info:")
            print(f"- Size: {src.width} x {src.height} pixels")
            print(f"- Resolution: {src.res[0]:.6f}° x {src.res[1]:.6f}°")
            print(f"- CRS: {src.crs}")
            print(f"- Bounds: {src.bounds}")
            
            # Read elevation stats
            data = src.read(1, masked=True)
            if not data.mask.all():
                print(f"- Elevation range: {data.min():.1f} to {data.max():.1f} meters")
                print(f"- Mean elevation: {data.mean():.1f} meters")
    except Exception as e:
        print(f"Could not read DEM info: {e}")

if __name__ == "__main__":
    # Parameters
    gauge_id = "0001"  # Change this to your gauge ID
    
    # Directories
    shapefile_dir = "/home/jberg/OneDrive/Raven_worldwide/01_data/topo/catchment_shapefile"
    output_dir = "/home/jberg/OneDrive/Raven_worldwide/01_data/topo/catchment_dem"
    
    # Buffer around shapefile extent (in degrees)
    buffer_degrees = 0.01  # ~1 km at equator
    
    # Process the gauge
    dem_path = download_srtm_for_catchment(
        gauge_id=gauge_id,
        shapefile_dir=shapefile_dir,
        output_dir=output_dir,
        buffer_degrees=buffer_degrees
    )
    
    if dem_path:
        print(f"\n🎉 DEM processing complete for gauge {gauge_id}!")
        print(f"DEM saved as: {dem_path}")
        print_dem_info(dem_path)
    else:
        print(f"\n❌ DEM processing failed for gauge {gauge_id}")