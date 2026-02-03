import os
import requests
import rasterio
import rasterio.mask
import rasterio.merge
import geopandas as gpd
import tempfile
import shutil
import traceback
import elevation
import yaml
import math
import time
from shapely.geometry import box

def read_namelist(namelist_path):
    """Read gauge_id, shape_dir, raster_dir from namelist.yaml"""
    with open(namelist_path, 'r') as f:
        config = yaml.safe_load(f)
    gauge_id = str(config.get('gauge_id'))
    main_dir = config.get('main_dir', '')
    shape_dir = config.get('shape_dir')
    raster_dir = config.get('raster_dir')
    # Build absolute paths if needed
    shape_path = shape_dir.format(gauge_id=gauge_id)
    raster_path = raster_dir.format(gauge_id=gauge_id)
    if not os.path.isabs(shape_path):
        shape_path = os.path.join(main_dir, shape_path)
    if not os.path.isabs(raster_path):
        raster_path = os.path.join(main_dir, raster_path)
    return gauge_id, shape_path, raster_path

def get_extent_from_shapefile(shapefile_path, buffer_degrees=0.01):
    try:
        gdf = gpd.read_file(shapefile_path)
        if gdf.crs != 'EPSG:4326':
            print(f"Converting from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs('EPSG:4326')
        bounds = gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        minx -= buffer_degrees
        miny -= buffer_degrees
        maxx += buffer_degrees
        maxy += buffer_degrees
        print(f"Shapefile extent: West={minx:.3f}, South={miny:.3f}, East={maxx:.3f}, North={maxy:.3f}")
        print(f"Buffer applied: {buffer_degrees}°")
        
        # Calculate size
        width_deg = maxx - minx
        height_deg = maxy - miny
        print(f"Coverage: {width_deg:.2f}° x {height_deg:.2f}°")
        
        return (minx, miny, maxx, maxy)
    except Exception as e:
        print(f"Error reading shapefile: {str(e)}")
        traceback.print_exc()
        return None

def split_bounds_into_tiles(bounds, max_tile_size=1.5):
    """
    Split large bounds into smaller tiles that won't hit API limits.
    Returns list of (minx, miny, maxx, maxy) tuples.
    """
    minx, miny, maxx, maxy = bounds
    width_deg = maxx - minx
    height_deg = maxy - miny
    
    # Calculate number of tiles needed
    n_tiles_x = max(1, int(math.ceil(width_deg / max_tile_size)))
    n_tiles_y = max(1, int(math.ceil(height_deg / max_tile_size)))
    
    print(f"Splitting area into {n_tiles_x} x {n_tiles_y} = {n_tiles_x * n_tiles_y} tiles")
    print(f"Each tile will be ~{width_deg/n_tiles_x:.2f}° x {height_deg/n_tiles_y:.2f}°")
    
    tiles = []
    for i in range(n_tiles_x):
        for j in range(n_tiles_y):
            tile_minx = minx + i * (width_deg / n_tiles_x)
            tile_maxx = min(maxx, minx + (i + 1) * (width_deg / n_tiles_x))
            tile_miny = miny + j * (height_deg / n_tiles_y)
            tile_maxy = min(maxy, miny + (j + 1) * (height_deg / n_tiles_y))
            
            tiles.append((tile_minx, tile_miny, tile_maxx, tile_maxy))
    
    return tiles

def download_tile_with_elevation_package(bounds, output_path, tile_id):
    """
    Download a single tile using elevation package (no API key needed!).
    """
    try:
        minx, miny, maxx, maxy = bounds
        width_deg = maxx - minx
        height_deg = maxy - miny
        
        print(f"  Tile {tile_id}: Downloading {width_deg:.2f}° x {height_deg:.2f}° area...")
        
        elevation.clip(bounds=bounds, output=output_path, product='SRTM1')
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            # Validate the tile
            with rasterio.open(output_path) as src:
                if src.width > 0 and src.height > 0:
                    print(f"    ✓ Tile {tile_id} downloaded ({src.width}x{src.height} pixels)")
                    return True
        
        print(f"    ✗ Tile {tile_id} failed")
        return False
        
    except Exception as e:
        print(f"    ✗ Tile {tile_id} error: {str(e)}")
        return False

def download_dem_tiled_elevation_package(bounds, output_path, max_tile_size=1.5):
    """
    Download DEM by splitting into tiles and using elevation package (no API limits!).
    """
    try:
        minx, miny, maxx, maxy = bounds
        width_deg = maxx - minx
        height_deg = maxy - miny
        
        # Check if we need to tile
        if width_deg <= max_tile_size and height_deg <= max_tile_size:
            # Small enough - download in one piece
            print("Downloading SRTM using elevation package (single piece)...")
            elevation.clip(bounds=bounds, output=output_path, product='SRTM1')
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                print(f"✅ Successfully downloaded DEM with elevation package")
                return True
            else:
                print("Elevation package download failed")
                return False
        
        else:
            # Too large - split into tiles
            print(f"Area too large ({width_deg:.2f}° x {height_deg:.2f}°)")
            print(f"Using tiled approach with elevation package (no API limits!)...")
            
            # Split into tiles
            tiles = split_bounds_into_tiles(bounds, max_tile_size=max_tile_size)
            
            tile_files = []
            temp_dir = tempfile.mkdtemp(prefix="elevation_tiles_")
            
            try:
                # Download each tile
                for idx, tile_bounds in enumerate(tiles, 1):
                    tile_output = os.path.join(temp_dir, f"tile_{idx}.tif")
                    
                    if download_tile_with_elevation_package(tile_bounds, tile_output, idx):
                        tile_files.append(tile_output)
                    
                    # Small delay between downloads to be nice to the server
                    if idx < len(tiles):
                        time.sleep(2)
                
                # Check if we got all tiles
                if len(tile_files) == 0:
                    print("❌ No tiles downloaded successfully")
                    return False
                
                print(f"\n  Successfully downloaded {len(tile_files)}/{len(tiles)} tiles")
                
                if len(tile_files) < len(tiles):
                    print(f"  ⚠️  Warning: {len(tiles) - len(tile_files)} tiles failed")
                
                # Merge tiles
                if len(tile_files) == 1:
                    print("  Single tile, copying...")
                    shutil.copy(tile_files[0], output_path)
                else:
                    print(f"  Merging {len(tile_files)} tiles...")
                    
                    src_files_to_mosaic = []
                    for tile_file in tile_files:
                        src = rasterio.open(tile_file)
                        src_files_to_mosaic.append(src)
                    
                    # Merge tiles
                    mosaic, out_trans = rasterio.merge.merge(src_files_to_mosaic)
                    
                    # Get metadata from first tile
                    out_meta = src_files_to_mosaic[0].meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": out_trans,
                        "compress": "lzw"
                    })
                    
                    # Write merged raster
                    with rasterio.open(output_path, "w", **out_meta) as dest:
                        dest.write(mosaic)
                    
                    # Close all source files
                    for src in src_files_to_mosaic:
                        src.close()
                
                print(f"✅ Successfully downloaded and merged DEM using elevation package")
                return True
                
            finally:
                # Clean up temp directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    
    except Exception as e:
        print(f"Error with tiled elevation download: {str(e)}")
        traceback.print_exc()
        return False

def download_dem(bounds, output_path):
    """
    Download DEM using tiled elevation package approach (no API limits!).
    """
    # Use tiled elevation package approach (works for any size!)
    if download_dem_tiled_elevation_package(bounds, output_path, max_tile_size=1.5):
        return True
    
    print("❌ ERROR: DEM download failed")
    return False

def clip_raster_to_shapefile(raster_path, shapefile_path, output_path):
    try:
        print("Clipping DEM to catchment extent...")
        gdf = gpd.read_file(shapefile_path)
        with rasterio.open(raster_path) as src:
            if gdf.crs != src.crs:
                print(f"Reprojecting shapefile from {gdf.crs} to {src.crs}")
                gdf = gdf.to_crs(src.crs)
            out_image, out_transform = rasterio.mask.mask(src, gdf.geometry, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw"
            })
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
        print(f"✅ Successfully clipped DEM")
        return True
    except Exception as e:
        print(f"❌ Error clipping raster: {str(e)}")
        traceback.print_exc()
        return False

def print_dem_info(dem_path):
    try:
        with rasterio.open(dem_path) as src:
            print(f"\nDEM Info:")
            print(f"- Size: {src.width} x {src.height} pixels")
            print(f"- Resolution: {src.res[0]:.6f}° x {src.res[1]:.6f}°")
            print(f"- CRS: {src.crs}")
            print(f"- Bounds: {src.bounds}")
            data = src.read(1, masked=True)
            if not data.mask.all():
                print(f"- Elevation range: {data.min():.1f} to {data.max():.1f} meters")
                print(f"- Mean elevation: {data.mean():.1f} meters")
            file_size_mb = os.path.getsize(dem_path) / (1024 * 1024)
            print(f"- File size: {file_size_mb:.1f} MB")
    except Exception as e:
        print(f"Could not read DEM info: {e}")

def download_upper_indus_dem(output_dir):
    """
    Download DEM for the entire Upper Indus Basin
    
    Parameters:
    -----------
    output_dir : str
        Directory where to save the DEM
    
    Returns:
    --------
    str or None
        Path to the downloaded DEM, or None if failed
    """
    # Upper Indus Basin bounding box
    upper_indus_bbox = (
        71.0,  # West
        30.5,  # South
        82.0,  # East
        37.5   # North
    )
    
    minx, miny, maxx, maxy = upper_indus_bbox
    width_deg = maxx - minx
    height_deg = maxy - miny
    
    print("\n" + "="*70)
    print("DOWNLOADING UPPER INDUS BASIN DEM")
    print("="*70)
    print(f"\nBounding Box:")
    print(f"  West:  {minx}° (Longitude)")
    print(f"  South: {miny}° (Latitude)")
    print(f"  East:  {maxx}° (Longitude)")
    print(f"  North: {maxy}° (Latitude)")
    print(f"\nCoverage: {width_deg:.1f}° x {height_deg:.1f}° (~{width_deg*111:.0f} x {height_deg*111:.0f} km)")
    print(f"Area: ~{width_deg*111 * height_deg*111:,.0f} km²")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output path
    dem_output = os.path.join(output_dir, "dem_Indus.tif")
    
    # Check if already exists
    if os.path.exists(dem_output) and os.path.getsize(dem_output) > 1000:
        print(f"\n✅ DEM already exists: {dem_output}")
        print_dem_info(dem_output)
        return dem_output
    
    print(f"\nOutput: {dem_output}")
    print("\n" + "-"*70)
    
    # Download
    if download_dem(upper_indus_bbox, dem_output):
        print("\n" + "="*70)
        print("✅ SUCCESS - UPPER INDUS DEM DOWNLOADED")
        print("="*70)
        print_dem_info(dem_output)
        return dem_output
    else:
        print("\n" + "="*70)
        print("❌ FAILED - UPPER INDUS DEM DOWNLOAD FAILED")
        print("="*70)
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download SRTM DEM for catchments or Upper Indus Basin')
    parser.add_argument('--mode', choices=['catchment', 'indus'], default='indus',
                       help='Download mode: catchment (from namelist) or indus (Upper Indus Basin)')
    parser.add_argument('--output-dir', default='/home/jberg/Raven-world/data/dem',
                       help='Output directory for Upper Indus DEM (default: /home/jberg/Raven-world/data/dem)')
    
    args = parser.parse_args()
    
    if args.mode == 'indus':
        # Download Upper Indus Basin DEM
        download_upper_indus_dem(args.output_dir)
        
    else:
        # Original catchment-based download from namelist
        namelist_path = "/home/jberg/OneDrive/Raven-world/namelist.yaml"
        gauge_id, shapefile_path, dem_path = read_namelist(namelist_path)
        buffer_degrees = 0.01  # ~1 km at equator

        print(f"\nGauge ID: {gauge_id}")
        print(f"Shapefile: {shapefile_path}")
        print(f"DEM output: {dem_path}")

        # Check if DEM already exists
        if os.path.exists(dem_path) and os.path.getsize(dem_path) > 1000:
            print(f"✅ DEM already exists: {dem_path}")
            print_dem_info(dem_path)
        else:
            # Extract extent from shapefile
            bounds = get_extent_from_shapefile(shapefile_path, buffer_degrees)
            if bounds is None:
                print("❌ ERROR: Could not extract extent from shapefile")
            else:
                temp_dir = tempfile.mkdtemp(prefix=f"srtm_{gauge_id}_")
                temp_dem_path = os.path.join(temp_dir, f"full_dem_{gauge_id}.tif")
                try:
                    if download_dem(bounds, temp_dem_path):
                        if clip_raster_to_shapefile(temp_dem_path, shapefile_path, dem_path):
                            print(f"✅ Successfully created DEM: {dem_path}")
                            print_dem_info(dem_path)
                        else:
                            print("❌ ERROR: Failed to clip DEM to catchment")
                    else:
                        print("❌ ERROR: Failed to download DEM")
                finally:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        print(f"Cleaned up temporary directory")