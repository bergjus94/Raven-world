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

def download_with_elevation_package(bounds, output_path):
    try:
        minx, miny, maxx, maxy = bounds
        width_deg = maxx - minx
        height_deg = maxy - miny
        
        # Skip if too large (elevation package has tile limits)
        if width_deg > 2 or height_deg > 2:
            print(f"Area too large for elevation package ({width_deg:.2f}° x {height_deg:.2f}°), skipping...")
            return False
        
        print("Downloading SRTM data using elevation package...")
        elevation.clip(bounds=bounds, output=output_path, product='SRTM1')
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            print(f"✅ Successfully downloaded DEM with elevation package")
            return True
        else:
            print("Elevation package download failed")
            return False
    except Exception as e:
        print(f"Error with elevation package: {str(e)}")
        return False

def download_opentopography_dem(bounds, output_path, max_tile_size=1.0):
    """
    Download DEM from OpenTopography, splitting into tiles if necessary.
    Includes retry logic and delays to avoid rate limiting.
    """
    try:
        minx, miny, maxx, maxy = bounds
        width_deg = maxx - minx
        height_deg = maxy - miny
        
        # Check if we need to tile
        if width_deg <= max_tile_size and height_deg <= max_tile_size:
            # Small enough - download in one piece
            print("Downloading SRTM from OpenTopography (single tile)...")
            url = "https://portal.opentopography.org/API/globaldem"
            params = {
                'demtype': 'SRTMGL1',
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
                try:
                    with rasterio.open(output_path) as src:
                        if src.width > 0 and src.height > 0:
                            print(f"✅ Successfully downloaded DEM from OpenTopography")
                            return True
                except:
                    pass
            print(f"OpenTopography download failed: HTTP {response.status_code}")
            return False
        
        else:
            # Too large - need to tile
            print(f"Downloading SRTM from OpenTopography (tiled - area: {width_deg:.2f}° x {height_deg:.2f}°)...")
            
            # Calculate number of tiles needed
            n_tiles_x = max(1, int(math.ceil(width_deg / max_tile_size)))
            n_tiles_y = max(1, int(math.ceil(height_deg / max_tile_size)))
            
            total_tiles = n_tiles_x * n_tiles_y
            print(f"  Splitting into {n_tiles_x} x {n_tiles_y} = {total_tiles} tiles")
            
            # Warning if too many tiles
            if total_tiles > 20:
                print(f"  ⚠️  Warning: {total_tiles} tiles may exceed API rate limits")
                print(f"  Consider using a smaller catchment or getting an OpenTopography API key")
            
            tile_files = []
            temp_dir = tempfile.mkdtemp(prefix="opentopo_tiles_")
            
            # Track failures
            failed_tiles = []
            
            try:
                # Download each tile with retry logic
                for i in range(n_tiles_x):
                    for j in range(n_tiles_y):
                        tile_minx = minx + i * (width_deg / n_tiles_x)
                        tile_maxx = min(maxx, minx + (i + 1) * (width_deg / n_tiles_x))
                        tile_miny = miny + j * (height_deg / n_tiles_y)
                        tile_maxy = min(maxy, miny + (j + 1) * (height_deg / n_tiles_y))
                        
                        tile_output = os.path.join(temp_dir, f"tile_{i}_{j}.tif")
                        
                        print(f"  Downloading tile {i+1},{j+1}/{n_tiles_x},{n_tiles_y}...", end=" ")
                        
                        # Try downloading with retries
                        success = False
                        for attempt in range(3):  # 3 attempts per tile
                            try:
                                url = "https://portal.opentopography.org/API/globaldem"
                                params = {
                                    'demtype': 'SRTMGL1',
                                    'south': tile_miny,
                                    'north': tile_maxy,
                                    'west': tile_minx,
                                    'east': tile_maxx,
                                    'outputFormat': 'GTiff',
                                    'API_Key': 'demoapikeyot2022'
                                }
                                
                                response = requests.get(url, params=params, timeout=180)
                                
                                if response.status_code == 200:
                                    with open(tile_output, 'wb') as f:
                                        f.write(response.content)
                                    
                                    # Validate tile
                                    with rasterio.open(tile_output) as src:
                                        if src.width > 0 and src.height > 0:
                                            tile_files.append(tile_output)
                                            print(f"✓")
                                            success = True
                                            break
                                
                                elif response.status_code == 401:
                                    # Rate limit or unauthorized - wait longer
                                    if attempt < 2:
                                        wait_time = 5 * (attempt + 1)
                                        print(f"(401, waiting {wait_time}s)...", end=" ")
                                        time.sleep(wait_time)
                                    else:
                                        print(f"✗ (HTTP 401 - rate limit)")
                                        failed_tiles.append((i+1, j+1))
                                        break
                                else:
                                    print(f"✗ (HTTP {response.status_code})")
                                    failed_tiles.append((i+1, j+1))
                                    break
                                    
                            except Exception as e:
                                if attempt < 2:
                                    print(f"(error, retry {attempt+1})...", end=" ")
                                    time.sleep(2)
                                else:
                                    print(f"✗ ({str(e)[:30]})")
                                    failed_tiles.append((i+1, j+1))
                                    break
                        
                        # Small delay between tiles to avoid rate limiting
                        if success:
                            time.sleep(1)  # 1 second between successful downloads
                
                # Report status
                print(f"\n  Downloaded {len(tile_files)}/{total_tiles} tiles successfully")
                
                if failed_tiles:
                    print(f"  Failed tiles: {failed_tiles}")
                    if len(tile_files) < total_tiles * 0.5:  # Less than 50% success
                        print("  ❌ Too many failed tiles - try again later or use smaller catchment")
                        return False
                
                # Check if we got any tiles
                if len(tile_files) == 0:
                    print("❌ No valid tiles downloaded")
                    return False
                
                elif len(tile_files) == 1:
                    # Only one tile - just copy it
                    print("  Single tile, copying...")
                    shutil.copy(tile_files[0], output_path)
                    print(f"✅ Successfully downloaded DEM from OpenTopography")
                    return True
                    
                else:
                    # Multiple tiles - merge them
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
                    
                    if failed_tiles:
                        print(f"✅ Partially downloaded DEM from OpenTopography ({len(tile_files)}/{total_tiles} tiles)")
                    else:
                        print(f"✅ Successfully downloaded and merged DEM from OpenTopography")
                    return True
                    
            finally:
                # Clean up temp directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    
    except Exception as e:
        print(f"Error downloading from OpenTopography: {str(e)}")
        traceback.print_exc()
        return False

def download_dem(bounds, output_path):
    # Try elevation package first (for small areas)
    if download_with_elevation_package(bounds, output_path):
        return True
    # Try OpenTopography (with automatic tiling if needed)
    if download_opentopography_dem(bounds, output_path, max_tile_size=1.0):
        return True
    print("❌ ERROR: All DEM sources failed")
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
    except Exception as e:
        print(f"Could not read DEM info: {e}")

if __name__ == "__main__":
    # Read parameters from namelist.yaml
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