import os
import requests
import rasterio
import rasterio.mask
import geopandas as gpd
import numpy as np
import tempfile
import shutil
import traceback
import yaml
from scipy import stats

def read_namelist(namelist_path):
    """Read gauge_id, shape_dir, landuse_dir from namelist.yaml"""
    with open(namelist_path, 'r') as f:
        config = yaml.safe_load(f)
    gauge_id = str(config.get('gauge_id'))
    main_dir = config.get('main_dir', '')
    shape_dir = config.get('shape_dir')
    landuse_dir = config.get('landuse_dir')
    # Build absolute paths if needed
    shape_path = shape_dir.format(gauge_id=gauge_id)
    landuse_path = landuse_dir.format(gauge_id=gauge_id)
    if not os.path.isabs(shape_path):
        shape_path = os.path.join(main_dir, shape_path)
    if not os.path.isabs(landuse_path):
        landuse_path = os.path.join(main_dir, landuse_path)
    return gauge_id, shape_path, landuse_path

def get_extent_from_shapefile(shapefile_path, buffer_degrees=0.01):
    """
    Read shapefile and extract bounding box extent
    """
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
        
        return (minx, miny, maxx, maxy)
        
    except Exception as e:
        print(f"Error reading shapefile: {str(e)}")
        traceback.print_exc()
        return None

def download_esa_worldcover_direct(bounds, output_path):
    """
    Download ESA WorldCover tiles and merge if catchment spans multiple tiles
    """
    try:
        print("Downloading ESA WorldCover from direct servers...")
        
        minx, miny, maxx, maxy = bounds
        
        # Determine ALL tiles needed
        tiles_needed = set()
        
        # Check longitude tiles (3-degree tiles)
        lon_start = int((minx + 180) // 3) * 3 - 180
        lon_end = int((maxx + 180) // 3) * 3 - 180
        
        # Check latitude tiles
        lat_start = int((miny + 60) // 3) * 3 - 60
        lat_end = int((maxy + 60) // 3) * 3 - 60
        
        # Generate all tile combinations needed
        for tile_lon in range(lon_start, lon_end + 3, 3):
            for tile_lat in range(lat_start, lat_end + 3, 3):
                lat_str = f"S{abs(tile_lat):02d}" if tile_lat < 0 else f"N{tile_lat:02d}"
                lon_str = f"W{abs(tile_lon):03d}" if tile_lon < 0 else f"E{tile_lon:03d}"
                tile_name = f"ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map"
                tiles_needed.add((tile_name, tile_lon, tile_lat))
        
        print(f"Catchment spans {len(tiles_needed)} tile(s):")
        for tile_name, _, _ in tiles_needed:
            print(f"  - {tile_name}")
        
        # Download each tile
        downloaded_tiles = []
        temp_dir = os.path.dirname(output_path)
        
        for tile_name, _, _ in tiles_needed:
            tile_path = os.path.join(temp_dir, f"{tile_name}.tif")
            
            if os.path.exists(tile_path):
                print(f"  Tile already downloaded: {tile_name}")
                downloaded_tiles.append(tile_path)
                continue
            
            # Try direct download URLs
            urls = [
                f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{tile_name}.tif",
                f"http://2018-cfs.esa-worldcover.org/v200/2021/map/{tile_name}.tif",
                f"https://worldcover2021.esa.int/data/v200/2021/map/{tile_name}.tif"
            ]
            
            downloaded = False
            for i, url in enumerate(urls):
                try:
                    print(f"  Downloading {tile_name} from URL {i+1}/{len(urls)}...")
                    response = requests.get(url, stream=True, timeout=300)
                    
                    if response.status_code == 200:
                        with open(tile_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        if os.path.getsize(tile_path) > 1000:
                            print(f"  ✅ Downloaded {tile_name}")
                            downloaded_tiles.append(tile_path)
                            downloaded = True
                            break
                        else:
                            os.remove(tile_path)
                            
                except Exception as e:
                    print(f"  URL {i+1} error: {str(e)}")
                    continue
            
            if not downloaded:
                print(f"  ⚠️  Failed to download {tile_name}")
        
        if not downloaded_tiles:
            return False
        
        # If only one tile, just use it
        if len(downloaded_tiles) == 1:
            shutil.copy(downloaded_tiles[0], output_path)
            print(f"Successfully downloaded ESA WorldCover tile")
            return True
        
        # Multiple tiles - merge them
        print(f"Merging {len(downloaded_tiles)} tiles...")
        
        from rasterio.merge import merge
        
        src_files_to_mosaic = []
        for tile_path in downloaded_tiles:
            src = rasterio.open(tile_path)
            src_files_to_mosaic.append(src)
        
        mosaic, out_trans = merge(src_files_to_mosaic)
        
        # Copy metadata from first tile
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw"
        })
        
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        
        # Close all source files
        for src in src_files_to_mosaic:
            src.close()
        
        print(f"✅ Successfully merged {len(downloaded_tiles)} tiles")
        return True
        
    except Exception as e:
        print(f"Error with direct download: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def clip_raster_to_shapefile(raster_path, shapefile_path, output_path):
    """
    Clip raster to shapefile extent
    """
    try:
        print("Clipping ESA WorldCover to catchment extent...")
        
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
        
        print(f"Successfully clipped to catchment extent")
        return True
        
    except Exception as e:
        print(f"Error clipping raster: {str(e)}")
        traceback.print_exc()
        return False

def aggregate_to_30m_mode(input_path, output_path):
    """
    Fixed aggregation from 10m to 30m using mode - handles no-data properly
    """
    try:
        print("Aggregating from 10m to 30m resolution using mode...")
        
        with rasterio.open(input_path) as src:
            data = src.read(1)
            nodata = src.nodata
            original_transform = src.transform
            original_height, original_width = data.shape
            
            print(f"Processing area: {original_width} x {original_height} pixels")
            print(f"Original no-data value: {nodata}")
            
            scale_factor = 3
            new_height = original_height // scale_factor
            new_width = original_width // scale_factor
            
            # Create new transform
            new_transform = rasterio.Affine(
                original_transform.a * scale_factor,
                original_transform.b,
                original_transform.c,
                original_transform.d,
                original_transform.e * scale_factor,
                original_transform.f
            )
            
            # Initialize output array with 60 (bare/sparse vegetation) as default
            output_data = np.full((new_height, new_width), 60, dtype=data.dtype)
            
            print("Processing blocks...")
            
            for i in range(new_height):
                if i % 100 == 0:
                    print(f"  Processing row {i}/{new_height} ({i/new_height*100:.1f}%)")
                
                for j in range(new_width):
                    # Extract 3x3 block
                    start_row = i * scale_factor
                    end_row = min(start_row + scale_factor, original_height)
                    start_col = j * scale_factor
                    end_col = min(start_col + scale_factor, original_width)
                    
                    block = data[start_row:end_row, start_col:end_col]
                    
                    # Filter out no-data values (0, negative values, or specified nodata)
                    valid_mask = (block > 0) & (block <= 100)
                    if nodata is not None:
                        valid_mask &= (block != nodata)
                    
                    valid_data = block[valid_mask]
                    
                    if len(valid_data) > 0:
                        # Calculate mode of valid ESA classes only
                        unique_vals, counts = np.unique(valid_data, return_counts=True)
                        mode_idx = np.argmax(counts)
                        mode_value = unique_vals[mode_idx]
                        
                        # Ensure mode value is a valid ESA class
                        if mode_value in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]:
                            output_data[i, j] = mode_value
                        else:
                            output_data[i, j] = 60  # Default to bare/sparse vegetation
                    else:
                        # No valid data - use bare/sparse vegetation as default
                        output_data[i, j] = 60
            
            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": new_height,
                "width": new_width,
                "transform": new_transform,
                "compress": "lzw",
                "nodata": None  # Remove nodata value since we're handling it explicitly
            })
            
            # Write aggregated raster
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(output_data, 1)
        
        print(f"Successfully aggregated to 30m resolution")
        
        # Verify the output
        with rasterio.open(output_path) as src:
            check_data = src.read(1)
            unique_out = np.unique(check_data)
            print(f"Output classes: {sorted(unique_out)}")
            
            # Check for class 0
            if 0 in unique_out:
                print("⚠️  ERROR: Still found class 0 in output!")
                return False
            else:
                print("✅ No class 0 found - aggregation successful!")
        
        return True
        
    except Exception as e:
        print(f"Error aggregating raster: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def aggregate_to_30m_mode_parallel(input_path, output_path, n_jobs=-1):
    """
    Aggregate 10m ESA WorldCover to 30m using mode with parallel processing
    Even faster version using joblib for parallel processing
    """
    try:
        from joblib import Parallel, delayed
        import multiprocessing
        
        print("Aggregating from 10m to 30m resolution using mode (parallel)...")
        
        with rasterio.open(input_path) as src:
            data = src.read(1)
            original_transform = src.transform
            original_height, original_width = data.shape
            
            print(f"Processing area: {original_width} x {original_height} pixels")
            
            scale_factor = 3
            new_height = original_height // scale_factor
            new_width = original_width // scale_factor
            
            # Create new transform
            new_transform = rasterio.Affine(
                original_transform.a * scale_factor,
                original_transform.b,
                original_transform.c,
                original_transform.d,
                original_transform.e * scale_factor,
                original_transform.f
            )
            
            # Determine number of jobs
            if n_jobs == -1:
                n_jobs = multiprocessing.cpu_count()
            
            print(f"Using {n_jobs} CPU cores for parallel processing...")
            
            def process_block(i, j):
                """Process a single 3x3 block"""
                start_row = i * scale_factor
                end_row = min(start_row + scale_factor, original_height)
                start_col = j * scale_factor
                end_col = min(start_col + scale_factor, original_width)
                
                block = data[start_row:end_row, start_col:end_col]
                
                if block.size > 0:
                    mode_result = stats.mode(block.flatten(), keepdims=False)
                    return mode_result.mode
                else:
                    return 0
            
            # Create all coordinate pairs
            coordinates = [(i, j) for i in range(new_height) for j in range(new_width)]
            
            print(f"Processing {len(coordinates)} blocks in parallel...")
            
            # Process blocks in parallel
            results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(process_block)(i, j) for i, j in coordinates
            )
            
            # Reshape results back to 2D array
            output_data = np.array(results).reshape(new_height, new_width).astype(data.dtype)
            
            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": new_height,
                "width": new_width,
                "transform": new_transform,
                "compress": "lzw"
            })
            
            # Write aggregated raster
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(output_data, 1)
        
        print(f"Successfully aggregated to 30m resolution using parallel processing")
        return True
        
    except ImportError:
        print("joblib not available, falling back to sequential processing")
        return aggregate_to_30m_mode(input_path, output_path)
    except Exception as e:
        print(f"Error with parallel aggregation: {str(e)}")
        print("Falling back to sequential processing")
        return aggregate_to_30m_mode(input_path, output_path)

def print_landuse_info(landuse_path):
    """
    Print information about the landuse file
    """
    try:
        with rasterio.open(landuse_path) as src:
            print(f"\nLanduse Info:")
            print(f"- Size: {src.width} x {src.height} pixels")
            print(f"- Resolution: {src.res[0]:.6f}° x {src.res[1]:.6f}°")
            print(f"- CRS: {src.crs}")
            print(f"- Bounds: {src.bounds}")
            
            data = src.read(1, masked=True)
            if not data.mask.all():
                unique_classes, counts = np.unique(data.compressed(), return_counts=True)
                print(f"- Landuse classes found: {len(unique_classes)}")
                print(f"- Most common classes:")
                
                # ESA WorldCover class names
                class_names = {
                    10: "Tree cover",
                    20: "Shrubland", 
                    30: "Grassland",
                    40: "Cropland",
                    50: "Built-up",
                    60: "Bare/sparse vegetation",
                    70: "Snow and ice",
                    80: "Permanent water bodies",
                    90: "Herbaceous wetland",
                    95: "Mangroves",
                    100: "Moss and lichen"
                }
                
                sorted_indices = np.argsort(counts)[::-1]
                for i in range(min(5, len(unique_classes))):
                    idx = sorted_indices[i]
                    class_id = unique_classes[idx]
                    percentage = (counts[idx] / len(data.compressed())) * 100
                    class_name = class_names.get(class_id, f"Unknown class {class_id}")
                    print(f"  - {class_name} ({class_id}): {percentage:.1f}%")
                    
    except Exception as e:
        print(f"Could not read landuse info: {e}")

if __name__ == "__main__":
    # Read parameters from namelist.yaml
    namelist_path = "/home/jberg/OneDrive/Raven-world/namelist.yaml"
    gauge_id, shapefile_path, landuse_path = read_namelist(namelist_path)
    buffer_degrees = 0.01  # ~1 km at equator

    print(f"\n{'='*80}")
    print(f"PROCESSING ESA WORLDCOVER FOR GAUGE {gauge_id}")
    print(f"{'='*80}")
    print(f"Gauge ID: {gauge_id}")
    print(f"Shapefile: {shapefile_path}")
    print(f"Landuse output: {landuse_path}")

    # Check if shapefile exists
    if not os.path.exists(shapefile_path):
        print(f"ERROR: Shapefile not found: {shapefile_path}")
        exit(1)
    
    # Check if landuse already exists
    if os.path.exists(landuse_path) and os.path.getsize(landuse_path) > 1000:
        print(f"✅ Landuse raster already exists: {landuse_path}")
        print_landuse_info(landuse_path)
    else:
        # Extract extent from shapefile
        bounds = get_extent_from_shapefile(shapefile_path, buffer_degrees)
        if bounds is None:
            print("ERROR: Could not extract extent from shapefile")
        else:
            # Create output directory if needed
            output_dir = os.path.dirname(landuse_path)
            os.makedirs(output_dir, exist_ok=True)
            
            temp_dir = tempfile.mkdtemp(prefix=f"esa_{gauge_id}_")
            
            try:
                # Step 1: Download ESA WorldCover tile
                temp_worldcover_path = os.path.join(temp_dir, f"worldcover_full_{gauge_id}.tif")
                
                if not download_esa_worldcover_direct(bounds, temp_worldcover_path):
                    print("ERROR: Failed to download ESA WorldCover")
                else:
                    # Step 2: Clip to catchment extent FIRST (much smaller area to process)
                    clipped_path = os.path.join(temp_dir, f"worldcover_clipped_{gauge_id}.tif")
                    
                    if not clip_raster_to_shapefile(temp_worldcover_path, shapefile_path, clipped_path):
                        print("ERROR: Failed to clip ESA WorldCover to catchment")
                    else:
                        # Step 3: Aggregate the clipped area (parallel processing!)
                        if aggregate_to_30m_mode_parallel(clipped_path, landuse_path):
                            print(f"✅ Successfully created landuse raster: {landuse_path}")
                            print_landuse_info(landuse_path)
                        else:
                            print("ERROR: Failed to aggregate to 30m")
                        
            finally:
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary directory")