import os
import requests
import rasterio
import rasterio.mask
import geopandas as gpd
import numpy as np
import tempfile
import shutil
import traceback
from scipy import stats

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
    Download ESA WorldCover from direct ESA servers (the working method)
    """
    try:
        print("Downloading ESA WorldCover from direct servers...")
        
        minx, miny, maxx, maxy = bounds
        
        # Calculate the tile name based on coordinates
        center_lon = (minx + maxx) / 2
        center_lat = (miny + maxy) / 2
        
        # Tile coordinates (3-degree tiles)
        tile_lon = int((center_lon + 180) // 3) * 3 - 180
        tile_lat = int((center_lat + 60) // 3) * 3 - 60
        
        # Format tile name
        lat_str = f"S{abs(tile_lat):02d}" if tile_lat < 0 else f"N{tile_lat:02d}"
        lon_str = f"W{abs(tile_lon):03d}" if tile_lon < 0 else f"E{tile_lon:03d}"
        
        tile_name = f"ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map"
        print(f"Need tile: {tile_name}")
        
        # Try direct download URLs
        urls = [
            f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{tile_name}.tif",
            f"http://2018-cfs.esa-worldcover.org/v200/2021/map/{tile_name}.tif",
            f"https://worldcover2021.esa.int/data/v200/2021/map/{tile_name}.tif"
        ]
        
        for i, url in enumerate(urls):
            try:
                print(f"  Trying URL {i+1}/{len(urls)}...")
                response = requests.get(url, stream=True, timeout=300)
                
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    if os.path.getsize(output_path) > 1000:
                        print(f"Successfully downloaded ESA WorldCover tile")
                        return True
                    else:
                        os.remove(output_path)
                        
                print(f"  URL {i+1} failed: HTTP {response.status_code}")
                        
            except Exception as e:
                print(f"  URL {i+1} error: {str(e)}")
                continue
        
        return False
        
    except Exception as e:
        print(f"Error with direct download: {str(e)}")
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
    


def download_esa_worldcover_for_catchment(gauge_id, shapefile_dir, output_dir, buffer_degrees=0.01):
    """
    Download and process ESA WorldCover for a specific catchment
    Optimized workflow: Download -> Clip -> Aggregate (parallel)
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING ESA WORLDCOVER FOR GAUGE {gauge_id}")
    print(f"{'='*80}")

    shapefile_path = os.path.join(shapefile_dir, f"catchment_shape_{gauge_id}.shp")

    if not os.path.exists(shapefile_path):
        print(f"ERROR: Shapefile not found: {shapefile_path}")
        return None
    
    print(f"Found shapefile: {shapefile_path}")
    
    bounds = get_extent_from_shapefile(shapefile_path, buffer_degrees)
    if bounds is None:
        print("ERROR: Could not extract extent from shapefile")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix=f"esa_{gauge_id}_")
    
    try:
        # Step 1: Download ESA WorldCover tile
        temp_worldcover_path = os.path.join(temp_dir, f"worldcover_full_{gauge_id}.tif")
        
        if not download_esa_worldcover_direct(bounds, temp_worldcover_path):
            print("ERROR: Failed to download ESA WorldCover")
            return None
        
        # Step 2: Clip to catchment extent FIRST (much smaller area to process)
        clipped_path = os.path.join(temp_dir, f"worldcover_clipped_{gauge_id}.tif")
        
        if not clip_raster_to_shapefile(temp_worldcover_path, shapefile_path, clipped_path):
            print("ERROR: Failed to clip ESA WorldCover to catchment")
            return None
        
        # Step 3: Aggregate the clipped area (parallel processing!)
        final_landuse_path = os.path.join(output_dir, f"landuse_{gauge_id}.tif")
        
        if aggregate_to_30m_mode_parallel(clipped_path, final_landuse_path):
            print(f"✅ Successfully created landuse raster: {final_landuse_path}")
            return final_landuse_path
        else:
            print("ERROR: Failed to aggregate to 30m")
            return None
            
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory")

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
    # Parameters
    gauge_id = "0620"  # Change this to your gauge ID
    
    # Directories
    shapefile_dir = "/home/jberg/OneDrive/Raven_worldwide/01_data/topo/catchment_shapefile"
    output_dir = "/home/jberg/OneDrive/Raven_worldwide/01_data/topo/catchment_landuse"
    
    # Buffer around shapefile extent (in degrees)
    buffer_degrees = 0.01  # ~1 km at equator
    
    # Process the gauge
    landuse_path = download_esa_worldcover_for_catchment(
        gauge_id=gauge_id,
        shapefile_dir=shapefile_dir,
        output_dir=output_dir,
        buffer_degrees=buffer_degrees
    )
    
    if landuse_path:
        print(f"\n🎉 ESA WorldCover processing complete for gauge {gauge_id}!")
        print(f"Landuse raster saved as: {landuse_path}")
        print_landuse_info(landuse_path)
    else:
        print(f"\n❌ ESA WorldCover processing failed for gauge {gauge_id}")