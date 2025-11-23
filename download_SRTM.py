import os
import requests
import rasterio
import rasterio.mask
import geopandas as gpd
import tempfile
import shutil
import traceback
import elevation
import yaml

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
        return (minx, miny, maxx, maxy)
    except Exception as e:
        print(f"Error reading shapefile: {str(e)}")
        traceback.print_exc()
        return None

def download_with_elevation_package(bounds, output_path):
    try:
        print("Downloading SRTM data using elevation package...")
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
    try:
        print("Trying OpenTopography SRTM as backup...")
        minx, miny, maxx, maxy = bounds
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
    if download_with_elevation_package(bounds, output_path):
        return True
    if download_opentopography_dem(bounds, output_path):
        return True
    print("ERROR: All DEM sources failed")
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
        print(f"Successfully clipped DEM")
        return True
    except Exception as e:
        print(f"Error clipping raster: {str(e)}")
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
            print("ERROR: Could not extract extent from shapefile")
        else:
            temp_dir = tempfile.mkdtemp(prefix=f"srtm_{gauge_id}_")
            temp_dem_path = os.path.join(temp_dir, f"full_dem_{gauge_id}.tif")
            try:
                if download_dem(bounds, temp_dem_path):
                    if clip_raster_to_shapefile(temp_dem_path, shapefile_path, dem_path):
                        print(f"✅ Successfully created DEM: {dem_path}")
                        print_dem_info(dem_path)
                    else:
                        print("ERROR: Failed to clip DEM to catchment")
                else:
                    print("ERROR: Failed to download DEM")
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary directory")