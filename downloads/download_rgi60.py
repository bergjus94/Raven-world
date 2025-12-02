import os
import requests
import zipfile
import geopandas as gpd
import tempfile
import shutil
import traceback
import numpy as np

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

def get_rgi_regions_for_bounds(bounds):
    """
    Determine which RGI regions overlap with the given bounds
    """
    minx, miny, maxx, maxy = bounds
    
    # RGI region definitions (approximate bounds)
    rgi_regions = {
        '01': {'name': 'Alaska', 'bounds': (-180, 51, -120, 72)},
        '02': {'name': 'Western Canada and US', 'bounds': (-180, 42, -90, 70)},
        '03': {'name': 'Arctic Canada North', 'bounds': (-150, 65, -60, 85)},
        '04': {'name': 'Arctic Canada South', 'bounds': (-130, 50, -60, 75)},
        '05': {'name': 'Greenland', 'bounds': (-75, 59, -10, 85)},
        '06': {'name': 'Iceland', 'bounds': (-25, 63, -13, 67)},
        '07': {'name': 'Svalbard', 'bounds': (10, 76, 35, 81)},
        '08': {'name': 'Scandinavia', 'bounds': (4, 55, 32, 72)},
        '09': {'name': 'Russian Arctic', 'bounds': (30, 65, 180, 82)},
        '10': {'name': 'Asia North', 'bounds': (60, 35, 180, 65)},
        '11': {'name': 'Central Europe', 'bounds': (5, 40, 20, 50)},
        '12': {'name': 'Caucasus and Middle East', 'bounds': (25, 30, 75, 45)},
        '13': {'name': 'Central Asia', 'bounds': (60, 25, 110, 50)},
        '14': {'name': 'South Asia West', 'bounds': (65, 25, 85, 40)},
        '15': {'name': 'South Asia East', 'bounds': (85, 20, 105, 40)},
        '16': {'name': 'Low Latitudes', 'bounds': (-80, -25, 180, 25)},
        '17': {'name': 'Southern Andes', 'bounds': (-80, -56, -65, -40)},
        '18': {'name': 'New Zealand', 'bounds': (165, -48, 180, -34)},
        '19': {'name': 'Antarctic and Subantarctic', 'bounds': (-180, -90, 180, -60)}
    }
    
    overlapping_regions = []
    
    for region_id, info in rgi_regions.items():
        r_minx, r_miny, r_maxx, r_maxy = info['bounds']
        
        # Check if regions overlap
        if not (maxx < r_minx or minx > r_maxx or maxy < r_miny or miny > r_maxy):
            overlapping_regions.append({
                'id': region_id,
                'name': info['name']
            })
    
    return overlapping_regions

def download_rgi_region(region_id, output_dir):
    """
    Download RGI6 data for a specific region using updated URLs
    """
    try:
        # Try multiple sources for RGI6 data
        region_file = f"RGI2000-v6.0_rgi{region_id}.zip"
        
        # Updated URLs for RGI6 data
        urls = [
            # NSIDC (National Snow and Ice Data Center) - most reliable
            f"https://n5eil01u.ecs.nsidc.org/ATLAS/ATL14/003/RGI60/{region_file}",
            # Alternative NSIDC URL
            f"https://nsidc.org/sites/default/files/RGI60/{region_file}",
            # Zenodo repository
            f"https://zenodo.org/record/3939050/files/{region_file}",
            # GLIMS backup (original)
            f"https://www.glims.org/RGI/rgi60_files/{region_file}",
            # Direct GitHub release
            f"https://github.com/GLIMS-RGI/rgi_user_guide/releases/download/v6.0/{region_file}"
        ]
        
        output_path = os.path.join(output_dir, region_file)
        
        # Check if already downloaded
        if os.path.exists(output_path):
            print(f"Region {region_id} already downloaded")
            return output_path
        
        print(f"Downloading RGI region {region_id}...")
        
        for i, url in enumerate(urls):
            try:
                print(f"  Trying source {i+1}/{len(urls)}...")
                response = requests.get(url, stream=True, timeout=300)
                
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Verify it's a valid file
                    if os.path.getsize(output_path) > 1000:
                        print(f"Successfully downloaded RGI region {region_id}")
                        return output_path
                    else:
                        os.remove(output_path)
                
                print(f"  Source {i+1} failed: HTTP {response.status_code}")
                
            except Exception as e:
                print(f"  Source {i+1} error: {str(e)}")
                continue
        
        print(f"All sources failed for RGI region {region_id}")
        return None
            
    except Exception as e:
        print(f"Error downloading RGI region {region_id}: {str(e)}")
        return None

def download_rgi_via_zenodo(region_id, output_dir):
    """
    Download RGI6 from Zenodo (most reliable current source)
    """
    try:
        print(f"Downloading RGI region {region_id} from Zenodo...")
        
        # Zenodo record for RGI 6.0
        zenodo_record_id = "3939050"
        
        # Get file list from Zenodo API
        api_url = f"https://zenodo.org/api/records/{zenodo_record_id}"
        response = requests.get(api_url, timeout=60)
        
        if response.status_code != 200:
            print(f"Failed to access Zenodo API: HTTP {response.status_code}")
            return None
        
        record_data = response.json()
        
        # Find the specific region file
        region_file = f"RGI2000-v6.0_rgi{region_id}.zip"
        file_url = None
        
        for file_info in record_data.get('files', []):
            if file_info['key'] == region_file:
                file_url = file_info['links']['self']
                break
        
        if not file_url:
            print(f"Region {region_id} not found in Zenodo record")
            return None
        
        # Download the file
        output_path = os.path.join(output_dir, region_file)
        
        print(f"Downloading from: {file_url}")
        response = requests.get(file_url, stream=True, timeout=300)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Successfully downloaded RGI region {region_id} from Zenodo")
            return output_path
        else:
            print(f"Failed to download from Zenodo: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error downloading from Zenodo: {str(e)}")
        return None

def extract_rgi_region(zip_path, extract_dir):
    """
    Extract RGI region zip file and return path to shapefile
    """
    try:
        print(f"Extracting {os.path.basename(zip_path)}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find the shapefile
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.endswith('.shp') and ('rgi60' in file.lower() or 'rgi' in file.lower()):
                    shapefile_path = os.path.join(root, file)
                    print(f"Found RGI shapefile: {shapefile_path}")
                    return shapefile_path
        
        print("No RGI shapefile found in extracted files")
        return None
        
    except Exception as e:
        print(f"Error extracting RGI region: {str(e)}")
        return None

def clip_glaciers_to_catchment(rgi_shapefiles, catchment_shapefile, output_path):
    """
    Clip RGI glacier polygons to catchment extent
    """
    try:
        import pandas as pd
        
        print("Processing RGI glacier data...")
        
        # Read catchment shapefile
        catchment_gdf = gpd.read_file(catchment_shapefile)
        
        # Combine all RGI data
        all_glaciers = []
        
        for shp_path in rgi_shapefiles:
            if shp_path and os.path.exists(shp_path):
                print(f"  Reading {os.path.basename(shp_path)}...")
                glaciers_gdf = gpd.read_file(shp_path)
                
                # Ensure same CRS
                if glaciers_gdf.crs != catchment_gdf.crs:
                    print(f"  Reprojecting from {glaciers_gdf.crs} to {catchment_gdf.crs}")
                    glaciers_gdf = glaciers_gdf.to_crs(catchment_gdf.crs)
                
                all_glaciers.append(glaciers_gdf)
        
        if not all_glaciers:
            print("No valid RGI data found")
            return False
        
        # Combine all glacier data
        combined_glaciers = gpd.GeoDataFrame(pd.concat(all_glaciers, ignore_index=True))
        
        print(f"Total glaciers before clipping: {len(combined_glaciers)}")
        
        # Clip glaciers to catchment
        clipped_glaciers = gpd.clip(combined_glaciers, catchment_gdf)
        
        print(f"Glaciers after clipping to catchment: {len(clipped_glaciers)}")
        
        if len(clipped_glaciers) > 0:
            # Save clipped glaciers
            clipped_glaciers.to_file(output_path)
            print(f"Successfully saved clipped glaciers to: {output_path}")
            return True
        else:
            print("No glaciers found within catchment extent")
            # Create empty shapefile with correct schema
            create_empty_glacier_shapefile(output_path, catchment_gdf.crs)
            return True
            
    except Exception as e:
        print(f"Error clipping glaciers: {str(e)}")
        traceback.print_exc()
        return False

def create_empty_glacier_shapefile(output_path, crs):
    """
    Create an empty glacier shapefile when no glaciers are found
    """
    try:
        # Create empty GeoDataFrame with RGI schema
        empty_gdf = gpd.GeoDataFrame({
            'RGIId': [],
            'GLIMSId': [],
            'Name': [],
            'CenLon': [],
            'CenLat': [],
            'O1Region': [],
            'O2Region': [],
            'Area': [],
            'Zmin': [],
            'Zmax': [],
            'Zmean': [],
            'Slope': [],
            'Aspect': [],
            'Lmax': [],
            'Status': [],
            'Connect': [],
            'Form': [],
            'TermType': [],
            'Surging': [],
            'Linkages': [],
            'EndDate': []
        }, crs=crs)
        
        empty_gdf.to_file(output_path)
        print(f"Created empty glacier shapefile: {output_path}")
        
    except Exception as e:
        print(f"Error creating empty shapefile: {str(e)}")

def download_rgi_for_catchment(gauge_id, shapefile_dir, output_dir, buffer_degrees=0.01):
    """
    Download and process RGI6 glacier data for a specific catchment
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING RGI6 GLACIERS FOR GAUGE {gauge_id}")
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
    
    # Determine which RGI regions to download
    rgi_regions = get_rgi_regions_for_bounds(bounds)
    
    if not rgi_regions:
        print("No RGI regions found for this area")
        return None
    
    print(f"Need RGI regions: {', '.join([f'{r['id']} ({r['name']})' for r in rgi_regions])}")
    
    # Setup directories
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix=f"rgi_{gauge_id}_")
    
    try:
        # Download and extract RGI regions
        rgi_shapefiles = []
        
        for region in rgi_regions:
            region_id = region['id']
            
            # Try Zenodo first, then fallback to other sources
            zip_path = download_rgi_via_zenodo(region_id, temp_dir)
            if not zip_path:
                zip_path = download_rgi_region(region_id, temp_dir)
            
            if not zip_path:
                print(f"Failed to download region {region_id}, continuing with others...")
                continue
            
            # Extract region
            extract_path = os.path.join(temp_dir, f"rgi{region_id}")
            os.makedirs(extract_path, exist_ok=True)
            
            shapefile_path_rgi = extract_rgi_region(zip_path, extract_path)
            if shapefile_path_rgi:
                rgi_shapefiles.append(shapefile_path_rgi)
        
        if not rgi_shapefiles:
            print("ERROR: No RGI data successfully downloaded")
            return None
        
        # Clip glaciers to catchment
        final_glacier_path = os.path.join(output_dir, f"glaciers_{gauge_id}.shp")
        
        if clip_glaciers_to_catchment(rgi_shapefiles, shapefile_path, final_glacier_path):
            print(f"✅ Successfully created glacier shapefile: {final_glacier_path}")
            return final_glacier_path
        else:
            print("ERROR: Failed to clip glaciers to catchment")
            return None
            
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory")

def print_glacier_info(glacier_path):
    """
    Print information about the glacier shapefile
    """
    try:
        gdf = gpd.read_file(glacier_path)
        
        print(f"\nGlacier Info:")
        print(f"- Number of glaciers: {len(gdf)}")
        print(f"- CRS: {gdf.crs}")
        
        if len(gdf) > 0:
            print(f"- Bounds: {gdf.total_bounds}")
            
            # Calculate total glacier area
            if 'Area' in gdf.columns:
                total_area = gdf['Area'].sum()
                print(f"- Total glacier area: {total_area:.3f} km²")
                print(f"- Largest glacier: {gdf['Area'].max():.3f} km²")
                print(f"- Smallest glacier: {gdf['Area'].min():.3f} km²")
                print(f"- Mean glacier size: {gdf['Area'].mean():.3f} km²")
            
            # Show elevation stats if available
            if 'Zmin' in gdf.columns and 'Zmax' in gdf.columns:
                print(f"- Elevation range: {gdf['Zmin'].min():.0f} - {gdf['Zmax'].max():.0f} m")
                if 'Zmean' in gdf.columns:
                    print(f"- Mean elevation: {gdf['Zmean'].mean():.0f} m")
        else:
            print("- No glaciers found in catchment")
            
    except Exception as e:
        print(f"Could not read glacier info: {e}")

if __name__ == "__main__":
    # Parameters
    gauge_id = "0001"  # Change this to your gauge ID
    
    # Directories
    shapefile_dir = "/home/jberg/OneDrive/Raven_worldwide/01_data/topo/catchment_shapefile"
    output_dir = "/home/jberg/OneDrive/Raven_worldwide/01_data/topo/catchment_glaciers"
    
    # Buffer around shapefile extent (in degrees)
    buffer_degrees = 0.01  # ~1 km at equator
    
    # Process the gauge
    glacier_path = download_rgi_for_catchment(
        gauge_id=gauge_id,
        shapefile_dir=shapefile_dir,
        output_dir=output_dir,
        buffer_degrees=buffer_degrees
    )
    
    if glacier_path:
        print(f"\n🎉 RGI6 glacier processing complete for gauge {gauge_id}!")
        print(f"Glacier shapefile saved as: {glacier_path}")
        print_glacier_info(glacier_path)
    else:
        print(f"\n❌ RGI6 glacier processing failed for gauge {gauge_id}")