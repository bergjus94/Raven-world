import os
import requests
import rasterio
import rasterio.mask
import geopandas as gpd
import numpy as np
import tempfile
import shutil
import traceback
from datetime import datetime, timedelta
import time
from pathlib import Path
import json
import ssl
import itertools
import math
import netrc
import base64
from getpass import getpass
try:
    from urllib.parse import urlparse
    from urllib.request import urlopen, Request, build_opener, HTTPCookieProcessor
    from urllib.error import HTTPError, URLError
except ImportError:
    from urlparse import urlparse
    from urllib2 import urlopen, Request, HTTPError, URLError, build_opener, HTTPCookieProcessor

# CMR and authentication constants
CMR_URL = "https://cmr.earthdata.nasa.gov"
URS_URL = "https://urs.earthdata.nasa.gov"
CMR_PAGE_SIZE = 2000
CMR_FILE_URL = (
    "{0}/search/granules.json?"
    "&sort_key[]=start_date&sort_key[]=producer_granule_id"
    "&page_size={1}".format(CMR_URL, CMR_PAGE_SIZE)
)
CMR_COLLECTIONS_URL = "{0}/search/collections.json?".format(CMR_URL)
FILE_DOWNLOAD_MAX_RETRIES = 3

def get_login_credentials():
    """Get user credentials from .netrc or prompt for input."""
    credentials = None
    token = None

    try:
        info = netrc.netrc()
        username, _account, password = info.authenticators(urlparse(URS_URL).hostname)
        if username == "token":
            token = password
        else:
            credentials = "{0}:{1}".format(username, password)
            credentials = base64.b64encode(credentials.encode("ascii")).decode("ascii")
    except Exception:
        print("⚠️ No .netrc credentials found, will prompt for username/password")
        username = input("Earthdata username (or press Return to use a bearer token): ")
        if len(username):
            password = getpass("password: ")
            credentials = "{0}:{1}".format(username, password)
            credentials = base64.b64encode(credentials.encode("ascii")).decode("ascii")
        else:
            token = getpass("bearer token: ")

    return credentials, token

def build_version_query_params(version):
    """Build version query parameters for CMR."""
    desired_pad_length = 3
    if len(version) > desired_pad_length:
        print('Version string too long: "{0}"'.format(version))
        return ""

    version = str(int(version))  # Strip off any leading zeros
    query_params = ""

    while len(version) <= desired_pad_length:
        padded_version = version.zfill(desired_pad_length)
        query_params += "&version={0}".format(padded_version)
        desired_pad_length -= 1
    return query_params

def build_query_params_str(short_name, version, time_start="", time_end="", 
                          bounding_box=None, polygon=None, filename_filter=None, provider=None):
    """Create the query params string for the given inputs."""
    params = "&short_name={0}".format(short_name)
    params += build_version_query_params(version)
    if time_start or time_end:
        params += "&temporal[]={0},{1}".format(time_start, time_end)
    if polygon:
        params += "&polygon={0}".format(polygon)
    elif bounding_box:
        params += "&bounding_box={0}".format(bounding_box)
    if filename_filter:
        filters = filename_filter.split(",")
        params += "&options[producer_granule_id][pattern]=true"
        for filter_item in filters:
            if not filter_item.startswith("*"):
                filter_item = "*" + filter_item
            if not filter_item.endswith("*"):
                filter_item = filter_item + "*"
            params += "&producer_granule_id[]=" + filter_item
    if provider:
        params += "&provider={0}".format(provider)

    return params

def build_cmr_query_url(short_name, version, time_start, time_end, 
                       bounding_box=None, polygon=None, filename_filter=None, provider=None):
    """Build the complete CMR query URL."""
    params = build_query_params_str(
        short_name=short_name, version=version, time_start=time_start, time_end=time_end,
        bounding_box=bounding_box, polygon=polygon, filename_filter=filename_filter, provider=provider
    )
    return CMR_FILE_URL + params

def check_provider_for_collection(short_name, version, provider):
    """Return True if the collection is available for the given provider."""
    query_params = build_query_params_str(short_name=short_name, version=version, provider=provider)
    cmr_query_url = CMR_COLLECTIONS_URL + query_params

    req = Request(cmr_query_url)
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        response = urlopen(req, context=ctx)
    except Exception as e:
        print("Error checking provider: " + str(e))
        return False

    search_page = response.read()
    search_page = json.loads(search_page.decode("utf-8"))

    if "feed" not in search_page or "entry" not in search_page["feed"]:
        return False

    return len(search_page["feed"]["entry"]) > 0

def get_provider_for_collection(short_name, version):
    """Return the provider for the collection."""
    # Try cloud provider first
    cloud_provider = "NSIDC_CPRD"
    if check_provider_for_collection(short_name, version, cloud_provider):
        return cloud_provider

    # Fall back to ECS
    ecs_provider = "NSIDC_ECS"
    if check_provider_for_collection(short_name, version, ecs_provider):
        return ecs_provider

    raise RuntimeError(
        "Found no collection matching short_name ({0}) and version ({1})".format(short_name, version)
    )

def cmr_filter_urls(search_results):
    """Select only the desired data files from CMR response."""
    if "feed" not in search_results or "entry" not in search_results["feed"]:
        return []

    entries = [e["links"] for e in search_results["feed"]["entry"] if "links" in e]
    links = list(itertools.chain(*entries))

    urls = []
    unique_filenames = set()
    for link in links:
        if "href" not in link:
            continue
        if "inherited" in link and link["inherited"] is True:
            continue
        if "rel" in link and "data#" not in link["rel"]:
            continue
        if "title" in link and "opendap" in link["title"].lower():
            continue

        filename = link["href"].split("/")[-1]

        if "metadata#" in link["rel"] and filename.endswith(".dmrpp"):
            continue
        if "metadata#" in link["rel"] and filename == "s3credentials":
            continue
        if filename in unique_filenames:
            continue
        unique_filenames.add(filename)

        urls.append(link["href"])

    return urls

def cmr_search(short_name, version, time_start, time_end, bounding_box="", 
              polygon="", filename_filter="", quiet=False):
    """Perform a scrolling CMR query for files matching input criteria."""
    provider = get_provider_for_collection(short_name=short_name, version=version)
    cmr_query_url = build_cmr_query_url(
        provider=provider, short_name=short_name, version=version,
        time_start=time_start, time_end=time_end, bounding_box=bounding_box,
        polygon=polygon, filename_filter=filename_filter
    )
    
    if not quiet:
        print("🔍 Querying CMR for data:")
        print(f"   URL: {cmr_query_url}")

    cmr_paging_header = "cmr-search-after"
    cmr_page_id = None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    urls = []
    hits = 0
    while True:
        req = Request(cmr_query_url)
        if cmr_page_id:
            req.add_header(cmr_paging_header, cmr_page_id)
        try:
            response = urlopen(req, context=ctx)
        except Exception as e:
            print("❌ CMR query error: " + str(e))
            break

        headers = {k.lower(): v for k, v in dict(response.info()).items()}
        if not cmr_page_id:
            hits = int(headers["cmr-hits"])
            if not quiet:
                if hits > 0:
                    print(f"📦 Found {hits} matching files")
                else:
                    print("❌ Found no matching files")

        cmr_page_id = headers.get(cmr_paging_header)

        search_page = response.read()
        search_page = json.loads(search_page.decode("utf-8"))
        url_scroll_results = cmr_filter_urls(search_page)
        if not url_scroll_results:
            break
        if not quiet and hits > CMR_PAGE_SIZE:
            print(".", end="")
        urls += url_scroll_results

    if not quiet and hits > CMR_PAGE_SIZE:
        print()
    return urls

def get_login_response(url, credentials, token):
    """Get authenticated response from URL."""
    opener = build_opener(HTTPCookieProcessor())

    req = Request(url)
    if token:
        req.add_header("Authorization", "Bearer {0}".format(token))
    elif credentials:
        try:
            response = opener.open(req)
            url = response.url
        except HTTPError:
            pass
        except Exception as e:
            print("Authentication error: " + str(e))
            raise

        req = Request(url)
        req.add_header("Authorization", "Basic {0}".format(credentials))

    try:
        response = opener.open(req)
    except HTTPError as e:
        err = "HTTP error {0}, {1}".format(e.code, e.reason)
        if "Unauthorized" in e.reason:
            if token:
                err += ": Check your bearer token"
            else:
                err += ": Check your username and password"
        print(err)
        raise
    except Exception as e:
        print("Error: " + str(e))
        raise

    return response

def cmr_read_in_chunks(file_object, chunk_size=1024 * 1024):
    """Read a file in chunks using a generator."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

def get_speed(time_elapsed, chunk_size):
    """Calculate download speed."""
    if time_elapsed <= 0:
        return ""
    speed = chunk_size / time_elapsed
    if speed <= 0:
        speed = 1
    size_name = ("", "k", "M", "G", "T", "P", "E", "Z", "Y")
    i = int(math.floor(math.log(speed, 1000)))
    p = math.pow(1000, i)
    return "{0:.1f}{1}B/s".format(speed / p, size_name[i])

def output_progress(count, total, status="", bar_len=40):
    """Output download progress bar."""
    if total <= 0:
        return
    fraction = min(max(count / float(total), 0), 1)
    filled_len = int(round(bar_len * fraction))
    percents = int(round(100.0 * fraction))
    bar = "=" * filled_len + " " * (bar_len - filled_len)
    fmt = "  [{0}] {1:3d}%  {2}".format(bar, percents, status)
    print("\r" + fmt, end="")

def cmr_download(urls, output_dir, force=False, quiet=False):
    """Download files from list of urls."""
    if not urls:
        return []

    url_count = len(urls)
    if not quiet:
        print(f"📥 Downloading {url_count} files...")
    
    credentials = None
    token = None
    downloaded_files = []

    for index, url in enumerate(urls, start=1):
        if not credentials and not token:
            p = urlparse(url)
            if p.scheme == "https":
                credentials, token = get_login_credentials()

        filename = url.split("/")[-1]
        filepath = os.path.join(output_dir, filename)
        
        if not quiet:
            print(f"\n{index}/{url_count}: {filename}")

        for attempt in range(1, FILE_DOWNLOAD_MAX_RETRIES + 1):
            if not quiet and attempt > 1:
                print(f"  Retry {attempt-1}/{FILE_DOWNLOAD_MAX_RETRIES-1}")
            try:
                response = get_login_response(url, credentials, token)
                length = int(response.headers.get("content-length", 0))
                
                # Check if file already exists
                try:
                    if not force and length > 0 and length == os.path.getsize(filepath):
                        if not quiet:
                            print("  ✅ File exists, skipping")
                        downloaded_files.append(filepath)
                        break
                except OSError:
                    pass

                count = 0
                chunk_size = min(max(length, 1), 1024 * 1024)
                max_chunks = int(math.ceil(length / chunk_size)) if length > 0 else 0
                time_initial = time.time()
                
                with open(filepath, "wb") as out_file:
                    for data in cmr_read_in_chunks(response, chunk_size=chunk_size):
                        out_file.write(data)
                        if not quiet and max_chunks > 0:
                            count = count + 1
                            time_elapsed = time.time() - time_initial
                            download_speed = get_speed(time_elapsed, count * chunk_size)
                            output_progress(count, max_chunks, status=download_speed)
                
                if not quiet:
                    print(f"\n  ✅ Downloaded: {filename} ({os.path.getsize(filepath)/1024/1024:.1f} MB)")
                downloaded_files.append(filepath)
                break
                
            except HTTPError as e:
                print(f"  ❌ HTTP error {e.code}, {e.reason}")
            except URLError as e:
                print(f"  ❌ URL error: {e.reason}")
            except Exception as e:
                print(f"  ❌ Download error: {e}")

            if attempt == FILE_DOWNLOAD_MAX_RETRIES:
                print(f"  ❌ Failed to download {filename} after {FILE_DOWNLOAD_MAX_RETRIES} attempts")

    return downloaded_files

def get_extent_from_shapefile(shapefile_path, buffer_degrees=0.01):
    """Read shapefile and extract bounding box extent"""
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
        return (minx, miny, maxx, maxy)
        
    except Exception as e:
        print(f"Error reading shapefile: {str(e)}")
        return None

def process_hdf_to_geotiff(hdf_path, output_path):
    """Convert MODIS HDF to GeoTIFF"""
    try:
        import subprocess
        
        # Get HDF file info
        cmd_info = f"gdalinfo '{hdf_path}'"
        result_info = subprocess.run(cmd_info, shell=True, capture_output=True, text=True)
        
        if result_info.returncode != 0:
            print(f"❌ Could not read HDF file: {os.path.basename(hdf_path)}")
            return False
        
        # Find snow cover subdataset
        lines = result_info.stdout.split('\n')
        snow_dataset = None
        
        for line in lines:
            if 'SUBDATASET' in line and 'NDSI_Snow_Cover' in line:
                if '=' in line:
                    snow_dataset = line.split('=', 1)[1]
                    break
        
        if not snow_dataset:
            print(f"❌ Could not find snow cover dataset in {os.path.basename(hdf_path)}")
            return False
        
        # Convert to GeoTIFF
        cmd_translate = f"gdal_translate -of GTiff '{snow_dataset}' '{output_path}'"
        result_translate = subprocess.run(cmd_translate, shell=True, capture_output=True, text=True)
        
        if result_translate.returncode == 0 and os.path.exists(output_path):
            print(f"  ✅ Converted to GeoTIFF: {os.path.basename(output_path)}")
            return True
        else:
            print(f"  ❌ Failed to convert HDF: {result_translate.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing HDF: {e}")
        return False

def mosaic_and_clip_modis(file_paths, shapefile_path, output_path):
    """Mosaic and clip MODIS files to catchment"""
    try:
        if not file_paths:
            return False
        
        print(f"🔧 Processing {len(file_paths)} files...")
        
        # Process HDF files to GeoTIFF
        processed_files = []
        for file_path in file_paths:
            if file_path.endswith('.hdf'):
                geotiff_path = file_path.replace('.hdf', '_snow_cover.tif')
                if process_hdf_to_geotiff(file_path, geotiff_path):
                    processed_files.append(geotiff_path)
                else:
                    print(f"⚠️ Could not process {os.path.basename(file_path)}")
            else:
                processed_files.append(file_path)
        
        if not processed_files:
            print("❌ No files could be processed")
            return False
        
        print(f"📁 Successfully processed {len(processed_files)} files to GeoTIFF")
        
        # Load shapefile
        gdf = gpd.read_file(shapefile_path)
        
        if len(processed_files) == 1:
            # Single file - just clip
            with rasterio.open(processed_files[0]) as src:
                if gdf.crs != src.crs:
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
        else:
            # Multiple files - mosaic first
            import rasterio.merge
            
            src_files = []
            for file_path in processed_files:
                try:
                    src = rasterio.open(file_path)
                    src_files.append(src)
                except Exception as e:
                    print(f"⚠️ Could not open {file_path}: {e}")
                    continue
            
            if not src_files:
                print("❌ No valid files to mosaic")
                return False
            
            # Mosaic files
            mosaic_array, mosaic_transform = rasterio.merge.merge(src_files)
            
            # Close source files
            for src in src_files:
                src.close()
            
            # Create temporary mosaic
            temp_mosaic = output_path.replace('.tif', '_temp_mosaic.tif')
            out_meta = src_files[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic_array.shape[1],
                "width": mosaic_array.shape[2],
                "transform": mosaic_transform,
                "compress": "lzw"
            })
            
            with rasterio.open(temp_mosaic, "w", **out_meta) as dest:
                dest.write(mosaic_array)
            
            # Clip mosaic to catchment
            with rasterio.open(temp_mosaic) as src:
                if gdf.crs != src.crs:
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
            
            # Clean up temp file
            if os.path.exists(temp_mosaic):
                os.remove(temp_mosaic)
        
        print(f"✅ Successfully created: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing files: {e}")
        traceback.print_exc()
        return False

def download_modis_snow_for_catchment(gauge_id, shapefile_dir, output_dir, 
                                     start_date, end_date, 
                                     product="MOD10A1", buffer_degrees=0.01):
    """Main function to download and process MODIS snow cover data"""
    
    print(f"\n{'='*80}")
    print(f"PROCESSING MODIS SNOW COVER FOR GAUGE {gauge_id}")
    print(f"{'='*80}")
    print(f"Product: {product}")
    print(f"Date range: {start_date} to {end_date}")
    print("🔐 Using NASA Earthdata authentication")
    
    # Check shapefile
    shapefile_path = os.path.join(shapefile_dir, f"catchment_shape_{gauge_id}.shp")
    if not os.path.exists(shapefile_path):
        print(f"❌ ERROR: Shapefile not found: {shapefile_path}")
        return []
    
    print(f"📍 Found shapefile: {shapefile_path}")
    
    # Get extent
    bounds = get_extent_from_shapefile(shapefile_path, buffer_degrees)
    if bounds is None:
        return []
    
    # Format bounding box for CMR
    minx, miny, maxx, maxy = bounds
    bounding_box = f"{minx},{miny},{maxx},{maxy}"
    
    # Setup directories
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix=f"modis_{gauge_id}_")
    
    successful_files = []
    
    try:
        # Format dates for CMR
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        time_start = start_dt.strftime('%Y-%m-%dT00:00:00Z')
        time_end = end_dt.strftime('%Y-%m-%dT23:59:59Z')
        
        print(f"📦 Searching and downloading MODIS data...")
        
        # Search for files using CMR
        url_list = cmr_search(
            short_name=product,
            version="61",
            time_start=time_start,
            time_end=time_end,
            bounding_box=bounding_box,
            polygon="",
            filename_filter="",
            quiet=False
        )
        
        if not url_list:
            print("❌ No files found matching criteria")
            return []
        
        # Download files
        downloaded_files = cmr_download(url_list, temp_dir, force=False, quiet=False)
        
        if not downloaded_files:
            print("❌ No files were downloaded")
            return []
        
        print(f"\n📊 Processing {len(downloaded_files)} downloaded files...")
        
        # Group files by date
        files_by_date = {}
        for file_path in downloaded_files:
            filename = os.path.basename(file_path)
            if '.A' in filename:
                date_part = filename.split('.A')[1][:7]  # Get YYYYDDD
                year = int(date_part[:4])
                doy = int(date_part[4:])
                file_date = datetime(year, 1, 1) + timedelta(days=doy-1)
                date_str = file_date.strftime('%Y%m%d')
                
                if date_str not in files_by_date:
                    files_by_date[date_str] = []
                files_by_date[date_str].append(file_path)
        
        # Process each date
        for date_str, daily_files in files_by_date.items():
            print(f"\n🗓️ Processing {date_str} ({len(daily_files)} files)...")
            
            output_filename = f"snow_cover_{product}_{gauge_id}_{date_str}.tif"
            output_path = os.path.join(output_dir, output_filename)
            
            if mosaic_and_clip_modis(daily_files, shapefile_path, output_path):
                successful_files.append(output_path)
                print(f"✅ Successfully processed: {output_filename}")
            else:
                print(f"❌ Failed to process: {output_filename}")
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary directory")
    
    print(f"\n📊 SUMMARY:")
    print(f"Successfully processed: {len(successful_files)} files")
    print(f"Files saved to: {output_dir}")
    
    return successful_files

if __name__ == "__main__":
    # Parameters
    gauge_id = "0001"
    start_date = "2022-01-01"
    end_date = "2022-01-05"
    product = "MOD10A1"
    
    # Directories
    shapefile_dir = "/home/jberg/OneDrive/Raven_worldwide/01_data/topo/catchment_shapefile"
    output_dir = "/home/jberg/OneDrive/Raven_worldwide/01_data/snow/MODIS"
    
    # Download MODIS data
    snow_files = download_modis_snow_for_catchment(
        gauge_id=gauge_id,
        shapefile_dir=shapefile_dir,
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        product=product
    )
    
    if snow_files:
        print(f"\n🎉 MODIS download complete!")
        print(f"Files saved to: {output_dir}")
        for f in snow_files:
            print(f"  - {os.path.basename(f)}")
    else:
        print(f"\n❌ MODIS download failed")