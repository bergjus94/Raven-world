import os
import re
import requests
import rasterio
import rasterio.mask
import rasterio.merge
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
from http.cookiejar import CookieJar

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

# Retry policy with exponential backoff.  Earthdata URS rate-limits IPs that
# auth too aggressively; retry waits give the limit time to clear instead of
# hammering on top of it.
FILE_DOWNLOAD_MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS    = [5, 20, 60, 180]      # one per retry attempt
CIRCUIT_BREAK_AFTER      = 8                     # consecutive failures
CIRCUIT_BREAK_SLEEP_SEC  = 300                   # 5 min cooldown
TRANSIENT_HTTP_CODES     = {429, 500, 502, 503, 504}

# Per-product config: which HDF subdataset to extract and how to aggregate
# to a basin-mean fSCA.  MOD10A1/MYD10A1 = continuous NDSI 0-100; MOD10A2 =
# categorical 8-day max snow extent (25=land no-snow, 200=snow).
PRODUCT_CONFIG = {
    'MOD10A1': {'subdataset': 'NDSI_Snow_Cover',     'kind': 'ndsi'},
    'MYD10A1': {'subdataset': 'NDSI_Snow_Cover',     'kind': 'ndsi'},
    'MOD10A2': {'subdataset': 'Maximum_Snow_Extent', 'kind': 'categorical'},
    'MYD10A2': {'subdataset': 'Maximum_Snow_Extent', 'kind': 'categorical'},
}

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
        # Skip MODIS sidecar metadata XML files — they aren't rasters
        if filename.endswith(".hdf.xml") or filename.endswith(".xml"):
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

def make_authenticated_opener(credentials, token):
    """Build a single OpenerDirector with a shared CookieJar that performs
    the Earthdata URS auth dance once, then reuses the session cookies for
    every subsequent download.

    Per-file re-authentication (the previous pattern) hits NASA's URS rate
    limit after ~3000 requests and gets RST'd back.  Sharing one cookie jar
    means we only auth on the first request (or whenever URS issues a new
    session cookie), avoiding the rate-limit trigger entirely.

    Returns (opener, auth_header).  Pass to `fetch_url`.
    """
    cookie_jar = CookieJar()
    opener = build_opener(HTTPCookieProcessor(cookie_jar))
    if token:
        auth_header = "Bearer {0}".format(token)
    elif credentials:
        auth_header = "Basic {0}".format(credentials)
    else:
        auth_header = None
    return opener, auth_header


def fetch_url(opener, auth_header, url):
    """Fetch URL using the persistent opener.  Auth header attached on every
    request; the shared cookie jar carries the URS session between calls."""
    req = Request(url)
    if auth_header:
        req.add_header("Authorization", auth_header)
    return opener.open(req)


def _is_transient_error(exc):
    """Classify an exception so we know whether to back off and retry."""
    if isinstance(exc, HTTPError) and exc.code in TRANSIENT_HTTP_CODES:
        return True
    if isinstance(exc, URLError):
        # Connection reset, refused, timeout — all worth retrying.
        return True
    msg = str(exc).lower()
    if 'reset by peer' in msg or 'timed out' in msg or 'temporarily' in msg:
        return True
    return False

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
    """Download files from list of urls using a single persistent authenticated
    opener (avoids URS rate limit), with exponential backoff on transient
    errors and a circuit breaker that pauses 5 min after sustained failure."""
    if not urls:
        return []

    url_count = len(urls)
    if not quiet:
        print(f"📥 Downloading {url_count} files...")

    # Authenticate ONCE for the entire batch — opener + cookie jar are reused.
    credentials, token = get_login_credentials()
    opener, auth_header = make_authenticated_opener(credentials, token)

    downloaded_files = []
    consecutive_failures = 0

    for index, url in enumerate(urls, start=1):
        filename = url.split("/")[-1]
        filepath = os.path.join(output_dir, filename)

        if not quiet:
            print(f"\n{index}/{url_count}: {filename}")

        succeeded = False
        last_error = None

        for attempt in range(1, FILE_DOWNLOAD_MAX_RETRIES + 1):
            if attempt > 1 and not quiet:
                backoff = RETRY_BACKOFF_SECONDS[
                    min(attempt - 2, len(RETRY_BACKOFF_SECONDS) - 1)
                ]
                print(f"  ⏳ Retry {attempt-1}/{FILE_DOWNLOAD_MAX_RETRIES-1} "
                      f"after {backoff}s...")
                time.sleep(backoff)

            try:
                response = fetch_url(opener, auth_header, url)
                length = int(response.headers.get("content-length", 0))

                # Skip if existing file already matches expected size
                try:
                    if (not force and length > 0
                            and length == os.path.getsize(filepath)):
                        if not quiet:
                            print("  ✅ File exists, skipping")
                        downloaded_files.append(filepath)
                        succeeded = True
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
                    print(f"\n  ✅ Downloaded: {filename} "
                          f"({os.path.getsize(filepath)/1024/1024:.1f} MB)")
                downloaded_files.append(filepath)
                succeeded = True
                break

            except HTTPError as e:
                last_error = e
                print(f"  ❌ HTTP error {e.code}, {e.reason}")
                # Auth errors won't be fixed by retrying — bail early
                if e.code in (401, 403):
                    break
                if e.code not in TRANSIENT_HTTP_CODES:
                    break
            except URLError as e:
                last_error = e
                print(f"  ❌ URL error: {e.reason}")
            except Exception as e:
                last_error = e
                print(f"  ❌ Download error: {e}")
                if not _is_transient_error(e):
                    break

        if succeeded:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            print(f"  ❌ Failed to download {filename} after "
                  f"{FILE_DOWNLOAD_MAX_RETRIES} attempts")
            # Circuit breaker: if we hit a sustained failure run, pause to
            # let the rate limit clear instead of hammering URS.
            if consecutive_failures >= CIRCUIT_BREAK_AFTER:
                print(f"\n🚨 {consecutive_failures} consecutive failures — "
                      f"pausing {CIRCUIT_BREAK_SLEEP_SEC}s to let any rate "
                      f"limit clear.  Resuming at {time.strftime('%H:%M:%S', time.localtime(time.time() + CIRCUIT_BREAK_SLEEP_SEC))}.\n")
                time.sleep(CIRCUIT_BREAK_SLEEP_SEC)
                consecutive_failures = 0

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

def process_hdf_to_geotiff(hdf_path, output_path, subdataset_name='NDSI_Snow_Cover'):
    """Convert MODIS HDF to GeoTIFF, extracting the named subdataset."""
    try:
        import subprocess

        # Get HDF file info.  Some MODIS HDFs include non-UTF8 bytes in their
        # CoreMetadata blocks, so decode lossy to avoid blowing up the parse.
        cmd_info = f"gdalinfo '{hdf_path}'"
        result_info = subprocess.run(
            cmd_info, shell=True, capture_output=True,
            text=True, errors='replace',
        )

        if result_info.returncode != 0:
            print(f"❌ Could not read HDF file: {os.path.basename(hdf_path)}")
            return False

        # Find requested subdataset (must match SUBDATASET_*_NAME lines)
        lines = result_info.stdout.split('\n')
        snow_dataset = None

        for line in lines:
            if 'SUBDATASET' in line and subdataset_name in line and 'NAME' in line:
                if '=' in line:
                    snow_dataset = line.split('=', 1)[1]
                    break

        if not snow_dataset:
            print(f"❌ Could not find subdataset '{subdataset_name}' in {os.path.basename(hdf_path)}")
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

def _safe_write_geotiff(out_image, out_meta, output_path):
    """Write a GeoTIFF safely even when output_path is on a SMB/GVFS share.

    GDAL's GTiff driver does seeking writes (header, then strips, then
    re-writes the directory).  GVFS-mounted SMB shares mishandle that and
    produce an 8-byte file with just the TIFF magic.  We write to a local
    staging path first and copy the finished file to the final destination
    with a single sequential write.
    """
    needs_staging = output_path.startswith('/run/user/') and 'gvfs' in output_path
    if not needs_staging:
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
        return

    fd, staging_path = tempfile.mkstemp(suffix='.tif', prefix='modis_stage_')
    os.close(fd)
    try:
        with rasterio.open(staging_path, "w", **out_meta) as dest:
            dest.write(out_image)
        shutil.copyfile(staging_path, output_path)
    finally:
        if os.path.exists(staging_path):
            os.remove(staging_path)


def mosaic_and_clip_modis(file_paths, shapefile_path, output_path,
                          subdataset_name='NDSI_Snow_Cover'):
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
                if process_hdf_to_geotiff(file_path, geotiff_path, subdataset_name):
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

            _safe_write_geotiff(out_image, out_meta, output_path)
        else:
            # Multiple tiles - mosaic in memory, then clip and write the
            # final small clipped tif directly to output_path.  Writing an
            # intermediate temp tif is avoided because it breaks on
            # SMB/GVFS-mounted output dirs (write fails on large compressed
            # files).
            from rasterio.io import MemoryFile

            src_files = []
            for file_path in processed_files:
                try:
                    src_files.append(rasterio.open(file_path))
                except Exception as e:
                    print(f"⚠️ Could not open {file_path}: {e}")
                    continue

            if not src_files:
                print("❌ No valid files to mosaic")
                return False

            try:
                mosaic_array, mosaic_transform = rasterio.merge.merge(src_files)
                base_meta = src_files[0].meta.copy()
                src_crs = src_files[0].crs
            finally:
                for src in src_files:
                    src.close()

            base_meta.update({
                "driver": "GTiff",
                "height": mosaic_array.shape[1],
                "width": mosaic_array.shape[2],
                "transform": mosaic_transform,
                "crs": src_crs,
            })

            with MemoryFile() as memfile:
                with memfile.open(**base_meta) as ds:
                    ds.write(mosaic_array)
                with memfile.open() as src:
                    if gdf.crs != src.crs:
                        gdf = gdf.to_crs(src.crs)
                    out_image, out_transform = rasterio.mask.mask(
                        src, gdf.geometry, crop=True
                    )
                    out_meta = src.meta.copy()

            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",
            })

            _safe_write_geotiff(out_image, out_meta, output_path)
        
        print(f"✅ Successfully created: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing files: {e}")
        traceback.print_exc()
        return False

def compute_basin_fsca(geotiff_path, kind):
    """Compute basin-mean fractional snow cover from a clipped MODIS GeoTIFF.

    Returns a dict with: fsca (0-1 or NaN), n_valid, n_cloud, n_total.
      - kind='ndsi': MOD10A1/MYD10A1 NDSI_Snow_Cover band, values 0-100 are
        valid NDSI; 250 is cloud; other values >100 are other QA flags.
        fSCA proxy = mean(valid NDSI) / 100.
      - kind='categorical': MOD10A2/MYD10A2 Maximum_Snow_Extent, values
        25=land no-snow, 200=snow, 50=cloud. fSCA = #snow / (#snow + #no-snow).
    """
    with rasterio.open(geotiff_path) as src:
        arr = src.read(1)

    n_total = int(arr.size)
    if kind == 'ndsi':
        valid_mask = (arr >= 0) & (arr <= 100)
        cloud_mask = (arr == 250)
        n_valid = int(valid_mask.sum())
        n_cloud = int(cloud_mask.sum())
        fsca = float(arr[valid_mask].mean() / 100.0) if n_valid > 0 else float('nan')
    elif kind == 'categorical':
        snow_mask    = (arr == 200)
        no_snow_mask = (arr == 25)
        cloud_mask   = (arr == 50)
        n_snow    = int(snow_mask.sum())
        n_no_snow = int(no_snow_mask.sum())
        n_valid   = n_snow + n_no_snow
        n_cloud   = int(cloud_mask.sum())
        fsca = float(n_snow / n_valid) if n_valid > 0 else float('nan')
    else:
        raise ValueError(f"Unknown fSCA aggregation kind '{kind}'")

    return {'fsca': fsca, 'n_valid': n_valid, 'n_cloud': n_cloud, 'n_total': n_total}


def download_modis_snow_for_catchment(gauge_id, shapefile_dir, output_dir,
                                     start_date, end_date,
                                     product="MOD10A2", buffer_degrees=0.01):
    """Main function to download and process MODIS snow cover data"""

    if product not in PRODUCT_CONFIG:
        raise ValueError(
            f"Unsupported product '{product}'. Supported: {list(PRODUCT_CONFIG)}"
        )
    subdataset_name = PRODUCT_CONFIG[product]['subdataset']
    aggregation_kind = PRODUCT_CONFIG[product]['kind']

    print(f"\n{'='*80}")
    print(f"PROCESSING MODIS SNOW COVER FOR GAUGE {gauge_id}")
    print(f"{'='*80}")
    print(f"Product:     {product}  (subdataset: {subdataset_name})")
    print(f"Date range:  {start_date} to {end_date}")
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

        # Resume: drop granules whose clipped output tif already exists in
        # output_dir.  Lets interrupted runs restart cheaply (the slow step is
        # downloading raw HDFs, which we skip if the date is already processed).
        existing_dates = set()
        try:
            import re
            pat = re.compile(
                rf'^snow_cover_{re.escape(product)}_{re.escape(gauge_id)}_(\d{{8}})\.tif$'
            )
            for f in os.listdir(output_dir):
                m = pat.match(f)
                if m:
                    existing_dates.add(m.group(1))
        except FileNotFoundError:
            pass

        if existing_dates:
            def _date_of_url(url):
                fn = url.split("/")[-1]
                if ".A" not in fn:
                    return None
                dp = fn.split(".A")[1][:7]
                yr, doy = int(dp[:4]), int(dp[4:])
                return (datetime(yr, 1, 1) + timedelta(days=doy - 1)).strftime("%Y%m%d")

            kept = [u for u in url_list if _date_of_url(u) not in existing_dates]
            skipped = len(url_list) - len(kept)
            if skipped:
                print(f"⏭️  Skipping {skipped} already-processed granules; "
                      f"{len(kept)} new to download")
            url_list = kept

        if not url_list:
            print("✅ Nothing new to download — all dates already processed.")
            downloaded_files = []
        else:
            # Download files
            downloaded_files = cmr_download(url_list, temp_dir, force=False, quiet=False)

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
        
        # Process each date and collect basin-mean fSCA into a timeseries
        fsca_rows = []
        for date_str, daily_files in sorted(files_by_date.items()):
            print(f"\n🗓️ Processing {date_str} ({len(daily_files)} files)...")

            output_filename = f"snow_cover_{product}_{gauge_id}_{date_str}.tif"
            output_path = os.path.join(output_dir, output_filename)

            if mosaic_and_clip_modis(daily_files, shapefile_path, output_path,
                                     subdataset_name=subdataset_name):
                successful_files.append(output_path)
                # Aggregate to basin mean
                try:
                    agg = compute_basin_fsca(output_path, aggregation_kind)
                    fsca_rows.append({
                        'date':    f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
                        'fsca':    agg['fsca'],
                        'n_valid': agg['n_valid'],
                        'n_cloud': agg['n_cloud'],
                        'n_total': agg['n_total'],
                    })
                    print(f"  📊 fSCA={agg['fsca']:.3f}  "
                          f"valid={agg['n_valid']}/{agg['n_total']}  "
                          f"cloud={agg['n_cloud']}")
                except Exception as e:
                    print(f"  ⚠️ Basin-mean aggregation failed: {e}")
                print(f"✅ Successfully processed: {output_filename}")
            else:
                print(f"❌ Failed to process: {output_filename}")

        # Write basin-mean fSCA timeseries — scan ALL existing per-date tifs
        # in output_dir (not just newly downloaded ones), so a resumed run
        # produces a complete CSV.  Re-uses already-computed fsca for new dates
        # to avoid re-reading the rasters we just wrote.
        import re
        pat = re.compile(
            rf'^snow_cover_{re.escape(product)}_{re.escape(gauge_id)}_(\d{{8}})\.tif$'
        )
        new_by_date = {r['date'].replace('-', ''): r for r in fsca_rows}
        all_rows = []
        for f in sorted(os.listdir(output_dir)):
            m = pat.match(f)
            if not m:
                continue
            date_str = m.group(1)
            if date_str in new_by_date:
                all_rows.append(new_by_date[date_str])
            else:
                tif_path = os.path.join(output_dir, f)
                try:
                    agg = compute_basin_fsca(tif_path, aggregation_kind)
                    all_rows.append({
                        'date':    f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
                        'fsca':    agg['fsca'],
                        'n_valid': agg['n_valid'],
                        'n_cloud': agg['n_cloud'],
                        'n_total': agg['n_total'],
                    })
                except Exception as e:
                    print(f"  ⚠️ Could not aggregate existing {f}: {e}")

        if all_rows:
            csv_path = os.path.join(
                output_dir, f"fsca_{product}_{gauge_id}.csv"
            )
            header = "date,fsca,n_valid,n_cloud,n_total\n"
            with open(csv_path, "w") as fout:
                fout.write(header)
                for r in all_rows:
                    fsca_str = "" if math.isnan(r['fsca']) else f"{r['fsca']:.6f}"
                    fout.write(
                        f"{r['date']},{fsca_str},{r['n_valid']},"
                        f"{r['n_cloud']},{r['n_total']}\n"
                    )
            print(f"\n📝 Basin-mean fSCA timeseries ({len(all_rows)} dates): {csv_path}")

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary directory")

    print(f"\n📊 SUMMARY:")
    print(f"Successfully processed: {len(successful_files)} files")
    print(f"Files saved to: {output_dir}")

    return successful_files

def process_cached_hdfs(
    cache_dir,
    gauge_id,
    shapefile_dir,
    output_dir,
    product="MOD10A2",
):
    """Run the clip + aggregate path on already-downloaded HDFs in cache_dir.

    Lets us salvage a partial run after a download failure: the HDFs are
    intact, we just need to produce the per-date GeoTIFFs and the CSV.

    Returns the list of clipped GeoTIFFs written.
    """
    if product not in PRODUCT_CONFIG:
        raise ValueError(
            f"Unsupported product '{product}'. Supported: {list(PRODUCT_CONFIG)}"
        )
    subdataset_name  = PRODUCT_CONFIG[product]['subdataset']
    aggregation_kind = PRODUCT_CONFIG[product]['kind']

    cache_dir   = Path(cache_dir)
    output_dir  = str(output_dir)
    shapefile_path = os.path.join(shapefile_dir, f"catchment_shape_{gauge_id}.shp")
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"PROCESSING CACHED HDFs for gauge {gauge_id}")
    print(f"{'='*80}")
    print(f"Cache dir:  {cache_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Product:    {product}  (subdataset: {subdataset_name})")

    hdfs = sorted(cache_dir.glob(f"{product}.A*.hdf"))
    if not hdfs:
        print(f"❌ No {product} HDFs found in {cache_dir}")
        return []

    # Group by acquisition date (YYYY+DOY embedded in filename)
    files_by_date: dict = {}
    for f in hdfs:
        name = f.name
        if '.A' not in name:
            continue
        dp = name.split('.A')[1][:7]
        try:
            yr, doy = int(dp[:4]), int(dp[4:])
        except ValueError:
            continue
        d = datetime(yr, 1, 1) + timedelta(days=doy - 1)
        files_by_date.setdefault(d.strftime('%Y%m%d'), []).append(str(f))

    print(f"📁 {len(hdfs)} HDFs grouped into {len(files_by_date)} dates")

    # Resumability — skip dates whose clipped tif already exists in SMB
    import re
    pat = re.compile(
        rf'^snow_cover_{re.escape(product)}_{re.escape(gauge_id)}_(\d{{8}})\.tif$'
    )
    existing_dates = set()
    for f in os.listdir(output_dir):
        m = pat.match(f)
        if m:
            existing_dates.add(m.group(1))
    if existing_dates:
        skipped = sum(1 for d in files_by_date if d in existing_dates)
        print(f"⏭️  Skipping {skipped} dates already processed in output dir")

    successful_files = []
    fsca_rows = []
    for date_str, daily_files in sorted(files_by_date.items()):
        if date_str in existing_dates:
            continue
        print(f"\n🗓️ {date_str} ({len(daily_files)} tiles)")
        output_filename = f"snow_cover_{product}_{gauge_id}_{date_str}.tif"
        output_path = os.path.join(output_dir, output_filename)
        if mosaic_and_clip_modis(daily_files, shapefile_path, output_path,
                                 subdataset_name=subdataset_name):
            successful_files.append(output_path)
            try:
                agg = compute_basin_fsca(output_path, aggregation_kind)
                fsca_rows.append({
                    'date':    f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
                    'fsca':    agg['fsca'],
                    'n_valid': agg['n_valid'],
                    'n_cloud': agg['n_cloud'],
                    'n_total': agg['n_total'],
                })
                print(f"  📊 fSCA={agg['fsca']:.3f}  valid={agg['n_valid']}/{agg['n_total']}")
            except Exception as e:
                print(f"  ⚠️ Aggregation failed: {e}")

    # CSV — scan all per-date tifs (new + previously-existing) so the
    # resulting timeseries is complete.
    new_by_date = {r['date'].replace('-', ''): r for r in fsca_rows}
    all_rows = []
    for f in sorted(os.listdir(output_dir)):
        m = pat.match(f)
        if not m:
            continue
        date_str = m.group(1)
        if date_str in new_by_date:
            all_rows.append(new_by_date[date_str])
        else:
            try:
                agg = compute_basin_fsca(os.path.join(output_dir, f),
                                         aggregation_kind)
                all_rows.append({
                    'date':    f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
                    'fsca':    agg['fsca'],
                    'n_valid': agg['n_valid'],
                    'n_cloud': agg['n_cloud'],
                    'n_total': agg['n_total'],
                })
            except Exception as e:
                print(f"  ⚠️ Could not re-aggregate {f}: {e}")

    if all_rows:
        csv_path = os.path.join(output_dir, f"fsca_{product}_{gauge_id}.csv")
        with open(csv_path, "w") as fout:
            fout.write("date,fsca,n_valid,n_cloud,n_total\n")
            for r in all_rows:
                fsca_str = "" if math.isnan(r['fsca']) else f"{r['fsca']:.6f}"
                fout.write(f"{r['date']},{fsca_str},{r['n_valid']},"
                           f"{r['n_cloud']},{r['n_total']}\n")
        print(f"\n📝 CSV ({len(all_rows)} dates): {csv_path}")

    print(f"\nDone. {len(successful_files)} new tifs written, "
          f"{len(all_rows)} total dates in CSV.")
    return successful_files


def batch_download_basins(catchments, output_root, shapefile_dir,
                          product="MOD10A2"):
    """Download + clip + aggregate MODIS for a list of catchments.

    Each entry in `catchments` is a dict with keys:
      gauge_id, name, start_date, end_date

    Output goes to <output_root>/basins/<name>_<gauge_id>/ — one folder per
    basin containing clipped GeoTIFFs and the fSCA CSV.
    """
    summary = []
    for c in catchments:
        basin_dir = os.path.join(
            output_root, "basins", f"{c['name']}_{c['gauge_id']}"
        )
        os.makedirs(basin_dir, exist_ok=True)
        files = download_modis_snow_for_catchment(
            gauge_id=c['gauge_id'],
            shapefile_dir=shapefile_dir,
            output_dir=basin_dir,
            start_date=c['start_date'],
            end_date=c['end_date'],
            product=product,
        )
        summary.append({'basin': c['name'], 'gauge': c['gauge_id'],
                        'n_files': len(files), 'output_dir': basin_dir})

    print(f"\n{'='*80}\nBATCH SUMMARY\n{'='*80}")
    for s in summary:
        print(f"  {s['basin']:20s} {s['gauge']:6s} → "
              f"{s['n_files']:5d} files in {s['output_dir']}")
    return summary


def _default_smb_root() -> str:
    uid = os.getuid()
    return (f"/run/user/{uid}/gvfs/"
            f"smb-share:server=hydroshare.giub.unibe.ch,share=data"
            f"/Meteorology/Global/MODIS")


def parse_catchments_arg(items):
    """Parse `--catchments gauge:name:start:end` repeated args.

    Date fields optional; defaults applied at the call site.
    """
    parsed = []
    for item in items:
        parts = item.split(':')
        if len(parts) < 2:
            raise ValueError(
                f"--catchments item '{item}' must be gauge:name[:start:end]"
            )
        entry = {'gauge_id': parts[0], 'name': parts[1]}
        if len(parts) > 2 and parts[2]:
            entry['start_date'] = parts[2]
        if len(parts) > 3 and parts[3]:
            entry['end_date'] = parts[3]
        parsed.append(entry)
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Tile-keyed HDF cache (shared across catchments in the same region)
# ─────────────────────────────────────────────────────────────────────────────
#
# MODIS filename layout: MOD10A2.A2000057.h24v05.061.2020037220731.hdf
#                        ^^^^^^^ ^^^^^^^^ ^^^^^^^                ^^^
#                        product  date    tile_id                ext
#
# Keying the cache by (product, tile_id) lets every Indus catchment reuse
# the same h24v05 download instead of re-fetching it per gauge.  Layout:
#
#     <cache_root>/<product>/<tile_id>/<filename>.hdf
#
# Example: /home/.../01_data/snow/MODIS/tiles/MOD10A2/h24v05/MOD10A2.A2000057...hdf


_MODIS_FILENAME_RE = re.compile(
    r'^(?P<product>M[OY]D10A[12])'
    r'\.A(?P<year>\d{4})(?P<doy>\d{3})'
    r'\.(?P<tile>h\d{2}v\d{2})'
    r'\.\d+\.\d+\.hdf$'
)


def extract_tile_id(filename):
    """Parse the h..v.. tile id from a MODIS HDF filename. Returns None if
    the filename doesn't match the expected pattern."""
    m = _MODIS_FILENAME_RE.match(os.path.basename(filename))
    return m.group('tile') if m else None


def extract_date_from_filename(filename):
    """Parse the YYYY-MM-DD date from a MODIS HDF filename via its YYYYDDD
    julian. Returns None if the filename doesn't match the expected pattern."""
    m = _MODIS_FILENAME_RE.match(os.path.basename(filename))
    if not m:
        return None
    year, doy = int(m.group('year')), int(m.group('doy'))
    return (datetime(year, 1, 1) + timedelta(days=doy - 1)).strftime('%Y-%m-%d')


def tile_cache_dir(cache_root, product, tile_id):
    """Return the cache directory for a (product, tile_id) pair."""
    return Path(cache_root) / product / tile_id


def find_cached_hdfs(cache_root, product, time_start=None, time_end=None):
    """List cached HDFs for a product (optionally clipped to a date range).

    `time_start` / `time_end` are 'YYYY-MM-DD' strings or None for unbounded.
    """
    root = Path(cache_root) / product
    if not root.exists():
        return []
    out = []
    for hdf in sorted(root.glob('*/*.hdf')):
        dt = extract_date_from_filename(hdf.name)
        if dt is None:
            continue
        if time_start and dt < time_start:
            continue
        if time_end and dt > time_end:
            continue
        out.append(hdf)
    return out


def _url_filename(url):
    return url.rstrip('/').split('/')[-1]


def download_tiles_to_cache(product, bounding_box, time_start, time_end,
                            cache_root, quiet=False):
    """Download MODIS tiles for a product+bbox+date-range to the tile cache.

    Skips granules whose target HDF already exists. Returns the full list of
    cached HDF paths covering the requested window (cached + freshly downloaded).

    Parameters
    ----------
    product       : 'MOD10A2' | 'MOD10A1' | 'MYD10A1' | …
    bounding_box  : 'minx,miny,maxx,maxy' string for CMR.
    time_start    : 'YYYY-MM-DD'
    time_end      : 'YYYY-MM-DD'
    cache_root    : Path under which <product>/<tile>/ subfolders live.
    """
    if product not in PRODUCT_CONFIG:
        raise ValueError(f"Unsupported product '{product}'. "
                         f"Supported: {list(PRODUCT_CONFIG)}")
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    cmr_t0 = f"{time_start}T00:00:00Z"
    cmr_t1 = f"{time_end}T23:59:59Z"

    urls = cmr_search(
        short_name=product, version="61",
        time_start=cmr_t0, time_end=cmr_t1,
        bounding_box=bounding_box, polygon="", filename_filter="",
        quiet=quiet,
    )
    if not urls:
        print(f"  No granules in CMR for {product} {time_start}–{time_end}")
        return find_cached_hdfs(cache_root, product, time_start, time_end)

    # Map url → (tile_id, target_path); skip if target already exists
    to_download = []
    target_paths = []
    for url in urls:
        fn = _url_filename(url)
        tile = extract_tile_id(fn)
        if tile is None:
            print(f"  Skipping (cannot parse tile id): {fn}")
            continue
        dest = tile_cache_dir(cache_root, product, tile) / fn
        target_paths.append(dest)
        if not dest.exists():
            to_download.append((url, dest))

    if not to_download:
        if not quiet:
            print(f"  All {len(target_paths)} granules already cached.")
        return sorted(p for p in target_paths if p.exists())

    if not quiet:
        print(f"  {len(to_download)} new granules to download "
              f"({len(target_paths) - len(to_download)} already cached).")

    # cmr_download writes into a single staging dir; afterwards we move each
    # file into <cache_root>/<product>/<tile>/. Doing it this way reuses the
    # existing retry/circuit-breaker logic in cmr_download.
    staging = cache_root / '_staging'
    staging.mkdir(exist_ok=True)
    try:
        download_urls = [u for u, _ in to_download]
        cmr_download(download_urls, str(staging), force=False, quiet=quiet)
        for _, dest in to_download:
            src = staging / dest.name
            if src.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
    finally:
        # Sweep any remaining files (e.g. failed downloads) into the matching
        # tile cache dir so we don't lose partial work — but only if non-empty.
        for leftover in list(staging.glob('*.hdf')):
            tile = extract_tile_id(leftover.name)
            if tile is None:
                continue
            dest = tile_cache_dir(cache_root, product, tile) / leftover.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(leftover), str(dest))
        if staging.exists() and not any(staging.iterdir()):
            staging.rmdir()

    return sorted(p for p in target_paths if p.exists())


def get_extent_from_raster(raster_path, buffer_degrees=0.05):
    """Return (minx, miny, maxx, maxy) in EPSG:4326 from a raster file.

    Used to derive a region bbox from dem_<region>.tif for region-wide
    MODIS downloads (instead of per-catchment shapefiles).
    """
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs
    if crs is None:
        raise ValueError(f"Raster has no CRS: {raster_path}")
    if crs.to_epsg() != 4326:
        # Reproject the bounds polygon, not just the corners
        from rasterio.warp import transform_bounds
        bounds = transform_bounds(crs, 'EPSG:4326',
                                  bounds.left, bounds.bottom,
                                  bounds.right, bounds.top, densify_pts=21)
        minx, miny, maxx, maxy = bounds
    else:
        minx, miny, maxx, maxy = bounds.left, bounds.bottom, bounds.right, bounds.top
    return (minx - buffer_degrees, miny - buffer_degrees,
            maxx + buffer_degrees, maxy + buffer_degrees)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download MODIS snow cover for a list of catchments. "
                    "Same script runs locally (writes to GVFS SMB by default) "
                    "or on a server (pass --output-root to a local directory)."
    )
    parser.add_argument(
        '--output-root',
        default=_default_smb_root(),
        help="Root output directory.  Basin subfolders go in <root>/basins/."
             "  Default: GVFS hydroshare mount (laptop).  On a server, pass "
             "the local data path, e.g. /home/jberg@giub.local/Raven_world/"
             "01_data/snow/MODIS."
    )
    parser.add_argument(
        '--shapefile-dir',
        default="/home/jberg/OneDrive/Raven_worldwide/01_data/topo/catchment_shapefile",
        help="Dir containing catchment_shape_<gauge>.shp files."
    )
    parser.add_argument(
        '--catchments',
        nargs='+',
        default=['0102:Hunza:2000-02-26:2026-05-14',
                 '0130:Chenab:2000-02-26:2026-05-14'],
        help="Repeated 'gauge:name[:start:end]' entries.  Dates default to "
             "the full MODIS-Terra record if omitted."
    )
    parser.add_argument(
        '--start', default='2000-02-26',
        help="Default start_date for catchments that don't specify one."
    )
    parser.add_argument(
        '--end', default='2026-05-14',
        help="Default end_date for catchments that don't specify one."
    )
    parser.add_argument(
        '--product', default='MOD10A2',
        choices=sorted(PRODUCT_CONFIG.keys()),
        help="MODIS product."
    )
    parser.add_argument(
        '--cache-dir', default=None,
        help="If set, run process_cached_hdfs() instead of downloading.  "
             "Used to salvage a partial run from already-downloaded HDFs."
    )
    parser.add_argument(
        '--gauge', default=None,
        help="(only with --cache-dir) which gauge ID the cached HDFs belong to."
    )
    parser.add_argument(
        '--name', default=None,
        help="(only with --cache-dir) basin folder name for the output."
    )

    args = parser.parse_args()

    # --- Recovery mode: process cached HDFs, no download ---
    if args.cache_dir:
        if not args.gauge or not args.name:
            parser.error("--cache-dir requires --gauge and --name")
        out_dir = os.path.join(args.output_root, 'basins',
                               f"{args.name}_{args.gauge}")
        os.makedirs(out_dir, exist_ok=True)
        process_cached_hdfs(
            cache_dir=args.cache_dir,
            gauge_id=args.gauge,
            shapefile_dir=args.shapefile_dir,
            output_dir=out_dir,
            product=args.product,
        )
        sys.exit(0)

    # --- Normal download mode ---
    catchments = parse_catchments_arg(args.catchments)
    for c in catchments:
        c.setdefault('start_date', args.start)
        c.setdefault('end_date',   args.end)

    batch_download_basins(
        catchments=catchments,
        output_root=args.output_root,
        shapefile_dir=args.shapefile_dir,
        product=args.product,
    )