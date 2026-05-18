import pathlib
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.transform import rowcol
from shapely.geometry import Point
import requests
import zipfile
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import elevation
import time
import gc

# WhiteboxTools
import whitebox

print("✅ All libraries imported successfully!")

class CatchmentDelineator:
    """
    A comprehensive catchment delineation system using WhiteboxTools
    Supports custom bounding boxes for very large catchments
    """
    
    def __init__(self, base_dir, country="Nepal", bbox=None, constraint_shapefile=None):
        """
        Initialize CatchmentDelineator
        
        Parameters:
        -----------
        base_dir : str or Path
            Base directory for data storage
        country : str
            Country name for reference
        bbox : tuple, optional
            Custom bounding box as (minx, miny, maxx, maxy) in EPSG:4326
            If provided, uses this instead of calculating from stations
            Format: (west, south, east, north) in degrees
        constraint_shapefile : str or Path, optional
            Path to a shapefile defining an area to subtract from catchments
            Useful for removing false connections caused by DEM artifacts
        """
        self.base_dir = Path(base_dir)
        self.country = country
        self.custom_bbox = bbox
        
        # Load constraint shapefile if provided
        self.constraint_gdf = None
        if constraint_shapefile is not None:
            constraint_path = Path(constraint_shapefile)
            if constraint_path.exists():
                self.constraint_gdf = gpd.read_file(constraint_path)
                # Ensure it's in WGS84
                if self.constraint_gdf.crs != 'EPSG:4326':
                    self.constraint_gdf = self.constraint_gdf.to_crs('EPSG:4326')
                print(f"✅ Loaded constraint shapefile: {constraint_path.name}")
                print(f"   Constraint area: {len(self.constraint_gdf)} polygon(s)")
                constraint_area = self.constraint_gdf.to_crs('EPSG:4326').geometry.apply(
                    lambda geom: geom.area * (111.32 * np.cos(np.radians(geom.centroid.y))) * 111.32
                ).sum()
                print(f"   Constraint total area: {constraint_area:,.2f} km²")
            else:
                print(f"⚠️ Constraint shapefile not found: {constraint_path}")
        
        # Create directory structure
        self.data_dir = self.base_dir / "catchment_delineation_data"
        self.srtm_dir = self.data_dir / "srtm_tiles"
        self.processed_dir = self.data_dir / "processed"
        self.catchments_dir = self.data_dir / "catchments"
        self.temp_dir = self.data_dir / "temp"
        
        for dir_path in [self.data_dir, self.srtm_dir, self.processed_dir, 
                        self.catchments_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize WhiteboxTools
        self.wbt = whitebox.WhiteboxTools()
        self.wbt.set_working_dir(str(self.processed_dir))
        self.wbt.set_verbose_mode(False)
        
        print(f"✅ CatchmentDelineator initialized with WhiteboxTools")
        print(f"📁 Base directory: {self.base_dir}")
        print(f"🌍 Country: {self.country}")
        if bbox:
            print(f"📦 Custom bbox: ({bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f})")

        
    def load_gauge_stations(self, stations_file=None):
        """Load gauge station data from shapefile"""
        if stations_file:
            self.stations_gdf = gpd.read_file(stations_file)
            
            if self.stations_gdf.crs != 'EPSG:4326':
                print(f"Reprojecting from {self.stations_gdf.crs} to EPSG:4326")
                self.stations_gdf = self.stations_gdf.to_crs('EPSG:4326')
            
            self.stations_gdf['longitude'] = self.stations_gdf.geometry.x
            self.stations_gdf['latitude'] = self.stations_gdf.geometry.y
        
        print(f"✅ Loaded {len(self.stations_gdf)} gauge stations")
        return self.stations_gdf
    
    def download_srtm_in_chunks(self, chunk_size_degrees=1.0, buffer=None):
        """
        Download SRTM data in chunks and mosaic them
        """
        dem_output = self.processed_dir / "srtm_mosaic_chunked.tif"
        if dem_output.exists():
            print(f"✅ Using existing DEM mosaic: {dem_output}")
            self.dem_path = dem_output
            return dem_output
        
        if hasattr(self, 'custom_bbox') and self.custom_bbox is not None:
            minx, miny, maxx, maxy = self.custom_bbox
            print(f"\n📦 Using custom bounding box:")
            print(f"   West:  {minx:7.3f}° | South: {miny:7.3f}°")
            print(f"   East:  {maxx:7.3f}° | North: {maxy:7.3f}°")
            width = maxx - minx
            height = maxy - miny
            print(f"   Size: {width:.2f}° x {height:.2f}° (~{width*111:.0f} x {height*111:.0f} km)")
        else:
            if not hasattr(self, 'stations_gdf'):
                raise ValueError("Please load gauge stations first or provide bbox")
            
            bounds = self.stations_gdf.total_bounds
            
            if buffer is None:
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                area_deg2 = width * height
                
                if area_deg2 < 1:
                    buffer = 0.5
                elif area_deg2 < 10:
                    buffer = 1.0
                elif area_deg2 < 50:
                    buffer = 2.0
                else:
                    buffer = 3.0
                
                print(f"\n📊 Study area: {width:.2f}° x {height:.2f}° ({area_deg2:.2f}°²)")
                print(f"🎯 Using adaptive buffer: {buffer}° (~{buffer*111:.0f} km)")
            
            minx, miny, maxx, maxy = bounds
            minx -= buffer
            miny -= buffer
            maxx += buffer
            maxy += buffer
        
        total_width = maxx - minx
        total_height = maxy - miny
        
        print(f"\n📦 Final DEM extent:")
        print(f"   Bounds: ({minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f})")
        print(f"   Coverage: {total_width:.2f}° x {total_height:.2f}°")
        
        x_chunks = int(np.ceil(total_width / chunk_size_degrees))
        y_chunks = int(np.ceil(total_height / chunk_size_degrees))
        total_chunks = x_chunks * y_chunks
        
        print(f"\n📦 Downloading {total_chunks} chunks ({x_chunks}x{y_chunks}) at {chunk_size_degrees}° each...")
        
        chunk_dir = self.temp_dir / "srtm_chunks"
        chunk_dir.mkdir(exist_ok=True)
        
        chunk_files = []
        successful_downloads = 0
        failed_chunks = []
        
        start_time = time.time()
        
        for i in range(x_chunks):
            for j in range(y_chunks):
                chunk_minx = minx + i * chunk_size_degrees
                chunk_maxx = min(minx + (i + 1) * chunk_size_degrees, maxx)
                chunk_miny = miny + j * chunk_size_degrees
                chunk_maxy = min(miny + (j + 1) * chunk_size_degrees, maxy)
                
                chunk_file = chunk_dir / f"chunk_{i}_{j}.tif"
                
                if chunk_file.exists() and chunk_file.stat().st_size > 1000:
                    chunk_files.append(str(chunk_file))
                    successful_downloads += 1
                    print(f"   ✅ Chunk {successful_downloads}/{total_chunks}: Using cached")
                    continue
                
                chunk_num = successful_downloads + len(failed_chunks) + 1
                print(f"   📥 Chunk {chunk_num}/{total_chunks}: "
                      f"({chunk_minx:.3f}, {chunk_miny:.3f}, {chunk_maxx:.3f}, {chunk_maxy:.3f})")
                
                max_retries = 3
                success = False
                
                for attempt in range(max_retries):
                    try:
                        elevation.clip(
                            bounds=(chunk_minx, chunk_miny, chunk_maxx, chunk_maxy),
                            output=str(chunk_file),
                            product='SRTM1'
                        )
                        
                        if chunk_file.exists():
                            file_size = chunk_file.stat().st_size
                            if file_size > 1000:
                                chunk_files.append(str(chunk_file))
                                successful_downloads += 1
                                print(f"      ✅ Success: {file_size:,} bytes")
                                success = True
                                break
                            else:
                                print(f"      ⚠️ File too small: {file_size} bytes (attempt {attempt+1})")
                        
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            
                    except Exception as e:
                        print(f"      ❌ Error (attempt {attempt+1}): {e}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                
                if not success:
                    failed_chunks.append((i, j))
                
                if chunk_num < total_chunks:
                    time.sleep(1)
                
                if chunk_num % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / chunk_num
                    remaining = total_chunks - chunk_num
                    print(f"\n   📊 Progress: {chunk_num}/{total_chunks} | "
                          f"ETA: {avg_time * remaining / 60:.1f} min\n")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"📊 Download: ✅ {successful_downloads}/{total_chunks} | "
              f"❌ {len(failed_chunks)} | ⏱️ {total_time/60:.1f} min")
        
        if successful_downloads == 0:
            print("❌ No chunks downloaded!")
            return None
        
        # Mosaic all chunks
        print(f"\n🔗 Mosaicking {len(chunk_files)} chunks...")
        
        try:
            batch_size = 50
            n_batches = int(np.ceil(len(chunk_files) / batch_size))
            
            if n_batches > 1:
                print(f"   Batched approach: {n_batches} batches")
                intermediate_mosaics = []
                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(chunk_files))
                    batch_chunks = chunk_files[start_idx:end_idx]
                    
                    print(f"   🔗 Batch {batch_idx + 1}/{n_batches}: {len(batch_chunks)} chunks...")
                    
                    src_files = [rasterio.open(f) for f in batch_chunks]
                    mosaic, out_trans = merge(src_files)
                    
                    out_meta = src_files[0].meta.copy()
                    out_meta.update({
                        "driver": "GTiff", "height": mosaic.shape[1],
                        "width": mosaic.shape[2], "transform": out_trans, "compress": "lzw"
                    })
                    
                    intermediate_file = chunk_dir / f"intermediate_mosaic_{batch_idx}.tif"
                    with rasterio.open(intermediate_file, "w", **out_meta) as dest:
                        dest.write(mosaic)
                    
                    for src in src_files:
                        src.close()
                    intermediate_mosaics.append(str(intermediate_file))
                    del mosaic, src_files
                    gc.collect()
                
                # Final merge
                print(f"   🔗 Final merge of {len(intermediate_mosaics)} intermediates...")
                src_files = [rasterio.open(f) for f in intermediate_mosaics]
                mosaic, out_trans = merge(src_files)
                
                out_meta = src_files[0].meta.copy()
                out_meta.update({
                    "driver": "GTiff", "height": mosaic.shape[1],
                    "width": mosaic.shape[2], "transform": out_trans, "compress": "lzw"
                })
                
                with rasterio.open(dem_output, "w", **out_meta) as dest:
                    dest.write(mosaic)
                
                for src in src_files:
                    src.close()
                for f in intermediate_mosaics:
                    Path(f).unlink()
            else:
                src_files = [rasterio.open(f) for f in chunk_files]
                mosaic, out_trans = merge(src_files)
                
                out_meta = src_files[0].meta.copy()
                out_meta.update({
                    "driver": "GTiff", "height": mosaic.shape[1],
                    "width": mosaic.shape[2], "transform": out_trans, "compress": "lzw"
                })
                
                with rasterio.open(dem_output, "w", **out_meta) as dest:
                    dest.write(mosaic)
                
                for src in src_files:
                    src.close()
            
            self.dem_path = dem_output
            
            with rasterio.open(dem_output) as src:
                print(f"\n✅ Mosaic complete!")
                print(f"   📊 Size: {src.width:,} x {src.height:,} pixels")
                print(f"   📏 Resolution: ~{src.res[0]*111*1000:.0f}m")
                print(f"   💾 File size: {dem_output.stat().st_size / 1024 / 1024:.1f} MB")
            
            return dem_output
            
        except Exception as e:
            print(f"❌ Mosaicking failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def preprocess_dem(self, fill_depressions=True):
        """Preprocess DEM - Fill depressions and clean metadata"""
        if not hasattr(self, 'dem_path'):
            candidate = self.processed_dir / "srtm_mosaic_chunked.tif"
            if candidate.exists():
                self.dem_path = candidate
            else:
                raise ValueError("Please download DEM first")

        dem_path = Path(self.dem_path)
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM not found: {dem_path}")

        print("🔧 Preprocessing DEM...")

        if not fill_depressions:
            self.processed_dem_path = dem_path
            return self.processed_dem_path

        filled_dem_raw = self.processed_dir / "dem_filled_raw.tif"
        filled_dem = self.processed_dir / "dem_filled.tif"
        
        if filled_dem.exists() and filled_dem.stat().st_size > 0:
            try:
                with rasterio.open(filled_dem) as src:
                    sample = src.read(1, window=rasterio.windows.Window(0, 0, min(1000, src.width), min(1000, src.height)))
                    if not (np.all(sample == 0) or np.all(np.isnan(sample))):
                        print(f"✅ Using existing filled DEM: {filled_dem}")
                        self.processed_dem_path = filled_dem
                        return self.processed_dem_path
            except Exception as e:
                print(f"⚠️ Existing filled DEM has issues: {e} - regenerating...")

        print("🕳️  Filling depressions (Planchon & Darboux)...")
        
        if filled_dem_raw.exists():
            filled_dem_raw.unlink()
        if filled_dem.exists():
            filled_dem.unlink()
        
        self.wbt.set_verbose_mode(True)
        self.wbt.fill_depressions_planchon_and_darboux(
            dem=str(dem_path),
            output=str(filled_dem_raw)
        )
        self.wbt.set_verbose_mode(False)

        if not filled_dem_raw.exists() or filled_dem_raw.stat().st_size == 0:
            raise RuntimeError(f"WhiteboxTools failed to create filled DEM")

        print("🧹 Cleaning GeoTIFF metadata...")
        
        try:
            with rasterio.open(filled_dem_raw) as src:
                profile = src.profile.copy()
                profile.update(
                    driver='GTiff', dtype=src.dtypes[0],
                    compress='lzw', tiled=True,
                    blockxsize=512, blockysize=512, crs='EPSG:4326'
                )
                if 'photometric' in profile:
                    del profile['photometric']
                
                with rasterio.open(filled_dem, 'w', **profile) as dst:
                    for ji, window in src.block_windows(1):
                        data = src.read(1, window=window)
                        dst.write(data, 1, window=window)
            
            if filled_dem_raw.exists():
                filled_dem_raw.unlink()
                
        except Exception as e:
            print(f"⚠️ Metadata cleaning failed: {e}")
            import subprocess
            try:
                subprocess.run([
                    'gdal_translate', '-of', 'GTiff', '-co', 'COMPRESS=LZW',
                    '-co', 'TILED=YES', '-a_srs', 'EPSG:4326',
                    str(filled_dem_raw), str(filled_dem)
                ], capture_output=True, text=True, check=True)
                if filled_dem_raw.exists():
                    filled_dem_raw.unlink()
            except subprocess.CalledProcessError:
                import shutil
                shutil.move(str(filled_dem_raw), str(filled_dem))

        self.processed_dem_path = filled_dem
        print(f"✅ Preprocessed DEM: {self.processed_dem_path}")
        return self.processed_dem_path
    
    def calculate_flow_accumulation(self):
        """Calculate D8 flow direction and accumulation"""
        if not hasattr(self, 'processed_dem_path') or not Path(self.processed_dem_path).exists():
            self.preprocess_dem()

        print("💧 Calculating flow direction and accumulation...")

        flow_dir_path = self.processed_dir / "flow_direction.tif"
        flow_acc_path = self.processed_dir / "flow_accumulation.tif"

        if (flow_dir_path.exists() and flow_dir_path.stat().st_size > 0 and
            flow_acc_path.exists() and flow_acc_path.stat().st_size > 0):
            
            with rasterio.open(flow_acc_path) as src:
                data = src.read(1)
                max_acc = np.nanmax(data)
                print(f"   Existing flow accumulation max: {max_acc:,.0f}")
                
                if max_acc >= 100000:
                    print(f"✅ Using existing flow direction and accumulation")
                    self.flow_dir_path = flow_dir_path
                    self.flow_acc_path = flow_acc_path
                    return self.flow_dir_path, self.flow_acc_path
                else:
                    print(f"   ⚠️ Values too low - regenerating...")
        
        if flow_dir_path.exists():
            flow_dir_path.unlink()
        if flow_acc_path.exists():
            flow_acc_path.unlink()

        print("🧭 Calculating flow direction...")
        self.wbt.set_verbose_mode(True)
        self.wbt.d8_pointer(
            dem=str(self.processed_dem_path),
            output=str(flow_dir_path)
        )
        self.wbt.set_verbose_mode(False)

        if not flow_dir_path.exists() or flow_dir_path.stat().st_size == 0:
            raise RuntimeError("Failed to create flow direction")

        print("💧 Calculating flow accumulation...")
        self.wbt.set_verbose_mode(True)
        self.wbt.d8_flow_accumulation(
            i=str(flow_dir_path),
            output=str(flow_acc_path),
            pntr=True
        )
        self.wbt.set_verbose_mode(False)

        if not flow_acc_path.exists() or flow_acc_path.stat().st_size == 0:
            raise RuntimeError("Failed to create flow accumulation")

        with rasterio.open(flow_acc_path) as src:
            data = src.read(1)
            max_acc = np.nanmax(data)
            print(f"   Flow accumulation max: {max_acc:,.0f} cells")

        self.flow_dir_path = flow_dir_path
        self.flow_acc_path = flow_acc_path
        return self.flow_dir_path, self.flow_acc_path

    def _manual_snap_to_stream(self, lon, lat, search_radius_pixels=100,
                               min_flow_acc=10000, progressive=True):
        """
        Snap a point to the nearest stream cell using a progressive search strategy.

        Uses a small initial radius and only expands if no valid stream cell is found.
        This prevents jumping to a larger river at confluences/forks.

        Parameters:
        -----------
        lon, lat : float
            Original point coordinates in EPSG:4326
        search_radius_pixels : int
            Maximum search radius in pixels
        min_flow_acc : int
            Minimum flow accumulation to accept as a valid stream snap.
            The progressive search stops as soon as a cell >= this value is found.
        progressive : bool
            If True, search grows from 5px outward, stopping at the smallest radius
            that finds a cell >= min_flow_acc. This avoids snapping to a different
            (larger) river further away at confluences.
            If False, searches the full radius at once (original behaviour).

        Returns:
        --------
        tuple : (snapped_lon, snapped_lat, flow_acc_value, radius_used_px)
        """
        with rasterio.open(self.flow_acc_path) as src:
            try:
                row, col = rowcol(src.transform, lon, lat)
            except Exception:
                return lon, lat, 0, 0

            if row < 0 or row >= src.height or col < 0 or col >= src.width:
                return lon, lat, 0, 0

            nodata = src.nodata

            if not progressive:
                # --- Original behaviour: search full radius at once ---
                row_min = max(0, row - search_radius_pixels)
                row_max = min(src.height, row + search_radius_pixels + 1)
                col_min = max(0, col - search_radius_pixels)
                col_max = min(src.width, col + search_radius_pixels + 1)
                window = rasterio.windows.Window(
                    col_off=col_min, row_off=row_min,
                    width=col_max - col_min, height=row_max - row_min
                )
                flow_acc_data = src.read(1, window=window).astype(float)
                if nodata is not None:
                    flow_acc_data = np.where(flow_acc_data == nodata, 0, flow_acc_data)
                flow_acc_data = np.where(np.isnan(flow_acc_data), 0, flow_acc_data)
                max_idx = np.unravel_index(np.argmax(flow_acc_data), flow_acc_data.shape)
                snapped_lon, snapped_lat = src.xy(row_min + max_idx[0], col_min + max_idx[1])
                return snapped_lon, snapped_lat, float(flow_acc_data[max_idx]), search_radius_pixels

            # --- Progressive search: grow radius until we find a valid stream cell ---
            # These radii approximate: 150m, 300m, 600m, 1km, 1.5km, 2.25km, max
            radii_to_try = [5, 10, 20, 35, 50, 75, search_radius_pixels]
            radii_to_try = sorted(set(r for r in radii_to_try if r <= search_radius_pixels))
            if search_radius_pixels not in radii_to_try:
                radii_to_try.append(search_radius_pixels)

            last_data = None
            last_row_min = 0
            last_col_min = 0

            for radius in radii_to_try:
                row_min = max(0, row - radius)
                row_max = min(src.height, row + radius + 1)
                col_min = max(0, col - radius)
                col_max = min(src.width, col + radius + 1)

                window = rasterio.windows.Window(
                    col_off=col_min, row_off=row_min,
                    width=col_max - col_min, height=row_max - row_min
                )
                flow_acc_data = src.read(1, window=window).astype(float)
                if nodata is not None:
                    flow_acc_data = np.where(flow_acc_data == nodata, 0, flow_acc_data)
                flow_acc_data = np.where(np.isnan(flow_acc_data), 0, flow_acc_data)

                last_data = flow_acc_data
                last_row_min = row_min
                last_col_min = col_min

                max_val = float(np.max(flow_acc_data))
                if max_val >= min_flow_acc:
                    # Found a valid stream - stop here to avoid jumping rivers
                    max_idx = np.unravel_index(np.argmax(flow_acc_data), flow_acc_data.shape)
                    snapped_lon, snapped_lat = src.xy(
                        last_row_min + max_idx[0], last_col_min + max_idx[1]
                    )
                    return snapped_lon, snapped_lat, max_val, radius

            # Nothing found above threshold - return best cell found at max radius
            max_idx = np.unravel_index(np.argmax(last_data), last_data.shape)
            snapped_lon, snapped_lat = src.xy(
                last_row_min + max_idx[0], last_col_min + max_idx[1]
            )
            return snapped_lon, snapped_lat, float(last_data[max_idx]), search_radius_pixels

    def snap_outlets_to_streams(self, threshold=1000, search_radius_pixels=100,
                                 min_flow_acc=10000, progressive=True):
        """
        Snap gauge stations to streams using manual raster-based approach.

        Parameters:
        -----------
        threshold : int
            Flow accumulation threshold for stream extraction (visualization only)
        search_radius_pixels : int
            Maximum search radius in pixels (default 100 = ~3 km at 30m resolution).
            With progressive=True this is only reached if no stream found closer.
        min_flow_acc : int
            Minimum flow accumulation to accept as a valid stream snap (default 10,000
            cells ≈ ~9 km² contributing area at 30m resolution). Progressive search
            stops as soon as it finds a cell exceeding this threshold.
        progressive : bool
            If True (default), use progressive radius search:
              5px → 10px → 20px → 35px → 50px → 75px → search_radius_pixels
            Stops at the smallest radius that finds a cell >= min_flow_acc.
            Prevents jumping to a different (larger) river at confluences/forks.
            If False, searches the full search_radius_pixels at once.
        """
        if not hasattr(self, 'flow_acc_path'):
            self.calculate_flow_accumulation()

        print(f"🎯 Snapping outlets to streams (manual raster-based approach)...")
        if progressive:
            print(f"   Strategy: Progressive radius (stops at first valid stream cell)")
            print(f"   Radii tried: 5→10→20→35→50→75→{search_radius_pixels} px")
            print(f"   Min flow acc threshold: {min_flow_acc:,} cells "
                  f"(~{min_flow_acc * 900 / 1e6:.1f} km² at 30m)")
        else:
            print(f"   Strategy: Single full-radius search")
        print(f"   Max search radius: {search_radius_pixels} px "
              f"(~{search_radius_pixels * 30 / 1000:.1f} km)")

        # Extract streams (for reference/visualization)
        streams_path = str(self.processed_dir / "streams.tif")
        if not Path(streams_path).exists():
            print(f"🌊 Extracting streams (threshold: {threshold} cells)...")
            self.wbt.extract_streams(
                flow_accum=str(self.flow_acc_path),
                output=streams_path,
                threshold=threshold
            )

        snapped_records = []

        print(f"\n   {'Station':<15} {'Original (lon, lat)':<30} {'Snapped (lon, lat)':<30} "
              f"{'Flow Acc':>15} {'Dist (km)':>10} {'Radius':>8}")
        print(f"   {'─'*115}")

        for idx, station in self.stations_gdf.iterrows():
            orig_lon = station.geometry.x
            orig_lat = station.geometry.y

            snap_lon, snap_lat, flow_acc, radius_used = self._manual_snap_to_stream(
                orig_lon, orig_lat,
                search_radius_pixels=search_radius_pixels,
                min_flow_acc=min_flow_acc,
                progressive=progressive
            )

            dist_km = np.sqrt(
                ((snap_lon - orig_lon) * 111.32 * np.cos(np.radians(orig_lat)))**2 +
                ((snap_lat - orig_lat) * 111.32)**2
            )

            station_id = station.get('station_id', idx)

            if flow_acc > 100000:
                status = "✅"
            elif flow_acc >= min_flow_acc:
                status = "⚠️"
            else:
                status = "❌"

            print(f"   {status} {str(station_id):<13} "
                  f"({orig_lon:8.4f}, {orig_lat:7.4f})  →  "
                  f"({snap_lon:8.4f}, {snap_lat:7.4f})  "
                  f"{flow_acc:>15,.0f}  "
                  f"{dist_km:>8.2f}  "
                  f"~{radius_used:>3d}px")

            record = station.to_dict()
            record['geometry'] = Point(snap_lon, snap_lat)
            record['snap_lon'] = snap_lon
            record['snap_lat'] = snap_lat
            record['orig_lon'] = orig_lon
            record['orig_lat'] = orig_lat
            record['flow_acc'] = flow_acc
            record['snap_dist_km'] = dist_km
            record['snap_radius_px'] = radius_used
            snapped_records.append(record)

        self.snapped_outlets = gpd.GeoDataFrame(snapped_records, crs='EPSG:4326')

        snapped_file = self.processed_dir / "snapped_outlets.shp"
        self.snapped_outlets.to_file(snapped_file)

        n_good = len(self.snapped_outlets[self.snapped_outlets['flow_acc'] > 100000])
        n_ok   = len(self.snapped_outlets[(self.snapped_outlets['flow_acc'] >= min_flow_acc) &
                                           (self.snapped_outlets['flow_acc'] <= 100000)])
        n_bad  = len(self.snapped_outlets[self.snapped_outlets['flow_acc'] < min_flow_acc])

        print(f"\n   📊 Snapping summary:")
        print(f"      ✅ Well-snapped  (flow_acc > 100k):               {n_good}")
        print(f"      ⚠️  Moderate     ({min_flow_acc:,}–100k cells): {n_ok}")
        print(f"      ❌ Poorly snapped (< {min_flow_acc:,} cells):      {n_bad}")

        if n_bad > 0:
            print(f"\n   💡 For poorly snapped stations:")
            print(f"      - Increase search_radius_pixels (currently {search_radius_pixels})")
            print(f"      - Lower min_flow_acc (currently {min_flow_acc:,})")
            print(f"      - Run diagnose_station(id) to inspect the location in detail")

        self.streams_path = Path(streams_path)
        print(f"\n✅ Snapped {len(self.snapped_outlets)} outlets")
        return self.snapped_outlets

    def apply_constraint(self, catchment_gdf, station_name):
        """Apply constraint by subtracting the constraint area from catchment"""
        if self.constraint_gdf is None:
            return catchment_gdf
        
        try:
            catchment_geom = catchment_gdf.geometry.iloc[0]
            constraint_geom = self.constraint_gdf.geometry.unary_union
            
            if not catchment_geom.intersects(constraint_geom):
                return catchment_gdf
            
            intersection = catchment_geom.intersection(constraint_geom)
            intersection_area = intersection.area * (111.32 * np.cos(np.radians(intersection.centroid.y))) * 111.32
            
            print(f"    🔧 Constraint overlap: {intersection_area:,.2f} km²")
            
            corrected_geom = catchment_geom.difference(constraint_geom)
            
            if corrected_geom.is_empty:
                print(f"    ⚠️ Catchment entirely within constraint - keeping original")
                return catchment_gdf
            
            if corrected_geom.geom_type == 'MultiPolygon':
                largest_geom = max(corrected_geom.geoms, key=lambda g: g.area)
                corrected_geom = largest_geom
                print(f"    📐 MultiPolygon → using largest part")
            
            corrected_gdf = catchment_gdf.copy()
            corrected_gdf.geometry = [corrected_geom]
            
            new_area = corrected_geom.area * (111.32 * np.cos(np.radians(corrected_geom.centroid.y))) * 111.32
            old_area = catchment_gdf['area_km2'].iloc[0]
            corrected_gdf['area_km2'] = new_area
            
            print(f"    ✂️ Area: {old_area:,.2f} → {new_area:,.2f} km² (removed {old_area - new_area:,.2f} km²)")
            
            return corrected_gdf
            
        except Exception as e:
            print(f"    ⚠️ Constraint failed: {e}")
            return catchment_gdf
    
    def _create_pour_point_raster(self, outlet, station_name):
        """
        Create a pour point raster from a single outlet point.
        This is more reliable than using shapefile pour points with WhiteboxTools watershed().
        
        Parameters:
        -----------
        outlet : GeoSeries
            Single outlet with geometry
        station_name : str
            Station name for file naming
            
        Returns:
        --------
        Path : Path to the pour point raster
        """
        pour_point_raster = self.temp_dir / f"pourpoint_{station_name}.tif"
        
        with rasterio.open(self.flow_dir_path) as src:
            # Get the pixel location of the pour point
            lon = outlet.geometry.x
            lat = outlet.geometry.y
            
            row, col = rowcol(src.transform, lon, lat)
            
            # Create an empty raster matching the flow direction grid
            data = np.zeros((src.height, src.width), dtype=np.uint8)
            
            # Set the pour point pixel to 1
            if 0 <= row < src.height and 0 <= col < src.width:
                data[row, col] = 1
            else:
                print(f"    ⚠️ Pour point ({lon:.4f}, {lat:.4f}) outside raster bounds!")
                return None
            
            # Write the pour point raster
            profile = src.profile.copy()
            profile.update(dtype='uint8', nodata=0, compress='lzw')
            
            with rasterio.open(pour_point_raster, 'w', **profile) as dst:
                dst.write(data, 1)
        
        return pour_point_raster

    def delineate_catchments(self, use_raster_pourpoints=True):
        """
        Delineate catchments for all snapped outlets.
        
        Parameters:
        -----------
        use_raster_pourpoints : bool
            If True, creates raster-based pour points (more reliable for large DEMs).
            If False, uses shapefile-based pour points (original method).
        """
        if not hasattr(self, 'snapped_outlets'):
            self.snap_outlets_to_streams()
        
        print(f"\n🏔️  Delineating {len(self.snapped_outlets)} catchments...")
        if use_raster_pourpoints:
            print(f"   Method: Raster-based pour points (recommended for large DEMs)")
        else:
            print(f"   Method: Shapefile-based pour points")
        
        if self.constraint_gdf is not None:
            print(f"🔧 Constraint shapefile will be applied")
        
        catchments = []
        failed_stations = []
        
        for idx, outlet in self.snapped_outlets.iterrows():
            station_id_raw = outlet.get('station_id', idx)
            
            if isinstance(station_id_raw, (int, float)):
                if station_id_raw == int(station_id_raw):
                    station_id_str = str(int(station_id_raw))
                else:
                    station_id_str = str(station_id_raw).replace('.', '')
            else:
                station_id_str = str(station_id_raw).replace('.', '')
            
            station_name = f"shape_{station_id_str.zfill(4)}"
            flow_acc_val = outlet.get('flow_acc', 'N/A')
            print(f"\n  Processing {station_name} (ID: {station_id_raw}, flow_acc: {flow_acc_val:,.0f})...")
            
            # Check if catchment already exists
            watershed_vector = str(self.catchments_dir / f"catchment_{station_name}.shp")
            if Path(watershed_vector).exists():
                print(f"    ✅ Using existing catchment")
                catchment_gdf = gpd.read_file(watershed_vector)
                
                if self.constraint_gdf is not None:
                    catchment_gdf = self.apply_constraint(catchment_gdf, station_name)
                    catchment_gdf.to_file(watershed_vector)
                
                catchments.append(catchment_gdf)
                continue
            
            try:
                # === WATERSHED DELINEATION ===
                watershed_raster = self.temp_dir / f"watershed_{station_name}.tif"
                
                if use_raster_pourpoints:
                    # Method 1: Raster-based pour point (MORE RELIABLE)
                    pour_point_raster = self._create_pour_point_raster(outlet, station_name)
                    
                    if pour_point_raster is None:
                        print(f"    ❌ Could not create pour point raster")
                        failed_stations.append(station_name)
                        continue
                    
                    # Use unnest_basins or watershed with raster pour points
                    # WhiteboxTools watershed() accepts raster pour points too
                    self.wbt.watershed(
                        d8_pntr=str(self.flow_dir_path),
                        pour_pts=str(pour_point_raster),
                        output=str(watershed_raster)
                    )
                    
                    # Clean up pour point raster
                    if pour_point_raster.exists():
                        pour_point_raster.unlink()
                    
                else:
                    # Method 2: Shapefile-based pour point (original)
                    outlet_file = self.temp_dir / f"outlet_{station_name}.shp"
                    outlet_gdf = gpd.GeoDataFrame([outlet], crs='EPSG:4326')
                    outlet_gdf.to_file(outlet_file)
                    
                    self.wbt.watershed(
                        d8_pntr=str(self.flow_dir_path),
                        pour_pts=str(outlet_file),
                        output=str(watershed_raster)
                    )
                    
                    # Clean up
                    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                        f = outlet_file.with_suffix(ext)
                        if f.exists():
                            f.unlink()
                
                # === CHECK WATERSHED RASTER ===
                if not watershed_raster.exists():
                    print(f"    ❌ Watershed raster not created")
                    failed_stations.append(station_name)
                    continue
                
                # Validate: check that the raster has actual watershed pixels
                with rasterio.open(watershed_raster) as src:
                    data = src.read(1)
                    nodata = src.nodata
                    if nodata is not None:
                        valid_pixels = np.sum(data != nodata)
                    else:
                        valid_pixels = np.sum(data > 0)
                    
                    print(f"    📊 Watershed raster: {valid_pixels:,} valid pixels")
                    
                    if valid_pixels == 0:
                        print(f"    ❌ Empty watershed - pour point likely not on flow network")
                        print(f"       Try increasing search_radius_pixels in snap_outlets_to_streams()")
                        failed_stations.append(station_name)
                        if watershed_raster.exists():
                            watershed_raster.unlink()
                        continue
                
                # === CLIP WATERSHED RASTER BEFORE VECTORIZING ===
                # The watershed raster covers the full DEM extent (330M+ pixels).
                # WhiteboxTools raster_to_vector_polygons fails on such large rasters.
                # Clip to the tight bounding box of the valid pixels first.
                watershed_raster_clipped = self.temp_dir / f"watershed_{station_name}_clipped.tif"
                try:
                    with rasterio.open(watershed_raster) as src:
                        data = src.read(1)
                        nodata = src.nodata if src.nodata is not None else 0
                        mask = data != nodata
                        rows = np.any(mask, axis=1)
                        cols = np.any(mask, axis=0)
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        
                        # Add a small buffer of pixels around the extent
                        pad = 5
                        rmin = max(0, rmin - pad)
                        rmax = min(src.height - 1, rmax + pad)
                        cmin = max(0, cmin - pad)
                        cmax = min(src.width - 1, cmax + pad)
                        
                        window = rasterio.windows.Window(
                            col_off=cmin, row_off=rmin,
                            width=cmax - cmin + 1, height=rmax - rmin + 1
                        )
                        clipped_data = src.read(1, window=window)
                        clipped_transform = src.window_transform(window)
                        
                        profile = src.profile.copy()
                        profile.update(
                            height=clipped_data.shape[0],
                            width=clipped_data.shape[1],
                            transform=clipped_transform,
                            compress='lzw'
                        )
                        with rasterio.open(watershed_raster_clipped, 'w', **profile) as dst:
                            dst.write(clipped_data, 1)
                    
                    print(f"    ✂️  Clipped raster: {clipped_data.shape[1]}×{clipped_data.shape[0]} px "
                          f"(was {src.width}×{src.height})")
                    
                    # Remove the full-extent raster now
                    if watershed_raster.exists():
                        watershed_raster.unlink()
                    
                except Exception as e:
                    print(f"    ⚠️  Clipping failed ({e}), using full raster")
                    watershed_raster_clipped = watershed_raster
                
                # === CONVERT TO VECTOR ===
                # Try WhiteboxTools first, fall back to GDAL polygonize
                vector_success = False
                
                # Attempt 1: WhiteboxTools raster_to_vector_polygons on clipped raster
                try:
                    self.wbt.raster_to_vector_polygons(
                        i=str(watershed_raster_clipped),
                        output=watershed_vector
                    )
                    if Path(watershed_vector).exists() and Path(watershed_vector).stat().st_size > 0:
                        vector_success = True
                        print(f"    ✅ Vectorized with WhiteboxTools")
                except Exception as e:
                    print(f"    ⚠️  WhiteboxTools vectorize failed: {e}")
                
                # Attempt 2: GDAL polygonize (much more robust for large rasters)
                if not vector_success:
                    try:
                        import subprocess
                        # Convert raster values: set catchment pixels to 1, rest to nodata
                        watershed_binary = self.temp_dir / f"watershed_{station_name}_binary.tif"
                        with rasterio.open(watershed_raster_clipped) as src:
                            data = src.read(1)
                            nd = src.nodata if src.nodata is not None else 0
                            binary = np.where(data != nd, 1, 0).astype(np.uint8)
                            profile = src.profile.copy()
                            profile.update(dtype='uint8', nodata=0, compress='lzw')
                            with rasterio.open(watershed_binary, 'w', **profile) as dst:
                                dst.write(binary, 1)
                        
                        # Use gdal_polygonize
                        watershed_vector_gpkg = str(self.temp_dir / f"watershed_{station_name}.gpkg")
                        result = subprocess.run([
                            'gdal_polygonize.py',
                            str(watershed_binary),
                            '-f', 'GPKG',
                            watershed_vector_gpkg,
                            'watershed', 'value'
                        ], capture_output=True, text=True, timeout=120)
                        
                        if watershed_binary.exists():
                            watershed_binary.unlink()
                        
                        if Path(watershed_vector_gpkg).exists():
                            gdf_temp = gpd.read_file(watershed_vector_gpkg)
                            # Keep only polygons with value=1 (the catchment)
                            gdf_temp = gdf_temp[gdf_temp['value'] == 1]
                            if len(gdf_temp) > 0:
                                gdf_temp.to_file(watershed_vector)
                                Path(watershed_vector_gpkg).unlink()
                                vector_success = True
                                print(f"    ✅ Vectorized with GDAL polygonize")
                            else:
                                print(f"    ⚠️  GDAL polygonize: no valid polygons")
                        else:
                            print(f"    ⚠️  GDAL polygonize: output not created")
                            print(f"         stderr: {result.stderr[:200]}")
                    
                    except Exception as e:
                        print(f"    ⚠️  GDAL polygonize failed: {e}")
                
                # Attempt 3: rasterio + shapely directly (pure Python, always works)
                if not vector_success:
                    try:
                        from rasterio.features import shapes
                        from shapely.geometry import shape
                        
                        with rasterio.open(watershed_raster_clipped) as src:
                            data = src.read(1)
                            nd = src.nodata if src.nodata is not None else 0
                            mask = (data != nd).astype(np.uint8)
                            transform = src.transform
                            crs = src.crs
                        
                        polygons = []
                        for geom, val in shapes(mask, mask=mask, transform=transform):
                            if val == 1:
                                polygons.append(shape(geom))
                        
                        if polygons:
                            from shapely.ops import unary_union
                            merged = unary_union(polygons)
                            gdf_temp = gpd.GeoDataFrame(
                                [{'geometry': merged, 'value': 1}],
                                crs=crs
                            )
                            gdf_temp.to_file(watershed_vector)
                            vector_success = True
                            print(f"    ✅ Vectorized with rasterio.features.shapes")
                        else:
                            print(f"    ⚠️  rasterio shapes: no polygons found")
                    
                    except Exception as e:
                        print(f"    ⚠️  rasterio shapes failed: {e}")
                
                # Clean up clipped raster
                if watershed_raster_clipped.exists():
                    watershed_raster_clipped.unlink()
                
                if not vector_success or not Path(watershed_vector).exists():
                    print(f"    ❌ Vector conversion failed (all methods exhausted)")
                    failed_stations.append(station_name)
                    continue
                
                # === LOAD AND ADD METADATA ===
                catchment_gdf = gpd.read_file(watershed_vector)
                if len(catchment_gdf) == 0:
                    print(f"    ❌ Empty catchment vector")
                    failed_stations.append(station_name)
                    continue
                
                # If multiple polygons, take the one containing the pour point or the largest
                if len(catchment_gdf) > 1:
                    pour_point = outlet.geometry
                    containing = catchment_gdf[catchment_gdf.contains(pour_point)]
                    if len(containing) > 0:
                        catchment_gdf = containing.iloc[[0]].copy()
                        print(f"    📐 Multiple polygons → selected one containing pour point")
                    else:
                        # Take the largest
                        catchment_gdf['_area'] = catchment_gdf.geometry.area
                        catchment_gdf = catchment_gdf.nlargest(1, '_area').drop(columns=['_area'])
                        print(f"    📐 Multiple polygons → selected largest")
                
                catchment_gdf = catchment_gdf.reset_index(drop=True)
                catchment_gdf['station_name'] = station_name
                
                catchment_gdf['area_km2'] = catchment_gdf.to_crs('EPSG:4326').geometry.apply(
                    lambda geom: geom.area * (111.32 * np.cos(np.radians(geom.centroid.y))) * 111.32
                )
                
                # Copy all attributes from original station
                for col in outlet.index:
                    if col != 'geometry':
                        catchment_gdf[col] = outlet[col]
                
                # Apply constraint
                catchment_gdf = self.apply_constraint(catchment_gdf, station_name)
                
                # Save
                catchment_gdf.to_file(watershed_vector)
                
                catchments.append(catchment_gdf)
                print(f"    ✅ Success: {catchment_gdf['area_km2'].iloc[0]:,.2f} km²")
                
            except Exception as e:
                print(f"    ❌ Failed for {station_name}: {e}")
                import traceback
                traceback.print_exc()
                failed_stations.append(station_name)
                # Clean up temp files on failure
                for temp_file in [
                    self.temp_dir / f"watershed_{station_name}.tif",
                    self.temp_dir / f"pourpoint_{station_name}.tif",
                ]:
                    if temp_file.exists():
                        temp_file.unlink()
        
        # === SUMMARY ===
        print(f"\n{'='*60}")
        if catchments:
            self.catchments_gdf = gpd.GeoDataFrame(
                pd.concat(catchments, ignore_index=True),
                crs='EPSG:4326'
            )
            
            all_catchments_file = self.catchments_dir / "all_catchments.shp"
            self.catchments_gdf.to_file(all_catchments_file)
            
            print(f"✅ Delineated {len(catchments)} catchments")
            print(f"💾 Saved to: {all_catchments_file}")
            print(f"\n📊 Statistics:")
            print(f"   Total area: {self.catchments_gdf['area_km2'].sum():,.0f} km²")
            print(f"   Largest: {self.catchments_gdf['area_km2'].max():,.0f} km²")
            print(f"   Smallest: {self.catchments_gdf['area_km2'].min():,.0f} km²")
        
        if failed_stations:
            print(f"\n❌ Failed stations ({len(failed_stations)}):")
            for s in failed_stations:
                print(f"   - {s}")
            print(f"\n💡 Troubleshooting tips:")
            print(f"   1. Increase search_radius_pixels (e.g., 100 or 200)")
            print(f"   2. Check station coordinates in QGIS against DEM/streams")
            print(f"   3. Station may be outside DEM extent - check bbox")
            print(f"   4. Run: delineator.diagnose_station('station_id') for details")
        
        return self.catchments_gdf if catchments else None
    
    def diagnose_station(self, station_id):
        """
        Diagnose why a station fails during delineation.
        Prints detailed information about the station's location relative to the DEM and flow network.
        
        Parameters:
        -----------
        station_id : int or str
            Station ID to diagnose
        """
        print(f"\n🔍 Diagnosing station {station_id}...")
        print(f"{'='*60}")
        
        # Find the station
        station = None
        for idx, row in self.stations_gdf.iterrows():
            sid = row.get('station_id', idx)
            if str(int(sid) if isinstance(sid, float) else sid) == str(station_id):
                station = row
                break
        
        if station is None:
            print(f"❌ Station {station_id} not found in loaded stations")
            return
        
        orig_lon = station.geometry.x
        orig_lat = station.geometry.y
        print(f"\n📍 Original coordinates: ({orig_lon:.6f}, {orig_lat:.6f})")
        
        # Check if within DEM bounds
        with rasterio.open(self.dem_path) as src:
            dem_bounds = src.bounds
            in_dem = (dem_bounds.left <= orig_lon <= dem_bounds.right and
                      dem_bounds.bottom <= orig_lat <= dem_bounds.top)
            print(f"\n📦 DEM bounds: ({dem_bounds.left:.3f}, {dem_bounds.bottom:.3f}, "
                  f"{dem_bounds.right:.3f}, {dem_bounds.top:.3f})")
            print(f"   Station in DEM: {'✅ Yes' if in_dem else '❌ No'}")
            
            if not in_dem:
                print(f"   ⚠️ Station is OUTSIDE the DEM! Extend your bbox.")
                return
        
        # Check flow accumulation at various radii
        print(f"\n💧 Flow accumulation analysis (non-progressive, reports best cell in window):")
        for radius in [10, 25, 50, 100, 200]:
            snap_lon, snap_lat, flow_acc, radius_used = self._manual_snap_to_stream(
                orig_lon, orig_lat, search_radius_pixels=radius, progressive=False
            )
            dist = np.sqrt(
                ((snap_lon - orig_lon) * 111.32 * np.cos(np.radians(orig_lat)))**2 +
                ((snap_lat - orig_lat) * 111.32)**2
            )
            print(f"   Radius {radius:>3d}px (~{radius*30/1000:.1f}km): "
                  f"flow_acc={flow_acc:>15,.0f}  snap_dist={dist:.2f}km  "
                  f"→ ({snap_lon:.5f}, {snap_lat:.5f})")
        
        # Check flow accumulation directly at the point
        with rasterio.open(self.flow_acc_path) as src:
            try:
                row, col = rowcol(src.transform, orig_lon, orig_lat)
                if 0 <= row < src.height and 0 <= col < src.width:
                    local_acc = src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0]
                    print(f"\n   Flow acc at exact station location: {local_acc:,.0f}")
                else:
                    print(f"\n   ⚠️ Station pixel outside raster!")
            except Exception as e:
                print(f"\n   ❌ Error reading flow acc: {e}")
        
        # Check DEM value
        with rasterio.open(self.processed_dem_path) as src:
            try:
                row, col = rowcol(src.transform, orig_lon, orig_lat)
                if 0 <= row < src.height and 0 <= col < src.width:
                    dem_val = src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0]
                    print(f"   DEM elevation at station: {dem_val:.1f} m")
                    if dem_val == src.nodata:
                        print(f"   ⚠️ Station is on a NODATA pixel in the DEM!")
            except Exception as e:
                print(f"   ❌ Error reading DEM: {e}")
        
        print(f"\n{'='*60}")


# =============================================================================
# Predefined regions
# =============================================================================
REGIONS = {
    "upper_indus": {
        "label": "Upper Indus Basin",
        "bbox": (67.5, 30.0, 82.0, 37.5),
        "country": "Pakistan",
        "stations": "Indus_stations.shp",
        "constraint": "constraint_catchment_Indus.shp",
    },
    "western_himalaya": {
        "label": "Western Indian Himalaya (Kashmir to Uttarakhand)",
        "bbox": (72.5, 28.5, 83.0, 37.0),
        "country": "India",
        "stations": None,
        "constraint": None,
    },
    "ganges": {
        "label": "Ganges Basin (Nepal + Bhutan)",
        "bbox": (80.0, 26.3, 92.5, 30.5),
        "country": "Ganges",
        "stations": "Ganges_stations.shp",
        "constraint": None,
    },
}


def main():
    """Main function with region selection via CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Catchment delineation with predefined or custom bounding boxes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Predefined regions:
  {chr(10).join(f'  {k:20s} {v["label"]}  bbox={v["bbox"]}' for k, v in REGIONS.items())}

Examples:
  python catchment_delineation.py --region upper_indus
  python catchment_delineation.py --region western_himalaya --stations CWC_stations_pre2015.gpkg
  python catchment_delineation.py --bbox 72.5 28.5 81.0 37.0 --country India --stations stations.shp
        """,
    )

    parser.add_argument("--region", type=str, choices=list(REGIONS.keys()),
                        help="Predefined region name")
    parser.add_argument("--bbox", type=float, nargs=4,
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                        help="Custom bounding box (overrides --region bbox)")
    parser.add_argument("--base-dir", type=str,
                        default="/home/jberg/Raven-world/Catchment_delineation",
                        help="Base directory for outputs")
    parser.add_argument("--stations", type=str, default=None,
                        help="Path to gauge stations file (shapefile or geopackage)")
    parser.add_argument("--constraint", type=str, default=None,
                        help="Path to constraint shapefile")
    parser.add_argument("--country", type=str, default=None,
                        help="Country label (overrides --region country)")

    args = parser.parse_args()

    # Resolve region config
    if args.region:
        region = REGIONS[args.region]
        bbox = args.bbox or region["bbox"]
        country = args.country or region["country"]
        stations_file = args.stations
        constraint_file = args.constraint
        # Fall back to region defaults for stations/constraint if not given
        if stations_file is None and region["stations"]:
            default_stations = Path(args.base_dir) / region["stations"]
            if default_stations.exists():
                stations_file = str(default_stations)
        if constraint_file is None and region["constraint"]:
            default_constraint = Path(args.base_dir) / region["constraint"]
            if default_constraint.exists():
                constraint_file = str(default_constraint)
        label = region["label"]
    elif args.bbox:
        bbox = tuple(args.bbox)
        country = args.country or "Custom"
        stations_file = args.stations
        constraint_file = args.constraint
        label = f"Custom region ({bbox[0]:.1f}W, {bbox[1]:.1f}S, {bbox[2]:.1f}E, {bbox[3]:.1f}N)"
    else:
        parser.error("Must specify --region or --bbox")
        return

    try:
        print("\n" + "=" * 70)
        print(f"CATCHMENT DELINEATION: {label}")
        print("=" * 70)

        delineator = CatchmentDelineator(
            base_dir=args.base_dir,
            country=country,
            bbox=bbox,
            constraint_shapefile=constraint_file,
        )

        if stations_file and Path(stations_file).exists():
            stations_gdf = delineator.load_gauge_stations(stations_file=stations_file)
        else:
            if stations_file:
                print(f"Warning: stations file not found: {stations_file}")

        # Download DEM
        dem_path = delineator.download_srtm_in_chunks(chunk_size_degrees=1.5)
        if dem_path is None:
            print("DEM download failed")
            return

        # Process DEM
        processed_dem = delineator.preprocess_dem()

        # Calculate flow
        flow_dir, flow_acc = delineator.calculate_flow_accumulation()

        # Snap outlets with manual raster-based approach
        if hasattr(delineator, 'stations_gdf'):
            snapped_outlets = delineator.snap_outlets_to_streams(
                search_radius_pixels=100,
                min_flow_acc=10000,
                progressive=True,
            )

            # Delineate with raster-based pour points
            catchments = delineator.delineate_catchments(
                use_raster_pourpoints=True,
            )

            if catchments is not None:
                print(f"\nSUCCESS! Delineated {len(catchments)} catchments for {label}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()