import pathlib
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge
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
                # Calculate constraint area
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
        
        # Create all directories
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
            
            # Ensure it's in WGS84
            if self.stations_gdf.crs != 'EPSG:4326':
                print(f"Reprojecting from {self.stations_gdf.crs} to EPSG:4326")
                self.stations_gdf = self.stations_gdf.to_crs('EPSG:4326')
            
            # Extract coordinates
            self.stations_gdf['longitude'] = self.stations_gdf.geometry.x
            self.stations_gdf['latitude'] = self.stations_gdf.geometry.y
        
        print(f"✅ Loaded {len(self.stations_gdf)} gauge stations")
        return self.stations_gdf
    
    def download_srtm_in_chunks(self, chunk_size_degrees=1.0, buffer=None):
        """
        Download SRTM data in chunks and mosaic them
        
        Parameters:
        -----------
        chunk_size_degrees : float
            Size of each download chunk in degrees (default: 1.0)
        buffer : float, optional
            Buffer in degrees. If None, uses adaptive buffer based on area size
            Only used if custom_bbox is not provided
        """
        # Check if mosaic already exists
        dem_output = self.processed_dir / "srtm_mosaic_chunked.tif"
        if dem_output.exists():
            print(f"✅ Using existing DEM mosaic: {dem_output}")
            self.dem_path = dem_output
            return dem_output
        
        # Use custom bbox if provided
        if hasattr(self, 'custom_bbox') and self.custom_bbox is not None:
            minx, miny, maxx, maxy = self.custom_bbox
            print(f"\n📦 Using custom bounding box:")
            print(f"   West:  {minx:7.3f}° (Longitude)")
            print(f"   South: {miny:7.3f}° (Latitude)")
            print(f"   East:  {maxx:7.3f}° (Longitude)")
            print(f"   North: {maxy:7.3f}° (Latitude)")
            width = maxx - minx
            height = maxy - miny
            print(f"   Size: {width:.2f}° x {height:.2f}° (~{width*111:.0f} x {height*111:.0f} km)")
            print(f"   Area: ~{width*111 * height*111:,.0f} km²")
        else:
            # Calculate from stations with adaptive buffer
            if not hasattr(self, 'stations_gdf'):
                raise ValueError("Please load gauge stations first or provide bbox")
            
            bounds = self.stations_gdf.total_bounds
            
            # Adaptive buffer based on study area size
            if buffer is None:
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                area_deg2 = width * height
                
                if area_deg2 < 1:  # Small area (< ~12,000 km²)
                    buffer = 0.5
                elif area_deg2 < 10:  # Medium area (< ~120,000 km²)
                    buffer = 1.0
                elif area_deg2 < 50:  # Large area (< ~600,000 km²)
                    buffer = 2.0
                else:  # Very large area
                    buffer = 3.0
                
                print(f"\n📊 Study area: {width:.2f}° x {height:.2f}° ({area_deg2:.2f}°²)")
                print(f"🎯 Using adaptive buffer: {buffer}° (~{buffer*111:.0f} km)")
            else:
                print(f"\n🎯 Using custom buffer: {buffer}° (~{buffer*111:.0f} km)")
            
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
        
        # Calculate chunks
        x_chunks = int(np.ceil(total_width / chunk_size_degrees))
        y_chunks = int(np.ceil(total_height / chunk_size_degrees))
        total_chunks = x_chunks * y_chunks
        
        print(f"\n📦 Downloading {total_chunks} chunks ({x_chunks}x{y_chunks}) at {chunk_size_degrees}° each...")
        
        if total_chunks > 100:
            print(f"\n⚠️  WARNING: {total_chunks} chunks is a LOT!")
            print(f"   This will take approximately {total_chunks * 3 / 60:.0f}-{total_chunks * 10 / 60:.0f} minutes")
            print(f"   Consider:")
            print(f"   - Using larger chunk_size_degrees (currently {chunk_size_degrees}°)")
            print(f"   - Running overnight or in a screen/tmux session")
            print(f"   - Ensuring stable internet connection\n")
        
        # Create temp directory for chunks
        chunk_dir = self.temp_dir / "srtm_chunks"
        chunk_dir.mkdir(exist_ok=True)
        
        chunk_files = []
        successful_downloads = 0
        failed_chunks = []
        
        start_time = time.time()
        
        for i in range(x_chunks):
            for j in range(y_chunks):
                # Calculate chunk bounds
                chunk_minx = minx + i * chunk_size_degrees
                chunk_maxx = min(minx + (i + 1) * chunk_size_degrees, maxx)
                chunk_miny = miny + j * chunk_size_degrees
                chunk_maxy = min(miny + (j + 1) * chunk_size_degrees, maxy)
                
                chunk_file = chunk_dir / f"chunk_{i}_{j}.tif"
                
                # Skip if already downloaded
                if chunk_file.exists() and chunk_file.stat().st_size > 1000:
                    chunk_files.append(str(chunk_file))
                    successful_downloads += 1
                    print(f"   ✅ Chunk {successful_downloads}/{total_chunks}: Using cached")
                    continue
                
                chunk_num = successful_downloads + len(failed_chunks) + 1
                print(f"   📥 Chunk {chunk_num}/{total_chunks}: "
                      f"({chunk_minx:.3f}, {chunk_miny:.3f}, {chunk_maxx:.3f}, {chunk_maxy:.3f})")
                
                # Try download with retries
                max_retries = 3
                success = False
                
                for attempt in range(max_retries):
                    try:
                        elevation.clip(
                            bounds=(chunk_minx, chunk_miny, chunk_maxx, chunk_maxy),
                            output=str(chunk_file),
                            product='SRTM1'
                        )
                        
                        # Verify file was created and has content
                        if chunk_file.exists():
                            file_size = chunk_file.stat().st_size
                            if file_size > 1000:  # At least 1KB
                                chunk_files.append(str(chunk_file))
                                successful_downloads += 1
                                print(f"      ✅ Success: {file_size:,} bytes")
                                success = True
                                break
                            else:
                                print(f"      ⚠️ File too small: {file_size} bytes (attempt {attempt+1}/{max_retries})")
                        else:
                            print(f"      ❌ File not created (attempt {attempt+1}/{max_retries})")
                        
                        # Wait before retry
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            print(f"      ⏳ Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                            
                    except Exception as e:
                        print(f"      ❌ Error (attempt {attempt+1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            print(f"      ⏳ Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                
                if not success:
                    failed_chunks.append((i, j))
                
                # Small delay between chunks to be nice to server
                if chunk_num < total_chunks:
                    time.sleep(1)
                
                # Periodic progress update
                if chunk_num % 10 == 0:
                    elapsed = time.time() - start_time
                    success_rate = successful_downloads / chunk_num * 100
                    avg_time = elapsed / chunk_num
                    remaining_chunks = total_chunks - chunk_num
                    eta_seconds = avg_time * remaining_chunks
                    
                    print(f"\n   📊 Progress: {chunk_num}/{total_chunks} chunks processed")
                    print(f"      Success rate: {success_rate:.1f}%")
                    print(f"      Elapsed time: {elapsed/60:.1f} min")
                    print(f"      ETA: {eta_seconds/60:.1f} min\n")
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"📊 Download summary:")
        print(f"   ✅ Successful: {successful_downloads}/{total_chunks} ({successful_downloads/total_chunks*100:.1f}%)")
        print(f"   ❌ Failed: {len(failed_chunks)}/{total_chunks}")
        print(f"   ⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"{'='*60}\n")
        
        if failed_chunks:
            print(f"⚠️  Failed chunks: {failed_chunks[:10]}{'...' if len(failed_chunks) > 10 else ''}")
        
        if successful_downloads == 0:
            print("❌ No chunks downloaded successfully!")
            return None
        
        if successful_downloads < total_chunks * 0.8:
            print(f"⚠️  WARNING: Only {successful_downloads/total_chunks*100:.1f}% of chunks downloaded")
            print(f"   Some areas may have missing data")
            print(f"   Consider re-running to fill gaps")
        
        # Mosaic all chunks
        print("\n🔗 Mosaicking chunks...")
        print(f"   Processing {len(chunk_files)} chunk files...")
        
        try:
            # For very large mosaics, process in batches to avoid memory issues
            batch_size = 50
            n_batches = int(np.ceil(len(chunk_files) / batch_size))
            
            if n_batches > 1:
                print(f"   Using batched approach: {n_batches} batches of {batch_size} chunks each")
                
                intermediate_mosaics = []
                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(chunk_files))
                    batch_chunks = chunk_files[start_idx:end_idx]
                    
                    print(f"   🔗 Batch {batch_idx + 1}/{n_batches}: Mosaicking {len(batch_chunks)} chunks...")
                    
                    src_files_to_mosaic = []
                    for chunk_file in batch_chunks:
                        src = rasterio.open(chunk_file)
                        src_files_to_mosaic.append(src)
                    
                    # Mosaic batch
                    mosaic, out_trans = merge(src_files_to_mosaic)
                    
                    # Get metadata from first file
                    out_meta = src_files_to_mosaic[0].meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": out_trans,
                        "compress": "lzw"
                    })
                    
                    # Write intermediate mosaic
                    intermediate_file = chunk_dir / f"intermediate_mosaic_{batch_idx}.tif"
                    with rasterio.open(intermediate_file, "w", **out_meta) as dest:
                        dest.write(mosaic)
                    
                    # Close all source files
                    for src in src_files_to_mosaic:
                        src.close()
                    
                    intermediate_mosaics.append(str(intermediate_file))
                    
                    # Clear memory
                    del mosaic, src_files_to_mosaic
                    gc.collect()
                    
                    print(f"      ✅ Batch {batch_idx + 1}/{n_batches} complete")
                
                # Now mosaic the intermediate mosaics
                print(f"   🔗 Merging {len(intermediate_mosaics)} intermediate mosaics into final DEM...")
                
                src_files_to_mosaic = []
                for mosaic_file in intermediate_mosaics:
                    src = rasterio.open(mosaic_file)
                    src_files_to_mosaic.append(src)
                
                mosaic, out_trans = merge(src_files_to_mosaic)
                
                out_meta = src_files_to_mosaic[0].meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "compress": "lzw"
                })
                
                # Write final mosaic
                with rasterio.open(dem_output, "w", **out_meta) as dest:
                    dest.write(mosaic)
                
                # Close all source files
                for src in src_files_to_mosaic:
                    src.close()
                
                # Clean up intermediate files
                for mosaic_file in intermediate_mosaics:
                    Path(mosaic_file).unlink()
                
            else:
                # Single batch - mosaic directly
                print(f"   🔗 Mosaicking {len(chunk_files)} chunks directly...")
                
                src_files_to_mosaic = []
                for chunk_file in chunk_files:
                    src = rasterio.open(chunk_file)
                    src_files_to_mosaic.append(src)
                
                mosaic, out_trans = merge(src_files_to_mosaic)
                
                out_meta = src_files_to_mosaic[0].meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "compress": "lzw"
                })
                
                with rasterio.open(dem_output, "w", **out_meta) as dest:
                    dest.write(mosaic)
                
                for src in src_files_to_mosaic:
                    src.close()
            
            self.dem_path = dem_output
            
            # Print mosaic info
            with rasterio.open(dem_output) as src:
                print(f"\n✅ Mosaic complete!")
                print(f"   📁 File: {dem_output}")
                print(f"   📊 Size: {src.width:,} x {src.height:,} pixels")
                print(f"   📏 Resolution: {src.res[0]:.6f}° x {src.res[1]:.6f}° (~{src.res[0]*111:.0f}m)")
                print(f"   💾 File size: {dem_output.stat().st_size / 1024 / 1024:.1f} MB")
            
            return dem_output
            
        except Exception as e:
            print(f"❌ Mosaicking failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def preprocess_dem(self, fill_depressions=True):
        """
        Preprocess DEM - Fill depressions and clean metadata for WhiteboxTools compatibility.
        """
        # Ensure we have a DEM path
        if not hasattr(self, 'dem_path'):
            candidate = self.processed_dir / "srtm_mosaic_chunked.tif"
            if candidate.exists():
                self.dem_path = candidate
                print(f"✅ Using existing DEM mosaic: {candidate}")
            else:
                raise ValueError("Please download DEM first (srtm_mosaic_chunked.tif not found).")

        dem_path = Path(self.dem_path)
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM not found: {dem_path}")

        print("🔧 Preprocessing DEM...")

        if not fill_depressions:
            self.processed_dem_path = dem_path
            print(f"✅ Preprocessed DEM: {self.processed_dem_path}")
            return self.processed_dem_path

        # Output paths
        filled_dem_raw = self.processed_dir / "dem_filled_raw.tif"  # WhiteboxTools output
        filled_dem = self.processed_dir / "dem_filled.tif"  # Cleaned output
        
        # If valid cleaned filled DEM already exists, use it
        if filled_dem.exists() and filled_dem.stat().st_size > 0:
            try:
                with rasterio.open(filled_dem) as src:
                    # Check a sample to see if it's valid
                    sample = src.read(1, window=rasterio.windows.Window(0, 0, min(1000, src.width), min(1000, src.height)))
                    if not (np.all(sample == 0) or np.all(np.isnan(sample))):
                        print(f"✅ Using existing filled DEM: {filled_dem}")
                        self.processed_dem_path = filled_dem
                        return self.processed_dem_path
            except Exception as e:
                print(f"⚠️  Existing filled DEM has issues: {e} - regenerating...")

        # Step 1: Fill depressions with WhiteboxTools
        print("🕳️  Filling depressions (Planchon & Darboux)...")
        print("   This may take several minutes for large DEMs...")
        
        # Delete old files if they exist
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
            raise RuntimeError(f"WhiteboxTools failed to create filled DEM: {filled_dem_raw}")

        # Step 2: Clean the GeoTIFF metadata to fix WhiteboxTools compatibility issues
        print("🧹 Cleaning GeoTIFF metadata for WhiteboxTools compatibility...")
        
        try:
            with rasterio.open(filled_dem_raw) as src:
                # Read data in chunks to handle large files
                profile = src.profile.copy()
                
                # Simplify the profile - remove problematic metadata
                profile.update(
                    driver='GTiff',
                    dtype=src.dtypes[0],
                    compress='lzw',
                    tiled=True,
                    blockxsize=512,
                    blockysize=512,
                    # Force simple CRS representation
                    crs='EPSG:4326'
                )
                
                # Remove any extra tags that might cause issues
                if 'photometric' in profile:
                    del profile['photometric']
                
                # Write cleaned file in blocks to handle large files
                with rasterio.open(filled_dem, 'w', **profile) as dst:
                    # Process in windows/blocks for memory efficiency
                    for ji, window in src.block_windows(1):
                        data = src.read(1, window=window)
                        dst.write(data, 1, window=window)
            
            print(f"✅ Cleaned filled DEM saved: {filled_dem}")
            
            # Validate the result
            with rasterio.open(filled_dem) as src:
                print(f"   Filled DEM size: {src.width} x {src.height}")
                print(f"   Filled DEM CRS: {src.crs}")
            
            # Delete the raw file to save space
            if filled_dem_raw.exists():
                filled_dem_raw.unlink()
                print(f"   Deleted intermediate file: {filled_dem_raw}")
            
        except Exception as e:
            print(f"⚠️  Metadata cleaning failed: {e}")
            print(f"   Trying alternative approach with GDAL...")
            
            # Alternative: Use GDAL translate via command line
            import subprocess
            try:
                result = subprocess.run([
                    'gdal_translate',
                    '-of', 'GTiff',
                    '-co', 'COMPRESS=LZW',
                    '-co', 'TILED=YES',
                    '-a_srs', 'EPSG:4326',
                    str(filled_dem_raw),
                    str(filled_dem)
                ], capture_output=True, text=True, check=True)
                print(f"✅ GDAL translate successful")
                
                # Delete the raw file
                if filled_dem_raw.exists():
                    filled_dem_raw.unlink()
                    
            except subprocess.CalledProcessError as e:
                print(f"❌ GDAL translate failed: {e.stderr}")
                # Fall back to using the raw file directly
                if filled_dem_raw.exists():
                    import shutil
                    shutil.move(str(filled_dem_raw), str(filled_dem))
                    print(f"⚠️  Using raw filled DEM without cleaning")

        self.processed_dem_path = filled_dem
        print(f"✅ Preprocessed DEM: {self.processed_dem_path}")
        return self.processed_dem_path
    
    def calculate_flow_accumulation(self):
        """
        Calculate D8 flow direction and accumulation.
        ✅ KEY FIX: Use flow direction with pntr=True, exactly like the working script!
        """
        if not hasattr(self, 'processed_dem_path') or not Path(self.processed_dem_path).exists():
            self.preprocess_dem()

        print("💧 Calculating flow direction and accumulation...")

        flow_dir_path = self.processed_dir / "flow_direction.tif"
        flow_acc_path = self.processed_dir / "flow_accumulation.tif"

        # If both already exist and look valid, reuse them
        if (flow_dir_path.exists() and flow_dir_path.stat().st_size > 0 and
            flow_acc_path.exists() and flow_acc_path.stat().st_size > 0):
            
            # Validate flow accumulation
            with rasterio.open(flow_acc_path) as src:
                data = src.read(1)
                max_acc = np.nanmax(data)
                print(f"   Existing flow accumulation max value: {max_acc:,.0f}")
                
                # For a 200,000 km² basin at 30m resolution, expect millions of cells
                # At minimum, should be > 100,000 for any reasonable river
                if max_acc < 100000:
                    print(f"   ⚠️ Flow accumulation values too low - regenerating...")
                else:
                    print(f"✅ Using existing flow direction: {flow_dir_path}")
                    print(f"✅ Using existing flow accumulation: {flow_acc_path}")
                    self.flow_dir_path = flow_dir_path
                    self.flow_acc_path = flow_acc_path
                    return self.flow_dir_path, self.flow_acc_path
        
        # Delete existing files to regenerate
        if flow_dir_path.exists():
            flow_dir_path.unlink()
        if flow_acc_path.exists():
            flow_acc_path.unlink()

        # Compute flow direction
        print("🧭 Calculating flow direction (WhiteboxTools d8_pointer)...")
        self.wbt.set_verbose_mode(True)
        
        self.wbt.d8_pointer(
            dem=str(self.processed_dem_path),
            output=str(flow_dir_path)
        )
        
        self.wbt.set_verbose_mode(False)

        if not flow_dir_path.exists() or flow_dir_path.stat().st_size == 0:
            raise RuntimeError(
                f"WhiteboxTools failed to create flow direction: {flow_dir_path}\n"
                f"- Input DEM: {self.processed_dem_path}"
            )

        # ✅ KEY FIX: Use flow direction pointer with pntr=True
        # This is EXACTLY what the working script does!
        print("💧 Calculating flow accumulation (WhiteboxTools d8_flow_accumulation)...")
        self.wbt.set_verbose_mode(True)
        
        self.wbt.d8_flow_accumulation(
            i=str(flow_dir_path),  # ✅ Use flow DIRECTION, not DEM
            output=str(flow_acc_path),
            pntr=True  # ✅ This tells WBT the input is a D8 pointer raster
        )
        
        self.wbt.set_verbose_mode(False)

        if not flow_acc_path.exists() or flow_acc_path.stat().st_size == 0:
            raise RuntimeError(
                f"WhiteboxTools failed to create flow accumulation: {flow_acc_path}\n"
                f"- Pointer raster: {flow_dir_path}"
            )

        # Validate the output
        with rasterio.open(flow_acc_path) as src:
            data = src.read(1)
            max_acc = np.nanmax(data)
            print(f"   Flow accumulation max value: {max_acc:,.0f} cells")
            
            if max_acc < 100000:
                print(f"   ⚠️ WARNING: Max flow accumulation seems low!")
                print(f"      Expected: millions of cells for a large basin")
                print(f"      Possible causes:")
                print(f"      - DEM still has unfilled depressions")
                print(f"      - DEM has NoData gaps breaking connectivity")

        print(f"✅ Flow direction created: {flow_dir_path}")
        print(f"✅ Flow accumulation created: {flow_acc_path}")

        # Set attributes always
        self.flow_dir_path = flow_dir_path
        self.flow_acc_path = flow_acc_path
        return self.flow_dir_path, self.flow_acc_path

    def snap_outlets_to_streams(self, threshold=1000, search_radius=0.002):
        """Snap gauge stations to streams"""
        if not hasattr(self, 'flow_acc_path'):
            self.calculate_flow_accumulation()
        
        print(f"🎯 Snapping outlets to streams...")
        
        # Extract streams
        streams_path = str(self.processed_dir / "streams.tif")
        
        if Path(streams_path).exists():
            print("✅ Using existing streams")
        else:
            print(f"🌊 Extracting streams (threshold: {threshold} cells)...")
            self.wbt.extract_streams(
                flow_accum=str(self.flow_acc_path),
                output=streams_path,
                threshold=threshold
            )
        
        # Convert stations to shapefile
        stations_shp = self.temp_dir / "stations.shp"
        self.stations_gdf.to_file(stations_shp)
        
        # Snap pour points
        snapped_outlets = str(self.processed_dir / "snapped_outlets.shp")
        
        if Path(snapped_outlets).exists():
            print("✅ Using existing snapped outlets")
        else:
            print(f"🎯 Snapping pour points (search radius: {search_radius}°)...")
            result = self.wbt.snap_pour_points(
                pour_pts=str(stations_shp),
                flow_accum=str(self.flow_acc_path),
                output=snapped_outlets,
                snap_dist=search_radius
            )
            
            # If failed, try larger radius
            if not Path(snapped_outlets).exists():
                search_radius_large = search_radius * 10
                print(f"⚠️  Retrying with larger radius ({search_radius_large}°)...")
                result = self.wbt.snap_pour_points(
                    pour_pts=str(stations_shp),
                    flow_accum=str(self.flow_acc_path),
                    output=snapped_outlets,
                    snap_dist=search_radius_large
                )
                
                # If still failed, use original stations
                if not Path(snapped_outlets).exists():
                    print("⚠️  Using original stations as outlets")
                    self.stations_gdf.to_file(snapped_outlets)
        
        # Load snapped outlets
        self.snapped_outlets = gpd.read_file(snapped_outlets)
        self.streams_path = Path(streams_path)
        
        print(f"✅ Snapped {len(self.snapped_outlets)} outlets")
        return self.snapped_outlets

    def apply_constraint(self, catchment_gdf, station_name):
        """
        Apply constraint by subtracting the constraint area from catchment
        
        Parameters:
        -----------
        catchment_gdf : GeoDataFrame
            The delineated catchment
        station_name : str
            Station name for logging
            
        Returns:
        --------
        GeoDataFrame
            Catchment with constraint area subtracted (if overlapping)
        """
        if self.constraint_gdf is None:
            return catchment_gdf
        
        try:
            # Get the catchment geometry
            catchment_geom = catchment_gdf.geometry.iloc[0]
            
            # Get the constraint geometry (union of all constraint polygons)
            constraint_geom = self.constraint_gdf.geometry.unary_union
            
            # Check if they intersect
            if not catchment_geom.intersects(constraint_geom):
                return catchment_gdf
            
            # Calculate intersection area
            intersection = catchment_geom.intersection(constraint_geom)
            intersection_area = intersection.area * (111.32 * np.cos(np.radians(intersection.centroid.y))) * 111.32
            
            print(f"    🔧 Constraint overlap detected: {intersection_area:,.2f} km²")
            
            # Subtract the constraint area
            corrected_geom = catchment_geom.difference(constraint_geom)
            
            # Handle potential MultiPolygon result
            if corrected_geom.is_empty:
                print(f"    ⚠️ Catchment entirely within constraint - keeping original")
                return catchment_gdf
            
            # If result is MultiPolygon, take the largest part
            if corrected_geom.geom_type == 'MultiPolygon':
                # Find the largest polygon
                largest_geom = max(corrected_geom.geoms, key=lambda g: g.area)
                corrected_geom = largest_geom
                print(f"    📐 MultiPolygon result - using largest part")
            
            # Create corrected GeoDataFrame
            corrected_gdf = catchment_gdf.copy()
            corrected_gdf.geometry = [corrected_geom]
            
            # Recalculate area
            new_area = corrected_geom.area * (111.32 * np.cos(np.radians(corrected_geom.centroid.y))) * 111.32
            old_area = catchment_gdf['area_km2'].iloc[0]
            corrected_gdf['area_km2'] = new_area
            
            print(f"    ✂️ Area adjusted: {old_area:,.2f} → {new_area:,.2f} km² (removed {old_area - new_area:,.2f} km²)")
            
            return corrected_gdf
            
        except Exception as e:
            print(f"    ⚠️ Constraint application failed: {e}")
            return catchment_gdf
    
    def delineate_catchments(self):
        """Delineate catchments for all snapped outlets using formatted station IDs"""
        if not hasattr(self, 'snapped_outlets'):
            self.snap_outlets_to_streams()
        
        print(f"\n🏔️  Delineating {len(self.snapped_outlets)} catchments...")
        
        if self.constraint_gdf is not None:
            print(f"🔧 Constraint shapefile will be applied to overlapping catchments")
        
        catchments = []
        
        for idx, outlet in self.snapped_outlets.iterrows():
            # Get station ID and format it
            station_id_raw = outlet.get('station_id', idx)
            
            # Convert to string and handle decimal points
            if isinstance(station_id_raw, (int, float)):
                if station_id_raw == int(station_id_raw):
                    station_id_str = str(int(station_id_raw))
                else:
                    station_id_str = str(station_id_raw).replace('.', '')
            else:
                station_id_str = str(station_id_raw).replace('.', '')
            
            # Format with leading zeros to 4 digits
            station_name = f"shape_{station_id_str.zfill(4)}"
            print(f"  Processing {station_name} (ID: {station_id_raw})...")
            
            # Check if catchment already exists
            watershed_vector = str(self.catchments_dir / f"catchment_{station_name}.shp")
            if Path(watershed_vector).exists():
                print(f"    ✅ Using existing catchment")
                catchment_gdf = gpd.read_file(watershed_vector)
                
                # ✅ Still apply constraint to existing catchments if needed
                if self.constraint_gdf is not None:
                    catchment_gdf = self.apply_constraint(catchment_gdf, station_name)
                    # Save the corrected version
                    catchment_gdf.to_file(watershed_vector)
                
                catchments.append(catchment_gdf)
                continue
            
            try:
                # Create individual outlet file
                outlet_file = self.temp_dir / f"outlet_{station_name}.shp"
                gpd.GeoDataFrame([outlet]).to_file(outlet_file)
                
                # Delineate watershed to TEMPORARY file
                watershed_file = self.temp_dir / f"watershed_{station_name}.tif"
                self.wbt.watershed(
                    d8_pntr=str(self.flow_dir_path),
                    pour_pts=str(outlet_file),
                    output=str(watershed_file)
                )
                
                # Check if watershed raster was created
                if not Path(watershed_file).exists():
                    print(f"    ❌ Failed to create watershed raster")
                    continue
                
                # Convert to vector
                self.wbt.raster_to_vector_polygons(
                    i=str(watershed_file),
                    output=watershed_vector
                )
                
                # DELETE the temporary raster
                if Path(watershed_file).exists():
                    Path(watershed_file).unlink()
                
                # Delete the temporary outlet shapefile
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    outlet_related = outlet_file.with_suffix(ext)
                    if outlet_related.exists():
                        outlet_related.unlink()
                
                # Check if vector was created
                if not Path(watershed_vector).exists():
                    print(f"    ❌ Failed to create watershed vector")
                    continue
                
                # Load and add metadata
                catchment_gdf = gpd.read_file(watershed_vector)
                if len(catchment_gdf) == 0:
                    print(f"    ❌ Empty catchment")
                    continue
                
                # Add station name
                catchment_gdf['station_name'] = station_name
                
                # Calculate area (same method as working script)
                catchment_gdf['area_km2'] = catchment_gdf.to_crs('EPSG:4326').geometry.apply(
                    lambda geom: geom.area * (111.32 * np.cos(np.radians(geom.centroid.y))) * 111.32
                )
                
                # Copy ALL attributes from the original station
                for col in outlet.index:
                    if col != 'geometry':
                        catchment_gdf[col] = outlet[col]
                
                # ✅ Apply constraint - subtract the constraint area if overlapping
                catchment_gdf = self.apply_constraint(catchment_gdf, station_name)
                
                # Save the corrected catchment
                catchment_gdf.to_file(watershed_vector)
                
                catchments.append(catchment_gdf)
                print(f"    ✅ Success: {catchment_gdf['area_km2'].iloc[0]:,.2f} km²")
                
            except Exception as e:
                print(f"    ❌ Failed for {station_name}: {e}")
        
        if catchments:
            # Combine all catchments
            self.catchments_gdf = gpd.GeoDataFrame(
                pd.concat(catchments, ignore_index=True),
                crs='EPSG:4326'
            )
            
            # Save combined catchments
            all_catchments_file = self.catchments_dir / "all_catchments.shp"
            self.catchments_gdf.to_file(all_catchments_file)
            
            print(f"\n✅ Delineated {len(catchments)} catchments")
            print(f"💾 Saved to: {all_catchments_file}")
            
            # Statistics
            print(f"\n📊 Statistics:")
            print(f"   Total area: {self.catchments_gdf['area_km2'].sum():,.0f} km²")
            print(f"   Largest: {self.catchments_gdf['area_km2'].max():,.0f} km²")
            print(f"   Smallest: {self.catchments_gdf['area_km2'].min():,.0f} km²")
            
            return self.catchments_gdf
        else:
            print("❌ No catchments were successfully delineated")
            return None



def main():
    """
    Main function with examples for different regions
    """
    
    base_dir = "/home/jberg@giub.local/Catchment_delineation"
    
    stations_file_pakistan = "/home/jberg@giub.local/Catchment_delineation/gauging_stations/Pakistan_stations.shp"
    
    # ✅ Constraint shapefile to subtract from catchments
    constraint_shapefile = "/home/jberg@giub.local/Catchment_delineation/gauging_stations/constraint_catchment_Indus.shp"
    
    # Upper Indus Basin bounding box (extended east)
    upper_indus_bbox = (
        71.0,  # West
        30.5,  # South
        82.0,  # East (extended from 81.0)
        37.5   # North
    )

    
    try:
        print("\n" + "🏔️ "*40)
        print("UPPER INDUS BASIN DELINEATION")
        print("🏔️ "*40)
        
        # ✅ Initialize with custom bbox AND constraint shapefile
        delineator_indus = CatchmentDelineator(
            base_dir=base_dir, 
            country="Pakistan",
            bbox=upper_indus_bbox,
            constraint_shapefile=constraint_shapefile  # ✅ Add constraint
        )
        
        # Load gauge stations
        if Path(stations_file_pakistan).exists():
            stations_gdf = delineator_indus.load_gauge_stations(stations_file=stations_file_pakistan)
        
        # Download DEM
        dem_path = delineator_indus.download_srtm_in_chunks(chunk_size_degrees=1.5)
        
        if dem_path is None:
            print("❌ DEM download failed")
            return
        
        # Process DEM
        processed_dem = delineator_indus.preprocess_dem()
        
        # Calculate flow
        flow_dir, flow_acc = delineator_indus.calculate_flow_accumulation()
        
        # Delineate catchments (constraint will be applied automatically)
        if hasattr(delineator_indus, 'stations_gdf'):
            snapped_outlets = delineator_indus.snap_outlets_to_streams()
            catchments = delineator_indus.delineate_catchments()
            
            if catchments is not None:
                print(f"\n🎉 SUCCESS! Delineated {len(catchments)} Upper Indus catchments")
                print(f"📊 Statistics:")
                print(f"   Total area: {catchments['area_km2'].sum():,.0f} km²")
                print(f"   Largest catchment: {catchments['area_km2'].max():,.0f} km²")
                print(f"   Smallest catchment: {catchments['area_km2'].min():,.0f} km²")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()