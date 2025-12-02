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

# WhiteboxTools
import whitebox

print("✅ All libraries imported successfully!")

class CatchmentDelineator:
    """
    A comprehensive catchment delineation system using WhiteboxTools
    """
    
    def __init__(self, base_dir, country="Nepal"):
        self.base_dir = Path(base_dir)
        self.country = country
        
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
        
    # Keep your existing download_srtm_in_chunks method - it works great!
    def download_srtm_in_chunks(self, chunk_size_degrees=1.0):
        """Download SRTM data in chunks and mosaic them"""
        if not hasattr(self, 'stations_gdf'):
            raise ValueError("Please load gauge stations first")
        
        # Check if mosaic already exists
        dem_output = self.processed_dir / "srtm_mosaic_chunked.tif"
        if dem_output.exists():
            print(f"✅ Using existing DEM mosaic: {dem_output}")
            self.dem_path = dem_output
            return dem_output
        
        # Get bounding box with buffer for complete catchments
        bounds = self.stations_gdf.total_bounds
        buffer = 1.0  # Buffer for complete catchments
        minx, miny, maxx, maxy = bounds
        minx -= buffer
        miny -= buffer
        maxx += buffer
        maxy += buffer
        
        print(f"📦 Downloading SRTM data for area: {minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}")
        
        # Calculate chunks
        x_chunks = int(np.ceil((maxx - minx) / chunk_size_degrees))
        y_chunks = int(np.ceil((maxy - miny) / chunk_size_degrees))
        total_chunks = x_chunks * y_chunks
        
        print(f"📦 Downloading {total_chunks} chunks ({x_chunks}x{y_chunks}) at {chunk_size_degrees}° each...")
        
        # Create temp directory for chunks
        chunk_dir = self.temp_dir / "srtm_chunks"
        chunk_dir.mkdir(exist_ok=True)
        
        chunk_files = []
        successful_downloads = 0
        
        for i in range(x_chunks):
            for j in range(y_chunks):
                # Calculate chunk bounds
                chunk_minx = minx + i * chunk_size_degrees
                chunk_maxx = min(minx + (i + 1) * chunk_size_degrees, maxx)
                chunk_miny = miny + j * chunk_size_degrees
                chunk_maxy = min(miny + (j + 1) * chunk_size_degrees, maxy)
                
                chunk_file = chunk_dir / f"chunk_{i}_{j}.tif"
                
                print(f"   Downloading chunk {successful_downloads + 1}/{total_chunks}: "
                    f"({chunk_minx:.3f}, {chunk_miny:.3f}, {chunk_maxx:.3f}, {chunk_maxy:.3f})")
                
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
                            print(f"   ✅ Success: {file_size} bytes")
                        else:
                            print(f"   ⚠️ File too small: {file_size} bytes")
                    else:
                        print(f"   ❌ File not created")
                        
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    continue
        
        print(f"📊 Download summary: {successful_downloads}/{total_chunks} chunks successful")
        
        if successful_downloads == 0:
            print("❌ No chunks downloaded successfully!")
            return None
        
        # Mosaic all chunks
        print("🔗 Mosaicking chunks...")
        try:
            # Read all chunk files
            src_files_to_mosaic = []
            for chunk_file in chunk_files:
                src = rasterio.open(chunk_file)
                src_files_to_mosaic.append(src)
            
            # Mosaic
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
            
            # Write mosaic
            with rasterio.open(dem_output, "w", **out_meta) as dest:
                dest.write(mosaic)
            
            # Close all source files
            for src in src_files_to_mosaic:
                src.close()
            
            # Clean up chunk files
            for chunk_file in chunk_files:
                Path(chunk_file).unlink()
            chunk_dir.rmdir()
            
            self.dem_path = dem_output
            print(f"✅ Mosaic saved: {dem_output}")
            
            return dem_output
            
        except Exception as e:
            print(f"❌ Mosaicking failed: {e}")
            return None
        
    def preprocess_dem(self, fill_depressions=True):
        """Preprocess DEM using Planchon & Darboux method"""
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

        # Fill depressions using Planchon & Darboux
        filled_dem = self.processed_dir / "dem_filled.tif"
        
        # If valid filled DEM already exists, reuse
        if filled_dem.exists() and filled_dem.stat().st_size > 0:
            print(f"✅ Using existing filled DEM: {filled_dem}")
            self.processed_dem_path = filled_dem
            return self.processed_dem_path

        print("🕳️ Filling depressions (Planchon & Darboux)...")
        self.wbt.fill_depressions_planchon_and_darboux(
            dem=str(dem_path),
            output=str(filled_dem)
        )

        if not filled_dem.exists() or filled_dem.stat().st_size == 0:
            raise RuntimeError(f"WhiteboxTools failed to create filled DEM: {filled_dem}")

        self.processed_dem_path = filled_dem
        print(f"✅ Preprocessed DEM: {self.processed_dem_path}")
        return self.processed_dem_path
    
    
    def calculate_flow_accumulation(self):
        """Calculate D8 flow direction and accumulation; verify outputs exist."""
        if not hasattr(self, 'processed_dem_path') or not Path(self.processed_dem_path).exists():
            self.preprocess_dem()

        print("💧 Calculating flow direction and accumulation...")

        flow_dir_path = self.processed_dir / "flow_direction.tif"
        flow_acc_path = self.processed_dir / "flow_accumulation.tif"

        # If both already exist and look valid, reuse them
        if (flow_dir_path.exists() and flow_dir_path.stat().st_size > 0 and
            flow_acc_path.exists() and flow_acc_path.stat().st_size > 0):
            print(f"✅ Using existing flow direction: {flow_dir_path}")
            print(f"✅ Using existing flow accumulation: {flow_acc_path}")
        else:
            # Compute flow direction
            print("🧭 Calculating flow direction (WhiteboxTools d8_pointer)...")
            try:
                self.wbt.set_verbose_mode(True)
            except Exception:
                pass
            try:
                self.wbt.d8_pointer(
                    dem=str(self.processed_dem_path),
                    output=str(flow_dir_path)
                )
            finally:
                try:
                    self.wbt.set_verbose_mode(False)
                except Exception:
                    pass

            if not flow_dir_path.exists() or flow_dir_path.stat().st_size == 0:
                raise RuntimeError(
                    f"WhiteboxTools failed to create flow direction: {flow_dir_path}\n"
                    f"- Input DEM: {self.processed_dem_path}"
                )

            # Compute flow accumulation (using pointer as input with pntr=True)
            print("💧 Calculating flow accumulation (WhiteboxTools d8_flow_accumulation)...")
            try:
                self.wbt.set_verbose_mode(True)
            except Exception:
                pass
            try:
                self.wbt.d8_flow_accumulation(
                    i=str(flow_dir_path),
                    output=str(flow_acc_path),
                    pntr=True
                )
            finally:
                try:
                    self.wbt.set_verbose_mode(False)
                except Exception:
                    pass

            if not flow_acc_path.exists() or flow_acc_path.stat().st_size == 0:
                raise RuntimeError(
                    f"WhiteboxTools failed to create flow accumulation: {flow_acc_path}\n"
                    f"- Pointer raster: {flow_dir_path}"
                )

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
                print("🌊 Extracting streams...")
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
                print(f"🎯 Snapping pour points...")
                result = self.wbt.snap_pour_points(
                    pour_pts=str(stations_shp),
                    flow_accum=str(self.flow_acc_path),
                    output=snapped_outlets,
                    snap_dist=search_radius
                )
                
                # If failed, try larger radius
                if not Path(snapped_outlets).exists():
                    print(f"⚠️ Retrying with larger radius...")
                    result = self.wbt.snap_pour_points(
                        pour_pts=str(stations_shp),
                        flow_accum=str(self.flow_acc_path),
                        output=snapped_outlets,
                        snap_dist=search_radius * 10
                    )
                    
                    # If still failed, use original stations
                    if not Path(snapped_outlets).exists():
                        print("⚠️ Using original stations as outlets")
                        self.stations_gdf.to_file(snapped_outlets)
            
            # Load snapped outlets
            self.snapped_outlets = gpd.read_file(snapped_outlets)
            self.streams_path = Path(streams_path)
            
            print(f"✅ Snapped {len(self.snapped_outlets)} outlets")
            return self.snapped_outlets
    
    def delineate_catchments(self):
        """Delineate catchments for all snapped outlets using formatted station IDs"""
        if not hasattr(self, 'snapped_outlets'):
            self.snap_outlets_to_streams()
        
        print(f"🏔️ Delineating {len(self.snapped_outlets)} catchments...")
        
        catchments = []
        
        for idx, outlet in self.snapped_outlets.iterrows():
            # Get station ID and format it
            station_id_raw = outlet.get('station_id', idx)
            
            # Convert to string and handle decimal points
            if isinstance(station_id_raw, (int, float)):
                if station_id_raw == int(station_id_raw):
                    # It's a whole number (e.g., 667.0 -> 667)
                    station_id_str = str(int(station_id_raw))
                else:
                    # It has decimals (e.g., 668.3 -> 6683)
                    station_id_str = str(station_id_raw).replace('.', '')
            else:
                # It's already a string
                station_id_str = str(station_id_raw).replace('.', '')
            
            # Format with leading zeros to 4 digits
            station_name = f"shape_{station_id_str.zfill(4)}"
            print(f"  Processing {station_name} (ID: {station_id_raw})...")
            
            # Check if catchment already exists
            watershed_vector = str(self.catchments_dir / f"catchment_{station_name}.shp")
            if Path(watershed_vector).exists():
                print(f"    ✅ Using existing catchment")
                catchment_gdf = gpd.read_file(watershed_vector)
                catchments.append(catchment_gdf)
                continue
            
            try:
                # Create individual outlet file
                outlet_file = self.temp_dir / f"outlet_{station_name}.shp"
                gpd.GeoDataFrame([outlet]).to_file(outlet_file)
                
                # Delineate watershed
                watershed_file = str(self.catchments_dir / f"catchment_{station_name}.tif")
                self.wbt.watershed(
                    d8_pntr=str(self.flow_dir_path),
                    pour_pts=str(outlet_file),
                    output=watershed_file
                )
                
                # Check if watershed raster was created
                if not Path(watershed_file).exists():
                    print(f"    ❌ Failed to create watershed raster")
                    continue
                
                # Convert to vector
                self.wbt.raster_to_vector_polygons(
                    i=watershed_file,
                    output=watershed_vector
                )
                
                # Check if vector was created
                if not Path(watershed_vector).exists():
                    print(f"    ❌ Failed to create watershed vector")
                    continue
                
                # Load and add metadata
                catchment_gdf = gpd.read_file(watershed_vector)
                if len(catchment_gdf) == 0:
                    print(f"    ❌ Empty catchment")
                    continue
                
                # Add station name and calculated area first
                catchment_gdf['station_name'] = station_name
                catchment_gdf['area_km2'] = catchment_gdf.to_crs('EPSG:4326').geometry.apply(
                    lambda geom: geom.area * (111.32 * np.cos(np.radians(geom.centroid.y))) * 111.32
                )
                
                # **KEY FIX**: Copy ALL attributes from the original station
                # Get all non-geometry columns from the outlet (which came from stations)
                for col in outlet.index:
                    if col != 'geometry':  # Skip geometry column
                        catchment_gdf[col] = outlet[col]
                
                # Print what attributes we're copying
                copied_attrs = [col for col in outlet.index if col != 'geometry']
                print(f"    📋 Copied attributes: {copied_attrs}")
                
                catchments.append(catchment_gdf)
                print(f"    ✅ Success: {catchment_gdf['area_km2'].iloc[0]:.2f} km²")
                
            except Exception as e:
                print(f"    ❌ Failed for {station_name}: {e}")
        
        if catchments:
            # Combine all catchments
            self.catchments_gdf = gpd.pd.concat(catchments, ignore_index=True)
            
            # Save combined catchments
            all_catchments_file = self.catchments_dir / "all_catchments.shp"
            self.catchments_gdf.to_file(all_catchments_file)
            
            print(f"✅ Delineated {len(catchments)} catchments")
            print(f"💾 Saved to: {all_catchments_file}")
            print(f"📊 Final catchment columns: {list(self.catchments_gdf.columns)}")
            
            return self.catchments_gdf
        else:
            print("❌ No catchments were successfully delineated")
            return None
        
def main():
    # Your existing main function
    stations_file = "/home/jberg/OneDrive/Raven_worldwide/01_data/topo/gauging_stations/Nepal_stations_mideast.shp"
    base_dir = "/home/jberg/OneDrive/Raven_worldwide/01_data"
    
    try:
        delineator = CatchmentDelineator(base_dir=base_dir, country="Nepal")
        
        stations_gdf = delineator.load_gauge_stations(stations_file=stations_file)
        dem_path = delineator.download_srtm_in_chunks()
        processed_dem = delineator.preprocess_dem()
        flow_dir, flow_acc = delineator.calculate_flow_accumulation()
        snapped_outlets = delineator.snap_outlets_to_streams()
        catchments = delineator.delineate_catchments()
        
        if catchments is not None:
            print(f"\n🎉 SUCCESS! Delineated {len(catchments)} catchments")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()