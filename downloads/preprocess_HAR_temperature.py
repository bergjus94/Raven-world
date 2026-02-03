import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import os
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def find_har_base_path():
    """Find the mounted HAR data directory"""
    possible_base_paths = [
        "/mnt/hydroshare/data/Meteorology/HMA",
        "/media/hydroshare/data/Meteorology/HMA",
        "~/hydroshare/data/Meteorology/HMA",
    ]
    
    # Check for GVFS mounts
    try:
        gvfs_path = Path(f"/run/user/{os.getuid()}/gvfs")
        if gvfs_path.exists():
            for mount in gvfs_path.iterdir():
                if 'hydroshare' in mount.name.lower():
                    possible_path = mount / "Meteorology/HMA"
                    if possible_path.exists():
                        possible_base_paths.insert(0, str(possible_path))
    except Exception:
        pass
    
    # Check which path exists
    for path in possible_base_paths:
        expanded_path = Path(os.path.expanduser(path))
        if expanded_path.exists():
            print(f"✅ Found HAR data directory: {expanded_path}")
            return expanded_path
    
    raise FileNotFoundError(
        "HAR data directory not found! Please mount the network share:\n"
        "   smb://hydroshare.giub.unibe.ch/data"
    )


def process_hourly_to_daily(input_file, output_dir, year):
    """
    Process hourly temperature data to daily aggregates (memory-efficient)
    
    Parameters:
    -----------
    input_file : Path
        Path to hourly NetCDF file (HARv2_d10km_h_2d_t2_YYYY.nc)
    output_dir : Path
        Directory to save daily aggregated files
    year : int
        Year being processed
    
    Returns:
    --------
    dict with paths to created files
    """
    print(f"\n{'='*80}")
    print(f"📅 Processing year {year}")
    print(f"{'='*80}")
    print(f"📁 Input file: {input_file.name}")
    
    # Define output files
    output_files = {
        'mean': output_dir / f"HARv2_d10km_d_2d_t2_mean_{year}.nc",
        'min': output_dir / f"HARv2_d10km_d_2d_t2_min_{year}.nc",
        'max': output_dir / f"HARv2_d10km_d_2d_t2_max_{year}.nc"
    }
    
    # Check if all outputs already exist
    all_exist = all(f.exists() for f in output_files.values())
    if all_exist:
        print(f"✅ All daily files already exist for {year}, skipping...")
        return output_files
    
    # Load hourly data with chunking (MEMORY EFFICIENT!)
    print("📖 Loading hourly temperature data (chunked)...")
    
    # Open with dask for lazy loading
    ds = xr.open_dataset(
        input_file,
        chunks={'time': 24, 'south_north': 50, 'west_east': 50}  # Process in small chunks
    )
    
    # Check the structure
    print(f"   Time steps: {len(ds.time)}")
    print(f"   Spatial grid: {len(ds.south_north)} x {len(ds.west_east)}")
    print(f"   Time range: {pd.Timestamp(ds.time.values[0])} to {pd.Timestamp(ds.time.values[-1])}")
    
    # Get the temperature variable
    temp_var = None
    for var in ['t2', 'T2', 'temp', 'temperature']:
        if var in ds.data_vars:
            temp_var = var
            break
    
    if temp_var is None:
        raise ValueError(f"Temperature variable not found in {input_file}")
    
    print(f"   Temperature variable: '{temp_var}'")
    print(f"   Using chunked processing to avoid memory issues")
    
    # Resample to daily
    print("📊 Aggregating to daily values...")
    
    # Mean temperature
    if not output_files['mean'].exists():
        print("   ├─ Calculating daily mean...")
        
        # Resample and compute (this triggers the actual computation)
        daily_mean = ds[temp_var].resample(time='1D').mean()
        
        # Create minimal dataset (only coordinates we need)
        ds_mean = xr.Dataset(
            {
                't2_mean': daily_mean
            },
            coords={
                'time': daily_mean.time,
                'south_north': ds.south_north,
                'west_east': ds.west_east,
                'lat': ds.lat,
                'lon': ds.lon
            }
        )
        
        # Add metadata
        ds_mean['t2_mean'].attrs = {
            'long_name': 'Daily mean 2-meter air temperature',
            'units': ds[temp_var].attrs.get('units', 'K'),
            'description': f'Daily mean temperature aggregated from hourly HARv2 data for {year}',
            'source': 'HARv2 (High Asia Refined analysis version 2)',
            'temporal_resolution': 'daily',
            'aggregation_method': 'mean'
        }
        
        # Save with compression
        print(f"   ├─ Saving: {output_files['mean'].name}")
        encoding = {
            't2_mean': {
                'zlib': True, 
                'complevel': 5, 
                'dtype': 'float32',
                'chunksizes': (30, 50, 50)  # Chunk in output file too
            },
            'time': {'dtype': 'float64'},
            'lon': {'dtype': 'float32'},
            'lat': {'dtype': 'float32'}
        }
        
        # Use compute() to execute the lazy operations
        ds_mean.to_netcdf(
            output_files['mean'], 
            encoding=encoding,
            compute=True  # Force computation
        )
        
        print(f"   ✅ Daily mean saved ({output_files['mean'].stat().st_size / 1024**2:.1f} MB)")
        
        # Clean up
        del daily_mean, ds_mean
        import gc
        gc.collect()
    else:
        print(f"   ✓ Daily mean already exists")
    
    # Minimum temperature
    if not output_files['min'].exists():
        print("   ├─ Calculating daily minimum...")
        
        daily_min = ds[temp_var].resample(time='1D').min()
        
        ds_min = xr.Dataset(
            {
                't2_min': daily_min
            },
            coords={
                'time': daily_min.time,
                'south_north': ds.south_north,
                'west_east': ds.west_east,
                'lat': ds.lat,
                'lon': ds.lon
            }
        )
        
        ds_min['t2_min'].attrs = {
            'long_name': 'Daily minimum 2-meter air temperature',
            'units': ds[temp_var].attrs.get('units', 'K'),
            'description': f'Daily minimum temperature aggregated from hourly HARv2 data for {year}',
            'source': 'HARv2 (High Asia Refined analysis version 2)',
            'temporal_resolution': 'daily',
            'aggregation_method': 'minimum'
        }
        
        print(f"   ├─ Saving: {output_files['min'].name}")
        encoding = {
            't2_min': {
                'zlib': True, 
                'complevel': 5, 
                'dtype': 'float32',
                'chunksizes': (30, 50, 50)
            },
            'time': {'dtype': 'float64'},
            'lon': {'dtype': 'float32'},
            'lat': {'dtype': 'float32'}
        }
        
        ds_min.to_netcdf(
            output_files['min'], 
            encoding=encoding,
            compute=True
        )
        
        print(f"   ✅ Daily minimum saved ({output_files['min'].stat().st_size / 1024**2:.1f} MB)")
        
        del daily_min, ds_min
        gc.collect()
    else:
        print(f"   ✓ Daily minimum already exists")
    
    # Maximum temperature
    if not output_files['max'].exists():
        print("   ├─ Calculating daily maximum...")
        
        daily_max = ds[temp_var].resample(time='1D').max()
        
        ds_max = xr.Dataset(
            {
                't2_max': daily_max
            },
            coords={
                'time': daily_max.time,
                'south_north': ds.south_north,
                'west_east': ds.west_east,
                'lat': ds.lat,
                'lon': ds.lon
            }
        )
        
        ds_max['t2_max'].attrs = {
            'long_name': 'Daily maximum 2-meter air temperature',
            'units': ds[temp_var].attrs.get('units', 'K'),
            'description': f'Daily maximum temperature aggregated from hourly HARv2 data for {year}',
            'source': 'HARv2 (High Asia Refined analysis version 2)',
            'temporal_resolution': 'daily',
            'aggregation_method': 'maximum'
        }
        
        print(f"   └─ Saving: {output_files['max'].name}")
        encoding = {
            't2_max': {
                'zlib': True, 
                'complevel': 5, 
                'dtype': 'float32',
                'chunksizes': (30, 50, 50)
            },
            'time': {'dtype': 'float64'},
            'lon': {'dtype': 'float32'},
            'lat': {'dtype': 'float32'}
        }
        
        ds_max.to_netcdf(
            output_files['max'], 
            encoding=encoding,
            compute=True
        )
        
        print(f"   ✅ Daily maximum saved ({output_files['max'].stat().st_size / 1024**2:.1f} MB)")
        
        del daily_max, ds_max
        gc.collect()
    else:
        print(f"   ✓ Daily maximum already exists")
    
    # Close input dataset
    ds.close()
    
    print(f"\n✅ Year {year} complete!")
    
    return output_files


def main():
    """Process all HAR temperature files"""
    
    print("="*80)
    print("🌡️  HAR TEMPERATURE DATA PREPROCESSING")
    print("   Hourly → Daily Aggregation")
    print("="*80)
    
    # Find HAR data directory
    har_dir = find_har_base_path()
    
    # Find all hourly temperature files
    pattern = "HARv2_d10km_h_2d_t2_*.nc"
    hourly_files = sorted(har_dir.glob(pattern))
    
    if not hourly_files:
        raise FileNotFoundError(f"No files matching {pattern} found in {har_dir}")
    
    print(f"\n📂 Found {len(hourly_files)} hourly temperature files:")
    for f in hourly_files:
        print(f"   • {f.name}")
    
    # Extract years from filenames
    years = []
    for f in hourly_files:
        # Extract year from filename like: HARv2_d10km_h_2d_t2_1984.nc
        try:
            year = int(f.stem.split('_')[-1])
            years.append(year)
        except ValueError:
            print(f"⚠️  Could not extract year from {f.name}, skipping...")
    
    print(f"\n📅 Years to process: {min(years)} - {max(years)} ({len(years)} years)")
    
    # Create output directory
    output_dir = Path.home() / "HAR_daily_temperature"
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Process each year
    print(f"\n{'='*80}")
    print("🚀 Starting processing...")
    print(f"{'='*80}")
    
    all_results = {}
    
    for hourly_file in hourly_files:
        # Extract year
        try:
            year = int(hourly_file.stem.split('_')[-1])
        except ValueError:
            continue
        
        try:
            results = process_hourly_to_daily(hourly_file, output_dir, year)
            all_results[year] = results
        
        except Exception as e:
            print(f"\n❌ Error processing {year}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successfully processed: {len(all_results)} years")
    print(f"📁 Output directory: {output_dir}")
    print(f"\nCreated files:")
    print(f"   • {len(all_results)} × daily mean files")
    print(f"   • {len(all_results)} × daily minimum files")
    print(f"   • {len(all_results)} × daily maximum files")
    print(f"   = {len(all_results) * 3} total daily files")
    
    # Calculate total size
    total_size = 0
    for files in all_results.values():
        for f in files.values():
            if f.exists():
                total_size += f.stat().st_size
    
    print(f"\n💾 Total output size: {total_size / 1024**3:.2f} GB")
    
    print(f"\n{'='*80}")
    print("✅ ALL PROCESSING COMPLETE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()