"""
Simple GloGEM converter for servers with sufficient RAM (16+ GB)
UPDATED: Handles duplicate glaciers across files correctly
"""
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
import logging
import gc

def setup_logger():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('GloGEM')

def parse_dat_file(dat_file: Path, seen_glacier_years: set, logger) -> pd.DataFrame:
    """
    Parse a single .dat file, skipping duplicate (glacier_id, year) combinations
    
    Parameters
    ----------
    dat_file : Path
        Path to .dat file
    seen_glacier_years : set
        Set of (glacier_id, year) tuples already processed
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns [glacier_id, date, value]
    """
    records = []
    
    with open(dat_file, 'r') as f:
        for line in f:
            if line.startswith("ID") or line.startswith("//") or line.strip() == "":
                continue
            
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            glacier_id = parts[0]
            
            try:
                year = int(parts[1])
                
                # ✅ FIX: Check for (glacier_id, year) combination instead of just glacier_id
                if (glacier_id, year) in seen_glacier_years:
                    continue
                
                # Mark this (glacier_id, year) as seen
                seen_glacier_years.add((glacier_id, year))
                
                start_date = datetime(year-1, 10, 1)
                
                for day, val in enumerate(parts[3:]):
                    try:
                        value = float(val) if val != '*' else np.nan
                    except ValueError:
                        value = np.nan
                    
                    date = start_date + timedelta(days=day)
                    records.append((glacier_id, date, value))
                
            except (ValueError, IndexError):
                continue
    
    if records:
        return pd.DataFrame(records, columns=['glacier_id', 'date', 'value'])
    return pd.DataFrame()

def process_component(base_dir: Path, component: str, model: str, scenario: str = 'ssp126'):
    """Process all .dat files for a component/model/scenario combination"""
    logger = setup_logger()

    output_dir = base_dir / component / model / scenario
    output_file = base_dir / f'GloGEM_{component}_Indus_{model}_{scenario}.nc'
    
    if output_file.exists():
        logger.info(f"✅ Already exists: {output_file}")
        return xr.open_dataset(output_file)
    
    dat_files = sorted(output_dir.glob('*.dat'))
    logger.info(f"Processing {len(dat_files)} files for {component}")
    
    # ✅ Track (glacier_id, year) combinations instead of just glacier_id
    seen_glacier_years = set()
    
    # Parse all files
    all_dfs = []
    
    for i, dat_file in enumerate(dat_files, 1):
        logger.info(f"📄 {i}/{len(dat_files)}: {dat_file.name}")
        df = parse_dat_file(dat_file, seen_glacier_years, logger)
        if not df.empty:
            all_dfs.append(df)
            logger.debug(f"   Added {len(df)} records, {df['glacier_id'].nunique()} glaciers")
        
        # Periodic garbage collection
        if i % 10 == 0:
            gc.collect()
    
    logger.info(f"Combining {len(all_dfs)} DataFrames...")
    combined = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()
    
    logger.info(f"Total records: {len(combined):,}")
    logger.info(f"Unique glaciers: {combined['glacier_id'].nunique()}")
    logger.info(f"Unique dates: {combined['date'].nunique()}")
    logger.info(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    
    # ✅ Check for duplicates before pivoting
    logger.info("Checking for duplicate (date, glacier_id) combinations...")
    duplicates = combined.duplicated(subset=['date', 'glacier_id'], keep=False)
    n_duplicates = duplicates.sum()
    
    if n_duplicates > 0:
        logger.warning(f"Found {n_duplicates} duplicate entries!")
        logger.warning("Keeping first occurrence of each (date, glacier_id)")
        combined = combined.drop_duplicates(subset=['date', 'glacier_id'], keep='first')
        gc.collect()
    else:
        logger.info("✅ No duplicates found")
    
    # Pivot to wide format
    logger.info("Pivoting to wide format...")
    pivot = combined.pivot(index='date', columns='glacier_id', values='value')
    del combined
    gc.collect()
    
    logger.info(f"Pivot shape: {pivot.shape}")
    logger.info(f"   Time steps: {len(pivot.index)}")
    logger.info(f"   Glaciers: {len(pivot.columns)}")
    
    # Convert to xarray
    logger.info("Converting to xarray...")
    ds = xr.Dataset(
        {component.lower(): (['time', 'glacier_id'], pivot.values.astype(np.float32))},
        coords={
            'time': pivot.index,
            'glacier_id': pivot.columns.astype(str)
        }
    )
    
    del pivot
    gc.collect()
    
    # Add metadata
    ds.attrs.update({
        'title': f'GloGEM {component} for Indus Basin ({model})',
        'scenario': scenario,
        'model': model,
        'source': 'GloGEM glacier model',
        'units': 'mm/day',
        'date_range': f"{ds.time.values[0]} to {ds.time.values[-1]}",
        'n_glaciers': len(ds.glacier_id),
        'n_timesteps': len(ds.time),
    })
    
    # Save with compression
    logger.info(f"Saving to {output_file}...")
    encoding = {component.lower(): {'zlib': True, 'complevel': 5, 'dtype': 'float32'}}
    ds.to_netcdf(output_file, encoding=encoding)
    
    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logger.info(f"✅ Done! File size: {file_size_mb:.1f} MB")
    logger.info(f"   Shape: {ds[component.lower()].shape}")
    logger.info(f"   Date range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    # Calculate and log compression ratio
    dat_files_size = sum(f.stat().st_size for f in dat_files) / 1024 / 1024
    compression_ratio = (1 - file_size_mb / dat_files_size) * 100
    logger.info(f"   Original .dat size: {dat_files_size:.1f} MB")
    logger.info(f"   Compression: {compression_ratio:.1f}% reduction")
    
    return ds

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default='/home/jberg@giub.local/Raven_world/01_data/GloGEM/Indus/Snowline_update')
    parser.add_argument('--model', nargs='+', default=['gfdl-esm4'],
                        help='One or more GCM model names (e.g. gfdl-esm4 mpi-esm1-2-hr)')
    parser.add_argument('--scenario', nargs='+', default=['ssp126'],
                        help='One or more scenarios (e.g. ssp126 ssp585)')
    parser.add_argument('--component', choices=['Discharge', 'Icemelt', 'Snowmelt', 'Rain', 'all'], default='all')

    args = parser.parse_args()
    base_dir = Path(args.base_dir)

    components = ['Discharge', 'Icemelt', 'Snowmelt', 'Rain'] if args.component == 'all' else [args.component]

    for model in args.model:
        for scenario in args.scenario:
            for component in components:
                process_component(base_dir, component, model, scenario)
        