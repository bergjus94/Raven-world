"""
Update existing GloGEM NetCDF files with new glacier data from updated .dat files
This script reads new .dat files and replaces the data for matching glacier IDs in the NetCDF
"""
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
import logging
import gc
import shutil

def setup_logger():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('GloGEM_Updater')

def parse_dat_file(dat_file: Path, logger) -> pd.DataFrame:
    """
    Parse a single .dat file into long-format DataFrame
    
    Parameters
    ----------
    dat_file : Path
        Path to .dat file
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

def update_component(base_dir: Path, update_dir: Path, component: str, scenario: str = 'ssp126', backup: bool = True):
    """
    Update NetCDF file with new glacier data from .dat files
    
    Parameters
    ----------
    base_dir : Path
        Directory containing the original NetCDF files (e.g., .../GloGEM/Indus/GMB/)
    update_dir : Path
        Directory containing the updated .dat files (e.g., .../GloGEM/Indus/Snowline_update/)
    component : str
        Component name (Discharge, Icemelt, Snowmelt, Rain)
    scenario : str
        Scenario name (default: ssp126)
    backup : bool
        Whether to create backup of original NetCDF (default: True)
    """
    logger = setup_logger()
    
    # Paths - NetCDF is in base_dir with scenario folder structure
    nc_file = base_dir / f'GloGEM_{component}_Indus_{scenario}.nc'
    
    # Update files are directly in update_dir, with naming like: *_updated_discharge.dat
    component_pattern = f'*_updated_{component.lower()}.dat'
    update_files = sorted(update_dir.glob(component_pattern))
    
    if not nc_file.exists():
        logger.error(f"❌ NetCDF file not found: {nc_file}")
        return
    
    if len(update_files) == 0:
        logger.warning(f"⚠️  No update files found in: {update_dir}")
        logger.warning(f"⚠️  Looking for pattern: {component_pattern}")
        return
    
    logger.info(f"="*70)
    logger.info(f"UPDATING {component} - {scenario}")
    logger.info(f"="*70)
    logger.info(f"NetCDF file: {nc_file}")
    logger.info(f"Update files: {len(update_files)} .dat files")
    
    # Backup original NetCDF
    if backup:
        backup_file = nc_file.with_suffix('.nc.backup')
        if not backup_file.exists():
            logger.info(f"📦 Creating backup: {backup_file.name}")
            shutil.copy2(nc_file, backup_file)
        else:
            logger.info(f"📦 Backup already exists: {backup_file.name}")
    
    # Load existing NetCDF
    logger.info(f"📂 Loading existing NetCDF...")
    ds = xr.open_dataset(nc_file)
    logger.info(f"   Current shape: {ds[component.lower()].shape}")
    logger.info(f"   Glaciers: {len(ds.glacier_id)}")
    logger.info(f"   Time steps: {len(ds.time)}")
    logger.info(f"   Date range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    # Parse all update files
    logger.info(f"\n📄 Reading {len(update_files)} update files...")
    all_dfs = []
    
    for i, dat_file in enumerate(update_files, 1):
        logger.info(f"   [{i}/{len(update_files)}] {dat_file.name}")
        df = parse_dat_file(dat_file, logger)
        if not df.empty:
            all_dfs.append(df)
            logger.debug(f"      → {len(df)} records, {df['glacier_id'].nunique()} glaciers")
    
    if not all_dfs:
        logger.warning("⚠️  No data found in update files!")
        ds.close()
        return
    
    # Combine all update data
    logger.info(f"\n🔗 Combining update data...")
    update_df = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()
    
    logger.info(f"   Total records: {len(update_df):,}")
    logger.info(f"   Unique glaciers: {update_df['glacier_id'].nunique()}")
    logger.info(f"   Unique dates: {update_df['date'].nunique()}")
    logger.info(f"   Date range: {update_df['date'].min()} to {update_df['date'].max()}")
    
    # Check for duplicates
    logger.info(f"\n🔍 Checking for duplicates...")
    duplicates = update_df.duplicated(subset=['date', 'glacier_id'], keep=False)
    n_duplicates = duplicates.sum()
    
    if n_duplicates > 0:
        logger.warning(f"   Found {n_duplicates} duplicate entries!")
        logger.warning(f"   Keeping first occurrence of each (date, glacier_id)")
        update_df = update_df.drop_duplicates(subset=['date', 'glacier_id'], keep='first')
        gc.collect()
    else:
        logger.info(f"   ✅ No duplicates found")
    
    # Find which glaciers are being updated
    update_glacier_ids = update_df['glacier_id'].unique()
    existing_glacier_ids = ds.glacier_id.values.astype(str)
    
    matching_glaciers = set(update_glacier_ids) & set(existing_glacier_ids)
    new_glaciers = set(update_glacier_ids) - set(existing_glacier_ids)
    
    logger.info(f"\n📊 Glacier matching:")
    logger.info(f"   Glaciers in update files: {len(update_glacier_ids)}")
    logger.info(f"   Glaciers in NetCDF: {len(existing_glacier_ids)}")
    logger.info(f"   ✅ Matching (will be updated): {len(matching_glaciers)}")
    logger.info(f"   ⚠️  New (not in NetCDF): {len(new_glaciers)}")
    
    if len(matching_glaciers) == 0:
        logger.error("❌ No matching glaciers found! Cannot update.")
        ds.close()
        return
    
    if len(new_glaciers) > 0:
        logger.warning(f"\n⚠️  Warning: {len(new_glaciers)} glaciers in update files are not in the NetCDF:")
        for gid in list(new_glaciers)[:10]:  # Show first 10
            logger.warning(f"      - {gid}")
        if len(new_glaciers) > 10:
            logger.warning(f"      ... and {len(new_glaciers)-10} more")
        logger.warning(f"   These glaciers will be IGNORED (not added to NetCDF)")
    
    # Filter update data to only matching glaciers
    logger.info(f"\n🔄 Updating {len(matching_glaciers)} glaciers...")
    update_df_filtered = update_df[update_df['glacier_id'].isin(matching_glaciers)].copy()
    del update_df
    gc.collect()
    
    n_dates = update_df_filtered['date'].nunique()
    logger.info(f"   Pivoting update data to wide format ({n_dates} dates × {len(matching_glaciers)} glaciers)...")
    
    # ✅ FAST APPROACH: Pivot update data to wide format (time × glacier_id)
    update_pivot = update_df_filtered.pivot(index='date', columns='glacier_id', values='value')
    del update_df_filtered
    gc.collect()
    
    logger.info(f"   Aligning update data with NetCDF dimensions...")
    # Reindex to match NetCDF dimensions exactly (fills missing with NaN)
    update_pivot = update_pivot.reindex(
        index=pd.to_datetime(ds.time.values),
        columns=ds.glacier_id.values.astype(str),
        fill_value=np.nan
    )
    
    logger.info(f"   Applying vectorized update to NetCDF array...")
    # ✅ FAST: Get numpy array from xarray and update in-place using boolean mask
    # This is a single vectorized operation instead of millions of row-by-row updates
    arr = ds[component.lower()].values
    update_arr = update_pivot.values.astype(np.float32)
    
    # Create mask for non-NaN values in update array
    mask = ~np.isnan(update_arr)
    n_updated = mask.sum()
    
    logger.info(f"   Replacing {n_updated:,} values in NetCDF array...")
    # Single vectorized assignment - THIS IS THE KEY SPEEDUP
    arr[mask] = update_arr[mask]
    
    # Update the xarray Dataset with modified array
    ds[component.lower()].values = arr
    
    logger.info(f"   ✅ Updated {n_updated:,} values")
    
    del update_pivot, update_arr, mask
    gc.collect()
    
    # ✅ Prepare updated dataset for saving
    logger.info(f"\n💾 Preparing to save...")
    
    # Update metadata
    logger.info(f"   Updating metadata...")
    ds.attrs['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ds.attrs['update_info'] = f"Updated {len(matching_glaciers)} glaciers from {len(update_files)} files"
    
    # Load the updated array into memory before closing
    logger.info(f"   Loading updated data into memory...")
    updated_array = arr.copy()
    
    # Get all the metadata we need before closing
    time_values = ds.time.values
    glacier_id_values = ds.glacier_id.values
    attrs = dict(ds.attrs)
    
    # ✅ CRITICAL: Close the original dataset to release file lock
    logger.info(f"   Closing original dataset...")
    ds.close()
    del ds, arr
    gc.collect()
    
    # Create new dataset with updated values
    logger.info(f"   Creating new dataset with updated values...")
    ds_updated = xr.Dataset(
        {component.lower(): (['time', 'glacier_id'], updated_array)},
        coords={
            'time': time_values,
            'glacier_id': glacier_id_values
        },
        attrs=attrs
    )
    
    del updated_array
    gc.collect()
    
    # Save updated NetCDF
    logger.info(f"\n💾 Saving updated NetCDF...")
    encoding = {component.lower(): {'zlib': True, 'complevel': 5, 'dtype': 'float32'}}
    ds_updated.to_netcdf(nc_file, encoding=encoding)
    
    file_size_mb = nc_file.stat().st_size / 1024 / 1024
    logger.info(f"   ✅ Done! File size: {file_size_mb:.1f} MB")
    logger.info(f"   Shape: {ds_updated[component.lower()].shape}")
    
    logger.info(f"\n" + "="*70)
    logger.info(f"✅ UPDATE COMPLETE FOR {component}")
    logger.info(f"="*70)
    logger.info(f"Summary:")
    logger.info(f"   - Glaciers updated: {len(matching_glaciers)}")
    logger.info(f"   - Values replaced: {n_updated:,}")
    logger.info(f"   - Update files processed: {len(update_files)}")
    if backup:
        logger.info(f"   - Backup saved: {backup_file.name}")
    logger.info(f"="*70)

def update_all_components(base_dir: Path, update_dir: Path, scenario: str = 'ssp126', 
                         components: list = None, backup: bool = True):
    """
    Update all components (or specified subset)
    
    Parameters
    ----------
    base_dir : Path
        Directory containing the original NetCDF files
    update_dir : Path
        Directory containing the updated .dat files
    scenario : str
        Scenario name (default: ssp126)
    components : list, optional
        List of components to update. If None, updates all: ['Discharge', 'Icemelt', 'Snowmelt', 'Rain']
    backup : bool
        Whether to create backups (default: True)
    """
    if components is None:
        components = ['Discharge', 'Icemelt', 'Snowmelt', 'Rain']
    
    logger = setup_logger()
    logger.info(f"\n{'='*70}")
    logger.info(f"GLOGEM NETCDF UPDATE SCRIPT")
    logger.info(f"{'='*70}")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Update directory: {update_dir}")
    logger.info(f"Scenario: {scenario}")
    logger.info(f"Components: {', '.join(components)}")
    logger.info(f"Backup: {backup}")
    logger.info(f"{'='*70}\n")
    
    for component in components:
        try:
            update_component(base_dir, update_dir, component, scenario, backup)
            logger.info("\n")
        except Exception as e:
            logger.error(f"❌ Error updating {component}: {e}")
            import traceback
            traceback.print_exc()
            logger.info("\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Update GloGEM NetCDF files with new glacier data from .dat files'
    )
    parser.add_argument(
        '--base-dir', 
        required=True,
        help='Directory containing original NetCDF files (e.g., /path/to/01_data/GloGEM/Indus/GMB/)'
    )
    parser.add_argument(
        '--update-dir', 
        required=True,
        help='Directory containing new .dat files (e.g., /path/to/01_data/GloGEM/Indus/Snowline_update/)'
    )
    parser.add_argument(
        '--scenario', 
        default='ssp126',
        help='Scenario name (default: ssp126)'
    )
    parser.add_argument(
        '--component', 
        choices=['Discharge', 'Icemelt', 'Snowmelt', 'Rain', 'all'], 
        default='all',
        help='Component to update (default: all)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup of original NetCDF files'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    update_dir = Path(args.update_dir)
    
    if not base_dir.exists():
        print(f"❌ Error: Base directory does not exist: {base_dir}")
        exit(1)
    
    if not update_dir.exists():
        print(f"❌ Error: Update directory does not exist: {update_dir}")
        exit(1)
    
    components = ['Discharge', 'Icemelt', 'Snowmelt', 'Rain'] if args.component == 'all' else [args.component]
    
    update_all_components(
        base_dir=base_dir,
        update_dir=update_dir,
        scenario=args.scenario,
        components=components,
        backup=not args.no_backup
    )
