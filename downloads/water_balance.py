#!/usr/bin/env python3

"""
Water Balance Analysis Script
Converted from water_balance.ipynb

Calculates and visualizes annual water balance components for catchments:
- Observed streamflow (with ≥95% data coverage filter)
- Precipitation (ERA5 and HAR)  
- Potential Evapotranspiration (ERA5 and HAR)
- Glacier runoff components (ice melt, snow melt, rain)

Usage:
    python water_balance.py
    
Configuration:
    Edit the paths in the main() function below.
"""

import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def calculate_annual_precipitation(catchment_folder):
    """Calculate annual precipitation for glacier-free area from ERA5 data."""
    catchment_id = os.path.basename(catchment_folder)
    
    try:
        hru_path = os.path.join(catchment_folder, 'topo_files', 'HRU.shp')
        precip_path = os.path.join(catchment_folder, 'data_obs', 'era5_land_precip.nc')
        gridweights_path = os.path.join(catchment_folder, 'data_obs', 'GridWeights.txt')
        
        if not all([os.path.exists(p) for p in [hru_path, precip_path, gridweights_path]]):
            print(f"  WARNING: Missing files for {catchment_id}")
            return None
        
        gdf = gpd.read_file(hru_path)
        non_glacier_hrus = gdf[~gdf['Landuse_Cl'].isin([7, 8])].copy()
        non_glacier_hru_ids = non_glacier_hrus['HRU_ID'].values
        
        if len(non_glacier_hrus) == 0:
            print(f"  WARNING: No non-glacier HRUs found for {catchment_id}")
            return None
        
        total_glacier_free_area = non_glacier_hrus['Area_km2'].sum()
        
        with open(gridweights_path, 'r') as f:
            lines = f.readlines()
        
        data_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith(':'):
                parts = line.split()
                if len(parts) == 3:
                    try:
                        data_lines.append([int(parts[0]), int(parts[1]), float(parts[2])])
                    except:
                        pass
        
        grid_weights = pd.DataFrame(data_lines, columns=['HRU_ID', 'Cell_ID', 'Weight'])
        non_glacier_weights = grid_weights[grid_weights['HRU_ID'].isin(non_glacier_hru_ids)].copy()
        
        ds = xr.open_dataset(precip_path)
        precip_data = ds['tp'] if 'tp' in ds.variables else ds['precip'] if 'precip' in ds.variables else None
        
        if precip_data is None:
            print(f"  WARNING: Precipitation variable not found for {catchment_id}")
            ds.close()
            return None
        
        time_index = pd.to_datetime(precip_data.time.values)
        years = time_index.year.unique()
        annual_precip = {}
        
        for year in years:
            year_mask = time_index.year == year
            precip_year = precip_data.isel(time=year_mask)
            annual_total = precip_year.sum(dim='time')
            
            weighted_precip = 0
            total_weight = 0
            
            for _, row in non_glacier_weights.iterrows():
                hru_id = row['HRU_ID']
                grid_id = int(row['Cell_ID']) - 1
                weight = row['Weight']
                hru_area = non_glacier_hrus[non_glacier_hrus['HRU_ID'] == hru_id]['Area_km2'].values
                
                if len(hru_area) > 0:
                    hru_area = hru_area[0]
                    try:
                        grid_precip = annual_total.values.flat[grid_id]
                        weighted_precip += grid_precip * weight * hru_area
                        total_weight += weight * hru_area
                    except:
                        pass
            
            if total_weight > 0:
                annual_precip[year] = weighted_precip / total_weight
        
        mean_annual_precip = np.mean(list(annual_precip.values())) if annual_precip else np.nan
        
        result = {
            'catchment_id': catchment_id,
            'glacier_free_area_km2': total_glacier_free_area,
            'mean_annual_precip_mm': mean_annual_precip,
            **{f'precip_{year}_mm': precip for year, precip in annual_precip.items()}
        }
        
        ds.close()
        return result
    except Exception as e:
        print(f"  ERROR processing {catchment_id}: {str(e)}")
        return None


def calculate_annual_precipitation_har(catchment_folder, base_dir_har):
    """Calculate annual precipitation for glacier-free area from HAR data."""
    catchment_id = os.path.basename(catchment_folder)
    
    try:
        hru_path = os.path.join(catchment_folder, 'topo_files', 'HRU.shp')
        catchment_folder_har = os.path.join(base_dir_har, catchment_id)
        precip_path = os.path.join(catchment_folder_har, 'data_obs', 'har_precip.nc')
        gridweights_path = os.path.join(catchment_folder_har, 'data_obs', 'GridWeights_HAR.txt')
        
        if not all([os.path.exists(p) for p in [hru_path, precip_path, gridweights_path]]):
            print(f"  WARNING: Missing HAR files for {catchment_id}")
            return None
        
        gdf = gpd.read_file(hru_path)
        non_glacier_hrus = gdf[~gdf['Landuse_Cl'].isin([7, 8])].copy()
        non_glacier_hru_ids = non_glacier_hrus['HRU_ID'].values
        
        if len(non_glacier_hrus) == 0:
            return None
        
        total_glacier_free_area = non_glacier_hrus['Area_km2'].sum()
        
        with open(gridweights_path, 'r') as f:
            lines = f.readlines()
        
        data_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith(':'):
                parts = line.split()
                if len(parts) == 3:
                    try:
                        data_lines.append([int(parts[0]), int(parts[1]), float(parts[2])])
                    except:
                        pass
        
        grid_weights = pd.DataFrame(data_lines, columns=['HRU_ID', 'Cell_ID', 'Weight'])
        non_glacier_weights = grid_weights[grid_weights['HRU_ID'].isin(non_glacier_hru_ids)].copy()
        
        ds = xr.open_dataset(precip_path)
        
        # Find the precipitation variable - same approach as notebook
        precip_var = None
        for var in ds.data_vars:
            if len(ds[var].dims) >= 2:
                precip_var = var
                break
        
        if precip_var is None:
            print(f"  WARNING: Could not find precipitation variable for {catchment_id}")
            ds.close()
            return None
        
        precip_data = ds[precip_var]
        time_index = pd.to_datetime(ds.time.values)
        years = time_index.year.unique()
        annual_precip = {}
        
        for year in years:
            year_mask = time_index.year == year
            precip_year = precip_data.isel(time=year_mask)
            annual_total = precip_year.sum(dim='time')
            
            weighted_precip = 0
            total_weight = 0
            
            for _, row in non_glacier_weights.iterrows():
                hru_id = row['HRU_ID']
                grid_id = int(row['Cell_ID']) - 1
                weight = row['Weight']
                hru_area = non_glacier_hrus[non_glacier_hrus['HRU_ID'] == hru_id]['Area_km2'].values
                
                if len(hru_area) > 0:
                    hru_area = hru_area[0]
                    try:
                        grid_precip = annual_total.values.flat[grid_id]
                        weighted_precip += grid_precip * weight * hru_area
                        total_weight += weight * hru_area
                    except:
                        pass
            
            if total_weight > 0:
                annual_precip[year] = weighted_precip / total_weight
        
        mean_annual_precip = np.mean(list(annual_precip.values())) if annual_precip else np.nan
        
        result = {
            'catchment_id': catchment_id,
            'glacier_free_area_km2': total_glacier_free_area,
            'mean_annual_precip_mm': mean_annual_precip,
            **{f'precip_{year}_mm': precip for year, precip in annual_precip.items()}
        }
        
        ds.close()
        return result
    except Exception as e:
        print(f"  ERROR processing {catchment_id}: {str(e)}")
        return None


def calculate_annual_pet_era5(catchment_folder):
    """Calculate annual PET for glacier-free area from ERA5 data."""
    catchment_id = os.path.basename(catchment_folder)
    
    try:
        hru_path = os.path.join(catchment_folder, 'topo_files', 'HRU.shp')
        pet_path = os.path.join(catchment_folder, 'data_obs', 'era5_land_pet.nc')
        gridweights_path = os.path.join(catchment_folder, 'data_obs', 'GridWeights.txt')
        
        if not all([os.path.exists(p) for p in [hru_path, pet_path, gridweights_path]]):
            print(f"  WARNING: Missing ERA5 PET files for {catchment_id}")
            return None
        
        gdf = gpd.read_file(hru_path)
        non_glacier_hrus = gdf[~gdf['Landuse_Cl'].isin([7, 8])].copy()
        non_glacier_hru_ids = non_glacier_hrus['HRU_ID'].values
        
        if len(non_glacier_hrus) == 0:
            return None
        
        total_glacier_free_area = non_glacier_hrus['Area_km2'].sum()
        
        with open(gridweights_path, 'r') as f:
            lines = f.readlines()
        
        data_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith(':'):
                parts = line.split()
                if len(parts) == 3:
                    try:
                        data_lines.append([int(parts[0]), int(parts[1]), float(parts[2])])
                    except:
                        pass
        
        grid_weights = pd.DataFrame(data_lines, columns=['HRU_ID', 'Cell_ID', 'Weight'])
        non_glacier_weights = grid_weights[grid_weights['HRU_ID'].isin(non_glacier_hru_ids)].copy()
        
        ds = xr.open_dataset(pet_path)
        
        pet_data = None
        for var in ['pev', 'pet', 'PET']:
            if var in ds.variables:
                pet_data = ds[var]
                break
        
        if pet_data is None:
            ds.close()
            return None
        
        time_index = pd.to_datetime(pet_data.time.values)
        years = time_index.year.unique()
        annual_pet = {}
        
        for year in years:
            year_mask = time_index.year == year
            pet_year = pet_data.isel(time=year_mask)
            annual_total = pet_year.sum(dim='time')
            
            # Check units and convert if necessary
            if 'units' in pet_data.attrs:
                units = pet_data.attrs['units'].lower()
                if 'm of water equivalent' in units or units == 'm':
                    annual_total = annual_total * 1000
            
            weighted_pet = 0
            total_weight = 0
            
            for _, row in non_glacier_weights.iterrows():
                hru_id = row['HRU_ID']
                grid_id = int(row['Cell_ID']) - 1
                weight = row['Weight']
                hru_area = non_glacier_hrus[non_glacier_hrus['HRU_ID'] == hru_id]['Area_km2'].values
                
                if len(hru_area) > 0:
                    hru_area = hru_area[0]
                    try:
                        grid_pet = annual_total.values.flat[grid_id]
                        weighted_pet += grid_pet * weight * hru_area
                        total_weight += weight * hru_area
                    except:
                        pass
            
            if total_weight > 0:
                annual_pet[year] = weighted_pet / total_weight
        
        mean_annual_pet = np.mean(list(annual_pet.values())) if annual_pet else np.nan
        
        result = {
            'catchment_id': catchment_id,
            'glacier_free_area_km2': total_glacier_free_area,
            'mean_annual_pet_mm': mean_annual_pet,
            **{f'pet_{year}_mm': pet for year, pet in annual_pet.items()}
        }
        
        ds.close()
        return result
    except Exception as e:
        print(f"  ERROR processing {catchment_id}: {str(e)}")
        return None


def calculate_annual_pet_har(catchment_folder, base_dir_har):
    """Calculate annual PET for glacier-free area from HAR data."""
    catchment_id = os.path.basename(catchment_folder)
    
    try:
        hru_path = os.path.join(catchment_folder, 'topo_files', 'HRU.shp')
        catchment_folder_har = os.path.join(base_dir_har, catchment_id)
        pet_path = os.path.join(catchment_folder_har, 'data_obs', 'har_pet.nc')
        gridweights_path = os.path.join(catchment_folder_har, 'data_obs', 'GridWeights_HAR.txt')
        
        if not all([os.path.exists(p) for p in [hru_path, pet_path, gridweights_path]]):
            print(f"  WARNING: Missing HAR PET files for {catchment_id}")
            return None
        
        gdf = gpd.read_file(hru_path)
        non_glacier_hrus = gdf[~gdf['Landuse_Cl'].isin([7, 8])].copy()
        non_glacier_hru_ids = non_glacier_hrus['HRU_ID'].values
        
        if len(non_glacier_hrus) == 0:
            return None
        
        total_glacier_free_area = non_glacier_hrus['Area_km2'].sum()
        
        with open(gridweights_path, 'r') as f:
            lines = f.readlines()
        
        data_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith(':'):
                parts = line.split()
                if len(parts) == 3:
                    try:
                        data_lines.append([int(parts[0]), int(parts[1]), float(parts[2])])
                    except:
                        pass
        
        grid_weights = pd.DataFrame(data_lines, columns=['HRU_ID', 'Cell_ID', 'Weight'])
        non_glacier_weights = grid_weights[grid_weights['HRU_ID'].isin(non_glacier_hru_ids)].copy()
        
        ds = xr.open_dataset(pet_path)
        
        pet_data = None
        for var in ['potevap', 'pev', 'pet', 'PET']:
            if var in ds.variables:
                pet_data = ds[var]
                break
        
        if pet_data is None:
            ds.close()
            return None
        
        time_index = pd.to_datetime(pet_data.time.values)
        years = time_index.year.unique()
        annual_pet = {}
        
        for year in years:
            year_mask = time_index.year == year
            pet_year = pet_data.isel(time=year_mask)
            annual_total = pet_year.sum(dim='time')
            
            # Check units and convert if necessary
            if 'units' in pet_data.attrs:
                units = pet_data.attrs['units'].lower()
                if 'm of water equivalent' in units or units == 'm':
                    annual_total = annual_total * 1000
            
            weighted_pet = 0
            total_weight = 0
            
            for _, row in non_glacier_weights.iterrows():
                hru_id = row['HRU_ID']
                grid_id = int(row['Cell_ID']) - 1
                weight = row['Weight']
                hru_area = non_glacier_hrus[non_glacier_hrus['HRU_ID'] == hru_id]['Area_km2'].values
                
                if len(hru_area) > 0:
                    hru_area = hru_area[0]
                    try:
                        grid_pet = annual_total.values.flat[grid_id]
                        weighted_pet += grid_pet * weight * hru_area
                        total_weight += weight * hru_area
                    except:
                        pass
            
            if total_weight > 0:
                annual_pet[year] = weighted_pet / total_weight
        
        mean_annual_pet = np.mean(list(annual_pet.values())) if annual_pet else np.nan
        
        result = {
            'catchment_id': catchment_id,
            'glacier_free_area_km2': total_glacier_free_area,
            'mean_annual_pet_mm': mean_annual_pet,
            **{f'pet_{year}_mm': pet for year, pet in annual_pet.items()}
        }
        
        ds.close()
        return result
    except Exception as e:
        print(f"  ERROR processing {catchment_id}: {str(e)}")
        return None


def calculate_annual_streamflow(catchment_folder, min_data_coverage=0.95):
    """Calculate annual observed streamflow from Q_daily.rvt file."""
    catchment_id = os.path.basename(catchment_folder)
    
    try:
        q_path = os.path.join(catchment_folder, 'HBV', 'data_obs', 'Q_daily.rvt')
        hru_path = os.path.join(catchment_folder, 'topo_files', 'HRU.shp')
        
        if not os.path.exists(q_path):
            print(f"  WARNING: Q_daily.rvt not found for {catchment_id}")
            return None
        
        if not os.path.exists(hru_path):
            return None
        
        gdf = gpd.read_file(hru_path)
        total_area_km2 = gdf['Area_km2'].sum()
        
        with open(q_path, 'r') as f:
            lines = f.readlines()
        
        header_line = lines[1].strip().split()
        start_date = header_line[0]
        
        values = []
        for i in range(2, len(lines)):
            line = lines[i].strip()
            if line:
                try:
                    values.append(float(line))
                except:
                    pass
        
        start = pd.to_datetime(start_date)
        dates = pd.date_range(start=start, periods=len(values), freq='D')
        
        df_q = pd.DataFrame({'date': dates, 'Q_m3s': values})
        df_q['year'] = df_q['date'].dt.year
        df_q['Q_valid'] = df_q['Q_m3s'] != -1.2345
        df_q.loc[~df_q['Q_valid'], 'Q_m3s'] = np.nan
        
        annual_stats = []
        for year in df_q['year'].unique():
            year_data = df_q[df_q['year'] == year]
            total_days = len(year_data)
            valid_days = year_data['Q_valid'].sum()
            coverage = valid_days / total_days
            
            if coverage >= min_data_coverage:
                mean_q_m3s = year_data['Q_m3s'].mean()
                annual_volume_m3 = mean_q_m3s * 86400 * 365.25
                annual_depth_m = annual_volume_m3 / (total_area_km2 * 1e6)
                annual_depth_mm = annual_depth_m * 1000
                
                annual_stats.append({
                    'year': year,
                    'mean_Q_m3s': mean_q_m3s,
                    'mean_Q_mm': annual_depth_mm,
                    'data_coverage': coverage,
                    'valid_days': int(valid_days)
                })
        
        if len(annual_stats) == 0:
            return None
        
        mean_annual_q_m3s = np.mean([s['mean_Q_m3s'] for s in annual_stats])
        mean_annual_q_mm = np.mean([s['mean_Q_mm'] for s in annual_stats])
        
        result = {
            'catchment_id': catchment_id,
            'catchment_area_km2': total_area_km2,
            'mean_annual_Q_m3s': mean_annual_q_m3s,
            'mean_annual_Q_mm': mean_annual_q_mm,
            'num_years_with_data': len(annual_stats),
            'min_data_coverage': min_data_coverage
        }
        
        for stat in annual_stats:
            result[f'Q_{stat["year"]}_m3s'] = stat['mean_Q_m3s']
            result[f'Q_{stat["year"]}_mm'] = stat['mean_Q_mm']
        
        return result
    except Exception as e:
        print(f"  ERROR processing {catchment_id}: {str(e)}")
        return None


def calculate_annual_glacier_runoff(catchment_id, base_dir):
    """Calculate annual glacier runoff components from GloGEM data.
    
    The GloGEM data is in mm/year over the glacier area and needs to be scaled
    to the total catchment area for water balance calculations.
    """
    catchment_dir = os.path.join(base_dir, f'catchment_{catchment_id}')
    glogem_file = os.path.join(catchment_dir, 'topo_files', 'GloGEM_catchment_averaged.csv')
    hru_path = os.path.join(catchment_dir, 'topo_files', 'HRU.shp')
    
    if not os.path.exists(glogem_file):
        print(f"  WARNING: GloGEM file not found for catchment_{catchment_id}")
        return None
    
    if not os.path.exists(hru_path):
        print(f"  WARNING: HRU shapefile not found for catchment_{catchment_id}")
        return None
    
    # Read HRU shapefile to get catchment and glacier areas
    gdf = gpd.read_file(hru_path)
    total_catchment_area_km2 = gdf['Area_km2'].sum()
    glacier_hrus = gdf[gdf['Landuse_Cl'].isin([7, 8])]
    glacier_area_km2 = glacier_hrus['Area_km2'].sum()
    
    if glacier_area_km2 == 0:
        print(f"  WARNING: No glacier area found for catchment_{catchment_id}")
        return None
    
    # Calculate scaling factor (glacier area / total catchment area)
    area_fraction = glacier_area_km2 / total_catchment_area_km2
    
    # Read GloGEM data with new format
    df = pd.read_csv(glogem_file)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # Sum daily values to get annual totals (data is in mm/day over glacier area)
    # Use '_all' columns which include both small and large glaciers
    annual_data = df.groupby('year').agg({
        'icemelt_all': 'sum',
        'snowmelt_all': 'sum',
        'rain_all': 'sum',
        'melt_all': 'sum'
    }).reset_index()
    
    # Scale to catchment area (multiply by glacier fraction)
    # This converts from mm/year over glacier area to mm/year over catchment area
    # Note: melt_all is the GloGEM 'Discharge' variable = ice melt + snow melt + rain (total runoff)
    # rain_all is stored separately as a sub-component only — do NOT add it to melt_all
    annual_data['annual_icemelt_mm'] = annual_data['icemelt_all'] * area_fraction
    annual_data['annual_snowmelt_mm'] = annual_data['snowmelt_all'] * area_fraction
    annual_data['annual_rain_mm'] = annual_data['rain_all'] * area_fraction
    annual_data['annual_melt_mm'] = annual_data['melt_all'] * area_fraction
    annual_data['annual_total_mm'] = annual_data['annual_melt_mm']  # melt_all already includes rain
    
    # Keep only the final columns
    annual_data = annual_data[['year', 'annual_icemelt_mm', 'annual_snowmelt_mm', 
                               'annual_rain_mm', 'annual_melt_mm', 'annual_total_mm']]
    
    return {
        'catchment_id': catchment_id,
        'glacier_area_km2': glacier_area_km2,
        'total_catchment_area_km2': total_catchment_area_km2,
        'glacier_area_fraction': area_fraction,
        'annual_data': annual_data,
        'mean_annual_icemelt_mm': annual_data['annual_icemelt_mm'].mean(),
        'mean_annual_snowmelt_mm': annual_data['annual_snowmelt_mm'].mean(),
        'mean_annual_rain_mm': annual_data['annual_rain_mm'].mean(),
        'mean_annual_melt_mm': annual_data['annual_melt_mm'].mean(),
        'mean_annual_total_mm': annual_data['annual_total_mm'].mean(),
        'n_years': len(annual_data)
    }


def plot_catchment_water_balance(catchment_id, results_precip, results_streamflow, results_glacier, output_dir, 
                                results_precip_har=None, results_pet_era5=None, results_pet_har=None):
    """Create a bar plot showing annual water balance components for a catchment."""
    
    # Find data for this catchment
    precip_data_era5 = next((r for r in results_precip if r['catchment_id'] == catchment_id), None)
    precip_data_har = next((r for r in results_precip_har if r['catchment_id'] == catchment_id), None) if results_precip_har else None
    pet_data_era5 = next((r for r in results_pet_era5 if r['catchment_id'] == catchment_id), None) if results_pet_era5 else None
    pet_data_har = next((r for r in results_pet_har if r['catchment_id'] == catchment_id), None) if results_pet_har else None
    streamflow_data = next((r for r in results_streamflow if r['catchment_id'] == catchment_id), None)
    glacier_id = catchment_id.replace('catchment_', '')
    glacier_data = next((r for r in results_glacier if r['catchment_id'] == glacier_id), None)
    
    if glacier_data:
        all_years = sorted(glacier_data['annual_data']['year'].unique())
    elif precip_data_era5:
        precip_years = [int(k.replace('precip_', '').replace('_mm', '')) 
                       for k in precip_data_era5.keys() if k.startswith('precip_') and k.endswith('_mm')]
        all_years = sorted(precip_years)
    else:
        print(f"No data available for catchment {catchment_id}, skipping plot")
        return
    
    streamflow_years_dict = {}
    if streamflow_data:
        for key in streamflow_data.keys():
            if key.startswith('Q_') and key.endswith('_mm'):
                year = int(key.replace('Q_', '').replace('_mm', ''))
                streamflow_years_dict[year] = streamflow_data[key]
    
    if streamflow_data:
        total_catchment_area = streamflow_data['catchment_area_km2']
    elif glacier_data and not np.isnan(glacier_data.get('total_catchment_area_km2', np.nan)):
        total_catchment_area = glacier_data['total_catchment_area_km2']
    else:
        print(f"  WARNING: Cannot determine total catchment area for {catchment_id} (no streamflow or glacier data), skipping plot")
        return
    
    glacier_free_area_era5 = precip_data_era5['glacier_free_area_km2'] if precip_data_era5 else total_catchment_area
    glacier_free_area_har = precip_data_har['glacier_free_area_km2'] if precip_data_har else total_catchment_area
    
    streamflow, precipitation_era5, precipitation_har = [], [], []
    pet_era5, pet_har = [], []
    glacier_total, glacier_icemelt, glacier_snowmelt = [], [], []
    
    for year in all_years:
        streamflow.append(streamflow_years_dict.get(year, np.nan))
        
        if precip_data_era5:
            precip_key = f'precip_{year}_mm'
            if precip_key in precip_data_era5:
                precip_scaled = precip_data_era5[precip_key] * (glacier_free_area_era5 / total_catchment_area)
                precipitation_era5.append(precip_scaled)
            else:
                precipitation_era5.append(np.nan)
        else:
            precipitation_era5.append(np.nan)

        if precip_data_har:
            precip_key = f'precip_{year}_mm'
            if precip_key in precip_data_har:
                precip_scaled = precip_data_har[precip_key] * (glacier_free_area_har / total_catchment_area)
                precipitation_har.append(precip_scaled)
            else:
                precipitation_har.append(np.nan)
        else:
            precipitation_har.append(np.nan)

        if pet_data_era5:
            pet_key = f'pet_{year}_mm'
            if pet_key in pet_data_era5:
                pet_scaled = pet_data_era5[pet_key] * (glacier_free_area_era5 / total_catchment_area)
                pet_era5.append(pet_scaled)
            else:
                pet_era5.append(np.nan)
        else:
            pet_era5.append(np.nan)

        if pet_data_har:
            pet_key = f'pet_{year}_mm'
            if pet_key in pet_data_har:
                pet_scaled = pet_data_har[pet_key] * (glacier_free_area_har / total_catchment_area)
                pet_har.append(pet_scaled)
            else:
                pet_har.append(np.nan)
        else:
            pet_har.append(np.nan)

        if glacier_data:
            glacier_annual = glacier_data['annual_data']
            year_data = glacier_annual[glacier_annual['year'] == year]
            if len(year_data) > 0:
                glacier_total.append(year_data['annual_total_mm'].values[0])
                glacier_icemelt.append(year_data['annual_icemelt_mm'].values[0])
                glacier_snowmelt.append(year_data['annual_snowmelt_mm'].values[0])
            else:
                glacier_total.append(np.nan)
                glacier_icemelt.append(np.nan)
                glacier_snowmelt.append(np.nan)
        else:
            glacier_total.append(np.nan)
            glacier_icemelt.append(np.nan)
            glacier_snowmelt.append(np.nan)
    
    years = np.array(all_years)
    fig, ax = plt.subplots(figsize=(18, 6))
    
    x = np.arange(len(years))
    width = 0.10
    
    streamflow_arr = np.array(streamflow)
    precipitation_era5_arr = np.array(precipitation_era5)
    precipitation_har_arr = np.array(precipitation_har)
    pet_era5_arr = np.array(pet_era5)
    pet_har_arr = np.array(pet_har)
    glacier_total_arr = np.array(glacier_total)
    glacier_icemelt_arr = np.array(glacier_icemelt)
    glacier_snowmelt_arr = np.array(glacier_snowmelt)
    
    ax.bar(x - 3.5*width, streamflow_arr, width, label='Streamflow (obs, ≥95% data)', color='#1f77b4', alpha=0.8)
    ax.bar(x - 2.5*width, precipitation_era5_arr, width, label='Precip (ERA5)', color='#2ca02c', alpha=0.8)
    ax.bar(x - 1.5*width, precipitation_har_arr, width, label='Precip (HAR)', color='#8fbc8f', alpha=0.8)
    ax.bar(x - 0.5*width, pet_era5_arr, width, label='PET (ERA5)', color='#d62728', alpha=0.8)
    ax.bar(x + 0.5*width, pet_har_arr, width, label='PET (HAR)', color='#ff9896', alpha=0.8)
    ax.bar(x + 1.5*width, glacier_total_arr, width, label='Glacier Total', color='#9467bd', alpha=0.8)
    ax.bar(x + 2.5*width, glacier_icemelt_arr, width, label='Glacier Ice Melt', color='#ff7f0e', alpha=0.8)
    ax.bar(x + 3.5*width, glacier_snowmelt_arr, width, label='Glacier Snow Melt', color='#bcbd22', alpha=0.8)
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Water Depth (mm/year)', fontsize=12, fontweight='bold')
    display_id = catchment_id.replace('catchment_', '')
    ax.set_title(f'Annual Water Balance Components - Catchment {display_id}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha='right')
    ax.legend(loc='upper left', framealpha=0.9, ncol=3, fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    if not np.all(np.isnan(streamflow_arr)):
        mean_q = np.nanmean(streamflow_arr)
        ax.axhline(y=mean_q, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.5)

    if precip_data_era5 and not np.all(np.isnan(precipitation_era5_arr)):
        mean_p = np.nanmean(precipitation_era5_arr)
        ax.axhline(y=mean_p, color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.5)

    if precip_data_har and not np.all(np.isnan(precipitation_har_arr)):
        mean_p_har = np.nanmean(precipitation_har_arr)
        ax.axhline(y=mean_p_har, color='#8fbc8f', linestyle='--', linewidth=1.5, alpha=0.5)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f'catchment_{display_id}_water_balance.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_file}")
    plt.close()


def plot_average_water_balance_single(catchment_id, results_precip, results_streamflow, results_glacier, 
                                     output_dir, results_precip_har=None, results_pet_era5=None, results_pet_har=None):
    """Create a bar plot showing average annual water balance components for a single catchment."""
    
    precip_data_era5 = next((r for r in results_precip if r['catchment_id'] == catchment_id), None)
    precip_data_har = next((r for r in results_precip_har if r['catchment_id'] == catchment_id), None) if results_precip_har else None
    pet_data_era5 = next((r for r in results_pet_era5 if r['catchment_id'] == catchment_id), None) if results_pet_era5 else None
    pet_data_har = next((r for r in results_pet_har if r['catchment_id'] == catchment_id), None) if results_pet_har else None
    streamflow_data = next((r for r in results_streamflow if r['catchment_id'] == catchment_id), None)
    glacier_id = catchment_id.replace('catchment_', '')
    glacier_data = next((r for r in results_glacier if r['catchment_id'] == glacier_id), None)
    
    if not streamflow_data:
        print(f"  WARNING: No streamflow data for {catchment_id}")
        return
    
    components, values, colors = [], [], []
    
    # Streamflow - dark grey
    components.append('Streamflow')
    values.append(streamflow_data['mean_annual_Q_mm'])
    colors.append('#4d4d4d')  # dark grey
    
    total_area = streamflow_data['catchment_area_km2']
    
    # Precipitation ERA5 - blue
    if precip_data_era5:
        glacier_free_area = precip_data_era5['glacier_free_area_km2']
        precip_scaled = precip_data_era5['mean_annual_precip_mm'] * (glacier_free_area / total_area)
        components.append('Precip\n(ERA5)')
        values.append(precip_scaled)
        colors.append('#1f77b4')  # blue
    
    # Precipitation HAR - lighter blue
    if precip_data_har:
        glacier_free_area = precip_data_har['glacier_free_area_km2']
        precip_scaled = precip_data_har['mean_annual_precip_mm'] * (glacier_free_area / total_area)
        components.append('Precip\n(HAR)')
        values.append(precip_scaled)
        colors.append('#87CEEB')  # sky blue (lighter blue)
    
    # PET ERA5 - green
    if pet_data_era5:
        glacier_free_area = pet_data_era5['glacier_free_area_km2']
        pet_scaled = pet_data_era5['mean_annual_pet_mm'] * (glacier_free_area / total_area)
        components.append('PET\n(ERA5)')
        values.append(pet_scaled)
        colors.append('#2ca02c')  # green
    
    # PET HAR - lighter green
    if pet_data_har:
        glacier_free_area = pet_data_har['glacier_free_area_km2']
        pet_scaled = pet_data_har['mean_annual_pet_mm'] * (glacier_free_area / total_area)
        components.append('PET\n(HAR)')
        values.append(pet_scaled)
        colors.append('#90EE90')  # light green
    
    # Glacier components - keep same colors
    if glacier_data:
        components.extend(['Glacier\nTotal', 'Ice\nMelt', 'Snow\nMelt'])
        values.extend([glacier_data['mean_annual_total_mm'], 
                      glacier_data['mean_annual_icemelt_mm'],
                      glacier_data['mean_annual_snowmelt_mm']])
        colors.extend(['#9467bd', '#ff7f0e', '#bcbd22'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x = np.arange(len(components))
    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Formatting - NO TITLE as requested
    ax.set_ylabel('Water Depth (mm/year)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)  # Bigger y-axis tick labels
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(values) * 1.15)
    
    # Add glacier percentage text in top right corner
    if precip_data_era5:
        glacier_free_area = precip_data_era5['glacier_free_area_km2']
        glacier_pct = (1 - glacier_free_area / total_area) * 100
        ax.text(0.98, 0.98, f'Glacier: {glacier_pct:.1f}%',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    display_id = catchment_id.replace('catchment_', '')
    plot_file = os.path.join(output_dir, f'water_balance_average_{display_id}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_file}")
    plt.close()


def plot_average_water_balance_subpanels(results_precip, results_streamflow, results_glacier, output_dir, 
                                        results_precip_har=None, results_pet_era5=None, results_pet_har=None):
    """Create a multi-panel plot showing average annual water balance components."""
    catchment_ids = sorted([r['catchment_id'] for r in results_streamflow])
    n_catchments = len(catchment_ids)
    
    fig, axes = plt.subplots(1, n_catchments, figsize=(6*n_catchments, 6))
    if n_catchments == 1:
        axes = [axes]
    
    for idx, catchment_id in enumerate(catchment_ids):
        ax = axes[idx]
        
        precip_data_era5 = next((r for r in results_precip if r['catchment_id'] == catchment_id), None)
        precip_data_har = next((r for r in results_precip_har if r['catchment_id'] == catchment_id), None) if results_precip_har else None
        pet_data_era5 = next((r for r in results_pet_era5 if r['catchment_id'] == catchment_id), None) if results_pet_era5 else None
        pet_data_har = next((r for r in results_pet_har if r['catchment_id'] == catchment_id), None) if results_pet_har else None
        streamflow_data = next((r for r in results_streamflow if r['catchment_id'] == catchment_id), None)
        glacier_id = catchment_id.replace('catchment_', '')
        glacier_data = next((r for r in results_glacier if r['catchment_id'] == glacier_id), None)
        
        components, values, colors = [], [], []
        
        # Streamflow - dark grey
        if streamflow_data:
            components.append('Streamflow')
            values.append(streamflow_data['mean_annual_Q_mm'])
            colors.append('#4d4d4d')  # dark grey
        
        if precip_data_era5 and streamflow_data:
            total_area = streamflow_data['catchment_area_km2']
            glacier_free_area = precip_data_era5['glacier_free_area_km2']
            precip_scaled = precip_data_era5['mean_annual_precip_mm'] * (glacier_free_area / total_area)
            components.append('Precip\n(ERA5)')
            values.append(precip_scaled)
            colors.append('#1f77b4')  # blue
        
        if precip_data_har and streamflow_data:
            total_area = streamflow_data['catchment_area_km2']
            glacier_free_area = precip_data_har['glacier_free_area_km2']
            precip_scaled = precip_data_har['mean_annual_precip_mm'] * (glacier_free_area / total_area)
            components.append('Precip\n(HAR)')
            values.append(precip_scaled)
            colors.append('#87CEEB')  # sky blue (lighter blue)
        
        if pet_data_era5 and streamflow_data:
            total_area = streamflow_data['catchment_area_km2']
            glacier_free_area = pet_data_era5['glacier_free_area_km2']
            pet_scaled = pet_data_era5['mean_annual_pet_mm'] * (glacier_free_area / total_area)
            components.append('PET\n(ERA5)')
            values.append(pet_scaled)
            colors.append('#2ca02c')  # green
        
        if pet_data_har and streamflow_data:
            total_area = streamflow_data['catchment_area_km2']
            glacier_free_area = pet_data_har['glacier_free_area_km2']
            pet_scaled = pet_data_har['mean_annual_pet_mm'] * (glacier_free_area / total_area)
            components.append('PET\n(HAR)')
            values.append(pet_scaled)
            colors.append('#90EE90')  # light green
        
        if glacier_data:
            components.extend(['Glacier\nTotal', 'Ice\nMelt', 'Snow\nMelt'])
            values.extend([glacier_data['mean_annual_total_mm'], 
                          glacier_data['mean_annual_icemelt_mm'],
                          glacier_data['mean_annual_snowmelt_mm']])
            colors.extend(['#9467bd', '#ff7f0e', '#bcbd22'])
        
        x = np.arange(len(components))
        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        display_id = catchment_id.replace('catchment_', '')
        ax.set_title(f'Catchment {display_id}', fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel('Water Depth (mm/year)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(components, fontsize=11, fontweight='bold')
        ax.tick_params(axis='y', labelsize=11)  # Bigger y-axis tick labels
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(values) * 1.15)
        
        if streamflow_data and precip_data_era5:
            glacier_pct = (1 - glacier_free_area / total_area) * 100
            ax.text(0.98, 0.98, f'Glacier: {glacier_pct:.1f}%',
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    fig.suptitle('Mean Annual Water Balance Components by Catchment', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_file = os.path.join(output_dir, 'water_balance_average_all_catchments.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {plot_file}")
    plt.close()


def main():
    """Main execution function."""
    # ============================================================================
    # CONFIGURATION - EDIT THESE PATHS
    # ============================================================================
    base_dir = '/home/jberg@giub.local/Raven_world/02_model_setups_geodetic'
    base_dir_har = '/home/jberg@giub.local/Raven_world/02_model_setups_geodetic_har'
    output_dir = '/home/jberg@giub.local/Papers/Paper_2/Figures/Water_balance'
    
    print("\n" + "="*80)
    print("WATER BALANCE ANALYSIS")
    print("="*80)
    print(f"\nBase directory (ERA5): {base_dir}")
    print(f"Base directory (HAR): {base_dir_har}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    catchment_folders = sorted(glob.glob(os.path.join(base_dir, 'catchment_*')))
    print(f"\nFound {len(catchment_folders)} catchment folders")
    
    # ERA5 Precipitation
    print("\n" + "="*80)
    print("1. CALCULATING ERA5 PRECIPITATION")
    print("="*80)
    out_csv_precip_era5 = os.path.join(output_dir, 'catchment_annual_precipitation_era5.csv')
    
    if os.path.exists(out_csv_precip_era5):
        print(f"  File already exists: {out_csv_precip_era5}")
        print(f"  Loading existing results...")
        df_precip_era5 = pd.read_csv(out_csv_precip_era5)
        results_precip = df_precip_era5.to_dict('records')
        print(f"  Loaded {len(results_precip)} catchment results from file")
    else:
        results_precip = []
        for cf in catchment_folders:
            cid = os.path.basename(cf)
            print(f"\nProcessing {cid}...")
            res = calculate_annual_precipitation(cf)
            if res is not None:
                print(f"  Success: Mean annual precip = {res['mean_annual_precip_mm']:.2f} mm/year")
                results_precip.append(res)
        print(f"\nCompleted: {len(results_precip)}/{len(catchment_folders)} catchments")
        
        # Save ERA5 precipitation results
        if results_precip:
            df_precip_era5 = pd.DataFrame(results_precip)
            df_precip_era5 = df_precip_era5.sort_values('catchment_id').reset_index(drop=True)
            df_precip_era5.to_csv(out_csv_precip_era5, index=False)
            print(f"\nSaved ERA5 precipitation results: {out_csv_precip_era5}")
    
    # HAR Precipitation
    print("\n" + "="*80)
    print("2. CALCULATING HAR PRECIPITATION")
    print("="*80)
    out_csv_precip_har = os.path.join(output_dir, 'catchment_annual_precipitation_har.csv')
    
    if os.path.exists(out_csv_precip_har):
        print(f"  File already exists: {out_csv_precip_har}")
        print(f"  Loading existing results...")
        df_precip_har = pd.read_csv(out_csv_precip_har)
        results_precip_har = df_precip_har.to_dict('records')
        print(f"  Loaded {len(results_precip_har)} catchment results from file")
    else:
        results_precip_har = []
        for cf in catchment_folders:
            cid = os.path.basename(cf)
            print(f"\nProcessing {cid}...")
            res = calculate_annual_precipitation_har(cf, base_dir_har)
            if res is not None:
                print(f"  Success: Mean annual precip = {res['mean_annual_precip_mm']:.2f} mm/year")
                results_precip_har.append(res)
        print(f"\nCompleted: {len(results_precip_har)}/{len(catchment_folders)} catchments")
        
        # Save HAR precipitation results
        if results_precip_har:
            df_precip_har = pd.DataFrame(results_precip_har)
            df_precip_har = df_precip_har.sort_values('catchment_id').reset_index(drop=True)
            df_precip_har.to_csv(out_csv_precip_har, index=False)
            print(f"\nSaved HAR precipitation results: {out_csv_precip_har}")
    
    # ERA5 PET
    print("\n" + "="*80)
    print("3. CALCULATING ERA5 PET")
    print("="*80)
    out_csv_pet_era5 = os.path.join(output_dir, 'catchment_annual_pet_era5.csv')
    
    if os.path.exists(out_csv_pet_era5):
        print(f"  File already exists: {out_csv_pet_era5}")
        print(f"  Loading existing results...")
        df_pet_era5 = pd.read_csv(out_csv_pet_era5)
        results_pet_era5 = df_pet_era5.to_dict('records')
        print(f"  Loaded {len(results_pet_era5)} catchment results from file")
    else:
        results_pet_era5 = []
        for cf in catchment_folders:
            cid = os.path.basename(cf)
            print(f"\nProcessing {cid}...")
            res = calculate_annual_pet_era5(cf)
            if res is not None:
                print(f"  Success: Mean annual PET = {res['mean_annual_pet_mm']:.2f} mm/year")
                results_pet_era5.append(res)
        print(f"\nCompleted: {len(results_pet_era5)}/{len(catchment_folders)} catchments")
        
        # Save ERA5 PET results
        if results_pet_era5:
            df_pet_era5 = pd.DataFrame(results_pet_era5)
            df_pet_era5 = df_pet_era5.sort_values('catchment_id').reset_index(drop=True)
            df_pet_era5.to_csv(out_csv_pet_era5, index=False)
            print(f"\nSaved ERA5 PET results: {out_csv_pet_era5}")
    
    # HAR PET
    print("\n" + "="*80)
    print("4. CALCULATING HAR PET")
    print("="*80)
    out_csv_pet_har = os.path.join(output_dir, 'catchment_annual_pet_har.csv')
    
    if os.path.exists(out_csv_pet_har):
        print(f"  File already exists: {out_csv_pet_har}")
        print(f"  Loading existing results...")
        df_pet_har = pd.read_csv(out_csv_pet_har)
        results_pet_har = df_pet_har.to_dict('records')
        print(f"  Loaded {len(results_pet_har)} catchment results from file")
    else:
        results_pet_har = []
        for cf in catchment_folders:
            cid = os.path.basename(cf)
            print(f"\nProcessing {cid}...")
            res = calculate_annual_pet_har(cf, base_dir_har)
            if res is not None:
                print(f"  Success: Mean annual PET = {res['mean_annual_pet_mm']:.2f} mm/year")
                results_pet_har.append(res)
        print(f"\nCompleted: {len(results_pet_har)}/{len(catchment_folders)} catchments")
        
        # Save HAR PET results
        if results_pet_har:
            df_pet_har = pd.DataFrame(results_pet_har)
            df_pet_har = df_pet_har.sort_values('catchment_id').reset_index(drop=True)
            df_pet_har.to_csv(out_csv_pet_har, index=False)
            print(f"\nSaved HAR PET results: {out_csv_pet_har}")
    
    # Streamflow
    print("\n" + "="*80)
    print("5. CALCULATING STREAMFLOW")
    print("="*80)
    out_csv_streamflow = os.path.join(output_dir, 'catchment_annual_streamflow.csv')
    
    if os.path.exists(out_csv_streamflow):
        print(f"  File already exists: {out_csv_streamflow}")
        print(f"  Loading existing results...")
        df_streamflow = pd.read_csv(out_csv_streamflow)
        results_streamflow = df_streamflow.to_dict('records')
        print(f"  Loaded {len(results_streamflow)} catchment results from file")
    else:
        results_streamflow = []
        for cf in catchment_folders:
            cid = os.path.basename(cf)
            print(f"\nProcessing {cid}...")
            res = calculate_annual_streamflow(cf)
            if res is not None:
                print(f"  Success: Mean annual Q = {res['mean_annual_Q_mm']:.2f} mm/year | Years: {res['num_years_with_data']}")
                results_streamflow.append(res)
        print(f"\nCompleted: {len(results_streamflow)}/{len(catchment_folders)} catchments")
        
        # Save streamflow results
        if results_streamflow:
            df_streamflow = pd.DataFrame(results_streamflow)
            df_streamflow = df_streamflow.sort_values('catchment_id').reset_index(drop=True)
            df_streamflow.to_csv(out_csv_streamflow, index=False)
            print(f"\nSaved streamflow results: {out_csv_streamflow}")
    
    # Glacier Runoff
    print("\n" + "="*80)
    print("6. CALCULATING GLACIER RUNOFF")
    print("="*80)
    out_csv_glacier = os.path.join(output_dir, 'catchment_annual_glacier_runoff.csv')
    
    if os.path.exists(out_csv_glacier):
        print(f"  File already exists: {out_csv_glacier}")
        print(f"  Loading existing results...")
        df_glacier = pd.read_csv(out_csv_glacier)
        # Reconstruct results_glacier from the flattened CSV
        results_glacier = []
        for _, row in df_glacier.iterrows():
            # Extract the basic info
            glacier_result = {
                'catchment_id': row['catchment_id'],
                'glacier_area_km2': row.get('glacier_area_km2', np.nan),
                'total_catchment_area_km2': row.get('total_catchment_area_km2', np.nan),
                'glacier_area_fraction': row.get('glacier_area_fraction', np.nan),
                'mean_annual_icemelt_mm': row['mean_annual_icemelt_mm'],
                'mean_annual_snowmelt_mm': row['mean_annual_snowmelt_mm'],
                'mean_annual_rain_mm': row['mean_annual_rain_mm'],
                'mean_annual_melt_mm': row['mean_annual_melt_mm'],
                'mean_annual_total_mm': row['mean_annual_total_mm'],
                'n_years': int(row['n_years'])
            }
            # Reconstruct annual_data DataFrame from year columns
            annual_data_rows = []
            for col in df_glacier.columns:
                if col.startswith('icemelt_') and col.endswith('_mm'):
                    year = int(col.replace('icemelt_', '').replace('_mm', ''))
                    annual_data_rows.append({
                        'year': year,
                        'annual_icemelt_mm': row[f'icemelt_{year}_mm'],
                        'annual_snowmelt_mm': row[f'snowmelt_{year}_mm'],
                        'annual_rain_mm': row[f'rain_{year}_mm'],
                        'annual_melt_mm': row[f'melt_{year}_mm'],
                        'annual_total_mm': row[f'total_{year}_mm']
                    })
            glacier_result['annual_data'] = pd.DataFrame(annual_data_rows)
            results_glacier.append(glacier_result)
        print(f"  Loaded {len(results_glacier)} catchment results from file")
    else:
        results_glacier = []
        for cf in catchment_folders:
            cid = os.path.basename(cf).replace('catchment_', '')
            print(f"\nProcessing catchment_{cid}...")
            res = calculate_annual_glacier_runoff(cid, base_dir)
            if res is not None:
                print(f"  Success: Mean annual glacier total = {res['mean_annual_total_mm']:.2f} mm/year")
                results_glacier.append(res)
        print(f"\nCompleted: {len(results_glacier)}/{len(catchment_folders)} catchments")
        
        # Save glacier runoff results
        if results_glacier:
            # Flatten the glacier data for CSV export
            glacier_export = []
            for res in results_glacier:
                row = {
                    'catchment_id': res['catchment_id'],
                    'glacier_area_km2': res.get('glacier_area_km2', np.nan),
                    'total_catchment_area_km2': res.get('total_catchment_area_km2', np.nan),
                    'glacier_area_fraction': res.get('glacier_area_fraction', np.nan),
                    'mean_annual_icemelt_mm': res['mean_annual_icemelt_mm'],
                    'mean_annual_snowmelt_mm': res['mean_annual_snowmelt_mm'],
                    'mean_annual_rain_mm': res['mean_annual_rain_mm'],
                    'mean_annual_melt_mm': res['mean_annual_melt_mm'],
                    'mean_annual_total_mm': res['mean_annual_total_mm'],
                    'n_years': res['n_years']
                }
                # Add annual data for each year
                annual_data = res['annual_data']
                for _, year_row in annual_data.iterrows():
                    year = int(year_row['year'])
                    row[f'icemelt_{year}_mm'] = year_row['annual_icemelt_mm']
                    row[f'snowmelt_{year}_mm'] = year_row['annual_snowmelt_mm']
                    row[f'rain_{year}_mm'] = year_row['annual_rain_mm']
                    row[f'melt_{year}_mm'] = year_row['annual_melt_mm']
                    row[f'total_{year}_mm'] = year_row['annual_total_mm']
                glacier_export.append(row)
            
            df_glacier = pd.DataFrame(glacier_export)
            df_glacier = df_glacier.sort_values('catchment_id').reset_index(drop=True)
            df_glacier.to_csv(out_csv_glacier, index=False)
            print(f"\nSaved glacier runoff results: {out_csv_glacier}")
    
    # Generate Plots
    print("\n" + "="*80)
    print("7. GENERATING PLOTS")
    print("="*80)
    
    print("\nGenerating time series plots...")
    for catchment_id in sorted(set([r['catchment_id'] for r in results_streamflow])):
        print(f"\nPlotting {catchment_id}...")
        plot_catchment_water_balance(catchment_id, results_precip, results_streamflow, results_glacier, 
                                     output_dir, results_precip_har, results_pet_era5, results_pet_har)
    
    print("\nGenerating individual average water balance plots...")
    for catchment_id in sorted(set([r['catchment_id'] for r in results_streamflow])):
        print(f"\nPlotting average for {catchment_id}...")
        plot_average_water_balance_single(catchment_id, results_precip, results_streamflow, results_glacier,
                                         output_dir, results_precip_har, results_pet_era5, results_pet_har)
    
    print("\nGenerating multi-panel average plot...")
    plot_average_water_balance_subpanels(results_precip, results_streamflow, results_glacier, output_dir,
                                        results_precip_har, results_pet_era5, results_pet_har)
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nCSV files generated:")
    print(f"  - catchment_annual_precipitation_era5.csv")
    print(f"  - catchment_annual_precipitation_har.csv")
    print(f"  - catchment_annual_pet_era5.csv")
    print(f"  - catchment_annual_pet_har.csv")
    print(f"  - catchment_annual_streamflow.csv")
    print(f"  - catchment_annual_glacier_runoff.csv")
    print(f"\nPlot files generated:")
    print(f"  - catchment_XXXX_water_balance.png (time series plots)")
    print(f"  - water_balance_average_XXXX.png (individual average plots)")
    print(f"  - water_balance_average_all_catchments.png (multi-panel summary)")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
