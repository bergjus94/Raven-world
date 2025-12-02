"""
Create overview table for all delineated catchments with glacier coverage
Processes multiple country shapefiles and calculates relative glacier area using RGI data
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import numpy as np
from shapely.ops import unary_union

def load_all_catchments(catchment_dir):
    """
    Load all catchment shapefiles from directory
    
    Parameters:
    -----------
    catchment_dir : str or Path
        Directory containing all_catchment_*.shp files
        
    Returns:
    --------
    GeoDataFrame
        Combined catchments from all countries
    """
    catchment_dir = Path(catchment_dir)
    
    print("🗺️  Loading catchment shapefiles...")
    
    # Find all catchment shapefiles
    catchment_files = list(catchment_dir.glob("all_catchment_*.shp"))
    
    if not catchment_files:
        raise FileNotFoundError(f"No catchment shapefiles found in {catchment_dir}")
    
    print(f"   Found {len(catchment_files)} catchment files:")
    for f in catchment_files:
        print(f"   - {f.name}")
    
    # Load and combine all catchments
    catchments_list = []
    
    for shp_file in catchment_files:
        try:
            gdf = gpd.read_file(shp_file)
            
            # Extract country name from filename
            country = shp_file.stem.replace("all_catchment_", "")
            gdf['country'] = country
            
            catchments_list.append(gdf)
            print(f"   ✅ Loaded {len(gdf)} catchments from {country}")
            
        except Exception as e:
            print(f"   ❌ Error loading {shp_file.name}: {e}")
    
    if not catchments_list:
        raise ValueError("No catchments were successfully loaded")
    
    # Combine all catchments
    all_catchments = pd.concat(catchments_list, ignore_index=True)
    
    print(f"\n✅ Total catchments loaded: {len(all_catchments)}")
    
    return all_catchments


def load_glacier_data(rgi_dirs):
    """
    Load RGI glacier shapefiles
    
    Parameters:
    -----------
    rgi_dirs : list of tuples
        List of (directory, shapefile_name) for each RGI region
        
    Returns:
    --------
    GeoDataFrame
        Combined glacier outlines
    """
    print("\n🏔️  Loading RGI glacier data...")
    
    glacier_list = []
    
    for rgi_dir, shp_name in rgi_dirs:
        rgi_path = Path(rgi_dir) / shp_name
        
        if not rgi_path.exists():
            print(f"   ⚠️  Glacier file not found: {rgi_path}")
            continue
        
        try:
            glaciers = gpd.read_file(rgi_path)
            glacier_list.append(glaciers)
            print(f"   ✅ Loaded {len(glaciers)} glaciers from {shp_name}")
            
        except Exception as e:
            print(f"   ❌ Error loading {shp_name}: {e}")
    
    if not glacier_list:
        raise ValueError("No glacier data was successfully loaded")
    
    # Combine all glacier regions
    all_glaciers = pd.concat(glacier_list, ignore_index=True)
    
    print(f"\n✅ Total glaciers loaded: {len(all_glaciers)}")
    
    return all_glaciers


def calculate_glacier_coverage(catchments_gdf, glaciers_gdf):
    """
    Calculate relative glacier area for each catchment
    
    Parameters:
    -----------
    catchments_gdf : GeoDataFrame
        Catchment polygons
    glaciers_gdf : GeoDataFrame
        Glacier polygons
        
    Returns:
    --------
    pandas.Series
        Relative glacier area (0-1) for each catchment
    """
    print("\n❄️  Calculating glacier coverage for each catchment...")
    
    # Ensure same CRS
    if catchments_gdf.crs != glaciers_gdf.crs:
        print(f"   Reprojecting glaciers from {glaciers_gdf.crs} to {catchments_gdf.crs}")
        glaciers_gdf = glaciers_gdf.to_crs(catchments_gdf.crs)
    
    # Calculate glacier coverage for each catchment
    glacier_areas = []
    
    for idx, catchment in catchments_gdf.iterrows():
        try:
            # Find glaciers that intersect this catchment
            intersecting = glaciers_gdf[glaciers_gdf.intersects(catchment.geometry)]
            
            if len(intersecting) == 0:
                glacier_areas.append(0.0)
            else:
                # Calculate total glacier area within catchment
                glacier_area = 0.0
                
                for _, glacier in intersecting.iterrows():
                    intersection = glacier.geometry.intersection(catchment.geometry)
                    glacier_area += intersection.area
                
                # Calculate relative glacier area
                catchment_area = catchment.geometry.area
                relative_glacier_area = glacier_area / catchment_area if catchment_area > 0 else 0.0
                
                glacier_areas.append(relative_glacier_area)
            
            # Progress indicator
            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx + 1}/{len(catchments_gdf)} catchments...", end='\r')
                
        except Exception as e:
            print(f"\n   ⚠️  Error processing catchment {idx}: {e}")
            glacier_areas.append(np.nan)
    
    print(f"\n   ✅ Calculated glacier coverage for {len(glacier_areas)} catchments")
    
    return pd.Series(glacier_areas, index=catchments_gdf.index)


def identify_main_basins(catchments_gdf):
    """
    Identify the main basin for each catchment based on nesting relationships.
    If catchments overlap, the main basin is the station_id of the largest catchment.
    If standalone, the main basin is its own station_id.
    
    Parameters:
    -----------
    catchments_gdf : GeoDataFrame
        Catchment polygons with station_id and area columns
        
    Returns:
    --------
    pandas.Series
        Main basin station_id for each catchment
    """
    print("\n🗺️  Identifying main basins for nested catchments...")
    
    main_basins = {}
    
    # Sort catchments by area (largest first)
    sorted_indices = catchments_gdf.sort_values('area', ascending=False).index
    
    for idx in sorted_indices:
        catchment = catchments_gdf.loc[idx]
        station_id = catchment['station_id']
        catchment_geom = catchment.geometry
        
        # Check if this catchment is contained within any larger catchment
        parent_basin = None
        max_overlap_ratio = 0.0
        
        for other_idx in sorted_indices:
            if other_idx == idx:
                continue
            
            other_catchment = catchments_gdf.loc[other_idx]
            other_station_id = other_catchment['station_id']
            other_geom = other_catchment.geometry
            
            # Only check if other catchment is larger
            if other_catchment['area'] <= catchment['area']:
                continue
            
            try:
                # Check overlap
                intersection = catchment_geom.intersection(other_geom)
                overlap_ratio = intersection.area / catchment_geom.area
                
                # If mostly contained (>95% overlap), this is a nested catchment
                if overlap_ratio > 0.95 and overlap_ratio > max_overlap_ratio:
                    parent_basin = other_station_id
                    max_overlap_ratio = overlap_ratio
                    
            except Exception as e:
                print(f"   ⚠️ Error checking overlap between {station_id} and {other_station_id}: {e}")
        
        # Set main basin
        if parent_basin is not None:
            main_basins[idx] = parent_basin
            print(f"   🔗 {station_id} is nested in main basin {parent_basin} (overlap: {max_overlap_ratio*100:.1f}%)")
        else:
            main_basins[idx] = station_id  # Standalone - its own main basin
    
    print(f"   ✅ Identified main basins for {len(main_basins)} catchments")
    
    # Convert to Series with same index as input
    return pd.Series(main_basins)


def create_catchment_table(catchments_gdf, output_path):
    """
    Create final overview table with selected columns
    
    Parameters:
    -----------
    catchments_gdf : GeoDataFrame
        Catchments with all attributes
    output_path : str or Path
        Path to save CSV file
        
    Returns:
    --------
    DataFrame
        Final catchment overview table
    """
    print("\n📋 Creating catchment overview table...")
    
    # Select and rename columns
    column_mapping = {
        'station_id': 'station_id',
        'river_name': 'river_name',
        'station__1': 'station_name',  # Rename station__1 to station_name
        'lat': 'lat',
        'lon': 'lon',
        'elevation': 'elevation',
        'area': 'area',
        'country': 'country',
        'from': 'from',
        'to': 'to',
        'glacier_area': 'relative_glacier_area',
        'glacier_area_pct': 'glacier_area_percent',
        'main_basin': 'main_basin'
    }
    
    # Check which columns exist
    available_columns = {}
    for old_name, new_name in column_mapping.items():
        if old_name in catchments_gdf.columns:
            available_columns[old_name] = new_name
        else:
            print(f"   ⚠️  Column '{old_name}' not found in data")
    
    # Create final table
    table = catchments_gdf[list(available_columns.keys())].copy()
    table.rename(columns=available_columns, inplace=True)
    
    # Reorder columns to put main_basin after station_id
    if 'main_basin' in table.columns and 'station_id' in table.columns:
        cols = list(table.columns)
        cols.remove('main_basin')
        station_id_idx = cols.index('station_id')
        cols.insert(station_id_idx + 1, 'main_basin')
        table = table[cols]
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    table.to_csv(output_path, index=False)
    
    print(f"✅ Saved catchment table to: {output_path}")
    print(f"   Rows: {len(table)}")
    print(f"   Columns: {list(table.columns)}")
    
    return table


def print_summary_statistics(table):
    """Print summary statistics of the catchment table"""
    print("\n" + "="*80)
    print("CATCHMENT TABLE SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal catchments: {len(table)}")
    
    if 'country' in table.columns:
        print(f"\nCatchments by country:")
        country_counts = table['country'].value_counts()
        for country, count in country_counts.items():
            print(f"  {country}: {count}")
    
    if 'main_basin' in table.columns:
        print(f"\nMain basin distribution:")
        # Count how many unique main basins
        unique_basins = table['main_basin'].nunique()
        print(f"  Total main basins: {unique_basins}")
        
        # Count standalone vs nested
        standalone = (table['station_id'] == table['main_basin']).sum()
        nested = (table['station_id'] != table['main_basin']).sum()
        print(f"  Standalone catchments: {standalone}")
        print(f"  Nested catchments: {nested}")
        
        # Show basins with most subcatchments
        basin_counts = table['main_basin'].value_counts()
        print(f"\n  Basins with most subcatchments:")
        for basin_id, count in basin_counts.head(5).items():
            if count > 1:
                print(f"    {basin_id}: {count} catchments")
    
    if 'area' in table.columns:
        print(f"\nCatchment area statistics (km²):")
        print(f"  Min: {table['area'].min():.2f}")
        print(f"  Max: {table['area'].max():.2f}")
        print(f"  Mean: {table['area'].mean():.2f}")
        print(f"  Median: {table['area'].median():.2f}")
    
    if 'glacier_area_percent' in table.columns:
        print(f"\nGlacier coverage statistics:")
        glacier_coverage = table['glacier_area_percent']
        print(f"  Catchments with glaciers: {(glacier_coverage > 0).sum()}")
        print(f"  Mean coverage: {glacier_coverage.mean():.2f}%")
        print(f"  Max coverage: {glacier_coverage.max():.2f}%")
        print(f"  Catchments with >10% glacier: {(glacier_coverage > 10).sum()}")
        print(f"  Catchments with >50% glacier: {(glacier_coverage > 50).sum()}")
    
    if 'elevation' in table.columns:
        print(f"\nElevation statistics (m):")
        print(f"  Min: {table['elevation'].min():.0f}")
        print(f"  Max: {table['elevation'].max():.0f}")
        print(f"  Mean: {table['elevation'].mean():.0f}")
    
    print("\n" + "="*80)


def main():
    """Main execution function"""
    
    print("="*80)
    print("CATCHMENT OVERVIEW TABLE GENERATOR")
    print("="*80)
    
    # Configuration - Update these paths for your server
    catchment_dir = "/home/jberg@giub.local/Catchment_delineation/gauging_stations/catchment_delineation_data"
    
    # RGI glacier data directories
    rgi_regions = [
        (
            "/home/jberg@giub.local/Raven_world/01_data/topo/glacier_outline/nsidc0770_14.rgi60.SouthAsiaWest",
            "14_rgi60_SouthAsiaWest.shp"
        ),
        (
            "/home/jberg@giub.local/Raven_world/01_data/topo/glacier_outline/nsidc0770_15.rgi60.SouthAsiaEast",
            "15_rgi60_SouthAsiaEast.shp"
        )
    ]
    
    output_csv = "/home/jberg@giub.local/Catchment_delineation/catchment_overview_table.csv"
    
    try:
        # Step 1: Load all catchment shapefiles
        catchments = load_all_catchments(catchment_dir)
        
        # Step 2: Load glacier data
        glaciers = load_glacier_data(rgi_regions)
        
        # Step 3: Calculate glacier coverage (as fraction 0-1)
        catchments['glacier_area'] = calculate_glacier_coverage(catchments, glaciers)
        
        # Step 4: Convert glacier coverage to percentage
        catchments['glacier_area_pct'] = catchments['glacier_area'] * 100
        
        # Step 5: Identify main basins
        catchments['main_basin'] = identify_main_basins(catchments)
        
        # Step 6: Create final table
        table = create_catchment_table(catchments, output_csv)
        
        # Step 7: Print summary statistics
        print_summary_statistics(table)
        
        print("\n🎉 SUCCESS! Catchment overview table created.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()