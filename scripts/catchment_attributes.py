"""
Catchment attribute table from raw shapefiles, DEM, and RGI glacier outlines.

Computes for each catchment:
  - Area (km2)
  - Gauge lat/lon/elevation
  - Mean/min/max elevation from DEM
  - Mean slope from DEM
  - Glacier area (km2) and fraction from RGI
  - Number of glaciers
  - Data period (from config layers)

Usage:
    python scripts/catchment_attributes.py
    python scripts/catchment_attributes.py --catchments 0102 0122
    python scripts/catchment_attributes.py -o outputs/catchment_table.csv
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask

# Add src to path for config imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_catchment_shapefile(main_dir, gauge_id):
    """Load catchment boundary shapefile."""
    shape_path = main_dir / '01_data' / 'topo' / 'catchment_shapefile' / f'catchment_shape_{gauge_id}.shp'
    if not shape_path.exists():
        return None
    return gpd.read_file(shape_path)


def compute_dem_stats(main_dir, catchment_gdf):
    """Extract elevation and slope statistics from DEM for catchment."""
    dem_path = main_dir / '01_data' / 'topo' / 'catchment_dem' / 'dem_Indus.tif'
    if not dem_path.exists():
        return {}

    catchment_proj = catchment_gdf.to_crs('EPSG:4326')

    with rasterio.open(dem_path) as src:
        try:
            out_image, out_transform = rio_mask(src, catchment_proj.geometry, crop=True, nodata=src.nodata)
            elev = out_image[0]
            valid = elev != src.nodata
            if valid.sum() == 0:
                return {}

            elev_valid = elev[valid].astype(float)

            # Compute slope from DEM (degrees)
            dy, dx = np.gradient(elev.astype(float), out_transform[4], out_transform[0])
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            slope_deg = np.degrees(slope_rad)
            slope_valid = slope_deg[valid]

            return {
                'mean_elevation_m': float(np.mean(elev_valid)),
                'min_elevation_m': float(np.min(elev_valid)),
                'max_elevation_m': float(np.max(elev_valid)),
                'median_elevation_m': float(np.median(elev_valid)),
                'mean_slope_deg': float(np.mean(slope_valid)),
            }
        except Exception as e:
            print(f"  DEM extraction failed: {e}")
            return {}


def compute_glacier_stats(main_dir, catchment_gdf):
    """Compute glacier coverage from RGI outlines."""
    rgi_dirs = [
        main_dir / '01_data' / 'topo' / 'glacier_outlines' / 'nsidc0770_14.rgi60.SouthAsiaWest' / '14_rgi60_SouthAsiaWest.shp',
        main_dir / '01_data' / 'topo' / 'glacier_outlines' / 'nsidc0770_15.rgi60.SouthAsiaEast' / '15_rgi60_SouthAsiaEast.shp',
    ]

    glaciers_list = []
    for rgi_path in rgi_dirs:
        if rgi_path.exists():
            glaciers_list.append(gpd.read_file(rgi_path))

    if not glaciers_list:
        return {}

    glaciers = pd.concat(glaciers_list, ignore_index=True)
    glaciers = glaciers.to_crs(catchment_gdf.crs)

    # Project to UTM for area calculation
    utm_crs = catchment_gdf.estimate_utm_crs()
    catchment_utm = catchment_gdf.to_crs(utm_crs)
    glaciers_utm = glaciers.to_crs(utm_crs)

    catchment_geom = catchment_utm.geometry.iloc[0]
    catchment_area_km2 = catchment_geom.area / 1e6

    # Find intersecting glaciers
    intersecting = glaciers_utm[glaciers_utm.intersects(catchment_geom)]

    if len(intersecting) == 0:
        return {
            'catchment_area_km2': catchment_area_km2,
            'glacier_area_km2': 0.0,
            'glacier_fraction': 0.0,
            'n_glaciers': 0,
        }

    # Clip glaciers to catchment and compute area
    glacier_area_km2 = 0.0
    for _, glacier in intersecting.iterrows():
        clipped = glacier.geometry.intersection(catchment_geom)
        glacier_area_km2 += clipped.area / 1e6

    return {
        'catchment_area_km2': catchment_area_km2,
        'glacier_area_km2': glacier_area_km2,
        'glacier_fraction': glacier_area_km2 / catchment_area_km2 if catchment_area_km2 > 0 else 0,
        'n_glaciers': len(intersecting),
    }


def get_catchment_info(gauge_id):
    """Get display name and dates from config layers."""
    from config_merge import load_catchments_registry
    registry = load_catchments_registry()
    for c in registry:
        if str(c['gauge_id']) == str(gauge_id):
            return c
    return {}


def main():
    parser = argparse.ArgumentParser(description='Compute catchment attributes from raw data')
    parser.add_argument('--catchments', nargs='+', default=None,
                        help='Gauge IDs to process (default: all from config layers)')
    parser.add_argument('--shapefile', type=str, default=None,
                        help='Combined catchment shapefile (processes all catchments in file)')
    parser.add_argument('--id-column', type=str, default='station_id',
                        help='Column name for gauge ID in combined shapefile (default: station_id)')
    parser.add_argument('--main-dir', type=str, default=None,
                        help='Main data directory (auto-detected from config layers if not set)')
    parser.add_argument('--env', type=str, default=None,
                        help='Environment: "local" or "server" (auto-detected)')
    parser.add_argument('-o', '--output', type=str, default='outputs/catchment_attributes.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    # Resolve main_dir from config layers if not specified
    if args.main_dir:
        main_dir = Path(args.main_dir)
    else:
        from config_merge import load_config
        # Load any catchment config to get main_dir from env layer
        nml, tmp = load_config('0122', 'baseline', 'HBV', env=args.env)
        main_dir = Path(nml['main_dir'])
        Path(tmp).unlink(missing_ok=True)

    # Load catchments: from combined shapefile or individual files
    if args.shapefile:
        combined_gdf = gpd.read_file(args.shapefile)
        id_col = args.id_column
        if id_col not in combined_gdf.columns:
            print(f"Error: column '{id_col}' not found in {args.shapefile}")
            print(f"Available columns: {list(combined_gdf.columns)}")
            sys.exit(1)
        gauge_ids = [str(gid) for gid in combined_gdf[id_col].tolist()]
        if args.catchments:
            gauge_ids = [gid for gid in gauge_ids if gid in args.catchments]
        catchment_gdfs = {
            str(row[id_col]): combined_gdf.iloc[[idx]]
            for idx, row in combined_gdf.iterrows()
            if str(row[id_col]) in gauge_ids
        }
        print(f"Loaded {len(catchment_gdfs)} catchments from {args.shapefile}")
    else:
        if args.catchments:
            gauge_ids = args.catchments
        else:
            from config_merge import list_catchments
            gauge_ids = list_catchments()
        catchment_gdfs = {}

    print("=" * 70)
    print("CATCHMENT ATTRIBUTE TABLE")
    print("=" * 70)
    print(f"Catchments: {gauge_ids}")
    print(f"Data dir: {main_dir}")

    rows = []
    for gauge_id in gauge_ids:
        print(f"\n--- Catchment {gauge_id} ---")

        # Load shapefile (from combined dict or individual file)
        if gauge_id in catchment_gdfs:
            catchment_gdf = catchment_gdfs[gauge_id]
        else:
            catchment_gdf = load_catchment_shapefile(main_dir, gauge_id)
        if catchment_gdf is None:
            print(f"  Shapefile not found, skipping")
            continue

        # Get centroid for lat/lon
        centroid = catchment_gdf.to_crs('EPSG:4326').geometry.iloc[0].centroid

        row = {
            'gauge_id': gauge_id,
            'lat': round(centroid.y, 4),
            'lon': round(centroid.x, 4),
        }

        # Config layer info
        info = get_catchment_info(gauge_id)
        if info:
            display_name = info.get('display_name', '')
            if ' @ ' in display_name:
                row['river'], row['gauging_station'] = display_name.split(' @ ', 1)
            else:
                row['river'] = display_name
                row['gauging_station'] = ''
            row['validation_start'] = info.get('validation_start', '')
            row['validation_end'] = info.get('validation_end', '')

        # DEM stats
        print(f"  Computing DEM statistics...")
        dem_stats = compute_dem_stats(main_dir, catchment_gdf)
        row.update(dem_stats)

        # Glacier stats
        print(f"  Computing glacier coverage...")
        glacier_stats = compute_glacier_stats(main_dir, catchment_gdf)
        row.update(glacier_stats)

        if glacier_stats:
            print(f"  Area: {glacier_stats.get('catchment_area_km2', 0):.1f} km2")
            print(f"  Glacier: {glacier_stats.get('glacier_area_km2', 0):.1f} km2 ({glacier_stats.get('glacier_fraction', 0)*100:.1f}%)")
            print(f"  Glaciers: {glacier_stats.get('n_glaciers', 0)}")

        if dem_stats:
            print(f"  Elevation: {dem_stats.get('min_elevation_m', 0):.0f} - {dem_stats.get('max_elevation_m', 0):.0f} m (mean {dem_stats.get('mean_elevation_m', 0):.0f} m)")
            print(f"  Mean slope: {dem_stats.get('mean_slope_deg', 0):.1f} deg")

        rows.append(row)

    # Nesting detection: find main basin for each catchment
    if len(rows) > 1:
        print(f"\n--- Detecting nesting relationships ---")
        # Load all geometries in UTM for area comparison
        all_geoms = {}
        for row in rows:
            gdf = load_catchment_shapefile(main_dir, row['gauge_id'])
            if gdf is not None:
                utm_crs = gdf.estimate_utm_crs()
                all_geoms[row['gauge_id']] = gdf.to_crs(utm_crs).geometry.iloc[0]

        # Sort by area (largest first)
        sorted_ids = sorted(all_geoms.keys(),
                            key=lambda gid: all_geoms[gid].area, reverse=True)

        for row in rows:
            gid = row['gauge_id']
            geom = all_geoms.get(gid)
            if geom is None:
                row['main_basin'] = gid
                continue

            parent = None
            for other_id in sorted_ids:
                if other_id == gid:
                    continue
                other_geom = all_geoms[other_id]
                if other_geom.area <= geom.area:
                    continue
                try:
                    # Reproject to same CRS for comparison
                    overlap = geom.intersection(other_geom).area / geom.area
                    if overlap > 0.95:
                        parent = other_id
                        break
                except Exception:
                    pass

            row['main_basin'] = parent if parent else gid
            if parent:
                print(f"  {gid} nested in {parent}")
            else:
                print(f"  {gid} standalone")

    # Create table
    df = pd.DataFrame(rows)

    # Reorder columns
    col_order = ['gauge_id', 'river', 'gauging_station', 'main_basin', 'lat', 'lon',
                 'catchment_area_km2', 'mean_elevation_m', 'min_elevation_m',
                 'max_elevation_m', 'median_elevation_m', 'mean_slope_deg',
                 'glacier_area_km2', 'glacier_fraction', 'n_glaciers',
                 'validation_start', 'validation_end']
    df = df[[c for c in col_order if c in df.columns]]

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*70}")
    print(f"Saved: {output_path}")
    print(f"{'='*70}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
