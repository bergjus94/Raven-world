"""Derive per-catchment MODIS basin-mean fSCA from region NetCDFs.

Reads the region NetCDF (produced by scripts/build_modis_region.py), clips
to the catchment shape, optionally masks out glacier pixels using the
catchment's regional RGI glacier outlines, then computes a basin-mean fSCA
per MODIS timestep and writes a CSV consumable by the snow calibration
objective:

    <main_dir>/01_data/snow/MODIS/basins/<gauge>/fsca_<product>_<gauge>.csv

Output schema matches the existing fsca_MOD10A2_0101.csv:
    date, fsca, n_valid, n_cloud, n_total

Idempotent: re-running overwrites the CSV from the source NetCDFs.

Usage
-----
    python scripts/derive_basin_fsca.py 2268
    python scripts/derive_basin_fsca.py 2268 --no-glacier-mask
    python scripts/derive_basin_fsca.py 2268 --product MOD10A2 --years 2000-2020
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from config_merge import load_config  # noqa: E402


# MOD10A2 Maximum_Snow_Extent byte codes
MOD10A2_SNOW    = 200
MOD10A2_NO_SNOW = (25, 37)   # 25=land no-snow, 37=lake no-snow
MOD10A2_CLOUD   = 50


# ---------------------------------------------------------------------------

def discover_year_files(region_dir: Path, product: str,
                       years: Optional[List[int]] = None) -> List[Path]:
    files = sorted(region_dir.glob(f'{product}_*.nc'))
    if years is None:
        return files
    keep = []
    for f in files:
        try:
            y = int(f.stem.split('_')[-1])
        except ValueError:
            continue
        if y in years:
            keep.append(f)
    return keep


def open_region_dataset(year_files: List[Path]) -> xr.DataArray:
    """Concatenate per-year MODIS region NetCDFs along time."""
    parts = [xr.open_dataset(f)['snow_extent'] for f in year_files]
    if len(parts) == 1:
        return parts[0]
    return xr.concat(parts, dim='time')


def clip_to_catchment(
    da: xr.DataArray,
    catchment_shp: Path,
    glacier_shp: Optional[Path] = None,
) -> xr.DataArray:
    """Clip the region DataArray to the catchment polygon, optionally
    masking out glacier-covered pixels.

    The result keeps the original grid but sets off-catchment / on-glacier
    cells to NaN — preserves the spatial structure for downstream counting.
    """
    import geopandas as gpd
    import rioxarray  # noqa: F401  (registers .rio)

    if da.rio.crs is None:
        # Region NetCDFs are written in EPSG:4326 by build_modis_region
        da = da.rio.write_crs('EPSG:4326')

    catch_gdf = gpd.read_file(catchment_shp).to_crs(da.rio.crs)
    clipped = da.rio.clip(catch_gdf.geometry, drop=True)

    if glacier_shp is not None and Path(glacier_shp).exists():
        try:
            glacier_gdf = gpd.read_file(glacier_shp).to_crs(da.rio.crs)
            # Spatial join: keep only glacier polygons intersecting the
            # catchment bounding box (huge speedup vs running clip on a
            # continent-scale RGI shapefile).
            minx, miny, maxx, maxy = catch_gdf.total_bounds
            glacier_clip = glacier_gdf.cx[minx:maxx, miny:maxy]
            if len(glacier_clip) > 0:
                # `invert=True` keeps the OUTSIDE-glacier pixels
                clipped = clipped.rio.clip(glacier_clip.geometry, invert=True,
                                           drop=False)
        except Exception as e:
            print(f"  ⚠️ Glacier masking failed ({e}); falling back to "
                  f"full-catchment mean.")

    return clipped


def basin_mean_per_timestep(da: xr.DataArray) -> pd.DataFrame:
    """Count snow / no-snow / cloud / invalid pixels per time step and
    compute basin-mean fSCA (= n_snow / (n_snow + n_no_snow))."""
    # Mask of pixels inside the catchment (regardless of MODIS class)
    inside = ~da.isnull()
    snow    = (da == MOD10A2_SNOW)
    no_snow = sum((da == v) for v in MOD10A2_NO_SNOW)
    cloud   = (da == MOD10A2_CLOUD)

    n_total = inside.sum(dim=('y', 'x')).astype('int64')
    n_snow  = snow.sum(dim=('y', 'x')).astype('int64')
    n_nosno = no_snow.sum(dim=('y', 'x')).astype('int64')
    n_cloud = cloud.sum(dim=('y', 'x')).astype('int64')

    n_valid = (n_snow + n_nosno).astype('int64')

    # fsca: NaN when n_valid == 0 (all-cloud / off-catchment)
    fsca = (n_snow / xr.where(n_valid > 0, n_valid, 1)).astype('float64')
    fsca = xr.where(n_valid > 0, fsca, np.nan)

    df = pd.DataFrame({
        'date':    pd.to_datetime(da['time'].values).strftime('%Y-%m-%d'),
        'fsca':    fsca.values,
        'n_valid': n_valid.values,
        'n_cloud': n_cloud.values,
        'n_total': n_total.values,
    })
    return df


def resolve_region(nml: dict, fallback_gauge_prefix: dict) -> Optional[str]:
    """Determine region for a catchment.

    Precedence: namelist `region:` key > gauge_id prefix lookup.
    """
    if 'region' in nml and nml['region']:
        r = str(nml['region']).strip()
        # Capitalize for filesystem (switzerland → Switzerland)
        return r.capitalize() if r.islower() else r
    gauge = str(nml['gauge_id'])
    for prefix, region in fallback_gauge_prefix.items():
        if gauge.startswith(prefix):
            return region
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_GAUGE_PREFIX_REGION = {
    '0':  'Indus',
    '2':  'Switzerland',
    # Add Ganges/etc. as needed
}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gauge_id', help='Catchment gauge ID (e.g. 2268)')
    ap.add_argument('--product', default='MOD10A2',
                    choices=['MOD10A1', 'MYD10A1', 'MOD10A2', 'MYD10A2'])
    ap.add_argument('--years', default=None,
                    help="Year range '2000-2020' to subset; default: all available")
    ap.add_argument('--region', default=None,
                    help='Region override (default: from namelist `region:` key)')
    ap.add_argument('--main-dir', default=None,
                    help='Override main_dir (default: env layer autodetect)')
    ap.add_argument('--env', default=None,
                    help='Env layer override (server/local). Default: autodetect')
    ap.add_argument('--no-glacier-mask', action='store_true',
                    help='Skip the glacier-pixel exclusion step.')
    ap.add_argument('--configuration', default=None,
                    help='Configuration key for the namelist merge. Only '
                         'affects which `glacier_dir` is resolved. Any '
                         'compatible config works; defaults to the first '
                         'listed under `configurations:`.')
    args = ap.parse_args(argv)

    # ---- 1. Load namelist ----
    cat_layer_path = ROOT / 'src' / 'config' / 'layers' / 'catchments' / f'{args.gauge_id}.yaml'
    if not cat_layer_path.exists():
        print(f"❌ No catchment config at {cat_layer_path}")
        return 1

    # Pick *any* configuration the catchment supports — we only need main_dir,
    # region, catchment shape, and glacier outline, all of which are
    # config-agnostic.
    import yaml
    with open(cat_layer_path) as f:
        cat_layer = yaml.safe_load(f) or {}
    config_key = args.configuration or 'glogem_subdaily_opt1'

    # load_config returns (merged_dict, temp_yaml_path) — keep just the dict
    nml, _tmp = load_config(
        catchment=args.gauge_id, configuration=config_key,
        model='SPHY', env=args.env,
    )
    if args.main_dir:
        nml['main_dir'] = args.main_dir

    main_dir = Path(nml['main_dir'])
    region = args.region or resolve_region(nml, DEFAULT_GAUGE_PREFIX_REGION)
    if region is None:
        print(f"❌ Could not determine region for {args.gauge_id}. "
              f"Pass --region.")
        return 1

    # ---- 2. Locate region NetCDFs ----
    region_dir = main_dir / '01_data' / 'snow' / 'MODIS' / 'regions' / region
    if not region_dir.exists():
        print(f"❌ No region NetCDF dir: {region_dir}\n"
              f"   Run: python scripts/build_modis_region.py {region}")
        return 1

    years = None
    if args.years:
        from scripts.build_modis_region import parse_years  # type: ignore
        years = parse_years(args.years)
    year_files = discover_year_files(region_dir, args.product, years)
    if not year_files:
        print(f"❌ No {args.product}_*.nc files in {region_dir}")
        return 1

    print(f"Gauge:       {args.gauge_id}")
    print(f"Region:      {region}")
    print(f"NetCDFs:     {len(year_files)} ({year_files[0].name} … {year_files[-1].name})")

    # ---- 3. Resolve shapefiles ----
    shape_template = nml.get('shape_dir',
                             '01_data/topo/catchment_shapefile/'
                             'catchment_shape_{gauge_id}.shp')
    catchment_shp = main_dir / shape_template.format(gauge_id=args.gauge_id)
    if not catchment_shp.exists():
        print(f"❌ Catchment shape not found: {catchment_shp}")
        return 1

    glacier_shp: Optional[Path] = None
    if not args.no_glacier_mask:
        gl = nml.get('glacier_dir', '')
        if gl:
            glacier_shp = main_dir / gl
            if not glacier_shp.exists():
                print(f"  ⚠️ Glacier shapefile not found ({glacier_shp}); "
                      f"running without glacier mask.")
                glacier_shp = None

    print(f"Catchment:   {catchment_shp.name}")
    print(f"Glacier:     {glacier_shp.name if glacier_shp else '(none — full-basin mean)'}")

    # ---- 4. Load NetCDFs, clip, derive ----
    print(f"\nOpening {len(year_files)} region NetCDF(s)...")
    da = open_region_dataset(year_files)
    print(f"  → shape: {dict(da.sizes)}")

    print("Clipping to catchment...")
    clipped = clip_to_catchment(da, catchment_shp, glacier_shp)
    print(f"  → shape: {dict(clipped.sizes)}")

    print("Computing per-timestep basin mean...")
    df = basin_mean_per_timestep(clipped)
    print(f"  → {len(df)} timesteps")

    # ---- 5. Write CSV ----
    out_dir = main_dir / '01_data' / 'snow' / 'MODIS' / 'basins' / args.gauge_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'fsca_{args.product}_{args.gauge_id}.csv'
    df.to_csv(out_path, index=False, float_format='%.6f')
    print(f"\n💾 Wrote {out_path}  ({len(df)} dates)")
    print(f"   fsca range: {df['fsca'].min():.3f}–{df['fsca'].max():.3f}, "
          f"mean cloud frac: {(df['n_cloud'] / df['n_total']).mean():.2f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
