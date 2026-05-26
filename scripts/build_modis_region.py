"""Build region-wide MODIS fSCA NetCDFs.

Pipeline per year:

    1. Read the region bbox from <main_dir>/<dem_<region>.tif> (or override).
    2. Download missing HDF tiles into <main_dir>/01_data/snow/MODIS/tiles/
       (shared cache, keyed by (product, tile_id)).
    3. For each acquisition date, mosaic the available tiles and clip to the
       region bbox.
    4. Stack all dates into a single annual DataArray and write
       <main_dir>/01_data/snow/MODIS/regions/<region>/<product>_<year>.nc.

The NetCDFs hold the raw MOD10A2 byte values (uint8) so we can re-derive fSCA
later with whatever cloud-masking / categorical thresholds we choose — without
re-downloading anything.

Usage
-----
    python scripts/build_modis_region.py Switzerland --years 2000-2025
    python scripts/build_modis_region.py Indus --years 2000-2014 --product MOD10A2

The output is idempotent: years whose NetCDF already exists are skipped
unless --force is passed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'downloads'))

import download_MODIS as dm  # noqa: E402  (sibling import)

# ---------------------------------------------------------------------------
# Region registry
# ---------------------------------------------------------------------------
#
# bounding boxes are derived from the corresponding DEM raster at runtime;
# this dict only encodes the region → dem-file mapping (and is the place to
# wire a new region without touching code elsewhere).
REGION_DEM = {
    'Indus':       '01_data/topo/catchment_dem/dem_Indus.tif',
    'Switzerland': '01_data/topo/catchment_dem/dem_Switzerland.tif',
    'Ganges':      '01_data/topo/catchment_dem/dem_Ganges.tif',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_main_dir(env: Optional[str] = None) -> Path:
    """Look up main_dir from the env config layer (local/server autodetect)."""
    from config_merge import detect_env  # type: ignore
    env = env or detect_env()
    env_yaml = ROOT / 'src' / 'config' / 'layers' / 'env' / f'{env}.yaml'
    with open(env_yaml) as f:
        cfg = yaml.safe_load(f)
    return Path(cfg['main_dir'])


def region_bbox(region: str, main_dir: Path,
                buffer_degrees: float = 0.05) -> Tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) for the region in EPSG:4326."""
    if region not in REGION_DEM:
        raise ValueError(f"Unknown region '{region}'. "
                         f"Known: {sorted(REGION_DEM)}. "
                         f"Add an entry to REGION_DEM in this script.")
    dem_path = main_dir / REGION_DEM[region]
    if not dem_path.exists():
        raise FileNotFoundError(f"Region DEM not found: {dem_path}")
    return dm.get_extent_from_raster(str(dem_path), buffer_degrees=buffer_degrees)


def parse_years(spec: str) -> List[int]:
    """Parse '2000-2014' or '2000' or '2000,2005,2010' into a list of ints."""
    out: List[int] = []
    for chunk in spec.split(','):
        if '-' in chunk:
            a, b = chunk.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def group_hdfs_by_date(hdfs: List[Path]) -> Dict[str, List[Path]]:
    """Group HDFs by acquisition date (YYYY-MM-DD)."""
    out: Dict[str, List[Path]] = {}
    for h in hdfs:
        d = dm.extract_date_from_filename(h.name)
        if d:
            out.setdefault(d, []).append(h)
    return out


# ---------------------------------------------------------------------------
# Per-date assembly
# ---------------------------------------------------------------------------

def _hdfs_to_geotiff(hdfs: List[Path], staging: Path,
                     subdataset: str) -> List[Path]:
    """Convert each HDF to a GeoTIFF in `staging`. Returns the .tif paths."""
    tifs = []
    for h in hdfs:
        tif = staging / (h.stem + '_subds.tif')
        if not tif.exists():
            if not dm.process_hdf_to_geotiff(str(h), str(tif), subdataset):
                print(f"  ⚠️ HDF→GeoTIFF failed for {h.name}")
                continue
        tifs.append(tif)
    return tifs


def _mosaic_and_clip_to_array(
    tifs: List[Path],
    bbox: Tuple[float, float, float, float],
) -> Optional[xr.DataArray]:
    """Mosaic the listed GeoTIFFs and clip to the region bbox.

    Returns a single-band DataArray in EPSG:4326 (whatever the GeoTIFFs are in,
    reprojected on the fly via rioxarray) with NaN outside the bbox.
    """
    import rioxarray  # noqa: F401  (registers .rio accessor)
    if not tifs:
        return None

    arrays = []
    for tif in tifs:
        da = xr.open_dataarray(tif, engine='rasterio').squeeze(drop=True)
        arrays.append(da)

    # rioxarray.merge.merge_arrays handles reprojection-free merging when all
    # inputs share a CRS (MODIS tiles do — Sinusoidal). After merging we
    # reproject the whole thing to EPSG:4326 then clip to the bbox.
    from rioxarray.merge import merge_arrays
    merged = merge_arrays(arrays)

    # Reproject Sinusoidal → EPSG:4326. Nearest-neighbour preserves the
    # categorical MODIS values.
    merged_4326 = merged.rio.reproject('EPSG:4326', resampling=0)  # 0 = nearest

    minx, miny, maxx, maxy = bbox
    clipped = merged_4326.rio.clip_box(
        minx=minx, miny=miny, maxx=maxx, maxy=maxy
    )
    return clipped


# ---------------------------------------------------------------------------
# Annual NetCDF builder
# ---------------------------------------------------------------------------

def build_year_netcdf(
    region: str,
    year: int,
    product: str,
    bbox: Tuple[float, float, float, float],
    tile_cache: Path,
    out_path: Path,
    force: bool = False,
) -> Optional[Path]:
    """Build one <product>_<year>.nc for a region. Returns out_path or None
    if no data was assembled."""
    if out_path.exists() and not force:
        print(f"  ✅ Already exists: {out_path.name} (use --force to rebuild)")
        return out_path

    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    t0, t1 = f"{year}-01-01", f"{year}-12-31"

    # Make sure all tiles for this year are in the cache (downloads what's missing)
    print(f"  📦 Ensuring tile cache for {product} {year}...")
    cached = dm.download_tiles_to_cache(
        product=product, bounding_box=bbox_str,
        time_start=t0, time_end=t1, cache_root=tile_cache,
        quiet=True,
    )
    if not cached:
        print(f"  ⚠️ No granules for {product} in {year}; skipping year.")
        return None

    print(f"  📦 {len(cached)} HDFs available in cache for {year}")

    by_date = group_hdfs_by_date(cached)
    subdataset = dm.PRODUCT_CONFIG[product]['subdataset']

    # Stage HDF→GeoTIFF conversions in a temp dir we clean up afterwards
    daily_arrays: List[xr.DataArray] = []
    dates: List[str] = []

    with tempfile.TemporaryDirectory(prefix=f'modis_build_{region}_{year}_') as staging:
        staging_path = Path(staging)
        for date_str in sorted(by_date):
            hdfs = by_date[date_str]
            print(f"    {date_str}: {len(hdfs)} tile(s)", end='', flush=True)
            tifs = _hdfs_to_geotiff(hdfs, staging_path, subdataset)
            if not tifs:
                print(" — failed")
                continue
            try:
                da = _mosaic_and_clip_to_array(tifs, bbox)
            except Exception as e:
                print(f" — mosaic/clip failed: {e}")
                continue
            if da is None:
                print(" — no data")
                continue
            daily_arrays.append(da)
            dates.append(date_str)
            print(f" — {da.shape}")

    if not daily_arrays:
        print(f"  ⚠️ No usable scenes for {year}; skipping.")
        return None

    # Align all daily grids to the first one's coords (they should already be
    # identical post-bbox clip, but tiny rounding can drift the lat/lon arrays
    # by ~1e-7). Reindexing on the first array's coordinates fixes that.
    ref = daily_arrays[0]
    aligned = [da.reindex_like(ref, method='nearest', tolerance=1e-4)
               for da in daily_arrays]

    time_index = pd.to_datetime(dates)
    stacked = xr.concat(aligned, dim=pd.Index(time_index, name='time'))
    stacked.name = 'snow_extent'
    stacked.attrs.update({
        'long_name':   'MOD10A2 maximum snow extent (raw byte values)',
        'description': ('Categorical snow extent: 25=land, 37=lake, 39=ocean, '
                        '50=cloud, 100=lake ice, 200=snow. Other = invalid.'),
        'units':       '1',
        'source':      'MOD10A2 v061',
        'region':      region,
        'build_date':  datetime.utcnow().strftime('%Y-%m-%d'),
    })

    ds = stacked.to_dataset()

    # Compress everything except coords; uint8 stays uint8.
    encoding = {
        'snow_extent': {'zlib': True, 'complevel': 4, 'dtype': 'uint8'},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path, encoding=encoding)
    print(f"  💾 Wrote {out_path}  ({ds.sizes['time']} dates, "
          f"{ds.sizes.get('y', '?')}×{ds.sizes.get('x', '?')} px)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('region', help='Region name (key of REGION_DEM)')
    ap.add_argument('--product', default='MOD10A2',
                    choices=sorted(dm.PRODUCT_CONFIG))
    ap.add_argument('--years', default='2000-2025',
                    help="Year range, e.g. '2000-2025' or '2000,2005,2010'")
    ap.add_argument('--main-dir', default=None,
                    help='Override main_dir (default: env layer autodetect)')
    ap.add_argument('--env', default=None,
                    help='Env layer override (server/local). Default: autodetect')
    ap.add_argument('--force', action='store_true',
                    help='Rebuild even if the per-year NetCDF already exists.')
    ap.add_argument('--buffer-degrees', type=float, default=0.05,
                    help='Pad region bbox by this many degrees (default 0.05).')
    args = ap.parse_args(argv)

    main_dir = Path(args.main_dir) if args.main_dir else resolve_main_dir(args.env)
    bbox = region_bbox(args.region, main_dir, buffer_degrees=args.buffer_degrees)
    years = parse_years(args.years)

    print(f"Region:      {args.region}")
    print(f"Product:     {args.product}")
    print(f"Main dir:    {main_dir}")
    print(f"Bbox (4326): {bbox}")
    print(f"Years:       {years[0]}–{years[-1]}  ({len(years)} years)")

    tile_cache = main_dir / '01_data' / 'snow' / 'MODIS' / 'tiles'
    out_dir = main_dir / '01_data' / 'snow' / 'MODIS' / 'regions' / args.region
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Tile cache:  {tile_cache}")
    print(f"Output dir:  {out_dir}\n")

    built = 0
    skipped = 0
    for year in years:
        out_path = out_dir / f'{args.product}_{year}.nc'
        print(f"=== {args.region} {args.product} {year} ===")
        try:
            result = build_year_netcdf(
                region=args.region, year=year, product=args.product,
                bbox=bbox, tile_cache=tile_cache, out_path=out_path,
                force=args.force,
            )
            if result is None:
                skipped += 1
            else:
                built += 1
        except Exception as e:
            print(f"  ❌ Year {year} failed: {e}")
            skipped += 1
        print()

    print(f"\nDone. {built} built, {skipped} skipped/failed.")
    return 0 if built > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
