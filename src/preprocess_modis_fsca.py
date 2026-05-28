"""Per-catchment MODIS basin-mean and elevation-band fSCA derivation.

Reads the region NetCDF produced by ``scripts/build_modis_region.py``,
clips to the catchment shape, optionally masks out glacier pixels, then
emits a long-format CSV consumable by the snow calibration objective:

    <main_dir>/01_data/snow/MODIS/basins/<gauge>/fsca_<product>_<gauge>.csv

Output schema (long format works for both aggregation modes):
    date, band_m, fsca, n_valid, n_cloud, n_total

For ``aggregation='basin_mean'``: a single row per date with ``band_m='basin'``.
For ``aggregation='elevation_band'``: one row per (date, elevation_band).

The module is the canonical home for derivation logic; ``scripts/derive_basin_fsca.py``
is a thin CLI wrapper and ``src/create_input_files.py`` calls
``derive_for_catchment(nml)`` as a gated preprocessing step.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr


# MOD10A2 Maximum_Snow_Extent byte codes
MOD10A2_SNOW    = 200
MOD10A2_NO_SNOW = (25, 37)   # 25 = land no-snow, 37 = lake no-snow
MOD10A2_CLOUD   = 50


# ── Region NetCDF discovery / loading ──────────────────────────────────────

def discover_year_files(region_dir: Path, product: str,
                       years: Optional[List[int]] = None) -> List[Path]:
    """List <product>_<year>.nc files under region_dir, optionally filtered."""
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
    """Concatenate per-year MODIS region NetCDFs along time.

    Per-year mosaicking + reproject in build_modis_region.py can produce
    slightly different (x, y) grids across years for multi-tile regions
    (e.g. Indus year 2000: 2442×4692 vs years 2001+: 2334×4483, drift of
    ~30 m vs the 500 m MODIS pixel). xarray's default outer-join concat
    silently fills the union with NaN and will be a hard error in future
    versions. Reindex every non-reference year onto a chosen reference
    grid using nearest-neighbour — the drift is far below pixel resolution
    so the reindex is effectively lossless.
    """
    parts = [xr.open_dataset(f)['snow_extent'] for f in year_files]
    if len(parts) == 1:
        return parts[0]

    # Pick the smallest grid as reference (tight bbox, no over-extension).
    # Fall back to the first part if sizes are identical.
    ref = min(parts, key=lambda p: p.sizes.get('x', 0) * p.sizes.get('y', 0))
    aligned = []
    for p in parts:
        if p.sizes == ref.sizes and (p.x.equals(ref.x)) and (p.y.equals(ref.y)):
            aligned.append(p)
        else:
            aligned.append(p.reindex_like(ref, method='nearest',
                                          tolerance=1e-3))
    return xr.concat(aligned, dim='time', join='override')


def clip_to_catchment(
    da: xr.DataArray,
    catchment_shp: Path,
    glacier_shp: Optional[Path] = None,
) -> xr.DataArray:
    """Clip the region DataArray to the catchment polygon. When a glacier
    shapefile is supplied, also masks pixels inside any glacier outline that
    intersects the catchment bbox.

    Off-catchment / on-glacier cells become NaN — the structure is preserved
    so downstream counting still works.
    """
    import geopandas as gpd
    import rioxarray  # noqa: F401  (registers .rio accessor)

    if da.rio.crs is None:
        da = da.rio.write_crs('EPSG:4326')

    catch_gdf = gpd.read_file(catchment_shp).to_crs(da.rio.crs)
    clipped = da.rio.clip(catch_gdf.geometry, drop=True)

    if glacier_shp is not None and Path(glacier_shp).exists():
        try:
            glacier_gdf = gpd.read_file(glacier_shp).to_crs(da.rio.crs)
            minx, miny, maxx, maxy = catch_gdf.total_bounds
            glacier_clip = glacier_gdf.cx[minx:maxx, miny:maxy]
            if len(glacier_clip) > 0:
                clipped = clipped.rio.clip(glacier_clip.geometry, invert=True,
                                           drop=False)
        except Exception as e:
            print(f"  ⚠️ Glacier masking failed ({e}); using full-catchment "
                  f"pixels.")

    return clipped


# ── DEM resampling for elevation-band aggregation ──────────────────────────

def dem_on_modis_grid(dem_path: Path, target_da: xr.DataArray) -> xr.DataArray:
    """Resample a high-resolution DEM onto the MODIS grid using area-mean.

    Each output MODIS-resolution pixel's value is the mean of the underlying
    DEM cells. CRS is harmonised to whatever target_da uses (EPSG:4326 for
    region NetCDFs from build_modis_region.py).

    Returns an xr.DataArray with the same y/x coordinates as target_da.
    """
    import rioxarray  # noqa: F401
    from rasterio.enums import Resampling

    dem = (xr.open_dataarray(dem_path, engine='rasterio')
             .squeeze(drop=True))
    if dem.rio.crs is None:
        raise ValueError(f"DEM has no CRS: {dem_path}")
    if target_da.rio.crs is None:
        target_da = target_da.rio.write_crs('EPSG:4326')

    # reproject_match resamples + clips onto the target grid in one step
    dem_resampled = dem.rio.reproject_match(target_da,
                                            resampling=Resampling.average)
    return dem_resampled


# ── Aggregation helpers ────────────────────────────────────────────────────

def _per_timestep_counts(da: xr.DataArray, pixel_mask: xr.DataArray):
    """Return (n_snow, n_no_snow, n_cloud, n_total) per timestep within the
    boolean pixel_mask. Each output is a 1-D xr.DataArray indexed by time.
    """
    da_in = da.where(pixel_mask)
    inside  = (~da_in.isnull())
    snow    = (da_in == MOD10A2_SNOW)
    no_snow = (da_in == MOD10A2_NO_SNOW[0]) | (da_in == MOD10A2_NO_SNOW[1])
    cloud   = (da_in == MOD10A2_CLOUD)

    dims = ('y', 'x')
    return (
        snow.sum(dim=dims).astype('int64'),
        no_snow.sum(dim=dims).astype('int64'),
        cloud.sum(dim=dims).astype('int64'),
        inside.sum(dim=dims).astype('int64'),
    )


def basin_mean_per_timestep(da: xr.DataArray) -> pd.DataFrame:
    """Aggregate over the entire catchment (one row per date)."""
    pixel_mask = xr.ones_like(da.isel(time=0), dtype=bool)
    n_snow, n_nosno, n_cloud, n_total = _per_timestep_counts(da, pixel_mask)

    n_valid = (n_snow + n_nosno).astype('int64')
    fsca = (n_snow / xr.where(n_valid > 0, n_valid, 1)).astype('float64')
    fsca = xr.where(n_valid > 0, fsca, np.nan)

    dates = pd.to_datetime(da['time'].values).strftime('%Y-%m-%d')
    return pd.DataFrame({
        'date':    dates,
        'band_m':  ['basin'] * len(dates),
        'fsca':    fsca.values,
        'n_valid': n_valid.values,
        'n_cloud': n_cloud.values,
        'n_total': n_total.values,
    })


def elevation_band_per_timestep(
    da: xr.DataArray,
    elevation: xr.DataArray,
    band_width_m: int = 100,
) -> pd.DataFrame:
    """Aggregate per (date × elevation_band).

    Bands are labelled by their lower edge: e.g. with band_width_m=100, a
    pixel at 2543 m goes into band 2500. Pixels with NaN elevation (off-DEM
    or off-catchment) are excluded.

    Returns a long-format DataFrame: date, band_m, fsca, n_valid, n_cloud,
    n_total. Empty bands (no valid pixels at all) are omitted.

    Implementation note: vectorised over bands via ``np.bincount`` rather
    than a per-band Python loop. For Hunza-scale catchments (~55K pixels,
    63 bands) this is ~10× faster than the naive per-band ``xr.where``
    approach because it avoids reallocating a NaN-padded float copy of the
    data array on every iteration.
    """
    # ----- Per-pixel band assignment (lower edge of the bin) -----
    elev_vals = elevation.values
    finite_elev = np.isfinite(elev_vals)
    band_id_2d = np.full_like(elev_vals, -1, dtype=np.int32)
    band_id_2d[finite_elev] = (
        np.floor(elev_vals[finite_elev] / band_width_m) * band_width_m
    ).astype(np.int32)

    # ----- Pull the MODIS byte data once, flatten spatial dims -----
    # da is (time, y, x); we keep time as axis 0 and flatten y,x into one axis.
    data = da.values                                  # uint8 (time, y, x)
    n_times = data.shape[0]
    data_flat = data.reshape(n_times, -1)             # (time, n_pixels)
    band_flat = band_id_2d.ravel()                    # (n_pixels,)

    # Keep only pixels with a valid band assignment (drops off-DEM /
    # off-catchment pixels in one shot).
    keep_px = band_flat >= 0
    band_flat = band_flat[keep_px]
    data_flat = data_flat[:, keep_px]                 # (time, n_valid_px)

    if band_flat.size == 0:
        return pd.DataFrame(columns=['date', 'band_m', 'fsca',
                                     'n_valid', 'n_cloud', 'n_total'])

    # Dense band index for bincount.
    unique_bands, band_idx_dense = np.unique(band_flat, return_inverse=True)
    n_bands = len(unique_bands)

    # ----- Per-band counts -----
    # n_total per band is constant across time (which pixels exist in each
    # band doesn't depend on the timestep), so compute once.
    n_total_per_band = np.bincount(band_idx_dense, minlength=n_bands)

    # snow/no_snow/cloud counts per (time, band). Each bincount call is
    # O(n_valid_px) — much cheaper than allocating a full (time, y, x) mask
    # per band as the old loop did.
    snow_counts  = np.zeros((n_times, n_bands), dtype=np.int64)
    nosno_counts = np.zeros((n_times, n_bands), dtype=np.int64)
    cloud_counts = np.zeros((n_times, n_bands), dtype=np.int64)

    for t in range(n_times):
        row = data_flat[t]
        snow_counts[t]  = np.bincount(band_idx_dense,
                                       weights=(row == MOD10A2_SNOW),
                                       minlength=n_bands).astype(np.int64)
        nosno_counts[t] = np.bincount(
            band_idx_dense,
            weights=((row == MOD10A2_NO_SNOW[0]) |
                     (row == MOD10A2_NO_SNOW[1])),
            minlength=n_bands,
        ).astype(np.int64)
        cloud_counts[t] = np.bincount(band_idx_dense,
                                       weights=(row == MOD10A2_CLOUD),
                                       minlength=n_bands).astype(np.int64)

    n_valid_per = (snow_counts + nosno_counts).astype(np.int64)
    with np.errstate(divide='ignore', invalid='ignore'):
        fsca_per = np.where(n_valid_per > 0,
                             snow_counts / np.maximum(n_valid_per, 1),
                             np.nan)

    # ----- Reshape into long-format DataFrame -----
    dates = pd.to_datetime(da['time'].values).strftime('%Y-%m-%d')
    frames = []
    for bi, b in enumerate(unique_bands):
        if n_total_per_band[bi] == 0:
            continue
        frames.append(pd.DataFrame({
            'date':    dates,
            'band_m':  int(b),
            'fsca':    fsca_per[:, bi],
            'n_valid': n_valid_per[:, bi],
            'n_cloud': cloud_counts[:, bi],
            'n_total': int(n_total_per_band[bi]),
        }))

    if not frames:
        return pd.DataFrame(columns=['date', 'band_m', 'fsca',
                                     'n_valid', 'n_cloud', 'n_total'])
    return pd.concat(frames, ignore_index=True)


# ── Region resolution + main entry point ───────────────────────────────────

DEFAULT_GAUGE_PREFIX_REGION = {
    '0':  'Indus',
    '2':  'Switzerland',
}


def resolve_region(nml: dict,
                   fallback: Optional[dict] = None) -> Optional[str]:
    """Pick a region for a catchment. Precedence:
    namelist `region:` key → gauge-id prefix fallback → None.
    """
    if 'region' in nml and nml['region']:
        r = str(nml['region']).strip()
        return r.capitalize() if r.islower() else r
    fallback = fallback or DEFAULT_GAUGE_PREFIX_REGION
    gauge = str(nml['gauge_id'])
    for prefix, region in fallback.items():
        if gauge.startswith(prefix):
            return region
    return None


def derive_for_catchment(
    nml: dict,
    aggregation: str = 'basin_mean',
    band_width_m: int = 100,
    glacier_mask: bool = True,
    product: str = 'MOD10A2',
    years: Optional[List[int]] = None,
    region: Optional[str] = None,
    verbose: bool = True,
) -> Path:
    """Top-level entry point. Derives fSCA for a single catchment and writes
    the long-format CSV. Returns the path written.

    Raises FileNotFoundError if the region NetCDF directory or catchment
    shapefile is missing — preprocessing should fail loudly rather than
    silently producing a no-snow calibration target.
    """
    if aggregation not in ('basin_mean', 'elevation_band'):
        raise ValueError(f"aggregation must be 'basin_mean' or "
                         f"'elevation_band'; got {aggregation!r}")

    gauge_id = str(nml['gauge_id'])
    main_dir = Path(nml['main_dir'])

    region = region or resolve_region(nml)
    if region is None:
        raise ValueError(
            f"Could not determine region for {gauge_id}. Add a `region:` "
            f"key to the catchment namelist or extend "
            f"DEFAULT_GAUGE_PREFIX_REGION."
        )

    region_dir = main_dir / '01_data' / 'snow' / 'MODIS' / 'regions' / region
    if not region_dir.exists():
        raise FileNotFoundError(
            f"No MODIS region NetCDF dir: {region_dir}\n"
            f"Run: python scripts/build_modis_region.py {region}"
        )

    year_files = discover_year_files(region_dir, product, years)
    if not year_files:
        raise FileNotFoundError(
            f"No {product}_*.nc files under {region_dir}"
        )

    # Catchment + glacier shapefiles
    shape_template = nml.get(
        'shape_dir',
        '01_data/topo/catchment_shapefile/catchment_shape_{gauge_id}.shp',
    )
    catchment_shp = main_dir / shape_template.format(gauge_id=gauge_id)
    if not catchment_shp.exists():
        raise FileNotFoundError(f"Catchment shape not found: {catchment_shp}")

    glacier_shp: Optional[Path] = None
    if glacier_mask:
        gl = nml.get('glacier_dir', '')
        if gl:
            cand = main_dir / gl
            if cand.exists():
                glacier_shp = cand
            elif verbose:
                print(f"  ⚠️ Glacier shapefile not found ({cand}); "
                      f"deriving without glacier mask.")

    if verbose:
        print(f"Gauge:       {gauge_id}")
        print(f"Region:      {region}")
        print(f"Aggregation: {aggregation}"
              + (f"  (band_width={band_width_m} m)"
                 if aggregation == 'elevation_band' else ''))
        print(f"NetCDFs:     {len(year_files)} files")
        print(f"Catchment:   {catchment_shp.name}")
        print(f"Glacier:     "
              f"{glacier_shp.name if glacier_shp else '(no mask)'}")

    # Load + clip
    da = open_region_dataset(year_files)
    clipped = clip_to_catchment(da, catchment_shp, glacier_shp)

    # Aggregate
    if aggregation == 'basin_mean':
        df = basin_mean_per_timestep(clipped)
    else:
        # Resolve catchment DEM (region DEM is fine; we resample to MODIS grid)
        dem_path = main_dir / nml.get(
            'raster_dir',
            '01_data/topo/catchment_dem/dem_Indus.tif',
        )
        if not dem_path.exists():
            raise FileNotFoundError(
                f"DEM not found at {dem_path}; needed for elevation-band "
                f"aggregation."
            )
        elevation = dem_on_modis_grid(dem_path, clipped)
        df = elevation_band_per_timestep(clipped, elevation, band_width_m)

    # Write CSV
    out_dir = main_dir / '01_data' / 'snow' / 'MODIS' / 'basins' / gauge_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'fsca_{product}_{gauge_id}.csv'
    df.to_csv(out_path, index=False, float_format='%.6f')

    if verbose:
        if aggregation == 'basin_mean':
            print(f"💾 Wrote {out_path}  ({len(df)} dates)")
        else:
            n_bands = df['band_m'].nunique()
            n_dates = df['date'].nunique()
            print(f"💾 Wrote {out_path}  "
                  f"({n_dates} dates × {n_bands} bands = {len(df)} rows)")
    return out_path
