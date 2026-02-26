"""Generate synthetic test fixtures for Raven preprocessing pipeline.

Usage (standalone):
    python tests/fixtures/generate_fixtures.py [--root /path/to/output]

Creates a self-contained directory tree with synthetic data:
  - Catchment shapefiles (single basin + 3 nested sub-basins)
  - DEM raster (30×30, ~34.8–35.2 N, 73.8–74.2 E)
  - Land use raster (30×30, ICIMOD format)
  - Glacier shapefile (3 synthetic glaciers)
  - HAR NetCDF files (yearly, 10×10 curvilinear grid)
  - ERA5 NetCDF files (monthly hourly, 10×10 regular grid)
  - Streamflow CSV files
  - GloGEM NetCDF component files (Discharge, Icemelt, Snowmelt, Rain)

Gauge IDs used
--------------
  9999  – single-basin test
  9900  – multi-subbasin main (full extent)
  9901  – multi-subbasin sub 1 (upstream)
  9902  – multi-subbasin sub 2 (lateral)
"""

import argparse
import datetime
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.transform import from_bounds
from shapely.geometry import Polygon

# ── Constants ─────────────────────────────────────────────────────────────────
# Test period  (warm-up starts 1999, model run 2000-01-01 → 2001-12-31)
WARMUP_YEAR = 1999
START_DATE = datetime.date(2000, 1, 1)
END_DATE = datetime.date(2001, 12, 31)
ALL_YEARS = list(range(WARMUP_YEAR, END_DATE.year + 1))   # [1999, 2000, 2001]

# Spatial extents (minx, miny, maxx, maxy) in EPSG:4326
EXTENT_9999 = (73.80, 34.80, 74.20, 35.20)   # single-basin
EXTENT_9900 = (73.80, 34.80, 74.20, 35.20)   # multi-main  (same footprint)
EXTENT_9901 = (74.00, 34.95, 74.20, 35.20)   # sub 1 – upper-right quadrant
EXTENT_9902 = (73.80, 34.80, 74.00, 35.05)   # sub 2 – lower-left quadrant

# Meteo grid – slightly larger than the basin extents
GRID_MINX, GRID_MINY, GRID_MAXX, GRID_MAXY = 73.50, 34.50, 74.50, 35.50
GRID_NX, GRID_NY = 10, 10    # west_east × south_north

# RGI region for synthetic glaciers (14 = SouthAsiaWest)
RGI_REGION = "RGI60-14"

# Random seed for reproducibility
RNG = np.random.default_rng(42)


# ── Catchment shapefiles ───────────────────────────────────────────────────────
def _make_bbox_polygon(extent):
    minx, miny, maxx, maxy = extent
    return Polygon(
        [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    )


def create_catchment_shapefiles(root: Path) -> None:
    out_dir = root / "01_data" / "topo" / "catchment_shapefile"
    out_dir.mkdir(parents=True, exist_ok=True)

    basins = {
        "9999": EXTENT_9999,
        "9900": EXTENT_9900,
        "9901": EXTENT_9901,
        "9902": EXTENT_9902,
    }

    for gauge_id, extent in basins.items():
        gdf = gpd.GeoDataFrame(
            {"gauge_id": [gauge_id]},
            geometry=[_make_bbox_polygon(extent)],
            crs="EPSG:4326",
        )
        path = out_dir / f"catchment_shape_{gauge_id}.shp"
        gdf.to_file(path)
    print(f"  ✅ Catchment shapefiles → {out_dir}")


# ── DEM raster ─────────────────────────────────────────────────────────────────
def create_dem(root: Path) -> None:
    out_dir = root / "01_data" / "topo" / "catchment_dem"
    out_dir.mkdir(parents=True, exist_ok=True)

    ny, nx = 30, 30
    transform = from_bounds(*EXTENT_9900, width=nx, height=ny)

    # Synthetic elevation: increases from SW (2000 m) to NE (5500 m) + noise
    row_idx, col_idx = np.mgrid[0:ny, 0:nx]
    elevation = (2000 + 3500 * (row_idx / ny + col_idx / nx) / 2
                 + RNG.integers(-150, 150, (ny, nx))).astype(np.float32)
    # Flip rows so row 0 = north (standard raster orientation)
    elevation = elevation[::-1, :]

    path = out_dir / "dem_test.tif"
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=ny, width=nx,
        count=1, dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(elevation, 1)
    print(f"  ✅ DEM ({ny}×{nx}) → {path}")


# ── Land use raster (ICIMOD) ───────────────────────────────────────────────────
def create_landuse(root: Path) -> None:
    """ICIMOD classes: 1=Forest 2=Snow/glacier 3=Forest 4=Riverbed 5=Built 6=Crop
    7=Bare soil 8=Bare rock 9=Grassland"""
    out_dir = root / "01_data" / "topo" / "catchment_landuse"
    out_dir.mkdir(parents=True, exist_ok=True)

    ny, nx = 30, 30
    transform = from_bounds(*EXTENT_9900, width=nx, height=ny)

    # Elevation proxy: 0=low, 1=high (row 0 = north = high in this orientation)
    row_idx, _ = np.mgrid[0:ny, 0:nx]
    elev_norm = 1.0 - row_idx / ny   # row 0 → 1.0, row 29 → ~0

    landuse = np.where(elev_norm > 0.80, 2,   # snow/glacier at very high elevation
              np.where(elev_norm > 0.60, 8,   # bare rock
              np.where(elev_norm > 0.40, 7,   # bare soil
              np.where(elev_norm > 0.25, 9,   # grassland
                       6)))).astype(np.int16)  # cropland at lowest

    # Force a small glacier patch in the top-right corner (for coupled test)
    landuse[0:5, 20:25] = 2

    path = out_dir / "test_landuse.tif"
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=ny, width=nx,
        count=1, dtype="int16",
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(landuse, 1)
    print(f"  ✅ Land use (ICIMOD, {ny}×{nx}) → {path}")


# ── Glacier shapefile ──────────────────────────────────────────────────────────
def create_glacier_shapefile(root: Path) -> None:
    """Create 3 small synthetic glaciers with RGIId attributes."""
    out_dir = root / "01_data" / "topo" / "glacier_outlines"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Three small glacier rectangles in the NE (high elevation) part of basin
    glaciers = [
        {"RGIId": f"{RGI_REGION}.00029", "geometry": Polygon([
            (74.08, 35.10), (74.12, 35.10), (74.12, 35.14), (74.08, 35.14)])},
        {"RGIId": f"{RGI_REGION}.00030", "geometry": Polygon([
            (74.13, 35.12), (74.16, 35.12), (74.16, 35.16), (74.13, 35.16)])},
        {"RGIId": f"{RGI_REGION}.00031", "geometry": Polygon([
            (74.05, 35.06), (74.08, 35.06), (74.08, 35.09), (74.05, 35.09)])},
    ]
    gdf = gpd.GeoDataFrame(glaciers, crs="EPSG:4326")
    path = out_dir / "rgi_test_glaciers.shp"
    gdf.to_file(path)
    print(f"  ✅ Glacier shapefile ({len(gdf)} glaciers) → {path}")
    return gdf


# ── Streamflow CSVs ────────────────────────────────────────────────────────────
def create_streamflow_csvs(root: Path) -> None:
    out_dir = root / "01_data" / "streamflow"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate daily dates from warmup start through end
    dates = pd.date_range(
        start=datetime.date(WARMUP_YEAR, 1, 1), end=END_DATE, freq="D"
    )
    n = len(dates)

    for gauge_id, mean_q, noise_scale in [
        ("9999", 120.0, 30.0),
        ("9900", 120.0, 30.0),
        ("9901",  60.0, 15.0),
        ("9902",  40.0, 10.0),
    ]:
        # Seasonal cycle + noise (summer peak for Karakoram-style)
        doy = np.array([d.timetuple().tm_yday for d in dates.date])
        q_obs = (mean_q
                 + mean_q * 0.8 * np.sin(2 * np.pi * (doy - 60) / 365)
                 + RNG.normal(0, noise_scale, n))
        q_obs = np.clip(q_obs, 0.5, None)   # no negatives

        df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "Q_obs": q_obs})
        path = out_dir / f"streamflow_{gauge_id}.csv"
        df.to_csv(path, index=False)
    print(f"  ✅ Streamflow CSVs → {out_dir}")


# ── HAR NetCDF files ───────────────────────────────────────────────────────────
def _make_2d_grid(grid_minx, grid_miny, grid_maxx, grid_maxy, nx, ny):
    """Return 2D lat/lon arrays (south_north × west_east)."""
    lons_1d = np.linspace(grid_minx, grid_maxx, nx)
    lats_1d = np.linspace(grid_miny, grid_maxy, ny)
    lon_2d, lat_2d = np.meshgrid(lons_1d, lats_1d)  # (ny, nx)
    return lat_2d, lon_2d


def create_har_files(root: Path) -> None:
    """
    Create synthetic HAR yearly NetCDF files.

    File naming: HARv2_d10km_d_2d_{variable}_{year}.nc
    Dimensions:  (time, south_north, west_east)
    Coordinates: lat[south_north, west_east], lon[south_north, west_east]
    Units:       temperature in K, precipitation in mm h-1
    """
    out_dir = root / "01_data" / "meteo" / "HAR"
    out_dir.mkdir(parents=True, exist_ok=True)

    lat_2d, lon_2d = _make_2d_grid(
        GRID_MINX, GRID_MINY, GRID_MAXX, GRID_MAXY, GRID_NX, GRID_NY
    )

    variables = {
        "t2_mean": {"mean": 285.0,  "amplitude": 15.0, "units": "K"},
        "t2_min":  {"mean": 278.0,  "amplitude": 12.0, "units": "K"},
        "t2_max":  {"mean": 292.0,  "amplitude": 15.0, "units": "K"},
        "prcp":    {"mean": 0.04,   "amplitude": 0.06, "units": "mm h-1"},
        "potevap": {"mean": 0.03,   "amplitude": 0.02, "units": "mm h-1"},
    }

    for year in ALL_YEARS:
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        n_days = len(dates)
        doy = np.array([d.timetuple().tm_yday for d in dates.date])

        for var_name, meta in variables.items():
            # Seasonal signal + small spatial gradient + noise
            seasonal = (meta["mean"]
                        + meta["amplitude"] * np.sin(2 * np.pi * (doy - 60) / 365))

            # Shape: (time, south_north, west_east)
            data = (seasonal[:, None, None]
                    + RNG.normal(0, meta["amplitude"] * 0.1,
                                 (n_days, GRID_NY, GRID_NX)))
            data = data.astype(np.float32)

            # Precipitation must be non-negative
            if "prcp" in var_name:
                data = np.clip(data, 0.0, None)

            ds = xr.Dataset(
                {var_name: (["time", "south_north", "west_east"], data,
                             {"units": meta["units"]})},
                coords={
                    "time": dates,
                    "lat":  (["south_north", "west_east"], lat_2d.astype(np.float32)),
                    "lon":  (["south_north", "west_east"], lon_2d.astype(np.float32)),
                },
            )
            fname = f"HARv2_d10km_d_2d_{var_name}_{year}.nc"
            ds.to_netcdf(out_dir / fname)

    # Static terrain height file (single, no time dimension)
    elev_2d = (2000.0 + 3500.0 * (lat_2d - GRID_MINY) / (GRID_MAXY - GRID_MINY)
               ).astype(np.float32)
    ds_hgt = xr.Dataset(
        {"hgt": (["south_north", "west_east"], elev_2d, {"units": "m"})},
        coords={
            "lat": (["south_north", "west_east"], lat_2d.astype(np.float32)),
            "lon": (["south_north", "west_east"], lon_2d.astype(np.float32)),
        },
    )
    ds_hgt.to_netcdf(out_dir / "HARv2_d10km_static_hgt.nc")

    print(f"  ✅ HAR NetCDF files ({len(ALL_YEARS)} years × {len(variables)} vars"
          f" + static hgt) → {out_dir}")


# ── ERA5 NetCDF files ──────────────────────────────────────────────────────────
def create_era5_files(root: Path) -> None:
    """
    Create synthetic ERA5-Land monthly NetCDF files (hourly time steps).

    File naming:
        era5_land_2m_temperature_{year}_{month:02d}.nc       → variable: t2m  (K)
        era5_land_total_precipitation_{year}_{month:02d}.nc  → variable: tp   (m, cumulative)
        era5_land_potential_evaporation_{year}_{month:02d}.nc → variable: pev (m, cumulative)
    Dimensions: (time, latitude, longitude)  — time is hourly
    """
    out_dir = root / "01_data" / "meteo" / "ERA5"
    out_dir.mkdir(parents=True, exist_ok=True)

    lats_1d = np.linspace(GRID_MINY, GRID_MAXY, GRID_NY, dtype=np.float32)[::-1]
    lons_1d = np.linspace(GRID_MINX, GRID_MAXX, GRID_NX, dtype=np.float32)

    for year in ALL_YEARS:
        for month in range(1, 13):
            # Hourly time stamps for the month
            start = pd.Timestamp(year, month, 1)
            if month < 12:
                end = pd.Timestamp(year, month + 1, 1)
            else:
                end = pd.Timestamp(year + 1, 1, 1)
            times = pd.date_range(start, end, freq="h", inclusive="left")
            n_h = len(times)
            doy = np.array([t.dayofyear for t in times], dtype=float)

            # ── temperature: K, constant spatial gradient + seasonal + diurnal
            doy_arr = np.asarray(doy, dtype=float)
            hours_arr = np.asarray(times.hour, dtype=float)
            t2m_base = 283.0 + 10.0 * np.sin(2 * np.pi * (doy_arr - 60) / 365)
            t2m_diurnal = 5.0 * np.sin(2 * np.pi * hours_arr / 24)
            t2m = (t2m_base + t2m_diurnal)[:, None, None] * np.ones(
                (n_h, GRID_NY, GRID_NX), dtype=np.float32
            )
            t2m += RNG.normal(0, 0.5, t2m.shape).astype(np.float32)

            # ── precipitation: cumulative metres (ERA5 style)
            # Small random increments each hour, cumulative within forecast step
            # Simplified: hourly increments that are positive and small
            hourly_precip = np.clip(
                RNG.normal(0.0002, 0.0003, (n_h, GRID_NY, GRID_NX)), 0, None
            ).astype(np.float32)
            tp_cum = np.cumsum(hourly_precip, axis=0)

            # ── PET: cumulative metres with 24-hour reset cycle (ERA5 style).
            # Use POSITIVE increments so the ERA5LandAnalyzer's negative-mask
            # "forecast restart" handler does not replace hourly diffs with the
            # large cumulative values (which would give >1000 mm/day PET).
            # The processing code flips the sign at the daily-sum stage via
            # `if pet_mean_value < 0`.
            daily_rate = np.clip(
                RNG.normal(0.003, 0.001, (n_h // 24 + 1, GRID_NY, GRID_NX)),
                0.0005, None,
            ).astype(np.float32)   # ~3 mm/day target
            pev_cum = np.zeros((n_h, GRID_NY, GRID_NX), dtype=np.float32)
            for h in range(n_h):
                day_i = h // 24
                hour_in_day = h % 24
                # Accumulate within each day (positive → 3 mm by end of day)
                pev_cum[h] = daily_rate[min(day_i, len(daily_rate) - 1)] \
                             * (hour_in_day + 1) / 24

            coords = {"time": times, "latitude": lats_1d, "longitude": lons_1d}

            for var_name, data, units in [
                ("t2m", t2m,     "K"),
                ("tp",  tp_cum,  "m"),
                ("pev", pev_cum, "m"),
            ]:
                ds = xr.Dataset(
                    {var_name: (["time", "latitude", "longitude"], data,
                                {"units": units})},
                    coords=coords,
                )
                name_map = {
                    "t2m": "era5_land_2m_temperature",
                    "tp":  "era5_land_total_precipitation",
                    "pev": "era5_land_potential_evaporation",
                }
                fname = f"{name_map[var_name]}_{year}_{month:02d}.nc"
                ds.to_netcdf(out_dir / fname)

    # ── Geopotential file (elevation proxy, g * z in m²/s²)
    # ERA5LandAnalyzer uses this to compute station elevation.
    g = 9.80665
    # Elevation increases with latitude; broadcast to (lat × lon) grid
    lats_norm = (lats_1d - GRID_MINY) / (GRID_MAXY - GRID_MINY)   # shape (NY,)
    elev_m = (2000.0 + 3500.0 * lats_norm[:, None] * np.ones((1, GRID_NX))
              ).astype(np.float32)   # shape (NY, NX)
    geop = (elev_m * g).astype(np.float32)   # m²/s²
    ds_geo = xr.Dataset(
        {"z": (["latitude", "longitude"], geop, {"units": "m**2 s**-2"})},
        coords={"latitude": lats_1d, "longitude": lons_1d},
    )
    ds_geo.to_netcdf(out_dir / "era5_land_geopotential.nc")

    n_months = len(ALL_YEARS) * 12
    print(f"  ✅ ERA5 NetCDF files ({n_months} months × 3 vars + geopotential) → {out_dir}")


# ── GloGEM NetCDF files ────────────────────────────────────────────────────────
def create_glogem_files(root: Path, basin_name: str = "Test",
                        scenario: str = "ssp126") -> None:
    """
    Create synthetic GloGEM component NetCDF files.

    File naming: GloGEM_{component}_{basin_name}_{scenario}.nc
    Dimensions:  (time, glacier_id)
    Glacier IDs: 5-digit zero-padded integers matching the RGI glacier shapefile
    """
    out_dir = root / "01_data" / "GloGEM" / basin_name / "GMB"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Glacier IDs matching the shapefile (RGI60-14.00029 → 29, padded → '00029')
    glacier_numeric_ids = np.array([29, 30, 31], dtype=np.int32)

    # Daily dates covering full test period including warmup
    dates = pd.date_range(
        datetime.date(WARMUP_YEAR, 1, 1), END_DATE, freq="D"
    )
    n_days = len(dates)
    n_gl = len(glacier_numeric_ids)

    doy = np.array([d.timetuple().tm_yday for d in dates.date])
    seasonal = 0.5 + 0.4 * np.sin(2 * np.pi * (doy - 60) / 365)

    components = {
        "Discharge": (seasonal[:, None] * RNG.uniform(0.8, 1.2, (1, n_gl))).astype(np.float32),
        "Icemelt":   (0.5 * seasonal[:, None] * RNG.uniform(0.8, 1.2, (1, n_gl))).astype(np.float32),
        "Snowmelt":  (0.3 * seasonal[:, None] * RNG.uniform(0.8, 1.2, (1, n_gl))).astype(np.float32),
        "Rain":      (0.1 * seasonal[:, None] * RNG.uniform(0.5, 1.5, (1, n_gl))).astype(np.float32),
    }
    for comp, base in components.items():
        data = np.clip(base + RNG.normal(0, 0.05, (n_days, n_gl)).astype(np.float32), 0, None)
        ds = xr.Dataset(
            {comp.lower(): (["time", "glacier_id"], data,
                            {"units": "mm w.e. d-1"})},
            coords={
                "time": dates,
                "glacier_id": glacier_numeric_ids,
            },
        )
        fname = f"GloGEM_{comp}_{basin_name}_{scenario}.nc"
        ds.to_netcdf(out_dir / fname)

    print(f"  ✅ GloGEM files ({len(components)} components, {n_gl} glaciers)"
          f" → {out_dir}")


# ── glacier_id_mapping.csv (needed for GloGEMProcessor) ───────────────────────
def create_glacier_id_mapping(root: Path, gauge_id: str = "9999") -> None:
    """Write the glacier_id_mapping.csv that GloGEMProcessor reads."""
    out_dir = (root / "test_outputs" / f"catchment_{gauge_id}" / "topo_files")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "RGIId":      [f"{RGI_REGION}.00029", f"{RGI_REGION}.00030",
                       f"{RGI_REGION}.00031"],
        "numeric_id": [29, 30, 31],
        "is_large":   [True, False, False],
    })
    path = out_dir / "glacier_id_mapping.csv"
    df.to_csv(path, index=False)
    print(f"  ✅ glacier_id_mapping.csv → {path}")


# ── TPHiPr NetCDF files ────────────────────────────────────────────────────────
def create_tphipr_files(root: Path) -> None:
    """
    Create synthetic TPHiPr daily precipitation NetCDF files.

    File naming: tpmfd_prcp_{year}.nc
    Dimensions: (time, latitude, longitude) – time is daily
    Variable:   prcp (mm.hr-1)
    Grid:       regular lat/lon (N→S latitudes), 15×15 cells covering test domain
    """
    out_dir = root / "01_data" / "meteo" / "TPHiPr"
    out_dir.mkdir(parents=True, exist_ok=True)

    TPHIPR_NX, TPHIPR_NY = 15, 15
    # N→S latitudes, W→E longitudes – same domain as other meteo datasets
    lats_1d = np.linspace(GRID_MAXY, GRID_MINY, TPHIPR_NY, dtype=np.float64)
    lons_1d = np.linspace(GRID_MINX, GRID_MAXX, TPHIPR_NX, dtype=np.float64)

    for year in ALL_YEARS:
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        n_days = len(dates)
        doy = np.array([t.dayofyear for t in dates], dtype=float)

        # Seasonal precipitation in mm.hr-1; ~0.1–0.4 mm/hr = 2.4–9.6 mm/day
        seasonal = 0.1 + 0.15 * np.sin(2 * np.pi * (doy - 180) / 365)
        prcp = (
            seasonal[:, None, None] * np.ones((n_days, TPHIPR_NY, TPHIPR_NX))
            + RNG.exponential(0.03, (n_days, TPHIPR_NY, TPHIPR_NX))
        ).astype(np.float32)
        prcp = np.clip(prcp, 0, None)

        ds = xr.Dataset(
            {"prcp": (["time", "latitude", "longitude"], prcp,
                       {"units": "mm.hr-1", "long_name": "precipitation"})},
            coords={"time": dates, "latitude": lats_1d, "longitude": lons_1d},
        )
        ds.to_netcdf(out_dir / f"tpmfd_prcp_{year}.nc")

    print(f"  ✅ TPHiPr NetCDF files ({len(ALL_YEARS)} years) → {out_dir}")


# ── Main entry point ───────────────────────────────────────────────────────────
def generate_all_fixtures(root: Path) -> None:
    """Generate all synthetic fixtures under *root*."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"\n🔧 Generating synthetic test fixtures in: {root}")
    create_catchment_shapefiles(root)
    create_dem(root)
    create_landuse(root)
    create_glacier_shapefile(root)
    create_streamflow_csvs(root)
    create_har_files(root)
    create_era5_files(root)
    create_glogem_files(root)
    create_tphipr_files(root)
    print("\n✅ All fixtures generated.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path(__file__).parent / "data"),
        help="Output root directory (default: tests/fixtures/data/)",
    )
    args = parser.parse_args()
    generate_all_fixtures(Path(args.root))
