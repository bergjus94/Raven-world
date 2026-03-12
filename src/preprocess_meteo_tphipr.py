#### This file contains all functions for plotting and analyzing ERA5-Land meteorological data
#### Updated for plotting and time series analysis with namelist configuration
#### Justine Berg

#### This file contains all functions for plotting and analyzing ERA5-Land meteorological data
#### Updated for plotting and time series analysis with namelist configuration
#### Justine Berg

#--------------------------------------------------------------------------------
############################### import packages #################################
#--------------------------------------------------------------------------------

import geopandas as gpd
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import seaborn as sns
import logging
import yaml
from typing import Dict, List, Union, Optional, Any, Tuple
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings('ignore')

from preprocess_meteo_base import MeteoBase, normalize_coords

#--------------------------------------------------------------------------------
############################### TPHiPrAnalyzer ##################################
#--------------------------------------------------------------------------------

class TPHiPrAnalyzer(MeteoBase):
    """
    Clip and unit-convert TPHiPr yearly precipitation files.

    TPHiPr characteristics:
    - ~0.03° resolution (~3 km), regular lat/lon grid
    - Daily files: tpmfd_prcp_{year}.nc, variable 'prcp'
    - Source units: mm/hr  →  converted to mm/day (×24)
    - Output: tphipr_precip.nc in shared_data_dir
    """

    _logger_class_name = 'TPHiPrAnalyzer'

    def __init__(self, namelist_path: Union[str, Path], force_reprocess: bool = False) -> None:
        super().__init__(namelist_path, force_reprocess)

        # TPHiPr source directory
        tphipr_dir = self.config.get('tphipr_dir', '')
        self.tphipr_dir = Path(tphipr_dir)
        if not self.tphipr_dir.is_absolute():
            self.tphipr_dir = self.main_dir / self.tphipr_dir

        self.logger.info(f"TPHiPrAnalyzer initialized for gauge {self.gauge_id}")
        self.logger.info(f"TPHiPr source directory: {self.tphipr_dir}")

    #---------------------------------------------------------------------------------

    def process(self) -> Path:
        """
        Clip, unit-convert, and save TPHiPr precipitation.

        Returns
        -------
        Path
            Path to the saved tphipr_precip.nc file.
        """
        output_file = self.shared_data_dir / 'tphipr_precip.nc'

        if output_file.exists() and not self.force_reprocess:
            self.logger.info("✅ tphipr_precip.nc already exists – skipping")
            return output_file

        # Determine year range (extend to warm-up if configured)
        start_year = self.start_date.year
        if self.warmup_date is not None:
            start_year = min(start_year, self.warmup_date.year)
        end_year = self.end_date.year

        # Locate yearly files
        yearly_files = []
        for year in range(start_year, end_year + 1):
            f = self.tphipr_dir / f'tpmfd_prcp_{year}.nc'
            if f.exists():
                yearly_files.append(f)
                self.logger.info(f"  Found: {f.name}")
            else:
                self.logger.warning(f"  Not found: {f}")

        if not yearly_files:
            raise FileNotFoundError(
                f"No TPHiPr files found in {self.tphipr_dir} for years {start_year}–{end_year}"
            )

        # Use combine='nested' to avoid the coordinate-union problem that
        # combine='by_coords' causes when floating-point lat/lon values differ
        # by machine epsilon across files (producing NaN-padded coordinate arrays).
        self.logger.info(f"Opening {len(yearly_files)} TPHiPr files with open_mfdataset…")
        ds = xr.open_mfdataset(yearly_files, combine='nested', concat_dim='time')

        # Detect lat/lon dimension names
        lat_dim = next((n for n in ('latitude', 'lat') if n in ds.dims), None)
        lon_dim = next((n for n in ('longitude', 'lon') if n in ds.dims), None)
        self.logger.info(f"Detected dims: lat='{lat_dim}', lon='{lon_dim}'")

        # Clip to catchment bounding box (+0.1° buffer)
        bounds = tuple(self.catchment_extent.total_bounds) if self.catchment_extent is not None else None
        if bounds is not None and lat_dim and lon_dim:
            minx, miny, maxx, maxy = bounds
            buf = 0.1
            lat_vals = ds[lat_dim].values
            lat_increasing = lat_vals[0] < lat_vals[-1]
            lat_slice = (
                slice(miny - buf, maxy + buf) if lat_increasing
                else slice(maxy + buf, miny - buf)
            )
            lon_slice = slice(minx - buf, maxx + buf)
            ds = ds.sel({lat_dim: lat_slice, lon_dim: lon_slice})
            self.logger.info(
                f"Clipped to catchment: lat [{miny:.3f}, {maxy:.3f}], lon [{minx:.3f}, {maxx:.3f}]"
            )

        # Filter to simulation (+ warm-up) date range
        # Note: TPHiPr may not cover the full requested range; clip to available data.
        start_filter = self.warmup_date if self.warmup_date is not None else self.start_date
        ds = ds.sel(time=slice(start_filter, self.end_date))
        self.logger.info(
            f"Time range: {ds.time.values[0]} – {ds.time.values[-1]}"
        )
        avail_start = pd.Timestamp(ds.time.values[0])
        avail_end   = pd.Timestamp(ds.time.values[-1])
        if avail_start > start_filter:
            self.logger.warning(
                f"⚠️ TPHiPr starts {avail_start.date()} – earlier data unavailable"
            )
        if avail_end < self.end_date:
            self.logger.warning(
                f"⚠️ TPHiPr ends {avail_end.date()} – later data unavailable"
            )

        # Fill any missing days so Raven sees uniform daily time steps.
        # Missing days are filled with 0 precipitation.
        full_time = pd.date_range(avail_start, avail_end, freq='D')
        n_missing = len(full_time) - len(ds.time)
        if n_missing > 0:
            self.logger.warning(
                f"⚠️ {n_missing} missing day(s) in TPHiPr – filling with 0 mm/day"
            )
            ds = ds.reindex(time=full_time, fill_value=0.0)
            self.logger.info(f"Reindexed to {len(full_time)} continuous daily steps")

        # Convert units mm/hr → mm/day if needed
        src_units = ds['prcp'].attrs.get('units', '')
        needs_conversion = any(
            tok in src_units.lower()
            for tok in ('hr', 'h-1', '/h')
        ) or src_units.strip() in ('mm/hr', 'mm hr-1', 'mm.hr-1', 'mmhr-1')
        if needs_conversion:
            self.logger.info(f"Converting units from '{src_units}' to mm/day (×24)")
            ds['prcp'] = ds['prcp'] * 24.0
        else:
            self.logger.info(f"Units '{src_units}' – assuming already mm/day, no conversion")
        ds['prcp'].attrs['units'] = 'mm/day'

        # ── Add warm-up period by repeating the first year of data ──────
        # Same approach as _add_warmup_to_files() in preprocess_meteo_base.py
        # used by ERA5 and HAR preprocessors.
        avail_start = pd.Timestamp(ds.time.values[0])
        if self.warmup_date is not None and avail_start > self.warmup_date:
            self.logger.info("Adding warm-up period to TPHiPr data...")
            years_diff = (avail_start - self.warmup_date).days / 365.25
            n_repetitions = max(1, int(np.ceil(years_diff)))

            self.logger.info(f"📅 Warm-up: {self.warmup_date.date()} to "
                             f"{(avail_start - pd.Timedelta(days=1)).date()} "
                             f"({n_repetitions} repetition(s) of first year)")

            # Extract first year of available data
            first_year_end = avail_start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
            first_year = ds.sel(time=slice(avail_start, first_year_end))

            # Repeat first year n times with shifted time coordinates
            repeated_datasets = []
            current_time = pd.to_datetime(self.warmup_date)

            for i in range(n_repetitions):
                year_copy = first_year.copy(deep=True)
                time_deltas = first_year.time - first_year.time.values[0]
                new_times = current_time + time_deltas
                year_copy['time'] = new_times
                repeated_datasets.append(year_copy)
                current_time = new_times.values[-1] + pd.Timedelta(days=1)
                self.logger.debug(f"  Repetition {i+1}: {new_times.values[0]} to {new_times.values[-1]}")

            warmup_data = xr.concat(repeated_datasets, dim='time')

            # Trim warm-up to end exactly one day before available data starts
            warmup_end = avail_start - pd.Timedelta(days=1)
            warmup_data = warmup_data.sel(time=slice(self.warmup_date, warmup_end))

            self.logger.info(f"  Warm-up: {warmup_data.time.min().values} to "
                             f"{warmup_data.time.max().values} ({len(warmup_data.time)} days)")

            # Concatenate warm-up + simulation data
            ds = xr.concat([warmup_data, ds], dim='time')
            ds = ds.sortby('time')

            # Verify no duplicates
            combined_times = pd.to_datetime(ds.time.values)
            duplicates = combined_times.duplicated()
            if duplicates.any():
                self.logger.warning(f"⚠️ {duplicates.sum()} duplicate timestamps – removing")
                ds = ds.isel(time=~duplicates)

            # Verify consecutive time steps
            time_diffs = np.diff(ds.time.values).astype('timedelta64[D]').astype(int)
            non_consecutive = np.where(time_diffs != 1)[0]
            if len(non_consecutive) > 0:
                self.logger.warning(f"⚠️ {len(non_consecutive)} non-consecutive time steps")
            else:
                self.logger.info(f"  ✅ All time steps consecutive (1-day intervals)")

            self.logger.info(f"  Combined: {ds.time.min().values} to {ds.time.max().values} "
                             f"({len(ds.time)} days)")

        # Metadata
        ds.attrs.update({
            'title': 'TPHiPr daily precipitation',
            'source': 'TPHiPr High-Resolution Precipitation Dataset (~0.03°)',
            'gauge_id': str(self.gauge_id),
            'processed_by': 'TPHiPrAnalyzer',
            'creation_date': pd.Timestamp.now().isoformat(),
        })

        # Save – explicitly set _FillValue=None for coordinate variables so they
        # are written as plain numeric arrays without a fill-value attribute.
        encoding = {
            'prcp': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
        }
        if lat_dim:
            encoding[lat_dim] = {'dtype': 'float64', '_FillValue': None}
        if lon_dim:
            encoding[lon_dim] = {'dtype': 'float64', '_FillValue': None}
        self.logger.info(f"Saving to {output_file} …")
        ds.to_netcdf(output_file, encoding=encoding)
        ds.close()

        self.logger.info("✅ Saved tphipr_precip.nc")
        return output_file


#--------------------------------------------------------------------------------
######################### TPHiPrGridWeightsGenerator ############################
#--------------------------------------------------------------------------------

class TPHiPrGridWeightsGenerator(MeteoBase):
    """
    Generate a Raven GridWeights file for the TPHiPr regular lat/lon grid.
    Uses tphipr_precip.nc (written by TPHiPrAnalyzer) as the grid reference.
    Output: GridWeights_TPHiPr.txt in shared_data_dir.
    """

    _logger_class_name = 'TPHiPrGridWeightsGenerator'

    def __init__(self, namelist_path: Union[str, Path], force_reprocess: bool = False) -> None:
        super().__init__(namelist_path, force_reprocess)

        self.out_HRU_shape_dir = (
            self.model_dir / f'catchment_{self.gauge_id}' / 'topo_files' / 'HRU.shp'
        )
        self.logger.info(f"TPHiPrGridWeightsGenerator initialized for gauge {self.gauge_id}")

    #---------------------------------------------------------------------------------

    def generate(self) -> gpd.GeoDataFrame:
        """
        Generate GridWeights_TPHiPr.txt.

        Returns
        -------
        gpd.GeoDataFrame
            Relative area table (empty if file already existed).
        """
        gridweights_file = self.shared_data_dir / 'GridWeights_TPHiPr.txt'

        if gridweights_file.exists() and not self.force_reprocess:
            self.logger.info("✅ GridWeights_TPHiPr.txt already exists – skipping")
            return gpd.GeoDataFrame()

        # Load TPHiPr precip file as grid reference
        tphipr_file = self.shared_data_dir / 'tphipr_precip.nc'
        if not tphipr_file.exists():
            raise FileNotFoundError(
                f"tphipr_precip.nc not found at {tphipr_file}. "
                "Run TPHiPrAnalyzer.process() first."
            )

        self.logger.info(f"Loading TPHiPr grid from {tphipr_file}")
        ds = xr.open_dataset(tphipr_file)

        # Detect coordinate names
        lat_coord = next((n for n in ('latitude', 'lat') if n in ds.coords), None)
        lon_coord = next((n for n in ('longitude', 'lon') if n in ds.coords), None)
        if lat_coord is None or lon_coord is None:
            raise ValueError(
                f"Cannot detect lat/lon coords in tphipr_precip.nc. "
                f"Available: {list(ds.coords)}"
            )

        lat_1d = ds[lat_coord].values
        lon_1d = ds[lon_coord].values
        nlat, nlon = len(lat_1d), len(lon_1d)
        self.total_grid_cells = nlat * nlon
        self.logger.info(f"Grid: {nlat} lat × {nlon} lon = {self.total_grid_cells} cells")

        # Load HRU shapefile
        if not self.out_HRU_shape_dir.exists():
            raise FileNotFoundError(f"HRU shapefile not found: {self.out_HRU_shape_dir}")

        HRU = gpd.read_file(self.out_HRU_shape_dir)
        if 'HRU_ID' in HRU.columns:
            HRU = HRU.sort_values(by='HRU_ID').reset_index(drop=True)
            HRU['HRU ID'] = HRU['HRU_ID']
        elif 'HRU ID' not in HRU.columns:
            HRU['HRU ID'] = list(range(1, len(HRU) + 1))

        self.logger.info(f"Loaded {len(HRU)} HRUs")
        HRU_wgs84 = HRU.to_crs('EPSG:4326')
        hru_id_col = 'HRU ID' if 'HRU ID' in HRU_wgs84.columns else 'HRU_ID'

        # Cell half-sizes – use nanmedian of all diffs for robustness against
        # occasional NaN values at the coordinate boundaries.
        dlat = float(np.nanmedian(np.abs(np.diff(lat_1d)))) / 2 if nlat > 1 else 0.015
        dlon = float(np.nanmedian(np.abs(np.diff(lon_1d)))) / 2 if nlon > 1 else 0.015
        if np.isnan(dlat) or dlat == 0:
            dlat = 0.01667  # fallback: half of nominal ~0.0333° spacing
        if np.isnan(dlon) or dlon == 0:
            dlon = 0.01667
        self.logger.info(f"Cell half-size: dlat={dlat:.5f}°, dlon={dlon:.5f}°")

        # Catchment bounding box for fast filtering
        hru_minx, hru_miny, hru_maxx, hru_maxy = HRU_wgs84.total_bounds
        buf_lat = dlat * 4
        buf_lon = dlon * 4

        # Build grid polygon GeoDataFrame (catchment extent only)
        self.logger.info("Building TPHiPr grid polygons (catchment extent only)…")
        polygons = []
        cell_ids = []

        for j, lat in enumerate(lat_1d):
            for i, lon in enumerate(lon_1d):
                if (lon < hru_minx - buf_lon or lon > hru_maxx + buf_lon or
                        lat < hru_miny - buf_lat or lat > hru_maxy + buf_lat):
                    continue
                corners = [
                    (lon - dlon, lat - dlat),
                    (lon + dlon, lat - dlat),
                    (lon + dlon, lat + dlat),
                    (lon - dlon, lat + dlat),
                ]
                polygons.append(Polygon(corners))
                cell_ids.append(str(j * nlon + i))

        tphipr_grid = gpd.GeoDataFrame(
            {'cell_id': cell_ids, 'geometry': polygons}, crs='EPSG:4326'
        )
        self.logger.info(f"Created {len(tphipr_grid)} grid polygons")

        # Intersection with HRUs
        self.logger.info("Computing HRU–grid intersection…")
        res_union = HRU_wgs84.overlay(tphipr_grid, how='intersection')

        # Relative areas
        res_union_proj = res_union.to_crs('ESRI:54009')
        res_union['area'] = res_union_proj.geometry.area
        hru_totals = res_union.groupby(hru_id_col)['area'].transform('sum')
        res_union['area_rel'] = np.where(hru_totals > 0, res_union['area'] / hru_totals, 0)
        res_union['area_rel'] = res_union['area_rel'].round(5)

        def normalize_group(group):
            total = group['area_rel'].sum()
            group['normalized_relative_area'] = (
                (group['area_rel'] / total).round(5) if total > 0 else 0
            )
            return group

        relative_area = res_union.groupby(hru_id_col, group_keys=False).apply(normalize_group)

        # Handle HRUs with no grid overlap → assign nearest cell
        all_hru_ids = set(HRU_wgs84[hru_id_col].values)
        hrus_with_weights = set(relative_area[hru_id_col].values)
        missing_hrus = all_hru_ids - hrus_with_weights

        if missing_hrus:
            self.logger.warning(f"⚠️ {len(missing_hrus)} HRUs have no grid overlap; assigning nearest cell")
            new_rows = []
            for hru_id in missing_hrus:
                hru_geom = HRU_wgs84[HRU_wgs84[hru_id_col] == hru_id].geometry.values[0]
                distances = tphipr_grid.geometry.centroid.distance(hru_geom.centroid)
                nearest_cell_id = tphipr_grid.loc[distances.idxmin(), 'cell_id']
                new_rows.append({
                    hru_id_col: hru_id,
                    'cell_id': nearest_cell_id,
                    'area_rel': 1.0,
                    'normalized_relative_area': 1.0,
                    'area': 0,
                    'geometry': hru_geom,
                })
            if new_rows:
                relative_area = pd.concat(
                    [relative_area, gpd.GeoDataFrame(new_rows, crs=relative_area.crs)],
                    ignore_index=True,
                )
                self.logger.info(f"✅ Added {len(new_rows)} nearest-cell assignments")

        # Verify weight sums
        weight_sums = relative_area.groupby(hru_id_col)['normalized_relative_area'].sum()
        bad_sums = weight_sums[~np.isclose(weight_sums, 1.0, atol=0.001)]
        if len(bad_sums) > 0:
            self.logger.warning(f"⚠️ {len(bad_sums)} HRUs have weight sums ≠ 1.0")
        else:
            self.logger.info("✅ All HRU weight sums equal 1.0")

        self._write_gridweights(HRU, relative_area, hru_id_col)
        ds.close()
        self.logger.info("TPHiPr grid weights generation complete!")
        return relative_area

    #---------------------------------------------------------------------------------

    def _write_gridweights(
        self,
        hru: gpd.GeoDataFrame,
        relative_area: gpd.GeoDataFrame,
        hru_id_col: str,
    ) -> None:
        """Write GridWeights_TPHiPr.txt."""
        number_HRUs = len(hru)
        number_cells = getattr(self, 'total_grid_cells', len(relative_area))
        HRU_list = list(relative_area[hru_id_col])
        cell_id = list(relative_area['cell_id'])
        rel_area = list(relative_area['normalized_relative_area'])

        filename = self.shared_data_dir / 'GridWeights_TPHiPr.txt'
        self.logger.info(f"Writing {filename}")

        with open(filename, 'w') as ff:
            ff.write('# ---------------------------------------------- \n')
            ff.write('# Raven GridWeights file for TPHiPr data         \n')
            ff.write('# Generated by TPHiPrGridWeightsGenerator         \n')
            ff.write(f'# Catchment: {self.gauge_id}                    \n')
            ff.write(f'# Model type: {self.model_type}                 \n')
            ff.write('# ---------------------------------------------- \n')
            ff.write('\n')
            ff.write(':GridWeights                     \n')
            ff.write('   #                                \n')
            ff.write('   # [# HRUs]                       \n')
            ff.write('   :NumberHRUs       {}            \n'.format(number_HRUs))
            ff.write('   :NumberGridCells       {}            \n'.format(number_cells))
            ff.write('   #                                \n')
            ff.write('   # [HRU ID] [Cell #] [w_kl]       \n')
            for i in range(len(relative_area)):
                ff.write("   {}   {}   {}\n".format(HRU_list[i], cell_id[i], rel_area[i]))
            ff.write(':EndGridWeights \n')

        self.logger.info(f"✅ GridWeights_TPHiPr.txt written ({len(relative_area)} rows)")


#--------------------------------------------------------------------------------
######################## process_tphipr_precipitation ###########################
#--------------------------------------------------------------------------------

def process_tphipr_precipitation(namelist_path: Union[str, Path]) -> None:
    """
    Convenience wrapper: run TPHiPrAnalyzer and TPHiPrGridWeightsGenerator.

    Parameters
    ----------
    namelist_path : str or Path
        Path to namelist YAML file.
    """
    print("🌧️ Processing TPHiPr precipitation data…")
    analyzer = TPHiPrAnalyzer(namelist_path)
    analyzer.process()

    print("🧮 Generating TPHiPr grid weights…")
    gw_gen = TPHiPrGridWeightsGenerator(namelist_path)
    gw_gen.generate()

    print("✅ TPHiPr processing completed!")