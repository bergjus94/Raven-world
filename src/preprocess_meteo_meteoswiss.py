#### MeteoSwiss daily gridded products preprocessor for the Raven pipeline.
#### Ports Raven-Switzerland/src/preprocess_meteo.py onto the composable
#### MeteoBase architecture used by ERA5-Land / HAR / TPHiPr.
####
#### Source files (EPSG:2056 Swiss LV95, dims N/E, daily 1981-2020):
####   RhiresD — daily precipitation (mm/day)
####   TabsD   — daily mean temperature (°C)
####   TmaxD   — daily maximum temperature (°C)
####   TminD   — daily minimum temperature (°C)
####
#### Output files written to shared_data_dir:
####   prec_Meteoswiss.nc
####   temp_Meteoswiss.nc
####   temp_max_Meteoswiss.nc
####   temp_min_Meteoswiss.nc
####
#### Justine Berg

from pathlib import Path
from typing import Dict, List, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon

import warnings
warnings.filterwarnings('ignore')

from paths import get_paths
from preprocess_meteo_base import MeteoBase


#--------------------------------------------------------------------------------
############################### MeteoSwissAnalyzer ##############################
#--------------------------------------------------------------------------------

class MeteoSwissAnalyzer(MeteoBase):
    """
    Clip MeteoSwiss daily gridded products to catchment extent and time range.

    The MeteoSwiss NetCDFs already carry 2-D lat/lon coordinates (WGS84) alongside
    the projected E/N dims, so spatial clipping is done against the catchment
    bounding box in WGS84 without any reprojection.
    """

    _logger_class_name = 'MeteoSwissAnalyzer'
    _csv_prefix = 'meteoswiss_'

    # Mapping: (source variable in NetCDF) -> (output filename)
    VARIABLES: Dict[str, str] = {
        'RhiresD': 'prec_Meteoswiss.nc',
        'TabsD':   'temp_Meteoswiss.nc',
        'TmaxD':   'temp_max_Meteoswiss.nc',
        'TminD':   'temp_min_Meteoswiss.nc',
    }

    def __init__(self, namelist_path: Union[str, Path], force_reprocess: bool = False) -> None:
        super().__init__(namelist_path, force_reprocess)

        # Source directory for MeteoSwiss NetCDFs
        meteoswiss_dir = self.config.get('meteoswiss_dir', '01_data/meteo/Switzerland')
        self.meteoswiss_dir = Path(meteoswiss_dir)
        if not self.meteoswiss_dir.is_absolute():
            self.meteoswiss_dir = self.main_dir / self.meteoswiss_dir

        # Forcing filenames: either from namelist (list of 4) or auto-discover
        self.forcing_files = self._resolve_forcing_files()

        self.processed_files: List[Path] = []

        self.logger.info(f"MeteoSwissAnalyzer initialized for gauge {self.gauge_id}")
        self.logger.info(f"MeteoSwiss source directory: {self.meteoswiss_dir}")

    #---------------------------------------------------------------------------------

    def _resolve_forcing_files(self) -> Dict[str, Path]:
        """
        Return a dict mapping source variable names to absolute input file paths.

        Preference:
          1. `forcing_files` list in namelist (matched to VARIABLES by substring).
          2. Auto-discovery by glob in `meteoswiss_dir` using the variable name
             prefix (e.g. RhiresD_*.nc).
        """
        resolved: Dict[str, Path] = {}

        forcing_list = self.config.get('forcing_files') or []
        if forcing_list:
            for var in self.VARIABLES:
                match = next(
                    (Path(f) for f in forcing_list if var in Path(f).name),
                    None,
                )
                if match is not None:
                    if not match.is_absolute():
                        match = self.main_dir / match
                    resolved[var] = match

        # Fill any gaps via glob (prefer the *_elev.nc variant when available)
        for var in self.VARIABLES:
            if var in resolved:
                continue
            candidates = sorted(self.meteoswiss_dir.glob(f'{var}_*.nc'))
            elev_variants = [c for c in candidates if '_elev' in c.name]
            if elev_variants:
                resolved[var] = elev_variants[0]
            elif candidates:
                resolved[var] = candidates[0]
            else:
                self.logger.warning(
                    f"No MeteoSwiss input found for '{var}' under {self.meteoswiss_dir}"
                )

        return resolved

    #---------------------------------------------------------------------------------

    def process(self) -> List[Path]:
        """
        Clip all four MeteoSwiss variables and (optionally) add warm-up padding.

        Returns
        -------
        List[Path]
            Paths to the saved NetCDF files.
        """
        self.logger.info(
            f"Processing {len(self.VARIABLES)} MeteoSwiss variables for gauge {self.gauge_id}"
        )

        saved_files: List[Path] = []
        for var, out_name in self.VARIABLES.items():
            out_file = self.shared_data_dir / out_name

            if var not in self.forcing_files:
                self.logger.error(f"  ✗ {var}: no source file available – skipping")
                continue

            if out_file.exists() and not self.force_reprocess:
                self.logger.info(f"  ✅ {out_name} already exists – skipping")
                saved_files.append(out_file)
                continue

            try:
                self._clip_and_save(
                    source_file=self.forcing_files[var],
                    var_name=var,
                    out_file=out_file,
                )
                saved_files.append(out_file)
            except Exception as e:
                self.logger.error(f"  ✗ Failed to process {var}: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())

        self.processed_files = saved_files

        # Warm-up period handled by the shared base-class method (same approach
        # as ERA5 and HAR: repeat first year and prepend to simulation data).
        if self.warmup_date is not None and saved_files:
            self._add_warmup_to_files(saved_files)

        # Monthly climatologies for HBV (base class; uses _csv_prefix).
        if self.model_type and self.model_type.upper() in ('HBV', 'UBCWM'):
            self._calculate_monthly_temperature_averages_meteoswiss()

        self.logger.info(f"✅ MeteoSwiss processing complete ({len(saved_files)} files)")
        return saved_files

    #---------------------------------------------------------------------------------

    def _clip_and_save(self, source_file: Path, var_name: str, out_file: Path) -> None:
        """
        Clip one MeteoSwiss NetCDF to catchment bbox + time range and save.

        Uses the 2-D `lat`/`lon` coordinates that MeteoSwiss files carry
        alongside the projected E/N dims. No reprojection required.
        """
        self.logger.info(f"  Processing {var_name}: {source_file.name}")

        if not source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")

        ds = xr.open_dataset(source_file)

        # Keep only the main variable + its elevation companion (if present) to
        # avoid carrying the bulky `swiss_lv95_coordinates` auxiliary array.
        keep = [v for v in (var_name, 'elevation') if v in ds.data_vars]
        ds = ds[keep]

        # Spatial clip via 2-D lat/lon mask against catchment bounding box + buffer
        if self.catchment_extent is not None and 'lat' in ds.coords and 'lon' in ds.coords:
            minx, miny, maxx, maxy = self.catchment_extent.total_bounds
            buf = 0.05  # ~5 km — MeteoSwiss grid is ~2 km, 2-3 cell buffer
            lat_mask = (ds['lat'] >= miny - buf) & (ds['lat'] <= maxy + buf)
            lon_mask = (ds['lon'] >= minx - buf) & (ds['lon'] <= maxx + buf)
            mask_2d = lat_mask & lon_mask

            # The mask shares the 2-D (N, E) dims — collapse to per-axis 1-D
            # selectors so we can slice E and N independently and preserve the
            # regular grid shape.
            n_keep = mask_2d.any(dim='E')
            e_keep = mask_2d.any(dim='N')
            ds = ds.isel(N=n_keep.values, E=e_keep.values)
            self.logger.info(
                f"    Clipped to catchment: N={ds.sizes['N']}, E={ds.sizes['E']}"
            )

        # Temporal slice. For method='real' the base-class helper widens the
        # clip to include the warm-up window so pre-simulation data is kept;
        # for method='cycle' the clip stays at simulation start and synthetic
        # warm-up is prepended later by `_add_warmup_to_files`.
        effective_start = self._effective_start_date()
        ds = ds.sel(time=slice(effective_start, self.end_date))
        if ds.sizes.get('time', 0) == 0:
            raise ValueError(
                f"No time steps in {source_file.name} after slicing to "
                f"[{effective_start.date()}, {self.end_date.date()}]"
            )
        self.logger.info(
            f"    Time range: {pd.Timestamp(ds.time.values[0]).date()} → "
            f"{pd.Timestamp(ds.time.values[-1]).date()} ({ds.sizes['time']} days)"
        )

        # Strip grid_mapping attrs that can confuse NetCDF writers
        for v in ds.data_vars:
            ds[v].attrs.pop('grid_mapping', None)

        ds.attrs.update({
            'processed_by': 'MeteoSwissAnalyzer',
            'gauge_id': str(self.gauge_id),
            'creation_date': pd.Timestamp.now().isoformat(),
        })

        out_file.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out_file)
        ds.close()
        self.logger.info(f"    ✅ Saved {out_file.name}")

    #---------------------------------------------------------------------------------

    def _calculate_monthly_temperature_averages_meteoswiss(self) -> None:
        """
        Compute monthly mean temperature climatology from temp_Meteoswiss.nc and
        save to `{_csv_prefix}monthly_temperature_averages.csv` in output_path.

        The shared MeteoBase.calculate_monthly_temperature_averages() searches
        for files with 'temp_mean' in the name (ERA5/HAR convention), so we
        provide a MeteoSwiss-specific implementation here.
        """
        temp_file = self.shared_data_dir / 'temp_Meteoswiss.nc'
        if not temp_file.exists():
            self.logger.warning(
                f"temp_Meteoswiss.nc not found – skipping monthly temperature averages"
            )
            return

        self.logger.info("Calculating MeteoSwiss monthly temperature averages…")
        try:
            ds = xr.open_dataset(temp_file)
            temp_var = next((v for v in ds.data_vars if v == 'TabsD'), None)
            if temp_var is None:
                temp_var = next(
                    (v for v in ds.data_vars if 'time' in ds[v].dims and v != 'elevation'),
                    None,
                )
            if temp_var is None:
                self.logger.warning("No temperature variable found – skipping")
                ds.close()
                return

            spatial_dims = [d for d in ds[temp_var].dims if d != 'time']
            ts = ds[temp_var].mean(dim=spatial_dims, skipna=True)
            monthly = ts.groupby('time.month').mean().to_dataframe().reset_index()
            monthly = monthly.rename(columns={temp_var: 'Temperature'})
            monthly = (
                pd.DataFrame({'month': list(range(1, 13))})
                .merge(monthly, on='month', how='left')
            )
            if monthly['Temperature'].isna().any():
                monthly['Temperature'] = monthly['Temperature'].interpolate()
            monthly['Temperature'] = monthly['Temperature'].round(2)

            out = self.output_path / f'{self._csv_prefix}monthly_temperature_averages.csv'
            monthly.to_csv(out, index=False)
            self.logger.info(f"  ✅ Saved {out.name}")
            ds.close()
        except Exception as e:
            self.logger.error(f"Monthly temperature averages failed: {e}")


#--------------------------------------------------------------------------------
####################### MeteoSwissGridWeightsGenerator ##########################
#--------------------------------------------------------------------------------

class MeteoSwissGridWeightsGenerator(MeteoBase):
    """
    Build a Raven GridWeights file for the MeteoSwiss N/E grid using the 2-D
    lat/lon coordinates that the NetCDF carries.  Mirrors the HAR generator.

    Output: GridWeights_MeteoSwiss.txt in the HRU `topo` directory.
    """

    _logger_class_name = 'MeteoSwissGridWeightsGenerator'

    def __init__(self, namelist_path: Union[str, Path], force_reprocess: bool = False) -> None:
        super().__init__(namelist_path, force_reprocess)

        paths = get_paths(self.config)
        self.out_dir = self.shared_data_dir
        self.gridweights_dir = paths['topo_dir']
        self.out_HRU_shape_dir = paths['topo_dir'] / 'HRU.shp'

        self.logger.info(
            f"MeteoSwissGridWeightsGenerator initialized for gauge {self.gauge_id}"
        )

    #---------------------------------------------------------------------------------

    def generate(self) -> gpd.GeoDataFrame:
        """
        Generate GridWeights_MeteoSwiss.txt from prec_Meteoswiss.nc as grid ref.
        """
        gridweights_file = self.gridweights_dir / 'GridWeights_MeteoSwiss.txt'
        if gridweights_file.exists() and not self.force_reprocess:
            self.logger.info("✅ GridWeights_MeteoSwiss.txt already exists – skipping")
            return gpd.GeoDataFrame()

        # Grid reference: the clipped precip file
        grid_file = self.out_dir / 'prec_Meteoswiss.nc'
        if not grid_file.exists():
            raise FileNotFoundError(
                f"prec_Meteoswiss.nc not found at {grid_file}. "
                "Run MeteoSwissAnalyzer.process() first."
            )

        self.logger.info(f"Loading MeteoSwiss grid from {grid_file}")
        ds = xr.open_dataset(grid_file)

        if 'N' not in ds.dims or 'E' not in ds.dims:
            raise ValueError(
                f"Expected N/E dims in {grid_file.name}, got {list(ds.dims)}"
            )
        ny = ds.sizes['N']
        nx = ds.sizes['E']
        self.total_grid_cells = ny * nx
        self.logger.info(f"Grid: {ny} N × {nx} E = {self.total_grid_cells} cells")

        if 'lat' not in ds.coords or 'lon' not in ds.coords:
            raise ValueError("MeteoSwiss file missing 2-D lat/lon coordinates")
        lat_2d = ds.coords['lat'].values
        lon_2d = ds.coords['lon'].values

        # Half-cell sizes from median spacing along each axis (robust to NaNs)
        dlat = float(np.nanmedian(np.abs(np.diff(lat_2d, axis=0)))) / 2 if ny > 1 else 0.01
        dlon = float(np.nanmedian(np.abs(np.diff(lon_2d, axis=1)))) / 2 if nx > 1 else 0.015
        if not np.isfinite(dlat) or dlat == 0:
            dlat = 0.01     # ~1 km fallback
        if not np.isfinite(dlon) or dlon == 0:
            dlon = 0.015
        self.logger.info(f"Cell half-size: dlat={dlat:.5f}°, dlon={dlon:.5f}°")

        # Load HRUs
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

        # Build per-cell polygons (catchment extent only, for speed)
        hru_minx, hru_miny, hru_maxx, hru_maxy = HRU_wgs84.total_bounds
        buf_lat = dlat * 4
        buf_lon = dlon * 4

        polygons: List[Polygon] = []
        cell_ids: List[str] = []

        for j in range(ny):
            for i in range(nx):
                cx = float(lon_2d[j, i])
                cy = float(lat_2d[j, i])
                if not (np.isfinite(cx) and np.isfinite(cy)):
                    continue
                if (cx < hru_minx - buf_lon or cx > hru_maxx + buf_lon or
                        cy < hru_miny - buf_lat or cy > hru_maxy + buf_lat):
                    continue
                corners = [
                    (cx - dlon, cy - dlat),
                    (cx + dlon, cy - dlat),
                    (cx + dlon, cy + dlat),
                    (cx - dlon, cy + dlat),
                ]
                polygons.append(Polygon(corners))
                cell_ids.append(str(j * nx + i))

        meteoswiss_grid = gpd.GeoDataFrame(
            {'cell_id': cell_ids, 'geometry': polygons}, crs='EPSG:4326'
        )
        self.logger.info(f"Built {len(meteoswiss_grid)} grid polygons (catchment extent)")

        # Overlay with HRUs
        self.logger.info("Computing HRU–grid intersection…")
        res_union = HRU_wgs84.overlay(meteoswiss_grid, how='intersection')

        # Relative areas (equal-area projection)
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

        # Fill missing HRUs with nearest grid cell (weight = 1)
        all_hru_ids = set(HRU_wgs84[hru_id_col].values)
        hrus_with_weights = set(relative_area[hru_id_col].values)
        missing = all_hru_ids - hrus_with_weights
        if missing:
            self.logger.warning(
                f"⚠️ {len(missing)} HRUs have no grid overlap; assigning nearest cell"
            )
            new_rows = []
            for hid in missing:
                hru_geom = HRU_wgs84[HRU_wgs84[hru_id_col] == hid].geometry.values[0]
                distances = meteoswiss_grid.geometry.centroid.distance(hru_geom.centroid)
                nearest_cell_id = meteoswiss_grid.loc[distances.idxmin(), 'cell_id']
                new_rows.append({
                    hru_id_col: hid,
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

        weight_sums = relative_area.groupby(hru_id_col)['normalized_relative_area'].sum()
        bad_sums = weight_sums[~np.isclose(weight_sums, 1.0, atol=0.001)]
        if len(bad_sums) > 0:
            self.logger.warning(f"⚠️ {len(bad_sums)} HRUs have weight sums ≠ 1.0")
        else:
            self.logger.info("✅ All HRU weight sums equal 1.0")

        self._write_gridweights(HRU, relative_area, hru_id_col)
        ds.close()
        self.logger.info("MeteoSwiss grid weights generation complete!")
        return relative_area

    #---------------------------------------------------------------------------------

    def _write_gridweights(
        self,
        hru: gpd.GeoDataFrame,
        relative_area: gpd.GeoDataFrame,
        hru_id_col: str,
    ) -> None:
        number_HRUs = len(hru)
        number_cells = getattr(self, 'total_grid_cells', len(relative_area))
        HRU_list = list(relative_area[hru_id_col])
        cell_id = list(relative_area['cell_id'])
        rel_area = list(relative_area['normalized_relative_area'])

        filename = self.gridweights_dir / 'GridWeights_MeteoSwiss.txt'
        self.logger.info(f"Writing {filename}")

        with open(filename, 'w') as ff:
            ff.write('# ---------------------------------------------- \n')
            ff.write('# Raven GridWeights file for MeteoSwiss data     \n')
            ff.write('# Generated by MeteoSwissGridWeightsGenerator    \n')
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

        self.logger.info(f"✅ GridWeights_MeteoSwiss.txt written ({len(relative_area)} rows)")


#--------------------------------------------------------------------------------
######################## process_meteoswiss_forcing #############################
#--------------------------------------------------------------------------------

def process_meteoswiss_forcing(namelist_path: Union[str, Path]) -> None:
    """
    Convenience wrapper: run MeteoSwissAnalyzer and MeteoSwissGridWeightsGenerator.
    """
    print("🇨🇭 Processing MeteoSwiss meteorological data…")
    analyzer = MeteoSwissAnalyzer(namelist_path)
    analyzer.process()

    print("🧮 Generating MeteoSwiss grid weights…")
    gw = MeteoSwissGridWeightsGenerator(namelist_path)
    gw.generate()

    print("✅ MeteoSwiss processing completed!")
