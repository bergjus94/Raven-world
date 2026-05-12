#### CH2018 (and later CH2025) climate-projection preprocessing for Swiss
#### catchments.  CH2018 / CH2025 are already QM-bias-corrected to MeteoSwiss
#### observations, so this is a pass-through clip + time-slice — NO QDM,
#### NO regridding to a different observation grid.
####
#### Phase 1: CH2018 transient scenarios (this file)
####   - File pattern: CH2018/QMgrid/<var>_elev/
####                   CH2018_<var>_<RCM>_<GCM>_EUR11_<RCP>_QMgrid_1981-2099.nc
####   - vars: pr (mm/day), tas / tasmax / tasmin (°C)
####   - grid: EPSG:2056 (CH1903+ / LV95), dims (time, y, x), ~2 km
####   - time: proleptic_gregorian calendar, 1981-01-01 → 2099-12-31
####
#### Phase 2: CH2025 GWL-based scenarios (deferred to separate class)
####
#### Output files: <shared_data_dir>/ch2018_<model>_<scenario>_<vtype>.nc
####   where vtype ∈ {precip, temp_mean, temp_max, temp_min}
####
#### Source variable names (pr/tas/tasmax/tasmin) and projected (y, x)
#### dims are preserved — Raven-side renaming happens at the .rvt
#### assembly step.
####
#### Justine Berg

from pathlib import Path
from typing import Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from shapely.geometry import Polygon

import warnings
warnings.filterwarnings('ignore')

from paths import get_paths
from preprocess_meteo_base import MeteoBase


#--------------------------------------------------------------------------------
############################### CH2018PassThrough ###############################
#--------------------------------------------------------------------------------

class CH2018PassThrough(MeteoBase):
    """
    Clip CH2018 transient climate projections to catchment + time range.

    CH2018 is already QM-bias-corrected to MeteoSwiss observations on the
    ~2 km Swiss LV95 grid (EPSG:2056).  No QDM, no regridding here — just
    spatial + temporal clipping.
    """

    _logger_class_name = 'CH2018PassThrough'

    # CH2018 source var → output filename suffix (matches ERA5/CORDEX convention)
    VAR_MAP: Dict[str, str] = {
        'pr':     'precip',
        'tas':    'temp_mean',
        'tasmax': 'temp_max',
        'tasmin': 'temp_min',
    }

    def __init__(
        self,
        namelist_path: Union[str, Path],
        model_id:      str,
        scenario:      str,
        force_reprocess: bool = False,
    ) -> None:
        super().__init__(namelist_path, force_reprocess)

        self.model_id = model_id    # e.g. "SMHI-RCA_ECEARTH"
        self.scenario = scenario    # e.g. "RCP85"

        ch2018_dir = self.config.get('ch2018_dir')
        if not ch2018_dir:
            raise ValueError(
                "`ch2018_dir` must be set in namelist when future.source='CH2018'"
            )
        self.ch2018_dir = Path(ch2018_dir)
        if not self.ch2018_dir.exists():
            raise FileNotFoundError(
                f"CH2018 root not found: {self.ch2018_dir} "
                "(check SMB mount or local path)"
            )

        # Spatial buffer around catchment bbox (in metres, EPSG:2056).
        # CH2018 grid is ~1.7 km, so 5 km ≈ 3 cells.
        self.spatial_buffer_m = float(self.config.get('ch2018_spatial_buffer_m', 5000))

        self.logger.info(
            f"CH2018PassThrough initialized | model={model_id} | scenario={scenario}"
        )

    #---------------------------------------------------------------------------------

    def _source_file(self, var: str) -> Path:
        """Resolve the CH2018 source NetCDF for a given variable."""
        return (
            self.ch2018_dir
            / f'{var}_elev'
            / f'CH2018_{var}_{self.model_id}_EUR11_{self.scenario}_QMgrid_1981-2099.nc'
        )

    #---------------------------------------------------------------------------------

    def _output_file(self, var_type: str) -> Path:
        """Resolve the output NetCDF for a given variable type."""
        model_safe = self.model_id.replace('/', '_').replace(' ', '_')
        return (
            self.shared_data_dir
            / f'ch2018_{model_safe}_{self.scenario}_{var_type}.nc'
        )

    #---------------------------------------------------------------------------------

    def _catchment_bbox_2056(self) -> tuple:
        """Catchment bbox (minx, miny, maxx, maxy) in EPSG:2056 metres."""
        if self.catchment_extent is None:
            raise RuntimeError("Catchment shapefile not loaded")
        ext = self.catchment_extent.to_crs("EPSG:2056")
        return tuple(ext.total_bounds)

    #---------------------------------------------------------------------------------

    def process_variable(self, var: str) -> Optional[Path]:
        """
        Clip one CH2018 variable file to the catchment bbox + time range.

        Parameters
        ----------
        var : str
            Source variable in CH2018 — one of pr / tas / tasmax / tasmin.

        Returns
        -------
        Path to the saved NetCDF, or None on failure.
        """
        if var not in self.VAR_MAP:
            raise KeyError(f"Unknown CH2018 variable '{var}', expected one of {list(self.VAR_MAP)}")
        var_type = self.VAR_MAP[var]
        out = self._output_file(var_type)

        if out.exists() and not self.force_reprocess:
            self.logger.info(f"  ✅ {out.name} exists, skipping")
            return out

        src = self._source_file(var)
        if not src.exists():
            self.logger.error(f"  ✗ source missing: {src}")
            return None

        self.logger.info(f"  Processing {var}: {src.name}")
        ds = xr.open_dataset(src)

        # Catchment bbox in EPSG:2056 + buffer
        minx, miny, maxx, maxy = self._catchment_bbox_2056()
        buf = self.spatial_buffer_m

        # CH2018 y axis is decreasing (high → low northing) — slice top-to-bottom.
        # x axis is increasing.
        y_increasing = bool(ds.y.values[1] > ds.y.values[0])
        y_slice = slice(miny - buf, maxy + buf) if y_increasing else slice(maxy + buf, miny - buf)

        clipped = ds.sel(
            x=slice(minx - buf, maxx + buf),
            y=y_slice,
        )
        if clipped.sizes.get('x', 0) == 0 or clipped.sizes.get('y', 0) == 0:
            self.logger.error(
                f"  ✗ {var}: empty after spatial clip — bbox or CRS mismatch?"
            )
            ds.close()
            return None

        # Time slice (xarray decodes proleptic_gregorian to standard datetime)
        clipped = clipped.sel(time=slice(self.start_date, self.end_date))
        if clipped.sizes.get('time', 0) == 0:
            self.logger.error(
                f"  ✗ {var}: empty after time slice "
                f"[{self.start_date.date()}, {self.end_date.date()}] — "
                f"check sim period vs CH2018 range (1981-2099)"
            )
            ds.close()
            return None

        # CH2018 timestamps are noon-anchored ("days since 1981-01-01 12:00:00").
        # Raven expects daily forcing at the start of the day for a 1-day timestep,
        # otherwise it errors with "gridded forcing data not available at beginning
        # of model simulation".  Shift to midnight.
        clipped = clipped.assign_coords(
            time=clipped['time'] - pd.Timedelta(hours=12)
        )

        # Drop spatial_ref (we keep the CRS via attrs on the data var instead),
        # but preserve elevation since it's useful for lapse-rate corrections.
        keep = [v for v in (var, 'elevation') if v in clipped.data_vars]
        clipped = clipped[keep]

        # Strip grid_mapping attrs to keep NetCDF writer happy
        for v in clipped.data_vars:
            clipped[v].attrs.pop('grid_mapping', None)

        # Provenance
        clipped.attrs.update({
            'processed_by': 'CH2018PassThrough',
            'source_file': src.name,
            'ch2018_model': self.model_id,
            'ch2018_scenario': self.scenario,
            'gauge_id': str(self.gauge_id),
            'crs': 'EPSG:2056 (CH1903+ / LV95)',
        })

        out.parent.mkdir(parents=True, exist_ok=True)
        clipped.to_netcdf(out)
        ds.close()

        self.logger.info(
            f"    ✅ {out.name} ({clipped.sizes['time']} days, "
            f"{clipped.sizes['y']}×{clipped.sizes['x']} cells)"
        )
        return out

    #---------------------------------------------------------------------------------

    def process_all(
        self,
        variables: Optional[List[str]] = None,
    ) -> Dict[str, Optional[Path]]:
        """
        Process all four (or a subset of) CH2018 variables.

        Returns
        -------
        dict   { var_name -> output Path or None }
        """
        if variables is None:
            variables = list(self.VAR_MAP.keys())

        self.logger.info(
            f"❄️  CH2018 clip | model={self.model_id} | scenario={self.scenario} "
            f"| {len(variables)} variables"
        )
        results: Dict[str, Optional[Path]] = {}
        for var in variables:
            try:
                results[var] = self.process_variable(var)
            except Exception as e:
                self.logger.error(f"  ✗ {var} failed: {e}")
                results[var] = None
        return results


#--------------------------------------------------------------------------------
########################### CH2018GridWeightsGenerator ##########################
#--------------------------------------------------------------------------------

class CH2018GridWeightsGenerator(MeteoBase):
    """
    Build a Raven GridWeights file for the CH2018 x/y projected grid (EPSG:2056).

    The CH2018 grid is regular ~1.7 km in CH1903+ / LV95 metres, with no 2-D
    lat/lon companion — overlay is done entirely in EPSG:2056, no reprojection
    of coordinates to WGS84 needed.

    Cell-id convention matches `:DimNamesNC x y time` in the .rvt:
        cell_id = y_idx * nx + x_idx     (x is the fastest-varying dim)

    Output: GridWeights_CH2018.txt in the HRU `topo` directory.
    """

    _logger_class_name = 'CH2018GridWeightsGenerator'

    def __init__(self, namelist_path: Union[str, Path], force_reprocess: bool = False) -> None:
        super().__init__(namelist_path, force_reprocess)

        paths = get_paths(self.config)
        self.gridweights_dir = paths['topo_dir']
        self.out_HRU_shape_dir = paths['topo_dir'] / 'HRU.shp'
        self.output_file = self.gridweights_dir / 'GridWeights_CH2018.txt'

        self.logger.info(
            f"CH2018GridWeightsGenerator initialized for gauge {self.gauge_id}"
        )

    #---------------------------------------------------------------------------------

    def generate(self, reference_nc: Path) -> gpd.GeoDataFrame:
        """
        Build GridWeights_CH2018.txt from a clipped CH2018 NetCDF as grid reference.

        Parameters
        ----------
        reference_nc : Path
            Path to any of the four ch2018_*.nc files produced by
            CH2018PassThrough — they all share the same grid.
        """
        if self.output_file.exists() and not self.force_reprocess:
            self.logger.info(
                f"✅ {self.output_file.name} already exists — skipping"
            )
            return gpd.GeoDataFrame()

        if not reference_nc.exists():
            raise FileNotFoundError(
                f"CH2018 reference NetCDF not found: {reference_nc}. "
                "Run CH2018PassThrough.process_all() first."
            )

        ds = xr.open_dataset(reference_nc)
        if 'x' not in ds.dims or 'y' not in ds.dims:
            raise ValueError(
                f"Expected x/y dims in {reference_nc.name}, got {list(ds.dims)}"
            )
        nx, ny = ds.sizes['x'], ds.sizes['y']
        x = ds.x.values
        y = ds.y.values

        # Cell sizes from median spacing along each axis (robust to NaNs).
        # CH2018 is regular at ~1.7 km — use abs() in case y is decreasing.
        dx = float(np.nanmedian(np.abs(np.diff(x)))) if nx > 1 else 1737.0
        dy = float(np.nanmedian(np.abs(np.diff(y)))) if ny > 1 else 1737.0
        self.total_grid_cells = ny * nx
        self.logger.info(
            f"Grid: {ny} y × {nx} x = {self.total_grid_cells} cells "
            f"(dx={dx:.1f} m, dy={dy:.1f} m)"
        )

        # Load HRUs and reproject to EPSG:2056 for the overlay
        if not self.out_HRU_shape_dir.exists():
            raise FileNotFoundError(
                f"HRU shapefile not found: {self.out_HRU_shape_dir}"
            )
        HRU = gpd.read_file(self.out_HRU_shape_dir)
        if 'HRU_ID' in HRU.columns:
            HRU = HRU.sort_values(by='HRU_ID').reset_index(drop=True)
            HRU['HRU ID'] = HRU['HRU_ID']
        elif 'HRU ID' not in HRU.columns:
            HRU['HRU ID'] = list(range(1, len(HRU) + 1))
        hru_id_col = 'HRU ID' if 'HRU ID' in HRU.columns else 'HRU_ID'

        HRU_2056 = HRU.to_crs('EPSG:2056')
        HRU_2056['geometry'] = HRU_2056.geometry.buffer(0)  # heal invalid geoms
        self.logger.info(f"Loaded {len(HRU_2056)} HRUs (reprojected to EPSG:2056)")

        # Build cell polygons covering the catchment + buffer
        hru_minx, hru_miny, hru_maxx, hru_maxy = HRU_2056.total_bounds
        buf = max(dx, dy) * 4

        polygons: List[Polygon] = []
        cell_ids: List[str] = []
        half_dx, half_dy = dx / 2.0, dy / 2.0

        for j in range(ny):
            cy = float(y[j])
            if not np.isfinite(cy) or cy < hru_miny - buf or cy > hru_maxy + buf:
                continue
            for i in range(nx):
                cx = float(x[i])
                if not np.isfinite(cx) or cx < hru_minx - buf or cx > hru_maxx + buf:
                    continue
                polygons.append(Polygon([
                    (cx - half_dx, cy - half_dy),
                    (cx + half_dx, cy - half_dy),
                    (cx + half_dx, cy + half_dy),
                    (cx - half_dx, cy + half_dy),
                ]))
                # Match :DimNamesNC "x y time" → x is fastest
                cell_ids.append(str(j * nx + i))

        ch2018_grid = gpd.GeoDataFrame(
            {'cell_id': cell_ids, 'geometry': polygons}, crs='EPSG:2056'
        )
        self.logger.info(
            f"Built {len(ch2018_grid)} grid polygons (catchment + buffer)"
        )

        # Overlay HRUs × grid; areas are already in m² since both are EPSG:2056
        self.logger.info("Computing HRU–grid intersection…")
        res = HRU_2056.overlay(ch2018_grid, how='intersection')
        res['area'] = res.geometry.area

        hru_totals = res.groupby(hru_id_col)['area'].transform('sum')
        res['area_rel'] = np.where(hru_totals > 0, res['area'] / hru_totals, 0)
        res['area_rel'] = res['area_rel'].round(5)

        def _normalize(g):
            tot = g['area_rel'].sum()
            g['normalized_relative_area'] = (
                (g['area_rel'] / tot).round(5) if tot > 0 else 0
            )
            return g

        relative_area = res.groupby(hru_id_col, group_keys=False).apply(_normalize)

        # Fill missing HRUs (no grid overlap) with the nearest cell, weight=1
        all_hrus = set(HRU_2056[hru_id_col].values)
        seen = set(relative_area[hru_id_col].values)
        missing = all_hrus - seen
        if missing:
            self.logger.warning(
                f"⚠️ {len(missing)} HRUs have no CH2018-grid overlap "
                "(expected with ~1.7 km cells on small HRUs); assigning nearest cell"
            )
            new_rows = []
            grid_centroids = ch2018_grid.geometry.centroid
            for hid in missing:
                hru_geom = HRU_2056[HRU_2056[hru_id_col] == hid].geometry.values[0]
                distances = grid_centroids.distance(hru_geom.centroid)
                nearest_cell_id = ch2018_grid.loc[distances.idxmin(), 'cell_id']
                new_rows.append({
                    hru_id_col: hid,
                    'cell_id': nearest_cell_id,
                    'area_rel': 1.0,
                    'normalized_relative_area': 1.0,
                    'area': 0,
                    'geometry': hru_geom,
                })
            relative_area = pd.concat(
                [relative_area, gpd.GeoDataFrame(new_rows, crs='EPSG:2056')],
                ignore_index=True,
            )
            self.logger.info(f"✅ Added {len(new_rows)} nearest-cell assignments")

        # Sanity check
        sums = relative_area.groupby(hru_id_col)['normalized_relative_area'].sum()
        bad = sums[~np.isclose(sums, 1.0, atol=1e-3)]
        if len(bad) > 0:
            self.logger.warning(
                f"⚠️ {len(bad)} HRUs have weight sums ≠ 1.0 (max dev {abs(bad-1).max():.4f})"
            )
        else:
            self.logger.info("✅ All HRU weight sums equal 1.0")

        self._write_gridweights(HRU, relative_area, hru_id_col)
        ds.close()
        self.logger.info("CH2018 grid weights generation complete!")
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

        self.gridweights_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Writing {self.output_file}")
        with open(self.output_file, 'w') as ff:
            ff.write('# ---------------------------------------------- \n')
            ff.write('# Raven GridWeights file for CH2018 climate data \n')
            ff.write('# Generated by CH2018GridWeightsGenerator        \n')
            ff.write(f'# Catchment: {self.gauge_id}                    \n')
            ff.write(f'# Model type: {self.model_type}                 \n')
            ff.write('# Cell-id convention: y_idx * nx + x_idx         \n')
            ff.write('#   (matches :DimNamesNC "x y time" in .rvt)     \n')
            ff.write('# ---------------------------------------------- \n')
            ff.write('\n')
            ff.write(':GridWeights                     \n')
            ff.write('   #                                \n')
            ff.write('   # [# HRUs]                       \n')
            ff.write(f'   :NumberHRUs       {number_HRUs}            \n')
            ff.write(f'   :NumberGridCells       {number_cells}            \n')
            ff.write('   #                                \n')
            ff.write('   # [HRU ID] [Cell #] [w_kl]       \n')
            for _, row in relative_area.iterrows():
                ff.write(f"   {row[hru_id_col]}   {row['cell_id']}   "
                         f"{row['normalized_relative_area']}\n")
            ff.write(':EndGridWeights \n')

        self.logger.info(
            f"✅ {self.output_file.name} written ({len(relative_area)} rows)"
        )


#--------------------------------------------------------------------------------
############################### top-level entry #################################
#--------------------------------------------------------------------------------

def process_ch2018_climate(
    namelist_path: Union[str, Path],
    force_reprocess: bool = False,
) -> Dict[tuple, Dict[str, Optional[Path]]]:
    """
    Top-level entry point — call from create_input_files.py when
    namelist has `future.source == 'CH2018'`.

    Namelist keys consumed
    ----------------------
    ch2018_dir       : str  root dir of CH2018 QMgrid (under <var>_elev subfolders)
    ch2018_models    : list of '<RCM>_<GCM>' strings, e.g. ['SMHI-RCA_ECEARTH']
    ch2018_scenarios : list of RCP strings, e.g. ['RCP45', 'RCP85']
    ch2018_variables : list (optional)  subset of {pr, tas, tasmax, tasmin}

    Returns
    -------
    dict  { (model_id, scenario) -> { var -> output Path or None } }
    """
    with open(namelist_path) as f:
        config = yaml.safe_load(f)

    models = config.get('ch2018_models') or []
    scenarios = config.get('ch2018_scenarios') or []
    variables = config.get('ch2018_variables')   # None ⇒ all four

    if not models:
        raise ValueError(
            "CH2018 future run requires `ch2018_models` in namelist — "
            "e.g. ['SMHI-RCA_ECEARTH']"
        )
    if not scenarios:
        raise ValueError(
            "CH2018 future run requires `ch2018_scenarios` in namelist — "
            "e.g. ['RCP85']"
        )

    results: Dict[tuple, Dict[str, Optional[Path]]] = {}
    grid_weights_done = False
    for model_id in models:
        for scenario in scenarios:
            print(f"\n{'='*60}")
            print(f"  CH2018 clip | model={model_id} | scenario={scenario}")
            print(f"{'='*60}")
            proc = CH2018PassThrough(
                namelist_path=namelist_path,
                model_id=model_id,
                scenario=scenario,
                force_reprocess=force_reprocess,
            )
            results[(model_id, scenario)] = proc.process_all(variables=variables)

            # Generate the GridWeights once — all (model, scenario, var) combos
            # share the same CH2018 ~1.7 km grid.
            if not grid_weights_done:
                ref = next(
                    (p for p in results[(model_id, scenario)].values() if p is not None),
                    None,
                )
                if ref is not None:
                    print(f"\n  🧮 Generating GridWeights_CH2018.txt (one-time)…")
                    gw = CH2018GridWeightsGenerator(
                        namelist_path=namelist_path,
                        force_reprocess=force_reprocess,
                    )
                    gw.generate(reference_nc=ref)
                    grid_weights_done = True

    return results
