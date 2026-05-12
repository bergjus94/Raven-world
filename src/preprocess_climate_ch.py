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

import xarray as xr
import yaml

import warnings
warnings.filterwarnings('ignore')

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
    return results
