#!/usr/bin/env python3
"""
preprocess_climate.py  —  Bias-correct CORDEX WAS-44 projections to the
ERA5-Land 0.1° grid using xclim Quantile Delta Mapping (QDM).

Workflow (per variable/scenario)
---------------------------------
1. Load CORDEX file (clipped, from download_cordex.py)
2. Convert units:  K → °C  |  kg m⁻² s⁻¹ → mm day⁻¹
3. Regrid CORDEX rotated-pole → ERA5-Land 0.1° regular grid  (xesmf bilinear)
4. Lapse-rate elevation correction for temperature
       T_out = T_regridded + 0.0065 × (orog_CORDEX_on_ERA5grid − ERA5_elevation)
5. Train QDM on training period (default 1980–2005, monthly grouping)
       ref  = ERA5-Land  |  hist = CORDEX historical  (both on ERA5 grid)
6. Apply QDM to scenario data (historical or rcp26/45/85)
7. Save ERA5-Land-compatible NetCDF to  data_obs/

Output file convention
-----------------------
  cordex_{model}_{scenario}_temp_mean.nc   variable 't2m'  units '°C'
  cordex_{model}_{scenario}_temp_min.nc    variable 't2m'  units '°C'
  cordex_{model}_{scenario}_temp_max.nc    variable 't2m'  units '°C'
  cordex_{model}_{scenario}_precip.nc      variable 'tp'   units 'mm/day'

These are drop-in replacements for the ERA5-Land files.  The model
preprocessors (preprocess_HBV.py etc.) switch to them when
  future: true
  cordex_models: [...]
  cordex_scenarios: [...]
are set in the namelist YAML.

Required packages
-----------------
  xesmf   (conda install -c conda-forge xesmf)
  xclim   (conda install -c conda-forge xclim)
  dask    (conda install -c conda-forge dask)
"""

import logging
import traceback
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Optional heavy dependencies ─────────────────────────────────────────────

try:
    import xesmf as xe
    _XESMF_AVAILABLE = True
except ImportError:
    xe = None
    _XESMF_AVAILABLE = False

try:
    # xsdba is the standalone package split from xclim >= 0.60
    from xsdba import QuantileDeltaMapping
    _XCLIM_AVAILABLE = True
except ImportError:
    try:
        from xclim.sdba import QuantileDeltaMapping
        _XCLIM_AVAILABLE = True
    except ImportError:
        QuantileDeltaMapping = None
        _XCLIM_AVAILABLE = False


# ===========================================================================
# Constants
# ===========================================================================

_LAPSE_RATE      = 0.0065   # K m⁻¹ (standard environmental lapse rate)
_QDM_NQUANTILES  = 20
_TRAINING_START  = "1980-01-01"
_TRAINING_END    = "2005-12-31"
_TIME_CHUNK      = 365       # days per dask chunk

# CORDEX variable names that correspond to each variable type
_CORDEX_VAR: dict[str, str] = {
    "temp_mean": "tas",
    "temp_min":  "tasmin",
    "temp_max":  "tasmax",
    "precip":    "pr",
}

# ERA5-Land variable name used in output files
_ERA5_OUTPUT_VAR: dict[str, str] = {
    "temp_mean": "t2m",
    "temp_min":  "t2m",
    "temp_max":  "t2m",
    "precip":    "tp",
}

# ERA5-Land source file names (in data_obs/)
_ERA5_SOURCE_FILE: dict[str, str] = {
    "temp_mean": "era5_land_temp_mean.nc",
    "temp_min":  "era5_land_temp_min.nc",
    "temp_max":  "era5_land_temp_max.nc",
    "precip":    "era5_land_precip.nc",
}

# QDM kind: '+' = additive (temperature), '*' = multiplicative (precipitation)
_QDM_KIND: dict[str, str] = {
    "temp_mean": "+",
    "temp_min":  "+",
    "temp_max":  "+",
    "precip":    "*",
}


# ===========================================================================
# Logging helper
# ===========================================================================

def _setup_logger(name: str, debug: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger


# ===========================================================================
# CORDEXDownscaler
# ===========================================================================

class CORDEXDownscaler:
    """
    Downscale one CORDEX model to ERA5-Land 0.1° resolution using xclim QDM.

    Parameters
    ----------
    namelist_path : str or Path
        Raven-world namelist YAML for the target catchment.
    model_id : str
        GloGEM/Raven-world model ID (e.g. 'SMHI-RCA_MPIESM').
    force_reprocess : bool
        Re-compute even if output files already exist.
    """

    def __init__(
        self,
        namelist_path: "str | Path",
        model_id: str,
        force_reprocess: bool = False,
    ) -> None:
        if not _XESMF_AVAILABLE:
            raise ImportError(
                "xesmf is required for CORDEX regridding.\n"
                "  conda install -c conda-forge xesmf"
            )
        if not _XCLIM_AVAILABLE:
            raise ImportError(
                "xsdba (or xclim) is required for QDM bias correction.\n"
                "  conda install -c conda-forge xsdba xclim"
            )

        self.namelist_path  = Path(namelist_path)
        self.model_id       = model_id
        self.force_reprocess = force_reprocess

        with open(self.namelist_path) as f:
            self.config = yaml.safe_load(f)

        self.gauge_id  = self.config["gauge_id"]
        self.main_dir  = Path(self.config["main_dir"])
        self.model_dir = self.main_dir / self.config["config_dir"]
        self.shared_data_dir = (
            self.model_dir / f"catchment_{self.gauge_id}" / "data_obs"
        )
        self.shared_data_dir.mkdir(parents=True, exist_ok=True)

        cordex_dir = self.config.get("cordex_dir")
        if not cordex_dir:
            raise ValueError(
                "namelist must contain a 'cordex_dir' key pointing to the "
                "downloaded CORDEX data directory."
            )
        self.cordex_dir = Path(cordex_dir)

        self.train_start = pd.Timestamp(
            self.config.get("cordex_train_start", _TRAINING_START)
        )
        self.train_end = pd.Timestamp(
            self.config.get("cordex_train_end", _TRAINING_END)
        )

        self.debug  = self.config.get("debug", False)
        self.logger = _setup_logger(
            f"CORDEXDownscaler_{self.gauge_id}_{model_id}", self.debug
        )
        self.logger.info(f"CORDEXDownscaler  gauge={self.gauge_id}  model={model_id}")
        self.logger.info(f"  CORDEX dir  : {self.cordex_dir}")
        self.logger.info(f"  Output dir  : {self.shared_data_dir}")
        self.logger.info(
            f"  Train period: {self.train_start.date()} – {self.train_end.date()}"
        )

        # Lazy-built caches
        self._regridder: Optional[object] = None   # xesmf.Regridder
        self._era5_target_grid: Optional[xr.Dataset] = None
        self._era5_elev: Optional[xr.DataArray] = None
        self._cordex_orog_on_era5: Optional[xr.DataArray] = None

    # ── ERA5-Land target grid ────────────────────────────────────────────────

    def _get_era5_target_grid(self) -> xr.Dataset:
        """Return a minimal xesmf target Dataset from an ERA5-Land file."""
        if self._era5_target_grid is not None:
            return self._era5_target_grid

        # Find any existing ERA5-Land file to read the grid from
        for vtype in ("precip", "temp_mean", "temp_min"):
            era5_file = self.shared_data_dir / _ERA5_SOURCE_FILE[vtype]
            if era5_file.exists():
                break
        else:
            raise FileNotFoundError(
                f"No ERA5-Land file found in {self.shared_data_dir}. "
                "Run ERA5-Land preprocessing (preprocess_meteo.py) first."
            )

        with xr.open_dataset(era5_file) as ds:
            # Standardise to 'lat'/'lon' for xesmf
            lat_coord = "latitude" if "latitude" in ds.coords else "lat"
            lon_coord = "longitude" if "longitude" in ds.coords else "lon"
            lat = ds[lat_coord].values
            lon = ds[lon_coord].values

        # xesmf expects 'lat' and 'lon'
        self._era5_target_grid = xr.Dataset({
            "lat": xr.DataArray(lat, dims=["lat"],
                                attrs={"units": "degrees_north",
                                       "axis": "Y"}),
            "lon": xr.DataArray(lon, dims=["lon"],
                                attrs={"units": "degrees_east",
                                       "axis": "X"}),
        })
        return self._era5_target_grid

    # ── Elevation helpers ────────────────────────────────────────────────────

    def _load_era5_elevation(self) -> Optional[xr.DataArray]:
        """Return ERA5-Land surface elevation (m) on the ERA5 grid."""
        if self._era5_elev is not None:
            return self._era5_elev

        elev_path = self.shared_data_dir / "era5_land_elevation.nc"
        if not elev_path.exists():
            self.logger.warning(
                "era5_land_elevation.nc not found – lapse-rate correction skipped."
            )
            return None

        with xr.open_dataset(elev_path) as ds:
            if "elevation" in ds:
                da = ds["elevation"].load()
            elif "z" in ds:
                da = (ds["z"] / 9.80665).load()
                da.attrs["units"] = "m"
            else:
                self.logger.warning(
                    "No recognised elevation variable in era5_land_elevation.nc"
                )
                return None

        self._era5_elev = da
        return da

    def _load_cordex_orog_on_era5(self) -> Optional[xr.DataArray]:
        """Return CORDEX orography (m) regridded to the ERA5-Land grid."""
        if self._cordex_orog_on_era5 is not None:
            return self._cordex_orog_on_era5

        orog_path = self.cordex_dir / self.model_id / "orog.nc"
        if not orog_path.exists():
            self.logger.warning(
                f"CORDEX orog.nc not found at {orog_path} – "
                "lapse-rate correction skipped."
            )
            return None

        era5_elev = self._load_era5_elevation()
        if era5_elev is None:
            return None

        with xr.open_dataset(orog_path) as ds:
            orog_var = "orog" if "orog" in ds else list(ds.data_vars)[0]
            da_orog  = ds[orog_var].load()

        self.logger.info("Regridding CORDEX orog to ERA5-Land grid…")
        orog_on_era5 = self._regrid(da_orog)
        self._cordex_orog_on_era5 = orog_on_era5
        return orog_on_era5

    # ── xesmf regridder ──────────────────────────────────────────────────────

    def _build_regridder(self, da_sample: xr.DataArray) -> "xe.Regridder":
        """
        Build (and cache) an xesmf bilinear regridder from the CORDEX grid
        to the ERA5-Land grid.

        Parameters
        ----------
        da_sample : DataArray
            A representative CORDEX DataArray (any time step; used only for
            the spatial grid structure).
        """
        tgt = self._get_era5_target_grid()

        # Build a source Dataset that xesmf can read the 2-D lat/lon from.
        # CORDEX clipped files store geographic lat/lon as 2-D auxiliary
        # coordinates; xesmf picks them up automatically when they are
        # present as data variables named 'lat' and 'lon'.
        one_step = da_sample.isel(time=0).drop_vars("time", errors="ignore")
        src_ds = one_step.to_dataset(name="_v")
        if "lat" not in src_ds and "lat" in one_step.coords:
            src_ds["lat"] = one_step.coords["lat"]
        if "lon" not in src_ds and "lon" in one_step.coords:
            src_ds["lon"] = one_step.coords["lon"]

        regridder = xe.Regridder(
            src_ds, tgt,
            method         = "bilinear",
            extrap_method  = "nearest_s2d",
            ignore_degenerate = True,
        )
        self.logger.debug("xesmf bilinear regridder built.")
        return regridder

    def _regrid(self, da: xr.DataArray) -> xr.DataArray:
        """
        Regrid *da* from the CORDEX rotated-pole grid to the ERA5-Land grid.

        Returns a DataArray with dims (…, lat, lon) renamed to
        (…, latitude, longitude) to match ERA5-Land convention.
        """
        if self._regridder is None:
            self._regridder = self._build_regridder(da)

        da_out = self._regridder(da)

        # Rename lat/lon → latitude/longitude to match ERA5-Land files
        rename = {}
        if "lat" in da_out.dims:
            rename["lat"] = "latitude"
        if "lon" in da_out.dims:
            rename["lon"] = "longitude"
        if rename:
            da_out = da_out.rename(rename)

        return da_out

    # ── Unit conversion ──────────────────────────────────────────────────────

    @staticmethod
    def _convert_units(da: xr.DataArray, variable_type: str) -> xr.DataArray:
        """Convert CORDEX raw units to ERA5-Land-compatible units (°C, mm/day)."""
        units = da.attrs.get("units", "")

        if variable_type in ("temp_mean", "temp_min", "temp_max"):
            # CORDEX stores temperature in Kelvin
            if units in ("K", "Kelvin", "kelvin"):
                da = da - 273.15
                da.attrs["units"] = "degC"
            # If already °C, just normalise the attribute
            elif units in ("degC", "°C", "C", "celsius"):
                da.attrs["units"] = "degC"

        elif variable_type == "precip":
            # CORDEX stores precip as kg m-2 s-1 → mm/day (×86400)
            if units in ("kg m-2 s-1", "kg m**-2 s**-1", "kgm-2s-1",
                         "kg/m2/s", "kg/(m2*s)"):
                da = da * 86400.0
                da.attrs["units"] = "mm/day"
                da = da.clip(min=0)
            elif units in ("mm/day", "mm day-1", "mm d-1"):
                da.attrs["units"] = "mm/day"

        return da

    # ── Lapse-rate correction ────────────────────────────────────────────────

    def _lapse_rate_correct(self, da_temp: xr.DataArray) -> xr.DataArray:
        """
        Adjust temperature for the elevation difference between the CORDEX
        coarse orography and the ERA5-Land fine orography.

            T_out = T_in + 0.0065 × (orog_CORDEX − orog_ERA5)   [°C]

        If elevation data is unavailable the DataArray is returned unchanged.
        """
        era5_elev   = self._load_era5_elevation()
        cordex_orog = self._load_cordex_orog_on_era5()

        if era5_elev is None or cordex_orog is None:
            return da_temp

        # Align spatial coordinates
        try:
            dz = cordex_orog - era5_elev   # positive → CORDEX is higher
        except Exception:
            self.logger.warning(
                "Could not align elevation arrays for lapse-rate correction; "
                "skipping correction."
            )
            return da_temp

        correction  = _LAPSE_RATE * dz   # °C  (positive → CORDEX higher → ERA5 warmer)
        da_out      = da_temp + correction
        da_out.attrs.update(da_temp.attrs)
        return da_out

    # ── ERA5-Land reference loader ────────────────────────────────────────────

    def _load_era5_ref(
        self,
        variable_type: str,
        start: Optional[pd.Timestamp] = None,
        end:   Optional[pd.Timestamp] = None,
    ) -> xr.DataArray:
        """
        Load an ERA5-Land DataArray for use as QDM reference.

        Returns a DataArray with dim names (time, latitude, longitude)
        and CF-compatible attributes.
        """
        era5_file = self.shared_data_dir / _ERA5_SOURCE_FILE[variable_type]
        if not era5_file.exists():
            raise FileNotFoundError(
                f"ERA5-Land reference not found: {era5_file}. "
                "Run preprocess_meteo.py first."
            )

        era5_var = _ERA5_OUTPUT_VAR[variable_type]
        ds = xr.open_dataset(era5_file, chunks={"time": _TIME_CHUNK})
        da = ds[era5_var]

        # Standardise dim names
        rename = {}
        if "lat" in da.dims:
            rename["lat"] = "latitude"
        if "lon" in da.dims:
            rename["lon"] = "longitude"
        if rename:
            da = da.rename(rename)

        if start or end:
            da = da.sel(time=slice(
                start.strftime("%Y-%m-%d") if start else None,
                end.strftime("%Y-%m-%d")   if end   else None,
            ))

        # CF attributes for xclim
        if variable_type == "precip":
            da.attrs.setdefault("units", "mm/day")
            da.attrs.setdefault("cell_methods", "time: sum within days")
            da.attrs["standard_name"] = "precipitation_flux"
        else:
            da.attrs.setdefault("units", "degC")
            da.attrs.setdefault("cell_methods", "time: mean within days")
            da.attrs["standard_name"] = "air_temperature"

        da.name = era5_var
        return da

    # ── CORDEX loader ────────────────────────────────────────────────────────

    def _load_cordex(
        self,
        variable_type: str,
        scenario: str,
        start: Optional[pd.Timestamp] = None,
        end:   Optional[pd.Timestamp] = None,
    ) -> xr.DataArray:
        """
        Load a CORDEX variable, convert units, regrid and (for T) lapse-correct.

        Returns a DataArray on the ERA5-Land grid with dim names
        (time, latitude, longitude) and CF attributes.
        """
        cordex_var = _CORDEX_VAR[variable_type]
        nc_path    = self.cordex_dir / self.model_id / scenario / f"{cordex_var}.nc"

        if not nc_path.exists():
            raise FileNotFoundError(
                f"CORDEX file not found: {nc_path}. "
                "Run download_cordex.py first."
            )

        ds = xr.open_dataset(nc_path, chunks={"time": _TIME_CHUNK})
        da = ds[cordex_var]

        if start or end:
            da = da.sel(time=slice(
                start.strftime("%Y-%m-%d") if start else None,
                end.strftime("%Y-%m-%d")   if end   else None,
            ))

        # Unit conversion (operates lazily on dask arrays)
        da = self._convert_units(da, variable_type)

        # Regrid to ERA5 grid
        self.logger.debug(f"  Regridding {cordex_var} ({scenario})…")
        da = self._regrid(da)

        # Lapse-rate correction for temperature
        if variable_type != "precip":
            da = self._lapse_rate_correct(da)

        # CF attributes for xclim
        era5_var = _ERA5_OUTPUT_VAR[variable_type]
        da.name  = era5_var
        if variable_type == "precip":
            da.attrs["units"]        = "mm/day"
            da.attrs["cell_methods"] = "time: sum within days"
            da.attrs["standard_name"] = "precipitation_flux"
        else:
            da.attrs["units"]        = "degC"
            da.attrs["cell_methods"] = "time: mean within days"
            da.attrs["standard_name"] = "air_temperature"

        return da

    # ── QDM training and application ─────────────────────────────────────────

    def _train_qdm(
        self,
        ref:  xr.DataArray,
        hist: xr.DataArray,
        variable_type: str,
    ) -> "QuantileDeltaMapping":
        """Train a QDM on the training period."""
        kind = _QDM_KIND[variable_type]
        self.logger.info(
            f"  Training QDM (kind='{kind}', nq={_QDM_NQUANTILES}, "
            f"group=time.month, period={self.train_start.year}–{self.train_end.year})"
        )
        qdm = QuantileDeltaMapping.train(
            ref        = ref,
            hist       = hist,
            nquantiles = _QDM_NQUANTILES,
            kind       = kind,
            group      = "time.month",
        )
        return qdm

    # ── Output ───────────────────────────────────────────────────────────────

    def _output_path(self, variable_type: str, scenario: str) -> Path:
        model_safe = self.model_id.replace("/", "_").replace(" ", "_")
        return self.shared_data_dir / f"cordex_{model_safe}_{scenario}_{variable_type}.nc"

    def _save_output(
        self,
        da:            xr.DataArray,
        out_path:      Path,
        variable_type: str,
    ) -> None:
        """Write QDM-corrected DataArray as ERA5-Land-compatible NetCDF."""
        era5_var = _ERA5_OUTPUT_VAR[variable_type]
        da       = da.rename(era5_var)

        # Standardise dim names to (latitude, longitude, time)
        rename = {}
        for d in da.dims:
            if d in ("lat", "y") and d != "latitude":
                rename[d] = "latitude"
            elif d in ("lon", "x") and d != "longitude":
                rename[d] = "longitude"
        if rename:
            da = da.rename(rename)

        long_names = {
            "temp_mean": "2 metre temperature (daily mean) — CORDEX QDM-corrected",
            "temp_min":  "2 metre temperature (daily minimum) — CORDEX QDM-corrected",
            "temp_max":  "2 metre temperature (daily maximum) — CORDEX QDM-corrected",
            "precip":    "Total precipitation — CORDEX QDM-corrected",
        }
        da.attrs["long_name"] = long_names[variable_type]
        da.attrs["source"]    = (
            f"CORDEX WAS-44 {self.model_id}, bias-corrected with xclim QDM "
            f"(ref: ERA5-Land {self.train_start.year}–{self.train_end.year})"
        )

        ds = da.to_dataset()
        encoding: dict = {
            era5_var: {
                "zlib":     True,
                "complevel": 4,
                "dtype":    "float32",
            },
        }
        for coord in ("latitude", "longitude"):
            if coord in ds:
                encoding[coord] = {"_FillValue": None}

        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"  Saving {out_path.name} …")
        ds.to_netcdf(out_path, encoding=encoding)
        size_mb = out_path.stat().st_size / 1e6
        self.logger.info(f"  ✅ Saved: {out_path.name}  ({size_mb:.1f} MB)")

    # ── High-level entry points ───────────────────────────────────────────────

    def process_variable(self, variable_type: str, scenario: str) -> Path:
        """
        Full QDM pipeline for one variable / scenario combination.

        Steps
        -----
        1. Load ERA5-Land reference (training period)
        2. Load CORDEX historical (training period, regridded + corrected)
        3. Align training period
        4. Train QDM
        5. Load scenario data (full period)
        6. Apply QDM
        7. Save

        Returns
        -------
        Path to the output NetCDF.
        """
        out_path = self._output_path(variable_type, scenario)
        if out_path.exists() and not self.force_reprocess:
            size_mb = out_path.stat().st_size / 1e6
            self.logger.info(
                f"  ✅ Exists: {out_path.name}  ({size_mb:.1f} MB)  (skip)"
            )
            return out_path

        self.logger.info(
            f"\n{'─'*56}\n"
            f"  {variable_type}  |  {scenario}  |  {self.model_id}\n"
            f"{'─'*56}"
        )

        # 1 & 2: Load reference and historical data (training period)
        self.logger.info("  Loading ERA5-Land reference (training period)…")
        era5_train = self._load_era5_ref(
            variable_type, self.train_start, self.train_end
        )

        self.logger.info("  Loading CORDEX historical (training period)…")
        cordex_train = self._load_cordex(
            variable_type, "historical",
            self.train_start, self.train_end,
        )

        # 3: Align to common time period (may differ by a few days at boundaries)
        t_start = max(
            pd.Timestamp(era5_train.time.values[0]),
            pd.Timestamp(cordex_train.time.values[0]),
        )
        t_end = min(
            pd.Timestamp(era5_train.time.values[-1]),
            pd.Timestamp(cordex_train.time.values[-1]),
        )
        era5_train   = era5_train.sel(
            time=slice(t_start.strftime("%Y-%m-%d"), t_end.strftime("%Y-%m-%d"))
        )
        cordex_train = cordex_train.sel(
            time=slice(t_start.strftime("%Y-%m-%d"), t_end.strftime("%Y-%m-%d"))
        )
        n_train = len(era5_train.time)
        self.logger.info(
            f"  Aligned training period: {t_start.date()} – {t_end.date()} "
            f"({n_train} days)"
        )

        # 4: Train QDM
        qdm = self._train_qdm(era5_train, cordex_train, variable_type)

        # 5: Load scenario data (full period)
        self.logger.info(f"  Loading CORDEX {scenario} (full period)…")
        cordex_sim = self._load_cordex(variable_type, scenario)

        # 6: Apply QDM
        self.logger.info(f"  Applying QDM…")
        corrected = qdm.adjust(cordex_sim)
        if variable_type == "precip":
            corrected = corrected.clip(min=0)

        # 7: Save
        self._save_output(corrected, out_path, variable_type)
        return out_path

    def process_all(
        self,
        scenarios: Optional[list] = None,
        variables: Optional[list] = None,
    ) -> dict:
        """
        Process all variable / scenario combinations for this model.

        Parameters
        ----------
        scenarios : list of str, optional
            Defaults to namelist key 'cordex_scenarios' or ['historical','rcp45','rcp85'].
        variables : list of str, optional
            Defaults to all four: ['temp_mean','temp_min','temp_max','precip'].

        Returns
        -------
        dict mapping (variable_type, scenario) → output Path
        """
        if scenarios is None:
            scenarios = self.config.get(
                "cordex_scenarios", ["historical", "rcp45", "rcp85"]
            )
        if variables is None:
            variables = list(_CORDEX_VAR.keys())

        results: dict = {}
        for vtype in variables:
            for scen in scenarios:
                cordex_var = _CORDEX_VAR[vtype]
                src_file   = self.cordex_dir / self.model_id / scen / f"{cordex_var}.nc"
                if not src_file.exists():
                    self.logger.warning(
                        f"  ⏭️  Skip {vtype}/{scen}: "
                        f"source not found ({src_file.name})"
                    )
                    continue
                try:
                    out_path = self.process_variable(vtype, scen)
                    results[(vtype, scen)] = out_path
                except Exception as exc:
                    self.logger.error(f"  ❌ {vtype}/{scen}: {exc}")
                    self.logger.debug(traceback.format_exc())

        return results


# ===========================================================================
# Top-level entry point for create_input_files.py
# ===========================================================================

def process_cordex_climate(
    namelist_path: "str | Path",
    force_reprocess: bool = False,
) -> dict:
    """
    Downscale CORDEX projections for all models listed in the namelist.

    Called from create_input_files.py when ``future: true`` is set.

    Reads from namelist
    -------------------
    cordex_dir       : str   root dir of downloaded CORDEX data
    cordex_models    : list  GloGEM model IDs to process
    cordex_scenarios : list  e.g. ['historical', 'rcp45', 'rcp85']
    cordex_variables : list  optional subset; default = all four

    Returns
    -------
    dict  { model_id : { (variable_type, scenario) : Path } }
    """
    with open(namelist_path) as f:
        config = yaml.safe_load(f)

    models    = config.get("cordex_models", [])
    scenarios = config.get("cordex_scenarios", ["historical", "rcp45", "rcp85"])
    variables = config.get("cordex_variables", list(_CORDEX_VAR.keys()))

    if not models:
        raise ValueError(
            "namelist must contain 'cordex_models' list when future=True.\n"
            "Example:\n"
            "  cordex_models:\n"
            "    - SMHI-RCA_MPIESM\n"
        )

    all_results: dict = {}
    for model_id in models:
        print(f"\n{'='*60}")
        print(f"  CORDEX downscaling  |  model={model_id}")
        print(f"  scenarios : {scenarios}")
        print(f"  variables : {variables}")
        print(f"{'='*60}\n")

        downscaler = CORDEXDownscaler(
            namelist_path   = namelist_path,
            model_id        = model_id,
            force_reprocess = force_reprocess,
        )
        all_results[model_id] = downscaler.process_all(
            scenarios = scenarios,
            variables = variables,
        )

    return all_results
