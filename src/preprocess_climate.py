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
    from xsdba.processing import jitter_under_thresh as _jitter_under_thresh
    _XCLIM_AVAILABLE = True
except ImportError:
    try:
        from xclim.sdba import QuantileDeltaMapping
        from xclim.sdba.processing import jitter_under_thresh as _jitter_under_thresh
        _XCLIM_AVAILABLE = True
    except ImportError:
        QuantileDeltaMapping = None
        _jitter_under_thresh = None
        _XCLIM_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import matplotlib.gridspec as _gridspec
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _plt = None
    _gridspec = None
    _MATPLOTLIB_AVAILABLE = False

# Precipitation jitter threshold: values below this are perturbed to avoid
# multiplicative QDM division-by-zero when CORDEX has many dry days.
_PRECIP_JITTER_THRESH = "0.1 mm d-1"


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

        # The elevation file may cover a larger domain or use a different grid
        # than the catchment-clipped ERA5 files. Interpolate it to the target
        # ERA5 grid so that coordinate alignment works correctly.
        tgt = self._get_era5_target_grid()
        lat_dim = "latitude" if "latitude" in da.dims else "lat"
        lon_dim = "longitude" if "longitude" in da.dims else "lon"
        da = da.interp(
            {lat_dim: tgt["lat"].values, lon_dim: tgt["lon"].values},
            method="linear",
        )
        # Rename to standard (latitude, longitude) after interp
        rename = {}
        if "lat" in da.dims and "lat" != "latitude":
            rename["lat"] = "latitude"
        if "lon" in da.dims and "lon" != "longitude":
            rename["lon"] = "longitude"
        if rename:
            da = da.rename(rename)

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
        if "time" in da_sample.dims:
            one_step = da_sample.isel(time=0).drop_vars("time", errors="ignore")
        else:
            one_step = da_sample
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
                da = da.clip(min=0)
            # Normalise all precip unit strings to 'mm d-1' (CF-recognised rate unit)
            da.attrs["units"] = "mm d-1"

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

        # CF attributes for xclim — always override to ensure consistent units
        if variable_type == "precip":
            da.attrs["units"]        = "mm d-1"   # ERA5 file stores 'mm'; normalise to rate
            da.attrs["cell_methods"] = "time: sum within days"
            da.attrs["standard_name"] = "precipitation_flux"
        else:
            da.attrs["units"]        = "degC"
            da.attrs["cell_methods"] = "time: mean within days"
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

        # Normalise CORDEX timestamps to midnight (CORDEX often stores noon T12:00:00,
        # ERA5-Land uses midnight T00:00:00 — they must match for xsdba alignment)
        try:
            midnight_times = da.indexes["time"].normalize()
            da = da.assign_coords(time=midnight_times)
        except AttributeError:
            pass  # cftime index — handle below if needed

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

        # CF attributes for xclim — always override to ensure consistent units
        era5_var = _ERA5_OUTPUT_VAR[variable_type]
        da.name  = era5_var
        if variable_type == "precip":
            da.attrs["units"]        = "mm d-1"
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
        # xsdba requires a single chunk along the adjustment dimension (time)
        ref  = ref.chunk({"time": -1})
        hist = hist.chunk({"time": -1})
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
            "temp_mean": "2 metre temperature (daily mean) - CORDEX QDM-corrected",
            "temp_min":  "2 metre temperature (daily minimum) - CORDEX QDM-corrected",
            "temp_max":  "2 metre temperature (daily maximum) - CORDEX QDM-corrected",
            "precip":    "Total precipitation - CORDEX QDM-corrected",
        }
        da.attrs["long_name"] = long_names[variable_type]
        da.attrs["source"]    = (
            f"CORDEX WAS-44 {self.model_id}, bias-corrected with xclim QDM "
            f"(ref: ERA5-Land {self.train_start.year}-{self.train_end.year})"
        )

        # Drop stray non-spatial scalar coordinates (e.g. 'number' from ERA5)
        keep = {"time", "latitude", "longitude", "lat", "lon"}
        extra = [c for c in da.coords if c not in keep]
        if extra:
            da = da.drop_vars(extra, errors="ignore")

        # Also remove 'coordinates' attr if it references a now-dropped variable
        # (e.g. ERA5 sets coordinates='number' on tp/t2m; Raven fails if the
        # referenced variable is absent from the output file).
        da.attrs.pop("coordinates", None)

        # Ensure standard dimension order (time, latitude, longitude) to match
        # the ERA5-Land files that Raven expects.
        target_dims = [d for d in ("time", "latitude", "longitude") if d in da.dims]
        if list(da.dims) != target_dims:
            da = da.transpose(*target_dims)

        ds = da.to_dataset()

        # Add ERA5-Land elevation to the output file so that Raven can apply
        # its internal lapse-rate correction from the grid-cell elevation to
        # individual HRU elevations (required by :ElevationVarNameNC elevation
        # in the .rvt file).
        era5_elev = self._load_era5_elevation()
        if era5_elev is not None:
            # Drop any non-spatial coords (e.g. 'number') before merging
            elev_clean = era5_elev.drop_vars(
                [c for c in era5_elev.coords
                 if c not in ("latitude", "longitude")],
                errors="ignore",
            )
            ds["elevation"] = elev_clean
        else:
            self.logger.warning(
                "ERA5-Land elevation not available — "
                "elevation variable omitted from CORDEX output file."
            )

        encoding: dict = {
            era5_var: {
                "zlib":     True,
                "complevel": 4,
                "dtype":    "float32",
            },
        }
        if "elevation" in ds:
            encoding["elevation"] = {"zlib": True, "complevel": 4,
                                     "dtype": "float32", "_FillValue": None}
        for coord in ("latitude", "longitude"):
            if coord in ds:
                encoding[coord] = {"_FillValue": None}

        # Final scrub before writing:
        # 1. Remove 'coordinates' attr — xarray may regenerate it from encoding
        #    even after da.attrs.pop(), causing Raven to look up 'number' which
        #    doesn't exist in the CORDEX output file.
        # 2. Replace non-ASCII characters in all string attributes — unicode chars
        #    (em-dash, superscript 2, etc.) force NetCDF4 to use NC_STRING type
        #    instead of NC_CHAR, which Raven's libnetcdf cannot handle.
        def _ascii_safe(s: str) -> str:
            return s.encode("ascii", errors="replace").decode("ascii")

        for var in list(ds.data_vars) + list(ds.coords):
            obj = ds[var]
            obj.attrs.pop("coordinates", None)
            obj.encoding.pop("coordinates", None)
            for k, v in list(obj.attrs.items()):
                if isinstance(v, str) and not v.isascii():
                    obj.attrs[k] = _ascii_safe(v)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"  Saving {out_path.name} …")
        ds.to_netcdf(out_path, encoding=encoding)
        size_mb = out_path.stat().st_size / 1e6
        self.logger.info(f"  ✅ Saved: {out_path.name}  ({size_mb:.1f} MB)")

    # ── Diagnostic plots ──────────────────────────────────────────────────────

    def _save_diagnostic_plots(
        self,
        variable_type: str,
        scenario: str,
        era5_train: xr.DataArray,
        cordex_raw_train: xr.DataArray,
        corrected_full: xr.DataArray,
    ) -> Optional[Path]:
        """
        Save a multi-panel diagnostic figure for one variable / scenario.

        Layout (3 rows × 3 cols, 16 × 12 in)
        ─────────────────────────────────────
        Row 0  Monthly climatology [cols 0-1] │ QQ plot            [col 2]
        Row 1  Daily time series 2-yr zoom    │ Wet-day / bias     [col 2]
               [cols 0-1]                     │
        Row 2  Spatial mean maps: ERA5 | CORDEX raw | corrected
               [col 0]            [col 1]     │ [col 2]

        Parameters
        ----------
        era5_train, cordex_raw_train
            Training-period DataArrays (pre-jitter), dims (time, latitude, longitude).
        corrected_full
            Full-period QDM-corrected DataArray — sliced to training period internally
            for the climatology / QQ / spatial panels.
        """
        if not _MATPLOTLIB_AVAILABLE:
            self.logger.warning(
                "matplotlib not available — diagnostic plots skipped."
            )
            return None

        self.logger.info(
            f"  Generating diagnostic plots ({variable_type} / {scenario})…"
        )

        model_safe = self.model_id.replace("/", "_").replace(" ", "_")
        catchment_dir = self.model_dir / f"catchment_{self.gauge_id}"
        plot_dir = catchment_dir / "plots" / "future_climate"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path  = plot_dir / f"cordex_{model_safe}_{scenario}_{variable_type}.png"

        unit_label = "mm/day" if variable_type == "precip" else "°C"
        var_title  = {
            "temp_mean": "Mean Temperature",
            "temp_min":  "Min Temperature",
            "temp_max":  "Max Temperature",
            "precip":    "Precipitation",
        }[variable_type]

        # ── Compute spatial means (lazy → numpy) ─────────────────────────────
        sp_dims = [d for d in ("latitude", "longitude") if d in era5_train.dims]
        era5_sm = era5_train.mean(dim=sp_dims).compute()
        raw_sm  = cordex_raw_train.mean(dim=sp_dims).compute()

        # Slice corrected to training period
        t0 = pd.Timestamp(era5_train.time.values[0]).strftime("%Y-%m-%d")
        t1 = pd.Timestamp(era5_train.time.values[-1]).strftime("%Y-%m-%d")
        try:
            corr_sm = (
                corrected_full
                .sel(time=slice(t0, t1))
                .mean(dim=sp_dims)
                .compute()
            )
        except Exception:
            corr_sm = corrected_full.mean(dim=sp_dims).compute()

        # ── Spatial time-mean maps ────────────────────────────────────────────
        era5_map = era5_train.mean(dim="time").compute()
        raw_map  = cordex_raw_train.mean(dim="time").compute()
        try:
            corr_map = (
                corrected_full
                .sel(time=slice(t0, t1))
                .mean(dim="time")
                .compute()
            )
        except Exception:
            corr_map = corrected_full.mean(dim="time").compute()

        lat = era5_train.latitude.values
        lon = era5_train.longitude.values

        # ── Monthly climatology ───────────────────────────────────────────────
        month_labels = ["J","F","M","A","M","J","J","A","S","O","N","D"]
        months     = np.arange(1, 13)
        era5_clim  = era5_sm.groupby("time.month").mean().values
        raw_clim   = raw_sm.groupby("time.month").mean().values
        corr_clim  = corr_sm.groupby("time.month").mean().values

        # ── QQ plot (daily spatial-mean values) ───────────────────────────────
        era5_sorted = np.sort(era5_sm.values.ravel())
        raw_sorted  = np.sort(raw_sm.values.ravel())
        corr_sorted = np.sort(corr_sm.values.ravel())
        n = min(len(era5_sorted), len(raw_sorted), len(corr_sorted))
        step = max(1, n // 2000)   # subsample for plot clarity
        era5_qq = era5_sorted[:n:step]
        raw_qq  = raw_sorted[:n:step]
        corr_qq = corr_sorted[:n:step]

        # ── Daily time series (first 2 years) ────────────────────────────────
        ts_end = (
            pd.Timestamp(era5_sm.time.values[0]) + pd.DateOffset(years=2)
        ).strftime("%Y-%m-%d")
        era5_ts = era5_sm.sel(time=slice(None, ts_end))
        raw_ts  = raw_sm.sel(time=slice(None, ts_end))
        corr_ts = corr_sm.sel(time=slice(None, ts_end))

        def _to_datetime_index(da):
            try:
                return pd.DatetimeIndex(da.time.values)
            except Exception:
                return np.arange(len(da.time))

        t_era5 = _to_datetime_index(era5_ts)
        t_raw  = _to_datetime_index(raw_ts)
        t_corr = _to_datetime_index(corr_ts)

        # ── Wet-day fraction (precip) or monthly bias (temperature) ──────────
        if variable_type == "precip":
            thresh = 1.0
            def _wet_frac(da):
                return (da > thresh).astype(float).groupby("time.month").mean().values
            era5_wet = _wet_frac(era5_sm)
            raw_wet  = _wet_frac(raw_sm)
            corr_wet = _wet_frac(corr_sm)
        else:
            raw_bias  = raw_clim  - era5_clim
            corr_bias = corr_clim - era5_clim

        # ── Build figure ──────────────────────────────────────────────────────
        fig = _plt.figure(figsize=(16, 12))
        gs  = _gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

        ax_monthly = fig.add_subplot(gs[0, :2])
        ax_qq      = fig.add_subplot(gs[0, 2])
        ax_ts      = fig.add_subplot(gs[1, :2])
        ax_extra   = fig.add_subplot(gs[1, 2])
        ax_m1      = fig.add_subplot(gs[2, 0])
        ax_m2      = fig.add_subplot(gs[2, 1])
        ax_m3      = fig.add_subplot(gs[2, 2])

        fig.suptitle(
            f"CORDEX QDM — {var_title}  |  {model_safe}  |  {scenario}"
            f"  |  catchment {self.gauge_id}",
            fontsize=12, y=1.005,
        )

        col_era5 = "black"
        col_raw  = "tab:red"
        col_corr = "tab:blue"

        # Panel A — monthly climatology
        ax_monthly.plot(months, era5_clim, "o-",  color=col_era5, lw=2,   ms=5,
                        label="ERA5-Land")
        ax_monthly.plot(months, raw_clim,  "^--", color=col_raw,  lw=1.5, ms=5,
                        label="CORDEX raw")
        ax_monthly.plot(months, corr_clim, "s-",  color=col_corr, lw=1.5, ms=5,
                        label="CORDEX corrected")
        ax_monthly.set_xticks(months)
        ax_monthly.set_xticklabels(month_labels)
        ax_monthly.set_ylabel(unit_label)
        ax_monthly.set_title("Monthly climatology (training period)")
        ax_monthly.legend(fontsize=9)
        ax_monthly.grid(True, alpha=0.3)

        # Panel B — QQ plot
        ax_qq.scatter(era5_qq, raw_qq,  s=5, c=col_raw,  alpha=0.5,
                      label="CORDEX raw", rasterized=True)
        ax_qq.scatter(era5_qq, corr_qq, s=5, c=col_corr, alpha=0.5,
                      label="CORDEX corrected", rasterized=True)
        _diag = [min(era5_qq.min(), raw_qq.min(), corr_qq.min()),
                 max(era5_qq.max(), raw_qq.max(), corr_qq.max())]
        ax_qq.plot(_diag, _diag, "k--", lw=1, label="1:1")
        ax_qq.set_xlabel(f"ERA5-Land  ({unit_label})")
        ax_qq.set_ylabel(f"CORDEX  ({unit_label})")
        ax_qq.set_title("QQ plot (spatial mean, training)")
        ax_qq.legend(fontsize=8)
        ax_qq.grid(True, alpha=0.3)

        # Panel C — daily time series
        ax_ts.plot(t_era5, era5_ts.values, color=col_era5, lw=1.2,
                   label="ERA5-Land", alpha=0.85)
        ax_ts.plot(t_raw,  raw_ts.values,  color=col_raw,  lw=0.8,
                   label="CORDEX raw", alpha=0.6)
        ax_ts.plot(t_corr, corr_ts.values, color=col_corr, lw=0.9,
                   label="CORDEX corrected", alpha=0.75)
        ax_ts.set_ylabel(unit_label)
        ax_ts.set_title("Daily spatial-mean time series (first 2 yr of training)")
        ax_ts.legend(fontsize=9)
        ax_ts.grid(True, alpha=0.3)

        # Panel D — wet-day fraction or monthly bias
        x = np.arange(12)
        if variable_type == "precip":
            w = 0.25
            ax_extra.bar(x - w, era5_wet, w, color=col_era5, alpha=0.75, label="ERA5")
            ax_extra.bar(x,     raw_wet,  w, color=col_raw,  alpha=0.75, label="CORDEX raw")
            ax_extra.bar(x + w, corr_wet, w, color=col_corr, alpha=0.75, label="Corrected")
            ax_extra.set_xticks(x)
            ax_extra.set_xticklabels(month_labels, fontsize=8)
            ax_extra.set_ylabel("Wet-day fraction  (> 1 mm/d)")
            ax_extra.set_title("Wet-day statistics")
            ax_extra.legend(fontsize=8)
            ax_extra.grid(True, axis="y", alpha=0.3)
        else:
            w = 0.35
            ax_extra.bar(x - w/2, raw_bias,  w, color=col_raw,  alpha=0.8,
                         label="CORDEX raw bias")
            ax_extra.bar(x + w/2, corr_bias, w, color=col_corr, alpha=0.8,
                         label="Corrected bias")
            ax_extra.axhline(0, color="k", lw=1)
            ax_extra.set_xticks(x)
            ax_extra.set_xticklabels(month_labels, fontsize=8)
            ax_extra.set_ylabel(f"Bias vs ERA5  ({unit_label})")
            ax_extra.set_title("Monthly bias")
            ax_extra.legend(fontsize=8)
            ax_extra.grid(True, axis="y", alpha=0.3)

        # Panels E-G — spatial mean maps
        cmap = "YlOrRd" if variable_type == "precip" else "RdBu_r"
        all_vals = np.concatenate([
            era5_map.values.ravel(),
            raw_map.values.ravel(),
            corr_map.values.ravel(),
        ])
        all_vals = all_vals[np.isfinite(all_vals)]
        vmin, vmax = (
            (np.percentile(all_vals, 2), np.percentile(all_vals, 98))
            if len(all_vals) > 0 else (0, 1)
        )

        def _map_panel(ax, data_da, title):
            _lat = data_da.latitude.values if "latitude" in data_da.coords else lat
            _lon = data_da.longitude.values if "longitude" in data_da.coords else lon
            im = ax.pcolormesh(
                _lon, _lat, data_da.values,
                cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
            )
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Lon", fontsize=8)
            ax.set_ylabel("Lat", fontsize=8)
            ax.tick_params(labelsize=7)
            return im

        _map_panel(ax_m1, era5_map,  f"ERA5 mean ({unit_label})")
        _map_panel(ax_m2, raw_map,   f"CORDEX raw mean ({unit_label})")
        im3 = _map_panel(ax_m3, corr_map, f"Corrected mean ({unit_label})")
        _plt.colorbar(im3, ax=[ax_m1, ax_m2, ax_m3], shrink=0.65,
                      label=unit_label, pad=0.02)

        # ── Save ──────────────────────────────────────────────────────────────
        fig.savefig(plot_path, dpi=120, bbox_inches="tight")
        _plt.close(fig)
        self.logger.info(f"  📊 Diagnostic plot saved: {plot_path}")
        return plot_path

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
        # Save pre-jitter copies for diagnostic plots (jitter modifies the arrays).
        era5_train_orig  = era5_train
        cordex_train_orig = cordex_train

        # For precipitation, apply jitter to near-zero values before multiplicative
        # QDM to avoid division-by-zero when CORDEX has more dry days than ERA5.
        if variable_type == "precip" and _jitter_under_thresh is not None:
            self.logger.info(
                f"  Applying jitter (thresh={_PRECIP_JITTER_THRESH}) for precip QDM…"
            )
            era5_train    = _jitter_under_thresh(era5_train,    _PRECIP_JITTER_THRESH)
            cordex_train  = _jitter_under_thresh(cordex_train,  _PRECIP_JITTER_THRESH)

        qdm = self._train_qdm(era5_train, cordex_train, variable_type)

        # 5: Load scenario data (full period)
        self.logger.info(f"  Loading CORDEX {scenario} (full period)…")
        cordex_sim = self._load_cordex(variable_type, scenario)

        # 6: Apply QDM — xsdba requires a single chunk along the time dimension
        self.logger.info(f"  Applying QDM…")
        cordex_sim = cordex_sim.chunk({"time": -1})
        if variable_type == "precip" and _jitter_under_thresh is not None:
            cordex_sim = _jitter_under_thresh(cordex_sim, _PRECIP_JITTER_THRESH)
        corrected = qdm.adjust(cordex_sim)
        if variable_type == "precip":
            corrected = corrected.clip(min=0)

        # 6b: Prepend ERA5-Land warmup data if ERA5 starts before CORDEX.
        # This fills the gap so Raven can initialise model storages (snowpack,
        # soil moisture, groundwater) before the CORDEX projection period begins.
        # ERA5 and QDM-corrected CORDEX share the same statistical distribution,
        # so the concatenation at the CORDEX start date is seamless.
        era5_full = self._load_era5_ref(variable_type)
        era5_full_start  = pd.Timestamp(era5_full.time.values[0])
        cordex_data_start = pd.Timestamp(corrected.time.values[0])

        if era5_full_start < cordex_data_start:
            warmup_end = (
                cordex_data_start - pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d")
            era5_warmup = era5_full.sel(time=slice(None, warmup_end))

            # Drop any scalar coords that would block concatenation (e.g. 'number')
            keep_coords = {"time", "latitude", "longitude", "lat", "lon"}
            extra = [c for c in era5_warmup.coords if c not in keep_coords]
            if extra:
                era5_warmup = era5_warmup.drop_vars(extra, errors="ignore")

            n_warmup = len(era5_warmup.time)
            self.logger.info(
                f"  Prepending ERA5 warmup: {era5_full_start.date()} → "
                f"{warmup_end}  ({n_warmup} days)"
            )
            corrected = xr.concat(
                [era5_warmup.chunk({"time": -1}),
                 corrected.chunk({"time": -1})],
                dim="time",
            )

        # 7: Save
        self._save_output(corrected, out_path, variable_type)

        # 8: Diagnostic plots (non-fatal — never break the pipeline)
        try:
            self._save_diagnostic_plots(
                variable_type, scenario,
                era5_train_orig, cordex_train_orig, corrected,
            )
        except Exception as _plot_exc:
            self.logger.warning(f"  Diagnostic plots skipped: {_plot_exc}")

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
