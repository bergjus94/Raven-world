#!/usr/bin/env python3
"""
preprocess_climate.py  —  Bias-correct climate projections to the
ERA5-Land 0.1° grid using xclim Quantile Delta Mapping (QDM).

Supported sources
-----------------
CORDEX WAS-44  (CMIP5-era RCM downscaling)
  - Input:  CORDEX files clipped by download_cordex.py
  - Class:  CORDEXDownscaler
  - Output: cordex_{model}_{scenario}_{vartype}.nc

CMIP6 GCMs  (raw ISIMIP3b-compatible models)
  - Input:  CMIP6 files downloaded by download_cmip6.py
  - Class:  CMIP6Downscaler
  - Output: cmip6_{model}_{scenario}_{vartype}.nc

Workflow (per variable/scenario)
---------------------------------
1. Load source file (CORDEX or CMIP6)
2. Convert units:  K → °C  |  kg m⁻² s⁻¹ → mm day⁻¹
3. Regrid to ERA5-Land 0.1° regular grid  (xesmf bilinear)
4. Lapse-rate elevation correction for temperature
       T_out = T_regridded + 0.0065 × (orog_source_on_ERA5grid − ERA5_elevation)
5. Train QDM on training period (monthly grouping)
       ref  = ERA5-Land  |  hist = source historical  (both on ERA5 grid)
6. Apply QDM to scenario data
7. Save ERA5-Land-compatible NetCDF to  data_obs/

Output file convention
-----------------------
  {prefix}_{model}_{scenario}_temp_mean.nc   variable 't2m'  units '°C'
  {prefix}_{model}_{scenario}_temp_min.nc    variable 't2m'  units '°C'
  {prefix}_{model}_{scenario}_temp_max.nc    variable 't2m'  units '°C'
  {prefix}_{model}_{scenario}_precip.nc      variable 'tp'   units 'mm/day'

These are drop-in replacements for the ERA5-Land files.  The model
preprocessors (preprocess_HBV.py etc.) switch to them when
  future: true
  cordex_models: [...]   or   cmip6_models: [...]
  cordex_scenarios: [...]      cmip6_scenarios: [...]
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

from scipy.sparse import csr_matrix


class _ScipyBilinearRegridder:
    """Drop-in replacement for xe.Regridder using precomputed bilinear weights.

    Builds a sparse weight matrix once at init, then applies it as a fast
    matrix multiply for each timestep.  ~100x faster than per-step interpolation.
    Works on standard rectilinear lat/lon grids (CMIP6).
    """

    def __init__(self, src_ds, tgt_ds, logger=None):
        # Target coordinates (ERA5-Land)
        self._tgt_lat = tgt_ds["latitude"].values if "latitude" in tgt_ds else tgt_ds["lat"].values
        self._tgt_lon = tgt_ds["longitude"].values if "longitude" in tgt_ds else tgt_ds["lon"].values
        self._logger = logger
        self._weights = None  # Lazy — built on first call when we know the source grid

    def _build_weights(self, src_lat, src_lon):
        """Precompute sparse bilinear weight matrix: (n_tgt, n_src)."""
        n_tgt = len(self._tgt_lat) * len(self._tgt_lon)
        n_src_lat = len(src_lat)
        n_src_lon = len(src_lon)
        n_src = n_src_lat * n_src_lon

        rows, cols, vals = [], [], []

        for i, tlat in enumerate(self._tgt_lat):
            for j, tlon in enumerate(self._tgt_lon):
                tgt_idx = i * len(self._tgt_lon) + j

                # Find bounding source indices
                ilat = np.searchsorted(src_lat, tlat) - 1
                ilon = np.searchsorted(src_lon, tlon) - 1

                # Clamp to valid range
                ilat = max(0, min(ilat, n_src_lat - 2))
                ilon = max(0, min(ilon, n_src_lon - 2))

                # Bilinear weights
                lat0, lat1 = src_lat[ilat], src_lat[ilat + 1]
                lon0, lon1 = src_lon[ilon], src_lon[ilon + 1]

                dlat = (lat1 - lat0) if lat1 != lat0 else 1.0
                dlon = (lon1 - lon0) if lon1 != lon0 else 1.0
                wlat = (tlat - lat0) / dlat
                wlon = (tlon - lon0) / dlon
                wlat = np.clip(wlat, 0, 1)
                wlon = np.clip(wlon, 0, 1)

                # 4 corner weights
                corners = [
                    (ilat,     ilon,     (1 - wlat) * (1 - wlon)),
                    (ilat,     ilon + 1, (1 - wlat) * wlon),
                    (ilat + 1, ilon,     wlat * (1 - wlon)),
                    (ilat + 1, ilon + 1, wlat * wlon),
                ]
                for ci, cj, w in corners:
                    if w > 0:
                        rows.append(tgt_idx)
                        cols.append(ci * n_src_lon + cj)
                        vals.append(w)

        self._weights = csr_matrix((vals, (rows, cols)), shape=(n_tgt, n_src))
        self._src_shape = (n_src_lat, n_src_lon)
        if self._logger:
            self._logger.debug(
                f"Scipy bilinear weights built: {n_src} src → {n_tgt} tgt "
                f"({self._weights.nnz} nonzeros)"
            )

    def __call__(self, da: xr.DataArray) -> xr.DataArray:
        # Normalise dim names
        rename_in = {}
        if "latitude" in da.dims:
            rename_in["latitude"] = "lat"
        if "longitude" in da.dims:
            rename_in["longitude"] = "lon"
        if rename_in:
            da = da.rename(rename_in)

        src_lat = da["lat"].values
        src_lon = da["lon"].values

        # Build weights on first call
        if self._weights is None:
            self._build_weights(src_lat, src_lon)

        n_tgt_lat = len(self._tgt_lat)
        n_tgt_lon = len(self._tgt_lon)

        if "time" in da.dims:
            # Reshape to (time, n_src) and apply weights as matrix multiply
            flat = da.values.reshape(len(da.time), -1)  # (T, n_src)
            out_flat = (self._weights @ flat.T).T        # (T, n_tgt)
            out_data = out_flat.reshape(len(da.time), n_tgt_lat, n_tgt_lon)
            return xr.DataArray(
                out_data,
                dims=["time", "lat", "lon"],
                coords={"time": da.time, "lat": self._tgt_lat, "lon": self._tgt_lon},
            )
        else:
            flat = da.values.ravel()
            out_flat = self._weights @ flat
            out_data = out_flat.reshape(n_tgt_lat, n_tgt_lon)
            return xr.DataArray(
                out_data,
                dims=["lat", "lon"],
                coords={"lat": self._tgt_lat, "lon": self._tgt_lon},
            )

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
# multiplicative QDM division-by-zero when source has many dry days.
_PRECIP_JITTER_THRESH = "0.1 mm d-1"


# ===========================================================================
# Constants
# ===========================================================================

_LAPSE_RATE      = 0.0065   # K m⁻¹ (standard environmental lapse rate)
_QDM_NQUANTILES  = 20
_TRAINING_START  = "1980-01-01"
_TRAINING_END    = "2005-12-31"    # CORDEX default
_CMIP6_TRAINING_END = "2014-12-31" # CMIP6 historical ends 2014
_TIME_CHUNK      = 365       # days per dask chunk

# CORDEX variable names that correspond to each variable type
_CORDEX_VAR: dict[str, str] = {
    "temp_mean": "tas",
    "temp_min":  "tasmin",
    "temp_max":  "tasmax",
    "precip":    "pr",
}

# CMIP6 variable names — same as CORDEX (both follow CF/CMIP conventions)
_CMIP6_VAR: dict[str, str] = {
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
# _DownscalerBase  —  shared base for CORDEXDownscaler and CMIP6Downscaler
# ===========================================================================

class _DownscalerBase:
    """
    Shared base class for CORDEX and CMIP6 QDM downscalers.

    Subclasses must set these class-level attributes and implement the
    abstract methods listed below.

    Class attributes to override
    ----------------------------
    _DEFAULT_TRAIN_START : str     ISO date for fallback training start
    _DEFAULT_TRAIN_END   : str     ISO date for fallback training end
    _output_prefix       : str     file prefix ('cordex' or 'cmip6')
    _source_label        : str     human-readable label ('CORDEX' or 'CMIP6')
    _train_start_key     : str     namelist key for training start
    _train_end_key       : str     namelist key for training end
    _SOURCE_VAR          : dict    {variable_type: source_var_name}
    _default_scenarios   : list    default scenarios if not in namelist
    _scenarios_key       : str     namelist key for scenarios list

    Abstract methods (must implement)
    ----------------------------------
    _build_regridder(da_sample)     → xe.Regridder
    _load_source(vtype, scen, ...)  → xr.DataArray  on ERA5 grid
    _load_source_orog_on_era5()     → Optional[xr.DataArray]
    _source_file_path(vtype, scen)  → Path
    """

    # Defaults — override in subclasses
    _DEFAULT_TRAIN_START = "1980-01-01"
    _DEFAULT_TRAIN_END   = "2005-12-31"
    _output_prefix       = "base"
    _source_label        = "source"
    _train_start_key     = "train_start"
    _train_end_key       = "train_end"
    _SOURCE_VAR: dict    = _CORDEX_VAR
    _default_scenarios   = ["historical"]
    _scenarios_key       = "scenarios"

    def __init__(
        self,
        namelist_path: "str | Path",
        model_id: str,
        force_reprocess: bool = False,
    ) -> None:
        self.namelist_path   = Path(namelist_path)
        self.model_id        = model_id
        self.force_reprocess = force_reprocess

        with open(self.namelist_path) as f:
            self.config = yaml.safe_load(f)

        self.gauge_id  = self.config["gauge_id"]
        self.main_dir  = Path(self.config["main_dir"])
        from paths import get_paths
        paths = get_paths(self.config)
        self.model_dir = paths['catchment_dir'].parent
        self.shared_data_dir = paths['data_obs_dir']
        self.shared_data_dir.mkdir(parents=True, exist_ok=True)

        self.train_start = pd.Timestamp(
            self.config.get(self._train_start_key, self._DEFAULT_TRAIN_START)
        )
        self.train_end = pd.Timestamp(
            self.config.get(self._train_end_key, self._DEFAULT_TRAIN_END)
        )

        self.debug  = self.config.get("debug", False)
        class_name  = type(self).__name__
        self.logger = _setup_logger(
            f"{class_name}_{self.gauge_id}_{model_id}", self.debug
        )
        self.logger.info(f"{class_name}  gauge={self.gauge_id}  model={model_id}")
        self.logger.info(f"  Output dir  : {self.shared_data_dir}")
        self.logger.info(
            f"  Train period: {self.train_start.date()} - {self.train_end.date()}"
        )

        # Optional: raw ERA5 meteo directory for extending training data
        meteo_dir = self.config.get("meteo_dir")
        self.meteo_dir: Optional[Path] = Path(meteo_dir) if meteo_dir else None

        # Lazy-built caches
        self._regridder: Optional[object]              = None
        self._era5_target_grid: Optional[xr.Dataset]   = None
        self._era5_elev: Optional[xr.DataArray]        = None
        self._source_orog_cache: Optional[xr.DataArray] = None
        self._era5_ref_cache: dict                      = {}

    # ── ERA5-Land target grid ────────────────────────────────────────────────

    def _get_era5_target_grid(self) -> xr.Dataset:
        """Return a minimal xesmf target Dataset from an ERA5-Land file."""
        if self._era5_target_grid is not None:
            return self._era5_target_grid

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
            lat_coord = "latitude" if "latitude" in ds.coords else "lat"
            lon_coord = "longitude" if "longitude" in ds.coords else "lon"
            lat = ds[lat_coord].values
            lon = ds[lon_coord].values

        self._era5_target_grid = xr.Dataset({
            "lat": xr.DataArray(lat, dims=["lat"],
                                attrs={"units": "degrees_north", "axis": "Y"}),
            "lon": xr.DataArray(lon, dims=["lon"],
                                attrs={"units": "degrees_east",  "axis": "X"}),
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

        tgt = self._get_era5_target_grid()
        lat_dim = "latitude" if "latitude" in da.dims else "lat"
        lon_dim = "longitude" if "longitude" in da.dims else "lon"
        da = da.interp(
            {lat_dim: tgt["lat"].values, lon_dim: tgt["lon"].values},
            method="linear",
        )
        rename = {}
        if "lat" in da.dims and "lat" != "latitude":
            rename["lat"] = "latitude"
        if "lon" in da.dims and "lon" != "longitude":
            rename["lon"] = "longitude"
        if rename:
            da = da.rename(rename)

        self._era5_elev = da
        return da

    def _load_source_orog_on_era5(self) -> Optional[xr.DataArray]:
        """Return source model orography (m) regridded to ERA5-Land grid.
        Subclasses must implement this."""
        raise NotImplementedError

    # ── xesmf regridder ──────────────────────────────────────────────────────

    def _build_regridder(self, da_sample: xr.DataArray) -> "xe.Regridder":
        """Build (and cache) an xesmf bilinear regridder.
        Subclasses must implement this."""
        raise NotImplementedError

    def _regrid(self, da: xr.DataArray) -> xr.DataArray:
        """
        Regrid *da* from the source grid to the ERA5-Land grid.

        Returns a DataArray with dims renamed to (…, latitude, longitude)
        to match ERA5-Land convention.
        """
        if self._regridder is None:
            self._regridder = self._build_regridder(da)

        da_out = self._regridder(da)

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
        """Convert source raw units to ERA5-Land-compatible units (°C, mm/day)."""
        units = da.attrs.get("units", "")

        if variable_type in ("temp_mean", "temp_min", "temp_max"):
            if units in ("K", "Kelvin", "kelvin"):
                da = da - 273.15
                da.attrs["units"] = "degC"
            elif units in ("degC", "°C", "C", "celsius"):
                da.attrs["units"] = "degC"

        elif variable_type == "precip":
            if units in ("kg m-2 s-1", "kg m**-2 s**-1", "kgm-2s-1",
                         "kg/m2/s", "kg/(m2*s)"):
                da = da * 86400.0
                da = da.clip(min=0)
            da.attrs["units"] = "mm d-1"

        return da

    # ── Lapse-rate correction ────────────────────────────────────────────────

    def _lapse_rate_correct(self, da_temp: xr.DataArray) -> xr.DataArray:
        """
        Adjust temperature for elevation difference between source and ERA5-Land.

            T_out = T_in + 0.0065 × (orog_source − orog_ERA5)   [°C]
        """
        era5_elev   = self._load_era5_elevation()
        source_orog = self._load_source_orog_on_era5()

        if era5_elev is None or source_orog is None:
            return da_temp

        try:
            dz = source_orog - era5_elev
        except Exception:
            self.logger.warning(
                "Could not align elevation arrays for lapse-rate correction; "
                "skipping correction."
            )
            return da_temp

        correction  = _LAPSE_RATE * dz
        da_out      = da_temp + correction
        da_out.attrs.update(da_temp.attrs)
        return da_out

    # ── ERA5-Land reference loader ────────────────────────────────────────────

    # Raw ERA5-Land file naming conventions (monthly files in meteo_dir)
    _ERA5_RAW_TEMP_GLOB   = "era5_land_2m_temperature_{year}_{month:02d}.nc"
    _ERA5_RAW_PRECIP_GLOB = "era5_land_total_precipitation_{year}_{month:02d}.nc"

    def _load_era5_from_raw(
        self,
        variable_type: str,
        start_year: int,
        end_year: int,
        tgt_lat: np.ndarray,
        tgt_lon: np.ndarray,
    ) -> Optional[xr.DataArray]:
        """
        Build a daily ERA5-Land DataArray for *variable_type* from the raw
        monthly ERA5 files stored in ``self.meteo_dir``.

        Covers [start_year-01-01, end_year-12-31]. Returns None if files are
        missing or meteo_dir is not configured.
        """
        if self.meteo_dir is None:
            return None

        if variable_type in ("temp_mean", "temp_min", "temp_max"):
            raw_glob = self._ERA5_RAW_TEMP_GLOB
            raw_var  = "t2m"
        else:
            raw_glob = self._ERA5_RAW_PRECIP_GLOB
            raw_var  = "tp"

        yearly: list[xr.DataArray] = []
        for year in range(start_year, end_year + 1):
            monthly: list[xr.DataArray] = []
            for month in range(1, 13):
                fname = self.meteo_dir / raw_glob.format(year=year, month=month)
                if not fname.exists():
                    self.logger.debug(f"  Raw ERA5 file missing: {fname.name} — stop")
                    return xr.concat(yearly, dim="valid_time") if yearly else None
                ds = xr.open_dataset(fname)
                # Clip to a generous buffer around the target grid
                lat_min = float(tgt_lat.min()) - 0.5
                lat_max = float(tgt_lat.max()) + 0.5
                lon_min = float(tgt_lon.min()) - 0.5
                lon_max = float(tgt_lon.max()) + 0.5
                raw = ds[raw_var].sel(
                    latitude =slice(lat_max, lat_min),   # lat is decreasing
                    longitude=slice(lon_min, lon_max),
                ).load()
                ds.close()
                monthly.append(raw)

            year_da = xr.concat(monthly, dim="valid_time")

            # Resample hourly → daily
            if variable_type == "temp_mean":
                daily = year_da.resample(valid_time="1D").mean() - 273.15
            elif variable_type == "temp_min":
                daily = year_da.resample(valid_time="1D").min()  - 273.15
            elif variable_type == "temp_max":
                daily = year_da.resample(valid_time="1D").max()  - 273.15
            else:  # precip: deaccumulate hourly, then sum to daily, m → mm
                hourly = year_da.diff(dim="valid_time", label="upper")
                hourly = hourly.clip(min=0)  # handle forecast restart negatives
                daily = hourly.resample(valid_time="1D").sum() * 1000.0

            yearly.append(daily)
            self.logger.debug(f"  Raw ERA5 loaded: {year}  ({len(daily.valid_time)} days)")

        if not yearly:
            return None

        combined = xr.concat(yearly, dim="valid_time")

        # Snap to the exact target lat/lon grid
        combined = combined.interp(
            latitude=tgt_lat, longitude=tgt_lon, method="nearest"
        )
        combined = combined.rename(
            {"valid_time": "time", "latitude": "latitude", "longitude": "longitude"}
        )
        return combined

    def _load_era5_ref(
        self,
        variable_type: str,
        start: Optional[pd.Timestamp] = None,
        end:   Optional[pd.Timestamp] = None,
    ) -> xr.DataArray:
        """
        Load ERA5-Land DataArray for use as QDM reference.

        If the processed file in data_obs/ does not cover the requested *end*
        date (or self.train_end when end is None), and meteo_dir is configured
        in the namelist, the missing years are loaded transparently from the
        raw ERA5-Land monthly files and appended.  This lets the QDM training
        period extend beyond the model calibration period without any change to
        end_date or the model files.
        """
        era5_file = self.shared_data_dir / _ERA5_SOURCE_FILE[variable_type]
        if not era5_file.exists():
            raise FileNotFoundError(
                f"ERA5-Land reference not found: {era5_file}. "
                "Run preprocess_meteo.py first."
            )

        era5_var = _ERA5_OUTPUT_VAR[variable_type]

        # Use cache to avoid reloading/reprocessing raw files multiple times
        cache_key = variable_type
        if cache_key not in self._era5_ref_cache:
            ds = xr.open_dataset(era5_file, chunks={"time": _TIME_CHUNK})
            da = ds[era5_var]

            rename = {}
            if "lat" in da.dims and "latitude" not in da.dims:
                rename["lat"] = "latitude"
            if "lon" in da.dims and "longitude" not in da.dims:
                rename["lon"] = "longitude"
            if rename:
                da = da.rename(rename)

            processed_start = pd.Timestamp(da.time.values[0])
            processed_end   = pd.Timestamp(da.time.values[-1])
            want_start      = start if start is not None else self.train_start
            want_end        = end if end is not None else self.train_end

            # Extend backward if processed file starts after training start
            if want_start < processed_start and self.meteo_dir is not None:
                ext_start = want_start.year
                ext_end   = processed_start.year - 1
                self.logger.info(
                    f"  ERA5 processed file starts {processed_start.date()}; "
                    f"extending backward from raw files {ext_start}–{ext_end} ..."
                )
                tgt_lat = da["latitude"].values
                tgt_lon = da["longitude"].values
                extension = self._load_era5_from_raw(
                    variable_type, ext_start, ext_end, tgt_lat, tgt_lon
                )
                if extension is not None:
                    da = xr.concat(
                        [extension, da.load()], dim="time"
                    ).sortby("time")
                    new_start = pd.Timestamp(da.time.values[0])
                    self.logger.info(
                        f"  ERA5 extended back to {new_start.date()} "
                        f"({len(da.time)} days total)"
                    )
                else:
                    self.logger.warning(
                        "  Raw ERA5 backward extension failed; training limited to "
                        f"{processed_start.date()}"
                    )

            # Extend forward if processed file ends before training end
            if want_end > processed_end and self.meteo_dir is not None:
                ext_start = processed_end.year + 1
                ext_end   = want_end.year
                self.logger.info(
                    f"  ERA5 processed file ends {processed_end.date()}; "
                    f"extending from raw files {ext_start}–{ext_end} ..."
                )
                tgt_lat = da["latitude"].values
                tgt_lon = da["longitude"].values
                extension = self._load_era5_from_raw(
                    variable_type, ext_start, ext_end, tgt_lat, tgt_lon
                )
                if extension is not None:
                    da = xr.concat(
                        [da.load(), extension], dim="time"
                    ).sortby("time")
                    new_end = pd.Timestamp(da.time.values[-1])
                    self.logger.info(
                        f"  ERA5 extended to {new_end.date()} "
                        f"({len(da.time)} days total)"
                    )
                else:
                    self.logger.warning(
                        "  Raw ERA5 extension failed; training limited to "
                        f"{processed_end.date()}"
                    )

            self._era5_ref_cache[cache_key] = da

        da = self._era5_ref_cache[cache_key]

        if start or end:
            da = da.sel(time=slice(
                start.strftime("%Y-%m-%d") if start else None,
                end.strftime("%Y-%m-%d")   if end   else None,
            ))

        if variable_type == "precip":
            da.attrs["units"]         = "mm d-1"
            da.attrs["cell_methods"]  = "time: sum within days"
            da.attrs["standard_name"] = "precipitation_flux"
        else:
            da.attrs["units"]         = "degC"
            da.attrs["cell_methods"]  = "time: mean within days"
            da.attrs["standard_name"] = "air_temperature"

        da.name = era5_var
        return da

    # ── Source loader (abstract) ──────────────────────────────────────────────

    def _load_source(
        self,
        variable_type: str,
        scenario: str,
        start: Optional[pd.Timestamp] = None,
        end:   Optional[pd.Timestamp] = None,
    ) -> xr.DataArray:
        """Load source variable, convert units, regrid, and lapse-correct.
        Subclasses must implement this."""
        raise NotImplementedError

    # ── Source file path (abstract) ──────────────────────────────────────────

    def _source_file_path(self, vtype: str, scen: str) -> Path:
        """Return path to source file for one variable / scenario.
        Subclasses must implement this."""
        raise NotImplementedError

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
            f"group=time.month, period={self.train_start.year}-{self.train_end.year})"
        )
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
        return (
            self.shared_data_dir
            / f"{self._output_prefix}_{model_safe}_{scenario}_{variable_type}.nc"
        )

    def _save_output(
        self,
        da:            xr.DataArray,
        out_path:      Path,
        variable_type: str,
    ) -> None:
        """Write QDM-corrected DataArray as ERA5-Land-compatible NetCDF."""
        era5_var = _ERA5_OUTPUT_VAR[variable_type]
        da       = da.rename(era5_var)

        # Standardise dim names to (lat, lon, time) — matching ERA5-Land convention
        rename = {}
        for d in da.dims:
            if d in ("latitude", "y") and d != "lat":
                rename[d] = "lat"
            elif d in ("longitude", "x") and d != "lon":
                rename[d] = "lon"
        if rename:
            da = da.rename(rename)

        long_names = {
            "temp_mean": f"2 metre temperature (daily mean) - {self._source_label} QDM-corrected",
            "temp_min":  f"2 metre temperature (daily minimum) - {self._source_label} QDM-corrected",
            "temp_max":  f"2 metre temperature (daily maximum) - {self._source_label} QDM-corrected",
            "precip":    f"Total precipitation - {self._source_label} QDM-corrected",
        }
        da.attrs["long_name"] = long_names[variable_type]
        da.attrs["source"]    = (
            f"{self._source_label} {self.model_id}, bias-corrected with xclim QDM "
            f"(ref: ERA5-Land {self.train_start.year}-{self.train_end.year})"
        )

        # Drop stray non-spatial scalar coordinates (e.g. 'number' from ERA5)
        keep = {"time", "latitude", "longitude", "lat", "lon"}
        extra = [c for c in da.coords if c not in keep]
        if extra:
            da = da.drop_vars(extra, errors="ignore")

        da.attrs.pop("coordinates", None)

        # Ensure standard dimension order (time, lat, lon)
        target_dims = [d for d in ("time", "lat", "lon") if d in da.dims]
        if list(da.dims) != target_dims:
            da = da.transpose(*target_dims)

        ds = da.to_dataset()

        # Add ERA5-Land elevation variable for Raven's :ElevationVarNameNC
        era5_elev = self._load_era5_elevation()
        if era5_elev is not None:
            elev_clean = era5_elev.drop_vars(
                [c for c in era5_elev.coords
                 if c not in ("lat", "lon", "latitude", "longitude")],
                errors="ignore",
            )
            ds["elevation"] = elev_clean
        else:
            self.logger.warning(
                "ERA5-Land elevation not available — "
                f"elevation variable omitted from {self._output_prefix} output file."
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
        for coord in ("lat", "lon", "latitude", "longitude"):
            if coord in ds:
                encoding[coord] = {"_FillValue": None}

        # Final scrub before writing:
        # 1. Remove 'coordinates' attr — xarray may regenerate it from encoding
        # 2. Replace non-ASCII characters — unicode forces NC_STRING type which
        #    Raven's libnetcdf cannot handle.
        def _ascii_safe(s: str) -> str:
            return s.encode("ascii", errors="replace").decode("ascii")

        for var in list(ds.data_vars) + list(ds.coords):
            obj = ds[var]
            obj.attrs.pop("coordinates", None)
            obj.encoding.pop("coordinates", None)
            for k, v in list(obj.attrs.items()):
                if isinstance(v, str) and not v.isascii():
                    obj.attrs[k] = _ascii_safe(v)

        # Fill missing leap days (noleap → standard calendar conversion)
        if "time" in ds.dims:
            times = pd.DatetimeIndex(ds.time.values)
            full_idx = pd.date_range(times[0], times[-1], freq="D")
            n_missing = len(full_idx.difference(times))
            if n_missing > 0:
                self.logger.info(
                    f"  Interpolating {n_missing} missing leap days "
                    f"(noleap→standard)"
                )
                ds = ds.reindex(time=full_idx)
                ds = ds.chunk({"time": -1}).interpolate_na(dim="time", method="linear")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"  Saving {out_path.name} ...")
        ds.to_netcdf(out_path, encoding=encoding)
        size_mb = out_path.stat().st_size / 1e6
        self.logger.info(f"  OK Saved: {out_path.name}  ({size_mb:.1f} MB)")

    # ── Diagnostic plots ──────────────────────────────────────────────────────

    def _save_diagnostic_plots(
        self,
        variable_type: str,
        scenario: str,
        era5_train: xr.DataArray,
        source_raw_train: xr.DataArray,
        corrected_full: xr.DataArray,
    ) -> Optional[Path]:
        """
        Save a multi-panel diagnostic figure for one variable / scenario.

        Layout (3 rows x 3 cols, 16 x 12 in)
        ──────────────────────────────────────
        Row 0  Monthly climatology [cols 0-1] | QQ plot            [col 2]
        Row 1  Daily time series 2-yr zoom    | Wet-day / bias     [col 2]
               [cols 0-1]                     |
        Row 2  Spatial mean maps: ERA5 | source raw | corrected
               [col 0]            [col 1]     | [col 2]
        """
        if not _MATPLOTLIB_AVAILABLE:
            self.logger.warning(
                "matplotlib not available — diagnostic plots skipped."
            )
            return None

        self.logger.info(
            f"  Generating diagnostic plots ({variable_type} / {scenario})..."
        )

        model_safe = self.model_id.replace("/", "_").replace(" ", "_")
        catchment_dir = self.model_dir / f"catchment_{self.gauge_id}"
        plot_dir = catchment_dir / "plots" / "future_climate"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = (
            plot_dir
            / f"{self._output_prefix}_{model_safe}_{scenario}_{variable_type}.png"
        )

        unit_label = "mm/day" if variable_type == "precip" else "degC"
        var_title  = {
            "temp_mean": "Mean Temperature",
            "temp_min":  "Min Temperature",
            "temp_max":  "Max Temperature",
            "precip":    "Precipitation",
        }[variable_type]

        # ── Compute spatial means (lazy → numpy) ─────────────────────────────
        sp_dims = [d for d in ("latitude", "longitude") if d in era5_train.dims]
        era5_sm = era5_train.mean(dim=sp_dims).compute()
        raw_sm  = source_raw_train.mean(dim=sp_dims).compute()

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
        raw_map  = source_raw_train.mean(dim="time").compute()
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
        step = max(1, n // 2000)
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
            f"{self._source_label} QDM -- {var_title}  |  {model_safe}  |  {scenario}"
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
                        label=f"{self._source_label} raw")
        ax_monthly.plot(months, corr_clim, "s-",  color=col_corr, lw=1.5, ms=5,
                        label=f"{self._source_label} corrected")
        ax_monthly.set_xticks(months)
        ax_monthly.set_xticklabels(month_labels)
        ax_monthly.set_ylabel(unit_label)
        ax_monthly.set_title("Monthly climatology (training period)")
        ax_monthly.legend(fontsize=9)
        ax_monthly.grid(True, alpha=0.3)

        # Panel B — QQ plot
        ax_qq.scatter(era5_qq, raw_qq,  s=5, c=col_raw,  alpha=0.5,
                      label=f"{self._source_label} raw", rasterized=True)
        ax_qq.scatter(era5_qq, corr_qq, s=5, c=col_corr, alpha=0.5,
                      label=f"{self._source_label} corrected", rasterized=True)
        _diag = [min(era5_qq.min(), raw_qq.min(), corr_qq.min()),
                 max(era5_qq.max(), raw_qq.max(), corr_qq.max())]
        ax_qq.plot(_diag, _diag, "k--", lw=1, label="1:1")
        ax_qq.set_xlabel(f"ERA5-Land  ({unit_label})")
        ax_qq.set_ylabel(f"{self._source_label}  ({unit_label})")
        ax_qq.set_title("QQ plot (spatial mean, training)")
        ax_qq.legend(fontsize=8)
        ax_qq.grid(True, alpha=0.3)

        # Panel C — daily time series
        ax_ts.plot(t_era5, era5_ts.values, color=col_era5, lw=1.2,
                   label="ERA5-Land", alpha=0.85)
        ax_ts.plot(t_raw,  raw_ts.values,  color=col_raw,  lw=0.8,
                   label=f"{self._source_label} raw", alpha=0.6)
        ax_ts.plot(t_corr, corr_ts.values, color=col_corr, lw=0.9,
                   label=f"{self._source_label} corrected", alpha=0.75)
        ax_ts.set_ylabel(unit_label)
        ax_ts.set_title("Daily spatial-mean time series (first 2 yr of training)")
        ax_ts.legend(fontsize=9)
        ax_ts.grid(True, alpha=0.3)

        # Panel D — wet-day fraction or monthly bias
        x = np.arange(12)
        if variable_type == "precip":
            w = 0.25
            ax_extra.bar(x - w, era5_wet, w, color=col_era5, alpha=0.75, label="ERA5")
            ax_extra.bar(x,     raw_wet,  w, color=col_raw,  alpha=0.75,
                         label=f"{self._source_label} raw")
            ax_extra.bar(x + w, corr_wet, w, color=col_corr, alpha=0.75,
                         label="Corrected")
            ax_extra.set_xticks(x)
            ax_extra.set_xticklabels(month_labels, fontsize=8)
            ax_extra.set_ylabel("Wet-day fraction  (> 1 mm/d)")
            ax_extra.set_title("Wet-day statistics")
            ax_extra.legend(fontsize=8)
            ax_extra.grid(True, axis="y", alpha=0.3)
        else:
            w = 0.35
            ax_extra.bar(x - w/2, raw_bias,  w, color=col_raw,  alpha=0.8,
                         label=f"{self._source_label} raw bias")
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
        _map_panel(ax_m2, raw_map,   f"{self._source_label} raw mean ({unit_label})")
        im3 = _map_panel(ax_m3, corr_map, f"Corrected mean ({unit_label})")
        _plt.colorbar(im3, ax=[ax_m1, ax_m2, ax_m3], shrink=0.65,
                      label=unit_label, pad=0.02)

        fig.savefig(plot_path, dpi=120, bbox_inches="tight")
        _plt.close(fig)
        self.logger.info(f"  Diagnostic plot saved: {plot_path}")
        return plot_path

    # ── High-level entry points ───────────────────────────────────────────────

    def process_variable(self, variable_type: str, scenario: str) -> Path:
        """
        Full QDM pipeline for one variable / scenario combination.

        Steps
        -----
        1. Load ERA5-Land reference (training period)
        2. Load source historical (training period, regridded + corrected)
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
                f"  OK Exists: {out_path.name}  ({size_mb:.1f} MB)  (skip)"
            )
            return out_path

        self.logger.info(
            f"\n{'─'*56}\n"
            f"  {variable_type}  |  {scenario}  |  {self.model_id}\n"
            f"{'─'*56}"
        )

        # 1: Load ERA5-Land reference (training period)
        self.logger.info("  Loading ERA5-Land reference (training period)...")
        era5_train = self._load_era5_ref(
            variable_type, self.train_start, self.train_end
        )

        # 2: Load source historical (training period)
        self.logger.info(
            f"  Loading {self._source_label} historical (training period)..."
        )
        source_train = self._load_source(
            variable_type, "historical",
            self.train_start, self.train_end,
        )

        # 3: Align to common time period
        t_start = max(
            pd.Timestamp(era5_train.time.values[0]),
            pd.Timestamp(source_train.time.values[0]),
        )
        t_end = min(
            pd.Timestamp(era5_train.time.values[-1]),
            pd.Timestamp(source_train.time.values[-1]),
        )
        era5_train   = era5_train.sel(
            time=slice(t_start.strftime("%Y-%m-%d"), t_end.strftime("%Y-%m-%d"))
        )
        source_train = source_train.sel(
            time=slice(t_start.strftime("%Y-%m-%d"), t_end.strftime("%Y-%m-%d"))
        )

        # Inner-join on time: drops days present in one but not the other
        # (e.g. Feb 29 in ERA5 that a noleap-calendar model lacks).
        # xsdba requires ref and hist to have identical time coordinates.
        if len(era5_train.time) != len(source_train.time):
            era5_train, source_train = xr.align(
                era5_train, source_train, join="inner", copy=False
            )
            self.logger.debug(
                f"  Inner-joined training time axes: "
                f"{len(era5_train.time)} common days."
            )

        n_train = len(era5_train.time)
        self.logger.info(
            f"  Aligned training period: {t_start.date()} - {t_end.date()} "
            f"({n_train} days)"
        )

        # 4: Train QDM
        era5_train_orig   = era5_train
        source_train_orig = source_train

        if variable_type == "precip" and _jitter_under_thresh is not None:
            self.logger.info(
                f"  Applying jitter (thresh={_PRECIP_JITTER_THRESH}) for precip QDM..."
            )
            era5_train   = _jitter_under_thresh(era5_train,   _PRECIP_JITTER_THRESH)
            source_train = _jitter_under_thresh(source_train, _PRECIP_JITTER_THRESH)

        qdm = self._train_qdm(era5_train, source_train, variable_type)

        # 5: Load scenario data (full period)
        self.logger.info(
            f"  Loading {self._source_label} {scenario} (full period)..."
        )
        source_sim = self._load_source(variable_type, scenario)

        # 6: Apply QDM
        self.logger.info("  Applying QDM...")
        source_sim = source_sim.chunk({"time": -1})
        if variable_type == "precip" and _jitter_under_thresh is not None:
            source_sim = _jitter_under_thresh(source_sim, _PRECIP_JITTER_THRESH)
        corrected = qdm.adjust(source_sim)
        if variable_type == "precip":
            corrected = corrected.clip(min=0)

        # 6b: Prepend ERA5-Land warmup if ERA5 starts before the source data.
        era5_full = self._load_era5_ref(variable_type)
        era5_full_start    = pd.Timestamp(era5_full.time.values[0])
        source_data_start  = pd.Timestamp(corrected.time.values[0])

        if era5_full_start < source_data_start:
            warmup_end = (
                source_data_start - pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d")
            era5_warmup = era5_full.sel(time=slice(None, warmup_end))

            keep_coords = {"time", "latitude", "longitude", "lat", "lon"}
            extra = [c for c in era5_warmup.coords if c not in keep_coords]
            if extra:
                era5_warmup = era5_warmup.drop_vars(extra, errors="ignore")

            n_warmup = len(era5_warmup.time)
            self.logger.info(
                f"  Prepending ERA5 warmup: {era5_full_start.date()} -> "
                f"{warmup_end}  ({n_warmup} days)"
            )
            corrected = xr.concat(
                [era5_warmup.chunk({"time": -1}),
                 corrected.chunk({"time": -1})],
                dim="time",
            )

        # 7: Save
        self._save_output(corrected, out_path, variable_type)

        # 8: Diagnostic plots (non-fatal)
        try:
            self._save_diagnostic_plots(
                variable_type, scenario,
                era5_train_orig, source_train_orig, corrected,
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

        Returns
        -------
        dict mapping (variable_type, scenario) → output Path
        """
        if scenarios is None:
            scenarios = self.config.get(
                self._scenarios_key, self._default_scenarios
            )
        if variables is None:
            variables = list(self._SOURCE_VAR.keys())

        results: dict = {}
        for vtype in variables:
            for scen in scenarios:
                src_file = self._source_file_path(vtype, scen)
                if not src_file.exists():
                    self.logger.warning(
                        f"  Skip {vtype}/{scen}: "
                        f"source not found ({src_file.name})"
                    )
                    continue
                try:
                    out_path = self.process_variable(vtype, scen)
                    results[(vtype, scen)] = out_path
                except Exception as exc:
                    self.logger.error(f"  {vtype}/{scen}: {exc}")
                    self.logger.debug(traceback.format_exc())

        return results


# ===========================================================================
# CORDEXDownscaler  —  CORDEX WAS-44 rotated-pole RCM data
# ===========================================================================

class CORDEXDownscaler(_DownscalerBase):
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

    _DEFAULT_TRAIN_START = "1980-01-01"
    _DEFAULT_TRAIN_END   = "2005-12-31"
    _output_prefix       = "cordex"
    _source_label        = "CORDEX"
    _train_start_key     = "cordex_train_start"
    _train_end_key       = "cordex_train_end"
    _SOURCE_VAR          = _CORDEX_VAR
    _default_scenarios   = ["historical", "rcp45", "rcp85"]
    _scenarios_key       = "cordex_scenarios"

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

        super().__init__(namelist_path, model_id, force_reprocess)

        cordex_dir = self.config.get("cordex_dir")
        if not cordex_dir:
            raise ValueError(
                "namelist must contain a 'cordex_dir' key pointing to the "
                "downloaded CORDEX data directory."
            )
        self.cordex_dir = Path(cordex_dir)
        self.logger.info(f"  CORDEX dir  : {self.cordex_dir}")

    def _build_regridder(self, da_sample: xr.DataArray) -> "xe.Regridder":
        """
        Build xesmf bilinear regridder from CORDEX rotated-pole grid to ERA5.

        CORDEX clipped files store geographic lat/lon as 2-D auxiliary
        coordinates alongside the 1-D rotated rlat/rlon dimensions.
        """
        tgt = self._get_era5_target_grid()

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
            method            = "bilinear",
            extrap_method     = "nearest_s2d",
            ignore_degenerate = True,
        )
        self.logger.debug("xesmf bilinear regridder built (CORDEX rotated-pole).")
        return regridder

    def _load_source_orog_on_era5(self) -> Optional[xr.DataArray]:
        """Return CORDEX orography (m) regridded to the ERA5-Land grid."""
        if self._source_orog_cache is not None:
            return self._source_orog_cache

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

        self.logger.info("Regridding CORDEX orog to ERA5-Land grid...")
        orog_on_era5 = self._regrid(da_orog)
        self._source_orog_cache = orog_on_era5
        return orog_on_era5

    def _load_source(
        self,
        variable_type: str,
        scenario: str,
        start: Optional[pd.Timestamp] = None,
        end:   Optional[pd.Timestamp] = None,
    ) -> xr.DataArray:
        """Load a CORDEX variable, convert units, regrid and lapse-correct."""
        cordex_var = _CORDEX_VAR[variable_type]
        nc_path    = self.cordex_dir / self.model_id / scenario / f"{cordex_var}.nc"

        if not nc_path.exists():
            raise FileNotFoundError(
                f"CORDEX file not found: {nc_path}. "
                "Run download_cordex.py first."
            )

        ds = xr.open_dataset(nc_path, chunks={"time": _TIME_CHUNK})
        da = ds[cordex_var]

        # Normalise CORDEX timestamps to midnight
        try:
            midnight_times = da.indexes["time"].normalize()
            da = da.assign_coords(time=midnight_times)
        except AttributeError:
            pass

        if start or end:
            da = da.sel(time=slice(
                start.strftime("%Y-%m-%d") if start else None,
                end.strftime("%Y-%m-%d")   if end   else None,
            ))

        da = self._convert_units(da, variable_type)

        self.logger.debug(f"  Regridding {cordex_var} ({scenario})...")
        da = self._regrid(da)

        if variable_type != "precip":
            da = self._lapse_rate_correct(da)

        era5_var = _ERA5_OUTPUT_VAR[variable_type]
        da.name  = era5_var
        if variable_type == "precip":
            da.attrs["units"]         = "mm d-1"
            da.attrs["cell_methods"]  = "time: sum within days"
            da.attrs["standard_name"] = "precipitation_flux"
        else:
            da.attrs["units"]         = "degC"
            da.attrs["cell_methods"]  = "time: mean within days"
            da.attrs["standard_name"] = "air_temperature"

        return da

    def _source_file_path(self, vtype: str, scen: str) -> Path:
        cordex_var = _CORDEX_VAR[vtype]
        return self.cordex_dir / self.model_id / scen / f"{cordex_var}.nc"


# ===========================================================================
# CMIP6Downscaler  —  raw CMIP6 GCM data (ISIMIP3b-compatible models)
# ===========================================================================

class CMIP6Downscaler(_DownscalerBase):
    """
    Downscale one CMIP6 GCM to ERA5-Land 0.1° resolution using xclim QDM.

    Uses the same QDM workflow as CORDEXDownscaler but operates on standard
    lat/lon CMIP6 data (no rotated-pole transformation required).

    Key differences from CORDEXDownscaler
    --------------------------------------
    Training end   : 2014 (CMIP6 historical ends Dec 2014)
    Future start   : 2015
    Scenarios      : ssp126 / ssp370 / ssp585  (not rcp*)
    Output prefix  : 'cmip6_'
    Namelist key   : 'cmip6_dir'
    Grid           : Standard lat/lon (simpler regridder)

    Parameters
    ----------
    namelist_path : str or Path
        Raven-world namelist YAML for the target catchment.
    model_id : str
        ISIMIP3b / Raven-world model ID (e.g. 'GFDL-ESM4').
    force_reprocess : bool
        Re-compute even if output files already exist.
    """

    _DEFAULT_TRAIN_START = "1980-01-01"
    _DEFAULT_TRAIN_END   = "2014-12-31"
    _output_prefix       = "cmip6"
    _source_label        = "CMIP6"
    _train_start_key     = "cmip6_train_start"
    _train_end_key       = "cmip6_train_end"
    _SOURCE_VAR          = _CMIP6_VAR
    _default_scenarios   = ["historical", "ssp126"]
    _scenarios_key       = "cmip6_scenarios"

    def __init__(
        self,
        namelist_path: "str | Path",
        model_id: str,
        force_reprocess: bool = False,
    ) -> None:
        if not _XESMF_AVAILABLE:
            import logging
            logging.getLogger(__name__).info(
                "xesmf not available — using scipy bilinear interpolation for CMIP6 regridding"
            )
        if not _XCLIM_AVAILABLE:
            raise ImportError(
                "xsdba (or xclim) is required for QDM bias correction.\n"
                "  conda install -c conda-forge xsdba xclim"
            )

        super().__init__(namelist_path, model_id, force_reprocess)

        cmip6_dir = self.config.get("cmip6_dir")
        if not cmip6_dir:
            raise ValueError(
                "namelist must contain a 'cmip6_dir' key pointing to the "
                "downloaded CMIP6 data directory."
            )
        self.cmip6_dir = Path(cmip6_dir)
        self.logger.info(f"  CMIP6 dir   : {self.cmip6_dir}")

    def _build_regridder(self, da_sample: xr.DataArray):
        """
        Build bilinear regridder from standard lat/lon CMIP6 grid to ERA5.
        Uses xesmf if available, otherwise falls back to scipy interpolation.
        """
        tgt = self._get_era5_target_grid()

        if "time" in da_sample.dims:
            one_step = da_sample.isel(time=0).drop_vars("time", errors="ignore")
        else:
            one_step = da_sample

        # Ensure lat/lon are named 'lat'/'lon'
        rename = {}
        if "latitude" in one_step.dims and "lat" not in one_step.dims:
            rename["latitude"] = "lat"
        if "longitude" in one_step.dims and "lon" not in one_step.dims:
            rename["longitude"] = "lon"
        if rename:
            one_step = one_step.rename(rename)

        if _XESMF_AVAILABLE:
            src_ds = one_step.to_dataset(name="_v")
            regridder = xe.Regridder(
                src_ds, tgt,
                method            = "bilinear",
                extrap_method     = "nearest_s2d",
                ignore_degenerate = True,
            )
            self.logger.debug("xesmf bilinear regridder built (CMIP6 standard lat/lon).")
            return regridder
        else:
            regridder = _ScipyBilinearRegridder(one_step, tgt, logger=self.logger)
            self.logger.debug("scipy bilinear regridder built (CMIP6 standard lat/lon).")
            return regridder

    def _load_source_orog_on_era5(self) -> Optional[xr.DataArray]:
        """Return CMIP6 orography (m) regridded to the ERA5-Land grid."""
        if self._source_orog_cache is not None:
            return self._source_orog_cache

        orog_path = self.cmip6_dir / self.model_id / "orog.nc"
        if not orog_path.exists():
            self.logger.warning(
                f"CMIP6 orog.nc not found at {orog_path} – "
                "lapse-rate correction skipped."
            )
            return None

        era5_elev = self._load_era5_elevation()
        if era5_elev is None:
            return None

        with xr.open_dataset(orog_path) as ds:
            orog_var = "orog" if "orog" in ds else list(ds.data_vars)[0]
            da_orog  = ds[orog_var].load()

        # Normalise dimension names for xesmf
        rename = {}
        if "latitude" in da_orog.dims and "lat" not in da_orog.dims:
            rename["latitude"] = "lat"
        if "longitude" in da_orog.dims and "lon" not in da_orog.dims:
            rename["longitude"] = "lon"
        if rename:
            da_orog = da_orog.rename(rename)

        self.logger.info("Regridding CMIP6 orog to ERA5-Land grid...")
        orog_on_era5 = self._regrid(da_orog)
        self._source_orog_cache = orog_on_era5
        return orog_on_era5

    def _load_source(
        self,
        variable_type: str,
        scenario: str,
        start: Optional[pd.Timestamp] = None,
        end:   Optional[pd.Timestamp] = None,
    ) -> xr.DataArray:
        """Load a CMIP6 variable, convert units, regrid to ERA5 grid."""
        cmip6_var = _CMIP6_VAR[variable_type]
        nc_path   = self.cmip6_dir / self.model_id / scenario / f"{cmip6_var}.nc"

        if not nc_path.exists():
            raise FileNotFoundError(
                f"CMIP6 file not found: {nc_path}. "
                "Run download_cmip6.py first."
            )

        ds = xr.open_dataset(nc_path, chunks={"time": _TIME_CHUNK})

        # Variable name in the file might differ; try exact match then first var
        if cmip6_var in ds:
            da = ds[cmip6_var]
        else:
            da = ds[list(ds.data_vars)[0]]

        # Normalise dimension names to lat/lon for xesmf
        rename = {}
        if "latitude" in da.dims and "lat" not in da.dims:
            rename["latitude"] = "lat"
        if "longitude" in da.dims and "lon" not in da.dims:
            rename["longitude"] = "lon"
        if rename:
            da = da.rename(rename)

        # Convert non-standard calendars (e.g. noleap/360_day) to standard
        # datetime64 so that pandas/xsdba can handle the timestamps.
        try:
            time_index = da.indexes["time"]
            if hasattr(time_index, "calendar") and time_index.calendar not in (
                "standard", "gregorian", "proleptic_gregorian",
            ):
                self.logger.debug(
                    f"  Converting {time_index.calendar} calendar to standard..."
                )
                # use_cftime=False → numpy datetime64 (required by pandas + xsdba)
                da = da.convert_calendar("standard", use_cftime=False, align_on="date")
        except Exception as _cal_err:
            self.logger.debug(f"  Calendar conversion skipped: {_cal_err}")

        # Normalise timestamps to midnight (some CMIP6 files use noon)
        try:
            midnight_times = da.indexes["time"].normalize()
            da = da.assign_coords(time=midnight_times)
        except AttributeError:
            pass

        if start or end:
            da = da.sel(time=slice(
                start.strftime("%Y-%m-%d") if start else None,
                end.strftime("%Y-%m-%d")   if end   else None,
            ))

        da = self._convert_units(da, variable_type)

        self.logger.debug(f"  Regridding {cmip6_var} ({scenario})...")
        da = self._regrid(da)

        if variable_type != "precip":
            da = self._lapse_rate_correct(da)

        era5_var = _ERA5_OUTPUT_VAR[variable_type]
        da.name  = era5_var
        if variable_type == "precip":
            da.attrs["units"]         = "mm d-1"
            da.attrs["cell_methods"]  = "time: sum within days"
            da.attrs["standard_name"] = "precipitation_flux"
        else:
            da.attrs["units"]         = "degC"
            da.attrs["cell_methods"]  = "time: mean within days"
            da.attrs["standard_name"] = "air_temperature"

        return da

    def _source_file_path(self, vtype: str, scen: str) -> Path:
        cmip6_var = _CMIP6_VAR[vtype]
        return self.cmip6_dir / self.model_id / scen / f"{cmip6_var}.nc"


# ===========================================================================
# Top-level entry points for create_input_files.py
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


def process_cmip6_climate(
    namelist_path: "str | Path",
    force_reprocess: bool = False,
) -> dict:
    """
    Downscale CMIP6 projections for all models listed in the namelist.

    Called from create_input_files.py when ``future: true`` and
    ``cmip6_models`` are set.

    Reads from namelist
    -------------------
    cmip6_dir         : str   root dir of downloaded CMIP6 data
    cmip6_models      : list  model IDs to process (e.g. ['GFDL-ESM4'])
    cmip6_scenarios   : list  must include 'historical' for QDM training
                              plus future scenarios (e.g. ['historical', 'ssp126'])
    cmip6_variables   : list  optional subset; default = all four
    cmip6_train_start : str   optional; default '1980-01-01'
    cmip6_train_end   : str   optional; default '2014-12-31'

    Returns
    -------
    dict  { model_id : { (variable_type, scenario) : Path } }
    """
    with open(namelist_path) as f:
        config = yaml.safe_load(f)

    models    = config.get("cmip6_models", [])
    scenarios = config.get("cmip6_scenarios", ["historical", "ssp126"])
    variables = config.get("cmip6_variables", list(_CMIP6_VAR.keys()))

    if not models:
        raise ValueError(
            "namelist must contain 'cmip6_models' list when future=True.\n"
            "Example:\n"
            "  cmip6_models:\n"
            "    - GFDL-ESM4\n"
        )

    if "historical" not in scenarios:
        raise ValueError(
            "cmip6_scenarios must include 'historical' for QDM training. "
            "Add 'historical' to the list.\n"
            "Example:  cmip6_scenarios: [historical, ssp126]"
        )

    all_results: dict = {}
    for model_id in models:
        print(f"\n{'='*60}")
        print(f"  CMIP6 downscaling  |  model={model_id}")
        print(f"  scenarios : {scenarios}")
        print(f"  variables : {variables}")
        print(f"{'='*60}\n")

        downscaler = CMIP6Downscaler(
            namelist_path   = namelist_path,
            model_id        = model_id,
            force_reprocess = force_reprocess,
        )
        all_results[model_id] = downscaler.process_all(
            scenarios = scenarios,
            variables = variables,
        )

    return all_results
