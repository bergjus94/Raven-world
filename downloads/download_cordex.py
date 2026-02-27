#!/usr/bin/env python3
"""
download_cordex.py  —  Download and clip CORDEX South Asia (WAS-44) climate projections.

Primary source  : ESGF HTTP search API (public, no account required for most
                  CORDEX data).  Files are discovered via the ESGF Solr API,
                  downloaded in their native 5-year chunks, clipped to the
                  target bounding box and then merged into one file per
                  variable / experiment / model.

Fallback source : Copernicus CDS (cdsapi, same ~/.cdsapirc as ERA5-Land).
                  Downloads the full WAS-44 domain as a ZIP archive, extracts,
                  clips and removes the raw file to conserve disk space.
                  Note: CDS does NOT support area sub-setting for CORDEX, so
                  the full domain (~50-200 MB per file) is downloaded first.

CORDEX technical details
------------------------
Domain      : WAS-44 (West + South Asia)  ~0.44° ≈ 50 km
Projection  : Rotated pole  →  native coordinates are rlat / rlon;
              geographic lat / lon are stored as 2-D auxiliary fields.
Historical  : 1950–2005
Future      : 2006–2100  (RCP 2.6 / 4.5 / 8.5)
Variables   : tasmax (K), tasmin (K), tas (K), pr (kg m-2 s-1), orog (m)
              Unit conversions are performed in preprocess_climate.py, not here.

Models (GloGEM naming → CORDEX DRS identifiers)
------------------------------------------------
GloGEM name           RCM                 Driving GCM          Ensemble
SMHI-RCA_MPIESM       SMHI-RCA4           MPI-M-MPI-ESM-LR     r1i1p1   ✓ confirmed WAS-44
SMHI-RCA_ECEARTH      SMHI-RCA4           ICHEC-EC-EARTH        r12i1p1  ✓ confirmed WAS-44
SMHI-RCA_HADGEM       SMHI-RCA4           MOHC-HadGEM2-ES      r1i1p1   ✓ confirmed WAS-44
SMHI-RCA_IPSL         SMHI-RCA4           IPSL-IPSL-CM5A-MR    r1i1p1   ✓ confirmed WAS-44
MPICSC-REMO2_MPIESM   MPI-CSC-REMO2009    MPI-M-MPI-ESM-LR     r1i1p1   ✓ confirmed WAS-44
MPICSC-REMO1_MPIESM   MPI-CSC-REMO2009    MPI-M-MPI-ESM-LR     r2i1p1   ✓ confirmed WAS-44
DMI-HIRHAM_ECEARTH    DMI-HIRHAM5         ICHEC-EC-EARTH        r3i1p1   ? WAS coverage unconfirmed
CLMCOM-CCLM4_MPIESM   CLMcom-CCLM4-8-17  MPI-M-MPI-ESM-LR     r1i1p1   ? WAS coverage unconfirmed
CLMCOM-CCLM4_HadGem   CLMcom-CCLM4-8-17  MOHC-HadGEM2-ES      r1i1p1   ? WAS coverage unconfirmed
CLMCOM-CCLM4_ECEARTH  CLMcom-CCLM4-8-17  ICHEC-EC-EARTH        r12i1p1  ? WAS coverage unconfirmed

Recommended starting pair (two different RCMs + two different driving GCMs):
    SMHI-RCA_MPIESM   (prototype — confirmed available)
    SMHI-RCA_ECEARTH  (second run — different driving GCM)

Usage
-----
# Download two models, all confirmed experiments, to default output dir:
python downloads/download_cordex.py \\
    --models SMHI-RCA_MPIESM SMHI-RCA_ECEARTH \\
    --experiments historical rcp45 rcp85

# Override output directory:
python downloads/download_cordex.py \\
    --models SMHI-RCA_MPIESM \\
    --experiments historical rcp45 rcp85 \\
    --output-dir /path/to/01_data/CORDEX

# List available models:
python downloads/download_cordex.py --list-models

# Verify ESGF availability without downloading:
python downloads/download_cordex.py \\
    --models SMHI-RCA_MPIESM DMI-HIRHAM_ECEARTH \\
    --experiments historical \\
    --verify-only
"""

import argparse
import json
import os
import sys
import time
import tempfile
import traceback
import zipfile
from pathlib import Path

import numpy as np
import requests
import xarray as xr

# ── Optional cdsapi (only needed for CDS fallback) ─────────────────────────
try:
    import cdsapi
    _CDSAPI_AVAILABLE = True
except ImportError:
    _CDSAPI_AVAILABLE = False


# ===========================================================================
# Constants
# ===========================================================================

ESGF_SEARCH_URL = "https://esgf-node.llnl.gov/esg-search/search"
CORDEX_DOMAIN   = "WAS-44"
CORDEX_FREQ     = "day"

# Upper Indus bounding box (same as used for DEM / shapefile downloads)
DEFAULT_BBOX = dict(west=67.5, south=30.0, east=82.0, north=37.5)

# Extra buffer around the bbox to safely capture all rotated-pole cells
# that overlap the region edges.
BBOX_BUFFER = 1.0   # degrees

# CORDEX DRS variable names to CDS human-readable names (for fallback)
_VAR_TO_CDS = {
    "tasmax": "maximum_2m_temperature_in_the_last_24_hours",
    "tasmin": "minimum_2m_temperature_in_the_last_24_hours",
    "tas":    "2m_air_temperature",
    "pr":     "mean_precipitation_flux",
    "orog":   "orography",
}

# CDS experiment parameter strings
_EXP_TO_CDS = {
    "historical": "historical",
    "rcp26":      "rcp_2_6",
    "rcp45":      "rcp_4_5",
    "rcp85":      "rcp_8_5",
}

DEFAULT_VARIABLES  = ["tasmax", "tasmin", "tas", "pr"]
DEFAULT_EXPERIMENTS = ["historical", "rcp45", "rcp85"]


# ===========================================================================
# Model registry
# ===========================================================================
# Keys are the GloGEM / Raven-world internal names used in namelist YAML.
# Each entry maps to the ESGF DRS identifiers used in the search API and
# to the CDS API parameter strings used in the fallback download.

MODEL_REGISTRY = {
    # ── SMHI-RCA4 family  (confirmed WAS-44 coverage) ─────────────────────
    "SMHI-RCA_MPIESM": {
        "esgf_rcm":      "SMHI-RCA4",
        "esgf_gcm":      "MPI-M-MPI-ESM-LR",
        "esgf_ensemble": "r1i1p1",
        "cds_rcm":       "smhi_rca4",
        "cds_gcm":       "mpi_m_mpi_esm_lr",
        "cds_ensemble":  "r1i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "RCP2.6 not confirmed for WAS-44.",
    },
    "SMHI-RCA_ECEARTH": {
        "esgf_rcm":      "SMHI-RCA4",
        "esgf_gcm":      "ICHEC-EC-EARTH",
        "esgf_ensemble": "r12i1p1",
        "cds_rcm":       "smhi_rca4",
        "cds_gcm":       "ichec_ec_earth",
        "cds_ensemble":  "r12i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "EC-Earth uses ensemble member r12i1p1.",
    },
    "SMHI-RCA_HADGEM": {
        "esgf_rcm":      "SMHI-RCA4",
        "esgf_gcm":      "MOHC-HadGEM2-ES",
        "esgf_ensemble": "r1i1p1",
        "cds_rcm":       "smhi_rca4",
        "cds_gcm":       "mohc_hadgem2_es",
        "cds_ensemble":  "r1i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "",
    },
    "SMHI-RCA_IPSL": {
        "esgf_rcm":      "SMHI-RCA4",
        "esgf_gcm":      "IPSL-IPSL-CM5A-MR",
        "esgf_ensemble": "r1i1p1",
        "cds_rcm":       "smhi_rca4",
        "cds_gcm":       "ipsl_ipsl_cm5a_mr",
        "cds_ensemble":  "r1i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "",
    },
    # ── MPI-CSC REMO2009 family  (confirmed WAS-44 coverage) ──────────────
    "MPICSC-REMO2_MPIESM": {
        "esgf_rcm":      "MPI-CSC-REMO2009",
        "esgf_gcm":      "MPI-M-MPI-ESM-LR",
        "esgf_ensemble": "r1i1p1",
        "cds_rcm":       "mpi_csc_remo2009",
        "cds_gcm":       "mpi_m_mpi_esm_lr",
        "cds_ensemble":  "r1i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "",
    },
    "MPICSC-REMO1_MPIESM": {
        "esgf_rcm":      "MPI-CSC-REMO2009",
        "esgf_gcm":      "MPI-M-MPI-ESM-LR",
        "esgf_ensemble": "r2i1p1",
        "cds_rcm":       "mpi_csc_remo2009",
        "cds_gcm":       "mpi_m_mpi_esm_lr",
        "cds_ensemble":  "r2i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "Second ensemble member of REMO2009/MPI-ESM-LR.",
    },
    # ── Models with uncertain WAS-44 coverage ─────────────────────────────
    "DMI-HIRHAM_ECEARTH": {
        "esgf_rcm":      "DMI-HIRHAM5",
        "esgf_gcm":      "ICHEC-EC-EARTH",
        "esgf_ensemble": "r3i1p1",
        "cds_rcm":       "dmi_hirham5",
        "cds_gcm":       "ichec_ec_earth",
        "cds_ensemble":  "r3i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "WAS-44 availability unconfirmed; run --verify-only first.",
    },
    "CLMCOM-CCLM4_MPIESM": {
        "esgf_rcm":      "CLMcom-CCLM4-8-17",
        "esgf_gcm":      "MPI-M-MPI-ESM-LR",
        "esgf_ensemble": "r1i1p1",
        "cds_rcm":       "clmcom_cclm4_8_17",
        "cds_gcm":       "mpi_m_mpi_esm_lr",
        "cds_ensemble":  "r1i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "WAS-44 availability unconfirmed; run --verify-only first.",
    },
    "CLMCOM-CCLM4_HadGem": {
        "esgf_rcm":      "CLMcom-CCLM4-8-17",
        "esgf_gcm":      "MOHC-HadGEM2-ES",
        "esgf_ensemble": "r1i1p1",
        "cds_rcm":       "clmcom_cclm4_8_17",
        "cds_gcm":       "mohc_hadgem2_es",
        "cds_ensemble":  "r1i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "WAS-44 availability unconfirmed; run --verify-only first.",
    },
    "CLMCOM-CCLM4_ECEARTH": {
        "esgf_rcm":      "CLMcom-CCLM4-8-17",
        "esgf_gcm":      "ICHEC-EC-EARTH",
        "esgf_ensemble": "r12i1p1",
        "cds_rcm":       "clmcom_cclm4_8_17",
        "cds_gcm":       "ichec_ec_earth",
        "cds_ensemble":  "r12i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "WAS-44 availability unconfirmed; run --verify-only first.",
    },
}


# ===========================================================================
# ESGF search helpers
# ===========================================================================

def search_esgf(rcm_name: str, gcm_id: str, ensemble: str,
                experiment: str, variable: str,
                time_frequency: str = "day",
                timeout: int = 30) -> list[dict]:
    """
    Query the ESGF Solr search API and return a list of file records.

    Each record is a dict with at least:
        'title'     : filename (e.g. 'tasmax_WAS-44_..._19510101-19551231.nc')
        'http_url'  : direct HTTPS download URL  (None if not found)
        'opendap_url': OPeNDAP URL               (None if not found)
        'size'      : file size in bytes (int)
    """
    params = {
        "type":              "File",
        "project":           "CORDEX",
        "domain":            CORDEX_DOMAIN,
        "experiment":        experiment,
        "rcm_name":          rcm_name,
        "driving_model_id":  gcm_id,
        "ensemble":          ensemble,
        "variable":          variable,
        "time_frequency":    time_frequency,
        "format":            "application/solr+json",
        "limit":             200,
        "distrib":           "true",
    }

    try:
        resp = requests.get(ESGF_SEARCH_URL, params=params, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"   ⚠️  ESGF search failed: {e}")
        return []

    docs = resp.json().get("response", {}).get("docs", [])
    records = []
    for doc in docs:
        http_url    = None
        opendap_url = None
        for url_str in doc.get("url", []):
            # Format: "https://host/path/file.nc|mime_type|access_type"
            parts = url_str.split("|")
            if len(parts) < 3:
                continue
            url, _, access = parts[0], parts[1], parts[2]
            if access == "HTTPServer":
                http_url = url
            elif access == "OPENDAP":
                # Convert OPENDAP viewer URL to dodsC download URL
                opendap_url = url.replace(".html", "")
        records.append({
            "title":       doc.get("title", ""),
            "http_url":    http_url,
            "opendap_url": opendap_url,
            "size":        doc.get("size", [0])[0] if isinstance(doc.get("size"), list) else doc.get("size", 0),
        })

    # Sort chronologically by filename
    records.sort(key=lambda r: r["title"])
    return records


def search_esgf_orog(rcm_name: str, gcm_id: str, ensemble: str,
                     timeout: int = 30) -> list[dict]:
    """Search for the time-invariant orography (fx frequency) file."""
    params = {
        "type":             "File",
        "project":          "CORDEX",
        "domain":           CORDEX_DOMAIN,
        "rcm_name":         rcm_name,
        "driving_model_id": gcm_id,
        "ensemble":         ensemble,
        "variable":         "orog",
        "time_frequency":   "fx",
        "format":           "application/solr+json",
        "limit":            10,
        "distrib":          "true",
    }
    try:
        resp = requests.get(ESGF_SEARCH_URL, params=params, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"   ⚠️  ESGF orog search failed: {e}")
        return []

    docs = resp.json().get("response", {}).get("docs", [])
    records = []
    for doc in docs:
        http_url = None
        for url_str in doc.get("url", []):
            parts = url_str.split("|")
            if len(parts) >= 3 and parts[2] == "HTTPServer":
                http_url = parts[0]
        records.append({
            "title":    doc.get("title", ""),
            "http_url": http_url,
            "size":     doc.get("size", [0])[0] if isinstance(doc.get("size"), list) else 0,
        })
    return records


# ===========================================================================
# Rotated-pole clipping
# ===========================================================================

def _detect_dims(ds: xr.Dataset) -> tuple[str, str]:
    """Return (rlat_dim, rlon_dim) dimension names from a CORDEX dataset."""
    for lat_name in ("rlat", "y", "lat"):
        for lon_name in ("rlon", "x", "lon"):
            if lat_name in ds.dims and lon_name in ds.dims:
                return lat_name, lon_name
    raise ValueError(
        f"Cannot detect rotated-pole dimensions in dataset with dims: {list(ds.dims)}"
    )


def clip_to_bbox(ds: xr.Dataset,
                 west: float, south: float, east: float, north: float,
                 buffer: float = BBOX_BUFFER) -> xr.Dataset:
    """
    Clip a CORDEX rotated-pole dataset to a geographic bounding box.

    CORDEX files store actual lat/lon as 2-D auxiliary coordinate variables
    'lat' and 'lon' alongside the 1-D rotated 'rlat'/'rlon' dimensions.
    This function identifies which (rlat, rlon) index ranges overlap the
    geographic bbox and returns the corresponding rectangular subset.

    Parameters
    ----------
    ds : xr.Dataset   CORDEX dataset with rlat/rlon dimensions
    west, south, east, north : float   Geographic bounding box in WGS84
    buffer : float    Extra margin in degrees (default BBOX_BUFFER = 1.0°)
    """
    rlat_dim, rlon_dim = _detect_dims(ds)

    # Retrieve 2-D geographic coordinates (present in all standard CORDEX files)
    if "lat" in ds.coords and "lon" in ds.coords:
        lat_2d = ds["lat"].values
        lon_2d = ds["lon"].values
    else:
        raise ValueError(
            "Dataset does not contain 2-D 'lat'/'lon' auxiliary coordinates. "
            "Cannot clip by geographic bounding box."
        )

    w, s, e, n = west - buffer, south - buffer, east + buffer, north + buffer

    # Geographic mask over the 2-D grid
    mask = (lat_2d >= s) & (lat_2d <= n) & (lon_2d >= w) & (lon_2d <= e)

    # Find bounding indices along each axis
    rlat_indices = np.where(mask.any(axis=1))[0]
    rlon_indices = np.where(mask.any(axis=0))[0]

    if len(rlat_indices) == 0 or len(rlon_indices) == 0:
        raise ValueError(
            f"Bounding box ({w:.1f}°W, {s:.1f}°S, {e:.1f}°E, {n:.1f}°N) "
            f"does not overlap the dataset domain."
        )

    rlat_sl = slice(int(rlat_indices[0]),  int(rlat_indices[-1]) + 1)
    rlon_sl = slice(int(rlon_indices[0]),  int(rlon_indices[-1]) + 1)

    return ds.isel({rlat_dim: rlat_sl, rlon_dim: rlon_sl})


# ===========================================================================
# Download helpers
# ===========================================================================

def _download_file_http(url: str, dest_path: Path,
                        max_retries: int = 3,
                        chunk_size: int = 1 << 20) -> bool:
    """Download a single file via HTTPS with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"      Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(5 * attempt)
    return False


def _save_clipped(ds: xr.Dataset, out_path: Path,
                  variable: str, compress_level: int = 4) -> None:
    """Write a clipped dataset to NetCDF with compression."""
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {"zlib": True, "complevel": compress_level,
                         "dtype": "float32"}
    # Keep coordinate variables without fill values
    for coord in ds.coords:
        encoding[coord] = {"_FillValue": None}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path, encoding=encoding)


def download_and_clip_chunk(record: dict, variable: str,
                             bbox: dict, tmp_dir: Path) -> Path | None:
    """
    Download one ESGF file chunk, clip to bbox and save a temp clipped copy.

    Returns path to the clipped temp file, or None on failure.
    """
    title     = record["title"]
    http_url  = record.get("http_url")

    if not http_url:
        print(f"      ⚠️  No HTTP URL for {title} — skipping chunk")
        return None

    raw_path  = tmp_dir / title
    clip_path = tmp_dir / f"clipped_{title}"

    # Skip if already clipped (interrupted previous run)
    if clip_path.exists():
        return clip_path

    # Download raw chunk
    print(f"      📥 {title}  ({record['size'] / 1e6:.1f} MB)")
    if not _download_file_http(http_url, raw_path):
        print(f"      ❌ Download failed for {title}")
        return None

    # Clip and save
    try:
        with xr.open_dataset(raw_path) as ds:
            ds_clipped = clip_to_bbox(ds, **bbox)
            _save_clipped(ds_clipped, clip_path, variable)
    except Exception as e:
        print(f"      ❌ Clip failed for {title}: {e}")
        raw_path.unlink(missing_ok=True)
        return None
    finally:
        raw_path.unlink(missing_ok=True)   # always remove raw (large) file

    return clip_path


def merge_chunks(chunk_paths: list[Path], out_path: Path,
                 variable: str) -> bool:
    """Merge a list of clipped yearly/5-year chunks into one file."""
    try:
        datasets = [xr.open_dataset(p) for p in chunk_paths]
        merged   = xr.concat(datasets, dim="time")
        for ds in datasets:
            ds.close()
        # Sort time axis (should already be sorted, but be safe)
        merged = merged.sortby("time")
        _save_clipped(merged, out_path, variable)
        merged.close()
        return True
    except Exception as e:
        print(f"      ❌ Merge failed: {e}")
        traceback.print_exc()
        return False


# ===========================================================================
# CDS fallback
# ===========================================================================

def _download_variable_cds(model_id: str, experiment: str,
                            variable: str, out_path: Path,
                            bbox: dict) -> bool:
    """
    Download a single variable / experiment via Copernicus CDS.

    The full WAS-44 domain is downloaded (area sub-setting not supported for
    CORDEX on CDS), then clipped and saved.  The raw zip + extracted files
    are removed immediately to conserve disk space.
    """
    if not _CDSAPI_AVAILABLE:
        print("      ❌ cdsapi not installed; cannot use CDS fallback.")
        return False

    info = MODEL_REGISTRY[model_id]
    cds_var = _VAR_TO_CDS.get(variable)
    if cds_var is None:
        print(f"      ❌ No CDS variable name for '{variable}'")
        return False

    exp_start, exp_end = {"historical": (1950, 2005),
                          "rcp26":      (2006, 2100),
                          "rcp45":      (2006, 2100),
                          "rcp85":      (2006, 2100)}[experiment]

    # CDS requires separate start_year / end_year 5-year blocks
    start_years = [str(y) for y in range(exp_start, exp_end + 1, 5)]
    end_years   = [str(min(y + 4, exp_end)) for y in range(exp_start, exp_end + 1, 5)]

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / f"{model_id}_{experiment}_{variable}.zip"
        print(f"   📥 CDS download: {model_id} / {experiment} / {variable}")
        print(f"      (full WAS-44 domain — will clip after download)")
        try:
            c = cdsapi.Client()
            c.retrieve(
                "projections-cordex-domains-single-levels",
                {
                    "download_format":    "zip",
                    "data_format":        "netcdf_legacy",
                    "domain":             "south_asia",
                    "experiment":         _EXP_TO_CDS[experiment],
                    "horizontal_resolution": "0_44_degree_x_0_44_degree",
                    "temporal_resolution": "daily_mean",
                    "variable":           cds_var,
                    "gcm_model":          info["cds_gcm"],
                    "rcm_model":          info["cds_rcm"],
                    "ensemble_member":    info["cds_ensemble"],
                    "start_year":         start_years,
                    "end_year":           end_years,
                },
                str(zip_path),
            )
        except Exception as e:
            print(f"      ❌ CDS retrieval failed: {e}")
            return False

        # Extract zip
        extract_dir = Path(tmp) / "extracted"
        extract_dir.mkdir()
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except Exception as e:
            print(f"      ❌ Unzip failed: {e}")
            return False

        nc_files = sorted(extract_dir.glob("*.nc"))
        if not nc_files:
            print("      ❌ No NetCDF files in CDS zip archive")
            return False

        # Open, clip and merge
        try:
            datasets = [xr.open_dataset(p) for p in nc_files]
            merged   = xr.concat(datasets, dim="time").sortby("time")
            for ds in datasets:
                ds.close()
            ds_clipped = clip_to_bbox(merged, **bbox)
            _save_clipped(ds_clipped, out_path, variable)
            return True
        except Exception as e:
            print(f"      ❌ CDS clip/merge failed: {e}")
            traceback.print_exc()
            return False


# ===========================================================================
# Top-level download orchestration
# ===========================================================================

def download_variable(model_id: str, experiment: str,
                      variable: str, out_dir: Path,
                      bbox: dict,
                      force: bool = False) -> bool:
    """
    Download, clip and merge one variable for one model / experiment.

    Output file: {out_dir}/{model_id}/{experiment}/{variable}.nc

    Returns True on success, False on failure.
    """
    info     = MODEL_REGISTRY[model_id]
    out_path = out_dir / model_id / experiment / f"{variable}.nc"

    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / 1e6
        print(f"   ✅ Already exists: {out_path.name}  ({size_mb:.1f} MB)")
        return True

    print(f"\n   🌐 ESGF search: {model_id} / {experiment} / {variable}")
    records = search_esgf(
        rcm_name   = info["esgf_rcm"],
        gcm_id     = info["esgf_gcm"],
        ensemble   = info["esgf_ensemble"],
        experiment = experiment,
        variable   = variable,
    )

    if not records:
        print(f"   ⚠️  ESGF: 0 files found. Trying CDS fallback…")
        return _download_variable_cds(model_id, experiment, variable,
                                      out_path, bbox)

    print(f"   Found {len(records)} chunk(s) on ESGF")

    # Download + clip each chunk to a temp directory, then merge
    with tempfile.TemporaryDirectory(prefix="cordex_dl_") as tmp_str:
        tmp_dir = Path(tmp_str)
        chunk_paths = []
        for i, rec in enumerate(records, 1):
            print(f"   [{i}/{len(records)}] ", end="", flush=True)
            clipped = download_and_clip_chunk(rec, variable, bbox, tmp_dir)
            if clipped is not None:
                chunk_paths.append(clipped)
            else:
                print(f"   ⚠️  Skipping failed chunk: {rec['title']}")

        if not chunk_paths:
            print(f"   ❌ All chunks failed. Trying CDS fallback…")
            return _download_variable_cds(model_id, experiment, variable,
                                          out_path, bbox)

        if len(chunk_paths) < len(records):
            print(f"   ⚠️  Only {len(chunk_paths)}/{len(records)} chunks "
                  f"downloaded — output may be incomplete.")

        print(f"   🔗 Merging {len(chunk_paths)} chunk(s)…", flush=True)
        success = merge_chunks(chunk_paths, out_path, variable)

    if success:
        size_mb = out_path.stat().st_size / 1e6
        print(f"   ✅ Saved: {out_path}  ({size_mb:.1f} MB)")
    return success


def download_orography(model_id: str, out_dir: Path,
                       bbox: dict, force: bool = False) -> bool:
    """
    Download the time-invariant orography (orog, fx frequency) for a model.

    Output: {out_dir}/{model_id}/orog.nc
    Shared across experiments since it does not change.
    """
    info     = MODEL_REGISTRY[model_id]
    out_path = out_dir / model_id / "orog.nc"

    if out_path.exists() and not force:
        print(f"   ✅ Orography already exists: {out_path}")
        return True

    print(f"\n   🌐 ESGF search: {model_id} / orog (fx)")
    records = search_esgf_orog(
        rcm_name = info["esgf_rcm"],
        gcm_id   = info["esgf_gcm"],
        ensemble = info["esgf_ensemble"],
    )

    if not records:
        print(f"   ⚠️  No orog file found on ESGF for {model_id}. "
              f"Trying CDS…")
        return _download_variable_cds(model_id, "historical", "orog",
                                      out_path, bbox)

    rec = records[0]   # Only one orography file per model
    http_url = rec.get("http_url")
    if not http_url:
        print(f"   ❌ No HTTP URL for orography: {rec['title']}")
        return False

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / rec["title"]
        print(f"   📥 {rec['title']}  ({rec['size'] / 1e6:.1f} MB)")
        if not _download_file_http(http_url, raw_path):
            return False
        try:
            with xr.open_dataset(raw_path) as ds:
                ds_clipped = clip_to_bbox(ds, **bbox)
                _save_clipped(ds_clipped, out_path, "orog")
        except Exception as e:
            print(f"   ❌ Orography clip failed: {e}")
            return False

    print(f"   ✅ Orography saved: {out_path}")
    return True


def verify_availability(model_id: str,
                        experiments: list[str],
                        variables: list[str]) -> None:
    """
    Check ESGF availability for a model without downloading anything.
    Prints a summary table.
    """
    info = MODEL_REGISTRY[model_id]
    print(f"\n{'─'*60}")
    print(f"  Model   : {model_id}")
    print(f"  RCM     : {info['esgf_rcm']}")
    print(f"  GCM     : {info['esgf_gcm']}")
    print(f"  Ensemble: {info['esgf_ensemble']}")
    if info.get("notes"):
        print(f"  Notes   : {info['notes']}")
    print(f"{'─'*60}")
    print(f"  {'Variable':<10}  {'Experiment':<12}  {'Files':>6}  Status")
    print(f"  {'─'*8}  {'─'*10}  {'─'*6}  {'─'*20}")

    for exp in experiments:
        for var in variables:
            records = search_esgf(
                rcm_name   = info["esgf_rcm"],
                gcm_id     = info["esgf_gcm"],
                ensemble   = info["esgf_ensemble"],
                experiment = exp,
                variable   = var,
            )
            n = len(records)
            status = "✅ available" if n > 0 else "❌ not found on ESGF"
            total_mb = sum(r["size"] for r in records) / 1e6
            size_str = f"~{total_mb:.0f} MB" if n > 0 else "—"
            print(f"  {var:<10}  {exp:<12}  {n:>6}  {status}  {size_str}")

    # Also check orography
    orog_records = search_esgf_orog(info["esgf_rcm"], info["esgf_gcm"],
                                     info["esgf_ensemble"])
    n = len(orog_records)
    print(f"  {'orog':<10}  {'(fx)':<12}  {n:>6}  "
          f"{'✅ available' if n > 0 else '❌ not found'}")
    print()


def download_model(model_id: str, experiments: list[str],
                   variables: list[str], out_dir: Path,
                   bbox: dict, force: bool = False,
                   include_orog: bool = True) -> dict:
    """
    Download all requested variables and experiments for one model.

    Returns a summary dict: {(experiment, variable): True/False}
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_id}'. "
            f"Available: {', '.join(MODEL_REGISTRY)}"
        )

    info    = MODEL_REGISTRY[model_id]
    results = {}

    # Warn about unconfirmed models
    if "unconfirmed" in info.get("notes", "").lower():
        print(f"\n⚠️  WARNING: {model_id} has uncertain WAS-44 coverage.")
        print(f"   Run with --verify-only first to check availability.\n")

    # Orography (shared across experiments)
    if include_orog:
        download_orography(model_id, out_dir, bbox, force=force)

    for exp in experiments:
        avail = info.get("experiments", [])
        # Map rcp26/rcp45/rcp85 to what the registry stores
        exp_key = exp.replace("_", "")   # normalise
        if exp_key not in [e.replace("_", "") for e in avail]:
            print(f"\n   ⏭️  Skipping {exp} for {model_id} "
                  f"(not in model's experiment list: {avail})")
            continue

        for var in variables:
            key = (exp, var)
            print(f"\n{'='*60}")
            print(f"  {model_id}  |  {exp}  |  {var}")
            print(f"{'='*60}")
            results[key] = download_variable(
                model_id, exp, var, out_dir, bbox, force=force
            )

    return results


# ===========================================================================
# CLI
# ===========================================================================

def _default_output_dir() -> Path:
    """
    Try to infer the CORDEX output directory from the expected Raven-world
    project structure.  Falls back to ./01_data/CORDEX.
    """
    # Typical layout: Raven-world/ contains this script under downloads/
    project_root = Path(__file__).parent.parent
    candidate    = project_root.parent / "OneDrive" / "Raven_worldwide" \
                   / "01_data" / "CORDEX"
    if candidate.parent.exists():
        return candidate
    return Path("01_data") / "CORDEX"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["SMHI-RCA_MPIESM"],
        metavar="MODEL",
        help="GloGEM model ID(s) to download. Default: SMHI-RCA_MPIESM",
    )
    parser.add_argument(
        "--experiments", nargs="+",
        default=DEFAULT_EXPERIMENTS,
        metavar="EXP",
        choices=list(_EXP_TO_CDS.keys()),
        help=f"Experiments to download. Default: {DEFAULT_EXPERIMENTS}",
    )
    parser.add_argument(
        "--variables", nargs="+",
        default=DEFAULT_VARIABLES,
        metavar="VAR",
        help=f"Variables to download. Default: {DEFAULT_VARIABLES}",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=_default_output_dir(),
        metavar="DIR",
        help="Root output directory. Default: auto-detected from project layout",
    )
    parser.add_argument(
        "--bbox", nargs=4, type=float,
        default=[DEFAULT_BBOX["west"], DEFAULT_BBOX["south"],
                 DEFAULT_BBOX["east"], DEFAULT_BBOX["north"]],
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help=f"Bounding box (WGS84). Default: {list(DEFAULT_BBOX.values())}",
    )
    parser.add_argument(
        "--no-orog", action="store_true",
        help="Skip orography download",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if output files already exist",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Check ESGF availability without downloading",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Print available model IDs and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable CORDEX model IDs (GloGEM naming):")
        for mid, info in MODEL_REGISTRY.items():
            confirmed = "✓" if "unconfirmed" not in info.get("notes", "").lower() else "?"
            print(f"  {confirmed}  {mid:<28}  {info['esgf_rcm']} / {info['esgf_gcm']}")
        print("\n  ✓ = confirmed WAS-44 coverage")
        print("  ? = availability not confirmed; use --verify-only first\n")
        return

    bbox = dict(west=args.bbox[0], south=args.bbox[1],
                east=args.bbox[2], north=args.bbox[3])

    print(f"\n{'='*60}")
    print(f"  CORDEX WAS-44 downloader")
    print(f"  Models      : {args.models}")
    print(f"  Experiments : {args.experiments}")
    print(f"  Variables   : {args.variables}")
    print(f"  Bbox        : {bbox}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"{'='*60}\n")

    if args.verify_only:
        for model_id in args.models:
            if model_id not in MODEL_REGISTRY:
                print(f"❌ Unknown model: {model_id}")
                continue
            verify_availability(model_id, args.experiments, args.variables)
        return

    # Download
    all_results = {}
    for model_id in args.models:
        if model_id not in MODEL_REGISTRY:
            print(f"\n❌ Unknown model ID: '{model_id}'")
            print(f"   Run with --list-models to see available options.")
            continue
        results = download_model(
            model_id    = model_id,
            experiments = args.experiments,
            variables   = args.variables,
            out_dir     = args.output_dir,
            bbox        = bbox,
            force       = args.force,
            include_orog = not args.no_orog,
        )
        all_results[model_id] = results

    # Summary
    print(f"\n{'='*60}")
    print("  Download summary")
    print(f"{'='*60}")
    n_ok = n_fail = 0
    for model_id, results in all_results.items():
        for (exp, var), ok in results.items():
            status = "✅" if ok else "❌"
            print(f"  {status}  {model_id:<28}  {exp:<12}  {var}")
            if ok:
                n_ok   += 1
            else:
                n_fail += 1
    print(f"\n  Total: {n_ok} succeeded, {n_fail} failed\n")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
