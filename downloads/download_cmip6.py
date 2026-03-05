#!/usr/bin/env python3
"""
download_cmip6.py  —  Download CMIP6 daily climate data for the 5 ISIMIP3b models.

Source
------
Copernicus Climate Data Store (CDS) via cdsapi.
  • Credentials: ~/.cdsapirc  (same key used for ERA5-Land / CORDEX downloads)
  • Dataset: projections-cmip6
  • CDS DOES support area subsetting for CMIP6, so only the catchment bbox
    is downloaded (unlike CORDEX which requires download-then-clip).

CMIP6 technical details
-----------------------
Models      : 5 ISIMIP3b GCMs (same models used for GloGEM runs)
Historical  : up to 2014 (CMIP6 convention; CMIP5/CORDEX ends 2005)
Future      : 2015–2100  (SSP1-2.6 / SSP3-7.0 / SSP5-8.5)
Variables   : tas (K), tasmax (K), tasmin (K), pr (kg m-2 s-1), orog (m)
              Unit conversions are performed in preprocess_climate.py.

WARNING – CDS API parameter verification
------------------------------------------
The exact parameter names and values for the 'projections-cmip6' CDS dataset
may change as Copernicus updates their API.  Before running a full download,
use --verify-only to inspect the planned requests, then test with a single
small request.  Verified parameters as of 2025:
  dataset  : 'projections-cmip6'
  temporal_resolution : 'daily'
  experiment : 'historical', 'ssp1_2_6', 'ssp3_7_0', 'ssp5_8_5'
  variable   : 'near_surface_air_temperature', 'precipitation', etc.
  model      : lowercase underscore-separated (e.g. 'gfdl_esm4')  ← verified Mar 2026
  date       : 'YYYY-MM-DD/YYYY-MM-DD'  (per-request range)
  area       : [north, west, south, east]  (decimal degrees)
  format     : 'netcdf'

Output layout
-------------
{cmip6_dir}/{model_id}/historical/tas.nc
{cmip6_dir}/{model_id}/ssp126/tas.nc
{cmip6_dir}/{model_id}/orog.nc

Storage estimate (upper Indus bbox)
-------------------------------------
Historical (1980–2014, 35 years): ~15 MB/variable after area subsetting
SSP scenario (2015–2100, 86 years): ~35 MB/variable
5 models × 4 variables × 3 scenarios ≈ ~2 GB total

Models (ISIMIP3b)
-----------------
GFDL-ESM4       NOAA-GFDL       r1i1p1f1
IPSL-CM6A-LR    IPSL            r1i1p1f1
MPI-ESM1-2-HR   MPI-M           r1i1p1f1
MRI-ESM2-0      MRI             r1i1p1f1
UKESM1-0-LL     MOHC            r1i1p1f1

Usage
-----
# List available models:
python downloads/download_cmip6.py --list-models

# Show what would be downloaded (no network calls to CDS):
python downloads/download_cmip6.py \\
    --models GFDL-ESM4 IPSL-CM6A-LR \\
    --experiments historical ssp126 \\
    --verify-only

# Download one model, two experiments, catchment bbox:
python downloads/download_cmip6.py \\
    --models GFDL-ESM4 \\
    --experiments historical ssp126 \\
    --output-dir /path/to/01_data/CMIP6 \\
    --bbox 37.5 67.5 30.0 82.0

# Force re-download of existing files:
python downloads/download_cmip6.py \\
    --models GFDL-ESM4 \\
    --experiments historical ssp126 \\
    --force
"""

import argparse
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr

try:
    import cdsapi
    _CDSAPI_AVAILABLE = True
except ImportError:
    _CDSAPI_AVAILABLE = False


# ===========================================================================
# Constants
# ===========================================================================

# Upper Indus bounding box (same as used for DEM / shapefile / CORDEX downloads)
DEFAULT_BBOX = dict(west=67.5, south=30.0, east=82.0, north=37.5)

# CMIP6 historical period ends 2014 (unlike CMIP5/CORDEX which ends 2005)
_HIST_END = 2014

# Future scenarios start 2015 for CMIP6
_FUTURE_START = 2015

# Years per CDS request chunk (10 years keeps request sizes manageable)
_CHUNK_YEARS = 10

# CDS experiment identifier → CDS API string for projections-cmip6
_EXP_TO_CDS = {
    "historical": "historical",
    "ssp126":     "ssp1_2_6",
    "ssp370":     "ssp3_7_0",
    "ssp585":     "ssp5_8_5",
}

# Variable type → CDS long-name string for projections-cmip6
# NOTE: Verify these against the live CDS catalog if downloads fail.
_VAR_TO_CDS = {
    "tas":    "near_surface_air_temperature",
    "tasmax": "daily_maximum_near_surface_air_temperature",
    "tasmin": "daily_minimum_near_surface_air_temperature",
    "pr":     "precipitation",
    "orog":   "orography",
}

DEFAULT_VARIABLES   = ["tas", "tasmax", "tasmin", "pr"]
DEFAULT_EXPERIMENTS = ["historical", "ssp126"]


# ===========================================================================
# Model registry  (ISIMIP3b GCMs)
# ===========================================================================
# Keys: Raven-world internal model IDs (used in namelist YAML).
# Values: CDS API parameter strings.

MODEL_REGISTRY = {
    # CDS model names use underscores (verified against live CDS API Mar 2026)
    "GFDL-ESM4": {
        "cds_model":       "gfdl_esm4",
        "cds_institution": "noaa-gfdl",
        "ensemble":        "r1i1p1f1",
        "experiments":     ["historical", "ssp126", "ssp370", "ssp585"],
        "notes":           "NOAA-GFDL GFDL-ESM4; ISIMIP3b model.",
    },
    "IPSL-CM6A-LR": {
        "cds_model":       "ipsl_cm6a_lr",
        "cds_institution": "ipsl",
        "ensemble":        "r1i1p1f1",
        "experiments":     ["historical", "ssp126", "ssp370", "ssp585"],
        "notes":           "IPSL-CM6A-LR; ISIMIP3b model.",
    },
    "MPI-ESM1-2-HR": {
        "cds_model":       "mpi_esm1_2_hr",
        "cds_institution": "mpi-m",
        "ensemble":        "r1i1p1f1",
        "experiments":     ["historical", "ssp126", "ssp370", "ssp585"],
        "notes":           "MPI-ESM1-2-HR; ISIMIP3b model.",
    },
    "MRI-ESM2-0": {
        "cds_model":       "mri_esm2_0",
        "cds_institution": "mri",
        "ensemble":        "r1i1p1f1",
        "experiments":     ["historical", "ssp126", "ssp370", "ssp585"],
        "notes":           "MRI-ESM2-0; ISIMIP3b model.",
    },
    "UKESM1-0-LL": {
        "cds_model":       "ukesm1_0_ll",
        "cds_institution": "mohc",
        "ensemble":        "r1i1p1f1",
        "experiments":     ["historical", "ssp126", "ssp370", "ssp585"],
        "notes":           "UKESM1-0-LL; ISIMIP3b model.",
    },
}


# ===========================================================================
# CDS download helpers
# ===========================================================================

def _cds_client():
    """Return a cdsapi Client, raising a clear error if not configured."""
    if not _CDSAPI_AVAILABLE:
        raise ImportError(
            "cdsapi is not installed. Run:\n"
            "  conda install -c conda-forge cdsapi"
        )
    return cdsapi.Client()


def _build_chunk_ranges(
    experiment: str,
    historical_start: int = 1980,
    historical_end: int = _HIST_END,
) -> list[tuple[int, int]]:
    """
    Build a list of (start_year, end_year) tuples for chunked CDS requests.

    Historical: from historical_start to historical_end in _CHUNK_YEARS-year blocks.
    SSP:        from _FUTURE_START to 2100 in _CHUNK_YEARS-year blocks.
    """
    if experiment == "historical":
        period_start = historical_start
        period_end   = min(historical_end, _HIST_END)
    else:
        period_start = _FUTURE_START
        period_end   = 2100

    ranges: list[tuple[int, int]] = []
    y = period_start
    while y <= period_end:
        chunk_end = min(y + _CHUNK_YEARS - 1, period_end)
        ranges.append((y, chunk_end))
        y = chunk_end + 1
    return ranges


_ALL_MONTHS = ["01", "02", "03", "04", "05", "06",
               "07", "08", "09", "10", "11", "12"]


def _open_downloaded(path: Path) -> xr.Dataset:
    """
    Open a CDS-downloaded file as xr.Dataset.

    The projections-cmip6 dataset always returns a ZIP archive containing:
      - provenance.json
      - provenance.png
      - {variable}_day_{model}_{experiment}_{ensemble}_{grid}_{dates}.nc

    Falls back to direct NetCDF open if the file is not a ZIP.
    """
    # CDS CMIP6 always returns ZIP — try that first
    if zipfile.is_zipfile(path):
        extract_dir = path.parent / f"_extracted_{path.stem}"
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)
        nc_files = sorted(extract_dir.glob("*.nc"))
        if nc_files:
            return xr.open_dataset(nc_files[0])
        raise ValueError(f"No NetCDF found inside ZIP: {path}")

    # Fallback: try direct NetCDF
    try:
        return xr.open_dataset(path)
    except Exception as e:
        raise ValueError(
            f"Cannot open downloaded file as ZIP or NetCDF: {path}\n{e}"
        ) from e


def download_variable(
    model_id: str,
    experiment: str,
    variable: str,
    out_dir: Path,
    bbox: dict,
    force: bool = False,
    historical_start: int = 1980,
    historical_end: int = _HIST_END,
) -> bool:
    """
    Download and save one variable for one model / experiment via CDS.

    Output file: {out_dir}/{model_id}/{experiment}/{variable}.nc

    Data is requested in _CHUNK_YEARS-year blocks and concatenated.
    Area subsetting is applied at the CDS server side (no post-clip needed).

    Parameters
    ----------
    model_id         : Raven-world model ID (e.g. 'GFDL-ESM4')
    experiment       : 'historical', 'ssp126', 'ssp370', or 'ssp585'
    variable         : CMIP6 short name (e.g. 'tas', 'pr')
    out_dir          : root output directory
    bbox             : dict with keys west, south, east, north (degrees)
    force            : overwrite existing file
    historical_start : first year to keep from the historical experiment
    historical_end   : last year to include from the historical experiment

    Returns True on success, False on failure.
    """
    info     = MODEL_REGISTRY[model_id]
    out_path = out_dir / model_id / experiment / f"{variable}.nc"

    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / 1e6
        print(f"   OK Already exists: {out_path.name}  ({size_mb:.1f} MB)")
        return True

    cds_var = _VAR_TO_CDS.get(variable)
    if cds_var is None:
        print(f"   ERROR No CDS variable name for '{variable}'")
        return False

    if experiment not in _EXP_TO_CDS:
        print(f"   ERROR Unknown experiment '{experiment}'")
        return False

    model_exps = info.get("experiments", [])
    if experiment not in model_exps:
        print(f"   SKIP {experiment} not in model's experiment list: {model_exps}")
        return True   # not a failure, just not available

    chunk_ranges = _build_chunk_ranges(experiment, historical_start, historical_end)

    all_chunks: list[xr.Dataset] = []
    c = _cds_client()

    for (chunk_start, chunk_end) in chunk_ranges:
        date_str = f"{chunk_start}-01-01/{chunk_end}-12-31"
        request = {
            "temporal_resolution": "daily",
            "experiment":          _EXP_TO_CDS[experiment],
            "variable":            cds_var,
            "model":               info["cds_model"],
            "year":                [str(y) for y in range(chunk_start, chunk_end + 1)],
            "month":               _ALL_MONTHS,
            "area":                [
                bbox["north"], bbox["west"],
                bbox["south"], bbox["east"],
            ],
        }

        print(
            f"   Download: {model_id} / {experiment} / {variable} "
            f"{chunk_start}-{chunk_end}"
        )

        with tempfile.TemporaryDirectory(prefix="cmip6_cds_") as tmp_str:
            tmp_path = Path(tmp_str) / f"chunk_{chunk_start}_{chunk_end}.nc"
            try:
                c.retrieve("projections-cmip6", request, str(tmp_path))
            except Exception as e:
                print(f"   ERROR CDS retrieval failed for {chunk_start}-{chunk_end}: {e}")
                return False

            try:
                ds_chunk = _open_downloaded(tmp_path).load()
                all_chunks.append(ds_chunk)
            except Exception as e:
                print(f"   ERROR Cannot open chunk {chunk_start}-{chunk_end}: {e}")
                return False

    if not all_chunks:
        print(f"   ERROR No data chunks downloaded for {model_id}/{experiment}/{variable}")
        return False

    # Merge chunks along time
    try:
        if len(all_chunks) == 1:
            merged = all_chunks[0]
        else:
            merged = xr.concat(all_chunks, dim="time").sortby("time")

        # Close source chunks after concat (lazy-load safety)
        for ds in all_chunks:
            ds.close()

        # Trim time axis to [historical_start, historical_end]
        if experiment == "historical":
            merged = merged.sel(
                time=slice(
                    f"{historical_start}-01-01",
                    f"{historical_end}-12-31",
                )
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        encoding = {}
        for var in merged.data_vars:
            dtype_kind = getattr(merged[var].dtype, "kind", None)
            if dtype_kind in ("f", "i", "u"):
                encoding[var] = {"zlib": True, "complevel": 4, "dtype": "float32"}
        for coord in merged.coords:
            encoding[coord] = {"_FillValue": None}

        merged.to_netcdf(out_path, encoding=encoding)

    except Exception as e:
        print(f"   ERROR Merge/save failed: {e}")
        traceback.print_exc()
        return False

    size_mb = out_path.stat().st_size / 1e6
    print(f"   OK Saved: {out_path}  ({size_mb:.1f} MB)")
    return True


def download_orography(
    model_id: str,
    out_dir: Path,
    bbox: dict,
    force: bool = False,
) -> bool:
    """
    Download the time-invariant orography for one CMIP6 model via CDS.

    Output: {out_dir}/{model_id}/orog.nc

    Note: CMIP6 orography on CDS may be listed under 'fixed' temporal
    resolution or as a separate 'fx' variable.  If the download fails,
    lapse-rate correction will be skipped in preprocess_climate.py.
    """
    info     = MODEL_REGISTRY[model_id]
    out_path = out_dir / model_id / "orog.nc"

    if out_path.exists() and not force:
        print(f"   OK Orography already exists: {out_path}")
        return True

    # CMIP6 fixed fields: temporal_resolution='fixed', experiment='historical'
    # Note: CMIP6 fixed fields on CDS may require a specific ensemble member
    # or may not support area subsetting — if this fails, orography will be
    # skipped and lapse-rate correction omitted (preprocess_climate.py handles this).
    request = {
        "temporal_resolution": "fixed",
        "experiment":          "historical",
        "variable":            "orography",
        "model":               info["cds_model"],
        "area":                [
            bbox["north"], bbox["west"],
            bbox["south"], bbox["east"],
        ],
    }

    print(f"   Download: {model_id} / orography (fixed)")

    c = _cds_client()
    with tempfile.TemporaryDirectory(prefix="cmip6_orog_") as tmp_str:
        tmp_path = Path(tmp_str) / f"{model_id}_orog.nc"
        try:
            c.retrieve("projections-cmip6", request, str(tmp_path))
        except Exception as e:
            print(f"   WARNING Orography download failed (lapse-rate correction will be skipped): {e}")
            return False

        try:
            ds = _open_downloaded(tmp_path).load()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            encoding = {}
            for var in ds.data_vars:
                dtype_kind = getattr(ds[var].dtype, "kind", None)
                if dtype_kind in ("f", "i", "u"):
                    encoding[var] = {"zlib": True, "complevel": 4, "dtype": "float32"}
            for coord in ds.coords:
                encoding[coord] = {"_FillValue": None}
            ds.to_netcdf(out_path, encoding=encoding)
            ds.close()
        except Exception as e:
            print(f"   WARNING Cannot save orography: {e}")
            return False

    print(f"   OK Orography saved: {out_path}")
    return True


# ===========================================================================
# Availability summary (offline — no network call)
# ===========================================================================

def verify_availability(
    model_id: str,
    experiments: list[str],
    variables: list[str],
    bbox: dict,
    historical_start: int = 1980,
) -> None:
    """
    Print a summary of what would be downloaded for a model.

    This does NOT make any network calls to CDS; it shows the planned
    requests including the exact parameters that would be sent.
    """
    info = MODEL_REGISTRY[model_id]
    print(f"\n{'─'*70}")
    print(f"  Model       : {model_id}")
    print(f"  CDS model   : {info['cds_model']}")
    print(f"  Institution : {info['cds_institution']}")
    print(f"  Ensemble    : {info['ensemble']}")
    print(f"  Area (NWSE) : {bbox['north']}, {bbox['west']}, {bbox['south']}, {bbox['east']}")
    if info.get("notes"):
        print(f"  Notes       : {info['notes']}")
    print(f"{'─'*70}")

    model_exps = info.get("experiments", [])
    for exp in experiments:
        for var in variables:
            if exp not in model_exps:
                status = "SKIP (not in model's experiment list)"
            else:
                cds_exp  = _EXP_TO_CDS.get(exp, "??")
                cds_var  = _VAR_TO_CDS.get(var, "??")
                chunks   = _build_chunk_ranges(exp, historical_start)
                chunk_str = ", ".join(f"{s}-{e}" for s, e in chunks)
                status = f"PLANNED  CDS: exp={cds_exp}, var={cds_var}"
                print(f"  {var:<8}  {exp:<12}  {status}")
                print(f"           chunks: {chunk_str}")
                continue
            print(f"  {var:<8}  {exp:<12}  {status}")

    # Orography
    print(f"  {'orog':<8}  {'(fixed)':<12}  PLANNED  CDS: temporal_resolution=fixed")
    print()


# ===========================================================================
# Top-level download orchestration
# ===========================================================================

def download_model(
    model_id: str,
    experiments: list[str],
    variables: list[str],
    out_dir: Path,
    bbox: dict,
    force: bool = False,
    include_orog: bool = True,
    historical_start: int = 1980,
    historical_end: int = _HIST_END,
) -> dict:
    """
    Download all requested variables and experiments for one CMIP6 model.

    Returns a summary dict: {(experiment, variable): True/False}
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_id}'. "
            f"Available: {', '.join(MODEL_REGISTRY)}"
        )

    results: dict = {}

    # Orography (shared across experiments)
    if include_orog:
        download_orography(model_id, out_dir, bbox, force=force)

    model_exps = MODEL_REGISTRY[model_id].get("experiments", [])
    for exp in experiments:
        if exp not in model_exps:
            print(
                f"\n   SKIP {exp} for {model_id} "
                f"(not in model's experiment list: {model_exps})"
            )
            continue

        for var in variables:
            key = (exp, var)
            print(f"\n{'='*60}")
            print(f"  {model_id}  |  {exp}  |  {var}")
            print(f"{'='*60}")
            results[key] = download_variable(
                model_id         = model_id,
                experiment       = exp,
                variable         = var,
                out_dir          = out_dir,
                bbox             = bbox,
                force            = force,
                historical_start = historical_start,
                historical_end   = historical_end,
            )

    return results


# ===========================================================================
# CLI
# ===========================================================================

def _default_output_dir() -> Path:
    """Infer the CMIP6 output directory from the project layout."""
    project_root = Path(__file__).parent.parent
    candidate    = (project_root.parent / "OneDrive" / "Raven_worldwide"
                    / "01_data" / "CMIP6")
    if candidate.parent.exists():
        return candidate
    return Path("01_data") / "CMIP6"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["GFDL-ESM4"],
        metavar="MODEL",
        help="Model ID(s) to download. Default: GFDL-ESM4",
    )
    parser.add_argument(
        "--experiments", nargs="+",
        default=DEFAULT_EXPERIMENTS,
        metavar="EXP",
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
        default=[DEFAULT_BBOX["north"], DEFAULT_BBOX["west"],
                 DEFAULT_BBOX["south"], DEFAULT_BBOX["east"]],
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        help=f"Bounding box (WGS84, N W S E). Default: {list(DEFAULT_BBOX.values())}",
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
        help=(
            "Show what would be downloaded (model info + planned CDS requests) "
            "without making any network calls to CDS"
        ),
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Print available model IDs and exit",
    )
    parser.add_argument(
        "--historical-start", type=int, default=1980,
        metavar="YEAR",
        help=(
            "First year to keep from the historical experiment. "
            "Time axis is trimmed to YEAR-01-01. Default: 1980"
        ),
    )
    parser.add_argument(
        "--historical-end", type=int, default=_HIST_END,
        metavar="YEAR",
        help=(
            f"Last year to keep from the historical experiment. "
            f"Time axis is trimmed to YEAR-12-31. Default: {_HIST_END}"
        ),
    )

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable CMIP6 model IDs (ISIMIP3b):")
        for mid, info in MODEL_REGISTRY.items():
            print(f"  {mid:<20}  CDS: {info['cds_model']}  ({info['cds_institution']}, {info['ensemble']})")
        print()
        return

    bbox = dict(
        north=args.bbox[0], west=args.bbox[1],
        south=args.bbox[2], east=args.bbox[3],
    )

    print(f"\n{'='*70}")
    print(f"  CMIP6 downloader  (source: Copernicus CDS projections-cmip6)")
    print(f"  Models        : {args.models}")
    print(f"  Experiments   : {args.experiments}")
    print(f"  Variables     : {args.variables}")
    print(f"  Bbox (NWSE)   : {bbox['north']}, {bbox['west']}, {bbox['south']}, {bbox['east']}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"  Hist. start   : {args.historical_start}")
    print(f"  Hist. end     : {args.historical_end}")
    print(f"{'='*70}\n")

    if args.verify_only:
        print("  -- Verify-only mode ------------------------------------------")
        print("  No CDS requests will be made.  Showing planned downloads.\n")
        for model_id in args.models:
            if model_id not in MODEL_REGISTRY:
                print(f"  ERROR Unknown model: {model_id}")
                continue
            verify_availability(
                model_id         = model_id,
                experiments      = args.experiments,
                variables        = args.variables,
                bbox             = bbox,
                historical_start = args.historical_start,
            )
        return

    # Actual download
    all_results: dict = {}
    for model_id in args.models:
        if model_id not in MODEL_REGISTRY:
            print(f"\nERROR Unknown model ID: '{model_id}'")
            print(f"   Run with --list-models to see available options.")
            continue
        results = download_model(
            model_id         = model_id,
            experiments      = args.experiments,
            variables        = args.variables,
            out_dir          = args.output_dir,
            bbox             = bbox,
            force            = args.force,
            include_orog     = not args.no_orog,
            historical_start = args.historical_start,
            historical_end   = args.historical_end,
        )
        all_results[model_id] = results

    # Summary
    print(f"\n{'='*60}")
    print("  Download summary")
    print(f"{'='*60}")
    n_ok = n_fail = 0
    for model_id, results in all_results.items():
        for (exp, var), ok in results.items():
            status = "OK" if ok else "FAIL"
            print(f"  {status}  {model_id:<20}  {exp:<12}  {var}")
            if ok:
                n_ok   += 1
            else:
                n_fail += 1
    print(f"\n  Total: {n_ok} succeeded, {n_fail} failed\n")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
