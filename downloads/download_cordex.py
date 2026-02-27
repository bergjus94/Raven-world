#!/usr/bin/env python3
"""
download_cordex.py  —  Download and clip CORDEX South Asia (WAS-44) climate projections.

Source
------
Copernicus Climate Data Store (CDS) via cdsapi.
  • Credentials: ~/.cdsapirc  (same key used for ERA5-Land downloads)
  • Dataset: projections-cordex-domains-single-levels
  • CDS does NOT support area subsetting for CORDEX, so the full WAS-44
    domain is downloaded as a ZIP archive, then clipped to the target
    bounding box and saved.  The raw files are deleted immediately.

Note on ESGF
------------
The classic ESGF Solr search API (esgf-node.llnl.gov/esg-search) was
decommissioned in 2025/2026 as part of the ESGF 2.0 migration.  The new
intake-esgf client does not yet support CORDEX.  CDS remains the most
reliable programmatic access point for CORDEX South Asia data.

CORDEX technical details
------------------------
Domain      : WAS-44 (West + South Asia)  ~0.44° ≈ 50 km
Projection  : Rotated pole  →  native dims rlat / rlon;
              geographic lat / lon stored as 2-D auxiliary fields.
Historical  : 1950–2005
Future      : 2006–2100  (RCP 2.6 / 4.5 / 8.5)
Variables   : tasmax (K), tasmin (K), tas (K), pr (kg m-2 s-1), orog (m)
              Unit conversions are performed in preprocess_climate.py.

Storage estimate per variable (upper Indus bbox, clipped)
---------------------------------------------------------
Historical (1950–2005, 56 years): ~25 MB/variable after clipping
RCP scenario (2006–2100, 95 years): ~42 MB/variable after clipping
Two models × 3 variables × 3 experiments ≈ ~400 MB total (clipped)

Models (GloGEM naming → CDS identifiers)
-----------------------------------------
GloGEM name           RCM                  Driving GCM         Ensemble
SMHI-RCA_MPIESM       smhi_rca4            mpi_m_mpi_esm_lr    r1i1p1   ✓ WAS-44
SMHI-RCA_ECEARTH      smhi_rca4            ichec_ec_earth      r12i1p1  ✓ WAS-44
SMHI-RCA_HADGEM       smhi_rca4            mohc_hadgem2_es     r1i1p1   ✓ WAS-44
SMHI-RCA_IPSL         smhi_rca4            ipsl_ipsl_cm5a_mr   r1i1p1   ✓ WAS-44
MPICSC-REMO2_MPIESM   mpi_csc_remo2009     mpi_m_mpi_esm_lr    r1i1p1   ✓ WAS-44
MPICSC-REMO1_MPIESM   mpi_csc_remo2009     mpi_m_mpi_esm_lr    r2i1p1   ✓ WAS-44
DMI-HIRHAM_ECEARTH    dmi_hirham5          ichec_ec_earth      r3i1p1   ? unconfirmed
CLMCOM-CCLM4_MPIESM   clmcom_cclm4_8_17   mpi_m_mpi_esm_lr    r1i1p1   ? unconfirmed
CLMCOM-CCLM4_HadGem   clmcom_cclm4_8_17   mohc_hadgem2_es     r1i1p1   ? unconfirmed
CLMCOM-CCLM4_ECEARTH  clmcom_cclm4_8_17   ichec_ec_earth      r12i1p1  ? unconfirmed

Usage
-----
# List available models:
python downloads/download_cordex.py --list-models

# Show what would be downloaded (no network calls to CDS):
python downloads/download_cordex.py \\
    --models SMHI-RCA_MPIESM SMHI-RCA_ECEARTH \\
    --experiments historical rcp45 rcp85 \\
    --verify-only

# Download two models (writes to auto-detected CORDEX dir):
python downloads/download_cordex.py \\
    --models SMHI-RCA_MPIESM SMHI-RCA_ECEARTH \\
    --experiments historical rcp45 rcp85

# Override output directory:
python downloads/download_cordex.py \\
    --models SMHI-RCA_MPIESM \\
    --experiments historical rcp45 rcp85 \\
    --output-dir /path/to/01_data/CORDEX
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

# Upper Indus bounding box (same as used for DEM / shapefile downloads)
DEFAULT_BBOX = dict(west=67.5, south=30.0, east=82.0, north=37.5)

# Extra buffer around the bbox to safely capture all rotated-pole cells
# that overlap the region edges.
BBOX_BUFFER = 1.0   # degrees

# CDS experiment → CDS API string
_EXP_TO_CDS = {
    "historical": "historical",
    "rcp26":      "rcp_2_6",
    "rcp45":      "rcp_4_5",
    "rcp85":      "rcp_8_5",
}

# CORDEX variable → CDS long name
_VAR_TO_CDS = {
    "tasmax": "maximum_2m_temperature_in_the_last_24_hours",
    "tasmin": "minimum_2m_temperature_in_the_last_24_hours",
    "tas":    "2m_air_temperature",
    "pr":     "mean_precipitation_flux",
    "orog":   "orography",
}

DEFAULT_VARIABLES   = ["tasmax", "tasmin", "tas", "pr"]
DEFAULT_EXPERIMENTS = ["historical", "rcp45", "rcp85"]


# ===========================================================================
# Model registry
# ===========================================================================
# Keys: GloGEM / Raven-world internal names (used in namelist YAML).
# Values: CDS API parameter strings for the CDS download.

MODEL_REGISTRY = {
    # ── SMHI-RCA4 family  (confirmed WAS-44 coverage) ─────────────────────
    "SMHI-RCA_MPIESM": {
        "cds_rcm":       "smhi_rca4",
        "cds_gcm":       "mpi_m_mpi_esm_lr",
        "cds_ensemble":  "r1i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "RCP2.6 not confirmed for WAS-44.",
    },
    "SMHI-RCA_ECEARTH": {
        "cds_rcm":       "smhi_rca4",
        "cds_gcm":       "ichec_ec_earth",
        "cds_ensemble":  "r12i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "EC-Earth uses ensemble member r12i1p1.",
    },
    "SMHI-RCA_HADGEM": {
        "cds_rcm":       "smhi_rca4",
        "cds_gcm":       "mohc_hadgem2_es",
        "cds_ensemble":  "r1i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "",
    },
    "SMHI-RCA_IPSL": {
        "cds_rcm":       "smhi_rca4",
        "cds_gcm":       "ipsl_ipsl_cm5a_mr",
        "cds_ensemble":  "r1i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "",
    },
    # ── MPI-CSC REMO2009 family  (confirmed WAS-44 coverage) ──────────────
    "MPICSC-REMO2_MPIESM": {
        "cds_rcm":       "mpi_csc_remo2009",
        "cds_gcm":       "mpi_m_mpi_esm_lr",
        "cds_ensemble":  "r1i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "",
    },
    "MPICSC-REMO1_MPIESM": {
        "cds_rcm":       "mpi_csc_remo2009",
        "cds_gcm":       "mpi_m_mpi_esm_lr",
        "cds_ensemble":  "r2i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "Second ensemble member of REMO2009/MPI-ESM-LR.",
    },
    # ── Models with uncertain WAS-44 coverage ─────────────────────────────
    "DMI-HIRHAM_ECEARTH": {
        "cds_rcm":       "dmi_hirham5",
        "cds_gcm":       "ichec_ec_earth",
        "cds_ensemble":  "r3i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "WAS-44 availability unconfirmed on CDS.",
    },
    "CLMCOM-CCLM4_MPIESM": {
        "cds_rcm":       "clmcom_cclm4_8_17",
        "cds_gcm":       "mpi_m_mpi_esm_lr",
        "cds_ensemble":  "r1i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "WAS-44 availability unconfirmed on CDS.",
    },
    "CLMCOM-CCLM4_HadGem": {
        "cds_rcm":       "clmcom_cclm4_8_17",
        "cds_gcm":       "mohc_hadgem2_es",
        "cds_ensemble":  "r1i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "WAS-44 availability unconfirmed on CDS.",
    },
    "CLMCOM-CCLM4_ECEARTH": {
        "cds_rcm":       "clmcom_cclm4_8_17",
        "cds_gcm":       "ichec_ec_earth",
        "cds_ensemble":  "r12i1p1",
        "experiments":   ["historical", "rcp45", "rcp85"],
        "notes":         "WAS-44 availability unconfirmed on CDS.",
    },
}


# ===========================================================================
# Rotated-pole clipping  (unchanged from original)
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
    """
    rlat_dim, rlon_dim = _detect_dims(ds)

    if "lat" in ds.coords and "lon" in ds.coords:
        lat_2d = ds["lat"].values
        lon_2d = ds["lon"].values
    else:
        raise ValueError(
            "Dataset does not contain 2-D 'lat'/'lon' auxiliary coordinates."
        )

    w, s, e, n = west - buffer, south - buffer, east + buffer, north + buffer
    mask = (lat_2d >= s) & (lat_2d <= n) & (lon_2d >= w) & (lon_2d <= e)

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
# Output writer
# ===========================================================================

def _save_clipped(ds: xr.Dataset, out_path: Path,
                  compress_level: int = 4) -> None:
    """Write a clipped dataset to NetCDF with compression."""
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {"zlib": True, "complevel": compress_level,
                         "dtype": "float32"}
    for coord in ds.coords:
        encoding[coord] = {"_FillValue": None}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path, encoding=encoding)


# ===========================================================================
# CDS download functions
# ===========================================================================

def _cds_client():
    """Return a cdsapi Client, raising a clear error if not configured."""
    if not _CDSAPI_AVAILABLE:
        raise ImportError(
            "cdsapi is not installed. Run:\n"
            "  conda install -c conda-forge cdsapi"
        )
    return cdsapi.Client()


def download_variable(model_id: str, experiment: str,
                      variable: str, out_dir: Path,
                      bbox: dict,
                      force: bool = False) -> bool:
    """
    Download, clip and save one variable for one model / experiment via CDS.

    Output file: {out_dir}/{model_id}/{experiment}/{variable}.nc

    Returns True on success, False on failure.
    """
    info     = MODEL_REGISTRY[model_id]
    out_path = out_dir / model_id / experiment / f"{variable}.nc"

    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / 1e6
        print(f"   ✅ Already exists: {out_path.name}  ({size_mb:.1f} MB)")
        return True

    cds_var = _VAR_TO_CDS.get(variable)
    if cds_var is None:
        print(f"   ❌ No CDS variable name for '{variable}'")
        return False

    exp_ranges = {
        "historical": (1950, 2005),
        "rcp26":      (2006, 2100),
        "rcp45":      (2006, 2100),
        "rcp85":      (2006, 2100),
    }
    if experiment not in exp_ranges:
        print(f"   ❌ Unknown experiment '{experiment}'")
        return False

    exp_start, exp_end = exp_ranges[experiment]
    start_years = [str(y) for y in range(exp_start, exp_end + 1, 5)]
    end_years   = [str(min(y + 4, exp_end)) for y in range(exp_start, exp_end + 1, 5)]

    with tempfile.TemporaryDirectory(prefix="cordex_cds_") as tmp_str:
        tmp_dir  = Path(tmp_str)
        zip_path = tmp_dir / f"{model_id}_{experiment}_{variable}.zip"

        print(f"   📥 CDS download: {model_id} / {experiment} / {variable}")
        print(f"      (full WAS-44 domain → will clip to bbox after download)")
        try:
            c = _cds_client()
            c.retrieve(
                "projections-cordex-domains-single-levels",
                {
                    "download_format":       "zip",
                    "data_format":           "netcdf_legacy",
                    "domain":                "south_asia",
                    "experiment":            _EXP_TO_CDS[experiment],
                    "horizontal_resolution": "0_44_degree_x_0_44_degree",
                    "temporal_resolution":   "daily_mean",
                    "variable":              cds_var,
                    "gcm_model":             info["cds_gcm"],
                    "rcm_model":             info["cds_rcm"],
                    "ensemble_member":       info["cds_ensemble"],
                    "start_year":            start_years,
                    "end_year":              end_years,
                },
                str(zip_path),
            )
        except Exception as e:
            print(f"   ❌ CDS retrieval failed: {e}")
            return False

        # Extract zip
        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir()
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except Exception as e:
            print(f"   ❌ Unzip failed: {e}")
            return False

        nc_files = sorted(extract_dir.glob("*.nc"))
        if not nc_files:
            print("   ❌ No NetCDF files found in downloaded ZIP archive")
            return False

        print(f"   🗂️  {len(nc_files)} chunk(s) extracted; clipping to bbox…")
        try:
            datasets = [xr.open_dataset(p) for p in nc_files]
            merged   = xr.concat(datasets, dim="time").sortby("time")
            for ds in datasets:
                ds.close()
            ds_clipped = clip_to_bbox(merged, **bbox)
            _save_clipped(ds_clipped, out_path)
            merged.close()
        except Exception as e:
            print(f"   ❌ Clip/merge failed: {e}")
            traceback.print_exc()
            return False

    size_mb = out_path.stat().st_size / 1e6
    print(f"   ✅ Saved: {out_path}  ({size_mb:.1f} MB)")
    return True


def download_orography(model_id: str, out_dir: Path,
                       bbox: dict, force: bool = False) -> bool:
    """
    Download the time-invariant orography (orog) for one model via CDS.

    Output: {out_dir}/{model_id}/orog.nc
    Shared across experiments (static field, does not change with scenario).
    """
    info     = MODEL_REGISTRY[model_id]
    out_path = out_dir / model_id / "orog.nc"

    if out_path.exists() and not force:
        print(f"   ✅ Orography already exists: {out_path}")
        return True

    with tempfile.TemporaryDirectory(prefix="cordex_orog_") as tmp_str:
        tmp_dir  = Path(tmp_str)
        zip_path = tmp_dir / f"{model_id}_orog.zip"

        print(f"   📥 CDS orography: {model_id}")
        try:
            c = _cds_client()
            c.retrieve(
                "projections-cordex-domains-single-levels",
                {
                    "download_format":       "zip",
                    "data_format":           "netcdf_legacy",
                    "domain":                "south_asia",
                    "experiment":            "historical",   # orog is the same for all
                    "horizontal_resolution": "0_44_degree_x_0_44_degree",
                    "temporal_resolution":   "fixed",
                    "variable":              "orography",
                    "gcm_model":             info["cds_gcm"],
                    "rcm_model":             info["cds_rcm"],
                    "ensemble_member":       info["cds_ensemble"],
                },
                str(zip_path),
            )
        except Exception as e:
            print(f"   ❌ CDS orography retrieval failed: {e}")
            return False

        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir()
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except Exception as e:
            print(f"   ❌ Unzip failed: {e}")
            return False

        nc_files = sorted(extract_dir.glob("*.nc"))
        if not nc_files:
            print("   ❌ No orography NetCDF found in ZIP")
            return False

        try:
            with xr.open_dataset(nc_files[0]) as ds:
                ds_clipped = clip_to_bbox(ds, **bbox)
                _save_clipped(ds_clipped, out_path)
        except Exception as e:
            print(f"   ❌ Orography clip failed: {e}")
            return False

    print(f"   ✅ Orography saved: {out_path}")
    return True


# ===========================================================================
# Availability summary (offline — no network call)
# ===========================================================================

def verify_availability(model_id: str,
                        experiments: list[str],
                        variables: list[str]) -> None:
    """
    Print a summary of what would be downloaded for a model.

    Note: This does NOT make any network calls to CDS; it simply shows the
    model registry info and what variables/experiments are planned.
    To test actual CDS availability, attempt a real download — CDS will
    return a clear error message if the combination is not available.
    """
    info = MODEL_REGISTRY[model_id]
    print(f"\n{'─'*60}")
    print(f"  Model     : {model_id}")
    print(f"  CDS RCM   : {info['cds_rcm']}")
    print(f"  CDS GCM   : {info['cds_gcm']}")
    print(f"  Ensemble  : {info['cds_ensemble']}")
    if info.get("notes"):
        print(f"  Notes     : {info['notes']}")
    print(f"{'─'*60}")
    print(f"  {'Variable':<10}  {'Experiment':<12}  Status")
    print(f"  {'─'*8}  {'─'*10}  {'─'*30}")

    model_exps = info.get("experiments", [])
    for exp in experiments:
        for var in variables:
            if exp not in model_exps:
                status = "⏭️  skipped (not in model's experiment list)"
            elif "unconfirmed" in info.get("notes", "").lower():
                status = "⚠️  WAS-44 coverage unconfirmed — attempt download to test"
            else:
                status = "🟢 planned CDS download"
            print(f"  {var:<10}  {exp:<12}  {status}")

    print(f"  {'orog':<10}  {'(fixed)':<12}  🟢 planned CDS download")
    print()


# ===========================================================================
# Top-level download orchestration
# ===========================================================================

def download_model(model_id: str, experiments: list[str],
                   variables: list[str], out_dir: Path,
                   bbox: dict, force: bool = False,
                   include_orog: bool = True) -> dict:
    """
    Download all requested variables and experiments for one model via CDS.

    Returns a summary dict: {(experiment, variable): True/False}
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_id}'. "
            f"Available: {', '.join(MODEL_REGISTRY)}"
        )

    info    = MODEL_REGISTRY[model_id]
    results = {}

    if "unconfirmed" in info.get("notes", "").lower():
        print(f"\n⚠️  WARNING: {model_id} has uncertain WAS-44 coverage on CDS.")
        print(f"   Attempting download anyway — CDS will error if unavailable.\n")

    # Orography (shared across experiments)
    if include_orog:
        download_orography(model_id, out_dir, bbox, force=force)

    model_exps = info.get("experiments", [])
    for exp in experiments:
        if exp not in model_exps:
            print(f"\n   ⏭️  Skipping {exp} for {model_id} "
                  f"(not in model's experiment list: {model_exps})")
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
    """Infer the CORDEX output directory from the project layout."""
    project_root = Path(__file__).parent.parent
    candidate    = (project_root.parent / "OneDrive" / "Raven_worldwide"
                    / "01_data" / "CORDEX")
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
        help=(
            "Show what would be downloaded (model info + planned files) "
            "without making any CDS requests"
        ),
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Print available model IDs and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable CORDEX model IDs (GloGEM naming):")
        for mid, info in MODEL_REGISTRY.items():
            confirmed = "?" if "unconfirmed" in info.get("notes", "").lower() else "✓"
            print(f"  {confirmed}  {mid:<28}  "
                  f"{info['cds_rcm']} / {info['cds_gcm']}")
        print("\n  ✓ = confirmed WAS-44 coverage on CDS")
        print("  ? = availability not confirmed; attempt download to check\n")
        return

    bbox = dict(west=args.bbox[0], south=args.bbox[1],
                east=args.bbox[2], north=args.bbox[3])

    print(f"\n{'='*60}")
    print(f"  CORDEX WAS-44 downloader  (source: Copernicus CDS)")
    print(f"  Models      : {args.models}")
    print(f"  Experiments : {args.experiments}")
    print(f"  Variables   : {args.variables}")
    print(f"  Bbox        : {bbox}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"{'='*60}\n")

    if args.verify_only:
        print("  ── Verify-only mode ──────────────────────────────────────")
        print("  No CDS requests will be made.  Showing planned downloads.\n")
        for model_id in args.models:
            if model_id not in MODEL_REGISTRY:
                print(f"  ❌ Unknown model: {model_id}")
                continue
            verify_availability(model_id, args.experiments, args.variables)
        return

    # Actual download
    all_results = {}
    for model_id in args.models:
        if model_id not in MODEL_REGISTRY:
            print(f"\n❌ Unknown model ID: '{model_id}'")
            print(f"   Run with --list-models to see available options.")
            continue
        results = download_model(
            model_id     = model_id,
            experiments  = args.experiments,
            variables    = args.variables,
            out_dir      = args.output_dir,
            bbox         = bbox,
            force        = args.force,
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
