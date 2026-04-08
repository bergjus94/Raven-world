#!/usr/bin/env python3
"""
Run future climate projections for multiple CMIP6 / GloGEM models.

For each climate model:
  1. Download CMIP6 data (if needed)
  2. Generate a temporary namelist with model-specific settings
  3. Run create_input_files.py (CMIP6 downscaling + GloGEM + Raven files)
  4. Run Raven forward
  5. Move output to model-specific subfolder

All models share the same topo_files, HRU, grid weights, calibrated params.
Model-specific files: CMIP6 forcing NetCDFs, irrigation.nc, .rvt, Raven output.

Usage:
    python src/run_future_multi.py namelists_server/namelist_0101_glogem_subdaily_future.yaml
    python src/run_future_multi.py namelist.yaml --skip-download
    python src/run_future_multi.py namelist.yaml --models GFDL-ESM4 IPSL-CM6A-LR
"""

import sys
import argparse
import re
import shutil
import subprocess
import time
import logging
import tempfile
from pathlib import Path
from datetime import datetime

import yaml

script_dir = Path(__file__).parent.absolute()
project_dir = script_dir.parent

sys.path.insert(0, str(script_dir))
from paths import get_paths

# All 5 ISIMIP3b models (matching GloGEM)
ALL_MODELS = [
    "GFDL-ESM4",
    "IPSL-CM6A-LR",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0",
    "UKESM1-0-LL",
]

# Mapping: CMIP6 model ID (upper) → GloGEM model ID (lower)
GLOGEM_MODEL_ID = {
    "GFDL-ESM4": "gfdl-esm4",
    "IPSL-CM6A-LR": "ipsl-cm6a-lr",
    "MPI-ESM1-2-HR": "mpi-esm1-2-hr",
    "MRI-ESM2-0": "mri-esm2-0",
    "UKESM1-0-LL": "ukesm1-0-ll",
}


def setup_logging(debug=False, log_file=None):
    level = logging.DEBUG if debug else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def generate_model_namelist(base_nml_path, model_id, tmp_dir):
    """Generate a temporary namelist for a specific climate model."""
    base_nml_path = Path(base_nml_path)
    with open(base_nml_path) as f:
        nml = yaml.safe_load(f)

    glogem_id = GLOGEM_MODEL_ID[model_id]

    # Update model-specific fields
    nml["glogem_model"] = glogem_id
    nml["cmip6_models"] = [model_id]

    # Resolve relative params_dir to absolute (temp namelist lives in /tmp/)
    params_dir = nml.get("params_dir", "")
    if params_dir and not Path(params_dir).is_absolute():
        # Try resolving from base namelist location first
        resolved = (base_nml_path.parent / params_dir).resolve()
        if not resolved.exists():
            # Fall back to project directory (script_dir/../)
            resolved_proj = (project_dir / params_dir.lstrip('../')).resolve()
            if resolved_proj.exists():
                resolved = resolved_proj
        nml["params_dir"] = str(resolved)

    tmp_path = Path(tmp_dir) / f"namelist_future_{model_id}.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(nml, f, default_flow_style=False, sort_keys=False)

    return tmp_path, nml


def _swap_rvt_model(rvt_path, target_model_id, logger):
    """Update .rvt CMIP6 and irrigation filenames to point to the target model.

    Swaps:
    - cmip6_{MODEL}_ssp126_*.nc → cmip6_{TARGET}_ssp126_*.nc
    - irrigation.nc or irrigation_{model}.nc → irrigation_{target_glogem_id}.nc
    """
    text = rvt_path.read_text()
    # Match any CMIP6 model name in cmip6_<MODEL>_ patterns
    new_text = re.sub(
        r'(cmip6_)([A-Za-z0-9_-]+?)(_(?:ssp|historical))',
        rf'\g<1>{target_model_id}\3',
        text,
    )
    # Swap irrigation.nc to model-specific version
    glogem_id = GLOGEM_MODEL_ID.get(target_model_id, target_model_id.lower())
    new_text = re.sub(
        r'irrigation(?:_[a-z0-9-]+)?\.nc',
        f'irrigation_{glogem_id}.nc',
        new_text,
    )
    if new_text != text:
        rvt_path.write_text(new_text)
        logger.info(f"  .rvt updated to use {target_model_id} forcing + irrigation_{glogem_id}.nc")
    else:
        logger.info(f"  .rvt already references {target_model_id}")


def run_single_model(base_nml_path, model_id, skip_download=False,
                     force=False, bbox=None, verbose=False):
    """Run the full future pipeline for one climate model."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info(f"  MODEL: {model_id}  (GloGEM: {GLOGEM_MODEL_ID[model_id]})")
    logger.info("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generate model-specific namelist
        tmp_nml_path, nml = generate_model_namelist(base_nml_path, model_id, tmp_dir)
        logger.info(f"  Temporary namelist: {tmp_nml_path}")

        gauge_id = str(nml["gauge_id"])
        model_type = nml["model_type"]
        paths = get_paths(nml)

        # Check if this model already completed (output exists)
        existing_output = paths['output_dir'] / GLOGEM_MODEL_ID[model_id]
        if existing_output.exists() and not force:
            hydrographs = list(existing_output.glob(f"{gauge_id}_{model_type}_Hydrographs*"))
            if hydrographs:
                logger.info(f"  Output already exists for {model_id}, skipping (use --force to rerun)")
                return True

        # Step 1: Download CMIP6 (if needed)
        if not skip_download:
            cmip6_dir = nml.get("cmip6_dir", "01_data/CMIP6")
            if not Path(cmip6_dir).is_absolute():
                cmip6_dir = str(main_dir / cmip6_dir)

            model_data_dir = Path(cmip6_dir) / model_id
            # Check that all required files exist, not just the directory
            required_files = [
                model_data_dir / exp / f"{var}.nc"
                for exp in ["historical", "ssp126"]
                for var in ["tas", "tasmax", "tasmin", "pr"]
            ]
            all_present = model_data_dir.exists() and all(f.exists() for f in required_files)
            if all_present:
                logger.info(f"  CMIP6 data already exists: {model_data_dir} (8/8 files)")
            else:
                missing = [f.name for f in required_files if not f.exists()]
                if missing:
                    logger.info(f"  Missing CMIP6 files for {model_id}: {missing}")
                logger.info(f"  Downloading CMIP6 data for {model_id}...")
                download_script = project_dir / "downloads" / "download_cmip6.py"
                cmd = [
                    sys.executable, str(download_script),
                    "--models", model_id,
                    "--experiments", "historical", "ssp126",
                    "--output-dir", cmip6_dir,
                ]
                if bbox:
                    cmd.extend(["--bbox", *[str(b) for b in bbox]])
                try:
                    process = subprocess.run(
                        cmd, check=True, capture_output=True, text=True,
                        timeout=3600,
                    )
                    logger.info(f"  CMIP6 download complete for {model_id}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"  CMIP6 download failed for {model_id}")
                    if e.stdout:
                        logger.error(f"  STDOUT: {e.stdout[-2000:]}")
                    if e.stderr:
                        logger.error(f"  STDERR: {e.stderr[-2000:]}")
                    return False
                except subprocess.TimeoutExpired:
                    logger.error(f"  CMIP6 download timed out for {model_id}")
                    return False

        # Step 2: Create input files (downscaling + GloGEM + .rv* model files)
        # Always run — shared files (ERA5, CMIP6, topo) have internal skip logic,
        # but .rv* model files must be regenerated for the future namelist
        # (different dates, CMIP6 forcing references, output options).
        logger.info(f"  Creating input files for {model_id}...")
        create_script = script_dir / "create_input_files.py"
        cmd = [sys.executable, str(create_script), str(tmp_nml_path)]
        if force:
            cmd.append("--force")

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in process.stdout:
                logger.info(f"    {line.rstrip()}")
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
            logger.info(f"  Input creation complete for {model_id}")
        except subprocess.CalledProcessError:
            logger.error(f"  Input creation failed for {model_id}")
            return False

        # Step 3: Leap day fix removed — CMIP6Downscaler handles this internally

        # Step 4: Add output options + transport tracers to .rvi
        model_dir = paths['model_dir']
        _add_rvi_output_options(model_dir / f"{gauge_id}_{model_type}.rvi",
                                nml.get("coupled", False), model_type, logger)

        # Step 4b: Update .rvt to point to this model's CMIP6 forcing files
        rvt_path = model_dir / f"{gauge_id}_{model_type}.rvt"
        _swap_rvt_model(rvt_path, model_id, logger)

        # Step 5: Run Raven forward (output to model-specific subfolder)
        logger.info(f"  Running Raven for {model_id}...")
        model_output_dir = paths['output_dir'] / GLOGEM_MODEL_ID[model_id]
        model_output_dir.mkdir(parents=True, exist_ok=True)

        model_file = model_dir / f"{gauge_id}_{model_type}"
        raven_exe = nml.get("raven_executable", "Raven")

        cmd = [str(raven_exe), str(model_file), "-o", str(model_output_dir) + "/"]
        try:
            start = time.time()
            process = subprocess.run(
                cmd, check=True, capture_output=True, text=True,
                timeout=1200,
            )
            duration = time.time() - start
            logger.info(f"  Raven completed in {duration:.1f}s for {model_id}")
        except subprocess.CalledProcessError as e:
            logger.error(f"  Raven failed for {model_id}")
            logger.error(e.stderr[-500:] if e.stderr else "")
            return False

        # Keep a copy of the .rvt for reference
        rvt_file = model_dir / f"{gauge_id}_{model_type}.rvt"
        if rvt_file.exists():
            shutil.copy2(str(rvt_file), str(model_output_dir / rvt_file.name))

        logger.info(f"  {model_id} complete! Output: {model_output_dir}")
        return True


def _add_rvi_output_options(rvi_path, coupled, model_type, logger):
    """Add output options and transport tracers to .rvi file if missing.

    During calibration these are intentionally left out for speed, so they
    need to be injected before any forward (future) run.
    """
    rvi_path = Path(rvi_path)
    if not rvi_path.exists():
        logger.warning(f"  .rvi not found: {rvi_path}")
        return

    with open(rvi_path) as f:
        lines = f.readlines()

    # Check if output options are already present
    content = "".join(lines)
    if ":EvaluationMetrics" in content:
        logger.info(f"  .rvi already has output options, skipping injection")
        return

    output_options = [
        "  :EvaluationMetrics RMSE KLING_GUPTA NASH_SUTCLIFFE\n",
        "  :CustomOutput DAILY AVERAGE SNOW BY_HRU_GROUP\n",
        "  :CustomOutput DAILY AVERAGE SNOW BY_HRU\n",
        "  :CustomOutput DAILY AVERAGE PRECIP BY_HRU\n",
        "  :CustomOutput DAILY AVERAGE PRECIP BY_HRU_GROUP\n",
        "  :CustomOutput DAILY AVERAGE ATMOSPHERE BY_HRU\n",
        "  :CustomOutput DAILY AVERAGE ATMOSPHERE BY_HRU_GROUP\n",
        "  :CustomOutput DAILY AVERAGE SOIL[0] BY_HRU\n",
        "  :CustomOutput DAILY AVERAGE SOIL[1] BY_HRU\n",
        *(["  :CustomOutput DAILY AVERAGE SOIL[2] BY_HRU\n"] if model_type == 'HBV' else []),
        "  :CustomOutput DAILY AVERAGE AET BY_HRU\n",
        "  :CustomOutput DAILY AVERAGE AET BY_HRU_GROUP\n",
        "  :CustomOutput DAILY AVERAGE From:GLACIER_ICE BY_BASIN\n",
        "  :CustomOutput DAILY AVERAGE From:GLACIER_ICE BY_HRU\n",
        "  :CustomOutput DAILY AVERAGE TEMP_AVE BY_HRU_GROUP\n",
        "  :CustomOutput DAILY AVERAGE TEMP_AVE BY_HRU\n",
        "  :CustomOutput DAILY AVERAGE POTENTIAL_MELT BY_HRU\n",
        "  :CustomOutput DAILY AVERAGE POTENTIAL_MELT BY_HRU_GROUP\n",
        "  :CustomOutput DAILY AVERAGE RAINFALL BY_HRU\n",
        "  :CustomOutput DAILY AVERAGE RAINFALL BY_HRU_GROUP\n",
        "  :CustomOutput DAILY AVERAGE SNOWFALL BY_HRU\n",
        "  :CustomOutput DAILY AVERAGE SNOWFALL BY_HRU_GROUP\n",
        "  :CustomOutput DAILY AVERAGE SNOW_FRAC BY_HRU\n",
        "  :CustomOutput DAILY AVERAGE SNOW_FRAC BY_HRU_GROUP\n",
        "  :CustomOutput DAILY AVERAGE Between:SNOW_LIQ.And.PONDED_WATER BY_HRU\n",
        "  :CustomOutput DAILY AVERAGE Between:SNOW_LIQ.And.PONDED_WATER BY_HRU_GROUP\n",
        "  :CustomOutput DAILY AVERAGE Between:SNOW_LIQ.And.PONDED_WATER BY_BASIN\n",
        "  :CustomOutput DAILY AVERAGE Between:SNOW.And.ATMOSPHERE BY_HRU\n",
        "  :CustomOutput DAILY AVERAGE Between:SNOW.And.ATMOSPHERE BY_HRU_GROUP\n",
        "  :WriteMassLoadings\n",
    ]

    # Snowmelt tracer (all models)
    transport_tracers = [
        "\n#Transport for Snowmelt and Glacier Melt Tracking\n",
        "\n",
        ":Transport SNOWMELT TRACER\n",
        ":FixedConcentration SNOWMELT SNOW 1.0\n",
    ]
    # Glacier tracers
    if coupled:
        # Coupled: track GloGEM-driven melt via PONDED_WATER + HRU groups (all models)
        transport_tracers += [
            "\n",
            ":Transport GLACIERMELT_ALL TRACER\n",
            ":FixedConcentration GLACIERMELT_ALL PONDED_WATER 1.0 ALL_GLACIER\n",
            "\n",
            ":Transport GLACIERMELT_SMALL TRACER\n",
            ":FixedConcentration GLACIERMELT_SMALL PONDED_WATER 1.0 SMALL_GLACIER\n",
            "\n",
            ":Transport GLACIERMELT_LARGE TRACER\n",
            ":FixedConcentration GLACIERMELT_LARGE PONDED_WATER 1.0 LARGE_GLACIER\n",
        ]
    elif model_type == 'HBV':
        # Uncoupled HBV: track Raven's internal glacier melt via GLACIER state
        transport_tracers += [
            "\n",
            ":Transport GLACIERMELT_ALL TRACER\n",
            ":FixedConcentration GLACIERMELT_ALL GLACIER 1.0\n",
        ]

    # Find #Output Options line and insert after it
    new_lines = []
    inserted = False
    for line in lines:
        new_lines.append(line)
        if "#Output Options" in line and not inserted:
            inserted = True
            new_lines.extend(output_options)
            new_lines.extend(transport_tracers)

    if not inserted:
        new_lines.append("\n#Output Options\n")
        new_lines.extend(output_options)
        new_lines.extend(transport_tracers)

    with open(rvi_path, "w") as f:
        f.writelines(new_lines)

    logger.info(f"  Injected output options + transport tracers into .rvi")


def _fix_leap_days(data_obs_dir, model_id):
    """Interpolate missing leap days in CMIP6 NetCDF files."""
    import xarray as xr
    import pandas as pd

    logger = logging.getLogger(__name__)

    for pattern in [f"cmip6_{model_id}_ssp126_*.nc", f"cmip6_{model_id}_historical_*.nc"]:
        for f in sorted(data_obs_dir.glob(pattern)):
            ds = xr.open_dataset(f)
            var = list(ds.data_vars)[0]
            da = ds[var].load()
            ds.close()

            times = pd.DatetimeIndex(da.time.values)
            full_idx = pd.date_range(times[0], times[-1], freq="D")
            missing = full_idx.difference(times)

            if len(missing) > 0:
                da_reindexed = da.reindex(time=full_idx)
                da_filled = da_reindexed.interpolate_na(dim="time", method="linear")
                ds_out = da_filled.to_dataset(name=var)
                ds_out[var].attrs = da.attrs

                # Preserve elevation if present
                ds_orig = xr.open_dataset(f)
                if "elevation" in ds_orig:
                    ds_out["elevation"] = ds_orig["elevation"].load()
                ds_orig.close()

                tmp = str(f) + ".tmp"
                ds_out.to_netcdf(tmp)
                shutil.move(tmp, str(f))
                logger.info(f"    Fixed {len(missing)} leap days in {f.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run future climate projections for multiple CMIP6 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 5 models
  python src/run_future_multi.py namelists_server/namelist_0101_glogem_subdaily_future.yaml

  # Run specific models only
  python src/run_future_multi.py namelist.yaml --models IPSL-CM6A-LR MRI-ESM2-0

  # Skip download (data already exists)
  python src/run_future_multi.py namelist.yaml --skip-download
        """,
    )

    parser.add_argument("namelist", type=str, help="Base future namelist YAML")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to run (default: all). Choices: {ALL_MODELS}",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force reprocessing of existing files")
    parser.add_argument(
        "--bbox", type=float, nargs=4,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        default=[37.5, 67.5, 30.0, 82.0],
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--log-file", type=str)

    args = parser.parse_args()

    namelist_path = Path(args.namelist)
    if not namelist_path.exists():
        print(f"Error: Namelist not found: {namelist_path}")
        sys.exit(1)

    models = args.models or ALL_MODELS

    # Validate model names
    for m in models:
        if m not in ALL_MODELS:
            print(f"Error: Unknown model '{m}'. Valid: {ALL_MODELS}")
            sys.exit(1)

    log_file = args.log_file  # Use parent's log file if provided, otherwise stdout only
    logger = setup_logging(args.verbose, log_file)

    logger.info("=" * 70)
    logger.info("MULTI-MODEL FUTURE CLIMATE PROJECTIONS")
    logger.info("=" * 70)
    logger.info(f"Base namelist: {namelist_path}")
    logger.info(f"Models to run: {models}")
    logger.info(f"Log file: {log_file}")

    workflow_start = time.time()
    results = {}

    for model_id in models:
        model_start = time.time()
        ok = run_single_model(
            namelist_path, model_id,
            skip_download=args.skip_download,
            force=args.force,
            bbox=args.bbox,
            verbose=args.verbose,
        )
        duration = time.time() - model_start
        results[model_id] = {"success": ok, "duration": duration}

    # Summary
    total_duration = time.time() - workflow_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("MULTI-MODEL SUMMARY")
    logger.info("=" * 70)
    for model_id, res in results.items():
        status = "SUCCESS" if res["success"] else "FAILED"
        logger.info(f"  {model_id:20s}  {status}  ({res['duration']/60:.1f} min)")
    logger.info(f"  {'TOTAL':20s}  {total_duration/60:.1f} min")

    n_ok = sum(1 for r in results.values() if r["success"])
    n_total = len(results)
    logger.info(f"\n  {n_ok}/{n_total} models completed successfully")

    sys.exit(0 if n_ok == n_total else 1)


if __name__ == "__main__":
    main()
