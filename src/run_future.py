#!/usr/bin/env python3
"""
Future Climate Projection Run for Raven

Orchestrates the full future climate pipeline:
  1. (Optional) Download CMIP6 raw data
  2. Create input files (includes CMIP6 downscaling, GloGEM, model files)
  3. Run Raven forward with calibrated parameters (no calibration)

Usage:
    python src/run_future.py namelists_server/namelist_0101_glogem_subdaily_future.yaml
    python src/run_future.py namelist.yaml --skip-download
    python src/run_future.py namelist.yaml --download-only

Author: Justine Berg
"""

import sys
import argparse
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime

import yaml

script_dir = Path(__file__).parent.absolute()
project_dir = script_dir.parent


def setup_logging(debug=False, log_file=None):
    """Setup logging configuration."""
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


def load_namelist(namelist_path):
    """Load and validate namelist configuration."""
    with open(namelist_path, "r") as f:
        nml = yaml.safe_load(f)

    required = ["gauge_id", "model_type", "start_date", "end_date", "main_dir"]
    missing = [k for k in required if k not in nml]
    if missing:
        raise ValueError(f"Missing required namelist fields: {missing}")

    if not nml.get("future", False):
        raise ValueError("Namelist must have 'future: true' for a future climate run")

    if not nml.get("cmip6_models") and not nml.get("cordex_models"):
        raise ValueError("Namelist must specify cmip6_models or cordex_models")

    return nml


def run_cmip6_download(nml, bbox=None):
    """Download CMIP6 data using the download script."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STEP 1: DOWNLOADING CMIP6 DATA")
    logger.info("=" * 60)

    download_script = project_dir / "downloads" / "download_cmip6.py"
    if not download_script.exists():
        logger.error(f"Download script not found: {download_script}")
        return False

    models = nml.get("cmip6_models", [])
    scenarios = nml.get("cmip6_scenarios", ["historical", "ssp126"])

    # Resolve cmip6_dir relative to main_dir
    main_dir = Path(nml["main_dir"])
    cmip6_dir = nml.get("cmip6_dir", "01_data/CMIP6")
    if not Path(cmip6_dir).is_absolute():
        cmip6_dir = str(main_dir / cmip6_dir)

    cmd = [
        sys.executable,
        str(download_script),
        "--models",
        *models,
        "--experiments",
        *scenarios,
        "--output-dir",
        cmip6_dir,
    ]

    if bbox:
        cmd.extend(["--bbox", *[str(b) for b in bbox]])

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        process = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        logger.info(process.stdout)
        logger.info("CMIP6 download completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"CMIP6 download failed (return code {e.returncode})")
        logger.error(e.stdout)
        return False


def run_input_creation(namelist_path, force_reprocess=False):
    """Run create_input_files.py to generate all Raven inputs."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STEP 2: CREATING INPUT FILES (incl. CMIP6 downscaling)")
    logger.info("=" * 60)

    create_script = script_dir / "create_input_files.py"
    cmd = [sys.executable, str(create_script), str(namelist_path)]
    if force_reprocess:
        cmd.append("--force")

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            logger.info(f"  {line.rstrip()}")

        process.wait()
        if process.returncode == 0:
            logger.info("Input file creation completed successfully")
            return True
        else:
            logger.error(f"Input file creation failed (return code {process.returncode})")
            return False
    except Exception as e:
        logger.error(f"Error running input creation: {e}")
        return False


def run_raven_forward(nml, namelist_path):
    """Run Raven forward (no calibration) with calibrated parameters."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STEP 3: RUNNING RAVEN FORWARD SIMULATION")
    logger.info("=" * 60)

    gauge_id = str(nml["gauge_id"])
    model_type = nml["model_type"]
    main_dir = Path(nml["main_dir"])
    config_dir = nml.get("config_dir", "03_model_setups_future")

    model_dir = main_dir / config_dir / f"catchment_{gauge_id}" / model_type
    output_dir = model_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_file = model_dir / f"{gauge_id}_{model_type}"
    raven_exe = nml.get("raven_executable", "Raven")

    if not Path(raven_exe).exists():
        logger.error(f"Raven executable not found: {raven_exe}")
        return False

    # Check that model input file exists
    rvi_file = model_dir / f"{gauge_id}_{model_type}.rvi"
    if not rvi_file.exists():
        logger.error(f"Model input file not found: {rvi_file}")
        logger.error("Run input creation first (Step 2)")
        return False

    cmd = [str(raven_exe), str(model_file), "-o", str(output_dir) + "/"]
    logger.info(f"Running: {' '.join(cmd)}")
    logger.info(f"  Model directory: {model_dir}")
    logger.info(f"  Output directory: {output_dir}")

    try:
        start_time = time.time()
        process = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        duration = time.time() - start_time

        logger.info(f"Raven completed in {duration:.1f} seconds")

        # Check for output hydrograph
        hydrograph = output_dir / f"{gauge_id}_{model_type}_Hydrographs.csv"
        if hydrograph.exists():
            logger.info(f"Hydrograph output: {hydrograph}")
            # Print first and last few lines to verify date range
            with open(hydrograph, "r") as f:
                lines = f.readlines()
            if len(lines) > 2:
                logger.info(f"  First data line: {lines[1].strip()}")
                logger.info(f"  Last data line:  {lines[-1].strip()}")
                logger.info(f"  Total lines: {len(lines) - 1}")
        else:
            logger.warning(f"Hydrograph file not found: {hydrograph}")

        if process.stderr:
            logger.debug(f"Raven stderr:\n{process.stderr}")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Raven failed (return code {e.returncode})")
        if e.stdout:
            logger.error(f"stdout:\n{e.stdout}")
        if e.stderr:
            logger.error(f"stderr:\n{e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run future climate projection with Raven",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (download + preprocess + run)
  python src/run_future.py namelists_server/namelist_0101_glogem_subdaily_future.yaml

  # Skip download (CMIP6 data already exists)
  python src/run_future.py namelist.yaml --skip-download

  # Only download CMIP6 data
  python src/run_future.py namelist.yaml --download-only

  # Force reprocessing of all input files
  python src/run_future.py namelist.yaml --skip-download --force
        """,
    )

    parser.add_argument("namelist", type=str, help="Path to future namelist YAML file")
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip CMIP6 data download"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download CMIP6 data, then stop",
    )
    parser.add_argument(
        "--skip-input-creation",
        action="store_true",
        help="Skip input file creation (assume files already exist)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force reprocessing of existing files",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        default=[37.5, 67.5, 30.0, 82.0],
        help="Bounding box for CMIP6 download (default: Indus basin)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug output")
    parser.add_argument("--log-file", type=str, help="Path to log file")

    args = parser.parse_args()

    namelist_path = Path(args.namelist)
    if not namelist_path.exists():
        print(f"Error: Namelist file not found: {namelist_path}")
        sys.exit(1)

    nml = load_namelist(namelist_path)

    # Setup logging
    if args.log_file:
        log_file = args.log_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"future_run_{nml['gauge_id']}_{timestamp}.log"

    logger = setup_logging(args.verbose, log_file)

    logger.info("=" * 70)
    logger.info("RAVEN FUTURE CLIMATE PROJECTION")
    logger.info("=" * 70)
    logger.info(f"Namelist: {namelist_path}")
    logger.info(f"Gauge ID: {nml['gauge_id']}")
    logger.info(f"Model: {nml['model_type']}")
    logger.info(f"Period: {nml['start_date']} to {nml['end_date']}")
    logger.info(f"CMIP6 models: {nml.get('cmip6_models', [])}")
    logger.info(f"Scenarios: {nml.get('cmip6_scenarios', [])}")
    logger.info(f"GloGEM model: {nml.get('glogem_model', 'N/A')}")
    logger.info(f"GloGEM scenario: {nml.get('glogem_scenario', 'N/A')}")

    workflow_start = time.time()
    steps_ok = []

    # Step 1: Download CMIP6 data
    if not args.skip_download:
        ok = run_cmip6_download(nml, bbox=args.bbox)
        steps_ok.append(ok)
        if not ok:
            logger.error("CMIP6 download failed. Use --skip-download if data exists.")
            sys.exit(1)
        if args.download_only:
            logger.info("Download-only mode: stopping after download.")
            sys.exit(0)
    else:
        logger.info("Skipping CMIP6 download (--skip-download)")
        steps_ok.append(True)

    # Step 2: Create input files (preprocessing + CMIP6 downscaling + GloGEM)
    if not args.skip_input_creation:
        ok = run_input_creation(namelist_path, force_reprocess=args.force)
        steps_ok.append(ok)
        if not ok:
            logger.error("Input creation failed.")
            sys.exit(1)
    else:
        logger.info("Skipping input creation (--skip-input-creation)")
        steps_ok.append(True)

    # Step 3: Run Raven forward
    ok = run_raven_forward(nml, namelist_path)
    steps_ok.append(ok)

    # Summary
    total_duration = time.time() - workflow_start
    logger.info("=" * 70)
    logger.info("WORKFLOW SUMMARY")
    logger.info("=" * 70)

    step_names = ["CMIP6 Download", "Input Creation", "Raven Forward Run"]
    for i, (name, success) in enumerate(zip(step_names, steps_ok)):
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {i + 1}. {name}: {status}")

    logger.info(f"Total duration: {total_duration / 60:.1f} minutes")

    main_dir = Path(nml["main_dir"])
    config_dir = nml.get("config_dir", "")
    output_dir = (
        main_dir
        / config_dir
        / f"catchment_{nml['gauge_id']}"
        / nml["model_type"]
        / "output"
    )
    logger.info(f"Output directory: {output_dir}")

    if all(steps_ok):
        logger.info("Future climate run completed successfully!")
        sys.exit(0)
    else:
        logger.error("Future climate run completed with errors!")
        sys.exit(1)


if __name__ == "__main__":
    main()
