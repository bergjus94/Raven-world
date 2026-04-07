#!/usr/bin/env python3
"""
Full Automated Pipeline: Calibration → Future Projections

Takes a historical namelist and runs everything end-to-end:
  Phase 1: Create historical input files (ERA5/HAR forcing, topo, Raven .rv*)
  Phase 2: Calibrate with SCEUA + postprocessing diagnostics
  Phase 3: Convert best_params.csv → calibrated_params YAML
  Phase 4: Generate future namelist from historical namelist
  Phase 5: CMIP6 download + multi-model future runs
  Phase 6: Ensemble diagnostic plot

Usage:
    python run_full_pipeline.py namelists/namelist_0101_glogem_subdaily.yaml
    python run_full_pipeline.py namelist.yaml --skip-calibration --skip-download
    python run_full_pipeline.py namelist.yaml --models GFDL-ESM4 IPSL-CM6A-LR

Author: Justine Berg
Date: March 2026
"""

import sys
import argparse
import subprocess
import os
import time
import logging
import tempfile
import copy
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
script_dir = Path(__file__).parent.absolute()
src_dir = script_dir / 'src'
sys.path.insert(0, str(src_dir))

from paths import get_paths

# All 5 ISIMIP3b models
ALL_MODELS = [
    "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL",
]

# Default future projection settings
FUTURE_DEFAULTS = {
    'start_date': '1981-01-01',
    'end_date': '2100-09-30',
    'warm_up_date': '1980-01-01',
    'scenarios': ['ssp126'],
    'models': ALL_MODELS,
}


# =============================================================================
# Logging + Subprocess Helpers
# =============================================================================

def setup_logging(verbose=False, log_file=None):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def run_subprocess(cmd, label, logger, timeout=7200):
    """Run a subprocess with real-time logging. Returns True on success."""
    logger.info(f"Running: {' '.join(str(c) for c in cmd)}")
    try:
        start = time.time()
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True,
        )
        while True:
            line = process.stdout.readline()
            if line == '' and process.poll() is not None:
                break
            if line:
                logger.info(f"  [{label}] {line.rstrip()}")

        rc = process.poll()
        duration = time.time() - start
        if rc == 0:
            logger.info(f"  {label} completed in {duration/60:.1f} min")
            return True
        else:
            logger.error(f"  {label} failed (exit code {rc})")
            return False
    except subprocess.TimeoutExpired:
        process.kill()
        logger.error(f"  {label} timed out after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"  {label} error: {e}")
        return False


# =============================================================================
# Naming helpers
# =============================================================================

def derive_config_suffix(config_dir):
    """Extract naming suffix from config_dir path.

    E.g. 'chandra_1/03_model_setups_glogem_subdaily' → 'glogem_subdaily'
    """
    last_part = Path(config_dir).name
    prefix = '03_model_setups_'
    if last_part.startswith(prefix):
        return last_part[len(prefix):]
    return last_part


# =============================================================================
# Phase 1: Create Historical Input Files
# =============================================================================

def phase1_create_inputs(namelist_path, nml, force=False, future_proj=None, logger=None):
    """Create historical Raven input files via create_input_files.py.

    If future projections are enabled, extends the ERA5/HAR date range to cover
    the full historical period (e.g. from 1980) so forcing data isn't processed twice.
    """
    logger = logger or logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("PHASE 1: CREATE HISTORICAL INPUT FILES")
    logger.info("=" * 70)

    # Check if .rv* files already exist
    gauge_id = str(nml['gauge_id'])
    model_type = nml['model_type']
    paths = get_paths(nml)
    model_dir = paths['model_dir']

    rv_extensions = ['.rvi', '.rvp', '.rvc', '.rvh', '.rvt']
    file_prefix = f"{gauge_id}_{model_type}"
    existing = [ext for ext in rv_extensions if (model_dir / f"{file_prefix}{ext}").exists()]

    if len(existing) == len(rv_extensions) and not force:
        logger.info(f"  All .rv* files already exist in {model_dir}, skipping.")
        return True

    # If future projections are enabled, create a temporary namelist with extended dates
    # so ERA5/HAR forcing covers the full historical period
    actual_namelist = namelist_path
    tmp_dir = None

    if future_proj and future_proj.get('enabled', False):
        future_warm_up = future_proj.get('warm_up_date', FUTURE_DEFAULTS['warm_up_date'])
        future_start = future_proj.get('start_date', FUTURE_DEFAULTS['start_date'])

        # Only extend if future dates are earlier than historical dates
        hist_warm_up = nml.get('warm_up_date', nml['start_date'])
        if future_warm_up < hist_warm_up:
            logger.info(f"  Extending forcing dates for future reuse: warm_up={future_warm_up}")
            extended_nml = copy.deepcopy(nml)
            extended_nml['warm_up_date'] = future_warm_up
            # Keep the original start_date — preprocessing uses warm_up_date for data range

            # Resolve relative paths before copying to temp dir
            nml_dir = Path(namelist_path).parent
            if 'params_dir' in extended_nml:
                params_path = Path(extended_nml['params_dir'])
                if not params_path.is_absolute():
                    extended_nml['params_dir'] = str((nml_dir / params_path).resolve())

            tmp_dir = tempfile.mkdtemp(prefix='pipeline_phase1_')
            tmp_path = Path(tmp_dir) / Path(namelist_path).name
            with open(tmp_path, 'w') as f:
                yaml.dump(extended_nml, f, default_flow_style=False, sort_keys=False)
            actual_namelist = str(tmp_path)
            logger.info(f"  Using extended namelist: {actual_namelist}")

    # Run create_input_files.py
    create_script = src_dir / 'create_input_files.py'
    cmd = [sys.executable, str(create_script), str(actual_namelist)]
    if force:
        cmd.append('--force')

    ok = run_subprocess(cmd, 'CREATE_INPUTS', logger)

    # Clean up temp dir
    if tmp_dir:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return ok


# =============================================================================
# Phase 2: Calibration + Postprocessing
# =============================================================================

def phase2_calibrate(namelist_path, nml, iterations=None, ngs=None, logger=None):
    """Run SCEUA calibration + postprocessing diagnostics."""
    logger = logger or logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("PHASE 2: CALIBRATION + POSTPROCESSING")
    logger.info("=" * 70)

    gauge_id = str(nml['gauge_id'])
    model_type = nml['model_type']
    paths = get_paths(nml)
    output_dir = paths['output_dir']

    # Check if already calibrated
    verified_csv = output_dir / f"{gauge_id}_{model_type}_VERIFIED_best_params.csv"
    if verified_csv.exists():
        logger.info(f"  Calibration already complete: {verified_csv}")
        return True

    # Run spotpy_optimize.py
    calibration_script = script_dir / 'spotpy_optimize.py'
    cmd = [sys.executable, str(calibration_script), '--namelist', str(namelist_path)]
    if iterations is not None:
        cmd.extend(['--iterations', str(iterations)])
    if ngs is not None:
        cmd.extend(['--ngs', str(ngs)])

    ok = run_subprocess(cmd, 'CALIBRATION', logger, timeout=14400)
    if not ok:
        return False

    # Run postprocessing
    logger.info("  Running postprocessing diagnostics...")
    try:
        import postprocessing
        config = {
            'main_dir': nml['main_dir'],
            'gauge_id': gauge_id,
            'model_type': model_type,
            'start_date': nml.get('start_date', '2000-01-01'),
            'end_date': nml.get('end_date', '2020-12-31'),
            'cali_start_date': nml.get('cali_start_date', nml.get('start_date')),
            'cali_end_date': nml.get('cali_end_date', '2009-12-31'),
            'coupled': nml.get('coupled', False),
            'raven_executable': nml.get('raven_executable'),
            'glogem_dir': nml.get('glogem_dir', None),
            '_config_key': nml.get('_config_key', 'baseline'),
        }
        results = postprocessing.run_complete_postprocessing(
            config=config,
            validation_start=nml.get('cali_end_date'),
            validation_end=nml.get('end_date'),
        )
        if results:
            logger.info("  Postprocessing complete")
        else:
            logger.warning("  Postprocessing returned None (non-fatal)")
    except Exception as e:
        logger.warning(f"  Postprocessing error (non-fatal): {e}")

    return True


# =============================================================================
# Phase 3: Convert best_params.csv → calibrated_params YAML
# =============================================================================

def phase3_convert_params(nml, logger=None):
    """Convert VERIFIED_best_params.csv to a calibrated_params YAML file.

    Returns the path to the generated YAML, or None on failure.
    """
    logger = logger or logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("PHASE 3: CONVERT BEST PARAMS → CALIBRATED YAML")
    logger.info("=" * 70)

    gauge_id = str(nml['gauge_id'])
    model_type = nml['model_type']
    config_key = nml.get('_config_key', 'baseline')
    paths = get_paths(nml)

    output_dir = paths['output_dir']
    verified_csv = output_dir / f"{gauge_id}_{model_type}_VERIFIED_best_params.csv"

    # Output path
    calibrated_yaml = script_dir / 'src' / 'config' / f"calibrated_params_{config_key}_{model_type}_{gauge_id}.yaml"

    # Check if already exists
    if calibrated_yaml.exists():
        logger.info(f"  Calibrated params YAML already exists: {calibrated_yaml}")
        return calibrated_yaml

    if not verified_csv.exists():
        logger.error(f"  VERIFIED_best_params.csv not found: {verified_csv}")
        return None

    # Load default params and best params
    default_params_path = script_dir / 'src' / 'config' / 'default_params.yaml'
    with open(default_params_path) as f:
        default_params = yaml.safe_load(f)

    best_df = pd.read_csv(verified_csv)
    if len(best_df) == 0:
        logger.error("  best_params CSV is empty")
        return None

    best_row = best_df.iloc[0]
    obj_value = best_row.get('objective', None)
    iteration = best_row.get('iteration', None)

    # Get model section from defaults
    if model_type not in default_params:
        logger.error(f"  Model type '{model_type}' not found in default_params.yaml")
        return None

    model_defaults = default_params[model_type]
    names_map = model_defaults['names']  # {key: raven_name}

    # Build reverse mapping: {raven_name: key}
    reverse_map = {raven_name: key for key, raven_name in names_map.items()}

    # Start with default init values
    calibrated_init = dict(model_defaults['init'])

    # Replace with calibrated values from CSV
    for col in best_df.columns:
        if col in ('objective', 'iteration'):
            continue
        if col in reverse_map:
            key = reverse_map[col]
            calibrated_init[key] = float(best_row[col])
        elif col in calibrated_init:
            # Column name matches a key directly (e.g. tied params)
            calibrated_init[col] = float(best_row[col])

    # Compute derived/tied parameters from calibrated values
    tied = _compute_tied_params(model_type, calibrated_init, names_map)
    calibrated_init.update(tied)

    # Build output YAML structure (same format as default_params)
    out = {
        model_type: {
            'names': model_defaults['names'],
            'init': calibrated_init,
            'lower': model_defaults['lower'],
            'upper': model_defaults['upper'],
        }
    }
    if 'optional' in model_defaults:
        out[model_type]['optional'] = model_defaults['optional']

    # Write with header comment
    header = (
        f"# Calibrated Parameters for catchment {gauge_id} — {model_type} {config_key.replace('_', ' ').title()}\n"
        f"# Source: {verified_csv.relative_to(Path(nml['main_dir']))}\n"
    )
    if obj_value is not None:
        header += f"# Calibration: SCEUA, KGE = {obj_value:.3f}"
        if iteration is not None:
            header += f", iteration {int(iteration)}"
        header += "\n"
    header += (
        f"# Period: {nml.get('start_date', '?')} to {nml.get('cali_end_date', '?')} (calibration), "
        f"{nml.get('cali_end_date', '?')} to {nml.get('end_date', '?')} (validation)\n"
        f"# Mode: {'coupled' if nml.get('coupled') else 'uncoupled'}"
    )
    if nml.get('coupled'):
        header += f" ({nml.get('subdaily_method', '')})"
    header += "\n\n"

    calibrated_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(calibrated_yaml, 'w') as f:
        f.write(header)
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)

    logger.info(f"  Wrote calibrated params: {calibrated_yaml}")
    return calibrated_yaml


def _compute_tied_params(model_type, init_dict, names_map):
    """Compute tied/derived parameters from calibrated values."""
    tied = {}

    if model_type == 'HBV':
        # HBV_Time_To_Peak = X11 / 2  (X11 key maps to HBV_T_Conc_Max_Bas)
        if 'X11' in init_dict:
            tied['HBV_Time_To_Peak'] = init_dict['X11'] / 2.0
        # HBV_Initial_Thickness_Topsoil = X17 * 500
        if 'X17' in init_dict:
            tied['HBV_Initial_Thickness_Topsoil'] = init_dict['X17'] * 500.0

    elif model_type == 'GR4J':
        # Airsnow_Coeff = 1.0 - CEMANEIGE_X2
        if 'CEMANEIGE_X2' in init_dict:
            tied['Airsnow_Coeff'] = 1.0 - init_dict['CEMANEIGE_X2']

    elif model_type == 'MOHYSE':
        # Mohyse_Gamma_Scale = 1.0 / X10 (MOHYSE_Gamma_Scale_Aux)
        if 'X10' in init_dict and init_dict['X10'] > 0:
            tied['Mohyse_Gamma_Scale'] = 1.0 / init_dict['X10']

    return tied


# =============================================================================
# Phase 4: Generate Future Namelist
# =============================================================================

def phase4_generate_future_namelist(namelist_path, nml, calibrated_yaml, future_proj, logger=None):
    """Generate a future namelist from the historical namelist.

    Returns the path to the generated future namelist, or None on failure.
    """
    logger = logger or logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("PHASE 4: GENERATE FUTURE NAMELIST")
    logger.info("=" * 70)

    namelist_path = Path(namelist_path)
    stem = namelist_path.stem
    future_path = namelist_path.parent / f"{stem}_FUTURE.yaml"

    if future_path.exists():
        logger.info(f"  Future namelist already exists: {future_path}")
        return future_path

    gauge_id = str(nml['gauge_id'])
    config_key = nml.get('_config_key', 'baseline')
    main_dir = nml['main_dir']

    # Start from a deep copy of the historical namelist
    future_nml = copy.deepcopy(nml)

    # Core future settings
    future_nml['future'] = True
    future_nml['template'] = False
    future_nml['start_date'] = future_proj.get('start_date', FUTURE_DEFAULTS['start_date'])
    future_nml['end_date'] = future_proj.get('end_date', FUTURE_DEFAULTS['end_date'])
    future_nml['warm_up_date'] = future_proj.get('warm_up_date', FUTURE_DEFAULTS['warm_up_date'])
    future_nml['cali_end_date'] = future_nml['end_date']  # No calibration split in future mode

    # Scenario: pick first non-historical scenario for GloGEM file naming
    # (historical is only used for CMIP6 bias correction training, not for GloGEM)
    scenarios = future_proj.get('scenarios', FUTURE_DEFAULTS['scenarios'])
    future_scenarios = [s for s in scenarios if s != 'historical']
    future_nml['glogem_scenario'] = future_scenarios[0] if future_scenarios else scenarios[0]

    # Calibrated params path (relative to namelist location)
    if calibrated_yaml:
        try:
            rel_params = os.path.relpath(calibrated_yaml, namelist_path.parent)
        except ValueError:
            rel_params = str(calibrated_yaml)
        future_nml['params_dir'] = rel_params

    # CMIP6 settings
    future_nml['cmip6_dir'] = nml.get('cmip6_dir', f"{main_dir}/01_data/CMIP6")
    models = future_proj.get('models', FUTURE_DEFAULTS['models'])
    future_nml['cmip6_models'] = models
    # Ensure 'historical' is included (needed for bias correction) without duplicates
    all_scenarios = ['historical'] + [s for s in scenarios if s != 'historical']
    future_nml['cmip6_scenarios'] = all_scenarios
    future_nml['cmip6_train_start'] = '1980-01-01'
    future_nml['cmip6_train_end'] = '2014-12-31'

    # Remove calibration section (not needed for forward run)
    future_nml.pop('calibration', None)

    # Remove future_projections section (meta-config, not needed in the future namelist itself)
    future_nml.pop('future_projections', None)

    # Write with header comment
    header = (
        f"# =============================================================================\n"
        f"# AUTO-GENERATED future namelist for gauge {gauge_id}\n"
        f"# Source: {namelist_path.name}\n"
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"# =============================================================================\n\n"
    )

    with open(future_path, 'w') as f:
        f.write(header)
        yaml.dump(future_nml, f, default_flow_style=False, sort_keys=False)

    logger.info(f"  Wrote future namelist: {future_path}")
    return future_path


# =============================================================================
# Phase 5: CMIP6 Download + Multi-Model Future Runs
# =============================================================================

def phase5_future_runs(future_namelist, models=None, skip_download=False,
                       force=False, bbox=None, verbose=False, logger=None):
    """Run future projections via run_future_multi.py."""
    logger = logger or logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("PHASE 5: CMIP6 MULTI-MODEL FUTURE RUNS")
    logger.info("=" * 70)

    future_script = src_dir / 'run_future_multi.py'
    cmd = [sys.executable, str(future_script), str(future_namelist)]

    if models:
        cmd.extend(['--models'] + models)
    if skip_download:
        cmd.append('--skip-download')
    if force:
        cmd.append('--force')
    if bbox:
        cmd.extend(['--bbox'] + [str(b) for b in bbox])
    if verbose:
        cmd.append('--verbose')

    return run_subprocess(cmd, 'FUTURE_MULTI', logger, timeout=14400)


# =============================================================================
# Phase 6: Ensemble Diagnostic Plot
# =============================================================================

def phase6_ensemble_diagnostics(nml, future_proj, logger=None):
    """Create ensemble diagnostic plot from multi-model future outputs."""
    logger = logger or logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("PHASE 6: ENSEMBLE DIAGNOSTIC PLOT")
    logger.info("=" * 70)

    gauge_id = str(nml['gauge_id'])
    model_type = nml['model_type']
    paths = get_paths(nml)
    output_dir = paths['output_dir']

    if not output_dir.exists():
        logger.warning(f"  Future output directory not found: {output_dir}")
        return False

    # GloGEM model ID mapping (uppercase CMIP6 → lowercase GloGEM folder name)
    glogem_ids = {
        "GFDL-ESM4": "gfdl-esm4",
        "IPSL-CM6A-LR": "ipsl-cm6a-lr",
        "MPI-ESM1-2-HR": "mpi-esm1-2-hr",
        "MRI-ESM2-0": "mri-esm2-0",
        "UKESM1-0-LL": "ukesm1-0-ll",
    }

    models = future_proj.get('models', FUTURE_DEFAULTS['models'])

    # Collect hydrographs from each model
    all_hydro = {}
    for model_id in models:
        glogem_id = glogem_ids.get(model_id, model_id.lower())
        model_output = output_dir / glogem_id
        hydro_file = model_output / f"{gauge_id}_{model_type}_Hydrographs.csv"

        if not hydro_file.exists():
            logger.warning(f"  Missing hydrograph for {model_id}: {hydro_file}")
            continue

        try:
            df = pd.read_csv(hydro_file, parse_dates=['date'])
            # Simulated discharge column: 'XXXX [m3/s]' (gauge_id or sub_xxx)
            sim_cols = [c for c in df.columns if '[m3/s]' in c and 'observed' not in c.lower()]
            if sim_cols:
                all_hydro[model_id] = df[['date', sim_cols[0]]].rename(
                    columns={sim_cols[0]: 'Q_sim'}
                ).set_index('date')
                logger.info(f"  Loaded {model_id}: {len(df)} timesteps")
        except Exception as e:
            logger.warning(f"  Error reading {model_id}: {e}")

    if not all_hydro:
        logger.warning("  No model outputs found, skipping ensemble plot")
        return False

    # Create ensemble diagnostic plot
    try:
        _plot_ensemble(all_hydro, gauge_id, output_dir, logger)
        return True
    except Exception as e:
        logger.error(f"  Ensemble plot error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def _plot_ensemble(all_hydro, gauge_id, output_dir, logger):
    """Create a 4-panel ensemble diagnostic figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_hydro)))

    # Panel 1: Full time series
    ax = axes[0, 0]
    for (model_id, df), color in zip(all_hydro.items(), colors):
        ax.plot(df.index, df['Q_sim'], label=model_id, alpha=0.7, linewidth=0.5, color=color)
    ax.set_ylabel('Discharge [m³/s]')
    ax.set_title(f'Catchment {gauge_id} — Ensemble Hydrographs')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: Monthly climatology (2015–2100)
    ax = axes[0, 1]
    for (model_id, df), color in zip(all_hydro.items(), colors):
        monthly = df['Q_sim'].groupby(df.index.month).mean()
        ax.plot(monthly.index, monthly.values, label=model_id, marker='o',
                markersize=3, color=color)
    ax.set_xlabel('Month')
    ax.set_ylabel('Mean Discharge [m³/s]')
    ax.set_title('Monthly Climatology')
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.3)

    # Panel 3: Annual mean trends
    ax = axes[1, 0]
    for (model_id, df), color in zip(all_hydro.items(), colors):
        annual = df['Q_sim'].resample('YE').mean()
        ax.plot(annual.index, annual.values, label=model_id, alpha=0.8, color=color)
    ax.set_ylabel('Annual Mean Discharge [m³/s]')
    ax.set_title('Annual Trends')
    ax.grid(True, alpha=0.3)

    # Panel 4: Decadal seasonal change (2070-2099 vs 2015-2044)
    ax = axes[1, 1]
    for (model_id, df), color in zip(all_hydro.items(), colors):
        early = df.loc['2015':'2044']
        late = df.loc['2070':'2099']
        if len(early) > 0 and len(late) > 0:
            early_monthly = early['Q_sim'].groupby(early.index.month).mean()
            late_monthly = late['Q_sim'].groupby(late.index.month).mean()
            change_pct = ((late_monthly - early_monthly) / early_monthly) * 100
            ax.bar(change_pct.index + (list(all_hydro.keys()).index(model_id) - len(all_hydro)/2) * 0.15,
                   change_pct.values, width=0.15, label=model_id, alpha=0.8, color=color)
    ax.set_xlabel('Month')
    ax.set_ylabel('Change [%]')
    ax.set_title('Seasonal Change: 2070–2099 vs 2015–2044')
    ax.set_xticks(range(1, 13))
    ax.axhline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'ensemble_diagnostic.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved ensemble plot: {plot_path}")


# =============================================================================
# Multi-config catchment runner
# =============================================================================

def _build_sub_argv(args, namelist_path, log_file, skip_preprocessing=False):
    """Build subprocess argv from parsed args."""
    sub_argv = [str(namelist_path), '--log-file', log_file]
    if skip_preprocessing or args.skip_preprocessing:
        sub_argv.append('--skip-preprocessing')
    if args.skip_calibration:
        sub_argv.append('--skip-calibration')
    if args.skip_download:
        sub_argv.append('--skip-download')
    if args.skip_future:
        sub_argv.append('--skip-future')
    if args.force:
        sub_argv.append('--force')
    if args.iterations:
        sub_argv.extend(['--iterations', str(args.iterations)])
    if args.ngs:
        sub_argv.extend(['--ngs', str(args.ngs)])
    if args.bbox:
        sub_argv.extend(['--bbox'] + [str(b) for b in args.bbox])
    return [sys.executable, str(Path(__file__).resolve())] + sub_argv


def _run_multi_config(catchment_nml, args):
    """Run the full pipeline for each model × configuration in a catchment namelist.

    Two-phase approach to avoid race conditions on shared files:
      Phase A: For each config, run input creation (Phase 1) sequentially
               using the first model. This creates shared topo/meteo files.
      Phase B: Run all model × config calibration + future jobs in parallel
               (skipping preprocessing since Phase A already did it).

    Returns 0 if all succeeded, 1 if any failed.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from config_merge import load_config

    catchment_id = str(catchment_nml['catchment'])
    models = catchment_nml.get('models', ['HBV'])
    configurations = catchment_nml['configurations']
    max_workers = catchment_nml.get('parallel', args.parallel)

    # Extract overrides from the catchment namelist
    overrides = {}
    for key in ('start_date', 'end_date', 'cali_end_date', 'warm_up_date',
                'display_name', 'gauge_id'):
        if key in catchment_nml:
            overrides[key] = catchment_nml[key]
    if 'calibration' in catchment_nml:
        overrides['calibration'] = catchment_nml['calibration']
    if 'future' in catchment_nml:
        overrides['future_projections'] = catchment_nml['future']

    total = len(models) * len(configurations)

    print("=" * 70)
    print(f"MULTI-CONFIG PIPELINE: catchment {catchment_id}")
    print(f"  Models: {models}")
    print(f"  Configurations: {configurations}")
    print(f"  Total runs: {total}")
    print(f"  Parallel workers: {max_workers}")
    print("=" * 70)

    start_time = time.time()

    # =====================================================================
    # Phase A: Sequential input creation (one per config, using first model)
    # =====================================================================
    if not args.skip_preprocessing:
        print(f"\n{'='*70}")
        print(f"PHASE A: Sequential input creation ({len(configurations)} configs)")
        print(f"{'='*70}")

        for config in configurations:
            # Use first model for input creation (shared files are model-independent)
            first_model = models[0]
            run_key = f"{config}/{first_model}"

            try:
                nml, tmp_path = load_config(
                    catchment=catchment_id, configuration=config,
                    model=first_model, env=args.env,
                    overrides=overrides if overrides else None,
                )
            except Exception as e:
                print(f"  [{config}] Failed to load config: {e}")
                continue

            log_file = f"pipeline_{catchment_id}_{config}_inputs.log"
            # Run only Phase 1 (skip calibration + future)
            cmd = _build_sub_argv(args, Path(tmp_path).resolve(), log_file)
            # Override: only do preprocessing
            cmd_phase1 = [a for a in cmd if a not in ('--skip-preprocessing',)]
            cmd_phase1.extend(['--skip-calibration', '--skip-future'])

            elapsed = (time.time() - start_time) / 60
            print(f"  [{config:30s}] Creating input files...  ({elapsed:.0f}min)")

            try:
                proc = subprocess.run(
                    cmd_phase1, capture_output=True, text=True, timeout=7200,
                )
                status = "OK" if proc.returncode == 0 else "FAILED"
                if proc.returncode != 0:
                    print(f"  [{config:30s}] Input creation FAILED — see {log_file}")
            except subprocess.TimeoutExpired:
                status = "TIMEOUT"
                print(f"  [{config:30s}] Input creation TIMED OUT")
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass

            elapsed = (time.time() - start_time) / 60
            print(f"  [{config:30s}] {status}  ({elapsed:.0f}min)")

            # Also create input files for other model types (fast — only .rv* files differ)
            for model in models[1:]:
                try:
                    nml2, tmp2 = load_config(
                        catchment=catchment_id, configuration=config,
                        model=model, env=args.env,
                        overrides=overrides if overrides else None,
                    )
                    log2 = f"pipeline_{catchment_id}_{config}_{model}_inputs.log"
                    cmd2 = _build_sub_argv(args, Path(tmp2).resolve(), log2)
                    cmd2_phase1 = [a for a in cmd2 if a not in ('--skip-preprocessing',)]
                    cmd2_phase1.extend(['--skip-calibration', '--skip-future'])
                    subprocess.run(cmd2_phase1, capture_output=True, text=True, timeout=600)
                except Exception:
                    pass
                finally:
                    try:
                        Path(tmp2).unlink(missing_ok=True)
                    except OSError:
                        pass

    # =====================================================================
    # Phase B: Parallel calibration (skip preprocessing + future)
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"PHASE B: Parallel calibration ({total} jobs, {max_workers} workers)")
    print(f"{'='*70}")

    # Prepare all jobs — skip preprocessing AND future (future handled in Phase C)
    job_info = {}
    for config in configurations:
        for model in models:
            run_key = f"{config}/{model}"
            try:
                nml, tmp_path = load_config(
                    catchment=catchment_id, configuration=config,
                    model=model, env=args.env,
                    overrides=overrides if overrides else None,
                )
            except Exception as e:
                print(f"  [{run_key}] Failed to load config: {e}")
                job_info[run_key] = None
                continue

            log_file = f"pipeline_{catchment_id}_{config}_{model}.log"
            cmd = _build_sub_argv(args, Path(tmp_path).resolve(), log_file,
                                  skip_preprocessing=True)
            # Always skip future in Phase B — handled sequentially in Phase C
            if '--skip-future' not in cmd:
                cmd.append('--skip-future')
            job_info[run_key] = (cmd, tmp_path, log_file)

    # Run in parallel
    results = {}

    def _run_one(run_key):
        info = job_info[run_key]
        if info is None:
            return run_key, False
        cmd, tmp_path, log_file = info
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=28800,
            )
            ok = proc.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
        return run_key, ok

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one, key): key for key in job_info}
        for future in as_completed(futures):
            run_key, ok = future.result()
            results[run_key] = ok
            elapsed = time.time() - start_time
            status = "OK" if ok else "FAILED"
            done = len(results)
            log = job_info[run_key][2] if job_info[run_key] else ""
            print(f"  [{done}/{total}] {run_key:40s} {status:6s}  ({elapsed/60:.0f}min)  {log}")

    # =====================================================================
    # Phase C: Future projections
    #   C1: Sequential — first model per config (creates shared CMIP6 files)
    #   C2: Parallel — remaining models (CMIP6 files already exist)
    # =====================================================================
    if not args.skip_future:
        def _build_future_cmd(config, model, log_file):
            """Build a future-only pipeline command."""
            nml, tmp_path = load_config(
                catchment=catchment_id, configuration=config,
                model=model, env=args.env,
                overrides=overrides if overrides else None,
            )
            cmd = _build_sub_argv(args, Path(tmp_path).resolve(), log_file,
                                  skip_preprocessing=True)
            cmd = [a for a in cmd if a != '--skip-future']
            if '--skip-calibration' not in cmd:
                cmd.append('--skip-calibration')
            return cmd, tmp_path

        # C1: First model — sequential (creates CMIP6 downscaled files)
        first_model = models[0]
        print(f"\n{'='*70}")
        print(f"PHASE C1: Future projections — {first_model} ({len(configurations)} configs, sequential)")
        print(f"{'='*70}")

        for config in configurations:
            try:
                cmd, tmp_path = _build_future_cmd(
                    config, first_model,
                    f"pipeline_{catchment_id}_{config}_{first_model}_future.log")
            except Exception as e:
                print(f"  [{config}] Failed to load config: {e}")
                continue

            elapsed = (time.time() - start_time) / 60
            print(f"  [{config:30s}] {first_model} future...  ({elapsed:.0f}min)")

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
                status = "OK" if proc.returncode == 0 else "FAILED"
                if proc.returncode != 0:
                    print(f"  [{config:30s}] FAILED — see pipeline_{catchment_id}_{config}_{first_model}_future.log")
            except subprocess.TimeoutExpired:
                status = "TIMEOUT"
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass

            elapsed = (time.time() - start_time) / 60
            print(f"  [{config:30s}] {status}  ({elapsed:.0f}min)")

        # C2: Remaining models — parallel (CMIP6 files already exist)
        remaining_models = models[1:]
        if remaining_models:
            future_jobs = [(config, model) for config in configurations for model in remaining_models]
            print(f"\n{'='*70}")
            print(f"PHASE C2: Future projections — remaining models ({len(future_jobs)} jobs, {max_workers} workers)")
            print(f"{'='*70}")

            future_job_info = {}
            for config, model in future_jobs:
                run_key = f"{config}/{model}"
                try:
                    log_file = f"pipeline_{catchment_id}_{config}_{model}_future.log"
                    cmd, tmp_path = _build_future_cmd(config, model, log_file)
                    future_job_info[run_key] = (cmd, tmp_path, log_file)
                except Exception as e:
                    print(f"  [{run_key}] Failed to load config: {e}")
                    future_job_info[run_key] = None

            def _run_future_one(run_key):
                info = future_job_info[run_key]
                if info is None:
                    return run_key, False
                cmd, tmp_path, log_file = info
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
                    ok = proc.returncode == 0
                except subprocess.TimeoutExpired:
                    ok = False
                finally:
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except OSError:
                        pass
                return run_key, ok

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_run_future_one, key): key for key in future_job_info}
                n_future = len(future_job_info)
                for i, future in enumerate(as_completed(futures), 1):
                    run_key, ok = future.result()
                    elapsed = time.time() - start_time
                    status = "OK" if ok else "FAILED"
                    log = future_job_info[run_key][2] if future_job_info[run_key] else ""
                    print(f"  [{i}/{n_future}] {run_key:40s} {status:6s}  ({elapsed/60:.0f}min)  {log}")

    # Summary
    elapsed_total = (time.time() - start_time) / 60
    print(f"\n{'='*70}")
    print(f"MULTI-CONFIG SUMMARY: catchment {catchment_id}  ({elapsed_total:.1f} min)")
    print(f"{'='*70}")
    for run_key in sorted(results):
        ok = results[run_key]
        print(f"  {'✓' if ok else '✗'} {run_key}")
    n_ok = sum(results.values())
    print(f"\n  {n_ok}/{total} succeeded")

    return 0 if all(results.values()) else 1


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Full pipeline: calibration → future projections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_pipeline.py namelists/namelist_0101_glogem_subdaily.yaml
  python run_full_pipeline.py namelist.yaml --skip-calibration --skip-download
  python run_full_pipeline.py namelist.yaml --models GFDL-ESM4 IPSL-CM6A-LR --scenarios ssp126
        """,
    )

    parser.add_argument('namelist', nargs='?', default=None,
                        help='Path to historical namelist YAML (legacy mode)')
    parser.add_argument('--catchment', '-c', type=str,
                        help='Catchment ID (e.g. 0101)')
    parser.add_argument('--config', '-C', type=str, default='baseline',
                        help='Configuration name (e.g. glogem, baseline)')
    parser.add_argument('--model', '-m', type=str, default='HBV',
                        help='Hydrological model (HBV, HMETS, HYMOD, MOHYSE)')
    parser.add_argument('--env', type=str, default=None,
                        help='Environment: "local" or "server" (auto-detected)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip Phase 1 (assume .rv* files exist)')
    parser.add_argument('--skip-calibration', action='store_true',
                        help='Skip Phase 2 (assume best_params.csv exists)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip CMIP6 download in Phase 5')
    parser.add_argument('--skip-future', action='store_true',
                        help='Only run calibration (Phases 1-2), skip future projections')
    parser.add_argument('--models', nargs='+', default=None,
                        help=f'CMIP6 models to run (overrides namelist). Choices: {ALL_MODELS}')
    parser.add_argument('--scenarios', nargs='+', default=None,
                        help='SSP scenarios (overrides namelist). Default: ssp126')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Override SCEUA iterations')
    parser.add_argument('--ngs', type=int, default=None,
                        help='Override SCEUA complexes')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force reprocessing of existing files')
    parser.add_argument('--bbox', type=float, nargs=4,
                        metavar=('N', 'W', 'S', 'E'),
                        help='CMIP6 download bounding box')
    parser.add_argument('--parallel', '-j', type=int, default=30,
                        help='Max parallel workers for multi-config runs (default: 30)')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--log-file', type=str, default=None)

    args = parser.parse_args()

    # Load namelist (legacy, composable, or multi-config catchment mode)
    if args.namelist:
        namelist_path = Path(args.namelist).resolve()
        if not namelist_path.exists():
            print(f"Error: Namelist not found: {namelist_path}")
            sys.exit(1)
        with open(namelist_path) as f:
            nml = yaml.safe_load(f)

        # Detect multi-config catchment namelist (has 'catchment' + 'configurations')
        if 'catchment' in nml and 'configurations' in nml:
            sys.exit(_run_multi_config(nml, args))

    elif args.catchment:
        from config_merge import load_config
        nml, namelist_path = load_config(
            catchment=args.catchment,
            configuration=args.config,
            model=args.model,
            env=args.env,
        )
        namelist_path = Path(namelist_path).resolve()
        print(f"Merged config: catchment={args.catchment}, config={args.config}, model={args.model}")
    else:
        parser.error("Provide either a namelist path or --catchment")

    gauge_id = str(nml['gauge_id'])
    model_type = nml['model_type']

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log_file or f"pipeline_{gauge_id}_{timestamp}.log"
    logger = setup_logging(args.verbose, log_file)

    # Detect if this is a future namelist (has future: true and cmip6_models)
    is_future_namelist = nml.get('future', False) and 'cmip6_models' in nml

    # Resolve future projections config (namelist + CLI overrides)
    if is_future_namelist:
        # Future namelist passed directly — build future_proj from its contents
        future_proj = {
            'enabled': True,
            'models': nml.get('cmip6_models', None) or FUTURE_DEFAULTS['models'],
            'scenarios': [s for s in nml.get('cmip6_scenarios', ['ssp126']) if s != 'historical'],
        }
        future_enabled = not args.skip_future
    else:
        future_proj = nml.get('future_projections', {})
        future_enabled = future_proj.get('enabled', False) and not args.skip_future

    if future_enabled:
        # CLI overrides
        if args.models:
            future_proj['models'] = args.models
        if args.scenarios:
            future_proj['scenarios'] = args.scenarios

    logger.info("=" * 70)
    logger.info("RAVEN FULL PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now()}")
    logger.info(f"Namelist: {namelist_path}")
    logger.info(f"Gauge: {gauge_id}, Model: {model_type}")
    if is_future_namelist:
        logger.info(f"Mode: FUTURE NAMELIST (skipping calibration, using existing params)")
    logger.info(f"Future projections: {'ENABLED' if future_enabled else 'DISABLED'}")
    if future_enabled:
        logger.info(f"  Models: {future_proj.get('models', FUTURE_DEFAULTS['models'])}")
        logger.info(f"  Scenarios: {future_proj.get('scenarios', FUTURE_DEFAULTS['scenarios'])}")
    logger.info(f"Log file: {log_file}")

    workflow_start = time.time()
    phase_results = {}

    try:
        if is_future_namelist and future_enabled:
            # =================================================================
            # FUTURE NAMELIST MODE: skip Phases 1-4, run Phases 5-6 directly
            # =================================================================

            # Phase 3: Convert params (still needed if YAML doesn't exist yet)
            hist_nml = copy.deepcopy(nml)
            calibrated_yaml = phase3_convert_params(hist_nml, logger=logger)
            phase_results['Phase 3: Params YAML'] = calibrated_yaml is not None
            if calibrated_yaml is None:
                logger.error("Phase 3 failed — no calibrated params found.")
                logger.error("  Run calibration first, or check that VERIFIED_best_params.csv exists in")
                logger.error(f"  {Path(nml['main_dir']) / hist_config_dir}")
                _print_summary(phase_results, time.time() - workflow_start, logger)
                sys.exit(1)

            # Phase 4: Skip — future namelist already provided
            logger.info("Skipping Phase 4 (future namelist provided directly)")
            phase_results['Phase 4: Future Namelist'] = True

            # Phase 5: Future runs
            models = future_proj.get('models', FUTURE_DEFAULTS['models'])
            ok = phase5_future_runs(
                namelist_path, models=models,
                skip_download=args.skip_download,
                force=args.force, bbox=args.bbox,
                verbose=args.verbose, logger=logger,
            )
            phase_results['Phase 5: Future Runs'] = ok

            # Phase 6: Ensemble plot
            ok6 = phase6_ensemble_diagnostics(nml, future_proj, logger=logger)
            phase_results['Phase 6: Ensemble Plot'] = ok6

            # Phase 7: Full future postprocessing
            try:
                from postprocessing_future import run_future_postprocessing
                logger.info("=" * 70)
                logger.info("PHASE 7: FUTURE ENSEMBLE POSTPROCESSING")
                logger.info("=" * 70)
                run_future_postprocessing(str(namelist_path), skip_errors=True)
                phase_results['Phase 7: Future Postprocessing'] = True
            except Exception as e:
                logger.error(f"Phase 7 failed: {e}")
                phase_results['Phase 7: Future Postprocessing'] = False

        else:
            # =================================================================
            # HISTORICAL NAMELIST MODE: full pipeline Phases 1-6
            # =================================================================

            # Phase 1: Create input files
            if not args.skip_preprocessing:
                ok = phase1_create_inputs(
                    namelist_path, nml, force=args.force,
                    future_proj=future_proj if future_enabled else None,
                    logger=logger,
                )
                phase_results['Phase 1: Input Files'] = ok
                if not ok:
                    logger.error("Phase 1 failed. Stopping.")
                    _print_summary(phase_results, time.time() - workflow_start, logger)
                    sys.exit(1)
            else:
                logger.info("Skipping Phase 1 (--skip-preprocessing)")
                phase_results['Phase 1: Input Files'] = True

            # Phase 2: Calibration
            if not args.skip_calibration:
                ok = phase2_calibrate(
                    namelist_path, nml,
                    iterations=args.iterations, ngs=args.ngs, logger=logger,
                )
                phase_results['Phase 2: Calibration'] = ok
                if not ok:
                    logger.error("Phase 2 failed. Stopping.")
                    _print_summary(phase_results, time.time() - workflow_start, logger)
                    sys.exit(1)
            else:
                logger.info("Skipping Phase 2 (--skip-calibration)")
                phase_results['Phase 2: Calibration'] = True

            # Phases 3-6: Future projections
            if future_enabled:
                # Phase 3: Convert params
                calibrated_yaml = phase3_convert_params(nml, logger=logger)
                phase_results['Phase 3: Params YAML'] = calibrated_yaml is not None
                if calibrated_yaml is None:
                    logger.error("Phase 3 failed. Stopping.")
                    _print_summary(phase_results, time.time() - workflow_start, logger)
                    sys.exit(1)

                # Phase 4: Generate future namelist
                future_namelist = phase4_generate_future_namelist(
                    namelist_path, nml, calibrated_yaml, future_proj, logger=logger,
                )
                phase_results['Phase 4: Future Namelist'] = future_namelist is not None
                if future_namelist is None:
                    logger.error("Phase 4 failed. Stopping.")
                    _print_summary(phase_results, time.time() - workflow_start, logger)
                    sys.exit(1)

                # Phase 5: Future runs
                models = future_proj.get('models', FUTURE_DEFAULTS['models'])
                ok = phase5_future_runs(
                    future_namelist, models=models,
                    skip_download=args.skip_download,
                    force=args.force, bbox=args.bbox,
                    verbose=args.verbose, logger=logger,
                )
                phase_results['Phase 5: Future Runs'] = ok

                # Phase 6: Ensemble plot (even if some models failed)
                ok6 = phase6_ensemble_diagnostics(nml, future_proj, logger=logger)
                phase_results['Phase 6: Ensemble Plot'] = ok6

                # Phase 7: Full future postprocessing
                try:
                    from postprocessing_future import run_future_postprocessing
                    logger.info("=" * 70)
                    logger.info("PHASE 7: FUTURE ENSEMBLE POSTPROCESSING")
                    logger.info("=" * 70)
                    # Use the future namelist for postprocessing
                    run_future_postprocessing(str(future_namelist), skip_errors=True)
                    phase_results['Phase 7: Future Postprocessing'] = True
                except Exception as e:
                    logger.error(f"Phase 7 failed: {e}")
                    phase_results['Phase 7: Future Postprocessing'] = False

        # Summary
        total_duration = time.time() - workflow_start
        _print_summary(phase_results, total_duration, logger)

        if all(phase_results.values()):
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        _print_summary(phase_results, time.time() - workflow_start, logger)
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        _print_summary(phase_results, time.time() - workflow_start, logger)
        sys.exit(1)


def _print_summary(phase_results, total_duration, logger):
    """Print pipeline summary."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 70)
    for phase, ok in phase_results.items():
        status = "SUCCESS" if ok else "FAILED"
        logger.info(f"  {phase:35s}  {status}")
    logger.info(f"  {'Total time':35s}  {total_duration/60:.1f} min")

    if all(phase_results.values()):
        logger.info("\n  Pipeline completed successfully!")
    else:
        failed = [p for p, ok in phase_results.items() if not ok]
        logger.error(f"\n  Pipeline failed at: {', '.join(failed)}")


if __name__ == "__main__":
    main()
