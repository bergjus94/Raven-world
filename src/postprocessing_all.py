#!/usr/bin/env python3
"""
Master postprocessing orchestrator — runs all levels for a catchment.

Usage:
    python src/postprocessing_all.py namelists/catchment_0118.yaml
    python src/postprocessing_all.py namelists/catchment_0118.yaml --skip-single
    python src/postprocessing_all.py namelists/catchment_0118.yaml --only-metrics

Levels (in order):
    1. Single-run postprocessing (per config × model × metric)
    2. Metric comparison (per config × model)
    3. Model comparison (per config) — only if multiple models
    4. Configuration comparison (per model)
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from config_merge import load_config
from postprocessing_metrics import (
    discover_available_metrics, run_complete_metric_postprocessing
)
import postprocessing


def run_all_postprocessing(namelist_path, skip_single=False, skip_metrics=False,
                            skip_models=False, skip_configs=False,
                            only_metrics=False, env=None, skip_errors=True):
    """
    Run complete postprocessing pipeline for a catchment.
    """
    start_time = time.time()

    with open(namelist_path) as f:
        nml = yaml.safe_load(f)

    gauge_id = str(nml['catchment'])
    configurations = nml.get('configurations', ['baseline'])
    models = nml.get('models', ['HBV'])
    metrics_raw = nml.get('calibration', {}).get('metrics', ['KGE'])
    if isinstance(metrics_raw, str):
        metrics_list = [metrics_raw]
    elif isinstance(metrics_raw, list):
        metrics_list = metrics_raw
    else:
        metrics_list = ['KGE']

    print("=" * 80)
    print(f"COMPLETE POSTPROCESSING: catchment {gauge_id}")
    print(f"  Configurations: {configurations}")
    print(f"  Models: {models}")
    print(f"  Metrics: {metrics_list}")
    print("=" * 80)

    if only_metrics:
        skip_single = True
        skip_models = True
        skip_configs = True

    # ========================================================================
    # Level 1: Single-run postprocessing
    # ========================================================================
    if not skip_single:
        print(f"\n{'#'*80}")
        print(f"# LEVEL 1: Single-run postprocessing")
        print(f"{'#'*80}")

        for config in configurations:
            for model in models:
                for metric in metrics_list:
                    label = f"{config}/{model}/{metric}"
                    try:
                        overrides = {'_calibration_metric': metric}
                        loaded_nml, tmp = load_config(
                            gauge_id, config, model, env=env, overrides=overrides
                        )
                        Path(tmp).unlink(missing_ok=True)

                        pp_config = {
                            'main_dir': loaded_nml['main_dir'],
                            'gauge_id': gauge_id,
                            'model_type': model,
                            'start_date': loaded_nml.get('start_date'),
                            'end_date': loaded_nml.get('end_date'),
                            'cali_start_date': loaded_nml.get('cali_start_date', loaded_nml.get('start_date')),
                            'cali_end_date': loaded_nml.get('cali_end_date'),
                            'coupled': loaded_nml.get('coupled', False),
                            'raven_executable': loaded_nml.get('raven_executable'),
                            'glogem_dir': loaded_nml.get('glogem_dir'),
                            '_config_key': config,
                            '_calibration_metric': metric,
                        }

                        print(f"\n  [{label}]")
                        postprocessing.run_complete_postprocessing(
                            config=pp_config,
                            validation_start=loaded_nml.get('cali_end_date'),
                            validation_end=loaded_nml.get('end_date'),
                            skip_errors=skip_errors,
                        )
                    except Exception as e:
                        print(f"  [{label}] ERROR: {e}")
                        if not skip_errors:
                            raise

    # ========================================================================
    # Level 2: Metric comparison
    # ========================================================================
    if not skip_metrics:
        print(f"\n{'#'*80}")
        print(f"# LEVEL 2: Metric comparison")
        print(f"{'#'*80}")

        for config in configurations:
            for model in models:
                try:
                    run_complete_metric_postprocessing(
                        gauge_id=gauge_id,
                        config_key=config,
                        model_type=model,
                        env=env,
                        skip_errors=skip_errors,
                    )
                except Exception as e:
                    print(f"  [{config}/{model}] Metric comparison ERROR: {e}")
                    if not skip_errors:
                        raise

    # ========================================================================
    # Level 3: Model comparison (only if multiple models)
    # ========================================================================
    if not skip_models and len(models) > 1:
        print(f"\n{'#'*80}")
        print(f"# LEVEL 3: Model comparison")
        print(f"{'#'*80}")

        try:
            from postprocessing_models import run_complete_model_postprocessing
            for config in configurations:
                print(f"\n  [{config}] Comparing models: {models}")
                # TODO: integrate with postprocessing_models when metric-aware
                print(f"  [Not yet implemented with metric awareness]")
        except ImportError:
            print("  postprocessing_models not available")

    # ========================================================================
    # Level 4: Configuration comparison (only if multiple configs)
    # ========================================================================
    if not skip_configs and len(configurations) > 1:
        print(f"\n{'#'*80}")
        print(f"# LEVEL 4: Configuration comparison")
        print(f"{'#'*80}")

        try:
            from postprocessing_configurations import run_complete_multi_postprocessing
            for model in models:
                print(f"\n  [{model}] Comparing configurations: {configurations}")
                # TODO: integrate with postprocessing_configurations when metric-aware
                print(f"  [Not yet implemented with metric awareness]")
        except ImportError:
            print("  postprocessing_configurations not available")

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*80}")
    print(f"COMPLETE POSTPROCESSING DONE — {elapsed:.1f} min")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all postprocessing levels for a catchment")
    parser.add_argument('namelist', help='Path to catchment namelist YAML')
    parser.add_argument('--skip-single', action='store_true',
                        help='Skip Level 1 (single-run postprocessing)')
    parser.add_argument('--skip-metrics', action='store_true',
                        help='Skip Level 2 (metric comparison)')
    parser.add_argument('--skip-models', action='store_true',
                        help='Skip Level 3 (model comparison)')
    parser.add_argument('--skip-configs', action='store_true',
                        help='Skip Level 4 (configuration comparison)')
    parser.add_argument('--only-metrics', action='store_true',
                        help='Only run Level 2 (metric comparison)')
    parser.add_argument('--env', default=None, help='Environment (local/server)')
    args = parser.parse_args()

    run_all_postprocessing(
        args.namelist,
        skip_single=args.skip_single,
        skip_metrics=args.skip_metrics,
        skip_models=args.skip_models,
        skip_configs=args.skip_configs,
        only_metrics=args.only_metrics,
        env=args.env,
    )
