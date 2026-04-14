#!/usr/bin/env python3
"""
Finalize partial calibration runs that were interrupted before completion.

Finds calibration_results_*.csv files without a matching VERIFIED_best_params.csv,
extracts the best parameter set, runs the final Raven model, and generates
diagnostics and plots.

Usage:
    python scripts/finalize_calibration.py namelists/catchment_0118.yaml
    python scripts/finalize_calibration.py namelists/catchment_0118.yaml --min-iterations 1000
    python scripts/finalize_calibration.py namelists/catchment_0118.yaml --dry-run
"""

import sys
import argparse
from pathlib import Path

# Add src and project root to path
script_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(script_dir / 'src'))
sys.path.insert(0, str(script_dir))

import yaml
from config_merge import load_config
from paths import get_paths


def find_incomplete_runs(catchment_id, configurations, models, metrics, env=None):
    """Find calibration runs that have results but no VERIFIED file."""
    incomplete = []

    for config in configurations:
        for model in models:
            for metric in metrics:
                overrides = {'_calibration_metric': metric}
                if 'calibration' not in overrides:
                    overrides['calibration'] = {}
                overrides['calibration']['metrics'] = {'primary': metric}

                try:
                    nml, tmp_path = load_config(
                        catchment=catchment_id, configuration=config,
                        model=model, env=env, overrides=overrides,
                    )
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception as e:
                    print(f"  [{config}/{model}/{metric}] Config load failed: {e}")
                    continue

                paths = get_paths(nml)
                output_dir = paths['output_dir']
                gauge_id = str(nml['gauge_id'])
                model_type = nml['model_type']

                verified = output_dir / f"{gauge_id}_{model_type}_VERIFIED_best_params.csv"
                if verified.exists():
                    continue

                # Find calibration results files
                results_files = sorted(output_dir.glob(f"calibration_results_{gauge_id}_{model_type}_*.csv"))
                if not results_files:
                    continue

                # Use the most recent one
                results_file = results_files[-1]
                # Count iterations (subtract 1 for header)
                n_lines = sum(1 for _ in open(results_file)) - 1

                incomplete.append({
                    'config': config,
                    'model': model,
                    'metric': metric,
                    'nml': nml,
                    'results_file': results_file,
                    'n_iterations': n_lines,
                    'output_dir': output_dir,
                })

    return incomplete


def finalize_run(run_info, dry_run=False):
    """Run best parameters and generate diagnostics for a partial run."""
    nml = run_info['nml']
    results_file = run_info['results_file']
    label = f"{run_info['config']}/{run_info['model']}/{run_info['metric']}"

    print(f"\n{'='*60}")
    print(f"Finalizing: {label}")
    print(f"  Results file: {results_file}")
    print(f"  Iterations completed: {run_info['n_iterations']}")

    if dry_run:
        print(f"  [DRY RUN] Would finalize this run")
        return True

    try:
        from spotpy_optimize import RavenSCEUA

        gauge_id = str(nml['gauge_id'])
        model_type = nml['model_type']

        optimization = RavenSCEUA(
            gauge_id=gauge_id,
            model_type=model_type,
            cali_end_date=nml.get('cali_end_date'),
            vali_end_date=nml.get('end_date'),
            obj_function=nml.get('_calibration_metric', 'KGE'),
            main_dir=nml.get('main_dir'),
            config_dir=nml.get('_config_key'),
            coupled=nml.get('coupled', False),
            params_dir=nml.get('params_dir'),
            raven_exe=nml.get('raven_executable'),
            namelist=nml,
        )

        # Point to the existing results file
        optimization.results_file = results_file

        # Run final model with best parameters and generate diagnostics
        optimization.run_best_parameters()
        optimization.plot_results()

        print(f"  Finalized successfully")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Finalize partial calibration runs")
    parser.add_argument('namelist', help='Path to catchment namelist YAML')
    parser.add_argument('--min-iterations', type=int, default=500,
                        help='Minimum iterations to consider a run worth finalizing (default: 500)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be finalized without doing it')
    parser.add_argument('--env', type=str, default=None,
                        help='Environment (local/server)')
    args = parser.parse_args()

    # Load catchment namelist
    with open(args.namelist) as f:
        catchment_nml = yaml.safe_load(f)

    catchment_id = str(catchment_nml['catchment'])
    configurations = catchment_nml.get('configurations', ['baseline'])
    models = catchment_nml.get('models', ['HBV'])
    metrics_raw = catchment_nml.get('calibration', {}).get('metrics', ['KGE'])
    if isinstance(metrics_raw, str):
        metrics = [metrics_raw]
    elif isinstance(metrics_raw, list):
        metrics = metrics_raw
    else:
        metrics = ['KGE']

    print(f"Scanning catchment {catchment_id}")
    print(f"  Configurations: {configurations}")
    print(f"  Models: {models}")
    print(f"  Metrics: {metrics}")

    # Find incomplete runs
    incomplete = find_incomplete_runs(catchment_id, configurations, models, metrics, env=args.env)

    if not incomplete:
        print("\nNo incomplete runs found. All calibrations either completed or have no results.")
        return

    # Filter by minimum iterations
    worthy = [r for r in incomplete if r['n_iterations'] >= args.min_iterations]
    skipped = [r for r in incomplete if r['n_iterations'] < args.min_iterations]

    print(f"\nFound {len(incomplete)} incomplete runs:")
    for r in incomplete:
        status = "FINALIZE" if r['n_iterations'] >= args.min_iterations else f"SKIP (<{args.min_iterations} iter)"
        print(f"  {r['config']}/{r['model']}/{r['metric']}: {r['n_iterations']} iterations — {status}")

    if skipped:
        print(f"\nSkipping {len(skipped)} runs with <{args.min_iterations} iterations")

    if not worthy:
        print("\nNo runs meet the minimum iteration threshold.")
        return

    # Finalize
    print(f"\nFinalizing {len(worthy)} runs...")
    success = 0
    for run_info in worthy:
        if finalize_run(run_info, dry_run=args.dry_run):
            success += 1

    print(f"\n{'='*60}")
    print(f"Done: {success}/{len(worthy)} finalized successfully")


if __name__ == "__main__":
    main()
