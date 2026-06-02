#!/usr/bin/env python3
"""
Paper 5 — Morris elementary-effects sensitivity analysis.

Morris (1991, refined by Campolongo et al. 2007) is a global SA method
that samples *trajectories* in parameter space — each trajectory is a
chain of k+1 points where each successive point differs from the
previous by one parameter changing by Δ. The elementary effect of
parameter Xi at that trajectory is a finite-difference partial
derivative ∂Y/∂Xi.

After r trajectories you have r elementary effects per parameter. Two
statistics summarise them:
  - μ*  = mean of |EE|   → overall importance (Campolongo 2007 absolute-mean)
  - σ   = std-dev of EE  → non-linearity / interaction signal

Runs = r × (k+1) — typically 280-700 for r=20-50 and k=12-14, which is
~100× cheaper than Sobol while still globally exploring the prior.

This script is self-contained: it reuses spotpy_optimize.RavenSCEUA
for the model-evaluation infrastructure (parameter application + Raven
subprocess + objective extraction), but does its own Morris sampling
and analysis via SALib.

Usage:
  scripts/morris_sa.py --namelist namelists/catchment_2268_rhone.yaml \\
      --structure glogem_subdaily_opt1 \\
      --n-trajectories 20
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# SALib for Morris sampling + analysis
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze

# Reuse the production calibration setup for parameter application + Raven runs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spotpy_optimize import RavenSCEUA  # noqa: E402


OBJECTIVE_NAMES = ['Q', 'snow', 'baseflow']


def build_optimizer(namelist_path: Path, structure: str) -> RavenSCEUA:
    """Instantiate RavenSCEUA for a single namelist + chosen structure.

    Mirrors the merge order used by run_full_pipeline._run_multi_config:
    pulls the paper-5 namelist's overrides, then calls config_merge.load_config
    with the chosen configuration layer. This gives the same merged namelist
    that the NSGAII calibration would have used for this (catchment, structure),
    minus the iteration-loop logic.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
    from config_merge import load_config

    with namelist_path.open() as f:
        catchment_nml = yaml.safe_load(f)
    catchment_id = str(catchment_nml['catchment'])
    model = catchment_nml.get('models', ['SPHY'])[0]

    # Pull the same field set that run_full_pipeline._run_multi_config does
    overrides = {}
    for key in ('start_date', 'end_date', 'cali_end_date', 'warm_up_date',
                'display_name', 'gauge_id', 'warmup', 'precip_correction',
                'meteo_source', 'precip_source', 'lapse_rate_precip_source',
                'region'):
        if key in catchment_nml:
            overrides[key] = catchment_nml[key]
    if 'calibration' in catchment_nml:
        overrides['calibration'] = catchment_nml['calibration']

    nml, tmp_path = load_config(
        catchment=catchment_id,
        configuration=structure,
        model=model,
        overrides=overrides,
    )

    return RavenSCEUA(
        gauge_id=nml['gauge_id'],
        model_type=nml['model_type'],
        cali_end_date=nml['cali_end_date'],
        vali_end_date=nml['end_date'],
        obj_function='KGE_NP',
        main_dir=nml['main_dir'],
        config_dir=structure,
        coupled=nml.get('coupled', True),
        params_dir=nml.get('params_dir',
                            Path(__file__).resolve().parent.parent /
                            'src/config/default_params.yaml'),
        raven_exe=nml.get('raven_exe'),
        namelist=nml,
    )


def build_salib_problem(opt: RavenSCEUA) -> dict:
    """Construct SALib problem dict from the active parameter set.

    spotpy.parameters() returns a structured numpy array with fields
    'name', 'minbound', 'maxbound' (and others). Access via item lookup,
    not attribute, since elements are numpy.void records.
    """
    params = opt.parameters()
    return {
        'num_vars': len(params),
        'names':    [str(p['name']) for p in params],
        'bounds':   [[float(p['minbound']), float(p['maxbound'])] for p in params],
    }


def run_morris_evaluation(opt: RavenSCEUA, problem: dict,
                          n_trajectories: int, num_levels: int = 4,
                          out_dir: Path = Path('.')) -> pd.DataFrame:
    """Generate Morris design, evaluate each sample through Raven, return CSV.

    Saves an intermediate ``morris_samples.csv`` (parameters) and
    ``morris_results.csv`` (parameters + 3 objectives) under ``out_dir``.
    """
    samples = morris_sample.sample(problem, N=n_trajectories,
                                    num_levels=num_levels)
    n_samples = samples.shape[0]
    print(f"Generated {n_samples} samples "
          f"(r={n_trajectories} trajectories × ({problem['num_vars']}+1) points).")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(samples, columns=problem['names']).to_csv(
        out_dir / 'morris_samples.csv', index=False)

    results = np.full((n_samples, len(OBJECTIVE_NAMES)), np.nan)
    t0 = time.time()
    for i, vec in enumerate(samples):
        try:
            sim = opt.simulation(list(vec))
            # RavenSCEUA.simulation returns one objective per active obj
            for j, _ in enumerate(OBJECTIVE_NAMES[:len(sim)]):
                results[i, j] = float(sim[j])
        except Exception as e:
            print(f"  [{i+1}/{n_samples}] FAILED: {e}")
            continue
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate
            print(f"  [{i+1}/{n_samples}] eval ok  ({rate:.2f} samples/s, "
                  f"ETA {eta/60:.1f} min)")

    df = pd.DataFrame(samples, columns=problem['names'])
    for j, obj in enumerate(OBJECTIVE_NAMES[:results.shape[1]]):
        df[f'obj_{obj}'] = results[:, j]
    df.to_csv(out_dir / 'morris_results.csv', index=False)
    return df


def analyze_and_plot(problem: dict, results_df: pd.DataFrame,
                     n_trajectories: int, num_levels: int,
                     out_dir: Path, structure: str, catchment: str) -> None:
    """Compute Morris indices per objective and produce diagnostic plots."""
    param_names = problem['names']
    X = results_df[param_names].to_numpy()

    summary_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, obj in zip(axes, OBJECTIVE_NAMES):
        col = f'obj_{obj}'
        if col not in results_df.columns:
            ax.axis('off')
            continue
        Y = results_df[col].to_numpy()
        mask = np.isfinite(Y)
        if mask.sum() < len(X):
            print(f'  {obj}: dropping {(~mask).sum()} failed samples')
        Si = morris_analyze.analyze(problem, X[mask], Y[mask],
                                     num_levels=num_levels,
                                     print_to_console=False)

        # Plot μ* vs σ
        mu_star = Si['mu_star']
        sigma = Si['sigma']
        ax.scatter(mu_star, sigma, s=70, color='#2c7fb8',
                    edgecolor='k', linewidth=0.6, alpha=0.85)
        for n, mu, sg in zip(param_names, mu_star, sigma):
            ax.annotate(n.replace('Sphy_', ''), xy=(mu, sg),
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=7, alpha=0.8)
        ax.set_xlabel(r'$\mu^*$ (mean |EE|)  →  importance')
        if ax is axes[0]:
            ax.set_ylabel(r'$\sigma$ (std EE)  →  non-linearity / interaction')
        ax.set_title(obj)
        # Reference line σ = μ* (above = strong interaction/nonlinear)
        lim = max(np.nanmax(mu_star), np.nanmax(sigma)) * 1.1 if mu_star.size else 1
        ax.plot([0, lim], [0, lim], color='grey', ls=':', lw=0.6)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.grid(alpha=0.3)

        for n, mu, sg, ci in zip(param_names, mu_star, sigma, Si['mu_star_conf']):
            summary_rows.append({
                'catchment':    catchment,
                'structure':    structure,
                'objective':    obj,
                'parameter':    n,
                'mu_star':      float(mu),
                'sigma':        float(sg),
                'mu_star_conf': float(ci),
            })

    fig.suptitle(f'Morris SA — {catchment} / {structure}\n'
                  f'r={n_trajectories} trajectories, levels={num_levels}',
                  fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / 'morris_mu_vs_sigma.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(out_dir / 'morris_indices.csv', index=False)
    print(f'\nSaved indices: {out_dir / "morris_indices.csv"}')
    print(f'Saved plot:    {out_dir / "morris_mu_vs_sigma.png"}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--namelist', type=Path, required=True,
                        help='Paper-5 namelist YAML')
    parser.add_argument('--structure', type=str, required=True,
                        help='Configuration layer name '
                             '(e.g. glogem_subdaily_opt1 for S1)')
    parser.add_argument('--n-trajectories', '-r', type=int, default=20,
                        help='Number of Morris trajectories (default 20). '
                             'Total runs = r × (k+1) where k = num params.')
    parser.add_argument('--num-levels', '-l', type=int, default=4,
                        help='Number of grid levels per parameter axis (default 4)')
    parser.add_argument('--outdir', type=Path, default=None,
                        help='Output directory (default: <repo>/plots/morris_sa/<catchment>_<structure>/)')
    parser.add_argument('--analyze-only', type=Path, default=None,
                        help='Skip evaluation; read morris_results.csv from this dir '
                             'and re-analyze + re-plot only.')
    args = parser.parse_args()

    with args.namelist.open() as f:
        nml = yaml.safe_load(f)
    catchment = str(nml['catchment'])

    if args.outdir is None:
        repo_root = Path(__file__).resolve().parent.parent
        args.outdir = repo_root / 'plots' / 'morris_sa' / f'{catchment}_{args.structure}'
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {args.outdir}')

    if args.analyze_only:
        df = pd.read_csv(args.analyze_only / 'morris_results.csv')
        # Reconstruct problem from sample CSV
        param_names = [c for c in df.columns if not c.startswith('obj_')]
        opt = build_optimizer(args.namelist, args.structure)
        problem = build_salib_problem(opt)
        # Sanity: param names should match
        assert problem['names'] == param_names, \
            f"Parameter mismatch: {problem['names']} vs {param_names}"
    else:
        opt = build_optimizer(args.namelist, args.structure)
        problem = build_salib_problem(opt)
        print(f'\nMorris design: {args.n_trajectories} trajectories × '
              f'({problem["num_vars"]}+1) = '
              f'{args.n_trajectories * (problem["num_vars"] + 1)} model runs')

        df = run_morris_evaluation(opt, problem,
                                    n_trajectories=args.n_trajectories,
                                    num_levels=args.num_levels,
                                    out_dir=args.outdir)

    analyze_and_plot(problem, df,
                      n_trajectories=args.n_trajectories,
                      num_levels=args.num_levels,
                      out_dir=args.outdir,
                      structure=args.structure,
                      catchment=catchment)
    return 0


if __name__ == '__main__':
    sys.exit(main())
