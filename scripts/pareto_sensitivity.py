#!/usr/bin/env python3
"""
Paper 5 — PAWN sensitivity analysis on NSGAII Pareto samples.

Computes behavioural sensitivity indices for each calibrated parameter
against each Pareto objective, using the 10000-sample NSGAII output.

Implements PAWN (Pianosi & Wagener 2015, EMS):
  For each parameter Xi:
    1. Bin samples by Xi value (M equal-sample bins)
    2. For each bin, compute Kolmogorov-Smirnov distance between the
       conditional CDF of objective Y within the bin and the unconditional
       CDF of Y across all samples.
    3. PAWN index = stat(KS) across bins, where stat ∈ {median, max, mean}.

Designed for the Paper-5 NSGAII Pareto outputs (calibration_results_*.csv).
Auto-detects calibrated parameter columns (any column not in {obj_Q,
obj_snow, obj_baseflow, timestamp}).

Also computes complementary diagnostics:
  - Spearman rank correlation Xi vs each objective (sign + magnitude)
  - Parameter range fraction on Pareto front (max - min) / (bounds upper - bounds lower)
    → high = highly equifinal / poorly constrained
    → low  = well-constrained by the data

Usage:
  scripts/pareto_sensitivity.py --catchment 2268
  scripts/pareto_sensitivity.py --catchment 2268 --structure glogem_subdaily_opt1 --plot
  scripts/pareto_sensitivity.py --catchment 2268 --root /home/jberg/OneDrive/Raven_worldwide/model_runs
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr


OBJECTIVE_COLS = ['obj_Q', 'obj_snow', 'obj_baseflow']
NON_PARAM_COLS = {*OBJECTIVE_COLS, 'timestamp'}


def pawn_index(x: np.ndarray, y: np.ndarray, n_bins: int = 10,
               stat: str = 'median') -> float:
    """PAWN sensitivity index of y on x.

    Pianosi & Wagener 2015: split samples into ``n_bins`` equal-sample
    bins along x, compute KS distance between the conditional CDF of y
    within each bin and the unconditional CDF of y, return the chosen
    statistic across bins.

    Returns a value in [0, 1]; higher = more sensitive.
    """
    if len(x) != len(y) or len(x) < n_bins * 2:
        return float('nan')
    order = np.argsort(x)
    y_sorted = y[order]
    bin_size = len(y_sorted) // n_bins

    ks_distances = []
    for i in range(n_bins):
        lo = i * bin_size
        hi = (i + 1) * bin_size if i < n_bins - 1 else len(y_sorted)
        y_cond = y_sorted[lo:hi]
        if len(y_cond) < 2:
            continue
        ks, _ = ks_2samp(y_cond, y)
        ks_distances.append(ks)

    if not ks_distances:
        return float('nan')
    arr = np.array(ks_distances)
    return {
        'median': float(np.median(arr)),
        'max':    float(np.max(arr)),
        'mean':   float(np.mean(arr)),
    }[stat]


def analyze_csv(csv_path: Path, n_bins: int = 10) -> pd.DataFrame:
    """Compute PAWN, Spearman r, and range-fraction for every (param, objective).

    Returns a long-format DataFrame with columns:
      parameter, objective, pawn_median, pawn_max, spearman_r, range_frac
    """
    df = pd.read_csv(csv_path)
    param_cols = [c for c in df.columns if c not in NON_PARAM_COLS]

    rows = []
    for p in param_cols:
        x = df[p].to_numpy(dtype=float)
        # Pareto-range fraction: spread of param values on the front
        # normalised to the spread observed across all samples (proxy for
        # how much of the prior the calibration kept open).
        x_span = x.max() - x.min()
        x_range_frac = float('nan')  # filled below with bounds info if available

        for obj in OBJECTIVE_COLS:
            if obj not in df.columns:
                continue
            y = df[obj].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < n_bins * 2:
                continue
            xm, ym = x[mask], y[mask]
            pawn_med = pawn_index(xm, ym, n_bins=n_bins, stat='median')
            pawn_max = pawn_index(xm, ym, n_bins=n_bins, stat='max')
            rho, _ = spearmanr(xm, ym)
            rows.append({
                'parameter':   p,
                'objective':   obj,
                'pawn_median': pawn_med,
                'pawn_max':    pawn_max,
                'spearman_r':  float(rho),
                'x_span':      float(x_span),
            })

    return pd.DataFrame(rows)


def plot_pawn_heatmap(df_long: pd.DataFrame, title: str, out_path: Path,
                      value_col: str = 'pawn_median') -> None:
    """Heatmap: rows = parameters, cols = objectives, cells = PAWN index."""
    piv = df_long.pivot(index='parameter', columns='objective', values=value_col)
    # Order objectives consistently
    piv = piv.reindex(columns=[c for c in OBJECTIVE_COLS if c in piv.columns])
    # Order parameters by mean PAWN across objectives (most sensitive at top)
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(6, max(4, 0.4 * len(piv) + 1)))
    im = ax.imshow(piv.values, aspect='auto', cmap='viridis', vmin=0, vmax=0.6)

    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([c.replace('obj_', '') for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=9)

    # Cell annotations
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        color='white' if v < 0.3 else 'black', fontsize=8)

    cb = plt.colorbar(im, ax=ax, label=f'PAWN ({value_col.replace("pawn_", "")})')
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_cross_structure_overlay(combined: pd.DataFrame, catchment: str,
                                  out_path: Path, value_col: str = 'pawn_median') -> None:
    """Per-objective heatmap: rows = parameters, cols = structures.

    One figure with three subplots (one per objective). Each cell shows
    the PAWN value for that (parameter, structure) combination. Missing
    cells (parameter not in that structure) are grey. Parameters are
    ordered by their mean PAWN across all structures (most sensitive at
    top); structures appear in S1-S9 order.

    Useful for spotting structural patterns at a glance — e.g. does
    GlacROF only light up in glacier-GW structures? Does K2 dominate
    baseflow more in fast-routing variants?
    """
    # Map config keys to S1-S9 labels for compact display.
    structure_order = [
        ('glogem_subdaily_opt1', 'S1'),
        ('glogem_subdaily_opt1_glaciergw', 'S2'),
        ('glogem_subdaily_opt1_threshold', 'S3'),
        ('glogem_subdaily_opt1_threshold_glaciergw', 'S4'),
        ('glogem_subdaily_opt2_sphy_faithful', 'S5'),
        ('glogem_subdaily_opt2_sphy_faithful_glaciergw', 'S6'),
        ('glogem_subdaily_opt1_glaciergw_fast', 'S7'),
        ('glogem_subdaily_opt1_threshold_glaciergw_fast', 'S8'),
        ('glogem_subdaily_opt2_sphy_faithful_glaciergw_fast', 'S9'),
    ]
    config_to_label = dict(structure_order)
    col_order = [s for c, s in structure_order if c in combined['structure'].unique()]
    combined = combined.copy()
    combined['s_label'] = combined['structure'].map(config_to_label)

    fig, axes = plt.subplots(1, 3, figsize=(15, max(6, 0.4 * combined['parameter'].nunique() + 1)),
                              sharey=True)
    for ax, obj in zip(axes, OBJECTIVE_COLS):
        sub = combined[combined['objective'] == obj]
        if sub.empty:
            ax.axis('off')
            continue
        piv = sub.pivot(index='parameter', columns='s_label', values=value_col)
        piv = piv.reindex(columns=col_order)
        # Order parameters by mean PAWN across structures (max sensitivity at top)
        param_order = piv.mean(axis=1, skipna=True).sort_values(ascending=False).index
        piv = piv.loc[param_order]

        # Masked array so NaN cells render in grey via cmap.set_bad
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='#d9d9d9')
        ma = np.ma.array(piv.values, mask=~np.isfinite(piv.values))
        im = ax.imshow(ma, aspect='auto', cmap=cmap, vmin=0, vmax=0.6)

        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(piv.columns, fontsize=9)
        if ax is axes[0]:
            ax.set_yticks(range(piv.shape[0]))
            ax.set_yticklabels(piv.index, fontsize=8)
        ax.set_title(obj.replace('obj_', ''), fontsize=11)

        # Cell annotations
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.2f}',
                            ha='center', va='center',
                            color='white' if v < 0.3 else 'black', fontsize=7)

    fig.suptitle(f'PAWN sensitivity across 9 structures — catchment {catchment}',
                  fontsize=12, y=1.02)
    cb = fig.colorbar(im, ax=axes, label='PAWN (median KS)', shrink=0.7)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_spearman_heatmap(df_long: pd.DataFrame, title: str, out_path: Path) -> None:
    """Heatmap of signed Spearman correlations (red=negative, blue=positive)."""
    piv = df_long.pivot(index='parameter', columns='objective', values='spearman_r')
    piv = piv.reindex(columns=[c for c in OBJECTIVE_COLS if c in piv.columns])
    piv = piv.loc[piv.abs().mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(6, max(4, 0.4 * len(piv) + 1)))
    im = ax.imshow(piv.values, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([c.replace('obj_', '') for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=9)

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                        color='black' if abs(v) < 0.5 else 'white', fontsize=8)

    plt.colorbar(im, ax=ax, label='Spearman r')
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def find_csv(root: Path, catchment: str, structure: Optional[str] = None) -> list[Path]:
    """Locate calibration_results CSVs for a catchment.

    Layout: {root}/catchment_{ID}/configs/<structure>/SPHY_paper5_pareto/output/calibration_results_*.csv
    """
    cdir = root / f'catchment_{catchment}' / 'configs'
    if not cdir.exists():
        raise FileNotFoundError(f"No configs dir for catchment {catchment} under {root}")
    csvs = []
    structures = [structure] if structure else sorted(p.name for p in cdir.iterdir() if p.is_dir())
    for s in structures:
        cands = list((cdir / s / 'SPHY_paper5_pareto' / 'output').glob('calibration_results_*.csv'))
        if cands:
            # Use the most recent CSV (largest mtime)
            csvs.append(max(cands, key=lambda p: p.stat().st_mtime))
    return csvs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--catchment', required=True,
                        help='Catchment ID (e.g. 2268)')
    parser.add_argument('--structure', default=None,
                        help='Structure config dir (default: all structures found)')
    parser.add_argument('--root', type=Path,
                        default=Path.home() / 'Raven_world' / 'model_runs',
                        help='model_runs root. Defaults to ~/Raven_world/model_runs.')
    parser.add_argument('--outdir', type=Path,
                        default=Path('/tmp/pareto_sensitivity'),
                        help='Output directory for CSVs + plots')
    parser.add_argument('--n-bins', type=int, default=10,
                        help='PAWN conditional-bin count (Pianosi recommends 10)')
    parser.add_argument('--plot', action='store_true',
                        help='Save PAWN + Spearman heatmaps (one per structure)')
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    csvs = find_csv(args.root, args.catchment, args.structure)
    if not csvs:
        print(f"No CSVs found for catchment {args.catchment} "
              f"(structure={args.structure}) under {args.root}", file=sys.stderr)
        return 1

    all_results = []
    for csv in csvs:
        struct = csv.parent.parent.parent.name  # configs/<structure>/SPHY_paper5_pareto/output → <structure>
        print(f'  Analyzing {struct} ({csv.name}, '
              f'{sum(1 for _ in csv.open()) - 1} samples)...')
        df_long = analyze_csv(csv, n_bins=args.n_bins)
        df_long.insert(0, 'structure', struct)
        df_long.insert(0, 'catchment', args.catchment)
        all_results.append(df_long)

        if args.plot:
            base = f'{args.catchment}_{struct}'
            plot_pawn_heatmap(df_long,
                              title=f'PAWN sensitivity — catchment {args.catchment} / {struct}',
                              out_path=args.outdir / f'{base}_pawn.png')
            plot_spearman_heatmap(df_long,
                                  title=f'Spearman r — catchment {args.catchment} / {struct}',
                                  out_path=args.outdir / f'{base}_spearman.png')

    combined = pd.concat(all_results, ignore_index=True)
    out_csv = args.outdir / f'{args.catchment}_sensitivity_summary.csv'
    combined.to_csv(out_csv, index=False)
    print(f'\nSaved: {out_csv}')
    if args.plot:
        # Cross-structure overlay — only meaningful if we have >1 structure
        if combined['structure'].nunique() > 1:
            overlay_path = args.outdir / f'{args.catchment}_pawn_cross_structure.png'
            plot_cross_structure_overlay(combined, args.catchment, overlay_path)
            print(f'Cross-structure overlay: {overlay_path}')
        print(f'Plots in: {args.outdir}')

    # Quick top-3 most sensitive params summary per (structure, objective)
    print('\n═══ Top 3 most-sensitive parameters per structure × objective ═══')
    for struct in combined['structure'].unique():
        sub = combined[combined['structure'] == struct]
        print(f'\n--- {struct} ---')
        for obj in OBJECTIVE_COLS:
            ss = sub[sub['objective'] == obj].sort_values('pawn_median', ascending=False)
            if ss.empty:
                continue
            top = ss.head(3)
            entries = [f'{r.parameter}({r.pawn_median:.2f}, r={r.spearman_r:+.2f})'
                       for _, r in top.iterrows()]
            print(f'  {obj}: {", ".join(entries)}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
