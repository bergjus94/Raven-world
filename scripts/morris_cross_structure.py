#!/usr/bin/env python3
"""
Paper 5 — combine per-structure Morris-SA outputs into a cross-structure view.

After running scripts/morris_sa.py for each structure separately, this script:
  1. Reads each morris_indices.csv from plots/morris_sa/<catchment>_<structure>/
  2. Tags each row with the structure's two-letter code (LN/LS/LF/.../OF)
  3. Builds two diagnostic figures:

     (a) Cross-structure μ* heatmap (rows = parameters, cols = structures,
         one panel per objective). Reveals which parameters dominate
         sensitivity in which structures. Direct paper-figure candidate.

     (b) Cross-structure μ* vs σ scatter (one panel per objective).
         Markers colored by structure. Identifies which structures have
         the most non-linear / interaction-rich parameters.

Output: plots/morris_sa/<catchment>_cross_structure_<obj>.png

Usage:
  scripts/morris_cross_structure.py --catchment 2268
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper5_common import (
    STRUCTURE_INFO, STRUCTURE_ORDER, CONFIG_TO_LABEL,
)


OBJECTIVE_COLS = ['Q', 'snow', 'baseflow']


def collect_morris_results(catchment: str, morris_dir: Path) -> pd.DataFrame:
    """Load all per-structure morris_indices.csv files and concatenate.

    Returns a DataFrame with columns:
      catchment, structure, two_letter, objective, parameter,
      mu_star, sigma, mu_star_conf
    """
    rows = []
    for s_code in STRUCTURE_ORDER:
        s = STRUCTURE_INFO[s_code]
        cfg = s['config_key']
        two_letter = s['two_letter']
        # Per the morris_sa.py default, results land in
        # plots/morris_sa/<catchment>_<configkey>/morris_indices.csv
        csv_path = morris_dir / f'{catchment}_{cfg}' / 'morris_indices.csv'
        if not csv_path.exists():
            print(f'  ⚠️ missing: {csv_path.relative_to(morris_dir.parent)} '
                  f'({two_letter}) — skipping')
            continue
        df = pd.read_csv(csv_path)
        df['two_letter'] = two_letter
        rows.append(df)
        print(f'  ✓ loaded {two_letter} ({len(df)} rows)')

    if not rows:
        raise RuntimeError(f'No Morris results found under {morris_dir}')
    return pd.concat(rows, ignore_index=True)


def plot_mu_star_heatmap(combined: pd.DataFrame, catchment: str,
                          outdir: Path) -> None:
    """Per-objective μ* heatmap (parameters × structures). One PNG per objective."""
    # Force structure-column order to S1→S9 / LN→OF
    col_order = [STRUCTURE_INFO[s]['two_letter'] for s in STRUCTURE_ORDER]
    col_order = [c for c in col_order if c in combined['two_letter'].unique()]

    fig, axes = plt.subplots(1, 3, figsize=(16, max(6, 0.4 * combined['parameter'].nunique() + 1)),
                              sharey=True)
    for ax, obj in zip(axes, OBJECTIVE_COLS):
        sub = combined[combined['objective'] == obj]
        if sub.empty:
            ax.axis('off')
            continue
        piv = sub.pivot_table(index='parameter', columns='two_letter',
                               values='mu_star', aggfunc='mean')
        piv = piv.reindex(columns=col_order)
        # Order parameters by mean μ* across structures (most sensitive at top)
        param_order = piv.mean(axis=1, skipna=True).sort_values(ascending=False).index
        piv = piv.loc[param_order]

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='#d9d9d9')
        ma = np.ma.array(piv.values, mask=~np.isfinite(piv.values))
        # Use shared scale per panel (auto-vmax based on data)
        vmax = float(np.nanmax(piv.values)) if np.any(np.isfinite(piv.values)) else 1.0
        im = ax.imshow(ma, aspect='auto', cmap=cmap, vmin=0, vmax=vmax)

        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(piv.columns, fontsize=9)
        if ax is axes[0]:
            ax.set_yticks(range(piv.shape[0]))
            ax.set_yticklabels([p.replace('Sphy_', '') for p in piv.index], fontsize=8)
        ax.set_title(obj, fontsize=11)

        # Annotate cells
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            color='white' if v < vmax * 0.5 else 'black',
                            fontsize=7)
        plt.colorbar(im, ax=ax, label='μ*', shrink=0.7)

    fig.suptitle(f'Morris μ* across 9 structures — catchment {catchment}',
                  fontsize=12, y=1.02)
    fig.tight_layout()
    fp = outdir / f'{catchment}_morris_cross_structure_mu_star.png'
    fig.savefig(fp, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fp}')


def plot_mu_vs_sigma(combined: pd.DataFrame, catchment: str,
                      outdir: Path) -> None:
    """Per-objective μ* vs σ scatter, colored by structure (all 9 overlaid)."""
    col_order = [STRUCTURE_INFO[s]['two_letter'] for s in STRUCTURE_ORDER]
    col_order = [c for c in col_order if c in combined['two_letter'].unique()]
    palette = plt.cm.tab10(np.linspace(0, 1, max(9, len(col_order))))
    s_color = {c: palette[i] for i, c in enumerate(col_order)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    for ax, obj in zip(axes, OBJECTIVE_COLS):
        sub = combined[combined['objective'] == obj]
        if sub.empty:
            ax.axis('off')
            continue
        for s_lab, grp in sub.groupby('two_letter'):
            ax.scatter(grp['mu_star'], grp['sigma'],
                       s=60, alpha=0.75, color=s_color.get(s_lab, 'grey'),
                       label=s_lab, edgecolor='k', linewidth=0.4)
        # σ = μ* reference line (above = strong interaction/non-linearity)
        lim = max(float(np.nanmax(sub['mu_star'])),
                  float(np.nanmax(sub['sigma']))) * 1.1
        ax.plot([0, lim], [0, lim], color='grey', ls=':', lw=0.7)
        ax.set_xlabel(r'$\mu^*$  →  importance')
        if ax is axes[0]:
            ax.set_ylabel(r'$\sigma$  →  non-linearity / interaction')
        ax.set_title(obj)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        order = sorted(range(len(labels)), key=lambda i: col_order.index(labels[i]))
        axes[-1].legend([handles[i] for i in order], [labels[i] for i in order],
                         loc='upper right', fontsize=8, title='Structure',
                         frameon=True)

    fig.suptitle(f'Morris μ* vs σ — catchment {catchment} (all 9 structures overlaid)',
                  fontsize=12, y=1.02)
    fig.tight_layout()
    fp = outdir / f'{catchment}_morris_cross_structure_mu_sigma.png'
    fig.savefig(fp, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fp}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--catchment', required=True,
                        help='Catchment ID (e.g. 2268)')
    parser.add_argument('--morris-dir', type=Path, default=None,
                        help='Root dir of per-structure Morris outputs. '
                             'Default: <repo>/plots/morris_sa/')
    parser.add_argument('--outdir', type=Path, default=None,
                        help='Output directory for cross-structure plots. '
                             'Default: <repo>/plots/morris_sa/')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if args.morris_dir is None:
        args.morris_dir = repo_root / 'plots' / 'morris_sa'
    if args.outdir is None:
        args.outdir = repo_root / 'plots' / 'morris_sa'
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f'Loading per-structure Morris results from {args.morris_dir}')
    combined = collect_morris_results(args.catchment, args.morris_dir)

    summary_csv = args.outdir / f'{args.catchment}_morris_cross_structure.csv'
    combined.to_csv(summary_csv, index=False)
    print(f'\nSaved combined CSV: {summary_csv}')

    plot_mu_star_heatmap(combined, args.catchment, args.outdir)
    plot_mu_vs_sigma(combined, args.catchment, args.outdir)

    # Top-3 per (structure, objective) — quick text summary
    print('\n═══ Top 3 by μ* per (structure × objective) ═══')
    for two_letter in [STRUCTURE_INFO[s]['two_letter'] for s in STRUCTURE_ORDER
                        if STRUCTURE_INFO[s]['two_letter'] in combined['two_letter'].unique()]:
        sub = combined[combined['two_letter'] == two_letter]
        print(f'\n--- {two_letter} ---')
        for obj in OBJECTIVE_COLS:
            ss = sub[sub['objective'] == obj].sort_values('mu_star', ascending=False)
            if ss.empty:
                continue
            top = ss.head(3)
            entries = [f"{r.parameter.replace('Sphy_', '')}({r.mu_star:.2f})"
                       for _, r in top.iterrows()]
            print(f'  {obj:<10s} {", ".join(entries)}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
