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

# Pull the structure naming + ordering from the shared paper-5 module so all
# plotting scripts stay in sync (LN/LS/.../OF labels per paper_5_decisions.md §3).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper5_common import (  # noqa: E402
    STRUCTURE_INFO as _SI,
    CONFIG_TO_LABEL,
    METRIC_LABELS,
)
# Build (config_key, two_letter_label) pairs in canonical S1→S9 order
STRUCTURE_ORDER = [(_SI[s]['config_key'], _SI[s]['two_letter'])
                    for s in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']]


def load_param_bounds(params_yaml: Path, model: str = 'SPHY') -> dict:
    """Return {param_human_name: (lower, upper)} from default_params.yaml.

    Walks ``model.names`` (Xnn → human name) and pairs with ``model.lower``
    and ``model.upper`` (Xnn → bound). Returns the human-name keys used in
    the calibration_results CSV headers.
    """
    import yaml
    with params_yaml.open() as f:
        d = yaml.safe_load(f)
    block = d.get(model, {})
    names = block.get('names', {})
    lower = block.get('lower', {})
    upper = block.get('upper', {})
    bounds = {}
    for x_id, human in names.items():
        if x_id in lower and x_id in upper:
            bounds[human] = (float(lower[x_id]), float(upper[x_id]))
    return bounds


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


def analyze_csv(csv_path: Path, n_bins: int = 10,
                bounds: Optional[dict] = None) -> pd.DataFrame:
    """Compute PAWN, Spearman r, and range-fraction for every (param, objective).

    If ``bounds`` is provided ({param_name: (lower, upper)}), also fills
    ``range_frac`` = (Pareto max − min) / (prior upper − lower). A
    range_frac near 1 indicates the calibration kept the parameter
    essentially free; near 0 means it was tightly constrained.

    Returns a long-format DataFrame with one row per (param, objective).
    """
    df = pd.read_csv(csv_path)
    param_cols = [c for c in df.columns if c not in NON_PARAM_COLS]
    bounds = bounds or {}

    rows = []
    for p in param_cols:
        x = df[p].to_numpy(dtype=float)
        x_span = float(x.max() - x.min())
        if p in bounds:
            lo, hi = bounds[p]
            prior_span = hi - lo
            range_frac = float(x_span / prior_span) if prior_span > 0 else float('nan')
        else:
            range_frac = float('nan')

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
                'x_span':      x_span,
                'range_frac':  range_frac,
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
    col_order = [s for c, s in STRUCTURE_ORDER if c in combined['structure'].unique()]
    combined = combined.copy()
    combined['s_label'] = combined['structure'].map(CONFIG_TO_LABEL)

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


def plot_identifiability_vs_sensitivity(combined: pd.DataFrame, catchment: str,
                                          out_path: Path) -> None:
    """Scatter: range_frac (X) vs PAWN sensitivity (Y), one panel per objective.

    Quadrant interpretation (per panel):
      - top-left   (low range_frac, high PAWN): SENSITIVE + IDENTIFIABLE
                   → these parameters matter and the calibration found them
      - top-right  (high range_frac, high PAWN): SENSITIVE + EQUIFINAL
                   → these matter but multiple values work (compensating)
      - bottom-left  (low range_frac, low PAWN): WELL-CONSTRAINED but UNINFORMATIVE
                   → fixed at narrow range, but the fit is insensitive to it
      - bottom-right (high range_frac, low PAWN): FIXABLE
                   → could be fixed at a literature default with no loss

    Markers colored by structure (S1-S9). Each point labelled with the
    parameter name to make the diagnostic readable.
    """
    sub = combined.dropna(subset=['range_frac', 'pawn_median']).copy()
    if sub.empty:
        return
    sub['s_label'] = sub['structure'].map(CONFIG_TO_LABEL)

    # Distinct colors per structure (S1-S9)
    palette = plt.cm.tab10(np.linspace(0, 1, 9))
    s_color = {s: palette[i] for i, (_, s) in enumerate(STRUCTURE_ORDER)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, sharex=True)
    for ax, obj in zip(axes, OBJECTIVE_COLS):
        ss = sub[sub['objective'] == obj]
        if ss.empty:
            ax.axis('off')
            continue
        for s_lab, grp in ss.groupby('s_label'):
            ax.scatter(grp['range_frac'], grp['pawn_median'],
                       s=50, alpha=0.7, color=s_color.get(s_lab, 'grey'),
                       label=s_lab, edgecolor='k', linewidth=0.4)
        # Quadrant guide lines at the medians of all points in this objective
        med_rf = ss['range_frac'].median()
        med_pw = ss['pawn_median'].median()
        ax.axvline(med_rf, color='grey', ls=':', lw=0.8)
        ax.axhline(med_pw, color='grey', ls=':', lw=0.8)

        # Label only the top-5 most-sensitive points per panel (avoid clutter)
        top5 = ss.nlargest(5, 'pawn_median')
        for _, r in top5.iterrows():
            ax.annotate(r['parameter'].replace('Sphy_', ''),
                        xy=(r['range_frac'], r['pawn_median']),
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=7, alpha=0.8)

        ax.set_xlabel('range fraction on Pareto / prior')
        ax.set_title(obj.replace('obj_', ''))
        if ax is axes[0]:
            ax.set_ylabel('PAWN sensitivity (median KS)')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, max(0.75, ss['pawn_median'].max() * 1.05))
        ax.grid(alpha=0.3)

    # Single legend on the rightmost axis
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        # Order structures S1-S9
        order = sorted(range(len(labels)), key=lambda i: labels[i])
        axes[-1].legend([handles[i] for i in order], [labels[i] for i in order],
                         loc='upper right', fontsize=8, title='Structure')

    fig.suptitle(f'Identifiability vs Sensitivity — catchment {catchment}\n'
                  '(top-left = sensitive & identifiable; bottom-right = fixable)',
                  fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_dotty(csv_path: Path, struct: str, catchment: str,
                out_path: Path, sample_n: int = 2000) -> None:
    """Dotty plots: 2D scatter of each parameter vs each objective.

    Grid: rows = parameters (calibrated in this structure), cols = objectives.
    Subsamples to ``sample_n`` points for plot readability. Reveals the
    SHAPE of the response (monotonic / threshold / U-shape / fuzzy),
    complementing the PAWN magnitude.
    """
    df = pd.read_csv(csv_path)
    param_cols = [c for c in df.columns if c not in NON_PARAM_COLS]
    obj_cols = [c for c in OBJECTIVE_COLS if c in df.columns]
    if not obj_cols or not param_cols:
        return

    # Subsample for plot clarity
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=0)

    n_p, n_o = len(param_cols), len(obj_cols)
    fig, axes = plt.subplots(n_p, n_o, figsize=(3 * n_o, 1.6 * n_p),
                              sharex=False, sharey='col',
                              squeeze=False)

    for i, p in enumerate(param_cols):
        for j, obj in enumerate(obj_cols):
            ax = axes[i][j]
            ax.scatter(df[p], df[obj], s=4, alpha=0.4, color='#4575b4',
                        edgecolor='none')
            # Highlight top-1% in each objective (best solutions)
            top1pct = df.nlargest(max(20, len(df) // 100), obj)
            ax.scatter(top1pct[p], top1pct[obj], s=8, alpha=0.9, color='#d73027',
                        edgecolor='none')
            if i == 0:
                ax.set_title(obj.replace('obj_', ''), fontsize=10)
            if j == 0:
                ax.set_ylabel(p.replace('Sphy_', ''), fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)

    fig.suptitle(f'Dotty plots — catchment {catchment} / {struct}\n'
                  'blue = all samples, red = top 1% per objective',
                  fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_cross_catchment_overlay(combined_all: pd.DataFrame,
                                   out_path: Path,
                                   structure: Optional[str] = None,
                                   value_col: str = 'pawn_median') -> None:
    """Cross-catchment overlay: per-objective heatmap (params × catchments).

    If ``structure`` is given, plots that structure only. Otherwise averages
    PAWN values across structures for each (param, catchment) cell, which
    gives the catchment-level sensitivity signature regardless of structure.

    Designed to reveal whether the same parameters dominate sensitivity
    across climate regimes (Swiss vs UIB) or whether the ranking flips —
    direct evidence for the "does the answer depend on climate regime?"
    sub-question in Paper 5.
    """
    df = combined_all.copy()
    if 'catchment' not in df.columns:
        return
    if structure is not None:
        df = df[df['structure'] == structure]
        title_suffix = f' — structure {CONFIG_TO_LABEL.get(structure, structure)}'
    else:
        title_suffix = ' — averaged over 9 structures'

    # Catchment column order: UIB first, then Swiss in ascending ID
    catch_order = sorted(df['catchment'].unique(),
                          key=lambda c: (str(c).startswith('2'), str(c)))

    fig, axes = plt.subplots(1, 3, figsize=(15, max(6, 0.4 * df['parameter'].nunique() + 1)),
                              sharey=True)
    for ax, obj in zip(axes, OBJECTIVE_COLS):
        sub = df[df['objective'] == obj]
        if sub.empty:
            ax.axis('off')
            continue
        # Pivot: rows=parameter, cols=catchment, values=mean PAWN
        piv = sub.pivot_table(index='parameter', columns='catchment',
                               values=value_col, aggfunc='mean')
        piv = piv.reindex(columns=[c for c in catch_order if c in piv.columns])
        piv = piv.loc[piv.mean(axis=1, skipna=True).sort_values(ascending=False).index]

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='#d9d9d9')
        ma = np.ma.array(piv.values, mask=~np.isfinite(piv.values))
        im = ax.imshow(ma, aspect='auto', cmap=cmap, vmin=0, vmax=0.6)

        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(piv.columns, fontsize=9, rotation=45, ha='right')
        if ax is axes[0]:
            ax.set_yticks(range(piv.shape[0]))
            ax.set_yticklabels(piv.index, fontsize=8)
        ax.set_title(obj.replace('obj_', ''), fontsize=11)

        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            color='white' if v < 0.3 else 'black', fontsize=7)

    fig.suptitle(f'PAWN sensitivity across catchments{title_suffix}',
                  fontsize=12, y=1.02)
    fig.colorbar(im, ax=axes, label='PAWN (median KS)', shrink=0.7)
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
                        default=None,
                        help='Output directory for CSVs + plots. '
                             'Default: <repo>/plots/pareto_sensitivity/<catchment>/')
    parser.add_argument('--n-bins', type=int, default=10,
                        help='PAWN conditional-bin count (Pianosi recommends 10)')
    parser.add_argument('--plot', action='store_true',
                        help='Save PAWN + Spearman heatmaps (one per structure)')
    parser.add_argument('--params-yaml', type=Path,
                        default=Path(__file__).resolve().parent.parent / 'src/config/default_params.yaml',
                        help='Path to default_params.yaml for prior-bound lookup (range_frac).')
    parser.add_argument('--dotty', action='store_true',
                        help='Also generate dotty plots (per-param vs per-objective scatter). '
                             'Heavier output — adds 1 PNG per structure.')
    args = parser.parse_args()

    # Default outdir: <repo>/plots/pareto_sensitivity/<catchment>/
    if args.outdir is None:
        repo_root = Path(__file__).resolve().parent.parent
        args.outdir = repo_root / 'plots' / 'pareto_sensitivity' / args.catchment
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {args.outdir}')

    # Load parameter bounds for range_frac normalisation
    bounds = {}
    if args.params_yaml.exists():
        try:
            bounds = load_param_bounds(args.params_yaml, model='SPHY')
        except Exception as e:
            print(f'  (could not load bounds from {args.params_yaml}: {e})')
    else:
        print(f'  (params_yaml not found at {args.params_yaml}; range_frac will be NaN)')

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
        df_long = analyze_csv(csv, n_bins=args.n_bins, bounds=bounds)
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
        if args.dotty:
            base = f'{args.catchment}_{struct}'
            plot_dotty(csv, struct, args.catchment,
                        out_path=args.outdir / f'{base}_dotty.png')

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
        # Identifiability vs sensitivity scatter (requires bounds → range_frac)
        if combined['range_frac'].notna().any():
            idsens_path = args.outdir / f'{args.catchment}_identifiability_vs_sensitivity.png'
            plot_identifiability_vs_sensitivity(combined, args.catchment, idsens_path)
            print(f'Identifiability vs sensitivity: {idsens_path}')
        else:
            print('Identifiability plot skipped (no bounds available).')
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
