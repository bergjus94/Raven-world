#!/usr/bin/env python3
"""Cross-catchment analysis of the 3 Swiss Pareto runs (2268, 2256, 2161).

Produces a comprehensive set of comparison plots for Paper 5 methodology:

  A. Strategy comparison via post-hoc Pareto slicing.
     Simulates Q-only / snow-only / baseflow-only / Q+snow / Q+baseflow / all-3
     calibration outcomes by picking the appropriate point from the existing
     Pareto front. No additional model runs needed.

  B. Selection-rule comparison.
     Per catchment, applies 4 different selection rules to the same Pareto
     front: Pareto-range Tchebycheff, theoretical-bounds Tchebycheff,
     SCEUA-style weighted-sum (0.4/0.3/0.3), and ε-constraint (Q-best within
     behavioral set thresholded at 0.5/0.5/0.4).

  D. Parameter clustering across selection rules.
     Parallel coordinates of calibrated parameter values for each (catchment,
     rule) pair — visualizes which rules end up at similar vs different
     parameter values.

  F. SCEUA vs NSGAII algorithm comparison.
     For 2256 and 2161 where we have both, compares convergence behavior and
     parameter-space exploration.

Plots saved to a local folder; the wrapper script can rsync to server.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CATCHMENTS = ['2268', '2256', '2161']
LABELS = {'2268': 'Rhone @ Gletsch (2268)',
          '2256': 'Rosegbach @ Pontresina (2256)',
          '2161': 'Massa @ Aletsch (2161)'}
OBJ_COLS = ['obj_Q', 'obj_snow', 'obj_baseflow']
OBJ_LABELS = {'obj_Q': 'Q-KGE',
              'obj_snow': '1−RMSE snow',
              'obj_baseflow': 'baseflow KGE'}
OBJ_COLORS = {'obj_Q': '#1f77b4',
              'obj_snow': '#2ca02c',
              'obj_baseflow': '#d62728'}

# Default ε-constraint behavioral thresholds
EPS_THRESHOLDS = {'obj_Q': 0.85, 'obj_snow': 0.50, 'obj_baseflow': 0.40}

# Weighted-sum weights (matching production SCEUA setup)
SCEUA_WEIGHTS = {'obj_Q': 0.4, 'obj_snow': 0.3, 'obj_baseflow': 0.3}


def find_pareto_front(values: np.ndarray) -> np.ndarray:
    n = len(values)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dom = ((values >= values[i]).all(axis=1) & (values > values[i]).any(axis=1))
        dom[i] = False
        if dom.any():
            is_pareto[i] = False
    return is_pareto


def load_pareto(c: str, root: str = '/tmp/swiss_pareto') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load full results + Pareto-front subset for a catchment."""
    df = pd.read_csv(f'{root}/pareto_{c}.csv')
    obj_vals = df[OBJ_COLS].replace(-9999, np.nan)
    df = df.loc[obj_vals.notna().all(axis=1)].reset_index(drop=True)
    mask = find_pareto_front(df[OBJ_COLS].values)
    return df, df.loc[mask].reset_index(drop=True)


# ----- ANALYSIS A: STRATEGY COMPARISON VIA PARETO SLICING -----------------

def pick_max_single(front: pd.DataFrame, obj: str):
    return front.loc[front[obj].idxmax()]


def pick_pareto2d_tcheb(front: pd.DataFrame, objs: list[str]):
    """Find 2D Pareto subset on these two objectives, then Tchebycheff-best
    with theoretical bounds [0,1]."""
    vals = front[objs].values
    is_p2 = find_pareto_front(vals)
    sub = front.loc[is_p2].reset_index(drop=True)
    rescaled = np.clip(sub[objs].values, 0.0, 1.0)
    dist = (1.0 - rescaled).max(axis=1)
    return sub.iloc[dist.argmin()]


def pick_pareto3d_tcheb(front: pd.DataFrame, rescaling: str = 'theoretical'):
    """Tchebycheff-best on the full 3D front."""
    if rescaling == 'theoretical':
        rescaled = np.clip(front[OBJ_COLS].values, 0.0, 1.0)
    elif rescaling == 'pareto':
        lo = front[OBJ_COLS].min().values
        hi = front[OBJ_COLS].max().values
        span = hi - lo
        span[span == 0] = 1.0
        rescaled = (front[OBJ_COLS].values - lo) / span
    else:
        raise ValueError(f"Unknown rescaling: {rescaling}")
    dist = (1.0 - rescaled).max(axis=1)
    return front.iloc[dist.argmin()]


def pick_weighted_sum(front: pd.DataFrame, weights: dict[str, float] = None):
    weights = weights or SCEUA_WEIGHTS
    score = sum(weights[o] * front[o] for o in OBJ_COLS)
    return front.iloc[score.values.argmax()]


def pick_epsilon_constraint(front: pd.DataFrame, thresholds: dict[str, float] = None):
    thresholds = thresholds or EPS_THRESHOLDS
    behavioral = front.copy()
    for o, t in thresholds.items():
        behavioral = behavioral[behavioral[o] >= t]
    if len(behavioral) == 0:
        # Fall back to "best Q" if no behavioral points
        return front.loc[front.obj_Q.idxmax()], 0
    return behavioral.loc[behavioral.obj_Q.idxmax()], len(behavioral)


def analyze_strategies(front: pd.DataFrame) -> pd.DataFrame:
    """Apply 6 simulated strategies + 4 selection rules → return per-objective values."""
    strategies = {}
    strategies['Q-only']        = pick_max_single(front, 'obj_Q')
    strategies['snow-only']     = pick_max_single(front, 'obj_snow')
    strategies['baseflow-only'] = pick_max_single(front, 'obj_baseflow')
    strategies['Q+snow']        = pick_pareto2d_tcheb(front, ['obj_Q', 'obj_snow'])
    strategies['Q+baseflow']    = pick_pareto2d_tcheb(front, ['obj_Q', 'obj_baseflow'])
    strategies['all-3 (theo. Tcheb)'] = pick_pareto3d_tcheb(front, 'theoretical')

    selection_rules = {}
    selection_rules['Pareto-range Tcheb.']     = pick_pareto3d_tcheb(front, 'pareto')
    selection_rules['theo. Tcheb.']            = pick_pareto3d_tcheb(front, 'theoretical')
    selection_rules['weighted-sum 0.4/0.3/0.3'] = pick_weighted_sum(front)
    eps_pt, eps_n = pick_epsilon_constraint(front)
    selection_rules[f'ε-constraint (n_beh={eps_n})'] = eps_pt

    rows = []
    for name, row in strategies.items():
        rows.append({'category': 'strategy', 'name': name, **{o: row[o] for o in OBJ_COLS}})
    for name, row in selection_rules.items():
        rows.append({'category': 'selection_rule', 'name': name, **{o: row[o] for o in OBJ_COLS}})
    return pd.DataFrame(rows), strategies, selection_rules


def plot_a_strategy_comparison(per_catchment_results: dict, outdir: Path):
    """A. Grouped-bar plot: per-catchment per-strategy per-objective achievement."""
    outdir.mkdir(parents=True, exist_ok=True)

    strategy_names = ['Q-only', 'snow-only', 'baseflow-only',
                      'Q+snow', 'Q+baseflow', 'all-3 (theo. Tcheb)']

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    for ax, c in zip(axes, CATCHMENTS):
        results = per_catchment_results[c]
        sdf = results[results.category == 'strategy'].set_index('name')
        x = np.arange(len(strategy_names))
        width = 0.27
        for i, o in enumerate(OBJ_COLS):
            ax.bar(x + (i - 1) * width,
                   [sdf.loc[s, o] for s in strategy_names],
                   width=width, label=OBJ_LABELS[o], color=OBJ_COLORS[o])
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(strategy_names, rotation=20, ha='right')
        ax.set_ylim(0, 1.0)
        ax.set_title(f'{LABELS[c]}: per-objective values at each simulated strategy')
        ax.legend(loc='upper right', fontsize=9, ncol=3)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel('score')

    fig.suptitle('A. Simulated calibration-strategy comparison\n(picks from the Pareto front, no new runs)',
                 fontsize=13, y=0.995)
    fig.tight_layout()
    fp = outdir / 'A_strategy_comparison_per_catchment.png'
    plt.savefig(fp, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fp}")


def plot_a_strategy_delta(per_catchment_results: dict, outdir: Path):
    """A.2: Delta plot — how much each simulated strategy COSTS or GAINS vs all-3."""
    outdir.mkdir(parents=True, exist_ok=True)
    strategies_compare = ['Q-only', 'snow-only', 'baseflow-only', 'Q+snow', 'Q+baseflow']

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, c in zip(axes, CATCHMENTS):
        results = per_catchment_results[c]
        sdf = results[results.category == 'strategy'].set_index('name')
        # baseline = all-3 (theoretical Tchebycheff)
        baseline = sdf.loc['all-3 (theo. Tcheb)']
        x = np.arange(len(strategies_compare))
        width = 0.27
        for i, o in enumerate(OBJ_COLS):
            deltas = [sdf.loc[s, o] - baseline[o] for s in strategies_compare]
            ax.bar(x + (i - 1) * width, deltas, width=width,
                   label=OBJ_LABELS[o], color=OBJ_COLORS[o])
        ax.axhline(0, color='black', lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies_compare, rotation=20, ha='right')
        ax.set_title(f'{LABELS[c]}: Δ vs all-3 strategy (positive = strategy wins; negative = loses)')
        ax.legend(loc='best', fontsize=9, ncol=3)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel('Δ score (this strategy − all-3)')

    fig.suptitle('A.2 Cost/benefit of restricting the objective set\n(baseline = all-3 theo. Tcheb)',
                 fontsize=13, y=0.995)
    fig.tight_layout()
    fp = outdir / 'A2_strategy_delta_vs_all_three.png'
    plt.savefig(fp, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fp}")


# ----- ANALYSIS B: SELECTION-RULE COMPARISON ------------------------------

def plot_b_selection_rules(per_catchment_results: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    rule_names = ['Pareto-range Tcheb.', 'theo. Tcheb.', 'weighted-sum 0.4/0.3/0.3']
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, c in zip(axes, CATCHMENTS):
        results = per_catchment_results[c]
        sdf = results[results.category == 'selection_rule'].set_index('name')
        # Find the ε-constraint row (its name has 'ε-constraint' but variable trailing string)
        eps_row_name = [n for n in sdf.index if n.startswith('ε-constraint')][0]
        ordered = rule_names + [eps_row_name]
        x = np.arange(len(ordered))
        width = 0.27
        for i, o in enumerate(OBJ_COLS):
            vals = [sdf.loc[r, o] for r in ordered]
            ax.bar(x + (i - 1) * width, vals, width=width,
                   label=OBJ_LABELS[o], color=OBJ_COLORS[o])
        ax.set_xticks(x)
        ax.set_xticklabels(ordered, rotation=15, ha='right', fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_title(f'{LABELS[c]}: which point each rule selects from the SAME Pareto front')
        ax.legend(loc='upper right', fontsize=9, ncol=3)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel('score')

    fig.suptitle('B. Selection rule comparison — 4 rules, same Pareto front',
                 fontsize=13, y=0.995)
    fig.tight_layout()
    fp = outdir / 'B_selection_rule_comparison.png'
    plt.savefig(fp, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fp}")


# ----- ANALYSIS D: PARAMETER CLUSTERING ACROSS RULES ----------------------

def plot_d_param_clustering(per_catchment_selection_rules: dict, outdir: Path):
    """Parallel coordinates of param values per (catchment, selection_rule)."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Gather all params across all catchments
    rows = []
    for c in CATCHMENTS:
        rules = per_catchment_selection_rules[c]
        for rule_name, row in rules.items():
            r = {'catchment': c, 'rule': rule_name}
            for k in row.index:
                if k.startswith('Sphy_'):
                    r[k] = row[k]
            rows.append(r)
    df = pd.DataFrame(rows)
    param_cols = [c for c in df.columns if c.startswith('Sphy_')]

    # Rescale each param to [0, 1] using the union of all values for visualization
    df_rescaled = df.copy()
    for p in param_cols:
        lo = df[p].min(); hi = df[p].max()
        span = max(hi - lo, 1e-12)
        df_rescaled[p] = (df[p] - lo) / span

    catch_color = {'2268': '#1f77b4', '2256': '#ff7f0e', '2161': '#2ca02c'}
    rule_linestyle = {
        'Pareto-range Tcheb.':            ':',
        'theo. Tcheb.':                   '-',
        'weighted-sum 0.4/0.3/0.3':       '--',
    }

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(param_cols))

    for _, row in df_rescaled.iterrows():
        rule = row.rule
        ls = '-'
        for k, v in rule_linestyle.items():
            if rule.startswith(k.split(' ')[0][:5]):  # loose match
                ls = v
                break
        if rule.startswith('ε'):
            ls = '-.'
        ax.plot(x, [row[p] for p in param_cols],
                color=catch_color[row.catchment],
                linestyle=ls, lw=1.6, alpha=0.85,
                label=f'{row.catchment} — {row.rule}')

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('Sphy_', '') for p in param_cols], rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('parameter value (rescaled to [0, 1])')
    ax.set_title('D. Parallel coordinates: parameter values selected by 4 rules × 3 catchments\n(close lines = rules pick similar params; spread lines = rules disagree)')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', fontsize=8, framealpha=0.95)
    plt.tight_layout()

    fp = outdir / 'D_param_clustering_parallel_coords.png'
    plt.savefig(fp, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fp}")


# ----- ANALYSIS G: PARAMETER RANGES BY STRATEGY ---------------------------

def strategy_topk(front: pd.DataFrame, strategy: str, k: int = 10) -> pd.DataFrame:
    """Return the top-K parameter sets from the Pareto front under the
    specified simulated calibration strategy.
    """
    if strategy == 'Q-only':
        return front.nlargest(k, 'obj_Q')
    elif strategy == 'snow-only':
        return front.nlargest(k, 'obj_snow')
    elif strategy == 'baseflow-only':
        return front.nlargest(k, 'obj_baseflow')
    elif strategy == 'Q+snow':
        # equal-weight on 2D using theoretical bounds
        sc = 0.5 * np.clip(front.obj_Q, 0, 1) + 0.5 * np.clip(front.obj_snow, 0, 1)
        return front.iloc[sc.values.argsort()[::-1][:k]]
    elif strategy == 'Q+baseflow':
        sc = 0.5 * np.clip(front.obj_Q, 0, 1) + 0.5 * np.clip(front.obj_baseflow, 0, 1)
        return front.iloc[sc.values.argsort()[::-1][:k]]
    elif strategy == 'all-3':
        sc = (np.clip(front.obj_Q, 0, 1)
              + np.clip(front.obj_snow, 0, 1)
              + np.clip(front.obj_baseflow, 0, 1)) / 3.0
        return front.iloc[sc.values.argsort()[::-1][:k]]
    else:
        raise ValueError(strategy)


def plot_g_param_ranges_by_strategy(per_catchment_pareto: dict, outdir: Path, k: int = 10):
    """G. For each parameter: box-plot of values across strategies, per catchment.
    Reveals where strategies agree vs disagree on parameter selection.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    strategies = ['Q-only', 'snow-only', 'baseflow-only', 'Q+snow', 'Q+baseflow', 'all-3']

    # Collect: per (catchment, strategy, param) the K param values
    # data[param] = DataFrame with columns: catchment, strategy, value
    one_front = next(iter(per_catchment_pareto.values()))
    param_cols = [c for c in one_front.columns if c.startswith('Sphy_')]

    rows = []
    for c, front in per_catchment_pareto.items():
        for strat in strategies:
            topk = strategy_topk(front, strat, k=k)
            for _, row in topk.iterrows():
                for p in param_cols:
                    rows.append({'catchment': c, 'strategy': strat,
                                 'param': p, 'value': row[p]})
    long = pd.DataFrame(rows)

    # Compute divergence per parameter (across strategies, within each catchment, then averaged)
    div = []
    for p in param_cols:
        sub = long[long.param == p]
        # For each catchment: std across strategy-medians
        spreads = []
        for c in CATCHMENTS:
            sub_c = sub[sub.catchment == c]
            med_per_strat = sub_c.groupby('strategy').value.median()
            # normalize spread by parameter's value range
            v_min = sub_c.value.min(); v_max = sub_c.value.max()
            v_range = max(v_max - v_min, 1e-12)
            spread = (med_per_strat.max() - med_per_strat.min()) / v_range
            spreads.append(spread)
        div.append({'param': p, 'mean_spread': float(np.mean(spreads))})
    div_df = pd.DataFrame(div).sort_values('mean_spread', ascending=False).reset_index(drop=True)
    div_df.to_csv(outdir / 'G_param_divergence_ranking.csv', index=False)

    # Per-param subplot grid — sort by divergence (most-disagreeing first)
    ordered_params = div_df.param.tolist()
    n_params = len(ordered_params)
    cols = 4
    rows = (n_params + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.0))
    axes = axes.flatten()

    strat_colors = {'Q-only': '#1f77b4', 'snow-only': '#2ca02c', 'baseflow-only': '#d62728',
                    'Q+snow': '#9467bd', 'Q+baseflow': '#8c564b', 'all-3': '#000000'}
    catch_offsets = {'2268': -0.25, '2256': 0.0, '2161': +0.25}

    for ax_i, p in enumerate(ordered_params):
        ax = axes[ax_i]
        sub = long[long.param == p]
        # box per (strategy × catchment) — strategies on x-axis, catchments offset
        positions = []
        labels_x = []
        box_data = []
        box_colors = []
        for si, strat in enumerate(strategies):
            for c in CATCHMENTS:
                vals = sub[(sub.strategy == strat) & (sub.catchment == c)].value.values
                pos = si + catch_offsets[c]
                box_data.append(vals)
                positions.append(pos)
                box_colors.append(strat_colors[strat])
            labels_x.append(strat)
        bp = ax.boxplot(box_data, positions=positions, widths=0.18,
                        patch_artist=True, showfliers=False, medianprops=dict(color='black'))
        for patch, col in zip(bp['boxes'], box_colors):
            patch.set_facecolor(col); patch.set_alpha(0.55)
        ax.set_xticks(np.arange(len(strategies)))
        ax.set_xticklabels(strategies, rotation=20, ha='right', fontsize=8)
        title = p.replace('Sphy_', '')
        ax.set_title(f"{title}  (div={div_df[div_df.param==p].mean_spread.iloc[0]:.2f})", fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    # turn off unused axes
    for j in range(n_params, len(axes)):
        axes[j].axis('off')

    # legend (catchments)
    legend_elems = [Patch(facecolor='lightgray', edgecolor='black',
                          label=f'{c}: {LABELS[c]}') for c in CATCHMENTS]
    # also legend for strategies (using colors)
    legend_strat = [Patch(facecolor=strat_colors[s], alpha=0.55, label=s) for s in strategies]
    fig.legend(handles=legend_elems + [Patch(facecolor='none', label='')] + legend_strat,
               loc='lower center', ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('G. Parameter value distributions across simulated calibration strategies\n'
                 f'(top-{k} Pareto points per strategy; sorted by divergence, most-divergent first; '
                 'div = (max-min strategy-median) / param range)',
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fp = outdir / 'G_param_ranges_by_strategy.png'
    plt.savefig(fp, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fp}")


def plot_h_divergence_summary(per_catchment_pareto: dict, outdir: Path, k: int = 10):
    """H. Summary heatmap: parameter × strategy, color = top-K median value.
    One panel per catchment; row colors indicate divergence rank.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    strategies = ['Q-only', 'snow-only', 'baseflow-only', 'Q+snow', 'Q+baseflow', 'all-3']
    one_front = next(iter(per_catchment_pareto.values()))
    param_cols = [c for c in one_front.columns if c.startswith('Sphy_')]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    div_csv = pd.read_csv(outdir / 'G_param_divergence_ranking.csv')
    ordered_params = div_csv.param.tolist()

    for ax, c in zip(axes, CATCHMENTS):
        front = per_catchment_pareto[c]
        # Build matrix: rows = params (ordered by divergence), cols = strategies
        matrix = np.zeros((len(ordered_params), len(strategies)))
        for j, strat in enumerate(strategies):
            topk = strategy_topk(front, strat, k=k)
            for i, p in enumerate(ordered_params):
                matrix[i, j] = topk[p].median()
        # Normalize each row to [0, 1] for color comparability across params
        norm_matrix = np.zeros_like(matrix)
        for i in range(len(ordered_params)):
            row = matrix[i]
            lo, hi = row.min(), row.max()
            span = max(hi - lo, 1e-12)
            norm_matrix[i] = (row - lo) / span

        im = ax.imshow(norm_matrix, aspect='auto', cmap='RdBu_r', vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(strategies)))
        ax.set_xticklabels(strategies, rotation=25, ha='right', fontsize=9)
        ax.set_yticks(np.arange(len(ordered_params)))
        ax.set_yticklabels([p.replace('Sphy_', '') for p in ordered_params], fontsize=8)
        ax.set_title(f'{LABELS[c]}', fontsize=11)

        # annotate cells with actual values
        for i in range(len(ordered_params)):
            for j in range(len(strategies)):
                ax.text(j, i, f'{matrix[i, j]:.2g}',
                        ha='center', va='center', fontsize=7, color='black')

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('row-normalized value\n(per parameter, blue=lowest strategy / red=highest strategy)',
                   fontsize=9)
    fig.suptitle('H. Per-catchment heatmap: median top-K parameter value across strategies\n'
                 '(rows sorted by divergence rank, most-divergent at top)',
                 fontsize=12, y=1.00)
    fp = outdir / 'H_param_strategy_heatmap.png'
    plt.savefig(fp, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fp}")


# ----- ANALYSIS F: SCEUA VS NSGAII ----------------------------------------

def plot_f_sceua_vs_nsgaii(outdir: Path):
    """Compare SCEUA and NSGAII for 2256 and 2161."""
    outdir.mkdir(parents=True, exist_ok=True)

    available = []
    for c in ['2256', '2161']:
        sceua_path = Path(f'/tmp/swiss_pareto/sceua_results_{c}.csv')
        if sceua_path.exists():
            available.append(c)
    if not available:
        print("  No SCEUA results to compare — skipping F.")
        return

    fig, axes = plt.subplots(len(available), 3, figsize=(15, 4.5 * len(available)))
    if len(available) == 1:
        axes = axes.reshape(1, 3)

    for r, c in enumerate(available):
        sceua = pd.read_csv(f'/tmp/swiss_pareto/sceua_results_{c}.csv')
        _, pareto_front = load_pareto(c)

        # Panel 1: SCEUA convergence (objective vs iteration)
        ax = axes[r, 0]
        if 'objective' in sceua.columns:
            ax.plot(sceua.index + 1, sceua['objective'], alpha=0.3, lw=0.7, label='SCEUA all')
            running_best = sceua['objective'].cummax()
            ax.plot(sceua.index + 1, running_best, lw=2, color='red', label='SCEUA running-best')
        ax.set_xlabel('iteration')
        ax.set_ylabel('weighted-sum objective')
        ax.set_title(f'{LABELS[c]}: SCEUA convergence (composite)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        # Panel 2: NSGAII Pareto front in Q vs baseflow space, with SCEUA points
        ax = axes[r, 1]
        if all(o in sceua.columns for o in ['obj_Q', 'obj_baseflow']):
            ax.scatter(sceua['obj_Q'], sceua['obj_baseflow'],
                       alpha=0.15, s=8, color='blue', label='SCEUA all evals')
            sceua_best_idx = sceua['objective'].idxmax()
            ax.scatter([sceua.loc[sceua_best_idx, 'obj_Q']],
                       [sceua.loc[sceua_best_idx, 'obj_baseflow']],
                       s=120, color='red', edgecolor='black', zorder=5,
                       label='SCEUA single-best')
        ax.scatter(pareto_front.obj_Q, pareto_front.obj_baseflow,
                   alpha=0.7, s=12, color='orange', label=f'NSGAII Pareto front (n={len(pareto_front)})')
        ax.set_xlabel('Q-KGE')
        ax.set_ylabel('baseflow KGE')
        ax.set_title(f'{c}: search-space comparison — Q × baseflow')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        # Panel 3: Q-KGE distribution histogram comparing exploration
        ax = axes[r, 2]
        if 'obj_Q' in sceua.columns:
            ax.hist(sceua['obj_Q'], bins=40, alpha=0.6, color='blue',
                    density=True, label=f'SCEUA all evals (n={len(sceua)})')
        nsgaii_full, _ = load_pareto(c)
        ax.hist(nsgaii_full.obj_Q, bins=40, alpha=0.5, color='orange',
                density=True, label=f'NSGAII all evals (n={len(nsgaii_full)})')
        ax.set_xlabel('Q-KGE')
        ax.set_ylabel('density')
        ax.set_title(f'{c}: Q-KGE exploration histograms')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle('F. SCEUA vs NSGAII algorithm comparison', fontsize=13, y=0.998)
    fig.tight_layout()
    fp = outdir / 'F_sceua_vs_nsgaii.png'
    plt.savefig(fp, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fp}")


# ----- DRIVER -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outdir', type=Path,
                        default=Path('/tmp/cross_catchment_plots'),
                        help='Local output directory (default /tmp/cross_catchment_plots)')
    parser.add_argument('--pareto-dir', type=str,
                        default='/tmp/swiss_pareto',
                        help='Directory with pareto_NNNN.csv files (default /tmp/swiss_pareto)')
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {outdir}")
    print(f"Catchments: {CATCHMENTS}")

    # Load Pareto fronts + run analyses
    per_catchment_results = {}
    per_catchment_strategies = {}
    per_catchment_selection_rules = {}
    per_catchment_pareto = {}

    print("\n=== loading Pareto fronts and computing strategy/selection picks ===")
    for c in CATCHMENTS:
        _, front = load_pareto(c, root=args.pareto_dir)
        print(f"  {c}: Pareto front size = {len(front)}")
        per_catchment_pareto[c] = front
        results, strategies, selection_rules = analyze_strategies(front)
        per_catchment_results[c] = results
        per_catchment_strategies[c] = strategies
        per_catchment_selection_rules[c] = selection_rules
        # write per-catchment summary CSV
        results.to_csv(outdir / f'summary_{c}.csv', index=False)

    print("\n=== Plot A: simulated strategy comparison ===")
    plot_a_strategy_comparison(per_catchment_results, outdir)
    plot_a_strategy_delta(per_catchment_results, outdir)

    print("\n=== Plot B: selection-rule comparison ===")
    plot_b_selection_rules(per_catchment_results, outdir)

    print("\n=== Plot D: parameter clustering across rules ===")
    plot_d_param_clustering(per_catchment_selection_rules, outdir)

    print("\n=== Plot G: parameter ranges by strategy (top-K box plots) ===")
    plot_g_param_ranges_by_strategy(per_catchment_pareto, outdir, k=10)

    print("\n=== Plot H: parameter × strategy heatmap per catchment ===")
    plot_h_divergence_summary(per_catchment_pareto, outdir, k=10)

    print("\n=== Plot F: SCEUA vs NSGAII ===")
    plot_f_sceua_vs_nsgaii(outdir)

    print("\nAll done. Output dir:")
    print(f"  {outdir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
