#!/usr/bin/env python3
"""Paper 5 — per-structure parameter table figures.

For each of the 9 subsurface structures (S1-S9 / LN-OF), produces a figure
showing:
  - Structure metadata (config name, key flags, hydrologic-process changes vs S1)
  - Calibrated parameter table: X-name, symbol, description, units, lower, upper, init

Designed for the paper methods section (Figure 3 candidate) and the
supplementary methods doc.

Output: <outdir>/<S-code>_<two-letter>_parameter_table.png  +  a combined
3×3 overview (ALL_structures_overview.png).

Naming, parameter lists, and X-info live in scripts/paper5_common.py.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper5_common import (
    STRUCTURE_INFO, STRUCTURE_ORDER, PARAM_INFO,
    ARCHITECTURE, CONNECTION,
)


# Per-structure configuration flags + process-change narrative.
# Kept local to this script because they're prose, not data.
STRUCTURE_FLAGS = {
    'S1': {
        'flags': [
            'perc_option = 1 (PERC_CONSTANT FAST→SLOW)',
            "fast_reservoir_release = 'linear'",
            "land_surface_routing = 'flush_to_fast'",
            "glacier_routing = 'none'",
        ],
        'process_change': 'reference structure — no additions vs baseline',
    },
    'S2': {
        'flags': [
            'perc_option = 1',
            "fast_reservoir_release = 'linear'",
            "land_surface_routing = 'flush_to_fast'",
            "glacier_routing = 'split_to_slow'  ← KEY CHANGE",
        ],
        'process_change': '+ :Split RAVEN_DEFAULT PONDED_WATER → SURFACE_WATER + SLOW_RESERVOIR (on MASKED_GLACIER) — partitions glacier melt: GlacROF·surface + (1−GlacROF)·SLOW_RES',
    },
    'S3': {
        'flags': [
            'perc_option = 1',
            "fast_reservoir_release = 'threshold'  ← KEY CHANGE",
            "land_surface_routing = 'flush_to_fast'",
            "glacier_routing = 'none'",
        ],
        'process_change': '+ :Baseflow BASE_THRESH_STOR FAST_RES → SURFACE_WATER (alongside existing BASE_LINEAR). Q0 = K0·max(0, S_fast − UZL) when above threshold, Q1 = K1·S_fast always.',
    },
    'S4': {
        'flags': [
            'perc_option = 1',
            "fast_reservoir_release = 'threshold'",
            "land_surface_routing = 'flush_to_fast'",
            "glacier_routing = 'split_to_slow'",
        ],
        'process_change': 'combines S2 :Split + S3 BASE_THRESH_STOR. Tests additivity/interaction of the two structural changes.',
    },
    'S5': {
        'flags': [
            'perc_option = 2  (PERC_POWER_LAW TOPSOIL→FAST + PERC_LINEAR FAST→SLOW)',
            "fast_reservoir_release = 'linear'",
            "land_surface_routing = 'direct'  ← KEY CHANGE",
            "glacier_routing = 'none'",
        ],
        'process_change': '− :Flush SURFACE_WATER → FAST_RES (removed): sat-excess routes directly to outlet via :CatchmentRoute.   + :Percolation PERC_POWER_LAW TOPSOIL → FAST_RES (X12, X13).   + :Percolation PERC_LINEAR FAST_RES → SLOW_RES (X14). FAST_RES becomes a genuine subsurface store.',
    },
    'S6': {
        'flags': [
            'perc_option = 2',
            "fast_reservoir_release = 'linear'",
            "land_surface_routing = 'direct'",
            "glacier_routing = 'split_to_slow'",
        ],
        'process_change': 'combines S5 SPHY-style architecture + S2 :Split glacier-GW partitioning to SLOW_RES.',
    },
    'S7': {
        'flags': [
            'perc_option = 1',
            "fast_reservoir_release = 'linear'",
            "land_surface_routing = 'flush_to_fast'",
            "glacier_routing = 'split_to_fast'  ← KEY CHANGE (vs S2)",
        ],
        'process_change': "mirror of S2 with :Split target = FAST_RESERVOIR instead of SLOW_RESERVOIR. Glacier melt's (1−GlacROF) fraction now enters the fast store; drains via K1 (days) rather than K2 (months). Tests whether the *destination* of the glacier-GW link matters as much as its existence.",
    },
    'S8': {
        'flags': [
            'perc_option = 1',
            "fast_reservoir_release = 'threshold'",
            "land_surface_routing = 'flush_to_fast'",
            "glacier_routing = 'split_to_fast'  ← KEY CHANGE (vs S4)",
        ],
        'process_change': 'mirror of S4 with :Split target = FAST_RESERVOIR. With BASE_THRESH_STOR active, summer melt routed to FAST gushes out at K0 (above-threshold) — likely the most extreme contrast between SLOW vs FAST destinations.',
    },
    'S9': {
        'flags': [
            'perc_option = 2',
            "fast_reservoir_release = 'linear'",
            "land_surface_routing = 'direct'",
            "glacier_routing = 'split_to_fast'  ← KEY CHANGE (vs S6)",
        ],
        'process_change': "mirror of S6 with :Split target = FAST_RESERVOIR. In SPHY-style architecture, FAST_RES sits in the cascade percolation chain (TOPSOIL→FAST→SLOW); glacier melt routed to FAST joins the cascade rather than bypassing it.",
    },
}


def load_bounds(path: Path = Path('src/config/default_params.yaml')):
    with open(path) as f:
        params = yaml.safe_load(f)
    sphy = params['SPHY']
    return sphy['lower'], sphy['upper'], sphy['init']


def fmt_num(x):
    if x is None:
        return '–'
    if abs(x) >= 100 or abs(x) < 0.001:
        return f'{x:.3g}'
    if abs(x) < 0.01:
        return f'{x:.4f}'
    if abs(x) < 1.0:
        return f'{x:.3g}'
    return f'{x:.2f}'


def render_structure(struct_key: str, bounds_lo, bounds_hi, bounds_init, outdir: Path):
    s = STRUCTURE_INFO[struct_key]
    flags = STRUCTURE_FLAGS[struct_key]
    n_params = len(s['params'])
    arch_name = ARCHITECTURE[s['architecture']][0]
    conn_name = CONNECTION[s['connection']][0]

    fig = plt.figure(figsize=(13, max(7.0, 1.4 + 0.34 * n_params + 2.0)))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 0.9, n_params * 0.35 + 0.5],
                          hspace=0.20)
    ax_header = fig.add_subplot(gs[0]); ax_header.axis('off')
    ax_flags  = fig.add_subplot(gs[1]); ax_flags.axis('off')
    ax_table  = fig.add_subplot(gs[2]); ax_table.axis('off')

    # Header — uses both codes + descriptive title
    ax_header.text(0.0, 0.90, f"{s['two_letter']} ({s['s_code']}): {s['description']}",
                   fontsize=15, weight='bold', va='top')
    ax_header.text(0.0, 0.68,
                   f'Architecture: {arch_name}   |   Glacier-GW connection: {conn_name}',
                   fontsize=10.5, va='top', color='#444')
    ax_header.text(0.0, 0.52, f'Configuration: {s["config_key"]}.yaml',
                   fontsize=10, family='monospace', va='top', color='#444')
    ax_header.text(0.0, 0.36,
                   f'Calibrated parameters: {n_params}',
                   fontsize=11, va='top', color='#444')

    # Process / structural change description
    ax_header.text(0.0, 0.18, 'Structural change vs S1 baseline:',
                   fontsize=10, weight='bold', va='top')
    ax_header.text(0.0, 0.02, flags['process_change'],
                   fontsize=9.5, va='top', wrap=True)

    # Configuration flags
    ax_flags.text(0.0, 0.95, 'Configuration flags:', fontsize=10, weight='bold', va='top')
    for i, line in enumerate(flags['flags']):
        color = '#cc4422' if '← KEY CHANGE' in line else '#222'
        ax_flags.text(0.02, 0.78 - i * 0.18, line, fontsize=9.5,
                      family='monospace', va='top', color=color)

    # Parameter table
    header = ['X#', 'symbol', 'description', 'units', 'lower', 'upper', 'init']
    rows = []
    for x in s['params']:
        symbol, descr, units = PARAM_INFO[x]
        rows.append([x, symbol, descr, units,
                     fmt_num(bounds_lo.get(x)),
                     fmt_num(bounds_hi.get(x)),
                     fmt_num(bounds_init.get(x))])

    table = ax_table.table(cellText=rows, colLabels=header,
                           cellLoc='left', colLoc='left',
                           loc='upper left',
                           colWidths=[0.06, 0.10, 0.40, 0.10, 0.08, 0.08, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.5)
    for i in range(len(header)):
        cell = table[0, i]
        cell.set_facecolor('#dfe9f3')
        cell.set_text_props(weight='bold')

    fig.suptitle(' ', y=0.99)
    fp = outdir / f'{s["s_code"]}_{s["two_letter"]}_parameter_table.png'
    plt.savefig(fp, dpi=130, bbox_inches='tight')
    plt.close()
    return fp


def render_combined(bounds_lo, bounds_hi, bounds_init, outdir: Path):
    """Combined overview of all 9 structures, lightweight version (3×3 grid)."""
    fig, axes = plt.subplots(3, 3, figsize=(22, 17))
    axes = axes.flatten()
    for ax, key in zip(axes, STRUCTURE_ORDER):
        s = STRUCTURE_INFO[key]
        ax.axis('off')
        ax.text(0.5, 0.96,
                f"{s['two_letter']} ({s['s_code']})",
                fontsize=13, weight='bold', ha='center', va='top')
        ax.text(0.5, 0.90, s['description'],
                fontsize=9, ha='center', va='top', color='#444', style='italic')
        ax.text(0.5, 0.83, f"{len(s['params'])} calibrated parameters",
                fontsize=9, ha='center', va='top', color='#666')

        header = ['X#', 'symbol', 'lo', 'hi']
        rows = [[x, PARAM_INFO[x][0],
                 fmt_num(bounds_lo.get(x)), fmt_num(bounds_hi.get(x))]
                for x in s['params']]
        tbl = ax.table(cellText=rows, colLabels=header, cellLoc='left',
                       loc='upper center', bbox=[0.02, 0.05, 0.96, 0.72],
                       colWidths=[0.20, 0.30, 0.25, 0.25])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for i in range(len(header)):
            cell = tbl[0, i]
            cell.set_facecolor('#dfe9f3')
            cell.set_text_props(weight='bold')

    fig.suptitle('Paper 5 — calibrated parameters per subsurface structure '
                  '(3×3 factorial: architecture × glacier-GW connection)',
                  fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fp = outdir / 'ALL_structures_overview.png'
    plt.savefig(fp, dpi=130, bbox_inches='tight')
    plt.close()
    return fp


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outdir', type=Path,
                        default=Path(__file__).resolve().parent.parent /
                                'plots' / 'structures',
                        help='Output directory (default: <repo>/plots/structures/)')
    parser.add_argument('--params-yaml', type=Path,
                        default=Path('src/config/default_params.yaml'))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    bounds_lo, bounds_hi, bounds_init = load_bounds(args.params_yaml)

    for key in STRUCTURE_ORDER:
        fp = render_structure(key, bounds_lo, bounds_hi, bounds_init, args.outdir)
        print(f'  Saved {fp}')

    fp = render_combined(bounds_lo, bounds_hi, bounds_init, args.outdir)
    print(f'  Saved {fp}')


if __name__ == '__main__':
    main()
