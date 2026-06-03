#!/usr/bin/env python3
"""Paper 5 — combined structure schematic + parameter table per structure.

For each of S1-S6, produces a single landscape figure with:
  Left half:  boxes-and-arrows schematic (reservoirs + processes)
  Right half: stacked header / config flags / parameter table

This is the paper-figure-ready version — drops directly into the methods
section. Companion scripts produced the components separately
(structure_schematics.py, structure_parameter_tables.py).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

# Reuse the rendering primitives from the dedicated scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))
from structure_schematics import base_schematic, BOXES  # noqa: E402
from structure_parameter_tables import (  # noqa: E402
    STRUCTURE_FLAGS, load_bounds, fmt_num,
)
from paper5_common import STRUCTURE_INFO as STRUCTURES, PARAM_INFO  # noqa: E402


def render_table_panel(fig, gs_table_section, struct_key,
                       bounds_lo, bounds_hi, bounds_init):
    """Render the header+flags+table panel inside a gridspec section."""
    s = STRUCTURES[struct_key]
    flags = STRUCTURE_FLAGS[struct_key]
    n_params = len(s['params'])

    sub = gs_table_section.subgridspec(3, 1, height_ratios=[1.3, 1.0, n_params * 0.30 + 0.5],
                                        hspace=0.18)
    ax_header = fig.add_subplot(sub[0]); ax_header.axis('off')
    ax_flags  = fig.add_subplot(sub[1]); ax_flags.axis('off')
    ax_table  = fig.add_subplot(sub[2]); ax_table.axis('off')

    # Header
    ax_header.text(0.0, 0.92, f"{s['two_letter']} ({s['s_code']}): {s['description']}",
                   fontsize=14, weight='bold', va='top')
    ax_header.text(0.0, 0.72, f'Configuration: {s["config_key"]}.yaml',
                   fontsize=9, family='monospace', va='top', color='#444')
    ax_header.text(0.0, 0.58,
                   f'Calibrated parameters: {n_params}',
                   fontsize=10, va='top', color='#444')
    ax_header.text(0.0, 0.40, 'Structural change vs S1 baseline:',
                   fontsize=9.5, weight='bold', va='top')
    ax_header.text(0.0, 0.22, flags['process_change'],
                   fontsize=8.5, va='top', wrap=True)

    # Flags
    ax_flags.text(0.0, 0.92, 'Configuration flags:',
                  fontsize=9.5, weight='bold', va='top')
    for i, line in enumerate(flags['flags']):
        color = '#cc4422' if '← KEY CHANGE' in line else '#222'
        ax_flags.text(0.02, 0.74 - i * 0.18, line, fontsize=8.5,
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
                           colWidths=[0.06, 0.10, 0.42, 0.10, 0.08, 0.08, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.4)
    for i in range(len(header)):
        cell = table[0, i]
        cell.set_facecolor('#dfe9f3')
        cell.set_text_props(weight='bold')


def render_combined(struct_key, bounds_lo, bounds_hi, bounds_init, outdir: Path):
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.05)

    # Left half: schematic
    ax_schem = fig.add_subplot(gs[0])
    base_schematic(ax_schem, struct_key)

    # Right half: header + flags + table
    render_table_panel(fig, gs[1], struct_key,
                       bounds_lo, bounds_hi, bounds_init)

    s = STRUCTURES[struct_key]
    fp = outdir / f'{s["s_code"]}_{s["two_letter"]}_combined.png'
    plt.savefig(fp, dpi=130, bbox_inches='tight')
    plt.close()
    return fp


def main():
    from paper5_common import STRUCTURE_ORDER

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
        fp = render_combined(key, bounds_lo, bounds_hi, bounds_init, args.outdir)
        print(f'  Saved {fp}')


if __name__ == '__main__':
    main()
