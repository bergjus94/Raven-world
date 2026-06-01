#!/usr/bin/env python3
"""Paper 5 — per-structure parameter table figures.

For each of the 6 subsurface structures (S1-S6), produces a figure showing:
  - Structure metadata (config name, key flags, hydrologic-process changes vs S1)
  - Calibrated parameter table: X-name, symbol, description, units, lower, upper, init

Designed for the paper methods section (Figure 3 candidate) and the
supplementary methods doc.

Output: cross_catchment_plots/structures/S{n}_parameter_table.png + a
combined six-panel overview.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml


# Hand-curated parameter descriptions with proper LaTeX-friendly notation
PARAM_INFO = {
    'X01': ('T_{rs}', 'Rain/snow transition temperature', '°C'),
    'X02': ('M_f', 'Snow melt factor (degree-day)', 'mm·d⁻¹·°C⁻¹'),
    'X03': ('CFR', 'Refreeze coupling ratio', '–'),
    'X04': ('SWI', 'Snow water-holding fraction', '–'),
    'X05': ('β', 'HBV β infiltration exponent', '–'),
    'X06': ('FC', 'Field capacity (fraction of porosity)', '–'),
    'X07': ('K_1', 'FAST_RES linear baseflow coefficient', 'd⁻¹'),
    'X08': ('K_2', 'SLOW_RES linear baseflow coefficient', 'd⁻¹'),
    'X09': ('T_c', 'Time of concentration', 'd'),
    'X10': ('h_{topsoil}', 'Topsoil layer thickness', 'm'),
    'X11': ('P_{fast}', 'PERC_CONSTANT FAST→SLOW peak rate', 'mm·d⁻¹'),
    'X12': ('P_{tops}', 'PERC_POWER_LAW TOPSOIL→FAST peak rate', 'mm·d⁻¹'),
    'X13': ('n', 'PERC_POWER_LAW exponent on TOPSOIL', '–'),
    'X14': ('K_{perc}', 'PERC_LINEAR FAST→SLOW coefficient', 'd⁻¹'),
    'X15': ('CR', 'Capillary rise rate (FAST→TOPSOIL)', 'mm·d⁻¹'),
    'X16': ('GlacROF', 'Glacier-melt surface routing fraction', '–'),
    'X17': ('UZL', 'FAST_RES storage threshold for Q0 (BASE_THRESH_STOR)', 'mm'),
    'X18': ('K_0', 'FAST_RES above-threshold release rate', 'd⁻¹'),
}

# Per-structure metadata
STRUCTURES = {
    'S1': {
        'name': 'Baseline: HBV-linear, no glacier-GW',
        'config': 'glogem_subdaily_opt1',
        'flags': [
            'perc_option = 1 (PERC_CONSTANT FAST→SLOW)',
            "fast_reservoir_release = 'linear'",
            "land_surface_routing = 'flush_to_fast'",
            "glacier_routing = 'none'",
        ],
        'process_change': 'reference structure — no additions vs baseline',
        'params': ['X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08',
                   'X09', 'X10', 'X11', 'X15'],
    },
    'S2': {
        'name': 'Baseline + glacier-GW connectivity',
        'config': 'glogem_subdaily_opt1_glaciergw',
        'flags': [
            'perc_option = 1',
            "fast_reservoir_release = 'linear'",
            "land_surface_routing = 'flush_to_fast'",
            "glacier_routing = 'split_to_slow'  ← KEY CHANGE",
        ],
        'process_change': "+ :Split RAVEN_DEFAULT PONDED_WATER → SURFACE_WATER + SLOW_RESERVOIR (on MASKED_GLACIER) — partitions glacier melt: GlacROF·surface + (1−GlacROF)·SLOW_RES",
        'params': ['X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08',
                   'X09', 'X10', 'X11', 'X15', 'X16'],
    },
    'S3': {
        'name': 'Baseline + HBV-Light Q0+Q1 threshold release',
        'config': 'glogem_subdaily_opt1_threshold',
        'flags': [
            'perc_option = 1',
            "fast_reservoir_release = 'threshold'  ← KEY CHANGE",
            "land_surface_routing = 'flush_to_fast'",
            "glacier_routing = 'none'",
        ],
        'process_change': '+ :Baseflow BASE_THRESH_STOR FAST_RES → SURFACE_WATER (alongside existing BASE_LINEAR). Q0 = K0 · max(0, S_fast − UZL) when above threshold, Q1 = K1 · S_fast always.',
        'params': ['X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08',
                   'X09', 'X10', 'X11', 'X15', 'X17', 'X18'],
    },
    'S4': {
        'name': 'HBV-Light Q0+Q1 threshold + glacier-GW connectivity',
        'config': 'glogem_subdaily_opt1_threshold_glaciergw',
        'flags': [
            'perc_option = 1',
            "fast_reservoir_release = 'threshold'",
            "land_surface_routing = 'flush_to_fast'",
            "glacier_routing = 'split_to_slow'",
        ],
        'process_change': 'combines S2 :Split + S3 BASE_THRESH_STOR. Tests additivity/interaction of the two structural changes.',
        'params': ['X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08',
                   'X09', 'X10', 'X11', 'X15', 'X16', 'X17', 'X18'],
    },
    'S5': {
        'name': 'SPHY-faithful: direct overland routing + cascade percolation',
        'config': 'glogem_subdaily_opt2_sphy_faithful',
        'flags': [
            'perc_option = 2  (PERC_POWER_LAW TOPSOIL→FAST + PERC_LINEAR FAST→SLOW)',
            "fast_reservoir_release = 'linear'",
            "land_surface_routing = 'direct'  ← KEY CHANGE",
            "glacier_routing = 'none'",
        ],
        'process_change': '– :Flush SURFACE_WATER → FAST_RES (removed): sat-excess routes directly to outlet via :CatchmentRoute.   + :Percolation PERC_POWER_LAW TOPSOIL → FAST_RES (X12, X13).   + :Percolation PERC_LINEAR FAST_RES → SLOW_RES (X14). FAST_RES becomes a genuine subsurface store (Terink SPHY).',
        'params': ['X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08',
                   'X09', 'X10', 'X12', 'X13', 'X14', 'X15'],
    },
    'S6': {
        'name': 'SPHY-faithful + glacier-GW connectivity',
        'config': 'glogem_subdaily_opt2_sphy_faithful_glaciergw',
        'flags': [
            'perc_option = 2',
            "fast_reservoir_release = 'linear'",
            "land_surface_routing = 'direct'",
            "glacier_routing = 'split_to_slow'",
        ],
        'process_change': 'combines S5 SPHY-faithful architecture + S2 :Split glacier-GW partitioning.',
        'params': ['X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08',
                   'X09', 'X10', 'X12', 'X13', 'X14', 'X15', 'X16'],
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
    s = STRUCTURES[struct_key]
    n_params = len(s['params'])

    fig = plt.figure(figsize=(13, max(7.0, 1.4 + 0.34 * n_params + 2.0)))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 0.9, n_params * 0.35 + 0.5],
                          hspace=0.20)
    ax_header = fig.add_subplot(gs[0]); ax_header.axis('off')
    ax_flags  = fig.add_subplot(gs[1]); ax_flags.axis('off')
    ax_table  = fig.add_subplot(gs[2]); ax_table.axis('off')

    # Header
    ax_header.text(0.0, 0.85, f'{struct_key}: {s["name"]}',
                   fontsize=15, weight='bold', va='top')
    ax_header.text(0.0, 0.60, f'Configuration: {s["config"]}.yaml',
                   fontsize=10, family='monospace', va='top', color='#444')
    ax_header.text(0.0, 0.42,
                   f'Calibrated parameters: {n_params}',
                   fontsize=11, va='top', color='#444')

    # Process / structural change description
    ax_header.text(0.0, 0.18, 'Structural change vs S1 baseline:',
                   fontsize=10, weight='bold', va='top')
    ax_header.text(0.0, 0.02, s['process_change'],
                   fontsize=9.5, va='top', wrap=True)

    # Configuration flags
    ax_flags.text(0.0, 0.95, 'Configuration flags:',
                  fontsize=10, weight='bold', va='top')
    for i, line in enumerate(s['flags']):
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
    fp = outdir / f'{struct_key}_parameter_table.png'
    plt.savefig(fp, dpi=130, bbox_inches='tight')
    plt.close()
    return fp


def render_combined(bounds_lo, bounds_hi, bounds_init, outdir: Path):
    """Combined overview of all 6 structures, lightweight version."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()
    for ax, key in zip(axes, ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']):
        s = STRUCTURES[key]
        ax.axis('off')
        ax.text(0.5, 0.95, f"{key}: {s['name']}", fontsize=11.5, weight='bold',
                ha='center', va='top')
        ax.text(0.5, 0.86, f"{len(s['params'])} calibrated parameters",
                fontsize=9, ha='center', va='top', color='#666')

        header = ['X#', 'symbol', 'lo', 'hi']
        rows = [[x, PARAM_INFO[x][0], fmt_num(bounds_lo.get(x)), fmt_num(bounds_hi.get(x))]
                for x in s['params']]
        tbl = ax.table(cellText=rows, colLabels=header, cellLoc='left',
                       loc='upper center', bbox=[0.02, 0.05, 0.96, 0.78],
                       colWidths=[0.20, 0.30, 0.25, 0.25])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        for i in range(len(header)):
            cell = tbl[0, i]
            cell.set_facecolor('#dfe9f3')
            cell.set_text_props(weight='bold')

    fig.suptitle('Paper 5 — calibrated parameters per subsurface structure (S1–S6)',
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
                        default=Path('/tmp/cross_catchment_plots/structures'),
                        help='Output directory')
    parser.add_argument('--params-yaml', type=Path,
                        default=Path('src/config/default_params.yaml'))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    bounds_lo, bounds_hi, bounds_init = load_bounds(args.params_yaml)

    for key in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']:
        fp = render_structure(key, bounds_lo, bounds_hi, bounds_init, args.outdir)
        print(f'  Saved {fp}')

    fp = render_combined(bounds_lo, bounds_hi, bounds_init, args.outdir)
    print(f'  Saved {fp}')


if __name__ == '__main__':
    main()
