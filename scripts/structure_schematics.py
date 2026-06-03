#!/usr/bin/env python3
"""Paper 5 — per-structure model schematics.

For each of S1-S6, draws a boxes-and-arrows diagram of the SPHY-in-Raven
model setup: reservoirs (PONDED, TOPSOIL, FAST_RES, SLOW_RES, SURFACE_WATER)
and the processes (:Infiltration, :Flush, :Split, :Baseflow, :Percolation,
:CapillaryRise, :SoilEvaporation) connecting them.

Left column = LAND HRUs (soil-mediated water path).
Right column = MASKED_GLACIER HRUs (glacier-melt water path).
Shared subsurface (FAST_RES, SLOW_RES) drawn across the bottom.

Processes that DIFFER from S1 baseline are drawn in red (added) or
faded gray (removed).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D


# Box positions in figure-axes coords (0..10 wide, 0..10 tall canvas)
BOXES = {
    # name: (cx, cy, w, h, color, label)
    'ATMO':       (5.0, 9.4, 9.0, 0.5, '#dde6f0', 'ATMOSPHERE'),
    'PONDED_L':   (2.0, 8.0, 1.8, 0.6, '#d6eaf8', 'PONDED_WATER\n(land HRU)'),
    'PONDED_G':   (8.0, 8.0, 1.8, 0.6, '#f5d6d6', 'PONDED_WATER\n(MASKED_GLACIER)\nfrom GloGEM IRRIGATION'),
    'TOPSOIL':    (2.0, 6.3, 1.8, 0.9, '#caa472', 'TOPSOIL\n(SOIL[0])'),
    'FAST':       (5.0, 4.3, 2.0, 1.0, '#a3c4f3', 'FAST_RESERVOIR\n(SOIL[1])'),
    'SLOW':       (5.0, 2.4, 2.0, 1.0, '#76a8d8', 'SLOW_RESERVOIR\n(SOIL[2])'),
    'SURF':       (8.5, 4.3, 1.4, 0.7, '#b8e0d2', 'SURFACE_WATER\n→ outlet'),
}


def draw_box(ax, name):
    cx, cy, w, h, color, label = BOXES[name]
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle="round,pad=0.03",
                         linewidth=1.4, edgecolor='#333',
                         facecolor=color, alpha=0.85, zorder=2)
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=8.5,
            zorder=3, weight='bold' if 'FAST' in name or 'SLOW' in name else 'normal')


def draw_arrow(ax, from_box, to_box, label='', color='#222', style='-',
               offset_from=(0, 0), offset_to=(0, 0), curve=0.0,
               label_offset=(0.15, 0.15), fontsize=8, alpha=1.0):
    """Draw an arrow between two boxes. offset_* nudges the endpoints from
    the box centers; curve is the connection-style 'rad' parameter."""
    cx1, cy1, _, _, _, _ = BOXES[from_box]
    cx2, cy2, _, _, _, _ = BOXES[to_box]
    x1, y1 = cx1 + offset_from[0], cy1 + offset_from[1]
    x2, y2 = cx2 + offset_to[0], cy2 + offset_to[1]
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          connectionstyle=f'arc3,rad={curve}',
                          arrowstyle='-|>', mutation_scale=14,
                          linewidth=1.8, color=color,
                          linestyle=style, alpha=alpha, zorder=4)
    ax.add_patch(arr)
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=fontsize, color=color,
                ha='left', va='center', zorder=5,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.2))


def base_schematic(ax, struct_key):
    """Build a per-structure schematic figure."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis('off')

    # Draw all boxes
    for name in BOXES:
        draw_box(ax, name)

    # Header / structure name — pull from paper5_common single source of truth
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paper5_common import STRUCTURE_INFO as _SI
    s = _SI[struct_key]
    ax.text(5.0, 9.85,
            f"{s['two_letter']} ({s['s_code']}) — {s['description']}",
            fontsize=13, weight='bold', ha='center', va='center')
    ax.text(5.0, 8.95, f"{s['config_key']}.yaml",
            fontsize=9, ha='center', va='center', family='monospace', color='#555')

    # Color codes for "vs S1" change classification
    BLACK = '#222'   # standard process
    RED   = '#cc4422'  # NEW vs S1
    GRAY  = '#bbbbbb'  # REMOVED vs S1 (drawn faded)

    # Per-structure config flags — derive from the central registry's axis codes
    perc_opt2     = s['architecture'] == 'O'           # Overland (S5/S6/S9) uses opt2
    threshold_rel = s['architecture'] == 'T'           # Threshold (S3/S4/S8) adds BASE_THRESH_STOR
    direct_route  = s['architecture'] == 'O'           # Overland uses direct land-surface routing
    glacier_split_slow = s['connection'] == 'S'        # S2/S4/S6
    glacier_split_fast = s['connection'] == 'F'        # S7/S8/S9
    glacier_split = glacier_split_slow or glacier_split_fast

    # =============== LAND HRU PATH (left side) ===============

    # Atmosphere → PONDED_L (precip + snowmelt)
    draw_arrow(ax, 'ATMO', 'PONDED_L', label='precip + snowmelt',
               color=BLACK, offset_from=(-3, -0.25), offset_to=(0, 0.3),
               label_offset=(0.1, 0.4))

    # PONDED_L → TOPSOIL (infiltration partition, INF_HBV)
    draw_arrow(ax, 'PONDED_L', 'TOPSOIL',
               label=':Infiltration\nINF_HBV (β-power)\ninfiltration fraction',
               color=BLACK, offset_from=(0, -0.3), offset_to=(0, 0.45),
               label_offset=(0.25, -0.05))

    # TOPSOIL → ATMOSPHERE (soil evap)
    draw_arrow(ax, 'TOPSOIL', 'ATMO',
               label=':SoilEvaporation',
               color=BLACK, offset_from=(-0.6, 0.45), offset_to=(-3.5, -0.25), curve=0.3,
               label_offset=(-2.5, 0.35), fontsize=7.5)

    # PONDED_L → SURFACE_WATER (sat-excess fraction of INF_HBV partition)
    # In S5/S6 (direct routing), this is the DIRECT path to outlet;
    # in S1-S4, this SURFACE_WATER gets flushed below.
    if direct_route:
        # S5/S6: PONDED_L sat-excess → SURFACE_WATER directly
        draw_arrow(ax, 'PONDED_L', 'SURF',
                   label='sat-excess\n→ outlet\n(no :Flush)',
                   color=RED, offset_from=(0.5, -0.1), offset_to=(-0.6, 0.2),
                   curve=-0.25, label_offset=(0.5, 0.15), fontsize=8)
    else:
        # S1-S4: PONDED sat-excess → SURFACE_WATER → :Flush → FAST_RES
        # We draw the flush arrow conceptually as PONDED_L → FAST_RES through SURFACE_WATER
        draw_arrow(ax, 'PONDED_L', 'FAST',
                   label='sat-excess\n→ :Flush →\nFAST_RES',
                   color=BLACK, offset_from=(0.5, -0.1), offset_to=(-1.0, 0.45),
                   curve=-0.2, label_offset=(-0.5, 0.5), fontsize=7.5)

    # TOPSOIL → FAST (cascade percolation, opt2 only)
    if perc_opt2:
        draw_arrow(ax, 'TOPSOIL', 'FAST',
                   label=':Percolation\nPERC_POWER_LAW\nTOPSOIL→FAST  (X12, X13)',
                   color=RED, offset_from=(0.3, -0.45), offset_to=(-0.5, 0.5), curve=-0.15,
                   label_offset=(-1.5, 0.35), fontsize=7.5)

    # FAST → TOPSOIL (capillary rise, all structures)
    draw_arrow(ax, 'FAST', 'TOPSOIL',
               label=':CapillaryRise\nRISE_HBV (X15)',
               color=BLACK, offset_from=(-0.4, 0.4), offset_to=(0.3, -0.45),
               curve=-0.3, label_offset=(-1.8, -0.05), fontsize=7.5)

    # =============== GLACIER HRU PATH (right side) ===============

    # ATMOSPHERE → PONDED_G (glacier melt via IRRIGATION external)
    # Actually GloGEM melt is external — show as IRRIGATION input
    ax.annotate('GloGEM melt\n(external\nIRRIGATION)',
                xy=(8.7, 8.45), xytext=(7.0, 9.4),
                fontsize=8, ha='center', color='#cc4422',
                arrowprops=dict(arrowstyle='-|>', color='#cc4422', lw=1.6))

    # PONDED_G → SURFACE / SLOW / FAST  (glacier routing)
    if glacier_split_slow:
        # S2/S4/S6: :Split target = SLOW_RES
        draw_arrow(ax, 'PONDED_G', 'SURF',
                   label=':Split\nGlacROF·SURFACE\n(X16)',
                   color=RED, offset_from=(-0.5, -0.1), offset_to=(0.4, 0.25),
                   curve=-0.15, label_offset=(-1.8, 0.4), fontsize=7.5)
        draw_arrow(ax, 'PONDED_G', 'SLOW',
                   label=':Split\n(1−GlacROF)·SLOW_RES\n(X16) — glacier→SLOW',
                   color=RED, offset_from=(-0.6, -0.25), offset_to=(0.7, 0.45),
                   curve=-0.25, label_offset=(0.4, 1.5), fontsize=7.5)
    elif glacier_split_fast:
        # S7/S8/S9: :Split target = FAST_RES (mirror of slow variants)
        draw_arrow(ax, 'PONDED_G', 'SURF',
                   label=':Split\nGlacROF·SURFACE\n(X16)',
                   color=RED, offset_from=(-0.5, -0.1), offset_to=(0.4, 0.25),
                   curve=-0.15, label_offset=(-1.8, 0.4), fontsize=7.5)
        draw_arrow(ax, 'PONDED_G', 'FAST',
                   label=':Split\n(1−GlacROF)·FAST_RES\n(X16) — glacier→FAST',
                   color=RED, offset_from=(-0.6, -0.15), offset_to=(0.7, 0.55),
                   curve=-0.30, label_offset=(0.4, 1.3), fontsize=7.5)
    else:
        # S1/S3/S5: 100% to SURFACE_WATER (no :Split)
        draw_arrow(ax, 'PONDED_G', 'SURF',
                   label='100% glacier melt\n→ SURFACE\n(no :Split)',
                   color=BLACK, offset_from=(-0.5, -0.2), offset_to=(0.5, 0.2),
                   curve=-0.2, label_offset=(-2.0, 0.0), fontsize=7.5)

    # =============== SUBSURFACE PATH (shared) ===============

    # FAST → SLOW (percolation)
    perc_label = (':Percolation\nPERC_LINEAR (X14)' if perc_opt2
                  else ':Percolation\nPERC_CONSTANT (X11)')
    draw_arrow(ax, 'FAST', 'SLOW',
               label=perc_label,
               color=RED if perc_opt2 else BLACK,
               offset_from=(0, -0.5), offset_to=(0, 0.5),
               label_offset=(0.3, -0.1), fontsize=7.5)

    # FAST → SURFACE_WATER (BASE_LINEAR baseflow, Q1)
    draw_arrow(ax, 'FAST', 'SURF',
               label=':Baseflow\nBASE_LINEAR\nFAST→SURF (X07, Q1)',
               color=BLACK, offset_from=(0.95, 0), offset_to=(-0.55, -0.2),
               label_offset=(0.0, -0.7), fontsize=7.5)

    # FAST → SURFACE_WATER (BASE_THRESH_STOR, Q0) — S3/S4 only
    if threshold_rel:
        draw_arrow(ax, 'FAST', 'SURF',
                   label=':Baseflow\nBASE_THRESH_STOR\n(X17 UZL, X18 K0) — Q0',
                   color=RED, offset_from=(0.95, 0.35), offset_to=(-0.55, 0.2),
                   curve=-0.25, label_offset=(0.0, 0.85), fontsize=7.5)

    # SLOW → SURFACE_WATER
    draw_arrow(ax, 'SLOW', 'SURF',
               label=':Baseflow\nBASE_LINEAR\nSLOW→SURF (X08)',
               color=BLACK, offset_from=(0.95, 0.05), offset_to=(-0.4, -0.3),
               curve=0.25, label_offset=(0.0, -0.5), fontsize=7.5)

    # Legend
    legend_elems = [
        Line2D([0], [0], color=BLACK, lw=1.8, label='process active'),
        Line2D([0], [0], color=RED, lw=1.8, label='NEW vs S1 baseline'),
    ]
    ax.legend(handles=legend_elems, loc='lower left', fontsize=8.5,
              frameon=True, facecolor='white', edgecolor='gray')


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paper5_common import STRUCTURE_ORDER, STRUCTURE_INFO

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outdir', type=Path,
                        default=Path(__file__).resolve().parent.parent /
                                'plots' / 'structures',
                        help='Output directory (default: <repo>/plots/structures/)')
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Per-structure schematics — all 9
    for key in STRUCTURE_ORDER:
        s = STRUCTURE_INFO[key]
        fig, ax = plt.subplots(figsize=(13, 10))
        base_schematic(ax, key)
        fp = args.outdir / f'{key}_{s["two_letter"]}_schematic.png'
        plt.savefig(fp, dpi=130, bbox_inches='tight')
        plt.close()
        print(f'  Saved {fp}')

    # 3×3 overview — architecture × glacier-GW connection
    fig, axes = plt.subplots(3, 3, figsize=(36, 30))
    for ax, key in zip(axes.flatten(), STRUCTURE_ORDER):
        base_schematic(ax, key)
    fig.suptitle('Paper 5 — subsurface structures '
                 '(rows: Lumped / Threshold / Overland   |   cols: None / Slow / Fast)',
                 fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fp = args.outdir / 'ALL_structures_schematic.png'
    plt.savefig(fp, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fp}')


if __name__ == '__main__':
    main()
