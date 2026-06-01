#!/usr/bin/env python3
"""Paper 5 — Huang-style baseflow-filter comparison on observed discharge.

For each catchment, applies multiple baseflow separation methods to the
observed daily Q time series, then compares each method's winter-period
baseflow against raw winter Q (= the candidate "raw_winter" calibration
target). Quantifies the assumption "winter Q is essentially baseflow in
cold high-mountain catchments" before we commit to using raw_winter as
our calibration target.

Methods compared:
  - Raw winter Q (DJFM)   ← the candidate
  - Eckhardt digital filter
  - Lyne-Hollick digital filter
  - Sliding minimum (local-minimum-based)

Outputs per catchment:
  - Time series plot (full + winter zoom) with all baseflow estimates overlaid
  - Bar table of per-filter ratios: filter_winter_BF / raw_winter_Q
  - BFI (baseflow index) per filter
And a cross-catchment summary heatmap.
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

# Reuse the existing separators
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from baseflow_separation import BaseflowSeparator  # noqa: E402


CATCHMENTS = {
    '2268': 'Rhone @ Gletsch',
    '2256': 'Rosegbach @ Pontresina',
    '2161': 'Massa @ Aletsch',
    '0102': 'Hunza @ Dainyor',
}

# Window definitions for "winter"
WINTER_NDJFM = [11, 12, 1, 2, 3]
WINTER_DJFM = [12, 1, 2, 3]
WINTER_DEEP_DJF = [12, 1, 2]


def load_q_daily(rvt_path: Path) -> pd.Series:
    """Parse a Raven Q_daily.rvt format file into a pandas Series."""
    with open(rvt_path) as f:
        lines = f.readlines()

    # First line is :ObservationData header
    # Second line: start_date start_time interval nrecords
    header = lines[1].split()
    start_date = pd.to_datetime(header[0])
    nrec = int(header[3])

    values = []
    for line in lines[2:]:
        line = line.strip()
        if not line or line.startswith(':') or line.startswith('#'):
            continue
        try:
            v = float(line.split()[0])
            # Raven Q_daily.rvt files use -1.2345 (or similar negative) as the
            # missing-data sentinel. Q is always positive, so flag anything < 0.
            values.append(v if v >= 0 else np.nan)
        except (ValueError, IndexError):
            continue
        if len(values) >= nrec:
            break

    dates = pd.date_range(start_date, periods=len(values), freq='D')
    s = pd.Series(values, index=dates, name='Q_obs')
    return s.dropna()


def apply_filters(q: pd.Series, eckhardt_bfi_max: float = 0.95) -> dict:
    """Apply all available baseflow separation methods to the Q series.

    For Eckhardt, BFI_max defaults to 0.95 — appropriate for cold high-
    mountain catchments where winter Q is essentially all baseflow. The
    library default of 0.50 (ephemeral streams with porous aquifers,
    Eckhardt 2005) structurally caps the baseflow estimate at 50% of total
    Q, which is inappropriate for our regime. See plot title for the value
    actually used.

    Returns dict of filter_name → baseflow Series.
    """
    sep = BaseflowSeparator(q)
    out = {
        f'Eckhardt (BFI_max={eckhardt_bfi_max})': sep.eckhardt(BFI_max=eckhardt_bfi_max),
        'Lyne-Hollick':   sep.lyne_hollick(),
        'Sliding-Min':    sep.sliding_minimum(window_days=5),
    }
    # "Raw winter": Q itself (no filtering), but reported only during winter window
    out['Raw winter Q'] = q.copy()
    return out


def compute_winter_stats(q: pd.Series, baseflow: dict, window: list[int]) -> pd.DataFrame:
    """For each filter, compute winter-mean baseflow + ratio vs raw winter Q + annual BFI."""
    q_winter = q[q.index.month.isin(window)]
    raw_winter_mean = q_winter.mean()

    rows = []
    for name, bf in baseflow.items():
        bf_winter = bf[bf.index.month.isin(window)]
        bf_winter_mean = bf_winter.mean()
        ratio = bf_winter_mean / raw_winter_mean if raw_winter_mean > 0 else np.nan

        # Annual BFI = sum(baseflow) / sum(total Q)
        # Align indices
        common = q.index.intersection(bf.index)
        bfi = bf.loc[common].sum() / q.loc[common].sum() if q.loc[common].sum() > 0 else np.nan
        rows.append({'method': name,
                     'winter_BF_mean_m3s': bf_winter_mean,
                     'ratio_to_raw_winter_Q': ratio,
                     'annual_BFI': bfi})
    return pd.DataFrame(rows)


def plot_catchment_comparison(catch: str, q: pd.Series, baseflow: dict,
                              winter_window: list[int], outdir: Path):
    """Plot time series + winter zoom + ratio table for one catchment."""
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 0.8], hspace=0.4, wspace=0.2)
    ax_full = fig.add_subplot(gs[0, :])
    ax_zoom = fig.add_subplot(gs[1, :])
    ax_stats = fig.add_subplot(gs[2, 0]); ax_stats.axis('off')
    ax_bfi = fig.add_subplot(gs[2, 1])

    # Identify the Eckhardt key (it includes the BFI_max value)
    eckhardt_key = next(k for k in baseflow if k.startswith('Eckhardt'))
    colors = {'Raw winter Q': '#222', eckhardt_key: '#1f77b4', 'Lyne-Hollick': '#ff7f0e',
              'Sliding-Min': '#2ca02c'}

    # Top: full time series
    ax_full.plot(q.index, q.values, color='#999', lw=0.5, alpha=0.7,
                 label='Observed Q (daily)')
    for name, bf in baseflow.items():
        if name == 'Raw winter Q':
            # show Q only during winter
            q_w = q[q.index.month.isin(winter_window)]
            ax_full.plot(q_w.index, q_w.values, color=colors[name], lw=0.8, alpha=0.8,
                         label=f'{name} (DJFM Q)')
        else:
            ax_full.plot(bf.index, bf.values, color=colors[name], lw=0.8,
                         label=name)
    ax_full.set_ylabel('Q  [m³/s]')
    ax_full.set_title(f'{catch} {CATCHMENTS[catch]} — observed Q with baseflow separation')
    ax_full.legend(fontsize=9, loc='upper right', ncol=5)
    ax_full.grid(alpha=0.3)
    ax_full.set_yscale('log')

    # Middle: zoom into one representative winter (use last 4 full winters)
    n_years = (q.index.max().year - q.index.min().year)
    zoom_end_year = q.index.max().year - 1
    zoom_start = pd.to_datetime(f'{zoom_end_year - 3}-09-01')
    zoom_end   = pd.to_datetime(f'{zoom_end_year + 1}-04-30')
    q_zoom = q.loc[zoom_start:zoom_end]
    ax_zoom.plot(q_zoom.index, q_zoom.values, color='#999', lw=1.0, alpha=0.7,
                 label='Observed Q')
    for name, bf in baseflow.items():
        if name == 'Raw winter Q':
            q_w = q_zoom[q_zoom.index.month.isin(winter_window)]
            ax_zoom.scatter(q_w.index, q_w.values, color=colors[name], s=10,
                            label=f'{name} (DJFM)', zorder=4)
        else:
            bf_zoom = bf.loc[zoom_start:zoom_end]
            ax_zoom.plot(bf_zoom.index, bf_zoom.values, color=colors[name], lw=1.5,
                         label=name)
    # Highlight winter months
    for yr in range(zoom_start.year, zoom_end.year + 1):
        ax_zoom.axvspan(pd.Timestamp(f'{yr}-12-01'),
                        pd.Timestamp(f'{yr + 1}-04-01'),
                        color='#deebf7', alpha=0.4, zorder=0)
    ax_zoom.set_ylabel('Q  [m³/s]')
    ax_zoom.set_title('Zoom: 4 recent winters (DJFM highlighted in light blue)')
    ax_zoom.legend(fontsize=9, loc='upper right', ncol=5)
    ax_zoom.grid(alpha=0.3)

    # Bottom-left: stats table
    stats = compute_winter_stats(q, baseflow, winter_window)
    table_rows = [[r.method, f'{r.winter_BF_mean_m3s:.3g}',
                   f'{r.ratio_to_raw_winter_Q:.3f}',
                   f'{r.annual_BFI:.2f}']
                  for r in stats.itertuples()]
    tbl = ax_stats.table(cellText=table_rows,
                         colLabels=['Method', 'Winter BF mean\n[m³/s]',
                                    'Ratio to\nraw winter Q', 'Annual BFI'],
                         cellLoc='center', loc='center',
                         colWidths=[0.28, 0.22, 0.25, 0.20])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 2.0)
    for i in range(4):
        tbl[0, i].set_facecolor('#dfe9f3')
        tbl[0, i].set_text_props(weight='bold')

    # Bottom-right: BFI bar
    methods = stats.method.values
    bfis = stats.annual_BFI.values
    colors_bar = [colors[m] for m in methods]
    ax_bfi.bar(methods, bfis, color=colors_bar, alpha=0.75)
    ax_bfi.set_ylabel('Annual BFI')
    ax_bfi.set_title('Baseflow Index per filter')
    ax_bfi.grid(axis='y', alpha=0.3)
    ax_bfi.set_xticklabels(methods, rotation=20, ha='right', fontsize=9)
    ax_bfi.set_ylim(0, 1.0)

    fp = outdir / f'baseflow_filter_comparison_{catch}.png'
    plt.savefig(fp, dpi=130, bbox_inches='tight')
    plt.close()
    return stats, fp


def plot_cross_catchment_summary(all_stats: dict, outdir: Path):
    """Heatmap: rows = catchments, cols = filter methods, color = ratio to raw winter Q."""
    # Take the method names from the first catchment's stats
    methods = list(next(iter(all_stats.values())).method.values)
    matrix_ratio = np.zeros((len(CATCHMENTS), len(methods)))
    matrix_bfi = np.zeros((len(CATCHMENTS), len(methods)))
    catch_labels = []
    for ci, c in enumerate(CATCHMENTS.keys()):
        s = all_stats[c].set_index('method')
        catch_labels.append(f'{c}\n{CATCHMENTS[c]}')
        for mj, m in enumerate(methods):
            matrix_ratio[ci, mj] = s.loc[m, 'ratio_to_raw_winter_Q']
            matrix_bfi[ci, mj] = s.loc[m, 'annual_BFI']

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, mat, title, vmin, vmax, cmap in zip(
            axes,
            [matrix_ratio, matrix_bfi],
            ['Winter BF ratio to raw winter Q\n(values near 1.0 → raw_winter is justified)',
             'Annual BFI per filter\n(rough estimate of baseflow fraction of annual Q)'],
            [0.0, 0.0], [1.5, 1.0],
            ['RdYlGn', 'viridis']):
        im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.set_yticks(np.arange(len(catch_labels)))
        ax.set_yticklabels(catch_labels)
        ax.set_title(title)
        for i in range(len(catch_labels)):
            for j in range(len(methods)):
                ax.text(j, i, f'{mat[i, j]:.2f}',
                        ha='center', va='center', fontsize=10,
                        color='black' if 0.3 < mat[i, j] < 1.3 else 'white')
        plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle('Cross-catchment baseflow-filter comparison\n'
                 '(Huang-style validation of raw_winter assumption)', fontsize=12)
    fp = outdir / 'baseflow_filter_cross_catchment.png'
    plt.tight_layout()
    plt.savefig(fp, dpi=130, bbox_inches='tight')
    plt.close()
    return fp


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outdir', type=Path,
                        default=Path('/tmp/cross_catchment_plots/baseflow_filters'))
    parser.add_argument('--q-dir', type=Path,
                        default=Path('/tmp/swiss_pareto'))
    parser.add_argument('--window', choices=['DJFM', 'NDJFM', 'DJF'], default='DJFM',
                        help='Winter window definition')
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    window = {'DJFM': WINTER_DJFM, 'NDJFM': WINTER_NDJFM, 'DJF': WINTER_DEEP_DJF}[args.window]
    print(f'Using winter window: {args.window}  → months {window}\n')

    all_stats = {}
    for catch, name in CATCHMENTS.items():
        rvt = args.q_dir / f'Q_daily_{catch}.rvt'
        if not rvt.exists():
            print(f'  ⚠ {catch}: Q file missing at {rvt}, skipping')
            continue
        print(f'=== {catch} ({name}) ===')
        try:
            q = load_q_daily(rvt)
            print(f'  Q records: {len(q)}  [{q.index.min().date()} to {q.index.max().date()}]')
        except Exception as e:
            print(f'  ⚠ failed to load Q: {e}')
            continue
        try:
            baseflow = apply_filters(q)
        except Exception as e:
            print(f'  ⚠ filter computation failed: {e}')
            continue
        stats, fp = plot_catchment_comparison(catch, q, baseflow, window, args.outdir)
        all_stats[catch] = stats
        print('  ratios:', dict(zip(stats.method, stats.ratio_to_raw_winter_Q.round(3))))
        print(f'  saved {fp}')

    if len(all_stats) >= 2:
        fp = plot_cross_catchment_summary(all_stats, args.outdir)
        print(f'\nCross-catchment summary: {fp}')

    # BFI_max sensitivity — separate plot
    print('\nBFI_max sensitivity for Eckhardt filter:')
    fig, ax = plt.subplots(figsize=(10, 6))
    bfi_vals = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
    catch_colors = {'2268': '#1f77b4', '2256': '#ff7f0e',
                    '2161': '#2ca02c', '0102': '#d62728'}
    for c, name in CATCHMENTS.items():
        rvt = args.q_dir / f'Q_daily_{c}.rvt'
        if not rvt.exists():
            continue
        q = load_q_daily(rvt)
        sep = BaseflowSeparator(q)
        q_winter = q[q.index.month.isin(window)]
        raw_mean = q_winter.mean()
        ratios = []
        for bm in bfi_vals:
            bf = sep.eckhardt(BFI_max=bm)
            bf_winter = bf[bf.index.month.isin(window)]
            ratios.append(bf_winter.mean() / raw_mean)
        ax.plot(bfi_vals, ratios, marker='o', label=f'{c} {name}', color=catch_colors[c], lw=2)
    ax.axhline(1.0, color='black', ls='--', alpha=0.5, label='Raw winter Q (= 1.0)')
    ax.axvline(0.50, color='red', ls=':', alpha=0.5, label='Library default (0.50)')
    ax.axvline(0.95, color='green', ls=':', alpha=0.5, label='Recommended for cold catchments (0.95)')
    ax.set_xlabel('Eckhardt BFI_max parameter')
    ax.set_ylabel('Eckhardt winter baseflow / raw winter Q')
    ax.set_title('Eckhardt-filter sensitivity to BFI_max\n'
                 '(default 0.50 caps baseflow at 50% — inappropriate for cold high-mountain catchments)')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    fp = args.outdir / 'baseflow_eckhardt_bfimax_sensitivity.png'
    plt.savefig(fp, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fp}')

    # Combined CSV
    rows = []
    for c, stats in all_stats.items():
        for r in stats.itertuples():
            rows.append({'catchment': c, **{k: getattr(r, k) for k in stats.columns}})
    pd.DataFrame(rows).to_csv(args.outdir / 'baseflow_filter_stats.csv', index=False)
    print(f"Wrote {args.outdir / 'baseflow_filter_stats.csv'}")


if __name__ == '__main__':
    main()
