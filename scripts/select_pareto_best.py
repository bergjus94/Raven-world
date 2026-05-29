#!/usr/bin/env python
"""Post-Pareto best-parameter selection via compromise programming.

Reads an NSGAII calibration_results_*.csv, extracts the Pareto front, applies
min-max rescaling using the Pareto-derived per-objective ranges, then picks
the parameter set with smallest Tchebycheff (L-infinity) distance to the
ideal point (1, 1, ..., 1) in rescaled space.

The Tchebycheff selection rule is "minimize worst-case shortfall on any
axis" — gives a balanced compromise without re-introducing arbitrary weights.

Usage
-----
    python scripts/select_pareto_best.py path/to/calibration_results_*.csv \\
        --write-best-params output_dir/2268_SPHY_VERIFIED_best_params.csv \\
        --write-ranges-json output_dir/pareto_ranges.json

The VERIFIED_best_params.csv format matches what spotpy_optimize.py writes
for single-objective runs, so scripts/finalize_calibration.py picks it up
unchanged for the final calibrated-parameter run.

The pareto_ranges.json captures per-objective [min, max] from the Pareto
front — feed this into the SCEUA weighted-sum rescaling once that path is
wired up.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def find_pareto_front(values: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated rows (all objectives maximized).

    Vectorized O(n^2) — fast enough for n up to ~20k.
    """
    n = len(values)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        ge = (values >= values[i]).all(axis=1)
        gt = (values > values[i]).any(axis=1)
        dominated_by = ge & gt
        dominated_by[i] = False
        if dominated_by.any():
            is_pareto[i] = False
    return is_pareto


def min_max_rescale(values: np.ndarray,
                    ranges: dict[str, tuple[float, float]],
                    obj_cols: list[str]) -> np.ndarray:
    """Rescale each column to [0, 1] using its (min, max). Degenerate columns
    (max <= min) get rescaled to 0.5 (neutral)."""
    out = np.empty_like(values, dtype=float)
    for j, col in enumerate(obj_cols):
        lo, hi = ranges[col]
        if hi > lo:
            out[:, j] = (values[:, j] - lo) / (hi - lo)
        else:
            out[:, j] = 0.5
    return out


def tchebycheff_distance(rescaled: np.ndarray,
                         ideal: np.ndarray | None = None) -> np.ndarray:
    """L-infinity distance to ideal point.

    With ideal = (1, ..., 1) in [0, 1] rescaled space, this is the maximum
    shortfall on any axis. Clamped at 0 (a point that exceeds the ideal on
    some axis doesn't get rewarded for it — only penalized for being below).
    """
    if ideal is None:
        ideal = np.ones(rescaled.shape[1])
    shortfall = np.maximum(0.0, ideal - rescaled)
    return shortfall.max(axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('results_csv', type=Path,
                        help='Path to NSGAII calibration_results_*.csv')
    parser.add_argument('--objectives', nargs='+', default=None,
                        help='Objective column names. Default: all obj_* columns.')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Print top-K candidates by Tchebycheff distance (default 5).')
    parser.add_argument('--write-best-params', type=Path, default=None,
                        help='Write selected param row in VERIFIED_best_params.csv format.')
    parser.add_argument('--write-ranges-json', type=Path, default=None,
                        help='Write per-objective Pareto-derived ranges to JSON.')
    parser.add_argument('--write-pareto-csv', type=Path, default=None,
                        help='Write the full Pareto front (one row per non-dominated point) '
                             'with rescaled obj cols and Tchebycheff distance.')
    args = parser.parse_args()

    if not args.results_csv.exists():
        sys.exit(f"Results CSV not found: {args.results_csv}")

    df = pd.read_csv(args.results_csv)

    # Auto-detect objective columns
    if args.objectives:
        obj_cols = args.objectives
        missing = [c for c in obj_cols if c not in df.columns]
        if missing:
            sys.exit(f"Objective columns not found in CSV: {missing}")
    else:
        obj_cols = [c for c in df.columns if c.startswith('obj_')]
    if not obj_cols:
        sys.exit("No obj_* columns found; specify --objectives explicitly.")

    # Filter out failed evaluations (-9999 sentinel or NaN)
    obj_vals = df[obj_cols].replace(-9999, np.nan)
    valid_mask = obj_vals.notna().all(axis=1)
    df_v = df.loc[valid_mask].reset_index(drop=True)

    print(f"Loaded {len(df)} rows, {len(df_v)} valid after filtering failed runs.")
    print(f"Objectives: {obj_cols}")

    if len(df_v) == 0:
        sys.exit("No valid rows after filtering. Cannot proceed.")

    # Pareto front
    values = df_v[obj_cols].values
    is_pareto = find_pareto_front(values)
    pareto = df_v.loc[is_pareto].reset_index(drop=True)
    print(f"Pareto front size: {len(pareto)} / {len(df_v)} non-dominated.")

    # Per-objective ranges from the Pareto front (the "achievable trade-off space")
    ranges = {c: (float(pareto[c].min()), float(pareto[c].max())) for c in obj_cols}
    print("\nPareto-derived per-objective ranges (raw scores):")
    for c, (lo, hi) in ranges.items():
        print(f"  {c}: [{lo:+.4f}, {hi:+.4f}]   span = {hi - lo:.4f}")

    # Capture original row index in df_v so we can report iteration accurately
    pareto = pareto.copy()
    pareto['_original_row'] = df_v.index[is_pareto].values

    # Rescale + Tchebycheff distance
    rescaled = min_max_rescale(pareto[obj_cols].values, ranges, obj_cols)
    distances = tchebycheff_distance(rescaled)

    # Attach rescaled cols + distance for output
    for j, c in enumerate(obj_cols):
        pareto[f'{c}_rescaled'] = rescaled[:, j]
    pareto['_tcheb_dist'] = distances
    pareto_sorted = pareto.sort_values('_tcheb_dist').reset_index(drop=True)

    # Top-K diagnostic
    print(f"\nTop-{args.top_k} compromise candidates (lowest Tchebycheff distance):")
    cols_to_show = obj_cols + [f'{c}_rescaled' for c in obj_cols] + ['_tcheb_dist']
    with pd.option_context('display.max_columns', None,
                           'display.width', 200,
                           'display.float_format', lambda x: f'{x:+.4f}'):
        print(pareto_sorted[cols_to_show].head(args.top_k).to_string(index=False))

    # Check for fragile selection: many points clustered near the best
    best_dist = pareto_sorted['_tcheb_dist'].iloc[0]
    within_1pct = (pareto_sorted['_tcheb_dist'] <= best_dist + 0.01).sum()
    within_5pct = (pareto_sorted['_tcheb_dist'] <= best_dist + 0.05).sum()
    print(f"\nFragility check: {within_1pct} candidate(s) within 0.01 of best Tchebycheff "
          f"distance, {within_5pct} within 0.05.")
    if within_1pct > 1:
        print("  ⚠ Multiple candidates near-tied — consider validation-period evaluation "
              "to break the tie.")

    selected = pareto_sorted.iloc[0]
    print(f"\nSelected (Tchebycheff-optimal) parameter set:")
    print(f"  Tchebycheff distance: {selected['_tcheb_dist']:.4f}")
    for c in obj_cols:
        rescaled_val = selected[f'{c}_rescaled']
        print(f"  {c}: raw={selected[c]:+.4f}  rescaled={rescaled_val:+.4f}")

    # Optional outputs
    if args.write_best_params:
        # Match VERIFIED_best_params.csv format used elsewhere:
        # one row of <param_cols> + 'objective' + 'iteration'.
        non_param_cols = set(obj_cols) | {'timestamp', '_tcheb_dist', '_original_row'} \
                         | {f'{c}_rescaled' for c in obj_cols}
        param_cols = [c for c in df.columns if c not in non_param_cols]
        row = {c: selected[c] for c in param_cols if c in selected.index}
        # 'objective' here is the composite score (1 − Tchebycheff distance)
        # so it's higher-is-better, matching the SCEUA convention.
        row['objective'] = float(1.0 - selected['_tcheb_dist'])
        # 'iteration' is the row's position in the original results CSV (1-indexed).
        row['iteration'] = int(selected['_original_row']) + 1

        args.write_best_params.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(args.write_best_params, index=False)
        print(f"\nWrote VERIFIED_best_params row to: {args.write_best_params}")

    if args.write_ranges_json:
        args.write_ranges_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_ranges_json, 'w') as f:
            json.dump(ranges, f, indent=2)
        print(f"Wrote per-objective ranges to: {args.write_ranges_json}")

    if args.write_pareto_csv:
        args.write_pareto_csv.parent.mkdir(parents=True, exist_ok=True)
        pareto_sorted.to_csv(args.write_pareto_csv, index=False)
        print(f"Wrote full Pareto front to: {args.write_pareto_csv}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
