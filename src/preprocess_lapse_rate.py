####
####   Derive precipitation lapse-rate parameters from gridded precipitation
####   and a catchment-clipped DEM, for use with Raven's :OroPrecipCorrect
####   OROCORR_HBV (see RavenHydroFramework/docs/model_structure_decisions.md
####   section "Precipitation lapsing — data-derived approach").
####
####   Method:
####     1. Load catchment-clipped gridded annual-mean precip
####        (TPHiPr for HMA, MeteoSwiss for Switzerland)
####     2. Sample DEM at each precip grid-cell centroid
####     3. Fit 1-breakpoint piecewise-linear regression P(z) with pwlf
####     4. Convert segment slopes (mm/yr/m) to Raven's fractional /km
####        gradients via slope / P_ref * 1000, where P_ref = fit(z_ref)
####        and z_ref is the catchment mean elevation.
####
####   Writes: data_obs_dir / 'lapse_rate.yaml'
####   Diagnostic plot: plots_dir / 'lapse_rate.png'
####
####   Consumed by preprocess_HBV.py at .rvp generation time.
####

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import rasterio
import rasterio.warp
import geopandas as gpd
import pwlf
import matplotlib.pyplot as plt

from paths import get_paths


# Map source key → (precip filename, variable name).
_SOURCE_FILES = {
    'TPHIPR':     ('tphipr_precip.nc',   'prcp'),
    'METEOSWISS': ('prec_Meteoswiss.nc', 'RhiresD'),
}

# Raven's own canonical OROCORR_HBV defaults (RavenParameters.dat:281-283).
# Used when no gridded precip source is available — keeps the regression
# optional and gives the .rvp explicit, calibratable values.
_RAVEN_DEFAULT_LAPSE = {
    'HBVEC_LAPSE_RATE':  0.08,
    'HBVEC_LAPSE_UPPER': 0.0,
    'HBVEC_LAPSE_ELEV':  5000.0,
}


def _select_source(config: dict) -> Optional[Tuple[str, str, str]]:
    """Return (source_key, filename, varname), or None if nothing configured.

    Resolution order:
      1. ``lapse_rate_precip_source`` (the explicit override).
      2. ``precip_source`` if it's TPHiPr (legacy).
      3. ``meteo_source`` if it's MeteoSwiss (legacy).
      4. None → caller falls back to Raven defaults.
    """
    explicit = (config.get('lapse_rate_precip_source') or '').upper()
    if explicit in _SOURCE_FILES:
        return (explicit,) + _SOURCE_FILES[explicit]
    if explicit:
        raise ValueError(
            f"Unknown lapse_rate_precip_source={explicit!r}. "
            f"Supported: {sorted(_SOURCE_FILES)}."
        )

    precip = (config.get('precip_source') or '').upper()
    if precip == 'TPHIPR':
        return ('TPHIPR',) + _SOURCE_FILES['TPHIPR']

    meteo = (config.get('meteo_source') or '').upper()
    if meteo in ('METEOSWISS', 'METEO_SWISS'):
        return ('METEOSWISS',) + _SOURCE_FILES['METEOSWISS']

    return None


def _annual_mean_precip(precip_nc: Path, varname: str) -> xr.DataArray:
    """Return per-cell annual-mean precipitation (mm/yr) as a 2-D DataArray.

    Input file holds daily values in mm/day. Averaging over time and
    multiplying by 365.25 gives the annual mean — equivalent to mean over
    annual totals for files spanning whole years, and tolerant of partial
    years at the edges.
    """
    ds = xr.open_dataset(precip_nc)
    if varname not in ds.data_vars:
        raise KeyError(
            f"Variable '{varname}' not in {precip_nc.name}. "
            f"Available: {list(ds.data_vars)}"
        )
    da = ds[varname]
    # Mean daily precip across all time steps, scaled to mm/yr.
    annual = da.mean(dim='time', skipna=True) * 365.25
    annual.attrs['units'] = 'mm/yr'
    annual.attrs['long_name'] = 'annual-mean precipitation'
    return annual


def _sample_dem_at_cells(
    annual_da: xr.DataArray, dem_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """For each non-NaN precip grid cell, return (elevation_m, precip_mm_yr).

    DEM is read in its native CRS and re-sampled (nearest) at each precip
    cell centroid via rasterio.warp to handle differing resolutions/CRSs.
    """
    # Identify lat/lon coordinate names — TPHiPr uses 1-D lon/lat,
    # MeteoSwiss writes 2-D lat/lon but indexed by E/N — we handle both.
    lat_name = 'lat' if 'lat' in annual_da.coords else 'latitude'
    lon_name = 'lon' if 'lon' in annual_da.coords else 'longitude'
    lat = annual_da[lat_name].values
    lon = annual_da[lon_name].values

    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lon2d, lat2d = lon, lat

    precip = annual_da.values
    if precip.shape != lon2d.shape:
        # xarray may carry dims in (lat, lon) or (lon, lat) order.
        precip = precip.T
        if precip.shape != lon2d.shape:
            raise ValueError(
                f"Precip shape {annual_da.shape} incompatible with "
                f"coord grid {lon2d.shape}"
            )

    valid = np.isfinite(precip)
    lons_flat = lon2d[valid]
    lats_flat = lat2d[valid]
    precips = precip[valid]

    with rasterio.open(dem_path) as src:
        # Project lat/lon (WGS84) into DEM CRS, then sample.
        xs, ys = rasterio.warp.transform(
            'EPSG:4326', src.crs, lons_flat.tolist(), lats_flat.tolist()
        )
        samples = np.fromiter(
            (v[0] for v in src.sample(zip(xs, ys))),
            dtype=float, count=len(xs),
        )

    # Drop cells where DEM was nodata.
    good = np.isfinite(samples) & (samples > -1e4) & (samples < 1e5)
    return samples[good], precips[good]


def _linear_fallback(
    z: np.ndarray, p: np.ndarray, min_points_per_segment: int,
) -> Tuple[float, float, float, float, float, bool, int, float, float]:
    """Return a 1-segment linear fit packaged in the same tuple shape as the
    segmented case (slopes equal, breakpoint above catchment top). Used both
    as a small-data fallback and as the BIC-preferred model.
    """
    slope, intercept = np.polyfit(z, p, 1)
    z_ref = float(np.mean(z))
    p_ref = float(slope * z_ref + intercept)
    z_break = float(np.max(z)) + 1.0
    sse = float(np.sum((p - (slope * z + intercept)) ** 2))
    n = len(z)
    bic = n * float(np.log(sse / n)) + 2.0 * float(np.log(n))
    return (slope, slope, z_break, p_ref, z_ref, False,
            min_points_per_segment, sse, bic)


def _fit_one_breakpoint(
    elev: np.ndarray,
    precip: np.ndarray,
    min_points_per_segment: Optional[int] = None,
) -> Tuple[float, float, float, float, float, bool, int, float, float]:
    """Fit P(z) and pick 1 vs 2 segments via BIC.

    Always fits a 1-segment linear model. If at least
    ``2 * min_points_per_segment`` points are available, also fits a
    2-segment piecewise model (breakpoint constrained so each segment
    keeps ≥ ``min_points_per_segment`` cells). Returns whichever model
    has lower BIC.

    BIC = n·ln(SSE/n) + k·ln(n), with k=2 for the linear model and k=4
    for the segmented model (intercept, lower slope, upper slope,
    breakpoint location). The 2·ln(n) penalty difference is the minimum
    SSE reduction the segmented model must achieve to be preferred.

    If ``min_points_per_segment`` is None, uses the adaptive rule
    ``max(15, n // 10)``.

    Returns (slope_lower, slope_upper, breakpoint_m, p_ref, z_ref,
    used_breakpoint, min_points_resolved, bic_linear, bic_segmented).
    ``bic_segmented`` is np.nan when the 2-segment fit could not be tried.
    """
    order = np.argsort(elev)
    z = elev[order]
    p = precip[order]
    n = len(z)

    if min_points_per_segment is None:
        min_points_per_segment = max(15, n // 10)

    # 1-segment fit always computed.
    (slope1, _slope1_dup, z_break1, p_ref1, z_ref1, _used1,
     k_resolved, sse_lin, bic_lin) = _linear_fallback(
        z, p, min_points_per_segment
    )

    if n < 2 * min_points_per_segment:
        # No segmented option — linear is the only choice.
        return (slope1, slope1, z_break1, p_ref1, z_ref1, False,
                k_resolved, bic_lin, float('nan'))

    # Constrain breakpoint to keep ≥ k cells per side.
    k = min_points_per_segment
    b_lo = float(z[k - 1])
    b_hi = float(z[-k])
    if b_hi <= b_lo:
        # Degenerate (many ties in elevation) — linear only.
        return (slope1, slope1, z_break1, p_ref1, z_ref1, False,
                k_resolved, bic_lin, float('nan'))

    fit = pwlf.PiecewiseLinFit(z, p)
    fit.fit(2, bounds=[(b_lo, b_hi)])
    pred_seg = fit.predict(z)
    sse_seg = float(np.sum((p - pred_seg) ** 2))
    bic_seg = n * float(np.log(sse_seg / n)) + 4.0 * float(np.log(n))

    if bic_seg >= bic_lin:
        # Segmented does not justify its extra parameters — keep linear.
        return (slope1, slope1, z_break1, p_ref1, z_ref1, False,
                k_resolved, bic_lin, bic_seg)

    # Segmented wins — use its slopes and breakpoint.
    z_break = float(fit.fit_breaks[1])
    slope_lower = float(fit.slopes[0])
    slope_upper = float(fit.slopes[1])
    z_ref = float(np.mean(z))
    p_ref = float(fit.predict(np.array([z_ref]))[0])

    return (slope_lower, slope_upper, z_break, p_ref, z_ref, True,
            k_resolved, bic_lin, bic_seg)


def _make_diagnostic_plot(
    elev: np.ndarray,
    precip: np.ndarray,
    slope_lower: float,
    slope_upper: float,
    z_break: float,
    p_ref: float,
    z_ref: float,
    out_path: Path,
    title: str,
) -> None:
    """Save a scatter + fitted piecewise line for inspection."""
    z_min, z_max = float(np.min(elev)), float(np.max(elev))
    z_grid = np.array([z_min, z_break, z_max])
    # P at z_ref is p_ref; build the piecewise line from there. Step from
    # z_ref to z_break using whichever slope applies at z_ref.
    if z_ref <= z_break:
        p_break = p_ref + slope_lower * (z_break - z_ref)
    else:
        p_break = p_ref + slope_upper * (z_break - z_ref)
    p_min = p_break + slope_lower * (z_min - z_break)
    p_max = p_break + slope_upper * (z_max - z_break)
    p_line = np.array([p_min, p_break, p_max])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(elev, precip, s=6, alpha=0.3, color='steelblue',
               label=f'grid cells (n={len(elev)})')
    ax.plot(z_grid, p_line, color='firebrick', lw=2,
            label='piecewise linear fit')
    ax.axvline(z_break, color='grey', ls='--', alpha=0.6,
               label=f'breakpoint = {z_break:.0f} m')
    ax.set_xlabel('Elevation (m)')
    ax.set_ylabel('Annual mean precipitation (mm/yr)')
    ax.set_title(title)
    ax.legend(loc='best', frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def derive_lapse_rates(
    namelist_path: Union[str, Path],
    force_reprocess: bool = False,
    min_points_per_segment: Optional[int] = None,
) -> Dict[str, float]:
    """Compute and persist data-derived HBVEC lapse-rate parameters.

    Returns a dict with the three Raven parameters and provenance fields.
    """
    namelist_path = Path(namelist_path)
    with open(namelist_path, 'r') as f:
        config = yaml.safe_load(f)

    paths = get_paths(config)
    data_obs_dir: Path = paths['data_obs_dir']
    topo_shared_dir: Path = paths['topo_shared_dir']
    plots_dir: Path = paths['plots_dir']
    gauge_id = config.get('gauge_id')

    out_yaml = data_obs_dir / 'lapse_rate.yaml'
    if out_yaml.exists() and not force_reprocess:
        logging.info(f"✅ lapse_rate.yaml already exists – skipping ({out_yaml})")
        with open(out_yaml, 'r') as f:
            return yaml.safe_load(f)

    source = _select_source(config)
    if source is None:
        logging.info(
            f"No gridded precip configured for lapse-rate derivation "
            f"(gauge {gauge_id}). Writing Raven defaults to {out_yaml.name}."
        )
        result = {
            'gauge_id': gauge_id,
            'source': 'raven_default',
            'fit_kind': 'raven_default',
            'n_cells': 0,
            **_RAVEN_DEFAULT_LAPSE,
        }
        data_obs_dir.mkdir(parents=True, exist_ok=True)
        with open(out_yaml, 'w') as f:
            yaml.safe_dump(result, f, sort_keys=False)
        logging.info(
            f"  HBVEC_LAPSE_RATE  = {result['HBVEC_LAPSE_RATE']} /km (Raven default)"
        )
        logging.info(
            f"  HBVEC_LAPSE_UPPER = {result['HBVEC_LAPSE_UPPER']} /km (Raven default)"
        )
        logging.info(
            f"  HBVEC_LAPSE_ELEV  = {result['HBVEC_LAPSE_ELEV']:.0f} m (Raven default)"
        )
        logging.info(f"  Wrote {out_yaml}")
        return result

    source_key, filename, varname = source
    precip_nc = data_obs_dir / filename

    # DEM: prefer topo_shared_dir, but preprocess_catchment_hru currently
    # writes per-variant copies under topo_shared_dir/<variant>/clipped_dem.tif.
    # The terrain is the same regardless of HRU discretization, so any copy
    # works for the regression.
    dem_path = topo_shared_dir / 'clipped_dem.tif'
    if not dem_path.exists():
        for sub in sorted(topo_shared_dir.glob('*/clipped_dem.tif')):
            dem_path = sub
            break

    if not precip_nc.exists():
        raise FileNotFoundError(
            f"Precip file not found: {precip_nc}. Run the matching meteo "
            f"preprocessor first ({source_key} branch of preprocess_meteo.py)."
        )
    if not dem_path.exists():
        raise FileNotFoundError(
            f"Clipped DEM not found under {topo_shared_dir}/ "
            f"(searched topo_shared_dir and topo_shared_dir/*/). "
            f"Run preprocess_catchment_hru to produce it."
        )

    logging.info(f"Deriving precipitation lapse rates for gauge {gauge_id}")
    logging.info(f"  Source: {source_key} ({precip_nc.name}, var '{varname}')")
    logging.info(f"  DEM:    {dem_path.name}")

    annual = _annual_mean_precip(precip_nc, varname)
    elev, precip = _sample_dem_at_cells(annual, dem_path)
    logging.info(f"  Sampled {len(elev)} valid (elev, precip) pairs")

    (slope_lower, slope_upper, z_break, p_ref, z_ref,
     used_breakpoint, min_points_resolved,
     bic_linear, bic_segmented) = _fit_one_breakpoint(
        elev, precip, min_points_per_segment
    )
    logging.info(
        f"  min_points_per_segment = {min_points_resolved} "
        f"({'adaptive: max(15, n//10)' if min_points_per_segment is None else 'user override'})"
    )
    if np.isnan(bic_segmented):
        logging.info(
            f"  Linear fit (n={len(elev)} < 2 × {min_points_resolved} "
            f"required for breakpoint). BIC_linear={bic_linear:.1f}"
        )
    else:
        delta = bic_segmented - bic_linear
        winner = 'segmented' if used_breakpoint else 'linear'
        logging.info(
            f"  Model selection (BIC): linear={bic_linear:.1f}, "
            f"segmented={bic_segmented:.1f}, Δ={delta:+.1f} → {winner}"
        )

    # Convert mm/yr per m → fractional change per km
    # F.precip *= 1 + (rate/1000) * (z - z_ref)   (OrographicCorrections.cpp:243)
    # ⇒ dP/dz = P_ref * (rate / 1000)
    # ⇒ rate  = (dP/dz) / P_ref * 1000   (per km)
    if p_ref <= 0:
        raise ValueError(
            f"Fitted P_ref={p_ref:.2f} mm/yr at z_ref={z_ref:.0f} m is "
            f"non-positive — cannot normalize to fractional gradient."
        )
    rate_per_km = slope_lower / p_ref * 1000.0
    upper_per_km = slope_upper / p_ref * 1000.0

    result = {
        'gauge_id': gauge_id,
        'source': source_key,
        'precip_file': str(precip_nc),
        'dem_file': str(dem_path),
        'n_cells': int(len(elev)),
        'min_points_per_segment': int(min_points_resolved),
        'fit_kind': 'segmented' if used_breakpoint else 'linear',
        'bic_linear': float(bic_linear),
        'bic_segmented': (float(bic_segmented)
                          if not np.isnan(bic_segmented) else None),
        'bic_delta': (float(bic_segmented - bic_linear)
                      if not np.isnan(bic_segmented) else None),
        'z_ref_m': float(z_ref),
        'p_ref_mm_yr': float(p_ref),
        'slope_lower_mm_yr_per_m': float(slope_lower),
        'slope_upper_mm_yr_per_m': float(slope_upper),
        # Raven OROCORR_HBV parameters (fractional /km for rates, m for elev)
        'HBVEC_LAPSE_RATE':  float(rate_per_km),
        'HBVEC_LAPSE_UPPER': float(upper_per_km),
        'HBVEC_LAPSE_ELEV':  float(z_break),
    }

    logging.info(
        f"  HBVEC_LAPSE_RATE  = {rate_per_km:+.4f} /km "
        f"(slope {slope_lower:+.3f} mm/yr/m, P_ref {p_ref:.0f} mm/yr)"
    )
    logging.info(
        f"  HBVEC_LAPSE_UPPER = {upper_per_km:+.4f} /km "
        f"(slope {slope_upper:+.3f} mm/yr/m)"
    )
    logging.info(f"  HBVEC_LAPSE_ELEV  = {z_break:.0f} m")

    data_obs_dir.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, 'w') as f:
        yaml.safe_dump(result, f, sort_keys=False)
    logging.info(f"  Wrote {out_yaml}")

    plot_path = plots_dir / 'lapse_rate.png'
    _make_diagnostic_plot(
        elev, precip,
        slope_lower, slope_upper, z_break, p_ref, z_ref,
        plot_path,
        title=f"Gauge {gauge_id} — {source_key} precip vs DEM elevation",
    )
    logging.info(f"  Wrote diagnostic plot {plot_path}")

    return result


def load_lapse_rates(config: dict) -> Optional[Dict[str, float]]:
    """Load previously derived lapse-rate YAML, or None if absent.

    Helper for preprocess_HBV.py: returns the three HBVEC_* parameters
    when the YAML exists, otherwise None (caller falls back to defaults).
    """
    paths = get_paths(config)
    out_yaml: Path = paths['data_obs_dir'] / 'lapse_rate.yaml'
    if not out_yaml.exists():
        return None
    with open(out_yaml, 'r') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Derive HBVEC precipitation lapse-rate parameters from '
                    'gridded precip + DEM for a single catchment.'
    )
    parser.add_argument('namelist', type=str, help='Path to catchment namelist YAML')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if lapse_rate.yaml already exists')
    parser.add_argument('--min-points', type=int, default=None,
                        help='Minimum grid cells required in each segment. '
                             'Default: adaptive max(15, n // 10). Below '
                             '2×this, falls back to a single linear fit.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    derive_lapse_rates(
        args.namelist,
        force_reprocess=args.force,
        min_points_per_segment=args.min_points,
    )
