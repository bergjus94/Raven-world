"""
Annual water balance components for Raven catchments.

Computes for each catchment x configuration combination:
  - Static HRU attributes (area, elevation stats, land use fractions)
  - Annual observed streamflow Q (mm/year)
  - Annual precipitation P (mm/year) via area-weighted grid interpolation
  - Annual PET (mm/year) via area-weighted grid interpolation
  - Annual mean temperature (degC)
  - GloGEM glacier components (coupled configs only): icemelt, snowmelt, rain
  - Water balance residual: P - Q - PET

Outputs:
  - Per-catchment annual CSV
  - Multi-catchment summary CSV (multi-year averages)
  - Water balance bar charts per catchment x config
  - Multi-config comparison plot per catchment

Usage:
    python scripts/water_balance.py
    python scripts/water_balance.py --catchments 0101 0122
    python scripts/water_balance.py --env server -o outputs/water_balance
"""

import sys
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Add src to path for config imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from paths import get_paths, get_topo_variant
from config_merge import load_config, list_catchments, load_configurations_registry

MISSING_Q = -1.2345
GLACIER_LU_CLASSES = {'GLACIER', 'MASKED_GLACIER'}
SECONDS_PER_DAY = 86400
DAYS_PER_YEAR = 365.25
MIN_DATA_COVERAGE = 0.95


# =============================================================================
# Parsing helpers
# =============================================================================

def parse_hru_table(hru_path):
    """Load HRU_table.csv and return DataFrame with cleaned column names.

    Returns:
        DataFrame with columns: HRU_ID, AREA, ELEVATION, LATITUDE, LONGITUDE,
        BASIN_ID, LAND_USE_CLASS, VEG_CLASS, SOIL_PROFILE, SLOPE, ASPECT,
        GLACIER_SIZE, ...
    """
    df = pd.read_csv(hru_path)
    # First column is ':ATTRIBUTES' which is the HRU ID
    df = df.rename(columns={df.columns[0]: 'HRU_ID'})
    return df


def parse_grid_weights(gw_path):
    """Parse Raven GridWeights.txt file.

    Returns:
        DataFrame with columns: HRU_ID, Cell_ID, Weight
    """
    records = []
    in_data = False
    with open(gw_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(':EndGridWeights'):
                break
            if stripped.startswith(':NumberGridCells'):
                continue
            if stripped.startswith(':NumberHRUs'):
                continue
            if stripped.startswith(':GridWeights'):
                in_data = True
                continue
            if not in_data:
                continue
            if stripped.startswith('#') or stripped == '':
                continue
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    records.append({
                        'HRU_ID': int(parts[0]),
                        'Cell_ID': int(parts[1]),
                        'Weight': float(parts[2]),
                    })
                except ValueError:
                    continue
    return pd.DataFrame(records)


def parse_q_daily(rvt_path):
    """Parse Raven Q_daily.rvt observation file.

    Returns:
        pandas Series with DatetimeIndex and streamflow in m3/s.
        Missing values (== -1.2345) are set to NaN.
    """
    with open(rvt_path) as f:
        lines = f.readlines()

    # First line is header: :ObservationData HYDROGRAPH 1 m3/s
    # Second line is: start_date start_time timestep_days n_values
    header_line = lines[1].strip().split()
    start_date = pd.Timestamp(f"{header_line[0]} {header_line[1]}")
    n_values = int(header_line[3])

    values = []
    for line in lines[2:2 + n_values]:
        val = float(line.strip())
        values.append(val)

    dates = pd.date_range(start=start_date, periods=len(values), freq='D')
    series = pd.Series(values, index=dates, name='Q_m3s')
    series = series.replace(MISSING_Q, np.nan)
    return series


# =============================================================================
# Static catchment attributes
# =============================================================================

def compute_static_attributes(hru_df):
    """Compute static catchment attributes from HRU table.

    Returns:
        dict with area, elevation, land use statistics.
    """
    total_area = hru_df['AREA'].sum()
    glacier_mask = hru_df['LAND_USE_CLASS'].isin(GLACIER_LU_CLASSES)
    glacier_area = hru_df.loc[glacier_mask, 'AREA'].sum()
    glacier_free_area = total_area - glacier_area

    n_hrus = len(hru_df)
    elev_bands = hru_df['ELEVATION'].apply(lambda e: int(e // 100) * 100).nunique()

    # Area-weighted mean elevation
    mean_elev = np.average(hru_df['ELEVATION'], weights=hru_df['AREA'])

    # Land use fractions
    lu_fracs = (
        hru_df.groupby('LAND_USE_CLASS')['AREA'].sum() / total_area * 100
    ).to_dict()

    return {
        'total_area_km2': total_area,
        'glacier_area_km2': glacier_area,
        'glacier_free_area_km2': glacier_free_area,
        'glacier_fraction_pct': glacier_area / total_area * 100 if total_area > 0 else 0,
        'n_hrus': n_hrus,
        'n_elevation_bands': elev_bands,
        'min_elevation_m': hru_df['ELEVATION'].min(),
        'max_elevation_m': hru_df['ELEVATION'].max(),
        'mean_elevation_m': mean_elev,
        'landuse_fractions': lu_fracs,
    }


# =============================================================================
# Spatial averaging with grid weights
# =============================================================================

def compute_area_weighted_timeseries(nc_path, var_name, gw_df, hru_df,
                                     glacier_free_only=False):
    """Compute area-weighted spatial average from NetCDF using grid weights.

    For each timestep:
      1. Extract grid cell values from the NetCDF (flattened lat/lon grid).
      2. For each HRU, compute the HRU value as sum(cell_value * weight).
      3. Compute catchment average as area-weighted mean across HRUs.

    Args:
        nc_path: Path to NetCDF file (dims: time, lat, lon).
        var_name: Variable name in the NetCDF.
        gw_df: GridWeights DataFrame (HRU_ID, Cell_ID, Weight).
        hru_df: HRU table DataFrame.
        glacier_free_only: If True, exclude glacier HRUs from averaging.

    Returns:
        pandas Series with DatetimeIndex.
    """
    ds = xr.open_dataset(nc_path)
    data = ds[var_name]  # (time, lat, lon)
    times = pd.DatetimeIndex(ds['time'].values)

    # Flatten spatial dims to cell index (row-major: lat varies slowest)
    nlat = len(ds['lat'])
    nlon = len(ds['lon'])
    # Reshape to (time, n_cells)
    flat = data.values.reshape(len(times), nlat * nlon)

    # Filter HRUs if needed
    if glacier_free_only:
        valid_hrus = set(
            hru_df.loc[~hru_df['LAND_USE_CLASS'].isin(GLACIER_LU_CLASSES), 'HRU_ID']
        )
        gw_sub = gw_df[gw_df['HRU_ID'].isin(valid_hrus)].copy()
    else:
        gw_sub = gw_df.copy()

    if gw_sub.empty:
        warnings.warn("No HRUs remaining after filtering — returning NaN series.")
        return pd.Series(np.nan, index=times, name=var_name)

    # Get areas for selected HRUs
    area_map = hru_df.set_index('HRU_ID')['AREA'].to_dict()
    hru_ids = gw_sub['HRU_ID'].unique()

    # Build HRU-level values: for each HRU, weighted sum of grid cells
    # Then area-weighted average across HRUs
    hru_areas = np.array([area_map.get(h, 0) for h in hru_ids])
    total_area = hru_areas.sum()
    if total_area == 0:
        return pd.Series(np.nan, index=times, name=var_name)

    # Pre-compute weight matrix: for each HRU, which cells and what weight
    hru_cell_weights = {}
    for hru_id in hru_ids:
        mask = gw_sub['HRU_ID'] == hru_id
        cells = gw_sub.loc[mask, 'Cell_ID'].values
        weights = gw_sub.loc[mask, 'Weight'].values
        hru_cell_weights[hru_id] = (cells, weights)

    # Vectorized computation
    result = np.zeros(len(times))
    for i, hru_id in enumerate(hru_ids):
        cells, weights = hru_cell_weights[hru_id]
        # Cell IDs in GridWeights are 0-indexed into the flattened grid
        hru_vals = np.sum(flat[:, cells] * weights[np.newaxis, :], axis=1)
        result += hru_vals * hru_areas[i] / total_area

    ds.close()
    return pd.Series(result, index=times, name=var_name)


# =============================================================================
# Annual aggregation
# =============================================================================

def annual_streamflow(q_series, area_km2):
    """Convert daily Q (m3/s) to annual Q (mm/year), filtering low-coverage years.

    Returns:
        DataFrame with columns: year, Q_mm_year, Q_m3s_mean, n_days, coverage.
    """
    if q_series.empty:
        return pd.DataFrame(columns=['year', 'Q_mm_year', 'Q_m3s_mean', 'n_days', 'coverage'])

    q_series = q_series.copy()
    q_series.index = pd.DatetimeIndex(q_series.index)

    records = []
    for year, group in q_series.groupby(q_series.index.year):
        n_days_year = 366 if pd.Timestamp(year=year, month=1, day=1).is_leap_year else 365
        valid = group.dropna()
        coverage = len(valid) / n_days_year

        if coverage >= MIN_DATA_COVERAGE:
            mean_q = valid.mean()
            # mm/year = m3/s * 86400 * 365.25 / (area_km2 * 1e6) * 1000
            q_mm = mean_q * SECONDS_PER_DAY * DAYS_PER_YEAR / (area_km2 * 1e6) * 1000
            records.append({
                'year': year,
                'Q_mm_year': q_mm,
                'Q_m3s_mean': mean_q,
                'n_days': len(valid),
                'coverage': coverage,
            })

    return pd.DataFrame(records)


def annual_mean_timeseries(daily_series, name):
    """Aggregate daily series to annual mean.

    Returns:
        DataFrame with columns: year, {name}_daily_mean.
    """
    if daily_series.empty:
        return pd.DataFrame(columns=['year', f'{name}_daily_mean'])

    daily_series = daily_series.copy()
    daily_series.index = pd.DatetimeIndex(daily_series.index)

    records = []
    for year, group in daily_series.groupby(daily_series.index.year):
        records.append({
            'year': year,
            f'{name}_daily_mean': group.mean(),
        })
    return pd.DataFrame(records)


def annual_sum_timeseries(daily_series, name):
    """Aggregate daily series to annual sum (mm/day -> mm/year via sum).

    Returns:
        DataFrame with columns: year, {name}_mm_year.
    """
    if daily_series.empty:
        return pd.DataFrame(columns=['year', f'{name}_mm_year'])

    daily_series = daily_series.copy()
    daily_series.index = pd.DatetimeIndex(daily_series.index)

    records = []
    for year, group in daily_series.groupby(daily_series.index.year):
        records.append({
            'year': year,
            f'{name}_mm_year': group.sum(),
        })
    return pd.DataFrame(records)


def annual_glogem(glogem_path):
    """Load GloGEM_catchment_averaged.csv and compute annual glacier components.

    Uses the _catchment columns (already area-weighted to full catchment).
    Converts daily mm/day to mm/year by multiplying by 365.25.

    Returns:
        DataFrame with columns: year, icemelt_mm_year, snowmelt_glacier_mm_year,
        rain_glacier_mm_year, total_melt_mm_year.
    """
    df = pd.read_csv(glogem_path, parse_dates=['date'])
    df = df.set_index('date')

    cols = {
        'icemelt_all_catchment': 'icemelt_mm_year',
        'snowmelt_all_catchment': 'snowmelt_glacier_mm_year',
        'rain_all_catchment': 'rain_glacier_mm_year',
        'melt_all_catchment': 'total_melt_mm_year',
    }

    records = []
    for year, group in df.groupby(df.index.year):
        row = {'year': year}
        for src_col, dst_col in cols.items():
            if src_col in group.columns:
                row[dst_col] = group[src_col].mean() * DAYS_PER_YEAR
            else:
                row[dst_col] = np.nan
        records.append(row)

    return pd.DataFrame(records)


# =============================================================================
# Config discovery
# =============================================================================

def discover_configs(gauge_id, main_dir, env):
    """Auto-discover available configs for a catchment.

    Scans model_runs/catchment_{id}/configs/ for existing directories,
    then checks whether the matching topo_files variant exists.

    Returns:
        list of (config_key, nml_dict) tuples.
    """
    catchment_dir = main_dir / 'model_runs' / f'catchment_{gauge_id}'
    configs_dir = catchment_dir / 'configs'

    if not configs_dir.exists():
        return []

    # Get all known configuration keys from the registry
    registry = load_configurations_registry()
    registry_keys = {r['key'] for r in registry}

    discovered = []
    for config_dir in sorted(configs_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        config_key = config_dir.name
        if config_key not in registry_keys:
            continue

        try:
            nml, tmp_path = load_config(
                catchment=gauge_id,
                configuration=config_key,
                model='HBV',
                env=env,
            )
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)
        except (FileNotFoundError, KeyError) as e:
            print(f"  [WARN] Could not load config {config_key}: {e}")
            continue

        # Check topo variant exists
        paths = get_paths(nml)
        if not paths['topo_dir'].exists():
            print(f"  [SKIP] Topo dir missing for {config_key}: {paths['topo_dir']}")
            continue

        discovered.append((config_key, nml))

    return discovered


# =============================================================================
# Main processing
# =============================================================================

def process_catchment_config(gauge_id, config_key, nml, output_dir):
    """Process a single catchment x configuration combination.

    Returns:
        (annual_df, static_attrs) or (None, None) if data is missing.
    """
    paths = get_paths(nml)
    topo_dir = paths['topo_dir']
    data_obs_dir = paths['data_obs_dir']
    is_coupled = nml.get('coupled', False)

    print(f"  Processing {config_key} (coupled={is_coupled})")
    print(f"    Topo dir: {topo_dir}")
    print(f"    Data obs: {data_obs_dir}")

    # --- HRU table ---
    hru_path = topo_dir / 'HRU_table.csv'
    if not hru_path.exists():
        print(f"    [SKIP] HRU_table.csv not found")
        return None, None
    hru_df = parse_hru_table(hru_path)
    static = compute_static_attributes(hru_df)
    total_area = static['total_area_km2']

    print(f"    Area: {total_area:.1f} km2, {static['n_hrus']} HRUs, "
          f"glacier: {static['glacier_fraction_pct']:.1f}%")

    # --- Grid weights ---
    gw_path = topo_dir / 'GridWeights.txt'
    if not gw_path.exists():
        print(f"    [SKIP] GridWeights.txt not found")
        return None, None
    gw_df = parse_grid_weights(gw_path)

    # --- Annual streamflow ---
    rvt_path = data_obs_dir / 'Q_daily.rvt'
    q_annual = pd.DataFrame(columns=['year'])
    if rvt_path.exists():
        q_series = parse_q_daily(rvt_path)
        q_annual = annual_streamflow(q_series, total_area)
        print(f"    Q: {len(q_annual)} valid years")
    else:
        print(f"    [WARN] Q_daily.rvt not found")

    # --- Annual precipitation ---
    precip_path = data_obs_dir / 'era5_land_precip.nc'
    p_annual = pd.DataFrame(columns=['year'])
    if precip_path.exists():
        # For coupled configs, only use glacier-free HRUs for precip
        p_daily = compute_area_weighted_timeseries(
            precip_path, 'tp', gw_df, hru_df,
            glacier_free_only=is_coupled,
        )
        # ERA5 precip is in m/day, convert to mm/day
        p_daily = p_daily * 1000
        p_annual = annual_sum_timeseries(p_daily, 'P')
        if not p_annual.empty:
            print(f"    P: mean {p_annual['P_mm_year'].mean():.0f} mm/year")
    else:
        print(f"    [WARN] era5_land_precip.nc not found")

    # --- Annual PET ---
    pet_path = data_obs_dir / 'era5_land_pet.nc'
    pet_annual = pd.DataFrame(columns=['year'])
    if pet_path.exists():
        pet_daily = compute_area_weighted_timeseries(
            pet_path, 'pev', gw_df, hru_df,
            glacier_free_only=False,
        )
        # ERA5 PET is in m/day (negative convention), convert to positive mm/day
        pet_daily = pet_daily.abs() * 1000
        pet_annual = annual_sum_timeseries(pet_daily, 'PET')
        if not pet_annual.empty:
            print(f"    PET: mean {pet_annual['PET_mm_year'].mean():.0f} mm/year")
    else:
        print(f"    [WARN] era5_land_pet.nc not found")

    # --- Annual temperature ---
    temp_path = data_obs_dir / 'era5_land_temp_mean.nc'
    temp_annual = pd.DataFrame(columns=['year'])
    if temp_path.exists():
        temp_daily = compute_area_weighted_timeseries(
            temp_path, 't2m', gw_df, hru_df,
            glacier_free_only=False,
        )
        # ERA5 temp is in Kelvin, convert to Celsius
        temp_daily = temp_daily - 273.15
        temp_annual = annual_mean_timeseries(temp_daily, 'T')
        if not temp_annual.empty:
            print(f"    T: mean {temp_annual['T_daily_mean'].mean():.1f} degC")
    else:
        print(f"    [WARN] era5_land_temp_mean.nc not found")

    # --- GloGEM glacier components (coupled only) ---
    glogem_annual = pd.DataFrame(columns=['year'])
    if is_coupled:
        glogem_path = topo_dir / 'GloGEM_catchment_averaged.csv'
        if glogem_path.exists():
            glogem_annual = annual_glogem(glogem_path)
            if not glogem_annual.empty:
                print(f"    GloGEM: mean melt {glogem_annual['total_melt_mm_year'].mean():.0f} mm/year")
        else:
            print(f"    [WARN] GloGEM_catchment_averaged.csv not found (coupled config)")

    # --- Merge all annual DataFrames ---
    dfs = [q_annual, p_annual, pet_annual, temp_annual, glogem_annual]
    annual = None
    for df in dfs:
        if df.empty or 'year' not in df.columns:
            continue
        if annual is None:
            annual = df
        else:
            annual = annual.merge(df, on='year', how='outer')

    if annual is None or annual.empty:
        print(f"    [WARN] No annual data produced")
        return None, None

    annual = annual.sort_values('year').reset_index(drop=True)

    # --- Water balance residual ---
    if 'P_mm_year' in annual.columns and 'Q_mm_year' in annual.columns and 'PET_mm_year' in annual.columns:
        annual['residual_mm_year'] = annual['P_mm_year'] - annual['Q_mm_year'] - annual['PET_mm_year']

    # Add metadata columns
    annual.insert(0, 'gauge_id', gauge_id)
    annual.insert(1, 'config', config_key)

    return annual, static


def plot_water_balance(annual_df, gauge_id, config_key, is_coupled, output_dir):
    """Create water balance bar chart for a single catchment x config.

    X axis: years
    Bars: precipitation (blue)
    Lines: Q (black), PET (orange)
    If coupled: stacked area for glacier melt components.
    """
    df = annual_df.dropna(subset=['year']).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    years = df['year'].astype(int)

    # Precipitation bars
    if 'P_mm_year' in df.columns:
        ax.bar(years, df['P_mm_year'], color='steelblue', alpha=0.7,
               label='Precipitation', zorder=2)

    # Q line
    if 'Q_mm_year' in df.columns:
        valid = df['Q_mm_year'].notna()
        ax.plot(years[valid], df.loc[valid, 'Q_mm_year'], 'k-o', lw=2,
                markersize=4, label='Streamflow Q', zorder=4)

    # PET line
    if 'PET_mm_year' in df.columns:
        ax.plot(years, df['PET_mm_year'], '-s', color='darkorange', lw=2,
                markersize=4, label='PET', zorder=4)

    # Glacier components (coupled only)
    if is_coupled and 'icemelt_mm_year' in df.columns:
        ax.fill_between(years, 0, df['icemelt_mm_year'].fillna(0),
                        color='cyan', alpha=0.5, label='Icemelt', zorder=3)
        bottom = df['icemelt_mm_year'].fillna(0)
        if 'snowmelt_glacier_mm_year' in df.columns:
            ax.fill_between(years, bottom,
                            bottom + df['snowmelt_glacier_mm_year'].fillna(0),
                            color='lightblue', alpha=0.5,
                            label='Glacier snowmelt', zorder=3)
            bottom = bottom + df['snowmelt_glacier_mm_year'].fillna(0)
        if 'rain_glacier_mm_year' in df.columns:
            ax.fill_between(years, bottom,
                            bottom + df['rain_glacier_mm_year'].fillna(0),
                            color='mediumpurple', alpha=0.4,
                            label='Glacier rain', zorder=3)

    # Residual line
    if 'residual_mm_year' in df.columns:
        valid = df['residual_mm_year'].notna()
        ax.plot(years[valid], df.loc[valid, 'residual_mm_year'], '--',
                color='red', lw=1.5, label='Residual (P-Q-PET)', zorder=4)

    ax.set_xlabel('Year')
    ax.set_ylabel('mm / year')
    ax.set_title(f'Catchment {gauge_id} — Annual Water Balance ({config_key})')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(years.min() - 0.5, years.max() + 0.5)

    plt.tight_layout()
    out_path = output_dir / f'water_balance_{gauge_id}_{config_key}.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {out_path}")


def plot_multi_config_comparison(all_results, gauge_id, output_dir):
    """Create comparison plot across configurations for a single catchment.

    Side-by-side bars for multi-year mean P, Q, PET per config.
    """
    catchment_data = {k: v for k, v in all_results.items() if k[0] == gauge_id}
    if not catchment_data:
        return

    configs = []
    means = []
    for (gid, ckey), (annual_df, static) in catchment_data.items():
        if annual_df is None or annual_df.empty:
            continue
        row = {'config': ckey}
        for col in ['P_mm_year', 'Q_mm_year', 'PET_mm_year', 'residual_mm_year',
                     'icemelt_mm_year', 'snowmelt_glacier_mm_year',
                     'rain_glacier_mm_year', 'total_melt_mm_year']:
            if col in annual_df.columns:
                row[col] = annual_df[col].mean()
        means.append(row)
        configs.append(ckey)

    if not means:
        return

    means_df = pd.DataFrame(means)
    n_configs = len(means_df)

    fig, ax = plt.subplots(figsize=(max(8, n_configs * 2.5), 6))
    x = np.arange(n_configs)
    width = 0.22

    # Precipitation
    if 'P_mm_year' in means_df.columns:
        ax.bar(x - width, means_df['P_mm_year'], width, color='steelblue',
               label='Precip', alpha=0.8)

    # Streamflow
    if 'Q_mm_year' in means_df.columns:
        ax.bar(x, means_df['Q_mm_year'], width, color='black',
               label='Streamflow Q', alpha=0.7)

    # PET
    if 'PET_mm_year' in means_df.columns:
        ax.bar(x + width, means_df['PET_mm_year'], width, color='darkorange',
               label='PET', alpha=0.8)

    # Total glacier melt (if any)
    if 'total_melt_mm_year' in means_df.columns:
        vals = means_df['total_melt_mm_year'].fillna(0)
        if vals.sum() > 0:
            ax.bar(x + 2 * width, vals, width, color='cyan',
                   label='Glacier melt', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha='right')
    ax.set_ylabel('mm / year (multi-year mean)')
    ax.set_title(f'Catchment {gauge_id} — Configuration Comparison')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f'config_comparison_{gauge_id}.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved comparison: {out_path}")


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute annual water balance components for Raven catchments.',
    )
    parser.add_argument(
        '--catchments', nargs='+', default=None,
        help='Catchment IDs to process (default: all available).',
    )
    parser.add_argument(
        '--env', type=str, default=None,
        help='Environment (local/server). Auto-detected if not specified.',
    )
    parser.add_argument(
        '-o', '--output-dir', type=str, default='outputs/water_balance',
        help='Output directory for CSVs and plots.',
    )
    args = parser.parse_args()

    # Resolve output dir relative to project root
    project_root = Path(__file__).parent.parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine catchments
    catchment_ids = args.catchments or list_catchments()
    print(f"Catchments: {catchment_ids}")

    # Detect main_dir from a dummy config load
    dummy_nml, dummy_tmp = load_config(
        catchment=catchment_ids[0],
        configuration='baseline',
        model='HBV',
        env=args.env,
    )
    dummy_tmp.unlink(missing_ok=True)
    main_dir = Path(dummy_nml['main_dir'])
    print(f"Main dir: {main_dir}")
    print(f"Output dir: {output_dir}")
    print()

    all_results = {}  # (gauge_id, config_key) -> (annual_df, static_attrs)
    all_annual_rows = []
    summary_rows = []

    for gauge_id in catchment_ids:
        print(f"=== Catchment {gauge_id} ===")

        configs = discover_configs(gauge_id, main_dir, args.env)
        if not configs:
            print(f"  No configs found, skipping.")
            print()
            continue

        print(f"  Found configs: {[c[0] for c in configs]}")

        for config_key, nml in configs:
            annual_df, static = process_catchment_config(
                gauge_id, config_key, nml, output_dir,
            )

            if annual_df is None:
                continue

            all_results[(gauge_id, config_key)] = (annual_df, static)
            all_annual_rows.append(annual_df)

            # Per-catchment-config CSV
            csv_path = output_dir / f'annual_{gauge_id}_{config_key}.csv'
            annual_df.to_csv(csv_path, index=False, float_format='%.3f')
            print(f"    Saved: {csv_path}")

            # Water balance plot
            is_coupled = nml.get('coupled', False)
            plot_water_balance(annual_df, gauge_id, config_key, is_coupled, output_dir)

        # Multi-config comparison plot
        plot_multi_config_comparison(all_results, gauge_id, output_dir)
        print()

    # --- Summary CSV: multi-year averages per catchment x config ---
    if all_annual_rows:
        combined = pd.concat(all_annual_rows, ignore_index=True)

        # Save combined annual file
        combined_path = output_dir / 'annual_all.csv'
        combined.to_csv(combined_path, index=False, float_format='%.3f')
        print(f"Combined annual CSV: {combined_path}")

        # Multi-year averages
        numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
        if 'year' in numeric_cols:
            numeric_cols.remove('year')
        group_cols = ['gauge_id', 'config']
        summary = combined.groupby(group_cols)[numeric_cols].mean().reset_index()

        # Add year range info
        year_range = combined.groupby(group_cols)['year'].agg(['min', 'max', 'count'])
        year_range.columns = ['year_start', 'year_end', 'n_years']
        summary = summary.merge(year_range.reset_index(), on=group_cols)

        # Add static attributes
        static_rows = []
        for (gid, ckey), (_, static) in all_results.items():
            if static is not None:
                static_rows.append({
                    'gauge_id': gid,
                    'config': ckey,
                    'total_area_km2': static['total_area_km2'],
                    'glacier_area_km2': static['glacier_area_km2'],
                    'glacier_fraction_pct': static['glacier_fraction_pct'],
                    'n_hrus': static['n_hrus'],
                    'n_elevation_bands': static['n_elevation_bands'],
                    'mean_elevation_m': static['mean_elevation_m'],
                })
        if static_rows:
            static_df = pd.DataFrame(static_rows)
            summary = summary.merge(static_df, on=['gauge_id', 'config'], how='left')

        summary_path = output_dir / 'summary.csv'
        summary.to_csv(summary_path, index=False, float_format='%.3f')
        print(f"Summary CSV: {summary_path}")
    else:
        print("No data processed.")

    print("\nDone.")


if __name__ == '__main__':
    main()
