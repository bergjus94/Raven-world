"""
Switzerland GloGEM converter — .dat → netCDF.

Mirrors the output structure of preprocess_GloGEM_netcdf.py (Indus version)
so downstream tooling works identically on both basins.

Switzerland folder layout:
    <base>/<model>/<scenario>/centraleurope_<Component>(_variant)_r1_<catch>.dat

Daily components processed (match Indus set):
    Discharge : centraleurope_Discharge_r1_*.dat
    Icemelt   : centraleurope_Icemelt_day_r1_*.dat
    Rain      : centraleurope_Rain_day_r1_*.dat
    Snowmelt  : centraleurope_Snowmelt_day_r1_*.dat
"""
import argparse
import gc
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


COMPONENT_GLOB = {
    'Discharge': 'centraleurope_Discharge_r1_*.dat',
    'Icemelt':   'centraleurope_Icemelt_day_r1_*.dat',
    'Rain':      'centraleurope_Rain_day_r1_*.dat',
    'Snowmelt':  'centraleurope_Snowmelt_day_r1_*.dat',
}


def setup_logger():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    return logging.getLogger('GloGEM-CH')


def parse_dat_file(dat_file: Path, seen_glacier_years: set) -> pd.DataFrame:
    records = []
    with open(dat_file, 'r') as f:
        for line in f:
            if line.startswith('ID') or line.startswith('//') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            glacier_id = parts[0]
            try:
                year = int(parts[1])
            except (ValueError, IndexError):
                continue
            if (glacier_id, year) in seen_glacier_years:
                continue
            seen_glacier_years.add((glacier_id, year))
            start_date = datetime(year - 1, 10, 1)
            for day, val in enumerate(parts[3:]):
                try:
                    value = float(val) if val != '*' else np.nan
                except ValueError:
                    value = np.nan
                records.append((glacier_id, start_date + timedelta(days=day), value))
    if records:
        return pd.DataFrame(records, columns=['glacier_id', 'date', 'value'])
    return pd.DataFrame()


def process_component(base_dir: Path, component: str, model: str, scenario: str):
    logger = setup_logger()

    scenario_dir = base_dir / model / scenario
    output_file = base_dir / f'GloGEM_{component}_Switzerland_{model}_{scenario}.nc'

    if output_file.exists():
        logger.info(f'already exists, skipping: {output_file.name}')
        return xr.open_dataset(output_file)

    dat_files = sorted(scenario_dir.glob(COMPONENT_GLOB[component]))
    if not dat_files:
        logger.warning(f'no files matched in {scenario_dir} for {component}')
        return None
    logger.info(f'{component} / {model} / {scenario}: {len(dat_files)} catchment files')

    seen_glacier_years = set()
    all_dfs = []
    for i, dat_file in enumerate(dat_files, 1):
        logger.info(f'  [{i}/{len(dat_files)}] {dat_file.name}')
        df = parse_dat_file(dat_file, seen_glacier_years)
        if not df.empty:
            all_dfs.append(df)
        if i % 10 == 0:
            gc.collect()

    if not all_dfs:
        logger.warning(f'no records parsed for {component}/{model}/{scenario}')
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    logger.info(f'  records: {len(combined):,}  glaciers: {combined.glacier_id.nunique()}  '
                f'date range: {combined.date.min()} to {combined.date.max()}')

    dup_mask = combined.duplicated(subset=['date', 'glacier_id'], keep=False)
    if dup_mask.any():
        logger.warning(f'  dropping {dup_mask.sum()} duplicate (date, glacier_id) rows')
        combined = combined.drop_duplicates(subset=['date', 'glacier_id'], keep='first')
        gc.collect()

    pivot = combined.pivot(index='date', columns='glacier_id', values='value')
    del combined
    gc.collect()

    ds = xr.Dataset(
        {component.lower(): (['time', 'glacier_id'], pivot.values.astype(np.float32))},
        coords={
            'time': pivot.index,
            'glacier_id': pivot.columns.astype(str),
        },
    )
    del pivot
    gc.collect()

    ds.attrs.update({
        'title': f'GloGEM {component} for Switzerland ({model})',
        'scenario': scenario,
        'model': model,
        'source': 'GloGEM glacier model',
        'units': 'mm/day',
        'date_range': f'{ds.time.values[0]} to {ds.time.values[-1]}',
        'n_glaciers': len(ds.glacier_id),
        'n_timesteps': len(ds.time),
    })

    encoding = {component.lower(): {'zlib': True, 'complevel': 5, 'dtype': 'float32'}}
    ds.to_netcdf(output_file, encoding=encoding)

    mb = output_file.stat().st_size / 1024 / 1024
    raw_mb = sum(f.stat().st_size for f in dat_files) / 1024 / 1024
    logger.info(f'  wrote {output_file.name}  ({mb:.1f} MB, {100*(1-mb/raw_mb):.1f}% reduction vs {raw_mb:.1f} MB raw)')
    return ds


def discover(base_dir: Path, requested_models, requested_scenarios):
    if requested_models:
        models = requested_models
    else:
        models = sorted(p.name for p in base_dir.iterdir()
                        if p.is_dir() and not p.name.startswith('.') and not p.name.startswith('GloGEM_'))
    pairs = []
    for m in models:
        mdir = base_dir / m
        if not mdir.is_dir():
            continue
        scens = requested_scenarios or sorted(p.name for p in mdir.iterdir() if p.is_dir())
        for s in scens:
            if (mdir / s).is_dir():
                pairs.append((m, s))
    return pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir',
                        default='/home/jberg/OneDrive/Raven_worldwide/01_data/GloGEM_future')
    parser.add_argument('--model', nargs='+', default=None,
                        help='One or more RCM/GCM model folder names (default: all)')
    parser.add_argument('--scenario', nargs='+', default=None,
                        help='One or more scenarios, e.g. ssp126 ssp245 ssp585 (default: all present)')
    parser.add_argument('--component', choices=['Discharge', 'Icemelt', 'Snowmelt', 'Rain', 'all'],
                        default='all')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    components = ['Discharge', 'Icemelt', 'Snowmelt', 'Rain'] if args.component == 'all' else [args.component]
    pairs = discover(base_dir, args.model, args.scenario)

    logger = setup_logger()
    logger.info(f'processing {len(pairs)} (model, scenario) pairs × {len(components)} components')
    for model, scenario in pairs:
        for component in components:
            process_component(base_dir, component, model, scenario)
