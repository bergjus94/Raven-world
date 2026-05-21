#!/usr/bin/env python3
"""
Validate the composable config system by merging all (catchment × config) combinations
and comparing key fields against the existing namelists_server/ files.

Usage:
    python scripts/validate_config_merge.py
    python scripts/validate_config_merge.py --verbose
    python scripts/validate_config_merge.py --catchment 0101 --config glogem
"""

import sys
import argparse
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import yaml
from config_merge import load_config, list_catchments, list_configurations


NAMELISTS_SERVER = Path(__file__).parent.parent / 'namelists_server'

# Fields to compare (the meaningful ones that affect model runs)
COMPARE_FIELDS = [
    'gauge_id', 'start_date', 'end_date', 'cali_end_date', 'warm_up_date',
    'model_type', 'template', 'coupled', 'meteo_source', 'nconnect',
    'subdaily_method', 'future',
]

# Fields only present in coupled configs
COUPLED_FIELDS = [
    'glogem_model', 'glogem_scenario', 'basin_name', 'irrigation_variable',
]

# Optional fields (may or may not be present)
OPTIONAL_FIELDS = [
    'precip_source', 'pet_method',
    'lapse_rate_precip_source',  # decoupled lapse-rate gridded precip override
    'perc_option',               # SPHY percolation variant: 1 (HBV-Light) or 2 (SPHY-faithful)
]

# Map from generate_namelists config names to our config layer names
CONFIG_NAME_MAP = {
    'base': 'baseline',
}


def load_server_namelist(gauge_id, config_name):
    """Load an existing server namelist for comparison."""
    if config_name == 'base':
        path = NAMELISTS_SERVER / f'namelist_{gauge_id}.yaml'
    else:
        path = NAMELISTS_SERVER / f'namelist_{gauge_id}_{config_name}.yaml'

    if not path.exists():
        return None

    with open(path) as f:
        return yaml.safe_load(f)


def compare_configs(merged, server, gauge_id, config_name, verbose=False):
    """Compare merged config against server namelist. Return list of differences."""
    diffs = []

    for field in COMPARE_FIELDS:
        m_val = merged.get(field)
        s_val = server.get(field)
        if str(m_val) != str(s_val):
            diffs.append(f"  {field}: merged={m_val!r} vs server={s_val!r}")

    if merged.get('coupled', False):
        for field in COUPLED_FIELDS:
            m_val = merged.get(field)
            s_val = server.get(field)
            if str(m_val) != str(s_val):
                diffs.append(f"  {field}: merged={m_val!r} vs server={s_val!r}")

    for field in OPTIONAL_FIELDS:
        m_val = merged.get(field)
        s_val = server.get(field)
        if m_val is not None or s_val is not None:
            if str(m_val) != str(s_val):
                diffs.append(f"  {field}: merged={m_val!r} vs server={s_val!r}")

    # Compare calibration sub-fields
    m_cali = merged.get('calibration', {})
    s_cali = server.get('calibration', {})
    for field in ['iterations', 'ngs', 'cali_end_date', 'vali_end_date']:
        m_val = m_cali.get(field)
        s_val = s_cali.get(field)
        if str(m_val) != str(s_val):
            diffs.append(f"  calibration.{field}: merged={m_val!r} vs server={s_val!r}")

    return diffs


def main():
    parser = argparse.ArgumentParser(description='Validate composable config merge')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--catchment', '-c', type=str, default=None)
    parser.add_argument('--config', '-C', type=str, default=None)
    args = parser.parse_args()

    if not NAMELISTS_SERVER.exists():
        print(f"Warning: {NAMELISTS_SERVER} not found, skipping server comparison")
        print("Running merge-only validation...")
        server_compare = False
    else:
        server_compare = True

    # Server namelists only use these 10 configs
    server_configs = [
        'base', 'har', 'tphipr', 'oudin',
        'glogem', 'glogem_har', 'glogem_tphipr',
        'icemelt', 'glogem_oudin', 'glogem_gmb',
    ]

    catchments = [args.catchment] if args.catchment else list_catchments()
    if args.config:
        configs_to_test = [args.config]
    else:
        configs_to_test = list_configurations()

    total = 0
    passed = 0
    failed = 0
    skipped = 0

    for catchment in catchments:
        for config in configs_to_test:
            total += 1
            try:
                merged, tmp_path = load_config(
                    catchment=catchment,
                    configuration=config,
                    model='HBV',
                    env='server',
                )
                # Clean up temp file
                tmp_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"FAIL  {catchment} × {config}: merge error: {e}")
                failed += 1
                continue

            # Compare against server namelist if available
            # Map our config names back to generate_namelists names
            server_name = {v: k for k, v in CONFIG_NAME_MAP.items()}.get(config, config)

            if server_compare and server_name in server_configs:
                server = load_server_namelist(catchment, server_name)
                if server is None:
                    if args.verbose:
                        print(f"SKIP  {catchment} × {config}: no server namelist")
                    skipped += 1
                    continue

                diffs = compare_configs(merged, server, catchment, config, args.verbose)
                if diffs:
                    print(f"DIFF  {catchment} × {config}:")
                    for d in diffs:
                        print(d)
                    failed += 1
                else:
                    if args.verbose:
                        print(f"OK    {catchment} × {config}")
                    passed += 1
            else:
                # Config not in server namelists, just verify merge succeeds
                if args.verbose:
                    print(f"OK    {catchment} × {config} (merge only, no server file)")
                passed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped (of {total} total)")

    if failed:
        print("\nSome configs have differences. Review above.")
        sys.exit(1)
    else:
        print("\nAll configs validated successfully!")


if __name__ == '__main__':
    main()
