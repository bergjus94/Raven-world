"""CLI wrapper around src/preprocess_modis_fsca.derive_for_catchment.

Derives per-catchment MODIS fSCA — either basin-mean or per-elevation-band —
from the region NetCDF produced by ``scripts/build_modis_region.py``.

Output is a long-format CSV at:
    <main_dir>/01_data/snow/MODIS/basins/<gauge>/fsca_<product>_<gauge>.csv

with columns: ``date, band_m, fsca, n_valid, n_cloud, n_total``. The
``band_m`` column is the elevation-band lower edge in metres
(``'basin'`` for the basin-mean variant).

Usage
-----
    python scripts/derive_basin_fsca.py 2268                                  # basin-mean
    python scripts/derive_basin_fsca.py 2268 --aggregation elevation_band     # 100 m bands
    python scripts/derive_basin_fsca.py 2268 --aggregation elevation_band \\
                                              --band-width 200
    python scripts/derive_basin_fsca.py 2268 --no-glacier-mask
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from config_merge import load_config  # noqa: E402
from preprocess_modis_fsca import derive_for_catchment  # noqa: E402


def parse_years(spec: str) -> List[int]:
    """Parse '2000-2014' or '2000,2005,2010' into a sorted year list."""
    out: List[int] = []
    for chunk in spec.split(','):
        if '-' in chunk:
            a, b = chunk.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('gauge_id', help='Catchment gauge ID (e.g. 2268)')
    ap.add_argument('--aggregation', default='basin_mean',
                    choices=['basin_mean', 'elevation_band'],
                    help='Aggregation mode (default: basin_mean).')
    ap.add_argument('--band-width', type=int, default=100,
                    help='Elevation band width in metres '
                         '(only for --aggregation elevation_band; default 100).')
    ap.add_argument('--product', default='MOD10A2',
                    choices=['MOD10A1', 'MYD10A1', 'MOD10A2', 'MYD10A2'])
    ap.add_argument('--years', default=None,
                    help="Year range '2000-2020' to subset; "
                         "default: all available")
    ap.add_argument('--region', default=None,
                    help='Region override (default: from namelist '
                         '`region:` key).')
    ap.add_argument('--main-dir', default=None,
                    help='Override main_dir (default: env layer autodetect).')
    ap.add_argument('--env', default=None,
                    help='Env layer override (server/local).')
    ap.add_argument('--no-glacier-mask', action='store_true',
                    help='Skip the glacier-pixel exclusion step.')
    ap.add_argument('--configuration', default=None,
                    help='Configuration key for namelist merge. Any '
                         'compatible config works since this script only '
                         'needs main_dir / region / catchment shape / '
                         'glacier outline.')
    args = ap.parse_args(argv)

    # Load namelist for path resolution. We don't actually use the model
    # configuration — picking any one keeps load_config happy.
    config_key = args.configuration or 'glogem_subdaily_opt1'
    nml, _tmp = load_config(
        catchment=args.gauge_id, configuration=config_key,
        model='SPHY', env=args.env,
    )
    if args.main_dir:
        nml['main_dir'] = args.main_dir

    years = parse_years(args.years) if args.years else None

    try:
        derive_for_catchment(
            nml,
            aggregation=args.aggregation,
            band_width_m=args.band_width,
            glacier_mask=not args.no_glacier_mask,
            product=args.product,
            years=years,
            region=args.region,
        )
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
