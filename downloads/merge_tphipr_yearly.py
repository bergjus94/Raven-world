"""Merge daily TPHiPr precipitation NetCDFs into one file per year.

Source layout:
    <tphipr_dir>/<year>/tpmfd_prcp_d_YYYYMMDD.nc   (no time dimension)

Output (written to <tphipr_dir>):
    tpmfd_prcp_<year>.nc                             (time, latitude, longitude)

Usage:
    python merge_tphipr_yearly.py [--tphipr_dir PATH] [--years YEAR [YEAR ...]]

Defaults:
    tphipr_dir : /home/jberg/OneDrive/Raven_worldwide/01_data/meteo/TPHiPr
    years      : all numeric sub-directories found in tphipr_dir
"""

import argparse
import gc
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


# ── helpers ────────────────────────────────────────────────────────────────────

DATE_RE = re.compile(r"tpmfd_prcp_d_(\d{8})\.nc$")


def _date_from_path(nc_path: Path) -> datetime:
    m = DATE_RE.search(nc_path.name)
    if not m:
        raise ValueError(f"Cannot parse date from filename: {nc_path.name}")
    return datetime.strptime(m.group(1), "%Y%m%d")


def _open_daily(nc_path: Path) -> xr.Dataset:
    """Open one daily file and add a scalar time coordinate."""
    date = _date_from_path(nc_path)
    ds = xr.open_dataset(nc_path)
    ds = ds.expand_dims({"time": [np.datetime64(date, "ns")]})
    return ds


# ── main merge logic ───────────────────────────────────────────────────────────

def merge_year(year: int, tphipr_dir: Path) -> None:
    year_dir = tphipr_dir / str(year)
    if not year_dir.is_dir():
        print(f"  [skip] {year_dir} not found")
        return

    out_path = tphipr_dir / f"tpmfd_prcp_{year}.nc"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return

    daily_files = sorted(year_dir.glob("tpmfd_prcp_d_*.nc"))
    if not daily_files:
        print(f"  [skip] no daily files found in {year_dir}")
        return

    print(f"  Merging {len(daily_files)} files for {year} ...", flush=True)

    datasets = [_open_daily(f) for f in daily_files]
    merged = xr.concat(datasets, dim="time")

    merged.attrs["description"] = (
        f"TPHiPr daily precipitation for {year}, merged from daily files"
    )
    merged.attrs["history"] = (
        f"Merged by merge_tphipr_yearly.py on {datetime.now():%Y-%m-%d}"
    )
    merged["prcp"].attrs["units"] = merged["prcp"].attrs.get("units", "mm/day")
    merged["time"].attrs["long_name"] = "time"

    encoding = {
        "prcp": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "time": {"dtype": "float64"},
    }
    merged.to_netcdf(out_path, encoding=encoding)
    print(f"  -> written: {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    for ds in datasets:
        ds.close()
    merged.close()
    gc.collect()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    default_dir = Path(
        "/home/jberg/OneDrive/Raven_worldwide/01_data/meteo/TPHiPr"
    )

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tphipr_dir", type=Path, default=default_dir,
        help="Root TPHiPr directory containing per-year sub-folders (default: %(default)s)",
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=None,
        help="Years to process (default: all numeric sub-dirs found)",
    )
    args = parser.parse_args()

    tphipr_dir = args.tphipr_dir
    if not tphipr_dir.is_dir():
        raise SystemExit(f"TPHiPr directory not found: {tphipr_dir}")

    if args.years:
        years = sorted(args.years)
    else:
        years = sorted(
            int(d.name) for d in tphipr_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        )

    print(f"TPHiPr directory : {tphipr_dir}")
    print(f"Years to process : {years[0]}–{years[-1]}  ({len(years)} years)\n")

    for year in years:
        merge_year(year, tphipr_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
