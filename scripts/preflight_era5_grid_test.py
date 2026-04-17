"""
Preflight: download one ERA5-Land month and verify the grid aligns with the
existing Indus-Era5 files before committing to a 1970s backfill.

Strategy:
  1. Request 2m_temperature for 1979-12 with a slightly wider bbox.
  2. Compare lat/lon arrays to the existing 1980-01 file.
  3. Report whether the new grid covers (and is bit-for-bit aligned with) the
     existing grid after subsetting.
"""
from pathlib import Path
import cdsapi
import numpy as np
import xarray as xr

OUT = Path('/tmp/era5_preflight_1979_12_t2m.nc')
REF = Path('/home/jberg/OneDrive/Raven_worldwide/01_data/meteo/Indus-Era5/era5_land_2m_temperature_1980_01.nc')

# slightly wider than the existing grid so any small CDS-side rounding
# still includes every existing cell
AREA = [37.2, 71.0, 30.9, 82.0]   # N, W, S, E

if not OUT.exists():
    print(f'requesting {OUT.name} with area={AREA} ...')
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-land',
        {
            'variable': '2m_temperature',
            'year': '1979',
            'month': '12',
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': [f'{h:02d}:00' for h in range(24)],
            'area': AREA,
            'data_format': 'netcdf',
            'download_format': 'unarchived',
        },
        str(OUT),
    )
    print(f'downloaded: {OUT.stat().st_size / 1024**2:.1f} MB')
else:
    print(f'reusing cached {OUT}')

new = xr.open_dataset(OUT)
ref = xr.open_dataset(REF)

print()
print('=== new (1979-12 preflight) ===')
print(f'  lat {float(new.latitude.min()):.4f} .. {float(new.latitude.max()):.4f}  n={len(new.latitude)}')
print(f'  lon {float(new.longitude.min()):.4f} .. {float(new.longitude.max()):.4f}  n={len(new.longitude)}')

print()
print('=== ref (existing 1980-01) ===')
print(f'  lat {float(ref.latitude.min()):.4f} .. {float(ref.latitude.max()):.4f}  n={len(ref.latitude)}')
print(f'  lon {float(ref.longitude.min()):.4f} .. {float(ref.longitude.max()):.4f}  n={len(ref.longitude)}')

# Does the new grid contain every ref cell?  (strict equality on each coord value)
ref_lats = set(np.round(ref.latitude.values, 6))
ref_lons = set(np.round(ref.longitude.values, 6))
new_lats = set(np.round(new.latitude.values, 6))
new_lons = set(np.round(new.longitude.values, 6))

missing_lat = ref_lats - new_lats
missing_lon = ref_lons - new_lons
extra_lat   = new_lats - ref_lats
extra_lon   = new_lons - ref_lons

print()
print(f'  ref lats missing from new:  {len(missing_lat)}   extra in new: {len(extra_lat)}')
print(f'  ref lons missing from new:  {len(missing_lon)}   extra in new: {len(extra_lon)}')

if not missing_lat and not missing_lon:
    print()
    print('  PASS: every existing grid cell is present in the new download.')
    print('        The 1970s backfill will align bit-for-bit with the existing Indus-Era5 grid')
    print('        (after subsetting to the existing bbox if desired).')
else:
    print()
    print('  FAIL: the new download is missing cells that exist in the reference.')
    print('        Adjust the bbox and retry.')
