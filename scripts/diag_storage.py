"""One-off diagnostic for the empty-storage-plot bug in compare_sphy_options.py.

Loads SOIL[1] for catchment 2268 / glogem_subdaily_opt1 and prints whether
each step of the pipeline (file discovery → CSV parse → HRU ID parsing →
HRU area lookup → column intersection) produces sensible output.

Run as:
    python scripts/diag_storage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from compare_sphy_options import load_soil_by_hru, load_hru_areas
from config_merge import load_config
from paths import get_paths


def main():
    nml, _ = load_config(
        catchment='2268', configuration='glogem_subdaily_opt1', model='SPHY',
    )
    paths = get_paths(nml)
    print('output_dir:', paths['output_dir'])
    print('topo_dir: ', paths['topo_dir'])
    print()

    # --- Step 1: load the SOIL[1] storage CSV ---
    soil = load_soil_by_hru(paths['output_dir'], 1)
    if soil is None:
        print('SOIL[1] load returned None — file not found by load_soil_by_hru.')
        print('Listing all CSVs matching SOIL[1]:')
        for p in sorted(Path(paths['output_dir']).glob('*SOIL*ByHRU*.csv')):
            print(' ', p.name)
        return

    print(f'SOIL[1] shape:           {soil.shape}')
    print(f'SOIL[1] columns sample:  {list(soil.columns)[:6]}')
    print(f'  column dtype:          {type(soil.columns[0]).__name__}')
    print(f'  first column types:    {[type(c).__name__ for c in soil.columns[:5]]}')
    print(f'SOIL[1] first row:       {soil.iloc[0, :5].to_dict()}')
    print(f'SOIL[1] values mean:     {soil.values.mean():.2f}')
    print(f'SOIL[1] values max:      {soil.values.max():.2f}')
    print()

    # --- Step 2: load HRU areas ---
    areas = load_hru_areas(paths['topo_dir'])
    if areas is None:
        print('HRU_table.csv NOT FOUND under', paths['topo_dir'])
        # Probe likely alternative paths
        print('Probing alternatives...')
        for p in sorted(Path(paths['topo_dir']).parent.rglob('HRU_table.csv')):
            print(' ', p)
        return

    print(f'HRU_table.csv rows:      {len(areas)}')
    print(f'HRU_ID index sample:     {list(areas.index)[:6]}')
    print(f'  index dtype:           {areas.index.dtype}')
    print()

    # --- Step 3: intersection ---
    common = [h for h in soil.columns if h in areas.index]
    print(f'Common HRU IDs between SOIL columns and areas: {len(common)} '
          f'(out of {len(soil.columns)} SOIL cols, {len(areas)} areas)')
    if not common:
        print('  ⚠️ EMPTY intersection — this is why storage plots are empty.')
        print(f'  SOIL columns (full): {list(soil.columns)}')
        print(f'  Areas index  (full): {list(areas.index)}')
    else:
        print(f'  ✓ first matches: {common[:6]}')


if __name__ == '__main__':
    main()
