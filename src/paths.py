"""
Centralized path construction for Raven hydrological modeling.

Single source of truth for all directory paths. Every preprocessor and
postprocessing script should import get_paths() instead of building
paths with inline f-strings.

Usage:
    from paths import get_paths

    paths = get_paths(nml)
    paths['output_dir'] / f"{gauge_id}_{model_type}_Hydrographs.csv"
    paths['topo_dir'] / 'HRU_table.csv'
"""

from pathlib import Path


def get_topo_variant(nml: dict) -> str:
    """Derive the topo variant from config fields.

    - 'aspect' if 'aspect' is in criteria
    - 'icemelt' if coupled + irrigation_variable == 'icemelt'
    - 'default' otherwise
    """
    criteria = nml.get('criteria', [])
    if 'aspect' in criteria:
        return 'aspect'
    if nml.get('coupled', False) and nml.get('irrigation_variable') == 'icemelt':
        return 'icemelt'
    return 'default'


def get_paths(nml: dict) -> dict:
    """Return all standard paths for a given merged namelist.

    Args:
        nml: Merged namelist dict (from config_merge.load_config or a loaded YAML).

    Returns:
        Dict with Path objects for all standard directories:
        - catchment_dir:        main_dir/model_runs/catchment_{gauge_id}
        - topo_dir:             .../topo_files/{variant}  (variant-specific HRU files)
        - topo_shared_dir:      .../topo_files  (shared DEM, clipped shapefiles)
        - data_obs_dir:         .../data_obs  (shared meteo, streamflow, irrigation)
        - cmip6_dir:            .../cmip6  (future climate forcing per model)
        - config_dir:           .../configs/{config_key}
        - model_dir:            .../configs/{config_key}/{model_type}
        - output_dir:           .../configs/{config_key}/{model_type}/output
        - template_dir:         .../configs/{config_key}/{model_type}/templates
        - results_dir:          .../configs/{config_key}/{model_type}/results
        - plots_dir:            .../plots  (shared diagnostic plots)
        - model_comparisons_dir: .../model_comparisons
    """
    main_dir = Path(nml['main_dir'])
    gauge_id = nml['gauge_id']
    model_type = nml.get('model_type', 'HBV')
    config_key = nml.get('_config_key', 'baseline')

    catchment_dir = main_dir / 'model_runs' / f'catchment_{gauge_id}'
    topo_variant = get_topo_variant(nml)

    return {
        'catchment_dir': catchment_dir,
        'topo_dir': catchment_dir / 'topo_files' / topo_variant,
        'topo_shared_dir': catchment_dir / 'topo_files',
        'data_obs_dir': catchment_dir / 'data_obs',
        'cmip6_dir': catchment_dir / 'cmip6',
        'config_dir': catchment_dir / 'configs' / config_key,
        'model_dir': catchment_dir / 'configs' / config_key / model_type,
        'output_dir': catchment_dir / 'configs' / config_key / model_type / 'output',
        'template_dir': catchment_dir / 'configs' / config_key / model_type / 'templates',
        'results_dir': catchment_dir / 'configs' / config_key / model_type / 'results',
        'plots_dir': catchment_dir / 'plots',
        'model_comparisons_dir': catchment_dir / 'model_comparisons',
    }


def get_relative_data_obs(nml: dict) -> str:
    """Return the relative path from model_dir to data_obs_dir.

    Used in .rvt files to reference shared forcing data.
    E.g. from configs/glogem/HBV/ → ../../../data_obs/
    """
    return '../../../data_obs'


def get_relative_topo(nml: dict) -> str:
    """Return the relative path from model_dir to the topo variant dir.

    Used in .rv* files to reference HRU data.
    E.g. from configs/glogem/HBV/ → ../../../topo_files/default/
    """
    variant = get_topo_variant(nml)
    return f'../../../topo_files/{variant}'
