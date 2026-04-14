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

    Topo variants capture discretization differences that produce different
    HRU shapefiles.  Three axes matter:

    1. **Coupled mode** — coupled (non-icemelt) aggregates glaciers into 2 HRUs
       (large + small) instead of criteria-based discretization → different HRU count.
    2. **Criteria** — 'aspect' in criteria adds aspect-based splitting.
    3. **Icemelt mode** — glacier HRUs treated as ROCK with overlap table.
    4. **Landuse source** — ICIMOD vs ESA reclassification → different HRU boundaries.

    Naming: base[_landuse] where base is one of:
      'default'  — uncoupled, standard criteria
      'coupled'  — coupled with aggregated glacier HRUs
      'aspect'   — aspect in criteria (uncoupled)
      'icemelt'  — coupled + icemelt irrigation variable
    Landuse suffix appended for non-default source (e.g. 'coupled_icimod').
    """
    criteria = nml.get('criteria', [])
    coupled = nml.get('coupled', False)
    icemelt = coupled and nml.get('irrigation_variable') == 'icemelt'

    if 'aspect' in criteria:
        base = 'aspect'
    elif icemelt:
        base = 'icemelt'
    elif coupled:
        base = 'coupled'
    else:
        base = 'default'

    landuse = nml.get('landuse_source', '').upper()
    if landuse and landuse not in ('ESA', 'AUTO', ''):
        return f'{base}_{landuse.lower()}'
    return base


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
    config_key = nml.get('_config_key')

    # Metric suffix for non-default calibration metrics
    # Default (KGE) → no suffix; non-default → e.g. HBV_LogKGE/
    metric = nml.get('_calibration_metric', 'KGE')
    model_dir_name = model_type if metric == 'KGE' else f"{model_type}_{metric}"

    if config_key:
        # New composable layout: model_runs/catchment_{id}/configs/{key}/{model}/
        catchment_dir = main_dir / 'model_runs' / f'catchment_{gauge_id}'
        topo_variant = get_topo_variant(nml)

        return {
            'catchment_dir': catchment_dir,
            'topo_dir': catchment_dir / 'topo_files' / topo_variant,
            'topo_shared_dir': catchment_dir / 'topo_files',
            'data_obs_dir': catchment_dir / 'data_obs',
            'cmip6_dir': catchment_dir / 'cmip6',
            'config_dir': catchment_dir / 'configs' / config_key,
            'model_dir': catchment_dir / 'configs' / config_key / model_dir_name,
            'output_dir': catchment_dir / 'configs' / config_key / model_dir_name / 'output',
            'template_dir': catchment_dir / 'configs' / config_key / model_dir_name / 'templates',
            'results_dir': catchment_dir / 'configs' / config_key / model_dir_name / 'results',
            'plots_dir': catchment_dir / 'plots',
            'model_comparisons_dir': catchment_dir / 'model_comparisons',
            'metric_comparisons_dir': catchment_dir / 'metric_comparisons',
        }
    else:
        # Legacy layout: {config_dir}/catchment_{id}/{model}/
        config_dir_str = nml.get('config_dir', '')
        base = main_dir / config_dir_str / f'catchment_{gauge_id}'

        return {
            'catchment_dir': base,
            'topo_dir': base / 'topo_files',
            'topo_shared_dir': base / 'topo_files',
            'data_obs_dir': base / 'data_obs',
            'cmip6_dir': base / 'data_obs',
            'config_dir': base,
            'model_dir': base / model_type,
            'output_dir': base / model_type / 'output',
            'template_dir': base / model_type / 'templates',
            'results_dir': base / model_type / 'results',
            'plots_dir': base / model_type / 'output' / 'plots',
            'model_comparisons_dir': base / 'model_comparisons',
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
