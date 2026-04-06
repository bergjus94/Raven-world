"""
Composable configuration system for Raven hydrological modeling.

Merges YAML layer files in order: defaults <- env <- catchment <- configuration <- model <- overrides
Produces the same flat namelist dict that all downstream code expects.

Usage:
    from config_merge import load_config, merged_config

    # Direct usage (caller manages temp file cleanup)
    nml, tmp_path = load_config(catchment='0101', configuration='glogem', model='HBV')

    # Context manager (auto-cleanup)
    with merged_config(catchment='0101', configuration='glogem') as (nml, tmp_path):
        run_pipeline(nml, tmp_path)
"""

import socket
import tempfile
import copy
from pathlib import Path
from contextlib import contextmanager

import yaml


# ── Layer directory ──────────────────────────────────────────────────────────

LAYERS_DIR = Path(__file__).parent / 'config' / 'layers'


# ── Deep merge ───────────────────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base.

    - Dicts are merged recursively.
    - Lists are replaced (not appended).
    - Scalars are overwritten by override.
    - Keys in override that don't exist in base are added.

    Returns a new dict; neither input is modified.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ── Environment detection ────────────────────────────────────────────────────

def detect_env() -> str:
    """Auto-detect environment based on hostname."""
    hostname = socket.gethostname()
    # Server hostnames at GIUB typically contain 'giub' or the user dir exists
    if Path('/home/jberg@giub.local').exists():
        return 'server'
    return 'local'


# ── Layer loading ────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning empty dict if file doesn't exist."""
    if not path.exists():
        raise FileNotFoundError(f"Config layer not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def _list_available(subdir: str, layers_dir: Path = None) -> list:
    """List available YAML files in a layers subdirectory (without .yaml extension)."""
    d = (layers_dir or LAYERS_DIR) / subdir
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob('*.yaml'))


# ── Post-merge fixups ────────────────────────────────────────────────────────

def _apply_fixups(merged: dict) -> dict:
    """Apply post-merge fixups to ensure consistency."""

    # Auto-set tphipr_dir if precip_source is TPHiPr and not already set
    if merged.get('precip_source') == 'TPHiPr' and 'tphipr_dir' not in merged:
        main_dir = merged.get('main_dir', '')
        merged['tphipr_dir'] = f"{main_dir}/01_data/meteo/TPHiPr"

    # Sync calibration dates with top-level dates
    cali = merged.get('calibration', {})
    if 'cali_end_date' in merged and 'cali_end_date' not in cali:
        cali['cali_end_date'] = merged['cali_end_date']
    if 'end_date' in merged and 'vali_end_date' not in cali:
        cali['vali_end_date'] = merged['end_date']
    if cali:
        merged['calibration'] = cali

    # Resolve params_dir to absolute (relative to layers dir → src/config/)
    # This prevents breakage when temp file is in /tmp
    if 'params_dir' in merged:
        params = Path(merged['params_dir'])
        if not params.is_absolute():
            # params_dir is relative to the config directory
            resolved = (LAYERS_DIR.parent / params).resolve()
            if resolved.exists():
                merged['params_dir'] = str(resolved)

    return merged


# ── Main API ─────────────────────────────────────────────────────────────────

def load_config(
    catchment: str,
    configuration: str = 'baseline',
    model: str = 'HBV',
    env: str = None,
    overrides: dict = None,
    layers_dir: Path = None,
) -> tuple:
    """Load and merge YAML layers, return (merged_dict, temp_yaml_path).

    Args:
        catchment: Catchment ID (e.g. '0101')
        configuration: Configuration name (e.g. 'glogem', 'baseline')
        model: Hydrological model name (e.g. 'HBV', 'HMETS')
        env: Environment name ('local' or 'server'). Auto-detected if None.
        overrides: Optional dict of CLI overrides to apply last.
        layers_dir: Optional path to layers directory. Defaults to src/config/layers/.

    Returns:
        (merged_dict, temp_yaml_path) — the merged namelist dict and path to a
        temporary YAML file containing the same data (for preprocessors that
        expect a namelist_path).
    """
    ld = layers_dir or LAYERS_DIR

    if env is None:
        env = detect_env()

    # Load layers in merge order
    defaults = _load_yaml(ld / 'defaults.yaml')
    env_layer = _load_yaml(ld / 'env' / f'{env}.yaml')
    catchment_path = ld / 'catchments' / f'{catchment}.yaml'
    catchment_layer = _load_yaml(catchment_path) if catchment_path.exists() else {'gauge_id': catchment}
    config_layer = _load_yaml(ld / 'configurations' / f'{configuration}.yaml')
    model_layer = _load_yaml(ld / 'models' / f'{model}.yaml')

    # Merge in order: defaults <- env <- catchment <- configuration <- model
    merged = defaults
    merged = deep_merge(merged, env_layer)
    merged = deep_merge(merged, catchment_layer)
    merged = deep_merge(merged, config_layer)
    merged = deep_merge(merged, model_layer)

    # Apply CLI overrides last
    if overrides:
        merged = deep_merge(merged, overrides)

    # Post-merge fixups
    merged = _apply_fixups(merged)

    # Write merged config to temp file for preprocessors that need a path
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', prefix=f'namelist_{catchment}_{configuration}_{model}_',
        delete=False,
    )
    yaml.dump(merged, tmp, default_flow_style=False, sort_keys=False)
    tmp.close()

    return merged, Path(tmp.name)


@contextmanager
def merged_config(
    catchment: str,
    configuration: str = 'baseline',
    model: str = 'HBV',
    env: str = None,
    overrides: dict = None,
    layers_dir: Path = None,
):
    """Context manager that loads merged config and cleans up temp file on exit.

    Usage:
        with merged_config(catchment='0101', configuration='glogem') as (nml, path):
            ...
    """
    nml, tmp_path = load_config(
        catchment=catchment,
        configuration=configuration,
        model=model,
        env=env,
        overrides=overrides,
        layers_dir=layers_dir,
    )
    try:
        yield nml, tmp_path
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


# ── Registry builders (for postprocessing) ───────────────────────────────────

def load_configurations_registry(layers_dir: Path = None) -> list:
    """Scan configurations/*.yaml and build the registry list for postprocessing.

    Returns a list of dicts with keys: key, display_name, color, coupled,
    glacier_source, icemelt_mode — matching the format from configurations.yaml.
    """
    ld = layers_dir or LAYERS_DIR
    configs_dir = ld / 'configurations'
    if not configs_dir.exists():
        return []

    registry = []
    for yaml_file in sorted(configs_dir.glob('*.yaml')):
        cfg = _load_yaml(yaml_file)
        registry.append({
            'key': cfg.get('_config_key', yaml_file.stem),
            'display_name': cfg.get('display_name', yaml_file.stem),
            'color': cfg.get('color', '#333333'),
            'coupled': cfg.get('coupled', False),
            'glacier_source': cfg.get('glacier_source', 'none'),
            'icemelt_mode': cfg.get('icemelt_mode', False),
        })
    return registry


def load_catchments_registry(layers_dir: Path = None) -> list:
    """Scan catchments/*.yaml and build the registry list for postprocessing.

    Returns a list of dicts with keys: gauge_id, display_name,
    validation_start, validation_end.
    """
    ld = layers_dir or LAYERS_DIR
    catchments_dir = ld / 'catchments'
    if not catchments_dir.exists():
        return []

    registry = []
    for yaml_file in sorted(catchments_dir.glob('*.yaml')):
        c = _load_yaml(yaml_file)
        registry.append({
            'gauge_id': c.get('gauge_id', yaml_file.stem),
            'display_name': c.get('display_name', yaml_file.stem),
            'validation_start': c.get('validation_start', c.get('cali_end_date', '')),
            'validation_end': c.get('validation_end', c.get('end_date', '')),
        })
    return registry


# ── Utility ──────────────────────────────────────────────────────────────────

def list_catchments(layers_dir: Path = None) -> list:
    """List available catchment IDs."""
    return _list_available('catchments', layers_dir)


def list_configurations(layers_dir: Path = None) -> list:
    """List available configuration names."""
    return _list_available('configurations', layers_dir)


def list_models(layers_dir: Path = None) -> list:
    """List available model types."""
    return _list_available('models', layers_dir)
