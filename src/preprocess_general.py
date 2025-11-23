from pathlib import Path
import os
import yaml


def create_folder(namelist_path):
    """
    Creates folder structure for Raven model setup using a namelist file.

    Parameters:
    -----------
    namelist_path : Path or str
        Path to the namelist YAML or JSON file containing model setup info.
    """
    # Load namelist
    with open(namelist_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract variables
    main_dir = Path(config.get('main_dir'))
    model_dir = main_dir / config.get('config_dir')
    gauge_id = config.get('gauge_id')
    model_type = config.get('model_type')
    scenario_id = config.get('scenario_id', None)

    # Base catchment folder
    folder_path = Path(model_dir, f'catchment_{gauge_id}')
    os.makedirs(folder_path, exist_ok=True)

    # Topography files folder (common for all scenarios)
    folder_path = Path(model_dir, f'catchment_{gauge_id}', 'topo_files')
    os.makedirs(folder_path, exist_ok=True)

    if scenario_id:
        # Create scenario-specific folder structure
        folder_path = Path(model_dir, f'catchment_{gauge_id}_{scenario_id}')
        os.makedirs(folder_path, exist_ok=True)

        folder_path = Path(model_dir, f'catchment_{gauge_id}_{scenario_id}', f'{model_type}')
        os.makedirs(folder_path, exist_ok=True)

        folder_path = Path(model_dir, f'catchment_{gauge_id}_{scenario_id}', f'{model_type}', 'templates')
        os.makedirs(folder_path, exist_ok=True)

        folder_path = Path(model_dir, f'catchment_{gauge_id}_{scenario_id}', f'{model_type}', 'data_obs')
        os.makedirs(folder_path, exist_ok=True)

        folder_path = Path(model_dir, f'catchment_{gauge_id}_{scenario_id}', f'{model_type}', 'output')
        os.makedirs(folder_path, exist_ok=True)
    else:
        # Create standard folder structure (as before)
        folder_path = Path(model_dir, f'catchment_{gauge_id}', f'{model_type}')
        os.makedirs(folder_path, exist_ok=True)

        folder_path = Path(model_dir, f'catchment_{gauge_id}', f'{model_type}', 'templates')
        os.makedirs(folder_path, exist_ok=True)

        folder_path = Path(model_dir, f'catchment_{gauge_id}', f'{model_type}', 'data_obs')
        os.makedirs(folder_path, exist_ok=True)

        folder_path = Path(model_dir, f'catchment_{gauge_id}', f'{model_type}', 'output')
        os.makedirs(folder_path, exist_ok=True)