from pathlib import Path
import os
import yaml
from paths import get_paths, get_topo_variant


def create_folder(namelist_path):
    """
    Creates folder structure for Raven model setup using a namelist file.

    New layout (catchment at top, shared data, config-specific model dirs):
        catchment_{gauge_id}/
            topo_files/                  # shared across all configs
            topo_files/{variant}/        # variant-specific HRU files
            data_obs/                    # shared meteo, streamflow, irrigation
            cmip6/                       # shared future climate forcing
            plots/                       # shared diagnostic plots
            configs/{config}/{model}/    # config × model specific
                templates/
                output/

    Parameters:
    -----------
    namelist_path : Path or str
        Path to the namelist YAML file containing model setup info.
    """
    # Load namelist
    with open(namelist_path, 'r') as f:
        config = yaml.safe_load(f)

    paths = get_paths(config)

    # Shared directories (per catchment)
    os.makedirs(paths['catchment_dir'], exist_ok=True)
    os.makedirs(paths['topo_shared_dir'], exist_ok=True)
    os.makedirs(paths['topo_dir'], exist_ok=True)
    os.makedirs(paths['data_obs_dir'], exist_ok=True)
    os.makedirs(paths['cmip6_dir'], exist_ok=True)
    os.makedirs(paths['plots_dir'], exist_ok=True)

    # Config × model specific directories
    os.makedirs(paths['model_dir'], exist_ok=True)
    os.makedirs(paths['template_dir'], exist_ok=True)
    os.makedirs(paths['output_dir'], exist_ok=True)
