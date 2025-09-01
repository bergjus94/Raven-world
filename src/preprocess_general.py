from pathlib import Path
import os


def create_folder(model_dir, gauge_id, model_type, scenario_id=None):
    """
    Creates folder structure for Raven model setup.
    
    Parameters:
    -----------
    model_dir : Path or str
        Base directory for model setups
    gauge_id : str or int
        ID of the gauge/catchment
    model_type : str
        Type of the model (e.g., 'HBV')
    scenario_id : str, optional
        ID of the climate scenario. If provided, creates scenario-specific structure
    """
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