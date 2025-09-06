#### This file creates all input files for the Raven model
#### All functions are run with the namelist.yaml file
#### Justine Berg

#--------------------------------------------------------------------------------
############################### import packages #################################
#--------------------------------------------------------------------------------

import sys
import traceback
from pathlib import Path
import yaml

# Import your modules
import preprocess_general
import preprocess_meteo
from preprocess_meteo import ERA5LandAnalyzer, GridWeightsGenerator
from preprocess_catchment import CatchmentProcessor, HRUConnectivityCalculator
from preprocess_streamflow import StreamflowProcessor
import preprocess_rvx
from preprocess_glogem import GloGEMProcessor
from preprocess_HBV import HBVProcessor


#--------------------------------------------------------------------------------
################################# create files ##################################
#--------------------------------------------------------------------------------

def main(namelist_path):
    print(f"🔄 Reading namelist: {namelist_path}")
    with open(namelist_path, "r") as f:
        nml = yaml.safe_load(f)

    # Extract variables
    gauge_id = nml["gauge_id"]
    coupled = nml["coupled"]
    main_dir = nml["main_dir"]
    model_dirs = nml["model_dirs"]
    model_type = nml["model_type"]
    params_dir = nml["params_dir"]
    start_date = nml["start_date"]
    end_date = nml["end_date"]
    cali_end_date = nml["cali_end_date"]

    if coupled:
        model_dir = Path(main_dir, model_dirs["coupled"])
    else:
        model_dir = Path(main_dir, model_dirs["uncoupled"])

    # 1. Create folder structure
    print("📁 Creating folder structure...")
    preprocess_general.create_folder(namelist_path)

    # 2. Debug precipitation data
    print("🔎 Debugging precipitation data...")
    analyzer = ERA5LandAnalyzer(namelist_path)
    file_info = analyzer.get_processed_files_info()
    if 'precipitation' in file_info:
        precip_file = file_info['precipitation']
        analyzer.debug_precipitation_values(precip_file)
    else:
        print("No precipitation file found")

    # 3. Process streamflow
    print("💧 Processing streamflow...")
    processor_stream = StreamflowProcessor(namelist_path)
    success_stream = processor_stream.process()

    # 4. Process catchment
    print("🗺️ Processing catchment...")
    processor_catchment = CatchmentProcessor(namelist_path)
    print(f"   - Gauge ID: {processor_catchment.gauge_id}")
    print(f"   - Model directory: {processor_catchment.model_dir}")
    print(f"   - Criteria: {processor_catchment.criteria}")
    print(f"   - Elevation distance: {processor_catchment.elevation_distance} m")
    print(f"   - Debug mode: {processor_catchment.debug}")
    try:
        hru_table = processor_catchment.process_catchment()
        print(f"🎉 Catchment processing completed successfully!")
        print(f"📋 Created HRU table with {len(hru_table)} HRUs")
        print(f"📁 Files saved in: {processor_catchment.get_path('')}")
    except Exception as e:
        print(f"❌ Error during catchment processing: {e}")
        traceback.print_exc()

    # 5. Calculate HRU connectivity
    print("🔗 Calculating HRU connectivity...")
    connectivity_calc = HRUConnectivityCalculator(namelist_path)
    connectivity_df = connectivity_calc.calculate_connectivity()

    # 6. Generate grid weights
    print("🧮 Generating grid weights...")
    generator = GridWeightsGenerator(namelist_path)
    grid_weights = generator.generate()

    # 7. Process GloGEM glacier data
    print("❄️ Processing GloGEM glacier data...")
    glogem_processor = GloGEMProcessor(namelist_path)
    results_glogem = glogem_processor.prepare_coupled_forcing()
    print(f"🎉 GloGEM processing completed successfully!")
    print(f"   - Precipitation data shape: {results_glogem['precipitation'].shape}")
    print(f"   - Temperature data files: {len(results_glogem['temperature'])}")
    print(f"📁 Files created in: {glogem_processor.model_dir}/catchment_{glogem_processor.gauge_id}/{glogem_processor.model_type}/data_obs/")

    # 8. Load default parameters
    print("⚙️ Loading default parameters...")
    with open(params_dir, "r") as f:
        default_params = yaml.load(f, Loader=yaml.FullLoader)

    # 9. Process HBV files
    print("📝 Processing HBV files...")
    processor_hbv = HBVProcessor(namelist_path)
    processor_hbv.process_all_files(template=False)  # Regular files
    processor_hbv.process_all_files(template=True)   # Template files for optimization

    print("✅ All input files created successfully!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_input_files.py /path/to/namelist.yaml")
        sys.exit(1)
    namelist_path = sys.argv[1]
    main(namelist_path)