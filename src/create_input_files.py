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
import matplotlib
matplotlib.use('Agg')  # Prevent plots from showing in image viewer

# Import preprocessing modules
import preprocess_general
import preprocess_meteo
from preprocess_meteo import (
    ERA5LandAnalyzer,
    GridWeightsGenerator,
    HARAnalyzer,
    HARGridWeightsGenerator,
    process_tphipr_precipitation,
)
from preprocess_catchment import (
    CatchmentProcessor,
    HRUConnectivityCalculator,
    MultiSubbasinProcessor,
    MultiSubbasinConnectivityCalculator,
)
from preprocess_streamflow import StreamflowProcessor
from preprocess_glogem import GloGEMProcessor, MultiSubbasinGloGEMProcessor

# Import model-specific processors
from preprocess_HBV import HBVProcessor
from preprocess_HYMOD import HYMODPreprocessor
from preprocess_MOHYSE import MOHYSEPreprocessor
from preprocess_HMETS import HMETSPreprocessor


#--------------------------------------------------------------------------------
################################# helper functions ##############################
#--------------------------------------------------------------------------------

def get_model_processor(model_type: str, namelist_path: str):
    """
    Get the appropriate model processor based on model type.
    
    Parameters
    ----------
    model_type : str
        Model type (HBV, HYMOD, MOHYSE, HMETS)
    namelist_path : str
        Path to namelist YAML file
        
    Returns
    -------
    processor
        Model-specific processor instance
    """
    processors = {
        'HBV': HBVProcessor,
        'HYMOD': HYMODPreprocessor,
        'MOHYSE': MOHYSEPreprocessor,
        'HMETS': HMETSPreprocessor,
    }
    
    model_type_upper = model_type.upper()
    
    if model_type_upper not in processors:
        raise ValueError(f"Unknown model type: {model_type}. Supported: {list(processors.keys())}")
    
    return processors[model_type_upper](namelist_path)


def process_meteorological_data(namelist_path: str, meteo_source: str, debug: bool = False):
    """
    Process meteorological data based on the specified source.
    
    Parameters
    ----------
    namelist_path : str
        Path to namelist YAML file
    meteo_source : str
        Meteorological data source ('ERA5Land' or 'HAR')
    debug : bool
        Enable debug mode
        
    Returns
    -------
    dict
        Dictionary with processing results
    """
    results = {}
    
    meteo_source_upper = meteo_source.upper()
    
    if meteo_source_upper == 'ERA5LAND' or meteo_source_upper == 'ERA5_LAND' or meteo_source_upper == 'ERA5':
        print("🌍 Processing ERA5-Land meteorological data...")
        
        # Initialize ERA5Land analyzer
        analyzer = ERA5LandAnalyzer(namelist_path)
        
        # Debug precipitation data
        file_info = analyzer.get_processed_files_info()
        if 'precipitation' in file_info:
            precip_file = file_info['precipitation']
            if debug:
                analyzer.debug_precipitation_values(precip_file)
        else:
            print("   ⚠️ No precipitation file found")
        
        results['analyzer'] = analyzer
        results['source'] = 'ERA5Land'
        
        # Generate grid weights for ERA5Land
        print("🧮 Generating ERA5-Land grid weights...")
        generator = GridWeightsGenerator(namelist_path)
        grid_weights = generator.generate()
        results['grid_weights'] = grid_weights
        
    elif meteo_source_upper == 'HAR':
        print("🏔️ Processing HAR meteorological data...")
        
        # Initialize HAR analyzer
        har_analyzer = HARAnalyzer(namelist_path)
        results['analyzer'] = har_analyzer
        results['source'] = 'HAR'
        
        # Generate grid weights for HAR
        print("🧮 Generating HAR grid weights...")
        har_gw = HARGridWeightsGenerator(namelist_path)
        har_gw.generate()
        results['grid_weights'] = har_gw
        
    else:
        raise ValueError(f"Unknown meteo_source: {meteo_source}. Supported: 'ERA5Land', 'HAR'")
    
    print(f"   ✅ Meteorological data processing completed ({results['source']})")
    
    return results


def process_model_files(namelist_path: str, model_type: str):
    """
    Process model-specific input files.
    
    Parameters
    ----------
    namelist_path : str
        Path to namelist YAML file
    model_type : str
        Model type (HBV, HYMOD, MOHYSE, HMETS)
    """
    print(f"📝 Processing {model_type.upper()} model files...")
    
    try:
        processor = get_model_processor(model_type, namelist_path)
        
        # Generate regular files
        print(f"   📄 Creating regular {model_type.upper()} files...")
        processor.create_all_files(template=False)
        
        # Generate template files for optimization
        print(f"   📋 Creating template files for optimization...")
        processor.create_all_files(template=True)
        
        print(f"   ✅ {model_type.upper()} files created successfully!")
        
    except Exception as e:
        print(f"   ❌ Error processing {model_type.upper()} files: {e}")
        traceback.print_exc()
        raise


#--------------------------------------------------------------------------------
################################# main function #################################
#--------------------------------------------------------------------------------

def main(namelist_path: str, force_reprocess: bool = False):
    """
    Main function to create all Raven input files.
    
    Parameters
    ----------
    namelist_path : str
        Path to namelist YAML file
    force_reprocess : bool
        Force reprocessing of existing files
    """
    print("=" * 70)
    print("🚀 RAVEN INPUT FILE GENERATOR")
    print("=" * 70)
    
    # Read namelist
    print(f"\n📖 Reading namelist: {namelist_path}")
    with open(namelist_path, "r") as f:
        nml = yaml.safe_load(f)
    
    # Extract key configuration variables
    gauge_id = nml.get("gauge_id", "unknown")
    model_type = nml.get("model_type", "HBV")
    meteo_source = nml.get("meteo_source", "ERA5Land")
    coupled = nml.get("coupled", False)
    debug = nml.get("debug", False)
    
    print(f"\n📊 Configuration:")
    print(f"   - Gauge ID: {gauge_id}")
    print(f"   - Model type: {model_type}")
    print(f"   - Meteo source: {meteo_source}")
    print(f"   - Coupled mode: {coupled}")
    print(f"   - Debug mode: {debug}")
    print("-" * 70)

    # =========================================================================
    # STEP 1: Create folder structure
    # =========================================================================
    print("\n📁 STEP 1: Creating folder structure...")
    try:
        preprocess_general.create_folder(namelist_path)
        print("   ✅ Folder structure created")
    except Exception as e:
        print(f"   ❌ Error creating folders: {e}")
        traceback.print_exc()
        return False

    # =========================================================================
    # STEP 2: Process streamflow data
    # =========================================================================
    print("\n💧 STEP 2: Processing streamflow data...")
    try:
        if nml.get('subbasins'):
            # Multi-subbasin: process one RVT file per gauged subbasin
            for sb in nml['subbasins']:
                if sb.get('gauged', 0):
                    sb_gauge_id = str(sb['gauge_id'])
                    sb_id = sb['id']
                    print(f"   🏔️ Processing streamflow for subbasin {sb_id} (gauge {sb_gauge_id})...")
                    proc = StreamflowProcessor(namelist_path,
                                               basin_id=sb_id,
                                               output_filename=f"Q_daily_{sb_gauge_id}.rvt")
                    # Override so processor reads the subbasin's own streamflow file
                    proc.gauge_id = sb_gauge_id
                    proc.stream_file = proc.main_dir / proc.stream_dir_template.format(gauge_id=sb_gauge_id)
                    proc.plot_file = proc.plots_dir / f"streamflow_timeseries_gauge_{sb_gauge_id}.png"
                    success = proc.process()
                    if success:
                        print(f"   ✅ Streamflow processed for subbasin {sb_id} (gauge {sb_gauge_id})")
                    else:
                        print(f"   ⚠️ Streamflow processing completed with warnings for subbasin {sb_id}")
        else:
            processor_stream = StreamflowProcessor(namelist_path)
            success_stream = processor_stream.process()
            if success_stream:
                print("   ✅ Streamflow data processed")
            else:
                print("   ⚠️ Streamflow processing completed with warnings")
    except Exception as e:
        print(f"   ❌ Error processing streamflow: {e}")
        traceback.print_exc()

    # =========================================================================
    # STEP 3: Process catchment and create HRUs
    # =========================================================================
    print("\n🗺️ STEP 3: Processing catchment and creating HRUs...")
    try:
        if nml.get('subbasins'):
            # Multi-subbasin path
            print("   🌐 Multi-subbasin mode detected")
            multi_processor = MultiSubbasinProcessor(namelist_path)
            hru_table = multi_processor.process_all_subbasins()
            print(f"   ✅ Created {len(hru_table)} HRUs across {len(nml['subbasins'])} subbasins")
        else:
            # Single-basin path (existing behaviour)
            processor_catchment = CatchmentProcessor(namelist_path)

            if debug:
                print(f"   📊 Catchment configuration:")
                print(f"      - Criteria: {processor_catchment.criteria}")
                print(f"      - Elevation distance: {processor_catchment.elevation_distance} m")

            hru_table = processor_catchment.process_catchment()

            print(f"   ✅ Created {len(hru_table)} HRUs")
            print(f"   📁 Files saved in: {processor_catchment.get_path('')}")

    except Exception as e:
        print(f"   ❌ Error during catchment processing: {e}")
        traceback.print_exc()
        return False

    # =========================================================================
    # STEP 4: Calculate HRU connectivity
    # =========================================================================
    print("\n🔗 STEP 4: Calculating HRU connectivity...")
    try:
        if nml.get('subbasins'):
            # Multi-subbasin: compute per-subbasin connectivity, then merge
            print("   🌐 Multi-subbasin mode: computing per-subbasin connectivity")
            multi_conn = MultiSubbasinConnectivityCalculator(namelist_path)
            connections = multi_conn.calculate_connectivity()
            print(f"   ✅ HRU connectivity calculated ({len(connections)} total connections across all subbasins)")
        else:
            connectivity_calc = HRUConnectivityCalculator(namelist_path)
            connectivity_df = connectivity_calc.calculate_connectivity()
            print(f"   ✅ HRU connectivity calculated ({len(connectivity_df)} connections)")
    except Exception as e:
        print(f"   ❌ Error calculating connectivity: {e}")
        traceback.print_exc()

    # =========================================================================
    # STEP 5: Process meteorological data (ERA5Land or HAR)
    # =========================================================================
    print(f"\n🌡️ STEP 5: Processing meteorological data ({meteo_source})...")
    try:
        meteo_results = process_meteorological_data(namelist_path, meteo_source, debug=debug)
        print(f"   ✅ {meteo_results['source']} data processed")
    except Exception as e:
        print(f"   ❌ Error processing meteorological data: {e}")
        traceback.print_exc()
        return False

    # Optional: TPHiPr precipitation override
    if nml.get('precip_source', '').upper() == 'TPHIPR':
        print("\n🌧️ STEP 5b: Processing TPHiPr precipitation data...")
        try:
            process_tphipr_precipitation(namelist_path)
        except Exception as e:
            print(f"   ❌ Error processing TPHiPr data: {e}")
            traceback.print_exc()
            return False

    # Optional: CORDEX climate projection downscaling (future=True)
    if nml.get('future', False) and nml.get('cordex_models'):
        print("\n🌍 STEP 5c: Downscaling CORDEX climate projections (QDM)...")
        try:
            from preprocess_climate import process_cordex_climate
            process_cordex_climate(namelist_path, force_reprocess=force_reprocess)
            print("   ✅ CORDEX downscaling completed!")
        except Exception as e:
            print(f"   ❌ Error in CORDEX downscaling: {e}")
            traceback.print_exc()
            return False

    # Optional: CMIP6 climate projection downscaling (future=True)
    if nml.get('future', False) and nml.get('cmip6_models'):
        print("\n🌍 STEP 5d: Downscaling CMIP6 climate projections (QDM)...")
        try:
            from preprocess_climate import process_cmip6_climate
            process_cmip6_climate(namelist_path, force_reprocess=force_reprocess)
            print("   ✅ CMIP6 downscaling completed!")
        except Exception as e:
            print(f"   ❌ Error in CMIP6 downscaling: {e}")
            traceback.print_exc()
            return False

    # Compute per-subbasin monthly T/PET averages (multi-subbasin only)
    if nml.get('subbasins'):
        print("   🌡️ Computing per-subbasin monthly T/PET averages...")
        try:
            analyzer = meteo_results['analyzer']
            main_dir = Path(nml.get('main_dir', ''))
            config_dir = nml.get('config_dir', '')
            shared_data_dir = main_dir / config_dir / f"catchment_{gauge_id}" / 'data_obs'
            for sb in nml['subbasins']:
                sb_data_dir = shared_data_dir / f"subbasin_{sb['id']}"
                ok = analyzer.compute_monthly_averages_for_subbasin(str(sb['gauge_id']), sb_data_dir)
                status = '✅' if ok else '⚠️'
                print(f"   {status} Monthly T/PET subbasin {sb['id']} (gauge {sb['gauge_id']})")
        except Exception as e:
            print(f"   ❌ Error computing per-subbasin monthly averages: {e}")
            traceback.print_exc()

    # =========================================================================
    # STEP 6: Process GloGEM glacier data (only if coupled mode)
    # =========================================================================
    if coupled:
        print("\n❄️ STEP 6: Processing GloGEM glacier data...")
        try:
            if nml.get('subbasins'):
                print("   🌐 Multi-subbasin mode: using MultiSubbasinGloGEMProcessor")
                glogem_processor = MultiSubbasinGloGEMProcessor(namelist_path)
            else:
                glogem_processor = GloGEMProcessor(namelist_path)

            print(f"   📁 GloGEM shared_data_dir: {glogem_processor.shared_data_dir}")
            print(f"   📁 GloGEM model_dir: {glogem_processor.model_dir}")
            print(f"   📁 GloGEM glogem_dir: {glogem_processor.glogem_dir}")

            results_glogem = glogem_processor.process_all(force_reprocess=force_reprocess)

            if not results_glogem.get('skipped', False):
                print(f"   ✅ GloGEM processing completed!")
                if 'n_subbasins' in results_glogem:
                    print(f"      - Subbasins processed: {results_glogem['n_subbasins']}")
                else:
                    print(f"      - GloGEM records: {len(results_glogem['glogem_data'])}")
                    print(f"      - Glaciers matched: {len(results_glogem['validation']['matched'])}")
                    print(f"      - Missing in GloGEM: {len(results_glogem['validation']['missing_in_glogem'])}")
            else:
                print(f"   ⏭️ GloGEM files already exist (skipped)")

            # Verify irrigation files were created
            irrigation_nc = glogem_processor.shared_data_dir / 'irrigation.nc'
            irrigation_gw = glogem_processor.shared_data_dir / 'GridWeights_Irrigation.txt'
            if irrigation_nc.exists():
                print(f"   ✅ irrigation.nc exists: {irrigation_nc} ({irrigation_nc.stat().st_size/1e6:.1f} MB)")
            else:
                print(f"   ❌ irrigation.nc NOT FOUND at: {irrigation_nc}")
            if irrigation_gw.exists():
                print(f"   ✅ GridWeights_Irrigation.txt exists: {irrigation_gw}")
            else:
                print(f"   ❌ GridWeights_Irrigation.txt NOT FOUND at: {irrigation_gw}")

        except Exception as e:
            print(f"   ❌ Error processing GloGEM data: {e}")
            traceback.print_exc()
    else:
        print("\n⏭️ STEP 6: Skipping GloGEM processing (coupled=False)")

    # =========================================================================
    # STEP 7: Process model-specific files
    # =========================================================================
    print(f"\n📝 STEP 7: Processing {model_type.upper()} model files...")
    try:
        process_model_files(namelist_path, model_type)
    except Exception as e:
        print(f"   ❌ Error processing model files: {e}")
        traceback.print_exc()
        return False

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("🎉 ALL INPUT FILES CREATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   - Gauge ID: {gauge_id}")
    print(f"   - Model type: {model_type}")
    print(f"   - Meteo source: {meteo_source}")
    print(f"   - Coupled mode: {coupled}")
    print(f"   - Number of HRUs: {len(hru_table)}")
    
    # Print output directory
    with open(namelist_path, "r") as f:
        nml = yaml.safe_load(f)
    main_dir = Path(nml.get('main_dir', ''))
    config_dir = nml.get('config_dir', '')
    output_dir = main_dir / config_dir / f"catchment_{gauge_id}" / model_type
    
    print(f"\n📁 Output directory:")
    print(f"   {output_dir}")
    print("\n" + "=" * 70)
    
    return True


#--------------------------------------------------------------------------------
################################# CLI interface #################################
#--------------------------------------------------------------------------------

def print_usage():
    """Print usage information"""
    print("""
Usage: python create_input_files.py <namelist_path> [options]

Arguments:
    namelist_path       Path to the namelist YAML file

Options:
    --force, -f         Force reprocessing of existing files
    --help, -h          Show this help message

Example:
    python create_input_files.py ../namelist_0102.yaml
    python create_input_files.py ../namelist_0102.yaml --force

Supported configurations:
    meteo_source:   'ERA5Land', 'HAR'
    model_type:     'HBV', 'HYMOD', 'MOHYSE', 'HMETS'
    coupled:        true/false (enables GloGEM glacier processing)
""")


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    # Check for help flag
    if sys.argv[1] in ['--help', '-h']:
        print_usage()
        sys.exit(0)
    
    namelist_path = sys.argv[1]
    
    # Check for force flag
    force_reprocess = False
    if len(sys.argv) > 2:
        if sys.argv[2] in ['--force', '-f']:
            force_reprocess = True
            print("🔄 Force reprocessing enabled")
    
    # Check if namelist exists
    if not Path(namelist_path).exists():
        print(f"❌ Error: Namelist file not found: {namelist_path}")
        sys.exit(1)
    
    # Run main function
    try:
        success = main(namelist_path, force_reprocess=force_reprocess)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)