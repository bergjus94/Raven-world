#!/usr/bin/env python3
"""
Complete Raven Model Calibration Workflow

This script runs the complete workflow for Raven hydrological modeling:
1. Creates all input files (preprocessing)
2. Runs SCEUA optimization/calibration
3. Generates diagnostic plots and metrics

Usage:
    python run_complete_calibration.py path/to/namelist.yaml [options]

Author: Justine Berg
Date: August 2025
"""

import sys
import argparse
import subprocess
import os
from pathlib import Path
import yaml
import logging
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


# Add the src directory to the Python path for imports
script_dir = Path(__file__).parent.absolute()
src_dir = script_dir / 'src'
sys.path.insert(0, str(src_dir))

import postprocessing_single

def setup_logging(debug=False, log_file=None):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

def load_namelist(namelist_path):
    """Load namelist configuration."""
    logger = logging.getLogger(__name__)
    
    try:
        with open(namelist_path, "r") as f:
            nml = yaml.safe_load(f)
        logger.info(f"Successfully loaded namelist from {namelist_path}")
        return nml
    except Exception as e:
        logger.error(f"Error loading namelist: {e}")
        sys.exit(1)

def validate_namelist(nml):
    """Validate that the namelist contains required fields."""
    logger = logging.getLogger(__name__)
    
    required_fields = [
        'gauge_id', 'model_type', 'start_date', 'end_date', 
        'cali_end_date', 'main_dir', 'coupled'
    ]
    
    missing_fields = [field for field in required_fields if field not in nml]
    
    if missing_fields:
        logger.error(f"Missing required fields in namelist: {missing_fields}")
        sys.exit(1)
    
    # Validate calibration configuration if present
    if 'calibration' in nml:
        calibration_config = nml['calibration']
        if 'iterations' not in calibration_config:
            logger.warning("No iterations specified in calibration config, using default (20)")
        if 'ngs' not in calibration_config:
            logger.warning("No ngs (complexes) specified in calibration config, using default (8)")
    
    logger.info("Namelist validation passed")

def run_input_creation(namelist_path, verbose=False):
    """Run the input file creation script."""
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("STEP 1: CREATING INPUT FILES")
    logger.info("="*60)
    
    # Construct command
    create_input_script = src_dir / 'create_input_Raven.py'
    cmd = [sys.executable, str(create_input_script), str(namelist_path)]
    
    if verbose:
        cmd.append('--verbose')
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the input creation script
        start_time = time.time()
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Log output
        if result.stdout:
            logger.info("Input creation output:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
        
        if result.stderr:
            logger.warning("Input creation warnings/errors:")
            for line in result.stderr.split('\n'):
                if line.strip():
                    logger.warning(f"  {line}")
        
        logger.info(f"Input file creation completed successfully in {duration:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Input file creation failed with return code {e.returncode}")
        if e.stdout:
            logger.error("STDOUT:")
            logger.error(e.stdout)
        if e.stderr:
            logger.error("STDERR:")
            logger.error(e.stderr)
        return False
    except Exception as e:
        logger.error(f"Error running input creation: {e}")
        return False

def run_calibration(namelist_path, iterations=None, ngs=None, obj_function=None):
    """Run the SCEUA calibration."""
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("STEP 2: RUNNING SCEUA CALIBRATION")
    logger.info("="*60)
    
    # Construct command - just pass the namelist file like you normally do
    calibration_script = script_dir / 'spotpy_optimize.py'
    cmd = [
        sys.executable, str(calibration_script),
        '--namelist', str(namelist_path)
    ]
    
    # Add optional overrides if provided
    if iterations is not None:
        cmd.extend(['--iterations', str(iterations)])
    
    if ngs is not None:
        cmd.extend(['--ngs', str(ngs)])
    
    if obj_function is not None:
        cmd.extend(['--obj_function', obj_function])
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the calibration script
        start_time = time.time()
        
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Log output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(f"CALIBRATION: {output.strip()}")
        
        # Wait for process to complete
        return_code = process.poll()
        
        end_time = time.time()
        duration = end_time - start_time
        
        if return_code == 0:
            logger.info(f"Calibration completed successfully in {duration/60:.1f} minutes")
            return True
        else:
            logger.error(f"Calibration failed with return code {return_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error running calibration: {e}")
        return False

def run_postprocessing(namelist_path, validation_start=None, validation_end=None):
    """
    Run postprocessing analysis using the postprocessing_single module.
    
    Parameters:
    -----------
    namelist_path : str or Path
        Path to the namelist YAML file
    validation_start : str, optional
        Start date for validation period (defaults to cali_end_date from namelist)
    validation_end : str, optional
        End date for validation period (defaults to end_date from namelist)
        
    Returns:
    --------
    dict or None
        Dictionary containing postprocessing results, or None if failed
    """
    try:
        # Load namelist
        with open(namelist_path, 'r') as f:
            nml = yaml.safe_load(f)
        
        # Extract configuration
        gauge_id = str(nml['gauge_id'])
        model_type = nml.get('model_type', 'HBV')
        main_dir = nml['main_dir']
        config_dir = nml.get('config_dir', '02_model_setups')
        
        # Use validation dates from arguments or namelist
        if validation_start is None:
            validation_start = nml.get('cali_end_date', nml.get('start_date'))
        if validation_end is None:
            validation_end = nml.get('end_date')
        
        # Create configuration dictionary for postprocessing
        config = {
            'main_dir': main_dir,
            'config_dir': config_dir,
            'gauge_id': gauge_id,
            'model_type': model_type,
            'start_date': nml.get('start_date', '2000-01-01'),
            'end_date': nml.get('end_date', '2020-12-31'),
            'cali_start_date': nml.get('cali_start_date', nml.get('start_date', '2000-01-01')),
            'cali_end_date': nml.get('cali_end_date', '2009-12-31'),
            'coupled': nml.get('coupled', False),
            'raven_executable': nml.get('raven_executable', '/home/jberg/OneDrive/RavenHydro/build/Raven'),
            'glogem_dir': nml.get('glogem_dir', None)
        }
        
        logging.info(f"Running postprocessing for catchment {gauge_id}")
        logging.info(f"  Model type: {model_type}")
        logging.info(f"  Config directory: {config_dir}")
        logging.info(f"  Validation period: {validation_start} to {validation_end}")
        logging.info(f"  Raven executable: {config['raven_executable']}")
        
        # Run complete postprocessing
        results = postprocessing_single.run_complete_postprocessing(
            config=config,
            validation_start=validation_start,
            validation_end=validation_end
        )
        
        if results is None:
            logging.error("Postprocessing returned None - analysis failed")
            return None
        
        # Check if postprocessing was successful
        success_count = sum(1 for success in results['success'].values() if success is True)
        total_analyses = len([k for k, v in results['success'].items() if v is not None])
        
        logging.info(f"Postprocessing completed: {success_count}/{total_analyses} analyses successful")
        
        if results.get('errors'):
            logging.warning(f"Postprocessing encountered {len(results['errors'])} errors:")
            for analysis, error in results['errors'].items():
                logging.warning(f"  - {analysis}: {error}")
        
        return results
        
    except Exception as e:
        logging.error(f"Postprocessing failed with error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def print_summary(nml, success_steps, total_duration):
    """Print a summary of the workflow."""
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("WORKFLOW SUMMARY")
    logger.info("="*80)
    
    logger.info(f"Gauge ID: {nml['gauge_id']}")
    logger.info(f"Model Type: {nml['model_type']}")
    logger.info(f"Coupled Mode: {nml.get('coupled', False)}")
    logger.info(f"Calibration Period: {nml['start_date']} to {nml['cali_end_date']}")
    logger.info(f"Validation Period: {nml['cali_end_date']} to {nml['end_date']}")
    
    # Print step results
    steps = [
        "Input File Creation",
        "SCEUA Calibration",
        "Postprocessing Analysis"
    ]
    
    logger.info(f"\nStep Results:")
    for i, step in enumerate(steps):
        status = "✓ SUCCESS" if i < len(success_steps) and success_steps[i] else "✗ FAILED"
        logger.info(f"  {i+1}. {step}: {status}")
    
    logger.info(f"\nTotal Workflow Duration: {total_duration/60:.1f} minutes")
    
    # Calculate model directory using config_dir
    config_dir = nml.get('config_dir', '02_model_setups')
    
    model_dir = Path(nml['main_dir']) / config_dir / f"catchment_{nml['gauge_id']}" / nml['model_type']
    output_dir = model_dir / 'output'
    
    if all(success_steps):
        logger.info(f"\n🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
        logger.info(f"Results available in: {output_dir}")
        logger.info(f"Key files to check:")
        logger.info(f"  - Calibration results: {output_dir}/calibration_results_*.csv")
        logger.info(f"  - Best parameters: {output_dir}/*_best_params.csv")
        logger.info(f"  - Model metrics: {output_dir}/*_metrics.csv")
        logger.info(f"  - Diagnostic plots: {output_dir}/plots/")
    else:
        logger.error(f"\n❌ WORKFLOW FAILED!")
        failed_step = next((i for i, success in enumerate(success_steps) if not success), None)
        if failed_step is not None:
            logger.error(f"Failed at step {failed_step + 1}: {steps[failed_step]}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run complete Raven model calibration workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_complete_calibration.py namelist.yaml
  
  # With custom calibration parameters
  python run_complete_calibration.py namelist.yaml --iterations 50 --ngs 10
  
  # Verbose output
  python run_complete_calibration.py namelist.yaml --verbose
  
  # Skip input creation (if files already exist)
  python run_complete_calibration.py namelist.yaml --skip-input-creation
        """
    )
    
    # Required arguments
    parser.add_argument(
        'namelist',
        type=str,
        help='Path to namelist.yaml configuration file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--iterations',
        type=int,
        help='Number of calibration iterations (overrides namelist)'
    )
    parser.add_argument(
        '--ngs',
        type=int,
        help='Number of complexes for SCEUA (overrides namelist)'
    )
    parser.add_argument(
        '--obj-function',
        type=str,
        help='Objective function (KGE, NSE, etc.) (overrides namelist)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose debug output'
    )
    parser.add_argument(
        '--skip-input-creation',
        action='store_true',
        help='Skip input file creation (assume files already exist)'
    )
    parser.add_argument(
        '--skip-diagnostics',
        action='store_true',
        help='Skip final diagnostics generation'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file (default: auto-generated)'
    )

    parser.add_argument(
        '--skip-postprocessing',
        action='store_true',
        help='Skip postprocessing analysis after calibration'
    )
    parser.add_argument(
        '--validation-start',
        type=str,
        help='Start date for validation period in postprocessing (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--validation-end',
        type=str,
        help='End date for validation period in postprocessing (YYYY-MM-DD)'
    )
    
    args = parser.parse_args()
    
    # Check if namelist file exists
    namelist_path = Path(args.namelist)
    if not namelist_path.exists():
        print(f"Error: Namelist file {namelist_path} not found")
        sys.exit(1)
    
    # Load and validate namelist
    nml = load_namelist(namelist_path)
    validate_namelist(nml)
    
    # Setup logging
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        # Auto-generate log file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(f"calibration_{nml['gauge_id']}_{nml['model_type']}_{timestamp}.log")
    
    logger = setup_logging(args.verbose, log_file)
    
    # Print initial information
    logger.info("="*80)
    logger.info("RAVEN COMPLETE CALIBRATION WORKFLOW")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now()}")
    logger.info(f"Namelist: {namelist_path}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Gauge ID: {nml['gauge_id']}")
    logger.info(f"Model Type: {nml['model_type']}")
    logger.info(f"Coupled: {nml.get('coupled', False)}")
    
    # Track timing and success
    workflow_start = time.time()
    success_steps = []
    
    try:
        # Step 1: Create input files (optional)
        if not args.skip_input_creation:
            step1_success = run_input_creation(namelist_path, args.verbose)
            success_steps.append(step1_success)
            
            if not step1_success:
                logger.error("Input file creation failed. Stopping workflow.")
                sys.exit(1)
        else:
            logger.info("Skipping input file creation as requested")
            success_steps.append(True)  # Assume success
        
        # Step 2: Run calibration
        step2_success = run_calibration(
            namelist_path, 
            args.iterations, 
            args.ngs, 
            args.obj_function
        )
        success_steps.append(step2_success)
        
        if not step2_success:
            logger.error("Calibration failed. Stopping workflow.")
            print_summary(nml, success_steps, time.time() - workflow_start)
            sys.exit(1)
        
        # Step 3: Run postprocessing (NEW)
        if not args.skip_postprocessing:
            step3_success = run_postprocessing(
                namelist_path,
                validation_start=args.validation_start,
                validation_end=args.validation_end
            )
            success_steps.append(step3_success)
            
            if not step3_success:
                logger.warning("Postprocessing failed, but continuing...")
        else:
            logger.info("Skipping postprocessing as requested")
            success_steps.append(True)  # Assume success



        # Calculate total duration
        total_duration = time.time() - workflow_start
        
        # Print summary
        print_summary(nml, success_steps, total_duration)
        
        # Exit with appropriate code
        if all(success_steps):
            logger.info("Workflow completed successfully!")
            sys.exit(0)
        else:
            logger.error("Workflow completed with errors!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("Workflow interrupted by user")
        print_summary(nml, success_steps, time.time() - workflow_start)
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print_summary(nml, success_steps, time.time() - workflow_start)
        sys.exit(1)

if __name__ == "__main__":
    main()