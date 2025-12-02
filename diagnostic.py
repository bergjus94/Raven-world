import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import HydroErr as hr
import hydroeval as he


class ModelDiagnostics:
    """
    A class for calculating and analyzing calibration and validation metrics
    for hydrological model streamflow outputs.
    """
    
    def __init__(self, namelist_file=None, gauge_id=None, model_type=None, 
                cali_end_date=None, vali_end_date=None, coupled=None, base_dir=None, model_dir=None):
        """
        Initialize the ModelDiagnostics object.
        """
        # Read settings from namelist if provided
        self.namelist = None
        if namelist_file:
            self.read_namelist(namelist_file)
            
            # Use values from namelist if not explicitly provided
            if gauge_id is None:
                gauge_id = self.namelist.get('gauge_id')
            if model_type is None:
                model_type = self.namelist.get('model_type')
            if cali_end_date is None:
                cali_end_date = self.namelist.get('cali_end_date')
            if vali_end_date is None:
                vali_end_date = self.namelist.get('end_date')
            if coupled is None:
                coupled = self.namelist.get('coupled', False)
            if base_dir is None:
                base_dir = self.namelist.get('main_dir')
        
        # Store parameters
        self.gauge_id = str(gauge_id) if gauge_id is not None else None
        self.model_type = model_type
        self.cali_end_date = pd.to_datetime(cali_end_date) if cali_end_date else None
        self.vali_end_date = pd.to_datetime(vali_end_date) if vali_end_date else None
        self.coupled = coupled if coupled is not None else False
        
        # Check for required parameters
        if not all([self.gauge_id, self.model_type, self.cali_end_date, self.vali_end_date]):
            raise ValueError("Missing required parameters. Please provide gauge_id, model_type, "
                            "cali_end_date, and vali_end_date either directly or via namelist.")
        
        # Just use the model_dir directly
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            raise ValueError("model_dir must be provided")
            
        # Set output_dir based on model_dir
        self.output_dir = self.model_dir / "output"
        
        # Initialize data containers
        self.streamflow_data = None
        self.streamflow_metrics = None
        
        # Set default metrics
        self.primary_metric = 'KGE'
        self.secondary_metrics = ['NSE', 'PBIAS', 'RMSE']
        
        # Override with namelist metrics if available
        if self.namelist and 'calibration' in self.namelist and 'metrics' in self.namelist['calibration']:
            metrics_config = self.namelist['calibration']['metrics']
            if 'primary' in metrics_config:
                self.primary_metric = metrics_config['primary']
            if 'secondary' in metrics_config:
                self.secondary_metrics = metrics_config['secondary']
        
    def read_streamflow_data(self):
        """
        Read streamflow data from Raven output file and split into 
        calibration and validation periods.
        
        Returns
        -------
        dict
            Dictionary containing calibration and validation data
        """
        # Print the directory structure for debugging
        print(f"Looking for hydrographs in: {self.output_dir}")
        
        # Construct path
        input_path = self.output_dir / f'{self.gauge_id}_{self.model_type}_Hydrographs.csv'
        
        # Check if file exists
        if not input_path.exists():
            # Try alternative naming patterns
            alt_paths = [
                self.output_dir / "Hydrographs.csv",
                self.output_dir / f'{self.model_type}_Hydrographs.csv',
                self.output_dir / "hydrographs.csv"
            ]
            
            for path in alt_paths:
                if path.exists():
                    input_path = path
                    print(f"Found hydrograph file using alternative name: {path}")
                    break
            else:
                raise FileNotFoundError(f"Streamflow file not found: {input_path} or any alternatives")
        
        # Read data
        try:
            # Read the full CSV file
            df = pd.read_csv(input_path)
            print(f"CSV loaded with columns: {df.columns.tolist()}")
            
            # Extract the appropriate columns - based on the file format
            # The date column is 'date' (column 1)
            # The simulated flow is usually column 4 ('1234 [m3/s]' in this case)
            # The observed flow is usually column 5 ('1234 (observed) [m3/s]' in this case)
            
            # Set date as index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Identify flow columns - find the columns with 'm3/s' in the name
            flow_cols = [col for col in df.columns if 'm3/s' in col or '(observed)' in col]
            
            if len(flow_cols) >= 2:
                # Assume first is simulated, second is observed
                sim_col = flow_cols[0]
                obs_col = flow_cols[1]
            else:
                # Fall back to positional columns if we can't identify by name
                print("Falling back to positional columns")
                # Simulated flow is typically in column 4 (index 3 after loading)
                # Observed flow is typically in column 5 (index 4 after loading)
                sim_col = df.columns[3] if len(df.columns) > 3 else df.columns[0]
                obs_col = df.columns[4] if len(df.columns) > 4 else df.columns[1]
            
            print(f"Using column '{sim_col}' for simulated flow")
            print(f"Using column '{obs_col}' for observed flow")
            
            # Extract calibration and validation periods
            df_cali = df[df.index < self.cali_end_date]
            df_vali = df[(df.index >= self.cali_end_date) & (df.index < self.vali_end_date)]
            
            # Extract simulations and observations
            simulations_cali = df_cali[sim_col]
            observations_cali = df_cali[obs_col]
            simulations_vali = df_vali[sim_col]
            observations_vali = df_vali[obs_col]
            
            # Print some statistics for debugging
            print(f"Calibration period: {len(simulations_cali)} records from {df_cali.index.min()} to {df_cali.index.max()}")
            print(f"Validation period: {len(simulations_vali)} records from {df_vali.index.min() if not df_vali.empty else 'N/A'} to {df_vali.index.max() if not df_vali.empty else 'N/A'}")
            print(f"Simulated flow calibration - mean: {simulations_cali.mean():.2f}, max: {simulations_cali.max():.2f}")
            print(f"Observed flow calibration - mean: {observations_cali.mean():.2f}, max: {observations_cali.max():.2f}")
            
        except Exception as e:
            print(f"Error reading streamflow data: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        self.streamflow_data = {
            'calibration': {
                'simulations': simulations_cali,
                'observations': observations_cali
            },
            'validation': {
                'simulations': simulations_vali,
                'observations': observations_vali
            }
        }
        
        return self.streamflow_data
    
    def calculate_streamflow_metrics(self):
        """
        Calculate performance metrics for streamflow data.
        
        Returns
        -------
        dict
            Dictionary of calculated metrics
        """
        # Make sure we have streamflow data
        if self.streamflow_data is None:
            self.read_streamflow_data()
        
        metrics = {}
        
        # Process each period
        for period in ['calibration', 'validation']:
            sim = self.streamflow_data[period]['simulations']
            obs = self.streamflow_data[period]['observations']
            
            # Skip if no data
            if len(sim) == 0 or len(obs) == 0:
                print(f"Warning: No data for {period} period. Skipping metrics.")
                metrics[period] = {
                    'KGE_NP': float('nan'),
                    'KGE': float('nan'),
                    'NSE': float('nan'),
                    'PBIAS': float('nan'),
                    'RMSE': float('nan'),
                    'VE': float('nan'),
                    'rs': float('nan'),
                    'alpha': float('nan'),
                    'beta': float('nan'),
                    'KGE_NP_Cost': float('nan'),
                    'KGE_Cost': float('nan'),
                    'NSE_Cost': float('nan'),
                    'PBIAS_Cost': float('nan')
                }
                continue
            
            # Filter out NaN values
            valid_idx = ~np.isnan(sim) & ~np.isnan(obs)
            sim_valid = sim[valid_idx]
            obs_valid = obs[valid_idx]
            
            # Check if we have enough valid data
            if len(sim_valid) < 30:
                print(f"Warning: Not enough valid data for {period} period. Metrics may be unreliable.")
            
            # Calculate metrics
            try:
                kge_np, rs, alpha, beta = he.evaluator(obj_fn=he.kgenp, simulations=sim_valid, evaluation=obs_valid)
                kge, r, alpha_kge, beta_kge = he.evaluator(obj_fn=he.kge, simulations=sim_valid, evaluation=obs_valid)
                nse = he.evaluator(obj_fn=he.nse, simulations=sim_valid, evaluation=obs_valid)[0]
                pbias = he.evaluator(obj_fn=he.pbias, simulations=sim_valid, evaluation=obs_valid)[0]
                rmse = he.evaluator(obj_fn=he.rmse, simulations=sim_valid, evaluation=obs_valid)[0]
                try:
                    ve = hr.ve(simulated_array=sim_valid, observed_array=obs_valid)
                except:
                    ve = float('nan')
                
                # Store results
                metrics[period] = {
                    'KGE_NP': kge_np[0],
                    'KGE': kge[0],
                    'NSE': nse,
                    'PBIAS': pbias,
                    'RMSE': rmse,
                    'VE': ve,
                    'rs': rs[0],
                    'alpha': alpha[0],
                    'beta': beta[0],
                    'KGE_NP_Cost': math.fabs(kge_np[0] - 1),
                    'KGE_Cost': math.fabs(kge[0] - 1),
                    'NSE_Cost': math.fabs(nse - 1),
                    'PBIAS_Cost': math.fabs(pbias)
                }
            except Exception as e:
                print(f"Error calculating metrics for {period} period: {e}")
                metrics[period] = {
                    'KGE_NP': float('nan'),
                    'KGE': float('nan'),
                    'NSE': float('nan'),
                    'PBIAS': float('nan'),
                    'RMSE': float('nan'),
                    'VE': float('nan'),
                    'rs': float('nan'),
                    'alpha': float('nan'),
                    'beta': float('nan'),
                    'KGE_NP_Cost': float('nan'),
                    'KGE_Cost': float('nan'),
                    'NSE_Cost': float('nan'),
                    'PBIAS_Cost': float('nan')
                }
        
        self.streamflow_metrics = metrics
        return metrics
    
    def get_objective_value(self):
        """
        Calculate the objective function value for calibration based on the primary metric.
        
        Returns
        -------
        float
            Objective function value (-cost for metrics where higher is better)
        """
        if self.streamflow_metrics is None:
            self.calculate_streamflow_metrics()
            
        # Get the calibration period metrics
        cali_metrics = self.streamflow_metrics['calibration']
        
        # Get the cost value for the primary metric
        cost_metric = f"{self.primary_metric}_Cost" if f"{self.primary_metric}_Cost" in cali_metrics else self.primary_metric
        
        # Return the cost (will be minimized in calibration)
        return cali_metrics[cost_metric]
    
    def run_diagnostics(self):
        """
        Run the diagnostic process, calculating metrics for streamflow.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing all calculated metrics
        """
        # Read streamflow data and calculate metrics
        self.read_streamflow_data()
        self.calculate_streamflow_metrics()
          
        # Create DataFrame for streamflow metrics
        cal_metrics = pd.DataFrame([self.streamflow_metrics['calibration']], index=['CALIBRATION'])
        val_metrics = pd.DataFrame([self.streamflow_metrics['validation']], index=['VALIDATION'])
        
        # Combine metrics
        self.final_metrics = pd.concat([cal_metrics, val_metrics])
        
        # Add model info as metadata
        self.final_metrics.attrs = {
            'gauge_id': self.gauge_id,
            'model_type': self.model_type,
            'coupled': str(self.coupled),
            'calibration_end_date': str(self.cali_end_date),
            'validation_end_date': str(self.vali_end_date),
            'primary_metric': self.primary_metric
        }
        
        return self.final_metrics
    
    def write_metrics(self, output_file=None):
        """
        Write calculated metrics to CSV file.
        
        Parameters
        ----------
        output_file : str or Path, optional
            Path to output file. If None, uses default naming convention.
            
        Returns
        -------
        Path
            Path to the saved metrics file
        """
        # Make sure we have calculated metrics
        if not hasattr(self, 'final_metrics') or self.final_metrics is None:
            self.run_diagnostics()
        
        # Determine output file path
        if output_file is None:
            coupled_str = "_coupled" if self.coupled else ""
            output_file = self.output_dir / f'{self.gauge_id}_{self.model_type}{coupled_str}_metrics.csv'
        else:
            output_file = Path(output_file)
        
        try:
            # Create output directory if it doesn't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write metrics to file
            self.final_metrics.to_csv(output_file)
            print(f"Metrics written to: {output_file}")
            
            # Write metadata as a separate text file
            metadata_file = output_file.with_suffix('.meta.txt')
            with open(metadata_file, 'w') as f:
                f.write(f"Gauge ID: {self.gauge_id}\n")
                f.write(f"Model Type: {self.model_type}\n")
                f.write(f"Coupled: {self.coupled}\n")
                f.write(f"Calibration End Date: {self.cali_end_date}\n")
                f.write(f"Validation End Date: {self.vali_end_date}\n")
                f.write(f"Primary Metric: {self.primary_metric}\n")
                f.write(f"Output Directory: {self.output_dir}\n")
                f.write(f"Generated: {pd.Timestamp.now()}\n")
                
                # Add calibration parameters if available
                if self.namelist and 'calibration' in self.namelist and 'parameters' in self.namelist['calibration']:
                    f.write("\nCalibration Parameters:\n")
                    for param in self.namelist['calibration']['parameters']:
                        f.write(f"  {param['name']}: min={param['min']}, max={param['max']}\n")
            
            return output_file
        except Exception as e:
            print(f"Error writing metrics to file: {e}")
            return None


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate diagnostic metrics for a Raven model")
    parser.add_argument("--namelist", type=str, help="Path to namelist YAML file")
    parser.add_argument("--gauge_id", type=str, help="Gauge ID")
    parser.add_argument("--model_type", type=str, help="Model type (e.g., HBV)")
    parser.add_argument("--cali_end_date", type=str, help="End date of calibration period")
    parser.add_argument("--vali_end_date", type=str, help="End date of validation period")
    parser.add_argument("--coupled", action="store_true", help="Whether this is a coupled model")
    parser.add_argument("--base_dir", type=str, help="Base directory for model setups")
    parser.add_argument("--output", type=str, help="Output file path for metrics")
    
    args = parser.parse_args()
    
    # If namelist is provided, use it as the primary source
    if args.namelist:
        diagnostics = ModelDiagnostics(
            namelist_file=args.namelist,
            gauge_id=args.gauge_id,
            model_type=args.model_type,
            cali_end_date=args.cali_end_date,
            vali_end_date=args.vali_end_date,
            coupled=args.coupled,
            base_dir=args.base_dir
        )
    else:
        # If no namelist, all parameters must be provided
        if not all([args.gauge_id, args.model_type, args.cali_end_date, args.vali_end_date]):
            parser.error("If no namelist is provided, you must specify gauge_id, model_type, cali_end_date, and vali_end_date")
        
        diagnostics = ModelDiagnostics(
            gauge_id=args.gauge_id,
            model_type=args.model_type,
            cali_end_date=args.cali_end_date,
            vali_end_date=args.vali_end_date,
            coupled=args.coupled,
            base_dir=args.base_dir
        )
    
    # Run diagnostics and save results
    diagnostics.run_diagnostics()
    diagnostics.write_metrics(args.output)