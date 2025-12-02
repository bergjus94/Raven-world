#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCEUA Optimization Script for Raven Model
Optimizes hydrological model parameters using the SCEUA algorithm
Dynamically loads parameters from default_params.yaml based on model type
"""
import os
import sys
import subprocess
import argparse
import spotpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import yaml
import matplotlib.dates as mdates

class RavenSCEUA(object):
    """SPOTPY setup for SCEUA algorithm with Raven hydrological model"""
    
    def __init__(self, gauge_id, model_type, cali_end_date, vali_end_date,
                obj_function='KGE', main_dir=None, config_dir=None, coupled=False,
                params_dir=None):
        """Initialize the setup"""
        # Basic setup
        self.gauge_id = gauge_id
        self.model_type = model_type
        self.cali_end_date = cali_end_date
        self.vali_end_date = vali_end_date
        self.obj_function = obj_function
        self.coupled = coupled

        # Paths
        self.script_dir = Path(__file__).parent.absolute()
        self.main_dir = Path(main_dir) if main_dir else self.script_dir

        # Require config_dir and construct model_dir from it
        if not config_dir:
            raise ValueError("config_dir is required. Please provide a valid configuration directory path.")
        self.config_dir = config_dir
        self.model_dir = self.main_dir / config_dir / f'catchment_{gauge_id}' / model_type

        # Derived paths
        self.output_path = self.model_dir / 'output'
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.template_dir = self.model_dir / 'templates'

        # Raven executable
        self.raven_exe = self.main_dir.parent / 'RavenHydro' / 'build' / 'Raven'

        # Parameters directory - use params_dir if provided, otherwise default
        if params_dir:
            self.params_dir = Path(params_dir)
        else:
            self.params_dir = self.main_dir / 'config' / 'default_params.yaml'

        # Load parameters configuration
        self.params_config = self._load_parameters_config(self.params_dir)
        
        # Results tracking
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.output_path / f"calibration_results_{gauge_id}_{model_type}_{timestamp}.csv"
        self.best_obj = -999  # Initialize with very low value
        self.best_params = None
        
        # Create parameter objects for the model
        self.params = self._setup_parameters()
        
        # Logging
        print(f"SCEUA optimization setup for Gauge {gauge_id}, Model {model_type}")
        print(f"Loaded {len(self.params)} parameters for {model_type} model")
        print(f"Calibration period ends: {cali_end_date}")
        print(f"Validation period ends: {vali_end_date}")
        print(f"Objective function: {obj_function}")
        print(f"Main directory: {self.main_dir}")
        print(f"Config directory: {self.config_dir}")
        print(f"Model directory: {self.model_dir}")
        print(f"Coupled mode: {coupled}")
        print(f"Results will be saved to: {self.results_file}")
        
        # Print parameter summary
        print("\nParameters to be optimized:")
        for i, param in enumerate(self.params, 1):
            print(f"  {i}. {param.name}: [{param.minbound:.4f}, {param.maxbound:.4f}] (init: {param.optguess:.4f})")
        
        # Create results file with headers
        self._create_results_file()
    
    def _load_parameters_config(self, params_file):
        """Load parameter configuration from YAML file"""
        params_file = Path(params_file)
        
        if not params_file.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_file}")
        
        with open(params_file, 'r') as f:
            all_params = yaml.safe_load(f)
        
        # Check if model type exists in parameters file
        if self.model_type not in all_params:
            raise ValueError(f"Model type '{self.model_type}' not found in {params_file}. "
                           f"Available models: {', '.join(all_params.keys())}")
        
        model_params = all_params[self.model_type]
        
        print(f"Loaded parameter configuration for {self.model_type} from {params_file}")
        
        return model_params
    
    def _create_results_file(self):
        """Create the results CSV file with headers"""
        param_names = [p.name for p in self.params]
        headers = param_names + ['objective', 'obj_function', 'timestamp', 'validation_obj']
        
        pd.DataFrame(columns=headers).to_csv(self.results_file, index=False)
        print(f"Created results file with headers: {self.results_file}")
    
    def _setup_parameters(self):
        """Setup parameters for optimization dynamically based on model type"""
        params = []
        
        # Get parameter names, bounds, and initial values
        param_names = self.params_config['names']
        lower_bounds = self.params_config['lower']
        upper_bounds = self.params_config['upper']
        init_values = self.params_config['init']
        
        # Create SPOTPY parameter objects
        for param_key, param_name in param_names.items():
            # Get bounds and initial value
            try:
                lower = float(lower_bounds[param_key])
                upper = float(upper_bounds[param_key])
                init = float(init_values[param_key])
                
                # Validate bounds
                if lower >= upper:
                    print(f"Warning: Invalid bounds for {param_name}: lower={lower}, upper={upper}. Skipping.")
                    continue
                
                # Ensure init is within bounds
                if init < lower or init > upper:
                    print(f"Warning: Initial value {init} for {param_name} is outside bounds [{lower}, {upper}]. "
                          f"Using midpoint instead.")
                    init = (lower + upper) / 2
                
                # Create parameter object
                param = spotpy.parameter.Uniform(param_name, lower, upper, init)
                params.append(param)
                
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not create parameter {param_name} ({param_key}): {e}")
                continue
        
        if not params:
            raise ValueError(f"No valid parameters found for model type {self.model_type}")
        
        return params
    
    def parameters(self):
        """Return the parameter objects for SPOTPY in the required format"""
        return spotpy.parameter.generate(self.params)
    
    def _get_tied_parameters(self, params_dict):
        """Calculate tied parameters based on model type and primary parameters"""
        tied_params = {}
        
        if self.model_type == 'HBV':
            # HBV tied parameters
            if 'HBV_T_Conc_Max_Bas' in params_dict:
                tied_params['HBV_Time_To_Peak'] = params_dict['HBV_T_Conc_Max_Bas'] / 2.0
            
            if 'HBV_Thickness_Topsoil' in params_dict:
                tied_params['HBV_Initial_Thickness_Topsoil'] = params_dict['HBV_Thickness_Topsoil'] * 500.0

        elif self.model_type == 'GR4J':
            # GR4J tied parameter
            if 'Cemaneige_X2' in params_dict:
                tied_params['Airsnow_Coeff'] = 1.0 - params_dict['Cemaneige_X2']
        
        elif self.model_type == 'HMETS':
            # HMETS has NO tied parameters
            pass
        
        elif self.model_type == 'MOHYSE':
            # MOHYSE tied parameter: use the FULL parameter name from YAML
            if 'MOHYSE_Gamma_Scale_Aux' in params_dict:
                if params_dict['MOHYSE_Gamma_Scale_Aux'] > 0:
                    tied_params['Mohyse_Gamma_Scale'] = 1.0 / params_dict['MOHYSE_Gamma_Scale_Aux']
                else:
                    print(f"Warning: MOHYSE_Gamma_Scale_Aux = {params_dict['MOHYSE_Gamma_Scale_Aux']} is not positive.")
                    tied_params['Mohyse_Gamma_Scale'] = 1.0
        
        return tied_params
        

    def _write_parameters_to_file(self, parameters):
        """Write parameters to Raven parameter files"""
        param_names = [p.name for p in self.params]
        params_dict = dict(zip(param_names, parameters))
        
        # Add tied parameters based on model type
        tied_params = self._get_tied_parameters(params_dict)
        params_dict.update(tied_params)
        
        # Define template files directory
        templates_dir = self.model_dir / 'templates'
        
        # List of template files to process
        template_files = [
            {'template': templates_dir / f'{self.gauge_id}_{self.model_type}.rvp.tpl', 
            'output': self.model_dir / f'{self.gauge_id}_{self.model_type}.rvp'},
            {'template': templates_dir / f'{self.gauge_id}_{self.model_type}.rvc.tpl', 
            'output': self.model_dir / f'{self.gauge_id}_{self.model_type}.rvc'},
            {'template': templates_dir / f'{self.gauge_id}_{self.model_type}.rvh.tpl', 
            'output': self.model_dir / f'{self.gauge_id}_{self.model_type}.rvh'},
            {'template': templates_dir / f'{self.gauge_id}_{self.model_type}.rvt.tpl',
            'output': self.model_dir / f'{self.gauge_id}_{self.model_type}.rvt'},
        ]
        
        # Process each template file
        templates_processed = 0
        
        for file_info in template_files:
            template_path = file_info['template']
            output_path = file_info['output']
            
            if not template_path.exists():
                print(f"Warning: Template file not found: {template_path}")
                continue
            
            try:
                # Read template file and replace parameter placeholders
                with open(template_path, 'r') as template, open(output_path, 'w') as output:
                    replacements_made = 0
                    
                    for line in template:
                        modified_line = line
                        
                        # Check if any parameter name is in the line
                        for param_name, param_value in params_dict.items():
                            if param_name in line:
                                # Replace the parameter name with its value
                                if isinstance(param_value, float):
                                    val_str = f"{param_value:.6g}"
                                else:
                                    val_str = str(param_value)
                                
                                # Count replacements in this line
                                occurrences = line.count(param_name)
                                if occurrences > 0:
                                    replacements_made += occurrences
                                
                                modified_line = modified_line.replace(param_name, val_str)
                        
                        # Write the modified line to the output file
                        output.write(modified_line)
                    
                if replacements_made > 0:
                    print(f"Made {replacements_made} parameter replacements in {template_path.name}")
                templates_processed += 1
                
            except Exception as e:
                print(f"Error updating file {output_path}: {e}")
        
        if templates_processed == 0:
            print("WARNING: No template files were processed! Parameters will not be applied.")
        
        return self.model_dir / f'{self.gauge_id}_{self.model_type}'
    
    def _run_model(self, parameters=None):
        """Run the Raven model with given parameters"""
        if parameters is not None:
            self._write_parameters_to_file(parameters)
        
        try:
            # Get model file path
            model_file = self.model_dir / f'{self.gauge_id}_{self.model_type}'
            
            # Run Raven model
            cmd = [str(self.raven_exe), str(model_file), "-o", str(self.output_path)]
            
            process = subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Run diagnostics to get objective function value
            sys.path.append(str(self.script_dir))
            from diagnostic import ModelDiagnostics
            
            # Create diagnostics object
            diagnostics = ModelDiagnostics(
                gauge_id=self.gauge_id,
                model_type=self.model_type,
                cali_end_date=self.cali_end_date,
                vali_end_date=self.vali_end_date,
                coupled=self.coupled,
                base_dir=self.main_dir,
                model_dir=self.model_dir
            )
            
            # Get calibration and validation metrics
            metrics = diagnostics.calculate_streamflow_metrics()
            
            # Get primary objective function value for calibration period
            cali_metrics = metrics['calibration']
            obj_value = cali_metrics[self.obj_function]
            
            # Get validation period objective value for logging
            vali_metrics = metrics['validation']
            vali_obj = vali_metrics[self.obj_function]
            
            # Log both values to the results file
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            param_names = [p.name for p in self.params]
            results_dict = dict(zip(param_names, parameters))
            results_dict.update({
                'objective': obj_value,
                'obj_function': self.obj_function,
                'timestamp': timestamp,
                'validation_obj': vali_obj
            })
            
            # Append to results file
            pd.DataFrame([results_dict]).to_csv(self.results_file, mode='a', header=False, index=False)
            
            return obj_value, vali_obj
            
        except Exception as e:
            print(f"Error running model: {e}")
            return -999, -999  # Return very low objective value on error
    
    def _apply_constraints(self, params_dict):
        """Apply model-specific parameter constraints"""
        """Apply model-specific parameter constraints"""
        if self.model_type == 'HBV':
            # HBV constraint: HBV_Sat_Wilt (X14) < HBV_Field_Capacity (X06)
            if 'X14' in params_dict and 'X06' in params_dict:
                if params_dict['X14'] >= params_dict['X06']:
                    return False
        
        # HMETS constraints
        elif self.model_type == 'HMETS':
            # Min melt factor < Max melt factor
            if 'X05' in params_dict and 'X06' in params_dict:
                if params_dict['X05'] >= params_dict['X06']:
                    return False
            
            # Min SWI < Max SWI
            if 'X09' in params_dict and 'X10' in params_dict:
                if params_dict['X09'] >= params_dict['X10']:
                    return False
        
        elif self.model_type == 'HYMOD':
            # HYMOD doesn't have specific constraints beyond parameter bounds
            pass
        
        elif self.model_type == 'GR4J':
            # GR4J doesn't have specific constraints beyond parameter bounds
            pass

        elif self.model_type == 'MOHYSE':
            # MOHYSE doesn't have specific constraints beyond parameter bounds
            pass
        
        
        return True
    
    def simulation(self, parameters):
        """Run the model with parameters and return objective function value"""
        # Convert parameters to dict for constraint checks
        param_names = [p.name for p in self.params]
        params_dict = dict(zip(param_names, parameters))

        # Apply model-specific constraints
        if not self._apply_constraints(params_dict):
            return [-9999]

        # Run the model and get objective function value
        obj_value, vali_obj = self._run_model(parameters)
        
        # Validate objective value (for KGE, should be <= 1.0)
        if self.obj_function in ['KGE', 'KGE_NP', 'NSE']:
            if obj_value > 1.0:
                print(f"Warning: {self.obj_function} value {obj_value:.4f} exceeds theoretical maximum of 1.0.")
                obj_value = min(obj_value, 1.0)
        
        # Track best parameters
        if obj_value > self.best_obj:
            self.best_obj = obj_value
            self.best_params = parameters
            print(f"New best {self.obj_function}: {obj_value:.4f}")
    
        # Return the objective value for SPOTPY
        return [obj_value]
    
    def evaluation(self):
        """Dummy evaluation method required by SPOTPY"""
        return [0]
    
    def objectivefunction(self, simulation, evaluation):
        """Return objective function value for SCEUA (which minimizes)"""
        # Handle the case where simulation is a list
        if isinstance(simulation, list):
            obj_value = simulation[0]
        else:
            obj_value = simulation
        
        # Handle failed runs
        if obj_value < -900:
            return 999999  # Large positive value for minimization
                
        # For KGE and similar metrics, we need to convert maximization to minimization
        if self.obj_function in ['KGE', 'KGE_NP', 'NSE']:
            # Return -1 * KGE for minimization (smaller is better)
            # This way, KGE=1 (perfect) becomes -1 (minimum)
            return -1 * obj_value
        # For metrics we want to minimize (RMSE, PBIAS_Cost, etc.), return directly
        else:
            return obj_value
    
    def save_best_parameters(self):
        """Save the best parameters to a file"""
        if self.best_params is not None:
            param_names = [p.name for p in self.params]
            best_params_dict = dict(zip(param_names, self.best_params))
            
            # Add tied parameters
            tied_params = self._get_tied_parameters(best_params_dict)
            best_params_dict.update(tied_params)
            
            # Ensure KGE value is valid
            if self.obj_function in ['KGE', 'KGE_NP', 'NSE'] and self.best_obj > 1.0:
                print(f"Warning: Best {self.obj_function} value {self.best_obj:.4f} exceeds theoretical maximum of 1.0.")
                self.best_obj = min(self.best_obj, 1.0)
            
            # Create DataFrame and save
            best_params_df = pd.DataFrame([{**best_params_dict, 'objective': self.best_obj}])
            best_file = self.output_path / f"{self.gauge_id}_{self.model_type}_best_params.csv"
            best_params_df.to_csv(best_file, index=False)
            print(f"Saved best parameters to: {best_file}")
    
    def plot_results(self):
        """Create multiple visualization plots for model evaluation"""
        try:
            if not self.results_file.exists():
                print("Results file not found, cannot create plots.")
                return
                
            results = pd.read_csv(self.results_file)
            if len(results) <= 1:
                print("Not enough data points for plotting.")
                return
                
            # Create output directory for plots if it doesn't exist
            plots_dir = self.output_path / "plots_calibration"
            plots_dir.mkdir(exist_ok=True)
            
            # 1. Create the existing convergence plot
            self._create_convergence_plot(results, plots_dir)
            
            # 2. Create hydrograph comparison plot
            self._create_hydrograph_plot(plots_dir)
            
            # 3. Create seasonal performance analysis
            self._create_seasonal_plot(plots_dir)
            
            # 4. Create metric performance visualization
            self._create_metrics_plot(plots_dir)
            
            # 5. Create observed vs simulated scatter plot
            self._create_scatter_plot(plots_dir)
            
            # 6. Create parameter convergence plot
            self._create_parameter_convergence_plot(results, plots_dir)
            
            print(f"All plots saved to {plots_dir}")
            
        except Exception as e:
            print(f"Error creating plots: {e}")
            import traceback
            traceback.print_exc()
            
        
    def _create_convergence_plot(self, results, plots_dir):
        """Create the optimization convergence plot"""
        try:
            # Calculate running best values to show convergence
            if self.obj_function in ['KGE', 'KGE_NP', 'NSE']:
                # These metrics we want to maximize
                running_best = results['objective'].cummax()
                y_label = f'{self.obj_function} (higher is better)'
                final_best = running_best.iloc[-1]
            else:
                # For metrics we want to minimize (RMSE, etc.)
                running_best = results['objective'].cummin()
                y_label = f'{self.obj_function} (lower is better)'
                final_best = running_best.iloc[-1]
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot the full dataset as light gray points
            plt.scatter(range(1, len(results) + 1), results['objective'], 
                    color='lightgray', alpha=0.3, label='All evaluations')
            
            # Plot the running best values to show convergence
            plt.plot(range(1, len(results) + 1), running_best, 'r-', 
                    linewidth=2, label=f'Best {self.obj_function} so far')
            
            # Identify points where improvement occurs
            improvement_mask = running_best.diff().fillna(0) != 0
            improvement_indices = np.where(improvement_mask)[0]
            
            # Plot points where improvements happen
            plt.scatter(improvement_indices + 1, running_best[improvement_mask], 
                    color='blue', s=50, zorder=3, 
                    label='Improvements')
            
            # Plot final best value
            plt.axhline(y=final_best, color='green', linestyle='--', 
                    label=f'Final best: {final_best:.4f}')
            
            # Add labels and title
            plt.xlabel('Iteration')
            plt.ylabel(y_label)
            plt.title(f'SCEUA Optimization Convergence for {self.gauge_id}_{self.model_type}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save figure
            plot_file = plots_dir / f"{self.gauge_id}_{self.model_type}_sceua_convergence.png"
            plt.savefig(plot_file, dpi=300)
            plt.close()
            print(f"Created optimization convergence plot: {plot_file}")
        except Exception as e:
            print(f"Error creating convergence plot: {e}")

    def _create_hydrograph_plot(self, plots_dir):
        """Create hydrograph comparison plot with calibration and validation periods in two rows"""
        try:
            # Load hydrograph data
            hydrograph_file = list(self.output_path.glob(f'{self.gauge_id}_{self.model_type}_Hydrographs*.csv'))
            if not hydrograph_file:
                hydrograph_file = list(self.output_path.glob('Hydrographs*.csv'))
            
            if not hydrograph_file:
                print("No hydrograph file found.")
                return
                
            # Read the data
            df = pd.read_csv(hydrograph_file[0])
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Find flow columns
            sim_col = None
            obs_col = None
            
            for col in df.columns:
                if 'observed' in col.lower() and 'm3/s' in col.lower():
                    obs_col = col
                elif 'm3/s' in col.lower() and 'observed' not in col.lower():
                    sim_col = col
            
            if not sim_col or not obs_col:
                print("Could not identify simulation and observation columns.")
                return
                    
            # Set the date as index
            df = df.set_index('date')
            
            # Determine calibration and validation periods
            cali_end = pd.to_datetime(self.cali_end_date)
            vali_end = pd.to_datetime(self.vali_end_date)
            
            # Create masks for calibration and validation
            cali_mask = df.index <= cali_end
            vali_mask = (df.index > cali_end) & (df.index <= vali_end)
            
            # Split data into calibration and validation periods
            cali_data = df[cali_mask]
            vali_data = df[vali_mask]
            
            # Create figure with 2 rows, 1 column
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
            
            # Function to plot data for a given period
            def plot_period(ax, data, sim_col, obs_col, title, period_type='calibration'):
                if data.empty:
                    ax.text(0.5, 0.5, f"No data for {title}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12)
                    return
                    
                color = 'blue' if period_type == 'calibration' else 'red'
                
                # Plot simulated flow
                ax.plot(data.index, data[sim_col], color=color, linestyle='-', 
                    label=f'Simulated ({period_type})', alpha=0.7)
                
                # Plot observed flow
                ax.plot(data.index, data[obs_col], 'k-', 
                    label='Observed', alpha=0.7)
                
                # Add light background shading based on period type
                ax.set_facecolor('#e6f2ff' if period_type == 'calibration' else '#ffebeb')
                
                # Format x-axis to show dates properly
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Add labels and title
                ax.set_xlabel('Date')
                ax.set_ylabel('Streamflow (m³/s)')
                ax.set_title(title)
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
                
                # Add KGE value if we have sufficient data
                if len(data) > 10:
                    sim_vals = data[sim_col].values
                    obs_vals = data[obs_col].values
                    mask = ~np.isnan(sim_vals) & ~np.isnan(obs_vals)
                    
                    if np.sum(mask) > 10:
                        try:
                            # Calculate correlation
                            r = np.corrcoef(sim_vals[mask], obs_vals[mask])[0, 1]
                            
                            # Calculate alpha (ratio of standard deviations)
                            alpha = np.std(sim_vals[mask]) / np.std(obs_vals[mask])
                            
                            # Calculate beta (ratio of means)
                            beta = np.mean(sim_vals[mask]) / np.mean(obs_vals[mask])
                            
                            # Calculate KGE
                            kge = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
                            
                            # Add KGE to plot
                            ax.text(0.02, 0.95, f'KGE = {kge:.3f}', 
                                transform=ax.transAxes, fontsize=12,
                                bbox=dict(facecolor='white', alpha=0.8))
                        except:
                            pass
            
            # Plot calibration period (first subplot)
            plot_period(ax1, cali_data, sim_col, obs_col, 
                    f'Calibration Period: {cali_data.index.min().strftime("%Y-%m-%d")} to {cali_data.index.max().strftime("%Y-%m-%d")}', 
                    'calibration')
            
            # Plot validation period (second subplot)
            plot_period(ax2, vali_data, sim_col, obs_col, 
                    f'Validation Period: {vali_data.index.min().strftime("%Y-%m-%d") if not vali_data.empty else "N/A"} to {vali_data.index.max().strftime("%Y-%m-%d") if not vali_data.empty else "N/A"}', 
                    'validation')
            
            # Add overall title
            plt.suptitle(f'Hydrograph Comparison - {self.gauge_id} {self.model_type}', fontsize=16)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save figure
            plot_file = plots_dir / f"{self.gauge_id}_{self.model_type}_hydrograph.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Created hydrograph comparison plot: {plot_file}")
            
        except Exception as e:
            print(f"Error creating hydrograph plot: {e}")
            import traceback
            traceback.print_exc()

    def _create_seasonal_plot(self, plots_dir):
        """Create seasonal performance analysis plot"""
        try:
            # Load hydrograph data
            hydrograph_file = list(self.output_path.glob(f'{self.gauge_id}_{self.model_type}_Hydrographs*.csv'))
            if not hydrograph_file:
                hydrograph_file = list(self.output_path.glob('Hydrographs*.csv'))
            
            if not hydrograph_file:
                print("No hydrograph file found.")
                return
                
            # Read the data
            df = pd.read_csv(hydrograph_file[0])
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Find flow columns
            sim_col = None
            obs_col = None
            
            for col in df.columns:
                if 'observed' in col.lower() and 'm3/s' in col.lower():
                    obs_col = col
                elif 'm3/s' in col.lower() and 'observed' not in col.lower():
                    sim_col = col
            
            if not sim_col or not obs_col:
                print("Could not identify simulation and observation columns.")
                return
                
            # Add month and year columns
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            
            # Determine calibration and validation periods
            cali_end = pd.to_datetime(self.cali_end_date)
            vali_end = pd.to_datetime(self.vali_end_date)
            
            # Create calibration and validation masks
            cali_mask = df['date'] <= cali_end
            vali_mask = (df['date'] > cali_end) & (df['date'] <= vali_end)
            
            # Calculate monthly means
            monthly_means_cali = df[cali_mask].groupby('month').agg({
                sim_col: 'mean',
                obs_col: 'mean'
            })
            
            monthly_means_vali = df[vali_mask].groupby('month').agg({
                sim_col: 'mean',
                obs_col: 'mean'
            })
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot calibration period
            months = range(1, 13)
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Plot for calibration period
            ax1.plot(months, monthly_means_cali[sim_col], 'b-', marker='o', label='Simulated')
            ax1.plot(months, monthly_means_cali[obs_col], 'k-', marker='s', label='Observed')
            ax1.set_xticks(months)
            ax1.set_xticklabels(month_names)
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Mean Monthly Streamflow (m³/s)')
            ax1.set_title('Calibration Period')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot for validation period
            ax2.plot(months, monthly_means_vali[sim_col], 'r-', marker='o', label='Simulated')
            ax2.plot(months, monthly_means_vali[obs_col], 'k-', marker='s', label='Observed')
            ax2.set_xticks(months)
            ax2.set_xticklabels(month_names)
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Mean Monthly Streamflow (m³/s)')
            ax2.set_title('Validation Period')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Adjust layout and add overall title
            plt.suptitle(f'Monthly Seasonal Performance - {self.gauge_id} {self.model_type}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save figure
            plot_file = plots_dir / f"{self.gauge_id}_{self.model_type}_seasonal.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Created seasonal performance plot: {plot_file}")
            
        except Exception as e:
            print(f"Error creating seasonal plot: {e}")

    def _create_metrics_plot(self, plots_dir):
        """Create performance metrics visualization"""
        try:
            # Load metrics file
            metrics_file = list(self.output_path.glob(f'{self.gauge_id}_{self.model_type}_metrics*.csv'))
            
            if not metrics_file:
                print("No metrics file found.")
                return
                
            # Read the data
            metrics_df = pd.read_csv(metrics_file[0], index_col=0)
            print(f"Loaded metrics file: {metrics_file[0].name}")
            
            # Select specific metrics to display
            metrics_to_plot = ['KGE', 'NSE', 'RMSE', 'PBIAS']
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get values for calibration and validation
            cali_values = metrics_df.loc['CALIBRATION', metrics_to_plot].values
            vali_values = metrics_df.loc['VALIDATION', metrics_to_plot].values
            
            # Set bar positions and width
            bar_width = 0.35
            positions = np.arange(len(metrics_to_plot))
            
            # Plot bars
            bars1 = ax.bar(positions - bar_width/2, cali_values, bar_width, label='Calibration')
            bars2 = ax.bar(positions + bar_width/2, vali_values, bar_width, label='Validation')
            
            # Add labels and title
            ax.set_xlabel('Metric')
            ax.set_ylabel('Value')
            ax.set_title(f'Performance Metrics - {self.gauge_id} {self.model_type}')
            ax.set_xticks(positions)
            ax.set_xticklabels(metrics_to_plot)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on top of bars
            def add_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            add_labels(bars1)
            add_labels(bars2)
            
            # Save figure
            plot_file = plots_dir / f"{self.gauge_id}_{self.model_type}_metrics.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Created metrics performance plot: {plot_file}")
            
        except Exception as e:
            print(f"Error creating metrics plot: {e}")
            import traceback
            traceback.print_exc()

    def _create_scatter_plot(self, plots_dir):
        """Create observed vs simulated scatter plot"""
        try:
            # Load hydrograph data
            hydrograph_file = list(self.output_path.glob(f'{self.gauge_id}_{self.model_type}_Hydrographs*.csv'))
            if not hydrograph_file:
                hydrograph_file = list(self.output_path.glob('Hydrographs*.csv'))
            
            if not hydrograph_file:
                print("No hydrograph file found.")
                return
                
            # Read the data
            df = pd.read_csv(hydrograph_file[0])
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Find flow columns
            sim_col = None
            obs_col = None
            
            for col in df.columns:
                if 'observed' in col.lower() and 'm3/s' in col.lower():
                    obs_col = col
                elif 'm3/s' in col.lower() and 'observed' not in col.lower():
                    sim_col = col
            
            if not sim_col or not obs_col:
                print("Could not identify simulation and observation columns.")
                return
                
            # Determine calibration and validation periods
            cali_end = pd.to_datetime(self.cali_end_date)
            vali_end = pd.to_datetime(self.vali_end_date)
            
            # Create calibration and validation masks
            cali_mask = df['date'] <= cali_end
            vali_mask = (df['date'] > cali_end) & (df['date'] <= vali_end)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Get max value for plot limits
            max_val = max(df[sim_col].max(), df[obs_col].max()) * 1.1
            
            # Scatter plot for calibration period
            ax1.scatter(df.loc[cali_mask, obs_col], df.loc[cali_mask, sim_col], 
                    alpha=0.5, edgecolor='k', facecolor='blue')
            
            # Add 1:1 line
            ax1.plot([0, max_val], [0, max_val], 'r--')
            
            # Add labels and title
            ax1.set_xlabel('Observed Streamflow (m³/s)')
            ax1.set_ylabel('Simulated Streamflow (m³/s)')
            ax1.set_title('Calibration Period')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, max_val)
            ax1.set_ylim(0, max_val)
            
            # Add regression line and R² value
            obs_cali = df.loc[cali_mask, obs_col]
            sim_cali = df.loc[cali_mask, sim_col]
            mask = ~np.isnan(obs_cali) & ~np.isnan(sim_cali)
            
            if mask.sum() > 1:
                # Calculate linear regression
                slope, intercept = np.polyfit(obs_cali[mask], sim_cali[mask], 1)
                r_squared = np.corrcoef(obs_cali[mask], sim_cali[mask])[0, 1]**2
                
                # Add regression line
                x_vals = np.array([0, max_val])
                y_vals = intercept + slope * x_vals
                ax1.plot(x_vals, y_vals, 'g-', label=f'Regression (R² = {r_squared:.3f})')
                ax1.legend()
            
            # Scatter plot for validation period
            ax2.scatter(df.loc[vali_mask, obs_col], df.loc[vali_mask, sim_col], 
                    alpha=0.5, edgecolor='k', facecolor='red')
            
            # Add 1:1 line
            ax2.plot([0, max_val], [0, max_val], 'r--')
            
            # Add labels and title
            ax2.set_xlabel('Observed Streamflow (m³/s)')
            ax2.set_ylabel('Simulated Streamflow (m³/s)')
            ax2.set_title('Validation Period')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, max_val)
            ax2.set_ylim(0, max_val)
            
            # Add regression line and R² value
            obs_vali = df.loc[vali_mask, obs_col]
            sim_vali = df.loc[vali_mask, sim_col]
            mask = ~np.isnan(obs_vali) & ~np.isnan(sim_vali)
            
            if mask.sum() > 1:
                # Calculate linear regression
                slope, intercept = np.polyfit(obs_vali[mask], sim_vali[mask], 1)
                r_squared = np.corrcoef(obs_vali[mask], sim_vali[mask])[0, 1]**2
                
                # Add regression line
                x_vals = np.array([0, max_val])
                y_vals = intercept + slope * x_vals
                ax2.plot(x_vals, y_vals, 'g-', label=f'Regression (R² = {r_squared:.3f})')
                ax2.legend()
            
            # Adjust layout and add overall title
            plt.suptitle(f'Observed vs Simulated Streamflow - {self.gauge_id} {self.model_type}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save figure
            plot_file = plots_dir / f"{self.gauge_id}_{self.model_type}_scatter.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Created scatter plot: {plot_file}")
            
        except Exception as e:
            print(f"Error creating scatter plot: {e}")

    def _create_parameter_convergence_plot(self, results, plots_dir):
        """Create parameter convergence plot showing parameter evolution during optimization"""
        try:
            # Get parameter names
            param_names = [p.name for p in self.params]
            
            # Check which parameters actually exist in the results file
            available_params = [p for p in param_names if p in results.columns]
            
            if not available_params:
                print("No parameter columns found in results file.")
                return
            
            # Select a subset of important parameters to plot (maximum 6)
            key_params = [
                'HBV_RainSnow_Temp',
                'HBV_Melt_Fact',
                'HBV_Beta',
                'HBV_T_Conc_Max_Bas',
                'HBV_Thickness_Topsoil',
                'HBV_BaseFlow_Coeff_Fast_Res'
            ]
            
            # Filter to parameters that exist in the results
            plot_params = [p for p in key_params if p in available_params]
            
            # If none of the key parameters are found, use the first 6 available
            if not plot_params and available_params:
                plot_params = available_params[:min(6, len(available_params))]
            
            if not plot_params:
                print("No parameters available for convergence plot.")
                return
            
            # Create figure with subplots
            n_params = len(plot_params)
            n_rows = (n_params + 1) // 2  # Calculate number of rows needed
            
            fig, axes = plt.subplots(n_rows, 2, figsize=(12, n_rows * 4))
            axes = axes.flatten() if n_rows > 1 else [axes]  # Flatten for easy indexing
            
            # Get iterations (x-axis values)
            iterations = np.arange(1, len(results) + 1)
            
            # Get parameter bounds for scaling
            param_bounds = {}
            for param in self.params:
                if param.name in plot_params:
                    param_bounds[param.name] = (param.minbound, param.maxbound)
            
            # Plot each parameter
            for i, param_name in enumerate(plot_params):
                if i >= len(axes):
                    print(f"Warning: Not enough subplots for parameter {param_name}")
                    continue
                    
                ax = axes[i]
                
                # Plot parameter values over iterations
                ax.plot(iterations, results[param_name], 'b-', alpha=0.5)
                
                # Add scatter points
                ax.scatter(iterations, results[param_name], color='blue', s=20, alpha=0.7)
                
                # Calculate running best parameter based on objective function
                if self.obj_function in ['KGE', 'KGE_NP', 'NSE']:
                    # These metrics we want to maximize
                    running_best = results['objective'].cummax()
                    best_mask = results['objective'] == running_best
                else:
                    # For metrics we want to minimize (RMSE, etc.)
                    running_best = results['objective'].cummin()
                    best_mask = results['objective'] == running_best
                
                # Get indices where best values occur
                best_indices = np.where(best_mask)[0]
                
                if len(best_indices) > 0:
                    # Plot best parameter values
                    ax.plot(best_indices + 1, results.loc[best_mask, param_name], 'r-', 
                        linewidth=2, label='Best Parameter')
                    ax.scatter(best_indices + 1, results.loc[best_mask, param_name], 
                            color='red', s=30, zorder=3)
                
                # Add min/max bounds as horizontal lines if available
                if param_name in param_bounds:
                    min_bound, max_bound = param_bounds[param_name]
                    ax.axhline(y=min_bound, color='gray', linestyle='--', alpha=0.5, label='Min Bound')
                    ax.axhline(y=max_bound, color='gray', linestyle='--', alpha=0.5, label='Max Bound')
                
                # Add labels and title
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Parameter Value')
                ax.set_title(f'{param_name}')
                ax.grid(True, alpha=0.3)
                
                # Only add legend to first plot to avoid clutter
                if i == 0:
                    ax.legend()
            
            # Hide any unused subplots
            for i in range(len(plot_params), len(axes)):
                if i < len(axes):
                    axes[i].axis('off')
            
            # Adjust layout and add overall title
            plt.suptitle(f'Parameter Convergence - {self.gauge_id} {self.model_type}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save figure
            plot_file = plots_dir / f"{self.gauge_id}_{self.model_type}_parameter_convergence.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Created parameter convergence plot: {plot_file}")
            
        except Exception as e:
            print(f"Error creating parameter convergence plot: {e}")
            import traceback
            traceback.print_exc()

    def run_best_parameters(self):
        """Run the model one final time with the true best parameter set from results file"""
        try:
            # Load results file to get the best parameters
            if not self.results_file.exists():
                print(f"No results file found at {self.results_file}. Cannot run best parameters.")
                return
                
            print("\nExtracting best parameters from results file...")
            
            # Read results file
            results = pd.read_csv(self.results_file)
            print(f"Loaded results file with {len(results)} evaluations")
            
            # Find best parameter set based on objective function
            if self.obj_function in ['KGE', 'KGE_NP', 'NSE']:
                # These metrics we want to maximize
                best_idx = results['objective'].idxmax()
                best_obj = results.loc[best_idx, 'objective']
                print(f"Best {self.obj_function} (max): {best_obj:.4f} at iteration {best_idx+1}")
            else:
                # For metrics we want to minimize (RMSE, etc.)
                best_idx = results['objective'].idxmin()
                best_obj = results.loc[best_idx, 'objective']
                print(f"Best {self.obj_function} (min): {best_obj:.4f} at iteration {best_idx+1}")
            
            # Get parameter names
            param_names = [p.name for p in self.params]
            
            # Extract best parameters from results
            best_params = results.loc[best_idx, param_names].values
            
            print("\nBest parameter values:")
            for name, value in zip(param_names, best_params):
                print(f"  {name}: {value:.6f}")
            
            print("\n" + "="*50)
            print(f"Running final model with best parameter set (iteration {best_idx+1})")
            print("="*50)
            
            # SECOND: Modify the RVI file to add extended output options for final run
            print("Adding extended output options to RVI file for final run...")
            self._add_extended_output_options()
            
            # THIRD: Run the model with the modified RVI file
            obj_value, vali_obj = self._run_model(best_params)  # Don't pass parameters again since files are already written
            
            print(f"Final run with best parameters:")
            print(f"  Calibration {self.obj_function}: {obj_value:.4f}")
            print(f"  Validation {self.obj_function}: {vali_obj:.4f}")
            
            # Verify if values match
            if abs(obj_value - best_obj) > 0.01:
                print(f"WARNING: Final run {self.obj_function} ({obj_value:.4f}) doesn't match")
                print(f"         expected value from optimization ({best_obj:.4f})!")
                print(f"         This suggests a problem with parameter application.")
                
                # Double-check parameter application
                print("\nDouble-checking parameter application:")
                
                # Read the parameter files that were just written
                for file_info in [
                    {'name': 'RVP file', 'path': self.model_dir / f"{self.gauge_id}_{self.model_type}.rvp"},
                    {'name': 'RVC file', 'path': self.model_dir / f"{self.gauge_id}_{self.model_type}.rvc"},
                    {'name': 'RVH file', 'path': self.model_dir / f"{self.gauge_id}_{self.model_type}.rvh"},
                    {'name': 'RVT file', 'path': self.model_dir / f"{self.gauge_id}_{self.model_type}.rvt"}
                ]:
                    if file_info['path'].exists():
                        print(f"\nChecking {file_info['name']} for parameter values:")
                        with open(file_info['path'], 'r') as f:
                            content = f.read()
                            
                        # Check for each parameter
                        for name, value in zip(param_names, best_params):
                            if name in content:
                                # Extract context around parameter
                                start = max(0, content.find(name) - 20)
                                end = min(len(content), content.find(name) + len(name) + 30)
                                context = content[start:end].strip()
                                print(f"  {name} found in context: '...{context}...'")
                            else:
                                print(f"  {name} not found in file!")
            
            # Create a special marker file to indicate this is the final best run
            marker_file = self.output_path / f"{self.gauge_id}_{self.model_type}_BEST_RUN.txt"
            with open(marker_file, 'w') as f:
                f.write(f"Best parameter run completed at {datetime.now()}\n")
                f.write(f"Calibration {self.obj_function}: {obj_value:.4f}\n")
                f.write(f"Validation {self.obj_function}: {vali_obj:.4f}\n")
                f.write(f"Best {self.obj_function} from optimization: {best_obj:.4f} (iteration {best_idx+1})\n")
                
                # Add parameters to the marker file
                f.write("\nParameters:\n")
                for name, value in zip(param_names, best_params):
                    f.write(f"{name}: {value:.6f}\n")
            
            print(f"Created marker file: {marker_file}")
            
            # Save best parameters to CSV
            best_params_df = pd.DataFrame([dict(zip(param_names, best_params))])
            best_params_df['objective'] = best_obj
            best_params_df['iteration'] = best_idx + 1
            best_params_csv = self.output_path / f"{self.gauge_id}_{self.model_type}_VERIFIED_best_params.csv"
            best_params_df.to_csv(best_params_csv, index=False)
            print(f"Saved verified best parameters to: {best_params_csv}")
            
            # Generate full diagnostic metrics for the best run
            print("\nGenerating diagnostic metrics file for best run...")
            try:
                # Import diagnostic module
                sys.path.append(str(self.script_dir))
                from diagnostic import ModelDiagnostics
                
                # Create diagnostics object
                diagnostics = ModelDiagnostics(
                    gauge_id=self.gauge_id,
                    model_type=self.model_type,
                    cali_end_date=self.cali_end_date,
                    vali_end_date=self.vali_end_date,
                    coupled=self.coupled,
                    base_dir=self.main_dir,
                    model_dir=self.model_dir
                )
                
                # Run diagnostics and explicitly write metrics to file
                metrics_df = diagnostics.run_diagnostics()
                metrics_file = diagnostics.write_metrics()
                
                if metrics_file:
                    print(f"Generated metrics file: {metrics_file}")
                else:
                    print("Warning: Failed to generate metrics file")
                    
                # Also create a best-run specific metrics file
                best_metrics_file = self.output_path / f"{self.gauge_id}_{self.model_type}_BEST_RUN_metrics.csv"
                if hasattr(diagnostics, 'final_metrics'):
                    diagnostics.final_metrics.to_csv(best_metrics_file)
                    print(f"Also saved metrics to: {best_metrics_file}")
                    
            except Exception as e:
                print(f"Error generating diagnostic metrics: {e}")
                import traceback
                traceback.print_exc()
            
            return obj_value, vali_obj
        
        except Exception as e:
            print(f"Error running best parameters: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _add_extended_output_options(self):
        """Modify the RVI file to add extended output options for the final run"""
        try:
            rvi_file = self.model_dir / f'{self.gauge_id}_{self.model_type}.rvi'
            
            if not rvi_file.exists():
                print(f"Warning: RVI file not found at {rvi_file}")
                return
            
            # Read the current RVI file
            with open(rvi_file, 'r') as f:
                lines = f.readlines()
            
            # Define the extended output options
            extended_output_options = [
                "  :EvaluationMetrics RMSE KLING_GUPTA NASH_SUTCLIFFE\n",
                "  :WriteForcingFunctions\n",
                "  :CustomOutput DAILY AVERAGE SNOW BY_HRU_GROUP\n",
                "  :CustomOutput DAILY AVERAGE SNOW BY_HRU\n",
                "  :CustomOutput DAILY AVERAGE RUNOFF BY_HRU\n",
                "  :CustomOutput DAILY AVERAGE PRECIP BY_HRU\n",
                "  :CustomOutput DAILY AVERAGE ATMOSPHERE BY_HRU\n",
                "  :CustomOutput DAILY AVERAGE SOIL[0] BY_HRU\n",
                "  :CustomOutput DAILY AVERAGE SOIL[1] BY_HRU\n",
                "  :CustomOutput DAILY AVERAGE SOIL[2] BY_HRU\n",
                "  :CustomOutput DAILY AVERAGE AET BY_HRU\n",
                "  :CustomOutput DAILY AVERAGE SURFACE_WATER BY_HRU\n",
                "  :CustomOutput DAILY AVERAGE From:GLACIER_ICE BY_BASIN\n",
                "  :CustomOutput DAILY AVERAGE From:GLACIER_ICE BY_HRU\n",
                "  :CustomOutput DAILY AVERAGE To:LAKE_STORAGE BY_BASIN\n",
                "  :CustomOutput DAILY AVERAGE To:LAKE_STORAGE BY_HRU_GROUP\n",
                "  :CustomOutput DAILY AVERAGE To:LAKE_STORAGE BY_HRU\n",
                "  :CustomOutput DAILY AVERAGE TEMP_AVE  BY_HRU_GROUP\n",
                "  :CustomOutput DAILY AVERAGE TEMP_AVE  BY_HRU\n",
                "  :CustomOutput DAILY AVERAGE POTENTIAL_MELT BY_HRU\n",
                "  :CustomOutput DAILY AVERAGE POTENTIAL_MELT BY_HRU_GROUP\n",
                "  :CustomOutput DAILY AVERAGE RAINFALL BY_HRU\n",
                "  :CustomOutput DAILY AVERAGE SNOWFALL BY_HRU\n",
                "  :CustomOutput DAILY AVERAGE SNOW_FRAC BY_HRU\n",
                "  :CustomOutput DAILY AVERAGE SNOW_FRAC BY_HRU_GROUP\n",
            ]
            
            # Find the #Output Options section and replace it
            new_lines = []
            in_output_section = False
            output_section_found = False
            
            for line in lines:
                if '#Output Options' in line or ':Output Options' in line:
                    in_output_section = True
                    output_section_found = True
                    new_lines.append(line)  # Keep the header
                    # Add all extended output options
                    new_lines.extend(extended_output_options)
                elif in_output_section and (line.strip().startswith('#') or line.strip().startswith(':')):
                    # Check if this is the start of a new section
                    if not line.strip().startswith(':CustomOutput') and not line.strip().startswith(':EvaluationMetrics') and not line.strip().startswith(':WriteForcingFunctions'):
                        # This is a new section, stop adding output options
                        in_output_section = False
                        new_lines.append(line)
                    # Skip the old output option lines (they're replaced by extended_output_options)
                elif not in_output_section:
                    new_lines.append(line)
            
            # If no output section was found, add it at the end
            if not output_section_found:
                new_lines.append("\n#Output Options\n")
                new_lines.extend(extended_output_options)
            
            # Write the modified RVI file
            with open(rvi_file, 'w') as f:
                f.writelines(new_lines)
            
            print(f"Successfully added extended output options to {rvi_file}")
            print(f"Added {len(extended_output_options)} output directives for comprehensive model output")
            
        except Exception as e:
            print(f"Error modifying RVI file: {e}")
            import traceback
            traceback.print_exc()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run SCEUA optimization for Raven models")
    
    # Required arguments
    parser.add_argument('--gauge_id', type=str, help='Gauge ID for the catchment')
    parser.add_argument('--model_type', type=str, help='Model type (HBV, GR4J, etc.)')
    parser.add_argument('--cali_end_date', type=str, help='End date of calibration period (YYYY-MM-DD)')
    parser.add_argument('--vali_end_date', type=str, help='End date of validation period (YYYY-MM-DD)')
    
    # Optional arguments
    parser.add_argument('--obj_function', type=str, default='KGE', help='Objective function (KGE, NSE, etc.)')
    parser.add_argument('--iterations', type=int, default=20, help='Number of iterations')
    parser.add_argument('--ngs', type=int, default=8, help='Number of complexes for SCEUA')
    parser.add_argument('--main_dir', type=str, help='Main directory for Raven Switzerland project')
    parser.add_argument('--config_dir', type=str, help='Configuration directory (e.g., coupled or uncoupled)')
    parser.add_argument('--coupled', action='store_true', help='Use coupled model')
    parser.add_argument('--namelist', type=str, help='Path to namelist file')
    parser.add_argument('--params-dir', type=str, help='Path to parameters YAML file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Default values
    namelist = None
    main_dir = args.main_dir
    config_dir = args.config_dir
    params_dir = getattr(args, 'params_dir', None)
    
    # Load from namelist if provided
    if args.namelist:
        namelist_path = Path(args.namelist)
        
        if namelist_path.exists():
            with open(namelist_path, 'r') as f:
                namelist = yaml.safe_load(f)
            
            # Set defaults from namelist if not provided in command line
            if not args.gauge_id and 'gauge_id' in namelist:
                args.gauge_id = str(namelist['gauge_id'])
            
            if not args.model_type and 'model_type' in namelist:
                args.model_type = namelist['model_type']
            
            if not main_dir and 'main_dir' in namelist:
                main_dir = namelist['main_dir']
            
            if not config_dir and 'config_dir' in namelist:
                config_dir = namelist['config_dir']
                print(f"Using config_dir from namelist: {config_dir}")
            
            if not args.coupled and 'coupled' in namelist:
                args.coupled = namelist['coupled']
                
            # Get calibration dates from namelist
            if not args.cali_end_date and 'cali_end_date' in namelist:
                args.cali_end_date = namelist['cali_end_date']
            
            if not args.vali_end_date and 'end_date' in namelist:
                args.vali_end_date = namelist['end_date']
            
            # Load params_dir from namelist if not provided
            if not params_dir and 'params_dir' in namelist:
                params_dir = namelist['params_dir']
                print(f"Using params_dir from namelist: {params_dir}")
            
            # Set iterations from namelist if available
            if 'calibration' in namelist and 'iterations' in namelist['calibration']:
                args.iterations = namelist['calibration']['iterations']
                print(f"Using iterations={args.iterations} from namelist")

            # Set ngs (number of complexes) from namelist if available
            if 'calibration' in namelist and 'ngs' in namelist['calibration']:
                args.ngs = namelist['calibration']['ngs']
                print(f"Using ngs={args.ngs} from namelist")

            # Set objective function from namelist if available
            if 'calibration' in namelist and 'metrics' in namelist['calibration'] and 'primary' in namelist['calibration']['metrics']:
                args.obj_function = namelist['calibration']['metrics']['primary']
                print(f"Using objective function={args.obj_function} from namelist")
    
    # Validate required arguments
    if not args.gauge_id:
        parser.error("Gauge ID is required")
    
    if not args.model_type:
        parser.error("Model type is required")
    
    if not args.cali_end_date:
        parser.error("Calibration end date is required")
    
    if not args.vali_end_date:
        parser.error("Validation end date is required")
    
    if not config_dir:
        parser.error("config_dir is required (must be in namelist or command line)")
    
    # Store additional values in args
    args.main_dir = main_dir
    args.config_dir = config_dir
    args.params_dir = params_dir
    args.namelist = namelist
    
    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Initialize SCEUA setup
    optimization = RavenSCEUA(
        gauge_id=args.gauge_id,
        model_type=args.model_type,
        cali_end_date=args.cali_end_date,
        vali_end_date=args.vali_end_date,
        obj_function=args.obj_function,
        main_dir=args.main_dir,
        config_dir=args.config_dir,
        coupled=args.coupled,
        params_dir=args.params_dir
    )
    
    # Run SCEUA algorithm
    print(f"\nStarting SCEUA optimization with {args.ngs} complexes")
    print(f"Objective function: {args.obj_function}")
    print(f"Maximum iterations: {args.iterations}")
    print(f"Model: {args.model_type}")
    print(f"Number of parameters: {len(optimization.params)}")
    
    sampler = spotpy.algorithms.sceua(
        optimization,
        dbname=str(optimization.output_path / f"raven_sceua_{args.gauge_id}_{args.model_type}"),
        dbformat='csv',
        parallel='seq',
        save_sim=False
    )
    
    sampler.sample(args.iterations, ngs=args.ngs)
    
    # Save best parameters
    optimization.save_best_parameters()
    
    # Run final model with best parameters
    optimization.run_best_parameters()

    # Plot results
    optimization.plot_results()
    
    print("\nOptimization complete!")