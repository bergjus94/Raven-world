#### This file contains the StreamflowProcessor class for preprocessing streamflow data worldwide to run Raven
#### July 2025
#### Justine Berg

#--------------------------------------------------------------------------------
############################### import packages #################################
#--------------------------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import os
import logging
import yaml
from datetime import datetime
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------
############################### StreamflowProcessor Class #######################
#--------------------------------------------------------------------------------

class StreamflowProcessor:
    """
    A class for preprocessing streamflow data for hydrological modeling with Raven
    """
    
    def __init__(self, config):
        """
        Initialize the streamflow data preprocessor
        
        Parameters
        ----------
        config : dict or str
            Configuration dictionary with all required parameters or path to YAML config file
        """
        # Load config if it's a file path
        if isinstance(config, (str, Path)):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        
        # Store configuration parameters
        self.main_dir = Path(config.get('main_dir'))
        self.gauge_id = config.get('gauge_id')
        self.start_date = config.get('start_date')
        self.end_date = config.get('end_date')
        self.model_type = config.get('model_type')
        self.debug = config.get('debug', False)
        self.coupled = config.get('coupled', False)
        
        # Set model directory based on coupled/uncoupled
        if self.coupled:
            self.model_dir = self.main_dir / config['model_dirs']['coupled']
        else:
            self.model_dir = self.main_dir / config['model_dirs']['uncoupled']
        
        # Get streamflow file path
        stream_dir = config.get('stream_dir', '').format(gauge_id=self.gauge_id)
        if not os.path.isabs(stream_dir):
            self.stream_file = self.main_dir / stream_dir
        else:
            self.stream_file = Path(stream_dir)
            
        # Define output paths
        self.output_path = self.model_dir / f'catchment_{self.gauge_id}' / self.model_type / 'data_obs'
        self.plots_dir = self.model_dir / f'catchment_{self.gauge_id}' / self.model_type / 'plots'
        
        # Create output directories if they don't exist
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Define standard output files
        self.output_file = self.output_path / 'Q_daily.rvt'
        self.plot_file = self.plots_dir / f'streamflow_timeseries_gauge_{self.gauge_id}.png'

    def subset_dataframe_time(self, dataframe):
        """Subsetting a dataframe using a time interval."""
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        mask = dataframe['date'].between(start_date, end_date, inclusive="both")
        return dataframe[mask]

    def export_to_rvt_file(self, df, out_path):
        """Writes RVT file from DataFrame"""
        start_time = "0:00:00"
        
        with open(out_path, 'w') as f:
            f.write(f":ObservationData\tHYDROGRAPH\t1\tm3/s\n{self.start_date}\t{start_time}\t1\t{len(df)}\n")
            df_as_string = df.to_string(justify="right", header=False, index=False,
                                        columns=['Q_obs'], na_rep="-1.2345")
            f.write(df_as_string)
            f.write("\n:EndObservationData")

    def check_and_fill_missing_data(self, df):
        """
        Check for NaN values and missing dates, fill with -1.2345 to create complete time series
        
        Parameters
        ----------
        df : DataFrame
            Streamflow dataframe with 'date' and 'Q_obs' columns
            
        Returns
        -------
        DataFrame
            Complete dataframe with all dates filled and NaN values replaced with -1.2345
        """
        if self.debug:
            print("\n🔍 Checking for missing data and dates...")
        
        # Convert dates to datetime if not already
        df['date'] = pd.to_datetime(df['date'])
        
        # Check for NaN values in original data
        nan_count = df['Q_obs'].isna().sum()
        if nan_count > 0 and self.debug:
            print(f"   📊 Found {nan_count} NaN values in original data")
        
        # Create complete date range
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        complete_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        if self.debug:
            print(f"   📅 Expected date range: {start_date.date()} to {end_date.date()}")
            print(f"   📅 Expected total days: {len(complete_dates)}")
            print(f"   📅 Original data days: {len(df)}")
        
        # Create complete dataframe with all dates
        complete_df = pd.DataFrame({'date': complete_dates})
        
        # Merge with original data
        complete_df = complete_df.merge(df[['date', 'Q_obs']], on='date', how='left')
        
        # Count missing dates
        missing_dates = complete_df['Q_obs'].isna().sum()
        
        if self.debug:
            print(f"   🔍 Missing dates found: {missing_dates}")
            if missing_dates > 0:
                missing_periods = complete_df[complete_df['Q_obs'].isna()]['date']
                print(f"   📋 First few missing dates: {missing_periods.head().dt.strftime('%Y-%m-%d').tolist()}")
                if len(missing_periods) > 5:
                    print(f"   📋 Last few missing dates: {missing_periods.tail().dt.strftime('%Y-%m-%d').tolist()}")
        
        # Fill all NaN values with -1.2345
        complete_df['Q_obs'] = complete_df['Q_obs'].fillna(-1.2345)
        
        # Final summary
        total_filled = nan_count + missing_dates
        if self.debug and total_filled > 0:
            print(f"   ✅ Filled {total_filled} total missing values with -1.2345")
            print(f"      • Original NaN values: {nan_count}")
            print(f"      • Missing dates: {missing_dates}")
        elif self.debug:
            print("   ✅ No missing data found - time series is complete!")
        
        return complete_df

        
    def plot_streamflow_timeseries(self, df):
        """
        Create a time series plot of streamflow data with gaps for missing data
        
        Parameters
        ----------
        df : DataFrame
            Complete dataframe with 'date' and 'Q_obs' columns
        """
        if self.debug:
            print("\n📊 Creating streamflow time series plot...")
        
        try:
            # Create a copy for plotting (replace -1.2345 with NaN for gaps)
            plot_df = df.copy()
            plot_df.loc[plot_df['Q_obs'] == -1.2345, 'Q_obs'] = np.nan
            
            # Calculate statistics for valid data only
            valid_data = plot_df.dropna()
            if len(valid_data) > 0:
                mean_flow = valid_data['Q_obs'].mean()
                max_flow = valid_data['Q_obs'].max()
                min_flow = valid_data['Q_obs'].min()
                std_flow = valid_data['Q_obs'].std()
                missing_pct = (len(df) - len(valid_data)) / len(df) * 100
            else:
                mean_flow = max_flow = min_flow = std_flow = 0
                missing_pct = 100
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot streamflow with gaps for missing data
            ax.plot(plot_df['date'], plot_df['Q_obs'], 
                   color='steelblue', linewidth=0.8, alpha=0.8, label='Observed Streamflow')
            
            # Add mean line
            if len(valid_data) > 0:
                ax.axhline(y=mean_flow, color='red', linestyle='--', alpha=0.7, 
                          linewidth=1.5, label=f'Mean: {mean_flow:.2f} m³/s')
            
            # Formatting
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Streamflow (m³/s)', fontsize=12, fontweight='bold')
            ax.set_title(f'Streamflow Time Series - Gauge {self.gauge_id}\n'
                        f'Period: {self.start_date} to {self.end_date}', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Legend
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            
            # Format x-axis
            ax.tick_params(axis='x', rotation=45)
            
            # Add statistics text box
            stats_text = (
                f'Statistics (Valid Data Only):\n'
                f'• Records: {len(valid_data):,} / {len(df):,}\n'
                f'• Data Coverage: {100-missing_pct:.1f}%\n'
                f'• Mean: {mean_flow:.2f} m³/s\n'
                f'• Min: {min_flow:.2f} m³/s\n'
                f'• Max: {max_flow:.2f} m³/s\n'
                f'• Std Dev: {std_flow:.2f} m³/s'
            )
            
            # Position text box
            props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=props, family='monospace')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(self.plot_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            
            if self.debug:
                print(f"   ✅ Plot saved to: {self.plot_file}")
            
            # Show plot
            plt.show()
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error creating streamflow plot: {str(e)}")
            return False
        
    def process(self):
        """Process streamflow data and convert to Raven format"""
        try:
            if not self.stream_file.exists():
                raise FileNotFoundError(f"Streamflow file not found: {self.stream_file}")
            
            df_streamflow = pd.read_csv(self.stream_file)
            
            if self.debug:
                print(f"📂 Loaded streamflow data with shape: {df_streamflow.shape}")
                print(f"📅 Date range: {df_streamflow['date'].min()} to {df_streamflow['date'].max()}")

            # Convert date column and subset time period
            df_streamflow['date'] = pd.to_datetime(df_streamflow['date'])
            df_streamflow = self.subset_dataframe_time(df_streamflow)
            
            if self.debug:
                print(f"📅 After time subsetting: {len(df_streamflow)} records")
            
            # Check and fill missing data/dates
            df_streamflow = self.check_and_fill_missing_data(df_streamflow)
            
            # Final data summary
            missing_data = (df_streamflow['Q_obs'] == -1.2345).sum()
            valid_data = len(df_streamflow) - missing_data
            
            if self.debug:
                print(f"\n📊 Final dataset summary:")
                print(f"   • Total records: {len(df_streamflow)}")
                print(f"   • Valid observations: {valid_data}")
                print(f"   • Missing/filled values: {missing_data}")
                print(f"   • Data completeness: {(valid_data/len(df_streamflow)*100):.1f}%")
            
            if missing_data > 0:
                print(f"⚠️  Warning: {missing_data} missing values filled with -1.2345 in streamflow data")
            
            # Create streamflow time series plot
            plot_success = self.plot_streamflow_timeseries(df_streamflow)
            
            # Export to RVT file
            self.export_to_rvt_file(df_streamflow, self.output_file)
            
            if self.debug:
                print(f"✅ Successfully processed streamflow data for gauge {self.gauge_id}")
                print(f"📁 RVT file: {self.output_file}")
                if plot_success:
                    print(f"📊 Plot saved: {self.plot_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing streamflow data: {str(e)}")
            return False

#--------------------------------------------------------------------------------
############################### Convenience function ############################
#--------------------------------------------------------------------------------

def process_streamflow(config_file=None, **kwargs):
    """
    Convenience function for quick streamflow processing
    
    Parameters
    ----------
    config_file : str, optional
        Path to YAML config file
    **kwargs : dict
        Configuration parameters as keyword arguments
    
    Returns
    -------
    bool
        Success status
    """
    if config_file:
        processor = StreamflowProcessor(config_file)
    else:
        processor = StreamflowProcessor(kwargs)
    
    return processor.process()

#--------------------------------------------------------------------------------
############################### Main execution ################################
#--------------------------------------------------------------------------------

if __name__ == "__main__":
    config_file = "/home/jberg/OneDrive/Raven-world/namelist.yaml"
    
    try:
        success = process_streamflow(config_file)
        if success:
            print("🎉 Streamflow preprocessing completed successfully!")
        else:
            print("❌ Streamflow preprocessing failed!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")