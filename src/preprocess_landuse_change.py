import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import shape
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds
import logging
from typing import Dict, List, Optional
import re


class GloGEMAreaChangeProcessor:
    """
    Process GloGEM glacier area change ASCII files and create shapefiles
    with polygons for each glacier and year.
    """
    
    def __init__(self, input_dir: str, output_dir: str = None, debug: bool = False):
        """
        Initialize the GloGEM area change processor.
        
        Parameters
        ----------
        input_dir : str
            Path to directory containing GloGEM ASCII files (gl{year}_{rgi_id}.asc)
        output_dir : str, optional
            Path to output directory for shapefiles. If None, uses input_dir/output
        debug : bool
            Enable debug logging
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger(debug)
        
        # File pattern: gl{year}_{rgi_id}.asc
        self.file_pattern = re.compile(r'gl(\d{4})_(\d+)\.asc')
        
        self.logger.info(f"GloGEM Area Change Processor initialized")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def _setup_logger(self, debug: bool) -> logging.Logger:
        """Setup logger with appropriate level"""
        logger = logging.getLogger('GloGEMAreaChange')
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def find_asc_files(self) -> Dict[str, Dict[str, Path]]:
        """
        Find all GloGEM ASCII files and organize by glacier ID and year.
        
        Returns
        -------
        dict
            Nested dictionary: {glacier_id: {year: filepath}}
        """
        files_dict = {}
        
        # Search for all .asc files matching the pattern
        for file_path in self.input_dir.glob("gl*.asc"):
            match = self.file_pattern.match(file_path.name)
            if match:
                year = match.group(1)
                glacier_id = match.group(2)
                
                if glacier_id not in files_dict:
                    files_dict[glacier_id] = {}
                
                files_dict[glacier_id][year] = file_path
                self.logger.debug(f"Found: Glacier {glacier_id}, Year {year}")
        
        self.logger.info(f"Found {len(files_dict)} unique glaciers")
        for glacier_id, years in files_dict.items():
            self.logger.info(f"  Glacier {glacier_id}: {len(years)} years ({sorted(years.keys())})")
        
        return files_dict
    
    def read_asc_to_polygon(self, filepath: Path, year: str, glacier_id: str) -> Optional[dict]:
        """
        Read ASCII file and convert glacier cells directly to polygon using rasterio.
        
        Parameters
        ----------
        filepath : Path
            Path to ASCII file
        year : str
            Year of the data
        glacier_id : str
            Glacier RGI ID
            
        Returns
        -------
        dict or None
            Dictionary with glacier info and geometry, or None if no glacier cells
        """
        try:
            # Open with rasterio - it handles ASCII grids natively
            with rasterio.open(filepath) as src:
                # Read the data
                data = src.read(1)  # Read first band
                
                # Get transform and CRS
                transform = src.transform
                crs = src.crs if src.crs else 'EPSG:32632'  # Default to UTM 32N for Swiss Alps
                
                # Create mask for glacier cells (not -9999 and not -5000)
                glacier_mask = (data != -9999) & (data != -5000)
                
                if not glacier_mask.any():
                    self.logger.warning(f"No glacier cells in {filepath.name}")
                    return None
                
                # Convert mask to uint8 for rasterio.features.shapes
                glacier_mask = glacier_mask.astype(np.uint8)
                
                # Extract polygon shapes - this is MUCH faster!
                polygon_geoms = []
                for geom, value in shapes(glacier_mask, mask=glacier_mask, transform=transform):
                    if value == 1:
                        polygon_geoms.append(shape(geom))
                
                if not polygon_geoms:
                    return None
                
                # Merge all polygons
                from shapely.ops import unary_union
                if len(polygon_geoms) == 1:
                    glacier_polygon = polygon_geoms[0]
                else:
                    glacier_polygon = unary_union(polygon_geoms)
                
                # Calculate area (assuming UTM coordinates in meters)
                area_km2 = glacier_polygon.area / 1e6
                num_cells = int(glacier_mask.sum())
                
                self.logger.debug(f"Glacier {glacier_id}, year {year}: {area_km2:.3f} km², {num_cells} cells")
                
                return {
                    'glacier_id': glacier_id,
                    'year': int(year),
                    'geometry': glacier_polygon,
                    'area_km2': area_km2,
                    'num_cells': num_cells,
                    'crs': crs
                }
                
        except Exception as e:
            self.logger.error(f"Error processing {filepath.name}: {e}")
            return None
    
    def process_all_files(self) -> gpd.GeoDataFrame:
        """
        Process all GloGEM ASCII files and create a GeoDataFrame.
        
        Returns
        -------
        GeoDataFrame
            GeoDataFrame with columns: glacier_id, year, geometry, area_km2
        """
        self.logger.info("Starting to process all files...")
        
        # Find all files
        files_dict = self.find_asc_files()
        
        if not files_dict:
            raise ValueError(f"No GloGEM ASCII files found in {self.input_dir}")
        
        # Process each file
        rows = []
        crs = None
        
        for glacier_id, year_files in files_dict.items():
            self.logger.info(f"Processing glacier {glacier_id}...")
            
            for year, filepath in sorted(year_files.items()):
                result = self.read_asc_to_polygon(filepath, year, glacier_id)
                
                if result:
                    if crs is None:
                        crs = result['crs']
                    rows.append(result)
                    self.logger.info(f"  ✅ Year {year}: {result['area_km2']:.3f} km²")
        
        if not rows:
            raise ValueError("No glacier polygons were created")
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(rows, crs=crs)
        
        self.logger.info(f"✅ Created GeoDataFrame with {len(gdf)} polygons")
        self.logger.info(f"   Glaciers: {gdf['glacier_id'].nunique()}, Years: {sorted(gdf['year'].unique())}")
        
        return gdf
    
    def save_shapefiles(self, gdf: gpd.GeoDataFrame, 
                       separate_by: str = 'glacier') -> List[Path]:
        """
        Save GeoDataFrame as shapefiles.
        
        Parameters
        ----------
        gdf : GeoDataFrame
            GeoDataFrame with glacier polygons
        separate_by : str
            'glacier': one shapefile per glacier
            'year': one shapefile per year
            'all': one shapefile with all data
            
        Returns
        -------
        list
            List of created shapefile paths
        """
        created_files = []
        
        if separate_by == 'glacier':
            for glacier_id in gdf['glacier_id'].unique():
                glacier_gdf = gdf[gdf['glacier_id'] == glacier_id]
                output_path = self.output_dir / f"glacier_{glacier_id}_area_change.shp"
                glacier_gdf.to_file(output_path)
                created_files.append(output_path)
                self.logger.info(f"Saved: {output_path}")
                
        elif separate_by == 'year':
            for year in sorted(gdf['year'].unique()):
                year_gdf = gdf[gdf['year'] == year]
                output_path = self.output_dir / f"glaciers_year_{year}.shp"
                year_gdf.to_file(output_path)
                created_files.append(output_path)
                self.logger.info(f"Saved: {output_path}")
                
        elif separate_by == 'all':
            output_path = self.output_dir / "glacier_area_change_all.shp"
            gdf.to_file(output_path)
            created_files.append(output_path)
            self.logger.info(f"Saved: {output_path}")
        
        else:
            raise ValueError(f"Invalid separate_by option: {separate_by}")
        
        return created_files
    
    def create_area_difference_shapefiles(self, gdf: gpd.GeoDataFrame, save_plot: bool = True) -> List[Path]:
        """
        Create shapefiles showing the area difference (loss/gain) between consecutive years.
        
        Parameters
        ----------
        gdf : GeoDataFrame
            GeoDataFrame with all glacier polygons
        save_plot : bool
            If True, creates visualization plots
            
        Returns
        -------
        list
            List of paths to created difference shapefiles
        """
        from shapely.ops import unary_union
        import matplotlib.pyplot as plt
        
        self.logger.info("Creating area difference shapefiles...")
        
        # Create directory for difference files
        diff_dir = self.output_dir / "area_differences"
        diff_dir.mkdir(parents=True, exist_ok=True)
        
        glacier_ids = sorted(gdf['glacier_id'].unique())
        years = sorted(gdf['year'].unique())
        
        created_files = []
        all_differences = []  # Store all differences for combined shapefile
        
        # Process each glacier
        for glacier_id in glacier_ids:
            glacier_data = gdf[gdf['glacier_id'] == glacier_id].sort_values('year')
            glacier_years = sorted(glacier_data['year'].unique())
            
            if len(glacier_years) < 2:
                self.logger.warning(f"Glacier {glacier_id} has only one year, skipping difference calculation")
                continue
            
            self.logger.info(f"Processing differences for glacier {glacier_id}...")
            
            glacier_differences = []
            
            # Calculate differences between consecutive years
            for i in range(len(glacier_years) - 1):
                year_old = glacier_years[i]
                year_new = glacier_years[i + 1]
                
                geom_old = glacier_data[glacier_data['year'] == year_old].iloc[0]['geometry']
                geom_new = glacier_data[glacier_data['year'] == year_new].iloc[0]['geometry']
                
                # Calculate areas that were lost (in old but not in new)
                area_lost = geom_old.difference(geom_new)
                
                # Calculate areas that were gained (in new but not in old)
                area_gained = geom_new.difference(geom_old)
                
                # Calculate lost area in km²
                area_lost_km2 = area_lost.area / 1e6 if not area_lost.is_empty else 0
                area_gained_km2 = area_gained.area / 1e6 if not area_gained.is_empty else 0
                net_change_km2 = area_gained_km2 - area_lost_km2
                
                self.logger.info(f"  {year_old} → {year_new}: Lost {area_lost_km2:.3f} km², "
                               f"Gained {area_gained_km2:.3f} km², Net: {net_change_km2:.3f} km²")
                
                # Store loss geometry
                if not area_lost.is_empty:
                    glacier_differences.append({
                        'glacier_id': glacier_id,
                        'year_from': year_old,
                        'year_to': year_new,
                        'change_type': 'loss',
                        'geometry': area_lost,
                        'area_km2': area_lost_km2,
                        'net_change_km2': -area_lost_km2
                    })
                    all_differences.append(glacier_differences[-1])
                
                # Store gain geometry
                if not area_gained.is_empty:
                    glacier_differences.append({
                        'glacier_id': glacier_id,
                        'year_from': year_old,
                        'year_to': year_new,
                        'change_type': 'gain',
                        'geometry': area_gained,
                        'area_km2': area_gained_km2,
                        'net_change_km2': area_gained_km2
                    })
                    all_differences.append(glacier_differences[-1])
            
            # Save individual glacier difference shapefile
            if glacier_differences:
                glacier_diff_gdf = gpd.GeoDataFrame(glacier_differences, crs=gdf.crs)
                output_path = diff_dir / f"glacier_{glacier_id}_area_differences.shp"
                glacier_diff_gdf.to_file(output_path)
                created_files.append(output_path)
                self.logger.info(f"  Saved: {output_path}")
        
        # Create combined difference shapefile
        if all_differences:
            combined_diff_gdf = gpd.GeoDataFrame(all_differences, crs=gdf.crs)
            combined_path = diff_dir / "all_glaciers_area_differences.shp"
            combined_diff_gdf.to_file(combined_path)
            created_files.append(combined_path)
            self.logger.info(f"Saved combined difference shapefile: {combined_path}")
            
            # Create visualization if requested
            if save_plot:
                self._plot_area_differences(combined_diff_gdf, diff_dir)
        
        return created_files
    
    def _plot_area_differences(self, diff_gdf: gpd.GeoDataFrame, output_dir: Path) -> None:
        """
        Create visualization plots for area differences.
        
        Parameters
        ----------
        diff_gdf : GeoDataFrame
            GeoDataFrame with area difference polygons
        output_dir : Path
            Directory to save plots
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        
        self.logger.info("Creating area difference plots...")
        
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        glacier_ids = sorted(diff_gdf['glacier_id'].unique())
        
        # --- Plot 1: Individual glacier difference maps ---
        for glacier_id in glacier_ids:
            glacier_diff = diff_gdf[diff_gdf['glacier_id'] == glacier_id]
            
            # Get unique year pairs
            year_pairs = glacier_diff[['year_from', 'year_to']].drop_duplicates().values
            n_pairs = len(year_pairs)
            
            if n_pairs == 0:
                continue
            
            # Create subplots
            n_cols = min(3, n_pairs)
            n_rows = int(np.ceil(n_pairs / n_cols))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            if n_pairs == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_pairs > 1 else [axes]
            
            for idx, (year_from, year_to) in enumerate(year_pairs):
                ax = axes[idx]
                
                # Get data for this year pair
                pair_data = glacier_diff[
                    (glacier_diff['year_from'] == year_from) & 
                    (glacier_diff['year_to'] == year_to)
                ]
                
                # Plot losses in red
                loss_data = pair_data[pair_data['change_type'] == 'loss']
                if len(loss_data) > 0:
                    loss_data.plot(ax=ax, color='red', alpha=0.6, edgecolor='darkred', linewidth=1.5)
                
                # Plot gains in green
                gain_data = pair_data[pair_data['change_type'] == 'gain']
                if len(gain_data) > 0:
                    gain_data.plot(ax=ax, color='green', alpha=0.6, edgecolor='darkgreen', linewidth=1.5)
                
                # Calculate net change
                net_change = pair_data['net_change_km2'].sum()
                
                ax.set_title(f'{year_from} → {year_to}\nNet change: {net_change:.3f} km²',
                           fontsize=11, fontweight='bold')
                ax.set_xlabel('Easting (m)', fontsize=9)
                ax.set_ylabel('Northing (m)', fontsize=9)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                
                # Add legend
                legend_elements = [
                    Patch(facecolor='red', edgecolor='darkred', alpha=0.6, label='Area Lost'),
                    Patch(facecolor='green', edgecolor='darkgreen', alpha=0.6, label='Area Gained')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
            # Hide unused subplots
            for idx in range(n_pairs, len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle(f'Glacier {glacier_id} - Area Changes Between Years',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            output_path = plots_dir / f"glacier_{glacier_id}_area_differences.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved difference plot: {output_path}")
            
            if self.logger.level == logging.DEBUG:
                plt.show()
            
            plt.close()
        
        # --- Plot 2: Combined overview map ---
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Plot all losses in red
        loss_data = diff_gdf[diff_gdf['change_type'] == 'loss']
        if len(loss_data) > 0:
            loss_data.plot(ax=ax, color='red', alpha=0.4, edgecolor='darkred', linewidth=0.5)
        
        # Plot all gains in green
        gain_data = diff_gdf[diff_gdf['change_type'] == 'gain']
        if len(gain_data) > 0:
            gain_data.plot(ax=ax, color='green', alpha=0.4, edgecolor='darkgreen', linewidth=0.5)
        
        ax.set_title('All Glaciers - Cumulative Area Changes\n(Red = Loss, Green = Gain)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Easting (m)', fontsize=12)
        ax.set_ylabel('Northing (m)', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            Patch(facecolor='red', edgecolor='darkred', alpha=0.6, label='Area Lost'),
            Patch(facecolor='green', edgecolor='darkgreen', alpha=0.6, label='Area Gained')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        # Add statistics
        total_loss = loss_data['area_km2'].sum() if len(loss_data) > 0 else 0
        total_gain = gain_data['area_km2'].sum() if len(gain_data) > 0 else 0
        net_change = total_gain - total_loss
        
        stats_text = (
            f"Summary:\n"
            f"Total area lost: {total_loss:.3f} km²\n"
            f"Total area gained: {total_gain:.3f} km²\n"
            f"Net change: {net_change:.3f} km²\n"
            f"Glaciers: {len(glacier_ids)}"
        )
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add north arrow
        ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
                   fontsize=20, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
        ax.annotate('↑', xy=(0.95, 0.92), xycoords='axes fraction',
                   fontsize=24, ha='center', va='center')
        
        plt.tight_layout()
        
        output_path = plots_dir / "all_glaciers_cumulative_changes.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved cumulative changes plot: {output_path}")
        
        if self.logger.level == logging.DEBUG:
            plt.show()
        
        plt.close()
        
        self.logger.info(f"✅ Area difference plots saved to {plots_dir}")
    
    def create_combined_map(self, gdf: gpd.GeoDataFrame, save_plot: bool = True) -> Path:
        """
        Create a combined shapefile with all glaciers and years, and visualize on a map.
        
        Parameters
        ----------
        gdf : GeoDataFrame
            GeoDataFrame with all glacier polygons
        save_plot : bool
            If True, saves the plot to output directory
            
        Returns
        -------
        Path
            Path to the saved combined shapefile
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        from matplotlib.cm import ScalarMappable
        
        self.logger.info("Creating combined map...")
        
        # Save combined shapefile
        combined_shp_path = self.output_dir / "all_glaciers_all_years.shp"
        gdf.to_file(combined_shp_path)
        self.logger.info(f"Saved combined shapefile: {combined_shp_path}")
        
        # Create plots directory
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Get unique years and glaciers
        years = sorted(gdf['year'].unique())
        glacier_ids = sorted(gdf['glacier_id'].unique())
        n_years = len(years)
        
        # Create figure with subplots for each year
        n_cols = min(3, n_years)  # Max 3 columns
        n_rows = int(np.ceil(n_years / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_years == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_years > 1 else [axes]
        
        # Create color map for different glaciers
        colors = plt.cm.Set3(np.linspace(0, 1, len(glacier_ids)))
        glacier_colors = {glacier_id: colors[i] for i, glacier_id in enumerate(glacier_ids)}
        
        # Plot each year in a separate subplot
        for idx, year in enumerate(years):
            ax = axes[idx]
            year_data = gdf[gdf['year'] == year]
            
            # Plot each glacier
            for glacier_id in glacier_ids:
                glacier_year_data = year_data[year_data['glacier_id'] == glacier_id]
                
                if len(glacier_year_data) > 0:
                    glacier_year_data.plot(ax=ax, 
                                          color=glacier_colors[glacier_id],
                                          edgecolor='black',
                                          linewidth=1.5,
                                          alpha=0.7,
                                          label=f'Glacier {glacier_id}')
            
            ax.set_title(f'Year {year}\nTotal Area: {year_data["area_km2"].sum():.2f} km²', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Easting (m)', fontsize=10)
            ax.set_ylabel('Northing (m)', fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Add legend only to first subplot
            if idx == 0:
                ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        
        # Hide unused subplots
        for idx in range(n_years, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('All Glaciers - Spatial Evolution Over Time', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_plot:
            output_path = plots_dir / "all_glaciers_map_evolution.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved combined map: {output_path}")
        
        if self.logger.level == logging.DEBUG:
            plt.show()
        
        plt.close()
        
        # --- Create a single map with all years overlaid ---
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create color gradient from blue (old) to red (recent) for years
        year_cmap = plt.cm.YlOrRd
        year_norm = Normalize(vmin=years[0], vmax=years[-1])
        
        # Plot all glacier extents for all years
        for year in years:
            year_data = gdf[gdf['year'] == year]
            color = year_cmap(year_norm(year))
            
            year_data.plot(ax=ax,
                          color=color,
                          edgecolor='black',
                          linewidth=0.8,
                          alpha=0.4,
                          label=f'{year}')
        
        ax.set_title('All Glaciers - All Years Overlaid\n(Color gradient: blue=oldest, red=most recent)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Easting (m)', fontsize=12)
        ax.set_ylabel('Northing (m)', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = ScalarMappable(cmap=year_cmap, norm=year_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_label('Year', fontsize=12)
        
        # Add legend with year labels
        handles, labels = ax.get_legend_handles_labels()
        # Only show every other year in legend if too many
        if len(years) > 10:
            step = len(years) // 10
            handles = handles[::step]
            labels = labels[::step]
        ax.legend(handles, labels, loc='upper right', fontsize=9, framealpha=0.9, 
                 title='Years', title_fontsize=10)
        
        # Add north arrow
        ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
                   fontsize=20, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
        ax.annotate('↑', xy=(0.95, 0.92), xycoords='axes fraction',
                   fontsize=24, ha='center', va='center')
        
        # Add statistics text
        total_initial_area = gdf[gdf['year'] == years[0]]['area_km2'].sum()
        total_final_area = gdf[gdf['year'] == years[-1]]['area_km2'].sum()
        total_change = total_final_area - total_initial_area
        total_change_pct = (total_change / total_initial_area) * 100
        
        stats_text = (
            f"Summary:\n"
            f"Glaciers: {len(glacier_ids)}\n"
            f"Initial ({years[0]}): {total_initial_area:.2f} km²\n"
            f"Final ({years[-1]}): {total_final_area:.2f} km²\n"
            f"Change: {total_change:.2f} km² ({total_change_pct:.1f}%)"
        )
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot:
            output_path = plots_dir / "all_glaciers_overlaid_map.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved overlaid map: {output_path}")
        
        if self.logger.level == logging.DEBUG:
            plt.show()
        
        plt.close()
        
        self.logger.info(f"✅ Combined map visualizations saved to {plots_dir}")
        
        return combined_shp_path

    def plot_area_changes(self, gdf: gpd.GeoDataFrame, save_plots: bool = True) -> None:
        """
        Create plots showing area changes over time for each glacier.
        
        Parameters
        ----------
        gdf : GeoDataFrame
            GeoDataFrame with glacier polygons
        save_plots : bool
            If True, saves plots to output directory
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.colors import LinearSegmentedColormap
        
        self.logger.info("Creating area change plots...")
        
        # Create plots directory
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Get unique glaciers
        glacier_ids = sorted(gdf['glacier_id'].unique())
        
        # --- Individual plots for each glacier ---
        for glacier_id in glacier_ids:
            glacier_data = gdf[gdf['glacier_id'] == glacier_id].sort_values('year')
            
            if len(glacier_data) < 2:
                self.logger.warning(f"Glacier {glacier_id} has only one year of data, skipping plot")
                continue
            
            years = glacier_data['year'].values
            areas = glacier_data['area_km2'].values
            
            # Create figure with two subplots
            fig = plt.figure(figsize=(16, 8))
            gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.2], wspace=0.3)
            
            # ===== PLOT 1: Area over time =====
            ax1 = fig.add_subplot(gs[0, 0])
            
            ax1.plot(years, areas, 'o-', linewidth=2, markersize=8, color='steelblue')
            ax1.fill_between(years, 0, areas, alpha=0.3, color='steelblue')
            ax1.set_xlabel('Year', fontsize=12)
            ax1.set_ylabel('Area (km²)', fontsize=12)
            ax1.set_title(f'Glacier {glacier_id} - Area Change Over Time', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for year, area in zip(years, areas):
                ax1.text(year, area, f'{area:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Add summary statistics
            total_change = areas[-1] - areas[0]
            total_change_pct = ((areas[-1] - areas[0]) / areas[0]) * 100
            time_span = years[-1] - years[0]
            avg_change_per_year = total_change / time_span if time_span > 0 else 0
            
            stats_text = (
                f"Summary Statistics:\n"
                f"Initial area ({years[0]}): {areas[0]:.3f} km²\n"
                f"Final area ({years[-1]}): {areas[-1]:.3f} km²\n"
                f"Total change: {total_change:.3f} km² ({total_change_pct:.1f}%)\n"
                f"Avg change/year: {avg_change_per_year:.4f} km²/year"
            )
            
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # ===== PLOT 2: Map visualization of glacier retreat =====
            ax2 = fig.add_subplot(gs[0, 1])
            
            # Create color map from blue (old) to red (recent)
            n_years = len(years)
            colors_map = plt.cm.Blues_r(np.linspace(0.3, 1.0, n_years))
            
            # Plot all glacier extents from oldest to newest
            for i, (year, row) in enumerate(glacier_data.iterrows()):
                geom = row['geometry']
                area = row['area_km2']
                
                # Plot filled polygon with transparency
                if geom.geom_type == 'Polygon':
                    x, y = geom.exterior.xy
                    ax2.fill(x, y, color=colors_map[i], alpha=0.5, 
                            edgecolor='black', linewidth=1.5,
                            label=f'{year} ({area:.2f} km²)')
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        x, y = poly.exterior.xy
                        ax2.fill(x, y, color=colors_map[i], alpha=0.5,
                                edgecolor='black', linewidth=1.5)
                    # Add label only once for MultiPolygon
                    ax2.plot([], [], color=colors_map[i], linewidth=10,
                            label=f'{year} ({area:.2f} km²)')
            
            ax2.set_xlabel('Easting (m)', fontsize=12)
            ax2.set_ylabel('Northing (m)', fontsize=12)
            ax2.set_title(f'Glacier {glacier_id} - Spatial Retreat Over Time', 
                         fontsize=14, fontweight='bold')
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)
            
            # Add legend
            ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
            
            # Add north arrow
            ax2.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
                        fontsize=20, fontweight='bold', ha='center', va='center',
                        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
            ax2.annotate('↑', xy=(0.95, 0.92), xycoords='axes fraction',
                        fontsize=24, ha='center', va='center')
            
            plt.suptitle(f'Glacier {glacier_id} - Area Change Analysis', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            if save_plots:
                output_path = plots_dir / f"glacier_{glacier_id}_area_change.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved plot: {output_path}")
            
            if self.logger.level == logging.DEBUG:
                plt.show()
            
            plt.close()
        
        # --- Combined plot: Total glacier area over time ---
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Calculate total area for each year
        total_area_by_year = gdf.groupby('year')['area_km2'].sum().sort_index()
        years = total_area_by_year.index.values
        total_areas = total_area_by_year.values
        
        # Plot total glacier area
        ax.plot(years, total_areas, 'o-', linewidth=3, markersize=10, 
               color='steelblue', label='Total Glacier Area')
        ax.fill_between(years, 0, total_areas, alpha=0.3, color='steelblue')
        
        # Add value labels
        for year, area in zip(years, total_areas):
            ax.text(year, area, f'{area:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Total Area (km²)', fontsize=12)
        ax.set_title('Total Glacier Area - All Glaciers Combined', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Add summary statistics
        if len(years) > 1:
            total_change = total_areas[-1] - total_areas[0]
            total_change_pct = ((total_areas[-1] - total_areas[0]) / total_areas[0]) * 100
            time_span = years[-1] - years[0]
            avg_change_per_year = total_change / time_span if time_span > 0 else 0
            
            stats_text = (
                f"Combined Statistics:\n"
                f"Initial total area ({years[0]}): {total_areas[0]:.3f} km²\n"
                f"Final total area ({years[-1]}): {total_areas[-1]:.3f} km²\n"
                f"Total change: {total_change:.3f} km² ({total_change_pct:.1f}%)\n"
                f"Avg change/year: {avg_change_per_year:.4f} km²/year\n"
                f"Number of glaciers: {len(glacier_ids)}"
            )
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            output_path = plots_dir / "total_glacier_area.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved combined plot: {output_path}")
        
        if self.logger.level == logging.DEBUG:
            plt.show()
        
        plt.close()
        
        self.logger.info(f"✅ All plots saved to {plots_dir}")


    def create_area_change_summary(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Create a summary table of area changes over time.
        
        Parameters
        ----------
        gdf : GeoDataFrame
            GeoDataFrame with glacier polygons
            
        Returns
        -------
        DataFrame
            Summary table with area changes
        """
        # Pivot to get area by year for each glacier
        area_pivot = gdf.pivot(index='glacier_id', columns='year', values='area_km2')
        
        # Calculate changes
        years = sorted(area_pivot.columns)
        summary_rows = []
        
        for glacier_id in area_pivot.index:
            row = {'glacier_id': glacier_id}
            
            for year in years:
                row[f'area_{year}'] = area_pivot.loc[glacier_id, year]
            
            # Calculate total change
            if len(years) > 1:
                initial_area = area_pivot.loc[glacier_id, years[0]]
                final_area = area_pivot.loc[glacier_id, years[-1]]
                row['total_change_km2'] = final_area - initial_area
                row['total_change_pct'] = ((final_area - initial_area) / initial_area) * 100
            
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Save to CSV
        output_path = self.output_dir / "glacier_area_change_summary.csv"
        summary_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved summary table: {output_path}")
        
        return summary_df