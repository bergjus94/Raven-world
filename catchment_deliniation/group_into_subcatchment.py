"""
Script to identify nested catchments and create subcatchment shapefiles.
For each catchment system, creates:
1. Individual subcatchment files (one file per subcatchment)
2. One combined basin file with all subcatchments as separate features
"""

import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SubcatchmentGrouper:
    """Identify nested catchments and create proper subcatchment boundaries"""
    
    def __init__(self, input_dir, output_dir):
        """
        Parameters
        ----------
        input_dir : str or Path
            Directory containing catchment_shape_XXXX.shp files
        output_dir : str or Path
            Directory to save subcatchment shapefiles
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_all_catchments(self):
        """Load all catchment shapefiles"""
        print("📂 Loading all catchment shapefiles...")
        catchments = {}
        
        for shp_file in sorted(self.input_dir.glob("catchment_shape_*.shp")):
            gauge_id = shp_file.stem.split('_')[-1]  # Extract gauge ID (e.g., "0001")
            try:
                gdf = gpd.read_file(shp_file)
                if not gdf.empty:
                    # Get the catchment geometry (assuming single polygon per file)
                    catchments[gauge_id] = {
                        'geometry': gdf.unary_union,  # Combine all features into one
                        'gdf': gdf,
                        'area': gdf.geometry.area.sum(),
                        'file': shp_file
                    }
                    print(f"  ✅ Loaded {gauge_id}: {len(gdf)} features, area={catchments[gauge_id]['area']:.2f}")
            except Exception as e:
                print(f"  ❌ Error loading {shp_file.name}: {e}")
        
        print(f"\n📊 Total catchments loaded: {len(catchments)}")
        return catchments
    
    def find_nested_relationships(self, catchments):
        """
        Find which catchments are nested within others.
        Returns a dictionary of relationships: {parent_id: [child_ids]}
        """
        print("\n🔍 Identifying nested catchment relationships...")
        relationships = {}  # parent_id: [child_ids]
        
        # Sort catchments by area (largest first)
        sorted_ids = sorted(catchments.keys(), 
                          key=lambda x: catchments[x]['area'], 
                          reverse=True)
        
        for i, parent_id in enumerate(sorted_ids):
            parent_geom = catchments[parent_id]['geometry']
            children = []
            
            # Check all smaller catchments
            for child_id in sorted_ids[i+1:]:
                child_geom = catchments[child_id]['geometry']
                
                # Check if child is mostly contained within parent (>95% overlap)
                try:
                    intersection = parent_geom.intersection(child_geom)
                    overlap_ratio = intersection.area / child_geom.area
                    
                    if overlap_ratio > 0.95:  # 95% threshold for containment
                        children.append(child_id)
                        print(f"  🔗 {parent_id} contains {child_id} (overlap: {overlap_ratio*100:.1f}%)")
                except Exception as e:
                    print(f"  ⚠️ Error checking {parent_id} vs {child_id}: {e}")
            
            if children:
                relationships[parent_id] = children
        
        print(f"\n📊 Found {len(relationships)} parent catchments with nested subcatchments")
        return relationships
    
    def create_subcatchments(self, catchments, relationships):
        """
        Create subcatchment shapefiles by subtracting upstream areas.
        Creates both individual files and combined basin files.
        """
        print("\n✂️ Creating subcatchment shapefiles...")
        
        # Process catchments with nested subcatchments
        processed = set()
        
        for parent_id, child_ids in relationships.items():
            print(f"\n📁 Processing catchment system: {parent_id}")
            parent_geom = catchments[parent_id]['geometry']
            parent_crs = catchments[parent_id]['gdf'].crs
            
            # Sort children by area (largest to smallest)
            child_ids_sorted = sorted(child_ids, 
                                     key=lambda x: catchments[x]['area'], 
                                     reverse=True)
            
            # Track which areas have been "used" by upstream catchments
            remaining_area = parent_geom
            
            # Store all subcatchments for the combined basin file
            basin_subcatchments = []
            
            # Process each child (from headwater to downstream)
            for child_id in reversed(child_ids_sorted):  # Start with smallest (headwater)
                child_geom = catchments[child_id]['geometry']
                
                # Headwater catchments keep their full area
                subcatch_geom = child_geom
                subcatch_area_km2 = subcatch_geom.area / 1e6
                
                # Save individual subcatchment shapefile
                output_file = self.output_dir / f"subcatchment_{parent_id}_{child_id}.shp"
                self._save_subcatchment(catchments[child_id]['gdf'], 
                                       subcatch_geom, 
                                       output_file,
                                       parent_id,
                                       child_id,
                                       subcatch_area_km2)
                
                # Add to basin subcatchments list
                basin_subcatchments.append({
                    'gauge_id': child_id,
                    'subcatch_id': child_id,
                    'area_km2': subcatch_area_km2,
                    'geometry': subcatch_geom
                })
                
                # Remove this area from remaining parent area
                try:
                    remaining_area = remaining_area.difference(child_geom)
                except Exception as e:
                    print(f"  ⚠️ Geometry error for {child_id}: {e}")
                
                processed.add(child_id)
            
            # Create subcatchment for the remaining parent area (outlet subcatchment)
            if remaining_area.area > 0:
                outlet_area_km2 = remaining_area.area / 1e6
                output_file = self.output_dir / f"subcatchment_{parent_id}_outlet.shp"
                self._save_subcatchment(catchments[parent_id]['gdf'], 
                                       remaining_area, 
                                       output_file,
                                       parent_id,
                                       f"{parent_id}_outlet",
                                       outlet_area_km2)
                
                # Add outlet to basin subcatchments list
                basin_subcatchments.append({
                    'gauge_id': parent_id,
                    'subcatch_id': f"{parent_id}_outlet",
                    'area_km2': outlet_area_km2,
                    'geometry': remaining_area
                })
            
            processed.add(parent_id)
            
            # Create combined basin file with all subcatchments
            if basin_subcatchments:
                basin_file = self.output_dir / f"basin_{parent_id}_subcatchments.shp"
                self._save_basin_file(basin_subcatchments, basin_file, parent_crs, parent_id)
        
        # Copy standalone catchments (no nesting) as-is
        print("\n📋 Copying standalone catchments...")
        for gauge_id, data in catchments.items():
            if gauge_id not in processed:
                subcatch_area_km2 = data['geometry'].area / 1e6
                output_file = self.output_dir / f"subcatchment_{gauge_id}.shp"
                self._save_subcatchment(data['gdf'], 
                                       data['geometry'], 
                                       output_file,
                                       gauge_id,
                                       gauge_id,
                                       subcatch_area_km2)
                
                # Also create a basin file (with single subcatchment)
                basin_file = self.output_dir / f"basin_{gauge_id}_subcatchments.shp"
                basin_subcatchments = [{
                    'gauge_id': gauge_id,
                    'subcatch_id': gauge_id,
                    'area_km2': subcatch_area_km2,
                    'geometry': data['geometry']
                }]
                self._save_basin_file(basin_subcatchments, basin_file, data['gdf'].crs, gauge_id)
                
                print(f"  ✅ Copied standalone catchment: {gauge_id}")
    
    def _save_subcatchment(self, original_gdf, geometry, output_file, parent_id, subcatch_id, area_km2):
        """Save a subcatchment shapefile with proper attributes"""
        try:
            # Create new GeoDataFrame with the subcatchment geometry
            subcatch_gdf = gpd.GeoDataFrame(
                {'parent_id': [parent_id],
                 'subcatch_id': [subcatch_id],
                 'area_km2': [area_km2],
                 'geometry': [geometry]},
                crs=original_gdf.crs
            )
            
            # Save to file
            subcatch_gdf.to_file(output_file)
            print(f"  ✅ Saved: {output_file.name} (area: {area_km2:.2f} km²)")
            
        except Exception as e:
            print(f"  ❌ Error saving {output_file.name}: {e}")
    
    def _save_basin_file(self, basin_subcatchments, output_file, crs, parent_id):
        """Save a combined basin file with all subcatchments as separate features"""
        try:
            # Create GeoDataFrame with all subcatchments
            basin_gdf = gpd.GeoDataFrame(basin_subcatchments, crs=crs)
            
            # Add basin_id column
            basin_gdf['basin_id'] = parent_id
            
            # Reorder columns
            basin_gdf = basin_gdf[['basin_id', 'gauge_id', 'subcatch_id', 'area_km2', 'geometry']]
            
            # Save to file
            basin_gdf.to_file(output_file)
            
            total_area = basin_gdf['area_km2'].sum()
            print(f"  🗂️ Saved combined basin file: {output_file.name}")
            print(f"     {len(basin_gdf)} subcatchments, total area: {total_area:.2f} km²")
            
        except Exception as e:
            print(f"  ❌ Error saving basin file {output_file.name}: {e}")
    
    def process(self):
        """Main processing pipeline"""
        print("="*80)
        print("🌊 SUBCATCHMENT GROUPER")
        print("="*80)
        
        # Load all catchments
        catchments = self.load_all_catchments()
        
        if not catchments:
            print("❌ No catchments found!")
            return
        
        # Find nested relationships
        relationships = self.find_nested_relationships(catchments)
        
        # Create subcatchment files
        self.create_subcatchments(catchments, relationships)
        
        print("\n" + "="*80)
        print("✅ SUBCATCHMENT PROCESSING COMPLETE")
        print(f"📁 Output directory: {self.output_dir}")
        print("="*80)


def main():
    """Main execution"""
    
    # Define paths
    input_dir = "/home/jberg/OneDrive/Raven_worldwide/01_data/topo/catchment_shapefile"
    output_dir = "/home/jberg/OneDrive/Raven_worldwide/01_data/topo/subcatchment_shapefiles"
    
    # Create processor and run
    processor = SubcatchmentGrouper(input_dir, output_dir)
    processor.process()


if __name__ == "__main__":
    main()