#### MultiSubbasinProcessor for Raven hydrological model
#### Justine Berg

import geopandas as gpd
import numpy as np
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, List, Union, Optional, Any, Tuple
import yaml

from preprocess_catchment_hru import CatchmentProcessor

#--------------------------------------------------------------------------------
############################### MultiSubbasinProcessor ##########################
#--------------------------------------------------------------------------------

class MultiSubbasinProcessor:
    """
    Orchestrates multi-subbasin catchment processing for Raven.

    Reads a namelist with a ``subbasins:`` list, processes each subbasin with a
    separate :class:`CatchmentProcessor` (using the local polygon as the clipping
    boundary), then merges the resulting HRU tables and shapefiles into globally
    unique IDs.  Also auto-derives the routing topology from spatial overlap of
    the per-subbasin catchment shapefiles and writes it to
    ``topo_files/subbasin_routing.yaml``.

    Single-basin namelists (no ``subbasins:`` key) continue to use
    :class:`CatchmentProcessor` directly; this class is never instantiated for
    them.
    """

    def __init__(self, namelist_path: Union[str, Path]):
        self.namelist_path = Path(namelist_path)
        with open(self.namelist_path) as f:
            self.config = yaml.safe_load(f)

        self.main_dir = Path(self.config['main_dir'])
        self.gauge_id = str(self.config['gauge_id'])
        self.subbasin_configs = self.config['subbasins']

        # Shared output directory  – same location CatchmentProcessor would use
        # for the main gauge, so HBVProcessor still finds HRU_table.csv there.
        catchment_dir = (
            self.main_dir / self.config['config_dir'] / f'catchment_{self.gauge_id}'
        )
        self.topo_dir = catchment_dir / 'topo_files'
        self.topo_dir.mkdir(parents=True, exist_ok=True)

        # Populated by compute_routing_topology()
        self.local_polygons: Dict[int, gpd.GeoDataFrame] = {}

        self.logger = logging.getLogger('MultiSubbasinProcessor')

    # -------------------------------------------------------------------------

    def compute_routing_topology(self) -> Dict[int, int]:
        """
        Derive routing topology and local (residual) subbasin polygons.

        For each subbasin, loads ``catchment_shape_{gauge_id}.shp`` and checks
        95 % area-overlap containment (largest-first).  The largest polygon
        becomes the outlet (``downstream_id = -1``); smaller ones point to their
        smallest enclosing parent.

        Local polygons are computed by subtracting *direct* children from the
        parent polygon, so the outlet subbasin only covers the residual area.

        Saves the routing dict to ``topo_files/subbasin_routing.yaml``.

        Returns
        -------
        dict
            ``{subbasin_id: downstream_subbasin_id}``  (-1 for outlet).
        """
        shape_template = self.config['shape_dir']  # e.g. '01_data/.../catchment_shape_{gauge_id}.shp'

        # --- load catchment shapefiles ---
        catchments: Dict[int, Dict] = {}
        for sb in self.subbasin_configs:
            sb_id = sb['id']
            gauge_id = sb['gauge_id']
            shp_path = self.main_dir / shape_template.format(gauge_id=gauge_id)
            if not shp_path.exists():
                raise FileNotFoundError(
                    f"Catchment shapefile not found for subbasin {sb_id} "
                    f"(gauge_id={gauge_id}): {shp_path}"
                )
            gdf = gpd.read_file(shp_path)
            geom = gdf.geometry.unary_union
            catchments[sb_id] = {
                'geometry': geom,
                'gdf': gdf,
                'area': geom.area,
                'gauge_id': gauge_id,
            }
            self.logger.info(f"Loaded subbasin {sb_id} ({gauge_id}): area={geom.area:.2e}")

        # --- sort by area descending ---
        sorted_ids = sorted(catchments.keys(),
                            key=lambda x: catchments[x]['area'],
                            reverse=True)

        # --- find direct parent for each subbasin (smallest enclosing polygon) ---
        # A subbasin may be contained by several larger ones (e.g. 0104 fits inside
        # both 0105 and 0109).  We want the *most immediate* parent, i.e. the
        # smallest polygon that still contains the child with ≥95 % overlap.
        # parent_of[child_id] = parent_id
        parent_of: Dict[int, int] = {}
        for child_idx in range(1, len(sorted_ids)):        # skip the largest (it has no parent)
            child_id   = sorted_ids[child_idx]
            child_geom = catchments[child_id]['geometry']

            # collect ALL polygons that contain this child
            containing: list = []   # list of (parent_area, parent_id)
            for parent_id in sorted_ids[:child_idx]:    # only strictly larger polygons
                parent_geom = catchments[parent_id]['geometry']
                try:
                    intersection = parent_geom.intersection(child_geom)
                    overlap_ratio = intersection.area / child_geom.area
                    if overlap_ratio > 0.95:
                        containing.append((catchments[parent_id]['area'], parent_id))
                except Exception as exc:
                    self.logger.warning(f"Geometry error checking {parent_id} vs {child_id}: {exc}")

            if containing:
                # direct parent = smallest containing polygon
                containing.sort()                          # ascending area
                direct_parent_id = containing[0][1]
                parent_of[child_id] = direct_parent_id
                self.logger.info(
                    f"Routing: subbasin {child_id} -> {direct_parent_id} "
                    f"(direct parent, {len(containing)} containing polygon(s))"
                )

        # --- build routing dict ---
        routing: Dict[int, int] = {}
        for sb_id in sorted_ids:
            routing[sb_id] = parent_of.get(sb_id, -1)

        # --- compute local (residual) polygons ---
        # direct_children[parent_id] = [child_id, ...]
        direct_children: Dict[int, list] = {sb_id: [] for sb_id in sorted_ids}
        for child_id, par_id in parent_of.items():
            direct_children[par_id].append(child_id)

        for sb_id in sorted_ids:
            geom = catchments[sb_id]['geometry']
            crs = catchments[sb_id]['gdf'].crs
            for child_id in direct_children[sb_id]:
                try:
                    geom = geom.difference(catchments[child_id]['geometry'])
                except Exception as exc:
                    self.logger.warning(f"Could not subtract subbasin {child_id} from {sb_id}: {exc}")
            self.local_polygons[sb_id] = gpd.GeoDataFrame(geometry=[geom], crs=crs)

        # --- persist routing ---
        routing_path = self.topo_dir / 'subbasin_routing.yaml'
        with open(routing_path, 'w') as f:
            yaml.dump(routing, f)
        self.logger.info(f"Routing saved to {routing_path}")

        return routing

    # -------------------------------------------------------------------------

    def process_all_subbasins(self) -> pd.DataFrame:
        """
        Process every subbasin and return the combined HRU table.

        Steps
        -----
        1. ``compute_routing_topology()``
        2. For each subbasin: run :class:`CatchmentProcessor` with its local
           polygon pre-set (skips shapefile re-read) and per-subbasin model_dir.
        3. ``_merge_hru_tables()``
        4. ``_merge_hru_shapefiles()``

        Returns the combined HRU table (also written to
        ``topo_files/HRU_table.csv`` and ``topo_files/HRU.txt``).
        """
        self.logger.info("Multi-subbasin processor starting")
        self.compute_routing_topology()

        results = []  # list of (processor, hru_table, subbasin_id)

        for sb in self.subbasin_configs:
            sb_id = int(sb['id'])
            sb_gauge_id = str(sb['gauge_id'])
            self.logger.info(f"Processing subbasin {sb_id} (gauge {sb_gauge_id})")

            processor = CatchmentProcessor(self.namelist_path)

            # Override per-subbasin attributes
            processor.gauge_id = sb_gauge_id
            processor.shape_dir = (
                self.main_dir / self.config['shape_dir'].format(gauge_id=sb_gauge_id)
            )
            processor._basin_id = sb_id

            # Redirect output to a subbasin-specific subdirectory
            processor.model_dir = self.topo_dir / f'subbasin_{sb_id}'
            processor._create_output_dir()

            # Pre-set the local polygon → process_catchment() will skip extract_catchment_shape()
            if sb_id in self.local_polygons:
                processor.catchment_extent = self.local_polygons[sb_id]

            hru_table = processor.process_catchment()
            results.append((processor, hru_table, sb_id))
            self.logger.info(f"Subbasin {sb_id}: {len(hru_table)} HRUs created")

        combined = self._merge_hru_tables(results)
        self._merge_hru_shapefiles(results)

        self.logger.info(f"Combined HRU table: {len(combined)} HRUs across {len(self.subbasin_configs)} subbasins")
        return combined

    # -------------------------------------------------------------------------

    def _merge_hru_tables(
        self, results: List[Tuple]
    ) -> pd.DataFrame:
        """
        Concatenate per-subbasin HRU tables with globally unique HRU IDs.

        Shifts ``:ATTRIBUTES`` (HRU ID) by a running offset and sets
        ``BASIN_ID`` to the subbasin id.  Writes merged files to
        ``topo_files/HRU_table.csv`` and ``topo_files/HRU.txt``.
        """
        combined_rows = []
        hru_id_offset = 0

        for processor, hru_table, sb_id in results:
            tbl = hru_table.copy()
            tbl[':ATTRIBUTES'] = tbl[':ATTRIBUTES'] + hru_id_offset
            tbl['BASIN_ID'] = sb_id
            combined_rows.append(tbl)
            hru_id_offset += len(tbl)

        combined = pd.concat(combined_rows, ignore_index=True)

        combined.to_csv(self.topo_dir / 'HRU_table.csv', index=False)

        with open(self.topo_dir / 'HRU.txt', 'w') as f:
            f.write(' '.join(combined.columns) + '\n')
            for _, row in combined.iterrows():
                f.write(' '.join(map(str, row)) + '\n')

        self.logger.info(f"Merged HRU table written to {self.topo_dir / 'HRU_table.csv'}")
        return combined

    # -------------------------------------------------------------------------

    def _merge_hru_shapefiles(self, results: List[Tuple]) -> None:
        """
        Concatenate per-subbasin HRU shapefiles with globally unique HRU IDs.

        Shifts ``HRU_ID`` by a running offset and adds a ``Basin_ID`` column.
        Writes merged shapefile to ``topo_files/HRU.shp``.
        """
        gdfs = []
        hru_id_offset = 0

        for processor, hru_table, sb_id in results:
            shp_path = processor.get_path('HRU.shp')
            if not shp_path.exists():
                self.logger.warning(f"HRU shapefile not found for subbasin {sb_id}: {shp_path}")
                hru_id_offset += len(hru_table)
                continue

            gdf = gpd.read_file(shp_path)
            gdf['HRU_ID'] = gdf['HRU_ID'] + hru_id_offset
            gdf['Basin_ID'] = sb_id
            gdfs.append(gdf)
            hru_id_offset += len(gdf)

        if gdfs:
            combined_gdf = gpd.GeoDataFrame(
                pd.concat(gdfs, ignore_index=True),
                crs=gdfs[0].crs,
            )
            out_path = self.topo_dir / 'HRU.shp'
            combined_gdf.to_file(out_path, driver='ESRI Shapefile')
            self.logger.info(f"Merged HRU shapefile written to {out_path}")


#---------------------------------------------------------------------------------

