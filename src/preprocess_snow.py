#### SnowProcessor for SLF OSHD daily SWE — Swiss catchments.
#### Ports Raven-Switzerland/src/preprocess_snow.py onto the composable
#### MeteoBase architecture used by ERA5-Land / HAR / MeteoSwiss.
####
#### Source: SWECLQMD_ch01h.swiss.lv95_WY_<years>.nc
####   - var SWECLQMD (m), dims (time, N, E), EPSG:2056
####   - converted to mm on output
####
#### Outputs (under paths.py-resolved dirs):
####   topo_shared_dir/swe.nc                    — catchment-clipped SWE
####   data_obs_dir/swe_by_elevation_band.csv    — per-band time series
####   data_obs_dir/elevation_band_areas.csv     — band areas
####   data_obs_dir/swe_by_hru.csv               — per-HRU time series
####   data_obs_dir/hru_info.csv                 — HRU summary
####
#### Glacier HRUs (Landuse_Cl in {7, 8}) are excluded from elevation-band
#### averaging (SWE on glaciers is treated separately by GloGEM).  Per-HRU
#### averaging keeps all HRUs for diagnostic completeness.
####
#### Justine Berg

from pathlib import Path
from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers .rio accessor on xarray)
import xarray as xr
from shapely.geometry import mapping

import warnings
warnings.filterwarnings('ignore')

from paths import get_paths
from preprocess_meteo_base import MeteoBase


#--------------------------------------------------------------------------------
############################### SnowProcessor ###################################
#--------------------------------------------------------------------------------

class SnowProcessor(MeteoBase):
    """
    Preprocess SLF OSHD daily SWE for a Swiss catchment.

    Clips the gridded SWE NetCDF to the catchment, converts m → mm, then
    aggregates to (a) elevation bands defined by HRU Elev_Min/Elev_Max and
    (b) individual HRUs.  Designed as a preprocessing-only step — the
    outputs are not yet wired to calibration.
    """

    _logger_class_name = 'SnowProcessor'
    SWE_VAR = 'SWECLQMD'

    def __init__(self, namelist_path: Union[str, Path], force_reprocess: bool = False) -> None:
        super().__init__(namelist_path, force_reprocess)

        # Source SWE file
        swe_dir = self.config.get('swe_dir')
        if swe_dir is None:
            raise ValueError(
                "`swe_dir` not set in namelist — required for SnowProcessor "
                "(typically provided by the switzerland region layer)"
            )
        self.swe_dir = Path(swe_dir)
        if not self.swe_dir.is_absolute():
            self.swe_dir = self.main_dir / self.swe_dir

        if not self.swe_dir.exists():
            raise FileNotFoundError(f"SWE source file not found: {self.swe_dir}")

        # Output paths via paths.py
        paths = get_paths(self.config)
        self.swe_data_path: Path = paths['topo_shared_dir'] / 'swe.nc'
        self.hru_path: Path = paths['topo_dir'] / 'HRU.shp'
        self.elev_csv_path: Path = paths['data_obs_dir'] / 'swe_by_elevation_band.csv'
        self.elev_area_csv_path: Path = paths['data_obs_dir'] / 'elevation_band_areas.csv'
        self.hru_swe_csv_path: Path = paths['data_obs_dir'] / 'swe_by_hru.csv'
        self.hru_info_csv_path: Path = paths['data_obs_dir'] / 'hru_info.csv'

        self.logger.info(f"SnowProcessor initialized for gauge {self.gauge_id}")
        self.logger.info(f"SWE source: {self.swe_dir}")

    #---------------------------------------------------------------------------------

    def clip_swe_data(self) -> xr.Dataset:
        """
        Clip the source SWE NetCDF to the catchment extent + time range,
        convert m → mm, and save to topo_shared_dir/swe.nc.
        """
        if self.swe_data_path.exists() and not self.force_reprocess:
            self.logger.info(f"  ✅ {self.swe_data_path.name} exists, skipping clip")
            return xr.open_dataset(self.swe_data_path)

        if self.catchment_extent is None:
            raise RuntimeError("Catchment shapefile not loaded — cannot clip SWE")

        # MeteoBase loads catchment_extent in WGS84; SWE is in Swiss LV95 (EPSG:2056)
        extent = self.catchment_extent.to_crs("EPSG:2056").buffer(10)

        self.logger.info(f"  Opening SWE file ({self.swe_dir.name})…")
        xds = xr.open_mfdataset(self.swe_dir, chunks={"time": 100}).astype('float32')
        xds.rio.set_spatial_dims(x_dim="E", y_dim="N", inplace=True)
        xds.rio.write_crs("EPSG:2056", inplace=True)

        self.logger.info("  Clipping to catchment extent…")
        clipped = xds.rio.clip(
            extent.geometry.apply(mapping),
            extent.crs,
            drop=True,
            all_touched=True,
        )

        # Time slice (snow is observational so no warm-up logic needed)
        clipped = clipped.sel(time=slice(self.start_date, self.end_date))
        if clipped.sizes.get('time', 0) == 0:
            raise ValueError(
                f"No SWE time steps in [{self.start_date.date()}, {self.end_date.date()}]"
            )

        # m → mm
        if self.SWE_VAR not in clipped.data_vars:
            raise KeyError(f"Expected variable '{self.SWE_VAR}' not in SWE dataset")
        clipped[self.SWE_VAR] = clipped[self.SWE_VAR] * 1000
        clipped[self.SWE_VAR].attrs['units'] = 'mm'

        # Strip grid_mapping attrs to keep NetCDF writer happy
        for v in clipped.data_vars:
            clipped[v].attrs.pop('grid_mapping', None)

        self.swe_data_path.parent.mkdir(parents=True, exist_ok=True)
        clipped.to_netcdf(self.swe_data_path)
        self.logger.info(
            f"  ✅ Saved {self.swe_data_path.name} "
            f"({clipped.sizes['time']} days, "
            f"{clipped.sizes.get('N', '?')}×{clipped.sizes.get('E', '?')} cells)"
        )
        return clipped

    #---------------------------------------------------------------------------------

    def _load_hrus_lv95(self) -> gpd.GeoDataFrame:
        """Load the HRU shapefile and reproject to EPSG:2056 if needed."""
        if not self.hru_path.exists():
            raise FileNotFoundError(
                f"HRU shapefile not found: {self.hru_path}. "
                "Run catchment preprocessing first."
            )
        hru = gpd.read_file(self.hru_path)
        if hru.crs is None:
            self.logger.warning("HRU shapefile has no CRS — assuming EPSG:2056")
            hru.set_crs("EPSG:2056", inplace=True)
        elif hru.crs.to_string() != "EPSG:2056":
            hru = hru.to_crs("EPSG:2056")
        # Fix any invalid geometries (zero-width buffer is the standard trick)
        hru['geometry'] = hru.geometry.buffer(0)
        return hru

    #---------------------------------------------------------------------------------

    def _clip_swe_to_geometry(
        self,
        swe_ds: xr.Dataset,
        geometry,
        crs,
    ) -> Optional[xr.DataArray]:
        """
        Clip SWE to a single geometry and return the spatial-mean time series.
        Tries a 1-metre buffer as fallback for tiny / pathological HRUs.
        Returns None if both attempts fail.
        """
        try:
            clipped = swe_ds.rio.clip([geometry], crs, all_touched=True)
            return clipped[self.SWE_VAR].mean(dim=['N', 'E'], skipna=True)
        except Exception as e1:
            self.logger.debug(f"  primary clip failed ({e1}); retrying with 1 m buffer")
            try:
                clipped = swe_ds.rio.clip(
                    [geometry.buffer(1)], crs, all_touched=True
                )
                return clipped[self.SWE_VAR].mean(dim=['N', 'E'], skipna=True)
            except Exception as e2:
                self.logger.debug(f"  buffer fallback also failed ({e2})")
                return None

    #---------------------------------------------------------------------------------

    def average_swe_per_elevation_band(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate SWE to elevation bands (defined by HRU Elev_Min/Elev_Max).
        Excludes glacier HRUs (Landuse_Cl in {7, 8}).
        Writes swe_by_elevation_band.csv and elevation_band_areas.csv.
        """
        if (
            self.elev_csv_path.exists()
            and self.elev_area_csv_path.exists()
            and not self.force_reprocess
        ):
            self.logger.info(
                f"  ✅ {self.elev_csv_path.name} + areas exist, skipping band aggregation"
            )
            return (
                pd.read_csv(self.elev_csv_path, index_col=0, parse_dates=True),
                pd.read_csv(self.elev_area_csv_path, index_col=0),
            )

        hru = self._load_hrus_lv95()
        # Glacier HRUs handled separately by GloGEM
        non_glacier = hru[~hru['Landuse_Cl'].isin([7, 8])].copy()
        if non_glacier.empty:
            self.logger.warning("No non-glacier HRUs — skipping band aggregation")
            return pd.DataFrame(), pd.DataFrame()

        non_glacier['ElevationBand'] = non_glacier.apply(
            lambda r: f"{int(r['Elev_Min'])}-{int(r['Elev_Max'])}m", axis=1
        )

        # Total non-glacier area per band (pre-dissolve)
        area_by_band = non_glacier.groupby('ElevationBand')['Area_km2'].sum().to_dict()

        bands_gdf = non_glacier.dissolve(by='ElevationBand').reset_index()
        bands_gdf['Area_km2'] = bands_gdf['ElevationBand'].map(area_by_band)

        # Load clipped SWE
        swe_ds = xr.open_dataset(self.swe_data_path)
        swe_ds.rio.write_crs("EPSG:2056", inplace=True)

        swe_by_band: dict = {}
        for _, row in bands_gdf.iterrows():
            band = row['ElevationBand']
            series = self._clip_swe_to_geometry(swe_ds, row['geometry'], bands_gdf.crs)
            if series is None:
                self.logger.warning(f"  band {band}: clip failed, skipping")
                continue
            swe_by_band[band] = series.values

        if not swe_by_band:
            self.logger.error("No elevation bands successfully processed")
            swe_ds.close()
            return pd.DataFrame(), pd.DataFrame()

        df_swe = pd.DataFrame(swe_by_band, index=swe_ds.time.values)
        df_swe.index.name = 'time'

        area_df = pd.DataFrame(
            {'area_km2': [area_by_band[b] for b in swe_by_band.keys()]},
            index=list(swe_by_band.keys()),
        )
        area_df.index.name = 'ElevationBand'

        self.elev_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_swe.to_csv(self.elev_csv_path)
        area_df.to_csv(self.elev_area_csv_path)
        swe_ds.close()

        self.logger.info(
            f"  ✅ Saved {self.elev_csv_path.name} ({len(swe_by_band)} bands)"
        )
        return df_swe, area_df

    #---------------------------------------------------------------------------------

    def average_swe_per_hru(self) -> pd.DataFrame:
        """
        Aggregate SWE per HRU (including glacier HRUs for completeness).
        Writes swe_by_hru.csv and hru_info.csv.
        """
        if (
            self.hru_swe_csv_path.exists()
            and self.hru_info_csv_path.exists()
            and not self.force_reprocess
        ):
            self.logger.info(
                f"  ✅ {self.hru_swe_csv_path.name} + info exist, skipping HRU aggregation"
            )
            return pd.read_csv(self.hru_swe_csv_path, index_col=0, parse_dates=True)

        hru = self._load_hrus_lv95()

        swe_ds = xr.open_dataset(self.swe_data_path)
        swe_ds.rio.write_crs("EPSG:2056", inplace=True)

        swe_by_hru: dict = {}
        info_rows: list = []
        n_failed = 0

        for _, row in hru.iterrows():
            hru_id = int(row['HRU_ID'])
            key = f'HRU_{hru_id}'
            series = self._clip_swe_to_geometry(swe_ds, row['geometry'], hru.crs)

            base_info = {
                'hru_id': hru_id,
                'area_km2': float(row['Area_km2']),
                'elev_min': float(row['Elev_Min']),
                'elev_max': float(row['Elev_Max']),
                'elev_mean': (float(row['Elev_Min']) + float(row['Elev_Max'])) / 2,
                'landuse_cl': int(row['Landuse_Cl']),
            }

            if series is None:
                n_failed += 1
                info_rows.append({**base_info, 'success': False})
                continue

            swe_by_hru[key] = series.values
            info_rows.append({
                **base_info,
                'success': True,
                'mean_swe': float(np.nanmean(series.values)),
                'max_swe': float(np.nanmax(series.values)),
                'min_swe': float(np.nanmin(series.values)),
            })

        if not swe_by_hru:
            self.logger.error("No HRUs successfully processed")
            swe_ds.close()
            return pd.DataFrame()

        df_swe = pd.DataFrame(swe_by_hru, index=swe_ds.time.values)
        df_swe.index.name = 'time'

        info_df = pd.DataFrame(info_rows).set_index(
            pd.Index([f'HRU_{r["hru_id"]}' for r in info_rows], name='hru')
        )

        self.hru_swe_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_swe.to_csv(self.hru_swe_csv_path)
        info_df.to_csv(self.hru_info_csv_path)
        swe_ds.close()

        self.logger.info(
            f"  ✅ Saved {self.hru_swe_csv_path.name} "
            f"({len(swe_by_hru)} HRUs ok, {n_failed} failed)"
        )
        return df_swe

    #---------------------------------------------------------------------------------

    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the full pipeline: clip → elevation bands → per-HRU.

        Returns
        -------
        (df_bands, df_hru) : tuple of DataFrames
        """
        self.logger.info(f"❄️  Processing SWE for catchment {self.gauge_id}")
        self.clip_swe_data()
        df_bands, _ = self.average_swe_per_elevation_band()
        df_hru = self.average_swe_per_hru()
        self.logger.info("✅ Snow processing complete")
        return df_bands, df_hru

    #---------------------------------------------------------------------------------

    def plot_diagnostics(self, save: bool = True) -> Optional[Path]:
        """
        Produce a 4-panel diagnostic figure of the processed SWE.

        Not called by `process()` — invoke explicitly when you want visuals.
        Reads the cached outputs, so it's cheap to call after a run.

        Returns
        -------
        Path to the saved PNG, or None if `save=False`.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        for p in (self.swe_data_path, self.elev_csv_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing output {p} — run process() first")

        ds = xr.open_dataset(self.swe_data_path)
        df_bands = pd.read_csv(self.elev_csv_path, index_col=0, parse_dates=True)
        area = pd.read_csv(self.elev_area_csv_path, index_col=0)

        # Order bands by elevation
        def _band_lo(name: str) -> int:
            return int(name.split('-')[0])
        ordered = sorted(df_bands.columns, key=_band_lo)
        df_bands = df_bands[ordered]
        area = area.loc[ordered]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (1) Time-mean spatial SWE
        mean_map = ds[self.SWE_VAR].mean(dim='time')
        im = mean_map.plot(ax=axes[0, 0], cmap='Blues', add_colorbar=True,
                           cbar_kwargs={'label': 'Mean SWE (mm)'})
        if self.catchment_extent is not None:
            self.catchment_extent.to_crs("EPSG:2056").boundary.plot(
                ax=axes[0, 0], color='red', linewidth=1.5)
        axes[0, 0].set_title('Time-mean SWE')
        axes[0, 0].set_xlabel('Easting (m)')
        axes[0, 0].set_ylabel('Northing (m)')

        # (2) Catchment-mean time series (area-weighted across bands)
        weights = area['area_km2'].values
        weights = weights / weights.sum()
        catchment_ts = (df_bands.values * weights).sum(axis=1)
        axes[0, 1].plot(df_bands.index, catchment_ts, color='steelblue', linewidth=1)
        axes[0, 1].set_title('Catchment-mean SWE (area-weighted)')
        axes[0, 1].set_ylabel('SWE (mm)')
        axes[0, 1].grid(alpha=0.4)

        # (3) SWE vs elevation, mean annual cycle (heatmap)
        df_bands_monthly = df_bands.groupby(df_bands.index.month).mean()
        elev_mid = [(_band_lo(c) + int(c.split('-')[1].rstrip('m'))) / 2 for c in ordered]
        pcm = axes[1, 0].pcolormesh(
            range(1, 13), elev_mid, df_bands_monthly.values.T,
            cmap='Blues', shading='auto')
        plt.colorbar(pcm, ax=axes[1, 0], label='SWE (mm)')
        axes[1, 0].set_title('Mean annual cycle by elevation')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Elevation (m)')
        axes[1, 0].set_xticks(range(1, 13))

        # (4) Mean & peak SWE vs elevation
        means = df_bands.mean()
        peaks = df_bands.max()
        axes[1, 1].plot(elev_mid, means.values, 'o-', label='Mean', color='steelblue')
        axes[1, 1].plot(elev_mid, peaks.values, 's--', label='Peak', color='navy')
        axes[1, 1].set_title('SWE vs elevation')
        axes[1, 1].set_xlabel('Elevation (m)')
        axes[1, 1].set_ylabel('SWE (mm)')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.4)

        fig.suptitle(f'Snow diagnostics — catchment {self.gauge_id} '
                     f'({df_bands.index[0].date()} → {df_bands.index[-1].date()})',
                     fontsize=13, fontweight='bold')
        fig.tight_layout()

        ds.close()

        if not save:
            return None

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        out = self.plots_dir / f'snow_diagnostics_{self.gauge_id}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"  ✅ Saved snow diagnostics to {out}")
        return out


#--------------------------------------------------------------------------------
############################ process_snow_forcing ###############################
#--------------------------------------------------------------------------------

def process_snow_forcing(namelist_path: Union[str, Path], force_reprocess: bool = False) -> None:
    """Convenience wrapper: instantiate SnowProcessor and run process()."""
    print("❄️  Processing SLF OSHD SWE data…")
    proc = SnowProcessor(namelist_path, force_reprocess=force_reprocess)
    proc.process()
    print("✅ Snow processing completed!")
