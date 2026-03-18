"""
Presentation-ready figures for catchment overview and meteo validation.

Functions for generating publication/presentation quality plots of:
- Catchment DEM, landuse, glacier outlines
- AWS vs ERA5-Land / HAR temperature and precipitation comparisons
"""

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import xarray as xr
import yaml
from pathlib import Path


# ============================================================
# Landuse class definitions
# ============================================================
LU_NAMES = {
    1: 'Forest', 2: 'Open vegetation', 3: 'Cropland', 4: 'Built-up',
    5: 'Rock / Bare', 6: 'Water', 7: 'Glacier', 9: 'Bare open',
}

LU_COLORS = {
    1: '#228B22', 2: '#90EE90', 3: '#FFD700', 4: '#FF4500',
    5: '#A0522D', 6: '#4169E1', 7: '#87CEEB', 9: '#DEB887',
}


def _setup_presentation_style():
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 13,
        'figure.facecolor': 'white',
    })


def _reproject_to_wgs84(src_path, resampling=Resampling.bilinear):
    """Read a raster and reproject to EPSG:4326 if needed."""
    with rasterio.open(src_path) as src:
        if src.crs and src.crs.to_epsg() != 4326:
            transform, width, height = calculate_default_transform(
                src.crs, 'EPSG:4326', src.width, src.height, *src.bounds)
            data = np.empty((height, width), dtype=np.float64)
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs='EPSG:4326',
                resampling=resampling,
            )
            bounds = rasterio.transform.array_bounds(height, width, transform)
            nodata = src.nodata
        else:
            data = src.read(1).astype(float)
            bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
            nodata = src.nodata
    data = data.astype(float)
    if nodata is not None:
        data[data == nodata] = np.nan
    # bounds: (left, bottom, right, top) -> extent: [left, right, bottom, top]
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    return data, extent


def _load_dem(topo_dir):
    dem, extent = _reproject_to_wgs84(topo_dir / 'clipped_dem.tif', Resampling.bilinear)
    return dem, extent


def _load_landuse(topo_dir):
    lu, extent = _reproject_to_wgs84(topo_dir / 'reclassified_landuse.tif', Resampling.nearest)
    lu[lu == 0] = np.nan
    return lu, extent


def _catchment_xlim_ylim(catchment, pad_frac=0.05):
    """Get xlim/ylim from catchment bounds with padding."""
    b = catchment.total_bounds  # [minx, miny, maxx, maxy]
    dx = (b[2] - b[0]) * pad_frac
    dy = (b[3] - b[1]) * pad_frac
    return [b[0] - dx, b[2] + dx], [b[1] - dy, b[3] + dy]


def _format_geo_axes(ax):
    """Format axes with bold tick labels including °E / °N suffixes. No title or axis labels."""
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}\u00b0E'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}\u00b0N'))
    ax.tick_params(axis='both', labelsize=12)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_rotation(90)
        label.set_verticalalignment('center')


# ============================================================
# Catchment overview figures
# ============================================================

def _compute_hillshade(dem):
    """Compute hillshade from DEM array (NW light, 45° altitude)."""
    from numpy import gradient
    dy, dx = gradient(np.where(np.isnan(dem), 0, dem))
    slope = np.sqrt(dx**2 + dy**2)
    aspect = np.arctan2(-dx, dy)
    az, alt = np.radians(315), np.radians(45)
    hs = np.sin(alt) + np.cos(alt) * np.cos(aspect - az) * slope
    return (hs - hs.min()) / (hs.max() - hs.min())


def plot_catchment_discretization(topo_dir, outdir):
    """Two clean panels: (a) DEM, (b) Landuse + glacier overlay. No axes."""
    _setup_presentation_style()
    topo_dir = Path(topo_dir)
    outdir = Path(outdir)

    dem, dem_extent = _load_dem(topo_dir)
    lu, lu_extent = _load_landuse(topo_dir)

    # Landuse colormap
    all_classes = sorted(LU_COLORS.keys())
    lu_cmap = ListedColormap([LU_COLORS[c] for c in all_classes])
    bounds = [c - 0.5 for c in all_classes] + [all_classes[-1] + 0.5]
    lu_norm = BoundaryNorm(bounds, lu_cmap.N)

    # Glacier overlay
    glacier_path = topo_dir / 'clipped_glacier.shp'
    glaciers = None
    if glacier_path.exists():
        glaciers = gpd.read_file(glacier_path)
        if glaciers.crs and glaciers.crs.to_epsg() != 4326:
            glaciers = glaciers.to_crs('EPSG:4326')

    # ---- DEM plot ----
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    ax1.imshow(dem, cmap='terrain', extent=dem_extent, interpolation='bilinear',
               vmin=np.nanmin(dem), vmax=np.nanmax(dem))
    ax1.set_axis_off()
    out1 = outdir / 'presentation_dem.png'
    fig1.savefig(out1, dpi=300, bbox_inches='tight', pad_inches=0.05,
                 facecolor='white')
    print(f"Saved: {out1}")
    plt.close(fig1)

    # ---- Landuse + glaciers plot ----
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.imshow(lu, cmap=lu_cmap, norm=lu_norm, extent=lu_extent, interpolation='nearest')
    if glaciers is not None:
        glaciers.plot(ax=ax2, facecolor='#87CEEB', edgecolor='darkblue',
                      linewidth=0.8, alpha=0.6)
    ax2.set_axis_off()
    out2 = outdir / 'presentation_landuse.png'
    fig2.savefig(out2, dpi=300, bbox_inches='tight', pad_inches=0.05,
                 facecolor='white')
    print(f"Saved: {out2}")
    plt.close(fig2)


def plot_catchment_dem(topo_dir, catchment_shp, outdir, glacier_shp=None):
    """Clean DEM with catchment boundary (no glaciers by default)."""
    _setup_presentation_style()
    dem, extent = _load_dem(topo_dir)

    catchment = gpd.read_file(catchment_shp)
    if catchment.crs and catchment.crs.to_epsg() != 4326:
        catchment = catchment.to_crs('EPSG:4326')
    xlim, ylim = _catchment_xlim_ylim(catchment)

    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(dem, cmap='terrain', extent=extent, interpolation='bilinear',
                   vmin=np.nanmin(dem), vmax=np.nanmax(dem))
    catchment.boundary.plot(ax=ax, color='black', linewidth=2)

    legend_elements = [Line2D([0], [0], color='black', linewidth=2, label='Catchment boundary')]

    if glacier_shp is not None:
        glaciers = gpd.read_file(glacier_shp)
        if glaciers.crs != catchment.crs:
            glaciers = glaciers.to_crs(catchment.crs)
        glaciers.plot(ax=ax, facecolor='none', edgecolor='cyan', linewidth=1, alpha=0.9)
        legend_elements.append(Line2D([0], [0], color='cyan', linewidth=1.5, label='RGI 6.0 glacier outlines'))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax).set_label('Elevation (m a.s.l.)')

    ax.legend(handles=legend_elements, loc='lower left', fontsize=11, frameon=True,
              facecolor='white', edgecolor='gray', framealpha=0.9)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _format_geo_axes(ax)

    plt.tight_layout()
    out_path = Path(outdir) / 'presentation_dem.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_catchment_landuse(topo_dir, catchment_shp, outdir, title_prefix='ESA WorldCover'):
    """Landuse raster with class percentages from raster pixels."""
    _setup_presentation_style()
    lu, extent = _load_landuse(topo_dir)
    catchment = gpd.read_file(catchment_shp)
    if catchment.crs and catchment.crs.to_epsg() != 4326:
        catchment = catchment.to_crs('EPSG:4326')
    xlim, ylim = _catchment_xlim_ylim(catchment)

    all_classes = sorted(LU_COLORS.keys())
    cmap = ListedColormap([LU_COLORS[c] for c in all_classes])
    bounds = [c - 0.5 for c in all_classes] + [all_classes[-1] + 0.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(lu, cmap=cmap, norm=norm, extent=extent, interpolation='nearest')
    catchment.boundary.plot(ax=ax, color='black', linewidth=2)

    valid = lu[~np.isnan(lu)].astype(int)
    unique, counts = np.unique(valid, return_counts=True)
    total_px = counts.sum()

    patches = []
    for cl in all_classes:
        idx = np.where(unique == cl)[0]
        pct = counts[idx[0]] / total_px * 100 if len(idx) > 0 else 0
        if pct > 0.01:
            patches.append(mpatches.Patch(color=LU_COLORS[cl],
                           label=f'{LU_NAMES[cl]} ({pct:.1f}%)'))

    ax.legend(handles=patches, loc='lower left', fontsize=10, frameon=True,
              facecolor='white', edgecolor='gray', framealpha=0.9)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _format_geo_axes(ax)

    plt.tight_layout()
    out_path = Path(outdir) / 'presentation_landuse.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_catchment_landuse_glaciers(topo_dir, catchment_shp, glacier_shp, hru_shp, outdir,
                                     title_prefix='ESA WorldCover'):
    """Landuse raster with RGI6 glacier outlines overlaid.

    Uses HRU shapefile Landuse_Cl attribute for area statistics to correctly
    distinguish all reclassified classes (e.g. classes 2, 3, 9 which share
    the same LAND_USE_CLASS name in HRU_table.csv).
    """
    _setup_presentation_style()
    lu, extent = _load_landuse(topo_dir)
    catchment = gpd.read_file(catchment_shp)
    if catchment.crs and catchment.crs.to_epsg() != 4326:
        catchment = catchment.to_crs('EPSG:4326')
    xlim, ylim = _catchment_xlim_ylim(catchment)

    glaciers = gpd.read_file(glacier_shp)
    if glaciers.crs != catchment.crs:
        glaciers = glaciers.to_crs(catchment.crs)

    # Compute areas from HRU shapefile using Landuse_Cl attribute
    hru = gpd.read_file(hru_shp)
    hru_utm = hru.to_crs(hru.estimate_utm_crs())
    hru_utm['area_km2'] = hru_utm.geometry.area / 1e6
    area_by_class = hru_utm.groupby('Landuse_Cl')['area_km2'].sum()
    total_area = area_by_class.sum()

    # Glacier area from RGI
    glaciers_utm = glaciers.to_crs(hru_utm.crs)
    catchment_utm = catchment.to_crs(hru_utm.crs)
    glacier_area = glaciers_utm.geometry.area.sum() / 1e6
    catchment_area = catchment_utm.geometry.area.sum() / 1e6
    glacier_pct = glacier_area / catchment_area * 100

    # Remove glacier class from landuse (class 7) since we show RGI separately
    all_classes = sorted(c for c in LU_COLORS.keys() if c != 7)
    glacier_color = '#87CEEB'

    cmap = ListedColormap([LU_COLORS[c] for c in sorted(LU_COLORS.keys())])
    bounds_cm = [c - 0.5 for c in sorted(LU_COLORS.keys())] + [max(LU_COLORS.keys()) + 0.5]
    norm = BoundaryNorm(bounds_cm, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(lu, cmap=cmap, norm=norm, extent=extent, interpolation='nearest')
    glaciers.plot(ax=ax, facecolor=glacier_color, edgecolor='darkblue', linewidth=0.6, alpha=0.7)
    catchment.boundary.plot(ax=ax, color='black', linewidth=2)

    # Build legend with areas from HRU shapefile
    patches = []
    for cl in all_classes:
        if cl in area_by_class.index:
            area = area_by_class[cl]
            pct = area / catchment_area * 100
            if pct > 0.01:
                patches.append(mpatches.Patch(color=LU_COLORS[cl],
                               label=f'{LU_NAMES[cl]} ({area:.0f} km\u00b2, {pct:.1f}%)'))
    # Add glacier entry
    patches.append(mpatches.Patch(facecolor=glacier_color, edgecolor='darkblue',
                   label=f'Glacier RGI 6.0 ({glacier_area:.0f} km\u00b2, {glacier_pct:.1f}%)'))

    ax.legend(handles=patches, loc='lower left', fontsize=9, frameon=True,
              facecolor='white', edgecolor='gray', framealpha=0.9)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _format_geo_axes(ax)

    plt.tight_layout()
    out_path = Path(outdir) / 'presentation_landuse_glaciers.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_catchment_glaciers(topo_dir, catchment_shp, glacier_shp, outdir):
    """Glacier coverage on DEM with area statistics."""
    _setup_presentation_style()
    dem, extent = _load_dem(topo_dir)

    catchment = gpd.read_file(catchment_shp)
    if catchment.crs and catchment.crs.to_epsg() != 4326:
        catchment = catchment.to_crs('EPSG:4326')
    xlim, ylim = _catchment_xlim_ylim(catchment)

    glaciers = gpd.read_file(glacier_shp)
    if glaciers.crs != catchment.crs:
        glaciers = glaciers.to_crs(catchment.crs)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(dem, cmap='terrain', extent=extent, interpolation='bilinear',
                   vmin=np.nanmin(dem), vmax=np.nanmax(dem))
    glaciers.plot(ax=ax, facecolor='#87CEEB', edgecolor='darkblue', linewidth=0.8, alpha=0.5)
    catchment.boundary.plot(ax=ax, color='black', linewidth=2)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax).set_label('Elevation (m a.s.l.)')

    glaciers_utm = glaciers.to_crs(glaciers.estimate_utm_crs())
    catchment_utm = catchment.to_crs(glaciers_utm.crs)
    glacier_area = glaciers_utm.geometry.area.sum() / 1e6
    catchment_area = catchment_utm.geometry.area.sum() / 1e6
    glacier_pct = glacier_area / catchment_area * 100

    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2, label='Catchment boundary'),
        mpatches.Patch(facecolor='#87CEEB', edgecolor='darkblue', alpha=0.5,
                       label=f'RGI 6.0 ({glacier_area:.0f} km\u00b2, {glacier_pct:.0f}%)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=11, frameon=True,
              facecolor='white', edgecolor='gray', framealpha=0.9)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _format_geo_axes(ax)

    plt.tight_layout()
    out_path = Path(outdir) / 'presentation_glaciers.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_catchment_overview_2panel(topo_dir, catchment_shp, glacier_shp, hru_shp, outdir,
                                   basin_name='Chandra Basin'):
    """Combined 2-panel overview: (a) DEM, (b) Landuse + RGI6 glaciers.

    Uses HRU shapefile Landuse_Cl for area statistics.
    """
    _setup_presentation_style()
    dem, dem_extent = _load_dem(topo_dir)
    lu, lu_extent = _load_landuse(topo_dir)

    catchment = gpd.read_file(catchment_shp)
    if catchment.crs and catchment.crs.to_epsg() != 4326:
        catchment = catchment.to_crs('EPSG:4326')
    xlim, ylim = _catchment_xlim_ylim(catchment)

    glaciers = gpd.read_file(glacier_shp)
    if glaciers.crs != catchment.crs:
        glaciers = glaciers.to_crs(catchment.crs)

    # Area stats from HRU shapefile
    hru = gpd.read_file(hru_shp)
    hru_utm = hru.to_crs(hru.estimate_utm_crs())
    hru_utm['area_km2'] = hru_utm.geometry.area / 1e6
    area_by_class = hru_utm.groupby('Landuse_Cl')['area_km2'].sum()

    glaciers_utm = glaciers.to_crs(hru_utm.crs)
    catchment_utm = catchment.to_crs(hru_utm.crs)
    glacier_area = glaciers_utm.geometry.area.sum() / 1e6
    catchment_area = catchment_utm.geometry.area.sum() / 1e6
    glacier_pct = glacier_area / catchment_area * 100

    # Landuse colormap
    all_classes = sorted(LU_COLORS.keys())
    cmap = ListedColormap([LU_COLORS[c] for c in all_classes])
    bounds = [c - 0.5 for c in all_classes] + [all_classes[-1] + 0.5]
    norm = BoundaryNorm(bounds, cmap.N)

    glacier_color = '#87CEEB'
    non_glacier_classes = sorted(c for c in LU_COLORS.keys() if c != 7)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # (a) DEM
    ax = axes[0]
    im1 = ax.imshow(dem, cmap='terrain', extent=dem_extent, interpolation='bilinear',
                    vmin=np.nanmin(dem), vmax=np.nanmax(dem))
    catchment.boundary.plot(ax=ax, color='black', linewidth=2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4%", pad=0.3)
    plt.colorbar(im1, cax=cax, orientation='horizontal').set_label('Elevation (m a.s.l.)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _format_geo_axes(ax)

    # (b) Landuse + Glaciers
    ax = axes[1]
    ax.imshow(lu, cmap=cmap, norm=norm, extent=lu_extent, interpolation='nearest')
    glaciers.plot(ax=ax, facecolor=glacier_color, edgecolor='darkblue', linewidth=0.6, alpha=0.7)
    catchment.boundary.plot(ax=ax, color='black', linewidth=2)

    patches = []
    for cl in non_glacier_classes:
        if cl in area_by_class.index:
            area = area_by_class[cl]
            pct = area / catchment_area * 100
            if pct > 0.01:
                patches.append(mpatches.Patch(color=LU_COLORS[cl],
                               label=f'{LU_NAMES[cl]} ({area:.0f} km\u00b2, {pct:.1f}%)'))
    patches.append(mpatches.Patch(facecolor=glacier_color, edgecolor='darkblue',
                   label=f'Glacier RGI 6.0 ({glacier_area:.0f} km\u00b2, {glacier_pct:.1f}%)'))

    ax.legend(handles=patches, loc='lower left', fontsize=8, frameon=True,
              facecolor='white', edgecolor='gray', framealpha=0.9)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _format_geo_axes(ax)

    plt.tight_layout()
    out_path = Path(outdir) / 'presentation_overview_2panel.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ============================================================
# AWS vs reanalysis comparison figures
# ============================================================

def load_aws_temperature(aws_excel_path):
    """Load AWS daily temperature from Campbell logger Excel file."""
    aws = pd.read_excel(aws_excel_path, sheet_name='Sheet1', header=1, skiprows=[2, 3])
    aws['date'] = pd.to_datetime(aws['TIMESTAMP']).dt.normalize()
    aws = aws.set_index('date')
    aws = aws[~aws.index.duplicated(keep='first')]
    return aws[['AirTC_Max', 'AirTC_Min', 'AirTC_Avg']].rename(
        columns={'AirTC_Max': 'tmax', 'AirTC_Min': 'tmin', 'AirTC_Avg': 'tmean'})


def load_aws_precipitation(aws_csv_path):
    """Load AWS daily precipitation from Campbell logger CSV (AccuTotalPrecip diff)."""
    aws = pd.read_csv(aws_csv_path, skiprows=[0, 2, 3], parse_dates=['TIMESTAMP'], dayfirst=True)
    aws = aws.sort_values('TIMESTAMP').reset_index(drop=True)

    accu = aws['AccuTotalPrecip'].values.astype(float)
    hourly = np.zeros(len(accu))
    for i in range(1, len(accu)):
        diff = accu[i] - accu[i - 1]
        hourly[i] = diff if diff >= 0 else accu[i]

    aws['precip'] = hourly
    aws['date'] = aws['TIMESTAMP'].dt.normalize()
    daily = aws.groupby('date')['precip'].sum()
    return daily[~daily.index.duplicated(keep='first')]


def load_era5_temperature(era5_dir, lat, lon, years):
    """Load ERA5-Land daily Tmax/Tmin/Tmean at a point."""
    era5_dir = Path(era5_dir)
    files = []
    for y in years:
        files.extend(sorted(era5_dir.glob(f'era5_land_2m_temperature_{y}_*.nc')))

    ds = xr.open_mfdataset(files, combine='by_coords')
    point = ds['t2m'].sel(latitude=lat, longitude=lon, method='nearest')

    daily = point.resample(valid_time='1D')
    tmax = (daily.max().to_series() - 273.15).rename('tmax')
    tmin = (daily.min().to_series() - 273.15).rename('tmin')
    tmean = (daily.mean().to_series() - 273.15).rename('tmean')
    ds.close()

    for s in [tmax, tmin, tmean]:
        s.index = s.index.normalize()

    result = pd.DataFrame({'tmax': tmax, 'tmin': tmin, 'tmean': tmean})
    return result[~result.index.duplicated(keep='first')]


def load_era5_precipitation(era5_dir, lat, lon, years):
    """Load ERA5-Land daily precipitation at a point (handles cumulative tp)."""
    era5_dir = Path(era5_dir)
    files = []
    for y in years:
        files.extend(sorted(era5_dir.glob(f'era5_land_total_precipitation_{y}_*.nc')))

    ds = xr.open_mfdataset(files, combine='by_coords')
    point = ds['tp'].sel(latitude=lat, longitude=lon, method='nearest')

    hourly = point.to_series() * 1000  # m -> mm
    diff = hourly.diff()
    diff[diff < 0] = hourly[diff < 0]
    diff.iloc[0] = hourly.iloc[0]
    diff = diff.clip(lower=0)
    ds.close()

    daily = diff.resample('D').sum()
    daily.index = daily.index.normalize()
    return daily[~daily.index.duplicated(keep='first')]


def load_har_temperature(har_dir, lat, lon, years):
    """Load HAR v2 daily Tmax/Tmin at a point."""
    har_dir = Path(har_dir)

    # Find grid index
    ds_tmp = xr.open_dataset(sorted(har_dir.glob('HARv2_d10km_d_2d_t2_max_*.nc'))[0])
    lat2d, lon2d = ds_tmp['lat'].values, ds_tmp['lon'].values
    dist = (lat2d - lat) ** 2 + (lon2d - lon) ** 2
    idx = np.unravel_index(dist.argmin(), dist.shape)
    ds_tmp.close()

    data = {}
    for var in ['t2_max', 't2_min']:
        parts = []
        for y in years:
            f = har_dir / f'HARv2_d10km_d_2d_{var}_{y}.nc'
            if f.exists():
                ds = xr.open_dataset(f)
                ts = ds[var][:, idx[0], idx[1]].to_series()
                if ts.mean() > 200:
                    ts = ts - 273.15
                parts.append(ts)
                ds.close()
        combined = pd.concat(parts)
        combined.index = combined.index.normalize()
        data[var] = combined[~combined.index.duplicated(keep='first')]

    result = pd.DataFrame({
        'tmax': data['t2_max'],
        'tmin': data['t2_min'],
        'tmean': (data['t2_max'] + data['t2_min']) / 2,
    })
    return result[~result.index.duplicated(keep='first')]


def load_har_precipitation(har_dir, lat, lon, years):
    """Load HAR v2 daily precipitation at a point (units: mm/h mean -> *24)."""
    har_dir = Path(har_dir)

    ds_tmp = xr.open_dataset(sorted(har_dir.glob('HARv2_d10km_d_2d_prcp_*.nc'))[0])
    lat2d, lon2d = ds_tmp['lat'].values, ds_tmp['lon'].values
    dist = (lat2d - lat) ** 2 + (lon2d - lon) ** 2
    idx = np.unravel_index(dist.argmin(), dist.shape)
    ds_tmp.close()

    parts = []
    for y in years:
        f = har_dir / f'HARv2_d10km_d_2d_prcp_{y}.nc'
        if f.exists():
            ds = xr.open_dataset(f)
            ts = ds['prcp'][:, idx[0], idx[1]].to_series() * 24  # mm/h -> mm/day
            parts.append(ts)
            ds.close()

    daily = pd.concat(parts).clip(lower=0)
    daily.index = daily.index.normalize()
    return daily[~daily.index.duplicated(keep='first')]


def get_era5_elevation(era5_dir, lat, lon):
    """Get ERA5-Land grid cell elevation from geopotential file."""
    geo_file = Path(era5_dir) / 'era5_land_geopotential.nc'
    if not geo_file.exists():
        return None
    ds = xr.open_dataset(geo_file)
    var = list(ds.data_vars)[0]
    point = ds[var].sel(latitude=lat, longitude=lon, method='nearest')
    elev = float(point.values.flatten()[0]) / 9.80665
    ds.close()
    return elev


def plot_temperature_comparison(aws_temp, era5_temp, har_temp, outdir,
                                stn_name='AWS', stn_elev=None, era5_elev=None,
                                lapse_rate=0.0065):
    """Time series, scatter, and regime plots comparing AWS vs ERA5/HAR temperature."""
    _setup_presentation_style()
    outdir = Path(outdir)

    # Apply lapse rate correction
    era5_corr = lapse_rate * (era5_elev - stn_elev) if era5_elev and stn_elev else 0
    era5_corrected = era5_temp + era5_corr
    har_corrected = har_temp + era5_corr  # approximate

    common_idx = aws_temp.index
    era5_c = era5_corrected.reindex(common_idx)
    era5_r = era5_temp.reindex(common_idx)
    har_c = har_corrected.reindex(common_idx)
    har_r = har_temp.reindex(common_idx)

    # --- Time series ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    for ax, var, title in zip(axes, ['tmax', 'tmin', 'tmean'],
                              ['Daily Maximum Temperature', 'Daily Minimum Temperature', 'Daily Mean Temperature']):
        ax.plot(common_idx, aws_temp[var], 'k-', alpha=0.8, lw=1, label=stn_name)
        ax.plot(common_idx, era5_r[var], color='lightblue', alpha=0.5, lw=0.7, label='ERA5-Land raw')
        ax.plot(common_idx, era5_c[var], 'b-', alpha=0.7, lw=0.8, label=f'ERA5-Land corr. (+{era5_corr:.1f}°C)')
        ax.plot(common_idx, har_r[var], color='lightsalmon', alpha=0.5, lw=0.7, label='HAR v2 raw')
        ax.plot(common_idx, har_c[var], 'r-', alpha=0.7, lw=0.8, label=f'HAR v2 corr. (+{era5_corr:.1f}°C)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f'{stn_name} (~{stn_elev}m) vs ERA5-Land ({era5_elev:.0f}m) & HAR v2',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(outdir / 'aws_vs_era5_har_timeseries.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {outdir / 'aws_vs_era5_har_timeseries.png'}")
    plt.close(fig)

    # --- Scatter ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for col, (var, title) in enumerate(zip(['tmax', 'tmin', 'tmean'], ['Tmax', 'Tmin', 'Tmean'])):
        for row, (sim, name, color) in enumerate([
            (era5_c, 'ERA5-Land corr.', 'blue'), (har_c, 'HAR v2 corr.', 'red')
        ]):
            ax = axes[row, col]
            mask = aws_temp[var].notna() & sim[var].notna()
            o, s = aws_temp[var][mask].values.astype(float), sim[var][mask].values.astype(float)
            ax.scatter(o, s, s=3, alpha=0.3, color=color)
            lims = [min(o.min(), s.min()) - 2, max(o.max(), s.max()) + 2]
            ax.plot(lims, lims, 'k--', lw=0.8)
            bias = (s - o).mean()
            corr = np.corrcoef(o, s)[0, 1]
            ax.set_title(f'{name} {title}\nBias={bias:+.1f}°C, R={corr:.3f}')
            ax.set_xlabel(f'{stn_name} (°C)')
            ax.set_ylabel(f'{name} (°C)')
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_aspect('equal')
            ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(outdir / 'aws_vs_era5_har_scatter.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {outdir / 'aws_vs_era5_har_scatter.png'}")
    plt.close(fig)

    # --- Monthly regime ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, var, title in zip(axes, ['tmax', 'tmin', 'tmean'], ['Tmax', 'Tmin', 'Tmean']):
        for data, name, style in [
            (aws_temp, stn_name, 'ko-'),
            (era5_c, 'ERA5-Land corr.', 'bs-'),
            (har_c, 'HAR v2 corr.', 'r^-'),
        ]:
            monthly = data[var].groupby(data.index.month).mean()
            ax.plot(monthly.index, monthly.values, style, lw=1.5 if 'AWS' not in name else 2,
                    ms=6 if 'AWS' not in name else 8, label=name)
        ax.set_xlabel('Month')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xticks(range(1, 13))

    plt.tight_layout()
    fig.savefig(outdir / 'aws_vs_era5_har_regime.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {outdir / 'aws_vs_era5_har_regime.png'}")
    plt.close(fig)


def plot_precipitation_comparison(aws_precip, era5_precip, har_precip, outdir,
                                  stn_name='AWS'):
    """Daily, monthly, regime, scatter, and cumulative precipitation comparison."""
    _setup_presentation_style()
    outdir = Path(outdir)

    common_idx = aws_precip.index
    era5_p = era5_precip.reindex(common_idx)
    har_p = har_precip.reindex(common_idx)

    # --- Daily ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for ax, (data, name, color) in zip(axes, [
        (aws_precip, stn_name, 'black'),
        (era5_p, 'ERA5-Land', 'blue'),
        (har_p, 'HAR v2', 'red'),
    ]):
        ax.bar(common_idx, data.fillna(0), color=color, alpha=0.6, width=1, label=name)
        ann = data.mean() * 365
        ax.set_ylabel('Precip (mm/day)')
        ax.set_title(f'{name} — mean {data.mean():.1f} mm/day (~{ann:.0f} mm/yr)')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
    ymax = max(aws_precip.max(), era5_p.max(), har_p.max()) * 1.1
    for ax in axes:
        ax.set_ylim(0, ymax)
    plt.tight_layout()
    fig.savefig(outdir / 'aws_vs_era5_har_precip_daily.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {outdir / 'aws_vs_era5_har_precip_daily.png'}")
    plt.close(fig)

    # --- Monthly totals ---
    fig, ax = plt.subplots(figsize=(14, 6))
    width = pd.Timedelta(days=8)
    aws_m = aws_precip.resample('MS').sum()
    era5_m = era5_p.resample('MS').sum()
    har_m = har_p.resample('MS').sum()
    ax.bar(aws_m.index - width, aws_m.values, width=width, color='black', alpha=0.7, label=stn_name)
    ax.bar(era5_m.index, era5_m.values, width=width, color='blue', alpha=0.5, label='ERA5-Land')
    ax.bar(har_m.index + width, har_m.values, width=width, color='red', alpha=0.5, label='HAR v2')
    ax.set_ylabel('Precipitation (mm/month)')
    ax.set_title('Monthly Precipitation Totals', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / 'aws_vs_era5_har_precip_monthly.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {outdir / 'aws_vs_era5_har_precip_monthly.png'}")
    plt.close(fig)

    # --- Monthly regime ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for data, name, style in [
        (aws_precip, stn_name, 'ko-'), (era5_p, 'ERA5-Land', 'bs-'), (har_p, 'HAR v2', 'r^-')
    ]:
        regime = data.groupby(data.index.month).mean() * 30
        ax.plot(regime.index, regime.values, style, lw=1.5 if stn_name not in name else 2,
                ms=6 if stn_name not in name else 8, label=name)
    ax.set_xlabel('Month')
    ax.set_ylabel('Precipitation (mm/month)')
    ax.set_title('Monthly Precipitation Regime', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    plt.tight_layout()
    fig.savefig(outdir / 'aws_vs_era5_har_precip_regime.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {outdir / 'aws_vs_era5_har_precip_regime.png'}")
    plt.close(fig)

    # --- Scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (sim, name, color) in zip(axes, [
        (era5_p, 'ERA5-Land', 'blue'), (har_p, 'HAR v2', 'red')
    ]):
        mask = aws_precip.notna() & sim.notna()
        o, s = aws_precip[mask].values.astype(float), sim[mask].values.astype(float)
        ax.scatter(o, s, s=5, alpha=0.3, color=color)
        lim = max(o.max(), s.max()) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', lw=0.8)
        bias = (s - o).mean()
        corr = np.corrcoef(o, s)[0, 1]
        ratio = s.sum() / o.sum()
        ax.set_title(f'{name}\nBias={bias:+.1f}mm/d, R={corr:.3f}, P_ratio={ratio:.2f}')
        ax.set_xlabel(f'{stn_name} (mm/day)')
        ax.set_ylabel(f'{name} (mm/day)')
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / 'aws_vs_era5_har_precip_scatter.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {outdir / 'aws_vs_era5_har_precip_scatter.png'}")
    plt.close(fig)

    # --- Cumulative ---
    fig, ax = plt.subplots(figsize=(14, 6))
    aws_c = aws_precip.fillna(0).cumsum()
    era5_c = era5_p.fillna(0).cumsum()
    har_c = har_p.fillna(0).cumsum()
    ax.plot(common_idx, aws_c, 'k-', lw=2, label=f'{stn_name} ({aws_c.iloc[-1]:.0f} mm)')
    ax.plot(common_idx, era5_c, 'b-', lw=1.5, label=f'ERA5-Land ({era5_c.iloc[-1]:.0f} mm)')
    ax.plot(common_idx, har_c, 'r-', lw=1.5, label=f'HAR v2 ({har_c.iloc[-1]:.0f} mm)')
    ax.set_ylabel('Cumulative Precipitation (mm)')
    ax.set_title('Cumulative Precipitation Comparison', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / 'aws_vs_era5_har_precip_cumulative.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {outdir / 'aws_vs_era5_har_precip_cumulative.png'}")
    plt.close(fig)


def plot_meteo_validation_slide(aws_temp, aws_precip, era5_temp, era5_precip,
                                har_temp, har_precip, outdir,
                                stn_name='AWS Himansh', stn_elev=None, era5_elev=None,
                                lapse_rate=0.0065):
    """2x2 presentation slide: T regime, T scatter (top), monthly P bars, cumulative P (bottom).

    Uses neutral colors: black (AWS), orange (ERA5-Land), green (HAR v2).
    """
    _setup_presentation_style()
    outdir = Path(outdir)

    # Colors
    c_aws, c_era5, c_har = 'black', '#e67e22', '#27ae60'

    # Lapse rate correction
    era5_corr = lapse_rate * (era5_elev - stn_elev) if era5_elev and stn_elev else 0
    era5_t = (era5_temp + era5_corr).reindex(aws_temp.index)
    har_t = (har_temp + era5_corr).reindex(aws_temp.index)

    common_p = aws_precip.index
    era5_p = era5_precip.reindex(common_p)
    har_p = har_precip.reindex(common_p)

    # Temperature stats
    def _temp_stats(obs, sim, var='tmean'):
        mask = obs[var].notna() & sim[var].notna()
        o, s = obs[var][mask].values.astype(float), sim[var][mask].values.astype(float)
        return np.corrcoef(o, s)[0, 1], (s - o).mean()

    era5_t_r, era5_t_bias = _temp_stats(aws_temp, era5_t)
    har_t_r, har_t_bias = _temp_stats(aws_temp, har_t)

    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    FS_LABEL = 20
    FS_TICK = 18
    FS_LEGEND = 18
    FS_TITLE = 22

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.3)
    ax_tregime = fig.add_subplot(gs[0, 0])
    ax_tscatter = fig.add_subplot(gs[0, 1])
    ax_precip = fig.add_subplot(gs[1, :])

    # --- (a) Temperature regime ---
    ax = ax_tregime
    for data, name, color, marker in [
        (aws_temp, stn_name, c_aws, 'o'),
        (era5_t, 'ERA5-Land', c_era5, 's'),
        (har_t, 'HAR v2', c_har, '^'),
    ]:
        regime = data['tmean'].groupby(data.index.month).mean()
        ax.plot(regime.index, regime.values, color=color, marker=marker,
                lw=3, ms=9, label=name)
    ax.set_ylabel('Temperature (\u00b0C)', fontsize=FS_LABEL, fontweight='bold')
    ax.legend(fontsize=FS_LEGEND, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, fontsize=FS_TICK, fontweight='bold')
    ax.tick_params(axis='y', labelsize=FS_TICK)

    # --- (b) Temperature scatter ---
    ax = ax_tscatter
    for sim, name, color, marker in [
        (era5_t, 'ERA5-Land', c_era5, 's'),
        (har_t, 'HAR v2', c_har, '^'),
    ]:
        mask = aws_temp['tmean'].notna() & sim['tmean'].notna()
        o = aws_temp['tmean'][mask].values.astype(float)
        s = sim['tmean'][mask].values.astype(float)
        ax.scatter(o, s, s=10, alpha=0.4, color=color, marker=marker, label=name)
    lims = [min(aws_temp['tmean'].min(), era5_t['tmean'].min(), har_t['tmean'].min()) - 2,
            max(aws_temp['tmean'].max(), era5_t['tmean'].max(), har_t['tmean'].max()) + 2]
    ax.plot(lims, lims, 'k--', lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f'{stn_name} (\u00b0C)', fontsize=FS_LABEL, fontweight='bold')
    ax.set_ylabel('Reanalysis (\u00b0C)', fontsize=FS_LABEL, fontweight='bold')
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=FS_TICK)

    # --- (c) Mean daily precipitation per month (full width) ---
    ax = ax_precip
    aws_m = aws_precip.resample('MS').mean()
    era5_m = era5_p.resample('MS').mean()
    har_m = har_p.resample('MS').mean()
    width = pd.Timedelta(days=8)
    ax.bar(aws_m.index - width, aws_m.values, width=width, color=c_aws, alpha=0.7, label=stn_name)
    ax.bar(era5_m.index, era5_m.values, width=width, color=c_era5, alpha=0.7, label='ERA5-Land')
    ax.bar(har_m.index + width, har_m.values, width=width, color=c_har, alpha=0.7, label='HAR v2')
    ax.set_ylabel('Precipitation (mm/day)', fontsize=FS_LABEL, fontweight='bold')
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=FS_TICK)

    out_path = outdir / 'presentation_meteo_validation.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ============================================================
# Glacier melt spatial plots
# ============================================================

def plot_glacier_melt_spatial(irrigation_nc, hru_shp, catchment_shp, outdir,
                               date='2017-07-15', vmax=None):
    """Plot glacier melt (irrigation.nc) mapped onto HRU polygons for a single day.

    Non-glaciated HRUs (melt=0) shown in light gray. Glaciated HRUs use a
    blue colorscale starting from a low threshold to show only the melt range.

    Args:
        irrigation_nc: Path to irrigation.nc (GloGEM melt per HRU).
        hru_shp: Path to HRU shapefile with HRU_ID attribute.
        catchment_shp: Path to catchment boundary shapefile.
        outdir: Output directory.
        date: Date string to plot (e.g. '2017-07-15').
        vmax: Max value for colorbar. If None, auto from data.
    """
    _setup_presentation_style()
    outdir = Path(outdir)

    # Load irrigation data for the selected date
    ds = xr.open_dataset(irrigation_nc)
    melt = ds['data'].sel(time=date).squeeze()
    hru_ids = ds['x'].values
    melt_values = melt.values
    ds.close()

    # Load HRU shapefile and join melt values
    hru = gpd.read_file(hru_shp)
    melt_df = pd.DataFrame({'HRU_ID': hru_ids, 'melt_mm': melt_values})
    hru = hru.merge(melt_df, on='HRU_ID', how='left')
    hru['melt_mm'] = hru['melt_mm'].fillna(0)

    # Reproject to WGS84
    if hru.crs and hru.crs.to_epsg() != 4326:
        hru = hru.to_crs('EPSG:4326')

    catchment = gpd.read_file(catchment_shp)
    if catchment.crs and catchment.crs.to_epsg() != 4326:
        catchment = catchment.to_crs('EPSG:4326')
    xlim, ylim = _catchment_xlim_ylim(catchment)

    # Split into glaciated and non-glaciated
    hru_no_melt = hru[hru['melt_mm'] <= 0]
    hru_melt = hru[hru['melt_mm'] > 0]

    nonzero = hru_melt['melt_mm'].values
    vmin_melt = np.floor(nonzero.min()) if len(nonzero) > 0 else 0
    if vmax is None:
        vmax = np.ceil(nonzero.max()) if len(nonzero) > 0 else 1

    fig, ax = plt.subplots(figsize=(10, 10))

    # Non-glaciated HRUs in light gray
    hru_no_melt.plot(ax=ax, facecolor='#d9d9d9', edgecolor='gray', linewidth=0.3)

    # Glaciated HRUs with blue scale
    if len(hru_melt) > 0:
        hru_melt.plot(column='melt_mm', ax=ax, cmap='Blues', vmin=vmin_melt, vmax=vmax,
                      edgecolor='darkblue', linewidth=0.5, legend=False)

    catchment.boundary.plot(ax=ax, color='black', linewidth=2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin_melt, vmax=vmax))
    sm._A = []
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Glacier melt (mm/day)', fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _format_geo_axes(ax)

    date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
    plt.tight_layout()
    out_path = outdir / f'presentation_glacier_melt_{date_str}.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ============================================================
# Configuration comparison plots
# ============================================================

def _get_coupled_config_colors(configurations_yaml):
    """Build consistent batlow color mapping for all coupled configs.

    Returns dict mapping config key -> batlow color, in registry order.
    """
    import cmcrameri.cm as cmc
    with open(configurations_yaml) as f:
        cfg = yaml.safe_load(f)
    coupled = [c for c in cfg['configurations'] if c.get('coupled', False)]
    n = len(coupled)
    batlow = cmc.batlow
    return {c['key']: batlow(i / (n - 1)) if n > 1 else batlow(0.5)
            for i, c in enumerate(coupled)}


def _get_config_output_dir(main_dir, key, gauge_id, basin='chandra_1'):
    """Get the HBV output directory for a given config key."""
    return Path(main_dir) / f'{basin}/03_model_setups_{key}/catchment_{gauge_id}/HBV/output'


def plot_config_metrics_barchart(configurations_yaml, gauge_id, outdir,
                                 metrics=None):
    """Bar chart of KGE (and optionally NSE/PBIAS) for coupled configs, calibration vs validation.

    Args:
        configurations_yaml: Path to configurations.yaml registry.
        gauge_id: Gauge ID string (e.g. '0101').
        outdir: Output directory.
        metrics: List of metrics to plot. Default: ['KGE'].
    """
    _setup_presentation_style()
    outdir = Path(outdir)
    if metrics is None:
        metrics = ['KGE']

    with open(configurations_yaml) as f:
        cfg = yaml.safe_load(f)

    main_dir = Path(cfg['main_dir'])
    basin = cfg.get('basin', 'chandra_1')
    coupled_configs = [c for c in cfg['configurations'] if c.get('coupled', False)]

    # Collect metrics for each config
    rows = []
    for c in coupled_configs:
        key = c['key']
        setup_dir = main_dir / f'{basin}/03_model_setups_{key}/catchment_{gauge_id}/HBV/output'
        metrics_files = list(setup_dir.glob('*coupled_metrics.csv')) + \
                        list(setup_dir.glob('*BEST_RUN_metrics.csv'))
        if not metrics_files:
            continue
        df = pd.read_csv(metrics_files[0])
        cali = df[df['Unnamed: 0'] == 'CALIBRATION'].iloc[0]
        vali = df[df['Unnamed: 0'] == 'VALIDATION'].iloc[0]
        row = {'key': key, 'display_name': c['display_name']}
        for m in metrics:
            row[f'{m}_cali'] = cali[m]
            row[f'{m}_vali'] = vali[m]
        rows.append(row)

    if not rows:
        print("No metrics files found.")
        return

    data = pd.DataFrame(rows)
    n_metrics = len(metrics)
    n_configs = len(data)

    # Use consistent batlow color mapping
    color_map = _get_coupled_config_colors(configurations_yaml)
    bar_colors = [color_map[k] for k in data['key']]

    fig, axes = plt.subplots(1, n_metrics, figsize=(max(12, n_configs * 1.2), 6),
                              squeeze=False)

    for col, m in enumerate(metrics):
        ax = axes[0, col]
        x = np.arange(n_configs)
        bar_width = 0.35

        bars_c = ax.bar(x - bar_width / 2, data[f'{m}_cali'], bar_width,
                        color=bar_colors, alpha=0.9, edgecolor='black',
                        linewidth=0.5, label='Calibration')
        bars_v = ax.bar(x + bar_width / 2, data[f'{m}_vali'], bar_width,
                        color=bar_colors, alpha=0.5, edgecolor='black',
                        linewidth=0.5, label='Validation', hatch='///')

        # Value labels on bars
        for bar in list(bars_c) + list(bars_v):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(data['display_name'], rotation=45, ha='right',
                           fontweight='bold', fontsize=9)
        ax.set_ylabel(m, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, min(1.0, data[[f'{m}_cali', f'{m}_vali']].max().max() + 0.15))
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

    plt.tight_layout()
    metric_str = '_'.join(m.lower() for m in metrics)
    out_path = outdir / f'presentation_config_metrics_{metric_str}.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


def _load_hydrograph(output_dir, gauge_id):
    """Load Raven hydrograph CSV, return DataFrame with date index, sim/obs columns."""
    f = output_dir / f'{gauge_id}_HBV_Hydrographs.csv'
    df = pd.read_csv(f, skiprows=0)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    sim_col = f'{gauge_id} [m3/s]'
    obs_col = f'{gauge_id} (observed) [m3/s]'
    result = pd.DataFrame()
    result['sim'] = pd.to_numeric(df[sim_col], errors='coerce')
    result['obs'] = pd.to_numeric(df[obs_col], errors='coerce')
    return result


def _load_mass_loadings(output_dir, gauge_id, component):
    """Load a Raven MassLoadings CSV (SNOWMELT or GLACIERMELT_ALL).

    Returns Series in m3/s with date index.
    """
    f = output_dir / f'{gauge_id}_HBV_{component}MassLoadings.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    col = f'{gauge_id} m3/s'
    return pd.to_numeric(df[col], errors='coerce')


def plot_hydrograph_comparison(configurations_yaml, gauge_id, outdir,
                                config_keys=None, start_date=None, end_date=None):
    """Overlay hydrographs for selected configs with observed Q.

    Args:
        configurations_yaml: Path to configurations.yaml.
        gauge_id: Gauge ID string.
        outdir: Output directory.
        config_keys: List of config keys to plot. If None, plots all with data.
        start_date: Optional start date for plot window.
        end_date: Optional end date for plot window.
    """
    _setup_presentation_style()
    outdir = Path(outdir)

    with open(configurations_yaml) as f:
        cfg = yaml.safe_load(f)
    main_dir = Path(cfg['main_dir'])
    basin = cfg.get('basin', 'chandra_1')

    color_map = _get_coupled_config_colors(configurations_yaml)
    all_configs = {c['key']: c for c in cfg['configurations'] if c.get('coupled', False)}

    if config_keys is None:
        config_keys = list(all_configs.keys())

    fig, ax = plt.subplots(figsize=(16, 6))
    obs_plotted = False

    for key in config_keys:
        if key not in all_configs:
            continue
        out_dir = _get_config_output_dir(main_dir, key, gauge_id, basin)
        try:
            hydro = _load_hydrograph(out_dir, gauge_id)
        except Exception:
            continue

        if start_date:
            hydro = hydro[hydro.index >= start_date]
        if end_date:
            hydro = hydro[hydro.index <= end_date]

        if not obs_plotted:
            ax.plot(hydro.index, hydro['obs'], 'k-', lw=1.5, alpha=0.8, label='Observed')
            obs_plotted = True

        ax.plot(hydro.index, hydro['sim'], color=color_map.get(key, 'gray'),
                lw=1, alpha=0.8, label=all_configs[key]['display_name'])

    ax.set_ylabel('Discharge (m\u00b3/s)', fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.grid(alpha=0.3)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.tight_layout()
    out_path = outdir / 'presentation_hydrograph_comparison.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_hydrograph_single(configurations_yaml, gauge_id, outdir,
                           config_key='glogem_subdaily', start_date=None, end_date=None):
    """Presentation-ready hydrograph for a single configuration.

    Args:
        configurations_yaml: Path to configurations.yaml.
        gauge_id: Gauge ID string.
        outdir: Output directory.
        config_key: Config key to plot.
        start_date: Optional start date (defaults to cali_end_date / validation).
        end_date: Optional end date (defaults to end_date).
    """
    _setup_presentation_style()
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 17,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    })
    outdir = Path(outdir)

    with open(configurations_yaml) as f:
        cfg = yaml.safe_load(f)
    main_dir = Path(cfg['main_dir'])
    basin = cfg.get('basin', 'chandra_1')

    color_map = _get_coupled_config_colors(configurations_yaml)
    all_configs = {c['key']: c for c in cfg['configurations']}
    cfg_entry = all_configs.get(config_key)
    if cfg_entry is None:
        print(f"Config key '{config_key}' not found in registry.")
        return

    # Resolve validation period from namelist
    namelists_dir_override = cfg.get('namelists_dir')
    if namelists_dir_override:
        nl_dir = Path(namelists_dir_override)
        if not nl_dir.is_absolute():
            nl_dir = Path(configurations_yaml).parent.parent.parent / nl_dir
    else:
        nl_dir = Path(configurations_yaml).parent.parent.parent / 'namelists_server'

    nl_name = f'namelist_{gauge_id}_{config_key}.yaml'
    nl_path = nl_dir / nl_name
    if nl_path.exists():
        with open(nl_path) as f:
            nml = yaml.safe_load(f)
        if start_date is None:
            start_date = nml.get('cali_end_date', nml.get('start_date'))
        if end_date is None:
            end_date = nml.get('end_date')

    # Load KGE validation
    out_dir = _get_config_output_dir(main_dir, config_key, gauge_id, basin)
    kge_vali = None
    for pattern in ['*BEST_RUN_metrics.csv', '*coupled_metrics.csv']:
        mfiles = list(out_dir.glob(pattern))
        if mfiles:
            try:
                mdf = pd.read_csv(mfiles[0], index_col=0)
                kge_vali = float(mdf.loc['VALIDATION', 'KGE'])
            except Exception:
                pass
            break

    # Load hydrograph
    hydro = _load_hydrograph(out_dir, gauge_id)
    if start_date:
        hydro = hydro[hydro.index >= start_date]
    if end_date:
        hydro = hydro[hydro.index <= end_date]

    sim_color = color_map.get(config_key, '#e377c2')

    fig, ax = plt.subplots(figsize=(16, 5.5))

    # Observed as filled area
    ax.fill_between(hydro.index, 0, hydro['obs'], color='lightgray', alpha=0.5)
    ax.plot(hydro.index, hydro['obs'], 'k-', lw=2, label='Observed')

    # Simulated as colored dashed line
    ax.plot(hydro.index, hydro['sim'], color=sim_color, lw=2.5, linestyle='--', label=cfg_entry['display_name'])

    ax.set_ylabel('Discharge (m\u00b3/s)', fontweight='bold', fontsize=17)
    ax.set_xlabel('')

    # Add KGE to legend label instead of title
    if kge_vali is not None:
        # Update the sim line label to include KGE
        ax.get_lines()[-1].set_label(f'{cfg_entry["display_name"]}  (KGE = {kge_vali:.3f})')

    ax.legend(fontsize=15, loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim(hydro.index.min(), hydro.index.max())
    ax.set_ylim(bottom=0)

    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_rotation(30)
        label.set_ha('right')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.tight_layout()
    out_path = outdir / f'presentation_hydrograph_{config_key}.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_contribution_regime_subplots(configurations_yaml, gauge_id, outdir,
                                       config_keys=None, start_date=None, end_date=None,
                                       n_best=None):
    """Subplot grid showing monthly contribution regimes for coupled configs.

    Each subplot shows stacked areas: snowmelt, glacier melt, and observed Q overlay.
    For coupled configs, uses GloGEM data (icemelt + snowmelt) matched with HBV snowmelt.
    For icemelt-mode configs, uses HBV snowmelt only (glacier HRUs are ROCK).
    For non-coupled configs, uses Raven MassLoadings.
    Configs are ranked by KGE validation and filtered to top n_best.

    Args:
        configurations_yaml: Path to configurations.yaml.
        gauge_id: Gauge ID string.
        outdir: Output directory.
        config_keys: List of config keys to plot. If None, ranks all coupled by KGE vali.
        start_date: Optional start date to filter data (defaults to cali_end_date).
        end_date: Optional end date to filter data (defaults to end_date).
        n_best: Number of best configs to show. If None, shows all with data.
    """
    from postprocessing import load_glogem_data, load_snowmelt_mass_loadings

    _setup_presentation_style()
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 17,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
    })
    outdir = Path(outdir)

    with open(configurations_yaml) as f:
        cfg = yaml.safe_load(f)
    main_dir = Path(cfg['main_dir'])
    basin = cfg.get('basin', 'chandra_1')

    color_map = _get_coupled_config_colors(configurations_yaml)
    all_configs = {c['key']: c for c in cfg['configurations'] if c.get('coupled', False)}

    # Resolve namelists directory
    namelists_dir_override = cfg.get('namelists_dir')
    if namelists_dir_override:
        nl_dir = Path(namelists_dir_override)
        if not nl_dir.is_absolute():
            nl_dir = Path(configurations_yaml).parent.parent.parent / nl_dir
    else:
        nl_dir = Path(configurations_yaml).parent.parent.parent / 'namelists_server'

    if config_keys is None:
        # Rank coupled configs by KGE validation (descending)
        ranked = []
        for key in all_configs:
            out_dir = _get_config_output_dir(main_dir, key, gauge_id, basin)
            for pattern in ['*BEST_RUN_metrics.csv', '*coupled_metrics.csv']:
                mfiles = list(out_dir.glob(pattern))
                if mfiles:
                    break
            if not mfiles:
                continue
            try:
                mdf = pd.read_csv(mfiles[0], index_col=0)
                kge_vali = float(mdf.loc['VALIDATION', 'KGE'])
                ranked.append((key, kge_vali))
            except Exception:
                continue
        ranked.sort(key=lambda x: x[1], reverse=True)
        if n_best is not None:
            ranked = ranked[:n_best]
        config_keys = [k for k, _ in ranked]
        kge_values = {k: v for k, v in ranked}
        print(f"Configs ranked by KGE validation: {[(k, f'{v:.3f}') for k, v in ranked]}")
    else:
        kge_values = {}

    if not config_keys:
        print("No configs with metrics found.")
        return

    # Use full simulation period if not provided
    if start_date is None or end_date is None:
        first_key = config_keys[0]
        nl_name = f'namelist_{gauge_id}_{first_key}.yaml'
        nl_path = nl_dir / nl_name
        if nl_path.exists():
            with open(nl_path) as f:
                nml = yaml.safe_load(f)
            if start_date is None:
                start_date = nml.get('start_date')
            if end_date is None:
                end_date = nml.get('end_date')
            print(f"Using full period: {start_date} to {end_date}")

    # Get catchment area for m3/s -> mm/day conversion
    hru_shp = main_dir / f'{basin}/03_model_setups_{config_keys[0]}/catchment_{gauge_id}/topo_files/HRU.shp'
    hru = gpd.read_file(hru_shp)
    hru_utm = hru.to_crs(hru.estimate_utm_crs())
    catchment_area_m2 = hru_utm.geometry.area.sum()
    conv = 86400 / catchment_area_m2 * 1000  # m3/s -> mm/day

    n = len(config_keys)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5.5 * nrows), squeeze=False)

    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    val_start = pd.to_datetime(start_date)
    val_end = pd.to_datetime(end_date)

    for idx, key in enumerate(config_keys):
        ax = axes[idx // ncols, idx % ncols]
        cfg_entry = all_configs.get(key)
        if cfg_entry is None:
            ax.set_visible(False)
            continue

        is_coupled = cfg_entry.get('coupled', False)
        is_icemelt = cfg_entry.get('icemelt_mode', False)
        out_dir = _get_config_output_dir(main_dir, key, gauge_id, basin)

        try:
            hydro = _load_hydrograph(out_dir, gauge_id)
        except Exception:
            ax.set_visible(False)
            continue

        # Filter to validation period
        hydro = hydro[(hydro.index >= val_start) & (hydro.index <= val_end)]

        # Build individual_config dict for postprocessing functions
        config_dir_str = f'{basin}/03_model_setups_{key}'
        individual_config = {
            'main_dir': str(main_dir),
            'config_dir': config_dir_str,
            'gauge_id': gauge_id,
            'model_type': 'HBV',
            'start_date': start_date,
            'end_date': end_date,
            'coupled': is_coupled,
            'glogem_dir': None,
        }

        if is_coupled:
            # COUPLED: use GloGEM data for icemelt, and handle snowmelt correctly
            glogem_data = load_glogem_data(individual_config, unit='mm', plot=False)
            if glogem_data is None:
                print(f"  WARNING: No GloGEM data for {key}, falling back to MassLoadings")
                # Fall back to MassLoadings
                snowmelt = _load_mass_loadings(out_dir, gauge_id, 'SNOWMELT')
                glacier = _load_mass_loadings(out_dir, gauge_id, 'GLACIERMELT_ALL')
                if snowmelt is not None:
                    snowmelt = snowmelt[(snowmelt.index >= val_start) & (snowmelt.index <= val_end)]
                if glacier is not None:
                    glacier = glacier[(glacier.index >= val_start) & (glacier.index <= val_end)]
                snow_regime = (snowmelt * conv).groupby(snowmelt.index.month).mean() if snowmelt is not None else pd.Series(0, index=range(1, 13))
                glac_regime = (glacier * conv).groupby(glacier.index.month).mean() if glacier is not None else pd.Series(0, index=range(1, 13))
            else:
                # Filter GloGEM to validation period
                glogem_mask = (glogem_data['date'] >= val_start) & (glogem_data['date'] <= val_end)
                glogem_filtered = glogem_data[glogem_mask].copy()
                glogem_filtered['month'] = glogem_filtered['date'].dt.month

                # GloGEM icemelt (catchment-normalized, already in mm/day)
                glac_regime = glogem_filtered.groupby('month')['icemelt_normalized'].mean()

                # Load HBV snowmelt mass loadings
                hbv_snow_df = load_snowmelt_mass_loadings(individual_config, start_date, end_date, unit='mm')
                if hbv_snow_df is not None:
                    snow_col = 'snowmelt_mm_day' if 'snowmelt_mm_day' in hbv_snow_df.columns else 'snowmelt_m3s'
                    hbv_snow_df['month'] = hbv_snow_df['date'].dt.month
                    hbv_snow_regime = hbv_snow_df.groupby('month')[snow_col].mean()
                else:
                    hbv_snow_regime = pd.Series(0, index=range(1, 13))

                glogem_snow_regime = glogem_filtered.groupby('month')['snowmelt_normalized'].mean()

                if is_icemelt:
                    # ICEMELT: glacier HRUs are ROCK, Raven simulates snowmelt there.
                    # Use HBV snowmelt only (includes glacier areas). Don't add GloGEM snowmelt.
                    snow_regime = hbv_snow_regime
                    print(f"  {key}: icemelt mode — HBV snowmelt only (includes glacier-ROCK HRUs)")
                else:
                    # STANDARD COUPLED: glacier HRUs are MASKED_GLACIER, Raven does NOT
                    # simulate snowmelt there. Add GloGEM snowmelt for glacier areas.
                    snow_regime = glogem_snow_regime.add(hbv_snow_regime, fill_value=0)
                    print(f"  {key}: coupled mode — HBV + GloGEM snowmelt")

        else:
            # NON-COUPLED: use Raven MassLoadings
            snowmelt = _load_mass_loadings(out_dir, gauge_id, 'SNOWMELT')
            glacier = _load_mass_loadings(out_dir, gauge_id, 'GLACIERMELT_ALL')
            if snowmelt is not None:
                snowmelt = snowmelt[(snowmelt.index >= val_start) & (snowmelt.index <= val_end)]
            if glacier is not None:
                glacier = glacier[(glacier.index >= val_start) & (glacier.index <= val_end)]
            snow_regime = (snowmelt * conv).groupby(snowmelt.index.month).mean() if snowmelt is not None else pd.Series(0, index=range(1, 13))
            glac_regime = (glacier * conv).groupby(glacier.index.month).mean() if glacier is not None else pd.Series(0, index=range(1, 13))

        # Compute monthly regimes for streamflow
        # Only use days where observed is valid for both obs and sim
        obs_mm = hydro['obs'] * conv
        sim_mm = hydro['sim'] * conv
        valid_mask = obs_mm.notna() & (obs_mm > 0)
        obs_valid = obs_mm[valid_mask]
        sim_valid = sim_mm[valid_mask]

        # Two-step mean: monthly mean per year, then average across years
        # (avoids bias from unequal monthly coverage across years)
        min_days = 15  # require at least 15 valid days in a year-month
        obs_ym = obs_valid.groupby([obs_valid.index.year, obs_valid.index.month]).agg(['mean', 'count'])
        obs_ym = obs_ym[obs_ym['count'] >= min_days]
        obs_regime = obs_ym['mean'].groupby(level=1).mean()

        sim_ym = sim_valid.groupby([sim_valid.index.year, sim_valid.index.month]).agg(['mean', 'count'])
        sim_ym = sim_ym[sim_ym['count'] >= min_days]
        sim_regime = sim_ym['mean'].groupby(level=1).mean()

        months = list(range(1, 13))
        snow_vals = np.array([snow_regime.get(m, 0) for m in months])
        glac_vals = np.array([glac_regime.get(m, 0) for m in months])

        # Separate filled polygons (both from 0, not stacked)
        ax.fill_between(months, 0, snow_vals, color='#a8d8ea', alpha=0.6, label='Snowmelt')
        ax.fill_between(months, 0, glac_vals, color='#d0d0d0', alpha=0.7, label='Glacier melt')

        # Observed and simulated Q lines
        ax.plot(months, [obs_regime.get(m, 0) for m in months], 'ko-', lw=3, ms=7, label='Observed Q')
        ax.plot(months, [sim_regime.get(m, 0) for m in months],
                color=color_map.get(key, 'gray'), marker='s', lw=3, ms=7,
                linestyle='--', label='Simulated Q')

        # Title with KGE value if available
        title = cfg_entry['display_name']
        if key in kge_values:
            title += f'  (KGE = {kge_values[key]:.3f})'
        ax.set_title(title, fontweight='bold', fontsize=18,
                     color=color_map.get(key, 'black'))
        ax.set_xticks(list(months))
        ax.set_xticklabels(month_labels, fontweight='bold', fontsize=15)
        ax.set_ylabel('mm/day', fontweight='bold', fontsize=17)
        ax.grid(alpha=0.3)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        # Only show legend in the first subplot
        if idx == 0:
            ax.legend(fontsize=15, loc='upper left', framealpha=0.9)

    # Shared y-axis scale across all subplots
    y_max = max(ax.get_ylim()[1] for row in axes for ax in row if ax.get_visible())
    for row in axes:
        for ax in row:
            if ax.get_visible():
                ax.set_ylim(0, y_max)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout()
    out_path = outdir / 'presentation_contribution_regimes.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ============================================================
# Convenience runner
# ============================================================

def run_catchment_overview(topo_dir, catchment_shp, glacier_shp, outdir,
                           hru_shp=None, basin_name='Chandra Basin'):
    """Generate all catchment overview figures.

    Args:
        hru_shp: Path to HRU shapefile with Landuse_Cl attribute.
                 If None, skips landuse+glacier and 2-panel plots.
    """
    topo_dir = Path(topo_dir)
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    plot_catchment_dem(topo_dir, catchment_shp, outdir)
    plot_catchment_landuse(topo_dir, catchment_shp, outdir)
    plot_catchment_glaciers(topo_dir, catchment_shp, glacier_shp, outdir)

    if hru_shp is not None:
        plot_catchment_landuse_glaciers(topo_dir, catchment_shp, glacier_shp, hru_shp, outdir)
        plot_catchment_overview_2panel(topo_dir, catchment_shp, glacier_shp, hru_shp, outdir,
                                       basin_name=basin_name)
    print(f"\nAll overview figures saved to: {outdir}")


def run_meteo_validation(aws_temp_path, aws_precip_path, era5_dir, har_dir,
                         stn_lat, stn_lon, stn_elev, outdir, stn_name='AWS Himansh',
                         years=None):
    """Generate all AWS vs reanalysis comparison figures."""
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    aws_temp = load_aws_temperature(aws_temp_path)
    aws_precip = load_aws_precipitation(aws_precip_path)

    if years is None:
        all_years = set(aws_temp.index.year) | set(aws_precip.index.year)
        years = sorted(all_years)

    era5_temp = load_era5_temperature(era5_dir, stn_lat, stn_lon, years)
    era5_precip = load_era5_precipitation(era5_dir, stn_lat, stn_lon, years)
    har_temp = load_har_temperature(har_dir, stn_lat, stn_lon, years)
    har_precip = load_har_precipitation(har_dir, stn_lat, stn_lon, years)
    era5_elev = get_era5_elevation(era5_dir, stn_lat, stn_lon)

    plot_temperature_comparison(aws_temp, era5_temp, har_temp, outdir,
                                stn_name=stn_name, stn_elev=stn_elev,
                                era5_elev=era5_elev)
    plot_precipitation_comparison(aws_precip, era5_precip, har_precip, outdir,
                                 stn_name=stn_name)
    plot_meteo_validation_slide(aws_temp, aws_precip, era5_temp, era5_precip,
                                har_temp, har_precip, outdir,
                                stn_name=stn_name, stn_elev=stn_elev,
                                era5_elev=era5_elev)

    print(f"\nAll meteo validation figures saved to: {outdir}")


# ============================================================
# Example usage
# ============================================================
if __name__ == '__main__':
    base = Path('/home/jberg/OneDrive/Raven_worldwide')
    topo_dir = base / 'chandra_1/03_model_setups/catchment_0101/topo_files'
    catchment_shp = base / '01_data/topo/catchment_shapefile/catchment_shape_0101.shp'
    glacier_shp = topo_dir / 'clipped_glacier.shp'
    hru_shp = topo_dir / 'HRU.shp'
    outdir = base / 'chandra_1/multi_config_plots_catchment_0101'

    run_catchment_overview(topo_dir, catchment_shp, glacier_shp, outdir, hru_shp=hru_shp)

    aws_temp = '/home/jberg/OneDrive/Data_collection_Himalaya/Meteo/Chandra/AWS daily_Himansh _Chandra.xlsx'
    aws_precip = '/home/jberg/OneDrive/Data_collection_Himalaya/Meteo/Chandra/Precip_1hr_Himansch.csv'
    era5_dir = base / '01_data/meteo/Indus-Era5'
    har_dir = base / '01_data/meteo/HAR'

    run_meteo_validation(aws_temp, aws_precip, era5_dir, har_dir,
                         stn_lat=32.45, stn_lon=77.58, stn_elev=4050, outdir=outdir)
