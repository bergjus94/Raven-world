"""
QDM bias correction explanation figure for presentation.
Shows: raw GCM vs ERA5-Land vs QDM-corrected, for temperature and precipitation.
Catchment 0101 (Chandra @ Tandi), MRI-ESM2-0, SSP126.
"""

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

# --- Paths ---
base_obs = '/home/jberg/OneDrive/Raven_worldwide/test_config_all/03_model_setups_glogem_subdaily_future/catchment_0101/data_obs'
base_raw = '/home/jberg/OneDrive/Raven_worldwide/01_data/CMIP6/MRI-ESM2-0'

# --- Load data ---
print("Loading data...")

# ERA5-Land (reference)
era5_t = xr.open_dataset(f'{base_obs}/era5_land_temp_mean.nc')['t2m']
era5_p = xr.open_dataset(f'{base_obs}/era5_land_precip.nc')['tp']

# QDM-corrected CMIP6
qdm_hist_t = xr.open_dataset(f'{base_obs}/cmip6_MRI-ESM2-0_historical_temp_mean.nc')['t2m']
qdm_ssp_t = xr.open_dataset(f'{base_obs}/cmip6_MRI-ESM2-0_ssp126_temp_mean.nc')['t2m']
qdm_hist_p = xr.open_dataset(f'{base_obs}/cmip6_MRI-ESM2-0_historical_precip.nc')['tp']
qdm_ssp_p = xr.open_dataset(f'{base_obs}/cmip6_MRI-ESM2-0_ssp126_precip.nc')['tp']

# Raw CMIP6 (before correction)
raw_hist_t = xr.open_dataset(f'{base_raw}/historical/tas.nc')['tas']
raw_ssp_t = xr.open_dataset(f'{base_raw}/ssp126/tas.nc')['tas']
raw_hist_p = xr.open_dataset(f'{base_raw}/historical/pr.nc')['pr']
raw_ssp_p = xr.open_dataset(f'{base_raw}/ssp126/pr.nc')['pr']

# --- Spatial average (catchment mean) ---
print("Computing spatial means...")
era5_t_mean = era5_t.mean(dim=['lat', 'lon'])
era5_p_mean = era5_p.mean(dim=['lat', 'lon'])

# QDM-corrected: handle potential duplicate dim names
qdm_hist_t_mean = qdm_hist_t.mean(dim=['lat', 'lon'])
qdm_ssp_t_mean = qdm_ssp_t.mean(dim=['lat', 'lon'])
qdm_hist_p_mean = qdm_hist_p.mean(dim=['lat', 'lon'])
qdm_ssp_p_mean = qdm_ssp_p.mean(dim=['lat', 'lon'])

# Raw CMIP6: select nearest grid cells to catchment center, convert units
# Catchment 0101 center ~32.4N, 77.3E; raw grid is coarse (~1.1 deg)
# Use nearest-neighbor selection to get the 2 closest lat × 2 closest lon points
raw_hist_t_sel = (raw_hist_t.sel(lat=[31.96, 33.08], lon=[76.5, 77.625], method='nearest') - 273.15).mean(dim=['lat', 'lon'])
raw_ssp_t_sel = (raw_ssp_t.sel(lat=[31.96, 33.08], lon=[76.5, 77.625], method='nearest') - 273.15).mean(dim=['lat', 'lon'])
raw_hist_p_sel = (raw_hist_p.sel(lat=[31.96, 33.08], lon=[76.5, 77.625], method='nearest') * 86400).mean(dim=['lat', 'lon'])
raw_ssp_p_sel = (raw_ssp_p.sel(lat=[31.96, 33.08], lon=[76.5, 77.625], method='nearest') * 86400).mean(dim=['lat', 'lon'])

# --- Compute monthly climatologies for the training period ---
print("Computing climatologies...")
train_start, train_end = '1980', '2014'

era5_t_clim = era5_t_mean.sel(time=slice(train_start, train_end)).groupby('time.month').mean()
era5_p_clim = era5_p_mean.sel(time=slice(train_start, train_end)).groupby('time.month').mean()

raw_t_clim = raw_hist_t_sel.sel(time=slice(train_start, train_end)).groupby('time.month').mean()
raw_p_clim = raw_hist_p_sel.sel(time=slice(train_start, train_end)).groupby('time.month').mean()

qdm_t_clim = qdm_hist_t_mean.sel(time=slice(train_start, train_end)).groupby('time.month').mean()
qdm_p_clim = qdm_hist_p_mean.sel(time=slice(train_start, train_end)).groupby('time.month').mean()

# Future climatology (2070-2100)
fut_start, fut_end = '2070', '2100'
raw_ssp_t_clim = raw_ssp_t_sel.sel(time=slice(fut_start, fut_end)).groupby('time.month').mean()
raw_ssp_p_clim = raw_ssp_p_sel.sel(time=slice(fut_start, fut_end)).groupby('time.month').mean()
qdm_ssp_t_clim = qdm_ssp_t_mean.sel(time=slice(fut_start, fut_end)).groupby('time.month').mean()
qdm_ssp_p_clim = qdm_ssp_p_mean.sel(time=slice(fut_start, fut_end)).groupby('time.month').mean()

# --- Annual means for time series ---
print("Computing annual means...")
era5_t_annual = era5_t_mean.groupby('time.year').mean()
raw_hist_t_annual = raw_hist_t_sel.groupby('time.year').mean()
raw_ssp_t_annual = raw_ssp_t_sel.groupby('time.year').mean()
qdm_hist_t_annual = qdm_hist_t_mean.groupby('time.year').mean()
qdm_ssp_t_annual = qdm_ssp_t_mean.groupby('time.year').mean()

era5_p_annual = era5_p_mean.groupby('time.year').sum()
raw_hist_p_annual = raw_hist_p_sel.groupby('time.year').sum()
raw_ssp_p_annual = raw_ssp_p_sel.groupby('time.year').sum()
qdm_hist_p_annual = qdm_hist_p_mean.groupby('time.year').sum()
qdm_ssp_p_annual = qdm_ssp_p_mean.groupby('time.year').sum()

# =====================================================================
# FIGURE
# =====================================================================
print("Creating figure...")

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
})

fig = plt.figure(figsize=(14, 9))
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.4)

month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
months = np.arange(1, 13)

c_era5 = 'black'
c_raw = '#d62728'
c_qdm = '#1f77b4'
c_fut_raw = '#ff9896'

# --- Panel A: Monthly temperature climatology (historical) ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(months, raw_t_clim.values, 's--', color=c_raw, linewidth=2,
         markersize=6, label='Raw GCM (MRI-ESM2-0)', zorder=2)
ax1.plot(months, qdm_t_clim.values, '^-', color=c_qdm, linewidth=2,
         markersize=6, label='QDM-corrected', zorder=4)
ax1.plot(months, era5_t_clim.values, 'o-', color=c_era5, linewidth=2.5,
         markersize=6, label='ERA5-Land (reference)', zorder=3)
ax1.set_xticks(months)
ax1.set_xticklabels(month_labels, fontweight='bold')
ax1.set_ylabel('Temperature (°C)', fontweight='bold')
ax1.set_title('Temperature — Historical (1980–2014)')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
# Shade the bias
ax1.fill_between(months, era5_t_clim.values, raw_t_clim.values,
                 alpha=0.15, color=c_raw, label='_nolegend_')
mid_bias = (era5_t_clim.values[1] + raw_t_clim.values[1]) / 2
ax1.annotate('Bias', xy=(2, mid_bias),
             fontsize=11, color=c_raw, fontweight='bold', ha='center')

# --- Panel B: Monthly temperature climatology (future) ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(months, raw_ssp_t_clim.values, 's--', color=c_fut_raw, linewidth=2,
         markersize=6, label='Raw GCM SSP126 (2070–2100)', zorder=2)
ax2.plot(months, qdm_ssp_t_clim.values, '^-', color=c_qdm, linewidth=2.5,
         markersize=7, label='QDM SSP126 (2070–2100)', zorder=4)
ax2.plot(months, era5_t_clim.values, 'o-', color=c_era5, linewidth=2.5,
         markersize=6, label='ERA5-Land (historical)', zorder=3)
ax2.set_xticks(months)
ax2.set_xticklabels(month_labels, fontweight='bold')
ax2.set_title('Temperature — Future (2070–2100, SSP126)')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)
# Show the preserved delta
for m in [7, 8]:
    delta = qdm_ssp_t_clim.values[m-1] - era5_t_clim.values[m-1]
    ax2.annotate(f'Δ = {delta:+.1f}°C', xy=(m, qdm_ssp_t_clim.values[m-1]),
                 xytext=(m+0.8, qdm_ssp_t_clim.values[m-1]+2),
                 fontsize=10, color=c_qdm, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=c_qdm, lw=1.5))

# --- Panel C: Monthly precipitation climatology (historical) ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(months - 0.25, era5_p_clim.values, width=0.25, color=c_era5,
        label='ERA5-Land', alpha=0.8, edgecolor='black', linewidth=0.5)
ax3.bar(months, raw_p_clim.values, width=0.25, color=c_raw,
        label='Raw GCM', alpha=0.7, edgecolor='black', linewidth=0.5)
ax3.bar(months + 0.25, qdm_p_clim.values, width=0.25, color=c_qdm,
        label='QDM-corrected', alpha=0.8, edgecolor='black', linewidth=0.5)
ax3.set_xticks(months)
ax3.set_xticklabels(month_labels, fontweight='bold')
ax3.set_ylabel('Precipitation (mm/day)', fontweight='bold')
ax3.set_title('Precipitation — Historical (1980–2014)')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3, axis='y')

# --- Panel D: Monthly precipitation climatology (future) ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(months - 0.25, era5_p_clim.values, width=0.25, color=c_era5,
        label='ERA5-Land (historical)', alpha=0.8, edgecolor='black', linewidth=0.5)
ax4.bar(months, raw_ssp_p_clim.values, width=0.25, color=c_fut_raw,
        label='Raw GCM SSP126 (2070–2100)', alpha=0.7, edgecolor='black', linewidth=0.5)
ax4.bar(months + 0.25, qdm_ssp_p_clim.values, width=0.25, color=c_qdm,
        label='QDM SSP126 (2070–2100)', alpha=0.8, edgecolor='black', linewidth=0.5)
ax4.set_xticks(months)
ax4.set_xticklabels(month_labels, fontweight='bold')
ax4.set_title('Precipitation — Future (2070–2100, SSP126)')
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# Save
outpath = '/home/jberg/Raven-world/outputs/qdm_explanation_figure.png'
fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {outpath}")
