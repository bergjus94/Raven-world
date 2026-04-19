"""Compare cosine curve daily mean (Tmin+Tmax)/2 against ERA5 daily mean temperature."""
import matplotlib; matplotlib.use('Agg')
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys

gauge_id = sys.argv[1] if len(sys.argv) > 1 else '0122'
data_dir = sys.argv[2] if len(sys.argv) > 2 else f'/home/jberg/OneDrive/Raven_worldwide/model_runs/catchment_{gauge_id}/data_obs'

t_mean = xr.open_dataset(f'{data_dir}/era5_land_temp_mean.nc')['t2m']
t_min = xr.open_dataset(f'{data_dir}/era5_land_temp_min.nc')['t2m']
t_max = xr.open_dataset(f'{data_dir}/era5_land_temp_max.nc')['t2m']

cosine_mean = (t_max + t_min) / 2.0

era5_sp = t_mean.mean(dim=['lat', 'lon'])
cos_sp = cosine_mean.mean(dim=['lat', 'lon'])
bias = cos_sp - era5_sp

print(f"Time range: {str(t_mean.time.values[0])[:10]} to {str(t_mean.time.values[-1])[:10]}")
print(f"Grid: {t_mean.sizes['lat']}x{t_mean.sizes['lon']} = {t_mean.sizes['lat'] * t_mean.sizes['lon']} cells")
print(f"\n=== Spatially-averaged statistics ===")
print(f"ERA5 daily mean temp:      {float(era5_sp.mean()):.3f} C")
print(f"Cosine mean (Tmin+Tmax)/2: {float(cos_sp.mean()):.3f} C")
print(f"Mean bias:                 {float(bias.mean()):.3f} C")
print(f"Median bias:               {float(np.median(bias.values)):.3f} C")
print(f"Std of bias:               {float(bias.std()):.3f} C")
print(f"RMSE:                      {float(np.sqrt((bias**2).mean())):.3f} C")
print(f"Correlation:               {float(xr.corr(era5_sp, cos_sp)):.5f}")

bias_full = cosine_mean - t_mean
print(f"\n=== Per-pixel statistics ===")
print(f"Mean bias:                 {float(bias_full.mean()):.3f} C")
print(f"Std of bias:               {float(bias_full.std()):.3f} C")

monthly_bias = bias.groupby('time.month').mean()
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
print(f"\n=== Monthly mean bias ===")
for m, name in enumerate(months, 1):
    print(f"  {name}: {float(monthly_bias.sel(month=m)):+.3f} C")

# --- Plot ---
from pathlib import Path

FONTSIZE = 14
LABEL_PROPS = {'fontsize': FONTSIZE, 'fontweight': 'bold'}
TICK_SIZE = 12
LEGEND_PROPS = {'fontsize': 12, 'prop': {'weight': 'bold'}}

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Upper panel: time series (2-year window for readability)
t_slice = slice('1980-01-01', '1981-12-31')
axes[0].plot(era5_sp.sel(time=t_slice).time, era5_sp.sel(time=t_slice), label='ERA5 daily mean', alpha=0.8, lw=0.8, color='#1f77b4')
axes[0].plot(cos_sp.sel(time=t_slice).time, cos_sp.sel(time=t_slice), label='Cosine curve mean', alpha=0.8, lw=0.8, color='#ff7f0e')
axes[0].set_ylabel('Temperature (\u00b0C)', **LABEL_PROPS)
axes[0].legend(**LEGEND_PROPS)
axes[0].tick_params(axis='both', labelsize=TICK_SIZE)
axes[0].tick_params(axis='x', labelbottom=True)

# Lower panel: monthly boxplot of bias with mean bias annotation
mean_bias = float(bias.mean())
monthly_data = [bias.sel(time=bias.time.dt.month == m).values for m in range(1, 13)]
bp = axes[1].boxplot(monthly_data, tick_labels=months, patch_artist=True,
                     medianprops=dict(color='black', lw=1.5),
                     whiskerprops=dict(lw=1.2),
                     capprops=dict(lw=1.2),
                     flierprops=dict(markersize=3, alpha=0.4))
for patch in bp['boxes']:
    patch.set_facecolor('lightcoral')
    patch.set_alpha(0.7)
    patch.set_linewidth(1.2)
axes[1].axhline(0, color='black', lw=0.8)
axes[1].axhline(mean_bias, color='blue', lw=1.5, ls='--')
axes[1].set_ylabel('Bias (\u00b0C)', **LABEL_PROPS)
axes[1].tick_params(axis='both', labelsize=TICK_SIZE)
axes[1].tick_params(axis='x', rotation=0)

# Add mean bias text
axes[1].text(0.98, 0.95, f'Mean bias: {mean_bias:.2f} \u00b0C',
             transform=axes[1].transAxes, ha='right', va='top',
             fontsize=FONTSIZE, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

plt.tight_layout()
plot_dir = Path(data_dir).parent / 'plots'
plot_dir.mkdir(parents=True, exist_ok=True)
outpath = str(plot_dir / f'cosine_mean_vs_era5_mean_{gauge_id}.png')
plt.savefig(outpath, dpi=200, bbox_inches='tight')
print(f"\nPlot saved: {outpath}")
