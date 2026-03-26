#!/usr/bin/env python3
"""Generate server namelists for all catchments × configurations."""

from pathlib import Path

OUT_DIR = Path("/home/jberg/Raven-world/namelists_server")
OUT_DIR.mkdir(exist_ok=True)

# ── Catchment dates ────────────────────────────────────────────────────────
CATCHMENTS = {
    "0101": {"start": "2016-06-22", "end": "2022-12-31", "cali_end": "2020-12-31", "warmup": "2015-06-22"},
    "0102": {"start": "1980-01-01", "end": "2014-12-31", "cali_end": "1999-12-31", "warmup": "1978-01-01"},
    "0104": {"start": "1980-01-01", "end": "2008-12-31", "cali_end": "1995-12-31", "warmup": "1978-01-01"},
    "0105": {"start": "1980-01-01", "end": "2009-12-31", "cali_end": "1995-12-31", "warmup": "1978-01-01"},
    "0116": {"start": "1980-01-01", "end": "1998-12-31", "cali_end": "1990-12-31", "warmup": "1978-01-01"},
    "0117": {"start": "1982-01-01", "end": "1997-12-31", "cali_end": "1990-12-31", "warmup": "1980-01-01"},
    "0118": {"start": "1985-01-01", "end": "1997-12-31", "cali_end": "1992-12-31", "warmup": "1983-01-01"},
    "0119": {"start": "1980-01-01", "end": "1997-12-31", "cali_end": "1989-12-31", "warmup": "1978-01-01"},
    "0120": {"start": "1980-01-01", "end": "1997-12-31", "cali_end": "1989-12-31", "warmup": "1978-01-01"},
    "0121": {"start": "1980-01-01", "end": "1996-12-31", "cali_end": "1989-12-31", "warmup": "1978-01-01"},
    "0122": {"start": "1980-01-01", "end": "1997-12-31", "cali_end": "1989-12-31", "warmup": "1978-01-01"},
    "0130": {"start": "1980-01-01", "end": "2019-12-31", "cali_end": "2002-12-31", "warmup": "1978-01-01"},
}

# ── Configurations ─────────────────────────────────────────────────────────
# Each config: (suffix for filename, config_dir_suffix, coupled, extra_fields)
CONFIGS = [
    ("base",           "base",           False, {}),
    ("har",            "har",            False, {"meteo_source": "HAR"}),
    ("tphipr",         "tphipr",         False, {"precip_source": "TPHiPr"}),
    ("oudin",          "oudin",          False, {"pet_method": "OUDIN"}),
    ("glogem",         "glogem",         True,  {}),
    ("glogem_har",     "glogem_har",     True,  {"meteo_source": "HAR"}),
    ("glogem_tphipr",  "glogem_tphipr",  True,  {"precip_source": "TPHiPr"}),
    ("icemelt",        "icemelt",        True,  {"irrigation_variable": "icemelt"}),
    ("glogem_oudin",   "glogem_oudin",   True,  {"pet_method": "OUDIN"}),
    ("glogem_gmb",     "glogem_gmb",     True,  {"glogem_dir_override": "01_data/GloGEM/Indus/GMB"}),
]

MAIN_DIR = "/home/jberg@giub.local/Raven_world"
RAVEN_EXE = "/home/jberg@giub.local/RavenHydroFramework/build/Raven"


def generate_namelist(gauge_id, dates, config_name, config_dir_suffix, coupled, extras):
    """Generate a single namelist YAML string."""

    meteo_source = extras.get("meteo_source", "ERA5")
    glogem_dir = extras.get("glogem_dir_override", "01_data/GloGEM/Indus/TSLA")
    irrigation_variable = extras.get("irrigation_variable", "discharge")

    # Build the config description
    if coupled:
        if "glogem_dir_override" in extras and "GMB" in extras["glogem_dir_override"]:
            desc = f"coupled, GloGEM GMB"
        elif irrigation_variable == "icemelt":
            desc = f"coupled, GloGEM icemelt"
        else:
            desc = f"coupled, GloGEM TSLA"
    else:
        desc = "uncoupled (Raven internal glacier melt)"

    if meteo_source == "HAR":
        desc += ", HAR forcing"
    if extras.get("precip_source") == "TPHiPr":
        desc += ", TPHiPr precip"
    if extras.get("pet_method") == "OUDIN":
        desc += ", Oudin PET"

    lines = []
    lines.append(f"# =============================================================================")
    lines.append(f"# Namelist for gauge {gauge_id} ({desc})")
    lines.append(f"# Subdaily melt enabled, future projections enabled")
    lines.append(f"# =============================================================================")
    lines.append(f"")
    lines.append(f"gauge_id: '{gauge_id}'")
    lines.append(f"start_date: '{dates['start']}'")
    lines.append(f"end_date: '{dates['end']}'")
    lines.append(f"cali_end_date: '{dates['cali_end']}'")
    lines.append(f"warm_up_date: '{dates['warmup']}'")
    lines.append(f"")
    lines.append(f"model_type: 'HBV'")
    lines.append(f"template: true")
    lines.append(f"main_dir: '{MAIN_DIR}'")
    lines.append(f"config_dir: 'test_future_configs/{config_dir_suffix}'")
    lines.append(f"debug: true")
    lines.append(f"")

    # Coupling and glacier settings
    lines.append(f"coupled: {'true' if coupled else 'false'}")
    if coupled:
        lines.append(f"glogem_model: 'gfdl-esm4'")
        lines.append(f"glogem_scenario: 'ssp126'")
        lines.append(f"basin_name: 'Indus'")
        lines.append(f"irrigation_variable: '{irrigation_variable}'")

    # Subdaily — always enabled
    lines.append(f"subdaily_method: 'SUBDAILY_HOURLY_MELT'")
    lines.append(f"")

    # HRU settings
    lines.append(f"nconnect: 'multiple'")
    lines.append(f"criteria:")
    lines.append(f"  - elevation")
    lines.append(f"  - landuse")
    lines.append(f"")

    # Meteo source
    lines.append(f"meteo_source: '{meteo_source}'")

    # Optional overrides
    if extras.get("precip_source") == "TPHiPr":
        lines.append(f"precip_source: 'TPHiPr'")
        lines.append(f"tphipr_dir: '{MAIN_DIR}/01_data/meteo/TPHiPr'")

    if extras.get("pet_method") == "OUDIN":
        lines.append(f"pet_method: 'OUDIN'")

    # Future projections
    lines.append(f"future: false")
    lines.append(f"")

    # Future projections block (for run_full_pipeline.py)
    lines.append(f"# =============================================================================")
    lines.append(f"# Future projections (used by run_full_pipeline.py)")
    lines.append(f"# =============================================================================")
    lines.append(f"future_projections:")
    lines.append(f"  enabled: true")
    lines.append(f"  start_date: '1981-01-01'")
    lines.append(f"  end_date: '2100-09-30'")
    lines.append(f"  warm_up_date: '1980-01-01'")
    lines.append(f"  models:")
    lines.append(f"    - GFDL-ESM4")
    lines.append(f"    - IPSL-CM6A-LR")
    lines.append(f"    - MPI-ESM1-2-HR")
    lines.append(f"    - MRI-ESM2-0")
    lines.append(f"    - UKESM1-0-LL")
    lines.append(f"  scenarios:")
    lines.append(f"    - historical")
    lines.append(f"    - ssp126")
    lines.append(f"")

    lines.append(f"cmip6_dir: '{MAIN_DIR}/01_data/CMIP6'")
    lines.append(f"cmip6_train_start: '1980-01-01'")
    lines.append(f"cmip6_train_end: '2014-12-31'")
    lines.append(f"")

    # Paths
    lines.append(f"# =============================================================================")
    lines.append(f"# Paths (relative to main_dir or absolute)")
    lines.append(f"# =============================================================================")
    lines.append(f"meteo_dir: '{MAIN_DIR}/01_data/meteo/Indus-Era5'")
    lines.append(f"meteo_har_dir: '{MAIN_DIR}/01_data/meteo/HAR'")
    lines.append(f"shape_dir: '01_data/topo/catchment_shapefile/catchment_shape_{{gauge_id}}.shp'")
    lines.append(f"raster_dir: '01_data/topo/catchment_dem/dem_Indus.tif'")
    lines.append(f"stream_dir: '01_data/streamflow/streamflow_{{gauge_id}}.csv'")
    lines.append(f"landuse_dir: '01_data/topo/catchment_landuse/landuse_Indus_30m.tif'")
    lines.append(f"glacier_dir: '01_data/topo/glacier_outlines/nsidc0770_14.rgi60.SouthAsiaWest/14_rgi60_SouthAsiaWest.shp'")
    lines.append(f"glogem_dir: '{glogem_dir}'")
    lines.append(f"params_dir: '../src/config/default_params.yaml'")
    lines.append(f"")

    # Forcing files
    lines.append(f"# =============================================================================")
    lines.append(f"# Forcing file templates (populated by meteo preprocessors)")
    lines.append(f"# =============================================================================")
    lines.append(f"forcing_files:")
    lines.append(f"  - 'era5_land_2m_temperature_2000-2000_daily_max_combined.nc'")
    lines.append(f"  - 'era5_land_2m_temperature_2000-2000_daily_min_combined.nc'")
    lines.append(f"  - 'era5_land_2m_temperature_2000-2000_daily_mean_combined.nc'")
    lines.append(f"  - 'era5_land_total_precipitation_2000-2000_daily_combined.nc'")
    lines.append(f"")
    lines.append(f"forcing_files_har:")
    lines.append(f"  - 'har_land_2m_temperature_2000-2000_daily_max_combined.nc'")
    lines.append(f"  - 'har_land_2m_temperature_2000-2000_daily_min_combined.nc'")
    lines.append(f"  - 'har_land_2m_temperature_2000-2000_daily_mean_combined.nc'")
    lines.append(f"  - 'har_land_total_precipitation_2000-2000_daily_combined.nc'")
    lines.append(f"")

    # Calibration
    lines.append(f"# =============================================================================")
    lines.append(f"# Calibration settings")
    lines.append(f"# =============================================================================")
    lines.append(f"calibration:")
    lines.append(f"   metrics:")
    lines.append(f"     primary: 'KGE'")
    lines.append(f"     secondary: ['NSE', 'PBIAS', 'RMSE']")
    lines.append(f"   iterations: 5000")
    lines.append(f"   ngs: 15")
    lines.append(f"   cali_end_date: '{dates['cali_end']}'")
    lines.append(f"   vali_end_date: '{dates['end']}'")
    lines.append(f"")
    lines.append(f"raven_executable: '{RAVEN_EXE}'")

    return "\n".join(lines) + "\n"


# ── Generate all namelists ─────────────────────────────────────────────────
count = 0
for gauge_id, dates in sorted(CATCHMENTS.items()):
    for config_name, config_dir_suffix, coupled, extras in CONFIGS:
        content = generate_namelist(gauge_id, dates, config_name, config_dir_suffix, coupled, extras)

        if config_name == "base":
            filename = f"namelist_{gauge_id}.yaml"
        else:
            filename = f"namelist_{gauge_id}_{config_name}.yaml"

        out_path = OUT_DIR / filename
        out_path.write_text(content)
        count += 1

print(f"Generated {count} namelists in {OUT_DIR}")
