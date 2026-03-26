#!/usr/bin/env python3
"""
Validate glacier coverage: compare RGI glaciers in each catchment against
GloGEM discharge file to identify missing glaciers.

Produces per-catchment validation maps and a summary CSV.
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GLACIER_SHP = Path(
    "/home/jberg/OneDrive/Raven_worldwide/01_data/topo/glacier_outlines/"
    "nsidc0770_14.rgi60.SouthAsiaWest/14_rgi60_SouthAsiaWest.shp"
)
CATCHMENT_SHP = Path(
    "/home/jberg/OneDrive/Raven_worldwide/01_data/topo/catchment_shapefile/"
    "catchments_Indus/all_catchments.shp"
)
GLOGEM_NC = Path(
    "/home/jberg/OneDrive/Raven_worldwide/01_data/GloGEM/Indus/TSLA/"
    "GloGEM_Discharge_Indus_gfdl-esm4_ssp126.nc"
)
OUT_DIR = Path("/home/jberg/Raven-world/outputs/glacier_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading glacier shapefile...")
glaciers = gpd.read_file(GLACIER_SHP)

print("Loading catchment shapefile...")
catchments = gpd.read_file(CATCHMENT_SHP)

# Ensure same CRS
if glaciers.crs != catchments.crs:
    catchments = catchments.to_crs(glaciers.crs)

print("Loading GloGEM glacier IDs (coords only)...")
import netCDF4
nc = netCDF4.Dataset(str(GLOGEM_NC), "r")
glogem_ids_raw = nc.variables["glacier_id"][:]
# Handle char arrays: join characters along last axis
if glogem_ids_raw.ndim == 2:
    glogem_ids = set(
        b"".join(row).decode("utf-8").strip()
        for row in glogem_ids_raw
    )
else:
    glogem_ids = set(str(gid).strip() for gid in glogem_ids_raw)
nc.close()
print(f"  GloGEM has {len(glogem_ids)} glaciers")

# Extract numeric ID from RGIId for matching (e.g. RGI60-14.00123 -> 00123)
glaciers["rgi_num"] = glaciers["RGIId"].str.extract(r"RGI60-14\.(\d+)")[0]

# ---------------------------------------------------------------------------
# 2. Clip glaciers to each catchment & compare
# ---------------------------------------------------------------------------
print("\nClipping glaciers to catchments...")
summary_rows = []

for _, catch in catchments.iterrows():
    cid = catch["station_id"]
    cname = catch.get("station_na", cid)

    # Clip glaciers to this catchment
    catch_geom = gpd.GeoDataFrame([catch], crs=catchments.crs)
    clipped = gpd.sjoin(glaciers, catch_geom, how="inner", predicate="intersects")

    if len(clipped) == 0:
        summary_rows.append({
            "catchment_id": cid,
            "catchment_name": cname,
            "n_rgi_glaciers": 0,
            "n_in_glogem": 0,
            "n_missing": 0,
            "pct_missing": 0.0,
            "total_area_km2": 0.0,
            "missing_area_km2": 0.0,
            "pct_area_missing": 0.0,
        })
        continue

    # Tag each glacier: in GloGEM or not
    clipped = clipped.copy()
    clipped["in_glogem"] = clipped["rgi_num"].isin(glogem_ids)
    n_total = len(clipped)
    n_found = clipped["in_glogem"].sum()
    n_missing = n_total - n_found

    total_area = clipped["Area"].sum()
    missing_area = clipped.loc[~clipped["in_glogem"], "Area"].sum()

    summary_rows.append({
        "catchment_id": cid,
        "catchment_name": cname,
        "n_rgi_glaciers": n_total,
        "n_in_glogem": int(n_found),
        "n_missing": int(n_missing),
        "pct_missing": round(100 * n_missing / n_total, 1) if n_total > 0 else 0,
        "total_area_km2": round(total_area, 2),
        "missing_area_km2": round(missing_area, 2),
        "pct_area_missing": round(100 * missing_area / total_area, 1) if total_area > 0 else 0,
    })

    # --- Validation map ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    catch_geom.boundary.plot(ax=ax, color="black", linewidth=1.5)

    found = clipped[clipped["in_glogem"]]
    missing = clipped[~clipped["in_glogem"]]

    if len(found) > 0:
        found.plot(ax=ax, color="steelblue", alpha=0.6, edgecolor="none")
    if len(missing) > 0:
        missing.plot(ax=ax, color="red", alpha=0.7, edgecolor="none")

    patches = [
        mpatches.Patch(color="steelblue", alpha=0.6, label=f"In GloGEM ({n_found})"),
        mpatches.Patch(color="red", alpha=0.7, label=f"Missing ({n_missing})"),
        mpatches.Patch(facecolor="none", edgecolor="black", linewidth=1.5,
                       label="Catchment boundary"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=9)

    ax.set_title(
        f"Catchment {cid} ({cname})\n"
        f"{n_total} RGI glaciers | {n_missing} missing from GloGEM "
        f"({summary_rows[-1]['pct_missing']}%)\n"
        f"Missing area: {missing_area:.1f} km² / {total_area:.1f} km² "
        f"({summary_rows[-1]['pct_area_missing']}%)",
        fontsize=11,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"glacier_validation_{cid}.png", dpi=150)
    plt.close(fig)

    print(f"  {cid} ({cname}): {n_total} glaciers, {n_missing} missing "
          f"({summary_rows[-1]['pct_missing']}%), "
          f"missing area {missing_area:.1f}/{total_area:.1f} km²")

# ---------------------------------------------------------------------------
# 3. Summary table
# ---------------------------------------------------------------------------
summary = pd.DataFrame(summary_rows)
summary = summary.sort_values("pct_area_missing", ascending=False)
summary.to_csv(OUT_DIR / "glacier_coverage_summary.csv", index=False)

print(f"\n{'='*80}")
print("SUMMARY — Glaciers missing from GloGEM by catchment")
print(f"{'='*80}")
print(summary.to_string(index=False))
print(f"\nMaps saved to: {OUT_DIR}")
print(f"Summary CSV:   {OUT_DIR / 'glacier_coverage_summary.csv'}")
