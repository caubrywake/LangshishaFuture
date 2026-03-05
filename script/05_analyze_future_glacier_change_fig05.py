# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
===============================================================================
Script: Future Glacier Change and SPHY–Rounce Comparison (Multi-GCM, Multi-SSP)
Author: Dr. Caroline Aubry-Wake
Affiliation: University of Lethbridge
Date: 2025-11-20
Python: 3.9+

Purpose
-------
Quantify and visualize glacier retreat under future climate scenarios
(SSP1-2.6, SSP3-7.0, SSP5-8.5) using SPHY glacierized hydrology simulations,
and compare projected glacier volume change against Rounce et al. (2023)
multi-model ensemble projections for the Langshisha basin.

The script computes ensemble glacier area, volume, thickness, and mean
glacier elevation across multiple GCMs to assess the magnitude, temporal
evolution, and spatial pattern of glacier loss.

Main workflow
-------------
For each SSP (ssp126, ssp370, ssp585) and all available GCMs:

1. Load annual glacier area (GlacArea_*.map), ice volume (vIce_*.map),
   and thickness (iceDepth_*.map) rasters, masked by the basin footprint.
2. Compute basin-total glacier area [km²] and volume [km³] time series
   for each GCM.
3. Aggregate ensemble mean ± standard deviation across GCMs per year.
4. Normalize ensemble area and volume time series to the earliest available
   simulation year (baseline_year, typically 2015) to obtain % change.
5. Compute ensemble-mean thickness maps for 2025 and 2095.
6. Compute mean glacier elevation in 2025 and 2095 using a DEM.
7. Load Rounce et al. (2023) glacier mass and area projections (NetCDF),
   filter to Langshisha glaciers, normalize to 2015 baseline, and compute
   ensemble mean ± standard deviation.
8. Produce SPHY–Rounce comparison figures and export summary tables.

Inputs
------
SPHY raster inputs:
    - GlacArea_YYYYMMDD.map   → glacierized area (binary per pixel)
    - vIce_YYYYMMDD.map       → ice volume (m³ per pixel)
    - iceDepth_YYYYMMDD.map   → ice thickness (m)
    - footprint.tif           → basin mask
    - demfilled.map           → DEM (optional, for elevation statistics)

Directory layout:
    base_dir = "/scratch/depfg/aubry001/out/fut"
        base_dir/<GCM>/<SSP>/GlacArea_*.map
        base_dir/<GCM>/<SSP>/vIce_*.map
        base_dir/<GCM>/<SSP>/iceDepth_*.map

Rounce projections:
    - R15_glac_mass_annual_50sets_2000_2100-<ssp>.nc
    - R15_glac_area_annual_50sets_2000_2100-<ssp>.nc
    Filtered to Langshisha glaciers using RGI IDs.

Vector overlays:
    - Basin outline
    - Glacier outlines
    - River network
    - AWS and basin outlet locations

Outputs
-------
Figures (saved in fig_dir):
    - glacier_change_timeseries.png / .pdf
        → Ensemble % change in glacier area and volume
          from baseline_year (typically 2015).
    - glacier_thickness_maps_2095.png / .pdf
        → Ensemble-mean ice thickness (2095) for SSP126, SSP370, SSP585.
    - glacier_volume_sphy_vs_rounce_1x3.png
        → SPHY vs Rounce glacier volume change (% from baseline)
          for each SSP (ensemble mean ± 1σ).

Tables (CSV):
    - glacier_change_summary_<baseline_year>_<final_year>.csv
        → Per-SSP mean and std of % area and volume change.
    - glacier_mean_elevation_2025_2095.csv (if DEM available)
        → Ensemble mean glacier elevation (2025, 2095, Δ elevation).
    - glacier_volume_comparison_SPHY_vs_Rounce.csv
        → Final-year % change comparison (SPHY vs Rounce, mean ± std).

Notes
-----
- All area and volume metrics are basin-integrated and masked by footprint.tif.
- Percent changes are normalized to ensemble mean glacier area/volume
  in baseline_year (typically 2015).
- Rounce comparisons are basin-aggregated over Langshisha glaciers.
- Glacier defined as pixels with ice thickness > 0.1 m.
- Ensemble thickness maps are simple means across available GCMs.

===============================================================================
"""

# need to add rounce prediction to this! 
import os
import glob
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import xarray as xr

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import rasterio


# === SETTINGS ===
base_dir = "/scratch/depfg/aubry001/out/fut"
footprint_file = "/scratch/depfg/aubry001/geospatial/Langshisha_basin_outline/footprint.tif"
fig_dir = "/scratch/depfg/aubry001/fig/futlong_glacier"
os.makedirs(fig_dir, exist_ok=True)
rounce_dir = "/scratch/depfg/aubry001/geospatial/Rounce_massbalance/256182556"

ssps = ["ssp126", "ssp370", "ssp585"]
PIXEL_AREA = 20 * 20  # m² (20x20m grid)

# === Load basin footprint ===
with rasterio.open(footprint_file) as src:
    footprint = src.read(1)
mask = footprint > 0

# Paths
footprintpath = "/scratch/depfg/aubry001/geospatial/Langshisha_basin_outline/footprint.tif"
basinoutline = gpd.read_file("/scratch/depfg/aubry001/geospatial/Langshisha_basin_outline/LangshisaBasinandSide_outline_2mDEM.shp")
glaciers = gpd.read_file("/scratch/depfg/aubry001/geospatial/RGI_withinbasin/RGI_withingbasin.shp")
location = gpd.read_file("/scratch/depfg/aubry001/geospatial/location_utm/location_utm.shp")
rivshp = gpd.read_file("/scratch/depfg/aubry001/geospatial/Langshisha_surfacestream/langshihsa_surfacestream_20230308.shp")
rivshp['id'] = range(1, len(rivshp)+1)

fig_dir = "/scratch/depfg/aubry001/fig/futlong_glacier"
os.makedirs(fig_dir, exist_ok=True)

# === Containers ===
ensemble_area = {}
ensemble_vol = {}
ensemble_thickness_2025 = {}
ensemble_thickness_2095 = {}

# === Loop over SSPs ===
for ssp in ssps:
    print(f"\n=== Processing {ssp.upper()} ===")
    gcms = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    gcm_area_series = []
    gcm_vol_series = []
    thickness_maps_2025 = []
    thickness_maps_2095 = []

    # --- Loop over GCMs ---
    for gcm in gcms:
        gcm_dir = os.path.join(base_dir, gcm, ssp)
        if not os.path.exists(gcm_dir):
            continue

        # Find all GlacArea and vIce maps
        area_files = sorted(glob.glob(os.path.join(gcm_dir, "GlacArea_*.map")))
        vol_files = sorted(glob.glob(os.path.join(gcm_dir, "vIce_*.map")))
        thick_files = sorted(glob.glob(os.path.join(gcm_dir, "iceDepth_*.map")))

        if not area_files or not vol_files:
            print(f"⚠️ Missing glacier files for {gcm}, {ssp}")
            continue

        # Extract years
        def get_year(f):
            digits = ''.join([c for c in os.path.basename(f) if c.isdigit()])
            return int(digits[:4]) if len(digits) >= 4 else None

        years = [get_year(f) for f in area_files]
        area_series = []
        vol_series = []

        # --- Loop over years ---
        for af, vf, yr in zip(area_files, vol_files, years):
            if yr is None:
                continue
            with rasterio.open(af) as src:
                a = np.where(mask, src.read(1), np.nan)
            with rasterio.open(vf) as src:
                v = np.where(mask, src.read(1), np.nan)

            # Total glacier area (m² → km²)
            area_total = np.nansum(a > 0) * PIXEL_AREA / 1e6
            # Total glacier volume (m³ → km³)
            vol_total = np.nansum(v) / 1e9

            area_series.append(area_total)
            vol_series.append(vol_total)

        df = pd.DataFrame({"Year": years, "Area_km2": area_series, "Volume_km3": vol_series})
        df["GCM"] = gcm
        gcm_area_series.append(df[["Year", "Area_km2"]])
        gcm_vol_series.append(df[["Year", "Volume_km3"]])

        # Save early/late thickness maps
        thick_2025 = [f for f in thick_files if "2025" in f]
        thick_2095 = [f for f in thick_files if "2095" in f]
        if thick_2025:
            with rasterio.open(thick_2025[0]) as src:
                thickness_maps_2025.append(np.where(mask, src.read(1), np.nan))
        if thick_2095:
            with rasterio.open(thick_2095[0]) as src:
                thickness_maps_2095.append(np.where(mask, src.read(1), np.nan))

    # --- Ensemble mean and std ---
    df_area = pd.concat(gcm_area_series)
    df_vol = pd.concat(gcm_vol_series)

    area_stats = df_area.groupby("Year")["Area_km2"].agg(["mean", "std"])
    vol_stats = df_vol.groupby("Year")["Volume_km3"].agg(["mean", "std"])

    # Normalize to 2025 baseline
    baseA = area_stats["mean"].iloc[0]
    baseV = vol_stats["mean"].iloc[0]
    area_stats["pct_change"] = (area_stats["mean"] - baseA) / baseA * 100
    vol_stats["pct_change"] = (vol_stats["mean"] - baseV) / baseV * 100

    ensemble_area[ssp] = area_stats
    ensemble_vol[ssp] = vol_stats

    # Ensemble mean thickness maps
    if thickness_maps_2025:
        ensemble_thickness_2025[ssp] = np.nanmean(np.stack(thickness_maps_2025), axis=0)
    if thickness_maps_2095:
        ensemble_thickness_2095[ssp] = np.nanmean(np.stack(thickness_maps_2095), axis=0)

############################################################
# === 1️⃣ FIGURE: % CHANGE IN AREA & VOLUME ===
############################################################

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
colors = {"ssp126": "#2b83ba", "ssp370": "#fdae61", "ssp585": "#d7191c"}

# Determine baseline year from ensemble data (first year for all SSPs)
baseline_year = min(stats.index.min() for stats in ensemble_area.values())

# ------------------------
#  AREA PANEL  (axes[0])
# ------------------------
for ssp in ssps:
    if ssp in ensemble_area:
        area_stats = ensemble_area[ssp]
        years_a = area_stats.index

        # SPHY mean area change
        axes[0].plot(
            years_a,
            area_stats["pct_change"],
            color=colors[ssp],
            lw=2,
            label=f"SPHY {ssp.upper()}",
        )

        # SPHY ±1σ
        pct_std_a = area_stats["std"] / area_stats["mean"].loc[baseline_year] * 100
        axes[0].fill_between(
            years_a,
            area_stats["pct_change"] - pct_std_a,
            area_stats["pct_change"] + pct_std_a,
            color=colors[ssp],
            alpha=0.15,
        )


axes[0].set_ylabel("Glacier area change\n(2015–2099, %)")
axes[0].text(0.02, 0.97, "(a)", transform=axes[0].transAxes, ha="left", va="top")

# ------------------------
#  VOLUME PANEL (axes[1])
# ------------------------
for ssp in ssps:
    if ssp in ensemble_vol:
        vol_stats = ensemble_vol[ssp]
        years_v = vol_stats.index

        # SPHY mean volume change
        axes[1].plot(
            years_v,
            vol_stats["pct_change"],
            color=colors[ssp],
            lw=2,
            label=f"SPHY {ssp.upper()}",
        )

        # SPHY ±1σ envelope
        pct_std_v = vol_stats["std"] / vol_stats["mean"].loc[baseline_year] * 100
        axes[1].fill_between(
            years_v,
            vol_stats["pct_change"] - pct_std_v,
            vol_stats["pct_change"] + pct_std_v,
            color=colors[ssp],
            alpha=0.15,
        )

axes[1].set_ylabel("Glacier volume change\n(2015–2099, %)")
axes[1].text(0.02, 0.97, "(b)", transform=axes[1].transAxes, ha="left", va="top")

# ------------------------
#  Shared formatting
# ------------------------
for ax in axes:
    ax.set_xlabel("Year")
    ax.grid(alpha=0.3)
    ax.set_xlim(baseline_year, 2100)
    ax.set_ylim(-105, 5)

# Show single legend from left panel
axes[0].legend(frameon=False, ncol=1, title="Scenario")

plt.tight_layout()
out_path = os.path.join(fig_dir, "glacier_change_timeseries.png")
plt.savefig(out_path, dpi=300)
out_path = os.path.join(fig_dir, "glacier_change_timeseries.pdf")
plt.savefig(out_path, dpi=300)
plt.show()
print(f"📈 Saved glacier change time series → {out_path}")



# === SUMMARY TABLE: % change between baseline and final year ===
summary_rows = []

# Baseline = earliest year; final = latest year across SSPs
baseline_year = min(stats.index.min() for stats in ensemble_area.values())
final_year = max(stats.index.max() for stats in ensemble_area.values())

for ssp in ssps:
    if ssp not in ensemble_area or ssp not in ensemble_vol:
        continue

    area_stats = ensemble_area[ssp]
    vol_stats = ensemble_vol[ssp]

    # Ensure baseline_year and final_year exist in each series
    if baseline_year not in area_stats.index or final_year not in area_stats.index:
        continue
    if baseline_year not in vol_stats.index or final_year not in vol_stats.index:
        continue

    # Baseline absolute means
    baseA = area_stats.loc[baseline_year, "mean"]
    baseV = vol_stats.loc[baseline_year, "mean"]

    # Mean % change at final year (already normalized in pct_change)
    area_pct_mean = area_stats.loc[final_year, "pct_change"]
    vol_pct_mean = vol_stats.loc[final_year, "pct_change"]

    # Approximate std in % at final year (std of absolute values divided by baseline mean)
    area_pct_std = area_stats.loc[final_year, "std"] / baseA * 100.0 if baseA != 0 else np.nan
    vol_pct_std = vol_stats.loc[final_year, "std"] / baseV * 100.0 if baseV != 0 else np.nan

    summary_rows.append({
        "ssp": ssp,
        "baseline_year": baseline_year,
        "final_year": final_year,
        "area_pct_change_mean": area_pct_mean,
        "area_pct_change_std": area_pct_std,
        "volume_pct_change_mean": vol_pct_mean,
        "volume_pct_change_std": vol_pct_std,
    })

summary_df = pd.DataFrame(summary_rows)

# Save to CSV
summary_csv = os.path.join(fig_dir, f"glacier_change_summary_{baseline_year}_{final_year}.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"✅ Saved glacier change summary → {summary_csv}")

# Optional: pretty print to console
print("\n=== Glacier change summary (% from baseline) ===")
print(summary_df.round(1).to_string(index=False))

# === OPTIONAL: mean glacier elevation from ensemble thickness (2025 vs 2095) ===

# Path to DEM in same grid / projection as iceDepth maps
dem_file = "/scratch/depfg/aubry001/minp/demfilled.map"  # <-- update to your DEM

if os.path.exists(dem_file):
    with rasterio.open(dem_file) as src_dem:
        dem = src_dem.read(1).astype(float)
    # Apply basin mask
    dem = np.where(mask, dem, np.nan)

    def mean_glacier_elev(thickness_map, dem_array, threshold=0.1):
        """Mean elevation of pixels with ice thickness > threshold."""
        gl_mask = thickness_map > threshold
        elev_vals = np.where(gl_mask, dem_array, np.nan)
        return float(np.nanmean(elev_vals))

    elev_rows = []
    for ssp in ssps:
        if ssp in ensemble_thickness_2025 and ssp in ensemble_thickness_2095:
            thk_2025 = ensemble_thickness_2025[ssp]
            thk_2095 = ensemble_thickness_2095[ssp]

            mean_elev_2025 = mean_glacier_elev(thk_2025, dem)
            mean_elev_2095 = mean_glacier_elev(thk_2095, dem)

            elev_rows.append({
                "ssp": ssp,
                "mean_glacier_elev_2025_m": mean_elev_2025,
                "mean_glacier_elev_2095_m": mean_elev_2095,
                "delta_elev_m": mean_elev_2095 - mean_elev_2025
            })

    if elev_rows:
        elev_df = pd.DataFrame(elev_rows)
        elev_csv = os.path.join(fig_dir, "glacier_mean_elevation_2025_2095.csv")
        elev_df.to_csv(elev_csv, index=False)
        print(f"✅ Saved mean glacier elevation summary → {elev_csv}")
        print("\n=== Mean glacier elevation (ensemble, 2025 → 2095) ===")
        print(elev_df.round(1).to_string(index=False))
    else:
        print("⚠️ No ensemble thickness maps for both 2025 and 2095; skipping elevation summary.")
else:
    print(f"⚠️ DEM file not found: {dem_file} — skipping elevation summary.")



############################################################
#%% === FIGURE: END-OF-CENTURY ICE THICKNESS MAPS (2095, 3 SSPs) ===
############################################################


# Get grid extent from footprint
with rasterio.open(footprintpath) as src:
    footprint = src.read(1, masked=True)
    grid_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

# Define colormap (0 ice → gray, thicker ice → blue)
colors = [(0.9, 0.9, 0.9), (0.8, 0.9, 1.0), (0.2, 0.5, 0.9)]
cmap = mcolors.LinearSegmentedColormap.from_list("ice_cmap", colors)
vmin, vmax = 0, 80  # adjust if your glaciers are thicker/thinner

# === Plot ===
#%% Make panels a bit larger and reserve space on the right for the colorbar
fig, axes = plt.subplots(1, 3, figsize=(9, 4.8), sharex=True, sharey=True)
ssps = ["ssp126", "ssp370", "ssp585"]
panel_labels = ["(c) SSP126", "(d) SSP370", "(e) SSP585"]

im = None  # placeholder for colorbar

for col, ssp in enumerate(ssps):
    ax = axes[col]

    if ssp not in ensemble_thickness_2095:
        ax.axis("off")
        continue

    ice_map = ensemble_thickness_2095[ssp]
    masked = np.ma.masked_where(ice_map <= 0.1, ice_map)

    im = ax.imshow(
        masked,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=grid_extent
    )

    # Thicker glacier outline for legend meaning
    glaciers.boundary.plot(ax=ax, color="k", linewidth=1, alpha=0.9)
    basinoutline.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
    rivshp.plot(ax=ax, color='black', linewidth=0.5)

    # AWS & basin outlet as black dots
    location.plot(ax=ax, color='black', markersize=14)

    # Panel label in upper-left (no suptitle, no per-panel title)
    ax.text(
        0.02, 0.95,
        panel_labels[col],
        transform=ax.transAxes,
        ha="left",
        va="top"
    )

    ax.axis("off")


# Manually position subplots: tightly spaced
# Format: [left, bottom, width, height] in figure coordinates
axes[0].set_position([0.1, 0.15, 0.28, 0.75])  # Left panel
axes[1].set_position([0.35, 0.15, 0.28, 0.75])  # Center panel
axes[2].set_position([0.6, 0.15, 0.28, 0.75])  # Right panel


# Custom legend (horizontal layout)
legend_elements = [
    Line2D([0], [0], color="k", lw=1.5, label="Glacier extent (2015)"),
    Line2D([0], [0], marker="o", color="k", linestyle="None",
           markersize=6, label="AWS & basin outlet"),
]

# Place legend slightly below the middle panel, horizontally
axes[1].legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.10),  # moves it slightly outside the axis
    ncol=2,  # horizontal: two items in one row
    frameon=True,
    framealpha=0.9,
    facecolor="white",
)

# Colorbar placed manually (shorter height)
# [left, bottom, width, height] in figure coordinates
cax = fig.add_axes([0.86, 0.27, 0.015, 0.50])  # ← shorter & slightly higher
cbar = fig.colorbar(im, cax=cax, orientation="vertical")
cbar.set_label("Ice thickness (m)")

out_path = os.path.join(fig_dir, "glacier_thickness_maps_2095.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
out_path = os.path.join(fig_dir, "glacier_thickness_maps_2095.pdf")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"📈 Saved glacier thickness figure → {out_path}")
# %%

# === Rounce comparison files (multi-glacier basin) ===

# Langshisha glaciers: last 5 digits
glaciers_last5 = [
    "04215", "04244", "04245", "04614",
    "04240", "04241", "04216", "04293",
    "04292", "04290", "04291", "04288",
    "04289", "04252", "04311", "04280"
]
glacier_ids = [f"RGI60-15.{gid}" for gid in glaciers_last5]

rounce_data = {}
baseline_year = 2015
for ssp in ssps:

    # Correct filenames (without "_mad")
    rounce_file = os.path.join(
        rounce_dir,
        f"R15_glac_mass_annual_50sets_2000_2100-{ssp}.nc"
    )

    if not os.path.exists(rounce_file):
        print(f"⚠️ Rounce file not found for {ssp}: {rounce_file}")
        continue

    ds = xr.open_dataset(rounce_file)
    mass = ds["glac_mass_annual"]  # dims: ('model', 'glacier', 'year')

    # Filter to Langshisha glaciers
    sel = mass.where(mass["RGIId"].isin(glacier_ids), drop=True)

    if sel.sizes.get("glacier", 0) == 0:
        print(f"⚠️ No Langshisha glaciers found for {ssp}")
        continue

    print(f"✓ {ssp}: using {sel.sizes['glacier']} Langshisha glaciers")

    # Select years starting at baseline_year (e.g. 2015)
    sel = sel.sel(year=slice(baseline_year, None))  # <-- CORRECT
    years = sel["year"].values

    # Average over glaciers → model × time
    mean_glac = sel.mean(dim="glacier")

    # Normalize to 0% at baseline
    ref = mean_glac.sel(year=baseline_year)
    pct = (mean_glac / ref) * 100.0 - 100.0

    rounce_data[ssp] = {
        "years": years,
        "mean": pct.mean(dim="model").values,
        "std": pct.std(dim="model").values,
    }

# === Rounce glacier AREA comparison (multi-glacier basin) ===
rounce_area_data = {}

for ssp in ssps:
    rounce_area_file = os.path.join(
        rounce_dir,
        f"R15_glac_area_annual_50sets_2000_2100-{ssp}.nc"
    )

    if not os.path.exists(rounce_area_file):
        print(f"⚠️ Rounce AREA file not found for {ssp}: {rounce_area_file}")
        continue

    dsA = xr.open_dataset(rounce_area_file)
    area = dsA["glac_area_annual"]   # dims: ensemble, glacier, year

    # filter to Langshisha glaciers
    selA = area.where(area["RGIId"].isin(glacier_ids), drop=True)

    if selA.sizes.get("glacier", 0) == 0:
        print(f"⚠️ No Langshisha glaciers found in Rounce AREA for {ssp}")
        continue

    yearsA = selA["year"].values
    idxA = yearsA >= baseline_year
    selA = selA.isel(year=idxA)
    yearsA = yearsA[idxA]

    # average over glaciers
    mean_glac_A = selA.mean(dim="glacier")

    # normalize to baseline (2015 = 0%)
    refA = mean_glac_A.isel(year=0)
    pctA = (mean_glac_A / refA) * 100.0 - 100.0

    rounce_area_data[ssp] = {
        "years": yearsA,
        "mean": pctA.mean(dim="model").values,
        "std": pctA.std(dim="model").values,
    }

############################################################
# === FIGURE: SPHY vs Rounce glacier volume change (1×3) ===
############################################################

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
colors = {"ssp126": "#2b83ba", "ssp370": "#fdae61", "ssp585": "#d7191c"}

# Baseline year already used to normalise SPHY volume (% change)
baseline_year = min(stats.index.min() for stats in ensemble_area.values())

panel_order  = ["ssp126", "ssp370", "ssp585"]
panel_titles = {
    "ssp126": "SSP1-2.6",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
panel_labels = {
    "ssp126": "(a)",
    "ssp370": "(b)",
    "ssp585": "(c)",
}

for i, ssp in enumerate(panel_order):
    ax = axes[i]
    color = colors[ssp]

    # ----- SPHY volume (ensemble) -----
    if ssp in ensemble_vol:
        vol_stats = ensemble_vol[ssp]
        years_v = vol_stats.index

        ax.plot(
            years_v,
            vol_stats["pct_change"],
            color=color,
            lw=2,
            label="SPHY",
        )

        # SPHY ±1σ in %
        pct_std_v = vol_stats["std"] / vol_stats["mean"].loc[baseline_year] * 100.0
        ax.fill_between(
            years_v,
            vol_stats["pct_change"] - pct_std_v,
            vol_stats["pct_change"] + pct_std_v,
            color=color,
            alpha=0.20,
        )

    # ----- Rounce volume (ensemble) -----
    if ssp in rounce_data:
        yrs_r = rounce_data[ssp]["years"]
        mean_r = rounce_data[ssp]["mean"]
        std_r  = rounce_data[ssp]["std"]

        ax.plot(
            yrs_r,
            mean_r,
            color=color,
            ls="--",
            lw=1.8,
            label="Rounce",
        )
        ax.fill_between(
            yrs_r,
            mean_r - std_r,
            mean_r + std_r,
            color=color,
            alpha=0.12,
        )

    # Titles & panel labels
    ax.set_title(panel_titles.get(ssp, ssp.upper()))
    ax.text(
        0.03, 0.96,
        panel_labels.get(ssp, ""),
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    ax.grid(alpha=0.3)
    ax.set_xlim(baseline_year, 2100)
    ax.set_ylim(-105, 5)

    # Local legend: just SPHY vs Rounce
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels,
            frameon=False,
            loc="lower left",
        )

# Shared axis labels
axes[0].set_ylabel(f"Glacier volume change from {baseline_year} (%)")
for ax in axes:
    ax.set_xlabel("Year")

plt.tight_layout()
out_path = os.path.join(fig_dir, "glacier_volume_sphy_vs_rounce_1x3.png")
plt.savefig(out_path, dpi=300)
plt.show()
print(f"📈 Saved SPHY–Rounce 1×3 volume comparison → {out_path}")

# === Build comparison table: SPHY vs Rounce (2100 % change ± std) ===

comparison_rows = []
for ssp in ssps:
    if ssp not in rounce_data or ssp not in ensemble_vol:
        continue
    
    # SPHY projected change at 2100
    vol_stats = ensemble_vol[ssp]
    final_year = vol_stats.index.max()
    sphy_mean = vol_stats.loc[final_year, "pct_change"]          # %
    sphy_std = vol_stats.loc[final_year, "std"] / vol_stats["mean"].iloc[0] * 100  # %

    # Rounce (final timestep in dataset)
    r_years = rounce_data[ssp]["years"]
    r_mean_series = rounce_data[ssp]["mean"]
    r_std_series = rounce_data[ssp]["std"]

    r_year = r_years[-1]        # final year in Rounce data
    r_mean = r_mean_series[-1]  # %
    r_std = r_std_series[-1]    # %

    comparison_rows.append({
        "SSP": ssp.upper(),
        "SPHY_final_year": final_year,
        "SPHY_change_%": round(float(sphy_mean), 1),
        "SPHY_std_%": round(float(sphy_std), 1),
        "Rounce_final_year": int(r_year),
        "Rounce_change_%": round(float(r_mean), 1),
        "Rounce_std_%": round(float(r_std), 1),
    })

comparison_df = pd.DataFrame(comparison_rows)

# Save to CSV
comp_csv = os.path.join(fig_dir, "glacier_volume_comparison_SPHY_vs_Rounce.csv")
comparison_df.to_csv(comp_csv, index=False)

print("📄 SPHY vs Rounce table saved →", comp_csv)
print(comparison_df)

print("\n=== Glacier volume change comparison (SPHY vs Rounce, rounded) ===")
print(
    comparison_df.round({
        "SPHY_change_%": 0,
        "SPHY_std_%": 0,
        "Rounce_change_%": 0,
        "Rounce_std_%": 0
    }).astype({
        "SPHY_final_year": "int",
        "Rounce_final_year": "int"
    }).to_string(index=False)
)
