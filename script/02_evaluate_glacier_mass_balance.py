# -*- coding: utf-8 -*-
"""
===============================================================================
Glacier Mass Balance Evaluation Script (SPHY)
===============================================================================
Author:        Caroline Aubry-Wake
Affiliation:   University of Lethbridge
Created:       2025-XX-XX
Python:        3.9+ (tested)
Environment:   eejit HPC cluster (interactive, VS Code)
Dependencies:  numpy, pandas, geopandas, rasterio, xarray, matplotlib, hydroeval

Description:
------------
This script evaluates glacier mass balance simulated by the SPHY hydrological
model for the Langshisha Glacier (RGI60-15.04176) using multiple independent
datasets. 

The analysis includes:
    (1) Extraction of annual glacier ice thickness from SPHY raster outputs
        (iceDepth_YYYYMMDD.map) and CSV summaries (ICE_DEPTH.csv).
    (2) Comparison of simulated cumulative glacier mass balance against
        observed annual mass balance from Dussaillant et al. (2019, 2024),
        evaluated over 2011–2023.
    (3) Evaluation of modelled glacier elevation change between 2011 and 2018
        against independent geodetic estimates from Ragettli et al. (2016),
        separately for clean-ice and debris-covered glacier areas.
    (4) Contextual comparison of simulated relative glacier mass change with
        the ensemble of projections from Rounce et al. (2023; SSP2-4.5),
        used as a plausibility benchmark rather than an observational
        validation.

Observed Data Sources:
----------------------
- Dussaillant et al. (2024):
  Annual glacier mass balance for Langshisha Glacier (m w.e. yr⁻¹),
  aggregated to cumulative mass balance for comparison with SPHY outputs.
- Ragettli et al. (2016):
  Geodetic elevation change rates (m yr⁻¹) for full glacier and
  debris-covered ice over 2011–2018.
- Rounce et al. (2023):
  Modelled glacier mass projections (SSP2-4.5 ensemble), used for
  contextual comparison only.

Evaluation Metrics:
-------------------
- Cumulative mass balance difference (m w.e.)
- Mean annual mass balance bias (m w.e. yr⁻¹)
- Mean elevation change rate bias (m yr⁻¹), full glacier and debris-covered ice
- Relative glacier mass change (%) for ensemble comparison

Spatial Domain:
---------------
Langshisha Glacier, Nepal Himalaya
RGI ID: RGI60-15.04176
Glacier ID (SPHY): 15

Temporal Coverage:
------------------
- SPHY simulations: 2011–2024
- Dussaillant mass balance: 2011–2023
- Ragettli elevation change: 2011–2018
- Rounce ensemble: 2011–2025 (contextual)

Outputs Created
---------------
Tables:
- MB_dussaillant_yearly_comparison.csv
- MB_dussaillant_summary.csv
- MB_ragettli_comparison.csv
- MB_rounce_ensemble_vs_sphy.csv
- MB_rounce_summary.csv

Figures:
- mass_balance_comparison.png
- mass_balance_SPHY_Rounce.png

===============================================================================
"""


import os
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
import xarray as xr
import hydroeval as he
import matplotlib.pyplot as plt

# ======================================================================
# PATHS
# ======================================================================
BASE_DIR = "/scratch/depfg/aubry001/out/calibgw_0"
FOLDER = "GlacSnowFrac_0.7_0.6.cfg"  
folder_path = os.path.join(BASE_DIR, FOLDER)
glacid_path = "/scratch/depfg/aubry001/geospatial/glacierID/Glacier_ID_shapefile.shp"
dussaillant_path = "/scratch/depfg/aubry001/geospatial/dusaillant_massbalance/ASE_gla_MEAN-CAL-mass-change-series_obs_unobs.csv"
rounce_path = "/scratch/depfg/aubry001/geospatial/Rounce_massbalance/256182556/R15_glac_mass_annual_50sets_2000_2100-ssp245.nc"

figdir = "/scratch/depfg/aubry001/fig/glaccmassbalance_present/"
os.makedirs(figdir, exist_ok=True)

# ======================================================================
# 1. Glacier ice thickness from SPHY outputs (iceDepth map files)
# ======================================================================
glac_id = gpd.read_file(glacid_path)
glacid_value = 15  # Langshisha glacier ID
geometry = glac_id[glac_id['GLACID'] == glacid_value]['geometry'].values[0]

file_paths = sorted([
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.endswith(".map") and f.startswith("iceDepth") and f != "iceDepth_20160101.map" and f != "iceDepth_20110101.map"
])

sums_per_file = []
years = []

for idx, file_path in enumerate(file_paths):
    with rasterio.open(file_path) as src:
        icethickness = src.read(1)
        mask = rasterio.mask.geometry_mask(
            [geometry],
            out_shape=icethickness.shape,
            transform=src.transform,
            invert=True
        )
        masked_icethickness = np.ma.masked_where(~mask, icethickness)
        sum_icethickness = np.sum(masked_icethickness)
        fname = os.path.basename(file_path)  # iceDepth_20111001.map
        year = int(fname.split("_")[1][:4])
        sums_per_file.append(sum_icethickness)
        years.append(year)

sums_array = np.array(sums_per_file)
years_array = np.array(years)
percentages = (sums_array / sums_array[0]) * 100

# ======================================================================
# 2. Ice thickness from CSV output (SPHY ICE_DEPTH.csv)
# ======================================================================
csv_path = os.path.join(folder_path, "ICE_DEPTH.csv")
glacier_id = "15"

df = pd.read_csv(csv_path, parse_dates=[0])
df.rename(columns={df.columns[0]: "Date"}, inplace=True)

if glacier_id not in df.columns:
    raise ValueError(f"Glacier ID {glacier_id} not found in columns: {df.columns.tolist()}")

df_jan1 = df[df["Date"].dt.strftime("%m-%d") == "01-01"]
base_row = df_jan1[df_jan1["Date"].dt.year == 2011]
base_value = base_row[glacier_id].values[0]
df_jan1["Thickness_Change"] = df_jan1[glacier_id] - base_value



# ======================================================================
# 3. Observed Dussaillant Mass Balance
# ======================================================================
df_dus = pd.read_csv(dussaillant_path)
rgi_id = "RGI60-15.04176"
df_dus = df_dus[df_dus["RGIId"] == rgi_id]
df_dus = df_dus.loc[:, df_dus.columns.str.match(r"^\d{4}$")]
df_dus = df_dus.loc[:, (df_dus.columns.astype(int) >= 2011) & (df_dus.columns.astype(int) <= 2024)]
df_dus = df_dus.apply(pd.to_numeric, errors="coerce") / 1000
df_dus = df_dus.T
df_dus.columns = ["Dussaillant_MB"]
df_dus["Cumulative_MB"] = df_dus["Dussaillant_MB"].cumsum()
df_dus["Cumulative_MB"] -= df_dus["Cumulative_MB"].iloc[0]
df_dus["Year"] = df_dus.index.astype(int)

# ======================================================================
# 4. Plot Dussaillant vs SPHY modeled ice thickness
# ======================================================================
plt.figure(figsize=(10, 4))
plt.plot(df_dus["Year"], df_dus["Cumulative_MB"], marker="o", linestyle="-", color="black", label="Dussaillant")
plt.plot(df_jan1["Date"].dt.year, df_jan1["Thickness_Change"], marker="s", linestyle="-", color="red", label="SPHY")
plt.ylabel("Cumulative Mass Balance (m w.e.)")
plt.xticks(range(2011, 2025), rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figdir, "mass_balance_comparison.png"), dpi=300)
plt.show()

# --- Quantitative comparison with Dussaillant ---
# Align overlapping years
df_model_mb = pd.DataFrame({
    'Year': df_jan1["Date"].dt.year.values,
    'SPHY_MB': df_jan1["Thickness_Change"].values
})

df_compare = pd.merge(
    df_dus[['Year', 'Cumulative_MB']],
    df_model_mb,
    on='Year',
    how='inner'
)

# Final cumulative difference (end of period)
final_diff = df_compare['SPHY_MB'].iloc[-1] - df_compare['Cumulative_MB'].iloc[-1]

# Mean annual bias (difference in cumulative slope)
annual_bias = (df_compare['SPHY_MB'].diff() - df_compare['Cumulative_MB'].diff()).mean()

# Save year-by-year comparison + summary
df_compare_out = df_compare.copy()
df_compare_out.to_csv(os.path.join(figdir, "MB_dussaillant_yearly_comparison.csv"), index=False)

summary_dus = pd.DataFrame([{
    "Simulation": os.path.basename(os.path.normpath(folder_path)),
    "RGIId": rgi_id,
    "Year_start": int(df_compare["Year"].min()),
    "Year_end": int(df_compare["Year"].max()),
    "Final_cumulative_diff_mwe": float(final_diff),
    "Mean_annual_bias_mwe_yr": float(annual_bias),
    "n_years": int(len(df_compare))
}])
summary_dus.to_csv(os.path.join(figdir, "MB_dussaillant_summary.csv"), index=False)

print("\n=== Dussaillant Mass Balance Comparison ===")
print(f"Evaluation period: {df_compare['Year'].min()}–{df_compare['Year'].max()}")
print(f"Final cumulative mass balance difference: {final_diff:.2f} m w.e.")
print(f"Mean annual mass balance bias: {annual_bias:.2f} m w.e. yr⁻¹")

# ======================================================================
# 5. Rounce et al. model comparison (SSP2-4.5)
# ======================================================================
dataset = xr.open_dataset(rounce_path)
data = dataset['glac_mass_annual']
selected = data.where(data['RGIId'] == 'RGI60-15.04176', drop=True)

years_rounce = selected.year.values
sel_idx = np.where((years_rounce >= 2011) & (years_rounce <= 2025))[0]
years_2012_2020 = years_rounce[sel_idx]
data_values = selected.values[:, 0, sel_idx]

perc_change = (data_values / data_values[:, 0][:, np.newaxis]) * 100
mean_change = np.mean(perc_change, axis=0)
std_change = np.std(perc_change, axis=0)

plt.figure(figsize=(8, 5))
# individual models
for i in range(perc_change.shape[0]):
    plt.plot(years_2012_2020, perc_change[i, :], color='black', linestyle=':', linewidth=0.5)
# mean + std envelope
plt.fill_between(years_2012_2020, mean_change - std_change, mean_change + std_change, color='lightgrey', label='±1 std dev')
plt.plot(years_2012_2020, mean_change, color='black', linewidth=2, label='Rounce SSP2-4.5')
plt.plot(years_array, percentages, marker='o', linestyle='-', color='blue', label='SPHY')
plt.xlabel('Year')
plt.ylabel('Relative Ice Mass (%)')
plt.title('SPHY vs Rounce et al. (2023)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figdir, "mass_balance_SPHY_Rounce.png"), dpi=300)
plt.show()
print("✅ Glacier mass balance analysis completed.")
print(f"Figures saved in: {figdir}")

# Save Rounce ensemble summary + SPHY relative mass (aligned on years)
df_rounce = pd.DataFrame({
    "Year": years_2012_2020.astype(int),
    "Rounce_mean_rel_mass_pct": mean_change,
    "Rounce_std_rel_mass_pct": std_change
})

df_sphy = pd.DataFrame({
    "Year": years_array.astype(int),
    "SPHY_rel_mass_pct": percentages
})

df_rounce_compare = pd.merge(df_rounce, df_sphy, on="Year", how="left")
df_rounce_compare.to_csv(os.path.join(figdir, "MB_rounce_ensemble_vs_sphy.csv"), index=False)

# Optional: a single-row summary metric (end-year difference)
end_year = int(df_rounce_compare["Year"].max())
end_row = df_rounce_compare[df_rounce_compare["Year"] == end_year].iloc[0]
summary_rounce = pd.DataFrame([{
    "Simulation": os.path.basename(os.path.normpath(folder_path)),
    "RGIId": "RGI60-15.04176",
    "Year_end": end_year,
    "SPHY_rel_mass_pct_end": float(end_row["SPHY_rel_mass_pct"]) if pd.notna(end_row["SPHY_rel_mass_pct"]) else np.nan,
    "Rounce_mean_rel_mass_pct_end": float(end_row["Rounce_mean_rel_mass_pct"]),
    "Rounce_std_rel_mass_pct_end": float(end_row["Rounce_std_rel_mass_pct"]),
    "Diff_SPHY_minus_RounceMean_pct_end": (
        float(end_row["SPHY_rel_mass_pct"]) - float(end_row["Rounce_mean_rel_mass_pct"])
        if pd.notna(end_row["SPHY_rel_mass_pct"]) else np.nan
    )
}])
summary_rounce.to_csv(os.path.join(figdir, "MB_rounce_summary.csv"), index=False)
# ======================================================================
# 6. Ragettli reference comparison (elevation change 2011–2018)
# ======================================================================

# Reference values from Ragettli et al. (m w.e./yr)
dh_fullglacier_rag = -0.55
dh_debris_rag = -1.30

# Paths for glacier mask and debris cover mask (Langshisha)
glacid_mask_tif = "/scratch/depfg/aubry001/geospatial/ragettliglacier/glacierlangshisha_matchingRagettli.tif"
debris_mask_tif = "/scratch/depfg/aubry001/geospatial/ragettliglacier/langshishadebriscover.tif"

# Choose the simulation folder(s) to evaluate (can loop over several)
sphy_sim_folder = folder_path  # main calibrated run here

file_2011 = os.path.join(sphy_sim_folder, "iceDepth_20111001.map")
file_2018 = os.path.join(sphy_sim_folder, "iceDepth_20181001.map")

# --- Load ice thickness rasters ---
with rasterio.open(file_2011) as src:
    icethickness_2011 = src.read(1)
    transform = src.transform

with rasterio.open(file_2018) as src:
    icethickness_2018 = src.read(1)

# --- Compute elevation change ---
dh = icethickness_2018 - icethickness_2011

# --- Mask: full glacier ---
with rasterio.open(glacid_mask_tif) as src:
    glac_mask = src.read(1)

mask_full = glac_mask == 1
masked_dh_full = np.ma.masked_where(~mask_full, dh)
avg_dh_full = masked_dh_full.mean() / 7.0  # convert to m/yr over 7 years

# --- Mask: debris cover ---
with rasterio.open(debris_mask_tif) as src:
    debris_mask = src.read(1)

mask_debris = debris_mask == 1
masked_dh_debris = np.ma.masked_where(~mask_debris, dh)
avg_dh_debris = masked_dh_debris.mean() / 7.0  # m/yr

# --- Bias vs Ragettli ---
bias_full = avg_dh_full - dh_fullglacier_rag
bias_debris = avg_dh_debris - dh_debris_rag

# --- Store results in DataFrame ---
results_df_ragettli = pd.DataFrame([{
    'Simulation': os.path.basename(os.path.normpath(sphy_sim_folder)),
    'Modelled dh Full (m/yr)': round(avg_dh_full, 2),
    'Modelled dh Debris (m/yr)': round(avg_dh_debris, 2),
    'Ref Ragettli Full (m/yr)': dh_fullglacier_rag,
    'Ref Ragettli Debris (m/yr)': dh_debris_rag,
    'Bias Full (m/yr)': round(bias_full, 2),
    'Bias Debris (m/yr)': round(bias_debris, 2)
}])

print("\n=== Ragettli Mass Balance Comparison ===")
print(results_df_ragettli.to_string(index=False))
results_df_ragettli.to_csv(os.path.join(figdir, "MB_ragettli_comparison.csv"), index=False)