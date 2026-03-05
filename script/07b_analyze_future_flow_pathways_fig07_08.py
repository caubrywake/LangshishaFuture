"""
===============================================================================
Script: 08b — Future Streamflow Pathway Components (Multi-GCM, Multi-SSP)
Author: Dr. Caroline Aubry-Wake
Affiliation: University of Lethbridge
Date: 2025-11-20
Python: 3.9+

Purpose
-------
Analyze routed SPHY streamflow components under future climate scenarios
(SSP1-2.6, SSP3-7.0, SSP5-8.5) for 2025–2098.

This script:
    • Loads daily routed flow components (.tss) for all available GCMs.
    • Computes annual mean discharge (m³ s⁻¹) and annual volume (m³ yr⁻¹).
    • Aggregates ensemble mean ± standard deviation across GCMs.
    • Computes baseflow index (BFI) and fractional contributions.
    • Generates multi-SSP stacked annual pathway plots with imbalanced
      glacier overlays (dashed/dotted).
    • Computes early (2025–2034) vs late (2090–2099) changes and trends.
    • Computes DOY climatologies for early and late periods.
    • Exports summary statistics tables.

-------------------------------------------------------------------------------
Inputs
-------------------------------------------------------------------------------
SPHY future simulation outputs:
    base_dir = "/scratch/depfg/aubry001/out/fut"
        <GCM>/<SSP>/*.tss

Required daily routed components:
    - STotDTS.tss  → Snow Off Glacier
    - GTotDTS.tss  → Snow + Ice On Glacier
    - RTotDTS.tss  → Rain Off Glacier
    - BTotRDTS.tss → GW from Rain
    - BTotSDTS.tss → GW from Snow
    - BTotGDTS.tss → GW from Glacier
    - QAllDTS.tss  → Total discharge

Balanced fraction inputs (for imbalanced overlays):
    "/scratch/depfg/aubry001/fig/futlong_balance/ensemble_balance_<ssp>.csv"

-------------------------------------------------------------------------------
Outputs (saved in fig_dir = "/scratch/depfg/aubry001/fig/futlong_flow")
-------------------------------------------------------------------------------

Ensemble Tables
---------------
1. FlowComponents_ensemble_ssp126.csv
2. FlowComponents_ensemble_ssp370.csv
3. FlowComponents_ensemble_ssp585.csv

   Annual ensemble mean ± std across GCMs for:
       - All routed components (m³ s⁻¹)
       - Annual volumes (m³ yr⁻¹)
       - Baseflow index (BFI_m3yr)
       - Fractional contributions (m3yr basis)

Change Summary Tables
---------------------
4. FlowComponents_change_summary.csv

   Early (2025–2034) vs late (2089–2098) comparison for each component:
       - EarlyMean_m3s
       - LateMean_m3s
       - Absolute and relative change
       - Linear trend (slope, total change, %, R², p-value)

5. FlowComponents_change_summary_with_fractions.csv

   Same as above, plus:
       - EarlyFracOfTotal_% and LateFracOfTotal_%
       - Additional “Streamflow” row (sum of all components)

Figures
-------
6. FlowComponents_allSSP_stacked_fbal.png + .pdf

   Three-panel annual stacked pathway plot (one per SSP)
   showing:
       - GW glacier
       - GW snow
       - GW rain
       - Snow off glacier
       - Snow + Ice on glacier
       - Rain off glacier

   Includes dashed/dotted overlays indicating imbalanced
   glacier groundwater and runoff contributions.

7. FlowComponents_DOY_stacked_Early_Late.png +.pdf

   2×3 panel DOY climatology:
       Top row: 2025–2034
       Bottom row: 2090–2099
       Columns: SSP126, SSP370, SSP585

9. FlowComponents_DOY_stacked_4panel.png +.pdf

   4-panel DOY climatology:
       (a) Multi-SSP early mean (2025–2034)
       (b) SSP126 late (2090–2099)
       (c) SSP370 late (2090–2099)
       (d) SSP585 late (2090–2099)

DOY Statistics Table
---------------------
12. FlowComponents_DOY_component_stats.csv

   For each SSP and period (Early / Late), per component:
       - Mean_discharge_m3s
       - Mean_total_m3s
       - Fraction_of_total_%
       - Peak_DOY
       - Peak_date

===============================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress

# === SETTINGS ===
base_dir = "/scratch/depfg/aubry001/out/fut"
fig_dir = "/scratch/depfg/aubry001/fig/futlong_flow"
os.makedirs(fig_dir, exist_ok=True)

ssps = ["ssp126", "ssp370", "ssp585"]

# SPHY TSS component names (same structure as “current condition” script)
FLOW_FILES = {
    "Snow Off Glacier": "STotDTS.tss",
    "Snow + Ice On Glacier": "GTotDTS.tss",
    "Rain Off Glacier": "RTotDTS.tss",
    "GW from Rain": "BTotRDTS.tss",
    "GW from Snow": "BTotSDTS.tss",
    "GW from Glacier": "BTotGDTS.tss",
    "Total": "QAllDTS.tss"
}

OUTLET_NAME = "Outlet"
START_DATE = pd.to_datetime("2015-01-01")
YEARS = range(2025, 2099)

# Optional: test subset
TEST_SSP = None
TEST_GCM = None
def import_tss(file_path):
    """Robust SPHY .tss reader (skips header and timestep blocks)."""
    rows = []
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Find where numeric table starts — it’s the first line
    # with more than 3 numeric values (day + many columns)
    start_idx = None
    for i, line in enumerate(lines):
        parts = line.strip().split()
        # require multiple numeric values in the row
        if len(parts) > 3 and parts[0].lstrip("-").isdigit():
            start_idx = i
            break

    if start_idx is None:
        raise ValueError(f"Could not find numeric data in {file_path}")

    # Read from that line onwards
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        skiprows=start_idx,
        header=None
    )

    # First column = time step (integer days since model start)
    df.columns = ["day"] + [f"col_{i}" for i in range(1, df.shape[1])]
    start_day = df["day"].iloc[0]
    df["datetime"] = pd.to_datetime("2015-01-01") + pd.to_timedelta(df["day"] - start_day, unit="D")
    df.set_index("datetime", inplace=True)

    return df

# === Loop over SSPs ===
for ssp in ssps:
    if TEST_SSP and ssp != TEST_SSP:
        continue

    print(f"\n=== Processing {ssp.upper()} ===")
    all_gcm_dfs = []

    gcms = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for gcm in gcms:
        if TEST_GCM and gcm != TEST_GCM:
            continue

        gcm_dir = os.path.join(base_dir, gcm, ssp)
        if not os.path.exists(gcm_dir):
            print(f"⚠️ Missing directory {gcm_dir}")
            continue

        comp_data = {}
        for comp, fname in FLOW_FILES.items():
            path = os.path.join(gcm_dir, fname)
            if not os.path.exists(path):
                print(f"⚠️ Missing {fname} for {gcm}, {ssp}")
                continue

            df = import_tss(path)

            if OUTLET_NAME not in df.columns:
                df.rename(columns={df.columns[1]: OUTLET_NAME}, inplace=True)

            # FORCE numeric
            comp_data[comp] = pd.to_numeric(df[OUTLET_NAME], errors="coerce")

        if "Total" not in comp_data or len(comp_data) < 7:
            print(f"⚠️ Incomplete flow data for {gcm}, {ssp}")
            continue

        df_comp = pd.DataFrame(comp_data)
        df_comp = df_comp.loc["2025":"2098"]

        # annual mean discharge (m3/s)
        df_yearly = df_comp.resample("Y").mean()
        df_yearly.index = df_yearly.index.year

        # annual volume (m3/yr) = integrate daily mean Q
        sec_per_day = 86400.0
        df_yearly_vol = df_comp.resample("D").mean().resample("Y").sum() * sec_per_day
        df_yearly_vol.index = df_yearly_vol.index.year
        df_yearly_vol = df_yearly_vol.add_suffix("_m3yr")

        df_out = pd.concat([df_yearly, df_yearly_vol], axis=1)
        df_out.index.name = "Year"

        # BFI + fractions on m3/yr basis
        gw_cols_m3yr = ["GW from Rain_m3yr", "GW from Snow_m3yr", "GW from Glacier_m3yr"]
        df_out["Baseflow_m3yr"] = df_out[gw_cols_m3yr].sum(axis=1)
        df_out["BFI_m3yr"] = df_out["Baseflow_m3yr"] / df_out["Total_m3yr"]

        for c in [c for c in df_out.columns if c.endswith("_m3yr") and c != "Total_m3yr"]:
            df_out[c.replace("_m3yr", "") + "_frac_m3yr"] = df_out[c] / df_out["Total_m3yr"]

        df_out["GCM"] = gcm
        all_gcm_dfs.append(df_out.reset_index())

    if not all_gcm_dfs:
        print(f"⚠️ No valid GCM data for {ssp}")
        continue

    combined = pd.concat(all_gcm_dfs, ignore_index=True)

    # keep only numeric for ensemble stats
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    ensemble = combined.groupby("Year")[numeric_cols].agg(["mean", "std"])
    ensemble.columns = ["_".join(c) for c in ensemble.columns]

    out_csv = os.path.join(fig_dir, f"FlowComponents_ensemble_{ssp}.csv")
    ensemble.to_csv(out_csv, index=True)
    print(f"✅ Saved: {out_csv}")


# === After the SSP loop ===
print("\n✅ All SSPs processed successfully!")

#%%
#1b
# ---------------------------------------------------------
# 1. Multi-SSP figure (3 subplots side-by-side)
# ---------------------------------------------------------
ssp_list = ["ssp126", "ssp370", "ssp585"]
fig, axes = plt.subplots(1, len(ssp_list), figsize=(10, 3), sharey=True)
frac_dir = "/scratch/depfg/aubry001/fig/futlong_balance"
panel_labels = ["(a)", "(b)", "(c)"]  # <-- panel labels

for i, ssp in enumerate(ssp_list):
    ax = axes[i]

    # ---- Read streamflow components ----
    csv_path = os.path.join(fig_dir, f"FlowComponents_ensemble_{ssp}.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️ Missing CSV for {ssp}, skipping")
        continue
    df = pd.read_csv(csv_path)
    df = df.rename(columns=lambda x: x.strip())
    yrs = df["datetime"] if "datetime" in df.columns else df["Year"]

    # ---- Extract means (ensemble means only) ----
    glac_rain = df["GW from Rain_mean"]
    glac_snow = df["GW from Snow_mean"]
    glac_glac = df["GW from Glacier_mean"]
    surf_snowoff = df["Snow Off Glacier_mean"]
    surf_snowice = df["Snow + Ice On Glacier_mean"]
    surf_rainoff = df["Rain Off Glacier_mean"]
    

    # ---- Read balanced fraction for that SSP ----
    fbal_path = os.path.join(frac_dir, f"ensemble_balance_{ssp}.csv")
    if not os.path.exists(fbal_path):
        print(f"⚠️ Missing f_bal CSV for {ssp}, skipping dashed line")
        fbal = None
    else:
        fbal = pd.read_csv(fbal_path)
        # Match column names — ensemble_balance has 'year' and 'frac_balanced_mean'
        fbal = fbal.rename(columns=lambda x: x.strip())
        fbal_years = fbal["year"] if "year" in fbal.columns else fbal.index
        fbal = fbal.set_index(fbal_years)["frac_balanced_mean"]
        # Align with flow years (ensure same index length)
        fbal = fbal.reindex(yrs, method="nearest")  # nearest match per year

    # ---- Stacked area for flow components ----
    ax.stackplot(
        yrs,
        glac_glac, glac_snow, glac_rain,
        surf_snowoff, surf_snowice, surf_rainoff,
        labels=[
            "GW glacier", "GW snow", "GW rain",
            "Snow off glac.", "Snow+Ice on glac.", "Rain off glac."
        ],
        colors=["#83c1ff", "#707070", "#004992", "#ffba7e", "#ffdcb0", "#ff7e0e"],
        alpha=0.9
    )

        # ---- Panel label in upper-left ----
    ax.text(
        0.02, 0.95,
        panel_labels[i],
        transform=ax.transAxes,
        fontsize=11,
        va="top", ha="left"
    )

    # ---- X-limits: exactly first to last year, no extra space ----
    ax.set_xlim(yrs.iloc[0], yrs.iloc[-1])
    ax.margins(x=0)   # remove extra padding on x-axis

        # ---- Overlay wastage on glacier GW + glacier melt runoff (dashed / dotted) ----
    # ---- Overlay wastage on glacier GW + glacier melt runoff (dashed / dotted) ----
    if fbal is not None:
        frac_wastage = 1 - fbal.values   # or fbal.values if that's already "wastage"

        # components as Series/arrays
        gw_glac   = glac_glac
        runoff_gl = surf_snowice

        # wastage magnitudes
        gw_wastage_mag     = gw_glac * frac_wastage
        runoff_wastage_mag = runoff_gl * frac_wastage

        # --- positions inside stacked plot ---
        # GW from Glacier is the bottom layer, so baseline = 0
        y_gw_wastage = gw_wastage_mag

        # Snow+Ice on Glacier sits on top of:
        # GW glacier + GW snow + GW rain + Snow off glacier
        base_snowice = gw_glac + glac_snow + glac_rain + surf_snowoff
        y_runoff_wastage = base_snowice + runoff_wastage_mag

        if i == 0:
            ax.plot(yrs, y_gw_wastage, "k--", lw=1, label="Imbalanced GW (glacier)")
            ax.plot(yrs, y_runoff_wastage, "k:",  lw=1, label="Imbalanced runoff (glacier)")
        else:
            ax.plot(yrs, y_gw_wastage, "k--", lw=1)
            ax.plot(yrs, y_runoff_wastage, "k:",  lw=1)
            
# ---- Common legend ----
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.03),
    frameon=False
)

#plt.suptitle("Future Streamflow Components (ensemble mean, 2025–2098)", fontsize=14, weight="bold")
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig(os.path.join(fig_dir, "FlowComponents_allSSP_stacked_fbal.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(fig_dir, "FlowComponents_allSSP_stacked_fbal.pdf"), dpi=300, bbox_inches="tight")
plt.show()


#%%

# ---------------------------------------------------------
# 2. Change summary table (first 10 yrs vs last 10 yrs + trend)
# ---------------------------------------------------------


summary_rows = []
components = [
    "GW from Glacier", "GW from Snow", "GW from Rain",
    "Snow Off Glacier", "Snow + Ice On Glacier", "Rain Off Glacier"
]

for ssp in ssp_list:
    csv_path = os.path.join(fig_dir, f"FlowComponents_ensemble_{ssp}.csv")
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)
    df = df.rename(columns=lambda x: x.strip())
    yrs = df["datetime"] if "datetime" in df.columns else df["Year"]

    for comp in components:
        mean_col = f"{comp}_mean"
        if mean_col not in df.columns:
            continue

        early_mask = (yrs >= 2025) & (yrs <= 2034)
        late_mask  = (yrs >= 2089) & (yrs <= 2098)

        early_mean = df.loc[early_mask, mean_col].mean()
        early_std  = df.loc[early_mask, mean_col].std()
        late_mean  = df.loc[late_mask, mean_col].mean()
        late_std   = df.loc[late_mask, mean_col].std()

        abs_change = late_mean - early_mean
        rel_change = (abs_change / early_mean * 100) if early_mean != 0 else np.nan

        slope, intercept, r_value, p_value, stderr = linregress(yrs, df[mean_col])
        trend_total = slope * (yrs.max() - yrs.min())
        trend_percent = (trend_total / early_mean * 100) if early_mean != 0 else np.nan

        summary_rows.append({
            "SSP": ssp,
            "Component": comp,
            "EarlyMean_m3s": round(early_mean, 2),
            "LateMean_m3s": round(late_mean, 2),
            "AbsChange_m3s": round(abs_change, 2),
            "RelChange_%": round(rel_change, 1),
            "TrendSlope_m3s_per_yr": round(slope, 4),
            "TrendTotal_m3s": round(trend_total, 2),
            "TrendTotal_%": round(trend_percent, 1),
            "R2": round(r_value**2, 2),
            "p_value": round(p_value, 4)
        })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values(["SSP", "Component"])
out_summary = os.path.join(fig_dir, "FlowComponents_change_summary.csv")
summary_df.to_csv(out_summary, index=False)
print(f"\n✅ Saved summary table: {out_summary}")

# Nicely print a Markdown-style table
def markdown_table(df):
    headers = " | ".join(df.columns)
    sep = " | ".join(["---"] * len(df.columns))
    rows = [" | ".join(map(str, row)) for row in df.values]
    return f"{headers}\n{sep}\n" + "\n".join(rows)

print("\n=== Flow Component Change Summary ===")
print(markdown_table(summary_df))

#%% 
# ---------------------------------------------------------
# 2. Change summary table (first 10 yrs vs last 10 yrs + trend)
#    Now also includes:
#       - early/late fraction of total streamflow for each component
#       - an extra "Streamflow" row (sum of all components)
# ---------------------------------------------------------
from scipy.stats import linregress

summary_rows = []

components = [
    "GW from Glacier", "GW from Snow", "GW from Rain",
    "Snow Off Glacier", "Snow + Ice On Glacier", "Rain Off Glacier"
]

for ssp in ssp_list:
    csv_path = os.path.join(fig_dir, f"FlowComponents_ensemble_{ssp}.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️ Missing {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    df = df.rename(columns=lambda x: x.strip())

    yrs = df["datetime"] if "datetime" in df.columns else df["Year"]

    # build list of mean-column names for components that exist
    mean_cols = []
    for comp in components:
        mean_col = f"{comp}_mean"
        if mean_col in df.columns:
            mean_cols.append(mean_col)
        else:
            print(f"⚠️ {mean_col} not found in {csv_path}, skipping for totals.")

    # total streamflow as sum of all component means
    df["TotalStreamflow"] = df[mean_cols].sum(axis=1)

    # masks for early / late decades
    early_mask = (yrs >= 2025) & (yrs <= 2034)
    late_mask  = (yrs >= 2089) & (yrs <= 2098)

    # early / late mean of total streamflow
    early_total_mean = df.loc[early_mask, "TotalStreamflow"].mean()
    late_total_mean  = df.loc[late_mask,  "TotalStreamflow"].mean()

    # ---------- component rows ----------
    for comp in components:
        mean_col = f"{comp}_mean"
        if mean_col not in df.columns:
            continue

        early_mean = df.loc[early_mask, mean_col].mean()
        late_mean  = df.loc[late_mask,  mean_col].mean()

        abs_change = late_mean - early_mean
        rel_change = (abs_change / early_mean * 100.0) if early_mean != 0 else np.nan

        # fraction of total streamflow in early / late periods
        early_frac = (early_mean / early_total_mean * 100.0) if early_total_mean > 0 else np.nan
        late_frac  = (late_mean  / late_total_mean  * 100.0) if late_total_mean  > 0 else np.nan

        # linear trend over full period
        slope, intercept, r_value, p_value, stderr = linregress(yrs, df[mean_col])
        trend_total   = slope * (yrs.max() - yrs.min())
        trend_percent = (trend_total / early_mean * 100.0) if early_mean != 0 else np.nan

        summary_rows.append({
            "SSP": ssp,
            "Component": comp,
            "EarlyMean_m3s": round(early_mean, 3),
            "LateMean_m3s": round(late_mean, 3),
            "AbsChange_m3s": round(abs_change, 3),
            "RelChange_%": round(rel_change, 1),
            "EarlyFracOfTotal_%": round(early_frac, 1),
            "LateFracOfTotal_%": round(late_frac, 1),
            "TrendSlope_m3s_per_yr": round(slope, 4),
            "TrendTotal_m3s": round(trend_total, 3),
            "TrendTotal_%": round(trend_percent, 1),
            "R2": round(r_value**2, 2),
            "p_value": round(p_value, 4),
        })

    # ---------- total streamflow row ----------
    slope_tot, intercept_tot, r_tot, p_tot, stderr_tot = linregress(yrs, df["TotalStreamflow"])
    trend_tot_total   = slope_tot * (yrs.max() - yrs.min())
    trend_tot_percent = (trend_tot_total / early_total_mean * 100.0) if early_total_mean != 0 else np.nan

    summary_rows.append({
        "SSP": ssp,
        "Component": "Streamflow",
        "EarlyMean_m3s": round(early_total_mean, 3),
        "LateMean_m3s": round(late_total_mean, 3),
        "AbsChange_m3s": round(late_total_mean - early_total_mean, 3),
        "RelChange_%": round((late_total_mean - early_total_mean) / early_total_mean * 100.0, 1)
                       if early_total_mean != 0 else np.nan,
        "EarlyFracOfTotal_%": 100.0,
        "LateFracOfTotal_%": 100.0,
        "TrendSlope_m3s_per_yr": round(slope_tot, 4),
        "TrendTotal_m3s": round(trend_tot_total, 3),
        "TrendTotal_%": round(trend_tot_percent, 1),
        "R2": round(r_tot**2, 2),
        "p_value": round(p_tot, 4),
    })

# -------- save + print --------
summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values(["SSP", "Component"])

out_summary = os.path.join(fig_dir, "FlowComponents_change_summary_with_fractions.csv")
summary_df.to_csv(out_summary, index=False)
print(f"\n✅ Saved summary table: {out_summary}")


def markdown_table(df):
    headers = " | ".join(df.columns)
    sep = " | ".join(["---"] * len(df.columns))
    rows = [" | ".join(map(str, row)) for row in df.values]
    return f"{headers}\n{sep}\n" + "\n".join(rows)

print("\n=== Flow Component Change Summary (with % of streamflow) ===")
print(markdown_table(summary_df))


#%% DOY cklimatology ealry and late

# --------- Settings ---------
ssp_list = ["ssp126", "ssp370", "ssp585"]

FLOW_FILES = {
    "Snow Off Glacier": "STotDTS.tss",
    "Snow + Ice On Glacier": "GTotDTS.tss",
    "Rain Off Glacier": "RTotDTS.tss",
    "GW from Rain": "BTotRDTS.tss",
    "GW from Snow": "BTotSDTS.tss",
    "GW from Glacier": "BTotGDTS.tss",
    "Total": "QAllDTS.tss",
}

components_to_plot = [
    "GW from Glacier",
    "GW from Snow",
    "GW from Rain",
    "Snow Off Glacier",
    "Snow + Ice On Glacier",
    "Rain Off Glacier",
]

colors = {
    "GW from Glacier": "#83c1ff",
    "GW from Snow": "#707070",
    "GW from Rain": "#004992",
    "Snow Off Glacier": "#ffba7e",
    "Snow + Ice On Glacier": "#ffdcb0",
    "Rain Off Glacier": "#ff7e0e",
}

early_range = (2025, 2034)
late_range  = (2090, 2099)

# --------- Helper: load ensemble daily components for one SSP ---------
def load_ensemble_daily_components(ssp):
    """
    For a given SSP, load all GCMs' daily TSS for each flow component,
    align them on a common datetime index, and return:
      df_mean: DataFrame with columns = components (ensemble mean)
      df_std : DataFrame with columns = components (ensemble std; optional use)
    Index = datetime (daily).
    """
    # discover GCM directories
    gcm_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not gcm_dirs:
        raise RuntimeError(f"No GCM directories found in {base_dir}")

    # dict: comp -> list of Series (one per GCM)
    comp_series = {comp: [] for comp in FLOW_FILES.keys()}

    for gcm in gcm_dirs:
        ssp_dir = os.path.join(base_dir, gcm, ssp)
        if not os.path.isdir(ssp_dir):
            # allow some GCMs to lack a given SSP
            continue

        for comp, fname in FLOW_FILES.items():
            tss_path = os.path.join(ssp_dir, fname)
            if not os.path.exists(tss_path):
                print(f"⚠️ Missing {tss_path}, skipping ({gcm}, {ssp}, {comp})")
                continue

            df_comp = import_tss(tss_path)  # index=datetime, col_1
            s = df_comp["col_1"].copy()
            s.name = f"{gcm}_{comp}"
            comp_series[comp].append(s)

    # build ensemble mean/std per component
    df_mean = pd.DataFrame()
    df_std  = pd.DataFrame()

    for comp, series_list in comp_series.items():
        if not series_list:
            print(f"⚠️ No data for component '{comp}' in {ssp}")
            continue
        df_comp_all = pd.concat(series_list, axis=1)  # align on datetime index
        df_mean[comp] = df_comp_all.mean(axis=1)
        df_std[comp]  = df_comp_all.std(axis=1)

    # ensure datetime index and sort
    df_mean = df_mean.sort_index()
    df_std  = df_std.reindex(df_mean.index)

    return df_mean, df_std

# --------- Helper: DOY climatology for early/late decades ---------
def doy_climatology(df_daily, year_min, year_max):
    """
    df_daily: DataFrame with daily data, index=datetime, columns=components.
    Returns DOY climatology (mean over years in [year_min, year_max]) and DOY index.
    """
    df = df_daily.copy()
    df["Year"] = df.index.year
    df["DOY"]  = df.index.dayofyear

    # restrict to requested years
    mask = (df["Year"] >= year_min) & (df["Year"] <= year_max)
    df = df.loc[mask]

    # drop Feb 29 to avoid uneven DOY counts
    df = df[df["DOY"] <= 365]

    # group by DOY and average
    grouped = df.groupby("DOY")
    doy_mean = grouped[components_to_plot].mean()

    # ensure full 1..365
    doy_mean = doy_mean.reindex(range(1, 366))
    return doy_mean

# --------- Build DOY climatologies and plot ---------
fig, axes = plt.subplots(
    2, len(ssp_list),
    figsize=(12, 6),
    sharex=True,
    sharey=True
)

top_labels = ['(a)', '(b)', '(c)']
bot_labels = ['(d)', '(e)', '(f)']

for col, ssp in enumerate(ssp_list):
    print(f"Processing DOY climatology for {ssp}...")
    df_mean, df_std = load_ensemble_daily_components(ssp)

    # early and late DOY climatologies
    doy_early = doy_climatology(df_mean, *early_range)
    doy_late  = doy_climatology(df_mean, *late_range)

    # --- plot early ---
    ax_top = axes[0, col]
    ax_top.stackplot(
        doy_early.index,
        [doy_early[c] for c in components_to_plot],
        colors=[colors[c] for c in components_to_plot],
        alpha=0.9
    )
    if col == 0:
        ax_top.set_ylabel("Discharge (m³ s⁻¹)")
    ax_top.text(
        0.01, 0.95, top_labels[col],
        transform=ax_top.transAxes,
        fontsize=12, va='top', ha='left'
    )

    # --- plot late ---
    ax_bottom = axes[1, col]
    ax_bottom.stackplot(
        doy_late.index,
        [doy_late[c] for c in components_to_plot],
        colors=[colors[c] for c in components_to_plot],
        alpha=0.9
    )
    if col == 0:
        ax_bottom.set_ylabel("Discharge (m³ s⁻¹)")
    ax_bottom.text(
        0.01, 0.95, bot_labels[col],
        transform=ax_bottom.transAxes,
        fontsize=12, va='top', ha='left'
    )

    # x-axis formatting
    ax_bottom.set_xlabel("Day of Year")
    ax_bottom.set_xlim(1, 365)
    ax_bottom.set_xticks([1, 60, 121, 182, 244, 305, 365])
    ax_bottom.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov", "Dec"])

    for ax in [ax_top, ax_bottom]:
        ax.grid(alpha=0.3)

# --------- Common legend ---------
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in components_to_plot]
labels  = components_to_plot
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.05),
    frameon=False
)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
out_png = os.path.join(fig_dir, "FlowComponents_DOY_stacked_Early_Late.png")
out_pdf = os.path.join(fig_dir, "FlowComponents_DOY_stacked_Early_Late.pdf")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.show()

print("Saved DOY figure to:")
print("  ", out_png)
print("  ", out_pdf)

# --------- Build DOY climatologies and plot (4 panels) ---------

# store DOY climatologies for each SSP
doy_store = {ssp: {"Early": None, "Late": None} for ssp in ssp_list}

for ssp in ssp_list:
    print(f"Processing DOY climatology for {ssp}...")
    df_mean, df_std = load_ensemble_daily_components(ssp)

    # early and late DOY climatologies
    doy_early = doy_climatology(df_mean, *early_range)
    doy_late  = doy_climatology(df_mean, *late_range)

    doy_store[ssp]["Early"] = doy_early
    doy_store[ssp]["Late"]  = doy_late

# multi-SSP mean early climatology
early_mean = sum(doy_store[ssp]["Early"] for ssp in ssp_list) / len(ssp_list)

# for consistent y-limits across panels
all_clims = [early_mean] + [doy_store[ssp]["Late"] for ssp in ssp_list]
ymax = max(clim[components_to_plot].sum(axis=1).max() for clim in all_clims) * 1.05

# 4 rows x 1 column
fig, axes = plt.subplots(
    4, 1,
    figsize=(3,8),
    sharex=True,
    sharey=True
)

panel_info = [
    ("(a) 2025–2034", early_mean),
    ("(b) SSP126, 2090–2099",          doy_store["ssp126"]["Late"]),
    ("(c) SSP370, 2090–2099",          doy_store["ssp370"]["Late"]),
    ("(d) SSP585, 2090–2099",          doy_store["ssp585"]["Late"]),
]

for ax, (label, clim) in zip(axes, panel_info):
    ax.stackplot(
        clim.index,
        [clim[c] for c in components_to_plot],
        colors=[colors[c] for c in components_to_plot],
        alpha=0.9
    )
    ax.text(
        0.01, 0.95, label,
        transform=ax.transAxes,
        fontsize=10, va='top', ha='left'
    )
    ax.set_xlim(1, 365)
    ax.set_ylim(0, ymax)
    ax.grid(alpha=0.3)

# x-axis formatting on bottom panel
axes[-1].set_xlabel("Day of Year")
axes[-1].set_xticks([1, 60, 121, 182, 244, 305, 365])
axes[-1].set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov", "Dec"])

# y-labels
for ax in axes:
    ax.set_ylabel("Discharge (m³ s⁻¹)")

# --------- Common legend ---------
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in components_to_plot]
labels  = components_to_plot
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.02),
    frameon=False
)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
out_png = os.path.join(fig_dir, "FlowComponents_DOY_stacked_4panel.png")
out_pdf = os.path.join(fig_dir, "FlowComponents_DOY_stacked_4panel.pdf")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.show()

print("Saved DOY figure to:")
print("  ", out_png)
print("  ", out_pdf)
# %%
# ==========================================
# Component statistics (totals, %, peak DOY)
# ==========================================

stats_rows = []

def compute_period_stats(df_daily, year_min, year_max, ssp, period_label):
    """
    df_daily: ensemble-mean daily components, index=datetime
    Computes mean discharge and fraction of total for each component,
    plus the DOY and date of peak contribution, over the given year range.
    """
    df = df_daily.copy()
    df["Year"] = df.index.year
    df["DOY"]  = df.index.dayofyear

    # restrict to requested years and drop Feb 29
    mask = (df["Year"] >= year_min) & (df["Year"] <= year_max) & (df["DOY"] <= 365)
    df = df.loc[mask]

    # mean discharge over the period (m³/s)
    mean_total = df["Total"].mean()

    for comp in components_to_plot:
        if comp not in df.columns:
            continue

        mean_comp = df[comp].mean()
        frac_pct  = 100.0 * mean_comp / mean_total if mean_total > 0 else np.nan

        # peak day and date within the period
        idx_peak  = df[comp].idxmax()
        doy_peak  = int(df.loc[idx_peak, "DOY"])
        date_peak = idx_peak.date().isoformat()

        stats_rows.append({
            "SSP": ssp,
            "Period": period_label,          # "Early" or "Late"
            "Year_min": year_min,
            "Year_max": year_max,
            "Component": comp,
            "Mean_discharge_m3s": mean_comp,
            "Mean_total_m3s": mean_total,
            "Fraction_of_total_%": frac_pct,
            "Peak_DOY": doy_peak,
            "Peak_date": date_peak,
        })

# --- loop over SSPs and periods using the same ensemble loader ---
for ssp in ssp_list:
    print(f"\nComputing component stats for {ssp}...")
    df_mean, df_std = load_ensemble_daily_components(ssp)

    compute_period_stats(df_mean, early_range[0], early_range[1], ssp, "Early")
    compute_period_stats(df_mean, late_range[0],  late_range[1],  ssp, "Late")

# --- build table and save to CSV ---
stats_df = pd.DataFrame(stats_rows)

out_stats_csv = os.path.join(fig_dir, "FlowComponents_DOY_component_stats.csv")
stats_df.to_csv(out_stats_csv, index=False)
print(f"\nSaved component stats to:\n  {out_stats_csv}")

# --- print a compact console summary (rounded) ---
print("\n=== Mean discharge and % of total by component ===")
for ssp in ssp_list:
    for period in ["Early", "Late"]:
        sub = stats_df[(stats_df["SSP"] == ssp) & (stats_df["Period"] == period)].copy()
        if sub.empty:
            continue
        print(f"\n-- {ssp.upper()} | {period} ({int(sub['Year_min'].iloc[0])}-{int(sub['Year_max'].iloc[0])}) --")
        print(
            sub[["Component", "Mean_discharge_m3s", "Fraction_of_total_%", "Peak_DOY", "Peak_date"]]
            .sort_values("Fraction_of_total_%", ascending=False)
            .round({"Mean_discharge_m3s": 3, "Fraction_of_total_%": 1})
            .to_string(index=False)
        )

