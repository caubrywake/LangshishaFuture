"""
===============================================================================
Script Name: glacier_balance_ensemble.py
Author: Caroline Aubry-Wake
Affiliation: University of Lethbridge
Date: 2025-11-07
Python Version: 3.10
Dependencies: numpy, pandas, matplotlib, scipy, glob, os

-------------------------------------------------------------------------------
Purpose:
    Quantify and visualize the partitioning of glacier melt into
    (1) balanced melt — supported by annual snowfall accumulation, and
    (2) imbalanced melt — resulting from long-term ice volume loss,
    across multiple GCMs and SSP scenarios using SPHY model outputs.

-------------------------------------------------------------------------------
Method Summary:
    • The glacier melt flux (GlacMelt.csv) represents total glacier melt
      (balanced + imbalanced components).
    • Annual change in glacier ice volume (ΔV) is computed from the glacTable
      outputs (ICE_DEPTH × FRAC_GLAC × pixel area, per GLAC_ID).
    • Imbalanced melt corresponds to the ice volume loss (−ΔV, where ΔV < 0).
    • Balanced melt = total melt (GlacMelt) − imbalanced melt.
    • All melt fluxes are normalized by total glacierized area and expressed
      in mm yr⁻¹.
    • Pixels with ICE_DEPTH < 1 m or FRAC_GLAC < 0.05 are excluded to avoid
      spurious edge effects.
    • Ensemble statistics (mean ± 1σ) are calculated across all GCMs for each
      SSP scenario.
    • Trends in the balanced melt fraction are derived using linear regression.

-------------------------------------------------------------------------------
Input Requirements:
    Base directory (BASE_DIR) must contain SPHY model outputs in the structure:
        /scratch/.../out/fut/{GCM}/{SSP}/
            ├── GlacMelt.csv
            ├── glacTable_YYYYMMDD.csv (annual)
            └── other standard SPHY outputs (not directly used)

    Expected file contents:
        • GlacMelt.csv: daily melt flux [mm day⁻¹] for each GLAC_ID.
        • glacTable_*.csv: per-pixel fractional area (FRAC_GLAC) and
          ice thickness (ICE_DEPTH) for each GLAC_ID.

-------------------------------------------------------------------------------
Outputs:
    1. Per-SSP ensemble CSVs in FIG_DIR:
        - ensemble_balance_ssp126.csv
        - ensemble_balance_ssp370.csv
        - ensemble_balance_ssp585.csv
       Each file contains mean and standard deviation of:
         balanced_mm, imbalanced_mm, total_mm, frac_balanced

    2. Per-SSP figures (3 subplots):
        (a) Balanced & imbalanced melt time series (±1σ)
        (b) Stacked annual melt composition
        (c) Balanced fraction with regression trend

    3. Cross-SSP summary figure (4 subplots):
        (a–c) Stacked balanced vs imbalanced melt for each SSP
        (d) Balanced fraction curves for all SSPs combined

-------------------------------------------------------------------------------
User Parameters:
    BASE_DIR      – Path to GCM/SSP output structure.
    PIXEL_AREA    – Pixel area in m² (default 400 for 20×20 m resolution).
    GLACIER_IDS   – List of glacier IDs to include in analysis.
    START_YEAR,
    END_YEAR      – Time window for analysis (e.g., 2025–2099).
    FIG_DIR       – Output directory for figures and CSV files.


----------------

"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import warnings
import matplotlib.colors as mcolors
from scipy.stats import theilslopes


warnings.filterwarnings("ignore", category=DeprecationWarning)

# === USER SETTINGS ===
BASE_DIR = "/scratch/depfg/aubry001/out/fut"
PIXEL_AREA = 20 * 20  # m²
GLACIER_IDS = [11, 12, 13, 3, 2, 7, 8, 4, 24, 23, 21, 22, 20, 19, 16, 26, 18, 15] # glacier that drain within the basin
START_YEAR, END_YEAR = 2025, 2099   
FIG_DIR = "/scratch/depfg/aubry001/fig/futlong_balance"
os.makedirs(FIG_DIR, exist_ok=True)

# === HELPER: filter and compute glacier area + volume per year ===
def compute_area_volume(gt):
    mask = (gt["ICE_DEPTH"] >= 1.0) & (gt["FRAC_GLAC"] >= 0.05)
    gt = gt[mask]
    if gt.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    area_g = gt.groupby("GLAC_ID")["FRAC_GLAC"].sum() * PIXEL_AREA
    vol_g = gt.groupby("GLAC_ID", group_keys=False).apply(
        lambda df: np.nansum(df["ICE_DEPTH"] * df["FRAC_GLAC"] * PIXEL_AREA)
    )
    return area_g, vol_g


# === HELPER: process one GCM ===
def process_gcm(gcm_dir):
    print(f"Processing {os.path.basename(gcm_dir)}")

    melt_path = os.path.join(gcm_dir, "GlacMelt.csv")
    if not os.path.exists(melt_path):
        print(f"⚠ Missing GlacMelt.csv in {gcm_dir}")
        return None

    glacmelt = pd.read_csv(melt_path, index_col=0, parse_dates=True)
    glacmelt = glacmelt[[str(i) for i in GLACIER_IDS if str(i) in glacmelt.columns]]
    if glacmelt.empty:
        print(f"⚠ No matching glacier IDs in {gcm_dir}")
        return None

    years = sorted(set(glacmelt.index.year))
    results = []

    for y in years:
        if not (START_YEAR <= y <= END_YEAR):
            continue

        gt_files = sorted(glob.glob(os.path.join(gcm_dir, f"glacTable_{y}*.csv")))
        gt_prev_files = sorted(glob.glob(os.path.join(gcm_dir, f"glacTable_{y-1}*.csv")))
        if not gt_files or not gt_prev_files:
            continue

        gt_now = pd.read_csv(gt_files[0])
        gt_prev = pd.read_csv(gt_prev_files[0])
        gt_now = gt_now[gt_now["GLAC_ID"].isin(GLACIER_IDS)]
        gt_prev = gt_prev[gt_prev["GLAC_ID"].isin(GLACIER_IDS)]

        area_now, vol_now = compute_area_volume(gt_now)
        area_prev, vol_prev = compute_area_volume(gt_prev)

        if area_now.empty or vol_now.empty or area_prev.empty or vol_prev.empty:
            continue

        A_glac = area_now.sum()
        dV = vol_now.sum() - vol_prev.sum()  # m³ change per year

        # convert GlacMelt mm/day → m³/day and sum to annual m³
        valid_cols = [c for c in glacmelt.columns if c in area_now.index.astype(str)]
        area_series = pd.Series({str(int(k)): v for k, v in area_now.items()})
        glac_yr = (glacmelt.loc[str(y), valid_cols] / 1000 * area_series).sum(axis=1)
        glac_m3 = glac_yr.sum()

        # Partition melt
        imbalanced_m3 = max(0, -dV)        # long-term ice loss
        balanced_m3   = max(0, glac_m3 - imbalanced_m3)

        # Convert to mm/yr over glacier area
        balanced_mm   = balanced_m3   / A_glac * 1000
        imbalanced_mm = imbalanced_m3 / A_glac * 1000
        total_mm      = balanced_mm + imbalanced_mm
        frac_bal      = balanced_mm / total_mm if total_mm > 0 else np.nan

        results.append({
            "year": y,

            # absolute volumes (m³/yr)
            "balanced_m3": balanced_m3,
            "imbalanced_m3": imbalanced_m3,
            "total_m3": glac_m3,

            # area-normalized rates (mm/yr over current-year glacier area)
            "balanced_mm": balanced_mm,
            "imbalanced_mm": imbalanced_mm,
            "total_mm": total_mm,
            "frac_balanced": frac_bal,

            # optional diagnostics
            "A_glac_m2": A_glac,
            "dV_m3": dV,
        })

    if not results:
        return None

    df = pd.DataFrame(results).set_index("year")
    df["gcm"] = os.path.basename(gcm_dir)
    return df


# === DETECT AVAILABLE SSPs ===
example_gcm = next(
    (g for g in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, g))), None
)
if example_gcm is None:
    raise RuntimeError("No GCM folders found in BASE_DIR")

ssp_list = [
    ssp for ssp in os.listdir(os.path.join(BASE_DIR, example_gcm))
    if os.path.isdir(os.path.join(BASE_DIR, example_gcm, ssp))
]

print(f"🧭 SSPs found: {ssp_list}")

# === MAIN LOOP OVER SSPs ===
for SSP in ssp_list:
    print(f"\n🚀 Processing ensemble for {SSP.upper()}")

    gcm_dirs = [
        os.path.join(BASE_DIR, g, SSP)
        for g in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, g, SSP))
    ]
    if not gcm_dirs:
        print(f"⚠️ No GCM directories found for {SSP}")
        continue

    gcm_results = []
    for gcm_dir in gcm_dirs:
        res = process_gcm(gcm_dir)
        if res is not None:
            gcm_results.append(res.assign(gcm=os.path.basename(os.path.dirname(gcm_dir))))

    if not gcm_results:
        print(f"⚠️ No valid outputs for {SSP}")
        continue

    df_all = pd.concat(gcm_results)
    df_all.index.name = "year"
    df_all = df_all.reset_index()
    df_all = df_all[(df_all["year"] >= START_YEAR) & (df_all["year"] <= END_YEAR)]

    # === ENSEMBLE MEAN ± STD ===
    numeric_cols = df_all.select_dtypes(include=[np.number]).copy()

    # Make sure 'year' exists for grouping
    if "year" not in numeric_cols.columns:
        numeric_cols["year"] = df_all["year"]

    ensemble = (
        numeric_cols
        .groupby("year", as_index=True)     # ✅ group by actual year
        .agg(["mean", "std"])
    )
    ensemble.columns = ["_".join(col) for col in ensemble.columns]
    ensemble = ensemble.sort_index()        # optional: ensure chronological order

    csv_path = os.path.join(FIG_DIR, f"ensemble_balance_{SSP}.csv")
    ensemble.to_csv(csv_path)
    print(f"✅ Saved ensemble CSV: {csv_path}")

    ensemble = (
            numeric_cols
            .groupby("year", as_index=True)     # ✅ group by actual year
            .agg(["mean", "std"])
        )
    ensemble.columns = ["_".join(col) for col in ensemble.columns]
    ensemble = ensemble.sort_index()        # optional: ensure chronological order

    csv_path = os.path.join(FIG_DIR, f"ensemble_balance_{SSP}.csv")
    ensemble.to_csv(csv_path)
    print(f"✅ Saved ensemble CSV: {csv_path}")

     # ===============================================================
    # EXPORT SUMMARY TABLE FOR THIS SSP
    # ===============================================================
    summary_rows = []

    # Define "early" (first 10 yrs) and "late" (last 20 yrs) windows
    early_mask = (ensemble.index >= START_YEAR) & (ensemble.index <= START_YEAR + 9)
    late_mask  = (ensemble.index >= END_YEAR - 9) & (ensemble.index <= END_YEAR)

    # Metrics to summarize
    metrics = ["balanced_mm_mean", "imbalanced_mm_mean", "frac_balanced_mean", "balanced_m3_mean", "imbalanced_m3_mean", ]

    for m in metrics:
        # Means
        early_mean = ensemble.loc[early_mask, m].mean()
        late_mean  = ensemble.loc[late_mask, m].mean()

        # Standard deviations
        early_std = ensemble.loc[early_mask, m].std()
        late_std  = ensemble.loc[late_mask, m].std()

        # Changes
        abs_change = late_mean - early_mean
        rel_change = (abs_change / early_mean * 100) if early_mean != 0 else np.nan

        # Trend over full analysis period
        slope, intercept, r, p, stderr = linregress(ensemble.index, ensemble[m])
        trend_total = slope * (END_YEAR - START_YEAR)
        trend_total_pct = (trend_total / early_mean * 100) if early_mean != 0 else np.nan

        summary_rows.append({
            "SSP": SSP,
            "Metric": m,
            "EarlyMean": early_mean,
            "EarlyStd": early_std,
            "LateMean": late_mean,
            "LateStd": late_std,
            "AbsChange": abs_change,
            "RelChange_%": rel_change,
            "TrendSlope_per_yr": slope,
            "TrendTotal": trend_total,
            "TrendTotal_%": trend_total_pct,
            "R2": r**2,
            "p_value": p,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(FIG_DIR, f"BalanceSummary_{SSP}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"📄 Saved summary metrics: {summary_path}")

    # === TREND ===
    slope, intercept, r, p, stderr = linregress(
        ensemble.index, ensemble["frac_balanced_mean"]
    )

    # === PLOT ===
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)
    years = ensemble.index
    colors = {"balanced": "#4daf4a", "imbalanced": "#e41a1c"}

    # (a)
    axes[0].plot(years, ensemble["balanced_m3_mean"], label="Balanced", color=colors["balanced"])
    axes[0].fill_between(
        years,
        ensemble["balanced_m3_mean"] - ensemble["balanced_m3_std"],
        ensemble["balanced_m3_mean"] + ensemble["balanced_m3_std"],
        color=colors["balanced"], alpha=0.2
    )
    axes[0].plot(years, ensemble["imbalanced_m3_mean"], label="Imbalanced", color=colors["imbalanced"])
    axes[0].fill_between(
        years,
        ensemble["imbalanced_m3_mean"] - ensemble["imbalanced_m3_std"],
        ensemble["imbalanced_m3_mean"] + ensemble["imbalanced_m3_std"],
        color=colors["imbalanced"], alpha=0.2
    )
    axes[0].set_ylabel("Melt (mm yr⁻¹)")
    axes[0].legend()
    axes[0].set_title(f"(a) Glacier melt components ({SSP.upper()})")

    # (b)
    axes[1].bar(years, ensemble["balanced_m3_mean"], color=colors["balanced"], label="Balanced")
    axes[1].bar(years, ensemble["imbalanced_m3_mean"],
                bottom=ensemble["balanced_m3_mean"],
                color=colors["imbalanced"], label="Imbalanced")
    axes[1].set_ylabel("Melt (m^3 yr⁻¹)")
    axes[1].legend()
    axes[1].set_title("(b) Annual balanced vs imbalanced melt")

    # (c)
    axes[2].plot(years, ensemble["frac_balanced_mean"], color="k", lw=2)
    axes[2].fill_between(
        years,
        ensemble["frac_balanced_mean"] - ensemble["frac_balanced_std"],
        ensemble["frac_balanced_mean"] + ensemble["frac_balanced_std"],
        color="gray", alpha=0.3
    )
    axes[2].plot(years, intercept + slope * years, "r--", label=f"Trend (p={p:.3f})")
    axes[2].set_ylabel("Balanced melt fraction")
    axes[2].set_xlabel("Year")
    axes[2].legend()
    axes[2].set_title("(c) Fraction of total melt that is balanced")

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, f"ensemble_balance_trend_{SSP}.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print(f"✅ Saved figure for {SSP}: {fig_path}")

print("\n🎉 All SSP ensembles processed successfully.")


#%% === CROSS-SSP COMPARISON: stacked bars + combined balanced fraction ===
print("\n📊 Building improved cross-SSP comparison figure...")

# Base SSP colors (consistent with other figures)
ssp_colors = {"ssp126": "#2b83ba", "ssp370": "#fdae61", "ssp585": "#d7191c"}

def adjust_color(hex_color, factor):
    """Brighten (>1) or darken (<1) an RGB hex color."""
    rgb = mcolors.to_rgb(hex_color)
    return tuple(min(1, max(0, c * factor)) for c in rgb)

# Per-SSP melt colors: light = balanced, dark = imbalanced
melt_colors = {
    ssp: {
        "balanced":   adjust_color(col, 1),  # lighter
        "imbalanced": adjust_color(col, 1),  # darker
    }
    for ssp, col in ssp_colors.items()
}

# --- Load ensemble balance CSVs ---
ssp_results = []
for SSP in ssp_list:
    csv_path = os.path.join(FIG_DIR, f"ensemble_balance_{SSP}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["SSP"] = SSP
        ssp_results.append(df)
    else:
        print(f"⚠️ Missing ensemble file for {SSP}: {csv_path}")

if not ssp_results:
    print("⚠️ No ensemble CSVs found; skipping comparison plot.")
else:
    ssp_df = pd.concat(ssp_results, ignore_index=True)

    ssp_order = ["ssp126", "ssp370", "ssp585"]

    # 4-row subplot layout: 3× melt, 1× fraction
    fig, axes = plt.subplots(4, 1, figsize=(7, 11), sharex=True)

    # ---- Top three panels: stacked balanced + imbalanced melt ----
    for i, ssp in enumerate(ssp_order):
        group = ssp_df[ssp_df["SSP"] == ssp]
        if group.empty:
            continue

        years = group["year_mean"] if "year_mean" in group.columns else group["year"]
        ax = axes[i]

        # Stacked bars
        ax.bar(
            years,
            group["balanced_m3_mean"],
            color=melt_colors[ssp]["balanced"],
            label="Balanced melt"
        )

        ax.bar(
            years,
            group["imbalanced_m3_mean"],
            bottom=group["balanced_m3_mean"],
            color=melt_colors[ssp]["imbalanced"],
            alpha=0.35,       # <= lighter via transparency
            label="Imbalanced melt"
        )
                # ---- Theil–Sen trend for TOTAL melt (balanced + imbalanced) ----
        total_m3 = group["balanced_m3_mean"].values + group["imbalanced_m3_mean"].values
        yrs = np.asarray(years, dtype=float)

        # guard against missing/NaN values
        mask = np.isfinite(yrs) & np.isfinite(total_m3)
        if mask.sum() >= 3:
            ts_slope, ts_intercept, lo_slope, hi_slope = theilslopes(total_m3[mask], yrs[mask], alpha=0.05)
            trend = ts_intercept + ts_slope * yrs

        ax.set_ylabel("Melt (m³ yr⁻¹)")
        ax.set_xlim(2025, 2100)
        ax.grid(alpha=0.3)

        # Panel labels (a), (b), (c)
        panel_label = f"({chr(97 + i)})  {ssp.upper()}"
        ax.text(
            0.02, 0.95,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
        )

        # Only show bar legend once (top panel)
        if i == 0:
            ax.legend(loc="upper right", frameon=False)

    # Unified y-limits for the melt panels
    # for i in range(3):
    #    axes[i].set_ylim(0, 3500)  # tweak if needed

    # ---- Bottom panel: balanced fraction across SSPs ----
    axf = axes[-1]
    for ssp in ssp_order:
        group = ssp_df[ssp_df["SSP"] == ssp]
        if group.empty:
            continue

        years = group["year_mean"] if "year_mean" in group.columns else group["year"]
        c = ssp_colors[ssp]

        axf.plot(
            years,
            group["frac_balanced_mean"],
            color=c,
            lw=1.5,
            label=ssp.upper(),
        )

    axf.set_ylim(0, 1)
    axf.set_ylabel("Balanced melt fraction")
    axf.set_xlabel("Year")
    axf.set_xlim(2025, 2100)
    axf.grid(alpha=0.3)

    # Panel label (d)
    axf.text(
        0.02, 0.95,
        "(d) ",
        transform=axf.transAxes,
        ha="left",
        va="top",
    )

    # Legend for SSP lines
    axf.legend(
    loc="lower left",
    ncol=1,
    frameon=True,           # Enable legend box
    facecolor="white",      # White background
    framealpha=1.0          # Fully opaque
)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "crossSSP_balance_combined.png")
    plt.savefig(fig_path, dpi=300)
    fig_path = os.path.join(FIG_DIR, "crossSSP_balance_combined.pdf")
    plt.savefig(fig_path, dpi=300)
    plt.show()

    print(f"✅ Saved cross-SSP comparison figure: {fig_path}")

#%% === CROSS-SSP COMPARISON: stacked bars mm melt

print("\n📊 Building improved cross-SSP comparison figure...")

# Base SSP colors (consistent with other figures)
ssp_colors = {"ssp126": "#2b83ba", "ssp370": "#fdae61", "ssp585": "#d7191c"}

def adjust_color(hex_color, factor):
    """Brighten (>1) or darken (<1) an RGB hex color."""
    rgb = mcolors.to_rgb(hex_color)
    return tuple(min(1, max(0, c * factor)) for c in rgb)

# Per-SSP melt colors: light = balanced, dark = imbalanced
melt_colors = {
    ssp: {
        "balanced":   adjust_color(col, 1),  # lighter
        "imbalanced": adjust_color(col, 1),  # darker
    }
    for ssp, col in ssp_colors.items()
}

# --- Load ensemble balance CSVs ---
ssp_results = []
for SSP in ssp_list:
    csv_path = os.path.join(FIG_DIR, f"ensemble_balance_{SSP}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["SSP"] = SSP
        ssp_results.append(df)
    else:
        print(f"⚠️ Missing ensemble file for {SSP}: {csv_path}")

if not ssp_results:
    print("⚠️ No ensemble CSVs found; skipping comparison plot.")
else:
    ssp_df = pd.concat(ssp_results, ignore_index=True)

    ssp_order = ["ssp126", "ssp370", "ssp585"]

    # 4-row subplot layout: 3× melt, 1× fraction
    fig, axes = plt.subplots(4, 1, figsize=(7, 11), sharex=True)

    # ---- Top three panels: stacked balanced + imbalanced melt ----
    for i, ssp in enumerate(ssp_order):
        group = ssp_df[ssp_df["SSP"] == ssp]
        if group.empty:
            continue

        years = group["year_mean"] if "year_mean" in group.columns else group["year"]
        ax = axes[i]

        # Stacked bars
        ax.bar(
            years,
            group["balanced_mm_mean"],
            color=melt_colors[ssp]["balanced"],
            label="Balanced melt"
        )

        ax.bar(
            years,
            group["imbalanced_mm_mean"],
            bottom=group["balanced_mm_mean"],
            color=melt_colors[ssp]["imbalanced"],
            alpha=0.35,       # <= lighter via transparency
            label="Imbalanced melt"
        )
                # ---- Theil–Sen trend for TOTAL melt (balanced + imbalanced) ----
        total_mm = group["balanced_mm_mean"].values + group["imbalanced_mm_mean"].values
        yrs = np.asarray(years, dtype=float)

        # guard against missing/NaN values
        mask = np.isfinite(yrs) & np.isfinite(total_mm)
        if mask.sum() >= 3:
            ts_slope, ts_intercept, lo_slope, hi_slope = theilslopes(total_m3[mask], yrs[mask], alpha=0.05)
            trend = ts_intercept + ts_slope * yrs

        ax.set_ylabel("Melt (mm yr⁻¹)")
        ax.set_xlim(2025, 2100)
        ax.set_ylim(0, 3500)  # Adjusted y-limits for clarity
        ax.grid(alpha=0.3)

        # Panel labels (a), (b), (c)
        panel_label = f"({chr(97 + i)})  {ssp.upper()}"
        ax.text(
            0.02, 0.95,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
        )

        # Only show bar legend once (top panel)
        if i == 0:
            ax.legend(loc="upper right", frameon=False)

    # Unified y-limits for the melt panels
    # for i in range(3):
    #    axes[i].set_ylim(0, 3500)  # tweak if needed

    # ---- Bottom panel: balanced fraction across SSPs ----
    axf = axes[-1]
    for ssp in ssp_order:
        group = ssp_df[ssp_df["SSP"] == ssp]
        if group.empty:
            continue

        years = group["year_mean"] if "year_mean" in group.columns else group["year"]
        c = ssp_colors[ssp]

        axf.plot(
            years,
            group["frac_balanced_mean"],
            color=c,
            lw=1.5,
            label=ssp.upper(),
        )

    axf.set_ylim(0, 1)
    axf.set_ylabel("Balanced melt fraction")
    axf.set_xlabel("Year")
    axf.set_xlim(2025, 2100)
    axf.grid(alpha=0.3)

    # Panel label (d)
    axf.text(
        0.02, 0.95,
        "(d) ",
        transform=axf.transAxes,
        ha="left",
        va="top",
    )

    # Legend for SSP lines
    axf.legend(
    loc="lower left",
    ncol=1,
    frameon=True,           # Enable legend box
    facecolor="white",      # White background
    framealpha=1.0          # Fully opaque
)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "crossSSP_balance_combined_specificmelt.png")
    plt.savefig(fig_path, dpi=300)
    fig_path = os.path.join(FIG_DIR, "crossSSP_balance_combined_specificmelt.pdf")
    plt.savefig(fig_path, dpi=300)
    plt.show()

    print(f"✅ Saved cross-SSP comparison figure: {fig_path}")

#%%
# ===============================================================
# COMBINE ALL SSP SUMMARY TABLES INTO ONE MASTER TABLE
# ===============================================================
print("\n📊 Combining summary tables into one master table...")


summary_files = [
    os.path.join(FIG_DIR, f"BalanceSummary_{ssp}.csv")
    for ssp in ["ssp126", "ssp370", "ssp585"]
]

summary_dfs = []
for f in summary_files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        summary_dfs.append(df)
    else:
        print(f"⚠️ Missing summary file: {f}")

if summary_dfs:
    summary_all = pd.concat(summary_dfs, ignore_index=True)
    out_path = os.path.join(FIG_DIR, "BalanceSummary_ALL.csv")
    summary_all.to_csv(out_path, index=False)
    print(f"✅ Saved combined summary table to: {out_path}")

    # === PRINTED (rounded) version ===
    summary_all_rounded = summary_all.round(1)
    print("\n=== Combined Balance Summary (rounded) ===")
    print(summary_all_rounded.to_string(index=False))

else:
    print("⚠️ No summary tables found — combined table not created.")

# %%
