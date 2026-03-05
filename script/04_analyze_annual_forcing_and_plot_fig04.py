# -*- coding: utf-8 -*-
"""
Annual forcing statistics from per-GCM daily time series

Author: Caroline Aubry-Wake
Last updated: 2025-11-20

Purpose
-------
This script processes daily basin-mean climate forcing time series from multiple
GCMs (per SSP) to compute annual ensemble statistics for temperature (tavg,
tmin, tmax) and precipitation (prec). It is used to quantify long-term trends
and end-of-century changes in annual forcing for hydrological and groundwater
impact analysis.

Inputs
------
Directory with per-GCM daily CSV files:
    base_dir = "/scratch/depfg/aubry001/out/basin_mean_temp/"

File naming pattern:
    {var}_{gcm}_{ssp}.csv
Examples:
    "tavg_mri-esm2-0_ssp126.csv"
    "prec_mpi-esm1-2-hr_ssp585.csv"

Each CSV must contain at least:
    - "date" column (parsable as datetime)
    - One value column, selected in this priority order:
          1) "{var}_mean"   (e.g., "prec_mean")
          2) "{var}"        (e.g., "prec")
          3) first non-"date" column (fallback)

Variables:
    - tavg, tmin, tmax, prec

Scenarios (SSPs):
    - ssp126, ssp370, ssp585

Processing Logic
----------------
For each variable × SSP combination:

1) File discovery:
       - Find all matching per-GCM CSVs:
             {var}_*_{ssp}.csv
         within:
             base_dir = "/scratch/depfg/aubry001/out/basin_mean_temp/"

2) Per-GCM annual aggregation:
       - Read daily time series and derive "year" from "date".
       - Aggregate to annual:
             * Temperature variables: annual mean
             * Precipitation: annual sum

3) GCM alignment:
       - Concatenate all GCM annual series into a single DataFrame.
       - Drop years where any GCM is missing data
         → use only years common to all GCMs.

4) Ensemble annual series:
       - Compute ensemble mean and standard deviation across GCMs
         for each year:
             ensemble_mean(year), ensemble_std(year)
       - Store these for later multi-SSP plotting.

5) First / last window statistics:
       - Using each GCM’s annual series, compute:
             * First-window mean (first 10 years or N/2 if shorter)
             * Last-window mean  (last 10 years or N/2 if shorter)
             * Absolute change (last - first)
             * Percent change (relative to first; meaningful mainly for precip)
       - Aggregate across GCMs:
             * mean ± std for:
                   first_10_mean, last_10_mean,
                   change_abs, change_pct

6) Trend estimation:
       - Perform a linear regression (scipy.stats.linregress) of the
         ensemble annual series (years vs ensemble_mean):
             * slope (per year)
             * p-value
             * r-squared
       - Store slope and significance for each var × SSP.

7) Plot per var × SSP annual series:
       - Thin grey lines: individual GCM annual series
       - Thick coloured line: ensemble mean annual series
       - One plot per (var, ssp).

8) Summary tables:
       - Build a summary table with one row per var × SSP containing:
             * number of GCMs, period covered
             * first/last window means ± std
             * absolute and percent changes ± std
             * ensemble trend slope and p-value
       - Print copy-paste-friendly tables to the console with rounded values.

9) Multi-SSP summary figure:
       - Create a 2-panel figure:
             (a) Annual basin-mean tavg (ensemble mean ± 1σ across GCMs),
                 with linear trend lines per SSP.
             (b) Annual basin-mean precipitation (mm/year),
                 ensemble mean ± 1σ with trend lines per SSP.
       - X-axis restricted to 2025–2100.
       - Legend shown only in panel (a); panel labels “(a)” and “(b)” added.

Outputs
-------
Summary CSV:
    - "forcing_annual_stats_per_ssp.csv"
      saved in:
    - "/scratch/depfg/aubry001/out/basin_mean_temp/"

Per var × SSP annual series plots:
    - "forcing_annual_{var}_{ssp}.png"
      e.g., "forcing_annual_tavg_ssp126.png"

2-panel summary figure (tavg & prec, all SSPs):
    - "forcing_annual_tavg_prec_all_ssp_mean_std_trends.png" -> manuscript Fig 04

Console tables:
    - Rounded (copy-paste-friendly) annual statistics printed per variable.


"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, theilslopes


# === USER SETTINGS ===
base_dir = "/scratch/depfg/aubry001/out/basin_mean_temp/"
variables = ["tavg", "tmin", "tmax", "prec"]
ssps = ["ssp126", "ssp370", "ssp585"]
ssp_colors = {"ssp126": "#2b83ba", "ssp370": "#fdae61", "ssp585": "#d7191c"}

# length of first/last window (years)
WINDOW = 10  # will be reduced if series shorter


def infer_value_column(df, var):
    """
    Try to infer which column holds the daily variable values.
    Priority:
        1) f"{var}_mean"
        2) var
        3) first non-'date' column
    """
    mean_col = f"{var}_mean"
    if mean_col in df.columns:
        return mean_col
    if var in df.columns:
        return var
    non_date_cols = [c for c in df.columns if c.lower() != "date"]
    if not non_date_cols:
        raise ValueError("No data column found (only 'date' present).")
    return non_date_cols[0]


def compute_first_last_stats(years, values, window=WINDOW):
    """
    Given annual series (years, values), compute:
        - first_10_mean, last_10_mean (10 = window, adjusted if short)
        - change_abs, change_pct
    """
    years = np.asarray(years)
    vals = np.asarray(values)

    # sort by year
    order = np.argsort(years)
    years = years[order]
    vals = vals[order]

    n = len(vals)
    if n == 0:
        return {
            "first_mean": np.nan,
            "last_mean": np.nan,
            "change_abs": np.nan,
            "change_pct": np.nan,
        }

    # choose window length: <= WINDOW and <= n/2 (if possible)
    if n >= 2 * window:
        w = window
    else:
        w = max(1, n // 2)

    first_vals = vals[:w]
    last_vals = vals[-w:]

    first_mean = np.nanmean(first_vals)
    last_mean = np.nanmean(last_vals)
    change_abs = last_mean - first_mean
    if first_mean != 0:
        change_pct = (change_abs / abs(first_mean)) * 100.0
    else:
        change_pct = np.nan

    return {
        "first_mean": first_mean,
        "last_mean": last_mean,
        "change_abs": change_abs,
        "change_pct": change_pct,
    }


# container for summary rows
summary_records = []
ensemble_store = {var: {} for var in variables}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
})

for var in variables:
    for ssp in ssps:
        # find all GCM files for this var × ssp
        pattern = os.path.join(base_dir, f"{var}_*_{ssp}.csv")
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"⚠️ No files found for {var} {ssp} with pattern {pattern}")
            continue

        gcm_names = []
        gcm_annual_series = {}

        # --- load per-GCM daily and aggregate to annual ---
        for fpath in files:
            fname = os.path.basename(fpath)
            # expected pattern var_gcm_ssp.csv → split by '_'
            parts = fname.replace(".csv", "").split("_")
            if len(parts) < 3:
                print(f"⚠️ Unexpected filename format (skipping): {fname}")
                continue
            # var, gcm, ssp
            gcm = "_".join(parts[1:-1])  # robust to gcm with underscores
            gcm_names.append(gcm)

            df = pd.read_csv(fpath, parse_dates=["date"])
            val_col = infer_value_column(df, var)

            # derive year
            df["year"] = df["date"].dt.year

            # annual aggregation
            if var == "prec":
                annual = df.groupby("year")[val_col].sum()
            else:
                annual = df.groupby("year")[val_col].mean()

            gcm_annual_series[gcm] = annual

        if not gcm_annual_series:
            print(f"⚠️ No valid annual series for {var} {ssp}")
            continue

        # --- build common annual DataFrame across GCMs ---
        # outer join then drop years where any GCM is missing
        annual_df = pd.concat(gcm_annual_series, axis=1)
        annual_df.columns = gcm_annual_series.keys()
        annual_df = annual_df.dropna(how="any")

        if annual_df.empty:
            print(f"⚠️ Annual series empty after alignment for {var} {ssp}")
            continue

        years = annual_df.index.values.astype(int)
        n_years = len(years)

        # --- per-GCM first/last stats ---
        first_means = []
        last_means = []
        change_abss = []
        change_pcts = []

        for gcm in annual_df.columns:
            vals = annual_df[gcm].values
            stats_gl = compute_first_last_stats(years, vals)
            first_means.append(stats_gl["first_mean"])
            last_means.append(stats_gl["last_mean"])
            change_abss.append(stats_gl["change_abs"])
            change_pcts.append(stats_gl["change_pct"])

        first_means = np.array(first_means, dtype=float)
        last_means = np.array(last_means, dtype=float)
        change_abss = np.array(change_abss, dtype=float)
        change_pcts = np.array(change_pcts, dtype=float)

        # --- ensemble annual series (mean across GCMs) ---
        ensemble_mean = annual_df.mean(axis=1).values
        ensemble_std = annual_df.std(axis=1, ddof=1).values

        # store ensemble series (mean + std) for later multi-SSP plots
        ensemble_store[var][ssp] = {
            "years": years.copy(),
            "mean": ensemble_mean.copy(),
            "std": ensemble_std.copy(),
        }

        # trend on ensemble annual series
       # trend on ensemble annual series (Theil–Sen; robust to outliers, less assumption-heavy)
        if n_years >= 3:
            ts_slope, ts_intercept, ts_lo, ts_hi = theilslopes(ensemble_mean, years, alpha=0.05)
            slope = ts_slope
            intercept = ts_intercept
            p_value = np.nan          # do not report p-values unless using an autocorrelation-aware test
            r_squared = np.nan        # not defined/standard for Theil–Sen
        else:
            slope = np.nan
            intercept = np.nan
            p_value = np.nan
            r_squared = np.nan


        # --- aggregate GCM metrics: mean ± std across GCMs ---
        def mean_and_std(arr):
            return np.nanmean(arr), np.nanstd(arr, ddof=1) if np.isfinite(arr).sum() > 1 else (np.nanmean(arr), np.nan)

        first_mean_mean, first_mean_std = mean_and_std(first_means)
        last_mean_mean, last_mean_std = mean_and_std(last_means)
        change_abs_mean, change_abs_std = mean_and_std(change_abss)
        change_pct_mean, change_pct_std = mean_and_std(change_pcts)

        summary_records.append({
            "variable": var,
            "ssp": ssp,
            "n_gcm": len(annual_df.columns),
            "year_min": years.min(),
            "year_max": years.max(),
            "first_window_years": min(WINDOW, n_years // 2 if n_years >= 2 else 1),

            "first_10_mean_mean": first_mean_mean,
            "first_10_mean_std": first_mean_std,
            "last_10_mean_mean": last_mean_mean,
            "last_10_mean_std": last_mean_std,
            "change_abs_mean": change_abs_mean,
            "change_abs_std": change_abs_std,
            "change_pct_mean": change_pct_mean,
            "change_pct_std": change_pct_std,

            "ensemble_trend_slope_per_year": slope,
            "ensemble_trend_p_value": p_value,
            "ensemble_trend_r_squared": r_squared,
            "ensemble_trend_significant_0.05": False,
               })

        # --- Plot all GCM annual series + ensemble ---
        plt.figure(figsize=(9, 5))
        # GCMs: thin grey lines
        for gcm in annual_df.columns:
            plt.plot(years, annual_df[gcm].values, color="grey", alpha=0.4, lw=1, label="_nolegend_")

        # Ensemble: thick colored line
        plt.plot(years, ensemble_mean, color=ssp_colors[ssp], lw=2.5,
                 label=f"{ssp.upper()} ensemble")

        if var == "prec":
            ylabel = "Annual basin-mean precip (mm/year)"
        else:
            ylabel = f"Annual basin-mean {var} (units)"

        plt.ylabel(ylabel)
        plt.xlabel("Year")
        plt.title(f"Annual {var} – {ssp.upper()} (GCMs + ensemble)")
        plt.grid(alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()

        out_fig = os.path.join(base_dir, f"forcing_annual_{var}_{ssp}.png")
        plt.savefig(out_fig, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"📈 Saved {var} {ssp} annual series plot → {out_fig}")



# %% === Save summary stats table to CSV ===
summary_df = pd.DataFrame(summary_records)

out_csv = os.path.join(base_dir, "forcing_annual_stats_per_ssp.csv")
summary_df.to_csv(out_csv, index=False)
print(f"✅ Saved summary stats to {out_csv}")

# %% === Nicely formatted, copy-paste-friendly tables per variable ===

# choose which columns to show in the printed tables
display_cols = [
    "ssp",
    "first_10_mean_mean",
    "first_10_mean_std",
    "last_10_mean_mean",
    "last_10_mean_std",
    "change_abs_mean",
    "change_abs_std",
    "change_pct_mean",
    "change_pct_std",
    "ensemble_trend_slope_per_year",
    "ensemble_trend_p_value",
]

# keep only columns that exist (in case you tweak above)
display_cols = [c for c in display_cols if c in summary_df.columns]

# identify numeric columns for rounding
numeric_cols = summary_df.select_dtypes(include=[np.number]).columns

print("\n================ ANNUAL COPY-PASTE TABLES ================\n")
for var in variables:
    sub = summary_df[summary_df["variable"] == var].copy()
    if sub.empty:
        continue

    # round numeric columns for readability
    sub[numeric_cols] = sub[numeric_cols].round(2)

    # keep only the display columns
    sub = sub[display_cols]

    print(f"\n=== {var} ===")
    print(sub.to_string(index=False))
    print()  # blank line between variables

# %%

#%% === NEW: 2-panel figure: annual tavg and prec (ensemble mean ± std + trends) ===
fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

panel_vars = [
    ("tavg", "Annual basin-mean\ntemperature (°C)"),
    ("prec", "Annual basin-mean\nprecipitation (mm/year)")
]

for ax, (var, ylabel) in zip(axes, panel_vars):
    if var not in ensemble_store or not ensemble_store[var]:
        continue

    for ssp in ssps:
        if ssp not in ensemble_store[var]:
            continue

        data = ensemble_store[var][ssp]
        yrs = data["years"]
        mean_vals = data["mean"]
        std_vals = data["std"]

        color = ssp_colors.get(ssp, "k")
        label = ssp.upper()

        # Ensemble mean line
        ax.plot(yrs, mean_vals, color=color, lw=1.0, label=label)

        # ±1σ shading across GCMs
        ax.fill_between(
            yrs,
            mean_vals - std_vals,
            mean_vals + std_vals,
            color=color,
            alpha=0.2,
            linewidth=0,
        )

        # Linear trend on ensemble mean (same years / same data)
        if len(yrs) >= 3:
            ts_slope, ts_intercept, _, _ = theilslopes(mean_vals, yrs, alpha=0.05)
            trend_vals = ts_intercept + ts_slope * yrs
            ax.plot(
                yrs,
                trend_vals,
                color=color,
                lw=1,
                ls="--",
                alpha=0.9,
            )

    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Year")

panel_labels = ["(a)", "(b)"]

for i, (ax, lab) in enumerate(zip(axes, panel_labels)):
    # Panel label placement (not bold)
    ax.text(
        0.02, 0.9, lab,
        transform=ax.transAxes,
        ha="left", va="bottom"
    )

    # Legend only for first panel (a)
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
    handles,
    labels,
    title="Scenario",
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.78)
)
    # no legend at all for panel (b)
# Force x-axis range on both panels
for ax in axes:
    ax.set_xlim(2025, 2100)
plt.tight_layout(rect=[0, 0, 1, 0.94])

out_fig_panel = os.path.join(base_dir, "forcing_annual_tavg_prec_all_ssp_mean_std_trends.png")
plt.savefig(out_fig_panel, dpi=300, bbox_inches="tight")
plt.show()
print(f"📈 Saved 2-panel summary figure with trends → {out_fig_panel}")

# %% === Save summary stats table to CSV ===
summary_df = pd.DataFrame(summary_records)

out_csv = os.path.join(base_dir, "forcing_annual_stats_per_ssp.csv")
summary_df.to_csv(out_csv, index=False)
print(f"✅ Saved summary stats to {out_csv}")

# %% === Nicely formatted, copy-paste-friendly tables per variable ===

# choose which columns to show in the printed tables
display_cols = [
    "ssp",
    "first_10_mean_mean",
    "first_10_mean_std",
    "last_10_mean_mean",
    "last_10_mean_std",
    "change_abs_mean",
    "change_abs_std",
    "change_pct_mean",
    "change_pct_std",
    "ensemble_trend_slope_per_year",
    "ensemble_trend_p_value",
]

# keep only columns that exist (in case you tweak above)
display_cols = [c for c in display_cols if c in summary_df.columns]

# identify numeric columns for rounding
numeric_cols = summary_df.select_dtypes(include=[np.number]).columns

print("\n================ ANNUAL COPY-PASTE TABLES ================\n")
for var in variables:
    sub = summary_df[summary_df["variable"] == var].copy()
    if sub.empty:
        continue

    # round numeric columns for readability
    sub[numeric_cols] = sub[numeric_cols].round(2)

    # keep only the display columns
    sub = sub[display_cols]

    print(f"\n=== {var} ===")
    print(sub.to_string(index=False))
    print()  # blank line between variables
