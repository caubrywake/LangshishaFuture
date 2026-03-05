# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Seasonal forcing statistics from per-GCM daily time series

Author: Caroline Aubry-Wake
Last updated: 2025-11-20

Purpose
-------
This script processes climate forcing time series from multiple GCMs (per SSP)
to compute seasonal-scale ensemble statistics for basin-mean temperature
(tavg, tmin, tmax) and precipitation (prec). Results are used for climate trend
analysis and input to hydrological and groundwater impact assessments.

Processing Workflow
-------------------
For each variable (tavg, tmin, tmax, prec) and each SSP (ssp126, ssp370, ssp585):
    1. Read daily CSVs matching:
           "{var}_{gcm}_{ssp}.csv"
       from:
           base_dir = "/scratch/depfg/aubry001/out/basin_mean_temp/"
       Each file must contain:
           - 'date' column (ISO-format or parsable to pandas datetime)
           - Value column in order of priority:
               * "{var}_mean"
               * "{var}"
               * first non-'date' column

    2. Aggregate daily series to SEASONAL values:
           - Temperature → seasonal mean
           - Precipitation → seasonal sum
       Seasons defined as:
           DJF (Dec–Feb), MAM (Mar–May), JJA (Jun–Aug), SON (Sep–Nov)
       Note: December is assigned to the same calendar year.

    3. Build GCM-aligned seasonal annual series (only years available for all GCMs retained).

    4. Compute ensemble statistics:
           - first_window_mean ± std       (first 10 years or N/2 if shorter)
           - last_window_mean ± std
           - absolute and percent change (percent change *not interpreted for temp*)
           - Linear trend (slope per year) of ensemble seasonal time series
             using scipy.stats.linregress

    5. Store ensemble seasonal series for later plotting.

Outputs
-------
• Summary table:
      "forcing_seasonal_stats_per_ssp.csv"
  Saved to:
      base_dir = "/scratch/depfg/aubry001/out/basin_mean_temp/"

• Per var × ssp × season time series plots:
      "forcing_seasonal_{var}_{ssp}_{season}.png"

• Combined SSP plots per var × season:
      "forcing_seasonal_{var}_all_ssp_{season}.png"

• Console printout:
      Copy-paste ready tables with basic rounding

"""


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# === USER SETTINGS ===
base_dir = "/scratch/depfg/aubry001/out/basin_mean_temp/"
variables = ["tavg", "tmin", "tmax", "prec"]
ssps = ["ssp126", "ssp370", "ssp585"]
ssp_colors = {"ssp126": "#2b83ba", "ssp370": "#fdae61", "ssp585": "#d7191c"}

# seasons of interest (monsoon-relevant: JJA kept intact)
seasons = ["DJF", "MAM", "JJA", "SON"]

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


def assign_season(month):
    """
    Map month (1–12) to climatological season.
    Note: DJF uses December of the same calendar year.
    Adjust if you prefer Dec to be attached to the following year's DJF.
    """
    if month in [12, 1, 2]:
        return "DJF"
    elif month in [3, 4, 5]:
        return "MAM"
    elif month in [6, 7, 8]:
        return "JJA"
    else:
        return "SON"


def compute_first_last_stats(years, values, window=WINDOW):
    """
    Given a seasonal series (years, values), compute:
        - first_mean, last_mean (window adjusted if series is short)
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

# container for ensemble seasonal series per var × season × ssp
ensemble_store = {
    var: {season: {} for season in seasons} for var in variables
}

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

        gcm_season_series = {}

        # --- load per-GCM daily and aggregate to SEASONAL ---
        for fpath in files:
            fname = os.path.basename(fpath)
            parts = fname.replace(".csv", "").split("_")
            if len(parts) < 3:
                print(f"⚠️ Unexpected filename format (skipping): {fname}")
                continue

            gcm = "_".join(parts[1:-1])  # robust to gcm with underscores

            df = pd.read_csv(fpath, parse_dates=["date"])
            val_col = infer_value_column(df, var)

            # derive year & season
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["season"] = df["month"].apply(assign_season)

            # seasonal aggregation
            if var == "prec":
                seasonal = df.groupby(["year", "season"])[val_col].sum()
            else:
                seasonal = df.groupby(["year", "season"])[val_col].mean()

            gcm_season_series[gcm] = seasonal

        if not gcm_season_series:
            print(f"⚠️ No valid seasonal series for {var} {ssp}")
            continue

        # --- loop over seasons separately ---
        for season in seasons:
            # build common seasonal DataFrame across GCMs for this season
            frames = []
            for gcm, ser in gcm_season_series.items():
                # extract this season's series → index is 'year'
                try:
                    ser_season = ser.xs(season, level="season")
                except KeyError:
                    continue  # this GCM has no data for this season

                frames.append(ser_season.rename(gcm))

            if not frames:
                print(f"⚠️ No data for {var} {ssp} {season}")
                continue

            seasonal_df = pd.concat(frames, axis=1)
            # keep only years where all GCMs have data
            seasonal_df = seasonal_df.dropna(how="any")

            if seasonal_df.empty:
                print(f"⚠️ Seasonal series empty after alignment for {var} {ssp} {season}")
                continue

            years = seasonal_df.index.values.astype(int)
            n_years = len(years)

            # --- per-GCM first/last stats ---
            first_means = []
            last_means = []
            change_abss = []
            change_pcts = []

            for gcm in seasonal_df.columns:
                vals = seasonal_df[gcm].values
                stats_gl = compute_first_last_stats(years, vals)
                first_means.append(stats_gl["first_mean"])
                last_means.append(stats_gl["last_mean"])
                change_abss.append(stats_gl["change_abs"])
                change_pcts.append(stats_gl["change_pct"])

            first_means = np.array(first_means, dtype=float)
            last_means = np.array(last_means, dtype=float)
            change_abss = np.array(change_abss, dtype=float)
            change_pcts = np.array(change_pcts, dtype=float)

            # --- ensemble seasonal series (mean across GCMs) ---
            ensemble_vals = seasonal_df.mean(axis=1).values

            # store ensemble for later multi-SSP plots
            ensemble_store[var][season][ssp] = {
                "years": years.copy(),
                "values": ensemble_vals.copy(),
            }

            # trend on ensemble seasonal series
            if n_years >= 3:
                slope, intercept, r_value, p_value, std_err = linregress(years, ensemble_vals)
                r_squared = r_value ** 2
            else:
                slope = np.nan
                p_value = np.nan
                r_squared = np.nan

            # --- aggregate GCM metrics: mean ± std across GCMs ---
            def mean_and_std(arr):
                valid = np.isfinite(arr)
                if valid.sum() > 1:
                    return np.nanmean(arr), np.nanstd(arr, ddof=1)
                else:
                    return np.nanmean(arr), np.nan

            first_mean_mean, first_mean_std = mean_and_std(first_means)
            last_mean_mean, last_mean_std = mean_and_std(last_means)
            change_abs_mean, change_abs_std = mean_and_std(change_abss)
            change_pct_mean, change_pct_std = mean_and_std(change_pcts)

            # effective window used
            if n_years >= 2 * WINDOW:
                first_window_years = WINDOW
            else:
                first_window_years = max(1, n_years // 2)

            summary_records.append({
                "variable": var,
                "season": season,
                "ssp": ssp,
                "n_gcm": len(seasonal_df.columns),
                "year_min": years.min(),
                "year_max": years.max(),
                "first_window_years": first_window_years,

                "first_period_mean": first_mean_mean,
                "first_period_std": first_mean_std,
                "last_period_mean": last_mean_mean,
                "last_period_std": last_mean_std,
                "change_abs_mean": change_abs_mean,
                "change_abs_std": change_abs_std,
                "change_pct_mean": change_pct_mean,
                "change_pct_std": change_pct_std,

                "ensemble_trend_slope_per_year": slope,
                "ensemble_trend_p_value": p_value,
                "ensemble_trend_r_squared": r_squared,
                "ensemble_trend_significant_0.05": (p_value < 0.05) if np.isfinite(p_value) else False,
            })

            # --- Plot all GCM seasonal series + ensemble (per var × ssp × season) ---
            plt.figure(figsize=(9, 5))

            # GCMs: thin grey lines
            for gcm in seasonal_df.columns:
                plt.plot(years, seasonal_df[gcm].values, color="grey", alpha=0.4, lw=1,
                         label="_nolegend_")

            # Ensemble: thick colored line
            plt.plot(years, ensemble_vals, color=ssp_colors.get(ssp, "k"), lw=2.5,
                     label=f"{ssp.upper()} ensemble")

            if var == "prec":
                ylabel = f"{season} basin-mean precip (mm/season)"
            else:
                ylabel = f"{season} basin-mean {var} (°C)"

            plt.ylabel(ylabel)
            plt.xlabel("Year")
            plt.title(f"{season} {var} – {ssp.upper()} (GCMs + ensemble)")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()

            out_fig = os.path.join(base_dir, f"forcing_seasonal_{var}_{ssp}_{season}.png")
            plt.savefig(out_fig, dpi=300, bbox_inches="tight")
            plt.show()
            print(f"📈 Saved seasonal plot → {out_fig}")

# --- Combined plots per variable × season with all SSP ensembles ---
for var in variables:
    for season in seasons:
        if season not in ensemble_store[var]:
            continue
        if not ensemble_store[var][season]:
            continue

        plt.figure(figsize=(9, 5))
        for ssp in ssps:
            if ssp not in ensemble_store[var][season]:
                continue
            yrs = ensemble_store[var][season][ssp]["years"]
            vals = ensemble_store[var][season][ssp]["values"]
            plt.plot(yrs, vals, lw=2.5, color=ssp_colors.get(ssp, "k"),
                     label=f"{ssp.upper()} ensemble")

        if var == "prec":
            ylabel = f"{season} basin-mean precip (mm/season)"
        else:
            ylabel = f"{season} basin-mean {var} (units)"

        plt.ylabel(ylabel)
        plt.xlabel("Year")
        plt.title(f"{season} {var} – all SSP ensembles")
        plt.grid(alpha=0.3)
        plt.legend(title="Scenario")
        plt.tight_layout()

        out_fig_all = os.path.join(base_dir, f"forcing_seasonal_{var}_all_ssp_{season}.png")
        plt.savefig(out_fig_all, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"📈 Saved combined SSP seasonal plot → {out_fig_all}")

# --- Save seasonal summary table ---
summary_df = pd.DataFrame(summary_records)
out_csv = os.path.join(base_dir, "forcing_seasonal_stats_per_ssp.csv")
summary_df.to_csv(out_csv, index=False)
print(f"✅ Saved seasonal summary stats to {out_csv}")

# --- Nicely formatted, copy-paste-friendly tables per variable ---
display_cols = [
    "season",
    "ssp",
    "first_period_mean",
    "first_period_std",
    "last_period_mean",
    "last_period_std",
    "change_abs_mean",
    "change_abs_std",
    "change_pct_mean",
    "change_pct_std",
    "ensemble_trend_slope_per_year",
    "ensemble_trend_p_value",
    ]

display_cols = [c for c in display_cols if c in summary_df.columns]
numeric_cols = summary_df.select_dtypes(include=[np.number]).columns

print("\n================ SEASONAL COPY-PASTE TABLES ================\n")
for var in variables:
    sub = summary_df[summary_df["variable"] == var].copy()
    if sub.empty:
        continue

    # round numeric columns to 2 decimals for readability
    sub[numeric_cols] = sub[numeric_cols].round(2)
    sub = sub[display_cols]

    print(f"\n=== {var} ===")
    print(sub.to_string(index=False))
    print()
