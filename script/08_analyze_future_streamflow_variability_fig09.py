# -*- coding: utf-8 -*-
"""
===============================================================================
Script: 09 — Future Streamflow Variability Analysis (Monthly CV, 2025–2098)
Author: Dr. Caroline Aubry-Wake
Affiliation: University of Lethbridge
Date: 2025-11-20
Python: 3.9+

Purpose
-------
Quantify changes in intra-month streamflow variability under future climate
scenarios (SSP1-2.6, SSP3-7.0, SSP5-8.5) using SPHY future simulations.

This script:
    • Computes the coefficient of variation (CV = σ/μ) of daily discharge
      for each calendar month and year (2025–2098).
    • Aggregates ensemble mean ± standard deviation across GCMs.
    • Computes robust temporal trends using Theil–Sen slope estimation.
    • Quantifies early (2025–2034) vs late (2089–2098) changes.
    • Generates multi-panel variability figures.
    • Exports summary tables for reproducibility.

This script generates Figure 9 in the manuscript.

-------------------------------------------------------------------------------
Inputs
-------------------------------------------------------------------------------
SPHY future simulation outputs:
    base_dir = "/scratch/depfg/aubry001/out/fut"
        <GCM>/<SSP>/QAllDTS.tss

Required file:
    - QAllDTS.tss → routed daily discharge at basin outlet

Time settings:
    start_date = 2015-01-01
    Analysis period = 2025–2098

-------------------------------------------------------------------------------
Methods
-------------------------------------------------------------------------------
1. Read daily outlet discharge (m³ s⁻¹) for each GCM and SSP.
2. For each calendar month and year:
       CV_month,year = std(daily Q) / mean(daily Q)
3. Aggregate ensemble mean ± std across GCMs.
4. Estimate trend using Theil–Sen slope (robust to outliers).
5. Compute early vs late decade change metrics.

-------------------------------------------------------------------------------
Outputs (saved in out_dir = "/scratch/depfg/aubry001/fig/futlong_flow")
-------------------------------------------------------------------------------

Figures
-------
1. CV_monthly_all12months_allSSP.png
2. CV_monthly_all12months_allSSP.pdf

   12-panel (6×2) monthly CV time series:
       • Ensemble mean ± 1σ
       • Theil–Sen slope annotation (CV per year)

3. CV_monthly_Apr_Jul_allSSP.png
4. CV_monthly_Apr_Jul_allSSP.pdf

   Reduced manuscript figure (April and July only).

Tables
------
5. TheilSenSlope_CV_monthly_bySSP.csv

   For each SSP and month:
       • slope_mean_per_yr
       • slope_std_per_yr
       • slope_median_per_yr
       • ensmean_slope_per_yr
       • n_gcm

6. TheilSenSlope_CV_monthly_perGCM_<ssp>.csv

   Per-GCM Theil–Sen slopes (one file per SSP).

7. CV_monthly_change_summary.csv

   Early (2025–2034) vs late (2089–2098) comparison:
       • EarlyMean_CV
       • LateMean_CV
       • Absolute change
       • Percent change
       • Theil–Sen slope (CV per year)
       • Percent change per decade

-------------------------------------------------------------------------------
Notes
-------------------------------------------------------------------------------
• CV is unitless (σ/μ of daily discharge).
• CV is computed independently for each month (Jan–Dec).
• Years with insufficient daily data are excluded.
• Trends are estimated using Theil–Sen (robust non-parametric slope).
• Ensemble statistics are computed across all available GCMs.
• Analysis period: 2025–2099 (inclusive).

===============================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import theilslopes

# -------------------------
# SETTINGS
# -------------------------
base_dir   = "/scratch/depfg/aubry001/out/fut"
ssps       = ["ssp126", "ssp370", "ssp585"]
tss_name   = "QAllDTS.tss"
start_date = pd.to_datetime("2015-01-01")
year_min   = 2025
year_max   = 2098

out_dir = "/scratch/depfg/aubry001/fig/futlong_flow"
os.makedirs(out_dir, exist_ok=True)

# 12-month diagnostic
months = {pd.Timestamp(2001, m, 1).strftime("%b"): m for m in range(1, 13)}  # {'Jan':1,...}

# -------------------------
# HELPERS
# -------------------------
def read_qall_outlet(file_path: str, start_date: pd.Timestamp) -> pd.Series:
    """Read SPHY .tss, return Series of outlet discharge (first data column col_1)."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"No numeric data found in {file_path}")

    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=start_idx, header=None)
    df.columns = ["day"] + [f"col_{i}" for i in range(1, df.shape[1])]

    t0 = df["day"].iloc[0]
    dt = start_date + pd.to_timedelta(df["day"] - t0, unit="D")

    q = pd.Series(df["col_1"].astype(float).values, index=dt, name="Q").sort_index()
    return q

def calc_cv(x, eps=1e-12, min_n=10) -> float:
    x = pd.Series(x).dropna()
    if len(x) < min_n:
        return np.nan
    m = x.mean()
    if m < eps:
        return np.nan
    return x.std(ddof=1) / m

def monthly_cv_by_year(q: pd.Series, month: int, year_min: int, year_max: int) -> pd.Series:
    q = q.loc[f"{year_min}-01-01":f"{year_max}-12-31"]
    q = q[q.index.month == month]
    cv = q.groupby(q.index.year).apply(calc_cv)
    cv = cv.loc[(cv.index >= year_min) & (cv.index <= year_max)]
    cv.index.name = "Year"
    return cv

def ensemble_mean_std(series_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame(series_dict)  # Year x GCM
    return pd.DataFrame({
        "mean": df.mean(axis=1),
        "std":  df.std(axis=1),
        "n_gcm": df.count(axis=1)
    }).sort_index()

def theil_sen_slope(y: pd.Series) -> dict:
    """
    Robust slope of y vs year using Theil–Sen.
    Returns slope, intercept, lo, hi, n.
    Slope units: CV per year.
    """
    yy = y.dropna()
    if yy.size < 10:
        return {"slope": np.nan, "intercept": np.nan, "lo": np.nan, "hi": np.nan, "n": int(yy.size)}

    x = yy.index.values.astype(float)
    slope, intercept, lo_slope, hi_slope = theilslopes(yy.values.astype(float), x)
    return {"slope": slope, "intercept": intercept, "lo": lo_slope, "hi": hi_slope, "n": int(yy.size)}

# -------------------------
# DISCOVER GCMs
# -------------------------
gcms = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
print(f"Found {len(gcms)} GCM directories in {base_dir}")

# -------------------------
# COMPUTE: monthly CV time series + slopes
# -------------------------
# raw_cv[month_label][ssp][gcm] = Series(year->cv)
raw_cv = {m_lab: {ssp: {} for ssp in ssps} for m_lab in months.keys()}

# ens_cv[month_label][ssp] = DataFrame(index=Year, cols=mean,std,n_gcm)
ens_cv = {m_lab: {} for m_lab in months.keys()}

# slopes_gcm[(ssp, month_label)] = DataFrame rows=gcm, cols=slope,lo,hi,n
slopes_gcm = {}

for ssp in ssps:
    print(f"\n=== {ssp.upper()} ===")
    for gcm in gcms:
        tss_path = os.path.join(base_dir, gcm, ssp, tss_name)
        if not os.path.exists(tss_path):
            continue
        try:
            q = read_qall_outlet(tss_path, start_date)
        except Exception as e:
            print(f"  ⚠️ {gcm}: failed to read ({e})")
            continue

        for m_lab, m_num in months.items():
            cv = monthly_cv_by_year(q, m_num, year_min, year_max)
            if not cv.dropna().empty:
                raw_cv[m_lab][ssp][gcm] = cv

    # build ensemble stats + per-GCM slopes per month
    for m_lab in months.keys():
        if raw_cv[m_lab][ssp]:
            ens_cv[m_lab][ssp] = ensemble_mean_std(raw_cv[m_lab][ssp])

            # per-GCM slopes
            rows = []
            for gcm, cv_series in raw_cv[m_lab][ssp].items():
                ts = theil_sen_slope(cv_series)
                rows.append({"GCM": gcm, **ts})
            slopes_gcm[(ssp, m_lab)] = pd.DataFrame(rows).set_index("GCM").sort_index()
        else:
            ens_cv[m_lab][ssp] = None
            slopes_gcm[(ssp, m_lab)] = pd.DataFrame(columns=["slope","intercept","lo","hi","n"])

# -------------------------
# SUMMARY TABLE: slope stats across GCMs
# -------------------------
summary_rows = []
for ssp in ssps:
    for m_lab in months.keys():
        df_s = slopes_gcm[(ssp, m_lab)]
        if df_s.empty:
            continue

        s = df_s["slope"].astype(float).dropna()
        if s.empty:
            summary_rows.append({
                "SSP": ssp, "Month": m_lab,
                "slope_mean_per_yr": np.nan, "slope_std_per_yr": np.nan,
                "slope_median_per_yr": np.nan, "n_gcm": 0,
                "ensmean_slope_per_yr": np.nan
            })
            continue

        # optional: slope of ensemble mean series (single line)
        ens_slope = np.nan
        if ens_cv[m_lab][ssp] is not None:
            ens_slope = theil_sen_slope(ens_cv[m_lab][ssp]["mean"])["slope"]

        summary_rows.append({
            "SSP": ssp,
            "Month": m_lab,
            "slope_mean_per_yr": s.mean(),
            "slope_std_per_yr": s.std(ddof=1),
            "slope_median_per_yr": s.median(),
            "n_gcm": int(s.size),
            "ensmean_slope_per_yr": ens_slope
        })

slope_summary = pd.DataFrame(summary_rows)

# Save slope table
out_csv = os.path.join(out_dir, "TheilSenSlope_CV_monthly_bySSP.csv")
slope_summary.to_csv(out_csv, index=False)
print(f"\nSaved slope summary table:\n  {out_csv}")

# Also save per-GCM slopes (optional, one file per SSP)
for ssp in ssps:
    wide = []
    for m_lab in months.keys():
        df_s = slopes_gcm[(ssp, m_lab)].copy()
        if df_s.empty:
            continue
        df_s = df_s[["slope","lo","hi","n"]].rename(columns=lambda c: f"{m_lab}_{c}")
        wide.append(df_s)
    if wide:
        wide_df = pd.concat(wide, axis=1)
        out_gcm = os.path.join(out_dir, f"TheilSenSlope_CV_monthly_perGCM_{ssp}.csv")
        wide_df.to_csv(out_gcm)
        print(f"Saved per-GCM slopes:\n  {out_gcm}")

# -------------------------
# PLOT: 12 subplots (6 rows x 2 cols), each month shows 3 SSP lines ±1σ
#       + annotate panel with slope (ensemble-mean slope)
# -------------------------
fig, axes = plt.subplots(6, 2, figsize=(10, 14), sharex=True, sharey=False)
axes = axes.ravel()

for ax, (m_lab, m_num) in zip(axes, months.items()):
    ax.set_title(m_lab)

    for ssp in ssps:
        ens = ens_cv[m_lab].get(ssp, None)
        if ens is None:
            continue
        x = ens.index.values
        mu = ens["mean"].values
        sd = ens["std"].values

        ax.plot(x, mu, lw=2.0, label=ssp.upper())
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.15)

    # annotate slope (use ensemble-mean slope per SSP)
    txt_lines = []
    for ssp in ssps:
        ens = ens_cv[m_lab].get(ssp, None)
        if ens is None:
            continue
        sl = theil_sen_slope(ens["mean"])["slope"]
        if np.isfinite(sl):
            txt_lines.append(f"{ssp.upper()}: {sl:+.4f}/yr")
    if txt_lines:
        ax.text(0.02, 0.05, "\n".join(txt_lines),
                transform=ax.transAxes, fontsize=8, va="bottom", ha="left")

    ax.grid(alpha=0.25)
    ax.set_xlim(year_min, year_max)
    ax.set_ylabel("CV")

# Legend once, at top
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)

plt.tight_layout()
plt.subplots_adjust(top=0.95)

out_png = os.path.join(out_dir, "CV_monthly_all12months_allSSP.png")
out_pdf = os.path.join(out_dir, "CV_monthly_all12months_allSSP.pdf")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.show()

print("Saved figure to:")
print(" ", out_png)
print(" ", out_pdf)

# -----------------------------------
# % CHANGE + TREND SUMMARY TABLE
# -----------------------------------

from scipy.stats import theilslopes

early_years = (2025, 2034)
late_years  = (2089, 2098)

rows = []

for month_label in results.keys():
    for ssp in ssps:
        ens = results[month_label].get(ssp, None)
        if ens is None:
            continue

        # time series
        yrs = ens.index.values
        cv  = ens["mean"].values

        # masks
        early_mask = (yrs >= early_years[0]) & (yrs <= early_years[1])
        late_mask  = (yrs >= late_years[0])  & (yrs <= late_years[1])

        early_mean = np.nanmean(cv[early_mask])
        late_mean  = np.nanmean(cv[late_mask])

        abs_change = late_mean - early_mean

        # % change (robust to small values)
        if early_mean > 0:
            pct_change = 100 * abs_change / early_mean
        else:
            pct_change = np.nan

        # Theil–Sen slope
        slope, intercept, lo, hi = theilslopes(cv, yrs)

        # % change per decade (relative to early mean)
        if early_mean > 0:
            pct_per_decade = (slope * 10 / early_mean) * 100
        else:
            pct_per_decade = np.nan

        rows.append({
            "Month": month_label,
            "SSP": ssp.upper(),
            "EarlyMean_CV": round(early_mean, 3),
            "LateMean_CV": round(late_mean, 3),
            "AbsChange_CV": round(abs_change, 3),
            "PctChange_CV_%": round(pct_change, 1),
            "TheilSenSlope_CV_per_year": round(slope, 5),
            "PctChange_per_decade_%": round(pct_per_decade, 2),
        })

cv_change_df = pd.DataFrame(rows)

# sort nicely
month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
cv_change_df["Month"] = pd.Categorical(cv_change_df["Month"], categories=month_order, ordered=True)
cv_change_df = cv_change_df.sort_values(["Month","SSP"])

# save
out_csv = os.path.join(out_dir, "CV_monthly_change_summary.csv")
cv_change_df.to_csv(out_csv, index=False)

print("\n✅ Monthly CV change summary saved to:")
print(out_csv)

# pretty print
print("\n=== Monthly CV % change summary ===")
print(cv_change_df.to_string(index=False))

#%% Just 3 month:import matplotlib.pyplot as plt

# -------------------------
# SETTINGS
# -------------------------
months_to_plot = [
    ("Apr", "(a)"),
    ("Jul", "(b)"),
]

ssp_colors = {
    "ssp126": "#1f77b4",  # blue
    "ssp370": "#ff7f0e",  # orange
    "ssp585": "#d62728"   # red
}

ssp_labels = {
    "ssp126": "SSP126",
    "ssp370": "SSP370",
    "ssp585": "SSP585"
}

# -------------------------
# FIGURE
# -------------------------
fig, axes = plt.subplots(
    nrows=2, ncols=1,
    figsize=(5, 3),
    sharex=True
)

for ax, (month_label, panel_label) in zip(axes, months_to_plot):

    for ssp in ["ssp126", "ssp370", "ssp585"]:
        ens = ens_cv[month_label].get(ssp, None)
        if ens is None:
            continue

        years = ens.index.values
        mean  = ens["mean"].values
        std   = ens["std"].values

        ax.plot(
            years, mean,
            color=ssp_colors[ssp],
            lw=0.8,
            label=ssp_labels[ssp]
        )
        ax.fill_between(
            years,
            mean - std,
            mean + std,
            color=ssp_colors[ssp],
            alpha=0.20,
            edgecolor="none"
        )

    # panel label
    ax.text(
        0.02, 0.95,
        panel_label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10
    )

    #ax.set_ylabel("CV of daily discharge (–)")
    ax.grid(alpha=0.3)
    ax.set_xlim(year_min, year_max)
    ax.set_ylim(0, 0.75)
# Figure-level horizontal legend
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles, labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.6, .95)
)
axes[-1].set_xlabel("Year")

fig.text(
    -0.01, 0.55,
    "CV of daily discharge (–)",
    va="center",
    rotation="vertical",
    fontsize=11
)
plt.tight_layout()
plt.savefig(
    f"{out_dir}/CV_monthly_Apr_Jul_allSSP.png",
    dpi=300, bbox_inches="tight"
)
plt.savefig(
    f"{out_dir}/CV_monthly_Apr_Jul_allSSP.pdf",
    dpi=300, bbox_inches="tight"
)
plt.show()

# %%
