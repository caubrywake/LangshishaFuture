# -*- coding: utf-8 -*-
"""
Basin-mean temperature time series and SSP ensemble statistics

Author: Caroline Aubry-Wake
Last updated: 2025-11-20

Purpose
-------
Compute daily basin-mean time series from gridded climate forcing rasters
for multiple GCM–SSP combinations, then build daily ensemble mean and
standard deviation time series per SSP. Designed as a preprocessing step
for later annual/seasonal statistics and trend analysis.

Inputs
------
Directory layout (daily rasters):
    base_dir = "/scratch/depfg/aubry001/minp/ds"
    Structure:
        {base_dir}/{gcm}/{ssp}/{var}/<daily raster files>

Example:
    /scratch/depfg/aubry001/minp/ds/mri-esm2-0/ssp126/tavg/*.tif

Basin mask:
    footprint_file = "/scratch/depfg/aubry001/geospatial/Langshisha_basin_outline/footprint.tif"
    - Non-zero cells define the basin; used to mask each raster.

Variables:
    variables = ["tavg", "tmin", "tmax"]
    (add "prec" to include precipitation in the same workflow)

Time axis:
    start_date = datetime(2015, 1, 1)
    - Files in each {gcm}/{ssp}/{var} directory are assumed to be daily,
      consecutive, and sorted in time; index i → start_date + i days.

Processing
----------
For each variable in `variables`:
    1) Loop over all GCM directories in `base_dir`.
    2) For each GCM, loop over all SSP subdirectories.
    3) For each {gcm, ssp, var}:
           - Read all raster files in {gcm}/{ssp}/{var}, excluding *.aux.xml.
           - Apply the basin mask and compute basin-mean for each day
             (nan-masked outside basin).
           - Build a daily time series DataFrame:
                 columns: ["date", var]
           - Save per-GCM, per-SSP CSV:
                 "{var}_{gcm}_{ssp}.csv" in `out_dir`.

    4) For each SSP:
           - Align all GCM time series on common dates (inner join).
           - Compute daily ensemble mean and standard deviation across GCMs.
           - Save ensemble CSV:
                 "{var}_{ssp}_ensemble.csv"
           - Produce a quick-look plot of ensemble mean ± 1σ over time.

Outputs
-------
Output directory:
    out_dir = "/scratch/depfg/aubry001/out/basin_mean_temp"

Per-GCM, per-SSP daily basin-mean CSV:
    "{var}_{gcm}_{ssp}.csv"

Per-SSP ensemble CSV:
    "{var}_{ssp}_ensemble.csv"
    with columns:
        - "date"
        - "{var}_mean" (ensemble daily mean)
        - "{var}_std"  (ensemble daily standard deviation)

Per-SSP ensemble plots:
    "{var}_{ssp}_ensemble.png"
    - Daily ensemble mean with ±1σ shading.

"""

import os
import glob
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# === USER SETTINGS ===
base_dir = "/scratch/depfg/aubry001/minp/ds"
footprint_file = "/scratch/depfg/aubry001/geospatial/Langshisha_basin_outline/footprint.tif"
out_dir = "/scratch/depfg/aubry001/out/basin_mean_temp"
variables = ["tavg", "tmin", "tmax"]   # add "prec" here if you want precip too
start_date = datetime(2015, 1, 1)

os.makedirs(out_dir, exist_ok=True)

# === Load basin mask ===
with rasterio.open(footprint_file) as src:
    basin_mask = src.read(1)
mask = basin_mask > 0


# === MAIN LOOP: per variable ===
gcms = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for var in variables:
    print(f"\n=== Variable: {var} ===")
    # store per-SSP list of DataFrames (one per GCM) for ensemble calc
    ssp_series = {}

    for gcm in gcms:
        gcm_dir = os.path.join(base_dir, gcm)
        ssps = [d for d in os.listdir(gcm_dir) if os.path.isdir(os.path.join(gcm_dir, d))]

        for ssp in ssps:
            var_dir = os.path.join(gcm_dir, ssp, var)
            if not os.path.isdir(var_dir):
                print(f"  ⚠️ {var_dir} does not exist, skipping.")
                continue

            print(f"→ {gcm} — {ssp} — {var}")
            # grab all raster forcings in that var directory
            files = sorted(glob.glob(os.path.join(var_dir, "*")))
            # 🔧 NEW: filter out aux.xml (and any other obvious non-raster sidecars if you want)
            files = [f for f in files if not f.lower().endswith(".aux.xml")]
            
            if not files:
                print(f"  ⚠️ No files found in {var_dir}")
                continue

            dates = []
            values = []

            # enumerate files in sorted order -> use index as day offset
            for i, f in enumerate(tqdm(files, desc=f"{gcm}-{ssp}-{var}", ncols=80)):
                date = start_date + timedelta(days=i)

                with rasterio.open(f) as src:
                    arr = src.read(1)

                arr = np.where(mask, arr, np.nan)
                mean_val = np.nanmean(arr)
                dates.append(date)
                values.append(mean_val)

            if not dates:
                print(f"  ⚠️ No valid dates parsed for {var_dir}")
                continue

            df = pd.DataFrame({"date": dates, var: values}).sort_values("date")
            df.reset_index(drop=True, inplace=True)

            # save per-GCM, per-SSP CSV
            out_csv = os.path.join(out_dir, f"{var}_{gcm}_{ssp}.csv")
            df.to_csv(out_csv, index=False, float_format="%.4f")
            print(f"  ✅ Saved {out_csv} ({len(df)} days)")

            # store for ensemble
            df_for_ens = df.set_index("date")
            ssp_series.setdefault(ssp, []).append(df_for_ens)

    # === ENSEMBLE MEAN + STD PER SSP ===
    for ssp, df_list in ssp_series.items():
        # align on common dates (inner join)
        combined = pd.concat(df_list, axis=1, join="inner")
        combined.columns = [f"{var}_{i}" for i in range(len(df_list))]

        ens_mean = combined.mean(axis=1)
        ens_std = combined.std(axis=1)

        out_df = pd.DataFrame({
            "date": ens_mean.index,
            f"{var}_mean": ens_mean.values,
            f"{var}_std": ens_std.values
        })

        ens_csv = os.path.join(out_dir, f"{var}_{ssp}_ensemble.csv")
        out_df.to_csv(ens_csv, index=False, float_format="%.4f")
        print(f"  📊 Saved ensemble: {ens_csv}")

        # quick plot
        plt.figure(figsize=(10, 4))
        plt.plot(out_df["date"], out_df[f"{var}_mean"], label=f"{ssp} mean")
        plt.fill_between(
            out_df["date"],
            out_df[f"{var}_mean"] - out_df[f"{var}_std"],
            out_df[f"{var}_mean"] + out_df[f"{var}_std"],
            alpha=0.3,
            label="±1σ"
        )
        plt.title(f"{var.upper()} basin-mean ensemble — {ssp}")
        plt.xlabel("Date")
        ylabel = f"{var} (°C)" if var.startswith("t") else var
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(out_dir, f"{var}_{ssp}_ensemble.png")
        plt.savefig(fig_path, dpi=200)
        plt.show()

print("\n🎉 Done: basin-mean time series and SSP ensemble stats generated.")
