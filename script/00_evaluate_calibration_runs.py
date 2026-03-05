# -*- coding: utf-8 -*-
"""

===============================================================================
SPHY Calibration Evaluation Script — Hydrology + Glacier Mass Balance
===============================================================================

Author:        Caroline Aubry-Wake
Affiliation:   University of Lethbridge
Date:          2025-10-27
Python:        3.9+

Description
-----------
This script evaluates hydrological and glacier mass balance model performance
for multiple SPHY calibration simulations. It loops over all simulation folders
in the specified base directory, computes performance metrics for streamflow,
snow, and soil moisture, and compares glacier mass balance to multiple
independent reference datasets. Results are compiled into a summary table and
visualized through diagnostic plots.

Evaluation Components
---------------------
1. **Hydrology Evaluation**
   - Streamflow NSE, KGE, RMSE, MARE, PBIAS vs measured daily discharge.
   - Snow depth NSE vs pluvio-derived weekly SWE (Nov–Apr, summer set to zero).
   - Soil moisture NSE vs normalized (10–90%) in situ soil moisture data.

2. **Glacier Mass Balance Evaluation**
   - Mass change derived from `iceDepth_*.map` files:
       • Full glacier average thickness change
       • Debris-covered area thickness change
   - Bias calculated relative to reference values from:
       • Ragettli et al. (2015) — dh for Langshisha Glacier (full + debris)
       • Dussaillant et al. (2019) — cumulative MB for 2011–2024
   - Compatible with additional time series from Rounce et al. (2023) if needed.

3. **Outputs**
   - CSV summary table of all hydrology and glacier MB metrics across runs.
   - Horizontal bar plots:
       • NSE comparison (streamflow, snow, soil)
       • MB bias comparison (Ragettli full vs debris)
   - Figures saved to `FIG_DIR`.

Data Inputs
-----------
- Measured hydrometeorological data:
  - Daily streamflow at Langshisha outlet
  - Snow depth (pluvio sensor, converted to SWE)
  - Soil moisture (3 probes, normalized)
- SPHY model outputs:
  - QAllDTS.tss, SnowSDTS.tss, RootwDTS.tss
  - ICE_DEPTH.csv (annual thickness)
  - iceDepth_YYYYMMDD.map (gridded thickness)
- Reference glacier mass balance datasets:
  - Dussaillant et al. (2023) cumulative MB time series
  - Ragettli et al. (2015) dh values for full and debris glacier zones
  - Glacier and debris masks (GeoTIFF rasters)

Usage
-----
1. Set the `BASE_DIR` to the parent directory containing calibration folders.
2. Set paths to measured data and reference datasets.
3. Run the script:
       $ python evaluate_calibration_runs.py
4. The script will:
   - Loop through each folder
   - Compute hydrology + MB metrics
   - Save results to CSV and figures to `FIG_DIR`

Notes
-----
- Time series are aligned to daily resolution; snow evaluated at weekly frequency.
- Snow observations outside accumulation season (May–Oct) are set to zero.
- Soil moisture is normalized by segment (pre- and post-gap).
- Glacier MB change computed as mean dh between 2011–2018 maps, divided by 7 years.
- Intended for calibration evaluation and ranking; not for final publication plots.

Outputs Created (written to FIG_DIR)
-----------------------------------
Tables:
- calibration_metrics_summary.csv
    Summary metrics for every calibration run:
    streamflow (full + low-flow), snow, soil, Ragettli dh biases, Dussaillant bias.

Figures:
- MB_selection_scatter.png
    Scatter of |Ragettli full bias| vs |Dussaillant bias| (annualized), colored by |Ragettli debris bias| (DDF runs only).
- MB_bias_barplot_DDF.png
    Stacked horizontal bars of absolute MB bias components (DDF runs only).
- Top5_DDF_Qall_Hydrographs.png
    Observed discharge vs top-5 DDF runs (selected by MB_Error).
- Hydro_score_FracGS_lowflow_barplot.png
    Bar plot of Hydro_Score ranking for FracGS runs (low-flow weighted).
- Hydrograph_Top5_FracGS_lowflow.png
    Observed discharge vs top-5 FracGS runs (selected by Hydro_Score).
- Hydro_metrics_FracGS_3col.png
    3-panel summary plot of streamflow metrics across FracGS runs (NSE/NSE_low, RMSE/RMSE_low, KGE/|PBIAS|).
- Hydrograph_Top3_FracGS_2014to2019.png
    Observed discharge vs top-3 FracGS runs (by Hydro_Score) for 2014–2019.
==============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import hydroeval as he
import re

# ============================================================
# 1. CONFIGURATION
# ============================================================
BASE_DIR = "/scratch/depfg/aubry001/out/calibgw_0"
FIG_DIR = "/scratch/depfg/aubry001/fig/fig_calib_eval/"
os.makedirs(FIG_DIR, exist_ok=True)

# Measured data (hydrology)
MEASURED_STREAMFLOW = "/scratch/depfg/aubry001/eval_data/Q_processed_20132018_CAW.csv"
MEASURED_SNOW = "/scratch/depfg/aubry001/eval_data/20231129_Pluvio_Langshisha.csv"
MEASURED_SOIL_2024 = "/scratch/depfg/aubry001/eval_data/20241108_LangshishaPluvio_soilmoisture.csv"

# Glacier MB reference datasets
DUSSAILLANT_PATH = "/scratch/depfg/aubry001/geospatial/dusaillant_massbalance/ASE_gla_MEAN-CAL-mass-change-series_obs_unobs.csv"
GLACID_MASK_TIF = "/scratch/depfg/aubry001/geospatial/ragettliglacier/glacierlangshisha_matchingRagettli.tif"
DEBRIS_MASK_TIF = "/scratch/depfg/aubry001/geospatial/ragettliglacier/langshishadebriscover.tif"
GLACIER_ID = "15"

# Reference values from Ragettli (m w.e./yr)
RAG_FULL_REF = -0.55
RAG_DEBRIS_REF = -1.30

# Snow filtering parameters
SNOW_SENSOR_HEIGHT_M = 2.65
SNOW_DENSITY_W_E = 0.50

# Model simulation starts on 2011-01-01 (for TSS alignment)
MODEL_START_DATE = pd.to_datetime("2011-01-01")

# Evaluation starts after spinup (e.g., Jan 1, 2014)
EVAL_START_DATE = pd.to_datetime("2014-01-01")
EVAL_END_DATE   = pd.to_datetime("2024-11-01")  # or whatever your end date is

low_flow_months = [11, 12, 1, 2, 3, 4]

# ============================================================
# 2. UTILITIES
# ============================================================
def compute_metrics(sim, obs):
    valid = ~np.isnan(sim) & ~np.isnan(obs)
    sim = sim[valid]
    obs = obs[valid]
    if len(sim) == 0:
        return {m: np.nan for m in ['NSE','KGE','RMSE','MARE','PBIAS']}
    return {
        'NSE': he.evaluator(he.nse, sim, obs).item(),
        'KGE': he.evaluator(he.kge, sim, obs)[0].item(),
        'RMSE': he.rmse(sim, obs).item(),
        'MARE': he.mare(sim, obs).item(),
        'PBIAS': he.pbias(sim, obs).item()
    }

def import_tss(year, month, day, file_path, colnames):
    import datetime
    start_date = datetime.datetime.strptime(f"{month}/{day}/{year}", "%m/%d/%Y")
    data_started = False
    first_offset = None
    rows = []
    dates = []

    with open(file_path, "r") as f:
        for raw in f:
            if not raw.strip():
                continue
            parts = raw.split()
            if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
                offset = int(parts[0])
                values = [float(x) for x in parts[1:]]
                if not data_started:
                    data_started = True
                    first_offset = offset
                day_offset = offset - first_offset
                dates.append(start_date + pd.Timedelta(days=day_offset))
                rows.append(values)
    df = pd.DataFrame(rows, columns=colnames[:len(rows[0])])
    df["datetime"] = dates
    df.set_index("datetime", inplace=True)
    return df

# ============================================================
# 3. LOAD MEASURED DATA
# ============================================================
print("Loading measured data...")

# --- Streamflow ---
q = pd.read_csv(MEASURED_STREAMFLOW, parse_dates=['T'], index_col='T')
q = q.rename(columns={'q_m3s3': 'Q'}).resample('D').mean()
q = q.where((q > 0) & (q < 15), np.nan)

# --- Snow ---
meas = pd.read_csv(MEASURED_SNOW)
meas['datetime'] = pd.to_datetime(meas['Date'] + ' ' + meas['Time'])
meas = meas.set_index('datetime')[['SnowD']]
meas = meas.resample('D').mean()
depth_m = (-meas['SnowD'] + SNOW_SENSOR_HEIGHT_M)
swe_mm = depth_m * 1000.0 * SNOW_DENSITY_W_E
snow_meas = pd.DataFrame({'SnowD': swe_mm})

# zero May–Oct
mask_melt = (snow_meas.index.month >= 5) & (snow_meas.index.month < 11)
snow_meas.loc[mask_melt, 'SnowD'] = 0
snow_meas_weekly = snow_meas['SnowD'].resample('W').mean().mask(lambda x: x == 0)

# --- Soil ---
meas = pd.read_csv(MEASURED_SOIL_2024)
meas['datetime'] = pd.to_datetime(meas['datetime'])
meas = meas.set_index('datetime')[['avg']]

gap_index = meas.index[meas['avg'].isna()].tolist()[0]
before = meas.loc[:gap_index]
after = meas.loc[gap_index + pd.DateOffset(days=1):]

before = before.loc[before.index < '2018-10-15']
p10, p90 = before.quantile(0.1), before.quantile(0.9)
scaled_before = ((before - p10) / (p90 - p10)).clip(0, 1)

p10, p90 = after.quantile(0.1), after.quantile(0.9)
scaled_after = ((after - p10) / (p90 - p10)).clip(0, 1)
scaled_after.loc["2022-03-06":"2022-04-10"] = np.nan

soil_meas = meas.copy()
soil_meas['scaled'] = pd.concat([scaled_before, scaled_after])
soil_meas['scaled'] = soil_meas['scaled'].rolling(window=5, center=True).mean()

# --- Dussaillant reference ---
df_dus = pd.read_csv(DUSSAILLANT_PATH)
rgi_id = "RGI60-15.04176"
df_dus = df_dus[df_dus["RGIId"] == rgi_id]
df_dus = df_dus.loc[:, df_dus.columns.str.match(r"^\d{4}$")]
df_dus = df_dus.loc[:, (df_dus.columns.astype(int) >= 2011) & (df_dus.columns.astype(int) <= 2023)]
df_dus = df_dus.apply(pd.to_numeric, errors="coerce") / 1000
df_dus = df_dus.T
df_dus.columns = ["Dussaillant_MB"]
df_dus["Cumulative_MB"] = df_dus["Dussaillant_MB"].cumsum()
df_dus["Cumulative_MB"] -= df_dus["Cumulative_MB"].iloc[0]
df_dus["Year"] = df_dus.index.astype(int)

# ============================================================
# 4. GLACIER MB FUNCTIONS
# ============================================================
def process_glacier_mass_ragettli(sim_folder):
    file_2011 = os.path.join(sim_folder, "iceDepth_20111001.map")
    file_2018 = os.path.join(sim_folder, "iceDepth_20181001.map")
    if not (os.path.exists(file_2011) and os.path.exists(file_2018)):
        return {k: np.nan for k in ['dh_full','dh_debris','bias_full','bias_debris']}

    with rasterio.open(file_2011) as src:
        icethickness_2011 = src.read(1)
    with rasterio.open(file_2018) as src:
        icethickness_2018 = src.read(1)
    dh = icethickness_2018 - icethickness_2011

    glac_mask = rasterio.open(GLACID_MASK_TIF).read(1)
    debris_mask = rasterio.open(DEBRIS_MASK_TIF).read(1)

    masked_full = np.ma.masked_where(glac_mask != 1, dh)
    masked_debris = np.ma.masked_where(debris_mask != 1, dh)

    dh_full = masked_full.mean() / 7.0
    dh_debris = masked_debris.mean() / 7.0
    bias_full = dh_full - RAG_FULL_REF
    bias_debris = dh_debris - RAG_DEBRIS_REF

    return {'dh_full': dh_full, 'dh_debris': dh_debris, 'bias_full': bias_full, 'bias_debris': bias_debris}

def process_glacier_mass_dussaillant(sim_folder):
    csv_path = os.path.join(sim_folder, "ICE_DEPTH.csv")
    if not os.path.exists(csv_path):
        return {'bias_2023': np.nan}

    try:
        df = pd.read_csv(csv_path, parse_dates=[0])
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        df_jan1 = df[df["Date"].dt.strftime("%m-%d") == "01-01"]

        # ✅ Check for required years and column
        if GLACIER_ID not in df_jan1.columns:
            print(f"⚠️ {sim_folder}: Glacier ID {GLACIER_ID} not found in ICE_DEPTH.csv")
            return {'bias_2023': np.nan}

        if (df_jan1["Date"].dt.year == 2011).sum() == 0 or (df_jan1["Date"].dt.year == 2023).sum() == 0:
            print(f"⚠️ {sim_folder}: missing Jan 1 data for 2011 or 2023")
            return {'bias_2023': np.nan}

        base_value = df_jan1[df_jan1["Date"].dt.year == 2011][GLACIER_ID].values[0]
        val_2023  = df_jan1[df_jan1["Date"].dt.year == 2023][GLACIER_ID].values[0]

        model_change = val_2023 - base_value
        obs_2023 = df_dus.loc[df_dus["Year"] == 2023, "Cumulative_MB"].values[0]

        return {'bias_2023': model_change - obs_2023}

    except Exception as e:
        print(f"❌ Dussaillant MB failed for {sim_folder}: {e}")
        return {'bias_2023': np.nan}

# ============================================================
# 5. HYDROLOGY EVALUATION
# ============================================================
VARIABLES  = ['Outlet', 'Outlet old', 'Well', 'AWS', 'Below Toe', 'Toe Moraine', 
              'Near lake', 'Mid Ice', 'Mid Moraine','Up Ice', 'Up Moraine', 
              'Head Ice', 'Top1','Top2','Top3']

def evaluate_hydrology(sim_folder):
    try:
        qall = import_tss('2011','01','01', os.path.join(sim_folder, "QAllDTS.tss"), VARIABLES)[['Outlet']]
        snowmod = import_tss('2011','01','01', os.path.join(sim_folder, "SnowSDTS.tss"), VARIABLES)[['AWS']]
        soilmod = import_tss('2011','01','01', os.path.join(sim_folder, "RootwDTS.tss"), VARIABLES)[['AWS']]

        # ======================================================
        # Streamflow metrics — full evaluation period
        # ======================================================
        aligned_q = qall.join(q, how='inner').loc[EVAL_START_DATE:EVAL_END_DATE].dropna()
        q_metrics = compute_metrics(aligned_q['Outlet'].values, aligned_q['Q'].values)

        # ======================================================
        # Streamflow metrics — LOW FLOW season only
        # ======================================================
        low_flow_months = [11, 12, 1, 2, 3, 4]
        low_flow_q = aligned_q[aligned_q.index.month.isin(low_flow_months)]
        q_low_metrics = compute_metrics(low_flow_q['Outlet'].values, low_flow_q['Q'].values)

        # ======================================================
        # Snow metrics
        # ======================================================
        aligned_snow = snowmod.join(snow_meas_weekly.to_frame('SnowD'), how='inner')
        aligned_snow = aligned_snow.loc[EVAL_START_DATE:EVAL_END_DATE].dropna()
        snow_metrics = compute_metrics(aligned_snow['AWS'].values, aligned_snow['SnowD'].values)

        # ======================================================
        # Soil metrics
        # ======================================================
        p10, p90 = soilmod.quantile(0.10), soilmod.quantile(0.90)
        soilmod['scaled'] = ((soilmod - p10) / (p90 - p10)).clip(0, 1)
        aligned_soil = soilmod[['scaled']].join(
            soil_meas[['scaled']], how='inner', lsuffix='_mod', rsuffix='_meas'
        )
        aligned_soil = aligned_soil.loc[EVAL_START_DATE:EVAL_END_DATE]
        aligned_soil = aligned_soil.where(
            (aligned_soil['scaled_mod'] >= 0.1) & (aligned_soil['scaled_meas'] >= 0.1)
        ).dropna()
        soil_metrics = compute_metrics(aligned_soil['scaled_mod'].values, aligned_soil['scaled_meas'].values)

        # return both full-period and low-flow metrics
        return q_metrics, q_low_metrics, snow_metrics, soil_metrics

    except Exception as e:
        print(f"  ❌ Hydrology failed for {sim_folder}: {e}")
        nan_metrics = {'NSE': np.nan,
                       'KGE': np.nan,
                       'RMSE': np.nan,
                       'MARE': np.nan,
                       'PBIAS': np.nan}
        return (nan_metrics.copy(), nan_metrics.copy(),
                nan_metrics.copy(), nan_metrics.copy())

# ============================================================
# 6. MAIN LOOP
# ============================================================
master_metrics = []

for folder in sorted(os.listdir(BASE_DIR)):
    fpath = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(fpath):
        continue
    print(f"--- Evaluating {folder} ---")

    # Hydrology
    q_metrics, q_low_metrics, snow_metrics, soil_metrics = evaluate_hydrology(fpath)

    # Glacier MB: Ragettli
    rag_metrics = process_glacier_mass_ragettli(fpath)

    # Glacier MB: Dussaillant
    dus_metrics = process_glacier_mass_dussaillant(fpath)

    master_metrics.append({
        'Simulation': folder,
        'Q_NSE': q_metrics['NSE'],
        'Q_KGE': q_metrics['KGE'],
        'Q_RMSE': q_metrics['RMSE'],
        'Q_PBIAS': q_metrics['PBIAS'],
        'Q_NSE_low': q_low_metrics['NSE'],
        'Q_RMSE_low': q_low_metrics['RMSE'],
        'Snow_NSE': snow_metrics['NSE'],
        'Snow_RMSE': snow_metrics['RMSE'],
        'Snow_PBIAS': snow_metrics['PBIAS'],
        'Soil_NSE': soil_metrics['NSE'],
        'Soil_RMSE': soil_metrics['RMSE'],
        'Soil_PBIAS': soil_metrics['PBIAS'],
        'Rag_Full_dh': rag_metrics['dh_full'],
        'Rag_Debris_dh': rag_metrics['dh_debris'],
        'Rag_Bias_Full': rag_metrics['bias_full'],
        'Rag_Bias_Debris': rag_metrics['bias_debris'],
        'Dussaillant_Bias_2023': dus_metrics['bias_2023']
    })

metrics_df = pd.DataFrame(master_metrics)
metrics_df = metrics_df.sort_values(by='Q_NSE', ascending=False)
metrics_df.to_csv(os.path.join(FIG_DIR, "calibration_metrics_summary.csv"), index=False)
print("\n=== Summary Table ===")
print(metrics_df.round(3).to_string(index=False))


def shorten_name(name):
    # Degree-day factors
    if name.startswith("GlacierMeltFactor"):
        nums = re.findall(r'[\d.]+', name)
        return "DDF" + ".".join(nums)

    # Fraction Glacier Snow
    elif name.startswith("GlacSnowFrac"):
        nums = re.findall(r'[\d.]+', name)
        # Remove leading "0." if present
        nums = [n[2:] if n.startswith("0.") else n for n in nums]
        return "FracGS" + ".".join(nums)

    # Fallback
    else:
        return name

metrics_df['ShortName'] = metrics_df['Simulation'].apply(shorten_name)

# ============================================================
# 7. Selecting MB simulations
# ============================================================
# Keep only DDF simulations
ddf_df = metrics_df[metrics_df['ShortName'].str.startswith("DDF")].copy()
ddf_df = ddf_df.dropna(subset=[
    'Rag_Bias_Full',
    'Rag_Bias_Debris',
    'Dussaillant_Bias_2023'
])
# Weights (example)
w1, w2, w3 = 0.3, 0.2, 0.5
n_years = 2023 - 2011

ddf_df['MB_Error'] = (
    w1 * ddf_df['Rag_Bias_Full'].abs() +
    w2 * ddf_df['Rag_Bias_Debris'].abs() +
    w3 * (ddf_df['Dussaillant_Bias_2023'].abs() / n_years)
)

# Sort ascending (best first)
ddf_df = ddf_df.sort_values(by='MB_Error', ascending=True)

# Select top 5
top_ddf = ddf_df.head(5)

print("=== Top 5 simulations based on MB fit ===")
print(top_ddf[['ShortName', 'Rag_Bias_Full', 'Rag_Bias_Debris', 'Dussaillant_Bias_2023', 'MB_Error']].round(3))


# ============================================================
# 7.1 Scatter plot of MB fit
# ============================================================
plt.figure(figsize=(10, 8))
x = ddf_df['Rag_Bias_Full'].abs()
n_years = 2023 - 2011
y = ddf_df['Dussaillant_Bias_2023'].abs() / n_years
c = ddf_df['Rag_Bias_Debris'].abs()
labels = ddf_df['ShortName']

sc = plt.scatter(x, y, c=c, cmap='viridis', s=80, edgecolor='k', alpha=0.85)
cbar = plt.colorbar(sc)
cbar.set_label('Ragettli Debris (m w.e./yr)', fontsize=12)
plt.xlabel('Ragettli Full Glacier Bias (m w.e./yr)', fontsize=14)
plt.ylabel('Dussaillant Bias (m w.e./yr)', fontsize=14)
plt.title('Glacier Mass Balance Evaluation', fontsize=15)

for xi, yi, lab in zip(x, y, labels):
    plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(5, 5),
                 ha='left', fontsize=8, alpha=0.8)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.xlim(0 ,0.4)
plt.savefig(os.path.join(FIG_DIR, "MB_selection_scatter.png"), dpi=300)
plt.show()


# ============================================================
# 7.2 Bar plot of MB error components
# ============================================================
x_labels = ddf_df['ShortName']
rag_full = ddf_df['Rag_Bias_Full'].abs()
rag_debris = ddf_df['Rag_Bias_Debris'].abs()
dussaillant = ddf_df['Dussaillant_Bias_2023'].abs()

plt.figure(figsize=(10, 12))
plt.barh(x_labels, rag_full, label='|Rag Full|', color='steelblue')
plt.barh(x_labels, rag_debris, left=rag_full, label='|Rag Debris|', color='lightcoral')
plt.barh(x_labels, dussaillant, left=rag_full+rag_debris, label='|Dussaillant|', color='orange')
plt.xlabel('Absolute Bias (m w.e.)', fontsize=14)
plt.ylabel('Simulation', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=8)
plt.title('Mass Balance Bias Components', fontsize=15)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "MB_bias_barplot_DDF.png"), dpi=300)
plt.show()


# ============================================================
# 7.3 Plot hydrograph of top 5 DDF runs
# ============================================================
def load_qall(sim_folder, base_dir):
    tss_path = os.path.join(base_dir, sim_folder, "QAllDTS.tss")
    df = import_tss('2011','01','01', tss_path, VARIABLES)
    return df[['Outlet']]

# Load observed discharge
q_meas = pd.read_csv(MEASURED_STREAMFLOW, parse_dates=['T'], index_col='T')
q_meas = q_meas.rename(columns={'q_m3s3': 'Q'}).resample('D').mean()
q_meas = q_meas.where((q_meas > 0) & (q_meas < 15), np.nan)

top_ddf_runs = top_ddf['Simulation'].tolist()

plt.figure(figsize=(14,6))
plt.plot(q_meas.index, q_meas['Q'], color='black', linewidth=2, label='Observed')

for sim in top_ddf_runs:
    qall = load_qall(sim, BASE_DIR)
    merged = qall.join(q_meas, how='inner')
    # Compute NSE to display in legend
    try:
        nse_val = he.evaluator(he.nse, merged['Outlet'].values, merged['Q'].values).item()
        rmse_val = he.rmse(merged['Outlet'].values, merged['Q'].values).item()
        label = f"{shorten_name(sim)} (NSE={nse_val:.2f})"
    except Exception:
        label = shorten_name(sim)
    plt.plot(merged.index, merged['Outlet'], label=label, alpha=0.8)

plt.xlabel('Date', fontsize=14)
plt.ylabel('Discharge (m³/s)', fontsize=14)
plt.title('Top 5 DDF Simulations — Streamflow Comparison', fontsize=16)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Top5_DDF_Qall_Hydrographs.png"), dpi=300)
plt.show()


# =================================
# 8. Select hydro simulation (with low-flow weighting)
# =================================

frac_df = metrics_df[metrics_df['ShortName'].str.startswith("FracGS")].copy()

# Weights — emphasizing low-flow NSE
w1, w2, w3, w4, w5, w6 = 0.25, 0.25, 0.3, 0.1, 0.05, 0.05

# Scale RMSE and PBIAS terms to [0,1]
frac_df['Q_RMSE_scaled'] = frac_df['Q_RMSE'] / frac_df['Q_RMSE'].max()
frac_df['Q_PBIAS_scaled'] = frac_df['Q_PBIAS'].abs() / frac_df['Q_PBIAS'].abs().max()
frac_df['Q_RMSE_low_scaled'] = frac_df['Q_RMSE_low'] / frac_df['Q_RMSE_low'].max()

# Composite Hydro Score with low-flow emphasis
frac_df['Hydro_Score'] = (
    w1 * frac_df['Q_KGE'] +
    w2 * frac_df['Q_NSE'] +
    w3 * frac_df['Q_NSE_low'] -
    w4 * frac_df['Q_RMSE_scaled'] -
    w5 * frac_df['Q_PBIAS_scaled'] -
    w6 * frac_df['Q_RMSE_low_scaled']
)

frac_df = frac_df.sort_values(by='Hydro_Score', ascending=False)
top_frac = frac_df.head(5)

print("=== Top 5 Fraction Glacier Snow simulations ===")
print(top_frac[['ShortName', 'Q_NSE', 'Q_NSE_low', 'Q_KGE', 'Q_RMSE', 'Q_RMSE_low', 'Q_PBIAS', 'Hydro_Score']].round(3))


plt.figure(figsize=(10, 6))
plt.barh(frac_df['ShortName'], frac_df['Hydro_Score'], color='steelblue')
plt.xlabel('Hydro Score', fontsize=14)
plt.ylabel('Simulation', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=8)
plt.title('Hydrology Performance Ranking — Fraction Runs', fontsize=15)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Hydro_score_FracGS_lowflow_barplot.png"), dpi=300)
plt.show()

q_meas = pd.read_csv(MEASURED_STREAMFLOW, parse_dates=['T'], index_col='T')
q_meas = q_meas.rename(columns={'q_m3s3': 'Q'}).resample('D').mean()

top5_frac = top_frac['Simulation'].tolist()

plt.figure(figsize=(14, 6))
plt.plot(q_meas.index, q_meas['Q'], color='black', label='Observed', linewidth=2)

for run in top5_frac:
    run_path = os.path.join(BASE_DIR, run, "QAllDTS.tss")
    df_sim = import_tss('2011','01','01', run_path, VARIABLES)
    df_sim = df_sim.loc[q_meas.index.min():q_meas.index.max()]
    plt.plot(df_sim.index, df_sim['Outlet'], label=shorten_name(run), alpha=0.8)

plt.xlabel('Date', fontsize=14)
plt.ylabel('Discharge (m³/s)', fontsize=14)
plt.title('Top 5 FracGS Simulations — Streamflow Comparison', fontsize=15)
plt.legend(fontsize=9, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Hydrograph_Top5_FracGS_lowflow.png"), dpi=300)
plt.show()

# Sort by hydro score (best first)
frac_df_plot = frac_df.copy().sort_values(by='Hydro_Score', ascending=False)
sim_names = frac_df_plot['ShortName']

# Create figure: 1 row, 3 columns
fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)

# === 1. NSE and NSE_low ===
axes[0].barh(sim_names, frac_df_plot['Q_NSE'], label='NSE', color='steelblue', alpha=0.8)
axes[0].barh(sim_names, frac_df_plot['Q_NSE_low'], label='NSE (Low Flow)', color='lightcoral', alpha=0.7)
axes[0].set_title("NSE vs NSE (Low Flow)", fontsize=13)
axes[0].set_xlabel("Efficiency", fontsize=12)
axes[0].invert_yaxis()
axes[0].legend()

# === 2. RMSE and RMSE_low ===
axes[1].barh(sim_names, frac_df_plot['Q_RMSE'], label='RMSE', color='darkorange', alpha=0.8)
axes[1].barh(sim_names, frac_df_plot['Q_RMSE_low'], label='RMSE (Low Flow)', color='gold', alpha=0.7)
axes[1].set_title("RMSE vs RMSE (Low Flow)", fontsize=13)
axes[1].set_xlabel("RMSE (m³/s)", fontsize=12)
axes[1].legend()

# === 3. KGE and |PBIAS| ===
axes[2].barh(sim_names, frac_df_plot['Q_KGE'], label='KGE', color='seagreen', alpha=0.8)
axes[2].barh(sim_names, frac_df_plot['Q_PBIAS'].abs(), label='|PBIAS|', color='lightgreen', alpha=0.7)
axes[2].set_title("KGE vs |PBIAS|", fontsize=13)
axes[2].set_xlabel("Metric Value", fontsize=12)
axes[2].legend()

plt.suptitle("Streamflow Evaluation Metrics — Fraction Runs", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(FIG_DIR, "Hydro_metrics_FracGS_3col.png"), dpi=300)
plt.show()


# Reload observed discharge
q_meas = pd.read_csv(MEASURED_STREAMFLOW, parse_dates=['T'], index_col='T')
q_meas = q_meas.rename(columns={'q_m3s3': 'Q'}).resample('D').mean()
q_meas = q_meas.where((q_meas > 0) & (q_meas < 15), np.nan)
q_meas = q_meas.loc['2014-01-01':'2019-12-31']

# Sort simulations by Hydro Score
frac_df_plot = frac_df.copy().sort_values(by='Hydro_Score', ascending=False)
top3 = frac_df_plot.head(3)

# Prepare plotting figure: 21 rows × 1 column
fig, ax = plt.subplots(1, 1, figsize=(8, 4), sharex=True)
# ===============================
# Top 3 Simulations
# ===============================
ax.plot(q_meas.index, q_meas['Q'], color='black', linewidth=2, label='Observed Q')
for _, row in top3.iterrows():
    sim = row['Simulation']
    short = shorten_name(sim)
    run_path = os.path.join(BASE_DIR, sim, "QAllDTS.tss")
    df_sim = import_tss('2011','01','01', run_path, VARIABLES)
    df_sim = df_sim.loc['2014-01-01':'2019-12-31']
    ax.plot(df_sim.index, df_sim['Outlet'], label=f"{short} (Score={row['Hydro_Score']:.2f})", alpha=0.8)
ax.set_title("Top 3 Simulations — Streamflow (2014–2019)", fontsize=14)
ax.set_ylabel("Discharge (m³/s)", fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)

plt.suptitle("SPHY Calibration — Streamflow Performance (Top vs Worst Runs)", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(FIG_DIR, "Hydrograph_Top3_FracGS_2014to2019.png"), dpi=300)
plt.show()

# Identify best simulation (highest Hydro_Score)
best_run = frac_df.loc[frac_df['Hydro_Score'].idxmax()]

# Print summary for the best simulation
print("\n=== Best Simulation Summary ===")
print(f"Simulation: {best_run['Simulation']}")
print(f"ShortName: {best_run['ShortName']}")
print("\n--- Hydrology Metrics ---")
print(f"NSE:           {best_run['Q_NSE']:.3f}")
print(f"NSE (Low):     {best_run['Q_NSE_low']:.3f}")
print(f"KGE:           {best_run['Q_KGE']:.3f}")
print(f"RMSE:          {best_run['Q_RMSE']:.3f}")
print(f"RMSE (Low):    {best_run['Q_RMSE_low']:.3f}")
print(f"PBIAS:         {best_run['Q_PBIAS']:.3f}")
print(f"\nHydro Score (weighted): {best_run['Hydro_Score']:.3f}")