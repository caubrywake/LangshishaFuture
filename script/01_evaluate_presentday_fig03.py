# -*- coding: utf-8 -*-

"""
===============================================================================
SPHY Model Evaluation Script
===============================================================================
Author:        Caroline Aubry-Wake
Affiliation:   University of Lethbridge
Created:       2025-10-27
Python:        3.9+ (tested)
Dependencies:  numpy, pandas, matplotlib, hydroeval, rasterio

Description:
------------
This script evaluates the outputs of the SPHY hydrological model against 
measured data. This script reproduces Fig. 3 for the selected calibrated run.

The script:
    - Imports and processes SPHY .tss output files for discharge components,
      snow depth, and soil moisture.
    - Loads measured streamflow, snow depth, and soil moisture observations.
    - Calculates performance metrics (NSE, KGE, RMSE, MARE, PBIAS).
    - Produces diagnostic plots comparing modelled vs. observed time series.

Inputs:
-------
- BASE_DIR: Directory containing SPHY output folders.
- MEASURED_STREAMFLOW: CSV file with measured discharge time series.
- MEASURED_SNOW: CSV file with measured snow depth time series.
- MEASURED_SOIL: CSV file with measured soil moisture time series.
- FOLDER: Folder name for single run mode.

Outputs:
--------
- Performance metrics printed to console.
- Time series plots of discharge components, snow depth, and soil moisture.

Usage:
------
1. Set the configuration paths and flags in Section 1.
2. Run the script:
       $ python evaluate_sphy.py
3. To run sensitivity mode, set `SENSITIVITY_FLAG = 1`.

Notes:
------
- The `import_tss()` utility automatically handles variable header lengths in 
  .tss files and aligns timesteps to a user-defined start date.
- Performance metrics are computed only on overlapping non-NaN values.
- Plotting uses daily-resampled observed data.

===============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hydroeval as he
import datetime
import rasterio
import matplotlib as mpl

# =============================
# 1. CONFIGURATION
# =============================
# Base path for model outputs
BASE_DIR = "/scratch/depfg/aubry001/out/calibgw_0"

# Measured data paths
MEASURED_STREAMFLOW = "/scratch/depfg/aubry001/eval_data/Q_processed_20132018_CAW.csv"
MEASURED_SNOW = "/scratch/depfg/aubry001/eval_data/20231129_Pluvio_Langshisha.csv"
MEASURED_SOIL = "/scratch/depfg/aubry001/eval_data/20231208_LangshishaPluvio_soilmoisture.csv"
MEASURED_SOIL_2024 = "/scratch/depfg/aubry001/eval_data/20241108_LangshishaPluvio_soilmoisture.csv"

# Where you want to save plots (e.g. figures)
FIG_DIR = "/scratch/depfg/aubry001/eval_data/figures"

FOLDER = "GlacSnowFrac_0.7_0.6.cfg"   # for best run

START_DATE = pd.to_datetime('2011-01-01')
END_DATE   = pd.to_datetime('2024-11-01')
VARIABLES  = ['Outlet', 'Outlet old', 'Well', 'AWS', 'Below Toe', 'Toe Moraine', 
              'Near lake', 'Mid Ice', 'Mid Moraine','Up Ice', 'Up Moraine', 
              'Head Ice', 'Top1','Top2','Top3']



mpl.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 14,
    "lines.linewidth": 1
})
# =============================
# 1b. MEASURED DATA PATHS (scratch)
# =============================
# Snow (pluvio: Date, Time, SnowD)
MEASURED_SNOW_RAW = "/scratch/depfg/aubry001/eval_data/20231129_Pluvio_Langshisha.csv"

# Soil (legacy) — Date/Time separate columns with SM1..SM3
MEASURED_SOIL_OLD = "/scratch/depfg/aubry001/eval_data/20231208_LangshishaPluvio_soilmoisture.csv"
# Soil (2024 consolidated) — 'datetime' + 'avg'
MEASURED_SOIL_2024 = "/scratch/depfg/aubry001/eval_data/20241108_LangshishaPluvio_soilmoisture.csv"

# Tuning knobs
SNOW_SENSOR_HEIGHT_M = 2.7  #2.65       # pluvio head height (m)
SNOW_DENSITY_W_E = 0.50           # 0.5 = ~500 kg/m3 -> mm w.e. factor
SOIL_ROLLING_WINDOW = 5           # days for smoothing
SOIL_SPIKE_START = "2022-03-06"
SOIL_SPIKE_END   = "2022-04-10"

# =============================
# 2. UTILITIES
# =============================

def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def import_tss(year, month, day, file_path, colnames):
    """
    Import SPHY .tss as a DataFrame with a datetime index.
    - Works regardless of header length (e.g., lines '16', 'timestep', '1..15').
    - Normalizes so the first data row aligns with the provided start date
      (i.e., treats the first timestep as day 0).
    """
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

            # A real data row should have at least 2 tokens:
            # [timestep, value1, value2, ...]
            # The header numeric lines (like "16" or "1", "2", … "15") have only 1 token.
            if len(parts) >= 2 and parts[0].lstrip("-").isdigit() and _is_float(parts[1]):
                offset = int(parts[0])
                values = [float(x) for x in parts[1:]]

                # Only now mark data start and record the first offset
                if not data_started:
                    data_started = True
                    first_offset = offset  # e.g., 1 in your file

                # Normalize so first row corresponds to day 0
                day_offset = offset - first_offset  # makes first row = 0
                dates.append(start_date + datetime.timedelta(days=day_offset))
                rows.append(values)

            # Otherwise it's header/junk; skip

    # Build DataFrame
    df = pd.DataFrame(rows, columns=colnames[:len(rows[0])])
    df["datetime"] = dates
    df.set_index("datetime", inplace=True)
    return df


def process_measured_snow(
    fn: str,
    sensor_height_m: float = SNOW_SENSOR_HEIGHT_M,
    density_we_factor: float = SNOW_DENSITY_W_E
) -> pd.DataFrame:
    """
    Read pluvio snow file with 'Date','Time','SnowD' (SnowD is distance-to-surface, m),
    convert to daily mm w.e.
    """
    meas = pd.read_csv(fn)
    meas['datetime'] = pd.to_datetime(meas['Date'] + ' ' + meas['Time'])
    meas = meas.set_index('datetime')[['SnowD']]

    # Daily mean
    meas = meas.resample('D').mean()

    # Convert to snow depth (m) from distance-to-surface, then to mm w.e.
    # depth = (sensor_height - distance)  -> your code: (-SnowD + 2.65)
    depth_m = (-meas['SnowD'] + sensor_height_m)
    swe_mm = depth_m * 1000.0 * density_we_factor

    out = pd.DataFrame({'SnowD': swe_mm})
    return out


def _normalize_segment(df: pd.DataFrame, col: str = 'avg') -> pd.DataFrame:
    """Quantile (10–90%) normalization with clipping to [0,1]."""
    p10 = df[col].quantile(0.10)
    p90 = df[col].quantile(0.90)
    norm = (df[col] - p10) / (p90 - p10)
    return pd.DataFrame({'scaled': norm.clip(0, 1)})


def process_measured_soil(
    fn_old: str,
    fn_2024: str,
    rolling_window: int = SOIL_ROLLING_WINDOW,
    spike_start: str = SOIL_SPIKE_START,
    spike_end: str = SOIL_SPIKE_END
) -> pd.DataFrame:
    """
    Merge legacy soil-moisture file (Date/Time + SM1..SM3) with 2024 file (datetime + avg),
    then normalize in two sections split at the first long NaN gap, remove a known spike,
    and smooth with a centered rolling mean.
    Returns a DataFrame with columns ['avg','scaled'] at daily frequency.
    """

    # --- Load OLD format ---
    old = pd.read_csv(fn_old)
    old['datetime'] = pd.to_datetime(old['Date'] + ' ' + old['Time'])
    old = old.set_index('datetime')
    old['avg'] = old[['SM1', 'SM2', 'SM3']].mean(axis=1)
    old = old[['avg']].resample('D').mean()

    # --- Load 2024 consolidated (if available) ---
    try:
        new = pd.read_csv(fn_2024)
        new['datetime'] = pd.to_datetime(new['datetime'])
        new = new.set_index('datetime')[['avg']].resample('D').mean()
        # Prefer newer data where present
        merged = old.combine_first(new).combine_first(old)
        # If 2024 truly supersedes overlap, you could do: merged.update(new)
        merged.update(new)
    except Exception:
        merged = old

    # Identify a gap to split normalization (use first NaN run ≥ 3 days, else fallback)
    is_nan = merged['avg'].isna()
    if is_nan.any():
        # Find the first NaN index; robustly search for a run ≥ 3 days
        run_start = None
        run_len = 0
        split_idx = None
        for ts, nan_ in is_nan.items():
            if nan_:
                if run_start is None:
                    run_start = ts
                    run_len = 1
                else:
                    run_len += 1
            else:
                if run_start is not None and run_len >= 3:
                    split_idx = run_start
                    break
                run_start = None
                run_len = 0
        # Fallback: take first NaN if no long run found
        if split_idx is None:
            split_idx = is_nan[is_nan].index[0]
    else:
        split_idx = None

    if split_idx is not None:
        before = merged.loc[:split_idx]
        after  = merged.loc[split_idx + pd.DateOffset(days=1):]
        norm_before = _normalize_segment(before, 'avg')
        norm_after  = _normalize_segment(after, 'avg')
        scaled = pd.concat([norm_before, norm_after])
    else:
        scaled = _normalize_segment(merged, 'avg')

    # Remove known spike window (optional but matches your logic)
    scaled.loc[pd.to_datetime(spike_start):pd.to_datetime(spike_end), 'scaled'] = np.nan

    # Smooth
    scaled['scaled'] = scaled['scaled'].rolling(window=rolling_window, center=True, min_periods=1).mean()

    out = merged.join(scaled, how='left')
    return out

def compute_metrics(sim, obs):
    """Compute NSE, KGE, RMSE, MARE, PBIAS between two arrays."""
    valid = ~np.isnan(sim) & ~np.isnan(obs)
    sim = sim[valid]; obs = obs[valid]
    if len(sim) == 0:
        return {m: np.nan for m in ['NSE','KGE','RMSE','MARE','PBIAS']}
    nse = he.evaluator(he.nse, sim, obs).item()
    kge = he.evaluator(he.kge, sim, obs)[0].item()
    return {
        'NSE': nse,
        'KGE': kge,
        'RMSE': he.rmse(sim, obs).item(),
        'MARE': he.mare(sim, obs).item(),
        'PBIAS': he.pbias(sim, obs).item()
    }

def load_measured_data():
    """
    Load measured streamflow, snow depth (as mm w.e.), and soil moisture (avg + scaled).
    Streamflow is daily mean and lightly range-filtered; snow uses density conversion;
    soil uses section-wise normalization with smoothing.
    """
    # --- Streamflow ---
    q = pd.read_csv(MEASURED_STREAMFLOW, parse_dates=['T'], index_col='T')
    q = q.rename(columns={'q_m3s3': 'Q'}).resample('D').mean()
    # Light QC for crazy values (adjust as needed)
    q = q.where((q > 0) & (q < 15), np.nan)

    # --- Snow ---
    snow = process_measured_snow(MEASURED_SNOW_RAW)
     # ----- Clean measured snow for plotting (seasonal zeroing + weekly resample) -----
    snowmeas_filtered = snow.copy()

    # Extract month for masking
    months = snowmeas_filtered.index.month

    # Set snow to zero outside accumulation season (Nov–Apr)
    mask_melt = (months >= 5) & (months < 11)  # May through October inclusive
    snowmeas_filtered.loc[mask_melt, 'SnowD'] = 0

    # Weekly resample and remove 0 values
    snowmeas_weekly = snowmeas_filtered['SnowD'].resample('W').mean()
    snowmeas_weekly = snowmeas_weekly.mask(snowmeas_weekly == 0)

    # --- Soil ---
    soil = process_measured_soil(MEASURED_SOIL_OLD, MEASURED_SOIL_2024)
    return q, snow, snowmeas_weekly, soil

# =============================
# 3. PROCESSING
# =============================
def evaluate_folder(path, streamflow_meas, soil_meas, snow_meas_weekly):
    """Evaluate model outputs for a single folder."""
    results = {}

    # --- Streamflow components ---
    qall = import_tss('2011','01','01', os.path.join(path, "QAllDTS.tss"), VARIABLES)[['Outlet']]
    glac = import_tss('2011','01','01', os.path.join(path, "GTotDTS.tss"), VARIABLES)[['Outlet']]
    rain = import_tss('2011','01','01', os.path.join(path, "RTotDTS.tss"), VARIABLES)[['Outlet']]
    snow = import_tss('2011','01','01', os.path.join(path, "STotDTS.tss"), VARIABLES)[['Outlet']]
    base = import_tss('2011','01','01', os.path.join(path, "BTotDTS.tss"), VARIABLES)[['Outlet']]

    # --- Metrics: streamflow ---
    mod_cut = qall.loc[START_DATE:END_DATE]
    meas_cut = streamflow_meas.loc[START_DATE:END_DATE]
    aligned = mod_cut.join(meas_cut, how='inner').dropna()
    results['streamflow_metrics'] = compute_metrics(aligned['Outlet'].values, aligned['Q'].values)

    # --- Soil moisture ---
    soilmod = import_tss('2011','01','01', os.path.join(path, "RootwDTS.tss"), VARIABLES)[['AWS']]
    soilmod.iloc[0:7] = soilmod.iloc[8]  # your warm-up fix
    p10, p90 = soilmod.quantile(0.10), soilmod.quantile(0.90)
    soilmod['scaled'] = ((soilmod - p10) / (p90 - p10)).clip(0, 1)

    # Compare scaled (model) to scaled (measured)
    aligned = soilmod[['scaled']].join(
        soil_meas[['scaled']],
        how='inner',
        lsuffix='_mod',
        rsuffix='_meas'
    ).where(
        (lambda x: (x['scaled_mod'] >= 0.1) & (x['scaled_meas'] >= 0.1))
    ).dropna()
    results['soil_metrics'] = compute_metrics(
        aligned['scaled_mod'].values,
        aligned['scaled_meas'].values
    )
    # --- Snow depth ---
    # --- Snow SWE (AWS) ---
    # Keep a copy named 'AWS' for plotting consistency
    snowmod_all = import_tss('2011', '01', '01', os.path.join(path, "SnowSDTS.tss"), VARIABLES)

    snowmod = snowmod_all[['AWS']].copy()               # for plotting
    results['snowmod'] = snowmod                        # expected by plot_single_run()

    # --- Snow SWE metrics (weekly, only when measured weekly SWE exists, accumulation season only) ---
    # Use the WEEKLY observed series passed into evaluate_folder: snow_meas_weekly

    # Ensure obs is a Series named SWE_obs
    if isinstance(snow_meas_weekly, pd.Series):
        swe_obs_weekly = snow_meas_weekly.rename('SWE_obs')
    elif isinstance(snow_meas_weekly, pd.DataFrame):
        swe_obs_weekly = snow_meas_weekly.iloc[:, 0].rename('SWE_obs')
    else:
        raise TypeError("snow_meas_weekly must be a pandas Series or DataFrame of weekly SWE observations.")

    # Model weekly SWE (match the exact timestamps of obs by reindexing)
    swe_mod_weekly = snowmod['AWS'].resample('W').mean().rename('SWE_mod')

    snow_aligned = pd.concat([swe_obs_weekly, swe_mod_weekly], axis=1)

    # Accumulation season mask (Nov–Apr) on the WEEKLY timestamps
    months = snow_aligned.index.month
    is_accum = (months >= 11) | (months <= 4)

    # Apply mask + evaluation window + require both present
    snow_aligned = snow_aligned.loc[is_accum]
    snow_aligned = snow_aligned.loc[START_DATE:END_DATE]
    snow_aligned = snow_aligned.dropna(subset=['SWE_obs', 'SWE_mod'])

    snow_metrics_all = compute_metrics(
        snow_aligned['SWE_mod'].values,
        snow_aligned['SWE_obs'].values
    )

    results['snow_metrics'] = {
        'RMSE': snow_metrics_all['RMSE'],
        'PBIAS': snow_metrics_all['PBIAS'],
        'n_weeks': int(len(snow_aligned))
    }

    results['qall'] = qall
    results['glac'] = glac
    results['rain'] = rain
    results['snow'] = snow
    results['base'] = base
    results['soilmod'] = soilmod
    results['snowmod'] = snowmod

    return results

# =============================
# 4. PLOTTING
# =============================
def plot_single_run(results, streamflow_meas, snow_meas_weekly, soil_meas):
    """
    - Uses processed weekly snow (snow_meas_weekly)
    - Only plots where observations exist (gaps show as breaks)
    """

    qall    = results['qall'][['Outlet']].rename(columns={'Outlet': 'Q_model'})
    snowmod = results['snowmod'][['AWS']].rename(columns={'AWS': 'Snow_model'})
    soilmod = results['soilmod'][['scaled']].rename(columns={'scaled': 'Soil_model'})
    xmin = pd.to_datetime("2013-02-01")
    xmax = pd.to_datetime("2024-12-01")
    # -----------------------------
    # 1) STREAMFLOW (model vs obs)
    #    -> restrict to observed dates
    # -----------------------------
    q_obs = streamflow_meas[['Q']]
    q_join = pd.DataFrame(index=q_obs.index)
    q_join['Q_obs']   = q_obs['Q']
    q_join['Q_model'] = qall['Q_model'].reindex(q_join.index)

    # -----------------------------
    # 2) SNOW (processed weekly, obs index drives everything)
    # -----------------------------
    if isinstance(snow_meas_weekly, pd.Series):
        snow_obs = snow_meas_weekly.to_frame(name='Snow_obs')
    else:
        snow_obs = snow_meas_weekly.rename(columns={snow_meas_weekly.columns[0]: 'Snow_obs'})

    snow_join = pd.DataFrame(index=snow_obs.index)
    snow_join['Snow_obs']   = snow_obs['Snow_obs']
    snow_join['Snow_model'] = snowmod['Snow_model'].reindex(snow_join.index)

    # -----------------------------
    # 3) SOIL (normalized, obs index drives everything)
    # -----------------------------
    soil_obs = soil_meas[['scaled']].rename(columns={'scaled': 'Soil_obs'})
    soil_join = pd.DataFrame(index=soil_obs.index)
    soil_join['Soil_obs']   = soil_obs['Soil_obs']
    soil_join['Soil_model'] = soilmod['Soil_model'].reindex(soil_join.index)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, axs = plt.subplots(3, 1, figsize=(9, 5), sharex=True)

    # (a) Streamflow
    ax = axs[0]
    ax.plot(q_join.index, q_join['Q_model'], label='Model', linewidth=1.5)
    ax.plot(q_join.index, q_join['Q_obs'],   label='Observed', linewidth=2.0, color='black')
    ax.set_ylabel('Discharge (m³/s)')
    ax.text(0.01, 0.95, '(a)', transform=ax.transAxes,
            va='top', ha='left', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=11, frameon=False)
    ax.set_xlim([xmin, xmax])
    # (b) Snow
    ax = axs[1]
    ax.plot(snow_join.index, snow_join['Snow_model'], label='Model', linewidth=1.5)
    ax.plot(snow_join.index, snow_join['Snow_obs'],   label='Observed (weekly)',
            linewidth=2.0, color='black')
    ax.set_ylabel('Snow (mm w.e.)')
    ax.text(0.01, 0.95, '(b)', transform=ax.transAxes,
            va='top', ha='left', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=11, frameon=False)
    ax.set_xlim([xmin, xmax])
    # (c) Soil
    ax = axs[2]
    ax.plot(soil_join.index, soil_join['Soil_model'], label='Model', linewidth=1.5)
    ax.plot(soil_join.index, soil_join['Soil_obs'],   label='Observed', linewidth=2.0,
            color='black')
    ax.set_ylabel('Soil moisture (norm.)')
    ax.set_xlabel('Date')
    ax.text(0.01, 0.95, '(c)', transform=ax.transAxes,
            va='top', ha='left', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='center left', fontsize=11, frameon=False)
    ax.set_xlim([xmin, xmax])
    fig.tight_layout()
    # Ensure save directory exists
    os.makedirs(FIG_DIR, exist_ok=True)

    # Save figure
    fig_path = os.path.join(FIG_DIR, f"F03_evaluation.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    fig_path = os.path.join(FIG_DIR, f"F03_evaluation.pdf")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {fig_path}")
    plt.show()

# =============================
# 5. MAIN SCRIPT
# =============================
if __name__ == "__main__":
    streamflow_meas, snow_meas, snow_meas_weekly, soil_meas = load_measured_data()
    folder_path = os.path.join(BASE_DIR, FOLDER)
    res = evaluate_folder(folder_path, streamflow_meas, soil_meas, snow_meas_weekly)
    plot_single_run(res, streamflow_meas, snow_meas_weekly, soil_meas)
    print("Streamflow metrics:", res['streamflow_metrics'])
    print("Snow metrics:", res['snow_metrics'])
    print("Soil metrics:", res['soil_metrics'])

