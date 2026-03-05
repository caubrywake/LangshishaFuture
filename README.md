# Langshisha SPHY Modeling and Analysis Repository

**Author:** Dr. Caroline Aubry-Wake  
**Institution:** University of Lethbridge  

**Model:** SPHY (Spatial Processes in HYdrology)  
**Study focus:** Glacier–hydrology interactions and future change in the Langshisha Basin, Langtang, Nepal.

This repository contains the model implementation, analysis scripts, and evaluation datasets used to simulate and analyze hydrological processes in the Langshisha basin.

---

# Repository Structure

```
repo/
│
├── script/        → Model output analysis and figure-generation scripts
├── slurm/         → HPC job submission scripts and preprocessing utilities
├── model_code/    → SPHY source code used for simulations
├── data-eval/     → Observational datasets used for calibration and evaluation
├── model_input/   → Model parameter input files
└── model_cfg/     → Model configuration files
```

---

# Repository Content

## script/ — Analysis Workflow

This folder contains numbered scripts used to reproduce manuscript figures and tables.

Scripts are organized sequentially:

```
00_evaluate_calibration_runs.py
01_evaluate_presentday_fig03.py
02_evaluate_glacier_mass_balance.py
03_extract_basin_mean_forcing_daily.py
04_analyze_annual_forcing_and_plot_fig04.py
04b_analyze_seasonal_forcing_stats.py
05_analyze_future_glacier_change_fig05.py
06_analyze_future_melt_partitioning_fig06.py
07a_sources_doy_and_annual_composition_fig07.py
07b_analyze_future_flow_pathways_fig07_08.py
08_analyze_future_streamflow_variability_fig09.py
X_SettingupConfig → utilities for generating configuration files
```

Each script includes a metadata header describing required inputs, outputs, and assumptions.

---

## model_code/ — SPHY Model Implementation

Contains the version of SPHY used in this study.

Study-specific modifications are annotated in the code as:

```
CAW edit – date
```

Key components include:

```
sphy.py
rootzone.py
subzone.py
ET.py
hargreaves.py

modules/
utilities/
```

---

## slurm/ — HPC Execution Scripts

Contains job submission scripts used to run SPHY simulations on HPC systems and preprocess forcing datasets.

Examples include:

```
run_single_sphy.sh          → single calibration run
sphy_pergcmssp*.sh          → future scenario simulations per GCM / SSP

netcdf-to-pcr-ishimip-ssp*.sh → convert ISIMIP forcing data to PCRaster format
run_nc2pcr.slurm             → batch wrapper for NetCDF → PCRaster conversion
```

---

## data-eval/ — Observational Datasets

Contains observational datasets used for model calibration and evaluation.

Examples include:

- Streamflow observations  
- Langshisha automatic weather station data  
- Soil moisture measurements  

---

## model_input/ — Model Input Parameters

Contains SPHY parameter inputs in PCRaster format used for model simulations.

---

## model_cfg/ — Model Configuration Files

Contains `.cfg` files defining model configurations for calibration and future scenario runs.

---

# Forcing and Model Output Data (Not Included)

Due to file size constraints, forcing datasets and model outputs are **not included in this repository**.

---
