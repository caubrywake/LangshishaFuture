#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=32G
#SBATCH --threads-per-core=1
#SBATCH --partition=defq
#SBATCH --time="48:00:00"
#SBATCH --job-name=nc2pcr
#SBATCH --output=nc2pcr_%j.out
#SBATCH --error=nc2pcr_%j.err

# Load modules if needed (example for conda)
source ~/.bashrc
conda activate sptools-test

# Go to the directory where your script is
cd /scratch/eejit/aubry001/code/

# Run your conversion script
bash netcdf-to-pcr-ishimip-permodssp.sh
