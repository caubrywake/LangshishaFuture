#!/bin/bash

#SBATCH --ntasks=48
#SBATCH --mem=32G
#SBATCH --threads-per-core=1
#SBATCH --partition=defq
#SBATCH --time="96:00:00"
#SBATCH --job-name=nc2pcr_ssp370
#SBATCH --output=nc2pcr_%j.out
#SBATCH --error=nc2pcr_%j.err

# Load conda environment
source ~/.bashrc
conda activate sptools-test

# --------------------------
# Directory settings
# --------------------------
in_root=/scratch/depfg/aubry001/langshisha-downscale/data/downscaled-daily-mondelta-nc
out_root=/scratch/depfg/aubry001/langshisha-downscale/data/downscaled-daily-mondelta-pcr
tmproot=/scratch/depfg/aubry001/tmp

mkdir -p "$tmproot"

variables="pr tas tasmin tasmax"

# Only ssp126 models
models="gfdl-esm4_ssp370 ipsl-cm6a-lr_ssp370 mpi-esm1-2-hr_ssp370 mri-esm2-0_ssp370 ukesm1-0-ll_ssp370"

export GDAL_MAX_BAND_COUNT=65536
export GDAL_PAM_ENABLED=NO

n_timesteps=31411   # total daily timesteps
ncores=48           # matches --ntasks in SLURM
i=0                 # background job counter

initdir=`pwd`       # save starting directory

# --------------------------
# Loop over models and variables
# --------------------------
for mod in $models; do
    gcm=${mod%_ssp*}
    ssp=${mod#*_}

    # Create a temporary directory per model for faster I/O
    tmpdir=$(mktemp -d -p "$tmproot")
    cd "$tmpdir"

    for var in $variables; do
        [ "$var" == "pr" ] && sphyvar='prec'
        [ "$var" == "tas" ] && sphyvar='tavg'
        [ "$var" == "tasmin" ] && sphyvar='tmin'
        [ "$var" == "tasmax" ] && sphyvar='tmax'

        in_fn=$in_root/${var}_${mod}_2015-2100_rsmp_biascorr_mondelta.nc
        out_dir=$out_root/$gcm/$ssp/$sphyvar

        # Skip missing input file
        [ -f "$in_fn" ] || continue

        mkdir -p "$out_dir"

        # --------------------------
        # Loop over daily timesteps
        # --------------------------
        for t in $(seq 1 $n_timesteps); do
            pcr_index=$(printf %07d $t | sed 's/./&./4')
            out_fn=$out_dir/${sphyvar}${pcr_index}

            # Skip already-existing output
            [ -f "$out_fn" ] && continue

            # Run conversion in background
            gdal_translate -b $t -of PCRaster "$in_fn" "$out_fn" &

            ((i=i+1))
            if (( i % ncores == 0 )); then
                wait   # wait for batch of background jobs
            fi
        done
    done

    # Wait for any remaining jobs for this model
    wait

    # Clean up temporary directory for this model
    cd "$initdir"
    rm -rf "$tmpdir"
done

# Return to starting directory
cd "$initdir"
# return to original dir
cd "$initdir"