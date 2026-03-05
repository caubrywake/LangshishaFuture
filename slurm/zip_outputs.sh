#!/bin/bash
#SBATCH --job-name=zip_GwreM
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   # pigz will use these CPUs
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -o /scratch/depfg/aubry001/joblogs/zip_%j.out
#SBATCH --error=/scratch/depfg/aubry001/joblogs/zip_%j.err

module purge
# module load pigz  # Not needed if pigz is in /usr/bin

logroot="/scratch/depfg/aubry001/joblogs"
out_root="/scratch/depfg/aubry001/output/futlong"

# List of GCMs and SSPs
gcms=("ukesm1-0-ll" "mri-esm2-0" "mpi-esm1-2-hr" "ipsl-cm6a-lr" "gfdl-esm4")
ssps=("ssp126" "ssp370" "ssp585")

# Loop over each GCM and SSP
for gcm in "${gcms[@]}"; do
    for ssp in "${ssps[@]}"; do
        folder="$out_root/$gcm/$ssp"
        echo "Compressing $gcm/$ssp ..." | tee -a "$logroot/zip_${gcm}_${ssp}.log"

        # Count GwreM* files for progress
        num_files=$(find "$folder" -maxdepth 1 -type f -name "GwreM*" | wc -l)
        if [ "$num_files" -eq 0 ]; then
            echo "No GwreM* files found in $folder, skipping" | tee -a "$logroot/zip_${gcm}_${ssp}.log"
            continue
        fi

        # Compress GwreM* files in the folder
        tar cf - -C "$folder" $(ls "$folder"/GwreM* 2>/dev/null) \
        | pv -l -s $num_files \
        | pigz -p 8 > "$folder/${gcm}_${ssp}_GwreM_compressed.tar.gz"

        echo "Done $gcm/$ssp" | tee -a "$logroot/zip_${gcm}_${ssp}.log"
    done
done