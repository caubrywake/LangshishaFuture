#!/bin/bash
#SBATCH --nodes=1        
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --threads-per-core=1
#SBATCH --partition=defq
#SBATCH --time="96:00:00"
#SBATCH --job-name=run370

# Set this at the top of your script:
RUN_SSP370_ONLY=true  # set to false to run all SSPs

cfg_template=/scratch/depfg/aubry001/cfg/fut.cfg
forcing_root=/scratch/depfg/aubry001/minp/ds
out_root=/scratch/depfg/aubry001/out/fut
tmproot=/scratch/depfg/aubry001/tmp
modroot=/scratch/depfg/aubry001/model_code
logroot=/scratch/depfg/aubry001/joblogs

initdir=$(pwd)

eval "$(conda shell.bash hook)"
conda activate sphy

mkdir -p $tmproot $logroot

i=0

for gcm_dir in $forcing_root/*; do
  gcm=$(basename $gcm_dir)
  for ssp_dir in $gcm_dir/*; do
    ssp=$(basename $ssp_dir)
   
    if [ "$RUN_SSP370_ONLY" = true ] && [[ "$ssp" != *"ssp370"* ]]; then
      echo "Skipping $ssp (not ssp370)"
      continue
    fi

    runname="${gcm}_${ssp}"
    echo "Starting experiment: $runname"

    outdir="${out_root}/${gcm}/${ssp}/"
    mkdir -p "$outdir"

    # Sanity check before running the model	
    if [ ! -d "$outdir" ]; then
      echo "ERROR: Output dir missing: $outdir" >&2
      exit 1
    fi

    # Forcing directories (no duplicated /prec/prec pattern)
    prec_dir="ds/$gcm/$ssp/prec/prec"
    tair_dir="ds/$gcm/$ssp/tavg/tavg"
    tmin_dir="ds/$gcm/$ssp/tmin/tmin"
    tmax_dir="ds/$gcm/$ssp/tmax/tmax"

    tmpdir=$(mktemp -d -p $tmproot)
    cp -r $modroot $tmpdir

    # Replace placeholders in template config
    sed -e "s|__PREC_PATH__|$prec_dir|g" \
        -e "s|__TAIR_PATH__|$tair_dir|g" \
        -e "s|__TMIN_PATH__|$tmin_dir|g" \
        -e "s|__TMAX_PATH__|$tmax_dir|g" \
        -e "s|__OUTPUT_DIR__|$outdir|g" \
        $cfg_template > $outdir/configuration.cfg

    cd $tmpdir/$(basename $modroot)

    srun -n1 --exclusive --mem=5G -e $logroot/job.%N.%j.${runname}.err -o $logroot/job.%N.%j.${runname}.out \
      python sphy.py $outdir/configuration.cfg &

    ((i=i+1))
    if (( i % 48 == 0 )); then
        wait
    fi

    cd $initdir
  done
done

wait
rm -rf $tmproot
