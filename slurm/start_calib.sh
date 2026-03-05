#!/bin/bash
#SBATCH --ntasks=48
#SBATCH --mem=64G
#SBATCH --threads-per-core=1
#SBATCH --partition=defq
#SBATCH --time="96:00:00"
#SBATCH --job-name SPHYcal

cfgroot=/scratch/depfg/aubry001/config_file/calib_v3
outroot=/scratch/depfg/aubry001/output/calib_v3
tmproot=/scratch/depfg/aubry001/tmp
modroot=/scratch/depfg/aubry001/model_code
logroot=/scratch/depfg/aubry001/joblogs
initdir=`pwd`

eval "$(conda shell.bash hook)"
conda activate sphy

mkdir -p $tmproot

for run in `ls $cfgroot`
do
  echo "Starting experiment: ${run}"

  # create output dir
  outdir="${outroot}/${run}"
  mkdir -p $outdir
  
  # create log dir
  mkdir -p $logroot

  # define config to use for running from variable and copy it to output dir
  config_fn=$cfgroot/${run}
  cp $config_fn $outdir
  mv ${outdir}/`basename ${config_fn}` ${outdir}/configuration.cfg

  # copy sphy directory to temporary dir
  tmpdir=`mktemp -d -p $tmproot`
  cp -r $modroot $tmpdir

  # make tmp dir the active dir to make sure tss files are correctly exported when doing parallel runs
  cd $tmpdir/`basename $modroot`

  # one processor is enough, upgrade memory capacity if needed (without going over main allocation limits!)
  srun -n1 --exclusive --mem=5G -e $logroot/job.%N.%j.%s.out -o $logroot/job.%N.%j.%s.out python $tmpdir/$(basename $modroot)/sphy.py ${outdir}/configuration.cfg &
done

# wait for all background srun processes to finish
echo "Waiting for all runs to finish..."
wait

# go back to start directory
cd $initdir

# remove temporary dir and its contents
rm -rf $tmproot

