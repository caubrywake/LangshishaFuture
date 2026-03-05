#!/bin/bash
#SBATCH --job-name=SPHYsingle
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --threads-per-core=1
#SBATCH --partition=defq
#SBATCH --time="48:00:00"
#SBATCH --output=/scratch/depfg/aubry001/joblogs/sphy_single_%j.out
#SBATCH --error=/scratch/depfg/aubry001/joblogs/sphy_single_%j.err

# === CONFIGURATION ===
cfgroot=/scratch/depfg/aubry001/config_file/calibgw_0_Frac/
outroot=/scratch/depfg/aubry001/output/calibgw_0/GlacSnowFrac_0.7_0.6.cfg/
tmproot=/scratch/depfg/aubry001/tmp
modroot=/scratch/depfg/aubry001/model_code
logroot=/scratch/depfg/aubry001/joblogs

# name of the config you want to run:
cfgfile=GlacSnowFrac_0.7_0.6.cfg

# === ENVIRONMENT SETUP ===
eval "$(conda shell.bash hook)"
conda activate sphy

mkdir -p $tmproot $outroot $logroot

echo "Running SPHY for config: $cfgfile"

# Create working dirs
outdir="${outroot}/${cfgfile}"
mkdir -p $outdir

config_fn=$cfgroot/${cfgfile}
cp $config_fn $outdir/configuration.cfg

# copy model code to tmp
tmpdir=`mktemp -d -p $tmproot`
cp -r $modroot $tmpdir
cd $tmpdir/$(basename $modroot)

# Run SPHY
python $tmpdir/$(basename $modroot)/sphy.py ${outdir}/configuration.cfg

# Clean up temporary files
cd $HOME
rm -rf $tmpdir
echo "? SPHY single run complete: ${cfgfile}"
