#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J inference_flash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 00:05:00
##SBATCH -q debug
#SBATCH -o out/inf-%j.out
#SBATCH -e out/inf-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536

module load PrgEnv-gnu
module load rocm/6.2.4
module unload darshan-runtime
module unload libfabric

source /lustre/orion/world-shared/lrn036/jyc/frontier/sw/anaconda3/2023.09/etc/profile.d/conda.sh
eval "$(/lustre/orion/world-shared/lrn036/jyc/frontier/sw/anaconda3/2023.09/bin/conda shell.bash hook)"

conda activate /lustre/orion/lrn036/world-shared/kurihana/super-res-torchlight/flash-attention-rocm6.2.4-tak

module use -a /lustre/orion/world-shared/lrn036/jyc/frontier/sw/modulefiles
module load libfabric/1.22.0p

## DDStore and GPTL Timer
module use -a /lustre/orion/world-shared/lrn036/jyc/frontier/sw/modulefiles
module load SR_tools


lowresdir="/lustre/orion/lrn036/world-shared/kurihana/regridding/dataset/datasets/ERA5-Daymet-1dy-superres/15.0_arcmin"
highresdir="/lustre/orion/lrn036/world-shared/kurihana/regridding/dataset/datasets/ERA5-Daymet-1dy-superres/3.75_arcmin"

#expname=3174672
#variable='tmin'
expname='3178042' #3174672
variable='prcp' #'tmin'

# DIR
basedir="/lustre/orion/cli138/proj-shared/kurihana/super-res-torchlight/tutorial"
checkpoint_path=${basedir}/checkpoints/climate/imagegradient/${expname}/interm_rank_0_epoch_49.ckpt
outputdir=${basedir}/checkpoints/climate/imagegradient/${expname}/test
# YAML
#config_path="/ccs/home/kurihana/proje-shared_cli138/super-res-torchlight/configs/interm_fine_tune_template.yaml"
#config_path="/ccs/home/kurihana/proje-shared_cli138/super-res-torchlight/configs/interm_fine_tune_prcp.yaml"
config_path="/lustre/orion/cli138/proj-shared/kurihana/super-res-torchlight/configs/interm_fine_tune_prcp_scratch.yaml"

mkdir -p $outputdir

export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH
export HOSTNAME=$(hostname)
export PYTHONNOUSERSITE=1


export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD/../src:$PYTHONPATH

export ORBIT_USE_DDSTORE=0 ## 1 (enabled) or 0 (disable)


time srun -n $((SLURM_JOB_NUM_NODES*1)) --export=ALL,LD_PRELOAD=/lib64/libgcc_s.so.1:/usr/lib64/libstdc++.so.6 \
python ./inference_era5_daymet.py \
    --lowresdir ${lowresdir} \
    --highresdir ${highresdir} \
    --checkpoint_path ${checkpoint_path} \
    --outputdir $outputdir \
    --variable ${variable} \
    --config_path ${config_path} \
    --era5_total_precip 
    # --landcover --soil_moist