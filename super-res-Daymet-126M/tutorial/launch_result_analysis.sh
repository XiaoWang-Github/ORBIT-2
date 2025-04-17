#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J inference_flash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 01:00:00
##SBATCH -q debug
#SBATCH -o out/res-%j.out
#SBATCH -e out/res-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536

source ~/miniconda3/etc/profile.d/conda.sh

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.2.0

eval "$(/lustre/orion/world-shared/stf218/atsaris/env_test_march/miniconda/bin/conda shell.bash hook)"

source source_env.sh
#conda activate /lustre/orion/lrn036/world-shared/kurihana/super-res-torchlight/flash-attention-rocm6.2.4-tak

module use -a /lustre/orion/world-shared/lrn036/jyc/frontier/sw/modulefiles
module load SR_tools

## DDStore and GPTL Timer
#module use -a /lustre/orion/world-shared/lrn036/jyc/frontier/sw/modulefiles
#module load SR_tools


#lowresdir="/lustre/orion/lrn036/world-shared/kurihana/regridding/dataset/datasets/ERA5-Daymet-1dy-superres/15.0_arcmin"
#highresdir="/lustre/orion/lrn036/world-shared/kurihana/regridding/dataset/datasets/ERA5-Daymet-1dy-superres/3.75_arcmin"

lowresdir="/lustre/orion/lrn036/world-shared/data/superres/daymet/10.0_arcmin"
highresdir="/lustre/orion/lrn036/world-shared/data/superres/daymet/2.5_arcmin"


#expname=3174672
#variable='tmin'
expname='3174672'
variable='2m_temperature_min'  
basedir="/lustre/orion/cli138/proj-shared/kurihana/super-res-torchlight/tutorial"
checkpoint_path=checkpoints/climate/interm_rank_0_epoch_11.ckpt
outputdir=checkpoints/test
config_path="../configs/interm_8m.yaml"
#config_path="../configs/interm_fine_tune_template.yaml"

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
python ./Downscaling_Result.py
