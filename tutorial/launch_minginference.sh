#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J inference_flash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 00:20:00
#SBATCH -q debug
#SBATCH -o out/inf-%j.out
#SBATCH -e out/inf-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536

module load libfabric/1.22.0p
module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.2.0

eval "$(/lustre/orion/world-shared/stf218/atsaris/env_test_march/miniconda/bin/conda shell.bash hook)"

#conda activate /lustre/orion/lrn036/world-shared/kurihana/super-res-torchlight/flash-attention-torch25-tak_march12
conda activate /lustre/orion/lrn036/scratch/hossainm/project_orbit/frontier_envs/flashattn03_env

## DDStore and GPTL Timer
module use -a /lustre/orion/world-shared/lrn036/jyc/frontier/sw/modulefiles
module load SR_tools


lowresdir="/lustre/orion/lrn036/world-shared/kurihana/regridding/dataset/datasets/ERA5-Daymet-1dy-superres/15.0_arcmin"
highresdir="/lustre/orion/lrn036/world-shared/kurihana/regridding/dataset/datasets/ERA5-Daymet-1dy-superres/3.75_arcmin"
#lowresdir="/lustre/orion/lrn036/world-shared/kurihana/super-res-torchlight/superres/era5/0.25_deg_test/"
#highresdir="/lustre/orion/lrn036/world-shared/kurihana/super-res-torchlight/superres/era5/0.25_deg_test"


#epoch=67 #18 
#loss=imagegradient
#loss=mse
# total prcp 24hr
#expname=3215715 # 8m
#expname=3215780
#expname=3222161
variable='total_precipitation_24hr' 

# DIR
#basedir="/lustre/orion/cli138/proj-shared/kurihana/super-res-torchlight/tutorial"
#checkpoint_path=${basedir}/checkpoints/climate/${loss}/${expname}/interm_rank_0_epoch_${epoch}.ckpt
#checkpoint_path="/lustre/orion/lrn036/world-shared/kurihana/super-res-torchlight/tak-fine-tune-weights/3215715/interm_rank_0_epoch_48.ckpt"
#checkpoint_path="/lustre/orion/lrn036/scratch/hossainm/project_orbit/super-res-torchlight-fine_tune_tak_vMar18/tutorial/to_ming/3215780/interm_rank_0_epoch_67.ckpt"
checkpoint_path="/lustre/orion/lrn036/scratch/hossainm/project_orbit/super-res-torchlight-fine_tune_tak_vMar18/tutorial/to_ming/3215715/interm_rank_0_epoch_48.ckpt"
#checkpoint_path="/lustre/orion/lrn036/scratch/hossainm/project_orbit/super-res-torchlight-fine_tune_tak_vMar18/tutorial/to_ming/3222161/interm_rank_0_epoch_90.ckpt"
#outputdir=${basedir}/checkpoints/climate/${loss}/${expname}/test
outputdir="./test3215715"
# YAML
#config_path="../configs/interm_8m_total_precip_24hr.yaml"
#config_path="../configs/interm_117m_total_precip_24hr.yaml"
#config_path="../configs/interm_117m_total_precip_24hr_grd.yaml"i
#config_path="../configs/inference.yaml"
config_path="/lustre/orion/lrn036/scratch/hossainm/project_orbit/super-res-torchlight-fine_tune_tak_vMar18/tutorial/to_ming/config_daymet/interm_8m_total_precip_24hr.yaml"
#config_path="/lustre/orion/lrn036/scratch/hossainm/project_orbit/super-res-torchlight-fine_tune_tak_vMar18/tutorial/to_ming/config_daymet/interm_117m_total_precip_24hr_grd.yaml"
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


#time srun -n $((SLURM_JOB_NUM_NODES*4)) \
time srun -n $((SLURM_JOB_NUM_NODES*1)) \
python ./mcinference_era5_daymet.py \
    --lowresdir ${lowresdir} \
    --highresdir ${highresdir} \
    --checkpoint_path ${checkpoint_path} \
    --outputdir $outputdir \
    --variable ${variable} \
    --config_path ${config_path} \
    #--data_key INFER
    --data_key DAYMET_3
