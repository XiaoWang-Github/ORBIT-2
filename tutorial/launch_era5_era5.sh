#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J flash
#SBATCH --nodes=16
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 02:00:00
#SBATCH -q debug
#SBATCH -o flash-%j.out
#SBATCH -e flash-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536



source ~/miniconda3/etc/profile.d/conda.sh

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.2.0

eval "$(/lustre/orion/world-shared/stf218/atsaris/env_test_march/miniconda/bin/conda shell.bash hook)"

conda activate /lustre/orion/nro108/world-shared/xf9/flash-attention-torch25

export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/junqi/climax/rccl-plugin-rocm6/lib/:/opt/rocm-6.2.0/lib:$LD_LIBRARY_PATH





export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH


export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD:$PYTHONPATH


time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python ./era5_era5_downscaling.py --max_epochs 40 --checkpoint "./res_slimvit_downscaling_tp/checkpoints/epoch_005-v1.ckpt" /lustre/orion/lrn036/world-shared/data/superres/era5/1.0_deg/ /lustre/orion/lrn036/world-shared/data/superres/era5/0.25_deg/ res_slimvit tp 


#time srun -n $((SLURM_JOB_NUM_NODES*8)) \
#python ./era5_era5_downscaling.py --max_epochs 40 /lustre/orion/lrn036/world-shared/data/superres/era5/1.0_deg/ /lustre/orion/lrn036/world-shared/data/superres/era5/0.25_deg/ unet t2m 

#time srun -n $((SLURM_JOB_NUM_NODES*8)) \
#python ./era5_era5_downscaling.py --max_epochs 40 /lustre/orion/lrn036/world-shared/data/superres/era5/1.0_deg/ /lustre/orion/lrn036/world-shared/data/superres/era5/0.25_deg/ vit t2m 
