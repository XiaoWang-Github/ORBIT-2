#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J fine-tune
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:40:00
##SBATCH -t 00:05:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -o ./out/fine-tune-%j.out
#SBATCH -e ./out/fine-tune-%j.error

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536



source ~/miniconda3/etc/profile.d/conda.sh

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.2.0

eval "$(/lustre/orion/world-shared/stf218/atsaris/env_test_march/miniconda/bin/conda shell.bash hook)"

#conda activate /lustre/orion/lrn036/world-shared/xf9/flash-attention-torch25
conda activate /lustre/orion/lrn036/world-shared/kurihana/super-res-torchlight/flash-attention-torch25-tak_march12

#export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/junqi/climax/rccl-plugin-rocm6/lib/:/opt/rocm-6.2.0/lib:$LD_LIBRARY_PATH

## DDStore and GPTL Timer
module use -a /lustre/orion/world-shared/lrn036/jyc/frontier/sw/modulefiles
module load SR_tools


export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH
export HOSTNAME=$(hostname)
export PYTHONNOUSERSITE=1


export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD/../src:$PYTHONPATH

export ORBIT_USE_DDSTORE=0 ## 1 (enabled) or 0 (disable)

expname=3190616 #$SLURM_JOB_ID

time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python ./intermediate_downscaling.py ../configs/interm_prcp.yaml ${expname}
