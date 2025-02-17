#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J flash
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
<<<<<<< HEAD
#SBATCH -t 00:30:00
#SBATCH -p batch
=======
#SBATCH -t 01:00:00
#SBATCH -q debug
>>>>>>> ac4fb25 (fix bug when image size does not divide patch size. Add training script for PRISM)
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
python ./prism_prism_downscaling.py --max_epochs 30 /lustre/orion/world-shared/lrn036/jyc/frontier/ClimaX-v2/data/prism-superres/10.0_arcmin/ /lustre/orion/world-shared/lrn036/jyc/frontier/ClimaX-v2/data/prism-superres/2.5_arcmin/ res_slimvit prcp 
