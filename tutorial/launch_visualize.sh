#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J flash
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -o logs/vis-%j.out
#SBATCH -e logs/vis-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536

source ~/miniconda3/etc/profile.d/conda.sh

module load PrgEnv-gnu
module load rocm/6.2.4
module unload darshan-runtime
module unload libfabric

eval "$(/lustre/orion/world-shared/stf218/atsaris/env_test_march/miniconda/bin/conda shell.bash hook)"

conda activate /lustre/orion/lrn036/world-shared/xf9/torch26

#export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/junqi/climax/rccl-plugin-rocm6/lib/:/opt/rocm-6.2.0/lib:$LD_LIBRARY_PATH

## DDStore and GPTL Timer

module use -a /lustre/orion/world-shared/lrn036/jyc/frontier/sw/modulefiles
module load libfabric/1.22.0p




export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH
export HOSTNAME=$(hostname)
export PYTHONNOUSERSITE=1


export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD/../src:$PYTHONPATH

export ORBIT_USE_DDSTORE=0 ## 1 (enabled) or 0 (disable)

export LD_PRELOAD=/lib64/libgcc_s.so.1:/usr/lib64/libstdc++.so.6

# time srun -n $((SLURM_JOB_NUM_NODES*8)) python ./visualize.py ../configs/interm_1b.yaml
time srun -n $((SLURM_JOB_NUM_NODES*8)) python ./visualize.py ../configs/interm_1b_finetune_test.yaml

#time srun -n $((SLURM_JOB_NUM_NODES*1)) python ./visualize.py ../configs/interm_117m.yaml
#time srun -n $((SLURM_JOB_NUM_NODES*8)) python ./visualize.py ../configs/interm_8m.yaml
#time srun -n $((SLURM_JOB_NUM_NODES*1)) python ./visualize.py ../configs/inference.yaml




