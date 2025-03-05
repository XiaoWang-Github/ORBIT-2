#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J flash
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH -o flash-%j.out
#SBATCH -e flash-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536

module load PrgEnv-gnu
module load rocm/6.2.4
module unload darshan-runtime
module unload libfabric

source /lustre/orion/world-shared/lrn036/jyc/frontier/sw/anaconda3/2023.09/etc/profile.d/conda.sh

eval "$(/lustre/orion/world-shared/lrn036/jyc/frontier/sw/anaconda3/2023.09/bin/conda shell.bash hook)"

conda activate /lustre/orion/world-shared/lrn036/jyc/frontier/sw/superres 

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

time srun -n $((SLURM_JOB_NUM_NODES*8)) --export=ALL,LD_PRELOAD=/lib64/libgcc_s.so.1:/usr/lib64/libstdc++.so.6 \
python ./intermediate_downscaling.py ../configs/interm.yaml

#time srun -n $((SLURM_JOB_NUM_NODES*8)) \
#python ./intermediate_downscaling.py ../configs/era5_era5.yaml
#time srun -n $((SLURM_JOB_NUM_NODES*8)) \
#python ./intermediate_downscaling.py ../configs/prism_prism.yaml

