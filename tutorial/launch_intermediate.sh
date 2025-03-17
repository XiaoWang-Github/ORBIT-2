#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J flash
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:15:00
#SBATCH -o flash-%j.out
#SBATCH -e flash-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536

module load PrgEnv-gnu
module load rocm/6.2.4
module unload darshan-runtime
module unload libfabric

source /lustre/orion/proj-shared/stf006/irl1/conda/bin/activate

#eval "$(/lustre/orion/world-shared/lrn036/jyc/frontier/sw/anaconda3/2023.09/bin/conda shell.bash hook)"

#conda activate /lustre/orion/world-shared/lrn036/jyc/frontier/sw/superres 
conda activate /lustre/orion/stf006/proj-shared/irl1/super-env

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
python ./intermediate_downscaling.py ../configs/experiments/prism_temp_phys.yaml
#python ./intermediate_downscaling.py ../configs/experiments/era5_2_temp_normal.yaml
#python ./intermediate_downscaling.py ../configs/experiments/prism_temp_1012.yaml
#python ./intermediate_downscaling.py ../configs/experiments/era5_2_prcp_normal.yaml
#python ./intermediate_downscaling.py ../configs/experiments/era5_2_temp_1012.yaml
#python ./intermediate_downscaling.py ../configs/experiments/era5_2_prcp_1012.yaml


#PRISM
#python ./intermediate_downscaling.py ../configs/experiments/prism_temp_normal.yaml
#python ./intermediate_downscaling.py ../configs/experiments/prism_temp_1012.yaml
#python ./intermediate_downscaling.py ../configs/experiments/prism_prcp_normal.yaml
#python ./intermediate_downscaling.py ../configs/experiments/prism_prcp_1012.yaml


#era5_1
#python ./intermediate_downscaling.py ../configs/experiments/era5_1_prcp_127.yaml
#python ./intermediate_downscaling.py ../configs/experiments/era5_1_prcp_normal.yaml
#python ./intermediate_downscaling.py ../configs/experiments/era5_1_temp_127.yaml
#python ./intermediate_downscaling.py ../configs/experiments/era5_1_temp_normal.yaml


