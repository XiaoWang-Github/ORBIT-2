#!/bin/bash
#SBATCH -A g200
#SBATCH -J flash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH -t 00:10:00
#SBATCH -p debug
#SBATCH -o flash-%j.out
#SBATCH -e flash-%j.error
#SBATCH --uenv=pytorch/v2.6.0:/user-environment
#SBATCH --view=default

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


ulimit -n 262144

export DISTRIBUTED_INITIALIZATION_METHOD=SLURM
# OPENMP Environment Variables
export OMP_NUM_THREADS=64

#load environment
#source /capstor/store/cscs/userlab/g200/xf9/orbit_env/bin/activate

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export TRITON_HOME=/dev/shm/

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
#################################
# MPICH environment variables   #
#################################
export MPICH_GPU_SUPPORT_ENABLED=0 

#################################
# CUDA environment variables    #
#################################
export CUDA_CACHE_DISABLE=1

############################################
# NCCL and Fabric environment variables    #
############################################

# This forces NCCL to use the libfabric plugin, enabling full use of the
# Slingshot network. If the plugin can not be found, applications will fail to
# start. With the default value, applications would instead fall back to e.g.
# TCP, which would be significantly slower than with the plugin. More information
# about `NCCL_NET` can be found at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-net
export NCCL_NET="AWS Libfabric"
# Use GPU Direct RDMA when GPU and NIC are on the same NUMA node. More
# information about `NCCL_NET_GDR_LEVEL` can be found at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-net-gdr-level-formerly-nccl-ib-gdr-level
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
# These `FI` (libfabric) environment variables have been found to give the best
# performance on the Alps network across a wide range of applications. Specific
# applications may perform better with other values.
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=32768
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd


#export NCCL_PROTO=Simple
export HOSTNAME=$(hostname)
export PYTHONNOUSERSITE=1


export PYTHONPATH=$PWD/../src:$PYTHONPATH

export ORBIT_USE_DDSTORE=0 ## 1 (enabled) or 0 (disable)


srun -n $((SLURM_JOB_NUM_NODES*4))  bash -c "
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID
    . /capstor/store/cscs/userlab/g200/xf9/orbit_env/bin/activate 
    python ./intermediate_downscaling.py ../configs/interm_8m.yaml
"



#time srun -n $((SLURM_JOB_NUM_NODES*8)) \
#python ./intermediate_downscaling.py ../configs/interm_117m.yaml

#time srun -n $((SLURM_JOB_NUM_NODES*8)) \
#python ./intermediate_downscaling.py ../configs/interm_1b.yaml

#time srun -n $((SLURM_JOB_NUM_NODES*8)) \
#python ./intermediate_downscaling.py ../configs/interm_10b.yaml




