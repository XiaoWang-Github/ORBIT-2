#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J flash
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:10:00
#SBATCH -q debug
#SBATCH -o flash-%j.out
#SBATCH -e flash-%j.error

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536



source ~/miniconda3/etc/profile.d/conda.sh


module load PrgEnv-gnu
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

module unload darshan-runtime
module unload libfabric


#eval "$(/lustre/orion/world-shared/stf218/atsaris/env_test_march/miniconda/bin/conda shell.bash hook)"

conda activate /lustre/orion/lrn036/world-shared/xf9/torch27

#source activate /lustre/orion/lrn036/world-shared/xf9/torch27-rocm63
#conda activate /lustre/orion/lrn036/world-shared/xf9/torch26

#export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/junqi/climax/rccl-plugin-rocm6/lib/:/opt/rocm-6.2.0/lib:$LD_LIBRARY_PATH

## DDStore and GPTL Timer

#module use -a /lustre/orion/world-shared/lrn036/jyc/frontier/sw/modulefiles
module load libfabric/1.22.0
module use -a /lustre/orion/world-shared/lrn036/jyc/frontier/sw/modulefiles
module load SR_tools/devel-mpich8.1.31
module load aws-ofi-rccl/devel

echo $LD_LIBRARY_PATH


export FI_MR_CACHE_MONITOR=kdreg2     # Required to avoid a deadlock.
export FI_CXI_DEFAULT_CQ_SIZE=131072  # Ask the network stack to allocate additional space to process message completions.
export FI_CXI_DEFAULT_TX_SIZE=2048    # Ask the network stack to allocate additional space to hold pending outgoing messages.
export FI_CXI_RX_MATCH_MODE=hybrid    # Allow the network stack to transition to software mode if necessary.

export NCCL_NET_GDR_LEVEL=3           # Typically improves performance, but remove this setting if you encounter a hang/crash.
export NCCL_CROSS_NIC=1               # On large systems, this NCCL setting has been found to improve performance
export NCCL_SOCKET_IFNAME=hsn0        # NCCL/RCCL will use the high speed network to coordinate startup.
export TORCH_NCCL_HIGH_PRIORITY=1     # Use high priority stream for the NCCL/RCCL Communicator.

export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH
export HOSTNAME=$(hostname)
export PYTHONNOUSERSITE=1
export MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX=-1
export MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX=-1
export MIOPEN_DEBUG_CONV_WINOGRAD=0


export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD/../src:$PYTHONPATH

export ORBIT_USE_DDSTORE=0 ## 1 (enabled) or 0 (disable)

export LD_PRELOAD=/lib64/libgcc_s.so.1:/usr/lib64/libstdc++.so.6


#time srun -n $((SLURM_JOB_NUM_NODES*8)) \
#python ./intermediate_downscaling.py ../configs/interm_8m_ft.yaml

time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python ./intermediate_downscaling.py ../configs/interm_8m.yaml

