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
#SBATCH -e flash-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

source ~/miniconda3/etc/profile.d/conda.sh

CONDA_ENV_DIR=/lustre/orion/stf006/world-shared/irl1/bayes-cast-env

module load PrgEnv-gnu
module load rocm/7.13.0
module load craype-accel-amd-gfx90a
module unload libfabric

conda activate ${CONDA_ENV_DIR}

SDK_CORE_LIB=${CONDA_ENV_DIR}/lib/python3.11/site-packages/_rocm_sdk_core/lib
srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node 1 bash -c "
'${CONDA_ENV_DIR}/bin/python' -c \"
import sys, torch
assert '${CONDA_ENV_DIR}' in sys.prefix, f'unexpected prefix {sys.prefix!r}'
assert '${CONDA_ENV_DIR}' in torch.__file__, f'unexpected torch path {torch.__file__!r}'
print('sys.prefix:', sys.prefix)
\" &&
  add_unversioned_symlink() {
    local dir=\"\$1\"; local versioned=\"\$2\"; local unversioned=\"\$3\"
    if [ ! -e \"\${dir}/\${unversioned}\" ]; then
      ln -sf \"\${versioned}\" \"\${dir}/\${unversioned}\"
    fi
  }
  add_unversioned_symlink '${SDK_CORE_LIB}' libamdhip64.so.7 libamdhip64.so
  add_unversioned_symlink '${SDK_CORE_LIB}' libhsa-runtime64.so.1 libhsa-runtime64.so
"

module load cray-mpich/8.1.31 libfabric
module load libfabric/1.20.1 rccl-net-plugin

SDK_CORE=${CONDA_ENV_DIR}/lib/python3.11/site-packages/_rocm_sdk_core
SDK_LIBS=${CONDA_ENV_DIR}/lib/python3.11/site-packages/_rocm_sdk_libraries_gfx90a

export LD_LIBRARY_PATH=${SDK_CORE}/lib:${SDK_LIBS}/lib:${LD_LIBRARY_PATH}
unset LD_PRELOAD
unset NCCL_NET_PLUGIN
export LD_PRELOAD=/opt/rocm-7.13.0/lib/librccl.so.1

export HSA_NO_SCRATCH_RECLAIM=1
unset NCCL_DEBUG

export MIOPEN_DISABLE_CACHE=1
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


# Visualization command examples:
# 1. Use checkpoint path from config file (default):
# time srun -n $((SLURM_JOB_NUM_NODES*8)) python ./visualize.py ../../configs/interm_8m_ft.yaml

# 2. Override with custom checkpoint path:
# time srun -n $((SLURM_JOB_NUM_NODES*8)) python ./visualize.py ../../configs/interm_8m_ft.yaml --checkpoint /path/to/custom/checkpoint.ckpt

# 3. With additional options (index, variable, etc.):
# time srun -n $((SLURM_JOB_NUM_NODES*8)) python ./visualize.py ../../configs/interm_8m_ft.yaml --checkpoint /path/to/custom/checkpoint.ckpt --index 10 --variable 2m_temperature_max

time srun -n $((SLURM_JOB_NUM_NODES*8)) python ./visualize.py ../../configs/interm_8m_ft.yaml

