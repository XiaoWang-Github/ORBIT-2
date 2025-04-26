#!/bin/bash
#SBATCH -A LRN036
#SBATCH -J flash
#SBATCH --nodes=64
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH -o flash-%j.out
#SBATCH -e flash-%j.error
#SBATCH -C nvme

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


ulimit -n 262144


source ~/miniconda3/etc/profile.d/conda.sh

module load PrgEnv-gnu
module load rocm/6.2.4
module unload darshan-runtime
module unload libfabric

#start of sbcast
echo "copying env to each node in the job"
sbcast -pf /lustre/orion/lrn036/world-shared/xf9/torch26.tar.gz /mnt/bb/${USER}/torch26.tar.gz
if [ ! "$?" == "0" ]; then
    # CHECK EXIT CODE. When SBCAST fails, it may leave partial files on the compute nodes, and if you continue to launch srun,
    # your application may pick up partially complete shared library files, which would give you confusing errors.
    echo "SBCAST failed!"
    exit 1
fi

# Untar the environment file (only need 1 task per node to do this)
srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node 1 mkdir /mnt/bb/${USER}/torch26
echo "untaring env"
srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node 1 tar -xzf /mnt/bb/${USER}/torch26.tar.gz -C  /mnt/bb/${USER}/torch26

# Unpack the env
source activate /mnt/bb/${USER}/torch26
srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node 1 conda-unpack

##### END OF SBCAST AND CONDA-UNPACK #####




## DDStore and GPTL Timer

module use -a /lustre/orion/world-shared/lrn036/jyc/frontier/sw/modulefiles
module load libfabric/1.22.0p


export NCCL_PROTO=Simple
export HOSTNAME=$(hostname)
export PYTHONNOUSERSITE=1


#Needed to bypass MIOpen, Disk I/O Error

export MIOPEN_USER_DB_PATH=/tmp/$JOBID
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
#rm -rf ${MIOPEN_USER_DB_PATH}
srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node 1 mkdir -p ${MIOPEN_USER_DB_PATH}

export MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX=-1
export MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX=-1

export MIOPEN_DEBUG_CONV_WINOGRAD=0

export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD/../src:$PYTHONPATH

export ORBIT_USE_DDSTORE=0 ## 1 (enabled) or 0 (disable)

export LD_PRELOAD=/lib64/libgcc_s.so.1:/usr/lib64/libstdc++.so.6



#time srun -n $((SLURM_JOB_NUM_NODES*8)) \
#python ./intermediate_downscaling.py ../configs/interm_8m.yaml



#time srun -n $((SLURM_JOB_NUM_NODES*8)) \
#python ./intermediate_downscaling.py ../configs/interm_117m.yaml

#time srun -n $((SLURM_JOB_NUM_NODES*8)) \
#python ./intermediate_downscaling.py ../configs/interm_1b.yaml

time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python ./intermediate_downscaling.py ../configs/interm_10b.yaml




