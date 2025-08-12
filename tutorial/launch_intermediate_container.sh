#!/bin/bash
#SBATCH -A g200
#SBATCH -J flash
#SBATCH --nodes=1
##SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH -t 00:10:00
#SBATCH -o flash-%j.out
#SBATCH -e flash-%j.error

#source /capstor/store/cscs/userlab/g200/ilyngaas/my-venv-2/bin/activate

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

export OMP_NUM_THREADS=8
export HOSTNAME=$(hostname)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export MPICH_GPU_SUPPORT_ENABLED=0
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS

export PYTHONPATH=$PWD/../src:$PYTHONPATH


export TRITON_HOME=/dev/shm/
cat > run-cmd.sh <<EOF
#!/bin/bash
RANK=\$SLURM_PROCID
LOCAL_RANK=\$SLURM_LOCALID

python ./intermediate_downscaling.py ../configs/interm_8m.yaml
#python ./intermediate_downscaling.py ../configs/interm_117m.yaml

EOF

chmod +x run-cmd.sh

srun --environment=${HOME}/.edf/superres-2.toml --container-workdir=$PWD ./run-cmd.sh
