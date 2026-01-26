#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J flash
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:20:00
#SBATCH -p extended
#SBATCH --export=ALL
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

conda activate /lustre/orion/proj-shared/lrn036/yoonh/torch27

# Set cache directories
export TMPDIR="/lustre/orion/proj-shared/lrn036/yoonh/cache"
export PIP_CACHE_DIR="/lustre/orion/proj-shared/lrn036/yoonh/cache"
export PYTHONPYCACHEPREFIX="/lustre/orion/proj-shared/lrn036/yoonh/cache"
export TORCH_HOME="/lustre/orion/proj-shared/lrn036/yoonh/cache"
export TORCH_EXTENSIONS_DIR="/lustre/orion/proj-shared/lrn036/yoonh/cache/torch_extensions"
export TRITON_CACHE_DIR="/lustre/orion/proj-shared/lrn036/yoonh/cache/triton"
# INT8 debug/perf toggles (override when submitting: env VAR=value sbatch ...)
export ATTENTION_DEBUG=${ATTENTION_DEBUG:-0}                 # set 1 to enable NaN/Inf logging in attention
export INT8_DISABLE_STOCHASTIC_ROUND=${INT8_DISABLE_STOCHASTIC_ROUND:-1} # default off for perf; set 0 to keep stochastic rounding
export INT8_WEIGHT_CACHE_STRATEGY=${INT8_WEIGHT_CACHE_STRATEGY:-epoch}   # default to epoch cache for stability (off|step|epoch)
export INT8_INPUT_SCALE_EMA=${INT8_INPUT_SCALE_EMA:-0.8}                 # EMA alpha for input absmax (0 disables)
export INT8_INPUT_SCALE_UPDATE_EVERY=${INT8_INPUT_SCALE_UPDATE_EVERY:-1} # update EMA every N steps (>=1)
export INT8_INPUT_SCALE_FREEZE_AFTER=${INT8_INPUT_SCALE_FREEZE_AFTER:-0} # freeze EMA after N updates (0 disables)
export INT8_INPUT_SCALE_SAMPLE_STRIDE=${INT8_INPUT_SCALE_SAMPLE_STRIDE:-1} # sample stride for absmax (1 disables)
export INT8_OUTPUT_SCALE_SAMPLE_STRIDE=${INT8_OUTPUT_SCALE_SAMPLE_STRIDE:-1} # sample stride for output absmax (1 disables)
export INT8_OUTPUT_REQUANT_METHOD=${INT8_OUTPUT_REQUANT_METHOD:-auto} # auto|direct (direct skips output absmax)
export INT8_OUTPUT_REQUANT_TRITON=0 # 1 to use Triton int8 output kernel when bias is None
export INT8_VARATTN_PREQUANT=${INT8_VARATTN_PREQUANT:-1} # 1 to pre-quantize VarAttention input x
export INT8_VARATTN_KV_TRITON=${INT8_VARATTN_KV_TRITON:-1} # 1 to use VarAttention KV Triton kernel
export INT8_VARATTN_KV_TRITON_LOG=${INT8_VARATTN_KV_TRITON_LOG:-1} # 1 to log VarAttention KV Triton shapes once
export INT8_VARATTN_KV_BLOCK_M=${INT8_VARATTN_KV_BLOCK_M:-}
export INT8_VARATTN_KV_BLOCK_N=${INT8_VARATTN_KV_BLOCK_N:-}
export INT8_VARATTN_KV_BLOCK_K=${INT8_VARATTN_KV_BLOCK_K:-}
export INT8_VARATTN_KV_NUM_WARPS=${INT8_VARATTN_KV_NUM_WARPS:-}
export INT8_VARATTN_KV_NUM_STAGES=${INT8_VARATTN_KV_NUM_STAGES:-}
# Optional lightweight timing/logging
export PROFILE_BATCH_TIME=${PROFILE_BATCH_TIME:-0}                      # set 1 for CUDA event timing every 10 batches
export LOG_MEM_EVERY=${LOG_MEM_EVERY:-0}                                # set >0 to log reserved memory every N batches
export PROFILE_MAX_STEPS=${PROFILE_MAX_STEPS:-0}                        # 0 means full epoch (no early stop)
# Quantization visualization (module graph + profiler + summary)
export QUANT_VIZ=1
export QUANT_VIZ_STEPS=1
export QUANT_VIZ_DIR=${QUANT_VIZ_DIR:-/lustre/orion/proj-shared/lrn036/yoonh/super-res-torchlight/examples/quant_viz}
# INT8 softmax (LUT approximation) in attention
export INT8_SOFTMAX=1
export INT8_SOFTMAX_LUT_RANGE=${INT8_SOFTMAX_LUT_RANGE:-6.0}
export INT8_SOFTMAX_LUT_SIZE=${INT8_SOFTMAX_LUT_SIZE:-512}
export INT8_SOFTMAX_LUT_SCALE=${INT8_SOFTMAX_LUT_SCALE:-32768}
export INT8_SOFTMAX_LOG=1
export INT8_SOFTMAX_TRITON=${INT8_SOFTMAX_TRITON:-1}
export INT8_SOFTMAX_OUTPUT_INT8=${INT8_SOFTMAX_OUTPUT_INT8:-1}
export INT8_SOFTMAX_OUTPUT_SCALE=${INT8_SOFTMAX_OUTPUT_SCALE:-127.0}
export INT8_SOFTMAX_OUTPUT_TRITON=${INT8_SOFTMAX_OUTPUT_TRITON:-1}
export INT8_ATTENTION_E2E=${INT8_ATTENTION_E2E:-1}
export INT8_ATTENTION_E2E_LOG=${INT8_ATTENTION_E2E_LOG:-1}
export INT8_ATTENTION_AV_TRITON=${INT8_ATTENTION_AV_TRITON:-1}
export INT8_ATTENTION_TIMING=${INT8_ATTENTION_TIMING:-0}
export INT8_ATTENTION_TIMING_EVERY=${INT8_ATTENTION_TIMING_EVERY:-50}
export INT8_ATTENTION_QKV_LOG=${INT8_ATTENTION_QKV_LOG:-1}
export INT8_QKV_REQUANT_TRITON=${INT8_QKV_REQUANT_TRITON:-0}
export INT8_QKV_BLOCK_M=${INT8_QKV_BLOCK_M:-}
export INT8_QKV_BLOCK_N=${INT8_QKV_BLOCK_N:-}
export INT8_QKV_BLOCK_K=${INT8_QKV_BLOCK_K:-}
export INT8_QKV_GROUP_M=${INT8_QKV_GROUP_M:-}
export INT8_QKV_NUM_WARPS=${INT8_QKV_NUM_WARPS:-}
export INT8_QKV_NUM_STAGES=${INT8_QKV_NUM_STAGES:-}
echo "INT8_SOFTMAX=$INT8_SOFTMAX LUT_RANGE=$INT8_SOFTMAX_LUT_RANGE LUT_SIZE=$INT8_SOFTMAX_LUT_SIZE LUT_SCALE=$INT8_SOFTMAX_LUT_SCALE TRITON=$INT8_SOFTMAX_TRITON OUT_INT8=$INT8_SOFTMAX_OUTPUT_INT8 OUT_SCALE=$INT8_SOFTMAX_OUTPUT_SCALE OUT_TRITON=$INT8_SOFTMAX_OUTPUT_TRITON REQ_TRITON=$INT8_OUTPUT_REQUANT_TRITON QKV_LOG=$INT8_ATTENTION_QKV_LOG QKV_TRITON=$INT8_QKV_REQUANT_TRITON VARATTN_PREQ=$INT8_VARATTN_PREQUANT VARATTN_KV_TRITON=$INT8_VARATTN_KV_TRITON VARATTN_KV_LOG=$INT8_VARATTN_KV_TRITON_LOG"
echo "VARATTN_KV_BLOCK_M=$INT8_VARATTN_KV_BLOCK_M BLOCK_N=$INT8_VARATTN_KV_BLOCK_N BLOCK_K=$INT8_VARATTN_KV_BLOCK_K WARPS=$INT8_VARATTN_KV_NUM_WARPS STAGES=$INT8_VARATTN_KV_NUM_STAGES"
echo "QKV_BLOCK_M=$INT8_QKV_BLOCK_M BLOCK_N=$INT8_QKV_BLOCK_N BLOCK_K=$INT8_QKV_BLOCK_K GROUP_M=$INT8_QKV_GROUP_M WARPS=$INT8_QKV_NUM_WARPS STAGES=$INT8_QKV_NUM_STAGES"
# DataLoader tuning to reduce I/O stalls
export DATA_PREFETCH_FACTOR=${DATA_PREFETCH_FACTOR:-4}                  # prefetch per worker (>=2 when num_workers>0)
export DATA_PERSISTENT_WORKERS=${DATA_PERSISTENT_WORKERS:-1}            # keep workers alive across epochs
# rocBLAS/Lt logging (set to 1 to trace kernel selection)
export PYTORCH_ROCBLASLT_LOG_LEVEL=${PYTORCH_ROCBLASLT_LOG_LEVEL:-1}
export ROCBLASLT_LOG_LEVEL=${ROCBLASLT_LOG_LEVEL:-1}
export ROCBLASLT_LOG_MASK=${ROCBLASLT_LOG_MASK:-0}
export HIPBLASLT_LOG_MASK=${HIPBLASLT_LOG_MASK:-0}
export ROCBLASLT_LOG_DIR=${ROCBLASLT_LOG_DIR:-/lustre/orion/proj-shared/lrn036/yoonh/super-res-torchlight/examples/rocblaslt_logs}
mkdir -p "${ROCBLASLT_LOG_DIR}"
export ROCBLASLT_LOG_FILE=${ROCBLASLT_LOG_FILE:-${ROCBLASLT_LOG_DIR}/rocblaslt_${SLURM_JOB_ID}_rank${SLURM_PROCID}.log}
export HIPBLASLT_LOG_FILE=${HIPBLASLT_LOG_FILE:-${ROCBLASLT_LOG_DIR}/hipblaslt_${SLURM_JOB_ID}_rank${SLURM_PROCID}.log}
# Fallback: if rocBLAS (non-Lt) is used, capture its log too.
export ROCBLAS_LOG_LEVEL=${ROCBLAS_LOG_LEVEL:-1}
export ROCBLAS_LOG_MASK=${ROCBLAS_LOG_MASK:-0}
export ROCBLAS_LOG_FILE=${ROCBLAS_LOG_FILE:-${ROCBLASLT_LOG_DIR}/rocblas_${SLURM_JOB_ID}_rank${SLURM_PROCID}.log}

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

# load omnistat
ml use /autofs/nccs-svm1_sw/crusher/amdsw/modules
ml omnistat-wrapper

# start omnistat - enable data collection
${OMNISTAT_WRAPPER} usermode --start --interval 1


export FI_MR_CACHE_MONITOR=kdreg2     # Required to avoid a deadlock.
export FI_CXI_DEFAULT_CQ_SIZE=131072  # Ask the network stack to allocate additional space to process message completions.
export FI_CXI_DEFAULT_TX_SIZE=2048    # Ask the network stack to allocate additional space to hold pending outgoing messages.
export FI_CXI_RX_MATCH_MODE=hybrid    # Allow the network stack to transition to software mode if necessary.

export NCCL_NET_GDR_LEVEL=3           # Typically improves performance, but remove this setting if you encounter a hang/crash.
export NCCL_CROSS_NIC=1               # On large systems, this NCCL setting has been found to improve performance
export NCCL_SOCKET_IFNAME=hsn0        # NCCL/RCCL will use the high speed network to coordinate startup.
export TORCH_NCCL_HIGH_PRIORITY=1     # Use high priority stream for the NCCL/RCCL Communicator.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}                # lower verbosity for perf runs; set INFO when debugging hangs
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}  # fail fast on async errors
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-}                   # leave empty unless debugging
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}    # turn on DETAIL only when debugging
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576} # enable flight recorder (1MB) for hang diagnostics

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
python ./intermediate_downscaling.py ../configs/test_int8_8m.yaml


# stop omnistat - generate summary report and stop data collection
${OMNISTAT_WRAPPER} usermode --stopexporters
${OMNISTAT_WRAPPER} query --interval 1 --job ${SLURM_JOB_ID} --pdf omnistat.${SLURM_JOB_ID}.pdf
${OMNISTAT_WRAPPER} usermode --stopserver
mv /tmp/omnistat/${SLURM_JOB_ID} data_omnistat.${SLURM_JOB_ID}
