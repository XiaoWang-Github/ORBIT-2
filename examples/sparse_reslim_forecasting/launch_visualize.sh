#!/usr/bin/env bash
#SBATCH -A lrn036
#SBATCH -J sparse-reslim-visualize
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 00:10:00
#SBATCH -q debug
#SBATCH -o sparse-reslim-visualize-%j.out
#SBATCH -e sparse-reslim-visualize-%j.out

set -euo pipefail

module load PrgEnv-gnu
module load rocm/7.1.1
module load craype-accel-amd-gfx90a

export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="/tmp/miopen_${USER}_${SLURM_JOB_ID}"
export MPLCONFIGDIR="/tmp/matplotlib_${USER}_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="/tmp/cache_${USER}_${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-7}"
export PYTHONNOUSERSITE=1
mkdir -p "${MIOPEN_USER_DB_PATH}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${DEFAULT_REPO_ROOT}}}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/sparse_reslim_forecasting.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/lustre/orion/lrn036/world-shared/xf9/torch210/bin/python}"
SPLIT="${SPLIT:-test}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

VISUALIZE_ARGS=(
  "${CONFIG_PATH}"
  --split "${SPLIT}"
  --sample-index "${SAMPLE_INDEX}"
)
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  VISUALIZE_ARGS+=(--checkpoint "${CHECKPOINT_PATH}")
fi
if [[ -n "${VARIABLE:-}" ]]; then
  VISUALIZE_ARGS+=(--variable "${VARIABLE}")
fi
if [[ -n "${OUTPUT_PATH:-}" ]]; then
  VISUALIZE_ARGS+=(--output "${OUTPUT_PATH}")
fi

cd "${REPO_ROOT}"
srun "${PYTHON_BIN}" \
  examples/sparse_reslim_forecasting/visualize.py "${VISUALIZE_ARGS[@]}"
