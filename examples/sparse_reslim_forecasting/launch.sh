#!/usr/bin/env bash
#SBATCH -A lrn036
#SBATCH -J sparse-reslim-forecast
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 01:00:00
#SBATCH -q debug
#SBATCH -o sparse-reslim-forecast-%j.out
#SBATCH -e sparse-reslim-forecast-%j.out

set -euo pipefail

module load PrgEnv-gnu
module load rocm/7.1.1
module load craype-accel-amd-gfx90a

export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="/tmp/miopen_${USER}_${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-7}"
export PYTHONNOUSERSITE=1
mkdir -p "${MIOPEN_USER_DB_PATH}"

if [[ "${SLURM_NTASKS:-1}" -gt 1 ]]; then
  MASTER_HOST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
  export MASTER_ADDR="${MASTER_ADDR:-${MASTER_HOST}}"
  export MASTER_PORT="${MASTER_PORT:-29500}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${DEFAULT_REPO_ROOT}}}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/sparse_reslim_forecasting.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/lustre/orion/lrn036/world-shared/xf9/torch210/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
srun --kill-on-bad-exit=1 \
  "${PYTHON_BIN}" examples/sparse_reslim_forecasting/train.py "${CONFIG_PATH}"
