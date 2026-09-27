#!/usr/bin/env bash
#SBATCH -A lrn036
#SBATCH -J sparse-reslim-forecast
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 01:00:00
#SBATCH -q debug
#SBATCH -o sparse-reslim-forecast-%j.out
#SBATCH -e sparse-reslim-forecast-%j.out

set -euo pipefail

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.4.0

export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="/tmp/miopen_${USER}_${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-7}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ERA5_DIR="${ERA5_DIR:-/lustre/orion/world-shared/lrn036/jyc/frontier/ClimaX-v2/data/ERA5-1hr-superres/1.0_deg/}"

cd "${REPO_ROOT}"
srun python examples/sparse_reslim_forecasting/train.py "${ERA5_DIR}" \
  --max-epochs 30 \
  --batch-size 1 \
  --pred-range 120 \
  --input-vars 2m_temperature \
  --output-vars 2m_temperature \
  --keep-ratio 0.25 \
  --accelerator gpu \
  --devices 1
