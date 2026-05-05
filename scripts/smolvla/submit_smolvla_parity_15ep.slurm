#!/bin/bash
#SBATCH --job-name=smolvla-parity-15ep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=smolvla_parity_15ep_%j.out

set -euo pipefail

if [[ -n "${SMOLVLA_PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${SMOLVLA_PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SUBMIT_DIR}/../.." && pwd)"
fi

cd "${PROJECT_ROOT}"

export SMOLVLA_PARITY_EPISODES="${SMOLVLA_PARITY_EPISODES:-15}"
export SMOLVLA_PARITY_SEED="${SMOLVLA_PARITY_SEED:-1000}"
export SMOLVLA_PARITY_TASK="${SMOLVLA_PARITY_TASK:-push-v3}"
export SMOLVLA_PARITY_MAX_STEPS="${SMOLVLA_PARITY_MAX_STEPS:-120}"
export SMOLVLA_SAVE_FRAMES="${SMOLVLA_SAVE_FRAMES:-true}"
export SMOLVLA_METAWORLD_CAMERA_NAME="${SMOLVLA_METAWORLD_CAMERA_NAME:-corner2}"
export SMOLVLA_FLIP_CORNER2="${SMOLVLA_FLIP_CORNER2:-true}"
export SMOLVLA_INIT_CHECKPOINT="${SMOLVLA_INIT_CHECKPOINT:-jadechoghari/smolvla_metaworld}"

exec bash "${PROJECT_ROOT}/scripts/smolvla/run_pushv3_smolvla_parity_benchmark.sh"
