#!/bin/bash
#SBATCH --job-name=oracle-baseline-15ep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=oracle_baseline_15ep_%j.out

set -euo pipefail

if [[ -n "${ORACLE_PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${ORACLE_PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SUBMIT_DIR}/../.." && pwd)"
fi

cd "${PROJECT_ROOT}"

export ORACLE_BASELINE_EPISODES="${ORACLE_BASELINE_EPISODES:-15}"
export ORACLE_BASELINE_SEED="${ORACLE_BASELINE_SEED:-1000}"
export ORACLE_BASELINE_TASK="${ORACLE_BASELINE_TASK:-push-v3}"
export ORACLE_BASELINE_EPISODE_LENGTH="${ORACLE_BASELINE_EPISODE_LENGTH:-120}"
export ORACLE_SAVE_FRAMES="${ORACLE_SAVE_FRAMES:-true}"
export ORACLE_METAWORLD_CAMERA_NAME="${ORACLE_METAWORLD_CAMERA_NAME:-corner2}"
export ORACLE_FLIP_CORNER2="${ORACLE_FLIP_CORNER2:-true}"
export ORACLE_BASELINE_VIDEO="${ORACLE_BASELINE_VIDEO:-true}"

exec bash "${PROJECT_ROOT}/scripts/oracle/run_oracle_baseline_eval.sh"
