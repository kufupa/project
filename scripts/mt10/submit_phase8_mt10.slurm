#!/usr/bin/env bash
#SBATCH --job-name=mt10-p8-seggrpo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --partition=t4
#SBATCH --output=logs/mt10_phase8_%j.out
#SBATCH --error=logs/mt10_phase8_%j.err
#
# Sequential Segment-GRPO across MT10 tasks (one allocation, no nested sbatch).
#
set -euo pipefail

if [[ -n "${MT10_PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${MT10_PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SUBMIT_DIR}/../.." && pwd)"
fi

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"
VOL_CACHE="${WORKSPACE_ROOT}/.cache"
export HF_HOME="${HF_HOME:-${VOL_CACHE}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${VOL_CACHE}/torch}"
export JEPA_REPO="${JEPA_REPO:-${TORCH_HOME}/hub/facebookresearch_jepa-wms_main}"
export SMOLVLA_CHECKPOINT="${SMOLVLA_CHECKPOINT:-jadechoghari/smolvla_metaworld}"

echo "[mt10:p8-slurm] host=$(hostname) job=${SLURM_JOB_ID:-local}"

exec bash "${PROJECT_ROOT}/scripts/mt10/run_phase8_mt10.sh"
