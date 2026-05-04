#!/usr/bin/env bash
#SBATCH --job-name=mt50-phase07-smolvla
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=mt50_phase07_smolvla_baseline_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh" ]]; then
    COMMON_ENV="${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
  fi
fi

if [[ -z "${COMMON_ENV:-}" ]]; then
  COMMON_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../slurm/common_env.sh"
fi

source "${COMMON_ENV}"
slurm_resolve_project_root "scripts/mt50/run_phase07_smolvla_baseline.sh"
cd "${PROJECT_ROOT}"

slurm_export_pythonpath
slurm_export_hf_torch_cache "mt50-phase07-smolvla-baseline"

export WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"
export MT50_PHASE07_OUTPUT_ROOT="${MT50_PHASE07_OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/MT50_Phase07}"
export MT50_INCLUDE_DIFFICULTIES="${MT50_INCLUDE_DIFFICULTIES:-easy,medium,hard,very_hard,unclassified}"
export MT50_PHASE07_EPISODES="${MT50_PHASE07_EPISODES:-10}"
export MT50_PHASE07_SEED="${MT50_PHASE07_SEED:-1000}"
export MT50_PHASE07_MAX_STEPS="${MT50_PHASE07_MAX_STEPS:-120}"
export MT50_PHASE07_CHECKPOINT="${MT50_PHASE07_CHECKPOINT:-${SMOLVLA_INIT_CHECKPOINT:-jadechoghari/smolvla_metaworld}}"
export MT50_FLIP_CORNER2="${MT50_FLIP_CORNER2:-${SMOLVLA_FLIP_CORNER2:-true}}"
export SMOLVLA_FLIP_CORNER2="${SMOLVLA_FLIP_CORNER2:-${MT50_FLIP_CORNER2}}"
export MT50_TASK_DIFFICULTY_JSON="${MT50_TASK_DIFFICULTY_JSON:-${PROJECT_ROOT}/scripts/mt50/mt50_phase07_task_difficulties.json}"
export MT50_PHASE07_SAVE_FRAMES="${MT50_PHASE07_SAVE_FRAMES:-false}"
export MT50_PHASE07_VIDEO="${MT50_PHASE07_VIDEO:-false}"
export MT50_PHASE07_SAVE_ACTIONS="${MT50_PHASE07_SAVE_ACTIONS:-false}"

# Cached HF: resolve hub repo ids to local snapshots; fail fast if cache missing (no multi-hour hub stall).
export PROJECT_ROOT="${PROJECT_ROOT}"
export MT50_PHASE07_USE_LOCAL_HF_SNAPSHOT="${MT50_PHASE07_USE_LOCAL_HF_SNAPSHOT:-true}"
export SMOLVLA_LOCAL_FILES_ONLY="${SMOLVLA_LOCAL_FILES_ONLY:-true}"
export SMOLVLA_PREFER_LOCAL_HF_SNAPSHOT="${SMOLVLA_PREFER_LOCAL_HF_SNAPSHOT:-true}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "[mt50:phase07-slurm] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[mt50:phase07-slurm] project_root=${PROJECT_ROOT}"
echo "[mt50:phase07-slurm] output_root=${MT50_PHASE07_OUTPUT_ROOT}"
echo "[mt50:phase07-slurm] include_difficulties=${MT50_INCLUDE_DIFFICULTIES}"

mkdir -p "${MT50_PHASE07_OUTPUT_ROOT}"

exec bash "${PROJECT_ROOT}/scripts/mt50/run_phase07_smolvla_baseline.sh"
