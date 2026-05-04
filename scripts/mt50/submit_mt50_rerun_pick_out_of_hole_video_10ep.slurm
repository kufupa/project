#!/usr/bin/env bash
# Dedicated rerun: pick-out-of-hole-v3, 10 episodes, video on, GPU.
# All knobs are set HERE so `sbatch --export=NIL` still works (login-shell exports are NOT passed).
#
# Submit from login node:
#   sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL \
#     scripts/mt50/submit_mt50_rerun_pick_out_of_hole_video_10ep.slurm

#SBATCH --job-name=smolvla-pooh-vid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=smolvla_pick_out_of_hole_video_10ep_%j.out
#SBATCH --error=smolvla_pick_out_of_hole_video_10ep_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

_COMMON=""
if [[ -f "/vol/bitbucket/aa6622/project/scripts/slurm/common_env.sh" ]]; then
  _COMMON="/vol/bitbucket/aa6622/project/scripts/slurm/common_env.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh" ]]; then
  _COMMON="${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
else
  _SELF="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  _COMMON="${_SELF}/../slurm/common_env.sh"
fi
if [[ ! -f "${_COMMON}" ]]; then
  echo "error: cannot find scripts/slurm/common_env.sh" >&2
  exit 2
fi
# shellcheck source=../slurm/common_env.sh
source "${_COMMON}"

slurm_resolve_project_root "scripts/smolvla/legacy_run_pushv3_smolvla_parity_benchmark.sh"
cd "${PROJECT_ROOT}"

slurm_export_pythonpath
slurm_export_hf_torch_cache "mt50-rerun-pick-out-of-hole-video"

export MT50_PHASE07_USE_LOCAL_HF_SNAPSHOT="${MT50_PHASE07_USE_LOCAL_HF_SNAPSHOT:-true}"
export SMOLVLA_LOCAL_FILES_ONLY="${SMOLVLA_LOCAL_FILES_ONLY:-true}"
export SMOLVLA_PREFER_LOCAL_HF_SNAPSHOT="${SMOLVLA_PREFER_LOCAL_HF_SNAPSHOT:-true}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

export WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"

# --- Rerun-specific (do not rely on sbatch environment) ---
export SMOLVLA_PARITY_TASK="pick-out-of-hole-v3"
export SMOLVLA_PARITY_EPISODES="10"
export SMOLVLA_PARITY_SEED="1000"
export SMOLVLA_PARITY_MAX_STEPS="120"
export SMOLVLA_INIT_CHECKPOINT="${SMOLVLA_INIT_CHECKPOINT:-jadechoghari/smolvla_metaworld}"
export SMOLVLA_PARITY_OUTPUT_ROOT="${PROJECT_ROOT}/artifacts/MT50_Phase07/reruns/smolvla_pick_out_of_hole_video_only"
export SMOLVLA_PARITY_VIDEO="true"
export SMOLVLA_SAVE_FRAMES="${SMOLVLA_SAVE_FRAMES:-false}"
export SMOLVLA_SAVE_ACTIONS="${SMOLVLA_SAVE_ACTIONS:-false}"
export SMOLVLA_PARITY_FPS="${SMOLVLA_PARITY_FPS:-30}"
export SMOLVLA_PARITY_OVERLAY_MODE="${SMOLVLA_PARITY_OVERLAY_MODE:-reward_delta}"
export SMOLVLA_FLIP_CORNER2="${SMOLVLA_FLIP_CORNER2:-${MT50_FLIP_CORNER2:-true}}"

# Default inner timeout is episodes*600+1800 (10 ep => 7800s); allow extra headroom for cold load + video encode.
export SMOLVLA_PARITY_EVAL_TIMEOUT_SEC="${SMOLVLA_PARITY_EVAL_TIMEOUT_SEC:-12000}"

echo "[mt50-rerun-pooh-video] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[mt50-rerun-pooh-video] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[mt50-rerun-pooh-video] OUTPUT_ROOT=${SMOLVLA_PARITY_OUTPUT_ROOT}"
echo "[mt50-rerun-pooh-video] task=${SMOLVLA_PARITY_TASK} episodes=${SMOLVLA_PARITY_EPISODES} seed=${SMOLVLA_PARITY_SEED} video=${SMOLVLA_PARITY_VIDEO}"
echo "[mt50-rerun-pooh-video] eval_timeout_s=${SMOLVLA_PARITY_EVAL_TIMEOUT_SEC}"

exec bash "${PROJECT_ROOT}/scripts/smolvla/legacy_run_pushv3_smolvla_parity_benchmark.sh"
