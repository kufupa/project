#!/usr/bin/env bash
# One GPU allocation: single MetaWorld task, 1 episode, same env pins as MT50 phase07 baseline.
# Usage (login node):
#   sbatch --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/mt50/submit_phase07_one_task_1ep_gpu_smoke.slurm
#
# IMPORTANT: `--export=NIL` clears the submit shell environment. Variables you set on the login
# node (e.g. SMOLVLA_PARITY_TASK) are NOT visible unless you pass them explicitly, e.g.:
#   sbatch --chdir=/vol/bitbucket/aa6622/project --export=NIL,SMOLVLA_PARITY_TASK=pick-out-of-hole-v3,SMOLVLA_PARITY_EPISODES=10,SMOLVLA_PARITY_VIDEO=true,SMOLVLA_PARITY_OUTPUT_ROOT=/vol/bitbucket/aa6622/project/artifacts/MT50_Phase07/reruns/my_run \
#     scripts/mt50/submit_phase07_one_task_1ep_gpu_smoke.slurm
# For a fixed multi-episode + video rerun, prefer scripts/mt50/submit_mt50_rerun_pick_out_of_hole_video_10ep.slurm (all vars set inside the batch file).

#SBATCH --job-name=smolvla-1ep-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=smolvla_phase07_one_task_1ep_%j.out
#SBATCH --error=smolvla_phase07_one_task_1ep_%j.err

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
slurm_export_hf_torch_cache "phase07-one-task-1ep-smoke"

# Match MT50 Phase07: local hub snapshots + fail fast if cache cold (avoids long HF stalls).
export MT50_PHASE07_USE_LOCAL_HF_SNAPSHOT="${MT50_PHASE07_USE_LOCAL_HF_SNAPSHOT:-true}"
export SMOLVLA_LOCAL_FILES_ONLY="${SMOLVLA_LOCAL_FILES_ONLY:-true}"
export SMOLVLA_PREFER_LOCAL_HF_SNAPSHOT="${SMOLVLA_PREFER_LOCAL_HF_SNAPSHOT:-true}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

export WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"

# Match MT50 phase07 defaults for I/O flags
export SMOLVLA_SAVE_FRAMES="${SMOLVLA_SAVE_FRAMES:-false}"
export SMOLVLA_PARITY_VIDEO="${SMOLVLA_PARITY_VIDEO:-false}"
export SMOLVLA_SAVE_ACTIONS="${SMOLVLA_SAVE_ACTIONS:-false}"
export SMOLVLA_FLIP_CORNER2="${SMOLVLA_FLIP_CORNER2:-${MT50_FLIP_CORNER2:-true}}"

# Single task / one episode (override SMOLVLA_PARITY_TASK if needed)
export SMOLVLA_PARITY_TASK="${SMOLVLA_PARITY_TASK:-assembly-v3}"
export SMOLVLA_PARITY_EPISODES="${SMOLVLA_PARITY_EPISODES:-1}"
export SMOLVLA_PARITY_SEED="${SMOLVLA_PARITY_SEED:-1000}"
export SMOLVLA_PARITY_MAX_STEPS="${SMOLVLA_PARITY_MAX_STEPS:-120}"
export SMOLVLA_INIT_CHECKPOINT="${SMOLVLA_INIT_CHECKPOINT:-${MT50_PHASE07_CHECKPOINT:-jadechoghari/smolvla_metaworld}}"

export SMOLVLA_PARITY_OUTPUT_ROOT="${SMOLVLA_PARITY_OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/phase07_one_task_1ep_smoke}"

echo "[phase07-1ep-smoke] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[phase07-1ep-smoke] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[phase07-1ep-smoke] SMOLVLA_LEROBOT_ENV_DIR=${SMOLVLA_LEROBOT_ENV_DIR}"
echo "[phase07-1ep-smoke] HF_HOME=${HF_HOME:-}"
echo "[phase07-1ep-smoke] task=${SMOLVLA_PARITY_TASK} episodes=${SMOLVLA_PARITY_EPISODES} seed=${SMOLVLA_PARITY_SEED}"

exec bash "${PROJECT_ROOT}/scripts/smolvla/legacy_run_pushv3_smolvla_parity_benchmark.sh"
