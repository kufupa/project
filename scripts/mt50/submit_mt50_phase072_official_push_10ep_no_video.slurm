#!/usr/bin/env bash
# Phase072: official LeRobot push-v3, 10 episodes, no videos.
#
#   sbatch --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/mt50/submit_mt50_phase072_official_push_10ep_no_video.slurm
#
#SBATCH --job-name=mt50-p072-push-novid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=mt50_phase072_push_10ep_no_video_%j.out
#SBATCH --error=mt50_phase072_push_10ep_no_video_%j.err
#SBATCH --export=NIL

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
# shellcheck source=../slurm/common_env.sh
source "${_COMMON}"

slurm_resolve_project_root "scripts/mt50/run_official_lerobot_mt50_eval.sh"
cd "${PROJECT_ROOT}"

export MT50_PHASE071_TASK="${MT50_PHASE071_TASK:-push-v3}"
export MT50_PHASE071_EPISODES="${MT50_PHASE071_EPISODES:-10}"
export MT50_PHASE071_SEED="${MT50_PHASE071_SEED:-1000}"
export MT50_PHASE071_OUTPUT_ROOT="${MT50_PHASE071_OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/MT50_Phase072_official_lerobot_push_10ep}"
export MT50_LEROBOT_MAX_EPISODES_RENDERED="${MT50_LEROBOT_MAX_EPISODES_RENDERED:-0}"

echo "[mt50-p072-push-novid] host=$(hostname) job=${SLURM_JOB_ID:-local} PROJECT_ROOT=${PROJECT_ROOT}"
echo "[mt50-p072-push-novid] task=${MT50_PHASE071_TASK} episodes=${MT50_PHASE071_EPISODES} max_episodes_rendered=${MT50_LEROBOT_MAX_EPISODES_RENDERED}"
echo "[mt50-p072-push-novid] output_root=${MT50_PHASE071_OUTPUT_ROOT}"

exec bash "${PROJECT_ROOT}/scripts/mt50/run_official_lerobot_mt50_eval.sh"
