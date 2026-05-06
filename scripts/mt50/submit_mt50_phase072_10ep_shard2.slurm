#!/usr/bin/env bash
# Phase072: official LeRobot MT50 shard 2/3 — 16 tasks × 10 episodes.
#
#   sbatch --chdir=/vol/bitbucket/aa6622/project scripts/mt50/submit_mt50_phase072_10ep_shard2.slurm
#
#SBATCH --job-name=mt50-p072-s2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=mt50_phase072_10ep_shard2_%j.out
#SBATCH --error=mt50_phase072_10ep_shard2_%j.err
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

export MT50_PHASE071_EPISODES="${MT50_PHASE071_EPISODES:-10}"
export MT50_PHASE071_SEED="${MT50_PHASE071_SEED:-1000}"
export MT50_PHASE071_OUTPUT_ROOT="${MT50_PHASE071_OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/MT50_Phase072_official_lerobot_10ep/shard2}"
export MT50_PHASE071_TASK='plate-slide-back-v3,plate-slide-side-v3,plate-slide-v3,push-back-v3,push-v3,push-wall-v3,reach-v3,reach-wall-v3,shelf-place-v3,soccer-v3,stick-pull-v3,stick-push-v3,sweep-into-v3,sweep-v3,window-open-v3,window-close-v3'

echo "[mt50-p072-s2] host=$(hostname) job=${SLURM_JOB_ID:-local} PROJECT_ROOT=${PROJECT_ROOT} tasks=16 episodes=${MT50_PHASE071_EPISODES}"

exec bash "${PROJECT_ROOT}/scripts/mt50/run_official_lerobot_mt50_eval.sh"
