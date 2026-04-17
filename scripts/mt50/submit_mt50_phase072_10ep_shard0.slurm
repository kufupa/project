#!/usr/bin/env bash
# Phase072: official LeRobot MT50 shard 0/3 — 17 tasks × 10 episodes.
# Task list is embedded (avoid sbatch --export comma split).
#
#   sbatch --chdir=/vol/bitbucket/aa6622/project scripts/mt50/submit_mt50_phase072_10ep_shard0.slurm
#
#SBATCH --job-name=mt50-p072-s0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=mt50_phase072_10ep_shard0_%j.out
#SBATCH --error=mt50_phase072_10ep_shard0_%j.err
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
export MT50_PHASE071_OUTPUT_ROOT="${MT50_PHASE071_OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/MT50_Phase072_official_lerobot_10ep/shard0}"
export MT50_PHASE071_TASK='assembly-v3,basketball-v3,bin-picking-v3,box-close-v3,button-press-topdown-v3,button-press-topdown-wall-v3,button-press-v3,button-press-wall-v3,coffee-button-v3,coffee-pull-v3,coffee-push-v3,dial-turn-v3,disassemble-v3,door-close-v3,door-lock-v3,door-open-v3,door-unlock-v3'

echo "[mt50-p072-s0] host=$(hostname) job=${SLURM_JOB_ID:-local} PROJECT_ROOT=${PROJECT_ROOT} tasks=17 episodes=${MT50_PHASE071_EPISODES}"

exec bash "${PROJECT_ROOT}/scripts/mt50/run_official_lerobot_mt50_eval.sh"
