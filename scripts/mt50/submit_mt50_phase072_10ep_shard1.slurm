#!/usr/bin/env bash
# Phase072: official LeRobot MT50 shard 1/3 — 17 tasks × 10 episodes, no videos by default.
#
#   sbatch --chdir=/vol/bitbucket/aa6622/project scripts/mt50/submit_mt50_phase072_10ep_shard1.slurm
#
#SBATCH --job-name=mt50-p072-s1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=mt50_phase072_10ep_shard1_%j.out
#SBATCH --error=mt50_phase072_10ep_shard1_%j.err
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
export MT50_PHASE071_OUTPUT_ROOT="${MT50_PHASE071_OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/MT50_Phase072_official_lerobot_10ep/shard1}"
export MT50_PHASE071_TASK='drawer-close-v3,drawer-open-v3,faucet-close-v3,faucet-open-v3,hammer-v3,hand-insert-v3,handle-press-side-v3,handle-press-v3,handle-pull-side-v3,handle-pull-v3,lever-pull-v3,peg-insert-side-v3,peg-unplug-side-v3,pick-out-of-hole-v3,pick-place-v3,pick-place-wall-v3,plate-slide-back-side-v3'
export MT50_LEROBOT_MAX_EPISODES_RENDERED="${MT50_LEROBOT_MAX_EPISODES_RENDERED:-0}"

echo "[mt50-p072-s1] host=$(hostname) job=${SLURM_JOB_ID:-local} PROJECT_ROOT=${PROJECT_ROOT} tasks=17 episodes=${MT50_PHASE071_EPISODES} max_episodes_rendered=${MT50_LEROBOT_MAX_EPISODES_RENDERED}"

exec bash "${PROJECT_ROOT}/scripts/mt50/run_official_lerobot_mt50_eval.sh"
