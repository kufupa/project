#!/usr/bin/env bash
# Phase17: official LeRobot MT50 shard 0/2 — very-hard first, 25 tasks × 10 episodes, no videos.
#
#   sbatch --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/mt50/submit_mt50_phase17_full_10ep_2gpu_shard0.slurm
#
#SBATCH --job-name=mt50-p17-s0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=mt50_phase17_full_10ep_2gpu_shard0_%j.out
#SBATCH --error=mt50_phase17_full_10ep_2gpu_shard0_%j.err
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
export MT50_PHASE071_OUTPUT_ROOT="${MT50_PHASE071_OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/MT50_Phase17_official_lerobot_10ep_2gpu_full/shard0}"
export MT50_PHASE071_TASK='pick-place-wall-v3,shelf-place-v3,disassemble-v3,stick-pull-v3,stick-push-v3,assembly-v3,basketball-v3,bin-picking-v3,box-close-v3,button-press-topdown-v3,button-press-topdown-wall-v3,button-press-v3,button-press-wall-v3,coffee-button-v3,coffee-pull-v3,coffee-push-v3,dial-turn-v3,door-close-v3,door-lock-v3,door-open-v3,door-unlock-v3,drawer-close-v3,drawer-open-v3,faucet-close-v3,faucet-open-v3'
export MT50_LEROBOT_MAX_EPISODES_RENDERED="${MT50_LEROBOT_MAX_EPISODES_RENDERED:-0}"

echo "[mt50-p17-s0] host=$(hostname) job=${SLURM_JOB_ID:-local} PROJECT_ROOT=${PROJECT_ROOT}"
echo "[mt50-p17-s0] tasks=25 episodes=${MT50_PHASE071_EPISODES} seed=${MT50_PHASE071_SEED} max_episodes_rendered=${MT50_LEROBOT_MAX_EPISODES_RENDERED}"
echo "[mt50-p17-s0] task=${MT50_PHASE071_TASK}"
echo "[mt50-p17-s0] output_root=${MT50_PHASE071_OUTPUT_ROOT}"

exec bash "${PROJECT_ROOT}/scripts/mt50/run_official_lerobot_mt50_eval.sh"
