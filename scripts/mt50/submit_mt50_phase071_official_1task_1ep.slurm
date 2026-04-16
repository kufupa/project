#!/usr/bin/env bash
# GPU smoke: official LeRobot eval, one Meta-World task × 1 episode, videos on (library default).
# Usage:
#   sbatch --chdir=/vol/bitbucket/aa6622/project scripts/mt50/submit_mt50_phase071_official_1task_1ep.slurm
#
#SBATCH --job-name=mt50-p071-1task
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=mt50_phase071_official_1task_%j.out
#SBATCH --error=mt50_phase071_official_1task_%j.err
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

export MT50_PHASE071_TASK="${MT50_PHASE071_TASK:-assembly-v3}"
export MT50_PHASE071_EPISODES="${MT50_PHASE071_EPISODES:-1}"
export MT50_PHASE071_SEED="${MT50_PHASE071_SEED:-1000}"
export MT50_PHASE071_OUTPUT_ROOT="${MT50_PHASE071_OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/MT50_Phase071_official_lerobot_1task_1ep}"

echo "[mt50-p071-1task] host=$(hostname) job=${SLURM_JOB_ID:-local} PROJECT_ROOT=${PROJECT_ROOT}"

exec bash "${PROJECT_ROOT}/scripts/mt50/run_official_lerobot_mt50_eval.sh"
