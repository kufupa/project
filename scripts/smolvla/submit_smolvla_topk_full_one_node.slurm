#!/usr/bin/env bash
#SBATCH --job-name=smolvla-topk-full
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=smolvla_topk_full_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_BY_SCRIPT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${PROJECT_ROOT_BY_SCRIPT}/scripts/smolvla/run_oracle_topk_smolvla_full.sh" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT_BY_SCRIPT}"
else
  PROJECT_ROOT="$(pwd)"
fi
if [[ ! -f "${PROJECT_ROOT}/scripts/smolvla/run_oracle_topk_smolvla_full.sh" ]]; then
  echo "error: cannot locate project root from ${PROJECT_ROOT_BY_SCRIPT} or ${PROJECT_ROOT}" >&2
  exit 2
fi
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SMOLVLA_TOPK_LAUNCH_MODE="${SMOLVLA_TOPK_LAUNCH_MODE:-dry-run}"
export SMOLVLA_SAVE_FRAMES=false
export EPISODES_PER_TARGET="${EPISODES_PER_TARGET:-4}"
export ORACLE_RUN_DIR="${ORACLE_RUN_DIR:-${WORKSPACE_ROOT}/project/artifacts/phase06_oracle_baseline/run_20260411T131839Z_ep60_voracle_tpush_v3_s1000_r402093}"
if [[ ! -d "${ORACLE_RUN_DIR}" ]]; then
  echo "ORACLE_RUN_DIR does not exist: ${ORACLE_RUN_DIR}" >&2
  exit 2
fi
export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"

bash "${PROJECT_ROOT}/scripts/smolvla/run_oracle_topk_smolvla_full.sh"
