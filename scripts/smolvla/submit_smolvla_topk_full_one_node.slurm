#!/usr/bin/env bash
#SBATCH --job-name=smolvla-topk-full
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=smolvla_topk_full_%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh" ]]; then
  _SLURM_COMMON="${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
else
  _SLURM_COMMON="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/../slurm/common_env.sh"
fi
# shellcheck source=../slurm/common_env.sh
source "${_SLURM_COMMON}"
slurm_resolve_project_root "scripts/smolvla/run_oracle_topk_smolvla_full.sh"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "smolvla-topk-full"

WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export SMOLVLA_TOPK_LAUNCH_MODE="${SMOLVLA_TOPK_LAUNCH_MODE:-dry-run}"
export SMOLVLA_ALLOW_NESTED_SBATCH="${SMOLVLA_ALLOW_NESTED_SBATCH:-0}"
export SMOLVLA_SAVE_FRAMES=false
export EPISODES_PER_TARGET="${EPISODES_PER_TARGET:-4}"
export ORACLE_RUN_DIR="${ORACLE_RUN_DIR:-${WORKSPACE_ROOT}/project/artifacts/phase06_oracle_baseline/run_20260411T131839Z_ep60_voracle_tpush_v3_s1000_r402093}"
if [[ ! -d "${ORACLE_RUN_DIR}" ]]; then
  echo "ORACLE_RUN_DIR does not exist: ${ORACLE_RUN_DIR}" >&2
  exit 2
fi
export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"
echo "smolvla-topk-submit: launch_mode=${SMOLVLA_TOPK_LAUNCH_MODE} allow_nested_sbatch=${SMOLVLA_ALLOW_NESTED_SBATCH}"

bash "${PROJECT_ROOT}/scripts/smolvla/run_oracle_topk_smolvla_full.sh"
