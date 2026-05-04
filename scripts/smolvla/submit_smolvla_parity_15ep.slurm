#!/bin/bash
#SBATCH --job-name=smolvla-parity-15ep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=smolvla_parity_15ep_%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh" ]]; then
  _SLURM_COMMON="${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
else
  _SLURM_COMMON="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/../slurm/common_env.sh"
fi
# shellcheck source=../slurm/common_env.sh
source "${_SLURM_COMMON}"
slurm_resolve_project_root "scripts/smolvla/legacy_run_pushv3_smolvla_parity_benchmark.sh"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "smolvla-parity-15ep"

export SMOLVLA_PARITY_EPISODES="${SMOLVLA_PARITY_EPISODES:-15}"
export SMOLVLA_PARITY_SEED="${SMOLVLA_PARITY_SEED:-1000}"
export SMOLVLA_PARITY_TASK="${SMOLVLA_PARITY_TASK:-push-v3}"
export SMOLVLA_PARITY_MAX_STEPS="${SMOLVLA_PARITY_MAX_STEPS:-120}"
export SMOLVLA_SAVE_FRAMES="${SMOLVLA_SAVE_FRAMES:-true}"
export SMOLVLA_METAWORLD_CAMERA_NAME="${SMOLVLA_METAWORLD_CAMERA_NAME:-corner2}"
export SMOLVLA_FLIP_CORNER2="${SMOLVLA_FLIP_CORNER2:-true}"
export SMOLVLA_INIT_CHECKPOINT="${SMOLVLA_INIT_CHECKPOINT:-jadechoghari/smolvla_metaworld}"

exec bash "${PROJECT_ROOT}/scripts/smolvla/legacy_run_pushv3_smolvla_parity_benchmark.sh"
