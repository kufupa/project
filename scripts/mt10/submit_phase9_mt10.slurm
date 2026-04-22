#!/usr/bin/env bash
#SBATCH --job-name=mt10-p9-oracle-wm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --partition=t4
#SBATCH --output=logs/mt10_phase9_%j.out
#SBATCH --error=logs/mt10_phase9_%j.err
#
# Sequential Phase9 oracle-vs-WM across MT10 tasks (one allocation, no nested sbatch).
#
set -euo pipefail

if [[ -n "${MT10_PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${MT10_PROJECT_ROOT}"
elif [[ -f "$(pwd)/scripts/mt10/mt10_tasks.sh" ]]; then
  PROJECT_ROOT="$(pwd)"
else
  _here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${_here}/../.." && pwd)"
fi

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

ART="${PROJECT_ROOT}/artifacts"
export MT10_PHASE9_OUTPUT_ROOT="${MT10_PHASE9_OUTPUT_ROOT:-${ART}/mt10_phase09_oracle_vs_wm_baseline}"

WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"
VOL_CACHE="${WORKSPACE_ROOT}/.cache"
export TORCH_HOME="${TORCH_HOME:-${VOL_CACHE}/torch}"
export JEPA_REPO="${JEPA_REPO:-${TORCH_HOME}/hub/facebookresearch_jepa-wms_main}"

if [[ -z "${JEPA_REPO:-}" || ! -d "${JEPA_REPO}" ]]; then
  echo "[mt10:p9-slurm] ERROR: JEPA_REPO must be an existing directory: ${JEPA_REPO:-}" >&2
  exit 3
fi

echo "[mt10:p9-slurm] host=$(hostname) job=${SLURM_JOB_ID:-local}"

exec bash "${PROJECT_ROOT}/scripts/mt10/run_phase9_mt10.sh"
