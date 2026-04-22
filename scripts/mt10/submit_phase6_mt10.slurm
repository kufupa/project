#!/usr/bin/env bash
#SBATCH --job-name=mt10-p6-oracle
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --partition=t4
#SBATCH --output=logs/mt10_phase6_%j.out
#SBATCH --error=logs/mt10_phase6_%j.err
#
# One outer GPU allocation; sequential per-task work inside bash (no nested sbatch).
# Override partition: sbatch --partition=OTHER scripts/mt10/submit_phase6_mt10.slurm
#
set -euo pipefail

# Slurm may copy this script under /var/spool/slurm; do not derive repo from BASH_SOURCE alone.
# With sbatch --chdir=<project>, cwd is project; mt10_tasks.sh marks project root.
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
export PUSHV3_OUTPUT_ROOT="${PUSHV3_OUTPUT_ROOT:-${ART}/mt10_phase06_oracle_baseline}"

WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"

echo "[mt10:p6-slurm] host=$(hostname) job=${SLURM_JOB_ID:-local} partition=${SLURM_JOB_PARTITION:-unknown}"

exec bash "${PROJECT_ROOT}/scripts/mt10/run_phase6_mt10.sh"
