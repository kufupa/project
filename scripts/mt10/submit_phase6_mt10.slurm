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

if [[ -n "${MT10_PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${MT10_PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SUBMIT_DIR}/../.." && pwd)"
fi

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"

echo "[mt10:p6-slurm] host=$(hostname) job=${SLURM_JOB_ID:-local} partition=${SLURM_JOB_PARTITION:-unknown}"

exec bash "${PROJECT_ROOT}/scripts/mt10/run_phase6_mt10.sh"
