#!/bin/bash
#SBATCH --job-name=smolvla-top15
#SBATCH --array=0-14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=smolvla_top15_%A_%a.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${TARGETS_JSON:-}" ]]; then
  echo "error: TARGETS_JSON env var is required." >&2
  exit 2
fi

bash "${PROJECT_ROOT}/scripts/smolvla/run_smolvla_target_episode.sh" "${TARGETS_JSON}" "${SLURM_ARRAY_TASK_ID}"
