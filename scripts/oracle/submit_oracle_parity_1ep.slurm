#!/bin/bash
#SBATCH --job-name=oracle-parity-1ep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=oracle_parity_1ep_%j.out

set -euo pipefail

# Slurm copies this batch script out of the repo; resolve the project via SLURM_SUBMIT_DIR
# (cwd when `sbatch` was run). Submit from the repo root, or set ORACLE_PROJECT_ROOT.

if [[ -n "${ORACLE_PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${ORACLE_PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SUBMIT_DIR}/../.." && pwd)"
fi

PARITY_PY="${PROJECT_ROOT}/scripts/oracle/run_oracle_parity_1ep.py"
if [[ ! -f "${PARITY_PY}" ]]; then
  echo "error: parity script not found at ${PARITY_PY}" >&2
  echo "hint: cd <repo> && sbatch scripts/oracle/submit_oracle_parity_1ep.slurm" >&2
  echo "      or export ORACLE_PROJECT_ROOT=/path/to/project" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
exec python3 "${PARITY_PY}" "$@"
