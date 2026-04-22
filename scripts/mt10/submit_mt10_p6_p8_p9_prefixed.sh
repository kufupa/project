#!/usr/bin/env bash
# Submit MT10 phase 6 then 8+9 with afterok deps. mt10_* artifact roots default
# inside submit_phase{6,8,9}_mt10.slurm (no sbatch --export; avoids user_env issues).
# Run from login node with sbatch available.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SLURM_P6="${ROOT}/scripts/mt10/submit_phase6_mt10.slurm"
SLURM_P8="${ROOT}/scripts/mt10/submit_phase8_mt10.slurm"
SLURM_P9="${ROOT}/scripts/mt10/submit_phase9_mt10.slurm"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "[submit_mt10_p6_p8_p9_prefixed] ERROR: sbatch not in PATH" >&2
  exit 3
fi

mkdir -p "${ROOT}/logs"

# A1: no --export; roots default inside submit_phase*.slurm. --chdir so relative #SBATCH logs/ land under project/.
P6="$(sbatch --parsable --chdir="${ROOT}" "${SLURM_P6}")"
echo "[submit_mt10_p6_p8_p9_prefixed] P6 job_id=${P6}"

J8="$(sbatch --parsable --chdir="${ROOT}" --dependency="afterok:${P6}" "${SLURM_P8}")"
echo "[submit_mt10_p6_p8_p9_prefixed] P8 job_id=${J8} dependency=afterok:${P6}"

J9="$(sbatch --parsable --chdir="${ROOT}" --dependency="afterok:${P6}" "${SLURM_P9}")"
echo "[submit_mt10_p6_p8_p9_prefixed] P9 job_id=${J9} dependency=afterok:${P6}"
