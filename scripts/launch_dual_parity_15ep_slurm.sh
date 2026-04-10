#!/usr/bin/env bash
# Submit 15-episode oracle + SmolVLA parity jobs in parallel (matched seed 1000, 120 steps, corner2+flip, frames on).
# Run from repo root: bash scripts/launch_dual_parity_15ep_slurm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

O_JOB="$(sbatch --parsable "${PROJECT_ROOT}/scripts/oracle/submit_oracle_baseline_15ep.slurm")"
S_JOB="$(sbatch --parsable "${PROJECT_ROOT}/scripts/smolvla/submit_smolvla_parity_15ep.slurm")"

echo "Submitted oracle baseline 15ep: job_id=${O_JOB}"
echo "Submitted SmolVLA parity 15ep: job_id=${S_JOB}"
echo "Monitor: squeue -j ${O_JOB},${S_JOB}"
echo "Logs: ${PROJECT_ROOT}/oracle_baseline_15ep_${O_JOB}.out"
echo "      ${PROJECT_ROOT}/smolvla_parity_15ep_${S_JOB}.out"
