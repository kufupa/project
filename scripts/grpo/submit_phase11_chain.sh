#!/usr/bin/env bash
# Login-node orchestration: smoke -> short -> long with Slurm dependencies.
# Does not nest sbatch inside allocations.
#
# Usage:
#   ./scripts/grpo/submit_phase11_chain.sh /path/to/smolvla_ckpt /path/to/artifacts_root
#
# Requires: sbatch, scripts/grpo/submit_phase11_grpo.slurm, common_env marker.

set -euo pipefail

CKPT="${1:?SmolVLA checkpoint}"
ROOT="${2:?artifact root directory}"
SLURM_SH="${SLURM_SH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_phase11_grpo.slurm}"

mkdir -p "${ROOT}"
OUT_SMOKE="${ROOT}/smoke"
OUT_SHORT="${ROOT}/short_to5"
OUT_LONG="${ROOT}/long_to25"

PROJECT_CHDIR="${PROJECT_CHDIR:-/vol/bitbucket/aa6622/project}"

JOB1=$(sbatch --parsable --chdir="${PROJECT_CHDIR}" --export=NIL "${SLURM_SH}" train "${CKPT}" "${OUT_SMOKE}" 2000 0 1)
JOB2=$(sbatch --parsable --chdir="${PROJECT_CHDIR}" --export=NIL --dependency=afterok:"${JOB1}" "${SLURM_SH}" train \
  "${CKPT}" "${OUT_SHORT}" 2000 1 4 "${OUT_SMOKE}/checkpoints/latest.pt")
JOB3=$(sbatch --parsable --chdir="${PROJECT_CHDIR}" --export=NIL --dependency=afterok:"${JOB2}" "${SLURM_SH}" train \
  "${CKPT}" "${OUT_LONG}" 2000 5 20 "${OUT_SHORT}/checkpoints/latest.pt")

echo "submitted: smoke=${JOB1} short=${JOB2} long=${JOB3}"
