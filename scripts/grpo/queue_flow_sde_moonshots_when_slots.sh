#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/vol/bitbucket/aa6622/project}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-43200}"

cd "${PROJECT_ROOT}"
start_ts="$(date +%s)"

echo "[moonshot-queue] start poll_seconds=${POLL_SECONDS} max_wait_seconds=${MAX_WAIT_SECONDS}"
while true; do
  now_ts="$(date +%s)"
  waited="$((now_ts - start_ts))"
  if (( waited > MAX_WAIT_SECONDS )); then
    echo "[moonshot-queue] timeout after ${waited}s without available submit slots"
    exit 1
  fi

  submitted_count="$(squeue -h -u "$USER" | wc -l | tr -d ' ')"
  if (( submitted_count <= 6 )); then
    echo "[moonshot-queue] submit slots available count=${submitted_count}; submitting moonshots"
    dense_out="$(sbatch scripts/grpo/submit_flow_sde_chunk_grpo_moonshot30_dense_chain_a30.slurm)"
    sparse_out="$(sbatch scripts/grpo/submit_flow_sde_chunk_grpo_moonshot30_sparse_chain_a30.slurm)"
    echo "[moonshot-queue] ${dense_out}"
    echo "[moonshot-queue] ${sparse_out}"
    echo "FLOW_SDE_MOONSHOT_QUEUE_OK"
    exit 0
  fi

  echo "[moonshot-queue] waiting submitted_count=${submitted_count} waited=${waited}s"
  sleep "${POLL_SECONDS}"
done
