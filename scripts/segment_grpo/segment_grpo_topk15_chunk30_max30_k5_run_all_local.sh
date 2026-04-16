#!/usr/bin/env bash
# Run all 14 top-15 targets locally (no sbatch). Use inside salloc/srun or any machine with CUDA + env set up.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for TASK_ID in $(seq 0 13); do
  echo "[seggrpo-topk15-local] === starting TASK_ID=${TASK_ID} ==="
  bash "${SCRIPT_DIR}/segment_grpo_topk15_chunk30_max30_k5_run_task.sh" "${TASK_ID}"
done
echo "[seggrpo-topk15-local] all tasks finished OK"
