#!/usr/bin/env bash
# Submit five independent Push-v3 SmolVLA chunk-length baseline jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs/pbs/grpo
chunks=(2 5 10 15 20)
for chunk in "${chunks[@]}"; do
  qsub \
    -N "p58c${chunk}" \
    -o "logs/pbs/grpo/smolvla_pushv3_chunk_${chunk}.out" \
    -v "PHASE58_CHUNK_LEN=${chunk}" \
    scripts/grpo/submit_smolvla_baseline_pushv3_chunk_sweep.pbs
done
