#!/usr/bin/env bash
# Submit Phase57 MT50 raw-vs-bounded decode as 5 separate 1-GPU PBS jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

for shard in 0 1 2 3 4; do
  qsub -v "PHASE57_SHARD_INDEX=${shard},PHASE57_SHARD_COUNT=${PHASE57_SHARD_COUNT:-5}" \
    scripts/grpo/submit_phase57_mt50_decode_shard.pbs
done
