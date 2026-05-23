#!/usr/bin/env bash
# Submit all ten Phase072 jobs (50 tasks × 10 ep, official LeRobot), one GPU per job.
# See README_MT50_Phase072_10ep_10gpu_shards.txt for task lists and post-merge steps.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"

for _i in 0 1 2 3 4 5 6 7 8 9; do
  /usr/bin/sbatch --chdir="${_PROJECT_ROOT}" "${_PROJECT_ROOT}/scripts/mt50/submit_mt50_phase072_10ep_10gpu_shard${_i}.slurm"
done
