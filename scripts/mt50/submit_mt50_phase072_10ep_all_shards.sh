#!/usr/bin/env bash
# Submit all three Phase072 shard jobs (50 tasks × 10 ep, official LeRobot).
# See README_MT50_Phase072_10ep_shards.txt for task lists and post-merge steps.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"

/usr/bin/sbatch --chdir="${_PROJECT_ROOT}" "${_PROJECT_ROOT}/scripts/mt50/submit_mt50_phase072_10ep_shard0.slurm"
/usr/bin/sbatch --chdir="${_PROJECT_ROOT}" "${_PROJECT_ROOT}/scripts/mt50/submit_mt50_phase072_10ep_shard1.slurm"
/usr/bin/sbatch --chdir="${_PROJECT_ROOT}" "${_PROJECT_ROOT}/scripts/mt50/submit_mt50_phase072_10ep_shard2.slurm"
