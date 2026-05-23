#!/usr/bin/env bash
# Queue all ten Phase072 PBS jobs on CX3 (50 tasks × 10 ep). Run from repo root:
#   cd .../project && bash scripts/mt50/submit_mt50_phase072_10ep_all_10gpu_pbs.sh
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_PROJECT_ROOT}"

for _i in 0 1 2 3 4 5 6 7 8 9; do
  /opt/pbs/bin/qsub "${_PROJECT_ROOT}/scripts/mt50/submit_mt50_phase072_10ep_10gpu_shard${_i}.pbs"
done
