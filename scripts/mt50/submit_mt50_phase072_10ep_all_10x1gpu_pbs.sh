#!/usr/bin/env bash
# Queue ten Phase072 PBS jobs: 10 ep, 1 GPU each, 4c/16g/12h. Run from repo root:
#   cd .../project && bash scripts/mt50/submit_mt50_phase072_10ep_all_10x1gpu_pbs.sh
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_PROJECT_ROOT}"

for _i in 0 1 2 3 4 5 6 7 8 9; do
  /opt/pbs/bin/qsub \
    -N "mt50p072g10_${_i}" \
    -v MT50_PHASE072_SHARD_INDEX="${_i}" \
    "${_PROJECT_ROOT}/scripts/mt50/submit_mt50_phase072_10ep_10x1gpu_single_shard.pbs"
done
