#!/usr/bin/env bash
# MT10 goal frame 25 phase9 oracle-vs-WM wrapper (max_steps 50 default).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ART="${PROJECT_ROOT}/artifacts"

export MT10_PHASE9_GOAL_FRAME="${MT10_PHASE9_GOAL_FRAME:-25}"
export MT10_PHASE9_MAX_STEPS="${MT10_PHASE9_MAX_STEPS:-50}"
export MT10_PHASE9_CHUNK_LEN="${MT10_PHASE9_CHUNK_LEN:-50}"
export MT10_PHASE9_OUTPUT_ROOT="${MT10_PHASE9_OUTPUT_ROOT:-${ART}/mt10f25_phase09_oracle_vs_wm}"

exec bash "${SCRIPT_DIR}/run_phase9_mt10.sh" "$@"
