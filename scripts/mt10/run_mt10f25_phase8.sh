#!/usr/bin/env bash
# MT10 goal frame 25 + optional prefetch WM-prefix scoring (phase8 wrapper).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ART="${PROJECT_ROOT}/artifacts"

export MT10_PHASE8_GOAL_FRAME="${MT10_PHASE8_GOAL_FRAME:-25}"
export MT10_PHASE8_CHUNK_LEN="${MT10_PHASE8_CHUNK_LEN:-35}"
export MT10_PHASE8_MAX_STEPS="${MT10_PHASE8_MAX_STEPS:-35}"
export MT10_PHASE8_SMOLVLA_N_ACTION_STEPS="${MT10_PHASE8_SMOLVLA_N_ACTION_STEPS:-35}"
export MT10_PHASE8_OUTPUT_ROOT="${MT10_PHASE8_OUTPUT_ROOT:-${ART}/mt10f25_phase08_segment_grpo}"
export MT10_PHASE8_WM_SELECTION_ENV_STEPS="${MT10_PHASE8_WM_SELECTION_ENV_STEPS:-25}"
# Reuse K=3 candidate chunks from completed MT10 phase8 baseline (f50); children slice to chunk_len rows.
export MT10_PHASE8_PREFETCH_INDEX_JSON="${MT10_PHASE8_PREFETCH_INDEX_JSON:-${ART}/mt10_runs/mt10_phase8_index.json}"
# Prefetch mode does not use SmolVLA; drop checkpoint so run_all60 does not pass --checkpoint to children.
unset SMOLVLA_CHECKPOINT || true

exec bash "${SCRIPT_DIR}/run_phase8_mt10.sh" "$@"
