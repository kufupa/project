#!/usr/bin/env bash
# Phase 9: oracle actions + WM strips for all 60 episodes (no SmolVLA).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

PYTHON_BIN="${SEGMENT_GRPO_PYTHON:-${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[phase9] ERROR: python not executable: ${PYTHON_BIN}" >&2
  echo "[phase9] Set SMOLVLA_LEROBOT_ENV_DIR or SEGMENT_GRPO_PYTHON." >&2
  exit 1
fi

: "${JEPA_REPO:=${HOME}/.cache/torch/hub/facebookresearch_jepa-wms_main}"
: "${ORACLE_RUN_DIR:=${PROJECT_ROOT}/artifacts/phase06_oracle_baseline/run_20260411T131839Z_ep60_voracle_tpush_v3_s1000_r402093}"

if [[ ! -d "${JEPA_REPO}" ]]; then
  echo "[phase9] ERROR: JEPA_REPO is not a directory: ${JEPA_REPO}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" scripts/run_phase9_oracle_vs_wm.py \
  --oracle-run-root "${ORACLE_RUN_DIR}" \
  --jepa-repo "${JEPA_REPO}" \
  --episodes 60 \
  --goal-frame-index 50 \
  --max-steps 50 \
  --chunk-len 50 \
  --device cuda \
  --wm-rollout-mode iterative \
  --wm-sim-camera-parity \
  --wm-goal-hflip \
  "$@"
