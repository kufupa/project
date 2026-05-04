#!/usr/bin/env bash
# LEGACY per-task SmolVLA driver: loads policy each invocation. For MT50 Phase07 use
# scripts/mt50/run_phase07_smolvla_baseline.sh (campaign, single load) unless debugging.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
SMOLVLA_PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${SMOLVLA_PYTHON_BIN:-${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}/bin/python}"
CHECKPOINT="${SMOLVLA_INIT_CHECKPOINT:-jadechoghari/smolvla_metaworld}"
OUTPUT_ROOT="${SMOLVLA_PARITY_OUTPUT_ROOT:-${SMOLVLA_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}/phase07_smolvla_baseline/parity}"
TASK="${SMOLVLA_PARITY_TASK:-push-v3}"
SEED="${SMOLVLA_PARITY_SEED:-1000}"
EPISODES="${SMOLVLA_PARITY_EPISODES:-15}"
FPS="${SMOLVLA_PARITY_FPS:-30}"
OVERLAY_MODE="${SMOLVLA_PARITY_OVERLAY_MODE:-reward_delta}"
MAX_STEPS="${SMOLVLA_PARITY_MAX_STEPS:-120}"
MIN_VIDEO_BYTES="${SMOLVLA_PARITY_MIN_VIDEO_BYTES:-1024}"

CAMERA_NAME="${SMOLVLA_METAWORLD_CAMERA_NAME:-corner2}"
FLIP_CORNER2="${SMOLVLA_FLIP_CORNER2:-true}"
LOAD_VLM_WEIGHTS="${SMOLVLA_LOAD_VLM_WEIGHTS:-true}"
SAVE_FRAMES="${SMOLVLA_SAVE_FRAMES:-false}"
VIDEO="${SMOLVLA_PARITY_VIDEO:-false}"
SAVE_ACTIONS="${SMOLVLA_SAVE_ACTIONS:-false}"
TASK_TEXT="${SMOLVLA_TASK_TEXT:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

RUN_DIR="$(
  PYTHONPATH="${SMOLVLA_PYTHONPATH}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  TASK="${TASK}" \
  SEED="${SEED}" \
  EPISODES="${EPISODES}" \
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

from src.smolvla_pipeline.run_layout import ensure_unique_run_dir

run_dir = ensure_unique_run_dir(
    Path(os.environ["OUTPUT_ROOT"]),
    episodes=int(os.environ["EPISODES"]),
    task=os.environ["TASK"],
    seed=int(os.environ["SEED"]),
    variant="smolvla_parity",
)
print(str(run_dir))
PY
)"

echo "parity benchmark run dir: ${RUN_DIR}"

XVFB_ERR="${RUN_DIR}/xvfb.err"
# Stable matplotlib config dir: per-run ${RUN_DIR}/.mplconfig forces a cold font cache every job
# (multi-minute "building font cache", low CPU). Prefer ${XDG_CACHE_HOME} (set by Slurm common_env)
# or workspace .cache. Override with SMOLVLA_MPLCONFIGDIR for a custom path.
_cache_root="${XDG_CACHE_HOME:-${WORKSPACE_ROOT}/.cache}"
MPL_DIR="${SMOLVLA_MPLCONFIGDIR:-${_cache_root}/matplotlib_smolvla_parity}"
mkdir -p "${MPL_DIR}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPL_DIR}"
echo "smolvla_parity: MPLCONFIGDIR=${MPL_DIR}" >&2

# Whole-evaluator cap (kills hung xvfb-run before Slurm walltime). Override with SMOLVLA_PARITY_EVAL_TIMEOUT_SEC.
# Default ~10 min/episode + 30 min model/env overhead; clamp so typical 20-ep fits under 4h Slurm jobs.
_ev_timeout="${SMOLVLA_PARITY_EVAL_TIMEOUT_SEC:-}"
if [[ -z "${_ev_timeout}" ]]; then
  _ev_timeout=$((EPISODES * 600 + 1800))
fi
if [[ "${_ev_timeout}" -lt 1200 ]]; then
  _ev_timeout=1200
fi

EVAL_ENV=(
  env
  "PYTHONPATH=${SMOLVLA_PYTHONPATH}"
  "MPLBACKEND=${MPLBACKEND}"
  "MPLCONFIGDIR=${MPLCONFIGDIR}"
  "SMOLVLA_METAWORLD_CAMERA_NAME=${CAMERA_NAME}"
  "SMOLVLA_FLIP_CORNER2=${FLIP_CORNER2}"
  "SMOLVLA_LOAD_VLM_WEIGHTS=${LOAD_VLM_WEIGHTS}"
)

EVAL_ARGS=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/smolvla/run_metaworld_smolvla_eval.py"
  --task "${TASK}"
  --episodes "${EPISODES}"
  --seed "${SEED}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${RUN_DIR}"
  --video "${VIDEO}"
  --fps "${FPS}"
  --overlay-mode "${OVERLAY_MODE}"
  --max-steps "${MAX_STEPS}"
  --save-frames "${SAVE_FRAMES}"
  --save-actions "${SAVE_ACTIONS}"
)
if [[ -n "${TASK_TEXT}" ]]; then
  EVAL_ARGS+=(--task-text "${TASK_TEXT}")
fi

REQUIRE_VIDEO="${SMOLVLA_PARITY_REQUIRE_VIDEO:-}"
if [[ -z "${REQUIRE_VIDEO}" ]]; then
  if [[ "${VIDEO}" == "true" ]]; then
    REQUIRE_VIDEO="true"
  else
    REQUIRE_VIDEO="false"
  fi
fi

echo "smolvla_parity: starting evaluator timeout_s=${_ev_timeout} xvfb_err=${XVFB_ERR} video=${VIDEO} save_frames=${SAVE_FRAMES} save_actions=${SAVE_ACTIONS}" >&2

set +e
if command -v xvfb-run >/dev/null 2>&1; then
  echo "smolvla_parity: using xvfb-run (-e logs Xvfb/xauth)" >&2
  if command -v timeout >/dev/null 2>&1; then
    timeout -k 120 "${_ev_timeout}" \
      xvfb-run -a -e "${XVFB_ERR}" -s "-screen 0 1280x1024x24" \
      "${EVAL_ENV[@]}" "${EVAL_ARGS[@]}"
  else
    xvfb-run -a -e "${XVFB_ERR}" -s "-screen 0 1280x1024x24" \
      "${EVAL_ENV[@]}" "${EVAL_ARGS[@]}"
  fi
else
  echo "smolvla_parity: xvfb-run not found; running without virtual display" >&2
  if command -v timeout >/dev/null 2>&1; then
    timeout -k 120 "${_ev_timeout}" "${EVAL_ENV[@]}" "${EVAL_ARGS[@]}"
  else
    "${EVAL_ENV[@]}" "${EVAL_ARGS[@]}"
  fi
fi
_eval_status=$?
set -e

if [[ "${_eval_status}" -ne 0 ]]; then
  echo "smolvla_parity: evaluator failed exit_code=${_eval_status}" >&2
  if [[ -f "${XVFB_ERR}" ]]; then
    echo "smolvla_parity: tail ${XVFB_ERR}" >&2
    tail -n 120 "${XVFB_ERR}" >&2 || true
  fi
  exit "${_eval_status}"
fi

echo "smolvla_parity: evaluator finished OK" >&2

PYTHONPATH="${SMOLVLA_PYTHONPATH}" \
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/smolvla/verify_smolvla_run_artifacts.py" \
  --run-dir "${RUN_DIR}" \
  --task "${TASK}" \
  --episodes "${EPISODES}" \
  --require-video "${REQUIRE_VIDEO}" \
  --require-frames "${SAVE_FRAMES}" \
  --require-actions "${SAVE_ACTIONS}" \
  --min-video-bytes "${MIN_VIDEO_BYTES}"

echo "parity benchmark complete: ${RUN_DIR}"
