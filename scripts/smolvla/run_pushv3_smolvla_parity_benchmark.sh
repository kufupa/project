#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

PYTHON_BIN="${SMOLVLA_PYTHON_BIN:-${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}/bin/python}"
CHECKPOINT="${SMOLVLA_INIT_CHECKPOINT:-jadechoghari/smolvla_metaworld}"
OUTPUT_ROOT="${SMOLVLA_PARITY_OUTPUT_ROOT:-${SMOLVLA_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}/phase07_smolvla_baseline/parity}"
TASK="${SMOLVLA_PARITY_TASK:-push-v3}"
SEED="${SMOLVLA_PARITY_SEED:-1000}"
EPISODES="${SMOLVLA_PARITY_EPISODES:-15}"
FPS="${SMOLVLA_PARITY_FPS:-30}"
OVERLAY_MODE="${SMOLVLA_PARITY_OVERLAY_MODE:-cumulative_reward}"
MAX_STEPS="${SMOLVLA_PARITY_MAX_STEPS:-120}"
MIN_VIDEO_BYTES="${SMOLVLA_PARITY_MIN_VIDEO_BYTES:-1024}"

CAMERA_NAME="${SMOLVLA_METAWORLD_CAMERA_NAME:-corner2}"
FLIP_CORNER2="${SMOLVLA_FLIP_CORNER2:-true}"
LOAD_VLM_WEIGHTS="${SMOLVLA_LOAD_VLM_WEIGHTS:-true}"
SAVE_FRAMES="${SMOLVLA_SAVE_FRAMES:-true}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

RUN_DIR="$(
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
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
if command -v xvfb-run >/dev/null 2>&1; then
  xvfb-run -a -s "-screen 0 1280x1024x24" \
    env PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    SMOLVLA_METAWORLD_CAMERA_NAME="${CAMERA_NAME}" \
    SMOLVLA_FLIP_CORNER2="${FLIP_CORNER2}" \
    SMOLVLA_LOAD_VLM_WEIGHTS="${LOAD_VLM_WEIGHTS}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/smolvla/run_metaworld_smolvla_eval.py" \
      --task "${TASK}" \
      --episodes "${EPISODES}" \
      --seed "${SEED}" \
      --checkpoint "${CHECKPOINT}" \
      --output-dir "${RUN_DIR}" \
      --video true \
      --fps "${FPS}" \
      --overlay-mode "${OVERLAY_MODE}" \
      --max-steps "${MAX_STEPS}" \
      --save-frames "${SAVE_FRAMES}"
else
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
  SMOLVLA_METAWORLD_CAMERA_NAME="${CAMERA_NAME}" \
  SMOLVLA_FLIP_CORNER2="${FLIP_CORNER2}" \
  SMOLVLA_LOAD_VLM_WEIGHTS="${LOAD_VLM_WEIGHTS}" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/smolvla/run_metaworld_smolvla_eval.py" \
    --task "${TASK}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${RUN_DIR}" \
    --video true \
    --fps "${FPS}" \
    --overlay-mode "${OVERLAY_MODE}" \
    --max-steps "${MAX_STEPS}" \
    --save-frames "${SAVE_FRAMES}"
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/smolvla/verify_smolvla_run_artifacts.py" \
  --run-dir "${RUN_DIR}" \
  --task "${TASK}" \
  --episodes "${EPISODES}" \
  --require-video true \
  --require-frames "${SAVE_FRAMES}" \
  --min-video-bytes "${MIN_VIDEO_BYTES}"

echo "parity benchmark complete: ${RUN_DIR}"
