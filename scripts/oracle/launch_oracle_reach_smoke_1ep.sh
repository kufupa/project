#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

: "${ORACLE_BASELINE_TASKS:=reach-v3 reach-wall-v3}"
: "${ORACLE_BASELINE_SEED:=1000}"
: "${ORACLE_BASELINE_EPISODE_LENGTH:=120}"
: "${ORACLE_BASELINE_VIDEO:=true}"
: "${ORACLE_BASELINE_FPS:=30}"
: "${ORACLE_BASELINE_SAVE_FRAMES:=true}"
: "${ORACLE_METAWORLD_CAMERA_NAME:=corner2}"
: "${ORACLE_FLIP_CORNER2:=true}"
: "${ORACLE_ARTIFACT_ROOT:=${PROJECT_ROOT}/artifacts/phase06_oracle_baseline}"

echo "[INFO] running local one-episode Oracle smoke for tasks: ${ORACLE_BASELINE_TASKS}"
for task in ${ORACLE_BASELINE_TASKS}; do
  echo "[INFO] running task=${task}"
  bash "${PROJECT_ROOT}/scripts/oracle/run_oracle_baseline_eval.sh" \
    --task "${task}" \
    --episodes 1 \
    --seed "${ORACLE_BASELINE_SEED}" \
    --episode-length "${ORACLE_BASELINE_EPISODE_LENGTH}" \
    --video "${ORACLE_BASELINE_VIDEO}" \
    --fps "${ORACLE_BASELINE_FPS}" \
    --save-frames "${ORACLE_BASELINE_SAVE_FRAMES}" \
    --camera-name "${ORACLE_METAWORLD_CAMERA_NAME}" \
    --flip-corner2 "${ORACLE_FLIP_CORNER2}" \
    --output-root "${ORACLE_ARTIFACT_ROOT}"
  echo "[INFO] completed task=${task}"
done
