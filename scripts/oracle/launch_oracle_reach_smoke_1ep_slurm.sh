#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "[ERROR] sbatch not found. Run this script only on the Slurm login node." >&2
  exit 2
fi

: "${ORACLE_BASELINE_TASKS:=reach-v3 reach-wall-v3}"
: "${ORACLE_BASELINE_SEED:=1000}"
: "${ORACLE_BASELINE_EPISODE_LENGTH:=120}"
: "${ORACLE_BASELINE_VIDEO:=true}"
: "${ORACLE_BASELINE_FPS:=30}"
: "${ORACLE_BASELINE_SAVE_FRAMES:=true}"
: "${ORACLE_METAWORLD_CAMERA_NAME:=corner2}"
: "${ORACLE_FLIP_CORNER2:=true}"
: "${ORACLE_ARTIFACT_ROOT:=${PROJECT_ROOT}/artifacts/phase06_oracle_baseline}"

echo "[INFO] queueing 1-episode Oracle reach/reach-wall jobs to Slurm (seed=${ORACLE_BASELINE_SEED}, max_steps=${ORACLE_BASELINE_EPISODE_LENGTH})"

JOBS=()
for task in ${ORACLE_BASELINE_TASKS}; do
  job_id="$(sbatch --parsable \
    --job-name="oracle-1ep-${task//-/_}" \
    --output="${PROJECT_ROOT}/oracle_reach_1ep_%x_%j.out" \
    "${PROJECT_ROOT}/scripts/oracle/submit_oracle_parity_1ep.slurm" \
    --task "${task}" \
    --episodes 1 \
    --seed "${ORACLE_BASELINE_SEED}" \
    --max-steps "${ORACLE_BASELINE_EPISODE_LENGTH}" \
    --video "${ORACLE_BASELINE_VIDEO}" \
    --fps "${ORACLE_BASELINE_FPS}" \
    --camera-name "${ORACLE_METAWORLD_CAMERA_NAME}" \
    --flip-corner2 "${ORACLE_FLIP_CORNER2}" \
    --save-frames "${ORACLE_BASELINE_SAVE_FRAMES}" \
    --output-root "${ORACLE_ARTIFACT_ROOT}" \
    --output-dir "" )"
  JOBS+=("${job_id}")
  echo "[INFO] queued task=${task} job_id=${job_id}"
done

echo "[INFO] queued jobs: ${JOBS[*]}"
echo "Monitor: squeue -j $(IFS=,; echo "${JOBS[*]}")"
