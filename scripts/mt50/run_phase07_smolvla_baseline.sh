#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
PYTHON_BIN="${MT50_PYTHON_BIN:-${SMOLVLA_PYTHON_BIN:-${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}/bin/python}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[mt50:phase07] ERROR: python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

RUN_ROOT="${MT50_PHASE07_OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/MT50_Phase07}"
DIFFICULTY_JSON="${MT50_TASK_DIFFICULTY_JSON:-${SCRIPT_DIR}/mt50_phase07_task_difficulties.json}"
INCLUDE_DIFFICULTIES="${MT50_INCLUDE_DIFFICULTIES:-easy,medium,hard,very_hard,unclassified}"
EPISODES="${MT50_PHASE07_EPISODES:-10}"
SEED="${MT50_PHASE07_SEED:-1000}"
MAX_STEPS="${MT50_PHASE07_MAX_STEPS:-120}"
CHECKPOINT="${MT50_PHASE07_CHECKPOINT:-${SMOLVLA_INIT_CHECKPOINT:-jadechoghari/smolvla_metaworld}}"
FPS="${MT50_PHASE07_FPS:-30}"
OVERLAY_MODE="${MT50_PHASE07_OVERLAY_MODE:-reward_delta}"
SAVE_FRAMES="${MT50_PHASE07_SAVE_FRAMES:-false}"
VIDEO="${MT50_PHASE07_VIDEO:-false}"
SAVE_ACTIONS="${MT50_PHASE07_SAVE_ACTIONS:-false}"
CAMERA_NAME="${MT50_PHASE07_CAMERA_NAME:-${SMOLVLA_METAWORLD_CAMERA_NAME:-corner2}}"
FLIP_CORNER2="${MT50_FLIP_CORNER2:-${SMOLVLA_FLIP_CORNER2:-true}}"
MIN_VIDEO_BYTES="${MT50_MIN_VIDEO_BYTES:-1024}"
INDEX_JSON="${MT50_PHASE07_INDEX_JSON:-${RUN_ROOT}/MT50_Phase07_index.json}"
TASK_TEXT="${MT50_PHASE07_TASK_TEXT:-}"

PARITY_SCRIPT="${PROJECT_ROOT}/scripts/smolvla/legacy_run_pushv3_smolvla_parity_benchmark.sh"

if [[ ! -f "${DIFFICULTY_JSON}" ]]; then
  echo "[mt50:phase07] ERROR: missing difficulty map json: ${DIFFICULTY_JSON}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}"

mapfile -t INCLUDE_BUCKETS < <(
  echo "${INCLUDE_DIFFICULTIES}" |
    tr ',' '\n' |
    awk '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0); if (length($0) > 0) print tolower($0)}'
)
if [[ "${#INCLUDE_BUCKETS[@]}" -eq 0 ]]; then
  INCLUDE_BUCKETS=(easy medium hard very_hard unclassified)
fi

is_included_bucket() {
  local candidate="${1,,}"
  for bucket in "${INCLUDE_BUCKETS[@]}"; do
    if [[ "${candidate}" == "${bucket}" ]]; then
      return 0
    fi
  done
  return 1
}

resolve_default_difficulty() {
  "$PYTHON_BIN" - "${DIFFICULTY_JSON}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
print((data.get("default") or "").strip() or "unclassified")
PY
}

resolve_task_difficulty() {
  local task="$1"
  local fallback="$2"
  local difficulty
  difficulty="$(
    "$PYTHON_BIN" - "${DIFFICULTY_JSON}" "${task}" "${fallback}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)

task = sys.argv[2]
fallback = (sys.argv[3] or "unclassified").strip().lower()
task_difficulty = (data.get("task_difficulties") or {}).get(task, fallback)
print((str(task_difficulty or fallback)).strip().lower())
PY
  )"
  echo "${difficulty}"
}

resolve_task_slug() {
  local task="$1"
  "$PYTHON_BIN" - "${PROJECT_ROOT}" "${task}" <<'PY'
import os
import sys

sys.path.insert(0, os.path.join(sys.argv[1], ""))
from src.smolvla_pipeline.run_layout import slug_task

print(slug_task(sys.argv[2]))
PY
}

DEFAULT_DIFFICULTY="$(resolve_default_difficulty)"
TASKS_FILE="$(mktemp)"
RESULTS_FILE="$(mktemp)"
trap 'rm -f "${TASKS_FILE}" "${RESULTS_FILE}"' EXIT

"$PYTHON_BIN" - <<'PY' > "${TASKS_FILE}"
import metaworld

for task in sorted(metaworld.MT50().train_classes.keys()):
    print(task)
PY

mapfile -t TASKS < "${TASKS_FILE}"
if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "[mt50:phase07] ERROR: no MT50 tasks discovered from metaworld" >&2
  exit 2
fi

echo "[mt50:phase07] discovered ${#TASKS[@]} tasks from metaworld"
echo "[mt50:phase07] output_root=${RUN_ROOT}"
echo "[mt50:phase07] include_difficulties=${INCLUDE_DIFFICULTIES}"

echo "[mt50:phase07] ============================================================"
echo "[mt50:phase07] MT50 Phase07 SmolVLA baseline"
echo "[mt50:phase07] DEFAULT=campaign (single process, load SmolVLA once)"
echo "[mt50:phase07] LEGACY per-task loop ONLY if MT50_PHASE07_LEGACY_PER_TASK=true"
echo "[mt50:phase07] (legacy uses ${PARITY_SCRIPT} and reloads weights per task)"
echo "[mt50:phase07] ============================================================"

# Single-process campaign: load SmolVLA once, swap Meta-World task per iteration.
LEGACY_PER_TASK="${MT50_PHASE07_LEGACY_PER_TASK:-false}"
LEGACY_PER_TASK="${LEGACY_PER_TASK,,}"
if [[ "${LEGACY_PER_TASK}" != "true" ]]; then
  export PROJECT_ROOT="${PROJECT_ROOT}"
  export SMOLVLA_PYTHON_BIN="${PYTHON_BIN}"
  WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
  export WORKSPACE_ROOT="${WORKSPACE_ROOT}"
  _cache_root="${XDG_CACHE_HOME:-${WORKSPACE_ROOT}/.cache}"
  export MPLCONFIGDIR="${SMOLVLA_MPLCONFIGDIR:-${_cache_root}/matplotlib_smolvla_parity}"
  mkdir -p "${MPLCONFIGDIR}"
  export MPLBACKEND="${MPLBACKEND:-Agg}"
  SMOLVLA_PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  XVFB_ERR="${RUN_ROOT}/mt50_phase07_campaign_xvfb.err"
  set +e
  if command -v xvfb-run >/dev/null 2>&1; then
    if [[ -n "${SMOLVLA_MT50_CAMPAIGN_TIMEOUT_SEC:-}" ]] && command -v timeout >/dev/null 2>&1; then
      timeout -k 120 "${SMOLVLA_MT50_CAMPAIGN_TIMEOUT_SEC}" \
        xvfb-run -a -e "${XVFB_ERR}" -s "-screen 0 1280x1024x24" \
        env "PYTHONPATH=${SMOLVLA_PYTHONPATH}" "MPLBACKEND=${MPLBACKEND}" "MPLCONFIGDIR=${MPLCONFIGDIR}" \
        "${PYTHON_BIN}" -m src.smolvla_pipeline.mt50_phase07_campaign
    else
      xvfb-run -a -e "${XVFB_ERR}" -s "-screen 0 1280x1024x24" \
        env "PYTHONPATH=${SMOLVLA_PYTHONPATH}" "MPLBACKEND=${MPLBACKEND}" "MPLCONFIGDIR=${MPLCONFIGDIR}" \
        "${PYTHON_BIN}" -m src.smolvla_pipeline.mt50_phase07_campaign
    fi
  else
    if [[ -n "${SMOLVLA_MT50_CAMPAIGN_TIMEOUT_SEC:-}" ]] && command -v timeout >/dev/null 2>&1; then
      timeout -k 120 "${SMOLVLA_MT50_CAMPAIGN_TIMEOUT_SEC}" \
        env "PYTHONPATH=${SMOLVLA_PYTHONPATH}" "MPLBACKEND=${MPLBACKEND}" "MPLCONFIGDIR=${MPLCONFIGDIR}" \
        "${PYTHON_BIN}" -m src.smolvla_pipeline.mt50_phase07_campaign
    else
      env "PYTHONPATH=${SMOLVLA_PYTHONPATH}" "MPLBACKEND=${MPLBACKEND}" "MPLCONFIGDIR=${MPLCONFIGDIR}" \
        "${PYTHON_BIN}" -m src.smolvla_pipeline.mt50_phase07_campaign
    fi
  fi
  _campaign_rc=$?
  set -e
  echo "[mt50:phase07] complete. index=${INDEX_JSON}"
  exit "${_campaign_rc}"
fi

echo "[mt50:phase07] WARN: MT50_PHASE07_LEGACY_PER_TASK=true -> per-task parity driver (slow for MT50)" >&2

for task in "${TASKS[@]}"; do
  if [[ -z "${task}" ]]; then
    continue
  fi

  difficulty="$(resolve_task_difficulty "${task}" "${DEFAULT_DIFFICULTY}")"
  difficulty="${difficulty,,}"
  if ! is_included_bucket "${difficulty}"; then
    echo "[mt50:phase07] skip task=${task} difficulty=${difficulty}"
    printf '%s\t%s\t%s\t%s\t%s\n' "${task}" "${difficulty}" "" "skipped" "" >> "${RESULTS_FILE}"
    continue
  fi

  task_slug="$(resolve_task_slug "${task}")"
  task_output_root="${RUN_ROOT}/${difficulty}"
  mkdir -p "${task_output_root}"
  task_log="$(mktemp)"

  echo "[mt50:phase07] run task=${task} difficulty=${difficulty} task_slug=${task_slug}"

  set +e
  MT50_PHASE_TASK_STATUS="ok"
  MT50_PHASE_TASK_RUN_DIR=""
  MT50_PHASE_TASK_PC_SUCCESS=""

  SMOLVLA_PARITY_TASK="${task}" \
  SMOLVLA_PARITY_EPISODES="${EPISODES}" \
  SMOLVLA_PARITY_SEED="${SEED}" \
  SMOLVLA_PARITY_MAX_STEPS="${MAX_STEPS}" \
  SMOLVLA_PARITY_FPS="${FPS}" \
  SMOLVLA_PARITY_OVERLAY_MODE="${OVERLAY_MODE}" \
  SMOLVLA_SAVE_FRAMES="${SAVE_FRAMES}" \
  SMOLVLA_PARITY_VIDEO="${VIDEO}" \
  SMOLVLA_SAVE_ACTIONS="${SAVE_ACTIONS}" \
  SMOLVLA_METAWORLD_CAMERA_NAME="${CAMERA_NAME}" \
  SMOLVLA_FLIP_CORNER2="${FLIP_CORNER2}" \
  SMOLVLA_INIT_CHECKPOINT="${CHECKPOINT}" \
  SMOLVLA_PARITY_OUTPUT_ROOT="${task_output_root}" \
  SMOLVLA_PARITY_MIN_VIDEO_BYTES="${MIN_VIDEO_BYTES}" \
  RUN_NAME_PREFIX="${task_slug}_${difficulty}" \
  ${TASK_TEXT:+SMOLVLA_TASK_TEXT="${TASK_TEXT}"} \
  bash "${PARITY_SCRIPT}" 2>&1 | tee "${task_log}"
  task_rc="${PIPESTATUS[0]}"
  set -e

  MT50_PHASE_TASK_RUN_DIR="$(
    awk -F': ' '/^parity benchmark complete:/{print $NF; exit} /^parity benchmark run dir:/{print $NF; exit}' "${task_log}" | tail -n 1
  )"

  if [[ "${task_rc}" -ne 0 ]]; then
    echo "[mt50:phase07] task failed rc=${task_rc} task=${task}" >&2
    MT50_PHASE_TASK_STATUS="failed"
  elif [[ -z "${MT50_PHASE_TASK_RUN_DIR}" ]]; then
    echo "[mt50:phase07] ERROR: no run dir parsed for task=${task}" >&2
    MT50_PHASE_TASK_STATUS="failed"
  else
    if [[ -f "${MT50_PHASE_TASK_RUN_DIR}/eval_info.json" ]]; then
      MT50_PHASE_TASK_PC_SUCCESS="$(
        "$PYTHON_BIN" - "${MT50_PHASE_TASK_RUN_DIR}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(f"{path}/eval_info.json", encoding="utf-8") as f:
    eval_info = json.load(f)

overall = eval_info.get("overall", {})
pc = overall.get("pc_success", "")
if pc == "":
    print("")
else:
    print(float(pc))
PY
      )"
    else
      echo "[mt50:phase07] ERROR: missing eval_info.json in ${MT50_PHASE_TASK_RUN_DIR}" >&2
      MT50_PHASE_TASK_STATUS="failed"
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' "${task}" "${difficulty}" "${MT50_PHASE_TASK_RUN_DIR}" "${MT50_PHASE_TASK_STATUS}" "${MT50_PHASE_TASK_PC_SUCCESS}" >> "${RESULTS_FILE}"
  rm -f "${task_log}"
done

"$PYTHON_BIN" - "${RESULTS_FILE}" "${INDEX_JSON}" "${RUN_ROOT}" "${EPISODES}" "${SEED}" "${MAX_STEPS}" "${CHECKPOINT}" "${DEFAULT_DIFFICULTY}" "${INCLUDE_DIFFICULTIES}" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

results_file, index_path, run_root, episodes, seed, max_steps, checkpoint, default_difficulty, include_difficulties = sys.argv[1:10]

rows = []
status_counts = {"ok": 0, "failed": 0, "skipped": 0}
for raw in Path(results_file).read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    parts = raw.split("\t", 4)
    if len(parts) != 5:
        continue
    task, difficulty, run_dir, status, pc_success_raw = parts
    pc_success = None
    if pc_success_raw:
        try:
            pc_success = float(pc_success_raw)
        except ValueError:
            pc_success = None
    if status not in status_counts:
        status_counts[status] = 0
    status_counts[status] += 1
    rows.append(
        {
            "task": task,
            "difficulty": difficulty,
            "run_dir": run_dir,
            "status": status,
            "pc_success": pc_success,
        }
    )

payload = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "phase": "MT50_Phase07",
    "run_root": str(Path(run_root)),
    "episodes": int(episodes),
    "seed": int(seed),
    "max_steps": int(max_steps),
    "checkpoint": checkpoint,
    "default_difficulty": default_difficulty,
    "difficulty_filter": [bucket.strip() for bucket in include_difficulties.split(",") if bucket.strip()],
    "tasks": rows,
    "task_count": len(rows),
    "task_status": status_counts,
}

Path(index_path).parent.mkdir(parents=True, exist_ok=True)
Path(index_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(f"[mt50:phase07] wrote index: {index_path}")
print(f"[mt50:phase07] tasks={len(rows)} ok={status_counts.get('ok',0)} failed={status_counts.get('failed',0)} skipped={status_counts.get('skipped',0)}")
PY

echo "[mt50:phase07] complete. index=${INDEX_JSON}"
