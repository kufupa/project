#!/usr/bin/env bash
# Phase 6: full push-v3-style oracle pipeline once per MT10 task (60 ep, seed 1000, 120 steps).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
# shellcheck source=mt10_tasks.sh
source "${SCRIPT_DIR}/mt10_tasks.sh"
if [[ -n "${MT10_ONLY_TASK:-}" ]]; then
  MT10_TASK_IDS=("${MT10_ONLY_TASK}")
fi

PIPELINE="${PROJECT_ROOT}/scripts/oracle/pushv3_oracle_data_pipeline.sh"
PYTHON_BIN="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}/bin/python"
INDEX_DIR="${MT10_INDEX_DIR:-${PROJECT_ROOT}/artifacts/mt10_runs}"
INDEX_JSON="${MT10_PHASE6_INDEX_JSON:-${INDEX_DIR}/mt10_phase6_index.json}"
MAP_TMP="${MT10_PHASE6_MAP_TMP:-$(mktemp)}"

DRY_RUN=false
for a in "$@"; do
  if [[ "$a" == "--dry-run" ]]; then
    DRY_RUN=true
  fi
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[mt10:phase6] ERROR: python not executable: ${PYTHON_BIN}" >&2
  exit 3
fi

mkdir -p "${INDEX_DIR}"
rm -f "${MAP_TMP}"
touch "${MAP_TMP}"

if [[ "${MT10_DISABLE_RUN_PREFIX:-0}" != "1" ]]; then
  export ORACLE_RUN_PREFIX="${ORACLE_RUN_PREFIX:-${RUN_NAME_PREFIX:-${MT10_RUN_NAME_PREFIX:-mt10}}}"
  export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-${ORACLE_RUN_PREFIX}}"
fi

echo "[mt10:phase6] ENV_POLICY_MAP gate (${#MT10_TASK_IDS[@]} tasks)..."
if [[ "${MT10_SKIP_POLICY_GATE:-0}" == "1" ]]; then
  echo "[mt10:phase6] WARN: skipping ENV_POLICY_MAP gate (MT10_SKIP_POLICY_GATE=1)"
else
"${PYTHON_BIN}" <<'PY'
from __future__ import annotations

import sys

TASKS = (
    "button-press-topdown-v3",
    "door-open-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "peg-insert-side-v3",
    "pick-place-v3",
    "push-v3",
    "reach-v3",
    "window-close-v3",
    "window-open-v3",
)

try:
    from metaworld.policies import ENV_POLICY_MAP
except Exception as exc:  # noqa: BLE001
    print(f"[mt10:phase6] ERROR: cannot import metaworld.policies: {exc}", file=sys.stderr)
    sys.exit(3)

missing = [t for t in TASKS if t not in ENV_POLICY_MAP]
if missing:
    print("[mt10:phase6] ERROR: scripted policies missing for:", ", ".join(missing), file=sys.stderr)
    sys.exit(4)
print("[mt10:phase6] ENV_POLICY_MAP ok for all MT10 tasks.")
PY
fi

for task in "${MT10_TASK_IDS[@]}"; do
  echo "[mt10:phase6] --- task=${task} ---"
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[mt10:phase6] dry-run: would run pushv3_oracle_data_pipeline.sh --task ${task}"
    printf '%s\t%s\n' "${task}" "(dry-run-placeholder)" >>"${MAP_TMP}"
    continue
  fi
  logf="$(mktemp)"
  set +o pipefail
  PUSHV3_EPISODES="${MT10_PHASE6_EPISODES:-60}" \
    PUSHV3_SEED="${MT10_PHASE6_SEED:-1000}" \
    PUSHV3_TASK="${task}" \
    PUSHV3_EPISODE_LENGTH="${MT10_PHASE6_EPISODE_LENGTH:-120}" \
    ORACLE_SAVE_FRAMES="${ORACLE_SAVE_FRAMES:-true}" \
    PUSHV3_VIDEO="${PUSHV3_VIDEO:-true}" \
    bash "${PIPELINE}" \
    --episodes "${MT10_PHASE6_EPISODES:-60}" \
    --seed "${MT10_PHASE6_SEED:-1000}" \
    --task "${task}" \
    --episode-length "${MT10_PHASE6_EPISODE_LENGTH:-120}" \
    --save-frames true \
    --video true \
    2>&1 | tee "${logf}"
  set -o pipefail
  oracle_dir="$(awk '/Baseline output directory:/{print $NF}' "${logf}" | tail -n 1 || true)"
  rm -f "${logf}"
  if [[ -z "${oracle_dir}" || ! -d "${oracle_dir}" ]]; then
    echo "[mt10:phase6] ERROR: could not resolve oracle output dir for task=${task}" >&2
    exit 3
  fi
  printf '%s\t%s\n' "${task}" "${oracle_dir}" >>"${MAP_TMP}"
done

"${PYTHON_BIN}" <<PY
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

map_path = Path("${MAP_TMP}")
rows = []
for line in map_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    task, root = line.split("\t", 1)
    rows.append((task.strip(), root.strip()))

out = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "phase": 6,
    "run_name_prefix": __import__("os").environ.get("ORACLE_RUN_PREFIX")
    or __import__("os").environ.get("RUN_NAME_PREFIX")
    or "",
    "episodes": int("${MT10_PHASE6_EPISODES:-60}"),
    "seed": int("${MT10_PHASE6_SEED:-1000}"),
    "episode_length": int("${MT10_PHASE6_EPISODE_LENGTH:-120}"),
    "tasks": {t: r for t, r in rows if r != "(dry-run-placeholder)"},
}
# dry-run: still emit tasks map with placeholders for operator inspection
if not out["tasks"] and rows:
    out["tasks"] = {t: r for t, r in rows}

Path("${INDEX_JSON}").parent.mkdir(parents=True, exist_ok=True)
Path("${INDEX_JSON}").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
print(f"[mt10:phase6] wrote index: ${INDEX_JSON}")
PY
rm -f "${MAP_TMP}"
