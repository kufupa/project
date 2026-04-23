#!/usr/bin/env bash
# Phase 9: oracle-vs-WM replay per MT10 task (60 ep, uses each task's phase6 oracle run).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
# shellcheck source=mt10_tasks.sh
source "${SCRIPT_DIR}/mt10_tasks.sh"
# shellcheck source=mt10_env_defaults.sh
source "${SCRIPT_DIR}/mt10_env_defaults.sh"
if [[ -n "${MT10_ONLY_TASK:-}" ]]; then
  MT10_TASK_IDS=("${MT10_ONLY_TASK}")
fi

INDEX_JSON="${MT10_PHASE6_INDEX_JSON:-${PROJECT_ROOT}/artifacts/mt10_runs/mt10_phase6_index.json}"
INDEX9_JSON="${MT10_PHASE9_INDEX_JSON:-${PROJECT_ROOT}/artifacts/mt10_runs/mt10_phase9_index.json}"
MAP_TMP="${MT10_PHASE9_MAP_TMP:-$(mktemp)}"
PY="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}/bin/python"
PHASE9_PY="${PROJECT_ROOT}/scripts/run_phase9_oracle_vs_wm.py"

if [[ ! -f "${INDEX_JSON}" ]]; then
  echo "[mt10:phase9] ERROR: missing phase6 index: ${INDEX_JSON}" >&2
  exit 3
fi
if [[ ! -x "${PY}" ]]; then
  echo "[mt10:phase9] ERROR: python not executable: ${PY}" >&2
  exit 3
fi

rm -f "${MAP_TMP}"
touch "${MAP_TMP}"

if [[ "${MT10_DISABLE_RUN_PREFIX:-0}" != "1" ]]; then
  export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-${ORACLE_RUN_PREFIX:-${MT10_RUN_NAME_PREFIX:-mt10}}}"
  export ORACLE_RUN_PREFIX="${ORACLE_RUN_PREFIX:-${RUN_NAME_PREFIX}}"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

DRY_RUN=false
for a in "$@"; do
  if [[ "$a" == "--dry-run" ]]; then
    DRY_RUN=true
  fi
done

for task in "${MT10_TASK_IDS[@]}"; do
  oracle_root="$("${PY}" -c "import json,sys; m=json.load(open(sys.argv[1],encoding='utf-8')); print(m.get('tasks',{}).get(sys.argv[2],''))" "${INDEX_JSON}" "${task}")"
  if [[ -z "${oracle_root}" ]]; then
    echo "[mt10:phase9] ERROR: no phase6 oracle path for task=${task} in ${INDEX_JSON}" >&2
    exit 4
  fi
  echo "[mt10:phase9] --- task=${task} oracle_run=${oracle_root} ---"
  if [[ "${DRY_RUN}" == "true" ]]; then
    logf="$(mktemp)"
    set +o pipefail
    cmd=(
      "${PY}" "${PHASE9_PY}"
      --oracle-run-root "${oracle_root}"
      --artifacts-root "${PROJECT_ROOT}/artifacts"
      --output-root "${MT10_PHASE9_OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/phase09_oracle_vs_wm_baseline}"
      --task "${task}"
      --episodes "${MT10_PHASE9_EPISODES:-60}"
      --goal-frame-index "${MT10_PHASE9_GOAL_FRAME:-50}"
      --max-steps "${MT10_PHASE9_MAX_STEPS:-50}"
      --chunk-len "${MT10_PHASE9_CHUNK_LEN:-50}"
      --wm-rollout-mode iterative
      --wm-scoring-latent visual
      --device "${MT10_PHASE9_DEVICE:-cuda}"
      --dry-run
    )
    "${cmd[@]}" 2>&1 | tee "${logf}"
    py_rc="${PIPESTATUS[0]:-1}"
    set -o pipefail
    if [[ "${py_rc}" -ne 0 ]]; then
      echo "[mt10:phase9] ERROR: dry-run driver failed rc=${py_rc} task=${task}" >&2
      tail -n 80 "${logf}" >&2 || true
      rm -f "${logf}"
      exit "${py_rc}"
    fi
    run_dir="$(grep '\[phase9\] run directory:' "${logf}" | tail -n 1 | awk '{print $NF}' || true)"
    rm -f "${logf}"
    if [[ -z "${run_dir}" ]]; then
      echo "[mt10:phase9] ERROR: could not parse run directory from log for task=${task}" >&2
      exit 3
    fi
    printf '%s\t%s\n' "${task}" "(dry-run:${run_dir})" >>"${MAP_TMP}"
    continue
  fi
  if [[ ! -d "${oracle_root}" ]]; then
    echo "[mt10:phase9] ERROR: oracle root not a directory: ${oracle_root}" >&2
    exit 4
  fi
  logf="$(mktemp)"
  set +o pipefail
  cmd=(
    "${PY}" "${PHASE9_PY}"
    --oracle-run-root "${oracle_root}"
    --artifacts-root "${PROJECT_ROOT}/artifacts"
    --output-root "${MT10_PHASE9_OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/phase09_oracle_vs_wm_baseline}"
    --task "${task}"
    --episodes "${MT10_PHASE9_EPISODES:-60}"
    --goal-frame-index "${MT10_PHASE9_GOAL_FRAME:-50}"
    --max-steps "${MT10_PHASE9_MAX_STEPS:-50}"
    --chunk-len "${MT10_PHASE9_CHUNK_LEN:-50}"
    --wm-rollout-mode iterative
    --wm-scoring-latent visual
    --device "${MT10_PHASE9_DEVICE:-cuda}"
  )
  if [[ -z "${JEPA_REPO:-}" ]]; then
    echo "[mt10:phase9] ERROR: JEPA_REPO must be set for non-dry phase9 runs." >&2
    rm -f "${logf}"
    exit 3
  else
    cmd+=(--jepa-repo "${JEPA_REPO}")
  fi
  "${cmd[@]}" 2>&1 | tee "${logf}"
  py_rc="${PIPESTATUS[0]:-1}"
  set -o pipefail
  if [[ "${py_rc}" -ne 0 ]]; then
    echo "[mt10:phase9] ERROR: phase9 driver failed rc=${py_rc} task=${task}" >&2
    tail -n 80 "${logf}" >&2 || true
    rm -f "${logf}"
    exit "${py_rc}"
  fi
  run_dir="$(grep '\[phase9\] run directory:' "${logf}" | tail -n 1 | awk '{print $NF}' || true)"
  rm -f "${logf}"
  if [[ -z "${run_dir}" ]]; then
    echo "[mt10:phase9] ERROR: could not parse run directory from log for task=${task}" >&2
    exit 3
  fi
  printf '%s\t%s\n' "${task}" "${run_dir}" >>"${MAP_TMP}"
done

"${PY}" <<PY
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

p6 = json.loads(Path("${INDEX_JSON}").read_text(encoding="utf-8"))
rows = []
for line in Path("${MAP_TMP}").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    t, r = line.split("\t", 1)
    rows.append((t.strip(), r.strip()))

out = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "phase": 9,
    "phase6_index": str(Path("${INDEX_JSON}").resolve()),
    "run_name_prefix": __import__("os").environ.get("RUN_NAME_PREFIX")
    or __import__("os").environ.get("ORACLE_RUN_PREFIX")
    or "",
    "phase6_oracle_roots": p6.get("tasks", {}),
    "phase9_run_dirs": {t: r for t, r in rows},
}
Path("${INDEX9_JSON}").parent.mkdir(parents=True, exist_ok=True)
Path("${INDEX9_JSON}").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
print(f"[mt10:phase9] wrote index: ${INDEX9_JSON}")
PY
rm -f "${MAP_TMP}"
