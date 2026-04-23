#!/usr/bin/env bash
# Phase 8: Segment-GRPO all60 campaign per MT10 task (uses phase6 oracle roots index).
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
INDEX8_JSON="${MT10_PHASE8_INDEX_JSON:-${PROJECT_ROOT}/artifacts/mt10_runs/mt10_phase8_index.json}"
MAP_TMP="${MT10_PHASE8_MAP_TMP:-$(mktemp)}"
PY="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}/bin/python"
CAMPAIGN="${PROJECT_ROOT}/scripts/segment_grpo/run_all60_frame50_k3.py"

if [[ ! -f "${INDEX_JSON}" ]]; then
  echo "[mt10:phase8] ERROR: missing phase6 index: ${INDEX_JSON}" >&2
  exit 3
fi
if [[ ! -x "${PY}" ]]; then
  echo "[mt10:phase8] ERROR: python not executable: ${PY}" >&2
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
    echo "[mt10:phase8] ERROR: no phase6 oracle path for task=${task} in ${INDEX_JSON}" >&2
    exit 4
  fi
  echo "[mt10:phase8] --- task=${task} oracle_run=${oracle_root} ---"
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[mt10:phase8] dry-run: would run run_all60_frame50_k3.py --task ${task} --oracle-run-root ${oracle_root}"
    printf '%s\t%s\n' "${task}" "(dry-run-placeholder)" >>"${MAP_TMP}"
    continue
  fi
  if [[ "${oracle_root}" == "(dry-run-placeholder)" ]]; then
    echo "[mt10:phase8] ERROR: phase6 index still has dry-run placeholders; run phase6 for real first." >&2
    exit 4
  fi
  if [[ ! -d "${oracle_root}" ]]; then
    echo "[mt10:phase8] ERROR: oracle root not a directory: ${oracle_root}" >&2
    exit 4
  fi
  logf="$(mktemp)"
  set +o pipefail
  cmd=(
    "${PY}" "${CAMPAIGN}"
    --seed-base "${MT10_PHASE8_SEED_BASE:-1000}"
    --episode-start 0
    --episodes "${MT10_PHASE8_EPISODES:-60}"
    --goal-frame-index "${MT10_PHASE8_GOAL_FRAME:-50}"
    --num-candidates "${MT10_PHASE8_NUM_CANDIDATES:-3}"
    --chunk-len "${MT10_PHASE8_CHUNK_LEN:-50}"
    --max-steps "${MT10_PHASE8_MAX_STEPS:-50}"
    --smolvla-n-action-steps "${MT10_PHASE8_SMOLVLA_N_ACTION_STEPS:-50}"
    --task "${task}"
    --carry-mode sim
    --wm-rollout-mode iterative
    --wm-scoring-latent visual
    --comparison-strip-overlay
    --artifacts-root "${PROJECT_ROOT}/artifacts"
    --oracle-run-root "${oracle_root}"
    --output-root "${MT10_PHASE8_OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/phase08_segment_grpo_baseline}"
  )
  if [[ -n "${SMOLVLA_CHECKPOINT:-}" ]]; then
    cmd+=(--checkpoint "${SMOLVLA_CHECKPOINT}")
  fi
  if [[ -n "${JEPA_REPO:-}" ]]; then
    cmd+=(--jepa-repo "${JEPA_REPO}")
  fi
  PREFETCH_INDEX="${MT10_PHASE8_PREFETCH_INDEX_JSON:-}"
  if [[ -n "${PREFETCH_INDEX}" ]]; then
    if [[ ! -f "${PREFETCH_INDEX}" ]]; then
      echo "[mt10:phase8] ERROR: MT10_PHASE8_PREFETCH_INDEX_JSON not a file: ${PREFETCH_INDEX}" >&2
      exit 4
    fi
    prefetch_root="$("${PY}" -c "import json,sys; m=json.load(open(sys.argv[1],encoding='utf-8')); print(m.get('segment_grpo_run_dirs',{}).get(sys.argv[2],''))" "${PREFETCH_INDEX}" "${task}")"
    if [[ -z "${prefetch_root}" ]]; then
      echo "[mt10:phase8] ERROR: no segment_grpo_run_dirs entry for task=${task} in ${PREFETCH_INDEX}" >&2
      exit 4
    fi
    if [[ ! -d "${prefetch_root}" ]]; then
      echo "[mt10:phase8] ERROR: prefetch run root not a directory: ${prefetch_root}" >&2
      exit 4
    fi
    cmd+=(--prefetch-run-root "${prefetch_root}")
  fi
  if [[ -n "${MT10_PHASE8_WM_SELECTION_ENV_STEPS:-}" ]]; then
    cmd+=(--wm-selection-env-steps "${MT10_PHASE8_WM_SELECTION_ENV_STEPS}")
  fi
  if [[ -n "${MT10_PHASE8_PREFETCH_SEGMENT_INDEX:-}" ]]; then
    cmd+=(--prefetch-segment-index "${MT10_PHASE8_PREFETCH_SEGMENT_INDEX}")
  fi
  if [[ -n "${MT10_PHASE8_RUN_NAME:-}" ]]; then
    cmd+=(--run-name "${MT10_PHASE8_RUN_NAME}")
  fi
  "${cmd[@]}" 2>&1 | tee "${logf}"
  py_rc="${PIPESTATUS[0]:-1}"
  set -o pipefail
  if [[ "${py_rc}" -ne 0 ]]; then
    echo "[mt10:phase8] ERROR: campaign failed rc=${py_rc} task=${task}" >&2
    tail -n 80 "${logf}" >&2 || true
    rm -f "${logf}"
    exit "${py_rc}"
  fi
  run_dir="$(grep '\[campaign\] run_dir=' "${logf}" | tail -n 1 | sed 's/^.*run_dir=//' || true)"
  rm -f "${logf}"
  if [[ -z "${run_dir}" ]]; then
    echo "[mt10:phase8] ERROR: could not parse run_dir from campaign log for task=${task}" >&2
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
    "phase": 8,
    "phase6_index": str(Path("${INDEX_JSON}").resolve()),
    "run_name_prefix": __import__("os").environ.get("RUN_NAME_PREFIX")
    or __import__("os").environ.get("ORACLE_RUN_PREFIX")
    or "",
    "phase6_oracle_roots": p6.get("tasks", {}),
    "segment_grpo_run_dirs": {t: r for t, r in rows},
}
Path("${INDEX8_JSON}").parent.mkdir(parents=True, exist_ok=True)
Path("${INDEX8_JSON}").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
print(f"[mt10:phase8] wrote index: ${INDEX8_JSON}")
PY
rm -f "${MAP_TMP}"
