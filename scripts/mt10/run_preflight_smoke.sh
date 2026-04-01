#!/usr/bin/env bash
# Cheap 1-episode-per-phase GPU smoke for each MT10 task (sequential, no nested sbatch).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
# shellcheck source=mt10_tasks.sh
source "${SCRIPT_DIR}/mt10_tasks.sh"
if [[ -n "${MT10_ONLY_TASK:-}" ]]; then
  MT10_TASK_IDS=("${MT10_ONLY_TASK}")
fi

PY="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}/bin/python"
ORACLE_SH="${PROJECT_ROOT}/scripts/oracle/run_oracle_baseline_eval.sh"
CAMPAIGN="${PROJECT_ROOT}/scripts/segment_grpo/run_all60_frame50_k3.py"
PHASE9_PY="${PROJECT_ROOT}/scripts/run_phase9_oracle_vs_wm.py"
# Default path includes SLURM_JOB_ID when set so concurrent preflights do not clobber the same JSON.
REPORT_JSON="${MT10_PREFLIGHT_REPORT_JSON:-${PROJECT_ROOT}/artifacts/mt10_runs/mt10_preflight_report${SLURM_JOB_ID:+_job${SLURM_JOB_ID}}.json}"
ROWS_TMP="$(mktemp)"

PREFIX="${MT10_PREFLIGHT_PREFIX:-mt10_preflight}"
export ORACLE_RUN_PREFIX="${ORACLE_RUN_PREFIX:-$PREFIX}"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-$ORACLE_RUN_PREFIX}"

mkdir -p "$(dirname "${REPORT_JSON}")"
rm -f "${ROWS_TMP}"
touch "${ROWS_TMP}"

if [[ ! -x "${PY}" ]]; then
  echo "[mt10:preflight] ERROR: python not executable: ${PY}" >&2
  exit 3
fi
if [[ ! -f "${ORACLE_SH}" ]]; then
  echo "[mt10:preflight] ERROR: missing ${ORACLE_SH}" >&2
  exit 3
fi
if [[ -z "${JEPA_REPO:-}" || ! -d "${JEPA_REPO}" ]]; then
  echo "[mt10:preflight] ERROR: JEPA_REPO must be set to an existing directory." >&2
  exit 3
fi
if [[ -z "${SMOLVLA_CHECKPOINT:-}" ]]; then
  echo "[mt10:preflight] WARN: SMOLVLA_CHECKPOINT unset; Phase8 child may fail to load policy." >&2
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

record() {
  # task p6 p8 p9 err
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "${5:-}" >>"${ROWS_TMP}"
}

for task in "${MT10_TASK_IDS[@]}"; do
  echo "[mt10:preflight] === task=${task} ==="
  p6="fail"
  p8="skip"
  p9="skip"
  err=""
  log6="$(mktemp)"
  set +o pipefail
  if ! ORACLE_BASELINE_EPISODES=1 ORACLE_BASELINE_SEED=1000 ORACLE_BASELINE_TASK="${task}" \
    ORACLE_BASELINE_EPISODE_LENGTH=120 ORACLE_SAVE_FRAMES=true ORACLE_BASELINE_VIDEO=true \
    bash "${ORACLE_SH}" --episodes 1 --seed 1000 --task "${task}" --episode-length 120 \
    --save-frames true --video true 2>&1 | tee "${log6}"; then
    err="phase6_oracle_baseline_failed"
    record "${task}" "${p6}" "${p8}" "${p9}" "${err}"
    rm -f "${log6}"
    echo "[mt10:preflight] ERROR: phase6 failed for ${task}" >&2
    exit 4
  fi
  set -o pipefail
  oracle_dir="$(awk '/Baseline eval output directory:/{print $NF}' "${log6}" | tail -n 1 || true)"
  rm -f "${log6}"
  if [[ -z "${oracle_dir}" || ! -d "${oracle_dir}" ]]; then
    err="phase6_output_dir_unresolved"
    record "${task}" "${p6}" "${p8}" "${p9}" "${err}"
    echo "[mt10:preflight] ERROR: could not resolve oracle dir for ${task}" >&2
    exit 4
  fi
  p6="ok"

  log8="$(mktemp)"
  set +o pipefail
  cmd8=(
    "${PY}" "${CAMPAIGN}"
    --seed-base 1000
    --episode-start 0
    --episodes 1
    --goal-frame-index 50
    --num-candidates 3
    --chunk-len 50
    --max-steps 50
    --smolvla-n-action-steps 50
    --task "${task}"
    --carry-mode sim
    --wm-rollout-mode iterative
    --wm-scoring-latent visual
    --comparison-strip-overlay
    --artifacts-root "${PROJECT_ROOT}/artifacts"
    --oracle-run-root "${oracle_dir}"
    --output-root "${PROJECT_ROOT}/artifacts/phase08_segment_grpo_baseline"
    --run-name-prefix "${PREFIX}_p8smoke"
  )
  [[ -n "${SMOLVLA_CHECKPOINT:-}" ]] && cmd8+=(--checkpoint "${SMOLVLA_CHECKPOINT}")
  cmd8+=(--jepa-repo "${JEPA_REPO}")
  if ! "${cmd8[@]}" 2>&1 | tee "${log8}"; then
    err="phase8_segment_grpo_failed"
    record "${task}" "${p6}" "${p8}" "${p9}" "${err}"
    rm -f "${log8}"
    echo "[mt10:preflight] ERROR: phase8 failed for ${task}" >&2
    exit 5
  fi
  set -o pipefail
  rm -f "${log8}"
  p8="ok"

  log9="$(mktemp)"
  set +o pipefail
  if ! "${PY}" "${PHASE9_PY}" \
    --oracle-run-root "${oracle_dir}" \
    --artifacts-root "${PROJECT_ROOT}/artifacts" \
    --output-root "${PROJECT_ROOT}/artifacts/phase09_oracle_vs_wm_baseline" \
    --task "${task}" \
    --episodes 1 \
    --goal-frame-index 50 \
    --max-steps 50 \
    --chunk-len 50 \
    --jepa-repo "${JEPA_REPO}" \
    --device cuda \
    --wm-rollout-mode iterative \
    --wm-scoring-latent visual \
    2>&1 | tee "${log9}"; then
    err="phase9_oracle_vs_wm_failed"
    record "${task}" "${p6}" "${p8}" "${p9}" "${err}"
    rm -f "${log9}"
    echo "[mt10:preflight] ERROR: phase9 failed for ${task}" >&2
    exit 6
  fi
  set -o pipefail
  rm -f "${log9}"
  p9="ok"
  record "${task}" "${p6}" "${p8}" "${p9}" ""
done

"${PY}" <<PY
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

rows = []
for line in Path("${ROWS_TMP}").read_text(encoding="utf-8").splitlines():
    parts = line.split("\t")
    if len(parts) < 5:
        continue
    task, p6, p8, p9, err = parts[0], parts[1], parts[2], parts[3], parts[4]
    rows.append({"task": task, "phase6": p6, "phase8": p8, "phase9": p9, "error": err or None})

out = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "kind": "mt10_preflight_smoke",
    "run_prefix": "${PREFIX}",
    "rows": rows,
}
Path("${REPORT_JSON}").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
print(f"[mt10:preflight] wrote report: ${REPORT_JSON}")
PY
rm -f "${ROWS_TMP}"
