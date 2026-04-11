#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

ORACLE_RUN_DIR="${ORACLE_RUN_DIR:-/vol/bitbucket/aa6622/project/artifacts/phase06_oracle_baseline/run_20260411T131839Z_ep60_voracle_tpush_v3_s1000_r402093}"
TOP_K="${TOP_K:-15}"
PYTHON_BIN="${SMOLVLA_PYTHON_BIN:-${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}/bin/python}"
SAVE_FRAMES="false"
FORCE_DRY_RUN="${FORCE_DRY_RUN:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: smolvla python not found" >&2
  exit 2
fi

if [[ "${FORCE_DRY_RUN}" == "1" ]]; then
  CAMPAIGN_OUT="$(
    bash "${SCRIPT_DIR}/launch_pushv3_smolvla_topk15.sh" --oracle-run-dir "${ORACLE_RUN_DIR}" --top-k "${TOP_K}" --dry-run
  )"
else
  CAMPAIGN_OUT="$(
    bash "${SCRIPT_DIR}/launch_pushv3_smolvla_topk15.sh" --oracle-run-dir "${ORACLE_RUN_DIR}" --top-k "${TOP_K}" \
    || {
      echo "warning: launcher submission failed (possibly scheduler/QOS), retrying in dry-run mode..." >&2
      bash "${SCRIPT_DIR}/launch_pushv3_smolvla_topk15.sh" --oracle-run-dir "${ORACLE_RUN_DIR}" --top-k "${TOP_K}" --dry-run
    }
  )"
fi
readarray -t INFO <<<"${CAMPAIGN_OUT}"
CAMPAIGN_DIR="${INFO[0]#campaign_dir=}"
TARGETS_JSON="${INFO[1]#targets_json=}"

SMOLVLA_EPISODES_PER_TARGET=1
export SMOLVLA_EPISODES_PER_TARGET SAVE_FRAMES CAMPAIGN_DIR
export SMOLVLA_SAVE_FRAMES=false

bash "${SCRIPT_DIR}/run_smolvla_target_episode.sh" "${TARGETS_JSON}" 0
