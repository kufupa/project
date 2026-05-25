#!/usr/bin/env bash
# Queue 25ep/25nenv eval sweeps for gamble runs K and M only (leave L untouched).
# - eval now: all checkpoints present at submit time
# - eval afterok: remainder after train finishes (skip-existing merges into same sweep)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

GAMBLE_ROOT="${PHASE50_GAMBLE_ROOT:-${PROJECT_ROOT}/artifacts/phase50_eggroll_gamble/20260524_082551}"
STATE="${GAMBLE_ROOT}/queued_jobs.tsv"
EVAL_STATE="${GAMBLE_ROOT}/queued_eval_jobs.tsv"
EVAL_PBS="${PROJECT_ROOT}/scripts/eggroll/submit_phase50_eggroll_eval_sweep.pbs"
WATCH_SCRIPT="${PROJECT_ROOT}/scripts/eggroll/watch_phase50_skip_inline_eval.sh"
BASE_CKPT="${PHASE50_CHECKPOINT:-/rds/general/user/aa6622/home/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
SWEEP_NAME="${PHASE50_SWEEP_NAME:-eval_seeded25_nenv25_every2}"

if [[ ! -f "${STATE}" || ! -f "${EVAL_PBS}" ]]; then
  echo "error: missing ${STATE} or ${EVAL_PBS}" >&2
  exit 2
fi

max_ckpt() {
  local run_dir="$1"
  "${PYTHON:-python3}" - "${run_dir}" <<'PY'
from pathlib import Path
import re
import sys

vals = []
for p in (Path(sys.argv[1]) / "checkpoints").glob("update_*.pt"):
    m = re.match(r"update_(\d+)\.pt$", p.name)
    if m:
        vals.append(int(m.group(1)))
print(max(vals) if vals else 0)
PY
}

submit_eval() {
  local label="$1"
  local run_dir="$2"
  local train_job="$3"
  local max_update="$4"
  local skip_existing="$5"
  local depend="${6:-}"

  local env_vars
  env_vars="PHASE50_RUN_DIR=${run_dir},PHASE50_CHECKPOINT=${BASE_CKPT},PHASE50_TASK=push-v3,PHASE50_SWEEP_NAME=${SWEEP_NAME},PHASE50_INCLUDE_BASE_EVAL=1,PHASE50_MIN_UPDATE=0,PHASE50_MAX_UPDATE=${max_update},PHASE50_STRIDE=2,PHASE50_EVAL_EPISODES=25,PHASE50_EVAL_N_ENVS=25,PHASE50_EVAL_SEED_START=1000,PHASE50_EVAL_CHUNK_LEN=5,PHASE50_EVAL_MAX_STEPS=120,PHASE50_EVAL_SKIP_EXISTING=${skip_existing},SMOLVLA_METAWORLD_RESET_MODE=random_seeded"

  local qsub_args=(
    -N "${label}"
    -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000
    -o "${run_dir}/${label}.pbs.out"
    -v "${env_vars}"
  )
  if [[ -n "${depend}" ]]; then
    qsub_args+=(-W "depend=afterok:${depend}")
  fi

  qsub "${qsub_args[@]}" "${EVAL_PBS}"
}

: > "${EVAL_STATE}"

while IFS=$'\t' read -r job_id run_name run_dir _rest; do
  [[ -z "${job_id}" ]] && continue
  case "${run_name}" in
    run_k_*|run_m_*)
      ;;
    *)
      continue
      ;;
  esac

  train_job="${job_id%%.*}"
  mkdir -p "${run_dir}"
  touch "${run_dir}/.phase50_skip_final_eval"

  max_now="$(max_ckpt "${run_dir}")"
  if [[ "${max_now}" -lt 2 ]]; then
    echo "warning: ${run_name} has max_update=${max_now}; skipping now-eval" >&2
    now_id=""
  else
    now_label="$(printf 'p50ev%sN' "${run_name:4:6}" | tr -cd '[:alnum:]' | cut -c1-14)"
    now_id="$(submit_eval "${now_label}" "${run_dir}" "${train_job}" "${max_now}" "0" "")"
    printf '%s\t%s\t%s\tphase=now\tmax_update=%s\ttrain_job=%s\n' \
      "${now_id}" "${run_name}" "${run_dir}" "${max_now}" "${train_job}" | tee -a "${EVAL_STATE}"
    echo "queued now-eval job=${now_id} run=${run_name} max_update=${max_now}"
  fi

  tail_label="$(printf 'p50ev%sT' "${run_name:4:6}" | tr -cd '[:alnum:]' | cut -c1-14)"
  tail_id="$(submit_eval "${tail_label}" "${run_dir}" "${train_job}" "" "1" "${train_job}")"
  printf '%s\t%s\t%s\tphase=afterok\tmax_update=auto\ttrain_job=%s\tdepends_now=%s\n' \
    "${tail_id}" "${run_name}" "${run_dir}" "${train_job}" "${now_id:-none}" | tee -a "${EVAL_STATE}"
  echo "queued afterok-eval job=${tail_id} run=${run_name} after train=${train_job}"
done < "${STATE}"

if [[ -x "${WATCH_SCRIPT}" ]]; then
  nohup "${WATCH_SCRIPT}" "${STATE}" >> "${GAMBLE_ROOT}/skip_inline_eval_watch.log" 2>&1 &
  echo "started skip-inline-eval watcher pid=$! log=${GAMBLE_ROOT}/skip_inline_eval_watch.log"
else
  echo "warning: watcher not executable: ${WATCH_SCRIPT}" >&2
fi

echo "PHASE50_EGGROLL_GAMBLE_EVAL_KM_QUEUED eval_state=${EVAL_STATE}"
qstat -u "${USER}"
