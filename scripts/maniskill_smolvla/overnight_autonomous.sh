#!/usr/bin/env bash
# Overnight autonomous monitor for SmolVLA ManiSkill PBS chain.
# Polls jobs, writes RCA on failure, requeues chain after known fixes (bounded retries).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/general/user/aa6622/home/project}"
# shellcheck source=common.sh
source "${PROJECT_ROOT}/scripts/maniskill_smolvla/common.sh"
msm_prepare_runtime

POLL_SECONDS="${MSM_OVERNIGHT_POLL_SECONDS:-1200}"
MAX_CHAIN_RETRIES="${MSM_OVERNIGHT_MAX_CHAIN_RETRIES:-3}"
STATE_FILE="${MSM_RUN_ROOT}/manifests/overnight_autonomous.env"

log() {
  echo "[msm-overnight] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
}

job_state() {
  { qstat -f "$1" 2>/dev/null || true; } | awk -F'= ' '/job_state =/{print $2; exit}'
}

job_exit_status() {
  { qstat -xf "$1" 2>/dev/null || qstat -Hf "$1" 2>/dev/null || true; } \
    | awk -F'= ' 'tolower($1) ~ /exit_status[[:space:]]*$/{print $2; exit}'
}

tail_stage_log() {
  local job_id="$1"
  local log_dir="${MSM_PROJECT_ARTIFACT_ROOT}/runs/${job_id}/logs"
  local newest=""
  newest="$(ls -t "${log_dir}"/*.log 2>/dev/null | head -n 1 || true)"
  if [[ -n "${newest}" ]]; then
    tail -n 80 "${newest}" || true
  fi
  tail -n 80 "${MSM_PROJECT_ARTIFACT_ROOT}/msm_"*.out 2>/dev/null || true
}

write_rca() {
  local label="$1"
  local job_id="$2"
  local exit_status="$3"
  local rca="${MSM_RUN_ROOT}/rca/overnight_${label}_${job_id}.md"
  {
    echo "# Overnight RCA: ${label}"
    echo "- job_id: ${job_id}"
    echo "- exit_status: ${exit_status}"
    echo "- utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "## log tail"
    echo '```text'
    tail_stage_log "${job_id}"
    echo '```'
  } > "${rca}"
  log "RCA ${rca}"
}

submit_chain() {
  local data_job convert_job
  python3 "${HOME}/.agents/skills/checking-pbs-gpu-availability/scripts/pbs_gpu_snapshot.py" -q v1_gpu72 || true
  data_job="$(cd "${PROJECT_ROOT}" && qsub "${MSM_SCRIPT_ROOT}/02_data_full.pbs")"
  convert_job="$(cd "${PROJECT_ROOT}" && qsub -W "depend=afterok:${data_job}" "${MSM_SCRIPT_ROOT}/03_convert_full.pbs")"
  bash "${MSM_SCRIPT_ROOT}/queue_afterok_gpu_tail.sh" --after-job "${convert_job}"
  log "submitted data=${data_job} convert=${convert_job}"
  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "data_job=${data_job}"
    echo "convert_job=${convert_job}"
    echo "chain_attempt=${1:-0}"
  } > "${STATE_FILE}"
  echo "${data_job}"
}

wait_job() {
  local job_id="$1"
  while true; do
    local state
    state="$(job_state "${job_id}")"
    if [[ -z "${state}" ]]; then
      break
    fi
    log "job=${job_id} state=${state}"
    sleep "${POLL_SECONDS}"
  done
  job_exit_status "${job_id}"
}

chain_done_ok() {
  local eval_job="$1"
  local st
  st="$(job_exit_status "${eval_job}")"
  [[ "${st:-}" == "0" ]]
}

main() {
  mkdir -p "${MSM_RUN_ROOT}/rca" "${MSM_RUN_ROOT}/manifests"
  local attempt=0
  local data_job=""
  while (( attempt <= MAX_CHAIN_RETRIES )); do
    log "chain_attempt=${attempt}"
    data_job="$(submit_chain "${attempt}")"
    wait_job "${data_job}"
    local data_exit
    data_exit="$(job_exit_status "${data_job}")"
    if [[ "${data_exit:-1}" != "0" ]]; then
      write_rca "data_full" "${data_job}" "${data_exit:-unknown}"
      attempt=$((attempt + 1))
      log "data failed exit=${data_exit}; retry chain if attempts remain"
      sleep 60
      continue
    fi
    log "data ok; downstream afterok chain running — monitor convert+GPU via qstat"
  if [[ -f "${STATE_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${STATE_FILE}"
  fi
    if [[ -n "${convert_job:-}" ]]; then
      wait_job "${convert_job}"
      local cexit
      cexit="$(job_exit_status "${convert_job}")"
      if [[ "${cexit:-1}" != "0" ]]; then
        write_rca "convert" "${convert_job}" "${cexit:-unknown}"
        attempt=$((attempt + 1))
        continue
      fi
    fi
    log "MSM_OVERNIGHT_PIPELINE_SUBMITTED_OK data=${data_job}"
    return 0
  done
  log "MSM_OVERNIGHT_EXCEEDED_RETRIES"
  return 1
}

main "$@"
