#!/usr/bin/env bash
# Poll sparse GRPO moonshot PBS jobs; RCA + log status every 20m.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/general/user/aa6622/home/project}"
POLL_SECONDS="${SPARSE_GRPO_POLL_SECONDS:-1200}"
JOB_IDS_FILE="${PROJECT_ROOT}/docs/sparse_grpo_moonshot_job_ids.txt"
STATUS_LOG="${PROJECT_ROOT}/logs/overnight_sparse_grpo_status.log"
RCA_DIR="${PROJECT_ROOT}/docs/rca"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${STATUS_LOG}"
}

job_state() {
  { qstat -f "$1" 2>/dev/null || true; } | awk -F'= ' '/job_state =/{print $2; exit}'
}

job_exit_status() {
  { qstat -xf "$1" 2>/dev/null || qstat -Hf "$1" 2>/dev/null || true; } \
    | awk -F'= ' 'tolower($1) ~ /exit_status[[:space:]]*$/{print $2; exit}'
}

tail_pbs_log() {
  local pattern="$1"
  local f
  f="$(ls -t "${PROJECT_ROOT}/logs/pbs/grpo/"*${pattern}* 2>/dev/null | head -n 1 || true)"
  if [[ -n "${f}" ]]; then
    tail -n 60 "${f}" || true
  fi
}

write_rca() {
  local label="$1" job_id="$2" exit_st="$3"
  mkdir -p "${RCA_DIR}"
  local rca="${RCA_DIR}/sparse_grpo_${label}_${job_id}_$(date -u +%Y%m%dT%H%M%SZ).md"
  {
    echo "# Sparse GRPO RCA: ${label}"
    echo "- job_id: ${job_id}"
    echo "- exit_status: ${exit_st}"
    echo "- utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "## qstat -xf"
    echo '```text'
    qstat -xf "${job_id}" 2>&1 || true
    echo '```'
    echo
    echo "## log tail"
    echo '```text'
    tail_pbs_log "${label}"
    echo '```'
  } > "${rca}"
  log "RCA written ${rca}"
}

parse_jobs() {
  [[ -f "${JOB_IDS_FILE}" ]] || return 0
  grep -E '^(A|B)_TRAIN=' "${JOB_IDS_FILE}" || true
}

check_u1() {
  local run_dir="$1"
  local prog="${run_dir}/progress.jsonl"
  if [[ ! -f "${prog}" ]]; then
    return 1
  fi
  python3 - <<PY "${prog}"
import json, sys
path = sys.argv[1]
for line in open(path):
    row = json.loads(line)
    if int(row.get("update", -1)) >= 1:
        print("ok", "skipped" if row.get("skipped") else "trained")
        sys.exit(0)
sys.exit(1)
PY
}

monitor_once() {
  local tick=0
  tick="$(grep -c '^[0-9]' "${STATUS_LOG}" 2>/dev/null || echo 0)"
  tick=$((tick + 1))
  local parts=()
  while IFS= read -r line; do
    parts+=("${line}")
  done < <(parse_jobs)

  local summary="TICK${tick}:"
  for line in "${parts[@]}"; do
    local tag="${line%%_TRAIN=*}"
    local rest="${line#*=}"
    local train_job="${rest%% *}"
    local state
    state="$(job_state "${train_job}")"
    if [[ -z "${state}" ]]; then
      local ex
      ex="$(job_exit_status "${train_job}")"
      if [[ -n "${ex}" && "${ex}" != "0" ]]; then
        write_rca "${tag}" "${train_job}" "${ex}"
        summary+=" ${tag}_train=FAIL(${ex})"
      else
        summary+=" ${tag}_train=done"
      fi
    else
      summary+=" ${tag}_train=${state}"
    fi
  done
  log "${summary}"
}

main() {
  mkdir -p "${PROJECT_ROOT}/logs" "${RCA_DIR}" "${PROJECT_ROOT}/logs/pbs/grpo"
  log "sparse GRPO overnight monitor start poll=${POLL_SECONDS}s"
  while true; do
    monitor_once
    sleep "${POLL_SECONDS}"
  done
}

main "$@"
