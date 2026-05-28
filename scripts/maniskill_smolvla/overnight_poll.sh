#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/rds/general/user/aa6622/home/project}"
source "${PROJECT_ROOT}/scripts/maniskill_smolvla/common.sh"
msm_prepare_runtime
POLL="${MSM_OVERNIGHT_POLL_SECONDS:-1200}"
JOBS="2869605.pbs-7 2869606.pbs-7 2869607.pbs-7 2869608.pbs-7 2869609.pbs-7"
DATA_DIR="${MSM_RAW_ROOT}/full_cpu124_v1/PutOnPlateInScene25Main-v3/16400/data"
log() { echo "[msm-poll] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
while true; do
  log "=== qstat ==="
  qstat -u "${USER}" 2>/dev/null | grep -E 'msm_|286959|286960' || log "no msm jobs in queue"
  if [[ -d "${DATA_DIR}" ]]; then
    n=$(find "${DATA_DIR}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
    log "npz=${n}/16400"
  fi
  for j in ${JOBS}; do
    st=$(qstat -f "${j}" 2>/dev/null | awk -F'= ' '/job_state =/{print $2; exit}' || true)
  if [[ -z "${st}" ]]; then
    ex=$(qstat -xf "${j}" 2>/dev/null | awk -F'= ' '/Exit_status =/{print $2; exit}' || qstat -Hf "${j}" 2>/dev/null | awk -F'= ' '/Exit_status =/{print $2; exit}' || true)
    log "job=${j} finished exit=${ex:-?}"
  fi
  done
  sleep "${POLL}"
done
