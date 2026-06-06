#!/usr/bin/env bash
# Poll DGPO overnight jobs every 120s. RCA + log on failure.
set -euo pipefail

PROJECT_ROOT="/vol/bitbucket/aa6622/project"
LOG="${PROJECT_ROOT}/docs/dgpo_overnight_log.md"
POLL="${POLL_SECONDS:-120}"
JOBS=(247467 247468)

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"; }

rca_job() {
  local jid="$1"
  local out="${PROJECT_ROOT}/"*"_${jid}.out"
  local err="${PROJECT_ROOT}/"*"_${jid}.err"
  log "### FAIL job ${jid}"
  log "- out tail:"
  tail -30 ${out} 2>/dev/null | sed 's/^/  /' >> "${LOG}" || true
  log "- err tail:"
  tail -30 ${err} 2>/dev/null | sed 's/^/  /' >> "${LOG}" || true
  if grep -qiE 'OutOfMemory|QOSMaxMemory|Connection refused|Traceback' ${err} 2>/dev/null; then
    grep -iE 'OutOfMemory|QOSMaxMemory|Connection refused|Traceback' ${err} 2>/dev/null | head -5 | sed 's/^/  RCA: /' >> "${LOG}"
  fi
}

wait_job() {
  local jid="$1"
  while squeue -j "${jid}" -h 2>/dev/null | grep -q .; do
    local out
    out="$(ls ${PROJECT_ROOT}/*_${jid}.out 2>/dev/null | head -1)"
    if [[ -n "${out}" ]]; then
      local last
      last="$(grep -E 'phase111_grpo_update|\[dgpo\]|error:|CHAIN_OK|SMOKE_OK' "${out}" 2>/dev/null | tail -1 || true)"
      log "poll ${jid}: RUNNING last=${last:-(no lines yet)}"
    else
      log "poll ${jid}: RUNNING (no out yet)"
    fi
    sleep "${POLL}"
  done
  local state
  state="$(sacct -j "${jid}" --format=State,ExitCode -n -P 2>/dev/null | head -1 || echo UNKNOWN:?:?)"
  if [[ "${state}" == COMPLETED* ]] && [[ "${state}" == *":0"* ]]; then
    log "job ${jid} GREEN state=${state}"
    grep -E 'phase111_grpo_update update=|DGPO_CHUNK|FLOW_SDE_MOONSHOT' ${PROJECT_ROOT}/*_${jid}.out 2>/dev/null | tail -3 | sed "s/^/  /" >> "${LOG}" || true
  else
    log "job ${jid} FAILED state=${state}"
    rca_job "${jid}"
  fi
}

cd "${PROJECT_ROOT}"
log "monitor start jobs=${JOBS[*]} poll=${POLL}s"
for jid in "${JOBS[@]}"; do
  wait_job "${jid}"
done
log "monitor E0/E1 done — chain script handles E2/E3"
