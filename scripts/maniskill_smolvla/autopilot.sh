#!/usr/bin/env bash
# Submit and monitor the full SmolVLA x ManiSkill PBS pipeline.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/general/user/aa6622/home/project}"
# shellcheck source=common.sh
source "${PROJECT_ROOT}/scripts/maniskill_smolvla/common.sh"
msm_prepare_runtime

STAGES=(
  "00_build_envs.pbs"
  "01_data_probe.pbs"
  "02_data_full.pbs"
  "03_convert_full.pbs"
  "04_sft_smoke.pbs"
  "05_sft_train.pbs"
  "06_benchmark.pbs"
)

job_state() {
  { qstat -f "$1" 2>/dev/null || true; } | awk -F'= ' '/job_state =/{print $2; exit}'
}

job_exit_status() {
  { qstat -xf "$1" 2>/dev/null || qstat -Hf "$1" 2>/dev/null || true; } \
    | awk -F'= ' '/exit_status =/{print $2; exit}'
}

write_rca() {
  local stage="$1"
  local job_id="$2"
  local exit_status="$3"
  local rca="${MSM_RUN_ROOT}/rca/${stage%.pbs}_${job_id}.md"
  {
    echo "# RCA: ${stage}"
    echo
    echo "- job_id: ${job_id}"
    echo "- exit_status: ${exit_status}"
    echo "- timestamp_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- run_root: ${MSM_RUN_ROOT}"
    echo
    echo "## Likely logs"
    echo "- ${MSM_RUN_ROOT}/logs/"
    echo "- ${PROJECT_ROOT}/artifacts/smolvla_maniskill/"
    echo
    echo "## Next action"
    echo "Inspect newest stage log, fix root cause, then rerun from ${stage}."
  } > "${rca}"
  echo "MSM_RCA_NEEDED stage=${stage} job_id=${job_id} exit_status=${exit_status} rca=${rca}"
}

submit_stage() {
  local stage="$1"
  local script="${MSM_SCRIPT_ROOT}/${stage}"
  msm_require_file "${script}"
  (cd "${PROJECT_ROOT}" && qsub "${script}")
}

monitor_job() {
  local stage="$1"
  local job_id="$2"
  echo "[msm-autopilot] monitor stage=${stage} job=${job_id}"
  while true; do
    state="$(job_state "${job_id}")"
    if [[ -z "${state}" ]]; then
      break
    fi
    echo "[msm-autopilot] stage=${stage} job=${job_id} state=${state}"
    sleep "${MSM_AUTOPILOT_POLL_SECONDS:-120}"
  done
  exit_status="$(job_exit_status "${job_id}")"
  if [[ -z "${exit_status}" ]]; then
    sleep 30
    exit_status="$(job_exit_status "${job_id}")"
  fi
  if [[ "${exit_status:-unknown}" != "0" ]]; then
    write_rca "${stage}" "${job_id}" "${exit_status:-unknown}"
    return 1
  fi
  echo "[msm-autopilot] stage=${stage} job=${job_id} done"
}

main() {
  echo "[msm-autopilot] run_root=${MSM_RUN_ROOT}"
  local start_index="${MSM_START_STAGE_INDEX:-0}"
  for idx in "${!STAGES[@]}"; do
    if (( idx < start_index )); then
      continue
    fi
    stage="${STAGES[$idx]}"
    job_id="$(submit_stage "${stage}")"
    echo "[msm-autopilot] submitted stage=${stage} job=${job_id}"
    monitor_job "${stage}" "${job_id}"
  done
  echo "MSM_AUTOPILOT_DONE run_root=${MSM_RUN_ROOT}"
}

main "$@"
