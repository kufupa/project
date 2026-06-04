#!/usr/bin/env bash
# Submit and monitor the full SmolVLA x ManiSkill PBS pipeline.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/general/user/aa6622/home/project}"
# shellcheck source=common.sh
source "${PROJECT_ROOT}/scripts/maniskill_smolvla/common.sh"
msm_prepare_runtime

STAGES=(
  "03a_audit_full.pbs"
  "03_convert_full.pbs"
  "04_sft_smoke.pbs"
  "05_sft_train.pbs"
  "06_benchmark.pbs"
)

GPU_STAGES=(
  "04_sft_smoke.pbs"
  "05_sft_train.pbs"
  "06_benchmark.pbs"
)

job_state() {
  { qstat -f "$1" 2>/dev/null || true; } | awk -F'= ' '/job_state =/{print $2; exit}'
}

job_exit_status() {
  { qstat -xf "$1" 2>/dev/null || qstat -Hf "$1" 2>/dev/null || true; } \
    | awk -F'= ' 'tolower($1) ~ /exit_status[[:space:]]*$/{print $2; exit}'
}

write_rca() {
  local stage="$1"
  local job_id="$2"
  local exit_status="$3"
  local rca="${MSM_RUN_ROOT}/rca/${stage%.pbs}_${job_id}.md"
  local newest_log=""
  newest_log="$(ls -t "${MSM_RUN_ROOT}/logs"/*.log 2>/dev/null | head -n 1 || true)"
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
    if [[ -n "${newest_log}" ]]; then
      echo
      echo "## Latest stage log tail"
      echo
      echo '```text'
      tail -n 120 "${newest_log}" || true
      echo '```'
    fi
    echo
    echo "## PBS record"
    echo
    echo '```text'
    qstat -xf "${job_id}" 2>/dev/null || qstat -Hf "${job_id}" 2>/dev/null || true
    echo '```'
    echo
    echo "## Failure classification hints"
    echo "- scheduler/resource: qstat comments, Can Never Run, Qlist, walltime, mem"
    echo "- dependency/import: ModuleNotFoundError, ImportError, Illegal instruction, segmentation fault"
    echo "- data schema: bad shapes, missing keys, duplicate decoded signatures"
    echo "- LeRobot config: draccus parse errors, feature key mismatch, checkpoint config mismatch"
    echo "- CUDA/OOM: CUDA out of memory, CUBLAS, NCCL, killed"
    echo "- eval/action: NaN action, clipped action saturation, env.step failure"
    echo
    echo "## Next action"
    echo "Inspect newest stage log, fix root cause, then rerun from ${stage}."
  } > "${rca}"
  {
    echo
    echo "## $(date -u +%Y-%m-%dT%H:%M:%SZ) RCA needed"
    echo
    echo "- stage: ${stage}"
    echo "- job_id: ${job_id}"
    echo "- exit_status: ${exit_status}"
    echo "- rca: ${rca}"
  } >> "${PROJECT_ROOT}/docs/eggroll/smolvla_maniskill_handoff.md"
  echo "MSM_RCA_NEEDED stage=${stage} job_id=${job_id} exit_status=${exit_status} rca=${rca}"
}

is_gpu_stage() {
  local stage="$1"
  local gpu_stage
  for gpu_stage in "${GPU_STAGES[@]}"; do
    [[ "${stage}" == "${gpu_stage}" ]] && return 0
  done
  return 1
}

check_rtx6000_capacity() {
  local snapshot="${HOME}/.agents/skills/checking-pbs-gpu-availability/scripts/pbs_gpu_snapshot.py"
  if [[ -x "${snapshot}" || -f "${snapshot}" ]]; then
    python3 "${snapshot}" -q v1_gpu72 || true
  else
    echo "[msm-autopilot] warning: missing ${snapshot}; submitting RTX6000 PBS job without snapshot"
  fi
}

submit_stage() {
  local stage="$1"
  local script="${MSM_SCRIPT_ROOT}/${stage}"
  msm_require_file "${script}"
  if is_gpu_stage "${stage}"; then
    check_rtx6000_capacity
  fi
  local host="${MSM_PBS_HOST:-}"
  local select_spec=""
  if [[ -n "${host}" ]]; then
    case "${stage}" in
      01_data_probe.pbs) select_spec="1:ncpus=4:mem=32gb:ngpus=1:vnode=${host}:gpu_type=RTX6000" ;;
      02_data_full.pbs) select_spec="1:ncpus=16:mem=128gb:ngpus=1:vnode=${host}:gpu_type=RTX6000" ;;
      04_sft_smoke.pbs) select_spec="1:ncpus=8:mem=96gb:ngpus=1:vnode=${host}:gpu_type=RTX6000" ;;
      05_sft_train.pbs) select_spec="1:ncpus=16:mem=128gb:ngpus=1:vnode=${host}:gpu_type=RTX6000" ;;
      06_benchmark.pbs) select_spec="1:ncpus=8:mem=96gb:ngpus=1:vnode=${host}:gpu_type=RTX6000" ;;
    esac
  fi
  if [[ -n "${select_spec}" ]]; then
    (cd "${PROJECT_ROOT}" && qsub -l "select=${select_spec}" "${script}")
  else
    (cd "${PROJECT_ROOT}" && qsub "${script}")
  fi
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
  local max_retries="${MSM_STAGE_MAX_RETRIES:-2}"
  for idx in "${!STAGES[@]}"; do
    if (( idx < start_index )); then
      continue
    fi
    stage="${STAGES[$idx]}"
    local attempt=0
    while true; do
      attempt=$((attempt + 1))
      job_id="$(submit_stage "${stage}")"
      echo "[msm-autopilot] submitted stage=${stage} attempt=${attempt} job=${job_id}"
      if monitor_job "${stage}" "${job_id}"; then
        break
      fi
      if (( attempt > max_retries )); then
        echo "[msm-autopilot] stage=${stage} exceeded max_retries=${max_retries}"
        return 1
      fi
      echo "[msm-autopilot] retry stage=${stage} next_attempt=$((attempt + 1))"
      sleep "${MSM_AUTOPILOT_RETRY_SLEEP_SECONDS:-300}"
    done
  done
  echo "MSM_AUTOPILOT_DONE run_root=${MSM_RUN_ROOT}"
}

main "$@"
