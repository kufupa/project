#!/usr/bin/env bash
# Queue scarce-GPU SmolVLA stages immediately, chained by PBS afterok deps.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/general/user/aa6622/home/project}"
# shellcheck source=common.sh
source "${PROJECT_ROOT}/scripts/maniskill_smolvla/common.sh"
msm_prepare_runtime

usage() {
  cat <<'USAGE'
usage: queue_afterok_gpu_tail.sh --after-job JOB_ID

Queues:
  04_sft_smoke.pbs afterok:JOB_ID
  05_sft_train.pbs afterok:<smoke_job>
  06_benchmark.pbs afterok:<train_job>
USAGE
}

after_job=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --after-job)
      after_job="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${after_job}" ]]; then
  echo "missing --after-job JOB_ID" >&2
  usage >&2
  exit 2
fi

submit_afterok() {
  local dep="$1"
  local script="$2"
  msm_require_file "${MSM_SCRIPT_ROOT}/${script}"
  (cd "${PROJECT_ROOT}" && qsub -W "depend=afterok:${dep}" "${MSM_SCRIPT_ROOT}/${script}")
}

smoke_job="$(submit_afterok "${after_job}" "04_sft_smoke.pbs")"
train_job="$(submit_afterok "${smoke_job}" "05_sft_train.pbs")"
eval_job="$(submit_afterok "${train_job}" "06_benchmark.pbs")"

manifest="${MSM_RUN_ROOT}/manifests/afterok_gpu_tail.env"
msm_write_manifest \
  "${manifest}" \
  "stage=afterok_gpu_tail" \
  "after_job=${after_job}" \
  "sft_smoke_job=${smoke_job}" \
  "sft_train_job=${train_job}" \
  "benchmark_job=${eval_job}"

echo "MSM_AFTEROK_GPU_TAIL_QUEUED after_job=${after_job} sft_smoke=${smoke_job} sft_train=${train_job} benchmark=${eval_job} manifest=${manifest}"
