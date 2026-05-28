#!/usr/bin/env bash
# After primary data walltime-fail: drop held downstream jobs, convert+GPU tail after resume.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/general/user/aa6622/home/project}"
source "${PROJECT_ROOT}/scripts/maniskill_smolvla/common.sh"
msm_prepare_runtime

DATA_JOB="${1:?usage: rechain_after_data_fail.sh DATA_JOB RESUME_JOB [OLD_CONVERT ...]}"
RESUME_JOB="${2:?usage: rechain_after_data_fail.sh DATA_JOB RESUME_JOB [OLD_CONVERT ...]}"
OLD_CONVERT="${3:-}"
OLD_SMOKE="${4:-}"
OLD_TRAIN="${5:-}"
OLD_EVAL="${6:-}"

LOCK="${MSM_RUN_ROOT}/manifests/rechain_after_data_fail.env"
if [[ -f "${LOCK}" ]]; then
  echo "[rechain] already ran: $(cat "${LOCK}")"
  exit 0
fi

data_ex="$(qstat -xf "${DATA_JOB}" 2>/dev/null | awk -F'= ' '/Exit_status =/{print $2; exit}' || qstat -Hf "${DATA_JOB}" 2>/dev/null | awk -F'= ' '/Exit_status =/{print $2; exit}' || true)"
resume_st="$(qstat -f "${RESUME_JOB}" 2>/dev/null | awk -F'= ' '/job_state =/{print $2; exit}' || true)"
resume_ex="$(qstat -xf "${RESUME_JOB}" 2>/dev/null | awk -F'= ' '/Exit_status =/{print $2; exit}' || qstat -Hf "${RESUME_JOB}" 2>/dev/null | awk -F'= ' '/Exit_status =/{print $2; exit}' || true)"

echo "[rechain] data=${DATA_JOB} exit=${data_ex:-?} resume=${RESUME_JOB} state=${resume_st:-done} exit=${resume_ex:-?}"

if [[ "${data_ex}" == "0" ]]; then
  echo "[rechain] primary data exit 0; original afterok chain owns convert"
  exit 0
fi

if [[ "${resume_st}" == "R" || "${resume_st}" == "Q" || "${resume_st}" == "H" ]]; then
  echo "[rechain] wait for resume to finish"
  exit 0
fi

if [[ "${resume_ex}" != "0" ]]; then
  echo "[rechain] resume failed exit=${resume_ex}" >&2
  exit 1
fi

D="${MSM_RAW_ROOT}/full_cpu124_v1/${MSM_ENV_ID}/16400/data"
n="$(find "${D}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')"
echo "[rechain] npz=${n}"
if [[ "${n}" -lt 16384 ]]; then
  echo "[rechain] insufficient npz" >&2
  exit 1
fi

for j in "${OLD_CONVERT}" "${OLD_SMOKE}" "${OLD_TRAIN}" "${OLD_EVAL}"; do
  [[ -z "${j}" ]] && continue
  if qstat "${j}" 2>/dev/null | grep -q .; then
    echo "[rechain] qdel held downstream ${j}"
    qdel "${j}" 2>/dev/null || true
  fi
done

conv="$(cd "${PROJECT_ROOT}" && qsub -W "depend=afterok:${RESUME_JOB}" "${MSM_SCRIPT_ROOT}/03_convert_full.pbs")"
echo "[rechain] convert=${conv}"
"${MSM_SCRIPT_ROOT}/queue_afterok_gpu_tail.sh" --after-job "${conv}"
msm_write_manifest "${LOCK}" \
  "stage=rechain_after_data_fail" \
  "data_job=${DATA_JOB}" \
  "resume_job=${RESUME_JOB}" \
  "convert_job=${conv}"
