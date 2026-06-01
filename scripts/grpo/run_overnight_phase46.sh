#!/usr/bin/env bash
# Submit Phase46 GPU DAG (smoke -> train -> tiered RLinf eval) and follow with autopilot.
set -euo pipefail

LOGPROB_MODE="gaussian"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --logprob-mode)
      LOGPROB_MODE="${2:-gaussian}"
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "${LOGPROB_MODE}" != "gaussian" ]]; then
  echo "only gaussian chain enabled in phase46 overnight (flow_sde needs venv hook)" >&2
  exit 2
fi

PROJECT="/vol/bitbucket/aa6622/project"
RLINF="/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo"
BASE_CKPT="/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_ROOT="${PROJECT}/artifacts/phase46/${STAMP}_${LOGPROB_MODE}"
SMOKE_OUT="${RUN_ROOT}/smoke"
TRAIN_OUT="${RUN_ROOT}/train"
EVAL_OUT="${RUN_ROOT}/eval"
MANIFEST="${RUN_ROOT}/jobs_manifest.jsonl"

mkdir -p "${RUN_ROOT}"
: > "${MANIFEST}"

bash -n "${PROJECT}/scripts/grpo/submit_phase46_smoke_a30.slurm"
bash -n "${PROJECT}/scripts/grpo/submit_phase46_train_20upd_a30.slurm"
bash -n "${RLINF}/scripts/slurm/smolvla_phase46_tiered_eval_rlinf_a30.slurm"
chmod +x "${RLINF}/scripts/run_phase46_tiered_eval_rlinf.sh"

J1="$(sbatch --parsable --chdir="${PROJECT}" \
  "${PROJECT}/scripts/grpo/submit_phase46_smoke_a30.slurm" \
  "${BASE_CKPT}" "${SMOKE_OUT}")"
echo "submitted smoke ${J1}"
echo "{\"stage\":\"smoke\",\"job_id\":\"${J1}\",\"poll_state\":\"PENDING\",\"logprob_mode\":\"${LOGPROB_MODE}\",\"smoke_out\":\"${SMOKE_OUT}\"}" >> "${MANIFEST}"

J2="$(sbatch --parsable --chdir="${PROJECT}" --dependency=afterok:"${J1}" \
  "${PROJECT}/scripts/grpo/submit_phase46_train_20upd_a30.slurm" \
  "${BASE_CKPT}" "${TRAIN_OUT}")"
echo "submitted train ${J2}"
echo "{\"stage\":\"train\",\"job_id\":\"${J2}\",\"poll_state\":\"PENDING\",\"logprob_mode\":\"${LOGPROB_MODE}\",\"train_out\":\"${TRAIN_OUT}\"}" >> "${MANIFEST}"

J3="$(sbatch --parsable --dependency=afterok:"${J2}" \
  "${RLINF}/scripts/slurm/smolvla_phase46_tiered_eval_rlinf_a30.slurm" \
  "${TRAIN_OUT}/checkpoints" "${EVAL_OUT}" 2 20)"
echo "submitted eval ${J3}"
echo "{\"stage\":\"eval\",\"job_id\":\"${J3}\",\"poll_state\":\"PENDING\",\"logprob_mode\":\"${LOGPROB_MODE}\",\"eval_out\":\"${EVAL_OUT}\",\"train_out\":\"${TRAIN_OUT}\"}" >> "${MANIFEST}"

ln -sfn "${RUN_ROOT}" "${PROJECT}/artifacts/phase46/latest"
echo "PHASE46_CHAIN_SUBMITTED run_root=${RUN_ROOT} jobs=${J1},${J2},${J3}"

if [[ "${PHASE46_FOLLOW:-1}" == "1" ]]; then
  exec "${PROJECT}/scripts/grpo/phase46_autopilot.py" \
    --manifest "${MANIFEST}" \
    --follow \
    --interval 300 \
    --log-root "${PROJECT}" \
    --log-root "${RLINF}/logs/slurm" \
    --log-root "${RLINF}"
fi
