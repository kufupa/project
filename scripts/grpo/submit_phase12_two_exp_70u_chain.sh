#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

python3 "$HOME/.agents/skills/checking-pbs-gpu-availability/scripts/pbs_gpu_snapshot.py"

BASE_OUT="${PHASE12_BASE_OUT:-${PROJECT_ROOT}/artifacts/phase12_wm_only_overnight_70u}"
mkdir -p "${BASE_OUT}" logs/pbs/grpo

qsub_job() {
  local output
  output="$("$@")"
  echo "${output}" | awk '{print $1}'
}

require_job_visible() {
  local job_id="$1"
  qstat "${job_id}" >/dev/null 2>&1 || {
    echo "error: submitted job not visible in qstat: ${job_id}" >&2
    exit 3
  }
}

submit_eval_for_chunk() {
  local train_job="$1"
  local run_dir="$2"
  local label="$3"
  local min_update="$4"
  local max_update="$5"
  local eval_job
  eval_job="$(
    qsub -W "depend=afterok:${train_job}" \
      -v "PHASE12_RUN_DIR=${run_dir},PHASE12_MIN_UPDATE=${min_update},PHASE12_MAX_UPDATE=${max_update},PHASE12_STRIDE=2,PHASE12_EVAL_EPISODES=25,PHASE12_EVAL_N_ENVS=25,PHASE12_SWEEP_NAME=eval_last5_${min_update}_${max_update}_25ep_nenv25" \
      scripts/grpo/phase12_eval_last5_25ep.pbs
  )"
  eval_job="$(echo "${eval_job}" | awk '{print $1}')"
  echo "eval ${label} ${min_update}-${max_update}: ${eval_job}"
  require_job_visible "${eval_job}" || true
}

submit_chain() {
  local label="$1"
  local action_profile="$2"
  local batch_size="$3"
  local group_size="$4"
  local lr="$5"
  local clip_eps="$6"
  local noise="$7"
  local wm_mode="$8"
  local wm_bs="$9"
  local initial_dep="${10:-}"
  local run_dir="${BASE_OUT}/${label}"
  mkdir -p "${run_dir}"

  local prev_train=""
  local start end resume train_job min_eval max_eval
  for chunk in 0 1 2 3 4 5 6; do
    start=$((chunk * 10))
    end=$((start + 10))
    if [[ "${chunk}" -eq 0 ]]; then
      resume=""
    else
      resume="${run_dir}/checkpoints/update_$(printf '%04d' "${start}").pt"
    fi
    local varlist="PHASE12_RUN_DIR=${run_dir},PHASE12_START_UPDATE=${start},PHASE12_NUM_UPDATES=10,PHASE12_SAVE_EVERY=2,PHASE12_BATCH_SIZE=${batch_size},PHASE12_GROUP_SIZE=${group_size},PHASE12_ACTION_PROFILE=${action_profile},PHASE12_LR=${lr},PHASE12_CLIP_EPS=${clip_eps},PHASE12_EULER_NOISE=${noise},PHASE12_WM_SCORE_MODE=${wm_mode},PHASE12_WM_SCORE_BATCH_SIZE=${wm_bs}"
    if [[ -n "${resume}" ]]; then
      varlist="${varlist},PHASE12_RESUME=${resume}"
    fi
    if [[ -n "${prev_train}" ]]; then
      train_job="$(
        qsub -W "depend=afterok:${prev_train}" -v "${varlist}" scripts/grpo/phase12_train_chunk_10u.pbs
      )"
    elif [[ -n "${initial_dep}" ]]; then
      train_job="$(
        qsub -W "depend=afterok:${initial_dep}" -v "${varlist}" scripts/grpo/phase12_train_chunk_10u.pbs
      )"
    else
      train_job="$(
        qsub -v "${varlist}" scripts/grpo/phase12_train_chunk_10u.pbs
      )"
    fi
    train_job="$(echo "${train_job}" | awk '{print $1}')"
    echo "train ${label} ${start}-${end}: ${train_job}"
    require_job_visible "${train_job}" || true
    min_eval=$((start + 2))
    max_eval="${end}"
    submit_eval_for_chunk "${train_job}" "${run_dir}" "${label}" "${min_eval}" "${max_eval}"
    prev_train="${train_job}"
  done
}

echo "[phase12-chain] submit smoke"
smoke_job="$(
  qsub -v "PHASE12_RUN_DIR=${BASE_OUT}/smoke_b2_g8,PHASE12_START_UPDATE=0,PHASE12_NUM_UPDATES=2,PHASE12_SAVE_EVERY=1,PHASE12_BATCH_SIZE=2,PHASE12_GROUP_SIZE=8,PHASE12_ACTION_PROFILE=official_jepa_mirror,PHASE12_LR=1e-5,PHASE12_CLIP_EPS=0.2,PHASE12_EULER_NOISE=0.2,PHASE12_WM_SCORE_MODE=serial" scripts/grpo/phase12_train_chunk_10u.pbs
)"
smoke_job="$(echo "${smoke_job}" | awk '{print $1}')"
echo "smoke: ${smoke_job}"
require_job_visible "${smoke_job}" || true

echo "[phase12-chain] submit conservative chains after smoke=${smoke_job}"
submit_chain "official_g8_lr1e5" official_jepa_mirror 4 8 1e-5 0.2 0.2 serial 8 "${smoke_job}"
submit_chain "bounded_g8_lr1e5" bounded_executed 4 8 1e-5 0.2 0.2 serial 8 "${smoke_job}"
if [[ "${PHASE12_ENABLE_MAXPERF:-0}" == "1" ]]; then
  submit_chain "maxperf_b4_g16_lr5e6_clip01_lownoise" official_jepa_mirror 4 16 5e-6 0.1 0.1 batched 8 "${smoke_job}"
fi

echo "PHASE12_CHAIN_BOOTSTRAP_SUBMITTED smoke=${smoke_job} base_out=${BASE_OUT}"
