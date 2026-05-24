#!/usr/bin/env bash
set -euo pipefail

cd /rds/general/user/aa6622/home/project
mkdir -p logs/pbs/grpo

TRAIN_PBS="scripts/grpo/phase11_seedbatch_train_param.pbs"
EVAL_PBS="scripts/grpo/phase11_seedbatch_eval25_param.pbs"
if [[ ! -f "${TRAIN_PBS}" || ! -f "${EVAL_PBS}" ]]; then
  echo "error: missing PBS templates" >&2
  exit 2
fi

submit_pair() {
  local name="$1"
  local run_dir="$2"
  local batch_size="$3"
  local group_size="$4"
  local updates="$5"
  local lr="$6"
  local clip="$7"
  local log_std="$8"
  local euler="$9"
  local train_wall="${10}"
  local train_mem="${11}"
  local eval_wall="${12}"

  local label="${name}"
  local sweep="eval_sweep_0002_$(printf '%04d' "${updates}")_25ep_nenv25_async"
  local qsub_vars
  qsub_vars="PHASE11_RUN_DIR=${run_dir},PHASE11_RUN_LABEL=${label},PHASE11_BATCH_SIZE=${batch_size},PHASE11_GROUP_SIZE=${group_size},PHASE11_NUM_UPDATES=${updates},PHASE11_SAVE_EVERY=2,PHASE11_LR=${lr},PHASE11_CLIP_EPS=${clip},PHASE11_INIT_LOG_STD=${log_std},PHASE11_EULER_NOISE=${euler},PHASE11_ROLLOUT_POLICY_BATCH_SIZE=16,PHASE11_LOGPROB_BATCH_SIZE=16"

  echo "[submit] train ${name} run_dir=${run_dir}"
  local train_id
  train_id=$(qsub \
    -N "${name:0:15}" \
    -l "select=1:ncpus=48:mem=${train_mem}:ngpus=1:gpu_type=RTX6000" \
    -l "walltime=${train_wall}" \
    -o "logs/pbs/grpo/${name}.train.out" \
    -v "${qsub_vars}" \
    "${TRAIN_PBS}")

  local eval_vars
  eval_vars="PHASE11_RUN_DIR=${run_dir},PHASE11_SWEEP_NAME=${sweep},PHASE11_MIN_UPDATE=2,PHASE11_MAX_UPDATE=${updates},PHASE11_EVAL_EPISODES=25"
  echo "[submit] eval ${name} afterok=${train_id}"
  local eval_id
  eval_id=$(qsub \
    -N "${name:0:12}ev" \
    -l "walltime=${eval_wall}" \
    -o "logs/pbs/grpo/${name}.eval25.out" \
    -W "depend=afterok:${train_id}" \
    -v "${eval_vars}" \
    "${EVAL_PBS}")

  printf '%s train=%s eval=%s run_dir=%s sweep=%s\n' "${name}" "${train_id}" "${eval_id}" "${run_dir}" "${sweep}"
}

submit_pair \
  "p11b16g4" \
  "artifacts/phase11_pushv3_seedbatch_b16_g4_lr3e7_clip005_u50" \
  16 4 50 3e-7 0.05 -2.0 0.2 "06:00:00" "64gb" "01:40:00"

submit_pair \
  "p11b8g8" \
  "artifacts/phase11_pushv3_seedbatch_b8_g8_lr6e7_clip005_u50" \
  8 8 50 6e-7 0.05 -2.0 0.2 "06:00:00" "64gb" "01:40:00"

submit_pair \
  "p11b4g16" \
  "artifacts/phase11_pushv3_seedbatch_b4_g16_lr125e6_clip005_u50" \
  4 16 50 1.25e-6 0.05 -2.0 0.2 "06:00:00" "64gb" "01:40:00"

submit_pair \
  "p11b4g32" \
  "artifacts/phase11_pushv3_seedbatch_b4_g32_lr125e6_clip01_u30" \
  4 32 30 1.25e-6 0.1 -2.0 0.2 "08:00:00" "128gb" "01:00:00"

submit_pair \
  "p11b2g32ln" \
  "artifacts/phase11_pushv3_seedbatch_b2_g32_lr25e7_clip01_lownoise_u60" \
  2 32 60 2.5e-6 0.1 -2.5 0.1 "08:00:00" "96gb" "02:00:00"
