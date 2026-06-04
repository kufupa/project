#!/usr/bin/env bash
set -euo pipefail

cd /rds/general/user/aa6622/home/project
mkdir -p logs/pbs/grpo

TRAIN_PBS="scripts/grpo/phase11_seedbatch_train_param.pbs"
EVAL_PBS="scripts/grpo/phase11_seedbatch_eval25_param.pbs"
EPHEMERAL_ROOT="${PHASE11_EPHEMERAL_ROOT:-/rds/general/user/aa6622/ephemeral/phase11_grpo_20260519_reward_chains}"

if [[ ! -f "${TRAIN_PBS}" || ! -f "${EVAL_PBS}" ]]; then
  echo "error: missing PBS templates" >&2
  exit 2
fi
mkdir -p "${EPHEMERAL_ROOT}"

submit_chain() {
  local name="$1"
  local lr="$2"
  local clip="$3"
  local success_bonus="$4"
  local clip_penalty="$5"

  local run_dir="${EPHEMERAL_ROOT}/${name}"
  local prev_train=""
  local train_ids=()
  local eval_ids=()
  mkdir -p "${run_dir}"

  for chunk_idx in 0 1 2 3 4; do
    local start=$((chunk_idx * 10))
    local end=$(((chunk_idx + 1) * 10))
    local min_eval=$((start + 2))
    local max_eval="${end}"
    local num_updates=10
    local resume=""
    local train_vars
    local train_dep=()
    local train_name="${name}_t${start}_${end}"

    if [[ "${chunk_idx}" -gt 0 ]]; then
      resume="${run_dir}/checkpoints/latest.pt"
    fi
    if [[ -n "${prev_train}" ]]; then
      train_dep=(-W "depend=afterok:${prev_train}")
    fi

    train_vars="PHASE11_RUN_DIR=${run_dir},PHASE11_RUN_LABEL=${name},PHASE11_BATCH_SIZE=4,PHASE11_GROUP_SIZE=16,PHASE11_NUM_UPDATES=${num_updates},PHASE11_SAVE_EVERY=2,PHASE11_LR=${lr},PHASE11_CLIP_EPS=${clip},PHASE11_INIT_LOG_STD=-2.5,PHASE11_EULER_NOISE=0.1,PHASE11_SUCCESS_BONUS=${success_bonus},PHASE11_CLIP_PENALTY=${clip_penalty},PHASE11_ROLLOUT_POLICY_BATCH_SIZE=16,PHASE11_LOGPROB_BATCH_SIZE=16"
    if [[ -n "${resume}" ]]; then
      train_vars="${train_vars},PHASE11_RESUME=${resume},PHASE11_START_UPDATE=${start}"
    else
      train_vars="${train_vars},PHASE11_START_UPDATE=0"
    fi

    echo "[submit] train ${train_name} run_dir=${run_dir}"
    local train_id
    train_id=$(qsub \
      -N "${train_name:0:15}" \
      -l "select=1:ncpus=48:mem=64gb:ngpus=1:gpu_type=RTX6000" \
      -l "walltime=01:45:00" \
      -o "logs/pbs/grpo/${train_name}.out" \
      -v "${train_vars}" \
      "${train_dep[@]}" \
      "${TRAIN_PBS}")
    train_ids+=("${train_id}")
    prev_train="${train_id}"

    local sweep="eval_sweep_$(printf '%04d' "${min_eval}")_$(printf '%04d' "${max_eval}")_25ep_nenv25_async"
    local eval_vars="PHASE11_RUN_DIR=${run_dir},PHASE11_SWEEP_NAME=${sweep},PHASE11_MIN_UPDATE=${min_eval},PHASE11_MAX_UPDATE=${max_eval},PHASE11_EVAL_EPISODES=25"
    local eval_name="${name}_e${min_eval}_${max_eval}"
    echo "[submit] eval ${eval_name} afterok=${train_id}"
    local eval_id
    eval_id=$(qsub \
      -N "${eval_name:0:15}" \
      -l "select=1:ncpus=32:mem=32gb:ngpus=1:gpu_type=RTX6000" \
      -l "walltime=00:40:00" \
      -o "logs/pbs/grpo/${eval_name}.out" \
      -W "depend=afterok:${train_id}" \
      -v "${eval_vars}" \
      "${EVAL_PBS}")
    eval_ids+=("${eval_id}")
  done

  printf '%s run_dir=%s\n' "${name}" "${run_dir}"
  printf '%s train_ids=%s\n' "${name}" "$(IFS=,; echo "${train_ids[*]}")"
  printf '%s eval_ids=%s\n' "${name}" "$(IFS=,; echo "${eval_ids[*]}")"
}

submit_chain "p11b4g16_ln_dense" "1.25e-6" "0.05" "0.0" "0.0"
submit_chain "p11b4g16_ln_succ" "1.25e-6" "0.05" "50.0" "0.0"
submit_chain "p11b4g16_ln_succ_clip" "1.25e-6" "0.05" "50.0" "2.5"
