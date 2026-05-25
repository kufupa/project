#!/usr/bin/env bash
# Profile small RLinf ManiSkill shapes on CX3 RTX6000 nodes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=pirl_maniskill_common.sh
source "${SCRIPT_DIR}/pirl_maniskill_common.sh"

pirl_setup_modules
pirl_prepare_runtime

PROFILE_ROOT="${PIRL_ARTIFACT_ROOT}/profile/${PBS_JOBID:-local}"
mkdir -p "${PROFILE_ROOT}"

echo "[pirl-profile] Jupyter GPU queue excluded; use v1_gpu72 PBS RTX6000 nodes."
echo "[pirl-profile] output=${PROFILE_ROOT}"

env_shapes=(${PIRL_PROFILE_ENVS:-16 32 48 64 80 96 128 160 192 224 256 320})
for env_count in "${env_shapes[@]}"; do
  run_name="profile_env${env_count}"
  rollout_epoch="${PIRL_PROFILE_ROLLOUT_EPOCH:-}"
  max_steps="${PIRL_PROFILE_MAX_STEPS:-}"
  global_batch="${PIRL_PROFILE_GLOBAL_BATCH:-}"
  shape_note="small smoke fallback; override PIRL_PROFILE_ROLLOUT_EPOCH/PIRL_PROFILE_MAX_STEPS/PIRL_PROFILE_GLOBAL_BATCH to test another shape"

  if [[ -z "${rollout_epoch}" || -z "${max_steps}" || -z "${global_batch}" ]]; then
    case "${env_count}" in
      64)
        rollout_epoch="${rollout_epoch:-5}"
        max_steps="${max_steps:-80}"
        global_batch="${global_batch:-5120}"
        shape_note="planned 5120 shape"
        ;;
      80)
        rollout_epoch="${rollout_epoch:-4}"
        max_steps="${max_steps:-80}"
        global_batch="${global_batch:-5120}"
        shape_note="planned 5120 shape"
        ;;
      128)
        rollout_epoch="${rollout_epoch:-5}"
        max_steps="${max_steps:-40}"
        global_batch="${global_batch:-5120}"
        shape_note="planned 5120 shape"
        ;;
      160)
        rollout_epoch="${rollout_epoch:-2}"
        max_steps="${max_steps:-80}"
        global_batch="${global_batch:-5120}"
        shape_note="planned 5120 shape"
        ;;
      320)
        rollout_epoch="${rollout_epoch:-1}"
        max_steps="${max_steps:-80}"
        global_batch="${global_batch:-5120}"
        shape_note="planned 5120 shape"
        ;;
      *)
        rollout_epoch="${rollout_epoch:-1}"
        max_steps="${max_steps:-20}"
        global_batch="${global_batch:-$((env_count * 4))}"
        ;;
    esac
  fi

  echo "[pirl-profile] run=${run_name} envs=${env_count} rollout_epoch=${rollout_epoch} max_steps=${max_steps} global_batch=${global_batch} note=${shape_note}"
  set +e
  pirl_run_rlinf "${run_name}" \
    "runner.max_epochs=1" \
    "runner.val_check_interval=-1" \
    "runner.save_interval=-1" \
    "runner.logger.log_path=${PROFILE_ROOT}/${run_name}" \
    "runner.logger.experiment_name=${run_name}" \
    "env.train.total_num_envs=${env_count}" \
    "env.eval.total_num_envs=${env_count}" \
    "env.train.max_steps_per_rollout_epoch=${max_steps}" \
    "env.eval.max_steps_per_rollout_epoch=${max_steps}" \
    "env.eval.video_cfg.save_video=False" \
    "algorithm.rollout_epoch=${rollout_epoch}" \
    "actor.micro_batch_size=${PIRL_PROFILE_MICRO_BATCH:-1}" \
    "actor.global_batch_size=${global_batch}" \
    "actor.model.model_path=${PIRL_SFT_CKPT}" \
    "rollout.model.model_path=${PIRL_SFT_CKPT}"
  profile_status=$?
  set -e
  echo "envs=${env_count} status=${profile_status} rollout_epoch=${rollout_epoch} max_steps=${max_steps} global_batch=${global_batch}" \
    | tee -a "${PROFILE_ROOT}/profile_status.tsv"
  if [[ "${profile_status}" -ne 0 ]]; then
    echo "[pirl-profile] trial failed/OOM-like; continuing to preserve last-pass/first-fail evidence."
    continue
  fi
done

echo "PIRL_MANISKILL_PROFILE_SWEEP_DONE root=${PROFILE_ROOT}"
