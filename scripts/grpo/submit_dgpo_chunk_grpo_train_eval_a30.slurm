#!/usr/bin/env bash
#SBATCH --job-name=dgpo-chunk-s30
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=dgpo_chunk_grpo_train_%j.out
#SBATCH --error=dgpo_chunk_grpo_train_%j.err
#SBATCH --export=NIL

set -euo pipefail

_PROJECT_FALLBACK="/vol/bitbucket/aa6622/project"
if [[ -f "${SLURM_SUBMIT_DIR:-}/scripts/slurm/common_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
elif [[ -f "${_PROJECT_FALLBACK}/scripts/slurm/common_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${_PROJECT_FALLBACK}/scripts/slurm/common_env.sh"
else
  echo "error: missing scripts/slurm/common_env.sh" >&2
  exit 2
fi

slurm_resolve_project_root "scripts/grpo/train_phase11_env_on_policy_grpo.py"
cd "${PROJECT_ROOT}"
slurm_export_hf_torch_cache "dgpo-chunk-sparse30"
slurm_export_pythonpath

RLINF_ROOT="${RLINF_ROOT:-/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo}"
# shellcheck source=/dev/null
source "${RLINF_ROOT}/scripts/slurm/rlinf_smolvla_common.sh"
setup_rlinf_smolvla_env
require_python_bin
EVAL_PYTHON_BIN="${PYTHON_BIN}"

GRPO_PYTHON_BIN="${GRPO_PYTHON_BIN:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
MODEL_PATH="${MODEL_PATH:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
BASE_CKPT="${1:-${MODEL_PATH}}"
OUT_ROOT="${2:-${PROJECT_ROOT}/artifacts/dgpo_chunk_grpo_sparse30}"
RUN_LABEL="${3:-dgpo_chunk_grpo_sparse30_seed1000}"

TRAIN_OUT="${OUT_ROOT}/train_${SLURM_JOB_ID}"
EVAL25_OUT="${OUT_ROOT}/eval25_${SLURM_JOB_ID}"
EVAL100_OUT="${OUT_ROOT}/eval100_${SLURM_JOB_ID}"
mkdir -p "${TRAIN_OUT}" "${EVAL25_OUT}" "${EVAL100_OUT}"

echo "[dgpo-chunk-s30] base_ckpt=${BASE_CKPT} train_out=${TRAIN_OUT}"
cd "${PROJECT_ROOT}"
"${GRPO_PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/grpo/train_phase11_env_on_policy_grpo.py" \
  --checkpoint "${BASE_CKPT}" \
  --output-dir "${TRAIN_OUT}" \
  --run-label "${RUN_LABEL}" \
  --task push-v3 \
  --env-backend official_lerobot \
  --num-updates 30 \
  --batch-size 1 \
  --group-size 16 \
  --train-seed-base 2000 \
  --lr 7.5e-6 \
  --clip-eps 0.2 \
  --max-steps 120 \
  --reward-mode sparse_success_delta \
  --rollout-unit chunk \
  --rollout-chunk-len 5 \
  --rollout-execution vector_async \
  --async-start-method forkserver \
  --logprob-mode flow_sde \
  --flow-sde-trace-step 9 \
  --flow-sde-noise-level 1.0 \
  --euler-step-noise-std 0.0 \
  --action-transform no_tanh \
  --save-every 5 \
  --fail-on-parity-violation \
  --dgpo \
  --dgpo-tau "${DGPO_TAU:-0.5}" \
  --dgpo-kappa "${DGPO_KAPPA:-0.0}" \
  --dgpo-ref "${DGPO_REF:-frozen_sft}"

CKPT_DIR="${TRAIN_OUT}/checkpoints"
if [[ ! -f "${CKPT_DIR}/update_0030.pt" ]]; then
  echo "error: expected update_0030.pt not found in ${CKPT_DIR}" >&2
  exit 20
fi

cd "${RLINF_ROOT}"
"${EVAL_PYTHON_BIN}" -u scripts/eval_smolvla_metaworld_ckpt_sweep.py \
  --run-name "${RUN_LABEL}_eval25" \
  --output-dir "${EVAL25_OUT}/eval_25ep" \
  --num-episodes 25 \
  --include-baseline \
  --checkpoint-dir "${CKPT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --task-name push-v3 \
  --task-description "Push the puck to a goal" \
  --num-envs 25 \
  --seed-base 1000 \
  --max-episode-steps 150 \
  --chunk-len 5

"${EVAL_PYTHON_BIN}" -u scripts/eval_smolvla_metaworld_ckpt_sweep.py \
  --run-name "${RUN_LABEL}_eval100" \
  --output-dir "${EVAL100_OUT}/eval_100ep" \
  --num-episodes 100 \
  --include-baseline \
  --checkpoint-dir "${CKPT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --task-name push-v3 \
  --task-description "Push the puck to a goal" \
  --num-envs 25 \
  --seed-base 1000 \
  --max-episode-steps 150 \
  --chunk-len 5 \
  --only-updates "10,20,30"

echo "DGPO_CHUNK_SPARSE30_CHAIN_OK train=${TRAIN_OUT} eval25=${EVAL25_OUT} eval100=${EVAL100_OUT}"
