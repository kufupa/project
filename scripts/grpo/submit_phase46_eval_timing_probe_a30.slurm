#!/usr/bin/env bash
#SBATCH --job-name=ph46-eval-timing
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=phase46_eval_timing_probe_%j.out
#SBATCH --error=phase46_eval_timing_probe_%j.err
#SBATCH --export=NIL

set -euo pipefail

PROJECT_ROOT="/vol/bitbucket/aa6622/project"
RLINF_ROOT="${RLINF_ROOT:-/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo}"
PYTHON_BIN="${PYTHON_BIN:-/vol/bitbucket/aa6622/.envs/rlinf_smolvla_mw_py312_fresh_nodeps/bin/python}"
CHECKPOINT="${CHECKPOINT:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/artifacts/phase46_logprob_ablation/20260601_164622_logprob_ablation/unsquashed/train/checkpoints}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/artifacts/phase46_eval_timing_probe/${SLURM_JOB_ID:-local}}"

cd "${RLINF_ROOT}"
source "${RLINF_ROOT}/scripts/slurm/rlinf_smolvla_common.sh"
setup_rlinf_smolvla_env
require_python_bin

mkdir -p "${OUT_DIR}"

echo "PHASE46_EVAL_TIMING_PROBE_BEGIN host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "PHASE46_EVAL_TIMING_PROBE_CKPT_DIR ${CKPT_DIR}"
echo "PHASE46_EVAL_TIMING_PROBE_OUT_DIR ${OUT_DIR}"

"${PYTHON_BIN}" -u scripts/eval_smolvla_metaworld_ckpt_sweep.py \
  --run-name "phase46_timing_probe_25ep" \
  --checkpoint-dir "${CKPT_DIR}" \
  --output-dir "${OUT_DIR}/eval_25ep_u1_u2" \
  --model-path "${CHECKPOINT}" \
  --task-name push-v3 \
  --task-description "Push the puck to a goal" \
  --num-envs 25 \
  --num-episodes 25 \
  --seed-base 1000 \
  --max-episode-steps 150 \
  --chunk-len 5 \
  --include-baseline \
  --only-updates "1,2"

echo "PHASE46_EVAL_TIMING_PROBE_OK out=${OUT_DIR}"
