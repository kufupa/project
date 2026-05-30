#!/usr/bin/env bash
# Submit from login node (not inside a running job): smoke -> G8 train -> 25ep eval sweep.
set -euo pipefail

PROJECT_ROOT="/vol/bitbucket/aa6622/project"
cd "${PROJECT_ROOT}"

CHECKPOINT="${1:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
SMOKE_OUT="${PROJECT_ROOT}/artifacts/phase111_grpo_correctness_smoke_${STAMP}"
TRAIN_OUT="${PROJECT_ROOT}/artifacts/phase111_grpo_correctness_g8_${STAMP}"

for f in scripts/grpo/submit_phase111_correctness_smoke.slurm \
  scripts/grpo/submit_phase111_correctness_g8_train.slurm \
  scripts/grpo/submit_phase111_eval_sweep.slurm; do
  bash -n "${f}"
done

sbatch --test-only --chdir="${PROJECT_ROOT}" --export=NIL \
  scripts/grpo/submit_phase111_correctness_smoke.slurm "${CHECKPOINT}" "${SMOKE_OUT}"

SMOKE_JID="$(sbatch --parsable --chdir="${PROJECT_ROOT}" --export=NIL \
  --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 \
  scripts/grpo/submit_phase111_correctness_smoke.slurm "${CHECKPOINT}" "${SMOKE_OUT}")"
echo "submitted smoke job ${SMOKE_JID} -> ${SMOKE_OUT}"

TRAIN_JID="$(sbatch --parsable --chdir="${PROJECT_ROOT}" --export=NIL \
  --dependency=afterok:${SMOKE_JID} \
  --gres=gpu:1 --cpus-per-task=12 --mem=48G --time=24:00:00 \
  scripts/grpo/submit_phase111_correctness_g8_train.slurm "${CHECKPOINT}" "${TRAIN_OUT}" 15 0)"
echo "submitted train job ${TRAIN_JID} -> ${TRAIN_OUT}"

EVAL_JID="$(sbatch --parsable --chdir="${PROJECT_ROOT}" --export=NIL \
  --dependency=afterok:${TRAIN_JID} \
  --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=12:00:00 \
  scripts/grpo/submit_phase111_eval_sweep.slurm \
  "${TRAIN_OUT}" "${CHECKPOINT}" push-v3 25 1000 0 0 \
  eval_correctness_25ep 2 14)"
echo "submitted eval job ${EVAL_JID}"

echo "CHAIN_OK smoke=${SMOKE_JID} train=${TRAIN_JID} eval=${EVAL_JID}"
