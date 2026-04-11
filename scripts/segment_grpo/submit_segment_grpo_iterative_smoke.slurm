#!/bin/bash
#SBATCH --job-name=seggrpo-iter-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/vol/bitbucket/aa6622/project/logs/seggrpo_iter_smoke_%j.out
#
# One-episode real inference smoke (SmolVLA + JEPA-WM) with iterative WM rollout.
# Submit from repo login node:
#   export SMOLVLA_CHECKPOINT=...
#   export JEPA_REPO=...
#   sbatch /vol/bitbucket/aa6622/project/scripts/segment_grpo/submit_segment_grpo_iterative_smoke.slurm
#
set -euo pipefail
PROJECT_ROOT="/vol/bitbucket/aa6622/project"
cd "${PROJECT_ROOT}"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"
PY="${SMOLVLA_LEROBOT_ENV_DIR}/bin/python"

export SMOLVLA_CHECKPOINT="${SMOLVLA_CHECKPOINT:-jadechoghari/smolvla_metaworld}"
export JEPA_REPO="${JEPA_REPO:-${HOME}/.cache/torch/hub/facebookresearch_jepa-wms_main}"

"${PY}" scripts/run_segment_grpo.py \
  --checkpoint "${SMOLVLA_CHECKPOINT}" \
  --jepa-repo "${JEPA_REPO}" \
  --output-json "${PROJECT_ROOT}/artifacts/segment_grpo_iterative_gpu_smoke.json" \
  --episodes 1 \
  --episode-index 3 \
  --reset-seed 1003 \
  --chunk-len 8 \
  --num-candidates 2 \
  --max-steps 120 \
  --wm-rollout-mode iterative \
  --device cuda
