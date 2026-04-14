#!/bin/bash
#SBATCH --job-name=seggrpo-c1003-c20-olay-g50-220
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/vol/bitbucket/aa6622/project/logs/seggrpo_campaign1003_chunk20_overlaydist_g50_220_%j.out
#SBATCH --error=/vol/bitbucket/aa6622/project/logs/seggrpo_campaign1003_chunk20_overlaydist_g50_220_%j.err
#
# Same run as submit_segment_grpo_campaign1003_chunk20_overlaydist.slurm but with
# goal-frame-index=50 and max-steps=220.
#
# Submit:
#   export SMOLVLA_CHECKPOINT=...
#   export JEPA_REPO=...
#   sbatch /vol/bitbucket/aa6622/project/scripts/segment_grpo/submit_segment_grpo_campaign1003_chunk20_overlaydist_g50_max220.slurm
#
set -euo pipefail
PROJECT_ROOT="/vol/bitbucket/aa6622/project"
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"
PY="${SMOLVLA_LEROBOT_ENV_DIR}/bin/python"

export SMOLVLA_CHECKPOINT="${SMOLVLA_CHECKPOINT:-jadechoghari/smolvla_metaworld}"
export JEPA_REPO="${JEPA_REPO:-${HOME}/.cache/torch/hub/facebookresearch_jepa-wms_main}"

echo "[seggrpo-c1003-c20-olay-g50-220] SMOLVLA_CHECKPOINT=${SMOLVLA_CHECKPOINT}"
echo "[seggrpo-c1003-c20-olay-g50-220] JEPA_REPO=${JEPA_REPO}"
echo "[seggrpo-c1003-c20-olay-g50-220] goal-frame-index=50 max-steps=220"
echo "[seggrpo-c1003-c20-olay-g50-220] output-json -> artifacts/segment_grpo_campaign1003_ep3_s1003_chunk20_overlaydist_goal50_max220.json"

"${PY}" scripts/run_segment_grpo.py \
  --checkpoint "${SMOLVLA_CHECKPOINT}" \
  --jepa-repo "${JEPA_REPO}" \
  --output-json "${PROJECT_ROOT}/artifacts/segment_grpo_campaign1003_ep3_s1003_chunk20_overlaydist_goal50_max220.json" \
  --episodes 1 \
  --episode-index 3 \
  --reset-seed 1003 \
  --goal-frame-index 50 \
  --chunk-len 20 \
  --smolvla-n-action-steps 20 \
  --num-candidates 2 \
  --max-steps 220 \
  --carry-mode sim \
  --wm-rollout-mode iterative \
  --device cuda \
  --wm-sim-camera-parity \
  --wm-sim-img-size 224 \
  --wm-goal-hflip \
  --smolvla-policy-hflip-corner2 \
  --smolvla-noise-std 0.0 \
  --comparison-strip-overlay

echo "[seggrpo-c1003-c20-olay-g50-220] finished OK"
