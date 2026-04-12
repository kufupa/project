#!/bin/bash
#SBATCH --job-name=seggrpo-c1003-ep3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/vol/bitbucket/aa6622/project/logs/seggrpo_campaign1003_ep3_%j.out
#SBATCH --error=/vol/bitbucket/aa6622/project/logs/seggrpo_campaign1003_ep3_%j.err
#
# Real inference: SmolVLA + JEPA-WM + MetaWorld sim (no --dry-run).
# Matches phase07 campaign folder 1003: oracle_episode_index=3, oracle_reset_seed=1003
# (see .../pushv3_smolvla_topk15_*/1003/best_episode.json).
#
# Submit:
#   export SMOLVLA_CHECKPOINT=...   # optional; default HF id below
#   export JEPA_REPO=...            # optional; default torch.hub cache
#   sbatch /vol/bitbucket/aa6622/project/scripts/segment_grpo/submit_segment_grpo_campaign1003.slurm
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

echo "[seggrpo-c1003] SMOLVLA_CHECKPOINT=${SMOLVLA_CHECKPOINT}"
echo "[seggrpo-c1003] JEPA_REPO=${JEPA_REPO}"
echo "[seggrpo-c1003] episode-index=3 reset-seed=1003 (campaign 1003) carry_mode=sim device=cuda"

"${PY}" scripts/run_segment_grpo.py \
  --checkpoint "${SMOLVLA_CHECKPOINT}" \
  --jepa-repo "${JEPA_REPO}" \
  --output-json "${PROJECT_ROOT}/artifacts/segment_grpo_campaign1003_ep3_s1003.json" \
  --episodes 1 \
  --episode-index 3 \
  --reset-seed 1003 \
  --goal-frame-index 25 \
  --chunk-len 8 \
  --num-candidates 2 \
  --max-steps 120 \
  --carry-mode sim \
  --wm-rollout-mode iterative \
  --device cuda \
  --wm-sim-camera-parity \
  --wm-sim-img-size 224 \
  --wm-goal-hflip \
  --smolvla-policy-hflip-corner2 \
  --smolvla-noise-std 0.0

echo "[seggrpo-c1003] finished OK"
