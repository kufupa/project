#!/bin/bash
#SBATCH --job-name=seggrpo-c1003-c30m30k5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/vol/bitbucket/aa6622/project/logs/seggrpo_campaign1003_chunk30_max30_k5_%j.out
#SBATCH --error=/vol/bitbucket/aa6622/project/logs/seggrpo_campaign1003_chunk30_max30_k5_%j.err
#
# Campaign 1003: one segment (max_steps=30), chunk_len=30, SmolVLA horizon 30, K=5 candidates,
# goal_frame_index=25. WM f=5 => 6 decode columns; PNG footer = all candidates' int L2(goal) per WM step.
#
# Submit:
#   export SMOLVLA_CHECKPOINT=...
#   export JEPA_REPO=...
#   sbatch /vol/bitbucket/aa6622/project/scripts/segment_grpo/submit_segment_grpo_campaign1003_chunk30_max30_k5.slurm
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

echo "[seggrpo-c1003-c30m30k5] SMOLVLA_CHECKPOINT=${SMOLVLA_CHECKPOINT}"
echo "[seggrpo-c1003-c30m30k5] JEPA_REPO=${JEPA_REPO}"
echo "[seggrpo-c1003-c30m30k5] chunk-len=30 smolvla-n-action-steps=30 max-steps=30 num-candidates=5 goal-frame-index=25 episode-index=3 reset-seed=1003"

# Fail fast if this checkout is not the footer+L2-all-candidates build (avoids silent old PNGs).
echo "[seggrpo-c1003-c30m30k5] checking segment_grpo_loop.py resolves to project src and exports _all_candidates_wm_goal_l2_rows"
"${PY}" -c "import sys; p='${PROJECT_ROOT}/src'; sys.path.insert(0, p); import segment_grpo_loop as m; assert hasattr(m, '_all_candidates_wm_goal_l2_rows'), 'stale segment_grpo_loop'; print('[seggrpo-c1003-c30m30k5] OK', m.__file__)"

"${PY}" scripts/run_segment_grpo.py \
  --checkpoint "${SMOLVLA_CHECKPOINT}" \
  --jepa-repo "${JEPA_REPO}" \
  --output-json "${PROJECT_ROOT}/artifacts/segment_grpo_campaign1003_ep3_s1003_chunk30_max30_k5.json" \
  --episodes 1 \
  --episode-index 3 \
  --reset-seed 1003 \
  --goal-frame-index 25 \
  --chunk-len 30 \
  --smolvla-n-action-steps 30 \
  --num-candidates 5 \
  --max-steps 30 \
  --carry-mode sim \
  --wm-rollout-mode iterative \
  --device cuda \
  --wm-sim-camera-parity \
  --wm-sim-img-size 224 \
  --wm-goal-hflip \
  --smolvla-policy-hflip-corner2 \
  --smolvla-noise-std 0.0 \
  --comparison-strip-overlay

echo "[seggrpo-c1003-c30m30k5] finished OK"
