#!/usr/bin/env bash
# Run one top-15 target (ranks 2–15): same CLI as submit_segment_grpo_campaign1003_chunk30_max30_k5.slurm.
# Usage: segment_grpo_topk15_chunk30_max30_k5_run_task.sh TASK_ID
#   TASK_ID in 0..13 maps to (episode_index, reset_seed) from targets.json ranks 2–15.
#
# Oracle goal: --goal-frame-index 25 is always the 25th frame (1-based) PNG for *this* row's
# episode_index under the resolved oracle baseline (see load_oracle_reference_frames:
# frames/episode_{episode_index}/frame_000024.png), not episode 3's goal.
set -euo pipefail

TASK_ID="${1:-}"
if [[ -z "${TASK_ID}" ]] || ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 TASK_ID   (integer 0..13)" >&2
  exit 2
fi

PROJECT_ROOT="/vol/bitbucket/aa6622/project"
cd "${PROJECT_ROOT}"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"
PY="${SMOLVLA_LEROBOT_ENV_DIR}/bin/python"

export SMOLVLA_CHECKPOINT="${SMOLVLA_CHECKPOINT:-jadechoghari/smolvla_metaworld}"
export JEPA_REPO="${JEPA_REPO:-${HOME}/.cache/torch/hub/facebookresearch_jepa-wms_main}"

EPISODES=(53 16 44 39 8 58 10 42 9 59 5 55 33 28)
RESET_SEEDS=(1053 1016 1044 1039 1008 1058 1010 1042 1009 1059 1005 1055 1033 1028)

if (( TASK_ID < 0 || TASK_ID >= ${#EPISODES[@]} )); then
  echo "[seggrpo-topk15-c30m30k5] invalid TASK_ID=${TASK_ID} (need 0..$(( ${#EPISODES[@]} - 1 )))" >&2
  exit 2
fi

EPISODE_INDEX="${EPISODES[$TASK_ID]}"
RESET_SEED="${RESET_SEEDS[$TASK_ID]}"
OUTPUT_JSON="${PROJECT_ROOT}/artifacts/segment_grpo_campaign${RESET_SEED}_ep${EPISODE_INDEX}_s${RESET_SEED}_chunk30_max30_k5.json"

echo "[seggrpo-topk15-c30m30k5] SMOLVLA_CHECKPOINT=${SMOLVLA_CHECKPOINT}"
echo "[seggrpo-topk15-c30m30k5] JEPA_REPO=${JEPA_REPO}"
echo "[seggrpo-topk15-c30m30k5] task_id=${TASK_ID} chunk-len=30 smolvla-n-action-steps=30 max-steps=30 num-candidates=5 goal-frame-index=25 episode-index=${EPISODE_INDEX} reset-seed=${RESET_SEED}"

echo "[seggrpo-topk15-c30m30k5] checking segment_grpo_loop.py exports _all_candidates_wm_goal_l2_rows"
"${PY}" -c "import sys; p='${PROJECT_ROOT}/src'; sys.path.insert(0, p); import segment_grpo_loop as m; assert hasattr(m, '_all_candidates_wm_goal_l2_rows'), 'stale segment_grpo_loop'; print('[seggrpo-topk15-c30m30k5] OK', m.__file__)"

"${PY}" scripts/run_segment_grpo.py \
  --checkpoint "${SMOLVLA_CHECKPOINT}" \
  --jepa-repo "${JEPA_REPO}" \
  --output-json "${OUTPUT_JSON}" \
  --episodes 1 \
  --episode-index "${EPISODE_INDEX}" \
  --reset-seed "${RESET_SEED}" \
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

echo "[seggrpo-topk15-c30m30k5] task_id=${TASK_ID} finished OK"
