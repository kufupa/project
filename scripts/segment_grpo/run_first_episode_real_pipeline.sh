#!/usr/bin/env bash
# Real SmolVLA + JEPA-WM Segment-GRPO on a single episode: oracle index 0 by default.
#
# Requires PyTorch + CUDA, MetaWorld, checkpoint paths, and network if loading from HF.
#
# Usage (login or GPU node):
#   ./scripts/segment_grpo/run_first_episode_real_pipeline.sh
# Optional overrides:
#   SMOLVLA_CHECKPOINT=... JEPA_REPO=... FIRST_EPISODE_INDEX=0 FIRST_EPISODE_RESET_SEED=1000
#
# Output: nested run under artifacts/phase08_segment_grpo_baseline/run_* (see printed path).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

# Same venv as SmolVLA Slurm jobs (numpy, torch, metaworld, lerobot).
PYTHON_BIN="${SEGMENT_GRPO_PYTHON:-${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[run_first_episode_real] ERROR: python not executable: ${PYTHON_BIN}" >&2
  echo "[run_first_episode_real] Set SMOLVLA_LEROBOT_ENV_DIR or SEGMENT_GRPO_PYTHON." >&2
  exit 1
fi

# Defaults match this repo's SmolVLA runs (phase07 manifests) and torch.hub JEPA-WM checkout.
: "${SMOLVLA_CHECKPOINT:=jadechoghari/smolvla_metaworld}"
: "${JEPA_REPO:=${HOME}/.cache/torch/hub/facebookresearch_jepa-wms_main}"
if [[ ! -d "${JEPA_REPO}" ]]; then
  echo "[run_first_episode_real] ERROR: JEPA_REPO is not a directory: ${JEPA_REPO}" >&2
  echo "[run_first_episode_real] Clone or let torch.hub populate: facebookresearch/jepa-wms" >&2
  exit 1
fi

# MT1 push-v3 convention used elsewhere in this repo: reset_seed ≈ 1000 + episode_index
EPISODE_INDEX="${FIRST_EPISODE_INDEX:-0}"
RESET_SEED="${FIRST_EPISODE_RESET_SEED:-1000}"

echo "[run_first_episode_real] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[run_first_episode_real] PYTHON_BIN=${PYTHON_BIN}"
echo "[run_first_episode_real] episode_index=${EPISODE_INDEX} reset_seed=${RESET_SEED}"
echo "[run_first_episode_real] wm_rollout_mode=iterative carry_mode=sim device=cuda"

exec "${PYTHON_BIN}" scripts/run_segment_grpo.py \
  --checkpoint "${SMOLVLA_CHECKPOINT}" \
  --jepa-repo "${JEPA_REPO}" \
  --output-json "${PROJECT_ROOT}/artifacts/segment_grpo_first_episode_real.json" \
  --episodes 1 \
  --episode-index "${EPISODE_INDEX}" \
  --reset-seed "${RESET_SEED}" \
  --chunk-len 8 \
  --num-candidates 2 \
  --max-steps 200 \
  --carry-mode sim \
  --wm-rollout-mode iterative \
  --device cuda \
  "$@"
