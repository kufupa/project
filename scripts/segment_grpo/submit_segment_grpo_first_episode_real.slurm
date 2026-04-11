#!/bin/bash
#SBATCH --job-name=seggrpo-ep0-real
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/bitbucket/aa6622/project/logs/seggrpo_first_ep_real_%j.out
#SBATCH --error=/vol/bitbucket/aa6622/project/logs/seggrpo_first_ep_real_%j.err
#
# Full real pipeline: SmolVLA chunk sampling + JEPA-WM iterative rollout scoring + MetaWorld sim.
# Targets oracle episode_index 0 (reset_seed 1000) only — not the Top-15 rank-1 row.
#
# Submit (export vars in your shell so Slurm forwards them, or use sbatch --export):
#   export SMOLVLA_CHECKPOINT=...
#   export JEPA_REPO=...
#   sbatch /vol/bitbucket/aa6622/project/scripts/segment_grpo/submit_segment_grpo_first_episode_real.slurm
#
# Optional: FIRST_EPISODE_INDEX / FIRST_EPISODE_RESET_SEED for a different single episode.
#
set -euo pipefail
PROJECT_ROOT="/vol/bitbucket/aa6622/project"
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"

# Defaults: same as run_first_episode_real_pipeline.sh (phase07 checkpoint + hub cache).
export SMOLVLA_CHECKPOINT="${SMOLVLA_CHECKPOINT:-jadechoghari/smolvla_metaworld}"
export JEPA_REPO="${JEPA_REPO:-${HOME}/.cache/torch/hub/facebookresearch_jepa-wms_main}"

bash scripts/segment_grpo/run_first_episode_real_pipeline.sh
