#!/bin/bash
#SBATCH --job-name=phase9-oracle-vs-wm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/vol/bitbucket/aa6622/project/logs/phase9_oracle_vs_wm_%j.out
#SBATCH --error=/vol/bitbucket/aa6622/project/logs/phase9_oracle_vs_wm_%j.err
#
# Phase 9: oracle action replay + WM comparison strips (60 eps, no SmolVLA).
# Optional: JEPA_REPO, ORACLE_RUN_DIR, SEGMENT_GRPO_PYTHON / SMOLVLA_LEROBOT_ENV_DIR
#
set -euo pipefail
PROJECT_ROOT="/vol/bitbucket/aa6622/project"
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
VOL_CACHE="${WORKSPACE_ROOT}/.cache"
export HF_HOME="${HF_HOME:-${VOL_CACHE}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${VOL_CACHE}/torch}"
export JEPA_REPO="${JEPA_REPO:-${TORCH_HOME}/hub/facebookresearch_jepa-wms_main}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310}"

bash scripts/segment_grpo/run_phase9_oracle_vs_wm.sh
