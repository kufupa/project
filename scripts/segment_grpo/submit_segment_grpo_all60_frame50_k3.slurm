#!/usr/bin/env bash
#SBATCH --job-name=seggrpo-all60-f50-k3
#SBATCH --output=/vol/bitbucket/aa6622/project/logs/seggrpo_all60_f50_k3_%j.out
#SBATCH --error=/vol/bitbucket/aa6622/project/logs/seggrpo_all60_f50_k3_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#
# One GPU job, sequential 60 episodes (subprocess per episode for memory isolation).
# Submit from login node (avoid nested sbatch inside another allocation).
#
#   sbatch /vol/bitbucket/aa6622/project/scripts/segment_grpo/submit_segment_grpo_all60_frame50_k3.slurm
#
set -euo pipefail

PROJECT_ROOT="/vol/bitbucket/aa6622/project"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"
PY="${SMOLVLA_LEROBOT_ENV_DIR}/bin/python"

# Pin HF / torch caches to /vol/bitbucket because /homes/aa6622 is not readable
# on some compute nodes (e.g. clapper) and causes "[Errno 13] Permission denied:
# '/homes/aa6622'" inside HuggingFace from_pretrained(), which _try_load_smolvla_exec
# swallows as a silent None and surfaces as "SmolVLA helper returned None".
VOL_CACHE="${WORKSPACE_ROOT}/.cache"
export HF_HOME="${HF_HOME:-${VOL_CACHE}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${VOL_CACHE}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${VOL_CACHE}}"

export SMOLVLA_CHECKPOINT="${SMOLVLA_CHECKPOINT:-jadechoghari/smolvla_metaworld}"
export JEPA_REPO="${JEPA_REPO:-${TORCH_HOME}/hub/facebookresearch_jepa-wms_main}"

echo "[seggrpo-all60] SMOLVLA_CHECKPOINT=${SMOLVLA_CHECKPOINT}"
echo "[seggrpo-all60] JEPA_REPO=${JEPA_REPO}"
echo "[seggrpo-all60] HF_HOME=${HF_HOME}"
echo "[seggrpo-all60] TORCH_HOME=${TORCH_HOME}"
echo "[seggrpo-all60] host=$(hostname) job=${SLURM_JOB_ID:-local} start=$(date -Is)"

"${PY}" scripts/segment_grpo/run_all60_frame50_k3.py \
    --seed-base 1000 \
    --episode-start 0 \
    --episodes 60 \
    --goal-frame-index 50 \
    --num-candidates 3 \
    --chunk-len 50 \
    --max-steps 50 \
    --smolvla-n-action-steps 50 \
    --carry-mode sim \
    --wm-rollout-mode iterative \
    --wm-scoring-latent visual \
    --comparison-strip-overlay \
    --checkpoint "${SMOLVLA_CHECKPOINT}" \
    --jepa-repo "${JEPA_REPO}" \
    --output-root "${PROJECT_ROOT}/artifacts/phase08_segment_grpo_baseline"

echo "[seggrpo-all60] end=$(date -Is)"
