#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/vol/bitbucket/aa6622/project"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"
PY="${SMOLVLA_LEROBOT_ENV_DIR}/bin/python"

# Pin HF / torch caches to /vol/bitbucket because /homes/aa6622 is not readable
# on some compute nodes (e.g. clapper), which causes HF from_pretrained() to
# throw Permission denied and _try_load_smolvla_exec to silently return None.
VOL_CACHE="${WORKSPACE_ROOT}/.cache"
export HF_HOME="${HF_HOME:-${VOL_CACHE}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${VOL_CACHE}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${VOL_CACHE}}"

export SMOLVLA_CHECKPOINT="${SMOLVLA_CHECKPOINT:-jadechoghari/smolvla_metaworld}"
export JEPA_REPO="${JEPA_REPO:-${TORCH_HOME}/hub/facebookresearch_jepa-wms_main}"

EPISODES="${EPISODES:-60}"
EPISODE_START="${EPISODE_START:-0}"
MAX_STEPS="${MAX_STEPS:-50}"
CHUNK_LEN="${CHUNK_LEN:-50}"
NUM_CAND="${NUM_CAND:-3}"
N_ACTION="${N_ACTION:-${MAX_STEPS}}"
RUN_NAME="${RUN_NAME:-}"

RUN_NAME_FLAG=()
if [ -n "${RUN_NAME}" ]; then RUN_NAME_FLAG=(--run-name "${RUN_NAME}"); fi

"${PY}" scripts/segment_grpo/run_all60_frame50_k3.py \
    --seed-base 1000 \
    --episode-start "${EPISODE_START}" \
    --episodes "${EPISODES}" \
    --goal-frame-index 50 \
    --num-candidates "${NUM_CAND}" \
    --chunk-len "${CHUNK_LEN}" \
    --max-steps "${MAX_STEPS}" \
    --smolvla-n-action-steps "${N_ACTION}" \
    --carry-mode sim \
    --wm-rollout-mode iterative \
    --wm-scoring-latent visual \
    --comparison-strip-overlay \
    --checkpoint "${SMOLVLA_CHECKPOINT}" \
    --jepa-repo "${JEPA_REPO}" \
    --output-root "${PROJECT_ROOT}/artifacts/phase08_segment_grpo_baseline" \
    "${RUN_NAME_FLAG[@]}"
