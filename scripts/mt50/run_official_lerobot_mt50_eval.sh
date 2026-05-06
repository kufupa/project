#!/usr/bin/env bash
# Official SmolVLA × MetaWorld MT50 evaluation via HuggingFace LeRobot `lerobot-eval`.
# Canonical reproduction path (Phase071). See summarize_official_lerobot_eval.py for MT50-style index.
#
# Env (optional overrides):
#   MT50_PHASE071_TASK        — default assembly-v3; use "all" for full 50-task list from train_config.json
#   MT50_PHASE071_EPISODES    — default 1
#   MT50_PHASE071_SEED        — default 1000
#   MT50_PHASE071_OUTPUT_ROOT — default artifacts/MT50_Phase071_official_lerobot_1task_1ep or ..._1ep
#   MT50_PHASE071_CHECKPOINT  — directory with train_config.json + pretrained weights
#   MT50_PHASE071_DRY_RUN     — if true, print command and exit
#   SMOLVLA_LEROBOT_ENV_DIR   — venv with lerobot (default: ${WORKSPACE_ROOT}/.envs/lerobot_mw_py310)
#
set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_ENV="${SCRIPT_DIR}/../slurm/common_env.sh"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh" ]]; then
  COMMON_ENV="${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
fi

# shellcheck source=../slurm/common_env.sh
source "${COMMON_ENV}"
slurm_resolve_project_root "scripts/mt50/run_official_lerobot_mt50_eval.sh"
cd "${PROJECT_ROOT}"

WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export WORKSPACE_ROOT

slurm_export_pythonpath
slurm_export_hf_torch_cache "mt50-phase071-official"

export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"
PYTHON_BIN="${SMOLVLA_LEROBOT_ENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

SNAPSHOT="${MT50_PHASE071_CHECKPOINT:-${WORKSPACE_ROOT}/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
if [[ "${MT50_PHASE071_DRY_RUN:-false}" != "true" && ! -d "${SNAPSHOT}" ]]; then
  echo "error: checkpoint snapshot missing: ${SNAPSHOT}" >&2
  exit 2
fi

export SMOLVLA_LOCAL_FILES_ONLY="${SMOLVLA_LOCAL_FILES_ONLY:-true}"
export SMOLVLA_PREFER_LOCAL_HF_SNAPSHOT="${SMOLVLA_PREFER_LOCAL_HF_SNAPSHOT:-true}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME:-${WORKSPACE_ROOT}/.cache}/matplotlib_smolvla_official}"
mkdir -p "${MPLCONFIGDIR}"

# Headless MuJoCo render (GPU cluster); rgb_array still works with EGL.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

TASK="${MT50_PHASE071_TASK:-assembly-v3}"
EPISODES="${MT50_PHASE071_EPISODES:-1}"
SEED="${MT50_PHASE071_SEED:-1000}"

if [[ "${TASK}" == "all" ]]; then
  if [[ ! -f "${SNAPSHOT}/train_config.json" ]]; then
    echo "error: train_config.json missing under snapshot: ${SNAPSHOT}" >&2
    exit 2
  fi
  TASK="$(
    SNAPSHOT="${SNAPSHOT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

cfg = json.loads((Path(os.environ["SNAPSHOT"]) / "train_config.json").read_text())
print(cfg["env"]["task"])
PY
  )"
  DEFAULT_OUT="${PROJECT_ROOT}/artifacts/MT50_Phase071_official_lerobot_1ep"
else
  DEFAULT_OUT="${PROJECT_ROOT}/artifacts/MT50_Phase071_official_lerobot_1task_1ep"
fi

OUTPUT_ROOT="${MT50_PHASE071_OUTPUT_ROOT:-${DEFAULT_OUT}}"
mkdir -p "${OUTPUT_ROOT}"

cmd=(
  "${PYTHON_BIN}" -m lerobot.scripts.lerobot_eval
  --policy.path="${SNAPSHOT}"
  --env.type=metaworld
  --env.task="${TASK}"
  --eval.n_episodes="${EPISODES}"
  --eval.batch_size=1
  --eval.use_async_envs=false
  --seed="${SEED}"
  --output_dir="${OUTPUT_ROOT}"
)

printf '[mt50:phase071] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
echo "[mt50:phase071] output_root=${OUTPUT_ROOT}"
echo "[mt50:phase071] task=${TASK}"
echo "[mt50:phase071] episodes=${EPISODES}"
echo "[mt50:phase071] seed=${SEED}"
echo "[mt50:phase071] expected_horizon=500 via LeRobot MetaWorld _max_episode_steps"

if [[ "${MT50_PHASE071_DRY_RUN:-false}" == "true" ]]; then
  exit 0
fi

if command -v xvfb-run >/dev/null 2>&1; then
  exec xvfb-run -a -s "-screen 0 1280x1024x24" "${cmd[@]}"
else
  exec "${cmd[@]}"
fi
