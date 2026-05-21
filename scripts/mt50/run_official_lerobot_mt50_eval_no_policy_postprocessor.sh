#!/usr/bin/env bash
# Official SmolVLA x MetaWorld MT50 eval with policy postprocessor disabled.
#
# MT50_XVFB_SERVER_NUM — when MUJOCO_GL=glfw, fixed xvfb display (not -a); see run_official_lerobot_mt50_eval.sh.
#
# Env mirrors run_official_lerobot_mt50_eval.sh. This wrapper is isolated so
# running Phase27 jobs keep using the original wrapper unchanged.
set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_ENV="${SCRIPT_DIR}/../slurm/common_env.sh"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh" ]]; then
  COMMON_ENV="${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
fi

# shellcheck source=../slurm/common_env.sh
source "${COMMON_ENV}"
slurm_resolve_project_root "scripts/mt50/run_official_lerobot_mt50_eval_no_policy_postprocessor.sh"
cd "${PROJECT_ROOT}"

WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export WORKSPACE_ROOT

slurm_export_pythonpath
slurm_export_hf_torch_cache "mt50-phase27-no-policy-postprocessor"

export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py312}"
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
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME:-${WORKSPACE_ROOT}/.cache}/matplotlib_smolvla_official_no_policy_postprocessor}"
mkdir -p "${MPLCONFIGDIR}"

# Headless MuJoCo render (GPU cluster); default egl avoids Xvfb races.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

TASK="${MT50_PHASE071_TASK:-assembly-v3}"
EPISODES="${MT50_PHASE071_EPISODES:-1}"
SEED="${MT50_PHASE071_SEED:-1000}"
RENDER_LIMIT="${MT50_LEROBOT_MAX_EPISODES_RENDERED:-}"
FREEZE_RAND_VEC_RAW="${MT50_METAWORLD_FREEZE_RAND_VEC:-}"
if [[ -n "${RENDER_LIMIT}" && ! "${RENDER_LIMIT}" =~ ^[0-9]+$ ]]; then
  echo "error: MT50_LEROBOT_MAX_EPISODES_RENDERED must be a non-negative integer: ${RENDER_LIMIT}" >&2
  exit 2
fi
if [[ -n "${FREEZE_RAND_VEC_RAW}" ]]; then
  FREEZE_RAND_VEC_NORMALIZED="$(printf '%s' "${FREEZE_RAND_VEC_RAW}" | tr '[:upper:]' '[:lower:]')"
  if [[ ! "${FREEZE_RAND_VEC_NORMALIZED}" =~ ^(true|false|1|0|yes|no|on|off)$ ]]; then
    echo "error: MT50_METAWORLD_FREEZE_RAND_VEC must be one of true|false|1|0|yes|no|on|off: ${FREEZE_RAND_VEC_RAW}" >&2
    exit 2
  fi
  export MT50_METAWORLD_FREEZE_RAND_VEC="${FREEZE_RAND_VEC_NORMALIZED}"
fi

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
  DEFAULT_OUT="${PROJECT_ROOT}/artifacts/MT50_Phase27_no_policy_postprocessor_official_lerobot_1ep"
else
  DEFAULT_OUT="${PROJECT_ROOT}/artifacts/MT50_Phase27_no_policy_postprocessor_official_lerobot_1task_1ep"
fi

OUTPUT_ROOT="${MT50_PHASE071_OUTPUT_ROOT:-${DEFAULT_OUT}}"
mkdir -p "${OUTPUT_ROOT}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/lerobot_eval_no_policy_postprocessor.py"
  --policy.path="${SNAPSHOT}"
  --env.type=metaworld
  --env.task="${TASK}"
  --eval.n_episodes="${EPISODES}"
  --eval.batch_size=1
  --eval.use_async_envs=false
  --seed="${SEED}"
  --output_dir="${OUTPUT_ROOT}"
)

printf '[mt50:phase27-no-policy-postprocessor] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
echo "[mt50:phase27-no-policy-postprocessor] output_root=${OUTPUT_ROOT}"
echo "[mt50:phase27-no-policy-postprocessor] task=${TASK}"
echo "[mt50:phase27-no-policy-postprocessor] episodes=${EPISODES}"
echo "[mt50:phase27-no-policy-postprocessor] seed=${SEED}"
echo "[mt50:phase27-no-policy-postprocessor] max_episodes_rendered=${RENDER_LIMIT:-lerobot_default_10}"
echo "[mt50:phase27-no-policy-postprocessor] metaworld_freeze_rand_vec=${MT50_METAWORLD_FREEZE_RAND_VEC:-lerobot_default_false}"
echo "[mt50:phase27-no-policy-postprocessor] policy_postprocessor=disabled_identity"
echo "[mt50:phase27-no-policy-postprocessor] expected_horizon=500 via LeRobot MetaWorld _max_episode_steps"

if [[ "${MT50_PHASE071_DRY_RUN:-false}" == "true" ]]; then
  exit 0
fi

MUJOCO_GL_LOWER="$(printf '%s' "${MUJOCO_GL:-egl}" | tr '[:upper:]' '[:lower:]')"
if [[ "${MUJOCO_GL_LOWER}" == "egl" ]]; then
  export MUJOCO_GL="egl"
  exec "${cmd[@]}"
fi

if [[ "${MUJOCO_GL_LOWER}" == "glfw" ]]; then
  export MUJOCO_GL="glfw"
  if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "warn: MUJOCO_GL=glfw but xvfb-run missing; running without Xvfb (may fail headless)" >&2
    exec "${cmd[@]}"
  fi
  srv="${MT50_XVFB_SERVER_NUM:-}"
  if [[ -z "${srv}" ]]; then
    echo "error: MUJOCO_GL=glfw headless needs xvfb-run with fixed display; set MT50_XVFB_SERVER_NUM (e.g. 100 per parallel worker)" >&2
    exit 2
  fi
  exec xvfb-run -n "${srv}" -s "-screen 0 1280x1024x24" "${cmd[@]}"
fi

exec "${cmd[@]}"
