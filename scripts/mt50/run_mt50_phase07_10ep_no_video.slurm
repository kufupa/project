#!/usr/bin/env bash
# LEGACY/CUSTOM EVALUATOR (not official LeRobot reproduction).
# Official path: scripts/mt50/run_official_lerobot_mt50_eval.sh
# Historical note: custom backend did not mirror LeRobot's MetaWorld env setup
# (e.g. env._freeze_rand_vec after set_task); prefer Phase071 official artifacts.
#
#SBATCH --job-name=smolvla-mt50-10ep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=smolvla_mt50_phase07_10ep_%A_%a.out
#SBATCH --export=NIL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_ENV="${SCRIPT_DIR}/../slurm/common_env.sh"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh" ]]; then
  COMMON_ENV="${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
fi

if [[ -f "${COMMON_ENV}" ]]; then
  # shellcheck source=../slurm/common_env.sh
  source "${COMMON_ENV}"
  slurm_resolve_project_root "scripts/mt50/run_mt50_phase07_10ep_no_video.slurm"
  cd "${PROJECT_ROOT}"
  WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
  slurm_export_pythonpath
  slurm_export_hf_torch_cache "mt50-phase07-10ep-no-video"
  export WORKSPACE_ROOT
else
  echo "[mt50:phase07] WARN: common_env.sh missing at ${COMMON_ENV}, using fallback roots" >&2
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
  export PROJECT_ROOT
  export WORKSPACE_ROOT
  export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
fi

export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"
MAIN_ENV_DIR="${WORKSPACE_ROOT}/.envs/lerobot_mw_py310"
if [[ -n "${SMOLVLA_LEROBOT_ENV_DIR:-}" && -x "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${SMOLVLA_LEROBOT_ENV_DIR}/bin/python"
elif [[ -x "${MAIN_ENV_DIR}/bin/python" ]]; then
  if [[ "${SMOLVLA_LEROBOT_ENV_DIR}" != "${MAIN_ENV_DIR}" ]]; then
    echo "[mt50:phase07] fallback: SMOLVLA_LEROBOT_ENV_DIR invalid, using ${MAIN_ENV_DIR}" >&2
  fi
  PYTHON_BIN="${MAIN_ENV_DIR}/bin/python"
else
  PYTHON_BIN=""
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: python executable not found: ${MAIN_ENV_DIR}/bin/python" >&2
  echo "error: expected valid venv in SMOLVLA_LEROBOT_ENV_DIR=${SMOLVLA_LEROBOT_ENV_DIR:-<unset>}" >&2
  exit 2
fi
export SMOLVLA_PYTHON_BIN="${PYTHON_BIN}"

MT50_ROOT="${PROJECT_ROOT}/scripts/mt50"
if [[ ! -d "${MT50_ROOT}" ]]; then
  mkdir -p "${MT50_ROOT}"
fi
if [[ ! -w "${MT50_ROOT}" ]]; then
  echo "error: cannot write to task-config directory: ${MT50_ROOT}" >&2
  exit 2
fi
DIFF_JSON="${MT50_ROOT}/mt50_phase07_task_difficulties.json"
if [[ ! -f "${DIFF_JSON}" ]]; then
  cat <<'JSON' > "${DIFF_JSON}"
{
  "default": "unclassified",
  "task_difficulties": {}
}
JSON
fi
if [[ ! -r "${DIFF_JSON}" ]]; then
  echo "error: difficulty json unreadable: ${DIFF_JSON}" >&2
  exit 2
fi

export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export MT50_TASK_DIFFICULTY_JSON="${DIFF_JSON}"
export MT50_PHASE07_OUTPUT_ROOT="${PROJECT_ROOT}/artifacts/MT50_Phase07_500_10ep"
export MT50_PHASE07_EPISODES=10
export MT50_PHASE07_SEED=1000
export MT50_PHASE07_MAX_STEPS=500
export MT50_PHASE07_VIDEO=false
export MT50_PHASE07_FPS=30
export MT50_PHASE07_OVERLAY_MODE=reward_delta
export MT50_PHASE07_SAVE_FRAMES=false
export MT50_PHASE07_SAVE_ACTIONS=false
export MT50_PHASE07_RESUME=false
export MT50_PHASE07_PURGE_PARTIAL=true
export MT50_PHASE07_CHECKPOINT=jadechoghari/smolvla_metaworld
export MT50_PHASE07_USE_LOCAL_HF_SNAPSHOT="${MT50_PHASE07_USE_LOCAL_HF_SNAPSHOT:-true}"
export SMOLVLA_LOCAL_FILES_ONLY="${SMOLVLA_LOCAL_FILES_ONLY:-true}"
export SMOLVLA_PREFER_LOCAL_HF_SNAPSHOT="${SMOLVLA_PREFER_LOCAL_HF_SNAPSHOT:-true}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ -z "${MT50_PHASE07_CHECKPOINT}" ]]; then
  echo "error: MT50_PHASE07_CHECKPOINT is empty" >&2
  exit 2
fi
if ! mkdir -p "${MT50_PHASE07_OUTPUT_ROOT}"; then
  echo "error: cannot create output root: ${MT50_PHASE07_OUTPUT_ROOT}" >&2
  exit 2
fi
if [[ ! -w "${MT50_PHASE07_OUTPUT_ROOT}" ]]; then
  echo "error: output root not writable: ${MT50_PHASE07_OUTPUT_ROOT}" >&2
  exit 2
fi

echo "start phase07 10ep no-video run"
echo "output_root=${MT50_PHASE07_OUTPUT_ROOT}"
echo "episodes=${MT50_PHASE07_EPISODES}"
echo "seed=${MT50_PHASE07_SEED}"
echo "max_steps=${MT50_PHASE07_MAX_STEPS}"
echo "video=${MT50_PHASE07_VIDEO}"

_cache_root="${XDG_CACHE_HOME:-${WORKSPACE_ROOT}/.cache}"
export MPLCONFIGDIR="${SMOLVLA_MPLCONFIGDIR:-${_cache_root}/matplotlib_smolvla_parity}"
mkdir -p "${MPLCONFIGDIR}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
XVFB_ERR="${MT50_PHASE07_OUTPUT_ROOT}/mt50_phase07_campaign_xvfb.err"

if command -v xvfb-run >/dev/null 2>&1; then
  xvfb-run -a -e "${XVFB_ERR}" -s "-screen 0 1280x1024x24" \
    env "PYTHONPATH=${PYTHONPATH}" "MPLBACKEND=${MPLBACKEND}" "MPLCONFIGDIR=${MPLCONFIGDIR}" \
    "${PYTHON_BIN}" -m src.smolvla_pipeline.mt50_phase07_campaign
else
  env "PYTHONPATH=${PYTHONPATH}" "MPLBACKEND=${MPLBACKEND}" "MPLCONFIGDIR=${MPLCONFIGDIR}" \
    "${PYTHON_BIN}" -m src.smolvla_pipeline.mt50_phase07_campaign
fi
