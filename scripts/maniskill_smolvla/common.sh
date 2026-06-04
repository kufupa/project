#!/usr/bin/env bash
# Shared helpers for autonomous SmolVLA x ManiSkill PBS jobs.

set -euo pipefail

msm_setup_modules() {
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/dev/null
    . /etc/profile.d/modules.sh
  fi
  module purge >/dev/null 2>&1 || true
  module load tools/prod
  module load Python/3.12.3-GCCcore-13.3.0
  module load Mesa/24.1.3-GCCcore-13.3.0
  if [[ -n "${EBROOTPython:-}" && -d "${EBROOTPython}/lib" ]]; then
    export LD_LIBRARY_PATH="${EBROOTPython}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

msm_resolve_paths() {
  export PROJECT_ROOT="${PROJECT_ROOT:-/rds/general/user/aa6622/home/project}"
  export MSM_SCRIPT_ROOT="${MSM_SCRIPT_ROOT:-${PROJECT_ROOT}/scripts/maniskill_smolvla}"
  export RL4VLA_ROOT="${RL4VLA_ROOT:-${PROJECT_ROOT}/RL4VLA}"
  export MSM_EPHEMERAL_ROOT="${MSM_EPHEMERAL_ROOT:-/rds/general/user/aa6622/ephemeral/eggroll/smolvla_maniskill}"
  export MSM_PROJECT_ARTIFACT_ROOT="${MSM_PROJECT_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/smolvla_maniskill}"
  export MSM_VENV_ROOT="${MSM_VENV_ROOT:-/rds/general/user/aa6622/ephemeral/eggroll/venvs}"

  export MSM_DATA_PYTHON="${MSM_DATA_PYTHON:-${PROJECT_ROOT}/.venvs/pirl-rlinf-py312/bin/python}"
  export MSM_TRAIN_PYTHON="${MSM_TRAIN_PYTHON:-/rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python}"
  export MSM_TRAIN_VENV="${MSM_TRAIN_VENV:-${MSM_VENV_ROOT}/smolvla_lerobot_py312}"

  export MSM_RAW_ROOT="${MSM_RAW_ROOT:-${MSM_EPHEMERAL_ROOT}/raw_npz}"
  export MSM_LEROBOT_ROOT="${MSM_LEROBOT_ROOT:-${MSM_EPHEMERAL_ROOT}/lerobot_dataset}"
  export MSM_CHECKPOINT_ROOT="${MSM_CHECKPOINT_ROOT:-${MSM_EPHEMERAL_ROOT}/checkpoints}"
  export MSM_VIDEO_ROOT="${MSM_VIDEO_ROOT:-${MSM_EPHEMERAL_ROOT}/videos}"
  export MSM_HF_HOME="${MSM_HF_HOME:-${MSM_EPHEMERAL_ROOT}/hf_home}"
  export MSM_HF_LEROBOT_HOME="${MSM_HF_LEROBOT_HOME:-${MSM_EPHEMERAL_ROOT}/hf_lerobot_home}"
  export MSM_TORCH_HOME="${MSM_TORCH_HOME:-${MSM_EPHEMERAL_ROOT}/torch_home}"

  export MSM_ENV_ID="${MSM_ENV_ID:-PutOnPlateInScene25Main-v3}"
  export MSM_REPO_ID="${MSM_REPO_ID:-local/smolvla_maniskill_25main}"
  export MSM_FPS="${MSM_FPS:-5}"

  local pirl_ms_assets="${PROJECT_ROOT}/artifacts/pirl_maniskill/maniskill_assets"
  local pirl_task_assets="${PROJECT_ROOT}/artifacts/pirl_maniskill/maniskill_task_assets"
  export MS_ASSET_DIR="${MS_ASSET_DIR:-${pirl_ms_assets}}"
  export MANISKILL_ASSET_DIR="${MANISKILL_ASSET_DIR:-${pirl_task_assets}}"
  if [[ ! -d "${MS_ASSET_DIR}" ]]; then
    export MS_ASSET_DIR="${MSM_EPHEMERAL_ROOT}/maniskill_assets"
  fi
  if [[ ! -d "${MANISKILL_ASSET_DIR}" ]]; then
    export MANISKILL_ASSET_DIR="${MSM_EPHEMERAL_ROOT}/maniskill_task_assets"
  fi

  local job_tag="${PBS_JOBID:-local}"
  export MSM_RUN_ID="${MSM_RUN_ID:-${job_tag}}"
  export MSM_RUN_ROOT="${MSM_RUN_ROOT:-${MSM_PROJECT_ARTIFACT_ROOT}/runs/${MSM_RUN_ID}}"
  export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/msm_${USER}_${MSM_RUN_ID}/ray_tmp}"
  export TMPDIR="${TMPDIR:-/tmp/msm_${USER}_${MSM_RUN_ID}/tmp}"
  export HF_HOME="${HF_HOME:-${MSM_HF_HOME}}"
  export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${MSM_HF_LEROBOT_HOME}}"
  export TORCH_HOME="${TORCH_HOME:-${MSM_TORCH_HOME}}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${MSM_EPHEMERAL_ROOT}/xdg_cache}"
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
  export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
  export MS_SKIP_ASSET_DOWNLOAD_PROMPT="${MS_SKIP_ASSET_DOWNLOAD_PROMPT:-1}"

  case ":${PYTHONPATH:-}:" in
    *":${RL4VLA_ROOT}/ManiSkill:"*) ;;
    *) export PYTHONPATH="${RL4VLA_ROOT}/ManiSkill${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
  case ":${PYTHONPATH:-}:" in
    *":${RL4VLA_ROOT}:"*) ;;
    *) export PYTHONPATH="${RL4VLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
}

msm_prepare_runtime() {
  msm_resolve_paths
  mkdir -p \
    "${MSM_EPHEMERAL_ROOT}" "${MSM_PROJECT_ARTIFACT_ROOT}" "${MSM_RUN_ROOT}" \
    "${MSM_RUN_ROOT}/logs" "${MSM_RUN_ROOT}/rca" "${MSM_RUN_ROOT}/manifests" \
    "${MSM_RAW_ROOT}" "${MSM_LEROBOT_ROOT}" "${MSM_CHECKPOINT_ROOT}" \
    "${MSM_VIDEO_ROOT}" "${HF_HOME}" "${HF_LEROBOT_HOME}" "${TORCH_HOME}" \
    "${XDG_CACHE_HOME}" "${RAY_TMPDIR}" "${TMPDIR}" \
    "${MS_ASSET_DIR}" "${MANISKILL_ASSET_DIR}"
}

msm_require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "error: missing ${path}" >&2
    exit 2
  fi
}

msm_require_python() {
  local python_bin="$1"
  msm_require_file "${python_bin}"
  "${python_bin}" - <<'PY'
import sys
print(f"[msm] python={sys.executable} version={sys.version.split()[0]}")
PY
}

msm_stage_log() {
  local name="$1"
  mkdir -p "${MSM_RUN_ROOT}/logs"
  echo "${MSM_RUN_ROOT}/logs/${name}_$(date -u +%Y%m%d_%H%M%S).log"
}

msm_write_manifest() {
  local path="$1"
  shift
  mkdir -p "$(dirname "${path}")"
  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "project_root=${PROJECT_ROOT}"
    echo "git_sha=$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || true)"
    echo "git_branch=$(git -C "${PROJECT_ROOT}" branch --show-current 2>/dev/null || true)"
    printf '%s\n' "$@"
  } > "${path}"
}
