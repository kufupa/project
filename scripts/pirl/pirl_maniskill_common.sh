#!/usr/bin/env bash
# Common CX3 helpers for PIRL ManiSkill RLinf jobs.

set -euo pipefail

pirl_setup_modules() {
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/dev/null
    . /etc/profile.d/modules.sh
  fi
  module purge >/dev/null 2>&1 || true
  module load tools/prod
  module load Python/3.12.3-GCCcore-13.3.0
  module load Mesa/24.1.3-GCCcore-13.3.0
}

pirl_assert_python312() {
  local python_bin="${PIRL_PYTHON:-python3}"
  "${python_bin}" - <<'PY'
import sys

version = sys.version_info
if version[:2] != (3, 12):
    raise SystemExit(
        f"error: Python 3.12 required, got {sys.version.split()[0]} "
        f"from {sys.executable}"
    )
print(f"[pirl] python={sys.executable} version={sys.version.split()[0]}")
PY
}

pirl_resolve_paths() {
  export PROJECT_ROOT="${PROJECT_ROOT:-/rds/general/user/aa6622/home/project}"
  export RLINF_ROOT="${RLINF_ROOT:-${PROJECT_ROOT}/RLinf}"
  export RLINF_CONFIG="${RLINF_CONFIG:-maniskill_ppo_openpi_pi05_rtx6000_flow_sde}"
  export PIRL_ARTIFACT_ROOT="${PIRL_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/pirl_maniskill}"
  export PIRL_RUNTIME_VENV="${PIRL_RUNTIME_VENV:-${PROJECT_ROOT}/.venvs/pirl-rlinf-py312}"
  if [[ -z "${PIRL_PYTHON:-}" && -x "${PIRL_RUNTIME_VENV}/bin/python" ]]; then
    export PIRL_PYTHON="${PIRL_RUNTIME_VENV}/bin/python"
  fi
  export PIRL_SFT_CKPT="${PIRL_SFT_CKPT:-${PIRL_ARTIFACT_ROOT}/hf_models/RLinf-Pi05-ManiSkill-25Main-SFT}"

  local job_tag
  job_tag="${PBS_JOBID:-local}"
  export PIRL_RUN_ROOT="${PIRL_RUN_ROOT:-${PIRL_ARTIFACT_ROOT}/runs/${job_tag}}"
  export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/pirl_${USER}_${job_tag}/ray_tmp}"
  export EMBODIED_PATH="${EMBODIED_PATH:-${RLINF_ROOT}/examples/embodiment}"

  case ":${PYTHONPATH:-}:" in
    *":${RLINF_ROOT}:"*) ;;
    *) export PYTHONPATH="${RLINF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
}

pirl_prepare_runtime() {
  pirl_resolve_paths
  pirl_assert_python312
  mkdir -p "${PIRL_RUN_ROOT}" "${PIRL_ARTIFACT_ROOT}" "${RAY_TMPDIR}"
  mkdir -p "${PIRL_RUN_ROOT}/logs" "${PIRL_RUN_ROOT}/snapshots"
  export TMPDIR="${TMPDIR:-${PIRL_RUN_ROOT}/tmp}"
  mkdir -p "${TMPDIR}"
  export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
  export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
  export HF_HOME="${HF_HOME:-${PIRL_ARTIFACT_ROOT}/hf_home}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PIRL_ARTIFACT_ROOT}/xdg_cache}"
  export MS_ASSET_DIR="${MS_ASSET_DIR:-${PIRL_ARTIFACT_ROOT}/maniskill_assets}"
  export MANISKILL_ASSET_DIR="${MANISKILL_ASSET_DIR:-${PIRL_ARTIFACT_ROOT}/maniskill_task_assets}"
  mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${MANISKILL_ASSET_DIR}"
  mkdir -p "${MS_ASSET_DIR}"

  if [[ ! -d "${RLINF_ROOT}" ]]; then
    echo "error: missing RLINF_ROOT=${RLINF_ROOT}" >&2
    exit 2
  fi
  if [[ ! -f "${RLINF_ROOT}/examples/embodiment/config/${RLINF_CONFIG}.yaml" ]]; then
    echo "error: missing RLINF_CONFIG=${RLINF_CONFIG}" >&2
    exit 2
  fi
  pirl_ensure_sft_checkpoint
}

pirl_ensure_sft_checkpoint() {
  if [[ -e "${PIRL_SFT_CKPT}" ]]; then
    return 0
  fi
  echo "[pirl] SFT checkpoint missing; downloading RLinf/RLinf-Pi05-ManiSkill-25Main-SFT to ${PIRL_SFT_CKPT}" >&2
  mkdir -p "${PIRL_SFT_CKPT}"
  "${PIRL_PYTHON:-python3}" - <<'PY'
import os
from huggingface_hub import snapshot_download

target = os.environ["PIRL_SFT_CKPT"]
snapshot_download(
    repo_id="RLinf/RLinf-Pi05-ManiSkill-25Main-SFT",
    local_dir=target,
    local_dir_use_symlinks=False,
)
print(f"[pirl] downloaded SFT checkpoint to {target}")
PY
}

pirl_gpu_snapshot() {
  pirl_prepare_runtime
  local snapshot="${PIRL_RUN_ROOT}/snapshots/gpu_snapshot_$(date -u +%Y%m%d_%H%M%S).txt"
  echo "[pirl] GPU snapshot: v1_gpu72 RTX6000 queue; v1_jupytergpu default excluded."
  "${PIRL_PYTHON:-python3}" "$HOME/.agents/skills/checking-pbs-gpu-availability/scripts/pbs_gpu_snapshot.py" -q v1_gpu72 | tee "${snapshot}"
}

pirl_cleanup_ray() {
  if command -v ray >/dev/null 2>&1; then
    ray stop --force || true
  fi
}

pirl_latest_checkpoint() {
  pirl_resolve_paths
  local root="${1:-${PIRL_ARTIFACT_ROOT}}"
  local -a candidates=()
  shopt -s nullglob
  candidates+=("${root}"/checkpoints/global_step_*)
  candidates+=("${root}"/*/checkpoints/global_step_*)
  candidates+=("${root}"/*/*/checkpoints/global_step_*)
  candidates+=("${PIRL_ARTIFACT_ROOT}"/checkpoints/global_step_*)
  candidates+=("${PIRL_ARTIFACT_ROOT}"/*/checkpoints/global_step_*)
  candidates+=("${PIRL_ARTIFACT_ROOT}"/*/*/checkpoints/global_step_*)
  shopt -u nullglob

  local -a dirs=()
  local candidate
  for candidate in "${candidates[@]}"; do
    [[ -d "${candidate}" ]] && dirs+=("${candidate}")
  done
  if [[ "${#dirs[@]}" -eq 0 ]]; then
    return 1
  fi
  printf '%s\n' "${dirs[@]}" | sort -V | tail -n 1
}

pirl_run_rlinf() {
  local run_name="${1:-run}"
  shift || true
  pirl_prepare_runtime
  pirl_gpu_snapshot
  pirl_cleanup_ray

  local log_dir="${PIRL_RUN_ROOT}/logs"
  mkdir -p "${log_dir}"
  local log="${log_dir}/${run_name}_$(date -u +%Y%m%d_%H%M%S).log"
  export PIRL_LAST_LOG="${log}"

  echo "[pirl] PROJECT_ROOT=${PROJECT_ROOT}"
  echo "[pirl] RLINF_ROOT=${RLINF_ROOT}"
  echo "[pirl] PIRL_RUN_ROOT=${PIRL_RUN_ROOT}"
  echo "[pirl] PIRL_ARTIFACT_ROOT=${PIRL_ARTIFACT_ROOT}"
  echo "[pirl] RAY_TMPDIR=${RAY_TMPDIR}"
  echo "[pirl] log=${log}"
  echo "[pirl] native entrypoint: examples/embodiment/train_embodied_agent.py --config-name ${RLINF_CONFIG}"

  cd "${RLINF_ROOT}"
  set +e
  "${PIRL_PYTHON:-python3}" examples/embodiment/train_embodied_agent.py \
    --config-name "${RLINF_CONFIG}" \
    "$@" 2>&1 | tee "${log}"
  local status="${PIPESTATUS[0]}"
  set -e
  pirl_cleanup_ray
  return "${status}"
}
