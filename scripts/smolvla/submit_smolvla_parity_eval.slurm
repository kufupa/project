#!/bin/bash
# SmolVLA parity benchmark (Meta-World + LeRobot). Pass args after the script name, e.g.:
#   cd /path/to/project && sbatch --export=NIL scripts/smolvla/submit_smolvla_parity_eval.slurm \
#     --task reach-v3 --episodes 20 --seed 1000 --max-steps 120 \
#     --overlay-mode reward_delta --save-frames true
# Submit from the repo `project/` root (so SLURM_SUBMIT_DIR resolves), or set SMOLVLA_PROJECT_ROOT.
#
#SBATCH --job-name=smolvla-parity
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=smolvla_parity_eval_%j.out

set -euo pipefail

# Slurm may execute this file from /var/spool/slurm/...; do not resolve common_env.sh from BASH_SOURCE alone.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh" ]]; then
  _SLURM_COMMON="${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
else
  _SLURM_COMMON="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/../slurm/common_env.sh"
fi
# shellcheck source=../slurm/common_env.sh
source "${_SLURM_COMMON}"
slurm_resolve_project_root "scripts/smolvla/legacy_run_pushv3_smolvla_parity_benchmark.sh"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "smolvla-parity-eval"

TASK="${SMOLVLA_PARITY_TASK:-push-v3}"
EPISODES="${SMOLVLA_PARITY_EPISODES:-15}"
SEED="${SMOLVLA_PARITY_SEED:-1000}"
MAX_STEPS="${SMOLVLA_PARITY_MAX_STEPS:-120}"
FPS="${SMOLVLA_PARITY_FPS:-30}"
OVERLAY_MODE="${SMOLVLA_PARITY_OVERLAY_MODE:-reward_delta}"
SAVE_FRAMES="${SMOLVLA_SAVE_FRAMES:-true}"
CAMERA_NAME="${SMOLVLA_METAWORLD_CAMERA_NAME:-corner2}"
FLIP_CORNER2="${SMOLVLA_FLIP_CORNER2:-true}"
CHECKPOINT="${SMOLVLA_INIT_CHECKPOINT:-jadechoghari/smolvla_metaworld}"
LOAD_VLM_WEIGHTS="${SMOLVLA_LOAD_VLM_WEIGHTS:-true}"
OUTPUT_ROOT="${SMOLVLA_PARITY_OUTPUT_ROOT:-${SMOLVLA_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}/phase07_smolvla_baseline/parity}"
MIN_VIDEO_BYTES="${SMOLVLA_PARITY_MIN_VIDEO_BYTES:-1024}"
TASK_TEXT="${SMOLVLA_TASK_TEXT:-}"

log_error() { echo "error: $*" >&2; }

assert_positive_int() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" -lt 1 ]]; then
    log_error "Invalid ${name}: ${value}. Expected positive integer."
    exit 2
  fi
}

assert_bool() {
  local name="$1"
  local value="${2,,}"
  case "${value}" in
    true|false|1|0|yes|no|on|off) ;;
    *)
      log_error "Invalid ${name}: ${2}. Expected boolean-like (true/false/1/0/yes/no/on/off)."
      exit 2
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="${2}"
      shift 2
      ;;
    --episodes)
      EPISODES="${2}"
      shift 2
      ;;
    --seed)
      SEED="${2}"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="${2}"
      shift 2
      ;;
    --fps)
      FPS="${2}"
      shift 2
      ;;
    --overlay-mode)
      OVERLAY_MODE="${2}"
      shift 2
      ;;
    --save-frames)
      SAVE_FRAMES="${2}"
      shift 2
      ;;
    --camera-name)
      CAMERA_NAME="${2}"
      shift 2
      ;;
    --flip-corner2)
      FLIP_CORNER2="${2}"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="${2}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2}"
      shift 2
      ;;
    --min-video-bytes)
      MIN_VIDEO_BYTES="${2}"
      shift 2
      ;;
    --task-text)
      TASK_TEXT="${2}"
      shift 2
      ;;
    *)
      log_error "Unknown arg: ${1}"
      exit 2
      ;;
  esac
done

if [[ -z "${TASK}" ]]; then
  log_error "task must be non-empty."
  exit 2
fi

assert_positive_int "episodes" "${EPISODES}"
assert_positive_int "seed" "${SEED}"
assert_positive_int "max_steps" "${MAX_STEPS}"
assert_positive_int "fps" "${FPS}"
assert_bool "save_frames" "${SAVE_FRAMES}"
assert_bool "flip_corner2" "${FLIP_CORNER2}"
assert_positive_int "min_video_bytes" "${MIN_VIDEO_BYTES}"

export SMOLVLA_PARITY_TASK="${TASK}"
export SMOLVLA_PARITY_EPISODES="${EPISODES}"
export SMOLVLA_PARITY_SEED="${SEED}"
export SMOLVLA_PARITY_MAX_STEPS="${MAX_STEPS}"
export SMOLVLA_PARITY_FPS="${FPS}"
export SMOLVLA_PARITY_OVERLAY_MODE="${OVERLAY_MODE}"
export SMOLVLA_SAVE_FRAMES="${SAVE_FRAMES}"
export SMOLVLA_METAWORLD_CAMERA_NAME="${CAMERA_NAME}"
export SMOLVLA_FLIP_CORNER2="${FLIP_CORNER2}"
export SMOLVLA_INIT_CHECKPOINT="${CHECKPOINT}"
export SMOLVLA_LOAD_VLM_WEIGHTS="${LOAD_VLM_WEIGHTS}"
export SMOLVLA_PARITY_OUTPUT_ROOT="${OUTPUT_ROOT}"
export SMOLVLA_PARITY_MIN_VIDEO_BYTES="${MIN_VIDEO_BYTES}"
if [[ -n "${TASK_TEXT}" ]]; then
  export SMOLVLA_TASK_TEXT="${TASK_TEXT}"
fi

exec bash "${PROJECT_ROOT}/scripts/smolvla/legacy_run_pushv3_smolvla_parity_benchmark.sh"
