#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

ORACLE_EVAL_SCRIPT="${SCRIPT_DIR}/run_metaworld_oracle_eval.py"
LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"
ORACLE_XVFB_ENABLED="${ORACLE_XVFB_ENABLED:-1}"

EPISODES="${ORACLE_BASELINE_EPISODES:-15}"
# Match phase06 oracle campaign convention (e.g. run_*_voracle_*_s1000_*).
SEED="${ORACLE_BASELINE_SEED:-1000}"
TASK="${ORACLE_BASELINE_TASK:-push-v3}"
OUTPUT_ROOT="${ORACLE_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/phase06_oracle_baseline}"
VIDEO="${ORACLE_BASELINE_VIDEO:-true}"
EPISODE_LENGTH="${ORACLE_BASELINE_EPISODE_LENGTH:-120}"
FPS="${ORACLE_BASELINE_FPS:-30}"
SAVE_FRAMES="${ORACLE_SAVE_FRAMES:-true}"
CAMERA_NAME="${ORACLE_METAWORLD_CAMERA_NAME:-corner2}"
FLIP_CORNER2="${ORACLE_FLIP_CORNER2:-true}"

log_info() { echo "[INFO] $*"; }
log_warn() { echo "[WARN] $*" >&2; }
log_error() { echo "[ERROR] $*" >&2; }

assert_non_negative_int() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    log_error "Invalid ${name}: ${value}. Expected non-negative integer."
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
    --episodes)
      EPISODES="${2}"
      shift 2
      ;;
    --seed)
      SEED="${2}"
      shift 2
      ;;
    --task)
      TASK="${2}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2}"
      shift 2
      ;;
    --video)
      VIDEO="${2}"
      shift 2
      ;;
    --episode-length)
      EPISODE_LENGTH="${2}"
      shift 2
      ;;
    --fps)
      FPS="${2}"
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
    *)
      log_error "Unknown arg: ${1}"
      exit 2
      ;;
  esac
done

assert_non_negative_int "episodes" "${EPISODES}"
assert_non_negative_int "seed" "${SEED}"
assert_non_negative_int "episode_length" "${EPISODE_LENGTH}"
assert_non_negative_int "fps" "${FPS}"
assert_bool "video" "${VIDEO}"
assert_bool "save_frames" "${SAVE_FRAMES}"
assert_bool "flip_corner2" "${FLIP_CORNER2}"
if [[ -z "${CAMERA_NAME}" ]]; then
  log_error "Invalid camera_name: must be non-empty."
  exit 2
fi

if [[ ! -x "${LEROBOT_ENV_DIR}/bin/python" ]]; then
  log_error "Python not found in env: ${LEROBOT_ENV_DIR}/bin/python"
  exit 3
fi
if [[ ! -f "${ORACLE_EVAL_SCRIPT}" ]]; then
  log_error "Oracle eval script missing: ${ORACLE_EVAL_SCRIPT}"
  exit 3
fi

mkdir -p "${OUTPUT_ROOT}"
timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
task_slug="$(printf '%s' "${TASK}" | tr -c 'A-Za-z0-9' '_' | tr '[:upper:]' '[:lower:]')"
run_prefix_raw="${ORACLE_RUN_PREFIX:-${RUN_NAME_PREFIX:-}}"
run_prefix_slug=""
if [[ -n "${run_prefix_raw}" ]]; then
  run_prefix_slug="$(printf '%s' "${run_prefix_raw}" | tr -c 'A-Za-z0-9' '_' | tr '[:upper:]' '[:lower:]' | sed 's/__*/_/g' | sed 's/^_\|_$//g')"
fi
output_dir=""
for _ in $(seq 1 10); do
  nonce="$(date +%s%N | tail -c 7)"
  if [[ -n "${run_prefix_slug}" ]]; then
    candidate="${OUTPUT_ROOT}/${run_prefix_slug}_run_${timestamp}_ep${EPISODES}_voracle_t${task_slug}_s${SEED}_r${nonce}"
  else
    candidate="${OUTPUT_ROOT}/run_${timestamp}_ep${EPISODES}_voracle_t${task_slug}_s${SEED}_r${nonce}"
  fi
  if mkdir "${candidate}" 2>/dev/null; then
    output_dir="${candidate}"
    break
  fi
done
if [[ -z "${output_dir}" ]]; then
  log_error "Failed to create unique run output directory under ${OUTPUT_ROOT}"
  exit 3
fi

run_cmd="'${LEROBOT_ENV_DIR}/bin/python' '${ORACLE_EVAL_SCRIPT}' \
  --task '${TASK}' \
  --episodes ${EPISODES} \
  --seed ${SEED} \
  --max-steps ${EPISODE_LENGTH} \
  --video '${VIDEO}' \
  --fps ${FPS} \
  --camera-name '${CAMERA_NAME}' \
  --flip-corner2 '${FLIP_CORNER2}' \
  --save-frames '${SAVE_FRAMES}' \
  --output-dir '${output_dir}'"

log_info "Running oracle baseline task=${TASK}: episodes=${EPISODES}, seed=${SEED}, episode_length=${EPISODE_LENGTH}, video=${VIDEO}, fps=${FPS}, camera_name=${CAMERA_NAME}, flip_corner2=${FLIP_CORNER2}, save_frames=${SAVE_FRAMES}"
xvfb_enabled="$(printf '%s' "${ORACLE_XVFB_ENABLED}" | tr '[:upper:]' '[:lower:]')"
if [[ "${xvfb_enabled}" == "1" || "${xvfb_enabled}" == "true" || "${xvfb_enabled}" == "yes" ]]; then
  xvfb-run -a -s "-screen 0 1280x1024x24" bash -lc "${run_cmd}"
else
  log_warn "ORACLE_XVFB_ENABLED=${ORACLE_XVFB_ENABLED}; running without xvfb-run."
  bash -lc "${run_cmd}"
fi

echo "Baseline eval output directory: ${output_dir}"
