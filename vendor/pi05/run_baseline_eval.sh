#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SMOLVLA_SCRIPT_DIR="${SMOLVLA_SCRIPT_DIR:-${SCRIPT_DIR}}"
SMOLVLA_REPO_ROOT="${SMOLVLA_REPO_ROOT:-${REPO_ROOT}}"
SMOLVLA_WORKSPACE_ROOT="${SMOLVLA_WORKSPACE_ROOT:-$(cd "${SMOLVLA_REPO_ROOT}/.." && pwd)}"
SMOLVLA_EVAL_RUNNER="${SMOLVLA_REPO_ROOT}/scripts/lerobot_eval_full_videos.py"

if [[ -f "${SCRIPT_DIR}/config.sh" ]]; then
  source "${SCRIPT_DIR}/config.sh"
fi

if [[ -f "${SCRIPT_DIR}/common.sh" ]]; then
  source "${SCRIPT_DIR}/common.sh"
fi

if ! declare -F log_info >/dev/null; then
  log_info() {
    echo "[INFO] $*"
  }
fi
if ! declare -F log_warn >/dev/null; then
  log_warn() {
    echo "[WARN] $*"
  }
fi
if ! declare -F log_error >/dev/null; then
  log_error() {
    echo "[ERROR] $*" >&2
  }
fi

SMOLVLA_BASELINE_EPISODES="${SMOLVLA_BASELINE_EPISODES:-15}"
SMOLVLA_BASELINE_SEED="${SMOLVLA_BASELINE_SEED:-123}"
SMOLVLA_BASELINE_DEVICE="${SMOLVLA_BASELINE_DEVICE:-auto}"
SMOLVLA_BASELINE_VIDEO="${SMOLVLA_BASELINE_VIDEO:-true}"
SMOLVLA_BASELINE_USE_AMP="${SMOLVLA_BASELINE_USE_AMP:-false}"
SMOLVLA_BASELINE_EPISODE_LENGTH="${SMOLVLA_BASELINE_EPISODE_LENGTH:-400}"
SMOLVLA_BASELINE_VIDEO_LENGTH="${SMOLVLA_BASELINE_VIDEO_LENGTH:-220}"
SMOLVLA_BASELINE_VIDEO_INTERVAL="${SMOLVLA_BASELINE_VIDEO_INTERVAL:-2}"
SMOLVLA_BASELINE_TASK="${SMOLVLA_BASELINE_TASK:-push-v3}"
SMOLVLA_ARTIFACT_ROOT="${SMOLVLA_ARTIFACT_ROOT:-${SMOLVLA_REPO_ROOT}/artifacts}"
SMOLVLA_INIT_CHECKPOINT="${SMOLVLA_INIT_CHECKPOINT:-jadechoghari/smolvla_metaworld}"
SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${SMOLVLA_WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"
SMOLVLA_XVFB_ENABLED="${SMOLVLA_XVFB_ENABLED:-1}"

episodes="${SMOLVLA_BASELINE_EPISODES}"
seed="${SMOLVLA_BASELINE_SEED}"
device="${SMOLVLA_BASELINE_DEVICE}"
video="${SMOLVLA_BASELINE_VIDEO}"
use_amp="${SMOLVLA_BASELINE_USE_AMP}"
episode_length="${SMOLVLA_BASELINE_EPISODE_LENGTH}"
video_length="${SMOLVLA_BASELINE_VIDEO_LENGTH}"
video_interval="${SMOLVLA_BASELINE_VIDEO_INTERVAL}"
task="${SMOLVLA_BASELINE_TASK}"
output_root="${SMOLVLA_ARTIFACT_ROOT}/phase06_baseline"
checkpoint="${SMOLVLA_INIT_CHECKPOINT}"

baseline_workdir="${SMOLVLA_SCRIPT_DIR:-${SMOLVLA_REPO_ROOT}}"
if [[ ! -d "${baseline_workdir}" ]]; then
  baseline_workdir="${SMOLVLA_REPO_ROOT}"
fi
if [[ ! -d "${baseline_workdir}" ]]; then
  baseline_workdir="${SCRIPT_DIR}"
fi

while [[ $# -gt 0 ]]; do
  case "${1}" in
    --episodes)
      episodes="${2}"
      shift 2
      ;;
    --seed)
      seed="${2}"
      shift 2
      ;;
    --device)
      device="${2}"
      shift 2
      ;;
    --use-amp)
      use_amp="${2}"
      shift 2
      ;;
    --episode-length)
      episode_length="${2}"
      shift 2
      ;;
    --video)
      video="${2}"
      shift 2
      ;;
    --video-length)
      video_length="${2}"
      shift 2
      ;;
    --video-interval)
      video_interval="${2}"
      shift 2
      ;;
    --task)
      task="${2}"
      shift 2
      ;;
    --output-root)
      output_root="${2}"
      shift 2
      ;;
    --checkpoint)
      checkpoint="${2}"
      shift 2
      ;;
    *)
      echo "Unknown arg: ${1}" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${device}" || "${device}" == "auto" ]]; then
  if [[ -x "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" ]]; then
    resolved_device="$(${SMOLVLA_LEROBOT_ENV_DIR}/bin/python - <<PY
import torch
if torch.cuda.is_available():
    print("cuda")
else:
    print("cpu")
PY
    )"
    device="${resolved_device}"
  else
    device="cpu"
  fi
fi

mkdir -p "${output_root}"
timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
task_slug="$(printf '%s' "${task}" | tr -c 'A-Za-z0-9' '_' | tr '[:upper:]' '[:lower:]')"
seed_suffix="${seed}"
max_tries=10
output_dir=""
for _ in $(seq 1 "${max_tries}"); do
  nonce="$(date +%s%N | tail -c 7)"
  candidate="${output_root}/run_${timestamp}_ep${episodes}_v${video}_t${task_slug}_s${seed_suffix}_r${nonce}"
  if mkdir "${candidate}" 2>/dev/null; then
    output_dir="${candidate}"
    break
  fi
done
if [[ -z "${output_dir}" ]]; then
  log_error "Failed to create unique run output directory under ${output_root}"
  exit 3
fi
if [[ ! -x "${SMOLVLA_EVAL_RUNNER}" ]]; then
  log_error "Eval runner wrapper is not executable: ${SMOLVLA_EVAL_RUNNER}"
  exit 3
fi

tmp_dir="$(mktemp -d)"
tmp_cleanup() {
  rm -rf "${tmp_dir}"
}
trap tmp_cleanup EXIT
site_packages="$(${SMOLVLA_LEROBOT_ENV_DIR}/bin/python - <<'PY'
import site
print(":".join(site.getsitepackages()))
PY
)"
cat > "${tmp_dir}/sitecustomize.py" <<'PY'
import importlib.util
import pathlib
import site
import sys


def _restore_datasets_module() -> None:
    # Workaround for a known import-shadowing issue where a local `lerobot.datasets`
    # package can mask Hugging Face's external `datasets` package during policy import.
    candidates = []
    for site_dir in site.getsitepackages() + [site.getusersitepackages() or ""]:
        if not site_dir:
            continue
        candidates.append(pathlib.Path(site_dir) / "datasets" / "__init__.py")
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("datasets", str(candidate))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.__file__ = str(candidate)
                sys.modules["datasets"] = module
                break


_restore_datasets_module()
PY

baseline_cmd="source '${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate' && \
  cd '${baseline_workdir}' && \
  SMOLVLA_MAX_EPISODES_RENDERED='${episodes}' \
  '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' '${SMOLVLA_EVAL_RUNNER}' \
  --policy.type smolvla \
  --policy.pretrained_path ${checkpoint} \
  --policy.load_vlm_weights true \
  --policy.vlm_model_name HuggingFaceTB/SmolVLA2-500M-Instruct \
  --policy.expert_width_multiplier 0.5 \
  --policy.self_attn_every_n_layers 0 \
  --policy.n_action_steps 1 \
  --env.type metaworld \
  --env.task '${task}' \
  --eval.n_episodes ${episodes} \
  --eval.batch_size 1 \
  --eval.use_async_envs false \
  --env.episode_length ${episode_length} \
  --policy.device ${device} \
  --policy.use_amp ${use_amp} \
  --env.multitask_eval false \
  --output_dir '${output_dir}' \
  --seed ${seed}"

log_info "Running baseline task=${task} eval: episodes=${episodes}, episode_length=${episode_length}, device=${device}, use_amp=${use_amp}, seed=${seed}, video=${video}, checkpoint=${checkpoint}, workdir=${baseline_workdir}"
xvfb_enabled="$(printf '%s' "${SMOLVLA_XVFB_ENABLED}" | tr '[:upper:]' '[:lower:]')"
if [[ "${xvfb_enabled}" == "1" || "${xvfb_enabled}" == "true" || "${xvfb_enabled}" == "yes" ]]; then
  xvfb-run -a -s "-screen 0 1280x1024x24" env \
    LEAKY=1 \
    UV_INSECURE_HOST=http://localhost:9999 \
    MH_REPO=metaworld-v2 \
    METAWORLD_RENDER_MODE=rgb_array \
    PYTHONPATH="${tmp_dir}:${site_packages}:${PYTHONPATH:-}" \
    bash -lc "${baseline_cmd}"
else
  log_warn "SMOLVLA_XVFB_ENABLED=${SMOLVLA_XVFB_ENABLED}; running baseline eval without xvfb-run."
  env \
    LEAKY=1 \
    UV_INSECURE_HOST=http://localhost:9999 \
    MH_REPO=metaworld-v2 \
    METAWORLD_RENDER_MODE=rgb_array \
    PYTHONPATH="${tmp_dir}:${site_packages}:${PYTHONPATH:-}" \
    bash -lc "${baseline_cmd}"
fi

echo "Baseline eval output directory: ${output_dir}"
