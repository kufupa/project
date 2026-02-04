#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_BASELINE_SCRIPT="${PROJECT_ROOT}/vendor/pi05/run_baseline_eval.sh"
SUMMARIZE_SCRIPT="${SCRIPT_DIR}/summarize_pushv3_eval.py"
EXTRACTOR_SCRIPT="${SCRIPT_DIR}/extract_parquet_episode_video.py"

BASELINE_ARTIFACT_ROOT="${SMOLVLA_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"

EPISODES="${PUSHV3_EPISODES:-${SMOLVLA_BASELINE_EPISODES:-15}}"
SEED="${PUSHV3_SEED:-${SMOLVLA_BASELINE_SEED:-123}}"
TASK="${PUSHV3_TASK:-push-v3}"
CHECKPOINT="${PUSHV3_CHECKPOINT:-${SMOLVLA_INIT_CHECKPOINT:-jadechoghari/smolvla_metaworld}}"
OUTPUT_ROOT="${PUSHV3_OUTPUT_ROOT:-${BASELINE_ARTIFACT_ROOT}/phase06_baseline}"
TOP_K="${PUSHV3_TOP_K:-5}"
VIDEO="${PUSHV3_VIDEO:-true}"
DEVICE="${PUSHV3_DEVICE:-auto}"
EPISODE_LENGTH="${PUSHV3_EPISODE_LENGTH:-400}"
VIDEO_LENGTH="${PUSHV3_VIDEO_LENGTH:-220}"
VIDEO_INTERVAL="${PUSHV3_VIDEO_INTERVAL:-2}"
DRY_RUN="${PUSHV3_DRY_RUN:-false}"
DATASET_ROOT="${PUSHV3_DATASET_ROOT:-}"
SOURCE_EPISODES_ROOT="${PUSHV3_SOURCE_EPISODES_ROOT:-}"
FPS="${PUSHV3_EXPORT_FPS:-30}"

usage() {
  cat <<'EOF'
Usage: scripts/pushv3_data_pipeline.sh [options]

Run smolvla_metaworld push-v3 baseline eval, summarise top-k episodes, and export
trajectory videos.

Flags:
  --episodes N             Number of eval episodes
  --seed N                 Evaluation seed
  --task NAME              Task/group name (default: push-v3)
  --checkpoint CKPT        checkpoint / HF path
  --output-root PATH       Baseline artifact output root
  --top-k N                Number of top episodes to export
  --dataset-root PATH      Optional parquet dataset root for renderer fallback
  --source-episodes-root P Optional episode_*.pt directory for extractor matching
  --fps N                  FPS for rendered fallback video conversion
  --device NAME            smolvla device override (auto/cpu/cuda)
  --video true|false       Whether baseline writes videos
  --episode-length N
  --video-length N
  --video-interval N
  --dry-run                Parse and print commands only
  --help                   Show this help

Environment variables:
  PUSHV3_EPISODES, PUSHV3_SEED, PUSHV3_TASK, PUSHV3_CHECKPOINT,
  PUSHV3_OUTPUT_ROOT, PUSHV3_TOP_K, PUSHV3_DRY_RUN.
EOF
}

log_info() { echo "[INFO] $*"; }
log_warn() { echo "[WARN] $*" >&2; }
log_error() { echo "[ERROR] $*" >&2; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    log_error "Missing dependency: $1"
    exit 3
  }
}

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
    --checkpoint)
      CHECKPOINT="${2}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2}"
      shift 2
      ;;
    --top-k)
      TOP_K="${2}"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="${2}"
      shift 2
      ;;
    --source-episodes-root)
      SOURCE_EPISODES_ROOT="${2}"
      shift 2
      ;;
    --fps)
      FPS="${2}"
      shift 2
      ;;
    --device)
      DEVICE="${2}"
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
    --video-length)
      VIDEO_LENGTH="${2}"
      shift 2
      ;;
    --video-interval)
      VIDEO_INTERVAL="${2}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      log_error "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
done

assert_non_negative_int "episodes" "${EPISODES}"
assert_non_negative_int "seed" "${SEED}"
assert_non_negative_int "top_k" "${TOP_K}"
assert_non_negative_int "episode_length" "${EPISODE_LENGTH}"
assert_non_negative_int "video_length" "${VIDEO_LENGTH}"
assert_non_negative_int "video_interval" "${VIDEO_INTERVAL}"
assert_non_negative_int "fps" "${FPS}"
assert_bool "video" "${VIDEO}"
assert_bool "dry_run" "${DRY_RUN}"

if [[ "${TASK}" == "" ]]; then
  log_error "TASK must be set to a valid environment task name."
  exit 2
fi

require_cmd python3
require_cmd awk
if [[ ! -x "${RUN_BASELINE_SCRIPT}" ]]; then
  log_error "Baseline runner not found or not executable: ${RUN_BASELINE_SCRIPT}"
  exit 3
fi
if [[ ! -f "${SUMMARIZE_SCRIPT}" ]]; then
  log_error "Summarizer script not found: ${SUMMARIZE_SCRIPT}"
  exit 3
fi
if [[ -n "${DATASET_ROOT}" ]] && [[ ! -d "${DATASET_ROOT}" ]]; then
  log_error "dataset-root does not exist: ${DATASET_ROOT}"
  exit 3
fi
if [[ -n "${DATASET_ROOT}" ]] && [[ ! -x "${EXTRACTOR_SCRIPT}" ]]; then
  log_error "Extractor script not executable: ${EXTRACTOR_SCRIPT}"
  exit 3
fi

BASELINE_CMD=(
  "${RUN_BASELINE_SCRIPT}"
  --task "${TASK}"
  --episodes "${EPISODES}"
  --seed "${SEED}"
  --checkpoint "${CHECKPOINT}"
  --output-root "${OUTPUT_ROOT}"
  --video "${VIDEO}"
  --device "${DEVICE}"
  --episode-length "${EPISODE_LENGTH}"
  --video-length "${VIDEO_LENGTH}"
  --video-interval "${VIDEO_INTERVAL}"
)

log_info "Resolved configuration:"
log_info "  task=${TASK}"
log_info "  episodes=${EPISODES}"
log_info "  seed=${SEED}"
log_info "  checkpoint=${CHECKPOINT}"
log_info "  output_root=${OUTPUT_ROOT}"
log_info "  top_k=${TOP_K}"
log_info "  video=${VIDEO}"
log_info "  dataset_root=${DATASET_ROOT:-<none>}"
log_info "  dry_run=${DRY_RUN}"

if [[ "${DRY_RUN}" == "true" ]]; then
  log_info "Dry run enabled. The following command will not execute."
  log_info "  ${BASELINE_CMD[*]}"
  predicted_timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
  task_slug="$(printf '%s' "${TASK}" | tr -c 'A-Za-z0-9' '_' | tr '[:upper:]' '[:lower:]')"
  predicted_output="${OUTPUT_ROOT}/run_${predicted_timestamp}_ep${EPISODES}_v${VIDEO}_t${task_slug}_s${SEED}_r*"
  log_info "Expected eval output directory: ${predicted_output}"
  log_info "Skipping export/summarization."
  exit 0
fi

run_log="$(mktemp)"
log_info "Running baseline: ${BASELINE_CMD[*]}"
"${BASELINE_CMD[@]}" 2>&1 | tee "${run_log}"

eval_output_dir="$(awk '/Baseline eval output directory:/{print $NF}' "${run_log}" | tail -n 1 || true)"
rm -f "${run_log}"

if [[ -z "${eval_output_dir}" ]]; then
  log_error "Unable to detect baseline output directory from runner output."
  exit 3
fi

eval_output_dir="$(printf '%s' "${eval_output_dir}")"
log_info "Baseline output directory: ${eval_output_dir}"

eval_info="${eval_output_dir}/eval_info.json"
if [[ ! -f "${eval_info}" ]]; then
  log_error "Missing eval_info.json: ${eval_info}"
  exit 3
fi

optimal_report="${eval_output_dir}/optimal_report.json"
log_info "Writing optimal trajectory report to ${optimal_report}"
python3 "${SUMMARIZE_SCRIPT}" \
  --eval-info "${eval_info}" \
  --task "${TASK}" \
  --top-k "${TOP_K}" \
  --output "${optimal_report}"

if [[ ! -f "${optimal_report}" ]]; then
  log_error "Summarizer did not produce output: ${optimal_report}"
  exit 3
fi

python3 - "${optimal_report}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = payload.get("summary", {})
task = payload.get("resolved_task", payload.get("requested_task", "push-v3"))
mean_sum = summary.get("mean_sum_reward")
mean_max = summary.get("mean_max_reward")
success_rate = summary.get("success_rate_percent")
success_count = summary.get("success_count")
n_episodes = summary.get("n_episodes")
print(f"[SUMMARY] task={task}")
if mean_sum is not None:
    print(f"  reward mean: {mean_sum:.6f}")
if mean_max is not None:
    print(f"  max reward mean: {mean_max:.6f}")
if success_rate is not None:
    if success_count is not None:
        print(
            f"  success rate: {success_rate:.2f}% ({success_count}/{n_episodes or 0})"
        )
    else:
        print(f"  success rate: {success_rate:.2f}%")
PY

trajectories_dir="${eval_output_dir}/trajectories"
export_manifest="${trajectories_dir}/export_manifest.json"

log_info "Exporting top-${TOP_K} trajectories to ${trajectories_dir}"
python3 - "${optimal_report}" "${eval_output_dir}" "${trajectories_dir}" "${DATASET_ROOT}" "${EXTRACTOR_SCRIPT}" "${SOURCE_EPISODES_ROOT}" "${FPS}" <<'PY'
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

report_path = Path(sys.argv[1])
eval_output_dir = Path(sys.argv[2])
trajectories_dir = Path(sys.argv[3])
dataset_root = (sys.argv[4] or "").strip()
extractor_script = (sys.argv[5] or "").strip()
source_episodes_root = (sys.argv[6] or "").strip()
fps = int(sys.argv[7]) if len(sys.argv) > 7 else 30


def _parse_episode_video_index(path: Path) -> Optional[int]:
    stem = path.stem
    prefix = "eval_episode_"
    if not stem.startswith(prefix):
        return None
    suffix = stem[len(prefix) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _build_local_video_index(eval_root: Path) -> tuple[dict[int, Path], list[Path]]:
    videos_root = eval_root / "videos"
    if not videos_root.is_dir():
        return {}, []

    direct = {}
    ordered = []
    for path in videos_root.glob("**/eval_episode_*.mp4"):
        episode_index = _parse_episode_video_index(path)
        if episode_index is not None:
            direct.setdefault(episode_index, path)
        ordered.append(path)

    def _path_sort_key(p: Path) -> int:
        parsed = _parse_episode_video_index(p)
        return int(parsed) if parsed is not None else 2**31 + hash(p.as_posix())

    ordered.sort(key=_path_sort_key)
    return direct, ordered

payload = json.loads(report_path.read_text(encoding="utf-8"))
episodes = payload.get("episodes", [])

if not isinstance(episodes, list):
    raise RuntimeError(f"Invalid episodes payload in {report_path}")

trajectories_dir.mkdir(parents=True, exist_ok=True)
local_video_by_index, local_videos = _build_local_video_index(eval_output_dir)
exported = []
counts = {
    "copied": 0,
    "rendered": 0,
    "missing": 0,
    "failed": 0,
}

for episode in episodes:
    episode_index = episode.get("episode_index")
    if episode_index is None:
        continue
    try:
        episode_index = int(episode_index)
    except (TypeError, ValueError):
        continue
    rank = episode.get("rank", 0)
    max_reward = episode.get("max_reward")
    sum_reward = episode.get("sum_reward")
    success = episode.get("success")
    source_video = episode.get("video_path")

    output_path = trajectories_dir / f"trajectory_{int(rank):02d}_episode_{int(episode_index):04d}.mp4"
    status = "missing_video"

    if isinstance(source_video, str) and source_video.strip():
        source_path = Path(source_video).expanduser().resolve()
        if source_path.is_file():
            shutil.copy2(source_path, output_path)
            status = "copied"
            counts["copied"] += 1
        elif episode_index in local_video_by_index:
            status = "copied"
            shutil.copy2(local_video_by_index[episode_index], output_path)
            counts["copied"] += 1
        elif dataset_root:
            status = "render_requested"
        elif local_videos:
            # Fallback: when naming still follows sequential rollout order, map by episode index.
            if 0 <= episode_index < len(local_videos):
                fallback_path = local_videos[episode_index]
                if fallback_path.is_file():
                    shutil.copy2(fallback_path, output_path)
                    status = "copied"
                    counts["copied"] += 1
                else:
                    status = "missing_video"
            else:
                status = "missing_video"
        else:
            status = "missing_video"
    elif episode_index in local_video_by_index:
        shutil.copy2(local_video_by_index[episode_index], output_path)
        status = "copied"
        counts["copied"] += 1
    elif dataset_root and extractor_script:
        status = "render_requested"

    if status == "render_requested":
        render_cmd = [
            sys.executable,
            extractor_script,
            "--dataset-root",
            dataset_root,
            "--episode-index",
            str(episode_index),
            "--output",
            str(output_path),
            "--fps",
            str(fps),
        ]
        if source_episodes_root:
            render_cmd.extend(["--source-episodes-root", source_episodes_root])
        try:
            subprocess.run(render_cmd, check=True)
            status = "rendered"
            counts["rendered"] += 1
        except subprocess.CalledProcessError:
            status = "render_failed"
            counts["failed"] += 1
        except FileNotFoundError:
            status = "render_script_missing"
            counts["failed"] += 1

    if status not in ("copied", "rendered"):
        counts["missing"] += 1

    exported.append(
        {
            "rank": rank,
            "episode_index": episode_index,
            "max_reward": max_reward,
            "sum_reward": sum_reward,
            "success": success,
            "source_video": source_video,
            "export_video": str(output_path),
            "status": status,
        }
    )

manifest = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "task": payload.get("resolved_task", payload.get("requested_task", "push-v3")),
    "eval_info": payload.get("eval_info"),
    "output_dir": str(trajectories_dir),
    "top_k": payload.get("top_k"),
    "counts": counts,
    "episodes": exported,
}
manifest_path = trajectories_dir / "export_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Export manifest: {manifest_path}")
PY

log_info "Trajectory export manifest written: ${export_manifest}"
log_info "Done."
