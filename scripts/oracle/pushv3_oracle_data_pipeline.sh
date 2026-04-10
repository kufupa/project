#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_BASELINE_SCRIPT="${SCRIPT_DIR}/run_oracle_baseline_eval.sh"
SUMMARIZE_SCRIPT="${PROJECT_ROOT}/scripts/summarize_pushv3_eval.py"
EXTRACTOR_SCRIPT="${PROJECT_ROOT}/scripts/extract_parquet_episode_video.py"

BASELINE_ARTIFACT_ROOT="${ORACLE_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"

EPISODES="${PUSHV3_EPISODES:-${ORACLE_BASELINE_EPISODES:-15}}"
SEED="${PUSHV3_SEED:-${ORACLE_BASELINE_SEED:-1000}}"
TASK="${PUSHV3_TASK:-push-v3}"
OUTPUT_ROOT="${PUSHV3_OUTPUT_ROOT:-${BASELINE_ARTIFACT_ROOT}/phase06_oracle_baseline}"
TOP_K="${PUSHV3_TOP_K:-5}"
VIDEO="${PUSHV3_VIDEO:-true}"
SAVE_FRAMES="${ORACLE_SAVE_FRAMES:-true}"
EPISODE_LENGTH="${PUSHV3_EPISODE_LENGTH:-120}"
FPS="${PUSHV3_EXPORT_FPS:-30}"
DRY_RUN="${PUSHV3_DRY_RUN:-false}"
DATASET_ROOT="${PUSHV3_DATASET_ROOT:-}"
SOURCE_EPISODES_ROOT="${PUSHV3_SOURCE_EPISODES_ROOT:-}"
CAMERA_NAME="${ORACLE_METAWORLD_CAMERA_NAME:-corner2}"
FLIP_CORNER2="${ORACLE_FLIP_CORNER2:-true}"

usage() {
  cat <<'EOF'
Usage: scripts/oracle/pushv3_oracle_data_pipeline.sh [options]

Run Meta-World scripted oracle push-v3 eval, summarize top-k episodes, and export
trajectory videos.

Flags:
  --episodes N             Number of eval episodes
  --seed N                 Evaluation seed
  --task NAME              Task/group name (default: push-v3)
  --output-root PATH       Artifact output root
  --top-k N                Number of top episodes to export
  --dataset-root PATH      Optional parquet dataset root for renderer fallback
  --source-episodes-root P Optional episode_*.pt directory for extractor matching
  --fps N                  FPS for rendered fallback video conversion
  --video true|false       Whether baseline writes videos
  --save-frames true|false Whether to write frames/episode_XXXX PNGs (also ORACLE_SAVE_FRAMES)
  --episode-length N       Max steps per episode
  --camera-name NAME       Camera for baseline render (default: corner2)
  --flip-corner2 true|false Flip corner2 camera frames for parity
  --dry-run                Parse and print commands only
  --help                   Show this help
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
    --video)
      VIDEO="${2}"
      shift 2
      ;;
    --save-frames)
      SAVE_FRAMES="${2}"
      shift 2
      ;;
    --episode-length)
      EPISODE_LENGTH="${2}"
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
assert_non_negative_int "fps" "${FPS}"
assert_bool "video" "${VIDEO}"
assert_bool "save_frames" "${SAVE_FRAMES}"
assert_bool "flip_corner2" "${FLIP_CORNER2}"
if [[ -z "${CAMERA_NAME}" ]]; then
  log_error "camera-name must be non-empty"
  exit 2
fi
assert_bool "dry_run" "${DRY_RUN}"

if [[ -z "${TASK}" ]]; then
  log_error "TASK must be set to a valid environment task name."
  exit 2
fi

require_cmd python3
require_cmd awk
if [[ ! -x "${RUN_BASELINE_SCRIPT}" ]]; then
  log_error "Oracle baseline runner not found or not executable: ${RUN_BASELINE_SCRIPT}"
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
  --output-root "${OUTPUT_ROOT}"
  --video "${VIDEO}"
  --save-frames "${SAVE_FRAMES}"
  --episode-length "${EPISODE_LENGTH}"
  --camera-name "${CAMERA_NAME}"
  --flip-corner2 "${FLIP_CORNER2}"
  --fps "${FPS}"
)

log_info "Resolved configuration:"
log_info "  task=${TASK}"
log_info "  episodes=${EPISODES}"
log_info "  seed=${SEED}"
log_info "  output_root=${OUTPUT_ROOT}"
log_info "  top_k=${TOP_K}"
log_info "  video=${VIDEO}"
log_info "  save_frames=${SAVE_FRAMES}"
log_info "  camera_name=${CAMERA_NAME}"
log_info "  flip_corner2=${FLIP_CORNER2}"
log_info "  dataset_root=${DATASET_ROOT:-<none>}"
log_info "  dry_run=${DRY_RUN}"

if [[ "${DRY_RUN}" == "true" ]]; then
  log_info "Dry run enabled. The following command will not execute."
  log_info "  ${BASELINE_CMD[*]}"
  predicted_timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
  task_slug="$(printf '%s' "${TASK}" | tr -c 'A-Za-z0-9' '_' | tr '[:upper:]' '[:lower:]')"
  predicted_output="${OUTPUT_ROOT}/run_${predicted_timestamp}_ep${EPISODES}_voracle_t${task_slug}_s${SEED}_r*"
  log_info "Expected eval output directory: ${predicted_output}"
  log_info "Skipping export/summarization."
  exit 0
fi

run_log="$(mktemp)"
log_info "Running oracle baseline: ${BASELINE_CMD[*]}"
"${BASELINE_CMD[@]}" 2>&1 | tee "${run_log}"

eval_output_dir="$(awk '/Baseline eval output directory:/{print $NF}' "${run_log}" | tail -n 1 || true)"
rm -f "${run_log}"

if [[ -z "${eval_output_dir}" ]]; then
  log_error "Unable to detect baseline output directory from runner output."
  exit 3
fi
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

report_path = Path(sys.argv[1])
eval_output_dir = Path(sys.argv[2])
trajectories_dir = Path(sys.argv[3])
dataset_root = (sys.argv[4] or "").strip()
extractor_script = (sys.argv[5] or "").strip()
source_episodes_root = (sys.argv[6] or "").strip()
fps = int(sys.argv[7]) if len(sys.argv) > 7 else 30


def _parse_episode_video_index(path: Path):
    stem = path.stem
    if not stem.startswith("eval_episode_"):
        return None
    suffix = stem[len("eval_episode_") :]
    return int(suffix) if suffix.isdigit() else None


def _build_local_video_index(eval_root: Path):
    videos_root = eval_root / "videos"
    if not videos_root.is_dir():
        return {}, []
    direct = {}
    ordered = []
    for path in videos_root.glob("**/eval_episode_*.mp4"):
        parsed = _parse_episode_video_index(path)
        if parsed is not None:
            direct.setdefault(parsed, path)
        ordered.append(path)
    ordered.sort(key=lambda p: _parse_episode_video_index(p) if _parse_episode_video_index(p) is not None else 2**31)
    return direct, ordered


payload = json.loads(report_path.read_text(encoding="utf-8"))
episodes = payload.get("episodes", [])
if not isinstance(episodes, list):
    raise RuntimeError(f"Invalid episodes payload in {report_path}")

trajectories_dir.mkdir(parents=True, exist_ok=True)
local_video_by_index, local_videos = _build_local_video_index(eval_output_dir)
exported = []
counts = {"copied": 0, "rendered": 0, "missing": 0, "failed": 0}

for episode in episodes:
    ep_idx = episode.get("episode_index")
    if ep_idx is None:
        continue
    try:
        ep_idx = int(ep_idx)
    except (TypeError, ValueError):
        continue

    rank = int(episode.get("rank", 0))
    source_video = episode.get("video_path")
    output_path = trajectories_dir / f"trajectory_{rank:02d}_episode_{ep_idx:04d}.mp4"
    status = "missing_video"

    if isinstance(source_video, str) and source_video.strip():
        source_path = Path(source_video).expanduser().resolve()
        if source_path.is_file():
            shutil.copy2(source_path, output_path)
            status = "copied"
            counts["copied"] += 1
        elif ep_idx in local_video_by_index:
            shutil.copy2(local_video_by_index[ep_idx], output_path)
            status = "copied"
            counts["copied"] += 1
        elif dataset_root and extractor_script:
            status = "render_requested"
        elif 0 <= ep_idx < len(local_videos):
            fallback = local_videos[ep_idx]
            if fallback.is_file():
                shutil.copy2(fallback, output_path)
                status = "copied"
                counts["copied"] += 1
    elif ep_idx in local_video_by_index:
        shutil.copy2(local_video_by_index[ep_idx], output_path)
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
            str(ep_idx),
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
        except Exception:
            status = "render_failed"
            counts["failed"] += 1

    if status not in ("copied", "rendered"):
        counts["missing"] += 1

    exported.append(
        {
            "rank": rank,
            "episode_index": ep_idx,
            "max_reward": episode.get("max_reward"),
            "sum_reward": episode.get("sum_reward"),
            "success": episode.get("success"),
            "source_video": source_video,
            "export_video": str(output_path),
            "status": status,
        }
    )

run_manifest_path = eval_output_dir / "run_manifest.json"
run_manifest_rel = (
    str(run_manifest_path.relative_to(eval_output_dir))
    if run_manifest_path.is_file()
    else None
)

manifest = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "task": payload.get("resolved_task", payload.get("requested_task", "push-v3")),
    "eval_info": payload.get("eval_info"),
    "run_manifest": run_manifest_rel,
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

python3 - "${eval_output_dir}" "${optimal_report}" "${export_manifest}" "${TOP_K}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
optimal = Path(sys.argv[2]).resolve()
export_manifest = Path(sys.argv[3]).resolve()
top_k = int(sys.argv[4])

rm_path = run_dir / "run_manifest.json"
if not rm_path.is_file():
    sys.exit(0)


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(run_dir))
    except ValueError:
        return str(p)


data = json.loads(rm_path.read_text(encoding="utf-8"))
data["pipeline"] = {
    "top_k": top_k,
    "optimal_report": _rel(optimal),
    "trajectories_export_manifest": _rel(export_manifest),
}
rm_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(f"Updated run_manifest.json: {rm_path}")
PY

log_info "Done."
