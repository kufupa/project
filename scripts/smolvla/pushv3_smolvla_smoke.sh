#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
SMOLVLA_PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${SMOLVLA_PYTHON_BIN:-${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}/bin/python}"
CHECKPOINT="${SMOLVLA_INIT_CHECKPOINT:-jadechoghari/smolvla_metaworld}"
OUTPUT_ROOT="${SMOLVLA_SMOKE_OUTPUT_ROOT:-${SMOLVLA_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}/phase07_smolvla_baseline}"
MIN_VIDEO_BYTES="${SMOLVLA_MIN_VIDEO_BYTES:-1024}"
TASK="push-v3"
SEED="${SMOLVLA_SMOKE_SEED:-1000}"
EPISODES=1
EVAL_MODE="${SMOLVLA_EVAL_MODE:-parity}"

case "${EVAL_MODE}" in
  parity)
    MAX_STEPS="${SMOLVLA_SMOKE_MAX_STEPS:-120}"
    CAMERA_NAME="${SMOLVLA_METAWORLD_CAMERA_NAME:-corner2}"
    FLIP_CORNER2="${SMOLVLA_FLIP_CORNER2:-true}"
    LOAD_VLM_WEIGHTS="${SMOLVLA_LOAD_VLM_WEIGHTS:-true}"
    ;;
  fast)
    MAX_STEPS="${SMOLVLA_SMOKE_MAX_STEPS:-120}"
    CAMERA_NAME="${SMOLVLA_METAWORLD_CAMERA_NAME:-corner2}"
    FLIP_CORNER2="${SMOLVLA_FLIP_CORNER2:-true}"
    LOAD_VLM_WEIGHTS="${SMOLVLA_LOAD_VLM_WEIGHTS:-false}"
    ;;
  *)
    echo "error: SMOLVLA_EVAL_MODE must be parity or fast (got: ${EVAL_MODE})" >&2
    exit 2
    ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

RUN_DIR="$(
  PYTHONPATH="${SMOLVLA_PYTHONPATH}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  TASK="${TASK}" \
  SEED="${SEED}" \
  EPISODES="${EPISODES}" \
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

from src.smolvla_pipeline.run_layout import ensure_unique_run_dir

run_dir = ensure_unique_run_dir(
    Path(os.environ["OUTPUT_ROOT"]),
    episodes=int(os.environ["EPISODES"]),
    task=os.environ["TASK"],
    seed=int(os.environ["SEED"]),
    variant="smolvla",
)
print(str(run_dir))
PY
)"

if command -v xvfb-run >/dev/null 2>&1; then
  xvfb-run -a -s "-screen 0 1280x1024x24" \
    env PYTHONPATH="${SMOLVLA_PYTHONPATH}" \
    SMOLVLA_METAWORLD_CAMERA_NAME="${CAMERA_NAME}" \
    SMOLVLA_FLIP_CORNER2="${FLIP_CORNER2}" \
    SMOLVLA_LOAD_VLM_WEIGHTS="${LOAD_VLM_WEIGHTS}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/smolvla/run_metaworld_smolvla_eval.py" \
      --task "${TASK}" \
      --episodes "${EPISODES}" \
      --seed "${SEED}" \
      --checkpoint "${CHECKPOINT}" \
      --output-dir "${RUN_DIR}" \
      --video true \
      --overlay-mode cumulative_reward \
      --max-steps "${MAX_STEPS}"
else
  PYTHONPATH="${SMOLVLA_PYTHONPATH}" \
  SMOLVLA_METAWORLD_CAMERA_NAME="${CAMERA_NAME}" \
  SMOLVLA_FLIP_CORNER2="${FLIP_CORNER2}" \
  SMOLVLA_LOAD_VLM_WEIGHTS="${LOAD_VLM_WEIGHTS}" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/smolvla/run_metaworld_smolvla_eval.py" \
    --task "${TASK}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${RUN_DIR}" \
    --video true \
    --overlay-mode cumulative_reward \
    --max-steps "${MAX_STEPS}"
fi

PYTHONPATH="${SMOLVLA_PYTHONPATH}" \
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/smolvla/verify_smolvla_run_artifacts.py" \
  --run-dir "${RUN_DIR}" \
  --task "${TASK}" \
  --episodes "${EPISODES}" \
  --require-video true \
  --min-video-bytes "${MIN_VIDEO_BYTES}"

PYTHONPATH="${SMOLVLA_PYTHONPATH}" \
RUN_DIR="${RUN_DIR}" \
TASK="${TASK}" \
SEED="${SEED}" \
EPISODES="${EPISODES}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"]).resolve()
task = os.environ["TASK"]
seed = int(os.environ["SEED"])
episodes = int(os.environ["EPISODES"])

eval_info = json.loads((run_dir / "eval_info.json").read_text(encoding="utf-8"))
overall = eval_info.get("overall", {})
video_paths = overall.get("video_paths", [])
if not isinstance(video_paths, list):
    video_paths = []
video_paths = [path for path in video_paths if isinstance(path, str) and path.strip()]

video_path = video_paths[0] if video_paths else None
video_recorded = len(video_paths) == episodes
episodes_ok = int(overall.get("n_episodes", 0)) == episodes
pc_success = float(overall.get("pc_success", 0.0))
status = "success" if ("overall" in eval_info and episodes_ok and video_recorded and pc_success > 0.0) else "failed"

summary = {
    "status": status,
    "task": task,
    "seed": seed,
    "episodes": episodes,
    "video_recorded": video_recorded,
    "video_path": video_path,
    "run_dir": str(run_dir),
}

summary_path = run_dir / "smoke_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
