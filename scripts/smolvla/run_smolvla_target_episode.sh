#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

PYTHON_BIN="${SMOLVLA_PYTHON_BIN:-${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}/bin/python}"
CHECKPOINT="${SMOLVLA_INIT_CHECKPOINT:-jadechoghari/smolvla_metaworld}"
OUTPUT_ROOT="${SMOLVLA_TOPK_OUTPUT_ROOT:-${SMOLVLA_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}/phase07_smolvla_baseline}"
MIN_VIDEO_BYTES="${SMOLVLA_MIN_VIDEO_BYTES:-1024}"
EVAL_MODE="${SMOLVLA_EVAL_MODE:-parity}"
EPISODES_PER_TARGET="${SMOLVLA_EPISODES_PER_TARGET:-1}"
SAVE_FRAMES="${SMOLVLA_SAVE_FRAMES:-false}"

case "${EVAL_MODE}" in
  parity)
    MAX_STEPS="${SMOLVLA_TARGET_MAX_STEPS:-120}"
    CAMERA_NAME="${SMOLVLA_METAWORLD_CAMERA_NAME:-corner2}"
    FLIP_CORNER2="${SMOLVLA_FLIP_CORNER2:-true}"
    LOAD_VLM_WEIGHTS="${SMOLVLA_LOAD_VLM_WEIGHTS:-true}"
    ;;
  fast)
    MAX_STEPS="${SMOLVLA_TARGET_MAX_STEPS:-120}"
    CAMERA_NAME="${SMOLVLA_METAWORLD_CAMERA_NAME:-corner2}"
    FLIP_CORNER2="${SMOLVLA_FLIP_CORNER2:-true}"
    LOAD_VLM_WEIGHTS="${SMOLVLA_LOAD_VLM_WEIGHTS:-false}"
    ;;
  *)
    echo "error: SMOLVLA_EVAL_MODE must be parity or fast (got: ${EVAL_MODE})" >&2
    exit 2
    ;;
esac

TARGETS_JSON="${1:-${TARGETS_JSON:-}}"
TASK_INDEX="${2:-${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}}"
CAMPAIGN_DIR="${SMOLVLA_CAMPAIGN_DIR:-}"

if [[ -z "${TARGETS_JSON}" || -z "${TASK_INDEX}" ]]; then
  echo "usage: $0 <targets_json> <task_index>" >&2
  echo "or set TARGETS_JSON and TASK_INDEX/SLURM_ARRAY_TASK_ID env vars." >&2
  exit 2
fi

if [[ ! -f "${TARGETS_JSON}" ]]; then
  echo "error: targets file not found: ${TARGETS_JSON}" >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

TARGET_INFO="$(
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
  TARGETS_JSON="${TARGETS_JSON}" \
  TASK_INDEX="${TASK_INDEX}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["TARGETS_JSON"]).read_text(encoding="utf-8"))
targets = payload.get("targets")
if not isinstance(targets, list):
    raise ValueError("targets JSON must contain a list under 'targets'.")

task_index = int(os.environ["TASK_INDEX"])
if task_index < 0 or task_index >= len(targets):
    raise IndexError(f"task index out of range: {task_index} (targets={len(targets)})")

row = targets[task_index]
required = ("task", "reset_seed", "episode_index")
for key in required:
    if key not in row:
        raise ValueError(f"target row missing required key: {key}")

print(str(row["task"]))
print(str(int(row["reset_seed"])))
print(str(int(row["episode_index"])))
print(str(int(row.get("rank", task_index + 1))))
PY
)"
readarray -t TARGET_FIELDS <<<"${TARGET_INFO}"

TASK="${TARGET_FIELDS[0]}"
SEED="${TARGET_FIELDS[1]}"
EPISODE_INDEX="${TARGET_FIELDS[2]}"
RANK="${TARGET_FIELDS[3]}"

# Pin eval episodes to single oracle target identity.
export SMOLVLA_TARGET_EPISODE_INDEX="${EPISODE_INDEX}"
export SMOLVLA_FIXED_RESET_SEED="${SEED}"

RUN_DIR="$(
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  TASK="${TASK}" \
  SEED="${SEED}" \
  EPISODES="${EPISODES_PER_TARGET}" \
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

from src.smolvla_pipeline.run_layout import ensure_unique_run_dir

run_dir = ensure_unique_run_dir(
    Path(os.environ["OUTPUT_ROOT"]),
    episodes=int(os.environ["EPISODES"]),
    task=os.environ["TASK"],
    seed=int(os.environ["SEED"]),
    variant="smolvla_target",
)
print(str(run_dir))
PY
)"

if command -v xvfb-run >/dev/null 2>&1; then
  xvfb-run -a -s "-screen 0 1280x1024x24" \
    env PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    SMOLVLA_METAWORLD_CAMERA_NAME="${CAMERA_NAME}" \
    SMOLVLA_FLIP_CORNER2="${FLIP_CORNER2}" \
    SMOLVLA_LOAD_VLM_WEIGHTS="${LOAD_VLM_WEIGHTS}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/smolvla/run_metaworld_smolvla_eval.py" \
      --task "${TASK}" \
      --episodes "${EPISODES_PER_TARGET}" \
      --seed "${SEED}" \
      --checkpoint "${CHECKPOINT}" \
      --output-dir "${RUN_DIR}" \
      --video true \
      --overlay-mode cumulative_reward \
      --save-frames "${SAVE_FRAMES}" \
      --max-steps "${MAX_STEPS}"
else
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
  SMOLVLA_METAWORLD_CAMERA_NAME="${CAMERA_NAME}" \
  SMOLVLA_FLIP_CORNER2="${FLIP_CORNER2}" \
  SMOLVLA_LOAD_VLM_WEIGHTS="${LOAD_VLM_WEIGHTS}" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/smolvla/run_metaworld_smolvla_eval.py" \
    --task "${TASK}" \
      --episodes "${EPISODES_PER_TARGET}" \
    --seed "${SEED}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${RUN_DIR}" \
    --video true \
    --overlay-mode cumulative_reward \
      --save-frames "${SAVE_FRAMES}" \
    --max-steps "${MAX_STEPS}"
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/smolvla/verify_smolvla_run_artifacts.py" \
  --run-dir "${RUN_DIR}" \
  --task "${TASK}" \
  --episodes "${EPISODES_PER_TARGET}" \
  --require-video true \
  --min-video-bytes "${MIN_VIDEO_BYTES}"

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
RUN_DIR="${RUN_DIR}" \
TARGETS_JSON="${TARGETS_JSON}" \
TASK_INDEX="${TASK_INDEX}" \
TASK="${TASK}" \
SEED="${SEED}" \
EPISODE_INDEX="${EPISODE_INDEX}" \
RANK="${RANK}" \
CAMPAIGN_DIR="${CAMPAIGN_DIR}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"]).resolve()
targets_json = Path(os.environ["TARGETS_JSON"]).resolve()
task_index = int(os.environ["TASK_INDEX"])

summary = {
    "task_index": task_index,
    "rank": int(os.environ["RANK"]),
    "task": os.environ["TASK"],
    "seed": int(os.environ["SEED"]),
    "episode_index": int(os.environ["EPISODE_INDEX"]),
    "targets_json": str(targets_json),
    "run_dir": str(run_dir),
}

summary_path = run_dir / "target_episode_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

campaign_dir_raw = os.environ.get("CAMPAIGN_DIR", "").strip()
if campaign_dir_raw:
    campaign_dir = Path(campaign_dir_raw).resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    task_summary_path = campaign_dir / f"task_{task_index:04d}.json"
    task_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
PY
