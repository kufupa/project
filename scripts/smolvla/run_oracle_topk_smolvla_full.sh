#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

ORACLE_RUN_DIR="${ORACLE_RUN_DIR:-/vol/bitbucket/aa6622/project/artifacts/phase06_oracle_baseline/run_20260411T131839Z_ep60_voracle_tpush_v3_s1000_r402093}"
TOP_K="${TOP_K:-15}"
EPISODES_PER_TARGET="${EPISODES_PER_TARGET:-4}"
SAVE_FRAMES="false"

PYTHON_BIN="${SMOLVLA_PYTHON_BIN:-${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: smolvla python not found" >&2
  exit 2
fi

LAUNCH_MODE="${SMOLVLA_TOPK_LAUNCH_MODE:-dry-run}"
LAUNCH_ARGS=(--oracle-run-dir "${ORACLE_RUN_DIR}" --top-k "${TOP_K}")

if [[ "${LAUNCH_MODE}" == "dry-run" ]]; then
  LAUNCH_ARGS+=(--dry-run)
elif [[ "${LAUNCH_MODE}" != "sbatch" ]]; then
  echo "error: unsupported SMOLVLA_TOPK_LAUNCH_MODE=${LAUNCH_MODE}" >&2
  exit 2
fi

CAMPAIGN_OUT="$(
  bash "${SCRIPT_DIR}/launch_pushv3_smolvla_topk15.sh" "${LAUNCH_ARGS[@]}"
)"
readarray -t INFO <<<"${CAMPAIGN_OUT}"
CAMPAIGN_DIR="${INFO[0]#campaign_dir=}"
TARGETS_JSON="${INFO[1]#targets_json=}"

TARGET_COUNT=$(
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
  TARGETS_JSON="${TARGETS_JSON}" \
  "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
import os

targets_path = Path(os.environ["TARGETS_JSON"]).resolve()
targets = json.loads(targets_path.read_text(encoding="utf-8")).get("targets", [])
print(len(targets))
PY
)

if [[ "${TARGET_COUNT}" -eq 0 ]]; then
  echo "error: no oracle targets in ${TARGETS_JSON}" >&2
  exit 2
fi

SUMMARY_PATH="${CAMPAIGN_DIR}/smolvla_topk_best_summary.json"
printf '[]' >"${SUMMARY_PATH}"

for TASK_INDEX in $(seq 0 $((TARGET_COUNT - 1))); do
  export SMOLVLA_EPISODES_PER_TARGET="${EPISODES_PER_TARGET}"
  export SMOLVLA_SAVE_FRAMES="${SAVE_FRAMES}"
  export SMOLVLA_CAMPAIGN_DIR="${CAMPAIGN_DIR}"
  bash "${SCRIPT_DIR}/run_smolvla_target_episode.sh" "${TARGETS_JSON}" "${TASK_INDEX}"

  TARGET_SUMMARY_PATH="${CAMPAIGN_DIR}/task_$(printf '%04d' "${TASK_INDEX}").json"
  if [[ ! -f "${TARGET_SUMMARY_PATH}" ]]; then
    echo "error: missing task summary ${TARGET_SUMMARY_PATH}" >&2
    exit 2
  fi

  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
  CAMPAIGN_DIR="${CAMPAIGN_DIR}" \
  TARGETS_JSON="${TARGETS_JSON}" \
  TARGET_SUMMARY_PATH="${TARGET_SUMMARY_PATH}" \
  SUMMARY_PATH="${SUMMARY_PATH}" \
  TASK_INDEX="${TASK_INDEX}" \
  TASK_SUMMARY_PATH="${TARGET_SUMMARY_PATH}" \
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import json
import os
import shutil

from src.smolvla_pipeline.topk_selection import pick_best_episode

campaign_dir = Path(os.environ["CAMPAIGN_DIR"]).resolve()
targets_json = Path(os.environ["TARGETS_JSON"]).resolve()
task_index = int(os.environ["TASK_INDEX"])
task_summary_path = Path(os.environ["TARGET_SUMMARY_PATH"]).resolve()
summary_path = Path(os.environ["SUMMARY_PATH"]).resolve()
task_summary_path_out = Path(os.environ["TASK_SUMMARY_PATH"]).resolve()

task_summary = json.loads(task_summary_path.read_text(encoding="utf-8"))
run_dir = Path(task_summary["run_dir"]).resolve()

run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
episodes = run_manifest.get("episodes", [])
best_episode = pick_best_episode(episodes)

targets = json.loads(targets_json.read_text(encoding="utf-8")).get("targets", [])
oracle_target = targets[task_index]
oracle_sum_reward = float(oracle_target.get("oracle_sum_reward", 0.0))
oracle_max_reward = float(oracle_target.get("oracle_max_reward", 0.0))

best_video_source = run_dir / best_episode["paths"]["video"]
best_video = run_dir / "best_episode.mp4"
if best_video.exists():
    best_video.unlink()
shutil.copy(best_video_source, best_video)

row = {
    "task_index": task_index,
    "oracle_rank": int(oracle_target["rank"]),
    "oracle_episode_index": int(oracle_target["episode_index"]),
    "oracle_reset_seed": int(oracle_target["reset_seed"]),
    "oracle_sum_reward": oracle_sum_reward,
    "oracle_max_reward": oracle_max_reward,
    "smolvla_best_episode_index": int(best_episode["episode_index"]),
    "smolvla_sum_reward": float(best_episode["sum_reward"]),
    "smolvla_max_reward": float(best_episode["max_reward"]),
    "smolvla_success": bool(best_episode["success"]),
    "best_video": str(best_video),
    "smolvla_outperforms_oracle_sum": bool(best_episode["sum_reward"] > oracle_sum_reward),
    "smolvla_outperforms_oracle_max": bool(best_episode["max_reward"] > oracle_max_reward),
}

best_episode_path = run_dir / "best_episode.json"
best_episode_path.write_text(json.dumps(row, indent=2), encoding="utf-8")

task_summary = json.loads(task_summary_path.read_text(encoding="utf-8"))
task_summary["smolvla_best"] = {
    "task_index": row["task_index"],
    "smolvla_best_episode_index": row["smolvla_best_episode_index"],
    "smolvla_sum_reward": row["smolvla_sum_reward"],
    "smolvla_max_reward": row["smolvla_max_reward"],
    "smolvla_success": row["smolvla_success"],
    "smolvla_outperforms_oracle_sum": row["smolvla_outperforms_oracle_sum"],
    "smolvla_outperforms_oracle_max": row["smolvla_outperforms_oracle_max"],
    "best_video": row["best_video"],
}
task_summary_path_out.write_text(json.dumps(task_summary, indent=2), encoding="utf-8")

rows = json.loads(summary_path.read_text(encoding="utf-8"))
rows.append(row)
rows.sort(key=lambda value: int(value["oracle_rank"]))
summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"selected {best_video_source} -> {best_video}")
PY
done

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
SUMMARY_PATH="${SUMMARY_PATH}" \
python3 - <<'PY'
import json
import os
from pathlib import Path

summary = json.loads(Path(os.environ["SUMMARY_PATH"]).read_text(encoding="utf-8"))
print("rank, oracle_episode, oracle_sum, smolvla_best_episode, smolvla_sum, delta_sum, smolvla_video")
for row in summary:
    delta = float(row["smolvla_sum_reward"]) - float(row["oracle_sum_reward"])
    print(
        f'{row["oracle_rank"]}, {row["oracle_episode_index"]}, '
        f'{row["oracle_sum_reward"]}, {row["smolvla_best_episode_index"]}, '
        f'{row["smolvla_sum_reward"]}, {delta}, {row["best_video"]}'
    )
print(f"summary_path={os.environ['SUMMARY_PATH']}")
print(f"environments={len(summary)}")
PY
