#!/usr/bin/env bash
#SBATCH --job-name=phase111-grpo-single
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=phase111_single_task_grpo_%j.out
#SBATCH --error=phase111_single_task_grpo_%j.err
#SBATCH --export=NIL

set -euo pipefail

_PHASE111_PROJECT_FALLBACK="/vol/bitbucket/aa6622/project"
_COMMON=""
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh" ]]; then
  _COMMON="${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
elif [[ -n "${PROJECT_ROOT:-}" && -f "${PROJECT_ROOT}/scripts/slurm/common_env.sh" ]]; then
  _COMMON="${PROJECT_ROOT}/scripts/slurm/common_env.sh"
elif [[ -f "${_PHASE111_PROJECT_FALLBACK}/scripts/slurm/common_env.sh" ]]; then
  _COMMON="${_PHASE111_PROJECT_FALLBACK}/scripts/slurm/common_env.sh"
else
  _SELF="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  if [[ -f "${_SELF}/../slurm/common_env.sh" ]]; then
    _COMMON="${_SELF}/../slurm/common_env.sh"
  fi
fi
if [[ -z "${_COMMON}" || ! -f "${_COMMON}" ]]; then
  echo "error: cannot find scripts/slurm/common_env.sh (set PROJECT_ROOT or sbatch from project/)" >&2
  exit 2
fi
# shellcheck source=../slurm/common_env.sh
source "${_COMMON}"

slurm_resolve_project_root "scripts/grpo/train_phase11_env_on_policy_grpo.py"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "phase111-single-task-grpo"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
if [[ ! -x "${GRPO_PYTHON}" ]]; then
  echo "error: GRPO_PYTHON not executable: ${GRPO_PYTHON}" >&2
  exit 2
fi

CHECKPOINT="${GRPO_PHASE111_CHECKPOINT:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
TASK="${GRPO_PHASE111_TASK:-push-v3}"
SEED_BASE="${GRPO_PHASE111_SEED_BASE:-2000}"
START_UPDATE="${GRPO_PHASE111_START_UPDATE:-0}"
NUM_UPDATES="${GRPO_PHASE111_NUM_UPDATES:-100}"
GROUP_SIZE="${GRPO_PHASE111_GROUP_SIZE:-4}"
SAVE_EVERY="${GRPO_PHASE111_SAVE_EVERY:-5}"
# Phase111 trains with a 120-step cap to keep update cost comparable to legacy
# Phase11 and the CS238 GRPO runs. Use GRPO_PHASE111_MAX_STEPS=0 only when
# intentionally training on the full official LeRobot horizon (500 for v3).
MAX_STEPS="${GRPO_PHASE111_MAX_STEPS:-120}"
ROLLOUT_EXEC="${GRPO_PHASE111_ROLLOUT_EXECUTION:-vector_async}"
ASYNC_START="${GRPO_PHASE111_ASYNC_START_METHOD:-forkserver}"
LR="${GRPO_PHASE111_LR:-1e-5}"
INIT_LOG_STD="${GRPO_PHASE111_INIT_LOG_STD:--2.0}"
EULER_NOISE="${GRPO_PHASE111_EULER_NOISE:-0.0}"
ACTION_TRANSFORM="${GRPO_PHASE111_ACTION_TRANSFORM:-no_tanh}"
RUN_LABEL="${GRPO_PHASE111_RUN_LABEL:-no_tanh_main}"
RESUME="${GRPO_PHASE111_RESUME:-}"
OUT="${GRPO_PHASE111_OUT:-${PROJECT_ROOT}/artifacts/phase111_on_grpo_lerobot/${TASK}/g${GROUP_SIZE}_u${NUM_UPDATES}_seed${SEED_BASE}}"

# Optional positional overrides to avoid Slurm --export=NIL env retrieval issues:
#   $1 task, $2 num_updates, $3 group_size, $4 max_steps, $5 save_every,
#   $6 seed_base, $7 start_update, $8 out_dir, $9 resume_path,
#   $10 rollout_execution, $11 async_start_method, $12 action_transform,
#   $13 run_label
if [[ $# -ge 1 && -n "${1:-}" ]]; then TASK="${1}"; fi
if [[ $# -ge 2 && -n "${2:-}" ]]; then NUM_UPDATES="${2}"; fi
if [[ $# -ge 3 && -n "${3:-}" ]]; then GROUP_SIZE="${3}"; fi
if [[ $# -ge 4 && -n "${4:-}" ]]; then MAX_STEPS="${4}"; fi
if [[ $# -ge 5 && -n "${5:-}" ]]; then SAVE_EVERY="${5}"; fi
if [[ $# -ge 6 && -n "${6:-}" ]]; then SEED_BASE="${6}"; fi
if [[ $# -ge 7 && -n "${7:-}" ]]; then START_UPDATE="${7}"; fi
if [[ $# -ge 8 && -n "${8:-}" ]]; then OUT="${8}"; fi
if [[ $# -ge 9 && -n "${9:-}" ]]; then RESUME="${9}"; fi
if [[ $# -ge 10 && -n "${10:-}" ]]; then ROLLOUT_EXEC="${10}"; fi
if [[ $# -ge 11 && -n "${11:-}" ]]; then ASYNC_START="${11}"; fi
if [[ $# -ge 12 && -n "${12:-}" ]]; then ACTION_TRANSFORM="${12}"; fi
if [[ $# -ge 13 && -n "${13:-}" ]]; then RUN_LABEL="${13}"; fi

echo "[phase111-single] checkpoint=${CHECKPOINT}"
echo "[phase111-single] out=${OUT}"
echo "[phase111-single] task=${TASK}"
echo "[phase111-single] num_updates=${NUM_UPDATES}"
echo "[phase111-single] group_size=${GROUP_SIZE}"
echo "[phase111-single] max_steps=${MAX_STEPS}"
echo "[phase111-single] rollout_execution=${ROLLOUT_EXEC}"
echo "[phase111-single] async_start_method=${ASYNC_START}"
echo "[phase111-single] action_transform=${ACTION_TRANSFORM}"
echo "[phase111-single] run_label=${RUN_LABEL}"

EXTRA=()
if [[ -n "${RESUME}" ]]; then
  EXTRA+=(--resume "${RESUME}")
fi
if [[ "${ROLLOUT_EXEC}" != "serial" ]]; then
  EXTRA+=(--rollout-execution "${ROLLOUT_EXEC}")
fi
if [[ "${ROLLOUT_EXEC}" == "vector_async" ]]; then
  EXTRA+=(--async-start-method "${ASYNC_START}")
fi

"${GRPO_PYTHON}" scripts/grpo/train_phase11_env_on_policy_grpo.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT}" \
  --task "${TASK}" \
  --env-backend official_lerobot \
  --train-seed-base "${SEED_BASE}" \
  --start-update "${START_UPDATE}" \
  --num-updates "${NUM_UPDATES}" \
  --group-size "${GROUP_SIZE}" \
  --save-every "${SAVE_EVERY}" \
  --max-steps "${MAX_STEPS}" \
  --lr "${LR}" \
  --init-log-std "${INIT_LOG_STD}" \
  --euler-step-noise-std "${EULER_NOISE}" \
  --action-transform "${ACTION_TRANSFORM}" \
  --run-label "${RUN_LABEL}" \
  "${EXTRA[@]}"

test -f "${OUT}/train_manifest.json"
test -f "${OUT}/progress.jsonl"
test -f "${OUT}/checkpoints/latest.pt"
if [[ "${NUM_UPDATES}" -ge 5 ]]; then
  test -f "${OUT}/checkpoints/update_0005.pt"
fi

"${GRPO_PYTHON}" - "${OUT}" "${NUM_UPDATES}" "${TASK}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
num_updates = int(sys.argv[2])
task = sys.argv[3]
manifest = json.loads((out / "train_manifest.json").read_text())
rows = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines() if line.strip()]
assert manifest["env_backend"] == "official_lerobot"
assert manifest["task"] == task
assert manifest["rollout_execution"] in {"serial", "vector_sync", "vector_async"}
assert manifest["action_transform"] in {"no_tanh", "tanh_norm_ablation"}
assert manifest["run_label"]
if manifest["rollout_execution"] == "vector_async":
    assert manifest["async_start_method"] == "forkserver"
assert rows, "progress.jsonl has no rows"
end_update = int(manifest["end_update"])
assert len(rows) == end_update, (
    f"expected {end_update} progress rows (manifest end_update), got {len(rows)}"
)
row = rows[-1]
assert int(row["update"]) == end_update - 1, (
    f"expected last progress update index {end_update - 1}, got {row.get('update')}"
)
assert row["env_backend"] == "official_lerobot"
assert row["rollout_execution"] == manifest["rollout_execution"]
assert row["action_transform"] == manifest["action_transform"]
assert row["run_label"] == manifest["run_label"]
assert "returns" in row
assert "successes" in row
assert "success_rate" in row
assert "episode_lengths" in row
assert "resolved_max_steps" in row
assert "num_env_steps" in row
assert "action_clip_fraction" in row
assert "action_clip_any_fraction" in row
assert "rollout_seconds" in row
assert "optimize_seconds" in row
assert "update_seconds" in row
last_ckpt = out / "checkpoints" / f"update_{end_update:04d}.pt"
assert last_ckpt.is_file(), f"missing terminal checkpoint {last_ckpt}"
this_run_updates = end_update - int(manifest["start_update"])
print(
    "PHASE111_SINGLE_TASK_GRPO_OK",
    f"rows={len(rows)}",
    f"expected_updates={this_run_updates}",
    f"success_rate={row['success_rate']}",
)
PY
