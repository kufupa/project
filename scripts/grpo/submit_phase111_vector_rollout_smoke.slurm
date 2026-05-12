#!/usr/bin/env bash
#SBATCH --job-name=phase111-vec-roll-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=phase111_vector_rollout_smoke_%j.out
#SBATCH --error=phase111_vector_rollout_smoke_%j.err
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
slurm_export_hf_torch_cache "phase111-vector-rollout-smoke"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
if [[ ! -x "${GRPO_PYTHON}" ]]; then
  echo "error: GRPO_PYTHON not executable: ${GRPO_PYTHON}" >&2
  exit 2
fi

CHECKPOINT="${GRPO_PHASE111_CHECKPOINT:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
OUT_BASE="${GRPO_PHASE111_OUT:-${PROJECT_ROOT}/artifacts/phase111_vector_rollout_smoke}"
TASK="${GRPO_PHASE111_TASK:-push-v3}"
MAX_STEPS="${GRPO_PHASE111_MAX_STEPS:-8}"
GROUP_SIZE="${GRPO_PHASE111_GROUP_SIZE:-2}"

echo "[phase111-vector-rollout-smoke] checkpoint=${CHECKPOINT}"
echo "[phase111-vector-rollout-smoke] task=${TASK} max_steps=${MAX_STEPS} group_size=${GROUP_SIZE}"

OUT_SYNC="${OUT_BASE}_vec_sync"
OUT_ASYNC="${OUT_BASE}_vec_async"

"${GRPO_PYTHON}" scripts/grpo/train_phase11_env_on_policy_grpo.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT_SYNC}" \
  --task "${TASK}" \
  --env-backend official_lerobot \
  --rollout-execution vector_sync \
  --max-steps "${MAX_STEPS}" \
  --group-size "${GROUP_SIZE}" \
  --num-updates 1 \
  --train-seed-base 2000 \
  --start-update 0

"${GRPO_PYTHON}" scripts/grpo/train_phase11_env_on_policy_grpo.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT_ASYNC}" \
  --task "${TASK}" \
  --env-backend official_lerobot \
  --rollout-execution vector_async \
  --async-start-method forkserver \
  --max-steps "${MAX_STEPS}" \
  --group-size "${GROUP_SIZE}" \
  --num-updates 1 \
  --train-seed-base 2000 \
  --start-update 0

"${GRPO_PYTHON}" - "${OUT_SYNC}" "${OUT_ASYNC}" <<'PY'
import json
import os
import sys
from pathlib import Path

sync = Path(sys.argv[1])
async_out = Path(sys.argv[2])

def load_manifest(p: Path) -> dict:
    return json.loads((p / "train_manifest.json").read_text(encoding="utf-8"))

m_sync = load_manifest(sync)
m_async = load_manifest(async_out)
assert m_sync["rollout_execution"] == "vector_sync"
assert m_sync.get("async_start_method") is None
assert m_async["rollout_execution"] == "vector_async"
assert m_async.get("async_start_method") == "forkserver"
ctx = os.environ.get("SMOLVLA_GRPO_ASYNC_MP_CONTEXT", "forkserver")
if ctx != "forkserver":
    print("warning: SMOLVLA_GRPO_ASYNC_MP_CONTEXT overrides async ctx:", ctx)
print(
    "PHASE111_VECTOR_ROLLOUT_SMOKE_OK",
    f"sync={sync}",
    f"async={async_out}",
    f"manifest_async_start={m_async.get('async_start_method')}",
)
PY

echo "PHASE111_VECTOR_ROLLOUT_SLURM_OK"
