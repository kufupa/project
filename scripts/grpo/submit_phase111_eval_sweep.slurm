#!/usr/bin/env bash
#SBATCH --job-name=phase111-eval-sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=phase111_eval_sweep_%j.out
#SBATCH --error=phase111_eval_sweep_%j.err
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

slurm_resolve_project_root "scripts/grpo/eval_phase111_grpo_sweep.py"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "phase111-eval-sweep"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
if [[ ! -x "${GRPO_PYTHON}" ]]; then
  echo "error: GRPO_PYTHON not executable: ${GRPO_PYTHON}" >&2
  exit 2
fi

RUN_DIR="${1:-}"
BASE_CKPT="${2:-}"
TASK="${3:-push-v3}"
EPISODES="${4:-10}"
EVAL_SEED_START="${5:-1000}"
TOP_K="${6:-0}"
TOP_K_EPISODES="${7:-50}"
SWEEP_NAME="${8:-eval_sweep_current}"

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "error: RUN_DIR missing or not a directory: ${RUN_DIR}" >&2
  exit 2
fi
if [[ -z "${BASE_CKPT}" ]]; then
  echo "error: BASE_CKPT is required" >&2
  exit 2
fi

echo "[phase111-eval-sweep] run_dir=${RUN_DIR}"
echo "[phase111-eval-sweep] base_ckpt=${BASE_CKPT}"
echo "[phase111-eval-sweep] task=${TASK}"
echo "[phase111-eval-sweep] episodes=${EPISODES}"
echo "[phase111-eval-sweep] eval_seed_start=${EVAL_SEED_START}"
echo "[phase111-eval-sweep] top_k=${TOP_K}"
echo "[phase111-eval-sweep] top_k_episodes=${TOP_K_EPISODES}"
echo "[phase111-eval-sweep] sweep_name=${SWEEP_NAME}"

"${GRPO_PYTHON}" scripts/grpo/eval_phase111_grpo_sweep.py \
  --base-checkpoint "${BASE_CKPT}" \
  --run-dir "${RUN_DIR}" \
  --task "${TASK}" \
  --episodes "${EPISODES}" \
  --eval-seed-start "${EVAL_SEED_START}" \
  --top-k "${TOP_K}" \
  --top-k-episodes "${TOP_K_EPISODES}" \
  --sweep-name "${SWEEP_NAME}"

test -f "${RUN_DIR}/${SWEEP_NAME}/eval_sweep_summary.json"

"${GRPO_PYTHON}" - "${RUN_DIR}" "${SWEEP_NAME}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
sweep_name = sys.argv[2]
summary = json.loads((run_dir / sweep_name / "eval_sweep_summary.json").read_text())
rows = summary.get("rows", [])
assert rows, "eval sweep has no rows"
print(
    "PHASE111_EVAL_SWEEP_OK",
    f"rows={len(rows)}",
    f"sweep_name={sweep_name}",
    f"out={run_dir / sweep_name / 'eval_sweep_summary.json'}",
)
PY
