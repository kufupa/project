#!/usr/bin/env bash
#SBATCH --job-name=phase111-grpo-lr-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=phase111_on_grpo_lerobot_smoke_%j.out
#SBATCH --error=phase111_on_grpo_lerobot_smoke_%j.err
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
slurm_export_hf_torch_cache "phase111-on-grpo-lerobot-smoke"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
if [[ ! -x "${GRPO_PYTHON}" ]]; then
  echo "error: GRPO_PYTHON not executable: ${GRPO_PYTHON}" >&2
  exit 2
fi

CHECKPOINT="${GRPO_PHASE111_CHECKPOINT:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
OUT="${GRPO_PHASE111_OUT:-${PROJECT_ROOT}/artifacts/phase111_on_grpo_lerobot_smoke}"
TASK="${GRPO_PHASE111_TASK:-assembly-v3}"
MAX_STEPS="${GRPO_PHASE111_MAX_STEPS:-0}"

echo "[phase111] checkpoint=${CHECKPOINT}"
echo "[phase111] out=${OUT}"
echo "[phase111] task=${TASK}"
echo "[phase111] max_steps=${MAX_STEPS}"

"${GRPO_PYTHON}" scripts/grpo/train_phase11_env_on_policy_grpo.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT}" \
  --task "${TASK}" \
  --env-backend official_lerobot \
  --max-steps "${MAX_STEPS}" \
  --group-size 2 \
  --num-updates 1 \
  --train-seed-base 2000 \
  --start-update 0

test -f "${OUT}/train_manifest.json"
test -f "${OUT}/progress.jsonl"
test -f "${OUT}/checkpoints/latest.pt"
"${GRPO_PYTHON}" - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
manifest = json.loads((out / "train_manifest.json").read_text())
rows = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines() if line.strip()]
assert manifest["env_backend"] == "official_lerobot"
assert rows, "progress.jsonl has no rows"
row = rows[-1]
assert row["env_backend"] == "official_lerobot"
assert "returns" in row
assert "successes" in row
assert "success_rate" in row
print(
    "PHASE111_GRPO_LEROBOT_ARTIFACTS_OK",
    f"rows={len(rows)}",
    f"success_rate={row['success_rate']}",
    f"returns={row['returns']}",
)
PY

echo "PHASE111_GRPO_LEROBOT_SMOKE_OK out=${OUT}"
