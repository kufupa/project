#!/usr/bin/env bash
#SBATCH --job-name=phase111-g8-correct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=phase111_grpo_correctness_g8_%j.out
#SBATCH --error=phase111_grpo_correctness_g8_%j.err
#SBATCH --export=NIL

set -euo pipefail

# Positional: $1 checkpoint, $2 out_dir, $3 num_updates (default 15), $4 start_update (default 0)

_PHASE111_PROJECT_FALLBACK="/vol/bitbucket/aa6622/project"
_COMMON=""
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh" ]]; then
  _COMMON="${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
elif [[ -f "${_PHASE111_PROJECT_FALLBACK}/scripts/slurm/common_env.sh" ]]; then
  _COMMON="${_PHASE111_PROJECT_FALLBACK}/scripts/slurm/common_env.sh"
else
  _SELF="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  _COMMON="${_SELF}/../slurm/common_env.sh"
fi
# shellcheck source=../slurm/common_env.sh
source "${_COMMON}"

slurm_resolve_project_root "scripts/grpo/train_phase11_env_on_policy_grpo.py"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "phase111-grpo-correctness-g8"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
CHECKPOINT="${1:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
OUT="${2:-${PROJECT_ROOT}/artifacts/phase111_grpo_correctness_g8_u15}"
NUM_UPDATES="${3:-15}"
START_UPDATE="${4:-0}"

echo "[correctness-g8] checkpoint=${CHECKPOINT}"
echo "[correctness-g8] out=${OUT}"
echo "[correctness-g8] num_updates=${NUM_UPDATES} start=${START_UPDATE}"

"${GRPO_PYTHON}" scripts/grpo/train_phase11_env_on_policy_grpo.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT}" \
  --task push-v3 \
  --env-backend official_lerobot \
  --rollout-execution vector_async \
  --group-size 8 \
  --num-updates "${NUM_UPDATES}" \
  --start-update "${START_UPDATE}" \
  --train-seed-base 2000 \
  --max-steps 120 \
  --lr 1e-5 \
  --clip-eps 0.2 \
  --init-log-std -2.0 \
  --euler-step-noise-std 0.0 \
  --action-transform no_tanh \
  --run-label correctness_g8 \
  --save-every 2 \
  --fail-on-parity-violation

test -f "${OUT}/progress.jsonl"
test -f "${OUT}/checkpoints/latest.pt"

"${GRPO_PYTHON}" - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines() if line.strip()]
bad = [r for r in rows if r.get("skipped") or not (r.get("parity") or {}).get("within_tolerance")]
assert not bad, f"parity failures: {bad[:2]}"
print("PHASE111_GRPO_CORRECTNESS_G8_TRAIN_OK", f"rows={len(rows)}")
PY

echo "PHASE111_GRPO_CORRECTNESS_G8_TRAIN_OK out=${OUT}"
