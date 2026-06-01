#!/usr/bin/env bash
#SBATCH --job-name=ph46-train20
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=phase46_train20_%j.out
#SBATCH --error=phase46_train20_%j.err
#SBATCH --export=NIL

set -euo pipefail

_PHASE111_PROJECT_FALLBACK="/vol/bitbucket/aa6622/project"
if [[ -f "${SLURM_SUBMIT_DIR:-}/scripts/slurm/common_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
elif [[ -f "${_PHASE111_PROJECT_FALLBACK}/scripts/slurm/common_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${_PHASE111_PROJECT_FALLBACK}/scripts/slurm/common_env.sh"
fi

slurm_resolve_project_root "scripts/grpo/train_phase11_env_on_policy_grpo.py"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "phase46-train20"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
CHECKPOINT="${1:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
OUT="${2:-${PROJECT_ROOT}/artifacts/phase46_train20}"

echo "[phase46-train] out=${OUT}"

"${GRPO_PYTHON}" scripts/grpo/train_phase11_env_on_policy_grpo.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT}" \
  --task push-v3 \
  --env-backend official_lerobot \
  --rollout-execution vector_async \
  --group-size 8 \
  --num-updates 20 \
  --train-seed-base 2000 \
  --max-steps 120 \
  --lr 1e-5 \
  --clip-eps 0.2 \
  --init-log-std -2.0 \
  --euler-step-noise-std 0.0 \
  --min-log-std -4.0 \
  --action-transform no_tanh \
  --run-label phase46_g8 \
  --save-every 2 \
  --fail-on-parity-violation

test -f "${OUT}/checkpoints/update_0020.pt"
test -f "${OUT}/progress.jsonl"

"${GRPO_PYTHON}" - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines() if line.strip()]
assert len(rows) == 20, len(rows)
print("PHASE46_TRAIN_20UPD_OK", f"rows={len(rows)}")
PY

echo "PHASE46_TRAIN_20UPD_OK out=${OUT}"
