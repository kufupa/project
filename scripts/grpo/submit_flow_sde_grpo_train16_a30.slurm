#!/usr/bin/env bash
#SBATCH --job-name=flow-sde-u16
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=flow_sde_grpo_train16_%j.out
#SBATCH --error=flow_sde_grpo_train16_%j.err
#SBATCH --export=NIL

set -euo pipefail

_PROJECT_FALLBACK="/vol/bitbucket/aa6622/project"
if [[ -f "${SLURM_SUBMIT_DIR:-}/scripts/slurm/common_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
elif [[ -f "${_PROJECT_FALLBACK}/scripts/slurm/common_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${_PROJECT_FALLBACK}/scripts/slurm/common_env.sh"
fi

slurm_resolve_project_root "scripts/grpo/train_phase11_env_on_policy_grpo.py"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "flow-sde-grpo-train16"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
CHECKPOINT="${1:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
OUT="${2:-${PROJECT_ROOT}/artifacts/flow_sde_grpo_train16/${SLURM_JOB_ID:-local}}"

echo "[flow-sde-train16] checkpoint=${CHECKPOINT} out=${OUT}"

"${GRPO_PYTHON}" scripts/grpo/train_phase11_env_on_policy_grpo.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT}" \
  --task push-v3 \
  --env-backend official_lerobot \
  --rollout-execution vector_async \
  --group-size 8 \
  --num-updates 16 \
  --train-seed-base 2000 \
  --max-steps 120 \
  --lr 5e-6 \
  --clip-eps 0.2 \
  --init-log-std -2.0 \
  --euler-step-noise-std 0.0 \
  --min-log-std -4.0 \
  --action-transform no_tanh \
  --logprob-mode flow_sde \
  --flow-sde-noise-level 0.5 \
  --flow-sde-trace-step 0 \
  --run-label flow_sde_u16 \
  --save-every 2 \
  --fail-on-parity-violation

test -f "${OUT}/checkpoints/update_0016.pt"
test -f "${OUT}/progress.jsonl"

"${GRPO_PYTHON}" - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines() if line.strip()]
assert len(rows) == 16, len(rows)
bad = [r for r in rows if not (r.get("parity") or {}).get("within_tolerance")]
assert not bad, bad[:1]
print("FLOW_SDE_GRPO_TRAIN16_OK", f"rows={len(rows)}", f"out={out}")
PY

echo "FLOW_SDE_GRPO_TRAIN16_OK out=${OUT}"
