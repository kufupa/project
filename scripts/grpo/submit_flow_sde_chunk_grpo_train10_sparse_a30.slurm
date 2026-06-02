#!/usr/bin/env bash
#SBATCH --job-name=flow-sde-sparse-u10
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=flow_sde_chunk_grpo_train10_sparse_%j.out
#SBATCH --error=flow_sde_chunk_grpo_train10_sparse_%j.err
#SBATCH --export=NIL

set -euo pipefail

_PROJECT_FALLBACK="/vol/bitbucket/aa6622/project"
if [[ -f "${SLURM_SUBMIT_DIR:-}/scripts/slurm/common_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
elif [[ -f "${_PROJECT_FALLBACK}/scripts/slurm/common_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${_PROJECT_FALLBACK}/scripts/slurm/common_env.sh"
else
  echo "error: missing scripts/slurm/common_env.sh" >&2
  exit 2
fi

slurm_resolve_project_root "scripts/grpo/train_phase11_env_on_policy_grpo.py"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "flow-sde-chunk-grpo-train10-sparse"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
CHECKPOINT="${1:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
OUT="${2:-${PROJECT_ROOT}/artifacts/flow_sde_chunk_grpo_train10_sparse/${SLURM_JOB_ID:-local}}"

echo "[flow-sde-chunk-train10-sparse] checkpoint=${CHECKPOINT} out=${OUT}"

"${GRPO_PYTHON}" scripts/grpo/train_phase11_env_on_policy_grpo.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT}" \
  --task push-v3 \
  --env-backend official_lerobot \
  --rollout-execution serial \
  --rollout-unit chunk \
  --rollout-chunk-len 5 \
  --group-size 8 \
  --num-updates 10 \
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
  --flow-sde-trace-step -1 \
  --reward-mode sparse_success_delta \
  --run-label flow_sde_chunk_sparse_u10 \
  --save-every 2 \
  --fail-on-parity-violation

test -f "${OUT}/checkpoints/update_0010.pt"
test -f "${OUT}/progress.jsonl"

"${GRPO_PYTHON}" - "${OUT}" "sparse_success_delta" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
reward_mode = sys.argv[2]
manifest = json.loads((out / "train_manifest.json").read_text(encoding="utf-8"))
assert manifest["reward_mode"] == reward_mode, manifest
rows = [json.loads(line) for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(rows) == 10, len(rows)
bad = [r for r in rows if not (r.get("parity") or {}).get("within_tolerance")]
assert not bad, bad[:1]
assert all(r.get("chunk_count", 0) > 0 for r in rows), rows[-1]
assert all(r.get("reward_mode") == reward_mode for r in rows), rows[-1]
print("FLOW_SDE_CHUNK_GRPO_TRAIN10_OK", f"reward_mode={reward_mode}", f"rows={len(rows)}", f"out={out}")
PY

echo "FLOW_SDE_CHUNK_GRPO_TRAIN10_OK reward_mode=sparse_success_delta out=${OUT}"
