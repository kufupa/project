#!/usr/bin/env bash
#SBATCH --job-name=phase111-grpo-correct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=phase111_grpo_correctness_smoke_%j.out
#SBATCH --error=phase111_grpo_correctness_smoke_%j.err
#SBATCH --export=NIL

set -euo pipefail

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
slurm_export_hf_torch_cache "phase111-grpo-correctness-smoke"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
CHECKPOINT="${1:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
OUT="${2:-${PROJECT_ROOT}/artifacts/phase111_grpo_correctness_smoke}"

echo "[correctness-smoke] checkpoint=${CHECKPOINT}"
echo "[correctness-smoke] out=${OUT}"

"${GRPO_PYTHON}" -m pytest tests/test_grpo_logprob_correctness.py tests/test_grpo_policy_wrapper_action_transform.py -q

"${GRPO_PYTHON}" scripts/grpo/train_phase11_env_on_policy_grpo.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT}" \
  --task push-v3 \
  --env-backend official_lerobot \
  --rollout-execution vector_async \
  --group-size 2 \
  --num-updates 2 \
  --train-seed-base 2000 \
  --max-steps 120 \
  --lr 1e-5 \
  --clip-eps 0.2 \
  --init-log-std -2.0 \
  --euler-step-noise-std 0.0 \
  --action-transform no_tanh \
  --run-label correctness_smoke \
  --save-every 1 \
  --fail-on-parity-violation

"${GRPO_PYTHON}" - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines() if line.strip()]
assert rows, "no progress rows"
for row in rows:
    assert row.get("skipped") is not True, row
    parity = row.get("parity") or {}
    assert parity.get("within_tolerance") is True, parity
    assert (row.get("parity") or {}).get("mean_ratio", 1.0) > 0.0

print("PHASE111_GRPO_CORRECTNESS_SMOKE_OK", f"updates={len(rows)}")
PY

echo "PHASE111_GRPO_CORRECTNESS_SMOKE_OK out=${OUT}"
