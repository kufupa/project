#!/usr/bin/env bash
#SBATCH --job-name=ph46-logprob-abl
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=phase46_logprob_ablation_%j.out
#SBATCH --error=phase46_logprob_ablation_%j.err
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
  echo "error: cannot locate scripts/slurm/common_env.sh" >&2
  exit 2
fi

slurm_resolve_project_root "scripts/grpo/train_phase11_env_on_policy_grpo.py"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "phase46-logprob-ablation"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
CHECKPOINT="${CHECKPOINT:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
LOGPROB_ACTION="${1:?usage: submit_phase46_logprob_ablation_a30.slurm executed|unsquashed [out_dir]}"
OUT="${2:-${PROJECT_ROOT}/artifacts/phase46_logprob_ablation/${LOGPROB_ACTION}_${SLURM_JOB_ID:-local}}"
RLINF_ROOT="${RLINF_ROOT:-/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo}"

if [[ "${LOGPROB_ACTION}" != "executed" && "${LOGPROB_ACTION}" != "unsquashed" ]]; then
  echo "error: LOGPROB_ACTION must be executed or unsquashed, got ${LOGPROB_ACTION}" >&2
  exit 2
fi
if [[ ! -x "${GRPO_PYTHON}" ]]; then
  echo "error: GRPO_PYTHON not executable: ${GRPO_PYTHON}" >&2
  exit 2
fi
if [[ ! -x "${RLINF_ROOT}/scripts/run_phase46_tiered_eval_rlinf.sh" ]]; then
  echo "error: missing RLinf eval script under ${RLINF_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUT}"

echo "[phase46-logprob-ablation] action=${LOGPROB_ACTION}"
echo "[phase46-logprob-ablation] out=${OUT}"
echo "[phase46-logprob-ablation] checkpoint=${CHECKPOINT}"

"${GRPO_PYTHON}" scripts/grpo/train_phase11_env_on_policy_grpo.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT}/train" \
  --task push-v3 \
  --env-backend official_lerobot \
  --rollout-execution vector_async \
  --group-size 8 \
  --num-updates 5 \
  --train-seed-base 2000 \
  --max-steps 120 \
  --lr 1e-5 \
  --clip-eps 0.2 \
  --init-log-std -2.0 \
  --euler-step-noise-std 0.0 \
  --min-log-std -4.0 \
  --action-transform no_tanh \
  --gaussian-logprob-action "${LOGPROB_ACTION}" \
  --run-label "phase46_${LOGPROB_ACTION}_u5" \
  --save-every 1 \
  --fail-on-parity-violation

test -f "${OUT}/train/checkpoints/update_0001.pt"
test -f "${OUT}/train/checkpoints/update_0005.pt"
test -f "${OUT}/train/progress.jsonl"

"${GRPO_PYTHON}" - "${OUT}/train" "${LOGPROB_ACTION}" <<'PY'
import json
import sys
from pathlib import Path

train = Path(sys.argv[1])
expected_action = sys.argv[2]
rows = [json.loads(line) for line in (train / "progress.jsonl").read_text().splitlines() if line.strip()]
assert len(rows) == 5, len(rows)
bad = [r for r in rows if r.get("gaussian_logprob_action") != expected_action]
assert not bad, bad[:1]
print(
    "PHASE46_LOGPROB_ABLATION_TRAIN_OK",
    f"action={expected_action}",
    f"rows={len(rows)}",
    f"last_success={rows[-1].get('success_rate')}",
)
PY

"${RLINF_ROOT}/scripts/run_phase46_tiered_eval_rlinf.sh" \
  "${OUT}/train/checkpoints" \
  "${OUT}/eval" \
  1 \
  5 \
  "${CHECKPOINT}"

test -f "${OUT}/eval/tiered_eval_summary.json"

"${GRPO_PYTHON}" - "${OUT}" "${LOGPROB_ACTION}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
action = sys.argv[2]
summary = json.loads((out / "eval" / "tiered_eval_summary.json").read_text())
rows25 = summary.get("rows_25ep", [])
rows100 = summary.get("rows_100ep", [])
assert rows25, "missing 25ep rows"
assert rows100, "missing 100ep rows"
print(
    "PHASE46_LOGPROB_ABLATION_EVAL_OK",
    f"action={action}",
    f"rows25={len(rows25)}",
    f"rows100={len(rows100)}",
    f"summary={out / 'eval' / 'tiered_eval_summary.json'}",
)
PY

echo "PHASE46_LOGPROB_ABLATION_OK action=${LOGPROB_ACTION} out=${OUT}"
