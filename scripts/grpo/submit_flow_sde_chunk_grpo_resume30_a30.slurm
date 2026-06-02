#!/usr/bin/env bash
#SBATCH --job-name=flow-sde-resume30
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=flow_sde_chunk_grpo_resume30_%j.out
#SBATCH --error=flow_sde_chunk_grpo_resume30_%j.err
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
slurm_export_hf_torch_cache "flow-sde-chunk-grpo-resume30"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
RESUME="${1:?resume checkpoint}"
OUT="${2:?output dir}"
REWARD_MODE="${3:?reward mode}"
RUN_LABEL="${4:?run label}"
CHECKPOINT="${5:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"

echo "[flow-sde-chunk-resume30] resume=${RESUME} out=${OUT} reward_mode=${REWARD_MODE} run=${RUN_LABEL}"

"${GRPO_PYTHON}" scripts/grpo/train_phase11_env_on_policy_grpo.py \
  --checkpoint "${CHECKPOINT}" \
  --resume "${RESUME}" \
  --output-dir "${OUT}" \
  --task push-v3 \
  --env-backend official_lerobot \
  --rollout-execution vector_async \
  --async-start-method forkserver \
  --rollout-unit chunk \
  --rollout-chunk-len 5 \
  --group-size 8 \
  --num-updates 30 \
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
  --reward-mode "${REWARD_MODE}" \
  --run-label "${RUN_LABEL}" \
  --save-every 5 \
  --fail-on-parity-violation

test -f "${OUT}/checkpoints/update_0040.pt"
test -f "${OUT}/checkpoints_eval/update_0040.pt"
test -f "${OUT}/progress.jsonl"

"${GRPO_PYTHON}" - "${OUT}" "${REWARD_MODE}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
reward_mode = sys.argv[2]
manifest = json.loads((out / "train_manifest.json").read_text(encoding="utf-8"))
assert manifest["start_update"] == 10, manifest
assert manifest["end_update"] == 40, manifest
assert manifest["reward_mode"] == reward_mode, manifest
rows = [json.loads(line) for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
updates = {int(r["update"]) for r in rows}
assert set(range(10, 40)).issubset(updates), sorted(updates)
bad = [
    r for r in rows
    if int(r.get("update", -1)) >= 10
    and not r.get("skipped")
    and not (r.get("parity") or {}).get("within_tolerance")
]
assert not bad, bad[:1]
print("FLOW_SDE_CHUNK_GRPO_RESUME30_OK", f"reward_mode={reward_mode}", f"out={out}")
PY

echo "FLOW_SDE_CHUNK_GRPO_RESUME30_OK reward_mode=${REWARD_MODE} out=${OUT}"
