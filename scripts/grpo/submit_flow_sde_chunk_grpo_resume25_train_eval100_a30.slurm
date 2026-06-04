#!/usr/bin/env bash
#SBATCH --job-name=flow-sde-r25-e100
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=flow_sde_chunk_grpo_resume25_train_eval100_%j.out
#SBATCH --error=flow_sde_chunk_grpo_resume25_train_eval100_%j.err
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
slurm_export_hf_torch_cache "flow-sde-r25-e100"
slurm_export_pythonpath

RLINF_ROOT="${RLINF_ROOT:-/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo}"
# shellcheck source=/dev/null
source "${RLINF_ROOT}/scripts/slurm/rlinf_smolvla_common.sh"
setup_rlinf_smolvla_env
require_python_bin
EVAL_PYTHON_BIN="${PYTHON_BIN}"

GRPO_PYTHON_BIN="${GRPO_PYTHON_BIN:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
MODEL_PATH="${MODEL_PATH:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"

RUN_KEY="${1:?run key}"
OUT_ROOT="${2:?output root}"
RUN_LABEL="${3:?run label}"
GROUP_SIZE="${4:?group size}"
LR="${5:?learning rate}"
CLIP_EPS="${6:?clip eps}"
REWARD_MODE="${7:?reward mode}"
FLOW_SDE_TRACE_STEP="${8:?flow sde trace step}"
FLOW_SDE_NOISE_LEVEL="${9:?flow sde noise level}"
TRAIN_SEED_BASE="${10:?train seed base}"
ONLY_UPDATES="${11:?eval update list}"
EXPECTED_FINAL_UPDATE="${12:?expected final update, e.g. 0080}"
BASE_CKPT="${13:-${MODEL_PATH}}"
RESUME_CKPT="${14:?resume checkpoint required}"

NUM_TRAIN_UPDATES=25

TRAIN_OUT="${OUT_ROOT}/train_${SLURM_JOB_ID}"
EVAL100_OUT="${OUT_ROOT}/eval100_${SLURM_JOB_ID}"
mkdir -p "${TRAIN_OUT}" "${EVAL100_OUT}"

echo "[flow-sde-r25-e100] run=${RUN_KEY} train_out=${TRAIN_OUT} num_updates=${NUM_TRAIN_UPDATES}"
echo "[flow-sde-r25-e100] group=${GROUP_SIZE} lr=${LR} clip=${CLIP_EPS} reward=${REWARD_MODE} trace=${FLOW_SDE_TRACE_STEP} noise=${FLOW_SDE_NOISE_LEVEL} seed=${TRAIN_SEED_BASE} resume=${RESUME_CKPT}"

cd "${PROJECT_ROOT}"
train_cmd=(
  "${GRPO_PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/grpo/train_phase11_env_on_policy_grpo.py"
  --checkpoint "${BASE_CKPT}"
  --output-dir "${TRAIN_OUT}"
  --run-label "${RUN_LABEL}"
  --task push-v3
  --env-backend official_lerobot
  --num-updates "${NUM_TRAIN_UPDATES}"
  --batch-size 1
  --group-size "${GROUP_SIZE}"
  --train-seed-base "${TRAIN_SEED_BASE}"
  --lr "${LR}"
  --clip-eps "${CLIP_EPS}"
  --max-steps 120
  --reward-mode "${REWARD_MODE}"
  --rollout-unit chunk
  --rollout-chunk-len 5
  --rollout-execution vector_async
  --async-start-method forkserver
  --logprob-mode flow_sde
  --flow-sde-trace-step "${FLOW_SDE_TRACE_STEP}"
  --flow-sde-noise-level "${FLOW_SDE_NOISE_LEVEL}"
  --euler-step-noise-std 0.0
  --init-log-std -2.0
  --min-log-std -4.0
  --action-transform no_tanh
  --save-every 5
  --fail-on-parity-violation
  --resume "${RESUME_CKPT}"
)
"${train_cmd[@]}"

test -f "${TRAIN_OUT}/progress.jsonl"
test -f "${TRAIN_OUT}/train_manifest.json"
test -f "${TRAIN_OUT}/checkpoints/update_${EXPECTED_FINAL_UPDATE}.pt"
CKPT_DIR="${TRAIN_OUT}/checkpoints_eval"
test -f "${CKPT_DIR}/update_${EXPECTED_FINAL_UPDATE}.pt"

cd "${RLINF_ROOT}"
"${EVAL_PYTHON_BIN}" -u scripts/eval_smolvla_metaworld_ckpt_sweep.py \
  --run-name "${RUN_LABEL}_eval100" \
  --output-dir "${EVAL100_OUT}/eval_100ep" \
  --num-episodes 100 \
  --include-baseline \
  --checkpoint-dir "${CKPT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --task-name push-v3 \
  --task-description "Push the puck to a goal" \
  --num-envs 25 \
  --seed-base 1000 \
  --max-episode-steps 150 \
  --chunk-len 5 \
  --only-updates "${ONLY_UPDATES}"

"${EVAL_PYTHON_BIN}" - "${EVAL100_OUT}" "${RUN_LABEL}_eval100" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
run_name = sys.argv[2]
rows_path = out / "eval_100ep" / "results.jsonl"
rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
for row in rows:
    row["pc_success"] = float(row.get("success_rate", 0.0)) * 100.0
summary = {
    "protocol": {
        "backend": "rlinf_fast",
        "seed_base": 1000,
        "max_episode_steps": 150,
        "chunk_len": 5,
        "num_envs": 25,
        "num_episodes": 100,
    },
    "run_name": run_name,
    "rows_100ep": rows,
}
(out / "eval100_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("FLOW_SDE_RESUME25_TRAIN_EVAL100_OK", f"rows={len(rows)}", f"out={out}", f"run={run_name}")
PY

echo "FLOW_SDE_RESUME25_TRAIN_EVAL100_OK run=${RUN_KEY} train=${TRAIN_OUT} eval100=${EVAL100_OUT}"
