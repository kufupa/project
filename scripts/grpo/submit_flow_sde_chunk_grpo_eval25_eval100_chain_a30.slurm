#!/usr/bin/env bash
#SBATCH --job-name=flow-sde-e25-e100
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=flow_sde_chunk_grpo_eval25_eval100_chain_%j.out
#SBATCH --error=flow_sde_chunk_grpo_eval25_eval100_chain_%j.err
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
slurm_export_hf_torch_cache "flow-sde-chunk-grpo-e25-e100"

RLINF_ROOT="${RLINF_ROOT:-/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo}"
# shellcheck source=/dev/null
source "${RLINF_ROOT}/scripts/slurm/rlinf_smolvla_common.sh"
setup_rlinf_smolvla_env
require_python_bin

CKPT_DIR="${1:?checkpoint dir}"
OUT25_DIR="${2:?25ep output dir}"
RUN25_NAME="${3:?25ep run name}"
OUT100_DIR="${4:?100ep output dir}"
RUN100_NAME="${5:?100ep run name}"
ONLY_UPDATES="${6:-10,20,30,40}"
MODEL_PATH="${7:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"

mkdir -p "${OUT25_DIR}" "${OUT100_DIR}"
echo "[flow-sde-e25-e100] ckpt=${CKPT_DIR} out25=${OUT25_DIR} out100=${OUT100_DIR}"

cd "${RLINF_ROOT}"
"${PYTHON_BIN}" -u scripts/eval_smolvla_metaworld_ckpt_sweep.py \
  --run-name "${RUN25_NAME}" \
  --output-dir "${OUT25_DIR}/eval_25ep" \
  --num-episodes 25 \
  --include-baseline \
  --checkpoint-dir "${CKPT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --task-name push-v3 \
  --task-description "Push the puck to a goal" \
  --num-envs 25 \
  --seed-base 1000 \
  --max-episode-steps 150 \
  --chunk-len 5

"${PYTHON_BIN}" - "${OUT25_DIR}" "${RUN25_NAME}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
run_name = sys.argv[2]
rows_path = out / "eval_25ep" / "results.jsonl"
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
        "num_episodes": 25,
    },
    "run_name": run_name,
    "rows_25ep": rows,
}
(out / "eval25_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("FLOW_SDE_CHUNK_GRPO_EVAL25_CHAIN_OK", f"rows={len(rows)}", f"out={out}", f"run={run_name}")
PY

cmd=(
  "${PYTHON_BIN}" -u scripts/eval_smolvla_metaworld_ckpt_sweep.py
  --run-name "${RUN100_NAME}"
  --output-dir "${OUT100_DIR}/eval_100ep"
  --num-episodes 100
  --include-baseline
  --checkpoint-dir "${CKPT_DIR}"
  --model-path "${MODEL_PATH}"
  --task-name push-v3
  --task-description "Push the puck to a goal"
  --num-envs 25
  --seed-base 1000
  --max-episode-steps 150
  --chunk-len 5
)
if [[ -n "${ONLY_UPDATES}" ]]; then
  cmd+=(--only-updates "${ONLY_UPDATES}")
fi
"${cmd[@]}"

"${PYTHON_BIN}" - "${OUT100_DIR}" "${RUN100_NAME}" <<'PY'
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
        "only_updates": run_name,
    },
    "run_name": run_name,
    "rows_100ep": rows,
}
(out / "eval100_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("FLOW_SDE_CHUNK_GRPO_EVAL100_CHAIN_OK", f"rows={len(rows)}", f"out={out}", f"run={run_name}")
PY

echo "FLOW_SDE_CHUNK_GRPO_EVAL25_EVAL100_CHAIN_OK out25=${OUT25_DIR} out100=${OUT100_DIR}"
