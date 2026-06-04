#!/usr/bin/env bash
#SBATCH --job-name=flow-sde-e1000
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=flow_sde_chunk_grpo_eval1000_%j.out
#SBATCH --error=flow_sde_chunk_grpo_eval1000_%j.err
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
slurm_export_hf_torch_cache "flow-sde-chunk-grpo-eval1000"
slurm_export_pythonpath

RLINF_ROOT="${RLINF_ROOT:-/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo}"
# shellcheck source=/dev/null
source "${RLINF_ROOT}/scripts/slurm/rlinf_smolvla_common.sh"
setup_rlinf_smolvla_env
require_python_bin

CKPT_DIR="${1:?checkpoint dir}"
OUT_DIR="${2:?output dir}"
RUN_NAME="${3:?run name}"
ONLY_UPDATES="${4:?only updates, e.g. 80}"
SEED_BASE="${5:?seed base}"
NUM_EPISODES="${6:?num episodes}"
NUM_ENVS="${7:-25}"
MODEL_PATH="${8:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"

mkdir -p "${OUT_DIR}"
echo "[flow-sde-e1000] ckpt=${CKPT_DIR} out=${OUT_DIR} run=${RUN_NAME}"
echo "[flow-sde-e1000] only_updates=${ONLY_UPDATES} seed_base=${SEED_BASE} num_episodes=${NUM_EPISODES} num_envs=${NUM_ENVS}"
echo "[flow-sde-e1000] seed_end=$((SEED_BASE + NUM_EPISODES - 1))"

cd "${RLINF_ROOT}"
"${PYTHON_BIN}" -u scripts/eval_smolvla_metaworld_ckpt_sweep.py \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUT_DIR}/eval_1000ep" \
  --num-episodes "${NUM_EPISODES}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --task-name push-v3 \
  --task-description "Push the puck to a goal" \
  --num-envs "${NUM_ENVS}" \
  --seed-base "${SEED_BASE}" \
  --max-episode-steps 150 \
  --chunk-len 5 \
  --only-updates "${ONLY_UPDATES}"

"${PYTHON_BIN}" - "${OUT_DIR}" "${RUN_NAME}" "${SEED_BASE}" "${NUM_EPISODES}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
run_name = sys.argv[2]
seed_base = int(sys.argv[3])
num_episodes = int(sys.argv[4])
rows_path = out / "eval_1000ep" / "results.jsonl"
rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
for row in rows:
    row["pc_success"] = float(row.get("success_rate", 0.0)) * 100.0
summary = {
    "protocol": {
        "backend": "rlinf_fast",
        "seed_base": seed_base,
        "seed_end": seed_base + num_episodes - 1,
        "max_episode_steps": 150,
        "chunk_len": 5,
        "num_envs": rows[0].get("num_episodes", None),
        "num_episodes": num_episodes,
    },
    "run_name": run_name,
    "rows_1000ep": rows,
}
(out / "eval1000_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("FLOW_SDE_EVAL1000_OK", f"rows={len(rows)}", f"out={out}", f"run={run_name}")
for row in rows:
    sr = float(row.get("success_rate", 0)) * 100
    u = row.get("update", "?")
    print(f"  u{u}: {sr:.2f}% n={row.get('num_episodes')} wall_s={row.get('eval_wall_s')}")
PY

echo "FLOW_SDE_EVAL1000_OK out=${OUT_DIR} run=${RUN_NAME}"
