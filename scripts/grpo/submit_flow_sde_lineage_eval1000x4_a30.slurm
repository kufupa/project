#!/usr/bin/env bash
#SBATCH --job-name=fsde-e1000x4
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=flow_sde_lineage_eval1000x4_%j.out
#SBATCH --error=flow_sde_lineage_eval1000x4_%j.err
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
slurm_export_hf_torch_cache "flow-sde-lineage-e1000x4"
slurm_export_pythonpath

RLINF_ROOT="${RLINF_ROOT:-/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo}"
# shellcheck source=/dev/null
source "${RLINF_ROOT}/scripts/slurm/rlinf_smolvla_common.sh"
setup_rlinf_smolvla_env
require_python_bin

OUT_ROOT="${1:-/vol/bitbucket/aa6622/project/artifacts/flow_sde_over40_moonshots_20260603/u45_exploit_lr5e6/eval1000_lineage_seeds4000_4999}"
MODEL_PATH="${MODEL_PATH:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
SEED_BASE="${SEED_BASE:-4000}"
NUM_EPISODES="${NUM_EPISODES:-1000}"
NUM_ENVS="${NUM_ENVS:-25}"

CKPT_U25="/vol/bitbucket/aa6622/project/artifacts/flow_sde_chunk_grpo_moonshot_sparse30/train_246694/checkpoints_eval"
CKPT_U45="/vol/bitbucket/aa6622/project/artifacts/flow_sde_over40_moonshots_20260603/u25_aggressive_lr1e5/train_246945/checkpoints_eval"
CKPT_U60="/vol/bitbucket/aa6622/project/artifacts/flow_sde_over40_moonshots_20260603/u45_exploit_lr5e6/train_247023/checkpoints_eval"

mkdir -p "${OUT_ROOT}"

write_summary() {
  local out_dir="$1"
  local run_name="$2"
  "${PYTHON_BIN}" - "${out_dir}" "${run_name}" "${SEED_BASE}" "${NUM_EPISODES}" <<'PY'
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
        "num_envs": 25,
        "num_episodes": num_episodes,
    },
    "run_name": run_name,
    "rows_1000ep": rows,
}
(out / "eval1000_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
for row in rows:
    sr = float(row.get("success_rate", 0)) * 100
    u = row.get("update", row.get("checkpoint", "?"))
    print(f"  {u}: {sr:.2f}% wall_s={row.get('eval_wall_s')}")
PY
}

run_eval() {
  local label="$1"
  local ckpt_dir="$2"
  local only_updates="$3"
  local include_baseline="${4:-0}"
  local out_dir="${OUT_ROOT}/${label}"
  mkdir -p "${out_dir}"
  echo "=== [flow-sde-e1000x4] ${label} ckpt=${ckpt_dir} only=${only_updates} baseline=${include_baseline} ==="
  local -a cmd=(
    "${PYTHON_BIN}" -u scripts/eval_smolvla_metaworld_ckpt_sweep.py
    --run-name "lineage_${label}_eval1000"
    --output-dir "${out_dir}/eval_1000ep"
    --num-episodes "${NUM_EPISODES}"
    --checkpoint-dir "${ckpt_dir}"
    --model-path "${MODEL_PATH}"
    --task-name push-v3
    --task-description "Push the puck to a goal"
    --num-envs "${NUM_ENVS}"
    --seed-base "${SEED_BASE}"
    --max-episode-steps 150
    --chunk-len 5
  )
  if [[ -n "${only_updates}" ]]; then
    cmd+=(--only-updates "${only_updates}")
  fi
  if [[ "${include_baseline}" == "1" ]]; then
    cmd+=(--include-baseline)
  fi
  cd "${RLINF_ROOT}"
  "${cmd[@]}"
  write_summary "${out_dir}" "lineage_${label}_eval1000"
}

# baseline only (no GRPO ckpts in empty dir)
BASELINE_DIR="${OUT_ROOT}/_baseline_ckpt_dir"
mkdir -p "${BASELINE_DIR}"
run_eval "baseline" "${BASELINE_DIR}" "" "1"

run_eval "u25" "${CKPT_U25}" "25" "0"
run_eval "u45" "${CKPT_U45}" "45" "0"
run_eval "u60" "${CKPT_U60}" "60" "0"

echo "FLOW_SDE_LINEAGE_EVAL1000X4_OK out=${OUT_ROOT} seeds=${SEED_BASE}-$((SEED_BASE + NUM_EPISODES - 1))"
