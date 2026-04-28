#!/usr/bin/env bash
#SBATCH --job-name=phase12-eval-sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=phase12_eval_sweep_%j.out
#SBATCH --error=phase12_eval_sweep_%j.err
#SBATCH --export=NIL

set -euo pipefail

_PROJECT_FALLBACK="/vol/bitbucket/aa6622/project"
source "${_PROJECT_FALLBACK}/scripts/slurm/common_env.sh"
slurm_resolve_project_root "scripts/grpo/eval_phase12_checkpoint_sweep.py"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "phase12-eval-sweep"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
BASE_CKPT="${PHASE12_CHECKPOINT:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
RUN_DIR="${1:-${PHASE12_RUN_DIR:-}}"
SWEEP_NAME="${2:-${PHASE12_SWEEP_NAME:-eval_every10_u0_u300}}"
MIN_UPDATE="${3:-${PHASE12_MIN_UPDATE:-0}}"
MAX_UPDATE="${4:-${PHASE12_MAX_UPDATE:-300}}"
STRIDE="${5:-${PHASE12_STRIDE:-10}}"
EPISODES="${6:-${PHASE12_EVAL_EPISODES:-10}}"
EVAL_SEED_START="${7:-${PHASE12_EVAL_SEED_START:-1000}}"
TASK="${PHASE12_TASK:-push-v3}"
EXECUTION_MODE="${PHASE12_EVAL_EXECUTION_MODE:-subprocess}"
N_ENVS="${PHASE12_EVAL_N_ENVS:-1}"
ROLLOUT_EXECUTION="${PHASE12_EVAL_ROLLOUT_EXECUTION:-serial}"
MAX_STEPS="${PHASE12_EVAL_MAX_STEPS:-}"
if [[ $# -ge 8 && -n "${8:-}" ]]; then EXECUTION_MODE="${8}"; fi
if [[ $# -ge 9 && -n "${9:-}" ]]; then N_ENVS="${9}"; fi
if [[ $# -ge 10 && -n "${10:-}" ]]; then ROLLOUT_EXECUTION="${10}"; fi
if [[ $# -ge 11 && -n "${11:-}" ]]; then MAX_STEPS="${11}"; fi

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "error: RUN_DIR missing or not a directory: ${RUN_DIR}" >&2
  exit 2
fi

echo "[phase12-eval-sweep] run_dir=${RUN_DIR}"
echo "[phase12-eval-sweep] base_ckpt=${BASE_CKPT}"
echo "[phase12-eval-sweep] sweep_name=${SWEEP_NAME}"
echo "[phase12-eval-sweep] min_update=${MIN_UPDATE}"
echo "[phase12-eval-sweep] max_update=${MAX_UPDATE}"
echo "[phase12-eval-sweep] stride=${STRIDE}"
echo "[phase12-eval-sweep] episodes=${EPISODES}"
echo "[phase12-eval-sweep] eval_seed_start=${EVAL_SEED_START}"
echo "[phase12-eval-sweep] execution_mode=${EXECUTION_MODE}"
echo "[phase12-eval-sweep] n_envs=${N_ENVS}"
echo "[phase12-eval-sweep] rollout_execution=${ROLLOUT_EXECUTION}"
echo "[phase12-eval-sweep] max_steps=${MAX_STEPS:-<default>}"

EXTRA_ARGS=(
  --execution-mode "${EXECUTION_MODE}"
  --n-envs "${N_ENVS}"
  --rollout-execution "${ROLLOUT_EXECUTION}"
)
if [[ -n "${MAX_STEPS}" ]]; then
  EXTRA_ARGS+=(--max-steps "${MAX_STEPS}")
fi

"${GRPO_PYTHON}" scripts/grpo/eval_phase12_checkpoint_sweep.py \
  --base-checkpoint "${BASE_CKPT}" \
  --run-dir "${RUN_DIR}" \
  --task "${TASK}" \
  --episodes "${EPISODES}" \
  --eval-seed-start "${EVAL_SEED_START}" \
  --sweep-name "${SWEEP_NAME}" \
  --min-update "${MIN_UPDATE}" \
  --max-update "${MAX_UPDATE}" \
  --stride "${STRIDE}" \
  "${EXTRA_ARGS[@]}"

test -f "${RUN_DIR}/${SWEEP_NAME}/eval_sweep_summary.json"
echo "PHASE12_EVAL_SWEEP_DONE run_dir=${RUN_DIR} sweep=${SWEEP_NAME}"
