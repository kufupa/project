#!/bin/bash
#SBATCH --job-name=phase11_eval_1k
#SBATCH --output=phase11_eval_sweep_1k_%j.out
#SBATCH --error=phase11_eval_sweep_1k_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# Sequential eval sweep for milestones update_NNNN with NNNN >= MIN_UPDATE (default 50),
# up to MAX_UPDATE (default 100). Uses eval_seed_start=1000, episodes=20 (seeds 1000..1019).
# Dedupes checkpoint paths like submit_phase11_eval_sweep_seed1020.slurm (later train dirs win).
#
# Usage:
#   sbatch --chdir=/vol/bitbucket/aa6622/project --export=NIL \
#     --dependency=afterok:<PRIOR_JOBID> scripts/grpo/submit_phase11_eval_sweep_seed1000_ge50.slurm

set -euo pipefail

_PHASE11_PROJECT_FALLBACK="/vol/bitbucket/aa6622/project"
_COMMON=""
if [[ -f "${_PHASE11_PROJECT_FALLBACK}/scripts/slurm/common_env.sh" ]]; then
  _COMMON="${_PHASE11_PROJECT_FALLBACK}/scripts/slurm/common_env.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh" ]]; then
  _COMMON="${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
elif [[ -n "${PROJECT_ROOT:-}" && -f "${PROJECT_ROOT}/scripts/slurm/common_env.sh" ]]; then
  _COMMON="${PROJECT_ROOT}/scripts/slurm/common_env.sh"
else
  _SELF="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  if [[ -f "${_SELF}/../slurm/common_env.sh" ]]; then
    _COMMON="${_SELF}/../slurm/common_env.sh"
  fi
fi
if [[ -z "${_COMMON}" || ! -f "${_COMMON}" ]]; then
  echo "error: cannot find scripts/slurm/common_env.sh" >&2
  exit 2
fi
# shellcheck source=../slurm/common_env.sh
source "${_COMMON}"

slurm_resolve_project_root "scripts/grpo/eval_phase11_checkpoints.py"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "phase11_eval_sweep_1k"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
if [[ ! -x "${GRPO_PYTHON}" ]]; then
  echo "error: GRPO_PYTHON not executable: ${GRPO_PYTHON}" >&2
  exit 2
fi

"${GRPO_PYTHON}" - <<'PY'
import sys
import lerobot.policies.smolvla.modeling_smolvla as m

assert sys.executable.startswith("/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/")
assert "/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/" in m.__file__
PY

ARTIFACT_ROOT="${PHASE11_ARTIFACT_ROOT:-/vol/bitbucket/aa6622/project/artifacts/phase11_env_on_policy_grpo}"
BASE_CKPT="${PHASE11_BASE_CHECKPOINT:-jadechoghari/smolvla_metaworld}"
EVAL_SEED_START="${PHASE11_EVAL_SEED_START:-1000}"
EPISODES="${PHASE11_EVAL_EPISODES:-20}"
MIN_UPDATE="${PHASE11_MIN_UPDATE:-50}"
MAX_UPDATE="${PHASE11_MAX_UPDATE:-100}"

SEARCH_DIRS=(
  "${ARTIFACT_ROOT}/smoke_update1"
  "${ARTIFACT_ROOT}/train_u1_to_u4"
  "${ARTIFACT_ROOT}/run_progressive/long_to25"
  "${ARTIFACT_ROOT}/run_progressive/long_to45"
  "${ARTIFACT_ROOT}/run_progressive/long_to100"
)

declare -A CKPT_BY_NUM=()

for root in "${SEARCH_DIRS[@]}"; do
  cdir="${root}/checkpoints"
  [[ -d "${cdir}" ]] || continue
  shopt -s nullglob
  for f in "${cdir}"/update_*.pt; do
    b=$(basename "${f}" .pt)
    num_str="${b#update_}"
    num=$((10#${num_str}))
    if (( num < MIN_UPDATE || num > MAX_UPDATE )); then
      continue
    fi
    CKPT_BY_NUM[${num}]="${f}"
  done
  shopt -u nullglob
done

nums=("${!CKPT_BY_NUM[@]}")
sorted=()
while IFS= read -r line; do
  [[ -n "${line}" ]] || continue
  sorted+=("${line}")
done < <(printf '%s\n' "${nums[@]}" | sort -n)

if ((${#sorted[@]} == 0)); then
  echo "error: no update_*.pt with ${MIN_UPDATE} <= N <= ${MAX_UPDATE} under ${ARTIFACT_ROOT}" >&2
  exit 2
fi

echo "[phase11_eval_sweep_1k] ARTIFACT_ROOT=${ARTIFACT_ROOT}"
echo "[phase11_eval_sweep_1k] MIN_UPDATE=${MIN_UPDATE} MAX_UPDATE=${MAX_UPDATE}"
echo "[phase11_eval_sweep_1k] checkpoints: ${#sorted[@]} (update nums: ${sorted[*]})"
echo "[phase11_eval_sweep_1k] eval_seed_start=${EVAL_SEED_START} episodes=${EPISODES}"

for n in "${sorted[@]}"; do
  ck="${CKPT_BY_NUM[${n}]}"
  out="${ARTIFACT_ROOT}/eval_update$(printf '%04d' "${n}")_seed${EVAL_SEED_START}_ep${EPISODES}"
  echo "[phase11_eval_sweep_1k] --- update_${n} ---"
  echo "[phase11_eval_sweep_1k] ckpt=${ck}"
  echo "[phase11_eval_sweep_1k] out=${out}"
  "${GRPO_PYTHON}" scripts/grpo/eval_phase11_checkpoints.py \
    --base-checkpoint "${BASE_CKPT}" \
    --grpo-checkpoint "${ck}" \
    --output-dir "${out}" \
    --eval-seed-start "${EVAL_SEED_START}" \
    --episodes "${EPISODES}"
done

echo "[phase11_eval_sweep_1k] done all ${#sorted[@]} evals"
