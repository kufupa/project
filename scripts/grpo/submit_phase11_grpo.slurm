#!/bin/bash
#SBATCH --job-name=phase11_grpo
#SBATCH --output=phase11_grpo_%j.out
#SBATCH --error=phase11_grpo_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# One allocation worker: runs a single train or eval command. No nested sbatch.
# Usage (from login node, after sbatch --test-only):
#   sbatch --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/grpo/submit_phase11_grpo.slurm train \
#     /path/to/checkpoint /path/to/output_dir 2000 0 1
#
# Or eval:
#   sbatch --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/grpo/submit_phase11_grpo.slurm eval \
#     /path/to/base_ckpt /path/to/update_0005.pt /path/to/eval_out 1000 20
#
# Rollout smoke (short env steps, small group):
#   sbatch --chdir=... --export=NIL scripts/grpo/submit_phase11_grpo.slurm rollout-smoke \
#     jadechoghari/smolvla_metaworld 2 10 2000

set -euo pipefail

# SLURM_SUBMIT_DIR = dir sbatch invoked from, not always --chdir. Spooled BASH_SOURCE breaks ../slurm.
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
  echo "error: cannot find scripts/slurm/common_env.sh (set PROJECT_ROOT or sbatch from project/)" >&2
  exit 2
fi
# shellcheck source=../slurm/common_env.sh
source "${_COMMON}"

slurm_resolve_project_root "scripts/grpo/train_phase11_env_on_policy_grpo.py"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "phase11_grpo"

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

MODE="${1:-train}"
shift || true

if [[ "${MODE}" == "train" ]]; then
  CKPT="${1:?checkpoint}"
  OUT="${2:?output_dir}"
  SEED_BASE="${3:-2000}"
  START="${4:-0}"
  NUPD="${5:-1}"
  RESUME="${6:-}"
  EXTRA=()
  if [[ -n "${RESUME}" ]]; then
    EXTRA+=(--resume "${RESUME}")
  fi
  exec "${GRPO_PYTHON}" scripts/grpo/train_phase11_env_on_policy_grpo.py \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT}" \
    --train-seed-base "${SEED_BASE}" \
    --start-update "${START}" \
    --num-updates "${NUPD}" \
    "${EXTRA[@]}"
elif [[ "${MODE}" == "eval" ]]; then
  BASE="${1:?base checkpoint}"
  GRPO_PT="${2:?grpo pt}"
  EVAL_OUT="${3:?eval output}"
  EVAL_START="${4:-1000}"
  EPS="${5:-20}"
  exec "${GRPO_PYTHON}" scripts/grpo/eval_phase11_checkpoints.py \
    --base-checkpoint "${BASE}" \
    --grpo-checkpoint "${GRPO_PT}" \
    --output-dir "${EVAL_OUT}" \
    --eval-seed-start "${EVAL_START}" \
    --episodes "${EPS}"
elif [[ "${MODE}" == "rollout-smoke" ]]; then
  CKPT="${1:?checkpoint}"
  GS="${2:-2}"
  MS="${3:-10}"
  SD="${4:-2000}"
  exec "${GRPO_PYTHON}" scripts/grpo/smoke_phase11_rollout.py \
    --checkpoint "${CKPT}" \
    --group-size "${GS}" \
    --max-steps "${MS}" \
    --seed "${SD}"
else
  echo "usage: $0 train <ckpt> <out> [seed_base] [start] [n_updates] [resume_pt]" >&2
  echo "       $0 eval <base_ckpt> <grpo.pt> <eval_out> [eval_seed_start] [episodes]" >&2
  echo "       $0 rollout-smoke <ckpt> [group_size] [max_steps] [reset_seed]" >&2
  exit 2
fi
