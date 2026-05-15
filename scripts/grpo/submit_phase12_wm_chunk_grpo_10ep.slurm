#!/usr/bin/env bash
#SBATCH --job-name=phase12-wm-grpo-10ep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=phase12_wm_chunk_grpo_10ep_%j.out
#SBATCH --error=phase12_wm_chunk_grpo_10ep_%j.err
#SBATCH --export=NIL

set -euo pipefail

_PROJECT_FALLBACK="/vol/bitbucket/aa6622/project"
_COMMON=""
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh" ]]; then
  _COMMON="${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
elif [[ -n "${PROJECT_ROOT:-}" && -f "${PROJECT_ROOT}/scripts/slurm/common_env.sh" ]]; then
  _COMMON="${PROJECT_ROOT}/scripts/slurm/common_env.sh"
elif [[ -f "${_PROJECT_FALLBACK}/scripts/slurm/common_env.sh" ]]; then
  _COMMON="${_PROJECT_FALLBACK}/scripts/slurm/common_env.sh"
fi
if [[ -z "${_COMMON}" || ! -f "${_COMMON}" ]]; then
  echo "error: cannot find scripts/slurm/common_env.sh" >&2
  exit 2
fi
# shellcheck source=../slurm/common_env.sh
source "${_COMMON}"

slurm_resolve_project_root "scripts/grpo/train_phase12_wm_chunk_grpo.py"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "phase12-wm-chunk-grpo-10ep"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export JEPA_WM_DISABLE_IMAGE_HEAD="${JEPA_WM_DISABLE_IMAGE_HEAD:-1}"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
CHECKPOINT="${PHASE12_CHECKPOINT:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
JEPA_CKPT="${PHASE12_JEPA_CKPT:-jepa_wm_metaworld.pth.tar}"
JEPA_REPO="${PHASE12_JEPA_REPO:-/vol/bitbucket/aa6622/VGG JEPA/jepa-wms}"
TASK="${PHASE12_TASK:-push-v3}"
ACTION_PROFILE="${1:-${PHASE12_ACTION_PROFILE:-official_jepa_mirror}}"
OUT="${PHASE12_OUT:-${PROJECT_ROOT}/artifacts/phase12_wm_chunk_grpo_10ep/${ACTION_PROFILE}}"
DRY_RUN_FLAG=()
if [[ "${PHASE12_DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_FLAG+=(--dry-run)
fi

echo "[phase12-10ep] action_profile=${ACTION_PROFILE}"
echo "[phase12-10ep] out=${OUT}"

"${GRPO_PYTHON}" scripts/grpo/train_phase12_wm_chunk_grpo.py \
  --checkpoint "${CHECKPOINT}" \
  --jepa-ckpt "${JEPA_CKPT}" \
  --jepa-repo "${JEPA_REPO}" \
  --output-dir "${OUT}" \
  --task "${TASK}" \
  --env-backend official_lerobot_guarded \
  --action-profile "${ACTION_PROFILE}" \
  --chunk-len 25 \
  --group-size 4 \
  --num-episodes 10 \
  --num-updates 10 \
  --goal-latent-mode visual_proprio \
  --proprio-alpha 0.1 \
  --reward-key wm_latent_progress \
  --ratio-mode chunk \
  --action-transform no_tanh \
  --reset-mismatch fail \
  --decode-candidates selected \
  "${DRY_RUN_FLAG[@]}"

test -f "${OUT}/train_manifest.json"
test -f "${OUT}/progress.jsonl"

echo "PHASE12_WM_CHUNK_GRPO_10EP_DONE profile=${ACTION_PROFILE} out=${OUT}"

