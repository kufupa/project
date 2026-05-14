#!/usr/bin/env bash
#SBATCH --job-name=phase12-wm-grpo-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=phase12_wm_chunk_grpo_train_%j.out
#SBATCH --error=phase12_wm_chunk_grpo_train_%j.err
#SBATCH --export=NIL

set -euo pipefail

_PROJECT_FALLBACK="/vol/bitbucket/aa6622/project"
source "${_PROJECT_FALLBACK}/scripts/slurm/common_env.sh"
slurm_resolve_project_root "scripts/grpo/train_phase12_wm_chunk_grpo.py"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "phase12-wm-chunk-grpo-train"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export JEPA_WM_DISABLE_IMAGE_HEAD="${JEPA_WM_DISABLE_IMAGE_HEAD:-0}"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
CHECKPOINT="${PHASE12_CHECKPOINT:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
JEPA_CKPT="${PHASE12_JEPA_CKPT:-jepa_wm_metaworld.pth.tar}"
JEPA_REPO="${PHASE12_JEPA_REPO:-/vol/bitbucket/aa6622/VGG JEPA/jepa-wms}"
ACTION_PROFILE="${1:-${PHASE12_ACTION_PROFILE:-official_jepa_mirror}}"
UPDATES="${2:-${PHASE12_NUM_UPDATES:-100}}"
NUM_EPISODES="${PHASE12_NUM_EPISODES:-${UPDATES}}"
GROUP_SIZE="${PHASE12_GROUP_SIZE:-4}"
SAVE_EVERY="${PHASE12_SAVE_EVERY:-5}"
MAX_STEPS="${PHASE12_MAX_STEPS:-120}"
SEED_BASE="${PHASE12_SEED_BASE:-2000}"
LR="${PHASE12_LR:-1e-5}"
INIT_LOG_STD="${PHASE12_INIT_LOG_STD:--2.0}"
EULER_NOISE="${PHASE12_EULER_NOISE:-0.2}"
ACTION_TRANSFORM="${PHASE12_ACTION_TRANSFORM:-no_tanh}"

OUT="${3:-${PHASE12_OUT:-${PROJECT_ROOT}/artifacts/phase12_wm_chunk_grpo_train/push-v3/g${GROUP_SIZE}_u${UPDATES}_seed${SEED_BASE}_${ACTION_PROFILE}}}"
RESUME="${4:-${PHASE12_RESUME:-}}"
EXTRA=()
if [[ -n "${RESUME}" ]]; then
  EXTRA+=(--resume "${RESUME}")
fi

STRICT_DECODE=()
if [[ "${UPDATES}" == "1" ]]; then
  STRICT_DECODE+=(--strict-decode)
fi

echo "[phase12-train] checkpoint=${CHECKPOINT}"
echo "[phase12-train] jepa_repo=${JEPA_REPO}"
echo "[phase12-train] jepa_ckpt=${JEPA_CKPT}"
echo "[phase12-train] action_profile=${ACTION_PROFILE}"
echo "[phase12-train] updates=${UPDATES}"
echo "[phase12-train] out=${OUT}"
echo "[phase12-train] resume=${RESUME:-<none>}"
echo "[phase12-train] HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
echo "[phase12-train] TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
echo "[phase12-train] JEPA_WM_DISABLE_IMAGE_HEAD=${JEPA_WM_DISABLE_IMAGE_HEAD}"

"${GRPO_PYTHON}" scripts/grpo/train_phase12_wm_chunk_grpo.py \
  --mode wm_grpo_train \
  --checkpoint "${CHECKPOINT}" \
  --jepa-ckpt "${JEPA_CKPT}" \
  --jepa-repo "${JEPA_REPO}" \
  --output-dir "${OUT}" \
  --action-profile "${ACTION_PROFILE}" \
  --num-episodes "${NUM_EPISODES}" \
  --num-updates "${UPDATES}" \
  --train-seed-base "${SEED_BASE}" \
  --chunk-len 25 \
  --group-size "${GROUP_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --save-every "${SAVE_EVERY}" \
  --lr "${LR}" \
  --init-log-std "${INIT_LOG_STD}" \
  --euler-step-noise-std "${EULER_NOISE}" \
  --goal-latent-mode visual_proprio \
  --proprio-alpha 0.1 \
  --reward-key wm_latent_progress \
  --ratio-mode chunk \
  --action-transform "${ACTION_TRANSFORM}" \
  --reset-mismatch fail \
  --decode-candidates selected \
  "${STRICT_DECODE[@]}" \
  "${EXTRA[@]}"

test -f "${OUT}/train_manifest.json"
test -f "${OUT}/progress.jsonl"
test -f "${OUT}/checkpoints/latest.pt"
if [[ "${UPDATES}" -ge 5 ]]; then
  test -f "${OUT}/checkpoints/update_0005.pt"
fi
"${GRPO_PYTHON}" - "${OUT}" "${UPDATES}" "${SEED_BASE}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
expected_updates = int(sys.argv[2])
seed_base = int(sys.argv[3])
progress = out / "progress.jsonl"
rows = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines() if line.strip()]
complete = [row for row in rows if row.get("event") == "update_complete"]
if not complete:
    raise SystemExit("progress.jsonl missing update_complete event")
if not any(row.get("optimizer_step") is True for row in complete):
    raise SystemExit("progress.jsonl missing optimizer_step=true")
tail = complete[-expected_updates:]
if len(tail) != expected_updates:
    raise SystemExit(f"expected {expected_updates} update_complete rows in this run tail, found {len(tail)}")
last_update = int(tail[-1]["update_index"])
expected_seed = seed_base + last_update
if int(tail[-1].get("reset_seed", -1)) != expected_seed:
    raise SystemExit(f"expected final reset_seed {expected_seed}, got {tail[-1].get('reset_seed')}")
checkpoints = out / "checkpoints"
terminal_checkpoint = checkpoints / f"update_{last_update + 1:04d}.pt"
if not terminal_checkpoint.is_file():
    raise SystemExit(f"missing terminal checkpoint {terminal_checkpoint}")
PY
echo "PHASE12_WM_CHUNK_GRPO_TRAIN_DONE profile=${ACTION_PROFILE} updates=${UPDATES} out=${OUT}"
