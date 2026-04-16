#!/bin/bash
#SBATCH --job-name=seggrpo-c1003-topk15-c30m30k5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/vol/bitbucket/aa6622/project/logs/seggrpo_topk15_c30m30k5_%A_%a.out
#SBATCH --error=/vol/bitbucket/aa6622/project/logs/seggrpo_topk15_c30m30k5_%A_%a.err
#SBATCH --array=0-13
#
# Campaign top-15 (ranks 2-15), one segment each — same params as
# submit_segment_grpo_campaign1003_chunk30_max30_k5.slurm (per-task episode/seed).
# Oracle goal for each task: --goal-frame-index 25 PNG under that task's
# frames/episode_{episode_index}/ in the baseline oracle run (paired reset-seed).
#
# Submit once from the Slurm login node only. Do not sbatch this script from inside
# another batch job (nested submit can fail with QOSMaxSubmitJobPerUserLimit).
# For a single submit that runs all 14 targets in one allocation, use instead:
#   scripts/segment_grpo/submit_segment_grpo_topk15_chunk30_max30_k5_serial.slurm
# From an interactive GPU shell without any sbatch:
#   bash scripts/segment_grpo/segment_grpo_topk15_chunk30_max30_k5_run_all_local.sh
#
# Submit:
#   export SMOLVLA_CHECKPOINT=...
#   export JEPA_REPO=...
#   cd /vol/bitbucket/aa6622/project && sbatch scripts/segment_grpo/submit_segment_grpo_topk15_chunk30_max30_k5_array.slurm
#
# rank-1 seed 1003 is intentionally excluded — use:
#   scripts/segment_grpo/submit_segment_grpo_campaign1003_chunk30_max30_k5.slurm
#
# Optional concurrency: change --array=0-13 to --array=0-13%N.
set -euo pipefail
# Slurm may stage this script under /var/spool/slurm/...; helpers live in the repo only.
PROJECT_ROOT="/vol/bitbucket/aa6622/project"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/segment_grpo"
mkdir -p "${PROJECT_ROOT}/logs"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[seggrpo-c1003-topk15-c30m30k5] SLURM_ARRAY_TASK_ID is required"
  exit 1
fi

bash "${SCRIPT_DIR}/segment_grpo_topk15_chunk30_max30_k5_run_task.sh" "${SLURM_ARRAY_TASK_ID}"
