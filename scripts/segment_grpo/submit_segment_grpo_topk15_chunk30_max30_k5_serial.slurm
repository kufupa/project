#!/bin/bash
#SBATCH --job-name=seggrpo-topk15-serial-c30m30k5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/vol/bitbucket/aa6622/project/logs/seggrpo_topk15_serial_c30m30k5_%j.out
#SBATCH --error=/vol/bitbucket/aa6622/project/logs/seggrpo_topk15_serial_c30m30k5_%j.err
#
# One Slurm job runs all 14 remaining top-15 targets sequentially on one GPU.
# Each target uses goal frame 25 from its own oracle episode folder (see run_task.sh).
# Use this when you must avoid nested sbatch (e.g. only one submit from login node,
# or launching from an environment that cannot queue additional batch jobs).
#
# Submit from the cluster login node (not from inside another sbatch body):
#   export SMOLVLA_CHECKPOINT=...
#   export JEPA_REPO=...
#   cd /vol/bitbucket/aa6622/project && sbatch scripts/segment_grpo/submit_segment_grpo_topk15_chunk30_max30_k5_serial.slurm
#
# If you already have a GPU allocation (salloc/srun), run without sbatch:
#   bash scripts/segment_grpo/segment_grpo_topk15_chunk30_max30_k5_run_all_local.sh
#
# rank-1 (1003) excluded — use submit_segment_grpo_campaign1003_chunk30_max30_k5.slurm for that.
set -euo pipefail
# Slurm may execute a *copy* of this script under /var/spool/slurm/...; do not derive
# SCRIPT_DIR from BASH_SOURCE (helpers would be missing). Use repo layout instead.
PROJECT_ROOT="/vol/bitbucket/aa6622/project"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/segment_grpo"
mkdir -p "${PROJECT_ROOT}/logs"

for TASK_ID in $(seq 0 13); do
  echo "[seggrpo-topk15-serial] === starting TASK_ID=${TASK_ID} ==="
  bash "${SCRIPT_DIR}/segment_grpo_topk15_chunk30_max30_k5_run_task.sh" "${TASK_ID}"
done

echo "[seggrpo-topk15-serial] all tasks finished OK"
