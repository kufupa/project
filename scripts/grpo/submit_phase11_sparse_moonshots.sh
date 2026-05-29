#!/usr/bin/env bash
# Submit 2× sparse GRPO moonshot trains + afterok 25ep evals.
set -euo pipefail
cd /rds/general/user/aa6622/home/project
mkdir -p logs/pbs/grpo docs

: > docs/sparse_grpo_moonshot_job_ids.txt
echo "# sparse GRPO moonshots $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> docs/sparse_grpo_moonshot_job_ids.txt

TRAIN_A="$(qsub scripts/grpo/phase11_sparse_b4g16_lr25e6_clip005_moonshot_0050_ephemeral.pbs)"
RUN_A="/rds/general/ephemeral/user/aa6622/ephemeral/smolvla_metaworld/phase11_sparse_b4g16_lr25e6_clip005_u50_${TRAIN_A}"
EVAL_A="$(qsub -W "depend=afterok:${TRAIN_A%%.*}" \
  -v "TRAIN_JOBID=${TRAIN_A},PHASE11_RUN_DIR=${RUN_A}" \
  scripts/grpo/phase11_sparse_b4g16_lr25e6_clip005_moonshot_eval25_stride2.pbs)"
echo "A_TRAIN=${TRAIN_A} A_EVAL=${EVAL_A} A_RUN_DIR=${RUN_A}" | tee -a docs/sparse_grpo_moonshot_job_ids.txt

TRAIN_B="$(qsub scripts/grpo/phase11_sparse_g16_lr5e6_clip02_moonshot_0050_ephemeral.pbs)"
RUN_B="/rds/general/ephemeral/user/aa6622/ephemeral/smolvla_metaworld/phase11_sparse_g16_lr5e6_clip02_u50_${TRAIN_B}"
EVAL_B="$(qsub -W "depend=afterok:${TRAIN_B%%.*}" \
  -v "TRAIN_JOBID=${TRAIN_B},PHASE11_RUN_DIR=${RUN_B}" \
  scripts/grpo/phase11_sparse_g16_lr5e6_clip02_moonshot_eval25_stride2.pbs)"
echo "B_TRAIN=${TRAIN_B} B_EVAL=${EVAL_B} B_RUN_DIR=${RUN_B}" | tee -a docs/sparse_grpo_moonshot_job_ids.txt

qstat -u "$(whoami)"
