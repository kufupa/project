#!/usr/bin/env bash
# Queue exactly two Phase 50 EGGROLL overnight RTX6000 jobs.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATE_TAG="${PHASE50_DATE_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
ROOT="${PHASE50_OVERNIGHT_ROOT:-${PROJECT_ROOT}/artifacts/phase50_eggroll_overnight/${DATE_TAG}}"
STATE="${ROOT}/queued_jobs.tsv"
SCRIPT="${PROJECT_ROOT}/scripts/eggroll/submit_phase50_eggroll_overnight_train_eval.pbs"
BASE_CKPT="${PHASE50_CHECKPOINT:-/rds/general/user/aa6622/home/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
RESUME_CKPT="${PROJECT_ROOT}/artifacts/phase50_eggroll_train/push-v3/pop32_b16_r2_seed2000_alpha0015_rank/checkpoints/update_0010.pt"

mkdir -p "${ROOT}"
: > "${STATE}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "error: missing ${SCRIPT}" >&2
  exit 2
fi
if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "error: missing resume checkpoint ${RESUME_CKPT}" >&2
  exit 2
fi

GPU_SNAPSHOT="${ROOT}/gpu_snapshot.txt"
python3 "${HOME}/.agents/skills/checking-pbs-gpu-availability/scripts/pbs_gpu_snapshot.py" -q v1_gpu72 | tee "${GPU_SNAPSHOT}"

free_rtx6000="$(${PYTHON:-python3} - "${GPU_SNAPSHOT}" <<'PY'
from pathlib import Path
import re
import sys

total = 0
for line in Path(sys.argv[1]).read_text().splitlines():
    if "gpu_model=RTX6000" not in line:
        continue
    match = re.search(r"free_gpus=(\d+)", line)
    if match:
        total += int(match.group(1))
print(total)
PY
)"
if [[ "${free_rtx6000}" -lt 2 ]]; then
  echo "error: need at least 2 eligible free RTX6000 GPUs on v1_gpu72; got ${free_rtx6000}" >&2
  echo "Do not use v1_jupytergpu from login/batch; Jupyter nodes are excluded." >&2
  exit 3
fi

submit_run() {
  local run_name="$1"
  local pbs_name="$2"
  local out="$3"
  local resume="$4"
  local pop="$5"
  local batch="$6"
  local rank="$7"
  local sigma="$8"
  local alpha="$9"
  local iters="${10}"
  local seed_base="${11}"
  mkdir -p "${out}"
  local env_vars
  env_vars="PHASE50_RUN_NAME=${run_name},PHASE50_OUT=${out},PHASE50_CHECKPOINT=${BASE_CKPT},PHASE50_EVAL_BASE_CHECKPOINT=${BASE_CKPT},PHASE50_RESUME=${resume},PHASE50_TASK=push-v3,PHASE50_POPULATION_SIZE=${pop},PHASE50_POPULATION_BATCH_SIZE=${batch},PHASE50_RANK=${rank},PHASE50_SIGMA=${sigma},PHASE50_ALPHA=${alpha},PHASE50_NUM_ITERATIONS=${iters},PHASE50_MAX_STEPS=120,PHASE50_EPISODES_PER_MEMBER=2,PHASE50_SEED_MODE=shared_per_iteration,PHASE50_ACTION_CHUNK_SIZE=5,PHASE50_ROLLOUT_EXECUTION=vector_async,PHASE50_FITNESS_SHAPING=rank,PHASE50_BASELINE_TYPE=mean,PHASE50_TRAIN_SEED_BASE=${seed_base},PHASE50_SAVE_EVERY=2,PHASE50_VIDEO_EVERY=10,PHASE50_EVAL_EPISODES=50,PHASE50_EVAL_N_ENVS=3,PHASE50_EVAL_STRIDE=2,PHASE50_EVAL_SWEEP_NAME=eval_seeded50_every2,SMOLVLA_METAWORLD_RESET_MODE=random_seeded"
  local job_id
  job_id="$(qsub -N "${pbs_name}" -o "${out}/pbs.out" -v "${env_vars}" "${SCRIPT}")"
  printf '%s\t%s\t%s\tpop=%s\tbatch=%s\trank=%s\tsigma=%s\talpha=%s\titers=%s\tseed_mode=shared_per_iteration\n' "${job_id}" "${run_name}" "${out}" "${pop}" "${batch}" "${rank}" "${sigma}" "${alpha}" "${iters}" | tee -a "${STATE}"
}

submit_run "run_a_fresh_pop32_r2_seedbatch2_full120" "p50eggAnew" "${ROOT}/run_a_fresh_pop32_r2_seedbatch2_full120" "" 32 16 2 0.01 0.01 36 2000
submit_run "run_b_resume_u10_pop64_r1_seedbatch2_full120" "p50eggB64r1" "${ROOT}/run_b_resume_u10_pop64_r1_seedbatch2_full120" "${RESUME_CKPT}" 64 16 1 0.01 0.01 20 3000

echo "PHASE50_EGGROLL_OVERNIGHT_QUEUED root=${ROOT} state=${STATE}"
qstat -u "${USER}"
