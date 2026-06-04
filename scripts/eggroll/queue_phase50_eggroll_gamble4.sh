#!/usr/bin/env bash
# Queue four high-risk Phase 50 EGGROLL RTX6000 gamble jobs.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATE_TAG="${PHASE50_DATE_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
ROOT="${PHASE50_GAMBLE_ROOT:-${PROJECT_ROOT}/artifacts/phase50_eggroll_gamble/${DATE_TAG}}"
STATE="${ROOT}/queued_jobs.tsv"
SCRIPT="${PROJECT_ROOT}/scripts/eggroll/submit_phase50_eggroll_overnight_train_eval.pbs"
BASE_CKPT="${PHASE50_CHECKPOINT:-/rds/general/user/aa6622/home/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
G_RUN_DIR="${PHASE50_G_RUN_DIR:-${PROJECT_ROOT}/artifacts/phase50_eggroll_next/20260523_203005/run_g_fresh_pop128_b32_r1_seedbatch2_alpha0035_full120}"
G_BEST_CKPT="${PHASE50_G_BEST_CKPT:-${G_RUN_DIR}/checkpoints/update_0004.pt}"

mkdir -p "${ROOT}"
: > "${STATE}"

for required in "${SCRIPT}" "${BASE_CKPT}" "${G_BEST_CKPT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "error: missing required file ${required}" >&2
    exit 2
  fi
done

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
if [[ "${free_rtx6000}" -lt 4 ]]; then
  echo "warning: fewer than 4 currently free eligible RTX6000 GPUs on v1_gpu72; got ${free_rtx6000}; submitting anyway so PBS can queue." >&2
  echo "Do not use v1_jupytergpu from login/batch; Jupyter nodes are excluded." >&2
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
  local episodes_per_member="${12}"
  local abort_update_norm="${13}"
  local ncpus="${14}"
  local mem="${15}"
  mkdir -p "${out}"

  local env_vars
  env_vars="PHASE50_RUN_NAME=${run_name},PHASE50_OUT=${out},PHASE50_CHECKPOINT=${BASE_CKPT},PHASE50_EVAL_BASE_CHECKPOINT=${BASE_CKPT},PHASE50_RESUME=${resume},PHASE50_TASK=push-v3,PHASE50_POPULATION_SIZE=${pop},PHASE50_POPULATION_BATCH_SIZE=${batch},PHASE50_RANK=${rank},PHASE50_SIGMA=${sigma},PHASE50_ALPHA=${alpha},PHASE50_ABORT_UPDATE_NORM=${abort_update_norm},PHASE50_NUM_ITERATIONS=${iters},PHASE50_MAX_STEPS=120,PHASE50_EPISODES_PER_MEMBER=${episodes_per_member},PHASE50_SEED_MODE=shared_per_iteration,PHASE50_ACTION_CHUNK_SIZE=5,PHASE50_ROLLOUT_EXECUTION=vector_async,PHASE50_FITNESS_SHAPING=rank,PHASE50_BASELINE_TYPE=mean,PHASE50_TRAIN_SEED_BASE=${seed_base},PHASE50_SAVE_EVERY=2,PHASE50_VIDEO_EVERY=10,PHASE50_EVAL_EPISODES=50,PHASE50_EVAL_N_ENVS=25,PHASE50_EVAL_STRIDE=2,PHASE50_EVAL_SWEEP_NAME=eval_seeded50_nenv25_every2,SMOLVLA_METAWORLD_RESET_MODE=random_seeded"

  local job_id
  job_id="$(qsub -N "${pbs_name}" -l select=1:ncpus="${ncpus}":mem="${mem}":ngpus=1:gpu_type=RTX6000 -o "${out}/pbs.out" -v "${env_vars}" "${SCRIPT}")"
  printf '%s\t%s\t%s\tpbs_name=%s\tpop=%s\tbatch=%s\trank=%s\tsigma=%s\talpha=%s\titers=%s\tseed_base=%s\tepisodes_per_member=%s\tabort_update_norm=%s\tmax_steps=120\teval_n_envs=25\tseed_mode=shared_per_iteration\tresume=%s\tfallback_of=\tncpus=%s\tmem=%s\n' \
    "${job_id}" "${run_name}" "${out}" "${pbs_name}" "${pop}" "${batch}" "${rank}" "${sigma}" "${alpha}" "${iters}" "${seed_base}" "${episodes_per_member}" "${abort_update_norm}" "${resume}" "${ncpus}" "${mem}" | tee -a "${STATE}"
}

submit_run \
  "run_k_g004_pop256_b32_r1_sigma003_alpha01_ep1_u30" \
  "p50eggK256b32" \
  "${ROOT}/run_k_g004_pop256_b32_r1_sigma003_alpha01_ep1_u30" \
  "${G_BEST_CKPT}" \
  256 32 1 0.03 0.01 30 15000 1 0.08 64 192gb

submit_run \
  "run_l_g004_pop256_b32_r1_sigma005_alpha015_ep1_u24" \
  "p50eggL256b32" \
  "${ROOT}/run_l_g004_pop256_b32_r1_sigma005_alpha015_ep1_u24" \
  "${G_BEST_CKPT}" \
  256 32 1 0.05 0.015 24 17000 1 0.10 64 192gb

submit_run \
  "run_m_fresh_pop256_b32_r1_sigma010_alpha01_ep1_u30" \
  "p50eggM256b32" \
  "${ROOT}/run_m_fresh_pop256_b32_r1_sigma010_alpha01_ep1_u30" \
  "" \
  256 32 1 0.10 0.01 30 19000 1 0.10 64 192gb

submit_run \
  "run_n_g004_pop256_b64_r1_sigma003_alpha012_ep1_u30" \
  "p50eggN256b64" \
  "${ROOT}/run_n_g004_pop256_b64_r1_sigma003_alpha012_ep1_u30" \
  "${G_BEST_CKPT}" \
  256 64 1 0.03 0.012 30 21000 1 0.10 32 192gb

echo "PHASE50_EGGROLL_GAMBLE_QUEUED root=${ROOT} state=${STATE}"
qstat -u "${USER}"
