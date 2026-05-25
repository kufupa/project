#!/usr/bin/env bash
# Queue next 2.5h PIRL ManiSkill training chunk when gate allows.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=pirl_maniskill_common.sh
source "${SCRIPT_DIR}/pirl_maniskill_common.sh"

pirl_setup_modules
pirl_prepare_runtime

if [[ "${PIRL_AUTO_QUEUE:-0}" != "1" ]]; then
  echo "[pirl-queue] PIRL_AUTO_QUEUE!=1; not queueing."
  exit 0
fi

if [[ "${PIRL_GATE_STATUS:-}" == "halt" || -f "${PIRL_RUN_ROOT}/HALT" ]]; then
  echo "[pirl-queue] halt gate set; not queueing."
  exit 0
fi

if [[ "${PIRL_GATE_STATUS:-}" != "pass" && ! -f "${PIRL_RUN_ROOT}/GATE_PASS" ]]; then
  echo "[pirl-queue] no eval gate pass; set PIRL_GATE_STATUS=pass or create ${PIRL_RUN_ROOT}/GATE_PASS."
  exit 0
fi

PIRL_CHUNK_INDEX="${PIRL_CHUNK_INDEX:-0}"
PIRL_MAX_CHUNKS="${PIRL_MAX_CHUNKS:-4}"
if [[ "${PIRL_CHUNK_INDEX}" -ge "${PIRL_MAX_CHUNKS}" ]]; then
  echo "[pirl-queue] chunk index ${PIRL_CHUNK_INDEX} >= max ${PIRL_MAX_CHUNKS}; done."
  exit 0
fi

if [[ -z "${PBS_JOBID:-}" && "${PIRL_ALLOW_LOCAL_QUEUE:-0}" != "1" ]]; then
  echo "[pirl-queue] local run detected; not recursively queueing."
  exit 0
fi

pirl_gpu_snapshot

latest="${PIRL_RESUME_DIR:-}"
if [[ -z "${latest}" ]]; then
  latest="$(pirl_latest_checkpoint "${PIRL_ARTIFACT_ROOT}" || true)"
fi

next_index=$((PIRL_CHUNK_INDEX + 1))
script="${PIRL_CHUNK_SCRIPT:-${SCRIPT_DIR}/pirl_maniskill_2gpu_chunk.pbs}"
if [[ ! -f "${script}" ]]; then
  echo "error: missing chunk script ${script}" >&2
  exit 2
fi

env_vars="PROJECT_ROOT=${PROJECT_ROOT},PIRL_AUTO_QUEUE=${PIRL_AUTO_QUEUE},PIRL_GATE_STATUS=pass,PIRL_CHUNK_INDEX=${next_index},PIRL_MAX_CHUNKS=${PIRL_MAX_CHUNKS},PIRL_ARTIFACT_ROOT=${PIRL_ARTIFACT_ROOT},PIRL_SFT_CKPT=${PIRL_SFT_CKPT}"
if [[ -n "${latest}" ]]; then
  env_vars="${env_vars},PIRL_RESUME_DIR=${latest}"
fi
if [[ -n "${PIRL_TRAIN_ENVS:-}" ]]; then env_vars="${env_vars},PIRL_TRAIN_ENVS=${PIRL_TRAIN_ENVS}"; fi
if [[ -n "${PIRL_ROLLOUT_EPOCH:-}" ]]; then env_vars="${env_vars},PIRL_ROLLOUT_EPOCH=${PIRL_ROLLOUT_EPOCH}"; fi
if [[ -n "${PIRL_MAX_STEPS:-}" ]]; then env_vars="${env_vars},PIRL_MAX_STEPS=${PIRL_MAX_STEPS}"; fi
if [[ -n "${PIRL_MICRO_BATCH:-}" ]]; then env_vars="${env_vars},PIRL_MICRO_BATCH=${PIRL_MICRO_BATCH}"; fi
if [[ -n "${PIRL_GLOBAL_BATCH:-}" ]]; then env_vars="${env_vars},PIRL_GLOBAL_BATCH=${PIRL_GLOBAL_BATCH}"; fi

queue_root="${PIRL_ARTIFACT_ROOT}/queued_chunks"
mkdir -p "${queue_root}"
job_id="$(qsub -q v1_gpu72 -N "pirlmsk${next_index}" -o "${queue_root}/chunk_${next_index}.pbs.out" -v "${env_vars}" "${script}")"
echo "PIRL_MANISKILL_NEXT_CHUNK_QUEUED job=${job_id} index=${next_index} resume=${latest:-none}"
