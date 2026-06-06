#!/usr/bin/env bash
# Queue E2/E3 DGPO tau ablations when GPU slots free. Run from login node.
set -euo pipefail

PROJECT_ROOT="/vol/bitbucket/aa6622/project"
LOG="${PROJECT_ROOT}/docs/dgpo_overnight_log.md"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

cd "${PROJECT_ROOT}"

wait_for_slot() {
  while true; do
    local cnt
    cnt="$(squeue -h -u "$USER" | wc -l | tr -d ' ')"
    if (( cnt < MAX_PARALLEL )); then
      return 0
    fi
    echo "[dgpo-chain] wait slots count=${cnt} max=${MAX_PARALLEL}"
    sleep "${POLL_SECONDS}"
  done
}

submit_arm() {
  local tag="$1" tau="$2"
  wait_for_slot
  local jid
  jid="$(DGPO_TAU="${tau}" DGPO_KAPPA=0.0 DGPO_REF=frozen_sft \
    sbatch --parsable scripts/grpo/submit_dgpo_chunk_grpo_train_eval_a30.slurm \
    "" "${PROJECT_ROOT}/artifacts/dgpo_chunk_grpo_${tag}" "dgpo_${tag}_tau${tau}")"
  echo "[dgpo-chain] ${tag} tau=${tau} job=${jid}" | tee -a "${LOG}"
}

# Wait for E0/E1 to finish first
for jid in 247467 247468; do
  while squeue -j "${jid}" -h 2>/dev/null | grep -q .; do
    echo "[dgpo-chain] waiting for ${jid}"
    sleep "${POLL_SECONDS}"
  done
  echo "[dgpo-chain] ${jid} done" | tee -a "${LOG}"
done

submit_arm "e2_tau1" "1.0"
submit_arm "e3_tau025" "0.25"
echo "[dgpo-chain] E2/E3 submitted OK"
