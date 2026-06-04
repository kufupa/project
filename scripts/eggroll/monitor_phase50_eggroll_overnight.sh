#!/usr/bin/env bash
# Monitor queued Phase 50 EGGROLL overnight jobs.

set -euo pipefail

STATE_FILE="${1:?usage: monitor_phase50_eggroll_overnight.sh artifacts/.../queued_jobs.tsv}"
SLEEP_SECONDS="${PHASE50_MONITOR_SLEEP_SECONDS:-300}"
MAX_LOOPS="${PHASE50_MONITOR_MAX_LOOPS:-240}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBMIT_SCRIPT="${PROJECT_ROOT}/scripts/eggroll/submit_phase50_eggroll_overnight_train_eval.pbs"
BASE_CKPT="${PHASE50_CHECKPOINT:-/rds/general/user/aa6622/home/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
STATE_DIR="$(cd "$(dirname "${STATE_FILE}")" && pwd)"

if [[ ! -f "${STATE_FILE}" ]]; then
  echo "error: missing state file ${STATE_FILE}" >&2
  exit 2
fi

job_ids=()
run_names=()
run_dirs=()
run_meta=()
load_state() {
  job_ids=()
  run_names=()
  run_dirs=()
  run_meta=()
  while IFS=$'\t' read -r job_id run_name run_dir rest; do
    [[ -z "${job_id}" ]] && continue
    job_ids+=("${job_id}")
    run_names+=("${run_name}")
    run_dirs+=("${run_dir}")
    run_meta+=("${rest:-}")
  done < "${STATE_FILE}"
}
load_state

if [[ "${#job_ids[@]}" -lt 1 ]]; then
  echo "error: no jobs in ${STATE_FILE}" >&2
  exit 2
fi

meta_value() {
  local meta="$1"
  local key="$2"
  local field
  for field in ${meta}; do
    if [[ "${field}" == "${key}="* ]]; then
      printf '%s\n' "${field#*=}"
      return 0
    fi
  done
  return 1
}

scan_run() {
  local run_name="$1"
  local run_dir="$2"
  python3 - "${run_name}" "${run_dir}" <<'PY'
from pathlib import Path
import json
import sys

run_name = sys.argv[1]
run_dir = Path(sys.argv[2])
progress = run_dir / "progress.jsonl"
timings = run_dir / "timings.jsonl"
manifest = run_dir / "train_manifest.json"
summary = run_dir / "overnight_summary.json"
pbs = run_dir / "pbs.out"
rows = progress.read_text().splitlines() if progress.exists() else []
timing_rows = timings.read_text().splitlines() if timings.exists() else []
last = {}
if rows:
    try:
        last = json.loads(rows[-1])
    except Exception:
        last = {"parse_error": rows[-1][:200]}
signals = []
if pbs.exists():
    text = pbs.read_text(errors="replace")[-200000:]
    for needle in (
        "Traceback",
        "RuntimeError",
        "CUDA out of memory",
        "BrokenPipeError",
        "FileNotFoundError",
        "Killed",
        "error:",
        "PHASE50_EGGROLL_OVERNIGHT_DONE",
    ):
        if needle in text:
            signals.append(needle)
best = None
if summary.exists():
    try:
        best = json.loads(summary.read_text()).get("best_eval_row")
    except Exception as exc:
        best = {"parse_error": str(exc)}
manifest_bits = {}
if manifest.exists():
    try:
        data = json.loads(manifest.read_text())
        cfg = data.get("config", {})
        manifest_bits = {
            "manifest_seed_mode": data.get("reset_seed_mode"),
            "config_seed_mode": cfg.get("seed_mode"),
            "episodes_per_member": cfg.get("episodes_per_member"),
            "checkpoint_sync_dir": cfg.get("checkpoint_sync_dir"),
        }
    except Exception as exc:
        manifest_bits = {"manifest_parse_error": str(exc)}
checkpoint_count = len(list((run_dir / "checkpoints").glob("update_*.pt")))
print(json.dumps({
    "run_name": run_name,
    "run_dir": str(run_dir),
    "progress_rows": len(rows),
    "timing_rows": len(timing_rows),
    "checkpoint_count": checkpoint_count,
    "last_iteration": last.get("iteration"),
    "last_checkpoint_update": last.get("checkpoint_update"),
    "last_fitness_best": last.get("fitness_best"),
    "last_success_rate": last.get("success_rate"),
    "last_iteration_seconds": last.get("iteration_seconds"),
    "last_seed_mode": last.get("seed_mode"),
    "last_episodes_per_member": last.get("episodes_per_member"),
    "signals": signals,
    "best_eval_row": best,
    **manifest_bits,
}, sort_keys=True))
PY
}

diagnose_run() {
  local run_name="$1"
  local run_dir="$2"
  python3 - "${run_name}" "${run_dir}" <<'PY'
from pathlib import Path
import json
import sys

run_name = sys.argv[1]
run_dir = Path(sys.argv[2])
pbs = run_dir / "pbs.out"
progress = run_dir / "progress.jsonl"
summary = run_dir / "overnight_summary.json"
text = pbs.read_text(errors="replace") if pbs.exists() else ""
tail = text[-20000:]
signals = []
for needle in (
    "CUDA out of memory",
    "OutOfMemoryError",
    "BrokenPipeError",
    "FileNotFoundError",
    "Traceback",
    "RuntimeError",
    "Killed",
    "PBS: job killed",
    "walltime",
    "error:",
):
    if needle in tail:
        signals.append(needle)
if "CUDA out of memory" in tail or "OutOfMemoryError" in tail or "Killed" in tail:
    root_cause = "resource_or_oom"
elif "BrokenPipeError" in tail:
    root_cause = "worker_process_failure"
elif "FileNotFoundError" in tail:
    root_cause = "filesystem_or_source_visibility"
elif "walltime" in tail or "PBS: job killed" in tail:
    root_cause = "walltime"
elif "Traceback" in tail or "RuntimeError" in tail or "error:" in tail:
    root_cause = "runtime_error"
else:
    root_cause = "unknown"
progress_rows = progress.read_text(errors="replace").splitlines() if progress.exists() else []
payload = {
    "run_name": run_name,
    "run_dir": str(run_dir),
    "root_cause": root_cause,
    "signals": signals,
    "progress_rows": len(progress_rows),
    "summary_exists": summary.exists(),
    "checkpoint_count": len(list((run_dir / "checkpoints").glob("update_*.pt"))),
    "pbs_tail": tail[-4000:],
}
(run_dir / "root_cause_diagnosis.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
}

append_fallback_run() {
  local old_job_id="$1"
  local run_name="$2"
  local run_dir="$3"
  local meta="$4"
  local batch
  batch="$(meta_value "${meta}" batch || echo "")"
  local fallback_of
  fallback_of="$(meta_value "${meta}" fallback_of || echo "")"
  local fallback_batch fallback_ncpus fallback_mem
  case "${batch}" in
    64)
      fallback_batch=32
      fallback_ncpus=64
      fallback_mem=192gb
      ;;
    32)
      fallback_batch=16
      fallback_ncpus=32
      fallback_mem=96gb
      ;;
    *)
      return 0
      ;;
  esac
  if [[ -n "${fallback_of}" ]]; then
    return 0
  fi
  if grep -Fq "fallback_of=${old_job_id}" "${STATE_FILE}"; then
    return 0
  fi

  local pop rank sigma alpha iters seed_base episodes_per_member abort_update_norm resume
  pop="$(meta_value "${meta}" pop)"
  rank="$(meta_value "${meta}" rank)"
  sigma="$(meta_value "${meta}" sigma)"
  alpha="$(meta_value "${meta}" alpha)"
  iters="$(meta_value "${meta}" iters)"
  seed_base="$(meta_value "${meta}" seed_base)"
  episodes_per_member="$(meta_value "${meta}" episodes_per_member)"
  abort_update_norm="$(meta_value "${meta}" abort_update_norm)"
  resume="$(meta_value "${meta}" resume || echo "")"

  local fallback_name fallback_dir fallback_pbs env_vars job_id
  fallback_name="${run_name}_fallback_b${fallback_batch}"
  fallback_dir="${run_dir}_fallback_b${fallback_batch}"
  fallback_pbs="$(printf '%s' "${fallback_name}" | tr -cd '[:alnum:]' | cut -c1-14)"
  mkdir -p "${fallback_dir}"
  env_vars="PHASE50_RUN_NAME=${fallback_name},PHASE50_OUT=${fallback_dir},PHASE50_CHECKPOINT=${BASE_CKPT},PHASE50_EVAL_BASE_CHECKPOINT=${BASE_CKPT},PHASE50_RESUME=${resume},PHASE50_TASK=push-v3,PHASE50_POPULATION_SIZE=${pop},PHASE50_POPULATION_BATCH_SIZE=${fallback_batch},PHASE50_RANK=${rank},PHASE50_SIGMA=${sigma},PHASE50_ALPHA=${alpha},PHASE50_ABORT_UPDATE_NORM=${abort_update_norm},PHASE50_NUM_ITERATIONS=${iters},PHASE50_MAX_STEPS=120,PHASE50_EPISODES_PER_MEMBER=${episodes_per_member},PHASE50_SEED_MODE=shared_per_iteration,PHASE50_ACTION_CHUNK_SIZE=5,PHASE50_ROLLOUT_EXECUTION=vector_async,PHASE50_FITNESS_SHAPING=rank,PHASE50_BASELINE_TYPE=mean,PHASE50_TRAIN_SEED_BASE=${seed_base},PHASE50_SAVE_EVERY=2,PHASE50_VIDEO_EVERY=10,PHASE50_EVAL_EPISODES=50,PHASE50_EVAL_N_ENVS=25,PHASE50_EVAL_STRIDE=2,PHASE50_EVAL_SWEEP_NAME=eval_seeded50_nenv25_every2,SMOLVLA_METAWORLD_RESET_MODE=random_seeded"
  job_id="$(cd "${PROJECT_ROOT}" && qsub -N "${fallback_pbs}" -l select=1:ncpus=${fallback_ncpus}:mem=${fallback_mem}:ngpus=1:gpu_type=RTX6000 -o "${fallback_dir}/pbs.out" -v "${env_vars}" "${SUBMIT_SCRIPT}")"
  printf '%s\t%s\t%s\tpbs_name=%s\tpop=%s\tbatch=%s\trank=%s\tsigma=%s\talpha=%s\titers=%s\tseed_base=%s\tepisodes_per_member=%s\tabort_update_norm=%s\tmax_steps=120\teval_n_envs=25\tseed_mode=shared_per_iteration\tresume=%s\tfallback_of=%s\n' \
    "${job_id}" "${fallback_name}" "${fallback_dir}" "${fallback_pbs}" "${pop}" "${fallback_batch}" "${rank}" "${sigma}" "${alpha}" "${iters}" "${seed_base}" "${episodes_per_member}" "${abort_update_norm}" "${resume}" "${old_job_id}" | tee -a "${STATE_FILE}"
  echo "PHASE50_EGGROLL_FALLBACK_QUEUED failed_job=${old_job_id} fallback_job=${job_id} run_name=${fallback_name} out=${fallback_dir}"
}

loop=0
while [[ "${loop}" -lt "${MAX_LOOPS}" ]]; do
  loop=$((loop + 1))
  load_state
  echo "===== monitor_loop=${loop} utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
  qstat -u "${USER}" || true
  active_count=0
  for idx in "${!job_ids[@]}"; do
    job_id="${job_ids[$idx]}"
    run_name="${run_names[$idx]}"
    run_dir="${run_dirs[$idx]}"
    meta="${run_meta[$idx]}"
    if qstat "${job_id}" >/dev/null 2>&1; then
      active_count=$((active_count + 1))
      echo "job_active job_id=${job_id} run_name=${run_name}"
    else
      echo "job_inactive job_id=${job_id} run_name=${run_name}"
      if [[ ! -f "${run_dir}/overnight_summary.json" && -f "${run_dir}/pbs.out" ]]; then
        if grep -Eq "Traceback|RuntimeError|CUDA out of memory|OutOfMemoryError|BrokenPipeError|FileNotFoundError|Killed|PBS: job killed|error:" "${run_dir}/pbs.out"; then
          echo "PHASE50_EGGROLL_FAILURE_DIAGNOSIS job_id=${job_id} run_name=${run_name}"
          diagnose_run "${run_name}" "${run_dir}" || true
          fallback_count_before="$(grep -Fc "fallback_of=${job_id}" "${STATE_FILE}" || true)"
          append_fallback_run "${job_id}" "${run_name}" "${run_dir}" "${meta}" || true
          fallback_count_after="$(grep -Fc "fallback_of=${job_id}" "${STATE_FILE}" || true)"
          if [[ "${fallback_count_after}" -gt "${fallback_count_before}" ]]; then
            active_count=$((active_count + 1))
          fi
        fi
      fi
    fi
    scan_run "${run_name}" "${run_dir}" || true
  done
  if [[ "${active_count}" -eq 0 ]]; then
    echo "PHASE50_EGGROLL_OVERNIGHT_MONITOR_DONE state=${STATE_FILE}"
    exit 0
  fi
  sleep "${SLEEP_SECONDS}"
done

echo "error: monitor reached MAX_LOOPS=${MAX_LOOPS}" >&2
exit 4
