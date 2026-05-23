#!/usr/bin/env bash
# Monitor queued Phase 50 EGGROLL overnight jobs.

set -euo pipefail

STATE_FILE="${1:?usage: monitor_phase50_eggroll_overnight.sh artifacts/.../queued_jobs.tsv}"
SLEEP_SECONDS="${PHASE50_MONITOR_SLEEP_SECONDS:-300}"
MAX_LOOPS="${PHASE50_MONITOR_MAX_LOOPS:-240}"

if [[ ! -f "${STATE_FILE}" ]]; then
  echo "error: missing state file ${STATE_FILE}" >&2
  exit 2
fi

job_ids=()
run_names=()
run_dirs=()
while IFS=$'\t' read -r job_id run_name run_dir _rest; do
  [[ -z "${job_id}" ]] && continue
  job_ids+=("${job_id}")
  run_names+=("${run_name}")
  run_dirs+=("${run_dir}")
done < "${STATE_FILE}"

if [[ "${#job_ids[@]}" -lt 1 ]]; then
  echo "error: no jobs in ${STATE_FILE}" >&2
  exit 2
fi

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

loop=0
while [[ "${loop}" -lt "${MAX_LOOPS}" ]]; do
  loop=$((loop + 1))
  echo "===== monitor_loop=${loop} utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
  qstat -u "${USER}" || true
  active_count=0
  for idx in "${!job_ids[@]}"; do
    job_id="${job_ids[$idx]}"
    run_name="${run_names[$idx]}"
    run_dir="${run_dirs[$idx]}"
    if qstat "${job_id}" >/dev/null 2>&1; then
      active_count=$((active_count + 1))
      echo "job_active job_id=${job_id} run_name=${run_name}"
    else
      echo "job_inactive job_id=${job_id} run_name=${run_name}"
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
