#!/usr/bin/env bash
# qdel train jobs once training finishes so inline 50ep eval does not run.
# Running jobs use the PBS script copied at job start; skip markers alone are not enough.

set -euo pipefail

STATE_FILE="${1:?usage: watch_phase50_skip_inline_eval.sh queued_jobs.tsv}"
POLL_SECONDS="${PHASE50_SKIP_WATCH_POLL_SECONDS:-60}"
STABLE_SECONDS="${PHASE50_SKIP_WATCH_STABLE_SECONDS:-180}"

job_ids=()
run_names=()
run_dirs=()
while IFS=$'\t' read -r job_id run_name run_dir _rest; do
  [[ -z "${job_id}" ]] && continue
  [[ ! -f "${run_dir}/.phase50_skip_final_eval" ]] && continue
  job_ids+=("${job_id%%.*}")
  run_names+=("${run_name}")
  run_dirs+=("${run_dir}")
done < "${STATE_FILE}"

if [[ "${#job_ids[@]}" -lt 1 ]]; then
  echo "watch_skip_inline_eval: no skip-marked runs in ${STATE_FILE}"
  exit 0
fi

train_done() {
  local run_dir="$1"
  python3 - "${run_dir}" <<'PY'
from pathlib import Path
import json
import sys

run_dir = Path(sys.argv[1])
manifest = run_dir / "train_manifest.json"
progress = run_dir / "progress.jsonl"
if not manifest.is_file() or not progress.is_file():
    print("no")
    raise SystemExit(0)
manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
target = int(manifest_data.get("config", {}).get("num_iterations", 0))
rows = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines() if line.strip()]
if target < 1 or not rows:
    print("no")
    raise SystemExit(0)
last = rows[-1]
last_iter = int(last.get("iteration", -1))
if last_iter >= target - 1:
    print("yes")
else:
    print("no")
PY
}

echo "watch_skip_inline_eval: watching ${#job_ids[@]} run(s) poll=${POLL_SECONDS}s stable=${STABLE_SECONDS}s"
while true; do
  active=0
  for idx in "${!job_ids[@]}"; do
    job_id="${job_ids[$idx]}"
    run_name="${run_names[$idx]}"
    run_dir="${run_dirs[$idx]}"
    if ! qstat "${job_id}" >/dev/null 2>&1; then
      continue
    fi
    active=$((active + 1))
    if [[ "$(train_done "${run_dir}")" != "yes" ]]; then
      continue
    fi
    stable_key="${run_dir}/.phase50_skip_train_done_since"
    now="$(date -u +%s)"
    if [[ ! -f "${stable_key}" ]]; then
      echo "${now}" > "${stable_key}"
      echo "watch_skip_inline_eval: train_done_pending run=${run_name} job=${job_id}"
      continue
    fi
    since="$(<"${stable_key}")"
    if [[ $((now - since)) -lt "${STABLE_SECONDS}" ]]; then
      continue
    fi
    echo "watch_skip_inline_eval: qdel job=${job_id} run=${run_name} (skip inline eval)"
    qdel "${job_id}" || true
    rm -f "${stable_key}"
  done
  if [[ "${active}" -eq 0 ]]; then
    echo "watch_skip_inline_eval: done"
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done
