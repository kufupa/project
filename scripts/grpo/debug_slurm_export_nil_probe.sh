#!/usr/bin/env bash
# Appends one NDJSON line to Cursor debug log (session 519321). Run on submit host.
# region agent log
set -euo pipefail
LOG="/vol/bitbucket/aa6622/.cursor/debug-519321.log"
T0=$(date +%s%N)
bash -l -c 'true' >/dev/null 2>&1 || true
T1=$(date +%s%N)
MS=$(( (T1 - T0) / 1000000 ))
python3 - <<PY
import json, os, time
log_path = "/vol/bitbucket/aa6622/.cursor/debug-519321.log"
row = {
    "sessionId": "519321",
    "hypothesisId": "H_nil_export_combo",
    "location": "scripts/grpo/debug_slurm_export_nil_probe.sh",
    "message": "Slurm sbatch(1): NIL forbids explicit VAR after NIL; NONE implies get-user-env; get-user-env failure -> requeued+held",
    "data": {
        "bash_login_ms": int(os.environ.get("PROBE_MS", "0")),
        "submit_host": os.uname().nodename,
        "inside_slurm": bool(os.environ.get("SLURM_JOB_ID")),
        "slurm_export_nil_doc": "https://slurm.schedmd.com/sbatch.html --export=NIL",
    },
    "timestamp": int(time.time() * 1000),
}
row["data"]["bash_login_ms"] = int("${MS}")
with open(log_path, "a", encoding="utf-8") as fp:
    fp.write(json.dumps(row) + "\n")
PY
# endregion agent log
echo "wrote one NDJSON line to ${LOG}"
