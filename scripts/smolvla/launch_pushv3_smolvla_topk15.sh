#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
SMOLVLA_PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${SMOLVLA_PYTHON_BIN:-${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}/bin/python}"
ARTIFACT_ROOT="${SMOLVLA_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
OUTPUT_ROOT="${SMOLVLA_TOPK_OUTPUT_ROOT:-${ARTIFACT_ROOT}/phase07_smolvla_baseline}"
CAMPAIGNS_ROOT="${OUTPUT_ROOT}/campaigns"
SLURM_SCRIPT="${PROJECT_ROOT}/scripts/smolvla/submit_pushv3_smolvla_topk15.slurm"

ORACLE_RUN_DIR=""
TOP_K=15
DRY_RUN=0
SMOLVLA_ARRAY_SPEC="${SMOLVLA_ARRAY_SPEC:-}"
SMOLVLA_SBATCH_EXPORT_ALL="${SMOLVLA_SBATCH_EXPORT_ALL:-0}"

usage() {
  cat <<'EOF'
Usage: launch_pushv3_smolvla_topk15.sh --oracle-run-dir <path> [--top-k <int>] [--dry-run]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --oracle-run-dir)
      ORACLE_RUN_DIR="${2:-}"
      shift 2
      ;;
    --top-k)
      TOP_K="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${ORACLE_RUN_DIR}" ]]; then
  echo "error: --oracle-run-dir is required." >&2
  usage >&2
  exit 2
fi

if [[ ! -d "${ORACLE_RUN_DIR}" ]]; then
  echo "error: oracle run dir not found: ${ORACLE_RUN_DIR}" >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

if ! [[ "${TOP_K}" =~ ^[0-9]+$ ]] || [[ "${TOP_K}" -le 0 ]]; then
  echo "error: --top-k must be a positive integer, got: ${TOP_K}" >&2
  exit 2
fi

mkdir -p "${CAMPAIGNS_ROOT}"

CAMPAIGN_OUTPUT="$(
  PYTHONPATH="${SMOLVLA_PYTHONPATH}" \
  ORACLE_RUN_DIR="${ORACLE_RUN_DIR}" \
  TOP_K="${TOP_K}" \
  CAMPAIGNS_ROOT="${CAMPAIGNS_ROOT}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import secrets

from src.smolvla_pipeline.targets import load_topk_targets, write_targets_file

oracle_run_dir = Path(os.environ["ORACLE_RUN_DIR"]).expanduser().resolve()
campaigns_root = Path(os.environ["CAMPAIGNS_ROOT"]).expanduser().resolve()
top_k = int(os.environ["TOP_K"])

targets = load_topk_targets(oracle_run_dir, top_k=top_k)
if not targets:
    raise ValueError("No targets found in oracle run report.")

campaign_id = (
    f"pushv3_smolvla_topk{top_k}_"
    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
    f"{secrets.randbelow(1_000_000):06d}"
)
campaign_dir = campaigns_root / campaign_id
campaign_dir.mkdir(parents=True, exist_ok=False)

targets_path = campaign_dir / "targets.json"
write_targets_file(targets_path, targets)

manifest = {
    "schema_version": "smolvla_topk_campaign_v1",
    "campaign_id": campaign_id,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "oracle_run_dir": str(oracle_run_dir),
    "top_k_requested": top_k,
    "targets_count": len(targets),
    "targets_json": str(targets_path),
    "slurm_template": "scripts/smolvla/submit_pushv3_smolvla_topk15.slurm",
}
manifest_path = campaign_dir / "campaign_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print(str(campaign_dir))
print(str(targets_path))
print(str(manifest_path))
print(str(len(targets)))
PY
)"
readarray -t CAMPAIGN_INFO <<<"${CAMPAIGN_OUTPUT}"

CAMPAIGN_DIR="${CAMPAIGN_INFO[0]}"
TARGETS_JSON="${CAMPAIGN_INFO[1]}"
MANIFEST_PATH="${CAMPAIGN_INFO[2]}"
TARGET_COUNT="${CAMPAIGN_INFO[3]}"

ARRAY_MAX=$((TARGET_COUNT - 1))
if [[ "${ARRAY_MAX}" -lt 0 ]]; then
  echo "error: no targets available for submission." >&2
  exit 2
fi

if [[ "${SMOLVLA_SBATCH_EXPORT_ALL:-0}" == "1" ]]; then
  SBATCH_EXPORTS="ALL,TARGETS_JSON=${TARGETS_JSON},SMOLVLA_CAMPAIGN_DIR=${CAMPAIGN_DIR}"
else
  SBATCH_EXPORTS="NONE,TARGETS_JSON=${TARGETS_JSON},SMOLVLA_CAMPAIGN_DIR=${CAMPAIGN_DIR}"
fi
SBATCH_ARRAY="${SMOLVLA_ARRAY_SPEC:-0-${ARRAY_MAX}}"
if ! [[ "${SBATCH_ARRAY}" =~ ^[0-9]+-[0-9]+$ ]]; then
  echo "error: invalid SMOLVLA_ARRAY_SPEC='${SBATCH_ARRAY}'" >&2
  exit 2
fi

SBATCH_ARRAY_START="${SBATCH_ARRAY%-*}"
SBATCH_ARRAY_END="${SBATCH_ARRAY#*-}"
if [[ "${SBATCH_ARRAY_START}" -lt 0 || "${SBATCH_ARRAY_END}" -lt 0 || "${SBATCH_ARRAY_START}" -gt "${SBATCH_ARRAY_END}" ]]; then
  echo "error: invalid array bounds in SMOLVLA_ARRAY_SPEC='${SBATCH_ARRAY}'" >&2
  exit 2
fi
if [[ "${SBATCH_ARRAY_END}" -gt "${ARRAY_MAX}" ]]; then
  echo "error: array end '${SBATCH_ARRAY_END}' exceeds available targets '${ARRAY_MAX}'" >&2
  exit 2
fi

SBATCH_CMD=(sbatch --array "${SBATCH_ARRAY}" --export "${SBATCH_EXPORTS}")
if [[ -n "${SMOLVLA_SBATCH_CPUS_PER_TASK:-}" ]]; then
  SBATCH_CMD+=(--cpus-per-task "${SMOLVLA_SBATCH_CPUS_PER_TASK}")
fi
if [[ -n "${SMOLVLA_SBATCH_MEM:-}" ]]; then
  SBATCH_CMD+=(--mem "${SMOLVLA_SBATCH_MEM}")
fi
if [[ -n "${SMOLVLA_SBATCH_TIME:-}" ]]; then
  SBATCH_CMD+=(--time "${SMOLVLA_SBATCH_TIME}")
fi
if [[ -n "${SMOLVLA_SBATCH_QOS:-}" ]]; then
  SBATCH_CMD+=(--qos "${SMOLVLA_SBATCH_QOS}")
fi
if [[ -n "${SMOLVLA_SBATCH_PARTITION:-}" ]]; then
  SBATCH_CMD+=(--partition "${SMOLVLA_SBATCH_PARTITION}")
fi
if [[ -n "${SMOLVLA_SBATCH_GRES:-}" ]]; then
  SBATCH_CMD+=(--gres "${SMOLVLA_SBATCH_GRES}")
fi
if [[ -n "${SMOLVLA_SBATCH_ACCOUNT:-}" ]]; then
  SBATCH_CMD+=(--account "${SMOLVLA_SBATCH_ACCOUNT}")
fi
if [[ -n "${SMOLVLA_SBATCH_EXTRA_ARGS:-}" ]]; then
  read -r -a _extra_args <<< "${SMOLVLA_SBATCH_EXTRA_ARGS}"
  SBATCH_CMD+=("${_extra_args[@]}")
fi
SBATCH_CMD+=("${SLURM_SCRIPT}")
SUBMISSION_MODE="dry-run"
SUBMISSION_OUTPUT=""

if [[ "${DRY_RUN}" -eq 1 ]]; then
  SUBMISSION_OUTPUT="dry-run requested; did not call sbatch"
elif command -v sbatch >/dev/null 2>&1; then
  SUBMISSION_MODE="sbatch"
  SUBMISSION_OUTPUT="$("${SBATCH_CMD[@]}")"
else
  SUBMISSION_OUTPUT="sbatch unavailable; dry-run fallback"
fi

PYTHONPATH="${SMOLVLA_PYTHONPATH}" \
MANIFEST_PATH="${MANIFEST_PATH}" \
SUBMISSION_MODE="${SUBMISSION_MODE}" \
SUBMISSION_OUTPUT="${SUBMISSION_OUTPUT}" \
ARRAY_SPEC="${SBATCH_ARRAY}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(os.environ["MANIFEST_PATH"]).resolve()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["submission"] = {
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "mode": os.environ["SUBMISSION_MODE"],
    "array": os.environ["ARRAY_SPEC"],
    "output": os.environ["SUBMISSION_OUTPUT"],
}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY

echo "campaign_dir=${CAMPAIGN_DIR}"
echo "targets_json=${TARGETS_JSON}"
echo "targets_count=${TARGET_COUNT}"
echo "array=${SBATCH_ARRAY}"

if [[ "${SUBMISSION_MODE}" == "sbatch" ]]; then
  echo "${SUBMISSION_OUTPUT}"
else
  printf 'dry-run sbatch command:'
  for token in "${SBATCH_CMD[@]}"; do
    printf ' %q' "${token}"
  done
  printf '\n'
  echo "${SUBMISSION_OUTPUT}"
fi
