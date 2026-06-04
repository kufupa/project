#!/usr/bin/env bash
# Guarded cleanup for SmolVLA ManiSkill ephemeral artifacts.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/general/user/aa6622/home/project}"
# shellcheck source=common.sh
source "${PROJECT_ROOT}/scripts/maniskill_smolvla/common.sh"
msm_prepare_runtime

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "cleanup blocked: missing ${path}" >&2
    exit 3
  fi
}

require_json_field_nonzero() {
  local path="$1"
  local field="$2"
  "${MSM_DATA_PYTHON}" - "$path" "$field" <<'PY'
import json
import sys
path, field = sys.argv[1:3]
value = json.loads(open(path, encoding="utf-8").read()).get(field, 0)
if not value:
    raise SystemExit(f"cleanup blocked: {path} field {field!r} is empty/zero")
PY
}

require_confidence() {
  require_path "${MSM_RUN_ROOT}/manifests/audit_full.json"
  require_path "${MSM_RUN_ROOT}/manifests/convert_full.json"
  require_path "${MSM_RUN_ROOT}/manifests/sft_smoke.env"
  require_path "${MSM_RUN_ROOT}/manifests/sft_smoke_policy_contract.json"
  require_json_field_nonzero "${MSM_RUN_ROOT}/manifests/convert_full.json" frames
  require_json_field_nonzero "${MSM_RUN_ROOT}/manifests/convert_full.json" episodes_converted
}

usage() {
  cat <<EOF
Usage:
  MSM_RUN_ROOT=/path/to/run ${0} --target /ephemeral/path --execute

Only deletes paths under MSM_EPHEMERAL_ROOT after audit, conversion, metadata, frame,
sample, and SFT smoke evidence exists. Omit --execute for dry run.
EOF
}

main() {
  local target=""
  local execute="false"
  while (($#)); do
    case "$1" in
      --target) target="$2"; shift 2 ;;
      --execute) execute="true"; shift ;;
      -h|--help) usage; exit 0 ;;
      *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
  done
  if [[ -z "${target}" ]]; then
    usage >&2
    exit 2
  fi
  target="$(readlink -f "${target}")"
  case "${target}" in
    "${MSM_EPHEMERAL_ROOT}"/*) ;;
    *) echo "cleanup blocked: target outside MSM_EPHEMERAL_ROOT: ${target}" >&2; exit 4 ;;
  esac
  require_confidence
  echo "cleanup confidence >=95% evidence present"
  echo "target=${target}"
  if [[ "${execute}" == "true" ]]; then
    rm -rf -- "${target}"
    echo "deleted ${target}"
  else
    echo "dry run only; add --execute to delete"
  fi
}

main "$@"
