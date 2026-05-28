#!/usr/bin/env bash
# Delete a prior raw record tree before regen. Refuses unsafe paths.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/general/user/aa6622/home/project}"
# shellcheck source=common.sh
source "${PROJECT_ROOT}/scripts/maniskill_smolvla/common.sh"
msm_prepare_runtime

usage() {
  cat <<'USAGE'
usage: purge_stale_record_dir.sh --stale DIR --new DIR [--dry-run]

Deletes --stale only when:
  - both paths are under MSM_RAW_ROOT
  - stale != new
  - stale is not MSM_RAW_ROOT itself
USAGE
}

stale=""
new_dir=""
dry_run="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stale)
      stale="${2:-}"
      shift 2
      ;;
    --new)
      new_dir="${2:-}"
      shift 2
      ;;
    --dry-run)
      dry_run="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${stale}" || -z "${new_dir}" ]]; then
  echo "missing --stale or --new" >&2
  usage >&2
  exit 2
fi

stale="$(readlink -f "${stale}")"
new_dir="$(readlink -f "${new_dir}")"
raw_root="$(readlink -f "${MSM_RAW_ROOT}")"

case "${stale}" in
  "${raw_root}"/*) ;;
  *)
    echo "purge blocked: stale outside MSM_RAW_ROOT: ${stale}" >&2
    exit 3
    ;;
esac

case "${new_dir}" in
  "${raw_root}"/*) ;;
  *)
    echo "purge blocked: new outside MSM_RAW_ROOT: ${new_dir}" >&2
    exit 3
    ;;
esac

if [[ "${stale}" == "${raw_root}" ]]; then
  echo "purge blocked: refusing to delete MSM_RAW_ROOT" >&2
  exit 3
fi

if [[ "${stale}" == "${new_dir}" ]]; then
  echo "purge blocked: stale equals new record dir" >&2
  exit 3
fi

if [[ ! -e "${stale}" ]]; then
  echo "MSM_PURGE_STALE_SKIP missing stale=${stale}"
  exit 0
fi

echo "MSM_PURGE_STALE_BEGIN stale=${stale} new=${new_dir}"
du -sh "${stale}" || true
if [[ "${dry_run}" == "true" ]]; then
  echo "MSM_PURGE_STALE_DRY_RUN would rm -rf ${stale}"
  exit 0
fi

start_ts="$(date +%s)"
rm -rf -- "${stale}"
end_ts="$(date +%s)"
elapsed="$((end_ts - start_ts))"
echo "MSM_PURGE_STALE_DONE stale=${stale} elapsed_s=${elapsed}"
