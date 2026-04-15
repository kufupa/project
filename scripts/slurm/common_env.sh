#!/usr/bin/env bash
# Shared helpers for Slurm batch scripts and local bash runs:
#   - Resolve repo `project/` root without trusting BASH_SOURCE from the submitted job
#     (Slurm may execute the batch file from /var/spool/slurm/...).
#   - Export PYTHONPATH (project + src).
#   - Pin Hugging Face / torch caches under ${WORKSPACE_ROOT}/.cache (same idea as MT10 / segment_grpo).
#
# Usage (from a script under scripts/<area>/):
#   _COMMON="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../slurm/common_env.sh"
#   # shellcheck source=../slurm/common_env.sh
#   source "${_COMMON}"
#   slurm_resolve_project_root "scripts/smolvla/legacy_run_pushv3_smolvla_parity_benchmark.sh"
#   cd "${PROJECT_ROOT}"
#   slurm_export_pythonpath
#   slurm_export_hf_torch_cache "my-job-tag"

set -euo pipefail

# Directory containing this file (…/project/scripts/slurm).
_SLURM_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default project root if only relative resolution is possible (…/project).
_SLURM_DEFAULT_PROJECT_ROOT="$(cd "${_SLURM_COMMON_DIR}/../.." && pwd)"

# Resolve PROJECT_ROOT to the repo `project/` directory.
# Args:
#   $1 — marker file path relative to project root (must exist under SLURM_SUBMIT_DIR when valid).
slurm_resolve_project_root() {
  local marker="${1:-scripts/smolvla/legacy_run_pushv3_smolvla_parity_benchmark.sh}"
  if [[ -n "${PROJECT_ROOT:-}" && -f "${PROJECT_ROOT}/${marker}" ]]; then
    export PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
    return 0
  fi
  if [[ -n "${SMOLVLA_PROJECT_ROOT:-}" && -f "${SMOLVLA_PROJECT_ROOT}/${marker}" ]]; then
    export PROJECT_ROOT="$(cd "${SMOLVLA_PROJECT_ROOT}" && pwd)"
    return 0
  fi
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -n "${marker}" && -f "${SLURM_SUBMIT_DIR}/${marker}" ]]; then
    export PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
    return 0
  fi
  export PROJECT_ROOT="${_SLURM_DEFAULT_PROJECT_ROOT}"
  if [[ ! -f "${PROJECT_ROOT}/${marker}" ]]; then
    echo "error: could not resolve PROJECT_ROOT; expected marker missing: ${PROJECT_ROOT}/${marker}" >&2
    echo "error: set PROJECT_ROOT or SMOLVLA_PROJECT_ROOT, or sbatch from repo project/ with SLURM_SUBMIT_DIR set." >&2
    return 2
  fi
}

slurm_export_pythonpath() {
  if [[ -z "${PROJECT_ROOT:-}" ]]; then
    echo "error: PROJECT_ROOT unset before slurm_export_pythonpath" >&2
    return 2
  fi
  export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
}

# Pin caches under workspace .cache (override with HF_HOME / TORCH_HOME / XDG_CACHE_HOME if already set).
slurm_export_hf_torch_cache() {
  local tag="${1:-slurm}"
  if [[ -z "${PROJECT_ROOT:-}" ]]; then
    echo "error: PROJECT_ROOT unset before slurm_export_hf_torch_cache" >&2
    return 2
  fi
  local workspace_root
  workspace_root="$(cd "${PROJECT_ROOT}/.." && pwd)"
  local vol_cache="${workspace_root}/.cache"
  export HF_HOME="${HF_HOME:-${vol_cache}/huggingface}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
  export TORCH_HOME="${TORCH_HOME:-${vol_cache}/torch}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${vol_cache}}"
  echo "[${tag}] PROJECT_ROOT=${PROJECT_ROOT} HF_HOME=${HF_HOME} TORCH_HOME=${TORCH_HOME} host=$(hostname) job=${SLURM_JOB_ID:-local}"
}
