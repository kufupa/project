#!/bin/bash
# Download jadechoghari/smolvla_metaworld into the shared HF cache from a Slurm compute node.
# Submit from repo `project/` root:
#   cd /vol/bitbucket/aa6622/project && sbatch --chdir=/vol/bitbucket/aa6622/project --export=NIL \
#     scripts/slurm/hf_smolvla_metaworld_cache_warm.slurm
#
#SBATCH --job-name=hf-smolvla-cache
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=hf_smolvla_metaworld_cache_warm_%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh" ]]; then
  _SLURM_COMMON="${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
else
  _SLURM_COMMON="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/common_env.sh"
fi
# shellcheck source=common_env.sh
source "${_SLURM_COMMON}"
slurm_resolve_project_root "scripts/smolvla/legacy_run_pushv3_smolvla_parity_benchmark.sh"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "hf-smolvla-metaworld-cache-warm"

PY="${LEROBOT_MW_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
if [[ ! -x "${PY}" ]]; then
  echo "error: python not executable: ${PY} (set LEROBOT_MW_PYTHON)" >&2
  exit 2
fi

REPO_ID="${HF_CACHE_WARM_REPO_ID:-jadechoghari/smolvla_metaworld}"
REVISION="${HF_CACHE_WARM_REVISION:-}"
echo "using PYTHON=${PY} repo_id=${REPO_ID} revision=${REVISION:-<default>}"

cmd=("${PY}" -u "${PROJECT_ROOT}/scripts/slurm/run_hf_snapshot_cache_warm.py" --repo-id "${REPO_ID}")
if [[ -n "${REVISION}" ]]; then
  cmd+=(--revision "${REVISION}")
fi
exec "${cmd[@]}"
