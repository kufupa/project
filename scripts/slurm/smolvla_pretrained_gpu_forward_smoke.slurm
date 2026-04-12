#!/bin/bash
# Load jadechoghari/smolvla_metaworld (MetaWorld SmolVLA) + preprocessor, one GPU forward + distr params.
# Submit from repo `project/` root:
#   cd /vol/bitbucket/aa6622/project && sbatch --chdir=/vol/bitbucket/aa6622/project --export=NIL \
#     scripts/slurm/smolvla_pretrained_gpu_forward_smoke.slurm
#
#SBATCH --job-name=smolvla-pretrained-fwd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=smolvla_pretrained_gpu_forward_smoke_%j.out

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
slurm_export_hf_torch_cache "smolvla-pretrained-gpu-forward-smoke"

PY="${LEROBOT_MW_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
if [[ ! -x "${PY}" ]]; then
  echo "error: python not executable: ${PY} (set LEROBOT_MW_PYTHON)" >&2
  exit 2
fi

CHECKPOINT="${SMOLVLA_PRETRAINED_SMOKE_CHECKPOINT:-jadechoghari/smolvla_metaworld}"
echo "using PYTHON=${PY} checkpoint=${CHECKPOINT}"
exec "${PY}" -u "${PROJECT_ROOT}/scripts/slurm/run_smolvla_pretrained_gpu_forward_smoke.py" --checkpoint "${CHECKPOINT}"
