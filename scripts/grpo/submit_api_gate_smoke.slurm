#!/bin/bash
# GPU smoke: patched SmolVLA + forward/backward on distr params (see check_smolvla_grpo_forward.py).
# Submit from project root:
#   cd /vol/bitbucket/aa6622/project && sbatch --chdir=/vol/bitbucket/aa6622/project --export=NIL \
#     scripts/grpo/submit_api_gate_smoke.slurm
#
#SBATCH --job-name=grpo_api_gate
#SBATCH --output=logs/grpo_api_gate_%j.out
#SBATCH --error=logs/grpo_api_gate_%j.err
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh" ]]; then
  _COMMON="${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
else
  _COMMON="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/../slurm/common_env.sh"
fi
# shellcheck source=../slurm/common_env.sh
source "${_COMMON}"
slurm_resolve_project_root "scripts/grpo/check_smolvla_grpo_forward.py"
cd "${PROJECT_ROOT}"
mkdir -p logs
slurm_export_pythonpath
slurm_export_hf_torch_cache "grpo_api_gate"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
CHECKPOINT="${SMOLVLA_INIT_CHECKPOINT:-jadechoghari/smolvla_metaworld}"

if [[ ! -x "${GRPO_PYTHON}" ]]; then
  echo "error: GRPO_PYTHON not executable: ${GRPO_PYTHON}" >&2
  exit 2
fi

echo "using PYTHON=${GRPO_PYTHON} checkpoint=${CHECKPOINT}"

"${GRPO_PYTHON}" - <<'PY'
import sys

import lerobot.policies.smolvla.modeling_smolvla as m

print("[grpo_api_gate] executable", sys.executable)
print("[grpo_api_gate] modeling", m.__file__)
assert sys.executable.startswith("/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/")
assert "/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/" in m.__file__
PY

exec "${GRPO_PYTHON}" -u scripts/grpo/check_smolvla_grpo_forward.py --checkpoint "${CHECKPOINT}"
