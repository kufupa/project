#!/bin/bash
# GPU import smoke for production LeRobot venv after site-packages SmolVLA patch.
# Submit from repo `project/` root:
#   cd /vol/bitbucket/aa6622/project && sbatch --export=NIL scripts/slurm/smolvla_lerobot_pkg_import_gpu_smoke.slurm
# Or set SMOLVLA_PROJECT_ROOT to project/ before sbatch.
#
#SBATCH --job-name=smolvla-pkg-import
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=smolvla_lerobot_pkg_import_gpu_smoke_%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh" ]]; then
  _SLURM_COMMON="${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
else
  _SLURM_COMMON="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/common_env.sh"
fi
# shellcheck source=common_env.sh
source "${_SLURM_COMMON}"
slurm_resolve_project_root "scripts/smolvla/run_pushv3_smolvla_parity_benchmark.sh"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "smolvla-lerobot-pkg-import-gpu-smoke"

# Production MetaWorld / GRPO venv (site-packages lerobot was patched here).
PY="${LEROBOT_MW_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
if [[ ! -x "${PY}" ]]; then
  echo "error: python not executable: ${PY} (set LEROBOT_MW_PYTHON)" >&2
  exit 2
fi

echo "using PYTHON=${PY}"
exec "${PY}" -u - <<'PY'
import inspect
import sys

import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching

print("python", sys.executable)
print("lerobot_smolvla_import_ok")
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))

sig = inspect.signature(VLAFlowMatching.sample_actions)
print("VLAFlowMatching.sample_actions", sig)
print("has_select_action_distr_params", hasattr(SmolVLAPolicy, "select_action_distr_params"))
print("has__get_distr_params_chunk", hasattr(SmolVLAPolicy, "_get_distr_params_chunk"))
m = VLAFlowMatching.__init__
print("VLAFlowMatching.__init__", inspect.signature(m))
print("smolvla_gpu_symbol_check_ok")
PY
