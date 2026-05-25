#!/usr/bin/env bash
# Build the Python 3.12 runtime venv used by PIRL ManiSkill PBS jobs.

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/rds/general/user/aa6622/home/project}"
COMMON="${PROJECT_ROOT}/scripts/pirl/pirl_maniskill_common.sh"
# shellcheck source=pirl_maniskill_common.sh
source "${COMMON}"

pirl_setup_modules

export PIRL_RUNTIME_VENV="${PIRL_RUNTIME_VENV:-${PROJECT_ROOT}/.venvs/pirl-rlinf-py312}"
python3 -m venv "${PIRL_RUNTIME_VENV}"
PY="${PIRL_RUNTIME_VENV}/bin/python"

"${PY}" -m pip install --upgrade pip wheel setuptools
"${PY}" -m pip install --ignore-requires-python -e "${PROJECT_ROOT}/RLinf[embodied]"

# OpenPI pins conflict on Python 3.12 (`numpy<2` vs JAX/augmax); install code
# and known import deps explicitly. This preserves the user's Python 3.12 constraint.
"${PY}" -m pip install --no-deps --force-reinstall git+https://github.com/RLinf/openpi
"${PY}" -m pip install --no-deps --force-reinstall \
  git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
"${PY}" -m pip install git+https://github.com/haosulab/ManiSkill.git@v3.0.0b22
"${PY}" -m pip install --no-deps \
  tqdm-loggable beartype==0.19.0 jaxtyping==0.2.36 ml-collections==1.0.0 \
  sentencepiece treescope flatbuffers dm-tree numpydantic openpi-client polars \
  jax==0.5.3 jaxlib==0.5.3 flax==0.10.2 orbax-checkpoint==0.11.13 \
  chex==0.1.90 equinox optax augmax==0.3.4 gym-aloha \
  av datasets deepdiff diffusers draccus==0.10.0 einops flask gdown h5py \
  imageio jsonlines numba omegaconf opencv-python-headless pymunk pynput \
  pyzmq rerun-sdk termcolor torchvision wandb zarr \
  mergedeep pyyaml-include typing-inspect toml mypy-extensions \
  ml_dtypes etils humanize msgpack simplejson nest_asyncio tensorstore \
  opt_einsum toolz wadler_lindig \
  transformers==4.53.2 'tokenizers>=0.21,<0.22'

"${PY}" - <<'PY'
from pathlib import Path
import shutil
import transformers

site_transformers = Path(transformers.__file__).resolve().parent
replace_root = site_transformers.parent / "openpi" / "models_pytorch" / "transformers_replace"
for child in replace_root.iterdir():
    target = site_transformers / child.name
    if child.is_dir():
        shutil.copytree(child, target, dirs_exist_ok=True)
    else:
        shutil.copy2(child, target)
PY

"${PY}" - <<'PY'
mods = [
    "hydra", "ray", "torch", "gymnasium", "mani_skill", "openpi",
    "transformers", "safetensors", "huggingface_hub", "jax", "flax",
    "orbax.checkpoint", "chex", "equinox", "optax", "tqdm_loggable",
    "lerobot.common.datasets.lerobot_dataset",
    "transformers.models.siglip.check",
]
for mod in mods:
    __import__(mod)
    print(f"{mod} OK")
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
PY

export PIRL_PYTHON="${PY}"
pirl_prepare_runtime
"${PY}" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="RLinf/maniskill_assets",
    repo_type="dataset",
    local_dir=os.environ["MANISKILL_ASSET_DIR"],
    local_dir_use_symlinks=False,
)
PY
"${PY}" -m mani_skill.utils.download_asset bridge_v2_real2sim -y
"${PY}" -m mani_skill.utils.download_asset widowx250s -y

echo "PIRL_RUNTIME_SETUP_DONE python=${PY}"
