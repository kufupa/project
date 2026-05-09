#!/usr/bin/env bash
# Smoke required Python runtimes and build fallback SmolVLA env when needed.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/general/user/aa6622/home/project}"
# shellcheck source=common.sh
source "${PROJECT_ROOT}/scripts/maniskill_smolvla/common.sh"

msm_setup_modules
msm_prepare_runtime

echo "[msm-env] data python primary: ${MSM_DATA_PYTHON}"
msm_require_python "${MSM_DATA_PYTHON}"

"${MSM_DATA_PYTHON}" - <<'PY'
mods = [
    "gymnasium",
    "mani_skill",
    "tyro",
    "cv2",
    "torch",
    "sapien",
]
for mod in mods:
    __import__(mod)
    print(f"[msm-env] data import OK: {mod}")
PY

if [[ ! -x "${MSM_TRAIN_PYTHON}" ]]; then
  echo "[msm-env] train python missing; building ${MSM_TRAIN_VENV}"
  mkdir -p "${MSM_VENV_ROOT}"
  python3 -m venv --copies "${MSM_TRAIN_VENV}"
  MSM_TRAIN_PYTHON="${MSM_TRAIN_VENV}/bin/python"
  "${MSM_TRAIN_PYTHON}" -m pip install --upgrade pip wheel setuptools
  "${MSM_TRAIN_PYTHON}" -m pip install -r "${PROJECT_ROOT}/requirements-smolvla-lock.txt"
fi

echo "[msm-env] train python primary: ${MSM_TRAIN_PYTHON}"
msm_require_python "${MSM_TRAIN_PYTHON}"

"${MSM_TRAIN_PYTHON}" - <<'PY'
mods = [
    "torch",
    "transformers",
    "huggingface_hub",
    "lerobot.datasets.lerobot_dataset",
    "lerobot.policies.smolvla.configuration_smolvla",
    "lerobot.scripts.lerobot_train",
]
for mod in mods:
    __import__(mod)
    print(f"[msm-env] train import OK: {mod}")
PY

"${MSM_TRAIN_PYTHON}" - <<'PY'
import os
from huggingface_hub import snapshot_download

cache_dir = os.environ["HF_HOME"]
for repo_id in ["lerobot/smolvla_base", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"]:
    path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
    print(f"[msm-env] warmed {repo_id}: {path}")
PY

msm_write_manifest \
  "${MSM_RUN_ROOT}/manifests/envs.env" \
  "stage=build_envs" \
  "data_python=${MSM_DATA_PYTHON}" \
  "train_python=${MSM_TRAIN_PYTHON}" \
  "hf_home=${HF_HOME}" \
  "hf_lerobot_home=${HF_LEROBOT_HOME}" \
  "ms_asset_dir=${MS_ASSET_DIR}" \
  "maniskill_asset_dir=${MANISKILL_ASSET_DIR}"

echo "MSM_BUILD_ENVS_DONE run_root=${MSM_RUN_ROOT}"
