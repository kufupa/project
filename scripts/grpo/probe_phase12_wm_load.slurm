#!/usr/bin/env bash
#SBATCH --job-name=phase12-wm-load-probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=phase12_wm_load_probe_%j.out
#SBATCH --error=phase12_wm_load_probe_%j.err
#SBATCH --export=NIL

set -euo pipefail

source /vol/bitbucket/aa6622/project/scripts/slurm/common_env.sh
slurm_resolve_project_root "scripts/grpo/train_phase12_wm_chunk_grpo.py"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "phase12-wm-load-probe"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export JEPA_WM_DISABLE_IMAGE_HEAD="${JEPA_WM_DISABLE_IMAGE_HEAD:-1}"

/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python - <<'PY'
import os
import traceback
from pathlib import Path

import torch

from segment_grpo_loop import load_wm_bundle
from vendor.pi05 import jepa_cem_paired_pushv3_export as helper

ckpt = "jepa_wm_metaworld.pth.tar"
repo = Path("/vol/bitbucket/aa6622/VGG JEPA/jepa-wms")
print(
    "probe_env",
    "HOME=" + os.environ.get("HOME", ""),
    "HF_HOME=" + os.environ.get("HF_HOME", ""),
    "HUB=" + os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
    "TORCH_HOME=" + os.environ.get("TORCH_HOME", ""),
    "cuda=" + str(torch.cuda.is_available()),
    "device_count=" + str(torch.cuda.device_count()),
    flush=True,
)
print("resolved_ckpt", helper._resolve_ckpt(ckpt), flush=True)
os.environ["JEPAWM_CKPT"] = helper._resolve_ckpt(ckpt)
try:
    model, preprocessor = torch.hub.load(
        str(repo),
        "jepa_wm_metaworld",
        source="local",
        pretrained=True,
        device="cuda",
    )
    model.eval()
    print("direct_torch_hub_ok", type(model).__name__, type(preprocessor).__name__, flush=True)
except Exception as exc:
    print("direct_torch_hub_fail", type(exc).__name__, exc, flush=True)
    traceback.print_exc()
try:
    wm = load_wm_bundle(repo, ckpt, "cuda", required=True)
    print("load_wm_bundle_ok", wm.planner_action_dim, wm.proprio_dim, flush=True)
except Exception as exc:
    print("load_wm_bundle_fail", type(exc).__name__, exc, flush=True)
    traceback.print_exc()
    raise
PY

