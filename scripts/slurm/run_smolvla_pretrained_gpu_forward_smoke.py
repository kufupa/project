#!/usr/bin/env python3
"""Load hub SmolVLA (MetaWorld) checkpoint + preprocessor, run one GPU forward and distr-param path."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import torch


def _infer_image_chw(policy: object) -> tuple[int, int, int]:
    feats = getattr(getattr(policy, "config", None), "input_features", None) or {}
    for name, ft in feats.items():
        if "image" not in str(name).lower():
            continue
        shape = getattr(ft, "shape", None) or ()
        if len(shape) >= 3:
            c, h, w = int(shape[0]), int(shape[1]), int(shape[2])
            if c > 0 and h > 0 and w > 0:
                return c, h, w
    return 3, 480, 640


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="jadechoghari/smolvla_metaworld",
        help="HF repo id (default: MetaWorld SmolVLA used for parity / Phase11).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("error: CUDA not available (need a GPU allocation)", flush=True)
        return 1

    from smolvla_pipeline.evaluator import (  # noqa: PLC0415
        _load_smolvla_bundle,
        _resolve_task_text,
        _smolvla_state_dims,
    )

    ckpt = args.checkpoint.strip()
    t0 = time.perf_counter()
    print(f"smolvla_pretrained_smoke: load_begin checkpoint={ckpt!r}", flush=True)
    bundle = _load_smolvla_bundle(ckpt)
    print(f"smolvla_pretrained_load_ok elapsed_s={time.perf_counter() - t0:.2f}", flush=True)

    policy = bundle.policy
    c, h, w = _infer_image_chw(policy)
    agent_dim, env_dim = _smolvla_state_dims(policy)
    device = bundle.device
    task_text = _resolve_task_text("push-v3", override=None)

    raw = {
        bundle.obs_image_key: torch.rand(1, c, h, w, device=device, dtype=torch.float32).clamp(0.0, 1.0),
        bundle.obs_state_key: torch.zeros(1, agent_dim, device=device, dtype=torch.float32),
        bundle.obs_env_state_key: torch.zeros(1, env_dim, device=device, dtype=torch.float32),
        "task": task_text,
    }
    proc = bundle.preprocessor(raw)

    policy.reset()
    t1 = time.perf_counter()
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(proc)
    print(
        f"smolvla_pretrained_forward_ok chunk_shape={tuple(chunk.shape)} "
        f"elapsed_s={time.perf_counter() - t1:.2f}",
        flush=True,
    )

    policy.reset()
    t2 = time.perf_counter()
    with torch.inference_mode():
        mean, log_std = policy.select_action_distr_params(proc)
    print(
        "smolvla_pretrained_distr_params_ok "
        f"mean_shape={tuple(mean.shape)} log_std_shape={tuple(log_std.shape)} "
        f"mean_dtype={mean.dtype} log_std_dtype={log_std.dtype} "
        f"elapsed_s={time.perf_counter() - t2:.2f}",
        flush=True,
    )
    print("smolvla_pretrained_gpu_forward_smoke_ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
