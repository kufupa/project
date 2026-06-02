#!/usr/bin/env python3
"""Export slim RLinf eval checkpoints from full GRPO checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from smolvla_grpo.checkpointing import torch_load_mmap_default
from smolvla_grpo.phase11_rollout import load_bundle_for_grpo
from smolvla_grpo.policy_wrapper import freeze_all_but_grpo_trainables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export slim RLinf eval checkpoints from GRPO full checkpoints.")
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--task", default="push-v3")
    parser.add_argument("--rollout-chunk-len", type=int, default=5)
    parser.add_argument("--only-updates", default="")
    return parser.parse_args()


def checkpoint_update(path: Path) -> int:
    stem = path.stem
    return int(stem.split("_")[-1])


def wanted_updates(raw: str) -> set[int] | None:
    if not raw.strip():
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = (args.output_dir or (checkpoint_dir.parent / "checkpoints_eval")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle, _action_dim = load_bundle_for_grpo(
        args.model_path,
        task=args.task,
        env_backend="official_lerobot",
        n_action_steps=int(args.rollout_chunk_len),
    )
    policy = bundle.policy
    freeze_all_but_grpo_trainables(policy)
    trainable_names = {
        name
        for name, param in policy.named_parameters()
        if param.requires_grad
    }
    if not trainable_names:
        raise RuntimeError("No trainable parameters after freeze_all_but_grpo_trainables")

    allowed = wanted_updates(args.only_updates)
    ckpts = sorted(checkpoint_dir.glob("update_*.pt"), key=checkpoint_update)
    for ckpt_path in ckpts:
        update = checkpoint_update(ckpt_path)
        if allowed is not None and update not in allowed:
            continue
        checkpoint = torch_load_mmap_default(ckpt_path, map_location="cpu", weights_only=False)
        policy_state = checkpoint.get("policy_state_dict")
        if not isinstance(policy_state, dict):
            raise KeyError(f"{ckpt_path} missing policy_state_dict")
        missing = sorted(name for name in trainable_names if name not in policy_state)
        if missing:
            raise KeyError(f"{ckpt_path} missing trainable policy keys: {missing[:5]}")
        trainable_model = {
            f"policy.{name}": policy_state[name].detach().cpu().clone()
            for name in sorted(trainable_names)
        }
        payload = {
            "checkpoint_type": "trainable_delta",
            "update": int(update),
            "source_update_index": int(checkpoint.get("update_index", update - 1)),
            "source_checkpoint": str(ckpt_path),
            "trainable_model": trainable_model,
            "metrics": checkpoint.get("extra", {}),
        }
        out_path = output_dir / ckpt_path.name
        torch.save(payload, out_path)
        print(
            f"EXPORT_GRPO_EVAL_CHECKPOINT_OK update={update} out={out_path} tensors={len(trainable_model)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
