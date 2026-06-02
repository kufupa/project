"""GRPO checkpoint save/load (policy + optimizer + training position)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch

LoadMode = Literal["mmap", "eager"]


def save_grpo_checkpoint(
    path: Path,
    *,
    policy_state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    update_index: int,
    args: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_state_dict": policy_state,
        "optimizer_state_dict": optimizer_state,
        "update_index": int(update_index),
        "args": args,
        "extra": extra or {},
    }
    torch.save(payload, path)
    meta = path.with_suffix(path.suffix + ".meta.json")
    meta.write_text(
        json.dumps(
            {"update_index": int(update_index), "path": str(path)},
            indent=2,
        ),
        encoding="utf-8",
    )


def build_rlinf_eval_trainable_model(
    policy: torch.nn.Module,
    *,
    key_prefix: str = "policy.",
) -> dict[str, torch.Tensor]:
    trainable: dict[str, torch.Tensor] = {}
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        trainable[f"{key_prefix}{name}"] = param.detach().cpu().clone()
    if not trainable:
        raise ValueError("No trainable parameters found for RLinf eval checkpoint")
    return trainable


def save_rlinf_eval_checkpoint(
    path: Path,
    *,
    policy: torch.nn.Module,
    update_index: int,
    metrics: dict[str, Any] | None = None,
    source_checkpoint: str | Path | None = None,
    key_prefix: str = "policy.",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_type": "trainable_delta",
        "update": int(update_index) + 1,
        "source_update_index": int(update_index),
        "trainable_model": build_rlinf_eval_trainable_model(policy, key_prefix=key_prefix),
        "metrics": metrics or {},
    }
    if source_checkpoint is not None:
        payload["source_checkpoint"] = str(source_checkpoint)
    torch.save(payload, path)


def torch_load_mmap_default(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    weights_only: bool = False,
) -> Any:
    """Load a path-like checkpoint with mmap=True when PyTorch supports it."""
    payload, _ = torch_load_mmap_with_mode(
        path,
        map_location=map_location,
        weights_only=weights_only,
    )
    return payload


def torch_load_mmap_with_mode(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    weights_only: bool = False,
) -> tuple[Any, LoadMode]:
    """Load a path-like checkpoint and report whether mmap or eager load was used."""
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only, mmap=True), "mmap"
    except TypeError:
        try:
            return torch.load(path, map_location=map_location, weights_only=weights_only), "eager"
        except TypeError:
            return torch.load(path, map_location=map_location), "eager"


def load_grpo_checkpoint(path: Path, *, map_location: str | torch.device | None = None) -> dict[str, Any]:
    return torch_load_mmap_default(path, map_location=map_location, weights_only=False)
