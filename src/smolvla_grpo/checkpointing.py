"""GRPO checkpoint save/load (policy + optimizer + training cursor)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


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


def load_grpo_checkpoint(path: Path, *, map_location: str | torch.device | None = None) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
