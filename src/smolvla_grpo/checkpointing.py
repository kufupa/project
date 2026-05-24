"""GRPO checkpoint save/load (policy + optimizer + training position)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def _fsync_parent(path: Path) -> None:
    try:
        fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


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
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp_path)
    with tmp_path.open("ab") as f:
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    meta = path.with_suffix(path.suffix + ".meta.json")
    tmp_meta = meta.with_name(f".{meta.name}.{os.getpid()}.tmp")
    tmp_meta.write_text(
        json.dumps(
            {"update_index": int(update_index), "path": str(path)},
            indent=2,
        ),
        encoding="utf-8",
    )
    with tmp_meta.open("ab") as f:
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_meta, meta)
    _fsync_parent(path)


def load_grpo_checkpoint(path: Path, *, map_location: str | torch.device | None = None) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
