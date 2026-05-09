#!/usr/bin/env python3
"""Validate RL4VLA raw npz demos for 7D SmolVLA SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_episode(path: Path) -> dict[str, Any]:
    return np.load(path, allow_pickle=True)["arr_0"].tolist()


def summarize(root: Path, min_episodes: int = 1) -> dict[str, Any]:
    files = sorted(root.glob("*.npz"))
    if len(files) < min_episodes:
        raise SystemExit(f"found {len(files)} npz files under {root}, need >= {min_episodes}")

    first = load_episode(files[0])
    required = {"image", "instruction", "state", "action", "info"}
    missing = sorted(required - set(first))
    if missing:
        raise SystemExit(f"{files[0]} missing keys: {missing}")

    action = np.asarray(first["action"], dtype=np.float32)
    state = np.asarray(first["state"], dtype=np.float32)
    image = np.asarray(first["image"][0])
    if action.ndim != 2 or action.shape[1] != 7:
        raise SystemExit(f"{files[0]} action shape {action.shape}, expected (*, 7)")
    if state.ndim != 2 or state.shape[1] != 7:
        raise SystemExit(f"{files[0]} state shape {state.shape}, expected (*, 7)")
    if len(action) != len(state):
        raise SystemExit(f"{files[0]} action/state length mismatch {len(action)} != {len(state)}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise SystemExit(f"{files[0]} image shape {image.shape}, expected HWC RGB")

    total_frames = 0
    bad = []
    for path in files:
        episode = load_episode(path)
        a = np.asarray(episode["action"], dtype=np.float32)
        s = np.asarray(episode["state"], dtype=np.float32)
        if a.ndim != 2 or s.ndim != 2 or a.shape[1:] != (7,) or s.shape[1:] != (7,) or len(a) != len(s):
            bad.append({"path": str(path), "action_shape": list(a.shape), "state_shape": list(s.shape)})
            continue
        total_frames += len(a)

    if bad:
        raise SystemExit(json.dumps({"bad_episodes": bad[:10], "bad_count": len(bad)}, indent=2))

    return {
        "root": str(root),
        "episodes": len(files),
        "frames": total_frames,
        "first_file": str(files[0]),
        "first_action_shape": list(action.shape),
        "first_state_shape": list(state.shape),
        "first_image_shape": list(image.shape),
        "instruction": str(first["instruction"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    summary = summarize(args.root, args.min_episodes)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
