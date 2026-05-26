#!/usr/bin/env python3
"""Audit RL4VLA raw NPZ demos before LeRobot conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


NAME_RE = re.compile(r"success_proc_(?P<proc>\d+)_numid_(?P<numid>\d+)_epsid_(?P<epsid>\d+)\.npz$")


def load_episode(path: Path) -> dict[str, Any]:
    return np.load(path, allow_pickle=True)["arr_0"].tolist()


def bool_success(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0])
    return bool(value)


def filter_small_actions_mask(
    actions: np.ndarray,
    pos_thresh: float = 0.01,
    rot_thresh: float = 0.06,
) -> np.ndarray:
    valid = np.zeros(actions.shape[0], dtype=bool)
    for i, action in enumerate(actions):
        moved = np.linalg.norm(action[:3]) > pos_thresh or np.linalg.norm(action[3:6]) > rot_thresh
        gripper_changed = i > 0 and actions[i - 1, 6] != action[6]
        valid[i] = moved or gripper_changed
    return valid


def decoded_signature(episode: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for key in ("action", "state"):
        arr = np.asarray(episode[key], dtype=np.float32)
        h.update(key.encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    images = episode["image"]
    h.update(str(len(images)).encode("utf-8"))
    for image in (images[0], images[-1]):
        arr = np.asarray(image, dtype=np.uint8)
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    h.update(str(episode.get("instruction", "")).encode("utf-8"))
    return h.hexdigest()


def audit_episode(path: Path) -> dict[str, Any]:
    episode = load_episode(path)
    required = {"image", "instruction", "state", "action", "info"}
    missing = sorted(required - set(episode))
    if missing:
        return {"path": str(path), "name": path.name, "missing_keys": missing}

    actions = np.asarray(episode["action"], dtype=np.float32)
    states = np.asarray(episode["state"], dtype=np.float32)
    images = episode["image"]
    first_image = np.asarray(images[0]) if images else np.asarray([])
    info = episode["info"]
    successes = [bool_success(row.get("success", False)) for row in info]
    mask = filter_small_actions_mask(actions) if actions.ndim == 2 and actions.shape[1] >= 7 else np.ones(len(actions), dtype=bool)
    keep_idx = np.flatnonzero(mask)
    gaps = np.diff(keep_idx) if len(keep_idx) > 1 else np.asarray([], dtype=np.int64)
    name_match = NAME_RE.match(path.name)
    return {
        "path": str(path),
        "name": path.name,
        "proc": int(name_match.group("proc")) if name_match else None,
        "numid": int(name_match.group("numid")) if name_match else None,
        "epsid": int(name_match.group("epsid")) if name_match else None,
        "frames": int(len(actions)),
        "action_shape": list(actions.shape),
        "state_shape": list(states.shape),
        "image_shape": list(first_image.shape),
        "image_dtype": str(first_image.dtype),
        "state_gripper_equal_action_gripper": bool(
            actions.ndim == 2
            and states.ndim == 2
            and actions.shape[1] >= 7
            and states.shape[1] >= 7
            and len(actions) == len(states)
            and np.allclose(states[:, 6], actions[:, 6])
        ),
        "filter_small_actions_skip_fraction": float((~mask).mean()) if len(mask) else 0.0,
        "filter_small_actions_gap_count": int((gaps > 1).sum()),
        "success_true_count": int(sum(successes)),
        "first_success_index": next((i for i, ok in enumerate(successes) if ok), None),
        "decoded_signature": decoded_signature(episode),
    }


def is_bad_shape(row: dict[str, Any]) -> bool:
    if row.get("missing_keys"):
        return True
    return (
        row.get("action_shape", [])[1:] != [7]
        or row.get("state_shape", [])[1:] != [7]
        or row.get("image_shape", [])[-1:] != [3]
    )


def audit_root(root: Path, min_episodes: int = 1, sample_limit: int = 0) -> dict[str, Any]:
    files = sorted(root.glob("*.npz"))
    if len(files) < min_episodes:
        raise SystemExit(f"found {len(files)} npz files under {root}, need >= {min_episodes}")
    if sample_limit > 0:
        files = files[:sample_limit]

    rows = [audit_episode(path) for path in files]
    seen: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    for row in rows:
        sig = row.get("decoded_signature")
        if sig is None:
            continue
        if sig in seen:
            duplicates.append({"first": seen[sig], "second": row["name"]})
        else:
            seen[sig] = row["name"]

    skip_fracs = [float(row.get("filter_small_actions_skip_fraction", 0.0)) for row in rows]
    success_counts = [int(row.get("success_true_count", 0)) for row in rows]
    leakage_count = sum(bool(row.get("state_gripper_equal_action_gripper", False)) for row in rows)
    bad_shapes = [row for row in rows if is_bad_shape(row)]
    frames = [int(row.get("frames", 0)) for row in rows]
    return {
        "root": str(root),
        "episodes": len(rows),
        "frames": int(sum(frames)),
        "duplicate_decoded_signature_count": len(duplicates),
        "duplicate_decoded_signature_examples": duplicates[:10],
        "state_gripper_equal_action_gripper_count": int(leakage_count),
        "filter_small_actions_skip_fraction_min": min(skip_fracs),
        "filter_small_actions_skip_fraction_max": max(skip_fracs),
        "episodes_with_filter_gaps": int(sum(int(row.get("filter_small_actions_gap_count", 0)) > 0 for row in rows)),
        "success_true_count_min": min(success_counts),
        "success_true_count_max": max(success_counts),
        "bad_shape_count": len(bad_shapes),
        "bad_shape_examples": bad_shapes[:10],
        "sample_rows": rows[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--fail-on-duplicates", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    summary = audit_root(args.root, min_episodes=args.min_episodes, sample_limit=args.sample_limit)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(text + "\n", encoding="utf-8")
    if args.fail_on_duplicates and summary["duplicate_decoded_signature_count"] > 0:
        raise SystemExit("duplicate decoded NPZ signatures detected")
    if summary["bad_shape_count"] > 0:
        raise SystemExit("bad raw NPZ shapes detected")


if __name__ == "__main__":
    main()
