#!/usr/bin/env python3
"""Convert RL4VLA raw npz demos into local LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def load_episode(path: Path) -> dict[str, Any]:
    return np.load(path, allow_pickle=True)["arr_0"].tolist()


def filter_small_actions(
    actions: np.ndarray,
    pos_thresh: float = 0.01,
    rot_thresh: float = 0.06,
    check_gripper: bool = True,
) -> np.ndarray:
    valid = np.zeros(actions.shape[0], dtype=bool)
    for i, action in enumerate(actions):
        moved = np.linalg.norm(action[:3]) > pos_thresh or np.linalg.norm(action[3:6]) > rot_thresh
        gripper_changed = check_gripper and i > 0 and actions[i - 1, 6] != action[6]
        valid[i] = moved or gripper_changed
    return valid


def normalize_instruction(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        return str(value[0])
    return str(value)


def as_image_array(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got {image.shape}")
    return image


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    files = sorted(args.input.glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    if len(files) < args.min_episodes:
        raise SystemExit(f"found {len(files)} npz files under {args.input}, need >= {args.min_episodes}")

    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)

    features = {
        "observation.images.front": {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["eef_x", "eef_y", "eef_z", "eef_roll", "eef_pitch", "eef_yaw", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.output,
        robot_type="widowx",
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=args.image_writer_threads,
    )

    kept_frames = 0
    skipped_frames = 0
    converted_episodes = 0
    for path in files:
        episode = load_episode(path)
        actions = np.asarray(episode["action"], dtype=np.float32)
        states = np.asarray(episode["state"], dtype=np.float32)
        images = episode["image"]
        if actions.shape[1:] != (7,) or states.shape[1:] != (7,) or len(actions) != len(states):
            raise SystemExit(f"{path} bad action/state shapes: {actions.shape} {states.shape}")
        mask = filter_small_actions(actions) if args.filter_small_actions else np.ones(len(actions), dtype=bool)
        instruction = normalize_instruction(episode["instruction"])
        success_count = 0
        frame_count = 0
        for i, keep in enumerate(mask):
            if not keep:
                skipped_frames += 1
                continue
            dataset.add_frame(
                {
                    "observation.images.front": as_image_array(images[i]),
                    "observation.state": states[i].astype(np.float32),
                    "action": actions[i].astype(np.float32),
                    "task": instruction,
                }
            )
            kept_frames += 1
            frame_count += 1
            if episode["info"][i]["success"]:
                success_count += 1
            else:
                success_count = 0
            if args.stop_after_success_count > 0 and success_count >= args.stop_after_success_count:
                break
        if frame_count:
            dataset.save_episode()
            converted_episodes += 1
    dataset.finalize()

    return {
        "input": str(args.input),
        "output": str(args.output),
        "repo_id": args.repo_id,
        "episodes_seen": len(files),
        "episodes_converted": converted_episodes,
        "frames": kept_frames,
        "skipped_frames": skipped_frames,
        "filter_small_actions": args.filter_small_actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--stop-after-success-count", type=int, default=15)
    parser.add_argument("--filter-small-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    summary = build_dataset(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
