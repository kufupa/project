#!/usr/bin/env python3
"""Convert RL4VLA raw npz demos into local LeRobot dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def load_episode(path: Path) -> dict[str, Any]:
    return np.load(path, allow_pickle=True)["arr_0"].tolist()


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


def apply_state_gripper_mode(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    mode: str,
    initial_gripper: float,
) -> np.ndarray:
    if mode == "as-recorded":
        return states.astype(np.float32, copy=True)
    if mode != "previous-action":
        raise ValueError(f"unsupported state gripper mode: {mode}")
    fixed = states.astype(np.float32, copy=True)
    fixed[0, 6] = np.float32(initial_gripper)
    if len(fixed) > 1:
        fixed[1:, 6] = actions[:-1, 6].astype(np.float32)
    return fixed


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

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
    appended_completion_frames = 0
    converted_episodes = 0
    skipped_duplicate_episodes = 0
    seen_signatures: set[str] = set()
    for path in files:
        episode = load_episode(path)
        if args.dedupe_decoded_signatures:
            signature = decoded_signature(episode)
            if signature in seen_signatures:
                skipped_duplicate_episodes += 1
                continue
            seen_signatures.add(signature)
        actions = np.asarray(episode["action"], dtype=np.float32)
        raw_states = np.asarray(episode["state"], dtype=np.float32)
        states = apply_state_gripper_mode(
            raw_states,
            actions,
            mode=args.state_gripper_mode,
            initial_gripper=args.initial_gripper,
        )
        images = episode["image"]
        if actions.shape[1:] != (7,) or states.shape[1:] != (7,) or len(actions) != len(states):
            raise SystemExit(f"{path} bad action/state shapes: {actions.shape} {states.shape}")
        mask = filter_small_actions(actions) if args.filter_small_actions else np.ones(len(actions), dtype=bool)
        instruction = normalize_instruction(episode["instruction"])
        success_count = 0
        frame_count = 0
        last_frame: dict[str, Any] | None = None
        for i, keep in enumerate(mask):
            if not keep:
                skipped_frames += 1
                continue
            frame = {
                "observation.images.front": as_image_array(images[i]),
                "observation.state": states[i].astype(np.float32),
                "action": actions[i].astype(np.float32),
                "task": instruction,
            }
            dataset.add_frame(frame)
            last_frame = frame
            kept_frames += 1
            frame_count += 1
            if episode["info"][i]["success"]:
                success_count += 1
            else:
                success_count = 0
            if args.stop_after_success_count > 0 and success_count >= args.stop_after_success_count:
                break
        if frame_count:
            if args.append_completion_frames > 0 and last_frame is not None:
                completion_action = np.zeros_like(last_frame["action"], dtype=np.float32)
                completion_action[-1] = last_frame["action"][-1]
                for _ in range(args.append_completion_frames):
                    dataset.add_frame(
                        {
                            "observation.images.front": last_frame["observation.images.front"].copy(),
                            "observation.state": last_frame["observation.state"].copy(),
                            "action": completion_action.copy(),
                            "task": instruction,
                        }
                    )
                    kept_frames += 1
                    appended_completion_frames += 1
            dataset.save_episode()
            converted_episodes += 1
    dataset.finalize()

    return {
        "input": str(args.input),
        "output": str(args.output),
        "repo_id": args.repo_id,
        "episodes_seen": len(files),
        "episodes_converted": converted_episodes,
        "episodes_skipped_duplicate_decoded_signature": skipped_duplicate_episodes,
        "frames": kept_frames,
        "skipped_frames": skipped_frames,
        "appended_completion_frames": appended_completion_frames,
        "filter_small_actions": args.filter_small_actions,
        "state_gripper_mode": args.state_gripper_mode,
        "initial_gripper": args.initial_gripper,
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
    parser.add_argument("--stop-after-success-count", type=int, default=0)
    parser.add_argument("--append-completion-frames", type=int, default=0)
    parser.add_argument("--state-gripper-mode", choices=("as-recorded", "previous-action"), default="previous-action")
    parser.add_argument("--initial-gripper", type=float, default=1.0)
    parser.add_argument("--dedupe-decoded-signatures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter-small-actions", action=argparse.BooleanOptionalAction, default=False)
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
