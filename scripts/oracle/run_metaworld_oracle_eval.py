#!/usr/bin/env python3
"""Run Meta-World scripted oracle policy rollouts and emit eval_info.json."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import imageio.v2 as imageio
import metaworld
import numpy as np
from metaworld.policies import ENV_POLICY_MAP

SCHEMA_VERSION = "oracle_run_v1"


def _as_bool(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw!r}")


def _safe_success(info: dict[str, Any]) -> bool:
    for key in ("success", "is_success"):
        value = info.get(key)
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
    return False


def _render_rgb_frame(
    env: Any, *, camera_name: str, flip_corner2: bool
) -> np.ndarray | None:
    frame: Any | None = None
    if camera_name:
        try:
            frame = env.render(camera_name=camera_name)
        except TypeError:
            frame = None
        except Exception:
            frame = None
    if frame is None:
        try:
            frame = env.render()
        except Exception:
            return None

    if frame is None:
        return None
    frame_np = np.asarray(frame)
    if frame_np.ndim != 3:
        return None
    if frame_np.shape[-1] == 4:
        frame_np = frame_np[..., :3]
    if frame_np.dtype != np.uint8:
        if np.issubdtype(frame_np.dtype, np.floating) and float(np.max(frame_np)) <= 1.5:
            frame_np = (np.clip(frame_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
    if flip_corner2 and camera_name == "corner2":
        frame_np = np.ascontiguousarray(np.flip(frame_np, (0, 1)))
    return np.ascontiguousarray(frame_np)


def _write_frame_png(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Meta-World oracle rollout evaluator for push-v3 style tasks."
    )
    parser.add_argument("--task", default="push-v3")
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--video", default="true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--camera-name",
        default=os.environ.get("ORACLE_METAWORLD_CAMERA_NAME", "corner2"),
        help="Meta-World camera name for rgb rendering (default: corner2).",
    )
    parser.add_argument(
        "--flip-corner2",
        default=os.environ.get("ORACLE_FLIP_CORNER2", "true"),
        help="When true and camera-name is corner2, flip frame orientation for parity.",
    )
    parser.add_argument(
        "--save-frames",
        default="true",
        help="Write per-timestep PNGs under frames/episode_XXXX/ (default: true).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("error: --episodes must be >= 1")
    if args.max_steps < 1:
        raise SystemExit("error: --max-steps must be >= 1")
    if args.fps < 1:
        raise SystemExit("error: --fps must be >= 1")

    write_video = _as_bool(args.video)
    save_frames = _as_bool(args.save_frames)
    flip_corner2 = _as_bool(args.flip_corner2)
    camera_name = str(args.camera_name).strip()
    if not camera_name:
        raise SystemExit("error: --camera-name must be non-empty")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos" / f"{args.task}_0"
    if write_video:
        videos_dir.mkdir(parents=True, exist_ok=True)

    if args.task not in ENV_POLICY_MAP:
        available = ", ".join(sorted(ENV_POLICY_MAP))
        raise SystemExit(
            f"error: no Meta-World scripted policy for task '{args.task}'. Available: {available}"
        )

    start_t = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    policy = ENV_POLICY_MAP[args.task]()

    ml1 = metaworld.ML1(args.task, seed=int(args.seed))
    env_cls = ml1.train_classes[args.task]
    tasks = list(getattr(ml1, "train_tasks", []) or [])
    try:
        env = env_cls(render_mode="rgb_array", camera_name=camera_name)
    except Exception:
        env = env_cls()
    if hasattr(env, "render_mode"):
        try:
            env.render_mode = "rgb_array"
        except Exception:
            pass
    if camera_name == "corner2":
        try:
            env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        except Exception:
            pass

    sum_rewards: list[float] = []
    max_rewards: list[float] = []
    successes: list[bool] = []
    video_paths: list[str] = []
    episode_manifest_rows: list[dict[str, Any]] = []

    try:
        for episode_index in range(args.episodes):
            ep_name = f"episode_{episode_index:04d}"
            episode_dir = output_dir / "episodes" / ep_name
            frames_dir = output_dir / "frames" / ep_name
            episode_dir.mkdir(parents=True, exist_ok=True)
            if save_frames:
                frames_dir.mkdir(parents=True, exist_ok=True)

            if tasks:
                env.set_task(tasks[episode_index % len(tasks)])

            reset_seed = int(args.seed) + int(episode_index)
            obs, info = env.reset(seed=reset_seed)
            _ = info
            rewards: list[float] = []
            episode_success = False
            frames: list[np.ndarray] = []
            frame_seq = 0

            if write_video or save_frames:
                frame = _render_rgb_frame(
                    env, camera_name=camera_name, flip_corner2=flip_corner2
                )
                if frame is not None:
                    frames.append(frame)
                    if save_frames:
                        _write_frame_png(frames_dir / f"frame_{frame_seq:06d}.png", frame)
                    frame_seq += 1

            actions_path = episode_dir / "actions.jsonl"
            with actions_path.open("w", encoding="utf-8") as action_fp:
                for step_idx in range(args.max_steps):
                    obs_np = np.asarray(obs, dtype=np.float64)
                    action = policy.get_action(obs_np)
                    action_list = np.asarray(action, dtype=np.float32).reshape(-1).tolist()
                    obs, reward, terminated, truncated, info = env.step(action)
                    info_d = info if isinstance(info, dict) else {}
                    step_success = _safe_success(info_d)
                    episode_success = episode_success or step_success
                    rewards.append(float(reward))

                    line = {
                        "step": step_idx,
                        "action": action_list,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "success": bool(step_success),
                    }
                    action_fp.write(json.dumps(line) + "\n")

                    if write_video or save_frames:
                        frame = _render_rgb_frame(
                            env, camera_name=camera_name, flip_corner2=flip_corner2
                        )
                        if frame is not None:
                            frames.append(frame)
                            if save_frames:
                                _write_frame_png(
                                    frames_dir / f"frame_{frame_seq:06d}.png", frame
                                )
                            frame_seq += 1

                    if bool(terminated) or bool(truncated):
                        break

            if rewards:
                sum_reward = float(np.sum(rewards, dtype=np.float64))
                max_reward = float(np.max(rewards))
            else:
                sum_reward = 0.0
                max_reward = 0.0

            sum_rewards.append(sum_reward)
            max_rewards.append(max_reward)
            successes.append(bool(episode_success))

            rel_video: str | None = None
            if write_video and frames:
                episode_video = videos_dir / f"eval_episode_{episode_index}.mp4"
                imageio.mimsave(episode_video, frames, fps=args.fps)
                video_paths.append(str(episode_video))
                rel_video = str(episode_video.relative_to(output_dir))
            else:
                video_paths.append("")

            episode_meta = {
                "episode_index": episode_index,
                "reset_seed": reset_seed,
                "n_steps": len(rewards),
                "n_frames": frame_seq,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": bool(episode_success),
                "paths": {
                    "actions": str((episode_dir / "actions.jsonl").relative_to(output_dir)),
                    "episode_meta": str((episode_dir / "episode_meta.json").relative_to(output_dir)),
                    "frames_dir": str(frames_dir.relative_to(output_dir)) if save_frames else None,
                    "video": rel_video,
                },
            }
            (episode_dir / "episode_meta.json").write_text(
                json.dumps(episode_meta, indent=2), encoding="utf-8"
            )

            episode_manifest_rows.append(
                {
                    "episode_index": episode_index,
                    "reset_seed": reset_seed,
                    "n_steps": len(rewards),
                    "n_frames": frame_seq,
                    "sum_reward": sum_reward,
                    "max_reward": max_reward,
                    "success": bool(episode_success),
                    "paths": episode_meta["paths"],
                }
            )
    finally:
        env.close()

    elapsed_s = float(time.time() - start_t)
    success_percent = 100.0 * float(sum(1 for x in successes if x)) / float(len(successes))
    per_task_metrics = {
        "sum_rewards": sum_rewards,
        "max_rewards": max_rewards,
        "successes": successes,
        "video_paths": video_paths,
    }
    eval_info = {
        "per_task": [{"task_group": args.task, "task_id": 0, "metrics": per_task_metrics}],
        "per_group": {
            args.task: {
                "avg_sum_reward": float(mean(sum_rewards)),
                "avg_max_reward": float(mean(max_rewards)),
                "pc_success": float(success_percent),
                "n_episodes": int(len(sum_rewards)),
                "video_paths": [p for p in video_paths if p],
            }
        },
        "overall": {
            "avg_sum_reward": float(mean(sum_rewards)),
            "avg_max_reward": float(mean(max_rewards)),
            "pc_success": float(success_percent),
            "n_episodes": int(len(sum_rewards)),
            "eval_s": elapsed_s,
            "eval_ep_s": elapsed_s / float(max(1, len(sum_rewards))),
            "video_paths": [p for p in video_paths if p],
        },
    }
    eval_info_path = output_dir / "eval_info.json"
    eval_info_path.write_text(json.dumps(eval_info, indent=2), encoding="utf-8")

    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "task": args.task,
        "seed": int(args.seed),
        "episodes_requested": int(args.episodes),
        "max_steps": int(args.max_steps),
        "fps": int(args.fps),
        "camera_name": camera_name,
        "flip_corner2": bool(flip_corner2),
        "video_enabled": write_video,
        "save_frames": save_frames,
        "output_dir": str(output_dir),
        "eval_info": "eval_info.json",
        "episodes": episode_manifest_rows,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )

    print(f"Oracle eval output directory: {output_dir}")
    print(f"eval_info.json: {eval_info_path}")
    print(f"run_manifest.json: {output_dir / 'run_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
