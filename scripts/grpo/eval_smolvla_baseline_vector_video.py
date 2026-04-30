#!/usr/bin/env python3
"""Vectorized SmolVLA baseline eval with per-episode videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout, resolve_lerobot_horizon
from smolvla_grpo.phase11_rollout import load_bundle_for_grpo
from smolvla_grpo.phase12_diagnostics import write_phase12_episode_video
from smolvla_grpo.phase12_logging import utc_now_iso
from smolvla_grpo.phase12_vector_eval import (
    build_episode_waves,
    coerce_exec_action_batch,
    concatenate_proc_rows,
    select_eval_action_queue_free,
)


BASE_CHECKPOINT = (
    "/rds/general/user/aa6622/home/.cache/huggingface/hub/"
    "models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900"
)


def _frame_from_obs(obs: dict[str, Any]) -> np.ndarray | None:
    pixels = obs.get("pixels") if isinstance(obs, dict) else None
    if pixels is None:
        return None
    arr = np.asarray(pixels)
    if arr.ndim == 4:
        frame = arr[0]
    elif arr.ndim == 3:
        frame = arr
    else:
        return None
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and float(np.max(frame)) <= 1.5:
            frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=BASE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, default="push-v3")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--eval-seed-start", type=int, default=1000)
    parser.add_argument("--n-envs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle, _action_dim = load_bundle_for_grpo(
        args.checkpoint,
        task=args.task,
        env_backend="official_lerobot",
        n_action_steps=1,
    )
    bundle.policy.eval()

    rows: list[dict[str, Any]] = []
    t0 = perf_counter()
    for wave in build_episode_waves(
        episodes=int(args.episodes),
        eval_seed_start=int(args.eval_seed_start),
        n_envs=int(args.n_envs),
    ):
        wave_n = len(wave)
        envs = [OfficialLeRobotMetaWorldGRPORollout(task=args.task, n_envs=1) for _ in range(wave_n)]
        try:
            resolved_steps = resolve_lerobot_horizon(envs[0], int(args.max_steps))
            obs_by_row = [env.reset(seed) for env, (_ep, seed) in zip(envs, wave, strict=True)]
            reset = getattr(bundle.policy, "reset", None)
            if callable(reset):
                reset()
            active = np.ones((wave_n,), dtype=np.bool_)
            actions: list[list[list[float]]] = [[] for _ in range(wave_n)]
            rewards: list[list[float]] = [[] for _ in range(wave_n)]
            successes: list[list[bool]] = [[] for _ in range(wave_n)]
            frames: list[list[np.ndarray]] = [[] for _ in range(wave_n)]
            for row, obs in enumerate(obs_by_row):
                frame = _frame_from_obs(obs)
                if frame is None:
                    frame = envs[row].render_frame()
                frames[row].append(frame)

            for _step in range(int(resolved_steps)):
                if not bool(np.any(active)):
                    break
                active_rows = [idx for idx in range(wave_n) if bool(active[idx])]
                proc = concatenate_proc_rows(
                    [envs[idx].build_proc(obs_by_row[idx], bundle=bundle) for idx in active_rows]
                )
                with torch.inference_mode():
                    action = select_eval_action_queue_free(bundle.policy, proc)
                    post = bundle.postprocessor(action)
                exec_action_np = coerce_exec_action_batch(
                    post,
                    action_dim=int(envs[0].action_dim),
                    n_envs=len(active_rows),
                )
                for batch_row, row in enumerate(active_rows):
                    step = envs[row].step(exec_action_np[batch_row : batch_row + 1])
                    obs_by_row[row] = step.observation
                    actions[row].append(exec_action_np[batch_row].reshape(-1).tolist())
                    rewards[row].append(float(step.reward))
                    successes[row].append(bool(step.success))
                    frame = _frame_from_obs(step.observation)
                    if frame is None:
                        frame = envs[row].render_frame()
                    frames[row].append(frame)
                    if step.success or step.terminated or step.truncated:
                        active[row] = False

            for row, (episode_index, reset_seed) in enumerate(wave):
                episode_dir = output_dir / f"episode_{int(episode_index):04d}_seed_{int(reset_seed)}"
                video_path = episode_dir / "selected_action_rollout.mp4"
                write_phase12_episode_video(
                    video_path=video_path,
                    frames=frames[row],
                    rewards=rewards[row],
                    successes=successes[row],
                    fps=int(args.fps),
                )
                row_obj = {
                    "episode_index": int(episode_index),
                    "reset_seed": int(reset_seed),
                    "sum_reward": float(sum(rewards[row])),
                    "max_reward": float(max(rewards[row])) if rewards[row] else 0.0,
                    "success": bool(any(successes[row])),
                    "n_steps": len(rewards[row]),
                    "video_path": str(video_path),
                }
                rows.append(row_obj)
                (episode_dir / "episode_summary.json").write_text(
                    json.dumps(row_obj, indent=2),
                    encoding="utf-8",
                )
        finally:
            for env in envs:
                env.close()

    success_count = sum(1 for row in rows if row["success"])
    summary = {
        "created_at": utc_now_iso(),
        "checkpoint": str(args.checkpoint),
        "task": str(args.task),
        "episodes": int(args.episodes),
        "eval_seed_start": int(args.eval_seed_start),
        "eval_seed_end": int(args.eval_seed_start) + int(args.episodes) - 1,
        "n_envs": int(args.n_envs),
        "max_steps": int(args.max_steps),
        "pc_success": 100.0 * float(success_count) / max(len(rows), 1),
        "avg_sum_reward": float(np.mean([row["sum_reward"] for row in rows])) if rows else 0.0,
        "avg_max_reward": float(np.mean([row["max_reward"] for row in rows])) if rows else 0.0,
        "reset_randomization_mode": "random_seeded",
        "video_enabled": True,
        "elapsed_seconds": float(perf_counter() - t0),
        "episodes_rows": rows,
    }
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("smolvla_baseline_vector_video_ok", f"out={output_dir}", f"pc_success={summary['pc_success']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
