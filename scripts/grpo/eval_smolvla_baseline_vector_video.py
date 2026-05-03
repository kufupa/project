#!/usr/bin/env python3
"""Vectorized SmolVLA baseline eval with per-episode videos."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout, resolve_lerobot_horizon
from smolvla_grpo.phase11_rollout import load_bundle_for_grpo
from smolvla_grpo.phase12_diagnostics import write_phase12_episode_video
from smolvla_grpo.phase12_logging import utc_now_iso, write_jsonl_row
from smolvla_grpo.phase12_vector_eval import (
    TimingAccumulator,
    build_episode_waves,
    coerce_exec_action_batch,
    coerce_exec_action_chunk_batch,
    concatenate_proc_rows,
    select_eval_action_chunk_queue_free_timed,
    select_eval_action_queue_free_timed,
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


def _frame_from_vector_obs(obs: dict[str, Any], row: int) -> np.ndarray | None:
    pixels = obs.get("pixels") if isinstance(obs, dict) else None
    if pixels is None and isinstance(obs, dict):
        pixels = obs.get("observation.image")
    if pixels is None:
        return None
    arr = np.asarray(pixels)
    if arr.ndim == 4:
        frame = arr[int(row)]
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


def _select_proc_rows(proc: dict[str, Any], rows: list[int], *, batch_size: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must be non-empty")
    out: dict[str, Any] = {}
    for key, value in proc.items():
        if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == int(batch_size):
            out[key] = value[rows]
        elif isinstance(value, np.ndarray) and value.ndim > 0 and int(value.shape[0]) == int(batch_size):
            out[key] = value[rows]
        elif isinstance(value, (list, tuple)) and len(value) == int(batch_size):
            out[key] = [value[int(row)] for row in rows]
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce_actions_for_proc(
    *,
    args: argparse.Namespace,
    bundle: Any,
    proc: dict[str, Any],
    action_dim: int,
    n_envs: int,
    effective_chunk: int,
    timings: TimingAccumulator,
) -> np.ndarray:
    with torch.inference_mode():
        if int(args.chunk_len) == 1:
            action = select_eval_action_queue_free_timed(bundle.policy, proc, timings=timings)
            post_t0 = perf_counter()
            post = bundle.postprocessor(action)
            timings.add("postprocess_seconds", perf_counter() - post_t0)
            coerce_t0 = perf_counter()
            exec_action_np = coerce_exec_action_batch(
                post,
                action_dim=int(action_dim),
                n_envs=int(n_envs),
            )[:, None, :]
            timings.add("action_coerce_seconds", perf_counter() - coerce_t0)
        else:
            action = select_eval_action_chunk_queue_free_timed(
                bundle.policy,
                proc,
                chunk_len=effective_chunk,
                timings=timings,
            )
            post_t0 = perf_counter()
            post = bundle.postprocessor(action)
            timings.add("postprocess_seconds", perf_counter() - post_t0)
            coerce_t0 = perf_counter()
            exec_action_np = coerce_exec_action_chunk_batch(
                post,
                action_dim=int(action_dim),
                n_envs=int(n_envs),
                chunk_len=effective_chunk,
            )
            timings.add("action_coerce_seconds", perf_counter() - coerce_t0)
    return exec_action_np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=BASE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, default="push-v3")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--eval-seed-start", type=int, default=1000)
    parser.add_argument("--n-envs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--chunk-len", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--env-vector-mode",
        choices=("serial", "sync", "async"),
        default="serial",
        help="MetaWorld env stepping mode: serial keeps legacy per-row envs; sync/async use one vector env per wave.",
    )
    parser.add_argument("--timing-sync-cuda", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if int(args.chunk_len) < 1:
        parser.error("--chunk-len must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timings = TimingAccumulator(cuda_sync_requested=bool(args.timing_sync_cuda))
    load_t0 = perf_counter()
    bundle, _action_dim = load_bundle_for_grpo(
        args.checkpoint,
        task=args.task,
        env_backend="official_lerobot",
        n_action_steps=int(args.chunk_len),
    )
    timings.add("load_bundle_seconds", perf_counter() - load_t0)
    bundle.policy.eval()

    rows: list[dict[str, Any]] = []
    t0 = perf_counter()
    rollout_t0 = perf_counter()
    waves = build_episode_waves(
        episodes=int(args.episodes),
        eval_seed_start=int(args.eval_seed_start),
        n_envs=int(args.n_envs),
    )
    for wave_index, wave in enumerate(waves):
        wave_n = len(wave)
        if str(args.env_vector_mode) != "serial":
            env = OfficialLeRobotMetaWorldGRPORollout(
                task=args.task,
                n_envs=wave_n,
                use_async_envs=str(args.env_vector_mode) == "async",
            )
            try:
                resolved_steps = resolve_lerobot_horizon(env, int(args.max_steps))
                reset_t0 = perf_counter()
                obs = env.reset_many([seed for _ep, seed in wave])
                timings.add("reset_seconds", perf_counter() - reset_t0)
                reset = getattr(bundle.policy, "reset", None)
                if callable(reset):
                    reset()
                active = np.ones((wave_n,), dtype=np.bool_)
                actions: list[list[list[float]]] = [[] for _ in range(wave_n)]
                rewards: list[list[float]] = [[] for _ in range(wave_n)]
                successes: list[list[bool]] = [[] for _ in range(wave_n)]
                frames: list[list[np.ndarray]] = [[] for _ in range(wave_n)]
                for row in range(wave_n):
                    frame_t0 = perf_counter()
                    frame = _frame_from_vector_obs(obs, row)
                    if frame is None:
                        raise RuntimeError("vector env observation did not include row-addressable pixels")
                    frames[row].append(frame)
                    timings.add("frame_extract_seconds", perf_counter() - frame_t0)
                    timings.incr("n_video_frames")

                step_count = 0
                while step_count < int(resolved_steps):
                    if not bool(np.any(active)):
                        break
                    active_rows = [idx for idx in range(wave_n) if bool(active[idx])]
                    proc_t0 = perf_counter()
                    proc = _select_proc_rows(env.build_proc(obs, bundle=bundle), active_rows, batch_size=wave_n)
                    timings.add("proc_build_seconds", perf_counter() - proc_t0)
                    effective_chunk = min(int(args.chunk_len), int(resolved_steps) - int(step_count))
                    exec_action_np = _coerce_actions_for_proc(
                        args=args,
                        bundle=bundle,
                        proc=proc,
                        action_dim=int(env.action_dim),
                        n_envs=len(active_rows),
                        effective_chunk=effective_chunk,
                        timings=timings,
                    )
                    for chunk_step in range(effective_chunk):
                        if not bool(np.any(active)):
                            break
                        active_before_step = active.copy()
                        step_active_rows = [idx for idx in range(wave_n) if bool(active_before_step[idx])]
                        action_matrix = np.zeros((wave_n, int(env.action_dim)), dtype=np.float32)
                        for batch_row, row in enumerate(active_rows):
                            if not bool(active_before_step[row]):
                                continue
                            action_matrix[row] = exec_action_np[batch_row, chunk_step]
                        step_t0 = perf_counter()
                        step = env.step_batch(action_matrix)
                        step_seconds = perf_counter() - step_t0
                        timings.add("metaworld_step_seconds_including_obs_render", step_seconds)
                        timings.add("metaworld_step_batch_seconds_including_obs_render", step_seconds)
                        timings.incr("n_env_batch_steps")
                        timings.incr("n_env_steps", len(step_active_rows))
                        obs = step.observation
                        for batch_row, row in enumerate(active_rows):
                            if not bool(active_before_step[row]):
                                continue
                            actions[row].append(exec_action_np[batch_row, chunk_step].reshape(-1).tolist())
                            rewards[row].append(float(step.reward[row]))
                            successes[row].append(bool(step.success[row]))
                            frame_t0 = perf_counter()
                            frame = _frame_from_vector_obs(step.observation, row)
                            if frame is None:
                                raise RuntimeError("vector env observation did not include row-addressable pixels")
                            frames[row].append(frame)
                            timings.add("frame_extract_seconds", perf_counter() - frame_t0)
                            timings.incr("n_video_frames")
                            if step.success[row] or step.terminated[row] or step.truncated[row]:
                                active[row] = False
                        step_count += 1
                        if step_count >= int(resolved_steps):
                            break

                for row, (episode_index, reset_seed) in enumerate(wave):
                    episode_dir = output_dir / f"episode_{int(episode_index):04d}_seed_{int(reset_seed)}"
                    video_path = episode_dir / "selected_action_rollout.mp4"
                    video_t0 = perf_counter()
                    write_phase12_episode_video(
                        video_path=video_path,
                        frames=frames[row],
                        rewards=rewards[row],
                        successes=successes[row],
                        fps=int(args.fps),
                    )
                    video_seconds = perf_counter() - video_t0
                    timings.add("video_write_seconds", video_seconds)
                    row_obj = {
                        "episode_index": int(episode_index),
                        "reset_seed": int(reset_seed),
                        "sum_reward": float(sum(rewards[row])),
                        "max_reward": float(max(rewards[row])) if rewards[row] else 0.0,
                        "success": bool(any(successes[row])),
                        "n_steps": len(rewards[row]),
                        "n_frames": len(frames[row]),
                        "video_write_seconds": float(video_seconds),
                        "video_path": str(video_path),
                    }
                    rows.append(row_obj)
                    (episode_dir / "episode_summary.json").write_text(
                        json.dumps(row_obj, indent=2),
                        encoding="utf-8",
                    )
            finally:
                close_t0 = perf_counter()
                env.close()
                timings.add("close_seconds", perf_counter() - close_t0)

            write_jsonl_row(
                output_dir / "timings.jsonl",
                {
                    "event": "phase58_wave_timing",
                    "created_at": utc_now_iso(),
                    "wave_index": int(wave_index),
                    "episode_indices": [int(ep) for ep, _seed in wave],
                    **timings.summary(),
                },
            )
            continue

        envs = [OfficialLeRobotMetaWorldGRPORollout(task=args.task, n_envs=1) for _ in range(wave_n)]
        try:
            resolved_steps = resolve_lerobot_horizon(envs[0], int(args.max_steps))
            reset_t0 = perf_counter()
            obs_by_row = [env.reset(seed) for env, (_ep, seed) in zip(envs, wave, strict=True)]
            timings.add("reset_seconds", perf_counter() - reset_t0)
            reset = getattr(bundle.policy, "reset", None)
            if callable(reset):
                reset()
            active = np.ones((wave_n,), dtype=np.bool_)
            actions: list[list[list[float]]] = [[] for _ in range(wave_n)]
            rewards: list[list[float]] = [[] for _ in range(wave_n)]
            successes: list[list[bool]] = [[] for _ in range(wave_n)]
            frames: list[list[np.ndarray]] = [[] for _ in range(wave_n)]
            for row, obs in enumerate(obs_by_row):
                frame_t0 = perf_counter()
                frame = _frame_from_obs(obs)
                if frame is None:
                    frame = envs[row].render_frame()
                frames[row].append(frame)
                timings.add("frame_extract_seconds", perf_counter() - frame_t0)
                timings.incr("n_video_frames")

            step_count = 0
            while step_count < int(resolved_steps):
                if not bool(np.any(active)):
                    break
                active_rows = [idx for idx in range(wave_n) if bool(active[idx])]
                proc_t0 = perf_counter()
                proc = concatenate_proc_rows(
                    [envs[idx].build_proc(obs_by_row[idx], bundle=bundle) for idx in active_rows]
                )
                timings.add("proc_build_seconds", perf_counter() - proc_t0)
                effective_chunk = min(int(args.chunk_len), int(resolved_steps) - int(step_count))
                exec_action_np = _coerce_actions_for_proc(
                    args=args,
                    bundle=bundle,
                    proc=proc,
                    action_dim=int(envs[0].action_dim),
                    n_envs=len(active_rows),
                    effective_chunk=effective_chunk,
                    timings=timings,
                )
                for chunk_step in range(effective_chunk):
                    if not bool(np.any(active)):
                        break
                    for batch_row, row in enumerate(active_rows):
                        if not bool(active[row]):
                            continue
                        step_t0 = perf_counter()
                        step = envs[row].step(exec_action_np[batch_row, chunk_step : chunk_step + 1])
                        timings.add("metaworld_step_seconds_including_obs_render", perf_counter() - step_t0)
                        timings.incr("n_env_steps")
                        obs_by_row[row] = step.observation
                        actions[row].append(exec_action_np[batch_row, chunk_step].reshape(-1).tolist())
                        rewards[row].append(float(step.reward))
                        successes[row].append(bool(step.success))
                        frame_t0 = perf_counter()
                        frame = _frame_from_obs(step.observation)
                        if frame is None:
                            frame = envs[row].render_frame()
                        frames[row].append(frame)
                        timings.add("frame_extract_seconds", perf_counter() - frame_t0)
                        timings.incr("n_video_frames")
                        if step.success or step.terminated or step.truncated:
                            active[row] = False
                    step_count += 1
                    if step_count >= int(resolved_steps):
                        break

            for row, (episode_index, reset_seed) in enumerate(wave):
                episode_dir = output_dir / f"episode_{int(episode_index):04d}_seed_{int(reset_seed)}"
                video_path = episode_dir / "selected_action_rollout.mp4"
                video_t0 = perf_counter()
                write_phase12_episode_video(
                    video_path=video_path,
                    frames=frames[row],
                    rewards=rewards[row],
                    successes=successes[row],
                    fps=int(args.fps),
                )
                video_seconds = perf_counter() - video_t0
                timings.add("video_write_seconds", video_seconds)
                row_obj = {
                    "episode_index": int(episode_index),
                    "reset_seed": int(reset_seed),
                    "sum_reward": float(sum(rewards[row])),
                    "max_reward": float(max(rewards[row])) if rewards[row] else 0.0,
                    "success": bool(any(successes[row])),
                    "n_steps": len(rewards[row]),
                    "n_frames": len(frames[row]),
                    "video_write_seconds": float(video_seconds),
                    "video_path": str(video_path),
                }
                rows.append(row_obj)
                (episode_dir / "episode_summary.json").write_text(
                    json.dumps(row_obj, indent=2),
                    encoding="utf-8",
                )
        finally:
            close_t0 = perf_counter()
            for env in envs:
                env.close()
            timings.add("close_seconds", perf_counter() - close_t0)

        write_jsonl_row(
            output_dir / "timings.jsonl",
            {
                "event": "phase58_wave_timing",
                "created_at": utc_now_iso(),
                "wave_index": int(wave_index),
                "episode_indices": [int(ep) for ep, _seed in wave],
                **timings.summary(),
            },
        )

    success_count = sum(1 for row in rows if row["success"])
    timings.add("rollout_seconds", perf_counter() - rollout_t0)
    timing_summary = timings.summary()
    timing_summary_path = output_dir / "timing_summary.json"
    timing_summary_path.write_text(json.dumps(timing_summary, indent=2), encoding="utf-8")
    write_jsonl_row(
        output_dir / "timings.jsonl",
        {"event": "phase58_timing_summary", "created_at": utc_now_iso(), **timing_summary},
    )
    summary = {
        "created_at": utc_now_iso(),
        "checkpoint": str(args.checkpoint),
        "task": str(args.task),
        "episodes": int(args.episodes),
        "eval_seed_start": int(args.eval_seed_start),
        "eval_seed_end": int(args.eval_seed_start) + int(args.episodes) - 1,
        "n_envs": int(args.n_envs),
        "env_vector_mode": str(args.env_vector_mode),
        "max_steps": int(args.max_steps),
        "chunk_len": int(args.chunk_len),
        "rollout_execution": "chunk_open_loop" if int(args.chunk_len) > 1 else "one_step_queue_free",
        "pc_success": 100.0 * float(success_count) / max(len(rows), 1),
        "avg_sum_reward": float(np.mean([row["sum_reward"] for row in rows])) if rows else 0.0,
        "avg_max_reward": float(np.mean([row["max_reward"] for row in rows])) if rows else 0.0,
        "reset_randomization_mode": "random_seeded",
        "video_enabled": True,
        "elapsed_seconds": float(perf_counter() - t0),
        "timings": timing_summary,
        "timing_summary_path": str(timing_summary_path),
        "episodes_rows": rows,
    }
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("smolvla_baseline_vector_video_ok", f"out={output_dir}", f"pc_success={summary['pc_success']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
