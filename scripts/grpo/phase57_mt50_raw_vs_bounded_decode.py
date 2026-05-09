#!/usr/bin/env python3
"""Phase57 MT50 SmolVLA raw-vs-bounded JEPA decode audit."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from smolvla_grpo.phase12_decode_compare import (
    aligned_real_indices,
    build_action_variants,
    compute_raw_bounded_l2_metrics,
    encode_real_latents_for_indices,
    unroll_phase12_latent_trace,
    write_actions_npz,
    write_three_row_decode_strip_with_l2,
)
from smolvla_grpo.phase12_diagnostics import write_phase12_episode_video
from smolvla_grpo.phase12_logging import utc_now_iso, write_jsonl_row, write_manifest
from smolvla_grpo.phase12_pixels import policy_rgb_from_obs, wm_rgb_from_policy_rgb_corner2
from smolvla_grpo.phase12_vector_eval import build_episode_waves, concatenate_proc_rows, select_proc_rows


MT50_TASKS = [
    "assembly-v3",
    "basketball-v3",
    "bin-picking-v3",
    "box-close-v3",
    "button-press-topdown-v3",
    "button-press-topdown-wall-v3",
    "button-press-v3",
    "button-press-wall-v3",
    "coffee-button-v3",
    "coffee-pull-v3",
    "coffee-push-v3",
    "dial-turn-v3",
    "disassemble-v3",
    "door-close-v3",
    "door-lock-v3",
    "door-open-v3",
    "door-unlock-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "faucet-close-v3",
    "faucet-open-v3",
    "hammer-v3",
    "hand-insert-v3",
    "handle-press-side-v3",
    "handle-press-v3",
    "handle-pull-side-v3",
    "handle-pull-v3",
    "lever-pull-v3",
    "peg-insert-side-v3",
    "peg-unplug-side-v3",
    "pick-out-of-hole-v3",
    "pick-place-v3",
    "pick-place-wall-v3",
    "plate-slide-back-side-v3",
    "plate-slide-back-v3",
    "plate-slide-side-v3",
    "plate-slide-v3",
    "push-back-v3",
    "push-v3",
    "push-wall-v3",
    "reach-v3",
    "reach-wall-v3",
    "shelf-place-v3",
    "soccer-v3",
    "stick-pull-v3",
    "stick-push-v3",
    "sweep-into-v3",
    "sweep-v3",
    "window-close-v3",
    "window-open-v3",
]


def split_tasks_for_shard(tasks: list[str], *, shard_index: int, shard_count: int) -> list[str]:
    if int(shard_count) < 1:
        raise ValueError("shard_count must be >= 1")
    if int(shard_index) < 0 or int(shard_index) >= int(shard_count):
        raise ValueError("shard_index out of range")
    n = len(tasks)
    base = n // int(shard_count)
    rem = n % int(shard_count)
    start = 0
    for idx in range(int(shard_count)):
        count = base + (1 if idx < rem else 0)
        if idx == int(shard_index):
            return tasks[start : start + count]
        start += count
    return []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default="jadechoghari/smolvla_metaworld")
    p.add_argument("--jepa-ckpt", type=str, default="jepa_wm_metaworld.pth.tar")
    p.add_argument("--jepa-repo", type=str, default="")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task", type=str, default="")
    p.add_argument("--tasks", type=str, default="")
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--eval-seed-start", type=int, default=1000)
    p.add_argument("--n-envs", type=int, default=3)
    p.add_argument("--chunk-len", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=180)
    p.add_argument(
        "--env-vector-mode",
        choices=("serial", "sync", "async"),
        default="async",
        help="MetaWorld env stepping mode: serial keeps legacy per-row envs; sync/async use one vector env per wave.",
    )
    p.add_argument("--goal-latent-mode", choices=("visual_proprio", "visual_only_ablation"), default="visual_proprio")
    p.add_argument("--proprio-alpha", type=float, default=0.1)
    p.add_argument("--action-transform", choices=("no_tanh", "tanh_norm_ablation"), default="no_tanh")
    p.add_argument("--old-policy-inference-mode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--init-log-std", type=float, default=-2.0)
    p.add_argument("--euler-step-noise-std", type=float, default=0.2)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _task_list(args: argparse.Namespace) -> list[str]:
    raw = str(args.tasks or args.task or "push-v3")
    if raw == "all":
        return list(MT50_TASKS)
    return [x.strip() for x in raw.split(",") if x.strip()]


def _action_bounds(env_h: Any, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    space = getattr(getattr(env_h, "inner", None), "single_action_space", None)
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is None or high is None:
        return np.full((action_dim,), -1.0, dtype=np.float32), np.full((action_dim,), 1.0, dtype=np.float32)
    return np.asarray(low, dtype=np.float32).reshape(action_dim), np.asarray(high, dtype=np.float32).reshape(action_dim)


def _policy_rgb_from_vector_obs(obs: dict[str, Any], row: int) -> np.ndarray:
    if not isinstance(obs, dict) or "pixels" not in obs:
        raise KeyError("observation must contain 'pixels'")
    pixels = np.asarray(obs["pixels"])
    if pixels.ndim == 4:
        return policy_rgb_from_obs({"pixels": pixels[int(row)]})
    return policy_rgb_from_obs(obs)


def _agent_pos_from_vector_obs(obs: dict[str, Any], row: int) -> np.ndarray:
    if isinstance(obs, dict) and "agent_pos" in obs:
        agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32)
        if agent_pos.ndim >= 2 and int(agent_pos.shape[0]) > int(row):
            return np.asarray(agent_pos[int(row)], dtype=np.float32)
        return agent_pos.reshape(-1).astype(np.float32, copy=False)
    raise KeyError("observation must contain row-addressable 'agent_pos'")


def _episode_row(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_index": int(episode["episode_index"]),
        "reset_seed": int(episode["reset_seed"]),
        "sum_reward": float(episode["reward_sum"]),
        "max_reward": float(episode["max_reward"]),
        "success": bool(episode["success_any"]),
        "n_steps": int(episode["n_steps"]),
        "video_path": str(episode["video_path"]),
        "segment_count": int(episode["segment_count"]),
    }


def _write_eval_info(path: Path, *, task: str, episodes: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    rows = [_episode_row(ep) for ep in sorted(episodes, key=lambda x: int(x["episode_index"]))]
    sum_rewards = [float(r["sum_reward"]) for r in rows]
    max_rewards = [float(r["max_reward"]) for r in rows]
    successes = [bool(r["success"]) for r in rows]
    video_paths = [str(r["video_path"]) for r in rows]
    pc_success = 100.0 * sum(1 for ok in successes if ok) / max(len(successes), 1)
    eval_info = {
        "reset_randomization_mode": "random_seeded",
        "env_dispatched_source": "raw_postprocessed",
        "per_task": [
            {
                "task_group": str(task),
                "task_id": 0,
                "metrics": {
                    "sum_rewards": sum_rewards,
                    "max_rewards": max_rewards,
                    "successes": successes,
                    "video_paths": video_paths,
                },
            }
        ],
        "per_group": {
            str(task): {
                "avg_sum_reward": float(np.mean(sum_rewards)) if sum_rewards else 0.0,
                "avg_max_reward": float(np.mean(max_rewards)) if max_rewards else 0.0,
                "pc_success": float(pc_success),
                "n_episodes": len(rows),
                "video_paths": video_paths,
            }
        },
        "overall": {
            "avg_sum_reward": float(np.mean(sum_rewards)) if sum_rewards else 0.0,
            "avg_max_reward": float(np.mean(max_rewards)) if max_rewards else 0.0,
            "pc_success": float(pc_success),
            "n_episodes": len(rows),
            "eval_s": float(elapsed_s),
            "video_paths": video_paths,
        },
    }
    write_manifest(path, eval_info)
    return eval_info


def _summarize_l2(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    raw_vals: list[float] = []
    bounded_vals: list[float] = []
    winners: list[str] = []
    for ep in episodes:
        for seg in ep.get("segments", []):
            raw_vals.extend(float(x) for x in seg.get("raw", {}).get("combined_l2", []))
            bounded_vals.extend(float(x) for x in seg.get("bounded", {}).get("combined_l2", []))
            winners.extend(str(x) for x in seg.get("winner_by_column", []))
    total = max(len(winners), 1)
    return {
        "mean_raw_combined_l2": float(np.mean(raw_vals)) if raw_vals else None,
        "mean_bounded_combined_l2": float(np.mean(bounded_vals)) if bounded_vals else None,
        "raw_win_fraction": float(sum(1 for x in winners if x == "raw") / total),
        "bounded_win_fraction": float(sum(1 for x in winners if x == "bounded") / total),
        "tie_fraction": float(sum(1 for x in winners if x == "tie") / total),
        "metric_column_count": int(len(winners)),
    }


def _task_summary(
    *,
    args: argparse.Namespace,
    task: str,
    output_dir: Path,
    episodes: list[dict[str, Any]],
    elapsed_s: float,
) -> dict[str, Any]:
    rows = [_episode_row(ep) for ep in sorted(episodes, key=lambda x: int(x["episode_index"]))]
    successes = [bool(r["success"]) for r in rows]
    sum_rewards = [float(r["sum_reward"]) for r in rows]
    max_rewards = [float(r["max_reward"]) for r in rows]
    return {
        "created_at": utc_now_iso(),
        "task": str(task),
        "output_dir": str(output_dir),
        "episodes": int(args.episodes),
        "episodes_completed": len(rows),
        "eval_seed_start": int(args.eval_seed_start),
        "eval_seed_end": int(args.eval_seed_start) + int(args.episodes) - 1,
        "n_envs": int(args.n_envs),
        "env_vector_mode": str(args.env_vector_mode),
        "chunk_len": int(args.chunk_len),
        "max_steps": int(args.max_steps),
        "env_dispatched_source": "raw_postprocessed",
        "raw_wm_action_source": "raw_postprocessed",
        "bounded_wm_action_source": "clipped",
        "goal_latent_mode": str(args.goal_latent_mode),
        "reset_randomization_mode": "random_seeded",
        "pc_success": 100.0 * sum(1 for ok in successes if ok) / max(len(successes), 1),
        "avg_sum_reward": float(np.mean(sum_rewards)) if sum_rewards else 0.0,
        "avg_max_reward": float(np.mean(max_rewards)) if max_rewards else 0.0,
        "elapsed_seconds": float(elapsed_s),
        "episodes_rows": rows,
        "episodes_detail": episodes,
        **_summarize_l2(episodes),
    }


def _sample_chunk_batch(wrapper: Any, proc: Any, *, n_envs: int, chunk_len: int, seed: int, inference_mode: bool) -> Any:
    import torch

    if bool(inference_mode):
        with torch.inference_mode():
            return wrapper.sample_action_chunk_batch_from_proc(
                proc,
                n_envs=int(n_envs),
                chunk_len=int(chunk_len),
                reset_seed=int(seed),
            )
    return wrapper.sample_action_chunk_batch_from_proc(
        proc,
        n_envs=int(n_envs),
        chunk_len=int(chunk_len),
        reset_seed=int(seed),
    )


def run_task(
    *,
    args: argparse.Namespace,
    task: str,
    task_dir: Path,
    bundle: Any,
    wm_bundle: Any,
    action_dim: int,
) -> dict[str, Any]:
    from scripts.grpo.train_phase12_wm_chunk_grpo import build_train_wrapper
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout, resolve_lerobot_horizon

    task_args = copy.copy(args)
    task_args.task = str(task)
    wrapper, _trainable = build_train_wrapper(task_args, bundle, action_dim)
    wrapper._policy = bundle.policy.eval().to(bundle.device)
    task_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_row(
        task_dir / "progress.jsonl",
        {"created_at": utc_now_iso(), "event": "task_start", "task": str(task)},
    )
    episodes_out: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for wave in build_episode_waves(
        episodes=int(args.episodes),
        eval_seed_start=int(args.eval_seed_start),
        n_envs=int(args.n_envs),
    ):
        wave_n = len(wave)
        use_vector_env = str(args.env_vector_mode) != "serial"
        envs = [] if use_vector_env else [OfficialLeRobotMetaWorldGRPORollout(task=task, n_envs=1) for _ in range(wave_n)]
        env = (
            OfficialLeRobotMetaWorldGRPORollout(
                task=task,
                n_envs=wave_n,
                use_async_envs=str(args.env_vector_mode) == "async",
            )
            if use_vector_env
            else None
        )
        try:
            env_for_bounds = env if env is not None else envs[0]
            resolved_steps = min(int(args.max_steps), int(resolve_lerobot_horizon(env_for_bounds, int(args.max_steps))))
            if env is not None:
                obs = env.reset_many([seed for _ep, seed in wave])
                obs_by_row: list[dict[str, Any]] = []
            else:
                obs = {}
                obs_by_row = [row_env.reset(seed) for row_env, (_ep, seed) in zip(envs, wave, strict=True)]
            policy_reset = getattr(getattr(wrapper, "_policy", None), "reset", None)
            if callable(policy_reset):
                policy_reset()
            active = np.ones((wave_n,), dtype=np.bool_)
            low, high = _action_bounds(env_for_bounds, action_dim)
            if env is not None:
                policy_frames = [[_policy_rgb_from_vector_obs(obs, row)] for row in range(wave_n)]
                proprios = [[_agent_pos_from_vector_obs(obs, row)] for row in range(wave_n)]
            else:
                policy_frames = [[policy_rgb_from_obs(row_obs)] for row_obs in obs_by_row]
                proprios = [[np.asarray(row_env.last_agent_pos(), dtype=np.float32)] for row_env in envs]
            wm_frames = [[wm_rgb_from_policy_rgb_corner2(frames[0])] for frames in policy_frames]
            rewards: list[list[float]] = [[] for _ in range(wave_n)]
            successes: list[list[bool]] = [[] for _ in range(wave_n)]
            segments: list[list[dict[str, Any]]] = [[] for _ in range(wave_n)]

            while bool(np.any(active)):
                active_rows = [idx for idx in range(wave_n) if bool(active[idx])]
                if env is not None:
                    proc = select_proc_rows(env.build_proc(obs, bundle=bundle), active_rows, batch_size=wave_n)
                else:
                    proc = concatenate_proc_rows([envs[idx].build_proc(obs_by_row[idx], bundle=bundle) for idx in active_rows])
                sample_seed = int(args.eval_seed_start) * 1000003 + min(ep for ep, _seed in wave) * 7919
                sample_seed += max(len(segments[idx]) for idx in active_rows)
                chunk = _sample_chunk_batch(
                    wrapper,
                    proc,
                    n_envs=len(active_rows),
                    chunk_len=int(args.chunk_len),
                    seed=sample_seed,
                    inference_mode=bool(args.old_policy_inference_mode),
                )
                row_context: dict[int, dict[str, Any]] = {}
                for batch_row, row in enumerate(active_rows):
                    episode_index, reset_seed = wave[row]
                    root_image = np.asarray(wm_frames[row][-1], dtype=np.uint8).copy()
                    root_proprio = np.asarray(proprios[row][-1], dtype=np.float32).copy()
                    start = len(wm_frames[row]) - 1
                    variants = build_action_variants(
                        raw_actions=chunk.raw_postprocessed_action_np[batch_row],
                        clipped_actions=chunk.exec_action_np[batch_row],
                        action_low=low,
                        action_high=high,
                    )
                    row_context[row] = {
                        "episode_index": int(episode_index),
                        "reset_seed": int(reset_seed),
                        "root_image": root_image,
                        "root_proprio": root_proprio,
                        "start": int(start),
                        "variants": variants,
                        "executed": 0,
                    }

                if env is None:
                    for row, ctx in row_context.items():
                        variants = ctx["variants"]
                        executed = 0
                        for action in variants.env_actions:
                            if len(rewards[row]) >= resolved_steps:
                                active[row] = False
                                break
                            step = envs[row].step(np.asarray(action, dtype=np.float32).reshape(1, -1))
                            obs_by_row[row] = step.observation
                            pframe = policy_rgb_from_obs(step.observation)
                            policy_frames[row].append(pframe)
                            wm_frames[row].append(wm_rgb_from_policy_rgb_corner2(pframe))
                            proprios[row].append(np.asarray(envs[row].last_agent_pos(), dtype=np.float32))
                            rewards[row].append(float(step.reward))
                            successes[row].append(bool(step.success))
                            executed += 1
                            if bool(step.success or step.terminated or step.truncated):
                                active[row] = False
                                break
                        ctx["executed"] = executed
                else:
                    max_chunk_steps = max(len(ctx["variants"].env_actions) for ctx in row_context.values())
                    for chunk_step in range(max_chunk_steps):
                        if not bool(np.any(active)):
                            break
                        active_before_step = active.copy()
                        action_matrix = np.zeros((wave_n, int(action_dim)), dtype=np.float32)
                        rows_to_step: list[int] = []
                        for row, ctx in row_context.items():
                            variants = ctx["variants"]
                            if not bool(active_before_step[row]) or len(rewards[row]) >= resolved_steps:
                                active[row] = False
                                continue
                            if int(chunk_step) >= len(variants.env_actions):
                                continue
                            action_matrix[row] = np.asarray(variants.env_actions[chunk_step], dtype=np.float32).reshape(-1)
                            rows_to_step.append(row)
                        if not rows_to_step:
                            break
                        step = env.step_batch(action_matrix)
                        obs = step.observation
                        for row in rows_to_step:
                            pframe = _policy_rgb_from_vector_obs(step.observation, row)
                            policy_frames[row].append(pframe)
                            wm_frames[row].append(wm_rgb_from_policy_rgb_corner2(pframe))
                            proprios[row].append(_agent_pos_from_vector_obs(step.observation, row))
                            rewards[row].append(float(step.reward[row]))
                            successes[row].append(bool(step.success[row]))
                            row_context[row]["executed"] = int(row_context[row]["executed"]) + 1
                            if bool(step.success[row] or step.terminated[row] or step.truncated[row]):
                                active[row] = False
                            elif len(rewards[row]) >= resolved_steps:
                                active[row] = False

                for row, ctx in row_context.items():
                    variants = ctx["variants"]
                    executed = int(ctx["executed"])
                    if executed < 1:
                        continue
                    raw_exec = variants.raw_wm_actions[:executed]
                    bounded_exec = variants.bounded_wm_actions[:executed]
                    real_frames = list(wm_frames[row][int(ctx["start"]) :])
                    real_proprios = list(proprios[row][int(ctx["start"]) :])
                    raw_trace = unroll_phase12_latent_trace(
                        wm_bundle,
                        image=ctx["root_image"],
                        proprio=ctx["root_proprio"],
                        actions=raw_exec,
                        mode=args.goal_latent_mode,
                        decode_frames=True,
                    )
                    bounded_trace = unroll_phase12_latent_trace(
                        wm_bundle,
                        image=ctx["root_image"],
                        proprio=ctx["root_proprio"],
                        actions=bounded_exec,
                        mode=args.goal_latent_mode,
                        decode_frames=True,
                    )
                    raw_pred = list(raw_trace.frames or [])
                    bounded_pred = list(bounded_trace.frames or [])
                    pred_count = min(len(raw_pred), len(bounded_pred))
                    indices = aligned_real_indices(
                        pred_count=pred_count,
                        env_steps_per_wm_step=int(raw_trace.wm_factor),
                        carried_steps=min(executed, max(0, len(real_frames) - 1)),
                        real_frame_count=len(real_frames),
                    )
                    real_latents = encode_real_latents_for_indices(
                        wm_bundle,
                        frames=real_frames,
                        proprios=real_proprios,
                        indices=indices,
                        mode=args.goal_latent_mode,
                    )
                    metrics = compute_raw_bounded_l2_metrics(
                        raw_pred_latents=raw_trace.structured_latents[: len(indices)],
                        bounded_pred_latents=bounded_trace.structured_latents[: len(indices)],
                        real_latents=real_latents,
                        mode=args.goal_latent_mode,
                        proprio_alpha=float(args.proprio_alpha),
                    )
                    episode_dir = task_dir / f"episode_{int(ctx['episode_index']):04d}_seed_{int(ctx['reset_seed'])}"
                    segment_dir = episode_dir / f"segment_{len(segments[row]):04d}"
                    strip_path = write_three_row_decode_strip_with_l2(
                        segment_dir / "real_raw_bounded_decode_strip_l2.png",
                        real_frames=real_frames,
                        raw_pred_frames=raw_pred,
                        bounded_pred_frames=bounded_pred,
                        env_steps_per_wm_step=int(raw_trace.wm_factor),
                        carried_steps=min(executed, max(0, len(real_frames) - 1)),
                        raw_combined_l2=metrics["raw"]["combined_l2"],
                        bounded_combined_l2=metrics["bounded"]["combined_l2"],
                    )
                    actions_path = write_actions_npz(
                        segment_dir / "actions_raw_bounded_env.npz",
                        raw_actions=raw_exec,
                        bounded_actions=bounded_exec,
                        env_actions=variants.env_actions[:executed],
                        sampled_raw_actions=variants.raw_wm_actions,
                        sampled_bounded_actions=variants.bounded_wm_actions,
                    )
                    segment_meta = {
                        "segment_index": len(segments[row]),
                        "episode_index": int(ctx["episode_index"]),
                        "reset_seed": int(ctx["reset_seed"]),
                        "sample_seed": int(sample_seed),
                        "executed_steps": int(executed),
                        "wm_factor": int(raw_trace.wm_factor),
                        "aligned_real_indices": [int(x) for x in indices],
                        "strip_path": str(strip_path),
                        "actions_path": str(actions_path),
                        **variants.metadata,
                        **metrics,
                    }
                    write_manifest(segment_dir / "latent_l2_metrics.json", segment_meta)
                    segments[row].append(segment_meta)

            for row, (episode_index, reset_seed) in enumerate(wave):
                episode_dir = task_dir / f"episode_{int(episode_index):04d}_seed_{int(reset_seed)}"
                video_path = write_phase12_episode_video(
                    video_path=episode_dir / "selected_action_rollout.mp4",
                    frames=policy_frames[row],
                    rewards=rewards[row],
                    successes=successes[row],
                    fps=int(args.fps),
                )
                ep = {
                    "episode_index": int(episode_index),
                    "reset_seed": int(reset_seed),
                    "video_path": str(video_path),
                    "segment_count": int(len(segments[row])),
                    "segments": segments[row],
                    "reward_sum": float(sum(rewards[row])),
                    "max_reward": float(max(rewards[row])) if rewards[row] else 0.0,
                    "success_any": bool(any(successes[row])),
                    "success_last": bool(successes[row][-1]) if successes[row] else False,
                    "n_steps": int(len(rewards[row])),
                }
                write_manifest(episode_dir / "episode_manifest.json", ep)
                episodes_out.append(ep)
                write_jsonl_row(
                    task_dir / "progress.jsonl",
                    {
                        "created_at": utc_now_iso(),
                        "event": "episode_complete",
                        "task": str(task),
                        "episode_index": int(episode_index),
                        "reset_seed": int(reset_seed),
                        "success_any": bool(ep["success_any"]),
                        "n_steps": int(ep["n_steps"]),
                    },
                )
        finally:
            if env is not None:
                env.close()
            for env in envs:
                env.close()

    elapsed = time.perf_counter() - t0
    summary = _task_summary(args=args, task=task, output_dir=task_dir, episodes=episodes_out, elapsed_s=elapsed)
    write_manifest(task_dir / "task_summary.json", summary)
    _write_eval_info(task_dir / "eval_info.json", task=task, episodes=episodes_out, elapsed_s=elapsed)
    return summary


def run(args: argparse.Namespace) -> int:
    from scripts.grpo.train_phase12_wm_chunk_grpo import load_phase12_train_resources

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    tasks = _task_list(args)
    manifest = {
        "created_at": utc_now_iso(),
        "mode": "phase57_mt50_raw_vs_bounded_decode",
        "tasks": tasks,
        "episodes": int(args.episodes),
        "eval_seed_start": int(args.eval_seed_start),
        "n_envs": int(args.n_envs),
        "env_vector_mode": str(args.env_vector_mode),
        "chunk_len": int(args.chunk_len),
        "max_steps": int(args.max_steps),
        "env_dispatched_source": "raw_postprocessed",
    }
    write_manifest(out / "phase57_manifest.json", manifest)
    if args.dry_run:
        print("PHASE57_DRY_RUN", f"out={out}", f"tasks={len(tasks)}", flush=True)
        return 0
    if not str(args.jepa_repo).strip():
        raise RuntimeError("Missing --jepa-repo")
    bundle, wm_bundle, action_dim = load_phase12_train_resources(args)
    summaries = []
    for task in tasks:
        summaries.append(
            run_task(
                args=args,
                task=task,
                task_dir=out / task,
                bundle=bundle,
                wm_bundle=wm_bundle,
                action_dim=action_dim,
            )
        )
    write_manifest(out / "phase57_tasks_summary.json", {"tasks": summaries})
    print("PHASE57_MT50_RAW_VS_BOUNDED_DECODE_DONE", f"out={out}", f"tasks={len(tasks)}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
