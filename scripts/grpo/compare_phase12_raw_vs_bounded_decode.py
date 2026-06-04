#!/usr/bin/env python3
"""Compare Phase12 JEPA decodes for raw vs bounded actions from one sampled chunk."""

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
    build_action_variants,
    build_summary,
    decode_phase12_prediction_frames,
    write_actions_npz,
    write_three_row_decode_strip,
)
from smolvla_grpo.phase12_diagnostics import write_phase12_episode_video
from smolvla_grpo.phase12_logging import utc_now_iso, write_jsonl_row, write_manifest
from smolvla_grpo.phase12_pixels import policy_rgb_from_obs, wm_rgb_from_policy_rgb_corner2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default="jadechoghari/smolvla_metaworld")
    p.add_argument("--jepa-ckpt", type=str, default="jepa_wm_metaworld.pth.tar")
    p.add_argument("--jepa-repo", type=str, default="")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/phase12_raw_vs_bounded_decode/dry_run"))
    p.add_argument("--task", type=str, default="push-v3")
    p.add_argument("--num-episodes", type=int, default=6)
    p.add_argument("--chunk-len", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--train-seed-base", type=int, default=2000)
    p.add_argument("--goal-latent-mode", choices=("visual_proprio", "visual_only_ablation"), default="visual_proprio")
    p.add_argument("--action-transform", choices=("no_tanh", "tanh_norm_ablation"), default="no_tanh")
    p.add_argument("--old-policy-inference-mode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--init-log-std", type=float, default=-2.0)
    p.add_argument("--euler-step-noise-std", type=float, default=0.2)
    p.add_argument("--strict-decode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def build_run_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "created_at": utc_now_iso(),
        "mode": "raw_vs_bounded_decode_compare",
        "comparison_type": "raw_vs_bounded_same_actions",
        "checkpoint": str(args.checkpoint),
        "jepa_ckpt": str(args.jepa_ckpt),
        "jepa_repo": str(args.jepa_repo),
        "task": str(args.task),
        "num_episodes": int(args.num_episodes),
        "chunk_len": int(args.chunk_len),
        "max_steps": int(args.max_steps),
        "train_seed_base": int(args.train_seed_base),
        "goal_latent_mode": str(args.goal_latent_mode),
        "action_transform": str(args.action_transform),
        "old_policy_inference_mode": bool(args.old_policy_inference_mode),
        "env_dispatched_source": "raw_postprocessed",
        "raw_wm_action_source": "raw_postprocessed",
        "bounded_wm_action_source": "clipped",
        "phase12_policy_frame_contract": "lerobot_corner2_vhflip",
        "phase12_wm_frame_contract": "jepa_corner2_vflip",
    }


def _validate_args(args: argparse.Namespace) -> str | None:
    if int(args.num_episodes) < 1:
        return "--num-episodes must be >= 1"
    if int(args.chunk_len) < 1:
        return "--chunk-len must be >= 1"
    if int(args.max_steps) < 1:
        return "--max-steps must be >= 1"
    if not str(args.jepa_repo).strip():
        return "Missing --jepa-repo for JEPA decode compare."
    if not str(args.jepa_ckpt).strip():
        return "Missing --jepa-ckpt for JEPA decode compare."
    return None


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _action_bounds(env_h: Any, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    space = getattr(getattr(env_h, "inner", None), "single_action_space", None)
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is None or high is None:
        return np.full((action_dim,), -1.0, dtype=np.float32), np.full((action_dim,), 1.0, dtype=np.float32)
    return np.asarray(low, dtype=np.float32).reshape(action_dim), np.asarray(high, dtype=np.float32).reshape(action_dim)


def _wm_factor(wm_bundle: Any, action_dim: int) -> int:
    from segment_grpo_loop import _infer_model_action_dim, _wm_action_block_factor

    model_action_dim = _infer_model_action_dim(wm_bundle.model)
    wm_dim = int(model_action_dim) if model_action_dim else int(wm_bundle.planner_action_dim)
    return int(_wm_action_block_factor(int(action_dim), int(wm_dim)))


def _sample_chunk(wrapper: Any, proc: Any, *, chunk_len: int, seed: int, inference_mode: bool) -> Any:
    import torch

    gen = torch.Generator(device=wrapper.bundle.device)
    gen.manual_seed(int(seed))
    if bool(inference_mode):
        with torch.inference_mode():
            return wrapper.sample_action_chunk_from_proc(proc, chunk_len=int(chunk_len), rng=gen)
    return wrapper.sample_action_chunk_from_proc(proc, chunk_len=int(chunk_len), rng=gen)


def _episode_manifest(
    *,
    episode_index: int,
    reset_seed: int,
    video_path: Path,
    segments: list[dict[str, Any]],
    rewards: list[float],
    successes: list[bool],
) -> dict[str, Any]:
    return {
        "episode_index": int(episode_index),
        "reset_seed": int(reset_seed),
        "video_path": str(video_path),
        "segment_count": int(len(segments)),
        "segments": segments,
        "reward_sum": float(sum(rewards)),
        "success_any": bool(any(successes)),
        "success_last": bool(successes[-1]) if successes else False,
    }


def run_episode(
    *,
    args: argparse.Namespace,
    env_h: Any,
    wrapper: Any,
    wm_bundle: Any,
    action_dim: int,
    episode_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    reset_seed = int(args.train_seed_base) + int(episode_index)
    episode_dir = Path(output_dir) / f"episode_{int(episode_index):04d}"
    obs = env_h.reset(reset_seed)
    policy_reset = getattr(getattr(wrapper, "_policy", None), "reset", None)
    if callable(policy_reset):
        policy_reset()
    policy_frame = policy_rgb_from_obs(obs)
    wm_frame = wm_rgb_from_policy_rgb_corner2(policy_frame)
    proprio = np.asarray(env_h.last_agent_pos(), dtype=np.float32)
    policy_frames = [policy_frame]
    wm_frames = [wm_frame]
    rewards: list[float] = []
    successes: list[bool] = []
    segments: list[dict[str, Any]] = []
    low, high = _action_bounds(env_h, action_dim)
    factor = _wm_factor(wm_bundle, action_dim)
    max_steps = int(args.max_steps)
    segment_index = 0
    done = False

    while len(rewards) < max_steps and not done:
        proc = env_h.build_proc(obs, bundle=wrapper.bundle)
        root_image = np.asarray(wm_frames[-1], dtype=np.uint8).copy()
        root_proprio = np.asarray(proprio, dtype=np.float32).copy()
        sample_seed = reset_seed * 1000003 + int(segment_index) * 7919
        sample = _sample_chunk(
            wrapper,
            proc,
            chunk_len=int(args.chunk_len),
            seed=sample_seed,
            inference_mode=bool(args.old_policy_inference_mode),
        )
        variants = build_action_variants(
            raw_actions=sample.raw_postprocessed_action_np,
            clipped_actions=sample.exec_action_np,
            action_low=low,
            action_high=high,
        )
        start = len(wm_frames) - 1
        executed = 0
        for action in variants.env_actions:
            if len(rewards) >= max_steps:
                break
            step = env_h.step(np.asarray(action, dtype=np.float32).reshape(1, -1))
            obs = step.observation
            policy_frame = policy_rgb_from_obs(obs)
            wm_frame = wm_rgb_from_policy_rgb_corner2(policy_frame)
            proprio = np.asarray(env_h.last_agent_pos(), dtype=np.float32)
            policy_frames.append(policy_frame)
            wm_frames.append(wm_frame)
            rewards.append(float(step.reward))
            successes.append(bool(step.success))
            executed += 1
            if bool(step.terminated or step.truncated):
                done = True
                break
        if executed < 1:
            break

        raw_exec = variants.raw_wm_actions[:executed]
        bounded_exec = variants.bounded_wm_actions[:executed]
        real_segment_frames = list(wm_frames[start:])
        raw_pred = decode_phase12_prediction_frames(
            wm_bundle,
            image=root_image,
            proprio=root_proprio,
            actions=raw_exec,
            mode=args.goal_latent_mode,
        )
        bounded_pred = decode_phase12_prediction_frames(
            wm_bundle,
            image=root_image,
            proprio=root_proprio,
            actions=bounded_exec,
            mode=args.goal_latent_mode,
        )
        segment_dir = episode_dir / f"segment_{int(segment_index):04d}"
        strip_path = write_three_row_decode_strip(
            segment_dir / "real_raw_bounded_decode_strip.png",
            real_frames=real_segment_frames,
            raw_pred_frames=raw_pred,
            bounded_pred_frames=bounded_pred,
            env_steps_per_wm_step=factor,
            carried_steps=min(executed, max(0, len(real_segment_frames) - 1)),
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
            "segment_index": int(segment_index),
            "sample_seed": int(sample_seed),
            "executed_steps": int(executed),
            "wm_factor": int(factor),
            "decoded_raw_frames": int(len(raw_pred)),
            "decoded_bounded_frames": int(len(bounded_pred)),
            "strip_path": str(strip_path),
            "actions_path": str(actions_path),
            **variants.metadata,
        }
        (segment_dir / "segment_manifest.json").write_text(
            json.dumps(_json_ready(segment_meta), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        segments.append(segment_meta)
        segment_index += 1

    video_path = write_phase12_episode_video(
        video_path=episode_dir / "selected_action_rollout.mp4",
        frames=policy_frames,
        rewards=rewards,
        successes=successes,
        fps=int(args.fps),
    )
    manifest = _episode_manifest(
        episode_index=episode_index,
        reset_seed=reset_seed,
        video_path=video_path,
        segments=segments,
        rewards=rewards,
        successes=successes,
    )
    write_manifest(episode_dir / "episode_manifest.json", manifest)
    return manifest


def run_compare(args: argparse.Namespace, out: Path) -> int:
    import torch
    from scripts.grpo.train_phase12_wm_chunk_grpo import build_train_wrapper, load_phase12_train_resources
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout, resolve_lerobot_horizon

    bundle, wm_bundle, action_dim = load_phase12_train_resources(args)
    wrapper, _trainable = build_train_wrapper(args, bundle, action_dim)
    wrapper._policy = copy.deepcopy(bundle.policy).eval().to(bundle.device)
    for param in wrapper._policy.parameters():
        param.requires_grad_(False)
    env_h = OfficialLeRobotMetaWorldGRPORollout(task=args.task, n_envs=1, enable_expert_oracle=False)
    try:
        args.max_steps = min(int(args.max_steps), int(resolve_lerobot_horizon(env_h, int(args.max_steps))))
        episodes: list[dict[str, Any]] = []
        for episode_index in range(int(args.num_episodes)):
            t0 = time.perf_counter()
            episode = run_episode(
                args=args,
                env_h=env_h,
                wrapper=wrapper,
                wm_bundle=wm_bundle,
                action_dim=action_dim,
                episode_index=episode_index,
                output_dir=out,
            )
            episodes.append(episode)
            write_jsonl_row(
                out / "progress.jsonl",
                {
                    "created_at": utc_now_iso(),
                    "event": "episode_complete",
                    "episode_index": int(episode_index),
                    "reset_seed": int(episode["reset_seed"]),
                    "segment_count": int(episode["segment_count"]),
                    "reward_sum": float(episode["reward_sum"]),
                    "success_any": bool(episode["success_any"]),
                    "success_last": bool(episode["success_last"]),
                    "episode_seconds": float(time.perf_counter() - t0),
                },
            )
        summary = build_summary(output_dir=out, args=args, episodes=episodes)
        write_manifest(out / "compare_summary.json", summary)
    finally:
        env_h.close()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(args)
    write_manifest(out / "train_manifest.json", manifest)
    write_jsonl_row(
        out / "progress.jsonl",
        {
            "created_at": utc_now_iso(),
            "event": "dry_run" if args.dry_run else "run_start",
            "mode": "raw_vs_bounded_decode_compare",
            "num_episodes": int(args.num_episodes),
            "chunk_len": int(args.chunk_len),
            "train_seed_base": int(args.train_seed_base),
        },
    )
    if args.dry_run:
        print("PHASE12_RAW_VS_BOUNDED_DECODE_DRY_RUN", f"out={out}", flush=True)
        return 0
    error = _validate_args(args)
    if error is not None:
        raise RuntimeError(error)
    rc = run_compare(args, out)
    print("PHASE12_RAW_VS_BOUNDED_DECODE_DONE", f"out={out}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
