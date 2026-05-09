#!/usr/bin/env python3
"""Phase56: audit raw Push-v3 oracle action ranges."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from smolvla_grpo.phase12_diagnostics import write_phase12_episode_video
from smolvla_grpo.phase12_logging import utc_now_iso, write_jsonl_row, write_manifest
from smolvla_grpo.phase12_pixels import policy_rgb_from_obs


DIM_NAMES = ("arm_x", "arm_y", "arm_z", "gripper")
ARM_DIMS = (0, 1, 2)
GRIPPER_DIMS = (3,)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/phase56_oracle_action_audit/dry_run"))
    p.add_argument("--task", type=str, default="push-v3")
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _empty_dim_stats(dim: int, name: str) -> dict[str, Any]:
    return {
        "dim": int(dim),
        "name": name,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "max_abs": None,
        "outside_low_count": 0,
        "outside_high_count": 0,
        "outside_count": 0,
        "outside_fraction": 0.0,
    }


def _dim_stats(values: np.ndarray, dim: int, name: str) -> dict[str, Any]:
    if values.size == 0:
        return _empty_dim_stats(dim, name)
    low = values < -1.0
    high = values > 1.0
    outside = low | high
    return {
        "dim": int(dim),
        "name": name,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "max_abs": float(np.max(np.abs(values))),
        "outside_low_count": int(np.sum(low)),
        "outside_high_count": int(np.sum(high)),
        "outside_count": int(np.sum(outside)),
        "outside_fraction": float(np.mean(outside)),
    }


def _group_stats(raw_actions: np.ndarray, dims: tuple[int, ...], name: str) -> dict[str, Any]:
    if raw_actions.size == 0:
        return {
            "name": name,
            "dims": list(dims),
            "min": None,
            "max": None,
            "max_abs": None,
            "outside_step_count": 0,
            "outside_value_count": 0,
            "outside_step_fraction": 0.0,
        }
    values = raw_actions[:, list(dims)]
    outside = (values < -1.0) | (values > 1.0)
    outside_steps = np.any(outside, axis=1)
    return {
        "name": name,
        "dims": list(dims),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "max_abs": float(np.max(np.abs(values))),
        "outside_step_count": int(np.sum(outside_steps)),
        "outside_value_count": int(np.sum(outside)),
        "outside_step_fraction": float(np.mean(outside_steps)),
    }


def compute_action_range_summary(
    *,
    raw_actions: np.ndarray,
    clipped_actions: np.ndarray,
    rewards: np.ndarray,
    successes: np.ndarray,
    task: str,
    seed: int,
    max_steps: int,
    raw_actions_path: Path,
    clipped_actions_path: Path,
    steps_path: Path,
    video_path: Path | None,
) -> dict[str, Any]:
    raw = np.asarray(raw_actions, dtype=np.float32)
    clipped = np.asarray(clipped_actions, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError(f"raw_actions must have shape (T, 4), got {raw.shape}")
    if clipped.shape != raw.shape:
        raise ValueError(f"clipped_actions shape {clipped.shape} != raw shape {raw.shape}")
    outside = (raw < -1.0) | (raw > 1.0)
    dims = {
        name: _dim_stats(raw[:, dim], dim, name)
        for dim, name in enumerate(DIM_NAMES)
    }
    success_any = bool(np.any(successes)) if np.asarray(successes).size else False
    success_indices = np.flatnonzero(np.asarray(successes, dtype=np.bool_))
    success_frame = int(success_indices[0] + 2) if success_indices.size else None
    return {
        "created_at": utc_now_iso(),
        "task": str(task),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "action_count": int(raw.shape[0]),
        "raw_action_dim": 4,
        "action_dim_names": list(DIM_NAMES),
        "bounds": {"low": [-1.0] * 4, "high": [1.0] * 4},
        "action_source": "lerobot_metaworld_adapter_expert_raw",
        "env_dispatched_source": "clipped_oracle_action",
        "any_outside_minus1_1": bool(np.any(outside)),
        "outside_value_count": int(np.sum(outside)),
        "outside_step_count": int(np.sum(np.any(outside, axis=1))) if raw.shape[0] else 0,
        "dims": dims,
        "groups": {
            "arm": _group_stats(raw, ARM_DIMS, "arm"),
            "gripper": _group_stats(raw, GRIPPER_DIMS, "gripper"),
        },
        "success_any": success_any,
        "success_frame_1based": success_frame,
        "reward_sum": float(np.sum(rewards)) if np.asarray(rewards).size else 0.0,
        "raw_actions_path": str(raw_actions_path),
        "clipped_actions_path": str(clipped_actions_path),
        "steps_path": str(steps_path),
        "video_path": None if video_path is None else str(video_path),
    }


def _write_steps_csv(
    path: Path,
    *,
    raw_actions: np.ndarray,
    clipped_actions: np.ndarray,
    rewards: np.ndarray,
    successes: np.ndarray,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "raw_arm_x",
                "raw_arm_y",
                "raw_arm_z",
                "raw_gripper",
                "clipped_arm_x",
                "clipped_arm_y",
                "clipped_arm_z",
                "clipped_gripper",
                "outside_any",
                "reward",
                "success",
            ]
        )
        for idx, (raw, clipped) in enumerate(zip(raw_actions, clipped_actions, strict=True)):
            outside = bool(np.any((raw < -1.0) | (raw > 1.0)))
            writer.writerow(
                [
                    int(idx),
                    *[float(x) for x in raw],
                    *[float(x) for x in clipped],
                    outside,
                    float(rewards[idx]) if idx < len(rewards) else 0.0,
                    bool(successes[idx]) if idx < len(successes) else False,
                ]
            )
    return path


def _write_steps_jsonl(
    path: Path,
    *,
    raw_actions: np.ndarray,
    clipped_actions: np.ndarray,
    rewards: np.ndarray,
    successes: np.ndarray,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    for idx, (raw, clipped) in enumerate(zip(raw_actions, clipped_actions, strict=True)):
        write_jsonl_row(
            path,
            {
                "step": int(idx),
                "raw_action": [float(x) for x in raw],
                "clipped_action": [float(x) for x in clipped],
                "outside_mask": [bool(x) for x in ((raw < -1.0) | (raw > 1.0))],
                "outside_any": bool(np.any((raw < -1.0) | (raw > 1.0))),
                "reward": float(rewards[idx]) if idx < len(rewards) else 0.0,
                "success": bool(successes[idx]) if idx < len(successes) else False,
            },
        )
    return path


def run_audit(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout, resolve_lerobot_horizon

    env_h = OfficialLeRobotMetaWorldGRPORollout(task=args.task, n_envs=1, enable_expert_oracle=True)
    try:
        max_steps = min(int(args.max_steps), int(resolve_lerobot_horizon(env_h, int(args.max_steps))))
        obs = env_h.reset(int(args.seed))
        raw_rows: list[np.ndarray] = []
        clipped_rows: list[np.ndarray] = []
        rewards: list[float] = []
        successes: list[bool] = []
        frames: list[np.ndarray] = [policy_rgb_from_obs(obs)] if args.save_video else []
        for _step_idx in range(max_steps):
            raw = np.asarray(env_h.expert_action(), dtype=np.float32).reshape(4)
            clipped = np.clip(raw, -1.0, 1.0).astype(np.float32, copy=False)
            step = env_h.step(clipped.reshape(1, -1))
            raw_rows.append(raw.copy())
            clipped_rows.append(clipped.copy())
            rewards.append(float(step.reward))
            successes.append(bool(step.success))
            if args.save_video:
                frames.append(policy_rgb_from_obs(step.observation))
            if bool(step.success or step.terminated or step.truncated):
                break
        raw_actions = np.asarray(raw_rows, dtype=np.float32).reshape(-1, 4)
        clipped_actions = np.asarray(clipped_rows, dtype=np.float32).reshape(-1, 4)
        rewards_np = np.asarray(rewards, dtype=np.float64)
        successes_np = np.asarray(successes, dtype=np.bool_)

        raw_path = out / "oracle_actions_raw.npy"
        clipped_path = out / "oracle_actions_clipped.npy"
        steps_jsonl_path = out / "oracle_action_steps.jsonl"
        steps_csv_path = out / "oracle_action_steps.csv"
        np.save(raw_path, raw_actions)
        np.save(clipped_path, clipped_actions)
        _write_steps_jsonl(steps_jsonl_path, raw_actions=raw_actions, clipped_actions=clipped_actions, rewards=rewards_np, successes=successes_np)
        _write_steps_csv(steps_csv_path, raw_actions=raw_actions, clipped_actions=clipped_actions, rewards=rewards_np, successes=successes_np)
        video_path: Path | None = None
        if args.save_video and frames:
            video_path = write_phase12_episode_video(
                video_path=out / "oracle_action_rollout.mp4",
                frames=frames,
                rewards=rewards,
                successes=successes,
                fps=int(args.fps),
                overlay_mode="cumulative_reward",
            )
        summary = compute_action_range_summary(
            raw_actions=raw_actions,
            clipped_actions=clipped_actions,
            rewards=rewards_np,
            successes=successes_np,
            task=str(args.task),
            seed=int(args.seed),
            max_steps=max_steps,
            raw_actions_path=raw_path,
            clipped_actions_path=clipped_path,
            steps_path=steps_jsonl_path,
            video_path=video_path,
        )
        summary["steps_csv_path"] = str(steps_csv_path)
        write_manifest(out / "oracle_action_range_summary.json", summary)
        return summary
    finally:
        env_h.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": utc_now_iso(),
        "mode": "phase56_oracle_action_audit",
        "task": str(args.task),
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "save_video": bool(args.save_video),
    }
    write_manifest(out / "phase56_manifest.json", manifest)
    if args.dry_run:
        print("PHASE56_ORACLE_ACTION_AUDIT_DRY_RUN", f"out={out}", flush=True)
        return 0
    summary = run_audit(args, out)
    print(
        "PHASE56_ORACLE_ACTION_AUDIT_DONE",
        f"out={out}",
        f"any_outside={summary['any_outside_minus1_1']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
