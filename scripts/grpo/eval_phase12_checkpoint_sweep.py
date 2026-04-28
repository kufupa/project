#!/usr/bin/env python3
"""Run official-backend eval sweep over Phase12 checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

_REPO = Path(__file__).resolve().parents[2]


def _move_proc_to_device(value: Any, device: Any) -> Any:
    if hasattr(value, "to") and hasattr(value, "detach"):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_proc_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_proc_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_proc_to_device(v, device) for v in value)
    return value


def _write_eval_info(
    *,
    base_checkpoint: str,
    grpo_checkpoint: Path | None,
    output_dir: Path,
    task: str,
    episodes: int,
    eval_seed_start: int,
    sum_rewards: list[float],
    max_rewards: list[float],
    success_flags: list[bool],
) -> dict[str, Any]:
    pc = 100.0 * sum(1 for v in success_flags if v) / max(len(success_flags), 1)
    summary = {
        "grpo_checkpoint": str(grpo_checkpoint) if grpo_checkpoint is not None else "base_checkpoint",
        "base_checkpoint": base_checkpoint,
        "task": task,
        "env_backend": "official_lerobot",
        "eval_seed_start": int(eval_seed_start),
        "episodes": int(episodes),
        "avg_sum_reward": float(mean(sum_rewards)) if sum_rewards else 0.0,
        "avg_max_reward": float(mean(max_rewards)) if max_rewards else 0.0,
        "pc_success": float(pc),
    }
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    eval_info = {
        "per_task": [
            {
                "task_group": task,
                "task_id": 0,
                "metrics": {
                    "sum_rewards": sum_rewards,
                    "max_rewards": max_rewards,
                    "successes": success_flags,
                    "video_paths": [],
                },
            }
        ],
        "per_group": {
            task: {
                "avg_sum_reward": summary["avg_sum_reward"],
                "avg_max_reward": summary["avg_max_reward"],
                "pc_success": summary["pc_success"],
                "n_episodes": len(sum_rewards),
                "video_paths": [],
            }
        },
        "overall": {
            "avg_sum_reward": summary["avg_sum_reward"],
            "avg_max_reward": summary["avg_max_reward"],
            "pc_success": summary["pc_success"],
            "n_episodes": len(sum_rewards),
            "video_paths": [],
        },
    }
    (output_dir / "eval_info.json").write_text(json.dumps(eval_info, indent=2), encoding="utf-8")
    return summary


def _eval_loaded_policy(
    *,
    bundle: Any,
    env_h: Any,
    task: str,
    base_checkpoint: str,
    grpo_checkpoint: Path | None,
    output_dir: Path,
    episodes: int,
    eval_seed_start: int,
    max_steps: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from smolvla_pipeline.evaluator import EpisodeRollout, _coerce_exec_action, write_episode_artifacts

    sum_rewards: list[float] = []
    max_rewards: list[float] = []
    success_flags: list[bool] = []
    rows: list[dict[str, Any]] = []
    for ep in range(int(episodes)):
        reset_seed = int(eval_seed_start) + ep
        obs = env_h.reset(reset_seed)
        policy_reset = getattr(bundle.policy, "reset", None)
        if callable(policy_reset):
            try:
                policy_reset()
            except Exception:
                pass
        rewards: list[float] = []
        successes: list[bool] = []
        actions: list[list[float]] = []
        episode_terminated = False
        episode_truncated = False
        for _step in range(int(max_steps)):
            proc = _move_proc_to_device(env_h.build_proc(obs, bundle=bundle), bundle.device)
            with torch.no_grad():
                action = bundle.policy.select_action(proc)
            post = bundle.postprocessor(action)
            exec_action = _coerce_exec_action(post, action_dim=env_h.action_dim, np_module=np)
            env_step = env_h.step(exec_action.reshape(1, -1).astype(np.float32))
            obs = env_step.observation
            rewards.append(float(env_step.reward))
            successes.append(bool(env_step.success))
            actions.append(exec_action.reshape(-1).tolist())
            if env_step.success or env_step.terminated or env_step.truncated:
                episode_terminated = bool(env_step.terminated)
                episode_truncated = bool(env_step.truncated)
                break
        rollout = EpisodeRollout(
            actions=actions,
            rewards=rewards,
            successes=successes,
            frames=[],
            terminated=episode_terminated,
            truncated=episode_truncated or (len(rewards) >= int(max_steps) and not any(successes)),
        )
        ep_dir = output_dir / "episodes" / f"episode_{ep:04d}"
        write_episode_artifacts(
            episode_dir=ep_dir,
            actions=rollout.actions,
            rewards=rollout.rewards,
            successes=rollout.successes,
            overlay_mode="cumulative_reward",
        )
        sr = float(sum(rewards))
        mr = float(max(rewards)) if rewards else 0.0
        ok = any(bool(s) for s in successes)
        sum_rewards.append(sr)
        max_rewards.append(mr)
        success_flags.append(ok)
        rows.append(
            {
                "episode_index": ep,
                "reset_seed": reset_seed,
                "sum_reward": sr,
                "max_reward": mr,
                "success": ok,
                "n_steps": len(rewards),
                "env_backend": "official_lerobot",
            }
        )
    (output_dir / "eval_episodes.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return _write_eval_info(
        base_checkpoint=base_checkpoint,
        grpo_checkpoint=grpo_checkpoint,
        output_dir=output_dir,
        task=task,
        episodes=episodes,
        eval_seed_start=eval_seed_start,
        sum_rewards=sum_rewards,
        max_rewards=max_rewards,
        success_flags=success_flags,
    )


def _checkpoint_path(run_dir: Path, update: int) -> Path:
    return run_dir / "checkpoints" / f"update_{int(update):04d}.pt"


def run_sweep(
    *,
    base_checkpoint: str,
    run_dir: Path,
    task: str,
    episodes: int,
    eval_seed_start: int,
    sweep_name: str,
    min_update: int,
    max_update: int,
    stride: int,
) -> dict[str, Any]:
    if int(stride) <= 0:
        raise ValueError("stride must be positive")
    import torch

    from smolvla_grpo.checkpointing import load_grpo_checkpoint
    from smolvla_grpo.lerobot_metaworld_adapter import (
        OfficialLeRobotMetaWorldGRPORollout,
        resolve_lerobot_horizon,
    )
    from smolvla_grpo.phase11_rollout import load_bundle_for_grpo

    sweep_dir = run_dir / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    updates = list(range(int(min_update), int(max_update) + 1, int(stride)))
    for update in updates:
        ckpt = None if update == 0 else _checkpoint_path(run_dir, update)
        if ckpt is not None and not ckpt.is_file():
            missing.append(str(ckpt))
    if missing:
        raise FileNotFoundError("missing requested checkpoints:\n" + "\n".join(missing))
    bundle, _action_dim = load_bundle_for_grpo(
        base_checkpoint,
        task=task,
        env_backend="official_lerobot",
    )
    env_h = OfficialLeRobotMetaWorldGRPORollout(task=task)
    max_steps = resolve_lerobot_horizon(env_h, 0)
    try:
        for update in updates:
            ckpt = None if update == 0 else _checkpoint_path(run_dir, update)
            if ckpt is not None:
                loaded = load_grpo_checkpoint(ckpt, map_location="cpu")
                bundle.policy.load_state_dict(loaded["policy_state_dict"], strict=False)
                del loaded
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            bundle.policy.eval()
            out_dir = sweep_dir / f"update_{update:04d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            summary = _eval_loaded_policy(
                bundle=bundle,
                env_h=env_h,
                task=task,
                base_checkpoint=base_checkpoint,
                grpo_checkpoint=ckpt,
                output_dir=out_dir,
                episodes=episodes,
                eval_seed_start=eval_seed_start,
                max_steps=max_steps,
            )
            print(json.dumps(summary, indent=2), flush=True)
            rows.append(
                {
                    "update": int(update),
                    "checkpoint": str(ckpt) if ckpt is not None else "base_checkpoint",
                    "pc_success": float(summary.get("pc_success", 0.0)),
                    "avg_sum_reward": float(summary.get("avg_sum_reward", 0.0)),
                    "avg_max_reward": float(summary.get("avg_max_reward", 0.0)),
                    "episodes": int(summary.get("episodes", episodes)),
                    "eval_summary_path": str(out_dir / "eval_summary.json"),
                }
            )
    finally:
        env_h.close()
    result = {
        "task": task,
        "episodes": int(episodes),
        "eval_seed_start": int(eval_seed_start),
        "sweep_name": sweep_name,
        "min_update": int(min_update),
        "max_update": int(max_update),
        "stride": int(stride),
        "rows": rows,
    }
    out_path = sweep_dir / "eval_sweep_summary.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["output_path"] = str(out_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, default="push-v3")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval-seed-start", type=int, default=1000)
    parser.add_argument("--sweep-name", type=str, default="eval_sweep_every10")
    parser.add_argument("--min-update", type=int, default=0)
    parser.add_argument("--max-update", type=int, default=300)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    result = run_sweep(
        base_checkpoint=args.base_checkpoint,
        run_dir=args.run_dir.expanduser().resolve(),
        task=args.task,
        episodes=args.episodes,
        eval_seed_start=args.eval_seed_start,
        sweep_name=args.sweep_name,
        min_update=args.min_update,
        max_update=args.max_update,
        stride=args.stride,
    )
    print(
        "phase12_eval_sweep_ok",
        f"rows={len(result['rows'])}",
        f"task={result['task']}",
        f"out={result['output_path']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
