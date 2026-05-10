#!/usr/bin/env python3
"""Eval SmolVLA Push-v3 checkpoints produced by Phase11 GRPO (.pt with policy_state_dict)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


def _move_proc_to_device(value, device):
    if hasattr(value, "to") and hasattr(value, "detach"):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_proc_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_proc_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_proc_to_device(v, device) for v in value)
    return value


def main() -> int:
    from smolvla_grpo.checkpointing import load_grpo_checkpoint
    from smolvla_pipeline.evaluator import (
        EpisodeRollout,
        _LeRobotMetaWorldBackend,
        _coerce_exec_action,
        _resolve_max_steps,
        write_episode_artifacts,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint", type=str, required=True, help="HF checkpoint dir used during GRPO")
    p.add_argument("--grpo-checkpoint", type=Path, required=True, help="Path to update_XXXX.pt")
    p.add_argument("--task", type=str, default="push-v3")
    p.add_argument("--eval-seed-start", type=int, default=1000)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--env-backend", choices=("custom", "official_lerobot"), default="custom")
    p.add_argument("--save-official-eval-info", action="store_true")
    args = p.parse_args()

    max_steps = _resolve_max_steps() if args.max_steps is None else int(args.max_steps)
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    ck = load_grpo_checkpoint(args.grpo_checkpoint.expanduser().resolve())
    state = ck["policy_state_dict"]

    sum_rewards: list[float] = []
    max_rewards: list[float] = []
    success_flags: list[bool] = []
    rows: list[dict] = []

    if args.env_backend == "official_lerobot":
        import numpy as np
        import torch

        from smolvla_grpo.lerobot_metaworld_adapter import (
            OfficialLeRobotMetaWorldGRPORollout,
            resolve_lerobot_horizon,
        )

        from smolvla_grpo.phase11_rollout import load_bundle_for_grpo

        bundle, _action_dim = load_bundle_for_grpo(
            args.base_checkpoint,
            task=args.task,
            env_backend="official_lerobot",
        )
        bundle.policy.load_state_dict(state, strict=False)
        bundle.policy.eval()
        env_h = OfficialLeRobotMetaWorldGRPORollout(task=args.task)
        resolved_steps = resolve_lerobot_horizon(env_h, max_steps)
        try:
            for ep in range(args.episodes):
                reset_seed = int(args.eval_seed_start) + ep
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
                for _step in range(resolved_steps):
                    proc = _move_proc_to_device(env_h.build_proc(obs, bundle=bundle), bundle.device)
                    with torch.no_grad():
                        action = bundle.policy.select_action(proc)
                    post = bundle.postprocessor(action)
                    exec_action = _coerce_exec_action(post, action_dim=env_h.action_dim, np_module=np)
                    action_batch = exec_action.reshape(1, -1).astype(np.float32)
                    env_step = env_h.step(action_batch)
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
                    truncated=episode_truncated or (len(rewards) >= resolved_steps and not any(successes)),
                )
                ep_dir = out / "episodes" / f"episode_{ep:04d}"
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
                        "env_backend": args.env_backend,
                    }
                )
        finally:
            env_h.close()
    else:
        backend = _LeRobotMetaWorldBackend(
            task=args.task,
            checkpoint=args.base_checkpoint,
            seed=int(args.eval_seed_start),
            max_steps=max_steps,
            task_text=None,
        )
        backend._bundle.policy.load_state_dict(state, strict=False)  # noqa: SLF001
        backend._bundle.policy.eval()  # noqa: SLF001

        for ep in range(args.episodes):
            reset_seed = int(args.eval_seed_start) + ep
            rollout: EpisodeRollout = backend.rollout_episode(episode_index=ep, reset_seed=reset_seed)
            ep_dir = out / "episodes" / f"episode_{ep:04d}"
            write_episode_artifacts(
                episode_dir=ep_dir,
                actions=rollout.actions,
                rewards=rollout.rewards,
                successes=rollout.successes,
                overlay_mode="cumulative_reward",
            )
            sr = float(sum(rollout.rewards))
            mr = float(max(rollout.rewards)) if rollout.rewards else 0.0
            ok = any(bool(s) for s in rollout.successes)
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
                    "n_steps": len(rollout.rewards),
                    "env_backend": args.env_backend,
                }
            )

        backend.close()

    pc = 100.0 * sum(1 for v in success_flags if v) / max(len(success_flags), 1)
    summary = {
        "grpo_checkpoint": str(args.grpo_checkpoint),
        "base_checkpoint": args.base_checkpoint,
        "task": args.task,
        "env_backend": args.env_backend,
        "eval_seed_start": args.eval_seed_start,
        "episodes": args.episodes,
        "avg_sum_reward": float(mean(sum_rewards)) if sum_rewards else 0.0,
        "avg_max_reward": float(mean(max_rewards)) if max_rewards else 0.0,
        "pc_success": float(pc),
    }
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "eval_episodes.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    if args.save_official_eval_info:
        eval_info = {
            "per_task": [
                {
                    "task_group": args.task,
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
                args.task: {
                    "avg_sum_reward": float(mean(sum_rewards)) if sum_rewards else 0.0,
                    "avg_max_reward": float(mean(max_rewards)) if max_rewards else 0.0,
                    "pc_success": float(pc),
                    "n_episodes": len(sum_rewards),
                    "video_paths": [],
                }
            },
            "overall": {
                "avg_sum_reward": float(mean(sum_rewards)) if sum_rewards else 0.0,
                "avg_max_reward": float(mean(max_rewards)) if max_rewards else 0.0,
                "pc_success": float(pc),
                "n_episodes": len(sum_rewards),
                "video_paths": [],
            },
        }
        (out / "eval_info.json").write_text(json.dumps(eval_info, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
