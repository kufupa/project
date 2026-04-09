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


def main() -> int:
    from smolvla_grpo.checkpointing import load_grpo_checkpoint
    from smolvla_pipeline.evaluator import (
        EpisodeRollout,
        _LeRobotMetaWorldBackend,
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
    args = p.parse_args()

    max_steps = _resolve_max_steps() if args.max_steps is None else int(args.max_steps)
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    ck = load_grpo_checkpoint(args.grpo_checkpoint.expanduser().resolve())
    state = ck["policy_state_dict"]

    backend = _LeRobotMetaWorldBackend(
        task=args.task,
        checkpoint=args.base_checkpoint,
        seed=int(args.eval_seed_start),
        max_steps=max_steps,
        task_text=None,
    )
    backend._bundle.policy.load_state_dict(state, strict=False)  # noqa: SLF001
    backend._bundle.policy.eval()  # noqa: SLF001

    sum_rewards: list[float] = []
    success_flags: list[bool] = []
    rows: list[dict] = []

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
        ok = any(bool(s) for s in rollout.successes)
        sum_rewards.append(sr)
        success_flags.append(ok)
        rows.append(
            {
                "episode_index": ep,
                "reset_seed": reset_seed,
                "sum_reward": sr,
                "success": ok,
                "n_steps": len(rollout.rewards),
            }
        )

    backend.close()

    pc = 100.0 * sum(1 for v in success_flags if v) / max(len(success_flags), 1)
    summary = {
        "grpo_checkpoint": str(args.grpo_checkpoint),
        "base_checkpoint": args.base_checkpoint,
        "task": args.task,
        "eval_seed_start": args.eval_seed_start,
        "episodes": args.episodes,
        "avg_sum_reward": float(mean(sum_rewards)) if sum_rewards else 0.0,
        "pc_success": float(pc),
    }
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "eval_episodes.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
