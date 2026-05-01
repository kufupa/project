#!/usr/bin/env python3
"""Short GPU/CPU rollout group smoke for Phase11 (MetaWorld + SmolVLA policy)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    import torch

    from smolvla_grpo.phase11_rollout import collect_rollout_group, load_bundle_for_grpo
    from smolvla_pipeline.evaluator import _resolve_task_text

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--task", type=str, default="push-v3")
    p.add_argument("--group-size", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=2000, help="reset_seed base for the group")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle, action_dim = load_bundle_for_grpo(args.checkpoint, task=args.task)
    task_text = _resolve_task_text(args.task, override=None)
    policy = bundle.policy.to(device).eval()

    rollouts = collect_rollout_group(
        bundle=bundle,
        policy_old=policy,
        task=args.task,
        task_text=task_text,
        reset_seed=int(args.seed),
        episode_index=0,
        max_steps=int(args.max_steps),
        group_size=int(args.group_size),
        action_dim=action_dim,
        device=device,
    )

    if len(rollouts) != int(args.group_size):
        raise SystemExit(f"expected {args.group_size} trajectories, got {len(rollouts)}")

    for tr in rollouts:
        if len(tr.rewards) <= 0:
            raise SystemExit("each trajectory must have rewards length > 0")
        if len(tr.log_probs) != len(tr.rewards):
            raise SystemExit(
                f"log_probs len {len(tr.log_probs)} != rewards len {len(tr.rewards)}"
            )
        if len(tr.unsquashed_actions) != len(tr.rewards):
            raise SystemExit("unsquashed_actions length mismatch")

    print(
        "phase11_rollout_smoke_ok",
        f"group_size={len(rollouts)}",
        f"steps={[len(t.rewards) for t in rollouts]}",
        f"device={device}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
