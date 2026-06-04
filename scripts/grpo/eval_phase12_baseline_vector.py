#!/usr/bin/env python3
"""Fast vectorized SmolVLA baseline eval (no videos, no GRPO checkpoint)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, default="push-v3")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=1000)
    parser.add_argument("--n-envs", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--chunk-len", type=int, default=5)
    parser.add_argument(
        "--rollout-execution",
        choices=("vector_sync", "vector_async"),
        default="vector_async",
    )
    args = parser.parse_args()

    from smolvla_grpo.phase11_rollout import load_bundle_for_grpo
    from smolvla_grpo.phase12_vector_eval import evaluate_loaded_policy_vectorized

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle, _action_dim = load_bundle_for_grpo(
        args.checkpoint,
        task=args.task,
        env_backend="official_lerobot",
        n_action_steps=int(args.chunk_len),
    )
    bundle.policy.eval()

    summary = evaluate_loaded_policy_vectorized(
        bundle=bundle,
        base_checkpoint=str(args.checkpoint),
        grpo_checkpoint=None,
        output_dir=output_dir,
        task=args.task,
        episodes=int(args.episodes),
        eval_seed_start=int(args.eval_seed_start),
        n_envs=int(args.n_envs),
        rollout_execution=str(args.rollout_execution),
        max_steps=int(args.max_steps),
        chunk_len=int(args.chunk_len),
    )
    print(
        "phase12_baseline_vector_eval_ok",
        f"out={output_dir}",
        f"pc_success={summary.get('pc_success')}",
        f"episodes={summary.get('episodes')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
