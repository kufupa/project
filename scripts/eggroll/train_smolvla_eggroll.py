#!/usr/bin/env python3
"""Train SmolVLA on MetaWorld with EGGROLL."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from smolvla_grpo.eggroll_trainer import EggrollTrainer, EggrollTrainerConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default="push-v3")
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--population-batch-size", type=int, default=4)
    parser.add_argument("--rank", "--rank-r", dest="rank", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--baseline-type", choices=("none", "mean", "median"), default="mean")
    parser.add_argument("--fitness-shaping", choices=("centered", "rank"), default="centered")
    parser.add_argument("--antithetic", dest="antithetic", action="store_true", default=True)
    parser.add_argument("--no-antithetic", dest="antithetic", action="store_false")
    parser.add_argument("--episodes-per-member", type=int, default=1)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--action-chunk-size", "--chunk-len", dest="action_chunk_size", type=int, default=5)
    parser.add_argument("--rollout-execution", choices=("vector_sync", "vector_async"), default="vector_async")
    parser.add_argument("--async-start-method", default="forkserver")
    parser.add_argument("--train-scope", choices=("action_expert", "action_head", "all_linear"), default="action_expert")
    parser.add_argument("--train-seed-base", type=int, default=2000)
    parser.add_argument("--noise-seed", type=int, default=17)
    parser.add_argument("--flow-noise-seed", type=int, default=23)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--video-every", type=int, default=10)
    parser.add_argument("--video-member-id", type=int, default=0)
    parser.add_argument("--write-oracle-video", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--abort-update-norm", type=float, default=0.05)
    parser.add_argument("--seed-mode", choices=("member_offset", "shared_per_iteration"), default="member_offset")
    parser.add_argument("--checkpoint-sync-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = EggrollTrainerConfig(**vars(args))
    result = EggrollTrainer(cfg).run()
    print(
        "EGGROLL_TRAIN_DONE",
        f"task={cfg.task}",
        f"out={cfg.output_dir}",
        f"iteration={result.get('iteration')}",
        f"checkpoint={result.get('checkpoint_path')}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
