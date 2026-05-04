#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, PROJECT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.smolvla_pipeline.evaluator import run_smolvla_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SmolVLA evaluator for Meta-World tasks.")
    parser.add_argument("--task", default="push-v3")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video", default="false")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--overlay-mode",
        default="cumulative_reward",
        choices=["cumulative_reward", "reward", "reward_delta"],
    )
    parser.add_argument(
        "--save-frames",
        default="false",
        choices=["true", "false"],
        help="Write frames/episode_XXXX/frame_*.png (default: false).",
    )
    parser.add_argument(
        "--save-actions",
        default="false",
        choices=["true", "false"],
        help="Write episodes/episode_XXXX/actions.jsonl (default: false).",
    )
    parser.add_argument(
        "--task-text",
        default=None,
        help="Override language instruction passed to SmolVLA (default: LeRobot TASK_DESCRIPTIONS).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_smolvla_eval(
        task=args.task,
        episodes=args.episodes,
        seed=args.seed,
        checkpoint=args.checkpoint,
        output_dir=Path(args.output_dir),
        video=args.video,
        fps=args.fps,
        overlay_mode=args.overlay_mode,
        max_steps=args.max_steps,
        save_frames=args.save_frames == "true",
        save_actions=args.save_actions == "true",
        task_text=args.task_text,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
