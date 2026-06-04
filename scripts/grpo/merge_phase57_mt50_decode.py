#!/usr/bin/env python3
"""Merge Phase57 MT50 shard summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from smolvla_grpo.phase12_logging import utc_now_iso, write_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--expected-tasks", type=int, default=50)
    p.add_argument("--expected-episodes", type=int, default=1250)
    return p.parse_args(argv)


def discover_task_summaries(parent: Path) -> list[Path]:
    return sorted(Path(parent).glob("shard_*_*tasks/*/task_summary.json"))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_merged_summary(
    *,
    parent: Path,
    task_summaries: list[dict[str, Any]],
    expected_tasks: int,
    expected_episodes: int,
) -> dict[str, Any]:
    tasks = sorted(str(item["task"]) for item in task_summaries)
    episodes_found = int(sum(int(item.get("episodes_completed", item.get("episodes", 0))) for item in task_summaries))
    successes = 0
    total_eps = 0
    macro_success = []
    raw_l2_weighted_sum = 0.0
    bounded_l2_weighted_sum = 0.0
    raw_l2_weight = 0
    bounded_l2_weight = 0
    raw_wins = bounded_wins = ties = total_cols = 0
    for item in task_summaries:
        eps_rows = item.get("episodes_rows", [])
        total_eps += len(eps_rows)
        successes += sum(1 for row in eps_rows if bool(row.get("success", False)))
        macro_success.append(float(item.get("pc_success", 0.0)))
        cols = int(item.get("metric_column_count", 0) or 0)
        if item.get("mean_raw_combined_l2") is not None and cols > 0:
            raw_l2_weighted_sum += float(item["mean_raw_combined_l2"]) * cols
            raw_l2_weight += cols
        if item.get("mean_bounded_combined_l2") is not None and cols > 0:
            bounded_l2_weighted_sum += float(item["mean_bounded_combined_l2"]) * cols
            bounded_l2_weight += cols
        total_cols += cols
        raw_wins += int(round(float(item.get("raw_win_fraction", 0.0)) * cols))
        bounded_wins += int(round(float(item.get("bounded_win_fraction", 0.0)) * cols))
        ties += int(round(float(item.get("tie_fraction", 0.0)) * cols))
    return {
        "created_at": utc_now_iso(),
        "parent": str(Path(parent)),
        "tasks_found": len(tasks),
        "tasks_expected": int(expected_tasks),
        "missing_task_count": max(0, int(expected_tasks) - len(tasks)),
        "tasks": tasks,
        "episodes_found": episodes_found,
        "episodes_expected": int(expected_episodes),
        "micro_pc_success": 100.0 * successes / max(total_eps, 1),
        "macro_pc_success": float(np.mean(macro_success)) if macro_success else 0.0,
        "mean_raw_combined_l2": raw_l2_weighted_sum / raw_l2_weight if raw_l2_weight else None,
        "mean_bounded_combined_l2": bounded_l2_weighted_sum / bounded_l2_weight if bounded_l2_weight else None,
        "raw_win_fraction": raw_wins / max(total_cols, 1),
        "bounded_win_fraction": bounded_wins / max(total_cols, 1),
        "tie_fraction": ties / max(total_cols, 1),
        "metric_column_count": int(total_cols),
        "task_summaries": task_summaries,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    parent = args.parent.expanduser().resolve()
    paths = discover_task_summaries(parent)
    summaries = [_load(path) for path in paths]
    merged = build_merged_summary(
        parent=parent,
        task_summaries=summaries,
        expected_tasks=int(args.expected_tasks),
        expected_episodes=int(args.expected_episodes),
    )
    write_manifest(parent / "phase57_mt50_summary.json", merged)
    print(
        "PHASE57_MERGE_DONE",
        f"parent={parent}",
        f"tasks={merged['tasks_found']}/{merged['tasks_expected']}",
        f"episodes={merged['episodes_found']}/{merged['episodes_expected']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
