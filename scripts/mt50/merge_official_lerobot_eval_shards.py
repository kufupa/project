#!/usr/bin/env python3
"""Merge official LeRobot eval_info.json files from parallel shard runs into one artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _agg_from_list(xs: list[float]) -> float:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def merge_eval_info_paths(
    paths: list[Path],
    *,
    expected_tasks: int | None = 50,
) -> dict[str, Any]:
    per_task: list[dict[str, Any]] = []
    per_group: dict[str, Any] = {}
    all_sum: list[float] = []
    all_max: list[float] = []
    all_succ: list[bool] = []
    all_vid: list[str] = []
    eval_s_total = 0.0

    for path in paths:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing eval_info: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get("per_task", []):
            per_task.append(row)
            m = row.get("metrics") or {}
            all_sum.extend(m.get("sum_rewards") or [])
            all_max.extend(m.get("max_rewards") or [])
            succ = m.get("successes") or []
            all_succ.extend(succ)
            all_vid.extend(m.get("video_paths") or [])
        for gk, gv in (data.get("per_group") or {}).items():
            if gk in per_group:
                raise ValueError(f"duplicate task group across inputs: {gk}")
            per_group[gk] = gv
        eval_s_total += float((data.get("overall") or {}).get("eval_s") or 0.0)

    n_eps = len(all_sum)
    overall: dict[str, Any] = {
        "avg_sum_reward": _agg_from_list(all_sum),
        "avg_max_reward": _agg_from_list(all_max),
        "pc_success": float(sum(1 for s in all_succ if s) / len(all_succ) * 100.0)
        if all_succ
        else float("nan"),
        "n_episodes": n_eps,
        "eval_s": eval_s_total,
        "eval_ep_s": eval_s_total / max(1, n_eps),
        "video_paths": list(all_vid),
    }

    out: dict[str, Any] = {
        "per_task": per_task,
        "per_group": per_group,
        "overall": overall,
    }
    if expected_tasks is not None and len(per_task) != expected_tasks:
        out["warnings"] = [
            f"expected {expected_tasks} per_task rows, got {len(per_task)} (rerun missing shards?)"
        ]
    return out


def merge_shard_eval_infos(
    parent: Path,
    shard_names: list[str],
    *,
    expected_tasks: int | None = 50,
) -> dict[str, Any]:
    parent = parent.resolve()
    paths = [parent / name / "eval_info.json" for name in shard_names]
    return merge_eval_info_paths(paths, expected_tasks=expected_tasks)


def discover_phase27_per_task_eval_infos(parent: Path) -> list[Path]:
    """Paths like parent/shard_0_13tasks/<task>/eval_info.json (Phase27 4-GPU PBS layout)."""
    parent = parent.resolve()
    out: list[Path] = []
    for shard in sorted(parent.glob("shard_*_*tasks")):
        if not shard.is_dir():
            continue
        for task_dir in sorted(shard.iterdir()):
            ei = task_dir / "eval_info.json"
            if task_dir.is_dir() and ei.is_file():
                out.append(ei)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parent",
        type=Path,
        required=True,
        help="e.g. artifacts/MT50_Phase072_official_lerobot_10ep",
    )
    ap.add_argument(
        "--shards",
        nargs="+",
        default=["shard0", "shard1", "shard2"],
        help="Subdirectory names under parent containing eval_info.json",
    )
    ap.add_argument(
        "--phase27-per-task-dirs",
        action="store_true",
        help="Merge shard_*_*tasks/<task>/eval_info.json under --parent (Phase27 layout).",
    )
    ap.add_argument(
        "--merged-eval-info",
        type=Path,
        default=None,
        help="Write merged JSON here (default: parent/eval_info.json)",
    )
    ap.add_argument(
        "--expected-tasks",
        type=int,
        default=50,
        help="If set, warn in JSON when per_task count differs (use -1 to disable)",
    )
    args = ap.parse_args()

    exp = args.expected_tasks if args.expected_tasks >= 0 else None
    if args.phase27_per_task_dirs:
        paths = discover_phase27_per_task_eval_infos(args.parent)
        if not paths:
            raise SystemExit(
                f"no per-task eval_info.json under {args.parent.resolve()} "
                "(expected shard_*_*tasks/<task>/eval_info.json)"
            )
        merged = merge_eval_info_paths(paths, expected_tasks=exp)
    else:
        merged = merge_shard_eval_infos(args.parent, list(args.shards), expected_tasks=exp)
    out_path = args.merged_eval_info or (args.parent.resolve() / "eval_info.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path} per_task={len(merged['per_task'])}")


if __name__ == "__main__":
    main()
