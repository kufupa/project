#!/usr/bin/env python3
"""Build MT50-style index JSON from official LeRobot `eval_info.json`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _task_pc_success(successes: list[bool]) -> float:
    if not successes:
        return float("nan")
    return float(sum(1 for s in successes if s) / len(successes)) * 100.0


def _rel_video_path(path_str: str, run_root: Path) -> str:
    p = Path(path_str)
    try:
        if p.is_absolute():
            rel = p.relative_to(run_root.resolve())
            return rel.as_posix()
    except ValueError:
        pass
    return path_str.replace("\\", "/")


def build_official_index(
    eval_info: dict[str, Any],
    *,
    run_root: Path,
    seed: int | None,
    phase: str = "MT50_Phase071",
) -> dict[str, Any]:
    """Normalize official eval_info.json into MT50_Phase071 index schema."""
    warnings: list[str] = []
    run_root = run_root.resolve()

    per_task = eval_info.get("per_task")
    tasks_out: list[dict[str, Any]] = []

    if isinstance(per_task, list) and per_task:
        for row in per_task:
            tg = row.get("task_group", "unknown")
            tid = row.get("task_id", 0)
            metrics = row.get("metrics") or {}
            successes = metrics.get("successes") or []
            raw_paths = metrics.get("video_paths") or []
            rel_paths = [_rel_video_path(s, run_root) for s in raw_paths]
            for vp in rel_paths:
                abs_p = run_root / vp if not Path(vp).is_absolute() else Path(vp)
                if not abs_p.is_file():
                    warnings.append(f"missing_video:{vp}")

            tasks_out.append(
                {
                    "task_group": tg,
                    "task_id": int(tid),
                    "n_episodes": len(successes),
                    "pc_success": _task_pc_success(list(successes)),
                    "video_paths": rel_paths,
                }
            )
        overall_src = eval_info.get("overall") or {}
        overall = {
            "pc_success": float(overall_src.get("pc_success", float("nan"))),
            "n_episodes": int(overall_src.get("n_episodes", 0)),
        }
    else:
        # Single-env / legacy shape: aggregated + optional per_episode at top level
        agg = eval_info.get("aggregated") or eval_info.get("overall") or {}
        pe = eval_info.get("per_episode")
        if isinstance(pe, list) and pe:
            successes = [bool(x.get("success")) for x in pe]
            n_ep = len(successes)
            pc = _task_pc_success(successes)
        else:
            successes = []
            n_ep = int(agg.get("n_episodes", 0))
            pc = float(agg.get("pc_success", float("nan")))

        raw_paths = eval_info.get("video_paths") or []
        rel_paths = [_rel_video_path(s, run_root) for s in raw_paths]
        for vp in rel_paths:
            abs_p = run_root / vp if not Path(vp).is_absolute() else Path(vp)
            if not abs_p.is_file():
                warnings.append(f"missing_video:{vp}")

        tasks_out.append(
            {
                "task_group": "unknown",
                "task_id": 0,
                "n_episodes": n_ep if successes else n_ep,
                "pc_success": pc,
                "video_paths": rel_paths,
            }
        )
        overall = {"pc_success": pc, "n_episodes": tasks_out[0]["n_episodes"]}

    ns = [t["n_episodes"] for t in tasks_out]
    if ns and all(n == ns[0] for n in ns):
        episodes_per_task = ns[0]
    elif ns:
        episodes_per_task = min(ns)
    else:
        episodes_per_task = 0

    out: dict[str, Any] = {
        "phase": phase,
        "source": "official_lerobot_eval",
        "run_root": str(run_root),
        "task_count": len(tasks_out),
        "episodes": episodes_per_task,
        "tasks": tasks_out,
        "overall": overall,
    }
    if seed is not None:
        out["seed"] = seed
    if warnings:
        out["warnings"] = warnings
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-info", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Run directory (for resolving relative video paths). Default: parent of eval-info.",
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--phase",
        type=str,
        default="MT50_Phase071",
        help="Label written into index JSON (e.g. MT50_Phase072).",
    )
    args = ap.parse_args()

    run_root = args.run_root or args.eval_info.parent
    data = json.loads(args.eval_info.read_text(encoding="utf-8"))
    index = build_official_index(data, run_root=run_root, seed=args.seed, phase=args.phase)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
