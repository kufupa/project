#!/usr/bin/env python3
"""Analyze Phase12 pure-WM collapse signatures from run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean_segment_reward(row: dict[str, Any]) -> float | None:
    values: list[float] = []
    for segment in row.get("segment_candidate_rewards", []) or []:
        values.extend(float(value) for value in segment)
    if not values:
        return None
    return float(sum(values) / len(values))


def _find_eval_summary(run_dir: Path) -> Path | None:
    summaries = sorted(run_dir.glob("**/eval_sweep_summary.json"))
    if not summaries:
        return None

    def rank(path: Path) -> tuple[int, int, str]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            episodes = int(payload.get("episodes", 0))
        except Exception:
            episodes = 0
        name = str(path)
        return (1 if "eval100" in name else 0, episodes, name)

    return max(summaries, key=rank)


def analyze_run(run_dir: Path | str) -> dict[str, Any]:
    run = Path(run_dir).expanduser().resolve()
    progress_rows = [row for row in _read_jsonl(run / "progress.jsonl") if row.get("event") == "update_complete"]
    progress_by_update: dict[int, dict[str, Any]] = {
        int(row.get("update_index", -1)) + 1: row for row in progress_rows if int(row.get("update_index", -1)) >= 0
    }

    eval_summary_path = _find_eval_summary(run)
    eval_rows: list[dict[str, Any]] = []
    eval_payload: dict[str, Any] = {}
    if eval_summary_path is not None:
        eval_payload = json.loads(eval_summary_path.read_text(encoding="utf-8"))
        eval_rows = [dict(row) for row in eval_payload.get("rows", [])]

    if eval_rows:
        best = max(eval_rows, key=lambda row: (float(row.get("pc_success", 0.0)), int(row.get("update", -1))))
        last = eval_rows[-1]
        best_update = int(best.get("update", -1))
        best_success = float(best.get("pc_success", 0.0))
        last_update = int(last.get("update", -1))
        last_success = float(last.get("pc_success", 0.0))
    else:
        best_update = -1
        best_success = 0.0
        last_update = -1
        last_success = 0.0

    best_progress = progress_by_update.get(best_update, {})
    later_progress = [
        row for update, row in sorted(progress_by_update.items()) if best_update >= 0 and update > best_update
    ]
    best_wm_reward = _mean_segment_reward(best_progress)
    later_rewards = [_mean_segment_reward(row) for row in later_progress]
    later_rewards = [float(value) for value in later_rewards if value is not None]
    latest_later_reward = later_rewards[-1] if later_rewards else None

    best_clip = float(best_progress.get("action_clip_fraction", 0.0)) if best_progress else 0.0
    later_clips = [float(row.get("action_clip_fraction", 0.0)) for row in later_progress]
    max_later_clip = max(later_clips, default=0.0)
    collapse_pp = float(best_success - last_success)

    report = {
        "run_dir": str(run),
        "eval_summary_path": "" if eval_summary_path is None else str(eval_summary_path),
        "eval_episodes": int(eval_payload.get("episodes", 0)) if eval_payload else 0,
        "best_update": best_update,
        "best_pc_success": best_success,
        "last_update": last_update,
        "last_pc_success": last_success,
        "collapse_pp": collapse_pp,
        "best_wm_reward_mean": best_wm_reward,
        "latest_after_best_wm_reward_mean": latest_later_reward,
        "best_action_clip_fraction": best_clip,
        "max_after_best_action_clip_fraction": max_later_clip,
        "wm_reward_increased_while_eval_dropped": bool(
            best_wm_reward is not None
            and latest_later_reward is not None
            and latest_later_reward > best_wm_reward
            and last_success < best_success
        ),
        "clip_fraction_grew_after_best": bool(max_later_clip > best_clip),
        "unstable_clip_fraction": bool(max_later_clip > 0.5 or sum(1 for value in later_clips if value > 0.35) >= 2),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = analyze_run(args.run_dir)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
