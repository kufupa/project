#!/usr/bin/env python3
"""Summarize Phase111 GRPO progress.jsonl into compact health metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("progress.jsonl has no rows")
    rows_sorted = sorted(rows, key=lambda r: int(r.get("update", -1)))
    skipped_rows = [r for r in rows_sorted if bool(r.get("skipped", False))]
    identical_return_updates: list[int] = []
    for row in rows_sorted:
        vals = [float(v) for v in row.get("returns", [])]
        if vals and len(set(vals)) == 1:
            identical_return_updates.append(int(row.get("update", -1)))

    def _avg_return(row: dict[str, Any]) -> float:
        return float(row.get("avg_return", float("-inf")))

    def _success_rate(row: dict[str, Any]) -> float:
        return float(row.get("success_rate", float("-inf")))

    best_ret = max(rows_sorted, key=_avg_return)
    best_succ = max(rows_sorted, key=_success_rate)
    tail5 = rows_sorted[-5:]
    episode_lengths_last5 = [int(v) for r in tail5 for v in r.get("episode_lengths", [])]
    last = rows_sorted[-1]

    return {
        "n_updates": len(rows_sorted),
        "first_update": int(rows_sorted[0].get("update", -1)),
        "last_update": int(last.get("update", -1)),
        "skipped_updates": len(skipped_rows),
        "skip_fraction": float(len(skipped_rows) / max(len(rows_sorted), 1)),
        "best_avg_return_update": int(best_ret.get("update", -1)),
        "best_avg_return": float(best_ret.get("avg_return", 0.0)),
        "best_success_rate_update": int(best_succ.get("update", -1)),
        "best_success_rate": float(best_succ.get("success_rate", 0.0)),
        "last_avg_return": float(last.get("avg_return", 0.0)),
        "last_success_rate": float(last.get("success_rate", 0.0)),
        "mean_episode_length_last5": (
            float(mean(episode_lengths_last5)) if episode_lengths_last5 else 0.0
        ),
        "identical_return_updates": identical_return_updates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    progress_path = run_dir / "progress.jsonl"
    if not progress_path.exists():
        raise SystemExit(f"missing progress file: {progress_path}")
    rows = _load_rows(progress_path)
    summary = build_summary(rows)
    out_path = run_dir / "progress_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "phase111_progress_summary",
        f"updates={summary['n_updates']}",
        f"skipped={summary['skipped_updates']}",
        f"best_avg_return={summary['best_avg_return']}",
        f"best_success_rate={summary['best_success_rate']}",
        f"out={out_path}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
