#!/usr/bin/env python3
"""Print CSV-style summary for smolvla_topk_best_summary.json (stdout)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    raw = os.environ.get("SUMMARY_PATH", "").strip()
    if not raw:
        print("error: SUMMARY_PATH is required", file=sys.stderr)
        return 2
    path = Path(raw).resolve()
    summary = json.loads(path.read_text(encoding="utf-8"))
    print(
        "rank, oracle_episode, oracle_sum, smolvla_best_episode, "
        "smolvla_sum, delta_sum, smolvla_video"
    )
    for row in summary:
        delta = float(row["smolvla_sum_reward"]) - float(row["oracle_sum_reward"])
        print(
            f'{row["oracle_rank"]}, {row["oracle_episode_index"]}, '
            f'{row["oracle_sum_reward"]}, {row["smolvla_best_episode_index"]}, '
            f'{row["smolvla_sum_reward"]}, {delta}, {row["best_video"]}'
        )
    print(f"summary_path={path}")
    print(f"environments={len(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
