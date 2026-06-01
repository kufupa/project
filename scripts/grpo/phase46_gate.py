#!/usr/bin/env python3
"""Parse tiered_eval_summary.json and update results ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def best_row(rows: list[dict[str, Any]], *, key: str = "pc_success") -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda r: (float(r.get(key, 0.0)), float(r.get("success_rate", 0.0))))


def parse_tiered_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows_25 = list(data.get("rows_25ep", []))
    rows_100 = list(data.get("rows_100ep", []))
    for row in rows_25 + rows_100:
        if "pc_success" not in row and "success_rate" in row:
            row["pc_success"] = float(row["success_rate"]) * 100.0
    baseline = next((r for r in rows_100 if r.get("update") == 0 or r.get("checkpoint") == "baseline"), None)
    first = min(rows_100, key=lambda r: int(r.get("update", 9999)), default=None) if rows_100 else None
    last = max(rows_100, key=lambda r: int(r.get("update", 0)), default=None) if rows_100 else None
    return {
        "best_25ep": best_row(rows_25),
        "baseline_100ep": baseline,
        "first_100ep": first,
        "last_100ep": last,
        "rows_25ep": rows_25,
        "rows_100ep": rows_100,
        "protocol": data.get("protocol", {}),
    }


def append_ledger(ledger_path: Path, row: dict[str, Any]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if ledger_path.is_file():
        text = ledger_path.read_text(encoding="utf-8")
    else:
        text = "| run | mode | best_25ep | base_100 | first_100 | last_100 | notes |\n|-----|------|-----------|----------|-----------|----------|-------|\n"
    line = (
        f"| {row.get('run_id')} | {row.get('logprob_mode')} | "
        f"{row.get('best_25ep_pc', '—')} | {row.get('baseline_100_pc', '—')} | "
        f"{row.get('first_100_pc', '—')} | {row.get('last_100_pc', '—')} | {row.get('notes', '')} |\n"
    )
    ledger_path.write_text(text + line, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--run-id", type=str, default="phase46")
    p.add_argument("--logprob-mode", type=str, default="gaussian")
    p.add_argument(
        "--ledger",
        type=Path,
        default=Path(
            "/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/docs/superpowers/plans/results-ledger.md"
        ),
    )
    p.add_argument("--threshold", type=float, default=40.0)
    args = p.parse_args()

    parsed = parse_tiered_summary(args.summary)
    b25 = parsed.get("best_25ep") or {}
    b100 = parsed.get("baseline_100ep") or {}
    f100 = parsed.get("first_100ep") or {}
    l100 = parsed.get("last_100ep") or {}

    def pc(r: dict) -> float:
        return float(r.get("pc_success", r.get("success_rate", 0.0) * 100.0))

    append_ledger(
        args.ledger,
        {
            "run_id": args.run_id,
            "logprob_mode": args.logprob_mode,
            "best_25ep_pc": f"{pc(b25):.1f}" if b25 else "—",
            "baseline_100_pc": f"{pc(b100):.1f}" if b100 else "—",
            "first_100_pc": f"{pc(f100):.1f}" if f100 else "—",
            "last_100_pc": f"{pc(l100):.1f}" if l100 else "—",
            "notes": json.dumps(parsed.get("protocol", {}), sort_keys=True),
        },
    )
    last_pc = pc(l100) if l100 else 0.0
    print(
        "phase46_gate",
        f"last_100ep={last_pc:.1f}",
        f"threshold={args.threshold}",
        flush=True,
    )
    return 0 if last_pc >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
