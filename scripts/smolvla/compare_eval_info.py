#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: dict, key: str) -> float:
    overall = payload.get("overall", {})
    value = overall.get(key, 0.0)
    try:
        return float(value)
    except Exception:
        return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two eval_info.json files.")
    parser.add_argument("--baseline", required=True, help="Path to baseline eval_info.json")
    parser.add_argument("--candidate", required=True, help="Path to candidate eval_info.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_path = Path(args.baseline).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    baseline = _read_json(baseline_path)
    candidate = _read_json(candidate_path)

    keys = ("avg_sum_reward", "pc_success", "avg_max_reward", "n_episodes")
    rows: list[dict[str, float | str]] = []
    for key in keys:
        b = _metric(baseline, key)
        c = _metric(candidate, key)
        rows.append(
            {
                "metric": key,
                "baseline": b,
                "candidate": c,
                "delta": c - b,
            }
        )

    payload = {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "metrics": rows,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
