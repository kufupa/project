#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def _int_cell(row: dict[str, str], key: str) -> int:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return 0
    try:
        return int(float(raw))
    except ValueError:
        return 0


def _percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(math.ceil((pct / 100.0) * len(ordered))) - 1
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])


def summarize_process_tree_memory_csv(csv_path: Path) -> dict[str, object]:
    tree_rss: list[int] = []
    tree_vmsize: list[int] = []
    self_rss: list[int] = []
    descendant_counts: list[int] = []
    max_descendant_rss: list[int] = []
    root_pids: set[int] = set()

    with csv_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            tree_rss.append(_int_cell(row, "tree_rss_kb"))
            tree_vmsize.append(_int_cell(row, "tree_vmsize_kb"))
            self_rss.append(_int_cell(row, "self_rss_kb"))
            descendant_counts.append(_int_cell(row, "descendant_count"))
            max_descendant_rss.append(_int_cell(row, "max_descendant_rss_kb"))
            root_pid = _int_cell(row, "root_pid")
            if root_pid:
                root_pids.add(root_pid)

    max_tree_rss = max(tree_rss) if tree_rss else 0
    max_tree_vmsize = max(tree_vmsize) if tree_vmsize else 0
    return {
        "telemetry_csv": str(csv_path.resolve()),
        "sample_count": len(tree_rss),
        "root_pids": sorted(root_pids),
        "max_tree_rss_kb": int(max_tree_rss),
        "max_tree_rss_mib": float(max_tree_rss / 1024.0),
        "mean_tree_rss_kb": float(mean(tree_rss) if tree_rss else 0.0),
        "p95_tree_rss_kb": _percentile(tree_rss, 95.0),
        "max_tree_vmsize_kb": int(max_tree_vmsize),
        "max_tree_vmsize_mib": float(max_tree_vmsize / 1024.0),
        "max_self_rss_kb": int(max(self_rss) if self_rss else 0),
        "max_descendant_count": int(max(descendant_counts) if descendant_counts else 0),
        "max_max_descendant_rss_kb": int(max(max_descendant_rss) if max_descendant_rss else 0),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_process_tree_memory_csv(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "cpu_mem_telemetry_summary_ok",
        f"samples={summary['sample_count']}",
        f"max_tree_rss_mib={summary['max_tree_rss_mib']:.1f}",
        f"max_tree_vmsize_mib={summary['max_tree_vmsize_mib']:.1f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
