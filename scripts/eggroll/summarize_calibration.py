#!/usr/bin/env python3
"""Summarize EGGROLL calibration probe artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def _load_jsonl_last(path: Path) -> dict:
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[-1] if rows else {}


def _candidate_metrics(run_dir: Path, batch: int) -> dict:
    progress = _load_jsonl_last(run_dir / "progress.jsonl")
    timing = _load_jsonl_last(run_dir / "timings.jsonl")
    latest = run_dir / "checkpoints" / "latest.pt"
    completed = bool(progress) and latest.is_file()
    failure_kind = None
    if not completed:
        failure_kind = "missing_artifact"
    if "relative_update_norm" in progress and not isinstance(progress["relative_update_norm"], (int, float)):
        failure_kind = "bad_update_norm"
    metrics = {
        "population_batch_size": int(batch),
        "run_dir": str(run_dir),
        "completed": completed,
        "failure_kind": failure_kind,
        "checkpoint_exists": latest.is_file(),
        "progress_rows": sum(1 for line in (run_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
        if (run_dir / "progress.jsonl").is_file()
        else 0,
        "fitness_mean_last": progress.get("fitness_mean"),
        "relative_update_norm_last": progress.get("relative_update_norm"),
        "iteration_seconds": timing.get("iteration_seconds"),
        "cuda_max_memory_allocated_gb": timing.get("cuda_max_memory_allocated_gb", 0.0),
        "cuda_max_memory_reserved_gb": timing.get("cuda_max_memory_reserved_gb", 0.0),
        "cuda_total_memory_gb": timing.get("cuda_total_memory_gb", 0.0),
    }
    total = float(metrics["cuda_total_memory_gb"] or 0.0)
    reserved = float(metrics["cuda_max_memory_reserved_gb"] or 0.0)
    metrics["vram_reserved_fraction"] = reserved / total if total > 0 else 0.0
    return metrics


def choose_candidate(metrics: list[dict]) -> tuple[int, str]:
    safe = [
        item
        for item in metrics
        if item.get("completed")
        and float(item.get("vram_reserved_fraction") or 0.0) <= 0.85
        and isinstance(item.get("relative_update_norm_last"), (int, float))
    ]
    if not safe:
        raise SystemExit("no safe calibration candidate")
    safe.sort(key=lambda item: int(item["population_batch_size"]))
    selected = safe[-1]
    reason = f"selected largest safe candidate b{selected['population_batch_size']}"
    if int(selected["population_batch_size"]) == 16:
        prev = [item for item in safe if int(item["population_batch_size"]) == 8]
        if prev:
            p = prev[0]
            sel_time = float(selected.get("iteration_seconds") or 1e18)
            prev_time = float(p.get("iteration_seconds") or 1e18)
            if sel_time > prev_time * 0.9:
                selected = p
                reason = "selected b8 because b16 was not meaningfully faster"
    return int(selected["population_batch_size"]), reason


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: summarize_calibration.py ROOT BATCH...", file=sys.stderr)
        return 2
    root = Path(argv[1])
    batches = [int(x) for x in argv[2:]]
    metrics = [_candidate_metrics(root / f"pop32_b{b}_r2_seed2000", b) for b in batches]
    selected, reason = choose_candidate(metrics)
    summary = {
        "selected_population_batch_size": selected,
        "selection_reason": reason,
        "candidates": batches,
        "metrics_by_candidate": {str(item["population_batch_size"]): item for item in metrics},
        "safe_candidates": [
            item["population_batch_size"]
            for item in metrics
            if item.get("completed")
            and float(item.get("vram_reserved_fraction") or 0.0) <= 0.85
            and isinstance(item.get("relative_update_norm_last"), (int, float))
        ],
        "failed_candidates": [item for item in metrics if not item.get("completed")],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
