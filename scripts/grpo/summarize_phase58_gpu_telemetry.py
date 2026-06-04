#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterable


def _float_cell(row: dict[str, str], keys: Iterable[str]) -> float | None:
    for key in keys:
        if key not in row:
            continue
        raw = str(row[key]).strip()
        if not raw or raw == "-":
            return None
        raw = raw.replace("%", "").replace("MiB", "").replace("W", "").strip()
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(math.ceil((pct / 100.0) * len(ordered))) - 1
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])


def summarize_nvidia_smi_csv(csv_path: Path) -> dict[str, object]:
    gpu_utils: list[float] = []
    mem_utils: list[float] = []
    mem_used: list[float] = []
    mem_total: list[float] = []
    power_draw: list[float] = []
    temperatures: list[float] = []
    gpu_names: set[str] = set()
    gpu_uuids: set[str] = set()

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {str(key).strip(): str(value).strip() for key, value in raw_row.items() if key is not None}
            gpu_name = str(row.get("name", "")).strip()
            gpu_uuid = str(row.get("uuid", "")).strip()
            if gpu_name:
                gpu_names.add(gpu_name)
            if gpu_uuid:
                gpu_uuids.add(gpu_uuid)

            gpu_util = _float_cell(row, ("utilization.gpu [%]", "utilization.gpu"))
            mem_util = _float_cell(row, ("utilization.memory [%]", "utilization.memory"))
            used = _float_cell(row, ("memory.used [MiB]", "memory.used"))
            total = _float_cell(row, ("memory.total [MiB]", "memory.total"))
            power = _float_cell(row, ("power.draw [W]", "power.draw"))
            temp = _float_cell(row, ("temperature.gpu",))

            if gpu_util is not None:
                gpu_utils.append(gpu_util)
            if mem_util is not None:
                mem_utils.append(mem_util)
            if used is not None:
                mem_used.append(used)
            if total is not None:
                mem_total.append(total)
            if power is not None:
                power_draw.append(power)
            if temp is not None:
                temperatures.append(temp)

    max_mem_used = max(mem_used) if mem_used else 0.0
    max_mem_total = max(mem_total) if mem_total else 0.0
    return {
        "telemetry_csv": str(csv_path.resolve()),
        "sample_count": len(gpu_utils),
        "gpu_names": sorted(gpu_names),
        "gpu_uuids": sorted(gpu_uuids),
        "max_gpu_utilization_pct": float(max(gpu_utils) if gpu_utils else 0.0),
        "mean_gpu_utilization_pct": float(mean(gpu_utils) if gpu_utils else 0.0),
        "p95_gpu_utilization_pct": _percentile(gpu_utils, 95.0),
        "samples_gpu_utilization_ge_80_pct": int(sum(1 for value in gpu_utils if value >= 80.0)),
        "samples_gpu_utilization_ge_95_pct": int(sum(1 for value in gpu_utils if value >= 95.0)),
        "max_memory_utilization_pct": float(max(mem_utils) if mem_utils else 0.0),
        "mean_memory_utilization_pct": float(mean(mem_utils) if mem_utils else 0.0),
        "max_memory_used_mib": float(max_mem_used),
        "max_memory_total_mib": float(max_mem_total),
        "max_memory_used_fraction": float(max_mem_used / max_mem_total) if max_mem_total > 0 else 0.0,
        "max_power_draw_w": float(max(power_draw) if power_draw else 0.0),
        "mean_power_draw_w": float(mean(power_draw) if power_draw else 0.0),
        "max_temperature_gpu_c": float(max(temperatures) if temperatures else 0.0),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_nvidia_smi_csv(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "gpu_telemetry_summary_ok",
        f"samples={summary['sample_count']}",
        f"max_gpu={summary['max_gpu_utilization_pct']:.1f}",
        f"mean_gpu={summary['mean_gpu_utilization_pct']:.1f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
