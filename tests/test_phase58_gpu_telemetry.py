from __future__ import annotations

import csv
import json
from pathlib import Path


def test_gpu_telemetry_summary_reports_max_mean_p95_and_memory(tmp_path: Path) -> None:
    from scripts.grpo.summarize_phase58_gpu_telemetry import summarize_nvidia_smi_csv

    csv_path = tmp_path / "nvidia_smi_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "index",
                "uuid",
                "name",
                "utilization.gpu [%]",
                "utilization.memory [%]",
                "memory.used [MiB]",
                "memory.total [MiB]",
                "power.draw [W]",
                "temperature.gpu",
            ]
        )
        writer.writerow(["2026/05/17 10:00:00.000", "0", "GPU-a", "RTX6000", "10", "20", "1000", "24000", "70.5", "40"])
        writer.writerow(["2026/05/17 10:00:05.000", "0", "GPU-a", "RTX6000", "90", "70", "12000", "24000", "220.0", "62"])
        writer.writerow(["2026/05/17 10:00:10.000", "0", "GPU-a", "RTX6000", "100", "80", "18000", "24000", "250.0", "68"])

    summary = summarize_nvidia_smi_csv(csv_path)

    assert summary["sample_count"] == 3
    assert summary["max_gpu_utilization_pct"] == 100.0
    assert summary["mean_gpu_utilization_pct"] == 200.0 / 3.0
    assert summary["p95_gpu_utilization_pct"] == 100.0
    assert summary["samples_gpu_utilization_ge_80_pct"] == 2
    assert summary["max_memory_used_mib"] == 18000.0
    assert summary["max_memory_used_fraction"] == 0.75
    assert summary["max_memory_utilization_pct"] == 80.0
    assert summary["max_power_draw_w"] == 250.0


def test_gpu_telemetry_cli_writes_json(tmp_path: Path) -> None:
    from scripts.grpo.summarize_phase58_gpu_telemetry import main

    csv_path = tmp_path / "nvidia_smi_samples.csv"
    out_path = tmp_path / "gpu_telemetry_summary.json"
    csv_path.write_text(
        "timestamp, index, uuid, name, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB], power.draw [W], temperature.gpu\n"
        "2026/05/17 10:00:00.000, 0, GPU-a, RTX6000, 50, 40, 8000, 24000, 180, 55\n",
        encoding="utf-8",
    )

    assert main(["--csv", str(csv_path), "--output", str(out_path)]) == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 1
    assert payload["max_gpu_utilization_pct"] == 50.0
    assert payload["telemetry_csv"] == str(csv_path.resolve())
