from __future__ import annotations

import csv
from pathlib import Path

from scripts.grpo.sample_process_tree_memory import write_process_tree_memory_sample
from scripts.grpo.summarize_process_tree_memory import summarize_process_tree_memory_csv
from smolvla_grpo.process_memory import (
    prefixed_process_tree_memory_fields,
    process_tree_memory_snapshot,
)


def _write_status(proc_root: Path, pid: int, *, ppid: int, rss: int, hwm: int, vmsize: int, vmpeak: int) -> None:
    proc_dir = proc_root / str(pid)
    proc_dir.mkdir(parents=True)
    (proc_dir / "status").write_text(
        "\n".join(
            [
                f"Name:\tproc{pid}",
                f"PPid:\t{ppid}",
                f"VmRSS:\t{rss} kB",
                f"VmHWM:\t{hwm} kB",
                f"VmSize:\t{vmsize} kB",
                f"VmPeak:\t{vmpeak} kB",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_process_tree_memory_snapshot_sums_descendants(tmp_path: Path) -> None:
    _write_status(tmp_path, 100, ppid=1, rss=1000, hwm=1100, vmsize=5000, vmpeak=5100)
    _write_status(tmp_path, 101, ppid=100, rss=2000, hwm=2200, vmsize=6000, vmpeak=6200)
    _write_status(tmp_path, 102, ppid=101, rss=3000, hwm=3300, vmsize=7000, vmpeak=7300)
    _write_status(tmp_path, 200, ppid=1, rss=9000, hwm=9900, vmsize=19000, vmpeak=19900)

    snap = process_tree_memory_snapshot(root_pid=100, proc_root=tmp_path, timestamp_unix_ms=123)

    assert snap["timestamp_unix_ms"] == 123
    assert snap["root_pid"] == 100
    assert snap["process_count"] == 3
    assert snap["descendant_count"] == 2
    assert snap["missing_process_count"] == 0
    assert snap["self_rss_kb"] == 1000
    assert snap["self_vmhwm_kb"] == 1100
    assert snap["self_vmsize_kb"] == 5000
    assert snap["tree_rss_kb"] == 6000
    assert snap["tree_vmhwm_kb"] == 6600
    assert snap["tree_vmsize_kb"] == 18000
    assert snap["tree_vmpeak_kb"] == 18600
    assert snap["max_descendant_rss_kb"] == 3000


def test_prefixed_process_tree_memory_fields_are_flat(tmp_path: Path) -> None:
    _write_status(tmp_path, 100, ppid=1, rss=1000, hwm=1100, vmsize=5000, vmpeak=5100)

    fields = prefixed_process_tree_memory_fields(
        "proc_mem_update_start",
        root_pid=100,
        proc_root=tmp_path,
        timestamp_unix_ms=456,
    )

    assert fields == {
        "proc_mem_update_start_timestamp_unix_ms": 456,
        "proc_mem_update_start_root_pid": 100,
        "proc_mem_update_start_process_count": 1,
        "proc_mem_update_start_descendant_count": 0,
        "proc_mem_update_start_missing_process_count": 0,
        "proc_mem_update_start_self_rss_kb": 1000,
        "proc_mem_update_start_self_vmhwm_kb": 1100,
        "proc_mem_update_start_self_vmsize_kb": 5000,
        "proc_mem_update_start_self_vmpeak_kb": 5100,
        "proc_mem_update_start_tree_rss_kb": 1000,
        "proc_mem_update_start_tree_vmhwm_kb": 1100,
        "proc_mem_update_start_tree_vmsize_kb": 5000,
        "proc_mem_update_start_tree_vmpeak_kb": 5100,
        "proc_mem_update_start_max_descendant_rss_kb": 0,
    }


def test_write_process_tree_memory_sample_appends_csv(tmp_path: Path) -> None:
    _write_status(tmp_path / "proc", 100, ppid=1, rss=1000, hwm=1100, vmsize=5000, vmpeak=5100)
    output = tmp_path / "memory.csv"

    write_process_tree_memory_sample(
        output,
        root_pid=100,
        label="train",
        proc_root=tmp_path / "proc",
        timestamp_unix_ms=1000,
    )
    write_process_tree_memory_sample(
        output,
        root_pid=100,
        label="train",
        proc_root=tmp_path / "proc",
        timestamp_unix_ms=2000,
    )

    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["label"] == "train"
    assert rows[0]["root_pid"] == "100"
    assert rows[0]["tree_rss_kb"] == "1000"
    assert rows[1]["timestamp_unix_ms"] == "2000"


def test_summarize_process_tree_memory_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "memory.csv"
    csv_path.write_text(
        "\n".join(
            [
                "timestamp_unix_ms,label,root_pid,process_count,descendant_count,missing_process_count,self_rss_kb,self_vmhwm_kb,self_vmsize_kb,self_vmpeak_kb,tree_rss_kb,tree_vmhwm_kb,tree_vmsize_kb,tree_vmpeak_kb,max_descendant_rss_kb",
                "1000,train,100,1,0,0,1000,1100,5000,5100,1000,1100,5000,5100,0",
                "2000,train,100,3,2,0,2000,2200,6000,6200,9000,9900,16000,16200,4000",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_process_tree_memory_csv(csv_path)

    assert summary["sample_count"] == 2
    assert summary["max_tree_rss_kb"] == 9000
    assert summary["max_tree_rss_mib"] == 9000 / 1024.0
    assert summary["max_tree_vmsize_kb"] == 16000
    assert summary["max_descendant_count"] == 2
    assert summary["max_max_descendant_rss_kb"] == 4000
