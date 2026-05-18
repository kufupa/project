from __future__ import annotations

from pathlib import Path

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
