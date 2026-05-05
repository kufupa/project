from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Mapping

_STATUS_FIELDS = ("PPid", "VmRSS", "VmHWM", "VmSize", "VmPeak")


def _read_status_kb(pid: int, *, proc_root: Path = Path("/proc")) -> dict[str, int]:
    status_path = proc_root / str(int(pid)) / "status"
    values: dict[str, int] = {}
    try:
        with status_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                name, sep, rest = line.partition(":")
                if not sep or name not in _STATUS_FIELDS:
                    continue
                parts = rest.split()
                if not parts:
                    continue
                try:
                    values[name] = int(parts[0])
                except ValueError:
                    continue
    except OSError:
        return {}
    return values


def _read_process_table(proc_root: Path) -> tuple[dict[int, dict[str, int]], dict[int, list[int]]]:
    statuses: dict[int, dict[str, int]] = {}
    children: dict[int, list[int]] = defaultdict(list)
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return statuses, children

    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        status = _read_status_kb(pid, proc_root=proc_root)
        if not status:
            continue
        statuses[pid] = status
        ppid = status.get("PPid")
        if ppid is not None:
            children[int(ppid)].append(pid)
    return statuses, children


def _descendant_pids(root_pid: int, children: Mapping[int, list[int]]) -> list[int]:
    descendants: list[int] = []
    queue: deque[int] = deque(children.get(int(root_pid), []))
    while queue:
        pid = queue.popleft()
        descendants.append(pid)
        queue.extend(children.get(pid, []))
    return descendants


def process_tree_memory_snapshot(
    root_pid: int | None = None,
    *,
    proc_root: Path = Path("/proc"),
    timestamp_unix_ms: int | None = None,
) -> dict[str, int]:
    root = int(root_pid if root_pid is not None else os.getpid())
    timestamp = int(timestamp_unix_ms if timestamp_unix_ms is not None else time.time() * 1000)
    statuses, children = _read_process_table(proc_root)
    pids = [root, *_descendant_pids(root, children)]
    present = [pid for pid in pids if pid in statuses]
    root_status = statuses.get(root, {})
    descendant_statuses = [statuses[pid] for pid in present if pid != root]
    present_statuses = [statuses[pid] for pid in present]

    def _sum(field: str, rows: list[dict[str, int]]) -> int:
        return int(sum(int(row.get(field, 0)) for row in rows))

    return {
        "timestamp_unix_ms": timestamp,
        "root_pid": root,
        "process_count": len(present),
        "descendant_count": max(len(present) - 1, 0),
        "missing_process_count": max(len(pids) - len(present), 0),
        "self_rss_kb": int(root_status.get("VmRSS", 0)),
        "self_vmhwm_kb": int(root_status.get("VmHWM", 0)),
        "self_vmsize_kb": int(root_status.get("VmSize", 0)),
        "self_vmpeak_kb": int(root_status.get("VmPeak", 0)),
        "tree_rss_kb": _sum("VmRSS", present_statuses),
        "tree_vmhwm_kb": _sum("VmHWM", present_statuses),
        "tree_vmsize_kb": _sum("VmSize", present_statuses),
        "tree_vmpeak_kb": _sum("VmPeak", present_statuses),
        "max_descendant_rss_kb": int(
            max((int(row.get("VmRSS", 0)) for row in descendant_statuses), default=0)
        ),
    }


def prefixed_process_tree_memory_fields(
    prefix: str,
    root_pid: int | None = None,
    *,
    proc_root: Path = Path("/proc"),
    timestamp_unix_ms: int | None = None,
) -> dict[str, int]:
    clean_prefix = str(prefix).strip().rstrip("_")
    if not clean_prefix:
        raise ValueError("prefix must be non-empty")
    snapshot = process_tree_memory_snapshot(
        root_pid=root_pid,
        proc_root=proc_root,
        timestamp_unix_ms=timestamp_unix_ms,
    )
    return {f"{clean_prefix}_{key}": value for key, value in snapshot.items()}
