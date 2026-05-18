#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from smolvla_grpo.process_memory import process_tree_memory_snapshot

_STOP = False
_FIELDNAMES = [
    "timestamp_unix_ms",
    "label",
    "root_pid",
    "process_count",
    "descendant_count",
    "missing_process_count",
    "self_rss_kb",
    "self_vmhwm_kb",
    "self_vmsize_kb",
    "self_vmpeak_kb",
    "tree_rss_kb",
    "tree_vmhwm_kb",
    "tree_vmsize_kb",
    "tree_vmpeak_kb",
    "max_descendant_rss_kb",
]


def _handle_stop(signum, frame) -> None:
    del signum, frame
    global _STOP
    _STOP = True


def _pid_exists(pid: int, *, proc_root: Path = Path("/proc")) -> bool:
    return (proc_root / str(int(pid))).exists()


def write_process_tree_memory_sample(
    output: Path,
    *,
    root_pid: int,
    label: str,
    proc_root: Path = Path("/proc"),
    timestamp_unix_ms: int | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    row = process_tree_memory_snapshot(
        root_pid=int(root_pid),
        proc_root=proc_root,
        timestamp_unix_ms=timestamp_unix_ms,
    )
    row = {"label": str(label), **row}
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, 0) for key in _FIELDNAMES})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--label", type=str, default="train")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    interval = max(float(args.interval), 0.25)
    while not _STOP and _pid_exists(int(args.root_pid)):
        write_process_tree_memory_sample(
            args.output,
            root_pid=int(args.root_pid),
            label=str(args.label),
        )
        time.sleep(interval)
    if _pid_exists(int(args.root_pid)):
        write_process_tree_memory_sample(
            args.output,
            root_pid=int(args.root_pid),
            label=str(args.label),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
