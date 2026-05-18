# Phase11 CPU Memory Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CPU RAM time-series telemetry for Phase11 GRPO so pop128 jobs show whether RSS/VMEM grows across updates before PBS cgroup OOM.

**Architecture:** Keep GPU telemetry in `nvidia-smi`; add CPU telemetry through Linux `/proc` because PBS OOM is CPU cgroup memory, not GPU VRAM. Trainer writes per-update `proc_mem_*` fields to `progress.jsonl`; PBS scripts run a separate background process-tree sampler beside existing GPU sampling and summarize it at exit.

**Tech Stack:** Python 3.12, Linux `/proc`, PBS bash scripts, pytest static/unit tests, JSONL and CSV artifacts.

---

## Scope Check

This plan touches only Phase11 GRPO resource telemetry. It does not change GRPO loss math, rollout batching, hyperparameters, checkpoint format, or evaluation scripts.

## File Structure

- Create: `src/smolvla_grpo/process_memory.py`
  - Reusable `/proc` reader for self/process-tree memory snapshots.
  - No third-party dependency; avoids adding `psutil`.
- Modify: `scripts/grpo/train_phase11_env_on_policy_grpo.py`
  - Import helper.
  - Record process memory snapshots at update start, after rollout, and after optimize.
  - Add flat `proc_mem_*` fields to each `progress.jsonl` row and checkpoint metadata.
- Create: `scripts/grpo/sample_process_tree_memory.py`
  - Background CSV sampler for PBS jobs.
  - Samples trainer process tree every `CPU_MEM_TELEMETRY_INTERVAL`.
- Create: `scripts/grpo/summarize_process_tree_memory.py`
  - Summarizes CPU memory CSV into JSON, mirroring existing GPU summary pattern.
- Modify: Phase11 train PBS scripts that run long GRPO jobs:
  - `scripts/grpo/phase11_pop128_rolloutpbs32_smoke_u1.pbs`
  - `scripts/grpo/phase11_P128A_lr2e6_clip005_train_0001_0050.pbs`
  - `scripts/grpo/phase11_P128B_lr5e6_clip01_train_0001_0050.pbs`
  - `scripts/grpo/phase11_P128C_lr5e6_clip01_lownoise_train_0001_0050.pbs`
  - `scripts/grpo/phase11_R1_g32_lr2e6_clip005_train_0001_0050.pbs`
  - `scripts/grpo/phase11_R2_g32_lr5e6_clip01_lownoise_train_0001_0050.pbs`
  - `scripts/grpo/phase11_R3_g64_lr5e6_clip01_train_0001_0050.pbs`
  - `scripts/grpo/phase11_A_g32_lr5e6_clip01_train_0000_0030.pbs`
  - `scripts/grpo/phase11_A_g32_lr5e6_clip01_resume_train_0014_0030.pbs`
  - `scripts/grpo/phase11_g16_lr5e6_clip02_train_0000_0010.pbs`
  - `scripts/grpo/phase11_g16_lr5e6_clip02_train_0010_0020_resume.pbs`
  - `scripts/grpo/phase11_batched_logprob_smoke_u2.pbs`
- Create: `tests/test_process_memory.py`
  - Unit tests with fake `/proc`.
- Modify: `tests/test_phase11_true_action_chunking.py`
  - Static tests that trainer exposes `proc_mem_*` progress fields.
- Modify: `tests/test_phase11_slurm_scripts.py`
  - Static tests that selected PBS train scripts start CPU memory telemetry and summarize it.

---

### Task 1: Reusable Process Memory Helper

**Files:**
- Create: `src/smolvla_grpo/process_memory.py`
- Create: `tests/test_process_memory.py`

- [ ] **Step 1: Write failing tests for fake `/proc` tree**

Create `tests/test_process_memory.py`:

```python
from __future__ import annotations

from pathlib import Path

from smolvla_grpo.process_memory import (
    prefixed_process_tree_memory_fields,
    process_tree_memory_snapshot,
)


def _write_status(proc_root: Path, pid: int, *, ppid: int, rss: int, hwm: int, vmsize: int, vmpeak: int) -> None:
    proc_dir = proc_root / str(pid)
    proc_dir.mkdir()
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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_process_memory.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'smolvla_grpo.process_memory'`.

- [ ] **Step 3: Implement helper**

Create `src/smolvla_grpo/process_memory.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
pytest tests/test_process_memory.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smolvla_grpo/process_memory.py tests/test_process_memory.py
git commit -m "$(cat <<'EOF'
feat(grpo): add process memory snapshots

EOF
)"
```

---

### Task 2: Trainer Per-Update `proc_mem_*` Fields

**Files:**
- Modify: `scripts/grpo/train_phase11_env_on_policy_grpo.py`
- Modify: `tests/test_phase11_true_action_chunking.py`

- [ ] **Step 1: Write failing static test**

Append to `tests/test_phase11_true_action_chunking.py`:

```python
def test_phase11_train_script_logs_process_memory_static():
    script = (_REPO / "scripts" / "grpo" / "train_phase11_env_on_policy_grpo.py").read_text(
        encoding="utf-8"
    )
    assert "from smolvla_grpo.process_memory import prefixed_process_tree_memory_fields" in script
    assert 'proc_mem_update_start' in script
    assert 'proc_mem_after_rollout' in script
    assert 'proc_mem_after_optimize' in script
    assert '**proc_mem_update_start' in script
    assert '**proc_mem_after_rollout' in script
    assert '**proc_mem_after_optimize' in script
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest tests/test_phase11_true_action_chunking.py::test_phase11_train_script_logs_process_memory_static -q
```

Expected: FAIL on missing import or missing string.

- [ ] **Step 3: Import helper**

In `scripts/grpo/train_phase11_env_on_policy_grpo.py`, add this import with other `smolvla_grpo` imports:

```python
from smolvla_grpo.process_memory import prefixed_process_tree_memory_fields
```

- [ ] **Step 4: Add production helper near `_json_ready_args`**

Insert after `_json_ready_args`:

```python
def _proc_mem_fields(stage: str) -> dict[str, int]:
    return prefixed_process_tree_memory_fields(f"proc_mem_{stage}")
```

- [ ] **Step 5: Capture update start and rollout memory**

In the update loop, replace the temporary `AGENT_DEBUG_MEM` update-start block with:

```python
        proc_mem_update_start = _proc_mem_fields("update_start")
```

After `rollout_seconds = float(time.perf_counter() - rollout_t0)`, add:

```python
        proc_mem_after_rollout = _proc_mem_fields("after_rollout")
```

- [ ] **Step 6: Include memory fields in zero-advantage rows**

In the zero-advantages branch, before `skipped_extra = {`, add:

```python
            proc_mem_after_optimize = _proc_mem_fields("after_optimize")
```

Then add these spreads inside `skipped_extra` after `"update_seconds": update_seconds,`:

```python
                **proc_mem_update_start,
                **proc_mem_after_rollout,
                **proc_mem_after_optimize,
```

- [ ] **Step 7: Include memory fields in optimized rows**

After `optimize_seconds = float(time.perf_counter() - optimize_t0)`, replace the temporary `AGENT_DEBUG_MEM` optimize block with:

```python
        proc_mem_after_optimize = _proc_mem_fields("after_optimize")
```

Then add these spreads inside `checkpoint_extra` after `"update_seconds": update_seconds,`:

```python
            **proc_mem_update_start,
            **proc_mem_after_rollout,
            **proc_mem_after_optimize,
```

- [ ] **Step 8: Add log-line hints**

In both `print("phase111_grpo_update", ...)` calls, add these arguments after `update_s`:

```python
                f"rss_tree_mb={checkpoint_extra.get('proc_mem_after_optimize_tree_rss_kb', 0) / 1024.0:.1f}",
                f"vmem_tree_mb={checkpoint_extra.get('proc_mem_after_optimize_tree_vmsize_kb', 0) / 1024.0:.1f}",
```

For the skipped branch, use `skipped_extra` instead of `checkpoint_extra`:

```python
                f"rss_tree_mb={skipped_extra.get('proc_mem_after_optimize_tree_rss_kb', 0) / 1024.0:.1f}",
                f"vmem_tree_mb={skipped_extra.get('proc_mem_after_optimize_tree_vmsize_kb', 0) / 1024.0:.1f}",
```

- [ ] **Step 9: Remove obsolete agent-only memory logger**

Delete the `_debug_mem_log` function and all remaining `_debug_mem_log(...)` calls. The replacement telemetry is always written to run artifacts, so `AGENT_DEBUG_MEM` is no longer needed.

- [ ] **Step 10: Run tests**

Run:

```bash
pytest tests/test_process_memory.py tests/test_phase11_true_action_chunking.py::test_phase11_train_script_logs_process_memory_static -q
```

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add scripts/grpo/train_phase11_env_on_policy_grpo.py tests/test_phase11_true_action_chunking.py
git commit -m "$(cat <<'EOF'
feat(grpo): log phase11 process memory

EOF
)"
```

---

### Task 3: Background CPU Memory CSV Sampler

**Files:**
- Create: `scripts/grpo/sample_process_tree_memory.py`
- Create: `scripts/grpo/summarize_process_tree_memory.py`
- Modify: `tests/test_process_memory.py`

- [ ] **Step 1: Write failing sampler tests**

Append to `tests/test_process_memory.py`:

```python
import csv
import json

from scripts.grpo.sample_process_tree_memory import write_process_tree_memory_sample
from scripts.grpo.summarize_process_tree_memory import summarize_process_tree_memory_csv


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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_process_memory.py -q
```

Expected: FAIL because sampler and summarizer scripts do not exist.

- [ ] **Step 3: Implement sampler script**

Create `scripts/grpo/sample_process_tree_memory.py`:

```python
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
```

- [ ] **Step 4: Implement summarizer script**

Create `scripts/grpo/summarize_process_tree_memory.py`:

```python
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
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_process_memory.py -q
python scripts/grpo/sample_process_tree_memory.py --help >/dev/null
python scripts/grpo/summarize_process_tree_memory.py --help >/dev/null
```

Expected: PASS and both `--help` commands exit 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/grpo/sample_process_tree_memory.py scripts/grpo/summarize_process_tree_memory.py tests/test_process_memory.py
git commit -m "$(cat <<'EOF'
feat(grpo): add CPU memory sampler

EOF
)"
```

---

### Task 4: Wire CPU Sampler Into Phase11 PBS

**Files:**
- Modify the Phase11 train PBS scripts listed in File Structure.
- Modify: `tests/test_phase11_slurm_scripts.py`

- [ ] **Step 1: Write failing static tests**

Append to `tests/test_phase11_slurm_scripts.py`:

```python
def test_phase11_train_pbs_scripts_track_cpu_memory() -> None:
    names = (
        "phase11_pop128_rolloutpbs32_smoke_u1.pbs",
        "phase11_P128A_lr2e6_clip005_train_0001_0050.pbs",
        "phase11_P128B_lr5e6_clip01_train_0001_0050.pbs",
        "phase11_P128C_lr5e6_clip01_lownoise_train_0001_0050.pbs",
        "phase11_R1_g32_lr2e6_clip005_train_0001_0050.pbs",
        "phase11_R2_g32_lr5e6_clip01_lownoise_train_0001_0050.pbs",
        "phase11_R3_g64_lr5e6_clip01_train_0001_0050.pbs",
        "phase11_A_g32_lr5e6_clip01_train_0000_0030.pbs",
        "phase11_A_g32_lr5e6_clip01_resume_train_0014_0030.pbs",
        "phase11_g16_lr5e6_clip02_train_0000_0010.pbs",
        "phase11_g16_lr5e6_clip02_train_0010_0020_resume.pbs",
        "phase11_batched_logprob_smoke_u2.pbs",
    )
    for name in names:
        path = _REPO_ROOT / "scripts" / "grpo" / name
        text = path.read_text(encoding="utf-8")
        assert 'CPU_MEM_TELEMETRY_INTERVAL="${CPU_MEM_TELEMETRY_INTERVAL:-5}"' in text
        assert 'CPU_MEM_TELEMETRY_DIR="${RUN_DIR}/cpu_mem_telemetry/train"' in text
        assert 'scripts/grpo/sample_process_tree_memory.py' in text
        assert 'scripts/grpo/summarize_process_tree_memory.py' in text
        assert 'process_tree_memory.csv' in text
        assert 'process_tree_memory_summary.json' in text
        subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest tests/test_phase11_slurm_scripts.py::test_phase11_train_pbs_scripts_track_cpu_memory -q
```

Expected: FAIL on missing CPU memory telemetry strings.

- [ ] **Step 3: Add CPU telemetry variables to each PBS script**

In each script, after `GPU_TELEMETRY_DIR="${RUN_DIR}/gpu_telemetry/train"`, add:

```bash
CPU_MEM_TELEMETRY_INTERVAL="${CPU_MEM_TELEMETRY_INTERVAL:-5}"
CPU_MEM_TELEMETRY_DIR="${RUN_DIR}/cpu_mem_telemetry/train"
```

Change each `mkdir -p` line from:

```bash
mkdir -p logs/pbs/grpo "${RUN_DIR}" "${GPU_TELEMETRY_DIR}"
```

to:

```bash
mkdir -p logs/pbs/grpo "${RUN_DIR}" "${GPU_TELEMETRY_DIR}" "${CPU_MEM_TELEMETRY_DIR}"
```

- [ ] **Step 4: Add sampler PIDs and trap cleanup**

In each script, after `NVIDIA_SMI_PID=""`, add:

```bash
TRAIN_PID=""
CPU_MEM_PID=""
```

In `stop_gpu_telemetry()`, after the NVIDIA SMI kill block, add:

```bash
  if [[ -n "${CPU_MEM_PID}" ]] && kill -0 "${CPU_MEM_PID}" >/dev/null 2>&1; then
    kill "${CPU_MEM_PID}" >/dev/null 2>&1 || true
    wait "${CPU_MEM_PID}" >/dev/null 2>&1 || true
  fi
```

After the `qstat` snapshot block or after the `nvidia_smi_end.txt` block when a script has no `qstat` block, add:

```bash
  if [[ -f "${CPU_MEM_TELEMETRY_DIR}/process_tree_memory.csv" ]]; then
    "${PYTHON_BIN}" scripts/grpo/summarize_process_tree_memory.py \
      --csv "${CPU_MEM_TELEMETRY_DIR}/process_tree_memory.csv" \
      --output "${CPU_MEM_TELEMETRY_DIR}/process_tree_memory_summary.json" || true
  fi
```

- [ ] **Step 5: Add CPU sampler start function**

After `start_gpu_telemetry()`, add:

```bash
start_cpu_mem_telemetry() {
  if [[ -n "${TRAIN_PID}" ]]; then
    "${PYTHON_BIN}" scripts/grpo/sample_process_tree_memory.py \
      --root-pid "${TRAIN_PID}" \
      --output "${CPU_MEM_TELEMETRY_DIR}/process_tree_memory.csv" \
      --interval "${CPU_MEM_TELEMETRY_INTERVAL}" \
      --label train \
      > "${CPU_MEM_TELEMETRY_DIR}/process_tree_memory.log" \
      2> "${CPU_MEM_TELEMETRY_DIR}/process_tree_memory.err" &
    CPU_MEM_PID=$!
  fi
}
```

- [ ] **Step 6: Run trainer in background so sampler can target PID**

For each PBS script, replace the direct trainer invocation:

```bash
"${PYTHON_BIN}" scripts/grpo/train_phase11_env_on_policy_grpo.py \
  --checkpoint "${BASE_CKPT}" \
  --output-dir "${RUN_DIR}" \
  ...
  --run-label example_label
```

with this structure, preserving the existing arguments exactly:

```bash
"${PYTHON_BIN}" scripts/grpo/train_phase11_env_on_policy_grpo.py \
  --checkpoint "${BASE_CKPT}" \
  --output-dir "${RUN_DIR}" \
  ...
  --run-label example_label &
TRAIN_PID=$!
start_cpu_mem_telemetry
set +e
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
set -e
TRAIN_PID=""
exit "${TRAIN_STATUS}"
```

Important: keep the existing `test -f ...` assertions after this block only in scripts where they can still run. If using `exit "${TRAIN_STATUS}"`, move the `test -f` assertions before the `exit`:

```bash
set +e
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
set -e
TRAIN_PID=""
if [[ "${TRAIN_STATUS}" -eq 0 ]]; then
  test -f "${RUN_DIR}/train_manifest.json"
  test -f "${RUN_DIR}/progress.jsonl"
  test -f "${RUN_DIR}/checkpoints/update_0050.pt"
  echo "PHASE11_P128A_TRAIN_DONE run_dir=${RUN_DIR}"
fi
exit "${TRAIN_STATUS}"
```

Use each script's existing final checkpoint and success echo. For the pop128 smoke script, keep its Python assertion block inside the `if [[ "${TRAIN_STATUS}" -eq 0 ]]` block.

- [ ] **Step 7: Run static tests**

Run:

```bash
pytest tests/test_phase11_slurm_scripts.py::test_phase11_train_pbs_scripts_track_cpu_memory -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts/grpo/phase11_pop128_rolloutpbs32_smoke_u1.pbs \
  scripts/grpo/phase11_P128A_lr2e6_clip005_train_0001_0050.pbs \
  scripts/grpo/phase11_P128B_lr5e6_clip01_train_0001_0050.pbs \
  scripts/grpo/phase11_P128C_lr5e6_clip01_lownoise_train_0001_0050.pbs \
  scripts/grpo/phase11_R1_g32_lr2e6_clip005_train_0001_0050.pbs \
  scripts/grpo/phase11_R2_g32_lr5e6_clip01_lownoise_train_0001_0050.pbs \
  scripts/grpo/phase11_R3_g64_lr5e6_clip01_train_0001_0050.pbs \
  scripts/grpo/phase11_A_g32_lr5e6_clip01_train_0000_0030.pbs \
  scripts/grpo/phase11_A_g32_lr5e6_clip01_resume_train_0014_0030.pbs \
  scripts/grpo/phase11_g16_lr5e6_clip02_train_0000_0010.pbs \
  scripts/grpo/phase11_g16_lr5e6_clip02_train_0010_0020_resume.pbs \
  scripts/grpo/phase11_batched_logprob_smoke_u2.pbs \
  tests/test_phase11_slurm_scripts.py
git commit -m "$(cat <<'EOF'
chore(grpo): sample CPU memory in phase11 PBS

EOF
)"
```

---

### Task 5: End-to-End Verification

**Files:**
- No new files.
- Verify files changed in Tasks 1-4.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/test_process_memory.py \
  tests/test_phase11_true_action_chunking.py::test_phase11_train_script_logs_process_memory_static \
  tests/test_phase11_slurm_scripts.py::test_phase11_train_pbs_scripts_track_cpu_memory \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run PBS bash syntax checks for changed scripts**

Run:

```bash
for f in \
  scripts/grpo/phase11_pop128_rolloutpbs32_smoke_u1.pbs \
  scripts/grpo/phase11_P128A_lr2e6_clip005_train_0001_0050.pbs \
  scripts/grpo/phase11_P128B_lr5e6_clip01_train_0001_0050.pbs \
  scripts/grpo/phase11_P128C_lr5e6_clip01_lownoise_train_0001_0050.pbs \
  scripts/grpo/phase11_R1_g32_lr2e6_clip005_train_0001_0050.pbs \
  scripts/grpo/phase11_R2_g32_lr5e6_clip01_lownoise_train_0001_0050.pbs \
  scripts/grpo/phase11_R3_g64_lr5e6_clip01_train_0001_0050.pbs \
  scripts/grpo/phase11_A_g32_lr5e6_clip01_train_0000_0030.pbs \
  scripts/grpo/phase11_A_g32_lr5e6_clip01_resume_train_0014_0030.pbs \
  scripts/grpo/phase11_g16_lr5e6_clip02_train_0000_0010.pbs \
  scripts/grpo/phase11_g16_lr5e6_clip02_train_0010_0020_resume.pbs \
  scripts/grpo/phase11_batched_logprob_smoke_u2.pbs; do
  bash -n "$f"
done
```

Expected: exit 0.

- [ ] **Step 3: Run a tiny sampler self-check**

Run:

```bash
tmp_dir="$(mktemp -d)"
python scripts/grpo/sample_process_tree_memory.py \
  --root-pid "$$" \
  --output "${tmp_dir}/process_tree_memory.csv" \
  --interval 1 \
  --label self &
sampler_pid=$!
sleep 2
kill "$sampler_pid"
wait "$sampler_pid" || true
python scripts/grpo/summarize_process_tree_memory.py \
  --csv "${tmp_dir}/process_tree_memory.csv" \
  --output "${tmp_dir}/process_tree_memory_summary.json"
python - <<PY
import json
from pathlib import Path
summary = json.loads(Path("${tmp_dir}/process_tree_memory_summary.json").read_text())
assert summary["sample_count"] >= 1
assert summary["max_tree_rss_kb"] > 0
print("CPU_MEM_SAMPLER_SELF_CHECK_OK", summary["sample_count"])
PY
```

Expected: prints `CPU_MEM_SAMPLER_SELF_CHECK_OK`.

- [ ] **Step 4: Commit verification-only fixes if needed**

If Step 1-3 reveal syntax or import issues, fix only the changed telemetry files and commit:

```bash
git add src/smolvla_grpo/process_memory.py \
  scripts/grpo/train_phase11_env_on_policy_grpo.py \
  scripts/grpo/sample_process_tree_memory.py \
  scripts/grpo/summarize_process_tree_memory.py \
  scripts/grpo/phase11*.pbs \
  tests/test_process_memory.py \
  tests/test_phase11_true_action_chunking.py \
  tests/test_phase11_slurm_scripts.py
git commit -m "$(cat <<'EOF'
fix(grpo): stabilize CPU memory telemetry

EOF
)"
```

Expected: no commit if no fixes were needed.

---

## Expected Artifacts

Trainer `progress.jsonl` rows get fields like:

```json
{
  "proc_mem_update_start_tree_rss_kb": 12345678,
  "proc_mem_after_rollout_tree_rss_kb": 56789012,
  "proc_mem_after_optimize_tree_rss_kb": 67890123,
  "proc_mem_after_optimize_tree_vmsize_kb": 98765432,
  "proc_mem_after_optimize_descendant_count": 128
}
```

PBS run directories get:

```text
artifacts/<run>/cpu_mem_telemetry/train/process_tree_memory.csv
artifacts/<run>/cpu_mem_telemetry/train/process_tree_memory.log
artifacts/<run>/cpu_mem_telemetry/train/process_tree_memory.err
artifacts/<run>/cpu_mem_telemetry/train/process_tree_memory_summary.json
```

GPU telemetry remains unchanged:

```text
artifacts/<run>/gpu_telemetry/train/nvidia_smi_samples.csv
artifacts/<run>/gpu_telemetry/train/gpu_telemetry_summary.json
```

## Reading Results

- If `proc_mem_update_start_tree_rss_kb` increases update-to-update, memory persists across updates.
- If `proc_mem_after_rollout_tree_rss_kb` jumps then returns down by next update start, rollout has transient peak memory.
- If CSV `tree_rss_kb` rises until PBS kills the job, the sampler confirms cgroup OOM trend.
- If `descendant_count` grows each update, env workers are leaking or not closing.
- If `self_rss_kb` grows but `descendant_count` stays stable, trainer/model-side Python objects are accumulating.

## Self-Review

- Spec coverage: trainer per-update memory fields, PBS background sampler, end-of-job summary, no `nvidia-smi` misuse, and PBS cgroup-friendly evidence are all covered.
- Placeholder scan: no `TBD`, no deferred edge handling, no unnamed test work.
- Type consistency: helper returns flat `dict[str, int]`; trainer spreads those fields; sampler writes the same keys to CSV; summarizer reads the same CSV columns.

