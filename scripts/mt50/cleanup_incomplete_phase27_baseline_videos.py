#!/usr/bin/env python3
"""Scan Phase27 baseline artifact tree (old layout) and remove incomplete video dirs.

Old layout: ``<parent>/shard_<i>_<n>tasks/videos/<task>_0/eval_episode_*.mp4``.

Only shard dirs matching ``^shard_\\d+_\\d+tasks$`` are scanned (not ``shard_recovery_*``).

By default writes ``incomplete_video_tasks.md`` under ``--parent`` and ``rm -rf``\s each
``videos/<task>_0`` dir with fewer than ``--expected`` MP4s (only if that dir exists).

Use ``--dry-run`` to report without deleting.
"""

import argparse
import re
import shutil
import sys
from collections import namedtuple
from pathlib import Path
from typing import List, Optional

# Canonical MT50 task order (must match submit_mt50_phase27_smolvla_baseline_25ep_s1000_4gpu_rtx6000.pbs).
PHASE27_MT50_TASKS = (
    "assembly-v3",
    "basketball-v3",
    "bin-picking-v3",
    "box-close-v3",
    "button-press-topdown-v3",
    "button-press-topdown-wall-v3",
    "button-press-v3",
    "button-press-wall-v3",
    "coffee-button-v3",
    "coffee-pull-v3",
    "coffee-push-v3",
    "dial-turn-v3",
    "disassemble-v3",
    "door-close-v3",
    "door-lock-v3",
    "door-open-v3",
    "door-unlock-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "faucet-close-v3",
    "faucet-open-v3",
    "hammer-v3",
    "hand-insert-v3",
    "handle-press-side-v3",
    "handle-press-v3",
    "handle-pull-side-v3",
    "handle-pull-v3",
    "lever-pull-v3",
    "peg-insert-side-v3",
    "peg-unplug-side-v3",
    "pick-out-of-hole-v3",
    "pick-place-v3",
    "pick-place-wall-v3",
    "plate-slide-back-side-v3",
    "plate-slide-back-v3",
    "plate-slide-side-v3",
    "plate-slide-v3",
    "push-back-v3",
    "push-v3",
    "push-wall-v3",
    "reach-v3",
    "reach-wall-v3",
    "shelf-place-v3",
    "soccer-v3",
    "stick-pull-v3",
    "stick-push-v3",
    "sweep-into-v3",
    "sweep-v3",
    "window-close-v3",
    "window-open-v3",
)

SHARD_DIR_RE = re.compile(r"^shard_\d+_\d+tasks$")


TaskVideoInfo = namedtuple(
    "TaskVideoInfo",
    ["task", "task_index", "shard_dirname", "video_task_dir", "n_videos", "dir_exists"],
)


def task_index_to_baseline_shard(parent: Path, task_index: int, worker_count: int = 4):
    """Return (shard_dirname, path to .../videos/<task>_0) for original 4-GPU baseline split."""
    tasks = PHASE27_MT50_TASKS
    n = len(tasks)
    base = n // worker_count
    remainder = n % worker_count
    start = 0
    for idx in range(worker_count):
        count = base + (1 if idx < remainder else 0)
        if start <= task_index < start + count:
            shard_name = f"shard_{idx}_{count}tasks"
            tname = tasks[task_index]
            video_dir = parent / shard_name / "videos" / f"{tname}_0"
            return shard_name, video_dir
        start += count
    raise IndexError(f"task_index {task_index} out of range for {n} tasks")


def count_eval_mp4s(video_task_dir: Path) -> int:
    if not video_task_dir.is_dir():
        return 0
    return sum(1 for p in video_task_dir.glob("eval_episode_*.mp4") if p.is_file())


def collect_task_infos(parent: Path, expected: int) -> List[TaskVideoInfo]:
    out: List[TaskVideoInfo] = []
    for i, task in enumerate(PHASE27_MT50_TASKS):
        shard_name, video_dir = task_index_to_baseline_shard(parent, i)
        exists = video_dir.is_dir()
        n = count_eval_mp4s(video_dir) if exists else 0
        out.append(
            TaskVideoInfo(task, i, shard_name, video_dir, n, exists)
        )
    return out


def list_legacy_shard_dirs(parent: Path) -> List[Path]:
    if not parent.is_dir():
        return []
    return sorted(p for p in parent.iterdir() if p.is_dir() and SHARD_DIR_RE.match(p.name))


def write_markdown_report(
    path: Path,
    *,
    parent: Path,
    expected: int,
    infos: List[TaskVideoInfo],
    incomplete: List[TaskVideoInfo],
    deleted_paths: List[Path],
    dry_run: bool,
) -> None:
    complete = [x for x in infos if x.n_videos >= expected]
    lines: List[str] = [
        "# Phase27 baseline: incomplete video task dirs",
        "",
        f"- **Parent**: `{parent}`",
        f"- **Expected** `eval_episode_*.mp4` per task: **{expected}**",
        f"- **Dry run**: **{dry_run}**",
        "",
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|--------|------:|",
        f"| Total canonical tasks | {len(infos)} |",
        f"| Complete (>= {expected} videos) | {len(complete)} |",
        f"| Incomplete (< {expected} videos) | {len(incomplete)} |",
        f"| Dirs removed (`rm -rf`) | {len(deleted_paths)} |",
        "",
        f"## Incomplete tasks (< {expected} videos)",
        "",
        "| Task | Shard | Videos | Dir exists | Path |",
        "|------|-------|-------:|:----------:|------|",
    ]
    for x in incomplete:
        # After optional rmtree, reflect filesystem (scan-time dir_exists can be stale).
        exists_now = x.video_task_dir.is_dir()
        exists_s = "yes" if exists_now else "no"
        lines.append(
            f"| `{x.task}` | `{x.shard_dirname}` | {x.n_videos} | {exists_s} | `{x.video_task_dir}` |"
        )
    lines.extend(
        [
            "",
            "## Complete tasks",
            "",
            f"| Task | Shard | Videos |",
            f"|------|-------|-------:|",
        ]
    )
    for x in sorted(complete, key=lambda z: z.task_index):
        lines.append(f"| `{x.task}` | `{x.shard_dirname}` | {x.n_videos} |")
    lines.extend(["", "## Legacy shard dirs seen", ""])
    shards = list_legacy_shard_dirs(parent)
    if not shards:
        lines.append("_(none matching `shard_<n>_<m>tasks`)_")
    else:
        for s in shards:
            lines.append(f"- `{s.name}`")
    lines.append("")
    if deleted_paths:
        lines.extend(["## Removed paths", ""])
        for p in deleted_paths:
            lines.append(f"- `{p}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    default_parent = (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "MT50_Phase27_smolvla_baseline_official_lerobot_25ep_s1000_4gpu_rtx6000"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parent",
        type=Path,
        default=default_parent,
        help=f"Baseline artifact parent (default: {default_parent})",
    )
    p.add_argument("--expected", type=int, default=25, help="Minimum MP4 count to keep dir (default: 25)")
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Markdown report path (default: <parent>/incomplete_video_tasks.md)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write report only; do not delete directories",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    parent: Path = args.parent.resolve()
    expected: int = args.expected
    report_path = args.report if args.report is not None else parent / "incomplete_video_tasks.md"

    if not parent.is_dir():
        print(f"error: parent not a directory: {parent}", file=sys.stderr)
        return 2

    infos = collect_task_infos(parent, expected)
    incomplete = [x for x in infos if x.n_videos < expected]

    deleted: List[Path] = []
    if not args.dry_run:
        for x in incomplete:
            if x.dir_exists and x.video_task_dir.is_dir():
                shutil.rmtree(x.video_task_dir)
                deleted.append(x.video_task_dir)

    write_markdown_report(
        report_path,
        parent=parent,
        expected=expected,
        infos=infos,
        incomplete=incomplete,
        deleted_paths=deleted,
        dry_run=bool(args.dry_run),
    )
    print(f"wrote {report_path}")
    if args.dry_run:
        print(f"dry-run: would remove {sum(1 for x in incomplete if x.dir_exists)} existing dirs")
    else:
        print(f"removed {len(deleted)} dirs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
