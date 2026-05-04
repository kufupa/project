#!/usr/bin/env python3
"""Summarize SmolVLA parity Slurm jobs: queue state, stdout run dir, live x/N progress."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def _run_squeue(job_ids: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["squeue", "-j", ",".join(job_ids), "-o", "%.10i %.20j %.2t %.10M %.20R"],
            check=False,
            capture_output=True,
            text=True,
        )
        return (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return "(squeue not found)\n"


def _parse_run_dir(stdout_path: Path) -> str | None:
    if not stdout_path.is_file():
        return None
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^parity benchmark run dir:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _count_completed_episodes(run_dir: Path) -> tuple[int, int, bool]:
    """Return (n_meta, n_videos, manifest_exists)."""
    ep = run_dir / "episodes"
    meta = 0
    if ep.is_dir():
        meta = sum(1 for p in ep.glob("episode_*/episode_meta.json") if p.is_file())
    vroot = run_dir / "videos"
    videos = 0
    if vroot.is_dir():
        videos = sum(1 for p in vroot.rglob("*.mp4") if p.is_file())
    manifest = (run_dir / "run_manifest.json").is_file()
    return meta, videos, manifest


def _tail_progress_jsonl(run_dir: Path, n: int) -> list[dict]:
    path = run_dir / "progress.jsonl"
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[dict] = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append({"raw": line})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "job_ids",
        nargs="+",
        help="Slurm job IDs (e.g. 236688 236689)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repo project/ root (default: parent of scripts/smolvla)",
    )
    parser.add_argument(
        "--stdout-pattern",
        default="smolvla_parity_eval_{job}.out",
        help="Stdout filename pattern under project root; {job} is replaced",
    )
    parser.add_argument(
        "--progress-tail",
        type=int,
        default=2,
        help="How many progress.jsonl lines to show per job when present",
    )
    args = parser.parse_args()
    root: Path = args.project_root

    print(_run_squeue(args.job_ids))
    for job in args.job_ids:
        pat = args.stdout_pattern.format(job=job)
        out_path = root / pat if not Path(pat).is_absolute() else Path(pat)
        run_dir_s = _parse_run_dir(out_path)
        print(f"--- job {job} ---")
        print(f"stdout: {out_path} exists={out_path.is_file()}")
        if not run_dir_s:
            print("run_dir: (not found in stdout yet)")
            continue
        rd = Path(run_dir_s)
        print(f"run_dir: {rd}")
        meta, videos, manifest = _count_completed_episodes(rd)
        total = None
        if manifest:
            try:
                man = json.loads((rd / "run_manifest.json").read_text(encoding="utf-8"))
                total = int(man.get("episodes_requested", 0) or 0)
            except (OSError, json.JSONDecodeError):
                total = None
        if total and total > 0:
            print(f"progress: {meta}/{total} (episode_meta.json), videos={videos}, manifest={manifest}")
        else:
            print(f"progress: episode_meta={meta}, videos={videos}, manifest={manifest}")
        tail = _tail_progress_jsonl(rd, args.progress_tail)
        if tail:
            print("progress.jsonl (tail):")
            for row in tail:
                print(" ", json.dumps(row, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
