#!/usr/bin/env python3
"""Poll Phase46 SLURM jobs, verify markers, RCA on failure, write morning report."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STAGE_MARKERS: dict[str, str] = {
    "smoke": "PHASE46_SMOKE_OK",
    "train": "PHASE46_TRAIN_20UPD_OK",
    "eval": "PHASE46_TIERED_EVAL_RLINF_OK",
}

TERMINAL_STATES = frozenset(
    {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY"}
)


@dataclass
class SacctRow:
    job_id: str
    state: str
    exit_code: str


def append_manifest(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def poll_sacct(job_id: str) -> SacctRow | None:
    proc = subprocess.run(
        ["sacct", "-j", job_id, "-X", "-n", "-o", "JobID,State,ExitCode"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    line = proc.stdout.strip().splitlines()[-1]
    parts = line.split()
    if len(parts) < 3:
        return None
    return SacctRow(job_id=parts[0], state=parts[1], exit_code=parts[2])


def find_slurm_log(job_id: str, roots: list[Path]) -> Path | None:
    patterns = [f"*{job_id}*.out", f"*_{job_id}.out", f"phase46_*_{job_id}.out"]
    for root in roots:
        if not root.is_dir():
            continue
        for pat in patterns:
            hits = sorted(root.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
            if hits:
                return hits[0]
        for pat in patterns:
            hits = sorted(root.rglob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
            if hits:
                return hits[0]
    return None


def log_contains_marker(log_path: Path | None, marker: str) -> bool:
    if log_path is None or not log_path.is_file():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return marker in text


def write_incident(
    run_root: Path,
    job_id: str,
    stage: str,
    sacct: SacctRow | None,
    log_path: Path | None,
    hypothesis: str,
) -> Path:
    inc_dir = run_root / "incidents"
    inc_dir.mkdir(parents=True, exist_ok=True)
    out = inc_dir / f"{job_id}.md"
    tail = ""
    if log_path and log_path.is_file():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-50:])
    body = (
        f"# Incident job {job_id} stage={stage}\n\n"
        f"## Symptom\n"
        f"- sacct: {sacct}\n"
        f"- log: {log_path}\n\n"
        f"## Hypothesis\n{hypothesis}\n\n"
        f"## Log tail\n```\n{tail}\n```\n"
    )
    out.write_text(body, encoding="utf-8")
    return out


def verify_stage_artifacts(stage: str, manifest_row: dict[str, Any]) -> str | None:
    if stage == "train":
        out = Path(manifest_row.get("train_out", ""))
        ckpt = out / "checkpoints" / "update_0020.pt"
        if not ckpt.is_file():
            return f"missing {ckpt}"
        prog = out / "progress.jsonl"
        if prog.is_file():
            n = sum(1 for ln in prog.read_text().splitlines() if ln.strip())
            if n < 2:
                return f"progress.jsonl rows={n}"
    if stage == "eval":
        eval_out = Path(manifest_row.get("eval_out", ""))
        summary = eval_out / "tiered_eval_summary.json"
        if not summary.is_file():
            return f"missing {summary}"
    return None


def follow_manifest(
    manifest_path: Path,
    *,
    interval_s: int,
    log_roots: list[Path],
    max_retries: int,
) -> int:
    run_root = manifest_path.parent
    pending = load_manifest(manifest_path)
    retries: dict[str, int] = {}

    while True:
        all_done = True
        for row in pending:
            jid = str(row.get("job_id", ""))
            stage = str(row.get("stage", ""))
            state = str(row.get("poll_state", "PENDING"))
            if state in {"OK", "FAILED_FINAL"}:
                continue
            all_done = False
            sacct = poll_sacct(jid)
            if sacct is None or sacct.state not in TERMINAL_STATES:
                continue
            log_path = find_slurm_log(jid, log_roots)
            marker = STAGE_MARKERS.get(stage, "")
            ok_marker = log_contains_marker(log_path, marker) if marker else True
            exit_ok = sacct.exit_code.startswith("0:")
            art_err = verify_stage_artifacts(stage, row)

            if sacct.state == "COMPLETED" and exit_ok and ok_marker and art_err is None:
                row["poll_state"] = "OK"
                row["sacct_state"] = sacct.state
                print(f"phase46_autopilot OK job={jid} stage={stage}", flush=True)
                if stage == "eval":
                    eval_out = Path(str(row.get("eval_out", "")))
                    summary = eval_out / "tiered_eval_summary.json"
                    if summary.is_file():
                        import subprocess

                        gate = Path(__file__).resolve().parent / "phase46_gate.py"
                        subprocess.run(
                            [
                                "python3",
                                str(gate),
                                "--summary",
                                str(summary),
                                "--run-id",
                                run_root.name,
                                "--logprob-mode",
                                str(row.get("logprob_mode", "gaussian")),
                            ],
                            check=False,
                        )
                continue

            nretry = retries.get(jid, 0)
            hypothesis = (
                f"state={sacct.state} exit={sacct.exit_code} marker={ok_marker} art={art_err}"
            )
            write_incident(run_root, jid, stage, sacct, log_path, hypothesis)
            if nretry < max_retries:
                retries[jid] = nretry + 1
                row["poll_state"] = "RETRY_NEEDED"
                print(
                    f"phase46_autopilot RETRY job={jid} stage={stage} try={nretry + 1}",
                    flush=True,
                )
            else:
                row["poll_state"] = "FAILED_FINAL"
                print(f"phase46_autopilot FAIL job={jid} stage={stage}", flush=True)

        manifest_path.write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in pending) + "\n",
            encoding="utf-8",
        )
        if all_done:
            break
        time.sleep(interval_s)

    failed = [r for r in pending if r.get("poll_state") == "FAILED_FINAL"]
    return 1 if failed else 0


def write_morning_report(run_root: Path, manifest_path: Path) -> None:
    rows = load_manifest(manifest_path)
    lines = ["# Phase46 Morning Report\n"]
    for r in rows:
        lines.append(
            f"- stage={r.get('stage')} job={r.get('job_id')} state={r.get('poll_state')} "
            f"logprob_mode={r.get('logprob_mode', 'gaussian')}"
        )
    eval_row = next((r for r in rows if r.get("stage") == "eval" and r.get("poll_state") == "OK"), None)
    if eval_row:
        summary = Path(eval_row.get("eval_out", "")) / "tiered_eval_summary.json"
        if summary.is_file():
            data = json.loads(summary.read_text(encoding="utf-8"))
            lines.append("\n## 100ep milestones\n")
            for row in data.get("rows_100ep", []):
                sr = float(row.get("success_rate", 0.0))
                lines.append(
                    f"- {row.get('checkpoint', row.get('update'))}: "
                    f"{100.0 * sr:.1f}% ({row.get('success_count', '?')}/{row.get('num_episodes', '?')})"
                )
    (run_root / "MORNING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--follow", action="store_true")
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument(
        "--log-root",
        type=Path,
        action="append",
        default=[],
        help="Directories to search for slurm *.out (repeatable)",
    )
    args = p.parse_args()
    log_roots = list(args.log_root) or [
        Path("/vol/bitbucket/aa6622/project"),
        Path("/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/logs/slurm"),
        Path("/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo"),
    ]
    if args.follow:
        rc = follow_manifest(
            args.manifest,
            interval_s=args.interval,
            log_roots=log_roots,
            max_retries=args.max_retries,
        )
        write_morning_report(args.manifest.parent, args.manifest)
        return rc
    rows = load_manifest(args.manifest)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
