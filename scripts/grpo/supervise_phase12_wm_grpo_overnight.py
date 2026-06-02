#!/usr/bin/env python3
"""Submit/diagnose/resume Phase12 WM-GRPO overnight PBS runs.

This script is deliberately conservative: it can resubmit known-safe resume
cases, but it stops on unknown failures and writes a root-cause ledger for the
agent/human to inspect.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_ROOT / "artifacts" / "phase12_wm_g8_u20_strict_parity_20260602"
STRICT_PBS = PROJECT_ROOT / "scripts" / "grpo" / "phase12_g8_u20_wm_train_eval100_stride5.pbs"
EVAL_PBS = PROJECT_ROOT / "scripts" / "grpo" / "phase12_eval_last5_25ep.pbs"
MAX_AUTO_RESUMES_PER_KEY = 1


@dataclass
class SupervisorState:
    run_dir: str
    phase: str = "init"
    job_ids: list[str] = field(default_factory=list)
    latest_checkpoint: str | None = None
    failures: list[dict[str, Any]] = field(default_factory=list)
    resume_attempts: dict[str, int] = field(default_factory=dict)
    last_action: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_path(run_dir: Path) -> Path:
    return run_dir / "overnight_supervisor_state.json"


def ledger_path(run_dir: Path) -> Path:
    return run_dir / "overnight_root_cause_ledger.md"


def load_state(run_dir: Path) -> SupervisorState:
    path = state_path(run_dir)
    if not path.is_file():
        return SupervisorState(run_dir=str(run_dir))
    return SupervisorState(**json.loads(path.read_text(encoding="utf-8")))


def save_state(state: SupervisorState) -> None:
    run_dir = Path(state.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path(run_dir).write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def append_ledger(run_dir: Path, *, job_id: str | None, symptom: str, root_cause: str, action: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = ledger_path(run_dir)
    if not path.exists():
        path.write_text("# Phase12 WM-GRPO Overnight Root-Cause Ledger\n\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"## {utc_now()}\n\n")
        f.write(f"- job_id: `{job_id or ''}`\n")
        f.write(f"- symptom: {symptom}\n")
        f.write(f"- root_cause: `{root_cause}`\n")
        f.write(f"- action: {action}\n\n")


def run_cmd(args: list[str], *, cwd: Path = PROJECT_ROOT, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(cwd), text=True, capture_output=True, check=check)


def submit_strict(run_dir: Path, *, resume: Path | None = None, start_update: int | None = None) -> str:
    env = f"PHASE12_RUN_DIR={run_dir}"
    if resume is not None:
        env += f",PHASE12_RESUME={resume}"
    if start_update is not None:
        remaining = max(20 - int(start_update), 0)
        env += f",PHASE12_START_UPDATE={int(start_update)},PHASE12_NUM_UPDATES={remaining}"
    cmd = ["qsub", "-q", "v1_gpu72", "-v", env, str(STRICT_PBS)]
    proc = run_cmd(cmd)
    return proc.stdout.strip().split()[0]


def submit_eval100(run_dir: Path) -> str:
    env = ",".join(
        [
            f"PHASE12_RUN_DIR={run_dir}",
            "PHASE12_MIN_UPDATE=5",
            "PHASE12_MAX_UPDATE=20",
            "PHASE12_STRIDE=5",
            "PHASE12_EVAL_EPISODES=100",
            "PHASE12_EVAL_N_ENVS=25",
            "PHASE12_EVAL_CHUNK_LEN=5",
            "PHASE12_EVAL_MAX_STEPS=120",
            "PHASE12_SWEEP_NAME=eval100_u0005_0020_stride5_nenv25_async",
        ]
    )
    proc = run_cmd(["qsub", "-q", "v1_gpu72", "-v", env, str(EVAL_PBS)])
    return proc.stdout.strip().split()[0]


def qstat_visible(job_id: str) -> bool:
    proc = run_cmd(["qstat", job_id], check=False)
    return proc.returncode == 0


def _read_checkpoint_meta(path: Path) -> int | None:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return int(meta["update_index"]) + 1
    except Exception:
        return None


def latest_checkpoint(run_dir: Path) -> tuple[Path | None, int | None]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None, None
    best: tuple[int, Path] | None = None
    for path in ckpt_dir.glob("update_*.pt"):
        match = re.search(r"update_(\d{4})\.pt$", path.name)
        if match is None:
            continue
        idx = int(match.group(1))
        if best is None or idx > best[0]:
            best = (idx, path)
    latest = ckpt_dir / "latest.pt"
    latest_start = _read_checkpoint_meta(latest)
    if latest.is_file() and latest_start is not None and (best is None or latest_start > best[0]):
        best = (latest_start, latest)
    if best is None:
        return None, None
    return best[1], best[0]


def read_recent_logs() -> str:
    log_dir = PROJECT_ROOT / "logs" / "pbs" / "grpo"
    if not log_dir.is_dir():
        return ""
    files = sorted(log_dir.glob("phase12*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
    chunks: list[str] = []
    for path in files:
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace")[-12000:])
        except OSError:
            continue
    return "\n".join(chunks)


def has_walltime_evidence(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "walltime",
            "wall time",
            "time limit",
            "timelimit",
            "exceeded limit",
            "resources_used.walltime",
        )
    )


def classify_failure(run_dir: Path) -> tuple[str, str]:
    text = read_recent_logs()
    progress = run_dir / "progress.jsonl"
    if "CUDA out of memory" in text or "OutOfMemoryError" in text:
        return "oom_or_cuda_memory", "CUDA OOM in recent Phase12 logs."
    if "Missing --jepa-repo" in text or "Failed to load WM" in text or "jepa" in text.lower() and "not found" in text.lower():
        return "wm_load_failure", "JEPA-WM repo/checkpoint failed to load."
    if "mujoco" in text.lower() or "egl" in text.lower():
        return "mujoco_egl_failure", "MuJoCo/EGL initialization failed."
    if "No space left on device" in text or "Permission denied" in text:
        return "hf_cache_failure", "Cache/path write failure."
    if "nan" in text.lower() or "inf" in text.lower():
        return "nan_or_inf_loss", "NaN/Inf appeared in train logs."
    if progress.is_file():
        try:
            rows = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines() if line.strip()]
            if rows and rows[-1].get("event") == "configuration_error":
                return "eval_cli_failure", str(rows[-1].get("reason", "configuration_error"))
        except Exception:
            pass
    ckpt, update = latest_checkpoint(run_dir)
    if ckpt is not None and update is not None and update < 20 and has_walltime_evidence(text):
        return "pbs_walltime_timeout", f"Latest checkpoint update_{update:04d}.pt exists before target update_0020.pt."
    if ckpt is not None and update is not None and update < 20:
        return (
            "partial_checkpoint_no_walltime_evidence",
            f"Latest checkpoint can resume at update {update}, but logs lack walltime evidence.",
        )
    if ckpt is None:
        return "missing_checkpoint", "No Phase12 checkpoint found."
    return "unknown", "No known root-cause signature found."


def _attempt_count(state: SupervisorState, key: str) -> int:
    return int(state.resume_attempts.get(key, 0))


def _record_attempt(state: SupervisorState, key: str) -> None:
    state.resume_attempts[key] = _attempt_count(state, key) + 1


def handle_once(run_dir: Path, *, auto_resume: bool) -> SupervisorState:
    state = load_state(run_dir)
    active = [job_id for job_id in state.job_ids if qstat_visible(job_id)]
    if active:
        state.phase = "running"
        state.last_action = f"jobs still visible: {','.join(active)}"
        save_state(state)
        return state

    final_ckpt = run_dir / "checkpoints" / "update_0020.pt"
    eval100 = run_dir / "eval100_u0005_0020_stride5_nenv25_async" / "eval_sweep_summary.json"
    if final_ckpt.is_file() and eval100.is_file():
        state.phase = "complete"
        state.latest_checkpoint = str(final_ckpt)
        state.last_action = "train+eval100 complete"
        save_state(state)
        return state
    if final_ckpt.is_file() and not eval100.is_file():
        key = f"eval100:{final_ckpt}"
        if auto_resume and _attempt_count(state, key) < MAX_AUTO_RESUMES_PER_KEY:
            job_id = submit_eval100(run_dir)
            _record_attempt(state, key)
            state.job_ids.append(job_id)
            state.phase = "resubmitted"
            state.latest_checkpoint = str(final_ckpt)
            state.last_action = f"submitted eval100-only job={job_id}"
            append_ledger(
                run_dir,
                job_id=job_id,
                symptom="Training reached update_0020.pt but eval100 summary is missing.",
                root_cause="eval_cli_failure",
                action="auto-submit eval100-only recovery",
            )
            save_state(state)
            return state
        state.phase = "blocked"
        state.latest_checkpoint = str(final_ckpt)
        state.failures.append(
            {
                "created_at": utc_now(),
                "root_cause": "eval_cli_failure",
                "symptom": "Training reached update_0020.pt but eval100 summary is missing.",
            }
        )
        state.last_action = "blocked: eval100 missing after final checkpoint"
        append_ledger(
            run_dir,
            job_id=state.job_ids[-1] if state.job_ids else None,
            symptom="Training reached update_0020.pt but eval100 summary is missing.",
            root_cause="eval_cli_failure",
            action="diagnosed",
        )
        save_state(state)
        return state

    cause, symptom = classify_failure(run_dir)
    state.failures.append({"created_at": utc_now(), "root_cause": cause, "symptom": symptom})
    ckpt, update = latest_checkpoint(run_dir)
    state.latest_checkpoint = str(ckpt) if ckpt is not None else None
    append_ledger(run_dir, job_id=state.job_ids[-1] if state.job_ids else None, symptom=symptom, root_cause=cause, action="diagnosed")

    key = f"train:{ckpt}:{update}" if ckpt is not None else ""
    if (
        auto_resume
        and cause == "pbs_walltime_timeout"
        and ckpt is not None
        and update is not None
        and update < 20
        and _attempt_count(state, key) < MAX_AUTO_RESUMES_PER_KEY
    ):
        next_update = int(update)
        job_id = submit_strict(run_dir, resume=ckpt, start_update=next_update)
        _record_attempt(state, key)
        state.job_ids.append(job_id)
        state.phase = "resubmitted"
        state.last_action = f"resubmitted from {ckpt} start_update={next_update} job={job_id}"
        append_ledger(
            run_dir,
            job_id=job_id,
            symptom=symptom,
            root_cause=cause,
            action=f"auto-resume from `{ckpt}` at start_update={next_update}",
        )
    else:
        state.phase = "blocked"
        state.last_action = f"blocked: {cause}"
    save_state(state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--submit", action="store_true", help="Submit strict G8-u20 WM run.")
    parser.add_argument("--status", action="store_true", help="Print current supervisor state.")
    parser.add_argument("--auto-resume", action="store_true", help="Auto-resume known-safe timeout failures.")
    parser.add_argument("--loop", action="store_true", help="Poll until complete/blocked.")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(run_dir)

    if args.submit:
        job_id = submit_strict(run_dir)
        state.job_ids.append(job_id)
        state.phase = "submitted"
        state.last_action = f"submitted strict G8-u20 WM run job={job_id}"
        save_state(state)
        append_ledger(run_dir, job_id=job_id, symptom="initial_submit", root_cause="none", action="submitted strict run")
        print(json.dumps(asdict(state), indent=2))
        return 0

    if args.status:
        print(json.dumps(asdict(state), indent=2))
        return 0

    while True:
        state = handle_once(run_dir, auto_resume=bool(args.auto_resume))
        print(json.dumps(asdict(state), indent=2))
        if not args.loop or state.phase in {"complete", "blocked"}:
            return 0 if state.phase != "blocked" else 2
        time.sleep(max(int(args.poll_seconds), 30))


if __name__ == "__main__":
    raise SystemExit(main())
