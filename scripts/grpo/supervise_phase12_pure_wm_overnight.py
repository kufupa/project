#!/usr/bin/env python3
"""Conservative supervisor for Phase12 pure-WM train/eval PBS runs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_EVAL_PBS = PROJECT_ROOT / "scripts" / "grpo" / "phase12_pure_wm_teacher_forced_train_eval.pbs"


@dataclass
class SupervisorState:
    run_dir: str
    phase: str
    job_ids: list[str]
    failures: list[dict[str, Any]]
    last_action: str


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _latest_checkpoint(run_dir: Path) -> tuple[Path | None, int | None]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None, None
    best: tuple[int, Path] | None = None
    for path in ckpt_dir.glob("update_*.pt"):
        match = re.match(r"update_(\d{4})\.pt$", path.name)
        if match is None:
            continue
        update = int(match.group(1))
        if best is None or update > best[0]:
            best = (update, path)
    if best is None:
        latest = ckpt_dir / "latest.pt"
        if latest.is_file():
            return latest, None
        return None, None
    return best[1], best[0]


def diagnose_run(
    run_dir: Path,
    *,
    expected_update: int = 20,
    expected_sweep: str = "eval100_explicit_u5_20_nenv25_async",
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    expected_ckpt = run_dir / "checkpoints" / f"update_{int(expected_update):04d}.pt"
    expected_eval = run_dir / expected_sweep / "eval_sweep_summary.json"
    manifest_exists = (run_dir / "manifest.json").is_file()
    progress_exists = (run_dir / "progress.jsonl").is_file()
    latest, latest_update = _latest_checkpoint(run_dir)

    if expected_ckpt.is_file() and expected_eval.is_file():
        phase = "complete"
        known_safe = True
    elif expected_ckpt.is_file():
        phase = "needs_eval_resume"
        known_safe = True
    elif latest is not None and latest_update is not None and latest_update < int(expected_update):
        phase = "needs_train_resume"
        known_safe = True
    elif latest is not None:
        phase = "blocked_unknown_checkpoint_state"
        known_safe = False
    elif manifest_exists or progress_exists:
        phase = "blocked_no_checkpoint"
        known_safe = False
    else:
        phase = "not_started"
        known_safe = True

    return {
        "run_dir": str(run_dir),
        "phase": phase,
        "known_safe": bool(known_safe),
        "latest_checkpoint": "" if latest is None else str(latest),
        "latest_update": latest_update,
        "expected_checkpoint": str(expected_ckpt),
        "expected_eval_summary": str(expected_eval),
        "manifest_exists": manifest_exists,
        "progress_exists": progress_exists,
    }


def qsub_train_eval(
    run_dir: Path,
    *,
    root_mode: str,
    loss_mode: str,
    action_l2: float,
    lr: str,
    clip_eps: str,
    init_log_std: str,
    euler_noise: str,
    resume: Path | None = None,
    start_update: int | None = None,
    expected_update: int = 20,
) -> str:
    var_items = [
        f"PHASE12_RUN_DIR={run_dir}",
        f"PHASE12_WM_ONLY_ROOT_MODE={root_mode}",
        f"PHASE12_LOSS_NORMALIZER_MODE={loss_mode}",
        f"PHASE12_WM_ACTION_L2_PENALTY={float(action_l2)}",
        f"PHASE12_LR={lr}",
        f"PHASE12_CLIP_EPS={clip_eps}",
        f"PHASE12_INIT_LOG_STD={init_log_std}",
        f"PHASE12_EULER_NOISE={euler_noise}",
        f"PHASE12_TOTAL_UPDATES={int(expected_update)}",
    ]
    if resume is not None and start_update is not None:
        var_items.extend(
            [
                f"PHASE12_RESUME={resume}",
                f"PHASE12_START_UPDATE={int(start_update)}",
                f"PHASE12_NUM_UPDATES={max(int(expected_update) - int(start_update), 0)}",
            ]
        )
    result = subprocess.run(
        ["qsub", "-v", ",".join(var_items), str(TRAIN_EVAL_PBS)],
        cwd=str(PROJECT_ROOT),
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip().split()[0]


def _supervise_once(args: argparse.Namespace, job_ids: list[str]) -> tuple[SupervisorState, list[str]]:
    diagnosis = diagnose_run(args.run_dir, expected_update=args.expected_update, expected_sweep=args.expected_sweep)
    _write_jsonl(args.run_dir / "overnight_root_cause_ledger.jsonl", {"event": "diagnosis", **diagnosis})
    action = "diagnosed"
    failures: list[dict[str, Any]] = []
    if args.auto_resume and diagnosis["known_safe"] and diagnosis["phase"] in {"not_started", "needs_eval_resume", "needs_train_resume"}:
        resume = Path(diagnosis["latest_checkpoint"]) if diagnosis["phase"] == "needs_train_resume" else None
        start_update = int(diagnosis["latest_update"]) if diagnosis["phase"] == "needs_train_resume" else None
        job_ids.append(
            qsub_train_eval(
                args.run_dir,
                root_mode=args.root_mode,
                loss_mode=args.loss_mode,
                action_l2=float(args.action_l2),
                lr=args.lr,
                clip_eps=args.clip_eps,
                init_log_std=args.init_log_std,
                euler_noise=args.euler_noise,
                resume=resume,
                start_update=start_update,
                expected_update=int(args.expected_update),
            )
        )
        action = f"submitted_{diagnosis['phase']}"
    elif not diagnosis["known_safe"]:
        failures.append(diagnosis)

    state = SupervisorState(
        run_dir=str(args.run_dir),
        phase=str(diagnosis["phase"]),
        job_ids=job_ids,
        failures=failures,
        last_action=action,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "overnight_supervisor_state.json").write_text(
        json.dumps(asdict(state), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return state, job_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--root-mode", default="oracle_teacher_forced")
    parser.add_argument("--loss-mode", default="group")
    parser.add_argument("--action-l2", type=float, default=0.003)
    parser.add_argument("--lr", default="1e-5")
    parser.add_argument("--clip-eps", default="0.2")
    parser.add_argument("--init-log-std", default="-2.0")
    parser.add_argument("--euler-noise", default="0.2")
    parser.add_argument("--expected-update", type=int, default=20)
    parser.add_argument("--expected-sweep", default="eval100_explicit_u5_20_nenv25_async")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--sleep-seconds", "--poll-seconds", dest="sleep_seconds", type=int, default=300)
    args = parser.parse_args()

    job_ids: list[str] = []
    while True:
        state, job_ids = _supervise_once(args, job_ids)
        if not args.loop or state.phase == "complete" or state.failures:
            break
        time.sleep(int(args.sleep_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
