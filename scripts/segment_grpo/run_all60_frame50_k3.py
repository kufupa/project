#!/usr/bin/env python3
"""Campaign orchestrator: run scripts/run_segment_grpo.py once per episode.

MT10 / explicit oracle: when ``--oracle-run-root`` is set (e.g. from
``run_phase8_mt10.sh`` using ``mt10_phase6_index.json``), goal frames load only
from that run's ``frames/episode_*/`` tree; ``--task`` + episode index must match
that run. ``--goal-frame-index`` is a fixed PNG index, not a per-task success
time.

When ``--prefetch-run-root`` is set, each child loads
``<root>/out_episode_{campaign_offset:04d}.json``; the prior campaign must use the
same ``--episode-start`` / episode ordering so prefetch rows align with
``expected_episode_index`` validation in the child.

Phase-specific: edits no existing file, spawns a fresh Python process per
episode to guarantee memory isolation, handles skip/resume/failure policy.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from segment_grpo_reference import resolve_latest_oracle_pushv3_run  # noqa: E402
from smolvla_pipeline.run_layout import (  # noqa: E402
    effective_run_name_prefix_slug,
    slug_run_directory_prefix,
    slug_task,
)


@dataclass
class EpisodeOutcome:
    episode_offset: int
    target_episode_index: int
    reset_seed: int
    status: str  # "ok" | "resume_skip" | "missing_goal" | "missing_prefetch" | "failed"
    output_json: str | None
    stderr_tail: str | None
    wall_seconds: float


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segment-GRPO all-60 campaign orchestrator.")
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--episode-start", type=int, default=0)
    p.add_argument("--goal-frame-index", type=int, default=50)
    p.add_argument("--num-candidates", type=int, default=3)
    p.add_argument("--chunk-len", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--smolvla-n-action-steps", type=int, default=50)
    p.add_argument("--task", default="push-v3")
    p.add_argument("--carry-mode", default="sim", choices=["sim", "replay"])
    p.add_argument("--wm-rollout-mode", default="iterative", choices=["iterative", "batched"])
    p.add_argument("--wm-scoring-latent", default="visual", choices=["visual", "proprio", "concat"])
    p.add_argument(
        "--comparison-strip-overlay",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    p.add_argument("--checkpoint", default=os.environ.get("SMOLVLA_CHECKPOINT", ""))
    p.add_argument("--jepa-repo", default=os.environ.get("JEPA_REPO", ""))
    p.add_argument("--jepa-ckpt", default="jepa_wm_metaworld.pth.tar")
    p.add_argument("--artifacts-root", type=Path, default=ROOT / "artifacts")
    p.add_argument("--oracle-run-root", type=Path, default=None)
    p.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "phase08_segment_grpo_baseline",
    )
    p.add_argument("--run-name", default="")
    p.add_argument(
        "--run-name-prefix",
        default="",
        help="When --run-name is empty, prepend this slug to the auto-generated run_* name "
        "(overrides RUN_NAME_PREFIX / ORACLE_RUN_PREFIX for the default name only).",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--child-script", type=Path, default=ROOT / "scripts" / "run_segment_grpo.py")
    p.add_argument("--python", default=sys.executable)
    p.add_argument(
        "--per-episode-timeout",
        type=float,
        default=1800.0,
        help="Seconds; subprocess.run timeout per episode. Timeouts logged as failures.",
    )
    p.add_argument(
        "--prefetch-run-root",
        type=Path,
        default=None,
        help=(
            "Prior segment_grpo campaign run directory containing out_episode_XXXX.json per offset; "
            "passed through as --prefetch-candidates-json to each child."
        ),
    )
    p.add_argument("--prefetch-segment-index", type=int, default=0)
    p.add_argument(
        "--wm-selection-env-steps",
        type=int,
        default=None,
        help="Forwarded to run_segment_grpo.py --wm-selection-env-steps when set.",
    )
    return p.parse_args(argv)


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    name = args.run_name.strip()
    if not name:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = (
            f"run_{stamp}_all60_f{int(args.goal_frame_index)}"
            f"_k{int(args.num_candidates)}_s{int(args.seed_base)}"
            f"_t{slug_task(str(args.task))}"
        )
        cli_prefix = str(getattr(args, "run_name_prefix", "") or "").strip()
        if cli_prefix:
            pslug = slug_run_directory_prefix(cli_prefix)
        else:
            pslug = effective_run_name_prefix_slug()
        if pslug:
            name = f"{pslug}_{name}"
    run_dir = args.output_root.expanduser().resolve(strict=False) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_oracle_run(args: argparse.Namespace) -> Path:
    if args.oracle_run_root is not None:
        return Path(args.oracle_run_root).expanduser().resolve(strict=False)
    return resolve_latest_oracle_pushv3_run(args.artifacts_root, task=args.task)


def _goal_frame_path(oracle_run: Path, episode_index: int, goal_frame_index: int) -> Path:
    # Must match segment_grpo_reference.load_oracle_reference_frames:
    #   run_dir / "frames" / f"episode_{idx:04d}" / f"frame_{goal_idx:06d}.png"
    frames_dir = oracle_run / "frames" / f"episode_{int(episode_index):04d}"
    return frames_dir / f"frame_{int(goal_frame_index) - 1:06d}.png"


def _episode_json_path(run_dir: Path, episode_offset: int) -> Path:
    return run_dir / f"out_episode_{int(episode_offset):04d}.json"


def _build_child_argv(
    args: argparse.Namespace,
    episode_offset: int,
    target_episode: int,
    reset_seed: int,
    episode_json: Path,
    oracle_run: Path,
) -> list[str]:
    argv = [
        args.python,
        str(args.child_script),
        "--output-json",
        str(episode_json),
        "--flat-output",
        "--artifacts-root",
        str(args.artifacts_root),
        "--oracle-run-root",
        str(oracle_run),
        "--top15-report",
        "/dev/null",
        "--episodes",
        "1",
        "--episode-index",
        str(int(target_episode)),
        "--reset-seed",
        str(int(reset_seed)),
        "--goal-frame-index",
        str(int(args.goal_frame_index)),
        "--num-candidates",
        str(int(args.num_candidates)),
        "--chunk-len",
        str(int(args.chunk_len)),
        "--max-steps",
        str(int(args.max_steps)),
        "--smolvla-n-action-steps",
        str(int(args.smolvla_n_action_steps)),
        "--task",
        str(args.task),
        "--carry-mode",
        str(args.carry_mode),
        "--wm-rollout-mode",
        str(args.wm_rollout_mode),
        "--wm-scoring-latent",
        str(args.wm_scoring_latent),
        "--jepa-ckpt",
        str(args.jepa_ckpt),
    ]
    if args.checkpoint:
        argv += ["--checkpoint", str(args.checkpoint)]
    if args.jepa_repo:
        argv += ["--jepa-repo", str(args.jepa_repo)]
    if args.comparison_strip_overlay:
        argv += ["--comparison-strip-overlay"]
    else:
        argv += ["--no-comparison-strip-overlay"]
    if args.dry_run:
        argv += ["--dry-run"]
    wm_sel = getattr(args, "wm_selection_env_steps", None)
    if wm_sel is not None:
        argv += ["--wm-selection-env-steps", str(int(wm_sel))]
    prefetch_root = getattr(args, "prefetch_run_root", None)
    if prefetch_root is not None:
        prefetch_json = Path(prefetch_root).expanduser().resolve() / f"out_episode_{int(episode_offset):04d}.json"
        argv += [
            "--prefetch-candidates-json",
            str(prefetch_json),
            "--prefetch-segment-index",
            str(int(getattr(args, "prefetch_segment_index", 0))),
        ]
    return argv


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _tail(text: str, n_chars: int = 2000) -> str:
    if not text:
        return ""
    return text[-n_chars:]


def _timeout_stderr_tail(exc: subprocess.TimeoutExpired) -> str:
    err = exc.stderr
    if err is None:
        return ""
    if isinstance(err, bytes):
        return _tail(err.decode("utf-8", "replace"))
    return _tail(str(err))


def _run_episode(
    args: argparse.Namespace,
    run_dir: Path,
    oracle_run: Path,
    episodes_jsonl: Path,
    skipped_jsonl: Path,
    failed_jsonl: Path,
    i: int,
    total: int,
) -> EpisodeOutcome:
    target_episode = int(args.episode_start) + int(i)
    reset_seed = int(args.seed_base) + target_episode
    ep_json = _episode_json_path(run_dir, i)

    print(
        f"[campaign] {i + 1}/{total} episode=START offset={i} "
        f"target_episode={target_episode} reset_seed={reset_seed}",
        flush=True,
    )

    if ep_json.is_file():
        print(f"[campaign] {i + 1}/{total} RESUME-SKIP {ep_json}", flush=True)
        return EpisodeOutcome(i, target_episode, reset_seed, "resume_skip", str(ep_json), None, 0.0)

    goal_path = _goal_frame_path(oracle_run, target_episode, int(args.goal_frame_index))
    if not goal_path.is_file():
        record = {
            "episode_offset": i,
            "target_episode_index": target_episode,
            "reset_seed": reset_seed,
            "goal_frame_index": int(args.goal_frame_index),
            "reason": "goal_frame_missing",
            "expected_path": str(goal_path),
        }
        _append_jsonl(skipped_jsonl, record)
        print(
            f"[campaign] {i + 1}/{total} SKIP-MISSING-GOAL target_episode={target_episode} "
            f"expected={goal_path}",
            flush=True,
        )
        return EpisodeOutcome(i, target_episode, reset_seed, "missing_goal", None, None, 0.0)

    if getattr(args, "prefetch_run_root", None) is not None:
        prefetch_json = (
            Path(args.prefetch_run_root).expanduser().resolve() / f"out_episode_{int(i):04d}.json"
        )
        if not prefetch_json.is_file():
            record = {
                "episode_offset": i,
                "target_episode_index": target_episode,
                "reset_seed": reset_seed,
                "reason": "prefetch_json_missing",
                "expected_path": str(prefetch_json),
            }
            _append_jsonl(skipped_jsonl, record)
            print(
                f"[campaign] {i + 1}/{total} SKIP-MISSING-PREFETCH target_episode={target_episode} "
                f"expected={prefetch_json}",
                flush=True,
            )
            return EpisodeOutcome(i, target_episode, reset_seed, "missing_prefetch", None, None, 0.0)

    argv = _build_child_argv(args, i, target_episode, reset_seed, ep_json, oracle_run)
    t0 = time.time()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=float(args.per_episode_timeout),
        )
    except subprocess.TimeoutExpired as exc:
        wall = time.time() - t0
        stderr_tail = _timeout_stderr_tail(exc)
        _append_jsonl(
            failed_jsonl,
            {
                "episode_offset": i,
                "target_episode_index": target_episode,
                "reset_seed": reset_seed,
                "rc": None,
                "reason": "timeout",
                "timeout_seconds": float(args.per_episode_timeout),
                "stderr_tail": stderr_tail,
                "wall_seconds": wall,
            },
        )
        print(
            f"[campaign] {i + 1}/{total} TIMEOUT wall={wall:.1f}s target_episode={target_episode} "
            f"timeout_s={args.per_episode_timeout}",
            flush=True,
        )
        return EpisodeOutcome(i, target_episode, reset_seed, "failed", None, stderr_tail, wall)
    wall = time.time() - t0
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    if proc.returncode != 0:
        stderr_tail = _tail(proc.stderr or "")
        _append_jsonl(
            failed_jsonl,
            {
                "episode_offset": i,
                "target_episode_index": target_episode,
                "reset_seed": reset_seed,
                "rc": proc.returncode,
                "stderr_tail": stderr_tail,
                "wall_seconds": wall,
            },
        )
        print(
            f"[campaign] {i + 1}/{total} FAILED rc={proc.returncode} wall={wall:.1f}s "
            f"target_episode={target_episode}",
            flush=True,
        )
        sys.stderr.write(proc.stderr or "")
        sys.stderr.flush()
        return EpisodeOutcome(i, target_episode, reset_seed, "failed", None, stderr_tail, wall)

    summary: dict = {
        "episode_offset": i,
        "target_episode_index": target_episode,
        "reset_seed": reset_seed,
        "output_json": str(ep_json),
        "wall_seconds": wall,
    }
    if ep_json.is_file():
        try:
            payload = json.loads(ep_json.read_text())
            for key in (
                "latent_scores",
                "selected_scores",
                "candidate_distances",
                "selected_indices",
                "steps",
                "done",
                "goal_source",
            ):
                if key in payload:
                    summary[key] = payload[key]
        except Exception as exc:  # noqa: BLE001
            summary["summary_extract_error"] = repr(exc)
    _append_jsonl(episodes_jsonl, summary)
    print(
        f"[campaign] {i + 1}/{total} DONE wall={wall:.1f}s target_episode={target_episode}",
        flush=True,
    )
    return EpisodeOutcome(i, target_episode, reset_seed, "ok", str(ep_json), None, wall)


def _write_manifest(
    run_dir: Path,
    args: argparse.Namespace,
    oracle_run: Path,
    outcomes: list[EpisodeOutcome],
    episodes_jsonl: Path,
    skipped_jsonl: Path,
    failed_jsonl: Path,
) -> Path:
    manifest = {
        "campaign": "all60_frame50_k3",
        "carry_mode": args.carry_mode,
        "chunk_len": int(args.chunk_len),
        "comparison_strip_overlay": bool(args.comparison_strip_overlay),
        "counts": {
            "ok": sum(1 for o in outcomes if o.status == "ok"),
            "resume_skip": sum(1 for o in outcomes if o.status == "resume_skip"),
            "missing_goal": sum(1 for o in outcomes if o.status == "missing_goal"),
            "missing_prefetch": sum(1 for o in outcomes if o.status == "missing_prefetch"),
            "failed": sum(1 for o in outcomes if o.status == "failed"),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "episode_start": int(args.episode_start),
        "episodes": int(args.episodes),
        "episodes_jsonl": str(episodes_jsonl),
        "failed_episodes_jsonl": str(failed_jsonl),
        "goal_frame_index": int(args.goal_frame_index),
        "max_steps": int(args.max_steps),
        "num_candidates": int(args.num_candidates),
        "oracle_run_root": str(oracle_run),
        "outcomes": [asdict(o) for o in outcomes],
        "run_dir": str(run_dir),
        "seed_base": int(args.seed_base),
        "skipped_episodes_jsonl": str(skipped_jsonl),
        "smolvla_n_action_steps": int(args.smolvla_n_action_steps),
        "task": args.task,
        "wm_rollout_mode": args.wm_rollout_mode,
        "wm_scoring_latent": args.wm_scoring_latent,
        "prefetch_run_root": str(args.prefetch_run_root) if args.prefetch_run_root is not None else None,
        "prefetch_segment_index": int(getattr(args, "prefetch_segment_index", 0)),
        "wm_selection_env_steps": (
            int(args.wm_selection_env_steps) if getattr(args, "wm_selection_env_steps", None) is not None else None
        ),
    }
    manifest_path = run_dir / "segment_grpo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[campaign] wrote manifest: {manifest_path}", flush=True)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = _resolve_run_dir(args)
    oracle_run = _resolve_oracle_run(args)
    print(f"[campaign] run_dir={run_dir}", flush=True)
    print(f"[campaign] oracle_run={oracle_run}", flush=True)

    episodes_jsonl = run_dir / "episodes.jsonl"
    skipped_jsonl = run_dir / "skipped_episodes.jsonl"
    failed_jsonl = run_dir / "failed_episodes.jsonl"

    outcomes: list[EpisodeOutcome] = []
    total = int(args.episodes)
    for i in range(total):
        outcome = _run_episode(
            args,
            run_dir,
            oracle_run,
            episodes_jsonl,
            skipped_jsonl,
            failed_jsonl,
            i,
            total,
        )
        outcomes.append(outcome)
        if outcome.status == "failed" and args.stop_on_error:
            print(f"[campaign] --stop-on-error set, aborting after offset={i}", flush=True)
            break

    _write_manifest(run_dir, args, oracle_run, outcomes, episodes_jsonl, skipped_jsonl, failed_jsonl)

    n_failed = sum(1 for o in outcomes if o.status == "failed")
    if n_failed and args.stop_on_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
