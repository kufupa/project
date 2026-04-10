#!/usr/bin/env python3
"""
Run one Meta-World scripted-oracle episode with SmolVLA-parity rendering (corner2 + flip, 120 steps).

This is a thin orchestrator around ``run_metaworld_oracle_eval.py``. It produces the **same
artifact layout** as ``run_oracle_baseline_eval.sh`` / the original oracle pipeline:

- ``eval_info.json``, ``run_manifest.json``
- ``episodes/episode_0000/actions.jsonl`` (scripted policy actions per step)
- ``frames/episode_0000/frame_*.png`` (when save-frames is true)
- ``videos/push-v3_0/eval_episode_0.mp4``

Run directory naming matches the bash runner::

    run_{UTC}_ep1_voracle_t{task_slug}_s{seed}_r{nonce}

Default seed is **1000** to match the phase06 oracle campaign
(``run_*_ep45_voracle_tpush_v3_s1000_*``). Override with ``--seed`` or ``ORACLE_PARITY_SEED``.
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults — SmolVLA parity camera + horizon; oracle-standard outputs
# ---------------------------------------------------------------------------
DEFAULT_TASK = "push-v3"
DEFAULT_EPISODES = 1
DEFAULT_SEED = 1000
DEFAULT_MAX_STEPS = 120
DEFAULT_FPS = 30
DEFAULT_CAMERA_NAME = "corner2"
DEFAULT_FLIP_CORNER2 = True
DEFAULT_VIDEO = True
DEFAULT_SAVE_FRAMES = True


def _project_roots() -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    workspace_root = project_root.parent
    return project_root, workspace_root


def _task_slug(task: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", task.strip()).strip("_").lower()
    return slug or "task"


def _unique_run_dir(
    output_root: Path, *, episodes: int, task: str, seed: int
) -> Path:
    """Match run_oracle_baseline_eval.sh directory creation."""
    output_root.mkdir(parents=True, exist_ok=True)
    task_slug = _task_slug(task)
    for _ in range(10):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        nonce = f"{secrets.randbelow(900_000) + 100_000:06d}"
        candidate = output_root / (
            f"run_{timestamp}_ep{episodes}_voracle_t{task_slug}_s{seed}_r{nonce}"
        )
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Failed to allocate unique run directory under {output_root}")


def _resolve_lerobot_python(workspace_root: Path) -> Path:
    env_dir = os.environ.get("SMOLVLA_LEROBOT_ENV_DIR", "").strip()
    if not env_dir:
        env_dir = str(workspace_root / ".envs" / "lerobot_mw_py310")
    py = Path(env_dir).expanduser() / "bin" / "python"
    if not py.is_file():
        raise FileNotFoundError(
            f"LeRobot python not found at {py}. Set SMOLVLA_LEROBOT_ENV_DIR."
        )
    if not os.access(py, os.X_OK):
        raise PermissionError(f"Python is not executable: {py}")
    return py


def _as_bool_str(value: bool) -> str:
    return "true" if value else "false"


def parse_args() -> argparse.Namespace:
    env_seed = os.environ.get("ORACLE_PARITY_SEED", "").strip()
    default_seed = int(env_seed) if env_seed.isdigit() else DEFAULT_SEED

    parser = argparse.ArgumentParser(
        description="Oracle 1-episode parity run (SmolVLA camera + 120 steps, full oracle outputs)."
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--camera-name", default=DEFAULT_CAMERA_NAME)
    parser.add_argument(
        "--flip-corner2",
        default=None,
        choices=("true", "false"),
        help="Override corner2 flip (default: true).",
    )
    parser.add_argument(
        "--video",
        default=_as_bool_str(DEFAULT_VIDEO),
        choices=("true", "false"),
        help="Write MP4 under videos/ (default: true).",
    )
    parser.add_argument(
        "--save-frames",
        default=_as_bool_str(DEFAULT_SAVE_FRAMES),
        choices=("true", "false"),
        help="Write PNG frames under frames/episode_XXXX/ (default: true).",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get(
            "ORACLE_ARTIFACT_ROOT",
            str(_project_roots()[0] / "artifacts" / "phase06_oracle_baseline"),
        ),
        help="Parent directory for run_*_voracle_* (default: phase06_oracle_baseline).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="If set, use this directory instead of creating a new run_* folder.",
    )
    parser.add_argument(
        "--no-xvfb",
        action="store_true",
        help="Run without xvfb-run (only if you already have a display).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root, workspace_root = _project_roots()
    eval_script = project_root / "scripts" / "oracle" / "run_metaworld_oracle_eval.py"
    if not eval_script.is_file():
        print(f"error: missing {eval_script}", file=sys.stderr)
        return 2

    flip = DEFAULT_FLIP_CORNER2 if args.flip_corner2 is None else args.flip_corner2 == "true"

    if args.output_dir:
        run_dir = Path(args.output_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = _unique_run_dir(
            Path(args.output_root).expanduser().resolve(),
            episodes=int(args.episodes),
            task=args.task,
            seed=int(args.seed),
        )

    py = _resolve_lerobot_python(workspace_root)

    cmd: list[str] = [
        str(py),
        str(eval_script),
        "--task",
        args.task,
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--max-steps",
        str(args.max_steps),
        "--video",
        args.video,
        "--fps",
        str(args.fps),
        "--camera-name",
        str(args.camera_name).strip(),
        "--flip-corner2",
        _as_bool_str(flip),
        "--save-frames",
        args.save_frames,
        "--output-dir",
        str(run_dir),
    ]

    if args.no_xvfb:
        print("[INFO] Running without xvfb-run.", file=sys.stderr)
        proc = subprocess.run(cmd, cwd=str(project_root))
        return int(proc.returncode)

    if not shutil.which("xvfb-run"):
        print("error: xvfb-run not found; use --no-xvfb on a machine with a display.", file=sys.stderr)
        return 3

    xvfb_cmd = [
        "xvfb-run",
        "-a",
        "-s",
        "-screen 0 1280x1024x24",
        *cmd,
    ]
    print(f"[INFO] Oracle parity run dir: {run_dir}", file=sys.stderr)
    print(f"[INFO] Python: {py}", file=sys.stderr)
    proc = subprocess.run(xvfb_cmd, cwd=str(project_root))
    if proc.returncode != 0:
        return int(proc.returncode)

    print(f"Baseline eval output directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
