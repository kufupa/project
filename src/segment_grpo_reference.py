from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TOPK_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|")


@dataclass(frozen=True)
class TopEpisode:
    rank: int
    episode_index: int
    reset_seed: int
    row: dict[str, Any]


@dataclass(frozen=True)
class OracleReferenceFrames:
    run_dir: Path
    episode_index: int
    goal_frame_idx_zero_based: int
    task: str
    goal_frame_path: Path
    start_frame_path: Path
    goal_frame: np.ndarray
    start_frame: np.ndarray


def _load_png_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    img_path = Path(path)
    with Image.open(img_path) as img:
        frame = img.convert("RGB")
    return np.asarray(frame, dtype=np.uint8)


def resolve_latest_oracle_pushv3_run(artifacts_root: Path, task: str = "push-v3") -> Path:
    """Return latest available `phase06_oracle_baseline` push-v3 run."""
    root = Path(artifacts_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Artifacts root not found: {root}")
    phase_root = root / "phase06_oracle_baseline"
    if not phase_root.exists():
        raise FileNotFoundError(f"Oracle phase06 baseline folder missing: {phase_root}")

    task_snippet = f"_t{str(task).replace('-', '_')}_"
    run_dirs = [
        p
        for p in phase_root.glob("run_*")
        if p.is_dir() and task_snippet in p.name and (p / "run_manifest.json").exists()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No push-v3 oracle run found in {phase_root}.")
    return sorted(run_dirs)[-1]


def parse_top15_report(path: Path) -> list[TopEpisode]:
    """Parse rows from the markdown top-15 oracle report."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Top-15 report not found: {p}")
    payload: list[TopEpisode] = []
    raw = p.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(raw):
        m = TOPK_ROW_RE.match(line.strip())
        if not m:
            continue
        rank = int(m.group(1))
        episode_index = int(m.group(2))
        reset_seed = int(m.group(3))
        payload.append(
            TopEpisode(
                rank=rank,
                episode_index=episode_index,
                reset_seed=reset_seed,
                row={
                    "rank": rank,
                    "episode_index": episode_index,
                    "reset_seed": reset_seed,
                    "line_no": idx,
                },
            )
        )
    if not payload:
        raise ValueError(f"No top-k rows found in report: {p}")
    return payload


def load_oracle_reference_frames(
    run_dir: Path,
    episode_index: int,
    goal_frame_index: int,
    *,
    start_frame_index: int = 0,
) -> OracleReferenceFrames:
    """
    Load oracle start + goal frames for the requested episode.

    `goal_frame_index` is 1-based (e.g. 25 => ``frame_000024.png``).
    """
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"Oracle run dir not found: {run_path}")

    if goal_frame_index <= 0:
        raise ValueError(f"goal_frame_index must be >= 1; got {goal_frame_index!r}")
    if start_frame_index < 0:
        raise ValueError(f"start_frame_index must be >= 0; got {start_frame_index!r}")
    task = "push-v3"
    manifest = run_path / "run_manifest.json"
    if manifest.exists():
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        task = str(manifest_payload.get("task", task))

    frames_dir = run_path / "frames" / f"episode_{int(episode_index):04d}"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Episode frame directory missing: {frames_dir}")

    goal_idx = int(goal_frame_index) - 1
    start_path = frames_dir / f"frame_{int(start_frame_index):06d}.png"
    goal_path = frames_dir / f"frame_{goal_idx:06d}.png"
    if not start_path.exists():
        raise FileNotFoundError(f"Oracle start frame not found: {start_path}")
    if not goal_path.exists():
        raise FileNotFoundError(f"Oracle goal frame not found: {goal_path}")

    return OracleReferenceFrames(
        run_dir=run_path,
        episode_index=int(episode_index),
        goal_frame_idx_zero_based=goal_idx,
        task=task,
        start_frame_path=start_path,
        goal_frame_path=goal_path,
        start_frame=_load_png_rgb(start_path),
        goal_frame=_load_png_rgb(goal_path),
    )
