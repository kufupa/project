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
    # Optional: flatten_obs_state aligned with goal_frame PNG (from flat_obs.jsonl).
    goal_flat_obs: np.ndarray | None = None


@dataclass(frozen=True)
class OracleActionSequence:
    """Oracle scripted-policy actions for one episode (from actions.jsonl)."""

    run_dir: Path
    episode_index: int
    action_source_path: Path
    actions: np.ndarray  # (T, env_action_dim), float32
    n_steps: int
    env_action_dim: int


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


def load_oracle_goal_flat_obs(run_dir: Path, episode_index: int, goal_frame_idx_zero_based: int) -> np.ndarray | None:
    """
    Load flattened observation for the oracle frame index that matches ``frame_{idx:06d}.png``.

    Written by ``run_metaworld_oracle_eval.py`` as ``episodes/episode_XXXX/flat_obs.jsonl`` when
    ``--save-frames`` is enabled. Returns ``None`` if the file is missing or has no matching row.
    """
    run_path = Path(run_dir).expanduser().resolve()
    path = run_path / "episodes" / f"episode_{int(episode_index):04d}" / "flat_obs.jsonl"
    if not path.is_file():
        return None
    want = int(goal_frame_idx_zero_based)
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(obj.get("frame_index", -1)) != want:
                continue
            flat = obj.get("flat_obs")
            if not isinstance(flat, list) or not flat:
                return None
            return np.asarray(flat, dtype=np.float32).reshape(-1)
    return None


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

    goal_flat_obs = load_oracle_goal_flat_obs(run_path, int(episode_index), goal_idx)

    return OracleReferenceFrames(
        run_dir=run_path,
        episode_index=int(episode_index),
        goal_frame_idx_zero_based=goal_idx,
        task=task,
        start_frame_path=start_path,
        goal_frame_path=goal_path,
        start_frame=_load_png_rgb(start_path),
        goal_frame=_load_png_rgb(goal_path),
        goal_flat_obs=goal_flat_obs,
    )


def load_oracle_action_sequence(run_dir: Path, episode_index: int) -> OracleActionSequence:
    """
    Load oracle per-step actions for ``episode_index`` from ``episodes/episode_XXXX/actions.jsonl``.

    Each JSON line must contain an ``action`` key: list of floats (push-v3: length 4).
    """
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"Oracle run dir not found: {run_path}")

    actions_path = run_path / "episodes" / f"episode_{int(episode_index):04d}" / "actions.jsonl"
    if not actions_path.is_file():
        raise FileNotFoundError(f"Oracle actions.jsonl missing: {actions_path}")

    rows: list[list[float]] = []
    with actions_path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{actions_path}:{line_no} invalid JSON: {exc}") from exc
            action = obj.get("action")
            if not isinstance(action, list) or not action:
                raise ValueError(f"{actions_path}:{line_no} missing or empty 'action' list")
            rows.append([float(x) for x in action])

    if not rows:
        raise ValueError(f"Oracle actions.jsonl empty: {actions_path}")

    dims = {len(r) for r in rows}
    if len(dims) != 1:
        raise ValueError(f"Oracle action rows have mixed widths {dims} in {actions_path}")
    env_action_dim = int(next(iter(dims)))
    arr = np.asarray(rows, dtype=np.float32)

    return OracleActionSequence(
        run_dir=run_path,
        episode_index=int(episode_index),
        action_source_path=actions_path,
        actions=arr,
        n_steps=int(arr.shape[0]),
        env_action_dim=env_action_dim,
    )


def load_prefetch_candidate_actions(
    path: Path,
    *,
    num_candidates: int,
    target_rows: int,
    segment_index: int = 0,
    expected_task: str | None = None,
    expected_episode_index: int | None = None,
) -> list[np.ndarray]:
    """Load per-candidate action chunks from a prior segment_grpo episode JSON.

    Reads ``segments[segment_index].candidates[*].actions`` (index order 0..K-1),
    validates row counts, returns float32 arrays shaped ``(target_rows, D)``.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Prefetch JSON not found: {p}")
    if int(num_candidates) < 1:
        raise ValueError("num_candidates must be >= 1")
    if int(target_rows) < 1:
        raise ValueError("target_rows must be >= 1")

    with p.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if expected_task is not None:
        jt = str(payload.get("task", "") or "").strip()
        if jt and jt != str(expected_task).strip():
            raise ValueError(f"Prefetch JSON task {jt!r} != expected {expected_task!r} ({p})")
    if expected_episode_index is not None and payload.get("episode_index") is not None:
        if int(payload["episode_index"]) != int(expected_episode_index):
            raise ValueError(
                f"Prefetch JSON episode_index={payload['episode_index']} != expected "
                f"{expected_episode_index} ({p})"
            )

    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"Prefetch JSON missing non-empty segments: {p}")
    if int(segment_index) < 0 or int(segment_index) >= len(segments):
        raise IndexError(f"segment_index {segment_index} out of range for {len(segments)} segments ({p})")

    seg = segments[int(segment_index)]
    cands = seg.get("candidates")
    if not isinstance(cands, list) or len(cands) != int(num_candidates):
        raise ValueError(
            f"Prefetch segment {segment_index} expected {num_candidates} candidates, got "
            f"{len(cands) if isinstance(cands, list) else type(cands)} ({p})"
        )

    out: list[np.ndarray] = []
    env_dim: int | None = None
    for i, c in enumerate(cands):
        if not isinstance(c, dict):
            raise ValueError(f"candidate {i} not an object ({p})")
        if int(c.get("index", i)) != i:
            raise ValueError(f"candidate index mismatch at slot {i} ({p})")
        actions = c.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"candidate {i} missing actions ({p})")
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"candidate {i} actions must be 2-D, got {arr.shape} ({p})")
        if arr.shape[0] < int(target_rows):
            raise ValueError(
                f"candidate {i} has {arr.shape[0]} rows < target_rows={target_rows} ({p})"
            )
        d = int(arr.shape[1])
        if env_dim is None:
            env_dim = d
        elif d != env_dim:
            raise ValueError(f"candidate {i} action width {d} != {env_dim} ({p})")
        out.append(np.asarray(arr[: int(target_rows)], dtype=np.float32).copy())

    return out
