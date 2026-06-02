"""Oracle subgoal scheduling and reset-parity helpers for Phase12."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Phase12GoalSchedule:
    primary_frames_1based: list[int]
    companion_frames_1based: list[int]


@dataclass(frozen=True)
class Phase12GoalBundle:
    schedule: Phase12GoalSchedule
    output_dir: Path
    reset_metrics: dict[str, float]
    goals: list[Any]


@dataclass(frozen=True)
class Phase12Goal:
    subgoal_index: int
    frame_index_1based: int
    frame_path: Path
    companion_frame_index_1based: int | None
    companion_frame_path: Path | None
    proprio: np.ndarray
    goal_visual: Any
    goal_proprio: Any
    source: str


@dataclass(frozen=True)
class Phase12LocalTransition:
    subgoal_index: int
    root_frame_1based: int
    goal_frame_1based: int


def frame_index_to_filename(frame_index_1based: int) -> str:
    idx = int(frame_index_1based)
    if idx < 1:
        raise ValueError("frame index must be 1-based and >= 1")
    return f"frame_{idx - 1:06d}.png"


def build_subgoal_schedule(
    *,
    max_frame_1based: int,
    chunk_len: int,
    success_frame_1based: int | None = None,
) -> Phase12GoalSchedule:
    max_frame = int(success_frame_1based if success_frame_1based is not None else max_frame_1based)
    if max_frame < 1:
        raise ValueError("max_frame_1based must be >= 1")
    stride = int(chunk_len)
    if stride < 1:
        raise ValueError("chunk_len must be >= 1")
    primary = list(range(stride, max_frame + 1, stride))
    if not primary or primary[-1] != max_frame:
        primary.append(max_frame)
    companion = [frame + 1 for frame in primary if frame + 1 <= int(max_frame_1based)]
    return Phase12GoalSchedule(
        primary_frames_1based=[int(x) for x in primary],
        companion_frames_1based=[int(x) for x in companion],
    )


def build_local_transition_schedule(
    *,
    max_frame_1based: int,
    chunk_len: int,
    success_frame_1based: int | None = None,
) -> list[Phase12LocalTransition]:
    terminal = int(success_frame_1based if success_frame_1based is not None else max_frame_1based)
    if terminal < 1:
        raise ValueError("max_frame_1based must be >= 1")
    stride = int(chunk_len)
    if stride < 1:
        raise ValueError("chunk_len must be >= 1")

    goals = list(range(stride, terminal + 1, stride))
    if not goals or goals[-1] != terminal:
        goals.append(terminal)
    roots = [1]
    roots.extend(goals[:-1])

    return [
        Phase12LocalTransition(
            subgoal_index=int(i),
            root_frame_1based=int(root),
            goal_frame_1based=int(goal),
        )
        for i, (root, goal) in enumerate(zip(roots, goals, strict=True))
    ]


def select_required_oracle_frame_indices(
    *,
    max_frame_1based: int,
    schedule: Phase12GoalSchedule,
    include_initial_pair: bool = True,
    neighbor_radius: int = 1,
) -> list[int]:
    max_frame = int(max_frame_1based)
    if max_frame < 1:
        raise ValueError("max_frame_1based must be >= 1")
    radius = int(neighbor_radius)
    if radius < 0:
        raise ValueError("neighbor_radius must be >= 0")

    selected: set[int] = set()
    if include_initial_pair:
        selected.update(idx for idx in (1, 2) if idx <= max_frame)

    for goal_frame in schedule.primary_frames_1based:
        center = int(goal_frame)
        for offset in range(-radius, radius + 1):
            idx = center + offset
            if 1 <= idx <= max_frame:
                selected.add(idx)

    return sorted(selected)


def compute_reset_parity(
    init_image: np.ndarray,
    reset_image: np.ndarray,
    init_proprio: np.ndarray,
    reset_proprio: np.ndarray,
) -> dict[str, float]:
    img_a = np.asarray(init_image, dtype=np.float32)
    img_b = np.asarray(reset_image, dtype=np.float32)
    if img_a.shape != img_b.shape:
        raise ValueError(f"reset image shape mismatch: {img_a.shape} != {img_b.shape}")
    prop_a = np.asarray(init_proprio, dtype=np.float32).reshape(-1)
    prop_b = np.asarray(reset_proprio, dtype=np.float32).reshape(-1)
    if prop_a.shape != prop_b.shape:
        raise ValueError(f"reset proprio shape mismatch: {prop_a.shape} != {prop_b.shape}")
    img_diff = np.abs(img_a - img_b)
    prop_diff = np.abs(prop_a - prop_b)
    return {
        "image_mean_abs_diff": float(np.mean(img_diff)) if img_diff.size else 0.0,
        "image_max_abs_diff": float(np.max(img_diff)) if img_diff.size else 0.0,
        "proprio_max_abs_diff": float(np.max(prop_diff)) if prop_diff.size else 0.0,
    }


def should_fail_reset_parity(
    metrics: dict[str, float],
    *,
    image_mean_threshold: float,
    image_max_threshold: float,
    proprio_max_threshold: float,
) -> bool:
    return (
        float(metrics.get("image_mean_abs_diff", 0.0)) > float(image_mean_threshold)
        or float(metrics.get("image_max_abs_diff", 0.0)) > float(image_max_threshold)
        or float(metrics.get("proprio_max_abs_diff", 0.0)) > float(proprio_max_threshold)
    )


def collect_phase12_oracle_goals(
    *,
    env_backend: str,
    task: str,
    seed: int,
    chunk_len: int,
    output_dir: Path,
    strict_reset: bool,
    wm_bundle: Any | None = None,
) -> Phase12GoalBundle:
    """Placeholder API for real oracle rollout wiring in trainer integration.

    The deterministic schedule/reset helpers above are covered by unit tests.
    Real environment collection is implemented in the Phase12 trainer path.
    """

    raise NotImplementedError(
        "collect_phase12_oracle_goals requires Phase12 trainer env wiring; "
        f"got env_backend={env_backend!r}, task={task!r}, seed={int(seed)}, "
        f"chunk_len={int(chunk_len)}, strict_reset={bool(strict_reset)}, "
        f"output_dir={Path(output_dir)}."
    )

