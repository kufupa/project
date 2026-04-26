from __future__ import annotations

import numpy as np
import pytest

from smolvla_grpo.phase12_goals import (
    build_subgoal_schedule,
    compute_reset_parity,
    frame_index_to_filename,
    select_required_oracle_frame_indices,
    should_fail_reset_parity,
)


def test_goal_schedule_uses_primary_and_companion_frames() -> None:
    schedule = build_subgoal_schedule(
        max_frame_1based=80,
        chunk_len=25,
        success_frame_1based=None,
    )

    assert schedule.primary_frames_1based == [25, 50, 75, 80]
    assert schedule.companion_frames_1based == [26, 51, 76]
    assert frame_index_to_filename(25) == "frame_000024.png"


def test_goal_schedule_uses_success_frame_as_terminal_goal() -> None:
    schedule = build_subgoal_schedule(
        max_frame_1based=100,
        chunk_len=25,
        success_frame_1based=63,
    )

    assert schedule.primary_frames_1based == [25, 50, 63]
    assert schedule.companion_frames_1based == [26, 51, 64]


def test_required_oracle_frames_keep_init_and_goal_neighbors() -> None:
    schedule = build_subgoal_schedule(
        max_frame_1based=80,
        chunk_len=25,
        success_frame_1based=None,
    )

    assert select_required_oracle_frame_indices(
        max_frame_1based=80,
        schedule=schedule,
    ) == [1, 2, 24, 25, 26, 49, 50, 51, 74, 75, 76, 79, 80]


def test_required_oracle_frames_clip_to_valid_frame_range() -> None:
    schedule = build_subgoal_schedule(
        max_frame_1based=2,
        chunk_len=25,
        success_frame_1based=None,
    )

    assert select_required_oracle_frame_indices(
        max_frame_1based=2,
        schedule=schedule,
    ) == [1, 2]


def test_frame_index_rejects_zero_or_negative() -> None:
    with pytest.raises(ValueError, match="1-based"):
        frame_index_to_filename(0)


def test_reset_parity_reports_image_and_proprio_diffs() -> None:
    a_img = np.zeros((2, 2, 3), dtype=np.uint8)
    b_img = np.ones((2, 2, 3), dtype=np.uint8)
    a_prop = np.array([0.0, 1.0], dtype=np.float32)
    b_prop = np.array([0.5, 1.0], dtype=np.float32)

    metrics = compute_reset_parity(a_img, b_img, a_prop, b_prop)

    assert metrics["image_mean_abs_diff"] == pytest.approx(1.0)
    assert metrics["image_max_abs_diff"] == pytest.approx(1.0)
    assert metrics["proprio_max_abs_diff"] == pytest.approx(0.5)


def test_reset_parity_threshold_failure() -> None:
    metrics = {
        "image_mean_abs_diff": 0.2,
        "image_max_abs_diff": 3.0,
        "proprio_max_abs_diff": 0.0,
    }

    assert should_fail_reset_parity(
        metrics,
        image_mean_threshold=0.1,
        image_max_threshold=10.0,
        proprio_max_threshold=1e-5,
    )

