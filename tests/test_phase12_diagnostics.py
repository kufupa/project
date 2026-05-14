from __future__ import annotations

import numpy as np
import pytest

from smolvla_grpo.phase12_diagnostics import (
    build_decode_artifacts,
    expected_wm_decode_steps,
    write_phase12_episode_video,
)


def _frame(value: int) -> np.ndarray:
    return np.full((64, 64, 3), value, dtype=np.uint8)


def test_expected_decoded_future_frames_for_25_env_steps() -> None:
    assert expected_wm_decode_steps(chunk_len=25, env_steps_per_wm_step=5) == 5


def test_decode_failure_nonfatal_unless_strict(tmp_path) -> None:
    result = build_decode_artifacts(
        decode_fn=lambda: (_ for _ in ()).throw(RuntimeError("decode broke")),
        output_dir=tmp_path,
        real_frames=[_frame(0), _frame(1)],
        strict_decode=False,
    )

    assert result.metadata["decode_status"] == "failed"
    assert "decode broke" in result.metadata["decode_failure_reason"]


def test_decode_failure_strict_raises(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="decode broke"):
        build_decode_artifacts(
            decode_fn=lambda: (_ for _ in ()).throw(RuntimeError("decode broke")),
            output_dir=tmp_path,
            real_frames=[_frame(0), _frame(1)],
            strict_decode=True,
        )


def test_selected_decode_artifacts_write_strip_paths(tmp_path) -> None:
    result = build_decode_artifacts(
        decode_fn=lambda: [_frame(10), _frame(20)],
        output_dir=tmp_path,
        real_frames=[_frame(0), _frame(1), _frame(2), _frame(3), _frame(4), _frame(5)],
        strict_decode=True,
        segment_index=0,
        selected_candidate_index=1,
        env_steps_per_wm_step=2,
        carried_steps=5,
    )

    assert result.metadata["decode_status"] == "ok"
    assert result.metadata["decoded_frame_count"] == 2
    selected = result.paths["wm_decode_selected_strip_path"]
    real_vs_pred = result.paths["wm_real_vs_pred_selected_strip_path"]
    assert selected.is_file()
    assert real_vs_pred.is_file()


def test_selected_decode_strip_resizes_prediction_frames_to_real_frame_height(tmp_path) -> None:
    result = build_decode_artifacts(
        decode_fn=lambda: [np.full((32, 32, 3), 10, dtype=np.uint8)],
        output_dir=tmp_path,
        real_frames=[_frame(0), _frame(1), _frame(2)],
        strict_decode=True,
        env_steps_per_wm_step=1,
        carried_steps=1,
    )

    assert result.metadata["decode_status"] == "ok"
    assert result.paths["wm_decode_selected_strip_path"].is_file()


def test_write_phase12_episode_video_writes_nonempty_mp4(tmp_path) -> None:
    video = tmp_path / "selected_action_rollout.mp4"

    out = write_phase12_episode_video(
        video_path=video,
        frames=[_frame(0), _frame(20), _frame(40)],
        rewards=[0.0, 1.0],
        successes=[False, True],
        fps=6,
        overlay_mode="cumulative_reward",
    )

    assert out == video
    assert video.is_file()
    assert video.stat().st_size > 0

