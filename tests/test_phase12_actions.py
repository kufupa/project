from __future__ import annotations

import numpy as np
import pytest

from smolvla_grpo.phase12_actions import apply_phase12_action_profile


class _DummyPreprocessor:
    action_mean = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    action_std = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32)

    def normalize_actions(self, actions):
        return (np.asarray(actions, dtype=np.float32) - self.action_mean) / self.action_std


def test_official_jepa_mirror_scores_raw_postprocessed_actions() -> None:
    raw = np.array([[2.0, -2.0, 0.5, 1.5]], dtype=np.float32)

    result = apply_phase12_action_profile(
        raw,
        action_profile="official_jepa_mirror",
        action_low=-1.0,
        action_high=1.0,
    )

    np.testing.assert_allclose(result.exec_actions_for_env, raw)
    np.testing.assert_allclose(result.exec_actions_for_wm, raw)
    np.testing.assert_allclose(result.exec_actions_clipped, np.clip(raw, -1.0, 1.0))
    assert result.metadata["clip_fraction"] > 0.0
    assert result.metadata["clip_any"] is True
    assert result.metadata["exec_action_source"] == "raw_postprocessed"
    assert result.metadata["wm_action_source"] == "raw_postprocessed"


def test_bounded_executed_scores_exactly_what_it_executes() -> None:
    raw = np.array([[2.0, -2.0, 0.5, 1.5]], dtype=np.float32)

    result = apply_phase12_action_profile(
        raw,
        action_profile="bounded_executed",
        action_low=-1.0,
        action_high=1.0,
    )

    np.testing.assert_allclose(result.exec_actions_for_env, np.clip(raw, -1.0, 1.0))
    np.testing.assert_allclose(result.exec_actions_for_wm, result.exec_actions_for_env)
    assert result.metadata["exec_action_source"] == "clipped"
    assert result.metadata["wm_action_source"] == "clipped"


def test_action_metadata_includes_jepa_normalized_and_packed_shape() -> None:
    raw = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 7.0, 12.0],
        ],
        dtype=np.float32,
    )

    result = apply_phase12_action_profile(
        raw,
        action_profile="official_jepa_mirror",
        action_low=np.full((4,), -1.0, dtype=np.float32),
        action_high=np.full((4,), 1.0, dtype=np.float32),
        preprocessor=_DummyPreprocessor(),
        env_action_dim=4,
        wm_action_dim=20,
    )

    assert result.metadata["env_action_dim"] == 4
    assert result.metadata["wm_action_dim"] == 20
    assert result.metadata["pack_factor"] == 5
    assert result.metadata["packed_action_shape"] == [1, 20]
    assert result.metadata["jepa_norm_max_abs"] == pytest.approx(1.0)
    assert result.metadata["jepa_norm_min"] == pytest.approx(0.0)
    assert result.metadata["jepa_norm_max"] == pytest.approx(1.0)


def test_unknown_action_profile_fails() -> None:
    with pytest.raises(ValueError, match="action_profile"):
        apply_phase12_action_profile(
            np.zeros((1, 4), dtype=np.float32),
            action_profile="bad",
            action_low=-1.0,
            action_high=1.0,
        )

