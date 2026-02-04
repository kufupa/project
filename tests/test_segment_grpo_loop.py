from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from segment_grpo_loop import (
    _ensure_action_matrix,
    rollout_with_chunks,
    score_chunk_by_goal_latent,
    WMBundle,
)



def test_replay_rollout_contracts_shape_and_scores() -> None:
    """Replay carry mode should advance by chunk and record selected metadata."""
    episode, _adapter = rollout_with_chunks(
        smolvla_bundle=None,
        wm_bundle=None,
        task="push-v3",
        episode_index=0,
        chunk_len=4,
        num_candidates=3,
        max_steps=10,
        carry_mode="replay",
        replay_root=None,
        goal_latent_source=None,
        seed=123,
        train_steps=0,
        dry_run=True,
    )

    assert episode.steps == 10
    assert episode.done is True
    assert len(episode.segments) == 3
    assert episode.steps == len(episode.actions)
    assert len(episode.latent_scores) == len(episode.segments)
    assert len(episode.selected_scores) == len(episode.segments)
    assert len(episode.selected_indices) == len(episode.segments)

    for segment in episode.segments:
        assert len(segment.candidates) == 3
        assert len(segment.executed_actions) <= 4
        assert segment.selected_index in {c.index for c in segment.candidates}


def test_ensure_action_matrix_padding_and_truncation_regression() -> None:
    """Action matrix helper must preserve leading 4-D action values and pad/truncate safely."""
    short = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ],
        dtype=np.float32,
    )
    padded = _ensure_action_matrix(short, action_dim=20, length=2)
    assert padded.shape == (2, 20)
    np.testing.assert_allclose(padded[:, :4], short)
    np.testing.assert_allclose(padded[:, 4:], 0.0)

    long = np.arange(12, dtype=np.float32).reshape(3, 4)
    truncated = _ensure_action_matrix(long, action_dim=2, length=2)
    assert truncated.shape == (2, 2)
    expected = np.array([[0.0, 1.0], [4.0, 5.0]], dtype=np.float32)
    np.testing.assert_allclose(truncated, expected)



def test_wm_encode_unroll_smoke() -> None:
    torch = pytest.importorskip("torch")

    class _FakeModel:
        def encode(self, obs: dict[str, object]) -> torch.Tensor:
            return torch.tensor([[[0.25, -0.10, 0.40]]], dtype=torch.float32)

        def unroll(self, z: torch.Tensor, act_suffix: torch.Tensor, debug: bool = False) -> torch.Tensor:
            assert z.shape == (1, 1, 3)
            assert act_suffix.shape == (1, 1, 4)
            return torch.tensor([0.50, 0.70, 0.90], dtype=torch.float32)

    bundle = WMBundle(
        model=_FakeModel(),
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    goal_latent = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

    distance = score_chunk_by_goal_latent(
        wm_bundle=bundle,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        chunk_actions=np.array([[0.1, -0.2, 0.3, 0.4]], dtype=np.float32),
        goal_latent=goal_latent,
        chunk_len=1,
    )

    assert isinstance(distance, float)
    assert distance > 0.0
