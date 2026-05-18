"""Tests for smolvla_grpo.grpo_math."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from smolvla_grpo.grpo_math import (
    compute_clipped_grpo_loss,
    compute_group_advantages,
    compute_seed_batch_advantages,
    summarize_ratio_stats,
)


def test_advantage_zero_variance() -> None:
    r = torch.tensor([1.0, 1.0, 1.0])
    a = compute_group_advantages(r)
    assert torch.allclose(a, torch.zeros_like(a))


def test_advantage_normalized() -> None:
    r = torch.tensor([0.0, 4.0, 8.0])
    a = compute_group_advantages(r)
    assert abs(float(a.mean())) < 1e-5
    assert float(a.std()) > 0.9


def test_compute_seed_batch_advantages_matches_per_group_manual() -> None:
    returns = torch.tensor([10.0, 12.0, 8.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    got = compute_seed_batch_advantages(returns, group_size=3)
    expected = torch.cat(
        [
            compute_group_advantages(returns[:3]),
            compute_group_advantages(returns[3:]),
        ],
        dim=0,
    )
    torch.testing.assert_close(got, expected)


def test_compute_seed_batch_advantages_one_group_zero_variance() -> None:
    returns = torch.tensor([0.0, 4.0, 8.0, 5.0, 5.0, 5.0], dtype=torch.float32)
    got = compute_seed_batch_advantages(returns, group_size=3)
    assert not torch.allclose(got[:3], torch.zeros(3))
    assert torch.allclose(got[3:], torch.zeros(3))


def test_compute_seed_batch_advantages_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="multiple of group_size"):
        compute_seed_batch_advantages(torch.tensor([1.0, 2.0]), group_size=3)


def test_clipped_loss_changes_when_ratio_large() -> None:
    eps = 0.2
    old_lp = torch.zeros(5)
    new_lp_far = old_lp + 3.0  # ratio exp(3) >> clip
    new_lp_near = old_lp.clone()
    A = torch.tensor(1.0)
    loss_big, _ = compute_clipped_grpo_loss(new_lp_far, old_lp, A, epsilon=eps)
    loss_small, _ = compute_clipped_grpo_loss(new_lp_near, old_lp, A, epsilon=eps)
    assert loss_big.item() != loss_small.item()


def test_ratio_stats_clip_fraction() -> None:
    ratio = torch.tensor([0.5, 1.0, 1.5])
    s = summarize_ratio_stats(ratio, epsilon=0.2)
    assert s.clip_fraction > 0.0


def test_clipped_loss_finite() -> None:
    new_lp = torch.randn(10, requires_grad=True)
    old_lp = torch.randn(10)
    A = torch.tensor(0.5)
    loss, stats = compute_clipped_grpo_loss(new_lp, old_lp, A, epsilon=0.2)
    assert torch.isfinite(loss)
    assert stats.n == 10
