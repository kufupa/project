"""Unit tests for Phase11 sparse_success_delta reward backend."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from smolvla_grpo.reward_backends import (  # noqa: E402
    SparseSuccessDeltaBackend,
    compute_sparse_success_delta_return,
    make_phase11_reward_backend,
)


class _Traj:
    def __init__(self, successes: list[bool], rewards: list[float] | None = None):
        self.successes = successes
        self.rewards = rewards or [0.0] * len(successes)


def test_all_fail_zero_return_rel():
    assert compute_sparse_success_delta_return([False, False, False], use_rel_reward=True) == 0.0


def test_first_success_spike_rel():
    # success at step 2 only -> rel delta 0,0,1,0 after success held
    ret = compute_sparse_success_delta_return(
        [False, False, True, True],
        use_rel_reward=True,
    )
    assert ret == pytest.approx(1.0)


def test_non_rel_sums_success_indicators():
    ret = compute_sparse_success_delta_return(
        [False, True, True],
        use_rel_reward=False,
    )
    assert ret == pytest.approx(2.0)


def test_fail_vs_success_differ():
    fail = SparseSuccessDeltaBackend().episode_return(_Traj([False] * 5))
    ok = SparseSuccessDeltaBackend().episode_return(
        _Traj([False, False, True, True, True])
    )
    assert fail == 0.0
    assert ok > fail


def test_dense_backend_unchanged():
    dense = make_phase11_reward_backend(reward_mode="dense_return")
    t = _Traj([False], rewards=[1.0, 2.0, 3.0])
    assert dense.episode_return(t) == pytest.approx(6.0)
