from __future__ import annotations

from types import SimpleNamespace

import torch

from smolvla_grpo.reward_backends import episode_return_for_mode


def _chunk(*, rewards: list[float], successes: list[bool], valid: list[bool]) -> SimpleNamespace:
    return SimpleNamespace(
        rewards=torch.tensor(rewards, dtype=torch.float32),
        successes=torch.tensor(successes, dtype=torch.bool),
        valid_action_mask=torch.tensor(valid, dtype=torch.bool),
    )


def _traj(chunks: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(chunks=chunks)


def test_dense_return_can_rank_dense_failure_above_success() -> None:
    success = _traj([_chunk(rewards=[2.0, 3.0, 0.0], successes=[False, True, False], valid=[True, True, False])])
    failure = _traj([_chunk(rewards=[9.0, 9.0, 9.0], successes=[False, False, False], valid=[True, True, True])])

    assert episode_return_for_mode(success, reward_mode="dense_return") == 5.0
    assert episode_return_for_mode(failure, reward_mode="dense_return") == 27.0


def test_sparse_success_delta_ranks_success_above_dense_failure() -> None:
    success = _traj([_chunk(rewards=[2.0, 3.0, 0.0], successes=[False, True, False], valid=[True, True, False])])
    failure = _traj([_chunk(rewards=[9.0, 9.0, 9.0], successes=[False, False, False], valid=[True, True, True])])

    assert episode_return_for_mode(success, reward_mode="sparse_success_delta") == 1.0
    assert episode_return_for_mode(failure, reward_mode="sparse_success_delta") == 0.0


def test_success_first_dense_ranks_any_success_above_any_failure() -> None:
    success = _traj([_chunk(rewards=[2.0, 3.0, 0.0], successes=[False, True, False], valid=[True, True, False])])
    failure = _traj([_chunk(rewards=[900.0, 900.0, 900.0], successes=[False, False, False], valid=[True, True, True])])

    assert episode_return_for_mode(success, reward_mode="success_first_dense") > episode_return_for_mode(
        failure,
        reward_mode="success_first_dense",
    )


def test_reward_modes_ignore_invalid_terminal_tail() -> None:
    traj = _traj([_chunk(rewards=[1.0, 2.0, 100.0], successes=[False, False, True], valid=[True, True, False])])

    assert episode_return_for_mode(traj, reward_mode="dense_return") == 3.0
    assert episode_return_for_mode(traj, reward_mode="sparse_success_delta") == 0.0
