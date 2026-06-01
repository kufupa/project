from __future__ import annotations

import torch

from smolvla_grpo.chunk_math import masked_chunk_reward_sum, masked_chunk_sum, valid_chunk_any


def test_masked_chunk_sum_accepts_per_action_logprobs() -> None:
    log_probs = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    valid = torch.tensor([[True, True, False], [False, True, True]])

    out = masked_chunk_sum(log_probs, valid)

    torch.testing.assert_close(out, torch.tensor([3.0, 50.0]))


def test_masked_chunk_sum_accepts_per_dim_logprobs() -> None:
    log_probs = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
        ]
    )
    valid = torch.tensor([[True, False, True], [False, True, False]])

    out = masked_chunk_sum(log_probs, valid)

    torch.testing.assert_close(out, torch.tensor([14.0, 70.0]))


def test_masked_chunk_reward_sum_ignores_terminal_tail() -> None:
    rewards = torch.tensor([[1.0, 1.0, 100.0], [2.0, 3.0, 4.0]])
    valid = torch.tensor([[True, True, False], [True, False, False]])

    out = masked_chunk_reward_sum(rewards, valid)

    torch.testing.assert_close(out, torch.tensor([2.0, 2.0]))


def test_valid_chunk_any_marks_chunks_with_executed_action() -> None:
    valid = torch.tensor([[False, False], [True, False]])

    out = valid_chunk_any(valid)

    torch.testing.assert_close(out, torch.tensor([False, True]))
