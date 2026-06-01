from __future__ import annotations

import torch

from smolvla_grpo.phase11_chunk_rollout import build_valid_tail_mask, chunk_success_any


def test_build_valid_tail_mask_excludes_actions_after_terminal() -> None:
    mask = build_valid_tail_mask(chunk_len=5, executed_count=2)

    torch.testing.assert_close(mask, torch.tensor([True, True, False, False, False]))


def test_build_valid_tail_mask_all_valid_when_no_terminal() -> None:
    mask = build_valid_tail_mask(chunk_len=4, executed_count=4)

    torch.testing.assert_close(mask, torch.tensor([True, True, True, True]))


def test_chunk_success_any_uses_only_valid_rows() -> None:
    successes = [False, False, True]
    valid = torch.tensor([True, False, False])

    assert chunk_success_any(successes, valid) is False


def test_chunk_success_any_detects_valid_success() -> None:
    successes = [False, True, False]
    valid = torch.tensor([True, True, False])

    assert chunk_success_any(successes, valid) is True
