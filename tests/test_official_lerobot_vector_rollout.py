from __future__ import annotations

import numpy as np
import torch

from smolvla_grpo.official_lerobot_vector_rollout import (
    concat_sampled_action_batches,
    concat_sampled_action_chunk_batches,
    iter_compact_row_slices,
    slice_proc_row,
)
from smolvla_grpo.policy_wrapper import SampledActionChunkBatch, SampledBatchStep


def test_slice_proc_row_keeps_batch_dim_one() -> None:
    proc = {
        "observation.image": torch.zeros(3, 2, 4, 4),
        "observation.state": torch.ones(3, 5),
        "task": ["a", "b", "c"],
    }
    row1 = slice_proc_row(proc, 1)
    assert row1["observation.image"].shape == (1, 2, 4, 4)
    assert row1["observation.state"].shape == (1, 5)
    assert row1["task"] == ["b"]


def test_iter_compact_row_slices_splits_and_covers() -> None:
    assert iter_compact_row_slices(0, 32) == []
    assert iter_compact_row_slices(6, 2) == [(0, 2), (2, 4), (4, 6)]
    assert iter_compact_row_slices(10, 32) == [(0, 10)]


def test_concat_sampled_action_batches_restores_order() -> None:
    first = SampledBatchStep(
        exec_action_np=np.asarray([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32),
        raw_postprocessed_action_np=np.asarray([[11, 11, 11, 11], [12, 12, 12, 12]], dtype=np.float32),
        policy_tensor=torch.tensor([[21, 21, 21, 21], [22, 22, 22, 22]], dtype=torch.float32),
        unsquashed=torch.tensor([[31, 31, 31, 31], [32, 32, 32, 32]], dtype=torch.float32),
        log_prob=torch.tensor([41, 42], dtype=torch.float32),
        action_clip_fraction=np.asarray([0.1, 0.2], dtype=np.float64),
        action_clip_any=np.asarray([False, True], dtype=np.bool_),
    )
    second = SampledBatchStep(
        exec_action_np=np.asarray([[3, 3, 3, 3]], dtype=np.float32),
        raw_postprocessed_action_np=np.asarray([[13, 13, 13, 13]], dtype=np.float32),
        policy_tensor=torch.tensor([[23, 23, 23, 23]], dtype=torch.float32),
        unsquashed=torch.tensor([[33, 33, 33, 33]], dtype=torch.float32),
        log_prob=torch.tensor([43], dtype=torch.float32),
        action_clip_fraction=np.asarray([0.3], dtype=np.float64),
        action_clip_any=np.asarray([False], dtype=np.bool_),
    )
    out = concat_sampled_action_batches([first, second])
    assert out.exec_action_np.shape == (3, 4)
    assert out.exec_action_np[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert out.log_prob.tolist() == [41.0, 42.0, 43.0]
    assert out.action_clip_any.tolist() == [False, True, False]


def test_concat_sampled_action_chunk_batches_restores_order() -> None:
    first_actions = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    second_actions = torch.arange(100, 100 + 1 * 3 * 4, dtype=torch.float32).reshape(1, 3, 4)
    first = SampledActionChunkBatch(
        exec_action_np=first_actions.numpy(),
        raw_postprocessed_action_np=(first_actions + 10).numpy(),
        policy_tensor=first_actions + 20,
        unsquashed_chunk=first_actions + 30,
        log_prob_steps=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
        log_prob_sum=torch.tensor([6, 15], dtype=torch.float32),
        action_clip_fraction=np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
        action_clip_any=np.asarray([[False, False, True], [False, True, True]], dtype=np.bool_),
    )
    second = SampledActionChunkBatch(
        exec_action_np=second_actions.numpy(),
        raw_postprocessed_action_np=(second_actions + 10).numpy(),
        policy_tensor=second_actions + 20,
        unsquashed_chunk=second_actions + 30,
        log_prob_steps=torch.tensor([[7, 8, 9]], dtype=torch.float32),
        log_prob_sum=torch.tensor([24], dtype=torch.float32),
        action_clip_fraction=np.asarray([[0.7, 0.8, 0.9]], dtype=np.float64),
        action_clip_any=np.asarray([[False, False, False]], dtype=np.bool_),
    )
    out = concat_sampled_action_chunk_batches([first, second])
    assert out.exec_action_np.shape == (3, 3, 4)
    assert out.exec_action_np[:, 0, 0].tolist() == [0.0, 12.0, 100.0]
    assert out.log_prob_steps.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
