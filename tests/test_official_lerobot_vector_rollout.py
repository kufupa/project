from __future__ import annotations

import numpy as np
import torch

from smolvla_grpo.official_lerobot_vector_rollout import slice_proc_row


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
