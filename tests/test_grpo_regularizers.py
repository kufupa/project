from __future__ import annotations

import torch

from smolvla_grpo.grpo_math import gaussian_entropy, kl_to_reference
from smolvla_grpo.policy_wrapper import MetaWorldSmolVLAGRPOPolicy


def test_log_std_floor_applied() -> None:
    out = MetaWorldSmolVLAGRPOPolicy.clamp_log_std(torch.full((4,), -10.0), min_log_std=-4.0)
    assert torch.all(out >= -4.0 - 1e-6)


def test_entropy_and_kl() -> None:
    log_std = torch.full((4, 4), -2.0)
    assert gaussian_entropy(log_std).shape == (4,)
    assert torch.allclose(kl_to_reference(torch.zeros(4), torch.zeros(4)), torch.zeros(4), atol=1e-6)
