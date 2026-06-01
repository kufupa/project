from __future__ import annotations

import torch

from smolvla_grpo.grpo_math import gaussian_entropy, kl_to_reference, apply_grpo_regularizers
from smolvla_grpo.policy_wrapper import MetaWorldSmolVLAGRPOPolicy


def test_log_std_floor_applied() -> None:
    out = MetaWorldSmolVLAGRPOPolicy.clamp_log_std(torch.full((4,), -10.0), min_log_std=-4.0)
    assert torch.all(out >= -4.0 - 1e-6)


def test_entropy_and_kl() -> None:
    log_std = torch.full((4, 4), -2.0)
    assert gaussian_entropy(log_std).shape == (4,)
    assert torch.allclose(kl_to_reference(torch.zeros(4), torch.zeros(4)), torch.zeros(4), atol=1e-6)


def test_regularizer_coefficients_change_toy_loss() -> None:
    base_loss = torch.tensor(2.0)
    current_log_probs = torch.tensor([-1.5, -1.0, -0.5])
    reference_log_probs = torch.tensor([-1.0, -1.0, -1.0])
    log_std = torch.full((3, 4), -2.0)

    unchanged = apply_grpo_regularizers(
        base_loss,
        current_log_probs=current_log_probs,
        reference_log_probs=reference_log_probs,
        log_std=log_std,
        kl_beta=0.0,
        entropy_coef=0.0,
    )
    with_regularizers = apply_grpo_regularizers(
        base_loss,
        current_log_probs=current_log_probs,
        reference_log_probs=reference_log_probs,
        log_std=log_std,
        kl_beta=0.5,
        entropy_coef=0.01,
    )

    assert torch.allclose(unchanged, base_loss)
    assert not torch.allclose(with_regularizers, base_loss)
