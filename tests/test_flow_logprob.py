from __future__ import annotations

import math

import torch

from smolvla_grpo.flow_logprob import sde_step_logprob


def test_flow_sde_transition_logprob_matches_gaussian() -> None:
    mu = torch.zeros(2, 4)
    sigma = torch.full((2, 4), 0.1)
    x_next = mu + sigma * torch.randn_like(mu)
    lp = sde_step_logprob(x_next, mu, sigma)
    man = (-torch.log(sigma) - 0.5 * math.log(2 * math.pi) - 0.5 * ((x_next - mu) / sigma) ** 2).sum(
        -1
    )
    assert torch.allclose(lp, man, atol=1e-5)


def test_sigma_zero_is_safe() -> None:
    mu = torch.zeros(1, 4)
    lp = sde_step_logprob(mu.clone(), mu, torch.zeros(1, 4))
    assert torch.isfinite(lp).all()
