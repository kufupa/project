from __future__ import annotations

import math

import torch

from smolvla_grpo.flow_logprob import sde_step_logprob, sde_step_logprob_per_dim, sde_step_params


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


def _openpi_flow_sde_reference(
    x_tau: torch.Tensor,
    v_tau: torch.Tensor,
    *,
    timestep_index: int,
    denoise_steps: int,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=x_tau.device, dtype=x_tau.dtype)
    timesteps = torch.cat([timesteps, torch.tensor([0.0], device=x_tau.device, dtype=x_tau.dtype)])
    idx = torch.full((x_tau.shape[0],), int(timestep_index), device=x_tau.device, dtype=torch.long)
    t_input = timesteps[idx][:, None, None].expand_as(x_tau)
    delta = (timesteps[idx] - timesteps[idx + 1])[:, None, None].expand_as(x_tau)
    x0_pred = x_tau - v_tau * t_input
    x1_pred = x_tau + v_tau * (1 - t_input)
    denom_timesteps = torch.where(timesteps == 1, timesteps[1], timesteps)
    sigma_ratio = timesteps / (1 - denom_timesteps)
    sigmas = noise_level * torch.sqrt(sigma_ratio)[:-1]
    sigma_i = sigmas[idx][:, None, None].expand_as(x_tau)
    x0_weight = torch.ones_like(t_input) - (t_input - delta)
    x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
    return x0_pred * x0_weight + x1_pred * x1_weight, torch.sqrt(delta) * sigma_i


def test_sde_step_params_matches_openpi_first_step_sigma() -> None:
    x_tau = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10.0
    v_tau = torch.full_like(x_tau, 0.25)
    tau = torch.ones_like(x_tau)
    delta = torch.full_like(x_tau, 0.1)

    mu, sigma = sde_step_params(x_tau, v_tau, tau, delta, noise_level=0.5)
    ref_mu, ref_sigma = _openpi_flow_sde_reference(
        x_tau,
        v_tau,
        timestep_index=0,
        denoise_steps=10,
        noise_level=0.5,
    )

    assert torch.allclose(mu, ref_mu, atol=1e-6)
    assert torch.allclose(sigma, ref_sigma, atol=1e-6)


def test_sde_step_params_matches_openpi_middle_step() -> None:
    x_tau = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10.0
    v_tau = torch.full_like(x_tau, -0.1)
    tau = torch.full_like(x_tau, 0.7)
    delta = torch.full_like(x_tau, 0.1)

    mu, sigma = sde_step_params(x_tau, v_tau, tau, delta, noise_level=0.3)
    ref_mu, ref_sigma = _openpi_flow_sde_reference(
        x_tau,
        v_tau,
        timestep_index=3,
        denoise_steps=10,
        noise_level=0.3,
    )

    assert torch.allclose(mu, ref_mu, atol=1e-6)
    assert torch.allclose(sigma, ref_sigma, atol=1e-6)


def test_logprob_per_dim_matches_openpi_zero_sigma_mask() -> None:
    mu = torch.zeros(1, 4)
    sigma = torch.tensor([[0.1, 0.0, 0.2, 0.0]])
    sample = torch.tensor([[0.1, 10.0, -0.1, -10.0]])
    per_dim = sde_step_logprob_per_dim(sample, mu, sigma)

    sigma_safe = torch.where(sigma == 0, torch.ones_like(sigma), sigma)
    expected = -torch.log(sigma_safe) - 0.5 * torch.log(2 * torch.pi * torch.ones_like(sample))
    expected = expected - 0.5 * ((sample - mu) / sigma_safe) ** 2
    expected = torch.where(sigma == 0, torch.zeros_like(expected), expected)

    assert torch.allclose(per_dim, expected, atol=1e-6)
    assert torch.allclose(sde_step_logprob(sample, mu, sigma), expected.sum(dim=-1), atol=1e-6)


def test_sde_logprob_uses_fp32_for_low_precision_inputs() -> None:
    mu = torch.zeros(2, 4, dtype=torch.bfloat16)
    sigma = torch.full((2, 4), 0.1, dtype=torch.bfloat16)
    x_next = torch.full((2, 4), 0.05, dtype=torch.bfloat16)

    per_dim = sde_step_logprob_per_dim(x_next, mu, sigma)
    summed = sde_step_logprob(x_next, mu, sigma)

    assert per_dim.dtype == torch.float32
    assert summed.dtype == torch.float32
    assert torch.isfinite(summed).all()


def test_sde_step_params_uses_fp32_for_low_precision_inputs() -> None:
    x_tau = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4) / 10
    v_tau = torch.full_like(x_tau, 0.25)
    tau = torch.ones_like(x_tau)
    delta = torch.full_like(x_tau, 0.1)

    mu, sigma = sde_step_params(x_tau, v_tau, tau, delta, noise_level=0.5)

    assert mu.dtype == torch.float32
    assert sigma.dtype == torch.float32
    assert torch.isfinite(mu).all()
    assert torch.isfinite(sigma).all()
