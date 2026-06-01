"""Flow-SDE single-step transition logprob (ports openpi / piRL reference)."""

from __future__ import annotations

import math

import torch


def _fp32(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float32 else t.float()


def sde_step_logprob_per_dim(
    x_next: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """OpenPI-style per-dimension diagonal Gaussian log-density."""
    x_next = _fp32(x_next)
    mu = _fp32(mu)
    sigma = _fp32(sigma)
    mask = sigma <= eps
    sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
    log_norm = -torch.log(sigma_safe) - 0.5 * math.log(2 * math.pi)
    quad = -0.5 * ((x_next - mu) / sigma_safe) ** 2
    per_dim = log_norm + quad
    return torch.where(mask.expand_as(per_dim), torch.zeros_like(per_dim), per_dim)


def sde_step_logprob(
    x_next: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Diagonal Gaussian log-density summed over trailing action dims."""
    return sde_step_logprob_per_dim(x_next, mu, sigma, eps=eps).sum(dim=-1)


def _flow_sde_denom_timestep(tau: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """OpenPI replaces tau=1 with the next timestep before sigma_ratio."""
    next_tau = tau - delta
    return torch.where(torch.isclose(tau, torch.ones_like(tau)), next_tau, tau)


def sde_step_params(
    x_tau: torch.Tensor,
    v_tau: torch.Tensor,
    tau: torch.Tensor,
    delta: torch.Tensor,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map openpi Flow-SDE hybrid step to (mu, sigma) for one transition."""
    x_tau = _fp32(x_tau)
    v_tau = _fp32(v_tau)
    tau = _fp32(tau)
    delta = _fp32(delta)
    t_input = tau
    delta_b = delta
    x0_pred = x_tau - v_tau * t_input
    x1_pred = x_tau + v_tau * (1.0 - t_input)
    denom_timestep = _flow_sde_denom_timestep(tau, delta_b)
    sigma_ratio = tau / (1.0 - denom_timestep).clamp(min=1e-6)
    sigmas = float(noise_level) * torch.sqrt(sigma_ratio.clamp(min=0.0))
    sigma_i = sigmas
    x0_weight = torch.ones_like(t_input) - (t_input - delta_b)
    x1_weight = t_input - delta_b - (sigma_i**2) * delta_b / (2 * t_input.clamp(min=1e-6))
    mu = x0_pred * x0_weight + x1_pred * x1_weight
    x_t_std = torch.sqrt(delta_b.clamp(min=0.0)) * sigma_i
    return mu, x_t_std
