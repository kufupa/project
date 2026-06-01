"""Flow-SDE single-step transition logprob (ports openpi / piRL reference)."""

from __future__ import annotations

import math

import torch


def sde_step_logprob(
    x_next: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Diagonal Gaussian log-density summed over trailing action dims."""
    sigma_safe = torch.clamp(sigma, min=eps)
    var = sigma_safe * sigma_safe
    log_norm = -torch.log(sigma_safe) - 0.5 * math.log(2 * math.pi)
    quad = -0.5 * ((x_next - mu) ** 2) / var
    per_dim = log_norm + quad
    mask = (sigma <= eps).expand_as(per_dim)
    out = per_dim.sum(dim=-1)
    return torch.where(mask.any(dim=-1), torch.zeros_like(out), out)


def sde_step_params(
    x_tau: torch.Tensor,
    v_tau: torch.Tensor,
    tau: torch.Tensor,
    delta: torch.Tensor,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map openpi Flow-SDE hybrid step to (mu, sigma) for one transition."""
    t_input = tau
    delta_b = delta
    x0_pred = x_tau - v_tau * t_input
    x1_pred = x_tau + v_tau * (1.0 - t_input)
    denom = torch.where(tau == 1, torch.tensor(1.0, device=tau.device, dtype=tau.dtype), 1.0 - tau)
    sigma_ratio = tau / denom.clamp(min=1e-6)
    sigmas = float(noise_level) * torch.sqrt(sigma_ratio.clamp(min=0.0))
    sigma_i = sigmas
    x0_weight = torch.ones_like(t_input) - (t_input - delta_b)
    x1_weight = t_input - delta_b - (sigma_i**2) * delta_b / (2 * t_input.clamp(min=1e-6))
    mu = x0_pred * x0_weight + x1_pred * x1_weight
    x_t_std = torch.sqrt(delta_b.clamp(min=0.0)) * sigma_i
    return mu, x_t_std
