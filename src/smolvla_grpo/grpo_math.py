"""Pure GRPO/PPO-style clipped surrogate helpers (no torch dependency on policy)."""

from __future__ import annotations

from dataclasses import dataclass
import torch


def compute_group_advantages(returns: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Normalize episode returns within a group (same as safe-robot-steering).

    Args:
        returns: shape (G,) float tensor of total reward per trajectory.

    Returns:
        advantages: shape (G,). All zeros if std is ~0 or non-finite.
    """
    if returns.numel() == 0:
        return returns
    mean_r = returns.mean()
    std_r = returns.std(unbiased=len(returns) > 1)
    mean_r = torch.nan_to_num(mean_r, nan=0.0)
    std_r = torch.nan_to_num(std_r, nan=0.0)
    if std_r.item() < eps:
        return torch.zeros_like(returns)
    adv = (returns - mean_r) / (std_r + eps)
    return torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class RatioStats:
    mean_ratio: float
    max_ratio: float
    clip_fraction: float
    n: int


def summarize_ratio_stats(
    ratio: torch.Tensor, *, epsilon: float
) -> RatioStats:
    """ratio = exp(new_logp - old_logp), same shape as timesteps flattened."""
    flat = ratio.detach().float().reshape(-1)
    n = int(flat.numel())
    if n == 0:
        return RatioStats(0.0, 0.0, 0.0, 0)
    low = 1.0 - epsilon
    high = 1.0 + epsilon
    clipped = torch.clamp(flat, low, high)
    clip_fraction = float((flat < low).sum().item() + (flat > high).sum().item()) / max(n, 1)
    return RatioStats(
        mean_ratio=float(flat.mean().item()),
        max_ratio=float(flat.max().item()),
        clip_fraction=clip_fraction,
        n=n,
    )


def compute_clipped_grpo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantage_scalar: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, RatioStats]:
    """Per-timestep clipped surrogate: -min(r*A, clip(r)*A).

    Args:
        new_log_probs: (T,) or (T,1)
        old_log_probs: same shape
        advantage_scalar: scalar tensor broadcast to timesteps

    Returns:
        mean_loss scalar, ratio stats
    """
    nlp = new_log_probs.reshape(-1).float()
    olp = old_log_probs.reshape(-1).float()
    if torch.isnan(nlp).any() or torch.isnan(olp).any():
        raise ValueError("NaN in log_probs before GRPO loss")
    ratio = torch.exp(nlp - olp)
    A = advantage_scalar.reshape(()).float()
    unclipped = ratio * A
    clipped_r = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    clipped = clipped_r * A
    step_losses = -torch.min(unclipped, clipped)
    stats = summarize_ratio_stats(ratio, epsilon=epsilon)
    return step_losses.mean(), stats


