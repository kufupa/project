"""Pure GRPO/PPO-style clipped surrogate helpers (no torch dependency on policy)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    """Per-batch-row entropy of diagonal Gaussian (sum over action dims)."""
    return (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum(dim=-1)


def kl_to_reference(current_log_probs: torch.Tensor, reference_log_probs: torch.Tensor) -> torch.Tensor:
    """KL(ref || cur) ≈ E[log pi_ref - log pi_cur] on stored actions."""
    return (reference_log_probs.reshape(-1) - current_log_probs.reshape(-1)).float()


def apply_grpo_regularizers(
    base_loss: torch.Tensor,
    *,
    current_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    log_std: torch.Tensor,
    kl_beta: float,
    entropy_coef: float,
) -> torch.Tensor:
    """Add measured KL penalty and entropy bonus to an already-scaled GRPO loss."""
    loss = base_loss
    if float(kl_beta) != 0.0:
        kl = kl_to_reference(current_log_probs, reference_log_probs).mean()
        loss = loss + float(kl_beta) * kl
    if float(entropy_coef) != 0.0:
        entropy = gaussian_entropy(log_std.reshape(-1, log_std.shape[-1])).mean()
        loss = loss - float(entropy_coef) * entropy
    return loss


def update_metrics(
    *,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    log_std: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float,
) -> dict[str, float | bool | int]:
    """Phase-A per-update diagnostics for GRPO collapse triage."""
    nlp = new_log_probs.detach().reshape(-1).float()
    olp = old_log_probs.detach().reshape(-1).float()
    if nlp.shape != olp.shape:
        raise ValueError(f"logprob shape mismatch: old={tuple(olp.shape)} new={tuple(nlp.shape)}")

    if nlp.numel() == 0:
        ratio = torch.ones(0, dtype=torch.float32, device=nlp.device)
        approx_kl = 0.0
        ratio_min = ratio_max = ratio_mean = 1.0
        ratio_clip_fraction = 0.0
    else:
        log_ratio = nlp - olp
        ratio = torch.exp(log_ratio)
        approx_kl = float((olp - nlp).mean().item())
        ratio_min = float(ratio.min().item())
        ratio_max = float(ratio.max().item())
        ratio_mean = float(ratio.mean().item())
        low = 1.0 - float(epsilon)
        high = 1.0 + float(epsilon)
        ratio_clip_fraction = float(((ratio < low) | (ratio > high)).float().mean().item())

    ls = log_std.detach().reshape(-1).float()
    if ls.numel() == 0:
        log_std_mean = log_std_min = log_std_max = 0.0
    else:
        log_std_mean = float(ls.mean().item())
        log_std_min = float(ls.min().item())
        log_std_max = float(ls.max().item())

    ret = returns.detach().reshape(-1).float()
    if ret.numel() <= 1:
        group_return_std = 0.0
    else:
        group_return_std = float(torch.nan_to_num(ret.std(unbiased=True), nan=0.0).item())

    adv = advantages.detach().reshape(-1).float()
    zero_adv_skip = bool(adv.numel() > 0 and torch.allclose(adv, torch.zeros_like(adv)))

    return {
        "approx_kl": approx_kl,
        "ratio_mean": ratio_mean,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "ratio_clip_fraction": ratio_clip_fraction,
        "log_std_mean": log_std_mean,
        "log_std_min": log_std_min,
        "log_std_max": log_std_max,
        "group_return_std": group_return_std,
        "zero_adv_skip": zero_adv_skip,
        "n_logprob": int(nlp.numel()),
    }


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


@dataclass
class LogprobRatioParityStats:
    """Old vs recomputed logprob under identical weights and stored actions."""

    n: int
    mean_ratio: float
    std_ratio: float
    min_ratio: float
    max_ratio: float
    max_abs_log_ratio: float
    within_tolerance: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "n": self.n,
            "mean_ratio": self.mean_ratio,
            "std_ratio": self.std_ratio,
            "min_ratio": self.min_ratio,
            "max_ratio": self.max_ratio,
            "max_abs_log_ratio": self.max_abs_log_ratio,
            "within_tolerance": self.within_tolerance,
        }


def summarize_logprob_ratio_parity(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    *,
    tolerance: float = 0.02,
) -> LogprobRatioParityStats:
    """Check exp(new-old) ~ 1 for identical weights and stored policy-space actions."""
    olp = old_log_probs.reshape(-1).float()
    nlp = new_log_probs.reshape(-1).float()
    if olp.shape != nlp.shape:
        raise ValueError(f"logprob shape mismatch: old={tuple(olp.shape)} new={tuple(nlp.shape)}")
    n = int(olp.numel())
    if n == 0:
        return LogprobRatioParityStats(
            n=0,
            mean_ratio=1.0,
            std_ratio=0.0,
            min_ratio=1.0,
            max_ratio=1.0,
            max_abs_log_ratio=0.0,
            within_tolerance=True,
        )
    log_ratio = (nlp - olp).detach()
    ratio = torch.exp(log_ratio)
    mean_ratio = float(ratio.mean().item())
    std_ratio = float(ratio.std(unbiased=n > 1).item()) if n > 1 else 0.0
    min_ratio = float(ratio.min().item())
    max_ratio = float(ratio.max().item())
    max_abs_log_ratio = float(log_ratio.abs().max().item())
    within = (
        abs(mean_ratio - 1.0) <= tolerance
        and max_abs_log_ratio <= max(tolerance, 1e-6)
    )
    return LogprobRatioParityStats(
        n=n,
        mean_ratio=mean_ratio,
        std_ratio=std_ratio,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        max_abs_log_ratio=max_abs_log_ratio,
        within_tolerance=within,
    )


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


