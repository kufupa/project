"""Distribution-Guided Policy Optimization (arXiv 2605.03327) math, adapted to
continuous diagonal-Gaussian action-chunk policies.

DGPO keeps the GRPO trajectory advantage and redistributes it across the
episode's chunks via a softmax of the per-chunk Hellinger deviation between the
current and a reference policy. This module is pure (CPU/GPU tensors only).
"""
from __future__ import annotations

import torch


def gaussian_hellinger_sq(
    mu_a: torch.Tensor,
    mu_b: torch.Tensor,
    log_std_a: torch.Tensor,
    log_std_b: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Squared Hellinger distance between two diagonal Gaussians, per chunk.

    Inputs are shape [B, T, D] (batch, chunks, action-dim). Returns [B, T] in
    [0, 1]. d = 1 - BC, BC = prod_k sqrt(2 s_a s_b/(s_a^2+s_b^2)) *
    exp(-(mu_a-mu_b)^2 / (4 (s_a^2+s_b^2))).
    """
    var_a = torch.exp(2.0 * log_std_a).clamp(min=eps)
    var_b = torch.exp(2.0 * log_std_b).clamp(min=eps)
    var_sum = var_a + var_b
    coef = torch.sqrt((2.0 * torch.sqrt(var_a * var_b)) / var_sum)
    expo = torch.exp(-((mu_a - mu_b) ** 2) / (2.0 * var_sum))
    bc = (coef * expo).prod(dim=-1)  # Bhattacharyya coefficient over D
    return (1.0 - bc).clamp(0.0, 1.0)


def _normalized_entropy(log_std: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-chunk diagonal-Gaussian differential entropy, min-max normalized to
    [0,1] within each trajectory's valid chunks. Shape [B, T, D] -> [B, T].
    Constant log_std -> constant -> normalized to ones (gate inert)."""
    ent = (log_std + 0.5 * (1.0 + torch.log(torch.tensor(2.0 * torch.pi)))).sum(dim=-1)  # [B,T]
    neg_inf = torch.finfo(ent.dtype).min
    masked = torch.where(mask, ent, torch.full_like(ent, neg_inf))
    e_max = masked.max(dim=1, keepdim=True).values
    masked_pos = torch.where(mask, ent, torch.full_like(ent, torch.finfo(ent.dtype).max))
    e_min = masked_pos.min(dim=1, keepdim=True).values
    rng = (e_max - e_min).clamp(min=eps)
    norm = ((ent - e_min) / rng).clamp(0.0, 1.0)
    # if range collapsed (constant entropy), treat gate as fully open (=1)
    collapsed = (e_max - e_min) <= eps
    return torch.where(collapsed, torch.ones_like(norm), norm)


def dgpo_redistribution_weights(
    deviations: torch.Tensor,
    mask: torch.Tensor,
    tau: float = 0.5,
    kappa: float = 0.0,
    entropy_norm: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Unit-mean softmax redistribution weights w_{i,t} per trajectory.

    deviations, mask: [B, T]. entropy_norm: [B, T] or None.
    Returns w: [B, T], masked chunks = 0, (1/T_i) sum_t w = 1 over valid chunks.
    kappa=0 -> pure deviation (default; SmolVLA entropy gate is inert).
    """
    dev = deviations
    if kappa != 0.0 and entropy_norm is not None:
        score = dev * entropy_norm.clamp(min=eps) ** kappa
    else:
        score = dev
    logits = score / max(float(tau), eps)
    neg_inf = torch.finfo(logits.dtype).min
    logits = torch.where(mask, logits, torch.full_like(logits, neg_inf))
    # numerically-stable softmax over chunk axis (valid only)
    logits = logits - logits.max(dim=1, keepdim=True).values
    exps = torch.where(mask, torch.exp(logits), torch.zeros_like(logits))
    denom = exps.sum(dim=1, keepdim=True).clamp(min=eps)
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1).to(exps.dtype)
    w = counts * exps / denom
    return torch.where(mask, w, torch.zeros_like(w))
