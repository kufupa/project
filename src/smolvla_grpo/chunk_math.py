"""Chunk-level helpers for SmolVLA GRPO."""

from __future__ import annotations

import torch


def _validate_valid_mask(valid_action_mask: torch.Tensor) -> torch.Tensor:
    if valid_action_mask.ndim != 2:
        raise ValueError(f"valid_action_mask must be [B,H], got {tuple(valid_action_mask.shape)}")
    return valid_action_mask.bool()


def masked_chunk_sum(log_probs: torch.Tensor, valid_action_mask: torch.Tensor) -> torch.Tensor:
    """Sum logprobs over valid chunk rows.

    Accepts logprobs already summed over action dims as [B,H], or per-dim
    logprobs as [B,H,D].
    """
    valid = _validate_valid_mask(valid_action_mask).to(device=log_probs.device)
    if log_probs.ndim == 2:
        if tuple(log_probs.shape) != tuple(valid.shape):
            raise ValueError(f"log_probs [B,H] shape {tuple(log_probs.shape)} != valid {tuple(valid.shape)}")
        return (log_probs.float() * valid.to(dtype=log_probs.dtype)).sum(dim=1)
    if log_probs.ndim == 3:
        if tuple(log_probs.shape[:2]) != tuple(valid.shape):
            raise ValueError(
                f"log_probs [B,H,D] shape {tuple(log_probs.shape)} incompatible with valid {tuple(valid.shape)}"
            )
        return (log_probs.float() * valid.to(dtype=log_probs.dtype).unsqueeze(-1)).sum(dim=(1, 2))
    raise ValueError(f"log_probs must be [B,H] or [B,H,D], got {tuple(log_probs.shape)}")


def masked_chunk_reward_sum(rewards: torch.Tensor, valid_action_mask: torch.Tensor) -> torch.Tensor:
    valid = _validate_valid_mask(valid_action_mask).to(device=rewards.device, dtype=rewards.dtype)
    if rewards.ndim != 2 or tuple(rewards.shape) != tuple(valid.shape):
        raise ValueError(
            f"rewards must match valid mask [B,H], got rewards={tuple(rewards.shape)} valid={tuple(valid.shape)}"
        )
    return (rewards.float() * valid.float()).sum(dim=1)


def valid_chunk_any(valid_action_mask: torch.Tensor) -> torch.Tensor:
    return _validate_valid_mask(valid_action_mask).any(dim=1)
