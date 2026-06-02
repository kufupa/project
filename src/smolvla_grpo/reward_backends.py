"""Reward backends: Phase11 (env) and Phase12 (WM latent) — same GRPO math, different return signal."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, Protocol, Sequence

import torch


class TrajectoryForReward(Protocol):
    rewards: Sequence[float]
    metadata: dict[str, Any]


class RewardBackend(ABC):
    """Compute scalar return used for group-relative advantages."""

    @abstractmethod
    def episode_return(self, traj: Any) -> float:
        ...


class EnvRewardBackend(RewardBackend):
    """Phase11: sum of environment step rewards."""

    def episode_return(self, traj: Any) -> float:
        rewards = getattr(traj, "rewards", None)
        if rewards is None and isinstance(traj, dict):
            rewards = traj.get("rewards", [])
        return float(sum(float(r) for r in rewards))


def _iter_chunks(traj: Any) -> Sequence[Any]:
    chunks = getattr(traj, "chunks", None)
    if chunks is None and isinstance(traj, dict):
        chunks = traj.get("chunks")
    return chunks or ()


def _valid_dense_return(traj: Any) -> float:
    chunks = _iter_chunks(traj)
    if chunks:
        total = 0.0
        for chunk in chunks:
            rewards = torch.as_tensor(getattr(chunk, "rewards"))
            valid = torch.as_tensor(getattr(chunk, "valid_action_mask")).bool()
            total += float((rewards.float() * valid.float()).sum().item())
        return total
    return EnvRewardBackend().episode_return(traj)


def _valid_success_once(traj: Any) -> bool:
    chunks = _iter_chunks(traj)
    if chunks:
        for chunk in chunks:
            successes = torch.as_tensor(getattr(chunk, "successes")).bool()
            valid = torch.as_tensor(getattr(chunk, "valid_action_mask")).bool()
            if bool((successes & valid).any().item()):
                return True
        return False
    successes = getattr(traj, "successes", None)
    if successes is None and isinstance(traj, dict):
        successes = traj.get("successes", [])
    return any(bool(x) for x in (successes or []))


def episode_return_for_mode(traj: Any, *, reward_mode: str) -> float:
    """Scalar episode return for Phase11 reward ablations."""
    dense = _valid_dense_return(traj)
    if reward_mode == "dense_return":
        return dense
    success = _valid_success_once(traj)
    if reward_mode == "sparse_success_delta":
        return 1.0 if success else 0.0
    if reward_mode == "success_first_dense":
        dense_tiebreak = math.tanh(dense / 1000.0) * 1e-3
        return (1.0 if success else 0.0) + dense_tiebreak
    raise ValueError(
        "reward_mode must be one of: dense_return, sparse_success_delta, success_first_dense"
    )


class WMLatentRewardBackend(RewardBackend):
    """Phase12: latent-distance-based return (set on rollout metadata for now).

    Expects ``traj.metadata['latent_return']`` or ``traj.metadata['wm_latent_progress']``,
    or the same keys on a dict trajectory. Full WM wiring reuses ``segment_grpo_loop``
    helpers in a later trainer entrypoint.
    """

    def episode_return(self, traj: Any) -> float:
        meta = getattr(traj, "metadata", None) or (
            traj.get("metadata") if isinstance(traj, dict) else {}
        )
        if not isinstance(meta, dict):
            meta = {}
        if "latent_return" in meta:
            return float(meta["latent_return"])
        if "wm_latent_progress" in meta:
            return float(meta["wm_latent_progress"])
        raise KeyError(
            "WMLatentRewardBackend: set metadata['latent_return'] or implement WM scoring on rollout."
        )
