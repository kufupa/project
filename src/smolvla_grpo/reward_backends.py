"""Reward backends: Phase11 (env) and Phase12 (WM latent) — same GRPO math, different return signal."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, Sequence


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
