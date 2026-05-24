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
    """Phase11: sum of environment step rewards, optionally aligned to success."""

    def __init__(
        self,
        *,
        success_bonus: float = 0.0,
        clip_penalty: float = 0.0,
    ) -> None:
        self.success_bonus = float(success_bonus)
        self.clip_penalty = float(clip_penalty)

    def episode_return(self, traj: Any) -> float:
        rewards = getattr(traj, "rewards", None)
        if rewards is None and isinstance(traj, dict):
            rewards = traj.get("rewards", [])
        total = float(sum(float(r) for r in rewards))
        if self.success_bonus:
            successes = getattr(traj, "successes", None)
            if successes is None and isinstance(traj, dict):
                successes = traj.get("successes", [])
            if any(bool(s) for s in (successes or [])):
                total += self.success_bonus
        if self.clip_penalty:
            clip_fractions = getattr(traj, "action_clip_fractions", None)
            if clip_fractions is None and isinstance(traj, dict):
                clip_fractions = traj.get("action_clip_fractions", [])
            total -= self.clip_penalty * float(
                sum(float(v) for v in (clip_fractions or []))
            )
        return total


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
