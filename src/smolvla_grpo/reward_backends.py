"""Reward backends: Phase11 (env) and Phase12 (WM latent) — same GRPO math, different return signal."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, Literal, Protocol, Sequence

RewardMode = Literal["dense_return", "sparse_success_delta", "success_first_dense"]

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


def _trajectory_successes(traj: Any) -> list[bool]:
    successes = getattr(traj, "successes", None)
    if successes is None and isinstance(traj, dict):
        successes = traj.get("successes", [])
    return [bool(s) for s in (successes or [])]


def compute_sparse_success_delta_return(
    successes: Sequence[bool],
    *,
    reward_coef: float = 1.0,
    use_rel_reward: bool = True,
) -> float:
    """Episode return from sparse success (+ optional relative shaping).

    Matches RLinf ``smolvla_metaworld_env._step_reward`` for sparse_success_delta.
    """
    coef = float(reward_coef)
    prev = 0.0
    total = 0.0
    for success in successes:
        reward = coef * float(bool(success))
        if use_rel_reward:
            step = reward - prev
            prev = reward
        else:
            step = reward
        total += step
    return total


class SparseSuccessDeltaBackend(RewardBackend):
    """Phase11 sparse MetaWorld reward: success indicator with optional rel delta."""

    def __init__(
        self,
        *,
        reward_coef: float = 1.0,
        use_rel_reward: bool = True,
        success_bonus: float = 0.0,
        clip_penalty: float = 0.0,
    ) -> None:
        self.reward_coef = float(reward_coef)
        self.use_rel_reward = bool(use_rel_reward)
        self.success_bonus = float(success_bonus)
        self.clip_penalty = float(clip_penalty)

    def episode_return(self, traj: Any) -> float:
        total = compute_sparse_success_delta_return(
            _trajectory_successes(traj),
            reward_coef=self.reward_coef,
            use_rel_reward=self.use_rel_reward,
        )
        if self.success_bonus and any(_trajectory_successes(traj)):
            total += self.success_bonus
        if self.clip_penalty:
            clip_fractions = getattr(traj, "action_clip_fractions", None)
            if clip_fractions is None and isinstance(traj, dict):
                clip_fractions = traj.get("action_clip_fractions", [])
            total -= self.clip_penalty * float(
                sum(float(v) for v in (clip_fractions or []))
            )
        return total


def make_phase11_reward_backend(
    *,
    reward_mode: RewardMode = "dense_return",
    reward_coef: float = 1.0,
    use_rel_reward: bool = True,
    success_bonus: float = 0.0,
    clip_penalty: float = 0.0,
) -> RewardBackend:
    if reward_mode == "dense_return":
        return EnvRewardBackend(
            success_bonus=float(success_bonus),
            clip_penalty=float(clip_penalty),
        )
    if reward_mode == "sparse_success_delta":
        return SparseSuccessDeltaBackend(
            reward_coef=float(reward_coef),
            use_rel_reward=bool(use_rel_reward),
            success_bonus=float(success_bonus),
            clip_penalty=float(clip_penalty),
        )
    if reward_mode == "success_first_dense":
        return _EpisodeReturnModeBackend(reward_mode="success_first_dense")
    raise ValueError(f"unsupported reward_mode={reward_mode!r}")


class _EpisodeReturnModeBackend(RewardBackend):
    """Delegate scalar return to ``episode_return_for_mode`` (chunk-aware dense/success)."""

    def __init__(self, *, reward_mode: str) -> None:
        self._reward_mode = str(reward_mode)

    def episode_return(self, traj: Any) -> float:
        return episode_return_for_mode(traj, reward_mode=self._reward_mode)


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
