"""Lightweight Phase12 rollout records and helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from smolvla_grpo.phase12_actions import apply_phase12_action_profile


@dataclass(frozen=True)
class Phase12Candidate:
    candidate_index: int
    proc_root_snapshot: Any
    unsquashed_chunk: Any
    old_logprob_steps: Any
    old_logprob_sum: float
    exec_actions_raw_postprocessed: Any
    exec_actions_clipped: Any
    exec_actions_for_env: Any
    exec_actions_for_wm: Any
    action_metadata: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass(frozen=True)
class Phase12SegmentRecord:
    update_index: int
    episode_index: int
    segment_index: int
    goal_frame_index_1based: int
    selected_candidate_index: int
    scores: list[Any]
    candidates: list[Phase12Candidate]
    success_any: bool
    success_last: bool
    env_reward_sum: float
    decode_metadata: dict[str, Any]


@dataclass(frozen=True)
class Phase12EpisodeResult:
    segments: list[Phase12SegmentRecord]
    total_env_reward: float
    success_any: bool
    success_last: bool
    metadata: dict[str, Any]


def chunk_grpo_loss(
    old_logprob_sums: torch.Tensor,
    new_logprob_sums: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped GRPO surrogate over summed chunk log probabilities."""

    old = torch.as_tensor(old_logprob_sums).float()
    new = torch.as_tensor(new_logprob_sums).float()
    adv = torch.as_tensor(advantages).float()
    ratio = torch.exp(new - old)
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()

    detached_ratio = ratio.detach().reshape(-1)
    low = 1.0 - float(clip_eps)
    high = 1.0 + float(clip_eps)
    clip_fraction = ((detached_ratio < low) | (detached_ratio > high)).float().mean()
    stats = {
        "ratio_mean": float(detached_ratio.mean().item()),
        "ratio_max": float(detached_ratio.max().item()),
        "ratio_min": float(detached_ratio.min().item()),
        "ratio_clip_fraction": float(clip_fraction.item()),
        "approx_kl": float((old.detach() - new.detach()).float().mean().item()),
    }
    return loss, stats


def select_best_candidate(scores: Sequence[Any], reward_key: str = "wm_latent_progress") -> int:
    """Select argmax reward, breaking ties by final distance then candidate index."""

    if not scores:
        raise ValueError("scores must contain at least one candidate")

    def key(score: Any) -> tuple[float, float, int]:
        reward = _field(score, reward_key)
        final_distance = _field(score, "final_combined_distance")
        candidate_index = int(_field(score, "candidate_index"))
        return (-float(reward), float(final_distance), candidate_index)

    best = min(scores, key=key)
    return int(_field(best, "candidate_index"))


def collect_phase12_episode(
    *,
    env: Any,
    policy_sampler: Callable[..., Iterable[Any]],
    score_fn: Callable[..., Any],
    score_candidates_fn: Callable[..., Sequence[Any]] | None = None,
    goals: Sequence[Any],
    num_candidates: int,
    update_index: int = 0,
    episode_index: int = 0,
    root_id_fn: Callable[[Any], Any] | None = None,
    reward_key: str = "wm_latent_progress",
    metadata: Mapping[str, Any] | None = None,
    action_profile: str = "official_jepa_mirror",
    action_low: float | np.ndarray | None = None,
    action_high: float | np.ndarray | None = None,
    preprocessor: Any | None = None,
    env_action_dim: int | None = None,
    wm_action_dim: int | None = None,
) -> Phase12EpisodeResult:
    """Collect a lightweight callback-driven Phase12 episode.

    The policy and scoring callbacks receive the same root observation and root
    id for every candidate in a segment. The env is advanced only after selecting
    the best candidate, so the next segment starts from the fresh env observation.
    """

    observation = env.reset()
    segments: list[Phase12SegmentRecord] = []
    total_reward = 0.0
    success_any = False
    success_last = False

    for segment_index, goal in enumerate(goals):
        root_observation = observation
        root_id = _root_id(root_observation, root_id_fn)
        sampled = policy_sampler(
            root_observation,
            root_id=root_id,
            num_candidates=num_candidates,
            segment_index=segment_index,
            goal=goal,
        )
        low, high = _resolve_action_bounds(
            env,
            action_low=action_low,
            action_high=action_high,
            env_action_dim=env_action_dim,
        )
        candidates = [
            _candidate_from_sample(
                sample,
                default_index=i,
                root_snapshot=root_id,
                action_profile=action_profile,
                action_low=low,
                action_high=high,
                preprocessor=preprocessor,
                env_action_dim=env_action_dim,
                wm_action_dim=wm_action_dim,
            )
            for i, sample in enumerate(sampled)
        ]
        if len(candidates) != int(num_candidates):
            raise ValueError(f"policy_sampler returned {len(candidates)} candidates, expected {num_candidates}")

        if score_candidates_fn is not None:
            scores = list(
                score_candidates_fn(
                    root_observation,
                    candidates,
                    goal,
                    root_id=root_id,
                    segment_index=segment_index,
                )
            )
        else:
            scores = [
                score_fn(
                    root_observation,
                    candidate,
                    goal,
                    root_id=root_id,
                    segment_index=segment_index,
                )
                for candidate in candidates
            ]
        selected_index = select_best_candidate(scores, reward_key=reward_key)
        selected = _candidate_by_index(candidates, selected_index)
        segment_reward, observation, segment_success_any, success_last = _execute_chunk(env, selected.exec_actions_for_env)
        total_reward += segment_reward
        success_any = success_any or segment_success_any

        segments.append(
            Phase12SegmentRecord(
                update_index=int(update_index),
                episode_index=int(episode_index),
                segment_index=int(segment_index),
                goal_frame_index_1based=int(getattr(goal, "frame_index_1based", segment_index + 1)),
                selected_candidate_index=int(selected_index),
                scores=list(scores),
                candidates=candidates,
                success_any=bool(success_any),
                success_last=bool(success_last),
                env_reward_sum=float(segment_reward),
                decode_metadata={"root_id": root_id},
            )
        )

    return Phase12EpisodeResult(
        segments=segments,
        total_env_reward=float(total_reward),
        success_any=bool(success_any),
        success_last=bool(success_last),
        metadata=dict(metadata or {}),
    )


def _field(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value[key]
    return getattr(value, key)


def _optional_field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _root_id(observation: Any, root_id_fn: Callable[[Any], Any] | None) -> Any:
    if root_id_fn is not None:
        return root_id_fn(observation)
    if isinstance(observation, Mapping):
        return observation.get("id", observation.get("root_id", observation))
    return getattr(observation, "id", observation)


def _sum_logprobs(old_logprob_steps: Any) -> float:
    if old_logprob_steps is None:
        return 0.0
    return float(np.asarray(old_logprob_steps, dtype=np.float32).sum())


def _candidate_from_sample(
    sample: Any,
    *,
    default_index: int,
    root_snapshot: Any,
    action_profile: str,
    action_low: float | np.ndarray,
    action_high: float | np.ndarray,
    preprocessor: Any | None,
    env_action_dim: int | None,
    wm_action_dim: int | None,
) -> Phase12Candidate:
    candidate_index = int(_optional_field(sample, "candidate_index", default_index))
    old_logprob_steps = _optional_field(sample, "old_logprob_steps", None)
    raw = _optional_field(sample, "exec_actions_raw_postprocessed", None)
    if raw is None:
        raw = _optional_field(sample, "raw_postprocessed_action_np", None)
    if raw is None:
        raw = _optional_field(sample, "exec_action_np", None)
    if raw is None:
        raw = _optional_field(sample, "exec_actions_for_env", None)
    if raw is None:
        raw = _optional_field(sample, "unsquashed_chunk", None)
    if raw is None:
        raise ValueError("sample must provide an action chunk")
    profile = apply_phase12_action_profile(
        np.asarray(raw, dtype=np.float32),
        action_profile=action_profile,
        action_low=action_low,
        action_high=action_high,
        preprocessor=preprocessor,
        env_action_dim=env_action_dim,
        wm_action_dim=wm_action_dim,
    )
    metadata = {**profile.metadata, **dict(_optional_field(sample, "action_metadata", {}) or {})}
    return Phase12Candidate(
        candidate_index=candidate_index,
        proc_root_snapshot=_optional_field(sample, "proc_root_snapshot", root_snapshot),
        unsquashed_chunk=_optional_field(sample, "unsquashed_chunk", None),
        old_logprob_steps=old_logprob_steps,
        old_logprob_sum=float(_optional_field(sample, "old_logprob_sum", _sum_logprobs(old_logprob_steps))),
        exec_actions_raw_postprocessed=profile.exec_actions_raw_postprocessed,
        exec_actions_clipped=profile.exec_actions_clipped,
        exec_actions_for_env=profile.exec_actions_for_env,
        exec_actions_for_wm=profile.exec_actions_for_wm,
        action_metadata=metadata,
    )


def _resolve_action_bounds(
    env: Any,
    *,
    action_low: float | np.ndarray | None,
    action_high: float | np.ndarray | None,
    env_action_dim: int | None,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    if action_low is not None and action_high is not None:
        return action_low, action_high
    space = getattr(env, "action_space", None)
    inner = getattr(env, "inner", None)
    if space is None and inner is not None:
        space = getattr(inner, "action_space", None)
    if space is None and hasattr(env, "vec_env"):
        space = getattr(env.vec_env, "single_action_space", getattr(env.vec_env, "action_space", None))
    if space is not None and hasattr(space, "low") and hasattr(space, "high"):
        return np.asarray(space.low, dtype=np.float32).reshape(-1), np.asarray(space.high, dtype=np.float32).reshape(-1)
    dim = int(env_action_dim or getattr(env, "action_dim", 4))
    return np.full((dim,), -1.0, dtype=np.float32), np.full((dim,), 1.0, dtype=np.float32)


def _candidate_by_index(candidates: Sequence[Phase12Candidate], candidate_index: int) -> Phase12Candidate:
    for candidate in candidates:
        if candidate.candidate_index == candidate_index:
            return candidate
    raise ValueError(f"selected candidate {candidate_index} not found")


def _execute_chunk(env: Any, actions: Any) -> tuple[float, Any, bool, bool]:
    reward_sum = 0.0
    success_any = False
    success_last = False
    observation = None
    for action in np.asarray(actions):
        step_out = env.step(action)
        if len(step_out) == 5:
            observation, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            observation, reward, done, info = step_out
        reward_sum += float(reward)
        success_last = _success_from_step(observation, info)
        success_any = success_any or success_last
        if done:
            break
    return float(reward_sum), observation, bool(success_any), bool(success_last)


def _success_from_step(observation: Any, info: Any) -> bool:
    if isinstance(info, Mapping):
        if "success" in info:
            return bool(info["success"])
        if "is_success" in info:
            return bool(info["is_success"])
    if isinstance(observation, Mapping):
        if "success" in observation:
            return bool(observation["success"])
        if "is_success" in observation:
            return bool(observation["is_success"])
    return False
