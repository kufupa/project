"""Phase12 WM-only rollout helpers: score chunks without selected env stepping."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from smolvla_grpo.phase12_rollout import (
    Phase12EpisodeResult,
    Phase12SegmentRecord,
    _candidate_from_sample,
    select_best_candidate,
)


def _score_value(score: Any, key: str) -> float:
    if isinstance(score, dict):
        return float(score[key])
    return float(getattr(score, key))


def collect_phase12_wm_only_episode(
    *,
    root_source: Any,
    reset_seed: int,
    policy_sampler: Callable[..., Iterable[Any]],
    score_fn: Callable[..., Any],
    score_candidates_fn: Callable[..., Sequence[Any]] | None = None,
    goals: Sequence[Any],
    group_size: int,
    reward_key: str,
    action_profile: str = "official_jepa_mirror",
    action_low: float | np.ndarray | None = None,
    action_high: float | np.ndarray | None = None,
    preprocessor: Any | None = None,
    env_action_dim: int | None = None,
    wm_action_dim: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Phase12EpisodeResult:
    root = root_source.reset(int(reset_seed))
    low = action_low if action_low is not None else np.full((int(env_action_dim or 4),), -1.0, dtype=np.float32)
    high = action_high if action_high is not None else np.full((int(env_action_dim or 4),), 1.0, dtype=np.float32)

    segments: list[Phase12SegmentRecord] = []
    selected_candidate_indices: list[int] = []
    segment_candidate_rewards: list[list[float]] = []
    old_logprob_sums: list[float] = []
    proc_root_snapshots: list[Any] = []
    unsquashed_chunks: list[Any] = []

    for segment_index, goal in enumerate(goals):
        samples = list(
            policy_sampler(
                root,
                num_candidates=int(group_size),
                segment_index=int(segment_index),
                goal=goal,
            )
        )
        candidates = [
            _candidate_from_sample(
                sample,
                default_index=i,
                root_snapshot=root.get("proc", root),
                action_profile=action_profile,
                action_low=low,
                action_high=high,
                preprocessor=preprocessor,
                env_action_dim=env_action_dim,
                wm_action_dim=wm_action_dim,
            )
            for i, sample in enumerate(samples)
        ]
        if len(candidates) != int(group_size):
            raise ValueError(f"policy_sampler returned {len(candidates)} candidates, expected {group_size}")
        if score_candidates_fn is not None:
            scores = list(score_candidates_fn(root, candidates, goal, segment_index=int(segment_index)))
        else:
            scores = [
                score_fn(root, candidate, goal, segment_index=int(segment_index))
                for candidate in candidates
            ]
        selected = select_best_candidate(scores, reward_key=reward_key)
        selected_candidate_indices.append(int(selected))
        segment_candidate_rewards.append([_score_value(score, reward_key) for score in scores])
        old_logprob_sums.extend(float(candidate.old_logprob_sum) for candidate in candidates)
        proc_root_snapshots.extend(candidate.proc_root_snapshot for candidate in candidates)
        unsquashed_chunks.extend(candidate.unsquashed_chunk for candidate in candidates)
        segments.append(
            Phase12SegmentRecord(
                update_index=0,
                episode_index=0,
                segment_index=int(segment_index),
                goal_frame_index_1based=int(getattr(goal, "frame_index_1based", segment_index + 1)),
                selected_candidate_index=int(selected),
                scores=list(scores),
                candidates=list(candidates),
                success_any=False,
                success_last=False,
                env_reward_sum=0.0,
                decode_metadata={"wm_only": True},
            )
        )

    meta = dict(metadata or {})
    meta.update(
        {
            "phase12_train_mode": "wm_only",
            "candidate_rewards": [reward for row in segment_candidate_rewards for reward in row],
            "segment_candidate_rewards": segment_candidate_rewards,
            "selected_candidate_indices": selected_candidate_indices,
            "old_logprob_sums": old_logprob_sums,
            "proc_root_snapshots": proc_root_snapshots,
            "unsquashed_chunks": unsquashed_chunks,
            "success_any": False,
            "success_last": False,
        }
    )
    return Phase12EpisodeResult(
        segments=segments,
        total_env_reward=0.0,
        success_any=False,
        success_last=False,
        metadata=meta,
    )
