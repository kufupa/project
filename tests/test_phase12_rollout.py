from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smolvla_grpo.phase12_objective import Phase12Score
from smolvla_grpo.phase12_rollout import (
    collect_phase12_episode,
    chunk_grpo_loss,
    select_best_candidate,
)


def _score(candidate_index: int, progress: float, final_distance: float) -> Phase12Score:
    return Phase12Score(
        candidate_index=candidate_index,
        start_visual_distance=0.0,
        start_proprio_distance=0.0,
        start_combined_distance=0.0,
        final_visual_distance=final_distance,
        final_proprio_distance=0.0,
        final_combined_distance=final_distance,
        wm_latent_progress=progress,
        latent_return=-final_distance,
        wm_status="ok",
    )


def test_chunk_grpo_loss_uses_summed_chunk_logprobs_and_clips() -> None:
    old = torch.tensor([0.0, 0.0])
    new = torch.tensor([np.log(2.0), np.log(0.5)])
    advantages = torch.tensor([1.0, -1.0])

    loss, stats = chunk_grpo_loss(old, new, advantages, clip_eps=0.2)

    expected_surrogate = torch.tensor([1.2, -0.8])
    torch.testing.assert_close(loss, -expected_surrogate.mean())
    assert stats["ratio_mean"] == pytest.approx(1.25)
    assert stats["ratio_max"] == pytest.approx(2.0)
    assert stats["ratio_min"] == pytest.approx(0.5)
    assert stats["ratio_clip_fraction"] == pytest.approx(1.0)
    assert stats["approx_kl"] == pytest.approx(float((old - new).mean().item()))


def test_select_best_candidate_breaks_ties_by_distance_then_index() -> None:
    scores = [
        _score(candidate_index=2, progress=1.0, final_distance=0.3),
        _score(candidate_index=0, progress=1.0, final_distance=0.2),
        _score(candidate_index=1, progress=1.0, final_distance=0.2),
        _score(candidate_index=3, progress=0.9, final_distance=0.0),
    ]

    assert select_best_candidate(scores) == 0


def test_dummy_rollout_scores_all_candidates_from_same_root_observation() -> None:
    scored_roots: list[tuple[str, int]] = []

    class DummyEnv:
        def reset(self):
            return {"id": "root-0", "success": False}

        def step(self, action):
            return {"id": "root-1", "success": True}, 1.0, True, {"success": True}

    def policy_sampler(root_observation, *, root_id, num_candidates, segment_index, **_kwargs):
        del root_observation, segment_index
        for candidate_index in range(num_candidates):
            yield {
                "candidate_index": candidate_index,
                "proc_root_snapshot": root_id,
                "unsquashed_chunk": np.array([[candidate_index]], dtype=np.float32),
                "old_logprob_steps": np.array([-0.1], dtype=np.float32),
                "exec_actions_for_env": np.array([[candidate_index]], dtype=np.float32),
            }

    def score_fn(root_observation, candidate, goal, *, root_id, segment_index, **_kwargs):
        del root_observation, goal, segment_index
        scored_roots.append((root_id, candidate["candidate_index"]))
        return _score(
            candidate_index=candidate["candidate_index"],
            progress=float(candidate["candidate_index"]),
            final_distance=float(10 - candidate["candidate_index"]),
        )

    result = collect_phase12_episode(
        env=DummyEnv(),
        policy_sampler=policy_sampler,
        score_fn=score_fn,
        goals=["goal-0"],
        num_candidates=3,
        update_index=7,
        episode_index=4,
    )

    assert scored_roots == [("root-0", 0), ("root-0", 1), ("root-0", 2)]
    segment = result.segments[0]
    assert segment.update_index == 7
    assert segment.episode_index == 4
    assert segment.selected_candidate_index == 2
    assert len(segment.candidates) == 3
    assert result.success_any is True
    assert result.success_last is True


def test_collect_phase12_episode_records_goal_frame_index_from_goal() -> None:
    class DummyEnv:
        def reset(self):
            return {"id": "root", "success": False}

        def step(self, action):
            del action
            return {"id": "next", "success": False}, 0.0, True, {}

    def policy_sampler(*_args, **_kwargs):
        yield {
            "candidate_index": 0,
            "proc_root_snapshot": "root",
            "unsquashed_chunk": np.zeros((1, 4), dtype=np.float32),
            "old_logprob_steps": np.zeros(1, dtype=np.float32),
            "exec_actions_for_env": np.zeros((1, 4), dtype=np.float32),
        }

    def score_fn(_root_observation, candidate, _goal, **_kwargs):
        return _score(candidate_index=candidate["candidate_index"], progress=1.0, final_distance=0.0)

    goal = type("Goal", (), {"frame_index_1based": 25})()
    result = collect_phase12_episode(
        env=DummyEnv(),
        policy_sampler=policy_sampler,
        score_fn=score_fn,
        goals=[goal],
        num_candidates=1,
    )

    assert result.segments[0].goal_frame_index_1based == 25


def test_dummy_two_segment_rollout_uses_fresh_observation_after_best_chunk_execution() -> None:
    sampled_roots: list[str] = []
    scored_roots: list[str] = []

    class DummyEnv:
        def __init__(self) -> None:
            self.step_count = 0

        def reset(self):
            return {"id": "root-0", "success": False}

        def step(self, action):
            self.step_count += 1
            obs = {"id": f"root-{self.step_count}", "success": self.step_count >= 2}
            done = False
            return obs, float(np.asarray(action).reshape(-1)[0]), done, {"success": obs["success"]}

    def policy_sampler(root_observation, *, root_id, num_candidates, segment_index, **_kwargs):
        assert root_id == root_observation["id"]
        sampled_roots.append(root_id)
        for candidate_index in range(num_candidates):
            yield {
                "candidate_index": candidate_index,
                "proc_root_snapshot": root_id,
                "unsquashed_chunk": np.array([[candidate_index]], dtype=np.float32),
                "old_logprob_steps": np.array([-0.1], dtype=np.float32),
                "exec_actions_for_env": np.array([[candidate_index]], dtype=np.float32),
            }

    def score_fn(root_observation, candidate, goal, *, root_id, segment_index, **_kwargs):
        del root_observation, goal
        scored_roots.append(root_id)
        preferred = 1 if segment_index == 0 else 0
        return _score(
            candidate_index=candidate["candidate_index"],
            progress=1.0 if candidate["candidate_index"] == preferred else 0.0,
            final_distance=float(candidate["candidate_index"]),
        )

    result = collect_phase12_episode(
        env=DummyEnv(),
        policy_sampler=policy_sampler,
        score_fn=score_fn,
        goals=["goal-0", "goal-1"],
        num_candidates=2,
    )

    assert sampled_roots == ["root-0", "root-1"]
    assert scored_roots == ["root-0", "root-0", "root-1", "root-1"]
    assert [segment.selected_candidate_index for segment in result.segments] == [1, 0]
    assert result.total_env_reward == pytest.approx(1.0)
    assert result.success_any is True
    assert result.success_last is True


def test_collect_phase12_episode_applies_action_profile_before_score_and_env() -> None:
    seen_wm: list[np.ndarray] = []
    seen_env: list[np.ndarray] = []

    class DummyEnv:
        action_space = type(
            "A",
            (),
            {
                "low": np.full((4,), -1.0, dtype=np.float32),
                "high": np.full((4,), 1.0, dtype=np.float32),
            },
        )()

        def reset(self):
            return {"id": "root"}

        def step(self, action):
            seen_env.append(np.asarray(action, dtype=np.float32).copy())
            return {"id": "next"}, 0.0, True, {}

    def sampler(*_args, **_kwargs):
        yield {
            "candidate_index": 0,
            "unsquashed_chunk": np.zeros((1, 4), dtype=np.float32),
            "old_logprob_steps": np.zeros(1, dtype=np.float32),
            "exec_actions_raw_postprocessed": np.array([[2.0, -2.0, 0.5, 1.5]], dtype=np.float32),
        }

    def score_fn(_root, candidate, _goal, **_kwargs):
        seen_wm.append(candidate.exec_actions_for_wm.copy())
        return _score(candidate_index=0, progress=1.0, final_distance=0.0)

    collect_phase12_episode(
        env=DummyEnv(),
        policy_sampler=sampler,
        score_fn=score_fn,
        goals=["goal"],
        num_candidates=1,
        action_profile="bounded_executed",
    )

    np.testing.assert_allclose(seen_wm[0], np.array([[1.0, -1.0, 0.5, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(seen_env[0], np.array([1.0, -1.0, 0.5, 1.0], dtype=np.float32))


def test_collect_phase12_episode_official_profile_scores_raw_postprocessed_actions() -> None:
    seen_wm: list[np.ndarray] = []
    seen_env: list[np.ndarray] = []

    class DummyEnv:
        action_space = type(
            "A",
            (),
            {
                "low": np.full((4,), -1.0, dtype=np.float32),
                "high": np.full((4,), 1.0, dtype=np.float32),
            },
        )()

        def reset(self):
            return {"id": "root"}

        def step(self, action):
            seen_env.append(np.asarray(action, dtype=np.float32).copy())
            return {"id": "next"}, 0.0, True, {}

    def sampler(*_args, **_kwargs):
        yield {
            "candidate_index": 0,
            "unsquashed_chunk": np.zeros((1, 4), dtype=np.float32),
            "old_logprob_steps": np.zeros(1, dtype=np.float32),
            "exec_actions_raw_postprocessed": np.array([[2.0, -2.0, 0.5, 1.5]], dtype=np.float32),
            "exec_actions_clipped": np.array([[1.0, -1.0, 0.5, 1.0]], dtype=np.float32),
        }

    def score_fn(_root, candidate, _goal, **_kwargs):
        seen_wm.append(candidate.exec_actions_for_wm.copy())
        return _score(candidate_index=0, progress=1.0, final_distance=0.0)

    collect_phase12_episode(
        env=DummyEnv(),
        policy_sampler=sampler,
        score_fn=score_fn,
        goals=["goal"],
        num_candidates=1,
        action_profile="official_jepa_mirror",
    )

    expected = np.array([[2.0, -2.0, 0.5, 1.5]], dtype=np.float32)
    np.testing.assert_allclose(seen_wm[0], expected)
    np.testing.assert_allclose(seen_env[0], expected.reshape(4))
