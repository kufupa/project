from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from smolvla_grpo.phase12_wm_only_rollout import collect_phase12_wm_only_episode


def _score(candidate_index: int, progress: float) -> dict:
    return {
        "candidate_index": int(candidate_index),
        "wm_latent_progress": float(progress),
        "latent_return": float(progress),
        "final_combined_distance": float(10.0 - progress),
    }


def test_wm_only_collector_scores_candidates_without_env_step() -> None:
    step_calls = 0

    class RootSource:
        def reset(self, seed: int):
            assert seed == 123
            return {
                "id": "root-123",
                "image": np.zeros((8, 8, 3), dtype=np.uint8),
                "proprio": np.zeros(4, dtype=np.float32),
                "proc": {"x": torch.zeros(1, 1)},
            }

        def step(self, _action):
            nonlocal step_calls
            step_calls += 1
            raise AssertionError("wm_only collector must not step env")

    def sampler(root, *, num_candidates: int, segment_index: int, goal):
        del goal
        assert root["id"] == "root-123"
        assert segment_index == 0
        for i in range(num_candidates):
            yield {
                "candidate_index": i,
                "proc_root_snapshot": root["proc"],
                "unsquashed_chunk": torch.full((2, 4), float(i), dtype=torch.float32),
                "old_logprob_steps": np.array([-0.1, -0.2], dtype=np.float32),
                "exec_actions_raw_postprocessed": np.full((2, 4), float(i), dtype=np.float32),
            }

    def score_fn(root, candidate, goal, *, segment_index: int):
        assert root["id"] == "root-123"
        assert goal.frame_index_1based == 25
        assert segment_index == 0
        return _score(candidate.candidate_index, progress=float(candidate.candidate_index))

    result = collect_phase12_wm_only_episode(
        root_source=RootSource(),
        reset_seed=123,
        policy_sampler=sampler,
        score_fn=score_fn,
        goals=[SimpleNamespace(frame_index_1based=25)],
        group_size=3,
        reward_key="wm_latent_progress",
    )

    assert step_calls == 0
    assert result.success_any is False
    assert result.metadata["candidate_rewards"] == [0.0, 1.0, 2.0]
    assert result.metadata["selected_candidate_indices"] == [2]
    assert len(result.metadata["old_logprob_sums"]) == 3
    assert len(result.metadata["unsquashed_chunks"]) == 3
    assert result.segments[0].goal_frame_index_1based == 25
