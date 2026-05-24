from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from smolvla_grpo.phase12_wm_reward import score_phase12_chunk_with_wm, score_phase12_chunks_with_wm


class FakeWM:
    device = torch.device("cpu")
    proprio_dim = 2
    planner_action_dim = 4

    class Preprocessor:
        action_mean = np.zeros(4, dtype=np.float32)
        action_std = np.ones(4, dtype=np.float32)

    preprocessor = Preprocessor()

    class Model:
        action_dim = 4

        def encode(self, obs):
            return {"visual": obs["visual"] * 0.0, "proprio": obs["proprio"]}

        def unroll(self, z, *, act_suffix, debug=False):
            del debug
            delta = act_suffix.float().sum().reshape(1, 1, 1)
            return {
                "visual": z["visual"] + delta,
                "proprio": z["proprio"] + delta,
            }

    model = Model()


def test_score_phase12_chunk_uses_structured_visual_plus_alpha_proprio() -> None:
    wm = FakeWM()
    score = score_phase12_chunk_with_wm(
        wm_bundle=wm,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        proprio=np.zeros(2, dtype=np.float32),
        chunk_actions=np.ones((1, 4), dtype=np.float32),
        goal={"visual": torch.zeros(1, 1, 3, 256, 256), "proprio": torch.zeros(1, 1, 2)},
        candidate_index=3,
        proprio_alpha=0.1,
        mode="visual_proprio",
    )

    assert score.candidate_index == 3
    assert score.wm_status == "ok"
    assert score.final_combined_distance == score.final_visual_distance + 0.1 * score.final_proprio_distance
    assert score.wm_latent_progress < 0.0


def test_score_phase12_chunk_reduces_time_major_unroll_context_between_steps() -> None:
    class TimeMajorTraceWM(FakeWM):
        class Model:
            action_dim = 4

            def encode(self, obs):
                del obs
                return {
                    "visual": torch.zeros(1, 1, 1, dtype=torch.float32),
                    "proprio": torch.zeros(1, 1, 1, dtype=torch.float32),
                }

            def unroll(self, z, *, act_suffix, debug=False):
                del debug
                assert z["visual"].shape[:2] == (1, 1)
                assert z["proprio"].shape[:2] == (1, 1)
                prev_visual = z["visual"][-1:]
                prev_proprio = z["proprio"][-1:]
                delta = act_suffix.float().sum().reshape(1, 1, 1)
                return {
                    "visual": torch.cat([prev_visual, prev_visual + delta], dim=0),
                    "proprio": torch.cat([prev_proprio, prev_proprio + delta], dim=0),
                }

        model = Model()

    score = score_phase12_chunk_with_wm(
        wm_bundle=TimeMajorTraceWM(),
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        proprio=np.zeros(2, dtype=np.float32),
        chunk_actions=np.ones((2, 4), dtype=np.float32),
        goal={"visual": torch.full((1, 1, 1), 8.0), "proprio": torch.full((1, 1, 1), 8.0)},
        candidate_index=0,
        proprio_alpha=0.1,
        mode="visual_proprio",
    )

    assert score.final_visual_distance == 0.0
    assert score.final_proprio_distance == 0.0


def test_score_phase12_chunk_packs_five_env_actions_into_one_wm_action() -> None:
    class PackedActionWM(FakeWM):
        planner_action_dim = 20

        class Preprocessor:
            action_mean = np.zeros(4, dtype=np.float32)
            action_std = np.ones(4, dtype=np.float32)

        preprocessor = Preprocessor()

        class Model:
            action_dim = 20

            def __init__(self) -> None:
                self.actions: list[torch.Tensor] = []

            def encode(self, obs):
                del obs
                return {
                    "visual": torch.zeros(1, 1, 1, dtype=torch.float32),
                    "proprio": torch.zeros(1, 1, 1, dtype=torch.float32),
                }

            def unroll(self, z, *, act_suffix, debug=False):
                del debug
                self.actions.append(act_suffix.detach().cpu())
                delta = act_suffix.float().sum().reshape(1, 1, 1)
                return {
                    "visual": z["visual"] + delta,
                    "proprio": z["proprio"] + delta,
                }

        model = Model()

    wm = PackedActionWM()
    actions = np.arange(20, dtype=np.float32).reshape(5, 4)
    score_phase12_chunk_with_wm(
        wm_bundle=wm,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        proprio=np.zeros(1, dtype=np.float32),
        chunk_actions=actions,
        goal={"visual": torch.zeros(1, 1, 1), "proprio": torch.zeros(1, 1, 1)},
        candidate_index=0,
        proprio_alpha=0.1,
        mode="visual_proprio",
    )

    assert len(wm.model.actions) == 1
    assert tuple(wm.model.actions[0].shape) == (1, 1, 20)
    np.testing.assert_allclose(wm.model.actions[0].numpy().reshape(20), actions.reshape(20))


def test_phase12_wm_reward_module_does_not_import_old_concat_scorer() -> None:
    source = Path("src/smolvla_grpo/phase12_wm_reward.py").read_text(encoding="utf-8")

    assert "score_chunk_by_goal_latent" not in source
    assert "wm_scoring_latent" not in source


def _score_numbers(score):
    return (
        score.candidate_index,
        score.start_combined_distance,
        score.final_combined_distance,
        score.wm_latent_progress,
        score.latent_return,
    )


class BatchableFakeWM(FakeWM):
    class Model:
        action_dim = 4

        def __init__(self) -> None:
            self.encode_calls = 0
            self.unroll_batch_sizes: list[int] = []

        def encode(self, obs):
            self.encode_calls += 1
            proprio = obs["proprio"].clone()
            visual = proprio.sum(dim=-1, keepdim=True)
            return {"visual": visual, "proprio": proprio}

        def unroll(self, z, *, act_suffix, debug=False):
            del debug
            self.unroll_batch_sizes.append(int(act_suffix.shape[1]))
            start_visual = z["visual"].permute(1, 0, *range(2, z["visual"].dim())).contiguous()
            start_proprio = z["proprio"].permute(1, 0, *range(2, z["proprio"].dim())).contiguous()
            delta = act_suffix.float().sum(dim=-1, keepdim=True)
            final_visual = start_visual + delta.cumsum(dim=0)
            final_proprio = start_proprio + delta.cumsum(dim=0)
            return {
                "visual": torch.cat([start_visual, final_visual], dim=0),
                "proprio": torch.cat([start_proprio, final_proprio], dim=0),
            }

    def __init__(self) -> None:
        self.model = self.Model()


def _batched_inputs():
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    proprio = np.array([0.25, -0.5], dtype=np.float32)
    goal = {
        "visual": torch.full((1, 1, 1), 3.0),
        "proprio": torch.full((1, 1, 2), 3.0),
    }
    actions = [
        np.array([[0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.0, 0.1]], dtype=np.float32),
        np.array([[1.0, -0.5, 0.25, 0.0], [0.2, 0.2, 0.2, 0.2]], dtype=np.float32),
        np.array([[-0.3, 0.4, -0.5, 0.6], [0.7, -0.8, 0.9, -1.0]], dtype=np.float32),
        np.array([[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]], dtype=np.float32),
    ]
    return image, proprio, goal, actions


def test_score_phase12_chunks_batched_matches_serial_scores() -> None:
    image, proprio, goal, actions = _batched_inputs()
    candidate_indices = [10, 11, 12, 13]

    serial = [
        score_phase12_chunk_with_wm(
            wm_bundle=BatchableFakeWM(),
            image=image,
            proprio=proprio,
            chunk_actions=chunk,
            goal=goal,
            candidate_index=candidate_index,
            proprio_alpha=0.1,
            mode="visual_proprio",
        )
        for candidate_index, chunk in zip(candidate_indices, actions, strict=True)
    ]

    wm = BatchableFakeWM()
    batched = score_phase12_chunks_with_wm(
        wm_bundle=wm,
        image=image,
        proprio=proprio,
        chunk_actions=actions,
        candidate_indices=candidate_indices,
        goal=goal,
        proprio_alpha=0.1,
        mode="visual_proprio",
        batch_size=4,
    )

    for got, want in zip(batched, serial, strict=True):
        assert _score_numbers(got) == pytest.approx(_score_numbers(want))
    assert wm.model.encode_calls == 1
    assert max(wm.model.unroll_batch_sizes) == 4


def test_score_phase12_chunks_batch_size_microbatch_equivalent_and_permutation_invariant() -> None:
    image, proprio, goal, actions = _batched_inputs()
    candidate_indices = [7, 3, 99, 42]
    expected_by_id = None

    for batch_size in [1, 2, 3, 99]:
        telemetry: dict[str, object] = {}
        scores = score_phase12_chunks_with_wm(
            wm_bundle=BatchableFakeWM(),
            image=image,
            proprio=proprio,
            chunk_actions=actions,
            candidate_indices=candidate_indices,
            goal=goal,
            proprio_alpha=0.1,
            mode="visual_proprio",
            batch_size=batch_size,
            telemetry=telemetry,
        )
        by_id = {score.candidate_index: _score_numbers(score) for score in scores}
        if expected_by_id is None:
            expected_by_id = by_id
        assert by_id == pytest.approx(expected_by_id)
        assert telemetry["wm_score_candidate_count"] == len(actions)
        assert telemetry["wm_score_batch_size"] == batch_size
        assert telemetry["wm_score_mode"] == "batched"

    order = [2, 0, 3, 1]
    permuted = score_phase12_chunks_with_wm(
        wm_bundle=BatchableFakeWM(),
        image=image,
        proprio=proprio,
        chunk_actions=[actions[i] for i in order],
        candidate_indices=[candidate_indices[i] for i in order],
        goal=goal,
        proprio_alpha=0.1,
        mode="visual_proprio",
        batch_size=2,
    )

    assert {score.candidate_index: _score_numbers(score) for score in permuted} == pytest.approx(expected_by_id)


def test_score_phase12_chunks_accumulates_telemetry_across_segments() -> None:
    image, proprio, goal, actions = _batched_inputs()
    telemetry: dict[str, object] = {}

    score_phase12_chunks_with_wm(
        wm_bundle=BatchableFakeWM(),
        image=image,
        proprio=proprio,
        chunk_actions=actions[:3],
        candidate_indices=[0, 1, 2],
        goal=goal,
        proprio_alpha=0.1,
        mode="visual_proprio",
        batch_size=2,
        telemetry=telemetry,
    )
    first_seconds = float(telemetry["wm_score_seconds"])
    score_phase12_chunks_with_wm(
        wm_bundle=BatchableFakeWM(),
        image=image,
        proprio=proprio,
        chunk_actions=actions[3:],
        candidate_indices=[3],
        goal=goal,
        proprio_alpha=0.1,
        mode="visual_proprio",
        batch_size=2,
        telemetry=telemetry,
    )

    assert telemetry["wm_score_mode"] == "batched"
    assert telemetry["wm_score_candidate_count"] == 4
    assert telemetry["wm_score_batch_count"] == 3
    assert telemetry["wm_score_batch_sizes"] == [2, 1, 1]
    assert float(telemetry["wm_score_seconds"]) >= first_seconds
