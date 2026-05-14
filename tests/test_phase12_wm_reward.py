from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from smolvla_grpo.phase12_wm_reward import score_phase12_chunk_with_wm


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


def test_phase12_wm_reward_module_does_not_import_old_concat_scorer() -> None:
    source = Path("src/smolvla_grpo/phase12_wm_reward.py").read_text(encoding="utf-8")

    assert "score_chunk_by_goal_latent" not in source
    assert "wm_scoring_latent" not in source
