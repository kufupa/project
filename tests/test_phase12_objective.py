from __future__ import annotations

import pytest
import torch

from smolvla_grpo.phase12_objective import (
    combined_l2_distance,
    score_progress,
    split_structured_latent,
)


def test_combined_l2_distance_uses_visual_plus_alpha_proprio() -> None:
    pred = {
        "visual": torch.tensor([1.0, 3.0]),
        "proprio": torch.tensor([2.0, 6.0]),
    }
    goal = {
        "visual": torch.tensor([0.0, 1.0]),
        "proprio": torch.tensor([0.0, 2.0]),
    }

    dist = combined_l2_distance(pred, goal, proprio_alpha=0.1)

    assert dist.visual_distance == pytest.approx((1.0**2 + 2.0**2) / 2.0)
    assert dist.proprio_distance == pytest.approx((2.0**2 + 4.0**2) / 2.0)
    assert dist.combined_distance == pytest.approx(dist.visual_distance + 0.1 * dist.proprio_distance)


def test_score_progress_uses_start_minus_final_and_negative_final_return() -> None:
    goal = {"visual": torch.zeros(2), "proprio": torch.zeros(2)}
    start = {"visual": torch.ones(2), "proprio": torch.ones(2)}
    final = {"visual": torch.zeros(2), "proprio": torch.zeros(2)}

    score = score_progress(
        candidate_index=2,
        start=start,
        final=final,
        goal=goal,
        proprio_alpha=0.1,
    )

    assert score.candidate_index == 2
    assert score.wm_latent_progress == pytest.approx(score.start_combined_distance - score.final_combined_distance)
    assert score.wm_latent_progress > 0
    assert score.latent_return == pytest.approx(-score.final_combined_distance)
    assert score.wm_status == "ok"


def test_split_structured_latent_requires_proprio_by_default() -> None:
    with pytest.raises(KeyError, match="proprio"):
        split_structured_latent({"visual": torch.zeros(2)})


def test_split_structured_latent_visual_only_ablation_allows_missing_proprio() -> None:
    out = split_structured_latent(
        {"visual": torch.ones(2)},
        mode="visual_only_ablation",
    )

    assert set(out) == {"visual"}
    torch.testing.assert_close(out["visual"], torch.ones(2))

