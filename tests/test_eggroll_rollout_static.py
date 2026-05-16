from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_eggroll_rollout_is_queue_free_static() -> None:
    text = (PROJECT / "src" / "smolvla_grpo" / "eggroll_rollout.py").read_text(encoding="utf-8")

    assert "torch.inference_mode()" in text
    assert ".backward(" not in text
    assert "policy.model.sample_actions" in text
    assert "actions =" in text
    assert "mean, log_std" not in text
    assert "population_batch_size" in text
    assert "member_ids=" in text
    assert "flow_dtype" in text
    assert "rollout_seed_offset" in text
    assert "_frame_from_vector_obs" in text
    assert "render_frame()" not in text
    assert "select_action(" not in text
    assert "select_action_distr_params(" not in text
