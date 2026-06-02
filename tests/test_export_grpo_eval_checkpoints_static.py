from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "grpo" / "export_grpo_eval_checkpoints.py"


def test_export_grpo_eval_checkpoints_uses_base_policy_trainable_names() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "from smolvla_grpo.phase11_rollout import load_bundle_for_grpo" in text
    assert "train_phase11_env_on_policy_grpo import load_bundle_for_grpo" not in text
    assert "freeze_all_but_grpo_trainables" in text
    assert "policy.named_parameters()" in text
    assert "policy_state_dict" in text
    assert "policy.{name}" in text
    assert "checkpoint_type" in text
    assert "trainable_delta" in text


def test_export_grpo_eval_checkpoints_defaults_to_sibling_eval_dir() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'checkpoint_dir.parent / "checkpoints_eval"' in text
    assert "update_*.pt" in text
