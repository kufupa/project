from __future__ import annotations

from pathlib import Path


TRAINER = Path(__file__).resolve().parents[1] / "scripts" / "grpo" / "train_phase11_env_on_policy_grpo.py"


def test_trainer_exposes_rollout_unit_and_rollout_chunk_len() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert "--rollout-unit" in text
    assert 'choices=("step", "chunk")' in text
    assert "--rollout-chunk-len" in text


def test_flow_sde_step_mode_is_guarded() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert "flow_sde requires --rollout-unit chunk" in text


def test_chunk_mode_is_flow_sde_only_for_first_implementation() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert "chunk rollout currently requires --logprob-mode flow_sde" in text


def test_chunk_mode_loads_bundle_with_rollout_chunk_len() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert 'n_action_steps=(int(args.rollout_chunk_len) if args.rollout_unit == "chunk" else 1)' in text


def test_chunk_mode_uses_chunk_rollout_collector() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert "collect_chunk_rollout_group" in text
