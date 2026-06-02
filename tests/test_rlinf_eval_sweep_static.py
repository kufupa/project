from __future__ import annotations

from pathlib import Path

RLINF_EVAL = (
    Path(__file__).resolve().parents[2]
    / "RLinf-smolvla-metaworld-ppo-grpo"
    / "scripts"
    / "eval_smolvla_metaworld_ckpt_sweep.py"
)


def _trainable_block() -> str:
    text = RLINF_EVAL.read_text(encoding="utf-8")
    start = text.index("if isinstance(trainable, dict):")
    end = text.index('policy_state = checkpoint.get("policy_state_dict")', start)
    return text[start:end]


def test_eval_auto_prefers_sibling_checkpoints_eval() -> None:
    text = RLINF_EVAL.read_text(encoding="utf-8")
    assert "def resolve_checkpoint_dir" in text
    assert 'checkpoint_dir.name == "checkpoints"' in text
    assert 'checkpoint_dir.parent / "checkpoints_eval"' in text
    assert 'checkpoint_dir / "checkpoints_eval"' in text
    assert "resolved_checkpoint_dir" in text


def test_trainable_delta_path_resets_policy_forward_state() -> None:
    block = _trainable_block()
    assert 'policy = getattr(model, "policy", model)' in block
    assert 'policy_reset = getattr(policy, "reset", None)' in block
    assert "policy_reset()" in block
