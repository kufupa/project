"""Static checks for GRPO policy wrapper (PyTorch API compatibility)."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_POLICY_SRC = _REPO_ROOT / "src" / "smolvla_grpo" / "policy_wrapper.py"


def test_policy_wrapper_does_not_call_rsample_with_generator() -> None:
    src = _POLICY_SRC.read_text(encoding="utf-8")
    assert ".rsample(generator=" not in src
    assert "torch.randn(" in src or "torch.randn_like(" in src
    assert "generator=rng" in src


def test_policy_wrapper_checks_policy_and_model_for_euler_noise() -> None:
    src = _POLICY_SRC.read_text(encoding="utf-8")
    assert 'hasattr(self._policy, "euler_step_noise_std")' in src
    assert 'hasattr(self._policy.model, "euler_step_noise_std")' in src
