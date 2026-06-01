from __future__ import annotations

import os
from pathlib import Path


def _modeling_smolvla_path() -> Path:
    return Path(
        os.environ.get(
            "SMOLVLA_MODELING_SMOLVLA_PATH",
            "/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages/"
            "lerobot/policies/smolvla/modeling_smolvla.py",
        )
    )


def test_smolvla_venv_hook_exports_flow_sde_trace_contract() -> None:
    text = _modeling_smolvla_path().read_text(encoding="utf-8")

    assert "self.last_flow_sde_trace" in text
    assert "flow_sde_trace" in text
    assert "flow_sde_noise_level" in text
    assert "flow_sde_trace_step" in text
    assert "flow_sde_noise_seed" in text
    for key in ("tau_idx", "A_tau", "v_tau", "mu_tau", "sigma_tau", "A_next", "noise_seed"):
        assert f'"{key}"' in text


def test_gaussian_sampling_path_does_not_require_flow_sde_trace() -> None:
    text = _modeling_smolvla_path().read_text(encoding="utf-8")

    assert 'flow_sde_trace = bool(kwargs.get("flow_sde_trace", False))' in text
    assert "if flow_sde_trace:" in text


def test_smolvla_venv_hook_can_recompute_flow_sde_logprob() -> None:
    text = _modeling_smolvla_path().read_text(encoding="utf-8")

    assert "def flow_sde_logprob_from_trace(" in text
    assert "A_tau" in text
    assert "A_next" in text
    assert "self.denoise_step(" in text
