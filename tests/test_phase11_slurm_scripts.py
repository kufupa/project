"""Lightweight checks on Phase11 Slurm / shell entrypoints."""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_submit_phase11_grpo_uses_explicit_python_and_rollout_smoke() -> None:
    text = (_REPO_ROOT / "scripts" / "grpo" / "submit_phase11_grpo.slurm").read_text(encoding="utf-8")
    assert "GRPO_PYTHON=" in text
    assert 'exec "${GRPO_PYTHON}"' in text
    assert "rollout-smoke" in text
    assert "smoke_phase11_rollout.py" in text
    assert "_PHASE11_PROJECT_FALLBACK" in text
    assert "scripts/slurm/common_env.sh" in text


def test_submit_phase11_chain_uses_chdir() -> None:
    text = (_REPO_ROOT / "scripts" / "grpo" / "submit_phase11_chain.sh").read_text(encoding="utf-8")
    assert "--chdir=" in text
    assert "PROJECT_CHDIR" in text


def test_slurm_and_chain_scripts_bash_syntax() -> None:
    grpo = _REPO_ROOT / "scripts" / "grpo"
    for name in (
        "submit_phase11_grpo.slurm",
        "submit_phase11_chain.sh",
        "submit_api_gate_smoke.slurm",
        "submit_phase11_eval_sweep_seed1020.slurm",
        "submit_phase11_eval_sweep_seed1000_ge50.slurm",
    ):
        path = grpo / name
        subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))
    common = _REPO_ROOT / "scripts" / "slurm" / "common_env.sh"
    subprocess.run(["bash", "-n", str(common)], check=True, cwd=str(_REPO_ROOT))
