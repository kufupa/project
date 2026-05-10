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
        "submit_phase111_on_grpo_lerobot_smoke.slurm",
        "submit_phase111_single_task_grpo.slurm",
        "submit_phase11_chain.sh",
        "submit_api_gate_smoke.slurm",
        "submit_phase11_eval_sweep_seed1020.slurm",
        "submit_phase11_eval_sweep_seed1000_ge50.slurm",
    ):
        path = grpo / name
        subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))
    common = _REPO_ROOT / "scripts" / "slurm" / "common_env.sh"
    subprocess.run(["bash", "-n", str(common)], check=True, cwd=str(_REPO_ROOT))


def test_submit_phase111_on_grpo_lerobot_smoke_is_official_backend() -> None:
    text = (_REPO_ROOT / "scripts" / "grpo" / "submit_phase111_on_grpo_lerobot_smoke.slurm").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --export=NIL" in text
    assert "phase111_on_grpo_lerobot_smoke" in text
    assert "--env-backend official_lerobot" in text
    assert 'MAX_STEPS="${GRPO_PHASE111_MAX_STEPS:-0}"' in text
    assert '--max-steps "${MAX_STEPS}"' in text
    assert "PHASE111_GRPO_LEROBOT_ARTIFACTS_OK" in text
    assert "PHASE111_GRPO_LEROBOT_SMOKE_OK" in text


def test_submit_phase111_single_task_grpo_is_official_backend() -> None:
    text = (_REPO_ROOT / "scripts" / "grpo" / "submit_phase111_single_task_grpo.slurm").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --export=NIL" in text
    assert "scripts/slurm/common_env.sh" in text
    assert "--env-backend official_lerobot" in text
    assert 'TASK="${GRPO_PHASE111_TASK:-push-v3}"' in text
    assert 'NUM_UPDATES="${GRPO_PHASE111_NUM_UPDATES:-100}"' in text
    assert 'GROUP_SIZE="${GRPO_PHASE111_GROUP_SIZE:-4}"' in text
    assert 'SAVE_EVERY="${GRPO_PHASE111_SAVE_EVERY:-5}"' in text
    assert 'MAX_STEPS="${GRPO_PHASE111_MAX_STEPS:-120}"' in text
    assert "120-step cap to keep update cost comparable" in text
    assert '_PHASE111_PROJECT_FALLBACK="/vol/bitbucket/aa6622/project"' in text
    assert "PHASE111_SINGLE_TASK_GRPO_OK" in text


def test_phase111_eval_sweep_runs_eval_from_repo_root() -> None:
    text = (_REPO_ROOT / "scripts" / "grpo" / "eval_phase111_grpo_sweep.py").read_text(encoding="utf-8")
    assert "_REPO = Path(__file__).resolve().parents[2]" in text
    assert 'subprocess.run(cmd, check=True, cwd=str(_REPO))' in text
