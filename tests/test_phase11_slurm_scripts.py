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
        "submit_phase111_vector_rollout_smoke.slurm",
        "submit_phase111_eval_sweep.slurm",
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


def test_submit_phase111_vector_rollout_smoke_trains_sync_and_async() -> None:
    path = _REPO_ROOT / "scripts" / "grpo" / "submit_phase111_vector_rollout_smoke.slurm"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --export=NIL" in text
    assert "scripts/slurm/common_env.sh" in text
    assert "--rollout-execution vector_sync" in text
    assert "--rollout-execution vector_async" in text
    assert "--async-start-method forkserver" in text
    assert "PHASE111_VECTOR_ROLLOUT_SMOKE_OK" in text
    assert 'm_async.get("async_start_method") == "forkserver"' in text
    subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))


def test_submit_phase111_single_task_grpo_is_official_backend() -> None:
    text = (_REPO_ROOT / "scripts" / "grpo" / "submit_phase111_single_task_grpo.slurm").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --export=NIL" in text
    assert "#SBATCH --time=48:00:00" in text
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
    assert "GRPO_PHASE111_ROLLOUT_EXECUTION" in text
    assert '$10 rollout_execution' in text
    assert '$11 async_start_method' in text
    assert '$12 action_transform' in text
    assert '$13 run_label' in text
    assert 'if [[ $# -ge 10' in text
    assert 'if [[ $# -ge 11' in text
    assert 'if [[ $# -ge 12' in text
    assert 'if [[ $# -ge 13' in text
    assert "GRPO_PHASE111_ACTION_TRANSFORM" in text
    assert "GRPO_PHASE111_RUN_LABEL" in text
    assert 'manifest["rollout_execution"]' in text
    assert 'manifest["action_transform"]' in text
    assert 'manifest["run_label"]' in text
    assert 'manifest["async_start_method"] == "forkserver"' in text
    assert 'len(rows) == num_updates' in text
    assert 'update_0100.pt' in text
    assert '"rollout_seconds" in row' in text
    assert '"optimize_seconds" in row' in text
    assert '"update_seconds" in row' in text
    assert '"num_env_steps" in row' in text
    assert '"action_clip_fraction" in row' in text
    assert '"action_clip_any_fraction" in row' in text


def test_phase111_eval_sweep_runs_eval_from_repo_root() -> None:
    text = (_REPO_ROOT / "scripts" / "grpo" / "eval_phase111_grpo_sweep.py").read_text(encoding="utf-8")
    assert "_REPO = Path(__file__).resolve().parents[2]" in text
    assert 'subprocess.run(cmd, check=True, cwd=str(_REPO))' in text
    assert "--sweep-name" in text
    assert 'sweep_dir = run_dir / sweep_name' in text


def test_submit_phase111_eval_sweep_uses_common_env_and_sweep_name() -> None:
    path = _REPO_ROOT / "scripts" / "grpo" / "submit_phase111_eval_sweep.slurm"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --export=NIL" in text
    assert "scripts/slurm/common_env.sh" in text
    assert 'RUN_DIR="${1:-}"' in text
    assert 'BASE_CKPT="${2:-}"' in text
    assert 'SWEEP_NAME="${8:-eval_sweep_current}"' in text
    assert "--sweep-name" in text
    assert "PHASE111_EVAL_SWEEP_OK" in text
    subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))
