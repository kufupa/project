from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (PROJECT / "scripts" / "grpo" / name).read_text(encoding="utf-8")


def _body_without_sbatch_header(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith("#SBATCH"))


def test_train_slurm_contracts() -> None:
    text = _read("submit_phase12_wm_chunk_grpo_train.slurm")

    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --export=NIL" in text
    assert 'slurm_resolve_project_root "scripts/grpo/train_phase12_wm_chunk_grpo.py"' in text
    assert "slurm_export_pythonpath" in text
    assert 'slurm_export_hf_torch_cache "phase12-wm-chunk-grpo-train"' in text
    assert 'export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"' in text
    assert 'export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"' in text
    assert 'export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"' in text
    assert 'export JEPA_WM_DISABLE_IMAGE_HEAD="${JEPA_WM_DISABLE_IMAGE_HEAD:-0}"' in text
    assert "--mode wm_grpo_train" in text
    assert 'ACTION_PROFILE="${1:-${PHASE12_ACTION_PROFILE:-official_jepa_mirror}}"' in text
    assert 'UPDATES="${2:-${PHASE12_NUM_UPDATES:-100}}"' in text
    assert 'OUT="${3:-${PHASE12_OUT:-${PROJECT_ROOT}/artifacts/phase12_wm_chunk_grpo_train/push-v3/g${GROUP_SIZE}_u${UPDATES}_seed${SEED_BASE}_${ACTION_PROFILE}}}"' in text
    assert 'RESUME="${4:-${PHASE12_RESUME:-}}"' in text
    assert 'NUM_EPISODES="${PHASE12_NUM_EPISODES:-${UPDATES}}"' in text
    assert 'EXTRA=()' in text
    assert 'EXTRA+=(--resume "${RESUME}")' in text
    assert '--num-updates "${UPDATES}"' in text
    assert '--num-episodes "${NUM_EPISODES}"' in text
    assert '"${EXTRA[@]}"' in text
    assert '--max-steps "${MAX_STEPS}"' in text
    assert '--save-every "${SAVE_EVERY}"' in text
    assert '--lr "${LR}"' in text
    assert '--init-log-std "${INIT_LOG_STD}"' in text
    assert '--euler-step-noise-std "${EULER_NOISE}"' in text
    assert "PHASE12_WM_CHUNK_GRPO_TRAIN_DONE" in text
    assert "PHASE12_WM_CHUNK_ROLLOUT_VALIDATION_DONE" not in text
    assert "submit_phase12_wm_chunk_rollout_validation_10ep" not in text
    assert "rg -q" not in text
    assert 'progress.jsonl missing update_complete event' in text
    assert 'progress.jsonl missing optimizer_step=true' in text
    assert 'terminal_checkpoint = checkpoints / f"update_{last_update + 1:04d}.pt"' in text
    assert 'expected final reset_seed' in text
    assert "sbatch" not in _body_without_sbatch_header(text)

