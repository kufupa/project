from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_eggroll_cli_exposes_seed_mode_and_checkpoint_sync_dir() -> None:
    text = (PROJECT / "scripts" / "eggroll" / "train_smolvla_eggroll.py").read_text(encoding="utf-8")

    assert "--seed-mode" in text
    assert "shared_per_iteration" in text
    assert "--checkpoint-sync-dir" in text


def test_eggroll_trainer_wires_seed_mode_and_checkpoint_sync_dir() -> None:
    text = (PROJECT / "src" / "smolvla_grpo" / "eggroll_trainer.py").read_text(encoding="utf-8")

    assert 'seed_mode: str = "member_offset"' in text
    assert "checkpoint_sync_dir: Path | None = None" in text
    assert "seed_mode=str(cfg.seed_mode)" in text
    assert "checkpoint_sync_dir" in text
    assert "reset_seed_mode" in text
    assert "sync_checkpoint_artifacts" in text
