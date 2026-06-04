from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_eggroll_trainer_artifact_contract_static() -> None:
    text = (PROJECT / "src" / "smolvla_grpo" / "eggroll_trainer.py").read_text(encoding="utf-8")

    assert "selected_action_rollout.mp4" in text
    assert "oracle_baseline.mp4" in text
    assert "timings.jsonl" in text
    assert "progress.jsonl" in text
    assert "train_manifest.json" in text
    assert "smoke_manifest.json" in text
    assert "assert_smoke_manifest_contract" in text
    for key in (
        "vla_load_seconds",
        "env_init_seconds",
        "reset_seconds",
        "proc_build_seconds",
        "forward_seconds",
        "postprocess_seconds",
        "env_step_seconds",
        "rollout_seconds",
        "es_update_seconds",
        "checkpoint_seconds",
        "video_seconds",
        "iteration_seconds",
        "cuda_max_memory_allocated_gb",
        "cuda_max_memory_reserved_gb",
    ):
        assert key in text
