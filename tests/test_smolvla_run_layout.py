from pathlib import Path

from src.smolvla_pipeline.run_layout import build_run_dir_name, ensure_unique_run_dir


def test_build_run_dir_name_contract():
    run_name = build_run_dir_name(
        timestamp_utc="20260410T170000Z",
        episodes=1,
        task="push-v3",
        seed=1000,
        variant="smolvla",
        nonce="123456",
    )
    assert run_name == "run_20260410T170000Z_ep1_vsmolvla_tpush_v3_s1000_r123456"


def test_ensure_unique_run_dir_never_reuses_existing(tmp_path: Path):
    first = ensure_unique_run_dir(tmp_path, episodes=1, task="push-v3", seed=1000, variant="smolvla")
    second = ensure_unique_run_dir(tmp_path, episodes=1, task="push-v3", seed=1000, variant="smolvla")
    assert first != second
    assert first.exists()
    assert second.exists()
