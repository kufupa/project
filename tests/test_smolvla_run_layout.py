from datetime import datetime, timezone
from pathlib import Path

from src.smolvla_pipeline import run_layout
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


def test_build_run_dir_name_sanitizes_variant():
    run_name = build_run_dir_name(
        timestamp_utc="20260410T170000Z",
        episodes=1,
        task="push-v3",
        seed=1000,
        variant="../SMOL/VLA!!",
        nonce="123456",
    )
    assert run_name == "run_20260410T170000Z_ep1_vsmol_vla_tpush_v3_s1000_r123456"


def test_ensure_unique_run_dir_retries_collision_then_succeeds(tmp_path: Path, monkeypatch):
    fixed_now = datetime(2026, 4, 10, 17, 0, 0, tzinfo=timezone.utc)

    class FixedDatetime:
        @staticmethod
        def now(tz):
            return fixed_now

    monkeypatch.setattr(run_layout, "datetime", FixedDatetime)

    collided = tmp_path / "run_20260410T170000Z_ep1_vsmolvla_tpush_v3_s1000_r000001"
    collided.mkdir()

    values = iter([1, 2])
    monkeypatch.setattr(run_layout.secrets, "randbelow", lambda _max_value: next(values))

    created = ensure_unique_run_dir(tmp_path, episodes=1, task="push-v3", seed=1000, variant="smolvla")
    assert created.name == "run_20260410T170000Z_ep1_vsmolvla_tpush_v3_s1000_r000002"
    assert created.exists()


def test_ensure_unique_run_dir_raises_after_retry_exhaustion(tmp_path: Path, monkeypatch):
    fixed_now = datetime(2026, 4, 10, 17, 0, 0, tzinfo=timezone.utc)

    class FixedDatetime:
        @staticmethod
        def now(tz):
            return fixed_now

    monkeypatch.setattr(run_layout, "datetime", FixedDatetime)
    monkeypatch.setattr(run_layout.secrets, "randbelow", lambda _max_value: 1)

    (tmp_path / "run_20260410T170000Z_ep1_vsmolvla_tpush_v3_s1000_r000001").mkdir()

    try:
        ensure_unique_run_dir(tmp_path, episodes=1, task="push-v3", seed=1000, variant="smolvla")
        assert False, "Expected RuntimeError after exhausting retries."
    except RuntimeError as err:
        assert "unique run directory" in str(err).lower()
