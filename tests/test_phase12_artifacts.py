from __future__ import annotations

from pathlib import Path

import pytest

from smolvla_grpo.phase12_logging import assert_smoke_manifest_contract


def test_smoke_manifest_rejects_nonexistent_files(tmp_path: Path) -> None:
    manifest = {
        "rollout_validation_video": str(tmp_path / "missing.mp4"),
        "selected_action_rollout_video": str(tmp_path / "missing_selected.mp4"),
        "oracle_baseline_video": str(tmp_path / "oracle_baseline.mp4"),
        "oracle_baseline_video_status": "ok",
        "success_any": False,
        "success_last": False,
    }

    with pytest.raises(AssertionError, match="does not exist"):
        assert_smoke_manifest_contract(manifest, base_dir=tmp_path)


def test_smoke_manifest_accepts_existing_nonempty_files(tmp_path: Path) -> None:
    selected = tmp_path / "selected_action_rollout.mp4"
    oracle = tmp_path / "oracle_baseline.mp4"
    strip = tmp_path / "wm_decode_selected_strip.png"
    real_vs_pred = tmp_path / "wm_real_vs_pred_selected_strip.png"
    selected.write_bytes(b"selected")
    oracle.write_bytes(b"oracle")
    strip.write_bytes(b"decode")
    real_vs_pred.write_bytes(b"real-vs-pred")
    manifest = {
        "rollout_validation_video": str(selected),
        "selected_action_rollout_video": str(selected),
        "oracle_baseline_video": str(oracle),
        "oracle_baseline_video_status": "ok",
        "wm_decode_status": "ok",
        "wm_decode_selected_strip_path": str(strip),
        "wm_real_vs_pred_selected_strip_path": str(real_vs_pred),
        "success_any": False,
        "success_last": False,
    }

    assert_smoke_manifest_contract(manifest, base_dir=tmp_path)

