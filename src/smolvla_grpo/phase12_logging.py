"""Phase12 manifest/progress/artifact helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row) + "\n")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_manifest_path(value: str, base_dir: Path | None) -> Path:
    path = Path(value)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path


def _assert_nonempty_file(path: Path, key: str) -> None:
    assert path.exists(), f"smoke manifest {key} does not exist: {path}"
    assert path.is_file(), f"smoke manifest {key} is not a file: {path}"
    assert path.stat().st_size > 0, f"smoke manifest {key} is empty: {path}"


def assert_smoke_manifest_contract(manifest: dict[str, Any], *, base_dir: Path | None = None) -> None:
    required = (
        "rollout_validation_video",
        "selected_action_rollout_video",
        "oracle_baseline_video",
        "oracle_baseline_video_status",
        "success_any",
        "success_last",
    )
    for key in required:
        assert key in manifest, f"smoke manifest missing {key}"
    for key in (
        "rollout_validation_video",
        "selected_action_rollout_video",
        "oracle_baseline_video",
    ):
        value = manifest[key]
        assert isinstance(value, str) and value.strip(), f"smoke manifest {key} must be a non-empty path"
        _assert_nonempty_file(_resolve_manifest_path(value, base_dir), key)
    assert manifest["oracle_baseline_video_status"] == "ok"
    if manifest.get("wm_decode_status") == "ok":
        for key in ("wm_decode_selected_strip_path", "wm_real_vs_pred_selected_strip_path"):
            assert key in manifest, f"smoke manifest missing {key}"
            value = manifest[key]
            assert isinstance(value, str) and value.strip(), f"smoke manifest {key} must be a non-empty path"
            _assert_nonempty_file(_resolve_manifest_path(value, base_dir), key)
    assert isinstance(manifest["success_any"], bool), "success_any must be boolean"
    assert isinstance(manifest["success_last"], bool), "success_last must be boolean"

