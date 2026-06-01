from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.grpo.phase46_autopilot import append_manifest, load_manifest


def test_manifest_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "jobs_manifest.jsonl"
        append_manifest(p, {"stage": "smoke", "job_id": "123", "poll_state": "PENDING"})
        rows = load_manifest(p)
        assert rows[0]["job_id"] == "123"
        assert rows[0]["stage"] == "smoke"


def test_load_manifest_empty() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "missing.jsonl"
        assert load_manifest(p) == []
