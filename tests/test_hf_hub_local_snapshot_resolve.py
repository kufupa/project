"""Tests for Hugging Face hub local snapshot resolution (cached load path)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from smolvla_pipeline.hf_hub_local_resolve import (
    resolve_hf_hub_repo_to_local_snapshot,
    should_resolve_hf_hub_to_local,
)


def test_resolve_returns_single_snapshot(tmp_path: Path):
    hub = tmp_path / "hub"
    snap = hub / "models--foo--bar" / "snapshots" / "deadbeefcafe"
    snap.mkdir(parents=True)
    (snap / "marker.txt").write_text("ok", encoding="utf-8")

    out = resolve_hf_hub_repo_to_local_snapshot("foo/bar", hub_cache=hub)
    assert out is not None
    assert out == str(snap.resolve())


def test_resolve_picks_newest_when_multiple_snapshots(tmp_path: Path):
    hub = tmp_path / "hub"
    base = hub / "models--org--model" / "snapshots"
    old = base / "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    new = base / "ffffffffffffffffffffffffffffffffffffffff"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "a.txt").write_text("a", encoding="utf-8")
    (new / "b.txt").write_text("b", encoding="utf-8")
    old_time = time.time() - 10_000
    os.utime(old, (old_time, old_time))
    os.utime(new, (time.time(), time.time()))

    out = resolve_hf_hub_repo_to_local_snapshot("org/model", hub_cache=hub)
    assert out is not None
    assert Path(out).name == new.name


def test_resolve_non_remote_id_passthrough():
    assert resolve_hf_hub_repo_to_local_snapshot("/abs/path/to/ckpt", hub_cache=None) == "/abs/path/to/ckpt"


def test_resolve_strict_raises_when_no_snapshot(tmp_path: Path):
    hub = tmp_path / "hub"
    with pytest.raises(RuntimeError, match="no local Hugging Face hub snapshot"):
        resolve_hf_hub_repo_to_local_snapshot("missing/repo", hub_cache=hub, strict=True)


def test_resolve_returns_remote_when_missing_and_not_strict(tmp_path: Path):
    hub = tmp_path / "hub"
    out = resolve_hf_hub_repo_to_local_snapshot("a/b", hub_cache=hub, strict=False)
    assert out == "a/b"


def test_should_resolve_hf_hub_to_local_respects_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SMOLVLA_LOCAL_FILES_ONLY", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("MT50_PHASE07_USE_LOCAL_HF_SNAPSHOT", raising=False)
    monkeypatch.delenv("SMOLVLA_PREFER_LOCAL_HF_SNAPSHOT", raising=False)
    assert should_resolve_hf_hub_to_local() is False

    monkeypatch.setenv("SMOLVLA_PREFER_LOCAL_HF_SNAPSHOT", "true")
    assert should_resolve_hf_hub_to_local() is True

    monkeypatch.delenv("SMOLVLA_PREFER_LOCAL_HF_SNAPSHOT", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert should_resolve_hf_hub_to_local() is True
