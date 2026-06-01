"""Tests for smolvla_grpo.checkpointing."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from smolvla_grpo.checkpointing import (
    load_grpo_checkpoint,
    save_grpo_checkpoint,
    torch_load_mmap_default,
    torch_load_mmap_with_mode,
)


def test_save_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "update_0001.pt"
    policy_state = {"w": torch.tensor([1.0, 2.0])}
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    save_grpo_checkpoint(
        path,
        policy_state=policy_state,
        optimizer_state=opt.state_dict(),
        update_index=1,
        args={"lr": 0.1},
        extra={"note": "test"},
    )
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["update_index"] == 1

    loaded = load_grpo_checkpoint(path, map_location="cpu")
    assert loaded["update_index"] == 1
    assert "policy_state_dict" in loaded
    assert loaded["policy_state_dict"]["w"].tolist() == [1.0, 2.0]
    assert loaded["args"]["lr"] == 0.1
    assert loaded["extra"]["note"] == "test"
    assert "optimizer_state_dict" in loaded


def test_torch_load_mmap_default_requests_mmap(tmp_path, monkeypatch) -> None:
    path = tmp_path / "ckpt.pt"
    expected = {"update_index": 3}
    torch.save(expected, path)
    seen: dict[str, bool] = {}

    def fake_load(*args, **kwargs):
        seen["mmap"] = kwargs.get("mmap") is True
        return expected

    monkeypatch.setattr(torch, "load", fake_load)
    loaded = torch_load_mmap_default(path, map_location="cpu", weights_only=False)
    assert loaded == expected
    assert seen["mmap"] is True


def test_torch_load_mmap_with_mode_falls_back_when_mmap_unsupported(tmp_path, monkeypatch) -> None:
    path = tmp_path / "ckpt.pt"
    expected = {"update_index": 4}
    torch.save(expected, path)
    calls: list[dict[str, object]] = []

    def fake_load(*args, **kwargs):
        calls.append(dict(kwargs))
        if kwargs.get("mmap") is True:
            raise TypeError("mmap not supported")
        return expected

    monkeypatch.setattr(torch, "load", fake_load)
    loaded, mode = torch_load_mmap_with_mode(path, map_location="cpu", weights_only=False)
    assert loaded == expected
    assert mode == "eager"
    assert calls[0]["mmap"] is True
    assert "mmap" not in calls[1]
