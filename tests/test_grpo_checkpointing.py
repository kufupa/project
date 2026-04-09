"""Tests for smolvla_grpo.checkpointing."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from smolvla_grpo.checkpointing import load_grpo_checkpoint, save_grpo_checkpoint


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
