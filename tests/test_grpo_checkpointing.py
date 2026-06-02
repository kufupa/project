"""Tests for smolvla_grpo.checkpointing."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from smolvla_grpo.checkpointing import (
    build_rlinf_eval_trainable_model,
    load_grpo_checkpoint,
    save_grpo_checkpoint,
    save_rlinf_eval_checkpoint,
    torch_load_mmap_default,
    torch_load_mmap_with_mode,
    validate_rlinf_eval_checkpoint,
)


class _ToyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(2, 2)
        self.model = nn.Module()
        self.model.log_std = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.model.vlm_with_expert = nn.Module()
        self.model.vlm_with_expert.lm_expert = nn.Linear(2, 1)
        for param in self.frozen.parameters():
            param.requires_grad = False


def test_build_rlinf_eval_trainable_model_prefixes_policy_keys() -> None:
    policy = _ToyPolicy()
    payload = build_rlinf_eval_trainable_model(policy)

    assert "policy.model.log_std" in payload
    assert "policy.model.vlm_with_expert.lm_expert.weight" in payload
    assert "policy.model.vlm_with_expert.lm_expert.bias" in payload
    assert all(name.startswith("policy.") for name in payload)
    assert not any("frozen" in name for name in payload)
    assert payload["policy.model.log_std"].device.type == "cpu"
    assert payload["policy.model.log_std"].tolist() == [1.0, 2.0]


def test_save_rlinf_eval_checkpoint_is_slim_and_one_based_update(tmp_path) -> None:
    policy = _ToyPolicy()
    path = tmp_path / "checkpoints_eval" / "update_0016.pt"

    save_rlinf_eval_checkpoint(
        path,
        policy=policy,
        update_index=15,
        metrics={"success_rate": 0.28},
        source_checkpoint="checkpoints/update_0016.pt",
    )

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["checkpoint_type"] == "trainable_delta"
    assert loaded["update"] == 16
    assert loaded["source_update_index"] == 15
    assert loaded["metrics"] == {"success_rate": 0.28}
    assert loaded["source_checkpoint"] == "checkpoints/update_0016.pt"
    assert "trainable_model" in loaded
    assert "policy_state_dict" not in loaded
    assert "optimizer_state_dict" not in loaded
    assert "policy.model.log_std" in loaded["trainable_model"]


def test_validate_rlinf_eval_checkpoint_rejects_missing_delta(tmp_path) -> None:
    path = tmp_path / "checkpoints_eval" / "update_0010.pt"
    path.parent.mkdir(parents=True)
    torch.save({"checkpoint_type": "trainable_delta", "update": 10}, path)

    with pytest.raises(ValueError, match="trainable_model"):
        validate_rlinf_eval_checkpoint(path, expected_update=10)


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
