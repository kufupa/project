from __future__ import annotations

import torch
import torch.nn as nn

from smolvla_grpo.eggroll_noise import (
    EggrollLayerSpec,
    EggrollNoiseManager,
    discover_eggroll_layers,
)


def test_noise_is_deterministic_and_member_specific() -> None:
    spec = EggrollLayerSpec(layer_id=3, name="x", out_features=5, in_features=7)
    manager = EggrollNoiseManager(base_seed=17, rank=2, antithetic=False)

    a1, b1, s1 = manager.generate_factors(
        spec,
        member_id=4,
        iteration=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    a2, b2, s2 = manager.generate_factors(
        spec,
        member_id=4,
        iteration=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    a3, b3, _s3 = manager.generate_factors(
        spec,
        member_id=5,
        iteration=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(a1, a2)
    assert torch.equal(b1, b2)
    assert s1 == s2 == 1.0
    assert not torch.equal(a1, a3)
    assert not torch.equal(b1, b3)


def test_antithetic_pair_reuses_factors_with_opposite_sign() -> None:
    spec = EggrollLayerSpec(layer_id=0, name="x", out_features=3, in_features=4)
    manager = EggrollNoiseManager(base_seed=17, rank=1, antithetic=True)

    a0, b0, s0 = manager.generate_factors(
        spec,
        member_id=8,
        iteration=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    a1, b1, s1 = manager.generate_factors(
        spec,
        member_id=9,
        iteration=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(a0, a1)
    assert torch.equal(b0, b1)
    assert s0 == 1.0
    assert s1 == -1.0


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vlm_with_expert = nn.Module()
        self.vlm_with_expert.lm_expert = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 6))
        self.state_proj = nn.Linear(4, 4)


class _FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeModel()
        self.other = nn.Linear(2, 2)


def test_discover_action_expert_scope_only() -> None:
    specs = discover_eggroll_layers(_FakePolicy(), train_scope="action_expert")
    names = [spec.name for spec in specs]

    assert names == [
        "model.vlm_with_expert.lm_expert.0",
        "model.vlm_with_expert.lm_expert.2",
    ]


def test_discover_action_head_scope() -> None:
    specs = discover_eggroll_layers(_FakePolicy(), train_scope="action_head")
    assert [spec.name for spec in specs] == ["model.state_proj"]
