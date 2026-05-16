from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from smolvla_grpo.eggroll_linear import eggroll_linear_context, install_eggroll_linear_patch
from smolvla_grpo.eggroll_noise import EggrollLayerSpec, EggrollNoiseManager


def _explicit_delta(
    x: torch.Tensor,
    layer: nn.Linear,
    spec: EggrollLayerSpec,
    manager: EggrollNoiseManager,
    *,
    member_id: int,
    iteration: int,
    sigma: float,
) -> torch.Tensor:
    a, b, sign = manager.generate_factors(
        spec,
        member_id=member_id,
        iteration=iteration,
        device=x.device,
        dtype=x.dtype,
    )
    base = F.linear(x, layer.weight, layer.bias)
    delta = (x.reshape(-1, spec.in_features) @ b) @ a.T
    return base + sigma * sign * delta.reshape_as(base) / (manager.rank**0.5)


def test_patch_no_context_matches_original_and_restores() -> None:
    layer = nn.Linear(4, 3)
    spec = EggrollLayerSpec(0, "layer", 3, 4)
    x = torch.randn(2, 4)
    expected = layer(x)

    handle = install_eggroll_linear_patch({0: layer}, [spec])
    try:
        assert isinstance(layer, nn.Linear)
        assert torch.allclose(layer(x), expected)
    finally:
        handle.remove()
    assert torch.allclose(layer(x), expected)


def test_scalar_context_matches_explicit_formula() -> None:
    layer = nn.Linear(4, 3)
    spec = EggrollLayerSpec(0, "layer", 3, 4)
    manager = EggrollNoiseManager(base_seed=5, rank=2, antithetic=True)
    handle = install_eggroll_linear_patch({0: layer}, [spec])
    x = torch.randn(2, 4)

    try:
        ctx = handle.context(noise_manager=manager, iteration=7, sigma=0.01, member_id=3)
        with eggroll_linear_context(ctx):
            got = layer(x)
        expected = _explicit_delta(x, layer, spec, manager, member_id=3, iteration=7, sigma=0.01)
        assert torch.allclose(got, expected, atol=1e-6)
    finally:
        handle.remove()


def test_batched_context_matches_per_row_formula() -> None:
    layer = nn.Linear(4, 3)
    spec = EggrollLayerSpec(0, "layer", 3, 4)
    manager = EggrollNoiseManager(base_seed=5, rank=2, antithetic=True)
    handle = install_eggroll_linear_patch({0: layer}, [spec])
    x = torch.randn(2, 5, 4)
    member_ids = torch.tensor([0, 1])

    try:
        ctx = handle.context(noise_manager=manager, iteration=2, sigma=0.02, member_ids=member_ids)
        with eggroll_linear_context(ctx):
            got = layer(x)
        expected = torch.stack(
            [
                _explicit_delta(
                    x[i],
                    layer,
                    spec,
                    manager,
                    member_id=int(member_ids[i]),
                    iteration=2,
                    sigma=0.02,
                )
                for i in range(2)
            ],
            dim=0,
        )
        assert torch.allclose(got, expected, atol=1e-6)
    finally:
        handle.remove()


def test_batched_flattened_rows_expand_member_ids() -> None:
    layer = nn.Linear(4, 3)
    spec = EggrollLayerSpec(0, "layer", 3, 4)
    manager = EggrollNoiseManager(base_seed=5, rank=2, antithetic=True)
    handle = install_eggroll_linear_patch({0: layer}, [spec])
    x = torch.randn(10, 4)
    member_ids = torch.tensor([0, 1])

    try:
        ctx = handle.context(noise_manager=manager, iteration=2, sigma=0.02, member_ids=member_ids)
        with eggroll_linear_context(ctx):
            got = layer(x)
        expected_rows = []
        for row_idx in range(10):
            member_id = 0 if row_idx < 5 else 1
            expected_rows.append(
                _explicit_delta(
                    x[row_idx : row_idx + 1],
                    layer,
                    spec,
                    manager,
                    member_id=member_id,
                    iteration=2,
                    sigma=0.02,
                ).reshape(-1)
            )
        assert torch.allclose(got, torch.stack(expected_rows, dim=0), atol=1e-6)
    finally:
        handle.remove()


def test_member_ids_batch_mismatch_raises() -> None:
    layer = nn.Linear(4, 3)
    spec = EggrollLayerSpec(0, "layer", 3, 4)
    manager = EggrollNoiseManager(base_seed=5, rank=1)
    handle = install_eggroll_linear_patch({0: layer}, [spec])
    x = torch.randn(2, 4)
    try:
        ctx = handle.context(
            noise_manager=manager,
            iteration=0,
            sigma=0.01,
            member_ids=torch.tensor([0, 1, 2]),
        )
        with eggroll_linear_context(ctx), pytest.raises(RuntimeError, match="member_ids batch mismatch"):
            layer(x)
    finally:
        handle.remove()
