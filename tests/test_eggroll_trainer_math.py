from __future__ import annotations

import torch
import torch.nn as nn

from smolvla_grpo.eggroll_noise import EggrollLayerSpec, EggrollNoiseManager
from smolvla_grpo.eggroll_trainer import (
    EggrollTrainerConfig,
    apply_es_update,
    compute_baseline,
    compute_fitness_stats,
    shape_fitness,
)


def test_eggroll_trainer_config_defaults_action_chunk_size_to_five(tmp_path) -> None:
    cfg = EggrollTrainerConfig(checkpoint="checkpoint", output_dir=tmp_path)

    assert cfg.action_chunk_size == 5


def test_baseline_and_centered_fitness() -> None:
    values = [1.0, 2.0, 5.0]
    assert compute_baseline(values, "mean") == 8.0 / 3.0
    assert compute_baseline(values, "median") == 2.0
    assert compute_baseline(values, "none") == 0.0
    centered = shape_fitness(values, baseline_type="mean", fitness_shaping="centered")
    assert abs(sum(centered)) < 1e-9


def test_rank_fitness_is_centered() -> None:
    shaped = shape_fitness([10.0, 30.0, 20.0], baseline_type="mean", fitness_shaping="rank")
    assert abs(sum(shaped)) < 1e-9
    assert shaped[1] > shaped[2] > shaped[0]


def test_rank_fitness_all_ties_returns_zero() -> None:
    assert shape_fitness([0.0, 0.0, 0.0], baseline_type="mean", fitness_shaping="rank") == [
        0.0,
        0.0,
        0.0,
    ]


def test_compute_fitness_stats_rejects_generator_success_bug() -> None:
    stats = compute_fitness_stats([1.0, 2.0], [True, False])
    assert stats["success_rate"] == 0.5


def test_apply_es_update_matches_explicit_sum() -> None:
    layer = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    spec = EggrollLayerSpec(0, "layer", 2, 3)
    manager = EggrollNoiseManager(base_seed=11, rank=2, antithetic=True)
    shaped = [1.0, -1.0, 0.5, -0.5]
    before = layer.weight.detach().clone()
    expected = torch.zeros_like(layer.weight)
    for member_id, fitness in enumerate(shaped):
        a, b, sign = manager.generate_factors(
            spec,
            member_id=member_id,
            iteration=0,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        expected += float(fitness) * float(sign) * (a @ b.T)
    expected *= 0.1 / len(shaped) / (manager.rank**0.5)

    stats = apply_es_update(
        modules={0: layer},
        specs=[spec],
        noise_manager=manager,
        shaped_fitness=shaped,
        iteration=0,
        alpha=0.1,
    )

    assert torch.allclose(layer.weight, before + expected)
    assert stats["relative_update_norm"] > 0.0
    assert stats["max_abs_update"] > 0.0


def test_apply_es_update_clips_relative_norm() -> None:
    layer = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    spec = EggrollLayerSpec(0, "layer", 2, 3)
    manager = EggrollNoiseManager(base_seed=11, rank=2, antithetic=True)

    stats = apply_es_update(
        modules={0: layer},
        specs=[spec],
        noise_manager=manager,
        shaped_fitness=[100.0, -100.0],
        iteration=0,
        alpha=1.0,
        max_relative_update_norm=0.01,
    )

    assert stats["relative_update_norm"] <= 0.010001
    assert stats["unclipped_relative_update_norm"] > stats["relative_update_norm"]
    assert stats["update_clipped_layer_count"] == 1
