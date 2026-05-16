"""Deterministic low-rank EGGROLL noise for SmolVLA."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EggrollLayerSpec:
    layer_id: int
    name: str
    out_features: int
    in_features: int


class EggrollNoiseManager:
    """Regenerate per-layer low-rank factors from stable folded seeds."""

    def __init__(self, *, base_seed: int, rank: int, antithetic: bool = True) -> None:
        if int(rank) < 1:
            raise ValueError("rank must be >= 1")
        self.base_seed = int(base_seed)
        self.rank = int(rank)
        self.antithetic = bool(antithetic)

    @property
    def perturbation_scale(self) -> float:
        return 1.0 / math.sqrt(float(self.rank))

    def member_base_id_and_sign(self, member_id: int) -> tuple[int, float]:
        member = int(member_id)
        if member < 0:
            raise ValueError("member_id must be >= 0")
        if not self.antithetic:
            return member, 1.0
        return member // 2, 1.0 if member % 2 == 0 else -1.0

    def fold_seed(self, *, layer_id: int, member_id: int, iteration: int, stream: int) -> int:
        base_member, _sign = self.member_base_id_and_sign(int(member_id))
        payload = (
            f"eggroll|{self.base_seed}|{int(layer_id)}|{base_member}|"
            f"{int(iteration)}|{int(stream)}|{self.rank}"
        ).encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF

    def _randn(
        self,
        shape: tuple[int, ...],
        *,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        return torch.randn(shape, generator=gen, device=device, dtype=dtype)

    def generate_factors(
        self,
        spec: EggrollLayerSpec,
        *,
        member_id: int,
        iteration: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Return `A`, `B`, `sign` where perturbation is `sign * A @ B.T`."""

        _base_member, sign = self.member_base_id_and_sign(int(member_id))
        a_seed = self.fold_seed(
            layer_id=spec.layer_id,
            member_id=int(member_id),
            iteration=int(iteration),
            stream=0,
        )
        b_seed = self.fold_seed(
            layer_id=spec.layer_id,
            member_id=int(member_id),
            iteration=int(iteration),
            stream=1,
        )
        a = self._randn(
            (int(spec.out_features), self.rank),
            seed=a_seed,
            device=device,
            dtype=dtype,
        )
        b = self._randn(
            (int(spec.in_features), self.rank),
            seed=b_seed,
            device=device,
            dtype=dtype,
        )
        return a, b, float(sign)


def _iter_named_linears(root: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    for name, module in root.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def _scope_prefixes(scope: str) -> tuple[str, ...]:
    if scope == "action_expert":
        return ("model.vlm_with_expert.lm_expert", "vlm_with_expert.lm_expert")
    if scope == "action_head":
        return (
            "model.state_proj",
            "model.action_in_proj",
            "model.action_out_proj",
            "model.action_time_mlp_in",
            "model.action_time_mlp_out",
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        )
    if scope == "all_linear":
        return ("",)
    raise ValueError("train_scope must be 'action_expert', 'action_head', or 'all_linear'")


def discover_eggroll_layers(policy: Any, *, train_scope: str = "action_expert") -> list[EggrollLayerSpec]:
    """Discover trainable `nn.Linear` modules for EGGROLL by dotted name."""

    prefixes = _scope_prefixes(str(train_scope))
    root = policy if isinstance(policy, nn.Module) else getattr(policy, "_policy", policy)
    specs: list[EggrollLayerSpec] = []
    for name, module in _iter_named_linears(root):
        if "" not in prefixes and not any(name == p or name.startswith(p + ".") for p in prefixes):
            continue
        specs.append(
            EggrollLayerSpec(
                layer_id=len(specs),
                name=name,
                out_features=int(module.out_features),
                in_features=int(module.in_features),
            )
        )
    if not specs:
        raise RuntimeError(f"no EGGROLL nn.Linear layers found for train_scope={train_scope!r}")
    return specs


def modules_for_specs(policy: Any, specs: Iterable[EggrollLayerSpec]) -> dict[int, nn.Linear]:
    """Map discovered specs back to live modules."""

    wanted = {spec.name: spec for spec in specs}
    root = policy if isinstance(policy, nn.Module) else getattr(policy, "_policy", policy)
    out: dict[int, nn.Linear] = {}
    for name, module in _iter_named_linears(root):
        spec = wanted.get(name)
        if spec is not None:
            out[int(spec.layer_id)] = module
    missing = sorted(set(wanted) - {spec.name for spec in specs if spec.layer_id in out})
    if missing:
        raise RuntimeError(f"missing EGGROLL modules for specs: {missing[:10]}")
    return out
