"""Per-row low-rank `nn.Linear` perturbation patch for EGGROLL."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
from types import MethodType
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from smolvla_grpo.eggroll_noise import EggrollLayerSpec, EggrollNoiseManager


@dataclass(frozen=True)
class EggrollLinearContext:
    noise_manager: EggrollNoiseManager
    specs_by_module: dict[int, EggrollLayerSpec]
    module_ids: dict[int, int]
    iteration: int
    sigma: float
    member_ids: torch.Tensor | None = None
    member_id: int | None = None


_ACTIVE_CONTEXT: ContextVar[EggrollLinearContext | None] = ContextVar(
    "eggroll_linear_context",
    default=None,
)


@contextmanager
def eggroll_linear_context(ctx: EggrollLinearContext) -> Iterator[None]:
    token = _ACTIVE_CONTEXT.set(ctx)
    try:
        yield
    finally:
        _ACTIVE_CONTEXT.reset(token)


class EggrollLinearPatchHandle:
    """Install/remove patched forwards on selected Linear modules."""

    def __init__(self, modules: dict[int, nn.Linear], specs: list[EggrollLayerSpec]) -> None:
        self.modules = dict(modules)
        self.specs = list(specs)
        self.specs_by_module = {id(self.modules[spec.layer_id]): spec for spec in self.specs}
        self.module_ids = {id(module): int(layer_id) for layer_id, module in self.modules.items()}
        self._original_forwards: dict[int, Any] = {}

    def install(self) -> None:
        for layer_id, module in self.modules.items():
            key = id(module)
            if key in self._original_forwards:
                continue
            self._original_forwards[key] = module.forward
            module.forward = MethodType(_eggroll_linear_forward, module)

    def remove(self) -> None:
        for module in self.modules.values():
            original = self._original_forwards.pop(id(module), None)
            if original is not None:
                module.forward = original

    def context(
        self,
        *,
        noise_manager: EggrollNoiseManager,
        iteration: int,
        sigma: float,
        member_ids: torch.Tensor | None = None,
        member_id: int | None = None,
    ) -> EggrollLinearContext:
        return EggrollLinearContext(
            noise_manager=noise_manager,
            specs_by_module=self.specs_by_module,
            module_ids=self.module_ids,
            iteration=int(iteration),
            sigma=float(sigma),
            member_ids=member_ids,
            member_id=member_id,
        )


def install_eggroll_linear_patch(
    modules: dict[int, nn.Linear],
    specs: list[EggrollLayerSpec],
) -> EggrollLinearPatchHandle:
    handle = EggrollLinearPatchHandle(modules, specs)
    handle.install()
    return handle


def _base_linear(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, module.weight, module.bias)


def _reshape_for_delta(x: torch.Tensor, *, batch: int, in_features: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    if int(x.shape[-1]) != int(in_features):
        raise RuntimeError(
            f"EGGROLL Linear input features mismatch: expected {in_features}, got {int(x.shape[-1])}"
        )
    original_shape = tuple(x.shape)
    return x.reshape(batch, -1, in_features), original_shape


def _scalar_delta(
    x: torch.Tensor,
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    flat = x.reshape(-1, int(b.shape[0]))
    delta = (flat @ b) @ a.T
    return delta.reshape(*x.shape[:-1], int(a.shape[0])) * float(scale)


def _batched_delta(
    x: torch.Tensor,
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch = int(x.shape[0])
    flat, original_shape = _reshape_for_delta(x, batch=batch, in_features=int(b.shape[1]))
    tmp = torch.einsum("bti,bir->btr", flat, b)
    delta = torch.einsum("btr,bor->bto", tmp, a)
    return delta.reshape(*original_shape[:-1], int(a.shape[1])) * float(scale)


def _rowwise_2d_delta(
    x: torch.Tensor,
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    tmp = torch.einsum("ni,nir->nr", x, b)
    delta = torch.einsum("nr,nor->no", tmp, a)
    return delta * float(scale)


def _eggroll_linear_forward(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    ctx = _ACTIVE_CONTEXT.get()
    base = _base_linear(module, x)
    if ctx is None:
        return base

    spec = ctx.specs_by_module.get(id(module))
    if spec is None:
        return base
    if ctx.member_ids is None and ctx.member_id is None:
        return base
    if ctx.member_ids is not None and ctx.member_id is not None:
        raise RuntimeError("EGGROLL context cannot set both member_ids and member_id")

    scale = float(ctx.sigma) / math.sqrt(float(ctx.noise_manager.rank))
    if ctx.member_id is not None:
        a, b, sign = ctx.noise_manager.generate_factors(
            spec,
            member_id=int(ctx.member_id),
            iteration=int(ctx.iteration),
            device=x.device,
            dtype=x.dtype,
        )
        return base + _scalar_delta(x, a=a, b=b, scale=scale * sign)

    member_ids = ctx.member_ids
    if member_ids is None:
        return base
    batch = int(member_ids.numel())
    expanded_member_ids = member_ids
    if int(x.shape[0]) != batch:
        if x.dim() == 2 and int(x.shape[0]) % batch == 0:
            rows_per_member = int(x.shape[0]) // batch
            expanded_member_ids = member_ids.repeat_interleave(rows_per_member)
        else:
            raise RuntimeError("member_ids batch mismatch")
    if int(x.shape[0]) != int(expanded_member_ids.numel()):
        raise RuntimeError("member_ids batch mismatch")

    a_rows: list[torch.Tensor] = []
    b_rows: list[torch.Tensor] = []
    signs: list[float] = []
    for member_tensor in expanded_member_ids.detach().cpu().reshape(-1):
        a, b, sign = ctx.noise_manager.generate_factors(
            spec,
            member_id=int(member_tensor.item()),
            iteration=int(ctx.iteration),
            device=x.device,
            dtype=x.dtype,
        )
        a_rows.append(a)
        b_rows.append(b)
        signs.append(float(sign))

    a_stack = torch.stack(a_rows, dim=0)
    b_stack = torch.stack(b_rows, dim=0)
    sign_t = torch.as_tensor(signs, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
    if x.dim() == 2:
        delta = _rowwise_2d_delta(x, a=a_stack, b=b_stack, scale=scale)
    else:
        delta = _batched_delta(x, a=a_stack, b=b_stack, scale=scale)
    return base + delta * sign_t.reshape((int(x.shape[0]),) + (1,) * (delta.dim() - 1))
