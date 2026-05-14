"""Official JEPA-WM MetaWorld objective helpers for Phase12."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class Phase12Distance:
    visual_distance: float
    proprio_distance: float
    combined_distance: float


@dataclass(frozen=True)
class Phase12Score:
    candidate_index: int
    start_visual_distance: float
    start_proprio_distance: float
    start_combined_distance: float
    final_visual_distance: float
    final_proprio_distance: float
    final_combined_distance: float
    wm_latent_progress: float
    latent_return: float
    wm_status: str
    debug_npz_path: str | None = None


def _as_tensor(value: Any, *, key: str) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().float()
    try:
        return torch.as_tensor(value, dtype=torch.float32).detach()
    except Exception as exc:  # pragma: no cover - defensive message path
        raise TypeError(f"Could not convert latent field {key!r} to tensor: {exc}") from exc


def _get_field(encoded: Any, key: str) -> Any:
    if isinstance(encoded, Mapping):
        if key not in encoded:
            raise KeyError(f"Structured JEPA-WM latent missing {key!r}")
        return encoded[key]
    if hasattr(encoded, "get"):
        value = encoded.get(key, None)
        if value is None:
            raise KeyError(f"Structured JEPA-WM latent missing {key!r}")
        return value
    try:
        return encoded[key]
    except Exception as exc:
        raise KeyError(f"Structured JEPA-WM latent missing {key!r}") from exc


def split_structured_latent(
    encoded: Any,
    *,
    mode: str = "visual_proprio",
) -> dict[str, torch.Tensor]:
    """Return detached visual/proprio tensors for the Phase12 objective.

    Default mode intentionally requires proprio because it mirrors the official
    JEPA-WM MetaWorld planning objective.
    """

    mode_norm = str(mode).strip().lower()
    if mode_norm not in ("visual_proprio", "visual_only_ablation"):
        raise ValueError("mode must be 'visual_proprio' or 'visual_only_ablation'")
    out = {"visual": _as_tensor(_get_field(encoded, "visual"), key="visual")}
    if mode_norm == "visual_proprio":
        out["proprio"] = _as_tensor(_get_field(encoded, "proprio"), key="proprio")
    return out


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).pow(2).mean().item())


def combined_l2_distance(
    pred: Mapping[str, torch.Tensor],
    goal: Mapping[str, torch.Tensor],
    *,
    proprio_alpha: float = 0.1,
    mode: str = "visual_proprio",
) -> Phase12Distance:
    """Official MetaWorld JEPA-WM L2 objective: visual + alpha * proprio."""

    mode_norm = str(mode).strip().lower()
    visual = _mse(pred["visual"], goal["visual"])
    if mode_norm == "visual_only_ablation":
        proprio = 0.0
    elif mode_norm == "visual_proprio":
        proprio = _mse(pred["proprio"], goal["proprio"])
    else:
        raise ValueError("mode must be 'visual_proprio' or 'visual_only_ablation'")
    combined = visual + float(proprio_alpha) * proprio
    return Phase12Distance(
        visual_distance=visual,
        proprio_distance=proprio,
        combined_distance=float(combined),
    )


def score_progress(
    *,
    candidate_index: int,
    start: Mapping[str, torch.Tensor],
    final: Mapping[str, torch.Tensor],
    goal: Mapping[str, torch.Tensor],
    proprio_alpha: float = 0.1,
    mode: str = "visual_proprio",
    debug_npz_path: str | None = None,
) -> Phase12Score:
    start_dist = combined_l2_distance(start, goal, proprio_alpha=proprio_alpha, mode=mode)
    final_dist = combined_l2_distance(final, goal, proprio_alpha=proprio_alpha, mode=mode)
    progress = start_dist.combined_distance - final_dist.combined_distance
    return Phase12Score(
        candidate_index=int(candidate_index),
        start_visual_distance=start_dist.visual_distance,
        start_proprio_distance=start_dist.proprio_distance,
        start_combined_distance=start_dist.combined_distance,
        final_visual_distance=final_dist.visual_distance,
        final_proprio_distance=final_dist.proprio_distance,
        final_combined_distance=final_dist.combined_distance,
        wm_latent_progress=float(progress),
        latent_return=float(-final_dist.combined_distance),
        wm_status="ok",
        debug_npz_path=debug_npz_path,
    )

