"""Phase12 action profiles and telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Phase12ActionProfileResult:
    exec_actions_raw_postprocessed: np.ndarray
    exec_actions_clipped: np.ndarray
    exec_actions_for_env: np.ndarray
    exec_actions_for_wm: np.ndarray
    metadata: dict[str, Any]


def _as_actions_2d(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"actions must be 1D or 2D, got shape {arr.shape}")
    return arr


def _bounds(bound: float | np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(bound, dtype=np.float32)
    if arr.ndim == 0:
        return np.full((dim,), float(arr), dtype=np.float32)
    flat = arr.reshape(-1)
    if flat.size == 1:
        return np.full((dim,), float(flat[0]), dtype=np.float32)
    if flat.size >= int(dim):
        return flat[: int(dim)].astype(np.float32, copy=False)
    return np.pad(flat, (0, int(dim) - int(flat.size)), mode="edge").astype(np.float32, copy=False)


def _stats(prefix: str, arr: np.ndarray) -> dict[str, float]:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
        }
    return {
        f"{prefix}_min": float(np.min(flat)),
        f"{prefix}_max": float(np.max(flat)),
        f"{prefix}_mean": float(np.mean(flat)),
        f"{prefix}_std": float(np.std(flat)),
    }


def _normalize_actions(preprocessor: Any, actions: np.ndarray) -> np.ndarray:
    if hasattr(preprocessor, "normalize_actions"):
        try:
            out = preprocessor.normalize_actions(actions)
        except TypeError:
            try:
                import torch

                device = getattr(getattr(preprocessor, "action_mean", None), "device", "cpu")
                batch = torch.as_tensor(actions, dtype=torch.float32, device=device)
                out = preprocessor.normalize_actions(batch)
            except Exception:
                out = None
        if out is not None:
            if hasattr(out, "detach"):
                return out.detach().cpu().numpy().astype(np.float32)
            return np.asarray(out, dtype=np.float32)
    mean = getattr(preprocessor, "action_mean", None)
    std = getattr(preprocessor, "action_std", None)
    if mean is None or std is None:
        raise AttributeError("preprocessor must expose normalize_actions() or action_mean/action_std")
    if hasattr(mean, "detach"):
        mean = mean.detach().cpu().numpy()
    if hasattr(std, "detach"):
        std = std.detach().cpu().numpy()
    return (actions - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)


def _packed_shape(n_steps: int, env_action_dim: int | None, wm_action_dim: int | None) -> tuple[list[int] | None, int | None]:
    if env_action_dim is None or wm_action_dim is None:
        return None, None
    env_dim = int(env_action_dim)
    wm_dim = int(wm_action_dim)
    if env_dim <= 0 or wm_dim <= 0 or wm_dim % env_dim != 0:
        return None, None
    factor = wm_dim // env_dim
    rows = int(np.ceil(float(n_steps) / float(factor)))
    return [rows, wm_dim], factor


def apply_phase12_action_profile(
    raw_postprocessed_actions: np.ndarray,
    *,
    action_profile: str,
    action_low: float | np.ndarray,
    action_high: float | np.ndarray,
    preprocessor: Any | None = None,
    env_action_dim: int | None = None,
    wm_action_dim: int | None = None,
) -> Phase12ActionProfileResult:
    """Choose action arrays for env execution and JEPA-WM scoring."""

    raw = _as_actions_2d(raw_postprocessed_actions)
    low = _bounds(action_low, raw.shape[1])
    high = _bounds(action_high, raw.shape[1])
    clipped = np.clip(raw, low.reshape(1, -1), high.reshape(1, -1)).astype(np.float32, copy=False)
    changed = np.not_equal(raw, clipped)

    profile = str(action_profile).strip().lower()
    if profile == "official_jepa_mirror":
        env_actions = raw
        wm_actions = raw
        source = "raw_postprocessed"
    elif profile == "bounded_executed":
        env_actions = clipped
        wm_actions = clipped
        source = "clipped"
    else:
        raise ValueError("action_profile must be 'official_jepa_mirror' or 'bounded_executed'")

    metadata: dict[str, Any] = {
        "action_profile": profile,
        "clip_fraction": float(np.mean(changed)),
        "clip_any": bool(np.any(changed)),
        "exec_action_source": source,
        "wm_action_source": source,
    }
    metadata.update(_stats("raw_action", raw))
    metadata.update(_stats("clipped_action", clipped))

    if preprocessor is not None:
        norm = _normalize_actions(preprocessor, wm_actions)
        metadata.update(_stats("jepa_norm", norm))
        metadata["jepa_norm_max_abs"] = float(np.max(np.abs(norm))) if norm.size else 0.0

    packed_shape, pack_factor = _packed_shape(int(wm_actions.shape[0]), env_action_dim, wm_action_dim)
    if env_action_dim is not None:
        metadata["env_action_dim"] = int(env_action_dim)
    if wm_action_dim is not None:
        metadata["wm_action_dim"] = int(wm_action_dim)
    if pack_factor is not None:
        metadata["pack_factor"] = int(pack_factor)
    if packed_shape is not None:
        metadata["packed_action_shape"] = packed_shape

    return Phase12ActionProfileResult(
        exec_actions_raw_postprocessed=raw.astype(np.float32, copy=False),
        exec_actions_clipped=clipped,
        exec_actions_for_env=np.asarray(env_actions, dtype=np.float32),
        exec_actions_for_wm=np.asarray(wm_actions, dtype=np.float32),
        metadata=metadata,
    )

