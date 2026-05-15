"""Phase12 pixel-contract helpers.

Policy RGB matches LeRobot/SmolVLA: corner2 + vertical+horizontal flip.
WM RGB matches JEPA-WM/Phase08: corner2 + vertical flip only.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_rgb_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"image must be HxWx3/4, got {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.max(arr)) <= 1.5:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def policy_rgb_from_raw_corner2(raw_rgb: Any) -> np.ndarray:
    return np.ascontiguousarray(np.flip(to_rgb_uint8(raw_rgb), (0, 1)))


def wm_rgb_from_raw_corner2(raw_rgb: Any) -> np.ndarray:
    return np.ascontiguousarray(np.flip(to_rgb_uint8(raw_rgb), 0))


def wm_rgb_from_policy_rgb_corner2(policy_rgb: Any) -> np.ndarray:
    return np.ascontiguousarray(np.flip(to_rgb_uint8(policy_rgb), 1))


def policy_rgb_from_obs(obs: Any) -> np.ndarray:
    if not isinstance(obs, dict) or "pixels" not in obs:
        raise KeyError("observation must contain 'pixels'")
    pixels = np.asarray(obs["pixels"])
    if pixels.ndim == 4:
        if pixels.shape[0] < 1:
            raise ValueError("observation pixels batch is empty")
        pixels = pixels[0]
    return to_rgb_uint8(pixels)
