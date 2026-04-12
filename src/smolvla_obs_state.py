"""Shared observation flattening for SmolVLA / segment-GRPO / MetaWorld backends.

Matches the contract used by ``smolvla_pipeline.evaluator`` (sorted dict keys,
concatenate numeric vector parts). Keeps segment-GRPO proprio aligned with
best_video evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def flatten_obs_state(obs: Any) -> np.ndarray:
    """Return a 1-D float32 vector from env observation (dict, ndarray, or scalar)."""
    if isinstance(obs, np.ndarray):
        return obs.reshape(-1).astype(np.float32, copy=False)
    if isinstance(obs, dict):
        chunks: list[np.ndarray] = []
        for key in sorted(obs):
            value = obs[key]
            try:
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if arr.size > 0:
                chunks.append(arr)
        if chunks:
            return np.concatenate(chunks, axis=0)
        return np.zeros(0, dtype=np.float32)
    return np.asarray(obs, dtype=np.float32).reshape(-1)
