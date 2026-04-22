"""MetaWorld / paper reproducibility: global RNG seeding and strict-env flags.

Ordering contract (match ``run_metaworld_oracle_eval`` / Gymnasium MetaWorld):

1. ``seed_metaworld_process(reset_seed)``
2. ``env.set_task(...)`` when task lists exist
3. ``env.reset(seed=reset_seed)``
4. ``env.step`` with actions clipped to the env action box (e.g. ``[-1, 1]``)

``env.reset(seed=...)`` alone does not re-seed Python ``random``, legacy
``numpy.random``, or PyTorch globals that some MetaWorld / MuJoCo paths use.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def metaworld_strict_ctor_requested() -> bool:
    """When true, oracle env construction uses a single constructor path (no silent fallback)."""
    v = os.environ.get("METAWORLD_STRICT_CTOR", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def seed_metaworld_process(seed: int) -> None:
    """Seed Python / NumPy / PyTorch globals for reproducible MetaWorld rollouts.

    Raises if PyTorch is required but unavailable (paper stack assumes torch is installed).
    """
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "seed_metaworld_process requires PyTorch installed (paper / MT10 reproducibility stack)."
        ) from exc

    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def gymnasium_reset_strict(env: Any, seed: int) -> Any:
    """Call ``env.reset(seed=...)``; no legacy unseeded fallback."""
    try:
        return env.reset(seed=seed)
    except TypeError as exc:
        raise RuntimeError(
            "env.reset(seed=...) is required for reproducible MetaWorld rollouts (Gymnasium API). "
            "Legacy envs without `seed` are not supported."
        ) from exc
