"""
JEPA-wms MetaWorld render parity (facebookresearch/jepa-wms `MetaWorldWrapper`).

Reference: `evals/simu_env_planning/envs/metaworld.py` — `corner2`, `cam_pos[2]`
patch, square `img_size`, `MujocoRenderer` with that camera, and
`env.render().copy()[::-1]` (vertical flip only).

Segment-GRPO sim with parity enabled uses `render_jepa_rgb` as the single RGB
contract for SmolVLA chunk sampling, WM encode/score, and comparison-strip
“real” frames (no separate legacy policy camera in that mode).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def build_jepa_metaworld_env(task: str, *, img_size: int, seed: int | None = None) -> tuple[Any, list]:
    """
    Build a MetaWorld env matching jepa-wms planning sim rendering.

    Caller must `set_task` from `train_tasks` (returned second) and `reset`.

    `seed` is reserved for API symmetry; use `env.reset(seed=...)`.
    """
    _ = seed  # caller resets with seed
    import metaworld
    from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

    os.environ.setdefault("MUJOCO_GL", "egl")
    mt1 = metaworld.MT1(task)
    env_cls = mt1.train_classes[task]
    env: Any
    try:
        env = env_cls(render_mode="rgb_array", camera_name="corner2")
    except TypeError:
        env = env_cls()
        try:
            if hasattr(env, "render_mode"):
                env.render_mode = "rgb_array"
        except Exception as exc:
            logger.warning("failed to set render_mode on jepa-parity env: %s", exc)
        try:
            env.camera_name = "corner2"
        except Exception as exc:
            logger.debug("could not set camera_name on ctor fallback env: %s", exc)

    env.model.cam_pos[2] = [0.75, 0.075, 0.7]
    env.camera_name = "corner2"
    env.width = env.height = int(img_size)

    if getattr(env, "mujoco_renderer", None) is None:
        try:
            env.render()
        except Exception as exc:
            logger.debug("warm render before MujocoRenderer: %s", exc)

    mr = getattr(env, "mujoco_renderer", None)
    if mr is None:
        err = RuntimeError(
            "MetaWorld env has no mujoco_renderer after warm render(); "
            "check metaworld / gymnasium / MuJoCo versions for jepa-parity sim."
        )
        logger.error("%s", err)
        raise err

    try:
        env.mujoco_renderer = MujocoRenderer(
            env.model,
            env.data,
            mr.default_cam_config,
            width=int(img_size),
            height=int(img_size),
            max_geom=mr.max_geom,
            camera_id=None,
            camera_name="corner2",
        )
    except Exception as exc:
        logger.error("MujocoRenderer rebuild failed: %s", exc)
        raise RuntimeError(
            "Failed to init jepa-parity MujocoRenderer (corner2, square size). "
            "See metaworld_jepa_render.build_jepa_metaworld_env. "
            f"Original error: {exc}"
        ) from exc

    train_tasks = getattr(mt1, "train_tasks", None) or []
    return env, list(train_tasks)


def render_jepa_rgb(env: Any) -> np.ndarray:
    """RGB uint8 HWC, vertical flip to match jepa-wms `MetaWorldWrapper.render`."""
    arr = np.asarray(env.render().copy())[::-1]  # V-flip
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.max(arr)) <= 1.5:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)
