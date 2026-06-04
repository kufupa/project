"""Pytest defaults for headless MetaWorld / MuJoCo on gpucluster login nodes."""

from __future__ import annotations

import os


def _force_egl_mujoco() -> None:
    # gpucluster3 login nodes use EGL; keep PYOPENGL in sync because imports like
    # smolvla_grpo.lerobot_metaworld_adapter set osmesa during test collection.
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"


_force_egl_mujoco()


def pytest_configure(config) -> None:
    _force_egl_mujoco()


def pytest_collection_modifyitems(items) -> None:
    """Run MetaWorld render smoke tests before long suites can poison GL state."""
    priority = []
    rest = []
    for item in items:
        nodeid = item.nodeid
        if "test_vec_env_smoke" in nodeid or "test_metaworld_jepa_render" in nodeid:
            priority.append(item)
        else:
            rest.append(item)
    items[:] = priority + rest
