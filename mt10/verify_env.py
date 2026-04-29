#!/usr/bin/env python3
"""Smoke-test Meta-World MT1 (push-v3) + Gymnasium vec MT10 with project determinism helpers."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from metaworld_determinism import gymnasium_reset_strict, seed_metaworld_process  # noqa: E402


def _print_versions() -> None:
    import gymnasium as gym
    import mujoco
    import metaworld

    print(
        "versions:",
        "metaworld",
        getattr(metaworld, "__version__", "?"),
        "gymnasium",
        gym.__version__,
        "mujoco",
        getattr(mujoco, "__version__", "?"),
    )


def _mt1_push_v3_smoke() -> None:
    import metaworld

    mt1 = metaworld.MT1("push-v3")
    if "push-v3" not in mt1.train_classes:
        raise RuntimeError(f"push-v3 missing from MT1.train_classes keys={sorted(mt1.train_classes)!r}")
    env = mt1.train_classes["push-v3"]()
    train_tasks = list(getattr(mt1, "train_tasks", []) or [])
    seed_metaworld_process(0)
    if train_tasks:
        env.set_task(train_tasks[0])
    gymnasium_reset_strict(env, 0)
    env.close()


def _mt10_vec_smoke() -> None:
    import gymnasium as gym

    envs = gym.make_vec("Meta-World/MT10", vector_strategy="sync", seed=0)
    try:
        seed_metaworld_process(0)
        gymnasium_reset_strict(envs, 0)
        actions = envs.action_space.sample()
        envs.step(actions)
    finally:
        envs.close()


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    _print_versions()
    _mt1_push_v3_smoke()
    _mt10_vec_smoke()
    print("mt10_ok")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("mt10_verify_failed:", repr(exc), file=sys.stderr)
        raise SystemExit(1)
