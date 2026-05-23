#!/usr/bin/env python3
"""Run LeRobot eval with a repo-controlled video render limit.

LeRobot's CLI entrypoint currently hardcodes ``max_episodes_rendered=10``.
This adapter preserves the official eval path and overrides only that render
limit at the ``eval_policy_all`` call boundary.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from typing import Any


def parse_max_episodes_rendered(raw: str | None) -> int:
    """Parse MT50_LEROBOT_MAX_EPISODES_RENDERED."""
    if raw is None or raw == "":
        raise ValueError("MT50_LEROBOT_MAX_EPISODES_RENDERED must be set")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("MT50_LEROBOT_MAX_EPISODES_RENDERED must be a non-negative integer") from exc
    if value < 0:
        raise ValueError("MT50_LEROBOT_MAX_EPISODES_RENDERED must be a non-negative integer")
    return value


def parse_freeze_rand_vec(raw: str | None) -> bool | None:
    """Parse MT50_METAWORLD_FREEZE_RAND_VEC."""
    if raw is None or raw == "":
        return None
    normalized = raw.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError("MT50_METAWORLD_FREEZE_RAND_VEC must be a boolean value")


def with_configurable_rendering(
    eval_policy_all: Callable[..., dict[str, Any]],
    *,
    max_episodes_rendered: int,
) -> Callable[..., dict[str, Any]]:
    """Wrap LeRobot eval_policy_all and force the desired render limit."""

    @wraps(eval_policy_all)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs["max_episodes_rendered"] = max_episodes_rendered
        if max_episodes_rendered == 0:
            kwargs["videos_dir"] = None
        return eval_policy_all(*args, **kwargs)

    return wrapped


def apply_metaworld_freeze_rand_vec(value: bool) -> None:
    """Monkeypatch MetaWorld env creation with fixed freeze flag."""
    from lerobot.envs import metaworld as _mw

    _orig = _mw.MetaworldEnv._make_envs_task

    def _patched(self: Any, env_name: str) -> Any:
        env = _orig(self, env_name)
        env._freeze_rand_vec = bool(value)
        return env

    _mw.MetaworldEnv._make_envs_task = _patched


def main() -> None:
    render_limit_raw = os.environ.get("MT50_LEROBOT_MAX_EPISODES_RENDERED")
    freeze_rand_vec = parse_freeze_rand_vec(os.environ.get("MT50_METAWORLD_FREEZE_RAND_VEC"))

    from lerobot.scripts import lerobot_eval

    if render_limit_raw is not None and render_limit_raw != "":
        render_limit = parse_max_episodes_rendered(render_limit_raw)
        lerobot_eval.eval_policy_all = with_configurable_rendering(
            lerobot_eval.eval_policy_all,
            max_episodes_rendered=render_limit,
        )
    if freeze_rand_vec is not None:
        apply_metaworld_freeze_rand_vec(freeze_rand_vec)
        print(f"[mt50:adapter] metaworld_freeze_rand_vec={freeze_rand_vec}")
    lerobot_eval.main()


if __name__ == "__main__":
    main()
