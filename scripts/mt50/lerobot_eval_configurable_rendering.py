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


def main() -> None:
    render_limit = parse_max_episodes_rendered(os.environ.get("MT50_LEROBOT_MAX_EPISODES_RENDERED"))

    from lerobot.scripts import lerobot_eval

    lerobot_eval.eval_policy_all = with_configurable_rendering(
        lerobot_eval.eval_policy_all,
        max_episodes_rendered=render_limit,
    )
    lerobot_eval.main()


if __name__ == "__main__":
    main()
