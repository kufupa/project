#!/usr/bin/env python3
"""Run lerobot-eval while rendering videos for all requested episodes.

This is a project-local wrapper to avoid depending on site-packages edits while
working around the upstream hardcoded max_episodes_rendered default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lerobot.scripts import lerobot_eval


OriginalEvalPolicyAll = lerobot_eval.eval_policy_all


def _patched_eval_policy_all(*args: Any, **kwargs: Any):
    """Use environment override for max_episodes_rendered if present."""
    rendered_cap = os.environ.get("SMOLVLA_MAX_EPISODES_RENDERED")
    if rendered_cap:
        try:
            rendered_cap_value = int(rendered_cap)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid SMOLVLA_MAX_EPISODES_RENDERED={rendered_cap!r}; expected int"
            ) from exc
        if rendered_cap_value < 0:
            raise RuntimeError(
                f"Invalid SMOLVLA_MAX_EPISODES_RENDERED={rendered_cap_value}; expected non-negative int"
            )
        kwargs["max_episodes_rendered"] = rendered_cap_value

    return OriginalEvalPolicyAll(*args, **kwargs)


def main() -> None:
    """Patch upstream eval path and run the wrapped CLI entrypoint."""
    lerobot_eval.eval_policy_all = _patched_eval_policy_all
    # Preserve current working directory for relative path behavior.
    Path.cwd()
    lerobot_eval.main()


if __name__ == "__main__":
    main()
