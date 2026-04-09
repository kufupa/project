#!/usr/bin/env python3
"""CPU-light check: forked LeRobot exposes GRPO distribution hooks."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected-commit",
        default="f30fc2a1b904bb2ccd752cfff94f6f4423bd523b",
        help="Optional hint string printed for humans (not verified against git).",
    )
    args = parser.parse_args()

    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError as exc:
        print("FAIL: cannot import SmolVLAPolicy:", exc, file=sys.stderr)
        return 2

    checks = {
        "SmolVLAPolicy.select_action_distr_params": hasattr(SmolVLAPolicy, "select_action_distr_params"),
        "SmolVLAPolicy._get_distr_params_chunk": hasattr(SmolVLAPolicy, "_get_distr_params_chunk"),
    }
    ok = all(bool(v) for v in checks.values())
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    print(f"hint: safe-robot pins lerobot @ {args.expected_commit}")
    if not ok:
        print(
            "FAIL: install jsnchon/lerobot fork at pinned commit (see scripts/grpo/README.md).",
            file=sys.stderr,
        )
        return 1
    print("OK: GRPO API surface looks present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
