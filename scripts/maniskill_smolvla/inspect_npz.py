#!/usr/bin/env python3
"""Validate RL4VLA raw npz demos for 7D SmolVLA SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.maniskill_smolvla.audit_npz_contract import audit_root


def load_episode(path: Path) -> dict[str, Any]:
    return np.load(path, allow_pickle=True)["arr_0"].tolist()


def summarize(root: Path, min_episodes: int = 1) -> dict[str, Any]:
    summary = audit_root(root, min_episodes=min_episodes)
    if summary["duplicate_decoded_signature_count"] > 0:
        raise SystemExit("duplicate decoded NPZ signatures detected")
    if summary["bad_shape_count"] > 0:
        raise SystemExit(json.dumps(summary["bad_shape_examples"], indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    summary = summarize(args.root, args.min_episodes)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
