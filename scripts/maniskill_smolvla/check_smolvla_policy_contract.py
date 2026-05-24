#!/usr/bin/env python3
"""Validate saved SmolVLA config for ManiSkill 25Main."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_N_ACTION_STEPS = 4


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model_config(root: Path) -> Path:
    candidates = [
        root / "config.json",
        root / "checkpoints" / "last" / "pretrained_model" / "config.json",
    ]
    candidates.extend(sorted(root.glob("checkpoints/*/pretrained_model/config.json")))
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"cannot find SmolVLA config.json under {root}")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    input_features = config["input_features"]
    output_features = config["output_features"]
    image_shape = input_features["observation.images.front"]["shape"]
    state_shape = input_features["observation.state"]["shape"]
    action_shape = output_features["action"]["shape"]
    if image_shape != [3, 480, 640]:
        raise SystemExit(f"bad image shape: {image_shape}")
    if state_shape != [7]:
        raise SystemExit(f"bad state shape: {state_shape}")
    if action_shape != [7]:
        raise SystemExit(f"bad action shape: {action_shape}")
    if int(config["n_action_steps"]) != EXPECTED_N_ACTION_STEPS:
        raise SystemExit(f"bad n_action_steps: {config['n_action_steps']}")
    if int(config["chunk_size"]) != 50:
        raise SystemExit(f"bad chunk_size: {config['chunk_size']}")
    if int(config["max_state_dim"]) != 32 or int(config["max_action_dim"]) != 32:
        raise SystemExit(
            f"bad max dims: state={config['max_state_dim']} action={config['max_action_dim']}"
        )
    return {
        "image_shape": image_shape,
        "state_shape": state_shape,
        "action_shape": action_shape,
        "n_action_steps": config["n_action_steps"],
        "chunk_size": config["chunk_size"],
        "max_state_dim": config["max_state_dim"],
        "max_action_dim": config["max_action_dim"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    config_path = resolve_model_config(args.run_root)
    summary = validate_config(load_json(config_path))
    summary["config_path"] = str(config_path)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
