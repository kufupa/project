#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


from src.smolvla_pipeline.targets import load_topk_targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Materialize oracle top-k episodes into an explicit SmolVLA target contract.",
    )
    p.add_argument("--oracle-run-dir", required=True)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError(f"--top-k must be positive, got: {args.top_k}")

    root = Path(args.oracle_run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"oracle run dir not found: {root}")

    run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    targets = load_topk_targets(root, top_k=args.top_k)

    required_keys = ("seed", "max_steps", "camera_name", "flip_corner2")
    missing = [key for key in required_keys if key not in run_manifest]
    if missing:
        raise KeyError(f"run_manifest.json missing required fields: {', '.join(missing)}")

    out = {
        "oracle_run_dir": str(root),
        "top_k": args.top_k,
        "task": str(run_manifest["task"]),
        "base_seed": int(run_manifest["seed"]),
        "oracle_max_steps": int(run_manifest["max_steps"]),
        "camera_name": str(run_manifest["camera_name"]),
        "flip_corner2": bool(run_manifest["flip_corner2"]),
        "count": len(targets),
        "targets": targets,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
