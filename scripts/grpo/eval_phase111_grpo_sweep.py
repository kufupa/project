#!/usr/bin/env python3
"""Run official-backend eval sweep over Phase111 GRPO checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]


def _checkpoint_update(path: Path) -> int:
    stem = path.stem
    if stem.startswith("update_"):
        return int(stem.split("_", 1)[1])
    raise ValueError(f"unexpected checkpoint name: {path.name}")


def _run_eval(
    *,
    base_checkpoint: str,
    checkpoint_path: Path,
    output_dir: Path,
    task: str,
    episodes: int,
    eval_seed_start: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/grpo/eval_phase11_checkpoints.py",
        "--base-checkpoint",
        base_checkpoint,
        "--grpo-checkpoint",
        str(checkpoint_path),
        "--task",
        task,
        "--eval-seed-start",
        str(eval_seed_start),
        "--episodes",
        str(episodes),
        "--output-dir",
        str(output_dir),
        "--max-steps",
        "0",
        "--env-backend",
        "official_lerobot",
        "--save-official-eval-info",
    ]
    subprocess.run(cmd, check=True, cwd=str(_REPO))
    summary_path = output_dir / "eval_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_sweep(
    *,
    base_checkpoint: str,
    run_dir: Path,
    task: str,
    episodes: int,
    eval_seed_start: int,
    top_k: int = 0,
    top_k_episodes: int = 50,
) -> dict[str, Any]:
    checkpoints_dir = run_dir / "checkpoints"
    ckpts = sorted(checkpoints_dir.glob("update_*.pt"), key=_checkpoint_update)
    if not ckpts:
        raise ValueError(f"no checkpoints found under {checkpoints_dir}")

    sweep_dir = run_dir / "eval_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for ckpt in ckpts:
        upd = _checkpoint_update(ckpt)
        out_dir = sweep_dir / f"update_{upd:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = _run_eval(
            base_checkpoint=base_checkpoint,
            checkpoint_path=ckpt,
            output_dir=out_dir,
            task=task,
            episodes=episodes,
            eval_seed_start=eval_seed_start,
        )
        rows.append(
            {
                "update": upd,
                "checkpoint": str(ckpt),
                "pc_success": float(summary.get("pc_success", 0.0)),
                "avg_sum_reward": float(summary.get("avg_sum_reward", 0.0)),
                "avg_max_reward": float(summary.get("avg_max_reward", 0.0)),
                "episodes": int(summary.get("episodes", episodes)),
                "eval_summary_path": str(out_dir / "eval_summary.json"),
            }
        )

    rows_sorted = sorted(rows, key=lambda r: r["update"])
    result: dict[str, Any] = {
        "task": task,
        "episodes": episodes,
        "eval_seed_start": eval_seed_start,
        "rows": rows_sorted,
    }

    if top_k > 0:
        ranked = sorted(rows_sorted, key=lambda r: (r["pc_success"], r["avg_sum_reward"]), reverse=True)
        top_rows = ranked[:top_k]
        reevaluated: list[dict[str, Any]] = []
        for row in top_rows:
            upd = int(row["update"])
            ckpt = Path(row["checkpoint"])
            out_dir = sweep_dir / f"topk_update_{upd:04d}_{top_k_episodes}ep"
            out_dir.mkdir(parents=True, exist_ok=True)
            summary = _run_eval(
                base_checkpoint=base_checkpoint,
                checkpoint_path=ckpt,
                output_dir=out_dir,
                task=task,
                episodes=top_k_episodes,
                eval_seed_start=eval_seed_start,
            )
            reevaluated.append(
                {
                    "update": upd,
                    "checkpoint": str(ckpt),
                    "pc_success": float(summary.get("pc_success", 0.0)),
                    "avg_sum_reward": float(summary.get("avg_sum_reward", 0.0)),
                    "avg_max_reward": float(summary.get("avg_max_reward", 0.0)),
                    "episodes": int(summary.get("episodes", top_k_episodes)),
                    "eval_summary_path": str(out_dir / "eval_summary.json"),
                }
            )
        result["topk"] = {
            "top_k": int(top_k),
            "top_k_episodes": int(top_k_episodes),
            "rows": reevaluated,
        }

    out_path = sweep_dir / "eval_sweep_summary.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["output_path"] = str(out_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, default="push-v3")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval-seed-start", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-k-episodes", type=int, default=50)
    args = parser.parse_args()

    result = run_sweep(
        base_checkpoint=args.base_checkpoint,
        run_dir=args.run_dir.expanduser().resolve(),
        task=args.task,
        episodes=args.episodes,
        eval_seed_start=args.eval_seed_start,
        top_k=args.top_k,
        top_k_episodes=args.top_k_episodes,
    )
    print(
        "phase111_eval_sweep_ok",
        f"rows={len(result['rows'])}",
        f"task={result['task']}",
        f"out={result['output_path']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
