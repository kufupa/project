#!/usr/bin/env python3
"""Run official-backend eval sweep over Phase12 checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]


def _run_eval(
    *,
    base_checkpoint: str,
    grpo_checkpoint: Path | None,
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
    if grpo_checkpoint is not None:
        cmd.extend(["--grpo-checkpoint", str(grpo_checkpoint)])
    else:
        cmd.extend(["--grpo-checkpoint", str(_make_base_eval_checkpoint(base_checkpoint, output_dir, task=task))])
    subprocess.run(cmd, check=True, cwd=str(_REPO))
    return json.loads((output_dir / "eval_summary.json").read_text(encoding="utf-8"))


def _make_base_eval_checkpoint(base_checkpoint: str, output_dir: Path, *, task: str) -> Path:
    """Write a tiny sentinel checkpoint by loading base policy state once via eval script format."""
    import torch

    from smolvla_grpo.checkpointing import save_grpo_checkpoint
    from smolvla_grpo.phase11_rollout import load_bundle_for_grpo

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "base_update_0000.pt"
    if ckpt_path.exists():
        return ckpt_path
    bundle, _action_dim = load_bundle_for_grpo(
        base_checkpoint,
        task=task,
        env_backend="official_lerobot",
    )
    save_grpo_checkpoint(
        ckpt_path,
        policy_state={k: v.detach().cpu() for k, v in bundle.policy.state_dict().items()},
        optimizer_state={},
        update_index=-1,
        args={"base_checkpoint": base_checkpoint, "synthetic_phase12_eval_base": True},
        extra={"synthetic_phase12_eval_base": True},
    )
    del bundle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ckpt_path


def _checkpoint_path(run_dir: Path, update: int) -> Path:
    return run_dir / "checkpoints" / f"update_{int(update):04d}.pt"


def run_sweep(
    *,
    base_checkpoint: str,
    run_dir: Path,
    task: str,
    episodes: int,
    eval_seed_start: int,
    sweep_name: str,
    min_update: int,
    max_update: int,
    stride: int,
) -> dict[str, Any]:
    if int(stride) <= 0:
        raise ValueError("stride must be positive")
    sweep_dir = run_dir / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    updates = list(range(int(min_update), int(max_update) + 1, int(stride)))
    for update in updates:
        ckpt = None if update == 0 else _checkpoint_path(run_dir, update)
        if ckpt is not None and not ckpt.is_file():
            missing.append(str(ckpt))
            continue
        out_dir = sweep_dir / f"update_{update:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = _run_eval(
            base_checkpoint=base_checkpoint,
            grpo_checkpoint=ckpt,
            output_dir=out_dir,
            task=task,
            episodes=episodes,
            eval_seed_start=eval_seed_start,
        )
        rows.append(
            {
                "update": int(update),
                "checkpoint": str(ckpt) if ckpt is not None else "base_checkpoint",
                "pc_success": float(summary.get("pc_success", 0.0)),
                "avg_sum_reward": float(summary.get("avg_sum_reward", 0.0)),
                "avg_max_reward": float(summary.get("avg_max_reward", 0.0)),
                "episodes": int(summary.get("episodes", episodes)),
                "eval_summary_path": str(out_dir / "eval_summary.json"),
            }
        )
    if missing:
        raise FileNotFoundError("missing requested checkpoints:\n" + "\n".join(missing))
    result = {
        "task": task,
        "episodes": int(episodes),
        "eval_seed_start": int(eval_seed_start),
        "sweep_name": sweep_name,
        "min_update": int(min_update),
        "max_update": int(max_update),
        "stride": int(stride),
        "rows": rows,
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
    parser.add_argument("--sweep-name", type=str, default="eval_sweep_every10")
    parser.add_argument("--min-update", type=int, default=0)
    parser.add_argument("--max-update", type=int, default=300)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    result = run_sweep(
        base_checkpoint=args.base_checkpoint,
        run_dir=args.run_dir.expanduser().resolve(),
        task=args.task,
        episodes=args.episodes,
        eval_seed_start=args.eval_seed_start,
        sweep_name=args.sweep_name,
        min_update=args.min_update,
        max_update=args.max_update,
        stride=args.stride,
    )
    print(
        "phase12_eval_sweep_ok",
        f"rows={len(result['rows'])}",
        f"task={result['task']}",
        f"out={result['output_path']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
