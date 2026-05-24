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
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


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
    execution_mode: str,
    n_envs: int,
    rollout_execution: str,
    max_steps: int | None,
    chunk_len: int,
) -> dict[str, Any]:
    if execution_mode == "inprocess_vector":
        from smolvla_grpo.phase11_rollout import load_bundle_for_grpo
        from smolvla_grpo.phase12_vector_eval import evaluate_loaded_policy_vectorized, load_policy_checkpoint_into_bundle

        bundle, _action_dim = load_bundle_for_grpo(
            base_checkpoint,
            task=task,
            env_backend="official_lerobot",
            n_action_steps=int(chunk_len),
        )
        load_policy_checkpoint_into_bundle(bundle, checkpoint_path)
        return evaluate_loaded_policy_vectorized(
            bundle=bundle,
            base_checkpoint=base_checkpoint,
            grpo_checkpoint=checkpoint_path,
            output_dir=output_dir,
            task=task,
            episodes=episodes,
            eval_seed_start=eval_seed_start,
            n_envs=n_envs,
            rollout_execution=rollout_execution,
            max_steps=0 if max_steps is None else int(max_steps),
            chunk_len=int(chunk_len),
        )
    if execution_mode != "subprocess":
        raise ValueError("execution_mode must be 'subprocess' or 'inprocess_vector'")
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
        str(0 if max_steps is None else int(max_steps)),
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
    sweep_name: str = "eval_sweep",
    min_update: int | None = None,
    max_update: int | None = None,
    execution_mode: str = "inprocess_vector",
    n_envs: int = 4,
    rollout_execution: str = "vector_async",
    max_steps: int | None = None,
    chunk_len: int = 1,
) -> dict[str, Any]:
    checkpoints_dir = run_dir / "checkpoints"
    ckpts = sorted(checkpoints_dir.glob("update_*.pt"), key=_checkpoint_update)
    if not ckpts:
        raise ValueError(f"no checkpoints found under {checkpoints_dir}")
    if min_update is not None:
        ckpts = [c for c in ckpts if _checkpoint_update(c) >= int(min_update)]
    if max_update is not None:
        ckpts = [c for c in ckpts if _checkpoint_update(c) <= int(max_update)]
    if not ckpts:
        raise ValueError(
            "no checkpoints remain after min_update/max_update filter "
            f"(min_update={min_update}, max_update={max_update})"
        )

    sweep_dir = run_dir / sweep_name
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
            execution_mode=execution_mode,
            n_envs=n_envs,
            rollout_execution=rollout_execution,
            max_steps=max_steps,
            chunk_len=chunk_len,
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
        "sweep_name": sweep_name,
        "rows": rows_sorted,
        "min_update": int(min_update) if min_update is not None else None,
        "max_update": int(max_update) if max_update is not None else None,
        "execution_mode": str(execution_mode),
        "n_envs": int(n_envs),
        "rollout_execution": str(rollout_execution),
        "max_steps": None if max_steps is None else int(max_steps),
        "chunk_len": int(chunk_len),
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
                execution_mode=execution_mode,
                n_envs=n_envs,
                rollout_execution=rollout_execution,
                max_steps=max_steps,
                chunk_len=chunk_len,
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
    parser.add_argument("--sweep-name", type=str, default="eval_sweep")
    parser.add_argument(
        "--min-update",
        type=int,
        default=None,
        help="If set, only evaluate checkpoints with update index >= this value.",
    )
    parser.add_argument(
        "--max-update",
        type=int,
        default=None,
        help="If set, only evaluate checkpoints with update index <= this value.",
    )
    parser.add_argument("--execution-mode", choices=("subprocess", "inprocess_vector"), default="inprocess_vector")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--rollout-execution", choices=("vector_sync", "vector_async"), default="vector_async")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--chunk-len", type=int, default=1)
    args = parser.parse_args()

    result = run_sweep(
        base_checkpoint=args.base_checkpoint,
        run_dir=args.run_dir.expanduser().resolve(),
        task=args.task,
        episodes=args.episodes,
        eval_seed_start=args.eval_seed_start,
        top_k=args.top_k,
        top_k_episodes=args.top_k_episodes,
        sweep_name=args.sweep_name,
        min_update=args.min_update,
        max_update=args.max_update,
        execution_mode=args.execution_mode,
        n_envs=args.n_envs,
        rollout_execution=args.rollout_execution,
        max_steps=args.max_steps,
        chunk_len=args.chunk_len,
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
