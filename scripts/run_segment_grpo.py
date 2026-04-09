#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from segment_grpo_loop import (  # noqa: E402
    WMBundle,
    load_smolvla_bundle,
    load_wm_bundle,
    rollout_with_chunks,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment-level GRPO loop for push-v3 style tasks.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to sample.")
    parser.add_argument("--chunk-len", type=int, default=8, help="Action chunk length A.")
    parser.add_argument("--num-candidates", type=int, default=6, help="Number of chunk candidates K.")
    parser.add_argument("--max-steps", type=int, default=200, help="Per-episode max environment steps.")
    parser.add_argument("--task", type=str, default="push-v3", help="Task name.")
    parser.add_argument("--checkpoint", type=str, default="", help="SmolVLA checkpoint (HF id or local path).")
    parser.add_argument("--jepa-repo", type=Path, default=None, help="Local/remote JEPA-WM hub repo path.")
    parser.add_argument("--jepa-ckpt", type=str, default="jepa_wm_metaworld.pth.tar", help="JEPA-WM checkpoint.")
    parser.add_argument("--goal-latent-source", type=str, default="", help="Path to goal latent target.")
    parser.add_argument(
        "--carry-mode",
        type=str,
        choices=["sim", "replay"],
        default="sim",
        help="Carry mode: sim (env step) or replay (index hop).",
    )
    parser.add_argument("--replay-root", type=Path, default=None, help="Replay source with images/proprio arrays.")
    parser.add_argument("--output-json", type=Path, required=True, help="Output JSON artifact path.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--dry-run", action="store_true", help="Enable synthetic fallback for missing models/paths.")
    parser.add_argument("--train-steps", type=int, default=0, help="Optional adapter training steps per segment.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device for model loading (auto/cpu/cuda).",
    )
    return parser.parse_args()


def _resolve_output_path(base: Path, episode_idx: int, episodes: int) -> Path:
    if episodes <= 1:
        path = base
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        return path

    if base.suffix == ".json":
        out_dir = base.parent
        stem = base.stem
    else:
        out_dir = base if (base.suffix == "" or base.is_dir()) else base.parent
        stem = base.stem if base.suffix else base.name
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_episode_{episode_idx:04d}.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    args = _parse_args()
    device = args.device

    smolvla_bundle = None
    try:
        if args.checkpoint:
            smolvla_bundle = load_smolvla_bundle(args.checkpoint, device)
        elif args.dry_run:
            print("[segment_grpo] --checkpoint missing: using synthetic action fallback.")
        else:
            print("Error: --checkpoint is required when not using --dry-run.")
            return 1
    except Exception as exc:
        if args.dry_run:
            print(f"[segment_grpo] SmolVLA load failed, using dry-run fallback: {exc}")
            smolvla_bundle = None
        else:
            raise

    wm_bundle: WMBundle | None = None
    try:
        wm_bundle = load_wm_bundle(
            args.jepa_repo,
            args.jepa_ckpt,
            device,
            required=False,
        )
    except Exception as exc:
        if args.dry_run:
            print(f"[segment_grpo] JEPA-WM load failed, using synthetic/fallback scoring: {exc}")
            wm_bundle = None
        else:
            raise

    if smolvla_bundle is None and not args.dry_run:
        print("Error: SmolVLA bundle unavailable and --dry-run is disabled.")
        return 1

    adapter = None
    episode_summaries: list[dict[str, object]] = []
    for episode_idx in range(int(args.episodes)):
        episode_log, adapter = rollout_with_chunks(
            smolvla_bundle,
            wm_bundle,
            task=args.task,
            episode_index=episode_idx,
            chunk_len=int(args.chunk_len),
            num_candidates=int(args.num_candidates),
            max_steps=int(args.max_steps),
            carry_mode=args.carry_mode,
            replay_root=args.replay_root,
            goal_latent_source=args.goal_latent_source,
            seed=int(args.seed) + episode_idx * 997,
            train_steps=int(args.train_steps),
            dry_run=args.dry_run,
            adapter=adapter,
        )
        episode_path = _resolve_output_path(args.output_json, episode_idx, int(args.episodes))
        _write_json(episode_path, episode_log.to_dict())
        episode_summaries.append(
            {
                "episode_index": int(episode_idx),
                "output_json": str(episode_path),
                "steps": int(episode_log.steps),
                "done": bool(episode_log.done),
                "selected_indices": list(episode_log.selected_indices),
                "latent_scores": list(episode_log.latent_scores),
                "selected_scores": list(episode_log.selected_scores),
            }
        )

    if int(args.episodes) > 1:
        if args.output_json.suffix == ".json":
            manifest_dir = args.output_json.parent
        else:
            manifest_dir = args.output_json if args.output_json.is_dir() else args.output_json.parent
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "segment_grpo_manifest.json"
        _write_json(
            manifest_path,
            {
                "task": args.task,
                "episodes": int(args.episodes),
                "chunk_len": int(args.chunk_len),
                "num_candidates": int(args.num_candidates),
                "max_steps": int(args.max_steps),
                "carry_mode": args.carry_mode,
                "dry_run": bool(args.dry_run),
                "output_json": str(args.output_json),
                "episodes_info": episode_summaries,
            },
        )
        print(f"[segment_grpo] wrote manifest: {manifest_path}")

    if episode_summaries:
        print(f"[segment_grpo] wrote episode artifacts for {len(episode_summaries)} episode(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
