#!/usr/bin/env python3
"""Phase 9: replay oracle action trajectories through WM + comparison strips (no SmolVLA / GRPO)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from segment_grpo_loop import _resolve_device, load_wm_bundle, rollout_with_chunks  # noqa: E402
from segment_grpo_reference import (  # noqa: E402
    load_oracle_action_sequence,
    load_oracle_reference_frames,
    resolve_latest_oracle_pushv3_run,
)
from smolvla_pipeline.run_layout import ensure_unique_run_dir  # noqa: E402

COMPARISON_STRIP_STITCH_GUTTER_PX_DEFAULT = 6


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 9 oracle-vs-WM replay (oracle actions + WM strips).")
    p.add_argument("--oracle-run-root", type=Path, default=None, help="Explicit phase06 oracle run directory.")
    p.add_argument("--artifacts-root", type=Path, default=ROOT / "artifacts", help="Root for resolve_latest_oracle_pushv3_run.")
    p.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "phase09_oracle_vs_wm_baseline",
        help="Parent for run_* output folders.",
    )
    p.add_argument("--task", type=str, default="push-v3")
    p.add_argument("--episodes", type=int, default=60, help="Max episodes to run (from run_manifest order).")
    p.add_argument("--goal-frame-index", type=int, default=50, help="1-based oracle goal frame index.")
    p.add_argument("--max-steps", type=int, default=50, help="Cap on env steps per episode.")
    p.add_argument("--chunk-len", type=int, default=50, help="Chunk length (single segment = full horizon).")
    p.add_argument("--jepa-repo", type=Path, default=None, help="Local JEPA-WM hub repo path.")
    p.add_argument("--jepa-ckpt", type=str, default="jepa_wm_metaworld.pth.tar")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--wm-rollout-mode", type=str, default="iterative", choices=["iterative", "batched"])
    p.add_argument("--wm-scoring-latent", type=str, default="visual", choices=["visual", "proprio", "concat"])
    p.add_argument("--wm-goal-hflip", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--wm-sim-camera-parity", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--wm-sim-img-size", type=int, default=224)
    p.add_argument("--smolvla-policy-hflip-corner2", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--comparison-strip-overlay", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument(
        "--comparison-strip-stitch-gutter-pixels",
        type=int,
        default=COMPARISON_STRIP_STITCH_GUTTER_PX_DEFAULT,
        help="Gutter between stitched segment strips (default matches run_segment_grpo).",
    )
    p.add_argument("--reset-frame-warning-threshold", type=float, default=0.08)
    p.add_argument("--strict-wm-scoring", action="store_true")
    p.add_argument("--strict-decode", action="store_true")
    p.add_argument("--train-steps", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="Skip WM/sim (for plumbing tests only).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    oracle_run = args.oracle_run_root.expanduser().resolve() if args.oracle_run_root is not None else None
    if oracle_run is None:
        oracle_run = resolve_latest_oracle_pushv3_run(args.artifacts_root.expanduser().resolve(), task=args.task)

    manifest_path = oracle_run / "run_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing run_manifest.json: {manifest_path}")
    oracle_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode_rows = list(oracle_manifest.get("episodes", []))[: int(args.episodes)]
    if not episode_rows:
        raise SystemExit(f"No episodes in {manifest_path}")

    manifest_task = str(oracle_manifest.get("task", "") or "").strip()
    if manifest_task and manifest_task != str(args.task).strip():
        raise SystemExit(
            f"Oracle manifest task mismatch: manifest has {manifest_task!r} but CLI --task is {str(args.task)!r} "
            f"({manifest_path})"
        )

    base_seed = int(oracle_manifest.get("seed", 1000))

    jepa_repo_resolved: str | None
    if args.dry_run:
        wm_bundle = None
        wm_device_record = "dry_run"
        jepa_repo_resolved = str(args.jepa_repo.expanduser().resolve()) if args.jepa_repo is not None else None
    else:
        if args.jepa_repo is None:
            raise SystemExit("Missing --jepa-repo (or pass --dry-run).")
        resolved_repo = args.jepa_repo.expanduser().resolve()
        jepa_repo_resolved = str(resolved_repo)
        wm_bundle = load_wm_bundle(
            resolved_repo,
            args.jepa_ckpt,
            args.device,
            required=True,
        )
        wm_device_record = str(_resolve_device(args.device))

    run_dir = ensure_unique_run_dir(
        args.output_root.expanduser().resolve(),
        episodes=len(episode_rows),
        task=args.task,
        seed=base_seed,
        variant="phase09_oracle_vs_wm",
    )
    print(f"[phase9] run directory: {run_dir}")

    episode_summaries: list[dict[str, object]] = []
    adapter = None

    for loop_idx, row in enumerate(episode_rows):
        ep_idx = int(row["episode_index"])
        seed = int(row["reset_seed"])

        goal_ref = load_oracle_reference_frames(
            oracle_run,
            episode_index=ep_idx,
            goal_frame_index=int(args.goal_frame_index),
            start_frame_index=0,
        )
        act_seq = load_oracle_action_sequence(oracle_run, ep_idx)
        cap = min(int(args.max_steps), int(args.chunk_len), int(act_seq.n_steps))
        if cap < 1:
            raise RuntimeError(f"episode {ep_idx}: effective_max_steps < 1 (oracle has {act_seq.n_steps} steps)")

        stem = f"segment_grpo_episode_{loop_idx:04d}"
        episode_output = run_dir / f"{stem}.json"
        artifact_dir = run_dir / f"{stem}_artifacts"
        comparison_root = artifact_dir / "comparison"

        episode_log, adapter = rollout_with_chunks(
            None,
            wm_bundle,
            task=args.task,
            episode_index=ep_idx,
            chunk_len=cap,
            num_candidates=1,
            max_steps=cap,
            carry_mode="sim",
            replay_root=None,
            goal_latent_source="",
            goal_frame=goal_ref.goal_frame,
            goal_proprio=goal_ref.goal_flat_obs,
            start_frame=goal_ref.start_frame,
            goal_frame_index=int(args.goal_frame_index),
            goal_source=str(goal_ref.goal_frame_path),
            comparison_root=comparison_root,
            reset_frame_warning_threshold=float(args.reset_frame_warning_threshold),
            seed=seed,
            train_steps=int(args.train_steps),
            dry_run=bool(args.dry_run),
            adapter=adapter,
            strict_wm_scoring=bool(args.strict_wm_scoring),
            strict_decode=bool(args.strict_decode),
            wm_rollout_mode=str(args.wm_rollout_mode),
            wm_scoring_latent=str(args.wm_scoring_latent),
            wm_goal_flip_horizontal=bool(args.wm_goal_hflip),
            wm_sim_camera_parity=bool(args.wm_sim_camera_parity),
            wm_sim_img_size=int(args.wm_sim_img_size),
            smolvla_policy_hflip_corner2=bool(args.smolvla_policy_hflip_corner2),
            smolvla_noise_std=0.0,
            smolvla_n_action_steps=1,
            comparison_strip_overlay=bool(args.comparison_strip_overlay),
            comparison_strip_stitch_gutter_pixels=int(args.comparison_strip_stitch_gutter_pixels),
            oracle_action_sequence=act_seq.actions,
            oracle_action_source=str(act_seq.action_source_path),
        )

        _write_json(episode_output, episode_log.to_dict())
        episode_summaries.append(
            {
                "episode_index": int(loop_idx),
                "target_episode_index": int(ep_idx),
                "target_reset_seed": int(seed),
                "effective_max_steps": int(cap),
                "oracle_action_source": str(act_seq.action_source_path),
                "output_json": str(episode_output),
                "goal_frame_index": episode_log.goal_frame_index,
                "goal_source": episode_log.goal_source,
                "start_frame_similarity": episode_log.start_frame_similarity,
                "reset_frame_warning": bool(episode_log.reset_frame_warning),
                "steps": int(episode_log.steps),
                "done": bool(episode_log.done),
                "selected_indices": list(episode_log.selected_indices),
                "selected_candidate_indices": list(episode_log.selected_candidate_indices),
                "latent_scores": list(episode_log.latent_scores),
                "selected_scores": list(episode_log.selected_scores),
                "comparison_strip_path": episode_log.comparison_strip_path,
                "comparison_video_path": episode_log.comparison_video_path,
            }
        )

    manifest_out = run_dir / "segment_grpo_manifest.json"
    _write_json(
        manifest_out,
        {
            "task": args.task,
            "episodes": len(episode_rows),
            "chunk_len": int(args.chunk_len),
            "num_candidates": 1,
            "max_steps": int(args.max_steps),
            "carry_mode": "sim",
            "dry_run": bool(args.dry_run),
            "reset_frame_warning_threshold": float(args.reset_frame_warning_threshold),
            "oracle_run_root": str(oracle_run),
            "goal_frame_index": int(args.goal_frame_index),
            "wm_rollout_mode": str(args.wm_rollout_mode),
            "wm_scoring_latent": str(args.wm_scoring_latent),
            "strict_wm_scoring": bool(args.strict_wm_scoring),
            "strict_decode": bool(args.strict_decode),
            "oracle_action_mode": True,
            "episodes_info": episode_summaries,
            "run_dir": str(run_dir),
            "jepa_repo": jepa_repo_resolved,
            "jepa_ckpt": str(args.jepa_ckpt),
            "wm_device": wm_device_record,
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
        },
    )
    print(f"[phase9] wrote manifest: {manifest_out}")
    print(f"[phase9] wrote {len(episode_summaries)} episode JSON(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
