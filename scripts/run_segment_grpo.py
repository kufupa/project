#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import os
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
from segment_grpo_reference import (  # noqa: E402
    TopEpisode,
    load_oracle_reference_frames,
    parse_top15_report,
    resolve_latest_oracle_pushv3_run,
)
from smolvla_pipeline.run_layout import ensure_unique_run_dir  # noqa: E402

# Jepa-wms / WM–oracle pixel contract: keep these True unless you are deliberately A/B testing.
WM_PARITY_GOAL_HFLIP_DEFAULT = True
WM_PARITY_SIM_CAMERA_DEFAULT = True
WM_PARITY_SIM_IMG_SIZE_DEFAULT = 224
SMOLVLA_POLICY_HFLIP_DEFAULT = True
SMOLVLA_NOISE_STD_DEFAULT = 0.0
COMPARISON_STRIP_OVERLAY_DEFAULT = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment-level GRPO loop for push-v3 style tasks.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to sample.")
    parser.add_argument("--chunk-len", type=int, default=8, help="Action chunk length A.")
    parser.add_argument("--num-candidates", type=int, default=2, help="Number of chunk candidates K.")
    parser.add_argument("--max-steps", type=int, default=200, help="Per-episode max environment steps.")
    parser.add_argument("--task", type=str, default="push-v3", help="Task name.")
    parser.add_argument("--artifacts-root", type=Path, default=ROOT / "artifacts", help="Oracle artifact root directory.")
    parser.add_argument(
        "--oracle-run-root",
        type=Path,
        default=None,
        help="Optional explicit push-v3 oracle run directory; defaults to latest under phase06_oracle_baseline.",
    )
    parser.add_argument(
        "--top15-report",
        type=Path,
        default=ROOT
        / "docs"
        / "superpowers"
        / "reports"
        / "2026-04-11-pushv3-oracle-top15-smolvla-setup.md",
        help="Top-15 pilot table used to select episode/seed.",
    )
    parser.add_argument("--episode-plan", type=int, default=1, help="Top-k rank to execute (1-based).")
    parser.add_argument("--episode-index", type=int, default=None, help="Explicit target episode index override.")
    parser.add_argument("--reset-seed", type=int, default=None, help="Explicit reset seed override.")
    parser.add_argument("--goal-frame-index", type=int, default=25, help="1-based oracle goal frame index (default 25).")
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
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output JSON artifact path (auto-nested in run dir when writing to artifacts root).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "phase08_segment_grpo_baseline",
        help="Parent directory for auto-created run_* Segment-GRPO output folders.",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Disable auto run-directory nesting and write directly to --output-json.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--dry-run", action="store_true", help="Enable synthetic fallback for missing models/paths.")
    parser.add_argument("--train-steps", type=int, default=0, help="Optional adapter training steps per segment.")
    parser.add_argument(
        "--reset-frame-warning-threshold",
        type=float,
        default=0.08,
        help="Reset-frame similarity warning threshold (default 0.08).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device for model loading (auto/cpu/cuda).",
    )
    parser.add_argument(
        "--wm-rollout-mode",
        type=str,
        choices=["iterative", "batched"],
        default="iterative",
        help="WM latent rollout for chunk scoring: iterative (one unroll per action) or batched (single unroll).",
    )
    parser.add_argument(
        "--wm-scoring-latent",
        type=str,
        choices=["visual", "proprio", "concat"],
        default="visual",
        help="WM latent modality used for distance scoring: visual, proprio, or concat.",
    )
    parser.add_argument(
        "--strict-wm-scoring",
        action="store_true",
        help="Fail rollout when WM scoring raises; otherwise fall back with metadata.",
    )
    parser.add_argument(
        "--strict-decode",
        action="store_true",
        help="Fail rollout when decode reconstruction fails; otherwise record failure metadata.",
    )
    parser.add_argument(
        "--wm-goal-hflip",
        default=WM_PARITY_GOAL_HFLIP_DEFAULT,
        action=argparse.BooleanOptionalAction,
        help="H-flip oracle/stored goal pixels before WM encode (default: on; --no-wm-goal-hflip to disable).",
    )
    parser.add_argument(
        "--wm-sim-camera-parity",
        default=WM_PARITY_SIM_CAMERA_DEFAULT,
        action=argparse.BooleanOptionalAction,
        help="Jepa-wms corner2 + V-flip sim RGB for policy + WM + strips (default: on; --no-wm-sim-camera-parity to disable).",
    )
    parser.add_argument(
        "--wm-sim-img-size",
        type=int,
        default=WM_PARITY_SIM_IMG_SIZE_DEFAULT,
        help=f"Square render size for WM sim camera parity (default {WM_PARITY_SIM_IMG_SIZE_DEFAULT}).",
    )
    parser.add_argument(
        "--smolvla-policy-hflip-corner2",
        default=SMOLVLA_POLICY_HFLIP_DEFAULT,
        action=argparse.BooleanOptionalAction,
        help=(
            "When WM sim camera parity is on, H-flip jepa V-only RGB before SmolVLA so pixels match "
            "best_video corner2 (V+H) contract (default: on; --no-smolvla-policy-hflip-corner2 to disable)."
        ),
    )
    parser.add_argument(
        "--smolvla-noise-std",
        type=float,
        default=SMOLVLA_NOISE_STD_DEFAULT,
        help="Std dev of Gaussian noise added per chunk timestep to SmolVLA base action (default 0 = deterministic).",
    )
    parser.add_argument(
        "--comparison-strip-overlay",
        default=COMPARISON_STRIP_OVERLAY_DEFAULT,
        action=argparse.BooleanOptionalAction,
        help=(
            "Draw WM decode metadata box on bottom panel of each comparison-strip column "
            f"(default: {COMPARISON_STRIP_OVERLAY_DEFAULT})."
        ),
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


def _resolve_output_filename(output_json: Path, task: str) -> str:
    if output_json.suffix == ".json":
        return output_json.name
    stem = output_json.name.strip() or f"segment_grpo_{task.replace('-', '_')}"
    return f"{stem}.json"


def _has_run_component(path: Path) -> bool:
    return any(part.startswith("run_") for part in path.parts)


def _prepare_output_json(args: argparse.Namespace) -> tuple[Path, Path | None]:
    output_json = args.output_json.expanduser()
    if args.flat_output:
        return output_json, None

    artifacts_root = args.artifacts_root.expanduser().resolve(strict=False)
    output_parent = output_json.parent.expanduser().resolve(strict=False)
    if _has_run_component(output_parent):
        return output_json, None
    if output_parent != artifacts_root:
        return output_json, None

    output_root = args.output_root.expanduser().resolve(strict=False)
    run_dir = ensure_unique_run_dir(
        output_root,
        episodes=int(args.episodes),
        task=args.task,
        seed=int(args.seed),
        variant="segment_grpo",
    )
    output_file = run_dir / _resolve_output_filename(output_json, args.task)
    return output_file, run_dir


def _load_top15_rows(path: Path) -> list[TopEpisode]:
    if not path:
        return []
    if not path.exists():
        return []
    try:
        return parse_top15_report(path)
    except Exception:
        return []


def _resolve_oracle_plan(
    episode_offset: int,
    *,
    base_seed: int,
    explicit_episode: int | None,
    explicit_seed: int | None,
    top15_rows: list[TopEpisode],
    episode_plan: int,
) -> tuple[int, int]:
    if explicit_episode is not None:
        return explicit_episode, explicit_seed if explicit_seed is not None else base_seed + episode_offset * 997
    if explicit_seed is not None and explicit_episode is None:
        return episode_offset, explicit_seed
    if top15_rows:
        idx = int(max(1, episode_plan)) - 1
        base_idx = min(max(0, idx), len(top15_rows) - 1)
        row_idx = min(base_idx + episode_offset, len(top15_rows) - 1)
        row = top15_rows[row_idx]
        return row.episode_index, row.reset_seed
    return episode_offset, base_seed + episode_offset * 997


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    device = args.device
    output_json_base, run_dir = _prepare_output_json(args)
    if run_dir is not None:
        print(f"[segment_grpo] run directory: {run_dir}")
    print(
        "[segment_grpo] wm flags: "
        f"wm_goal_hflip={args.wm_goal_hflip} "
        f"wm_sim_camera_parity={args.wm_sim_camera_parity} "
        f"wm_sim_img_size={args.wm_sim_img_size} "
        f"smolvla_policy_hflip_corner2={args.smolvla_policy_hflip_corner2} "
        f"smolvla_noise_std={args.smolvla_noise_std} "
        f"comparison_strip_overlay={args.comparison_strip_overlay}"
    )

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

    oracle_rows = _load_top15_rows(args.top15_report)
    try:
        oracle_run = (
            args.oracle_run_root
            if args.oracle_run_root is not None
            else resolve_latest_oracle_pushv3_run(args.artifacts_root, task=args.task)
        )
    except Exception as exc:
        if args.dry_run:
            print(f"[segment_grpo] Oracle run discovery skipped in dry-run mode: {exc}")
            oracle_run = None
        else:
            raise

    adapter = None
    episode_summaries: list[dict[str, object]] = []
    for episode_idx in range(int(args.episodes)):
        target_episode, target_seed = _resolve_oracle_plan(
            episode_idx,
            base_seed=int(args.seed),
            explicit_episode=args.episode_index,
            explicit_seed=args.reset_seed,
            top15_rows=oracle_rows,
            episode_plan=int(args.episode_plan),
        )
        if target_episode != episode_idx and args.episodes == 1:
            print(f"[segment_grpo] overriding episode index from {episode_idx} to top-k rank target {target_episode}")

        goal_reference = None
        if oracle_run is not None:
            try:
                goal_reference = load_oracle_reference_frames(
                    oracle_run,
                    episode_index=target_episode,
                    goal_frame_index=int(args.goal_frame_index),
                    start_frame_index=0,
                )
            except Exception as exc:
                if args.dry_run:
                    print(f"[segment_grpo] oracle frame resolution failed, proceeding without oracle goal image: {exc}")
                else:
                    raise
        if goal_reference is not None:
            print(
                f"[segment_grpo] episode={target_episode} goal_frame={goal_reference.goal_frame_path} "
                f"start_frame={goal_reference.start_frame_path} reset_seed={target_seed}"
            )
        else:
            print(f"[segment_grpo] episode={target_episode} using fallback goal source (no oracle frame)")

        episode_output = _resolve_output_path(output_json_base, episode_idx, int(args.episodes))
        artifact_dir = episode_output.parent / f"{episode_output.stem}_artifacts"
        comparison_root = artifact_dir / "comparison"

        episode_log, adapter = rollout_with_chunks(
            smolvla_bundle,
            wm_bundle,
            task=args.task,
            episode_index=target_episode,
            chunk_len=int(args.chunk_len),
            num_candidates=int(args.num_candidates),
            max_steps=int(args.max_steps),
            carry_mode=args.carry_mode,
            replay_root=args.replay_root,
            goal_latent_source=args.goal_latent_source,
            goal_frame=(goal_reference.goal_frame if goal_reference is not None else None),
            start_frame=(goal_reference.start_frame if goal_reference is not None else None),
            goal_frame_index=int(args.goal_frame_index),
            goal_source=(str(goal_reference.goal_frame_path) if goal_reference is not None else None),
            comparison_root=comparison_root,
            reset_frame_warning_threshold=float(args.reset_frame_warning_threshold),
            seed=target_seed,
            train_steps=int(args.train_steps),
            dry_run=args.dry_run,
            adapter=adapter,
            strict_wm_scoring=bool(args.strict_wm_scoring),
            strict_decode=bool(args.strict_decode),
            wm_rollout_mode=str(args.wm_rollout_mode),
            wm_scoring_latent=str(args.wm_scoring_latent),
            wm_goal_flip_horizontal=bool(args.wm_goal_hflip),
            wm_sim_camera_parity=bool(args.wm_sim_camera_parity),
            wm_sim_img_size=int(args.wm_sim_img_size),
            smolvla_policy_hflip_corner2=bool(args.smolvla_policy_hflip_corner2),
            smolvla_noise_std=float(args.smolvla_noise_std),
            comparison_strip_overlay=bool(args.comparison_strip_overlay),
        )
        episode_path = _resolve_output_path(output_json_base, episode_idx, int(args.episodes))
        _write_json(episode_path, episode_log.to_dict())
        episode_summaries.append(
            {
                "episode_index": int(episode_idx),
                "target_episode_index": int(target_episode),
                "target_reset_seed": int(target_seed),
                "output_json": str(episode_path),
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

    if int(args.episodes) > 1:
        if output_json_base.suffix == ".json":
            manifest_dir = output_json_base.parent
        else:
            manifest_dir = output_json_base if output_json_base.is_dir() else output_json_base.parent
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
                "reset_frame_warning_threshold": float(args.reset_frame_warning_threshold),
                "oracle_run_root": str(oracle_run) if oracle_run is not None else None,
                "top15_report": str(args.top15_report),
                "episode_plan": int(args.episode_plan),
                "output_json": str(output_json_base),
                "goal_frame_index": int(args.goal_frame_index),
                "wm_rollout_mode": str(args.wm_rollout_mode),
                "wm_scoring_latent": str(args.wm_scoring_latent),
                "strict_wm_scoring": bool(args.strict_wm_scoring),
                "strict_decode": bool(args.strict_decode),
                "episodes_info": episode_summaries,
                "run_dir": str(run_dir) if run_dir is not None else None,
            },
        )
        print(f"[segment_grpo] wrote manifest: {manifest_path}")

    if episode_summaries:
        print(f"[segment_grpo] wrote episode artifacts for {len(episode_summaries)} episode(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
