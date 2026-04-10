#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SmolVLA run artifacts for preflight and smoke gating."
    )
    parser.add_argument("--run-dir", required=True, help="Absolute or relative run directory path.")
    parser.add_argument("--task", default="push-v3")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--require-video", default="true")
    parser.add_argument(
        "--require-frames",
        default="false",
        help="Require frames_dir with frame_*.png per episode (matches save_frames runs).",
    )
    parser.add_argument("--min-video-bytes", type=int, default=1024)
    return parser.parse_args()


def _as_bool(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw!r}")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        _fail(f"run dir does not exist: {run_dir}")

    eval_info_path = run_dir / "eval_info.json"
    run_manifest_path = run_dir / "run_manifest.json"
    if not eval_info_path.is_file():
        _fail(f"missing eval_info.json at {eval_info_path}")
    if not run_manifest_path.is_file():
        _fail(f"missing run_manifest.json at {run_manifest_path}")

    eval_info = _read_json(eval_info_path)
    run_manifest = _read_json(run_manifest_path)
    overall = eval_info.get("overall", {})

    expected_episodes = int(args.episodes)
    actual_episodes = int(overall.get("n_episodes", 0))
    if actual_episodes != expected_episodes:
        _fail(f"expected n_episodes={expected_episodes}, got {actual_episodes}")

    if run_manifest.get("task") != args.task:
        _fail(f"expected task={args.task!r}, got {run_manifest.get('task')!r}")
    if run_manifest.get("runtime_backend") != "lerobot_metaworld":
        _fail(
            f"unexpected runtime_backend={run_manifest.get('runtime_backend')!r}; "
            "expected 'lerobot_metaworld'"
        )
    max_steps = int(run_manifest.get("max_steps", 0))
    if max_steps < 1:
        _fail(f"run_manifest max_steps must be >= 1, got {max_steps}")
    camera_name = run_manifest.get("camera_name")
    if not isinstance(camera_name, str) or not camera_name.strip():
        _fail("run_manifest.camera_name missing or invalid")
    if not isinstance(run_manifest.get("flip_corner2"), bool):
        _fail("run_manifest.flip_corner2 missing or not boolean")
    episodes = run_manifest.get("episodes", [])
    if not isinstance(episodes, list) or len(episodes) != expected_episodes:
        _fail(
            f"run_manifest episodes must contain exactly {expected_episodes} rows; "
            f"got {len(episodes) if isinstance(episodes, list) else 'non-list'}"
        )

    require_video = _as_bool(args.require_video)
    require_frames = _as_bool(args.require_frames)
    min_video_bytes = int(args.min_video_bytes)

    if require_frames:
        if run_manifest.get("save_frames") is not True:
            _fail("run_manifest.save_frames must be true when --require-frames true")
    video_paths = overall.get("video_paths", [])
    if not isinstance(video_paths, list):
        _fail("eval_info.overall.video_paths must be a list")
    cleaned_video_paths = [path for path in video_paths if isinstance(path, str) and path.strip()]
    if require_video and len(cleaned_video_paths) != expected_episodes:
        _fail(
            f"video path count must equal episodes when require_video=true "
            f"(expected={expected_episodes}, got={len(cleaned_video_paths)})"
        )

    checked_videos = 0
    checked_frame_dirs = 0
    for row in episodes:
        if not isinstance(row, dict):
            _fail("run_manifest episodes row is not an object")
        paths = row.get("paths", {})
        if not isinstance(paths, dict):
            _fail("run_manifest episode.paths must be an object")

        for key in ("actions", "reward_curve_csv", "reward_curve_png"):
            rel = paths.get(key)
            if not isinstance(rel, str) or not rel.strip():
                _fail(f"episode path {key!r} missing or invalid")
            artifact_path = run_dir / rel
            if not artifact_path.is_file():
                _fail(f"artifact missing: {artifact_path}")

        actions_rel = paths["actions"]
        action_lines = [
            line
            for line in (run_dir / actions_rel).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not action_lines:
            _fail(f"actions.jsonl is empty: {run_dir / actions_rel}")

        video_rel = paths.get("video")
        if require_video:
            if not isinstance(video_rel, str) or not video_rel.strip():
                _fail("episode video path missing while require_video=true")
            video_path = run_dir / video_rel
            if not video_path.is_file():
                _fail(f"video missing: {video_path}")
            if video_path.stat().st_size < min_video_bytes:
                _fail(
                    f"video too small ({video_path.stat().st_size} bytes) "
                    f"for {video_path}; minimum is {min_video_bytes}"
                )
            checked_videos += 1

        if require_frames:
            rel = paths.get("frames_dir")
            if not isinstance(rel, str) or not rel.strip():
                _fail("episode paths.frames_dir missing while require_frames=true")
            fdir = run_dir / rel
            if not fdir.is_dir():
                _fail(f"frames_dir not a directory: {fdir}")
            pngs = sorted(fdir.glob("frame_*.png"))
            n_expect = int(row.get("n_frames", 0))
            if n_expect < 1:
                _fail("episode n_frames must be >= 1 when require_frames=true")
            if len(pngs) != n_expect:
                _fail(
                    f"expected {n_expect} frame PNGs in {fdir}, found {len(pngs)}"
                )
            checked_frame_dirs += 1

    payload = {
        "status": "ok",
        "run_dir": str(run_dir),
        "task": args.task,
        "episodes": expected_episodes,
        "checked_videos": checked_videos,
        "checked_frame_dirs": checked_frame_dirs,
        "pc_success": float(overall.get("pc_success", 0.0)),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
