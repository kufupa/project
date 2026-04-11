"""Single-process MT50 Phase07 SmolVLA baseline: load policy once, evaluate all tasks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import metaworld

from src.smolvla_pipeline.evaluator import (
    _load_smolvla_bundle,
    _LeRobotMetaWorldBackend,
    run_smolvla_eval,
)
from src.smolvla_pipeline.run_layout import ensure_unique_run_dir, slug_task


def _truthy_env(name: str, default: str = "false") -> bool:
    raw = (os.environ.get(name) or default).strip().lower()
    return raw in ("1", "true", "yes", "on")


def _load_difficulty_map(path: Path) -> tuple[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    default = (data.get("default") or "").strip().lower() or "unclassified"
    td = data.get("task_difficulties") or {}
    out: dict[str, str] = {}
    for k, v in td.items():
        out[str(k)] = str(v).strip().lower()
    return default, out


def _include_buckets(raw: str) -> set[str]:
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return set(parts) if parts else {"easy", "medium", "hard", "very_hard", "unclassified"}


def _run_dir_complete(run_dir: Path, episodes: int) -> bool:
    path = run_dir / "eval_info.json"
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        n = data.get("overall", {}).get("n_episodes")
        return int(n) == int(episodes)
    except Exception:
        return False


def _matching_run_dirs(
    out_bucket: Path, *, slug: str, difficulty: str, episodes: int, seed: int
) -> list[Path]:
    prefix = f"{slug}_{difficulty}_run_"
    out: list[Path] = []
    for p in out_bucket.iterdir():
        if not p.is_dir():
            continue
        n = p.name
        if not n.startswith(prefix):
            continue
        if (
            f"_ep{episodes}_" in n
            and f"_s{seed}_" in n
            and "_vsmolvla_parity_" in n
        ):
            out.append(p)
    out.sort(key=lambda x: x.stat().st_mtime)
    return out


def _write_index(
    *,
    rows: list[dict[str, Any]],
    index_path: Path,
    run_root: Path,
    episodes: int,
    seed: int,
    max_steps: int,
    checkpoint: str,
    default_difficulty: str,
    include_difficulties: str,
) -> None:
    from datetime import datetime, timezone

    status_counts: dict[str, int] = {"ok": 0, "failed": 0, "skipped": 0}
    for row in rows:
        st = row.get("status", "")
        if st in status_counts:
            status_counts[st] += 1

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "MT50_Phase07",
        "run_root": str(run_root),
        "episodes": int(episodes),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "checkpoint": checkpoint,
        "default_difficulty": default_difficulty,
        "difficulty_filter": [
            b.strip() for b in include_difficulties.split(",") if b.strip()
        ],
        "tasks": rows,
        "task_count": len(rows),
        "task_status": status_counts,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[mt50:phase07] wrote index: {index_path}", flush=True)
    print(
        f"[mt50:phase07] tasks={len(rows)} ok={status_counts.get('ok', 0)} "
        f"failed={status_counts.get('failed', 0)} skipped={status_counts.get('skipped', 0)}",
        flush=True,
    )


def _verify_run(
    *,
    project_root: Path,
    run_dir: Path,
    task: str,
    episodes: int,
    require_video: str,
    require_frames: str,
    require_actions: str,
    min_video_bytes: str,
) -> None:
    py = Path(os.environ.get("SMOLVLA_PYTHON_BIN", sys.executable))
    if not py.is_file():
        py = Path(sys.executable)
    cmd = [
        str(py),
        str(project_root / "scripts/smolvla/verify_smolvla_run_artifacts.py"),
        "--run-dir",
        str(run_dir),
        "--task",
        task,
        "--episodes",
        str(episodes),
        "--require-video",
        require_video,
        "--require-frames",
        require_frames,
        "--require-actions",
        require_actions,
        "--min-video-bytes",
        min_video_bytes,
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    run_root = Path(
        os.environ.get(
            "MT50_PHASE07_OUTPUT_ROOT",
            project_root / "artifacts" / "MT50_Phase07",
        )
    )
    script_dir = project_root / "scripts" / "mt50"
    diff_path = Path(
        os.environ.get(
            "MT50_TASK_DIFFICULTY_JSON",
            script_dir / "mt50_phase07_task_difficulties.json",
        )
    )
    if not diff_path.is_file():
        print(f"[mt50:phase07] ERROR: missing difficulty map: {diff_path}", flush=True)
        return 2

    include_raw = os.environ.get(
        "MT50_INCLUDE_DIFFICULTIES", "easy,medium,hard,very_hard,unclassified"
    )
    buckets = _include_buckets(include_raw)
    episodes = int(os.environ.get("MT50_PHASE07_EPISODES", "10"))
    seed = int(os.environ.get("MT50_PHASE07_SEED", "1000"))
    max_steps = int(os.environ.get("MT50_PHASE07_MAX_STEPS", "120"))
    checkpoint = os.environ.get(
        "MT50_PHASE07_CHECKPOINT",
        os.environ.get("SMOLVLA_INIT_CHECKPOINT", "jadechoghari/smolvla_metaworld"),
    ).strip()
    fps = int(os.environ.get("MT50_PHASE07_FPS", "30"))
    overlay = os.environ.get("MT50_PHASE07_OVERLAY_MODE", "reward_delta").strip()
    save_frames = _truthy_env("MT50_PHASE07_SAVE_FRAMES", "false")
    video = _truthy_env("MT50_PHASE07_VIDEO", "false")
    save_actions = _truthy_env("MT50_PHASE07_SAVE_ACTIONS", "false")
    task_text_global = os.environ.get("MT50_PHASE07_TASK_TEXT", "").strip() or None
    index_json = Path(
        os.environ.get("MT50_PHASE07_INDEX_JSON", run_root / "MT50_Phase07_index.json")
    )
    min_video = os.environ.get("MT50_MIN_VIDEO_BYTES", "1024")

    default_d, task_diff = _load_difficulty_map(diff_path)
    run_root.mkdir(parents=True, exist_ok=True)

    tasks_sorted = sorted(metaworld.MT50().train_classes.keys())
    print(
        f"[mt50:phase07] discovered {len(tasks_sorted)} tasks (campaign, shared bundle)",
        flush=True,
    )
    print(f"[mt50:phase07] output_root={run_root}", flush=True)
    print(f"[mt50:phase07] include_difficulties={include_raw}", flush=True)

    shared_bundle_holder: list[Any] = [None]

    def backend_factory(
        *,
        task: str,
        checkpoint: str,
        seed: int,
        max_steps: int,
        task_text: str | None = None,
        collect_frames: bool = True,
        **_kwargs: Any,
    ) -> Any:
        if shared_bundle_holder[0] is None:
            shared_bundle_holder[0] = _load_smolvla_bundle(checkpoint)
        return _LeRobotMetaWorldBackend(
            task=task,
            checkpoint=checkpoint,
            seed=seed,
            max_steps=max_steps,
            task_text=task_text,
            collect_frames=collect_frames,
            bundle=shared_bundle_holder[0],
        )

    resume = _truthy_env("MT50_PHASE07_RESUME", "true")
    purge_partial = _truthy_env("MT50_PHASE07_PURGE_PARTIAL", "true")
    t_pre = time.perf_counter()
    print("[mt50:phase07] campaign: preload_bundle_begin", flush=True)
    shared_bundle_holder[0] = _load_smolvla_bundle(checkpoint)
    print(
        "[mt50:phase07] campaign: preload_bundle_done "
        f"elapsed_s={time.perf_counter() - t_pre:.2f}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    require_video = "true" if video else "false"
    require_frames = "true" if save_frames else "false"
    require_actions = "true" if save_actions else "false"

    for task in tasks_sorted:
        difficulty = str(task_diff.get(task, default_d)).strip().lower()
        if difficulty not in buckets:
            print(
                f"[mt50:phase07] skip task={task} difficulty={difficulty}",
                flush=True,
            )
            rows.append(
                {
                    "task": task,
                    "difficulty": difficulty,
                    "run_dir": "",
                    "status": "skipped",
                    "pc_success": None,
                }
            )
            continue

        slug = slug_task(task)
        out_bucket = run_root / difficulty
        out_bucket.mkdir(parents=True, exist_ok=True)

        matching = _matching_run_dirs(
            out_bucket, slug=slug, difficulty=difficulty, episodes=episodes, seed=seed
        )
        complete_dirs = [p for p in matching if _run_dir_complete(p, episodes)]
        partial_dirs = [p for p in matching if p not in complete_dirs]

        if resume and complete_dirs:
            run_dir = max(complete_dirs, key=lambda p: p.stat().st_mtime)
            eval_info = json.loads((run_dir / "eval_info.json").read_text(encoding="utf-8"))
            pc = eval_info.get("overall", {}).get("pc_success")
            rows.append(
                {
                    "task": task,
                    "difficulty": difficulty,
                    "run_dir": str(run_dir),
                    "status": "ok",
                    "pc_success": float(pc) if pc is not None and pc != "" else None,
                }
            )
            print(
                f"[mt50:phase07] resume skip task={task} run_dir={run_dir} pc_success={pc}",
                flush=True,
            )
            continue

        if resume and purge_partial and partial_dirs:
            for p in partial_dirs:
                shutil.rmtree(p, ignore_errors=True)
                print(f"[mt50:phase07] purge_partial run_dir={p}", flush=True)

        run_dir = ensure_unique_run_dir(
            out_bucket,
            episodes=episodes,
            task=task,
            seed=seed,
            variant="smolvla_parity",
            run_name_prefix=f"{slug}_{difficulty}",
        )
        print(f"[mt50:phase07] run task={task} difficulty={difficulty} run_dir={run_dir}", flush=True)

        tt = task_text_global
        try:
            run_smolvla_eval(
                task=task,
                episodes=episodes,
                seed=seed,
                checkpoint=checkpoint,
                output_dir=run_dir,
                video=video,
                fps=fps,
                overlay_mode=overlay,
                max_steps=max_steps,
                save_frames=save_frames,
                save_actions=save_actions,
                task_text=tt,
                backend_factory=backend_factory,
            )
            _verify_run(
                project_root=project_root,
                run_dir=run_dir,
                task=task,
                episodes=episodes,
                require_video=require_video,
                require_frames=require_frames,
                require_actions=require_actions,
                min_video_bytes=min_video,
            )
            eval_info = json.loads((run_dir / "eval_info.json").read_text(encoding="utf-8"))
            pc = eval_info.get("overall", {}).get("pc_success")
            rows.append(
                {
                    "task": task,
                    "difficulty": difficulty,
                    "run_dir": str(run_dir),
                    "status": "ok",
                    "pc_success": float(pc) if pc is not None and pc != "" else None,
                }
            )
            print(f"[mt50:phase07] task ok task={task} pc_success={pc}", flush=True)
        except Exception as exc:
            print(f"[mt50:phase07] task failed task={task} err={exc!r}", flush=True)
            rows.append(
                {
                    "task": task,
                    "difficulty": difficulty,
                    "run_dir": str(run_dir),
                    "status": "failed",
                    "pc_success": None,
                }
            )

    _write_index(
        rows=rows,
        index_path=index_json,
        run_root=run_root,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
        checkpoint=checkpoint,
        default_difficulty=default_d,
        include_difficulties=include_raw,
    )
    failed = sum(1 for r in rows if r.get("status") == "failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
