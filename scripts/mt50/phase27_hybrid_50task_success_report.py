#!/usr/bin/env python3
"""Combine Phase27 baseline official eval_info (recovery tasks) + video-derived success for legacy layout.

- **official_eval_info**: rows from merged ``eval_info.json`` (LeRobot ``metrics.successes``).
- **video_duration_recovery**: ``count_successes_from_videos`` on ``shard_*_*tasks/videos`` only
  (excludes ``shard_recovery_*`` top-level videos dirs — recovery lives under per-task subdirs).

Writes JSON + Markdown under ``--parent`` with per-task ``source`` and provenance notes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Match count_successes_from_videos / paper-style MT50 buckets (+ unclassified for map gaps).
DIFFICULTY_ORDER = ("easy", "medium", "hard", "very_hard", "unclassified")
# MT50-style macro: mean of these four bucket-level success rates (not mean over 50 tasks).
DIFFICULTY_MACRO_BUCKETS = ("easy", "medium", "hard", "very_hard")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    # Required so dataclasses (and similar) can resolve the module while the class body runs.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_phase27_tasks(script_dir: Path) -> tuple[str, ...]:
    cleanup = _load_module(
        "cleanup_incomplete_phase27_baseline_videos",
        script_dir / "cleanup_incomplete_phase27_baseline_videos.py",
    )
    return tuple(cleanup.PHASE27_MT50_TASKS)


def _load_difficulty_map(path: Path) -> tuple[dict[str, str], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("task_difficulties") or {}), str(data.get("default") or "unclassified")


def _official_by_task(eval_path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("per_task", []) or []:
        name = row.get("task_group")
        if isinstance(name, str) and name:
            out[name] = row
    return out


def _official_success_metrics(row: dict[str, Any]) -> tuple[int, int, float, list[bool]]:
    metrics = row.get("metrics") or {}
    succ = metrics.get("successes")
    if not isinstance(succ, list) or not succ:
        return 0, 0, float("nan"), []
    n = len(succ)
    boo = [bool(x) for x in succ]
    hits = sum(1 for x in boo if x)
    pc = 100.0 * hits / n if n else float("nan")
    return hits, n, pc, boo


def _discover_legacy_videos_dirs(parent: Path) -> list[Path]:
    """Only ``shard_<n>_<m>tasks/videos``, not ``shard_recovery_*``."""
    import re

    pat = re.compile(r"^shard_\d+_\d+tasks$")
    out: list[Path] = []
    for shard in sorted(parent.iterdir()):
        if not shard.is_dir() or not pat.match(shard.name):
            continue
        v = shard / "videos"
        if v.is_dir():
            out.append(v)
    return out


def _scan_legacy_videos(
    csvideos: Any,
    videos_dirs: list[Path],
    *,
    expected_episodes: int,
    failure_duration_s: float,
    epsilon_s: float,
    difficulty_map: Path,
) -> dict[str, Any]:
    by_task: dict[str, Any] = {}
    for vdir in videos_dirs:
        rows = csvideos.scan_videos_dir(
            vdir,
            expected_episodes=expected_episodes,
            failure_duration_s=failure_duration_s,
            epsilon_s=epsilon_s,
            difficulty_map_path=difficulty_map,
        )
        for row in rows:
            by_task[row.task] = row
    return by_task


def _video_success_metrics(row: Any) -> tuple[int | None, int, float | None, str]:
    status = row.status
    if status == "complete":
        return int(row.successes), int(row.expected_episodes), float(row.pc_success), status
    return None, int(row.expected_episodes), None, status


def _build_rows(
    tasks: tuple[str, ...],
    official: dict[str, dict[str, Any]],
    video_by_task: dict[str, Any],
    *,
    expected_episodes: int,
    difficulty_by_task: dict[str, str],
    default_difficulty: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        difficulty = difficulty_by_task.get(task, default_difficulty)
        if task in official:
            hits, n, pc, _boo = _official_success_metrics(official[task])
            st = "complete" if n == expected_episodes and n > 0 else "partial"
            rows.append(
                {
                    "task": task,
                    "difficulty": difficulty,
                    "source": "official_eval_info",
                    "successes": hits,
                    "episodes": n,
                    "pc_success": pc if math.isfinite(pc) else None,
                    "status": st,
                    "detail": {"note": "merged eval_info.json per_task.metrics.successes"},
                }
            )
            continue
        if task in video_by_task:
            vr = video_by_task[task]
            hits, n, pc, st = _video_success_metrics(vr)
            rows.append(
                {
                    "task": task,
                    "difficulty": difficulty,
                    "source": "video_duration_recovery",
                    "successes": hits,
                    "episodes": n,
                    "pc_success": pc,
                    "status": st,
                    "detail": {
                        "videos_subdir": str(vr.source_videos_dir),
                        "difficulty": vr.difficulty,
                        "episodes_found": vr.episodes_found,
                    },
                }
            )
            continue
        rows.append(
            {
                "task": task,
                "difficulty": difficulty,
                "source": "missing",
                "successes": None,
                "episodes": 0,
                "pc_success": None,
                "status": "missing",
                "detail": {},
            }
        )
    return rows


def _micro_global_episode_weighted(rows: list[dict[str, Any]]) -> tuple[float, int, int]:
    """Pooled successes / pooled episodes across all tasks with metrics."""
    with_m = [r for r in rows if r.get("successes") is not None and int(r.get("episodes", 0) or 0) > 0]
    if not with_m:
        return float("nan"), 0, 0
    succ = sum(int(r["successes"]) for r in with_m)  # type: ignore[arg-type]
    ep = sum(int(r["episodes"]) for r in with_m)
    return (100.0 * succ / ep) if ep else float("nan"), succ, ep


def _macro_mean_four_difficulty_buckets(by_difficulty: list[dict[str, Any]]) -> tuple[float, list[float]]:
    """Mean of easy/medium/hard/very_hard bucket % (each bucket is episode-weighted internally)."""
    dmap = {str(b["difficulty"]): b for b in by_difficulty}
    vals: list[float] = []
    for d in DIFFICULTY_MACRO_BUCKETS:
        if d not in dmap:
            return float("nan"), []
        mic = dmap[d].get("pc_success_micro_episode_weighted")
        if mic is None or not math.isfinite(float(mic)):
            return float("nan"), []
        vals.append(float(mic))
    return sum(vals) / 4.0, vals


def _breakdown_by_difficulty(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-category: episode-weighted % within bucket; per-bucket mean task % matches when episodes/task constant."""
    by_d: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        d = str(r.get("difficulty") or "unclassified")
        by_d[d].append(r)

    out: list[dict[str, Any]] = []
    for d in DIFFICULTY_ORDER:
        if d not in by_d:
            continue
        gr = by_d[d]
        with_metrics = [r for r in gr if r.get("successes") is not None and r.get("episodes", 0) > 0]
        succ_sum = sum(int(r["successes"]) for r in with_metrics)  # type: ignore[arg-type]
        ep_sum = sum(int(r["episodes"]) for r in with_metrics)
        micro = (100.0 * succ_sum / ep_sum) if ep_sum else float("nan")
        pcs = [
            float(r["pc_success"])
            for r in gr
            if r.get("pc_success") is not None and math.isfinite(float(r["pc_success"]))
        ]
        macro = (sum(pcs) / len(pcs)) if pcs else float("nan")
        out.append(
            {
                "difficulty": d,
                "n_tasks": len(gr),
                "n_official_eval_info": sum(1 for r in gr if r["source"] == "official_eval_info"),
                "n_video_duration_recovery": sum(1 for r in gr if r["source"] == "video_duration_recovery"),
                "n_missing": sum(1 for r in gr if r["source"] == "missing"),
                "successes_pooled": succ_sum,
                "episodes_pooled": ep_sum,
                "pc_success_micro_episode_weighted": micro if math.isfinite(micro) else None,
                "pc_success_macro_task_mean": macro if math.isfinite(macro) else None,
            }
        )
    return out


def _breakdown_by_source(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_s: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_s[str(r["source"])].append(r)
    order = ("official_eval_info", "video_duration_recovery", "missing")
    out: list[dict[str, Any]] = []
    for src in order:
        if src not in by_s:
            continue
        gr = by_s[src]
        with_metrics = [r for r in gr if r.get("successes") is not None and r.get("episodes", 0) > 0]
        succ_sum = sum(int(r["successes"]) for r in with_metrics) if with_metrics else 0
        ep_sum = sum(int(r["episodes"]) for r in with_metrics)
        micro = (100.0 * succ_sum / ep_sum) if ep_sum else None
        pcs = [
            float(r["pc_success"])
            for r in gr
            if r.get("pc_success") is not None and math.isfinite(float(r["pc_success"]))
        ]
        macro = (sum(pcs) / len(pcs)) if pcs else float("nan")
        out.append(
            {
                "source": src,
                "n_tasks": len(gr),
                "successes_pooled": succ_sum if with_metrics else None,
                "episodes_pooled": ep_sum if with_metrics else 0,
                "pc_success_micro_episode_weighted": micro,
                "pc_success_macro_task_mean": macro if math.isfinite(macro) else None,
            }
        )
    return out


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase27 hybrid 50-task success",
        "",
        f"- **Parent**: `{payload['parent']}`",
        f"- **Eval info**: `{payload['eval_info_path']}`",
        f"- **Difficulty map**: `{payload.get('difficulty_map', '')}`",
        f"- **Video method**: duration heuristic (failures ≈ {payload['failure_duration_s']}s horizon)",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|------:|",
    ]
    s = payload["summary"]
    lines.append(f"| Official eval_info tasks | {s['official_count']} |")
    lines.append(f"| Video-recovery tasks | {s['video_complete_count']} |")
    lines.append(f"| Missing / broken | {s['missing_count']} |")
    micro_g = s.get("pc_success_micro_global_episode_weighted")
    micro_s = f"{micro_g:.2f}" if isinstance(micro_g, (int, float)) and math.isfinite(micro_g) else "nan"
    lines.append(f"| **Micro** (episode-weighted, all tasks) | {micro_s} |")
    macro4 = s.get("pc_success_macro_mean_four_difficulty_buckets")
    macro_s = f"{macro4:.2f}" if isinstance(macro4, (int, float)) and math.isfinite(macro4) else "nan"
    lines.append(
        f"| **Macro** (mean of {len(DIFFICULTY_MACRO_BUCKETS)} difficulty-bucket %) | {macro_s} |"
    )
    lines.append("")
    lines.append("## By difficulty (category)")
    lines.append("")
    lines.append("Episode-weighted % = pooled successes / pooled episodes across tasks in that bucket.")
    lines.append("")
    lines.append(
        "| Difficulty | Tasks | Official | Video | Missing | Successes | Episodes | % (micro) | % (macro task-mean) |"
    )
    lines.append("|------------|------:|---------:|------:|--------:|----------:|---------:|----------:|--------------------:|")
    for row in payload.get("by_difficulty", []):
        d = row["difficulty"]
        micro = row.get("pc_success_micro_episode_weighted")
        macro = row.get("pc_success_macro_task_mean")
        micro_s = f"{float(micro):.2f}" if micro is not None and math.isfinite(float(micro)) else ""
        macro_s = f"{float(macro):.2f}" if macro is not None and math.isfinite(float(macro)) else ""
        lines.append(
            f"| {d} | {row['n_tasks']} | {row['n_official_eval_info']} | "
            f"{row['n_video_duration_recovery']} | {row['n_missing']} | "
            f"{row['successes_pooled']} | {row['episodes_pooled']} | {micro_s} | {macro_s} |"
        )
    lines.append("")
    lines.append("## By data source")
    lines.append("")
    lines.append("| Source | Tasks | Successes (pooled) | Episodes (pooled) | % (micro) | % (macro task-mean) |")
    lines.append("|--------|------:|---------------------:|------------------:|----------:|--------------------:|")
    for row in payload.get("by_source", []):
        src = row["source"]
        sp = row.get("successes_pooled")
        ep = row.get("episodes_pooled", 0)
        micro = row.get("pc_success_micro_episode_weighted")
        macro = row.get("pc_success_macro_task_mean")
        sp_s = "" if sp is None else str(int(sp))
        micro_s = f"{float(micro):.2f}" if micro is not None and math.isfinite(float(micro)) else ""
        macro_s = f"{float(macro):.2f}" if macro is not None and math.isfinite(float(macro)) else ""
        lines.append(f"| {src} | {row['n_tasks']} | {sp_s} | {ep} | {micro_s} | {macro_s} |")
    lines.append("")
    lines.append("## Per task")
    lines.append("")
    lines.append("| Task | Category | Source | Successes | Episodes | % | Status |")
    lines.append("|------|----------|--------|----------:|---------:|---:|--------|")
    for r in payload["tasks"]:
        src = r["source"]
        cat = r.get("difficulty", "")
        succ = r["successes"]
        succ_s = "" if succ is None else str(succ)
        ep = r["episodes"]
        pc = r["pc_success"]
        pc_s = "" if pc is None else f"{float(pc):.2f}"
        st = r["status"]
        lines.append(f"| `{r['task']}` | {cat} | {src} | {succ_s} | {ep} | {pc_s} | {st} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    default_parent = project_root / "artifacts" / "MT50_Phase27_smolvla_baseline_official_lerobot_25ep_s1000_4gpu_rtx6000"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", type=Path, default=default_parent, help="Phase27 baseline artifact root")
    ap.add_argument(
        "--eval-info",
        type=Path,
        default=None,
        help="Merged eval_info.json (default: <parent>/eval_info.json)",
    )
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--failure-duration", type=float, default=6.25)
    ap.add_argument("--epsilon", type=float, default=0.001)
    ap.add_argument(
        "--difficulty-map",
        type=Path,
        default=project_root / "scripts" / "mt50" / "mt50_phase07_task_difficulties.json",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--md-out", type=Path, default=None)
    args = ap.parse_args()

    parent = args.parent.resolve()
    eval_path = (args.eval_info or (parent / "eval_info.json")).resolve()
    if not eval_path.is_file():
        raise SystemExit(f"missing eval_info: {eval_path}")

    tasks = _load_phase27_tasks(script_dir)
    official = _official_by_task(eval_path)

    csvideos = _load_module("count_successes_from_videos", script_dir / "count_successes_from_videos.py")
    videos_dirs = _discover_legacy_videos_dirs(parent)
    if not videos_dirs:
        raise SystemExit(f"no legacy shard_*_*tasks/videos under {parent}")
    diff_map_path = args.difficulty_map.resolve()
    difficulty_by_task, default_difficulty = _load_difficulty_map(diff_map_path)
    video_by_task = _scan_legacy_videos(
        csvideos,
        videos_dirs,
        expected_episodes=args.episodes,
        failure_duration_s=args.failure_duration,
        epsilon_s=args.epsilon,
        difficulty_map=diff_map_path,
    )

    task_rows = _build_rows(
        tasks,
        official,
        video_by_task,
        expected_episodes=args.episodes,
        difficulty_by_task=difficulty_by_task,
        default_difficulty=default_difficulty,
    )

    official_count = sum(1 for r in task_rows if r["source"] == "official_eval_info")
    vid_complete = sum(1 for r in task_rows if r["source"] == "video_duration_recovery" and r["status"] == "complete")
    vid_broken = sum(1 for r in task_rows if r["source"] == "video_duration_recovery" and r["status"] != "complete")
    missing_count = sum(1 for r in task_rows if r["source"] == "missing")

    by_difficulty = _breakdown_by_difficulty(task_rows)
    by_source = _breakdown_by_source(task_rows)
    micro_global, succ_pool, ep_pool = _micro_global_episode_weighted(task_rows)
    macro_four, bucket_pcs = _macro_mean_four_difficulty_buckets(by_difficulty)

    payload: dict[str, Any] = {
        "parent": str(parent),
        "eval_info_path": str(eval_path),
        "difficulty_map": str(diff_map_path),
        "legacy_videos_dirs": [str(p) for p in videos_dirs],
        "failure_duration_s": args.failure_duration,
        "epsilon_s": args.epsilon,
        "expected_episodes": args.episodes,
        "provenance": (
            "Hybrid report: official LeRobot successes from eval_info.json where present; "
            "remaining tasks from MP4 duration heuristic (see count_successes_from_videos.py). "
            "Rewards / sum_rewards not recovered from video."
        ),
        "by_difficulty": by_difficulty,
        "by_source": by_source,
        "tasks": task_rows,
        "summary": {
            "official_count": official_count,
            "video_complete_count": vid_complete,
            "video_broken_count": vid_broken,
            "missing_count": missing_count,
            # Episode-weighted over all tasks (equals mean of per-task % when every task has same episode count).
            "pc_success_micro_global_episode_weighted": micro_global,
            "successes_pooled_global": succ_pool,
            "episodes_pooled_global": ep_pool,
            # MT50-style macro: average of easy/medium/hard/very_hard bucket rates.
            "pc_success_macro_mean_four_difficulty_buckets": macro_four,
            "macro_four_bucket_pcs": bucket_pcs,
            "macro_four_bucket_labels": list(DIFFICULTY_MACRO_BUCKETS),
        },
    }

    json_out = (args.json_out or (parent / "phase27_hybrid_50task_success.json")).resolve()
    md_out = (args.md_out or (parent / "phase27_hybrid_50task_success.md")).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_md(md_out, payload)
    print(f"wrote {json_out}")
    print(f"wrote {md_out}")
    print(
        f"summary: official={official_count} video_complete={vid_complete} "
        f"video_broken={vid_broken} missing={missing_count}"
    )


if __name__ == "__main__":
    main()
