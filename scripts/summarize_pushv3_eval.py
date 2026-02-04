#!/usr/bin/env python3
"""Build a compact summary of the best push-v3 evaluation episodes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize eval_info.json with top-k max-reward episodes."
    )
    parser.add_argument(
        "--eval-info",
        required=True,
        help="Path to eval_info.json from push-v3 baseline eval.",
    )
    parser.add_argument(
        "--task",
        default="push-v3",
        help="Task/group name used in the eval json (default: push-v3).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top episodes to keep (default: 5).",
    )
    parser.add_argument(
        "--output",
        default="optimal_report.json",
        help="Where to write the compact summary JSON.",
    )
    parser.add_argument(
        "--video-path",
        action="append",
        default=[],
        help=(
            "Optional override for episode video path in the format "
            "'INDEX:PATH' (can be repeated)."
        ),
    )
    return parser.parse_args()


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _parse_video_overrides(video_specs: Sequence[str]) -> Dict[int, str]:
    overrides: Dict[int, str] = {}
    for spec in video_specs:
        if ":" not in spec:
            raise ValueError(
                f"Invalid --video-path value '{spec}'. Expected INDEX:PATH."
            )
        index_raw, path = spec.split(":", 1)
        if not index_raw.isdigit():
            raise ValueError(
                f"Invalid --video-path episode index '{index_raw}'. Expected a non-negative int."
            )
        overrides[int(index_raw)] = path
    return overrides


def _find_task_payload(
    eval_data: Mapping[str, Any], requested_task: str
) -> Tuple[Mapping[str, Any], str, int]:
    per_task = eval_data.get("per_task")
    available_tasks: list[str] = []
    if isinstance(per_task, list):
        for entry in per_task:
            if not isinstance(entry, Mapping):
                continue
            task_group = str(entry.get("task_group", ""))
            if task_group:
                available_tasks.append(task_group)
            if task_group == requested_task:
                metrics = entry.get("metrics", {})
                if isinstance(metrics, Mapping):
                    return metrics, task_group, int(entry.get("task_id", 0))
    available = ", ".join(sorted(available_tasks)) if available_tasks else "<none>"
    raise ValueError(
        f"Task/group '{requested_task}' not found in eval_info.per_task. "
        f"Available tasks: {available}"
    )


def _coerce_episode_count(episodes: Any) -> Optional[int]:
    if isinstance(episodes, int):
        return episodes
    if isinstance(episodes, float) and episodes.is_integer():
        return int(episodes)
    return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _extract_episode_rows(
    task_metrics: Mapping[str, Any], video_overrides: Mapping[int, str]
) -> List[Dict[str, Any]]:
    max_rewards = task_metrics.get("max_rewards")
    if not isinstance(max_rewards, list):
        raise ValueError("Eval payload does not include a max_rewards list.")

    sum_rewards = task_metrics.get("sum_rewards", [])
    successes = task_metrics.get("successes", [])
    video_paths = task_metrics.get("video_paths", [])

    sum_rewards = sum_rewards if isinstance(sum_rewards, list) else []
    successes = successes if isinstance(successes, list) else []
    video_paths = video_paths if isinstance(video_paths, list) else []

    rows: List[Dict[str, Any]] = []
    for episode_idx, raw_max in enumerate(max_rewards):
        max_reward = _safe_float(raw_max)
        if max_reward is None:
            continue

        sum_reward = None
        if episode_idx < len(sum_rewards):
            sum_reward = _safe_float(sum_rewards[episode_idx])

        success = None
        if episode_idx < len(successes):
            success = _coerce_bool(successes[episode_idx])

        video_path = video_overrides.get(episode_idx)
        if video_path is None and episode_idx < len(video_paths):
            candidate = video_paths[episode_idx]
            if isinstance(candidate, str) and candidate.strip():
                video_path = candidate

        rows.append(
            {
                "episode_index": episode_idx,
                "max_reward": max_reward,
                "sum_reward": sum_reward,
                "success": success,
                "video_path": video_path,
            }
        )

    return rows


def _summary_block(
    eval_data: Mapping[str, Any],
    resolved_task: str,
    task_metrics: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    requested_top_k: int,
) -> Dict[str, Any]:
    overall = eval_data.get("overall", {})
    per_group = eval_data.get("per_group", {})

    mean_sum_reward = _safe_float(overall.get("avg_sum_reward"))
    mean_max_reward = _safe_float(overall.get("avg_max_reward"))
    success_rate = _safe_float(overall.get("pc_success"))

    n_episodes = _coerce_episode_count(overall.get("n_episodes"))
    if n_episodes is None:
        n_episodes = len(episodes)

    if success_rate is not None and success_rate <= 1.0:
        success_rate *= 100.0

    if mean_sum_reward is None:
        sum_candidates = [
            row.get("sum_reward") for row in episodes if row.get("sum_reward") is not None
        ]
        if sum_candidates:
            mean_sum_reward = mean([float(v) for v in sum_candidates])  # type: ignore[arg-type]

    if mean_max_reward is None:
        max_candidates = [float(row["max_reward"]) for row in episodes]
        if max_candidates:
            mean_max_reward = mean(max_candidates)

    if success_rate is None:
        success_values = [
            row.get("success")
            for row in episodes
            if isinstance(row.get("success"), bool)
        ]
        if success_values:
            success_count = sum(1 for x in success_values if x)
            success_rate = 100.0 * float(success_count) / float(len(success_values))
        else:
            success_count = 0
            success_rate = 0.0
    else:
        if episodes:
            success_count = sum(
                1 for row in episodes if row.get("success") is True and row.get("success") is not None
            )
        else:
            success_count = 0

    # Keep task-level metadata if available.
    task_entry = None
    if isinstance(per_group, Mapping) and isinstance(per_group.get(resolved_task), Mapping):
        task_entry = per_group.get(resolved_task)

    return {
        "mean_sum_reward": mean_sum_reward,
        "mean_max_reward": mean_max_reward,
        "success_rate_percent": success_rate,
        "success_count": success_count,
        "n_episodes": n_episodes,
        "top_k_requested": requested_top_k,
        "task_entry": task_entry,
    }


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("error: --top-k must be at least 1.")

    eval_path = Path(args.eval_info).expanduser()
    if not eval_path.exists():
        raise SystemExit(f"error: eval_info.json not found at {eval_path}")

    try:
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: invalid JSON in {eval_path}: {exc}") from exc
    if not isinstance(eval_data, Mapping):
        raise SystemExit(f"error: eval_info.json has unexpected structure: {eval_path}")

    try:
        video_overrides = _parse_video_overrides(args.video_path)
        task_metrics, resolved_task, task_id = _find_task_payload(
            eval_data, args.task
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    episode_rows = _extract_episode_rows(task_metrics, video_overrides)
    top_k = sorted(
        episode_rows,
        key=lambda row: (-float(row["max_reward"]), int(row["episode_index"])),
    )[: args.top_k]

    summary = _summary_block(
        eval_data, resolved_task, task_metrics, episode_rows, args.top_k
    )
    for idx, row in enumerate(top_k, start=1):
        row["rank"] = idx

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "eval_info": str(eval_path),
        "requested_task": args.task,
        "resolved_task": resolved_task,
        "task_id": task_id,
        "top_k": args.top_k,
        "episodes": top_k,
        "summary": summary,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
