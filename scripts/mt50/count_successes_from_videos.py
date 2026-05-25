#!/usr/bin/env python3
"""Infer MT50 video success counts from episode MP4 durations.

This is a recovery tool for runs that produced videos but exited before
LeRobot wrote eval_info.json. In these Phase27 videos, failures run to the
fixed horizon (6.25s); successes terminate earlier.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


DIFFICULTY_ORDER = ("easy", "medium", "hard", "very_hard")
DEFAULT_DIFFICULTY_MAP = Path("scripts/mt50/mt50_phase07_task_difficulties.json")


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    path: Path
    duration_s: float | None
    success: bool | None
    status: str


@dataclass(frozen=True)
class TaskResult:
    source_videos_dir: Path
    folder: str
    task: str
    difficulty: str
    status: str
    episodes_found: int
    expected_episodes: int
    successes: int
    episodes: list[EpisodeResult]

    @property
    def pc_success(self) -> float:
        if self.status != "complete" or self.expected_episodes <= 0:
            return float("nan")
        return 100.0 * self.successes / self.expected_episodes


def _iter_boxes(f: BinaryIO, end: int):
    while f.tell() < end:
        start = f.tell()
        header = f.read(8)
        if len(header) < 8:
            return
        size, box_type_raw = struct.unpack(">I4s", header)
        box_type = box_type_raw.decode("latin1")
        header_size = 8
        if size == 1:
            large_size = f.read(8)
            if len(large_size) < 8:
                return
            size = struct.unpack(">Q", large_size)[0]
            header_size = 16
        elif size == 0:
            size = end - start
        if size < header_size:
            return
        yield start, size, header_size, box_type
        f.seek(start + size)


def _parse_duration_from_versioned_box(payload: bytes) -> float | None:
    if not payload:
        return None
    version = payload[0]
    if version == 1:
        if len(payload) < 32:
            return None
        timescale = struct.unpack(">I", payload[20:24])[0]
        duration = struct.unpack(">Q", payload[24:32])[0]
    elif version == 0:
        if len(payload) < 20:
            return None
        timescale = struct.unpack(">I", payload[12:16])[0]
        duration = struct.unpack(">I", payload[16:20])[0]
    else:
        return None
    if timescale <= 0:
        return None
    return float(duration) / float(timescale)


def mp4_duration_s(path: Path) -> float:
    """Return MP4 movie duration from mvhd/mdhd boxes without ffmpeg."""
    durations: list[float] = []
    with path.open("rb") as f:
        f.seek(0, 2)
        total_size = f.tell()
        f.seek(0)

        def walk(end: int) -> None:
            for start, size, header_size, box_type in _iter_boxes(f, end):
                data_start = start + header_size
                if box_type in {"moov", "trak", "mdia"}:
                    f.seek(data_start)
                    walk(start + size)
                elif box_type in {"mvhd", "mdhd"}:
                    f.seek(data_start)
                    payload = f.read(min(size - header_size, 32))
                    duration = _parse_duration_from_versioned_box(payload)
                    if duration is not None:
                        durations.append(duration)
                f.seek(start + size)

        walk(total_size)
    if not durations:
        raise ValueError(f"no MP4 duration box found: {path}")
    # Prefer movie header when present; otherwise track/media duration is enough
    # for these single-stream eval videos.
    return max(durations)


def task_name_from_video_folder(folder_name: str) -> str:
    if folder_name.endswith("_0"):
        return folder_name[:-2]
    return folder_name


def _episode_index(path: Path) -> int:
    stem = path.stem
    prefix = "eval_episode_"
    if not stem.startswith(prefix):
        return -1
    try:
        return int(stem[len(prefix) :])
    except ValueError:
        return -1


def classify_duration(duration_s: float, *, failure_duration_s: float, epsilon_s: float) -> bool:
    return duration_s < failure_duration_s - epsilon_s


def load_difficulty_map(path: Path) -> tuple[dict[str, str], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("task_difficulties") or {}), str(data.get("default") or "unclassified")


def scan_task_folder(
    task_dir: Path,
    *,
    videos_dir: Path,
    expected_episodes: int,
    failure_duration_s: float,
    epsilon_s: float,
    difficulty_by_task: dict[str, str],
    default_difficulty: str,
) -> TaskResult:
    video_paths = sorted(task_dir.glob("eval_episode_*.mp4"), key=_episode_index)
    task = task_name_from_video_folder(task_dir.name)
    episodes: list[EpisodeResult] = []
    complete = len(video_paths) == expected_episodes
    for path in video_paths:
        episode = _episode_index(path)
        duration: float | None = None
        success: bool | None = None
        status = "ok"
        try:
            if path.stat().st_size <= 0:
                raise ValueError("empty file")
            duration = mp4_duration_s(path)
            if duration <= 0.0 or not math.isfinite(duration):
                raise ValueError(f"invalid duration {duration}")
            success = classify_duration(
                duration,
                failure_duration_s=failure_duration_s,
                epsilon_s=epsilon_s,
            )
        except Exception as exc:
            status = f"invalid:{type(exc).__name__}"
        episodes.append(
            EpisodeResult(
                episode=episode,
                path=path,
                duration_s=duration,
                success=success,
                status=status,
            )
        )
    invalid = any(ep.status != "ok" for ep in episodes)
    status = "complete" if complete and not invalid else "broken"
    successes = sum(1 for ep in episodes if ep.success is True) if status == "complete" else 0
    return TaskResult(
        source_videos_dir=videos_dir,
        folder=task_dir.name,
        task=task,
        difficulty=difficulty_by_task.get(task, default_difficulty),
        status=status,
        episodes_found=len(video_paths),
        expected_episodes=expected_episodes,
        successes=successes,
        episodes=episodes,
    )


def scan_videos_dir(
    videos_dir: Path,
    *,
    expected_episodes: int,
    failure_duration_s: float,
    epsilon_s: float,
    difficulty_map_path: Path,
) -> list[TaskResult]:
    difficulty_by_task, default_difficulty = load_difficulty_map(difficulty_map_path)
    task_dirs = sorted(path for path in videos_dir.iterdir() if path.is_dir())
    return [
        scan_task_folder(
            task_dir,
            videos_dir=videos_dir,
            expected_episodes=expected_episodes,
            failure_duration_s=failure_duration_s,
            epsilon_s=epsilon_s,
            difficulty_by_task=difficulty_by_task,
            default_difficulty=default_difficulty,
        )
        for task_dir in task_dirs
    ]


def _duration_stats(episodes: list[EpisodeResult]) -> tuple[float | None, float | None, float | None]:
    durations = [ep.duration_s for ep in episodes if ep.duration_s is not None]
    if not durations:
        return None, None, None
    return min(durations), statistics.median(durations), max(durations)


def _json_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def task_to_json(row: TaskResult) -> dict[str, Any]:
    min_d, med_d, max_d = _duration_stats(row.episodes)
    return {
        "task": row.task,
        "folder": row.folder,
        "source_videos_dir": str(row.source_videos_dir),
        "difficulty": row.difficulty,
        "status": row.status,
        "episodes_found": row.episodes_found,
        "expected_episodes": row.expected_episodes,
        "successes": row.successes if row.status == "complete" else None,
        "pc_success": _json_float(row.pc_success if row.status == "complete" else None),
        "duration_s": {"min": min_d, "median": med_d, "max": max_d},
        "episodes": [
            {
                "episode": ep.episode,
                "path": str(ep.path.relative_to(row.source_videos_dir)),
                "duration_s": _json_float(ep.duration_s),
                "success": ep.success,
                "status": ep.status,
            }
            for ep in row.episodes
        ],
    }


def results_to_json(
    rows: list[TaskResult],
    *,
    videos_dirs: list[Path],
    failure_duration_s: float,
    epsilon_s: float,
) -> dict[str, Any]:
    complete = [row for row in rows if row.status == "complete"]
    by_difficulty = aggregate_by_difficulty(rows)
    return {
        "videos_dirs": [str(path) for path in videos_dirs],
        "failure_duration_s": failure_duration_s,
        "epsilon_s": epsilon_s,
        "complete_tasks": len(complete),
        "broken_tasks": len(rows) - len(complete),
        "tasks": [task_to_json(row) for row in rows],
        "difficulty_summary": {
            difficulty: {
                **values,
                "pc_success": _json_float(float(values["pc_success"])),
            }
            for difficulty, values in by_difficulty.items()
        },
        "macro_average_pc_success": _json_float(macro_average_pc_success(by_difficulty)),
    }


def aggregate_by_difficulty(rows: list[TaskResult]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, dict[str, float | int]] = defaultdict(lambda: {"successes": 0, "episodes": 0, "tasks": 0})
    for row in rows:
        if row.status != "complete" or row.difficulty not in DIFFICULTY_ORDER:
            continue
        grouped[row.difficulty]["successes"] = int(grouped[row.difficulty]["successes"]) + row.successes
        grouped[row.difficulty]["episodes"] = int(grouped[row.difficulty]["episodes"]) + row.expected_episodes
        grouped[row.difficulty]["tasks"] = int(grouped[row.difficulty]["tasks"]) + 1
    out: dict[str, dict[str, float | int]] = {}
    for difficulty in DIFFICULTY_ORDER:
        successes = int(grouped[difficulty]["successes"])
        episodes = int(grouped[difficulty]["episodes"])
        out[difficulty] = {
            "successes": successes,
            "episodes": episodes,
            "tasks": int(grouped[difficulty]["tasks"]),
            "pc_success": (100.0 * successes / episodes) if episodes else float("nan"),
        }
    return out


def macro_average_pc_success(summary: dict[str, dict[str, float | int]]) -> float:
    values = [float(summary[difficulty]["pc_success"]) for difficulty in DIFFICULTY_ORDER]
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) != len(DIFFICULTY_ORDER):
        return float("nan")
    return sum(finite) / len(finite)


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _pct_cell(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.1f}\\%"


def _duration_cell(duration_s: float | None) -> str:
    if duration_s is None:
        return "--"
    return f"{duration_s:.3f}"


def _outcome_cell(ep: EpisodeResult) -> str:
    if ep.status != "ok":
        return _latex_escape(ep.status)
    return "success" if ep.success else "failure"


def render_latex_report(
    rows: list[TaskResult],
    *,
    caption_prefix: str,
) -> str:
    lines: list[str] = [
        "% Generated by scripts/mt50/count_successes_from_videos.py",
        "% Video-derived success recovery; failures are fixed-horizon videos.",
        r"\begin{longtable}{lllrl}",
        rf"\caption{{{_latex_escape(caption_prefix)} per-episode video-derived outcomes.}} \\",
        r"\hline",
        r"Task & Difficulty & Episode & Duration (s) & Outcome \\",
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        r"Task & Difficulty & Episode & Duration (s) & Outcome \\",
        r"\hline",
        r"\endhead",
    ]
    for row in rows:
        for ep in row.episodes:
            lines.append(
                " & ".join(
                    [
                        _latex_escape(row.task),
                        _latex_escape(row.difficulty),
                        str(ep.episode),
                        _duration_cell(ep.duration_s),
                        _outcome_cell(ep),
                    ]
                )
                + r" \\"
            )
        if row.status == "complete":
            total = f"{row.successes}/{row.expected_episodes}"
            pct = _pct_cell(row.pc_success)
        else:
            total = f"broken ({row.episodes_found}/{row.expected_episodes})"
            pct = "--"
        lines.append(
            r"\textbf{"
            + _latex_escape(row.task)
            + r" total} & "
            + _latex_escape(row.difficulty)
            + r" & -- & "
            + _latex_escape(total)
            + " & "
            + pct
            + r" \\"
        )
        lines.append(r"\hline")
    lines.extend([r"\end{longtable}", ""])

    summary = aggregate_by_difficulty(rows)
    macro = macro_average_pc_success(summary)
    lines.extend(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{_latex_escape(caption_prefix)} success by task difficulty.}}",
            r"\begin{tabular}{lrrr}",
            r"\hline",
            r"Difficulty & Success & Episodes & Success (\%) \\",
            r"\hline",
        ]
    )
    for difficulty in DIFFICULTY_ORDER:
        item = summary[difficulty]
        successes = int(item["successes"])
        episodes = int(item["episodes"])
        lines.append(
            " & ".join(
                [
                    _latex_escape(difficulty),
                    str(successes),
                    str(episodes),
                    _pct_cell(float(item["pc_success"])),
                ]
            )
            + r" \\"
        )
    lines.append(" & ".join(["Macro-average", "--", "--", _pct_cell(macro)]) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def print_summary(rows: list[TaskResult]) -> None:
    for row in rows:
        if row.status == "complete":
            min_d, med_d, max_d = _duration_stats(row.episodes)
            print(
                f"{row.task}\t{row.difficulty}\t{row.successes}/{row.expected_episodes}"
                f"\t{row.pc_success:.1f}%\tduration_min_med_max="
                f"{min_d:.3f},{med_d:.3f},{max_d:.3f}"
            )
        else:
            print(f"{row.task}\t{row.difficulty}\tbroken\tvideos={row.episodes_found}/{row.expected_episodes}")
    summary = aggregate_by_difficulty(rows)
    print("difficulty_summary")
    for difficulty in DIFFICULTY_ORDER:
        item = summary[difficulty]
        print(
            f"{difficulty}\t{int(item['successes'])}/{int(item['episodes'])}"
            f"\t{_pct_cell(float(item['pc_success'])).replace('\\%', '%')}"
        )
    macro = macro_average_pc_success(summary)
    macro_text = "nan" if not math.isfinite(macro) else f"{macro:.1f}%"
    print(f"macro_average\t{macro_text}")


def discover_videos_dirs(parent: Path) -> list[Path]:
    return sorted(
        path
        for path in parent.glob("shard_*_*tasks/videos")
        if path.is_dir()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--videos-dir",
        type=Path,
        nargs="+",
        help="One or more videos folders containing task subfolders.",
    )
    source.add_argument(
        "--parent",
        type=Path,
        help="Artifact root; scans shard_*_*tasks/videos under it.",
    )
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--failure-duration", type=float, default=6.25)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--latex-out", type=Path, default=None)
    parser.add_argument("--difficulty-map", type=Path, default=DEFAULT_DIFFICULTY_MAP)
    parser.add_argument("--caption-prefix", default="MT50 Phase27 video-derived")
    args = parser.parse_args()

    if args.parent is not None:
        videos_dirs = discover_videos_dirs(args.parent.resolve())
        if not videos_dirs:
            raise SystemExit(f"no shard_*_*tasks/videos dirs under: {args.parent.resolve()}")
    else:
        videos_dirs = [path.resolve() for path in args.videos_dir]
    for videos_dir in videos_dirs:
        if not videos_dir.is_dir():
            raise SystemExit(f"videos dir not found: {videos_dir}")

    rows: list[TaskResult] = []
    for videos_dir in videos_dirs:
        rows.extend(
            scan_videos_dir(
                videos_dir,
                expected_episodes=args.episodes,
                failure_duration_s=args.failure_duration,
                epsilon_s=args.epsilon,
                difficulty_map_path=args.difficulty_map,
            )
        )
    print_summary(rows)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = results_to_json(
            rows,
            videos_dirs=videos_dirs,
            failure_duration_s=args.failure_duration,
            epsilon_s=args.epsilon,
        )
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")

    if args.latex_out is not None:
        args.latex_out.parent.mkdir(parents=True, exist_ok=True)
        latex = render_latex_report(rows, caption_prefix=args.caption_prefix)
        args.latex_out.write_text(latex, encoding="utf-8")
        print(f"wrote {args.latex_out}")


if __name__ == "__main__":
    main()
