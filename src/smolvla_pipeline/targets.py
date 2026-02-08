from __future__ import annotations

import json
from pathlib import Path


def load_topk_targets(oracle_run_dir: Path, top_k: int) -> list[dict]:
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k}.")

    optimal_report = json.loads((oracle_run_dir / "optimal_report.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((oracle_run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    reset_seed_by_episode_index: dict[int, int] = {}
    for row_index, episode in enumerate(run_manifest.get("episodes", [])):
        if "episode_index" not in episode:
            raise ValueError(
                f"Missing required field 'episode_index' in run_manifest episode row at index {row_index}."
            )
        if "reset_seed" not in episode:
            raise ValueError(
                f"Missing required field 'reset_seed' in run_manifest episode row at index {row_index}."
            )
        reset_seed_by_episode_index[int(episode["episode_index"])] = int(episode["reset_seed"])

    if "task" not in run_manifest:
        raise ValueError("Missing required field 'task' in run_manifest.json.")
    task = run_manifest["task"]
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Field 'task' in run_manifest.json must be a non-empty string.")
    targets: list[dict] = []
    for row_index, episode in enumerate(optimal_report.get("episodes", [])[:top_k]):
        if "episode_index" not in episode:
            raise ValueError(
                f"Missing required field 'episode_index' in optimal_report episode row at index {row_index}."
            )
        if "rank" not in episode:
            raise ValueError(
                f"Missing required field 'rank' in optimal_report episode row at index {row_index}."
            )

        episode_index = int(episode["episode_index"])
        if episode_index not in reset_seed_by_episode_index:
            raise ValueError(
                f"Missing reset_seed mapping for episode_index {episode_index} in run_manifest.json."
            )
        targets.append(
            {
                "rank": int(episode["rank"]),
                "episode_index": episode_index,
                "reset_seed": reset_seed_by_episode_index[episode_index],
                "task": task,
            }
        )

    return targets


def write_targets_file(path: Path, targets: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"count": len(targets), "targets": targets}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
