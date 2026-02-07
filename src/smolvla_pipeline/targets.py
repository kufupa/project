from __future__ import annotations

import json
from pathlib import Path


def load_topk_targets(oracle_run_dir: Path, *, top_k: int) -> list[dict]:
    optimal_report = json.loads((oracle_run_dir / "optimal_report.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((oracle_run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    reset_seed_by_episode_index = {
        int(episode["episode_index"]): int(episode["reset_seed"])
        for episode in run_manifest.get("episodes", [])
    }

    task = run_manifest.get("task", "push-v3")
    targets: list[dict] = []
    for episode in optimal_report.get("episodes", [])[:top_k]:
        episode_index = int(episode["episode_index"])
        targets.append(
            {
                "rank": int(episode["rank"]),
                "episode_index": episode_index,
                "reset_seed": reset_seed_by_episode_index[episode_index],
                "task": task,
            }
        )

    return targets
