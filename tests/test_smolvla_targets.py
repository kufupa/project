import json
from pathlib import Path

from src.smolvla_pipeline.targets import load_topk_targets


def test_load_topk_targets_reads_reset_seed_from_manifest(tmp_path: Path):
    run_dir = tmp_path / "oracle_run"
    run_dir.mkdir()
    (run_dir / "optimal_report.json").write_text(
        json.dumps(
            {
                "top_k": 2,
                "episodes": [
                    {"rank": 1, "episode_index": 7, "max_reward": 2.0},
                    {"rank": 2, "episode_index": 3, "max_reward": 1.5},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "task": "push-v3",
                "episodes": [
                    {"episode_index": 3, "reset_seed": 1003},
                    {"episode_index": 7, "reset_seed": 1007},
                ],
            }
        ),
        encoding="utf-8",
    )
    targets = load_topk_targets(run_dir, top_k=2)
    assert [t["episode_index"] for t in targets] == [7, 3]
    assert [t["reset_seed"] for t in targets] == [1007, 1003]
