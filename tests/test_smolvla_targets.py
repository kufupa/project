import json
from pathlib import Path

import pytest

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
    assert targets == [
        {
            "rank": 1,
            "episode_index": 7,
            "reset_seed": 1007,
            "task": "push-v3",
            "oracle_max_reward": 2.0,
        },
        {
            "rank": 2,
            "episode_index": 3,
            "reset_seed": 1003,
            "task": "push-v3",
            "oracle_max_reward": 1.5,
        },
    ]


@pytest.mark.parametrize("invalid_top_k", [0, -1])
def test_load_topk_targets_rejects_non_positive_top_k(tmp_path: Path, invalid_top_k: int):
    run_dir = tmp_path / "oracle_run"
    run_dir.mkdir()
    (run_dir / "optimal_report.json").write_text(
        json.dumps({"top_k": 1, "episodes": [{"rank": 1, "episode_index": 7}]}),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"task": "push-v3", "episodes": [{"episode_index": 7, "reset_seed": 1007}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        load_topk_targets(run_dir, invalid_top_k)


def test_load_topk_targets_raises_on_missing_manifest_episode_mapping(tmp_path: Path):
    run_dir = tmp_path / "oracle_run"
    run_dir.mkdir()
    (run_dir / "optimal_report.json").write_text(
        json.dumps({"top_k": 1, "episodes": [{"rank": 1, "episode_index": 11}]}),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "task": "push-v3",
                "episodes": [{"episode_index": 7, "reset_seed": 1007}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="episode_index 11"):
        load_topk_targets(run_dir, 1)


@pytest.mark.parametrize(
    ("episode_row", "expected_message"),
    [
        ({"episode_index": 7}, "Missing required field 'rank'"),
        ({"rank": 1}, "Missing required field 'episode_index'"),
    ],
)
def test_load_topk_targets_raises_on_missing_required_optimal_report_fields(
    tmp_path: Path, episode_row: dict, expected_message: str
):
    run_dir = tmp_path / "oracle_run"
    run_dir.mkdir()
    (run_dir / "optimal_report.json").write_text(
        json.dumps({"top_k": 1, "episodes": [episode_row]}),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"task": "push-v3", "episodes": [{"episode_index": 7, "reset_seed": 1007}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=expected_message):
        load_topk_targets(run_dir, 1)


def test_load_topk_targets_raises_on_missing_manifest_reset_seed(tmp_path: Path):
    run_dir = tmp_path / "oracle_run"
    run_dir.mkdir()
    (run_dir / "optimal_report.json").write_text(
        json.dumps({"top_k": 1, "episodes": [{"rank": 1, "episode_index": 7}]}),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"task": "push-v3", "episodes": [{"episode_index": 7}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required field 'reset_seed'"):
        load_topk_targets(run_dir, 1)


@pytest.mark.parametrize(
    ("task_value", "expected_message"),
    [
        (None, "Missing required field 'task'"),
        ("", "non-empty string"),
        ("   ", "non-empty string"),
        (123, "non-empty string"),
    ],
)
def test_load_topk_targets_raises_on_missing_or_invalid_manifest_task(
    tmp_path: Path, task_value: object, expected_message: str
):
    run_dir = tmp_path / "oracle_run"
    run_dir.mkdir()
    (run_dir / "optimal_report.json").write_text(
        json.dumps({"top_k": 1, "episodes": [{"rank": 1, "episode_index": 7}]}),
        encoding="utf-8",
    )
    manifest: dict[str, object] = {"episodes": [{"episode_index": 7, "reset_seed": 1007}]}
    if task_value is not None:
        manifest["task"] = task_value
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        load_topk_targets(run_dir, 1)


def test_write_targets_file(tmp_path: Path):
    from src.smolvla_pipeline.targets import write_targets_file

    output = tmp_path / "targets_top15.json"
    targets = [
        {"rank": 1, "episode_index": 7, "reset_seed": 1007, "task": "push-v3"},
        {"rank": 2, "episode_index": 3, "reset_seed": 1003, "task": "push-v3"},
    ]
    write_targets_file(output, targets)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"count", "targets"}
    assert payload["count"] == len(targets)
    assert payload["targets"] == targets
