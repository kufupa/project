from __future__ import annotations

from pathlib import Path

import pytest


def test_split_tasks_for_five_shards_is_ten_each() -> None:
    from scripts.grpo.phase57_mt50_raw_vs_bounded_decode import MT50_TASKS, split_tasks_for_shard

    shards = [split_tasks_for_shard(MT50_TASKS, shard_index=i, shard_count=5) for i in range(5)]

    assert [len(x) for x in shards] == [10, 10, 10, 10, 10]
    assert shards[0][0] == "assembly-v3"
    assert shards[0][-1] == "coffee-pull-v3"
    assert shards[3] == [
        "pick-out-of-hole-v3",
        "pick-place-v3",
        "pick-place-wall-v3",
        "plate-slide-back-side-v3",
        "plate-slide-back-v3",
        "plate-slide-side-v3",
        "plate-slide-v3",
        "push-back-v3",
        "push-v3",
        "push-wall-v3",
    ]
    assert shards[4][-1] == "window-open-v3"


def test_phase57_parse_defaults(tmp_path: Path) -> None:
    from scripts.grpo.phase57_mt50_raw_vs_bounded_decode import parse_args

    args = parse_args(["--output-dir", str(tmp_path)])

    assert args.episodes == 25
    assert args.n_envs == 3
    assert args.chunk_len == 50
    assert args.max_steps == 180
    assert args.goal_latent_mode == "visual_proprio"


def test_merge_phase57_summary_counts_success_and_l2(tmp_path: Path) -> None:
    from scripts.grpo.merge_phase57_mt50_decode import build_merged_summary

    merged = build_merged_summary(
        parent=tmp_path,
        expected_tasks=2,
        expected_episodes=4,
        task_summaries=[
            {
                "task": "push-v3",
                "episodes_completed": 2,
                "pc_success": 50.0,
                "mean_raw_combined_l2": 1.0,
                "mean_bounded_combined_l2": 2.0,
                "metric_column_count": 10,
                "raw_win_fraction": 0.7,
                "bounded_win_fraction": 0.2,
                "tie_fraction": 0.1,
                "episodes_rows": [{"success": True}, {"success": False}],
            },
            {
                "task": "reach-v3",
                "episodes_completed": 2,
                "pc_success": 100.0,
                "mean_raw_combined_l2": 3.0,
                "mean_bounded_combined_l2": 1.0,
                "metric_column_count": 10,
                "raw_win_fraction": 0.3,
                "bounded_win_fraction": 0.6,
                "tie_fraction": 0.1,
                "episodes_rows": [{"success": True}, {"success": True}],
            },
        ],
    )

    assert merged["tasks_found"] == 2
    assert merged["episodes_found"] == 4
    assert merged["micro_pc_success"] == pytest.approx(75.0)
    assert merged["macro_pc_success"] == pytest.approx(75.0)
    assert merged["mean_raw_combined_l2"] == pytest.approx(2.0)
    assert merged["mean_bounded_combined_l2"] == pytest.approx(1.5)
    assert merged["raw_win_fraction"] == pytest.approx(0.5)
    assert merged["bounded_win_fraction"] == pytest.approx(0.4)


def test_merge_phase57_l2_means_are_weighted_by_metric_columns(tmp_path: Path) -> None:
    from scripts.grpo.merge_phase57_mt50_decode import build_merged_summary

    merged = build_merged_summary(
        parent=tmp_path,
        expected_tasks=2,
        expected_episodes=2,
        task_summaries=[
            {
                "task": "short-v3",
                "episodes_completed": 1,
                "metric_column_count": 1,
                "mean_raw_combined_l2": 100.0,
                "mean_bounded_combined_l2": 100.0,
                "episodes_rows": [{"success": True}],
            },
            {
                "task": "long-v3",
                "episodes_completed": 1,
                "metric_column_count": 9,
                "mean_raw_combined_l2": 0.0,
                "mean_bounded_combined_l2": 10.0,
                "episodes_rows": [{"success": True}],
            },
        ],
    )

    assert merged["mean_raw_combined_l2"] == pytest.approx(10.0)
    assert merged["mean_bounded_combined_l2"] == pytest.approx(19.0)
