from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _load_module():
    path = _REPO / "scripts" / "grpo" / "summarize_phase111_grpo_progress.py"
    spec = importlib.util.spec_from_file_location("summarize_phase111_grpo_progress", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_summary_reports_expected_metrics() -> None:
    mod = _load_module()
    rows = [
        {
            "update": 0,
            "avg_return": 1.0,
            "returns": [1, 1, 1, 1],
            "successes": [False, False, False, False],
            "success_rate": 0.0,
            "skipped": True,
            "episode_lengths": [500, 500, 500, 500],
        },
        {
            "update": 1,
            "avg_return": 3.0,
            "returns": [2, 3, 4, 3],
            "successes": [False, True, False, False],
            "success_rate": 0.25,
            "episode_lengths": [500, 120, 500, 500],
        },
    ]
    summary = mod.build_summary(rows)
    assert summary["n_updates"] == 2
    assert summary["skipped_updates"] == 1
    assert summary["best_avg_return_update"] == 1
    assert summary["best_success_rate_update"] == 1
    assert summary["last_update"] == 1
    assert summary["identical_return_updates"] == [0]
    encoded = json.dumps(summary)
    assert "mean_episode_length_last5" in encoded
