from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEG_SCRIPTS = ROOT / "scripts" / "segment_grpo"
_SPEC = importlib.util.spec_from_file_location(
    "aggregate_wm_goal_l2_by_action_range",
    SEG_SCRIPTS / "aggregate_wm_goal_l2_by_action_range.py",
)
assert _SPEC and _SPEC.loader
agg = importlib.util.module_from_spec(_SPEC)
sys.modules["aggregate_wm_goal_l2_by_action_range"] = agg
_SPEC.loader.exec_module(agg)


def _segment(selected: int, cand0: list[int], cand1: list[int] | None = None) -> dict:
    rows = [
        {"candidate_index": 0, "d_goal_l2_wm_int": cand0},
    ]
    if cand1 is not None:
        rows.append({"candidate_index": 1, "d_goal_l2_wm_int": cand1})
    return {
        "segment_index": 0,
        "start_step": 0,
        "selected_index": selected,
        "carried_steps": 50,
        "metadata": {
            "comparison_wm_env_steps_per_wm_step": 5,
            "comparison_env_step_start": 0,
            "comparison_env_step_end": 50,
            "candidate_wm_goal_l2_int": rows,
        },
    }


def test_selected_picks_candidate_row() -> None:
    seg = _segment(
        selected=1,
        cand0=[400, 410, 420, 430, 440, 450, 460, 470, 480, 490],
        cand1=[100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
    )
    m = agg._episode_bin_map({"segments": [seg]}, candidate_mode="selected")
    assert m is not None
    first = agg.BinKey(0, 5)
    assert m[first] == 100.0


def test_all_mean_averages_candidates() -> None:
    seg = _segment(
        selected=1,
        cand0=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        cand1=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    )
    m = agg._episode_bin_map({"segments": [seg]}, candidate_mode="all_mean")
    assert m is not None
    assert m[agg.BinKey(0, 5)] == 5.0


def test_aggregate_bins_se() -> None:
    ep0 = {agg.BinKey(0, 5): 10.0, agg.BinKey(5, 10): 20.0}
    ep1 = {agg.BinKey(0, 5): 14.0, agg.BinKey(5, 10): 20.0}
    st = agg._aggregate_bins([ep0, ep1])
    assert st[agg.BinKey(0, 5)]["mean"] == 12.0
    assert st[agg.BinKey(0, 5)]["n"] == 2.0
    assert st[agg.BinKey(0, 5)]["se"] > 0


def test_merge_paired_delta() -> None:
    policy = {agg.BinKey(0, 5): {"n": 2.0, "mean": 12.0, "se": 1.0}}
    oracle = {agg.BinKey(0, 5): {"n": 2.0, "mean": 10.0, "se": 0.5}}
    rows = agg._merge_paired(policy, oracle)
    assert len(rows) == 1
    assert rows[0]["delta_mean"] == pytest.approx(2.0)


def test_cli_end_to_end_paired(tmp_path: Path) -> None:
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()
    seg_a = _segment(0, [438, 435, 437, 436, 432, 420, 415, 391, 344, 287])
    seg_b = _segment(0, [436, 428, 431, 399, 389, 385, 379, 379, 354, 297])
    (run_a / "out_episode_0000.json").write_text(
        json.dumps({"episode_index": 0, "segments": [seg_a]}), encoding="utf-8"
    )
    (run_b / "segment_grpo_episode_0000.json").write_text(
        json.dumps({"episode_index": 0, "segments": [seg_b]}), encoding="utf-8"
    )
    tex = tmp_path / "out.tex"
    rc = agg.main(
        [
            "--run-dir",
            str(run_a),
            "--compare-run-dir",
            str(run_b),
            "--candidate-mode",
            "selected",
            "--out-prefix",
            "t",
            "--out-dir",
            str(tmp_path),
            "--latex-out",
            str(tex),
        ]
    )
    assert rc == 0
    assert tex.is_file()
    body = tex.read_text(encoding="utf-8")
    assert "PAIRED TABLE" in body
    assert r"\Delta" not in body
    assert "delta_mean" not in body
