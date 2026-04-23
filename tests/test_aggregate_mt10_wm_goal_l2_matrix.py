from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEG_SCRIPTS = ROOT / "scripts" / "segment_grpo"
_SPEC = importlib.util.spec_from_file_location(
    "aggregate_mt10_wm_goal_l2_matrix",
    SEG_SCRIPTS / "aggregate_mt10_wm_goal_l2_matrix.py",
)
assert _SPEC and _SPEC.loader
mt10 = importlib.util.module_from_spec(_SPEC)
sys.modules["aggregate_mt10_wm_goal_l2_matrix"] = mt10
_SPEC.loader.exec_module(mt10)


def _segment(selected: int, cand0: list[int]) -> dict:
    return {
        "segment_index": 0,
        "start_step": 0,
        "selected_index": selected,
        "carried_steps": 50,
        "metadata": {
            "comparison_wm_env_steps_per_wm_step": 5,
            "comparison_env_step_start": 0,
            "comparison_env_step_end": 50,
            "candidate_wm_goal_l2_int": [{"candidate_index": 0, "d_goal_l2_wm_int": cand0}],
        },
    }


def _write_minimal_campaign(
    root: Path,
    *,
    task: str,
    run_name: str,
    episode_pattern: str,
    first_bin_value: int,
) -> None:
    run = root / run_name
    run.mkdir(parents=True)
    (run / "segment_grpo_manifest.json").write_text(
        json.dumps({"task": task, "campaign": "test"}),
        encoding="utf-8",
    )
    vals = [first_bin_value] + [400 + i for i in range(1, 10)]
    body = {"episode_index": 0, "segments": [_segment(0, vals)]}
    fname = "out_episode_0000.json" if episode_pattern == "out" else "segment_grpo_episode_0000.json"
    (run / fname).write_text(json.dumps(body), encoding="utf-8")


def test_matrix_two_tasks_end_to_end(tmp_path: Path) -> None:
    p8 = tmp_path / "p8"
    p9 = tmp_path / "p9"
    _write_minimal_campaign(p8, task="reach-v3", run_name="mt10_run_a", episode_pattern="out", first_bin_value=200)
    _write_minimal_campaign(p8, task="push-v3", run_name="mt10_run_b", episode_pattern="out", first_bin_value=100)
    _write_minimal_campaign(p9, task="reach-v3", run_name="mt10_run_c", episode_pattern="seg", first_bin_value=220)
    _write_minimal_campaign(p9, task="push-v3", run_name="mt10_run_d", episode_pattern="seg", first_bin_value=110)

    payload = mt10.build_matrix_payload(
        phase8_root=p8,
        phase9_root=p9,
        candidate_mode_phase8="selected",
        candidate_mode_phase9="selected",
        strict=False,
    )
    assert payload["n_tasks"] == 2
    rows = payload["wide_csv_rows"]
    assert len(rows) == 4
    push_vla = next(r for r in rows if r["task"] == "push-v3" and r["phase"] == "VLA")
    assert push_vla["r0_5"] == pytest.approx(100.0)
    reach_orc = next(r for r in rows if r["task"] == "reach-v3" and r["phase"] == "oracle")
    assert reach_orc["r0_5"] == pytest.approx(220.0)

    csv_path = tmp_path / "m.csv"
    json_path = tmp_path / "m.json"
    tex_path = tmp_path / "m.tex"
    rc = mt10.main(
        [
            "--phase8-root",
            str(p8),
            "--phase9-root",
            str(p9),
            "--no-strict",
            "--out-csv",
            str(csv_path),
            "--out-json",
            str(json_path),
            "--latex-out",
            str(tex_path),
        ]
    )
    assert rc == 0
    assert csv_path.is_file()
    assert "push-v3" in tex_path.read_text(encoding="utf-8")
    assert r"\textbf{0:5}" in tex_path.read_text(encoding="utf-8")

    compare_csv_path = tmp_path / "m_compare.csv"
    compare_tex_path = tmp_path / "m_compare.tex"
    rc = mt10.main(
        [
            "--phase8-root",
            str(p8),
            "--phase9-root",
            str(p9),
            "--no-strict",
            "--out-csv-compare",
            str(compare_csv_path),
            "--out-latex-compare",
            str(compare_tex_path),
        ]
    )
    assert rc == 0
    assert compare_csv_path.is_file()
    assert compare_tex_path.is_file()


def test_compare_rows_include_all_tasks_average(tmp_path: Path) -> None:
    p8 = tmp_path / "p8"
    p9 = tmp_path / "p9"
    _write_minimal_campaign(p8, task="push-v3", run_name="mt10_run_b", episode_pattern="out", first_bin_value=10)
    _write_minimal_campaign(p8, task="reach-v3", run_name="mt10_run_a", episode_pattern="out", first_bin_value=20)
    _write_minimal_campaign(p9, task="push-v3", run_name="mt10_run_d", episode_pattern="seg", first_bin_value=12)
    _write_minimal_campaign(p9, task="reach-v3", run_name="mt10_run_c", episode_pattern="seg", first_bin_value=22)

    payload = mt10.build_matrix_payload(
        phase8_root=p8,
        phase9_root=p9,
        candidate_mode_phase8="selected",
        candidate_mode_phase9="selected",
        strict=False,
    )
    compare_rows = payload["compare_csv_rows"]
    assert len(compare_rows) == 3
    overall = compare_rows[-1]
    assert overall["task"] == "All tasks"
    assert overall["r0_5_VLA"] == 15
    assert overall["r0_5_oracle"] == 17
    assert overall["r0_5_delta"] == 2


def test_compare_long_rows_are_task_range_rows(tmp_path: Path) -> None:
    p8 = tmp_path / "p8"
    p9 = tmp_path / "p9"
    _write_minimal_campaign(p8, task="push-v3", run_name="mt10_run_b", episode_pattern="out", first_bin_value=10)
    _write_minimal_campaign(p8, task="reach-v3", run_name="mt10_run_a", episode_pattern="out", first_bin_value=20)
    _write_minimal_campaign(p9, task="push-v3", run_name="mt10_run_d", episode_pattern="seg", first_bin_value=12)
    _write_minimal_campaign(p9, task="reach-v3", run_name="mt10_run_c", episode_pattern="seg", first_bin_value=22)

    payload = mt10.build_matrix_payload(
        phase8_root=p8,
        phase9_root=p9,
        candidate_mode_phase8="selected",
        candidate_mode_phase9="selected",
        strict=False,
    )
    long_rows = payload["compare_csv_rows_long"]
    assert len(long_rows) == 30  # 2 tasks + All tasks, each with 10 bins
    first = long_rows[0]
    assert first["task"] == "push-v3"
    assert first["range"] == "0:5"
    assert first["vla"] == 10


def test_run_delta_latex_row_separators(tmp_path: Path) -> None:
    p8 = tmp_path / "p8"
    p9 = tmp_path / "p9"
    _write_minimal_campaign(p8, task="push-v3", run_name="mt10_run_b", episode_pattern="out", first_bin_value=10)
    _write_minimal_campaign(p8, task="reach-v3", run_name="mt10_run_a", episode_pattern="out", first_bin_value=20)
    _write_minimal_campaign(p9, task="push-v3", run_name="mt10_run_d", episode_pattern="seg", first_bin_value=12)
    _write_minimal_campaign(p9, task="reach-v3", run_name="mt10_run_c", episode_pattern="seg", first_bin_value=22)

    compare_tex_path = tmp_path / "m_compare.tex"
    assert (
        mt10.main(
            [
                "--phase8-root",
                str(p8),
                "--phase9-root",
                str(p9),
                "--no-strict",
                "--out-latex-compare",
                str(compare_tex_path),
                "--compare-format",
                "run-delta",
            ]
        )
        == 0
    )

    lines = [ln for ln in compare_tex_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    push_vla = next(i for i, ln in enumerate(lines) if ln.startswith("push & VLA &"))
    push_delta = next(i for i, ln in enumerate(lines) if ln.startswith("push & $\\Delta_{\\mathrm{oracle}}$ &"))
    reach_vla = next(i for i, ln in enumerate(lines) if ln.startswith("reach & VLA &"))
    reach_delta = next(i for i, ln in enumerate(lines) if ln.startswith("reach & $\\Delta_{\\mathrm{oracle}}$ &"))
    all_vla = next(i for i, ln in enumerate(lines) if ln.startswith("All tasks & VLA &"))
    all_delta = next(i for i, ln in enumerate(lines) if ln.startswith("All tasks & $\\Delta_{\\mathrm{oracle}}$ &"))

    # task-level pairs are grouped and separated by \\hline
    assert push_vla < push_delta < reach_vla
    assert push_delta + 1 == lines.index("\\hline", start=push_delta + 1, stop=reach_vla)
    assert all_vla < all_delta
    assert all_vla > reach_delta
    assert lines[all_vla - 1] == "\\hline"
    # all-tasks block is visually emphasized with a thicker divider
    assert lines[all_delta + 1] == "\\hline"
    assert lines[all_delta + 2] == "\\hline"


def test_compare_summary_rows_are_task_averages(tmp_path: Path) -> None:
    p8 = tmp_path / "p8"
    p9 = tmp_path / "p9"
    _write_minimal_campaign(p8, task="push-v3", run_name="mt10_run_b", episode_pattern="out", first_bin_value=10)
    _write_minimal_campaign(p8, task="reach-v3", run_name="mt10_run_a", episode_pattern="out", first_bin_value=20)
    _write_minimal_campaign(p9, task="push-v3", run_name="mt10_run_d", episode_pattern="seg", first_bin_value=12)
    _write_minimal_campaign(p9, task="reach-v3", run_name="mt10_run_c", episode_pattern="seg", first_bin_value=22)

    payload = mt10.build_matrix_payload(
        phase8_root=p8,
        phase9_root=p9,
        candidate_mode_phase8="selected",
        candidate_mode_phase9="selected",
        strict=False,
    )
    summary_rows = payload["compare_csv_rows_summary"]
    assert len(summary_rows) == 3
    assert summary_rows[-1]["task"] == "All tasks"
    assert summary_rows[0]["task"] == "push-v3"
    assert summary_rows[0]["vla"] == 364
    assert summary_rows[0]["oracle"] == 366


def test_strict_requires_ten_tasks(tmp_path: Path) -> None:
    p8 = tmp_path / "p8"
    p9 = tmp_path / "p9"
    _write_minimal_campaign(p8, task="push-v3", run_name="mt10_run_b", episode_pattern="out", first_bin_value=1)
    _write_minimal_campaign(p9, task="push-v3", run_name="mt10_run_d", episode_pattern="seg", first_bin_value=2)
    with pytest.raises(ValueError, match="Expected 10 paired tasks"):
        mt10.build_matrix_payload(
            phase8_root=p8,
            phase9_root=p9,
            candidate_mode_phase8="selected",
            candidate_mode_phase9="selected",
            strict=True,
        )


def test_task_mismatch_strict(tmp_path: Path) -> None:
    p8 = tmp_path / "p8"
    p9 = tmp_path / "p9"
    _write_minimal_campaign(p8, task="push-v3", run_name="mt10_run_b", episode_pattern="out", first_bin_value=1)
    _write_minimal_campaign(p8, task="reach-v3", run_name="mt10_run_a", episode_pattern="out", first_bin_value=2)
    _write_minimal_campaign(p9, task="push-v3", run_name="mt10_run_d", episode_pattern="seg", first_bin_value=2)
    with pytest.raises(ValueError, match="Task mismatch"):
        mt10.build_matrix_payload(
            phase8_root=p8,
            phase9_root=p9,
            candidate_mode_phase8="selected",
            candidate_mode_phase9="selected",
            strict=True,
        )
