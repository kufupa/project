from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_compute_action_range_summary_records_arm_and_gripper(tmp_path: Path) -> None:
    from scripts.grpo.phase56_oracle_action_audit import compute_action_range_summary

    raw = np.array(
        [
            [0.25, -1.25, 0.50, 1.50],
            [1.25, 0.00, -0.75, -1.50],
        ],
        dtype=np.float32,
    )
    clipped = np.clip(raw, -1.0, 1.0)
    summary = compute_action_range_summary(
        raw_actions=raw,
        clipped_actions=clipped,
        rewards=np.array([0.0, 1.0], dtype=np.float64),
        successes=np.array([False, True]),
        task="push-v3",
        seed=2000,
        max_steps=120,
        raw_actions_path=tmp_path / "raw.npy",
        clipped_actions_path=tmp_path / "clipped.npy",
        steps_path=tmp_path / "steps.jsonl",
        video_path=None,
    )

    assert summary["any_outside_minus1_1"] is True
    assert summary["outside_value_count"] == 4
    assert summary["groups"]["arm"]["outside_value_count"] == 2
    assert summary["groups"]["gripper"]["outside_value_count"] == 2
    assert summary["dims"]["arm_y"]["min"] == pytest.approx(-1.25)
    assert summary["dims"]["arm_x"]["max"] == pytest.approx(1.25)
    assert summary["dims"]["gripper"]["outside_low_count"] == 1
    assert summary["dims"]["gripper"]["outside_high_count"] == 1
    assert summary["success_frame_1based"] == 3


def test_phase56_parse_defaults() -> None:
    from scripts.grpo.phase56_oracle_action_audit import parse_args

    args = parse_args([])

    assert args.task == "push-v3"
    assert args.seed == 2000
    assert args.max_steps == 120
    assert args.save_video is False
    assert str(args.output_dir).endswith("artifacts/phase56_oracle_action_audit/dry_run")


def test_phase56_dry_run_writes_manifest(tmp_path: Path) -> None:
    from scripts.grpo.phase56_oracle_action_audit import main

    rc = main(["--dry-run", "--output-dir", str(tmp_path), "--task", "push-v3"])

    assert rc == 0
    manifest = json.loads((tmp_path / "phase56_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "phase56_oracle_action_audit"
    assert manifest["task"] == "push-v3"


def test_step_writers_include_raw_clipped_and_outside(tmp_path: Path) -> None:
    from scripts.grpo.phase56_oracle_action_audit import _write_steps_csv, _write_steps_jsonl

    raw = np.array([[2.0, -0.5, 0.0, -2.0]], dtype=np.float32)
    clipped = np.clip(raw, -1.0, 1.0)
    rewards = np.array([1.5], dtype=np.float64)
    successes = np.array([True])

    jsonl = _write_steps_jsonl(
        tmp_path / "steps.jsonl",
        raw_actions=raw,
        clipped_actions=clipped,
        rewards=rewards,
        successes=successes,
    )
    csv_path = _write_steps_csv(
        tmp_path / "steps.csv",
        raw_actions=raw,
        clipped_actions=clipped,
        rewards=rewards,
        successes=successes,
    )

    row = json.loads(jsonl.read_text(encoding="utf-8").strip())
    assert row["raw_action"] == [2.0, -0.5, 0.0, -2.0]
    assert row["clipped_action"] == [1.0, -0.5, 0.0, -1.0]
    assert row["outside_mask"] == [True, False, False, True]
    assert "raw_arm_x" in csv_path.read_text(encoding="utf-8").splitlines()[0]
