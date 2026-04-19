from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from segment_grpo_reference import load_oracle_action_sequence


def test_load_oracle_action_sequence_happy(tmp_path: Path) -> None:
    run_dir = tmp_path / "oracle"
    ep_dir = run_dir / "episodes" / "episode_0000"
    ep_dir.mkdir(parents=True)
    actions_path = ep_dir / "actions.jsonl"
    with actions_path.open("w", encoding="utf-8") as fp:
        for i in range(3):
            fp.write(json.dumps({"step": i, "action": [0.1, 0.2, 0.3, 0.4]}) + "\n")

    seq = load_oracle_action_sequence(run_dir, 0)
    assert seq.n_steps == 3
    assert seq.env_action_dim == 4
    assert seq.actions.shape == (3, 4)
    assert seq.action_source_path == actions_path.resolve()


def test_load_oracle_action_sequence_mixed_widths_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "oracle"
    ep_dir = run_dir / "episodes" / "episode_0000"
    ep_dir.mkdir(parents=True)
    actions_path = ep_dir / "actions.jsonl"
    with actions_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps({"step": 0, "action": [0.1, 0.2, 0.3, 0.4]}) + "\n")
        fp.write(json.dumps({"step": 1, "action": [0.1, 0.2]}) + "\n")

    with pytest.raises(ValueError, match="mixed widths"):
        load_oracle_action_sequence(run_dir, 0)


def test_load_oracle_action_sequence_missing_file_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "oracle"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="actions.jsonl"):
        load_oracle_action_sequence(run_dir, 0)
