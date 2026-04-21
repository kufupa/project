from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SCRIPT_ROOT = ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_phase9_oracle_vs_wm import main as phase9_main  # noqa: E402
from segment_grpo_loop import EpisodeLog  # noqa: E402


def test_phase9_main_dry_run_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_root = tmp_path / "oracle"
    oracle_root.mkdir()
    (oracle_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "seed": 1000,
                "task": "push-v3",
                "episodes": [
                    {"episode_index": 0, "reset_seed": 1000},
                    {"episode_index": 1, "reset_seed": 1001},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    def _fake_load_oracle_reference_frames(*_a: object, **_k: object) -> SimpleNamespace:
        return SimpleNamespace(
            goal_frame=__import__("numpy").zeros((2, 2, 3), dtype="uint8"),
            start_frame=__import__("numpy").zeros((2, 2, 3), dtype="uint8"),
            goal_frame_path=oracle_root / "frames" / "episode_0000" / "frame_000049.png",
        )

    def _fake_load_oracle_action_sequence(*_a: object, **_k: object) -> SimpleNamespace:
        import numpy as np

        return SimpleNamespace(
            actions=np.zeros((10, 4), dtype=np.float32),
            n_steps=10,
            action_source_path=oracle_root / "episodes" / "episode_0000" / "actions.jsonl",
        )

    def _fake_rollout_with_chunks(*args: object, **kwargs: object):
        calls.append({"args": args, "kwargs": kwargs})
        ep = EpisodeLog(
            episode_index=int(kwargs["episode_index"]),
            task=str(kwargs["task"]),
            carry_mode="sim",
            chunk_len=int(kwargs["chunk_len"]),
            num_candidates=int(kwargs["num_candidates"]),
            max_steps=int(kwargs["max_steps"]),
            goal_frame_index=int(kwargs["goal_frame_index"] or 0),
            goal_source=str(kwargs.get("goal_source") or ""),
            steps=int(kwargs["max_steps"]),
            done=True,
            latent_scores=[0.1],
            selected_scores=[-0.1],
            selected_indices=[0],
            selected_candidate_indices=[0],
            candidate_distances=[0.1],
            metadata={"oracle_action_mode": True},
        )
        return ep, None

    monkeypatch.setattr(
        "run_phase9_oracle_vs_wm.resolve_latest_oracle_pushv3_run",
        lambda *_a, **_k: oracle_root,
    )
    monkeypatch.setattr("run_phase9_oracle_vs_wm.load_oracle_reference_frames", _fake_load_oracle_reference_frames)
    monkeypatch.setattr("run_phase9_oracle_vs_wm.load_oracle_action_sequence", _fake_load_oracle_action_sequence)
    monkeypatch.setattr("run_phase9_oracle_vs_wm.rollout_with_chunks", _fake_rollout_with_chunks)
    monkeypatch.setattr(
        "run_phase9_oracle_vs_wm.ensure_unique_run_dir",
        lambda *_a, **_k: tmp_path / "run_phase9_test",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase9_oracle_vs_wm.py",
            "--oracle-run-root",
            str(oracle_root),
            "--artifacts-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "phase09"),
            "--episodes",
            "2",
            "--goal-frame-index",
            "50",
            "--max-steps",
            "50",
            "--chunk-len",
            "50",
            "--dry-run",
        ],
    )

    assert phase9_main() == 0
    assert len(calls) == 2
    for c in calls:
        assert c["args"][0] is None
        kw = c["kwargs"]
        assert kw["num_candidates"] == 1
        assert kw["carry_mode"] == "sim"
        assert kw["oracle_action_sequence"] is not None
        assert kw["oracle_action_source"]

    manifest_path = tmp_path / "run_phase9_test" / "segment_grpo_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["oracle_action_mode"] is True
    assert manifest["episodes"] == 2
    assert len(manifest["episodes_info"]) == 2
    assert manifest["episodes_info"][0]["effective_max_steps"] == 10
