from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import segment_grpo_loop as sgl  # noqa: E402
from segment_grpo_reference import load_oracle_goal_flat_obs  # noqa: E402


def test_load_oracle_goal_flat_obs_roundtrip(tmp_path: Path) -> None:
    run = tmp_path / "oracle"
    ep_dir = run / "episodes" / "episode_0000"
    ep_dir.mkdir(parents=True)
    path = ep_dir / "flat_obs.jsonl"
    want_idx = 49
    vec = np.arange(12, dtype=np.float32)
    with path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps({"frame_index": 0, "flat_obs": [0.0, 1.0]}) + "\n")
        fp.write(json.dumps({"frame_index": want_idx, "flat_obs": vec.tolist()}) + "\n")
    loaded = load_oracle_goal_flat_obs(run, 0, want_idx)
    assert loaded is not None
    np.testing.assert_array_equal(loaded, vec)


def test_load_oracle_goal_flat_obs_missing_file(tmp_path: Path) -> None:
    run = tmp_path / "oracle"
    (run / "episodes" / "episode_0000").mkdir(parents=True)
    assert load_oracle_goal_flat_obs(run, 0, 0) is None


def test_load_goal_latent_prefers_goal_proprio(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    recorded: dict[str, np.ndarray] = {}

    def _recorder(
        bundle: object, image: np.ndarray, proprio: np.ndarray, wm_scoring_latent: str = "visual"
    ) -> np.ndarray:
        recorded["proprio"] = np.asarray(proprio, dtype=np.float32).copy()
        return np.array([0.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(sgl, "_encode_state_to_latent", _recorder)
    wm = SimpleNamespace(proprio_dim=3)
    goal = np.zeros((2, 2, 3), dtype=np.uint8)
    fb = np.full((3,), 9.0, dtype=np.float32)
    gp = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    lat, _dbg = sgl._load_goal_latent(
        "",
        wm,
        None,
        fb,
        goal_frame=goal,
        goal_proprio=gp,
        wm_goal_flip_horizontal=False,
    )
    assert lat is not None
    np.testing.assert_array_equal(recorded["proprio"], gp)


def test_load_oracle_goal_flat_obs_no_matching_frame(tmp_path: Path) -> None:
    run = tmp_path / "oracle"
    ep_dir = run / "episodes" / "episode_0000"
    ep_dir.mkdir(parents=True)
    path = ep_dir / "flat_obs.jsonl"
    path.write_text(json.dumps({"frame_index": 0, "flat_obs": [1.0]}) + "\n", encoding="utf-8")
    assert load_oracle_goal_flat_obs(run, 0, 99) is None
