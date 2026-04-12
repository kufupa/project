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

from run_segment_grpo import _parse_args  # noqa: E402
from segment_grpo_loop import WMBundle, rollout_with_chunks  # noqa: E402


def test_parse_args_accepts_strict_controls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_segment_grpo.py",
            "--output-json",
            str(tmp_path / "out.json"),
            "--strict-wm-scoring",
            "--strict-decode",
            "--wm-scoring-latent",
            "concat",
        ],
    )
    args = _parse_args()
    assert args.strict_wm_scoring is True
    assert args.strict_decode is True
    assert args.wm_scoring_latent == "concat"


def _make_goal_latent_file(path: Path) -> None:
    path.write_text(json.dumps({"latent": [0.1, 0.2, 0.3]}), encoding="utf-8")


def test_rollout_with_chunks_records_fallback_metadata_for_non_strict_scoring(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class _ScoringFailureModel:
        def encode(self, *_args: object, **_kwargs: object) -> object:  # type: ignore[override]
            raise RuntimeError("scoring failed")

        def unroll(self, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError("should not be called")

    wm_bundle = WMBundle(
        model=_ScoringFailureModel(),
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    goal_path = tmp_path / "goal.json"
    _make_goal_latent_file(goal_path)

    episode, _ = rollout_with_chunks(
        smolvla_bundle=None,
        wm_bundle=wm_bundle,
        task="push-v3",
        episode_index=0,
        chunk_len=4,
        num_candidates=2,
        max_steps=8,
        carry_mode="replay",
        goal_latent_source=str(goal_path),
        seed=123,
        dry_run=True,
        strict_wm_scoring=False,
        strict_decode=False,
        wm_rollout_mode="iterative",
    )

    assert episode.metadata.get("strict_wm_scoring") is False
    assert episode.metadata.get("strict_decode") is False
    assert episode.metadata["wm_scoring_statuses"]
    assert episode.metadata["wm_scoring_statuses"][0] == "fallback"
    assert episode.metadata["decode_statuses"][0] == "failed"
    selected_segment = episode.segments[0]
    assert selected_segment.metadata["wm_scoring_status"] == "fallback"
    assert selected_segment.metadata["decode_status"] == "failed"
    selected_candidate = selected_segment.candidates[selected_segment.selected_index]
    assert selected_candidate.meta["wm_scoring_status"] == "fallback"
    assert selected_candidate.meta["scoring_failure_reason"] is not None


def test_rollout_with_chunks_strict_wm_scoring_raises(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class _ScoringFailureModel:
        def encode(self, *_args: object, **_kwargs: object) -> object:  # type: ignore[override]
            raise RuntimeError("scoring failed")

        def unroll(self, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError("should not be called")

    wm_bundle = WMBundle(
        model=_ScoringFailureModel(),
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    goal_path = tmp_path / "goal.json"
    _make_goal_latent_file(goal_path)

    with pytest.raises(RuntimeError, match="scoring failed"):
        rollout_with_chunks(
            smolvla_bundle=None,
            wm_bundle=wm_bundle,
            task="push-v3",
            episode_index=0,
            chunk_len=4,
            num_candidates=2,
            max_steps=4,
            carry_mode="replay",
            goal_latent_source=str(goal_path),
            seed=123,
            dry_run=True,
            strict_wm_scoring=True,
            strict_decode=False,
            wm_rollout_mode="iterative",
        )


def test_rollout_with_chunks_strict_decode_raises(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class _DecodeFailureModel:
        def encode(self, *_args: object, **_kwargs: object) -> object:
            return torch.ones((1, 1, 3), dtype=torch.float32)

        def unroll(self, *_args: object, **_kwargs: object) -> object:
            act_suffix = _args[1] if len(_args) > 1 else None
            steps = int(act_suffix.shape[0]) if hasattr(act_suffix, "shape") else 1
            return torch.ones((steps, 1, 3), dtype=torch.float32)

    wm_bundle = WMBundle(
        model=_DecodeFailureModel(),
        preprocessor=SimpleNamespace(),
        proprio_dim=4,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )
    goal_path = tmp_path / "goal.json"
    _make_goal_latent_file(goal_path)

    with pytest.raises(RuntimeError, match="Decode failed for selected candidate"):
        rollout_with_chunks(
            smolvla_bundle=None,
            wm_bundle=wm_bundle,
            task="push-v3",
            episode_index=0,
            chunk_len=4,
            num_candidates=2,
            max_steps=8,
            carry_mode="replay",
            goal_latent_source=str(goal_path),
            seed=123,
            dry_run=True,
            strict_wm_scoring=False,
            strict_decode=True,
            wm_rollout_mode="iterative",
        )


def test_main_manifest_includes_strict_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from run_segment_grpo import main as segment_main

    class _FakeEpisode:
        def __init__(self, idx: int) -> None:
            self.goal_frame_index = 25
            self.goal_source = f"episode_{idx}"
            self.start_frame_similarity = 0.0
            self.reset_frame_warning = False
            self.steps = 0
            self.done = True
            self.selected_indices = [0]
            self.selected_candidate_indices = [0]
            self.latent_scores = [0.0]
            self.selected_scores = [0.0]
            self.comparison_strip_path = None
            self.comparison_video_path = None

        def to_dict(self) -> dict[str, object]:
            return {"goal_frame_index": self.goal_frame_index}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_segment_grpo.py",
            "--episodes",
            "2",
            "--output-json",
            str(tmp_path / "segment_grpo_default2.json"),
            "--dry-run",
            "--strict-wm-scoring",
            "--strict-decode",
            "--wm-scoring-latent",
            "proprio",
        ],
    )

    manifest_path = tmp_path / "segment_grpo_manifest.json"
    captured: dict[Path, dict[str, object]] = {}

    def _fake_resolve_oracle_run(*_args: object, **_kwargs: object) -> None:
        return None

    calls: list[tuple[bool, bool, str]] = []

    def _fake_rollout_with_chunks(*_args: object, **kwargs: object):
        calls.append(
            (
                bool(kwargs["strict_wm_scoring"]),
                bool(kwargs["strict_decode"]),
                str(kwargs["wm_scoring_latent"]),
            )
        )
        episode_index = int(kwargs["episode_index"])
        return _FakeEpisode(episode_index), None

    def _fake_write_json(path: Path, payload: dict[str, object]) -> None:
        captured[path] = payload

    monkeypatch.setattr("run_segment_grpo.resolve_latest_oracle_pushv3_run", _fake_resolve_oracle_run)
    monkeypatch.setattr("run_segment_grpo.load_wm_bundle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("run_segment_grpo.rollout_with_chunks", _fake_rollout_with_chunks)
    monkeypatch.setattr("run_segment_grpo._write_json", _fake_write_json)
    monkeypatch.setattr("run_segment_grpo.load_oracle_reference_frames", lambda *_args, **_kwargs: None)

    code = segment_main()
    assert code == 0
    assert manifest_path in captured
    manifest = captured[manifest_path]
    assert manifest["strict_wm_scoring"] is True
    assert manifest["strict_decode"] is True
    assert manifest["wm_scoring_latent"] == "proprio"
    assert manifest["episodes"] == 2
    assert len(manifest["episodes_info"]) == 2  # type: ignore[arg-type]
    assert all(item == (True, True, "proprio") for item in calls)
