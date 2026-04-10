import json
from pathlib import Path

import numpy as np
import pytest

from src.smolvla_pipeline import evaluator
from src.smolvla_pipeline.evaluator import EpisodeRollout, run_smolvla_eval, write_episode_artifacts


def test_write_episode_artifacts_outputs_logs_and_plot(tmp_path: Path):
    episode_dir = tmp_path / "episodes" / "episode_0000"
    actions = [[0.1, 0.2, 0.3, 0.4], [0.0, -0.1, 0.2, 0.5]]
    rewards = [0.3, 0.7]
    successes = [False, True]

    write_episode_artifacts(
        episode_dir=episode_dir,
        actions=actions,
        rewards=rewards,
        successes=successes,
    )

    actions_path = episode_dir / "actions.jsonl"
    lines = [line for line in actions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2

    first_line = json.loads(lines[0])
    assert first_line["step"] == 0
    assert first_line["reward"] == 0.3
    assert first_line["action"] == [0.1, 0.2, 0.3, 0.4]
    assert first_line["cumulative_reward"] == 0.3
    assert first_line["success"] is False

    assert (episode_dir / "reward_curve.csv").exists()
    assert (episode_dir / "reward_curve.png").exists()


def test_run_smolvla_eval_requires_video_enabled(tmp_path: Path):
    output_dir = tmp_path / "run_video_disabled"
    with pytest.raises(ValueError, match="video must be enabled"):
        run_smolvla_eval(
            task="push-v3",
            episodes=1,
            seed=42,
            checkpoint="jadechoghari/smolvla_metaworld",
            output_dir=output_dir,
            video=False,
            fps=30,
            overlay_mode="cumulative_reward",
            backend_factory=lambda **_kwargs: None,  # type: ignore[arg-type]
        )
    assert not output_dir.exists()


class _FakeBackend:
    def __init__(self):
        self.closed = False

    def rollout_episode(self, *, episode_index: int, reset_seed: int) -> EpisodeRollout:
        assert episode_index == 0
        assert reset_seed == 1000
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        return EpisodeRollout(
            actions=[[0.25, -0.1, 0.4, 0.0], [0.0, 0.2, -0.3, 0.1]],
            rewards=[0.3, 0.4],
            successes=[False, True],
            frames=[frame, frame],
            terminated=True,
            truncated=False,
        )

    def close(self) -> None:
        self.closed = True


def test_run_smolvla_eval_writes_real_flow_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_dir = tmp_path / "run_real_backend_contract"
    fake_backend = _FakeBackend()

    seen_overlay_modes: list[str] = []
    seen_frame_counts: list[int] = []

    def _fake_video_writer(
        *,
        video_path: Path,
        frames: list[np.ndarray],
        rewards: list[float],
        successes: list[bool],
        overlay_mode: str,
        fps: int,
    ) -> None:
        _ = (rewards, successes, fps)
        seen_overlay_modes.append(overlay_mode)
        seen_frame_counts.append(len(frames))
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"FAKE_MP4_DATA")

    monkeypatch.setattr(evaluator, "_write_episode_video", _fake_video_writer)

    result = run_smolvla_eval(
        task="push-v3",
        episodes=1,
        seed=1000,
        checkpoint="jadechoghari/smolvla_metaworld",
        output_dir=output_dir,
        video=True,
        fps=30,
        overlay_mode="cumulative_reward",
        backend_factory=lambda **_kwargs: fake_backend,
    )

    assert fake_backend.closed is True
    assert result["output_dir"] == str(output_dir.resolve())
    assert seen_overlay_modes == ["cumulative_reward"]
    assert seen_frame_counts == [2]

    eval_info = json.loads((output_dir / "eval_info.json").read_text(encoding="utf-8"))
    assert eval_info["overall"]["n_episodes"] == 1
    assert len(eval_info["overall"]["video_paths"]) == 1

    run_manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["runtime_backend"] == "lerobot_metaworld"
    assert run_manifest["camera_name"] == "corner2"
    assert run_manifest["flip_corner2"] is True
    episode = run_manifest["episodes"][0]
    assert episode["n_steps"] == 2
    assert episode["success"] is True
    assert episode["terminated"] is True
    assert episode["truncated"] is False

    actions_path = output_dir / episode["paths"]["actions"]
    assert actions_path.is_file()
    lines = [line for line in actions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2

    video_path = output_dir / episode["paths"]["video"]
    assert video_path.is_file()
    assert video_path.stat().st_size > 0


def test_run_smolvla_eval_passes_explicit_max_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class _MinimalBackend:
        def rollout_episode(self, *, episode_index: int, reset_seed: int) -> EpisodeRollout:
            _ = (episode_index, reset_seed)
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            return EpisodeRollout(
                actions=[[0.0, 0.0, 0.0, 0.0]],
                rewards=[0.0],
                successes=[False],
                frames=[frame, frame],
                terminated=False,
                truncated=False,
            )

        def close(self) -> None:
            return None

    seen_max_steps: list[int] = []

    def _backend_factory(**kwargs):
        seen_max_steps.append(int(kwargs["max_steps"]))
        return _MinimalBackend()

    def _fake_video_writer(
        *,
        video_path: Path,
        frames: list[np.ndarray],
        rewards: list[float],
        successes: list[bool],
        overlay_mode: str,
        fps: int,
    ) -> None:
        _ = (frames, rewards, successes, overlay_mode, fps)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"FAKE_MP4_DATA")

    monkeypatch.setattr(evaluator, "_write_episode_video", _fake_video_writer)

    result = run_smolvla_eval(
        task="push-v3",
        episodes=1,
        seed=1000,
        checkpoint="jadechoghari/smolvla_metaworld",
        output_dir=tmp_path / "run_explicit_max_steps",
        video=True,
        fps=30,
        overlay_mode="reward",
        max_steps=321,
        backend_factory=_backend_factory,
    )
    assert seen_max_steps == [321]
    run_manifest = result["run_manifest"]
    assert run_manifest["max_steps"] == 321


def test_coerce_exec_action_refuses_silent_resize():
    with pytest.raises(RuntimeError, match="Policy action dim mismatch"):
        evaluator._coerce_exec_action([0.1, 0.2], action_dim=4, np_module=np)

    action = evaluator._coerce_exec_action([2.0, -2.0, 0.5, -0.25], action_dim=4, np_module=np)
    assert action.tolist() == [1.0, -1.0, 0.5, -0.25]
