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


def test_run_smolvla_eval_video_disabled_skips_mp4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_dir = tmp_path / "run_video_disabled"
    fake_backend = _FakeBackend()
    called_video: list[bool] = []

    def _no_video_writer(**kwargs: object) -> None:  # type: ignore[no-untyped-def]
        called_video.append(True)

    monkeypatch.setattr(evaluator, "_write_episode_video", _no_video_writer)

    run_smolvla_eval(
        task="push-v3",
        episodes=1,
        seed=1000,
        checkpoint="jadechoghari/smolvla_metaworld",
        output_dir=output_dir,
        video=False,
        fps=30,
        overlay_mode="cumulative_reward",
        save_actions=False,
        backend_factory=lambda **_kwargs: fake_backend,
    )

    assert called_video == []
    eval_info = json.loads((output_dir / "eval_info.json").read_text(encoding="utf-8"))
    assert eval_info["overall"]["video_paths"] == []
    run_manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["video_enabled"] is False
    assert run_manifest["save_actions"] is False
    ep0 = run_manifest["episodes"][0]
    assert "video" not in ep0["paths"]
    assert (output_dir / ep0["paths"]["reward_curve_csv"]).is_file()
    assert (output_dir / ep0["paths"]["reward_curve_png"]).is_file()


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
        save_actions=True,
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
    assert run_manifest["save_frames"] is False
    assert run_manifest["camera_name"] == "corner2"
    assert run_manifest["flip_corner2"] is True
    episode = run_manifest["episodes"][0]
    assert episode["n_steps"] == 2
    assert episode["success"] is True
    assert episode["success_any"] is True
    assert episode["success_last"] is True
    assert episode["first_success_step"] == 1
    assert episode["terminated"] is True
    assert episode["truncated"] is False

    actions_path = output_dir / episode["paths"]["actions"]
    assert actions_path.is_file()
    lines = [line for line in actions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2

    video_path = output_dir / episode["paths"]["video"]
    assert video_path.is_file()
    assert video_path.stat().st_size > 0

    progress_lines = [
        line
        for line in (output_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(progress_lines) == 1
    progress_row = json.loads(progress_lines[0])
    assert progress_row["episode_index"] == 0
    assert progress_row["episodes_total"] == 1
    assert progress_row["success"] is True
    assert progress_row["success_any"] is True
    assert progress_row["success_last"] is True
    assert progress_row["first_success_step"] == 1


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


def test_run_smolvla_eval_save_frames_writes_pngs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_backend = _FakeBackend()

    def _fake_write_frames(*, frames_dir: Path, frames: list) -> None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(len(frames)):
            (frames_dir / f"frame_{idx:06d}.png").write_bytes(b"fakepng")

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
    monkeypatch.setattr(evaluator, "_write_episode_frames_png", _fake_write_frames)

    output_dir = tmp_path / "run_save_frames"
    run_smolvla_eval(
        task="push-v3",
        episodes=1,
        seed=1000,
        checkpoint="jadechoghari/smolvla_metaworld",
        output_dir=output_dir,
        video=True,
        fps=30,
        overlay_mode="cumulative_reward",
        save_frames=True,
        backend_factory=lambda **_kwargs: fake_backend,
    )

    frames_dir = output_dir / "frames" / "episode_0000"
    assert frames_dir.is_dir()
    assert (frames_dir / "frame_000000.png").is_file()
    assert (frames_dir / "frame_000001.png").is_file()

    meta = json.loads(
        (output_dir / "episodes" / "episode_0000" / "episode_meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta["paths"]["frames_dir"] == "frames/episode_0000"

    run_manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["save_frames"] is True


def test_coerce_exec_action_refuses_silent_resize():
    with pytest.raises(RuntimeError, match="Policy action dim mismatch"):
        evaluator._coerce_exec_action([0.1, 0.2], action_dim=4, np_module=np)

    action = evaluator._coerce_exec_action([2.0, -2.0, 0.5, -0.25], action_dim=4, np_module=np)
    assert action.tolist() == [1.0, -1.0, 0.5, -0.25]


def test_build_overlay_text_uses_single_primary_metric_label():
    text = evaluator._build_overlay_text(
        step=5,
        reward=0.12,
        cumulative_reward=0.54,
        reward_delta=-0.03,
        success=True,
        overlay_mode="cumulative_reward",
    )
    tokens = text.split()
    assert sum(token.startswith("cumulative_reward=") for token in tokens) == 1
    assert "metric=0.5400" in text

    text_reward = evaluator._build_overlay_text(
        step=5,
        reward=0.12,
        cumulative_reward=0.54,
        reward_delta=-0.03,
        success=True,
        overlay_mode="reward",
    )
    reward_tokens = text_reward.split()
    assert sum(token.startswith("reward=") for token in reward_tokens) == 1
    assert "metric=0.1200" in text_reward


def test_validate_overlay_mode_accepts_reward_delta():
    assert evaluator._validate_overlay_mode("reward_delta") == "reward_delta"


def test_resolve_task_text_override_wins():
    assert (
        evaluator._resolve_task_text("reach-v3", override="Move to the goal of the light red sphere")
        == "Move to the goal of the light red sphere"
    )
    assert evaluator._resolve_task_text("push-v3", override="  Custom prompt  ") == "Custom prompt"


def test_run_smolvla_eval_task_text_override_in_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_dir = tmp_path / "run_custom_prompt"
    fake_backend = _FakeBackend()
    custom = "Move to the goal of the light red sphere"

    def _fake_video_writer(
        *,
        video_path: Path,
        frames: list[np.ndarray],
        rewards: list[float],
        successes: list[bool],
        overlay_mode: str,
        fps: int,
    ) -> None:
        _ = (rewards, successes, overlay_mode, fps)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"FAKE_MP4_DATA")

    monkeypatch.setattr(evaluator, "_write_episode_video", _fake_video_writer)

    run_smolvla_eval(
        task="reach-v3",
        episodes=1,
        seed=1000,
        checkpoint="jadechoghari/smolvla_metaworld",
        output_dir=output_dir,
        video=True,
        fps=30,
        overlay_mode="cumulative_reward",
        task_text=custom,
        backend_factory=lambda **_kwargs: fake_backend,
    )

    run_manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["task_text"] == custom
    assert run_manifest["task"] == "reach-v3"


def test_run_smolvla_eval_transient_success_uses_success_any(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class _TransientBackend:
        def rollout_episode(self, *, episode_index: int, reset_seed: int) -> EpisodeRollout:
            _ = (episode_index, reset_seed)
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            return EpisodeRollout(
                actions=[[0.0, 0.0, 0.0, 0.0] for _ in range(4)],
                rewards=[0.0, 1.0, 0.5, 0.1],
                successes=[False, True, True, False],
                frames=[frame, frame, frame, frame],
                terminated=False,
                truncated=False,
            )

        def close(self) -> None:
            return None

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

    output_dir = tmp_path / "run_transient_success"
    run_smolvla_eval(
        task="reach-v3",
        episodes=1,
        seed=1000,
        checkpoint="jadechoghari/smolvla_metaworld",
        output_dir=output_dir,
        video=True,
        fps=30,
        overlay_mode="reward_delta",
        backend_factory=lambda **_kwargs: _TransientBackend(),
    )

    eval_info = json.loads((output_dir / "eval_info.json").read_text(encoding="utf-8"))
    assert eval_info["overall"]["pc_success"] == 100.0

    run_manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    episode = run_manifest["episodes"][0]
    assert episode["success"] is True
    assert episode["success_any"] is True
    assert episode["success_last"] is False
    assert episode["first_success_step"] == 1


def test_run_smolvla_eval_respects_fixed_reset_seed_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    seen_reset_seeds: list[int] = []

    class _SeedRecordingBackend:
        def rollout_episode(self, *, episode_index: int, reset_seed: int) -> EpisodeRollout:
            _ = episode_index
            seen_reset_seeds.append(int(reset_seed))
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
    monkeypatch.setenv("SMOLVLA_FIXED_RESET_SEED", "4242")

    run_smolvla_eval(
        task="push-v3",
        episodes=3,
        seed=1000,
        checkpoint="jadechoghari/smolvla_metaworld",
        output_dir=tmp_path / "run_fixed_seed",
        video=True,
        fps=30,
        overlay_mode="cumulative_reward",
        backend_factory=lambda **_kwargs: _SeedRecordingBackend(),
    )

    assert seen_reset_seeds == [4242, 4242, 4242]
    run_manifest = json.loads(
        (tmp_path / "run_fixed_seed" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert [int(ep["reset_seed"]) for ep in run_manifest["episodes"]] == [4242, 4242, 4242]


def test_lerobot_backend_seeds_before_set_task_then_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeEnv:
        def set_task(self, task: object) -> None:
            calls.append(("set_task", task))

    backend = evaluator._LeRobotMetaWorldBackend.__new__(evaluator._LeRobotMetaWorldBackend)
    backend._env = FakeEnv()
    backend._tasks = ["task0"]
    backend._target_episode_index_override = None
    backend._max_steps = 0
    backend._np = np

    def fake_seed(seed: int) -> None:
        calls.append(("seed", int(seed)))

    def fake_reset(self: object, reset_seed: int) -> tuple[dict[str, bool], dict[str, bool]]:
        calls.append(("reset", int(reset_seed)))
        return ({"ok": True}, {"info": True})

    monkeypatch.setattr(evaluator, "seed_metaworld_process", fake_seed)
    monkeypatch.setattr(evaluator._LeRobotMetaWorldBackend, "_reset", fake_reset)
    monkeypatch.setattr(evaluator._LeRobotMetaWorldBackend, "_render_frame", lambda self: None)

    rollout = backend.rollout_episode(episode_index=0, reset_seed=7)

    assert calls == [("seed", 7), ("set_task", "task0"), ("reset", 7)]
    assert rollout.actions == []
    assert rollout.rewards == []


def test_lerobot_backend_reset_delegates_to_gymnasium_reset_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = object()
    calls: list[tuple[int, int]] = []

    backend = evaluator._LeRobotMetaWorldBackend.__new__(evaluator._LeRobotMetaWorldBackend)
    backend._env = env

    def fake_strict(seen_env: object, seed: int) -> tuple[dict[str, int], dict[str, bool]]:
        calls.append((id(seen_env), int(seed)))
        return ({"obs": 1}, {"ok": True})

    monkeypatch.setattr(evaluator, "gymnasium_reset_strict", fake_strict)

    obs, info = backend._reset(11)

    assert calls == [(id(env), 11)]
    assert obs == {"obs": 1}
    assert info == {"ok": True}
