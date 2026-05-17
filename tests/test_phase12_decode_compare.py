from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_build_action_variants_executes_raw_and_decodes_raw_vs_clipped() -> None:
    from smolvla_grpo.phase12_decode_compare import build_action_variants

    raw = np.array([[2.0, -2.0, 0.25, 1.5]], dtype=np.float32)
    variants = build_action_variants(
        raw_actions=raw,
        clipped_actions=None,
        action_low=np.full((4,), -1.0, dtype=np.float32),
        action_high=np.full((4,), 1.0, dtype=np.float32),
    )

    np.testing.assert_allclose(variants.env_actions, raw)
    np.testing.assert_allclose(variants.raw_wm_actions, raw)
    np.testing.assert_allclose(variants.bounded_wm_actions, [[1.0, -1.0, 0.25, 1.0]])
    assert variants.metadata["env_action_source"] == "raw_postprocessed"
    assert variants.metadata["raw_action_max_abs"] == pytest.approx(2.0)
    assert variants.metadata["bounded_action_max_abs"] == pytest.approx(1.0)
    assert variants.metadata["clip_delta_max_abs"] == pytest.approx(1.0)


def test_three_row_strip_aligns_real_to_wm_steps(tmp_path: Path) -> None:
    from smolvla_grpo.phase12_decode_compare import align_decode_rows, write_three_row_decode_strip

    real = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(12)]
    raw = [np.full((2, 2, 3), 100 + i, dtype=np.uint8) for i in range(2)]
    bounded = [np.full((2, 2, 3), 200 + i, dtype=np.uint8) for i in range(2)]

    rows, indices = align_decode_rows(
        real_frames=real,
        raw_pred_frames=raw,
        bounded_pred_frames=bounded,
        env_steps_per_wm_step=5,
        carried_steps=10,
    )

    assert indices == [5, 10]
    assert [int(frame[0, 0, 0]) for frame in rows[0]] == [5, 10]
    assert [int(frame[0, 0, 0]) for frame in rows[1]] == [100, 101]
    assert [int(frame[0, 0, 0]) for frame in rows[2]] == [200, 201]

    out = write_three_row_decode_strip(
        tmp_path / "real_raw_bounded_decode_strip.png",
        real_frames=real,
        raw_pred_frames=raw,
        bounded_pred_frames=bounded,
        env_steps_per_wm_step=5,
        carried_steps=10,
    )
    assert out.exists()


def test_write_actions_npz_contains_raw_bounded_env(tmp_path: Path) -> None:
    from smolvla_grpo.phase12_decode_compare import write_actions_npz

    path = write_actions_npz(
        tmp_path / "actions_raw_bounded_env.npz",
        raw_actions=np.array([[2.0, -2.0, 0.0, 1.5]], dtype=np.float32),
        bounded_actions=np.array([[1.0, -1.0, 0.0, 1.0]], dtype=np.float32),
        env_actions=np.array([[2.0, -2.0, 0.0, 1.5]], dtype=np.float32),
    )

    data = np.load(path)
    np.testing.assert_allclose(data["raw_actions"], [[2.0, -2.0, 0.0, 1.5]])
    np.testing.assert_allclose(data["bounded_actions"], [[1.0, -1.0, 0.0, 1.0]])
    np.testing.assert_allclose(data["env_actions"], [[2.0, -2.0, 0.0, 1.5]])
    np.testing.assert_allclose(data["clip_delta"], [[1.0, -1.0, 0.0, 0.5]])


def test_build_summary_records_episode_outputs(tmp_path: Path) -> None:
    from smolvla_grpo.phase12_decode_compare import build_summary

    summary = build_summary(
        output_dir=tmp_path,
        args=SimpleNamespace(num_episodes=6, chunk_len=50, max_steps=120, train_seed_base=2000, task="push-v3"),
        episodes=[
            {
                "episode_index": 0,
                "reset_seed": 2000,
                "video_path": str(tmp_path / "episode_0000" / "selected_action_rollout.mp4"),
                "segments": [
                    {
                        "segment_index": 0,
                        "strip_path": str(
                            tmp_path / "episode_0000" / "segment_0000" / "real_raw_bounded_decode_strip.png"
                        ),
                        "actions_path": str(
                            tmp_path / "episode_0000" / "segment_0000" / "actions_raw_bounded_env.npz"
                        ),
                    }
                ],
            }
        ],
    )

    assert summary["comparison_type"] == "raw_vs_bounded_same_actions"
    assert summary["num_episodes"] == 6
    assert summary["episodes"][0]["reset_seed"] == 2000


def test_compare_decode_parse_defaults() -> None:
    from scripts.grpo.compare_phase12_raw_vs_bounded_decode import parse_args

    args = parse_args([])

    assert args.num_episodes == 6
    assert args.chunk_len == 50
    assert args.train_seed_base == 2000
    assert args.task == "push-v3"
    assert args.goal_latent_mode == "visual_proprio"
    assert str(args.output_dir).endswith("artifacts/phase12_raw_vs_bounded_decode/dry_run")


def test_train_script_uses_public_phase12_decoder() -> None:
    text = Path("scripts/grpo/train_phase12_wm_chunk_grpo.py").read_text(encoding="utf-8")

    assert "from smolvla_grpo.phase12_decode_compare import decode_phase12_prediction_frames" in text
    assert "_decode_phase12_prediction_frames" not in text


def test_run_episode_samples_once_executes_raw_and_decodes_same_root(monkeypatch, tmp_path: Path) -> None:
    from scripts.grpo import compare_phase12_raw_vs_bounded_decode as compare

    raw_actions = np.array([[2.0, -2.0, 0.5, 1.5], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    clipped_actions = np.clip(raw_actions, -1.0, 1.0)
    seen_steps: list[np.ndarray] = []
    seen_decodes: list[dict[str, object]] = []
    seen_policy_reset: list[bool] = []

    class DummyEnv:
        inner = SimpleNamespace(
            single_action_space=SimpleNamespace(
                low=np.full((4,), -1.0, dtype=np.float32),
                high=np.full((4,), 1.0, dtype=np.float32),
            )
        )

        def __init__(self) -> None:
            self._step = 0

        def reset(self, seed):
            del seed
            self._step = 0
            return {"pixels": np.zeros((1, 2, 2, 3), dtype=np.uint8)}

        def build_proc(self, obs, *, bundle):
            del obs, bundle
            return {"proc": "root"}

        def last_agent_pos(self):
            return np.array([float(self._step)], dtype=np.float32)

        def step(self, action_batch):
            seen_steps.append(np.asarray(action_batch, dtype=np.float32).reshape(-1).copy())
            self._step += 1
            obs = {"pixels": np.full((1, 2, 2, 3), self._step, dtype=np.uint8)}
            return SimpleNamespace(
                observation=obs,
                reward=1.0,
                success=False,
                terminated=self._step >= 1,
                truncated=False,
                info={},
            )

    class DummyWrapper:
        bundle = SimpleNamespace(device="cpu")

        class _Policy:
            def reset(self):
                seen_policy_reset.append(True)

        _policy = _Policy()

        def sample_action_chunk_from_proc(self, proc, *, chunk_len, rng):
            del proc, chunk_len, rng
            return SimpleNamespace(
                raw_postprocessed_action_np=raw_actions,
                exec_action_np=clipped_actions,
            )

    monkeypatch.setattr(
        compare,
        "decode_phase12_prediction_frames",
        lambda wm_bundle, *, image, proprio, actions, mode: seen_decodes.append(
            {
                "image": np.asarray(image).copy(),
                "proprio": np.asarray(proprio).copy(),
                "actions": np.asarray(actions).copy(),
                "mode": mode,
            }
        )
        or [np.zeros((2, 2, 3), dtype=np.uint8)],
    )
    monkeypatch.setattr(
        compare,
        "write_phase12_episode_video",
        lambda *, video_path, frames, rewards, successes, fps: Path(video_path).write_bytes(b"video") or Path(video_path),
    )

    episode = compare.run_episode(
        args=SimpleNamespace(
            train_seed_base=2000,
            chunk_len=2,
            max_steps=1,
            old_policy_inference_mode=True,
            goal_latent_mode="visual_proprio",
            fps=10,
        ),
        env_h=DummyEnv(),
        wrapper=DummyWrapper(),
        wm_bundle=SimpleNamespace(model=SimpleNamespace(action_dim=4), planner_action_dim=4),
        action_dim=4,
        episode_index=0,
        output_dir=tmp_path,
    )

    assert episode["segment_count"] == 1
    assert seen_policy_reset == [True]
    assert len(seen_steps) == 1
    np.testing.assert_allclose(seen_steps[0], raw_actions[0])
    assert len(seen_decodes) == 2
    np.testing.assert_allclose(seen_decodes[0]["actions"], raw_actions[:1])
    np.testing.assert_allclose(seen_decodes[1]["actions"], clipped_actions[:1])
    np.testing.assert_allclose(seen_decodes[0]["image"], seen_decodes[1]["image"])
    np.testing.assert_allclose(seen_decodes[0]["proprio"], seen_decodes[1]["proprio"])
    assert (tmp_path / "episode_0000" / "segment_0000" / "actions_raw_bounded_env.npz").exists()
