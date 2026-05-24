from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_split_tasks_for_five_shards_is_ten_each() -> None:
    from scripts.grpo.phase57_mt50_raw_vs_bounded_decode import MT50_TASKS, split_tasks_for_shard

    shards = [split_tasks_for_shard(MT50_TASKS, shard_index=i, shard_count=5) for i in range(5)]

    assert [len(x) for x in shards] == [10, 10, 10, 10, 10]
    assert shards[0][0] == "assembly-v3"
    assert shards[0][-1] == "coffee-pull-v3"
    assert shards[3] == [
        "pick-out-of-hole-v3",
        "pick-place-v3",
        "pick-place-wall-v3",
        "plate-slide-back-side-v3",
        "plate-slide-back-v3",
        "plate-slide-side-v3",
        "plate-slide-v3",
        "push-back-v3",
        "push-v3",
        "push-wall-v3",
    ]
    assert shards[4][-1] == "window-open-v3"


def test_phase57_parse_defaults(tmp_path: Path) -> None:
    from scripts.grpo.phase57_mt50_raw_vs_bounded_decode import parse_args

    args = parse_args(["--output-dir", str(tmp_path)])

    assert args.episodes == 25
    assert args.n_envs == 3
    assert args.chunk_len == 50
    assert args.max_steps == 180
    assert args.env_vector_mode == "async"
    assert args.goal_latent_mode == "visual_proprio"


def test_phase57_vector_helpers_use_requested_row() -> None:
    from scripts.grpo.phase57_mt50_raw_vs_bounded_decode import _agent_pos_from_vector_obs, _policy_rgb_from_vector_obs

    obs = {
        "pixels": np.stack(
            [
                np.full((2, 2, 3), 10, dtype=np.uint8),
                np.full((2, 2, 3), 200, dtype=np.uint8),
            ]
        ),
        "agent_pos": np.asarray([[1.0, 2.0, 3.0, 4.0], [9.0, 8.0, 7.0, 6.0]], dtype=np.float32),
    }

    assert int(_policy_rgb_from_vector_obs(obs, 1)[0, 0, 0]) == 200
    np.testing.assert_allclose(_agent_pos_from_vector_obs(obs, 1), np.asarray([9.0, 8.0, 7.0, 6.0]))


def test_phase57_vector_mode_steps_one_batch_per_chunk_step(tmp_path: Path, monkeypatch) -> None:
    from scripts.grpo import phase57_mt50_raw_vs_bounded_decode as mod

    envs = []

    class FakePolicy:
        def eval(self):
            return self

        def to(self, device):
            del device
            return self

    class FakeBundle:
        device = torch.device("cpu")

        def __init__(self):
            self.policy = FakePolicy()

    class FakeVectorEnv:
        action_dim = 4

        def __init__(self, *, task, n_envs=1, use_async_envs=False):
            assert task == "push-v3"
            self.n_envs = int(n_envs)
            self.use_async_envs = bool(use_async_envs)
            self.steps = 0
            self.step_batches = []
            self.inner = SimpleNamespace(single_action_space=SimpleNamespace(low=np.full(4, -1.0), high=np.full(4, 1.0)))
            envs.append(self)

        def reset_many(self, seeds):
            self.seeds = list(seeds)
            return {
                "pixels": np.zeros((self.n_envs, 2, 2, 3), dtype=np.uint8),
                "agent_pos": np.zeros((self.n_envs, 4), dtype=np.float32),
            }

        def build_proc(self, obs, *, bundle):
            del bundle
            return {"observation.state": torch.as_tensor(obs["agent_pos"], dtype=torch.float32), "task": ["push"] * self.n_envs}

        def step_batch(self, action_batch):
            self.step_batches.append(np.asarray(action_batch, dtype=np.float32).copy())
            self.steps += 1
            success = np.asarray([self.steps >= 1, self.steps >= 2], dtype=bool)
            return SimpleNamespace(
                observation={
                    "pixels": np.full((self.n_envs, 2, 2, 3), self.steps, dtype=np.uint8),
                    "agent_pos": np.full((self.n_envs, 4), self.steps, dtype=np.float32),
                },
                reward=np.ones((self.n_envs,), dtype=np.float32),
                success=success,
                terminated=success,
                truncated=np.zeros((self.n_envs,), dtype=bool),
            )

        def close(self):
            return None

    def fake_build_train_wrapper(args, bundle, action_dim):
        del args, bundle, action_dim
        return SimpleNamespace(_policy=FakePolicy()), []

    def fake_sample_chunk_batch(_wrapper, _proc, *, n_envs, chunk_len, seed, inference_mode):
        del seed, inference_mode
        actions = np.zeros((int(n_envs), int(chunk_len), 4), dtype=np.float32)
        actions[:, :, 0] = np.arange(1, int(n_envs) + 1, dtype=np.float32)[:, None]
        return SimpleNamespace(
            raw_postprocessed_action_np=actions.copy(),
            exec_action_np=actions,
        )

    monkeypatch.setattr("smolvla_grpo.lerobot_metaworld_adapter.OfficialLeRobotMetaWorldGRPORollout", FakeVectorEnv)
    monkeypatch.setattr("smolvla_grpo.lerobot_metaworld_adapter.resolve_lerobot_horizon", lambda env, max_steps: int(max_steps))
    monkeypatch.setattr(mod, "build_action_variants", lambda raw_actions, clipped_actions, action_low, action_high: SimpleNamespace(env_actions=clipped_actions, raw_wm_actions=raw_actions, bounded_wm_actions=clipped_actions, metadata={}))
    monkeypatch.setattr(mod, "_sample_chunk_batch", fake_sample_chunk_batch)
    monkeypatch.setattr("scripts.grpo.train_phase12_wm_chunk_grpo.build_train_wrapper", fake_build_train_wrapper)
    monkeypatch.setattr(mod, "unroll_phase12_latent_trace", lambda *args, **kwargs: SimpleNamespace(frames=[], wm_factor=1, structured_latents=[]))
    monkeypatch.setattr(mod, "aligned_real_indices", lambda **kwargs: [])
    monkeypatch.setattr(mod, "encode_real_latents_for_indices", lambda *args, **kwargs: [])
    monkeypatch.setattr(mod, "compute_raw_bounded_l2_metrics", lambda **kwargs: {"raw": {"combined_l2": []}, "bounded": {"combined_l2": []}})
    monkeypatch.setattr(mod, "write_three_row_decode_strip_with_l2", lambda path, **kwargs: path)
    monkeypatch.setattr(mod, "write_actions_npz", lambda path, **kwargs: path)
    monkeypatch.setattr(mod, "write_phase12_episode_video", lambda video_path, **kwargs: video_path)

    args = mod.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--task",
            "push-v3",
            "--episodes",
            "2",
            "--n-envs",
            "2",
            "--chunk-len",
            "2",
            "--max-steps",
            "2",
            "--env-vector-mode",
            "async",
        ]
    )
    summary = mod.run_task(
        args=args,
        task="push-v3",
        task_dir=tmp_path / "push-v3",
        bundle=FakeBundle(),
        wm_bundle=SimpleNamespace(),
        action_dim=4,
    )

    env = envs[0]
    assert env.use_async_envs is True
    assert len(env.step_batches) == 2
    np.testing.assert_allclose(env.step_batches[1][0], np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(env.step_batches[1][1], np.asarray([2.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert summary["env_vector_mode"] == "async"
    assert [row["n_steps"] for row in summary["episodes_rows"]] == [1, 2]


def test_merge_phase57_summary_counts_success_and_l2(tmp_path: Path) -> None:
    from scripts.grpo.merge_phase57_mt50_decode import build_merged_summary

    merged = build_merged_summary(
        parent=tmp_path,
        expected_tasks=2,
        expected_episodes=4,
        task_summaries=[
            {
                "task": "push-v3",
                "episodes_completed": 2,
                "pc_success": 50.0,
                "mean_raw_combined_l2": 1.0,
                "mean_bounded_combined_l2": 2.0,
                "metric_column_count": 10,
                "raw_win_fraction": 0.7,
                "bounded_win_fraction": 0.2,
                "tie_fraction": 0.1,
                "episodes_rows": [{"success": True}, {"success": False}],
            },
            {
                "task": "reach-v3",
                "episodes_completed": 2,
                "pc_success": 100.0,
                "mean_raw_combined_l2": 3.0,
                "mean_bounded_combined_l2": 1.0,
                "metric_column_count": 10,
                "raw_win_fraction": 0.3,
                "bounded_win_fraction": 0.6,
                "tie_fraction": 0.1,
                "episodes_rows": [{"success": True}, {"success": True}],
            },
        ],
    )

    assert merged["tasks_found"] == 2
    assert merged["episodes_found"] == 4
    assert merged["micro_pc_success"] == pytest.approx(75.0)
    assert merged["macro_pc_success"] == pytest.approx(75.0)
    assert merged["mean_raw_combined_l2"] == pytest.approx(2.0)
    assert merged["mean_bounded_combined_l2"] == pytest.approx(1.5)
    assert merged["raw_win_fraction"] == pytest.approx(0.5)
    assert merged["bounded_win_fraction"] == pytest.approx(0.4)


def test_merge_phase57_l2_means_are_weighted_by_metric_columns(tmp_path: Path) -> None:
    from scripts.grpo.merge_phase57_mt50_decode import build_merged_summary

    merged = build_merged_summary(
        parent=tmp_path,
        expected_tasks=2,
        expected_episodes=2,
        task_summaries=[
            {
                "task": "short-v3",
                "episodes_completed": 1,
                "metric_column_count": 1,
                "mean_raw_combined_l2": 100.0,
                "mean_bounded_combined_l2": 100.0,
                "episodes_rows": [{"success": True}],
            },
            {
                "task": "long-v3",
                "episodes_completed": 1,
                "metric_column_count": 9,
                "mean_raw_combined_l2": 0.0,
                "mean_bounded_combined_l2": 10.0,
                "episodes_rows": [{"success": True}],
            },
        ],
    )

    assert merged["mean_raw_combined_l2"] == pytest.approx(10.0)
    assert merged["mean_bounded_combined_l2"] == pytest.approx(19.0)
