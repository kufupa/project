from __future__ import annotations

import numpy as np
import torch

from smolvla_grpo import phase11_rollout
from smolvla_grpo.policy_wrapper import MetaWorldSmolVLAGRPOPolicy, concatenate_proc_rows


class _DummyPostprocessor:
    def __init__(self) -> None:
        self.rows: list[torch.Tensor] = []

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        self.rows.append(action.detach().clone())
        return action


class _DummyBundle:
    obs_image_key = "observation.image"
    obs_state_key = "observation.state"
    obs_env_state_key = "observation.environment_state"
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.postprocessor = _DummyPostprocessor()


class _DummyPolicy:
    def __init__(self, *, action_dim: int = 4) -> None:
        self.action_dim = int(action_dim)
        self.chunk_calls: list[int] = []
        self.select_calls = 0

    def _get_distr_params_chunk(self, proc):
        chunk_len = int(getattr(self, "configured_chunk_len", 50))
        self.chunk_calls.append(chunk_len)
        mean = torch.arange(
            int(chunk_len) * self.action_dim,
            dtype=torch.float32,
        ).reshape(int(chunk_len), self.action_dim)
        mean = mean / 100.0
        log_std = torch.full_like(mean, -0.5)
        return mean, log_std

    def select_action_distr_params(self, proc):
        self.select_calls += 1
        b = int(proc["x"].shape[0])
        mean = torch.zeros((b, self.action_dim), dtype=torch.float32)
        log_std = torch.full_like(mean, -0.5)
        return mean, log_std

    def eval(self):
        return None


class _DummyBatchChunkPolicy(_DummyPolicy):
    def _get_distr_params_chunk(self, proc):
        b = int(proc["x"].shape[0])
        t = int(getattr(self, "configured_chunk_len", 3))
        step_base = torch.arange(t * self.action_dim, dtype=torch.float32).reshape(1, t, self.action_dim)
        row_offset = proc["x"].reshape(b, -1)[:, :1].reshape(b, 1, 1).float()
        base = step_base + row_offset
        return base / 100.0, torch.full_like(base, -0.5)


class _DummyEnv:
    def __init__(self, *, task: str) -> None:
        self.inner = self
        self.action_space = type(
            "ActionSpace",
            (),
            {"shape": (4,), "low": np.full((4,), -1.0), "high": np.full((4,), 1.0)},
        )()

    def close(self) -> None:
        return None


def _wrapper() -> tuple[MetaWorldSmolVLAGRPOPolicy, _DummyPolicy, _DummyBundle]:
    bundle = _DummyBundle()
    policy = _DummyPolicy(action_dim=4)
    wrapper = MetaWorldSmolVLAGRPOPolicy(
        bundle,
        task="push-v3",
        task_text="push",
        camera_name="corner2",
        flip_corner2=False,
        action_dim=4,
        policy_module=policy,
        action_transform="no_tanh",
        action_low=np.full((4,), -10.0, dtype=np.float32),
        action_high=np.full((4,), 10.0, dtype=np.float32),
    )
    return wrapper, policy, bundle


def _batch_wrapper() -> tuple[MetaWorldSmolVLAGRPOPolicy, _DummyBatchChunkPolicy, _DummyBundle]:
    bundle = _DummyBundle()
    policy = _DummyBatchChunkPolicy(action_dim=4)
    wrapper = MetaWorldSmolVLAGRPOPolicy(
        bundle,
        task="push-v3",
        task_text="push",
        camera_name="corner2",
        flip_corner2=False,
        action_dim=4,
        policy_module=policy,
        action_transform="no_tanh",
        action_low=np.full((4,), -10.0, dtype=np.float32),
        action_high=np.full((4,), 10.0, dtype=np.float32),
    )
    return wrapper, policy, bundle


def test_load_bundle_for_grpo_threads_n_action_steps(monkeypatch) -> None:
    calls: list[int] = []

    def fake_load_bundle(checkpoint: str, *, n_action_steps: int = 1):
        calls.append(int(n_action_steps))
        return object()

    monkeypatch.setattr(phase11_rollout, "_load_smolvla_bundle", fake_load_bundle)
    monkeypatch.setattr(phase11_rollout, "PushV3GRPOEnv", _DummyEnv)

    bundle, action_dim = phase11_rollout.load_bundle_for_grpo(
        "checkpoint",
        task="push-v3",
        env_backend="custom",
        n_action_steps=25,
    )

    assert bundle is not None
    assert action_dim == 4
    assert calls == [25]


def test_load_bundle_for_grpo_default_stays_one(monkeypatch) -> None:
    calls: list[int] = []

    def fake_load_bundle(checkpoint: str, *, n_action_steps: int = 1):
        calls.append(int(n_action_steps))
        return object()

    monkeypatch.setattr(phase11_rollout, "_load_smolvla_bundle", fake_load_bundle)
    monkeypatch.setattr(phase11_rollout, "PushV3GRPOEnv", _DummyEnv)

    bundle, action_dim = phase11_rollout.load_bundle_for_grpo(
        "checkpoint",
        task="push-v3",
        env_backend="custom",
    )

    assert bundle is not None
    assert action_dim == 4
    assert calls == [1]


def test_sample_action_chunk_from_proc_returns_full_chunk_shapes() -> None:
    wrapper, policy, _bundle = _wrapper()
    rng = torch.Generator(device="cpu").manual_seed(123)

    chunk = wrapper.sample_action_chunk_from_proc(
        {"x": torch.zeros(1, 1)},
        chunk_len=25,
        rng=rng,
    )

    assert chunk.exec_action_np.shape == (25, 4)
    assert chunk.raw_postprocessed_action_np.shape == (25, 4)
    assert chunk.policy_tensor.shape == (25, 4)
    assert chunk.unsquashed_chunk.shape == (25, 4)
    assert chunk.log_prob_steps.shape == (25,)
    assert chunk.log_prob_sum.shape == ()
    assert chunk.action_clip_fraction.shape == (25,)
    assert chunk.action_clip_any.shape == (25,)
    assert isinstance(chunk.unique_action_rows, int)
    assert chunk.unique_action_rows >= 1
    assert policy.chunk_calls == [50]
    assert policy.select_calls == 0


def test_sample_action_chunk_preserves_raw_postprocessed_actions_before_clip() -> None:
    wrapper, _policy, _bundle = _wrapper()
    wrapper.action_low = np.full((4,), -1.0, dtype=np.float32)
    wrapper.action_high = np.full((4,), 1.0, dtype=np.float32)

    chunk = wrapper.sample_action_chunk_from_proc(
        {"x": torch.zeros(1, 1)},
        chunk_len=25,
        rng=torch.Generator(device="cpu").manual_seed(123),
    )

    assert chunk.raw_postprocessed_action_np.shape == (25, 4)
    assert chunk.exec_action_np.shape == (25, 4)
    assert np.max(np.abs(chunk.raw_postprocessed_action_np)) > 1.0
    assert np.max(chunk.exec_action_np) <= 1.0
    assert np.any(chunk.raw_postprocessed_action_np != chunk.exec_action_np)


def test_sample_action_chunk_slices_installed_lerobot_full_chunk() -> None:
    wrapper, policy, _bundle = _wrapper()
    policy.configured_chunk_len = 50
    rng = torch.Generator(device="cpu").manual_seed(123)

    chunk = wrapper.sample_action_chunk_from_proc(
        {"x": torch.zeros(1, 1)},
        chunk_len=25,
        rng=rng,
    )

    assert chunk.unsquashed_chunk.shape == (25, 4)
    assert policy.chunk_calls == [50]


def test_sample_action_chunk_from_proc_supports_chunk_len_five() -> None:
    wrapper, policy, _bundle = _wrapper()
    policy.configured_chunk_len = 5

    chunk = wrapper.sample_action_chunk_from_proc(
        {"x": torch.zeros(1, 1)},
        chunk_len=5,
        rng=torch.Generator(device="cpu").manual_seed(123),
    )

    assert chunk.exec_action_np.shape == (5, 4)
    assert chunk.unsquashed_chunk.shape == (5, 4)
    assert chunk.log_prob_steps.shape == (5,)
    recomputed = wrapper.get_action_probs_for_chunk_from_proc(
        {"x": torch.zeros(1, 1)},
        chunk.unsquashed_chunk,
    )
    torch.testing.assert_close(recomputed, chunk.log_prob_steps)
    assert policy.chunk_calls == [5, 5]


def test_get_action_probs_for_chunk_from_proc_recomputes_per_step_logprobs() -> None:
    wrapper, _policy, _bundle = _wrapper()
    rng = torch.Generator(device="cpu").manual_seed(123)
    proc = {"x": torch.zeros(1, 1)}
    chunk = wrapper.sample_action_chunk_from_proc(proc, chunk_len=25, rng=rng)

    log_probs = wrapper.get_action_probs_for_chunk_from_proc(proc, chunk.unsquashed_chunk)

    assert log_probs.shape == (25,)
    torch.testing.assert_close(log_probs, chunk.log_prob_steps)


def test_two_chunks_from_same_root_can_differ_without_action_queue() -> None:
    wrapper, _policy, _bundle = _wrapper()
    proc = {"x": torch.zeros(1, 1)}

    first = wrapper.sample_action_chunk_from_proc(
        proc,
        chunk_len=25,
        rng=torch.Generator(device="cpu").manual_seed(123),
    )
    second = wrapper.sample_action_chunk_from_proc(
        proc,
        chunk_len=25,
        rng=torch.Generator(device="cpu").manual_seed(456),
    )

    assert not torch.allclose(first.unsquashed_chunk, second.unsquashed_chunk)


def test_sample_action_chunk_batch_from_proc_returns_batch_shapes() -> None:
    wrapper, _policy, _bundle = _batch_wrapper()

    chunk = wrapper.sample_action_chunk_batch_from_proc(
        {"x": torch.zeros(2, 1)},
        n_envs=2,
        chunk_len=3,
        reset_seed=123,
    )

    assert chunk.exec_action_np.shape == (2, 3, 4)
    assert chunk.raw_postprocessed_action_np.shape == (2, 3, 4)
    assert chunk.policy_tensor.shape == (2, 3, 4)
    assert chunk.unsquashed_chunk.shape == (2, 3, 4)
    assert chunk.log_prob_steps.shape == (2, 3)
    assert chunk.log_prob_sum.shape == (2,)
    assert chunk.action_clip_fraction.shape == (2, 3)
    assert chunk.action_clip_any.shape == (2, 3)


def test_reshape_chunk_params_batch_slices_flat_full_chunks_per_env() -> None:
    wrapper, _policy, _bundle = _batch_wrapper()
    base = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
    flat = base.reshape(2 * 5, 4)

    mean, log_std = wrapper._reshape_chunk_params_batch(
        flat,
        torch.full_like(flat, -0.5),
        n_envs=2,
        chunk_len=2,
    )

    assert mean.shape == (2, 2, 4)
    torch.testing.assert_close(mean[0], base[0, :2])
    torch.testing.assert_close(mean[1], base[1, :2])
    torch.testing.assert_close(log_std, torch.full((2, 2, 4), -0.5))


def test_sample_action_chunk_batch_is_seed_deterministic_per_row() -> None:
    wrapper, _policy, _bundle = _batch_wrapper()
    proc = {"x": torch.zeros(2, 1)}

    first = wrapper.sample_action_chunk_batch_from_proc(proc, n_envs=2, chunk_len=3, reset_seed=123)
    second = wrapper.sample_action_chunk_batch_from_proc(proc, n_envs=2, chunk_len=3, reset_seed=123)
    third = wrapper.sample_action_chunk_batch_from_proc(proc, n_envs=2, chunk_len=3, reset_seed=124)

    torch.testing.assert_close(first.unsquashed_chunk, second.unsquashed_chunk)
    assert not torch.allclose(first.unsquashed_chunk, third.unsquashed_chunk)


def test_sample_action_chunk_batch_preserves_raw_before_clip() -> None:
    wrapper, _policy, _bundle = _batch_wrapper()
    wrapper.action_low = np.full((4,), -1.0, dtype=np.float32)
    wrapper.action_high = np.full((4,), 1.0, dtype=np.float32)

    chunk = wrapper.sample_action_chunk_batch_from_proc(
        {"x": torch.zeros(2, 1)},
        n_envs=2,
        chunk_len=3,
        reset_seed=123,
    )

    assert np.max(np.abs(chunk.raw_postprocessed_action_np)) > 1.0
    assert np.max(chunk.exec_action_np) <= 1.0
    assert np.any(chunk.raw_postprocessed_action_np != chunk.exec_action_np)


def test_concatenate_proc_rows_handles_tensor_numpy_task_and_scalars() -> None:
    rows = [
        {
            "x": torch.tensor([[1.0, 2.0]]),
            "arr": np.asarray([[1, 2]], dtype=np.int64),
            "task": ["push"],
            "meta": "a",
        },
        {
            "x": torch.tensor([[3.0, 4.0]]),
            "arr": np.asarray([[3, 4]], dtype=np.int64),
            "task": ["push"],
            "meta": "b",
        },
    ]

    proc = concatenate_proc_rows(rows)

    torch.testing.assert_close(proc["x"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(proc["arr"], np.asarray([[1, 2], [3, 4]], dtype=np.int64))
    assert proc["task"] == ["push", "push"]
    assert proc["meta"] == ["a", "b"]


def test_get_action_probs_for_chunk_batch_matches_loop_recompute() -> None:
    wrapper, policy, _bundle = _batch_wrapper()
    policy.configured_chunk_len = 3
    proc_rows = [
        {"x": torch.zeros(1, 1), "task": ["push"]},
        {"x": torch.ones(1, 1), "task": ["push"]},
    ]
    unsquashed = torch.full((2, 3, 4), 0.25, dtype=torch.float32)

    batched = wrapper.get_action_probs_for_chunk_batch_from_proc_list(proc_rows, unsquashed)

    expected_policy = _DummyBatchChunkPolicy(action_dim=4)
    expected_wrapper = MetaWorldSmolVLAGRPOPolicy(
        _DummyBundle(),
        task="push-v3",
        task_text="push",
        camera_name="corner2",
        flip_corner2=False,
        action_dim=4,
        policy_module=expected_policy,
        action_transform="no_tanh",
    )
    looped = torch.stack(
        [
            expected_wrapper.get_action_probs_for_chunk_from_proc(proc_rows[0], unsquashed[0]),
            expected_wrapper.get_action_probs_for_chunk_from_proc(proc_rows[1], unsquashed[1]),
        ],
        dim=0,
    )

    assert batched.shape == (2, 3)
    torch.testing.assert_close(batched, looped)
