from __future__ import annotations

import numpy as np
import torch

from smolvla_grpo.policy_wrapper import MetaWorldSmolVLAGRPOPolicy


class _DummyPostprocessor:
    def __init__(self) -> None:
        self.last = None

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        self.last = action.detach().clone()
        return action


class _DummyBundle:
    obs_image_key = "observation.image"
    obs_state_key = "observation.state"
    obs_env_state_key = "observation.environment_state"
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.postprocessor = _DummyPostprocessor()


class _DummyPolicy:
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor) -> None:
        self._mean = mean
        self._log_std = log_std

    def select_action_distr_params(self, proc):
        b = int(proc["x"].shape[0])
        return self._mean.repeat(b, 1), self._log_std.repeat(b, 1)

    def eval(self):
        return None


def _wrapper(*, action_transform: str) -> tuple[MetaWorldSmolVLAGRPOPolicy, _DummyBundle]:
    bundle = _DummyBundle()
    policy = _DummyPolicy(torch.full((1, 4), 2.0), torch.full((1, 4), -20.0))
    wrapper = MetaWorldSmolVLAGRPOPolicy(
        bundle,
        task="push-v3",
        task_text="push",
        camera_name="corner2",
        flip_corner2=False,
        action_dim=4,
        policy_module=policy,
        action_transform=action_transform,
        action_low=np.full((4,), -1.0, dtype=np.float32),
        action_high=np.full((4,), 1.0, dtype=np.float32),
    )
    return wrapper, bundle


def test_no_tanh_passes_normalized_sample_to_postprocessor_and_clips_env_action() -> None:
    wrapper, bundle = _wrapper(action_transform="no_tanh")
    step = wrapper.sample_action_from_proc({"x": torch.zeros(1, 1)})

    np.testing.assert_allclose(bundle.postprocessor.last.numpy(), np.full((1, 4), 2.0), atol=1e-4)
    np.testing.assert_allclose(step.raw_postprocessed_action_np, np.full((4,), 2.0), atol=1e-4)
    np.testing.assert_allclose(step.exec_action_np, np.ones((4,), dtype=np.float32), atol=1e-6)
    assert step.action_clip_fraction == 1.0
    assert step.action_clip_any is True


def test_tanh_norm_ablation_preserves_old_pre_postprocessor_tanh() -> None:
    wrapper, bundle = _wrapper(action_transform="tanh_norm_ablation")
    step = wrapper.sample_action_from_proc({"x": torch.zeros(1, 1)})

    expected = np.tanh(np.full((1, 4), 2.0, dtype=np.float32))
    np.testing.assert_allclose(bundle.postprocessor.last.numpy(), expected, atol=1e-4)
    np.testing.assert_allclose(step.raw_postprocessed_action_np, expected.reshape(4), atol=1e-4)
    np.testing.assert_allclose(step.exec_action_np, expected.reshape(4), atol=1e-4)
    assert step.action_clip_fraction == 0.0
    assert step.action_clip_any is False


def test_batch_sampling_reuses_persistent_generators_across_timesteps() -> None:
    bundle = _DummyBundle()
    policy = _DummyPolicy(torch.zeros(1, 4), torch.zeros(1, 4))
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
    rngs = [torch.Generator(device="cpu").manual_seed(123), torch.Generator(device="cpu").manual_seed(456)]
    proc = {"x": torch.zeros(2, 1)}

    first = wrapper.sample_action_batch_from_proc(proc, rngs=rngs, n_envs=2)
    second = wrapper.sample_action_batch_from_proc(proc, rngs=rngs, n_envs=2)

    assert not torch.allclose(first.unsquashed, second.unsquashed)
