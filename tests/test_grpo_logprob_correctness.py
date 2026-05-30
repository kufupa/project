from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from smolvla_grpo.grpo_math import summarize_logprob_ratio_parity
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
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def select_action_distr_params(self, proc):
        b = int(proc["x"].shape[0])
        return self._mean.repeat(b, 1), self._log_std.repeat(b, 1)

    def eval(self):
        return None


def _wrapper(*, action_transform: str = "no_tanh") -> tuple[MetaWorldSmolVLAGRPOPolicy, _DummyBundle, _DummyPolicy]:
    bundle = _DummyBundle()
    policy = _DummyPolicy(torch.zeros(1, 4), torch.full((1, 4), -1.0))
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
    return wrapper, bundle, policy


def test_logprob_ratio_parity_identity() -> None:
    old_lp = torch.tensor([-1.2, -0.8, -2.1], dtype=torch.float32)
    stats = summarize_logprob_ratio_parity(old_lp, old_lp.clone(), tolerance=0.02)
    assert stats.within_tolerance
    assert abs(stats.mean_ratio - 1.0) < 1e-5


def test_clipped_exec_does_not_change_gaussian_logprob_target() -> None:
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
        action_transform="no_tanh",
        action_low=np.full((4,), -1.0, dtype=np.float32),
        action_high=np.full((4,), 1.0, dtype=np.float32),
    )
    step = wrapper.sample_action_from_proc({"x": torch.zeros(1, 1)})
    proc = {"x": torch.zeros(1, 1)}
    recomputed = wrapper.get_action_probs_from_proc_list([proc], step.unsquashed.reshape(1, -1))
    assert step.action_clip_fraction > 0.0
    assert torch.allclose(step.log_prob.reshape(()), recomputed.reshape(()), atol=1e-5)


def test_get_action_probs_resets_policy_per_timestep() -> None:
    wrapper, _bundle, policy = _wrapper()
    procs = [{"x": torch.zeros(1, 1)}, {"x": torch.zeros(1, 1)}]
    unsq = torch.zeros(2, 4)
    wrapper.get_action_probs_from_proc_list(procs, unsq)
    assert policy.reset_calls >= 2


def test_trainer_rejects_nonzero_euler_without_flag() -> None:
    text = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "grpo"
        / "train_phase11_env_on_policy_grpo.py"
    ).read_text(encoding="utf-8")
    assert 'default=0.0' in text
    assert "--allow-euler-noise" in text
    assert "summarize_logprob_ratio_parity" in text
    assert "parity_stats" in text
