from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from smolvla_grpo.grpo_math import summarize_logprob_ratio_parity
from smolvla_grpo.flow_logprob import sde_step_logprob
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


class _TracePolicy(_DummyPolicy):
    def __init__(self) -> None:
        super().__init__(torch.zeros(1, 4), torch.zeros(1, 4))
        self.model = self
        self.config = type("Config", (), {"n_action_steps": 1})()
        self.last_flow_sde_trace = None

    def select_action_distr_params(self, proc, **kwargs):
        assert kwargs["flow_sde_trace"] is True
        b = int(proc["x"].shape[0])
        mu = torch.zeros(b, 1, 4)
        sigma = torch.full((b, 1, 4), 0.5)
        a_next = torch.full((b, 1, 4), 0.25)
        self.last_flow_sde_trace = {
            "tau_idx": torch.zeros(b, dtype=torch.long),
            "A_tau": torch.zeros(b, 1, 4),
            "v_tau": torch.ones(b, 1, 4),
            "mu_tau": mu,
            "sigma_tau": sigma,
            "A_next": a_next,
            "noise_seed": kwargs.get("flow_sde_noise_seed"),
        }
        return a_next[:, 0, :], torch.zeros(b, 4)

    def flow_sde_logprob_from_trace(self, proc, flow_sde_trace, *, flow_sde_noise_level: float):
        assert flow_sde_noise_level == 0.5
        return (
            sde_step_logprob(
                flow_sde_trace["A_next"],
                flow_sde_trace["mu_tau"],
                flow_sde_trace["sigma_tau"],
            ),
            flow_sde_trace["mu_tau"],
            flow_sde_trace["sigma_tau"],
        )


def _wrapper(
    *,
    action_transform: str = "no_tanh",
    gaussian_logprob_action: str = "executed",
) -> tuple[MetaWorldSmolVLAGRPOPolicy, _DummyBundle, _DummyPolicy]:
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
        gaussian_logprob_action=gaussian_logprob_action,
        action_low=np.full((4,), -1.0, dtype=np.float32),
        action_high=np.full((4,), 1.0, dtype=np.float32),
    )
    return wrapper, bundle, policy


def test_logprob_ratio_parity_identity() -> None:
    old_lp = torch.tensor([-1.2, -0.8, -2.1], dtype=torch.float32)
    stats = summarize_logprob_ratio_parity(old_lp, old_lp.clone(), tolerance=0.02)
    assert stats.within_tolerance
    assert abs(stats.mean_ratio - 1.0) < 1e-5


def test_logprob_scores_executed_action_when_clipped() -> None:
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
    assert step.action_clip_fraction > 0.0
    assert not torch.allclose(step.unsquashed.flatten()[:2], step.logprob_action.flatten()[:2])
    recomputed = wrapper.get_action_probs_from_proc_list(
        [proc], step.logprob_action.reshape(1, -1)
    )
    assert torch.allclose(step.log_prob.reshape(()), recomputed.reshape(()), atol=1e-5)


def test_logprob_unsquashed_ablation_scores_sampled_action_when_clipped() -> None:
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
        gaussian_logprob_action="unsquashed",
        action_low=np.full((4,), -1.0, dtype=np.float32),
        action_high=np.full((4,), 1.0, dtype=np.float32),
    )
    step = wrapper.sample_action_from_proc({"x": torch.zeros(1, 1)})
    proc = {"x": torch.zeros(1, 1)}
    assert step.action_clip_fraction > 0.0
    assert torch.allclose(step.unsquashed, step.logprob_action)
    exec_t = torch.from_numpy(step.exec_action_np.reshape(step.unsquashed.shape))
    assert not torch.allclose(exec_t, step.logprob_action)
    recomputed = wrapper.get_action_probs_from_proc_list(
        [proc], step.logprob_action.reshape(1, -1)
    )
    assert torch.allclose(step.log_prob.reshape(()), recomputed.reshape(()), atol=1e-5)


def test_get_action_probs_resets_policy_before_batched_forward() -> None:
    wrapper, _bundle, policy = _wrapper()
    procs = [{"x": torch.zeros(1, 1)}, {"x": torch.zeros(1, 1)}]
    unsq = torch.zeros(2, 4)
    wrapper.get_action_probs_from_proc_list(procs, unsq)
    assert policy.reset_calls >= 1


def test_flow_sde_mode_scores_and_exports_trace() -> None:
    bundle = _DummyBundle()
    policy = _TracePolicy()
    wrapper = MetaWorldSmolVLAGRPOPolicy(
        bundle,
        task="push-v3",
        task_text="push",
        camera_name="corner2",
        flip_corner2=False,
        action_dim=4,
        policy_module=policy,
        logprob_mode="flow_sde",
        flow_sde_noise_level=0.5,
        flow_sde_trace_step=0,
        action_low=np.full((4,), -1.0, dtype=np.float32),
        action_high=np.full((4,), 1.0, dtype=np.float32),
    )

    step = wrapper.sample_action_from_proc({"x": torch.zeros(1, 1)}, rng=torch.Generator().manual_seed(123))
    trace = step.flow_sde_trace

    assert trace is not None
    for key in ("tau_idx", "A_tau", "v_tau", "mu_tau", "sigma_tau", "A_next", "noise_seed"):
        assert key in trace
    expected = sde_step_logprob(trace["A_next"], trace["mu_tau"], trace["sigma_tau"]).reshape(())
    assert torch.allclose(step.log_prob.reshape(()), expected, atol=1e-6)


def test_flow_sde_recompute_uses_stored_trace() -> None:
    bundle = _DummyBundle()
    policy = _TracePolicy()
    wrapper = MetaWorldSmolVLAGRPOPolicy(
        bundle,
        task="push-v3",
        task_text="push",
        camera_name="corner2",
        flip_corner2=False,
        action_dim=4,
        policy_module=policy,
        logprob_mode="flow_sde",
        flow_sde_noise_level=0.5,
        action_low=np.full((4,), -1.0, dtype=np.float32),
        action_high=np.full((4,), 1.0, dtype=np.float32),
    )
    step = wrapper.sample_action_from_proc({"x": torch.zeros(1, 1)}, rng=torch.Generator().manual_seed(123))

    recomputed, mu, log_std = wrapper.get_flow_sde_log_probs_from_proc_list(
        [{"x": torch.zeros(1, 1)}],
        [step.flow_sde_trace],
    )

    assert torch.allclose(recomputed.reshape(()), step.log_prob.reshape(()), atol=1e-6)
    assert torch.allclose(mu, step.distr_mean)
    assert torch.allclose(log_std, step.distr_log_std)


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


def test_trainer_live_parity_uses_recompute_path_not_stored_params() -> None:
    text = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "grpo"
        / "train_phase11_env_on_policy_grpo.py"
    ).read_text(encoding="utf-8")
    assert "def compute_live_logprob_parity(" in text
    parity_body = text.split("def compute_live_logprob_parity(", maxsplit=1)[1].split(
        "\ndef main()", maxsplit=1
    )[0]
    assert "get_action_probs_from_proc_list" in parity_body
    assert "calculate_gaussian_log_prob" not in parity_body
    assert "mean_stored" not in parity_body
