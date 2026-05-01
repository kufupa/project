"""MetaWorld SmolVLA GRPO policy: fork `select_action_distr_params` + Gaussian + tanh log-prob (safe-robot style)."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from smolvla_pipeline.evaluator import (
    _SmolVLABundle,
    _coerce_exec_action,
    _collect_policy_rgb,
    _flatten_obs_state,
    _maybe_flip_corner2_frame,
    _smolvla_state_dims,
    _vectors_for_smolvla,
)


@dataclass
class SampledStep:
    """One env step: exec action for MetaWorld + tensors for GRPO."""

    exec_action_np: np.ndarray
    squished_policy_tensor: torch.Tensor
    unsquashed: torch.Tensor
    log_prob: torch.Tensor


class MetaWorldSmolVLAGRPOPolicy:
    """Wraps LeRobot SmolVLAPolicy + preprocessor/postprocessor for Push-v3 rollouts."""

    def __init__(
        self,
        bundle: _SmolVLABundle,
        *,
        task: str,
        task_text: str,
        camera_name: str,
        flip_corner2: bool,
        action_dim: int,
        eps: float = 1e-8,
        policy_module: Any | None = None,
    ) -> None:
        self.bundle = bundle
        self.task_text = task_text
        self.camera_name = camera_name
        self.flip_corner2 = flip_corner2
        self.action_dim = int(action_dim)
        self.eps = float(eps)
        self._policy: Any = policy_module if policy_module is not None else bundle.policy
        self._agent_dim, self._env_dim = _smolvla_state_dims(self._policy)

    def build_proc_batch(self, obs: Any, env: Any) -> Any:
        flat = _flatten_obs_state(obs)
        if flat.size == 0:
            raise RuntimeError("Empty observation vector from environment.")
        agent_vec, env_vec = _vectors_for_smolvla(flat, self._agent_dim, self._env_dim)
        rgb = _collect_policy_rgb(env, obs)
        rgb = _maybe_flip_corner2_frame(
            rgb,
            camera_name=self.camera_name,
            flip_corner2=self.flip_corner2,
        )
        timg = torch.from_numpy(rgb).unsqueeze(0).permute(0, 3, 1, 2).contiguous().float() / 255.0
        timg = timg.to(self.bundle.device)
        st = torch.from_numpy(agent_vec).unsqueeze(0).to(self.bundle.device)
        es = torch.from_numpy(env_vec).unsqueeze(0).to(self.bundle.device)
        raw = {
            self.bundle.obs_image_key: timg,
            self.bundle.obs_state_key: st,
            self.bundle.obs_env_state_key: es,
            "task": self.task_text,
        }
        return self.bundle.preprocessor(raw)

    def _obs_to_proc_batch(self, obs: Any, env: Any) -> Any:
        return self.build_proc_batch(obs, env)

    @staticmethod
    def calculate_log_prob(
        mean: torch.Tensor,
        log_std: torch.Tensor,
        unsquashed: torch.Tensor,
        squished: torch.Tensor,
        *,
        eps: float,
    ) -> torch.Tensor:
        std = torch.exp(log_std)
        var = std * std
        log_prob_u = -0.5 * (
            ((unsquashed - mean) ** 2) / var + 2 * log_std + math.log(2 * math.pi)
        )
        log_prob_u = log_prob_u.sum(dim=-1)
        correction = torch.sum(torch.log(1 - squished.pow(2) + eps), dim=-1)
        return log_prob_u - correction

    def assert_grpo_api(self) -> None:
        """Raise if forked LeRobot GRPO hooks are missing."""
        pol = self._policy
        if not hasattr(pol, "select_action_distr_params"):
            raise RuntimeError(
                "SmolVLAPolicy.select_action_distr_params missing. "
                "Install jsnchon/lerobot@f30fc2a1b904bb2ccd752cfff94f6f4423bd523b (see scripts/grpo/README.md)."
            )
        model = getattr(pol, "model", None)
        if model is None or not hasattr(model, "log_std"):
            raise RuntimeError(
                "SmolVLAPolicy.model.log_std missing. Use forked LeRobot for GRPO."
            )

    def set_log_std(self, value: float) -> None:
        model = self._policy.model
        ls = model.log_std
        new = torch.full(
            ls.shape,
            fill_value=float(value),
            device=ls.device,
            dtype=ls.dtype,
        )
        model.log_std = nn.Parameter(new)

    def set_euler_step_noise_std(self, std: float) -> None:
        if hasattr(self._policy, "euler_step_noise_std"):
            self._policy.euler_step_noise_std = float(std)
        elif hasattr(self._policy, "model") and hasattr(self._policy.model, "euler_step_noise_std"):
            self._policy.model.euler_step_noise_std = float(std)
        else:
            raise AttributeError("SmolVLA policy has no euler_step_noise_std hook")

    def sample_action_from_proc(
        self,
        proc: Any,
        *,
        rng: torch.Generator | None = None,
    ) -> SampledStep:
        proc_d = self._proc_to_device(proc)
        mean, log_std = self._policy.select_action_distr_params(proc_d)
        std = torch.exp(log_std)
        if rng is None:
            noise = torch.randn_like(mean)
        else:
            noise = torch.randn(mean.shape, generator=rng, device=mean.device, dtype=mean.dtype)
        unsquashed = mean + std * noise
        squished = torch.tanh(unsquashed)
        log_prob = self.calculate_log_prob(mean, log_std, unsquashed, squished, eps=self.eps)
        out = self.bundle.postprocessor(squished)
        exec_np = _coerce_exec_action(
            out,
            action_dim=self.action_dim,
            np_module=np,
        )
        return SampledStep(
            exec_action_np=exec_np,
            squished_policy_tensor=squished.detach(),
            unsquashed=unsquashed.detach(),
            log_prob=log_prob.detach(),
        )

    def sample_action(
        self,
        obs: Any,
        env: Any,
        *,
        rng: torch.Generator | None = None,
    ) -> SampledStep:
        proc = self.build_proc_batch(obs, env)
        return self.sample_action_from_proc(proc, rng=rng)

    def _proc_to_device(self, proc: Any) -> Any:
        out: dict[str, Any] = {}
        for k, v in proc.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.bundle.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def get_action_probs_from_proc_list(
        self,
        proc_snapshots: Sequence[Any],
        unsquashed_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute log p(a|s) from stored preprocessor outputs (no env needed)."""
        if len(proc_snapshots) != int(unsquashed_actions.shape[0]):
            raise ValueError("proc_snapshots length must match unsquashed_actions batch dim")
        means: list[torch.Tensor] = []
        log_stds: list[torch.Tensor] = []
        for proc_cpu in proc_snapshots:
            proc = self._proc_to_device(proc_cpu)
            mean, log_std = self._policy.select_action_distr_params(proc)
            means.append(mean.reshape(1, -1))
            log_stds.append(log_std.reshape(1, -1))
        mean = torch.cat(means, dim=0)
        log_std = torch.cat(log_stds, dim=0)
        u = unsquashed_actions.to(mean.device)
        if u.shape != mean.shape:
            u = u.reshape(mean.shape)
        squished = torch.tanh(u)
        return self.calculate_log_prob(mean, log_std, u, squished, eps=self.eps)

    def get_action_probs_chunk(
        self,
        obs_list: Sequence[Any],
        env: Any,
        unsquashed_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute log p(a|s) using live obs+env (debug / legacy)."""
        procs = [self.build_proc_batch(obs, env) for obs in obs_list]
        return self.get_action_probs_from_proc_list(procs, unsquashed_actions)

    def train(self, mode: bool = True) -> None:
        self._policy.train(mode)

    def eval(self) -> None:
        self._policy.eval()


def freeze_all_but_grpo_trainables(policy: Any) -> list[nn.Parameter]:
    """Match safe-robot: lm_expert + log_std trainable."""
    for p in policy.parameters():
        p.requires_grad = False
    trainable: list[nn.Parameter] = []
    model = getattr(policy, "model", None)
    if model is None:
        return trainable
    vlm = getattr(model, "vlm_with_expert", None)
    if vlm is not None:
        expert = getattr(vlm, "lm_expert", None)
        if expert is not None:
            for p in expert.parameters():
                p.requires_grad = True
            trainable.extend(p for p in expert.parameters() if p.requires_grad)
    if hasattr(model, "log_std") and isinstance(model.log_std, nn.Parameter):
        model.log_std.requires_grad = True
        trainable.append(model.log_std)
    return trainable


def clone_policy_state(policy: Any) -> dict[str, torch.Tensor]:
    return {k: v.clone().detach() for k, v in policy.state_dict().items()}


def load_policy_state(policy: Any, state: dict[str, torch.Tensor]) -> None:
    policy.load_state_dict(state)


def deepcopy_policy_for_old(bundle: _SmolVLABundle) -> Any:
    """CPU deepcopy of policy weights for reference (then .to(device))."""
    p = copy.deepcopy(bundle.policy)
    p.eval()
    return p.to(bundle.device)
