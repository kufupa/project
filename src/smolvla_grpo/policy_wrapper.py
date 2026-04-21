"""MetaWorld SmolVLA GRPO policy: fork `select_action_distr_params` + Gaussian log-prob."""

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
    policy_tensor: torch.Tensor
    unsquashed: torch.Tensor
    log_prob: torch.Tensor
    action_clip_fraction: float = 0.0
    action_clip_any: bool = False


@dataclass
class SampledBatchStep:
    """Batched env step: actions aligned with vector env rows."""

    exec_action_np: np.ndarray
    policy_tensor: torch.Tensor
    unsquashed: torch.Tensor
    log_prob: torch.Tensor
    action_clip_fraction: np.ndarray
    action_clip_any: np.ndarray


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
        action_transform: str = "no_tanh",
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ) -> None:
        self.bundle = bundle
        self.task_text = task_text
        self.camera_name = camera_name
        self.flip_corner2 = flip_corner2
        self.action_dim = int(action_dim)
        self.eps = float(eps)
        self._policy: Any = policy_module if policy_module is not None else bundle.policy
        self._agent_dim, self._env_dim = _smolvla_state_dims(self._policy)
        if action_transform not in ("no_tanh", "tanh_norm_ablation"):
            raise ValueError("action_transform must be 'no_tanh' or 'tanh_norm_ablation'")
        self.action_transform = action_transform
        self.action_low = (
            np.asarray(action_low, dtype=np.float32).reshape(self.action_dim)
            if action_low is not None
            else np.full((self.action_dim,), -1.0, dtype=np.float32)
        )
        self.action_high = (
            np.asarray(action_high, dtype=np.float32).reshape(self.action_dim)
            if action_high is not None
            else np.full((self.action_dim,), 1.0, dtype=np.float32)
        )

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

    @staticmethod
    def calculate_gaussian_log_prob(
        mean: torch.Tensor,
        log_std: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(log_std)
        var = std * std
        return -0.5 * (((sample - mean) ** 2) / var + 2 * log_std + math.log(2 * math.pi)).sum(dim=-1)

    def _postprocess_and_clip(self, policy_tensor: torch.Tensor) -> tuple[np.ndarray, float, bool]:
        out = self.bundle.postprocessor(policy_tensor)
        if hasattr(out, "detach"):
            raw_np = out.detach().float().cpu().numpy().reshape(-1)
        else:
            raw_np = np.asarray(out, dtype=np.float32).reshape(-1)
        if raw_np.size != self.action_dim:
            raise RuntimeError(
                f"Policy action dim mismatch: expected {self.action_dim}, got {raw_np.size}. "
                "Refusing silent pad/truncate."
            )
        clipped = np.clip(raw_np, self.action_low, self.action_high).astype(np.float32, copy=False)
        changed = np.not_equal(clipped, raw_np)
        return clipped, float(np.mean(changed)), bool(np.any(changed))

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
        if self.action_transform == "tanh_norm_ablation":
            policy_tensor = torch.tanh(unsquashed)
            log_prob = self.calculate_log_prob(mean, log_std, unsquashed, policy_tensor, eps=self.eps)
        else:
            policy_tensor = unsquashed
            log_prob = self.calculate_gaussian_log_prob(mean, log_std, unsquashed)
        exec_np, clip_fraction, clip_any = self._postprocess_and_clip(policy_tensor)
        return SampledStep(
            exec_action_np=exec_np,
            policy_tensor=policy_tensor.detach(),
            unsquashed=unsquashed.detach(),
            log_prob=log_prob.detach(),
            action_clip_fraction=clip_fraction,
            action_clip_any=clip_any,
        )

    def sample_action_batch_from_proc(
        self,
        proc: Any,
        *,
        reset_seed: int,
        rollout_index_offset: int,
        n_envs: int,
    ) -> SampledBatchStep:
        """Sample one action per batch row; RNG matches serial `collect_rollout_group` indexing."""
        proc_d = self._proc_to_device(proc)
        mean, log_std = self._policy.select_action_distr_params(proc_d)
        b = int(mean.shape[0])
        if b != int(n_envs):
            raise ValueError(f"proc batch dim {b} != n_envs {int(n_envs)}")
        exec_rows: list[np.ndarray] = []
        policy_rows: list[torch.Tensor] = []
        unsq_rows: list[torch.Tensor] = []
        lp_rows: list[torch.Tensor] = []
        clip_frac_rows: list[float] = []
        clip_any_rows: list[bool] = []
        base = int(rollout_index_offset)
        for r in range(b):
            gen = torch.Generator(device=mean.device)
            gen.manual_seed(int(reset_seed) * 1000003 + (base + r) * 7919)
            m = mean[r : r + 1]
            ls = log_std[r : r + 1]
            std = torch.exp(ls)
            noise = torch.randn(m.shape, generator=gen, device=m.device, dtype=m.dtype)
            unsquashed = m + std * noise
            if self.action_transform == "tanh_norm_ablation":
                policy_tensor = torch.tanh(unsquashed)
                log_prob = self.calculate_log_prob(m, ls, unsquashed, policy_tensor, eps=self.eps)
            else:
                policy_tensor = unsquashed
                log_prob = self.calculate_gaussian_log_prob(m, ls, unsquashed)
            exec_np, clip_fraction, clip_any = self._postprocess_and_clip(policy_tensor)
            exec_rows.append(np.asarray(exec_np, dtype=np.float32))
            policy_rows.append(policy_tensor.detach())
            unsq_rows.append(unsquashed.detach())
            lp_rows.append(log_prob.detach())
            clip_frac_rows.append(clip_fraction)
            clip_any_rows.append(clip_any)
        return SampledBatchStep(
            exec_action_np=np.stack(exec_rows, axis=0),
            policy_tensor=torch.cat(policy_rows, dim=0),
            unsquashed=torch.cat(unsq_rows, dim=0),
            log_prob=torch.cat(lp_rows, dim=0).reshape(b),
            action_clip_fraction=np.asarray(clip_frac_rows, dtype=np.float64),
            action_clip_any=np.asarray(clip_any_rows, dtype=np.bool_),
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
        if self.action_transform == "tanh_norm_ablation":
            squished = torch.tanh(u)
            return self.calculate_log_prob(mean, log_std, u, squished, eps=self.eps)
        return self.calculate_gaussian_log_prob(mean, log_std, u)

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
