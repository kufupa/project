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
    raw_postprocessed_action_np: np.ndarray
    policy_tensor: torch.Tensor
    unsquashed: torch.Tensor
    log_prob: torch.Tensor
    action_clip_fraction: float = 0.0
    action_clip_any: bool = False


@dataclass
class SampledBatchStep:
    """Batched env step: actions aligned with vector env rows."""

    exec_action_np: np.ndarray
    raw_postprocessed_action_np: np.ndarray
    policy_tensor: torch.Tensor
    unsquashed: torch.Tensor
    log_prob: torch.Tensor
    action_clip_fraction: np.ndarray
    action_clip_any: np.ndarray


@dataclass
class SampledActionChunk:
    exec_action_np: np.ndarray
    raw_postprocessed_action_np: np.ndarray
    policy_tensor: torch.Tensor
    unsquashed_chunk: torch.Tensor
    log_prob_steps: torch.Tensor
    log_prob_sum: torch.Tensor
    action_clip_fraction: np.ndarray
    action_clip_any: np.ndarray
    unique_action_rows: int


@dataclass
class SampledActionChunkBatch:
    exec_action_np: np.ndarray
    raw_postprocessed_action_np: np.ndarray
    policy_tensor: torch.Tensor
    unsquashed_chunk: torch.Tensor
    log_prob_steps: torch.Tensor
    log_prob_sum: torch.Tensor
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

    def _postprocess_action(self, policy_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray, float, bool]:
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
        return raw_np.astype(np.float32, copy=False), clipped, float(np.mean(changed)), bool(np.any(changed))

    def _postprocess_and_clip(self, policy_tensor: torch.Tensor) -> tuple[np.ndarray, float, bool]:
        """Legacy helper: env-executable clipped action plus clip telemetry."""
        _raw_np, clipped, clip_fraction, clip_any = self._postprocess_action(policy_tensor)
        return clipped, clip_fraction, clip_any

    def _get_distr_params_chunk(
        self,
        proc: Any,
        *,
        chunk_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(chunk_len) < 1:
            raise ValueError("chunk_len must be >= 1")
        policy_hook = getattr(self._policy, "_get_distr_params_chunk", None)
        if callable(policy_hook):
            try:
                mean, log_std = policy_hook(proc)
            except TypeError:
                mean, log_std = policy_hook(proc, chunk_len=int(chunk_len))
            return self._reshape_chunk_params(mean, log_std, chunk_len=int(chunk_len))
        model = getattr(self._policy, "model", None)
        model_hook = getattr(model, "_get_distr_params_chunk", None)
        if callable(model_hook):
            try:
                mean, log_std = model_hook(proc)
            except TypeError:
                mean, log_std = model_hook(proc, chunk_len=int(chunk_len))
            return self._reshape_chunk_params(mean, log_std, chunk_len=int(chunk_len))
        mean, log_std = self._policy.select_action_distr_params(proc)
        return self._reshape_chunk_params(mean, log_std, chunk_len=int(chunk_len))

    def _reshape_chunk_params(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        *,
        chunk_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = mean.reshape(-1, self.action_dim)
        log_std = log_std.reshape(-1, self.action_dim)
        if int(mean.shape[0]) == 1 and int(chunk_len) > 1:
            mean = mean.expand(int(chunk_len), self.action_dim)
            log_std = log_std.expand(int(chunk_len), self.action_dim)
        if int(mean.shape[0]) > int(chunk_len):
            mean = mean[: int(chunk_len)]
            log_std = log_std[: int(chunk_len)]
        if int(mean.shape[0]) != int(chunk_len) or int(log_std.shape[0]) != int(chunk_len):
            raise RuntimeError(
                "Chunk distribution params must have chunk_len rows "
                f"({int(chunk_len)}), got mean={tuple(mean.shape)} log_std={tuple(log_std.shape)}."
            )
        return mean, log_std

    def _reshape_chunk_params_batch(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        *,
        n_envs: int,
        chunk_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = int(n_envs)
        t = int(chunk_len)
        a = int(self.action_dim)
        if b < 1:
            raise ValueError("n_envs must be >= 1")
        if t < 1:
            raise ValueError("chunk_len must be >= 1")

        def _reshape(x: torch.Tensor, name: str) -> torch.Tensor:
            x = x.reshape(*tuple(x.shape))
            if x.ndim == 3 and int(x.shape[0]) == b and int(x.shape[2]) == a:
                out = x
            elif x.ndim == 2 and int(x.shape[0]) == b * t and int(x.shape[1]) == a:
                out = x.reshape(b, t, a)
            elif x.ndim == 2 and int(x.shape[0]) == b and int(x.shape[1]) == a:
                out = x.unsqueeze(1).expand(b, t, a)
            elif x.ndim == 2 and int(x.shape[1]) == a and int(x.shape[0]) % b == 0 and int(x.shape[0]) // b >= t:
                total_t = int(x.shape[0]) // b
                out = x.reshape(b, total_t, a)[:, :t, :]
            elif x.ndim == 2 and int(x.shape[0]) >= b * t and int(x.shape[1]) == a:
                out = x[: b * t].reshape(b, t, a)
            else:
                raise RuntimeError(
                    f"Batch chunk {name} params must be (B,T,A), (B*T,A), or (B,A); got {tuple(x.shape)} "
                    f"for B={b} T={t} A={a}."
                )
            if int(out.shape[1]) > t:
                out = out[:, :t, :]
            if tuple(out.shape) != (b, t, a):
                raise RuntimeError(
                    f"Batch chunk {name} params must reshape to {(b, t, a)}, got {tuple(out.shape)}."
                )
            return out

        return _reshape(mean, "mean"), _reshape(log_std, "log_std")

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
        raw_np, exec_np, clip_fraction, clip_any = self._postprocess_action(policy_tensor)
        return SampledStep(
            exec_action_np=exec_np,
            raw_postprocessed_action_np=raw_np,
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
        n_envs: int,
        rngs: Sequence[torch.Generator] | None = None,
        reset_seed: int | None = None,
        rollout_index_offset: int = 0,
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
        if rngs is not None and len(rngs) != b:
            raise ValueError(f"rngs length {len(rngs)} != n_envs {b}")
        if rngs is None and reset_seed is None:
            raise ValueError("Either rngs or reset_seed must be provided")
        base = int(rollout_index_offset)
        raw_rows: list[np.ndarray] = []
        for r in range(b):
            if rngs is None:
                gen = torch.Generator(device=mean.device)
                gen.manual_seed(int(reset_seed) * 1000003 + (base + r) * 7919)
            else:
                gen = rngs[r]
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
            raw_np, exec_np, clip_fraction, clip_any = self._postprocess_action(policy_tensor)
            raw_rows.append(np.asarray(raw_np, dtype=np.float32))
            exec_rows.append(np.asarray(exec_np, dtype=np.float32))
            policy_rows.append(policy_tensor.detach())
            unsq_rows.append(unsquashed.detach())
            lp_rows.append(log_prob.detach())
            clip_frac_rows.append(clip_fraction)
            clip_any_rows.append(clip_any)
        return SampledBatchStep(
            exec_action_np=np.stack(exec_rows, axis=0),
            raw_postprocessed_action_np=np.stack(raw_rows, axis=0),
            policy_tensor=torch.cat(policy_rows, dim=0),
            unsquashed=torch.cat(unsq_rows, dim=0),
            log_prob=torch.cat(lp_rows, dim=0).reshape(b),
            action_clip_fraction=np.asarray(clip_frac_rows, dtype=np.float64),
            action_clip_any=np.asarray(clip_any_rows, dtype=np.bool_),
        )

    def sample_action_chunk_from_proc(
        self,
        proc: Any,
        *,
        chunk_len: int,
        rng: torch.Generator | None = None,
    ) -> SampledActionChunk:
        proc_d = self._proc_to_device(proc)
        mean, log_std = self._get_distr_params_chunk(proc_d, chunk_len=int(chunk_len))
        std = torch.exp(log_std)
        if rng is None:
            noise = torch.randn_like(mean)
        else:
            noise = torch.randn(mean.shape, generator=rng, device=mean.device, dtype=mean.dtype)
        unsquashed = mean + std * noise
        if self.action_transform == "tanh_norm_ablation":
            policy_tensor = torch.tanh(unsquashed)
            log_prob_steps = self.calculate_log_prob(mean, log_std, unsquashed, policy_tensor, eps=self.eps)
        else:
            policy_tensor = unsquashed
            log_prob_steps = self.calculate_gaussian_log_prob(mean, log_std, unsquashed)

        exec_rows: list[np.ndarray] = []
        raw_rows: list[np.ndarray] = []
        clip_frac_rows: list[float] = []
        clip_any_rows: list[bool] = []
        for row in policy_tensor:
            raw_np, exec_np, clip_fraction, clip_any = self._postprocess_action(row.reshape(1, -1))
            raw_rows.append(np.asarray(raw_np, dtype=np.float32))
            exec_rows.append(np.asarray(exec_np, dtype=np.float32))
            clip_frac_rows.append(clip_fraction)
            clip_any_rows.append(clip_any)

        exec_action_np = np.stack(exec_rows, axis=0)
        raw_postprocessed_action_np = np.stack(raw_rows, axis=0)
        return SampledActionChunk(
            exec_action_np=exec_action_np,
            raw_postprocessed_action_np=raw_postprocessed_action_np,
            policy_tensor=policy_tensor.detach(),
            unsquashed_chunk=unsquashed.detach(),
            log_prob_steps=log_prob_steps.detach().reshape(int(chunk_len)),
            log_prob_sum=log_prob_steps.detach().sum(),
            action_clip_fraction=np.asarray(clip_frac_rows, dtype=np.float64),
            action_clip_any=np.asarray(clip_any_rows, dtype=np.bool_),
            unique_action_rows=int(np.unique(exec_action_np, axis=0).shape[0]),
        )

    def sample_action_chunk_batch_from_proc(
        self,
        proc: Any,
        *,
        n_envs: int,
        chunk_len: int,
        rngs: Sequence[torch.Generator] | None = None,
        reset_seed: int | None = None,
        rollout_index_offset: int = 0,
    ) -> SampledActionChunkBatch:
        proc_d = self._proc_to_device(proc)
        b = int(n_envs)
        if rngs is not None and len(rngs) != b:
            raise ValueError(f"rngs length {len(rngs)} != n_envs {b}")
        if rngs is None and reset_seed is None:
            raise ValueError("Either rngs or reset_seed must be provided")

        policy_hook = getattr(self._policy, "_get_distr_params_chunk", None)
        if callable(policy_hook):
            try:
                mean, log_std = policy_hook(proc_d)
            except TypeError:
                mean, log_std = policy_hook(proc_d, chunk_len=int(chunk_len))
        else:
            model = getattr(self._policy, "model", None)
            model_hook = getattr(model, "_get_distr_params_chunk", None)
            if callable(model_hook):
                try:
                    mean, log_std = model_hook(proc_d)
                except TypeError:
                    mean, log_std = model_hook(proc_d, chunk_len=int(chunk_len))
            else:
                mean, log_std = self._policy.select_action_distr_params(proc_d)
        mean, log_std = self._reshape_chunk_params_batch(
            mean,
            log_std,
            n_envs=b,
            chunk_len=int(chunk_len),
        )

        policy_rows: list[torch.Tensor] = []
        unsq_rows: list[torch.Tensor] = []
        lp_rows: list[torch.Tensor] = []
        raw_rows: list[np.ndarray] = []
        exec_rows: list[np.ndarray] = []
        clip_frac_rows: list[np.ndarray] = []
        clip_any_rows: list[np.ndarray] = []
        base = int(rollout_index_offset)
        for row_idx in range(b):
            gen = rngs[row_idx] if rngs is not None else torch.Generator(device=mean.device)
            if rngs is None:
                gen.manual_seed(int(reset_seed) * 1000003 + (base + row_idx) * 7919)
            m = mean[row_idx]
            ls = log_std[row_idx]
            std = torch.exp(ls)
            noise = torch.randn(m.shape, generator=gen, device=m.device, dtype=m.dtype)
            unsquashed = m + std * noise
            if self.action_transform == "tanh_norm_ablation":
                policy_tensor = torch.tanh(unsquashed)
                log_prob_steps = self.calculate_log_prob(m, ls, unsquashed, policy_tensor, eps=self.eps)
            else:
                policy_tensor = unsquashed
                log_prob_steps = self.calculate_gaussian_log_prob(m, ls, unsquashed)

            raw_steps: list[np.ndarray] = []
            exec_steps: list[np.ndarray] = []
            frac_steps: list[float] = []
            any_steps: list[bool] = []
            for step_tensor in policy_tensor:
                raw_np, exec_np, clip_fraction, clip_any = self._postprocess_action(step_tensor.reshape(1, -1))
                raw_steps.append(np.asarray(raw_np, dtype=np.float32))
                exec_steps.append(np.asarray(exec_np, dtype=np.float32))
                frac_steps.append(float(clip_fraction))
                any_steps.append(bool(clip_any))
            policy_rows.append(policy_tensor.detach())
            unsq_rows.append(unsquashed.detach())
            lp_rows.append(log_prob_steps.detach())
            raw_rows.append(np.stack(raw_steps, axis=0))
            exec_rows.append(np.stack(exec_steps, axis=0))
            clip_frac_rows.append(np.asarray(frac_steps, dtype=np.float64))
            clip_any_rows.append(np.asarray(any_steps, dtype=np.bool_))

        return SampledActionChunkBatch(
            exec_action_np=np.stack(exec_rows, axis=0),
            raw_postprocessed_action_np=np.stack(raw_rows, axis=0),
            policy_tensor=torch.stack(policy_rows, dim=0),
            unsquashed_chunk=torch.stack(unsq_rows, dim=0),
            log_prob_steps=torch.stack(lp_rows, dim=0),
            log_prob_sum=torch.stack(lp_rows, dim=0).sum(dim=1),
            action_clip_fraction=np.stack(clip_frac_rows, axis=0),
            action_clip_any=np.stack(clip_any_rows, axis=0),
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

    def get_action_probs_for_chunk_from_proc(
        self,
        proc: Any,
        unsquashed_chunk: torch.Tensor,
    ) -> torch.Tensor:
        proc_d = self._proc_to_device(proc)
        chunk_len = int(unsquashed_chunk.reshape(-1, self.action_dim).shape[0])
        mean, log_std = self._get_distr_params_chunk(proc_d, chunk_len=chunk_len)
        u = unsquashed_chunk.to(mean.device).reshape(mean.shape)
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
