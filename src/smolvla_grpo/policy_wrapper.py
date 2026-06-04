"""MetaWorld SmolVLA GRPO policy: fork `select_action_distr_params` + Gaussian log-prob."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from smolvla_grpo.flow_logprob import sde_step_logprob
from smolvla_pipeline.evaluator import (
    _SmolVLABundle,
    _collect_policy_rgb,
    _flatten_obs_state,
    _maybe_flip_corner2_frame,
    _smolvla_state_dims,
    _vectors_for_smolvla,
)


def concatenate_proc_rows(proc_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate one-row preprocessor snapshots into one batch."""
    rows = list(proc_rows)
    if not rows:
        raise ValueError("proc_rows must be non-empty")
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError("proc_rows must be dictionaries")
    keys = tuple(rows[0].keys())
    for row in rows[1:]:
        if tuple(row.keys()) != keys:
            raise ValueError("proc_rows must have identical keys")

    out: dict[str, Any] = {}
    for key in keys:
        vals = [row[key] for row in rows]
        first = vals[0]
        if torch.is_tensor(first):
            if first.dim() == 0:
                out[key] = torch.stack(vals, dim=0)
            else:
                out[key] = torch.cat(vals, dim=0)
        elif isinstance(first, np.ndarray):
            if first.ndim == 0:
                out[key] = np.stack(vals, axis=0)
            else:
                out[key] = np.concatenate(vals, axis=0)
        elif isinstance(first, (list, tuple)):
            merged: list[Any] = []
            for value in vals:
                merged.extend(list(value))
            out[key] = merged
        else:
            out[key] = vals
    return out


@dataclass
class SampledStep:
    """One env step: exec action for MetaWorld + tensors for GRPO."""

    exec_action_np: np.ndarray
    raw_postprocessed_action_np: np.ndarray
    policy_tensor: torch.Tensor
    unsquashed: torch.Tensor
    logprob_action: torch.Tensor
    log_prob: torch.Tensor
    distr_mean: torch.Tensor
    distr_log_std: torch.Tensor
    action_clip_fraction: float = 0.0
    action_clip_any: bool = False
    postprocessor_oob_mean: float = 0.0
    flow_sde_trace: dict[str, Any] | None = None


@dataclass
class SampledBatchStep:
    """Batched env step: actions aligned with vector env rows."""

    exec_action_np: np.ndarray
    raw_postprocessed_action_np: np.ndarray
    policy_tensor: torch.Tensor
    unsquashed: torch.Tensor
    log_prob: torch.Tensor
    logprob_action: torch.Tensor
    distr_mean: torch.Tensor
    distr_log_std: torch.Tensor
    action_clip_fraction: np.ndarray
    action_clip_any: np.ndarray
    postprocessor_oob_mean: np.ndarray
    flow_sde_traces: list[dict[str, Any] | None] | None = None


@dataclass
class SampledActionChunk:
    exec_action_np: np.ndarray
    raw_postprocessed_action_np: np.ndarray
    policy_tensor: torch.Tensor
    unsquashed_chunk: torch.Tensor
    logprob_action: torch.Tensor
    log_prob_steps: torch.Tensor
    log_prob_sum: torch.Tensor
    distr_mean: torch.Tensor
    distr_log_std: torch.Tensor
    action_clip_fraction: np.ndarray
    action_clip_any: np.ndarray
    postprocessor_oob_mean: np.ndarray
    unique_action_rows: int
    flow_sde_trace: dict[str, Any] | None = None


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


def collate_proc_snapshots(proc_snapshots: Sequence[Any]) -> dict[str, Any]:
    """Merge single-row preprocessor dicts into one batch for SmolVLA forward."""
    if not proc_snapshots:
        return {}
    out: dict[str, Any] = {}
    for key in proc_snapshots[0].keys():
        vals = [proc[key] for proc in proc_snapshots]
        v0 = vals[0]
        if torch.is_tensor(v0):
            out[key] = torch.cat(
                [v if torch.is_tensor(v) else torch.as_tensor(v) for v in vals],
                dim=0,
            )
        elif isinstance(v0, (list, tuple)):
            merged: list[Any] = []
            for v in vals:
                merged.extend(list(v))
            out[key] = merged
        else:
            out[key] = v0
    return out


def collate_flow_sde_traces(traces: Sequence[dict[str, Any] | None]) -> dict[str, Any]:
    """Merge one-row Flow-SDE trace dicts into a batch trace."""
    if not traces or any(t is None for t in traces):
        raise ValueError("flow_sde traces must be present for every proc snapshot")
    first = traces[0]
    assert first is not None
    out: dict[str, Any] = {}
    for key in first.keys():
        vals = [t[key] for t in traces if t is not None]
        v0 = vals[0]
        if torch.is_tensor(v0):
            out[key] = torch.cat([v if torch.is_tensor(v) else torch.as_tensor(v) for v in vals], dim=0)
        else:
            out[key] = vals
    return out


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
        min_log_std: float = -4.0,
        gaussian_logprob_action: str = "executed",
        logprob_mode: str = "gaussian",
        flow_sde_noise_level: float = 0.5,
        flow_sde_trace_step: int = 0,
    ) -> None:
        self.bundle = bundle
        self.min_log_std = float(min_log_std)
        self.task_text = task_text
        self.camera_name = camera_name
        self.flip_corner2 = flip_corner2
        self.action_dim = int(action_dim)
        self.eps = float(eps)
        self._policy: Any = policy_module if policy_module is not None else bundle.policy
        self._agent_dim, self._env_dim = _smolvla_state_dims(self._policy)
        if action_transform not in ("no_tanh", "tanh_norm_ablation"):
            raise ValueError("action_transform must be 'no_tanh' or 'tanh_norm_ablation'")
        if gaussian_logprob_action not in ("executed", "unsquashed"):
            raise ValueError("gaussian_logprob_action must be 'executed' or 'unsquashed'")
        if logprob_mode not in ("gaussian", "flow_sde"):
            raise ValueError("logprob_mode must be 'gaussian' or 'flow_sde'")
        self.action_transform = action_transform
        self.gaussian_logprob_action = gaussian_logprob_action
        self.logprob_mode = logprob_mode
        self.flow_sde_noise_level = float(flow_sde_noise_level)
        self.flow_sde_trace_step = int(flow_sde_trace_step)
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
    def clamp_log_std(log_std: torch.Tensor, min_log_std: float) -> torch.Tensor:
        return torch.clamp(log_std, min=float(min_log_std))

    @staticmethod
    def _executed_tensor_from_np(exec_np: np.ndarray, *, ref: torch.Tensor) -> torch.Tensor:
        t = torch.from_numpy(np.asarray(exec_np, dtype=np.float32)).to(
            device=ref.device, dtype=ref.dtype
        )
        return t.reshape(ref.shape)

    def _gaussian_scored_action(
        self,
        *,
        unsquashed: torch.Tensor,
        executed: torch.Tensor,
    ) -> torch.Tensor:
        if self.gaussian_logprob_action == "unsquashed":
            return unsquashed
        return executed

    @staticmethod
    def calculate_gaussian_log_prob(
        mean: torch.Tensor,
        log_std: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(log_std)
        var = std * std
        return -0.5 * (((sample - mean) ** 2) / var + 2 * log_std + math.log(2 * math.pi)).sum(dim=-1)

    @staticmethod
    def postprocessor_oob_excess(
        raw_np: np.ndarray,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> np.ndarray:
        """Per-dim positive excess beyond env bounds (before clip)."""
        low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        raw = np.asarray(raw_np, dtype=np.float32).reshape(-1)
        below = np.maximum(low - raw, 0.0)
        above = np.maximum(raw - high, 0.0)
        return below + above

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

    def _postprocess_and_clip(
        self, policy_tensor: torch.Tensor
    ) -> tuple[np.ndarray, float, bool, float]:
        raw_np, clipped, clip_fraction, clip_any = self._postprocess_action(policy_tensor)
        oob_mean = float(
            np.mean(self.postprocessor_oob_excess(raw_np, action_low=self.action_low, action_high=self.action_high))
        )
        return clipped, clip_fraction, clip_any, oob_mean

    def _next_flow_sde_noise_seed(self, rng: torch.Generator | None, *, device: torch.device) -> int | None:
        if rng is None:
            return None
        return int(torch.randint(0, 2**31 - 1, (1,), generator=rng, device=device).item())

    def _sample_flow_sde_trace_step(self, rng: torch.Generator | None, *, device: torch.device) -> int:
        num_steps = int(getattr(getattr(self._policy, "config", None), "num_steps", 10))
        if self.flow_sde_trace_step >= 0:
            if self.flow_sde_trace_step >= num_steps:
                raise ValueError(f"flow_sde_trace_step must be < {num_steps}, got {self.flow_sde_trace_step}")
            return int(self.flow_sde_trace_step)
        if rng is None:
            return int(torch.randint(0, num_steps, (1,), device=device).item())
        return int(torch.randint(0, num_steps, (1,), generator=rng, device=device).item())

    def _get_last_flow_sde_trace(self) -> dict[str, Any]:
        model = getattr(self._policy, "model", self._policy)
        trace = getattr(model, "last_flow_sde_trace", None)
        if not isinstance(trace, dict):
            raise RuntimeError("flow_sde logprob requested but policy did not export last_flow_sde_trace")
        return trace

    def _slice_flow_sde_trace(
        self,
        trace: dict[str, Any],
        *,
        row: int | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in trace.items():
            if torch.is_tensor(value):
                t = value
                if row is not None and t.dim() > 0 and int(t.shape[0]) > row:
                    t = t[row : row + 1]
                out[key] = t.detach()
            else:
                out[key] = value
        return out

    @staticmethod
    def _flow_sde_log_prob_from_trace(trace: dict[str, Any], *, action_dim: int) -> torch.Tensor:
        return sde_step_logprob(
            trace["A_next"][:, 0, :action_dim],
            trace["mu_tau"][:, 0, :action_dim],
            trace["sigma_tau"][:, 0, :action_dim],
        ).reshape(-1)

    def _flow_sde_log_prob_steps_from_trace(
        self,
        trace: dict[str, Any],
        *,
        chunk_len: int,
    ) -> torch.Tensor:
        action = trace["A_next"][:, : int(chunk_len), : self.action_dim]
        mu = trace["mu_tau"][:, : int(chunk_len), : self.action_dim]
        sigma = trace["sigma_tau"][:, : int(chunk_len), : self.action_dim]
        return sde_step_logprob(action, mu, sigma).reshape(-1)

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

    def _get_distr_params_chunk_batch(
        self,
        proc: Any,
        *,
        n_envs: int,
        chunk_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        policy_hook = getattr(self._policy, "_get_distr_params_chunk", None)
        if callable(policy_hook):
            try:
                mean, log_std = policy_hook(proc)
            except TypeError:
                mean, log_std = policy_hook(proc, chunk_len=int(chunk_len))
        else:
            model = getattr(self._policy, "model", None)
            model_hook = getattr(model, "_get_distr_params_chunk", None)
            if callable(model_hook):
                try:
                    mean, log_std = model_hook(proc)
                except TypeError:
                    mean, log_std = model_hook(proc, chunk_len=int(chunk_len))
            else:
                mean, log_std = self._policy.select_action_distr_params(proc)
        return self._reshape_chunk_params_batch(
            mean,
            log_std,
            n_envs=int(n_envs),
            chunk_len=int(chunk_len),
        )

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
        if self.logprob_mode == "flow_sde":
            n_action_steps = int(getattr(getattr(self._policy, "config", None), "n_action_steps", 1))
            if n_action_steps != 1:
                raise RuntimeError("flow_sde GRPO currently requires n_action_steps=1")
            noise_seed = self._next_flow_sde_noise_seed(rng, device=self.bundle.device)
            mean, log_std = self._policy.select_action_distr_params(
                proc_d,
                flow_sde_trace=True,
                flow_sde_noise_level=self.flow_sde_noise_level,
                flow_sde_trace_step=self.flow_sde_trace_step,
                flow_sde_noise_seed=noise_seed,
            )
            flow_trace = self._slice_flow_sde_trace(self._get_last_flow_sde_trace())
            policy_tensor = mean
            raw_np, exec_np, clip_fraction, clip_any = self._postprocess_action(policy_tensor)
            oob_mean = float(
                np.mean(self.postprocessor_oob_excess(raw_np, action_low=self.action_low, action_high=self.action_high))
            )
            log_prob = self._flow_sde_log_prob_from_trace(flow_trace, action_dim=self.action_dim)
            trace_action = flow_trace["A_next"][:, 0, : self.action_dim].reshape(mean.shape)
            trace_mu = flow_trace["mu_tau"][:, 0, : self.action_dim].reshape(mean.shape)
            trace_sigma = flow_trace["sigma_tau"][:, 0, : self.action_dim].reshape(mean.shape)
            return SampledStep(
                exec_action_np=exec_np,
                raw_postprocessed_action_np=raw_np,
                policy_tensor=policy_tensor.detach(),
                unsquashed=policy_tensor.detach(),
                logprob_action=trace_action.detach(),
                log_prob=log_prob.detach(),
                distr_mean=trace_mu.detach(),
                distr_log_std=torch.log(trace_sigma.clamp(min=self.eps)).detach(),
                action_clip_fraction=clip_fraction,
                action_clip_any=clip_any,
                postprocessor_oob_mean=oob_mean,
                flow_sde_trace=flow_trace,
            )

        mean, log_std = self._policy.select_action_distr_params(proc_d)
        std = torch.exp(log_std)
        if rng is None:
            noise = torch.randn_like(mean)
        else:
            noise = torch.randn(mean.shape, generator=rng, device=mean.device, dtype=mean.dtype)
        unsquashed = mean + std * noise
        if self.action_transform == "tanh_norm_ablation":
            policy_tensor = torch.tanh(unsquashed)
        else:
            policy_tensor = unsquashed
        raw_np, exec_np, clip_fraction, clip_any = self._postprocess_action(policy_tensor)
        oob_mean = float(
            np.mean(self.postprocessor_oob_excess(raw_np, action_low=self.action_low, action_high=self.action_high))
        )
        exec_t = self._executed_tensor_from_np(exec_np, ref=mean)
        if self.action_transform == "tanh_norm_ablation":
            log_prob = self.calculate_log_prob(mean, log_std, unsquashed, policy_tensor, eps=self.eps)
            logprob_action = unsquashed
        else:
            logprob_action = self._gaussian_scored_action(unsquashed=unsquashed, executed=exec_t)
            log_prob = self.calculate_gaussian_log_prob(mean, log_std, logprob_action)
        return SampledStep(
            exec_action_np=exec_np,
            raw_postprocessed_action_np=raw_np,
            policy_tensor=policy_tensor.detach(),
            unsquashed=unsquashed.detach(),
            logprob_action=logprob_action.detach(),
            log_prob=log_prob.detach(),
            distr_mean=mean.detach(),
            distr_log_std=log_std.detach(),
            action_clip_fraction=clip_fraction,
            action_clip_any=clip_any,
            postprocessor_oob_mean=oob_mean,
            flow_sde_trace=None,
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
        if self.logprob_mode == "flow_sde":
            if rngs is not None and len(rngs) != int(n_envs):
                raise ValueError(f"rngs length {len(rngs)} != n_envs {int(n_envs)}")
            if rngs is None and reset_seed is None:
                raise ValueError("Either rngs or reset_seed must be provided")
            seed_source = rngs[0] if rngs is not None and len(rngs) > 0 else None
            noise_seed = self._next_flow_sde_noise_seed(seed_source, device=self.bundle.device)
            if noise_seed is None and reset_seed is not None:
                noise_seed = int(reset_seed) * 1000003 + int(rollout_index_offset) * 7919
            mean, _log_std = self._policy.select_action_distr_params(
                proc_d,
                flow_sde_trace=True,
                flow_sde_noise_level=self.flow_sde_noise_level,
                flow_sde_trace_step=self.flow_sde_trace_step,
                flow_sde_noise_seed=noise_seed,
            )
            b = int(mean.shape[0])
            if b != int(n_envs):
                raise ValueError(f"proc batch dim {b} != n_envs {int(n_envs)}")
            full_trace = self._get_last_flow_sde_trace()
            exec_rows: list[np.ndarray] = []
            raw_rows: list[np.ndarray] = []
            policy_rows: list[torch.Tensor] = []
            lp_rows: list[torch.Tensor] = []
            scored_rows: list[torch.Tensor] = []
            mean_rows: list[torch.Tensor] = []
            log_std_rows: list[torch.Tensor] = []
            clip_frac_rows: list[float] = []
            clip_any_rows: list[bool] = []
            oob_rows: list[float] = []
            flow_traces: list[dict[str, Any] | None] = []
            for r in range(b):
                row_trace = self._slice_flow_sde_trace(full_trace, row=r)
                policy_tensor = mean[r : r + 1]
                raw_np, exec_np, clip_fraction, clip_any = self._postprocess_action(policy_tensor)
                oob_mean = float(
                    np.mean(self.postprocessor_oob_excess(raw_np, action_low=self.action_low, action_high=self.action_high))
                )
                log_prob = self._flow_sde_log_prob_from_trace(row_trace, action_dim=self.action_dim)
                trace_action = row_trace["A_next"][:, 0, : self.action_dim].reshape(policy_tensor.shape)
                trace_mu = row_trace["mu_tau"][:, 0, : self.action_dim].reshape(policy_tensor.shape)
                trace_sigma = row_trace["sigma_tau"][:, 0, : self.action_dim].reshape(policy_tensor.shape)
                raw_rows.append(np.asarray(raw_np, dtype=np.float32))
                exec_rows.append(np.asarray(exec_np, dtype=np.float32))
                policy_rows.append(policy_tensor.detach())
                lp_rows.append(log_prob.detach())
                scored_rows.append(trace_action.detach())
                mean_rows.append(trace_mu.detach())
                log_std_rows.append(torch.log(trace_sigma.clamp(min=self.eps)).detach())
                clip_frac_rows.append(clip_fraction)
                clip_any_rows.append(clip_any)
                oob_rows.append(oob_mean)
                flow_traces.append(row_trace)
            return SampledBatchStep(
                exec_action_np=np.stack(exec_rows, axis=0),
                raw_postprocessed_action_np=np.stack(raw_rows, axis=0),
                policy_tensor=torch.cat(policy_rows, dim=0),
                unsquashed=torch.cat(policy_rows, dim=0),
                log_prob=torch.cat(lp_rows, dim=0).reshape(b),
                logprob_action=torch.cat(scored_rows, dim=0),
                distr_mean=torch.cat(mean_rows, dim=0),
                distr_log_std=torch.cat(log_std_rows, dim=0),
                action_clip_fraction=np.asarray(clip_frac_rows, dtype=np.float64),
                action_clip_any=np.asarray(clip_any_rows, dtype=np.bool_),
                postprocessor_oob_mean=np.asarray(oob_rows, dtype=np.float64),
                flow_sde_traces=flow_traces,
            )

        mean, log_std = self._policy.select_action_distr_params(proc_d)
        b = int(mean.shape[0])
        if b != int(n_envs):
            raise ValueError(f"proc batch dim {b} != n_envs {int(n_envs)}")
        exec_rows: list[np.ndarray] = []
        raw_rows: list[np.ndarray] = []
        policy_rows: list[torch.Tensor] = []
        unsq_rows: list[torch.Tensor] = []
        lp_rows: list[torch.Tensor] = []
        mean_rows: list[torch.Tensor] = []
        log_std_rows: list[torch.Tensor] = []
        clip_frac_rows: list[float] = []
        clip_any_rows: list[bool] = []
        oob_rows: list[float] = []
        scored_rows: list[torch.Tensor] = []
        if rngs is not None and len(rngs) != b:
            raise ValueError(f"rngs length {len(rngs)} != n_envs {b}")
        if rngs is None and reset_seed is None:
            raise ValueError("Either rngs or reset_seed must be provided")
        base = int(rollout_index_offset)
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
            else:
                policy_tensor = unsquashed
            raw_np, exec_np, clip_fraction, clip_any = self._postprocess_action(policy_tensor)
            oob_mean = float(
                np.mean(self.postprocessor_oob_excess(raw_np, action_low=self.action_low, action_high=self.action_high))
            )
            exec_t = self._executed_tensor_from_np(exec_np, ref=m)
            if self.action_transform == "tanh_norm_ablation":
                log_prob = self.calculate_log_prob(m, ls, unsquashed, policy_tensor, eps=self.eps)
                logprob_action = unsquashed
            else:
                logprob_action = self._gaussian_scored_action(unsquashed=unsquashed, executed=exec_t)
                log_prob = self.calculate_gaussian_log_prob(m, ls, logprob_action)
            raw_rows.append(np.asarray(raw_np, dtype=np.float32))
            exec_rows.append(np.asarray(exec_np, dtype=np.float32))
            policy_rows.append(policy_tensor.detach())
            unsq_rows.append(unsquashed.detach())
            lp_rows.append(log_prob.detach())
            scored_rows.append(logprob_action.detach())
            mean_rows.append(m.detach())
            log_std_rows.append(ls.detach())
            clip_frac_rows.append(clip_fraction)
            clip_any_rows.append(clip_any)
            oob_rows.append(oob_mean)
        return SampledBatchStep(
            exec_action_np=np.stack(exec_rows, axis=0),
            raw_postprocessed_action_np=np.stack(raw_rows, axis=0),
            policy_tensor=torch.cat(policy_rows, dim=0),
            unsquashed=torch.cat(unsq_rows, dim=0),
            log_prob=torch.cat(lp_rows, dim=0).reshape(b),
            logprob_action=torch.cat(scored_rows, dim=0),
            distr_mean=torch.cat(mean_rows, dim=0),
            distr_log_std=torch.cat(log_std_rows, dim=0),
            action_clip_fraction=np.asarray(clip_frac_rows, dtype=np.float64),
            action_clip_any=np.asarray(clip_any_rows, dtype=np.bool_),
            postprocessor_oob_mean=np.asarray(oob_rows, dtype=np.float64),
        )

    def sample_action_chunk_from_proc(
        self,
        proc: Any,
        *,
        chunk_len: int,
        rng: torch.Generator | None = None,
    ) -> SampledActionChunk:
        proc_d = self._proc_to_device(proc)
        if self.logprob_mode == "flow_sde":
            if self.action_transform != "no_tanh":
                raise RuntimeError("flow_sde chunk GRPO requires action_transform='no_tanh'")
            policy_hook = getattr(self._policy, "_get_distr_params_chunk", None)
            model_hook = getattr(getattr(self._policy, "model", None), "_get_distr_params_chunk", None)
            hook = policy_hook if callable(policy_hook) else model_hook
            if not callable(hook):
                raise RuntimeError("flow_sde chunk GRPO requires _get_distr_params_chunk hook")
            tau_idx = self._sample_flow_sde_trace_step(rng, device=self.bundle.device)
            noise_seed = self._next_flow_sde_noise_seed(rng, device=self.bundle.device)
            mean_full, log_std_full = hook(
                proc_d,
                flow_sde_trace=True,
                flow_sde_noise_level=self.flow_sde_noise_level,
                flow_sde_trace_step=tau_idx,
                flow_sde_noise_seed=noise_seed,
            )
            mean, _log_std = self._reshape_chunk_params(
                mean_full,
                torch.zeros_like(mean_full),
                chunk_len=int(chunk_len),
            )
            full_trace = self._slice_flow_sde_trace(self._get_last_flow_sde_trace())
            full_trace["flow_sde_noise_level"] = self.flow_sde_noise_level
            log_prob_steps_batched, trace_mu_batched, trace_log_std_batched = (
                self.get_flow_sde_log_probs_for_chunk_from_proc_list(
                    [proc_d],
                    [full_trace],
                    chunk_len=int(chunk_len),
                )
            )
            log_prob_steps = log_prob_steps_batched.reshape(-1)
            policy_tensor = mean

            exec_rows: list[np.ndarray] = []
            raw_rows: list[np.ndarray] = []
            clip_frac_rows: list[float] = []
            clip_any_rows: list[bool] = []
            oob_rows: list[float] = []
            for row in policy_tensor:
                raw_np, exec_np, clip_fraction, clip_any = self._postprocess_action(row.reshape(1, -1))
                oob_mean = float(
                    np.mean(self.postprocessor_oob_excess(raw_np, action_low=self.action_low, action_high=self.action_high))
                )
                raw_rows.append(np.asarray(raw_np, dtype=np.float32))
                exec_rows.append(np.asarray(exec_np, dtype=np.float32))
                clip_frac_rows.append(clip_fraction)
                clip_any_rows.append(clip_any)
                oob_rows.append(oob_mean)

            exec_action_np = np.stack(exec_rows, axis=0)
            raw_postprocessed_action_np = np.stack(raw_rows, axis=0)
            trace_action = full_trace["A_next"][:, : int(chunk_len), : self.action_dim].reshape(
                int(chunk_len), self.action_dim
            )
            trace_mu = trace_mu_batched.reshape(int(chunk_len), self.action_dim)
            trace_log_std = trace_log_std_batched.reshape(int(chunk_len), self.action_dim)
            return SampledActionChunk(
                exec_action_np=exec_action_np,
                raw_postprocessed_action_np=raw_postprocessed_action_np,
                policy_tensor=policy_tensor.detach(),
                unsquashed_chunk=policy_tensor.detach(),
                logprob_action=trace_action.detach(),
                log_prob_steps=log_prob_steps.detach(),
                log_prob_sum=log_prob_steps.detach().sum(),
                distr_mean=trace_mu.detach(),
                distr_log_std=trace_log_std.detach(),
                action_clip_fraction=np.asarray(clip_frac_rows, dtype=np.float64),
                action_clip_any=np.asarray(clip_any_rows, dtype=np.bool_),
                postprocessor_oob_mean=np.asarray(oob_rows, dtype=np.float64),
                unique_action_rows=int(np.unique(exec_action_np, axis=0).shape[0]),
                flow_sde_trace=full_trace,
            )

        mean, log_std = self._get_distr_params_chunk(proc_d, chunk_len=int(chunk_len))
        std = torch.exp(log_std)
        if rng is None:
            noise = torch.randn_like(mean)
        else:
            noise = torch.randn(mean.shape, generator=rng, device=mean.device, dtype=mean.dtype)
        unsquashed = mean + std * noise
        if self.action_transform == "tanh_norm_ablation":
            policy_tensor = torch.tanh(unsquashed)
        else:
            policy_tensor = unsquashed

        exec_rows: list[np.ndarray] = []
        raw_rows: list[np.ndarray] = []
        clip_frac_rows: list[float] = []
        clip_any_rows: list[bool] = []
        oob_rows: list[float] = []
        for row in policy_tensor:
            raw_np, exec_np, clip_fraction, clip_any = self._postprocess_action(row.reshape(1, -1))
            oob_mean = float(
                np.mean(self.postprocessor_oob_excess(raw_np, action_low=self.action_low, action_high=self.action_high))
            )
            raw_rows.append(np.asarray(raw_np, dtype=np.float32))
            exec_rows.append(np.asarray(exec_np, dtype=np.float32))
            clip_frac_rows.append(clip_fraction)
            clip_any_rows.append(clip_any)
            oob_rows.append(oob_mean)

        exec_action_np = np.stack(exec_rows, axis=0)
        raw_postprocessed_action_np = np.stack(raw_rows, axis=0)
        if self.action_transform == "tanh_norm_ablation":
            logprob_action = unsquashed
            log_prob_steps = self.calculate_log_prob(mean, log_std, unsquashed, policy_tensor, eps=self.eps)
        else:
            exec_t = self._executed_tensor_from_np(exec_action_np, ref=mean)
            logprob_action = self._gaussian_scored_action(unsquashed=unsquashed, executed=exec_t)
            log_prob_steps = self.calculate_gaussian_log_prob(mean, log_std, logprob_action)
        return SampledActionChunk(
            exec_action_np=exec_action_np,
            raw_postprocessed_action_np=raw_postprocessed_action_np,
            policy_tensor=policy_tensor.detach(),
            unsquashed_chunk=unsquashed.detach(),
            logprob_action=logprob_action.detach(),
            log_prob_steps=log_prob_steps.detach().reshape(int(chunk_len)),
            log_prob_sum=log_prob_steps.detach().sum(),
            distr_mean=mean.detach(),
            distr_log_std=log_std.detach(),
            action_clip_fraction=np.asarray(clip_frac_rows, dtype=np.float64),
            action_clip_any=np.asarray(clip_any_rows, dtype=np.bool_),
            postprocessor_oob_mean=np.asarray(oob_rows, dtype=np.float64),
            unique_action_rows=int(np.unique(exec_action_np, axis=0).shape[0]),
            flow_sde_trace=None,
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
        if self.logprob_mode == "flow_sde":
            raise RuntimeError("sample_action_chunk_batch_from_proc does not support logprob_mode='flow_sde'")
        proc_d = self._proc_to_device(proc)
        b = int(n_envs)
        if rngs is not None and len(rngs) != b:
            raise ValueError(f"rngs length {len(rngs)} != n_envs {b}")
        if rngs is None and reset_seed is None:
            raise ValueError("Either rngs or reset_seed must be provided")

        mean, log_std = self._get_distr_params_chunk_batch(
            proc_d,
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

    def _reset_policy_forward_state(self) -> None:
        policy_reset = getattr(self._policy, "reset", None)
        if callable(policy_reset):
            policy_reset()

    def get_action_probs_from_proc_list(
        self,
        proc_snapshots: Sequence[Any],
        scored_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute log p(a|s) for the configured Gaussian-scored action."""
        log_probs, _mean, _log_std = self.get_action_log_probs_and_params_from_proc_list(
            proc_snapshots,
            scored_actions,
        )
        return log_probs

    def get_action_log_probs_and_params_from_proc_list(
        self,
        proc_snapshots: Sequence[Any],
        scored_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute log p(a|s) and return live distribution params for diagnostics."""
        if len(proc_snapshots) != int(scored_actions.shape[0]):
            raise ValueError("proc_snapshots length must match scored_actions batch dim")
        self._reset_policy_forward_state()
        batched = self._proc_to_device(collate_proc_snapshots(proc_snapshots))
        mean, log_std = self._policy.select_action_distr_params(batched)
        log_std = self.clamp_log_std(log_std, self.min_log_std)
        mean = mean.reshape(len(proc_snapshots), -1)
        log_std = log_std.reshape(len(proc_snapshots), -1)
        u = scored_actions.to(mean.device)
        if u.shape != mean.shape:
            u = u.reshape(mean.shape)
        if self.action_transform == "tanh_norm_ablation":
            squished = torch.tanh(u)
            log_probs = self.calculate_log_prob(mean, log_std, u, squished, eps=self.eps)
        else:
            log_probs = self.calculate_gaussian_log_prob(mean, log_std, u)
        return log_probs, mean, log_std

    def get_action_probs_step_batch_from_proc_list(
        self,
        proc_snapshots: Sequence[Any],
        unsquashed_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Batched single-step log p(a|s) recompute from stored proc rows."""
        b = len(proc_snapshots)
        if b != int(unsquashed_actions.reshape(len(proc_snapshots), -1).shape[0]):
            raise ValueError("proc_snapshots length must match unsquashed_actions batch dim")
        proc = self._proc_to_device(concatenate_proc_rows(proc_snapshots))
        mean, log_std = self._policy.select_action_distr_params(proc)
        log_std = self.clamp_log_std(log_std, self.min_log_std)
        mean = mean.reshape(b, -1)
        log_std = log_std.reshape(b, -1)
        u = unsquashed_actions.to(mean.device).reshape(mean.shape)
        if self.action_transform == "tanh_norm_ablation":
            squished = torch.tanh(u)
            return self.calculate_log_prob(mean, log_std, u, squished, eps=self.eps)
        return self.calculate_gaussian_log_prob(mean, log_std, u)

    def get_flow_sde_log_probs_from_proc_list(
        self,
        proc_snapshots: Sequence[Any],
        flow_sde_traces: Sequence[dict[str, Any] | None],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute Flow-SDE transition log p(A_next|A_tau,s) from stored traces."""
        if len(proc_snapshots) != len(flow_sde_traces):
            raise ValueError("proc_snapshots length must match flow_sde_traces length")
        self._reset_policy_forward_state()
        batched = self._proc_to_device(collate_proc_snapshots(proc_snapshots))
        trace = self._proc_to_device(collate_flow_sde_traces(flow_sde_traces))
        hook = getattr(self._policy, "flow_sde_logprob_from_trace", None)
        if not callable(hook):
            raise RuntimeError("flow_sde recompute requires policy.flow_sde_logprob_from_trace")
        _log_probs_full, mu, sigma = hook(
            batched,
            trace,
            flow_sde_noise_level=self.flow_sde_noise_level,
        )
        mu = mu.reshape(len(proc_snapshots), -1)[:, : self.action_dim]
        sigma = sigma.reshape(len(proc_snapshots), -1)[:, : self.action_dim]
        log_std = torch.log(sigma.clamp(min=self.eps))
        action = trace["A_next"][:, 0, : self.action_dim]
        log_probs = sde_step_logprob(action, mu, sigma)
        return log_probs.reshape(-1), mu, log_std

    def get_flow_sde_log_probs_for_chunk_from_proc_list(
        self,
        proc_snapshots: Sequence[Any],
        flow_sde_traces: Sequence[dict[str, Any] | None],
        *,
        chunk_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute chunk Flow-SDE log p(A_next|A_tau,s) from stored traces."""
        if len(proc_snapshots) != len(flow_sde_traces):
            raise ValueError("proc_snapshots length must match flow_sde_traces length")
        self._reset_policy_forward_state()
        batched = self._proc_to_device(collate_proc_snapshots(proc_snapshots))
        trace = self._proc_to_device(collate_flow_sde_traces(flow_sde_traces))
        hook = getattr(self._policy, "flow_sde_logprob_from_trace", None)
        if not callable(hook):
            raise RuntimeError("flow_sde recompute requires policy.flow_sde_logprob_from_trace")
        _log_probs_full, mu, sigma = hook(
            batched,
            trace,
            flow_sde_noise_level=self.flow_sde_noise_level,
        )
        mu_env = mu[:, : int(chunk_len), : self.action_dim]
        sigma_env = sigma[:, : int(chunk_len), : self.action_dim]
        action = trace["A_next"][:, : int(chunk_len), : self.action_dim]
        log_probs = sde_step_logprob(action, mu_env, sigma_env)
        log_std = torch.log(sigma_env.clamp(min=self.eps))
        return log_probs, mu_env, log_std

    def get_action_probs_for_chunk_batch_from_proc_list(
        self,
        proc_snapshots: Sequence[Any],
        unsquashed_chunks: torch.Tensor,
    ) -> torch.Tensor:
        """Batched chunk log p(a|s) recompute from stored one-root proc rows."""
        b = len(proc_snapshots)
        if b < 1:
            raise ValueError("proc_snapshots must be non-empty")
        u = unsquashed_chunks.reshape(b, -1, self.action_dim)
        chunk_len = int(u.shape[1])
        proc = self._proc_to_device(concatenate_proc_rows(proc_snapshots))
        mean, log_std = self._get_distr_params_chunk_batch(
            proc,
            n_envs=b,
            chunk_len=chunk_len,
        )
        log_std = self.clamp_log_std(log_std, self.min_log_std)
        u = u.to(mean.device).reshape(mean.shape)
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
        log_std = self.clamp_log_std(log_std, self.min_log_std)
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
