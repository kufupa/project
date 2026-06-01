"""Deterministic Push-v3 on-policy rollout collection for GRPO."""

from __future__ import annotations

import os

# Slurm GPU nodes: no X11; MuJoCo must use EGL (same as segment_grpo / mt10).
os.environ.setdefault("MUJOCO_GL", "egl")

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from metaworld_determinism import gymnasium_reset_strict, seed_metaworld_process
from smolvla_pipeline.evaluator import (
    _SmolVLABundle,
    _load_smolvla_bundle,
    _resolve_camera_name,
    _resolve_flip_corner2,
    _resolve_task_text,
    _safe_success,
)
from smolvla_grpo.lerobot_metaworld_adapter import (
    OfficialLeRobotMetaWorldGRPORollout,
    resolve_lerobot_horizon,
)
from smolvla_grpo.policy_wrapper import MetaWorldSmolVLAGRPOPolicy


def detach_proc_snapshot(proc: Any) -> Any:
    """Store preprocessor output on CPU for later log-prob recompute."""
    if not isinstance(proc, dict):
        return copy.deepcopy(proc)
    out: dict[str, Any] = {}
    for k, v in proc.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class RolloutTrajectory:
    reset_seed: int
    rollout_index: int
    proc_snapshots: list[Any] = field(default_factory=list)
    exec_actions: list[list[float]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)
    unsquashed_actions: list[torch.Tensor] = field(default_factory=list)
    logprob_actions: list[torch.Tensor] = field(default_factory=list)
    distr_means: list[torch.Tensor] = field(default_factory=list)
    distr_log_stds: list[torch.Tensor] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    flow_sde_traces: list[dict[str, Any] | None] = field(default_factory=list)
    action_clip_fractions: list[float] = field(default_factory=list)
    action_clip_any: list[bool] = field(default_factory=list)
    postprocessor_oob_means: list[float] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rewards_seq(self) -> list[float]:
        return self.rewards

    def total_return(self) -> float:
        return float(sum(self.rewards))


class PushV3GRPOEnv:
    """Minimal MetaWorld MT1 env for GRPO (mirrors evaluator backend setup)."""

    def __init__(self, *, task: str) -> None:
        import metaworld

        self._task = task
        self._camera_name = _resolve_camera_name()
        self._flip_corner2 = _resolve_flip_corner2()
        self._mt1 = metaworld.MT1(task)
        if task not in self._mt1.train_classes:
            available = ", ".join(sorted(self._mt1.train_classes.keys()))
            raise ValueError(f"Task {task!r} not in MT1. Available: {available}")
        env_cls = self._mt1.train_classes[task]
        try:
            self._env = env_cls(render_mode="rgb_array", camera_name=self._camera_name)
        except Exception:
            self._env = env_cls()
        try:
            if hasattr(self._env, "render_mode"):
                self._env.render_mode = "rgb_array"
        except Exception:
            pass
        if self._camera_name == "corner2":
            try:
                self._env.model.cam_pos[2] = [0.75, 0.075, 0.7]
            except Exception:
                pass
        self._tasks = list(getattr(self._mt1, "train_tasks", []) or [])

    @property
    def inner(self) -> Any:
        return self._env

    def set_task_for_episode(self, episode_index: int) -> None:
        if self._tasks:
            self._env.set_task(self._tasks[int(episode_index) % len(self._tasks)])

    def reset(self, reset_seed: int) -> Any:
        out = gymnasium_reset_strict(self._env, int(reset_seed))
        if isinstance(out, tuple) and len(out) >= 1:
            return out[0]
        return out

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        step_out = self._env.step(a)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            return obs, float(reward), bool(terminated), bool(truncated), info if isinstance(info, dict) else {}
        obs, reward, done, info = step_out
        return obs, float(reward), bool(done), False, info if isinstance(info, dict) else {}

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass


def collect_rollout_group(
    *,
    bundle: _SmolVLABundle,
    policy_old: torch.nn.Module,
    task: str,
    task_text: str,
    reset_seed: int,
    episode_index: int,
    max_steps: int,
    group_size: int,
    action_dim: int,
    device: torch.device,
    env_backend: str = "custom",
    rollout_execution: str = "serial",
    async_start_method: str = "forkserver",
    action_transform: str = "no_tanh",
    gaussian_logprob_action: str = "executed",
) -> list[RolloutTrajectory]:
    """Collect `group_size` trajectories from same seed/task (GRPO group)."""
    if rollout_execution not in ("serial", "vector_sync", "vector_async"):
        raise ValueError("rollout_execution must be 'serial', 'vector_sync', or 'vector_async'")
    if rollout_execution != "serial" and env_backend != "official_lerobot":
        raise ValueError("vector rollout modes require env_backend='official_lerobot'")
    if env_backend == "official_lerobot" and rollout_execution in ("vector_sync", "vector_async"):
        from smolvla_grpo.official_lerobot_vector_rollout import collect_official_lerobot_vector_rollout_group

        return collect_official_lerobot_vector_rollout_group(
            bundle=bundle,
            policy_old=policy_old,
            task=task,
            task_text=task_text,
            reset_seed=reset_seed,
            episode_index=episode_index,
            max_steps=max_steps,
            group_size=group_size,
            action_dim=action_dim,
            device=device,
            rollout_execution=rollout_execution,
            async_start_method=async_start_method,
            action_transform=action_transform,
            gaussian_logprob_action=gaussian_logprob_action,
        )

    requested_max_steps = int(max_steps)
    if env_backend == "official_lerobot":
        env_h = OfficialLeRobotMetaWorldGRPORollout(task=task)
        max_steps = resolve_lerobot_horizon(env_h, max_steps)
    elif env_backend == "custom":
        if requested_max_steps <= 0:
            raise ValueError("custom env_backend requires max_steps >= 1")
        env_h = PushV3GRPOEnv(task=task)
    else:
        raise ValueError("env_backend must be 'custom' or 'official_lerobot'")
    camera_name = _resolve_camera_name()
    flip_corner2 = _resolve_flip_corner2()
    action_low, action_high = _action_bounds(env_h)
    old_wrapper = MetaWorldSmolVLAGRPOPolicy(
        bundle,
        task=task,
        task_text=task_text,
        camera_name=camera_name,
        flip_corner2=flip_corner2,
        action_dim=action_dim,
        policy_module=policy_old,
        action_transform=action_transform,
        gaussian_logprob_action=gaussian_logprob_action,
        action_low=action_low,
        action_high=action_high,
    )
    old_wrapper.eval()

    rollouts: list[RolloutTrajectory] = []
    gen = torch.Generator(device=device)
    for r in range(group_size):
        gen.manual_seed(int(reset_seed) * 1000003 + r * 7919)
        traj = RolloutTrajectory(reset_seed=reset_seed, rollout_index=r)
        traj.metadata["task"] = task
        traj.metadata["env_backend"] = env_backend
        traj.metadata["rollout_execution"] = rollout_execution
        traj.metadata["action_transform"] = action_transform
        traj.metadata["gaussian_logprob_action"] = gaussian_logprob_action
        traj.metadata["async_start_method"] = (
            str(async_start_method) if rollout_execution == "vector_async" else "none"
        )
        traj.metadata["requested_max_steps"] = requested_max_steps
        traj.metadata["resolved_max_steps"] = int(max_steps)
        if env_backend == "official_lerobot":
            obs = env_h.reset(reset_seed)
        else:
            seed_metaworld_process(int(reset_seed))
            env_h.set_task_for_episode(episode_index)
            obs = env_h.reset(reset_seed)
        policy_reset = getattr(policy_old, "reset", None)
        if callable(policy_reset):
            try:
                policy_reset()
            except Exception:
                pass
        terminated = False
        truncated = False
        for _step in range(max_steps):
            if env_backend == "official_lerobot":
                proc = env_h.build_proc(obs, bundle=bundle)
            else:
                proc = old_wrapper.build_proc_batch(obs, env_h.inner)
            traj.proc_snapshots.append(detach_proc_snapshot(proc))
            with torch.no_grad():
                step = old_wrapper.sample_action_from_proc(proc, rng=gen)
            traj.exec_actions.append(step.exec_action_np.reshape(-1).tolist())
            traj.unsquashed_actions.append(step.unsquashed.cpu())
            traj.logprob_actions.append(step.logprob_action.cpu())
            traj.distr_means.append(step.distr_mean.cpu())
            traj.distr_log_stds.append(step.distr_log_std.cpu())
            traj.log_probs.append(step.log_prob.cpu())
            traj.flow_sde_traces.append(step.flow_sde_trace)
            traj.action_clip_fractions.append(float(step.action_clip_fraction))
            traj.action_clip_any.append(bool(step.action_clip_any))
            traj.postprocessor_oob_means.append(float(step.postprocessor_oob_mean))
            if env_backend == "official_lerobot":
                action_batch = step.exec_action_np.reshape(1, -1).astype(np.float32)
                env_step = env_h.step(action_batch)
                obs = env_step.observation
                reward = env_step.reward
                terminated = env_step.terminated
                truncated = env_step.truncated
                success = env_step.success
            else:
                obs, reward, terminated, truncated, info = env_h.step(step.exec_action_np)
                success = _safe_success(info)
            traj.rewards.append(float(reward))
            traj.successes.append(bool(success))
            if success or terminated or truncated:
                traj.terminated = bool(terminated)
                traj.truncated = bool(truncated)
                break
        rollouts.append(traj)
    env_h.close()
    return rollouts


def _action_bounds(env_h: Any) -> tuple[np.ndarray, np.ndarray]:
    space = None
    inner = getattr(env_h, "inner", env_h)
    if hasattr(inner, "single_action_space"):
        space = inner.single_action_space
    elif hasattr(inner, "action_space"):
        space = inner.action_space
    elif hasattr(env_h, "inner") and hasattr(env_h.inner, "action_space"):
        space = env_h.inner.action_space
    if space is None or not hasattr(space, "low") or not hasattr(space, "high"):
        adim = int(getattr(env_h, "action_dim", 0) or np.prod(env_h.inner.action_space.shape))
        return np.full((adim,), -1.0, dtype=np.float32), np.full((adim,), 1.0, dtype=np.float32)
    low = np.asarray(space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(space.high, dtype=np.float32).reshape(-1)
    adim = int(getattr(env_h, "action_dim", low.size))
    if low.size != adim and low.size % max(adim, 1) == 0:
        low = low.reshape(-1, adim)[0]
        high = high.reshape(-1, adim)[0]
    if low.size != high.size:
        raise ValueError("action_space low/high shape mismatch")
    return low, high


def load_bundle_for_grpo(
    checkpoint: str,
    *,
    task: str = "push-v3",
    env_backend: str = "custom",
    n_action_steps: int = 1,
) -> tuple[_SmolVLABundle, int]:
    bundle = _load_smolvla_bundle(checkpoint, n_action_steps=int(n_action_steps))
    if env_backend == "official_lerobot":
        env_probe = OfficialLeRobotMetaWorldGRPORollout(task=task)
    elif env_backend == "custom":
        env_probe = PushV3GRPOEnv(task=task)
    else:
        raise ValueError("env_backend must be 'custom' or 'official_lerobot'")
    try:
        if env_backend == "official_lerobot":
            adim = int(env_probe.action_dim)
        else:
            adim = int(np.prod(env_probe.inner.action_space.shape))
    finally:
        env_probe.close()
    return bundle, adim
