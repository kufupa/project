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
    log_probs: list[torch.Tensor] = field(default_factory=list)
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
) -> list[RolloutTrajectory]:
    """Collect `group_size` trajectories from same seed/task (GRPO group)."""
    env_h = PushV3GRPOEnv(task=task)
    camera_name = _resolve_camera_name()
    flip_corner2 = _resolve_flip_corner2()
    old_wrapper = MetaWorldSmolVLAGRPOPolicy(
        bundle,
        task=task,
        task_text=task_text,
        camera_name=camera_name,
        flip_corner2=flip_corner2,
        action_dim=action_dim,
        policy_module=policy_old,
    )
    old_wrapper.eval()

    rollouts: list[RolloutTrajectory] = []
    gen = torch.Generator(device=device)
    for r in range(group_size):
        gen.manual_seed(int(reset_seed) * 1000003 + r * 7919)
        traj = RolloutTrajectory(reset_seed=reset_seed, rollout_index=r)
        seed_metaworld_process(int(reset_seed))
        env_h.set_task_for_episode(episode_index)
        obs = env_h.reset(reset_seed)
        terminated = False
        truncated = False
        for _step in range(max_steps):
            proc = old_wrapper.build_proc_batch(obs, env_h.inner)
            traj.proc_snapshots.append(detach_proc_snapshot(proc))
            with torch.no_grad():
                step = old_wrapper.sample_action_from_proc(proc, rng=gen)
            traj.exec_actions.append(step.exec_action_np.reshape(-1).tolist())
            traj.unsquashed_actions.append(step.unsquashed.cpu())
            traj.log_probs.append(step.log_prob.cpu())
            obs, reward, terminated, truncated, info = env_h.step(step.exec_action_np)
            traj.rewards.append(float(reward))
            traj.successes.append(_safe_success(info))
            if terminated or truncated:
                traj.terminated = bool(terminated)
                traj.truncated = bool(truncated)
                break
        rollouts.append(traj)
    env_h.close()
    return rollouts


def load_bundle_for_grpo(checkpoint: str, *, task: str = "push-v3") -> tuple[_SmolVLABundle, int]:
    bundle = _load_smolvla_bundle(checkpoint)
    env_probe = PushV3GRPOEnv(task=task)
    try:
        adim = int(np.prod(env_probe.inner.action_space.shape))
    finally:
        env_probe.close()
    return bundle, adim
