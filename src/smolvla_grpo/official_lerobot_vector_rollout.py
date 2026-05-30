"""Batched official LeRobot MetaWorld rollout collection for GRPO (vector_sync / vector_async)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from smolvla_grpo.lerobot_metaworld_adapter import (
    OfficialLeRobotMetaWorldGRPORollout,
    resolve_lerobot_horizon,
)
from smolvla_grpo.phase11_rollout import RolloutTrajectory, _action_bounds, detach_proc_snapshot
from smolvla_grpo.policy_wrapper import MetaWorldSmolVLAGRPOPolicy
from smolvla_pipeline.evaluator import (
    _SmolVLABundle,
    _resolve_camera_name,
    _resolve_flip_corner2,
)


def slice_proc_row(proc: dict[str, Any], row: int) -> dict[str, Any]:
    """Take batch dim `row` from preprocessor output (tensors / ndarrays / task lists)."""
    out: dict[str, Any] = {}
    for k, v in proc.items():
        if torch.is_tensor(v):
            if v.dim() > 0 and int(v.shape[0]) > row:
                out[k] = v[row : row + 1]
            else:
                out[k] = v
        elif isinstance(v, np.ndarray) and v.ndim > 0 and int(v.shape[0]) > row:
            out[k] = np.asarray(v[row : row + 1])
        elif isinstance(v, (list, tuple)) and len(v) > row:
            out[k] = [v[row]]
        else:
            out[k] = copy.deepcopy(v)
    return out


def collect_official_lerobot_vector_rollout_group(
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
    rollout_execution: str,
    async_start_method: str = "forkserver",
    action_transform: str = "no_tanh",
) -> list[RolloutTrajectory]:
    """Collect `group_size` one-episode trajectories with one vector env step per timestep."""
    if rollout_execution not in ("vector_sync", "vector_async"):
        raise ValueError("rollout_execution must be 'vector_sync' or 'vector_async'")
    if group_size < 1:
        raise ValueError("group_size must be >= 1")

    requested_max_steps = int(max_steps)
    use_async = rollout_execution == "vector_async"
    env_h = OfficialLeRobotMetaWorldGRPORollout(
        task=task,
        n_envs=int(group_size),
        use_async_envs=use_async,
        async_start_method=str(async_start_method),
    )
    try:
        resolved = resolve_lerobot_horizon(env_h, max_steps)
        max_steps_i = int(resolved)

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
            action_low=action_low,
            action_high=action_high,
        )
        old_wrapper.eval()

        rollouts = [
            RolloutTrajectory(reset_seed=int(reset_seed), rollout_index=r) for r in range(group_size)
        ]
        for tr in rollouts:
            tr.metadata["task"] = task
            tr.metadata["episode_index"] = int(episode_index)
            tr.metadata["env_backend"] = "official_lerobot"
            tr.metadata["rollout_execution"] = rollout_execution
            tr.metadata["action_transform"] = action_transform
            tr.metadata["async_start_method"] = async_start_method if use_async else "none"
            tr.metadata["requested_max_steps"] = requested_max_steps
            tr.metadata["resolved_max_steps"] = max_steps_i

        obs = env_h.reset(int(reset_seed))
        policy_reset = getattr(policy_old, "reset", None)
        if callable(policy_reset):
            try:
                policy_reset()
            except Exception:
                pass

        active = np.ones((group_size,), dtype=np.bool_)
        rngs = [torch.Generator(device=device) for _ in range(group_size)]
        for r, gen in enumerate(rngs):
            gen.manual_seed(int(reset_seed) * 1000003 + r * 7919)
        for _step in range(max_steps_i):
            if not bool(np.any(active)):
                break
            proc = env_h.build_proc(obs, bundle=bundle)
            with torch.no_grad():
                batch = old_wrapper.sample_action_batch_from_proc(
                    proc,
                    n_envs=int(group_size),
                    rngs=rngs,
                )
            env_batch = env_h.step_batch(batch.exec_action_np.astype(np.float32, copy=False))
            obs = env_batch.observation

            for i in range(group_size):
                if not active[i]:
                    continue
                row_proc = slice_proc_row(proc, i)
                rollouts[i].proc_snapshots.append(detach_proc_snapshot(row_proc))
                rollouts[i].exec_actions.append(batch.exec_action_np[i].reshape(-1).tolist())
                rollouts[i].unsquashed_actions.append(batch.unsquashed[i].detach().cpu())
                rollouts[i].distr_means.append(batch.distr_mean[i].detach().cpu())
                rollouts[i].distr_log_stds.append(batch.distr_log_std[i].detach().cpu())
                rollouts[i].log_probs.append(batch.log_prob[i].detach().cpu())
                rollouts[i].action_clip_fractions.append(float(batch.action_clip_fraction[i]))
                rollouts[i].action_clip_any.append(bool(batch.action_clip_any[i]))
                rollouts[i].rewards.append(float(env_batch.reward[i]))
                rollouts[i].successes.append(bool(env_batch.success[i]))

                if env_batch.success[i] or env_batch.terminated[i] or env_batch.truncated[i]:
                    active[i] = False
                    rollouts[i].terminated = bool(env_batch.terminated[i])
                    rollouts[i].truncated = bool(env_batch.truncated[i])

        return rollouts
    finally:
        env_h.close()
