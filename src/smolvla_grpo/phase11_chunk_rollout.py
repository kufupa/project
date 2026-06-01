"""Chunk-level official-LeRobot rollout collection for Phase11 GRPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class ChunkDecision:
    proc_snapshot: Any
    exec_actions: np.ndarray
    rewards: torch.Tensor
    successes: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    valid_action_mask: torch.Tensor
    logprob_actions: torch.Tensor
    log_probs: torch.Tensor
    log_prob_sum: torch.Tensor
    distr_mean: torch.Tensor
    distr_log_std: torch.Tensor
    flow_sde_trace: dict[str, Any] | None
    action_clip_fraction: torch.Tensor
    action_clip_any: torch.Tensor
    postprocessor_oob_mean: torch.Tensor


@dataclass
class ChunkRolloutTrajectory:
    reset_seed: int
    rollout_index: int
    chunks: list[ChunkDecision] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rewards(self) -> list[float]:
        out: list[float] = []
        for chunk in self.chunks:
            valid = chunk.valid_action_mask.bool()
            out.extend(chunk.rewards[valid].float().cpu().tolist())
        return [float(x) for x in out]

    @property
    def successes(self) -> list[bool]:
        out: list[bool] = []
        for chunk in self.chunks:
            valid = chunk.valid_action_mask.bool()
            out.extend(bool(x) for x in chunk.successes[valid].cpu().tolist())
        return out

    def total_return(self) -> float:
        total = 0.0
        for chunk in self.chunks:
            total += float((chunk.rewards.float() * chunk.valid_action_mask.float()).sum().item())
        return total


def build_valid_tail_mask(*, chunk_len: int, executed_count: int) -> torch.Tensor:
    if chunk_len < 1:
        raise ValueError("chunk_len must be >= 1")
    executed = max(0, min(int(executed_count), int(chunk_len)))
    mask = torch.zeros(int(chunk_len), dtype=torch.bool)
    mask[:executed] = True
    return mask


def chunk_success_any(successes: list[bool], valid_action_mask: torch.Tensor) -> bool:
    valid = valid_action_mask.bool().cpu().tolist()
    return any(bool(s) and bool(v) for s, v in zip(successes, valid, strict=False))


def collect_chunk_rollout_group(
    *,
    bundle: Any,
    policy_old: torch.nn.Module,
    task: str,
    task_text: str,
    reset_seed: int,
    episode_index: int,
    max_steps: int,
    group_size: int,
    action_dim: int,
    device: torch.device,
    chunk_len: int,
    action_transform: str = "no_tanh",
    gaussian_logprob_action: str = "executed",
    logprob_mode: str = "flow_sde",
    flow_sde_noise_level: float = 0.5,
    flow_sde_trace_step: int = -1,
) -> list[ChunkRolloutTrajectory]:
    if int(group_size) < 1:
        raise ValueError("group_size must be >= 1")
    if int(chunk_len) < 1:
        raise ValueError("chunk_len must be >= 1")

    from smolvla_grpo.lerobot_metaworld_adapter import (
        OfficialLeRobotMetaWorldGRPORollout,
        resolve_lerobot_horizon,
    )
    from smolvla_grpo.phase11_rollout import _action_bounds, detach_proc_snapshot
    from smolvla_grpo.policy_wrapper import MetaWorldSmolVLAGRPOPolicy
    from smolvla_pipeline.evaluator import _resolve_camera_name, _resolve_flip_corner2

    env_h = OfficialLeRobotMetaWorldGRPORollout(task=task, n_envs=1, use_async_envs=False)
    try:
        resolved_max_steps = resolve_lerobot_horizon(env_h, max_steps)
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
            logprob_mode=logprob_mode,
            flow_sde_noise_level=flow_sde_noise_level,
            flow_sde_trace_step=flow_sde_trace_step,
            action_low=action_low,
            action_high=action_high,
        )
        old_wrapper.eval()
        rollouts: list[ChunkRolloutTrajectory] = []
        for r in range(int(group_size)):
            gen = torch.Generator(device=device)
            gen.manual_seed(int(reset_seed) * 1000003 + r * 7919)
            obs = env_h.reset(int(reset_seed))
            policy_reset = getattr(policy_old, "reset", None)
            if callable(policy_reset):
                policy_reset()
            traj = ChunkRolloutTrajectory(reset_seed=int(reset_seed), rollout_index=r)
            traj.metadata.update(
                {
                    "task": task,
                    "episode_index": int(episode_index),
                    "env_backend": "official_lerobot",
                    "rollout_unit": "chunk",
                    "rollout_execution": "serial",
                    "chunk_len": int(chunk_len),
                    "requested_max_steps": int(max_steps),
                    "resolved_max_steps": int(resolved_max_steps),
                    "logprob_mode": logprob_mode,
                    "flow_sde_trace_step": int(flow_sde_trace_step),
                }
            )
            scalar_steps = 0
            done = False
            while scalar_steps < int(resolved_max_steps) and not done:
                proc = env_h.build_proc(obs, bundle=bundle)
                with torch.no_grad():
                    sampled = old_wrapper.sample_action_chunk_from_proc(proc, chunk_len=int(chunk_len), rng=gen)
                rewards: list[float] = []
                successes: list[bool] = []
                terms: list[bool] = []
                truncs: list[bool] = []
                executed_count = 0
                for i in range(int(chunk_len)):
                    if scalar_steps >= int(resolved_max_steps) or done:
                        break
                    step = env_h.step(sampled.exec_action_np[i : i + 1].astype(np.float32, copy=False))
                    obs = step.observation
                    rewards.append(float(step.reward))
                    successes.append(bool(step.success))
                    terms.append(bool(step.terminated))
                    truncs.append(bool(step.truncated))
                    executed_count += 1
                    scalar_steps += 1
                    if bool(step.success) or bool(step.terminated) or bool(step.truncated):
                        done = True
                        traj.terminated = bool(step.terminated)
                        traj.truncated = bool(step.truncated)
                pad_n = int(chunk_len) - executed_count
                rewards.extend([0.0] * pad_n)
                successes.extend([False] * pad_n)
                terms.extend([False] * pad_n)
                truncs.extend([False] * pad_n)
                valid_mask = build_valid_tail_mask(chunk_len=int(chunk_len), executed_count=executed_count)
                chunk = ChunkDecision(
                    proc_snapshot=detach_proc_snapshot(proc),
                    exec_actions=sampled.exec_action_np.astype(np.float32, copy=False),
                    rewards=torch.tensor(rewards, dtype=torch.float32),
                    successes=torch.tensor(successes, dtype=torch.bool),
                    terminations=torch.tensor(terms, dtype=torch.bool),
                    truncations=torch.tensor(truncs, dtype=torch.bool),
                    valid_action_mask=valid_mask,
                    logprob_actions=sampled.logprob_action.detach().cpu(),
                    log_probs=sampled.log_prob_steps.detach().cpu(),
                    log_prob_sum=(sampled.log_prob_steps.detach().cpu() * valid_mask.float()).sum(),
                    distr_mean=sampled.distr_mean.detach().cpu(),
                    distr_log_std=sampled.distr_log_std.detach().cpu(),
                    flow_sde_trace=sampled.flow_sde_trace,
                    action_clip_fraction=torch.as_tensor(sampled.action_clip_fraction, dtype=torch.float32),
                    action_clip_any=torch.as_tensor(sampled.action_clip_any, dtype=torch.bool),
                    postprocessor_oob_mean=torch.as_tensor(sampled.postprocessor_oob_mean, dtype=torch.float32),
                )
                traj.chunks.append(chunk)
            rollouts.append(traj)
        return rollouts
    finally:
        env_h.close()
