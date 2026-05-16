"""Queue-free EGGROLL population rollouts for SmolVLA on MetaWorld."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import torch

from smolvla_grpo.eggroll_linear import EggrollLinearPatchHandle, eggroll_linear_context
from smolvla_grpo.eggroll_noise import EggrollNoiseManager
from smolvla_grpo.lerobot_metaworld_adapter import (
    OfficialLeRobotMetaWorldGRPORollout,
    resolve_lerobot_horizon,
)
from smolvla_grpo.phase11_rollout import _action_bounds
from smolvla_grpo.policy_wrapper import MetaWorldSmolVLAGRPOPolicy
from smolvla_pipeline.evaluator import (
    _SmolVLABundle,
    _resolve_camera_name,
    _resolve_flip_corner2,
)


@dataclass
class EggrollMemberRollout:
    member_id: int
    reset_seed: int
    rewards: list[float] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)
    actions: list[list[float]] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False

    @property
    def fitness(self) -> float:
        return float(sum(self.rewards))


@dataclass
class EggrollPopulationRolloutResult:
    rollouts: list[EggrollMemberRollout]
    timings: dict[str, float]
    selected_frames: list[np.ndarray]
    selected_rewards: list[float]
    selected_successes: list[bool]

    @property
    def fitness(self) -> list[float]:
        return [item.fitness for item in sorted(self.rollouts, key=lambda r: r.member_id)]


def _proc_to_device(proc: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in proc.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _postprocess_row(wrapper: MetaWorldSmolVLAGRPOPolicy, row: torch.Tensor) -> tuple[np.ndarray, np.ndarray, float, bool]:
    return wrapper._postprocess_action(row.reshape(1, -1))


def _frame_from_vector_obs(obs: dict[str, Any], row: int) -> np.ndarray | None:
    pixels = obs.get("pixels") if isinstance(obs, dict) else None
    if pixels is None:
        pixels = obs.get("observation.image") if isinstance(obs, dict) else None
    if pixels is None:
        return None
    arr = np.asarray(pixels)
    if arr.ndim == 4:
        frame = arr[int(row)]
    elif arr.ndim == 3:
        frame = arr
    else:
        return None
    if frame.shape[0] in (1, 3, 4) and frame.ndim == 3:
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and float(np.max(frame)) <= 1.5:
            frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _flow_noise(
    *,
    member_ids: list[int],
    action_chunk: int,
    action_dim: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for member_id in member_ids:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) * 1000003 + int(member_id) * 7919)
        rows.append(
            torch.randn(
                (int(action_chunk), int(action_dim)),
                generator=gen,
                device=device,
                dtype=dtype,
            )
        )
    return torch.stack(rows, dim=0)


def stateless_smolvla_action_chunk(
    policy: Any,
    proc: dict[str, Any],
    *,
    flow_noise: torch.Tensor | None,
) -> torch.Tensor:
    """Queue-free SmolVLA action chunk. Never calls queued selection APIs."""

    batch = policy._prepare_batch(proc)
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]
    with torch.inference_mode():
        actions = policy.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=flow_noise,
        )
    if not torch.is_tensor(actions):
        raise RuntimeError("SmolVLA sample_actions must return a tensor for EGGROLL")
    action_dim = int(policy.config.action_feature.shape[0])
    return actions[:, :, :action_dim]


def collect_eggroll_population_rollouts(
    *,
    bundle: _SmolVLABundle,
    task: str,
    task_text: str,
    action_dim: int,
    population_size: int,
    population_batch_size: int,
    iteration: int,
    max_steps: int,
    train_seed_base: int,
    flow_noise_seed: int,
    rollout_seed_offset: int = 0,
    noise_manager: EggrollNoiseManager,
    patch_handle: EggrollLinearPatchHandle,
    sigma: float,
    rollout_execution: str = "vector_sync",
    async_start_method: str = "forkserver",
    video_member_id: int = 0,
) -> EggrollPopulationRolloutResult:
    if rollout_execution not in ("vector_sync", "vector_async"):
        raise ValueError("rollout_execution must be 'vector_sync' or 'vector_async'")
    if int(population_batch_size) < 1:
        raise ValueError("population_batch_size must be >= 1")
    if int(population_size) < 1:
        raise ValueError("population_size must be >= 1")

    timings = {
        "env_init_seconds": 0.0,
        "reset_seconds": 0.0,
        "proc_build_seconds": 0.0,
        "forward_seconds": 0.0,
        "postprocess_seconds": 0.0,
        "env_step_seconds": 0.0,
        "rollout_seconds": 0.0,
    }
    rollout_t0 = perf_counter()
    rollouts: list[EggrollMemberRollout] = []
    selected_frames: list[np.ndarray] = []
    selected_rewards: list[float] = []
    selected_successes: list[bool] = []
    device = bundle.device
    policy = bundle.policy
    action_proj = getattr(getattr(policy, "model", None), "action_in_proj", None)
    flow_dtype = getattr(getattr(action_proj, "weight", None), "dtype", torch.float32)
    camera_name = _resolve_camera_name()
    flip_corner2 = _resolve_flip_corner2()

    for start in range(0, int(population_size), int(population_batch_size)):
        member_ids_py = list(range(start, min(start + int(population_batch_size), int(population_size))))
        wave_size = len(member_ids_py)
        env_t0 = perf_counter()
        env_h = OfficialLeRobotMetaWorldGRPORollout(
            task=task,
            n_envs=wave_size,
            use_async_envs=(rollout_execution == "vector_async"),
            async_start_method=async_start_method,
        )
        timings["env_init_seconds"] += perf_counter() - env_t0
        try:
            resolved_max_steps = resolve_lerobot_horizon(env_h, int(max_steps))
            action_low, action_high = _action_bounds(env_h)
            wrapper = MetaWorldSmolVLAGRPOPolicy(
                bundle,
                task=task,
                task_text=task_text,
                camera_name=camera_name,
                flip_corner2=flip_corner2,
                action_dim=action_dim,
                action_low=action_low,
                action_high=action_high,
            )
            reset_seeds = [
                int(train_seed_base) + int(iteration) * 100003 + int(rollout_seed_offset) * 1009 + mid
                for mid in member_ids_py
            ]
            reset_t0 = perf_counter()
            obs = env_h.reset_many(reset_seeds)
            timings["reset_seconds"] += perf_counter() - reset_t0
            active = np.ones((wave_size,), dtype=np.bool_)
            wave_rollouts = [
                EggrollMemberRollout(member_id=mid, reset_seed=seed)
                for mid, seed in zip(member_ids_py, reset_seeds, strict=True)
            ]
            member_ids = torch.as_tensor(member_ids_py, device=device, dtype=torch.long)

            for step_idx in range(int(resolved_max_steps)):
                if not bool(np.any(active)):
                    break
                proc_t0 = perf_counter()
                proc = _proc_to_device(env_h.build_proc(obs, bundle=bundle), device)
                timings["proc_build_seconds"] += perf_counter() - proc_t0

                noise = _flow_noise(
                    member_ids=member_ids_py,
                    action_chunk=int(getattr(policy.config, "chunk_size", 1)),
                    action_dim=int(policy.config.max_action_dim),
                    seed=int(flow_noise_seed)
                    + int(iteration) * 100003
                    + int(rollout_seed_offset) * 1009
                    + step_idx,
                    device=device,
                    dtype=flow_dtype,
                )
                fwd_t0 = perf_counter()
                ctx = patch_handle.context(
                    noise_manager=noise_manager,
                    iteration=int(iteration),
                    sigma=float(sigma),
                    member_ids=member_ids,
                )
                with eggroll_linear_context(ctx), torch.inference_mode():
                    chunk = stateless_smolvla_action_chunk(policy, proc, flow_noise=noise)
                timings["forward_seconds"] += perf_counter() - fwd_t0

                post_t0 = perf_counter()
                first_actions = chunk[:, 0, :].detach()
                exec_rows: list[np.ndarray] = []
                for row_idx in range(wave_size):
                    _raw, exec_np, _clip_fraction, _clip_any = _postprocess_row(wrapper, first_actions[row_idx])
                    exec_rows.append(exec_np)
                action_batch = np.stack(exec_rows, axis=0).astype(np.float32, copy=False)
                timings["postprocess_seconds"] += perf_counter() - post_t0

                step_t0 = perf_counter()
                env_batch = env_h.step_batch(action_batch)
                timings["env_step_seconds"] += perf_counter() - step_t0
                obs = env_batch.observation

                for row_idx, rollout in enumerate(wave_rollouts):
                    if not active[row_idx]:
                        continue
                    rollout.actions.append(action_batch[row_idx].reshape(-1).tolist())
                    rollout.rewards.append(float(env_batch.reward[row_idx]))
                    rollout.successes.append(bool(env_batch.success[row_idx]))
                    if member_ids_py[row_idx] == int(video_member_id):
                        frame = _frame_from_vector_obs(obs, row_idx)
                        if frame is not None:
                            selected_frames.append(frame)
                        selected_rewards.append(float(env_batch.reward[row_idx]))
                        selected_successes.append(bool(env_batch.success[row_idx]))
                    if env_batch.success[row_idx] or env_batch.terminated[row_idx] or env_batch.truncated[row_idx]:
                        active[row_idx] = False
                        rollout.terminated = bool(env_batch.terminated[row_idx])
                        rollout.truncated = bool(env_batch.truncated[row_idx])

            rollouts.extend(wave_rollouts)
        finally:
            env_h.close()

    timings["rollout_seconds"] = perf_counter() - rollout_t0
    return EggrollPopulationRolloutResult(
        rollouts=rollouts,
        timings=timings,
        selected_frames=selected_frames,
        selected_rewards=selected_rewards,
        selected_successes=selected_successes,
    )
