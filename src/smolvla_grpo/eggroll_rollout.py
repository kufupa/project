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
from smolvla_grpo.phase12_vector_eval import select_proc_rows
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


def build_eggroll_reset_seeds(
    *,
    train_seed_base: int,
    iteration: int,
    rollout_seed_offset: int,
    member_ids: list[int],
    seed_mode: str,
) -> list[int]:
    """Build MetaWorld reset seeds for EGGROLL population fitness evaluation."""

    if seed_mode == "member_offset":
        return [
            int(train_seed_base) + int(iteration) * 100003 + int(rollout_seed_offset) * 1009 + int(mid)
            for mid in member_ids
        ]
    if seed_mode == "shared_per_iteration":
        seed = int(train_seed_base) + int(iteration) * 2 + int(rollout_seed_offset)
        return [seed for _mid in member_ids]
    raise ValueError("seed_mode must be 'member_offset' or 'shared_per_iteration'")


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
    action_chunk_size: int = 5,
    train_seed_base: int,
    flow_noise_seed: int,
    rollout_seed_offset: int = 0,
    seed_mode: str = "member_offset",
    noise_manager: EggrollNoiseManager,
    patch_handle: EggrollLinearPatchHandle,
    sigma: float,
    rollout_execution: str = "vector_async",
    async_start_method: str = "forkserver",
    video_member_id: int = 0,
) -> EggrollPopulationRolloutResult:
    if rollout_execution not in ("vector_sync", "vector_async"):
        raise ValueError("rollout_execution must be 'vector_sync' or 'vector_async'")
    if int(population_batch_size) < 1:
        raise ValueError("population_batch_size must be >= 1")
    if int(population_size) < 1:
        raise ValueError("population_size must be >= 1")
    if int(action_chunk_size) < 1:
        raise ValueError("action_chunk_size must be >= 1")

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
            reset_seeds = build_eggroll_reset_seeds(
                train_seed_base=int(train_seed_base),
                iteration=int(iteration),
                rollout_seed_offset=int(rollout_seed_offset),
                member_ids=member_ids_py,
                seed_mode=str(seed_mode),
            )
            reset_t0 = perf_counter()
            obs = env_h.reset_many(reset_seeds)
            timings["reset_seconds"] += perf_counter() - reset_t0
            active = np.ones((wave_size,), dtype=np.bool_)
            wave_rollouts = [
                EggrollMemberRollout(member_id=mid, reset_seed=seed)
                for mid, seed in zip(member_ids_py, reset_seeds, strict=True)
            ]
            member_ids = torch.as_tensor(member_ids_py, device=device, dtype=torch.long)

            reset_frame_idx = member_ids_py.index(int(video_member_id)) if int(video_member_id) in member_ids_py else -1
            if reset_frame_idx >= 0:
                frame = _frame_from_vector_obs(obs, reset_frame_idx)
                if frame is not None:
                    selected_frames.append(frame)

            step_count = 0
            policy_call_idx = 0
            while step_count < int(resolved_max_steps):
                if not bool(np.any(active)):
                    break
                active_rows = [idx for idx in range(wave_size) if bool(active[idx])]
                active_member_ids = [member_ids_py[idx] for idx in active_rows]
                proc_t0 = perf_counter()
                proc = _proc_to_device(
                    select_proc_rows(env_h.build_proc(obs, bundle=bundle), active_rows, batch_size=wave_size),
                    device,
                )
                timings["proc_build_seconds"] += perf_counter() - proc_t0

                noise = _flow_noise(
                    member_ids=active_member_ids,
                    action_chunk=int(getattr(policy.config, "chunk_size", 1)),
                    action_dim=int(policy.config.max_action_dim),
                    seed=int(flow_noise_seed)
                    + int(iteration) * 100003
                    + int(rollout_seed_offset) * 1009
                    + int(policy_call_idx),
                    device=device,
                    dtype=flow_dtype,
                )
                fwd_t0 = perf_counter()
                ctx = patch_handle.context(
                    noise_manager=noise_manager,
                    iteration=int(iteration),
                    sigma=float(sigma),
                    member_ids=member_ids[active_rows],
                )
                with eggroll_linear_context(ctx), torch.inference_mode():
                    chunk = stateless_smolvla_action_chunk(policy, proc, flow_noise=noise)
                timings["forward_seconds"] += perf_counter() - fwd_t0
                if chunk.ndim != 3:
                    raise RuntimeError(f"SmolVLA EGGROLL action chunk must be rank 3, got shape {tuple(chunk.shape)}")
                if int(chunk.shape[0]) != len(active_rows):
                    raise RuntimeError(
                        f"SmolVLA EGGROLL active batch mismatch: got {int(chunk.shape[0])}, expected {len(active_rows)}"
                    )
                effective_chunk = min(
                    int(action_chunk_size),
                    int(chunk.shape[1]),
                    int(resolved_max_steps) - int(step_count),
                )
                if effective_chunk < 1:
                    break

                for chunk_step in range(effective_chunk):
                    if not bool(np.any(active)):
                        break
                    active_before_step = active.copy()
                    action_matrix = np.zeros((wave_size, int(action_dim)), dtype=np.float32)
                    action_by_row: dict[int, np.ndarray] = {}
                    for compact_idx, row_idx in enumerate(active_rows):
                        if not active_before_step[row_idx]:
                            continue
                        post_t0 = perf_counter()
                        _raw, exec_np, _clip_fraction, _clip_any = _postprocess_row(
                            wrapper,
                            chunk[compact_idx, chunk_step, :].detach(),
                        )
                        exec_action = exec_np.reshape(-1).astype(np.float32, copy=False)
                        timings["postprocess_seconds"] += perf_counter() - post_t0
                        action_matrix[row_idx] = exec_action
                        action_by_row[row_idx] = exec_action

                    if not action_by_row:
                        break
                    step_t0 = perf_counter()
                    env_batch = env_h.step_batch(action_matrix)
                    timings["env_step_seconds"] += perf_counter() - step_t0
                    obs = env_batch.observation

                    for row_idx, exec_action in action_by_row.items():
                        rollout = wave_rollouts[row_idx]
                        rollout.actions.append(exec_action.tolist())
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
                    step_count += 1
                policy_call_idx += 1

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
