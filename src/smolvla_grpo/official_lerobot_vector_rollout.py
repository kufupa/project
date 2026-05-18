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
from smolvla_grpo.phase11_rollout import (
    RolloutTrajectory,
    _action_bounds,
    _record_executed_chunk,
    _validate_action_chunk_size,
    detach_proc_snapshot,
)
from smolvla_grpo.policy_wrapper import (
    MetaWorldSmolVLAGRPOPolicy,
    SampledActionChunkBatch,
    SampledBatchStep,
)
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


def iter_compact_row_slices(n_active: int, policy_batch_size: int) -> list[tuple[int, int]]:
    if n_active <= 0:
        return []
    bs = max(1, min(int(policy_batch_size), int(n_active)))
    return [(start, min(start + bs, n_active)) for start in range(0, n_active, bs)]


def concat_sampled_action_batches(parts: list[SampledBatchStep]) -> SampledBatchStep:
    if not parts:
        raise ValueError("parts must be non-empty")
    return SampledBatchStep(
        exec_action_np=np.concatenate([p.exec_action_np for p in parts], axis=0),
        raw_postprocessed_action_np=np.concatenate(
            [p.raw_postprocessed_action_np for p in parts], axis=0
        ),
        policy_tensor=torch.cat([p.policy_tensor for p in parts], dim=0),
        unsquashed=torch.cat([p.unsquashed for p in parts], dim=0),
        log_prob=torch.cat([p.log_prob.reshape(-1) for p in parts], dim=0),
        action_clip_fraction=np.concatenate(
            [np.asarray(p.action_clip_fraction).reshape(-1) for p in parts], axis=0
        ),
        action_clip_any=np.concatenate(
            [np.asarray(p.action_clip_any).reshape(-1) for p in parts], axis=0
        ),
    )


def concat_sampled_action_chunk_batches(parts: list[SampledActionChunkBatch]) -> SampledActionChunkBatch:
    if not parts:
        raise ValueError("parts must be non-empty")
    return SampledActionChunkBatch(
        exec_action_np=np.concatenate([p.exec_action_np for p in parts], axis=0),
        raw_postprocessed_action_np=np.concatenate(
            [p.raw_postprocessed_action_np for p in parts], axis=0
        ),
        policy_tensor=torch.cat([p.policy_tensor for p in parts], dim=0),
        unsquashed_chunk=torch.cat([p.unsquashed_chunk for p in parts], dim=0),
        log_prob_steps=torch.cat([p.log_prob_steps for p in parts], dim=0),
        log_prob_sum=torch.cat([p.log_prob_sum.reshape(-1) for p in parts], dim=0),
        action_clip_fraction=np.concatenate([p.action_clip_fraction for p in parts], axis=0),
        action_clip_any=np.concatenate([p.action_clip_any for p in parts], axis=0),
    )


def select_proc_rows(proc: dict[str, Any], rows: list[int], *, batch_size: int) -> dict[str, Any]:
    """Compact full vector preprocessor output down to active rows."""
    if not rows:
        raise ValueError("rows must be non-empty")
    out: dict[str, Any] = {}
    for key, value in proc.items():
        if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == int(batch_size):
            out[key] = value[rows]
        elif isinstance(value, np.ndarray) and value.ndim > 0 and int(value.shape[0]) == int(batch_size):
            out[key] = np.asarray(value[rows])
        elif isinstance(value, (list, tuple)) and len(value) == int(batch_size):
            out[key] = [value[int(row)] for row in rows]
        else:
            out[key] = copy.deepcopy(value)
    return out


def _sample_policy_for_active_rows(
    old_wrapper: MetaWorldSmolVLAGRPOPolicy,
    proc: dict[str, Any],
    *,
    active_rows: list[int],
    rngs: list[torch.Generator],
    action_chunk_size: int,
    effective_chunk: int,
    rollout_policy_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_active = len(active_rows)
    slices = iter_compact_row_slices(n_active, rollout_policy_batch_size)
    if action_chunk_size == 1:
        parts = []
        for start, end in slices:
            sub_rows = list(range(start, end))
            proc_sub = select_proc_rows(proc, sub_rows, batch_size=n_active)
            parts.append(
                old_wrapper.sample_action_batch_from_proc(
                    proc_sub,
                    n_envs=end - start,
                    rngs=[rngs[active_rows[row]] for row in sub_rows],
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        batch = concat_sampled_action_batches(parts)
        chunk_exec_actions = batch.exec_action_np[:, None, :].astype(np.float32, copy=False)
        chunk_unsquashed = batch.unsquashed.detach().cpu().reshape(n_active, 1, -1)
        chunk_log_probs = batch.log_prob.detach().cpu().reshape(n_active, 1)
        chunk_clip_fraction = np.asarray(batch.action_clip_fraction, dtype=np.float64).reshape(
            n_active, 1
        )
        chunk_clip_any = np.asarray(batch.action_clip_any, dtype=np.bool_).reshape(n_active, 1)
    else:
        parts = []
        for start, end in slices:
            sub_rows = list(range(start, end))
            proc_sub = select_proc_rows(proc, sub_rows, batch_size=n_active)
            parts.append(
                old_wrapper.sample_action_chunk_batch_from_proc(
                    proc_sub,
                    n_envs=end - start,
                    chunk_len=effective_chunk,
                    rngs=[rngs[active_rows[row]] for row in sub_rows],
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        batch = concat_sampled_action_chunk_batches(parts)
        chunk_exec_actions = batch.exec_action_np.astype(np.float32, copy=False).reshape(
            n_active, effective_chunk, -1
        )
        chunk_unsquashed = batch.unsquashed_chunk.detach().cpu().reshape(
            n_active, effective_chunk, -1
        )
        chunk_log_probs = batch.log_prob_steps.detach().cpu().reshape(n_active, effective_chunk)
        chunk_clip_fraction = np.asarray(batch.action_clip_fraction, dtype=np.float64).reshape(
            n_active, effective_chunk
        )
        chunk_clip_any = np.asarray(batch.action_clip_any, dtype=np.bool_).reshape(
            n_active, effective_chunk
        )
    return chunk_exec_actions, chunk_unsquashed, chunk_log_probs, chunk_clip_fraction, chunk_clip_any


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
    action_chunk_size: int = 1,
    rollout_policy_batch_size: int = 32,
) -> list[RolloutTrajectory]:
    """Collect `group_size` one-episode trajectories with one vector env step per timestep."""
    if rollout_execution not in ("vector_sync", "vector_async"):
        raise ValueError("rollout_execution must be 'vector_sync' or 'vector_async'")
    if group_size < 1:
        raise ValueError("group_size must be >= 1")

    requested_max_steps = int(max_steps)
    action_chunk_size_i = _validate_action_chunk_size(action_chunk_size)
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
            tr.metadata["action_chunk_size"] = action_chunk_size_i
            tr.metadata["rollout_policy_batch_size"] = int(rollout_policy_batch_size)
            tr.metadata["policy_sample_calls"] = 0

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
        step_count = 0
        while step_count < max_steps_i:
            if not bool(np.any(active)):
                break
            active_rows = [idx for idx in range(group_size) if bool(active[idx])]
            proc_full = env_h.build_proc(obs, bundle=bundle)
            proc = select_proc_rows(proc_full, active_rows, batch_size=group_size)
            effective_chunk = min(action_chunk_size_i, int(max_steps_i) - int(step_count))
            with torch.no_grad():
                (
                    chunk_exec_actions,
                    chunk_unsquashed,
                    chunk_log_probs,
                    chunk_clip_fraction,
                    chunk_clip_any,
                ) = _sample_policy_for_active_rows(
                    old_wrapper,
                    proc,
                    active_rows=active_rows,
                    rngs=rngs,
                    action_chunk_size=action_chunk_size_i,
                    effective_chunk=effective_chunk,
                    rollout_policy_batch_size=int(rollout_policy_batch_size),
                )

            active_proc_snapshots = [
                detach_proc_snapshot(slice_proc_row(proc, batch_row)) for batch_row in range(len(active_rows))
            ]
            executed_by_batch_row = np.zeros((len(active_rows),), dtype=np.int64)
            chunk_start_step = int(step_count)
            for batch_row, row in enumerate(active_rows):
                rollouts[row].metadata["policy_sample_calls"] = (
                    int(rollouts[row].metadata.get("policy_sample_calls", 0)) + 1
                )

            for chunk_step in range(effective_chunk):
                if not bool(np.any(active)):
                    break
                active_before_step = active.copy()
                action_matrix = np.zeros((group_size, int(action_dim)), dtype=np.float32)
                for batch_row, row in enumerate(active_rows):
                    if bool(active_before_step[row]):
                        action_matrix[row] = chunk_exec_actions[batch_row, chunk_step]

                env_batch = env_h.step_batch(action_matrix)
                obs = env_batch.observation

                for batch_row, row in enumerate(active_rows):
                    if not bool(active_before_step[row]):
                        continue
                    rollouts[row].proc_snapshots.append(detach_proc_snapshot(active_proc_snapshots[batch_row]))
                    rollouts[row].exec_actions.append(
                        chunk_exec_actions[batch_row, chunk_step].reshape(-1).tolist()
                    )
                    rollouts[row].unsquashed_actions.append(
                        chunk_unsquashed[batch_row, chunk_step : chunk_step + 1].detach().cpu()
                    )
                    rollouts[row].log_probs.append(
                        chunk_log_probs[batch_row, chunk_step : chunk_step + 1].detach().cpu()
                    )
                    rollouts[row].action_clip_fractions.append(float(chunk_clip_fraction[batch_row, chunk_step]))
                    rollouts[row].action_clip_any.append(bool(chunk_clip_any[batch_row, chunk_step]))
                    rollouts[row].rewards.append(float(env_batch.reward[row]))
                    rollouts[row].successes.append(bool(env_batch.success[row]))
                    executed_by_batch_row[batch_row] += 1

                    if env_batch.success[row] or env_batch.terminated[row] or env_batch.truncated[row]:
                        active[row] = False
                        rollouts[row].terminated = bool(env_batch.terminated[row])
                        rollouts[row].truncated = bool(env_batch.truncated[row])
                step_count += 1
                if step_count >= int(max_steps_i):
                    break

            for batch_row, row in enumerate(active_rows):
                _record_executed_chunk(
                    rollouts[row],
                    proc_snapshot=active_proc_snapshots[batch_row],
                    unsquashed_chunk=chunk_unsquashed[batch_row],
                    log_prob_steps=chunk_log_probs[batch_row],
                    start_step=chunk_start_step,
                    executed_steps=int(executed_by_batch_row[batch_row]),
                    logprob_mode="step" if action_chunk_size_i == 1 else "chunk",
                )

        return rollouts
    finally:
        env_h.close()
