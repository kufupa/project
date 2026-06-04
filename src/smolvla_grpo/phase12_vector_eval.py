from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np
import torch

from smolvla_grpo.checkpointing import load_grpo_checkpoint
from smolvla_pipeline.evaluator import write_episode_artifacts


@dataclass(frozen=True)
class EpisodeResult:
    episode_index: int
    reset_seed: int
    actions: list[list[float]]
    rewards: list[float]
    successes: list[bool]
    terminated: bool
    truncated: bool


TIMING_SECOND_FIELDS = (
    "load_bundle_seconds",
    "rollout_seconds",
    "reset_seconds",
    "proc_build_seconds",
    "policy_prepare_seconds",
    "policy_forward_seconds",
    "postprocess_seconds",
    "action_coerce_seconds",
    "metaworld_step_seconds_including_obs_render",
    "metaworld_step_batch_seconds_including_obs_render",
    "frame_extract_seconds",
    "video_write_seconds",
    "close_seconds",
)

TIMING_COUNT_FIELDS = ("n_policy_calls", "n_env_steps", "n_env_batch_steps", "n_video_frames")


@dataclass
class TimingAccumulator:
    seconds: dict[str, float] = field(default_factory=lambda: {key: 0.0 for key in TIMING_SECOND_FIELDS})
    counts: dict[str, int] = field(default_factory=lambda: {key: 0 for key in TIMING_COUNT_FIELDS})
    cuda_sync_requested: bool = True
    cuda_synchronized_forward_timing: bool = False

    def add(self, key: str, value: float) -> None:
        if key not in self.seconds:
            raise KeyError(f"unknown timing seconds field: {key}")
        self.seconds[key] += float(value)

    def incr(self, key: str, value: int = 1) -> None:
        if key not in self.counts:
            raise KeyError(f"unknown timing count field: {key}")
        self.counts[key] += int(value)

    def summary(self) -> dict[str, Any]:
        policy_calls = max(int(self.counts["n_policy_calls"]), 1)
        env_steps = max(int(self.counts["n_env_steps"]), 1)
        env_batch_steps = max(int(self.counts["n_env_batch_steps"]), 1)
        video_frames = max(int(self.counts["n_video_frames"]), 1)
        return {
            "schema_version": "phase58_timing_v1",
            **{key: float(self.seconds.get(key, 0.0)) for key in TIMING_SECOND_FIELDS},
            **{key: int(self.counts.get(key, 0)) for key in TIMING_COUNT_FIELDS},
            "mean_policy_forward_ms_per_call": float(1000.0 * self.seconds["policy_forward_seconds"] / policy_calls),
            "mean_metaworld_step_ms_per_env_step": float(
                1000.0 * self.seconds["metaworld_step_seconds_including_obs_render"] / env_steps
            ),
            "mean_metaworld_step_batch_ms": float(
                1000.0 * self.seconds["metaworld_step_batch_seconds_including_obs_render"] / env_batch_steps
            ),
            "mean_video_write_ms_per_frame": float(1000.0 * self.seconds["video_write_seconds"] / video_frames),
            "cuda_sync_requested": bool(self.cuda_sync_requested),
            "cuda_synchronized_forward_timing": bool(self.cuda_synchronized_forward_timing),
        }


def build_episode_waves(*, episodes: int, eval_seed_start: int, n_envs: int) -> list[list[tuple[int, int]]]:
    if int(n_envs) < 1:
        raise ValueError("n_envs must be >= 1")
    if int(episodes) < 1:
        raise ValueError("episodes must be >= 1")
    pairs = [(ep, int(eval_seed_start) + ep) for ep in range(int(episodes))]
    return [pairs[i : i + int(n_envs)] for i in range(0, len(pairs), int(n_envs))]


_ALLOWED_MISSING_KEYS = {"model.log_std"}


def _normalise_incompatible_keys(result: Any) -> tuple[list[str], list[str]]:
    if isinstance(result, tuple) and len(result) == 2:
        return list(result[0]), list(result[1])
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    return missing, unexpected


def validate_checkpoint_state(policy: Any, state: dict[str, Any]) -> None:
    expected = set(policy.state_dict().keys())
    supplied = set(state.keys())
    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - expected)
    bad_missing = [key for key in missing if key not in _ALLOWED_MISSING_KEYS]
    if bad_missing:
        raise RuntimeError(f"missing checkpoint keys: {bad_missing[:20]}")
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:20]}")


def load_policy_checkpoint_into_bundle(bundle: Any, checkpoint_path: Path) -> dict[str, Any]:
    payload = load_grpo_checkpoint(checkpoint_path.expanduser().resolve(), map_location="cpu")
    state = payload["policy_state_dict"]
    validate_checkpoint_state(bundle.policy, state)
    result = bundle.policy.load_state_dict(state, strict=False)
    missing, unexpected = _normalise_incompatible_keys(result)
    bad_missing = [key for key in missing if key not in _ALLOWED_MISSING_KEYS]
    if bad_missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch missing={bad_missing[:20]} unexpected={unexpected[:20]}")
    bundle.policy.eval()
    reset = getattr(bundle.policy, "reset", None)
    if callable(reset):
        reset()
    return payload


def write_eval_artifacts(
    *,
    base_checkpoint: str,
    grpo_checkpoint: Path | None,
    output_dir: Path,
    task: str,
    episodes: int,
    eval_seed_start: int,
    results: list[EpisodeResult],
) -> dict[str, Any]:
    ordered = sorted(results, key=lambda item: item.episode_index)
    if len(ordered) != int(episodes):
        raise RuntimeError(f"expected {episodes} episode results; got {len(ordered)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sum_rewards = [float(sum(item.rewards)) for item in ordered]
    max_rewards = [float(max(item.rewards)) if item.rewards else 0.0 for item in ordered]
    success_flags = [any(bool(v) for v in item.successes) for item in ordered]
    pc_success = 100.0 * sum(1 for value in success_flags if value) / max(len(success_flags), 1)

    rows = []
    for item, sr, mr, ok in zip(ordered, sum_rewards, max_rewards, success_flags, strict=True):
        ep_dir = output_dir / "episodes" / f"episode_{item.episode_index:04d}"
        write_episode_artifacts(
            episode_dir=ep_dir,
            actions=item.actions,
            rewards=item.rewards,
            successes=item.successes,
            overlay_mode="cumulative_reward",
        )
        rows.append(
            {
                "episode_index": int(item.episode_index),
                "reset_seed": int(item.reset_seed),
                "sum_reward": float(sr),
                "max_reward": float(mr),
                "success": bool(ok),
                "n_steps": len(item.rewards),
                "env_backend": "official_lerobot",
            }
        )

    summary = {
        "grpo_checkpoint": str(grpo_checkpoint) if grpo_checkpoint is not None else "base_checkpoint",
        "base_checkpoint": base_checkpoint,
        "task": task,
        "env_backend": "official_lerobot",
        "eval_seed_start": int(eval_seed_start),
        "episodes": int(episodes),
        "avg_sum_reward": float(mean(sum_rewards)) if sum_rewards else 0.0,
        "avg_max_reward": float(mean(max_rewards)) if max_rewards else 0.0,
        "pc_success": float(pc_success),
    }
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "eval_episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    eval_info = {
        "reset_randomization_mode": os.environ.get("SMOLVLA_METAWORLD_RESET_MODE", "random_seeded"),
        "per_task": [
            {
                "task_group": task,
                "task_id": 0,
                "metrics": {
                    "sum_rewards": sum_rewards,
                    "max_rewards": max_rewards,
                    "successes": success_flags,
                    "video_paths": [],
                },
            }
        ],
        "per_group": {
            task: {
                "avg_sum_reward": summary["avg_sum_reward"],
                "avg_max_reward": summary["avg_max_reward"],
                "pc_success": summary["pc_success"],
                "n_episodes": len(sum_rewards),
                "video_paths": [],
            }
        },
        "overall": {
            "avg_sum_reward": summary["avg_sum_reward"],
            "avg_max_reward": summary["avg_max_reward"],
            "pc_success": summary["pc_success"],
            "n_episodes": len(sum_rewards),
            "video_paths": [],
        },
    }
    (output_dir / "eval_info.json").write_text(json.dumps(eval_info, indent=2), encoding="utf-8")
    return summary


def coerce_exec_action_batch(action: Any, *, action_dim: int, n_envs: int) -> np.ndarray:
    if hasattr(action, "detach"):
        action_np = action.detach().float().cpu().numpy()
    else:
        action_np = np.asarray(action, dtype=np.float32)
    action_np = np.asarray(action_np, dtype=np.float32)
    expected_size = int(n_envs) * int(action_dim)
    if action_np.size != expected_size:
        raise RuntimeError(
            f"Policy action dim mismatch: expected batch ({n_envs}, {action_dim}) "
            f"with {expected_size} values, got shape {tuple(action_np.shape)} and size {action_np.size}. "
            "Refusing silent pad/truncate."
        )
    return np.clip(action_np.reshape(int(n_envs), int(action_dim)), -1.0, 1.0).astype(np.float32, copy=False)


def coerce_exec_action_chunk_batch(action: Any, *, action_dim: int, n_envs: int, chunk_len: int) -> np.ndarray:
    if hasattr(action, "detach"):
        action_np = action.detach().float().cpu().numpy()
    else:
        action_np = np.asarray(action, dtype=np.float32)
    action_np = np.asarray(action_np, dtype=np.float32)
    expected_size = int(n_envs) * int(chunk_len) * int(action_dim)
    if action_np.size != expected_size:
        raise RuntimeError(
            f"Policy action chunk dim mismatch: expected batch ({n_envs}, {chunk_len}, {action_dim}) "
            f"with {expected_size} values, got shape {tuple(action_np.shape)} and size {action_np.size}. "
            "Refusing silent pad/truncate."
        )
    return np.clip(action_np.reshape(int(n_envs), int(chunk_len), int(action_dim)), -1.0, 1.0).astype(
        np.float32,
        copy=False,
    )


def concatenate_proc_rows(proc_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not proc_rows:
        raise ValueError("proc_rows must be non-empty")
    out: dict[str, Any] = {}
    keys = proc_rows[0].keys()
    for key in keys:
        vals = [row[key] for row in proc_rows]
        first = vals[0]
        if torch.is_tensor(first):
            out[key] = torch.cat(vals, dim=0)
        elif isinstance(first, np.ndarray):
            out[key] = np.concatenate(vals, axis=0)
        elif isinstance(first, list):
            merged: list[Any] = []
            for value in vals:
                merged.extend(value)
            out[key] = merged
        else:
            out[key] = vals
    return out


def select_proc_rows(proc: dict[str, Any], rows: list[int], *, batch_size: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must be non-empty")
    out: dict[str, Any] = {}
    for key, value in proc.items():
        if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == int(batch_size):
            out[key] = value[rows]
        elif isinstance(value, np.ndarray) and value.ndim > 0 and int(value.shape[0]) == int(batch_size):
            out[key] = value[rows]
        elif isinstance(value, (list, tuple)) and len(value) == int(batch_size):
            out[key] = [value[int(row)] for row in rows]
        else:
            out[key] = value
    return out


def _reset_policy(policy: Any) -> None:
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()


def _sync_cuda_for_timing(policy: Any, timings: TimingAccumulator | None) -> None:
    if timings is not None and not timings.cuda_sync_requested:
        return
    if not torch.cuda.is_available():
        return
    device = getattr(policy, "device", None)
    try:
        torch.cuda.synchronize(device=device)
    except Exception:
        torch.cuda.synchronize()
    if timings is not None:
        timings.cuda_synchronized_forward_timing = True


def _timed_sample_actions(
    policy: Any,
    proc: dict[str, Any],
    *,
    timings: TimingAccumulator | None,
) -> tuple[torch.Tensor, int]:
    prepare_t0 = perf_counter()
    batch = policy._prepare_batch(proc)
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]
    if timings is not None:
        timings.add("policy_prepare_seconds", perf_counter() - prepare_t0)

    _sync_cuda_for_timing(policy, timings)
    forward_t0 = perf_counter()
    actions = policy.model.sample_actions(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
    )
    _sync_cuda_for_timing(policy, timings)
    if timings is not None:
        timings.add("policy_forward_seconds", perf_counter() - forward_t0)
        timings.incr("n_policy_calls")

    if not torch.is_tensor(actions):
        raise RuntimeError("SmolVLA sample_actions must return a tensor during vector eval")
    return actions, int(policy.config.action_feature.shape[0])


def select_eval_action_queue_free_timed(
    policy: Any,
    proc: dict[str, Any],
    *,
    timings: TimingAccumulator | None,
) -> torch.Tensor:
    """Return first eval action without using SmolVLA's cross-step action queue."""

    if all(hasattr(policy, name) for name in ("_prepare_batch", "prepare_images", "prepare_state")) and hasattr(
        getattr(policy, "model", None), "sample_actions"
    ):
        actions, action_dim = _timed_sample_actions(policy, proc, timings=timings)
        return actions[:, 0, :action_dim]

    if timings is not None:
        timings.incr("n_policy_calls")
    return policy.select_action(proc)


def select_eval_action_queue_free(policy: Any, proc: dict[str, Any]) -> torch.Tensor:
    """Return first eval action without using SmolVLA's cross-step action queue."""

    return select_eval_action_queue_free_timed(policy, proc, timings=None)


def select_eval_action_chunk_queue_free_timed(
    policy: Any,
    proc: dict[str, Any],
    *,
    chunk_len: int,
    timings: TimingAccumulator | None,
) -> torch.Tensor:
    """Return an eval action chunk without using SmolVLA's cross-step action queue."""

    if int(chunk_len) < 1:
        raise ValueError("chunk_len must be >= 1")
    if all(hasattr(policy, name) for name in ("_prepare_batch", "prepare_images", "prepare_state")) and hasattr(
        getattr(policy, "model", None), "sample_actions"
    ):
        actions, action_dim = _timed_sample_actions(policy, proc, timings=timings)
        actions = actions[:, :, :action_dim] if actions.ndim == 3 else actions.reshape(actions.shape[0], -1, action_dim)
        if int(actions.shape[1]) < int(chunk_len):
            raise RuntimeError(f"SmolVLA returned {int(actions.shape[1])} actions, requested chunk_len={int(chunk_len)}")
        return actions[:, : int(chunk_len), :]

    first = select_eval_action_queue_free(policy, proc)
    if int(chunk_len) != 1:
        raise RuntimeError("Chunked baseline eval requires SmolVLA model.sample_actions when chunk_len > 1")
    return first.unsqueeze(1)


def select_eval_action_chunk_queue_free(policy: Any, proc: dict[str, Any], *, chunk_len: int) -> torch.Tensor:
    """Return an eval action chunk without using SmolVLA's cross-step action queue."""

    return select_eval_action_chunk_queue_free_timed(policy, proc, chunk_len=chunk_len, timings=None)


def _resolve_action_dim(task: str) -> int:
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    probe = OfficialLeRobotMetaWorldGRPORollout(task=task, n_envs=1)
    try:
        return int(probe.action_dim)
    finally:
        probe.close()


def evaluate_loaded_policy_vectorized(
    *,
    bundle: Any,
    base_checkpoint: str,
    grpo_checkpoint: Path | None,
    output_dir: Path,
    task: str,
    episodes: int,
    eval_seed_start: int,
    n_envs: int,
    rollout_execution: str,
    max_steps: int,
    chunk_len: int = 1,
) -> dict[str, Any]:
    if rollout_execution not in ("vector_sync", "vector_async"):
        raise ValueError("vector eval supports rollout_execution='vector_sync' or 'vector_async'")
    if int(chunk_len) < 1:
        raise ValueError("chunk_len must be >= 1")

    from smolvla_grpo.lerobot_metaworld_adapter import (
        OfficialLeRobotMetaWorldGRPORollout,
        resolve_lerobot_horizon,
    )

    action_dim = _resolve_action_dim(task)
    waves = build_episode_waves(episodes=episodes, eval_seed_start=eval_seed_start, n_envs=n_envs)
    all_results: list[EpisodeResult] = []

    for wave in waves:
        wave_n = len(wave)
        env = OfficialLeRobotMetaWorldGRPORollout(
            task=task,
            n_envs=wave_n,
            use_async_envs=str(rollout_execution) == "vector_async",
        )
        try:
            resolved_steps = resolve_lerobot_horizon(env, max_steps)
            obs = env.reset_many([seed for _ep, seed in wave])
            _reset_policy(bundle.policy)
            active = np.ones((wave_n,), dtype=np.bool_)
            actions: list[list[list[float]]] = [[] for _ in range(wave_n)]
            rewards: list[list[float]] = [[] for _ in range(wave_n)]
            successes: list[list[bool]] = [[] for _ in range(wave_n)]
            terminated = [False for _ in range(wave_n)]
            truncated = [False for _ in range(wave_n)]

            step_count = 0
            while step_count < int(resolved_steps):
                if not bool(np.any(active)):
                    break
                active_rows = [idx for idx in range(wave_n) if bool(active[idx])]
                proc = select_proc_rows(env.build_proc(obs, bundle=bundle), active_rows, batch_size=wave_n)
                effective_chunk = min(int(chunk_len), int(resolved_steps) - int(step_count))
                with torch.inference_mode():
                    if int(chunk_len) == 1:
                        action = select_eval_action_queue_free(bundle.policy, proc)
                        post = bundle.postprocessor(action)
                        exec_action_np = coerce_exec_action_batch(
                            post,
                            action_dim=action_dim,
                            n_envs=len(active_rows),
                        )[:, None, :]
                    else:
                        action = select_eval_action_chunk_queue_free(
                            bundle.policy,
                            proc,
                            chunk_len=effective_chunk,
                        )
                        post = bundle.postprocessor(action)
                        exec_action_np = coerce_exec_action_chunk_batch(
                            post,
                            action_dim=action_dim,
                            n_envs=len(active_rows),
                            chunk_len=effective_chunk,
                        )

                for chunk_step in range(effective_chunk):
                    if not bool(np.any(active)):
                        break
                    active_before_step = active.copy()
                    action_matrix = np.zeros((wave_n, int(env.action_dim)), dtype=np.float32)
                    for batch_row, row in enumerate(active_rows):
                        if not bool(active_before_step[row]):
                            continue
                        action_matrix[row] = exec_action_np[batch_row, chunk_step]
                    step = env.step_batch(action_matrix)
                    obs = step.observation
                    for batch_row, row in enumerate(active_rows):
                        if not bool(active_before_step[row]):
                            continue
                        actions[row].append(exec_action_np[batch_row, chunk_step].reshape(-1).tolist())
                        rewards[row].append(float(step.reward[row]))
                        successes[row].append(bool(step.success[row]))
                        if step.success[row] or step.terminated[row] or step.truncated[row]:
                            active[row] = False
                            terminated[row] = bool(step.terminated[row])
                            truncated[row] = bool(step.truncated[row])
                    step_count += 1
                    if step_count >= int(resolved_steps):
                        break

            for row, (episode_index, reset_seed) in enumerate(wave):
                all_results.append(
                    EpisodeResult(
                        episode_index=int(episode_index),
                        reset_seed=int(reset_seed),
                        actions=actions[row],
                        rewards=rewards[row],
                        successes=successes[row],
                        terminated=terminated[row],
                        truncated=truncated[row]
                        or (len(rewards[row]) >= int(resolved_steps) and not any(successes[row])),
                    )
                )
        finally:
            env.close()

    summary = write_eval_artifacts(
        base_checkpoint=base_checkpoint,
        grpo_checkpoint=grpo_checkpoint,
        output_dir=output_dir,
        task=task,
        episodes=episodes,
        eval_seed_start=eval_seed_start,
        results=all_results,
    )
    summary["n_envs"] = int(n_envs)
    summary["max_steps"] = int(max_steps)
    summary["chunk_len"] = int(chunk_len)
    summary["rollout_execution"] = str(rollout_execution)
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

