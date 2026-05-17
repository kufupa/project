from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from statistics import mean
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


def _reset_policy(policy: Any) -> None:
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()


def select_eval_action_queue_free(policy: Any, proc: dict[str, Any]) -> torch.Tensor:
    """Return first eval action without using SmolVLA's cross-step action queue."""

    if all(hasattr(policy, name) for name in ("_prepare_batch", "prepare_images", "prepare_state")) and hasattr(
        getattr(policy, "model", None), "sample_actions"
    ):
        batch = policy._prepare_batch(proc)
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]
        actions = policy.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=None,
        )
        if not torch.is_tensor(actions):
            raise RuntimeError("SmolVLA sample_actions must return a tensor during vector eval")
        action_dim = int(policy.config.action_feature.shape[0])
        return actions[:, 0, :action_dim]

    return policy.select_action(proc)


def select_eval_action_chunk_queue_free(policy: Any, proc: dict[str, Any], *, chunk_len: int) -> torch.Tensor:
    """Return an eval action chunk without using SmolVLA's cross-step action queue."""

    if int(chunk_len) < 1:
        raise ValueError("chunk_len must be >= 1")
    if all(hasattr(policy, name) for name in ("_prepare_batch", "prepare_images", "prepare_state")) and hasattr(
        getattr(policy, "model", None), "sample_actions"
    ):
        batch = policy._prepare_batch(proc)
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]
        actions = policy.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=None,
        )
        if not torch.is_tensor(actions):
            raise RuntimeError("SmolVLA sample_actions must return a tensor during vector eval")
        action_dim = int(policy.config.action_feature.shape[0])
        actions = actions[:, :, :action_dim] if actions.ndim == 3 else actions.reshape(actions.shape[0], -1, action_dim)
        if int(actions.shape[1]) < int(chunk_len):
            raise RuntimeError(f"SmolVLA returned {int(actions.shape[1])} actions, requested chunk_len={int(chunk_len)}")
        return actions[:, : int(chunk_len), :]

    first = select_eval_action_queue_free(policy, proc)
    if int(chunk_len) != 1:
        raise RuntimeError("Chunked baseline eval requires SmolVLA model.sample_actions when chunk_len > 1")
    return first.unsqueeze(1)


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
    if rollout_execution != "vector_sync":
        raise ValueError("manual-pool eval currently supports rollout_execution='vector_sync' only")
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
        envs = [OfficialLeRobotMetaWorldGRPORollout(task=task, n_envs=1) for _ in range(wave_n)]
        try:
            resolved_steps = resolve_lerobot_horizon(envs[0], max_steps)
            obs_by_row = [env.reset(seed) for env, (_ep, seed) in zip(envs, wave, strict=True)]
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
                proc_rows = [envs[idx].build_proc(obs_by_row[idx], bundle=bundle) for idx in active_rows]
                proc = concatenate_proc_rows(proc_rows)
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
                    for batch_row, row in enumerate(active_rows):
                        if not bool(active[row]):
                            continue
                        step = envs[row].step(exec_action_np[batch_row, chunk_step : chunk_step + 1])
                        obs_by_row[row] = step.observation
                        actions[row].append(exec_action_np[batch_row, chunk_step].reshape(-1).tolist())
                        rewards[row].append(float(step.reward))
                        successes[row].append(bool(step.success))
                        if step.success or step.terminated or step.truncated:
                            active[row] = False
                            terminated[row] = bool(step.terminated)
                            truncated[row] = bool(step.truncated)
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
            for env in envs:
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
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

