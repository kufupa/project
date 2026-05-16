"""EGGROLL trainer for SmolVLA MetaWorld."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np
import torch

from smolvla_grpo.checkpointing import load_grpo_checkpoint, save_grpo_checkpoint
from smolvla_grpo.eggroll_linear import install_eggroll_linear_patch
from smolvla_grpo.eggroll_noise import (
    EggrollLayerSpec,
    EggrollNoiseManager,
    discover_eggroll_layers,
    modules_for_specs,
)
from smolvla_grpo.eggroll_rollout import collect_eggroll_population_rollouts
from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout
from smolvla_grpo.phase11_rollout import load_bundle_for_grpo
from smolvla_grpo.phase12_diagnostics import write_phase12_episode_video
from smolvla_grpo.phase12_logging import (
    assert_smoke_manifest_contract,
    utc_now_iso,
    write_jsonl_row,
    write_manifest,
)
from smolvla_pipeline.evaluator import _resolve_task_text


ROLLOUT_TIMING_FIELDS = (
    "env_init_seconds",
    "reset_seconds",
    "proc_build_seconds",
    "forward_seconds",
    "postprocess_seconds",
    "env_step_seconds",
    "rollout_seconds",
)


@dataclass
class EggrollTrainerConfig:
    checkpoint: str
    output_dir: Path
    task: str = "push-v3"
    population_size: int = 32
    population_batch_size: int = 4
    rank: int = 2
    sigma: float = 0.01
    alpha: float = 0.03
    baseline_type: str = "mean"
    fitness_shaping: str = "centered"
    antithetic: bool = True
    episodes_per_member: int = 1
    num_iterations: int = 100
    max_steps: int = 120
    rollout_execution: str = "vector_sync"
    async_start_method: str = "forkserver"
    train_scope: str = "action_expert"
    train_seed_base: int = 2000
    noise_seed: int = 17
    flow_noise_seed: int = 23
    save_every: int = 10
    video_every: int = 10
    video_member_id: int = 0
    write_oracle_video: bool = False
    resume: Path | None = None
    abort_update_norm: float = 0.05


def compute_baseline(values: list[float], baseline_type: str) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("fitness values cannot be empty")
    if baseline_type == "none":
        return 0.0
    if baseline_type == "mean":
        return float(np.mean(arr))
    if baseline_type == "median":
        return float(np.median(arr))
    raise ValueError("baseline_type must be 'none', 'mean', or 'median'")


def shape_fitness(values: list[float], *, baseline_type: str, fitness_shaping: str) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if fitness_shaping == "centered":
        return (arr - compute_baseline(values, baseline_type)).astype(np.float64).tolist()
    if fitness_shaping == "rank":
        if arr.size < 2 or float(np.max(arr) - np.min(arr)) == 0.0:
            return np.zeros_like(arr, dtype=np.float64).tolist()
        order = np.argsort(arr, kind="mergesort")
        ranks = np.empty(arr.shape, dtype=np.float64)
        i = 0
        while i < arr.size:
            j = i + 1
            while j < arr.size and arr[order[j]] == arr[order[i]]:
                j += 1
            avg_rank = 0.5 * float(i + j - 1)
            ranks[order[i:j]] = avg_rank
            i = j
        centered = ranks - float(np.mean(ranks))
        denom = float(np.std(centered))
        if denom > 0:
            centered = centered / denom
        return centered.tolist()
    raise ValueError("fitness_shaping must be 'centered' or 'rank'")


def compute_fitness_stats(values: list[float], successes: list[bool]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "fitness_mean": float(np.mean(arr)) if arr.size else 0.0,
        "fitness_best": float(np.max(arr)) if arr.size else 0.0,
        "fitness_min": float(np.min(arr)) if arr.size else 0.0,
        "fitness_std": float(np.std(arr)) if arr.size else 0.0,
        "success_rate": float(np.mean(np.asarray(successes, dtype=np.float64))) if successes else 0.0,
    }


def apply_es_update(
    *,
    modules: dict[int, torch.nn.Linear],
    specs: list[EggrollLayerSpec],
    noise_manager: EggrollNoiseManager,
    shaped_fitness: list[float],
    iteration: int,
    alpha: float,
    max_relative_update_norm: float | None = None,
) -> dict[str, float]:
    if len(shaped_fitness) < 1:
        raise ValueError("shaped_fitness cannot be empty")
    rels: list[float] = []
    max_abs = 0.0
    clipped = 0
    unclipped_rels: list[float] = []
    for spec in specs:
        module = modules[int(spec.layer_id)]
        update = torch.zeros_like(module.weight)
        for member_id, fitness in enumerate(shaped_fitness):
            a, b, sign = noise_manager.generate_factors(
                spec,
                member_id=member_id,
                iteration=int(iteration),
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            update.add_(a @ b.T, alpha=float(fitness) * float(sign))
        update.mul_(float(alpha) / float(len(shaped_fitness)) / math.sqrt(float(noise_manager.rank)))
        before = float(torch.linalg.vector_norm(module.weight.detach()).item())
        upd = float(torch.linalg.vector_norm(update.detach()).item())
        rel = upd / max(before, 1e-12)
        unclipped_rels.append(rel)
        if max_relative_update_norm is not None and rel > float(max_relative_update_norm):
            update.mul_(float(max_relative_update_norm) / max(rel, 1e-12))
            clipped += 1
        with torch.no_grad():
            module.weight.add_(update)
            upd = float(torch.linalg.vector_norm(update.detach()).item())
        rels.append(upd / max(before, 1e-12))
        max_abs = max(max_abs, float(torch.max(torch.abs(update)).item()) if update.numel() else 0.0)
    return {
        "relative_update_norm": float(max(rels) if rels else 0.0),
        "mean_relative_update_norm": float(mean(rels) if rels else 0.0),
        "unclipped_relative_update_norm": float(max(unclipped_rels) if unclipped_rels else 0.0),
        "update_clipped_layer_count": int(clipped),
        "max_abs_update": float(max_abs),
    }


def cuda_memory_metrics() -> dict[str, float | bool]:
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "cuda_max_memory_allocated_gb": 0.0,
            "cuda_max_memory_reserved_gb": 0.0,
            "cuda_total_memory_gb": 0.0,
        }
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "cuda_available": True,
        "cuda_max_memory_allocated_gb": float(torch.cuda.max_memory_allocated(idx) / 1e9),
        "cuda_max_memory_reserved_gb": float(torch.cuda.max_memory_reserved(idx) / 1e9),
        "cuda_total_memory_gb": float(props.total_memory / 1e9),
    }


def _policy_state(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in policy.state_dict().items()}


def _write_checkpoint(
    *,
    path: Path,
    bundle: Any,
    config: EggrollTrainerConfig,
    iteration: int,
    specs: list[EggrollLayerSpec],
    extra: dict[str, Any],
) -> None:
    save_grpo_checkpoint(
        path,
        policy_state=_policy_state(bundle.policy),
        optimizer_state={},
        update_index=int(iteration),
        args={k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        extra={
            "trainer": "eggroll",
            "eggroll": {
                "rank": int(config.rank),
                "sigma": float(config.sigma),
                "alpha": float(config.alpha),
                "population_size": int(config.population_size),
                "population_batch_size": int(config.population_batch_size),
                "train_scope": str(config.train_scope),
                "layer_count": len(specs),
                **extra,
            },
        },
    )


def write_oracle_baseline_video(
    *,
    output_dir: Path,
    task: str,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    env_h = OfficialLeRobotMetaWorldGRPORollout(task=task, n_envs=1, enable_expert_oracle=True)
    frames: list[np.ndarray] = []
    rewards: list[float] = []
    successes: list[bool] = []
    try:
        env_h.reset(int(seed))
        for _ in range(int(max_steps)):
            frames.append(env_h.render_frame())
            action = env_h.expert_action().reshape(1, -1).astype(np.float32)
            step = env_h.step(action)
            rewards.append(float(step.reward))
            successes.append(bool(step.success))
            if step.success or step.terminated or step.truncated:
                break
    finally:
        env_h.close()
    video_path = output_dir / "oracle" / f"seed_{int(seed)}" / "oracle_baseline.mp4"
    if frames:
        write_phase12_episode_video(
            video_path=video_path,
            frames=frames,
            rewards=rewards,
            successes=successes,
            fps=20,
        )
    manifest = {
        "seed": int(seed),
        "video_path": str(video_path),
        "frame_count": len(frames),
        "reward_sum": float(sum(rewards)),
        "success_any": bool(any(successes)),
    }
    write_manifest(video_path.with_name("oracle_manifest.json"), manifest)
    return manifest


class EggrollTrainer:
    def __init__(self, config: EggrollTrainerConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.progress_path = self.output_dir / "progress.jsonl"
        self.timings_path = self.output_dir / "timings.jsonl"

    def run(self) -> dict[str, Any]:
        cfg = self.config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        t0 = perf_counter()
        bundle, action_dim = load_bundle_for_grpo(
            cfg.checkpoint,
            task=cfg.task,
            env_backend="official_lerobot",
            n_action_steps=1,
        )
        vla_load_seconds = perf_counter() - t0
        if cfg.resume is not None:
            payload = load_grpo_checkpoint(Path(cfg.resume), map_location="cpu")
            bundle.policy.load_state_dict(payload["policy_state_dict"], strict=False)
        bundle.policy.eval()
        task_text = _resolve_task_text(cfg.task)
        specs = discover_eggroll_layers(bundle.policy, train_scope=cfg.train_scope)
        modules = modules_for_specs(bundle.policy, specs)
        noise_manager = EggrollNoiseManager(
            base_seed=int(cfg.noise_seed),
            rank=int(cfg.rank),
            antithetic=bool(cfg.antithetic),
        )
        patch_handle = install_eggroll_linear_patch(modules, specs)
        oracle_manifest: dict[str, Any] | None = None
        try:
            if cfg.write_oracle_video:
                oracle_manifest = write_oracle_baseline_video(
                    output_dir=self.output_dir,
                    task=cfg.task,
                    seed=int(cfg.train_seed_base),
                    max_steps=int(cfg.max_steps),
                )

            manifest = {
                "trainer": "eggroll",
                "created_at": utc_now_iso(),
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                "reset_randomization_mode": os.environ.get("SMOLVLA_METAWORLD_RESET_MODE", "random_seeded"),
                "action_dim": int(action_dim),
                "layer_count": len(specs),
                "vla_load_seconds": float(vla_load_seconds),
            }
            write_manifest(self.output_dir / "train_manifest.json", manifest)

            last_summary: dict[str, Any] = {}
            for iteration in range(int(cfg.num_iterations)):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                iter_t0 = perf_counter()
                episode_results = [
                    collect_eggroll_population_rollouts(
                        bundle=bundle,
                        task=cfg.task,
                        task_text=task_text,
                        action_dim=action_dim,
                        population_size=int(cfg.population_size),
                        population_batch_size=int(cfg.population_batch_size),
                        iteration=iteration,
                        max_steps=int(cfg.max_steps),
                        train_seed_base=int(cfg.train_seed_base),
                        flow_noise_seed=int(cfg.flow_noise_seed),
                        rollout_seed_offset=episode_repeat,
                        noise_manager=noise_manager,
                        patch_handle=patch_handle,
                        sigma=float(cfg.sigma),
                        rollout_execution=cfg.rollout_execution,
                        async_start_method=cfg.async_start_method,
                        video_member_id=int(cfg.video_member_id),
                    )
                    for episode_repeat in range(int(cfg.episodes_per_member))
                ]
                rollout = episode_results[0]
                fitness_arrays = [np.asarray(item.fitness, dtype=np.float64) for item in episode_results]
                fitness = np.mean(np.stack(fitness_arrays, axis=0), axis=0).astype(np.float64).tolist()
                shaped = shape_fitness(
                    fitness,
                    baseline_type=cfg.baseline_type,
                    fitness_shaping=cfg.fitness_shaping,
                )
                update_t0 = perf_counter()
                update_stats = apply_es_update(
                    modules=modules,
                    specs=specs,
                    noise_manager=noise_manager,
                    shaped_fitness=shaped,
                    iteration=iteration,
                    alpha=float(cfg.alpha),
                    max_relative_update_norm=float(cfg.abort_update_norm),
                )
                es_update_seconds = perf_counter() - update_t0
                if not math.isfinite(update_stats["relative_update_norm"]):
                    raise RuntimeError("non-finite relative_update_norm")

                successes = []
                for member_idx in range(int(cfg.population_size)):
                    successes.append(
                        any(
                            any(
                                any(r.successes)
                                for r in sorted(item.rollouts, key=lambda r: r.member_id)
                                if int(r.member_id) == member_idx
                            )
                            for item in episode_results
                        )
                    )
                stats = compute_fitness_stats(fitness, successes)
                ckpt_t0 = perf_counter()
                ckpt_dir = self.output_dir / "checkpoints"
                latest_path = ckpt_dir / "latest.pt"
                checkpoint_update = iteration + 1
                should_number = checkpoint_update % int(cfg.save_every) == 0 or iteration == int(cfg.num_iterations) - 1
                numbered_path = ckpt_dir / f"update_{checkpoint_update:04d}.pt"
                _write_checkpoint(
                    path=latest_path,
                    bundle=bundle,
                    config=cfg,
                    iteration=iteration,
                    specs=specs,
                    extra=update_stats,
                )
                if should_number:
                    _write_checkpoint(
                        path=numbered_path,
                        bundle=bundle,
                        config=cfg,
                        iteration=iteration,
                        specs=specs,
                        extra=update_stats,
                    )
                checkpoint_seconds = perf_counter() - ckpt_t0

                video_path = ""
                video_seconds = 0.0
                if cfg.video_every > 0 and (iteration == 0 or checkpoint_update % int(cfg.video_every) == 0):
                    video_t0 = perf_counter()
                    path = (
                        self.output_dir
                        / "rollouts"
                        / f"iteration_{iteration:04d}"
                        / f"member_{int(cfg.video_member_id):04d}"
                        / "selected_action_rollout.mp4"
                    )
                    if rollout.selected_frames:
                        write_phase12_episode_video(
                            video_path=path,
                            frames=rollout.selected_frames,
                            rewards=rollout.selected_rewards,
                            successes=rollout.selected_successes,
                            fps=20,
                        )
                        video_path = str(path)
                    video_seconds = perf_counter() - video_t0

                iteration_seconds = perf_counter() - iter_t0
                rollout_timing = {
                    key: float(sum(float(item.timings.get(key, 0.0)) for item in episode_results))
                    for key in ROLLOUT_TIMING_FIELDS
                }
                timing_row = {
                    "event": "eggroll_iteration_timing",
                    "iteration": iteration,
                    "vla_load_seconds": float(vla_load_seconds if iteration == 0 else 0.0),
                    **rollout_timing,
                    "es_update_seconds": float(es_update_seconds),
                    "checkpoint_seconds": float(checkpoint_seconds),
                    "video_seconds": float(video_seconds),
                    "iteration_seconds": float(iteration_seconds),
                    "population_size": int(cfg.population_size),
                    "population_batch_size": int(cfg.population_batch_size),
                    "num_env_steps": int(
                        sum(len(r.rewards) for item in episode_results for r in item.rollouts)
                    ),
                    **cuda_memory_metrics(),
                }
                write_jsonl_row(self.timings_path, timing_row)
                progress_row = {
                    **timing_row,
                    "event": "eggroll_iteration_complete",
                    "trainer": "eggroll",
                    "iteration": iteration,
                    "checkpoint_update": checkpoint_update,
                    "task": cfg.task,
                    "population_size": int(cfg.population_size),
                    "population_batch_size": int(cfg.population_batch_size),
                    "rank": int(cfg.rank),
                    "sigma": float(cfg.sigma),
                    "alpha": float(cfg.alpha),
                    "baseline_type": cfg.baseline_type,
                    "fitness_shaping": cfg.fitness_shaping,
                    "fitness_raw": [float(x) for x in fitness],
                    "fitness_shaped": [float(x) for x in shaped],
                    **stats,
                    **update_stats,
                    "checkpoint_path": str(numbered_path if should_number else latest_path),
                    "latest_checkpoint_path": str(latest_path),
                    "selected_action_rollout_video": video_path,
                    "oracle_baseline_video": str(oracle_manifest["video_path"]) if oracle_manifest else "",
                }
                write_jsonl_row(self.progress_path, progress_row)
                last_summary = progress_row

            if cfg.write_oracle_video and last_summary.get("selected_action_rollout_video"):
                smoke_manifest = {
                    "rollout_validation_video": last_summary["selected_action_rollout_video"],
                    "selected_action_rollout_video": last_summary["selected_action_rollout_video"],
                    "oracle_baseline_video": str(oracle_manifest["video_path"]) if oracle_manifest else "",
                    "oracle_baseline_video_status": "ok",
                    "success_any": bool(last_summary.get("success_rate", 0.0) > 0.0),
                    "success_last": bool(last_summary.get("success_rate", 0.0) > 0.0),
                }
                assert_smoke_manifest_contract(smoke_manifest)
                write_manifest(self.output_dir / "smoke_manifest.json", smoke_manifest)
            return last_summary
        finally:
            patch_handle.remove()
