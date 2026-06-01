#!/usr/bin/env python3
"""Phase11: env-on-policy GRPO for SmolVLA on MetaWorld Push-v3."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


def _append_progress(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row) + "\n")


# region agent log
def _agent_debug_log(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        path = Path("/vol/bitbucket/aa6622/.logs/debug-4facd7.log")
        path.parent.mkdir(parents=True, exist_ok=True)
        now_ms = int(time.time() * 1000)
        payload = {
            "sessionId": "4facd7",
            "id": f"log_{now_ms}_{hypothesis_id}",
            "timestamp": now_ms,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
        }
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload) + "\n")
    except Exception:
        pass
# endregion


def compute_live_logprob_parity(
    *,
    train_wrapper,
    rollouts,
    chunk_size: int,
    device,
    tolerance: float,
):
    """Recompute stored actions through the live update path before optimizer.step."""
    import torch

    from smolvla_grpo.grpo_math import summarize_logprob_ratio_parity

    parity_old: list[torch.Tensor] = []
    parity_new: list[torch.Tensor] = []
    with torch.no_grad():
        for traj in rollouts:
            t_len = len(traj.proc_snapshots)
            for cs in range(0, t_len, int(chunk_size)):
                ce = min(cs + int(chunk_size), t_len)
                procs = traj.proc_snapshots[cs:ce]
                scored_chunk = torch.stack([traj.logprob_actions[t] for t in range(cs, ce)]).to(device)
                old_lp = torch.stack([traj.log_probs[t] for t in range(cs, ce)]).to(device).reshape(-1)
                if train_wrapper.logprob_mode == "flow_sde":
                    live_lp, _mu, _log_std = train_wrapper.get_flow_sde_log_probs_from_proc_list(
                        procs,
                        traj.flow_sde_traces[cs:ce],
                    )
                    live_lp = live_lp.reshape(-1)
                else:
                    live_lp = train_wrapper.get_action_probs_from_proc_list(procs, scored_chunk).reshape(-1)
                parity_old.append(old_lp.detach().cpu())
                parity_new.append(live_lp.detach().cpu())
    return summarize_logprob_ratio_parity(
        torch.cat(parity_old) if parity_old else torch.zeros(0),
        torch.cat(parity_new) if parity_new else torch.zeros(0),
        tolerance=float(tolerance),
    )


def compute_live_chunk_logprob_parity(
    *,
    train_wrapper,
    rollouts,
    chunk_len: int,
    tolerance: float,
):
    """Recompute stored chunk traces through the live update path before optimizer.step."""
    import torch

    from smolvla_grpo.chunk_math import masked_chunk_sum
    from smolvla_grpo.grpo_math import summarize_logprob_ratio_parity

    parity_old: list[torch.Tensor] = []
    parity_new: list[torch.Tensor] = []
    per_action_abs: list[torch.Tensor] = []
    with torch.no_grad():
        for traj in rollouts:
            procs = [chunk.proc_snapshot for chunk in traj.chunks]
            traces = [chunk.flow_sde_trace for chunk in traj.chunks]
            if not procs:
                continue
            live_steps, _mu, _log_std = train_wrapper.get_flow_sde_log_probs_for_chunk_from_proc_list(
                procs,
                traces,
                chunk_len=int(chunk_len),
            )
            live_steps = live_steps.detach().cpu()
            for idx, chunk in enumerate(traj.chunks):
                valid = chunk.valid_action_mask.reshape(1, -1)
                old_step = chunk.log_probs.reshape(1, -1)
                new_step = live_steps[idx : idx + 1]
                parity_old.append(masked_chunk_sum(old_step, valid).reshape(1))
                parity_new.append(masked_chunk_sum(new_step, valid).reshape(1))
                per_action_abs.append(((new_step - old_step).abs() * valid.float()).reshape(-1))
    stats = summarize_logprob_ratio_parity(
        torch.cat(parity_old) if parity_old else torch.zeros(0),
        torch.cat(parity_new) if parity_new else torch.zeros(0),
        tolerance=float(tolerance),
    )
    payload = stats.as_dict()
    payload["max_abs_per_action_logprob"] = (
        float(torch.cat(per_action_abs).max().item()) if per_action_abs else 0.0
    )
    return stats, payload


def main() -> int:
    import torch
    from torch import nn

    from smolvla_grpo.checkpointing import load_grpo_checkpoint, save_grpo_checkpoint
    from smolvla_grpo.grpo_math import (
        apply_grpo_regularizers,
        compute_group_advantages,
        update_metrics,
    )
    from smolvla_grpo.chunk_math import masked_chunk_sum
    from smolvla_grpo.phase11_chunk_rollout import collect_chunk_rollout_group
    from smolvla_grpo.phase11_rollout import collect_rollout_group, load_bundle_for_grpo
    from smolvla_grpo.policy_wrapper import (
        MetaWorldSmolVLAGRPOPolicy,
        freeze_all_but_grpo_trainables,
    )
    from smolvla_grpo.reward_backends import EnvRewardBackend
    from smolvla_pipeline.evaluator import (
        _resolve_camera_name,
        _resolve_flip_corner2,
        _resolve_task_text,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="SmolVLA HF checkpoint dir or id")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task", type=str, default="push-v3")
    p.add_argument("--env-backend", choices=("custom", "official_lerobot"), default="custom")
    p.add_argument(
        "--rollout-execution",
        choices=("serial", "vector_sync", "vector_async"),
        default="serial",
        help="official_lerobot only: serial loops vs batched SyncVectorEnv vs batched async (forkserver)",
    )
    p.add_argument(
        "--async-start-method",
        type=str,
        default="forkserver",
        choices=("forkserver", "spawn"),
        help="multiprocessing context for vector_async (override with SMOLVLA_GRPO_ASYNC_MP_CONTEXT)",
    )
    p.add_argument("--train-seed-base", type=int, default=2000)
    p.add_argument("--start-update", type=int, default=0)
    p.add_argument("--num-updates", type=int, default=1, help="Number of updates to run from start-update")
    p.add_argument("--resume", type=Path, default=None, help="Path to .pt from save_grpo_checkpoint")
    p.add_argument("--max-steps", type=int, default=120, help="Use 0 with official_lerobot to use LeRobot env horizon")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1, help="Must be 1 for single-seed group")
    p.add_argument("--update-epochs", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=5)
    p.add_argument("--rollout-unit", choices=("step", "chunk"), default="step")
    p.add_argument("--rollout-chunk-len", type=int, default=5)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--init-log-std", type=float, default=-2.0)
    p.add_argument(
        "--euler-step-noise-std",
        type=float,
        default=0.0,
        help="Must be 0 for Gaussian logprob parity unless --allow-euler-noise",
    )
    p.add_argument(
        "--action-transform",
        choices=("no_tanh", "tanh_norm_ablation"),
        default="no_tanh",
        help="GRPO sampling transform before LeRobot postprocessor; tanh path is explicit ablation only.",
    )
    p.add_argument("--run-label", type=str, default="no_tanh_main")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--parity-tolerance", type=float, default=0.02)
    p.add_argument(
        "--allow-euler-noise",
        action="store_true",
        help="Allow nonzero euler_step_noise_std (denoise noise not in Gaussian logprob)",
    )
    p.add_argument(
        "--fail-on-parity-violation",
        action="store_true",
        help="Exit non-zero if pre-update logprob ratio parity fails",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--min-log-std", type=float, default=-4.0)
    p.add_argument("--kl-beta", type=float, default=0.0)
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument(
        "--logprob-mode",
        choices=("gaussian", "flow_sde"),
        default="gaussian",
        help="flow_sde requires venv denoise hook (Phase B)",
    )
    p.add_argument("--flow-sde-noise-level", type=float, default=0.5)
    p.add_argument("--flow-sde-trace-step", type=int, default=0)
    p.add_argument(
        "--gaussian-logprob-action",
        choices=("executed", "unsquashed"),
        default="executed",
        help="Action tensor scored by Gaussian logprob; unsquashed recreates the pre-A.3 G8 ablation.",
    )
    args = p.parse_args()
    if args.batch_size != 1:
        raise SystemExit("Only batch_size=1 supported (one seed context per update).")
    if args.logprob_mode == "flow_sde" and args.rollout_unit != "chunk":
        raise SystemExit("flow_sde requires --rollout-unit chunk")
    if args.rollout_unit == "chunk" and args.logprob_mode != "flow_sde":
        raise SystemExit("chunk rollout currently requires --logprob-mode flow_sde")
    if args.logprob_mode == "flow_sde" and args.action_transform != "no_tanh":
        raise SystemExit("flow_sde requires --action-transform no_tanh")
    if args.rollout_unit == "chunk" and args.env_backend != "official_lerobot":
        raise SystemExit("chunk rollout requires --env-backend official_lerobot")
    if args.rollout_unit == "chunk" and args.rollout_execution != "serial":
        raise SystemExit("first chunk rollout implementation requires --rollout-execution serial")
    if int(args.rollout_chunk_len) < 1:
        raise SystemExit("--rollout-chunk-len must be >= 1")
    if float(args.euler_step_noise_std) > 0.0 and not args.allow_euler_noise:
        raise SystemExit(
            "euler_step_noise_std must be 0 for corrected Gaussian GRPO "
            "(add --allow-euler-noise only for ablations)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    roll_dir = out / "rollouts"
    roll_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    manifest_path = out / "train_manifest.json"

    bundle, action_dim = load_bundle_for_grpo(
        args.checkpoint,
        task=args.task,
        env_backend=args.env_backend,
        n_action_steps=(int(args.rollout_chunk_len) if args.rollout_unit == "chunk" else 1),
    )
    task_text = _resolve_task_text(args.task, override=None)
    camera_name = _resolve_camera_name()
    flip_corner2 = _resolve_flip_corner2()

    train_wrapper = MetaWorldSmolVLAGRPOPolicy(
        bundle,
        task=args.task,
        task_text=task_text,
        camera_name=camera_name,
        flip_corner2=flip_corner2,
        action_dim=action_dim,
        action_transform=args.action_transform,
        min_log_std=float(args.min_log_std),
        gaussian_logprob_action=args.gaussian_logprob_action,
        logprob_mode=args.logprob_mode,
        flow_sde_noise_level=float(args.flow_sde_noise_level),
        flow_sde_trace_step=int(args.flow_sde_trace_step),
    )
    train_wrapper.assert_grpo_api()
    train_wrapper.set_log_std(args.init_log_std)
    train_wrapper.set_euler_step_noise_std(args.euler_step_noise_std)
    freeze_all_but_grpo_trainables(bundle.policy)
    trainable = [p for p in bundle.policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95))

    start_u = int(args.start_update)
    if args.resume is not None:
        ck = load_grpo_checkpoint(args.resume, map_location=str(device))
        bundle.policy.load_state_dict(ck["policy_state_dict"], strict=False)
        if ck.get("optimizer_state_dict"):
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_u = int(ck.get("update_index", start_u - 1)) + 1

    old_policy = copy.deepcopy(bundle.policy).eval().to(device)
    bundle.policy.train()

    reward_backend = EnvRewardBackend()
    end_u = start_u + int(args.num_updates)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "env_backend": args.env_backend,
        "rollout_execution": args.rollout_execution,
        "rollout_unit": args.rollout_unit,
        "rollout_chunk_len": int(args.rollout_chunk_len),
        "policy_n_action_steps": int(getattr(getattr(bundle.policy, "config", None), "n_action_steps", -1)),
        "async_start_method": (
            args.async_start_method if args.rollout_execution == "vector_async" else None
        ),
        "requested_max_steps": args.max_steps,
        "train_seed_base": args.train_seed_base,
        "start_update": start_u,
        "end_update": end_u,
        "max_steps": args.max_steps,
        "group_size": args.group_size,
        "clip_eps": args.clip_eps,
        "lr": args.lr,
        "action_transform": args.action_transform,
        "gaussian_logprob_action": args.gaussian_logprob_action,
        "logprob_mode": args.logprob_mode,
        "flow_sde_noise_level": float(args.flow_sde_noise_level),
        "flow_sde_trace_step": int(args.flow_sde_trace_step),
        "run_label": args.run_label,
        "euler_step_noise_std": float(args.euler_step_noise_std),
        "parity_tolerance": float(args.parity_tolerance),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # region agent log
    _agent_debug_log(
        run_id="pre-fix",
        hypothesis_id="H1",
        location="train_phase11_env_on_policy_grpo.py:manifest",
        message="train_config_snapshot",
        data={
            "run_label": args.run_label,
            "num_updates": int(args.num_updates),
            "start_update": int(start_u),
            "group_size": int(args.group_size),
            "max_steps": int(args.max_steps),
            "lr": float(args.lr),
            "clip_eps": float(args.clip_eps),
            "init_log_std": float(args.init_log_std),
            "min_log_std": float(args.min_log_std),
            "action_transform": args.action_transform,
            "gaussian_logprob_action": args.gaussian_logprob_action,
            "logprob_mode": args.logprob_mode,
            "euler_step_noise_std": float(args.euler_step_noise_std),
            "train_seed_base": int(args.train_seed_base),
            "save_every": int(args.save_every),
            "optimizer_chunk_size": int(args.chunk_size),
            "policy_n_action_steps": int(getattr(getattr(bundle.policy, "config", None), "n_action_steps", -1)),
            "trainable_param_count": int(sum(p.numel() for p in trainable)),
        },
    )
    # endregion

    for update in range(start_u, end_u):
        update_t0 = time.perf_counter()
        reset_seed = int(args.train_seed_base) + int(update)
        rollout_t0 = time.perf_counter()
        if args.rollout_unit == "chunk":
            rollouts = collect_chunk_rollout_group(
                bundle=bundle,
                policy_old=old_policy,
                task=args.task,
                task_text=task_text,
                reset_seed=reset_seed,
                episode_index=update,
                max_steps=args.max_steps,
                group_size=args.group_size,
                action_dim=action_dim,
                device=device,
                chunk_len=int(args.rollout_chunk_len),
                action_transform=args.action_transform,
                gaussian_logprob_action=args.gaussian_logprob_action,
                logprob_mode=args.logprob_mode,
                flow_sde_noise_level=float(args.flow_sde_noise_level),
                flow_sde_trace_step=int(args.flow_sde_trace_step),
            )
        else:
            rollouts = collect_rollout_group(
                bundle=bundle,
                policy_old=old_policy,
                task=args.task,
                task_text=task_text,
                reset_seed=reset_seed,
                episode_index=update,
                max_steps=args.max_steps,
                group_size=args.group_size,
                action_dim=action_dim,
                device=device,
                env_backend=args.env_backend,
                rollout_execution=args.rollout_execution,
                async_start_method=args.async_start_method,
                action_transform=args.action_transform,
                gaussian_logprob_action=args.gaussian_logprob_action,
                logprob_mode=args.logprob_mode,
                flow_sde_noise_level=float(args.flow_sde_noise_level),
                flow_sde_trace_step=int(args.flow_sde_trace_step),
            )
        rollout_seconds = float(time.perf_counter() - rollout_t0)
        if args.rollout_unit == "chunk":
            returns = torch.tensor([tr.total_return() for tr in rollouts], dtype=torch.float32, device=device)
            successes = [any(bool(s) for s in tr.successes) for tr in rollouts]
            episode_lengths = [len(tr.rewards) for tr in rollouts]
            num_env_steps = int(sum(episode_lengths))
            terminated_flags = [bool(tr.terminated) for tr in rollouts]
            truncated_flags = [bool(tr.truncated) for tr in rollouts]
            clip_values = [
                float(v)
                for tr in rollouts
                for chunk in tr.chunks
                for v in chunk.action_clip_fraction.reshape(-1).tolist()
            ]
            clip_any_values = [
                bool(v)
                for tr in rollouts
                for chunk in tr.chunks
                for v in chunk.action_clip_any.reshape(-1).tolist()
            ]
            oob_values = [
                float(v)
                for tr in rollouts
                for chunk in tr.chunks
                for v in chunk.postprocessor_oob_mean.reshape(-1).tolist()
            ]
            rollout_old_lp = (
                torch.cat([chunk.log_prob_sum.reshape(1) for tr in rollouts for chunk in tr.chunks]).to(device)
                if rollouts
                else torch.zeros(0, device=device)
            )
            rollout_log_std = (
                torch.cat([chunk.distr_log_std.reshape(-1, action_dim) for tr in rollouts for chunk in tr.chunks]).to(
                    device
                )
                if rollouts
                else torch.zeros(0, action_dim, device=device)
            )
            proc_snapshot_counts = [int(len(tr.chunks)) for tr in rollouts]
            policy_samples_one_per_env_step = False
            chunk_count = int(sum(len(tr.chunks) for tr in rollouts))
            valid_chunk_count = int(
                sum(1 for tr in rollouts for chunk in tr.chunks if bool(chunk.valid_action_mask.any()))
            )
        else:
            returns = torch.tensor(
                [reward_backend.episode_return(tr) for tr in rollouts],
                dtype=torch.float32,
                device=device,
            )
            successes = [any(bool(s) for s in tr.successes) for tr in rollouts]
            episode_lengths = [len(tr.rewards) for tr in rollouts]
            num_env_steps = int(sum(episode_lengths))
            terminated_flags = [bool(tr.terminated) for tr in rollouts]
            truncated_flags = [bool(tr.truncated) for tr in rollouts]
            clip_values = [float(v) for tr in rollouts for v in tr.action_clip_fractions]
            clip_any_values = [bool(v) for tr in rollouts for v in tr.action_clip_any]
            oob_values = [float(v) for tr in rollouts for v in tr.postprocessor_oob_means]
            rollout_old_lp = (
                torch.cat([torch.stack(tr.log_probs).reshape(-1) for tr in rollouts]).to(device)
                if rollouts
                else torch.zeros(0, device=device)
            )
            rollout_log_std = (
                torch.cat([torch.stack(tr.distr_log_stds).reshape(-1, action_dim) for tr in rollouts]).to(device)
                if rollouts
                else torch.zeros(0, action_dim, device=device)
            )
            proc_snapshot_counts = [int(len(tr.proc_snapshots)) for tr in rollouts]
            policy_samples_one_per_env_step = bool(
                all(len(tr.proc_snapshots) == len(tr.rewards) for tr in rollouts)
            )
            chunk_count = 0
            valid_chunk_count = 0
        success_rate = float(sum(1 for s in successes if s) / max(len(successes), 1))
        avg_ret = float(returns.mean().item())
        action_clip_fraction = float(sum(clip_values) / max(len(clip_values), 1))
        action_clip_any_fraction = float(
            sum(1 for v in clip_any_values if v) / max(len(clip_any_values), 1)
        )
        postprocessor_oob_mean = float(sum(oob_values) / max(len(oob_values), 1)) if oob_values else 0.0
        resolved_max_steps = int(args.max_steps)
        if rollouts:
            resolved_max_steps = int(
                rollouts[0].metadata.get("resolved_max_steps", resolved_max_steps)
            )
        advantages = compute_group_advantages(returns)
        pre_update_metrics = update_metrics(
            new_log_probs=rollout_old_lp,
            old_log_probs=rollout_old_lp,
            log_std=rollout_log_std,
            returns=returns,
            advantages=advantages,
            epsilon=float(args.clip_eps),
        )
        returns_cpu = returns.detach().cpu()
        advantages_cpu = advantages.detach().cpu()
        best_idx = int(torch.argmax(returns_cpu).item()) if returns_cpu.numel() else -1
        success_advantages = [
            float(advantages_cpu[i].item()) for i, ok in enumerate(successes) if bool(ok)
        ]
        # region agent log
        _agent_debug_log(
            run_id="pre-fix",
            hypothesis_id="H2,H4,H5",
            location="train_phase11_env_on_policy_grpo.py:pre_update",
            message="rollout_advantage_snapshot",
            data={
                "update": int(update),
                "reset_seed": int(reset_seed),
                "avg_return": float(avg_ret),
                "success_rate": float(success_rate),
                "returns": [float(x) for x in returns_cpu.tolist()],
                "successes": [bool(x) for x in successes],
                "advantages": [float(x) for x in advantages_cpu.tolist()],
                "best_return_index": best_idx,
                "best_return": float(returns_cpu[best_idx].item()) if best_idx >= 0 else None,
                "best_advantage": float(advantages_cpu[best_idx].item()) if best_idx >= 0 else None,
                "success_advantages": success_advantages,
                "group_return_std": float(pre_update_metrics["group_return_std"]),
                "zero_adv_skip": bool(pre_update_metrics["zero_adv_skip"]),
                "episode_lengths": [int(x) for x in episode_lengths],
                "proc_snapshot_counts": proc_snapshot_counts,
                "optimizer_chunk_size": int(args.chunk_size),
                "policy_samples_one_per_env_step": policy_samples_one_per_env_step,
                "rollout_unit": args.rollout_unit,
                "rollout_chunk_len": int(args.rollout_chunk_len),
                "chunk_count": int(chunk_count),
                "valid_chunk_count": int(valid_chunk_count),
                "terminated_count": int(sum(1 for x in terminated_flags if x)),
                "truncated_count": int(sum(1 for x in truncated_flags if x)),
                "action_clip_fraction": float(action_clip_fraction),
                "action_clip_any_fraction": float(action_clip_any_fraction),
                "log_std_mean": float(pre_update_metrics["log_std_mean"]),
                "log_std_min": float(pre_update_metrics["log_std_min"]),
                "log_std_max": float(pre_update_metrics["log_std_max"]),
            },
        )
        # endregion
        if torch.allclose(advantages, torch.zeros_like(advantages)):
            update_seconds = float(time.perf_counter() - update_t0)
            _append_progress(
                progress_path,
                {
                    "update": update,
                    "skipped": True,
                    "reason": "zero_advantages",
                    "reset_seed": reset_seed,
                    "env_backend": args.env_backend,
                    "rollout_execution": args.rollout_execution,
                    "rollout_unit": args.rollout_unit,
                    "rollout_chunk_len": int(args.rollout_chunk_len),
                    "action_transform": args.action_transform,
                    "gaussian_logprob_action": args.gaussian_logprob_action,
                    "run_label": args.run_label,
                    "async_start_method": (
                        args.async_start_method if args.rollout_execution == "vector_async" else None
                    ),
                    "max_steps": args.max_steps,
                    "resolved_max_steps": resolved_max_steps,
                    "avg_return": avg_ret,
                    "returns": returns.detach().cpu().tolist(),
                    "successes": successes,
                    "success_rate": success_rate,
                    "episode_lengths": episode_lengths,
                    "num_env_steps": num_env_steps,
                    "chunk_count": int(chunk_count),
                    "valid_chunk_count": int(valid_chunk_count),
                    "action_clip_fraction": action_clip_fraction,
                    "action_clip_any_fraction": action_clip_any_fraction,
                    "rollout_seconds": rollout_seconds,
                    "optimize_seconds": 0.0,
                    "update_seconds": update_seconds,
                    "terminated": terminated_flags,
                    "truncated": truncated_flags,
                    **pre_update_metrics,
                },
            )
            state = {k: v.clone() for k, v in bundle.policy.state_dict().items()}
            old_policy.load_state_dict(state)
            old_policy.eval()
            skipped_extra = {
                "avg_return": avg_ret,
                "success_rate": success_rate,
                "successes": successes,
                "skipped": True,
                "resolved_max_steps": resolved_max_steps,
                "episode_lengths": episode_lengths,
                "num_env_steps": num_env_steps,
                "action_clip_fraction": action_clip_fraction,
                "action_clip_any_fraction": action_clip_any_fraction,
                "rollout_seconds": rollout_seconds,
                "optimize_seconds": 0.0,
                "update_seconds": update_seconds,
                "terminated": terminated_flags,
                "truncated": truncated_flags,
                **pre_update_metrics,
            }
            save_grpo_checkpoint(
                ckpt_dir / "latest.pt",
                policy_state=bundle.policy.state_dict(),
                optimizer_state=optimizer.state_dict(),
                update_index=update,
                args=vars(args),
                extra=skipped_extra,
            )
            if (update + 1) % args.save_every == 0 or update == end_u - 1:
                save_grpo_checkpoint(
                    ckpt_dir / f"update_{update + 1:04d}.pt",
                    policy_state=bundle.policy.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    update_index=update,
                    args=vars(args),
                    extra=skipped_extra,
                )
            print(
                "phase111_grpo_update",
                f"update={update}",
                f"mode={args.rollout_execution}",
                f"label={args.run_label}",
                f"action_transform={args.action_transform}",
                f"gaussian_logprob_action={args.gaussian_logprob_action}",
                f"seed={reset_seed}",
                f"avg_return={avg_ret:.6g}",
                f"success_rate={success_rate:.3f}",
                f"episode_lengths={episode_lengths}",
                f"env_steps={num_env_steps}",
                f"clip_frac={action_clip_fraction:.4f}",
                f"clip_any_frac={action_clip_any_fraction:.4f}",
                f"rollout_s={rollout_seconds:.2f}",
                "opt_s=0.00",
                f"update_s={update_seconds:.2f}",
                "skipped=zero_advantages",
                flush=True,
            )
            continue

        if args.rollout_unit == "chunk":
            bundle.policy.eval()
            parity_stats, parity_payload = compute_live_chunk_logprob_parity(
                train_wrapper=train_wrapper,
                rollouts=rollouts,
                chunk_len=int(args.rollout_chunk_len),
                tolerance=float(args.parity_tolerance),
            )
            if not parity_stats.within_tolerance:
                msg = (
                    f"GRPO chunk logprob parity failed update={update}: "
                    f"mean_ratio={parity_stats.mean_ratio:.6f} "
                    f"max_abs_log_ratio={parity_stats.max_abs_log_ratio:.6f} "
                    f"max_abs_per_action_logprob={parity_payload['max_abs_per_action_logprob']:.6f}"
                )
                print(msg, flush=True)
                if args.fail_on_parity_violation:
                    raise RuntimeError(msg)

            if int(valid_chunk_count) <= 0:
                raise RuntimeError("chunk rollout produced zero valid chunks")

            bundle.policy.train()
            optimize_t0 = time.perf_counter()
            last_new_log_probs: list[torch.Tensor] = []
            last_old_log_probs: list[torch.Tensor] = []
            last_log_stds: list[torch.Tensor] = []
            total_grad_norm = torch.zeros((), device=device)
            for _epoch in range(args.update_epochs):
                optimizer.zero_grad()
                epoch_new_log_probs: list[torch.Tensor] = []
                epoch_old_log_probs: list[torch.Tensor] = []
                epoch_log_stds: list[torch.Tensor] = []
                for gi, traj in enumerate(rollouts):
                    A = advantages[gi].reshape(()).float()
                    procs = [chunk.proc_snapshot for chunk in traj.chunks]
                    traces = [chunk.flow_sde_trace for chunk in traj.chunks]
                    if not procs:
                        continue
                    new_steps, _mu_live, log_std_live = (
                        train_wrapper.get_flow_sde_log_probs_for_chunk_from_proc_list(
                            procs,
                            traces,
                            chunk_len=int(args.rollout_chunk_len),
                        )
                    )
                    for ci, chunk in enumerate(traj.chunks):
                        valid = chunk.valid_action_mask.reshape(1, -1).to(device)
                        if not bool(valid.any()):
                            continue
                        old_steps = chunk.log_probs.reshape(1, -1).to(device)
                        new_step = new_steps[ci : ci + 1]
                        old_lp = masked_chunk_sum(old_steps, valid)
                        new_lp = masked_chunk_sum(new_step, valid)
                        ratio = torch.exp((new_lp - old_lp).clamp(-20.0, 20.0))
                        unclipped = ratio * A
                        clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * A
                        chunk_loss = -torch.min(unclipped, clipped).sum() / max(int(valid_chunk_count), 1)
                        chunk_loss = apply_grpo_regularizers(
                            chunk_loss,
                            current_log_probs=new_lp,
                            reference_log_probs=old_lp,
                            log_std=log_std_live[ci : ci + 1].reshape(-1, action_dim),
                            kl_beta=float(args.kl_beta) / max(int(valid_chunk_count), 1),
                            entropy_coef=float(args.entropy_coef) / max(int(valid_chunk_count), 1),
                        )
                        chunk_loss.backward()
                        epoch_new_log_probs.append(new_lp.detach().cpu())
                        epoch_old_log_probs.append(old_lp.detach().cpu())
                        epoch_log_stds.append(log_std_live[ci].detach().cpu().reshape(-1, action_dim))

                total_grad_norm = nn.utils.clip_grad_norm_(bundle.policy.parameters(), args.grad_clip)
                optimizer.step()
                last_new_log_probs = epoch_new_log_probs
                last_old_log_probs = epoch_old_log_probs
                last_log_stds = epoch_log_stds

            optimize_seconds = float(time.perf_counter() - optimize_t0)
            update_seconds = float(time.perf_counter() - update_t0)
            post_update_metrics = update_metrics(
                new_log_probs=torch.cat(last_new_log_probs) if last_new_log_probs else rollout_old_lp.detach().cpu(),
                old_log_probs=torch.cat(last_old_log_probs) if last_old_log_probs else rollout_old_lp.detach().cpu(),
                log_std=torch.cat(last_log_stds) if last_log_stds else rollout_log_std.detach().cpu(),
                returns=returns.detach().cpu(),
                advantages=advantages.detach().cpu(),
                epsilon=float(args.clip_eps),
            )
            _append_progress(
                progress_path,
                {
                    "update": update,
                    "reset_seed": reset_seed,
                    "env_backend": args.env_backend,
                    "rollout_execution": args.rollout_execution,
                    "rollout_unit": args.rollout_unit,
                    "rollout_chunk_len": int(args.rollout_chunk_len),
                    "action_transform": args.action_transform,
                    "gaussian_logprob_action": args.gaussian_logprob_action,
                    "run_label": args.run_label,
                    "async_start_method": None,
                    "max_steps": args.max_steps,
                    "resolved_max_steps": resolved_max_steps,
                    "avg_return": avg_ret,
                    "returns": returns.detach().cpu().tolist(),
                    "successes": successes,
                    "success_rate": success_rate,
                    "advantages": advantages.detach().cpu().tolist(),
                    "episode_lengths": episode_lengths,
                    "num_env_steps": num_env_steps,
                    "chunk_count": int(chunk_count),
                    "valid_chunk_count": int(valid_chunk_count),
                    "action_clip_fraction": action_clip_fraction,
                    "action_clip_any_fraction": action_clip_any_fraction,
                    "postprocessor_oob_mean": postprocessor_oob_mean,
                    "parity": parity_payload,
                    "kl_beta": float(args.kl_beta),
                    "entropy_coef": float(args.entropy_coef),
                    "rollout_seconds": rollout_seconds,
                    "optimize_seconds": optimize_seconds,
                    "update_seconds": update_seconds,
                    "terminated": terminated_flags,
                    "truncated": truncated_flags,
                    "grad_norm_before_clip": float(total_grad_norm.detach().cpu().item())
                    if torch.is_tensor(total_grad_norm)
                    else float(total_grad_norm),
                    **post_update_metrics,
                },
            )
            state = {k: v.clone() for k, v in bundle.policy.state_dict().items()}
            old_policy.load_state_dict(state)
            old_policy.eval()
            extra = {
                "avg_return": avg_ret,
                "success_rate": success_rate,
                "successes": successes,
                "resolved_max_steps": resolved_max_steps,
                "episode_lengths": episode_lengths,
                "num_env_steps": num_env_steps,
                "chunk_count": int(chunk_count),
                "valid_chunk_count": int(valid_chunk_count),
                "action_clip_fraction": action_clip_fraction,
                "action_clip_any_fraction": action_clip_any_fraction,
                "terminated": terminated_flags,
                "truncated": truncated_flags,
            }
            save_grpo_checkpoint(
                ckpt_dir / "latest.pt",
                policy_state=bundle.policy.state_dict(),
                optimizer_state=optimizer.state_dict(),
                update_index=update,
                args=vars(args),
                extra=extra,
            )
            if (update + 1) % args.save_every == 0 or update == end_u - 1:
                save_grpo_checkpoint(
                    ckpt_dir / f"update_{update + 1:04d}.pt",
                    policy_state=bundle.policy.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    update_index=update,
                    args=vars(args),
                    extra=extra,
                )
            print(
                "phase111_grpo_update",
                f"update={update}",
                f"mode={args.rollout_execution}",
                f"unit={args.rollout_unit}",
                f"label={args.run_label}",
                f"seed={reset_seed}",
                f"avg_return={avg_ret:.6g}",
                f"success_rate={success_rate:.3f}",
                f"episode_lengths={episode_lengths}",
                f"env_steps={num_env_steps}",
                f"chunks={chunk_count}",
                f"valid_chunks={valid_chunk_count}",
                f"parity_mean_ratio={parity_stats.mean_ratio:.4f}",
                f"rollout_s={rollout_seconds:.2f}",
                f"opt_s={optimize_seconds:.2f}",
                f"update_s={update_seconds:.2f}",
                flush=True,
            )
            continue

        bundle.policy.eval()
        parity_stats = compute_live_logprob_parity(
            train_wrapper=train_wrapper,
            rollouts=rollouts,
            chunk_size=int(args.chunk_size),
            device=device,
            tolerance=float(args.parity_tolerance),
        )
        if not parity_stats.within_tolerance:
            msg = (
                f"GRPO logprob parity failed update={update}: "
                f"mean_ratio={parity_stats.mean_ratio:.6f} "
                f"max_abs_log_ratio={parity_stats.max_abs_log_ratio:.6f}"
            )
            print(msg, flush=True)
            if args.fail_on_parity_violation:
                raise RuntimeError(msg)

        bundle.policy.train()
        optimize_t0 = time.perf_counter()
        last_new_log_probs: list[torch.Tensor] = []
        last_old_log_probs: list[torch.Tensor] = []
        last_log_stds: list[torch.Tensor] = []
        for _epoch in range(args.update_epochs):
            optimizer.zero_grad()
            epoch_new_log_probs: list[torch.Tensor] = []
            epoch_old_log_probs: list[torch.Tensor] = []
            epoch_log_stds: list[torch.Tensor] = []
            for gi, traj in enumerate(rollouts):
                A = advantages[gi].reshape(()).float()
                T = len(traj.proc_snapshots)
                G = len(rollouts)
                for cs in range(0, T, args.chunk_size):
                    ce = min(cs + args.chunk_size, T)
                    procs = traj.proc_snapshots[cs:ce]
                    scored_chunk = torch.stack([traj.logprob_actions[t] for t in range(cs, ce)]).to(
                        device
                    )
                    old_lp = torch.stack([traj.log_probs[t] for t in range(cs, ce)]).to(device).reshape(-1)
                    new_lp, _mean_live, log_std_live = (
                        train_wrapper.get_flow_sde_log_probs_from_proc_list(
                            procs,
                            traj.flow_sde_traces[cs:ce],
                        )
                        if args.logprob_mode == "flow_sde"
                        else train_wrapper.get_action_log_probs_and_params_from_proc_list(procs, scored_chunk)
                    )
                    new_lp = new_lp.reshape(-1)
                    ratio = torch.exp(new_lp - old_lp)
                    unclipped = ratio * A
                    clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * A
                    step_losses = -torch.min(unclipped, clipped)
                    chunk_loss = step_losses.sum() / max(T * G, 1)
                    regularizer_scale = float(ce - cs) / float(max(T * G, 1))
                    chunk_loss = apply_grpo_regularizers(
                        chunk_loss,
                        current_log_probs=new_lp,
                        reference_log_probs=old_lp,
                        log_std=log_std_live,
                        kl_beta=float(args.kl_beta) * regularizer_scale,
                        entropy_coef=float(args.entropy_coef) * regularizer_scale,
                    )
                    chunk_loss.backward()
                    epoch_new_log_probs.append(new_lp.detach().cpu())
                    epoch_old_log_probs.append(old_lp.detach().cpu())
                    epoch_log_stds.append(log_std_live.detach().cpu())

            total_grad_norm = nn.utils.clip_grad_norm_(bundle.policy.parameters(), args.grad_clip)
            optimizer.step()
            last_new_log_probs = epoch_new_log_probs
            last_old_log_probs = epoch_old_log_probs
            last_log_stds = epoch_log_stds
            debug_after_new: list[torch.Tensor] = []
            debug_after_old: list[torch.Tensor] = []
            bundle.policy.eval()
            with torch.no_grad():
                for traj in rollouts:
                    debug_t = min(len(traj.proc_snapshots), max(int(args.chunk_size), 10))
                    for cs in range(0, debug_t, args.chunk_size):
                        ce = min(cs + args.chunk_size, debug_t)
                        procs = traj.proc_snapshots[cs:ce]
                        scored_chunk = torch.stack([traj.logprob_actions[t] for t in range(cs, ce)]).to(
                            device
                        )
                        old_lp = torch.stack([traj.log_probs[t] for t in range(cs, ce)]).to(device).reshape(-1)
                        if args.logprob_mode == "flow_sde":
                            new_lp_after, _mean_after, _log_std_after = (
                                train_wrapper.get_flow_sde_log_probs_from_proc_list(
                                    procs,
                                    traj.flow_sde_traces[cs:ce],
                                )
                            )
                        else:
                            new_lp_after, _mean_after, _log_std_after = (
                                train_wrapper.get_action_log_probs_and_params_from_proc_list(
                                    procs,
                                    scored_chunk,
                                )
                            )
                        debug_after_old.append(old_lp.detach().cpu())
                        debug_after_new.append(new_lp_after.reshape(-1).detach().cpu())
            bundle.policy.train()
            if debug_after_new and debug_after_old:
                debug_old_cat = torch.cat(debug_after_old).float()
                debug_new_cat = torch.cat(debug_after_new).float()
                debug_ratio = torch.exp(debug_new_cat - debug_old_cat)
                debug_after_step = {
                    "sample_n": int(debug_ratio.numel()),
                    "sample_ratio_mean": float(debug_ratio.mean().item()),
                    "sample_ratio_min": float(debug_ratio.min().item()),
                    "sample_ratio_max": float(debug_ratio.max().item()),
                    "sample_max_abs_log_ratio": float((debug_new_cat - debug_old_cat).abs().max().item()),
                    "sample_ratio_clip_fraction": float(
                        (
                            (debug_ratio < 1.0 - float(args.clip_eps))
                            | (debug_ratio > 1.0 + float(args.clip_eps))
                        )
                        .float()
                        .mean()
                        .item()
                    ),
                }
            else:
                debug_after_step = {"sample_n": 0}
            debug_pre_step_metrics = update_metrics(
                new_log_probs=torch.cat(last_new_log_probs) if last_new_log_probs else rollout_old_lp.detach().cpu(),
                old_log_probs=torch.cat(last_old_log_probs) if last_old_log_probs else rollout_old_lp.detach().cpu(),
                log_std=torch.cat(last_log_stds) if last_log_stds else rollout_log_std.detach().cpu(),
                returns=returns.detach().cpu(),
                advantages=advantages.detach().cpu(),
                epsilon=float(args.clip_eps),
            )
            # region agent log
            _agent_debug_log(
                run_id="pre-fix",
                hypothesis_id="H3,H5",
                location="train_phase11_env_on_policy_grpo.py:post_optimizer",
                message="optimizer_effect_snapshot",
                data={
                    "update": int(update),
                    "grad_norm_before_clip": float(total_grad_norm.detach().cpu().item())
                    if torch.is_tensor(total_grad_norm)
                    else float(total_grad_norm),
                    "grad_clip": float(args.grad_clip),
                    "pre_step_ratio_mean": float(debug_pre_step_metrics["ratio_mean"]),
                    "pre_step_ratio_min": float(debug_pre_step_metrics["ratio_min"]),
                    "pre_step_ratio_max": float(debug_pre_step_metrics["ratio_max"]),
                    "pre_step_ratio_clip_fraction": float(debug_pre_step_metrics["ratio_clip_fraction"]),
                    "after_step_sample": debug_after_step,
                },
            )
            # endregion
        optimize_seconds = float(time.perf_counter() - optimize_t0)
        update_seconds = float(time.perf_counter() - update_t0)
        post_update_metrics = update_metrics(
            new_log_probs=torch.cat(last_new_log_probs) if last_new_log_probs else rollout_old_lp.detach().cpu(),
            old_log_probs=torch.cat(last_old_log_probs) if last_old_log_probs else rollout_old_lp.detach().cpu(),
            log_std=torch.cat(last_log_stds) if last_log_stds else rollout_log_std.detach().cpu(),
            returns=returns.detach().cpu(),
            advantages=advantages.detach().cpu(),
            epsilon=float(args.clip_eps),
        )

        _append_progress(
            progress_path,
            {
                "update": update,
                "reset_seed": reset_seed,
                "env_backend": args.env_backend,
                "rollout_execution": args.rollout_execution,
                "action_transform": args.action_transform,
                "gaussian_logprob_action": args.gaussian_logprob_action,
                "run_label": args.run_label,
                "async_start_method": (
                    args.async_start_method if args.rollout_execution == "vector_async" else None
                ),
                "max_steps": args.max_steps,
                "resolved_max_steps": resolved_max_steps,
                "avg_return": avg_ret,
                "returns": returns.detach().cpu().tolist(),
                "successes": successes,
                "success_rate": success_rate,
                "advantages": advantages.detach().cpu().tolist(),
                "episode_lengths": episode_lengths,
                "num_env_steps": num_env_steps,
                "action_clip_fraction": action_clip_fraction,
                "action_clip_any_fraction": action_clip_any_fraction,
                "postprocessor_oob_mean": postprocessor_oob_mean,
                "parity": parity_stats.as_dict(),
                "kl_beta": float(args.kl_beta),
                "entropy_coef": float(args.entropy_coef),
                "rollout_seconds": rollout_seconds,
                "optimize_seconds": optimize_seconds,
                "update_seconds": update_seconds,
                "terminated": terminated_flags,
                "truncated": truncated_flags,
                **post_update_metrics,
            },
        )
        state = {k: v.clone() for k, v in bundle.policy.state_dict().items()}
        old_policy.load_state_dict(state)
        old_policy.eval()

        save_grpo_checkpoint(
            ckpt_dir / "latest.pt",
            policy_state=bundle.policy.state_dict(),
            optimizer_state=optimizer.state_dict(),
            update_index=update,
            args=vars(args),
            extra={
                "avg_return": avg_ret,
                "success_rate": success_rate,
                "successes": successes,
                "resolved_max_steps": resolved_max_steps,
                "episode_lengths": episode_lengths,
                "num_env_steps": num_env_steps,
                "action_clip_fraction": action_clip_fraction,
                "action_clip_any_fraction": action_clip_any_fraction,
                "terminated": terminated_flags,
                "truncated": truncated_flags,
            },
        )
        if (update + 1) % args.save_every == 0 or update == end_u - 1:
            save_grpo_checkpoint(
                ckpt_dir / f"update_{update + 1:04d}.pt",
                policy_state=bundle.policy.state_dict(),
                optimizer_state=optimizer.state_dict(),
                update_index=update,
                args=vars(args),
                extra={
                    "avg_return": avg_ret,
                    "success_rate": success_rate,
                    "successes": successes,
                    "resolved_max_steps": resolved_max_steps,
                    "episode_lengths": episode_lengths,
                    "terminated": terminated_flags,
                    "truncated": truncated_flags,
                    "action_clip_fraction": action_clip_fraction,
                    "action_clip_any_fraction": action_clip_any_fraction,
                },
            )
        print(
            "phase111_grpo_update",
            f"update={update}",
            f"mode={args.rollout_execution}",
            f"label={args.run_label}",
            f"action_transform={args.action_transform}",
            f"gaussian_logprob_action={args.gaussian_logprob_action}",
            f"seed={reset_seed}",
            f"avg_return={avg_ret:.6g}",
            f"success_rate={success_rate:.3f}",
            f"episode_lengths={episode_lengths}",
            f"env_steps={num_env_steps}",
            f"clip_frac={action_clip_fraction:.4f}",
            f"clip_any_frac={action_clip_any_fraction:.4f}",
            f"oob_mean={postprocessor_oob_mean:.4f}",
            f"parity_mean_ratio={parity_stats.mean_ratio:.4f}",
            f"rollout_s={rollout_seconds:.2f}",
            f"opt_s={optimize_seconds:.2f}",
            f"update_s={update_seconds:.2f}",
            flush=True,
        )

    print(f"Done. Artifacts under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
