#!/usr/bin/env python3
"""Phase11: env-on-policy GRPO for SmolVLA on MetaWorld Push-v3."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


def _append_progress(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row) + "\n")


def _json_ready_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


# region agent log
def _debug_mem_log(*, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    if os.environ.get("AGENT_DEBUG_MEM") != "1":
        return
    try:
        pid = os.getpid()

        def _status_for(process_id: int) -> dict[str, int]:
            fields: dict[str, int] = {}
            try:
                with open(f"/proc/{process_id}/status", "r", encoding="utf-8") as fp:
                    for line in fp:
                        if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "PPid:")):
                            parts = line.split()
                            fields[parts[0].rstrip(":")] = int(parts[1])
            except OSError:
                pass
            return fields

        child_rss_kb = 0
        child_count = 0
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            status = _status_for(int(entry))
            if status.get("PPid") == pid:
                child_count += 1
                child_rss_kb += int(status.get("VmRSS", 0))
        cuda_data = {}
        if torch.cuda.is_available():
            cuda_data = {
                "cuda_allocated": int(torch.cuda.memory_allocated()),
                "cuda_reserved": int(torch.cuda.memory_reserved()),
                "cuda_max_allocated": int(torch.cuda.max_memory_allocated()),
                "cuda_max_reserved": int(torch.cuda.max_memory_reserved()),
            }
        payload = {
            "sessionId": "1b1269",
            "runId": os.environ.get("AGENT_DEBUG_RUN_ID", "phase11_mem"),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": {
                **data,
                "pid": pid,
                "self_status": _status_for(pid),
                "child_count": child_count,
                "child_rss_kb": child_rss_kb,
                **cuda_data,
            },
            "timestamp": int(time.time() * 1000),
        }
        with open("/rds/general/user/aa6622/home/.cursor/debug-1b1269.log", "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload) + "\n")
    except Exception:
        return
# endregion


def _phase11_clipped_row_loss(
    new_lp: torch.Tensor,
    old_lp: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float,
) -> torch.Tensor:
    ratio = torch.exp(new_lp - old_lp)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * advantage
    return -torch.min(unclipped, clipped)


def _new_loss_telemetry(*, mode: str, batch_size: int) -> dict:
    return {
        "logprob_recompute_mode": str(mode),
        "logprob_batch_size": int(batch_size),
        "num_logprob_forward_batches": 0,
        "old_logprob_chunks": [],
        "new_logprob_chunks": [],
    }


def _record_loss_telemetry(telemetry: dict | None, *, old_lp: torch.Tensor, new_lp: torch.Tensor, forwards: int) -> None:
    if telemetry is None:
        return
    telemetry["num_logprob_forward_batches"] = int(telemetry.get("num_logprob_forward_batches", 0)) + int(forwards)
    telemetry.setdefault("old_logprob_chunks", []).append(old_lp.detach().float().cpu().reshape(-1))
    telemetry.setdefault("new_logprob_chunks", []).append(new_lp.detach().float().cpu().reshape(-1))


def _finalize_loss_telemetry(telemetry: dict | None, *, clip_eps: float) -> dict:
    if telemetry is None:
        return {}
    old_chunks = list(telemetry.get("old_logprob_chunks", []) or [])
    new_chunks = list(telemetry.get("new_logprob_chunks", []) or [])
    base = {
        "logprob_recompute_mode": telemetry.get("logprob_recompute_mode"),
        "logprob_batch_size": int(telemetry.get("logprob_batch_size", 0)),
        "num_logprob_forward_batches": int(telemetry.get("num_logprob_forward_batches", 0)),
    }
    if not old_chunks or not new_chunks:
        return {
            **base,
            "ratio_mean": None,
            "ratio_min": None,
            "ratio_max": None,
            "ratio_clip_fraction": None,
            "approx_kl": None,
            "old_logprob_sum_mean": None,
            "new_logprob_sum_mean": None,
        }
    old_lp = torch.cat(old_chunks, dim=0)
    new_lp = torch.cat(new_chunks, dim=0)
    ratio = torch.exp(new_lp - old_lp)
    if not torch.isfinite(ratio).all():
        raise RuntimeError("non-finite PPO ratio during Phase11 telemetry")
    clipped = (ratio < (1.0 - float(clip_eps))) | (ratio > (1.0 + float(clip_eps)))
    return {
        **base,
        "ratio_mean": float(ratio.mean().item()),
        "ratio_min": float(ratio.min().item()),
        "ratio_max": float(ratio.max().item()),
        "ratio_clip_fraction": float(clipped.float().mean().item()),
        "approx_kl": float((old_lp - new_lp).mean().item()),
        "old_logprob_sum_mean": float(old_lp.mean().item()),
        "new_logprob_sum_mean": float(new_lp.mean().item()),
    }


def _log_std_telemetry(policy) -> dict:
    model = getattr(policy, "model", None)
    log_std = getattr(model, "log_std", None)
    if log_std is None:
        return {
            "log_std_mean": None,
            "log_std_min": None,
            "log_std_max": None,
            "entropy_proxy_mean": None,
        }
    vals = log_std.detach().float().reshape(-1)
    entropy_proxy = vals + 0.5 * (1.0 + math.log(2.0 * math.pi))
    return {
        "log_std_mean": float(vals.mean().item()),
        "log_std_min": float(vals.min().item()),
        "log_std_max": float(vals.max().item()),
        "entropy_proxy_mean": float(entropy_proxy.mean().item()),
    }


def _fmt_metric(value: object) -> str:
    if value is None:
        return "null"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _backward_phase11_chunk_loss(
    *,
    train_wrapper,
    action_chunks,
    advantage: torch.Tensor,
    device: torch.device,
    optimizer_chunk_size: int,
    clip_eps: float,
    normalizer: int,
    logprob_recompute_mode: str = "batched",
    logprob_batch_size: int | None = None,
    telemetry: dict | None = None,
) -> int:
    rows = list(action_chunks)
    if not rows:
        return 0
    denom = int(normalizer)
    if denom <= 0:
        raise ValueError("normalizer must be > 0")
    opt_chunk = max(int(optimizer_chunk_size), 1)
    lp_batch = max(int(logprob_batch_size or opt_chunk), 1)

    def _recompute_row_logprob(row) -> torch.Tensor:
        unsquashed = row.unsquashed_chunk.to(device)
        if getattr(row, "logprob_mode", "chunk") == "step":
            return train_wrapper.get_action_probs_from_proc_list(
                [row.proc_snapshot],
                unsquashed.reshape(1, -1).to(device),
            ).sum()
        return train_wrapper.get_action_probs_for_chunk_from_proc(
            row.proc_snapshot,
            unsquashed,
        ).sum()

    def _backward_batch(batch_rows, new_lp: torch.Tensor, *, forwards: int) -> None:
        new_lp = new_lp.float().reshape(-1)
        old_lp = torch.stack([row.log_prob_sum for row in batch_rows]).to(
            device=new_lp.device,
            dtype=torch.float32,
        ).reshape(-1)
        if new_lp.shape != old_lp.shape:
            raise RuntimeError(f"new/old logprob shape mismatch: {tuple(new_lp.shape)} != {tuple(old_lp.shape)}")
        if not torch.isfinite(new_lp).all() or not torch.isfinite(old_lp).all():
            raise RuntimeError("non-finite logprob during Phase11 loss")
        adv = advantage.to(device=new_lp.device, dtype=torch.float32)
        losses = _phase11_clipped_row_loss(new_lp, old_lp, adv, clip_eps)
        _record_loss_telemetry(telemetry, old_lp=old_lp, new_lp=new_lp, forwards=forwards)
        (losses.sum() / denom).backward()

    if str(logprob_recompute_mode) == "loop":
        for cs in range(0, len(rows), opt_chunk):
            batch_rows = rows[cs : cs + opt_chunk]
            new_lp = torch.stack([_recompute_row_logprob(row) for row in batch_rows])
            _backward_batch(batch_rows, new_lp, forwards=len(batch_rows))
        return len(rows)
    if str(logprob_recompute_mode) != "batched":
        raise ValueError("logprob_recompute_mode must be 'batched' or 'loop'")

    groups: dict[tuple[str, tuple[int, ...]], list] = {}
    for row in rows:
        mode = str(getattr(row, "logprob_mode", "chunk"))
        shape = tuple(row.unsquashed_chunk.reshape(-1, row.unsquashed_chunk.shape[-1]).shape)
        groups.setdefault((mode, shape), []).append(row)

    for (mode, _shape), group_rows in groups.items():
        for cs in range(0, len(group_rows), lp_batch):
            batch_rows = group_rows[cs : cs + lp_batch]
            if mode == "step":
                if not hasattr(train_wrapper, "get_action_probs_step_batch_from_proc_list"):
                    new_lp = torch.stack([_recompute_row_logprob(row) for row in batch_rows])
                    _backward_batch(batch_rows, new_lp, forwards=len(batch_rows))
                    continue
                proc_snapshots = [row.proc_snapshot for row in batch_rows]
                unsquashed = torch.stack(
                    [row.unsquashed_chunk.reshape(-1) for row in batch_rows],
                    dim=0,
                ).to(device)
                new_lp = train_wrapper.get_action_probs_step_batch_from_proc_list(
                    proc_snapshots,
                    unsquashed,
                )
                _backward_batch(batch_rows, new_lp, forwards=1)
                continue
            if not hasattr(train_wrapper, "get_action_probs_for_chunk_batch_from_proc_list"):
                new_lp = torch.stack([_recompute_row_logprob(row) for row in batch_rows])
                _backward_batch(batch_rows, new_lp, forwards=len(batch_rows))
                continue
            proc_snapshots = [row.proc_snapshot for row in batch_rows]
            unsquashed = torch.stack([row.unsquashed_chunk for row in batch_rows], dim=0).to(device)
            new_lp_steps = train_wrapper.get_action_probs_for_chunk_batch_from_proc_list(
                proc_snapshots,
                unsquashed,
            )
            new_lp = new_lp_steps.reshape(len(batch_rows), -1).sum(dim=1)
            _backward_batch(batch_rows, new_lp, forwards=1)
    return len(rows)


def _backward_phase11_group_loss(
    *,
    train_wrapper,
    rollouts,
    advantages: torch.Tensor,
    device: torch.device,
    optimizer_chunk_size: int,
    clip_eps: float,
    logprob_recompute_mode: str,
    logprob_batch_size: int,
    telemetry: dict | None,
) -> int:
    G = len(rollouts)
    entries: list[tuple[object, torch.Tensor, int]] = []
    for gi, traj in enumerate(rollouts):
        A = advantages[gi].reshape(()).float()
        n_units = len(getattr(traj, "action_chunks", []) or [])
        if n_units <= 0:
            raise RuntimeError("Phase11 rollout produced no action_chunks")
        normalizer = int(n_units * G)
        entries.extend((row, A, normalizer) for row in traj.action_chunks)
    if not entries:
        return 0

    opt_chunk = max(int(optimizer_chunk_size), 1)
    lp_batch = max(int(logprob_batch_size), 1)

    def _recompute_row_logprob(row) -> torch.Tensor:
        unsquashed = row.unsquashed_chunk.to(device)
        if getattr(row, "logprob_mode", "chunk") == "step":
            return train_wrapper.get_action_probs_from_proc_list(
                [row.proc_snapshot],
                unsquashed.reshape(1, -1).to(device),
            ).sum()
        return train_wrapper.get_action_probs_for_chunk_from_proc(
            row.proc_snapshot,
            unsquashed,
        ).sum()

    def _backward_entries(batch_entries, new_lp: torch.Tensor, *, forwards: int) -> None:
        rows = [entry[0] for entry in batch_entries]
        new_lp = new_lp.float().reshape(-1)
        old_lp = torch.stack([row.log_prob_sum for row in rows]).to(
            device=new_lp.device,
            dtype=torch.float32,
        ).reshape(-1)
        adv = torch.stack([entry[1].to(device=new_lp.device, dtype=torch.float32).reshape(()) for entry in batch_entries])
        normalizers = torch.tensor(
            [int(entry[2]) for entry in batch_entries],
            device=new_lp.device,
            dtype=torch.float32,
        )
        if new_lp.shape != old_lp.shape or new_lp.shape != adv.shape:
            raise RuntimeError(
                "new/old/adv logprob shape mismatch: "
                f"new={tuple(new_lp.shape)} old={tuple(old_lp.shape)} adv={tuple(adv.shape)}"
            )
        if not torch.isfinite(new_lp).all() or not torch.isfinite(old_lp).all():
            raise RuntimeError("non-finite logprob during Phase11 group loss")
        losses = _phase11_clipped_row_loss(new_lp, old_lp, adv, clip_eps)
        _record_loss_telemetry(telemetry, old_lp=old_lp, new_lp=new_lp, forwards=forwards)
        (losses / normalizers).sum().backward()

    if str(logprob_recompute_mode) == "loop":
        for cs in range(0, len(entries), opt_chunk):
            batch_entries = entries[cs : cs + opt_chunk]
            new_lp = torch.stack([_recompute_row_logprob(entry[0]) for entry in batch_entries])
            _backward_entries(batch_entries, new_lp, forwards=len(batch_entries))
        return len(entries)
    if str(logprob_recompute_mode) != "batched":
        raise ValueError("logprob_recompute_mode must be 'batched' or 'loop'")

    groups: dict[tuple[str, tuple[int, ...]], list[tuple[object, torch.Tensor, int]]] = {}
    for entry in entries:
        row = entry[0]
        mode = str(getattr(row, "logprob_mode", "chunk"))
        shape = tuple(row.unsquashed_chunk.reshape(-1, row.unsquashed_chunk.shape[-1]).shape)
        groups.setdefault((mode, shape), []).append(entry)

    for (mode, _shape), group_entries in groups.items():
        for cs in range(0, len(group_entries), lp_batch):
            batch_entries = group_entries[cs : cs + lp_batch]
            rows = [entry[0] for entry in batch_entries]
            if mode == "step":
                if not hasattr(train_wrapper, "get_action_probs_step_batch_from_proc_list"):
                    new_lp = torch.stack([_recompute_row_logprob(row) for row in rows])
                    _backward_entries(batch_entries, new_lp, forwards=len(rows))
                    continue
                new_lp = train_wrapper.get_action_probs_step_batch_from_proc_list(
                    [row.proc_snapshot for row in rows],
                    torch.stack([row.unsquashed_chunk.reshape(-1) for row in rows], dim=0).to(device),
                )
                _backward_entries(batch_entries, new_lp, forwards=1)
                continue
            if not hasattr(train_wrapper, "get_action_probs_for_chunk_batch_from_proc_list"):
                new_lp = torch.stack([_recompute_row_logprob(row) for row in rows])
                _backward_entries(batch_entries, new_lp, forwards=len(rows))
                continue
            new_lp_steps = train_wrapper.get_action_probs_for_chunk_batch_from_proc_list(
                [row.proc_snapshot for row in rows],
                torch.stack([row.unsquashed_chunk for row in rows], dim=0).to(device),
            )
            _backward_entries(batch_entries, new_lp_steps.reshape(len(rows), -1).sum(dim=1), forwards=1)
    return len(entries)


def _legacy_backward_phase11_chunk_loss(
    *,
    train_wrapper,
    action_chunks,
    advantage: torch.Tensor,
    device: torch.device,
    optimizer_chunk_size: int,
    clip_eps: float,
    normalizer: int,
) -> int:
    rows = list(action_chunks)
    if not rows:
        return 0
    denom = int(normalizer)
    if denom <= 0:
        raise ValueError("normalizer must be > 0")
    opt_chunk = max(int(optimizer_chunk_size), 1)

    def _recompute_row_logprob(row) -> torch.Tensor:
        unsquashed = row.unsquashed_chunk.to(device)
        if getattr(row, "logprob_mode", "chunk") == "step":
            return train_wrapper.get_action_probs_from_proc_list(
                [row.proc_snapshot],
                unsquashed.reshape(1, -1).to(device),
            ).sum()
        return train_wrapper.get_action_probs_for_chunk_from_proc(
            row.proc_snapshot,
            unsquashed,
        ).sum()

    for cs in range(0, len(rows), opt_chunk):
        batch_rows = rows[cs : cs + opt_chunk]
        new_lp = torch.stack(
            [
                _recompute_row_logprob(row)
                for row in batch_rows
            ]
        ).float()
        old_lp = torch.stack([row.log_prob_sum for row in batch_rows]).to(
            device=new_lp.device,
            dtype=torch.float32,
        ).reshape(-1)
        adv = advantage.to(device=new_lp.device, dtype=torch.float32)
        losses = _phase11_clipped_row_loss(new_lp.reshape(-1), old_lp, adv, clip_eps)
        (losses.sum() / denom).backward()
    return len(rows)


def main() -> int:
    from torch import nn

    from smolvla_grpo.checkpointing import load_grpo_checkpoint, save_grpo_checkpoint
    from smolvla_grpo.grpo_math import compute_group_advantages
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
    p.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Optimizer microbatch size for logprob recompute/backward",
    )
    p.add_argument(
        "--logprob-recompute-mode",
        choices=("batched", "loop"),
        default="batched",
        help="Batched is the default fast path; loop is a slow debug fallback.",
    )
    p.add_argument(
        "--logprob-batch-size",
        type=int,
        default=16,
        help="Batch size for optimizer logprob recompute. Separate from --chunk-size.",
    )
    p.add_argument(
        "--rollout-policy-batch-size",
        type=int,
        default=32,
        help="SmolVLA forward batch during vector rollout. Independent of --group-size and --logprob-batch-size.",
    )
    p.add_argument(
        "--action-chunk-size",
        type=int,
        default=1,
        help="Open-loop rollout horizon. 1 keeps legacy Phase11 behavior; 5 samples one 5-step action chunk per root observation.",
    )
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--init-log-std", type=float, default=-2.0)
    p.add_argument("--euler-step-noise-std", type=float, default=0.2)
    p.add_argument(
        "--action-transform",
        choices=("no_tanh", "tanh_norm_ablation"),
        default="no_tanh",
        help="GRPO sampling transform before LeRobot postprocessor; tanh path is explicit ablation only.",
    )
    p.add_argument("--run-label", type=str, default="no_tanh_main")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    args = p.parse_args()
    if args.batch_size != 1:
        raise SystemExit("Only batch_size=1 supported (one seed context per update).")
    if int(args.chunk_size) < 1:
        raise SystemExit("--chunk-size must be >= 1")
    if int(args.logprob_batch_size) < 1:
        raise SystemExit("--logprob-batch-size must be >= 1")
    if int(args.rollout_policy_batch_size) < 1:
        raise SystemExit("--rollout-policy-batch-size must be >= 1")
    if int(args.action_chunk_size) < 1:
        raise SystemExit("--action-chunk-size must be >= 1")
    if args.logprob_recompute_mode == "loop":
        print("warning: loop logprob recompute is slow; use batched unless debugging", flush=True)

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
        n_action_steps=int(args.action_chunk_size),
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
        "async_start_method": (
            args.async_start_method if args.rollout_execution == "vector_async" else None
        ),
        "requested_max_steps": args.max_steps,
        "train_seed_base": args.train_seed_base,
        "start_update": start_u,
        "end_update": end_u,
        "max_steps": args.max_steps,
        "group_size": args.group_size,
        "optimizer_chunk_size": int(args.chunk_size),
        "logprob_recompute_mode": args.logprob_recompute_mode,
        "logprob_batch_size": int(args.logprob_batch_size),
        "rollout_policy_batch_size": int(args.rollout_policy_batch_size),
        "action_chunk_size": int(args.action_chunk_size),
        "loss_unit": "policy_chunk",
        "clip_eps": args.clip_eps,
        "lr": args.lr,
        "action_transform": args.action_transform,
        "run_label": args.run_label,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for update in range(start_u, end_u):
        update_t0 = time.perf_counter()
        reset_seed = int(args.train_seed_base) + int(update)
        # region agent log
        _debug_mem_log(
            hypothesis_id="H1_H2_H5",
            location="scripts/grpo/train_phase11_env_on_policy_grpo.py:249",
            message="update_start_mem",
            data={"update": int(update), "group_size": int(args.group_size)},
        )
        # endregion
        rollout_t0 = time.perf_counter()
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
            action_chunk_size=int(args.action_chunk_size),
            rollout_policy_batch_size=int(args.rollout_policy_batch_size),
        )
        rollout_seconds = float(time.perf_counter() - rollout_t0)
        # region agent log
        _debug_mem_log(
            hypothesis_id="H1_H2_H5",
            location="scripts/grpo/train_phase11_env_on_policy_grpo.py:270",
            message="after_rollout_mem",
            data={
                "update": int(update),
                "rollout_seconds": rollout_seconds,
                "rollouts": len(rollouts),
                "action_chunks": int(sum(len(getattr(tr, "action_chunks", []) or []) for tr in rollouts)),
                "proc_snapshots": int(sum(len(getattr(tr, "proc_snapshots", []) or []) for tr in rollouts)),
            },
        )
        # endregion
        returns = torch.tensor(
            [reward_backend.episode_return(tr) for tr in rollouts],
            dtype=torch.float32,
            device=device,
        )
        successes = [any(bool(s) for s in tr.successes) for tr in rollouts]
        success_rate = float(sum(1 for s in successes if s) / max(len(successes), 1))
        avg_ret = float(returns.mean().item())
        episode_lengths = [len(tr.rewards) for tr in rollouts]
        num_env_steps = int(sum(episode_lengths))
        num_policy_sample_calls = int(
            sum(
                int(
                    (getattr(tr, "metadata", {}) or {}).get(
                        "policy_sample_calls",
                        len(getattr(tr, "action_chunks", []) or []),
                    )
                )
                for tr in rollouts
            )
        )
        num_loss_units = int(
            sum(len(getattr(tr, "action_chunks", []) or []) for tr in rollouts)
        )
        terminated_flags = [bool(tr.terminated) for tr in rollouts]
        truncated_flags = [bool(tr.truncated) for tr in rollouts]
        clip_values = [float(v) for tr in rollouts for v in tr.action_clip_fractions]
        clip_any_values = [bool(v) for tr in rollouts for v in tr.action_clip_any]
        action_clip_fraction = float(sum(clip_values) / max(len(clip_values), 1))
        action_clip_any_fraction = float(
            sum(1 for v in clip_any_values if v) / max(len(clip_any_values), 1)
        )
        resolved_max_steps = int(args.max_steps)
        if rollouts:
            resolved_max_steps = int(
                rollouts[0].metadata.get("resolved_max_steps", resolved_max_steps)
            )
        metrics_common = {
            "env_backend": args.env_backend,
            "rollout_execution": args.rollout_execution,
            "action_transform": args.action_transform,
            "run_label": args.run_label,
            "async_start_method": (
                args.async_start_method if args.rollout_execution == "vector_async" else None
            ),
            "max_steps": args.max_steps,
            "resolved_max_steps": resolved_max_steps,
            "avg_return": avg_ret,
            "successes": successes,
            "success_rate": success_rate,
            "episode_lengths": episode_lengths,
            "num_env_steps": num_env_steps,
            "num_policy_sample_calls": num_policy_sample_calls,
            "num_loss_units": num_loss_units,
            "optimizer_chunk_size": int(args.chunk_size),
            "logprob_recompute_mode": args.logprob_recompute_mode,
            "logprob_batch_size": int(args.logprob_batch_size),
            "rollout_policy_batch_size": int(args.rollout_policy_batch_size),
            "action_chunk_size": int(args.action_chunk_size),
            "loss_unit": "policy_chunk",
            "action_clip_fraction": action_clip_fraction,
            "action_clip_any_fraction": action_clip_any_fraction,
            "terminated": terminated_flags,
            "truncated": truncated_flags,
        }
        advantages = compute_group_advantages(returns)
        if torch.allclose(advantages, torch.zeros_like(advantages)):
            update_seconds = float(time.perf_counter() - update_t0)
            skipped_extra = {
                **metrics_common,
                "skipped": True,
                "rollout_seconds": rollout_seconds,
                "optimize_seconds": 0.0,
                "update_seconds": update_seconds,
                "num_logprob_forward_batches": 0,
                "ratio_mean": None,
                "ratio_min": None,
                "ratio_max": None,
                "ratio_clip_fraction": None,
                "approx_kl": None,
                "old_logprob_sum_mean": None,
                "new_logprob_sum_mean": None,
                **_log_std_telemetry(bundle.policy),
            }
            _append_progress(
                progress_path,
                {
                    "update": update,
                    "skipped": True,
                    "reason": "zero_advantages",
                    "reset_seed": reset_seed,
                    "returns": returns.detach().cpu().tolist(),
                    **skipped_extra,
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
                args=_json_ready_args(args),
                extra=skipped_extra,
            )
            if (update + 1) % args.save_every == 0 or update == end_u - 1:
                save_grpo_checkpoint(
                    ckpt_dir / f"update_{update + 1:04d}.pt",
                    policy_state=bundle.policy.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    update_index=update,
                    args=_json_ready_args(args),
                    extra=skipped_extra,
                )
            print(
                "phase111_grpo_update",
                f"update={update}",
                f"mode={args.rollout_execution}",
                f"label={args.run_label}",
                f"action_transform={args.action_transform}",
                f"seed={reset_seed}",
                f"avg_return={avg_ret:.6g}",
                f"success_rate={success_rate:.3f}",
                f"episode_lengths={episode_lengths}",
                f"env_steps={num_env_steps}",
                f"policy_calls={num_policy_sample_calls}",
                f"loss_units={num_loss_units}",
                f"action_chunk_size={int(args.action_chunk_size)}",
                f"logprob_mode={args.logprob_recompute_mode}",
                f"logprob_bs={int(args.logprob_batch_size)}",
                "lp_batches=0",
                "ratio_clip=null",
                "approx_kl=null",
                f"log_std_mean={_fmt_metric(skipped_extra.get('log_std_mean'))}",
                f"clip_frac={action_clip_fraction:.4f}",
                f"clip_any_frac={action_clip_any_fraction:.4f}",
                f"rollout_s={rollout_seconds:.2f}",
                "opt_s=0.00",
                f"update_s={update_seconds:.2f}",
                "skipped=zero_advantages",
                flush=True,
            )
            continue

        bundle.policy.train()
        optimize_t0 = time.perf_counter()
        loss_telemetry = _new_loss_telemetry(
            mode=args.logprob_recompute_mode,
            batch_size=int(args.logprob_batch_size),
        )
        for _epoch in range(args.update_epochs):
            optimizer.zero_grad()
            _backward_phase11_group_loss(
                train_wrapper=train_wrapper,
                rollouts=rollouts,
                advantages=advantages,
                device=device,
                optimizer_chunk_size=int(args.chunk_size),
                clip_eps=float(args.clip_eps),
                logprob_recompute_mode=args.logprob_recompute_mode,
                logprob_batch_size=int(args.logprob_batch_size),
                telemetry=loss_telemetry,
            )

            nn.utils.clip_grad_norm_(bundle.policy.parameters(), args.grad_clip)
            optimizer.step()
        optimize_seconds = float(time.perf_counter() - optimize_t0)
        loss_telemetry_row = _finalize_loss_telemetry(loss_telemetry, clip_eps=float(args.clip_eps))
        # region agent log
        _debug_mem_log(
            hypothesis_id="H2_H4",
            location="scripts/grpo/train_phase11_env_on_policy_grpo.py:428",
            message="after_optimize_mem",
            data={"update": int(update), "optimize_seconds": optimize_seconds},
        )
        # endregion
        update_seconds = float(time.perf_counter() - update_t0)
        checkpoint_extra = {
            **metrics_common,
            "rollout_seconds": rollout_seconds,
            "optimize_seconds": optimize_seconds,
            "update_seconds": update_seconds,
            **loss_telemetry_row,
            **_log_std_telemetry(bundle.policy),
        }

        _append_progress(
            progress_path,
            {
                "update": update,
                "reset_seed": reset_seed,
                "returns": returns.detach().cpu().tolist(),
                "advantages": advantages.detach().cpu().tolist(),
                **checkpoint_extra,
            },
        )
        state = {k: v.clone() for k, v in bundle.policy.state_dict().items()}
        old_policy.load_state_dict(state)
        old_policy.eval()
        # region agent log
        _debug_mem_log(
            hypothesis_id="H3_H4",
            location="scripts/grpo/train_phase11_env_on_policy_grpo.py:452",
            message="after_old_policy_sync_mem",
            data={"update": int(update), "state_tensors": len(state)},
        )
        # endregion

        save_grpo_checkpoint(
            ckpt_dir / "latest.pt",
            policy_state=bundle.policy.state_dict(),
            optimizer_state=optimizer.state_dict(),
            update_index=update,
            args=_json_ready_args(args),
            extra=checkpoint_extra,
        )
        if (update + 1) % args.save_every == 0 or update == end_u - 1:
            save_grpo_checkpoint(
                ckpt_dir / f"update_{update + 1:04d}.pt",
                policy_state=bundle.policy.state_dict(),
                optimizer_state=optimizer.state_dict(),
                update_index=update,
                args=_json_ready_args(args),
                extra=checkpoint_extra,
            )
        # region agent log
        _debug_mem_log(
            hypothesis_id="H3",
            location="scripts/grpo/train_phase11_env_on_policy_grpo.py:473",
            message="after_checkpoint_mem",
            data={"update": int(update), "save_every": int(args.save_every)},
        )
        # endregion
        print(
            "phase111_grpo_update",
            f"update={update}",
            f"mode={args.rollout_execution}",
            f"label={args.run_label}",
            f"action_transform={args.action_transform}",
            f"seed={reset_seed}",
            f"avg_return={avg_ret:.6g}",
            f"success_rate={success_rate:.3f}",
            f"episode_lengths={episode_lengths}",
            f"env_steps={num_env_steps}",
            f"policy_calls={num_policy_sample_calls}",
            f"loss_units={num_loss_units}",
            f"action_chunk_size={int(args.action_chunk_size)}",
            f"logprob_mode={args.logprob_recompute_mode}",
            f"logprob_bs={int(args.logprob_batch_size)}",
            f"lp_batches={checkpoint_extra.get('num_logprob_forward_batches')}",
            f"ratio_clip={_fmt_metric(checkpoint_extra.get('ratio_clip_fraction'))}",
            f"approx_kl={_fmt_metric(checkpoint_extra.get('approx_kl'))}",
            f"log_std_mean={_fmt_metric(checkpoint_extra.get('log_std_mean'))}",
            f"clip_frac={action_clip_fraction:.4f}",
            f"clip_any_frac={action_clip_any_fraction:.4f}",
            f"rollout_s={rollout_seconds:.2f}",
            f"opt_s={optimize_seconds:.2f}",
            f"update_s={update_seconds:.2f}",
            flush=True,
        )

    print(f"Done. Artifacts under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
