#!/usr/bin/env python3
"""Phase11: env-on-policy GRPO for SmolVLA on MetaWorld Push-v3."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from smolvla_grpo.process_memory import prefixed_process_tree_memory_fields


def _append_progress(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row) + "\n")


def _json_ready_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _proc_mem_fields(stage: str) -> dict[str, int]:
    return prefixed_process_tree_memory_fields(f"proc_mem_{stage}")


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
    device: torch.device,
    tolerance: float,
):
    """Recompute stored actions through the live update path before optimizer.step."""
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
    from smolvla_grpo.chunk_math import masked_chunk_sum
    from smolvla_grpo.grpo_math import summarize_logprob_ratio_parity

    parity_old: list[torch.Tensor] = []
    parity_new: list[torch.Tensor] = []
    per_action_abs: list[torch.Tensor] = []
    with torch.no_grad():
        for traj in rollouts:
            for chunk in traj.chunks:
                live_steps, _mu, _log_std = train_wrapper.get_flow_sde_log_probs_for_chunk_from_proc_list(
                    [chunk.proc_snapshot],
                    [chunk.flow_sde_trace],
                    chunk_len=int(chunk_len),
                )
                live_steps = live_steps.detach().cpu()
                valid = chunk.valid_action_mask.reshape(1, -1)
                old_step = chunk.log_probs.reshape(1, -1)
                new_step = live_steps.reshape(1, -1)
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
    grpo_group_size: int | None = None,
) -> int:
    G = int(grpo_group_size if grpo_group_size is not None else len(rollouts))
    if G < 1:
        raise ValueError("grpo_group_size must be >= 1")
    if len(rollouts) % G != 0:
        raise ValueError(
            f"rollout count {len(rollouts)} must be a multiple of grpo_group_size={G}"
        )
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

    from smolvla_grpo.checkpointing import (
        load_grpo_checkpoint,
        save_grpo_checkpoint,
        save_rlinf_eval_checkpoint,
        validate_rlinf_eval_checkpoint,
    )
    from smolvla_grpo.grpo_math import (
        apply_grpo_regularizers,
        compute_group_advantages,
        compute_seed_batch_advantages,
        update_metrics,
    )
    from smolvla_grpo.chunk_math import masked_chunk_sum
    from smolvla_grpo.phase11_chunk_rollout import collect_chunk_rollout_group
    from smolvla_grpo.phase11_rollout import (
        collect_rollout_group,
        collect_rollout_seed_batch,
        load_bundle_for_grpo,
    )
    from smolvla_grpo.policy_wrapper import (
        MetaWorldSmolVLAGRPOPolicy,
        freeze_all_but_grpo_trainables,
    )
    from smolvla_grpo.reward_backends import episode_return_for_mode, make_phase11_reward_backend
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
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of reset seeds per update; each seed gets group_size rollouts.",
    )
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
    p.add_argument("--rollout-unit", choices=("step", "chunk"), default="step")
    p.add_argument("--rollout-chunk-len", type=int, default=5)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--init-log-std", type=float, default=-2.0)
    p.add_argument(
        "--euler-step-noise-std",
        type=float,
        default=0.2,
        help="Must be 0 for flow_sde/chunk unless --allow-euler-noise",
    )
    p.add_argument(
        "--allow-euler-noise",
        action="store_true",
        help="Allow nonzero euler_step_noise_std on chunk/flow_sde path",
    )
    p.add_argument(
        "--action-transform",
        choices=("no_tanh", "tanh_norm_ablation"),
        default="no_tanh",
        help="GRPO sampling transform before LeRobot postprocessor; tanh path is explicit ablation only.",
    )
    p.add_argument("--run-label", type=str, default="no_tanh_main")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument(
        "--save-rlinf-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write slim checkpoints_eval/*.pt for RLinf eval",
    )
    p.add_argument("--parity-tolerance", type=float, default=0.02)
    p.add_argument(
        "--fail-on-parity-violation",
        action="store_true",
        help="Exit non-zero if pre-update logprob ratio parity fails (chunk path)",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--min-log-std", type=float, default=-4.0)
    p.add_argument("--kl-beta", type=float, default=0.0)
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument(
        "--logprob-mode",
        choices=("gaussian", "flow_sde"),
        default="gaussian",
        help="flow_sde requires --rollout-unit chunk",
    )
    p.add_argument("--flow-sde-noise-level", type=float, default=0.5)
    p.add_argument("--flow-sde-trace-step", type=int, default=0)
    p.add_argument(
        "--gaussian-logprob-action",
        choices=("executed", "unsquashed"),
        default="executed",
        help="Action tensor scored by Gaussian logprob; unsquashed recreates the pre-A.3 G8 ablation.",
    )
    p.add_argument("--success-bonus", type=float, default=0.0)
    p.add_argument("--clip-penalty", type=float, default=0.0)
    p.add_argument(
        "--reward-mode",
        choices=("dense_return", "sparse_success_delta", "success_first_dense"),
        default="dense_return",
        help="dense_return sums env step rewards; sparse/success_first use episode_return_for_mode on chunk path.",
    )
    p.add_argument(
        "--reward-coef",
        type=float,
        default=1.0,
        help="Scale for sparse success reward (ignored for dense_return).",
    )
    p.add_argument(
        "--use-rel-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sparse only: reward = delta of success indicator across steps.",
    )
    args = p.parse_args()
    if int(args.batch_size) < 1:
        raise SystemExit("--batch-size must be >= 1")
    if int(args.group_size) < 2:
        raise SystemExit("--group-size must be >= 2 for GRPO advantage normalization")
    if int(args.chunk_size) < 1:
        raise SystemExit("--chunk-size must be >= 1")
    if int(args.logprob_batch_size) < 1:
        raise SystemExit("--logprob-batch-size must be >= 1")
    if int(args.rollout_policy_batch_size) < 1:
        raise SystemExit("--rollout-policy-batch-size must be >= 1")
    if int(args.action_chunk_size) < 1:
        raise SystemExit("--action-chunk-size must be >= 1")
    if int(args.rollout_chunk_len) < 1:
        raise SystemExit("--rollout-chunk-len must be >= 1")
    if args.rollout_unit == "chunk" and int(args.batch_size) != 1:
        raise SystemExit("chunk rollout requires --batch-size 1")
    if args.logprob_mode == "flow_sde" and args.rollout_unit != "chunk":
        raise SystemExit("flow_sde requires --rollout-unit chunk")
    if args.rollout_unit == "chunk" and args.logprob_mode != "flow_sde":
        raise SystemExit("chunk rollout currently requires --logprob-mode flow_sde")
    if args.logprob_mode == "flow_sde" and args.action_transform != "no_tanh":
        raise SystemExit("flow_sde requires --action-transform no_tanh")
    if args.rollout_unit == "chunk" and args.env_backend != "official_lerobot":
        raise SystemExit("chunk rollout requires --env-backend official_lerobot")
    if args.rollout_unit == "chunk" and float(args.euler_step_noise_std) > 0.0 and not args.allow_euler_noise:
        raise SystemExit(
            "euler_step_noise_std must be 0 for chunk/flow_sde GRPO "
            "(add --allow-euler-noise only for ablations)"
        )
    if args.logprob_recompute_mode == "loop":
        print("warning: loop logprob recompute is slow; use batched unless debugging", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_ckpt_dir = out / "checkpoints_eval"
    if args.save_rlinf_eval:
        eval_ckpt_dir.mkdir(parents=True, exist_ok=True)
    roll_dir = out / "rollouts"
    roll_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    manifest_path = out / "train_manifest.json"

    bundle, action_dim = load_bundle_for_grpo(
        args.checkpoint,
        task=args.task,
        env_backend=args.env_backend,
        n_action_steps=(
            int(args.rollout_chunk_len) if args.rollout_unit == "chunk" else int(args.action_chunk_size)
        ),
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

    def save_checkpoint_pair(name: str, *, update_index: int, extra: dict) -> None:
        full_path = ckpt_dir / name
        save_grpo_checkpoint(
            full_path,
            policy_state=bundle.policy.state_dict(),
            optimizer_state=optimizer.state_dict(),
            update_index=update_index,
            args=_json_ready_args(args),
            extra=extra,
        )
        if not args.save_rlinf_eval:
            return
        eval_path = eval_ckpt_dir / name
        save_rlinf_eval_checkpoint(
            eval_path,
            policy=bundle.policy,
            update_index=update_index,
            metrics=extra,
            source_checkpoint=full_path,
        )
        validate_rlinf_eval_checkpoint(eval_path, expected_update=update_index + 1)

    def persist_checkpoint(name: str, *, update_index: int, extra: dict) -> None:
        if args.save_rlinf_eval:
            save_checkpoint_pair(name, update_index=update_index, extra=extra)
            return
        save_grpo_checkpoint(
            ckpt_dir / name,
            policy_state=bundle.policy.state_dict(),
            optimizer_state=optimizer.state_dict(),
            update_index=update_index,
            args=_json_ready_args(args),
            extra=extra,
        )

    old_policy = copy.deepcopy(bundle.policy).eval().to(device)
    bundle.policy.train()

    reward_backend = make_phase11_reward_backend(
        reward_mode=args.reward_mode,
        reward_coef=float(args.reward_coef),
        use_rel_reward=bool(args.use_rel_reward),
        success_bonus=float(args.success_bonus),
        clip_penalty=float(args.clip_penalty),
    )
    end_u = start_u + int(args.num_updates)
    zero_advantage_skips = 0

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
        "batch_size": int(args.batch_size),
        "optimizer_chunk_size": int(args.chunk_size),
        "logprob_recompute_mode": args.logprob_recompute_mode,
        "logprob_batch_size": int(args.logprob_batch_size),
        "rollout_policy_batch_size": int(args.rollout_policy_batch_size),
        "action_chunk_size": int(args.action_chunk_size),
        "loss_unit": "policy_chunk" if args.rollout_unit == "step" else "flow_sde_chunk",
        "clip_eps": args.clip_eps,
        "lr": args.lr,
        "success_bonus": float(args.success_bonus),
        "clip_penalty": float(args.clip_penalty),
        "reward_mode": args.reward_mode,
        "reward_coef": float(args.reward_coef),
        "use_rel_reward": bool(args.use_rel_reward),
        "action_transform": args.action_transform,
        "gaussian_logprob_action": args.gaussian_logprob_action,
        "logprob_mode": args.logprob_mode,
        "flow_sde_noise_level": float(args.flow_sde_noise_level),
        "flow_sde_trace_step": int(args.flow_sde_trace_step),
        "euler_step_noise_std": float(args.euler_step_noise_std),
        "parity_tolerance": float(args.parity_tolerance),
        "save_rlinf_eval": bool(args.save_rlinf_eval),
        "run_label": args.run_label,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for update in range(start_u, end_u):
        update_t0 = time.perf_counter()
        group_size_i = int(args.group_size)
        batch_size_i = int(args.batch_size)
        seed_batch_base = int(args.train_seed_base) + int(update) * batch_size_i
        reset_seeds = [seed_batch_base + b for b in range(batch_size_i)]
        reset_seed = int(reset_seeds[0])
        proc_mem_update_start = _proc_mem_fields("update_start")
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
                group_size=group_size_i,
                action_dim=action_dim,
                device=device,
                chunk_len=int(args.rollout_chunk_len),
                rollout_execution=args.rollout_execution,
                async_start_method=args.async_start_method,
                action_transform=args.action_transform,
                gaussian_logprob_action=args.gaussian_logprob_action,
                logprob_mode=args.logprob_mode,
                flow_sde_noise_level=float(args.flow_sde_noise_level),
                flow_sde_trace_step=int(args.flow_sde_trace_step),
            )
        elif batch_size_i == 1:
            rollouts = collect_rollout_group(
                bundle=bundle,
                policy_old=old_policy,
                task=args.task,
                task_text=task_text,
                reset_seed=reset_seed,
                episode_index=update,
                max_steps=args.max_steps,
                group_size=group_size_i,
                action_dim=action_dim,
                device=device,
                env_backend=args.env_backend,
                rollout_execution=args.rollout_execution,
                async_start_method=args.async_start_method,
                action_transform=args.action_transform,
                action_chunk_size=int(args.action_chunk_size),
                rollout_policy_batch_size=int(args.rollout_policy_batch_size),
            )
        else:
            rollouts = collect_rollout_seed_batch(
                bundle=bundle,
                policy_old=old_policy,
                task=args.task,
                task_text=task_text,
                reset_seeds=reset_seeds,
                episode_index=update,
                max_steps=args.max_steps,
                group_size=group_size_i,
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
        proc_mem_after_rollout = _proc_mem_fields("after_rollout")

        if args.rollout_unit == "chunk":
            returns = torch.tensor(
                [episode_return_for_mode(tr, reward_mode=args.reward_mode) for tr in rollouts],
                dtype=torch.float32,
                device=device,
            )
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
                torch.cat(
                    [chunk.distr_log_std.reshape(-1, action_dim) for tr in rollouts for chunk in tr.chunks]
                ).to(device)
                if rollouts
                else torch.zeros(0, action_dim, device=device)
            )
            chunk_count = int(sum(len(tr.chunks) for tr in rollouts))
            valid_chunk_count = int(
                sum(1 for tr in rollouts for chunk in tr.chunks if bool(chunk.valid_action_mask.any()))
            )
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
            failure_advantages = [
                float(advantages_cpu[i].item()) for i, ok in enumerate(successes) if not bool(ok)
            ]
            success_negative_advantage_count = int(sum(1 for x in success_advantages if x < 0.0))
            failure_positive_advantage_count = int(sum(1 for x in failure_advantages if x > 0.0))
            success_returns = [float(returns_cpu[i].item()) for i, ok in enumerate(successes) if bool(ok)]
            failure_returns = [float(returns_cpu[i].item()) for i, ok in enumerate(successes) if not bool(ok)]
            max_success_return = max(success_returns) if success_returns else None
            max_failed_return = max(failure_returns) if failure_returns else None
            chunk_metrics_common = {
                "env_backend": args.env_backend,
                "rollout_execution": args.rollout_execution,
                "rollout_unit": args.rollout_unit,
                "rollout_chunk_len": int(args.rollout_chunk_len),
                "action_transform": args.action_transform,
                "gaussian_logprob_action": args.gaussian_logprob_action,
                "reward_mode": args.reward_mode,
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
                "chunk_count": int(chunk_count),
                "valid_chunk_count": int(valid_chunk_count),
                "action_clip_fraction": action_clip_fraction,
                "action_clip_any_fraction": action_clip_any_fraction,
                "postprocessor_oob_mean": postprocessor_oob_mean,
                "terminated": terminated_flags,
                "truncated": truncated_flags,
                "rollout_seconds": rollout_seconds,
                **proc_mem_update_start,
                **proc_mem_after_rollout,
            }
            if torch.allclose(advantages, torch.zeros_like(advantages)):
                zero_advantage_skips += 1
                update_seconds = float(time.perf_counter() - update_t0)
                proc_mem_after_optimize = _proc_mem_fields("after_optimize")
                skipped_extra = {
                    **chunk_metrics_common,
                    "skipped": True,
                    "optimize_seconds": 0.0,
                    "update_seconds": update_seconds,
                    **proc_mem_after_optimize,
                    "success_advantage_mean": float(sum(success_advantages) / max(len(success_advantages), 1)),
                    "success_negative_advantage_count": success_negative_advantage_count,
                    "failure_positive_advantage_count": failure_positive_advantage_count,
                    "max_success_return": max_success_return,
                    "max_failed_return": max_failed_return,
                    **pre_update_metrics,
                }
                _append_progress(
                    progress_path,
                    {
                        "update": update,
                        "skipped": True,
                        "reason": "zero_advantages",
                        "zero_advantage_skips": zero_advantage_skips,
                        "reset_seed": reset_seed,
                        "returns": returns.detach().cpu().tolist(),
                        **skipped_extra,
                    },
                )
                state = {k: v.clone() for k, v in bundle.policy.state_dict().items()}
                old_policy.load_state_dict(state)
                old_policy.eval()
                persist_checkpoint("latest.pt", update_index=update, extra=skipped_extra)
                if (update + 1) % args.save_every == 0 or update == end_u - 1:
                    persist_checkpoint(f"update_{update + 1:04d}.pt", update_index=update, extra=skipped_extra)
                print(
                    "phase111_grpo_update",
                    f"update={update}",
                    f"mode={args.rollout_execution}",
                    f"unit={args.rollout_unit}",
                    f"label={args.run_label}",
                    f"seed={reset_seed}",
                    f"avg_return={avg_ret:.6g}",
                    f"success_rate={success_rate:.3f}",
                    f"rollout_s={rollout_seconds:.2f}",
                    "opt_s=0.00",
                    f"update_s={update_seconds:.2f}",
                    "skipped=zero_advantages",
                    flush=True,
                )
                continue

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
                    for chunk in traj.chunks:
                        valid = chunk.valid_action_mask.reshape(1, -1).to(device)
                        if not bool(valid.any()):
                            continue
                        new_steps, _mu_live, log_std_live = (
                            train_wrapper.get_flow_sde_log_probs_for_chunk_from_proc_list(
                                [chunk.proc_snapshot],
                                [chunk.flow_sde_trace],
                                chunk_len=int(args.rollout_chunk_len),
                            )
                        )
                        old_steps = chunk.log_probs.reshape(1, -1).to(device)
                        new_step = new_steps.reshape(1, -1)
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
                            log_std=log_std_live.reshape(-1, action_dim),
                            kl_beta=float(args.kl_beta) / max(int(valid_chunk_count), 1),
                            entropy_coef=float(args.entropy_coef) / max(int(valid_chunk_count), 1),
                        )
                        chunk_loss.backward()
                        epoch_new_log_probs.append(new_lp.detach().cpu())
                        epoch_old_log_probs.append(old_lp.detach().cpu())
                        epoch_log_stds.append(log_std_live.detach().cpu().reshape(-1, action_dim))

                total_grad_norm = nn.utils.clip_grad_norm_(bundle.policy.parameters(), args.grad_clip)
                optimizer.step()
                last_new_log_probs = epoch_new_log_probs
                last_old_log_probs = epoch_old_log_probs
                last_log_stds = epoch_log_stds

            optimize_seconds = float(time.perf_counter() - optimize_t0)
            proc_mem_after_optimize = _proc_mem_fields("after_optimize")
            update_seconds = float(time.perf_counter() - update_t0)
            post_update_metrics = update_metrics(
                new_log_probs=torch.cat(last_new_log_probs) if last_new_log_probs else rollout_old_lp.detach().cpu(),
                old_log_probs=torch.cat(last_old_log_probs) if last_old_log_probs else rollout_old_lp.detach().cpu(),
                log_std=torch.cat(last_log_stds) if last_log_stds else rollout_log_std.detach().cpu(),
                returns=returns.detach().cpu(),
                advantages=advantages.detach().cpu(),
                epsilon=float(args.clip_eps),
            )
            checkpoint_extra = {
                **chunk_metrics_common,
                "optimize_seconds": optimize_seconds,
                "update_seconds": update_seconds,
                **proc_mem_after_optimize,
                "success_advantage_mean": float(sum(success_advantages) / max(len(success_advantages), 1)),
                "success_negative_advantage_count": success_negative_advantage_count,
                "failure_positive_advantage_count": failure_positive_advantage_count,
                "max_success_return": max_success_return,
                "max_failed_return": max_failed_return,
                "parity": parity_payload,
                "kl_beta": float(args.kl_beta),
                "entropy_coef": float(args.entropy_coef),
                "grad_norm_before_clip": float(total_grad_norm.detach().cpu().item())
                if torch.is_tensor(total_grad_norm)
                else float(total_grad_norm),
                **post_update_metrics,
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
            persist_checkpoint("latest.pt", update_index=update, extra=checkpoint_extra)
            if (update + 1) % args.save_every == 0 or update == end_u - 1:
                persist_checkpoint(f"update_{update + 1:04d}.pt", update_index=update, extra=checkpoint_extra)
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
                f"rss_tree_mb={checkpoint_extra.get('proc_mem_after_optimize_tree_rss_kb', 0) / 1024.0:.1f}",
                f"vmem_tree_mb={checkpoint_extra.get('proc_mem_after_optimize_tree_vmsize_kb', 0) / 1024.0:.1f}",
                flush=True,
            )
            continue

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
        per_seed_success_rate = [
            float(sum(1 for s in successes[i * group_size_i : (i + 1) * group_size_i] if s) / group_size_i)
            for i in range(batch_size_i)
        ]
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
            "batch_size": batch_size_i,
            "reset_seeds": reset_seeds,
            "per_seed_success_rate": per_seed_success_rate,
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
            "reward_mode": args.reward_mode,
            "reward_coef": float(args.reward_coef),
            "use_rel_reward": bool(args.use_rel_reward),
        }
        if batch_size_i == 1:
            advantages = compute_group_advantages(returns)
        else:
            advantages = compute_seed_batch_advantages(returns, group_size=group_size_i)
        if torch.allclose(advantages, torch.zeros_like(advantages)):
            zero_advantage_skips += 1
            update_seconds = float(time.perf_counter() - update_t0)
            proc_mem_after_optimize = _proc_mem_fields("after_optimize")
            skipped_extra = {
                **metrics_common,
                "skipped": True,
                "rollout_seconds": rollout_seconds,
                "optimize_seconds": 0.0,
                "update_seconds": update_seconds,
                **proc_mem_update_start,
                **proc_mem_after_rollout,
                **proc_mem_after_optimize,
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
                    "zero_advantage_skips": zero_advantage_skips,
                    "reset_seed": reset_seed,
                    "returns": returns.detach().cpu().tolist(),
                    **skipped_extra,
                },
            )
            state = {k: v.clone() for k, v in bundle.policy.state_dict().items()}
            old_policy.load_state_dict(state)
            old_policy.eval()
            persist_checkpoint("latest.pt", update_index=update, extra=skipped_extra)
            if (update + 1) % args.save_every == 0 or update == end_u - 1:
                persist_checkpoint(f"update_{update + 1:04d}.pt", update_index=update, extra=skipped_extra)
            print(
                "phase111_grpo_update",
                f"update={update}",
                f"mode={args.rollout_execution}",
                f"label={args.run_label}",
                f"action_transform={args.action_transform}",
                f"seeds={reset_seeds}",
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
                f"rss_tree_mb={skipped_extra.get('proc_mem_after_optimize_tree_rss_kb', 0) / 1024.0:.1f}",
                f"vmem_tree_mb={skipped_extra.get('proc_mem_after_optimize_tree_vmsize_kb', 0) / 1024.0:.1f}",
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
                grpo_group_size=group_size_i,
            )

            nn.utils.clip_grad_norm_(bundle.policy.parameters(), args.grad_clip)
            optimizer.step()
        optimize_seconds = float(time.perf_counter() - optimize_t0)
        loss_telemetry_row = _finalize_loss_telemetry(loss_telemetry, clip_eps=float(args.clip_eps))
        proc_mem_after_optimize = _proc_mem_fields("after_optimize")
        update_seconds = float(time.perf_counter() - update_t0)
        checkpoint_extra = {
            **metrics_common,
            "rollout_seconds": rollout_seconds,
            "optimize_seconds": optimize_seconds,
            "update_seconds": update_seconds,
            **proc_mem_update_start,
            **proc_mem_after_rollout,
            **proc_mem_after_optimize,
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

        persist_checkpoint("latest.pt", update_index=update, extra=checkpoint_extra)
        if (update + 1) % args.save_every == 0 or update == end_u - 1:
            persist_checkpoint(f"update_{update + 1:04d}.pt", update_index=update, extra=checkpoint_extra)
        print(
            "phase111_grpo_update",
            f"update={update}",
            f"mode={args.rollout_execution}",
            f"label={args.run_label}",
            f"action_transform={args.action_transform}",
            f"seeds={reset_seeds}",
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
            f"rss_tree_mb={checkpoint_extra.get('proc_mem_after_optimize_tree_rss_kb', 0) / 1024.0:.1f}",
            f"vmem_tree_mb={checkpoint_extra.get('proc_mem_after_optimize_tree_vmsize_kb', 0) / 1024.0:.1f}",
            flush=True,
        )

    manifest["zero_advantage_skips"] = zero_advantage_skips
    manifest["completed_updates"] = int(end_u - start_u)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Done. Artifacts under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
