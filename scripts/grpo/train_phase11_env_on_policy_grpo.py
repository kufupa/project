#!/usr/bin/env python3
"""Phase11: env-on-policy GRPO for SmolVLA on MetaWorld Push-v3."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


def _append_progress(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row) + "\n")


def main() -> int:
    import torch
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
    p.add_argument("--train-seed-base", type=int, default=2000)
    p.add_argument("--start-update", type=int, default=0)
    p.add_argument("--num-updates", type=int, default=1, help="Number of updates to run from start-update")
    p.add_argument("--resume", type=Path, default=None, help="Path to .pt from save_grpo_checkpoint")
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1, help="Must be 1 for single-seed group")
    p.add_argument("--update-epochs", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=5)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--init-log-std", type=float, default=-2.0)
    p.add_argument("--euler-step-noise-std", type=float, default=0.2)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    args = p.parse_args()
    if args.batch_size != 1:
        raise SystemExit("Only batch_size=1 supported (one seed context per update).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    roll_dir = out / "rollouts"
    roll_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    manifest_path = out / "train_manifest.json"

    bundle, action_dim = load_bundle_for_grpo(args.checkpoint, task=args.task)
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
        "train_seed_base": args.train_seed_base,
        "start_update": start_u,
        "end_update": end_u,
        "max_steps": args.max_steps,
        "group_size": args.group_size,
        "clip_eps": args.clip_eps,
        "lr": args.lr,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for update in range(start_u, end_u):
        reset_seed = int(args.train_seed_base) + int(update)
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
        )
        returns = torch.tensor(
            [reward_backend.episode_return(tr) for tr in rollouts],
            dtype=torch.float32,
            device=device,
        )
        advantages = compute_group_advantages(returns)
        if torch.allclose(advantages, torch.zeros_like(advantages)):
            _append_progress(
                progress_path,
                {
                    "update": update,
                    "skipped": True,
                    "reason": "zero_advantages",
                    "reset_seed": reset_seed,
                    "returns": returns.detach().cpu().tolist(),
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
                extra={"skipped": True},
            )
            continue

        bundle.policy.train()
        for _epoch in range(args.update_epochs):
            optimizer.zero_grad()
            for gi, traj in enumerate(rollouts):
                A = advantages[gi].reshape(()).float()
                T = len(traj.proc_snapshots)
                G = len(rollouts)
                for cs in range(0, T, args.chunk_size):
                    ce = min(cs + args.chunk_size, T)
                    procs = traj.proc_snapshots[cs:ce]
                    u_chunk = torch.stack([traj.unsquashed_actions[t] for t in range(cs, ce)]).to(
                        device
                    )
                    old_lp = torch.stack([traj.log_probs[t] for t in range(cs, ce)]).to(device).reshape(-1)
                    new_lp = train_wrapper.get_action_probs_from_proc_list(procs, u_chunk).reshape(-1)
                    ratio = torch.exp(new_lp - old_lp)
                    unclipped = ratio * A
                    clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * A
                    step_losses = -torch.min(unclipped, clipped)
                    chunk_loss = step_losses.sum() / max(T * G, 1)
                    chunk_loss.backward()

            nn.utils.clip_grad_norm_(bundle.policy.parameters(), args.grad_clip)
            optimizer.step()

        avg_ret = float(returns.mean().item())
        _append_progress(
            progress_path,
            {
                "update": update,
                "reset_seed": reset_seed,
                "avg_return": avg_ret,
                "returns": returns.detach().cpu().tolist(),
                "advantages": advantages.detach().cpu().tolist(),
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
            extra={"avg_return": avg_ret},
        )
        if (update + 1) % args.save_every == 0 or update == end_u - 1:
            save_grpo_checkpoint(
                ckpt_dir / f"update_{update + 1:04d}.pt",
                policy_state=bundle.policy.state_dict(),
                optimizer_state=optimizer.state_dict(),
                update_index=update,
                args=vars(args),
                extra={"avg_return": avg_ret},
            )

    print(f"Done. Artifacts under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
