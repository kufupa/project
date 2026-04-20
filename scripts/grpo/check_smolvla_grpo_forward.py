#!/usr/bin/env python3
"""GPU smoke: load SmolVLA checkpoint, run select_action_distr_params, backward on log_std."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/grpo/...` from repo root
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env-backend", choices=("custom", "official_lerobot"), default="custom")
    args = parser.parse_args()

    from metaworld_determinism import gymnasium_reset_strict, seed_metaworld_process
    from smolvla_grpo.phase11_rollout import PushV3GRPOEnv, load_bundle_for_grpo
    from smolvla_grpo.policy_wrapper import MetaWorldSmolVLAGRPOPolicy, freeze_all_but_grpo_trainables
    from smolvla_pipeline.evaluator import (
        _resolve_camera_name,
        _resolve_flip_corner2,
        _resolve_task_text,
    )

    task = "push-v3"
    bundle, action_dim = load_bundle_for_grpo(args.checkpoint, task=task, env_backend=args.env_backend)
    task_text = _resolve_task_text("push-v3", override=None)
    env_h = None
    cam = _resolve_camera_name()
    flip = _resolve_flip_corner2()

    wrapper = MetaWorldSmolVLAGRPOPolicy(
        bundle,
        task=task,
        task_text=task_text,
        camera_name=cam,
        flip_corner2=flip,
        action_dim=action_dim,
    )
    wrapper.assert_grpo_api()
    wrapper.set_log_std(-2.0)
    wrapper.set_euler_step_noise_std(0.2)
    freeze_all_but_grpo_trainables(bundle.policy)
    bundle.policy.train()

    try:
        if args.env_backend == "official_lerobot":
            from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

            env_h = OfficialLeRobotMetaWorldGRPORollout(task=task)
            obs = env_h.reset(0)
            proc = env_h.build_proc(obs, bundle=bundle)
        else:
            env_h = PushV3GRPOEnv(task=task)
            seed_metaworld_process(0)
            if env_h._tasks:  # noqa: SLF001
                env_h.set_task_for_episode(0)
            obs = gymnasium_reset_strict(env_h.inner, 0)
            if isinstance(obs, tuple):
                obs = obs[0]
            proc = wrapper.build_proc_batch(obs, env_h.inner)
        proc_d = wrapper._proc_to_device(proc)  # noqa: SLF001
        mean, log_std = bundle.policy.select_action_distr_params(proc_d)
        loss = mean.sum() + log_std.sum()
        loss.backward()
    finally:
        if env_h is not None:
            env_h.close()
    print("OK: forward+backward on distr params completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
