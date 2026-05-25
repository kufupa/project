#!/usr/bin/env python3
"""Best-effort ManiSkill rollout benchmark for a SmolVLA checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE
from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles, quaternion_to_matrix


def resolve_policy_path(path: Path) -> Path:
    if (path / "model.safetensors").exists():
        return path
    candidates = sorted(path.glob("checkpoints/*/pretrained_model"))
    last = path / "checkpoints" / "last" / "pretrained_model"
    if last.exists():
        return last.resolve()
    if candidates:
        return candidates[-1].resolve()
    raise SystemExit(f"cannot resolve policy checkpoint under {path}")


def tensor_to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def current_state(env: gym.Env, gripper: float) -> np.ndarray:
    pose = env.unwrapped.agent.ee_pose_at_robot_base
    state = torch.cat(
        [
            pose.p[0],
            matrix_to_euler_angles(quaternion_to_matrix(pose.q[0]), "XYZ"),
            torch.as_tensor([gripper], device=env.unwrapped.device),
        ]
    )
    return state.detach().cpu().numpy().astype(np.float32)


def current_image(env: gym.Env, camera: str) -> np.ndarray:
    obs = env.unwrapped.get_obs()
    rgb = obs["sensor_data"][camera]["rgb"].squeeze(0)
    image = tensor_to_numpy(rgb).astype(np.float32) / 255.0
    return np.moveaxis(image, -1, 0)


def is_success(info: dict[str, Any]) -> bool:
    value = info.get("success", False)
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().numpy().reshape(-1)[0])
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0])
    return bool(value)


def rollout(args: argparse.Namespace) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, download_videos=False)
    policy_path = resolve_policy_path(args.policy_path)
    policy = SmolVLAPolicy.from_pretrained(policy_path, device=device, local_files_only=args.local_files_only)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        dataset_stats=dataset.meta.stats,
    )

    env = gym.make(
        args.env_id,
        obs_mode="rgb+segmentation",
        num_envs=1,
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="none",
        sensor_configs={"shader_pack": "default"},
        human_render_camera_configs={"shader_pack": "default"},
        viewer_camera_configs={"shader_pack": "default"},
        sim_backend="cpu",
        sim_config={"sim_freq": 500, "control_freq": 5},
    )

    successes = 0
    episodes = []
    try:
        for episode_idx in range(args.episodes):
            policy.reset()
            obs, info = env.reset(options={"obj_set": args.obj_set})
            del obs
            task = env.unwrapped.get_language_instruction()
            gripper = 1.0
            episode_success = False
            last_info = info
            for step in range(args.max_steps):
                batch = {
                    "observation.images.front": torch.from_numpy(current_image(env, args.camera)),
                    OBS_STATE: torch.from_numpy(current_state(env, gripper)),
                    "task": task,
                }
                model_batch = preprocessor(batch)
                with torch.no_grad():
                    action = policy.select_action(model_batch)
                    action = postprocessor(action)
                action_np = action.detach().cpu().numpy().reshape(-1).astype(np.float32)
                action_np = np.nan_to_num(action_np, nan=0.0, posinf=1.0, neginf=-1.0)
                action_np[:6] = np.clip(action_np[:6], -1.0, 1.0)
                action_np[6] = 1.0 if action_np[6] >= 0 else -1.0
                gripper = float(action_np[6])
                _, _, terminated, truncated, last_info = env.step(action_np)
                episode_success = is_success(last_info)
                if episode_success or bool(terminated) or bool(truncated):
                    break
            successes += int(episode_success)
            episodes.append(
                {
                    "episode": episode_idx,
                    "success": episode_success,
                    "steps": step + 1,
                    "task": task,
                    "info": {k: str(v) for k, v in last_info.items()},
                }
            )
    finally:
        env.close()

    return {
        "policy_path": str(policy_path),
        "dataset_root": str(args.dataset_root),
        "env_id": args.env_id,
        "obj_set": args.obj_set,
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / max(args.episodes, 1),
        "episode_results": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--env-id", default="PutOnPlateInScene25Main-v3")
    parser.add_argument("--obj-set", default="train")
    parser.add_argument("--camera", default="3rd_view_camera")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    summary = rollout(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
