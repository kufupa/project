"""Official LeRobot MetaWorld vector-env adapter for SmolVLA GRPO rollouts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


def _coerce_success_value(value: Any) -> bool:
    success_arr = np.asarray(value).reshape(-1)
    return bool(success_arr[0]) if success_arr.size else False


def _first_success(value: Any) -> bool | None:
    if isinstance(value, dict):
        if "is_success" in value:
            return _coerce_success_value(value["is_success"])
        for item in value.values():
            found = _first_success(item)
            if found is not None:
                return found
        return None
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            for item in value.reshape(-1):
                found = _first_success(item)
                if found is not None:
                    return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_success(item)
            if found is not None:
                return found
    return None


@dataclass
class OfficialStep:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    info: dict[str, Any]


class OfficialLeRobotMetaWorldGRPORollout:
    """GRPO rollout adapter matching LeRobot's official vector eval path."""

    def __init__(self, *, task: str, obs_type: str = "pixels_agent_pos") -> None:
        from lerobot.envs.configs import MetaworldEnv
        from lerobot.envs.factory import make_env

        self.task_group = task
        self.task_id = 0
        self.env_cfg = MetaworldEnv(task=task, obs_type=obs_type)
        envs = make_env(
            self.env_cfg,
            n_envs=1,
            use_async_envs=False,
            trust_remote_code=False,
        )
        try:
            self.vec_env = envs[task][0]
        except KeyError as exc:
            available = ", ".join(str(k) for k in envs)
            raise KeyError(f"official_lerobot task {task!r} not found; available tasks: {available}") from exc

    @property
    def inner(self) -> Any:
        return self.vec_env

    @property
    def max_episode_steps(self) -> int:
        return int(self.vec_env.call("_max_episode_steps")[0])

    @property
    def action_dim(self) -> int:
        if hasattr(self.vec_env, "single_action_space"):
            shape = self.vec_env.single_action_space.shape
        else:
            shape = self.vec_env.action_space.shape
            if len(shape) > 1 and int(shape[0]) == 1:
                shape = shape[1:]
        return int(np.prod(shape))

    def reset(self, reset_seed: int) -> dict[str, Any]:
        obs, _info = self.vec_env.reset(seed=[int(reset_seed)])
        return obs

    def build_proc(self, observation: dict[str, Any], *, bundle: Any) -> Any:
        from lerobot.envs.utils import add_envs_task, preprocess_observation

        obs = preprocess_observation(observation)
        obs = add_envs_task(self.vec_env, obs)
        return bundle.preprocessor(obs)

    def step(self, action_batch: np.ndarray) -> OfficialStep:
        action_np = np.asarray(action_batch, dtype=np.float32)
        expected = (1, self.action_dim)
        if action_np.shape != expected:
            raise ValueError(f"official_lerobot vector action must have shape {expected}; got {action_np.shape}")

        obs, reward, terminated, truncated, info = self.vec_env.step(action_np)
        success = False
        if isinstance(info, dict):
            success = bool(_first_success(info.get("final_info")) or _first_success(info) or False)

        return OfficialStep(
            observation=obs,
            reward=float(np.asarray(reward).reshape(-1)[0]),
            terminated=bool(np.asarray(terminated).reshape(-1)[0]),
            truncated=bool(np.asarray(truncated).reshape(-1)[0]),
            success=success,
            info=info if isinstance(info, dict) else {},
        )

    def close(self) -> None:
        self.vec_env.close()


def resolve_lerobot_horizon(
    env: OfficialLeRobotMetaWorldGRPORollout,
    requested_max_steps: int | None,
) -> int:
    if requested_max_steps is None or int(requested_max_steps) <= 0:
        return env.max_episode_steps
    return int(requested_max_steps)
