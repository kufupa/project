from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]


class FakeVectorEnv:
    """Tiny SyncVectorEnv-like fake for official GRPO adapter tests."""

    class FakeSingleActionSpace:
        shape = (4,)

    single_action_space = FakeSingleActionSpace()
    metadata = {"render_fps": 80}

    def __init__(self, *, n_envs: int = 1, step_info: dict | None = None) -> None:
        self.num_envs = int(n_envs)
        self.actions: list[np.ndarray] = []
        self.closed = False
        self.reset_seeds = None
        self.step_info = step_info if step_info is not None else {"final_info": {"is_success": np.array([True])}}

    def reset(self, seed=None):
        self.reset_seeds = seed
        n = self.num_envs
        return {
            "pixels": np.zeros((n, 480, 480, 3), dtype=np.uint8),
            "agent_pos": np.tile(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64), (n, 1)),
        }, {}

    def step(self, action):
        action_np = np.asarray(action, dtype=np.float32)
        self.actions.append(action_np)
        n = self.num_envs
        assert action_np.shape == (n, 4), action_np.shape
        return (
            {
                "pixels": np.ones((n, 480, 480, 3), dtype=np.uint8),
                "agent_pos": np.tile(np.array([[4.0, 3.0, 2.0, 1.0]], dtype=np.float64), (n, 1)),
            },
            np.full((n,), 7.0, dtype=np.float32),
            np.ones((n,), dtype=bool),
            np.zeros((n,), dtype=bool),
            self.step_info,
        )

    def call(self, name):
        if name == "_max_episode_steps":
            return (500,) * self.num_envs
        if name == "task_description":
            return tuple(["assemble the nut onto the peg"] * self.num_envs)
        if name == "task":
            return tuple(["assembly-v3"] * self.num_envs)
        raise KeyError(name)

    def close(self):
        self.closed = True


class FakeDeferredVectorEnv:
    single_action_space = FakeVectorEnv.FakeSingleActionSpace()
    metadata = {"render_fps": 80}

    def __init__(self, fns):
        self.envs = [fn() for fn in fns]
        self.num_envs = len(self.envs)
        self.closed = False
        self.reset_seeds = None

    def reset(self, seed=None):
        self.reset_seeds = seed
        obs_rows = []
        info_rows = []
        for env, item_seed in zip(self.envs, seed, strict=True):
            obs, info = env.reset(seed=item_seed)
            obs_rows.append(obs)
            info_rows.append(info)
        return {
            "pixels": np.stack([obs["pixels"] for obs in obs_rows], axis=0),
            "agent_pos": np.stack([obs["agent_pos"] for obs in obs_rows], axis=0),
        }, {"info": info_rows}

    def step(self, action):
        outs = [env.step(np.asarray(action[i], dtype=np.float32)) for i, env in enumerate(self.envs)]
        obs, rew, term, trunc, info = zip(*outs, strict=True)
        return (
            {
                "pixels": np.stack([o["pixels"] for o in obs], axis=0),
                "agent_pos": np.stack([o["agent_pos"] for o in obs], axis=0),
            },
            np.asarray(rew, dtype=np.float32),
            np.asarray(term, dtype=bool),
            np.asarray(trunc, dtype=bool),
            {"final_info": np.asarray(info, dtype=object)},
        )

    def call(self, name):
        if name == "_max_episode_steps":
            return tuple(getattr(env, "_max_episode_steps", 500) for env in self.envs)
        if name == "task_description":
            return tuple(getattr(env, "task_description", "push the puck to the goal") for env in self.envs)
        if name == "last_agent_pos":
            return tuple(env.last_agent_pos() for env in self.envs)
        return tuple(getattr(env, name)() for env in self.envs)

    def close(self):
        self.closed = True
        for env in self.envs:
            env.close()


def _install_fake_official_lerobot(
    monkeypatch,
    *,
    task: str = "assembly-v3",
    envs_by_task: dict | None = None,
    step_info: dict | None = None,
    n_envs: int = 1,
):
    fake_vec = FakeVectorEnv(n_envs=n_envs, step_info=step_info)

    def fake_make_env(cfg, *, n_envs, use_async_envs, trust_remote_code=False):
        assert getattr(cfg, "type") == "metaworld"
        assert getattr(cfg, "task") == task
        assert getattr(cfg, "obs_type") == "pixels_agent_pos"
        assert n_envs == fake_vec.num_envs
        assert use_async_envs is False
        assert trust_remote_code is False
        return envs_by_task if envs_by_task is not None else {"assembly-v3": {0: fake_vec}}

    class FakeMetaworldEnvConfig:
        type = "metaworld"

        def __init__(self, *, task, obs_type="pixels_agent_pos"):
            self.task = task
            self.obs_type = obs_type

    def fake_preprocess_observation(obs):
        pixels = torch.from_numpy(obs["pixels"]).permute(0, 3, 1, 2).contiguous().float() / 255.0
        state = torch.from_numpy(obs["agent_pos"]).float()
        return {
            "observation.image": pixels,
            "observation.state": state,
        }

    def fake_add_envs_task(env, observation):
        observation = dict(observation)
        td = env.call("task_description")
        observation["task"] = list(td) if isinstance(td, tuple) else td
        return observation

    import lerobot.envs.configs as configs
    import lerobot.envs.factory as factory
    import lerobot.envs.utils as utils

    monkeypatch.setattr(configs, "MetaworldEnv", FakeMetaworldEnvConfig)
    monkeypatch.setattr(factory, "make_env", fake_make_env)
    monkeypatch.setattr(utils, "preprocess_observation", fake_preprocess_observation)
    monkeypatch.setattr(utils, "add_envs_task", fake_add_envs_task)
    return fake_vec


def _install_fake_deferred_deps(monkeypatch):
    class FakeSpaces:
        class Box:
            def __init__(self, *, low, high, shape, dtype):
                self.low = np.full(shape, low, dtype=dtype)
                self.high = np.full(shape, high, dtype=dtype)
                self.shape = tuple(shape)
                self.dtype = dtype

        class Dict(dict):
            pass

    class FakeExpertPolicy:
        def get_action(self, raw_obs):
            return np.asarray(raw_obs[:4], dtype=np.float32) * 0.1

    class FakeInner:
        max_path_length = 500

        def __init__(self, *args, **kwargs):
            self.raw = np.array([1.0, 2.0, 3.0, 4.0, 9.0], dtype=np.float64)
            self.seeded_rand_vec = False
            self.seed_calls: list[int] = []
            self.reset_calls: list[int | None] = []
            self.model = types.SimpleNamespace(
                cam_pos={2: [0.0, 0.0, 0.0]},
                goal_pos=np.array([9.0, 9.0, 9.0], dtype=np.float64),
            )
            self.data = types.SimpleNamespace(
                goal_xpos=np.array([-1.0, -1.0, -1.0], dtype=np.float64),
                forward_calls=0,
            )
            self._target_site_config = [("goal", np.array([1.0, 2.0, 3.0], dtype=np.float64))]
            self.render_goal_snapshots: list[np.ndarray] = []

        def set_task(self, task):
            self.task = task

        def _set_pos_site(self, name, pos):
            assert name == "goal"
            self.model.goal_pos = np.asarray(pos, dtype=np.float64).copy()

        def seed(self, seed):
            self.seed_calls.append(int(seed))
            return [int(seed)]

        def reset(self, seed=None):
            self.reset_calls.append(None if seed is None else int(seed))
            self.raw = np.array([1.0, 2.0, 3.0, 4.0, 9.0], dtype=np.float64)
            return self.raw.copy(), {}

        def step(self, action):
            self.raw = np.asarray(action, dtype=np.float64).reshape(-1)
            self.raw = np.pad(self.raw, (0, max(0, 5 - self.raw.size)), constant_values=0.0)
            return self.raw.copy(), 1.0, False, False, {"success": False}

        def render(self):
            self.render_goal_snapshots.append(self.data.goal_xpos.copy())
            return np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)

        def close(self):
            return None

    class FakeMT1:
        def __init__(self, task, seed=42):
            del seed
            self.train_classes = {task: FakeInner}
            self.train_tasks = [object()]

    import gymnasium
    import gymnasium.vector
    import lerobot.envs as lr_envs

    fake_lr_mw = types.ModuleType("lerobot.envs.metaworld")
    fake_lr_mw.TASK_DESCRIPTIONS = {"push-v3": "push the puck to the goal"}
    fake_lr_mw.TASK_POLICY_MAPPING = {"push-v3": FakeExpertPolicy}

    fake_metaworld = types.ModuleType("metaworld")
    fake_metaworld.MT1 = FakeMT1
    fake_mujoco = types.ModuleType("mujoco")

    def fake_mj_forward(model, data):
        data.forward_calls += 1
        data.goal_xpos = np.asarray(model.goal_pos, dtype=np.float64).copy()

    fake_mujoco.mj_forward = fake_mj_forward

    monkeypatch.setattr(gymnasium, "spaces", FakeSpaces)
    monkeypatch.setattr(gymnasium.vector, "SyncVectorEnv", FakeDeferredVectorEnv)
    monkeypatch.setitem(sys.modules, "lerobot.envs.metaworld", fake_lr_mw)
    monkeypatch.setattr(lr_envs, "metaworld", fake_lr_mw, raising=False)
    monkeypatch.setitem(sys.modules, "metaworld", fake_metaworld)
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)


class IdentityBundle:
    device = "cpu"

    def preprocessor(self, obs):
        return obs


def test_official_adapter_uses_make_env_and_vector_contract(monkeypatch):
    fake_vec = _install_fake_official_lerobot(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    rollout = OfficialLeRobotMetaWorldGRPORollout(task="assembly-v3", reset_randomization_mode="lerobot_default")
    assert rollout.max_episode_steps == 500
    assert rollout.action_dim == 4

    obs = rollout.reset(1000)
    assert fake_vec.reset_seeds == [1000]
    proc = rollout.build_proc(obs, bundle=IdentityBundle())

    assert proc["observation.image"].shape == (1, 3, 480, 480)
    assert proc["observation.state"].shape == (1, 4)
    assert proc["task"] == ["assemble the nut onto the peg"]
    assert "observation.environment_state" not in proc

    step = rollout.step(np.zeros((1, 4), dtype=np.float32))
    assert step.reward == 7.0
    assert step.terminated is True
    assert step.truncated is False
    assert step.success is True
    assert fake_vec.actions[-1].shape == (1, 4)
    rollout.close()
    assert fake_vec.closed is True


def test_official_adapter_reset_many_uses_per_row_seeds(monkeypatch):
    fake_vec = _install_fake_official_lerobot(monkeypatch, n_envs=3)
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    rollout = OfficialLeRobotMetaWorldGRPORollout(
        task="assembly-v3", n_envs=3, reset_randomization_mode="lerobot_default"
    )
    try:
        obs = rollout.reset_many([1000, 1001, 1002])
        assert fake_vec.reset_seeds == [1000, 1001, 1002]
        assert obs["pixels"].shape[0] == 3
    finally:
        rollout.close()


def test_deferred_metaworld_env_stores_raw_obs_for_expert_action(monkeypatch):
    _install_fake_deferred_deps(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import DeferredLeRobotMetaworldEnv

    env = DeferredLeRobotMetaworldEnv(task="push-v3")
    try:
        with pytest.raises(RuntimeError, match="before reset"):
            env.expert_action()
        obs, _info = env.reset(seed=123)
        np.testing.assert_allclose(obs["agent_pos"], np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(env.last_agent_pos(), np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(env.expert_action(), np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
        assert env.render_frame().shape == (8, 8, 3)
    finally:
        env.close()


def test_deferred_metaworld_reset_observation_pixels_are_lerobot_vh(monkeypatch):
    _install_fake_deferred_deps(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import DeferredLeRobotMetaworldEnv

    env = DeferredLeRobotMetaworldEnv(task="push-v3", camera_name="corner2")
    try:
        obs, _info = env.reset(seed=123)
        raw = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)

        np.testing.assert_array_equal(np.asarray(obs["pixels"]), np.flip(raw, (0, 1)))
    finally:
        env.close()


def test_deferred_metaworld_reset_syncs_goal_site_before_render(monkeypatch):
    _install_fake_deferred_deps(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import DeferredLeRobotMetaworldEnv

    env = DeferredLeRobotMetaworldEnv(task="push-v3", camera_name="corner2")
    try:
        env.reset(seed=123)
        inner = env._env  # noqa: SLF001
        np.testing.assert_allclose(inner.model.goal_pos, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(inner.data.goal_xpos, np.array([1.0, 2.0, 3.0]))
        assert inner.data.forward_calls >= 1
        np.testing.assert_allclose(inner.render_goal_snapshots[-1], np.array([1.0, 2.0, 3.0]))
    finally:
        env.close()


def test_deferred_metaworld_reset_forward_works_without_target_site_config(monkeypatch):
    _install_fake_deferred_deps(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import DeferredLeRobotMetaworldEnv

    env = DeferredLeRobotMetaworldEnv(task="push-v3")
    try:
        env.reset(seed=123)
        inner = env._env  # noqa: SLF001
        delattr(inner, "_target_site_config")
        obs, _info = env.reset(seed=124)
        assert obs["pixels"].shape == (8, 8, 3)
        assert inner.data.forward_calls >= 1
    finally:
        env.close()


def test_deferred_metaworld_env_seeds_underlying_random_vector(monkeypatch):
    _install_fake_deferred_deps(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import DeferredLeRobotMetaworldEnv

    env = DeferredLeRobotMetaworldEnv(task="push-v3", reset_randomization_mode="random_seeded")
    try:
        env.reset(seed=2000)
        inner = env._env  # noqa: SLF001
        assert inner.seed_calls[-1] == 2000
        assert inner.reset_calls[-1] == 2000
        assert inner.seeded_rand_vec is True
    finally:
        env.close()


def test_deferred_metaworld_env_defaults_to_random_seeded_reset(monkeypatch):
    _install_fake_deferred_deps(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import DeferredLeRobotMetaworldEnv

    env = DeferredLeRobotMetaworldEnv(task="push-v3")
    try:
        env.reset(seed=2000)
        inner = env._env  # noqa: SLF001
        assert inner._freeze_rand_vec is False
        assert inner.seed_calls[-1] == 2000
        assert inner.seeded_rand_vec is True
    finally:
        env.close()


def test_deferred_metaworld_env_does_not_autoreset_terminal_step(monkeypatch):
    _install_fake_deferred_deps(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import DeferredLeRobotMetaworldEnv

    env = DeferredLeRobotMetaworldEnv(task="push-v3")
    try:
        env.reset(seed=123)
        inner = env._env  # noqa: SLF001

        def terminal_step(action):
            inner.raw = np.asarray(action, dtype=np.float64).reshape(-1)
            inner.raw = np.pad(inner.raw, (0, max(0, 5 - inner.raw.size)), constant_values=0.0)
            return inner.raw.copy(), 1.0, False, False, {"success": True}

        inner.step = terminal_step
        before_reset_calls = list(inner.reset_calls)
        obs, _reward, terminated, _truncated, info = env.step(np.array([9.0, 8.0, 7.0, 6.0], dtype=np.float32))

        assert terminated is True
        assert info["final_info"]["is_success"] is True
        assert inner.reset_calls == before_reset_calls
        np.testing.assert_allclose(obs["agent_pos"], np.array([9.0, 8.0, 7.0, 6.0]))
        np.testing.assert_allclose(env.last_agent_pos(), np.array([9.0, 8.0, 7.0, 6.0]))
    finally:
        env.close()


def test_official_adapter_expert_oracle_uses_deferred_single_env(monkeypatch):
    _install_fake_deferred_deps(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    rollout = OfficialLeRobotMetaWorldGRPORollout(task="push-v3", enable_expert_oracle=True)
    try:
        obs = rollout.reset(77)
        assert obs["pixels"].shape == (1, 8, 8, 3)
        np.testing.assert_allclose(rollout.last_agent_pos(), np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(rollout.expert_action(), np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
        assert rollout.render_frame().shape == (8, 8, 3)
    finally:
        rollout.close()


def test_expert_oracle_rollout_reset_exposes_lerobot_vh_pixels(monkeypatch):
    _install_fake_deferred_deps(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    rollout = OfficialLeRobotMetaWorldGRPORollout(
        task="push-v3",
        n_envs=1,
        enable_expert_oracle=True,
    )
    try:
        obs = rollout.reset(123)
        raw = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        policy_frame = np.asarray(obs["pixels"][0])

        np.testing.assert_array_equal(policy_frame, np.flip(raw, (0, 1)))
    finally:
        rollout.close()


def test_official_adapter_step_batch_matches_n_envs(monkeypatch):
    fin = np.array([{"is_success": True}, {"is_success": False}], dtype=object)
    _install_fake_official_lerobot(monkeypatch, n_envs=2, step_info={"final_info": fin})
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    rollout = OfficialLeRobotMetaWorldGRPORollout(
        task="assembly-v3", n_envs=2, reset_randomization_mode="lerobot_default"
    )
    try:
        rollout.reset(42)
        b = rollout.step_batch(np.zeros((2, 4), dtype=np.float32))
        assert b.reward.shape == (2,)
        assert b.success.tolist() == [True, False]
    finally:
        rollout.close()


def test_official_adapter_defaults_to_random_seeded_reset_for_vector_env(monkeypatch):
    _install_fake_deferred_deps(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    rollout = OfficialLeRobotMetaWorldGRPORollout(task="push-v3", n_envs=3)
    try:
        obs = rollout.reset_many([2000, 2001, 2002])
        assert obs["pixels"].shape[0] == 3
        assert rollout.vec_env.num_envs == 3
        assert [env._env._freeze_rand_vec for env in rollout.vec_env.envs] == [False, False, False]  # noqa: SLF001
        assert [env._env.seed_calls[-1] for env in rollout.vec_env.envs] == [2000, 2001, 2002]  # noqa: SLF001
        assert [env._env.seeded_rand_vec for env in rollout.vec_env.envs] == [True, True, True]  # noqa: SLF001
    finally:
        rollout.close()


def test_official_adapter_rejects_step_when_n_envs_gt_1(monkeypatch):
    _install_fake_official_lerobot(monkeypatch, n_envs=2)
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    rollout = OfficialLeRobotMetaWorldGRPORollout(
        task="assembly-v3", n_envs=2, reset_randomization_mode="lerobot_default"
    )
    try:
        rollout.reset(0)
        try:
            rollout.step(np.zeros((1, 4), dtype=np.float32))
        except ValueError as exc:
            assert "step_batch" in str(exc)
        else:
            raise AssertionError("expected ValueError")
    finally:
        rollout.close()


def test_official_adapter_rejects_non_batched_vector_action(monkeypatch):
    _install_fake_official_lerobot(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    rollout = OfficialLeRobotMetaWorldGRPORollout(task="assembly-v3", reset_randomization_mode="lerobot_default")
    try:
        try:
            rollout.step(np.zeros(4, dtype=np.float32))
        except ValueError as exc:
            assert "shape (1, 4)" in str(exc)
        else:
            raise AssertionError("expected ValueError for non-batched action")
    finally:
        rollout.close()


def test_official_adapter_rejects_missing_task_without_fallback(monkeypatch):
    _install_fake_official_lerobot(
        monkeypatch,
        task="missing-v3",
        envs_by_task={"other-v3": {0: FakeVectorEnv()}},
    )
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    with pytest.raises(KeyError, match="missing-v3"):
        OfficialLeRobotMetaWorldGRPORollout(task="missing-v3", reset_randomization_mode="lerobot_default")


def test_official_adapter_reads_vector_final_info_success(monkeypatch):
    _install_fake_official_lerobot(
        monkeypatch,
        step_info={"final_info": np.array([{"is_success": True}], dtype=object)},
    )
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    rollout = OfficialLeRobotMetaWorldGRPORollout(task="assembly-v3", reset_randomization_mode="lerobot_default")
    try:
        step = rollout.step(np.zeros((1, 4), dtype=np.float32))
        assert step.success is True
    finally:
        rollout.close()


def test_phase11_rollout_exposes_official_backend_static():
    text = (_REPO / "src" / "smolvla_grpo" / "phase11_rollout.py").read_text(encoding="utf-8")
    assert "env_backend" in text
    assert "rollout_execution" in text
    assert "collect_official_lerobot_vector_rollout_group" in text
    assert "OfficialLeRobotMetaWorldGRPORollout" in text
    assert "resolve_lerobot_horizon" in text
    assert "env_h.build_proc" in text
    assert "chunk_exec_actions" in text or "step.exec_action_np.reshape(1, -1)" in text
    assert 'raise ValueError("custom env_backend requires max_steps >= 1")' in text
    assert 'policy_reset = getattr(policy_old, "reset", None)' in text


def test_grpo_scripts_expose_official_lerobot_backend_and_success_logging():
    train = (_REPO / "scripts" / "grpo" / "train_phase11_env_on_policy_grpo.py").read_text(encoding="utf-8")
    smoke = (_REPO / "scripts" / "grpo" / "smoke_phase11_rollout.py").read_text(encoding="utf-8")
    forward = (_REPO / "scripts" / "grpo" / "check_smolvla_grpo_forward.py").read_text(encoding="utf-8")
    for text in (train, smoke, forward):
        assert "--env-backend" in text
        assert "official_lerobot" in text
    assert "--rollout-execution" in train
    assert "--rollout-execution" in smoke
    assert "--action-transform" in train
    assert "--run-label" in train
    assert "env_backend=args.env_backend" in train
    assert "success_rate" in train
    assert "successes" in train
    assert "resolved_max_steps" in train
    assert "episode_lengths" in train
    assert "num_env_steps" in train
    assert "rollout_seconds" in train
    assert "optimize_seconds" in train
    assert "update_seconds" in train
    assert "phase111_grpo_update" in train
    assert "action_clip_fraction" in train
    assert "action_clip_any_fraction" in train
    assert "terminated" in train
    assert "truncated" in train
    rollout = (_REPO / "src" / "smolvla_grpo" / "phase11_rollout.py").read_text(encoding="utf-8")
    assert "if success or terminated or truncated:" in rollout
    assert "env_backend=args.env_backend" in smoke
    assert "env_h.build_proc" in forward


def test_grpo_eval_can_write_official_eval_info():
    text = (_REPO / "scripts" / "grpo" / "eval_phase11_checkpoints.py").read_text(encoding="utf-8")
    assert "--env-backend" in text
    assert "official_lerobot" in text
    assert "--save-official-eval-info" in text
    assert "eval_info.json" in text
    assert "per_task" in text
    assert "per_group" in text
    assert "overall" in text
    assert "pc_success" in text


def test_official_eval_path_matches_rollout_semantics_static():
    text = (_REPO / "scripts" / "grpo" / "eval_phase11_checkpoints.py").read_text(encoding="utf-8")
    assert "_coerce_exec_action" in text
    assert "_move_proc_to_device" in text
    assert "policy_reset = getattr(bundle.policy, \"reset\", None)" in text
    assert "if callable(policy_reset):" in text
    assert "env_step.success or env_step.terminated or env_step.truncated" in text


def test_skipped_grpo_updates_preserve_numbered_checkpoints_static():
    text = (_REPO / "scripts" / "grpo" / "train_phase11_env_on_policy_grpo.py").read_text(encoding="utf-8")
    skipped_branch = text.split('"reason": "zero_advantages"', 1)[1].split("continue", 1)[0]
    assert 'persist_checkpoint(f"update_{update + 1:04d}.pt"' in skipped_branch


def test_phase111_slurm_smoke_uses_official_backend_and_export_nil():
    path = _REPO / "scripts" / "grpo" / "submit_phase111_on_grpo_lerobot_smoke.slurm"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --export=NIL" in text
    assert "scripts/slurm/common_env.sh" in text
    assert "--env-backend official_lerobot" in text
    assert 'MAX_STEPS="${GRPO_PHASE111_MAX_STEPS:-0}"' in text
    assert '--max-steps "${MAX_STEPS}"' in text
    assert "phase111_on_grpo_lerobot_smoke" in text
    assert "PHASE111_GRPO_LEROBOT_ARTIFACTS_OK" in text
    assert "PHASE111_GRPO_LEROBOT_SMOKE_OK" in text
    subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO))
