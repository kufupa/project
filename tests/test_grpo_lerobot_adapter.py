from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]


class FakeVectorEnv:
    """Tiny SyncVectorEnv-like fake for official GRPO adapter tests."""

    class FakeActionSpace:
        shape = (1, 4)

    num_envs = 1
    action_space = FakeActionSpace()
    metadata = {"render_fps": 80}

    def __init__(self, *, step_info: dict | None = None) -> None:
        self.actions: list[np.ndarray] = []
        self.closed = False
        self.reset_seeds = None
        self.step_info = step_info if step_info is not None else {"final_info": {"is_success": np.array([True])}}

    def reset(self, seed=None):
        self.reset_seeds = seed
        return {
            "pixels": np.zeros((1, 480, 480, 3), dtype=np.uint8),
            "agent_pos": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64),
        }, {}

    def step(self, action):
        action_np = np.asarray(action, dtype=np.float32)
        self.actions.append(action_np)
        return (
            {
                "pixels": np.ones((1, 480, 480, 3), dtype=np.uint8),
                "agent_pos": np.array([[4.0, 3.0, 2.0, 1.0]], dtype=np.float64),
            },
            np.array([7.0], dtype=np.float32),
            np.array([True]),
            np.array([False]),
            self.step_info,
        )

    def call(self, name):
        if name == "_max_episode_steps":
            return [500]
        if name == "task_description":
            return ["assemble the nut onto the peg"]
        if name == "task":
            return ["assembly-v3"]
        raise KeyError(name)

    def close(self):
        self.closed = True


def _install_fake_official_lerobot(
    monkeypatch,
    *,
    task: str = "assembly-v3",
    envs_by_task: dict | None = None,
    step_info: dict | None = None,
):
    fake_vec = FakeVectorEnv(step_info=step_info)

    def fake_make_env(cfg, *, n_envs, use_async_envs, trust_remote_code=False):
        assert getattr(cfg, "type") == "metaworld"
        assert getattr(cfg, "task") == task
        assert getattr(cfg, "obs_type") == "pixels_agent_pos"
        assert n_envs == 1
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
        observation["task"] = env.call("task_description")
        return observation

    import lerobot.envs.configs as configs
    import lerobot.envs.factory as factory
    import lerobot.envs.utils as utils

    monkeypatch.setattr(configs, "MetaworldEnv", FakeMetaworldEnvConfig)
    monkeypatch.setattr(factory, "make_env", fake_make_env)
    monkeypatch.setattr(utils, "preprocess_observation", fake_preprocess_observation)
    monkeypatch.setattr(utils, "add_envs_task", fake_add_envs_task)
    return fake_vec


class IdentityBundle:
    device = "cpu"

    def preprocessor(self, obs):
        return obs


def test_official_adapter_uses_make_env_and_vector_contract(monkeypatch):
    fake_vec = _install_fake_official_lerobot(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    rollout = OfficialLeRobotMetaWorldGRPORollout(task="assembly-v3")
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


def test_official_adapter_rejects_non_batched_vector_action(monkeypatch):
    _install_fake_official_lerobot(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    rollout = OfficialLeRobotMetaWorldGRPORollout(task="assembly-v3")
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
        OfficialLeRobotMetaWorldGRPORollout(task="missing-v3")


def test_official_adapter_reads_vector_final_info_success(monkeypatch):
    _install_fake_official_lerobot(
        monkeypatch,
        step_info={"final_info": np.array([{"is_success": True}], dtype=object)},
    )
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    rollout = OfficialLeRobotMetaWorldGRPORollout(task="assembly-v3")
    try:
        step = rollout.step(np.zeros((1, 4), dtype=np.float32))
        assert step.success is True
    finally:
        rollout.close()


def test_phase11_rollout_exposes_official_backend_static():
    text = (_REPO / "src" / "smolvla_grpo" / "phase11_rollout.py").read_text(encoding="utf-8")
    assert "env_backend" in text
    assert "OfficialLeRobotMetaWorldGRPORollout" in text
    assert "resolve_lerobot_horizon" in text
    assert "env_h.build_proc" in text
    assert "step.exec_action_np.reshape(1, -1)" in text
    assert 'raise ValueError("custom env_backend requires max_steps >= 1")' in text
    assert 'policy_reset = getattr(policy_old, "reset", None)' in text


def test_grpo_scripts_expose_official_lerobot_backend_and_success_logging():
    train = (_REPO / "scripts" / "grpo" / "train_phase11_env_on_policy_grpo.py").read_text(encoding="utf-8")
    smoke = (_REPO / "scripts" / "grpo" / "smoke_phase11_rollout.py").read_text(encoding="utf-8")
    forward = (_REPO / "scripts" / "grpo" / "check_smolvla_grpo_forward.py").read_text(encoding="utf-8")
    for text in (train, smoke, forward):
        assert "--env-backend" in text
        assert "official_lerobot" in text
    assert "env_backend=args.env_backend" in train
    assert "success_rate" in train
    assert "successes" in train
    assert "resolved_max_steps" in train
    assert "episode_lengths" in train
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
    assert 'ckpt_dir / f"update_{update + 1:04d}.pt"' in skipped_branch


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
