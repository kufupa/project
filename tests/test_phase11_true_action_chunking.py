from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from smolvla_grpo import phase11_rollout
from smolvla_grpo import official_lerobot_vector_rollout as vector_rollout

_REPO = Path(__file__).resolve().parents[1]


class FakeBundle:
    device = torch.device("cpu")
    policy = SimpleNamespace()


class FakeActionSpace:
    shape = (4,)
    low = np.full((4,), -1.0, dtype=np.float32)
    high = np.full((4,), 1.0, dtype=np.float32)


class FakeCustomEnv:
    instances: list["FakeCustomEnv"] = []
    success_on_step: int | None = None

    def __init__(self, *, task: str) -> None:
        self.inner = self
        self.action_space = FakeActionSpace()
        self.step_count = 0
        self.actions: list[np.ndarray] = []
        FakeCustomEnv.instances.append(self)

    def set_task_for_episode(self, episode_index: int) -> None:
        return None

    def reset(self, reset_seed: int) -> dict[str, int]:
        self.step_count = 0
        return {"step": 0}

    def step(self, action: np.ndarray):
        self.step_count += 1
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        success = self.success_on_step is not None and self.step_count >= int(self.success_on_step)
        return {"step": self.step_count}, 1.0, False, False, {"success": success}

    def close(self) -> None:
        return None


class FakeOfficialEnv:
    instances: list["FakeOfficialEnv"] = []
    action_dim = 4

    def __init__(self, *, task: str) -> None:
        self.n_envs = 1
        self.step_count = 0
        self.actions: list[np.ndarray] = []
        self.single_action_space = FakeActionSpace()
        FakeOfficialEnv.instances.append(self)

    def reset(self, reset_seed: int) -> dict[str, int]:
        self.step_count = 0
        return {"step": 0}

    def build_proc(self, obs, *, bundle):
        return {"obs": torch.tensor([[float(obs["step"])]])}

    def step(self, action_batch: np.ndarray):
        self.step_count += 1
        self.actions.append(np.asarray(action_batch, dtype=np.float32).reshape(-1))
        return SimpleNamespace(
            observation={"step": self.step_count},
            reward=1.0,
            terminated=False,
            truncated=False,
            success=False,
        )

    def close(self) -> None:
        return None


class FakeVectorOfficialEnv:
    instances: list["FakeVectorOfficialEnv"] = []
    action_dim = 4
    success_row_after_step: dict[int, int] = {}

    def __init__(
        self,
        *,
        task: str,
        n_envs: int,
        use_async_envs: bool,
        async_start_method: str = "forkserver",
    ) -> None:
        self.n_envs = int(n_envs)
        self.step_count = 0
        self.actions: list[np.ndarray] = []
        self.single_action_space = FakeActionSpace()
        FakeVectorOfficialEnv.instances.append(self)

    def reset(self, reset_seed: int):
        self.step_count = 0
        return {"step": np.zeros((self.n_envs,), dtype=np.int64)}

    def build_proc(self, obs, *, bundle):
        step = np.asarray(obs["step"], dtype=np.float32).reshape(self.n_envs)
        return {
            "obs": torch.as_tensor(step, dtype=torch.float32).reshape(self.n_envs, 1),
            "task": [f"row-{idx}" for idx in range(self.n_envs)],
            "static": {"copied": True},
        }

    def step_batch(self, action_matrix: np.ndarray):
        self.step_count += 1
        action_matrix = np.asarray(action_matrix, dtype=np.float32).reshape(self.n_envs, 4)
        self.actions.append(action_matrix.copy())
        success = np.zeros((self.n_envs,), dtype=np.bool_)
        for row, done_step in self.success_row_after_step.items():
            if self.step_count >= int(done_step):
                success[int(row)] = True
        return SimpleNamespace(
            observation={"step": np.full((self.n_envs,), self.step_count, dtype=np.int64)},
            reward=np.ones((self.n_envs,), dtype=np.float32),
            terminated=np.zeros((self.n_envs,), dtype=np.bool_),
            truncated=np.zeros((self.n_envs,), dtype=np.bool_),
            success=success,
        )

    def close(self) -> None:
        return None


class FakePolicyWrapper:
    instances: list["FakePolicyWrapper"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.chunk_calls: list[int] = []
        self.single_calls = 0
        self.build_proc_calls = 0
        FakePolicyWrapper.instances.append(self)

    def eval(self) -> None:
        return None

    def build_proc_batch(self, obs, env):
        self.build_proc_calls += 1
        return {"obs": torch.tensor([[float(obs["step"])]])}

    def sample_action_from_proc(self, proc, *, rng=None):
        self.single_calls += 1
        idx = float(self.single_calls)
        action = np.full((4,), idx, dtype=np.float32)
        return SimpleNamespace(
            exec_action_np=action,
            unsquashed=torch.full((1, 4), idx, dtype=torch.float32),
            log_prob=torch.tensor([idx], dtype=torch.float32),
            action_clip_fraction=0.0,
            action_clip_any=False,
        )

    def sample_action_chunk_from_proc(self, proc, *, chunk_len: int, rng=None):
        self.chunk_calls.append(int(chunk_len))
        base = float(len(self.chunk_calls) * 10)
        actions = np.stack(
            [np.full((4,), base + float(i), dtype=np.float32) for i in range(int(chunk_len))],
            axis=0,
        )
        unsquashed = torch.as_tensor(actions, dtype=torch.float32)
        log_probs = torch.arange(int(chunk_len), dtype=torch.float32) + base
        return SimpleNamespace(
            exec_action_np=actions,
            unsquashed_chunk=unsquashed,
            log_prob_steps=log_probs,
            action_clip_fraction=np.zeros((int(chunk_len),), dtype=np.float64),
            action_clip_any=np.zeros((int(chunk_len),), dtype=np.bool_),
        )


class FakeVectorPolicyWrapper:
    instances: list["FakeVectorPolicyWrapper"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.batch_calls: list[tuple[int, int]] = []
        self.chunk_calls: list[tuple[int, int]] = []
        FakeVectorPolicyWrapper.instances.append(self)

    def eval(self) -> None:
        return None

    def sample_action_batch_from_proc(self, proc, *, n_envs: int, rngs=None):
        self.batch_calls.append((int(n_envs), 1))
        actions = np.stack(
            [np.full((4,), float(row + 1), dtype=np.float32) for row in range(int(n_envs))],
            axis=0,
        )
        tensor = torch.as_tensor(actions, dtype=torch.float32)
        log_prob = torch.arange(int(n_envs), dtype=torch.float32)
        return SimpleNamespace(
            exec_action_np=actions,
            raw_postprocessed_action_np=actions.copy(),
            policy_tensor=tensor,
            unsquashed=tensor,
            log_prob=log_prob,
            action_clip_fraction=np.zeros((int(n_envs),), dtype=np.float64),
            action_clip_any=np.zeros((int(n_envs),), dtype=np.bool_),
        )

    def sample_action_chunk_batch_from_proc(self, proc, *, n_envs: int, chunk_len: int, rngs=None):
        self.chunk_calls.append((int(n_envs), int(chunk_len)))
        actions = np.zeros((int(n_envs), int(chunk_len), 4), dtype=np.float32)
        for row in range(int(n_envs)):
            for step in range(int(chunk_len)):
                actions[row, step] = float(len(self.chunk_calls) * 100 + row * 10 + step)
        tensor = torch.as_tensor(actions, dtype=torch.float32)
        log_prob_steps = torch.arange(int(n_envs) * int(chunk_len), dtype=torch.float32).reshape(
            int(n_envs), int(chunk_len)
        )
        return SimpleNamespace(
            exec_action_np=actions,
            raw_postprocessed_action_np=actions.copy(),
            policy_tensor=tensor,
            unsquashed_chunk=tensor,
            log_prob_steps=log_prob_steps,
            log_prob_sum=log_prob_steps.sum(dim=1),
            action_clip_fraction=np.zeros((int(n_envs), int(chunk_len)), dtype=np.float64),
            action_clip_any=np.zeros((int(n_envs), int(chunk_len)), dtype=np.bool_),
        )


def _install_common_fakes(monkeypatch, env_cls) -> None:
    FakePolicyWrapper.instances.clear()
    FakeCustomEnv.instances.clear()
    FakeCustomEnv.success_on_step = None
    FakeOfficialEnv.instances.clear()
    monkeypatch.setattr(phase11_rollout, "MetaWorldSmolVLAGRPOPolicy", FakePolicyWrapper)
    monkeypatch.setattr(phase11_rollout, "seed_metaworld_process", lambda seed: None)
    monkeypatch.setattr(phase11_rollout, "_resolve_camera_name", lambda: "corner2")
    monkeypatch.setattr(phase11_rollout, "_resolve_flip_corner2", lambda: False)
    if env_cls is FakeCustomEnv:
        monkeypatch.setattr(phase11_rollout, "PushV3GRPOEnv", FakeCustomEnv)
    else:
        monkeypatch.setattr(phase11_rollout, "OfficialLeRobotMetaWorldGRPORollout", FakeOfficialEnv)
        monkeypatch.setattr(phase11_rollout, "resolve_lerobot_horizon", lambda env, max_steps: int(max_steps))


def _install_vector_fakes(monkeypatch) -> None:
    FakeVectorPolicyWrapper.instances.clear()
    FakeVectorOfficialEnv.instances.clear()
    FakeVectorOfficialEnv.success_row_after_step = {}
    monkeypatch.setattr(vector_rollout, "OfficialLeRobotMetaWorldGRPORollout", FakeVectorOfficialEnv)
    monkeypatch.setattr(vector_rollout, "MetaWorldSmolVLAGRPOPolicy", FakeVectorPolicyWrapper)
    monkeypatch.setattr(vector_rollout, "resolve_lerobot_horizon", lambda env, max_steps: int(max_steps))
    monkeypatch.setattr(vector_rollout, "_resolve_camera_name", lambda: "corner2")
    monkeypatch.setattr(vector_rollout, "_resolve_flip_corner2", lambda: False)


def test_phase11_serial_custom_action_chunk_samples_once_per_chunk(monkeypatch):
    _install_common_fakes(monkeypatch, FakeCustomEnv)

    rollouts = phase11_rollout.collect_rollout_group(
        bundle=FakeBundle(),
        policy_old=SimpleNamespace(),
        task="push-v3",
        task_text="push",
        reset_seed=1000,
        episode_index=0,
        max_steps=5,
        group_size=1,
        action_dim=4,
        device=torch.device("cpu"),
        env_backend="custom",
        action_chunk_size=2,
    )

    wrapper = FakePolicyWrapper.instances[-1]
    env = FakeCustomEnv.instances[-1]
    traj = rollouts[0]
    assert wrapper.chunk_calls == [2, 2, 1]
    assert env.step_count == 5
    assert len(traj.exec_actions) == 5
    assert [chunk.executed_steps for chunk in traj.action_chunks] == [2, 2, 1]
    assert traj.metadata["policy_sample_calls"] == 3
    assert traj.metadata["action_chunk_size"] == 2


def test_phase11_serial_official_action_chunk_samples_once_per_chunk(monkeypatch):
    _install_common_fakes(monkeypatch, FakeOfficialEnv)

    rollouts = phase11_rollout.collect_rollout_group(
        bundle=FakeBundle(),
        policy_old=SimpleNamespace(),
        task="push-v3",
        task_text="push",
        reset_seed=1000,
        episode_index=0,
        max_steps=5,
        group_size=1,
        action_dim=4,
        device=torch.device("cpu"),
        env_backend="official_lerobot",
        action_chunk_size=2,
    )

    wrapper = FakePolicyWrapper.instances[-1]
    env = FakeOfficialEnv.instances[-1]
    traj = rollouts[0]
    assert wrapper.chunk_calls == [2, 2, 1]
    assert env.step_count == 5
    assert len(traj.exec_actions) == 5
    assert [chunk.executed_steps for chunk in traj.action_chunks] == [2, 2, 1]
    assert traj.metadata["policy_sample_calls"] == 3
    assert traj.metadata["action_chunk_size"] == 2


def test_phase11_action_chunk_size_one_records_degenerate_chunks(monkeypatch):
    _install_common_fakes(monkeypatch, FakeCustomEnv)

    rollouts = phase11_rollout.collect_rollout_group(
        bundle=FakeBundle(),
        policy_old=SimpleNamespace(),
        task="push-v3",
        task_text="push",
        reset_seed=1000,
        episode_index=0,
        max_steps=5,
        group_size=1,
        action_dim=4,
        device=torch.device("cpu"),
        env_backend="custom",
        action_chunk_size=1,
    )

    wrapper = FakePolicyWrapper.instances[-1]
    traj = rollouts[0]
    assert wrapper.single_calls == 5
    assert wrapper.chunk_calls == []
    assert len(traj.action_chunks) == 5
    assert [chunk.executed_steps for chunk in traj.action_chunks] == [1, 1, 1, 1, 1]
    assert [chunk.logprob_mode for chunk in traj.action_chunks] == ["step"] * 5
    assert len(traj.proc_snapshots) == 5
    assert len(traj.log_probs) == 5
    assert traj.metadata["policy_sample_calls"] == 5
    assert traj.metadata["action_chunk_size"] == 1


def test_phase11_vector_action_chunk_size_gt_one_collects(monkeypatch):
    _install_vector_fakes(monkeypatch)

    rollouts = phase11_rollout.collect_rollout_group(
        bundle=FakeBundle(),
        policy_old=SimpleNamespace(),
        task="push-v3",
        task_text="push",
        reset_seed=1000,
        episode_index=0,
        max_steps=3,
        group_size=1,
        action_dim=4,
        device=torch.device("cpu"),
        env_backend="official_lerobot",
        rollout_execution="vector_sync",
        action_chunk_size=2,
    )

    traj = rollouts[0]
    wrapper = FakeVectorPolicyWrapper.instances[-1]
    assert wrapper.chunk_calls == [(1, 2), (1, 1)]
    assert len(traj.rewards) == 3
    assert [chunk.executed_steps for chunk in traj.action_chunks] == [2, 1]
    assert [chunk.logprob_mode for chunk in traj.action_chunks] == ["chunk", "chunk"]
    assert traj.metadata["action_chunk_size"] == 2
    assert traj.metadata["policy_sample_calls"] == 2


def test_phase11_vector_action_chunk_samples_active_rows_once_per_chunk(monkeypatch):
    _install_vector_fakes(monkeypatch)

    rollouts = phase11_rollout.collect_rollout_group(
        bundle=FakeBundle(),
        policy_old=SimpleNamespace(),
        task="push-v3",
        task_text="push",
        reset_seed=1000,
        episode_index=0,
        max_steps=5,
        group_size=3,
        action_dim=4,
        device=torch.device("cpu"),
        env_backend="official_lerobot",
        rollout_execution="vector_sync",
        action_chunk_size=2,
    )

    wrapper = FakeVectorPolicyWrapper.instances[-1]
    assert wrapper.chunk_calls == [(3, 2), (3, 2), (3, 1)]
    assert [len(traj.rewards) for traj in rollouts] == [5, 5, 5]
    assert [len(traj.action_chunks) for traj in rollouts] == [3, 3, 3]
    assert [traj.metadata["policy_sample_calls"] for traj in rollouts] == [3, 3, 3]
    assert [[chunk.executed_steps for chunk in traj.action_chunks] for traj in rollouts] == [
        [2, 2, 1],
        [2, 2, 1],
        [2, 2, 1],
    ]


def test_phase11_vector_rollout_microbatches_policy_forward(monkeypatch):
    _install_vector_fakes(monkeypatch)
    rollouts = phase11_rollout.collect_rollout_group(
        bundle=FakeBundle(),
        policy_old=SimpleNamespace(),
        task="push-v3",
        task_text="push",
        reset_seed=1000,
        episode_index=0,
        max_steps=5,
        group_size=6,
        action_dim=4,
        device=torch.device("cpu"),
        env_backend="official_lerobot",
        rollout_execution="vector_sync",
        rollout_policy_batch_size=2,
        action_chunk_size=2,
    )
    wrapper = FakeVectorPolicyWrapper.instances[-1]
    assert wrapper.chunk_calls == [
        (2, 2),
        (2, 2),
        (2, 2),
        (2, 2),
        (2, 2),
        (2, 2),
        (2, 1),
        (2, 1),
        (2, 1),
    ]
    assert [len(tr.rewards) for tr in rollouts] == [5, 5, 5, 5, 5, 5]
    assert {tr.metadata["rollout_policy_batch_size"] for tr in rollouts} == {2}


def test_phase11_vector_chunk_stops_sampling_done_rows(monkeypatch):
    _install_vector_fakes(monkeypatch)
    FakeVectorOfficialEnv.success_row_after_step = {1: 1}

    rollouts = phase11_rollout.collect_rollout_group(
        bundle=FakeBundle(),
        policy_old=SimpleNamespace(),
        task="push-v3",
        task_text="push",
        reset_seed=1000,
        episode_index=0,
        max_steps=5,
        group_size=3,
        action_dim=4,
        device=torch.device("cpu"),
        env_backend="official_lerobot",
        rollout_execution="vector_sync",
        action_chunk_size=2,
    )

    wrapper = FakeVectorPolicyWrapper.instances[-1]
    assert wrapper.chunk_calls == [(3, 2), (2, 2), (2, 1)]
    assert [len(traj.rewards) for traj in rollouts] == [5, 1, 5]
    assert [len(traj.action_chunks) for traj in rollouts] == [3, 1, 3]
    assert [traj.metadata["policy_sample_calls"] for traj in rollouts] == [3, 1, 3]
    assert [chunk.executed_steps for chunk in rollouts[1].action_chunks] == [1]


def test_phase11_serial_custom_chunk_records_executed_prefix_on_early_success(monkeypatch):
    _install_common_fakes(monkeypatch, FakeCustomEnv)
    FakeCustomEnv.success_on_step = 2

    rollouts = phase11_rollout.collect_rollout_group(
        bundle=FakeBundle(),
        policy_old=SimpleNamespace(),
        task="push-v3",
        task_text="push",
        reset_seed=1000,
        episode_index=0,
        max_steps=5,
        group_size=1,
        action_dim=4,
        device=torch.device("cpu"),
        env_backend="custom",
        action_chunk_size=5,
    )

    traj = rollouts[0]
    assert len(traj.rewards) == 2
    assert len(traj.action_chunks) == 1
    chunk = traj.action_chunks[0]
    assert chunk.executed_steps == 2
    assert tuple(chunk.unsquashed_chunk.shape) == (2, 4)
    assert tuple(chunk.log_prob_steps.shape) == (2,)
    assert chunk.start_step == 0


def _load_phase11_train_module():
    path = _REPO / "scripts" / "grpo" / "train_phase11_env_on_policy_grpo.py"
    spec = importlib.util.spec_from_file_location("train_phase11_env_on_policy_grpo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeChunkTrainWrapper:
    def __init__(self, init: float = -0.1) -> None:
        self.scale = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32))
        self.calls: list[tuple[str, tuple[int, ...]]] = []
        self.step_calls: list[tuple[tuple[str, ...], tuple[int, ...]]] = []
        self.batch_calls: list[tuple[tuple[str, ...], tuple[int, ...]]] = []
        self.step_batch_calls: list[tuple[tuple[str, ...], tuple[int, ...]]] = []

    def get_action_probs_for_chunk_from_proc(self, proc_snapshot, unsquashed_chunk):
        self.calls.append((proc_snapshot["id"], tuple(unsquashed_chunk.shape)))
        return self.scale.expand(int(unsquashed_chunk.shape[0])) / float(unsquashed_chunk.shape[0])

    def get_action_probs_from_proc_list(self, proc_snapshots, unsquashed_actions):
        self.step_calls.append(
            (tuple(proc_snapshot["id"] for proc_snapshot in proc_snapshots), tuple(unsquashed_actions.shape))
        )
        return self.scale.expand(int(unsquashed_actions.shape[0]))

    def get_action_probs_for_chunk_batch_from_proc_list(self, proc_snapshots, unsquashed_chunks):
        self.batch_calls.append(
            (tuple(proc_snapshot["id"] for proc_snapshot in proc_snapshots), tuple(unsquashed_chunks.shape))
        )
        b = int(unsquashed_chunks.shape[0])
        t = int(unsquashed_chunks.shape[1])
        return self.scale.expand(b, t) / float(t)

    def get_action_probs_step_batch_from_proc_list(self, proc_snapshots, unsquashed_actions):
        self.step_batch_calls.append(
            (tuple(proc_snapshot["id"] for proc_snapshot in proc_snapshots), tuple(unsquashed_actions.shape))
        )
        return self.scale.expand(int(unsquashed_actions.shape[0]))


def _action_chunk(proc_id: str, steps: int = 5, logprob_mode: str = "chunk"):
    return phase11_rollout.RolloutActionChunk(
        proc_snapshot={"id": proc_id},
        unsquashed_chunk=torch.zeros((steps, 4), dtype=torch.float32),
        log_prob_steps=torch.zeros(steps, dtype=torch.float32),
        log_prob_sum=torch.tensor(0.0, dtype=torch.float64),
        start_step=0,
        executed_steps=steps,
        logprob_mode=logprob_mode,
    )


def test_phase11_training_loss_recomputes_root_conditioned_chunk_logprob():
    train = _load_phase11_train_module()
    wrapper = FakeChunkTrainWrapper()
    traj = phase11_rollout.RolloutTrajectory(reset_seed=1000, rollout_index=0)
    traj.action_chunks.append(_action_chunk("root", steps=5))

    count = train._backward_phase11_chunk_loss(
        train_wrapper=wrapper,
        action_chunks=traj.action_chunks,
        advantage=torch.tensor(1.0, dtype=torch.float32),
        device=torch.device("cpu"),
        optimizer_chunk_size=2,
        clip_eps=0.2,
        normalizer=1,
    )

    assert count == 1
    assert wrapper.calls == []
    assert wrapper.batch_calls == [(("root",), (1, 5, 4))]
    assert wrapper.step_calls == []
    assert wrapper.scale.grad is not None
    assert wrapper.scale.grad.abs().item() > 0.0


def test_phase11_training_loss_step_mode_uses_legacy_step_recompute():
    train = _load_phase11_train_module()
    wrapper = FakeChunkTrainWrapper()

    count = train._backward_phase11_chunk_loss(
        train_wrapper=wrapper,
        action_chunks=[_action_chunk("root", steps=1, logprob_mode="step")],
        advantage=torch.tensor(1.0, dtype=torch.float32),
        device=torch.device("cpu"),
        optimizer_chunk_size=2,
        clip_eps=0.2,
        normalizer=1,
    )

    assert count == 1
    assert wrapper.calls == []
    assert wrapper.step_calls == []
    assert wrapper.step_batch_calls == [(("root",), (1, 4))]
    assert wrapper.scale.grad is not None
    assert wrapper.scale.grad.abs().item() > 0.0


def test_phase11_training_loss_chunk_mode_single_row_uses_chunk_recompute():
    train = _load_phase11_train_module()
    wrapper = FakeChunkTrainWrapper()

    count = train._backward_phase11_chunk_loss(
        train_wrapper=wrapper,
        action_chunks=[_action_chunk("root", steps=1, logprob_mode="chunk")],
        advantage=torch.tensor(1.0, dtype=torch.float32),
        device=torch.device("cpu"),
        optimizer_chunk_size=2,
        clip_eps=0.2,
        normalizer=1,
    )

    assert count == 1
    assert wrapper.calls == []
    assert wrapper.batch_calls == [(("root",), (1, 1, 4))]
    assert wrapper.step_calls == []
    assert wrapper.scale.grad is not None
    assert wrapper.scale.grad.abs().item() > 0.0


def test_phase11_training_loss_normalizer_scales_gradient():
    train = _load_phase11_train_module()
    chunks = [_action_chunk("root-a", steps=5), _action_chunk("root-b", steps=5)]

    wrapper_one = FakeChunkTrainWrapper()
    count = train._backward_phase11_chunk_loss(
        train_wrapper=wrapper_one,
        action_chunks=chunks,
        advantage=torch.tensor(1.0, dtype=torch.float64),
        device=torch.device("cpu"),
        optimizer_chunk_size=1,
        clip_eps=0.2,
        normalizer=2,
    )

    wrapper_two = FakeChunkTrainWrapper()
    train._backward_phase11_chunk_loss(
        train_wrapper=wrapper_two,
        action_chunks=chunks,
        advantage=torch.tensor(1.0, dtype=torch.float64),
        device=torch.device("cpu"),
        optimizer_chunk_size=1,
        clip_eps=0.2,
        normalizer=4,
    )

    assert count == 2
    assert wrapper_one.calls == []
    assert wrapper_one.batch_calls == [(("root-a",), (1, 5, 4)), (("root-b",), (1, 5, 4))]
    torch.testing.assert_close(wrapper_one.scale.grad, torch.tensor(-0.9048374))
    torch.testing.assert_close(wrapper_two.scale.grad, wrapper_one.scale.grad / 2.0)


def test_phase11_group_loss_batches_across_trajectories_without_global_denominator():
    train = _load_phase11_train_module()
    chunks_a = [_action_chunk("a0", steps=5)]
    chunks_b = [_action_chunk("b0", steps=5), _action_chunk("b1", steps=5)]
    rollouts = [
        SimpleNamespace(action_chunks=chunks_a),
        SimpleNamespace(action_chunks=chunks_b),
    ]

    batched = FakeChunkTrainWrapper()
    count = train._backward_phase11_group_loss(
        train_wrapper=batched,
        rollouts=rollouts,
        advantages=torch.tensor([1.0, 2.0], dtype=torch.float32),
        device=torch.device("cpu"),
        optimizer_chunk_size=1,
        clip_eps=0.2,
        logprob_recompute_mode="batched",
        logprob_batch_size=16,
        telemetry=None,
    )

    loop = FakeChunkTrainWrapper()
    train._backward_phase11_group_loss(
        train_wrapper=loop,
        rollouts=rollouts,
        advantages=torch.tensor([1.0, 2.0], dtype=torch.float32),
        device=torch.device("cpu"),
        optimizer_chunk_size=1,
        clip_eps=0.2,
        logprob_recompute_mode="loop",
        logprob_batch_size=16,
        telemetry=None,
    )

    assert count == 3
    assert batched.batch_calls == [(("a0", "b0", "b1"), (3, 5, 4))]
    torch.testing.assert_close(batched.scale.grad, loop.scale.grad)
    assert batched.scale.grad is not None
    assert batched.scale.grad.item() != -0.9048374


def test_phase11_group_loss_matches_closed_form_per_trajectory_normalizer():
    train = _load_phase11_train_module()
    rollouts = [
        SimpleNamespace(action_chunks=[_action_chunk("a0", steps=5)]),
        SimpleNamespace(action_chunks=[_action_chunk("b0", steps=5), _action_chunk("b1", steps=5)]),
    ]
    wrapper = FakeChunkTrainWrapper(init=-0.1)

    train._backward_phase11_group_loss(
        train_wrapper=wrapper,
        rollouts=rollouts,
        advantages=torch.tensor([1.0, 2.0], dtype=torch.float32),
        device=torch.device("cpu"),
        optimizer_chunk_size=1,
        clip_eps=0.2,
        logprob_recompute_mode="batched",
        logprob_batch_size=16,
        telemetry=None,
    )

    # d[-exp(new_lp) * A / normalizer] / d scale, with new_lp == scale.
    expected = -torch.exp(torch.tensor(-0.1)) * (
        torch.tensor(1.0) / torch.tensor(1 * 2)
        + torch.tensor(2.0) / torch.tensor(2 * 2)
        + torch.tensor(2.0) / torch.tensor(2 * 2)
    )
    wrong_global_denominator = -torch.exp(torch.tensor(-0.1)) * (
        torch.tensor(1.0)
        + torch.tensor(2.0)
        + torch.tensor(2.0)
    ) / torch.tensor(3 * 2)

    torch.testing.assert_close(wrapper.scale.grad, expected)
    assert not torch.allclose(wrapper.scale.grad, wrong_global_denominator)


def test_phase11_clipped_row_loss_positive_advantage_clamp_saturates_grad():
    train = _load_phase11_train_module()
    new_lp = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
    old_lp = torch.tensor([0.0], dtype=torch.float32)

    loss = train._phase11_clipped_row_loss(
        new_lp,
        old_lp,
        torch.tensor(1.0, dtype=torch.float32),
        clip_eps=0.2,
    ).sum()
    loss.backward()

    torch.testing.assert_close(new_lp.grad, torch.zeros_like(new_lp))


def test_phase11_training_loss_returns_zero_for_missing_action_chunks():
    train = _load_phase11_train_module()
    wrapper = FakeChunkTrainWrapper()

    count = train._backward_phase11_chunk_loss(
        train_wrapper=wrapper,
        action_chunks=[],
        advantage=torch.tensor(1.0, dtype=torch.float32),
        device=torch.device("cpu"),
        optimizer_chunk_size=2,
        clip_eps=0.2,
        normalizer=1,
    )

    assert count == 0
    assert wrapper.calls == []
    assert wrapper.batch_calls == []


def test_phase11_training_loss_rejects_non_positive_normalizer():
    train = _load_phase11_train_module()
    wrapper = FakeChunkTrainWrapper()

    try:
        train._backward_phase11_chunk_loss(
            train_wrapper=wrapper,
            action_chunks=[_action_chunk("root", steps=5)],
            advantage=torch.tensor(1.0, dtype=torch.float32),
            device=torch.device("cpu"),
            optimizer_chunk_size=2,
            clip_eps=0.2,
            normalizer=0,
        )
    except ValueError as exc:
        assert str(exc) == "normalizer must be > 0"
    else:
        raise AssertionError("expected ValueError")


def test_phase11_training_loop_rejects_missing_action_chunks_static():
    script = (_REPO / "scripts" / "grpo" / "train_phase11_env_on_policy_grpo.py").read_text(
        encoding="utf-8"
    )
    assert 'raise RuntimeError("Phase11 rollout produced no action_chunks")' in script


def test_phase11_train_script_exposes_action_chunk_size_static():
    script = (_REPO / "scripts" / "grpo" / "train_phase11_env_on_policy_grpo.py").read_text(
        encoding="utf-8"
    )
    assert "--action-chunk-size" in script
    assert "n_action_steps=int(args.action_chunk_size)" in script
    assert "action_chunk_size=int(args.action_chunk_size)" in script
    assert "num_policy_sample_calls" in script
    assert '"loss_unit": "policy_chunk"' in script
    assert "--logprob-recompute-mode" in script
    assert 'default="batched"' in script
    assert "--logprob-batch-size" in script
    assert "--rollout-policy-batch-size" in script
    assert "default=16" in script
    assert "default=32" in script
    assert '"logprob_recompute_mode": args.logprob_recompute_mode' in script
    assert '"logprob_batch_size": int(args.logprob_batch_size)' in script
    assert '"rollout_policy_batch_size": int(args.rollout_policy_batch_size)' in script


def test_phase11_json_ready_args_converts_path_values():
    train = _load_phase11_train_module()
    args = SimpleNamespace(
        checkpoint="ckpt",
        output_dir=_REPO / "out",
        resume=_REPO / "resume.pt",
        chunk_size=2,
    )

    ready = train._json_ready_args(args)

    assert ready["checkpoint"] == "ckpt"
    assert ready["output_dir"] == str(_REPO / "out")
    assert ready["resume"] == str(_REPO / "resume.pt")
    assert ready["chunk_size"] == 2


def test_phase11_train_script_logs_process_memory_static():
    script = (_REPO / "scripts" / "grpo" / "train_phase11_env_on_policy_grpo.py").read_text(
        encoding="utf-8"
    )
    assert "from smolvla_grpo.process_memory import prefixed_process_tree_memory_fields" in script
    assert 'proc_mem_update_start' in script
    assert 'proc_mem_after_rollout' in script
    assert 'proc_mem_after_optimize' in script
    assert '**proc_mem_update_start' in script
    assert '**proc_mem_after_rollout' in script
    assert '**proc_mem_after_optimize' in script


def test_phase11_train_script_checkpoint_metadata_static():
    script = (_REPO / "scripts" / "grpo" / "train_phase11_env_on_policy_grpo.py").read_text(
        encoding="utf-8"
    )
    assert "metrics_common = {" in script
    assert '"optimizer_chunk_size": int(args.chunk_size)' in script
    assert '"num_policy_sample_calls": num_policy_sample_calls' in script
    assert '"num_loss_units": num_loss_units' in script
    assert '"num_logprob_forward_batches": 0' in script
    assert '"ratio_clip_fraction": None' in script
    assert '"approx_kl": None' in script
    assert '"log_std_mean": None' in script
    assert "skipped_extra = {" in script
    assert "checkpoint_extra = {" in script
    assert '"rollout_seconds": rollout_seconds' in script
    assert '"optimize_seconds": optimize_seconds' in script
    assert '"update_seconds": update_seconds' in script
    assert '"optimize_seconds": 0.0' in script
    assert script.count("args=_json_ready_args(args)") == script.count("save_grpo_checkpoint(")
    assert "args=vars(args)" not in script
