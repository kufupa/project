from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def test_episode_waves_use_distinct_eval_seeds() -> None:
    from smolvla_grpo.phase12_vector_eval import build_episode_waves

    waves = build_episode_waves(episodes=10, eval_seed_start=1000, n_envs=4)

    assert waves == [
        [(0, 1000), (1, 1001), (2, 1002), (3, 1003)],
        [(4, 1004), (5, 1005), (6, 1006), (7, 1007)],
        [(8, 1008), (9, 1009)],
    ]


def test_episode_waves_reject_bad_n_envs() -> None:
    from smolvla_grpo.phase12_vector_eval import build_episode_waves

    with pytest.raises(ValueError, match="n_envs must be >= 1"):
        build_episode_waves(episodes=10, eval_seed_start=1000, n_envs=0)


class TinyPolicy:
    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {
            "linear.weight": object(),
            "linear.bias": object(),
        }

    def load_state_dict(self, state, strict=False):
        self.loaded = dict(state)
        missing = [k for k in self.state_dict() if k not in state]
        unexpected = [k for k in state if k not in self.state_dict()]
        return missing, unexpected


def test_validate_checkpoint_state_rejects_missing_non_whitelisted_key() -> None:
    from smolvla_grpo.phase12_vector_eval import validate_checkpoint_state

    policy = TinyPolicy()
    with pytest.raises(RuntimeError, match="missing checkpoint keys"):
        validate_checkpoint_state(policy, {"linear.weight": object()})


def test_validate_checkpoint_state_allows_log_std_missing_only() -> None:
    from smolvla_grpo.phase12_vector_eval import validate_checkpoint_state

    class PolicyWithLogStd(TinyPolicy):
        def state_dict(self):
            return {
                "linear.weight": object(),
                "model.log_std": object(),
            }

    validate_checkpoint_state(PolicyWithLogStd(), {"linear.weight": object()})


def test_write_eval_artifacts_preserves_schema(tmp_path, monkeypatch) -> None:
    from smolvla_grpo.phase12_vector_eval import EpisodeResult, write_eval_artifacts

    calls = []

    def fake_write_episode_artifacts(*, episode_dir, actions, rewards, successes, overlay_mode):
        calls.append((episode_dir.name, list(actions), list(rewards), list(successes), overlay_mode))
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "actions.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr("smolvla_grpo.phase12_vector_eval.write_episode_artifacts", fake_write_episode_artifacts)

    results = [
        EpisodeResult(
            episode_index=1,
            reset_seed=1001,
            actions=[[0.0, 0.0, 0.0, 0.0]],
            rewards=[2.0],
            successes=[False],
            terminated=False,
            truncated=True,
        ),
        EpisodeResult(
            episode_index=0,
            reset_seed=1000,
            actions=[[1.0, 0.0, 0.0, 0.0]],
            rewards=[1.0],
            successes=[True],
            terminated=True,
            truncated=False,
        ),
    ]

    summary = write_eval_artifacts(
        base_checkpoint="base",
        grpo_checkpoint=Path("update_0010.pt"),
        output_dir=tmp_path,
        task="push-v3",
        episodes=2,
        eval_seed_start=1000,
        results=results,
    )

    assert summary["episodes"] == 2
    assert summary["pc_success"] == 50.0
    assert summary["avg_sum_reward"] == 1.5
    rows = [json.loads(line) for line in (tmp_path / "eval_episodes.jsonl").read_text().splitlines()]
    assert [row["episode_index"] for row in rows] == [0, 1]
    assert [row["reset_seed"] for row in rows] == [1000, 1001]
    assert (tmp_path / "eval_summary.json").exists()
    assert (tmp_path / "eval_info.json").exists()
    info = json.loads((tmp_path / "eval_info.json").read_text(encoding="utf-8"))
    assert "reset_randomization_mode" in info
    assert calls[0][0] == "episode_0000"


def test_coerce_exec_action_batch_preserves_rows() -> None:
    from smolvla_grpo.phase12_vector_eval import coerce_exec_action_batch

    action = np.array([[2.0, 0.5, -0.5, -2.0], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    out = coerce_exec_action_batch(action, action_dim=4, n_envs=2)

    assert out.shape == (2, 4)
    np.testing.assert_allclose(out[0], np.array([1.0, 0.5, -0.5, -1.0], dtype=np.float32))
    np.testing.assert_allclose(out[1], np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))


def test_coerce_exec_action_chunk_batch_preserves_chunk_rows() -> None:
    from smolvla_grpo.phase12_vector_eval import coerce_exec_action_chunk_batch

    action = np.array(
        [
            [[2.0, 0.5, -0.5, -2.0], [0.1, 0.2, 0.3, 0.4]],
            [[-3.0, 0.0, 3.0, 0.5], [0.9, -0.9, 0.8, -0.8]],
        ],
        dtype=np.float32,
    )

    out = coerce_exec_action_chunk_batch(action, action_dim=4, n_envs=2, chunk_len=2)

    assert out.shape == (2, 2, 4)
    np.testing.assert_allclose(out[0, 0], np.array([1.0, 0.5, -0.5, -1.0], dtype=np.float32))
    np.testing.assert_allclose(out[1, 0], np.array([-1.0, 0.0, 1.0, 0.5], dtype=np.float32))


def test_concatenate_proc_rows_preserves_batch_order() -> None:
    import torch

    from smolvla_grpo.phase12_vector_eval import concatenate_proc_rows

    rows = [
        {
            "observation.state": torch.tensor([[1.0, 2.0]]),
            "observation.image": torch.zeros(1, 3, 2, 2),
            "task": ["seed1000"],
        },
        {
            "observation.state": torch.tensor([[3.0, 4.0]]),
            "observation.image": torch.ones(1, 3, 2, 2),
            "task": ["seed1001"],
        },
    ]

    out = concatenate_proc_rows(rows)

    assert out["observation.state"].shape == (2, 2)
    np.testing.assert_allclose(out["observation.state"].numpy(), np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert out["observation.image"].shape == (2, 3, 2, 2)
    assert out["task"] == ["seed1000", "seed1001"]


def test_select_eval_action_chunk_queue_free_uses_model_sample_actions() -> None:
    import torch
    from types import SimpleNamespace

    from smolvla_grpo.phase12_vector_eval import select_eval_action_chunk_queue_free

    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
            del images, img_masks, lang_tokens, lang_masks, state, noise
            self.calls += 1
            return torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)

    class FakePolicy:
        def __init__(self) -> None:
            self.model = FakeModel()
            self.config = SimpleNamespace(action_feature=SimpleNamespace(shape=(4,)))
            self.select_action_calls = 0

        def _prepare_batch(self, proc):
            return proc

        def prepare_images(self, batch):
            return batch["images"], batch["img_masks"]

        def prepare_state(self, batch):
            return batch["state"]

        def select_action(self, proc):
            del proc
            self.select_action_calls += 1
            return torch.zeros(2, 4)

    proc = {
        "images": torch.zeros(2, 3, 8, 8),
        "img_masks": torch.ones(2, 1),
        "state": torch.zeros(2, 8),
        "observation.language.tokens": torch.zeros(2, 4, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(2, 4, dtype=torch.long),
    }
    policy = FakePolicy()

    got = select_eval_action_chunk_queue_free(policy, proc, chunk_len=3)

    assert got.shape == (2, 3, 4)
    assert policy.model.calls == 1
    assert policy.select_action_calls == 0
    torch.testing.assert_close(got, torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)[:, :3, :])


def _load_phase12_sweep_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "scripts" / "grpo" / "eval_phase12_checkpoint_sweep.py"
    spec = importlib.util.spec_from_file_location("eval_phase12_checkpoint_sweep", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_phase12_vector_sweep_uses_resident_eval(tmp_path, monkeypatch) -> None:
    mod = _load_phase12_sweep_module()
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "update_0010.pt").write_bytes(b"fake")
    calls = []

    def fake_run_vector_sweep(**kwargs):
        calls.append(kwargs)
        sweep_dir = kwargs["run_dir"] / kwargs["sweep_name"]
        update_dir = sweep_dir / "update_0010"
        update_dir.mkdir(parents=True)
        (update_dir / "eval_summary.json").write_text(
            json.dumps({"pc_success": 0.0, "avg_sum_reward": 1.0, "avg_max_reward": 2.0, "episodes": 4}),
            encoding="utf-8",
        )
        result = {
            "task": kwargs["task"],
            "episodes": kwargs["episodes"],
            "eval_seed_start": kwargs["eval_seed_start"],
            "sweep_name": kwargs["sweep_name"],
            "min_update": kwargs["min_update"],
            "max_update": kwargs["max_update"],
            "stride": kwargs["stride"],
            "rows": [
                {
                    "update": 10,
                    "checkpoint": str(run_dir / "checkpoints" / "update_0010.pt"),
                    "pc_success": 0.0,
                    "avg_sum_reward": 1.0,
                    "avg_max_reward": 2.0,
                    "episodes": 4,
                    "eval_summary_path": str(update_dir / "eval_summary.json"),
                }
            ],
        }
        (sweep_dir / "eval_sweep_summary.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(mod, "run_sweep_inprocess_vector", fake_run_vector_sweep)
    result = mod.run_sweep(
        base_checkpoint="base",
        run_dir=run_dir,
        task="push-v3",
        episodes=4,
        eval_seed_start=1000,
        sweep_name="vec",
        min_update=10,
        max_update=10,
        stride=10,
        execution_mode="inprocess_vector",
        n_envs=2,
        rollout_execution="vector_sync",
        max_steps=5,
    )

    assert len(result["rows"]) == 1
    assert calls[0]["n_envs"] == 2
    assert calls[0]["rollout_execution"] == "vector_sync"


def test_vector_eval_uses_queue_free_action_when_active_rows_shrink(tmp_path, monkeypatch):
    import sys
    import types
    from collections import deque
    from types import SimpleNamespace

    import torch

    from smolvla_grpo.phase12_vector_eval import evaluate_loaded_policy_vectorized

    class FakePolicy:
        def __init__(self):
            self.reset_calls = 0
            self.select_action_calls = 0
            self.queue_free_calls = []
            self._queues = {"action": deque(maxlen=50)}

        def reset(self):
            self.reset_calls += 1
            self._queues["action"].clear()

        def select_action(self, proc):
            self.select_action_calls += 1
            batch = int(proc["observation.state"].shape[0])
            if not self._queues["action"]:
                for _ in range(50):
                    self._queues["action"].append(torch.zeros(batch, 4))
            return self._queues["action"].popleft()

    class FakeBundle:
        base_checkpoint = "base"
        grpo_checkpoint = None
        device = torch.device("cpu")

        def __init__(self):
            self.policy = FakePolicy()

        def postprocessor(self, action):
            return action

    class FakeEnv:
        action_dim = 4

        def __init__(self, *, task, n_envs=1):
            del task
            assert n_envs == 1
            self.seed = None
            self.steps = 0

        def reset(self, seed):
            self.seed = int(seed)
            self.steps = 0
            return {"state": np.array([float(seed), 0.0, 0.0, 0.0], dtype=np.float32)}

        def build_proc(self, obs, *, bundle):
            del bundle
            return {
                "observation.state": torch.tensor([[obs["state"][0], 0.0, 0.0, 0.0]], dtype=torch.float32),
                "task": [f"seed-{self.seed}"],
            }

        def step(self, action):
            self.steps += 1
            success = self.seed == 1001 and self.steps == 1
            return SimpleNamespace(
                observation={"state": np.array([float(self.seed), float(self.steps), 0.0, 0.0], dtype=np.float32)},
                reward=1.0,
                success=success,
                terminated=False,
                truncated=False,
            )

        def close(self):
            return None

    def fake_write_episode_artifacts(*, episode_dir, actions, rewards, successes, overlay_mode):
        del actions, rewards, successes, overlay_mode
        episode_dir.mkdir(parents=True, exist_ok=True)

    fake_adapter = types.ModuleType("smolvla_grpo.lerobot_metaworld_adapter")
    fake_adapter.OfficialLeRobotMetaWorldGRPORollout = FakeEnv
    fake_adapter.resolve_lerobot_horizon = lambda env, max_steps: int(max_steps)
    monkeypatch.setitem(sys.modules, "smolvla_grpo.lerobot_metaworld_adapter", fake_adapter)
    monkeypatch.setattr("smolvla_grpo.phase12_vector_eval._resolve_action_dim", lambda task: 4)
    monkeypatch.setattr("smolvla_grpo.phase12_vector_eval.write_episode_artifacts", fake_write_episode_artifacts)

    def fake_queue_free(policy, proc):
        batch = int(proc["observation.state"].shape[0])
        policy.queue_free_calls.append(batch)
        return torch.zeros(batch, 4)

    monkeypatch.setattr("smolvla_grpo.phase12_vector_eval.select_eval_action_queue_free", fake_queue_free)

    bundle = FakeBundle()
    summary = evaluate_loaded_policy_vectorized(
        bundle=bundle,
        base_checkpoint="base",
        grpo_checkpoint=None,
        output_dir=tmp_path,
        task="push-v3",
        episodes=3,
        eval_seed_start=1000,
        n_envs=3,
        rollout_execution="vector_sync",
        max_steps=3,
    )

    assert summary["episodes"] == 3
    assert bundle.policy.select_action_calls == 0
    assert bundle.policy.queue_free_calls == [3, 2, 2]


def test_vector_eval_chunked_execution_samples_once_per_chunk(tmp_path, monkeypatch):
    import sys
    import types
    from types import SimpleNamespace

    import torch

    from smolvla_grpo.phase12_vector_eval import evaluate_loaded_policy_vectorized

    class FakePolicy:
        def __init__(self):
            self.reset_calls = 0
            self.chunk_calls = []

        def reset(self):
            self.reset_calls += 1

    class FakeBundle:
        base_checkpoint = "base"
        grpo_checkpoint = None
        device = torch.device("cpu")

        def __init__(self):
            self.policy = FakePolicy()

        def postprocessor(self, action):
            return action

    class FakeEnv:
        action_dim = 4

        def __init__(self, *, task, n_envs=1):
            del task
            assert n_envs == 1
            self.seed = None
            self.steps = 0

        def reset(self, seed):
            self.seed = int(seed)
            self.steps = 0
            return {"state": np.array([float(seed), 0.0, 0.0, 0.0], dtype=np.float32)}

        def build_proc(self, obs, *, bundle):
            del bundle
            return {
                "observation.state": torch.tensor([[obs["state"][0], obs["state"][1], 0.0, 0.0]], dtype=torch.float32),
                "task": [f"seed-{self.seed}"],
            }

        def step(self, action):
            self.steps += 1
            return SimpleNamespace(
                observation={"state": np.array([float(self.seed), float(self.steps), 0.0, 0.0], dtype=np.float32)},
                reward=float(np.asarray(action).reshape(-1)[0]),
                success=False,
                terminated=False,
                truncated=False,
            )

        def close(self):
            return None

    fake_adapter = types.ModuleType("smolvla_grpo.lerobot_metaworld_adapter")
    fake_adapter.OfficialLeRobotMetaWorldGRPORollout = FakeEnv
    fake_adapter.resolve_lerobot_horizon = lambda env, max_steps: int(max_steps)
    monkeypatch.setitem(sys.modules, "smolvla_grpo.lerobot_metaworld_adapter", fake_adapter)
    monkeypatch.setattr("smolvla_grpo.phase12_vector_eval._resolve_action_dim", lambda task: 4)

    def fake_write_episode_artifacts(*, episode_dir, actions, rewards, successes, overlay_mode):
        del actions, rewards, successes, overlay_mode
        episode_dir.mkdir(parents=True, exist_ok=True)

    def fake_chunk(policy, proc, *, chunk_len):
        policy.chunk_calls.append((int(proc["observation.state"].shape[0]), int(chunk_len)))
        batch = int(proc["observation.state"].shape[0])
        return torch.ones(batch, int(chunk_len), 4)

    monkeypatch.setattr("smolvla_grpo.phase12_vector_eval.write_episode_artifacts", fake_write_episode_artifacts)
    monkeypatch.setattr("smolvla_grpo.phase12_vector_eval.select_eval_action_chunk_queue_free", fake_chunk)

    bundle = FakeBundle()
    summary = evaluate_loaded_policy_vectorized(
        bundle=bundle,
        base_checkpoint="base",
        grpo_checkpoint=None,
        output_dir=tmp_path,
        task="push-v3",
        episodes=3,
        eval_seed_start=1000,
        n_envs=3,
        rollout_execution="vector_sync",
        max_steps=5,
        chunk_len=2,
    )

    assert summary["episodes"] == 3
    assert summary["chunk_len"] == 2
    assert bundle.policy.chunk_calls == [(3, 2), (3, 2), (3, 1)]


def test_baseline_vector_video_parse_accepts_chunk_len(tmp_path):
    from scripts.grpo.eval_smolvla_baseline_vector_video import parse_args

    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--task",
            "push-v3",
            "--episodes",
            "25",
            "--eval-seed-start",
            "1000",
            "--n-envs",
            "3",
            "--max-steps",
            "180",
            "--chunk-len",
            "20",
        ]
    )

    assert args.task == "push-v3"
    assert args.episodes == 25
    assert args.eval_seed_start == 1000
    assert args.n_envs == 3
    assert args.max_steps == 180
    assert args.chunk_len == 20


def test_vector_eval_rejects_invalid_chunk_len(tmp_path) -> None:
    from types import SimpleNamespace

    from smolvla_grpo.phase12_vector_eval import evaluate_loaded_policy_vectorized

    bundle = SimpleNamespace(policy=SimpleNamespace(), postprocessor=lambda action: action)

    with pytest.raises(ValueError, match="chunk_len must be >= 1"):
        evaluate_loaded_policy_vectorized(
            bundle=bundle,
            base_checkpoint="base",
            grpo_checkpoint=None,
            output_dir=tmp_path,
            task="push-v3",
            episodes=1,
            eval_seed_start=1000,
            n_envs=1,
            rollout_execution="vector_sync",
            max_steps=1,
            chunk_len=0,
        )


def test_baseline_vector_video_parse_rejects_invalid_chunk_len(tmp_path) -> None:
    from scripts.grpo.eval_smolvla_baseline_vector_video import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--output-dir", str(tmp_path), "--chunk-len", "0"])

