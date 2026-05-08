from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]


def test_eggroll_rollout_is_queue_free_static() -> None:
    text = (PROJECT / "src" / "smolvla_grpo" / "eggroll_rollout.py").read_text(encoding="utf-8")

    assert "torch.inference_mode()" in text
    assert ".backward(" not in text
    assert "policy.model.sample_actions" in text
    assert "actions =" in text
    assert "mean, log_std" not in text
    assert "population_batch_size" in text
    assert "member_ids=" in text
    assert "flow_dtype" in text
    assert "rollout_seed_offset" in text
    assert "_frame_from_vector_obs" in text
    assert "render_frame()" not in text
    assert "select_action(" not in text
    assert "select_action_distr_params(" not in text


def test_eggroll_rollout_executes_chunk_prefix_static() -> None:
    text = (PROJECT / "src" / "smolvla_grpo" / "eggroll_rollout.py").read_text(encoding="utf-8")

    assert "action_chunk_size: int = 5" in text
    assert "effective_chunk = min(" in text
    assert "chunk[compact_idx, chunk_step" in text
    assert "chunk[:, 0, :]" not in text
    assert "policy_call_idx" in text


def test_eggroll_train_cli_exposes_action_chunk_size() -> None:
    text = (PROJECT / "scripts" / "eggroll" / "train_smolvla_eggroll.py").read_text(encoding="utf-8")

    assert "--action-chunk-size" in text
    assert "--chunk-len" in text
    assert "default=5" in text


def test_eggroll_rollout_executes_chunk_prefix_and_stops_done_rows(monkeypatch) -> None:
    from smolvla_grpo import eggroll_rollout as mod

    class FakeModel:
        def __init__(self) -> None:
            self.action_in_proj = SimpleNamespace(weight=torch.ones(1, dtype=torch.float32))
            self.calls: list[int] = []

        def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
            del images, img_masks, lang_tokens, lang_masks, noise
            batch = int(state.shape[0])
            self.calls.append(batch)
            out = torch.zeros(batch, 4, 4)
            for row in range(batch):
                for step in range(4):
                    out[row, step, 0] = float((row + 1) * 10 + step)
            return out

    class FakePolicy:
        def __init__(self) -> None:
            self.model = FakeModel()
            self.config = SimpleNamespace(
                action_feature=SimpleNamespace(shape=(4,)),
                max_action_dim=4,
                chunk_size=4,
                input_features={},
            )

        def _prepare_batch(self, proc):
            return proc

        def prepare_images(self, batch):
            return batch["images"], batch["img_masks"]

        def prepare_state(self, batch):
            return batch["state"]

    class FakeBundle:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.policy = FakePolicy()

        def postprocessor(self, action):
            return action

    class FakeEnv:
        action_dim = 4

        def __init__(self, *, task, n_envs=1, use_async_envs=False, async_start_method="forkserver"):
            del task, use_async_envs, async_start_method
            self.n_envs = int(n_envs)
            assert self.n_envs == 2
            self.inner = self
            self.action_space = SimpleNamespace(
                shape=(4,),
                low=np.full((4,), -100.0, dtype=np.float32),
                high=np.full((4,), 100.0, dtype=np.float32),
            )
            self.seeds = []
            self.steps = 0

        def reset_many(self, seeds):
            self.seeds = [int(seed) for seed in seeds]
            self.steps = 0
            return {"pixels": np.zeros((self.n_envs, 2, 2, 3), dtype=np.uint8), "agent_pos": np.zeros((self.n_envs, 4), dtype=np.float32)}

        def build_proc(self, obs, *, bundle):
            del obs, bundle
            return {
                "images": torch.zeros(self.n_envs, 3, 2, 2),
                "img_masks": torch.ones(self.n_envs, 1),
                "state": torch.zeros(self.n_envs, 4),
                "observation.language.tokens": torch.zeros(self.n_envs, 2, dtype=torch.long),
                "observation.language.attention_mask": torch.ones(self.n_envs, 2, dtype=torch.long),
                "task": [f"seed-{seed}" for seed in self.seeds],
            }

        def step_batch(self, action):
            self.steps += 1
            action0 = np.asarray(action, dtype=np.float32)[:, 0]
            success = np.asarray([seed == 100 and self.steps == 1 for seed in self.seeds], dtype=bool)
            return SimpleNamespace(
                observation={"pixels": np.full((self.n_envs, 2, 2, 3), self.steps, dtype=np.uint8), "agent_pos": np.zeros((self.n_envs, 4), dtype=np.float32)},
                reward=action0,
                success=success,
                terminated=success,
                truncated=np.zeros((self.n_envs,), dtype=bool),
            )

        def close(self):
            return None

    class FakePatchHandle:
        def context(self, **kwargs):
            return SimpleNamespace(**kwargs)

    monkeypatch.setattr(mod, "OfficialLeRobotMetaWorldGRPORollout", FakeEnv)
    monkeypatch.setattr(mod, "resolve_lerobot_horizon", lambda env, max_steps: int(max_steps))

    result = mod.collect_eggroll_population_rollouts(
        bundle=FakeBundle(),
        task="push-v3",
        task_text="push",
        action_dim=4,
        population_size=2,
        population_batch_size=2,
        iteration=0,
        max_steps=3,
        action_chunk_size=2,
        train_seed_base=100,
        flow_noise_seed=23,
        noise_manager=SimpleNamespace(rank=1),
        patch_handle=FakePatchHandle(),
        sigma=0.01,
        video_member_id=0,
    )

    rollouts = sorted(result.rollouts, key=lambda item: item.member_id)
    assert [len(item.actions) for item in rollouts] == [1, 3]
    assert rollouts[0].terminated is True
    assert rollouts[0].rewards == [10.0]
    assert rollouts[1].rewards == [20.0, 21.0, 10.0]
    assert result.selected_rewards == [10.0]
    assert len(result.selected_frames) == 2
