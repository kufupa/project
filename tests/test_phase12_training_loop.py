from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from scripts.grpo import train_phase12_wm_chunk_grpo as trainer
from smolvla_grpo.phase12_rollout import Phase12EpisodeResult


class _TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.log_std = torch.nn.Parameter(torch.zeros(1, 4))
        self.model.euler_step_noise_std = 0.0
        self.w = torch.nn.Parameter(torch.zeros(1))

    def select_action_distr_params(self, proc):
        del proc
        mean = self.w.reshape(1, 1).expand(4, 4)
        log_std = self.model.log_std.expand(4, 4)
        return mean, log_std


class _TinyBundle:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.policy = _TinyPolicy()
        self.preprocessor = lambda obs: obs
        self.postprocessor = lambda x: x


def test_wm_grpo_train_writes_update_row_and_nonempty_checkpoint(monkeypatch, tmp_path) -> None:
    selected = tmp_path / "selected_action_rollout.mp4"
    oracle = tmp_path / "oracle_baseline.mp4"
    selected.write_bytes(b"selected")
    oracle.write_bytes(b"oracle")
    chunks = [torch.full((4, 4), 0.1 * (i + 1), dtype=torch.float32) for i in range(4)]
    episode = SimpleNamespace(
        total_env_reward=1.0,
        success_any=False,
        success_last=False,
        metadata={
            "segment_candidate_rewards": [[0.0, 1.0, 2.0, 3.0]],
            "candidate_rewards": [0.0, 1.0, 2.0, 3.0],
            "old_logprob_sums": [-1.0, -1.1, -1.2, -1.3],
            "proc_root_snapshots": [{"x": torch.zeros(1, 1)} for _ in range(4)],
            "unsquashed_chunks": chunks,
            "rollout_validation_video": str(selected),
            "selected_action_rollout_video": str(selected),
            "oracle_baseline_video": str(oracle),
            "oracle_baseline_video_status": "ok",
            "wm_decode_status": "failed",
        },
    )
    monkeypatch.setattr(trainer, "load_phase12_train_resources", lambda args: (_TinyBundle(), object(), 4))
    monkeypatch.setattr(trainer, "collect_phase12_training_episode", lambda **_kwargs: episode)

    code = trainer.main(
        [
            "--mode",
            "wm_grpo_train",
            "--output-dir",
            str(tmp_path),
            "--jepa-repo",
            "/tmp/jepa",
            "--jepa-ckpt",
            "wm.pt",
            "--num-updates",
            "1",
            "--num-episodes",
            "1",
        ]
    )

    assert code == 0
    rows = [json.loads(x) for x in (tmp_path / "progress.jsonl").read_text().splitlines() if x.strip()]
    assert any(row.get("event") == "update_complete" and row.get("optimizer_step") is True for row in rows)
    ckpt = torch.load(tmp_path / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    assert ckpt["policy_state_dict"]
    assert ckpt["optimizer_state_dict"]


def test_phase12_reset_gate_tolerates_sparse_render_jitter_when_raw_state_matches() -> None:
    decision = trainer._phase12_reset_gate_decision(
        reset_metrics={
            "image_mean_abs_diff": 0.0634794533252716,
            "image_max_abs_diff": 170.0,
            "proprio_max_abs_diff": 0.0,
        },
        reset_debug_report={"raw_obs_max_abs_diff": 0.0},
    )

    assert decision["old_strict_image_gate_would_fail"] is True
    assert decision["fail"] is False


def test_phase12_reset_gate_fails_raw_state_mismatch_even_if_image_matches() -> None:
    decision = trainer._phase12_reset_gate_decision(
        reset_metrics={
            "image_mean_abs_diff": 0.0,
            "image_max_abs_diff": 0.0,
            "proprio_max_abs_diff": 0.0,
        },
        reset_debug_report={"raw_obs_max_abs_diff": 1e-3},
    )

    assert decision["raw_mismatch"] is True
    assert decision["fail"] is True


def test_structured_field_reads_tensordict_like_values() -> None:
    class TensorDictLike:
        def __init__(self) -> None:
            self.values = {"visual": "v", "proprio": "p"}

        def __contains__(self, key):
            return key in self.values

        def __getitem__(self, key):
            return self.values[key]

        def get(self, key, default=None):
            return self.values.get(key, default)

    td = TensorDictLike()

    assert trainer._structured_field(td, "visual") == "v"
    assert trainer._structured_field(td, "proprio") == "p"
    assert trainer._structured_field(td, "missing") is None


def test_with_episode_metadata_handles_frozen_phase12_episode_result() -> None:
    episode = Phase12EpisodeResult(
        segments=[],
        total_env_reward=0.0,
        success_any=False,
        success_last=False,
        metadata={"old": 1},
    )

    updated = trainer._with_episode_metadata(episode, {"new": 2})

    assert updated is not episode
    assert updated.metadata == {"new": 2}
    assert episode.metadata == {"old": 1}


def test_phase12_decode_metadata_is_promoted_to_smoke_contract_key() -> None:
    meta = {"existing": True}

    trainer._merge_phase12_decode_metadata(
        meta,
        {
            "decode_status": "ok",
            "wm_decode_selected_strip_path": "/tmp/selected.png",
            "wm_real_vs_pred_selected_strip_path": "/tmp/real_vs_pred.png",
        },
    )

    assert meta["wm_decode_status"] == "ok"
    assert meta["wm_decode_selected_strip_path"] == "/tmp/selected.png"
    assert meta["wm_real_vs_pred_selected_strip_path"] == "/tmp/real_vs_pred.png"


def test_phase12_decode_uses_final_unroll_timestep(monkeypatch) -> None:
    class _Preprocessor:
        action_mean = torch.zeros(4)
        action_std = torch.ones(4)

    class _Model:
        action_dim = 4

        def __init__(self) -> None:
            self.calls = 0

        def encode(self, obs):
            del obs
            return {
                "visual": torch.zeros(1, 1, 1, dtype=torch.float32),
                "proprio": torch.zeros(1, 1, 1, dtype=torch.float32),
            }

        def unroll(self, latent, *, act_suffix, debug=False):
            del latent, act_suffix, debug
            self.calls += 1
            current = torch.full((1, 1, 1), 100.0 + self.calls)
            predicted = torch.full((1, 1, 1), float(self.calls))
            return {
                "visual": torch.cat([current, predicted], dim=0),
                "proprio": torch.cat([current, predicted], dim=0),
            }

    def _fake_decode(_wm_bundle, trace):
        frames = [
            np.full((2, 2, 3), int(np.asarray(latent).reshape(-1)[0]), dtype=np.uint8)
            for latent in trace.visual_latents
        ]
        return frames, None

    monkeypatch.setattr(trainer, "_decode_latent_trace_to_frames", _fake_decode, raising=False)
    monkeypatch.setattr("segment_grpo_loop._decode_latent_trace_to_frames", _fake_decode)

    model = _Model()
    wm_bundle = SimpleNamespace(
        model=model,
        preprocessor=_Preprocessor(),
        proprio_dim=1,
        planner_action_dim=4,
        device=torch.device("cpu"),
    )

    frames = trainer._decode_phase12_prediction_frames(
        wm_bundle,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        proprio=np.zeros(1, dtype=np.float32),
        actions=np.zeros((2, 4), dtype=np.float32),
        mode="visual_proprio",
    )

    assert model.calls == 2
    assert [int(frame[0, 0, 0]) for frame in frames] == [1, 2]


def test_phase12_root_uses_wm_image_for_scoring_and_policy_obs_for_proc() -> None:
    class EnvHarness:
        inner = SimpleNamespace(single_action_space=SimpleNamespace(shape=(4,)))

        def __init__(self) -> None:
            self.proc_obs = []

        def build_proc(self, obs, *, bundle):
            del bundle
            self.proc_obs.append(obs)
            return {"policy_proc": True}

    policy_frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    expected_wm_frame = np.flip(policy_frame, axis=1)
    obs = {"pixels": policy_frame[None], "agent_pos": np.zeros((1, 4), dtype=np.float32)}
    env_h = EnvHarness()

    rollout = trainer._Phase12SelectedRolloutEnv(
        env_h=env_h,
        bundle=object(),
        seed=7,
        initial_obs=obs,
        initial_frame=policy_frame,
        initial_proprio=np.zeros(4, dtype=np.float32),
    )

    root = rollout.reset()

    np.testing.assert_array_equal(root["image"], expected_wm_frame)
    np.testing.assert_array_equal(root["policy_image"], policy_frame)
    assert root["proc"] == {"policy_proc": True}
    assert env_h.proc_obs[0] is obs


def test_phase12_oracle_records_policy_and_wm_frames(monkeypatch, tmp_path) -> None:
    reset_policy = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    step_policy = reset_policy + 20

    class Step:
        observation = {"pixels": step_policy[None]}
        reward = 1.0
        success = True
        terminated = True
        truncated = False
        info = {"success": True}

    class EnvHarness:
        def reset(self, seed):
            assert seed == 123
            return {"pixels": reset_policy[None]}

        def last_agent_pos(self):
            return np.arange(4, dtype=np.float32)

        def last_raw_obs(self):
            return np.arange(5, dtype=np.float64)

        def expert_action(self):
            return np.zeros(4, dtype=np.float32)

        def step(self, action):
            np.testing.assert_array_equal(action, np.zeros((1, 4), dtype=np.float32))
            return Step()

    monkeypatch.setattr(trainer, "write_phase12_episode_video", lambda **kwargs: kwargs["video_path"])

    oracle = trainer._rollout_phase12_oracle(
        env_h=EnvHarness(),
        seed=123,
        max_steps=1,
        output_dir=tmp_path,
        fps=6,
    )

    np.testing.assert_array_equal(oracle["frames"][0], reset_policy)
    np.testing.assert_array_equal(oracle["wm_frames"][0], np.flip(reset_policy, 1))
    np.testing.assert_array_equal(oracle["frames"][1], step_policy)
    np.testing.assert_array_equal(oracle["wm_frames"][1], np.flip(step_policy, 1))
    assert len(oracle["wm_frames"]) == len(oracle["frames"])


def test_phase12_selected_decode_uses_wm_frames_as_real_frames(monkeypatch, tmp_path) -> None:
    seen: dict[str, object] = {}
    policy_frame = np.full((2, 2, 3), 11, dtype=np.uint8)
    wm_frame = np.full((2, 2, 3), 22, dtype=np.uint8)

    def fake_build_decode_artifacts(**kwargs):
        seen["real_frames"] = kwargs["real_frames"]
        return SimpleNamespace(paths={}, metadata={"decode_status": "ok"})

    monkeypatch.setattr(trainer, "build_decode_artifacts", fake_build_decode_artifacts, raising=False)

    episode = SimpleNamespace(segments=[SimpleNamespace(selected_candidate_index=0)])
    rollout_env = SimpleNamespace(frames=[policy_frame], wm_frames=[wm_frame])
    score_inputs = {
        (0, 0): {
            "image": wm_frame,
            "proprio": np.zeros(4, dtype=np.float32),
            "actions": np.zeros((5, 4), dtype=np.float32),
        }
    }
    meta: dict[str, object] = {}

    trainer._build_phase12_selected_decode_artifacts(
        args=SimpleNamespace(save_wm_decodes=True, strict_decode=True, goal_latent_mode="visual_proprio", chunk_len=5),
        episode=episode,
        episode_dir=tmp_path,
        rollout_env=rollout_env,
        score_inputs=score_inputs,
        wm_bundle=SimpleNamespace(planner_action_dim=20),
        action_dim=4,
        meta=meta,
    )

    assert seen["real_frames"] == [wm_frame]
    assert meta["wm_decode_status"] == "ok"


def test_phase12_microbatch_backward_frees_each_logprob_graph_before_next_forward() -> None:
    class TrackGraph(torch.autograd.Function):
        active = 0
        max_active = 0

        @staticmethod
        def forward(ctx, weight, value):
            del ctx
            TrackGraph.active += 1
            TrackGraph.max_active = max(TrackGraph.max_active, TrackGraph.active)
            return weight * 0.0 + weight.new_tensor(float(value))

        @staticmethod
        def backward(ctx, grad_output):
            del ctx
            TrackGraph.active -= 1
            return torch.zeros_like(grad_output), None

    class Wrapper:
        def __init__(self, weight):
            self.weight = weight
            self.calls = 0

        def get_action_probs_for_chunk_from_proc(self, proc, chunk):
            del proc, chunk
            self.calls += 1
            return TrackGraph.apply(self.weight, -0.1 * self.calls)

    weight = torch.nn.Parameter(torch.tensor(1.0))
    wrapper = Wrapper(weight)
    old_lp = torch.zeros(4)
    advantages = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    procs = [{"x": torch.zeros(1)} for _ in range(4)]
    chunks = [torch.zeros(1, 4) for _ in range(4)]

    loss, stats, new_lp = trainer._backward_chunk_grpo_loss_microbatched(
        train_wrapper=wrapper,
        proc_snapshots=procs,
        unsquashed_chunks=chunks,
        old_lp=old_lp,
        advantages=advantages,
        clip_eps=0.2,
    )

    assert wrapper.calls == 4
    assert TrackGraph.max_active == 1
    assert TrackGraph.active == 0
    assert isinstance(loss, float)
    assert len(new_lp) == 4
    assert "ratio_mean" in stats


def test_phase12_old_policy_sampling_uses_inference_mode_when_enabled() -> None:
    seen = {}

    class Wrapper:
        bundle = SimpleNamespace(device=torch.device("cpu"))

        def sample_action_chunk_from_proc(self, proc, *, chunk_len, rng):
            del proc, chunk_len, rng
            seen["grad_enabled"] = torch.is_grad_enabled()
            seen["inference_mode"] = torch.is_inference_mode_enabled()
            return SimpleNamespace(
                unsquashed_chunk=torch.zeros(2, 4),
                log_prob_steps=torch.zeros(2),
                log_prob_sum=torch.tensor(0.0),
                exec_action_np=np.zeros((2, 4), dtype=np.float32),
                action_clip_fraction=np.zeros(2),
                action_clip_any=np.zeros(2, dtype=bool),
                unique_action_rows=1,
            )

    sample = trainer._sample_old_action_chunk(
        Wrapper(),
        {"x": torch.zeros(1)},
        chunk_len=2,
        rng=torch.Generator(device="cpu"),
        use_inference_mode=True,
    )

    assert sample.unique_action_rows == 1
    assert seen == {"grad_enabled": False, "inference_mode": True}

