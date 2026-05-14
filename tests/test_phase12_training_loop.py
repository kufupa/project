from __future__ import annotations

import json
from types import SimpleNamespace

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

