from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from scripts.grpo import train_phase12_wm_chunk_grpo as trainer
from scripts.grpo.train_phase12_wm_chunk_grpo import build_manifest, main, parse_args


def test_phase12_cli_defaults() -> None:
    args = parse_args([])

    assert args.env_backend == "official_lerobot_guarded"
    assert args.jepa_ckpt == "jepa_wm_metaworld.pth.tar"
    assert args.action_profile == "official_jepa_mirror"
    assert args.chunk_len == 25
    assert args.group_size == 4
    assert args.goal_latent_mode == "visual_proprio"
    assert args.proprio_alpha == 0.1
    assert args.reward_key == "wm_latent_progress"
    assert args.ratio_mode == "chunk"
    assert args.action_transform == "no_tanh"
    assert args.reset_mismatch == "fail"
    assert args.decode_candidates == "selected"
    assert args.mode == "wm_grpo_train"


def test_phase12_trainer_has_no_real_mode_not_implemented_guard() -> None:
    source = (trainer._REPO / "scripts" / "grpo" / "train_phase12_wm_chunk_grpo.py").read_text(
        encoding="utf-8"
    )

    assert "real GPU trainer wiring is not complete" not in source
    assert "PHASE12_WM_CHUNK_GRPO_TRAIN_DONE" in source
    assert "CarryMode.SIM" not in source
    assert 'carry_mode="sim"' in source


def test_write_selected_frames_png_only_writes_requested_indices(tmp_path) -> None:
    frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(4)]

    trainer._write_selected_frames_png(tmp_path, frames, [1, 3])

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "frame_000000.png",
        "frame_000002.png",
    ]


def test_manifest_records_phase12_contract(tmp_path) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--dry-run", "--num-episodes", "1"])

    manifest = build_manifest(args)

    assert manifest["method_label"] == "wm_scored_receding_horizon_chunk_grpo"
    assert manifest["mode"] == "wm_grpo_train"
    assert manifest["optimizer_updates"] == 1
    assert manifest["objective_type"] == "L2"
    assert manifest["goal_latent_mode"] == "visual_proprio"
    assert manifest["proprio_alpha"] == 0.1
    assert manifest["action_profile"] == "official_jepa_mirror"
    assert manifest["uses_cem"] is False
    assert manifest["chunk_len"] == 25
    assert json.dumps(manifest)


def test_manifest_records_phase12_pixel_contracts(tmp_path) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--dry-run", "--num-episodes", "1"])

    manifest = build_manifest(args)

    assert manifest["phase12_policy_frame_contract"] == "lerobot_corner2_vhflip"
    assert manifest["phase12_wm_frame_contract"] == "jepa_corner2_vflip"
    assert manifest["phase12_goal_frame_contract"] == "jepa_corner2_vflip"
    assert manifest["phase12_decode_real_frame_source"] == "wm_frames"


def test_phase12_oracle_and_selected_rollout_do_not_use_render_frame_for_pixels() -> None:
    source = (trainer._REPO / "scripts" / "grpo" / "train_phase12_wm_chunk_grpo.py").read_text(
        encoding="utf-8"
    )

    assert "env_h.render_frame()" not in source
    assert "self.env_h.render_frame()" not in source
    assert "policy_rgb_from_obs(reset_obs)" in source
    assert "policy_rgb_from_obs(step.observation)" in source


def test_rollout_validation_manifest_is_not_grpo_training(tmp_path) -> None:
    args = parse_args(["--mode", "rollout_validation", "--output-dir", str(tmp_path), "--dry-run"])

    manifest = build_manifest(args)

    assert manifest["mode"] == "rollout_validation"
    assert manifest["method_label"] == "wm_scored_receding_horizon_rollout_validation"
    assert manifest["optimizer_updates"] == 0
    assert manifest["uses_cem"] is False


def test_real_mode_requires_jepa_inputs(tmp_path) -> None:
    code = main(["--output-dir", str(tmp_path), "--num-episodes", "1"])

    assert code == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["event"] == "configuration_error"
    assert "jepa" in rows[-1]["reason"].lower()


def test_real_mode_writes_progress_checkpoint_and_smoke_manifest(monkeypatch, tmp_path) -> None:
    selected = tmp_path / "selected_action_rollout.mp4"
    oracle = tmp_path / "oracle_baseline.mp4"
    selected.write_bytes(b"selected")
    oracle.write_bytes(b"oracle")

    episode = SimpleNamespace(
        total_env_reward=3.0,
        success_any=True,
        success_last=False,
        metadata={
            "wm_latent_progress": 1.25,
            "latent_return": -0.75,
            "rollout_validation_video": str(selected),
            "selected_action_rollout_video": str(selected),
            "oracle_baseline_video": str(oracle),
            "oracle_baseline_video_status": "ok",
            "wm_decode_status": "failed",
        },
    )
    calls: list[dict] = []

    def fake_run_episode(**kwargs):
        calls.append(kwargs)
        return episode

    monkeypatch.setattr(trainer, "run_phase12_episode", fake_run_episode)
    monkeypatch.setattr(
        trainer,
        "save_grpo_checkpoint",
        lambda path, **_kwargs: path.parent.mkdir(parents=True, exist_ok=True) or path.write_bytes(b"ckpt"),
    )

    code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--mode",
            "rollout_validation",
            "--jepa-repo",
            "/tmp/jepa",
            "--jepa-ckpt",
            "wm.pt",
            "--num-episodes",
            "1",
            "--num-updates",
            "1",
        ]
    )

    assert code == 0
    assert calls and calls[0]["action_profile"] == "official_jepa_mirror"
    smoke = json.loads((tmp_path / "smoke_manifest.json").read_text(encoding="utf-8"))
    assert smoke["success_any"] is True
    assert smoke["success_last"] is False
    rows = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["event"] == "run_start"


class _TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.log_std = torch.nn.Parameter(torch.zeros(1, 4))
        self.model.euler_step_noise_std = 0.0
        self.w = torch.nn.Parameter(torch.zeros(1))

    def select_action_distr_params(self, proc):
        mean = self.w.reshape(1, 1).expand(4, 4)
        log_std = self.model.log_std.expand(4, 4)
        return mean, log_std


class _TinyBundle:
    device = torch.device("cpu")
    obs_image_key = "observation.image"
    obs_state_key = "observation.state"
    obs_env_state_key = "observation.environment_state"

    def __init__(self) -> None:
        self.policy = _TinyPolicy()
        self.preprocessor = lambda obs: obs
        self.postprocessor = lambda x: x


def test_one_update_training_validation_saves_real_checkpoint(monkeypatch, tmp_path) -> None:
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

    code = main(
        [
            "--mode",
            "wm_grpo_train",
            "--output-dir",
            str(tmp_path),
            "--jepa-repo",
            "/tmp/jepa",
            "--jepa-ckpt",
            "wm.pt",
            "--num-episodes",
            "1",
            "--num-updates",
            "1",
        ]
    )

    assert code == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    update_rows = [row for row in rows if row.get("event") == "update_complete"]
    assert update_rows and update_rows[-1]["optimizer_step"] is True
    ckpt = torch.load(tmp_path / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    assert ckpt["policy_state_dict"]
    assert ckpt["optimizer_state_dict"]

