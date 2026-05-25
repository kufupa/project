from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
MSM = PROJECT / "scripts" / "maniskill_smolvla"


def _read(name: str) -> str:
    return (MSM / name).read_text(encoding="utf-8")


def test_pbs_scripts_use_pbs_and_expected_queues() -> None:
    pbs_files = [
        "00_build_envs.pbs",
        "01_data_probe.pbs",
        "02_data_full.pbs",
        "03_convert_full.pbs",
        "04_sft_smoke.pbs",
        "05_sft_train.pbs",
    ]
    for name in pbs_files:
        text = _read(name)
        assert "#PBS" in text
        assert "#SBATCH" not in text
        assert "scripts/maniskill_smolvla/common.sh" in text
        assert "artifacts/smolvla_maniskill" in text

    assert "#PBS -q v1_gpu72" in _read("01_data_probe.pbs")
    assert "#PBS -q v1_gpu72" in _read("02_data_full.pbs")
    assert "#PBS -q v1_gpu72" in _read("04_sft_smoke.pbs")
    assert "#PBS -q v1_gpu72" in _read("05_sft_train.pbs")


def test_common_keeps_large_artifacts_in_ephemeral() -> None:
    common = _read("common.sh")
    assert "/rds/general/user/aa6622/ephemeral/eggroll/smolvla_maniskill" in common
    assert "MSM_RAW_ROOT" in common
    assert "MSM_LEROBOT_ROOT" in common
    assert "MSM_CHECKPOINT_ROOT" in common
    assert "HF_LEROBOT_HOME" in common
    assert "RAY_TMPDIR" in common
    assert "MS_SKIP_ASSET_DOWNLOAD_PROMPT" in common


def test_collector_patch_records_state() -> None:
    collector_path = PROJECT / "RL4VLA/ManiSkill/mani_skill/examples/data_collector/vla_data_collector.py"
    if not collector_path.exists():
        return
    collector = collector_path.read_text(encoding="utf-8")
    assert '"state": []' in collector
    assert "def update_state" in collector
    assert "ee_pose_at_robot_base" in collector
    assert "matrix_to_euler_angles" in collector
    assert "self.update_state(action)" in collector


def test_converter_uses_lerobot_7d_contract() -> None:
    converter = _read("convert_npz_to_lerobot.py")
    assert "LeRobotDataset.create" in converter
    assert '"observation.images.front"' in converter
    assert '"observation.state"' in converter
    assert '"action"' in converter
    assert '"shape": (7,)' in converter
    assert '"task": instruction' in converter
    assert "dataset.finalize()" in converter


def test_sft_keeps_pretrained_padding_dims() -> None:
    combined = _read("04_sft_smoke.pbs") + "\n" + _read("05_sft_train.pbs")
    assert "--policy.path=lerobot/smolvla_base" in combined
    assert "--policy.max_state_dim=32" in combined
    assert "--policy.max_action_dim=32" in combined
    assert "--save_freq=\"${MSM_SFT_SAVE_FREQ}\"" in combined


def test_gitignore_covers_large_artifacts() -> None:
    gitignore = (PROJECT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ["artifacts/", "*.npz", "*.safetensors", "*.pt", "*.mp4"]:
        assert pattern in gitignore
